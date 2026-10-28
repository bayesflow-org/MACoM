"""Simulation-based validation on each family's EMPIRICAL design (experiment 2).

Recovery / coverage / calibration of the model-class CogFormer on data simulated with
the real experiment's design, so the posteriors experiment 2 reports on real data come
with a check of how well that exact design identifies each coefficient.

"The real design" is taken literally, to respect the original authors' designs as far
as the simulator allows. Every validation dataset borrows one real subject's:

  - regressor values and coding (+/-0.5 contrasts, CDM's 3-level difficulty as two
    dummies, and their interactions), built by the same data_utils code the apply
    scripts use -- not the generator's approximation of them;
  - trial count;
  - per-trial context: RDM's 3 alternatives and the subject's per-trial ink colour
    (favored_idx); CDM's heading fixed at 0, matching deviation-coded responses;
  - parameter design (which coefficients are regressed), from ppc.MODEL_REGISTRY, i.e.
    the design the real-data fit used. --fix_s_v drops s_v exactly as the DDM refit does.

LBA (experiment 3) runs through the same code against the LBA-FINETUNED checkpoint
(4 families, fixed_A=0.5): stimulus-coded, favored_idx = the word/nonword stimulus from
the data, and --design selects the lba_mc_cf_apply design the real-data fit used
(condition_only by default, matching run_experiments.sh's LBA_RD_DESIGN).

Datasets cycle through the subjects. True coefficients come from the TRAINING prior via
ModelFamily.sample(design_matrix=...), which swaps in the fixed design but keeps the
training code path for the coefficient prior, the realized-support mask, fixed-value
injection and simulation.

Figures use experiment 1's plot_case, so the two validations read the same way.

    python -m experiments.experiment_2.realdata_validate --model rdm \\
        --checkpoint <ckpt> --outdir <figs> --pred_dir <data>
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from cogformer.adapter import Adapter
from cogformer.networks.transformers.cogformer import CogFormer
from ..utils import add_structure_conditioning_arg
from cogformer.simulator import ModelFamily, ModelClass
from cogformer.simulator.benchmarks.ssms.ddm import DDM
from cogformer.simulator.benchmarks.ssms.rdm import RDM
from cogformer.simulator.benchmarks.ssms.cdm import CDM
from cogformer.simulator.benchmarks.ssms.lba import LBA
from cogformer.simulator.benchmarks.ssms.priors import ddm_priors, rdm_priors, cdm_priors, lba_priors
from cogformer.simulator.benchmarks.ssms.link_funs import (ddm_link_fun, rdm_link_fun, cdm_link_fun,
                                                           lba_link_fun)
from cogformer.utils.plot_utils import cogformer_mc_colors

from ..utils import build_encoder_input_dim, chunked_sample, seed_case
from ..experiment_1.model_class_cf_validate import (_MODELS_WITH_PI, infer_mask_randomizer_kwargs,
                                                    plot_case)
from .data_utils import build_real_design_matrix, build_real_param_mask, build_subject_batch
from .ppc import MODEL_REGISTRY
from .lba_mc_cf_apply import LBA_DESIGNS
from ..experiment_3.model_class_lba_finetune_validate import _LBA_WITH_PI

FAMILY_CFG = {**_MODELS_WITH_PI, "lba": _LBA_WITH_PI}

NON_DESIGN_KEYS = ("rts", "choices", "favored_idx", "participant")


def build_model_class(seeded: bool, lba_fixed_A: float | None = None) -> ModelClass:
    """The 3-family class, or with lba_fixed_A the 4-family finetuned one (LBA last,
    matching the finetune and lba_mc_cf_apply, so model ids line up with the checkpoint)."""
    workers = 1 if seeded else None
    fams = {
        "DDM": ModelFamily(model=DDM(), name="DDM", prior=ddm_priors(), num_workers=workers,
                           intrinsic_params=_MODELS_WITH_PI["ddm"]["intrinsic_params"]),
        "RDM": ModelFamily(model=RDM(), name="RDM", prior=rdm_priors(), num_workers=workers,
                           intrinsic_params=_MODELS_WITH_PI["rdm"]["intrinsic_params"]),
        "CDM": ModelFamily(model=CDM(), name="CDM", prior=cdm_priors(), num_workers=workers,
                           intrinsic_params=_MODELS_WITH_PI["cdm"]["intrinsic_params"]),
    }
    links = {"DDM": ddm_link_fun(), "RDM": rdm_link_fun(), "CDM": cdm_link_fun()}
    if lba_fixed_A is not None:
        fams["LBA"] = ModelFamily(model=LBA(fixed_A=lba_fixed_A), name="LBA", prior=lba_priors(),
                                  num_workers=workers,
                                  intrinsic_params=_LBA_WITH_PI["intrinsic_params"])
        links["LBA"] = lba_link_fun()
    return ModelClass(model_families=fams, link_funs=links)


def regressor_spec(subject: dict, design_config: dict) -> tuple[list, dict]:
    """(categorical regressors, realized dummy-column count per main regressor), read off
    the data: a 2-D regressor is dummy-coded (categorical), a 1-D one is a single column."""
    cats, counts = [], {}
    for k in design_config:
        if k == "1" or ":" in k:
            continue
        x = np.asarray(subject[k])
        counts[k] = x.shape[1] if x.ndim == 2 else 1
        if x.ndim == 2:
            cats.append(k)
    return cats, counts


def simulation_context(model_key: str, subject: dict, num_obs: int):
    reg = MODEL_REGISTRY[model_key]
    if reg.get("num_alternatives") is not None:        # RDM: 3-colour Stroop
        return {"num_alternatives": int(reg["num_alternatives"]),
                "favored_idx": np.asarray(subject["favored_idx"])}
    if model_key == "cdm":                               # deviation-coded responses
        return CDM.build_context(num_obs, theta_mode="zeros")
    return None


def main():
    a = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fam, FAM = a.model, a.model.upper()
    reg, mcfg = MODEL_REGISTRY[fam], FAMILY_CFG[fam]
    is_lba = fam == "lba"
    if a.num_models is None:
        a.num_models = 4 if is_lba else 3

    design_config = {k: list(v) for k, v in
                     (LBA_DESIGNS[a.design] if is_lba else reg["design_config"]).items()}
    cfg_name = "realdata" if not is_lba or a.design == "condition_only" else f"realdata_{a.design}"
    if a.fix_s_v:
        if fam != "ddm":
            raise SystemExit("--fix_s_v mirrors the DDM refit; it is DDM-only")
        # Dropping s_v from the design fixes it: the simulator injects
        # default_fixed_values["s_v"] (0.0), exactly as ddm_mc_cf_apply --fix_s_v fits.
        design_config["1"] = [p for p in design_config["1"] if p != "s_v"]
        cfg_name = "realdata_sv_fixed"
    intrinsic = mcfg["intrinsic_params"]
    n_local = len(intrinsic)
    main_keys = [k for k in design_config if k != "1" and ":" not in k]

    model_class = build_model_class(seeded=a.seed is not None,
                                    lba_fixed_A=a.lba_fixed_A if is_lba else None)
    mf = model_class.model_families[FAM]
    link = model_class.link_funs[FAM]
    model_id = model_class.model_index[FAM]
    global_idx = model_class.local_to_global[FAM]
    n_global = model_class.n_global_params
    mask_kwargs = infer_mask_randomizer_kwargs(design_config, intrinsic, mcfg["default_fixed_values"])

    cogformer = CogFormer(
        encoder_structure_conditioning=a.encoder_structure_conditioning,
        encoder_input_dim=build_encoder_input_dim(a.max_num_regressors, a.max_num_categories, True),
        encoder_num_layers=a.encoder_num_layers, decoder_num_layers=a.decoder_num_layers,
        encoder_num_heads=a.encoder_num_heads, decoder_num_heads=a.decoder_num_heads,
        encoder_embed_dim=a.encoder_embed_dim,
        num_seeds=a.num_seeds, seed_dim=a.seed_dim, proj_dim=a.proj_dim,
        dropout=0.0, layer_dropout=0.0,
        decoder_layer_design="mixed_attention", decoder_layer_kwargs={"mab_first": True},
        time_embedding_dim=a.time_embedding_dim, pos_embedding_dim=a.pos_embedding_dim,
        num_models=a.num_models, model_embed_dim=a.model_embed_dim,
        gated_residual=a.gated_residual, pre_pool_ffn=a.pre_pool_ffn,
    ).to(device)
    state = torch.load(a.checkpoint, map_location=device)
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    cogformer.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in state.items()})
    cogformer.eval()
    adapter = Adapter()

    # Full-geometry mask, lifted to global space; build_subject_batch refines it to the
    # realized support of each dataset's design.
    local_mask = build_real_param_mask(design_config=design_config, intrinsic_params=intrinsic,
                                       max_num_regressors=a.max_num_regressors,
                                       max_num_categories=a.max_num_categories, keep_intercept=True)
    n_rows = len(local_mask) // n_local
    _, gmask = model_class.lift_to_global_space(FAM, local_mask[None], local_mask[None])
    gmask = gmask[0]

    data, ids = reg["loader"](reg["data_path"])
    logging.info(f"{FAM} {cfg_name}: {a.num_datasets} datasets over {len(ids)} real subject designs")

    true_set = np.zeros((a.num_datasets, n_rows, n_local), np.float32)
    pred_set = np.zeros((a.num_datasets, a.num_samples, n_rows, n_local), np.float32)
    realized_masks = []
    for i in range(a.num_datasets):
        subj = data[ids[i % len(ids)]]
        regs = {k: subj[k] for k in subj if k not in NON_DESIGN_KEYS}
        num_obs = len(subj["rts"])
        # The design at the model's full geometry (what the encoder reads) and at its
        # compact geometry (what ModelFamily.sample's layout is for this many regressors).
        # They agree column-for-column; the full one is the compact one zero-padded.
        dm_full = build_real_design_matrix(regressors=regs, design_config=design_config,
                                           max_num_regressors=a.max_num_regressors,
                                           max_num_categories=a.max_num_categories,
                                           keep_intercept=True)
        dm_small = build_real_design_matrix(regressors=regs, design_config=design_config,
                                            max_num_regressors=len(main_keys),
                                            max_num_categories=a.max_num_categories,
                                            keep_intercept=True)
        k = dm_small.shape[1]
        assert np.array_equal(dm_small, dm_full[:, :k]) and not dm_full[:, k:].any(), \
            "compact and full design layouts disagree; true coefficients would misalign"
        cats, counts = regressor_spec(subj, design_config)
        ctx = simulation_context(fam, subj, num_obs)

        seed_case(a.seed, FAM, cfg_name, i)
        s = mf.sample(design_config=design_config, num_obs=num_obs,
                      max_num_regressors=len(main_keys), max_num_categories=a.max_num_categories,
                      keep_intercept=True, fixed_config=True, link_fun=link,
                      mask_randomizer_kwargs=mask_kwargs, categorical_regressors=cats,
                      design_matrix=dm_small, category_counts=counts, context=ctx,
                      flatten_param_outputs=False)
        true_set[i, :k] = s["param_matrix"] * s["param_mask"]

        batch = build_subject_batch(
            rts=s["obs"]["rts"], choices=s["obs"]["choices"], design_matrix=dm_full,
            param_mask=gmask, n_params=n_global,
            max_num_regressors=a.max_num_regressors, max_num_categories=a.max_num_categories,
            design_config=design_config,
            num_alternatives=int(reg.get("num_alternatives") or 2), model_id=model_id,
            favored_idx=(ctx or {}).get("favored_idx"),
        )
        realized_masks.append(batch["param_masks"][0].reshape(n_rows, n_global)[:, global_idx])
        adapted = adapter.adapt(batch, device=device)
        seed_case(a.seed, FAM, cfg_name, i, salt="ode")
        pred = chunked_sample(cogformer, adapted, num_samples=a.num_samples,
                              steps=a.num_sample_steps, solver=a.solver,
                              samples_chunk_size=a.samples_chunk_size)
        pred_set[i] = pred.reshape(a.num_samples, n_rows, n_global)[:, :, global_idx]
        if (i + 1) % 20 == 0:
            logging.info(f"  {i + 1}/{a.num_datasets}")

    # One mask for the plots. The real designs are balanced, so every subject realizes
    # the same cells; fail loudly rather than average over datasets that disagree.
    params_mask = realized_masks[0]
    if not all(np.array_equal(params_mask, m) for m in realized_masks):
        raise SystemExit("realized parameter support differs across subjects; per-dataset "
                         "masks would be needed for these diagnostics")

    pred_dir = Path(a.pred_dir); pred_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{fam}_{cfg_name}"
    np.savez(pred_dir / f"{tag}_cf_pred.npz", pred_set=pred_set, true_set=true_set,
             params_mask=params_mask, subject_ids=np.array(ids))
    logging.info(f"[saved] {pred_dir / f'{tag}_cf_pred.npz'}")

    outdir = Path(a.outdir) / fam / cfg_name
    outdir.mkdir(parents=True, exist_ok=True)
    case = {"model_name": FAM, "cfg_name": cfg_name, "design_config": design_config,
            "intrinsic_params": intrinsic, "variable_names": mcfg["variable_names"],
            "true_set": true_set, "pred_set": pred_set, "params_mask": params_mask,
            "outdir": outdir, "n_rows": n_rows, "global_indices": global_idx,
            "categorical_regressors": regressor_spec(data[ids[0]], design_config)[0]}
    plot_case(case, cogformer_mc_colors(), max_num_categories=a.max_num_categories,
              max_num_regressors=a.max_num_regressors, skip_posteriors=True)
    logging.info(f"[saved] figures -> {outdir}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_structure_conditioning_arg(p)
    p.add_argument("--model", required=True, choices=["ddm", "rdm", "cdm", "lba"])
    p.add_argument("--design", default="condition_only", choices=sorted(LBA_DESIGNS),
                   help="LBA only: which lba_mc_cf_apply design the real-data fit used.")
    p.add_argument("--lba_fixed_A", type=float, default=0.5,
                   help="LBA only: MUST equal the finetuned checkpoint's fixed_A.")
    p.add_argument("--fix_s_v", action="store_true",
                   help="DDM only: the s_v-fixed design of the refit (original specification).")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--num_datasets", type=int, default=200)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--num_sample_steps", type=int, default=50)
    p.add_argument("--solver", default="midpoint", choices=["euler", "midpoint", "rk4"])
    p.add_argument("--samples_chunk_size", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_num_regressors", type=int, default=4)
    p.add_argument("--max_num_categories", type=int, default=4)
    for k, v in [("encoder_num_layers", 8), ("decoder_num_layers", 8), ("encoder_num_heads", 8), ("encoder_embed_dim", 64),
                 ("decoder_num_heads", 8), ("num_seeds", 32), ("seed_dim", 128), ("proj_dim", 256),
                 ("time_embedding_dim", 32), ("pos_embedding_dim", 32),
                 ("model_embed_dim", 8)]:
        p.add_argument(f"--{k}", type=int, default=v)
    p.add_argument("--num_models", type=int, default=None,
                   help="Model-embedding rows in the checkpoint; default 3, or 4 for lba.")
    # Must match the checkpoint; see model_class_cf_train. Default ON = the class default.
    p.add_argument("--gated_residual", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pre_pool_ffn", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


if __name__ == "__main__":
    main()
