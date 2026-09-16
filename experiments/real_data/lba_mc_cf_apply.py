"""Apply the LBA-finetuned ModelClass CogFormer to real lexical-decision data.

Data: Wagenmakers, Ratcliff, Gomez & McKoon (2008), distributed as ``speed_acc`` in
the R package ``rtdists`` (GPL-3). See load_lba_data for the coding decisions.

Each subject is processed individually (batch_size=1). Posteriors are saved in global
parameter space as (num_subjects, num_samples, n_rows, n_global_params), matching the
DDM/RDM/CDM apply scripts so the downstream PPC and plotting paths work unchanged.

THREE THINGS THAT DIFFER FROM THE OTHER APPLY SCRIPTS, all load-bearing:

1. FOUR model families, not three. The checkpoint is the LBA-finetuned one, whose
   model_embedding has 4 rows; ``--num_models 4`` must match it or load_state_dict
   fails. LBA is index 3.

2. ``--lba_fixed_A`` MUST equal the value the checkpoint was finetuned with (0.5).
   It changes what ``a`` means -- with fixed_A set, ``a`` is the mean distance
   travelled rather than the start-point range -- so a mismatch silently reinterprets
   the parameter rather than erroring.

3. GEOMETRY DEFAULTS ARE r4c4, not the r5c5 the other apply scripts default to. The
   model-class checkpoints are r4c4 (encoder input 72); r5c5 builds a 186-wide encoder
   and dies in load_state_dict. run_lba.sh omits these flags entirely, which is why
   the finetune could never be launched through it.

    python -m experiments.real_data.lba_mc_cf_apply --checkpoint "$LBA_FT_CKPT"
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from cogformer.adapter import Adapter
from cogformer.networks.transformers.cogformer import CogFormer
from cogformer.simulator import ModelFamily, ModelClass
from cogformer.simulator.benchmarks.ssms.ddm import DDM
from cogformer.simulator.benchmarks.ssms.rdm import RDM
from cogformer.simulator.benchmarks.ssms.cdm import CDM
from cogformer.simulator.benchmarks.ssms.lba import LBA
from cogformer.simulator.benchmarks.ssms.priors import (
    ddm_priors, rdm_priors, cdm_priors, lba_priors)
from cogformer.simulator.benchmarks.ssms.link_funs import (
    ddm_link_fun, rdm_link_fun, cdm_link_fun, lba_link_fun)

from ..configs import (MODELS as _BASE_MODELS, LBA_CONFIG, LBA_PPC_DESIGN_CONFIG,
                       GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
from ..utils import add_global_params, build_encoder_input_dim, chunked_sample
from .data_utils import (
    load_lba_data,
    build_real_design_matrix,
    build_real_param_mask,
    build_subject_batch,
)

# condition (speed/accuracy), frequency band and lexicality are regressed on drift,
# drift advantage, threshold and non-decision time; s_v/s_tau/pi stay intercept-only.
# Only the condition x frequency interaction is carried: it is the theoretically
# interesting one (does the speed-accuracy tradeoff change with evidence quality), and
# every added interaction costs block_width**2 = 9 design columns.
# Derived from the canonical table in experiments/configs.py rather than restated, so
# the simulated val scenario ("ppc_design") and this real-data fit cannot drift -- the
# same reason ppc.py imports this name instead of declaring its own copy. configs.py
# defines it without pi (module convention); the real-data application DOES estimate
# contamination, so pi joins the intercept row here, exactly as add_global_params does
# for the simulated configs.
LBA_DESIGN_CONFIG = {
    k: (list(v) + GLOBAL_PARAMS if k == "1" else list(v))
    for k, v in LBA_PPC_DESIGN_CONFIG.items()
}

# Every parameter on the intercept row, no regressors. A DIAGNOSTIC alternative to the
# full design, not a replacement: it asks what the fit looks like when the model is not
# asked to track condition, frequency or their interaction.
#
# Under STIMULUS coding this does NOT discard the lexicality manipulation. Which
# accumulator the stimulus favours is per-trial CONTEXT (favored_idx), not a design
# column, so word/nonword structure still reaches both encoder and simulator; what is
# dropped is how the drift advantage VARIES with frequency band and speed instruction.
# Under accuracy coding the same change would have removed lexicality entirely.
LBA_DESIGN_CONFIG_INTERCEPT = {"1": list(LBA_DESIGN_CONFIG["1"])}

# Intercept plus the speed/accuracy contrast, nothing else. The middle ground between
# the two above, and the one the PPC argues for: intercept_only fits RT better than the
# full design everywhere EXCEPT by condition, where it is structurally blind -- it
# predicts a Speed-Accuracy median gap of -0.000 against an observed -0.147. u_1 is what
# carries that gap.
#
# Dropping u_3 (lexicality) is principled rather than merely expedient under stimulus
# coding: which accumulator the stimulus favours already reaches the model as per-trial
# context, so u_3 only adds a second, collinear route to the same manipulation.
LBA_DESIGN_CONFIG_CONDITION = {
    "1": list(LBA_DESIGN_CONFIG["1"]),
    "u_1": list(LBA_DESIGN_CONFIG["u_1"]),
}

LBA_DESIGNS = {"full": LBA_DESIGN_CONFIG,
               "intercept_only": LBA_DESIGN_CONFIG_INTERCEPT,
               "condition_only": LBA_DESIGN_CONFIG_CONDITION}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="./data/data_application_lba.csv")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--outdir", type=str, default="./outputs/data/real_data/lba/mc/")
    p.add_argument("--design", type=str, default="full", choices=sorted(LBA_DESIGNS),
                   help="Which design to fit. ppc.py takes the same flag and MUST be "
                        "given the matching value: the posteriors carry no record of "
                        "which design produced them, so a mismatch reindexes the "
                        "coefficient rows silently.")
    p.add_argument("--lba_fixed_A", type=float, default=0.5,
                   help="MUST match the finetuned checkpoint (0.5). Changes what `a` "
                        "means rather than erroring if wrong.")

    # Design -- r4c4, matching the model-class checkpoints.
    p.add_argument("--max_num_regressors", type=int, default=4)
    p.add_argument("--max_num_categories", type=int, default=4)
    p.add_argument("--keep_intercept", action="store_true", default=True)

    # Inference
    p.add_argument("--num_sample_steps", type=int, default=50)
    p.add_argument("--num_samples", type=int, default=500)
    p.add_argument("--samples_chunk_size", type=int, default=1)
    p.add_argument("--solver", type=str, default="midpoint",
                   choices=["euler", "midpoint", "rk4"])

    # Architecture (must match training)
    p.add_argument("--encoder_num_layers", type=int, default=8)
    p.add_argument("--decoder_num_layers", type=int, default=8)
    p.add_argument("--encoder_num_heads", type=int, default=8)
    p.add_argument("--decoder_num_heads", type=int, default=8)
    p.add_argument("--num_seeds", type=int, default=32)
    p.add_argument("--seed_dim", type=int, default=128)
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--layer_dropout", type=float, default=0.0)
    p.add_argument("--time_embedding_dim", type=int, default=32)
    p.add_argument("--pos_embedding_dim", type=int, default=32)
    p.add_argument("--num_models", type=int, default=4)
    p.add_argument("--model_embed_dim", type=int, default=8)
    p.add_argument("--self_attn_padding_mask", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Must match the finetune run; changes the forward pass but not "
                        "the state_dict keys, so a mismatch loads cleanly and is wrong.")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _MODELS_WITH_PI = {
        m: add_global_params(cfg, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
        for m, cfg in _BASE_MODELS.items()
        if m != "lba"
    }
    _LBA_WITH_PI = add_global_params(LBA_CONFIG, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
    lba_local_params = _LBA_WITH_PI["intrinsic_params"]

    def _mask_kwargs(cfg):
        return {"free_intrinsics": cfg["free_params"],
                "fixed_intrinsics": cfg["fixed_params"],
                "fixed_values": cfg["default_fixed_values"],
                "global_params": GLOBAL_PARAMS}

    # Insertion order fixes the model indices; LBA must land at 3 to match the
    # finetuned embedding table, whose row 3 is the one finetuning trained.
    model_families = {
        "DDM": ModelFamily(model=DDM(), name="DDM", prior=ddm_priors(),
                           intrinsic_params=_MODELS_WITH_PI["ddm"]["intrinsic_params"],
                           mask_randomizer_kwargs=_mask_kwargs(_MODELS_WITH_PI["ddm"])),
        "RDM": ModelFamily(model=RDM(), name="RDM", prior=rdm_priors(),
                           intrinsic_params=_MODELS_WITH_PI["rdm"]["intrinsic_params"],
                           mask_randomizer_kwargs=_mask_kwargs(_MODELS_WITH_PI["rdm"])),
        "CDM": ModelFamily(model=CDM(), name="CDM", prior=cdm_priors(),
                           intrinsic_params=_MODELS_WITH_PI["cdm"]["intrinsic_params"],
                           mask_randomizer_kwargs=_mask_kwargs(_MODELS_WITH_PI["cdm"])),
        "LBA": ModelFamily(model=LBA(fixed_A=args.lba_fixed_A), name="LBA",
                           prior=lba_priors(),
                           intrinsic_params=lba_local_params,
                           mask_randomizer_kwargs=_mask_kwargs(_LBA_WITH_PI)),
    }
    link_funs = {"DDM": ddm_link_fun(), "RDM": rdm_link_fun(),
                 "CDM": cdm_link_fun(), "LBA": lba_link_fun()}
    model_class = ModelClass(model_families=model_families, link_funs=link_funs)

    n_global_params = model_class.n_global_params
    lba_model_id = model_class.model_index["LBA"]
    if lba_model_id != 3:
        raise ValueError(
            f"LBA resolved to model index {lba_model_id}, expected 3. The finetuned "
            "checkpoint trained row 3 of the embedding table; any other index reads "
            "an embedding that was never finetuned for LBA."
        )
    if model_class.num_models != args.num_models:
        raise ValueError(
            f"ModelClass has {model_class.num_models} families but --num_models is "
            f"{args.num_models}; the embedding table would not match the checkpoint."
        )

    encoder_input_dim = build_encoder_input_dim(
        args.max_num_regressors, args.max_num_categories, args.keep_intercept
    )
    cogformer = CogFormer(
        encoder_input_dim=encoder_input_dim,
        encoder_num_layers=args.encoder_num_layers,
        decoder_num_layers=args.decoder_num_layers,
        encoder_num_heads=args.encoder_num_heads,
        decoder_num_heads=args.decoder_num_heads,
        num_seeds=args.num_seeds,
        seed_dim=args.seed_dim,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
        layer_dropout=args.layer_dropout,
        decoder_layer_design="mixed_attention",
        decoder_layer_kwargs={"mab_first": True},
        time_embedding_dim=args.time_embedding_dim,
        pos_embedding_dim=args.pos_embedding_dim,
        num_models=args.num_models,
        model_embed_dim=args.model_embed_dim,
        self_attn_padding_mask=args.self_attn_padding_mask,
    ).to(device)

    state = torch.load(Path(args.checkpoint), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    cogformer.load_state_dict(state)
    cogformer.eval()
    logging.info(f"[loaded] {args.checkpoint}")

    adapter = Adapter()

    # The mask decides which coefficient cells are ACTIVE, so it has to be built from
    # the same design as the design matrix. Built from the full config while the matrix
    # was intercept-only, it would mark regressor rows active with no columns behind
    # them -- a silent reindexing, not an error.
    design_config = LBA_DESIGNS[args.design]
    logging.info(f"design={args.design}: rows {list(design_config)}")

    local_mask_flat = build_real_param_mask(
        design_config=design_config,
        intrinsic_params=lba_local_params,
        max_num_regressors=args.max_num_regressors,
        max_num_categories=args.max_num_categories,
        keep_intercept=args.keep_intercept,
    )
    local_pm = local_mask_flat[np.newaxis]
    _, param_mask_global = model_class.lift_to_global_space("LBA", local_pm, local_pm)
    param_mask_global = param_mask_global[0]
    n_rows = len(local_mask_flat) // len(lba_local_params)

    subject_data, subject_ids = load_lba_data(args.data)
    logging.info(f"Loaded {len(subject_ids)} subjects from {args.data}")

    all_posteriors = []
    for sid in subject_ids:
        d = subject_data[sid]
        num_obs = len(d["rts"])
        logging.info(f"Subject {sid}: {num_obs} trials")

        regressors = {"u_1": d["u_1"], "u_2": d["u_2"], "u_3": d["u_3"],
                      "u_1:u_2": d["u_1:u_2"]}
        dm = build_real_design_matrix(
            regressors=regressors,
            design_config=design_config,
            max_num_regressors=args.max_num_regressors,
            max_num_categories=args.max_num_categories,
            keep_intercept=args.keep_intercept,
        )

        batch = build_subject_batch(
            rts=d["rts"], choices=d["choices"],
            design_matrix=dm, param_mask=param_mask_global,
            n_params=n_global_params,
            max_num_regressors=args.max_num_regressors,
            max_num_categories=args.max_num_categories,
            design_config=design_config,
            num_alternatives=2,
            model_id=lba_model_id,
            # Without this the Adapter fills the channel with zeros and raises
            # nothing, telling the encoder accumulator 0 is favoured on every trial
            # while the stimulus-coded choices say otherwise.
            favored_idx=d["favored_idx"],
        )
        adapted = adapter.adapt(batch, device=device)

        pred = chunked_sample(
            cogformer, adapted,
            num_samples=args.num_samples,
            steps=args.num_sample_steps,
            solver=args.solver,
            samples_chunk_size=args.samples_chunk_size,
        )
        pred = pred.reshape(args.num_samples, n_rows, n_global_params)
        all_posteriors.append(pred)

    posteriors = np.stack(all_posteriors, axis=0)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "lba_mc_posteriors.npz"
    np.savez(
        out_path,
        posteriors=posteriors,
        subject_ids=np.array(subject_ids),
        global_param_names=np.array(model_class.global_param_names),
        lba_intrinsic_params=np.array(lba_local_params),
        lba_variable_names=np.array(_LBA_WITH_PI["variable_names"]),
        lba_local_to_global=np.array(model_class.local_to_global["LBA"]),
        # The design ACTUALLY fitted, not the full one: this field is the only record
        # the npz carries of how its coefficient rows are laid out.
        design_config_keys=np.array(list(design_config.keys())),
        design_name=np.array(args.design),
        lba_fixed_A=np.float32(args.lba_fixed_A),
    )
    logging.info(f"[saved] {out_path}  shape={posteriors.shape}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
