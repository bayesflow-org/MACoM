"""Apply the LBA-finetuned ModelClass CogFormer to real Stroop colour-word data.

Data: ``experiments/real_data/stroop/stroop.csv`` (participant, rt, ink, word,
response). See ``load_stroop_data`` for the coding decisions and for what this
reduced design can and cannot measure.

This is the three-accumulator sibling of ``lba_mc_cf_apply``. Two things differ from
every other apply script in this directory, and both are load-bearing:

1. K = 3, NOT 2. ``favored_idx`` (the ink colour, i.e. the correct response) must be
   passed explicitly. ``Adapter.adapt`` silently defaults it to zeros when absent —
   "blue is correct on every trial", true here on only ~33% of trials — so omitting it
   produces plausible, wrong posteriors rather than an error. ``build_subject_batch``
   now raises for K > 2 without it; this script supplies it.

2. s_v IS LEFT FREE, DELIBERATELY, despite the literature convention s_v = 1.
   The convention is NOT expressible with this checkpoint. Fixed values bypass the
   link function (``ModelFamily.sample`` injects them on the natural scale), and
   LBA's ``default_fixed_values`` is ``{"s_v": 0.0, "s_tau": 0.0}``. So masking s_v
   out asserts s_v == 0 EXACTLY — deterministic drift — not 1, and not
   shifted_softplus(0) = 0.693 either. The network has never seen s_v fixed at any
   other value, so "fix s_v = 1" would require retraining, not a flag.

   This costs nothing. s_v = 1 is a scaling CONVENTION that makes the LBA
   identifiable, not an empirical claim; here the prior on v/a already fixes the
   scale, so leaving s_v free estimates it on that scale instead of asserting it.
   Pinning it post hoc was measured on speed_acc and made the fit strictly worse
   (W1 up, accuracy worse, dispersion 2.10 -> 2.40).

    python -m experiments.real_data.stroop_lba_cf_apply --checkpoint "$LBA_FT_CKPT"
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

from ..configs import (MODELS as _BASE_MODELS, LBA_CONFIG,
                       GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
from ..utils import add_global_params, build_encoder_input_dim, chunked_sample
from .data_utils import (
    load_stroop_data,
    build_real_design_matrix,
    build_real_param_mask,
    build_subject_batch,
)

# The REDUCED Stroop design agreed for this fit: the ink colour is the favored
# accumulator, and congruency modulates that accumulator's drift ADVANTAGE. This
# measures interference as a shrinking target advantage.
#
# Congruency is deliberately not put on `v` as well. v is the drift shared by all
# three accumulators, so v and v_diff would both be shifted by the same single
# regressor and only their sum is constrained by the favored accumulator's RT --
# leaving the split to be resolved by the two distractors alone. Keeping u_1 on
# v_diff attributes the Stroop effect to one interpretable place.
STROOP_DESIGN_CONFIG = {
    "1":   ["v", "v_diff", "a", "tau", "s_v", "s_tau", "pi"],
    "u_1": ["v_diff"],   # congruency, effect coded (congruent +0.5, incongruent -0.5)
}

# 0=blue, 1=green, 2=red -- must match the ordering in load_stroop_data.
NUM_ALTERNATIVES = 3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str,
                   default="./experiments/real_data/stroop/stroop.csv")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--outdir", type=str, default="./outputs/data/real_data/stroop/mc/")
    p.add_argument("--min_trials", type=int, default=60,
                   help="Drop participants with fewer trials. Median is 96 against a "
                        "trained num_obs range of 200-1000, so every subject is OOD on "
                        "length; this bounds how bad it gets, it does not fix it.")
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

    local_mask_flat = build_real_param_mask(
        design_config=STROOP_DESIGN_CONFIG,
        intrinsic_params=lba_local_params,
        max_num_regressors=args.max_num_regressors,
        max_num_categories=args.max_num_categories,
        keep_intercept=args.keep_intercept,
    )
    local_pm = local_mask_flat[np.newaxis]
    _, param_mask_global = model_class.lift_to_global_space("LBA", local_pm, local_pm)
    param_mask_global = param_mask_global[0]
    n_rows = len(local_mask_flat) // len(lba_local_params)

    subject_data, subject_ids = load_stroop_data(args.data, min_trials=args.min_trials)
    logging.info(f"Loaded {len(subject_ids)} participants from {args.data}")

    all_posteriors, kept_pids = [], []
    for sid in subject_ids:
        d = subject_data[sid]
        num_obs = len(d["rts"])
        logging.info(f"Subject {sid} ({d['participant']}): {num_obs} trials")

        dm = build_real_design_matrix(
            regressors={"u_1": d["u_1"]},
            design_config=STROOP_DESIGN_CONFIG,
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
            design_config=STROOP_DESIGN_CONFIG,
            num_alternatives=NUM_ALTERNATIVES,
            model_id=lba_model_id,
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
        kept_pids.append(d["participant"])

    posteriors = np.stack(all_posteriors, axis=0)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "stroop_lba_posteriors.npz"
    np.savez(
        out_path,
        posteriors=posteriors,
        subject_ids=np.array(subject_ids),
        participants=np.array(kept_pids),
        global_param_names=np.array(model_class.global_param_names),
        lba_intrinsic_params=np.array(lba_local_params),
        lba_variable_names=np.array(_LBA_WITH_PI["variable_names"]),
        lba_local_to_global=np.array(model_class.local_to_global["LBA"]),
        design_config_keys=np.array(list(STROOP_DESIGN_CONFIG.keys())),
        lba_fixed_A=np.float32(args.lba_fixed_A),
        num_alternatives=np.int32(NUM_ALTERNATIVES),
    )
    logging.info(f"[saved] {out_path}  shape={posteriors.shape}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
