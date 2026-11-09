"""Encoder representation experiment: aDDM inference with frozen CogFormer encoder + BayesFlow.

The CogFormer encoder is frozen and used as a summary network; BayesFlow
FlowMatching is trained on the pooled encoder output to infer aDDM parameters.
aDDM is a probe: the encoder never saw it, and unlike LCA/FFI its key parameter
(omega, the attentional discount) is driven by a within-trial process the encoder
has no channel for. What survives into the summary is the question.

Use the ORIGINAL (non-finetuned) model-class checkpoint, so the encoder is the
DDM+RDM+CDM model and every probe family is cleanly out-of-distribution.

Cases (deliberately close to intercept-only — there is no fixation covariate):
  intercept_only  — all aDDM params at intercept, no regressors
  intercept_v     — intercept params + u_1 modulates v

first_fixation drives omega's identifiability, so it is a CLI switch:
  random  — standard aDDM; the first-fixation advantage averages out, leaving
            omega visible only through RT-distribution shape
  item0   — gaze always starts on item 0, turning that advantage into a
            systematic choice bias recoverable from marginal choice proportions
Measured at 500 trials/dataset, v_diff=0: P(choose 0) runs 0.578 -> 0.506 across
omega 0 -> 1 under "item0", and is flat (~0.50) under "random".
"""
from experiments.paths import RESULTS_ROOT
import os
os.environ["KERAS_BACKEND"] = "torch"

import argparse
import logging
import numpy as np
try:
    import wandb
except Exception:
    wandb = None
import torch
import keras
import bayesflow as bf
import matplotlib.pyplot as plt

from pathlib import Path

from cogformer.simulator import ModelFamily
from cogformer.simulator.benchmarks.ssms.addm import ADDM
from cogformer.simulator.benchmarks.ssms.priors import addm_priors
from cogformer.simulator.benchmarks.ssms.link_funs import addm_link_fun
from cogformer.networks.transformers.cogformer import CogFormer

from ..utils import build_encoder_input_dim
from ..timing import RunTimer


INTRINSIC_PARAMS = ["v", "v_diff", "omega", "a", "tau", "s_v", "s_tau"]
VARIABLE_NAMES   = [r"$v$", r"$v_\mathrm{diff}$", r"$\omega$",
                    r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"]

DESIGN_CONFIGS = {
    "intercept_only": {
        "1": INTRINSIC_PARAMS,
    },
    "intercept_v": {
        "1": INTRINSIC_PARAMS,
        "u_1": ["v"],
    },
    # Trial-varying item values -- the design real aDDM experiments actually use,
    # where each trial shows a different pair of items whose values are KNOWN
    # (from a pre-experiment bidding task) and therefore enter as regressors.
    #   v_diff_t = b_d0 + b_d1 * u_1t   (the value DIFFERENCE channel)
    #   v_t      = b_v0 + b_v1 * u_2t   (the common-value / SUM channel)
    # This is what removes the flat direction WITHOUT pinning v. Under
    # intercept_only the data see only mu = (1+w)/2*v_diff and
    # Delta = (1-w)(2v + v_diff), two functionals for three parameters. Here
    # v_diff enters BOTH channels with the same b_d1, and the ratio of its
    # Delta-effect to its mu-effect is 2(1-w)/(1+w) -- free of b_d1, so omega is
    # structurally identified. Whether that is PRACTICALLY strong at 500 trials
    # is the empirical question this case exists to answer; the Delta channel is
    # still second-order, which is what defeated the v-fixed variant.
    "varying_values": {
        "1": INTRINSIC_PARAMS,
        "u_1": ["v_diff"],
        "u_2": ["v"],
    },
}

# Extra batch_sample kwargs per case. Item values are CONTINUOUS quantities, but
# build_discrete_mask assigns each regressor continuous/categorical at random
# (discrete_prob=0.5) unless told otherwise -- an empty categorical_regressors
# list pins both value regressors continuous.
CASE_SAMPLE_KWARGS = {
    "varying_values": {"categorical_regressors": []},
}

MASK_RANDOMIZER_KWARGS = {
    "free_intrinsics": ["v", "v_diff", "omega", "a", "tau"],
    "fixed_intrinsics": ["s_v", "s_tau"],
    "fixed_values": {},
}


# ── v-fixed variant ───────────────────────────────────────────────────────────
# With gaze unobserved and item values constant across trials, the data depend on
# (v, v_diff, omega) only through two functionals -- the effective drift
#     mu    = (1 + omega)/2 * v_diff
# and the attentional swing
#     Delta = (1 - omega) * (2v + v_diff),
# which drives both the first-fixation choice bias and the extra within-trial
# drift variance. Three parameters, two functionals: exactly one flat direction,
# and v is the coordinate that spans it -- it never enters mu at all. Measured at
# 200k trials, a=1, tau=0.3, s_v=s_tau=0, first_fixation=item0:
#     v=1.00 v_diff=0.500 omega=0.25 -> P(c=1)=0.5713, RT q10/50/90 .554/1.007/2.173
#     v=1.67 v_diff=0.417 omega=0.50 -> P(c=1)=0.5719,               .554/1.006/2.167
#     v=3.57 v_diff=0.357 omega=0.75 -> P(c=1)=0.5727,               .553/1.006/2.172
# Pinning v makes (v_diff, omega) <-> (mu, Delta) a bijection; omega then moves
# P(c=1) by 0.17-0.28 across [0, 1]. This is also the scientifically correct move:
# in Krajbich's design item values come from a pre-experiment bidding task, i.e.
# they are known covariates, not free parameters.
#
# V_FIXED is the PRIOR MEDIAN of v on the natural scale -- shifted_softplus(0.6),
# 0.6 being the prior location -- so the marginal data distribution stays as close
# as possible to the v-free variant. Fixed-value injection in ModelFamily.sample()
# BYPASSES the link function, so this is a post-link value, not an unconstrained one.
V_FIXED = 1.0375

# S_V_FIXED is the prior median of s_v on the natural scale -- shifted_softplus(-1.0),
# -1.0 being the prior location -- chosen exactly as V_FIXED was.
#
# WHY pinning s_v is a different trade from pinning v. Under intercept_only with
# constant item values the data see only mu = (1+omega)/2 * v_diff and
# Delta = (1-omega)(2v + v_diff): two functionals for three parameters, one flat
# direction, which is what --fix_v removes -- at the cost of v no longer being
# INFERRED. The across-trial drift variance is a third functional,
# s_v^2 * (1 + omega^2). Pinning s_v turns that into a constraint on omega, which
# closes the same system from the other side and leaves v free to be recovered.
#
# Measured before running: s_v is itself only weakly recovered (r = 0.22-0.34 across
# the 7-dim and varying-values arms) and its within-posterior correlation with both
# omega and v is ~0.00, so the channel this reasoning leans on is real but thin.
# That is the empirical question these variants exist to answer.
S_V_FIXED = 0.3133

# Parameters that may be pinned, with the post-link value injected for each.
PINNABLE = {"v": V_FIXED, "s_v": S_V_FIXED}


def _pinned(fix_v: bool, fix_s_v: bool) -> list:
    return [q for q, on in (("v", fix_v), ("s_v", fix_s_v)) if on]


def resolve_case(case: str, fix_v: bool, fix_s_v: bool = False):
    """Design config + mask-randomizer kwargs for a case, with v optionally pinned.

    Dropping "v" from the design config leaves its column of the parameter mask
    empty, which is exactly the condition under which ModelFamily.sample() injects
    fixed_values["v"] instead of running the link function. The inference target
    goes from 7 active cells to 6 (v_diff, omega, a, tau, s_v, s_tau).

    Train and validate MUST call this with the same fix_v, or their simulated data
    distributions diverge -- the same failure mode as the empty-fixed_values bug.
    """
    pins = _pinned(fix_v, fix_s_v)
    if not pins:
        return DESIGN_CONFIGS[case], MASK_RANDOMIZER_KWARGS
    if case != "intercept_only":
        raise ValueError(
            f"pinning {pins} is only defined for intercept_only, got {case!r}: "
            "intercept_v regresses v on u_1, and pinning it would leave that "
            "term empty."
        )
    return (
        {"1": [q for q in INTRINSIC_PARAMS if q not in pins]},
        {
            "free_intrinsics": [q for q in MASK_RANDOMIZER_KWARGS["free_intrinsics"]
                                if q not in pins],
            "fixed_intrinsics": MASK_RANDOMIZER_KWARGS["fixed_intrinsics"] + pins,
            "fixed_values": {q: PINNABLE[q] for q in pins},
        },
    )


_VARIANT_SUFFIX = {(): "", ("v",): "_vfixed", ("s_v",): "_svfixed",
                   ("v", "s_v"): "_vsvfixed"}


def _variant(p: Path, fix_v: bool, fix_s_v: bool = False) -> Path:
    """Route the v-fixed run to its own output tree, leaving 7-dim results in place."""
    suf = _VARIANT_SUFFIX[tuple(_pinned(fix_v, fix_s_v))]
    return p.with_name(p.name + suf) if suf else p

# MUST match the geometry the checkpoint was trained at: build_encoder_input_dim
# gives r2c2->9, r2c3->14, r4c4->72, r5c5->186, and load_state_dict fails on any
# mismatch at encoder.input_embedding. The model-class checkpoints are r4c4.
# Frozen-encoder arm colour; kept in step with addm_diagnostics.FROZEN_COLOR.
# THE arm colours for experiment 4. Defined once here and imported everywhere
# else -- FROZEN_COLOR was previously duplicated in addm_diagnostics.py and
# BASELINE_COLOR in three modules, which is how a palette drifts.
FROZEN_COLOR = "#4E2A84"      # frozen CogFormer encoder
BASELINE_COLOR = "#6969ff"    # learned SetTransformer baseline

MAX_NUM_REGRESSORS = 4
MAX_NUM_CATEGORIES = 4
MAX_ALTERNATIVES   = 4
NUM_ALTERNATIVES   = 2

CKPT_DIR    = (RESULTS_ROOT / "checkpoints/encoder_rep/bf/addm")
DATA_DIR    = (RESULTS_ROOT / "data/encoder_rep/addm")
FIGURES_DIR = (RESULTS_ROOT / "figures/encoder_rep/addm")


class CogFormerEncoderSummary:
    """Data-only encoder probe, with no family or coefficient-structure context.

    For structure-conditioned checkpoints this deliberately evaluates encoder(x)
    with context=None, rather than the posterior's encode_conditions path. This
    keeps the held-out-family probe independent of any assumed parameter mapping.
    """

    def __init__(self, cogformer: CogFormer, device: torch.device):
        self.encoder = cogformer.encoder
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.device = device

        n_total = MAX_NUM_REGRESSORS * (MAX_NUM_REGRESSORS + 1) // 2
        self.max_num_cols = n_total * (MAX_NUM_CATEGORIES - 1) + 1
        self._choices_denom = float(max(NUM_ALTERNATIVES - 1, 1))
        self._alt_val = float(NUM_ALTERNATIVES - 1) / float(max(MAX_ALTERNATIVES - 1, 1))

    @torch.no_grad()
    def summarize(
        self,
        design_matrices: np.ndarray,  # (B, T, max_num_cols)
        rts: np.ndarray,               # (B, T, 1)
        choices: np.ndarray,           # (B, T, 1)
    ) -> np.ndarray:                   # (B, seed_dim)
        batch_size, num_obs = design_matrices.shape[:2]
        choices_norm = choices / self._choices_denom
        alt_col = np.full((batch_size, num_obs, 1), self._alt_val, dtype=np.float32)
        # The encoder expects FIVE observation channels, matching Adapter.adapt:
        #   [design, rt, choices_normalized, alt_col, favored_idx, theta]
        # favored_idx is RDM-only and theta is CDM-only, so both are zero for aDDM —
        # but they must still be present or the input is 2 columns short and
        # encoder.input_embedding rejects it.
        fav_col = np.zeros((batch_size, num_obs, 1), dtype=np.float32)
        theta_col = np.zeros((batch_size, num_obs, 1), dtype=np.float32)
        input_data = np.concatenate(
            [design_matrices, rts, choices_norm, alt_col, fav_col, theta_col], axis=-1
        )
        return self.summarize_raw(input_data)

    @torch.no_grad()
    def summarize_raw(self, input_data: np.ndarray) -> np.ndarray:
        """Encode an ALREADY-ASSEMBLED per-trial tensor.

        The concatenation above is byte-identical to the ``data`` tensor
        ADDMBaselineSimulator hands its SetTransformer, so routing both arms
        through this entry point is what lets addm_arm_compare.py prove the two
        arms saw the same input rather than merely asserting it.
        """
        x = torch.from_numpy(input_data).to(dtype=torch.float32, device=self.device)
        tokens = self.encoder(x)   # (B, n_seeds, seed_dim)
        return tokens.mean(dim=1).cpu().numpy()  # (B, seed_dim)


class ADDMSimulator(bf.simulators.Simulator):

    def __init__(self, encoder_summary: CogFormerEncoderSummary, case: str, num_obs: int = 500,
                 first_fixation: str = "item0", fix_v: bool = False,
                 fix_s_v: bool = False):
        self.encoder_summary = encoder_summary
        self.design_config, mask_kwargs = resolve_case(case, fix_v, fix_s_v)
        self.num_obs = num_obs
        self.extra_sample_kwargs = CASE_SAMPLE_KWARGS.get(case, {})
        self.model_family = ModelFamily(
            model=ADDM(first_fixation=first_fixation),
            prior=addm_priors(),
            intrinsic_params=INTRINSIC_PARAMS,
            mask_randomizer_kwargs=mask_kwargs,
        )

    def sample(self, batch_size, **kwargs):
        if isinstance(batch_size, tuple):
            batch_size = batch_size[0]

        samples = self.model_family.batch_sample(
            batch_size=batch_size,
            num_obs=self.num_obs,
            design_config=self.design_config,
            max_num_regressors=MAX_NUM_REGRESSORS,
            max_num_categories=MAX_NUM_CATEGORIES,
            keep_intercept=True,
            fixed_config=True,
            flatten_param_outputs=True,
            # build_link_functions requires link_fun keys to be a subset of
            # intrinsic_params; addm_link_fun carries "pi", which this probe does
            # not infer (contamination is off here), so drop anything unlisted.
            link_fun={k: f for k, f in addm_link_fun().items() if k in INTRINSIC_PARAMS},
            **self.extra_sample_kwargs,
        )

        summaries = self.encoder_summary.summarize(
            design_matrices=samples["design_matrices"],
            rts=samples["sim_data"]["rts"],
            choices=samples["sim_data"]["choices"],
        )

        param_masks = samples["param_masks"].astype(np.float32)
        # Infer ONLY the active cells. The adapter DROPS "masks", so the mask
        # never reaches the network -- it is used post hoc in validate() to slice
        # the metrics, which made this easy to miss. Handing FlowMatching all 469
        # cells means ~98.5% of its MSE is spent annihilating base noise on
        # dimensions that are structurally zero in every dataset. The full grid
        # exists for CogFormer's DECODER, which consumes param_mask in its loss;
        # this arm uses only CogFormer's encoder, so nothing here requires it.
        active = param_masks[0].astype(bool)   # constant within a case (fixed_config=True)
        return {
            "summaries": summaries.astype(np.float32),
            "params":    samples["param_matrices"].astype(np.float32)[:, active],
            "masks":     param_masks,
        }


def active_labels(mask_row):
    """Names for the active cells of a flattened 67x7 design grid.

    Flat index is ``row * n_params + param``. Kept in step with the identical
    helper in addm_bf_baseline.py -- duplicated rather than imported so the two
    arms stay independently runnable. Replaces an older expression that indexed a
    469-long mask with range(7), correct only when the active cells are exactly
    the intercept row (true for intercept_only, wrong for intercept_v).
    """
    n = len(INTRINSIC_PARAMS)
    idx = np.where(np.asarray(mask_row).astype(bool))[0]
    plain, pretty = [], []
    for i in idx:
        row, par = divmod(int(i), n)
        suffix = "" if row == 0 else f"@r{row}"
        plain.append(INTRINSIC_PARAMS[par] + suffix)
        pretty.append(VARIABLE_NAMES[par] if row == 0
                      else VARIABLE_NAMES[par][:-1] + r"^{(" + str(row) + r")}$")
    return plain, pretty


def build_adapter():
    return (
        bf.Adapter()
        .drop(["masks"])
        .convert_dtype("float64", "float32")
        # The summary is PRECOMPUTED by the frozen CogFormer encoder, so there is no
        # BayesFlow summary network. "summary_variables" is reserved for input a
        # summary network consumes; a precomputed vector is an inference condition.
        .rename("summaries", "inference_conditions")
        .rename("params", "inference_variables")
    )


def _sample_chunked(approximator, key: str, conditions, num_samples: int, chunk: int):
    """Posterior draws in dataset-sized chunks, concatenated.

    ``approximator.sample`` flattens (n_datasets, num_samples) into ONE batch, so
    500 datasets x 500 draws builds a 250,000 x 256 activation inside
    FlowMatching's FiLM layer and OOMs a 31.4 GiB card at ~29 GiB. Chunking over
    datasets keeps peak memory linear in ``chunk`` and leaves the draws per
    dataset -- and therefore every downstream metric -- byte-identical.
    """
    outs = []
    for i in range(0, len(conditions), chunk):
        draws = approximator.sample(
            conditions={key: conditions[i:i + chunk]}, num_samples=num_samples
        )
        outs.append(np.asarray(draws["params"]))
    return np.concatenate(outs, axis=0)


def ckpt_path(case: str, fix_v: bool = False, fix_s_v: bool = False) -> str:
    return str(_variant(CKPT_DIR, fix_v, fix_s_v) / case / "addm_encoder_rep_fm")


def train(args):
    device = torch.device(args.encoder_device)

    cogformer = _load_cogformer(args, device)
    encoder_summary = CogFormerEncoderSummary(cogformer, device)
    simulator = ADDMSimulator(encoder_summary, case=args.case, num_obs=args.num_obs,
                              first_fixation=args.first_fixation, fix_v=args.fix_v,
                              fix_s_v=args.fix_s_v)

    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=build_adapter(),
        inference_network=bf.networks.FlowMatching(),
        checkpoint_filepath=ckpt_path(args.case, args.fix_v, args.fix_s_v),
    )

    logging.info(f"[aDDM/{args.case}] Training: {args.epochs} epochs × {args.steps_per_epoch} steps")
    workflow.fit_online(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
    )
    logging.info(f"[aDDM/{args.case}] Training complete.")


def validate(args):
    device = torch.device(args.encoder_device)

    cogformer = _load_cogformer(args, device)
    encoder_summary = CogFormerEncoderSummary(cogformer, device)
    simulator = ADDMSimulator(encoder_summary, case=args.case, num_obs=args.num_obs,
                              first_fixation=args.first_fixation, fix_v=args.fix_v,
                              fix_s_v=args.fix_s_v)

    approximator = keras.saving.load_model(
        ckpt_path(args.case, args.fix_v, args.fix_s_v) + "/model.keras"
    )

    val_sims = simulator.sample(args.val_batch_size)
    summaries = val_sims["summaries"]
    masks = val_sims["masks"]

    pred_active = _sample_chunked(approximator, "summaries", summaries,
                                  args.num_samples, args.sample_chunk_size)
    # The simulator now emits only the active cells, so no masking here.
    true_active = val_sims["params"]

    out_dir = _variant(DATA_DIR, args.fix_v, args.fix_s_v) / args.case
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "addm_encoder_rep_data.npz",
        summaries=summaries,
        true_params=true_active,
        pred_params=pred_active,
        param_masks=masks,
    )
    logging.info(f"[aDDM/{args.case}] Saved data to {out_dir}")

    # Recovery and the other three diagnostics are drawn by addm_diagnostics.py from
    # the npz written above, using bf.diagnostics.plots with FROZEN_COLOR. This script
    # used to ALSO write its own hand-rolled recovery scatter here. Both files tidied
    # to the same finals name (probe/<variant>/frozen/recovery.pdf), so whichever the
    # assembler copied second won -- and it was the hand-rolled one, which is why the
    # frozen arm's recovery plot looked nothing like the baseline's. Removed rather
    # than renamed: it duplicated addm_diagnostics with a worse plot.
    _ = active_labels(masks[0])



def _load_cogformer(args, device):
    encoder_input_dim = build_encoder_input_dim(MAX_NUM_REGRESSORS, MAX_NUM_CATEGORIES, keep_intercept=True)
    cogformer = CogFormer(
        encoder_input_dim=encoder_input_dim,
        encoder_num_layers=args.encoder_num_layers,
        decoder_num_layers=args.decoder_num_layers,
        encoder_num_heads=args.encoder_num_heads,
        encoder_embed_dim=args.encoder_embed_dim,
        decoder_num_heads=args.decoder_num_heads,
        num_seeds=args.num_seeds,
        seed_dim=args.seed_dim,
        proj_dim=args.proj_dim,
        dropout=0.0,
        layer_dropout=0.0,
        time_embedding_dim=args.time_embedding_dim,
        pos_embedding_dim=args.pos_embedding_dim,
        num_models=args.num_models,
        model_embed_dim=args.model_embed_dim,
        decoder_layer_design="mixed_attention",
        decoder_layer_kwargs={"mab_first": True},
        encoder_structure_conditioning=getattr(args, "encoder_structure_conditioning", False),
    ).to(device)
    raw = torch.load(Path(args.cogformer_checkpoint), map_location=device, weights_only=False)
    # model-class checkpoints wrap the weights and carry a torch.compile prefix
    state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    cogformer.load_state_dict(state)
    cogformer.eval()
    return cogformer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, required=True, choices=["train", "validate"])
    p.add_argument("--case", type=str, default="intercept_only",
                   choices=list(DESIGN_CONFIGS.keys()))
    p.add_argument("--first_fixation", type=str, default="item0",
                   choices=["random", "item0", "item1"],
                   help="Gaze-onset rule. 'item0' makes omega recoverable from marginal "
                        "choice proportions; 'random' is standard aDDM but leaves omega "
                        "identifiable only through RT-distribution shape.")
    p.add_argument("--fix_s_v", action="store_true",
                   help="Pin across-trial drift variability s_v to S_V_FIXED instead of "
                        "inferring it. Combine with --fix_v to pin both. Each combination "
                        "writes to its own tree (_svfixed / _vsvfixed). intercept_only only.")
    p.add_argument("--fix_v", action="store_true",
                   help="Pin the common value v to V_FIXED instead of inferring it, "
                        "removing the (v, v_diff, omega) flat direction. Drops the "
                        "target from 7 active cells to 6 and routes all outputs to a "
                        "*_vfixed tree. intercept_only only.")
    p.add_argument("--num_models", type=int, default=3,
                   help="Family count the checkpoint was trained with (original=3, "
                        "LBA-finetuned=4). Only sizes model_embedding, which the encoder "
                        "never touches, but load_state_dict is strict about it.")
    p.add_argument("--cogformer_checkpoint", type=str, required=True,
                   help="Path to the encoder_rep CogFormer checkpoint (.pt)")
    p.add_argument("--encoder_device", type=str, default="cpu",
                   help="Device for CogFormer encoder (cpu or cuda)")

    # Training
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--steps_per_epoch", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_obs", type=int, default=500)
    p.add_argument("--dry_run_epochs", type=int, default=None,
                   help="If set, train for this many epochs only and record extrapolated time.")
    p.add_argument("--use_wandb", action="store_true")

    # Validation
    p.add_argument("--sample_chunk_size", type=int, default=50,
                   help="Datasets per approximator.sample call at validation. The call "
                        "flattens datasets x draws into one batch, so 500x500 OOMs a "
                        "31 GiB card; chunking changes peak memory only, not results.")
    p.add_argument("--val_batch_size", type=int, default=200)
    p.add_argument("--num_samples", type=int, default=500)

    # CogFormer architecture (must match checkpoint)
    p.add_argument("--encoder_structure_conditioning", action="store_true",
                   help="Load conditioned weights strictly; the representation probe uses data only (context=None).")
    p.add_argument("--encoder_num_layers", type=int, default=8)
    p.add_argument("--decoder_num_layers", type=int, default=8)
    p.add_argument("--encoder_num_heads", type=int, default=8)
    p.add_argument("--encoder_embed_dim", type=int, default=64)
    p.add_argument("--decoder_num_heads", type=int, default=8)
    p.add_argument("--num_seeds", type=int, default=32)
    p.add_argument("--seed_dim", type=int, default=128)
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--model_embed_dim", type=int, default=8)
    p.add_argument("--time_embedding_dim", type=int, default=32)
    p.add_argument("--pos_embedding_dim", type=int, default=32)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if args.mode == "train":
        if args.use_wandb:
            wandb.init(
                project="cogformer-encoder-rep",
                tags=["CogFormer", "LCA", "BF", "EncoderRep"],
                config={
                    "case": args.case,
                    "epochs": args.epochs,
                    "steps_per_epoch": args.steps_per_epoch,
                    "batch_size": args.batch_size,
                    "num_obs": args.num_obs,
                },
            )
        is_dry_run = args.dry_run_epochs is not None
        total_epochs = args.epochs
        if is_dry_run:
            args.epochs = args.dry_run_epochs
        with RunTimer(f"addm_bf_{args.case}", dry_run=is_dry_run,
                      total_epochs=total_epochs, dry_run_epochs=args.dry_run_epochs):
            train(args)
        if args.use_wandb:
            wandb.finish()
    else:
        validate(args)
