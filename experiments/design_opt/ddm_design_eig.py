"""Static design optimisation: what does discretising a predictor cost you?

A psychologist has two predictors of drift rate and must decide, for each, whether to
measure it continuously or reduce it to two levels -- the median-split question. This
sweeps that choice against trial count and reports the information gain of each design.

WHY THIS IS CHEAP
-----------------
CogFormer amortises over designs: the design matrix is a CONDITIONING INPUT, not
something baked into training. So every cell of the sweep is a forward pass on an
existing checkpoint. A per-design NPE or per-design MCMC would need one fit per cell;
here the whole grid costs one pass each.

FOUR DESIGNS, MATCHED DIMENSION
-------------------------------
u_1 and u_2 each continuous (C) or binary (B): CC, CB, BC, BB. Binary is the only
categorical level count that keeps this honest. A K-level categorical contributes
K-1 design columns, so for K > 2 the parameter vector changes DIMENSION between
designs and EIG becomes a log-volume in a different-dimensional space -- not
comparable. At K = 2 a categorical regressor contributes exactly one column, same as
a continuous one, so all four designs share an identical parameter mask and the four
numbers can be read against each other directly. The script asserts that.

THE MEASURE -- AND WHY IT IS NOT RAW EIG ON THE COEFFICIENTS
------------------------------------------------------------
Matching the parameter DIMENSION across designs is necessary but NOT sufficient.
Measured directly, a binary design differs from a continuous one on two more axes:

  * PRIOR SCALE. Under coefficient_prior="cell_means" a categorical contrast is a
    difference of two latent draws, so its prior variance is 2 against N(0,1) for a
    continuous slope (measured prior sd 1.415/1.360 vs 0.962/0.980). That alone gives
    the all-binary design 1.74 nats more prior entropy -- larger than any design
    effect we are trying to detect.
  * REGRESSOR SCALE. Continuous columns are standardised to sd 1; a binary column has
    sd sqrt(p(1-p)), measured at 0.329 and 0.488. A unit coefficient therefore means a
    much smaller change in drift under the binary design.

So raw EIG on the coefficients would rank designs by their prior widths, not by their
information. Everything below is computed on a COMMON ESTIMAND instead: each slope
cell is rescaled by the realised sd of its own design column, putting it in
"change in the linear predictor per sd of the predictor as actually run" -- the same
physical quantity for a continuous and a binary regressor. Intercept cells are
untouched (their priors already match across designs to <2%).

The PRIMARY metric is EIG, the log-volume RATIO

    EIG = 0.5 * logdet(Sigma_prior) - 0.5 * E_x[ logdet(Sigma_post(x)) ]

because a ratio cancels the rescaling above, and indeed any fixed linear reparameter-
isation -- which is exactly what makes it immune to the regressor-scale difference.
``post_logvol`` (-0.5 * E[logdet Sigma_post]) is written out too but is DIAGNOSTIC
ONLY: as an absolute volume it still carries the scale, and it ranks the all-binary
design "most certain" purely because rescaling by ~0.33 shrinks its estimand.

Read EIG with one caveat that no reparameterisation removes: it is information gain
relative to EACH DESIGN'S OWN prior, and those priors differ (measured 1.37 nats per
binary regressor, i.e. variance ratio exactly 2 per cell, the cell_means contrast).
A wider prior means more available to learn. ``logdet_prior`` is in the CSV per design
so that difference stays visible rather than buried in the headline number.

BALANCE
-------
Category proportions are Dirichlet-drawn, so "binary" here means a randomly
UNBALANCED dichotomisation, not a median split -- the realised split is recorded per
cell as ``mean_split_balance`` (0.5 = balanced). That is a real design cost and
belongs in the result, but it means these curves answer "factor vs parametric", not
"median split vs continuous".

FORMER MEASURE
--------------
CogFormer's decoder exposes `sample` but no `log_prob`, so the exact
E[log q(theta|x)] - E[log p(theta)] estimator used for the aDDM arms is unavailable.
Gaussian EIG instead:

    EIG = 0.5 * logdet(Sigma_prior) - 0.5 * E_x[ logdet(Sigma_post(x)) ]

the log-volume contraction of the posterior, in nats. Sigma_prior is estimated from
the same simulated draws, so both terms carry the same Gaussian approximation and the
DIFFERENCE between designs -- the quantity of interest -- stays meaningful even where
the absolute value is only approximate. Per-parameter contraction is reported
alongside as a sanity check.

TRIAL-COUNT RANGE
-----------------
The sweep stays inside [200, 1000], the num_obs range these checkpoints were trained
on (shared_config.sh MIN_OBS/MAX_OBS). Below 200 the curve would stop measuring
information and start measuring encoder degradation out of distribution -- which
looks identical on a plot and is not the same thing. --allow_ood lifts the guard for
deliberate OOD probing; the output marks those rows.
"""
import argparse
import logging
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cogformer.adapter import Adapter
from cogformer.networks.transformers.cogformer import CogFormer
from cogformer.simulator import ModelFamily
from cogformer.simulator.model_class import ModelClass
from cogformer.simulator.benchmarks.ssms.ddm import DDM
from cogformer.simulator.benchmarks.ssms.rdm import RDM
from cogformer.simulator.benchmarks.ssms.cdm import CDM
from cogformer.simulator.benchmarks.ssms.priors import ddm_priors, rdm_priors, cdm_priors
from cogformer.simulator.benchmarks.ssms.link_funs import (
    ddm_link_fun, rdm_link_fun, cdm_link_fun)

from ..configs import MODELS as _BASE_MODELS
from ..utils import (add_global_params, infer_free_fixed_intrinsics,
                     build_encoder_input_dim, chunked_sample)

GLOBAL_PARAMS = ["pi"]
GLOBAL_VARIABLE_NAMES = [r"$\pi$"]

# Two predictors, each loading on BOTH drift rate and threshold: "do my two factors
# act on the evidence or on the caution?" -- 4 slope cells (v,a) x (u_1,u_2) plus 7
# intercept cells = 11 active. This mirrors the trained `regressed` benchmark config
# (u_1/u_2 -> v,a,z), so it is in-distribution for the checkpoint. pi is appended to
# the intercept column because training designs always include the global params --
# a mask without pi is OOD.
#
# NOTE the estimand widens with it: the comparison is now over an 11-dim posterior,
# so the discretisation cost is aggregated over drift AND threshold effects rather
# than drift alone. Per-cell contraction is saved so the two can be read apart.
DESIGN_CONFIG = {
    "1":   ["v", "a", "z", "tau", "s_v", "s_tau", "pi"],
    "u_1": ["v", "a"],
    "u_2": ["v", "a"],
    "u_1:u_2": [],
}

# (label, which regressors are categorical). C = continuous, B = binary two-level.
DESIGNS = [
    ("CC", []),
    ("CB", ["u_2"]),
    ("BC", ["u_1"]),
    ("BB", ["u_1", "u_2"]),
]
DESIGN_COLORS = {"CC": "#1b6ca8", "CB": "#6969ff", "BC": "#e08214", "BB": "#b2182b"}

TRAIN_MIN_OBS, TRAIN_MAX_OBS = 200, 1000

# batch_size * samples_chunk_size to target when --samples_chunk_size is unset.
# 1000 keeps peak well under a 31 GiB card while sharing it with a training job.
CHUNK_BUDGET = 1000


def build_model_class(seed):
    workers = 1 if seed is not None else None
    cfgs = {m: add_global_params(c, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
            for m, c in _BASE_MODELS.items() if m != "lba"}
    families = {
        "DDM": ModelFamily(model=DDM(), name="DDM", prior=ddm_priors(),
                           num_workers=workers, intrinsic_params=cfgs["ddm"]["intrinsic_params"]),
        "RDM": ModelFamily(model=RDM(), name="RDM", prior=rdm_priors(),
                           num_workers=workers, intrinsic_params=cfgs["rdm"]["intrinsic_params"]),
        "CDM": ModelFamily(model=CDM(), name="CDM", prior=cdm_priors(),
                           num_workers=workers, intrinsic_params=cfgs["cdm"]["intrinsic_params"]),
    }
    links = {"DDM": ddm_link_fun(), "RDM": rdm_link_fun(), "CDM": cdm_link_fun()}
    return ModelClass(model_families=families, link_funs=links), cfgs["ddm"]


def load_cogformer(args, device):
    cogformer = CogFormer(
        encoder_input_dim=build_encoder_input_dim(
            args.max_num_regressors, args.max_num_categories, keep_intercept=True),
        encoder_num_layers=args.encoder_num_layers,
        decoder_num_layers=args.decoder_num_layers,
        encoder_num_heads=args.encoder_num_heads,
        decoder_num_heads=args.decoder_num_heads,
        num_seeds=args.num_seeds, seed_dim=args.seed_dim, proj_dim=args.proj_dim,
        dropout=0.0, layer_dropout=0.0,
        time_embedding_dim=args.time_embedding_dim,
        pos_embedding_dim=args.pos_embedding_dim,
        num_models=args.num_models, model_embed_dim=args.model_embed_dim,
        decoder_layer_design="mixed_attention",
        decoder_layer_kwargs={"mab_first": True},
    ).to(device)
    raw = torch.load(Path(args.checkpoint), map_location=device, weights_only=False)
    state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    cogformer.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state.items()})
    cogformer.eval()
    return cogformer


def resolve_chunk(args):
    """Samples per ODE pass. Peak GPU memory scales with batch_size * chunk *
    num_tokens, and the decoder's O(N^2) attention makes that the binding constraint:
    batch 40 x 100 samples in one pass asked for 21.85 GiB on a 31 GiB card. Auto mode
    holds batch_size * chunk near CHUNK_BUDGET so the peak is independent of how many
    datasets a cell uses.
    """
    if args.samples_chunk_size is not None:
        return args.samples_chunk_size
    return max(1, CHUNK_BUDGET // max(args.batch_size, 1))


def gaussian_logdet(x, ridge=1e-8):
    """logdet of a sample covariance, with a ridge for numerical safety.

    The ridge matters: a parameter the design leaves almost unconstrained gives a
    posterior column whose variance is nearly the prior's, but a cell that is
    structurally pinned would give a singular block and -inf.
    """
    c = np.cov(x, rowvar=False)
    c = np.atleast_2d(c)
    c = c + ridge * np.eye(c.shape[0])
    sign, logdet = np.linalg.slogdet(c)
    if sign <= 0:
        return np.nan
    return float(logdet)


def active_cell_scales(design_matrices, mask):
    """Per-dataset rescaling that puts every active cell on a common estimand.

    Active cell (row r, param p) multiplies design column r, so the coefficient's
    effect on the linear predictor is beta * sd(column r). Rescaling by that sd makes
    a continuous slope and a binary contrast the same physical quantity -- change in
    the linear predictor per sd of the predictor as actually run -- which is what
    makes the designs comparable at all. The intercept row (r = 0) is left alone.
    """
    b, _, n_cols = design_matrices.shape
    n_rows, n_local = mask.shape
    rows, _ = np.nonzero(mask)
    scales = np.ones((b, int(mask.sum())), dtype=np.float64)
    for k, r in enumerate(rows):
        if r == 0:
            continue
        if r < n_cols:
            scales[:, k] = design_matrices[:, :, r].std(axis=1)
    return scales


def simulate_cell(args, model_class, ddm_cfg, categorical_regressors, num_obs,
                  batch_size):
    """Simulate one (design, num_obs) cell and return the pieces both callers need."""
    mf = model_class.model_families["DDM"]
    free, fixed, fixed_values = infer_free_fixed_intrinsics(
        DESIGN_CONFIG, ddm_cfg["intrinsic_params"], ddm_cfg["default_fixed_values"])
    mask_kwargs = {"free_intrinsics": free, "fixed_intrinsics": fixed,
                   "fixed_values": fixed_values}

    samples = mf.batch_sample(
        design_config=DESIGN_CONFIG,
        batch_size=batch_size,
        num_obs=num_obs,
        mask_randomizer_kwargs=mask_kwargs,
        max_num_regressors=args.max_num_regressors,
        # K=2 for every design: a binary categorical contributes exactly one column,
        # matching a continuous regressor, so the parameter mask is identical across
        # all four designs and the EIG values are comparable.
        max_num_categories=2, min_num_categories=2,
        categorical_regressors=categorical_regressors,
        keep_intercept=True, add_interaction=False,
        flatten_param_outputs=True, fixed_config=True,
        link_fun=model_class.link_funs["DDM"],
        pad_to_max_num_categories=args.max_num_categories,
    )
    samples["model_ids"] = np.full(batch_size, model_class.model_index["DDM"],
                                   dtype=np.int64)
    samples["param_matrices"], samples["param_masks"] = model_class.lift_to_global_space(
        "DDM", samples["param_matrices"], samples["param_masks"])
    n_global = model_class.n_global_params
    samples["n_params"] = n_global
    gidx = model_class.local_to_global["DDM"]

    return samples, n_global, gidx


def _unpack(adapted, samples, model_class, gidx, batch_size):
    n_global = model_class.n_global_params
    true_flat = adapted["param_matrices"].detach().cpu().numpy()
    n_rows = true_flat.shape[1] // n_global
    true_set = true_flat.reshape(batch_size, n_rows, n_global)[:, :, gidx]
    mask = adapted["param_masks"].detach().cpu().numpy()
    mask = mask.reshape(batch_size, n_rows, n_global)[0][:, gidx].astype(bool)
    scales = active_cell_scales(samples["design_matrices"], mask)
    return true_set[:, mask] * scales, mask, n_rows, scales


def prior_logdet(args, model_class, ddm_cfg, adapter, categorical_regressors, num_obs):
    """logdet(Sigma_prior) for one DESIGN, from a large dedicated draw.

    Estimated once per design rather than per cell. The parameter prior does not
    depend on num_obs, but it DOES depend on the design: under cell_means a
    categorical contrast has prior variance 2 against 1 for a continuous slope. Fitting
    it from only batch_size draws made logdet_prior swing by ~3 nats between cells that
    should share it, and that noise landed entirely in EIG.
    """
    samples, _, gidx = simulate_cell(args, model_class, ddm_cfg,
                                     categorical_regressors, num_obs, args.prior_batch)
    adapted = adapter.adapt(samples)
    true_active, _, _, _ = _unpack(adapted, samples, model_class, gidx, args.prior_batch)
    return gaussian_logdet(true_active)


def run_cell(args, model_class, ddm_cfg, cogformer, adapter, device,
             design_label, categorical_regressors, num_obs, logdet_prior):
    samples, n_global, gidx = simulate_cell(args, model_class, ddm_cfg,
                                            categorical_regressors, num_obs,
                                            args.batch_size)
    adapted = adapter.adapt(samples)
    true_active, mask, n_rows, scales = _unpack(adapted, samples, model_class, gidx,
                                                args.batch_size)

    raw = chunked_sample(cogformer, adapted, num_samples=args.num_samples,
                         steps=args.num_sample_steps, solver=args.solver,
                         samples_chunk_size=resolve_chunk(args))
    pred_set = raw.reshape(args.batch_size, args.num_samples, n_rows,
                           model_class.n_global_params)[:, :, :, gidx]
    # Common estimand: slope cells in "per sd of the predictor as run" units, so a
    # continuous slope and a binary contrast measure the same physical thing.
    pred_active = pred_set[:, :, mask] * scales[:, None, :]

    logdet_post = np.array([gaussian_logdet(pred_active[i]) for i in range(args.batch_size)])
    post_logvol = -0.5 * np.nanmean(logdet_post)          # PRIMARY: higher = more certain
    post_se = 0.5 * np.nanstd(logdet_post, ddof=1) / np.sqrt(np.isfinite(logdet_post).sum())
    eig = 0.5 * logdet_prior + post_logvol

    contraction = 1.0 - (pred_active.std(axis=1) ** 2).mean(axis=0) / true_active.var(axis=0)

    # Realised split balance for the categorical regressors: sd = sqrt(p(1-p)) => the
    # closer to 0.5 the more balanced. 1.0 is recorded for all-continuous designs.
    slope_rows = sorted({r for r in np.nonzero(mask)[0] if r != 0})
    if categorical_regressors and slope_rows:
        sds = np.concatenate([samples["design_matrices"][:, :, r].std(axis=1)
                              for r in slope_rows if r < samples["design_matrices"].shape[2]])
        balance = float(np.mean(0.5 - np.sqrt(np.clip(0.25 - sds**2, 0, None))))
    else:
        balance = float("nan")

    return {
        "design": design_label, "num_obs": num_obs, "n_active": int(mask.sum()),
        "post_logvol": post_logvol, "post_se": post_se,
        "logdet_prior": logdet_prior, "eig_nats": eig,
        "mean_contraction": float(np.mean(contraction)),
        "mean_split_balance": balance,
        "ood": num_obs < TRAIN_MIN_OBS or num_obs > TRAIN_MAX_OBS,
        "mask_signature": "".join("1" if b else "0" for b in mask.ravel()),
    }, contraction


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    model_class, ddm_cfg = build_model_class(args.seed)
    cogformer = load_cogformer(args, device)
    adapter = Adapter()

    obs_grid = [int(n) for n in args.num_obs_grid.split(",")]
    bad = [n for n in obs_grid if (n < TRAIN_MIN_OBS or n > TRAIN_MAX_OBS)]
    if bad and not args.allow_ood:
        raise SystemExit(
            f"num_obs {bad} lies outside the trained range [{TRAIN_MIN_OBS}, {TRAIN_MAX_OBS}]. "
            "Those points would measure encoder degradation, not information. "
            "Pass --allow_ood to include them anyway; they will be marked in the output."
        )

    # One prior estimate per design, reused across the num_obs grid.
    prior_ld = {}
    for label, cats in DESIGNS:
        prior_ld[label] = prior_logdet(args, model_class, ddm_cfg, adapter, cats,
                                       obs_grid[0])
        logging.info(f"[{label}] logdet(Sigma_prior) = {prior_ld[label]:.4f} "
                     f"({args.prior_batch} draws)")

    rows, contractions, signatures = [], {}, set()
    for (label, cats), n in product(DESIGNS, obs_grid):
        if args.seed is not None:
            np.random.seed(args.seed + hash((label, n)) % 100_000)
            torch.manual_seed(args.seed + hash((label, n)) % 100_000)
        logging.info(f"[{label} / num_obs={n}] simulating + sampling")
        row, contr = run_cell(args, model_class, ddm_cfg, cogformer, adapter, device,
                              label, cats, n, prior_ld[label])
        rows.append(row); contractions[(label, n)] = contr
        signatures.add(row["mask_signature"])
        logging.info(
            f"[{label} / num_obs={n}] post_logvol {row['post_logvol']:.3f} +/- "
            f"{row['post_se']:.3f}  (logdet_prior {row['logdet_prior']:.3f}, "
            f"EIG {row['eig_nats']:.3f}, contraction {row['mean_contraction']:.3f})")

    # The comparison is only meaningful if every design produced the same parameter
    # mask; otherwise EIG values are log-volumes in different-dimensional spaces.
    if len(signatures) != 1:
        raise SystemExit(
            f"Designs produced {len(signatures)} distinct parameter masks. EIG is not "
            "comparable across them -- check that max_num_categories=2 held for every cell."
        )

    df = pd.DataFrame(rows).drop(columns=["mask_signature"])
    df.to_csv(outdir / "design_eig.csv", index=False, float_format="%.4f")
    np.savez(outdir / "design_contraction.npz",
             **{f"{l}_{n}": c for (l, n), c in contractions.items()})
    logging.info("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    _plot(df, outdir)
    logging.info(f"[saved] {outdir}")


def _plot(df, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for label in [d[0] for d in DESIGNS]:
        s = df[df.design == label].sort_values("num_obs")
        c = DESIGN_COLORS[label]
        axes[0].errorbar(s.num_obs, s.eig_nats, yerr=s.post_se, marker="o", ms=4,
                         capsize=3, color=c, label=label)
        axes[1].plot(s.num_obs, s.mean_contraction, marker="o", ms=4, color=c, label=label)
    axes[0].set_ylabel("expected information gain (nats)")
    axes[1].set_ylabel("mean posterior contraction")
    for ax in axes:
        ax.set_xlabel("trials per dataset")
        ax.legend(title="u₁u₂  (C=continuous, B=binary)", fontsize=8)
        ax.grid(alpha=0.25)
    axes[0].set_title("Information gain by design")
    axes[1].set_title("Contraction by design")
    fig.tight_layout()
    fig.savefig(outdir / "design_eig.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--outdir", type=str, default="./outputs/data/design_opt/ddm")
    p.add_argument("--num_obs_grid", type=str, default="200,400,600,800,1000")
    p.add_argument("--allow_ood", action="store_true",
                   help=f"Permit num_obs outside the trained [{TRAIN_MIN_OBS}, {TRAIN_MAX_OBS}] range.")
    p.add_argument("--batch_size", type=int, default=200, help="datasets per cell")
    p.add_argument("--num_samples", type=int, default=200, help="posterior draws per dataset")
    p.add_argument("--num_sample_steps", type=int, default=50)
    p.add_argument("--solver", type=str, default="euler")
    p.add_argument("--samples_chunk_size", type=int, default=None,
                   help=f"Samples per ODE pass. Default auto: CHUNK_BUDGET({CHUNK_BUDGET}) "
                        "// batch_size, which keeps peak memory independent of batch_size.")
    p.add_argument("--prior_batch", type=int, default=1500,
                   help="Draws used for the per-design prior covariance. Estimated once "
                        "per design, not per cell -- the prior is num_obs-independent, and "
                        "a small per-cell estimate injects ~3 nats of noise into EIG.")
    p.add_argument("--seed", type=int, default=0)
    # Geometry -- must match the checkpoint (the MC checkpoints are r4c4)
    p.add_argument("--max_num_regressors", type=int, default=4)
    p.add_argument("--max_num_categories", type=int, default=4)
    p.add_argument("--num_models", type=int, default=3)
    p.add_argument("--encoder_num_layers", type=int, default=8)
    p.add_argument("--decoder_num_layers", type=int, default=8)
    p.add_argument("--encoder_num_heads", type=int, default=8)
    p.add_argument("--decoder_num_heads", type=int, default=8)
    p.add_argument("--num_seeds", type=int, default=32)
    p.add_argument("--seed_dim", type=int, default=128)
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--model_embed_dim", type=int, default=8)
    p.add_argument("--time_embedding_dim", type=int, default=32)
    p.add_argument("--pos_embedding_dim", type=int, default=32)
    return p.parse_args()


if __name__ == "__main__":
    main()
