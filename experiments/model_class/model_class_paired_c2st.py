"""
Paired C2ST between the original and LBA-finetuned checkpoints — the "forgetting" arm.

Asks: did finetuning on LBA change the posteriors the model produces for the three
families it was originally trained on (DDM/CDM/RDM)?

Unlike ``model_family_c2st.py`` (which compares CogFormer against BayesFlow and so
needs ``reshape_bf_to_cf`` to bridge layouts), both inputs here are CogFormer-native
predictions with identical shapes, identical ``params_mask``, and — critically —
identical ``true_set``: the two checkpoints were evaluated on the same simulated
datasets. That means the comparison is exactly paired and no re-simulation is needed.
The script verifies this rather than assuming it.

Interpreting the output
-----------------------
C2ST measures *distinguishability*, not quality. For this question a low value is the
desired result: it means finetuning left the posterior alone. But 0.5 is NOT the
achievable floor. With tens of thousands of pooled draws an MLP will separate two
distributions slightly even when they are identical up to Monte Carlo noise, so a raw
0.56 is uninterpretable on its own.

So every comparison is reported against a null measured the same way, at the same
sample size: split a single arm's posterior *draws* in half and run C2ST of that arm
against itself. Same checkpoint, same datasets, same n — the only thing separating the
two halves is finite-draw noise. That is the empirical floor.

The draw split (not a dataset split) is the right null because the real comparison
holds datasets fixed. Splitting by dataset would fold between-dataset variation into
the floor and overstate it.

Read ``excess`` in the output, not ``c2st``:

    excess = c2st_real_matched - max(null_orig, null_ft)

``excess`` near zero => the two checkpoints are indistinguishable beyond noise.

Sample sizes are matched by construction: the null uses draws[:h] vs draws[h:] within
one arm (h = num_draws // 2), and ``c2st_real_matched`` uses orig draws[:h] vs ft
draws[:h]. ``c2st_real_full`` additionally uses all draws on both sides for maximum
power, but is not comparable to the null and is reported for reference only.
"""

import argparse
import itertools
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from cogformer.diagnostics.metric.adaptive_c2st import compute_c2st_accuracy

from ..configs import MODELS as _BASE_MODELS
from ..utils import add_global_params

GLOBAL_PARAMS = ["pi"]
GLOBAL_VARIABLE_NAMES = [r"$\pi$"]

# The forgetting arm covers only the families the original checkpoint was trained on.
FORGETTING_FAMILIES = ["ddm", "cdm", "rdm"]

# Scenarios present in the paired run.
DEFAULT_SCENARIOS = [
    "intercept_only", "regressed", "fixed",
    "fixed_regressed", "interaction", "mixed_predictor",
]

MODELS = {
    m: add_global_params(cfg, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
    for m, cfg in _BASE_MODELS.items()
    if m in FORGETTING_FAMILIES
}


def load_pred(path):
    """Load a CogFormer prediction file. Returns (pred_set, true_set, params_mask)."""
    d = np.load(path, allow_pickle=True)
    mask = d["params_mask"]
    if mask.ndim == 3:
        mask = mask[0]
    return d["pred_set"], d["true_set"], mask


def cell_labels(design_config, variable_names, mask, max_num_categories):
    """Label every active (r, c) cell the same way adaptive_c2st does."""
    regressor_keys = list(design_config.keys())
    labels = {}
    for r, c in zip(*np.where(mask == 1.0)):
        if r == 0:
            suffix = None
        else:
            category_id = (r - 1) % (max_num_categories - 1) + 1
            regressor_id = (r - 1) // (max_num_categories - 1) + 1
            if regressor_id >= len(regressor_keys):
                suffix = f"row{r}"
            else:
                suffix = f"${regressor_keys[regressor_id]}$ | $c_{{{category_id}}}$"
        col = variable_names[c] if c < len(variable_names) else f"col{c}"
        labels[(int(r), int(c))] = col if suffix is None else f"{col} -- {suffix}"
    return labels


def _cell_task(args):
    """One active cell: real (full + n-matched) and both within-arm nulls."""
    (r, c), orig_cell, ft_cell, half, n_splits, max_iter, seed = args
    # orig_cell / ft_cell are (batch, draws) slices for this cell.
    real_full = compute_c2st_accuracy(
        orig_cell.ravel(), ft_cell.ravel(),
        n_splits=n_splits, max_iter=max_iter, random_state=seed,
    )
    real_matched = compute_c2st_accuracy(
        orig_cell[:, :half].ravel(), ft_cell[:, :half].ravel(),
        n_splits=n_splits, max_iter=max_iter, random_state=seed,
    )
    null_orig = compute_c2st_accuracy(
        orig_cell[:, :half].ravel(), orig_cell[:, half:].ravel(),
        n_splits=n_splits, max_iter=max_iter, random_state=seed,
    )
    null_ft = compute_c2st_accuracy(
        ft_cell[:, :half].ravel(), ft_cell[:, half:].ravel(),
        n_splits=n_splits, max_iter=max_iter, random_state=seed,
    )
    return (r, c), real_full, real_matched, null_orig, null_ft


def run_pair(family, scenario, orig_path, ft_path, cfg, outdir,
             max_num_categories, n_splits, max_iter, seed, workers,
             joint_hidden, skip_joint, strict):
    orig_pred, orig_true, orig_mask = load_pred(orig_path)
    ft_pred, ft_true, ft_mask = load_pred(ft_path)

    # --- the assumption this whole script rests on ---
    if orig_pred.shape != ft_pred.shape:
        raise ValueError(
            f"{family}/{scenario}: pred_set shape mismatch "
            f"{orig_pred.shape} vs {ft_pred.shape} — not a paired comparison."
        )
    if not np.array_equal(orig_mask, ft_mask):
        raise ValueError(f"{family}/{scenario}: params_mask differs between arms.")
    paired = (orig_true.shape == ft_true.shape) and np.array_equal(orig_true, ft_true)
    if not paired:
        msg = (f"{family}/{scenario}: true_set DIFFERS between arms — the two "
               f"checkpoints were not evaluated on the same datasets, so C2ST here "
               f"conflates checkpoint difference with dataset difference.")
        if strict:
            raise ValueError(msg)
        logging.warning(msg)

    mask = orig_mask
    batch, draws, num_rows, num_cols = orig_pred.shape
    half = draws // 2
    if half < 2:
        raise ValueError(f"{family}/{scenario}: need >=4 draws to build a null, got {draws}.")

    design_config = cfg["benchmark_configs"][scenario]
    labels = cell_labels(design_config, cfg["variable_names"], mask, max_num_categories)
    active = sorted(labels.keys())
    logging.info(
        f"[{family}/{scenario}] {len(active)} active cells, "
        f"batch={batch} draws={draws} (null half={half}), paired={paired}"
    )

    tasks = [
        ((r, c), orig_pred[:, :, r, c], ft_pred[:, :, r, c],
         half, n_splits, max_iter, seed)
        for (r, c) in active
    ]

    results = {}
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_cell_task, t): t[0] for t in tasks}
            for fut in as_completed(futs):
                key, *vals = fut.result()
                results[key] = vals
    else:
        for t in tasks:
            key, *vals = _cell_task(t)
            results[key] = vals

    rows = []
    for (r, c) in active:
        real_full, real_matched, null_orig, null_ft = results[(r, c)]
        floor = max(null_orig, null_ft)
        rows.append({
            "parameter": labels[(r, c)], "row": r, "col": c,
            "c2st_real_full": real_full,
            "c2st_real_matched": real_matched,
            "null_orig": null_orig, "null_ft": null_ft,
            "null_floor": floor,
            "excess": real_matched - floor,
        })
    df = pd.DataFrame(rows).sort_values("excess", ascending=False)

    # --- joint C2ST over all active cells simultaneously ---
    joint = {}
    if not skip_joint:
        act = mask == 1.0
        n_active = int(act.sum())
        hidden = tuple(joint_hidden) if joint_hidden else "auto"
        a_full = orig_pred[:, :, act].reshape(-1, n_active)
        b_full = ft_pred[:, :, act].reshape(-1, n_active)
        a_h = orig_pred[:, :half, act].reshape(-1, n_active)
        b_h = ft_pred[:, :half, act].reshape(-1, n_active)
        a_h2 = orig_pred[:, half:, act].reshape(-1, n_active)
        b_h2 = ft_pred[:, half:, act].reshape(-1, n_active)
        logging.info(f"[{family}/{scenario}] joint C2ST over {n_active} dims, hidden={hidden}")
        kw = dict(n_splits=n_splits, hidden_layer_sizes=hidden,
                  max_iter=max_iter, random_state=seed)
        joint = {
            "joint_real_full": compute_c2st_accuracy(a_full, b_full, **kw),
            "joint_real_matched": compute_c2st_accuracy(a_h, b_h, **kw),
            "joint_null_orig": compute_c2st_accuracy(a_h, a_h2, **kw),
            "joint_null_ft": compute_c2st_accuracy(b_h, b_h2, **kw),
        }
        joint["joint_null_floor"] = max(joint["joint_null_orig"], joint["joint_null_ft"])
        joint["joint_excess"] = joint["joint_real_matched"] - joint["joint_null_floor"]

    cell_out = outdir / family
    cell_out.mkdir(parents=True, exist_ok=True)
    csv_path = cell_out / f"{family}_{scenario}_c2st.csv"
    df.to_csv(csv_path, index=False)
    logging.info(f"[saved] {csv_path}")

    summary = {
        "family": family.upper(), "scenario": scenario,
        "paired_datasets": paired, "n_active_cells": len(active),
        "n_samples_per_side_matched": batch * half,
        "mean_real_matched": float(df["c2st_real_matched"].mean()),
        "mean_null_floor": float(df["null_floor"].mean()),
        "mean_excess": float(df["excess"].mean()),
        "max_excess": float(df["excess"].max()),
        "worst_cell": str(df.iloc[0]["parameter"]),
        **joint,
    }
    return summary


def parse_args():
    p = argparse.ArgumentParser(
        description="Paired C2ST: original vs LBA-finetuned checkpoint (forgetting arm)."
    )
    base = "./outputs/figures/model_class/cf/paired_lba_20260816"
    p.add_argument("--orig_dir", type=str, default=f"{base}/original/predictions")
    p.add_argument("--ft_dir", type=str, default=f"{base}/finetuned/predictions")
    p.add_argument("--orig_suffix", type=str, default="_cf_pred.npz")
    p.add_argument("--ft_suffix", type=str, default="_ft_pred.npz")
    p.add_argument("--outdir", type=str, default=f"{base}/c2st")
    p.add_argument("--models", nargs="+", default=FORGETTING_FAMILIES,
                   choices=FORGETTING_FAMILIES)
    p.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    p.add_argument("--max_num_categories", type=int, default=4,
                   help="Row layout of the prediction grid. Must match the checkpoint's "
                        "training value (r4c4 => 4). Only affects row labels here, since "
                        "params_mask is read from the npz rather than rebuilt.")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--max_iter", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel processes over active cells.")
    p.add_argument("--joint_hidden", nargs="+", type=int, default=None,
                   help="Hidden layer sizes for the joint C2ST. Default 'auto' scales as "
                        "2**ceil(log2(10*dims)), which for ~90 dims is (1024, 1024) and is "
                        "slow; e.g. --joint_hidden 128 128 is much cheaper.")
    p.add_argument("--skip_joint", action="store_true",
                   help="Per-cell C2ST only; skip the expensive joint fit.")
    p.add_argument("--strict", action="store_true",
                   help="Error out (instead of warning) if true_set differs between arms.")
    return p.parse_args()


def main():
    args = parse_args()
    orig_dir, ft_dir = Path(args.orig_dir), Path(args.ft_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summaries, skipped = [], []
    for family, scenario in itertools.product(args.models, args.scenarios):
        cfg = MODELS[family]
        if scenario not in cfg["benchmark_configs"]:
            skipped.append((family, scenario, "not in benchmark_configs"))
            continue
        orig_path = orig_dir / f"{family}_{scenario}{args.orig_suffix}"
        ft_path = ft_dir / f"{family}_{scenario}{args.ft_suffix}"
        if not orig_path.exists():
            skipped.append((family, scenario, f"missing {orig_path}"))
            continue
        if not ft_path.exists():
            skipped.append((family, scenario, f"missing {ft_path}"))
            continue

        summaries.append(run_pair(
            family, scenario, orig_path, ft_path, cfg, outdir,
            args.max_num_categories, args.n_splits, args.max_iter, args.seed,
            args.workers, args.joint_hidden, args.skip_joint, args.strict,
        ))

    for family, scenario, why in skipped:
        logging.warning(f"skipped {family}/{scenario}: {why}")

    if not summaries:
        # Never exit 0 having silently done nothing — that was the trap in
        # model_family_c2st.py, whose stale default paths made it a no-op.
        raise SystemExit(
            f"No pairs evaluated ({len(skipped)} skipped). Check --orig_dir/--ft_dir."
        )

    summary_df = pd.DataFrame(summaries)
    summary_path = outdir / "forgetting_c2st_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logging.info(f"[saved] {summary_path}")

    cols = ["family", "scenario", "mean_real_matched", "mean_null_floor",
            "mean_excess", "max_excess"]
    if "joint_excess" in summary_df:
        cols += ["joint_real_matched", "joint_null_floor", "joint_excess"]
    logging.info("\n" + summary_df[cols].to_string(index=False))
    logging.info(
        f"\nRead 'excess' (real minus noise floor), not raw C2ST. "
        f"Overall mean excess = {summary_df['mean_excess'].mean():.4f}"
    )
    if skipped:
        logging.warning(f"{len(skipped)} pair(s) skipped — see warnings above.")
    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
