import argparse
import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from threadpoolctl import threadpool_limits

from cogformer.diagnostics.metric.adaptive_c2st import adaptive_c2st as compute_adaptive_c2st
from cogformer.diagnostics.metric.adaptive_c2st import compute_c2st_accuracy, compute_joint_c2st
from cogformer.utils.plot_utils import cogformer_fm_colors

from ..configs import MODELS as _BASE_MODELS
from ..utils import add_global_params, reshape_bf_to_cf


# Both arms of Class-vs-BF carry the contamination parameter: the BF baselines get
# it via _bf_cases_with_globals, and the model class via its own add_global_params.
# The bare configs list only the 6 local params, so reshape_bf_to_cf would map 6 of
# the 7 active BF params and trip its own assertion. Augment to match both arms.
GLOBAL_PARAMS = ["pi"]
GLOBAL_VARIABLE_NAMES = [r"$\pi$"]

_C2ST_MODELS = ("ddm", "cdm", "rdm", "lba")
_C2ST_OUTDIRS = {
    "ddm": "./outputs/figures/c2st/ddm/",
    "cdm": "./outputs/figures/c2st/cdm/",
    "rdm": "./outputs/figures/c2st/rdm/",
    "lba": "./outputs/figures/c2st/lba/",
}

MODELS = {}
for _key, _raw in {k: v for k, v in _BASE_MODELS.items() if k in _C2ST_MODELS}.items():
    _base = add_global_params(_raw, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
    _cfg = {
        "intrinsic_params": _base["intrinsic_params"],
        "variable_names": _base["variable_names"],
        "outdir": _C2ST_OUTDIRS[_key],
        # c2st uses a subset of benchmark cases (no "full")
        "benchmark_configs": {
            k: v for k, v in _base["benchmark_configs"].items() if k != "full"
        },
    }
    MODELS[_key] = _cfg


# The amortization-gap matrix: DDM/RDM/CDM get BF_vs_Original and Original_vs_Finetuned;
# LBA gets BF_vs_Finetuned, since it has no original-checkpoint arm (it was not one of
# the three model classes the original checkpoint was trained on). Each entry is
# (arm_a, arm_b, output-file tag).
SCENARIOS = {
    "BF_vs_Original":        ("original",  "bf",        "orig_vs_bf"),
    "BF_vs_Finetuned":       ("finetuned", "bf",        "ft_vs_bf"),
    "Original_vs_Finetuned": ("finetuned", "original",  "ft_vs_orig"),
}


def load_bf_data(data_path):
    dataset = np.load(data_path, allow_pickle=True)
    return {"param_masks": dataset["param_masks"], "pred_set": dataset["pred_set"]}


def load_cf_pred(pred_path):
    dataset = np.load(pred_path, allow_pickle=True)
    return dataset["pred_set"], dataset["params_mask"]


def _qda_c2st(x, y, n_splits=5, random_state=0):
    """1-D C2ST with a quadratic discriminant instead of the MLP.

    Per dataset there are only ~100 draws per side. At that size the MLP (early stopping
    on 1/5 of the training fold) barely learns: on ddm/interaction it scored 0.528 where
    the mean shift alone implies a Bayes accuracy of 0.645. QDA -- the optimal rule for
    two Gaussians, which these unimodal 1-D posteriors approximate -- scored 0.667
    (null 0.498). Standardised first: QDA's rank check is absolute, and pi's posterior
    variance (~1e-6) would trip it.
    """
    mu, sd = x.mean(), x.std() + 1e-12
    X = ((np.concatenate([x, y]) - mu) / sd)[:, None]
    t = np.concatenate([np.zeros(len(x)), np.ones(len(y))])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return float(cross_val_score(QuadraticDiscriminantAnalysis(), X, t, cv=cv).mean())


def _dataset_task(args):
    """One dataset: joint and per-cell C2ST, each next to its within-arm null.

    The pooled arms (compute_adaptive_c2st, compute_joint_c2st) mix draws from every
    dataset. Pooled over 200 datasets, a single parameter's draws trace the aggregate
    posterior -- roughly the prior for any calibrated approximator -- so the pooled
    per-cell C2ST sits near 0.52 even when each dataset's two posteriors sit ~0.5-0.8
    posterior SDs apart. The pooled joint recovers that difference only once the
    datasets separate in parameter space (~9+ dims), so it also undershoots on the
    small cases. This arm conditions on the dataset, which is the comparison the gap
    is about.

    Same recipe as experiment_3/model_class_paired_c2st: real = first halves of both
    arms' draws, null = each arm's first half vs its own second half (same n, same
    dataset), floor = the larger null. Read real next to its floor, not against 0.5:
    with ~100 draws per side the joint MLP's null drifts off 0.5. The joint uses the
    MLP as everywhere else; the per-cell test uses QDA (see _qda_c2st).
    """
    i, a, b, half = args                                   # a, b: (draws, num_active)
    with threadpool_limits(1):                             # one core per worker
        def real_and_floor(x, y, c2st):
            real = c2st(x[:half], y[:half])
            floor = max(c2st(x[:half], x[half:2 * half]), c2st(y[:half], y[half:2 * half]))
            return real, floor
        joint = real_and_floor(a, b, compute_c2st_accuracy)
        cells = [real_and_floor(a[:, d], b[:, d], _qda_c2st) for d in range(a.shape[1])]
    return i, joint, np.asarray(cells)


def per_dataset_c2st(pred_a, pred_b, params_mask, workers):
    """(per-dataset DataFrame, (num_active, 2) per-cell [real, floor] means over datasets)."""
    m = (params_mask[0] if params_mask.ndim == 3 else params_mask) == 1.0
    a, b = pred_a[:, :, m], pred_b[:, :, m]                # (N, draws, num_active)
    n_ds, draws, n_act = a.shape
    tasks = [(i, a[i], b[i], draws // 2) for i in range(n_ds)]
    if workers > 1:
        # spawn, not fork: the parent has imported jax/keras via cogformer.
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
            results = list(ex.map(_dataset_task, tasks, chunksize=4))
    else:
        results = [_dataset_task(t) for t in tasks]
    results.sort(key=lambda r: r[0])
    cells = np.stack([r[2] for r in results])              # (N, num_active, 2)
    df = pd.DataFrame({
        "dataset": [r[0] for r in results],
        "joint_c2st": [r[1][0] for r in results],
        "joint_null_floor": [r[1][1] for r in results],
        "mean_cell_c2st": cells[:, :, 0].mean(1),
        "mean_cell_null_floor": cells[:, :, 1].mean(1),
    })
    return df, cells.mean(0)


def run_comparison(pred_a, pred_b, design_config, intrinsic_params, variable_names,
                   colors, max_num_categories, params_mask, outdir, file_prefix,
                   per_dataset=True, workers=1):
    joint_score = compute_joint_c2st(
        pred_a=pred_a, pred_b=pred_b, design_config=design_config,
        intrinsic_params=intrinsic_params, max_num_categories=max_num_categories,
        parameter_mask=params_mask,
    )
    c2st_df = compute_adaptive_c2st(
        pred_a=pred_a, pred_b=pred_b, design_config=design_config,
        intrinsic_params=intrinsic_params, max_num_categories=max_num_categories,
        parameter_mask=params_mask, variable_names=variable_names,
    )
    mean_accuracy = float(c2st_df["C2ST Accuracy"].mean())
    scores = {"mean_accuracy": mean_accuracy, "joint_c2st": joint_score}

    if per_dataset:
        ds_df, cell_means = per_dataset_c2st(pred_a, pred_b, params_mask, workers)
        ds_df.to_csv(outdir / f"{file_prefix}_per_dataset_c2st.csv", index=False)
        # Cells come out of the boolean mask in row-major order, the same order
        # compute_adaptive_c2st walks them, so they line up with c2st_df's rows.
        c2st_df["Per-dataset C2ST"] = cell_means[:, 0]
        c2st_df["Per-dataset null floor"] = cell_means[:, 1]
        scores.update({
            "perds_mean_accuracy": float(ds_df["mean_cell_c2st"].mean()),
            "perds_mean_null_floor": float(ds_df["mean_cell_null_floor"].mean()),
            "perds_joint_c2st": float(ds_df["joint_c2st"].mean()),
            "perds_joint_null_floor": float(ds_df["joint_null_floor"].mean()),
        })
        c2st_df.loc["_mean_accuracy_"] = [mean_accuracy, scores["perds_mean_accuracy"],
                                          scores["perds_mean_null_floor"]]
        c2st_df.loc["_joint_c2st_"] = [joint_score, scores["perds_joint_c2st"],
                                       scores["perds_joint_null_floor"]]
    else:
        c2st_df.loc["_mean_accuracy_"] = {"C2ST Accuracy": mean_accuracy}
        c2st_df.loc["_joint_c2st_"] = {"C2ST Accuracy": joint_score}
    c2st_df.to_csv(outdir / f"{file_prefix}_c2st.csv")
    logging.info(f"[saved] {outdir / file_prefix}_c2st.csv  "
                 + "  ".join(f"{k}={v:.3f}" for k, v in scores.items()))
    return scores


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    p.add_argument("--bf_data_dir", type=str, default="./outputs/data/model_family",
                   help="Directory with BF validation .npz files (model_family_bf_validate).")
    p.add_argument("--mc_pred_dir", type=str, default=None,
                   help="Predictions from the ORIGINAL model-class checkpoint.")
    p.add_argument("--ft_pred_dir", type=str, default=None,
                   help="Predictions from the LBA-FINETUNED model-class checkpoint.")
    p.add_argument("--mc_pred_suffix", type=str, default="_cf_pred.npz",
                   help="Filename suffix in --mc_pred_dir (model_class_cf_validate).")
    p.add_argument("--ft_pred_suffix", type=str, default="_ft_pred.npz",
                   help="Filename suffix in --ft_pred_dir (model_class_lba_finetune_validate).")
    p.add_argument("--scenarios", nargs="+", default=None, choices=list(SCENARIOS),
                   help="Which comparisons to run. Default: every one whose inputs are "
                        "available from the directories given.")
    p.add_argument("--outdir", type=str, default=None,
                   help="Output directory (defaults to model-specific path)")
    p.add_argument("--max_num_categories", type=int, default=2)
    p.add_argument("--skip_per_dataset", action="store_true",
                   help="Pooled arms only (skips the per-dataset arm; see _dataset_task).")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                   help="Processes for the per-dataset arm.")
    return p.parse_args()


def main():
    args = parse_args()
    model_cfg = MODELS[args.model]
    model_name = args.model
    intrinsic_params = model_cfg["intrinsic_params"]
    variable_names = model_cfg["variable_names"]
    colors = cogformer_fm_colors()
    benchmark = model_cfg["benchmark_configs"]
    outdir_root = Path(args.outdir) if args.outdir else Path(model_cfg["outdir"])
    summary_rows = []

    have = {"bf": args.bf_data_dir is not None,
            "original": args.mc_pred_dir is not None,
            "finetuned": args.ft_pred_dir is not None}
    wanted = args.scenarios or [
        name for name, (a, b, _) in SCENARIOS.items() if have[a] and have[b]
    ]
    for name in wanted:
        a, b, _ = SCENARIOS[name]
        for side in (a, b):
            if not have[side]:
                raise SystemExit(
                    f"scenario '{name}' needs {side} predictions; pass the matching "
                    f"--{'bf_data_dir' if side == 'bf' else 'mc_pred_dir' if side == 'original' else 'ft_pred_dir'}"
                )
    if not wanted:
        raise SystemExit("no scenario has both its inputs; nothing to compare.")
    logging.info(f"[{model_name}] scenarios: {', '.join(wanted)}")

    for cfg_name, design_config in benchmark.items():
        # Load whichever arms exist for this case. Every scenario scores the same
        # simulated datasets, so the MC/finetuned predictions must come from runs
        # paired to this BF file (model_class_cf_validate --data_dir).
        sources, missing = {}, []
        bf_path = Path(args.bf_data_dir) / f"{model_name}_{cfg_name}_data.npz"
        params_mask = None

        if bf_path.exists():
            bf_data = load_bf_data(bf_path)
            active_idx = bf_data["param_masks"][0].astype(bool)
            sources["bf"] = reshape_bf_to_cf(
                bf_data["pred_set"][:, :, active_idx], design_config, intrinsic_params
            )
        else:
            missing.append("bf")

        # Each producer keeps its own naming: model_class_cf_validate writes
        # {model}_{case}_cf_pred.npz, model_class_lba_finetune_validate writes
        # {model}_{case}_ft_pred.npz. Read both natively rather than renaming files.
        for side, d, suffix in (("original", args.mc_pred_dir, args.mc_pred_suffix),
                                ("finetuned", args.ft_pred_dir, args.ft_pred_suffix)):
            if d is None:
                continue
            path = Path(d) / f"{model_name}_{cfg_name}{suffix}"
            if path.exists():
                sources[side], mask = load_cf_pred(path)
                if params_mask is None:
                    params_mask = mask
            else:
                missing.append(side)

        runnable = [n for n in wanted
                    if SCENARIOS[n][0] in sources and SCENARIOS[n][1] in sources]
        if not runnable:
            logging.warning(f"[{cfg_name}] no scenario runnable (missing: {', '.join(missing) or 'n/a'}), skipping.")
            continue
        if params_mask is None:
            logging.warning(f"[{cfg_name}] no CogFormer predictions, so no parameter mask; skipping.")
            continue

        # compute_joint_c2st indexes BOTH arms with the same (num_rows, num_cols) mask,
        # so a grid mismatch would silently misalign parameters rather than raise.
        grid = tuple(params_mask.shape[-2:])
        bad = {k: tuple(v.shape[-2:]) for k, v in sources.items() if tuple(v.shape[-2:]) != grid}
        if bad:
            logging.error(
                f"[{cfg_name}] grid mismatch against mask {grid}: {bad}. Regenerate the "
                f"predictions paired to this BF file (model_class_cf_validate --data_dir "
                f"{args.bf_data_dir}). Skipping."
            )
            continue

        outdir = outdir_root / cfg_name
        outdir.mkdir(parents=True, exist_ok=True)

        for name in runnable:
            a, b, tag = SCENARIOS[name]
            scores = run_comparison(
                sources[a], sources[b], design_config, intrinsic_params,
                variable_names, colors, args.max_num_categories, params_mask,
                outdir, f"{model_name}_{cfg_name}_{tag}",
                per_dataset=not args.skip_per_dataset, workers=args.workers,
            )
            summary_rows.append({
                "model": model_name.upper(), "case": cfg_name, "scenario": name,
                **scores,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = outdir_root / f"{model_name}_c2st_summary.csv"
    outdir_root.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    logging.info(f"[saved] {summary_path}  ({len(summary_rows)} rows)")
    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
