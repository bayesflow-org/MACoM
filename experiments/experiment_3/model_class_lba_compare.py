"""Compare zero-shot vs. fine-tuned LBA inference, and check forgetting on DDM/RDM/CDM.

Loads .npz prediction files from two pred_dirs and computes:
  1. Delta r heatmap (fine-tuned − zero-shot) for LBA recovery.
  2. Delta r heatmap (fine-tuned − base) for DDM, RDM, CDM (forgetting).
  3. Delta metrics table (fine-tuned − base) for all models as CSV, and per case a
     NRMSE / calibration-error delta heatmap on the LBA's coefficient grid.

Expected file naming conventions:
  Zero-shot LBA  : {zero_shot_lba_pred_dir}/lba_{cfg}_mid1_cf_pred.npz
  Fine-tuned LBA : {finetuned_pred_dir}/lba_{cfg}_ft_pred.npz
  Base MC        : {base_mc_pred_dir}/{model}_{cfg}_cf_pred.npz
  Fine-tuned MC  : {finetuned_pred_dir}/{model}_{cfg}_ft_pred.npz
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from cogformer.diagnostics.plot.adaptive_correlation import (
    compute_r_matrix, adaptive_correlation_delta, adaptive_metric_matrix, heatmap_axes,
)
from cogformer.diagnostics.metric.adaptive_metrics import adaptive_metrics as compute_adaptive_metrics
from cogformer.utils.plot_utils import plain_coef_labels

from ..configs import MODELS as _BASE_MODELS, LBA_CONFIG
from ..utils import add_global_params

GLOBAL_PARAMS = ["pi"]
GLOBAL_VARIABLE_NAMES = [r"$\pi$"]

_MODELS_WITH_PI = {
    m: add_global_params(cfg, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
    for m, cfg in _BASE_MODELS.items()
    if m != "lba"
}
_LBA_WITH_PI = add_global_params(LBA_CONFIG, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
LBA_LOCAL_PARAMS = _LBA_WITH_PI["intrinsic_params"]

MC_MODELS = {
    "DDM": _MODELS_WITH_PI["ddm"],
    "RDM": _MODELS_WITH_PI["rdm"],
    "CDM": _MODELS_WITH_PI["cdm"],
}


BENCH5 = ["intercept_only", "fixed", "regressed", "fixed_regressed", "interaction"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zero_shot_dir", type=str, required=True,
                   help="model_class_lba_cf_validate --outdir, INCLUDING the A<fixed_A> "
                        "level (holds <scen>/mid_rdm/ and predictions/).")
    p.add_argument("--finetuned_dir", type=str, required=True,
                   help="model_class_lba_finetune_validate --outdir (<fam>/<scen>/, predictions/).")
    p.add_argument("--base_fig_dir", type=str, required=True,
                   help="Original checkpoint's per-case figures (e1_validate --outdir).")
    p.add_argument("--base_pred_dir", type=str, required=True,
                   help="Original checkpoint's predictions (e1_validate --pred_dir).")
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--scenarios", nargs="+", default=BENCH5,
                   help="Cases to compare. Default bench5: the cases every arm loads from "
                        "the shared benchmark files, so each difference is exactly paired.")
    p.add_argument("--max_num_categories", type=int, default=2,
                   help="Row layout of the prediction grids. 2 for bench cases scored from "
                        "the shared files (compact grid: intercept, u_1, u_2, u_1:u_2).")
    return p.parse_args()


def _load_npz(path: Path):
    """Load pred_set, true_set, params_mask from an npz file."""
    data = np.load(path)
    return data["pred_set"], data["true_set"], data["params_mask"]


def _load_metrics_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        logging.warning(f"Metrics CSV not found: {path}")
        return None
    return pd.read_csv(path, index_col=0)


def _delta_metrics(df_a: pd.DataFrame, df_b: pd.DataFrame, label_a: str, label_b: str) -> pd.DataFrame:
    """Return a combined table with columns for a, b, and delta=b−a."""
    shared_idx = df_a.index.intersection(df_b.index)
    shared_cols = df_a.columns.intersection(df_b.columns)
    a = df_a.loc[shared_idx, shared_cols]
    b = df_b.loc[shared_idx, shared_cols]
    combined = pd.concat(
        [a.add_suffix(f" ({label_a})"), b.add_suffix(f" ({label_b})"),
         (b - a).add_suffix(f" (Δ {label_b}−{label_a})")],
        axis=1,
    )
    col_order = []
    for c in shared_cols:
        col_order += [f"{c} ({label_a})", f"{c} ({label_b})", f"{c} (Δ {label_b}−{label_a})"]
    return combined[[c for c in col_order if c in combined.columns]]


# Only the finetune target is plotted: the trained families' deltas are ~100x smaller
# and stay in delta_metrics.csv; their forgetting is shown by the delta-r maps and
# the paired C2ST.
DELTA_PLOT_MODEL = "LBA"
DELTA_PLOT_METRICS = ["NRMSE", "Calibration Error"]


def _delta_col(df: pd.DataFrame, metric: str) -> str | None:
    """The delta column for `metric`, whatever baseline it was computed against.

    LBA is differenced against ZERO-SHOT and the other three against BASE, so the
    suffixes differ per model ("(Δ fine-tuned−zero-shot)" vs "(Δ fine-tuned−base)").
    Match on the prefix instead of hardcoding either one -- which is also why the
    panels are not titled with a single baseline: one label would be wrong for three
    of the four columns.
    """
    hits = [c for c in df.columns if c.startswith(f"{metric} (Δ ")]
    # pd.concat unions the columns, so a model differenced against "base" still HAS the
    # "(Δ fine-tuned−zero-shot)" column -- filled with NaN. Taking hits[0] therefore
    # picked LBA's delta from the base-suffixed column and rendered an all-NaN panel.
    # Choose the one that actually carries values for these rows.
    for c in hits:
        if df[c].notna().any():
            return c
    return None


def plot_delta_metrics(combined: pd.DataFrame, layouts: dict, outdir: Path, condition: str,
                       max_num_categories: int, cell_size: float = 0.55,
                       title_fontsize: int = 14) -> Path | None:
    """2 x 1 column of the LBA's per-cell metric deltas (fine-tuned minus zero-shot):
    NRMSE on top, calibration error below.

    Each panel lays the deltas out on the LBA's coefficient grid (coefficients down,
    parameters across), the same layout as the delta-r heatmaps, so a cell sits where
    its coefficient sits in the design. ``layouts[(model, condition)]`` holds the
    (design_config, parameter_mask, model_config) the cell labels are rebuilt from.

    Negative is improvement for both metrics (both are lower-is-better), so the
    diverging map puts improvement in blue and degradation in red, around a neutral
    zero; each panel has its own symmetric scale.
    """
    rows = combined[(combined["condition"] == condition) & (combined["model"] == DELTA_PLOT_MODEL)]
    if rows.empty or (DELTA_PLOT_MODEL, condition) not in layouts:
        logging.warning(f"No {DELTA_PLOT_MODEL} delta rows for condition '{condition}'; "
                        f"skipping delta metric plot.")
        return None
    dc, mask, cfg = layouts[(DELTA_PLOT_MODEL, condition)]

    panels = {}
    for metric in DELTA_PLOT_METRICS:
        col = _delta_col(rows, metric)
        if col is None:
            continue
        *panel, unplaced = adaptive_metric_matrix(
            rows[col].dropna().to_dict(), mask, dc, cfg["intrinsic_params"], max_num_categories,
            variable_names=cfg["variable_names"], global_params=GLOBAL_PARAMS)
        if unplaced:
            raise SystemExit(f"{DELTA_PLOT_MODEL}/{condition} {metric}: no grid cell for "
                             f"{unplaced} -- metric labels and grid layout disagree")
        # Words instead of the design-cell symbols, as in experiment 1's figures. One
        # line each: horizontal tick labels have the width, and two-line labels crowd
        # a row that is only one cell tall.
        panel[2] = plain_coef_labels(panel[2], wrap=False)
        panels[metric] = panel
    if not panels:
        logging.warning(f"No {DELTA_PLOT_MODEL} metric deltas for '{condition}'; skipping.")
        return None

    n_rows, n_cols = next(iter(panels.values()))[0].shape
    fig, axes = plt.subplots(len(panels), 1, squeeze=False, layout="constrained",
                             figsize=(cell_size * n_cols + 1.6,
                                      len(panels) * (cell_size * max(n_rows, 2) + 0.9)))
    cmap = "RdBu_r"
    for ax, (metric, panel) in zip(axes[:, 0], panels.items()):
        vmax = np.nanmax(np.abs(panel[0])) if np.isfinite(panel[0]).any() else 1.0
        norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
        heatmap_axes(ax, *panel, cmap=cmap, norm=norm, annot_fontsize=9, fmt="+.3f")
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=13)
        ax.set_title(f"\u0394 {metric}", fontsize=title_fontsize)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)

    path = outdir / f"delta_metrics_{condition}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"[saved] {path}")
    return path


def _paired(true_a, true_b, what):
    """Refuse to difference two arms unless they scored the same datasets' parameters."""
    if true_a.shape != true_b.shape or not np.allclose(true_a, true_b, atol=1e-6):
        raise SystemExit(f"{what}: true_set differs between arms -- they did not score the "
                         f"same datasets, so a paired difference would be meaningless")


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    zs, ft = Path(args.zero_shot_dir), Path(args.finetuned_dir)
    base_fig, base_pred = Path(args.base_fig_dir), Path(args.base_pred_dir)
    max_cats = args.max_num_categories
    delta_rows = []
    layouts = {}  # (family, case) -> (design_config, parameter_mask, model_config)

    arms = {"LBA": dict(cfg=_LBA_WITH_PI, labels=("zero-shot", "fine-tuned"),
                        a_pred=lambda sc: zs / "predictions" / f"lba_{sc}_mid1_cf_pred.npz",
                        a_csv=lambda sc: zs / sc / "mid_rdm" / f"lba_{sc}_mid1_metrics.csv",
                        stem="delta_r")}
    for name, cfg in MC_MODELS.items():
        m = name.lower()
        arms[name] = dict(cfg=cfg, labels=("base", "fine-tuned"),
                          a_pred=lambda sc, m=m: base_pred / f"{m}_{sc}_cf_pred.npz",
                          a_csv=lambda sc, m=m: base_fig / m / sc / f"{m}_{sc}_fm_metrics.csv",
                          stem="forgetting_delta_r")

    for name, arm in arms.items():
        m, cfg = name.lower(), arm["cfg"]
        la, lb = arm["labels"]
        for sc in args.scenarios:
            a_path, b_path = arm["a_pred"](sc), ft / "predictions" / f"{m}_{sc}_ft_pred.npz"
            if not (a_path.exists() and b_path.exists()):
                logging.warning(f"{name}/{sc}: predictions missing ({a_path} / {b_path}); skipped")
                continue
            pred_a, true_a, mask_a = _load_npz(a_path)
            pred_b, true_b, mask_b = _load_npz(b_path)
            _paired(true_a, true_b, f"{name}/{sc}")
            dc = cfg["benchmark_configs"][sc]
            layouts[(name, sc)] = (dc, mask_a, cfg)
            r_a, pm = compute_r_matrix(true_a, pred_a, dc, cfg["intrinsic_params"], max_cats, mask_a[None])
            r_b, _ = compute_r_matrix(true_b, pred_b, dc, cfg["intrinsic_params"], max_cats, mask_b[None])
            fig = adaptive_correlation_delta(r_a, r_b, pm, dc, cfg["intrinsic_params"], max_cats,
                                             variable_names=cfg["variable_names"],
                                             label_a=la, label_b=lb)
            (outdir / m).mkdir(exist_ok=True)
            fig.savefig(outdir / m / f"{sc}_{arm['stem']}.pdf", bbox_inches="tight")
            plt.close(fig)

            df_a = _load_metrics_csv(arm["a_csv"](sc))
            df_b = _load_metrics_csv(ft / m / sc / f"{m}_{sc}_ft_metrics.csv")
            if df_a is not None and df_b is not None:
                d = _delta_metrics(df_a, df_b, la, lb)
                d.insert(0, "model", name); d.insert(1, "condition", sc)
                delta_rows.append(d)

    if not delta_rows:
        raise SystemExit("no paired metrics found; nothing to compare")
    combined = pd.concat(delta_rows, axis=0)
    combined.to_csv(outdir / "delta_metrics.csv")
    logging.info(f"[saved] {outdir / 'delta_metrics.csv'}")
    for sc in args.scenarios:
        plot_delta_metrics(combined, layouts, outdir, condition=sc, max_num_categories=max_cats)
    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
