"""Redraw the random-design ensemble figures from the saved npz — no GPU, no sampling.

``model_class_cf_ensemble.py`` samples 200 posterior draws for each of 12 random
design configs; changing how the figures LOOK must not require paying for that
again. Worse than the cost: the sampler would draw a DIFFERENT posterior, so a
rerun to restyle a figure would quietly change what the figure shows.

Everything the plots draw is already in ``model_class_ensemble_data.npz`` -- the
run persists it before plotting for exactly this reason. The plotting functions and
the parameter-space construction are imported rather than copied, so a replot and a
real run cannot drift apart.

    python -m experiments.experiment_1.replot_ensemble \\
        --datadir outputs/figures/model_class/cf/r4c4_b128_e15000_ensemble
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cogformer.diagnostics.plot.ensemble_recovery import (ensemble_recovery,
                                                          global_ensemble_recovery)
from cogformer.diagnostics.plot.ensemble_coverage import (ensemble_coverage,
                                                          global_ensemble_coverage)
from cogformer.diagnostics.plot.ensemble_ecdf import ensemble_ecdf, global_ensemble_ecdf
from cogformer.diagnostics.plot.ensemble_metrics import (ensemble_metrics,
                                                         global_ensemble_metrics)

from .model_class_cf_ensemble import (build_model_class, ensemble_colors,
                                      GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
from ..configs import (ALL_PARAMS, ALL_VARIABLE_NAMES as ALL_PARAMS_VARIABLE_NAMES)

DATA_NAME = "model_class_ensemble_data.npz"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--datadir", type=str, required=True,
                   help=f"Directory holding {DATA_NAME}.")
    p.add_argument("--outdir", type=str, default=None,
                   help="Where to write the PDFs (default: --datadir, i.e. in place).")
    p.add_argument("--plots", type=str, default="recovery,coverage,ecdf",
                   help="Comma-separated subset of recovery,coverage,ecdf,metrics. "
                        "Restyling one figure should not redraw the other five.")
    p.add_argument("--max_num_regressors", type=int, default=4)
    p.add_argument("--max_num_categories", type=int, default=4)
    # SIZES ARE FOR THE PAGE. The grid figure is read at a full 5.5 in \textwidth and
    # the per-config pi grid at half that, so they are drawn at 2x their target and
    # the page scales each by 0.5. The plotting functions' own defaults size per
    # panel (3 in each), which is how the grid ended up 24 in wide -- scaled to
    # \textwidth that is 0.23, and a 12 pt label lands at under 3 pt.
    p.add_argument("--width", type=float, default=None,
                   help="Figure width in inches; default is per-plot (see PAGE_SPEC).")
    p.add_argument("--title_fs", type=float, default=None)
    p.add_argument("--label_fs", type=float, default=None)
    p.add_argument("--legend_fs", type=float, default=None)
    p.add_argument("--tick_fs", type=float, default=None)
    return p.parse_args()


# width_in, title, label, legend, tick, annot -- drawn at 2x the intended page width.
#
# The pi grid stays 2 rows of 6, which is the shape the figure is meant to have. That
# does mean each panel is narrow once the page scales it down, so its text is sized to
# the panel rather than to a legibility target -- the panels are the point.
PAGE_SPEC = {
    # label/legend here are only STARTING sizes for recovery: fit_labels rescales
    # each one to its cell, so what matters is the cell, not this number.
    # height_ratio: rows are square and WIDTH-limited here, so the leftover cell
    # height is what shows up as the band between rows. Only the figure height
    # closes it, and the value is tuned against the SAVED figure -- the in-figure
    # geometry before the thumbnail strip is added is not what gets written out.
    "recovery":        dict(width=10.1, title=15, label=12, legend=10, tick=6,  annot=7,
                            height_ratio=0.46),
    "recovery_global": dict(width=11.0, title=12, label=11, legend=9,  tick=8,  annot=9),
}
DEFAULT_SPEC = dict(width=None, title=None, label=None, legend=None, tick=None,
                    annot=None, height_ratio=None)
GLOBAL_PER_ROW = 6

# The design-cell names ("1", "u_1(:c_1)") are what the mask is indexed by, and are
# unreadable as row labels on a figure. Same four rows, named for what they are.
# Wrapped: rotated, a row label's length is measured against the panel's HEIGHT, and
# "Main Effect 1" on one line is longer than a page-sized panel is tall, so the four
# labels overprint each other. The words are unchanged.
ROW_LABELS = ["Intercept", "Main Effect 1", "Main Effect 2", "Interaction"]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = parse_args()
    datadir = Path(a.datadir)
    outdir = Path(a.outdir) if a.outdir else datadir
    outdir.mkdir(parents=True, exist_ok=True)

    d = np.load(datadir / DATA_NAME, allow_pickle=True)
    n = int(d["n_configs"])
    true_list = [d[f"true_{i}"] for i in range(n)]
    pred_list = [d[f"pred_{i}"] for i in range(n)]
    masks = [d[f"mask_{i}"] for i in range(n)]
    labels = [str(x) for x in d["labels"]]
    design_configs = json.loads(str(d["design_configs"]))
    categorical_regressors = json.loads(str(d["categorical_regressors"]))
    families = [str(x) for x in d["families"]]
    logging.info(f"{n} configs | families {sorted(set(families))} | "
                 f"{true_list[0].shape[-1]} parameter columns")

    _, model_class = build_model_class()
    names = model_class.global_param_names
    label_map = dict(zip(ALL_PARAMS, ALL_PARAMS_VARIABLE_NAMES))
    label_map.update(dict(zip(GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)))
    if len(names) != true_list[0].shape[-1]:
        raise SystemExit(
            f"saved arrays have {true_list[0].shape[-1]} parameter columns but the "
            f"ModelClass defines {len(names)} ({names}). The npz predates the current "
            "parameter space; re-run the ensemble rather than mislabelling it.")

    shared = dict(
        true_list=true_list, pred_list=pred_list, design_configs=design_configs,
        intrinsic_params=names, max_num_categories=a.max_num_categories,
        parameter_masks=masks, variable_names=[label_map[p] for p in names],
        max_num_regressors=a.max_num_regressors,
        categorical_regressors_list=categorical_regressors,
        global_params=GLOBAL_PARAMS,
    )
    color_kwargs = {"colors": ensemble_colors(n), "labels": labels}
    thumb_kw = {"thumb_per_row": 2 if len(set(families)) > 1 else max(n, 1)}

    registry = {
        "recovery": (ensemble_recovery, global_ensemble_recovery, color_kwargs, thumb_kw),
        "coverage": (ensemble_coverage, global_ensemble_coverage, color_kwargs, thumb_kw),
        "ecdf":     (ensemble_ecdf, global_ensemble_ecdf, color_kwargs, thumb_kw),
        "metrics":  (ensemble_metrics, global_ensemble_metrics, {}, {}),
    }
    wanted = [w.strip() for w in a.plots.split(",") if w.strip()]
    unknown = [w for w in wanted if w not in registry]
    if unknown:
        raise SystemExit(f"unknown plot(s) {unknown}; available: {list(registry)}")

    def spec_for(key):
        s = dict(PAGE_SPEC.get(key, DEFAULT_SPEC))
        for cli, field in (("width", "width"), ("title_fs", "title"),
                           ("label_fs", "label"), ("legend_fs", "legend"),
                           ("tick_fs", "tick")):
            v = getattr(a, cli)
            if v is not None:
                s[field] = v
        return s

    def sized(fig_fn, key, n_rows, n_cols, **kw):
        """Draw one figure at its page size. Ticks go through rcParams because the
        plotting functions expose title/label sizes but not tick size."""
        s = spec_for(key)
        if s["width"] is not None:
            ratio = s.get("height_ratio") or (n_rows / n_cols)
            kw["figsize"] = (s["width"], s["width"] * ratio)
        for field, name in (("title", "title_fontsize"), ("label", "label_fontsize"),
                            ("legend", "legend_fontsize"), ("annot", "annot_fontsize")):
            if s[field] is not None and name in fig_fn.__code__.co_varnames:
                kw[name] = s[field]
        rc = {"xtick.labelsize": s["tick"], "ytick.labelsize": s["tick"]} \
            if s["tick"] is not None else {}
        with matplotlib.rc_context(rc):
            return fig_fn(**kw)

    # The DRAWN grid, not the padded one: ensemble_recovery keeps only rows active in
    # at least one config and drops the global columns into their own figure. Sizing
    # off the padded 21-row, 9-column array gave a figure 78 inches tall.
    n_rows_grid = int(np.sum(np.any([np.any(m == 1.0, axis=1) for m in masks], axis=0)))
    n_cols_main = sum(1 for q in names if q not in GLOBAL_PARAMS)
    logging.info(f"drawn grid: {n_rows_grid} rows x {n_cols_main} columns")
    if len(ROW_LABELS) != n_rows_grid:
        raise SystemExit(f"ROW_LABELS has {len(ROW_LABELS)} entries but the grid draws "
                         f"{n_rows_grid} rows")

    for name in wanted:
        fn, global_fn, extra, main_only = registry[name]
        logging.info(f"drawing {name}")
        # Labels fitted to their cell and the gutters pulled in as far as they go:
        # the panels are what the figure is for, so nothing else gets to claim space.
        row_kw = {"row_labels": ROW_LABELS, "max_ticks": 3, "fit_labels": True,
                  "fit_delta": 1.0, "wspace": 0.20, "hspace": 0.0,
                  "tick_pad": 1.9, "marker_size": 9, "err_linewidth": 0.7,
                  "thumb_scale": 0.8, "thumb_height_in": 0.60, "thumb_spread": False, "thumb_tight_rows": True,
                  "pack_rows": True, "row_gap_px": 10.0,
                  "pack_cols": True, "col_gap_px": 10.0,
                  # The strip divider is drawn at 0.02..0.98 of the figure. The target
                  # is BELOW 0.02 on purpose: this aligns the grid's tight bbox, and a
                  # text bbox reserves descender space that "Ground Truth" does not
                  # use, so matching the bboxes leaves the visible ink ~8px high.
                  "align_bottom_frac": 0.0086} if name == "recovery" else {}
        fig = sized(fn, name, n_rows_grid, n_cols_main,
                    **shared, **extra, **main_only, **row_kw)
        if fig is not None:
            path = outdir / f"model_class_ensemble_{name}.pdf"
            fig.savefig(path, bbox_inches="tight"); plt.close(fig)
            logging.info(f"[saved] {path}")
        # The pi grid wraps 12 configs into 2 rows of 6 and every panel shows the
        # same parameter, so it takes one shared scale with ticks on the first column.
        gkw = {"share_limits": True, "max_per_row": GLOBAL_PER_ROW,
               "wspace": 0.06, "equal_gaps": True} if name == "recovery" else {}
        g_rows = int(np.ceil(n / GLOBAL_PER_ROW))
        gfig = sized(global_fn, f"{name}_global", g_rows, GLOBAL_PER_ROW,
                     **shared, **extra, **gkw)
        if gfig is not None:
            path = outdir / f"model_class_ensemble_{name}_global.pdf"
            gfig.savefig(path, bbox_inches="tight"); plt.close(gfig)
            logging.info(f"[saved] {path}")


if __name__ == "__main__":
    main()
