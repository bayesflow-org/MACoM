"""Random-design ensemble recovery with the mask thumbnails in ONE row under the grid.

For single-family ensembles drawn with ``model_class_cf_ensemble --families DDM
--fix_params s_v,s_tau --no-add_interaction``: a 3-row (intercept + two main effects)
by 4-column (v, a, z, tau) grid, every design overlaid, and the thumbnail row spanning
exactly the four columns' plot areas (``ensemble_recovery(thumb_below=True)``).

    python -m experiments.experiment_1.replot_ensemble_below \\
        --datadir outputs/figures/model_class/cf/<MC_TAG>_ensemble_ddm6_fixsvstau
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cogformer.diagnostics.plot.ensemble_recovery import ensemble_recovery

from .model_class_cf_ensemble import (build_model_class, ensemble_colors,
                                      GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
from .replot_ensemble import DATA_NAME
from ..configs import ALL_PARAMS, ALL_VARIABLE_NAMES

ROW_LABELS = {3: ["Intercept", "Main\nEffect 1", "Main\nEffect 2"],
              4: ["Intercept", "Main\nEffect 1", "Main\nEffect 2", "Interaction"]}

# Drawn at 2x the 5.5 in \textwidth it is placed at, so every size prints at half.
PAGE_WIDTH_IN = 5.5
SPEC = dict(width=11.0, height=8.6, title=17, label=15, legend=13, tick=11, annot=12)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--datadir", type=str, required=True)
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--max_num_regressors", type=int, default=4)
    p.add_argument("--max_num_categories", type=int, default=4)
    p.add_argument("--thumb_gap_in", type=float, default=0.12)
    p.add_argument("--col_gap_px", type=float, default=6.0,
                   help="Blank gap between a panel's right edge and the next column's "
                        "tick labels, in pixels at 100 dpi (ensemble_recovery pack_cols). "
                        "The panels widen into whatever the gutters give up.")
    p.add_argument("--row_gap_px", type=float, default=10.0,
                   help="Blank gap between a row's x tick labels and the next row's "
                        "panels, in pixels at 100 dpi.")
    p.add_argument("--height", type=float, default=SPEC["height"],
                   help="Figure height in inches. Panels are square and height-limited, "
                        "so this sets their size; packing the columns then sets the width.")
    p.add_argument("--caption", choices=["design", "label"], default="design",
                   help="'design': Design 1..n (one family, so the family name is "
                        "redundant); 'label': the run's own labels (DDM | 1, ...).")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = parse_args()
    datadir = Path(a.datadir)
    outdir = Path(a.outdir) if a.outdir else datadir
    outdir.mkdir(parents=True, exist_ok=True)

    d = np.load(datadir / DATA_NAME, allow_pickle=True)
    n = int(d["n_configs"])
    families = sorted({str(x) for x in d["families"]})
    fix_params = set(json.loads(str(d["fix_params"]))) if "fix_params" in d else set()

    model_families, model_class = build_model_class()
    names = model_class.global_param_names
    # Saved in the global parameter space; keep this run's families' columns minus
    # the pinned ones -- the same slice the run itself plotted.
    keep_params = set().union(*(model_families[f].intrinsic_params for f in families)) \
        - fix_params
    keep = [i for i, q in enumerate(names) if q in keep_params]
    true_list = [d[f"true_{i}"][..., keep] for i in range(n)]
    pred_list = [d[f"pred_{i}"][..., keep] for i in range(n)]
    masks = [d[f"mask_{i}"][:, keep] for i in range(n)]
    kept = [names[i] for i in keep]
    label_map = dict(zip(ALL_PARAMS, ALL_VARIABLE_NAMES))
    label_map.update(dict(zip(GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)))

    n_rows = int(np.sum(np.any([np.any(m == 1.0, axis=1) for m in masks], axis=0)))
    labels = ([f"Design {i + 1}" for i in range(n)] if a.caption == "design"
              else [str(x) for x in d["labels"]])
    logging.info(f"{n} configs | {families} | fixed {sorted(fix_params)} | "
                 f"columns {kept} | {n_rows} rows")

    rc = {"xtick.labelsize": SPEC["tick"], "ytick.labelsize": SPEC["tick"]}
    with matplotlib.rc_context(rc):
        fig = ensemble_recovery(
            true_list=true_list, pred_list=pred_list,
            design_configs=json.loads(str(d["design_configs"])),
            intrinsic_params=kept, max_num_categories=a.max_num_categories,
            parameter_masks=masks, variable_names=[label_map[q] for q in kept],
            max_num_regressors=a.max_num_regressors,
            categorical_regressors_list=json.loads(str(d["categorical_regressors"])),
            global_params=GLOBAL_PARAMS,
            colors=ensemble_colors(n), labels=labels,
            row_labels=ROW_LABELS[n_rows], max_ticks=3,
            title_fontsize=SPEC["title"], label_fontsize=SPEC["label"],
            legend_fontsize=SPEC["legend"], annot_fontsize=SPEC["annot"],
            marker_size=14, err_linewidth=0.8, wspace=0.25, hspace=0.12,
            pack_cols=True, col_gap_px=a.col_gap_px,
            pack_rows=True, row_gap_px=a.row_gap_px, tick_pad=2.0,
            figsize=(SPEC["width"], a.height),
            thumb_below=True, thumb_gap_in=a.thumb_gap_in,
        )

    # The span the strip was built to match, checked on the drawn figure.
    fig.canvas.draw()
    grid_axes = fig.axes[:n_rows * len([q for q in kept if q not in GLOBAL_PARAMS])]
    thumbs = fig.axes[len(grid_axes):]
    gx = [ax.get_window_extent() for ax in grid_axes]
    tx = [ax.get_window_extent() for ax in thumbs]
    dl = (min(b.x0 for b in tx) - min(b.x0 for b in gx)) / fig.dpi * 72
    dr = (max(b.x1 for b in tx) - max(b.x1 for b in gx)) / fig.dpi * 72
    logging.info(f"strip vs grid edges: left {dl:+.2f} pt, right {dr:+.2f} pt")

    path = outdir / "model_class_ensemble_recovery_below.pdf"
    fig.savefig(path, bbox_inches="tight")
    saved_w = (fig.get_tightbbox(fig.canvas.get_renderer()).width
               + 2 * matplotlib.rcParams["savefig.pad_inches"])
    logging.info(f"[saved] {path}  ({saved_w:.2f} in wide -> x{PAGE_WIDTH_IN / saved_w:.3f}"
                 f" at {PAGE_WIDTH_IN} in)")
    plt.close(fig)


if __name__ == "__main__":
    main()
