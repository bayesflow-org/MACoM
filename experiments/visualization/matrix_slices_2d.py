"""Flat 2D versions of the three stacks: one draw's parameter mask, coefficient
matrix and design matrix, each as its own figure.

Reads the draws parameter_mask_stack saved (``*_draws.npz``) and takes one sheet of
the stack (``--layer``, 0 = top; by default the densest sheet, the one with the most
active mask cells plus nonzero entries in the shown rows of X). Cells use exactly
the 3D figures' colors. Matrices are in their usual orientation: the mask and
coefficients are terms x parameters, the design matrix is observations x terms.
Dash-dot lines mark the same intercept | main effect | interaction boundaries as
the section cuts.

Usage (after parameter_mask_stack has run):
    python -m experiments.visualization.matrix_slices_2d
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from .coefficient_matrix_stack import coefficient_faces
from .design_matrix_stack import design_faces, design_stack
from .parameter_mask_stack import FAMILIES, FIG_DIR, mask_faces, stack_from_draws, term_zones


def draw_matrix(faces, row_cuts=(), col_cuts=(), cell_in=0.25, margin=0.6):
    """Figure with ``faces`` (rows, cols, 4) as a grid of square cells (row 0 at the
    top) with white lines between them, and dash-dot cut lines between rows
    ``row_cuts`` and columns ``col_cuts``, overshooting the grid by ``margin``."""
    num_rows, num_cols = faces.shape[:2]
    fig, ax = plt.subplots(figsize=(num_cols * cell_in + 0.4, num_rows * cell_in + 0.4))
    polys = [[(j, i), (j + 1, i), (j + 1, i + 1), (j, i + 1)]
             for i in range(num_rows) for j in range(num_cols)]
    # Opaque white lines on every cell: flat, nothing lies beneath that needs to show
    # through, so the grid stays continuous across inactive cells too.
    ax.add_collection(PolyCollection(polys, facecolors=faces.reshape(-1, 4), edgecolors="white",
                                     linewidths=0.5))

    cut = {"color": "#333333", "linewidth": 0.7, "linestyle": "-."}
    for r in row_cuts:
        ax.plot([-margin, num_cols + margin], [r, r], **cut)
    for c in col_cuts:
        ax.plot([c, c], [-margin, num_rows + margin], **cut)

    ax.set_xlim(-margin, num_cols + margin)
    ax.set_ylim(num_rows + margin, -margin)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=Path, default=FIG_DIR / "parameter_mask_stack_draws.npz",
                        help="Draws saved by parameter_mask_stack.")
    parser.add_argument("--layer", type=int, default=None,
                        help="Sheet of the stack to show (0 = top); default: the densest.")
    parser.add_argument("--family", choices=list(FAMILIES), default=None,
                        help="Pick the densest sheet of this model family only.")
    parser.add_argument("--num_obs_shown", type=int, default=10,
                        help="Observations (rows of X) shown, from the start of the draw.")
    parser.add_argument("--coef_vmax", type=float, default=3.0)
    parser.add_argument("--design_vmax", type=float, default=1.0)
    parser.add_argument("--saturation_floor", type=float, default=0.15)
    parser.add_argument("--suffix", type=str, default="",
                        help="Appended to each output name, e.g. _r2c2 for a separate set.")
    args = parser.parse_args()

    draws = np.load(args.draws, allow_pickle=True)
    kinds = draws["row_kinds"]
    inactive, padding = float(draws["inactive_alpha"]), float(draws["padding_alpha"])
    cuts = [x0 for _, x0, _ in term_zones(kinds)[1:]]
    stack = stack_from_draws(draws)
    design = design_stack(draws, args.num_obs_shown)
    density = stack["masks"].sum(axis=(1, 2)) + (design["values"] != 0).sum(axis=(1, 2))
    if args.family is not None:
        density = np.where(stack["families"] == args.family, density, -1)
    k = int(np.argmax(density)) if args.layer is None else args.layer

    figures = {
        "parameter_mask_2d": draw_matrix(
            mask_faces(stack, kinds, inactive, padding)[k], row_cuts=cuts),
        "coefficient_matrix_2d": draw_matrix(
            coefficient_faces(stack, kinds, args.coef_vmax, args.saturation_floor,
                              inactive, padding)[k], row_cuts=cuts),
        "design_matrix_2d": draw_matrix(
            design_faces(design, kinds, args.design_vmax,
                         args.saturation_floor, inactive, padding)[k].transpose(1, 0, 2),
            col_cuts=cuts),
    }
    for name, fig in figures.items():
        for ext in ("pdf", "png"):
            fig.savefig(FIG_DIR / f"{name}{args.suffix}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"saved {', '.join(n + args.suffix for n in figures)} "
          f"({stack['families'][k]} draw, layer {k}) to {FIG_DIR}")


if __name__ == "__main__":
    main()
