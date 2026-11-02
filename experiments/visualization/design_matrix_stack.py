"""Isometric stack of the design matrices behind parameter_mask_stack's masks.

Reads the draws parameter_mask_stack saved (``*_draws.npz``) and stacks each draw's
design matrix X (observations x design columns) in the same order and layout. The
design columns run along the same term axis as the rows of the mask and coefficient
stacks (X is N x K, B is K x P), so the section cuts line up across all three; the
other axis shows the first ``--num_obs_shown`` observations. A nonzero entry takes
its family's shade for the column's term kind, with saturation scaled by |x| (full
color at ``--vmax`` and beyond); zero entries and unused design columns are the
inactive gray, and columns of terms absent from the design are padding.

Usage (after parameter_mask_stack has run):
    python -m experiments.visualization.design_matrix_stack --num_obs_shown 10
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.colors as mcolors

from .coefficient_matrix_stack import scale_saturation
from .parameter_mask_stack import (FAMILIES, FAMILY_COLORS, FIG_DIR, INACTIVE_GRAY, PADDING_GRAY,
                                   render_stack)


def design_stack(draws, num_obs_shown):
    """Design matrices in the saved stack order, transposed to (layers, design
    columns, observations) and cut to the first ``num_obs_shown`` observations.
    Also returns which design columns are used anywhere in each draw's full matrix,
    which terms exist, and each layer's family."""
    order = draws["stack_order"]
    shown, used, exists, families = [], [], [], []
    for name in FAMILIES:
        X = draws[f"{name}_design_matrices"]
        shown.append(X[:, :num_obs_shown].transpose(0, 2, 1))
        used.append((X != 0).any(axis=1))
        exists.append(draws[f"{name}_regressor_masks"].astype(bool))
        families += [name] * len(X)
    return {"values": np.concatenate(shown)[order], "used": np.concatenate(used)[order],
            "exists": np.concatenate(exists)[order], "families": np.array(families)[order]}


def design_faces(stack, kinds, vmax, floor, inactive_alpha, padding_alpha):
    """RGBA per cell: family shade by term kind, saturation by |x|, for nonzero
    entries; inactive gray for zeros and unused columns; padding for absent terms."""
    values = stack["values"]
    kind_key = {"intercept": "intercept", "main": "main_effect", "interaction": "interaction"}
    base = np.array([[mcolors.to_rgb(FAMILY_COLORS[family][kind_key[kind]]) for kind in kinds]
                     for family in stack["families"]])                     # (layers, columns, 3)
    base = np.broadcast_to(base[:, :, None, :], values.shape + (3,))

    faces = np.empty(values.shape + (4,))
    faces[:] = mcolors.to_rgba(INACTIVE_GRAY, inactive_alpha)
    active = (values != 0) & stack["used"][:, :, None] & stack["exists"][:, :, None]
    faces[active, :3] = scale_saturation(base[active], values[active], vmax, floor)
    faces[active, 3] = 1.0
    faces[~stack["exists"]] = mcolors.to_rgba(PADDING_GRAY, padding_alpha)
    return faces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=Path, default=FIG_DIR / "parameter_mask_stack_draws.npz",
                        help="Draws saved by parameter_mask_stack.")
    parser.add_argument("--num_obs_shown", type=int, default=10,
                        help="Observations (rows of X) shown per sheet, from the start of each draw.")
    parser.add_argument("--vmax", type=float, default=1.0,
                        help="|x| at which a cell reaches its full color; 1 puts every dummy at full color.")
    parser.add_argument("--saturation_floor", type=float, default=0.15,
                        help="Fraction of the full saturation kept as |x| approaches 0.")
    parser.add_argument("--out", type=str, default="design_matrix_stack")
    args = parser.parse_args()

    draws = np.load(args.draws, allow_pickle=True)
    kinds = draws["row_kinds"]
    faces = design_faces(design_stack(draws, args.num_obs_shown), kinds, args.vmax,
                         args.saturation_floor, float(draws["inactive_alpha"]),
                         float(draws["padding_alpha"]))

    fig, _ = render_stack(faces, kinds, float(draws["layer_gap"]), int(draws["rotation"]))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{args.out}.{ext}", dpi=300, bbox_inches="tight")
    print(f"saved {FIG_DIR / args.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
