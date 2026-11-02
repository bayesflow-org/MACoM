"""Isometric stack of the coefficient matrices behind parameter_mask_stack's masks.

Reads the draws parameter_mask_stack saved (``*_draws.npz``) and redraws the same
stack, sheet for sheet and in the same layout and colors: each active cell takes its
family's shade for the term kind, with saturation scaled by the coefficient's
magnitude |beta| (full color at ``--vmax`` and beyond, fading toward gray at zero;
sign is not encoded). Inactive and padding cells keep the mask figure's translucent
grays. Coefficients are on the linear-predictor scale, before the link function.

Usage (after parameter_mask_stack has run):
    python -m experiments.visualization.coefficient_matrix_stack --vmax 3
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.colors as mcolors

from .parameter_mask_stack import FIG_DIR, mask_faces, render_stack, stack_from_draws


def scale_saturation(rgb, magnitude, vmax, floor):
    """``rgb`` (..., 3) with HSV saturation scaled to ``floor`` at |beta| = 0 rising
    linearly to the full color at |beta| >= ``vmax``; hue and value are kept."""
    hsv = mcolors.rgb_to_hsv(rgb)
    t = np.clip(np.abs(magnitude) / vmax, 0.0, 1.0)
    hsv[..., 1] *= floor + (1.0 - floor) * t
    return mcolors.hsv_to_rgb(hsv)


def coefficient_faces(stack, kinds, vmax, floor, inactive_alpha, padding_alpha):
    """The mask figure's faces, with each active cell's saturation set by |beta|."""
    faces = mask_faces(stack, kinds, inactive_alpha, padding_alpha)
    active = stack["exists"] & (stack["masks"] > 0)
    faces[active, :3] = scale_saturation(faces[active, :3], stack["values"][active], vmax, floor)
    return faces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=Path, default=FIG_DIR / "parameter_mask_stack_draws.npz",
                        help="Draws saved by parameter_mask_stack.")
    parser.add_argument("--vmax", type=float, default=3.0,
                        help="|coefficient| at which a cell reaches its full color.")
    parser.add_argument("--saturation_floor", type=float, default=0.15,
                        help="Fraction of the full saturation kept at |coefficient| = 0, so a "
                             "near-zero active cell still shows its hue.")
    parser.add_argument("--out", type=str, default="coefficient_matrix_stack")
    args = parser.parse_args()

    draws = np.load(args.draws, allow_pickle=True)
    stack = stack_from_draws(draws)
    kinds = draws["row_kinds"]
    faces = coefficient_faces(stack, kinds, args.vmax, args.saturation_floor,
                              float(draws["inactive_alpha"]), float(draws["padding_alpha"]))

    fig, _ = render_stack(faces, kinds, float(draws["layer_gap"]), int(draws["rotation"]))
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{args.out}.{ext}", dpi=300, bbox_inches="tight")
    print(f"saved {FIG_DIR / args.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
