"""Cell-by-cell CHANGE in recovery and calibration between two arms.

Same coefficient-matrix layout as :func:`adaptive_metrics` -- rows are design rows,
columns are intrinsic parameters, every active cell carries its number -- but each
cell holds a DIFFERENCE, arm B minus arm A, rather than one arm's value.

Two things follow from plotting a difference rather than a level, and both are
deliberate:

DIVERGING, CENTRED ON ZERO. The sequential palettes in adaptive_metrics encode
magnitude, which is right for a quantity with a floor at zero and wrong for a signed
change: on a sequential ramp "no change" lands wherever the data happen to put it.
The limits here are symmetric about zero by construction, so the midpoint of the
colour bar IS no change, and the reader can see sign before reading a digit.

ONE PALETTE FOR BOTH PANELS. Both metrics here are lower-is-better, so a single
diverging ramp is unambiguous: negative (the arm improved) is blue/purple, positive
(it got worse) is magenta/red, and the same colour means the same thing wherever it
appears. This only works because both panels run the same way -- an earlier version
plotted the change in r, which improves UPWARD, and had to reverse one panel's ramp
to keep "red = better" true on both. Swapping r for NRMSE removes that special case.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from cogformer.simulator.design_utils import (build_parameter_mask,
                                              expand_design_config_to_mask,
                                              build_row_labels)
from cogformer.diagnostics.metric.adaptive_metrics import _calibration_error, _rmse
from .adaptive_metrics import _compact_grids, _draw_grids

# Diverging, built from the project's own identity ramps rather than a seaborn
# default, so the hues mean what they mean elsewhere: the purple/blue ramp on the
# negative side, the magenta/red one on the positive. Saturated at both extremes and
# near-white at zero, so the midpoint of the bar reads as "no change".
DELTA_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "cogformer_delta",
    [(0.00, "#4E2A84"),    # deep purple  -- most improved
     (0.28, "#6969FF"),    # blue
     (0.50, "#F7F7F7"),    # no change
     (0.72, "#FF6969"),    # red
     (1.00, "#EC008C")],   # magenta      -- most degraded
)


def _kind_row_labels(design_config, max_num_categories, parameter_mask,
                     categorical_regressors=None):
    """Spelled-out row names: Intercept / Main effect N / Interaction N.

    The default labels are the design's own coefficient keys ($u_1:c_1$), which say
    which regressor a row is but not what KIND of term it is -- and the kind is the
    thing a reader compares across rows. Numbered per kind, in row order.
    """
    _, kinds = build_row_labels(design_config, max_num_categories, return_kinds=True,
                                categorical_regressors=categorical_regressors,
                                parameter_mask=parameter_mask)
    return ["Intercept" if k == "intercept"
            else ("Main effect" if k == "main" else "Interaction") for k in kinds]


def _renumber(row_labels):
    """Number each kind sequentially over the rows that SURVIVED compacting.

    Numbering before the filter counts inactive rows too, so a grid showing three
    main-effect rows came out "Main effect 1 / 4 / 13" -- the row's index in the full
    design rather than its position in the figure. A kind with only one row keeps its
    bare name; a number there implies a second one is somewhere off-figure.
    """
    totals = {}
    for lab in row_labels:
        totals[lab] = totals.get(lab, 0) + 1
    seen, out = {}, []
    for lab in row_labels:
        if totals[lab] == 1:
            out.append(lab)
            continue
        seen[lab] = seen.get(lab, 0) + 1
        out.append(f"{lab} {seen[lab]}")
    return out


def adaptive_delta(
    true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    design_config: dict,
    intrinsic_params: list[str],
    max_num_categories: int,
    parameter_mask: np.ndarray = None,
    variable_names: list[str] = None,
    arm_names: tuple = ("zero-shot", "finetuned"),
    normalize: str | None = "range",
    kind_row_labels: bool = True,
    fit_row_labels: bool = True,
    aggregation=np.median,
    resolution: int = 20,
    min_quantile: float = 0.005,
    max_quantile: float = 0.995,
    intercept_color: str = "#4e2a84",
    main_effect_color: str = "#6969ff",
    interaction_color: str = "#ff6969",
    figsize: tuple | None = None,
    title_fontsize: int = 16,
    label_fontsize: int = 12,
    tick_fontsize: int = 10,
    annot_fontsize: int = 10,
    fmt: str = "+.3f",
    global_params: list[str] = None,
    global_only: bool = False,
    max_num_regressors: int = None,
    categorical_regressors: list[str] = None,
    symmetric: bool = True,
    cell_size: float = 0.75,
    orientation: str = "column",
) -> plt.Figure:
    """Delta heatmaps for ``pred_b`` minus ``pred_a`` on shared ``true``.

    Both arms must have been evaluated on the SAME datasets -- one ``true`` is taken,
    not one per arm -- because a cell here subtracts two numbers and is meaningless
    if they describe different data. Callers are expected to have checked that.

    ``symmetric`` puts the colour limits at +/- max|delta| per panel. Turning it off
    uses each panel's own min/max, which exaggerates a panel where nothing moved.

    ``orientation`` defaults to "column" here: two panels of the same grid read as a
    before/after pair stacked, and side by side they push the figure past a text
    column's width for no gain.

    ``cell_size`` is half the adaptive_metrics default. This plot carries two panels
    where that one carries three or four, so at 1.5 in a cell the same grid comes out
    a third wider than the figures it sits beside, and each cell holds one short
    signed number rather than needing room to be read as a level.

    Returns ``None`` if no column survives compacting, matching adaptive_metrics.
    """
    if parameter_mask is None:
        parameter_mask = build_parameter_mask(
            design_config=design_config, max_num_categories=max_num_categories,
            intrinsic_params=intrinsic_params, keep_intercept=True)
    if parameter_mask.ndim == 3:
        parameter_mask = parameter_mask[0]

    design_config = expand_design_config_to_mask(
        design_config, parameter_mask, max_num_categories,
        max_num_regressors=max_num_regressors)
    col_labels = variable_names if variable_names is not None else intrinsic_params
    regressor_keys = list(design_config.keys())
    block_width = max_num_categories - 1
    main_keys = [k for k in regressor_keys if k != "1" and ":" not in k]
    inter_keys = [k for k in regressor_keys if ":" in k]

    _, _, num_rows, num_cols = pred_a.shape
    n_design_rows = ((1 if "1" in regressor_keys else 0)
                     + len(main_keys) * block_width
                     + len(inter_keys) * block_width ** 2)
    num_rows = min(num_rows, n_design_rows)
    parameter_mask = parameter_mask[:num_rows]
    true = true[:, :num_rows, :]
    pred_a = pred_a[:, :, :num_rows, :]
    pred_b = pred_b[:, :, :num_rows, :]

    alphas = np.linspace(min_quantile, max_quantile, resolution)
    regions = 1 - alphas
    lowers = regions / 2
    uppers = 1 - lowers

    lo, hi = arm_names
    rmse_name = "RMSE" if normalize is None else "NRMSE"
    r_key = f"$\\Delta$ {rmse_name}   ({hi} $-$ {lo})"
    c_key = f"$\\Delta$ calibration error   ({hi} $-$ {lo})"
    grids = {r_key: np.full((num_rows, num_cols), np.nan),
             c_key: np.full((num_rows, num_cols), np.nan)}

    for r in range(num_rows):
        for c in range(num_cols):
            if parameter_mask[r, c] != 1.0:
                continue
            t = true[:, r, c]
            a, b = pred_a[:, :, r, c], pred_b[:, :, r, c]
            grids[r_key][r, c] = (_rmse(b, t, normalize, aggregation)
                                  - _rmse(a, t, normalize, aggregation))
            grids[c_key][r, c] = (
                _calibration_error(b, t, alphas, lowers, uppers, aggregation)
                - _calibration_error(a, t, alphas, lowers, uppers, aggregation))

    compacted = _compact_grids(
        grids, parameter_mask, design_config, num_rows, max_num_categories,
        intrinsic_params, col_labels, global_params, global_only,
        categorical_regressors,
        (intercept_color, main_effect_color, interaction_color),
        row_label_fn=(_kind_row_labels
                      if (kind_row_labels and not global_only) else None))
    if compacted is None:
        return None
    grids, parameter_mask, row_labels, row_colors, col_labels = compacted
    if kind_row_labels and not global_only:
        row_labels = _renumber(row_labels)

    specs = []
    for key in (r_key, c_key):
        g = grids[key]
        m = float(np.nanmax(np.abs(g))) if symmetric else None
        specs.append((key, DELTA_CMAP, -m if m else None, m if m else None))

    return _draw_grids(grids, specs, parameter_mask, row_labels, row_colors,
                       col_labels, figsize=figsize, title_fontsize=title_fontsize,
                       label_fontsize=label_fontsize, tick_fontsize=tick_fontsize,
                       annot_fontsize=annot_fontsize, fmt=fmt,
                       cell_size=cell_size, fit_row_labels=fit_row_labels,
                       orientation=orientation)
