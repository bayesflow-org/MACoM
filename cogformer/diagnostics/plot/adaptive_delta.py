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

ORIENTED SO THAT ONE HUE MEANS IMPROVEMENT. r and calibration error run in opposite
directions -- r should go up, calibration error down -- so a single shared palette
would paint improvement green on one panel and red on the other. Each spec therefore
carries its own sense, and the colour always means better or worse. The ANNOTATION
keeps the raw signed difference, so nothing is hidden by the reorientation: a cell
whose calibration error fell by 0.2 reads "-0.200" on an improvement-coloured tile.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from cogformer.simulator.design_utils import build_parameter_mask, expand_design_config_to_mask
from cogformer.diagnostics.metric.adaptive_metrics import _calibration_error
from .adaptive_metrics import _compact_grids, _draw_grids


def _pearson_r(estimates, targets):
    """corr(truth, posterior mean) -- the same r the recovery panels annotate."""
    y = estimates.mean(axis=1)
    if np.std(targets) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(targets, y)[0, 1])


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
) -> plt.Figure:
    """Delta heatmaps for ``pred_b`` minus ``pred_a`` on shared ``true``.

    Both arms must have been evaluated on the SAME datasets -- one ``true`` is taken,
    not one per arm -- because a cell here subtracts two numbers and is meaningless
    if they describe different data. Callers are expected to have checked that.

    ``symmetric`` puts the colour limits at +/- max|delta| per panel. Turning it off
    uses each panel's own min/max, which exaggerates a panel where nothing moved.

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
    r_key = f"$\\Delta$ r   ({hi} $-$ {lo})"
    c_key = f"$\\Delta$ calibration error   ({hi} $-$ {lo})"
    grids = {r_key: np.full((num_rows, num_cols), np.nan),
             c_key: np.full((num_rows, num_cols), np.nan)}

    for r in range(num_rows):
        for c in range(num_cols):
            if parameter_mask[r, c] != 1.0:
                continue
            t = true[:, r, c]
            a, b = pred_a[:, :, r, c], pred_b[:, :, r, c]
            grids[r_key][r, c] = _pearson_r(b, t) - _pearson_r(a, t)
            grids[c_key][r, c] = (
                _calibration_error(b, t, alphas, lowers, uppers, aggregation)
                - _calibration_error(a, t, alphas, lowers, uppers, aggregation))

    compacted = _compact_grids(
        grids, parameter_mask, design_config, num_rows, max_num_categories,
        intrinsic_params, col_labels, global_params, global_only,
        categorical_regressors,
        (intercept_color, main_effect_color, interaction_color))
    if compacted is None:
        return None
    grids, parameter_mask, row_labels, row_colors, col_labels = compacted

    # "vlag" runs blue (low) -> red (high). r improves upward, so it is used as-is
    # and red means better; calibration error improves DOWNWARD, so that panel takes
    # the reversed ramp and red means better there too.
    specs = []
    for key, cmap_name in ((r_key, "vlag"), (c_key, "vlag_r")):
        g = grids[key]
        m = float(np.nanmax(np.abs(g))) if symmetric else None
        specs.append((key, sns.color_palette(cmap_name, as_cmap=True),
                      -m if m else None, m if m else None))

    return _draw_grids(grids, specs, parameter_mask, row_labels, row_colors,
                       col_labels, figsize=figsize, title_fontsize=title_fontsize,
                       label_fontsize=label_fontsize, tick_fontsize=tick_fontsize,
                       annot_fontsize=annot_fontsize, fmt=fmt)
