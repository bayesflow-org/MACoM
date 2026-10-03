import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import seaborn as sns

from cogformer.simulator.design_utils import build_parameter_mask, expand_design_config_to_mask, build_row_labels


def compute_r_matrix(
    true: np.ndarray,
    pred: np.ndarray,
    design_config: dict,
    intrinsic_params: list[str],
    max_num_categories: int,
    parameter_mask: np.ndarray = None,
    max_num_regressors: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-cell Pearson r over the parameter grid.

    Returns (r_matrix, parameter_mask) both clipped to n_design_rows but not
    compacted — NaN for inactive cells.  Suitable for delta computation.

    true : (B, num_rows, num_cols)
    pred : (B, num_samples, num_rows, num_cols) or (B, num_rows, num_cols)
    """
    if parameter_mask is None:
        parameter_mask = build_parameter_mask(
            design_config=design_config,
            intrinsic_params=intrinsic_params,
            max_num_categories=max_num_categories,
            keep_intercept=True,
        )
    elif parameter_mask.ndim == 3:
        parameter_mask = parameter_mask[0]

    design_config = expand_design_config_to_mask(design_config, parameter_mask, max_num_categories,
                                                 max_num_regressors=max_num_regressors)
    regressor_keys = list(design_config.keys())
    block_width = max_num_categories - 1
    main_keys = [k for k in regressor_keys if k != "1" and ":" not in k]
    inter_keys = [k for k in regressor_keys if ":" in k]
    n_design_rows = (
        (1 if "1" in regressor_keys else 0)
        + len(main_keys) * block_width
        + len(inter_keys) * block_width ** 2
    )
    num_rows = min(parameter_mask.shape[0], n_design_rows)
    num_cols = parameter_mask.shape[1]
    parameter_mask = parameter_mask[:num_rows].copy()

    if pred.ndim == 4:
        pred_mean = pred.mean(axis=1)
    else:
        pred_mean = pred
    true = true[:, :num_rows, :]
    pred_mean = pred_mean[:, :num_rows, :]

    r_matrix = np.full((num_rows, num_cols), np.nan)
    for r in range(num_rows):
        for c in range(num_cols):
            if parameter_mask[r, c] == 1.0:
                x = true[:, r, c]
                y = pred_mean[:, r, c]
                if np.std(x) > 1e-8 and np.std(y) > 1e-8:
                    r_matrix[r, c] = np.corrcoef(x, y)[0, 1]

    return r_matrix, parameter_mask


def _compact(r_matrix, parameter_mask, row_labels, row_colors, variable_names,
             global_cols=None, global_only=False):
    """Drop all-NaN rows and columns for display; return compacted versions.

    ``global_cols`` (a set of column indices for global params like ``pi``) are
    excluded, or kept exclusively when ``global_only`` — so global params can be
    rendered separately from the coefficient matrix.
    """
    col_idx = np.where(np.any(parameter_mask, axis=0))[0]
    if global_cols is not None:
        keep = (lambda c: c in global_cols) if global_only else (lambda c: c not in global_cols)
        col_idx = np.array([c for c in col_idx if keep(c)], dtype=int)
    # Rows with no active cell in the KEPT columns are dropped — a global-only
    # panel keeps just the row(s) where the global param lives.
    row_idx = np.where(np.any(parameter_mask[:, col_idx], axis=1))[0] if col_idx.size else np.array([], dtype=int)
    return (
        r_matrix[np.ix_(row_idx, col_idx)],
        parameter_mask[np.ix_(row_idx, col_idx)],
        [row_labels[i] for i in row_idx],
        [row_colors[i] for i in row_idx],
        [variable_names[i] for i in col_idx],
    )


def _build_row_labels(design_config, num_rows, max_num_categories,
                      intercept_color, main_effect_color, interaction_color,
                      categorical_regressors=None, parameter_mask=None):
    """Row labels + colors aligned with the parameter-mask row layout
    (intercept, main-effect blocks, then interaction (i, j) blocks)."""
    labels, kinds = build_row_labels(design_config, max_num_categories, return_kinds=True,
                                     categorical_regressors=categorical_regressors,
                                     parameter_mask=parameter_mask)
    kind_color = {"intercept": intercept_color, "main": main_effect_color, "interaction": interaction_color}
    colors = [kind_color[k] for k in kinds]
    return labels[:num_rows], colors[:num_rows]


def _heatmap_axes(ax, r_matrix, parameter_mask, row_labels, row_colors, col_labels,
                  cmap, norm, annot_fontsize, fmt, na_alpha=0.08):
    """Draw the heatmap, annotations, and N/A tiles onto ax."""
    num_rows, num_cols = r_matrix.shape
    cm = plt.get_cmap(cmap)
    active = parameter_mask == 1.0
    masked = np.ma.masked_where(~active, r_matrix)
    ax.imshow(masked, cmap=cm, norm=norm, aspect="equal")

    for r in range(num_rows):
        for c in range(num_cols):
            if active[r, c] and not np.isnan(r_matrix[r, c]):
                rgba = cm(norm(r_matrix[r, c]))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(c, r, format(r_matrix[r, c], fmt),
                        ha="center", va="center", fontsize=annot_fontsize,
                        color="white" if luminance < 0.5 else "black")
            elif not active[r, c]:
                ax.add_patch(Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    facecolor=row_colors[r], alpha=na_alpha, zorder=2,
                ))
                ax.text(c, r, "N/A", ha="center", va="center",
                        fontsize=annot_fontsize + 1, fontweight="bold",
                        color=row_colors[r], alpha=0.7, zorder=3)

    ax.set_xticks(range(num_cols))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(num_rows))
    ax.set_yticklabels(row_labels)
    for yi, lbl in enumerate(ax.get_yticklabels()):
        lbl.set_color(row_colors[yi])
    ax.tick_params(axis="both", length=0)
    ax.set_xlim(-0.5, num_cols - 0.5)
    ax.set_ylim(num_rows - 0.5, -0.5)


def heatmap_axes(ax, matrix, parameter_mask, row_labels, row_colors, col_labels,
                 cmap, norm, annot_fontsize, fmt):
    """Draw a compacted coefficient-grid matrix (e.g. from :func:`adaptive_metric_matrix`)."""
    _heatmap_axes(ax, matrix, parameter_mask, row_labels, row_colors, col_labels,
                  cmap, norm, annot_fontsize, fmt)


def adaptive_metric_matrix(
    values: dict,
    parameter_mask: np.ndarray,
    design_config: dict,
    intrinsic_params: list[str],
    max_num_categories: int,
    variable_names: list[str] = None,
    global_params: list[str] = None,
    intercept_color: str = "#4e2a84",
    main_effect_color: str = "#6969ff",
    interaction_color: str = "#ff6969",
    max_num_regressors: int = None,
    categorical_regressors: list[str] = None,
):
    """Place per-cell values keyed like ``adaptive_metrics`` rows on the coefficient grid.

    ``values`` maps the row labels of an ``adaptive_metrics`` table (bare parameter name
    for the intercept row, ``"<param> -- <coefficient>"`` otherwise) to a number. The
    labels are rebuilt here exactly as ``adaptive_metrics`` builds them, so any key that
    finds no active cell means the two conventions have drifted.

    Global parameters are dropped, as in ``adaptive_correlation``.

    Returns ``(matrix, display_mask, row_labels, row_colors, col_labels, unplaced)``,
    compacted for display; ``unplaced`` lists the keys of ``values`` that matched no cell.
    """
    if variable_names is None:
        variable_names = intrinsic_params
    if parameter_mask.ndim == 3:
        parameter_mask = parameter_mask[0]

    design_config = expand_design_config_to_mask(design_config, parameter_mask, max_num_categories,
                                                 max_num_regressors=max_num_regressors)
    metric_row_labels = build_row_labels(design_config, max_num_categories,
                                         categorical_regressors=categorical_regressors,
                                         parameter_mask=parameter_mask)
    num_rows = min(parameter_mask.shape[0], len(metric_row_labels))
    parameter_mask = parameter_mask[:num_rows].copy()
    num_cols = parameter_mask.shape[1]

    matrix = np.full((num_rows, num_cols), np.nan)
    placed = set()
    for r in range(num_rows):
        for c in range(num_cols):
            if parameter_mask[r, c] != 1.0:
                continue
            key = variable_names[c] if r == 0 else f"{variable_names[c]} -- {metric_row_labels[r]}"
            if key in values:
                matrix[r, c] = values[key]
                placed.add(key)
    unplaced = [k for k in values if k not in placed]

    row_labels, row_colors = _build_row_labels(
        design_config, num_rows, max_num_categories,
        intercept_color, main_effect_color, interaction_color,
        categorical_regressors=categorical_regressors, parameter_mask=parameter_mask,
    )
    global_cols = {i for i, p in enumerate(intrinsic_params) if global_params and p in global_params}
    return (*_compact(matrix, parameter_mask, row_labels, row_colors, variable_names,
                      global_cols=global_cols), unplaced)


def adaptive_correlation_delta(
    r_matrix_a: np.ndarray,
    r_matrix_b: np.ndarray,
    parameter_mask: np.ndarray,
    design_config: dict,
    intrinsic_params: list[str],
    max_num_categories: int,
    variable_names: list[str] = None,
    label_a: str = "base",
    label_b: str = "fine-tuned",
    intercept_color: str = "#4e2a84",
    main_effect_color: str = "#6969ff",
    interaction_color: str = "#ff6969",
    cmap: str = "RdBu_r",
    figsize: tuple = None,
    title_fontsize: int = 18,
    label_fontsize: int = 15,
    tick_fontsize: int = 13,
    annot_fontsize: int = 11,
    fmt: str = "+.2f",
    max_num_regressors: int = None,
    categorical_regressors: list[str] = None,
) -> plt.Figure:
    """Heatmap of Δr = r_b − r_a, using the same cell layout as adaptive_correlation.

    Positive values (red, under the RdBu_r map) indicate improvement in b; negative values (blue) indicate
    degradation.  Cells inactive in either matrix are shown as N/A.

    r_matrix_a, r_matrix_b : outputs of compute_r_matrix (same shape, same design).
    parameter_mask          : clipped mask from compute_r_matrix.
    """
    if variable_names is None:
        variable_names = intrinsic_params

    design_config = expand_design_config_to_mask(design_config, parameter_mask, max_num_categories,
                                                 max_num_regressors=max_num_regressors)
    num_rows, num_cols = parameter_mask.shape

    row_labels, row_colors = _build_row_labels(
        design_config, num_rows, max_num_categories,
        intercept_color, main_effect_color, interaction_color,
        categorical_regressors=categorical_regressors, parameter_mask=parameter_mask,
    )

    delta = r_matrix_b - r_matrix_a
    active_mask = (parameter_mask == 1.0) & ~np.isnan(r_matrix_a) & ~np.isnan(r_matrix_b)
    delta_masked = np.where(active_mask, delta, np.nan)
    display_mask = parameter_mask.copy()

    delta_masked, display_mask, row_labels, row_colors, col_labels = _compact(
        delta_masked, display_mask, row_labels, row_colors, variable_names
    )
    num_rows, num_cols = delta_masked.shape

    cell_size = 1.0
    if figsize is None:
        figsize = (num_cols * cell_size + 2.5, num_rows * cell_size + 2.5)

    fig, ax = plt.subplots(figsize=figsize)
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    _heatmap_axes(ax, delta_masked, display_mask, row_labels, row_colors, col_labels,
                  cmap, norm, annot_fontsize, fmt)

    ax.set_title(f"Δr ({label_b} − {label_a})", fontsize=title_fontsize)
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=label_fontsize)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def adaptive_correlation(
    true: np.ndarray,
    pred: np.ndarray,
    design_config: dict,
    intrinsic_params: list[str],
    max_num_categories: int,
    parameter_mask: np.ndarray = None,
    variable_names: list[str] = None,
    intercept_color: str = "#4e2a84",
    main_effect_color: str = "#6969ff",
    interaction_color: str = "#ff6969",
    cmap: str = "RdBu_r",
    figsize: tuple = None,
    title_fontsize: int = 18,
    label_fontsize: int = 15,
    tick_fontsize: int = 13,
    annot_fontsize: int = 11,
    fmt: str = ".2f",
    global_params: list[str] = None,
    global_only: bool = False,
    max_num_regressors: int = None,
    categorical_regressors: list[str] = None,
) -> plt.Figure:
    """Heatmap of Pearson r, identical to the r values in adaptive_recovery scatter panels.

    ``global_params`` (e.g. the contamination probability ``pi``) are estimated
    jointly but presented separately: with ``global_only=False`` (default) their
    columns are excluded from the heatmap; with ``global_only=True`` only those are
    kept (used by :func:`global_correlation`). Returns ``None`` if no column is
    selected.

    true : (B, num_rows, num_cols)
    pred : (B, num_samples, num_rows, num_cols) or (B, num_rows, num_cols)
    """
    if variable_names is None:
        variable_names = intrinsic_params

    r_matrix, parameter_mask = compute_r_matrix(
        true, pred, design_config, intrinsic_params, max_num_categories, parameter_mask,
        max_num_regressors=max_num_regressors,
    )
    design_config = expand_design_config_to_mask(design_config, parameter_mask, max_num_categories,
                                                 max_num_regressors=max_num_regressors)
    num_rows = parameter_mask.shape[0]

    row_labels, row_colors = _build_row_labels(
        design_config, num_rows, max_num_categories,
        intercept_color, main_effect_color, interaction_color,
        categorical_regressors=categorical_regressors, parameter_mask=parameter_mask,
    )

    global_cols = (
        {i for i, p in enumerate(intrinsic_params) if p in global_params}
        if global_params else None
    )
    r_matrix, parameter_mask, row_labels, row_colors, col_labels = _compact(
        r_matrix, parameter_mask, row_labels, row_colors, variable_names,
        global_cols=global_cols, global_only=global_only,
    )
    num_rows, num_cols = r_matrix.shape
    if num_cols == 0:
        return None

    cell_size = 1.0
    if figsize is None:
        figsize = (num_cols * cell_size + 2.5, num_rows * cell_size + 2.5)

    fig, ax = plt.subplots(figsize=figsize)
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    _heatmap_axes(ax, r_matrix, parameter_mask, row_labels, row_colors, col_labels,
                  cmap, norm, annot_fontsize, fmt)

    ax.set_title("Recovery Correlation (r)", fontsize=title_fontsize)
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=label_fontsize)
    fig.tight_layout()
    return fig


def global_correlation(
    true: np.ndarray,
    pred: np.ndarray,
    design_config: dict,
    intrinsic_params: list[str],
    max_num_categories: int,
    global_params: list[str],
    parameter_mask: np.ndarray = None,
    variable_names: list[str] = None,
    global_color: str = "#4e2a84",
    **kwargs,
) -> plt.Figure:
    """Recovery-correlation heatmap for global parameters (e.g. ``pi``) only,
    plotted separately from the coefficient matrix. Thin wrapper over
    :func:`adaptive_correlation` with ``global_only=True``. Returns ``None`` when
    no global parameter is active.
    """
    return adaptive_correlation(
        true=true,
        pred=pred,
        design_config=design_config,
        intrinsic_params=intrinsic_params,
        max_num_categories=max_num_categories,
        parameter_mask=parameter_mask,
        variable_names=variable_names,
        intercept_color=global_color,
        main_effect_color=global_color,
        interaction_color=global_color,
        global_params=global_params,
        global_only=True,
        **kwargs,
    )
