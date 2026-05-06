import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import seaborn as sns

from cogformer.simulator.design_utils import build_parameter_mask, expand_design_config_to_mask


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
) -> plt.Figure:
    """
    Heatmap of Pearson r computed cell-by-cell over the parameter grid,
    identical to the r values annotated on each scatter panel in adaptive_recovery.

    Cell size matches adaptive_metrics (1 inch per cell).
    Inactive cells are rendered as tinted N/A tiles.

    true : (B, num_rows, num_cols)
    pred : (B, num_samples, num_rows, num_cols) or (B, num_rows, num_cols)
    """
    if variable_names is None:
        variable_names = intrinsic_params

    if parameter_mask is None:
        parameter_mask = build_parameter_mask(
            design_config=design_config,
            intrinsic_params=intrinsic_params,
            max_num_categories=max_num_categories,
            keep_intercept=True,
        )
    elif parameter_mask.ndim == 3:
        parameter_mask = parameter_mask[0]

    # Expand sparse config to match the mask row count (no-op if already full)
    design_config = expand_design_config_to_mask(design_config, parameter_mask, max_num_categories)

    num_rows, num_cols = parameter_mask.shape
    regressor_keys = list(design_config.keys())
    block_width = max_num_categories - 1
    main_keys = [k for k in regressor_keys if k != "1" and ":" not in k]
    inter_keys = [k for k in regressor_keys if ":" in k]
    n_design_rows = (
        (1 if "1" in regressor_keys else 0)
        + len(main_keys) * block_width
        + len(inter_keys) * block_width ** 2
    )
    num_rows = min(num_rows, n_design_rows)
    parameter_mask = parameter_mask[:num_rows]

    if pred.ndim == 4:
        pred_mean = pred.mean(axis=1)
    else:
        pred_mean = pred
    true = true[:, :num_rows, :]
    pred_mean = pred_mean[:, :num_rows, :]

    # Compute r for each (row, col) cell — same as adaptive_recovery
    r_matrix = np.full((num_rows, num_cols), np.nan)
    for r in range(num_rows):
        for c in range(num_cols):
            if parameter_mask[r, c] == 1.0:
                x = true[:, r, c]
                y = pred_mean[:, r, c]
                if np.std(x) > 1e-8 and np.std(y) > 1e-8:
                    r_matrix[r, c] = np.corrcoef(x, y)[0, 1]

    # Row labels and colours — matching adaptive_metrics / adaptive_recovery
    row_labels = []
    row_colors = []
    _r = 0
    if "1" in regressor_keys:
        row_labels.append(r"$1$")
        row_colors.append(intercept_color)
        _r += 1
    for key in main_keys:
        for _ in range(block_width):
            row_labels.append(fr"${key}$")
            row_colors.append(main_effect_color)
            _r += 1
            if _r >= num_rows:
                break
        if _r >= num_rows:
            break
    for key in inter_keys:
        for _ in range(block_width ** 2):
            row_labels.append(fr"${key}$")
            row_colors.append(interaction_color)
            _r += 1
            if _r >= num_rows:
                break
        if _r >= num_rows:
            break

    # Compact: drop rows and columns that are entirely N/A
    row_idx = np.where(np.any(parameter_mask, axis=1))[0]
    col_idx = np.where(np.any(parameter_mask, axis=0))[0]
    parameter_mask = parameter_mask[np.ix_(row_idx, col_idx)]
    r_matrix   = r_matrix[np.ix_(row_idx, col_idx)]
    row_labels = [row_labels[i] for i in row_idx]
    row_colors = [row_colors[i] for i in row_idx]
    col_labels = [variable_names[i] for i in col_idx]
    num_rows   = len(row_idx)
    num_cols   = len(col_idx)

    cell_size = 1.0  # inches per cell — matches adaptive_metrics
    if figsize is None:
        figsize = (num_cols * cell_size + 2.5, num_rows * cell_size + 2.5)

    fig, ax = plt.subplots(figsize=figsize)

    active = parameter_mask == 1.0
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    cm = plt.get_cmap(cmap)

    masked = np.ma.masked_where(~active, r_matrix)
    ax.imshow(masked, cmap=cm, norm=norm, aspect="equal")

    # Annotate active cells
    for r in range(num_rows):
        for c in range(num_cols):
            if active[r, c] and not np.isnan(r_matrix[r, c]):
                rgba = cm(norm(r_matrix[r, c]))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(
                    c, r, format(r_matrix[r, c], fmt),
                    ha="center", va="center",
                    fontsize=annot_fontsize,
                    color="white" if luminance < 0.5 else "black",
                )

    # Tinted N/A tiles for inactive cells
    for r in range(num_rows):
        for c in range(num_cols):
            if not active[r, c]:
                ax.add_patch(Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    facecolor=row_colors[r], alpha=0.08, zorder=2,
                ))
                ax.text(
                    c, r, "N/A",
                    ha="center", va="center",
                    fontsize=annot_fontsize + 1, fontweight="bold",
                    color=row_colors[r], alpha=0.7, zorder=3,
                )

    ax.set_title("Recovery Correlation (r)", fontsize=title_fontsize)
    ax.set_xticks(range(num_cols))
    ax.set_xticklabels(col_labels, fontsize=tick_fontsize)
    ax.set_yticks(range(num_rows))
    ax.set_yticklabels(row_labels, fontsize=label_fontsize)
    for yi, lbl in enumerate(ax.get_yticklabels()):
        lbl.set_color(row_colors[yi])
    ax.tick_params(axis="both", length=0)
    ax.set_xlim(-0.5, num_cols - 0.5)
    ax.set_ylim(num_rows - 0.5, -0.5)

    fig.tight_layout()
    return fig
