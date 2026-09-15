import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import seaborn as sns

from cogformer.simulator.design_utils import build_parameter_mask, expand_design_config_to_mask, build_row_labels
from cogformer.diagnostics.metric.adaptive_metrics import (
    _rmse,
    _contraction,
    _calibration_error,
    _log_gamma,
    _gamma_null_distribution,
)


def adaptive_metrics(
    true: np.ndarray,
    pred: np.ndarray,
    design_config: dict,
    intrinsic_params: list[str],
    max_num_categories: int,
    parameter_mask: np.ndarray = None,
    variable_names: list[str] = None,
    normalize: str | None = "range",
    aggregation=np.median,
    resolution: int = 20,
    min_quantile: float = 0.005,
    max_quantile: float = 0.995,
    num_null_draws: int = 1000,
    log_gamma_quantile: float = 0.05,
    skip_log_gamma: bool = True,
    intercept_color: str = "#4e2a84",
    main_effect_color: str = "#6969ff",
    interaction_color: str = "#ff6969",
    figsize: tuple | None = None,
    title_fontsize: int = 16,
    label_fontsize: int = 12,
    tick_fontsize: int = 10,
    annot_fontsize: int = 10,
    fmt: str = ".3f",
    global_params: list[str] = None,
    global_only: bool = False,
    max_num_regressors: int = None,
    categorical_regressors: list[str] = None,
) -> plt.Figure:
    """
    Heatmap visualization of four adaptive diagnostic metrics arranged in a 2×2 grid.

    ``global_params`` (e.g. the contamination probability ``pi``) are estimated
    jointly but presented separately: with ``global_only=False`` (default) their
    columns are excluded from the heatmaps; with ``global_only=True`` only those
    are kept (used by :func:`global_metrics`). Returns ``None`` if no column is
    selected.

    Each subplot shows a ``num_rows × num_cols`` heatmap (rows = regressors/intercept,
    cols = intrinsic parameters), colour-coded by metric value. Inactive cells are
    rendered as grey N/A tiles, consistent with ``adaptive_recovery`` and
    ``adaptive_coverage``.

    Metrics
    -------
    - (N)RMSE
    - Posterior Contraction
    - Calibration Error
    - Log Gamma

    Parameters
    ----------
    true : np.ndarray of shape (batch_size, num_rows, num_cols)
    pred : np.ndarray of shape (batch_size, num_draws, num_rows, num_cols)
    design_config : dict
    intrinsic_params : list[str]
    max_num_categories : int
    parameter_mask : np.ndarray, optional
    variable_names : list[str], optional
    normalize : str or None, optional (default "range")
    aggregation : callable, optional (default np.median)
    resolution : int, optional (default 20)
    min_quantile, max_quantile : float
    num_null_draws : int, optional (default 1000)
    log_gamma_quantile : float, optional (default 0.05)
    intercept_color, main_effect_color, interaction_color : str
        N/A tile accent colours, matching the other adaptive plot.
    figsize : tuple, optional
    title_fontsize, label_fontsize, tick_fontsize, annot_fontsize : int
    fmt : str, optional (default ".3f")
        Format string for cell annotations.

    Returns
    -------
    plt.Figure
    """
    if parameter_mask is None:
        parameter_mask = build_parameter_mask(
            design_config=design_config,
            max_num_categories=max_num_categories,
            intrinsic_params=intrinsic_params,
            keep_intercept=True,
        )
    if parameter_mask.ndim == 3:
        parameter_mask = parameter_mask[0]

    design_config = expand_design_config_to_mask(design_config, parameter_mask, max_num_categories,
                                                 max_num_regressors=max_num_regressors)
    col_labels = variable_names if variable_names is not None else intrinsic_params
    regressor_keys = list(design_config.keys())
    block_width = max_num_categories - 1
    main_keys = [k for k in regressor_keys if k != "1" and ":" not in k]
    inter_keys_dc = [k for k in regressor_keys if ":" in k]

    batch_size, num_draws, num_rows, num_cols = pred.shape
    n_design_rows = (
        (1 if "1" in regressor_keys else 0)
        + len(main_keys) * block_width
        + len(inter_keys_dc) * block_width ** 2
    )
    num_rows = min(num_rows, n_design_rows)
    parameter_mask = parameter_mask[:num_rows]
    true = true[:, :num_rows, :]
    pred = pred[:, :, :num_rows, :]

    # ------------------------------------------------------------------
    # Pre-compute shared quantities
    # ------------------------------------------------------------------
    alphas = np.linspace(min_quantile, max_quantile, resolution)
    regions = 1 - alphas
    lowers = regions / 2
    uppers = 1 - lowers

    if not skip_log_gamma:
        null_dist = _gamma_null_distribution(batch_size, num_draws, num_null_draws)
        null_quantile = np.quantile(null_dist, log_gamma_quantile)

    # ------------------------------------------------------------------
    # Build metric grids  (NaN where masked)
    # ------------------------------------------------------------------
    rmse_col = "RMSE" if normalize is None else "NRMSE"

    grids = {
        rmse_col: np.full((num_rows, num_cols), np.nan),
        "Posterior Contraction": np.full((num_rows, num_cols), np.nan),
        "Calibration Error": np.full((num_rows, num_cols), np.nan),
    }
    if not skip_log_gamma:
        grids["Log Gamma"] = np.full((num_rows, num_cols), np.nan)

    for r in range(num_rows):
        for c in range(num_cols):
            if parameter_mask[r, c] != 1.0:
                continue
            estimates = pred[:, :, r, c]
            targets = true[:, r, c]
            grids[rmse_col][r, c] = _rmse(estimates, targets, normalize, aggregation)
            grids["Posterior Contraction"][r, c] = _contraction(estimates, targets, aggregation)
            grids["Calibration Error"][r, c] = _calibration_error(
                estimates, targets, alphas, lowers, uppers, aggregation
            )
            if not skip_log_gamma:
                grids["Log Gamma"][r, c] = _log_gamma(estimates, targets, null_quantile)

    compacted = _compact_grids(
        grids, parameter_mask, design_config, num_rows, max_num_categories,
        intrinsic_params, col_labels, global_params, global_only,
        categorical_regressors,
        (intercept_color, main_effect_color, interaction_color),
    )
    if compacted is None:
        return None
    grids, parameter_mask, row_labels, row_colors, col_labels = compacted
    num_rows, num_cols = parameter_mask.shape

    metric_specs = [
        (rmse_col,                "mako",   0.0, 1.0),
        ("Posterior Contraction", "rocket", 0.0, 1.0),
        ("Calibration Error",     "crest",  0.0, 1.0),
    ]
    if not skip_log_gamma:
        metric_specs.append(("Log Gamma", "flare", None, None))

    return _draw_grids(grids, metric_specs, parameter_mask, row_labels, row_colors,
                       col_labels, figsize=figsize, title_fontsize=title_fontsize,
                       label_fontsize=label_fontsize, tick_fontsize=tick_fontsize,
                       annot_fontsize=annot_fontsize, fmt=fmt)


def _compact_grids(grids, parameter_mask, design_config, num_rows, max_num_categories,
                   intrinsic_params, col_labels, global_params, global_only,
                   categorical_regressors, kind_colors):
    """Row/column labels, then the drop of all-N/A rows and columns.

    Split out of :func:`adaptive_metrics` so a plot of DIFFERENCES between two arms
    gets the identical grid without a second copy of this bookkeeping. Returns None
    when nothing survives, which the callers pass straight on.
    """
    intercept_color, main_effect_color, interaction_color = kind_colors
    row_labels, row_kinds = build_row_labels(design_config, max_num_categories,
                                             return_kinds=True,
                                             categorical_regressors=categorical_regressors,
                                             parameter_mask=parameter_mask)
    _kind_color = {"intercept": intercept_color, "main": main_effect_color,
                   "interaction": interaction_color}
    row_colors = [_kind_color[k] for k in row_kinds]
    row_labels = row_labels[:num_rows]
    row_colors = row_colors[:num_rows]

    # Compact: drop columns that are entirely N/A. Global parameters (e.g. pi) are
    # estimated jointly but presented separately: exclude their columns from the
    # heatmaps, or keep only them when global_only.
    col_idx = np.where(np.any(parameter_mask, axis=0))[0]
    if global_params and intrinsic_params is not None:
        global_cols = {i for i, p in enumerate(intrinsic_params) if p in global_params}
        keep = (lambda c: c in global_cols) if global_only else (lambda c: c not in global_cols)
        col_idx = np.array([c for c in col_idx if keep(c)], dtype=int)
    if col_idx.size == 0:
        return None
    # Drop rows with no active cell in the KEPT columns — a global-only panel keeps
    # just the row(s) where the global param lives, with no N/A rows below it.
    row_idx = np.where(np.any(parameter_mask[:, col_idx], axis=1))[0]
    if row_idx.size == 0:
        return None

    parameter_mask = parameter_mask[np.ix_(row_idx, col_idx)]
    grids = {k: v[np.ix_(row_idx, col_idx)] for k, v in grids.items()}
    row_labels = [row_labels[i] for i in row_idx]
    row_colors = [row_colors[i] for i in row_idx]
    col_labels = [col_labels[i] for i in col_idx]
    if global_only:
        # Global params aren't coefficient-matrix rows; the column title names them.
        row_labels = [""] * len(row_labels)
    return grids, parameter_mask, row_labels, row_colors, col_labels


def _draw_grids(grids, metric_specs, parameter_mask, row_labels, row_colors,
                col_labels, figsize=None, title_fontsize=16, label_fontsize=12,
                tick_fontsize=10, annot_fontsize=10, fmt=".3f", cell_size=1.5):
    """One annotated heatmap per spec, sharing the coefficient-matrix layout.

    ``metric_specs`` is (name, palette, vmin, vmax); a None limit is taken from the
    grid. Cells carry their value, inactive cells become tinted N/A tiles. Split out
    of :func:`adaptive_metrics` so the delta plot renders through the same code --
    two implementations of this layout is how the two would drift apart.
    """
    num_rows, num_cols = parameter_mask.shape
    n_metrics = len(metric_specs)
    if figsize is None:
        figsize = (n_metrics * num_cols * cell_size + 2.0,
                   num_rows * cell_size + 2.5)

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize, squeeze=False)
    axes = axes[0]

    for ax, (metric_name, palette, vmin, vmax) in zip(axes, metric_specs):
        grid = grids[metric_name]
        active = parameter_mask == 1.0

        if vmin is None:
            vmin = np.nanmin(grid)
        if vmax is None:
            vmax = np.nanmax(grid)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = (palette if isinstance(palette, mcolors.Colormap)
                else sns.color_palette(palette, as_cmap=True))

        # Masked array — matplotlib leaves NaN cells transparent
        masked = np.ma.masked_where(~active, grid)
        ax.imshow(masked, cmap=cmap, norm=norm, aspect="equal")

        # Annotate active cells with numeric values
        for r in range(num_rows):
            for c in range(num_cols):
                if active[r, c]:
                    rgba = cmap(norm(grid[r, c]))
                    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    text_color = "white" if luminance < 0.5 else "black"
                    ax.text(
                        c, r,
                        format(grid[r, c], fmt),
                        ha="center", va="center",
                        fontsize=annot_fontsize,
                        color=text_color,
                    )

        # Render inactive cells as tinted N/A tiles
        for r in range(num_rows):
            for c in range(num_cols):
                if not active[r, c]:
                    tint = row_colors[r]
                    ax.add_patch(
                        Rectangle(
                            (c - 0.5, r - 0.5), 1, 1,
                            facecolor=tint, alpha=0.08, zorder=2,
                        )
                    )
                    ax.text(
                        c, r, "N/A",
                        ha="center", va="center",
                        fontsize=annot_fontsize + 1,
                        color=tint, alpha=0.7, zorder=3,
                    )

        ax.set_title(metric_name, fontsize=title_fontsize)

        ax.set_xticks(range(num_cols))
        ax.set_xticklabels(col_labels, fontsize=tick_fontsize)
        ax.set_yticks(range(num_rows))
        ax.set_yticklabels(row_labels, fontsize=label_fontsize)

        ax.tick_params(axis="both", length=0)
        ax.set_xlim(-0.5, num_cols - 0.5)
        ax.set_ylim(num_rows - 0.5, -0.5)

    fig.tight_layout()
    return fig


def global_metrics(
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
    """Diagnostic-metric heatmap(s) for global parameters (e.g. ``pi``), plotted
    separately from the coefficient matrix. Thin wrapper over
    :func:`adaptive_metrics` with ``global_only=True``. Returns ``None`` when no
    global parameter is active.
    """
    return adaptive_metrics(
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


if __name__ == "__main__":
    from cogformer.simulators.context_manager import ContextManager
    from cogformer.utils.plot_utils import cogformer_cm_colors

    color = cogformer_cm_colors()

    design_config = {
        "1": ["v", "a", "z", "tau"],
        "u_1": ["v", "a", "tau"],
        "u_2": ["v", "a", "z"],
        "u_1:u_2": ["v", "a"],
    }
    intrinsic_params = design_config["1"]
    variable_names = [r"$v$", r"$a$", r"$z$", r"$\tau$"]

    num_categories = 2
    parameter_mask = build_parameter_mask(
        design_config=design_config,
        max_num_categories=num_categories,
        intrinsic_params=intrinsic_params,
        keep_intercept=True,
    )
    if parameter_mask.ndim == 3:
        parameter_mask = parameter_mask[0]

    num_rows, num_cols = parameter_mask.shape
    batch_size, num_draws = 200, 256

    true = np.random.normal(0.0, 1.0, (batch_size, num_rows, num_cols))
    pred = true[:, None, :, :] + np.random.normal(0.0, 0.5, (batch_size, num_draws, num_rows, num_cols))

    fig = adaptive_metrics(
        true=true,
        pred=pred,
        design_config=design_config,
        intrinsic_params=intrinsic_params,
        max_num_categories=num_categories,
        parameter_mask=parameter_mask,
        variable_names=variable_names,
        intercept_color=color["intercept"],
        main_effect_color=color["main_effect"],
        interaction_color=color["interaction"],
    )

    fig.savefig("test_adaptive_metrics.pdf", bbox_inches="tight")
    print("Saved test_adaptive_metrics.pdf")