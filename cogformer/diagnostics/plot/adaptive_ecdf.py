import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
from cogformer.simulator.design_utils import build_parameter_mask, build_design_matrix, build_random_design_config, expand_design_config_to_mask, build_row_labels
from cogformer.utils.plot_utils import cogformer_cm_colors, place_legend_below


def compute_ecdf_bands(
    num_datasets: int,
    prob: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute exact pointwise confidence bands for the ECDF of a uniform distribution.

    Uses the fact that the k-th order statistic of n i.i.d. Uniform(0,1) samples
    follows a Beta(k, n−k+1) distribution, giving exact closed-form bands.

    Parameters
    ----------
    num_datasets : int
        Number of datasets (determines ECDF resolution).
    prob : float, optional, default: 0.95
        Confidence level for the bands.

    Returns
    -------
    x : np.ndarray of shape (num_datasets,)
        Sorted x-axis positions (k / num_datasets).
    lower : np.ndarray of shape (num_datasets,)
        Lower confidence band.
    upper : np.ndarray of shape (num_datasets,)
        Upper confidence band.
    """
    k = np.arange(1, num_datasets + 1)
    x = k / num_datasets
    lower = beta_dist.ppf((1 - prob) / 2, k, num_datasets - k + 1)
    upper = beta_dist.ppf((1 + prob) / 2, k, num_datasets - k + 1)
    return x, lower, upper


def adaptive_ecdf(
    true: np.ndarray,
    pred: np.ndarray,
    design_config: dict,
    variable_names: list,
    intrinsic_params: list = None,
    parameter_mask: np.ndarray = None,
    max_num_categories: int = 2,
    intercept_color: str = "#4e2a84",
    main_effect_color: str = "#6969ff",
    interaction_color: str = "#ff6969",
    figsize: tuple | None = None,
    title_fontsize: int = 20,
    label_fontsize: int = 14,
    tick_fontsize: int = 11,
    legend_fontsize: int = 12,
    legend_location: str = "lower right",
    prob: float = 0.95,
    difference: bool = True,
    global_params: list = None,
    global_only: bool = False,
    max_num_regressors: int = None,
    categorical_regressors: list = None,
    arms: list = None,
) -> plt.Figure:
    """
    Adaptive (masked) ECDF calibration plot over a regressor-by-parameter grid.

    ``global_params`` (e.g. the contamination probability ``pi``) are estimated
    jointly but presented separately: with ``global_only=False`` (default) their
    columns are excluded from the grid; with ``global_only=True`` only those are
    ``arms`` overlays SEVERAL prediction sets on one grid: a list of
    ``(pred, color, label)``. Each cell then carries one rank-ECDF curve per arm in
    that arm's colour against the SAME confidence band, which is what makes them
    directly comparable -- the band depends only on the number of datasets, and both
    arms are scored on identical data here. Row-kind tints are given up for the
    curves (colour has to mean which arm) but the N/A tiles keep them. ``pred`` is
    ignored; pass ``arms[0][0]`` for it.

    kept (used by :func:`global_ecdf`). Returns ``None`` if no column is selected.

    For each active cell, computes the fractional rank of the true parameter value
    within the posterior draws and plot the resulting empirical CDF against the
    expected uniform CDF, with simultaneous confidence bands.

    Parameters
    ----------
    true : np.ndarray of shape (num_datasets, num_rows, num_cols)
        True parameter values.
    pred : np.ndarray of shape (num_datasets, num_draws, num_rows, num_cols)
        Posterior draws.
    design_config : dict
        Design configuration mapping regressor names to lists of parameter names.
    variable_names : list of str
        Column (parameter) names for subplot titles.
    intrinsic_params : list of str, optional
        Intrinsic parameter names (used to build the mask if not provided).
    parameter_mask : np.ndarray, optional
        Boolean mask of shape (num_rows, num_cols). Built with build_parameter_mask if None.
    max_num_categories : int, optional, default: 2
        Maximum number of categories (used for row layout).
    intercept_color : str
        Color for intercept row.
    main_effect_color : str
        Color for main effect rows.
    interaction_color : str
        Color for interaction rows.
    figsize : tuple or None
        Figure size. Defaults to (3.2 * num_cols, 2.9 * num_rows).
    title_fontsize : int
        Font size for column titles.
    label_fontsize : int
        Font size for axis labels.
    tick_fontsize : int
        Font size for tick labels.
    legend_fontsize : int
        Font size for legend.
    legend_location : str
        Legend location (unused; legend is placed at figure bottom).
    prob : float, optional, default: 0.95
        Confidence level for simultaneous ECDF bands.
    difference : bool, optional, default: False
        If True, plot ECDF − uniform diagonal instead of raw ECDF.

    Returns
    -------
    fig : plt.Figure
    """
    if arms:
        pred = arms[0][0]
    if parameter_mask is None:
        parameter_mask = build_parameter_mask(
            design_config=design_config,
            max_num_categories=max_num_categories,
            intrinsic_params=intrinsic_params,
            keep_intercept=True,
        )
    elif parameter_mask.ndim == 3:
        parameter_mask = parameter_mask[0]

    design_config = expand_design_config_to_mask(design_config, parameter_mask, max_num_categories,
                                                 max_num_regressors=max_num_regressors)
    num_datasets, num_draws, num_rows, num_cols = pred.shape
    # Row labels + tints, aligned with the parameter-mask row layout
    # (intercept, main-effect blocks, then interaction (i, j) blocks).
    row_ylabels, row_kinds = build_row_labels(design_config, max_num_categories, return_kinds=True,
                                              categorical_regressors=categorical_regressors,
                                              parameter_mask=parameter_mask)
    _kind_color = {"intercept": intercept_color, "main": main_effect_color, "interaction": interaction_color}
    row_tints = [_kind_color[k] for k in row_kinds]
    num_rows = min(num_rows, len(row_ylabels))
    row_ylabels = row_ylabels[:num_rows]
    row_tints = row_tints[:num_rows]

    # Compact: drop columns that are entirely N/A. Global parameters (e.g. pi) are
    # estimated jointly but presented separately: exclude their columns from the
    # matrix, or keep only them when global_only.
    col_idx = np.where(np.any(parameter_mask[:num_rows], axis=0))[0]
    if global_params and intrinsic_params is not None:
        global_cols = {i for i, p in enumerate(intrinsic_params) if p in global_params}
        keep = (lambda c: c in global_cols) if global_only else (lambda c: c not in global_cols)
        col_idx = np.array([c for c in col_idx if keep(c)], dtype=int)
    if col_idx.size == 0:
        return None
    # Drop rows with no active cell in the KEPT columns — a global-only panel keeps
    # just the row(s) where the global param lives, with no N/A rows below it.
    row_idx = np.where(np.any(parameter_mask[:num_rows][:, col_idx], axis=1))[0]
    if row_idx.size == 0:
        return None

    parameter_mask = parameter_mask[np.ix_(row_idx, col_idx)]
    true = true[:, row_idx, :][:, :, col_idx]
    pred = pred[:, :, row_idx, :][:, :, :, col_idx]
    arms_f = ([(q[:, :, row_idx, :][:, :, :, col_idx], c, lab)
               for q, c, lab in arms] if arms else None)
    row_ylabels = [row_ylabels[i] for i in row_idx]
    row_tints   = [row_tints[i]   for i in row_idx]
    col_labels  = [variable_names[i] for i in col_idx]
    if global_only:
        # Global params aren't coefficient-matrix rows; the column title names them.
        row_ylabels = [""] * len(row_ylabels)
    num_rows    = len(row_idx)
    num_cols    = len(col_idx)

    if figsize is None:
        figsize = (3.2 * num_cols, 3.2 * num_rows)

    fig, axarr = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)

    # Precompute confidence bands (same for all cells since num_datasets is fixed)
    band_x, band_lower, band_upper = compute_ecdf_bands(
        num_datasets=num_datasets,
        prob=prob,
    )
    legend_handles = None

    for r_f in range(num_rows):
        ylabel = row_ylabels[r_f]
        tint   = row_tints[r_f]

        for c_f in range(num_cols):
            ax = axarr[r_f, c_f]
            # Square BY CONSTRUCTION, active and N/A cells alike: the box aspect is fixed
            # independently of figsize and of the margins tight_layout leaves, which is
            # what made single-row (intercept-only) grids and 1x1 global panels drift.
            ax.set_box_aspect(1)

            # Masked cell -> N/A tile
            if parameter_mask[r_f, c_f] != 1.0:
                ax.set_facecolor(tint)
                ax.patch.set_alpha(0.05)
                if r_f == num_rows - 1:
                    ax.set_xticks([0.0, 0.5, 1.0])
                    ax.set_xticklabels(["0.0", "0.5", "1.0"])
                    ax.tick_params(axis="x", length=0, labelcolor="none", labelsize=tick_fontsize)
                else:
                    ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                ax.text(
                    0.5, 0.5, "N/A",
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=18, weight="bold",
                    alpha=0.7, color=tint,
                )
                ax.set_xlabel("Fractional rank statistic" if r_f == num_rows - 1 else "", fontsize=label_fontsize)
                continue

            ecdf_y = np.arange(1, num_datasets + 1) / num_datasets
            # The band is drawn ONCE, before the curves: it is a property of
            # num_datasets, not of any arm, so one band is the common reference
            # both arms are read against.
            if difference:
                ax.fill_between(
                    band_x,
                    band_lower - band_x,
                    band_upper - band_x,
                    color="grey", alpha=0.07,
                    label=f"{int(prob * 100)}% Confidence Band",
                )
                ax.axhline(0.0, color="black", linestyle="dashed", label="Ideal (Uniform)")
            else:
                ax.fill_between(
                    band_x,
                    band_lower,
                    band_upper,
                    color="grey", alpha=0.07,
                    label=f"{int(prob * 100)}% Confidence Band",
                )
                ax.plot([0, 1], [0, 1], color="black", linestyle="dashed", label="Ideal (Uniform)")

            for q, acol, alab in (arms_f if arms_f else [(pred, tint, None)]):
                # Fractional rank: per dataset, the fraction of draws below truth.
                ranks_sorted = np.sort(np.mean(
                    q[:, :, r_f, c_f] < true[:, r_f, c_f, np.newaxis], axis=1))
                ax.plot(ranks_sorted,
                        ecdf_y - ranks_sorted if difference else ecdf_y,
                        color=acol, alpha=1.0,
                        label=alab or ("ECDF Difference" if difference
                                       else "Empirical CDF"))

            ax.set_xlim(-0.02, 1.02)
            ax.set_ylabel(ylabel if c_f == 0 else "", fontsize=label_fontsize)
            ax.set_xlabel("Fractional rank statistic" if r_f == num_rows - 1 else "", fontsize=label_fontsize)

            if col_labels is not None and r_f == 0:
                ax.set_title(col_labels[c_f], fontsize=title_fontsize)

            ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
            ax.set_xticks(ticks)
            if not difference:
                ax.set_yticks(ticks)
            ax.tick_params(axis="both", labelsize=tick_fontsize)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, color="lightgray", linestyle="--", linewidth=0.5, alpha=0.35)

            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

    fig.tight_layout()
    if legend_handles is not None:
        # Below the measured extent of the bottom row's labels (see place_legend_below).
        # The old fixed subplots_adjust(bottom=0.1) overrode tight_layout's margin and,
        # on single-row grids, put the legend on top of the x-axis label.
        place_legend_below(fig, legend_handles, legend_labels,
                           ncol=len(legend_handles), fontsize=legend_fontsize)
    return fig


def global_ecdf(
    true: np.ndarray,
    pred: np.ndarray,
    design_config: dict,
    variable_names: list,
    global_params: list,
    intrinsic_params: list = None,
    parameter_mask: np.ndarray = None,
    max_num_categories: int = 2,
    global_color: str = "#4e2a84",
    **kwargs,
) -> plt.Figure:
    """ECDF calibration panel(s) for global parameters (e.g. ``pi``), plotted
    separately from the coefficient matrix. Thin wrapper over :func:`adaptive_ecdf`
    with ``global_only=True``. Returns ``None`` when no global parameter is active.
    """
    return adaptive_ecdf(
        true=true,
        pred=pred,
        design_config=design_config,
        variable_names=variable_names,
        intrinsic_params=intrinsic_params,
        parameter_mask=parameter_mask,
        max_num_categories=max_num_categories,
        intercept_color=global_color,
        main_effect_color=global_color,
        interaction_color=global_color,
        global_params=global_params,
        global_only=True,
        **kwargs,
    )


if __name__ == "__main__":

    color = cogformer_cm_colors()

    design_config = {
        "1": ["v", "a", "z", "tau"],
        "u_1": ["v", "a", "tau"],
        "u_2": ["v", "a", "z"],
        "u_1:u_2": ["v", "a"],
    }

    intrinsic_params = design_config["1"]
    variable_names = [r"$v$", r"$a$", r"$z$", r"$\tau$"]

    num_categories = 3
    batch_size = 200
    num_draws = 256

    parameter_mask = build_parameter_mask(
        design_config=design_config,
        max_num_categories=num_categories,
        intrinsic_params=intrinsic_params,
        keep_intercept=True,
    )
    if parameter_mask.ndim == 3:
        parameter_mask = parameter_mask[0]

    num_rows, num_cols = parameter_mask.shape

    # Good calibration: true and draws from same marginal → exchangeable → ECDF ≈ diagonal
    true = np.random.normal(0.0, 1.0, size=(batch_size, num_rows, num_cols))
    pred = np.random.normal(0.0, 1.0, size=(batch_size, num_draws, num_rows, num_cols))

    print("true:", true.shape)
    print("pred:", pred.shape)
    print("mask:", parameter_mask.shape)

    fig = adaptive_ecdf(
        true=true,
        pred=pred,
        design_config=design_config,
        parameter_mask=parameter_mask,
        variable_names=variable_names,
        max_num_categories=num_categories,
        intercept_color=color["intercept"],
        main_effect_color=color["main_effect"],
        interaction_color=color["interaction"],
        prob=0.95,
        difference=True,
    )

    fig.savefig("test_adaptive_ecdf.pdf")
    print("Saved test_adaptive_ecdf.pdf")
