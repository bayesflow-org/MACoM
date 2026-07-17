import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from cogformer.simulator.design_utils import (
    build_parameter_mask, build_design_matrix, build_random_design_config,
    build_ensemble_row_labels,
)
from cogformer.diagnostics.plot.adaptive_ecdf import compute_ecdf_bands
from cogformer.utils.plot_utils import _add_mask_legend, _add_config_legend


def ensemble_ecdf(
    true_list: list[np.ndarray],
    pred_list: list[np.ndarray],
    design_configs: list[dict],
    intrinsic_params: list[str],
    max_num_categories: int,
    parameter_masks: list[np.ndarray] = None,
    variable_names: list[str] = None,
    colors: list[str] = None,
    n_colors: int = None,
    palette_lightness: float = 0.77,
    palette_start_hue: float = 0.05,
    labels: list[str] = None,
    figsize: tuple = None,
    title_fontsize: int = 24,
    label_fontsize: int = 12,
    tick_fontsize: int = 10,
    legend_fontsize: int = 17,
    thumb_scale: float = 1.0,
    prob: float = 0.95,
    difference: bool = True,
    band_alpha: float = 0.07,
    global_params: list[str] = None,
    global_only: bool = False,
    max_num_regressors: int = None,
    categorical_regressors_list: list = None,
) -> plt.Figure:
    """
    Overlay ECDF calibration plot for multiple design configs on a shared
    parameter matrix grid.

    A single confidence band (computed analytically from the Beta distribution)
    is drawn once per active cell and shared across all configs.

    Parameters
    ----------
    true_list : list of np.ndarray
        Ground-truth arrays (batch, num_rows_i, num_cols), one per config.
    pred_list : list of np.ndarray
        Posterior draw arrays (batch, draws, num_rows_i, num_cols), one per config.
    design_configs : list of dict
        Design configuration dicts (max 8).
    intrinsic_params : list of str
        Intrinsic parameter names (columns).
    max_num_categories : int
        Maximum number of categories used to size row blocks.
    parameter_masks : list of np.ndarray, optional
        Pre-built masks; constructed from design_configs if not provided.
    variable_names : list of str, optional
        Column header names.
    colors : list of str, optional
        One color per config. Defaults to seaborn husl palette of 8.
    labels : list of str, optional
        Legend labels, one per config.
    figsize : tuple, optional
    title_fontsize, label_fontsize, tick_fontsize, legend_fontsize : int
    prob : float
        Confidence level for the ECDF bands.
    difference : bool
        If True (default), plot ECDF − diagonal instead of raw ECDF.
    band_alpha : float
        Alpha for the shared confidence band fill (barely visible).
    """
    n_configs = len(design_configs)
    if n_configs > 12:
        raise ValueError("ensemble_ecdf supports at most 12 design configs.")
    if len(true_list) != n_configs or len(pred_list) != n_configs:
        raise ValueError("true_list, pred_list, and design_configs must have the same length.")

    if variable_names is None:
        variable_names = design_configs[0].get("1", intrinsic_params)

    if parameter_masks is None:
        parameter_masks = [
            build_parameter_mask(
                design_config=dc,
                max_num_categories=max_num_categories,
                intrinsic_params=intrinsic_params,
                keep_intercept=True
            )
            for dc in design_configs
        ]
    else:
        parameter_masks = [
            m[0] if m.ndim == 3 else m
            for m in parameter_masks
        ]

    if colors is None:
        colors = sns.husl_palette(n_configs if n_colors is None else n_colors, h=palette_start_hue, l=palette_lightness)

    if labels is None:
        labels = [f"Config {i + 1}" for i in range(n_configs)]

    # Grid dimensions: max num_rows across all masks, fixed num_cols. Pad every
    # mask to the shared row count so cell activity checks are uniform.
    num_rows = max(m.shape[0] for m in parameter_masks)
    num_cols = parameter_masks[0].shape[1]
    parameter_masks = [
        np.vstack([m, np.zeros((num_rows - m.shape[0], m.shape[1]), dtype=m.dtype)])
        if m.shape[0] < num_rows else m
        for m in parameter_masks
    ]

    # Global parameters (e.g. pi) are estimated jointly but presented separately.
    # Slice columns to exclude them from the grid, or keep only them when global_only.
    if global_params and intrinsic_params is not None:
        global_cols = {i for i, p in enumerate(intrinsic_params) if p in global_params}
        keep = [c for c in range(num_cols) if (c in global_cols) == global_only]
        if not keep:
            return None
        keep = np.array(keep, dtype=int)
        parameter_masks = [m[:, keep] for m in parameter_masks]
        true_list = [t[..., keep] for t in true_list]
        pred_list = [p[..., keep] for p in pred_list]
        variable_names = [variable_names[c] for c in keep]
        intrinsic_params = [intrinsic_params[c] for c in keep]
        num_cols = len(keep)

    # Union row labels across configs, aligned with the padded row layout.
    row_labels_all = build_ensemble_row_labels(
        design_configs, parameter_masks, max_num_categories,
        max_num_regressors=max_num_regressors,
        categorical_regressors_list=categorical_regressors_list,
    )

    # Compact rows: keep only rows active in at least one config.
    row_idx = np.where(np.any([np.any(m == 1.0, axis=1) for m in parameter_masks], axis=0))[0]
    if row_idx.size == 0:
        return None
    n_kept = row_idx.size

    # Shared confidence band — requires all configs to have the same batch size
    num_datasets = true_list[0].shape[0]
    band_x, band_lower, band_upper = compute_ecdf_bands(num_datasets=num_datasets, prob=prob)

    if figsize is None:
        figsize = (3 * num_cols, 3 * n_kept)

    fig, axarr = plt.subplots(n_kept, num_cols, figsize=figsize, squeeze=False)

    for r_i, r in enumerate(row_idx):
        ylabel = "" if global_only else row_labels_all[r]

        for c in range(num_cols):
            ax = axarr[r_i, c]

            active_configs = [
                k for k in range(n_configs)
                if parameter_masks[k][r, c] == 1.0
            ]

            if not active_configs:
                ax.set_facecolor("gray")
                ax.patch.set_alpha(0.05)
                if r_i == n_kept - 1:
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
                    alpha=0.7, color="gray"
                )
                ax.set_xlabel("Fractional rank statistic" if r_i == n_kept - 1 else "", fontsize=label_fontsize)
                continue

            # Shared confidence band drawn once per active cell
            if difference:
                ax.fill_between(
                    band_x, band_lower - band_x, band_upper - band_x,
                    color="black", alpha=band_alpha,
                )
                ax.axhline(0.0, color="black", linestyle="dashed", linewidth=1.0, alpha=0.7)
            else:
                ax.fill_between(
                    band_x, band_lower, band_upper,
                    color="black", alpha=band_alpha,
                )
                ax.plot([0, 1], [0, 1], color="black", linestyle="dashed", linewidth=1.0, alpha=0.7)

            for k in active_configs:
                ranks = np.mean(
                    pred_list[k][:, :, r, c] < true_list[k][:, r, c, np.newaxis],
                    axis=1
                )  # (batch,)
                ranks_sorted = np.sort(ranks)
                ecdf_y = np.arange(1, num_datasets + 1) / num_datasets

                if difference:
                    ax.plot(ranks_sorted, ecdf_y - ranks_sorted, color=colors[k], alpha=1.0)
                else:
                    ax.plot(ranks_sorted, ecdf_y, color=colors[k], alpha=1.0)

            ax.set_xlim(-0.02, 1.02)
            ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
            if not difference:
                ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.tick_params(axis="both", labelsize=tick_fontsize)
            ax.set_box_aspect(1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, color="lightgray", linestyle="--", linewidth=0.5, alpha=0.35)

            ax.set_ylabel(ylabel if c == 0 else "", fontsize=label_fontsize)
            ax.set_xlabel("Fractional rank statistic" if r_i == n_kept - 1 else "", fontsize=label_fontsize)
            if r_i == 0 and variable_names is not None:
                ax.set_title(variable_names[c], fontsize=title_fontsize)

    # Compute per-cell ECE → per-cell alpha map in [0.5, 1.0]
    # ECE is bounded by ~0.25, so we rescale: alpha = clip(1 - 4*ece, 0.5, 1.0)
    pixel_alphas = []
    ecdf_y = np.arange(1, num_datasets + 1) / num_datasets
    for k in range(n_configs):
        alpha_map = np.full((n_kept, num_cols), 0.5)
        for r_i, r in enumerate(row_idx):
            for c in range(num_cols):
                if parameter_masks[k][r, c] != 1.0:
                    continue
                ranks = np.mean(
                    pred_list[k][:, :, r, c] < true_list[k][:, r, c, np.newaxis],
                    axis=1,
                )
                ranks_sorted = np.sort(ranks)
                ece = float(np.mean(np.abs(ecdf_y - ranks_sorted)))
                alpha_map[r_i, c] = float(np.clip(1.0 - 4.0 * ece, 0.2, 1.0))
        pixel_alphas.append(alpha_map)

    if global_only:
        # No coefficient-matrix structure to show — plain color legend instead.
        _add_config_legend(fig, colors, labels, fontsize=legend_fontsize)
        strip_w_fig = 0.0
    else:
        compact_masks = [m[row_idx] for m in parameter_masks]
        strip_w_fig = _add_mask_legend(fig, compact_masks, colors, labels, n_kept, num_cols,
                                       label_fontsize=legend_fontsize, pixel_alphas=pixel_alphas,
                                       thumb_scale=thumb_scale)

    # Reserve space at the bottom of the main plot area for the band/ECDF legend
    legend_gap = 0.07
    fig.subplots_adjust(bottom=legend_gap + 0.04)

    band_handle = mpatches.Patch(facecolor="black", alpha=band_alpha * 6, label=f"{int(prob * 100)}% confidence band")
    ecdf_handle = mlines.Line2D([], [], color="black", alpha=0.7, label="Rank ECDF")
    fig.legend(
        handles=[band_handle, ecdf_handle],
        loc="lower center",
        bbox_to_anchor=((1.0 + strip_w_fig) / 2, 0.005),
        ncol=2,
        fontsize=legend_fontsize + 2,
        frameon=True,
    )
    return fig


def global_ensemble_ecdf(
    true_list: list[np.ndarray],
    pred_list: list[np.ndarray],
    design_configs: list[dict],
    intrinsic_params: list[str],
    max_num_categories: int,
    global_params: list[str],
    parameter_masks: list[np.ndarray] = None,
    variable_names: list[str] = None,
    colors: list[str] = None,
    n_colors: int = None,
    palette_lightness: float = 0.77,
    palette_start_hue: float = 0.05,
    labels: list[str] = None,
    title_fontsize: int = 18,
    label_fontsize: int = 14,
    tick_fontsize: int = 10,
    figsize: tuple = None,
    prob: float = 0.95,
    difference: bool = True,
    band_alpha: float = 0.07,
    max_per_row: int = 6,
    **kwargs,
):
    """Ensemble ECDF calibration for global parameters (e.g. ``pi``), plotted
    separately from the coefficient matrix: ONE PANEL PER CONFIG, arranged in
    rows of at most ``max_per_row`` panels (e.g. 5 configs -> one row of 5;
    12 configs -> two rows of 6), each panel in its config's color with the
    config label as title. Returns ``None`` when no global parameter is active.
    """
    n_configs = len(design_configs)
    if variable_names is None:
        variable_names = intrinsic_params
    if parameter_masks is None:
        parameter_masks = [
            build_parameter_mask(design_config=dc, max_num_categories=max_num_categories,
                                 intrinsic_params=intrinsic_params, keep_intercept=True)
            for dc in design_configs
        ]
    else:
        parameter_masks = [m[0] if m.ndim == 3 else m for m in parameter_masks]
    if colors is None:
        colors = sns.husl_palette(n_configs if n_colors is None else n_colors,
                                  h=palette_start_hue, l=palette_lightness)
    if labels is None:
        labels = [f"Config {i + 1}" for i in range(n_configs)]

    global_cols = [i for i, p in enumerate(intrinsic_params) if p in global_params]
    active_cols = [c for c in global_cols
                   if any(np.any(m[:, c] == 1.0) for m in parameter_masks)]
    if not active_cols:
        return None

    # Wrap config panels at max_per_row per row (e.g. 12 configs -> 2 rows of 6).
    n_cols_grid = min(n_configs, max_per_row)
    rows_per_param = int(np.ceil(n_configs / n_cols_grid))
    n_rows = len(active_cols) * rows_per_param
    if figsize is None:
        figsize = (3 * n_cols_grid, 3.2 * n_rows)
    fig, axarr = plt.subplots(n_rows, n_cols_grid, figsize=figsize, squeeze=False)

    for gi, c in enumerate(active_cols):
        for k in range(n_configs):
            ax = axarr[gi * rows_per_param + k // n_cols_grid, k % n_cols_grid]
            is_left = k % n_cols_grid == 0
            is_bottom = (gi == len(active_cols) - 1) and (k + n_cols_grid >= n_configs)
            rows_k = np.where(parameter_masks[k][:, c] == 1.0)[0]
            if rows_k.size == 0:
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", va="center",
                        fontsize=18, weight="bold", alpha=0.7, color="gray")
                ax.set_title(labels[k], fontsize=title_fontsize)
                continue
            r = int(rows_k[0])  # global params live on the intercept row

            num_datasets = true_list[k].shape[0]
            band_x, band_lower, band_upper = compute_ecdf_bands(num_datasets=num_datasets, prob=prob)
            ranks = np.mean(pred_list[k][:, :, r, c] < true_list[k][:, r, c, np.newaxis], axis=1)
            ranks_sorted = np.sort(ranks)
            ecdf_y = np.arange(1, num_datasets + 1) / num_datasets

            if difference:
                ax.fill_between(band_x, band_lower - band_x, band_upper - band_x,
                                color="black", alpha=band_alpha)
                ax.axhline(0.0, color="black", linestyle="dashed", linewidth=1.0, alpha=0.7)
                ax.plot(ranks_sorted, ecdf_y - ranks_sorted, color=colors[k], alpha=1.0)
            else:
                ax.fill_between(band_x, band_lower, band_upper, color="black", alpha=band_alpha)
                ax.plot([0, 1], [0, 1], color="black", linestyle="dashed", linewidth=1.0, alpha=0.7)
                ax.plot(ranks_sorted, ecdf_y, color=colors[k], alpha=1.0)
                ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

            ax.set_xlim(-0.02, 1.02)
            ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.tick_params(axis="both", labelsize=tick_fontsize)
            ax.set_box_aspect(1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, color="lightgray", linestyle="--", linewidth=0.5, alpha=0.35)
            ax.set_title(labels[k], fontsize=title_fontsize)
            ax.set_ylabel(variable_names[c] if is_left else "", fontsize=title_fontsize)
            ax.set_xlabel("Fractional rank statistic" if is_bottom else "",
                          fontsize=label_fontsize)

    # Hide unused trailing cells in each param block
    for gi in range(len(active_cols)):
        for k in range(n_configs, rows_per_param * n_cols_grid):
            axarr[gi * rows_per_param + k // n_cols_grid, k % n_cols_grid].axis("off")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    intrinsic_params = ["v", "a", "z", "tau", "p1", "p2", "p3", "p4"]
    variable_names = [r"$v$", r"$a$", r"$z$", r"$\tau$", r"$p_1$", r"$p_2$", r"$p_3$", r"$p_4$"]
    max_num_categories = 2
    batch_size = 200
    draws = 256

    design_configs = [
        build_random_design_config(
            intrinsic_params=intrinsic_params,
            num_regressors=2,
            keep_intercept=True,
            add_interaction=True,
        )
        for _ in range(12)
    ]
    labels = [f"Config {i + 1}" for i in range(12)]

    def make_data(dc, batch, draws, max_cats, miscalibrated=False):
        mask = build_parameter_mask(
            design_config=dc,
            max_num_categories=max_cats,
            intrinsic_params=intrinsic_params,
            keep_intercept=True
        )
        nr, nc = mask.shape
        true = np.random.normal(0, 1, (batch, nr, nc))
        pred = np.random.normal(0, 1, (batch, draws, nr, nc))
        if miscalibrated:
            # Apply overestimation bias to a random subset of cells
            rng = np.random.default_rng()
            bias = rng.uniform(0.0, 2.0, (nr, nc))  # per-cell bias magnitude
            pred = pred + bias[np.newaxis, np.newaxis, :, :]
        return true, pred

    data = [
        make_data(dc, batch_size, draws, max_num_categories, miscalibrated=(i >= 4))
        for i, dc in enumerate(design_configs)
    ]
    true_list, pred_list = zip(*data)

    fig = ensemble_ecdf(
        true_list=list(true_list),
        pred_list=list(pred_list),
        design_configs=design_configs,
        intrinsic_params=intrinsic_params,
        max_num_categories=max_num_categories,
        variable_names=variable_names,
        labels=labels,
    )
    fig.savefig("test_ensemble_ecdf.pdf")
    print("done")
