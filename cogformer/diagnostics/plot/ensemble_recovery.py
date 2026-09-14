import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from collections.abc import Callable

from cogformer.simulator.design_utils import (
    build_parameter_mask, build_design_matrix, build_random_design_config,
    build_ensemble_row_labels,
)
from cogformer.utils.plot_utils import (credible_interval, _add_mask_legend,
                                        _add_config_legend, fit_text)


def ensemble_recovery(
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
    uncertainty_agg: Callable = credible_interval,
    title_fontsize: int = 24,
    label_fontsize: int = 12,
    legend_fontsize: int = 17,
    row_labels: list = None,
    annot_fontsize: float = 12,
    shared_xlabel: bool = False,
    max_ticks: int = None,
    fit_labels: bool = False,
    fit_delta: float = 0.0,
    tick_pad: float = None,
    thumb_spread: bool = False,
    pack_rows: bool = False,
    row_gap_px: float = 6.0,
    pack_cols: bool = False,
    col_gap_px: float = 10.0,
    thumb_tight_rows: bool = False,
    marker_size: float = None,
    err_linewidth: float = 1.2,
    wspace: float = None,
    hspace: float = None,
    thumb_scale: float = 1.0,
    thumb_per_row: int | None = None,
    alpha: float = 0.5,
    figsize: tuple = None,
    global_params: list[str] = None,
    global_only: bool = False,
    max_num_regressors: int = None,
    categorical_regressors_list: list = None,
):
    """
    Overlay recovery plot for multiple design configs on a shared parameter matrix grid.

    Rows with no active cell in any config are dropped, so padded layouts stay
    compact. ``max_num_regressors`` is the main-block count the mask layout was
    padded to; ``categorical_regressors_list`` optionally gives, per config, the
    contrast-coded regressor keys for row labeling (mask-inferred when omitted).

    Parameters
    ----------
    true_list : list of np.ndarray
        Ground-truth arrays, one per design config. Each has shape
        (batch, num_rows_i, num_cols).
    pred_list : list of np.ndarray
        Posterior predictive arrays, one per design config. Each has shape
        (batch, draws, num_rows_i, num_cols) for full posteriors or
        (batch, num_rows_i, num_cols) for point estimates.
    design_configs : list of dict
        Design configuration dicts, one per entry (max 10).
    intrinsic_params : list of str
        Names of the intrinsic (column) parameters.
    max_num_categories : int
        Maximum number of categories used to size the row blocks.
    parameter_masks : list of np.ndarray, optional
        Pre-built parameter masks (one per design config). Built from
        design_configs if not provided.
    variable_names : list of str, optional
        Column header names; defaults to intrinsic_params.
    colors : list of str, optional
        One color per design config. Defaults to tab10 colormap.
    labels : list of str, optional
        Legend labels, one per design config.
    uncertainty_agg : Callable
        Function to compute credible intervals; signature (y, prob, axis).
    title_fontsize, label_fontsize : int
    alpha : float
        Scatter/errorbar transparency for overlaid configs.
    figsize : tuple, optional
    """
    n_configs = len(design_configs)
    if n_configs > 12:
        raise ValueError("ensemble_recovery supports at most 12 design configs.")
    if len(true_list) != n_configs or len(pred_list) != n_configs:
        raise ValueError("true_list, pred_list, and design_configs must have the same length.")

    if variable_names is None:
        variable_names = design_configs[0]["1"] if "1" in design_configs[0] else intrinsic_params

    # Build masks if not provided
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

    # Default colors from husl palette
    if colors is None:
        colors = sns.husl_palette(n_configs if n_colors is None else n_colors, h=palette_start_hue, l=palette_lightness)

    if labels is None:
        labels = [f"Config {i + 1}" for i in range(n_configs)]

    if figsize is None:
        figsize = (3 * num_cols, 3 * n_kept)

    fig, axarr = plt.subplots(n_kept, num_cols, figsize=figsize, squeeze=False)

    # Per-config r-value maps (compact-row space) for mask legend pixel alphas
    r_maps = [np.zeros((n_kept, num_cols)) for _ in range(n_configs)]

    # An explicit override replaces the design-cell names ("1", "u_1(:c_1)") with
    # whatever the caller wants down the left edge. Indexed in COMPACT row order --
    # the order the rows are drawn in -- because the caller counts the rows it sees,
    # not the padded grid rows behind them.
    if row_labels is not None and len(row_labels) != n_kept:
        raise ValueError(f"row_labels has {len(row_labels)} entries but {n_kept} rows "
                         "are drawn")

    for r_i, r in enumerate(row_idx):
        if row_labels is not None:
            ylabel = row_labels[r_i]
        else:
            ylabel = "" if global_only else row_labels_all[r]

        for c in range(num_cols):
            ax = axarr[r_i, c]

            # Collect data from all configs that are active at (r, c)
            active_entries = []
            for k in range(n_configs):
                mask_k = parameter_masks[k]
                if mask_k[r, c] != 1.0:
                    continue

                true_k = true_list[k][..., r, c]
                pred_k = pred_list[k]

                if pred_k.ndim == 4:
                    y_k = pred_k[..., r, c]
                    y_mean_k = y_k.mean(axis=1)
                    ci_k = uncertainty_agg(y_k, prob=0.9, axis=1)
                    y_lo_k, y_hi_k = ci_k[0], ci_k[1]
                    y_err_k = np.vstack([y_mean_k - y_lo_k, y_hi_k - y_mean_k])
                    active_entries.append((k, true_k, y_mean_k, y_err_k, colors[k], labels[k]))
                else:
                    y_mean_k = pred_k[..., r, c]
                    active_entries.append((k, true_k, y_mean_k, None, colors[k], labels[k]))

            if not active_entries:
                # N/A cell — no config covers this cell
                ax.set_facecolor("gray")
                ax.patch.set_alpha(0.05)
                if r_i == n_kept - 1:
                    ax.set_xticks([0.0, 0.5, 1.0])
                    ax.set_xticklabels(["0.0", "0.5", "1.0"])
                    ax.tick_params(axis="x", length=0, labelcolor="none")
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
                ax.set_xlabel("Ground Truth" if r_i == n_kept - 1 else "", fontsize=label_fontsize)
                continue

            # Compute global limits across all active configs for this cell
            all_x = np.concatenate([e[1] for e in active_entries])
            all_y = np.concatenate([e[2] for e in active_entries])
            lower = min(all_x.min(), all_y.min())
            upper = max(all_x.max(), all_y.max())
            span = upper - lower if upper != lower else 1.0
            eps = span * 0.1
            xlim = (lower - eps, upper + eps)
            ylim = (lower - eps, upper + eps)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.plot(xlim, ylim, color="black", alpha=0.4, linestyle="dashed", linewidth=1.0)

            # Per-config correlations — used for annotation and mask legend pixel alphas
            corrs = [np.corrcoef(e[1], e[2])[0, 1] for e in active_entries]
            mean_corr = float(np.mean(corrs))
            cell_alpha = 1.0 if n_configs <= 2 else 2.0 / n_configs

            for idx, (cfg_k, x_k, y_mean_k, y_err_k, color_k, label_k) in enumerate(active_entries):
                r_maps[cfg_k][r_i, c] = corrs[idx]
                if y_err_k is not None:
                    ax.errorbar(
                        x_k, y_mean_k, yerr=y_err_k,
                        fmt="none", alpha=cell_alpha * 0.6,
                        linewidth=err_linewidth, color=color_k
                    )
                sns.scatterplot(
                    x=x_k, y=y_mean_k, ax=ax,
                    color=color_k, alpha=cell_alpha,
                    **({"s": marker_size} if marker_size else {}),
                )
            ax.text(
                0.07, 0.95, f"r = {mean_corr:.3f}",
                ha="left", va="center",
                transform=ax.transAxes, size=annot_fontsize
            )

            ax.set_box_aspect(1)
            ax.grid(True, color="lightgray", linestyle="--", linewidth=0.5, alpha=0.2)
            sns.despine(ax=ax)
            if tick_pad is not None:
                # The blank band between panels is mostly this: matplotlib pads tick
                # labels 3.5 pt off the axis by default, which at these sizes is most
                # of the gap. Closing it hands the space back to the plot area.
                ax.tick_params(pad=tick_pad)
            if max_ticks is not None:
                # A default locator puts five labels under a panel an inch wide and
                # they run into the panel beside them.
                ax.xaxis.set_major_locator(MaxNLocator(max_ticks))
                ax.yaxis.set_major_locator(MaxNLocator(max_ticks))
            ax.set_ylabel(ylabel if c == 0 else "", fontsize=label_fontsize)
            ax.set_title(variable_names[c] if r_i == 0 else "", fontsize=title_fontsize)
            if not shared_xlabel:
                ax.set_xlabel("Ground Truth" if r_i == n_kept - 1 else "",
                              fontsize=label_fontsize)

    fitted_label_size = None
    grid_axes = {id(ax) for ax in np.ravel(axarr)}
    if fit_labels:
        # Layout only; the labels are fitted after the legend claims its strip.
        _tighten_grid(fig, wspace, hspace)
    if False:
        # THE PANELS ARE THE FIGURE. Labels are sized to the cell they belong to --
        # x labels to a cell's width, row labels to its height -- rather than to an
        # absolute point size, so they fill their cell exactly and never spill into
        # the neighbouring one. Done after the spacing below, since the cell size is
        # what is being measured.
        fitted_label_size = _fit_grid_labels(fig, axarr, row_idx.size, num_cols,
                                             wspace, hspace, fit_delta)

    if shared_xlabel:
        # ONE x label for the grid. Every column shares the same x quantity, and at a
        # page-sized figure "Ground Truth" is wider than a panel, so per-panel labels
        # simply overprint each other.
        fig.supxlabel("Ground Truth", fontsize=label_fontsize)

    if global_only:
        # No coefficient-matrix structure to show — plain color legend instead.
        _add_config_legend(fig, colors, labels, fontsize=legend_fontsize)
    else:
        pixel_alpha_maps = [np.clip(np.abs(r_maps[k]), 0.2, 1.0) for k in range(n_configs)]
        compact_masks = [m[row_idx] for m in parameter_masks]
        _add_mask_legend(fig, compact_masks, colors, labels, n_kept, num_cols,
                         label_fontsize=fitted_label_size or legend_fontsize,
                         pixel_alphas=pixel_alpha_maps,
                         thumb_scale=thumb_scale, per_row=thumb_per_row,
                         fit_labels=False, spread=thumb_spread,
                         tight_rows=thumb_tight_rows)

    if fit_labels:
        # AFTER the legend. _add_mask_legend re-runs tight_layout to reserve the
        # thumbnail strip, which shrinks the panels; fitting before that sizes the
        # labels against panels that are about to get smaller and they overrun each
        # other. Re-apply the gutters first -- the legend's tight_layout resets them.
        adj = {k: v for k, v in (("wspace", wspace), ("hspace", hspace))
               if v is not None}
        if adj:
            fig.subplots_adjust(**adj)
        if pack_rows:
            _pack_rows(fig, axarr, row_idx.size, num_cols, row_gap_px)
        if pack_cols:
            _pack_cols(fig, axarr, row_idx.size, num_cols, col_gap_px)
        fitted_label_size = _fit_grid_labels(fig, axarr, row_idx.size, num_cols,
                                             wspace, hspace, fit_delta)
        if fitted_label_size:
            # The thumbnail captions take the same size, for the same reason the row
            # labels do: they are all doing one job and should read as one size.
            for ax in fig.axes:
                if id(ax) not in grid_axes and ax.get_xlabel():
                    ax.xaxis.label.set_fontsize(fitted_label_size)
    return fig


def _pack_rows(fig, axarr, n_rows, n_cols, gap_px=6.0):
    """Stack the rows at a fixed gap, overriding tight_layout's uniform allocation.

    tight_layout gives every row the same height, and that height must accommodate
    the tallest decoration in the grid -- row 0's column titles. Rows 1..n have no
    title, so they carry that allowance as dead space above them, and it stays the
    same ~38 px however wide or tall the figure is made. Nothing in wspace, hspace
    or the figure aspect touches it, because it is an ALLOCATION, not slack.

    So the rows are placed by hand: measure row 0's box and how far its decorations
    reach below it, then stack the rest at exactly that pitch plus `gap_px`.
    """
    if n_rows < 2:
        return
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    p0 = axarr[0, 0].get_position()
    tb = axarr[0, 0].get_tightbbox(rend).transformed(inv)
    below = max(0.0, p0.y0 - tb.y0)          # x tick labels under the axes
    gap = gap_px / (fig.get_size_inches()[1] * fig.dpi)
    pitch = p0.height + below + gap
    for r in range(1, n_rows):
        for c in range(n_cols):
            pos = axarr[r, c].get_position()
            axarr[r, c].set_position([pos.x0, p0.y0 - r * pitch,
                                      pos.width, p0.height])


def _pack_cols(fig, axarr, n_rows, n_cols, gap_px=10.0):
    """Stack the columns at a fixed blank gap, the mirror of _pack_rows.

    wspace cannot express this: it is a fraction of the cell width, and the cell
    also holds the y tick labels, so the BLANK part of the gutter is whatever is
    left over after they are drawn. Measuring how far column 0's decorations reach
    to its left gives the pitch directly -- axes width + that reach + `gap_px`.
    """
    if n_cols < 2:
        return
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    p0 = axarr[0, 0].get_position()
    gap = gap_px / (fig.get_size_inches()[0] * fig.dpi)

    def left_reach(c):
        """How far column c's decorations stick out to the LEFT of its axes.

        Per column, not a single figure for all of them: the y tick labels differ
        in width from column to column ("-2.5" against "0"), so one pitch derived
        from column 0 leaves some columns crowded and others far apart.
        """
        out = 0.0
        for r in range(n_rows):
            ax = axarr[r, c]
            if not ax.get_visible():
                continue
            tb = ax.get_tightbbox(rend).transformed(inv)
            out = max(out, max(0.0, ax.get_position().x0 - tb.x0))
        return out

    x = p0.x0
    for c in range(n_cols):
        if c:
            x += p0.width + left_reach(c) + gap
        for r in range(n_rows):
            pos = axarr[r, c].get_position()
            axarr[r, c].set_position([x, pos.y0, p0.width, pos.height])


def _tighten_grid(fig, wspace, hspace):
    """Lay the grid out and pull the gutters in. Separate from the fitting step
    because the two have to happen at different points: the layout before the
    thumbnail strip is reserved, the fitting after it, once the panels are final."""
    fig.tight_layout()
    adj = {k: v for k, v in (("wspace", wspace), ("hspace", hspace)) if v is not None}
    if adj:
        fig.subplots_adjust(**adj)


def _fit_grid_labels(fig, axarr, n_rows, n_cols, wspace, hspace, fit_delta=0.0):
    """Tighten the grid, then size the axis labels to the cell they sit in.

    ONE size for both axes, set by the x label against the cell WIDTH, and returned
    so the thumbnail captions can take it too. Fitting each label to its own axis
    independently -- x to the width, y to the height -- is defensible but reads
    badly: the row labels come out visibly larger than the column labels beside
    them, when the two are doing the same job.
    NO LAYOUT HERE. Calling tight_layout at this point would discard the rect the
    thumbnail strip was reserved with and put the panels back on top of it.
    """
    fig.canvas.draw()

    figW, figH = fig.get_size_inches()
    # The PLOT AREA, not the grid cell. The axes are square by box_aspect, so they do
    # not fill their cell -- the cell also carries the tick labels and the gutter --
    # and a label sized to the cell is wider than the panel it belongs to.
    bb = axarr[0, 0].get_window_extent(renderer=fig.canvas.get_renderer())
    cell_w_in, cell_h_in = bb.width / fig.dpi, bb.height / fig.dpi

    xlabels, ylabels = [], []
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axarr[r, c]
            if ax.get_xlabel():
                xlabels.append(ax.xaxis.label)
            if ax.get_ylabel():
                ylabels.append(ax.yaxis.label)
    fit_text(fig, xlabels, cell_w_in, axis="x")
    size = xlabels[0].get_fontsize() if xlabels else None
    if size is not None and fit_delta:
        # An explicit trim on top of the fit. The fit gives the largest size that
        # fits the panel; this is the caller saying it wants a little less than that.
        size = max(1.0, size + fit_delta)
        for art in xlabels:
            art.set_fontsize(size)
    if size is not None:
        # The y labels TAKE the x size rather than being fitted to the cell height.
        # They are a little longer than "Ground Truth", so they end up a little
        # longer than a cell -- still inside the gap, and matching matters more.
        for art in ylabels:
            art.set_fontsize(size)
    else:
        fit_text(fig, ylabels, cell_h_in, axis="y")
    return size


def global_ensemble_recovery(
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
    uncertainty_agg: Callable = credible_interval,
    title_fontsize: int = 18,
    label_fontsize: int = 14,
    figsize: tuple = None,
    share_limits: bool = False,
    annot_fontsize: float = 12,
    wspace: float = None,
    hspace: float = None,
    equal_gaps: bool = False,
    alpha: float = 0.7,
    max_per_row: int = 6,
    **kwargs,
):
    """Ensemble recovery for global parameters (e.g. ``pi``), plotted separately
    from the coefficient matrix: ONE PANEL PER CONFIG, arranged in rows of at
    most ``max_per_row`` panels (e.g. 5 configs -> one row of 5; 12 configs ->
    two rows of 6), each panel in its config's color with the config label as
    title. Returns ``None`` when no global parameter is active.
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

    # ONE scale across every panel, ticks labelled on the first column only. These
    # panels all show the same parameter, so per-panel limits make a config whose pi
    # happens to run to 0.08 look identical to one running to 0.15 -- the comparison
    # the grid exists for is then invisible. Computed up front, over every config's
    # truths and predictions together.
    shared_lims = None
    if share_limits:
        vals = []
        for k in range(n_configs):
            for c in active_cols:
                rows_k = np.where(parameter_masks[k][:, c] == 1.0)[0]
                if rows_k.size == 0:
                    continue
                r = int(rows_k[0])
                vals.append(np.asarray(true_list[k][:, r, c]).ravel())
                pk = pred_list[k]
                vals.append(np.asarray(
                    pk[:, :, r, c].mean(axis=1) if pk.ndim == 4 else pk[:, r, c]).ravel())
        if vals:
            allv = np.concatenate(vals)
            lo, hi = float(allv.min()), float(allv.max())
            span = hi - lo if hi != lo else 1.0
            shared_lims = (lo - span * 0.1, hi + span * 0.1)

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

            x = true_list[k][:, r, c]
            pred_k = pred_list[k]
            if pred_k.ndim == 4:
                y = pred_k[:, :, r, c]
                y_mean = y.mean(axis=1)
                ci = uncertainty_agg(y, prob=0.9, axis=1)
                y_err = np.vstack([y_mean - ci[0], ci[1] - y_mean])
                ax.errorbar(x, y_mean, yerr=y_err, fmt="none", alpha=0.3,
                            linewidth=1.2, color=colors[k])
            else:
                y_mean = pred_k[:, r, c]

            if shared_lims is not None:
                lims = shared_lims
            else:
                lower = min(x.min(), y_mean.min())
                upper = max(x.max(), y_mean.max())
                span = upper - lower if upper != lower else 1.0
                eps = span * 0.1
                lims = (lower - eps, upper + eps)
            ax.set_xlim(lims); ax.set_ylim(lims)
            ax.plot(lims, lims, color="black", alpha=0.4, linestyle="dashed", linewidth=1.0)

            sns.scatterplot(x=x, y=y_mean, ax=ax, color=colors[k], alpha=alpha)
            corr = np.corrcoef(x, y_mean)[0, 1]
            # Inset from the spine: at 0.02 the "r" sits on the y axis line.
            ax.text(0.07, 0.95, f"r = {corr:.3f}", ha="left", va="center",
                    transform=ax.transAxes, size=annot_fontsize)

            ax.set_box_aspect(1)
            ax.grid(True, color="lightgray", linestyle="--", linewidth=0.5, alpha=0.2)
            sns.despine(ax=ax)
            ax.set_title(labels[k], fontsize=title_fontsize)
            ax.set_ylabel(variable_names[c] if is_left else "", fontsize=title_fontsize)
            if shared_lims is not None and not is_left:
                ax.tick_params(labelleft=False)
            ax.set_xlabel("Ground Truth" if is_bottom else "", fontsize=label_fontsize)

    # Hide unused trailing cells in each param block
    for gi in range(len(active_cols)):
        for k in range(n_configs, rows_per_param * n_cols_grid):
            axarr[gi * rows_per_param + k // n_cols_grid, k % n_cols_grid].axis("off")

    fig.tight_layout()
    # After tight_layout, which computes its own and would discard anything set up
    # front. The panels are square, so a wide gutter is pure dead space between them.
    adj = {k: v for k, v in (("wspace", wspace), ("hspace", hspace)) if v is not None}
    if adj:
        fig.subplots_adjust(**adj)

    if equal_gaps and n_rows > 1 and n_cols_grid > 1:
        # MATCH THE COLUMN GAP TO THE ROW GAP, measured rather than guessed.
        #
        # MEASURED BETWEEN INK, NOT BETWEEN AXES BOXES. The row gap is full of tick
        # labels and titles while the column gap is empty, so matching the axes-box
        # gaps leaves the columns looking far further apart than the rows even though
        # the numbers agree. tightbbox includes each panel's decorations, which is
        # what the eye is actually judging.
        #
        # wspace alone cannot close it either: the axes are square by box_aspect and
        # HEIGHT-limited here, so their width is fixed by the row height and the
        # leftover cell width shows up as gap even at wspace=0. The excess comes out
        # of the FIGURE WIDTH, the only thing that narrows the cells.
        fig.subplots_adjust(wspace=0.0)
        inv = fig.transFigure.inverted()
        for _ in range(6):
            fig.canvas.draw()
            rend = fig.canvas.get_renderer()
            t00 = axarr[0, 0].get_tightbbox(rend)
            t01 = axarr[0, 1].get_tightbbox(rend)
            t10 = axarr[1, 0].get_tightbbox(rend)
            gap_x, gap_y = t01.x0 - t00.x1, t00.y0 - t10.y1
            if gap_x <= 0 or abs(gap_x - gap_y) < 2.0:
                break
            shrink_px = (gap_x - gap_y) * (n_cols_grid - 1)
            w_in, h_in = fig.get_size_inches()
            new_w = max(1.0, w_in - shrink_px / fig.dpi)
            if abs(new_w - w_in) < 0.01:
                break
            fig.set_size_inches(new_w, h_in)
    return fig


if __name__ == "__main__":
    intrinsic_params = ["v", "a", "z", "tau", "p1", "p2", "p3", "p4"]
    variable_names = [r"$v$", r"$a$", r"$z$", r"$\tau$", r"$p_1$", r"$p_2$", r"$p_3$", r"$p_4$"]
    max_num_categories = 2
    batch_size = 20
    draws = 50

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

    def make_data(dc, batch, draws, max_cats, poor_recovery=False):
        mask = build_parameter_mask(
            design_config=dc,
            max_num_categories=max_cats,
            intrinsic_params=intrinsic_params,
            keep_intercept=True
        )
        nr, nc = mask.shape
        true = np.random.normal(0, 1, (batch, nr, nc))
        if poor_recovery:
            # Draws unrelated to true → low correlation
            pred = np.random.normal(0, 1, (batch, draws, nr, nc))
        else:
            # Good recovery: draws centered tightly on true
            pred = true[:, None, :, :] + np.random.normal(0, 0.15, (batch, draws, nr, nc))
        return true, pred

    data = [
        make_data(dc, batch_size, draws, max_num_categories, poor_recovery=(i >= 6))
        for i, dc in enumerate(design_configs)
    ]
    true_list, pred_list = zip(*data)

    fig = ensemble_recovery(
        true_list=list(true_list),
        pred_list=list(pred_list),
        design_configs=design_configs,
        intrinsic_params=intrinsic_params,
        max_num_categories=max_num_categories,
        variable_names=variable_names,
        labels=labels,
    )
    fig.savefig("test_ensemble_recovery.pdf")
    print("done")
