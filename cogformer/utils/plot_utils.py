import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D


def make_quadratic(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str = "black"):
    """
    Utility to make subplots quadratic to avoid visual illusions
    in, e.g., recovery plot.
    """

    lower = min(x.min(), y.min())
    upper = max(x.max(), y.max())

    span = upper - lower
    if span == 0:
        span = 1.0
    eps = span * 0.1

    ax.set_xlim((lower - eps, upper + eps))
    ax.set_ylim((lower - eps, upper + eps))

    ax.plot(
        [ax.get_xlim()[0], ax.get_xlim()[1]],
        [ax.get_ylim()[0], ax.get_ylim()[1]],
        color=color,
        alpha=0.7,
        linestyle="dashed",
    )

def credible_interval(x: np.ndarray, prob: float = 0.95, axis: int = None, **kwargs) -> np.ndarray:
    """
    Compute credible interval from samples using quantiles.

    Parameters
    ----------
    x : array_like
        Input array of samples from a posterior distribution or bootstrap samples.
    prob : float, default 0.95
        Coverage probability of the credible interval (between 0 and 1).
        For example, 0.95 gives a 95% credible interval.
    axis : Sequence[int]
        Axis or axes along which the credible interval is computed.
        Default is None (flatten array).

    Returns
    -------
    a numpy array of shape (2, ...) with the first dimension indicating the
    lower and upper bounds of the credible interval.

    Examples
    --------
    >>> import numpy as np
    >>> # Simulate posterior samples
    >>> samples = np.random.normal(size=(10, 1000, 3))

    >>> # Different coverage probabilities
    >>> credible_interval(samples, prob=0.5, axis=1)  # 50% CI
    >>> credible_interval(samples, prob=0.99, axis=1)  # 99% CI
    """

    # Input validation
    if not 0 <= prob <= 1:
        raise ValueError(f"prob must be between 0 and 1, got {prob}")

    # Calculate tail probabilities
    alpha = 1 - prob
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2

    # Compute quantiles
    return np.quantile(x, q=(lower_q, upper_q), axis=axis, **kwargs)

def _add_mask_legend(
    fig: plt.Figure,
    parameter_masks: list,
    colors: list,
    labels: list,
    num_rows: int,
    num_cols: int,
    label_fontsize: int = 11,
    pad: float = 0.012,
    gap: float = 0.018,
    pixel_alphas: list[float] | None = None,
    thumb_scale: float = 1.0,
    max_per_column: int = 6,
    per_row: int | None = None,
):
    """
    Add parameter mask thumbnails as a visual legend in a strip on the LEFT
    side of the figure, stacked top-down in one or more columns and separated
    from the main axes by a vertical divider line. Active cells are drawn in
    the config's color; inactive cells in light gray. Pixels are forced square.
    Config labels appear below each thumbnail.

    Parameters
    ----------
    fig : plt.Figure
    parameter_masks : list of np.ndarray
        One binary mask per config, shape (num_rows_k, num_cols).
    colors : list
        One matplotlib color per config.
    labels : list of str
        One label per config, displayed below the thumbnail.
    num_rows : int
        Global max num_rows (used to pad shorter masks).
    num_cols : int
        Number of columns (intrinsic parameters).
    label_fontsize : int
    pad : float
        Padding around thumbnails, in figure coords.
    gap : float
        Horizontal gap between legend columns, in figure coords.
    max_per_column : int
        Thumbnails per legend column when ``per_row`` is None (column-major fill).
    per_row : int, optional
        When given, thumbnails are laid out ROW-MAJOR with this many per row, so
        configs that are adjacent in ``parameter_masks`` stay adjacent in the legend.
        The ensemble plots sequence configs by model family, so ``per_row=2`` gives
        one family per pair of rows (DDM 1-2 / DDM 3-4 / RDM 1-2 / ...), and
        ``per_row=n`` puts a single family's configs on one row.

    Returns
    -------
    strip_w_fig : float
        Width of the legend strip in figure coordinates.
    """
    figW, figH = fig.get_size_inches()
    n = len(parameter_masks)
    if per_row is not None and per_row > 0:
        # Row-major: per_row thumbnails across, as many rows as needed.
        n_cols_leg = min(per_row, n)
        n_per_col = int(np.ceil(n / n_cols_leg))
        row_major = True
    else:
        n_cols_leg = int(np.ceil(n / max_per_column))
        n_per_col = int(np.ceil(n / n_cols_leg))
        row_major = False

    # Rough label height in figure coords (1 line of text below each thumbnail)
    label_h_fig = (label_fontsize * 2.2 / 72) / figH

    cell_gap_fig = 2 * pad

    # A single row of thumbnails goes in a horizontal band along the BOTTOM; anything
    # taller stays a left sidebar. Forcing a wide row into the sidebar makes the strip
    # as wide as the number of configs and squashes the main axes to half the figure.
    bottom_band = row_major and n_per_col == 1

    if bottom_band:
        # Size to fit n thumbnails across ~85% of the figure width, square pixels.
        thumb_w_fig = min(thumb_scale * (0.85 - (n - 1) * gap) / n, 0.20)
        thumb_w_in  = thumb_w_fig * figW
        thumb_h_in  = thumb_w_in * num_rows / num_cols
        thumb_h_fig = thumb_h_in / figH
        max_h_fig = 0.22 * thumb_scale
        if thumb_h_fig > max_h_fig:
            thumb_h_fig = max_h_fig
            thumb_h_in  = thumb_h_fig * figH
            thumb_w_in  = thumb_h_in * num_cols / num_rows
            thumb_w_fig = thumb_w_in / figW
        strip_h_fig = pad + thumb_h_fig + label_h_fig + pad
        strip_w_fig = 0.0

        fig.tight_layout(rect=[0, strip_h_fig, 1, 1])
        fig.add_artist(Line2D(
            [0.02, 0.98], [strip_h_fig, strip_h_fig], transform=fig.transFigure,
            color="lightgray", linewidth=0.7, zorder=10,
        ))
        row_w_fig = n * thumb_w_fig + (n - 1) * gap
        x_left = max((1.0 - row_w_fig) / 2, pad)
        y_bottom = pad + label_h_fig
    else:
        # Size thumbnails so a full legend column fits in ~90% of the figure height
        # (square pixels); cap the width so the strip stays a narrow sidebar. The
        # lower clamp keeps small figures (e.g. single-cell global panels) valid.
        thumb_h_fig = max(thumb_scale * (0.9 / n_per_col - label_h_fig - cell_gap_fig), 0.02)
        thumb_h_in  = thumb_h_fig * figH
        thumb_w_in  = thumb_h_in * num_cols / num_rows   # square pixels
        thumb_w_fig = thumb_w_in / figW
        max_w_fig = 0.11 * thumb_scale
        if thumb_w_fig > max_w_fig:
            thumb_w_fig = max_w_fig
            thumb_w_in  = thumb_w_fig * figW
            thumb_h_in  = thumb_w_in * num_rows / num_cols
            thumb_h_fig = thumb_h_in / figH

        strip_w_fig = pad + n_cols_leg * thumb_w_fig + (n_cols_leg - 1) * gap + pad

        # Re-layout main subplots leaving the left strip free
        fig.tight_layout(rect=[strip_w_fig, 0, 1, 1])

        # Vertical divider between the thumbnail strip and the main plot
        fig.add_artist(Line2D(
            [strip_w_fig, strip_w_fig], [0.02, 0.98], transform=fig.transFigure,
            color="lightgray", linewidth=0.7, zorder=10,
        ))

        # Centre each legend column vertically
        cell_h_fig = thumb_h_fig + label_h_fig + cell_gap_fig
        y_top = 0.5 + (n_per_col * cell_h_fig) / 2

    for k in range(n):
        mask_k = np.array(parameter_masks[k])
        # Pad to global num_rows
        if mask_k.shape[0] < num_rows:
            mask_k = np.vstack([
                mask_k,
                np.zeros((num_rows - mask_k.shape[0], num_cols))
            ])

        # Build RGBA image: active = config color, inactive = light gray
        img = np.ones((num_rows, num_cols, 4))
        img[..., :3] = 0.92   # inactive gray
        dark_gray = np.array([0.25, 0.25, 0.25])
        config_rgb = np.array(mcolors.to_rgb(colors[k]))
        if pixel_alphas is not None:
            alpha_k = pixel_alphas[k]
            if np.ndim(alpha_k) == 0:
                # Scalar: uniform fade for all active pixels
                t = float(np.clip(alpha_k, 0.0, 1.0))
                cell_rgb = t * config_rgb + (1 - t) * dark_gray
                img[mask_k == 1.0, :3] = cell_rgb
                img[mask_k == 1.0,  3] = 1.0
            else:
                # 2D array: per-cell fade
                alpha_arr = np.clip(np.array(alpha_k), 0.0, 1.0)
                active_rows, active_cols = np.where(mask_k == 1.0)
                for ar, ac in zip(active_rows, active_cols):
                    t = float(alpha_arr[ar, ac])
                    img[ar, ac, :3] = t * config_rgb + (1 - t) * dark_gray
                    img[ar, ac,  3] = 1.0
        else:
            img[mask_k == 1.0, :3] = config_rgb
            img[mask_k == 1.0,  3] = 1.0

        if bottom_band:
            x0 = x_left + k * (thumb_w_fig + gap)
            y0 = y_bottom
        else:
            if row_major:
                leg_row, leg_col = divmod(k, n_cols_leg)
            else:
                leg_col, leg_row = divmod(k, n_per_col)
            x0 = pad + leg_col * (thumb_w_fig + gap)
            y0 = y_top - leg_row * cell_h_fig - thumb_h_fig

        ax_t = fig.add_axes([x0, y0, thumb_w_fig, thumb_h_fig])
        ax_t.imshow(img, aspect="equal", interpolation="nearest")

        # Major ticks off, minor ticks at pixel boundaries for gridlines
        ax_t.set_xticks([])
        ax_t.set_yticks([])
        ax_t.set_xticks(np.arange(-0.5, num_cols, 1), minor=True)
        ax_t.set_yticks(np.arange(-0.5, num_rows, 1), minor=True)
        ax_t.grid(which="minor", color="white", linewidth=0.4, alpha=0.8)
        ax_t.tick_params(which="minor", length=0)

        for sp in ax_t.spines.values():
            sp.set_visible(False)
        ax_t.set_xlabel(labels[k], fontsize=label_fontsize, labelpad=3)

    return strip_w_fig


def _add_config_legend(fig, colors, labels, fontsize: int = 11):
    """Plain per-config color legend to the right of the axes — used instead of
    mask thumbnails when the grid has no coefficient-matrix structure to show
    (e.g. global-parameter-only ensemble panels)."""
    handles = [
        Line2D([], [], marker="o", linestyle="none", color=c, label=l)
        for c, l in zip(colors, labels)
    ]
    fig.tight_layout()
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
               fontsize=fontsize, frameon=False)


def hex_code():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def bf_colors():
    colors = {
        # BF
        "intercept": "#284734",
        "main_effect": "#275D38",
        "interaction": "#9CAF88",
    }
    return colors

def cogformer_cm_colors():
    colors = {
        "intercept": "#15435F",
        "main_effect": "#007396",
        "interaction": "#3EB1C8",
    }
    return colors

def cogformer_vi_colors():
    colors = {
        "intercept": "#4E2A84",
        "main_effect": "#6969FF",
        "interaction": "#47b5ff",
    }

    return colors

def cogformer_fm_colors():
    colors = {
        "intercept": "#59315F",
        "main_effect": "#EC008C",
        "interaction": "#FF6969",
    }

    return colors

def cogformer_mc_colors():
    colors = {
        "intercept": "#4E2A84",
        "main_effect": "#6969FF",
        "interaction": "#47b5ff",
    }

    return colors

def staedtler_fineliner():
    colors = {
        "bf": bf_colors(),
        "vi": cogformer_vi_colors(),
        "fm": cogformer_fm_colors(),
        "cm": cogformer_cm_colors()
    }
    return colors
