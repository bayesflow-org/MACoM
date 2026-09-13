import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D


# Typography lives in cogformer.utils.style, applied at package import so that
# scripts which never touch this module still get it. Re-exported here because
# callers already read plot_utils.INTER_AVAILABLE.
from .style import INTER_AVAILABLE  # noqa: F401


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

# margin < 1: a label sized to exactly its cell touches its neighbour, and the
# outermost one lands on the figure edge. The few percent held back is the gap.
def fit_text(fig, artists, target_in, axis="x", margin=0.90, min_size=3.0,
             max_size=None):
    """Scale each text artist so its extent along `axis` fills `target_in` inches.

    Used where a label's job is to sit inside a cell rather than to hit an absolute
    point size: the cell is the unit the reader compares against, so a label wider
    than its cell has to shrink and one much narrower is wasting the space. Measured
    per artist, so a long label and a short one in the same figure end up at
    different sizes -- which is the point, since it is the extent that is being
    matched, not the type size.
    """
    if not artists:
        return
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    dpi = fig.dpi
    for art in artists:
        txt = art.get_text() if hasattr(art, "get_text") else ""
        if not txt:
            continue
        bb = art.get_window_extent(renderer=rend)
        cur_in = (bb.width if axis == "x" else bb.height) / dpi
        if cur_in <= 0:
            continue
        size = art.get_fontsize() * (target_in * margin) / cur_in
        if max_size is not None:
            size = min(size, max_size)
        art.set_fontsize(max(min_size, size))


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
    fit_labels: bool = False,
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
    thumb_label_artists = []
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
        thumb_label_artists.append(ax_t.xaxis.label)

    # Thumbnail captions sized to their own thumbnail, for the same reason the grid's
    # axis labels are sized to a cell: the thumbnail is the unit, and a caption wider
    # than it runs into the next one.
    if fit_labels and thumb_label_artists:
        fit_text(fig, thumb_label_artists, thumb_w_fig * figW, axis="x")

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


# --------------------------------------------------------------------------- #
# Model-family identity colours
# --------------------------------------------------------------------------- #
# The palettes above are keyed by EXPERIMENT -- bf = the BayesFlow baseline arm,
# fm = model-family runs, cm/mc = model-class runs. Any figure that puts several
# model families side by side needs the opposite key: one hue per FAMILY, held
# fixed across every figure and every experiment, so a reader learns "magenta is
# DDM" once and it keeps meaning that.
#
# Each family reuses an existing ramp rather than inventing new hex, so the two
# keying schemes can never disagree about what a given hex means:
#
#     ddm  magenta/red  <- cogformer_fm_colors
#     rdm  purple/blue  <- cogformer_vi_colors
#     cdm  green        <- cdm_colors  (see below -- NOT bf_colors)
#     lba  orange/gold  <- lba_colors
#
# Each ramp runs dark -> saturated -> light across (intercept, main_effect,
# interaction) and keeps that effect-type meaning within a family. A figure that
# needs a SINGLE colour per family should take ``main_effect``, the saturated
# stop; ``interaction`` is the light stop and reads well as a fill underneath it.
def cdm_colors():
    """CDM's identity green.

    Deliberately NOT bf_colors(), whose greens are the BayesFlow baseline's and
    are built to recede behind the CogFormer result. Measured against the DDM and
    RDM ramps, that palette fails as a family identity on three counts: its mid
    stop is far too dark (HSV value 0.36 against DDM 0.93 / RDM 1.00), so a CDM
    median line barely separates from the near-black observed curve (contrast 2.20
    against DDM 4.02 / RDM 4.11); its dark and mid stops are nearly the same
    colour (contrast 1.33 against DDM 2.44 / RDM 2.55), leaving an effectively
    two-stop ramp; and its light stop is desaturated to 0.22 against DDM 0.59 /
    RDM 0.72, reading grey-green rather than green.

    This ramp keeps the original brand dark green and repairs the other two stops:
    mid/observed contrast 4.94, dark-mid separation 2.98.
    """
    colors = {
        "intercept": "#284734",
        "main_effect": "#1F9E55",
        "interaction": "#8FD9A8",
    }
    return colors


def lba_colors():
    """LBA's identity orange/gold.

    Replaces ``cogformer_cm_colors`` (blue/teal) as LBA's family identity. Two
    reasons, one practical and one structural.

    Practically, the LBA figures were never rendering in teal: every model-class
    validate script sets ``colors = cogformer_mc_colors()`` once, globally, and
    that function is a byte-for-byte duplicate of ``cogformer_vi_colors`` -- RDM's
    identity. So LBA has been drawn in RDM's purple/blue in every arm of
    experiment 3, which is precisely the experiment where the two sit side by side.

    Structurally, orange/gold is the only warm hue left unclaimed: DDM holds
    magenta/red, RDM purple/blue, CDM green. Its nearest neighbour is DDM at 67
    degrees of hue, against 114 and 151 for CDM and RDM.

    Gold is intrinsically luminous, so the naive bright ramp (#F59E0B ->  #FCD34D)
    puts the light stop at 1.44 contrast against a white page -- an invisible fill.
    The mid stop is therefore deepened to a burnt orange, which buys room beneath
    it: light/white 1.77 against CDM's 1.66, mid/observed 4.51 (DDM 4.02, RDM 4.11,
    CDM 4.94), dark-mid separation 2.81 (band 2.44-2.98) and mid-light 2.14.

    ``cogformer_cm_colors`` is NOT removed -- it is still the default colour inside
    ``adaptive_recovery``, ``adaptive_coverage`` and ``adaptive_pair_posterior``.
    """
    colors = {
        "intercept": "#5E3410",
        "main_effect": "#C96A04",
        "interaction": "#F3B94F",
    }
    return colors


FAMILY_PALETTES = {
    "ddm": cogformer_fm_colors,
    "rdm": cogformer_vi_colors,
    "cdm": cdm_colors,
    "lba": lba_colors,
}

# Observed data is not a model, so it never takes a family colour: it is the
# shared reference every family is judged against, and giving it a hue that moves
# between figures would imply it changes with the model. Neutral near-black
# everywhere, with a lighter grey for non-data guide lines.
OBSERVED_COLOR = "#1a1a2e"
GUIDE_COLOR = "#9a9aa5"


def family_colors(family: str) -> dict:
    """Three-stop identity ramp for one model family.

    Parameters
    ----------
    family : str
        Model family key, case-insensitive: ``ddm``, ``rdm``, ``cdm`` or ``lba``.

    Returns
    -------
    dict with keys ``intercept`` (dark), ``main_effect`` (saturated) and
    ``interaction`` (light).
    """
    key = family.lower()
    if key not in FAMILY_PALETTES:
        raise KeyError(
            f"no identity palette for model family {family!r}; "
            f"known families: {sorted(FAMILY_PALETTES)}"
        )
    return FAMILY_PALETTES[key]()


def family_line_fill(family: str) -> tuple[str, str]:
    """``(line, fill)`` for the common one-family-per-panel case.

    The saturated stop for the estimate and the light stop for the band beneath
    it, so the two read as one family rather than as two series.
    """
    c = family_colors(family)
    return c["main_effect"], c["interaction"]
