import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

from cogformer.simulator.design_utils import build_parameter_mask, expand_design_config_to_mask, build_row_labels
from cogformer.utils.plot_utils import cogformer_cm_colors


def _target_label(col_label: str, row_label: str, row_kind: str) -> str:
    """Same naming convention as the old unfolded pairplot columns: bare
    parameter name for intercept rows, ``"<param> -- <coef>"`` otherwise."""
    return col_label if row_kind == "intercept" else f"{col_label} -- {row_label}"


def compute_sample_correlation(
    samples: np.ndarray,
    design_config: dict,
    intrinsic_params: list[str],
    max_num_categories: int,
    parameter_mask: np.ndarray = None,
    variable_names: list[str] = None,
    max_num_regressors: int = None,
    categorical_regressors: list[str] = None,
) -> tuple[np.ndarray | None, list[str], list[str]]:
    """Flatten the active coefficient-matrix cells into named posterior-sample
    columns and compute their K x K Pearson correlation matrix.

    samples : np.ndarray of shape (num_draws, num_rows, num_cols)
        Posterior draws for a single dataset.

    Returns
    -------
    corr : np.ndarray of shape (K, K), or None if fewer than 2 targets are active.
    labels : list[str] of length K
        Display name per estimation target.
    kinds : list[str] of length K
        One of "intercept", "main", "interaction" per target, for tick coloring.
    """
    if variable_names is None:
        variable_names = intrinsic_params

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
    num_rows, num_cols = parameter_mask.shape

    row_labels, row_kinds = build_row_labels(design_config, max_num_categories, return_kinds=True,
                                             categorical_regressors=categorical_regressors,
                                             parameter_mask=parameter_mask)
    num_rows = min(num_rows, len(row_labels))
    row_labels = row_labels[:num_rows]
    row_kinds = row_kinds[:num_rows]
    parameter_mask = parameter_mask[:num_rows]
    samples = samples[..., :num_rows, :]

    labels: list[str] = []
    kinds: list[str] = []
    columns: list[np.ndarray] = []
    for r in range(num_rows):
        for c in range(num_cols):
            if parameter_mask[r, c] == 1.0:
                labels.append(_target_label(variable_names[c], row_labels[r], row_kinds[r]))
                kinds.append(row_kinds[r])
                columns.append(np.asarray(samples[:, r, c], dtype=float))

    if len(labels) < 2:
        return None, labels, kinds

    mat = np.column_stack(columns)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(mat, rowvar=False)
    return corr, labels, kinds


def adaptive_sample_correlation(
    samples: np.ndarray,
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
    tick_fontsize: int = 10,
    annot_fontsize: int = 9,
    fmt: str = ".2f",
    global_params: list[str] = None,
    max_num_regressors: int = None,
    categorical_regressors: list[str] = None,
) -> plt.Figure:
    """Lower-triangle heatmap of pairwise posterior-sample correlations across
    estimation targets — single correlation numbers replacing the pairwise 2D
    KDE panels previously shown in the ``adaptive_posterior`` pairplot.

    Only active coefficient-matrix cells become rows/columns of this matrix in
    the first place (same "at least one active target" convention as
    :func:`cogformer.diagnostics.plot.adaptive_posterior.adaptive_posterior`),
    so there is no separate N/A tiling step here — a degenerate (zero-variance)
    target still renders as N/A if it slips through.

    ``global_params`` is not yet supported here (global-parameter diagnostics
    are being reworked separately); pass it as ``None``.

    Returns
    -------
    fig : plt.Figure, or None if fewer than 2 estimation targets are active.
    """
    if global_params:
        raise NotImplementedError(
            "global_params is not yet supported in adaptive_sample_correlation; "
            "global-parameter diagnostics are being reworked separately."
        )

    corr, labels, kinds = compute_sample_correlation(
        samples=samples,
        design_config=design_config,
        intrinsic_params=intrinsic_params,
        max_num_categories=max_num_categories,
        parameter_mask=parameter_mask,
        variable_names=variable_names,
        max_num_regressors=max_num_regressors,
        categorical_regressors=categorical_regressors,
    )
    if corr is None:
        return None

    _kind_color = {"intercept": intercept_color, "main": main_effect_color, "interaction": interaction_color}
    tick_colors = [_kind_color[k] for k in kinds]
    K = corr.shape[0]

    # Lower triangle only: diagonal and upper triangle are blanked, not N/A'd
    # (they're redundant/self entries, not missing estimation targets).
    display = np.full_like(corr, np.nan)
    tril = np.tril_indices(K, k=-1)
    display[tril] = corr[tril]

    cell_size = 0.9
    if figsize is None:
        figsize = (K * cell_size + 2.5, K * cell_size + 2.5)

    fig, ax = plt.subplots(figsize=figsize)
    cm = plt.get_cmap(cmap)
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    masked = np.ma.masked_invalid(display)
    ax.imshow(masked, cmap=cm, norm=norm, aspect="equal")

    for r in range(K):
        for c in range(K):
            if c >= r:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="white", edgecolor="none", zorder=2))
                continue

            val = display[r, c]
            if np.isnan(val):
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="lightgray", alpha=0.3, zorder=2))
                ax.text(c, r, "N/A", ha="center", va="center", fontsize=annot_fontsize, color="gray")
                continue

            rgba = cm(norm(val))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(
                c, r, format(val, fmt),
                ha="center", va="center", fontsize=annot_fontsize,
                color="white" if luminance < 0.5 else "black",
            )

    ax.set_xticks(range(K))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(range(K))
    ax.set_yticklabels(labels)
    for i, lbl in enumerate(ax.get_xticklabels()):
        lbl.set_color(tick_colors[i])
    for i, lbl in enumerate(ax.get_yticklabels()):
        lbl.set_color(tick_colors[i])
    ax.tick_params(axis="both", length=0, labelsize=tick_fontsize)
    ax.set_xlim(-0.5, K - 0.5)
    ax.set_ylim(K - 0.5, -0.5)

    ax.set_title("Posterior Sample Correlation", fontsize=title_fontsize)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    colors = cogformer_cm_colors()

    intrinsic_params = ["v", "a", "tau", "s_v", "s_tau"]
    variable_names = [r"$v$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"]

    design_config = {
        "1": intrinsic_params,
        "u_1": ["v", "a", "tau", "s_v"],
        "u_2": ["v", "a", "tau"],
        "u_1:u_2": ["v", "a"],
    }

    parameter_mask = build_parameter_mask(
        design_config=design_config,
        intrinsic_params=intrinsic_params,
        max_num_categories=2,
        keep_intercept=True,
    )
    num_draws = 500
    R, C = parameter_mask.shape

    # Fabricate correlated posterior draws so the heatmap has something to show.
    K_active = int(parameter_mask.sum())
    latent = np.random.normal(size=(num_draws, 1))
    posterior_samples = np.random.normal(0.0, 1.0, (num_draws, R, C))
    posterior_samples += 0.6 * latent[:, :, None]

    fig = adaptive_sample_correlation(
        samples=posterior_samples,
        design_config=design_config,
        intrinsic_params=intrinsic_params,
        max_num_categories=2,
        variable_names=variable_names,
        intercept_color=colors["intercept"],
        main_effect_color=colors["main_effect"],
        interaction_color=colors["interaction"],
    )
    fig.savefig("test_adaptive_sample_correlation.pdf", bbox_inches="tight")
    print("saved test_adaptive_sample_correlation.pdf")
