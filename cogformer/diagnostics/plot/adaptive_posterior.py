import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from cogformer.utils.plot_utils import cogformer_cm_colors
from cogformer.simulator.design_utils import (
    build_parameter_mask, expand_design_config_to_mask, build_row_labels,
)


def adaptive_posterior(
    samples: np.ndarray,
    design_config: dict,
    intrinsic_params: list[str],
    max_num_categories: int,
    priors: np.ndarray = None,
    targets: np.ndarray = None,
    parameter_mask: np.ndarray = None,
    variable_names: list[str] = None,
    intercept_color: str = "#4e2a84",
    main_effect_color: str = "#6969ff",
    interaction_color: str = "#ff6969",
    show_prior: bool = True,
    show_prior_mean: bool = True,
    show_posterior_mean: bool = True,
    show_target: bool = True,
    prior_color: str = "0.3",
    prior_alpha: float = 0.12,
    posterior_alpha: float = 0.6,
    target_color: str = "black",
    title_fontsize: int = 20,
    label_fontsize: int = 14,
    tick_fontsize: int = 11,
    legend_fontsize: int = 12,
    figsize: tuple = None,
    global_params: list[str] = None,
    max_num_regressors: int = None,
    categorical_regressors: list[str] = None,
) -> plt.Figure:
    """Coefficient-matrix grid of per-target posterior KDEs.

    Same row/column layout, compaction, and N/A tiling convention as
    :func:`adaptive_recovery` / :func:`adaptive_ecdf` — one subplot per
    (design row, intrinsic param) cell, defaulting to only the rows/columns
    where at least one estimation target is active.

    Each active cell shows the posterior KDE, an optional prior-KDE backdrop
    (``show_prior``, requires ``priors``), and up to three independently
    toggleable dashed vertical lines: prior mean, posterior mean, and ground
    truth (requires ``targets``).

    Pairwise posterior correlations previously shown as 2D KDE panels in the
    old pairplot version of this function now live in
    :func:`cogformer.diagnostics.plot.adaptive_sample_correlation.adaptive_sample_correlation`.

    ``global_params`` is not yet supported here (global-parameter diagnostics
    are being reworked separately); pass it as ``None``.

    Parameters
    ----------
    samples : np.ndarray of shape (num_draws, num_rows, num_cols)
        Posterior draws for a single dataset.
    priors : np.ndarray of shape (num_prior_draws, num_rows, num_cols), optional
        Prior draws for the same dataset/design, for the prior-KDE backdrop.
    targets : np.ndarray of shape (num_rows, num_cols), optional
        Ground-truth parameter values for the dashed ground-truth line.

    Returns
    -------
    fig : plt.Figure, or None if no column is selected.
    """
    if global_params:
        raise NotImplementedError(
            "global_params is not yet supported in the rewritten adaptive_posterior; "
            "global-parameter diagnostics are being reworked separately."
        )

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

    row_ylabels, row_kinds = build_row_labels(design_config, max_num_categories, return_kinds=True,
                                              categorical_regressors=categorical_regressors,
                                              parameter_mask=parameter_mask)
    _kind_color = {"intercept": intercept_color, "main": main_effect_color, "interaction": interaction_color}
    row_colors = [_kind_color[k] for k in row_kinds]
    num_rows = min(num_rows, len(row_ylabels))
    row_ylabels = row_ylabels[:num_rows]
    row_colors = row_colors[:num_rows]
    parameter_mask = parameter_mask[:num_rows]

    # Default to rows/columns where at least one estimation target is active.
    col_idx = np.where(np.any(parameter_mask, axis=0))[0]
    if col_idx.size == 0:
        return None
    row_idx = np.where(np.any(parameter_mask[:, col_idx], axis=1))[0]
    if row_idx.size == 0:
        return None

    parameter_mask = parameter_mask[np.ix_(row_idx, col_idx)]
    samples_f = samples[..., :num_rows, :][..., row_idx, :][..., col_idx]
    priors_f = priors[..., :num_rows, :][..., row_idx, :][..., col_idx] if priors is not None else None
    targets_f = targets[:num_rows, :][np.ix_(row_idx, col_idx)] if targets is not None else None
    row_ylabels = [row_ylabels[i] for i in row_idx]
    row_colors  = [row_colors[i]  for i in row_idx]
    col_labels  = [variable_names[i] for i in col_idx]
    num_rows    = len(row_idx)
    num_cols    = len(col_idx)

    if figsize is None:
        figsize = (3 * num_cols, 2.6 * num_rows)

    fig, axarr = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)

    legend_handles: list = []
    legend_seen: set = set()

    def _add_legend(handle, key):
        if key not in legend_seen:
            legend_seen.add(key)
            legend_handles.append(handle)

    for r_f in range(num_rows):
        ylabel = row_ylabels[r_f]
        color = row_colors[r_f]

        for c_f in range(num_cols):
            ax = axarr[r_f, c_f]

            if parameter_mask[r_f, c_f] != 1.0:
                ax.set_facecolor(color)
                ax.patch.set_alpha(0.05)
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                ax.text(
                    0.5, 0.5, "N/A",
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=18, weight="bold",
                    alpha=0.7, color=color,
                )
                ax.set_ylabel(ylabel if c_f == 0 else "", fontsize=label_fontsize)
                ax.set_title(col_labels[c_f] if r_f == 0 else "", fontsize=title_fontsize)
                continue

            post = np.asarray(samples_f[:, r_f, c_f], dtype=float)
            post = post[~np.isnan(post)]

            if show_prior and priors_f is not None:
                pr = np.asarray(priors_f[:, r_f, c_f], dtype=float)
                pr = pr[~np.isnan(pr)]
                if pr.size >= 2 and np.std(pr) > 1e-12:
                    sns.kdeplot(x=pr, ax=ax, fill=True, color=prior_color, alpha=prior_alpha)
                    _add_legend(
                        Patch(facecolor=prior_color, edgecolor="none", alpha=prior_alpha * 3, label="Prior"),
                        "prior_kde",
                    )
                    if show_prior_mean:
                        ax.axvline(pr.mean(), color=prior_color, linestyle="--", linewidth=1.5)
                        _add_legend(
                            Line2D([0], [0], color=prior_color, linestyle="--", linewidth=1.5, label="Prior mean"),
                            "prior_mean",
                        )

            if post.size >= 2 and np.std(post) > 1e-12:
                sns.kdeplot(x=post, ax=ax, fill=True, color=color, alpha=posterior_alpha)
                _add_legend(Patch(facecolor=color, edgecolor="none", alpha=posterior_alpha, label="Posterior"), "post_kde")
            elif post.size >= 1:
                ax.axvline(post.mean(), color=color, linewidth=2.0)

            if show_posterior_mean and post.size >= 1:
                ax.axvline(post.mean(), color=color, linestyle="--", linewidth=1.5)
                _add_legend(
                    Line2D([0], [0], color=color, linestyle="--", linewidth=1.5, label="Posterior mean"),
                    "post_mean",
                )

            if show_target and targets_f is not None:
                t = targets_f[r_f, c_f]
                if not np.isnan(t):
                    ax.axvline(t, color=target_color, linestyle=":", linewidth=1.5)
                    _add_legend(
                        Line2D([0], [0], color=target_color, linestyle=":", linewidth=1.5, label="Ground truth"),
                        "target",
                    )

            ax.set_yticks([])
            ax.grid(True, color="lightgray", linestyle="--", linewidth=0.5, alpha=0.2)
            sns.despine(ax=ax, left=True)
            ax.set_ylabel(ylabel if c_f == 0 else "", fontsize=label_fontsize)
            ax.set_title(col_labels[c_f] if r_f == 0 else "", fontsize=title_fontsize)
            ax.tick_params(axis="x", labelsize=tick_fontsize)

    fig.tight_layout()
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=len(legend_handles),
            fontsize=legend_fontsize,
            bbox_to_anchor=(0.5, 0),
            bbox_transform=fig.transFigure,
        )
        fig.subplots_adjust(bottom=0.14)
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

    prior_samples = np.random.normal(0.0, 5.0, (num_draws, R, C))
    posterior_samples = 0.3 * np.random.normal(0.0, 1.0, (num_draws, R, C)) + 1.5
    targets = np.random.normal(1.5, 0.3, (R, C))

    fig = adaptive_posterior(
        samples=posterior_samples,
        priors=prior_samples,
        targets=targets,
        design_config=design_config,
        intrinsic_params=intrinsic_params,
        max_num_categories=2,
        variable_names=variable_names,
        intercept_color=colors["intercept"],
        main_effect_color=colors["main_effect"],
        interaction_color=colors["interaction"],
    )
    fig.savefig("test_adaptive_posterior.pdf", bbox_inches="tight")
    print("saved test_adaptive_posterior.pdf")
