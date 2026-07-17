import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from cogformer.utils.plot_utils import cogformer_cm_colors
from cogformer.simulator.design_utils import (
    build_parameter_mask, build_design_matrix, build_random_design_config,
    build_row_labels, expand_design_config_to_mask,
)


from matplotlib.patches import Patch


def effect_type_from_unfolded_name(name: str) -> str:
    """
    Determine effect type from unfolded column name.

    Expected naming scheme from unfolded_names() (labels from build_row_labels):
      - "<param>"                              -> intercept
      - "<param> -- $u_k$" / "-- $u_k:c_i$"    -> main (":c_i" marks a categorical contrast)
      - "<param> -- $u_k \\times u_j$" (any contrast suffixes) -> interaction

    Returns
    -------
    effect_type : {"intercept", "main", "interaction"}
    """
    if not isinstance(name, str) or " -- " not in name:
        return "intercept"

    regressor_part = name.split(" -- ", 1)[1]
    return "interaction" if (r"\times" in regressor_part or "×" in regressor_part) else "main"


def pick_pair_color(x_name: str, y_name: str, colors: dict[str, str]) -> str:
    """
    Pick a single color for a (x,y) panel.
    Rule: highest-order present wins: interaction > main > intercept.
    """
    tx = effect_type_from_unfolded_name(x_name)
    ty = effect_type_from_unfolded_name(y_name)

    if tx == "interaction" or ty == "interaction":
        return colors["interaction"]
    if tx == "main" or ty == "main":
        return colors["main"]
    return colors["intercept"]

def unfolded_names(
    design_config: dict,
    intrinsic_params: list[str],
    max_num_categories: int,
    col_labels: list[str] | None = None,
    categorical_regressors: list[str] | None = None,
    parameter_mask: np.ndarray | None = None,
):
    """
    Names aligned with the unfolded parameter matrix.

    Parameters
    ----------
    col_labels : list[str] | None
        Optional display labels for columns (e.g. LaTeX r"$v$").
        If None, uses intrinsic_params.
    categorical_regressors, parameter_mask :
        Forwarded to :func:`build_row_labels` for categorical contrast labeling.
    """
    if col_labels is None:
        col_labels = intrinsic_params
    num_cols = len(col_labels)

    # Row-major over the parameter-mask layout (intercept, main-effect blocks, then
    # interaction (i, j) blocks) so names align with parameter_mask.reshape(-1).
    row_labels, row_kinds = build_row_labels(design_config, max_num_categories, return_kinds=True,
                                             categorical_regressors=categorical_regressors,
                                             parameter_mask=parameter_mask)

    names = []
    for coef, kind in zip(row_labels, row_kinds):
        if kind == "intercept":
            # intercept row: named by the column parameter alone (a bare name with no
            # " -- " is what effect_type_from_unfolded_name treats as the intercept)
            names.extend(col_labels[c] for c in range(num_cols))
        else:
            names.extend(f"{col_labels[c]} -- {coef}" for c in range(num_cols))

    return names


def samples_to_unfolded_df(
    samples: np.ndarray | pd.DataFrame,
    design_config: dict,
    intrinsic_params: list[str],
    max_num_categories: int,
    parameter_mask: np.ndarray,
    unfold: bool,
    col_labels: list[str] | None = None,
    categorical_regressors: list[str] | None = None,
):
    """
    Convert samples into a DataFrame aligned with the unfolded parameter matrix.
    ...
    """
    names = unfolded_names(
        design_config=design_config,
        intrinsic_params=intrinsic_params,
        max_num_categories=max_num_categories,
        col_labels=col_labels,
        categorical_regressors=categorical_regressors,
        parameter_mask=parameter_mask,
    )

    if isinstance(samples, pd.DataFrame):
        # assume user already provided correct columns
        df = samples.copy()
        if df.shape[1] != len(names):
            raise ValueError("DataFrame has unexpected number of columns for unfolded layout.")
    else:
        x = np.asarray(samples)
        if x.ndim == 3:
            N, R, C = x.shape
            x2 = x.reshape(N, R * C)
        elif x.ndim == 2:
            x2 = x
        else:
            raise ValueError("samples must be (N,R,C), (N,R*C), or a DataFrame.")

        if x2.shape[1] != len(names):
            raise ValueError(
                f"Expected {len(names)} columns for unfolded layout, got {x2.shape[1]}."
            )

        df = pd.DataFrame(x2, columns=names)

    # build patched column list from mask (row-major)
    mask_flat = parameter_mask.reshape(-1)
    patched_cols = [names[i] for i, m in enumerate(mask_flat) if float(m) != 1.0]
    active_cols = [names[i] for i, m in enumerate(mask_flat) if float(m) == 1.0]

    if unfold:
        # keep all cols, but set patched columns to NaN so seaborn doesn't draw density/scatter
        df.loc[:, patched_cols] = np.nan
        return df, active_cols, patched_cols

    # only keep active (in-scope) columns
    return df.loc[:, active_cols], active_cols, patched_cols


def patch_pairgrid_axes(g: sns.PairGrid, patched_cols: set[str], patch_label: str = "N/A"):
    """
    Patch cells corresponding to inactive parameters with a grey background.
    Works even when PairGrid was created with corner=True.
    """
    for i, row_name in enumerate(g.y_vars):
        for j, col_name in enumerate(g.x_vars):

            ax = g.axes[i, j]

            # <-- NEW: skip non-existent axes (happens when corner=True)
            if ax is None:
                continue

            if (row_name in patched_cols) or (col_name in patched_cols):
                ax.cla()
                ax.set_facecolor("lightgray")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(
                    0.5, 0.5, patch_label,
                    ha="center", va="center",
                    transform=ax.transAxes,
                    fontsize=10, color="gray"
                )


def adaptive_posterior(
    samples: np.ndarray | pd.DataFrame,
    design_config: dict,
    intrinsic_params: list[str],
    max_num_categories: int,
    variable_names: list[str] = None,
    targets: np.ndarray = None,
    priors: np.ndarray = None,
    parameter_mask: np.ndarray = None,
    col_labels: list[str] | None = None,
    intercept_color: str = "#4e2a84",
    main_effect_color: str = "#6969ff",
    interaction_color: str = "#ff6969",
    num_bins: int = 10,
    height: float = 2.5,
    unfold: bool = True,
    add_legend: bool = False,
    show_upper_scatter: bool = True,
    show_prior: bool = False,
    prior_color: str = "0.3",
    prior_alpha: float = 0.12,
    target_color: str = "black",
    global_params: list[str] = None,
    global_only: bool = False,
    max_num_regressors: int = None,
    categorical_regressors: list[str] = None,
):
    """
    Pairplot of posterior samples with adaptive "patching" consistent with adaptive_recovery.

    New options
    -----------
    show_upper_scatter : bool
        If False, omits the upper triangle entirely (diag + lower only).
    show_prior : bool
        If True and `priors` is provided, draws prior samples behind posterior.
    global_params : list[str], optional
        Intrinsic params estimated jointly but presented separately (e.g. ``pi``).
        With ``global_only=False`` (default) their columns are excluded from the
        pairplot; with ``global_only=True`` only those columns are kept (used by
        :func:`global_posterior`). Returns ``None`` if no column is selected.
    """
    if col_labels is None and variable_names is not None:
        col_labels = variable_names

    if parameter_mask is None:
        parameter_mask = build_parameter_mask(
            design_config=design_config,
            max_num_categories=max_num_categories,
            intrinsic_params=intrinsic_params,
            keep_intercept=True,
        )
    if parameter_mask.ndim == 3:
        parameter_mask = parameter_mask[0]

    # collate() pads the grid to max dims; cap to rows the design_config covers
    # (intercept + main blocks + declared interaction blocks — a contiguous prefix).
    design_config = expand_design_config_to_mask(design_config, parameter_mask, max_num_categories,
                                                 max_num_regressors=max_num_regressors)
    _n_design_rows = len(build_row_labels(design_config, max_num_categories))
    parameter_mask = parameter_mask[:_n_design_rows]
    if not isinstance(samples, pd.DataFrame):
        _s = np.asarray(samples)
        if _s.ndim == 3:
            samples = _s[:, :_n_design_rows, :]

    df_post, active_cols, patched_cols = samples_to_unfolded_df(
        samples=samples,
        design_config=design_config,
        intrinsic_params=intrinsic_params,
        max_num_categories=max_num_categories,
        parameter_mask=parameter_mask,
        unfold=unfold,
        col_labels=col_labels,
        categorical_regressors=categorical_regressors,
    )

    # Global parameters (e.g. pi) are estimated jointly but presented separately.
    # Classify each unfolded column by its intrinsic param, then exclude those
    # columns (or keep only them when global_only).
    global_names = set()
    if global_params:
        _cl = col_labels if col_labels is not None else intrinsic_params
        _names_all = unfolded_names(
            design_config=design_config,
            intrinsic_params=intrinsic_params,
            max_num_categories=max_num_categories,
            col_labels=_cl,
            categorical_regressors=categorical_regressors,
            parameter_mask=parameter_mask,
        )
        _ncl = len(_cl)
        global_names = {
            _names_all[i] for i in range(len(_names_all))
            if intrinsic_params[i % _ncl] in global_params
        }

        if global_only:
            keep_cols = [c for c in active_cols if c in global_names]
            if not keep_cols:
                return None
            df_post = df_post.loc[:, keep_cols]
            active_cols = keep_cols
            patched_cols = []
        else:
            df_post = df_post.loc[:, [c for c in df_post.columns if c not in global_names]]
            active_cols = [c for c in active_cols if c not in global_names]
            patched_cols = [c for c in patched_cols if c not in global_names]
            if df_post.shape[1] == 0:
                return None

    # Build targets lookup: col_name -> scalar value for one dataset
    targets_dict = None
    if targets is not None:
        t = np.asarray(targets)
        t_flat = t.reshape(-1) if t.ndim == 2 else t
        names_all = unfolded_names(
            design_config=design_config,
            intrinsic_params=intrinsic_params,
            max_num_categories=max_num_categories,
            col_labels=col_labels,
            categorical_regressors=categorical_regressors,
            parameter_mask=parameter_mask,
        )
        mask_flat = parameter_mask.reshape(-1)
        if unfold:
            targets_dict = {names_all[i]: t_flat[i] for i in range(len(names_all))}
        else:
            targets_dict = {
                names_all[i]: t_flat[i]
                for i in range(len(names_all))
                if float(mask_flat[i]) == 1.0
            }

    df_prior = None
    if show_prior and priors is not None:
        df_prior, _, _ = samples_to_unfolded_df(
            samples=priors,
            design_config=design_config,
            intrinsic_params=intrinsic_params,
            max_num_categories=max_num_categories,
            parameter_mask=parameter_mask,
            unfold=unfold,
            col_labels=col_labels,
            categorical_regressors=categorical_regressors,
        )
        # Keep prior columns aligned with the (global-filtered) posterior columns.
        df_prior = df_prior.loc[:, [c for c in df_prior.columns if c in df_post.columns]]

    colors = {
        "intercept": intercept_color,
        "main": main_effect_color,
        "interaction": interaction_color,
    }

    # If we don't want upper triangle, let seaborn handle layout cleanly.
    g = sns.PairGrid(df_post, corner=(not show_upper_scatter), height=height)

    def _diag_hist(x, **kwargs):
        name = getattr(x, "name", "")
        c = colors[effect_type_from_unfolded_name(name)]

        x_post = x.dropna()
        if x_post.size < 2:
            return

        # Prior behind posterior (if available)
        if df_prior is not None and name in df_prior.columns:
            x_pr = df_prior[name].dropna()
            if x_pr.size >= 2:
                sns.histplot(x=x_pr, bins=num_bins, stat="density", color=prior_color, alpha=prior_alpha)

        # Avoid kde warnings for constant vectors
        if np.nanstd(x_post.values) == 0.0:
            sns.histplot(x=x_post, bins=num_bins, stat="density", color=c, alpha=0.9)
            return

        sns.histplot(x=x_post, kde=True, bins=num_bins, stat="density", color=c, alpha=0.9)

        if targets_dict is not None and name in targets_dict:
            plt.gca().axvline(
                targets_dict[name], color=target_color,
                linestyle="--", linewidth=1.5, label="Ground Truth",
            )

    def _lower_kde(x, y, **kwargs):
        xname = getattr(x, "name", "")
        yname = getattr(y, "name", "")

        # Posterior
        post = pd.concat([x, y], axis=1)
        post.columns = ["x", "y"]
        post = post.dropna()
        if post.shape[0] < 10:
            return
        if np.nanstd(post["x"].values) < 1e-12 or np.nanstd(post["y"].values) < 1e-12:
            return

        # Prior behind posterior (if available)
        if df_prior is not None and (xname in df_prior.columns) and (yname in df_prior.columns):
            pr = pd.DataFrame({"x": df_prior[xname], "y": df_prior[yname]}).dropna()
            if pr.shape[0] >= 10 and np.nanstd(pr["x"].values) >= 1e-12 and np.nanstd(pr["y"].values) >= 1e-12:
                sns.kdeplot(x=pr["x"], y=pr["y"], fill=True, color=prior_color, alpha=prior_alpha)

        c = pick_pair_color(xname, yname, colors)
        sns.kdeplot(x=post["x"], y=post["y"], fill=True, color=c, alpha=0.35)

        if targets_dict is not None and xname in targets_dict and yname in targets_dict:
            plt.gca().scatter(
                targets_dict[xname], targets_dict[yname],
                marker="x", color=target_color, s=80, linewidths=1.5, zorder=5,
            )

    def _upper_scatter(x, y, **kwargs):
        tmp = pd.DataFrame({"x": x, "y": y}).dropna()
        if tmp.shape[0] < 2:
            return

        c = pick_pair_color(getattr(x, "name", ""), getattr(y, "name", ""), colors)
        sns.scatterplot(x=tmp["x"], y=tmp["y"], linewidth=0, alpha=0.25, color=c)

        xname = getattr(x, "name", "")
        yname = getattr(y, "name", "")
        if targets_dict is not None and xname in targets_dict and yname in targets_dict:
            plt.gca().scatter(
                targets_dict[xname], targets_dict[yname],
                marker="x", color=target_color, s=80, linewidths=1.5, zorder=5,
            )

    g.map_diag(_diag_hist)
    g.map_lower(_lower_kde)

    if show_upper_scatter:
        g.map_upper(_upper_scatter)

    if unfold and len(patched_cols) > 0:
        patch_pairgrid_axes(g, patched_cols=patched_cols, patch_label="N/A")

    if add_legend:
        from matplotlib.lines import Line2D
        handles = [
            Patch(facecolor=colors["intercept"], edgecolor="none", label="Intercept"),
            Patch(facecolor=colors["main"], edgecolor="none", label="Main effect"),
            Patch(facecolor=colors["interaction"], edgecolor="none", label="Interaction"),
        ]
        if targets_dict is not None:
            handles.append(
                Line2D([0], [0], marker="x", color=target_color, linestyle="--",
                       linewidth=1.5, markersize=8, label="Ground Truth")
            )
        g.figure.legend(handles=handles, loc="upper right", frameon=False)

    return g


def global_posterior(
    samples,
    design_config: dict,
    intrinsic_params: list[str],
    max_num_categories: int,
    global_params: list[str],
    global_color: str = "#4e2a84",
    **kwargs,
):
    """Posterior pairplot for global parameters (e.g. ``pi``) only, plotted
    separately from the coefficient matrix. Thin wrapper over
    :func:`adaptive_posterior` with ``global_only=True``. Returns ``None`` when no
    global parameter is active.
    """
    return adaptive_posterior(
        samples=samples,
        design_config=design_config,
        intrinsic_params=intrinsic_params,
        max_num_categories=max_num_categories,
        intercept_color=global_color,
        main_effect_color=global_color,
        interaction_color=global_color,
        global_params=global_params,
        global_only=True,
        **kwargs,
    )


if __name__ == "__main__":
    debug = True
    colors = cogformer_cm_colors()

    intrinsic_params = ["v", "a", "tau", "s_v", "s_tau"]
    variable_names = [r"$v$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"]

    design_config = {
        "1": intrinsic_params,  # intercept
        "u_1": ["v", "a", "tau", "s_v"],
        "u_2": ["v", "a", "tau"],
        "u_1:u_2": ["v", "a"],
    }
    num_params = len(variable_names)


    parameter_mask = build_parameter_mask(
        design_config=design_config,
        intrinsic_params=intrinsic_params,
        max_num_categories=2,
        keep_intercept=True,
    )

    num_draws = 500
    R, C = parameter_mask.shape
    unfold = True

    # posterior over the full unfolded matrix
    prior_samples = np.random.normal(0.0, 5.0, (num_draws, R, C))
    posterior_samples = np.random.normal(0.0, 1.0, (num_draws, R, C))
    print(posterior_samples.shape)

    g = adaptive_posterior(
        samples=posterior_samples,  # (N, R, C) or (N, R*C)
        priors=prior_samples,
        design_config=design_config,
        intrinsic_params=intrinsic_params,
        max_num_categories=2,
        unfold=True,  # or False
        show_prior=True,
        show_upper_scatter=False
    )
    g.savefig(f"posterior_pairplot{'_unfolded' if unfold else ''}.pdf")
    print("awesome")