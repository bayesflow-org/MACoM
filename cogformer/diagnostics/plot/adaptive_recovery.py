import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from collections.abc import Callable

from cogformer.simulator.design_utils import build_parameter_mask, expand_design_config_to_mask
from cogformer.utils.plot_utils import make_quadratic, credible_interval, cogformer_cm_colors


def adaptive_recovery(
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
    uncertainty_agg: Callable = credible_interval,
    title_fontsize: int = 20,
    label_fontsize: int = 14,
    figsize: tuple = None
):
    if variable_names is None:
        variable_names = design_config["1"]

    if parameter_mask is None:
        parameter_mask = build_parameter_mask(
            design_config=design_config,
            max_num_categories=max_num_categories,
            intrinsic_params=intrinsic_params,
            keep_intercept=True
        )
    elif parameter_mask.ndim == 3:
        parameter_mask = parameter_mask[0]

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

    # Pre-generate labels for all rows before compact filtering
    row_ylabels = []
    row_colors  = []
    _r = 0
    if "1" in regressor_keys:
        row_ylabels.append(r"$1$")
        row_colors.append(intercept_color)
        _r += 1
    for key in main_keys:
        for cat_id in range(1, block_width + 1):
            row_ylabels.append(fr"${key}$" + (fr"$ | c_{cat_id}$" if max_num_categories > 2 else ""))
            row_colors.append(main_effect_color)
            _r += 1
            if _r >= num_rows:
                break
        if _r >= num_rows:
            break
    for key in inter_keys:
        for cat_id in range(1, block_width ** 2 + 1):
            row_ylabels.append(fr"${key}$" + (fr"$ | c_{cat_id}$" if max_num_categories > 2 else ""))
            row_colors.append(interaction_color)
            _r += 1
            if _r >= num_rows:
                break
        if _r >= num_rows:
            break

    # Compact: drop rows and columns that are entirely N/A
    row_idx = np.where(np.any(parameter_mask[:num_rows], axis=1))[0]
    col_idx = np.where(np.any(parameter_mask[:num_rows], axis=0))[0]
    parameter_mask = parameter_mask[np.ix_(row_idx, col_idx)]
    true_f  = true[..., row_idx, :][..., col_idx]  # keep original for ref below
    pred_f  = pred[..., row_idx, :][..., col_idx] if pred.ndim >= 3 else pred
    row_ylabels = [row_ylabels[i] for i in row_idx]
    row_colors  = [row_colors[i]  for i in row_idx]
    col_labels  = [variable_names[i] for i in col_idx]
    num_rows    = len(row_idx)
    num_cols    = len(col_idx)

    if figsize is None:
        figsize = (3 * num_cols, 2.8 * num_rows)

    fig, axarr = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)

    for r_f in range(num_rows):
        ylabel = row_ylabels[r_f]
        color  = row_colors[r_f]

        for c_f in range(num_cols):
            ax = axarr[r_f, c_f]
            x = true_f[..., r_f, c_f]
            mask = parameter_mask[r_f, c_f]
            if mask == 1.0:
                if pred_f.ndim == 4:  # Full posterior
                    y = pred_f[..., r_f, c_f]
                    y_mean = y.mean(axis=1)
                    make_quadratic(ax, x, y_mean, color=color)

                    ci = uncertainty_agg(y, prob=0.9, axis=1)
                    y_lo, y_hi = ci[0], ci[1]
                    y_err = np.vstack([y_mean - y_lo, y_hi - y_mean])

                    ax.errorbar(x, y_mean, yerr=y_err, fmt="none", alpha=0.3, linewidth=1.5, color=color)
                    sns.scatterplot(x=x, y=y_mean, ax=ax, color=color, alpha=0.7)

                    corr = np.corrcoef(x, y_mean)[0, 1]
                else:                 # Point estimate
                    y = pred_f[..., r_f, c_f]
                    make_quadratic(ax, x, y, color=color)

                    sns.scatterplot(x=x, y=y, ax=ax, color=color, alpha=0.5)
                    corr = np.corrcoef(x, y)[0, 1]

                metric_label = f"r = {corr:.3f}"
                ax.text(0.1, 0.95, metric_label, ha="left", va="center", transform=ax.transAxes, size=12)

                ax.grid(True, color="lightgray", linestyle="--", linewidth=0.5, alpha=0.2)
                sns.despine(ax=ax)
                ax.set_ylabel(ylabel if c_f == 0 else "", fontsize=label_fontsize)
                ax.set_title(col_labels[c_f] if r_f == 0 else "", fontsize=title_fontsize)
                ax.set_xlabel("Ground Truth" if r_f == num_rows - 1 else "", fontsize=label_fontsize)

            else:
                ax.set_facecolor(color)
                ax.patch.set_alpha(0.05)
                if r_f == num_rows - 1:
                    ax.set_xticks([0.0, 0.5, 1.0])
                    ax.set_xticklabels(['0.0', '0.5', '1.0'])
                    ax.tick_params(axis='x', length=0, labelcolor='none')
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
                    alpha=0.7, color=color,
                )
                ax.set_xlabel("Ground Truth" if r_f == num_rows - 1 else "", fontsize=label_fontsize)

    fig.tight_layout()
    return fig

if __name__ == "__main__":
    debug = True
    color = cogformer_cm_colors()

    if debug:
        design_config = {
            "1": ["v", "a", "z", "tau"],
            "u_1": ["v", "a", "tau"],
            "u_2": ["v", "a", "z"],
            "u_1:u_2": ["v", "a"]
        }

        intrinsic_params = design_config["1"]
        variable_names = [r"$v$", r"$a$", r"$z$", r"$\tau$"]
        num_params = len(intrinsic_params)
        num_regressors = len(list(design_config.keys())) - 1
        num_categories = 3
        batch_size = 10
        true = np.random.normal(1, 1, (batch_size, num_regressors * (num_categories - 1) + 1, num_params))
        pred = np.random.normal(1, 1, (batch_size, num_regressors * (num_categories - 1) + 1, num_params))
        print(true.shape, pred.shape)

        fig = adaptive_recovery(
            true,
            pred,
            design_config,
            intrinsic_params,
            variable_names=variable_names,
            max_num_categories=num_categories,
            intercept_color=color["intercept"],
            main_effect_color=color["main_effect"],
            interaction_color=color["interaction"],
            title_fontsize=18,
            label_fontsize=14,
        )
        fig.savefig("test_recovery.pdf")
        print("groovy")
