"""Diagnostic plots for real-data CogFormer posteriors.

Generates four figures saved under outputs/figures/real_data/:
  ddm_posteriors.png         — KDE grid of per-subject posterior medians (mc vs mf)
  cdm_posteriors.png         — same for CDM
  ddm_mc_vs_mf_scatter.png   — intercept-row mc vs mf posterior means per subject
  cdm_mc_vs_mf_scatter.png   — same for CDM
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde, wasserstein_distance
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "outputs/data/real_data"
FIG_DIR = ROOT / "outputs/figures/real_data"

MC_COLOR = "#4E2A84"
MF_COLOR = "#007396"
NA_COLOR = "#aaaaaa"

PARAM_MATH = {
    "v":       r"$v$",
    "a":       r"$a$",
    "z":       r"$z$",
    "tau":     r"$\tau$",
    "s_v":     r"$\sigma_v$",
    "s_tau":   r"$\sigma_\tau$",
    "pi":      r"$\pi$",
    "v_theta": r"$v_\theta$",
    "v_diff":  r"$\Delta v$",
}

# Active design rows: (row_index_in_181, row_label, active_param_names)
# Row layout (max_num_regressors=5, max_num_categories=5, keep_intercept=True,
# block_width=4, inter_block_width=16):
#   col 0: intercept
#   cols 1-4:   u_1 block (width 4)
#   cols 5-8:   u_2 block
#   cols 9-24:  u_1:u_2 block (width 16)

DDM_DESIGN = [
    (0,  "Intercept",                   ["v", "a", "z", "tau", "s_v", "s_tau", "pi"]),
    (1,  r"Stim type ($u_1$)",          ["v", "a", "z", "tau", "s_tau"]),
    (5,  r"Truth ($u_2$)",              ["v", "a", "z", "tau", "s_tau"]),
    (9,  r"$u_1 \times u_2$",           ["v", "a", "z", "tau", "s_tau"]),
]

CDM_DESIGN = [
    (0,  "Intercept",                      ["v", "v_theta", "a", "tau", "s_v", "s_tau", "pi"]),
    (1,  r"Speed ($u_1$)",                 ["v", "a", "tau"]),
    (5,  r"Difficulty $d_2$ ($u_2$)",      ["v", "a", "tau"]),
    (6,  r"Difficulty $d_3$ ($u_2$)",      ["v", "a", "tau"]),
    (9,  r"Speed $\times$ $d_2$",          ["v", "a", "tau"]),
    (10, r"Speed $\times$ $d_3$",          ["v", "a", "tau"]),
]


def _param_idx(param_names_list, name):
    return list(param_names_list).index(name)


def _per_subject_medians(posteriors, row_idx, param_names, all_param_names):
    """Return dict: param_name -> (n_subjects,) array of posterior medians."""
    out = {}
    for pname in param_names:
        pidx = _param_idx(all_param_names, pname)
        out[pname] = np.median(posteriors[:, :, row_idx, pidx], axis=1)
    return out


def _kde_plot(ax, data, color, label, fill_alpha=0.15, n_pts=300):
    """KDE of data on ax with filled area. Handles near-constant data gracefully."""
    data = np.asarray(data, dtype=float)
    spread = data.max() - data.min()
    if spread < 1e-10:
        ax.axvline(data[0], color=color, lw=1.5, label=label)
        return
    kde = gaussian_kde(data, bw_method="silverman")
    margin = spread * 0.3
    xs = np.linspace(data.min() - margin, data.max() + margin, n_pts)
    ys = kde(xs)
    ax.plot(xs, ys, color=color, lw=1.8, label=label)
    ax.fill_between(xs, ys, alpha=fill_alpha, color=color)


def posterior_summary_figure(
    mc_file, mf_file,
    design_rows, mc_param_key, mf_param_key,
    title, savepath,
):
    mc = np.load(mc_file, allow_pickle=True)
    mf = np.load(mf_file, allow_pickle=True)
    mc_posteriors = mc["posteriors"]   # (n_subj, n_samp, 181, n_mc_params)
    mf_posteriors = mf["posteriors"]   # (n_subj, n_samp, 181, n_mf_params)
    mc_pnames = list(mc[mc_param_key])
    mf_pnames = list(mf[mf_param_key])

    n_design_rows = len(design_rows)
    all_params = list(dict.fromkeys(p for _, _, ps in design_rows for p in ps))
    n_cols = len(all_params)

    fig, axes = plt.subplots(
        n_design_rows, n_cols,
        figsize=(2.4 * n_cols, 2.2 * n_design_rows),
        squeeze=False,
    )

    for i, (row_idx, row_label, param_names) in enumerate(design_rows):
        mc_med = _per_subject_medians(mc_posteriors, row_idx, param_names, mc_pnames)
        mf_med = _per_subject_medians(mf_posteriors, row_idx, param_names, mf_pnames)

        for j, pname in enumerate(all_params):
            ax = axes[i, j]
            col_title = PARAM_MATH.get(pname, f"${pname}$")

            if pname not in param_names:
                ax.set_facecolor(NA_COLOR)
                ax.patch.set_alpha(0.08)
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                ax.text(
                    0.5, 0.5, "N/A",
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=11, weight="bold",
                    alpha=0.5, color=NA_COLOR,
                )
                if i == 0:
                    ax.set_title(col_title, fontsize=9)
                continue

            mc_vals = mc_med[pname]
            mf_vals = mf_med[pname]

            _kde_plot(ax, mc_vals, MC_COLOR, "mc")
            _kde_plot(ax, mf_vals, MF_COLOR, "mf")

            ax.axvline(np.mean(mc_vals), color=MC_COLOR, lw=1.0, ls="--", alpha=0.8)
            ax.axvline(np.mean(mf_vals), color=MF_COLOR, lw=1.0, ls="--", alpha=0.8)
            ax.axvline(0, color="gray", lw=0.6, ls="--", alpha=0.4)

            wd = wasserstein_distance(mc_vals, mf_vals)
            ax.text(0.04, 0.96, f"W={wd:.3f}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=6.5, color="black")

            ax.set_yticks([])
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="x", labelsize=7)

            if i == 0:
                ax.set_title(col_title, fontsize=9)
            if j == 0:
                ax.set_ylabel(row_label, fontsize=8)

    mc_patch = mpatches.Patch(color=MC_COLOR, alpha=0.7, label="ModelClass (mc)")
    mf_patch = mpatches.Patch(color=MF_COLOR, alpha=0.7, label="ModelFamily (mf)")
    fig.legend(
        handles=[mc_patch, mf_patch],
        fontsize=8, loc="lower center",
        ncol=2, bbox_to_anchor=(0.5, 0),
        frameon=False,
    )

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(savepath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {savepath}")


def mc_vs_mf_scatter(
    mc_file, mf_file,
    mc_param_key, mf_param_key,
    intercept_param_names, title, savepath,
):
    mc = np.load(mc_file, allow_pickle=True)
    mf = np.load(mf_file, allow_pickle=True)
    mc_posteriors = mc["posteriors"]
    mf_posteriors = mf["posteriors"]
    mc_pnames = list(mc[mc_param_key])
    mf_pnames = list(mf[mf_param_key])

    n_params = len(intercept_param_names)
    fig, axes = plt.subplots(1, n_params, figsize=(2.6 * n_params, 2.8))

    for j, pname in enumerate(intercept_param_names):
        ax = axes[j]
        mc_idx = _param_idx(mc_pnames, pname)
        mf_idx = _param_idx(mf_pnames, pname)

        mc_means = np.mean(mc_posteriors[:, :, 0, mc_idx], axis=1)
        mf_means = np.mean(mf_posteriors[:, :, 0, mf_idx], axis=1)

        ax.scatter(mc_means, mf_means, alpha=0.55, s=18, color="#333333")

        lo = min(mc_means.min(), mf_means.min())
        hi = max(mc_means.max(), mf_means.max())
        margin = (hi - lo) * 0.08 or 0.05
        lims = [lo - margin, hi + margin]
        ax.plot(lims, lims, color="tomato", lw=0.9, ls="--", alpha=0.7)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        r = np.corrcoef(mc_means, mf_means)[0, 1]
        col_title = PARAM_MATH.get(pname, f"${pname}$")
        ax.set_title(f"{col_title}\nr={r:.2f}", fontsize=9)
        ax.set_xlabel("mc", fontsize=8)
        ax.set_ylabel("mf", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, lw=0.5)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(savepath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {savepath}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # DDM summary
    posterior_summary_figure(
        mc_file=DATA_DIR / "ddm/mc/ddm_mc_posteriors.npz",
        mf_file=DATA_DIR / "ddm/mf/ddm_mf_posteriors.npz",
        design_rows=DDM_DESIGN,
        mc_param_key="global_param_names",
        mf_param_key="intrinsic_params",
        title="DDM Real Data Posteriors (mc vs mf)  —  Stump et al., 2026",
        savepath=FIG_DIR / "ddm_posteriors.png",
    )

    # CDM summary
    posterior_summary_figure(
        mc_file=DATA_DIR / "cdm/mc/cdm_mc_posteriors.npz",
        mf_file=DATA_DIR / "cdm/mf/cdm_mf_posteriors.npz",
        design_rows=CDM_DESIGN,
        mc_param_key="global_param_names",
        mf_param_key="intrinsic_params",
        title="CDM Real Data Posteriors (mc vs mf)  —  Kvam et al., 2019",
        savepath=FIG_DIR / "cdm_posteriors.png",
    )

    # DDM mc vs mf scatter (intercept row)
    mc_vs_mf_scatter(
        mc_file=DATA_DIR / "ddm/mc/ddm_mc_posteriors.npz",
        mf_file=DATA_DIR / "ddm/mf/ddm_mf_posteriors.npz",
        mc_param_key="global_param_names",
        mf_param_key="intrinsic_params",
        intercept_param_names=["v", "a", "z", "tau", "s_v", "s_tau", "pi"],
        title="DDM Intercept: mc vs mf posterior means per subject",
        savepath=FIG_DIR / "ddm_mc_vs_mf_scatter.png",
    )

    # CDM mc vs mf scatter (intercept row)
    mc_vs_mf_scatter(
        mc_file=DATA_DIR / "cdm/mc/cdm_mc_posteriors.npz",
        mf_file=DATA_DIR / "cdm/mf/cdm_mf_posteriors.npz",
        mc_param_key="global_param_names",
        mf_param_key="intrinsic_params",
        intercept_param_names=["v", "v_theta", "a", "tau", "s_v", "s_tau", "pi"],
        title="CDM Intercept: mc vs mf posterior means per subject",
        savepath=FIG_DIR / "cdm_mc_vs_mf_scatter.png",
    )


if __name__ == "__main__":
    main()
