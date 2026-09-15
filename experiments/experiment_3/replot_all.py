"""Redraw every paired-validation diagnostic in Experiment 3 -- font only, no GPU.

These figures were produced before the project vendored Inter, so they are the last
DejaVu holdouts in outputs/finals. Everything they draw is a function of the saved
posterior draws, so the whole tree can be redrawn on CPU; the typography then comes
from cogformer.utils.style, which is applied at package import.

TWO THINGS THIS SCRIPT HAS TO GET RIGHT, AND CHECKS RATHER THAN ASSUMES
-----------------------------------------------------------------------
GEOMETRY. The grid a cell lands in depends on max_num_categories and on whether
max_num_regressors reaches the plotter at all, and the three arms do not agree:
run_lba.sh passes neither flag, and the LBA validate scripts then forward only
max_num_categories to the plot functions, leaving max_num_regressors at None, while
the script behind the `original` arm forwards both. Get it wrong and rows are
silently relabelled -- `$u_5$` instead of `$u_1 \\times u_2$` -- with no error.
So every case is VERIFIED before it is drawn: the saved metrics.csv records the
active-cell labels that run produced, and a case whose labels do not reproduce is
skipped and reported rather than redrawn wrong.

PALETTE. The tree is not uniformly coloured and must not be made so by accident.
`finetuned/lba` was recoloured to LBA's identity palette by replot_lba_finetuned.py;
every other arm, including LBA's own ZERO-SHOT arm, is still on the model-class
palette the validate scripts set globally. That is a real inconsistency, but it is
not this script's to fix -- a font pass that also repainted 384 figures would be
impossible to review. Each case keeps exactly the palette it has.

FONT SIZE. These figures are 3 in per column, so a six-column scenario is 18 in wide
and `\\includegraphics[width=\\textwidth]` scales it by 5.5/18 = 0.31 -- which put a
20 pt title at 6 pt on the page. Sizes are scaled by width/11 in, the width this
project draws at, so that every figure lands at the same on-page size whatever its
column count. The scale is floored at 1.0: narrow figures keep today's sizes, since
the defect being fixed is wide figures rendering small.

    python -m experiments.experiment_3.replot_all --dry_run
    python -m experiments.experiment_3.replot_all
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cogformer.diagnostics.plot.adaptive_recovery import adaptive_recovery, global_recovery
from cogformer.diagnostics.plot.adaptive_coverage import adaptive_coverage, global_coverage
from cogformer.diagnostics.plot.adaptive_ecdf import adaptive_ecdf, global_ecdf
from cogformer.diagnostics.plot.adaptive_metrics import (
    adaptive_metrics as plot_adaptive_metrics,
    global_metrics as plot_global_metrics,
)
from cogformer.diagnostics.metric.adaptive_metrics import adaptive_metrics as compute_metrics
from cogformer.utils.plot_utils import family_colors, cogformer_mc_colors

from .model_class_lba_finetune_validate import (
    GLOBAL_PARAMS, _LBA_WITH_PI, _MODELS_WITH_PI)

SRC = Path("outputs/figures/model_class/cf/paired_lba_20260816")
FINALS = Path("outputs/finals/experiment-3")
CFG = {"lba": _LBA_WITH_PI, **{k: _MODELS_WITH_PI[k] for k in ("ddm", "rdm", "cdm")}}

# arm -> (families, PREDICTION-file infix, FIGURE-file infix, max_num_regressors as
# that arm's validate script forwarded it to the plotters).
#
# The two infixes are NOT the same, and assuming they were wrote 192 files under new
# names beside the real ones instead of replacing them -- silently, since both the
# write and the later assembly succeeded. zeroshot's predictions are
# `lba_<scen>_mid1_cf_pred.npz` while its figures are `lba_<scen>_mid1_*`; original's
# are `<fam>_<scen>_cf_pred.npz` against figures `<fam>_<scen>_fm_*`. Only finetuned
# uses one infix for both.
ARMS = {
    "zeroshot":  (["lba"], "mid1_cf", "mid1", None),
    "finetuned": (["ddm", "rdm", "cdm", "lba"], "ft", "ft", None),
    "original":  (["ddm", "rdm", "cdm"], "cf", "fm", 4),
}
MAX_CATS = 4          # verified against all 48 metrics.csv indices
BASE_W_PER_COL = 3.0  # inches, the widest of the four plot types
TARGET_W = 11.0       # the width this project draws at


def scenario_dir(arm, fam, scen):
    """Where this arm's per-scenario figures live in the SOURCE tree.

    zeroshot has no family level -- it is LBA only -- but it DOES nest a variant
    directory below the scenario, which the other two arms do not. Writing to
    `zeroshot/<scen>/` instead of `zeroshot/<scen>/mid_rdm/` puts the figures
    somewhere finalize_experiment_3 does not look, so the replot appears to succeed
    and changes nothing downstream.
    """
    if arm == "zeroshot":
        return SRC / arm / scen / "mid_rdm"
    return SRC / arm / fam / scen


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--skip_log_gamma", action="store_true", default=True)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    a = parse_args()
    mc = cogformer_mc_colors()
    lba = family_colors("lba")

    written, skipped, verified = 0, [], 0
    for arm in a.arms:
        fams, pred_tag, fig_tag, regs = ARMS[arm]
        for fam in fams:
            cfg = CFG[fam]
            for scen, design_config, _ in cfg["val_scenarios"]:
                tag = f"{fam}_{scen}_{fig_tag}"
                npz = SRC / arm / "predictions" / f"{fam}_{scen}_{pred_tag}_pred.npz"
                csv = FINALS / arm / fam / scen / "metrics.csv"
                if not npz.exists() or not csv.exists():
                    skipped.append(f"{arm}/{fam}/{scen} (no predictions or metrics.csv)")
                    continue
                d = np.load(npz, allow_pickle=True)
                true_set, pred_set = d["true_set"], d["pred_set"]
                mask = d["params_mask"]

                common = dict(
                    true=true_set, pred=pred_set, design_config=design_config,
                    intrinsic_params=cfg["intrinsic_params"],
                    max_num_categories=MAX_CATS, parameter_mask=mask,
                    variable_names=cfg["variable_names"],
                    global_params=GLOBAL_PARAMS, max_num_regressors=regs)

                # VERIFY: reproduce the labels the original run recorded, or skip.
                got = list(compute_metrics(**common).index)
                want = list(pd.read_csv(csv, index_col=0).index)
                if got != want:
                    skipped.append(f"{arm}/{fam}/{scen} (labels differ from metrics.csv)")
                    continue
                verified += 1

                # Font scale from the figure's own width. num_cols counts active,
                # non-global columns, which is what the plotters compact down to.
                gcols = {i for i, p in enumerate(cfg["intrinsic_params"])
                         if p in GLOBAL_PARAMS}
                num_cols = sum(1 for c in range(mask.shape[1])
                               if c not in gcols and mask[:, c].any())
                s = max(1.0, BASE_W_PER_COL * num_cols / TARGET_W)

                if a.dry_run:
                    logging.info(f"  would redraw {arm}/{fam}/{scen} "
                                 f"({num_cols} cols, font x{s:.2f}) -> "
                                 f"{scenario_dir(arm, fam, scen)}")
                    written += 8
                    continue
                outdir = scenario_dir(arm, fam, scen)
                outdir.mkdir(parents=True, exist_ok=True)
                pal = lba if (arm == "finetuned" and fam == "lba") else mc
                kinds = dict(intercept_color=pal["intercept"],
                             main_effect_color=pal["main_effect"],
                             interaction_color=pal["interaction"])

                # Tick size is not a parameter of adaptive_recovery, so it is scaled
                # through rcParams, which covers every plot type uniformly.
                with plt.rc_context({"xtick.labelsize": 10 * s,
                                     "ytick.labelsize": 10 * s}):
                    for fn, name, kw in (
                        (adaptive_recovery, "recovery",
                         dict(title_fontsize=20 * s, label_fontsize=14 * s)),
                        (adaptive_coverage, "coverage",
                         dict(title_fontsize=20 * s, label_fontsize=14 * s,
                              tick_fontsize=11 * s, legend_fontsize=12 * s)),
                        (adaptive_ecdf, "ecdf",
                         dict(difference=True, title_fontsize=20 * s,
                              label_fontsize=14 * s, tick_fontsize=11 * s,
                              legend_fontsize=12 * s)),
                        (plot_adaptive_metrics, "metrics",
                         dict(skip_log_gamma=a.skip_log_gamma,
                              title_fontsize=16 * s, label_fontsize=12 * s,
                              tick_fontsize=10 * s, annot_fontsize=10 * s)),
                    ):
                        fig = fn(**common, **kinds, **kw)
                        if fig is None:
                            continue
                        fig.savefig(outdir / f"{tag}_{name}.pdf", bbox_inches="tight")
                        plt.close(fig)
                        written += 1

                    for gfn, name, kw in (
                        (global_recovery, "recovery",
                         dict(title_fontsize=20 * s, label_fontsize=14 * s)),
                        (global_coverage, "coverage", {}),
                        (global_ecdf, "ecdf", dict(difference=True)),
                        (plot_global_metrics, "metrics",
                         dict(skip_log_gamma=a.skip_log_gamma)),
                    ):
                        gfig = gfn(**common, global_color=pal["intercept"], **kw)
                        if gfig is not None:
                            gfig.savefig(outdir / f"{tag}_{name}_global.pdf",
                                         bbox_inches="tight")
                            plt.close(gfig)
                            written += 1
                logging.info(f"  {arm}/{fam}/{scen}: {num_cols} cols, font x{s:.2f}")

    verb = "would write" if a.dry_run else "wrote"
    logging.info(f"\n{verb} {written} figures from {verified} verified cases")
    for s_ in skipped:
        logging.info(f"  SKIPPED {s_}")
    if not a.dry_run:
        logging.info("run `python -m experiments.finalize_experiment_3` to propagate")


if __name__ == "__main__":
    main()
