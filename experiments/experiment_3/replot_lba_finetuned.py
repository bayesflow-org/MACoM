"""Redraw the LBA finetuned diagnostics from saved predictions -- no GPU, no sampling.

WHY THIS EXISTS
---------------
`model_class_lba_finetune_validate.py` costs a full ODE-integration pass over four
families x six scenarios, so changing how its figures LOOK must not require running
it again. Everything these four plot types draw is already in
`<pred_dir>/lba_<scenario>_ft_pred.npz` (true_set, pred_set, params_mask); the rest
is design metadata reconstructed from `experiments/configs.py`, which needs no model
and no checkpoint. Same contract as `replot_design_eig.py`.

SCOPE: LBA ONLY, deliberately
-----------------------------
The validate script sets `colors = cogformer_mc_colors()` ONCE, globally, outside the
family loop -- so recolouring by editing that line would repaint DDM, RDM and CDM too.
This script selects the palette per family via `family_colors()` and only ever walks
the LBA scenarios, so the sibling families in `finetuned/` are left untouched.

The plotting functions are imported from the same modules the real run uses rather
than reimplemented, so a replot and a real run cannot drift apart.

    python -m experiments.experiment_3.replot_lba_finetuned --dry_run
    python -m experiments.experiment_3.replot_lba_finetuned
"""
import argparse
import logging
from pathlib import Path

import numpy as np
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
from cogformer.utils.plot_utils import family_colors

from ..utils import resolve_val_scenario
# Import the LBA config from the validate module itself rather than rebuilding it:
# `_LBA_WITH_PI` is what the real run plots against, so taking it from the source
# makes it impossible for the replot to disagree about intrinsic_params, variable
# names or val_scenarios.
from .model_class_lba_finetune_validate import GLOBAL_PARAMS, _LBA_WITH_PI

FAMILY = "lba"
ARM = "ft"

PAIRED = Path("outputs/figures/model_class/cf/paired_lba_20260816/finetuned")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", type=str, default=str(PAIRED / "predictions"),
                   help="Directory holding lba_<scenario>_ft_pred.npz")
    p.add_argument("--outdir", type=str, default=str(PAIRED / FAMILY),
                   help="Root of the per-scenario figure directories to overwrite")
    # The TRAINING max, not the scenario's resolved value. The diagnostic plot
    # functions size their row blocks by this, and passing the scenario value
    # silently truncates rows rather than erroring.
    p.add_argument("--max_num_categories", type=int, default=4)
    p.add_argument("--max_num_regressors", type=int, default=4)
    p.add_argument("--skip_log_gamma", action="store_true", default=True)
    p.add_argument("--scenarios", nargs="+", default=None,
                   help="Restrict to these scenario names (default: all found)")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = _LBA_WITH_PI
    intrinsic_params = cfg["intrinsic_params"]
    variable_names = cfg["variable_names"]

    colors = family_colors(FAMILY)
    logging.info(f"palette family_colors({FAMILY!r}) = {colors}")

    pred_dir, outroot = Path(args.pred_dir), Path(args.outdir)
    written, skipped = 0, []

    for cfg_name, design_config, scenario_opts in cfg["val_scenarios"]:
        if args.scenarios and cfg_name not in args.scenarios:
            continue
        _, _, _, skip = resolve_val_scenario(scenario_opts, args.max_num_categories)
        n_main = sum(1 for k in design_config if k != "1" and ":" not in k)
        if skip or n_main > args.max_num_regressors:
            skipped.append(f"{cfg_name} (scenario exceeds training geometry)")
            continue

        tag = f"{FAMILY}_{cfg_name}_{ARM}"
        npz = pred_dir / f"{tag}_pred.npz"
        if not npz.exists():
            skipped.append(f"{cfg_name} (no {npz.name})")
            continue

        d = np.load(npz, allow_pickle=True)
        true_set, pred_set, params_mask = d["true_set"], d["pred_set"], d["params_mask"]
        outdir = outroot / cfg_name
        logging.info(f"[{cfg_name}] {pred_set.shape} -> {outdir}")
        if args.dry_run:
            written += 1
            continue
        outdir.mkdir(parents=True, exist_ok=True)

        common = dict(
            true=true_set, pred=pred_set, design_config=design_config,
            intrinsic_params=intrinsic_params,
            max_num_categories=args.max_num_categories,
            parameter_mask=params_mask, variable_names=variable_names,
            global_params=GLOBAL_PARAMS,
        )
        for fn, name, kw in [
            (adaptive_recovery,     "recovery", {}),
            (adaptive_coverage,     "coverage", {}),
            (adaptive_ecdf,         "ecdf",     {"difference": True}),
            (plot_adaptive_metrics, "metrics",  {"skip_log_gamma": args.skip_log_gamma}),
        ]:
            fig = fn(**common,
                     intercept_color=colors["intercept"],
                     main_effect_color=colors["main_effect"],
                     interaction_color=colors["interaction"],
                     **kw)
            fig.savefig(outdir / f"{tag}_{name}.pdf", bbox_inches="tight")
            plt.close(fig)
            written += 1

        for gfn, name, kw in [
            (global_recovery,     "recovery", {}),
            (global_coverage,     "coverage", {}),
            (global_ecdf,         "ecdf",     {"difference": True}),
            (plot_global_metrics, "metrics",  {"skip_log_gamma": args.skip_log_gamma}),
        ]:
            gfig = gfn(**common, global_color=colors["intercept"], **kw)
            if gfig is not None:
                gfig.savefig(outdir / f"{tag}_{name}_global.pdf", bbox_inches="tight")
                plt.close(gfig)
                written += 1

    for s in skipped:
        logging.info(f"  skipped: {s}")
    verb = "would write" if args.dry_run else "wrote"
    logging.info(f"\n{verb} {written} figures with the {FAMILY.upper()} "
                 f"identity palette ({colors['main_effect']})")
    if not args.dry_run:
        logging.info("re-run `bash run_experiments.sh e3_finalize` to propagate to finals/")


if __name__ == "__main__":
    main()
