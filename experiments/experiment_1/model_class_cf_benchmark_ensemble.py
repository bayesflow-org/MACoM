"""Benchmark-case ensemble diagnostics for the model_class CogFormer.

Overlays the benchmark validation cases of each model family on a shared
coefficient-matrix grid, reading the per-case predictions saved by
``model_class_cf_validate.py`` (``<pred_dir>/<family>_<case>_cf_pred.npz``) —
run that script first.

Two variants are produced per family and plot type:

- ``bench5`` — the five OG benchmark cases (intercept_only, fixed, regressed,
  fixed_regressed, interaction), all-continuous regressors.
- ``bench6`` — the same plus ``mixed_predictor`` (u_1 categorical at the
  training category budget), whose contrast rows land in the same main-effect /
  interaction blocks of the coefficient matrix.

Plot types: recovery, ECDF, coverage, metrics — each with a separate global
(``pi``) companion where a global parameter is active.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from cogformer.diagnostics.plot.ensemble_recovery import ensemble_recovery, global_ensemble_recovery
from cogformer.diagnostics.plot.ensemble_coverage import ensemble_coverage, global_ensemble_coverage
from cogformer.diagnostics.plot.ensemble_ecdf import ensemble_ecdf, global_ensemble_ecdf
from cogformer.diagnostics.plot.ensemble_metrics import ensemble_metrics, global_ensemble_metrics

from ..configs import MODELS as _BASE_MODELS
from ..utils import add_global_params

GLOBAL_PARAMS = ["pi"]
GLOBAL_VARIABLE_NAMES = [r"$\pi$"]

_MODELS_WITH_PI = {
    m: add_global_params(cfg, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
    for m, cfg in _BASE_MODELS.items()
    if m != "lba"
}

OG_CASES = ["intercept_only", "fixed", "regressed", "fixed_regressed", "interaction"]
MIXED_CASE = "mixed_predictor"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", type=str, default="./outputs/data/model_class/",
                   help="Directory with <family>_<case>_cf_pred.npz saved by model_class_cf_validate.py")
    p.add_argument("--outdir", type=str,
                   default="./outputs/figures/model_class/cf/ensemble_benchmark/")
    p.add_argument("--max_num_regressors", type=int, default=5,
                   help="Must match the validation run (the mask row layout is padded to it)")
    p.add_argument("--max_num_categories", type=int, default=5,
                   help="Must match the validation run (sizes the row blocks)")
    p.add_argument("--num_null_draws", type=int, default=1000)
    p.add_argument("--metrics", action=argparse.BooleanOptionalAction, default=True,
                   help="Also produce the (slower) metrics heatmaps")
    return p.parse_args()


def load_case(pred_dir: Path, family: str, case: str) -> dict | None:
    path = pred_dir / f"{family}_{case}_cf_pred.npz"
    if not path.exists():
        logging.warning(f"[missing] {path} — run model_class_cf_validate.py first; skipping {family}/{case}")
        return None
    data = np.load(path)
    return {
        "true": data["true_set"],
        "pred": data["pred_set"],
        "mask": data["params_mask"],
    }


def main():
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # One stable color per benchmark case, shared across the 5- and 6-case variants.
    all_cases = OG_CASES + [MIXED_CASE]
    palette = sns.husl_palette(len(all_cases), h=0.05, l=0.72)
    case_colors = dict(zip(all_cases, palette))

    plotters = {
        "recovery": (ensemble_recovery, global_ensemble_recovery, {}),
        "ecdf":     (ensemble_ecdf, global_ensemble_ecdf, {"difference": True}),
        "coverage": (ensemble_coverage, global_ensemble_coverage, {"difference": True}),
    }
    if args.metrics:
        plotters["metrics"] = (ensemble_metrics, global_ensemble_metrics,
                               {"num_null_draws": args.num_null_draws})

    for family, cfg in _MODELS_WITH_PI.items():
        scenarios = {name: (dc, opts) for name, dc, opts in cfg["val_scenarios"]}
        intrinsic_params = cfg["intrinsic_params"]
        variable_names = cfg["variable_names"]

        loaded = {}
        for case in all_cases:
            if case not in scenarios:
                continue
            entry = load_case(pred_dir, family, case)
            if entry is not None:
                loaded[case] = entry

        for variant, case_names in (("bench5", OG_CASES), ("bench6", all_cases)):
            cases = [c for c in case_names if c in loaded]
            if len(cases) < 2:
                logging.warning(f"[skip] {family} {variant}: only {len(cases)} case(s) available")
                continue

            true_list = [loaded[c]["true"] for c in cases]
            pred_list = [loaded[c]["pred"] for c in cases]
            masks = [loaded[c]["mask"] for c in cases]
            design_configs = [scenarios[c][0] for c in cases]
            categorical_list = [scenarios[c][1].get("categorical_regressors", []) for c in cases]
            colors = [case_colors[c] for c in cases]

            shared = dict(
                true_list=true_list,
                pred_list=pred_list,
                design_configs=design_configs,
                intrinsic_params=intrinsic_params,
                max_num_categories=args.max_num_categories,
                parameter_masks=masks,
                variable_names=variable_names,
                max_num_regressors=args.max_num_regressors,
                categorical_regressors_list=categorical_list,
                global_params=GLOBAL_PARAMS,
            )
            color_kwargs = {"colors": colors, "labels": cases}

            for plot_name, (plot_fn, global_fn, extra) in plotters.items():
                kwargs = dict(shared, **extra)
                if plot_fn is not ensemble_metrics:
                    kwargs.update(color_kwargs)

                logging.info(f"Plotting {family} {variant} ensemble {plot_name} ({len(cases)} cases)...")
                fig = plot_fn(**kwargs)
                if fig is not None:
                    path = outdir / f"{family}_{variant}_ensemble_{plot_name}.pdf"
                    fig.savefig(path, bbox_inches="tight")
                    plt.close(fig)
                    logging.info(f"[saved] {path}")

                glob_fig = global_fn(**kwargs)
                if glob_fig is not None:
                    path = outdir / f"{family}_{variant}_ensemble_{plot_name}_global.pdf"
                    glob_fig.savefig(path, bbox_inches="tight")
                    plt.close(glob_fig)
                    logging.info(f"[saved] {path}")

    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
