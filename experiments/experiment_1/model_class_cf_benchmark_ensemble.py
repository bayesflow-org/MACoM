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
    mask = data["params_mask"]
    return {
        "true": data["true_set"],
        "pred": data["pred_set"],
        "mask": mask[0] if mask.ndim == 3 else mask,
        # The grid the case was SCORED in: cases read from the shared benchmark files are
        # compact (2 categories), self-simulated ones padded to training (4). Older npz
        # files without the key were all compact.
        "cats": int(data["max_num_categories"]) if "max_num_categories" in data.files else 2,
    }


def lift_compact(entry: dict, num_categories: int, num_regressors: int) -> dict:
    """Re-lay a compact-grid case (rows 1, u_1, u_2, u_1 x u_2) onto the padded grid.

    Overlaying cases row-by-INDEX is only valid when they share one layout. In the padded
    grid a continuous regressor fills the first slot of its block and u_1 x u_2 the first
    cell of the (u_1, u_2) interaction block, so the compact rows go to
    [0, 1, 1 + bw, 1 + R*bw] (bw = K - 1). Without this, bench6 pooled the continuous
    designs' u_2 slope with mixed_predictor's u_1:c_2 contrast, and their u_1 x u_2 with
    u_1:c_3.
    """
    bw = num_categories - 1
    n_rows = 1 + num_regressors * bw + (num_regressors * (num_regressors - 1) // 2) * bw * bw
    if entry["mask"].shape[0] != 4:
        raise ValueError(f"expected a compact 4-row grid, got {entry['mask'].shape[0]} rows")
    rows = [0, 1, 1 + bw, 1 + num_regressors * bw]

    def scatter(a, row_axis):
        shape = list(a.shape)
        shape[row_axis] = n_rows
        out = np.zeros(shape, dtype=a.dtype)
        idx = [slice(None)] * a.ndim
        idx[row_axis] = rows
        out[tuple(idx)] = a
        return out

    return {"true": scatter(entry["true"], 1), "pred": scatter(entry["pred"], 2),
            "mask": scatter(entry["mask"], 0), "cats": num_categories}


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

            # One row layout per ensemble: the cases' own grid when they share it (bench5,
            # all compact), else every compact case lifted onto the padded grid (bench6).
            grid_cats = {loaded[c]["cats"] for c in cases}
            if len(grid_cats) == 1:
                entries = [loaded[c] for c in cases]
                num_cats = grid_cats.pop()
            else:
                num_cats = args.max_num_categories
                entries = [loaded[c] if loaded[c]["cats"] == num_cats else
                           lift_compact(loaded[c], num_cats, args.max_num_regressors)
                           for c in cases]
                padded_rows = {e["mask"].shape[0] for e in entries}
                if len(padded_rows) != 1:
                    raise ValueError(f"{family} {variant}: row layouts still differ: {padded_rows}")
            true_list = [e["true"] for e in entries]
            pred_list = [e["pred"] for e in entries]
            masks = [e["mask"] for e in entries]
            design_configs = [scenarios[c][0] for c in cases]
            categorical_list = [scenarios[c][1].get("categorical_regressors", []) for c in cases]
            colors = [case_colors[c] for c in cases]

            shared = dict(
                true_list=true_list,
                pred_list=pred_list,
                design_configs=design_configs,
                intrinsic_params=intrinsic_params,
                max_num_categories=num_cats,
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
