"""Overlay saved BayesFlow benchmark diagnostics on CogFormer's ensemble grids.

Reads <family>_<case>_data.npz from model_family_bf_validate, without loading
checkpoints or rerunning inference. Each family gets recovery, ECDF, coverage,
and metric PDFs for the five continuous benchmark cases, with pi separate.
"""
from experiments.paths import RESULTS_ROOT

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from cogformer.diagnostics.plot.ensemble_coverage import ensemble_coverage, global_ensemble_coverage
from cogformer.diagnostics.plot.ensemble_ecdf import ensemble_ecdf, global_ensemble_ecdf
from cogformer.diagnostics.plot.ensemble_metrics import ensemble_metrics, global_ensemble_metrics
from cogformer.diagnostics.plot.ensemble_recovery import ensemble_recovery, global_ensemble_recovery

from ..configs import MODELS as BASE_MODELS, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES
from ..utils import add_global_params, reshape_bf_to_cf


CASES = ["intercept_only", "fixed", "regressed", "fixed_regressed", "interaction"]
FAMILIES = ["ddm", "rdm", "cdm", "lba"]
GLOBAL_SHARE = {"recovery": {"share_limits": True},
                "ecdf": {"share_axes": True}, "coverage": {"share_axes": True}}


def load_case(data_dir: Path, family: str, case: str, cfg: dict) -> dict:
    """Select active flat cells and map them to the common four-row grid.

    Intercept-only files contain one row, whereas regression files contain the
    full coefficient grid. Validate both layouts against the design before
    unpacking: an incorrect mask could otherwise silently relabel coefficients.
    """
    path = data_dir / f"{family}_{case}_data.npz"
    with np.load(path) as data:
        truth, pred, masks = (data[k] for k in ("true_set", "pred_set", "param_masks"))
    if (truth.ndim != 2 or pred.ndim != 3 or truth.shape[0] < 2
            or pred.shape[1] < 2 or pred.shape[::2] != truth.shape):
        raise ValueError(f"{path}: expected truth (datasets, cells) and draws (datasets, draws, cells)")
    if masks.shape != truth.shape or not np.all(masks == masks[0]):
        raise ValueError(f"{path}: expected one identical parameter mask per dataset")

    design = cfg["benchmark_configs"][case]
    params = cfg["intrinsic_params"]
    grid_mask = np.array([[p in active for p in params] for active in design.values()])
    source_rows, remainder = divmod(truth.shape[-1], len(params))
    if (remainder or source_rows not in (1, len(design))
            or np.any(grid_mask[source_rows:])
            or not np.array_equal(masks[0], grid_mask[:source_rows].ravel())):
        raise ValueError(f"{path}: saved parameter mask does not match the benchmark design")
    active = masks[0].astype(bool)
    if not np.all(np.isfinite(truth[:, active])) or not np.all(np.isfinite(pred[:, :, active])):
        raise ValueError(f"{path}: nonfinite values in active parameters")

    return {
        "true": reshape_bf_to_cf(truth[:, active], design, params),
        "pred": reshape_bf_to_cf(pred[:, :, active], design, params),
        "mask": grid_mask,
        "source": str(path.resolve()),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data_dir", type=Path, default=(RESULTS_ROOT / "data/model_family"))
    p.add_argument("--outdir", type=Path,
                   default=(RESULTS_ROOT / "figures/model_family/bf/ensemble_benchmark"))
    p.add_argument("--families", nargs="+", choices=FAMILIES, default=FAMILIES)
    p.add_argument("--metrics", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num_null_draws", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0, help="Seed for the log-gamma null distribution")
    args = p.parse_args()
    if args.num_null_draws < 1:
        p.error("--num_null_draws must be positive")
    if not 0 <= args.seed < 2**32:
        p.error("--seed must be between 0 and 2**32 - 1")
    return args


def plot_bench5(cfg: dict, entries: list[dict], outdir: Path, family: str, *,
                metrics: bool = True, num_null_draws: int = 1000, seed: int = 0) -> list[str]:
    """Draw one family's bench5 ensemble set; return the saved file names.

    ``entries`` are the five CASES in order, each {"true", "pred", "mask"} on the
    compact four-row grid. Shared with experiment 3's CogFormer LBA ensembles so both
    methods' figures use the same palette, labels, metric normalisation, null seed,
    shared global axes and file names.
    """
    # Six colors matches CogFormer's bench5/bench6 palette exactly; use its first five.
    colors = sns.husl_palette(6, h=0.05, l=0.72)[:len(CASES)]
    plotters = {
        "recovery": (ensemble_recovery, global_ensemble_recovery, {}),
        "ecdf": (ensemble_ecdf, global_ensemble_ecdf, {"difference": True}),
        "coverage": (ensemble_coverage, global_ensemble_coverage, {"difference": True}),
    }
    if metrics:
        plotters["metrics"] = (ensemble_metrics, global_ensemble_metrics,
                               {"normalize": "range", "num_null_draws": num_null_draws})
    shared = dict(
        true_list=[e["true"] for e in entries], pred_list=[e["pred"] for e in entries],
        parameter_masks=[e["mask"] for e in entries],
        design_configs=[cfg["benchmark_configs"][case] for case in CASES],
        intrinsic_params=cfg["intrinsic_params"], variable_names=cfg["variable_names"],
        max_num_categories=2, max_num_regressors=2,
        categorical_regressors_list=[[] for _ in CASES], global_params=GLOBAL_PARAMS,
    )
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, (local_fn, global_fn, extra) in plotters.items():
        kwargs = dict(shared, **extra)
        if name != "metrics":
            kwargs.update(colors=colors, labels=CASES)
        for scope, plot_fn in (("local", local_fn), ("global", global_fn)):
            logging.info("Plotting %s bench5 %s (%s)", family, name, scope)
            # Reuse one seeded null reference for local and global heatmaps;
            # the family seed is independent of which other families are plotted.
            if name == "metrics":
                np.random.seed(seed)
            global_opts = GLOBAL_SHARE.get(name, {}) if scope == "global" else {}
            fig = plot_fn(**kwargs, **global_opts)
            if fig is None:
                continue
            suffix = "_global" if scope == "global" else ""
            path = outdir / f"{family}_bench5_ensemble_{name}{suffix}.pdf"
            try:
                fig.savefig(path, bbox_inches="tight")
            finally:
                plt.close(fig)
            outputs.append(path.name)
            logging.info("[saved] %s", path)
    return outputs


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    # Check every requested input before creating any plots. bench5 always means
    # all five cases, so a missing file must not silently produce a partial ensemble.
    families = {}
    for family in dict.fromkeys(args.families):
        cfg = add_global_params(BASE_MODELS[family], GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
        entries = [load_case(args.data_dir, family, case, cfg) for case in CASES]
        sizes = {e["pred"].shape[:2] for e in entries}
        if len(sizes) != 1:
            raise ValueError(f"{family}: metric ensembles require equal dataset/draw counts across cases")
        families[family] = (cfg, entries)

    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "cases": CASES, "global_params": GLOBAL_PARAMS, "normalize": "range",
        "metrics_aggregation": "Per-cell mean and population SD across active benchmark cases",
        "seed": args.seed, "num_null_draws": args.num_null_draws,
        "families": {},
    }
    for family, (cfg, entries) in families.items():
        outputs = plot_bench5(cfg, entries, args.outdir, family, metrics=args.metrics,
                              num_null_draws=args.num_null_draws, seed=args.seed)
        manifest["families"][family] = {
            "inputs": [e["source"] for e in entries],
            "datasets_per_case": int(entries[0]["pred"].shape[0]),
            "draws_per_dataset": int(entries[0]["pred"].shape[1]), "outputs": outputs,
        }
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logging.info("Done: %s", args.outdir)


if __name__ == "__main__":
    main()
