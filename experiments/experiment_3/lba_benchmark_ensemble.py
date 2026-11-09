"""CogFormer LBA bench5 ensembles (zero-shot and finetuned), BayesFlow layout.

Reads the per-case predictions that experiment 3's validators saved and draws them
with experiment 1a's ``plot_bench5``, so every figure is laid out exactly like
``figures/model_family/bf/ensemble_benchmark/lba_bench5_ensemble_*.pdf``: same
five cases, palette, labels, range-normalised metrics, null seed and shared
global axes. No checkpoint is loaded; CPU only.

    <zero_shot_dir>/predictions/lba_<case>_mid1_cf_pred.npz   (model_class_lba_cf_validate)
    <finetuned_dir>/predictions/lba_<case>_ft_pred.npz        (model_class_lba_finetune_validate)
      -> <outdir>/<arm>/lba_bench5_ensemble_<plot>[_global].pdf + manifest.json
"""
from experiments.paths import RESULTS_ROOT
from experiments import shared_config

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

from ..configs import MODELS, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES
from ..experiment_1a.model_family_bf_benchmark_ensemble import CASES, plot_bench5
from ..utils import add_global_params

_LBA_DIR = (RESULTS_ROOT / "figures/model_class/cf") / f"{shared_config.get('MC_TAG')}_lba"
_ZS_A = f"A{format(float(shared_config.get('LBA_FIXED_A')), 'g')}"
# arm -> filename infix written by its validator
ARMS = {"zeroshot": "mid1_cf_pred", "finetuned": "ft_pred"}


def load_case(pred_dir: Path, case: str, infix: str, cfg: dict) -> dict:
    """Load one CogFormer case, checking it is the compact bench5 grid of ``case``.

    The validators score bench5 from the shared benchmark files on the compact
    (2-category) grid: rows 1, u_1, u_2, u_1 x u_2. A case scored on the padded
    training grid, or with a mask that disagrees with the design, would overlay
    the wrong coefficients, so both are rejected rather than reshaped.
    """
    path = pred_dir / f"lba_{case}_{infix}.npz"
    with np.load(path) as data:
        truth, pred, mask = data["true_set"], data["pred_set"], data["params_mask"]
        cats = int(data["max_num_categories"]) if "max_num_categories" in data.files else 2
    params = cfg["intrinsic_params"]
    design = cfg["benchmark_configs"][case]
    grid_mask = np.array([[p in active for p in params] for active in design.values()])
    if cats != 2 or mask.shape != grid_mask.shape or not np.array_equal(mask.astype(bool), grid_mask):
        raise ValueError(f"{path}: parameter mask is not the compact bench5 grid of {case!r}")
    if truth.shape[1:] != grid_mask.shape or pred.ndim != 4 or pred.shape[::2] != truth.shape[:2] \
            or pred.shape[2:] != grid_mask.shape:
        raise ValueError(f"{path}: expected truth (datasets, 4, {len(params)}) "
                         f"and draws (datasets, draws, 4, {len(params)})")
    if not np.all(np.isfinite(truth[:, grid_mask])) or not np.all(np.isfinite(pred[:, :, grid_mask])):
        raise ValueError(f"{path}: nonfinite values in active parameters")
    return {"true": truth, "pred": pred, "mask": grid_mask, "source": str(path.resolve())}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--zero_shot_dir", type=Path, default=_LBA_DIR / "zeroshot" / _ZS_A,
                   help="model_class_lba_cf_validate --outdir INCLUDING A<fixed_A> (holds predictions/)")
    p.add_argument("--finetuned_dir", type=Path, default=_LBA_DIR / "finetuned",
                   help="model_class_lba_finetune_validate --outdir (holds predictions/)")
    p.add_argument("--outdir", type=Path, default=_LBA_DIR / "ensemble_benchmark")
    p.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    p.add_argument("--metrics", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num_null_draws", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0, help="Seed for the log-gamma null distribution")
    args = p.parse_args()
    if args.num_null_draws < 1:
        p.error("--num_null_draws must be positive")
    if not 0 <= args.seed < 2**32:
        p.error("--seed must be between 0 and 2**32 - 1")
    return args


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    cfg = add_global_params(MODELS["lba"], GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
    dirs = {"zeroshot": args.zero_shot_dir, "finetuned": args.finetuned_dir}

    # Load and check everything before plotting: bench5 always means all five cases.
    arms = {}
    for arm in dict.fromkeys(args.arms):
        entries = [load_case(dirs[arm] / "predictions", case, ARMS[arm], cfg) for case in CASES]
        if len({e["pred"].shape[:2] for e in entries}) != 1:
            raise ValueError(f"{arm}: metric ensembles require equal dataset/draw counts across cases")
        arms[arm] = entries
    # Both arms score the shared benchmark files, so their ground truth must agree.
    if len(arms) == 2:
        for case, zs, ft in zip(CASES, arms["zeroshot"], arms["finetuned"]):
            if not np.array_equal(zs["true"][:, zs["mask"]], ft["true"][:, ft["mask"]]):
                raise ValueError(f"{case}: zero-shot and finetuned arms scored different datasets")

    manifest = {"cases": CASES, "global_params": GLOBAL_PARAMS, "normalize": "range",
                "seed": args.seed, "num_null_draws": args.num_null_draws,
                "layout": "experiment_1a.model_family_bf_benchmark_ensemble.plot_bench5",
                "arms": {}}
    for arm, entries in arms.items():
        outputs = plot_bench5(cfg, entries, args.outdir / arm, "lba", metrics=args.metrics,
                              num_null_draws=args.num_null_draws, seed=args.seed)
        manifest["arms"][arm] = {
            "inputs": [e["source"] for e in entries],
            "datasets_per_case": int(entries[0]["pred"].shape[0]),
            "draws_per_dataset": int(entries[0]["pred"].shape[1]), "outputs": outputs,
        }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logging.info("Done: %s", args.outdir)


if __name__ == "__main__":
    main()
