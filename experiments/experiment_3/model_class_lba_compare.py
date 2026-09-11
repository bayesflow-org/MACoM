"""Compare zero-shot vs. fine-tuned LBA inference, and check forgetting on DDM/RDM/CDM.

Loads .npz prediction files from two pred_dirs and computes:
  1. Delta r heatmap (fine-tuned − zero-shot) for LBA recovery.
  2. Delta r heatmap (fine-tuned − base) for DDM, RDM, CDM (forgetting).
  3. Delta metrics table (fine-tuned − base) for all models as CSV.

Expected file naming conventions:
  Zero-shot LBA  : {zero_shot_lba_pred_dir}/lba_{cfg}_mid1_cf_pred.npz
  Fine-tuned LBA : {finetuned_pred_dir}/lba_{cfg}_ft_pred.npz
  Base MC        : {base_mc_pred_dir}/{model}_{cfg}_cf_pred.npz
  Fine-tuned MC  : {finetuned_pred_dir}/{model}_{cfg}_ft_pred.npz
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cogformer.diagnostics.plot.adaptive_correlation import compute_r_matrix, adaptive_correlation_delta
from cogformer.diagnostics.metric.adaptive_metrics import adaptive_metrics as compute_adaptive_metrics

from ..configs import MODELS as _BASE_MODELS, LBA_CONFIG
from ..utils import add_global_params

GLOBAL_PARAMS = ["pi"]
GLOBAL_VARIABLE_NAMES = [r"$\pi$"]

_MODELS_WITH_PI = {
    m: add_global_params(cfg, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
    for m, cfg in _BASE_MODELS.items()
    if m != "lba"
}
_LBA_WITH_PI = add_global_params(LBA_CONFIG, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
LBA_LOCAL_PARAMS = _LBA_WITH_PI["intrinsic_params"]

MC_MODELS = {
    "DDM": _MODELS_WITH_PI["ddm"],
    "RDM": _MODELS_WITH_PI["rdm"],
    "CDM": _MODELS_WITH_PI["cdm"],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zero_shot_lba_pred_dir", type=str,
                   default="./outputs/data/generalization/model_class_lba/cf/",
                   help="pred_dir from model_class_lba_cf_validate.py (zero-shot LBA).")
    p.add_argument("--base_mc_pred_dir", type=str,
                   default="./outputs/data/model_class/",
                   help="pred_dir from model_class_cf_validate.py (base DDM/RDM/CDM).")
    p.add_argument("--finetuned_pred_dir", type=str,
                   default="./outputs/data/model_class/cf/lba_finetune/",
                   help="pred_dir from model_class_lba_finetune_validate.py.")
    p.add_argument("--outdir", type=str,
                   default="./outputs/figures/model_class/cf/lba_comparison/")
    p.add_argument("--ref_condition", type=str, default="interaction",
                   help="Benchmark condition used for the delta heatmap (default: interaction).")
    p.add_argument("--max_num_categories", type=int, default=5)
    p.add_argument("--skip_log_gamma", action="store_true", default=True)
    return p.parse_args()


def _load_npz(path: Path):
    """Load pred_set, true_set, params_mask from an npz file."""
    data = np.load(path)
    return data["pred_set"], data["true_set"], data["params_mask"]


def _load_metrics_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        logging.warning(f"Metrics CSV not found: {path}")
        return None
    return pd.read_csv(path, index_col=0)


def _delta_metrics(df_a: pd.DataFrame, df_b: pd.DataFrame, label_a: str, label_b: str) -> pd.DataFrame:
    """Return a combined table with columns for a, b, and delta=b−a."""
    shared_idx = df_a.index.intersection(df_b.index)
    shared_cols = df_a.columns.intersection(df_b.columns)
    a = df_a.loc[shared_idx, shared_cols]
    b = df_b.loc[shared_idx, shared_cols]
    combined = pd.concat(
        [a.add_suffix(f" ({label_a})"), b.add_suffix(f" ({label_b})"),
         (b - a).add_suffix(f" (Δ {label_b}−{label_a})")],
        axis=1,
    )
    col_order = []
    for c in shared_cols:
        col_order += [f"{c} ({label_a})", f"{c} ({label_b})", f"{c} (Δ {label_b}−{label_a})"]
    return combined[[c for c in col_order if c in combined.columns]]


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    zero_shot_dir  = Path(args.zero_shot_lba_pred_dir)
    base_mc_dir    = Path(args.base_mc_pred_dir)
    finetuned_dir  = Path(args.finetuned_pred_dir)
    ref            = args.ref_condition
    max_cats       = args.max_num_categories

    all_delta_rows = []

    # ── LBA: zero-shot vs. fine-tuned ─────────────────────────────────────────
    zs_lba_path = zero_shot_dir / f"lba_{ref}_mid1_cf_pred.npz"
    ft_lba_path = finetuned_dir / f"lba_{ref}_ft_pred.npz"

    if zs_lba_path.exists() and ft_lba_path.exists():
        pred_zs, true_zs, mask_zs = _load_npz(zs_lba_path)
        pred_ft, true_ft, mask_ft = _load_npz(ft_lba_path)

        design_config = _LBA_WITH_PI["benchmark_configs"][ref]
        intrinsic_params = LBA_LOCAL_PARAMS
        variable_names   = _LBA_WITH_PI["variable_names"]

        r_zs, pm = compute_r_matrix(true_zs, pred_zs, design_config, intrinsic_params, max_cats,
                                     mask_zs[None])
        r_ft, _  = compute_r_matrix(true_ft, pred_ft, design_config, intrinsic_params, max_cats,
                                     mask_ft[None])

        fig = adaptive_correlation_delta(
            r_zs, r_ft, pm, design_config, intrinsic_params, max_cats,
            variable_names=variable_names,
            label_a="zero-shot", label_b="fine-tuned",
        )
        fig.suptitle(f"LBA — {ref}", fontsize=14)
        fig.savefig(outdir / f"lba_{ref}_delta_r.pdf", bbox_inches="tight")
        plt.close(fig)
        logging.info(f"[saved] {outdir / f'lba_{ref}_delta_r.pdf'}")

        # metrics delta for LBA across all available conditions
        for cfg_name in _LBA_WITH_PI["benchmark_configs"]:
            zs_m = zero_shot_dir / f"lba_{cfg_name}_mid1_cf_pred.npz"
            ft_m = finetuned_dir / f"lba_{cfg_name}_ft_pred.npz"
            csv_zs = zero_shot_dir.parent / "figures" / "lba" / cfg_name / f"lba_{cfg_name}_metrics.csv"
            csv_ft = finetuned_dir.parent / "figures" / "lba" / cfg_name / f"lba_{cfg_name}_ft_metrics.csv"
            # Try outdir-based paths that validate scripts actually write
            csv_zs_alt = Path(args.zero_shot_lba_pred_dir).parent.parent / "figures" / \
                         "generalization" / "model_class_lba" / "cf" / cfg_name / "mid_rdm" / \
                         f"lba_{cfg_name}_mid1_metrics.csv"
            csv_ft_alt = Path(args.finetuned_pred_dir).parent.parent / "figures" / \
                         "model_class" / "cf" / "lba_finetune" / "lba" / cfg_name / \
                         f"lba_{cfg_name}_ft_metrics.csv"
            df_zs = _load_metrics_csv(csv_zs) or _load_metrics_csv(csv_zs_alt)
            df_ft = _load_metrics_csv(csv_ft) or _load_metrics_csv(csv_ft_alt)
            if df_zs is not None and df_ft is not None:
                delta = _delta_metrics(df_zs, df_ft, "zero-shot", "fine-tuned")
                delta.insert(0, "model", "LBA")
                delta.insert(1, "condition", cfg_name)
                all_delta_rows.append(delta)
    else:
        logging.warning(f"LBA prediction files not found for condition '{ref}'. Skipping LBA delta.")

    # ── DDM / RDM / CDM: base vs. fine-tuned (forgetting) ─────────────────────
    for model_name, model_cfg in MC_MODELS.items():
        m = model_name.lower()
        base_path = base_mc_dir / f"{m}_{ref}_cf_pred.npz"
        ft_path   = finetuned_dir / f"{m}_{ref}_ft_pred.npz"

        if base_path.exists() and ft_path.exists():
            pred_base, true_base, mask_base = _load_npz(base_path)
            pred_ft,   true_ft,   mask_ft   = _load_npz(ft_path)

            design_config    = model_cfg["benchmark_configs"][ref]
            intrinsic_params = model_cfg["intrinsic_params"]
            variable_names   = model_cfg["variable_names"]

            r_base, pm = compute_r_matrix(true_base, pred_base, design_config,
                                           intrinsic_params, max_cats, mask_base[None])
            r_ft, _    = compute_r_matrix(true_ft,   pred_ft,   design_config,
                                           intrinsic_params, max_cats, mask_ft[None])

            fig = adaptive_correlation_delta(
                r_base, r_ft, pm, design_config, intrinsic_params, max_cats,
                variable_names=variable_names,
                label_a="base", label_b="fine-tuned",
            )
            fig.suptitle(f"{model_name} — {ref} (forgetting)", fontsize=14)
            fig.savefig(outdir / f"{m}_{ref}_forgetting_delta_r.pdf", bbox_inches="tight")
            plt.close(fig)
            logging.info(f"[saved] {outdir / f'{m}_{ref}_forgetting_delta_r.pdf'}")
        else:
            logging.warning(f"{model_name} prediction files not found for condition '{ref}'. Skipping.")

        # metrics delta across all conditions
        for cfg_name in model_cfg["benchmark_configs"]:
            csv_base = base_mc_dir.parent / "figures" / "model_class" / "cf" / m / cfg_name / \
                       f"{m}_{cfg_name}_fm_metrics.csv"
            csv_ft   = Path(args.finetuned_pred_dir).parent.parent / "figures" / \
                       "model_class" / "cf" / "lba_finetune" / m / cfg_name / \
                       f"{m}_{cfg_name}_ft_metrics.csv"
            df_base = _load_metrics_csv(csv_base)
            df_ft   = _load_metrics_csv(csv_ft)
            if df_base is not None and df_ft is not None:
                delta = _delta_metrics(df_base, df_ft, "base", "fine-tuned")
                delta.insert(0, "model", model_name)
                delta.insert(1, "condition", cfg_name)
                all_delta_rows.append(delta)

    # ── Save combined metrics delta table ──────────────────────────────────────
    if all_delta_rows:
        combined = pd.concat(all_delta_rows, axis=0)
        combined.to_csv(outdir / "delta_metrics.csv")
        logging.info(f"[saved] {outdir / 'delta_metrics.csv'}")
    else:
        logging.warning("No metrics CSVs found; delta_metrics.csv not written.")

    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
