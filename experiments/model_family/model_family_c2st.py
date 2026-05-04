import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cogformer.diagnostics.metric.adaptive_c2st import adaptive_c2st as compute_adaptive_c2st
from cogformer.diagnostics.metric.adaptive_c2st import compute_joint_c2st
from cogformer.utils.plot_utils import cogformer_fm_colors

from ..configs import MODELS as _BASE_MODELS
from ..utils import reshape_bf_to_cf


_CF_PRED_FILENAMES = {
    "ddm": "{model}_family_{case}_cf_pred.npz",   # DDM uses singular "family"
    "cdm": "{model}_families_{case}_cf_pred.npz",
    "rdm": "{model}_families_{case}_cf_pred.npz",
}
_C2ST_OUTDIRS = {
    "ddm": "./outputs/figures/c2st/ddm/",
    "cdm": "./outputs/figures/c2st/cdm/",
    "rdm": "./outputs/figures/c2st/rdm/",
}

MODELS = {}
for _key, _base in {k: v for k, v in _BASE_MODELS.items() if k in _CF_PRED_FILENAMES}.items():
    _cfg = {
        "intrinsic_params": _base["intrinsic_params"],
        "variable_names": _base["variable_names"],
        "cf_pred_filename": _CF_PRED_FILENAMES[_key],
        "outdir": _C2ST_OUTDIRS[_key],
        # c2st uses a subset of benchmark cases (no "full")
        "benchmark_configs": {
            k: v for k, v in _base["benchmark_configs"].items() if k != "full"
        },
    }
    MODELS[_key] = _cfg


def load_bf_data(data_path):
    dataset = np.load(data_path, allow_pickle=True)
    return {"param_masks": dataset["param_masks"], "pred_set": dataset["pred_set"]}


def load_cf_pred(pred_path):
    dataset = np.load(pred_path, allow_pickle=True)
    return dataset["pred_set"], dataset["params_mask"]


def run_comparison(pred_a, pred_b, design_config, intrinsic_params, variable_names,
                   colors, max_num_categories, params_mask, outdir, file_prefix):
    joint_score = compute_joint_c2st(
        pred_a=pred_a, pred_b=pred_b, design_config=design_config,
        intrinsic_params=intrinsic_params, max_num_categories=max_num_categories,
        parameter_mask=params_mask,
    )
    c2st_df = compute_adaptive_c2st(
        pred_a=pred_a, pred_b=pred_b, design_config=design_config,
        intrinsic_params=intrinsic_params, max_num_categories=max_num_categories,
        parameter_mask=params_mask, variable_names=variable_names,
    )
    mean_accuracy = float(c2st_df["C2ST Accuracy"].mean())
    c2st_df.loc["_mean_accuracy_"] = {"C2ST Accuracy": mean_accuracy}
    c2st_df.loc["_joint_c2st_"] = {"C2ST Accuracy": joint_score}
    c2st_df.to_csv(outdir / f"{file_prefix}_c2st.csv")
    logging.info(f"[saved] {outdir / file_prefix}_c2st.csv  mean_acc={mean_accuracy:.3f}  joint={joint_score:.3f}")
    return mean_accuracy, joint_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    p.add_argument("--bf_data_dir", type=str, default="./cogformer/experiments/data")
    p.add_argument("--cf_pred_dir", type=str, default="./cogformer/experiments/data")
    p.add_argument("--mc_pred_dir", type=str, default="./cogformer/experiments/data/model_class",
                   help="Directory with ModelClass pred files; if set, also runs Class vs Family C2ST")
    p.add_argument("--outdir", type=str, default=None,
                   help="Output directory (defaults to model-specific path)")
    p.add_argument("--max_num_categories", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    model_cfg = MODELS[args.model]
    model_name = args.model
    intrinsic_params = model_cfg["intrinsic_params"]
    variable_names = model_cfg["variable_names"]
    colors = cogformer_fm_colors()
    benchmark = model_cfg["benchmark_configs"]
    cf_pred_template = model_cfg["cf_pred_filename"]
    outdir_root = Path(args.outdir) if args.outdir else Path(model_cfg["outdir"])
    summary_rows = []

    for cfg_name, design_config in benchmark.items():
        bf_data_path = Path(args.bf_data_dir) / f"{model_name}_{cfg_name}_data.npz"
        cf_pred_filename = cf_pred_template.format(model=model_name, case=cfg_name)
        cf_pred_path = Path(args.cf_pred_dir) / cf_pred_filename

        if not bf_data_path.exists():
            logging.warning(f"No BF data for '{cfg_name}', skipping.")
            continue
        if not cf_pred_path.exists():
            logging.warning(f"No GPT pred for '{cfg_name}', skipping.")
            continue

        outdir = outdir_root / cfg_name
        outdir.mkdir(parents=True, exist_ok=True)

        cf_pred_set, params_mask = load_cf_pred(cf_pred_path)
        bf_data = load_bf_data(bf_data_path)
        active_idx = bf_data["param_masks"][0].astype(bool)
        bf_pred_reshaped = reshape_bf_to_cf(
            bf_data["pred_set"][:, :, active_idx], design_config, intrinsic_params
        )

        # Family vs BF
        mean_acc, joint = run_comparison(
            cf_pred_set, bf_pred_reshaped, design_config, intrinsic_params,
            variable_names, colors, args.max_num_categories, params_mask,
            outdir, f"{model_name}_families_{cfg_name}",
        )
        summary_rows.append({
            "model": model_name.upper(), "case": cfg_name, "scenario": "Family_vs_BF",
            "mean_accuracy": mean_acc, "joint_c2st": joint,
        })

        # Class vs Family
        if args.mc_pred_dir is not None:
            mc_pred_path = Path(args.mc_pred_dir) / f"{model_name}_{cfg_name}_cf_pred.npz"
            if mc_pred_path.exists():
                mc_pred_set, _ = load_cf_pred(mc_pred_path)
                mean_acc_mc, joint_mc = run_comparison(
                    mc_pred_set, cf_pred_set, design_config, intrinsic_params,
                    variable_names, colors, args.max_num_categories, params_mask,
                    outdir, f"{model_name}_{cfg_name}_mc",
                )
                summary_rows.append({
                    "model": model_name.upper(), "case": cfg_name, "scenario": "Class_vs_Family",
                    "mean_accuracy": mean_acc_mc, "joint_c2st": joint_mc,
                })
            else:
                logging.warning(f"No MC pred for '{cfg_name}' at {mc_pred_path}, skipping.")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = outdir_root / f"{model_name}_c2st_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logging.info(f"[saved] {summary_path}")
    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
