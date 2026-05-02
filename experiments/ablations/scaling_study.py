"""
Dimensionality scaling study (ablation 2-iii).

Evaluates a trained CogFormer checkpoint across designs with an increasing
number of regressors, measuring how posterior recovery quality scales with
the effective size of the parameter matrix.

Unlike the architecture ablations, this is a pure evaluation script: no
training is run. Pass --checkpoint_path pointing to a saved .pt file and
the architecture args that match it.

Metrics reported per regressor count
-------------------------------------
- NRMSE          : root mean squared error of the posterior mean, normalised
                   by the range of true values across the batch.
- Coverage (90%) : fraction of active parameters for which the true value
                   falls within the 5th–95th percentile of the posterior.
- Active params  : mean number of active parameter-matrix entries per sample,
                   providing an interpretable x-axis alternative to raw r.

Usage
-----
python -m experiments.ablations.scaling_study \\
    --model ddm \\
    --checkpoint_path ./outputs/checkpoints/model_family/cf/ddm/<ckpt>.pt \\
    --output_dir ./outputs/evaluations/ablations/scaling_study/ddm/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from cogformer.simulator import ModelFamily
from cogformer.adapter import Adapter
from cogformer.networks.transformers.cogformer import CogFormer
from ..utils import add_global_params, build_encoder_input_dim
from ..configs import MODELS as _BASE_MODELS, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES


def _nrmse(pred_mean: np.ndarray, true: np.ndarray, mask: np.ndarray) -> float:
    """
    Normalised RMSE over all active (mask=1) parameter slots.

    pred_mean : (batch, T)
    true      : (batch, T)
    mask      : (batch, T)  — 1 for active slots
    """
    active = mask > 0
    if not active.any():
        return float("nan")
    err = (pred_mean[active] - true[active]) ** 2
    rng = true[active].max() - true[active].min()
    return float(np.sqrt(err.mean()) / rng) if rng > 0 else float(np.sqrt(err.mean()))


def _coverage(pred_samples: np.ndarray, true: np.ndarray, mask: np.ndarray,
              alpha: float = 0.90) -> float:
    """
    Fraction of active parameters whose true value lies within the
    alpha-level equal-tailed credible interval of the posterior.

    pred_samples : (batch, num_samples, T)
    true         : (batch, T)
    mask         : (batch, T)
    """
    lo = (1 - alpha) / 2
    hi = 1 - lo
    lower = np.quantile(pred_samples, lo, axis=1)   # (batch, T)
    upper = np.quantile(pred_samples, hi, axis=1)   # (batch, T)
    active = mask > 0
    if not active.any():
        return float("nan")
    covered = ((lower[active] <= true[active]) & (true[active] <= upper[active]))
    return float(covered.mean())


def _mean_active(mask: np.ndarray) -> float:
    """Mean number of active parameter slots per sample."""
    return float(mask.sum(axis=1).mean())


@torch.no_grad()
def evaluate_at_r(
    cogformer: CogFormer,
    model_family: ModelFamily,
    adapter: Adapter,
    r: int,
    model_cfg: dict,
    model_family_config: dict,
    val_params_kwargs: dict,
    num_obs: int,
    eval_batch_size: int,
    n_eval_batches: int,
    fm_sample_steps: int,
    fm_num_samples: int,
    fm_solver: str,
    device: torch.device,
) -> dict:
    """Run evaluation for a single regressor count r, returning metric dict."""
    link_fn = model_cfg["link_fn"]

    batch_nrmse = []
    batch_cov90 = []
    batch_n_active = []

    for _ in range(n_eval_batches):
        test_samples = model_family.batch_sample(
            batch_size=eval_batch_size,
            num_regressors=r,
            num_obs=num_obs,
            max_num_regressors=model_family_config["max_num_regressors"],
            max_num_categories=model_family_config["max_num_categories"],
            keep_intercept=model_family_config["keep_intercept"],
            add_interaction=model_family_config["add_interaction"],
            mask_randomizer_kwargs=val_params_kwargs,
            flatten_param_outputs=True,
            link_fun=link_fn(),
        )
        adapted = adapter.adapt(test_samples, device=device)

        pred_samples = cogformer.sample(
            adapted["input_data"],
            adapted["param_indices"],
            adapted["regressor_indices"],
            adapted["param_masks"],
            steps=fm_sample_steps,
            num_samples=fm_num_samples,
            solver=fm_solver,
        )
        # pred_samples: (batch, num_samples, T, 1)
        pred_samples = pred_samples[..., 0]   # (batch, num_samples, T)
        pred_mean    = pred_samples.mean(1)   # (batch, T)

        true = adapted["param_matrices"].cpu().numpy()   # (batch, T)
        mask = adapted["param_masks"].cpu().numpy()       # (batch, T)

        batch_nrmse.append(_nrmse(pred_mean, true, mask))
        batch_cov90.append(_coverage(pred_samples, true, mask))
        batch_n_active.append(_mean_active(mask))

    return {
        "num_regressors": r,
        "nrmse_mean":   float(np.nanmean(batch_nrmse)),
        "nrmse_sem":    float(np.nanstd(batch_nrmse) / np.sqrt(len(batch_nrmse))),
        "cov90_mean":   float(np.nanmean(batch_cov90)),
        "cov90_sem":    float(np.nanstd(batch_cov90) / np.sqrt(len(batch_cov90))),
        "n_active_mean": float(np.nanmean(batch_n_active)),
        "raw_nrmse":    batch_nrmse,
        "raw_cov90":    batch_cov90,
    }


def plot_scaling(results: list[dict], output_path: Path, model_name: str):
    rs       = [d["num_regressors"] for d in results]
    nrmse    = np.array([d["nrmse_mean"] for d in results])
    nrmse_se = np.array([d["nrmse_sem"] for d in results])
    cov90    = np.array([d["cov90_mean"] for d in results])
    cov90_se = np.array([d["cov90_sem"] for d in results])
    n_active = np.array([d["n_active_mean"] for d in results])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.errorbar(rs, nrmse, yerr=nrmse_se, marker="o", capsize=4, color="steelblue")
    ax.set_xlabel("Number of regressors")
    ax.set_ylabel("NRMSE")
    ax.set_title(f"{model_name} — Recovery quality vs. regressors")
    ax.set_xticks(rs)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(rs)
    ax2.set_xticklabels([f"{int(n)}" for n in n_active], fontsize=8)
    ax2.set_xlabel("Mean active parameters", fontsize=9)

    ax = axes[1]
    ax.errorbar(rs, cov90, yerr=cov90_se, marker="o", capsize=4, color="darkorange")
    ax.axhline(0.90, color="gray", linestyle="--", linewidth=0.8, label="Nominal 90%")
    ax.set_xlabel("Number of regressors")
    ax.set_ylabel("Coverage (90% CI)")
    ax.set_title(f"{model_name} — Calibration vs. regressors")
    ax.set_xticks(rs)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, required=True,
                        choices=["ddm", "cdm", "rdm"])
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the trained CogFormer .pt checkpoint.")
    parser.add_argument("--output_dir", type=str, default="./outputs/evaluations/ablations/scaling_study/",
                        help="Directory to write CSV and figure outputs.")

    # Regressor sweep
    parser.add_argument("--regressor_counts", type=int, nargs="+",
                        default=[0, 1, 2, 3, 4, 5],
                        help="Regressor counts to evaluate. (default: 0 1 2 3 4 5)")

    # Evaluation budget
    parser.add_argument("--n_eval_batches", type=int, default=10,
                        help="Number of random batches per regressor count. (default: 10)")
    parser.add_argument("--eval_batch_size", type=int, default=200)
    parser.add_argument("--num_obs", type=int, default=500)
    parser.add_argument("--fm_sample_steps", type=int, default=50)
    parser.add_argument("--fm_num_samples", type=int, default=200)
    parser.add_argument("--fm_solver", type=str, default="midpoint",
                        choices=["euler", "midpoint", "rk4"])

    # Architecture (must match the checkpoint)
    parser.add_argument("--encoder_num_layers", type=int, default=8)
    parser.add_argument("--decoder_num_layers", type=int, default=8)
    parser.add_argument("--encoder_num_heads", type=int, default=8)
    parser.add_argument("--decoder_num_heads", type=int, default=8)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--num_seeds", type=int, default=32)
    parser.add_argument("--seed_dim", type=int, default=64)
    parser.add_argument("--time_embedding_dim", type=int, default=32)
    parser.add_argument("--pos_embedding_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Design space (must match training)
    parser.add_argument("--max_num_regressors", type=int, default=5)
    parser.add_argument("--max_num_categories", type=int, default=5)
    parser.add_argument("--keep_intercept", action="store_true", default=True)
    parser.add_argument("--add_interaction", action="store_true", default=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_cfg = _BASE_MODELS[args.model]
    model_cfg = add_global_params(base_cfg, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)

    val_params_kwargs = {
        "free_intrinsics": model_cfg["intrinsic_params"],
        "fixed_intrinsics": [],
        "fixed_values": {},
    }

    model_family_config = {
        "max_num_regressors": args.max_num_regressors,
        "max_num_categories": args.max_num_categories,
        "keep_intercept": args.keep_intercept,
        "add_interaction": args.add_interaction,
    }

    encoder_input_dim = build_encoder_input_dim(
        args.max_num_regressors, args.max_num_categories, args.keep_intercept
    )

    cogformer = CogFormer(
        encoder_input_dim=encoder_input_dim,
        encoder_num_layers=args.encoder_num_layers,
        decoder_num_layers=args.decoder_num_layers,
        encoder_num_heads=args.encoder_num_heads,
        decoder_num_heads=args.decoder_num_heads,
        num_seeds=args.num_seeds,
        seed_dim=args.seed_dim,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
        decoder_layer_design="mixed_attention",
        decoder_layer_kwargs={"mab_first": True},
        time_embedding_dim=args.time_embedding_dim,
        pos_embedding_dim=args.pos_embedding_dim,
    )
    state = torch.load(args.checkpoint_path, map_location=device)
    cogformer.load_state_dict(state)
    cogformer.to(device).eval()

    model_family = ModelFamily(
        model=model_cfg["model_cls"](),
        name=model_cfg["name"],
        prior=model_cfg["prior_fn"](),
    )
    adapter = Adapter()

    results = []
    for r in args.regressor_counts:
        print(f"Evaluating r={r} regressors …", flush=True)
        row = evaluate_at_r(
            cogformer=cogformer,
            model_family=model_family,
            adapter=adapter,
            r=r,
            model_cfg=model_cfg,
            model_family_config=model_family_config,
            val_params_kwargs=val_params_kwargs,
            num_obs=args.num_obs,
            eval_batch_size=args.eval_batch_size,
            n_eval_batches=args.n_eval_batches,
            fm_sample_steps=args.fm_sample_steps,
            fm_num_samples=args.fm_num_samples,
            fm_solver=args.fm_solver,
            device=device,
        )
        results.append(row)
        print(
            f"  r={r:2d}  NRMSE={row['nrmse_mean']:.4f}±{row['nrmse_sem']:.4f}"
            f"  Cov90={row['cov90_mean']:.3f}±{row['cov90_sem']:.3f}"
            f"  ActiveParams={row['n_active_mean']:.1f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = model_cfg["name"]

    # Summary CSV (one row per r)
    summary_cols = ["num_regressors", "n_active_mean",
                    "nrmse_mean", "nrmse_sem", "cov90_mean", "cov90_sem"]
    summary_df = pd.DataFrame([{k: r[k] for k in summary_cols} for r in results])
    summary_path = output_dir / f"{args.model}_scaling_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[saved] {summary_path}")

    # Raw per-batch CSV (one row per batch per r)
    raw_records = []
    for row in results:
        for i, (nrmse, cov) in enumerate(zip(row["raw_nrmse"], row["raw_cov90"])):
            raw_records.append({
                "num_regressors": row["num_regressors"],
                "batch_idx": i,
                "nrmse": nrmse,
                "cov90": cov,
            })
    raw_df = pd.DataFrame(raw_records)
    raw_path = output_dir / f"{args.model}_scaling_raw.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"[saved] {raw_path}")

    # Figure
    fig_path = output_dir / f"{args.model}_scaling.pdf"
    plot_scaling(results, fig_path, model_name)
    print(f"[saved] {fig_path}")

    # Checkpoint metadata sidecar
    meta = {
        "checkpoint": args.checkpoint_path,
        "model": args.model,
        "regressor_counts": args.regressor_counts,
        "n_eval_batches": args.n_eval_batches,
        "eval_batch_size": args.eval_batch_size,
        "num_obs": args.num_obs,
    }
    meta_path = output_dir / f"{args.model}_scaling_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[saved] {meta_path}")
