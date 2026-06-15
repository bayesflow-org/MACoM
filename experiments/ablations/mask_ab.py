"""A/B: decoder self-attention key_padding_mask ON vs OFF.

Trains two identical DDM CogFormers — one with the self-attention key_padding_mask
(current default), one without (lets the fused FlashAttention kernel fire) — then
reports, for each variant:

  * median train step time (steady-state, post-compile)
  * recovery quality on a fixed validation set: per-active-cell correlation R
    (true vs posterior mean) and std-normalized RMSE, averaged over active cells.

The only difference between the two runs is whether self-attention sees the padding
mask; inactive token *outputs* are still zeroed after every layer in both. Use the
recovery delta to decide whether dropping the mask (a ~10% step-time win, measured
separately) is acceptable.

Usage:
    python -m experiments.ablations.mask_ab --epochs 300 --steps_per_epoch 100
    python -m experiments.ablations.mask_ab --epochs 300 --val_batch_size 500 --seed 0
"""
import os
import argparse
import time

import numpy as np
import torch

from cogformer.adapter import Adapter
from cogformer.networks.transformers.cogformer import CogFormer
from cogformer.diagnostics.metric.adaptive_metrics import _rmse

from ..configs import MODELS as _BASE_MODELS, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES
from ..utils import (add_global_params, build_encoder_input_dim, chunked_sample,
                     train_step, recommend_sim_workers,
                     ModelFamilyDataset, _identity_collate)


def _build_loader(dataset, num_workers):
    """Like utils.make_sim_loader but non-persistent, so single-iterator runs shut
    down workers cleanly (persistent numba workers abort noisily when killed)."""
    use_mp = num_workers > 0
    return torch.utils.data.DataLoader(
        dataset, batch_size=None, collate_fn=_identity_collate,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        **(dict(prefetch_factor=2, persistent_workers=False,
                multiprocessing_context="spawn") if use_mp else {}),
    )


def build_configs(args):
    """Mirror experiments.model_family.model_family_cf_train config construction (DDM)."""
    model_cfg = add_global_params(_BASE_MODELS["ddm"], GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)

    mf_config = {
        "max_num_regressors": args.max_num_regressors,
        "max_num_categories": args.max_num_categories,
        "keep_intercept": True,
        "add_interaction": True,
    }
    train_params_kwargs = {
        "free_intrinsics": model_cfg["free_params"],
        "fixed_intrinsics": model_cfg["fixed_params"],
        "fixed_values": model_cfg["default_fixed_values"],
        "global_params": model_cfg.get("global_params"),
    }
    val_params_kwargs = {
        "free_intrinsics": model_cfg["intrinsic_params"],
        "fixed_intrinsics": [],
        "fixed_values": model_cfg["default_fixed_values"],
    }
    train_config = {
        "epochs": args.epochs,
        "batch_size": args.train_batch_size,
        "steps_per_epoch": args.steps_per_epoch,
        "learning_rate": args.lr,
        "gradient_clip_norm": 5.0,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "model_family_config": mf_config,
        "train_sample_config": {
            "mask_randomizer_kwargs": train_params_kwargs,
            "min_num_regressors": 0,
            "min_num_obs": args.min_num_obs,
            "max_num_obs": args.max_num_obs,
            "fixed_config": False,
        },
    }
    val_sample_config = {
        "mask_randomizer_kwargs": val_params_kwargs,
        "min_num_regressors": 2,
        "num_obs": args.num_obs,
        "fixed_config": False,
    }
    return model_cfg, train_config, mf_config, val_sample_config


def prepare_val_data(model_cfg, mf_config, val_sample_config, args, device):
    """One fixed val scenario (the 'interaction' design), built like the trainer's _prepare_val_data."""
    from cogformer.simulator import ModelFamily
    # num_workers=1 → single-threaded, deterministic RNG order so separate process
    # launches (same seed) generate byte-identical validation sets.
    mf = ModelFamily(
        name=model_cfg["name"], model=model_cfg["model_cls"](),
        prior=model_cfg["prior_fn"](), num_workers=1,
    )
    adapter = Adapter()
    link_fun = model_cfg["link_fn"]()
    n_cols = len(model_cfg["intrinsic_params"])
    training_max_cats = mf_config["max_num_categories"]

    tag, design_config, opts = model_cfg["val_scenarios"][0]
    max_cats = opts.get("max_num_categories", training_max_cats)
    model_family_cfg = {**mf_config, "max_num_categories": max_cats}

    samples = mf.batch_sample(
        **model_family_cfg,
        **val_sample_config,
        batch_size=args.val_batch_size,
        flatten_param_outputs=True,
        design_config=design_config,
        link_fun=link_fun,
        pad_to_max_num_categories=training_max_cats,
    )
    adapted = adapter.adapt(samples, device=device)
    true_flat = adapted["param_matrices"].detach().cpu().numpy()
    n_rows = true_flat.shape[1] // n_cols
    true_set = true_flat.reshape(args.val_batch_size, n_rows, n_cols)
    params_mask = adapted["param_masks"].detach().cpu().numpy()
    params_mask = params_mask.reshape(args.val_batch_size, n_rows, n_cols)[0]
    return adapted, true_set, params_mask, n_rows, n_cols, tag


@torch.no_grad()
def recovery_metrics(net, adapted, true_set, params_mask, n_rows, n_cols, args):
    """Return (mean_R, mean_NRMSE) over active parameter cells."""
    net.eval()
    pred = chunked_sample(
        net, adapted, num_samples=args.fm_num_samples, steps=args.fm_sample_steps,
        solver=args.fm_solver, samples_chunk_size=args.val_samples_chunk_size,
    )
    pred = pred.reshape(args.val_batch_size, args.fm_num_samples, n_rows, n_cols)
    net.train()

    corrs, nrmses = [], []
    for r in range(n_rows):
        for c in range(n_cols):
            if params_mask[r, c] <= 0:
                continue
            x = true_set[:, r, c]                      # (B,)
            draws = pred[:, :, r, c]                    # (B, num_samples)
            y_mean = draws.mean(axis=1)                # (B,)
            if np.std(x) > 1e-8 and np.std(y_mean) > 1e-8:
                corrs.append(np.corrcoef(x, y_mean)[0, 1])
            nrmses.append(_rmse(draws, x, normalize="std", aggregation=np.mean))
    return float(np.mean(corrs)), float(np.mean(nrmses))


def run_variant(use_mask, model_cfg, train_config, mf_config, val_sample_config, args):
    device = train_config["device"]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    encoder_input_dim = build_encoder_input_dim(
        args.max_num_regressors, args.max_num_categories, True)

    cogformer_config = dict(
        encoder_input_dim=encoder_input_dim,
        encoder_num_layers=args.encoder_num_layers, decoder_num_layers=args.decoder_num_layers,
        encoder_num_heads=args.encoder_num_heads, decoder_num_heads=args.decoder_num_heads,
        encoder_embed_dim=64, ffn_expansion_factor=4.0,
        num_seeds=args.num_seeds, seed_dim=args.seed_dim, proj_dim=args.proj_dim,
        dropout=0.05, layer_dropout=0.05,
        decoder_layer_design="mixed_attention", decoder_layer_kwargs={"mab_first": True},
        time_embedding_dim=32, pos_embedding_dim=32,
        self_attn_padding_mask=use_mask,
    )
    net = CogFormer(**cogformer_config).to(device).train()
    net = torch.compile(net, dynamic=True, mode=args.compile_mode)

    optimizer = torch.optim.AdamW(net.parameters(), lr=train_config["learning_rate"], fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config["epochs"])

    num_sim_workers = args.num_sim_workers or recommend_sim_workers()
    dataset = ModelFamilyDataset(model_cfg, train_config, num_sim_workers, world_size=1)
    loader = _build_loader(dataset, num_sim_workers)
    loader_iter = iter(loader)

    step_times = []
    warmup_steps = train_config["steps_per_epoch"]  # skip first epoch (compile) when timing
    total_steps = train_config["epochs"] * train_config["steps_per_epoch"]
    last_loss = float("nan")
    for step in range(total_steps):
        adapted_cpu = next(loader_iter)
        adapted = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                   for k, v in adapted_cpu.items()}
        if step >= warmup_steps:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        last_loss, _ = train_step(net, adapted, train_config, optimizer, scheduler)
        if step >= warmup_steps:
            torch.cuda.synchronize()
            step_times.append((time.perf_counter() - t0) * 1000)
        if (step + 1) % train_config["steps_per_epoch"] == 0:
            scheduler.step()

    med_step = float(np.median(step_times)) if step_times else float("nan")

    # explicitly shut down spawn workers to avoid noisy "Aborted" tracebacks at GC
    try:
        loader_iter._shutdown_workers()
    except Exception:
        pass
    del loader_iter, loader
    import gc
    gc.collect()

    return net, med_step, last_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--steps_per_epoch", type=int, default=100)
    p.add_argument("--train_batch_size", type=int, default=64)
    p.add_argument("--val_batch_size", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min_num_obs", type=int, default=200)
    p.add_argument("--max_num_obs", type=int, default=1000)
    p.add_argument("--num_obs", type=int, default=500)
    p.add_argument("--max_num_regressors", type=int, default=5)
    p.add_argument("--max_num_categories", type=int, default=5)
    p.add_argument("--encoder_num_layers", type=int, default=8)
    p.add_argument("--decoder_num_layers", type=int, default=8)
    p.add_argument("--encoder_num_heads", type=int, default=8)
    p.add_argument("--decoder_num_heads", type=int, default=8)
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--num_seeds", type=int, default=32)
    p.add_argument("--seed_dim", type=int, default=64)
    p.add_argument("--fm_sample_steps", type=int, default=50)
    p.add_argument("--fm_num_samples", type=int, default=200)
    p.add_argument("--fm_solver", type=str, default="midpoint")
    p.add_argument("--val_samples_chunk_size", type=int, default=4)
    p.add_argument("--compile_mode", type=str, default="max-autotune-no-cudagraphs")
    p.add_argument("--num_sim_workers", type=int, default=None)
    p.add_argument("--variant", choices=["both", "on", "off"], default="both",
                   help="Run one variant (for parallel launch across GPUs) or both "
                        "sequentially. The val set is seeded so 'on'/'off' launches "
                        "with the same --seed score against identical validation data.")
    args = p.parse_args()

    model_cfg, train_config, mf_config, val_sample_config = build_configs(args)
    device = train_config["device"]

    # Seed val-set generation so separate 'on'/'off' process launches (same --seed)
    # produce an identical fixed validation set — required for a fair comparison.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    val = prepare_val_data(model_cfg, mf_config, val_sample_config, args, device)

    variants = {"both": (True, False), "on": (True,), "off": (False,)}[args.variant]
    results = {}
    for use_mask in variants:
        label = "mask_ON" if use_mask else "mask_OFF"
        print(f"\n===== training variant: {label} "
              f"({args.epochs} epochs x {args.steps_per_epoch} steps) =====", flush=True)
        net, med_step, last_loss = run_variant(
            use_mask, model_cfg, train_config, mf_config, val_sample_config, args)
        mean_r, mean_nrmse = recovery_metrics(net, *val[:-1], args)
        results[label] = dict(step_ms=med_step, loss=last_loss, R=mean_r, nrmse=mean_nrmse)
        print(f"[{label}] median step={med_step:.1f} ms | final loss={last_loss:.4f} "
              f"| recovery R={mean_r:.4f} | NRMSE={mean_nrmse:.4f}", flush=True)
        del net
        torch.cuda.empty_cache()

    if args.variant != "both":
        return  # single-variant launch (parallel mode): per-variant line already printed

    on, off = results["mask_ON"], results["mask_OFF"]
    print("\n" + "=" * 72)
    print(f"{'variant':<10}{'step (ms)':>12}{'speedup':>10}{'recovery R':>14}{'NRMSE':>10}")
    print("-" * 72)
    print(f"{'mask_ON':<10}{on['step_ms']:>12.1f}{'1.00x':>10}{on['R']:>14.4f}{on['nrmse']:>10.4f}")
    spd = on['step_ms'] / off['step_ms'] if off['step_ms'] else float('nan')
    print(f"{'mask_OFF':<10}{off['step_ms']:>12.1f}{spd:>9.2f}x{off['R']:>14.4f}{off['nrmse']:>10.4f}")
    print("=" * 72)
    dR = off['R'] - on['R']
    dN = off['nrmse'] - on['nrmse']
    print(f"\nmask_OFF vs mask_ON:  {spd:.2f}x faster/step  |  "
          f"dR={dR:+.4f} (higher=better)  |  dNRMSE={dN:+.4f} (lower=better)")
    print("Verdict: drop the mask only if dR is ~0 (or positive) and dNRMSE is not meaningfully worse.")


if __name__ == "__main__":
    import sys
    main()
    # Hard-exit after flushing: bypasses noisy numba/DataLoader worker teardown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
