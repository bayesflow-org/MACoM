"""
Architectural speedup benchmark.

Key insight: 4 decoder SABs (self-attention on 1267 tokens, O(n^2)) dominate cost.
Compilation strategies can't escape this. Architectural levers are the only option.

Tests:
  - Baseline: 8-layer mixed_attention decoder, proj_dim=256
  - Cross-attn only: no SABs (all MABs), O(n x 32) instead of O(n^2)
  - Fewer decoder layers: 4 instead of 8
  - Smaller proj_dim: 128 instead of 256
  - Combined: fewer layers + smaller proj_dim

Usage:
    cd /home/a_huangm13/Documents/Research/Projects/CogFormer-2
    python -m experiments.bench_step
"""
import os, sys, time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogformer.networks.transformers.cogformer import CogFormer
from experiments.utils import build_encoder_input_dim

BATCH_SIZE         = 64
N_PARAMS           = 7
N_COL_TOKENS       = 181
N_TOKENS           = N_PARAMS * N_COL_TOKENS   # 1267
ENCODER_INPUT_DIM  = build_encoder_input_dim(5, 5, True)   # 184

WARMUP_STEPS  = 8
MEASURE_STEPS = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}  GPU: {torch.cuda.get_device_name(0)}")
print(f"Encoder input dim: {ENCODER_INPUT_DIM}, decoder n_tokens: {N_TOKENS}\n")
torch.set_float32_matmul_precision("high")


def make_fake_batch(num_obs: int):
    rng = np.random.default_rng(num_obs)
    return {
        "input_data":        torch.from_numpy(rng.standard_normal((BATCH_SIZE, num_obs, ENCODER_INPUT_DIM)).astype(np.float32)),
        "param_indices":     torch.from_numpy(rng.integers(0, 8, (BATCH_SIZE, N_TOKENS, 1))),
        "regressor_indices": torch.from_numpy(np.concatenate([
                                 rng.integers(0, 4, (BATCH_SIZE, N_TOKENS, 2)),      # reg_a, reg_b
                                 rng.integers(0, 4, (BATCH_SIZE, N_TOKENS, 2)),      # slot_a, slot_b
                                 rng.integers(0, N_TOKENS, (BATCH_SIZE, N_TOKENS, 1)),  # row_pos
                             ], axis=-1)),
        "param_masks":       torch.ones(BATCH_SIZE, N_TOKENS, dtype=torch.float32),
        "param_matrices":    torch.from_numpy(rng.standard_normal((BATCH_SIZE, N_TOKENS, 1)).astype(np.float32)),
        "model_ids":         None,
    }


def build_and_time(label, cfg_overrides, compile_strategy="dynamic"):
    cfg = dict(
        encoder_input_dim    = ENCODER_INPUT_DIM,
        encoder_num_layers   = 8,
        decoder_num_layers   = 8,
        encoder_num_heads    = 8,
        decoder_num_heads    = 8,
        encoder_embed_dim    = 64,
        num_seeds            = 32,
        seed_dim             = 64,
        proj_dim             = 256,
        dropout              = 0.05,
        layer_dropout        = 0.05,
        decoder_layer_design = "mixed_attention",
        decoder_layer_kwargs = {"mab_first": True},
        time_embedding_dim   = 32,
        pos_embedding_dim    = 32,
    )
    cfg.update(cfg_overrides)

    net = CogFormer(**cfg).to(device).train()
    if compile_strategy == "dynamic":
        net = torch.compile(net, dynamic=True)
    elif compile_strategy == "split":
        net.encoder = torch.compile(net.encoder, dynamic=True)
        net.decoder = torch.compile(net.decoder, dynamic=False)

    n_params_total = sum(p.numel() for p in net.parameters()) / 1e6

    rng = np.random.default_rng(99)
    num_obs_list = rng.integers(200, 1001, size=WARMUP_STEPS + MEASURE_STEPS).tolist()
    batches = [make_fake_batch(n) for n in num_obs_list]

    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, fused=True)

    print(f"\n{'='*65}")
    print(f"{label}  [{n_params_total:.1f}M params]")
    print(f"  decoder_num_layers={cfg['decoder_num_layers']}, "
          f"proj_dim={cfg['proj_dim']}, "
          f"decoder_layer_design={cfg.get('decoder_layer_design','mixed_attention')}")

    times = []
    print(f"  Warmup ({WARMUP_STEPS})...", end="", flush=True)
    for batch in batches[:WARMUP_STEPS]:
        b = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred_v, tgt_v = net(b["param_matrices"], b["input_data"],
                                b["param_indices"], b["regressor_indices"], b["param_masks"])
            loss = net.compute_loss(pred_v, tgt_v, b["param_masks"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        optimizer.step()
        torch.cuda.synchronize()
    print(" done")

    print(f"  Measuring ({MEASURE_STEPS})...", end="", flush=True)
    for batch in batches[WARMUP_STEPS:]:
        b = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred_v, tgt_v = net(b["param_matrices"], b["input_data"],
                                b["param_indices"], b["regressor_indices"], b["param_masks"])
            loss = net.compute_loss(pred_v, tgt_v, b["param_masks"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        optimizer.step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    print(" done")

    med = np.median(times)
    p25, p75 = np.percentile(times, 25), np.percentile(times, 75)
    print(f"  Median={med:.1f} ms  (IQR [{p25:.1f}, {p75:.1f}])  "
          f"~{med/1000*100:.1f} s/epoch")

    del net
    torch.cuda.empty_cache()
    return med, n_params_total


if __name__ == "__main__":
    configs = [
        # label,                            cfg_overrides,                                              strategy
        ("BASELINE: 8L mixed, proj=256",    {},                                                         "dynamic"),
        ("enc heads=2 (head_dim=32)",       {"encoder_num_heads": 2},                                  "dynamic"),
        ("cross-attn only (no SABs)",       {"decoder_layer_design": "cross_attention",
                                             "decoder_layer_kwargs": {}},                               "dynamic"),
        ("4 decoder layers",                {"decoder_num_layers": 4},                                  "dynamic"),
        ("proj_dim=128",                    {"proj_dim": 128, "decoder_num_heads": 4},                  "dynamic"),
        ("4L + proj=128",                   {"decoder_num_layers": 4, "proj_dim": 128,
                                             "decoder_num_heads": 4},                                   "dynamic"),
        ("cross-attn + enc heads=2",        {"decoder_layer_design": "cross_attention",
                                             "decoder_layer_kwargs": {},
                                             "encoder_num_heads": 2},                                   "dynamic"),
    ]

    results = []
    for label, overrides, strategy in configs:
        med, nparams = build_and_time(label, overrides, strategy)
        results.append((label, med, nparams))

    baseline_ms = results[0][0][0] if isinstance(results[0][0], tuple) else results[0][1]
    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    baseline_ms = results[0][1]
    for label, ms, nparams in results:
        speedup = baseline_ms / ms
        print(f"  {label:<42s} {ms:6.1f} ms  {speedup:.2f}x  {nparams:.1f}M params")
