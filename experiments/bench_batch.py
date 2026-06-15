"""Batch-size sweep: find the GPU 'knee' where per-step time stops being flat.

Pure-GPU timing with fake data (no CPU simulation), mirroring bench_step.py but
sweeping batch size at the production architecture (8L mixed, proj=256, mask ON).

Reports ms/step and ms/sample and peak VRAM per batch. While ms/sample keeps
dropping, a bigger batch buys more simulation budget for ~free; once it flattens,
the GPU is saturated and bigger batch just costs proportionally more time.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m experiments.bench_batch
"""
import os, sys, time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cogformer.networks.transformers.cogformer import CogFormer
from experiments.utils import build_encoder_input_dim

N_PARAMS          = 7
N_COL_TOKENS      = 181
N_TOKENS          = N_PARAMS * N_COL_TOKENS          # 1267
ENCODER_INPUT_DIM = build_encoder_input_dim(5, 5, True)

BATCHES       = [64, 96, 128]
WARMUP_STEPS  = 8
MEASURE_STEPS = 16

device = torch.device("cuda")
print(f"Device: {torch.cuda.get_device_name(0)}  ({torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB)")
print(f"encoder_input_dim={ENCODER_INPUT_DIM}  decoder n_tokens={N_TOKENS}\n")
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True


def make_fake_batch(bs, num_obs):
    rng = np.random.default_rng(num_obs * 1000 + bs)
    return {
        "input_data":        torch.from_numpy(rng.standard_normal((bs, num_obs, ENCODER_INPUT_DIM)).astype(np.float32)),
        "param_indices":     torch.from_numpy(rng.uniform(0, 1, (bs, N_TOKENS, 1)).astype(np.float32)),
        "regressor_indices": torch.from_numpy(rng.uniform(0, 1, (bs, N_TOKENS, 1)).astype(np.float32)),
        "param_masks":       torch.ones(bs, N_TOKENS, dtype=torch.float32),
        "param_matrices":    torch.from_numpy(rng.standard_normal((bs, N_TOKENS, 1)).astype(np.float32)),
        "model_ids":         None,
    }


def time_batch(bs):
    cfg = dict(
        encoder_input_dim=ENCODER_INPUT_DIM, encoder_num_layers=8, decoder_num_layers=8,
        encoder_num_heads=8, decoder_num_heads=8, encoder_embed_dim=64, num_seeds=32,
        seed_dim=64, proj_dim=256, dropout=0.05, layer_dropout=0.05,
        decoder_layer_design="mixed_attention", decoder_layer_kwargs={"mab_first": True},
        time_embedding_dim=32, pos_embedding_dim=32, self_attn_padding_mask=True,  # mask ON (production)
    )
    net = CogFormer(**cfg).to(device).train()
    net = torch.compile(net, dynamic=True)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, fused=True)

    rng = np.random.default_rng(99)
    obs = rng.integers(200, 1001, size=WARMUP_STEPS + MEASURE_STEPS).tolist()
    batches = [make_fake_batch(bs, n) for n in obs]
    torch.cuda.reset_peak_memory_stats()

    def step(b):
        b = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pv, tv = net(b["param_matrices"], b["input_data"], b["param_indices"],
                         b["regressor_indices"], b["param_masks"])
            loss = net.compute_loss(pv, tv, b["param_masks"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        torch.cuda.synchronize()

    for b in batches[:WARMUP_STEPS]:
        step(b)
    times = []
    for b in batches[WARMUP_STEPS:]:
        t0 = time.perf_counter(); step(b); times.append((time.perf_counter() - t0) * 1000)

    med = float(np.median(times))
    peak = torch.cuda.max_memory_allocated() / 1e9
    del net, opt; torch.cuda.empty_cache()
    return med, peak


print(f"{'batch':>6} {'ms/step':>9} {'ms/sample':>10} {'peakVRAM':>9} {'samples/s':>10}")
prev = None
for bs in BATCHES:
    med, peak = time_batch(bs)
    per = med / bs
    sps = bs / (med / 1000)
    tag = ""
    if prev is not None:
        drop = (prev - per) / prev * 100
        tag = f"  ms/sample {drop:+.0f}% vs prev"
    print(f"{bs:>6} {med:>9.1f} {per:>10.3f} {peak:>7.1f}GB {sps:>10.0f}{tag}")
    prev = per
