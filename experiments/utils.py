"""Shared utility functions for CogFormer experiment scripts."""

import math
import os

# Expandable segments let the CUDA caching allocator grow/shrink segments instead
# of binning fixed-size blocks. With per-batch shape variation (num_obs and, after
# token compaction, token count) the binned allocator fragments — reserved-but-
# unallocated memory creeps up and can OOM despite free VRAM. Must be set before
# the first CUDA allocation; setdefault respects an explicit user override.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import hashlib

import numpy as np
import torch
import torch.utils.data

from cogformer.simulator.design_utils import DESIGN_GENERATORS


def seed_case(base_seed: int | None, *keys: object, salt: str = "") -> None:
    """Seed numpy and torch for one validation case, order-independently.

    Comparing two checkpoints on the same test sets requires the sampled datasets to be
    identical across runs that build *different* ModelClass objects (3-family pretrained
    vs 4-family finetuned) and iterate cases in a different order. Advancing one global
    RNG stream cannot do that, because the nth case is not the same case in both runs.
    So the seed is derived from the case's identity (family, scenario) rather than from
    its position, and re-seeding per case makes each one independently reproducible.

    blake2b rather than ``hash()``: Python randomizes string hashing per process, which
    would silently give different seeds on every invocation.

    NOTE: this only determinises sampling that runs in THIS process. ``batch_sample``
    dispatches to a process pool when ``num_workers > 1`` (model_family.py:479), and
    those workers carry their own RNG state with nondeterministic task assignment, so a
    seeded run must build its ModelFamily objects with ``num_workers=1``.
    """
    if base_seed is None:
        return
    key = ("|".join(str(k) for k in keys) + "|" + salt).encode()
    digest = hashlib.blake2b(key, digest_size=4).digest()
    seed = (int(base_seed) + int.from_bytes(digest, "big")) % (2 ** 32)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_seed_arg(parser) -> None:
    """Register --seed, shared by the validation scripts (see ``seed_case``)."""
    parser.add_argument("--seed", type=int, default=None,
                        help="Base seed for test-set generation. When set, each case is "
                             "seeded from its (family, scenario) identity, so two runs "
                             "produce identical datasets and their metrics can be "
                             "compared as a paired difference. Forces serial simulation.")


def recommend_batch_size(base: int = 64) -> int:
    """Suggest a batch size based on available GPU VRAM.

    CogFormer's O(N²) decoder self-attention (N≈1267 tokens) saturates GPU memory
    bandwidth even at small batch sizes, so throughput per step is roughly flat
    regardless of batch size.  Larger batches are a *gradient quality* lever, not
    a wall-clock speed lever — only use them if you also reduce epochs proportionally.

    Formula: scale by sqrt(vram / 8 GB) so the increase is conservative.
    Cap at 4× base to avoid OOM on attention-heavy architectures.
    Falls back to base when no CUDA device is present.
    """
    if not torch.cuda.is_available():
        return base
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    scale = max(1.0, math.sqrt(vram_gb / 8.0))
    return int(min(base * 4, 2 ** round(math.log2(base * scale))))


def recommend_sim_workers(backend: str | None = None) -> int:
    """Number of DataLoader simulation workers scaled to available CPU cores.

    numba backend: prefer many workers with ONE numba thread each over few workers with
    many: multi-threaded numba pays a fork-join sync on every simulate() call (once per
    batch element), which measured 2.4x slower than element-level process parallelism
    at equal core counts (24-core box: 6w x 4t = 3.3 batches/s, 20w x 1t = 7.8
    batches/s). Leave a few cores for the main process and GPU feeding.

    gpu backend: the kernels run on the GPU, leaving each worker only the design/prior
    draws, contamination and adapt (~6% of the numba-era per-batch CPU work), and every
    worker holds its own CUDA context (~0.3-0.5 GB). A handful of workers is enough.
    """
    if backend is None:
        from cogformer.simulator.model import _default_backend
        backend = os.environ.get("COGFORMER_SIM_BACKEND") or _default_backend()
    cpu = os.cpu_count() or 4
    if backend == "gpu":
        return max(2, min(6, cpu - 4))
    return max(3, cpu - 4)


def add_sim_backend_arg(parser) -> None:
    """Register --sim_backend (see cogformer.simulator.model.Model.sim_backend)."""
    parser.add_argument("--sim_backend", choices=["numba", "gpu"], default=None,
                        help="Simulation backend. 'gpu' runs the Triton kernels, batched per "
                             "family per batch; distributionally equivalent to numba "
                             "(experiments/check_gpu_simulators.py). Default: "
                             "$COGFORMER_SIM_BACKEND, else gpu when CUDA is available.")


def add_decoder_groups_arg(parser) -> None:
    """Register --decoder_groups (see plan_token_groups)."""
    parser.add_argument("--decoder_groups", type=int, default=3,
                        help="Split each training batch into this many size-sorted decoder "
                             "groups, each decoded at its own token width (exact; see "
                             "plan_token_groups). 3 measured 1.82x on the r4c4 b128 step; "
                             "0 or 1 disables.")


def apply_sim_backend(args) -> str:
    """Export the chosen backend so models AND spawned simulation workers pick it up."""
    from cogformer.simulator.model import _default_backend
    if getattr(args, "sim_backend", None):
        os.environ["COGFORMER_SIM_BACKEND"] = args.sim_backend
    return os.environ.get("COGFORMER_SIM_BACKEND") or _default_backend()


def set_numba_threads(n: int) -> None:
    try:
        import numba
        # NUMBA_NUM_THREADS can cap the pool below the machine's CPU count.
        numba.set_num_threads(min(max(1, n), numba.config.NUMBA_NUM_THREADS))
    except ImportError:
        pass


def _identity_collate(x):
    return x


def make_sim_loader(dataset, num_workers):
    """Build a DataLoader for an infinite simulation IterableDataset."""
    use_mp = num_workers > 0
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        collate_fn=_identity_collate,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **(dict(prefetch_factor=2, persistent_workers=True,
                multiprocessing_context="spawn") if use_mp else {}),
    )


class ModelFamilyDataset(torch.utils.data.IterableDataset):
    """Infinite simulation dataset for a single ModelFamily.

    Each spawned worker initializes its own ModelFamily and Adapter on first
    iteration, then yields pre-adapted CPU batches indefinitely.
    """

    def __init__(self, model_cfg, train_config, num_workers, world_size=1):
        self._name = model_cfg["name"]
        self._model_cls = model_cfg["model_cls"]
        self._prior_fn = model_cfg["prior_fn"]
        self._link_fn = model_cfg["link_fn"]
        self._holdout_configs = model_cfg.get("holdout_configs")
        self._train_context = model_cfg.get("train_context")
        self._batch_size = train_config["batch_size"]
        self._mf_config = train_config["model_family_config"]
        self._sample_config = train_config["train_sample_config"]
        self._free_prob = train_config.get("train_free_prob")
        self._slope_patterns = train_config.get("train_slope_patterns")
        self._n_workers = num_workers
        self._world = world_size

    def __iter__(self):
        from cogformer.simulator import ModelFamily
        from cogformer.adapter import Adapter
        info = torch.utils.data.get_worker_info()
        n_workers = info.num_workers if info else 1
        num_par = max(self._world, int(os.environ.get("NUM_GPU_JOBS", "1")))
        set_numba_threads((os.cpu_count() or 1) // (n_workers * num_par))
        mf = ModelFamily(
            name=self._name, model=self._model_cls(),
            prior=self._prior_fn(), num_workers=1,
        )
        if self._holdout_configs:
            mf.set_holdout_configs(self._holdout_configs)
        adapter = Adapter()
        while True:
            samples = mf.batch_sample(
                **self._mf_config,
                **self._sample_config,
                prior=mf.prior,
                batch_size=self._batch_size,
                flatten_param_outputs=True,
                link_fun=self._link_fn(),
                context=self._train_context,
                free_prob=self._free_prob,
                allowed_slope_patterns=self._slope_patterns,
            )
            yield adapter.adapt(samples, device="cpu", compact_tokens=True)


class ModelClassDataset(torch.utils.data.IterableDataset):
    """Infinite simulation dataset for a ModelClass (multiple ModelFamilies).

    Each spawned worker initializes its own ModelClass and Adapter on first iteration.
    """

    def __init__(self, model_class_config, train_config, num_workers, world_size=1):
        self._families_cfg = model_class_config["families"]
        self._link_fns = model_class_config["link_funs"]
        self._train_contexts = model_class_config.get("train_contexts")
        self._weights = model_class_config.get("weights")
        self._batch_size = train_config["batch_size"]
        self._mf_config = train_config["model_family_config"]
        self._sample_config = train_config["train_sample_config"]
        self._n_workers = num_workers
        self._world = world_size
        self._decoder_groups = int(train_config.get("decoder_groups", 0) or 0)

    def __iter__(self):
        from cogformer.simulator import ModelFamily, ModelClass
        from cogformer.adapter import Adapter
        info = torch.utils.data.get_worker_info()
        n_workers = info.num_workers if info else 1
        num_par = max(self._world, int(os.environ.get("NUM_GPU_JOBS", "1")))
        set_numba_threads((os.cpu_count() or 1) // (n_workers * num_par))
        families = {
            name: ModelFamily(
                # model_kwargs lets a family be constructed with non-default settings
                # (e.g. LBA(fixed_A=...)) from inside a spawned worker, where the
                # already-instantiated object from the parent process is unavailable.
                model=cfg["model_cls"](**cfg.get("model_kwargs", {})), name=name,
                prior=cfg["prior_fn"](),
                intrinsic_params=cfg["intrinsic_params"],
                mask_randomizer_kwargs=cfg["mask_randomizer_kwargs"],
                num_workers=1,
            )
            for name, cfg in self._families_cfg.items()
        }
        mc = ModelClass(
            model_families=families,
            link_funs={k: fn() for k, fn in self._link_fns.items()},
            train_contexts=self._train_contexts,
            weights=self._weights,
        )
        adapter = Adapter()
        while True:
            samples = mc.batch_sample(
                **self._mf_config,
                **self._sample_config,
                batch_size=self._batch_size,
                flatten_param_outputs=True,
            )
            adapted = adapter.adapt(samples, device="cpu", compact_tokens=True)
            if self._decoder_groups > 1:
                adapted = plan_token_groups(adapted, self._decoder_groups)
            yield adapted


def plan_token_groups(adapted: dict, num_groups: int) -> dict:
    """Sort a compact-token batch by active rows and split it into decoder groups.

    Returns the batch reordered (datasets sorted by active design rows; the loss is a
    mean over datasets, so order is irrelevant) with ``token_groups`` = [(start, end,
    width)]: contiguous slices, each decoded at its own token width (its largest
    dataset's active rows, rounded up to a multiple of 8 like the adapter's compaction,
    times n_params). Split points minimise the total padded tokens sum(n_g * width_g).
    A size-sorted split is exact -- see CogFormer.forward(token_groups=...) -- and cut
    the r4c4 b128 training step 103 -> 57 ms at 3 groups (2026-09-28; real batches
    carried 6x padding waste).
    """
    n_params = adapted["n_params"]
    active = adapted["num_active_rows"].numpy().astype(np.int64)
    batch_size = active.shape[0]
    max_rows = adapted["param_masks"].shape[1] // n_params
    order = np.argsort(active, kind="stable")
    rows = np.minimum(-(-np.maximum(active[order], 1) // 8) * 8, max_rows)   # sorted, padded
    k = max(1, min(num_groups, batch_size))

    # dp[g][j]: min padded rows covering the first j datasets with g groups.
    inf = np.iinfo(np.int64).max // 4
    dp = np.full((k + 1, batch_size + 1), inf, dtype=np.int64)
    cut = np.zeros((k + 1, batch_size + 1), dtype=np.int64)
    dp[0, 0] = 0
    for g in range(1, k + 1):
        for j in range(g, batch_size + 1):
            i = np.arange(g - 1, j)
            cost = dp[g - 1, i] + (j - i) * rows[j - 1]
            best = int(np.argmin(cost))
            dp[g, j], cut[g, j] = cost[best], i[best]
    bounds, j = [], batch_size
    for g in range(k, 0, -1):
        i = int(cut[g, j])
        if j > i:
            bounds.append((i, j))
        j = i
    bounds.reverse()

    groups = []
    for i, j in bounds:   # ties in the DP can split a run of equal widths: merge it back
        width = int(rows[j - 1]) * n_params
        if groups and groups[-1][2] == width:
            groups[-1] = (groups[-1][0], j, width)
        else:
            groups.append((i, j, width))

    perm = torch.from_numpy(order)
    out = {key: (val[perm] if torch.is_tensor(val) and val.dim() > 0 and val.shape[0] == batch_size else val)
           for key, val in adapted.items()}
    out["token_groups"] = groups
    return out


def _git_commit() -> str:
    import subprocess
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"]).returncode != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def init_wandb(project: str, name: str, config: dict, tags: list[str],
               run_id: str | None = None):
    """Start (or, given ``run_id``, continue) a wandb run for a CogFormer training.

    run_id comes from the checkpoint being resumed (trainers store ``wandb.run.id`` in
    every checkpoint), so an interrupted training keeps ONE continuous run instead of a
    new one per restart -- while a fresh training can never attach to an old run.
    Records what the numbers depend on beyond the hyperparameters: simulation backend,
    host and git commit (``-dirty`` if the tree had uncommitted changes).
    """
    import socket
    import wandb
    from cogformer.simulator.model import _default_backend
    return wandb.init(
        project=project, name=name, tags=tags, id=run_id,
        resume="allow" if run_id else None,
        config={**config,
                "sim_backend": os.environ.get("COGFORMER_SIM_BACKEND") or _default_backend(),
                "git_commit": _git_commit(), "host": socket.gethostname()},
    )


def wandb_run_id_from(checkpoint_path) -> str | None:
    """The wandb run id stored in a checkpoint, if resuming from one that has it."""
    from pathlib import Path
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        return None
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return ckpt.get("wandb_run_id") if isinstance(ckpt, dict) else None


def compile_cogformer(model, mode: str, decoder_groups: int = 0):
    """torch.compile a CogFormer for training.

    Without decoder groups the whole model is one graph (the long-standing setup). With
    them, the encoder and decoder are compiled SEPARATELY, in place, and the per-group
    loop in CogFormer.forward stays eager: compiled as one graph, the data-dependent
    group slicing trips an Inductor assertion on symbolic sizes (torch 2.11,
    2026-09-28). In-place Module.compile also keeps state_dict keys free of the
    ``_orig_mod.`` prefix; every loader strips it either way.
    """
    if decoder_groups and decoder_groups > 1:
        model.encoder.compile(dynamic=True, mode=mode)
        model.decoder.compile(dynamic=True, mode=mode)
        return model
    return torch.compile(model, dynamic=True, mode=mode)


def train_step(net, adapted, config, optimizer, scheduler):
    """One flow-matching gradient step; returns (loss_scalar, current_lr)."""
    optimizer.zero_grad(set_to_none=True)
    device_type = config["device"].type
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=(device_type == "cuda")):
        pred_velocity, target_velocity = net(
            adapted["param_matrices"][..., None],
            adapted["input_data"],
            adapted["param_indices"],
            adapted["regressor_indices"],
            adapted["param_masks"],
            model_ids=adapted.get("model_ids"),
            token_groups=adapted.get("token_groups"),
        )
        _module = getattr(net, 'module', net)  # handles DDP and DataParallel
        L = _module.compute_loss(pred_velocity, target_velocity, adapted["param_masks"])
    L.backward()
    if config["gradient_clip_norm"] is not None:
        torch.nn.utils.clip_grad_norm_(net.parameters(), config["gradient_clip_norm"])
    optimizer.step()
    return L.detach().item(), scheduler.get_last_lr()[0]


def add_interaction_prior_args(parser):
    """Explicit random-design structural prior; legacy defaults remain unchanged."""
    parser.add_argument("--interaction_policy", choices=["all", "optional"], default="all",
                        help="Random configs: 'all' forces every eligible parameter interaction; "
                             "'optional' samples each eligible (regressor pair, parameter). "
                             "Explicit design configs are unchanged.")
    parser.add_argument("--interaction_prob", type=float, default=None,
                        help="Inclusion probability with --interaction_policy optional. "
                             "Default: one Beta(2,2) draw per design, shared across eligible interactions.")


def add_structure_conditioning_arg(parser):
    parser.add_argument("--encoder_structure_conditioning", action="store_true",
                        help="Condition the encoder on joint active parameter/term identities. "
                             "Experimental architecture; must also be enabled when loading its checkpoint.")


def structure_training_schema(model_family_config, encoder_structure_conditioning=False):
    """Serializable structural prior and encoder contract for resumable training."""
    enabled = bool(model_family_config.get("add_interaction", False))
    policy = model_family_config.get("interaction_policy", "all") if enabled else "all"
    probability = model_family_config.get("interaction_prob") if enabled else None
    if policy not in ("all", "optional"):
        raise ValueError("interaction_policy must be 'all' or 'optional'")
    if probability is not None and (policy != "optional" or not math.isfinite(probability) or not 0 <= probability <= 1):
        raise ValueError("interaction_prob requires optional policy and a finite probability in [0, 1]")
    return dict(version=1, add_interaction=enabled, interaction_policy=policy,
                interaction_prob=probability, encoder_structure_conditioning=bool(encoder_structure_conditioning))


def validate_structure_resume(checkpoint, requested):
    """Old main-trainer checkpoints used forced interactions and no encoder context."""
    legacy = structure_training_schema({"add_interaction": True})
    saved = checkpoint.get("structure_schema", legacy) if isinstance(checkpoint, dict) else legacy
    if saved != requested:
        raise ValueError("Checkpoint structural prior/encoder conditioning differs from this run. "
                         "Resume with matching settings or start a separate training run.")


def add_architecture_args(parser):
    """Register shared CogFormer architecture arguments."""
    parser.add_argument("--encoder_num_layers", type=int, default=8)
    parser.add_argument("--decoder_num_layers", type=int, default=8)
    parser.add_argument("--encoder_num_heads", type=int, default=8)
    parser.add_argument("--decoder_num_heads", type=int, default=8)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--num_seeds", type=int, default=32)
    parser.add_argument("--time_embedding_dim", type=int, default=32)
    parser.add_argument("--pos_embedding_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--layer_dropout", type=float, default=0.05)
    parser.add_argument("--compile_mode", type=str, default="max-autotune-no-cudagraphs",
                        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])


def add_training_args(parser):
    """Register shared training schedule and design-space arguments."""
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_batch_size", type=int, default=recommend_batch_size(64))
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--min_num_obs", type=int, default=200)
    parser.add_argument("--num_obs_bucket", type=int, default=1,
                        help="Quantize num_obs to a multiple of this so batch shapes repeat. "
                             "Uniform num_obs yields ~300 distinct shapes, and torch.compile "
                             "max-autotune re-tunes per new shape (spikes to 25s against an "
                             "88ms steady-state step). Measured: this is a ONE-OFF ~130s "
                             "warmup cost, not per-step -- steady-state throughput is identical "
                             "with and without bucketing (11.19 vs 10.96 it/s), so the default "
                             "is off. Set e.g. 64 only to shorten warmup on very short runs, "
                             "at the cost of narrowing the num_obs distribution.")
    parser.add_argument("--dry_run_epochs", type=int, default=None,
                        help="If set, train for this many epochs only and record extrapolated time.")
    parser.add_argument("--max_num_regressors", type=int, default=5)
    parser.add_argument("--max_num_categories", type=int, default=5)
    parser.add_argument("--keep_intercept", action="store_true", default=True)
    parser.add_argument("--add_interaction", action="store_true", default=True)
    parser.add_argument("--design_generator", choices=DESIGN_GENERATORS, default="mixture",
                        help="Design-matrix construction. 'mixture' (default) draws category "
                             "assignments first, then one Gaussian per category: nominal categories, "
                             "dialable association, balanced factorials. 'quantile' is the "
                             "pre-2026-07 shared-latent cut; 'gaussian_softmax' uses joint "
                             "LKJ-correlated Gaussian scores and categorical softmax blocks.")
    parser.add_argument("--coefficient_order_scale", type=float, default=1.0,
                        help="Effect-hierarchy multiplier: a two-way interaction block is "
                             "scaled by this factor. 1.0 (default) imposes no hierarchy, which "
                             "is what 'one Gaussian per category' implies on its own -- at r4c4 "
                             "that leaves interactions carrying ~69%% of design-driven predictor "
                             "variance. 0.816 equalizes interaction and main-effect contributions; "
                             "0.577 puts interactions at half. Only used with "
                             "--coefficient_prior cell_means.")
    parser.add_argument("--coefficient_prior", choices=["cell_means", "iid"], default="cell_means",
                        help="Coefficient prior. 'cell_means' (default) draws one latent Gaussian "
                             "per category and takes contrasts as differences, making categories "
                             "exchangeable. 'iid' is the original per-row N(0, 1), under which the "
                             "reference level is a priori closer to every other level. Same scale.")


def compact_adapted(adapted):
    """Drop fully masked design rows from a padded adapted batch, on its device.

    The GPU-side equivalent of Adapter.adapt(..., compact_tokens=True): active rows
    first in canonical order, padded to the batch maximum rounded up to a multiple
    of 8. Exact for sampling -- the decoder masks inactive tokens as self-attention
    keys and zeroes their outputs, so active-token velocities do not depend on them
    (verified 2026-09-28: max relative velocity difference 7e-7 in fp32).
    Returns a new dict carrying row_gather_indices / num_active_rows /
    num_design_rows, which chunked_sample uses to scatter back.
    """
    n_params = adapted["n_params"]
    masks = adapted["param_masks"]
    batch_size, num_tokens = masks.shape
    n_rows = num_tokens // n_params

    row_active = (masks.reshape(batch_size, n_rows, n_params) > 0).any(dim=-1)   # (B, R)
    num_active = row_active.sum(dim=1)
    compact_rows = max(int(num_active.max().item()), 1)
    compact_rows = min(-(-compact_rows // 8) * 8, n_rows)
    # Stable sort puts active rows first, each group in canonical order.
    order = torch.sort((~row_active).to(torch.int8), dim=1, stable=True).indices
    gather = order[:, :compact_rows]                                               # (B, Rc)
    pad = torch.arange(compact_rows, device=masks.device)[None, :] >= num_active[:, None]
    gather = gather.masked_fill(pad, 0)

    tok = (gather[:, :, None] * n_params
           + torch.arange(n_params, device=masks.device)).reshape(batch_size, -1)
    pad_tok = pad.repeat_interleave(n_params, dim=1)

    def take(x):
        idx = tok if x.dim() == 2 else tok[..., None].expand(-1, -1, x.shape[-1])
        return torch.gather(x, 1, idx)

    regressor_idx = take(adapted["regressor_indices"]).masked_fill(pad_tok[..., None], 0)
    return {
        **adapted,
        "param_indices": take(adapted["param_indices"]),
        "regressor_indices": regressor_idx,
        "param_masks": take(masks).masked_fill(pad_tok, 0),
        "param_matrices": take(adapted["param_matrices"]).masked_fill(pad_tok, 0),
        "row_gather_indices": gather,
        "num_active_rows": num_active,
        "num_design_rows": n_rows,
    }


MAX_SAMPLE_TOKENS_PER_PASS = 150_000   # B * samples_chunk_size * tokens, see chunked_sample


def chunked_sample(model, adapted, num_samples, steps, solver="euler",
                   samples_chunk_size=None, compact=True):
    """Call model.sample() in chunks of samples_chunk_size to limit peak GPU memory.

    Peak memory scales with batch_size * samples_chunk_size * num_tokens.
    If samples_chunk_size is None the full num_samples are drawn in one pass.

    compact : sample on active design rows only (compact_adapted) and scatter back to
        the canonical layout. Exact, and 15x faster on padded r4c4 grids (603 -> 72
        tokens). Output shape is unchanged; entries of fully inactive rows are 0
        instead of untouched prior noise (they are masked everywhere downstream).
    """
    if compact and "row_gather_indices" not in adapted:
        adapted = compact_adapted(adapted)
    input_data      = adapted["input_data"]
    param_indices   = adapted["param_indices"]
    regressor_idx   = adapted["regressor_indices"]
    params_mask     = adapted["param_masks"]
    model_ids       = adapted.get("model_ids")

    # Cap the chunk by sequence-tokens per pass. Small designs are launch-bound and gain
    # ~2x from chunking; large ones are compute-bound at chunk 1 and only gain memory
    # (RK4, B=200: 576 tokens peaks 2.7 GiB at chunk 1, 11.9 at 5, 23.7 at 10; measured
    # 2026-10-03). At B=200 the cap keeps chunk 5 up to 150 tokens and is 1 from ~375.
    if samples_chunk_size is not None:
        num_seq_tokens = param_indices.shape[0] * param_indices.shape[1]
        samples_chunk_size = max(1, min(samples_chunk_size,
                                        MAX_SAMPLE_TOKENS_PER_PASS // num_seq_tokens))

    if samples_chunk_size is None or samples_chunk_size >= num_samples:
        pred = model.sample(
            input_data, param_indices, regressor_idx, params_mask,
            num_samples=num_samples, steps=steps, solver=solver,
            model_ids=model_ids,
        )
    else:
        chunks = []
        for s_start in range(0, num_samples, samples_chunk_size):
            s_count = min(samples_chunk_size, num_samples - s_start)
            chunks.append(model.sample(
                input_data, param_indices, regressor_idx, params_mask,
                num_samples=s_count, steps=steps, solver=solver,
                model_ids=model_ids,
            ))
        pred = np.concatenate(chunks, axis=1)

    # Batches adapted with compact_tokens=True carry only active design rows;
    # scatter predictions back to the canonical row layout for downstream analysis.
    if "row_gather_indices" in adapted:
        from cogformer.adapter import Adapter
        pred = Adapter.scatter_to_design_rows(
            pred,
            adapted["row_gather_indices"].cpu().numpy(),
            adapted["num_active_rows"].cpu().numpy(),
            adapted["num_design_rows"],
            adapted["n_params"],
        )
    return pred


def infer_free_fixed_intrinsics(
    design_config: dict,
    all_intrinsics: list[str],
    default_fixed_values: dict[str, float],
) -> tuple[list[str], list[str], dict[str, float]]:
    """Partition intrinsics into free/fixed based on which appear in design_config.

    Returns (free_intrinsics, fixed_intrinsics, fixed_values).
    """
    used = {p for params in design_config.values() for p in params}
    free = [p for p in all_intrinsics if p in used]
    fixed = [p for p in all_intrinsics if p not in used]
    fixed_values = {p: default_fixed_values[p] for p in fixed if p in default_fixed_values}
    return free, fixed, fixed_values


def build_encoder_input_dim(
    max_num_regressors: int,
    max_num_categories: int,
    keep_intercept: bool,
) -> int:
    """Compute the encoder input dimension for a CogFormer model."""
    block_width = max_num_categories - 1
    n_main = max_num_regressors
    n_inter = max_num_regressors * (max_num_regressors - 1) // 2
    design_cols = n_main * block_width + n_inter * block_width ** 2 + (1 if keep_intercept else 0)
    return design_cols + 5  # +5: rt, choices_normalized, alt_col, favored_idx, theta


def add_global_params(
    cfg: dict,
    global_params: list[str],
    global_variable_names: list[str],
) -> dict:
    """Return a copy of cfg with global_params appended to intrinsic_params,
    variable_names, and the intercept ("1") column of benchmark_configs and
    val_scenarios.
    """
    cfg = dict(cfg)
    cfg["intrinsic_params"] = cfg["intrinsic_params"] + global_params
    cfg["variable_names"] = cfg["variable_names"] + global_variable_names
    cfg["global_params"] = list(global_params)

    def _add_to_intercept(dc: dict) -> dict:
        return {
            k: (ps + [p for p in global_params if p not in ps] if k == "1" else list(ps))
            for k, ps in dc.items()
        }

    if "benchmark_configs" in cfg:
        cfg["benchmark_configs"] = {
            name: _add_to_intercept(dc)
            for name, dc in cfg["benchmark_configs"].items()
        }

    if "val_scenarios" in cfg:
        def _process_scenario(s):
            tag, dc, opts = s if len(s) == 3 else (*s, {})
            new_dc = dc if opts.get("skip_global", False) else _add_to_intercept(dc)
            return (tag, new_dc, {k: v for k, v in opts.items() if k != "skip_global"})
        cfg["val_scenarios"] = [_process_scenario(s) for s in cfg["val_scenarios"]]

    if "holdout_configs" in cfg:
        cfg["holdout_configs"] = [
            {**hc, "design_config": _add_to_intercept(hc["design_config"])}
            for hc in cfg["holdout_configs"]
        ]

    return cfg


def resolve_val_scenario(
    scenario_opts: dict,
    training_max_cats: int,
    default_max_cats: int | None = None,
    default_min_cats: int = 2,
) -> tuple[int, int, dict, bool]:
    """Resolve a ``val_scenarios`` opts dict into concrete category bounds + mix kwargs.

    Single source of the per-scenario category/mix handling shared by every CogFormer
    validation path (model_family / model_class train val_step and validate scripts), so
    they can't drift. Returns ``(max_cats, min_cats, mix_opts, skip)``:

    - ``max_cats`` — category ceiling; scenario ``max_num_categories`` else ``default_max_cats``
      (which itself defaults to ``training_max_cats``).
    - ``min_cats`` — floor; scenario ``min_num_categories`` else ``default_min_cats``. When the
      scenario sets ``exact_num_categories`` the floor is pulled up to the ceiling (exact-K).
      Clamped to ``2 <= min_cats <= max_cats`` so ``build_design_matrix``'s randint is valid.
    - ``mix_opts`` — ``{discrete_prob, categorical_regressors}`` passed through to batch_sample
      (fixes the continuous/categorical mix); absent keys keep batch_sample's random default.
    - ``skip`` — True when the scenario needs more categories than the model was trained on
      (the encoder input dim is sized from ``training_max_cats``, so a wider case would crash).
    """
    default_max_cats = default_max_cats or training_max_cats
    max_cats = scenario_opts.get("max_num_categories", default_max_cats)
    min_cats = scenario_opts.get("min_num_categories", default_min_cats)
    if scenario_opts.get("exact_num_categories"):
        min_cats = max_cats
    min_cats = max(2, min(min_cats, max_cats))
    skip = max_cats > training_max_cats
    mix_opts = {k: scenario_opts[k] for k in ("discrete_prob", "categorical_regressors")
                if k in scenario_opts}
    return max_cats, min_cats, mix_opts, skip


def strip_global_params(cfg: dict) -> dict:
    """Return a copy of cfg with global_params removed from intrinsic_params,
    variable_names, free_params, benchmark_configs, and val_scenarios.
    """
    global_p = set(cfg.get("global_params", []))
    if not global_p:
        return cfg

    cfg = dict(cfg)
    keep = [i for i, p in enumerate(cfg["intrinsic_params"]) if p not in global_p]
    cfg["intrinsic_params"] = [cfg["intrinsic_params"][i] for i in keep]
    cfg["variable_names"] = [cfg["variable_names"][i] for i in keep]
    cfg["free_params"] = [p for p in cfg.get("free_params", []) if p not in global_p]
    cfg["global_params"] = []

    def _strip_from(dc: dict) -> dict:
        return {k: [p for p in ps if p not in global_p] for k, ps in dc.items()}

    if "benchmark_configs" in cfg:
        cfg["benchmark_configs"] = {
            name: _strip_from(dc) for name, dc in cfg["benchmark_configs"].items()
        }

    if "val_scenarios" in cfg:
        cfg["val_scenarios"] = [
            (tag, _strip_from(dc), opts) for tag, dc, opts in cfg["val_scenarios"]
        ]

    if "holdout_configs" in cfg:
        cfg["holdout_configs"] = [
            {**hc, "design_config": _strip_from(hc["design_config"])}
            for hc in cfg["holdout_configs"]
        ]

    return cfg


def reshape_bf_to_cf(
    bf_samples: np.ndarray,
    design_config: dict,
    intrinsic_params: list[str],
) -> np.ndarray:
    """Reshape flat BayesFlow posterior samples into (*, num_rows, num_cols) layout.

    bf_samples shape: (*leading, num_active_params)
    Returns shape:    (*leading, num_rows, num_cols)
    """
    *leading, num_active = bf_samples.shape
    num_rows = len(design_config)
    num_cols = len(intrinsic_params)
    col_idx = {p: j for j, p in enumerate(intrinsic_params)}

    result = np.zeros((*leading, num_rows, num_cols))
    flat_pos = 0
    for row_i, active_params in enumerate(design_config.values()):
        ordered = [p for p in intrinsic_params if p in active_params]
        for p in ordered:
            result[..., row_i, col_idx[p]] = bf_samples[..., flat_pos]
            flat_pos += 1

    assert flat_pos == num_active, f"Expected {num_active} active params but mapped {flat_pos}."
    return result


def stimulus_channels(samples) -> dict:
    """Per-trial stimulus identity, for families whose choices are STIMULUS-CODED.

    RDM and LBA draw the favoured accumulator per trial (favored_idx). A choice is only
    informative about the drift DIFFERENCE given which accumulator the stimulus
    favoured, so a network that sees rts/choices without it cannot learn v_diff: its
    choices look like coin flips. Until 2026-09-27 the BF baselines dropped it. CogFormer
    was believed to receive it all along, but ModelClass.collate dropped it from every
    model-class TRAINING batch until 2026-10-05 (validation and real data did pass it).
    Returned as (B, N, 1) float so it concatenates with rts
    and choices; empty for families that carry no per-trial stimulus (DDM, CDM).
    """
    ptc = samples.get("per_trial_context") or {}
    if "favored_idx" not in ptc:
        return {}
    return {"favored_idx": np.asarray(ptc["favored_idx"], dtype=np.float32)[..., None]}


def benchmark_mix_opts(model_key: str, case: str) -> dict:
    """Design-generator options for a bench case, from the family's val_scenarios.

    The BayesFlow baselines and CogFormer's benchmark validation must draw the SAME
    design distribution for a case, or the amortization gap measures a distribution
    shift as well as amortization. Until 2026-09-27 they did not: val_scenarios pins
    regressed / fixed_regressed / interaction to continuous regressors
    (categorical_regressors=[], SD 1.0), while the BF simulator used batch_sample's
    default mixed generator (2-level categorical or continuous, continuous SD ~0.83).
    Measured: per-dataset mean RT differed at p <= 0.007 in exactly those cases.
    Reading the options from val_scenarios keeps a single definition of each case.
    """
    from .configs import MODELS
    for name, _dc, opts in MODELS[model_key].get("val_scenarios", []):
        if name == case:
            return {k: opts[k] for k in ("discrete_prob", "categorical_regressors") if k in opts}
    return {}


def load_bf_validation_data(
    data_path,
    design_config: dict,
    intrinsic_params: list,
    model_id: int,
    batch_size: int,
    max_dm_cols: int | None = None,
) -> dict:
    """Load BF-generated validation data into full (batch, n_rows * n_local) param space.

    BF stores params in two formats depending on the case:
      - Intercept-only cases: (batch, n_local) — only intercept row params
      - Cases with regression rows: (batch, n_rows * n_local) — full grid already

    max_dm_cols: if provided, zero-pads design matrices to this width (needed for
    model class encoder which expects fixed-width input across all cases).
    """
    dataset = np.load(data_path, allow_pickle=True)
    rts = dataset["rts"]
    choices = dataset["choices"]
    design_matrices = dataset["design_matrices"]
    flat_params = dataset["true_set"]
    flat_masks = dataset["param_masks"]

    if max_dm_cols is not None and design_matrices.shape[-1] < max_dm_cols:
        pad_width = max_dm_cols - design_matrices.shape[-1]
        design_matrices = np.pad(design_matrices, ((0, 0), (0, 0), (0, pad_width)))

    n_local = len(intrinsic_params)
    n_rows = len(design_config)
    full_size = n_rows * n_local

    if flat_params.shape[1] == full_size:
        # BF stored as full (n_rows * n_local) grid — use directly
        pm_full = flat_params.copy()
        pmask_full = flat_masks.copy()
    else:
        # BF stored only intercept-row params — pad remaining rows with zeros
        pm_full = np.zeros((batch_size, full_size), dtype=flat_params.dtype)
        pmask_full = np.zeros((batch_size, full_size), dtype=flat_masks.dtype)
        pm_full[:, :flat_params.shape[1]] = flat_params
        pmask_full[:, :flat_masks.shape[1]] = flat_masks

    out = {
        "sim_data": {"rts": rts, "choices": choices},
        "design_matrices": design_matrices,
        "param_matrices": pm_full,
        "param_masks": pmask_full,
        "model_ids": np.full(batch_size, model_id, dtype=np.int64),
    }
    # Stimulus identity for stimulus-coded families (RDM): the adapter feeds
    # favored_idx and num_alternatives to the encoder as per-trial channels. Without
    # them it substitutes zeros / K=2 and the encoder reads stimulus-coded choices
    # blind. Files written before 2026-09-27 lack both keys and load as before.
    if "favored_idx" in dataset.files:
        out["per_trial_context"] = {"favored_idx": np.asarray(dataset["favored_idx"])}
    if "num_alternatives" in dataset.files:
        out["num_alternatives"] = np.asarray(dataset["num_alternatives"])
    return out


def bench_to_training_layout(bench: dict, canonical: dict, design_config: dict,
                             bench_max_num_categories: int = 2) -> tuple[dict, list[int]]:
    """Re-lay a loaded benchmark case (load_bf_validation_data) into the training layout.

    The shared benchmark files store the COMPACT grid of their case: design columns
    [1, u_1, u_2, u_1:u_2] at the scenario's category count (2 for bench5). The
    model-class network was trained on the canonical r/c grid, where u_2 sits at
    column 1 + (K-1), interactions follow all main blocks, and every regressor column
    is populated (unused ones as standardized distractors). Zero-padding the compact
    grid on the right put u_2 in u_1's category-2 slot and labelled its parameter token
    that way, which the network never saw in training (measured 2026-09-29: DDM
    regressed tau-intercept 90% coverage 0.21 on the files vs 0.82 self-simulated).

    ``canonical`` is a batch_sample output for the same scenario, drawn exactly as the
    self-simulated branch draws it (pad_to_max_num_categories=training max, flattened
    params, local space). Its layout, position codes and distractor columns are kept;
    the benchmark's regressor columns, parameter rows, RTs/choices and per-trial
    context replace its own. Interaction columns are recomputed from the updated main
    columns. Returns (samples, keep_rows): keep_rows are the canonical row indices of
    the compact rows, in compact order, to map predictions back to the compact grid.
    """
    from cogformer.simulator.design_utils import build_position_codes

    compact_codes = build_position_codes(design_config, max_num_categories=bench_max_num_categories)
    canon_codes = np.asarray(canonical["position_codes"][0], dtype=np.int64)
    if not (np.asarray(canonical["position_codes"]) == canon_codes).all():
        raise ValueError("canonical sample has per-element position codes; expected one fixed layout")

    def canon_row(code):
        hits = np.where((canon_codes == code).all(axis=1))[0]
        if len(hits) != 1:
            raise ValueError(f"position code {tuple(code)} matches {len(hits)} canonical rows")
        return int(hits[0])

    # Row 0 is the intercept on both grids; its all-zero code is shared with padding rows.
    keep_rows = [0] + [canon_row(c) for c in compact_codes[1:]]

    B, T = bench["design_matrices"].shape[:2]
    canon_dm = np.asarray(canonical["design_matrices"], dtype=np.float32)
    if canon_dm.shape[:2] != (B, T):
        raise ValueError(f"canonical design {canon_dm.shape[:2]} != benchmark {(B, T)}")
    dm = canon_dm.copy()
    bench_dm = np.asarray(bench["design_matrices"], dtype=np.float32)

    main_rows = [(i, r) for i, (r, c) in enumerate(zip(keep_rows, compact_codes)) if c[0] > 0 and c[1] == 0]
    for i, r in main_rows:
        dm[:, :, r] = bench_dm[:, :, i]
    for r, (a, b, sa, sb) in enumerate(canon_codes):
        if a > 0 and b > 0 and np.abs(canon_dm[:, :, r]).any():
            dm[:, :, r] = dm[:, :, canon_row((a, 0, sa, 0))] * dm[:, :, canon_row((b, 0, sb, 0))]
    # Cross-check: an interaction the benchmark itself stores must equal the recomputed one.
    for i, (r, c) in enumerate(zip(keep_rows, compact_codes)):
        if c[0] > 0 and c[1] > 0 and np.abs(bench_dm[:, :, i]).any():
            if not np.allclose(dm[:, :, r], bench_dm[:, :, i], atol=1e-4):
                raise ValueError(f"recomputed interaction column {tuple(c)} differs from the benchmark's")

    n_canon = canon_codes.shape[0]
    n_compact = len(keep_rows)
    n_local = bench["param_matrices"].shape[1] // n_compact
    pm = np.zeros((B, n_canon, n_local), dtype=np.float32)
    mask = np.zeros((B, n_canon, n_local), dtype=np.float32)
    pm[:, keep_rows] = np.asarray(bench["param_matrices"]).reshape(B, n_compact, n_local)
    mask[:, keep_rows] = np.asarray(bench["param_masks"]).reshape(B, n_compact, n_local)
    if canonical["param_matrices"].shape[1] != n_canon * n_local:
        raise ValueError("canonical and benchmark parameter spaces differ")

    out = {k: v for k, v in canonical.items() if k not in ("sim_data", "per_trial_context", "num_alternatives")}
    out["design_matrices"] = dm
    out["param_matrices"] = pm.reshape(B, n_canon * n_local)
    out["param_masks"] = mask.reshape(B, n_canon * n_local)
    for k in ("sim_data", "per_trial_context", "num_alternatives", "model_ids"):
        if k in bench:
            out[k] = bench[k]
    return out, keep_rows
