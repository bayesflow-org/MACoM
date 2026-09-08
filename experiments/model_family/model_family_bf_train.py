import os
os.environ["KERAS_BACKEND"] = "jax"
# ONE numba thread. Measured on this 24-core box, batch 256, num_obs 500:
#     ModelFamily threads=4  -> 285 ms/batch
#     ModelFamily threads=12 -> 257 ms/batch
#     ModelFamily threads=23 -> 1788 ms/batch   (7x WORSE)
# Multi-threaded numba pays a fork-join sync on every simulate() call -- once per
# batch element -- so it degrades superlinearly. experiments/utils.recommend_sim_workers
# records the same effect measured the other way: 6 workers x 4 threads = 3.3 batches/s
# against 20 workers x 1 thread = 7.8 batches/s. Parallelism belongs at the DATASET
# level (worker processes), not inside numba.
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import argparse
import numpy as np
try:
    import wandb
except Exception:
    wandb = None
import bayesflow as bf

from pathlib import Path

from cogformer.simulator.model_family import ModelFamily

from ..configs import MODELS as _BASE_MODELS
from ..utils import recommend_sim_workers
from ..timing import RunTimer


_BF_CKPT_PREFIXES = {"ddm": "ddm_families_bf", "cdm": "cdm_families_bf",
                     "rdm": "rdm_families_bf", "lba": "lba_families_bf"}

# Only the families that actually define BF case configs. The guard stays because the
# unguarded `_cfg["bf_case_configs"]` here raised KeyError on import for every
# invocation, whatever --model was passed. LBA originally had no entry -- it was reached
# zero-shot or by finetuning from the model-class checkpoint rather than by a per-family
# BayesFlow baseline -- but the amortization-gap matrix needs "LBA: BF vs Finetuned",
# so LBA_BF_CASE_CONFIGS now exists (derived from RDM's; see experiments/configs.py).
MODELS = {}
for _key, _base in _BASE_MODELS.items():
    if "bf_case_configs" not in _base or _key not in _BF_CKPT_PREFIXES:
        continue
    _cfg = dict(_base)
    _cfg["case_configs"] = _cfg["bf_case_configs"]
    _cfg["ckpt_prefix"] = _BF_CKPT_PREFIXES[_key]
    MODELS[_key] = _cfg


class ModelFamilyBF(bf.simulators.Simulator):
    """Generic BayesFlow simulator wrapping ModelFamily for any benchmark model."""

    def __init__(self, model_key: str, case: str):
        model_cfg = MODELS[model_key]
        case_configs = model_cfg["case_configs"]
        if case not in case_configs:
            raise ValueError(f"Unknown case: {case}. Must be one of {list(case_configs.keys())}")

        self.case = case
        self.config = case_configs[case]
        self.link_fn = model_cfg["link_fn"]

        # Online training simulates every batch on the fly, so the numba simulator is
        # a real throughput term alongside the GPU step. ModelFamily defaults to
        # max(1, cpu_count()//6) workers (4 of 24 cores here); sim_workers lets the
        # caller use the box.
        # num_workers=1: this family runs inside ONE dataset worker process, which
        # supplies the parallelism. Threading here as well only re-creates the
        # fork-join penalty measured above.
        self.model_family = ModelFamily(
            # model_kwargs carries per-family simulator settings that differ from the
            # class defaults -- today only LBA(fixed_A=0.5), which every other LBA arm
            # pins via --lba_fixed_A. Constructing bare here trained the baseline on a
            # different generative model than the arm it is compared against.
            model=model_cfg["model_cls"](**model_cfg.get("model_kwargs", {})),
            prior=model_cfg["prior_fn"](),
            num_workers=1,
            regressed_params=self.config["regressed_params"],
            mask_randomizer_kwargs=dict(
                free_intrinsics=self.config["free_intrinsics"],
                fixed_intrinsics=self.config["fixed_intrinsics"],
                fixed_values=self.config["fixed_values"],
            ),
        )

    def sample(self, batch_size, num_obs=500, **kwargs):
        if isinstance(batch_size, tuple):
            batch_size = batch_size[0]

        sample_kwargs = {
            "min_num_regressors": self.config["min_num_regressors"],
            "max_num_regressors": self.config["max_num_regressors"],
            "max_num_categories": self.config["max_num_categories"],
        }
        if self.config["fixed_config"]:
            sample_kwargs["fixed_config"] = True
        if self.config["design_config"] is not None:
            sample_kwargs["design_config"] = self.config["design_config"]

        samples = self.model_family.batch_sample(
            batch_size=batch_size,
            num_obs=num_obs,
            flatten_param_outputs=self.config["flatten_param_outputs"],
            link_fun=self.link_fn(),
            **sample_kwargs,
            **kwargs,
        )

        param_matrices = samples["param_matrices"]
        param_masks = samples["param_masks"]
        if self.config["squeeze_outputs"]:
            param_matrices = param_matrices.squeeze(axis=1)
            param_masks = param_masks.squeeze(axis=1)

        return {
            "design_matrices": samples["design_matrices"],
            "rts": samples["sim_data"]["rts"],
            "choices": samples["sim_data"]["choices"],
            "params": param_matrices,
            "masks": param_masks,
        }


def test(model_key: str, case: str):
    model_cfg = MODELS[model_key]
    config = model_cfg["case_configs"][case]
    simulator = ModelFamilyBF(model_key=model_key, case=case)
    samples = simulator.sample(4)

    print(f"Model: {model_key.upper()}, Case: {case}")
    for k, v in samples.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: {v.shape}")

    masks = samples["masks"]
    params = samples["params"]

    num_active = int(masks[0].astype(bool).sum())
    expected = config["expected_num_active"]
    if expected is not None:
        status = "OK" if num_active == expected else f"FAIL (expected {expected})"
        print(f"\n  [Check] Active params: {num_active}  →  {status}")
        assert num_active == expected
    else:
        print(f"\n  [Check] Active params: {num_active}  →  (variable, no assertion)")

    if config["fixed_config"] or config["design_config"] is not None:
        consistent = np.all(masks == masks[0])
        print(f"  [Check] Mask consistent across batch: {'OK' if consistent else 'FAIL'}")
        assert consistent

    zero_where_masked = np.allclose(params * (1 - masks), 0.0)
    print(f"  [Check] Params zero where masked: {'OK' if zero_where_masked else 'FAIL'}")
    assert zero_where_masked

    adapter = (
        bf.Adapter()
        .drop(["masks"])
        .convert_dtype("float64", "float32")
        .concatenate(["design_matrices", "rts", "choices"], into="summary_variables")
        .rename("params", "inference_variables")
    )
    adapted = adapter(simulator.sample(4))
    print("\nAdapted samples:")
    for k, v in adapted.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: {v.shape}")


def main(model_key: str, case: str, epochs: int = 1000, steps_per_epoch: int = 100,
         batch_size: int = 64, sim_workers: int | None = None):
    model_cfg = MODELS[model_key]
    simulator = ModelFamilyBF(model_key=model_key, case=case)

    adapter = (
        bf.Adapter()
        .drop(["masks"])
        .convert_dtype("float64", "float32")
        .concatenate(["design_matrices", "rts", "choices"], into="summary_variables")
        .rename("params", "inference_variables")
    )

    summary_net = bf.networks.SetTransformer(
        summary_dim=32, seed_dim=64,
        num_heads=(4, 4, 4, 4), mlp_depths=(1, 1, 1, 1),
        embed_dims=(128, 128, 128, 128), mlp_widths=(128, 128, 128, 128),
        num_seeds=4,
    )
    inference_net = bf.networks.FlowMatching()

    ckpt_prefix = model_cfg["ckpt_prefix"]
    checkpoint_path = f"./outputs/checkpoints/model_family/bf/{model_key}/{ckpt_prefix}_{case}"

    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        summary_network=summary_net,
        inference_network=inference_net,
        checkpoint_filepath=checkpoint_path,
    )

    # Build the dataset explicitly instead of calling fit_online, so that keras'
    # PyDataset parallelism is reachable. OnlineDataset IS a keras PyDataset and
    # forwards **kwargs to it, but fit_online routes its own **kwargs to fit()
    # rather than to the dataset -- so workers/max_queue_size can only be set here.
    #
    # This buys two things at once, matching what the CogFormer trainer already does
    # via ModelFamilyDataset/make_sim_loader:
    #   * PROCESS parallelism over simulation, each worker single-threaded in numba
    #   * PREFETCH -- batch n+1 is simulated while the GPU trains on batch n, which
    #     the synchronous fit_online path cannot overlap at all.
    #
    # NB on the batch count: the parameter is num_batches, and fit_online's equivalent
    # is num_batches_per_epoch -- NOT steps_per_epoch, which is not a fit_online
    # argument and silently fell through **kwargs into keras' fit() while the dataset
    # kept its own default of 100. Same bug fixed in encoder_rep/addm_bf_baseline.py.
    workers = sim_workers if sim_workers is not None else recommend_sim_workers()
    dataset = bf.datasets.OnlineDataset(
        simulator=simulator,
        batch_size=batch_size,
        num_batches=steps_per_epoch,
        adapter=adapter,
        workers=workers,
        use_multiprocessing=workers > 1,
        max_queue_size=max(2 * workers, 4),
    )
    print(f"[sim] {workers} dataset worker processes x 1 numba thread, "
          f"prefetch queue {max(2 * workers, 4)}")
    # _fit, not fit: BasicWorkflow.fit is an abstract stub that always raises
    # NotImplementedError, and fit_online is just OnlineDataset(...) + _fit(...). Calling
    # _fit directly is the only way to supply a dataset we constructed ourselves, and
    # these are exactly the arguments fit_online passes.
    history = workflow._fit(
        dataset, epochs, strategy="online", keep_optimizer=False, validation_data=None
    )
    return history


if __name__ == "__main__":
    all_cases = list(MODELS["ddm"]["case_configs"].keys())

    parser = argparse.ArgumentParser(description="Model Family BayesFlow Training")
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()),
                        help="Benchmark model (ddm, cdm, rdm)")
    parser.add_argument("--case", type=str, default="intercept_only", choices=all_cases)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sim_workers", type=int, default=None,
                        help="Dataset worker PROCESSES for simulation (each single-threaded "
                             "in numba). Default recommend_sim_workers() = cpu_count()-4. "
                             "This is not a numba thread count -- raising numba threads "
                             "makes things dramatically worse (23 threads measured 7x "
                             "slower than 4), which is why parallelism lives here.")
    parser.add_argument("--dry_run_epochs", type=int, default=None,
                        help="If set, train for this many epochs only and record extrapolated time.")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        test(model_key=args.model, case=args.case)
    else:
        if args.use_wandb:
            wandb.init(
                project=f"cogformer-bf-{args.model}",
                tags=["CogFormer", args.model.upper(), "BF"],
                config={
                    "model": args.model,
                    "case": args.case,
                    "epochs": args.epochs,
                    "steps_per_epoch": args.steps_per_epoch,
                    "batch_size": args.batch_size,
                },
            )
        is_dry_run = args.dry_run_epochs is not None
        run_epochs = args.dry_run_epochs if is_dry_run else args.epochs
        with RunTimer(f"mf_bf_{args.model}_{args.case}", dry_run=is_dry_run,
                      total_epochs=args.epochs, dry_run_epochs=args.dry_run_epochs):
            main(
                model_key=args.model,
                case=args.case,
                epochs=run_epochs,
                steps_per_epoch=args.steps_per_epoch,
                batch_size=args.batch_size,
                sim_workers=args.sim_workers,
            )
        if args.use_wandb:
            wandb.finish()
