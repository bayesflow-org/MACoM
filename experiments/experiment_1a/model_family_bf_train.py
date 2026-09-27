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

    # The inference target is the FLATTENED PARAMETER GRID, and most of it is
    # structurally zero. Measured 2026-09-10 over 4 independent batches per case:
    #
    #   case              grid  active  constant(std==0)
    #   intercept_only       7       7      0
    #   fixed                7       5      2
    #   regressed           28      13     15   <- more than half
    #   fixed_regressed     28      11     17   <- more than half
    #   interaction         28      19      9
    #
    # Slicing to the active cells avoids asking the flow to reproduce dimensions
    # that are always exactly 0.0, which is what experiments/encoder_rep/
    # addm_bf_baseline.py does. It is a tidiness win, NOT a fix:
    #
    #   MEASURED (ddm/regressed, 300 epochs, slicing as the only variable):
    #       full grid (28 dims, 15 dead)  calibration error 0.417
    #       sliced   (13 active dims)     calibration error 0.405   (-3%)
    #
    #   And the predecessor project (../CogFormer, experiments/model_family/
    #   family_bf_train.py) fed the FULL GRID including dead dims and still
    #   reached calibration error 0.019 on this case. So dead dimensions were
    #   never what broke these baselines. Neither was the unstandardized summary
    #   input -- see the standardize block in main(), where the A/B measured
    #   standardizing it as slightly WORSE. Do not read this block as a diagnosis.
    #
    # Kept on by default because -3% is in the right direction and costs nothing
    # at training time; --no-slice_active_target restores the original target and
    # matches the predecessor exactly. The one real cost is downstream: an
    # active-only checkpoint emits a narrower params vector, so
    # model_family_bf_validate has to detect the width and scatter back to the
    # full grid before saving the npz.
    #
    # Safe because the active pattern is CONSTANT for every bench5 case -- all of
    # them set fixed_config=True or carry an explicit design_config, and 4-batch
    # probes over both models found exactly 1 distinct pattern each. _active is
    # cached from the first batch and asserted on every later one, so a case that
    # ever violated this would fail loudly instead of silently misaligning columns.
    _active = None
    slice_active_target = True   # set False to reproduce the pre-2026-09-10 target

    # PER-PROCESS RESEED. keras' PyDataset with use_multiprocessing FORKS a fresh
    # worker pool every epoch, and each fork inherits the parent's numpy RNG state --
    # which never advances, because the parent does not simulate. Measured 2026-09-27
    # (4 workers x 6 batches x 3 epochs): every worker emitted the SAME batch
    # sequence, and every epoch replayed the previous one exactly. At 20 workers x 100
    # batches x 64 the baseline therefore trained on ~5 distinct batches (~320
    # datasets) for its whole run. That is the overconfidence -- calibration error
    # 0.4, contraction 0.997, getting WORSE with epochs -- that the comments below
    # attribute to training length: it was overfitting a tiny fixed set. The one
    # checkpoint trained before the worker pool existed (ddm/intercept_only,
    # 2026-09-02) validates at calibration error ~0.03.
    #
    # CogFormer's own loader (experiments.utils.make_sim_loader) is NOT affected:
    # spawn + persistent workers, verified 16/16 distinct batches.
    _rng_pid = None

    def _reseed_if_forked(self):
        pid = os.getpid()
        if self._rng_pid != pid:
            self._rng_pid = pid
            np.random.seed(int.from_bytes(os.urandom(4), "little"))

    def sample(self, batch_size, num_obs=500, **kwargs):
        self._reseed_if_forked()
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

        active = param_masks[0].astype(bool)
        if self._active is None:
            self._active = active
        elif not np.array_equal(self._active, active):
            raise RuntimeError(
                f"active-parameter pattern changed between batches for "
                f"{self.case}: cached {self._active.sum()} active, got "
                f"{active.sum()}. Slicing the inference target assumes a constant "
                f"pattern; this case needs the full grid instead."
            )
        if not np.all(param_masks == param_masks[0]):
            raise RuntimeError(f"mask varies within batch for {self.case}")

        return {
            "design_matrices": samples["design_matrices"],
            "rts": samples["sim_data"]["rts"],
            "choices": samples["sim_data"]["choices"],
            "params": param_matrices[:, active] if self.slice_active_target else param_matrices,
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
         batch_size: int = 64, sim_workers: int | None = None,
         standardize_summaries: bool = False, tag: str = "",
         time_power_law_alpha: float | None = None,
         slice_active_target: bool = True,
         rt_transform: str = "none"):
    model_cfg = MODELS[model_key]
    simulator = ModelFamilyBF(model_key=model_key, case=case)
    simulator.slice_active_target = slice_active_target

    # The summary network sees rts, choices and design columns concatenated RAW --
    # BayesFlow standardizes only inference_variables. Contamination replaces ~pi of
    # trials with RT ~ Uniform(0, 20) against normal DDM RTs of 0.3-2s, which the
    # predecessor project never had. Measured over 256x500 trials:
    #
    #                 rts sd    p99     max     frac>10s
    #   predecessor    0.736    3.94    10.4     0.0000
    #   ours           1.666   10.15    20.0     0.0102
    #
    # log(rts + 2) compresses that tail (a 20s outlier becomes 3.09, channel sd
    # 7.79 -> 1.03) and stays finite on the rare rts = -1 "no response" sentinel,
    # which plain log1p would send to -inf. The shift is +2 so that sentinel maps
    # to exactly 0.
    #
    # The adapter is serialized INTO the checkpoint, so model_family_bf_validate
    # picks the same transform up automatically -- there is no second place to keep
    # in sync.
    adapter = bf.Adapter().drop(["masks"]).convert_dtype("float64", "float32")
    if rt_transform == "log":
        adapter = adapter.shift("rts", by=2.0).log("rts")
    adapter = (
        adapter
        .concatenate(["design_matrices", "rts", "choices"], into="summary_variables")
        .rename("params", "inference_variables")
    )

    # mlp_depths and mlp_widths are NOT SetTransformer parameters in BayesFlow
    # 2.0.13 -- the constructor swallows unknown kwargs without raising (verified:
    # even `total_nonsense_param=123` is accepted silently), so those two were
    # discarded on every run since this file was written. The network has always
    # used library-default MLP widths and dropout=0.05. Passing only real
    # parameters makes the configuration match what actually gets built.
    summary_net = bf.networks.SetTransformer(
        summary_dim=32, seed_dim=64,
        num_heads=(4, 4, 4, 4),
        embed_dims=(128, 128, 128, 128),
        num_seeds=4,
    )
    # time_power_law_alpha sets the TRAINING distribution of the flow-matching
    # interpolation time: t = u ** (1 + alpha) for u ~ U(0,1), so E[t] = 1/(2+alpha).
    # With theta_t = t*theta + (1-t)*z, t=1 is the DATA end -- the region where the
    # velocity field learns to contract onto the posterior.
    #
    # BayesFlow 2.0.13 defaults this to 0.5, so a bare FlowMatching() trains with
    # E[t]=0.4, biased toward the noise end and starving the contraction region.
    # CogFormer draws t ~ U(0,1) (decoder.py: `t = torch.rand(...)`), i.e. alpha=0.
    # The two arms of the amortization-gap comparison were therefore optimising
    # DIFFERENT objectives. experiments/encoder_rep/addm_bf_baseline.py already
    # names alpha=0.5 as "the leading suspect" for a calibration failure in the
    # other BayesFlow baseline in this repo. Pass 0.0 to match CogFormer.
    fm_kwargs = {}
    if time_power_law_alpha is not None:
        fm_kwargs["time_power_law_alpha"] = time_power_law_alpha
    inference_net = bf.networks.FlowMatching(**fm_kwargs)

    ckpt_prefix = model_cfg["ckpt_prefix"]
    checkpoint_path = f"./outputs/checkpoints/model_family/bf/{model_key}/{ckpt_prefix}_{case}"
    if tag:
        checkpoint_path += f"__{tag}"

    # BayesFlow standardizes ONLY inference_variables by default, and so did the
    # predecessor project (../CogFormer, experiments/model_family/family_bf_train.py):
    # it never touched summary_variables. Routing them through the adaptive
    # Standardization layer as well is therefore a DEVIATION from the reference rather
    # than a step toward it -- and the A/B says it does not pay for itself:
    #
    #   MEASURED (ddm/intercept_only, 100 epochs, standardization the only variable;
    #   outputs/logs/bf_ab_round2_driver.log)
    #       raw (inference_variables only)        calibration error 0.119
    #       std (+ summary_variables)             calibration error 0.166   WORSE
    #       raw + rt_transform=log                calibration error 0.127
    #       std + rt_transform=log                calibration error 0.137   WORSE
    #
    # So it is OFF by default, matching the predecessor. Kept switchable because the
    # summary channel genuinely is heterogeneous -- RTs in seconds (~0.2-2), binary
    # choices and design-matrix columns concatenated into one per-trial vector -- so a
    # future change to the contamination tail could make it earn its place. Re-measure
    # before turning it on.
    #
    # It was never the cause of the overconfident baselines either way. The dominant
    # term is TRAINING LENGTH: calibration error degrades monotonically with epochs
    # (ddm/intercept_only 0.145 at 100 ep -> 0.388 at 2000; rdm/intercept_only 0.295 at
    # 300 -> 0.432 at 2000) while the posterior MEANS stay accurate and the spread
    # collapses to ~10% of prior sd. Sweep epochs before changing anything here.
    standardize = ["inference_variables"]
    if standardize_summaries:
        standardize.append("summary_variables")

    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        summary_network=summary_net,
        inference_network=inference_net,
        checkpoint_filepath=checkpoint_path,
        standardize=standardize,
    )
    print(f"[cfg] rt_transform={rt_transform} "
          f"slice_active_target={slice_active_target} standardize={standardize} "
          f"time_power_law_alpha={getattr(inference_net, 'time_power_law_alpha', '?')} "
          f"checkpoint={checkpoint_path}")

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
    parser.add_argument("--standardize_summaries", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Route summary_variables through BayesFlow's adaptive "
                             "standardizer as well as inference_variables. Default OFF, "
                             "matching the predecessor project, which standardized only "
                             "inference_variables. The A/B measured turning it on as a "
                             "small regression (calibration error 0.166 vs 0.119), so "
                             "--standardize_summaries is opt-in.")
    parser.add_argument("--rt_transform", type=str, default="none", choices=["none", "log"],
                        help="'log' applies log(rts + 2) before the summary network, "
                             "compressing the Uniform(0,20) contamination tail. Baked "
                             "into the saved adapter, so validation inherits it.")
    parser.add_argument("--slice_active_target", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Train the flow on ONLY the active parameter cells. The "
                             "full grid is up to 60%% structurally zero. Measured worth "
                             "-3%% calibration error, i.e. marginal -- the predecessor "
                             "project fed the full grid and still reached 0.019. "
                             "--no-slice_active_target restores the original target.")
    parser.add_argument("--time_power_law_alpha", type=float, default=None,
                        help="Flow-matching training time exponent: t = u**(1+alpha). "
                             "BayesFlow defaults to 0.5 (biased toward the noise end); "
                             "0.0 is uniform and matches CogFormer's decoder. Omit to "
                             "keep the library default.")
    parser.add_argument("--tag", type=str, default="",
                        help="Suffix for the checkpoint directory, so an experimental "
                             "run cannot overwrite a production baseline.")
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
                standardize_summaries=args.standardize_summaries,
                tag=args.tag,
                time_power_law_alpha=args.time_power_law_alpha,
                slice_active_target=args.slice_active_target,
                rt_transform=args.rt_transform,
            )
        if args.use_wandb:
            wandb.finish()
