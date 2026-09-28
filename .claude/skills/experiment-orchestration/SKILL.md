---
name: experiment-orchestration
description: Run, resume, monitor, or split across machines the CogFormer-2 finals pipeline (run_experiments.sh, experiments 1/1a/2/3/4). Use when launching or resuming any stage, preparing a second machine (e.g. starrail), moving artifacts between hosts, checking what is still missing, or deciding whether an output can be trusted.
---

# CogFormer-2 experiment orchestration

`run_experiments.sh` (repo root) is the ONLY launcher. It runs 31 stages that produce
everything under `outputs/finals/experiment-{1,1a,2,3,4}/`. Everything else that used to
launch experiments was deleted on 2026-09-28; do not recreate ad-hoc launchers.

## 1. Environment (every machine, every time)

```bash
git pull                                  # same commit on every machine
uv sync --locked                          # exact validated stack; add --extra tracking for wandb
bash fetch_fonts.sh                       # Inter fonts for the figure style (once per clone)
.venv/bin/python -c "import bayesflow, numba, numpy, keras, jax, torch; \
  print(bayesflow.__version__, numba.__version__, numpy.__version__, keras.__version__, jax.__version__, torch.__version__)"
# expect: 2.0.14 0.67.0 2.5.3 3.15.1 0.11.1 2.11.0
```

- **Never mix stacks within one study.** numpy/numba changes alter the simulated random
  streams, so datasets and posteriors from different stacks are not comparable. If a
  machine cannot reproduce the versions above, stop and fix it before running anything.
- The orchestrator exports `MPLBACKEND=Agg` itself (a dead Xwayland DISPLAY otherwise hangs
  every plotting stage at 0% CPU). Any script run OUTSIDE the orchestrator needs it too.
- Real data lives in `data/data_application_{ddm,rdm,cdm,lba}.csv` (tracked in git).

## 2. The pipeline

```bash
bash run_experiments.sh --list            # stage table with done/todo markers
DRY_RUN=1 bash run_experiments.sh all     # print every command, no side effects
bash run_experiments.sh exp3              # a group: exp1 exp1a exp2 exp3 exp4 finalize all
bash run_experiments.sh e1_validate e3_compare   # or stage names / numbers
FORCE=1 bash run_experiments.sh e2_ppc    # ignore completion markers
```

Logs: `logs/finals/<RUN_ID>/<stage>.log`. Settings live in `experiments/shared_config.sh`
(read by the Python finalize scripts through `experiments/shared_config.py`, so the two can
never disagree). Change the recipe THERE, never inline.

| Group | Stages (in order) | What it produces |
|---|---|---|
| exp1 | `e1_data` → `e1_train` → `e1_validate`, `e1_ensemble`, `e1_benchmark`, `e1_design_eig` | shared benchmark data; model-class checkpoint (r4c4, b128 × 15000 ep); per-case, ensemble, benchmark, information-gain figures |
| exp1a | `e1a_bf` → `e1a_c2st` | 15 BayesFlow baselines (DDM/RDM/CDM × bench5); gap C2ST vs e1_validate |
| exp2 | `e2_apply`, `e2_refit_svfix` → `e2_ppc`, `e2_validate` | real-data fits, PPCs (+ combined figure), validation on the real subjects' designs |
| exp3 | `e3_bf`, `e3_zeroshot`, `e3_finetune` → `e3_ft_validate`, `e3_original` (links to e1_validate) → `e3_paired_c2st`, `e3_compare`, `e3_gap_c2st`, `e3_ppc_apply` → `e3_ppc`, `e3_realdata_validate` | LBA zero-shot, 1000-epoch finetune, forgetting C2ST, Δr / calibration plane, LBA amortization gap, stimulus-coded real-data fit + PPC + validation |
| exp4 | `e4_probe` → `e4_compare` → `e4_representation` | 5-param (vsvfixed) aDDM probe, both arms; shared-validation C2ST; RSA/CKA + 5-family MDS |
| finalize | `e{1,1a,2,3,4}_finalize` | copy allowlisted outputs into `outputs/finals/`, write MANIFEST + README |

**[OUTPUTS.md](OUTPUTS.md) maps every stage to its outputs**: for each checkpoint, generated
dataset, figure and table it names the responsible script, the working location it is written
to, and its destination under `outputs/finals/` (with the renaming rules and a dependency
graph). Use it to find where an artifact comes from, what to copy between hosts, or which
stage to re-run for a missing file. `experiments/finalize_experiment_N.py` stays the
authoritative source→destination map if the two ever disagree.

Rough costs on this project's hardware (RTX 5000 Ada class):
- `e1_train`: ~14 s/epoch at b128, GPU-bound → ~58 h. Sim workers are nearly idle.
- each BF case: ~8 h at 1000 ep × b128, SIMULATION-bound (CPU); 20 cases in total (15 in e1a, 5 LBA in e3).
- an 18-case validation at 200 datasets × 200 draws (`samples_chunk_size 1`): ~4–5 h.
- `e3_finetune` (1000 ep, b128): several hours.

## 3. Invariants — do not break these

1. **The shared benchmark files ARE the test set.** `e1_data` writes bench5 × {ddm, rdm, cdm,
   lba} once to `outputs/data/benchmark/seed0/` (+ `.done`). Every arm scoring a bench5 case
   LOADS them (`--data_dir`): e1_validate (= Exp 3 original), e3_zeroshot, e3_ft_validate,
   and all BF validations. The seed pins parameters but NOT RTs/choices (numba RNG) or
   regressor designs, so re-running `e1_data` silently unpairs everything scored on the old
   files. Never regenerate mid-study; on a second machine, COPY the directory.
2. **Completion = sentinel, not file existence.** Trainers rewrite their final checkpoint
   path mid-run, so `e1_train` and `e3_finetune` are done only when `<ckpt>.done` exists
   (`e1_train` also resumes with `--resume`). Downstream `need` checks the sentinel.
3. **C2ST only on bench5.** `mixed_predictor` and real-data designs are simulated per arm
   (parameter-paired only); `--strict` compares only `true_set` and cannot detect this.
4. **Compact vs padded grids.** Predictions scored from the shared files are in the compact
   bench grid (2 categories); self-simulated ones are padded to training (4). C2ST stages pass
   `BENCH_MAX_CAT=2`; validators carry per-case `plot_cats` and save it in the pred npz.
   Labelling a compact grid at 4 names u_2 / u_1:u_2 as u_1 categories.
5. **Same architecture flags everywhere.** The r/c geometry (`MC_MAX_REG/MC_MAX_CAT`) and
   `LBA_FIXED_A=0.5` are passed explicitly; checkpoint filenames do not encode r/c. Loading
   is strict, so a mismatched parameter set fails loudly — do not work around it.
6. **Finalize copies, never deletes.** Stale files in `outputs/finals/` survive a rebuild.
   For a clean final run, clear `outputs/finals/experiment-*` first (it is in git history).

## 4. Launching and monitoring

Long stages run DETACHED so an SSH drop cannot kill them:

```bash
setsid nohup bash run_experiments.sh exp1 > logs/exp1_$(date +%Y%m%d_%H%M).out 2>&1 < /dev/null & disown
```

- Check the log's modification time against the wall clock, not just its contents: a hung
  process and a slow one print the same last line. Treat > ~40 min without a write as suspect
  (validation can go quiet ~20 min per case while sampling).
- Poll sparsely (about every 2 h) and report completion/failure, not routine progress.
- GPU selection: `CUDA_VISIBLE_DEVICES=<id> bash run_experiments.sh ...`. BF stages add
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` so jax does not grab 75% of a shared card.
- A failed stage aborts the run; fix the cause and re-run the same selection — completed
  stages are skipped by their markers, and `e1a_bf`/`e3_bf` skip finished cases individually
  (their marker is each case's validated `outputs/data/model_family/<fam>_<case>_data.npz`).

## 5. Splitting work across two machines

The orchestrator has no host roles; split by running stage selections per host and moving
artifacts with rsync. Both hosts: same commit, `uv sync --locked`, and paths are
repo-relative, so rsync `outputs/...` subtrees 1:1 between clones.

**Critical path:** `e1_train` (~58 h) → everything that needs the model-class checkpoint.
BF baselines (`e1a_bf`, `e3_bf`) need only `e1_data` and are CPU-bound, so they run in
parallel with training — ideally on the host with more CPU cores.

Suggested split (host A = more GPUs/cores, e.g. starrail 2 GPU / 48 cores; host B = this
machine, 1 GPU / 24 cores):

1. **One host** runs `e1_data`. Copy `outputs/data/benchmark/seed0/` (incl. `.done`) to the
   other host BEFORE either host runs anything that reads it.
2. Host A GPU 0: `e1_train`. Host A GPU 1 and host B: BF cases.
   - To split BF cases, run individual cases directly on one host — copy the exact
     train+validate commands for those cases from `DRY_RUN=1 bash run_experiments.sh e1a_bf`
     (run them with `MPLBACKEND=Agg` and the repo on `PYTHONPATH`) — and rsync back:
     `outputs/checkpoints/model_family/bf/<fam>/`, `outputs/data/model_family/<fam>_<case>_data.npz`,
     `outputs/figures/model_family/bf/<fam>/<case>/`,
     `outputs/evaluations/model_family/bf/<fam>_families_bf_<case>_evaluations.csv`.
     The stage on the other host then skips those cases.
   - Two BF jobs on one host compete for the same cores (each defaults to cpu_count − 4
     simulation workers); set `--sim_workers` so their sum stays below the core count.
3. After `e1_train`: copy `outputs/checkpoints/model_class/cf/<MC_TAG>/` INCLUDING the `.pt.done`
   sentinel to any host that runs downstream stages. Likewise the LBA finetune directory
   `outputs/checkpoints/model_class/cf/lba_finetune/` + `.done` after `e3_finetune`.
4. Run downstream groups wherever a GPU is free; every stage's inputs are files, so only the
   directories it `need`s must be present (check with `DRY_RUN=1`, which prints each missing
   input as "would need: ...").
5. Gather all source trees on ONE host and run `bash run_experiments.sh finalize` there.

## 6. Before the final launch

- `git status` clean and committed on both hosts; `uv sync --locked` done; versions verified.
- Any pending architecture decision applied as CLASS defaults in
  `cogformer/networks/transformers/cogformer.py` (not per-script flags).
- `outputs/finals/experiment-*` cleared if a clean rebuild is wanted.
- `DRY_RUN=1 bash run_experiments.sh all` completes; `--list` shows the expected todo stages.
- GPUs free (`nvidia-smi`), enough disk for ~15 GB of new outputs.

## 7. Why things are the way they are (short)

- BF trainer reseeds numpy per worker process: keras forks its PyDataset workers every epoch
  with a cloned RNG, which once made every baseline train on ~320 datasets.
- BF RDM/LBA receive `favored_idx` as an input channel (stimulus coding); BF case designs are
  read from `val_scenarios` (`utils.benchmark_mix_opts`) so both arms draw the same designs.
- Real-data fits refine the parameter mask to realized design columns, exactly as training
  does (`data_utils.build_subject_batch`).
- The PPC cache key includes a hash of the posteriors file, so a refit cannot reuse stale draws.
- Diagnostic figures are square and legend-safe by construction (`set_box_aspect(1)`,
  `plot_utils.place_legend_below`); check figure properties by reading that code, not by eye.
