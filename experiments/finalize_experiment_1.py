"""Assemble outputs/finals/experiment-1 -- the single-provenance snapshot for
Experiment 1 (generalising to design space).

WHY THIS EXISTS
---------------
Experiment 1's results were previously spread across runs from three different
checkpoints: per-case figures from Jul 17 and again Aug 7, random ensembles from
Aug 9, benchmark ensembles from Jul 17 only. Nothing recorded which checkpoint
produced what, and the default --pred_dir of the benchmark script points at a
mixed-vintage directory, so it was possible to build a "current" figure from stale
predictions without any error. This script copies one coherent set into a frozen
tree and writes down exactly where every file came from.

The snapshot SUPERSEDES the source directories: predictions are copied rather than
linked so the tree is self-sufficient and the sources can be removed.

Idempotent, and missing sources are reported rather than fatal, so it can be run
before every stage has finished and re-run afterwards.

    python -m experiments.finalize_experiment_1 --dry_run
    python -m experiments.finalize_experiment_1
"""
import argparse
import hashlib
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

FINALS = Path("outputs/finals/experiment-1")
FIG = Path("outputs/figures/model_class/cf")
DATA = Path("outputs/data/model_class")
DESIGN_OPT = Path("outputs/data/design_opt/ddm")
C2ST = Path("outputs/figures/c2st")
BF_FIG = Path("outputs/figures/model_family/bf")
BF_EVAL = Path("outputs/evaluations/model_family/bf")
# The CogFormer arm of the C2ST is the PAIRED validate run, not the one behind
# per_case/. per_case/ comes from {BASE}_validate, which simulated its own
# datasets; the C2ST re-ran validation with --data_dir pointed at the BF npz so
# both arms score the SAME datasets. Comparing per_case/ against the BF
# diagnostics here would be comparing across different data.
CF_PAIRED = Path("outputs/figures/model_class/cf_paired")
BASE = "r4c4_b256_e10000"
CKPT = Path(f"outputs/checkpoints/model_class/cf/{BASE}/"
            f"cogformer_model_class_fm_l8_h8_p256_s32_d128_b256_e10000_t100.pt")

FAMILIES = ["ddm", "rdm", "cdm"]
CASES = ["intercept_only", "fixed", "regressed", "fixed_regressed",
         "interaction", "mixed_predictor"]
# One random ensemble only: 12 configs spanning all three families, 4 each. There are
# no per-family random ensembles -- the per-family breakdown is the job of
# benchmark_ensemble (bench5/bench6 x ddm/rdm/cdm), so running 12 more configs per
# family would be a different object at ~7h of GPU for no added coverage.
ENSEMBLE_DIR = f"{BASE}_ensemble"
# The amortization gap covers bench5 only. model_family_c2st drops "full", and the
# BF baselines were never trained on "mixed_predictor", so those two cases have no
# BF arm to compare against and are absent by construction, not by omission.
C2ST_CASES = ["intercept_only", "fixed", "regressed", "fixed_regressed", "interaction"]


# Plot stems this snapshot contains. An ALLOWLIST rather than "copy the directory":
# model_class_cf_validate.py emits 10 posterior pairplots per case by default
# (--skip_posteriors defaults to False), so a sweep-everything copier silently pulls
# in 180 extra files the moment someone re-runs without that flag. Naming the wanted
# types makes the tree's contents a decision, not a consequence of run flags.
KEEP = {"recovery", "coverage", "ecdf", "metrics"}
# Non-plot payload that is wanted despite not being a plot stem.
KEEP_EXACT = {"data.npz", "configs.csv"}


def kept(rest: str) -> bool:
    if rest in KEEP_EXACT:
        return True
    return rest.rsplit(".", 1)[0].replace("_global", "") in KEEP


def tidy(name: str) -> str:
    """Strip the redundant prefix and rename the global-parameter panel to _pi.

    ``_global`` meant "the global-parameter panel", which today is exactly pi
    (contamination). It is renamed to ``_pi`` because ``_global`` says nothing about
    what is plotted. If a second global parameter is ever added this convention has
    to change -- the panel would no longer be pi-specific -- so the mapping is
    recorded in MANIFEST.csv rather than being folklore.
    """
    return name.replace("_global", "_pi")


def plan() -> tuple[list[tuple[Path, Path]], list[Path]]:
    """(source, destination) pairs, plus files rejected by the allowlist."""
    jobs: list[tuple[Path, Path]] = []
    skipped: list[Path] = []

    # per-case: 6 cases x 3 families, from the Aug-7 validate run
    for fam in FAMILIES:
        for case in CASES:
            src_dir = FIG / f"{BASE}_validate" / fam / case
            for src in sorted(src_dir.glob("*")) if src_dir.is_dir() else []:
                rest = src.name.replace(f"{fam}_{case}_fm_", "")
                if not kept(rest):
                    skipped.append(src); continue
                jobs.append((src, FINALS / "per_case" / fam / case / tidy(rest)))

    # benchmark ensembles: bench5 / bench6 x 3 families
    bdir = FIG / f"{BASE}_ensemble_benchmark"
    for src in sorted(bdir.glob("*")) if bdir.is_dir() else []:
        parts = src.name.split("_ensemble_", 1)
        if len(parts) != 2:
            continue
        fam, bench = parts[0].split("_", 1)
        if not kept(parts[1]):
            skipped.append(src); continue
        jobs.append((src, FINALS / "benchmark_ensemble" / bench / fam / tidy(parts[1])))

    # random ensembles: combined + per family.
    # GUARD: only take a directory that contains data.npz. The pre-2026-09 runs wrote
    # figures but no npz, so the marker distinguishes the current coverage-sampled run
    # from the stale one it overwrites. Without this the snapshot would silently freeze
    # figures from the wrong checkpoint -- the exact failure this tree exists to end.
    edir = FIG / ENSEMBLE_DIR
    if (edir / "model_class_ensemble_data.npz").exists():
        for src in sorted(edir.glob("*")):
            rest = src.name.replace("model_class_ensemble_", "")
            if not kept(rest):
                skipped.append(src); continue
            if rest == "data.npz":
                dest = FINALS / "predictions" / "random_ensemble.npz"
            elif rest == "configs.csv":
                dest = FINALS / "random_ensemble" / "design_configs.csv"
            else:
                dest = FINALS / "random_ensemble" / tidy(rest)
            jobs.append((src, dest))

    # information gain vs trial count (static design optimisation). Same Aug-5
    # checkpoint as everything else here, so it belongs in this snapshot rather than
    # a separate one.
    for src, name in [(DESIGN_OPT / "design_eig.pdf", "information_gain.pdf"),
                      (DESIGN_OPT / "design_eig.csv", "information_gain.csv"),
                      (DESIGN_OPT / "design_contraction.npz", "contraction_per_cell.npz"),
                      (DESIGN_OPT / "design_nrmse.npz", "nrmse_per_cell.npz"),
                      (DESIGN_OPT / "design_draws.npz", "posterior_draws.npz")]:
        jobs.append((src, FINALS / "information_gain" / name))

    # amortization gap: C2ST against the per-case BayesFlow baselines.
    # The scenario tag is globbed, not hardcoded: model_family_c2st writes
    # <fam>_<case>_{orig_vs_bf,ft_vs_bf,ft_vs_orig}_c2st.csv depending on which arms
    # existed at run time, and DDM/CDM currently have orig_vs_bf only. LBA is
    # deliberately absent -- it is not one of this experiment's three families.
    for fam in FAMILIES:
        jobs.append((C2ST / fam / f"{fam}_c2st_summary.csv",
                     FINALS / "amortization_gap" / fam / "summary.csv"))
        for case in C2ST_CASES:
            cdir = C2ST / fam / case
            for src in sorted(cdir.glob(f"{fam}_{case}_*_c2st.csv")) if cdir.is_dir() else []:
                tag = src.name[len(f"{fam}_{case}_"):-len("_c2st.csv")]
                jobs.append((src, FINALS / "amortization_gap" / fam / case / f"{tag}.csv"))

    # Both arms of the comparison, per case, so the folder stands alone: bf_* is the
    # per-case BayesFlow baseline, cf_* the paired CogFormer run scored on the same
    # datasets. Same allowlist as everywhere else, and tidy() still maps the
    # global-parameter panel to _pi.
    for fam in FAMILIES:
        for case in C2ST_CASES:
            dest_dir = FINALS / "amortization_gap" / fam / case

            bdir = BF_FIG / fam / case
            for src in sorted(bdir.glob("*")) if bdir.is_dir() else []:
                rest = src.name.replace(f"{fam}_family_{case}_bf_", "")
                if not kept(rest):
                    skipped.append(src); continue
                jobs.append((src, dest_dir / f"bf_{tidy(rest)}"))

            # NRMSE / calibration error / posterior contraction for the BF arm --
            # the numbers that show how overconfident the baseline is.
            jobs.append((BF_EVAL / f"{fam}_families_bf_{case}_evaluations.csv",
                         dest_dir / "bf_evaluations.csv"))

            cdir = CF_PAIRED / fam / case
            for src in sorted(cdir.glob("*")) if cdir.is_dir() else []:
                rest = src.name.replace(f"{fam}_{case}_fm_", "")
                if not kept(rest):
                    skipped.append(src); continue
                jobs.append((src, dest_dir / f"cf_{tidy(rest)}"))

    # per-case predictions
    pdir = DATA / f"{BASE}_validate"
    for src in sorted(pdir.glob("*_cf_pred.npz")) if pdir.is_dir() else []:
        jobs.append((src, FINALS / "predictions" / "per_case" /
                     src.name.replace("_cf_pred", "")))
    return jobs, skipped


def sha256(path: Path, limit: int = 1 << 30) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        read = 0
        while chunk := fh.read(1 << 20):
            h.update(chunk); read += len(chunk)
            if read >= limit:
                return h.hexdigest() + " (first 1GiB)"
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    jobs, skipped = plan()
    present = [(s, d) for s, d in jobs if s.exists()]
    print(f"{len(present)} files to place\n")

    groups: dict[str, int] = {}
    for _, d in present:
        groups[d.relative_to(FINALS).parts[0]] = groups.get(d.relative_to(FINALS).parts[0], 0) + 1
    for k in sorted(groups):
        print(f"  {k:22} {groups[k]:4d} files")

    if not (FIG / ENSEMBLE_DIR / "model_class_ensemble_data.npz").exists():
        print(f"\n  [pending] random_ensemble: {FIG / ENSEMBLE_DIR} has no data.npz, "
              f"so it is either unfinished or a pre-2026-09 run. SKIPPED -- re-run "
              f"this script once the rebuild finishes.")
    if not (FIG / f'{BASE}_ensemble_benchmark').exists():
        print(f"\n  [pending] benchmark_ensemble: {FIG / f'{BASE}_ensemble_benchmark'} "
              f"not present yet -- re-run this script once that stage finishes.")

    gap_missing = [f for f in FAMILIES
                   if not (C2ST / f / f"{f}_c2st_summary.csv").exists()]
    if gap_missing:
        print(f"\n  [pending] amortization_gap: no C2ST summary for "
              f"{', '.join(gap_missing)} -- their BayesFlow baselines are still "
              f"training. Re-run this script when they finish.")

    if skipped:
        print(f"\n  excluded by allowlist: {len(skipped)}")
        for x in skipped[:5]:
            print(f"    {x.name}")

    if args.dry_run:
        print("\ndry run; nothing written")
        return

    rows = []
    for src, dest in present:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        st = src.stat()
        rows.append({
            "destination": str(dest.relative_to(FINALS)),
            "source": str(src),
            "bytes": st.st_size,
            "source_mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc)
                            .strftime("%Y-%m-%d %H:%M:%SZ"),
            "renamed_global_to_pi": "_pi" in dest.name and "_global" in src.name,
        })
    # Keep the folder present even when no family has finished yet; harmless once
    # real files land beside it.
    (FINALS / "amortization_gap").mkdir(parents=True, exist_ok=True)
    (FINALS / "amortization_gap" / ".gitkeep").touch()

    # Carry forward rows this run cannot regenerate. The snapshot deliberately
    # SUPERSEDES its sources -- the module docstring says they can be removed once
    # copied -- so a later run finds those sources gone and, regenerating the
    # manifest purely from sources, would drop files that are still sitting in the
    # tree. Measured 2026-09-09: information_gain and random_ensemble had lost their
    # sources, so a plain re-run would have shrunk the record from 260 rows to 240
    # while the files stayed on disk. Keep any prior row whose destination still
    # exists; drop the rest, which is a correction -- 19 `predictions/` rows named
    # files that are no longer in the tree at all.
    manifest = FINALS / "MANIFEST.csv"
    carried = []
    if manifest.exists():
        have = {r["destination"] for r in rows}
        for _, r in pd.read_csv(manifest).iterrows():
            if r["destination"] not in have and (FINALS / r["destination"]).exists():
                carried.append(dict(r))
        if carried:
            print(f"\n  carried {len(carried)} manifest rows whose sources are gone "
                  f"but whose files remain in the tree")

    pd.DataFrame(rows + carried).sort_values("destination").to_csv(
        manifest, index=False)
    write_readme(len(rows))
    total = sum(r["bytes"] for r in rows)
    print(f"\nwrote {len(rows)} files ({total/1e9:.2f} GB) -> {FINALS}")


def _gap_row() -> str:
    """The README's amortization_gap line, describing what is actually present."""
    d = FINALS / "amortization_gap"
    fams = sorted(x.name for x in d.glob("*") if x.is_dir()) if d.is_dir() else []
    if not fams:
        return "reserved — BayesFlow comparison, not yet run"
    n = len(list(d.rglob("*.csv")))
    pend = [f for f in FAMILIES if f not in fams]
    tail = f"; {', '.join(x.upper() for x in pend)} pending" if pend else ""
    return (f"C2ST against the per-case BayesFlow baselines — "
            f"{', '.join(x.upper() for x in fams)}, {n} CSVs{tail}")


def write_readme(n_files: int):
    ck = f"{CKPT}\n  sha256: {sha256(CKPT)}" if CKPT.exists() else f"{CKPT} (MISSING)"
    (FINALS / "README.md").write_text(f"""# Experiment 1 — generalising to design space

Frozen snapshot. Every figure here comes from ONE checkpoint; this tree supersedes
the scattered per-run directories under `outputs/figures/model_class/cf/`.

Assembled {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} by
`experiments/finalize_experiment_1.py` — {n_files} files.

## Provenance

- checkpoint: `{ck}`
- git commit: `{git_commit()}`
- per-file sources, sizes and original mtimes: `MANIFEST.csv`

## Contents

| folder | what |
|---|---|
| `random_ensemble/` | 12 randomly drawn design configs, 4 per family (one combined set) |
| `benchmark_ensemble/` | `bench5` (5 OG cases) and `bench6` (+ `mixed_predictor`), per family |
| `per_case/` | the 18: 6 benchmark cases × 3 families, plotted separately |
| `information_gain/` | expected information gain and NRMSE vs trial count, four continuous/binary designs |
| `amortization_gap/` | {_gap_row()} |
| `predictions/` | raw posterior draws backing every figure, for re-plotting |

## Amortization gap — how to read it

`amortization_gap/<family>/summary.csv` has one row per case with `mean_accuracy`
(mean over per-parameter cells) and `joint_c2st` (all active cells at once).
Each `<family>/<case>/` holds the comparison and both arms that produced it:

| file | what |
|---|---|
| `orig_vs_bf.csv` | per-cell C2ST, plus `_mean_accuracy_` and `_joint_c2st_` rows |
| `bf_*.pdf` / `bf_metrics.csv` | the per-case BayesFlow baseline's own diagnostics |
| `bf_evaluations.csv` | BF NRMSE, calibration error, posterior contraction |
| `cf_*.pdf` / `cf_metrics*.csv` | the CogFormer arm, same datasets |

The `cf_*` files come from the PAIRED validate run, not from `per_case/`. `per_case/`
simulated its own datasets; the C2ST re-ran validation against the BF simulations so
both arms score identical data. Compare `bf_*` with `cf_*` here — comparing `bf_*`
against `per_case/` would be comparing across different data.

**Read `joint_c2st`, not `mean_accuracy`.** Both metrics pool draws across all 200
datasets, and the pooled 1-D marginal of any single parameter is dominated by
between-dataset spread, which both approximators reproduce — so `mean_accuracy`
sits near 0.55 while `joint_c2st` runs 0.96–0.998. That is not a contradiction and
not seven weak signals accumulating: seven independent cells at 0.55 would combine
to about 0.63, not 0.99. Within a single dataset the posteriors genuinely differ —
centres by 1–2.5 posterior SDs, widths by 2.3–10x — and only the joint classifier
can localise a draw well enough to see it. Same pooling effect that
`model_class_paired_c2st.py` documents for its per-dataset arm.

**These numbers have no null floor.** `model_family_c2st.py` has none of the
null-calibration machinery in `model_class_paired_c2st.py`, so nothing here is
baseline-corrected. The joint values are far past any plausible floor (~0.49), but
a per-cell 0.545 must not be quoted as "0.045 above chance".

**C2ST is direction-blind, and here the direction favours CogFormer.** On identical
datasets, the amortized model beats the per-case BayesFlow baseline on every
parameter of both families — DDM `v` correlation 0.965 vs 0.817, RMSE 0.209 vs
0.460 — while producing *wider* posteriors. The BF baselines are badly
overconfident: their own diagnostics in
`outputs/evaluations/model_family/bf/` report calibration error 0.39–0.47 and
posterior contraction 0.993–0.999, and their posterior SDs are ~10x smaller than
their own RMSE. So a large gap here is mostly a statement about BF, not a cost of
amortization. Budgets differ and are not matched: ~51M simulations per BF case
against ~256M for the one checkpoint covering every case and family.

## What is included

An allowlist of plot stems -- `recovery`, `coverage`, `ecdf`, `metrics` and their
`_pi` companions -- not "whatever was in the source directory". The validate script
emits posterior pairplots by default, so a sweep-everything copier would pull in 180
extra files whenever a run forgets `--skip_posteriors`.

## Naming

Redundant prefixes are stripped: the directory already says which family, case and
plot family a file belongs to, so `ddm_interaction_fm_recovery.pdf` becomes
`per_case/ddm/interaction/recovery.pdf`.

`_global` is renamed `_pi`. That panel shows the global-parameter estimates, which
today means exactly pi (contamination probability). **If a second global parameter is
ever added, this rename becomes wrong** — the panel would no longer be pi-specific.
`MANIFEST.csv` flags every renamed file so the mapping is recoverable.

## Information gain

`information_gain/` sweeps four designs (CC/CB/BC/BB — each of two predictors either
continuous or binary) against trial count, both predictors loading on `v` and `a`.
The left panel is a Gaussian expected information gain, a log-volume RATIO, chosen
because a ratio cancels the per-SD rescaling that puts continuous and binary regressors
on a common estimand.

The right panel is mean NRMSE over the 11 active cells (draw-based RMSE against the
truth, normalised by the SD of the targets -- the repo's canonical `_rmse`). It
REPLACED a posterior-contraction panel, which said nothing the EIG panel did not: EIG
is the log-volume contraction of the same posterior, so the two are near-monotone
transforms of one another and BOTH are pure posterior-WIDTH measures. Neither can see a
design that sharpens the posterior around the wrong value. Since draw-based MSE
decomposes as

    NRMSE^2 = bias^2 / var(true) + (1 - contraction)

the old panel reported only the second term. The recovered bias/sd(true) runs 0.15-0.44
and is non-zero at every design and trial count, so that omission was real rather than
theoretical. It also separates two readings the old panel confounded: BB's NRMSE
flattens at 0.443/0.444 between 800 and 1000 trials while CC keeps improving to 0.305,
so the all-binary design stops converting extra trials into ACCURACY, not merely into
posterior width.

`posterior_draws.npz` holds the draws and targets behind both panels for all 20 cells
(`<design>_<n>_pred` (200, 200, 11), `<design>_<n>_true` (200, 11)). The first version
of this sweep persisted only summary scalars, so adding NRMSE later cost a second
four-hour GPU run -- posterior draws against truth cannot be reconstructed from a
contraction value. Any further metric is now a CPU replot.

Two caveats travel with the numbers:

- The designs carry DIFFERENT priors over that estimand (1.704 nats between CC and BB,
  from the `cell_means` contrast having twice the variance of a continuous slope). That
  offset is constant in n, so it shifts the level of every gap but cancels from the
  slope. Sigma_prior is itself estimated from 1500 draws, so re-running the sweep moves
  EIG by ~0.2 nats (max 0.78 observed) without changing the design ordering at any
  trial count. Only — only the TREND and the derived trial multipliers are prior-free.
- Category proportions are Dirichlet-drawn, so "binary" means an unbalanced
  dichotomisation (realised balance 0.38–0.45), not a median split. These curves answer
  factor-vs-parametric, not median-split-vs-continuous.

## How the design configs were drawn

The random ensemble uses rejection sampling: the whole SET of 12 is redrawn until its
union covers every estimable cell, so the grid has no N/A cells and every config is
still a genuine random draw. This replaced a scheme that made the last config of each
family a deterministic saturated design. The configs actually drawn are recorded in
each `random_ensemble/*/design_configs.csv`.

Note the configs are not marginally independent — they are a uniform sample from the
*covering* sets. Measured density cost of that conditioning is about +10% relative to
unconditional sampling.
""")


if __name__ == "__main__":
    main()
