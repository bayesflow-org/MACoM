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

The main-figure panels are RESTYLED before they are copied (restyle()): the stages
that produce them draw with their scripts' generic defaults, and the page styling
lives in the replot scripts. Doing it here means a re-run of any stage cannot put
the unstyled defaults into finals.

    python -m experiments.finalize_experiment_1 --dry_run
    python -m experiments.finalize_experiment_1
"""
from experiments.paths import RESULTS_ROOT, repo_relative
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from experiments import shared_config

FINALS = (RESULTS_ROOT / "finals/experiment-1")
FIG = (RESULTS_ROOT / "figures/model_class/cf")
DATA = (RESULTS_ROOT / "data/model_class")
# Run identity comes from shared_config.sh, the same file the orchestrator sources, so
# a recipe change there cannot leave this assembler reading the previous run's tree.
BASE = shared_config.get("MC_TAG")
CKPT = Path(shared_config.get("MC_CKPT"))
DESIGN_OPT = (RESULTS_ROOT / f"data/design_opt/{BASE}/ddm")

FAMILIES = ["ddm", "rdm", "cdm"]
CASES = ["intercept_only", "fixed", "regressed", "fixed_regressed",
         "interaction", "mixed_predictor"]
# One random ensemble only: 12 configs spanning all three families, 4 each. There are
# no per-family random ensembles -- the per-family breakdown is the job of
# benchmark_ensemble (bench5/bench6 x ddm/rdm/cdm), so running 12 more configs per
# family would be a different object at ~7h of GPU for no added coverage.
ENSEMBLE_DIR = f"{BASE}_ensemble"


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


def restyle():
    """Redraw the main-figure panels with their page styling, in place in the source
    directories, from the data the stages saved -- no GPU, no resampling.

    The random-ensemble recovery and pi grids come from replot_ensemble, the
    information-gain figure from replot_design_eig; their sizes and layout are tuned
    for main_figure.tex. Only those panels: the other ensemble figures (coverage,
    ecdf) are appendix figures and keep the run's own styling.
    """
    env = {**os.environ, "MPLBACKEND": "Agg"}
    steps = [
        (FIG / ENSEMBLE_DIR / "model_class_ensemble_data.npz",
         ["experiments.experiment_1.replot_ensemble",
          "--datadir", str(FIG / ENSEMBLE_DIR), "--plots", "recovery"]),
        (DESIGN_OPT / "design_eig.csv",
         ["experiments.experiment_1.replot_design_eig", "--datadir", str(DESIGN_OPT)]),
    ]
    for needed, cmd in steps:
        if not needed.exists():
            print(f"  [restyle] skipped {cmd[0]}: {needed} not present yet")
            continue
        print(f"  [restyle] {cmd[0]}")
        subprocess.run([sys.executable, "-m", *cmd], check=True, env=env)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    if not args.dry_run:
        restyle()
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
            "source": repo_relative(src),
            "bytes": st.st_size,
            "source_mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc)
                            .strftime("%Y-%m-%d %H:%M:%SZ"),
            "renamed_global_to_pi": "_pi" in dest.name and "_global" in src.name,
        })
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
| `predictions/` | raw posterior draws backing every figure, for re-plotting |
| `main_figure.tex` | the main-text figure + short caption: random-ensemble recovery and pi, information gain (generated with `captions.tex`; no manifest row) |
| `main_figure_details.tex` | the methods/appendix paragraph that caption refers to (`app:exp1-main-details`) |
| `captions.tex` | appendix figures + captions: `mixed_predictor` and the bench5/bench6 benchmark ensembles (generated by `python -m experiments.experiment_1.appendix_captions`; no manifest row) |

The amortization gap (C2ST against the per-case BayesFlow baselines) is no longer
part of this tree: it needs 15 BayesFlow trainings that nothing else here needs, so
it lives in `outputs/finals/experiment-1a/`, assembled by
`experiments/finalize_experiment_1a.py`.

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
