"""Assemble outputs/finals/experiment-1a -- the amortization gap.

WHY A SEPARATE EXPERIMENT FROM 1
--------------------------------
The gap is the only part of the Experiment-1 story that cannot be produced from the
ModelClass checkpoint alone: it needs a case-specific BayesFlow baseline trained per
(family, case), which is 15 trainings. Keeping it inside experiment-1 meant the whole
of experiment-1 inherited that cost, and a cheap re-run of the per-case or ensemble
figures sat behind a stage nobody wanted to trigger. Splitting it out makes
experiment-1 re-runnable in minutes and gives the expensive arm its own tree, its own
manifest and its own README.

The tree is FLAT: families sit at the top (`ddm/`, `rdm/`, `cdm/`), because this
directory IS the gap study -- an `amortization_gap/` level inside
`outputs/finals/experiment-1a/` would say the same word twice.

WHAT THE GAP IS
---------------
A paired C2ST: for each bench5 case, how distinguishable are the posteriors of the
case-specific BayesFlow baseline from those of the single amortized ModelClass
CogFormer? 0.5 = indistinguishable = no gap. Both arms score the SAME simulated
datasets -- the shared benchmark files written by e1_data, which experiment 1's
per-case validation (CF_PAIRED) and the baselines both load.

HISTORY
-------
The pre-2026-09-27 contents of this tree were produced by BayesFlow baselines that
trained on ~320 distinct datasets (keras forked its simulation workers with cloned
RNG state every epoch), hence calibration error ~0.4 and contraction ~0.997. They were
deleted; model_family_bf_train now reseeds per worker process. main() still carries
forward manifest rows whose files remain in the tree but whose sources are gone, so a
partial rebuild never silently shrinks the record.

    python -m experiments.finalize_experiment_1a --dry_run
    python -m experiments.finalize_experiment_1a
"""
import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Reuse experiment-1's allowlist and renaming rather than restating them, so the two
# snapshots cannot drift in what they keep or how they name the global-param panel.
from experiments.finalize_experiment_1 import git_commit, kept, tidy
from experiments import shared_config

FINALS = Path("outputs/finals/experiment-1a")
C2ST = Path("outputs/figures/c2st")
BF_FIG = Path("outputs/figures/model_family/bf")
BF_EVAL = Path("outputs/evaluations/model_family/bf")
# The CogFormer arm is experiment 1's per-case validation: it scores the same shared
# benchmark files as the baselines (e1_data), so no separate paired run exists.
CF_PAIRED = Path("outputs/figures/model_class/cf") / f"{shared_config.get('MC_TAG')}_validate"

FAMILIES = ["ddm", "rdm", "cdm"]
# bench5. model_family_c2st drops "full", and the BF baselines were never trained on
# "mixed_predictor", so both are absent by construction, not by omission.
CASES = ["intercept_only", "fixed", "regressed", "fixed_regressed", "interaction"]

# amortization_gap_tables.tex is generated (experiments.experiment_1a.gap_tables) from the
# sources, not copied, so it has no manifest row. Listed here so a reader does not
# mistake it for an orphan.
AUTHORED = ["amortization_gap_tables.tex"]


def plan() -> tuple[list[tuple[Path, Path]], list[Path]]:
    """(source, destination) pairs, plus files rejected by the allowlist."""
    jobs: list[tuple[Path, Path]] = []
    skipped: list[Path] = []

    for fam in FAMILIES:
        # The scenario tag is globbed, not hardcoded: model_family_c2st writes
        # <fam>_<case>_{orig_vs_bf,ft_vs_bf,ft_vs_orig}_c2st.csv depending on which
        # arms existed at run time.
        jobs.append((C2ST / fam / f"{fam}_c2st_summary.csv", FINALS / fam / "summary.csv"))
        for case in CASES:
            cdir = C2ST / fam / case
            for src in sorted(cdir.glob(f"{fam}_{case}_*_c2st.csv")) if cdir.is_dir() else []:
                tag = src.name[len(f"{fam}_{case}_"):-len("_c2st.csv")]
                jobs.append((src, FINALS / fam / case / f"{tag}.csv"))

    # Both arms per case, so each case folder stands alone: bf_* is the per-case
    # BayesFlow baseline, cf_* the paired CogFormer run scored on the same datasets.
    for fam in FAMILIES:
        for case in CASES:
            dest_dir = FINALS / fam / case

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

    return jobs, skipped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    jobs, skipped = plan()
    present = [(s, d) for s, d in jobs if s.exists()]
    print(f"{len(present)} files to place\n")

    groups: dict[str, int] = {}
    for _, d in present:
        k = d.relative_to(FINALS).parts[0]
        groups[k] = groups.get(k, 0) + 1
    for k in sorted(groups):
        print(f"  {k:12} {groups[k]:4d} files")

    have_rows = 0
    manifest = FINALS / "MANIFEST.csv"
    if manifest.exists():
        # Only rows whose file is still in the tree get carried; count those, not the
        # raw manifest length, or the dry run over-promises.
        have_rows = sum((FINALS / d).exists() for d in pd.read_csv(manifest)["destination"])

    missing = [f for f in FAMILIES if not (C2ST / f / f"{f}_c2st_summary.csv").exists()]
    if missing:
        print(f"\n  [pending] no C2ST summary for {', '.join(missing)} -- their "
              f"BayesFlow baselines have not been (re)trained. Run the exp1a stages "
              f"of run_experiments.sh. Existing files in the tree are NOT removed.")
    if skipped:
        print(f"\n  excluded by allowlist: {len(skipped)}")

    if args.dry_run:
        print(f"\n{have_rows} manifest rows already in the tree would be carried forward.")
        print("dry run; nothing written")
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

    # Carry forward rows this run cannot regenerate, as long as their files are still
    # in the tree (see HISTORY in the module docstring).
    carried = []
    if manifest.exists():
        have = {r["destination"] for r in rows}
        for _, r in pd.read_csv(manifest).iterrows():
            if r["destination"] not in have and (FINALS / r["destination"]).exists():
                carried.append(dict(r))
        if carried:
            print(f"\n  carried {len(carried)} manifest rows whose sources are gone "
                  f"but whose files remain in the tree")

    pd.DataFrame(rows + carried).sort_values("destination").to_csv(manifest, index=False)
    write_readme(len(rows), len(carried))
    total = sum(r["bytes"] for r in rows)
    print(f"\nwrote {len(rows)} files ({total/1e9:.2f} GB) -> {FINALS}")


def _family_rows() -> str:
    out = []
    for fam in FAMILIES:
        d = FINALS / fam
        if not d.is_dir():
            out.append(f"| {fam.upper()} | — | not yet produced |")
            continue
        cases = sorted(c.name for c in d.iterdir() if c.is_dir())
        n = sum(1 for _ in d.rglob("*") if _.is_file())
        out.append(f"| {fam.upper()} | {len(cases)} | {n} files: {', '.join(cases)} |")
    return "\n".join(out)


def write_readme(n_new: int, n_carried: int):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    (FINALS / "README.md").write_text(f"""# Experiment 1a — amortization gap

Assembled {stamp} by `experiments/finalize_experiment_1a.py` at commit `{git_commit()}`.
{n_new} files copied this run, {n_carried} carried from earlier runs.

## What this measures

A paired C2ST per bench5 case: how distinguishable are the posteriors of a
case-specific BayesFlow baseline from those of the single amortized ModelClass
CogFormer? **0.5 = indistinguishable = no gap.** Both arms score the *same* simulated
datasets: the shared benchmark files written once by the `e1_data` stage.

## Layout

`<family>/summary.csv` — the C2ST summary for that family.
`<family>/<case>/` — one folder per bench5 case, holding both arms:

| file | arm |
|---|---|
| `bf_recovery.pdf`, `bf_coverage.pdf`, `bf_ecdf.pdf`, `bf_metrics.pdf` | BayesFlow baseline |
| `bf_evaluations.csv`, `bf_metrics.csv` | NRMSE / calibration error / contraction |
| `cf_*.pdf`, `cf_*.csv` | paired CogFormer (`_pi` = global-parameter panel) |
| `orig_vs_bf.csv` | the C2ST itself |

| family | cases | contents |
|---|---|---|
{_family_rows()}

`amortization_gap_tables.tex` has no manifest row: it is generated from the same
sources by `python -m experiments.experiment_1a.gap_tables` (re-run it after this script).

## How to read it

`<family>/summary.csv` has one row per case. Pooled arms (one classifier over draws
from all 200 datasets): `mean_accuracy` (mean over per-parameter cells) and `joint_c2st`
(all active cells at once). Per-dataset arm (one classifier per dataset, averaged over
datasets): `perds_joint_c2st` and `perds_mean_accuracy`, each with its draw-split
`*_null_floor`. `<case>/orig_vs_bf.csv` has the per-cell values of both arms, and
`<case>/orig_vs_bf_per_dataset.csv` one row per dataset.

The `cf_*` files are experiment 1's per-case validation of the same cases
(`experiment-1/per_case/`), which scored those same shared files. So `bf_*`, `cf_*` and
`experiment-1/per_case/` describe identical datasets and compare directly. (The
seed alone would NOT make that true: it pins the parameters but not the simulated RTs
or regressor designs. Loading the saved files is what pairs the arms.)

**Read the per-dataset arm.** Pooled over 200 datasets, one parameter's draws trace
the aggregate posterior, which is roughly the prior for any calibrated approximator, so
the pooled `mean_accuracy` sits near 0.52 however far apart the per-dataset posteriors
are. The pooled `joint_c2st` sees per-dataset differences only once the datasets
separate in parameter space (9+ active cells), so it undershoots on intercept_only and
fixed and saturates near 0.95 on the rest -- it is not comparable across cases with
different numbers of parameters. Measured 2026-10-01: within a dataset, CogFormer's
posterior means sit a median 0.3–0.8 baseline posterior SDs from the baseline's, and
its posteriors are 1.2–1.6x wider. Same pooling effect `model_class_paired_c2st.py`
documents for experiment 3.

**Per-dataset numbers have a null floor; read them against it.** Real = first halves of
both arms' draws (~100 per side); floor = the larger of each arm's first-half vs
second-half C2ST on the same dataset. The joint uses the MLP; the per-parameter test
uses a quadratic discriminant, because at ~100 draws per side the 1-D MLP barely learns
(0.53 vs 0.67 with QDA on ddm/interaction, null 0.50). The pooled arms have no null.

**C2ST is direction-blind.** A large value says the two posteriors differ, not which
one is better. Read it next to `bf_evaluations.csv` and `cf_metrics.csv` (NRMSE,
calibration error, contraction) on the same datasets.

## Budget

Matched by construction (decided 2026-09-27): the model-class CogFormer trains on
{shared_config.get("MC_EPOCHS")} epochs x 100 steps x {shared_config.get("MC_TRAIN_BATCH")} = 192M simulated datasets
across three families, and each of the 15 BayesFlow baselines on
{shared_config.get("MF_BF_EPOCHS")} x 100 x {shared_config.get("MF_BF_BATCH")} = 12.8M, i.e. 64M per family on both sides, at the same batch size.

## Excluded

`full` (dropped by `model_family_c2st`) and `mixed_predictor` (no BF baseline was
ever trained on it) have no BF arm to compare against, so they are absent by
construction rather than by omission.
""")


if __name__ == "__main__":
    main()
