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
datasets -- the CogFormer arm comes from the PAIRED validation run (CF_PAIRED), not
from the per-case run behind experiment-1's per_case/, which simulated its own data.

PROVENANCE WARNING (recorded 2026-09-11)
----------------------------------------
The DDM and CDM results in this tree were produced by a run whose SOURCES NO LONGER
EXIST: outputs/figures/c2st/ and outputs/figures/model_class/cf_paired/ are absent,
there are zero outputs/data/model_family/*_data.npz, and 1 of 15 BF checkpoints
survives. RDM was never run at all -- yet amortization_gap.tex quotes RDM numbers,
so that run happened and its artifacts were lost.

This is why main() carries forward manifest rows whose sources are gone: a plain
regenerate-from-sources would silently shrink the record from 172 rows to 0 while the
files stayed on disk. Do not "fix" that by dropping the carry-forward.

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

FINALS = Path("outputs/finals/experiment-1a")
C2ST = Path("outputs/figures/c2st")
BF_FIG = Path("outputs/figures/model_family/bf")
BF_EVAL = Path("outputs/evaluations/model_family/bf")
CF_PAIRED = Path("outputs/figures/model_class/cf_paired")

FAMILIES = ["ddm", "rdm", "cdm"]
# bench5. model_family_c2st drops "full", and the BF baselines were never trained on
# "mixed_predictor", so both are absent by construction, not by omission.
CASES = ["intercept_only", "fixed", "regressed", "fixed_regressed", "interaction"]

# Hand-written, not copied from any source, so it has no manifest row and no source
# to regenerate from. Listed here so a reader does not mistake it for an orphan.
AUTHORED = ["amortization_gap.tex"]


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
        have_rows = len(pd.read_csv(manifest))

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

    # Carry forward rows this run cannot regenerate. See the PROVENANCE WARNING in
    # the module docstring: for DDM and CDM this is currently the ONLY thing keeping
    # 172 rows of record attached to 172 files that still exist on disk.
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
            out.append(f"| {fam.upper()} | — | absent — never run |")
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
datasets — the CogFormer arm is the PAIRED validation run, not the per-case run
behind `experiment-1/per_case/`.

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

`amortization_gap.tex` is hand-written (the paper table) and has no manifest row —
it is authored, not copied.

## How to read it

`<family>/summary.csv` has one row per case with `mean_accuracy` (mean over
per-parameter cells) and `joint_c2st` (all active cells at once).

The `cf_*` files come from the PAIRED validate run, not from
`experiment-1/per_case/`. `per_case/` simulated its own datasets; the C2ST re-ran
validation against the BF simulations so both arms score identical data. Compare
`bf_*` with `cf_*` here — comparing `bf_*` against `per_case/` would be comparing
across different data.

**Read `joint_c2st`, not `mean_accuracy`.** Both metrics pool draws across all 200
datasets, and the pooled 1-D marginal of any single parameter is dominated by
between-dataset spread, which both approximators reproduce — so `mean_accuracy` sits
near 0.55 while `joint_c2st` runs 0.96–0.998. That is not a contradiction and not
seven weak signals accumulating: seven independent cells at 0.55 would combine to
about 0.63, not 0.99. Within a single dataset the posteriors genuinely differ —
centres by 1–2.5 posterior SDs, widths by 2.3–10x — and only the joint classifier can
localise a draw well enough to see it. Same pooling effect that
`model_class_paired_c2st.py` documents for its per-dataset arm.

**These numbers have no null floor.** `model_family_c2st.py` has none of the
null-calibration machinery in `model_class_paired_c2st.py`, so nothing here is
baseline-corrected. The joint values are far past any plausible floor (~0.49), but a
per-cell 0.545 must not be quoted as "0.045 above chance".

**C2ST is direction-blind, and here the direction favours CogFormer.** On identical
datasets the amortized model beats the per-case BayesFlow baseline on every parameter
of both families — DDM `v` correlation 0.965 vs 0.817, RMSE 0.209 vs 0.460 — while
producing *wider* posteriors. The BF baselines are badly overconfident: their own
diagnostics report calibration error 0.39–0.47 and posterior contraction 0.993–0.999,
and their posterior SDs are ~10x smaller than their own RMSE. So a large gap here is
mostly a statement about BF, not a cost of amortization. Budgets differ and are not
matched: ~51M simulations per BF case against ~256M for the one checkpoint covering
every case and family.

## Provenance warning

The DDM and CDM files here come from a run whose **sources no longer exist**:
`outputs/figures/c2st/` and `outputs/figures/model_class/cf_paired/` are absent,
there are zero `outputs/data/model_family/*_data.npz`, and 1 of 15 BayesFlow
checkpoints survives. **RDM was never run** — yet `amortization_gap.tex` quotes RDM
numbers, so that run happened and its artifacts were lost.

Re-deriving any of this is a full rebuild (~15 BayesFlow trainings):

```bash
bash run_experiments.sh exp1a
```

`MANIFEST.csv` therefore carries forward rows whose sources are gone. Re-running this
assembler never removes files it cannot regenerate.

## Excluded

`full` (dropped by `model_family_c2st`) and `mixed_predictor` (no BF baseline was
ever trained on it) have no BF arm to compare against, so they are absent by
construction rather than by omission. The 200-epoch baseline variant lands in
`e200/` via `experiments/finalize_experiment_1a_e200.py` — a separate budget, kept
in its own subtree and its own manifest so the two are never mixed.
""")


if __name__ == "__main__":
    main()
