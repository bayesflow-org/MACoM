"""Assemble outputs/finals/experiment-3 -- LBA zero-shot inference and finetuning.

Source is the canonical three-arm paired validation at
`outputs/figures/model_class/cf/paired_lba_20260816/`, consolidated 2026-08-17.

ALLOWLIST, NOT SWEEP
--------------------
Unlike the experiment-1/2 assemblers, this one copies only files whose stem is in
KEEP. The source validate script emits 10 posterior pairplots per case by default
(`--skip_posteriors` defaults to False), and a sweep-everything copier would pull
them in silently whenever someone re-runs without the flag. Naming the four wanted
plot types makes the tree's contents a deliberate choice rather than a side effect
of which flags a run happened to pass.

    python -m experiments.finalize_experiment_3 --dry_run
    python -m experiments.finalize_experiment_3
"""
import argparse
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

FINALS = Path("outputs/finals/experiment-3")
SRC = Path("outputs/figures/model_class/cf/paired_lba_20260816")
# Finetuning trajectory: diagnostics at epochs 0/500/1000/1500/2000, evaluated on
# identical seeded datasets. Separate source tree because it is a different KIND of
# result from the three-arm paired validation -- the arms are endpoints, this is the
# path between them -- and is produced by its own script.
TRAJ_SRC = Path("outputs/figures/model_class/cf/lba_trajectory")
# LBA applied to the empirical lexical-decision data (17 subjects). It lives HERE
# rather than in experiment-2 because experiment-2 is the DDM/RDM/CDM real-data tree
# -- the three families the model class was trained on -- while LBA is the held-out
# family this experiment is about. Same checkpoint as the finetuned arm above.
PPC_SRC = Path("outputs/figures/real_data/ppc")
PPC_DATA = Path("outputs/data/real_data/lba/mc/lba_mc_posteriors.npz")
# variant -> destination subdirectory under ppc/. Two PPCs were run from the SAME
# posterior draws, differing only in what the forward simulation holds fixed:
#   free          -- s_v and s_tau as the model estimated them
#   conventional  -- s_v=1.0, s_tau=0.0, the standard LBA identifiability constraint
# Both are kept, in labelled subtrees, because which one belongs in the paper is a
# modelling decision and a single unlabelled `ppc/` would make it unrecoverable.
PPC_VARIANTS = {"lba": "free", "lba_conventional": "conventional"}
# Same allowlist and rename as finalize_experiment_2, so the two real-data trees
# name the same plot the same way.
PPC_KEEP = ("ppc", "choice_ppc", "rt_quantiles", "rt_qq", "choice_quantiles")

# arm -> (families, filename infix). zeroshot covers LBA only.
ARMS = {"zeroshot": (["lba"], "mid1"),
        "finetuned": (["cdm", "ddm", "lba", "rdm"], "ft"),
        "original": (["cdm", "ddm", "rdm"], "fm")}
SCENARIOS = ["intercept_only", "fixed", "regressed", "fixed_regressed",
             "interaction", "mixed_predictor"]
KEEP = {"recovery", "coverage", "ecdf", "metrics"}


def tidy(rest: str) -> str:
    return rest.replace("_global", "_pi")


def kept(rest: str) -> bool:
    """Allowlist on the plot-type stem: recovery/coverage/ecdf/metrics, +_global."""
    return rest.split(".")[0].replace("_global", "") in KEEP


def ppc_kept(rest: str) -> bool:
    return any(rest.startswith(k + "_") for k in PPC_KEEP)


def ppc_tidy(rest: str) -> str:
    """Rename the bare ``ppc_`` plot to ``rt_ppc_``, as experiment-2 does.

    In the source the RT-distribution plot is just ``ppc_``, which is ambiguous
    sitting next to ``choice_ppc_`` -- nothing in the name says which dimension it
    checks. The other types already say so.
    """
    return f"rt_{rest}" if rest.startswith("ppc_") else rest


def plan_ppc() -> tuple[list[tuple[Path, Path]], list[Path]]:
    """Real-data PPC jobs: (source, destination) pairs plus allowlist rejects."""
    jobs, skipped = [], []
    for variant, dest_name in PPC_VARIANTS.items():
        d = PPC_SRC / variant
        if not d.is_dir():
            continue
        for src in sorted(d.glob("*.pdf")):
            rest = src.name.replace("lba_mc_", "")
            if not ppc_kept(rest):
                skipped.append(src); continue
            jobs.append((src, FINALS / "ppc" / dest_name / ppc_tidy(rest)))
    # One posteriors file, not one per variant: both PPCs were simulated from the
    # same draws, so duplicating it would imply two fits where there is one.
    if PPC_DATA.exists():
        jobs.append((PPC_DATA, FINALS / "ppc" / "posteriors.npz"))
    return jobs, skipped


def plan():
    jobs, skipped = [], []
    for arm, (families, infix) in ARMS.items():
        for fam in families:
            for scen in SCENARIOS:
                # zeroshot nests one extra level (a single variant dir) and has no
                # family level, since it is LBA-only; flatten both away.
                cands = ([SRC / arm / scen / d for d in ("mid_rdm",)]
                         if arm == "zeroshot" else [SRC / arm / fam / scen])
                for cdir in cands:
                    if not cdir.is_dir():
                        continue
                    for src in sorted(cdir.iterdir()):
                        rest = src.name.replace(f"{fam}_{scen}_{infix}_", "")
                        if not kept(rest):
                            skipped.append(src)
                            continue
                        jobs.append((src, FINALS / arm / fam / scen / tidy(rest)))
        pdir = SRC / arm / "predictions"
        if pdir.is_dir():
            for src in sorted(pdir.glob("*.npz")):
                jobs.append((src, FINALS / "predictions" / arm / src.name))

    cdir = SRC / "c2st"
    for src in sorted(cdir.rglob("*.csv")) if cdir.is_dir() else []:
        rel = src.relative_to(cdir)
        name = src.name
        if len(rel.parts) == 2:                       # <fam>/<fam>_<scen>_...csv
            name = name.replace(f"{rel.parts[0]}_", "", 1)
        jobs.append((src, FINALS / "c2st" / rel.parent / name))

    # Trajectory: the figure plus BOTH backing CSVs. The per-scenario file matters
    # because the family mean hides real disagreement between scenarios (LBA's
    # per-scenario NRMSE spans 0.20-0.48 at epoch 0), and a reader who only has the
    # summary cannot recover that.
    if TRAJ_SRC.is_dir():
        for src in sorted(TRAJ_SRC.iterdir()):
            if src.suffix in {".pdf", ".csv"}:
                jobs.append((src, FINALS / "trajectory" / src.name))

    readme = SRC / "README.md"
    if readme.exists():
        jobs.append((readme, FINALS / "paired_validation_notes.md"))

    ppc_jobs, ppc_skipped = plan_ppc()
    jobs += ppc_jobs
    skipped += ppc_skipped
    return jobs, skipped


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    jobs, skipped = plan()
    present = [(s, d) for s, d in jobs if s.exists()]
    counts: dict[str, int] = {}
    for _, d in present:
        counts[d.relative_to(FINALS).parts[0]] = counts.get(d.relative_to(FINALS).parts[0], 0) + 1
    gb = sum(s.stat().st_size for s, _ in present) / 1e9
    print(f"{len(present)} files to place ({gb:.2f} GB)")
    for k in sorted(counts):
        print(f"  {k:14} {counts[k]:4d}")
    print(f"\n  excluded by allowlist: {len(skipped)}")
    for s in skipped[:5]:
        print(f"    {s.name}")
    if args.dry_run:
        print("\ndry run; nothing written")
        return

    rows = []
    for src, dest in present:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        st = src.stat()
        rows.append({"destination": str(dest.relative_to(FINALS)), "source": str(src),
                     "bytes": st.st_size,
                     "source_mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc)
                                     .strftime("%Y-%m-%d %H:%M:%SZ"),
                     "renamed_global_to_pi": "_pi" in dest.name and "_global" in src.name})
    pd.DataFrame(rows).sort_values("destination").to_csv(FINALS / "MANIFEST.csv", index=False)
    write_readme(len(rows), len(skipped))
    print(f"\nwrote {len(rows)} files -> {FINALS}")


def _traj_note() -> str:
    """State plainly whether the trajectory is complete, rather than letting a partial
    curve sit in a frozen tree looking finished."""
    import glob as _glob
    figs = _glob.glob(str(FINALS / "trajectory" / "lba_finetuning_trajectory*.pdf"))
    full = any(f.endswith("lba_finetuning_trajectory.pdf") for f in figs)
    if full:
        return "Complete: all five checkpoints (0/500/1000/1500/2000)."
    eps = sorted({e for f in figs for e in re.findall(r"ep([0-9-]+)", f)})
    return ("**INTERIM — the evaluation is still running.** Epochs present: "
            f"{', '.join(eps) if eps else 'none'}. The remaining checkpoints are still "
            "being scored; this figure will be replaced by the five-epoch version.")


def write_readme(n, n_skipped):
    (FINALS / "README.md").write_text(f"""# Experiment 3 — LBA zero-shot inference and finetuning

Frozen snapshot of the three-arm paired validation. Assembled
{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} by
`experiments/finalize_experiment_3.py` — {n} files. Git commit `{git_commit()}`.

`paired_validation_notes.md` is the original run's write-up and carries the headline
tables; `MANIFEST.csv` records per-file provenance.

## ppc/ — LBA on empirical data

Posterior predictive checks for the finetuned checkpoint applied to the empirical
lexical-decision dataset (17 subjects, 500 stored draws each). It sits in experiment 3
rather than experiment 2 because experiment 2 is the DDM/RDM/CDM real-data tree — the
three families the model class was trained on — while LBA is the held-out family this
experiment exists to study. Same checkpoint as `finetuned/`.

**Two variants, same posterior draws.** They differ only in what the forward
simulation holds fixed:

| folder | s_v | s_tau | what it is |
|---|---|---|---|
| `ppc/free/` | estimated | estimated | the model's own fit, simulated as-is |
| `ppc/conventional/` | 1.0 | 0.0 | the standard LBA identifiability constraint |

`posteriors.npz` is stored ONCE, at `ppc/`, not per variant: both PPCs read the same
draws, so duplicating it would imply two fits where there is one.

Which variant belongs in the paper is a modelling decision and has NOT been made here.
Both are kept in labelled subtrees precisely so that decision stays visible — a single
unlabelled `ppc/` would make the constraint unrecoverable from the tree.

Plot names follow experiment 2 exactly, so the same check is called the same thing in
both real-data trees: `rt_ppc_*` (RT distribution), `choice_ppc_*` (choice
proportions), `rt_qq_*`, `rt_quantiles_*`, `choice_quantiles_*`, suffixed by condition
(`condition`, `frequency`, `lexicality`).

## trajectory/ — finetuning path, NOT just its endpoints

{_traj_note()}

The three arms above are ENDPOINTS: pre-finetune and post-finetune. `trajectory/` adds
the path between them, evaluating diagnostics at epochs 0/500/1000/1500/2000 of a
re-run finetune. It answers a question the endpoints assert but cannot show: whether
LBA's gain is gradual or front-loaded, and whether the three trained families drift at
any point rather than only at the end.

- `lba_finetuning_trajectory*.pdf` — NRMSE, posterior contraction and calibration
  error against epoch, one line per family. Note contraction runs the OPPOSITE way to
  the other two (higher is better), which is stated on each panel.
- `trajectory_summary.csv` — the plotted means, aggregated over parameters then over
  scenarios.
- `trajectory_per_scenario.csv` — one row per (epoch, family, scenario, metric). Kept
  because the family mean hides genuine disagreement between scenarios: LBA's
  per-scenario NRMSE spans 0.20-0.48 at epoch 0.

Epoch 0 is not "untrained": it is the pre-finetune checkpoint with LBA's embedding row
warm-started by COPYING RDM's, which is exactly the zero-shot condition. Every epoch was
scored on IDENTICAL datasets (`seed_case` with a fixed seed), verified by byte-comparing
`true_set` across epochs, so movement along a line is the checkpoint changing and not
the test set.

**Lineage warning.** The trajectory comes from a SECOND finetune run (2026-09-09,
seeded), while `finetuned/` above still comes from the 2026-08-14 run, which set no
seed. The two are therefore not the same model, and the trajectory's endpoint will not
exactly equal the `finetuned/` numbers. Reconciling them -- rebuilding the arms from the
seeded lineage -- is pending.

## Arms

| folder | checkpoint evaluated | families |
|---|---|---|
| `zeroshot/` | original, pre-finetune | LBA only |
| `finetuned/` | LBA-finetuned | LBA + DDM/CDM/RDM |
| `original/` | original, pre-finetune | DDM/CDM/RDM |
| `c2st/` | both, compared | DDM/CDM/RDM, 5 scenarios |
| `predictions/` | — | posterior draws per arm |

Each case folder holds `recovery`, `coverage`, `ecdf`, `metrics` as PDF (plus
`metrics.csv`) and their `_pi` global-parameter companions.

## Headline

LBA mean NRMSE 0.2386 → 0.1237 (**-48.2%**); the three original families move
+0.3% / +1.0% / +0.5%.

**Read that second number alongside `c2st/forgetting_c2st_summary.csv`, not instead
of it.** Aggregate NRMSE barely moves, but the per-dataset C2ST columns in that file
report `perds_mean_excess` of 0.24-0.37 over the null with `perds_frac_excess_gt_05`
at 0.985-1.0 — on essentially every dataset the finetuned and original posteriors are
distinguishable. The matched-sample C2ST in the same file shows excess of only
0.009-0.021, which is the expected degeneracy: both models are calibrated, so pooled
draws approach the prior and the test goes to its null by construction. The two
statistics are not in conflict, but "no catastrophic forgetting" is a stronger claim
than aggregate NRMSE alone supports.

**The NRMSE metric is also not precise enough to carry that claim.**
`bf.diagnostics.metrics.root_mean_squared_error` is STOCHASTIC: three calls on
byte-identical arrays return

    [1.0041 0.2435 1.0260 0.1322 ...]
    [1.0088 0.2446 1.0172 0.1329 ...]
    [0.9887 0.2427 1.0222 0.1347 ...]

i.e. roughly 1-2% spread run to run. (`posterior_contraction` on the same arrays is
deterministic to 4 dp, so this is the RMSE estimator, not the data.) The +0.3% / +1.0%
/ +0.5% movements reported for CDM/DDM/RDM therefore sit INSIDE the metric's own
noise: they cannot distinguish "no forgetting" from "a small amount of forgetting" in
either direction. Any re-derivation of those numbers should average over repeated
calls, or use contraction, which is reproducible.

## Excluded

{n_skipped} posterior pairplots ({n_skipped} files, 10 per case in `original/`) were
deleted at source and are excluded here. This assembler uses an ALLOWLIST of plot
stems rather than copying directory contents, so a future run that forgets
`--skip_posteriors` cannot reintroduce them silently.

## Naming

Family/scenario/arm infixes (`_fm_`, `_ft_`, `_mid1_`) are stripped — the path already
carries them. `_global` is renamed `_pi`; that panel is the global-parameter view,
which today is exactly pi. The zeroshot arm's single extra variant level (`mid_rdm/`)
is flattened.
""")


if __name__ == "__main__":
    main()
