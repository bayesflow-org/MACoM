"""Assemble outputs/finals/experiment-3 -- LBA zero-shot inference and finetuning.

Final-run layout (2026-09-27): everything is STIMULUS-CODED and comes from one lineage
-- the model-class checkpoint named by MC_TAG in shared_config.sh and its 1000-epoch LBA
finetune. The DDM/RDM/CDM bench5 cases of every arm score the shared benchmark files
(e1_data), so comparisons on those cases are exactly paired. Sources:

    <FIG_MC>/<MC_TAG>_lba/{zeroshot,finetuned,original,c2st}   three-arm validation
    outputs/figures/real_data/ppc/lba_stim_cond[_conventional] real-data PPC

ALLOWLIST, NOT SWEEP
--------------------
Only files whose stem is in KEEP are copied. The validate scripts emit 10 posterior
pairplots per case by default (`--skip_posteriors` defaults to False), and a
sweep-everything copier would pull them in whenever someone re-runs without the flag.

    python -m experiments.finalize_experiment_3 --dry_run
    python -m experiments.finalize_experiment_3
"""
from experiments.paths import RESULTS_ROOT, repo_relative
import argparse
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from experiments import shared_config

FINALS = (RESULTS_ROOT / "finals/experiment-3")
MC_TAG = shared_config.get("MC_TAG")
FIG_MC = (RESULTS_ROOT / "figures/model_class/cf")
SRC = FIG_MC / f"{MC_TAG}_lba"
# The zero-shot validator nests one level per fixed_A value (A0.5/) and one per
# embedding-initialisation variant (mid_rdm/: LBA's row copied from RDM's).
ZS_A = f"A{format(float(shared_config.get('LBA_FIXED_A')), 'g')}"
# LBA applied to the empirical lexical-decision data. It lives HERE rather than in
# experiment-2 because experiment-2 is the DDM/RDM/CDM real-data tree, while LBA is
# the held-out family this experiment is about. Same checkpoint as finetuned/.
PPC_SRC = (RESULTS_ROOT / "figures/real_data/ppc")
PPC_DATA = (RESULTS_ROOT / "data/real_data/lba_stim_cond/mc/lba_mc_posteriors.npz")
# Two PPCs from the SAME posterior draws, differing only in what the forward
# simulation holds fixed. Both kept, in labelled subtrees, because which one belongs
# in the paper is still open.
PPC_VARIANTS = {"lba_stim_cond": "free", "lba_stim_cond_conventional": "conventional"}
# Same allowlist and rename as finalize_experiment_2, so the two real-data trees
# name the same plot the same way.
PPC_KEEP = ("ppc", "choice_ppc", "rt_quantiles", "rt_qq", "choice_quantiles")

# LBA amortization gap: BayesFlow LBA baselines vs the FINETUNED checkpoint, on the
# LBA cases of the shared benchmark files. Same layout as experiment-1a's per-family
# trees so the two gap studies read the same way.
GAP_C2ST = (RESULTS_ROOT / "figures/c2st/lba")
GAP_BF_FIG = (RESULTS_ROOT / "figures/model_family/bf/lba")
GAP_BF_EVAL = (RESULTS_ROOT / "evaluations/model_family/bf")
# BayesFlow LBA bench5 ensembles (experiment_1a/model_family_bf_benchmark_ensemble.py),
# filed beside the CogFormer arms they share a layout and datasets with.
BF_ENSEMBLE = (RESULTS_ROOT / "figures/model_family/bf/ensemble_benchmark")
GAP_CASES = ["intercept_only", "fixed", "regressed", "fixed_regressed", "interaction"]

# Real-data-DESIGN validation (experiment_2/realdata_validate.py --model lba): simulated
# data on the real subjects' designs, filed beside the real-data fit it validates.
VAL_FIG = FIG_MC / f"{MC_TAG}_realdata" / "lba" / "realdata"
VAL_PRED = (RESULTS_ROOT / "data/model_class") / f"{MC_TAG}_realdata" / "lba_realdata_cf_pred.npz"

# arm -> (families, filename infix). zeroshot covers LBA only.
ARMS = {"zeroshot": (["lba"], "mid1"),
        "finetuned": (["cdm", "ddm", "lba", "rdm"], "ft"),
        "original": (["cdm", "ddm", "rdm"], "fm")}
KEEP = {"recovery", "coverage", "ecdf", "metrics"}


def _subdirs(d: Path) -> list[Path]:
    """Scenario folders as they exist on disk. Discovered, not listed, so the tree
    follows whatever val_scenarios the validators ran rather than a copy of the list."""
    return sorted(x for x in d.iterdir() if x.is_dir() and x.name != "predictions") \
        if d.is_dir() else []


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


def plan_ppc(variants=None, data=None, dest_root=None) -> tuple[list[tuple[Path, Path]], list[Path]]:
    """Real-data PPC jobs: (source, destination) pairs plus allowlist rejects.

    Parameterised so the accuracy-coded and stimulus-coded applications go through the
    SAME allowlist and rename rules; two copies of this would be two chances for the
    trees to disagree about what a plot is called.
    """
    variants = variants if variants is not None else PPC_VARIANTS
    data = data if data is not None else PPC_DATA
    dest_root = dest_root if dest_root is not None else FINALS / "ppc"
    jobs, skipped = [], []
    for variant, dest_name in variants.items():
        d = PPC_SRC / variant
        if not d.is_dir():
            continue
        for src in sorted(d.glob("*.pdf")):
            rest = src.name.replace("lba_mc_", "")
            if not ppc_kept(rest):
                skipped.append(src); continue
            jobs.append((src, dest_root / dest_name / ppc_tidy(rest)))
    # One posteriors file, not one per variant: both PPCs were simulated from the
    # same draws, so duplicating it would imply two fits where there is one.
    if data.exists():
        jobs.append((data, dest_root / "posteriors.npz"))
    return jobs, skipped


def plan():
    jobs, skipped = [], []
    for arm, (families, infix) in ARMS.items():
        for fam in families:
            if arm == "zeroshot":
                root = SRC / arm / ZS_A
                cases = [(sd.name, sd / "mid_rdm") for sd in _subdirs(root)]
            else:
                cases = [(sd.name, sd) for sd in _subdirs(SRC / arm / fam)]
            for scen, cdir in cases:
                if not cdir.is_dir():
                    continue
                for src in sorted(cdir.iterdir()):
                    rest = src.name.replace(f"{fam}_{scen}_{infix}_", "")
                    if not kept(rest):
                        skipped.append(src)
                        continue
                    jobs.append((src, FINALS / arm / fam / scen / tidy(rest)))
        pdir = (SRC / arm / ZS_A / "predictions") if arm == "zeroshot" else (SRC / arm / "predictions")
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

    # compare/: paired deltas (Δr heatmaps per family/case, metric-delta table and
    # figures) and the LBA calibration plane. Copied with their relative layout.
    cmp_dir = SRC / "compare"
    for src in sorted(cmp_dir.rglob("*")) if cmp_dir.is_dir() else []:
        if src.is_file() and src.suffix in {".pdf", ".csv"}:
            rel = src.relative_to(cmp_dir)
            jobs.append((src, FINALS / "compare" / rel.parent / tidy(rel.name)))

    # benchmark_ensemble/: LBA bench5 ensembles per arm, in the BayesFlow layout.
    edir = SRC / "ensemble_benchmark"
    for arm in ("zeroshot", "finetuned"):
        for src in sorted((edir / arm).glob("lba_bench5_ensemble_*.pdf")):
            rest = src.name.replace("lba_bench5_ensemble_", "")
            jobs.append((src, FINALS / "benchmark_ensemble" / arm / tidy(rest)))
    jobs.append((edir / "manifest.json", FINALS / "benchmark_ensemble" / "manifest.json"))
    for src in sorted(BF_ENSEMBLE.glob("lba_bench5_ensemble_*.pdf")):
        rest = src.name.replace("lba_bench5_ensemble_", "")
        jobs.append((src, FINALS / "benchmark_ensemble" / "bayesflow" / tidy(rest)))
    jobs.append((BF_ENSEMBLE / "manifest.json", FINALS / "benchmark_ensemble" / "bayesflow" / "manifest.json"))

    # gap/: C2ST summary + per-case C2ST, and the BF arm's diagnostics per case. The
    # CogFormer arm's diagnostics for the same datasets are finetuned/lba/<case>/.
    jobs.append((GAP_C2ST / "lba_c2st_summary.csv", FINALS / "gap" / "summary.csv"))
    for case in GAP_CASES:
        cdir = GAP_C2ST / case
        for src in sorted(cdir.glob(f"lba_{case}_*_c2st.csv")) if cdir.is_dir() else []:
            jobs.append((src, FINALS / "gap" / case / src.name[len(f"lba_{case}_"):]))
        bdir = GAP_BF_FIG / case
        for src in sorted(bdir.glob("*")) if bdir.is_dir() else []:
            rest = src.name.replace(f"lba_family_{case}_bf_", "")
            if not kept(rest):
                skipped.append(src); continue
            jobs.append((src, FINALS / "gap" / case / f"bf_{tidy(rest)}"))
        jobs.append((GAP_BF_EVAL / f"lba_families_bf_{case}_evaluations.csv",
                     FINALS / "gap" / case / "bf_evaluations.csv"))

    for src in sorted(VAL_FIG.glob("*")) if VAL_FIG.is_dir() else []:
        rest = src.name.replace("lba_realdata_fm_", "")
        if not kept(rest):
            skipped.append(src); continue
        jobs.append((src, FINALS / "ppc" / "validation" / tidy(rest)))
    jobs.append((VAL_PRED, FINALS / "ppc" / "validation" / "predictions.npz"))

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
        rows.append({"destination": str(dest.relative_to(FINALS)), "source": repo_relative(src),
                     "bytes": st.st_size,
                     "source_mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc)
                                     .strftime("%Y-%m-%d %H:%M:%SZ"),
                     "renamed_global_to_pi": "_pi" in dest.name and "_global" in src.name})
    pd.DataFrame(rows).sort_values("destination").to_csv(FINALS / "MANIFEST.csv", index=False)
    write_readme(len(rows), len(skipped))
    print(f"\nwrote {len(rows)} files -> {FINALS}")


def write_readme(n, n_skipped):
    (FINALS / "README.md").write_text(f"""# Experiment 3 — LBA zero-shot inference and finetuning

Assembled {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} by
`experiments/finalize_experiment_3.py` — {n} files. Git commit `{git_commit()}`.
Model-class run `{MC_TAG}`; `MANIFEST.csv` records per-file provenance.

Everything here is **stimulus-coded** (`favored_idx` drawn per trial, for LBA and for
RDM at every K) and comes from ONE lineage: the model-class checkpoint above, its
{shared_config.get("LBA_FT_EPOCHS")}-epoch LBA finetune. The DDM/RDM/CDM bench5 cases of every arm
score the shared benchmark files (e1_data), so comparisons on those cases are exactly
paired.

## Arms

| folder | checkpoint evaluated | families |
|---|---|---|
| `zeroshot/` | original, pre-finetune (LBA row copied from RDM's) | LBA only |
| `finetuned/` | LBA-finetuned | LBA + DDM/CDM/RDM |
| `original/` | original, pre-finetune | DDM/CDM/RDM |
| `c2st/` | original vs finetuned, paired (forgetting) | DDM/CDM/RDM |
| `gap/` | BayesFlow LBA baselines vs finetuned, paired (amortization gap) | LBA, 5 bench cases |
| `compare/` | paired deltas: zero-shot->finetuned (LBA), original->finetuned (others) | 4 families, 5 bench cases |
| `benchmark_ensemble/` | zero-shot, finetuned and BayesFlow, five bench cases overlaid per arm | LBA only |
| `predictions/` | — | posterior draws per arm |
| `main_figure.tex`, `main_figure_details.tex`, `captions.tex`, `tables.tex` | — | main-text figure (zero-shot and fine-tuned LBA recovery overlaid, plus the metric deltas; interaction design) and its methods paragraph, appendix figures + captions, and the forgetting-C2ST / LBA-gap tables (generated by `python -m experiments.experiment_3.appendix_captions`; no manifest rows) |

Each case folder holds `recovery`, `coverage`, `ecdf`, `metrics` as PDF (plus
`metrics.csv`) and their `_pi` global-parameter companions.

**Read `c2st/forgetting_c2st_summary.csv` alongside aggregate NRMSE, not instead of
it.** Pooled per-cell and matched-sample C2ST approach their null by construction when
both models are calibrated; the per-dataset columns (`perds_*`) are the ones that can
show a posterior moved. Always read `excess` over the empirical null, not raw `c2st`.

## compare/ — what finetuning changed, as paired differences

`<family>/<case>_delta_r.pdf` (LBA: finetuned minus zero-shot) and
`<family>/<case>_forgetting_delta_r.pdf` (DDM/RDM/CDM: finetuned minus original) map the
change in recovery correlation per coefficient. `delta_metrics.csv` and
`delta_metrics_<case>.pdf` give per-parameter NRMSE and calibration-error deltas
(negative = better). Every coefficient map has the same six intrinsic-parameter
columns (N/A where a case leaves one inactive); the contamination probability pi is
drawn separately in the `_pi` companions (`delta_metrics_pi.csv` for its metrics),
on the color scale of the matching coefficient panel. `lba/interaction_overlay_recovery.pdf` draws zero-shot and finetuned
LBA recovery on one grid (main-text figure, with `delta_metrics_interaction.pdf`).
`lba_calibration_plane.pdf` draws each LBA cell's move from
zero-shot to finetuned in (posterior contraction, calibration error) -- how the
improvement happens, which the mean NRMSE cannot show. Every difference is on
identical datasets; the comparison script refuses unpaired inputs.

## benchmark_ensemble/ — LBA bench5 ensembles

`<arm>/{{recovery,ecdf,coverage,metrics}}[_pi].pdf` overlay the five bench cases for
three arms: CogFormer `zeroshot/` and `finetuned/`, and `bayesflow/` (the five
case-specific BayesFlow baselines of `gap/`). All three are drawn by one function
(experiment-1a's `plot_bench5`): same palette, range-normalised metrics, null seed
and shared global axes. All three score identical datasets, so they compare panel for
panel. Blank zero-shot Log Gamma cells are -inf: the truth falls outside every
posterior draw in most datasets. `bayesflow/manifest.json` also lists the
DDM/RDM/CDM BayesFlow ensembles drawn in the same run.

## gap/ — LBA amortization gap

The held-out family's counterpart of experiment-1a: a case-specific BayesFlow baseline
per bench5 case against the FINETUNED checkpoint (LBA has no original-checkpoint arm).
Both score the LBA cases of the shared benchmark files, and the baselines train at the
same per-case budget as experiment-1a's. `summary.csv` has one row per case; read the
per-dataset columns (`perds_joint_c2st`, `perds_mean_accuracy`) against their null floors,
as in experiment-1a: pooled `mean_accuracy` sits near 0.5 for any calibrated pair, and the
pooled `joint_c2st` saturates near 0.95 once a case has ~9+ active cells, so neither is
comparable across cases. `<case>/bf_*` are the baseline's diagnostics; the
CogFormer arm's for the same datasets are `finetuned/lba/<case>/`.

## ppc/ — LBA on empirical data

The finetuned checkpoint applied to the empirical lexical-decision data, with the
`condition_only` design (intercept + speed/accuracy condition). Lexicality needs no
regressor under stimulus coding: it reaches the model as `favored_idx`.

| folder | s_v | s_tau | what it is |
|---|---|---|---|
| `ppc/free/` | estimated | estimated | the model's own fit, simulated as-is |
| `ppc/conventional/` | 1.0 | 0.0 | the standard LBA identifiability constraint |

Both read the same draws; `posteriors.npz` is stored once, at `ppc/`. Which variant
belongs in the paper is an open modelling decision, which is why both are kept.

`ppc/validation/` checks how well this design identifies the parameters: recovery,
coverage, calibration on data SIMULATED with the real subjects' designs (condition
coding, trial counts, per-trial word/nonword stimulus), true parameters from the
training prior. Same script and figures as experiment-2's `<family>/validation/`.

## Excluded

{n_skipped} files outside the allowlist (posterior pairplots and similar).

## Naming

Family/scenario/arm infixes (`_fm_`, `_ft_`, `_mid1_`) are stripped — the path already
carries them. `_global` is renamed `_pi`. The zero-shot arm's `A<fixed_A>/` and
`mid_rdm/` levels are flattened.
""")


if __name__ == "__main__":
    main()
