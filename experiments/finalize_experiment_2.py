"""Assemble outputs/finals/experiment-2 -- the real-data posterior predictive checks.

Mirrors experiments/finalize_experiment_1.py: a frozen, single-provenance tree that
supersedes the scattered source directories, with a manifest recording where every
file came from.

SCOPE
-----
The model-CLASS (mc) posterior predictive set only, organised by model family. The
`*_mf_*` figures under the same source directories come from the separate
model-FAMILY checkpoints -- a different lineage, and one that does not exist for RDM
at all -- so mixing them in would reintroduce exactly the provenance ambiguity this
tree exists to remove. They are listed in the README as excluded rather than
silently dropped.

The default snapshot contains the standard fits used by the overall combined PPC.
Constrained sensitivity analyses are opt-in with --include_constrained.

    python -m experiments.finalize_experiment_2 --dry_run
    python -m experiments.finalize_experiment_2
"""
from experiments.paths import RESULTS_ROOT, repo_relative
import argparse
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from experiments import shared_config
from experiments.finalize_experiment_1 import kept as val_kept, tidy as val_tidy

FINALS = (RESULTS_ROOT / "finals/experiment-2")
SRC = (RESULTS_ROOT / "figures/real_data/ppc")
DATA = (RESULTS_ROOT / "data/real_data")
FAMILIES = ["ddm", "rdm", "cdm"]

# Real-data-DESIGN validation (realdata_validate.py): simulated data on each family's
# empirical designs. Filed next to the fit it validates -- <fam>/validation/ for the
# canonical fit, ddm/sv_fixed/validation/ for the refit -- so each folder pairs a real
# fit with the check of how well that design identifies it.
_MC_TAG = shared_config.get("MC_TAG")
VAL_FIG = (RESULTS_ROOT / "figures/model_class/cf") / f"{_MC_TAG}_realdata"
VAL_PRED = (RESULTS_ROOT / "data/model_class") / f"{_MC_TAG}_realdata"
# (family, validation case) -> destination under FINALS
VAL_CASES = {("ddm", "realdata"): "ddm/validation",
             ("ddm", "realdata_sv_fixed"): "ddm/sv_fixed/validation",
             ("rdm", "realdata"): "rdm/validation",
             ("cdm", "realdata"): "cdm/validation"}

# Constrained PPC variants: same posterior draws as the family's canonical set,
# differing only in what the forward SIMULATION holds fixed. They land in a labelled
# subfolder of the family rather than beside it, so the family root always means "the
# model's own fit, simulated as-is" -- the same rule experiment-3's ppc/free/ follows.
#
# source dir -> (family, destination subfolder)
PPC_VARIANTS = {
    "cdm_sv0":      ("cdm", "sv0"),
    "ddm_sv0":      ("ddm", "sv0"),
    "ddm_sv_fixed": ("ddm", "sv_fixed"),
}


# Plot types this snapshot contains. An ALLOWLIST rather than "copy the directory",
# for the same reason as the other two assemblers: what a run emits depends on which
# flags it was given, and the tree's contents should not.
#
# Matched as PREFIXES because these names carry a condition suffix
# (`rt_quantiles_difficulty`). Order-safe: "choice_ppc_speed" does not start with
# "ppc_", so the two never collide.
KEEP = ("ppc", "choice_ppc", "rt_quantiles", "rt_qq", "choice_quantiles",
        "combined_ppc")


def kept(rest: str) -> bool:
    """Prefix match, plus an exact one for types that carry NO condition suffix.

    `combined_ppc` holds every condition in one figure, so unlike the five per-condition
    types its name ends at the type. A prefix-only test would silently drop it.
    """
    stem = rest[:-4] if rest.endswith(".pdf") else rest
    return any(stem == k or stem.startswith(k + "_") for k in KEEP)


def tidy(rest: str) -> str:
    """Rename the bare `ppc_` plot to `rt_ppc_`.

    In the source the RT-distribution plot is just `ppc_`, which is ambiguous sitting
    next to `choice_ppc_` -- nothing in the name says which dimension it checks. The
    other four types already say so, so this makes the set self-describing.
    """
    return f"rt_{rest}" if rest.startswith("ppc_") else rest


def plan(include_constrained: bool = False) -> tuple[list[tuple[Path, Path]], list[Path]]:
    jobs, excluded = [], []
    skipped: list[Path] = []

    # All three families in one figure, at the TREE ROOT rather than under a family
    # folder -- it belongs to none of them. Produced by experiments.experiment_2
    # .combined_all from the same cached draws the per-family figures use.
    all_fams = SRC / "combined_ppc.pdf"
    if all_fams.exists():
        jobs.append((all_fams, FINALS / "combined_ppc.pdf"))

    for fam in FAMILIES:
        d = SRC / fam
        if not d.is_dir():
            continue
        for src in sorted(d.glob("*.pdf")):
            if f"{fam}_mf_" in src.name:      # model-family checkpoint: different lineage
                excluded.append(src)
                continue
            rest = src.name.replace(f"{fam}_mc_", "")
            if not kept(rest):
                skipped.append(src)
                continue
            jobs.append((src, FINALS / fam / tidy(rest)))

        # Posterior draws the PPC figures were simulated from. Only the mc/ set: the
        # mf/ siblings are 2026-06-02 and stale, matching the *_mf_* figures excluded
        # above. The Aug-19 posteriors predate the Aug-26 figures, which is the right
        # order -- PPC reads posteriors and simulates forward.
        post = DATA / fam / "mc" / f"{fam}_mc_posteriors.npz"
        if post.exists():
            jobs.append((post, FINALS / "posteriors" / f"{fam}.npz"))
        for stale in [DATA / fam / "mf"]:
            if stale.is_dir():
                excluded.extend(sorted(stale.glob("*.npz")))

    # A constrained REFIT is a genuinely different fit, so its posterior draws are a
    # distinct object and DO belong here -- unlike the simulation-only variants below,
    # which share their family's draws.
    refit_post = DATA / "ddm" / "mc_sv_fixed" / "ddm_mc_posteriors.npz"
    if include_constrained and refit_post.exists():
        jobs.append((refit_post, FINALS / "posteriors" / "ddm_sv_fixed.npz"))

    # Constrained variants. No posteriors.npz for the simulation-only ones: each reads
    # the same draws as its family's canonical set, so a second copy would imply a
    # second fit where there is none.
    for vdir, (fam, dest) in PPC_VARIANTS.items():
        if not include_constrained:
            continue
        d = SRC / vdir
        if not d.is_dir():
            continue
        for src in sorted(d.glob("*.pdf")):
            if f"{fam}_mf_" in src.name:      # model-family lineage, as above
                excluded.append(src)
                continue
            rest = src.name.replace(f"{fam}_mc_", "")
            if not kept(rest):
                skipped.append(src)
                continue
            jobs.append((src, FINALS / fam / dest / tidy(rest)))
    # Real-data-design validation: experiment 1's plot allowlist and naming, plus the
    # draws (true_set / pred_set) so the figures can be redrawn without re-running.
    for (fam, case), dest in VAL_CASES.items():
        if case == "realdata_sv_fixed" and not include_constrained:
            continue
        cdir = VAL_FIG / fam / case
        for src in sorted(cdir.glob("*")) if cdir.is_dir() else []:
            rest = src.name.replace(f"{fam}_{case}_fm_", "")
            if not val_kept(rest):
                skipped.append(src); continue
            jobs.append((src, FINALS / dest / val_tidy(rest)))
        jobs.append((VAL_PRED / f"{fam}_{case}_cf_pred.npz", FINALS / dest / "predictions.npz"))
    return jobs, excluded + skipped


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--include_constrained", action="store_true",
                   help="Also assemble the s_v=0 sensitivity plots and DDM constrained refit.")
    args = p.parse_args()

    jobs, excluded = plan(include_constrained=args.include_constrained)
    present = [(s, d) for s, d in jobs if s.exists()]
    # Keyed by the destination's parent PATH, not its name: `ddm/sv0` and `cdm/sv0`
    # share the basename `sv0`, so keying on the name silently merged the two into a
    # single count while both were copied correctly -- the breakdown disagreed with
    # the total.
    counts: dict[str, int] = {}
    for _, d in present:
        key = str(d.parent.relative_to(FINALS))
        counts[key] = counts.get(key, 0) + 1
    total_gb = sum(s.stat().st_size for s, _ in present) / 1e9
    print(f"{len(present)} files to place ({total_gb:.2f} GB)")
    # Variant subfolders are keyed by their own directory name, not the family, so
    # they must be listed explicitly or they vanish from the summary while still
    # being copied -- the count and the total would silently disagree.
    labels = []
    for fam in FAMILIES:
        labels.append((fam, fam))
        for _v, (vfam, dest) in PPC_VARIANTS.items():
            if vfam == fam and args.include_constrained:
                labels.append((f"  {fam}/{dest}", f"{fam}/{dest}"))
    labels.append(("posteriors", "posteriors"))
    for label, key in labels:
        print(f"  {label:15} {counts.get(key, 0):3d}")
    print(f"\n  excluded (model-family checkpoint lineage): {len(excluded)}")
    for e in excluded:
        print(f"    {e}")

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
            "renamed_ppc_to_rt_ppc": dest.name.startswith("rt_ppc_"),
        })
    pd.DataFrame(rows).sort_values("destination").to_csv(FINALS / "MANIFEST.csv",
                                                         index=False)
    write_readme(rows, excluded, include_constrained=args.include_constrained)
    print(f"\nwrote {len(rows)} files -> {FINALS}")


def write_readme(rows, excluded, include_constrained=False):
    exc = "\n".join(f"- `{e}`" for e in excluded) or "- none"
    constrained = (
        "Constrained variants are included by request: `ddm/sv0/` and `cdm/sv0/` "
        "reuse the standard posteriors with `s_v=0` during simulation. "
        "`ddm/sv_fixed/` is a separate fit with `s_v` fixed at fitting, paired with "
        "`posteriors/ddm_sv_fixed.npz` and its own validation. None feeds the root figure."
        if include_constrained else
        "Constrained variants are excluded from the default final snapshot. The "
        "archived snapshot is in `../../archive/experiment-2/constrained/`, with "
        "its original provenance and analysis notes. Source predictions and PPC "
        "caches remain available. To assemble those analyses again, use "
        "`python -m experiments.finalize_experiment_2 --include_constrained`."
    )
    (FINALS / "README.md").write_text(f"""# Experiment 2 — real data (posterior predictive checks)

Organised by model family. Metadata refreshed
{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} by
`experiments/finalize_experiment_2.py` — {len(rows)} recorded result files.
Git commit `{git_commit()}` identifies the metadata code; original source paths and
figure mtimes are recorded separately in `MANIFEST.csv`.

## Where the overall combined PPC comes from

The root `combined_ppc.pdf` uses the **standard CogFormer model-class fits** for
DDM, RDM and CDM, with contamination included and no simulation-time parameter
constraints. It is generated by `experiments.experiment_2.combined_all`, which calls
`collect_ppc_cached` with `variant=mc`, `n_ppc_samples=None` (all stored draws),
`seed=0`, `no_contamination=False`, `fixed_natural=None`, `posteriors_path=None`,
and the training geometry (4 regressors, 4 categories).

| family | empirical input | posterior source | retained copy | grouping variables |
|---|---|---|---|---|
| DDM | `data/data_application_ddm.csv` | `outputs/data/real_data/ddm/mc/ddm_mc_posteriors.npz` | `posteriors/ddm.npz` | stimulus type, factual truth |
| RDM | `data/data_application_rdm.csv` | `outputs/data/real_data/rdm/mc/rdm_mc_posteriors.npz` | `posteriors/rdm.npz` | congruency, session |
| CDM | `data/data_application_cdm.csv` | `outputs/data/real_data/cdm/mc/cdm_mc_posteriors.npz` | `posteriors/cdm.npz` | difficulty, speed instruction |

Paths in the input/source columns are relative to the repository for this study.
The standard per-family PPCs and the overall figure use the same cached predictive
draws in `outputs/data/real_data/ppc_cache/`. Every draw simulates each subject's own
trials and design, then pools simulated trials over subjects. The root figure is
copied from `outputs/figures/real_data/ppc/combined_ppc.pdf`. Its paper layout has DDM
and RDM in the first two rows, CDM RT quantiles in the third and response-angle
quantiles in the fourth. `PROVENANCE.json` records the audit of the retained figure.

## Retained results

| file | what it checks |
|---|---|
| `combined_ppc.pdf` (root) | overall PPC for all three standard fits |
| `<family>/combined_ppc.pdf` | the same family's check on one page |
| `<family>/rt_ppc_<cond>.pdf` | observed and predictive RT distributions |
| `<family>/choice_ppc_<cond>.pdf` | choice proportions (DDM/RDM), response-angle distribution (CDM) |
| `<family>/rt_quantiles_<cond>.pdf` | observed and predictive RT quantiles |
| `<family>/rt_qq_<cond>.pdf` | RT quantile–quantile comparison |
| `cdm/choice_quantiles_<cond>.pdf` | continuous response-angle quantiles |
| `<family>/validation/` | simulation-based validation on that family's empirical design, with saved predictions |
| `posteriors/` | the three standard posterior-draw files |
| `main_figure.tex`, `main_figure_details.tex`, `captions.tex` | paper figure and appendix definitions |

The appendix uses per-condition RT distributions for all three families and CDM
response-angle distributions. Those complementary checks remain alongside the
combined figure. RDM is the three-alternative Stroop analysis. CDM angles are doubled
from the recorded half-angle to the model's response-angle scale.

## Optional constrained analyses

{constrained}

## Assembly rules

Only model-class (`mc`) PPC results enter this snapshot. Separate model-family
checkpoints (`mf`) are excluded. The plot allowlist retains `ppc`, `choice_ppc`,
`rt_quantiles`, `rt_qq`, `choice_quantiles`, and `combined_ppc`. The source `ppc_`
prefix becomes `rt_ppc_`, and `_mc_` is dropped; the manifest records the mapping.
LBA empirical PPCs belong to Experiment 3.

Other excluded source files found during assembly:

{exc}
""")

if __name__ == "__main__":
    main()
