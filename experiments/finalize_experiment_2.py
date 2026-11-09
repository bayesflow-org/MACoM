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

    python -m experiments.finalize_experiment_2 --dry_run
    python -m experiments.finalize_experiment_2
"""
from experiments.paths import RESULTS_ROOT
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


def plan() -> tuple[list[tuple[Path, Path]], list[Path]]:
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
    if refit_post.exists():
        jobs.append((refit_post, FINALS / "posteriors" / "ddm_sv_fixed.npz"))

    # Constrained variants. No posteriors.npz for the simulation-only ones: each reads
    # the same draws as its family's canonical set, so a second copy would imply a
    # second fit where there is none.
    for vdir, (fam, dest) in PPC_VARIANTS.items():
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
    args = p.parse_args()

    jobs, excluded = plan()
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
            if vfam == fam:
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
            "source": str(src),
            "bytes": st.st_size,
            "source_mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc)
                            .strftime("%Y-%m-%d %H:%M:%SZ"),
            "renamed_ppc_to_rt_ppc": dest.name.startswith("rt_ppc_"),
        })
    pd.DataFrame(rows).sort_values("destination").to_csv(FINALS / "MANIFEST.csv",
                                                         index=False)
    write_readme(rows, excluded)
    print(f"\nwrote {len(rows)} files -> {FINALS}")


def write_readme(rows, excluded):
    n = len(rows)
    exc = "\n".join(f"- `{e}`" for e in excluded) or "- none"
    (FINALS / "README.md").write_text(f"""# Experiment 2 — real data (posterior predictive checks)

Frozen snapshot, organised by model family. Assembled
{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} by
`experiments/finalize_experiment_2.py` — {n} files. Git commit `{git_commit()}`.
Per-file sources and original mtimes: `MANIFEST.csv`.

## Contents

Each family folder holds, per grouping variable:

| file | what it checks |
|---|---|
| `rt_ppc_<cond>.pdf` | RT distribution, observed vs posterior predictive |
| `choice_ppc_<cond>.pdf` | response dimension: proportions (DDM/RDM) or distribution (CDM) |
| `rt_quantiles_<cond>.pdf` | RT quantiles with predictive intervals |
| `rt_qq_<cond>.pdf` | RT quantile–quantile |
| `choice_quantiles_<cond>.pdf` | **CDM only** — see below |
| `combined_ppc.pdf` | every grouping variable in ONE figure — see below |
| `main_figure.tex`, `main_figure_details.tex`, `captions.tex` (root) | main-text figure (root `combined_ppc.pdf`) + short caption, its methods/appendix paragraph, and the appendix RT / CDM response-angle figures (generated by `python -m experiments.experiment_2.appendix_captions`; no manifest row) |

At the tree ROOT, `combined_ppc.pdf` puts all three families in a single figure:
DDM and RDM take one row each (four RT-quantile panels plus their 2x2 choice block),
CDM takes two (RT quantiles, then response-angle quantiles). Dashed rules separate
anything within a family; solid rules separate the families, the one boundary across
which nothing is comparable.

`posteriors/{{ddm,rdm,cdm}}.npz` holds the posterior draws every figure was simulated
from, so the plots can be regenerated without re-running inference.
`posteriors/ddm_sv_fixed.npz` is the separate fit behind `ddm/sv_fixed/`; the
simulation-only variants have no npz of their own because they reuse their family's.

### combined_ppc.pdf

The family's whole check on one page, for the per-condition figures above are the same
data split across files. Condition groups are separated by a dashed vertical divider.

- **DDM / RDM** — one row: an RT-quantile panel per level of each grouping variable,
  then a solid divider and the choice-proportion check as a 2x2 of per-level panels
  folded into the footprint ONE quantile panel would take. The solid rule marks a
  change of MEASURE, not of grouping. Inside the 2x2 the grouping variables are the
  rows, separated by a dashed rule; the four share a y scale, and only the top-left
  panel carries a legend.
- **CDM** — two rows of five: RT quantiles on top, response-angle quantiles beneath,
  difficulty then speed across.

No titles: each panel is labelled only by its condition level, and one legend on the
first panel covers every series in the figure.

Grouping variables: DDM `stim_type`/`truth`, RDM `congruency`/`session`,
CDM `difficulty`/`speed`.

## Constrained variants

A family folder's ROOT is always the model's own fit, simulated as-is. Subfolders hold
PPCs run from the **same posterior draws** with something pinned during the forward
simulation. They are not separate fits, which is why there is deliberately no second
`posteriors/*.npz` for them.

| folder | pinned | produced by |
|---|---|---|
| folder | `s_v` | kind |
|---|---|---|
| `ddm/sv0/` | 0 at simulation | constrained **PPC** (`--fix_natural s_v=0.0`) |
| `ddm/sv_fixed/` | 0 at fitting | constrained **REFIT** (`--fix_s_v`) |
| `cdm/sv0/` | 0 at simulation | constrained PPC (`--fix_natural s_v=0.0`) |

### A constrained PPC is not a constrained fit

`ddm/sv0/` and `ddm/sv_fixed/` both set `s_v` to zero, and they are NOT the same thing.

- **`sv0/`** overrides at SIMULATION time. The posterior draws still come from a fit
  where `s_v` was free, so `v`, `a` and `tau` carry the compensation they acquired
  while it was estimated. This answers "what does this fit predict with drift
  variability switched off".
- **`sv_fixed/`** is a genuine REFIT. `s_v` is dropped from the design, so it is never
  an estimation target and the simulator injects 0.0 directly, bypassing the link.
  Every other parameter is re-estimated under that constraint. This is the one that
  matches the original specification (Stump et al., 2026).

`s_v` is a substantial component -- natural-scale median 0.456 against a median |v| of
1.243, a ratio of 0.37 -- so neither is a formality. Measured over RT quantiles
0.1/0.3/0.5/0.7/0.9, both conditions, 72 subjects:

| run | mean abs quantile error | predictive SD (truth=True) |
|---|---|---|
| observed | — | 1.173 |
| canonical | 0.0869 | 1.503 |
| `sv0/` (simulation-only) | 0.0695 | 1.581 |
| `sv_fixed/` (refit) | 0.0784 | 1.533 |

Read the aggregate with care: it is sensitive to which quantiles you score. On a coarse
0.1/0.5/0.9 grid the canonical run wins; adding 0.3 and 0.7 reverses it, because the
canonical fit's largest miss sits at q0.7 (-0.146). What is STABLE across grids is the
shape, not the ranking:

- `sv0/` fits the BODY better but overshoots the right tail badly (q0.9: +0.112 and
  +0.161 against observed), and is the most over-dispersed of the three. Switching off
  drift variability without re-estimating leaves `v`, `a` and `tau` compensating for a
  component that is no longer there.
- `sv_fixed/` tracks the canonical fit closely everywhere and slightly improves the
  right tail (q0.9 +0.019 vs -0.049 on truth=True). Re-estimating under the constraint
  re-absorbs what `s_v` was doing, so the predictive barely moves. **That similarity is
  the result**: for this dataset, dropping across-trial drift variability properly costs
  almost nothing.

All three under-predict RT through the body and are over-dispersed relative to observed
(SD ~1.5 against 1.17), which is a pre-existing property of the lapse model, not
something these variants introduce.

`cdm/sv0/` asks the simulation-time question of CDM;

Note that `sv_fixed/`'s PPC must still pass `--fix_natural s_v=0.0`: a parameter absent
from the design has its coefficients masked to zero, and the link is still applied --
`shifted_softplus(0) = 0.693`, not 0, because that function never implemented the
-log(2) shift its name implies.

`cdm/sv0/` asks the same simulation-time question of CDM.

## Why choice quantiles are CDM-only

Not an omission. CDM's response is a continuous angle, so quantiles of theta are
meaningful the way RT quantiles are. DDM and RDM respond with a binary choice, where
quantiles carry no information — for those the response-dimension check IS the
proportion comparison in `choice_ppc_*`. `plot_choice_quantiles` in
`experiments/experiment_2/ppc.py` gates on this explicitly and logs a skip.

## Naming

The source calls the RT-distribution plot just `ppc_`, which says nothing about which
dimension it checks when it sits beside `choice_ppc_`. Renamed to `rt_ppc_` here; the
`_mc_` infix is dropped since the whole tree is model-class. `MANIFEST.csv` records
every rename.

## What is included

An allowlist of plot types -- `rt_ppc`, `choice_ppc`, `rt_quantiles`, `rt_qq`,
`choice_quantiles` -- not "whatever was in the source directory", so the tree's
contents do not depend on which flags a given PPC run happened to pass.

## Excluded

Model-FAMILY checkpoint figures and their posteriors — a different lineage from
everything else here (and one that does not exist for RDM at all, since
`RDM_MF_CKPT` was never retrained after the 2026-07-09 factorisation). The mf
posteriors are additionally stale, dated 2026-06-02:

{exc}

## Caveats

- **RDM predates the Stroop data.** Every RDM figure here was produced before the
  Stroop dataset landed and will need regenerating once that analysis is settled.
- **These figures come from uncommitted code.** As of this snapshot
  `experiments/experiment_2/ppc.py` carries an uncommitted restyle: per-family
  identity colours from `cogformer.utils.plot_utils.FAMILY_PALETTES` (DDM
  magenta, RDM purple-blue, CDM green), one `PPC_CI` interval level replacing a
  mix of 90% and 95% bands, condition names routed through `CONDITION_DISPLAY`,
  and titles no longer carrying the `(mc)` variant tag. The git commit recorded
  above is therefore the parent of the code that drew these, not that code
  itself. A clean checkout would reproduce the same NUMBERS -- the posterior
  draws in `posteriors/` are unchanged -- but not the same appearance.
- Adjacent analyses NOT included here: `ppc/r4c4_fulldraws/` (contamination
  sensitivity, contam vs nocontam per family, 20 files) and the DDM-only
  `ppc/ddm_combined/` and `ppc/ddm_original/` sets.
""")


if __name__ == "__main__":
    main()
