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
import argparse
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

FINALS = Path("outputs/finals/experiment-2")
SRC = Path("outputs/figures/real_data/ppc")
DATA = Path("outputs/data/real_data")
FAMILIES = ["ddm", "rdm", "cdm"]


# Plot types this snapshot contains. An ALLOWLIST rather than "copy the directory",
# for the same reason as the other two assemblers: what a run emits depends on which
# flags it was given, and the tree's contents should not.
#
# Matched as PREFIXES because these names carry a condition suffix
# (`rt_quantiles_difficulty`). Order-safe: "choice_ppc_speed" does not start with
# "ppc_", so the two never collide.
KEEP = ("ppc", "choice_ppc", "rt_quantiles", "rt_qq", "choice_quantiles")


def kept(rest: str) -> bool:
    return any(rest.startswith(k + "_") for k in KEEP)


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
    counts: dict[str, int] = {}
    for _, d in present:
        counts[d.parent.name] = counts.get(d.parent.name, 0) + 1
    total_gb = sum(s.stat().st_size for s, _ in present) / 1e9
    print(f"{len(present)} files to place ({total_gb:.2f} GB)")
    for k in FAMILIES + ["posteriors"]:
        print(f"  {k:11} {counts.get(k, 0):3d}")
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

`posteriors/{{ddm,rdm,cdm}}.npz` holds the posterior draws every figure was simulated
from, so the plots can be regenerated without re-running inference.

Grouping variables: DDM `stim_type`/`truth`, RDM `congruency`/`session`,
CDM `difficulty`/`speed`.

## Why choice quantiles are CDM-only

Not an omission. CDM's response is a continuous angle, so quantiles of theta are
meaningful the way RT quantiles are. DDM and RDM respond with a binary choice, where
quantiles carry no information — for those the response-dimension check IS the
proportion comparison in `choice_ppc_*`. `plot_choice_quantiles` in
`experiments/real_data/ppc.py` gates on this explicitly and logs a skip.

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
- **The CDM choice-quantile plots come from uncommitted code.** As of this snapshot
  `experiments/real_data/ppc.py` carries ~56 uncommitted lines adding
  `plot_choice_quantiles`, and the two `cdm/choice_quantiles_*.pdf` files were
  untracked. A clean checkout would not reproduce them.
- Adjacent analyses NOT included here: `ppc/r4c4_fulldraws/` (contamination
  sensitivity, contam vs nocontam per family, 20 files) and the DDM-only
  `ppc/ddm_combined/` and `ppc/ddm_original/` sets.
""")


if __name__ == "__main__":
    main()
