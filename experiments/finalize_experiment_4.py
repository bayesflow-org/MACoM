"""Assemble outputs/finals/experiment-4 — encoder representation.

Two levels of evidence, kept separate because they answer different questions:

  probe/          POSTERIOR level. What each arm DECODES: recovery, calibration,
                  coverage, metrics, and the paired shared-validation comparison
                  (C2ST vs an empirical null, log posterior density, CRPS).
  representation/ REPRESENTATION level. What each arm ENCODES: RSA with crossnobis
                  and CKA, both against a dimensionality-matched null. The frozen
                  summary is 128-d and the learned one 16-d, so no posterior metric
                  can compare them directly -- this is the only level at which the
                  universal-representation claim is testable.

Allowlist, not sweep, matching the other assemblers: contents are a decision rather
than a consequence of which flags a run happened to pass.

    python -m experiments.finalize_experiment_4 --dry_run
"""
import argparse
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

FINALS = Path("outputs/finals/experiment-4")
DATA = Path("outputs/data/encoder_rep")
FIG = Path("outputs/figures/encoder_rep")

# variant -> (frozen data dir, frozen fig dir, baseline fig dir, compare key)
VARIANTS = {
    "intercept_only_7d": ("addm/intercept_only", "addm/intercept_only",
                          "addm_baseline/intercept_only/item0_bd_7d_e500",
                          "intercept_only_7d"),
    "vfixed": ("addm_vfixed/intercept_only", "addm_vfixed/intercept_only",
               "addm_baseline_vfixed/intercept_only/item0_vfixed_e500",
               "intercept_only_vfixed"),
    "varying_values": ("addm/varying_values", "addm/varying_values",
                       "addm_baseline/varying_values/item0_varying_e500",
                       "varying_values"),
    # s_v pinned, v left FREE: the test of whether the drift-variance functional
    # s_v^2(1+omega^2) can stand in for pinning v. It cannot -- v recovery goes
    # 0.130 -> -0.005 (frozen) and 0.006 -> 0.091 (baseline), i.e. nothing either way.
    "svfixed": ("addm_svfixed/intercept_only", "addm_svfixed/intercept_only",
                "addm_baseline_svfixed/intercept_only/item0_svfixed_e500",
                "intercept_only_svfixed"),
    # Both pinned. The control for the above: it says whether pinning s_v sharpens
    # the parameters that were ALREADY identified, separately from whether it
    # rescues v. It does not -- every cell matches vfixed to within noise.
    "vsvfixed": ("addm_vsvfixed/intercept_only", "addm_vsvfixed/intercept_only",
                 "addm_baseline_vsvfixed/intercept_only/item0_vsvfixed_e500",
                 "intercept_only_vsvfixed"),
}
# Superseded exploratory baselines from the 469-dim / BayesFlow-defaults debugging.
# Only item0_bd_7d_e500 is the final 7-dim baseline arm.
SUPERSEDED = ["item0_bd_7d", "item0_best_defaults", "item0_best_defaults_a0",
              "item0_main_7d", "item0_main_do0"]
REP_DIRS = ["varying_values", "intercept_only_vfixed", "crossfamily", "lba",
            "intercept_only_svfixed", "intercept_only_vsvfixed"]
KEEP_EXT = {".pdf", ".csv", ".npz"}


def tidy(name, variant):
    """Strip run-specific prefixes; the path already carries variant and arm."""
    for pre in (f"addm_{variant}_", "addm_baseline_", "addm_", "lba_"):
        if name.startswith(pre):
            name = name[len(pre):]
    for pre in ("intercept_only_", "varying_values_", "encoder_rep_", "arm_compare_"):
        if name.startswith(pre):
            name = name[len(pre):]
    return name


def plan():
    jobs, skipped = [], []
    for var, (fdata, ffig, bfig, ckey) in VARIANTS.items():
        for src_dir, dest_sub in ((DATA / fdata, "frozen"), (FIG / ffig, "frozen"),
                                  (FIG / bfig, "baseline")):
            if not src_dir.is_dir():
                continue
            for src in sorted(src_dir.iterdir()):
                if src.suffix not in KEEP_EXT:
                    skipped.append(src); continue
                dest = FINALS / "probe" / var / dest_sub / tidy(src.name, var)
                jobs.append((src, dest))
        for src_dir in (DATA / "addm_compare" / ckey, FIG / "addm_compare" / ckey):
            if not src_dir.is_dir():
                continue
            for src in sorted(src_dir.iterdir()):
                if src.suffix not in KEEP_EXT:
                    skipped.append(src); continue
                jobs.append((src, FINALS / "probe" / var / "compare" / tidy(src.name, var)))

    for d in REP_DIRS:
        for root, sub in ((DATA / "representation" / d, None),
                          (FIG / "representation" / d, None)):
            if not root.is_dir():
                continue
            for src in sorted(root.iterdir()):
                if src.suffix not in KEEP_EXT:
                    skipped.append(src); continue
                jobs.append((src, FINALS / "representation" / d / tidy(src.name, d)))

    for name in SUPERSEDED:
        p = FIG / "addm_baseline" / "intercept_only" / name
        if p.is_dir():
            skipped.extend(sorted(p.iterdir()))
    return jobs, skipped


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    jobs, skipped = plan()
    present = [(s, d) for s, d in jobs if s.exists()]
    counts = {}
    for _, d in present:
        k = "/".join(d.relative_to(FINALS).parts[:2])
        counts[k] = counts.get(k, 0) + 1
    gb = sum(s.stat().st_size for s, _ in present) / 1e9
    print(f"{len(present)} files ({gb:.2f} GB)")
    for k in sorted(counts):
        print(f"  {k:34} {counts[k]:3d}")
    print(f"\n  excluded: {len(skipped)} "
          f"({len(SUPERSEDED)} superseded baseline runs + non-artifact files)")
    if a.dry_run:
        print("\ndry run; nothing written"); return

    rows = []
    for src, dest in present:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        st = src.stat()
        rows.append({"destination": str(dest.relative_to(FINALS)), "source": str(src),
                     "bytes": st.st_size,
                     "source_mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc)
                                     .strftime("%Y-%m-%d %H:%M:%SZ")})
    pd.DataFrame(rows).sort_values("destination").to_csv(FINALS / "MANIFEST.csv",
                                                        index=False)
    write_readme(len(rows), len(skipped))
    print(f"\nwrote {len(rows)} files -> {FINALS}")


def write_readme(n, n_skipped):
    (FINALS / "README.md").write_text(f"""# Experiment 4 — encoder representation

Frozen snapshot. Assembled {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}
by `experiments/finalize_experiment_4.py` — {n} files. Commit `{git_commit()}`.

## Layout

| folder | level | what |
|---|---|---|
| `probe/<variant>/frozen` | posterior | frozen CogFormer encoder + FlowMatching |
| `probe/<variant>/baseline` | posterior | learned SetTransformer + FlowMatching |
| `probe/<variant>/compare` | posterior | PAIRED shared-validation C2ST / log-density / CRPS |
| `representation/<case>` | representation | RSA (crossnobis) + CKA |

Variants: `intercept_only_7d` (original 7-cell target), `vfixed` (6 cells, v pinned),
`varying_values` (9 cells, trial-varying item values).

## Headline results

**The arms differ at BOTH levels.** Posteriors: C2ST excess over an empirical null of
+0.128 / +0.118 / +0.205 across the three variants (z = +25 to +38), learned summary
better on log-density and CRPS. Geometry: RSA 0.80-0.81 against a
dimensionality-matched null of 0.85-0.87 — at or BELOW the null. Not "same
representation, different decoder".

**Model families occupy orthogonal subspaces.** Between-family CKA is -0.011..0.000 for
every pair, including DDM-RDM which were trained together, against split-half noise
ceilings of 0.96-0.98. This argues against a single universal code.

**LBA finetuning did not change the representation.** original vs finetuned encoder on
SHARED datasets scores 0.98-0.996 on lba/ddm/rdm/cdm, at or above every noise ceiling —
including LBA itself. The -48.2% LBA error improvement came from the DECODER; the
zero-shot encoder already carried the information (weight drift: encoder 3.3%,
decoder 6.8%). This answers experiment 3's forgetting question at a level aggregate
NRMSE cannot reach.

**Held-out aDDM transfers fine.** encoder-vs-shared-parameter RSA: ddm 0.217, rdm 0.478,
cdm 0.377, aDDM 0.223 — above trained DDM, and the spread across trained families
dwarfs the trained/held-out gap.

## Reading the numbers

- **Always against the null.** A random 128->16 projection scores RSA ~0.87, so an
  unqualified "RSA 0.80" reads as strong evidence and is evidence of nothing.
- **Always with the ceiling.** Split-half scores (same conditions, re-simulated) run
  0.96-0.98. Without them a near-zero off-diagonal cannot be distinguished from a
  metric that simply fails across independent stimulus sets.
- **Joint regression R^2 is only 0.08-0.35**, so most of what the encoder represents is
  NOT parameter structure.
- Marginal per-parameter RSA is DILUTED (`a`: contraction 0.983, marginal RSA 0.166);
  the joint-regression betas are the unique contributions. Both are plotted.

## Excluded

{n_skipped} files: five superseded exploratory baseline runs under
`addm_baseline/intercept_only/` ({', '.join(SUPERSEDED)}) from the 469-dim and
BayesFlow-defaults debugging — only `item0_bd_7d_e500` is the final 7-dim arm — plus
non-artifact files.
""")


if __name__ == "__main__":
    main()
