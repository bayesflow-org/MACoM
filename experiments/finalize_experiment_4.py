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

# FINAL SCOPE (2026-09-27): the 5-parameter vsvfixed aDDM (v and s_v pinned) and the
# 5-family pooled geometry. The other variants are out of the paper.
# variant -> (frozen data dir, frozen fig dir, baseline fig dir, compare key)
VARIANTS = {
    "vsvfixed": ("addm_vsvfixed/intercept_only", "addm_vsvfixed/intercept_only",
                 "addm_baseline_vsvfixed/intercept_only/item0_vsvfixed_e500",
                 "intercept_only_vsvfixed"),
}
SUPERSEDED = []
REP_DIRS = ["crossfamily", "intercept_only_vsvfixed"]
KEEP_EXT = {".pdf", ".csv", ".npz"}

# ALLOWLIST of destinations, relative to FINALS. The source trees also hold outputs
# that are deliberately NOT in the paper -- the per-family transfer score, the
# between-family CKA matrix, the cross-family RAW datasets (app3_raw_data.npz, kept in
# outputs/data only: hundreds of MB in a git-tracked tree) -- and a sweep would copy
# them in. Anything produced but not listed is reported as excluded; anything listed
# but not produced is reported as missing, so a partial run cannot pass as complete.
_PROBE = ["bf_metrics.csv", "calibration_ecdf.pdf", "coverage.pdf", "data.npz",
          "recovery.pdf", "z_score_contraction.pdf"]
KEEP_DEST = (
    {f"probe/vsvfixed/{arm}/{f}" for arm in ("frozen", "baseline") for f in _PROBE}
    | {f"probe/vsvfixed/compare/{f}" for f in (
        "c2st_identified.npz", "c2st_joint.npz", "c2st_summary.csv",
        "shared_validation.npz", "intercept_only_vsvfixed.pdf",
        "intercept_only_vsvfixed_panel.pdf", "intercept_only_vsvfixed_recovery.pdf")}
    | {"representation/crossfamily/app3_pooled_geometry.pdf",
       "representation/crossfamily/app3_summaries.npz"}
    | {f"representation/intercept_only_vsvfixed/{f}" for f in (
        "cka_vs_null.pdf", "gram_cka.pdf", "rdm_rsa.pdf", "rsa_vs_null.pdf",
        "summaries.npz")}
)


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

    kept_jobs = []
    for src, dest in jobs:
        (kept_jobs if str(dest.relative_to(FINALS)) in KEEP_DEST else skipped).append(
            (src, dest) if str(dest.relative_to(FINALS)) in KEEP_DEST else src)
    return kept_jobs, skipped


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
    print(f"\n  excluded (not on the allowlist): {len(skipped)}")
    missing = sorted(KEEP_DEST - {str(d.relative_to(FINALS)) for _, d in present})
    if missing:
        print(f"  MISSING ({len(missing)} of {len(KEEP_DEST)} allowlisted): "
              + ", ".join(missing[:6]) + (" ..." if len(missing) > 6 else ""))
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

Assembled {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} by
`experiments/finalize_experiment_4.py` — {n} files. Commit `{git_commit()}`.
`MANIFEST.csv` records per-file provenance.

## Scope

The **5-parameter aDDM** (`vsvfixed`: v and s_v pinned at 1.0375 and 0.3133, their
prior medians; free: v_diff, omega, a, tau, s_tau) and the **5-family pooled geometry**
(DDM, RDM, CDM trained; aDDM, LBA held out), before and after LBA finetuning.

## Layout

| folder | level | what |
|---|---|---|
| `probe/vsvfixed/frozen` | posterior | frozen model-class encoder + FlowMatching head |
| `probe/vsvfixed/baseline` | posterior | learned SetTransformer summary + FlowMatching |
| `probe/vsvfixed/compare` | posterior | PAIRED shared-validation C2ST, log-density, CRPS; combined recovery and panel |
| `representation/intercept_only_vsvfixed` | representation | frozen vs learned summary: RSA (crossnobis) + CKA against a dimensionality-matched null |
| `representation/crossfamily` | representation | pooled MDS geometry of all five families, original vs LBA-finetuned encoder |

## Reading the numbers

- **Always against the null.** A random 128->16 projection already scores high RSA, so
  an unqualified RSA value is not evidence of shared structure.
- **C2ST: read `excess`** over the empirical draw-split null, not raw accuracy.
- The two probe arms are compared on ONE shared validation set
  (`compare/shared_validation.npz`); their separate `frozen/` and `baseline/` runs
  simulate their own datasets and are not directly comparable.

## Data

`representation/*/summaries.npz` / `app3_summaries.npz` hold the encoder summaries behind
the figures. The RAW simulated datasets they were computed from are NOT here (hundreds of
MB): they stay in `outputs/data/encoder_rep/representation/<case>/` as `raw_data.npz`
(aDDM) and `app3_raw_data.npz` (cross-family), and `--raw_data <file>` on
`addm_representation.py` / `crossfamily_representation.py` rebuilds the figures on the
identical datasets. They must be saved, because `--seed` does not fix them (the numba
kernels draw from their own RNG).

{n_skipped} produced files are not on the allowlist and are excluded.
""")


if __name__ == "__main__":
    main()
