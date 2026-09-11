"""Snapshot the 200-epoch BayesFlow baselines into finals/experiment-1a/e200.

WHY A SIBLING TREE RATHER THAN amortization_gap/.

``experiment-1a/<fam>/<case>/`` is internally consistent: its ``bf_*`` figures, the
paired ``cf_*`` figures and the C2ST files (``orig_vs_bf.csv``, ``summary.csv``) all
describe the SAME 2000-epoch BayesFlow checkpoints. Dropping the 200-epoch ``bf_*`` in
there would leave the C2ST numbers describing a comparison that no longer matches the
figures beside them, and would mix budgets across cases -- only intercept_only and
regressed have e200 runs, so the other three would stay at 2000 epochs. That is exactly
the "figures from the wrong checkpoint" failure the finals tree exists to prevent, so
this run gets its own directory and its own manifest.

It reuses finalize_experiment_1's ``kept`` allowlist and ``tidy`` (_global -> _pi)
rather than restating them, so the two snapshots cannot drift in what they keep or how
they name the global-parameter panel.

Usage: python -m experiments.finalize_experiment_1a_e200 [--dry_run]
"""
import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# kept/tidy/sha256/git_commit still live in finalize_experiment_1 (they serve the
# whole finals tree); the BF source paths moved out with the gap itself.
from experiments.finalize_experiment_1 import git_commit, kept, sha256, tidy
from experiments.finalize_experiment_1a import BF_EVAL, BF_FIG

TAG = "e200"
DEST_ROOT = Path("outputs/finals/experiment-1a") / TAG
# Only what the e200 run actually covers. Deliberately NOT finalize_experiment_1's
# C2ST_CASES: naming the three untrained cases here would produce a tree that looks
# half-empty rather than one that is complete for what was run.
FAMILIES = ["ddm", "cdm"]
CASES = ["intercept_only", "regressed"]


def plan() -> tuple[list[tuple[Path, Path]], list[Path]]:
    jobs: list[tuple[Path, Path]] = []
    skipped: list[Path] = []
    for fam in FAMILIES:
        for case in CASES:
            stem = f"{case}__{TAG}"
            dest_dir = DEST_ROOT / fam / case

            bdir = BF_FIG / fam / stem
            for src in sorted(bdir.glob("*")) if bdir.is_dir() else []:
                rest = src.name.replace(f"{fam}_family_{stem}_bf_", "")
                if not kept(rest):
                    skipped.append(src)
                    continue
                jobs.append((src, dest_dir / f"bf_{tidy(rest)}"))

            jobs.append((BF_EVAL / f"{fam}_families_bf_{stem}_evaluations.csv",
                         dest_dir / "bf_evaluations.csv"))
    return jobs, skipped


README = f"""# amortization_gap_{TAG}

BayesFlow per-family baselines retrained at **200 epochs x batch 256** ({TAG}), under
the 2026-09-10 pipeline changes:

- `standardize_summaries` OFF -- inference_variables only, matching the predecessor
  project. The A/B measured standardizing the summary channel as a small regression
  (calibration error 0.166 vs 0.119 at 100 epochs).
- `--time_power_law_alpha 0.0` -- uniform flow-matching time. bayesflow 2.0.13 changed
  this default from 0.0 to 0.5, so a bare `FlowMatching()` no longer matches what the
  predecessor trained.
- `model_family_bf_validate` pins `ModelFamily(num_workers=1)`, so `seed_case` actually
  pins the validation set (its thread pool previously made draws non-reproducible).
- The global parameter `pi` is reported on its own axes: `bf_*_pi.pdf` / `bf_metrics_pi.csv`.
  The BF arm was the only one missing these; the `cf_*` arm already had them.

## Why this is not in `amortization_gap/`

`amortization_gap/` is a coherent snapshot: its `bf_*`, `cf_*` and C2ST files all
describe the 2000-epoch checkpoints. These figures come from different checkpoints and
cover only 2 of the 5 cases, so merging them would leave `orig_vs_bf.csv` describing a
comparison that no longer matches the figures beside it. **There is no C2ST arm here** --
that would need the paired CogFormer validate re-run against these npz files.

## Contents

`{{ddm,cdm}}/{{intercept_only,regressed}}/` with `bf_recovery`, `bf_coverage`, `bf_ecdf`,
`bf_metrics` (+ `_pi` companions) and `bf_evaluations.csv`.

## Headline

200 epochs beats 2000 on every case and metric, but remains far from the predecessor:

| run | calib err | contraction |
|---|---|---|
| ddm/intercept_only 200ep | 0.259 | 0.967 |
| ddm/intercept_only 2000ep | 0.422 | 0.996 |
| cdm/intercept_only 200ep | 0.214 | 0.973 |
| cdm/intercept_only 2000ep | 0.437 | 0.997 |
| ddm/regressed 200ep | 0.378 | 0.989 |
| cdm/regressed 200ep | 0.348 | 0.991 |
| predecessor (all four) | 0.011-0.027 | 0.848-0.922 |

Posteriors are centred but too narrow: empirical coverage of a nominal 95% interval is
0.61-0.65 on intercept_only and 0.32-0.38 on regressed. See MANIFEST.csv for the source
path and mtime of every file.
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    jobs, skipped = plan()
    present = [(s, d) for s, d in jobs if s.exists()]
    missing = [s for s, _ in jobs if not s.exists()]

    print(f"{len(present)} files to place into {DEST_ROOT}\n")
    for fam in FAMILIES:
        for case in CASES:
            n = sum(1 for _, d in present if d.parent == DEST_ROOT / fam / case)
            print(f"  {fam}/{case:16} {n:3d} files")
    if skipped:
        print(f"\n  excluded by allowlist: {len(skipped)}")
        for x in skipped[:6]:
            print(f"    {x.name}")
    if missing:
        print(f"\n  [missing sources] {len(missing)}")
        for x in missing:
            print(f"    {x}")

    if args.dry_run:
        print("\ndry run; nothing written")
        return

    rows = []
    for src, dest in present:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        st = src.stat()
        rows.append({
            "destination": str(dest.relative_to(DEST_ROOT)),
            "source": str(src),
            "bytes": st.st_size,
            "sha256": sha256(src),
            "source_mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc)
                            .strftime("%Y-%m-%d %H:%M:%SZ"),
            "renamed_global_to_pi": "_pi" in dest.name and "_global" in src.name,
        })

    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("destination").to_csv(
        DEST_ROOT / "MANIFEST.csv", index=False)
    (DEST_ROOT / "README.md").write_text(
        README + f"\nSnapshot git commit: `{git_commit()}`\n")
    print(f"\nwrote {len(rows)} files + MANIFEST.csv + README.md to {DEST_ROOT}")


if __name__ == "__main__":
    main()
