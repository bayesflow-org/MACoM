"""Print a timing summary from ./outputs/timing/*.json.

Reads all JSON files written by RunTimer and prints a formatted table with
elapsed times and extrapolated full-run estimates for dry-run entries.

Usage:
    python -m experiments.timing_report
    python -m experiments.timing_report --timing_dir ./outputs/timing/
"""
import argparse
import json
from pathlib import Path


def _fmt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:3d}h {m:02d}m {s:02d}s"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timing_dir", default="./outputs/timing",
                        help="Directory containing timing JSON files. (default: ./outputs/timing)")
    args = parser.parse_args()

    timing_dir = Path(args.timing_dir)
    paths = sorted(timing_dir.glob("*.json"))
    if not paths:
        print(f"No timing records found in {timing_dir}")
        return

    records = []
    for p in paths:
        try:
            records.append(json.loads(p.read_text()))
        except Exception as e:
            print(f"[warn] could not read {p}: {e}")

    records.sort(key=lambda r: r.get("start_utc", ""))

    col_name = 52
    col_elapsed = 13
    col_ext = 14
    col_type = 8
    header = (
        f"{'Experiment':<{col_name}}"
        f"{'Elapsed':>{col_elapsed}}"
        f"{'Extrapolated':>{col_ext}}"
        f"{'Type':>{col_type}}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    total_elapsed = 0.0
    total_full = 0.0  # elapsed for full runs + extrapolated for dry runs
    has_extrapolated = False

    for r in records:
        name = r["name"][:col_name]
        elapsed = r["elapsed_seconds"]
        is_dry = r.get("dry_run", False)
        ext = r.get("extrapolated_seconds")

        elapsed_str = _fmt(elapsed)
        ext_str = _fmt(ext) if ext is not None else "             —"
        type_str = "dry" if is_dry else "full"

        print(f"{name:<{col_name}}{elapsed_str:>{col_elapsed}}{ext_str:>{col_ext}}{type_str:>{col_type}}")
        total_elapsed += elapsed

        if ext is not None:
            total_full += ext
            has_extrapolated = True
        else:
            total_full += elapsed

    print(sep)
    ext_label = _fmt(total_full) if has_extrapolated else "             —"
    print(f"{'TOTAL':<{col_name}}{_fmt(total_elapsed):>{col_elapsed}}{ext_label:>{col_ext}}")

    if has_extrapolated:
        total_days = total_full / 86400
        print(f"\nEstimated total wall time: {total_days:.2f} days  ({_fmt(total_full)})")

        # Hint: suggest DRY_EPOCHS for a ~24 h dry run
        dry_records = [r for r in records if r.get("dry_run") and r.get("dry_run_epochs")]
        if dry_records:
            n_train_scripts = 34  # total training invocations in dry_run.sh
            sample = dry_records[0]
            secs_per_epoch = sample["elapsed_seconds"] / sample["dry_run_epochs"]
            target_secs = 86400  # 24 h
            suggested = max(1, int(target_secs / (n_train_scripts * secs_per_epoch)))
            print(f"\n  Hint: to fill ~24 h, set DRY_EPOCHS ≈ {suggested}"
                  f"  (based on {secs_per_epoch:.1f} s/epoch from '{sample['name']}')")
    else:
        total_days = total_elapsed / 86400
        print(f"\nTotal wall time: {total_days:.2f} days  ({_fmt(total_elapsed)})")


if __name__ == "__main__":
    main()
