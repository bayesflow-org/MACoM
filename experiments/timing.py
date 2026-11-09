"""Experiment timing utilities."""
from experiments.paths import RESULTS_ROOT

import json
import time
from datetime import datetime, timezone
from pathlib import Path


class RunTimer:
    """Context manager that times a training run and writes a JSON log.

    For dry runs, records the actual elapsed time alongside an extrapolated
    full-run estimate based on the epoch ratio.
    """

    def __init__(
        self,
        name: str,
        timing_dir: str = str(RESULTS_ROOT / "timing"),
        dry_run: bool = False,
        total_epochs: int | None = None,
        dry_run_epochs: int | None = None,
    ):
        self.name = name
        self.timing_dir = Path(timing_dir)
        self.dry_run = dry_run
        self.total_epochs = total_epochs
        self.dry_run_epochs = dry_run_epochs
        self._start: float | None = None
        self._wall_start: str = ""

    def __enter__(self):
        self._start = time.monotonic()
        self._wall_start = datetime.now(timezone.utc).isoformat()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.monotonic() - self._start
        record: dict = {
            "name": self.name,
            "start_utc": self._wall_start,
            "end_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed,
            "dry_run": self.dry_run,
        }
        if self.dry_run and self.total_epochs and self.dry_run_epochs:
            scale = self.total_epochs / self.dry_run_epochs
            record["dry_run_epochs"] = self.dry_run_epochs
            record["total_epochs"] = self.total_epochs
            record["extrapolated_seconds"] = elapsed * scale

        self.timing_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.timing_dir / f"{self.name}.json"
        out_path.write_text(json.dumps(record, indent=2))

        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        print(f"\n[timing] {self.name}: {elapsed:.1f}s ({h}h {m:02d}m {s:02d}s)", flush=True)
        if "extrapolated_seconds" in record:
            ext = record["extrapolated_seconds"]
            ext_h = int(ext // 3600)
            ext_m = int((ext % 3600) // 60)
            print(f"[timing] {self.name}: extrapolated full run ≈ {ext_h}h {ext_m:02d}m", flush=True)

        return False
