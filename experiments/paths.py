"""Study paths shared by standalone scripts and the experiment launcher.

Relative STUDY_ROOT overrides are resolved from the repository, so changing the
working directory never selects different results. Empirical inputs remain in
the repository's data directory; generated datasets live in RESULTS_ROOT/data.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_study = Path(os.environ.get("STUDY_ROOT", "outputs")).expanduser()
RESULTS_ROOT = (_study if _study.is_absolute() else PROJECT_ROOT / _study).resolve()
INPUT_DATA_ROOT = PROJECT_ROOT / "data"


def repo_relative(path) -> str:
    """Path as recorded in finals manifests: repo-relative when inside the repository,
    so manifests do not change with the checkout location; absolute otherwise."""
    path = Path(path).absolute()
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)
