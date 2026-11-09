"""Read values from experiments/shared_config.sh, the single source of run settings.

The finalize assemblers need the same run identity the orchestrator used (MC_TAG,
MC_CKPT, ...). Copying those strings into Python is how finalize_experiment_1 ended up
pointing at r4c4_b256_e10000 after the recipe changed; sourcing the shell file keeps
the two from drifting.
"""
import functools
import os
import subprocess
from pathlib import Path

from experiments.paths import RESULTS_ROOT
_SH = Path(__file__).with_name("shared_config.sh")


@functools.lru_cache(maxsize=None)
def get(name: str) -> str:
    out = subprocess.run(
        ["bash", "-c", f'source "{_SH}" >/dev/null 2>&1; printf "%s" "${{{name}}}"'],
        env=dict(os.environ, STUDY_ROOT=str(RESULTS_ROOT)),
        capture_output=True, text=True, check=True).stdout
    if not out:
        raise KeyError(f"{name} is not set in {_SH}")
    return out
