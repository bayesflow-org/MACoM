"""Typography for every figure in this project.

Applied from ``cogformer/__init__.py`` rather than left in ``plot_utils``. The
rcParams have to be set before anything draws, and "whichever module happens to
import plot_utils" is not that guarantee: of the 59 experiment scripts that import
matplotlib, a dozen never touch plot_utils, and ``ddm_design_eig`` is one of them --
its figure came out in the old face while every figure around it had changed, which
is exactly the silent mismatch this is supposed to prevent.

Every one of those scripts imports *something* from ``cogformer``, so the package
root is the one hook that covers them all. The cost is importing matplotlib whenever
cogformer is imported, which is noise beside torch.
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

# The font is VENDORED under cogformer/utils/fonts and registered at import,
# not installed on the machine: a system font would make the figures depend on
# whoever set that machine up. The TTFs are not in git, though -- run
# ./fetch_fonts.sh to pull the pinned release. Absent them, everything still
# draws in matplotlib's default sans and says so ONCE, because the failure mode
# worth guarding against is not a crash but a silent one: figures that differ
# from outputs/finals in a way nobody notices until the paper is typeset.
#
# MATH STAYS COMPUTER MODERN. Text is Inter; anything inside $...$ -- the theta
# and pi in the CDM response-angle labels -- renders in matplotlib's built-in
# "cm" fontset, which is the face LaTeX uses by default, so the symbols match a
# LaTeX document instead of matching the surrounding sans text. mathtext.fontset
# is independent of font.family, so the two can differ on purpose like this.
_FONT_DIR = Path(__file__).resolve().parent / "fonts"


def _register_vendored_fonts():
    """Add the vendored faces to matplotlib's font list. Idempotent."""
    if not _FONT_DIR.is_dir():
        return []
    known = {f.name for f in font_manager.fontManager.ttflist}
    added = []
    for ttf in sorted(_FONT_DIR.glob("*.ttf")):
        try:
            name = font_manager.FontProperties(fname=str(ttf)).get_name()
        except Exception:
            continue
        if name not in known or not any(
                Path(f.fname) == ttf for f in font_manager.fontManager.ttflist):
            font_manager.fontManager.addfont(str(ttf))
            added.append(ttf.name)
    return added


_register_vendored_fonts()
INTER_AVAILABLE = any(f.name == "Inter" for f in font_manager.fontManager.ttflist)

if INTER_AVAILABLE:
    # Inter FIRST, the previous default kept behind it: a machine missing the
    # vendored files then degrades to the old look instead of erroring.
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = (
        ["Inter"] + [f for f in plt.rcParams["font.sans-serif"] if f != "Inter"])
    plt.rcParams["mathtext.fontset"] = "cm"
else:
    logging.warning(
        "Inter not found in %s -- figures will use matplotlib's default sans "
        "and will NOT match outputs/finals. Run ./fetch_fonts.sh to install it.",
        _FONT_DIR)


