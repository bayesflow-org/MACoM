"""LBA recovery for ONE design, zero-shot and fine-tuned overlaid on one grid.

The per-arm recovery figures already exist (zeroshot/ and finetuned/), but as separate
files their frames differ: each sets its axis limits from its own spread, so the
zero-shot panel is drawn on a wider frame and the same displacement looks smaller in
it. Overlaying puts both arms in one cell, on one frame.

Reads the FINAL run's predictions -- the same files model_class_lba_compare differences
-- and refuses to draw unless both arms scored byte-identical datasets with the same
parameter mask. (An earlier overlay_arms.py read the seeded trajectory lineage, a
different validation set; it was removed with that lineage.) The row layout
(max_num_categories) is taken from the npz, which the validators save for replots, so
the overlay's rows match the single-arm figures'.

Replots from saved posterior draws -- CPU only, no checkpoint:

    python -m experiments.experiment_3.overlay_recovery \\
        --zero_shot_dir <LBA_PAIRED>/zeroshot/A0.5 --finetuned_dir <LBA_PAIRED>/finetuned \\
        --outdir <LBA_PAIRED>/compare --scenario interaction
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cogformer.diagnostics.plot.adaptive_recovery import adaptive_recovery
from cogformer.utils.plot_utils import family_colors

from ..configs import LBA_CONFIG
from ..utils import add_global_params

GLOBAL_PARAMS = ["pi"]
GLOBAL_VARIABLE_NAMES = [r"$\pi$"]
# Fine-tuned takes LBA's identity colour; zero-shot is neutral grey. Two stops of one
# ramp are not separable where the clouds overlap, which is most cells. Grey also says
# the right thing: zero-shot, the network has no LBA identity and runs it through the
# RDM's embedding.
C_ZERO, C_FT = "#D9D9D9", family_colors("lba")["main_effect"]
ARM_LABELS = ("zero-shot", "fine-tuned")
# Drawn at the plotting function's default 3 in per panel (18 in wide for six
# columns), so \textwidth scales it by ~0.31: 23 / 26 pt land at ~7.0 / 8.0 pt.
LABEL_FS, TITLE_FS = 23, 26


def load(path):
    if not path.exists():
        raise SystemExit(f"missing {path}")
    d = np.load(path)
    cats = int(d["max_num_categories"]) if "max_num_categories" in d.files else None
    return d["true_set"], d["pred_set"], d["params_mask"], cats


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--zero_shot_dir", type=str, required=True,
                   help="model_class_lba_cf_validate --outdir incl. A<fixed_A> (holds predictions/).")
    p.add_argument("--finetuned_dir", type=str, required=True,
                   help="model_class_lba_finetune_validate --outdir (holds predictions/).")
    p.add_argument("--outdir", type=str, required=True,
                   help="Writes <outdir>/lba/<scenario>_overlay_recovery.pdf.")
    p.add_argument("--scenario", type=str, default="interaction")
    p.add_argument("--max_num_regressors", type=int, default=4)
    p.add_argument("--max_num_categories", type=int, default=None,
                   help="Row layout; default: the value saved in the npz.")
    a = p.parse_args()

    cfg = add_global_params(LBA_CONFIG, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
    sc = a.scenario
    if sc not in cfg["benchmark_configs"]:
        raise SystemExit(f"{sc!r} not in {sorted(cfg['benchmark_configs'])}")
    t0, p0, m0, c0 = load(Path(a.zero_shot_dir) / "predictions" / f"lba_{sc}_mid1_cf_pred.npz")
    t1, p1, m1, c1 = load(Path(a.finetuned_dir) / "predictions" / f"lba_{sc}_ft_pred.npz")
    # Overlaying is only meaningful if a cell compares like with like; an unpaired pair
    # would still draw, so this is an error, not a note. Same test as
    # model_class_lba_compare._paired.
    if t0.shape != t1.shape or not np.allclose(t0, t1, atol=1e-6):
        raise SystemExit(f"{sc}: arms scored DIFFERENT datasets; cannot overlay")
    if not np.array_equal(m0, m1):
        raise SystemExit(f"{sc}: arms have different parameter masks; cannot overlay")
    cats = a.max_num_categories or c0 or c1
    if cats is None:
        raise SystemExit("no max_num_categories in either npz; pass --max_num_categories")
    if c0 and c1 and c0 != c1:
        raise SystemExit(f"arms saved different row layouts ({c0} vs {c1})")
    logging.info(f"{sc}: {t0.shape[0]} datasets x {p0.shape[1]} draws, "
                 f"{int(np.asarray(m0).sum())} active cells, max_num_categories={cats}")

    fig = adaptive_recovery(
        true=t0, pred=p0, design_config=cfg["benchmark_configs"][sc],
        intrinsic_params=cfg["intrinsic_params"], max_num_categories=cats,
        max_num_regressors=a.max_num_regressors, parameter_mask=m0,
        variable_names=cfg["variable_names"], global_params=GLOBAL_PARAMS,
        arms=[(p0, C_ZERO, ARM_LABELS[0]), (p1, C_FT, ARM_LABELS[1])],
        plain_row_labels=True, label_fontsize=LABEL_FS, title_fontsize=TITLE_FS)
    out = Path(a.outdir) / "lba" / f"{sc}_overlay_recovery.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"[saved] {out}")


if __name__ == "__main__":
    main()
