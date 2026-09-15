"""Recovery and calibration for ONE scenario, zero-shot and finetuned on one grid.

The per-arm recovery and ECDF figures already exist, but they are separate files, so
comparing them means holding two pages side by side and trusting that each cell's axes
match. They do not have to: recovery limits are set per figure from that arm's own
spread, so the zero-shot panel is drawn on a wider frame than the finetuned one and the
same displacement looks smaller in it. Overlaying puts both arms in one cell, on one
frame, against one confidence band.

Both arms come from the SEEDED trajectory lineage -- epoch 0 is the zero-shot condition
(pre-finetune checkpoint, LBA's embedding row warm-started from RDM's) and epoch 2000
is the finetuned checkpoint -- and were scored on byte-identical datasets, which this
script asserts rather than assumes. The finals' separate zeroshot/ and finetuned/ arms
are from an earlier unseeded run on a DIFFERENT validation set and cannot be overlaid
with these: paired cells would be comparing different data.

Replots from the saved posterior draws. CPU only, no checkpoint, no GPU.

    python -m experiments.experiment_3.overlay_arms --scenario interaction
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cogformer.diagnostics.plot.adaptive_recovery import adaptive_recovery
from cogformer.diagnostics.plot.adaptive_ecdf import adaptive_ecdf
from cogformer.utils.plot_utils import family_colors

from ..configs import LBA_CONFIG
from ..utils import add_global_params

GLOBAL_PARAMS = ["pi"]
GLOBAL_VARIABLE_NAMES = [r"$\pi$"]
TRAJ = Path("outputs/data/model_class/cf/lba_trajectory")

# Finetuned takes LBA's identity colour; zero-shot is neutral grey. Two stops of the
# same orange ramp were tried first and are not separable when the two clouds overlap
# in a cell -- which is most of them. Grey also says the right thing: at epoch 0 the
# checkpoint has no LBA identity, it is running LBA through RDM's embedding row.
C_ZERO, C_FT = "#8C8C96", family_colors("lba")["main_effect"]
ARM_EPOCH = {"zero-shot": 0, "finetuned": 2000}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scenario", type=str, default="interaction")
    p.add_argument("--root", type=str, default=str(TRAJ))
    p.add_argument("--outdir", type=str,
                   default="outputs/figures/model_class/cf/lba_trajectory")
    p.add_argument("--max_num_categories", type=int, default=5)
    p.add_argument("--max_num_regressors", type=int, default=7)
    return p.parse_args()


def load(root, epoch, scenario):
    f = Path(root) / f"ep{epoch}" / "predictions" / f"lba_{scenario}_ft_pred.npz"
    if not f.exists():
        raise SystemExit(f"missing {f}")
    d = np.load(f)
    return d["true_set"], d["pred_set"], d["params_mask"]


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    a = parse_args()
    cfg = add_global_params(LBA_CONFIG, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
    scen = {name: dc for name, dc, _ in cfg["val_scenarios"]}
    if a.scenario not in scen:
        raise SystemExit(f"{a.scenario!r} not in {sorted(scen)}")
    design_config = scen[a.scenario]

    true0, pred0, mask0 = load(a.root, ARM_EPOCH["zero-shot"], a.scenario)
    true1, pred1, mask1 = load(a.root, ARM_EPOCH["finetuned"], a.scenario)
    # The whole point of overlaying is that a cell compares like with like. If the two
    # arms were scored on different datasets the panels would still draw, and would be
    # meaningless, so this is an assertion rather than a note in the docstring.
    if not np.array_equal(true0, true1):
        raise SystemExit("arms were scored on DIFFERENT datasets; cannot overlay")
    if not np.array_equal(mask0, mask1):
        raise SystemExit("arms have different parameter masks; cannot overlay")
    logging.info(f"{a.scenario}: {true0.shape[0]} datasets x {pred0.shape[1]} draws, "
                 f"{int(mask0.sum())} active cells")

    arms = [(pred0, C_ZERO, "zero-shot"), (pred1, C_FT, "finetuned")]
    common = dict(true=true0, pred=pred0, design_config=design_config,
                  intrinsic_params=cfg["intrinsic_params"],
                  max_num_categories=a.max_num_categories,
                  max_num_regressors=a.max_num_regressors,
                  parameter_mask=mask0, variable_names=cfg["variable_names"],
                  global_params=GLOBAL_PARAMS, arms=arms)

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, fn, extra in (("recovery", adaptive_recovery, {}),
                            ("ecdf", adaptive_ecdf, dict(difference=True))):
        fig = fn(**common, **extra)
        if fig is None:
            logging.info(f"[skip] {name}: no active columns")
            continue
        out = outdir / f"lba_{a.scenario}_overlay_{name}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"[saved] {out}")


if __name__ == "__main__":
    main()
