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
from cogformer.diagnostics.plot.adaptive_delta import adaptive_delta
from cogformer.utils.plot_utils import cogformer_vi_colors, cogformer_fm_colors

from ..configs import LBA_CONFIG
from ..utils import add_global_params

GLOBAL_PARAMS = ["pi"]
GLOBAL_VARIABLE_NAMES = [r"$\pi$"]
TRAJ = Path("outputs/data/model_class/cf/lba_trajectory")

# Zero-shot on the purple/blue ramp, finetuned on the magenta one. Taken from the
# ramps directly rather than through family_colors: these are ARMS of one family, not
# families, and naming them "rdm"/"ddm" in code would read as a claim about which
# family each arm is. Two stops of LBA's own orange ramp were tried first and are not
# separable where the two clouds overlap, which is most cells.
#
# NOTE these two hues are RDM's and DDM's identity colours elsewhere in the project.
# Nothing else appears in this figure, so there is no ambiguity inside it, but a
# caption should not lean on colour to mean a family here.
C_ZERO, C_FT = cogformer_vi_colors()["main_effect"], cogformer_fm_colors()["main_effect"]
ARM_EPOCH = {"zero-shot": 0, "finetuned": 2000}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scenario", type=str, default="interaction")
    p.add_argument("--root", type=str, default=str(TRAJ))
    p.add_argument("--outdir", type=str,
                   default="outputs/figures/model_class/cf/lba_trajectory")
    # BOTH matter, and both were wrong here before. max_num_categories sets the
    # block width, so 5 laid the design out on 4-wide blocks when the run used 3,
    # putting each active cell under the wrong regressor's name. And
    # max_num_regressors must be FORWARDED to the label builder: without it,
    # expand_design_config_to_mask invents phantom main effects (u_3, u_4, u_5...) to
    # cover rows it cannot account for, so the interaction block u_1:u_2 is labelled
    # as a main effect "u_5". 4 is the training geometry (MC_MAX_REG/MC_MAX_CAT).
    p.add_argument("--max_num_categories", type=int, default=4)
    p.add_argument("--max_num_regressors", type=int, default=4)
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

    # Guard on the CELL COUNT, not on the labels. The run's own metrics.csv is the
    # oracle for which cells exist, and this checks the layout does not drop any --
    # but its label strings are NOT used as a reference, because the run wrote them
    # without max_num_regressors and so recorded the interaction block under a
    # phantom main-effect name. Matching those would preserve the error.
    ref = (Path(a.root) / f"ep{ARM_EPOCH['finetuned']}" / "lba" / a.scenario
           / f"lba_{a.scenario}_ft_metrics.csv")
    if ref.exists():
        import pandas as pd
        n_ref = len(pd.read_csv(ref, index_col=0))
        gcols = {i for i, p_ in enumerate(cfg["intrinsic_params"])
                 if p_ in GLOBAL_PARAMS}
        n_here = sum(1 for r in range(mask0.shape[0]) for c in range(mask0.shape[1])
                     if mask0[r, c] == 1.0 and c not in gcols)
        if n_here != n_ref:
            raise SystemExit(f"layout mismatch: {n_here} non-global active cells "
                             f"but {ref.name} records {n_ref}")
        logging.info(f"  layout check: {n_here} non-global cells, matches {ref.name}")

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

    # The same two arms as a per-cell CHANGE, in the adaptive-metrics layout. The
    # overlays above show both arms and leave the comparison to the eye; this states
    # it as a number per cell, which is what the scenario-level means average away.
    delta = adaptive_delta(
        true=true0, pred_a=pred0, pred_b=pred1, design_config=design_config,
        intrinsic_params=cfg["intrinsic_params"],
        max_num_categories=a.max_num_categories,
        max_num_regressors=a.max_num_regressors, parameter_mask=mask0,
        variable_names=cfg["variable_names"], global_params=GLOBAL_PARAMS,
        arm_names=("zero-shot", "finetuned"))
    if delta is None:
        logging.info("[skip] delta: no active columns")
    else:
        out = outdir / f"lba_{a.scenario}_delta_metrics.pdf"
        delta.savefig(out, bbox_inches="tight")
        plt.close(delta)
        logging.info(f"[saved] {out}")


if __name__ == "__main__":
    main()
