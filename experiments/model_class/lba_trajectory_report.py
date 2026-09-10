"""Aggregate and plot the LBA-finetuning trajectory.

Reads the per-epoch validation output written by run_lba_traj_validate.sh and answers
the question experiment 3 previously asserted rather than showed: how do LBA's gain and
the three trained families' forgetting develop OVER finetuning, not just between its
endpoints.

    python -m experiments.model_class.lba_trajectory_report

Each epoch directory holds one `{family}_{scenario}_ft_metrics.csv` per validation case,
indexed by parameter with columns NRMSE / Posterior Contraction / Calibration Error.
Aggregation is a two-stage mean: over parameters within a case, then over cases within a
family. Doing it in one pass would weight a family's scenarios by how many active
parameter cells they happen to have, so a scenario with more free parameters would
silently count for more.

Every epoch was validated on IDENTICAL datasets (seed_case with a fixed --seed), so
differences along a line are the checkpoint changing, not the test set.
"""
import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from cogformer.utils.plot_utils import family_colors

ROOT = Path("outputs/data/model_class/cf/lba_trajectory")
EPOCHS = [0, 500, 1000, 1500, 2000]
FAMILIES = ["lba", "ddm", "rdm", "cdm"]
METRICS = ["NRMSE", "Posterior Contraction", "Calibration Error"]
# Contraction runs the OPPOSITE way to the other two -- higher is better -- so the
# three panels cannot be read with one convention. Stated on each panel rather than
# left to the caption, because the LBA line rises in one panel and falls in the other
# two and that looks like a contradiction until you know which way is up.
BETTER = {"NRMSE": "lower is better",
          "Posterior Contraction": "higher is better",
          "Calibration Error": "lower is better"}

# LBA has no identity colour of its own in FAMILY_PALETTES' sense here -- it is the
# family being ACQUIRED, while the other three are being retained -- so it takes the
# blue-teal ramp reserved for it and is drawn heavier than the three it is tested
# against for forgetting.
LINE = {
    "lba": dict(color=family_colors("lba")["main_effect"], lw=2.4, marker="o", ms=6, zorder=5),
    "ddm": dict(color=family_colors("ddm")["main_effect"], lw=1.5, marker="s", ms=4, zorder=3),
    "rdm": dict(color=family_colors("rdm")["main_effect"], lw=1.5, marker="^", ms=4, zorder=3),
    "cdm": dict(color=family_colors("cdm")["main_effect"], lw=1.5, marker="D", ms=4, zorder=3),
}
LABEL = {"lba": "LBA (acquired)", "ddm": "DDM", "rdm": "RDM", "cdm": "CDM"}


def collect(root: Path, epochs) -> pd.DataFrame:
    """Tidy frame: one row per (epoch, family, scenario, metric).

    rglob, not glob: the validate script writes each case into its own
    {family}/{scenario}/ subdirectory rather than flat into --outdir, so a
    non-recursive glob silently finds nothing and the run looks like it produced
    no metrics at all.
    """
    rows = []
    for ep in epochs:
        d = root / f"ep{ep}"
        if not d.is_dir():
            raise SystemExit(f"missing {d} -- run run_lba_traj_validate.sh first")
        for csv in sorted(d.rglob("*_ft_metrics.csv")):
            m = re.match(r"(?P<fam>[a-z]+)_(?P<scen>.+)_ft_metrics\.csv$", csv.name)
            if not m or m.group("fam") not in FAMILIES:
                continue
            df = pd.read_csv(csv, index_col=0)
            for metric in METRICS:
                if metric not in df.columns:
                    continue
                # Mean over PARAMETERS within this case.
                rows.append({"epoch": ep, "family": m.group("fam"),
                             "scenario": m.group("scen"), "metric": metric,
                             "value": float(df[metric].mean())})
    if not rows:
        raise SystemExit(f"no *_ft_metrics.csv under {root}")
    return pd.DataFrame(rows)


def aggregate(tidy: pd.DataFrame) -> pd.DataFrame:
    """Mean over SCENARIOS within a family, with the spread across them."""
    g = tidy.groupby(["epoch", "family", "metric"])["value"]
    out = g.agg(mean="mean", sd="std", n="count").reset_index()
    # SE across scenarios: the shaded band is uncertainty in the family mean, not
    # across-parameter spread, which is already averaged out one level down.
    out["se"] = out["sd"] / np.sqrt(out["n"].clip(lower=1))
    return out


def plot(agg: pd.DataFrame, outpath: Path, epochs):
    fig, axes = plt.subplots(1, len(METRICS), figsize=(14.0, 5.2))
    handles, labels = [], []
    for ax, metric in zip(axes, METRICS):
        sub_m = agg[agg.metric == metric]
        for fam in FAMILIES:
            s = sub_m[sub_m.family == fam].sort_values("epoch")
            if s.empty:
                continue
            style = LINE[fam]
            (line,) = ax.plot(s.epoch, s["mean"], **style)
            if s["se"].notna().all():
                ax.fill_between(s.epoch, s["mean"] - s["se"], s["mean"] + s["se"],
                                color=style["color"], alpha=0.15, lw=0)
            if metric == METRICS[0]:
                handles.append(line); labels.append(LABEL[fam])
        ax.set_title(metric, fontsize=13, pad=24)
        ax.set_xlabel("Finetuning epoch", fontsize=12)
        ax.set_xticks(epochs)
        # Epoch 0 is not "no training": it is the pre-finetune model with LBA's
        # embedding warm-started from RDM, i.e. exactly the zero-shot condition
        # experiment 3 reports separately.
        ax.set_xticklabels(["0\nzero-shot"] + [str(e) for e in epochs[1:]], fontsize=10)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_box_aspect(1)
        # Between the axes and the title, not inside the panel: in NRMSE the three
        # retained families sit along the bottom edge, exactly where a corner
        # annotation lands.
        ax.text(0.5, 1.012, BETTER[metric], transform=ax.transAxes,
                ha="center", va="bottom", fontsize=10, color="0.35", style="italic")
    axes[0].set_ylabel("Mean over parameters and scenarios", fontsize=12)

    # Reserve enough for the two-line x tick labels AND the x-axis label before the
    # legend row; at 0.10 the legend landed on "Finetuning epoch".
    fig.tight_layout(rect=[0, 0.17, 1, 1])
    leg = fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                     frameon=False, fontsize=11.5, bbox_to_anchor=(0.5, 0.02))
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"[saved] {outpath}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=str(ROOT))
    p.add_argument("--epochs", type=int, nargs="+", default=EPOCHS,
                   help="Checkpoints to include. Pass a subset to plot a run that is "
                        "still in progress, e.g. --epochs 0 500 1000.")
    p.add_argument("--outdir", type=str,
                   default="outputs/figures/model_class/cf/lba_trajectory")
    a = p.parse_args()
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

    tidy = collect(Path(a.root), a.epochs)
    agg = aggregate(tidy)
    tidy.to_csv(outdir / "trajectory_per_scenario.csv", index=False, float_format="%.6f")
    agg.to_csv(outdir / "trajectory_summary.csv", index=False, float_format="%.6f")

    logging.info("\nmean over parameters and scenarios:")
    for metric in METRICS:
        piv = (agg[agg.metric == metric]
               .pivot(index="epoch", columns="family", values="mean")
               .reindex(columns=[f for f in FAMILIES if f in set(agg.family)]))
        logging.info(f"\n--- {metric} ---\n{piv.round(4).to_string()}")

    stem = "lba_finetuning_trajectory"
    if list(a.epochs) != EPOCHS:
        stem += "_ep" + "-".join(str(e) for e in a.epochs)
    plot(agg, outdir / f"{stem}.pdf", a.epochs)


if __name__ == "__main__":
    main()
