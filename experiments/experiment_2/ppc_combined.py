"""One figure per condition with THREE series: observed, ours, and the original.

    observed  neutral near-black       the data
    ours      DDM saturated (#EC008C)  CogFormer model-class PPC
    theirs    DDM dark     (#59315F)   the original authors' resimulations

Both predictive series sit on the DDM identity ramp and are told apart by
lightness and marker, not by hue. An earlier version drew "theirs" in #6969ff,
which is now RDM's identity colour -- on a DDM figure that reads as a second
model family rather than as a second method.

Ours is regenerated here via collect_ppc rather than read from disk, because
ppc.py computes its quantiles inside the plotting function and never persists
them. Their predictions are restricted to the subjects our loader keeps, so both
sets of quantiles are pooled over exactly the same trials.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import ppc as P
from .ppc import MODEL_REGISTRY, DEFAULT_RT_QUANTILES, _quantile_stats, OBS_COLOR
from cogformer.utils.plot_utils import family_colors

_DDM = family_colors("ddm")
OUR_COLOR = _DDM["main_effect"]
THEIR_COLOR = _DDM["intercept"]
COL = {"stim_type": "stim_type", "truth": "factual_truth"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="ddm")
    p.add_argument("--variant", type=str, default="mc")
    p.add_argument("--pred_csv", type=str, default="./data/pred_data_exp_2.csv")
    p.add_argument("--outdir", type=str, default="./outputs/figures/real_data/ppc/ddm_combined")
    p.add_argument("--max_num_regressors", type=int, default=4)
    p.add_argument("--max_num_categories", type=int, default=4)
    p.add_argument("--n_ppc_samples", type=int, default=None)
    p.add_argument("--rt_quantiles", type=float, nargs="+", default=list(DEFAULT_RT_QUANTILES))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    P.MAX_NUM_REGRESSORS = args.max_num_regressors
    P.MAX_NUM_CATEGORIES = args.max_num_categories
    cfg = MODEL_REGISTRY[args.model]
    qs = np.asarray(args.rt_quantiles)

    logging.info("regenerating our PPC (collect_ppc) ...")
    obs_pooled, ours_pooled, conditions, _, _ = P.collect_ppc(
        model_name=args.model, variant=args.variant,
        n_ppc_samples=args.n_ppc_samples, seed=args.seed)

    _, subject_ids = cfg["loader"](cfg["data_path"])
    keep = set(int(s) for s in subject_ids)
    pred = pd.read_csv(args.pred_csv)
    pred = pred[pred["id"].isin(keep)]
    logging.info(f"their predictions: {pred['resim_id'].nunique()} resims over "
                 f"{pred['id'].nunique()} matched subjects")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cond_name, _, label_map in conditions:
        labels = [l for l in label_map.values() if len(obs_pooled.get(cond_name, {}).get(l, []))]
        fig, axes = plt.subplots(1, len(labels), figsize=(3.4 * len(labels), 3.3), squeeze=False)
        x = np.arange(len(qs))
        for ax, lbl in zip(axes[0], labels):
            obs = obs_pooled[cond_name][lbl]
            val = [k for k, v in label_map.items() if v == lbl][0]
            theirs = [g["rt"].to_numpy() for _, g in
                      pred[pred[COL[cond_name]] == int(round(val + 0.5))].groupby("resim_id")]
            for series, samples, color, mark in (
                    ("ours", ours_pooled[cond_name][lbl], OUR_COLOR, "o"),
                    ("theirs", theirs, THEIR_COLOR, "^")):
                obs_q, med, lo, hi = _quantile_stats(obs, samples, qs)
                if med is None:
                    continue
                ax.fill_between(x, lo, hi, color=color, alpha=0.22, zorder=1)
                ax.plot(x, med, mark + "-", color=color, lw=1.4, ms=4, zorder=2,
                        label=("ours (CogFormer)" if series == "ours" else "original authors"))
                for j, q in enumerate(qs):
                    rows.append({"condition": cond_name, "level": lbl, "quantile": q,
                                 "observed": obs_q[j], "series": series, "median": med[j],
                                 "lo": lo[j], "hi": hi[j],
                                 "inside_95ci": bool(lo[j] <= obs_q[j] <= hi[j])})
            ax.plot(x, np.quantile(obs, qs), "s--", color=OBS_COLOR, lw=1.6, ms=5,
                    zorder=3, label="Observed")
            ax.set_xticks(x); ax.set_xticklabels([f"{q:g}" for q in qs])
            ax.set_title(lbl, fontsize=9)
            ax.set_xlabel("RT quantile", fontsize=8); ax.set_ylabel("RT (s)", fontsize=8)
            ax.tick_params(labelsize=7); ax.spines[["top", "right"]].set_visible(False)
        axes[0, 0].legend(fontsize=7, frameon=False)
        fig.suptitle(f"{args.model.upper()} — RT quantiles by {cond_name}: "
                     f"observed vs ours vs original", fontsize=10)
        fig.tight_layout()
        sp = outdir / f"{args.model}_{args.variant}_combined_rt_quantiles_{cond_name}.pdf"
        fig.savefig(sp, bbox_inches="tight"); plt.close(fig)
        logging.info(f"[saved] {sp}")

    df = pd.DataFrame(rows)
    csv = outdir / f"{args.model}_{args.variant}_combined_rt_quantiles.csv"
    df.to_csv(csv, index=False, float_format="%.4f")
    logging.info(f"[saved] {csv}")
    hit = df.groupby("series")["inside_95ci"].agg(["sum", "count"])
    logging.info("\nquantiles with the observed value inside the 95% CI:")
    for s, r in hit.iterrows():
        logging.info(f"  {s:8} {int(r['sum'])}/{int(r['count'])}")


if __name__ == "__main__":
    main()
