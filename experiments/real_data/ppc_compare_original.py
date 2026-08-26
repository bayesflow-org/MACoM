"""Quantile / Q-Q plots for the ORIGINAL authors' posterior predictive (DDM).

Reads their resimulated predictions and renders the same two figures ppc.py
produces for ours, so the pair can be read side by side:

    ours    magenta-red  #e94560   (ppc.py default)
    theirs  blue-purple  #6969ff

Their file carries 81 subjects; our loader keeps 72, so their predictions are
RESTRICTED to the 72 ids we actually fit. Pooling over different subject sets
would make the two sets of quantiles incomparable for reasons that have nothing
to do with either model.

Condition coding is identical in both files -- data_utils.load_ddm_data does
``u_1 = stim_type - 0.5`` and ``u_2 = factual_truth - 0.5`` -- so stim_type 0 is
Novel / 1 Familiar and factual_truth 0 is False / 1 True in each.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from .ppc import (MODEL_REGISTRY, plot_rt_quantiles, plot_rt_qq,
                  DEFAULT_RT_QUANTILES, _quantile_stats)

THEIR_COLOR = "#6969ff"
COL = {"stim_type": "stim_type", "truth": "factual_truth"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_csv", type=str, default="./data/pred_data_exp_2.csv")
    p.add_argument("--model", type=str, default="ddm")
    p.add_argument("--outdir", type=str,
                   default="./outputs/figures/real_data/ppc/ddm_original")
    p.add_argument("--rt_quantiles", type=float, nargs="+",
                   default=list(DEFAULT_RT_QUANTILES))
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    cfg = MODEL_REGISTRY[args.model]

    subject_data, subject_ids = cfg["loader"](cfg["data_path"])
    keep = set(int(s) for s in subject_ids)
    logging.info(f"observed: {len(keep)} subjects, "
                 f"{sum(len(subject_data[s]['rts']) for s in subject_ids)} trials")

    pred = pd.read_csv(args.pred_csv)
    n_before = pred["id"].nunique()
    pred = pred[pred["id"].isin(keep)]
    logging.info(f"their predictions: {n_before} subjects -> {pred['id'].nunique()} after "
                 f"matching ours; {pred['resim_id'].nunique()} resimulations")

    obs_pooled, ppc_pooled = {}, {}
    for cond_name, cond_col, label_map in cfg["conditions"]:
        obs_pooled[cond_name], ppc_pooled[cond_name] = {}, {}
        their_col = COL[cond_name]
        for val, lbl in label_map.items():
            # ours: u_1/u_2 are effect-coded +-0.5; theirs: 0/1. val + 0.5 bridges them.
            obs = [subject_data[s][cond_col] == val for s in subject_ids]
            obs_pooled[cond_name][lbl] = np.concatenate(
                [subject_data[s]["rts"][m] for s, m in zip(subject_ids, obs) if m.any()])
            sub = pred[pred[their_col] == int(round(val + 0.5))]
            ppc_pooled[cond_name][lbl] = [
                g["rt"].to_numpy() for _, g in sub.groupby("resim_id") if len(g)]
            logging.info(f"  {cond_name}/{lbl}: observed {len(obs_pooled[cond_name][lbl])} "
                         f"trials; theirs {len(ppc_pooled[cond_name][lbl])} resims x "
                         f"~{int(np.median([len(a) for a in ppc_pooled[cond_name][lbl]]))} trials")

    outdir = Path(args.outdir)
    plot_rt_quantiles(obs_pooled, ppc_pooled, cfg["conditions"], args.model, "original",
                      outdir, quantiles=args.rt_quantiles, ppc_color=THEIR_COLOR)
    plot_rt_qq(obs_pooled, ppc_pooled, cfg["conditions"], args.model, "original",
               outdir, quantiles=args.rt_quantiles, ppc_color=THEIR_COLOR)

    qs = np.asarray(args.rt_quantiles)
    for cond_name, _, label_map in cfg["conditions"]:
        for lbl in label_map.values():
            o = obs_pooled[cond_name][lbl]
            obs_q, med, lo, hi = _quantile_stats(o, ppc_pooled[cond_name][lbl], qs)
            logging.info(f"\n  {cond_name} / {lbl}   (their predictions)")
            logging.info(f"  {'q':>5}{'observed':>10}{'theirs':>10}{'95% CI':>20}{'ok?':>6}")
            for j, q in enumerate(qs):
                ok = "yes" if lo[j] <= obs_q[j] <= hi[j] else "NO"
                logging.info(f"  {q:5.2f}{obs_q[j]:10.3f}{med[j]:10.3f}"
                             f"   [{lo[j]:7.3f},{hi[j]:7.3f}]{ok:>6}")


if __name__ == "__main__":
    main()
