"""LaTeX tables for the amortization gap (Experiment 1a).

Reads the two arms' per-case outputs on the shared bench5 test set and writes
outputs/finals/experiment-1a/amortization_gap_tables.tex with three tables:

  tab:absolute-metrics  NRMSE, NRMSE (BayesFlow convention), calibration error, contraction,
                        median +- SEM pooled over
                        the five bench5 cases and every active parameter in each case.
  tab:c2st              per-dataset joint / per-parameter C2ST and the pooled joint
                        (median +- SEM over the five cases).
  tab:c2st-cases        per-dataset C2ST per case, so the design dependence is visible.

The pooled per-parameter C2ST is deliberately NOT reported: pooled over 200 datasets a
single parameter's draws trace the aggregate posterior (~ the prior for any calibrated
approximator), so it reads ~0.52 however far apart the per-dataset posteriors are. See
model_family_c2st._dataset_task.

Both arms are read from files written by the SAME metric code
(cogformer.diagnostics.metric.adaptive_metrics, via compute_adaptive_metrics /
compute_global_metrics): <case>_fm_metrics.csv + _fm_metrics_global.csv for CogFormer,
<fam>_family_<case>_bf_metrics.csv + _bf_metrics_global.csv for the baseline. NOT the
baseline's evaluations.csv: that is BayesFlow's own root_mean_squared_error, which takes
the RMSE over each dataset's draws and then the median over datasets, where the repo's
takes the RMSE over datasets per draw and then the median over draws. On identical
draws the two differ by up to ~0.04, so mixing them (as this table did until
2026-10-01) understated the baseline's NRMSE and overstated the gap (CogFormer/baseline
1.86-2.02x instead of 1.33-1.58x). Contraction and calibration error agree exactly
between the two. The parameter counts are asserted equal.

SEM = sample standard deviation / sqrt(n).

    python -m experiments.experiment_1a.gap_tables
"""
from experiments.paths import RESULTS_ROOT
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import shared_config

FAMILIES = ["ddm", "rdm", "cdm"]
CASES = ["intercept_only", "fixed", "regressed", "fixed_regressed", "interaction"]
CASE_LABELS = {
    "intercept_only": "Intercept only",
    "fixed": "Fixed",
    "regressed": "Regressed",
    "fixed_regressed": "Fixed + regressed",
    "interaction": "Interaction",
}
METRICS = [("NRMSE", r"\textbf{NRMSE} ($\downarrow$)"),
           ("NRMSE_BF", r"\textbf{NRMSE}$^\dagger$ ($\downarrow$)"),
           ("Calibration Error", r"\textbf{Cal. Error} ($\downarrow$)"),
           ("Posterior Contraction", r"\textbf{Post. Contr.} ($\uparrow$)")]
CF_NAME = r"\textbf{CogFormer}-$\mathcal{C}$"
BF_NAME = r"\textbf{Baseline}"

BF_FIG = (RESULTS_ROOT / "figures/model_family/bf")
BF_DATA = (RESULTS_ROOT / "data/model_family")
CF_PRED = (RESULTS_ROOT / "data/model_class") / f"{shared_config.get('MC_TAG')}_validate"
CF_VAL = (RESULTS_ROOT / "figures/model_class/cf") / f"{shared_config.get('MC_TAG')}_validate"
C2ST = (RESULTS_ROOT / "figures/c2st")
OUT = (RESULTS_ROOT / "finals/experiment-1a/amortization_gap_tables.tex")


def sem(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def fmt(x) -> str:
    return rf"${np.median(x):.3f} \pm {sem(x):.3f}$"


def nrmse_bayesflow_convention(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """BayesFlow's NRMSE (root_mean_squared_error, normalize="range", median aggregation).

    pred (datasets, draws, params), true (datasets, params) -> (params,): the RMSE of each
    dataset's posterior draws against its truth, divided by the range of the true values,
    then the median over datasets. The repo's _rmse (the NRMSE column, and every figure)
    instead takes the RMSE over datasets per draw and the median over draws. Computed here
    with the same code for both arms, from their saved posteriors.
    """
    rmse = np.sqrt(np.mean((pred - true[:, None, :]) ** 2, axis=1))
    return np.median(rmse / (true.max(0) - true.min(0)), axis=0)


def bayesflow_nrmse(fam: str, case: str) -> dict[str, np.ndarray]:
    """{arm: BayesFlow-convention NRMSE per active parameter}. The parameter ORDER differs
    between the arms' files, which is harmless: the table pools them by median."""
    b = np.load(BF_DATA / f"{fam}_{case}_data.npz")
    act = b["param_masks"][0].astype(bool)
    c = np.load(CF_PRED / f"{fam}_{case}_cf_pred.npz")
    m = (c["params_mask"][0] if c["params_mask"].ndim == 3 else c["params_mask"]) == 1
    return {"bf": nrmse_bayesflow_convention(b["pred_set"][:, :, act], b["true_set"][:, act]),
            "cf": nrmse_bayesflow_convention(c["pred_set"][:, :, m], c["true_set"][:, m])}


def load_metrics(fam: str) -> dict[str, pd.DataFrame]:
    """{arm: DataFrame of per-parameter metrics pooled over the five cases}."""
    bf, cf = [], []
    for case in CASES:
        bd = BF_FIG / fam / case
        b = pd.concat([pd.read_csv(bd / f"{fam}_family_{case}_bf_metrics.csv", index_col=0),
                       pd.read_csv(bd / f"{fam}_family_{case}_bf_metrics_global.csv", index_col=0)])
        d = CF_VAL / fam / case
        c = pd.concat([pd.read_csv(d / f"{fam}_{case}_fm_metrics.csv", index_col=0),
                       pd.read_csv(d / f"{fam}_{case}_fm_metrics_global.csv", index_col=0)])
        if len(b) != len(c):
            raise SystemExit(f"{fam}/{case}: BF has {len(b)} parameters, CogFormer {len(c)}")
        conv = bayesflow_nrmse(fam, case)
        if not len(conv["bf"]) == len(conv["cf"]) == len(b):
            raise SystemExit(f"{fam}/{case}: posterior files disagree with the metric CSVs")
        b = b.assign(NRMSE_BF=np.sort(conv["bf"]))   # order-free: pooled by median below
        c = c.assign(NRMSE_BF=np.sort(conv["cf"]))
        bf.append(b)
        cf.append(c)
    return {"bf": pd.concat(bf), "cf": pd.concat(cf)}


def load_c2st(fam: str) -> pd.DataFrame:
    """One row per case: pooled joint, per-dataset joint and per-parameter C2ST + floors."""
    rows = []
    for case in CASES:
        f = C2ST / fam / case / f"{fam}_{case}_orig_vs_bf_c2st.csv"
        if not f.exists():
            raise SystemExit(f"missing {f} -- has e1a_c2st finished?")
        df = pd.read_csv(f, index_col=0)
        if "Per-dataset C2ST" not in df:
            raise SystemExit(f"{f} has no per-dataset arm -- re-run e1a_c2st (FORCE=1)")
        rows.append({
            "case": case,
            "pooled_joint": df.loc["_joint_c2st_", "C2ST Accuracy"],
            "joint": df.loc["_joint_c2st_", "Per-dataset C2ST"],
            "joint_floor": df.loc["_joint_c2st_", "Per-dataset null floor"],
            "cell": df.loc["_mean_accuracy_", "Per-dataset C2ST"],
            "cell_floor": df.loc["_mean_accuracy_", "Per-dataset null floor"],
        })
    return pd.DataFrame(rows).set_index("case")


def floor_range(c2st: dict, col: str) -> str:
    v = pd.concat([c2st[f][col] for f in FAMILIES])
    lo, hi = f"{v.min():.2f}", f"{v.max():.2f}"
    return lo if lo == hi else f"{lo}--{hi}"


def absolute_table(metrics: dict) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Absolute metric values (median $\pm$ SEM) aggregated across the five "
        r"benchmark design configurations and all parameters in each configuration, both "
        r"networks scored on the same test datasets with the same code. NRMSE: for each "
        r"posterior draw, the root mean squared error against the ground truth across "
        r"datasets, median over draws, divided by the range of the ground-truth values (the "
        r"definition used in all figures). NRMSE$^\dagger$: the BayesFlow convention, the "
        r"root mean squared error of each dataset's posterior draws, divided by the same "
        r"range, median over datasets; it reflects the typical dataset and is less "
        r"sensitive to poorly recovered ones.}",
        r"\label{tab:absolute-metrics}",
        r"\begin{tabularx}{\textwidth}{llCCCC}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Method} & " + " & ".join(h for _, h in METRICS) + r" \\",
    ]
    for fam in FAMILIES:
        lines.append(r"\midrule")
        for i, (arm, name) in enumerate((("bf", BF_NAME), ("cf", CF_NAME))):
            head = rf"\textbf{{{fam.upper()}}}" if i == 0 else ""
            cells = " & ".join(fmt(metrics[fam][arm][m]) for m, _ in METRICS)
            lines.append(rf"{head} & {name} & {cells} \\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    return "\n".join(lines)


def c2st_table(c2st: dict) -> str:
    fams = " & ".join(rf"\textbf{{{f.upper()}}}" for f in FAMILIES)
    row = lambda col: " & ".join(fmt(c2st[f][col]) for f in FAMILIES)
    return "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{C2ST accuracy of {CF_NAME} vs.\ {BF_NAME} (median $\pm$ SEM across the "
        r"five benchmark design configurations; 0.5 = indistinguishable). Per dataset: one "
        r"classifier per test dataset, averaged over 200 datasets; joint over all active "
        r"parameters (MLP), per parameter averaged over parameters (quadratic discriminant). "
        r"Draw-split null floors: " + floor_range(c2st, "joint_floor") + " (joint), "
        + floor_range(c2st, "cell_floor") + r" (per parameter). Pooled: one classifier over "
        r"draws from all datasets.}",
        r"\label{tab:c2st}",
        r"\begin{tabularx}{\textwidth}{lCCC}",
        r"\toprule",
        rf"\textbf{{C2ST}} & {fams} \\",
        r"\midrule",
        rf"Joint, per dataset & {row('joint')} \\",
        rf"Per parameter, per dataset & {row('cell')} \\",
        rf"Joint, pooled & {row('pooled_joint')} \\",
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
    ])


def c2st_case_table(c2st: dict) -> str:
    fams = " & ".join(rf"\multicolumn{{2}}{{c}}{{\textbf{{{f.upper()}}}}}" for f in FAMILIES)
    rules = " ".join(rf"\cmidrule(lr){{{2 + 2 * k}-{3 + 2 * k}}}" for k in range(len(FAMILIES)))
    sub = " & ".join(["Joint & Param."] * len(FAMILIES))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{Per-dataset C2ST accuracy of {CF_NAME} vs.\ {BF_NAME} for each benchmark "
        r"design configuration: joint over all active parameters, and averaged over "
        r"parameters. 0.5 = indistinguishable.}",
        r"\label{tab:c2st-cases}",
        r"\begin{tabularx}{\textwidth}{l" + "CC" * len(FAMILIES) + "}",
        r"\toprule",
        rf"\textbf{{Configuration}} & {fams} \\",
        rules,
        rf" & {sub} \\",
        r"\midrule",
    ]
    for case in CASES:
        cells = " & ".join(f"${c2st[f].loc[case, 'joint']:.3f}$ & ${c2st[f].loc[case, 'cell']:.3f}$"
                           for f in FAMILIES)
        lines.append(rf"{CASE_LABELS[case]} & {cells} \\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()

    metrics = {f: load_metrics(f) for f in FAMILIES}
    c2st = {f: load_c2st(f) for f in FAMILIES}
    header = ("% Generated by `python -m experiments.experiment_1a.gap_tables` -- do not edit by hand.\n"
              "% Needs \\usepackage{booktabs,tabularx} and a centered X column:\n"
              "% \\newcolumntype{C}{>{\\centering\\arraybackslash}X}\n")
    body = "\n\n".join([absolute_table(metrics), c2st_table(c2st), c2st_case_table(c2st)])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(header + "\n" + body + "\n")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
