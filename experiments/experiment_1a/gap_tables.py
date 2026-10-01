"""LaTeX tables for the amortization gap (Experiment 1a).

Reads the two arms' per-case outputs on the shared bench5 test set and writes
outputs/finals/experiment-1a/amortization_gap_tables.tex with three tables:

  tab:absolute-metrics  NRMSE / calibration error / contraction, median +- SEM pooled over
                        the five bench5 cases and every active parameter in each case.
  tab:c2st              joint and mean marginal C2ST (median +- SEM over the five cases).
  tab:c2st-cases        joint C2ST per case, so the design dependence of the gap is visible.

Both arms report the same parameters per case in the same order; the CogFormer validator
splits them into <case>_fm_metrics.csv (per-row params) + _fm_metrics_global.csv (pi), the
BayesFlow evaluator writes them in one unlabeled file. The row counts are asserted equal.

SEM = sample standard deviation / sqrt(n).

    python -m experiments.experiment_1a.gap_tables
"""
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
           ("Calibration Error", r"\textbf{Cal. Error} ($\downarrow$)"),
           ("Posterior Contraction", r"\textbf{Post. Contr.} ($\uparrow$)")]
CF_NAME = r"\textbf{CogFormer}-$\mathcal{C}$"
BF_NAME = r"\textbf{Baseline}"

BF_EVAL = Path("outputs/evaluations/model_family/bf")
CF_VAL = Path("outputs/figures/model_class/cf") / f"{shared_config.get('MC_TAG')}_validate"
C2ST = Path("outputs/figures/c2st")
OUT = Path("outputs/finals/experiment-1a/amortization_gap_tables.tex")


def sem(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def fmt(x) -> str:
    return rf"${np.median(x):.3f} \pm {sem(x):.3f}$"


def load_metrics(fam: str) -> dict[str, pd.DataFrame]:
    """{arm: DataFrame of per-parameter metrics pooled over the five cases}."""
    bf, cf = [], []
    for case in CASES:
        b = pd.read_csv(BF_EVAL / f"{fam}_families_bf_{case}_evaluations.csv", sep=";", index_col=0)
        d = CF_VAL / fam / case
        c = pd.concat([pd.read_csv(d / f"{fam}_{case}_fm_metrics.csv", index_col=0),
                       pd.read_csv(d / f"{fam}_{case}_fm_metrics_global.csv", index_col=0)])
        if len(b) != len(c):
            raise SystemExit(f"{fam}/{case}: BF has {len(b)} parameters, CogFormer {len(c)}")
        bf.append(b)
        cf.append(c)
    return {"bf": pd.concat(bf), "cf": pd.concat(cf)}


def load_c2st(fam: str) -> pd.DataFrame:
    """One row per case: mean marginal and joint C2ST, CogFormer vs baseline."""
    rows = []
    for case in CASES:
        f = C2ST / fam / case / f"{fam}_{case}_orig_vs_bf_c2st.csv"
        if not f.exists():
            raise SystemExit(f"missing {f} -- has e1a_c2st finished?")
        s = pd.read_csv(f, index_col=0)["C2ST Accuracy"]
        rows.append({"case": case, "mean": s["_mean_accuracy_"], "joint": s["_joint_c2st_"]})
    return pd.DataFrame(rows).set_index("case")


def absolute_table(metrics: dict) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Absolute metric values (median $\pm$ SEM) aggregated across the five "
        r"benchmark design configurations and all parameters in each configuration.}",
        r"\label{tab:absolute-metrics}",
        r"\begin{tabularx}{\textwidth}{llCCC}",
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
    joint = " & ".join(fmt(c2st[f]["joint"]) for f in FAMILIES)
    marg = " & ".join(fmt(c2st[f]["mean"]) for f in FAMILIES)
    return "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{C2ST accuracy of {CF_NAME} vs.\ {BF_NAME} (median $\pm$ SEM across the "
        r"five benchmark design configurations). Joint: one classifier over all active "
        r"parameters; marginal: mean over per-parameter classifiers. 0.5 = indistinguishable.}",
        r"\label{tab:c2st}",
        r"\begin{tabularx}{\textwidth}{lCCC}",
        r"\toprule",
        rf"\textbf{{C2ST}} & {fams} \\",
        r"\midrule",
        rf"Joint & {joint} \\",
        rf"Marginal (mean) & {marg} \\",
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
    ])


def c2st_case_table(c2st: dict) -> str:
    fams = " & ".join(rf"\textbf{{{f.upper()}}}" for f in FAMILIES)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{Joint C2ST accuracy of {CF_NAME} vs.\ {BF_NAME} per benchmark design "
        r"configuration. 0.5 = indistinguishable.}",
        r"\label{tab:c2st-cases}",
        r"\begin{tabularx}{\textwidth}{lCCC}",
        r"\toprule",
        rf"\textbf{{Configuration}} & {fams} \\",
        r"\midrule",
    ]
    for case in CASES:
        cells = " & ".join(f"${c2st[f].loc[case, 'joint']:.3f}$" for f in FAMILIES)
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
