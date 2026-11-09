"""Appendix figures + captions: per-case recovery, ECDF and metrics, CogFormer vs BayesFlow.

Writes outputs/finals/experiment-1a/captions.tex: one figure per bench5 case x family x
arm (CogFormer, then the BayesFlow baseline), stacking recovery (top), the ECDF
calibration check (middle) and the metric heatmaps (bottom), with no subcaptions: the
captions name panels by position. pi is estimated jointly but not shown.
The FIRST figure carries the full caption; every later one a short caption that names
the case and arm and points back to it. The per-design sentence is generated from the
same benchmark_configs the validators score, so it cannot drift from the code.

Paths in \\includegraphics are relative to outputs/finals/experiment-1a/ (set
\\graphicspath accordingly). Facts stated in the full caption are those of the plotting
code: cogformer/diagnostics/plot/adaptive_recovery.py (posterior mean, central 90%
interval via credible_interval(prob=0.9), dashed identity, Pearson r), adaptive_ecdf.py
(fractional rank = share of draws below the truth; band = pointwise 95% Beta
order-statistic band for the number of datasets) and adaptive_metrics.py with
cogformer/diagnostics/metric/adaptive_metrics.py (range-normalised draw-based RMSE,
median over draws; contraction 1 - var_post/var_truth clipped to [0, 1], median over
datasets; calibration error = median |empirical - nominal| coverage over 20 central
interval levels in [0.005, 0.995]; all on fixed 0-1 colour scales).

    python -m experiments.experiment_1a.appendix_captions
"""
from experiments.paths import RESULTS_ROOT
from pathlib import Path

from experiments.configs import MODELS as BASE
from experiments.experiment_1a.model_family_c2st import MODELS

FINALS = (RESULTS_ROOT / "finals/experiment-1a")
OUT = FINALS / "captions.tex"
FAMILIES = ["ddm", "rdm", "cdm"]
CASES = ["intercept_only", "fixed", "regressed", "fixed_regressed", "interaction"]
CASE_NAMES = {"intercept_only": "intercept-only", "fixed": "fixed",
              "regressed": "regressed", "fixed_regressed": "fixed + regressed",
              "interaction": "interaction"}
N_DATASETS, N_DRAWS, N_TRIALS = 200, 200, 500   # e1_validate / model_family_bf_validate
REGRESSORS = {"u_1": "$u_1$", "u_2": "$u_2$", "u_1:u_2": "the product $u_1 u_2$"}


def names(fam, params):
    v = dict(zip(MODELS[fam]["intrinsic_params"], MODELS[fam]["variable_names"]))
    out = [v[p] for p in params]
    return out[0] if len(out) == 1 else ", ".join(out[:-1]) + " and " + out[-1]


def design_sentence(fam, case):
    """What this design estimates, in words, from its benchmark config."""
    dc = MODELS[fam]["benchmark_configs"][case]
    fixed = [p for p in BASE[fam]["intrinsic_params"] if p not in dc["1"]]
    parts = []
    if fixed:
        vals = {format(BASE[fam]["default_fixed_values"][p], "g") for p in fixed}
        at = (f"at {vals.pop()}" if len(vals) == 1 else
              "at " + ", ".join(f"{names(fam, [p])} = "
                                f"{BASE[fam]['default_fixed_values'][p]:g}" for p in fixed))
        parts.append(f"{names(fam, fixed)} are fixed {at} and not estimated")
    slopes = {r: dc.get(r, []) for r in REGRESSORS}
    if not any(slopes.values()):
        parts.append("every estimated parameter enters as an intercept only")
    elif slopes["u_1"] == slopes["u_2"]:
        parts.append(f"{names(fam, slopes['u_1'])} carry slopes on $u_1$ and $u_2$")
    else:
        for r in ("u_1", "u_2"):
            if slopes[r]:
                parts.append(f"{names(fam, slopes[r])} carry slopes on {REGRESSORS[r]}")
    if slopes["u_1:u_2"]:
        parts.append(f"{names(fam, slopes['u_1:u_2'])} on {REGRESSORS['u_1:u_2']}")
    n_coef = sum(len([p for p in v if p != "pi"]) for v in dc.values())
    s = "; ".join(parts)
    return f"{s[0].upper()}{s[1:]} ({n_coef} coefficients shown)."


def label(fam, case, arm):
    return f"fig:app1-{fam}-{case.replace('_', '-')}-{arm}"


ARM_NAMES = {"cf": "CogFormer", "bf": "BayesFlow baseline"}

FULL = (
    r"\textbf{{{FAM}, {CASE} design: {ARM}.}} "
    r"Parameter recovery (top), calibration (middle) and summary metrics (bottom) on the {N} bench5 "
    r"test datasets ({T} trials each), from {D} posterior draws per dataset. CogFormer is a "
    r"single network amortized over all three model families and all designs; each BayesFlow "
    r"baseline is trained for one family and design alone, and both are evaluated on the same "
    r"datasets. In every panel, columns are the model parameters and rows the coefficients of "
    r"the design: the intercept ($1$) and, in regressed designs, the slopes on the continuous "
    r"regressors $u_1$, $u_2$ and their product $u_1 u_2$; shading separates intercepts, main "
    r"effects and interactions, and N/A marks coefficients the design does not contain. "
    r"Values are on the unconstrained (link) scale on which both networks are trained; the "
    r"contamination probability $\pi$ is estimated jointly but not shown. "
    r"Top: posterior means (dots) against ground truth, with central 90\% credible intervals "
    r"(bars); the dashed line is the identity and $r$ the Pearson correlation between "
    r"posterior means and ground truth. "
    r"Middle: empirical CDF of the fractional rank of the ground truth among the posterior draws "
    r"(the share of draws below the truth), minus the uniform CDF. A calibrated posterior "
    r"stays inside the grey band, the pointwise 95\% band for {N} uniform ranks, around the "
    r"dashed zero line. A curve above zero throughout indicates overestimation, below zero "
    r"underestimation; an inverted U indicates posteriors that are too wide, a U-shape "
    r"overconfident ones. "
    r"Bottom: per coefficient, on a common 0--1 colour scale: NRMSE, the root mean squared error "
    r"of the posterior draws against the ground truth across datasets (median over draws), "
    r"divided by the range of the ground-truth values (lower is better); posterior "
    r"contraction, one minus the ratio of posterior variance to the variance of the "
    r"ground-truth values, clipped to $[0, 1]$ and taken as the median over datasets (higher "
    r"is better); and calibration error, the median absolute difference between empirical "
    r"and nominal coverage of central credible intervals at 20 levels from 0.5\% to 99.5\% "
    r"(lower is better). "
    r"{DESIGN} All subsequent figures in this section (Figures~\ref{{{NEXT}}}--"
    r"\ref{{{LAST}}}) follow this layout and encoding."
)
SHORT = (r"\textbf{{{FAM}, {CASE} design: {ARM}.}} Recovery (top), calibration (middle), "
         r"metrics (bottom). {DESIGN} Layout and encoding as in Figure~\ref{{{FIRST}}}.")


def figure(fam, case, arm, caption):
    d = f"{fam}/{case}"
    tall = r"width=\textwidth,height=0.3\textheight,keepaspectratio"
    blocks = [r"\begin{figure}[p]", r"  \centering"]
    for i, (kind, opts) in enumerate((("recovery", tall), ("ecdf", tall),
                                      ("metrics", r"width=\textwidth"))):
        blocks += [rf"  \includegraphics[{opts}]{{{d}/{arm}_{kind}.pdf}}"
                   + (r"\par\vspace{1ex}" if i < 2 else "")]
    return "\n".join(blocks + [rf"  \caption{{{caption}}}",
                               rf"  \label{{{label(fam, case, arm)}}}", r"\end{figure}"])


def main():
    order = [(f, c, a) for f in FAMILIES for c in CASES for a in ("cf", "bf")]
    for f, c, a in order:
        for kind in ("recovery", "ecdf", "metrics"):
            p = FINALS / f / c / f"{a}_{kind}.pdf"
            if not p.exists():
                raise SystemExit(f"missing {p} -- run e1a_finalize first")
    blocks = [
        "% Generated by `python -m experiments.experiment_1a.appendix_captions` -- do not edit by hand.",
        "% Needs \\usepackage{graphicx}; paths are relative to outputs/finals/experiment-1a/.",
        "% Per bench5 case and arm: recovery (top), ECDF calibration (middle), metrics (bottom).",
        "",
    ]
    for k, (f, c, a) in enumerate(order):
        fmt = dict(FAM=BASE[f]["name"], CASE=r"\emph{" + CASE_NAMES[c] + "}",
                   ARM=ARM_NAMES[a], DESIGN=design_sentence(f, c),
                   N=N_DATASETS, D=N_DRAWS, T=N_TRIALS)
        cap = (FULL.format(**fmt, NEXT=label(*order[1]), LAST=label(*order[-1])) if k == 0
               else SHORT.format(**fmt, FIRST=label(*order[0])))
        blocks += [figure(f, c, a, cap), ""]
    OUT.write_text("\n".join(blocks))
    print(f"[saved] {OUT} ({len(order)} figures)")


if __name__ == "__main__":
    main()
