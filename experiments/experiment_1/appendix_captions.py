"""Appendix figures + captions for Experiment 1: mixed-predictor case and benchmark ensembles.

Writes outputs/finals/experiment-1/captions.tex, in appendix order:

1. mixed_predictor, per family: recovery, ECDF and metrics as three SEPARATE full-page
   figures (the grids are long: a 4-level categorical regressor adds three contrast rows
   to the main effect and three to the interaction).
2. benchmark ensembles (model_class_cf_benchmark_ensemble.py), the final portion:
   bench5 as one figure per family stacking recovery, ECDF and metrics (top to bottom); bench6
   (bench5 + mixed_predictor; long grids) as three separate figures per family.

Section 1: CogFormer only: no BayesFlow baseline was trained on this
design, so its 200 validation datasets are simulated by the validator (seeded) rather
than loaded from the shared benchmark files.

The first figure of each section carries the full caption; the others a short caption
pointing back to it. Panel encodings are those of experiment-1a's appendix figures
(experiments/experiment_1a/appendix_captions.py), which this file references by label.
The coefficient layout is checked against the saved parameter mask, so the caption
cannot drift from what was scored. Ensemble encodings follow
cogformer/diagnostics/plot/ensemble_{recovery,ecdf,metrics}.py: per-case points/curves in
husl case colours, r = mean of the per-case Pearson r, mask thumbnails with cell opacity
|r| (floored at 0.2), metrics = mean +- SD (ddof 1) over the cases containing the cell
with their count n, Log Gamma = log(gamma / gamma_0.05) (Saeilynoja et al. 2022 ECDF
statistic against the 5% quantile of its uniform-rank null, 1000 null draws).

Also writes outputs/finals/experiment-1/main_figure.tex, the main-text figure (short caption),
and main_figure_details.tex, the methods/appendix paragraph its caption refers to: (top)
random-design ensemble recovery of the coefficient matrix, (middle) its pi companion, (bottom)
information gain vs trial count. Facts there follow model_class_cf_ensemble.py (12
covering random designs, 4 per family, 2 regressors, binary categoricals, interactions
on), the pi prior (Beta(1, 50), identity link) and ddm_design_eig.py (CC/CB/BC/BB on v and
a, common-estimand rescaling, Gaussian EIG, NRMSE normalised by the SD of the truth, +-1
SE bands); the realised split balance is read from information_gain.csv.

    python -m experiments.experiment_1.appendix_captions
"""
from experiments.paths import RESULTS_ROOT
from pathlib import Path

import numpy as np

from experiments import shared_config
from experiments.configs import MODELS as BASE
from experiments.experiment_1a.appendix_captions import (FAMILIES, N_DATASETS, N_DRAWS,
                                                         N_TRIALS, label as label_1a, names)

FINALS = (RESULTS_ROOT / "finals/experiment-1")
OUT = FINALS / "captions.tex"
PRED = (RESULTS_ROOT / "data/model_class") / f"{shared_config.get('MC_TAG')}_validate"
CASE = "mixed_predictor"
NUM_CATEGORIES = 4          # exact_num_categories at the training grid's max_num_categories
KINDS = {"recovery": "Parameter recovery", "ecdf": "Simulation-based calibration",
         "metrics": "Summary metrics"}
# Encoding of each kind, as defined in the full caption of experiment-1a's first figure.
# Figures carry no subcaptions, so panels are named by their position in the stack.
PANEL = {"recovery": "top", "ecdf": "middle", "metrics": "bottom"}
REF_1A = label_1a("ddm", "intercept_only", "cf")


def mixed_design(fam):
    """(design config, number of coefficients shown) for the mixed-predictor case."""
    entry = next(e for e in BASE[fam]["val_scenarios"] if e[0] == CASE)
    _, dc, opts = entry
    assert opts.get("categorical_regressors") == ["u_1"], opts
    mask = np.load(PRED / f"{fam}_{CASE}_cf_pred.npz")["params_mask"]
    mask = mask[0] if mask.ndim == 3 else mask
    pi_col = BASE[fam]["intrinsic_params"].index("pi") if "pi" in BASE[fam]["intrinsic_params"] \
        else len(BASE[fam]["intrinsic_params"])
    rows = mask[mask.any(1)]
    k1 = NUM_CATEGORIES - 1
    # intercept, K-1 contrasts of u_1, u_2, K-1 contrast x u_2 interactions
    assert len(rows) == 1 + k1 + 1 + k1, f"{fam}: {len(rows)} active rows"
    shown = int(np.delete(mask, pi_col, axis=1).sum()) if mask.shape[1] > pi_col else int(mask.sum())
    expected = len(dc["1"]) + k1 * len(dc["u_1"]) + len(dc["u_2"]) + k1 * len(dc["u_1:u_2"])
    assert shown == expected, f"{fam}: mask has {shown} coefficients, config implies {expected}"
    return dc, shown


def label(fam, kind):
    return f"fig:app1-{fam}-mixed-predictor-{kind}"


def design_sentence(fam):
    dc, shown = mixed_design(fam)
    return (f"{names(fam, dc['u_1'])} carry effects of $u_1$; {names(fam, dc['u_2'])} carry "
            f"slopes on $u_2$; {names(fam, dc['u_1:u_2'])} carry the interaction "
            f"({shown} coefficients shown).")


FULL = (
    r"\textbf{{{KIND}, {FAM}, \emph{{mixed-predictor}} design (CogFormer).}} "
    r"The design combines a categorical and a continuous regressor: $u_1$ has "
    r"{K} levels and is dummy-coded against a reference level, giving one row per contrast "
    r"($u_1{{:}}c_1$ to $u_1{{:}}c_{K1}$); $u_2$ is continuous; and the rows "
    r"$u_1{{:}}c_k \times u_2$ are their interaction, one per contrast. {DESIGN} No BayesFlow "
    r"baseline was trained on this design, so only CogFormer is shown, on {N} datasets "
    r"({T} trials each) simulated for it, with {D} posterior draws per dataset. Encoding as "
    r"in the {PANEL} panel of Figure~\ref{{{REF}}}; the contamination probability $\pi$ is not "
    r"shown. Figures~\ref{{{NEXT}}}--\ref{{{LAST}}} show the remaining diagnostics of this "
    r"design and the other two families in the same way."
)
SHORT = (r"\textbf{{{KIND}, {FAM}, \emph{{mixed-predictor}} design (CogFormer).}} "
         r"{DESIGN} Design as in Figure~\ref{{{FIRST}}}; encoding as in the {PANEL} panel of "
         r"Figure~\ref{{{REF}}}.")


def figure(fam, kind, caption):
    return single_figure(f"per_case/{fam}/{CASE}/{kind}.pdf", caption, label(fam, kind))


def single_figure(path, caption, lab):
    return "\n".join([
        r"\begin{figure}[p]",
        r"  \centering",
        rf"  \includegraphics[width=\textwidth,height=0.82\textheight,keepaspectratio]{{{path}}}",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{lab}}}",
        r"\end{figure}",
    ])


# ── benchmark ensembles ──────────────────────────────────────────────────────
ENS = "benchmark_ensemble"


def ens_label(fam, variant, kind=None):
    return f"fig:app1-{fam}-{variant}" + (f"-{kind}" if kind else "")


ENS_FULL = (
    r"\textbf{{Benchmark ensemble, {FAM}: the five benchmark designs overlaid (CogFormer).}} "
    r"Parameter recovery (top), calibration (middle) and summary metrics (bottom) of the intercept-only, "
    r"fixed, regressed, fixed + regressed and interaction designs on one coefficient grid; "
    r"each design contributes its {N} bench5 test datasets ({T} trials, {D} posterior draws "
    r"each), and a cell pools every design that contains that coefficient. Rows and columns "
    r"as in Figure~\ref{{{REF}}}; $\pi$ is not shown. "
    r"Top: posterior means against ground truth, one colour per design, with central 90\% "
    r"credible intervals; $r$ is the mean over designs of the per-design Pearson correlation. "
    r"The strip on the left draws each design's coefficient pattern in its colour, with cell "
    r"opacity proportional to that design's $|r|$ (floored at 0.2), so poorly recovered "
    r"coefficients appear faint. "
    r"Middle: rank-ECDF difference curves, one per design, against the common pointwise 95\% band "
    r"for {N} uniform ranks; reading as in the middle panel of Figure~\ref{{{REF}}}. "
    r"Bottom: each metric of the bottom panel of Figure~\ref{{{REF}}} computed per design, then shown as "
    r"the mean $\pm$ standard deviation over the $n$ designs that contain the coefficient, "
    r"coloured by the mean. The fourth heatmap, log $\gamma$, is the log ratio of the "
    r"ECDF-based uniformity statistic $\gamma$ of the calibration ranks to the 5\% quantile of "
    r"its null distribution under uniform ranks: values below zero reject calibration at the "
    r"5\% level, values above zero do not; its colour scale spans the observed range. "
    r"Figures~\ref{{{NEXT}}}--\ref{{{LAST}}} show the other two families in the same way, "
    r"followed by the six-design ensembles."
)
ENS_SHORT = (r"\textbf{{Benchmark ensemble, {FAM}: the five benchmark designs overlaid "
             r"(CogFormer).}} Recovery (top), calibration (middle), metrics (bottom). Layout and encoding as "
             r"in Figure~\ref{{{FIRST}}}.")
BENCH6_FULL = (
    r"\textbf{{{KIND}, six-design benchmark ensemble, {FAM} (CogFormer).}} The five benchmark "
    r"designs of Figure~\ref{{{B5}}} plus the mixed-predictor design "
    r"(Figure~\ref{{{MIXED}}}), overlaid on one grid in the mixed-predictor design's layout. "
    r"The first contrast of the categorical $u_1$ shares a row with the continuous designs' "
    r"$u_1$ slope, labelled $u_1(:c_1)$ (a slope or the first contrast, depending on the "
    r"design), and likewise $u_1(:c_1) \times u_2$ for the interaction; the $u_2$ row pools "
    r"every design with a $u_2$ slope, and the further contrast rows ($u_1{{:}}c_2$, "
    r"$u_1{{:}}c_3$ and their interactions with $u_2$) come from the mixed-predictor design "
    r"alone. Encoding as in the {PANEL} panel of "
    r"Figure~\ref{{{B5}}}. Shown separately because the grid is long; "
    r"Figures~\ref{{{NEXT}}}--\ref{{{LAST}}} show the remaining diagnostics and families."
)
BENCH6_SHORT = (r"\textbf{{{KIND}, six-design benchmark ensemble, {FAM} (CogFormer).}} Designs "
                r"as in Figure~\ref{{{FIRST}}}; encoding as in the {PANEL} panel of "
                r"Figure~\ref{{{B5}}}.")


def bench5_figure(fam, caption):
    d = f"{ENS}/bench5/{fam}"
    tall = r"width=\textwidth,height=0.3\textheight,keepaspectratio"
    blocks = [r"\begin{figure}[p]", r"  \centering"]
    for i, (kind, opts) in enumerate((("recovery", tall), ("ecdf", tall),
                                      ("metrics", r"width=\textwidth"))):
        blocks += [rf"  \includegraphics[{opts}]{{{d}/{kind}.pdf}}"
                   + (r"\par\vspace{1ex}" if i < 2 else "")]
    return "\n".join(blocks + [rf"  \caption{{{caption}}}",
                               rf"  \label{{{ens_label(fam, 'bench5')}}}", r"\end{figure}"])


def ensemble_blocks():
    out = []
    for i, f in enumerate(FAMILIES):
        fmt = dict(FAM=BASE[f]["name"], N=N_DATASETS, T=N_TRIALS, D=N_DRAWS, REF=REF_1A)
        cap = (ENS_FULL.format(**fmt, NEXT=ens_label(FAMILIES[1], "bench5"),
                               LAST=ens_label(FAMILIES[-1], "bench5")) if i == 0
               else ENS_SHORT.format(**fmt, FIRST=ens_label(FAMILIES[0], "bench5")))
        out += [bench5_figure(f, cap), ""]
    order = [(f, k) for f in FAMILIES for k in KINDS]
    for i, (f, k) in enumerate(order):
        fmt = dict(KIND=KINDS[k], FAM=BASE[f]["name"], PANEL=PANEL[k],
                   B5=ens_label(f, "bench5"))
        cap = (BENCH6_FULL.format(**fmt, MIXED=label(f, "recovery"),
                                  NEXT=ens_label(*order[1][:1], "bench6", order[1][1]),
                                  LAST=ens_label(order[-1][0], "bench6", order[-1][1]))
               if i == 0 else
               BENCH6_SHORT.format(**fmt, FIRST=ens_label(order[0][0], "bench6", order[0][1])))
        out += [single_figure(f"{ENS}/bench6/{f}/{k}.pdf", cap, ens_label(f, "bench6", k)), ""]
    return out


# ── main-text figure ─────────────────────────────────────────────────────────
MAIN_OUT = FINALS / "main_figure.tex"
EIG_CSV = FINALS / "information_gain" / "information_gain.csv"
NUM_PRIOR_DRAWS = 1500      # ddm_design_eig.py --prior_batch default (e1_design_eig does not override it)
ENSEMBLE_CSV = FINALS / "random_ensemble" / "design_configs.csv"

MAIN = (
    r"\textbf{{Amortized inference across design space.}} "
    r"Top: parameter recovery for {NCFG} randomly drawn regression designs ({PER} per model "
    r"family; two continuous or binary regressors and their interaction), all estimated by one "
    r"CogFormer network: posterior means against ground truth with 90\% credible intervals, one "
    r"colour per design; $r$ is the mean per-design correlation. Columns are the parameters of "
    r"all three families and rows the coefficients (intercept, main effects of $u_1$ and $u_2$, "
    r"interaction); the strip on the left shows each design's coefficient pattern, with opacity "
    r"proportional to its $|r|$. Middle: the contamination probability $\pi$ for each design. "
    r"Bottom: expected information gain (left) and mean NRMSE (right) against trial count when each "
    r"of two DDM predictors is measured continuously (C) or as a binary factor (B); bands are "
    r"$\pm 1$ standard error. Every estimate is an amortized forward pass of the same network. "
    r"Details in Appendix~\ref{{{DETAILS}}}."
)
DETAILS_OUT = FINALS / "main_figure_details.tex"
DETAILS_LABEL = "app:exp1-main-details"
DETAILS = (
    r"Each random design in the top panel of Figure~\ref{{fig:exp1-main}} has two regressors, each continuous "
    r"or binary (one contrast, so $u_k(:c_1)$ denotes a slope or a contrast), and their "
    r"interaction, with each parameter entering each coefficient at random; the set of "
    r"{NCFG} designs was redrawn until together they cover every coefficient of the grid. Each "
    r"design contributes {N} simulated datasets of {T} trials, with {D} posterior draws per "
    r"dataset. Coefficients are shown on the unconstrained (link) scale on which the network is "
    r"trained; $\pi$ (middle panel) is shown on its natural scale, with prior "
    r"$\mathrm{{Beta}}(1, 50)$. "
    r"For the design comparison (bottom panel), both DDM predictors act on "
    r"drift rate and boundary separation ({NACT} estimated coefficients, including $\pi$). "
    r"Expected information gain is the Gaussian log-volume ratio "
    r"$\tfrac{{1}}{{2}}\log\det\Sigma_{{\text{{prior}}}} - \tfrac{{1}}{{2}}\,"
    r"\mathbb{{E}}_x\!\left[\log\det\Sigma_{{\text{{post}}}}(x)\right]$, in nats, with "
    r"$\Sigma_{{\text{{prior}}}}$ estimated from {NPRIOR} prior draws; NRMSE is the root mean "
    r"squared error of the posterior draws against ground truth, divided by the standard "
    r"deviation of the ground-truth values and averaged over the coefficients. Both are computed "
    r"on a common estimand, each slope scaled by the realised standard deviation of its "
    r"predictor, so that continuous and binary predictors are comparable. Information gain is "
    r"relative to each design's own prior, which is wider for binary contrasts (a contrast of two "
    r"cell means has twice the prior variance of a continuous slope). Each point uses {N} "
    r"datasets with {D} posterior draws; standard errors are over datasets (bootstrapped for "
    r"NRMSE). Binary predictors are randomly unbalanced dichotomisations (realised balance "
    r"{BAL_LO}--{BAL_HI}), not median splits."
)


def main_figure():
    eig = __import__("pandas").read_csv(EIG_CSV)
    cfgs = __import__("pandas").read_csv(ENSEMBLE_CSV)
    per = cfgs.groupby("family").size()
    assert per.nunique() == 1, per
    n_act = eig["n_active"].unique()
    assert len(n_act) == 1, n_act
    bal = eig["mean_split_balance"].dropna()
    vals = dict(NCFG=len(cfgs), PER=int(per.iloc[0]), N=N_DATASETS, T=N_TRIALS, D=N_DRAWS,
                NACT=int(n_act[0]), NPRIOR=NUM_PRIOR_DRAWS,
                BAL_LO=f"{bal.min():.2f}", BAL_HI=f"{bal.max():.2f}")
    cap = MAIN.format(**vals, DETAILS=DETAILS_LABEL)
    DETAILS_OUT.write_text("\n".join([
        "% Generated by `python -m experiments.experiment_1.appendix_captions` -- do not edit by hand.",
        "% Methods/appendix text for the main-text figure fig:exp1-main (main_figure.tex).",
        "% Place it under a heading carrying the label below, e.g.",
        f"% \\subsection{{Design-space figure: details}}\\label{{{DETAILS_LABEL}}}",
        DETAILS.format(**vals),
        "",
    ]))
    print(f"[saved] {DETAILS_OUT}")
    for f in ("random_ensemble/recovery.pdf", "random_ensemble/recovery_pi.pdf",
              "information_gain/information_gain.pdf"):
        if not (FINALS / f).exists():
            raise SystemExit(f"missing {FINALS / f} -- run e1_finalize first")
    tex = "\n".join([
        "% Generated by `python -m experiments.experiment_1.appendix_captions` -- do not edit by hand.",
        "% Main-text figure for Experiment 1. Needs \\usepackage{graphicx,amsmath,amssymb};",
        "% paths are relative to outputs/finals/experiment-1/.",
        r"\begin{figure}[tbp]",
        r"  \centering",
        r"  \includegraphics[width=\textwidth]{random_ensemble/recovery.pdf}\par\vspace{1ex}",
        r"  \includegraphics[width=0.85\textwidth]{random_ensemble/recovery_pi.pdf}\par\vspace{1ex}",
        r"  \includegraphics[width=0.75\textwidth]{information_gain/information_gain.pdf}",
        rf"  \caption{{{cap}}}",
        r"  \label{fig:exp1-main}",
        r"\end{figure}",
        "",
    ])
    MAIN_OUT.write_text(tex)
    print(f"[saved] {MAIN_OUT}")


def main():
    main_figure()
    order = [(f, k) for f in FAMILIES for k in KINDS]
    for f, k in order:
        for p in (FINALS / "per_case" / f / CASE / f"{k}.pdf",
                  FINALS / ENS / "bench5" / f / f"{k}.pdf",
                  FINALS / ENS / "bench6" / f / f"{k}.pdf"):
            if not p.exists():
                raise SystemExit(f"missing {p} -- run e1_finalize first")
    blocks = [
        "% Generated by `python -m experiments.experiment_1.appendix_captions` -- do not edit by hand.",
        "% Needs \\usepackage{graphicx}; paths are relative to outputs/finals/experiment-1/.",
        f"% Mixed-predictor case and benchmark ensembles, CogFormer only; panel encodings refer to {REF_1A}",
        "% (outputs/finals/experiment-1a/captions.tex).",
        "",
        "% ── mixed-predictor case ──",
        "",
    ]
    for i, (f, k) in enumerate(order):
        fmt = dict(KIND=KINDS[k], FAM=BASE[f]["name"], DESIGN=design_sentence(f),
                   PANEL=PANEL[k], REF=REF_1A)
        cap = (FULL.format(**fmt, K=NUM_CATEGORIES, K1=NUM_CATEGORIES - 1, N=N_DATASETS,
                           T=N_TRIALS, D=N_DRAWS, NEXT=label(*order[1]), LAST=label(*order[-1]))
               if i == 0 else SHORT.format(**fmt, FIRST=label(*order[0])))
        blocks += [figure(f, k, cap), ""]
    blocks += ["% ── benchmark ensembles ──", ""] + ensemble_blocks()
    OUT.write_text("\n".join(blocks))
    print(f"[saved] {OUT} ({blocks.count('') - 3} figures)")


if __name__ == "__main__":
    main()
