"""Appendix figures + captions for Experiment 3 (held-out LBA: zero-shot, fine-tuning).

Writes outputs/finals/experiment-3/captions.tex, in appendix order:

1. Zero-shot LBA: the ORIGINAL network (DDM+RDM+CDM) on the LBA under RDM's model
   embedding (model_class_lba_cf_validate --eval_model_id 1).
2. Fine-tuned: the network after the 1000-epoch scheme-2 fine-tune (all four families,
   LBA under its own embedding initialised from RDM's), for LBA and -- as the forgetting
   check -- DDM, RDM and CDM (model_class_lba_finetune_validate).

Per arm and family: the bench5 cases as one figure each stacking (a) recovery, (b) ECDF,
(c) metrics; mixed_predictor (long grids) as three separate figures, as in Experiment 1.
Panel encodings are those of experiment-1a's first appendix figure, referenced by label,
and the mixed-predictor design is described in experiment-1's. The first figure of each
section carries the full caption; the rest a short one. Design sentences and coefficient
counts come from the benchmark configs and are checked against the saved masks.

    python -m experiments.experiment_3.appendix_captions
"""
from pathlib import Path

import numpy as np

from experiments import shared_config
from experiments.configs import MODELS as BASE
from experiments.experiment_1.appendix_captions import KINDS, PANEL, label as mixed_label_1
from experiments.experiment_1a.appendix_captions import (CASE_NAMES, N_DATASETS, N_DRAWS, N_TRIALS,
                                                         design_sentence, label as label_1a, names)

FINALS = Path("outputs/finals/experiment-3")
OUT = FINALS / "captions.tex"
PRED = FINALS / "predictions"
BENCH = ["intercept_only", "fixed", "regressed", "fixed_regressed", "interaction"]
MIXED = "mixed_predictor"
REF_ENC = label_1a("ddm", "intercept_only", "cf")        # full panel encoding (experiment 1a)
REF_MIXED = mixed_label_1("ddm", "recovery")             # mixed-predictor design (experiment 1)
FT_FAMILIES = ["lba", "ddm", "rdm", "cdm"]
FT_EPOCHS = int(shared_config.get("LBA_FT_EPOCHS"))
FIXED_A = shared_config.get("LBA_FIXED_A")
PRED_FILE = {"zeroshot": "zeroshot/{fam}_{case}_mid1_cf_pred.npz",
             "finetuned": "finetuned/{fam}_{case}_ft_pred.npz"}


def label(arm, fam, case, kind=None):
    return f"fig:app3-{arm}-{fam}-{case.replace('_', '-')}" + (f"-{kind}" if kind else "")


def mixed_sentence(arm, fam):
    """Who carries which effect in the mixed-predictor design, checked against the mask."""
    _, dc, opts = next(e for e in BASE[fam]["val_scenarios"] if e[0] == MIXED)
    assert opts.get("categorical_regressors") == ["u_1"], opts
    m = np.load(PRED / PRED_FILE[arm].format(fam=fam, case=MIXED))["params_mask"]
    m = m[0] if m.ndim == 3 else m
    shown = int(m[:, :len(BASE[fam]["intrinsic_params"])].sum())    # pi is the last column
    expected = len(dc["1"]) + 3 * len(dc["u_1"]) + len(dc["u_2"]) + 3 * len(dc["u_1:u_2"])
    assert shown == expected, f"{arm}/{fam}: mask {shown} != config {expected}"
    return (f"{names(fam, dc['u_1'])} carry effects of the four-level $u_1$; "
            f"{names(fam, dc['u_2'])} slopes on $u_2$; {names(fam, dc['u_1:u_2'])} the "
            f"interaction ({shown} coefficients shown).")


def stacked(arm, fam, case, caption):
    d = f"{arm}/{fam}/{case}"
    tall = r"width=\textwidth,height=0.3\textheight,keepaspectratio"
    blocks = [r"\begin{figure}[p]", r"  \centering"]
    for i, (kind, opts, name) in enumerate((("recovery", tall, "Recovery"),
                                            ("ecdf", tall, "Calibration (ECDF)"),
                                            ("metrics", r"width=\textwidth", "Metrics"))):
        blocks += [r"  \begin{subfigure}{\textwidth}\centering",
                   rf"    \includegraphics[{opts}]{{{d}/{kind}.pdf}}",
                   rf"    \caption{{{name}}}",
                   r"  \end{subfigure}" + (r"\\[1ex]" if i < 2 else "")]
    return "\n".join(blocks + [rf"  \caption{{{caption}}}", rf"  \label{{{label(arm, fam, case)}}}",
                               r"\end{figure}"])


def single(path, caption, lab):
    return "\n".join([r"\begin{figure}[p]", r"  \centering",
                      rf"  \includegraphics[width=\textwidth,height=0.82\textheight,keepaspectratio]{{{path}}}",
                      rf"  \caption{{{caption}}}", rf"  \label{{{lab}}}", r"\end{figure}"])


# ── section 1: zero-shot LBA ────────────────────────────────────────────────
ZS_FULL = (
    r"\textbf{{Zero-shot LBA, {CASE} design.}} (a) Recovery, (b) calibration and (c) summary "
    r"metrics of the original CogFormer network, trained on DDM, RDM and CDM only, applied to "
    r"the held-out LBA without any LBA training. The network is run under the RDM's model "
    r"embedding, whose parameters ($v$, $v_{{\mathrm{{diff}}}}$, $a$, $\tau$, $s_v$, $s_\tau$, "
    r"$\pi$) the LBA shares; the LBA's start-point range is fixed at $A = {A}$, and $a$ is the "
    r"mean distance an accumulator travels to its threshold, the quantity the RDM's $a$ "
    r"measures. Evaluated on the {N} bench5 LBA test datasets ({T} trials each, two "
    r"stimulus-coded alternatives), with {D} posterior draws per dataset; panels encoded as in "
    r"Figure~\ref{{{REF}}}. {DESIGN} All subsequent zero-shot figures "
    r"(Figures~\ref{{{NEXT}}}--\ref{{{LAST}}}) follow this layout; the mixed-predictor design, "
    r"whose grids are long, is shown as three separate figures."
)
ZS_SHORT = (r"\textbf{{Zero-shot LBA, {CASE} design.}} (a) Recovery, (b) calibration, (c) "
            r"metrics. {DESIGN} As in Figure~\ref{{{FIRST}}}.")
MIX_CAP = (r"\textbf{{{KIND}, {WHO}, mixed-predictor design.}} {DESIGN} Datasets simulated for "
           r"this design (no shared benchmark file exists); design as in Figure~\ref{{{MIXED}}}, "
           r"encoding as in the {PANEL} panel of Figure~\ref{{{REF}}}.")

# ── section 2: fine-tuned ───────────────────────────────────────────────────
FT_FULL = (
    r"\textbf{{After fine-tuning: {FAM}, {CASE} design.}} (a) Recovery, (b) calibration and "
    r"(c) summary metrics of the network after {EP} epochs of fine-tuning on all four families "
    r"jointly (DDM, RDM, CDM and LBA, drawn with equal probability), the LBA now under its own "
    r"model embedding, initialised from the RDM's. Same test datasets, draws and encoding as "
    r"Figure~\ref{{{ZS}}}, so each figure compares directly with its zero-shot counterpart. "
    r"{DESIGN} The same diagnostics for DDM, RDM and CDM after fine-tuning "
    r"(Figures~\ref{{{DDM}}}--\ref{{{LAST}}}) quantify forgetting: compare them with the "
    r"original network on the identical bench5 datasets (Figure~\ref{{{ORIG}}} and the "
    r"following CogFormer figures)."
)
FT_SHORT = (r"\textbf{{After fine-tuning: {FAM}, {CASE} design.}} (a) Recovery, (b) calibration, "
            r"(c) metrics. {DESIGN} As in Figure~\ref{{{FIRST}}}{CMP}.")


# ── section 3: Δr maps ───────────────────────────────────────────────────────
DR_FULL = (
    r"\textbf{{Change in parameter recovery with fine-tuning: LBA.}} "
    r"$\Delta r = r_{{\text{{fine-tuned}}}} - r_{{\text{{zero-shot}}}}$ per coefficient for "
    r"the five bench5 designs, {PANELS}, where $r$ is the Pearson correlation between the "
    r"posterior mean ({D} draws) and the ground truth across the {N} test datasets; both "
    r"networks are scored on identical datasets. Red marks better recovery after fine-tuning, "
    r"blue worse, on a common scale from $-1$ to $1$; each cell is annotated with its "
    r"$\Delta r$. Rows are the coefficients of the design and columns the parameters, "
    r"including $\pi$; N/A marks coefficients the design does not contain. Values are point "
    r"estimates; no uncertainty is shown. Figures~\ref{{{NEXT}}}--\ref{{{LAST}}} show the same "
    r"maps for DDM, RDM and CDM, where $\Delta r$ compares the fine-tuned network with the "
    r"original one and so measures forgetting."
)
DR_SHORT = (
    r"\textbf{{Forgetting in parameter recovery: {FAM}.}} "
    r"$\Delta r = r_{{\text{{fine-tuned}}}} - r_{{\text{{original}}}}$ (the original network "
    r"is labelled ``base'' in the panels) for {PANELS}, on the identical bench5 datasets; "
    r"encoding as in Figure~\ref{{{FIRST}}}."
)

# ── section 4: metric deltas ────────────────────────────────────────────────
DM_FULL = (
    r"\textbf{{Change in summary metrics with fine-tuning, {CASE} design.}} Per-coefficient "
    r"change in NRMSE (top row) and calibration error (bottom row) for DDM, RDM and CDM "
    r"(fine-tuned minus original network) and the LBA (fine-tuned minus zero-shot); each pair "
    r"of networks is scored on identical bench5 test datasets, with the metric definitions of "
    r"the bottom panel of Figure~\ref{{{REF}}}. Negative values (blue) are improvements, positive "
    r"values (red) deteriorations. Each column lists its own parameters and has its own "
    r"$x$-axis scale; $\pi$ is not shown. Figures~\ref{{{NEXT}}}--\ref{{{LAST}}} show the other "
    r"designs."
)
DM_SHORT = (r"\textbf{{Change in summary metrics with fine-tuning, {CASE} design.}} As in "
            r"Figure~\ref{{{FIRST}}}.")

# ── section 7: LBA on empirical data ────────────────────────────────────────
RD_VARS = [("condition", "speed/accuracy condition"), ("frequency", "word frequency"),
           ("lexicality", "lexicality")]
RD_FULL = (
    r"\textbf{{LBA on empirical lexical-decision data: posterior predictive check by "
    r"{VAR}.}} The fine-tuned network applied to the lexical-decision data of Wagenmakers, "
    r"Ratcliff, Gomez and McKoon (2008) ({N} subjects; {TMIN}--{TMAX} trials each, median "
    r"{TMED}), fitted per subject with an intercept and the speed/accuracy condition "
    r"($\pm 0.5$) on $v$, $v_{{\mathrm{{diff}}}}$, $a$ and $\tau$; lexicality enters only "
    r"through stimulus coding, the correct response's accumulator being the favoured one. Each "
    r"of {D} posterior draws per subject simulates that subject's own trials (contamination "
    r"included), and the simulated trials are pooled over subjects, giving one predictive data "
    r"set per draw; correct and error responses are pooled. (a) The 0.1, 0.3, 0.5, 0.7 and 0.9 "
    r"RT quantiles per level: observed (black squares, dashed) against the predictive median "
    r"(line) and its {CI}\% interval across draws (band). (b) RT densities: observed (dark line) "
    r"against individual predictive draws (thin lines), their pointwise median and {CI}\% band; "
    r"Gaussian kernel density estimates (Silverman bandwidth), plotted between the 1st and 99th "
    r"percentiles of the observed RTs. Panels have independent $y$-axes. "
    r"Figures~\ref{{{NEXT}}} and~\ref{{{LAST}}} show the same check by word frequency and by "
    r"lexicality."
)
RD_SHORT = (
    r"\textbf{{LBA on empirical lexical-decision data: posterior predictive check by "
    r"{VAR}.}} (a) RT quantiles and (b) RT densities, encoded as in Figure~\ref{{{FIRST}}}. "
    r"{VAR_CAP} {NOTE} is not part of the fitted design, so the predictive distribution is "
    r"identical across its levels by construction; differences between levels in the observed "
    r"data are effects the fitted model does not represent."
)
RD_NOTE = {"frequency": "Word frequency",
           "lexicality": "Lexicality enters only as the favoured accumulator, with the same drift "
                         "advantage for words and nonwords; as a factor it"}


def paired(paths, caption, lab, names_=None):
    blocks = [r"\begin{figure}[p]", r"  \centering"]
    for i, p in enumerate(paths):
        blocks += [r"  \begin{subfigure}{\textwidth}\centering",
                   rf"    \includegraphics[width=\textwidth,height=0.18\textheight,keepaspectratio]{{{p}}}",
                   rf"    \caption{{{names_[i] if names_ else ''}}}",
                   r"  \end{subfigure}" + (r"\\[0.5ex]" if i < len(paths) - 1 else "")]
    return "\n".join(blocks + [rf"  \caption{{{caption}}}", rf"  \label{{{lab}}}", r"\end{figure}"])


def realdata_counts():
    from experiments.experiment_2.data_utils import load_lba_data
    data, ids = load_lba_data("data/data_application_lba.csv")
    n = np.array([len(data[s]["rts"]) for s in ids])
    post = np.load(FINALS / "ppc" / "posteriors.npz", allow_pickle=True)["posteriors"]
    if post.shape[0] != len(ids):
        raise SystemExit(f"LBA: {post.shape[0]} fitted subjects, loader yields {len(ids)}")
    return dict(N=len(ids), TMIN=int(n.min()), TMED=int(np.median(n)), TMAX=int(n.max()),
                D=post.shape[1])


def sections_3_4_7():
    from experiments.experiment_2 import ppc
    out = []
    letters = "abcde"
    panels = ", ".join(f"({letters[i]}) {CASE_NAMES[c]}" for i, c in enumerate(BENCH))
    # 3. Δr maps
    out += ["% ── 3. change in recovery (Δr) ──", ""]
    dr = lambda f: f"fig:app3-delta-r-{f}"
    for f in FT_FAMILIES:
        files = [f"compare/{f}/{c}_{'delta_r' if f == 'lba' else 'forgetting_delta_r'}.pdf" for c in BENCH]
        fmt = dict(FAM=BASE[f]["name"], PANELS=panels, D=N_DRAWS, N=N_DATASETS)
        cap = (DR_FULL.format(**fmt, NEXT=dr("ddm"), LAST=dr("cdm")) if f == "lba"
               else DR_SHORT.format(**fmt, FIRST=dr("lba")))
        out += [paired(files, cap, dr(f)), ""]
    # 4. metric deltas
    out += ["% ── 4. change in metrics ──", ""]
    dm = lambda c: f"fig:app3-delta-metrics-{c.replace('_', '-')}"
    for i, c in enumerate(BENCH):
        fmt = dict(CASE=r"\emph{" + CASE_NAMES[c] + "}", REF=REF_ENC)
        cap = (DM_FULL.format(**fmt, NEXT=dm(BENCH[1]), LAST=dm(BENCH[-1])) if i == 0
               else DM_SHORT.format(**fmt, FIRST=dm(BENCH[0])))
        out += [single(f"compare/delta_metrics_{c}.pdf", cap, dm(c)).replace(
            "height=0.82\\textheight", "height=0.45\\textheight"), ""]
    # 7. real data (free variant only)
    out += ["% ── 7. LBA on empirical data (free variant: s_v, s_tau as estimated) ──", ""]
    rc = realdata_counts()
    rd = lambda v: f"fig:app3-realdata-{v}"
    for i, (v, vname) in enumerate(RD_VARS):
        files = [f"ppc/free/rt_quantiles_{v}.pdf", f"ppc/free/rt_ppc_{v}.pdf"]
        if i == 0:
            cap = RD_FULL.format(**rc, VAR=vname, CI=f"{ppc.PPC_CI:g}", NEXT=rd(RD_VARS[1][0]),
                                 LAST=rd(RD_VARS[2][0]))
        else:
            cap = RD_SHORT.format(VAR=vname, FIRST=rd(RD_VARS[0][0]), VAR_CAP="", NOTE=RD_NOTE[v])
        out += [paired(files, cap.replace("  ", " "), rd(v), ["RT quantiles", "RT densities"]), ""]
    return out


# ── section 6: LBA amortization gap (baseline figures) ──────────────────────
BF_BUDGET = (int(shared_config.get("MF_BF_EPOCHS")), int(shared_config.get("STEPS")),
             int(shared_config.get("MF_BF_BATCH")))
GAP_FULL = (
    r"\textbf{{LBA amortization gap: BayesFlow baseline, {CASE} design.}} (a) Recovery, (b) "
    r"calibration and (c) summary metrics of the case-specific BayesFlow baseline, trained for "
    r"this LBA design alone ({E} epochs $\times$ {S} steps $\times$ batch {B}, i.e.\ {M}M "
    r"simulated data sets, the per-case budget of the DDM/RDM/CDM baselines), on the same {N} "
    r"bench5 LBA test datasets as the fine-tuned network (Figure~\ref{{{FT}}}); encoding as in "
    r"Figure~\ref{{{REF}}}. {DESIGN} Figures~\ref{{{NEXT}}}--\ref{{{LAST}}} show the other "
    r"designs; Table~\ref{{tab:app3-lba-gap-c2st}} compares the two networks' posteriors."
)
GAP_SHORT = (r"\textbf{{LBA amortization gap: BayesFlow baseline, {CASE} design.}} (a) Recovery, "
             r"(b) calibration, (c) metrics. {DESIGN} As in Figure~\ref{{{FIRST}}}; fine-tuned "
             r"CogFormer on the same datasets: Figure~\ref{{{FT}}}.")


def section_6():
    out = ["% ── 6. LBA amortization gap: baseline diagnostics ──", ""]
    lab = lambda c: f"fig:app3-gap-lba-{c.replace('_', '-')}"
    E, S, B = BF_BUDGET
    for i, c in enumerate(BENCH):
        for k in ("recovery", "ecdf", "metrics"):
            if not (FINALS / "gap" / c / f"bf_{k}.pdf").exists():
                raise SystemExit(f"missing gap/{c}/bf_{k}.pdf -- run e3_gap_finalize first")
        fmt = dict(CASE=r"\emph{" + CASE_NAMES[c] + "}", DESIGN=design_sentence("lba", c),
                   FT=label("finetuned", "lba", c))
        cap = (GAP_FULL.format(**fmt, E=E, S=S, B=B, M=f"{E * S * B / 1e6:g}", N=N_DATASETS,
                               REF=REF_ENC, NEXT=lab(BENCH[1]), LAST=lab(BENCH[-1])) if i == 0
               else GAP_SHORT.format(**fmt, FIRST=lab(BENCH[0])))
        fig = stacked("gap", c, "", cap)
        fig = fig.replace(f"gap/{c}//", f"gap/{c}/bf_").replace(label("gap", c, ""), lab(c))
        out += [fig, ""]
    return out


# ── tables: forgetting C2ST and the LBA amortization gap ────────────────────
TABLES_OUT = FINALS / "tables.tex"


def write_tables():
    import pandas as pd
    from experiments.experiment_1a.gap_tables import fmt as med_sem, nrmse_bayesflow_convention

    head = ["% Generated by `python -m experiments.experiment_3.appendix_captions` -- do not edit by hand.",
            "% Needs \\usepackage{booktabs,tabularx} and \\newcolumntype{C}{>{\\centering\\arraybackslash}X}.", ""]

    # T1: forgetting C2ST (original vs fine-tuned network, DDM/RDM/CDM)
    fc = pd.read_csv(FINALS / "c2st" / "forgetting_c2st_summary.csv")
    rows = []
    for fam in ["DDM", "RDM", "CDM"]:
        sub = fc[fc["family"] == fam].set_index("scenario").loc[BENCH]
        assert sub["paired_datasets"].all(), fam
        for i, (case, r) in enumerate(sub.iterrows()):
            head_ = rf"\textbf{{{fam}}}" if i == 0 else ""
            rows.append(rf"{head_} & {CASE_NAMES[case]} & {int(r['n_active_cells'])} & "
                        rf"${r['mean_real_matched']:.3f}$ / ${r['mean_null_floor']:.3f}$ & "
                        rf"${r['perds_mean_real']:.3f}$ / ${r['perds_mean_null_floor']:.3f}$ & "
                        rf"${r['perds_median_excess']:+.3f}$ & ${100 * r['perds_frac_excess_gt_10']:.0f}\%$ \\")
        rows.append(r"\midrule")
    n_ds, half = int(fc["perds_n_datasets"].iloc[0]), int(fc["n_samples_per_side_matched"].iloc[0] // fc["perds_n_datasets"].iloc[0])
    t1 = [r"\begin{table}[t]", r"\centering", r"\small",
          r"\caption{Forgetting: C2ST accuracy of the original against the fine-tuned network on the "
          rf"identical bench5 test datasets ({n_ds} per design), each side using {half} posterior draws "
          r"per dataset and the draw-split null floor (each network's first half of draws against its "
          r"second half; the larger of the two) as reference. Pooled: one classifier per parameter over "
          r"the draws of all datasets, averaged over the active parameters (incl.\ $\pi$). Per dataset: "
          r"one classifier per dataset over all active parameters jointly, averaged over datasets; the "
          r"last two columns give the median excess over the floor and the share of datasets whose "
          r"excess exceeds 0.10. All classifiers are MLPs with 5-fold cross-validation.}",
          r"\label{tab:app3-forgetting-c2st}",
          r"\begin{tabularx}{\textwidth}{llcCCCC}", r"\toprule",
          r"\textbf{Model} & \textbf{Design} & \textbf{Params} & \textbf{Pooled} (real / floor) & "
          r"\textbf{Per dataset} (real / floor) & \textbf{Median excess} & \textbf{Excess $>0.10$} \\",
          r"\midrule"] + rows[:-1] + [r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""]

    # T2: LBA absolute metrics, both arms from the SAME metric code (repo adaptive_metrics),
    # plus the BayesFlow-convention NRMSE from the saved posteriors for both arms.
    bf_fig = Path("outputs/figures/model_family/bf/lba")
    arm = {"bf": [], "cf": []}
    for c in BENCH:
        b = pd.concat([pd.read_csv(bf_fig / c / f"lba_family_{c}_bf_metrics.csv", index_col=0),
                       pd.read_csv(bf_fig / c / f"lba_family_{c}_bf_metrics_global.csv", index_col=0)])
        f_ = pd.concat([pd.read_csv(FINALS / "finetuned" / "lba" / c / "metrics.csv", index_col=0),
                        pd.read_csv(FINALS / "finetuned" / "lba" / c / "metrics_pi.csv", index_col=0)])
        if len(b) != len(f_):
            raise SystemExit(f"lba/{c}: baseline {len(b)} vs fine-tuned {len(f_)} parameters")
        B_ = np.load(f"outputs/data/model_family/lba_{c}_data.npz")
        act = B_["param_masks"][0].astype(bool)
        F_ = np.load(PRED / PRED_FILE["finetuned"].format(fam="lba", case=c))
        m = (F_["params_mask"][0] if F_["params_mask"].ndim == 3 else F_["params_mask"]) == 1
        nb = nrmse_bayesflow_convention(B_["pred_set"][:, :, act], B_["true_set"][:, act])
        nf = nrmse_bayesflow_convention(F_["pred_set"][:, :, m], F_["true_set"][:, m])
        assert len(nb) == len(nf) == len(b)
        arm["bf"].append(b.assign(NRMSE_BF=np.sort(nb)))
        arm["cf"].append(f_.assign(NRMSE_BF=np.sort(nf)))
    arm = {k: pd.concat(v) for k, v in arm.items()}
    mcols = ["NRMSE", "NRMSE_BF", "Calibration Error", "Posterior Contraction"]
    t2 = [r"\begin{table}[t]", r"\centering", r"\small",
          r"\caption{LBA amortization gap: absolute metric values (median $\pm$ SEM) pooled over the "
          r"five benchmark designs and all parameters in each (incl.\ $\pi$), for the case-specific "
          r"BayesFlow baselines and the fine-tuned CogFormer, scored on the same test datasets with the "
          r"same code. NRMSE and NRMSE$^\dagger$ as in Table~\ref{tab:absolute-metrics}.}",
          r"\label{tab:app3-lba-gap-metrics}",
          r"\begin{tabularx}{\textwidth}{lCCCC}", r"\toprule",
          r"\textbf{Method} & \textbf{NRMSE} ($\downarrow$) & \textbf{NRMSE}$^\dagger$ ($\downarrow$) & "
          r"\textbf{Cal. Error} ($\downarrow$) & \textbf{Post. Contr.} ($\uparrow$) \\", r"\midrule",
          r"\textbf{Baseline} & " + " & ".join(med_sem(arm["bf"][c]) for c in mcols) + r" \\",
          r"\textbf{CogFormer}-$\mathcal{C}$ (fine-tuned) & " + " & ".join(med_sem(arm["cf"][c]) for c in mcols) + r" \\",
          r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""]

    # T3: LBA C2ST per design (same columns and classifiers as experiment-1a's tab:c2st-cases)
    gs = pd.read_csv(FINALS / "gap" / "summary.csv").set_index("case").loc[BENCH]
    fl = lambda col: (f"{gs[col].min():.2f}" if f"{gs[col].min():.2f}" == f"{gs[col].max():.2f}"
                      else f"{gs[col].min():.2f}--{gs[col].max():.2f}")
    rows = [rf"{CASE_NAMES[c].capitalize()} & ${r['perds_joint_c2st']:.3f}$ & ${r['perds_mean_accuracy']:.3f}$ & "
            rf"${r['joint_c2st']:.3f}$ \\" for c, r in gs.iterrows()]
    t3 = [r"\begin{table}[t]", r"\centering", r"\small",
          r"\caption{LBA amortization gap: C2ST accuracy of the fine-tuned CogFormer against the "
          r"case-specific BayesFlow baseline per benchmark design (0.5 = indistinguishable), with the "
          r"classifiers of Table~\ref{tab:c2st}: per dataset, joint over all active parameters (MLP) and "
          r"per parameter (quadratic discriminant), averaged over datasets; pooled, one classifier over "
          r"the draws of all datasets. Draw-split null floors: " + fl("perds_joint_null_floor") +
          r" (joint), " + fl("perds_mean_null_floor") + r" (per parameter).}",
          r"\label{tab:app3-lba-gap-c2st}",
          r"\begin{tabularx}{\textwidth}{lCCC}", r"\toprule",
          r"\textbf{Design} & \textbf{Joint, per dataset} & \textbf{Per parameter, per dataset} & "
          r"\textbf{Joint, pooled} \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""]

    TABLES_OUT.write_text("\n".join(head + t1 + t2 + t3))
    print(f"[saved] {TABLES_OUT} (3 tables)")


def main():
    blocks = ["% Generated by `python -m experiments.experiment_3.appendix_captions` -- do not edit by hand.",
              "% Needs \\usepackage{graphicx,subcaption}; paths are relative to outputs/finals/experiment-3/.",
              f"% Panel encodings refer to {REF_ENC} (experiment-1a/captions.tex), the mixed-predictor",
              f"% design to {REF_MIXED} (experiment-1/captions.tex).", ""]
    for arm, fams in (("zeroshot", ["lba"]), ("finetuned", FT_FAMILIES)):
        for f in fams:
            for c in BENCH + [MIXED]:
                for k in KINDS:
                    p = FINALS / arm / f / c / f"{k}.pdf"
                    if not p.exists():
                        raise SystemExit(f"missing {p} -- run e3_finalize first")

    # 1. zero-shot
    blocks += ["% ── 1. zero-shot LBA ──", ""]
    zs_first, zs_next = label("zeroshot", "lba", BENCH[0]), label("zeroshot", "lba", BENCH[1])
    zs_last = label("zeroshot", "lba", MIXED, "metrics")
    for i, c in enumerate(BENCH):
        fmt = dict(CASE=r"\emph{" + CASE_NAMES[c] + "}", DESIGN=design_sentence("lba", c),
                   A=FIXED_A, N=N_DATASETS, T=N_TRIALS, D=N_DRAWS, REF=REF_ENC)
        cap = (ZS_FULL.format(**fmt, NEXT=zs_next, LAST=zs_last) if i == 0
               else ZS_SHORT.format(**fmt, FIRST=zs_first))
        blocks += [stacked("zeroshot", "lba", c, cap), ""]
    for k in KINDS:
        cap = MIX_CAP.format(KIND=KINDS[k], WHO="zero-shot LBA", DESIGN=mixed_sentence("zeroshot", "lba"),
                             MIXED=REF_MIXED, PANEL=PANEL[k], REF=REF_ENC)
        blocks += [single(f"zeroshot/lba/{MIXED}/{k}.pdf", cap, label("zeroshot", "lba", MIXED, k)), ""]

    # 2. fine-tuned
    blocks += ["% ── 2. after fine-tuning ──", ""]
    ft_first = label("finetuned", "lba", BENCH[0])
    for f in FT_FAMILIES:
        for i, c in enumerate(BENCH):
            fmt = dict(FAM=BASE[f]["name"], CASE=r"\emph{" + CASE_NAMES[c] + "}",
                       DESIGN=design_sentence(f, c), EP=FT_EPOCHS)
            if f == "lba" and i == 0:
                cap = FT_FULL.format(**fmt, ZS=zs_first, DDM=label("finetuned", "ddm", BENCH[0]),
                                     LAST=label("finetuned", "cdm", MIXED, "metrics"),
                                     ORIG=label_1a("ddm", "intercept_only", "cf"))
            else:
                cmp = (rf"; zero-shot counterpart: Figure~\ref{{{label('zeroshot', 'lba', c)}}}" if f == "lba"
                       else rf"; before fine-tuning: Figure~\ref{{{label_1a(f, c, 'cf')}}}")
                cap = FT_SHORT.format(**fmt, FIRST=ft_first, CMP=cmp)
            blocks += [stacked("finetuned", f, c, cap), ""]
        for k in KINDS:
            cap = MIX_CAP.format(KIND=KINDS[k], WHO=f"{BASE[f]['name']} after fine-tuning",
                                 DESIGN=mixed_sentence("finetuned", f), MIXED=REF_MIXED,
                                 PANEL=PANEL[k], REF=REF_ENC)
            blocks += [single(f"finetuned/{f}/{MIXED}/{k}.pdf", cap, label("finetuned", f, MIXED, k)), ""]

    tail = sections_3_4_7()
    cut = next(i for i, b in enumerate(tail) if b.startswith("% ── 7."))
    blocks += tail[:cut] + section_6() + tail[cut:]
    OUT.write_text("\n".join(blocks))
    write_tables()
    print(f"[saved] {OUT} ({sum(b.startswith(chr(92) + 'begin{figure}') for b in blocks)} figures)")


if __name__ == "__main__":
    main()
