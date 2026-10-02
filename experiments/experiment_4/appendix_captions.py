"""Figures + captions for Experiment 4 (frozen CogFormer encoder as a summary for the aDDM).

Writes, under outputs/finals/experiment-4/:

- main_figure.tex: the combined panel (probe/vsvfixed/compare/intercept_only_vsvfixed_panel.pdf):
  recovery of both arms, the pooled five-family MDS geometry before/after the LBA
  fine-tune, and the CKA Gram block; short caption;
- main_figure_details.tex: the methods/appendix paragraph that caption refers to;
- captions.tex: appendix figures -- each arm's four BayesFlow diagnostics (baseline, then
  frozen), the paired shared-validation comparison (C2ST + log density), and the RSA
  figure with both reference histograms.

Facts follow the code: addm_bf_apply.py / addm_bf_baseline.py (variant, pinned values,
matched 500 x 100 x 64 budget, mean-pooled seed summary), addm_diagnostics.py and
combined_recovery.py (posterior median, central 95% interval, baseline row on top),
addm_arm_compare.py (per-dataset C2ST against a frozen-vs-frozen draw-split null;
identified = contraction >= 0.5 in both arms on the shared set), addm_representation.py
(crossnobis RDMs over repeats, unbiased linear CKA, reference = random 128 -> 16
projections of the FROZEN summary itself) and crossfamily_representation.py (repeat-
averaged summaries, pooled standardisation, one MDS fit for both encoders).

THE REFERENCE IS NOT A NO-STRUCTURE NULL. matched_null compares the frozen summary with
random linear projections OF ITSELF, which keep the frozen geometry up to compression
loss. The captions therefore call it what it computes and do not read "observed <=
reference" as "different geometry".

Every number is read from the snapshot. The .npz inputs are gitignored and live only on
the host that ran the experiment; --draft renders the text anyway with ?? for anything
it cannot read, for review elsewhere -- never commit a draft.

    python -m experiments.experiment_4.appendix_captions [--draft]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import shared_config
from experiments.experiment_4.addm_bf_apply import S_V_FIXED, V_FIXED

FINALS = Path("outputs/finals/experiment-4")
PROBE = FINALS / "probe/vsvfixed"
REP = FINALS / "representation"
PANEL = "probe/vsvfixed/compare/intercept_only_vsvfixed_panel.pdf"
ARMS = {"baseline": "learned SetTransformer summary (baseline arm)",
        "frozen": "frozen CogFormer encoder (frozen arm)"}
TEX = {"v_diff": r"$v_{\mathrm{diff}}$", "omega": r"$\omega$", "a": "$a$", "tau": r"$\tau$",
       "s_tau": r"$s_\tau$"}
IDENTIFIED_THRESHOLD = 0.5      # addm_arm_compare --identified_threshold default
EPOCHS, STEPS, BATCH = 500, 100, 64
DETAILS_LABEL = "app:exp4-main-details"
MISSING = r"\textbf{??}"


class Facts(dict):
    """Values for the templates. In draft mode an unreadable source yields ?? instead of
    stopping, and every miss is listed at the end."""

    def __init__(self, draft):
        super().__init__()
        self.draft, self.missed = draft, []

    def load(self, keys, fn, what):
        try:
            self.update(fn())
        except (FileNotFoundError, KeyError) as e:
            if not self.draft:
                raise SystemExit(f"cannot read {what}: {e}\n(the .npz inputs exist only on the "
                                 "host that ran experiment 4; use --draft to preview elsewhere)")
            self.missed.append(f"{what}: {e}")
            self.update({k: MISSING for k in keys})


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


def arm_metrics():
    """bf_metrics.csv per arm -> {arm: DataFrame indexed by param}."""
    out = {arm: pd.read_csv(PROBE / arm / "bf_metrics.csv").set_index("param") for arm in ARMS}
    if list(out["frozen"].index) != list(out["baseline"].index):
        raise SystemExit("arms report different parameters")
    unknown = set(out["frozen"].index) - set(TEX)
    if unknown:
        raise SystemExit(f"no TeX name for {unknown}")
    return out


def c2st():
    df = pd.read_csv(PROBE / "compare/c2st_summary.csv").set_index("subset")
    vals = {}
    for s in ("joint", "identified"):
        r = df.loc[s]
        vals.update({f"{s}_dims": int(r.n_dims), f"{s}_arms": f3(r.c2st_arms),
                     f"{s}_null": f3(r.c2st_null), f"{s}_ex": f"{r.excess:+.3f}",
                     f"{s}_se": f3(r.se), f"{s}_z": f"{r.z:.1f}"})
    vals["Nc"] = int(df.loc["joint", "n_datasets"])
    return vals


def per_arm_counts():
    out = {}
    for arm in ARMS:
        d = np.load(PROBE / arm / "data.npz", allow_pickle=True)
        out[arm] = (d["true_params"].shape[0], d["pred_params"].shape[1])
    if out["frozen"] != out["baseline"]:
        raise SystemExit(f"arms validated on different sizes: {out}")
    return {"Nv": out["frozen"][0], "Dv": out["frozen"][1]}


def shared(names):
    """Shared-validation counts, identified cells and the paired log density."""
    s = np.load(PROBE / "compare/shared_validation.npz")
    t, dr_f, dr_b = s["true_params"], s["draws_frozen"], s["draws_baseline"]

    def contraction(draws):      # addm_arm_compare.contraction
        return 1.0 - (draws.std(axis=1) ** 2).mean(axis=0) / t.var(axis=0)

    c_f, c_b = contraction(dr_f), contraction(dr_b)
    ident = [n for j, n in enumerate(names) if min(c_f[j], c_b[j]) >= IDENTIFIED_THRESHOLD]
    dropped = [n for n in names if n not in ident]
    lp_f, lp_b = s["logprob_frozen"], s["logprob_baseline"]
    d = lp_f - lp_b
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {"Ns": t.shape[0], "Ds": dr_f.shape[1], "ident_n": len(ident),
            "dropped": " and ".join(TEX[n] for n in dropped),
            "lp_f": f2(lp_f.mean()), "lp_b": f2(lp_b.mean()),
            "lp_d": f"{d.mean():+.2f}", "lp_se": f2(se), "lp_z": f"{d.mean() / se:+.1f}",
            "lp_share": f"{100 * (d > 0).mean():.0f}"}


def representation():
    r = np.load(REP / "intercept_only_vsvfixed/summaries.npz", allow_pickle=True)
    fro, lea = r["frozen"], r["learned"]
    nr, nc = r["null_rsa"], r["null_cka"]
    cka = float(r["cka"])
    return {"C": fro.shape[0], "R": fro.shape[1], "df": fro.shape[2], "dl": lea.shape[2],
            "rsa": f3(float(r["rsa"])), "cka": f3(cka), "n_null": len(nr),
            "nr_m": f3(nr.mean()), "nr_s": f3(nr.std()), "nc_m": f3(nc.mean()),
            "nc_s": f3(nc.std()),
            # plot_cka_vs_null's z: population sd, as printed in that figure's title
            "cka_z": f"{(cka - nc.mean()) / (nc.std() + 1e-12):+.2f}"}


def crossfamily():
    g = np.load(REP / "crossfamily/app3_summaries.npz")
    fams = ["ddm", "rdm", "cdm", "addm", "lba"]
    shapes = {g[f].shape[:2] for f in fams} | {g[f + "__finetuned"].shape[:2] for f in fams}
    if len(shapes) != 1:
        raise SystemExit(f"families differ in conditions x repeats: {shapes}")
    (c, rep), = shapes
    return {"Cg": c, "Rg": rep}


# ── main-text figure ─────────────────────────────────────────────────────────
MAIN = (
    r"\textbf{{A frozen CogFormer encoder as the summary network for a held-out model.}} "
    r"\emph{{Top:}} parameter recovery for the attentional DDM (aDDM; five free parameters), "
    r"a family the encoder never saw, by two amortized posteriors that share the same "
    r"FlowMatching network and training budget and differ only in the summary: a "
    r"SetTransformer learned for this task (upper row, light) and CogFormer's frozen encoder "
    r"(lower row, dark). Points are posterior medians against ground truth with 95\% credible "
    r"intervals, on the unconstrained scale, for {Nv} validation datasets per arm; $r$ is "
    r"their correlation. $\omega$ is barely identified in this design (posterior contraction "
    r"{om_b} and {om_f} in the two arms). \emph{{Bottom left:}} two-dimensional MDS of the frozen encoder's "
    r"summaries of {Cg} datasets per family, for the three training families (circles) and "
    r"the two held-out ones (diamonds), before and after fine-tuning on the LBA "
    r"(Experiment~3); one joint fit, so both panels share axes, with per-family density "
    r"contours. \emph{{Bottom right:}} how the two summaries organise the same {C} aDDM "
    r"datasets: normalised Gram matrices (pairwise similarity between datasets) of the "
    r"learned and the frozen summary, their difference, and their off-diagonal entries "
    r"against each other, with linear CKA. Details in Appendix~\ref{{{DETAILS}}}."
)
DETAILS = (
    r"\emph{{Probe.}} The aDDM was not in the encoder's training mix, and its attentional "
    r"weight $\omega$ acts through a within-trial fixation process for which the encoder has "
    r"no input channel. Gaze always starts on item~0, which turns $\omega$ "
    r"into a choice bias, and with an intercept-only design and constant item values the "
    r"drift $v$, drift advantage $v_{{\mathrm{{diff}}}}$ and $\omega$ are not jointly "
    r"identified, so $v$ and the drift variability $s_v$ are fixed at their prior medians "
    r"($v = {V}$, $s_v = {SV}$ on the natural scale), leaving $v_{{\mathrm{{diff}}}}$, $\omega$, the threshold $a$, "
    r"non-decision time $\tau$ and its variability $s_\tau$. The \emph{{frozen}} arm takes the "
    r"Experiment~1 network's encoder (trained on DDM, RDM and CDM) with its weights frozen and "
    r"uses the mean of its output tokens ({df} dimensions) as the summary; the \emph{{baseline}} "
    r"arm learns a BayesFlow SetTransformer summary ({dl} dimensions) jointly with its "
    r"posterior. Both arms read the same per-trial input, use a FlowMatching posterior at "
    r"library defaults and train for {EP} epochs of {ST} batches of {BA}, i.e.\ on the same "
    r"{NSIM} simulated datasets of 500 trials; they are matched on data budget and input, not "
    r"on parameter count. Recovery (Figure~\ref{{fig:exp4-main}}, top) uses each arm's own "
    r"{Nv} validation datasets with {Dv} posterior draws, and the two rows share each "
    r"column's axis limits. Calibration does not separate the arms "
    r"(Figures~\ref{{fig:app4-arm-baseline}} and~\ref{{fig:app4-arm-frozen}}); a paired "
    r"comparison on shared datasets does (Figure~\ref{{fig:app4-compare}}). "
    r"\emph{{Geometry.}} Each family is simulated with an intercept-only design and 500 "
    r"trials: {Cg} parameter draws, each simulated {Rg} times independently, encoded by the "
    r"original and by the LBA-fine-tuned encoder (Experiment~3, {FT} epochs on all four "
    r"families) on the identical datasets. The aDDM is the five-parameter variant above and "
    r"the LBA has its start-point range fixed at $A = {A}$. A point is one parameter draw's "
    r"summary averaged over its repeats; dimensions are standardised over the pooled set of "
    r"both encoders, and a single metric MDS is fitted to all points and then split by "
    r"encoder. Contours are Gaussian kernel density estimates, drawn from 22\% of each "
    r"family's peak. "
    r"\emph{{Similarity.}} The Gram block uses {C} aDDM parameter draws ({R} repeats each, "
    r"averaged): each summary's centred Gram matrix is scaled to unit norm over its "
    r"off-diagonal entries with the diagonal set to zero, and both are divided by one common "
    r"constant so that they lie in $[-1, 1]$; the difference is SetTransformer minus "
    r"CogFormer. CKA is linear CKA with the unbiased HSIC estimator, $\mathrm{{CKA}} = {cka}$. "
    r"The summaries differ in dimension ({df} vs {dl}), so for scale: random linear "
    r"projections of the frozen summary itself to {dl} dimensions, which keep only the frozen "
    r"encoder's own geometry, score ${nc_m} \pm {nc_s}$ ({n_null} projections; "
    r"Figure~\ref{{fig:app4-rsa}}). "
    # TODO(verify): cite Kornblith et al. (2019) for CKA and Song et al. (2012) for unbiased
    # HSIC; Krajbich et al. (2010) for the aDDM.
)

# ── appendix ─────────────────────────────────────────────────────────────────
ARM_FULL = (
    r"\textbf{{aDDM, {ARM}: diagnostics.}} (a) Recovery: posterior medians against ground "
    r"truth with central 95\% credible intervals, $r$ their correlation; (b) calibration: "
    r"rank-ECDF difference (empirical minus uniform) with simultaneous 95\% bands; "
    r"(c) coverage: empirical against nominal coverage of central credible intervals; "
    r"(d) posterior $z$-score against posterior contraction, one point per dataset. "
    r"{Nv} validation datasets of 500 trials with {Dv} posterior draws each, on the "
    r"unconstrained scale. Contraction (NRMSE) per parameter: {METRICS}. $\omega$'s posterior "
    r"stays close to its prior, so its calibration is almost automatic; the informative tests "
    r"are the contracted parameters. Figure~\ref{{fig:app4-arm-frozen}} has the same layout "
    r"for the frozen arm."
)
ARM_SHORT = (
    r"\textbf{{aDDM, {ARM}: diagnostics.}} (a) Recovery, (b) calibration, (c) coverage, "
    r"(d) $z$-score against contraction, as in Figure~\ref{{fig:app4-arm-baseline}}. "
    r"Contraction (NRMSE): {METRICS}. Each arm is validated on its own simulated datasets, "
    r"so the two figures are not paired; Figure~\ref{{fig:app4-compare}} compares the arms "
    r"on shared data."
)
COMPARE = (
    r"\textbf{{aDDM: the two arms on shared data.}} Both trained posteriors are applied to "
    r"one set of {Ns} simulated datasets (500 trials), with {Ds} draws per arm. "
    r"(a, b) Per-dataset classifier two-sample test (C2ST) on {Nc} of these datasets: "
    r"held-out accuracy of a classifier (MLP, 5-fold cross-validation) trained to tell the "
    r"frozen arm's draws from the baseline's (purple), against an empirical null from two "
    r"independent sampling calls of the frozen arm (grey), which carries the same "
    r"finite-sample bias; 0.5 (dashed) means indistinguishable. (a) Jointly over all "
    r"{joint_dims} parameters: mean {joint_arms} against {joint_null}, excess "
    r"${joint_ex} \pm {joint_se}$ ($z = {joint_z}$). (b) Over the {identified_dims} identified "
    r"parameters (contraction at least 0.5 in both arms; {dropped} excluded), where agreement "
    r"cannot be inherited from a shared prior: {identified_arms} against {identified_null}, "
    r"excess ${identified_ex} \pm {identified_se}$ ($z = {identified_z}$). (c) Log "
    r"posterior density of the true parameters, frozen against baseline, paired over the "
    r"{Ns} datasets (dashed: equality): mean {lp_f} against {lp_b}, paired difference "
    r"${lp_d} \pm {lp_se}$ ($z = {lp_z}$; frozen higher on {lp_share}\% of datasets). C2ST "
    r"is computed per dataset because pooled over datasets, two calibrated posteriors both "
    r"average to the prior and the test reads 0.5 whatever their differences."
)
RSA = (
    r"\textbf{{aDDM: representational similarity of the two summaries.}} (a) "
    r"Representational dissimilarity matrices over {C} aDDM parameter draws, each simulated "
    r"{R} times independently (500 trials): the learned SetTransformer summary ({dl} "
    r"dimensions), the frozen CogFormer encoder ({df}), their difference (SetTransformer minus "
    r"CogFormer), and their off-diagonal entries against each other. Dissimilarities are "
    r"crossnobis distances, cross-validated over the {R} repeats so that measurement noise "
    r"does not inflate them. The two summaries' distances are on different scales, so each "
    r"matrix is z-scored over its off-diagonal entries (the "
    r"normalisation Pearson correlation applies) and both are divided by one common constant; "
    r"values are therefore signed and read as more or less dissimilar than typical, not as "
    r"distances from zero. $r = {rsa}$ is the Pearson correlation of the two off-diagonal "
    r"vectors. (b) The observed RSA against the reference labelled ``matched null'': the "
    r"frozen RDM compared with the RDMs of {n_null} random linear projections of the frozen "
    r"summary itself to {dl} dimensions (${nr_m} \pm {nr_s}$). These projections keep the "
    r"frozen encoder's own geometry up to compression loss, so the reference marks what a "
    r"{dl}-dimensional linear readout of the frozen representation scores, not the absence of "
    r"shared structure. Observed and reference CKA are marked for comparison. (c) The same "
    r"for linear CKA: observed {cka} against ${nc_m} \pm {nc_s}$ ($z = {cka_z}$). The CKA "
    r"Gram block is in Figure~\ref{{fig:exp4-main}} (bottom right)."
)


def metric_list(m):
    return ", ".join(f"{TEX[p]} {f2(r.contraction)} ({f2(r.nrmse)})" for p, r in m.iterrows())


def stacked(arm, caption):
    d = f"probe/vsvfixed/{arm}"
    parts = (("recovery", "Recovery"), ("calibration_ecdf", "Calibration (ECDF)"),
             ("coverage", "Coverage"), ("z_score_contraction", r"$z$-score vs contraction"))
    blocks = [r"\begin{figure}[p]", r"  \centering"]
    for i, (f, name) in enumerate(parts):
        blocks += [r"  \begin{subfigure}{\textwidth}\centering",
                   rf"    \includegraphics[width=\textwidth]{{{d}/{f}.pdf}}",
                   rf"    \caption{{{name}}}",
                   r"  \end{subfigure}" + (r"\\[1ex]" if i < len(parts) - 1 else "")]
    return "\n".join(blocks + [rf"  \caption{{{caption}}}", rf"  \label{{fig:app4-arm-{arm}}}",
                               r"\end{figure}"])


def single(path, caption, lab):
    return "\n".join([r"\begin{figure}[p]", r"  \centering",
                      rf"  \includegraphics[width=\textwidth]{{{path}}}",
                      rf"  \caption{{{caption}}}", rf"  \label{{{lab}}}", r"\end{figure}"])


def rsa_figure(caption):
    d = "representation/intercept_only_vsvfixed"
    return "\n".join([
        r"\begin{figure}[p]", r"  \centering",
        r"  \begin{subfigure}{\textwidth}\centering",
        rf"    \includegraphics[width=\textwidth]{{{d}/rdm_rsa.pdf}}",
        r"    \caption{}", r"  \end{subfigure}\\[1ex]",
        r"  \begin{subfigure}{0.48\textwidth}\centering",
        rf"    \includegraphics[width=\textwidth]{{{d}/rsa_vs_null.pdf}}",
        r"    \caption{}", r"  \end{subfigure}\hfill",
        r"  \begin{subfigure}{0.48\textwidth}\centering",
        rf"    \includegraphics[width=\textwidth]{{{d}/cka_vs_null.pdf}}",
        r"    \caption{}", r"  \end{subfigure}",
        rf"  \caption{{{caption}}}", r"  \label{fig:app4-rsa}", r"\end{figure}"])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--draft", action="store_true",
                   help="Render with ?? for values whose source file is absent (review only).")
    args = p.parse_args()

    needed = [PANEL, "probe/vsvfixed/compare/intercept_only_vsvfixed.pdf"] \
        + [f"probe/vsvfixed/{a}/{f}.pdf" for a in ARMS
           for f in ("recovery", "calibration_ecdf", "coverage", "z_score_contraction")] \
        + [f"representation/intercept_only_vsvfixed/{f}.pdf"
           for f in ("rdm_rsa", "rsa_vs_null", "cka_vs_null")]
    for f in needed:
        if not (FINALS / f).exists():
            raise SystemExit(f"missing {FINALS / f} -- run e4_finalize first")

    m = arm_metrics()
    names = list(m["frozen"].index)
    v = Facts(args.draft)
    v.update(c2st())
    v.update(V=f"{V_FIXED:g}", SV=f"{S_V_FIXED:g}", EP=EPOCHS, ST=STEPS, BA=BATCH,
             NSIM=f"{EPOCHS * STEPS * BATCH / 1e6:g}~million",
             FT=int(shared_config.get("LBA_FT_EPOCHS")), A=shared_config.get("LBA_FIXED_A"),
             om_b=f2(m["baseline"].loc["omega", "contraction"]),
             om_f=f2(m["frozen"].loc["omega", "contraction"]), DETAILS=DETAILS_LABEL)
    v.load(["Nv", "Dv"], per_arm_counts, "probe/vsvfixed/<arm>/data.npz")
    v.load(["Ns", "Ds", "ident_n", "dropped", "lp_f", "lp_b", "lp_d", "lp_se", "lp_z",
            "lp_share"], lambda: shared(names), "probe/vsvfixed/compare/shared_validation.npz")
    v.load(["C", "R", "df", "dl", "rsa", "cka", "n_null", "nr_m", "nr_s", "nc_m", "nc_s",
            "cka_z"], representation, "representation/intercept_only_vsvfixed/summaries.npz")
    v.load(["Cg", "Rg"], crossfamily, "representation/crossfamily/app3_summaries.npz")
    if v["ident_n"] != MISSING and v["ident_n"] != v["identified_dims"]:
        raise SystemExit(f"{v['ident_n']} identified cells recomputed, c2st_summary.csv has "
                         f"{v['identified_dims']}")

    header = ("% Generated by `python -m experiments.experiment_4.appendix_captions` -- "
              "do not edit by hand.\n")
    if args.draft:
        header += "% DRAFT: ?? marks values whose source file was absent. Do not commit.\n"
    (FINALS / "main_figure.tex").write_text("\n".join([
        header + "% Main-text figure for Experiment 4. Needs \\usepackage{graphicx};"
        " paths are relative to outputs/finals/experiment-4/.",
        r"\begin{figure}[tbp]", r"  \centering",
        rf"  \includegraphics[width=\textwidth]{{{PANEL}}}",
        rf"  \caption{{{MAIN.format(**v)}}}",
        r"  \label{fig:exp4-main}", r"\end{figure}", ""]))
    (FINALS / "main_figure_details.tex").write_text("\n".join([
        header + "% Methods/appendix text for fig:exp4-main (main_figure.tex). Place it under a heading",
        f"% carrying the label, e.g. \\subsection{{Encoder probe: details}}\\label{{{DETAILS_LABEL}}}",
        "% TODO(verify): Kornblith et al. (2019) CKA; Song et al. (2012) unbiased HSIC;"
        " Krajbich et al. (2010) aDDM.",
        DETAILS.format(**v), ""]))

    blocks = [header + "% Appendix figures for Experiment 4. Needs \\usepackage{graphicx,subcaption};"
              " paths are relative to outputs/finals/experiment-4/.", ""]
    blocks += [stacked("baseline", ARM_FULL.format(ARM=ARMS["baseline"],
                                                   METRICS=metric_list(m["baseline"]), **v)), ""]
    blocks += [stacked("frozen", ARM_SHORT.format(ARM=ARMS["frozen"],
                                                  METRICS=metric_list(m["frozen"]), **v)), ""]
    blocks += [single("probe/vsvfixed/compare/intercept_only_vsvfixed.pdf",
                      COMPARE.format(**v), "fig:app4-compare"), ""]
    blocks += [rsa_figure(RSA.format(**v)), ""]
    (FINALS / "captions.tex").write_text("\n".join(blocks))
    print(f"[saved] {FINALS}/main_figure.tex, main_figure_details.tex, captions.tex (4 figures)")
    if v.missed:
        print("DRAFT -- unreadable, rendered as ??:\n  " + "\n  ".join(v.missed), file=sys.stderr)


if __name__ == "__main__":
    main()
