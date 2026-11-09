"""Figures + captions for Experiment 2 (real data, posterior predictive checks).

Writes, under outputs/finals/experiment-2/:

- main_figure.tex: the combined PPC of all three families (root combined_ppc.pdf), short
  caption;
- main_figure_details.tex: the methods/appendix paragraph that caption refers to;
- captions.tex: appendix figures, one per family pairing its two grouping variables as
  (a, b): the RT distributions of DDM, RDM and CDM, and CDM's response-angle
  distribution. The first figure carries the full caption, the others a short one.

Facts follow experiments/experiment_2/ppc.py and data_utils.py: every posterior draw
simulates each subject's own trials and design (contamination included), the simulated
trials are pooled across subjects; densities are Silverman Gaussian KDEs; bands are the
central PPC_CI interval across draws; quantile panels plot 0.1/0.3/0.5/0.7/0.9; the CDM
response deviation is the doubled raw half-angle. Subject, trial and draw counts are read
from the data files and the saved posteriors, so the text cannot drift from the fit.

    python -m experiments.experiment_2.appendix_captions
"""
from experiments.paths import INPUT_DATA_ROOT, RESULTS_ROOT
from pathlib import Path

import numpy as np

from experiments.experiment_2 import ppc
from experiments.experiment_2.data_utils import load_cdm_data, load_ddm_data, load_rdm_data

FINALS = (RESULTS_ROOT / "finals/experiment-2")
FAMILIES = {   # family: (loader, data file, grouping variables in figure order)
    "ddm": (load_ddm_data, str(INPUT_DATA_ROOT / "data_application_ddm.csv"), ("stim_type", "truth")),
    "rdm": (load_rdm_data, str(INPUT_DATA_ROOT / "data_application_rdm.csv"), ("congruency", "session")),
    "cdm": (load_cdm_data, str(INPUT_DATA_ROOT / "data_application_cdm.csv"), ("difficulty", "speed")),
}
NAMES = {"ddm": "DDM", "rdm": "RDM", "cdm": "CDM"}
COND_NAMES = {"stim_type": "stimulus type", "truth": "factual truth", "congruency": "congruency",
              "session": "session", "difficulty": "difficulty", "speed": "speed instruction"}
DETAILS_LABEL = "app:exp2-main-details"


def counts():
    """Per family: subjects, trials per subject (min, median, max), posterior draws."""
    out = {}
    for fam, (loader, path, _) in FAMILIES.items():
        data, ids = loader(path)
        n_trials = np.array([len(data[s]["rts"]) for s in ids])
        post = np.load(FINALS / "posteriors" / f"{fam}.npz", allow_pickle=True)["posteriors"]
        if post.shape[0] != len(ids):
            raise SystemExit(f"{fam}: {post.shape[0]} fitted subjects but the loader yields {len(ids)}")
        out[fam] = dict(n=len(ids), tmin=int(n_trials.min()), tmed=int(np.median(n_trials)),
                        tmax=int(n_trials.max()), draws=post.shape[1])
    draws = {c["draws"] for c in out.values()}
    if len(draws) != 1:
        raise SystemExit(f"posterior draw counts differ across families: {draws}")
    return out


MAIN = (
    r"\textbf{{Posterior predictive checks on real data.}} "
    r"Observed data against the posterior predictive of CogFormer's fits, pooled over "
    r"subjects: DDM ({N_ddm} subjects; top row), RDM ({N_rdm}; second row) and CDM ({N_cdm}; "
    r"bottom two rows). Quantile panels show the 0.1, 0.3, 0.5, 0.7 and 0.9 quantiles of RT "
    r"(and, for the CDM, of the continuous response angle $\theta$, bottom row) for each level "
    r"of each experimental condition: observed (black squares, dashed) against the predictive "
    r"median (coloured line) with its {CI}\% interval (band). The $2 \times 2$ blocks on the "
    r"right compare, per condition level, the observed proportion of each response (bars; DDM: "
    r"the two responses, RDM: the three colour responses) with the predictive median and "
    r"{CI}\% interval (points). Dashed rules separate conditions within a family, solid rules the "
    r"families. Details in Appendix~\ref{{{DETAILS}}}."
)
DETAILS = (
    r"The posterior predictive in Figure~\ref{{fig:exp2-main}} is built from {D} posterior "
    r"draws per subject. Each draw simulates that subject's own trials under their own design "
    r"(the contamination process included), and the simulated trials are pooled over subjects, "
    r"giving one predictive data set per draw; quantiles and choice proportions are computed per "
    r"data set, and the band is their central {CI}\% interval across draws. Data: DDM, "
    r"Stump et al. (2026), {N_ddm} subjects with at least 100 trials ({T_ddm} trials per "
    r"subject), regressors stimulus type and factual truth (effect-coded $\pm 0.5$) and their "
    r"interaction; RDM, a three-alternative Stroop task, {N_rdm} subjects ({T_rdm} trials), "
    r"congruency and session ($\pm 0.5$) and their interaction, with stimulus-coded responses "
    r"(the ink colour is the favoured accumulator); CDM, {N_cdm} subjects ({T_cdm} trials), speed instruction ($\pm 0.5$) and "
    r"three difficulty levels (dummy-coded against the easiest) with their interaction. The "
    r"CDM's recorded response deviation spans $[-\pi/2, \pi/2]$ and is doubled to the model's "
    r"response angle on $[-\pi, \pi]$, i.e.\ treated as a half-angle. "
    # TODO(verify): RDM dataset citation; Kvam (2019) for the CDM data / response apparatus.
)

APP_FULL = (
    r"\textbf{{RT distributions, {FAM}: posterior predictive check.}} Pooled over the "
    r"{N} subjects, by (a) {C1} and (b) {C2}: the observed RT density (black) against the "
    r"posterior predictive of CogFormer's fit, one predictive data set per posterior draw "
    r"({D} draws, each simulating every subject's own trials and design, then pooled), shown "
    r"as a subset of individual draws (thin lines), their median (dark line) and the central "
    r"{CI}\% band. Densities are Gaussian kernel density estimates (Silverman bandwidth), "
    r"plotted between the 1st and 99th percentiles of the observed RTs. All subsequent figures "
    r"in this section (Figures~\ref{{{NEXT}}}--\ref{{{LAST}}}) follow this layout and encoding."
)
APP_SHORT = (r"\textbf{{RT distributions, {FAM}: posterior predictive check.}} Pooled over the "
             r"{N} subjects, by (a) {C1} and (b) {C2}. Encoding as in Figure~\ref{{{FIRST}}}.")
APP_ANGLE = (
    r"\textbf{{Response-angle distributions, CDM: posterior predictive check.}} Pooled over "
    r"the {N} subjects, by (a) {C1} and (b) {C2}: the density of the response deviation "
    r"$\theta$ (radians, on the model's $[-\pi, \pi]$ scale), encoded as in "
    r"Figure~\ref{{{FIRST}}}. Dotted lines mark $\theta = \pm\pi/2$; each panel reports the "
    r"observed and predictive share of responses beyond them, $P(|\theta| > \pi/2)$."
)


def label(fam, kind):
    return f"fig:app2-{fam}-{kind}"


def paired_figure(paths, caption, lab):
    blocks = [r"\begin{figure}[p]", r"  \centering"]
    for i, p in enumerate(paths):
        blocks += [r"  \begin{subfigure}{\textwidth}\centering",
                   rf"    \includegraphics[width=\textwidth]{{{p}}}",
                   r"    \caption{}",
                   r"  \end{subfigure}" + (r"\\[1ex]" if i < len(paths) - 1 else "")]
    return "\n".join(blocks + [rf"  \caption{{{caption}}}", rf"  \label{{{lab}}}", r"\end{figure}"])


def main():
    c = counts()
    ci = f"{ppc.PPC_CI:g}"
    trials = {f: f"{v['tmin']}--{v['tmax']}, median {v['tmed']}" for f, v in c.items()}
    vals = dict(CI=ci, D=c["ddm"]["draws"], **{f"N_{f}": v["n"] for f, v in c.items()},
                **{f"T_{f}": t for f, t in trials.items()})
    needed = ["combined_ppc.pdf"] + [f"{f}/rt_ppc_{x}.pdf" for f, (_, _, cs) in FAMILIES.items() for x in cs] \
        + [f"cdm/choice_ppc_{x}.pdf" for x in FAMILIES["cdm"][2]]
    for p in needed:
        if not (FINALS / p).exists():
            raise SystemExit(f"missing {FINALS / p} -- run e2_finalize first")

    header = "% Generated by `python -m experiments.experiment_2.appendix_captions` -- do not edit by hand.\n"
    (FINALS / "main_figure.tex").write_text("\n".join([
        header + "% Main-text figure for Experiment 2. Needs \\usepackage{graphicx};"
        " paths are relative to outputs/finals/experiment-2/.",
        r"\begin{figure}[tbp]", r"  \centering",
        r"  \includegraphics[width=\textwidth]{combined_ppc.pdf}",
        rf"  \caption{{{MAIN.format(**vals, DETAILS=DETAILS_LABEL)}}}",
        r"  \label{fig:exp2-main}", r"\end{figure}", ""]))
    (FINALS / "main_figure_details.tex").write_text("\n".join([
        header + "% Methods/appendix text for fig:exp2-main (main_figure.tex). Place it under a heading",
        f"% carrying the label, e.g. \\subsection{{Posterior predictive checks: details}}\\label{{{DETAILS_LABEL}}}",
        "% TODO(verify): RDM dataset citation; Kvam (2019) for the CDM data and response apparatus.",
        DETAILS.format(**vals), ""]))

    order = [(f, "rt") for f in FAMILIES] + [("cdm", "angle")]
    blocks = [header + "% Appendix figures for Experiment 2. Needs \\usepackage{graphicx,subcaption};"
              " paths are relative to outputs/finals/experiment-2/.", ""]
    first = label(*order[0])
    for k, (f, kind) in enumerate(order):
        c1, c2 = FAMILIES[f][2]
        fmt = dict(FAM=NAMES[f], N=c[f]["n"], C1=COND_NAMES[c1], C2=COND_NAMES[c2], D=c[f]["draws"],
                   CI=ci, FIRST=first)
        if kind == "angle":
            cap, paths = APP_ANGLE.format(**fmt), [f"cdm/choice_ppc_{x}.pdf" for x in (c1, c2)]
        else:
            cap = (APP_FULL.format(**fmt, NEXT=label(*order[1]), LAST=label(*order[-1])) if k == 0
                   else APP_SHORT.format(**fmt))
            paths = [f"{f}/rt_ppc_{x}.pdf" for x in (c1, c2)]
        blocks += [paired_figure(paths, cap, label(f, kind)), ""]
    (FINALS / "captions.tex").write_text("\n".join(blocks))
    print(f"[saved] {FINALS}/main_figure.tex, main_figure_details.tex, captions.tex ({len(order)} figures)")


if __name__ == "__main__":
    main()
