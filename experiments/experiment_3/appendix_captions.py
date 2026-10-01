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
           r"encoding as in panel {PANEL} of Figure~\ref{{{REF}}}.")

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

    OUT.write_text("\n".join(blocks))
    print(f"[saved] {OUT} ({sum(b.startswith(chr(92) + 'begin{figure}') for b in blocks)} figures)")


if __name__ == "__main__":
    main()
