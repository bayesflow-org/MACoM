"""Combine Experiment 3's saved panels into one vector PDF.

Recovery (A) and interaction metric deltas (B) share the top row at equal
height, preserving their aspect ratios. The empirical RT quantile row (C)
spans the bottom. The default PPC is the free-variability fit used by the
appendix captions; --ppc-variant conventional selects the constrained check.

    python -m experiments.experiment_3.combined_figure
"""
import argparse
from contextlib import ExitStack
from pathlib import Path

import pymupdf

from experiments.paths import RESULTS_ROOT

PANEL_FILES = ("compare/lba/interaction_overlay_recovery.pdf",
               "compare/delta_metrics_interaction.pdf")
PPC_VARIANTS = ("free", "conventional")


def panel_files(ppc_variant="free"):
    if ppc_variant not in PPC_VARIANTS:
        raise ValueError(f"unknown PPC variant: {ppc_variant}")
    return (*PANEL_FILES, f"ppc/{ppc_variant}/rt_quantiles_row.pdf")


def combine(finals, out, width=864, ppc_variant="free"):
    """Place the three existing PDFs with Experiment 1's margins and labels."""
    margin, label_gutter, column_gap, row_gap = 12, 18, 20, 22
    font_size = 14
    top_width = width - 2 * margin - 2 * label_gutter - column_gap
    if top_width <= 0:
        raise ValueError("page width must exceed the margins and panel gutters")
    finals, out = Path(finals), Path(out)
    with ExitStack() as stack:
        documents = [stack.enter_context(pymupdf.open(finals / path))
                     for path in panel_files(ppc_variant)]
        if any(len(doc) != 1 for doc in documents):
            raise ValueError("each input panel must have exactly one page")
        recovery, delta, rt = documents
        ratios = [doc[0].rect.width / doc[0].rect.height for doc in documents]
        top_height = top_width / sum(ratios[:2])
        recovery_width, delta_width = [top_height * r for r in ratios[:2]]
        delta_x = margin + label_gutter + recovery_width + column_gap
        bottom_y = margin + top_height + row_gap
        bottom_width = width - 2 * margin - label_gutter
        bottom_height = bottom_width / ratios[2]
        height = bottom_y + bottom_height + margin

        targets = (
            pymupdf.Rect(margin + label_gutter, margin,
                         margin + label_gutter + recovery_width, margin + top_height),
            pymupdf.Rect(delta_x + label_gutter, margin,
                         delta_x + label_gutter + delta_width, margin + top_height),
            pymupdf.Rect(margin + label_gutter, bottom_y,
                         width - margin, bottom_y + bottom_height),
        )
        result = stack.enter_context(pymupdf.open())
        page = result.new_page(width=width, height=height)
        for target, document in zip(targets, documents):
            page.show_pdf_page(target, document, 0)
        for letter, x, y in (("A", margin, margin), ("B", delta_x, margin),
                              ("C", margin, bottom_y)):
            page.insert_text((x, y + font_size), letter,
                             fontname="hebo", fontsize=font_size)
        result.set_metadata({
            "title": "CogFormer: transfer to the LBA model family",
            "subject": "A: zero-shot and fine-tuned recovery; B: interaction metric deltas; "
                       f"C: empirical RT quantiles ({ppc_variant} PPC)",
        })
        out.parent.mkdir(parents=True, exist_ok=True)
        result.save(out, garbage=4, deflate=True)
    return width, height


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--finals", type=Path, default=RESULTS_ROOT / "finals/experiment-3")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--width", type=float, default=864, help="Page width in PDF points.")
    parser.add_argument("--ppc-variant", choices=PPC_VARIANTS, default="free")
    args = parser.parse_args()
    out = args.out or args.finals / "main_figure.pdf"
    width, height = combine(args.finals, out, args.width, args.ppc_variant)
    print(f"[saved] {out} ({width / 72:.2f} x {height / 72:.2f} in)")


if __name__ == "__main__":
    main()
