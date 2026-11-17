"""Combine Experiment 1's saved panels into one vector PDF.

The random-design maps and coefficient recovery (A/B) span the top row; pi
recovery (C) and the information-gain comparison (D) share the bottom row.
Existing footer letters are clipped off and replaced with consistent labels in
the panels' upper-left margins. No sampling or rasterization is involved.

    python -m experiments.experiment_1.combined_figure

Uses PyMuPDF to preserve the panels' vector graphics.
"""
import argparse
from contextlib import ExitStack
from pathlib import Path

import pymupdf

from experiments.paths import RESULTS_ROOT

PANEL_FILES = ("random_ensemble/recovery.pdf", "random_ensemble/recovery_pi.pdf",
               "information_gain/information_gain.pdf")


def _panel(page, letters):
    """Content rectangle and footer-letter positions in an existing panel PDF."""
    markers = {}
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if span["text"] in letters and "bold" in span["font"].lower():
                    if span["text"] in markers:
                        raise ValueError(f"duplicate panel label {span['text']}")
                    markers[span["text"]] = pymupdf.Rect(span["bbox"])
    if set(markers) != set(letters):
        raise ValueError(f"expected footer labels {letters}, found {list(markers)}")
    clip = pymupdf.Rect(page.rect)
    clip.y1 = min(box.y0 for box in markers.values()) - 1
    return clip, markers


def combine(finals, out, width=864):
    """Place saved PDFs without changing their drawing commands or aspect ratios."""
    margin, label_gutter, column_gap, row_gap = 12, 18, 20, 22
    font_size = 14
    with ExitStack() as stack:
        top, pi, eig = [stack.enter_context(pymupdf.open(finals / path))
                        for path in PANEL_FILES]
        top_clip, top_letters = _panel(top[0], "AB")
        pi_clip, _ = _panel(pi[0], "C")
        eig_clip, _ = _panel(eig[0], "D")

        top_width = width - 2 * margin - label_gutter
        top_height = top_width * top_clip.height / top_clip.width
        bottom_width = width - 2 * margin - 2 * label_gutter - column_gap
        pi_ratio = pi_clip.width / pi_clip.height
        eig_ratio = eig_clip.width / eig_clip.height
        bottom_height = bottom_width / (pi_ratio + eig_ratio)
        pi_width, eig_width = bottom_height * pi_ratio, bottom_height * eig_ratio
        bottom_y = margin + top_height + row_gap
        eig_x = margin + label_gutter + pi_width + column_gap
        height = bottom_y + bottom_height + margin

        result = stack.enter_context(pymupdf.open())
        page = result.new_page(width=width, height=height)
        top_rect = pymupdf.Rect(margin + label_gutter, margin,
                               width - margin, margin + top_height)
        pi_rect = pymupdf.Rect(margin + label_gutter, bottom_y,
                              margin + label_gutter + pi_width, bottom_y + bottom_height)
        eig_rect = pymupdf.Rect(eig_x + label_gutter, bottom_y,
                               eig_x + label_gutter + eig_width, bottom_y + bottom_height)
        for target, document, clip in ((top_rect, top, top_clip),
                                       (pi_rect, pi, pi_clip), (eig_rect, eig, eig_clip)):
            page.show_pdf_page(target, document, 0, clip=clip)

        # B has a natural gutter to the left of the grid; A/C/D get an explicit
        # gutter so their letters sit next to the content without covering it.
        b_x = top_rect.x0 + top_letters["B"].x0 * top_width / top_clip.width
        for letter, x, y in (("A", margin, margin), ("B", b_x, margin),
                              ("C", margin, bottom_y), ("D", eig_x, bottom_y)):
            page.insert_text((x, y + font_size), letter,
                             fontname="hebo", fontsize=font_size)

        result.set_metadata({"title": "CogFormer: amortized inference across design space",
                             "subject": "A: design maps; B: coefficient recovery; "
                                        "C: contamination recovery; D: information gain and NRMSE"})
        out.parent.mkdir(parents=True, exist_ok=True)
        result.save(out, garbage=4, deflate=True)
    return width, height


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--finals", type=Path, default=RESULTS_ROOT / "finals/experiment-1")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--width", type=float, default=864, help="Page width in PDF points.")
    args = parser.parse_args()
    out = args.out or args.finals / "main_figure.pdf"
    width, height = combine(args.finals, out, args.width)
    print(f"[saved] {out} ({width / 72:.2f} x {height / 72:.2f} in)")


if __name__ == "__main__":
    main()
