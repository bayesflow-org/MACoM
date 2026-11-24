"""Place A--D in the bottom-left margins beside Experiment 4's plot labels.

    python -m experiments.experiment_4.panel_labels
"""
import argparse
from pathlib import Path

import pymupdf

from experiments.paths import RESULTS_ROOT


def _vertical_label_x(page, text):
    boxes = [box for box in page.search_for(text) if box.height > box.width]
    if not boxes or max(box.x0 for box in boxes) - min(box.x0 for box in boxes) > 0.01:
        raise ValueError(f"expected aligned vertical {text!r} labels, found {len(boxes)}")
    return min(box.x0 for box in boxes)


def _label(path, representation=False):
    """Align letters with x-axis labels, keeping plots and legends unobstructed."""
    path = Path(path)
    suffix = "BCD" if representation else "A"
    integrated_marker = "exp4-integrated-labels-" + suffix
    marker = integrated_marker + "-v2"
    old_marker = "exp4-bottom-labels-" + suffix
    with pymupdf.open(path) as doc:
        if len(doc) != 1:
            raise ValueError(f"{path}: expected one page")
        if marker in (doc.metadata.get("keywords") or ""):
            return
        page = doc[0]
        keywords = doc.metadata.get("keywords") or ""
        if integrated_marker in keywords:
            # Earlier integrated labels were appended as separate text streams.
            # Keep the original panel form and replace only those label streams.
            streams = page.get_contents()
            expected = 4 if representation else 2
            if len(streams) != expected or any(b"/hebo " not in doc.xref_stream(x)
                                               for x in streams[1:]):
                raise ValueError(f"{path}: unexpected existing label streams")
            page.set_contents(streams[0])
            keywords = keywords.replace(integrated_marker, "")
        def xlabel_y(text):
            boxes = [box for box in page.search_for(text) if box.width > box.height]
            if not boxes:
                raise ValueError(f"expected a horizontal {text!r} label")
            return max(box.y1 for box in boxes) - 3

        positions = [("A", _vertical_label_x(page, "Estimate"), xlabel_y("Ground truth"))
                     ] if not representation else [
            ("B", _vertical_label_x(page, "MDS 2"), xlabel_y("MDS 1")),
            ("C", _vertical_label_x(page, "Simulated dataset"), xlabel_y("Simulated dataset")),
            ("D", _vertical_label_x(page, "Set Transformer"), xlabel_y("CogFormer")),
        ]
        clip = pymupdf.Rect(page.rect)
        metadata = dict(doc.metadata)
        if old_marker in keywords:
            clip.y1 -= 26
            keywords = keywords.replace(old_marker, "")
        metadata["keywords"] = (keywords.strip() + " " + marker).strip()
        with pymupdf.open() as result:
            labelled = result.new_page(width=clip.width, height=clip.height)
            labelled.show_pdf_page(pymupdf.Rect(clip), doc, 0, clip=clip)
            # Give the label font its own page resource, independent of the
            # fonts inside the imported panel's nested forms.
            labelled.insert_font(fontname="Exp4Label", fontbuffer=pymupdf.Font("hebo").buffer)
            for letter, x, y in positions:
                labelled.insert_text((x, y), letter, fontname="Exp4Label", fontsize=14)
            result.set_metadata(metadata)
            content = result.tobytes(garbage=4, deflate=True)
    path.write_bytes(content)


def label_panels(recovery, representation, combined=None):
    _label(recovery)
    _label(representation, representation=True)
    if combined is not None:
        with pymupdf.open(recovery) as top, pymupdf.open(representation) as bottom:
            with pymupdf.open() as doc:
                width = max(top[0].rect.width, bottom[0].rect.width)
                heights = [width * source[0].rect.height / source[0].rect.width
                           for source in (top, bottom)]
                page = doc.new_page(width=width, height=sum(heights) + 12)
                page.show_pdf_page(pymupdf.Rect(0, 0, width, heights[0]), top, 0)
                page.show_pdf_page(pymupdf.Rect(0, heights[0] + 12,
                                               width, sum(heights) + 12), bottom, 0)
                content = doc.tobytes(garbage=4, deflate=True)
        Path(combined).write_bytes(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--directory", type=Path,
                        default=RESULTS_ROOT / "finals/experiment-4/probe/compare")
    args = parser.parse_args()
    label_panels(args.directory / "panel_recovery.pdf",
                 args.directory / "panel_representation.pdf", args.directory / "panel.pdf")
    print(f"[labelled] {args.directory}: A (recovery), B (MDS), C (similarity), D (CKA)")


if __name__ == "__main__":
    main()
