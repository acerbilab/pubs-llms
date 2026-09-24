#!/usr/bin/env python3
"""Extract per-page text, page renders, and figure crops from a paper PDF.

Output layout under OUT (shared by all parts of one paper, e.g. main + supplement):

    OUT/<prefix>/text/pNN.txt     text layer per page (running headers/footers and
                                  soft hyphens removed)
    OUT/<prefix>/png/pNN.png      page renders
    OUT/figs/<prefix>_fig<L>.png  one crop per figure caption found (L zero-padded if numeric)
    OUT/<prefix>/figures.tsv      label, page, kind (raster/vector/page), crop rect

Running headers/footers are detected as lines that recur (digits normalized) among the
first/last few lines of many pages; add --strip-regex for anything the heuristic misses.
Requires PyMuPDF (`fitz`).
"""
import argparse
import re
from collections import Counter
from pathlib import Path

import fitz

DEFAULT_CAPTION = r"(?:Fig\.?|Figure)\s*(S?\d+|[A-Z]\d*)\s*[.:|]"
EDGE_LINES = 3


def norm(line):
    return re.sub(r"\d+", "#", line.strip())


PAGE_NUMBER_SHAPES = re.compile(r"(?i)^(#|#\s*/\s*#|#\s+of\s+#|page\s+#(\s+of\s+#)?)$")


def detect_running_lines(pages_lines, min_frac):
    counts = Counter()
    for lines in pages_lines:
        nonempty = [l for l in lines if l.strip()]
        edge = nonempty[:EDGE_LINES] + nonempty[-EDGE_LINES:]
        counts.update({norm(l) for l in edge})
    threshold = max(3, int(min_frac * len(pages_lines)))
    # Letterless lines (figure tick labels such as "-20") recur on figure pages too;
    # among those, only page-number shapes count as running lines.
    return {k for k, c in counts.items()
            if c >= threshold and k and (re.search(r"[^\W\d_#]", k) or PAGE_NUMBER_SHAPES.match(k))}


def clean_page(lines, running, strip_res):
    nonempty_idx = [i for i, l in enumerate(lines) if l.strip()]
    edge_idx = set(nonempty_idx[:EDGE_LINES] + nonempty_idx[-EDGE_LINES:])
    kept = []
    for i, l in enumerate(lines):
        if i in edge_idx and norm(l) in running:
            continue
        if any(r.search(l) for r in strip_res):
            continue
        kept.append(l)
    text = "\n".join(kept)
    return text.replace("­\n", "").replace("­", "")


def caption_blocks(page, cap_re):
    out = []
    for b in page.get_text("blocks"):
        m = cap_re.match(b[4].lstrip())
        if m:
            out.append((m.group(1), fitz.Rect(b[:4])))
    return out


def cluster_above(rects, cap, gap=18):
    """Grow a box from the rects nearest above the caption, adding neighbors within gap."""
    cand = [r for r in rects if r.y1 <= cap.y0 + 4]
    if not cand:
        return None
    cand.sort(key=lambda r: -r.y1)
    seed = [r for r in cand if r.y1 >= cand[0].y1 - 60]
    box = fitz.Rect(seed[0])
    for r in seed[1:]:
        box |= r
    changed = True
    while changed:
        changed = False
        grown = box + (-gap, -gap, gap, gap)
        for r in cand:
            if not box.contains(r) and grown.intersects(r):
                box |= r
                changed = True
    return box


def figure_box(page, cap, prev_cap_y1):
    page_w = page.rect.width
    images = [r for img in page.get_images(full=True) for r in page.get_image_rects(img[0])
              if r.width > 120 and r.height > 60]
    drawings = [fitz.Rect(d["rect"]) for d in page.get_drawings()]
    # drop full-width rules (header/footer lines) and degenerate rects
    drawings = [r for r in drawings if r.width < page_w - 40 or r.height > 5]
    # short text blocks (tick labels, panel titles, legends) bridge the white space between
    # rows of a multi-panel figure; long blocks are body text and must not
    labels = [fitz.Rect(b[:4]) for b in page.get_text("blocks")
              if len(b[4].strip()) <= 60 and fitz.Rect(b[:4]).y1 <= cap.y0 - 2]
    graphics = [r for r in images + drawings if r.y0 >= prev_cap_y1 - 2]
    if not graphics:
        return None, "page"
    region = graphics + [r for r in labels if r.y0 >= prev_cap_y1 - 2]
    box = cluster_above(region, cap)
    if box is None or box.height < 30 or not any(box.intersects(g) for g in graphics):
        return None, "page"
    kind = "raster" if any(box.intersects(r) for r in images) else "vector"
    # include figure-internal text (tick labels, legends) overlapping the box
    for b in page.get_text("blocks"):
        br = fitz.Rect(b[:4])
        if br.y1 <= cap.y0 + 2 and br.intersects(box + (-30, -30, 30, 30)):
            box |= br
    return (box + (-6, -6, 6, 6)) & page.rect, kind


def render(page, clip, path, width_px):
    dpi = max(72, min(400, int(width_px / (clip.width / 72))))
    page.get_pixmap(clip=clip, dpi=dpi).save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf")
    ap.add_argument("out", help="work directory (shared across parts)")
    ap.add_argument("--prefix", required=True, help="part name, e.g. main or supp")
    ap.add_argument("--dpi", type=int, default=200, help="page render dpi")
    ap.add_argument("--fig-width", type=int, default=2000, help="figure crop width in px")
    ap.add_argument("--caption-regex", default=DEFAULT_CAPTION,
                    help="regex matched at the start of a text block; group 1 = figure label")
    ap.add_argument("--strip-regex", action="append", default=[],
                    help="extra regex; matching lines are removed from the text (repeatable)")
    ap.add_argument("--running-frac", type=float, default=0.4,
                    help="fraction of pages a header/footer line must recur on")
    ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    doc = fitz.open(args.pdf)
    w = max(2, len(str(doc.page_count)))
    tdir, pdir, fdir = out / args.prefix / "text", out / args.prefix / "png", out / "figs"
    for d in (tdir, pdir, fdir):
        d.mkdir(parents=True, exist_ok=True)

    pages_lines = [p.get_text().split("\n") for p in doc]
    running = detect_running_lines(pages_lines, args.running_frac)
    strip_res = [re.compile(r) for r in args.strip_regex]
    print(f"{args.pdf}: {doc.page_count} pages; running lines removed: {sorted(running)}")

    chars = []
    for i, page in enumerate(doc):
        text = clean_page(pages_lines[i], running, strip_res)
        chars.append(len(text))
        (tdir / f"p{i + 1:0{w}d}.txt").write_text(text, encoding="utf-8", newline="\n")
        page.get_pixmap(dpi=args.dpi).save(pdir / f"p{i + 1:0{w}d}.png")
    print("chars/page:", chars)

    if args.no_figs:
        return
    cap_re = re.compile(args.caption_regex)
    rows = []
    for i, page in enumerate(doc):
        caps = sorted(caption_blocks(page, cap_re), key=lambda c: c[1].y0)
        prev_y1 = 0
        for label, cap in caps:
            box, kind = figure_box(page, cap, prev_y1)
            prev_y1 = cap.y1
            if box is None:
                box = page.rect
            lab = f"{int(label):02d}" if label.isdigit() else label
            render(page, box, fdir / f"{args.prefix}_fig{lab}.png", args.fig_width)
            rows.append(f"{label}\t{i + 1}\t{kind}\t{tuple(round(v) for v in box)}")
    (out / args.prefix / "figures.tsv").write_text("label\tpage\tkind\trect\n" + "\n".join(rows) + "\n",
                                                   encoding="utf-8", newline="\n")
    print(f"{len(rows)} figure captions:")
    print("\n".join("  " + r for r in rows))


if __name__ == "__main__":
    main()
