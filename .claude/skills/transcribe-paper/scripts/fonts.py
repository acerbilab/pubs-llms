#!/usr/bin/env python3
"""Report which fonts a PDF uses for given glyphs, to settle bold/italic/upright questions.

Usage:
    python fonts.py paper.pdf                    # font inventory with sample text
    python fonts.py paper.pdf --chars θσV        # per-glyph font usage, by page
    python fonts.py paper.pdf --chars θ --pages 25-35

Font names are telling for LaTeX-derived math: CMMI = math italic, CMMIB = bold math
italic (\\boldsymbol), CMR = upright roman (\\mathrm), CMBX = bold upright (\\mathbf),
CMSY = symbols, MSBM = blackboard bold. Publisher fonts vary (e.g. "...-Italic", "...Bold").
"""
import argparse
from collections import Counter, defaultdict

import fitz


def page_range(spec, n):
    if not spec:
        return range(n)
    a, _, b = spec.partition("-")
    return range(int(a) - 1, int(b) if b else int(a))


def spans(doc, pages):
    for i in pages:
        for block in doc[i].get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for s in line["spans"]:
                    yield i + 1, s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf")
    ap.add_argument("--chars", help="glyphs to report on, e.g. θσV")
    ap.add_argument("--pages", help="1-based page range, e.g. 10-20")
    args = ap.parse_args()
    doc = fitz.open(args.pdf)
    pages = page_range(args.pages, doc.page_count)

    if not args.chars:
        count, samples = Counter(), defaultdict(list)
        for _, s in spans(doc, pages):
            count[s["font"]] += 1
            t = s["text"].strip()
            if t and t not in samples[s["font"]] and len(samples[s["font"]]) < 6:
                samples[s["font"]].append(t[:30])
        for font, n in count.most_common():
            print(f"{n:6d}  {font:28s}  {' | '.join(samples[font])}")
        return

    for ch in args.chars:
        usage = defaultdict(Counter)
        for page, s in spans(doc, pages):
            k = s["text"].count(ch)
            if k:
                usage[s["font"]][page] += k
        print(f"'{ch}':")
        if not usage:
            print("    not found")
        for font, by_page in sorted(usage.items(), key=lambda kv: -sum(kv[1].values())):
            pages_str = ", ".join(f"p{p}×{c}" for p, c in sorted(by_page.items()))
            print(f"    {font:28s} {sum(by_page.values()):4d}  {pages_str}")


if __name__ == "__main__":
    main()
