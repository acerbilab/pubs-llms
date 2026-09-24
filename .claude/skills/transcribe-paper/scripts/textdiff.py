#!/usr/bin/env python3
"""Compare a chunk's prose with the PDF text layer, word by word, to catch dropped,
duplicated, or paraphrased text.

Usage: python textdiff.py CHUNK.md WD/<part>/text FIRST LAST [--min 4]

Greek letters are also counted on both sides (text-layer glyphs vs `\\sigma`-style commands).
Both sides are reduced to lowercase alphabetic words of 3+ letters: math is dropped from the
chunk except for \\mathrm/\\text words (subscripts such as "motor" also appear in the text
layer), and figure-description blockquotes, markers, and URLs are ignored. Runs of at least
--min differing words are printed with context. Figure-internal text (axis labels in vector
figures) shows up as text-layer-only runs; judge each hit by reading the page.
"""
import argparse
import difflib
import re
import unicodedata
from collections import Counter
from pathlib import Path

WORD = re.compile(r"[A-Za-z]{3,}")


def chunk_words(text):
    text = re.sub(r"^<!--.*?-->\s*$", " ", text, flags=re.M)
    text = "\n".join(l for l in text.split("\n") if not l.lstrip().startswith(">"))
    text = re.sub(r"https?://\S+", " ", text)

    def keep_words(m):
        return " " + " ".join(re.findall(r"\\(?:mathrm|text|operatorname)\{([^{}]*)\}", m.group(0))) + " "
    text = re.sub(r"\$\$.*?\$\$", keep_words, text, flags=re.S)
    text = re.sub(r"(?<!\\)\$.*?(?<!\\)\$", keep_words, text, flags=re.S)
    return [w.lower() for w in WORD.findall(text)]


def layer_text(text_dir, first, last):
    files = sorted(Path(text_dir).glob("p*.txt"))
    pick = [f for f in files if first <= int(f.stem[1:]) <= last]
    return "\n".join(f.read_text(encoding="utf-8") for f in pick)


def layer_words(text_dir, first, last):
    text = layer_text(text_dir, first, last)
    text = re.sub(r"https?://\S+(\s*\S*\.\S+)?", " ", text)
    return [w.lower() for w in WORD.findall(text)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("chunk")
    ap.add_argument("text_dir")
    ap.add_argument("first", type=int)
    ap.add_argument("last", type=int)
    ap.add_argument("--min", type=int, default=4, help="smallest differing run to report")
    args = ap.parse_args()
    a = layer_words(args.text_dir, args.first, args.last)
    b = chunk_words(Path(args.chunk).read_text(encoding="utf-8"))
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    ops = [op for op in sm.get_opcodes() if op[0] != "equal"]
    removed = [" ".join(a[i1:i2]) for _, i1, i2, _, _ in ops if i2 > i1]
    added = [" ".join(b[j1:j2]) for _, _, _, j1, j2 in ops if j2 > j1]

    def moved(run, others):
        # a run that reappears elsewhere is relocated text (e.g. a caption placed at a
        # paragraph boundary), not a loss
        return len(run.split()) >= args.min and any(run in o or o in run for o in others
                                                    if len(o.split()) >= args.min)

    hits = 0
    for tag, i1, i2, j1, j2 in ops:
        lost, new = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        lost = "" if moved(lost, added) else lost
        new = "" if moved(new, removed) else new
        if max(len(lost.split()), len(new.split())) < args.min:
            continue
        hits += 1
        print(f"[{tag}] after '...{' '.join(a[max(0, i1 - 5):i1])}'")
        if lost:
            print(f"   text layer only: {lost[:300]}")
        if new:
            print(f"   chunk only:      {new[:300]}")

    # bag-of-words check, immune to reordering: which words occur a different number of times
    ca, cb = Counter(a), Counter(b)
    diff = sorted(((w, ca[w] - cb[w]) for w in set(ca) | set(cb) if ca[w] != cb[w]),
                  key=lambda x: -abs(x[1]))
    print(f"{args.chunk}: {len(a)} layer words, {len(b)} chunk words, "
          f"{hits} unexplained runs >= {args.min} words")
    if diff:
        shown = ", ".join(f"{w}{d:+d}" for w, d in diff[:40])
        print(f"   word-count differences (layer minus chunk): {shown}")

    # Greek letters: text-layer glyphs vs \name commands (or raw glyphs) in the chunk's prose
    layer = layer_text(args.text_dir, args.first, args.last)
    chunk = "\n".join(l for l in Path(args.chunk).read_text(encoding="utf-8").split("\n")
                      if not l.lstrip().startswith(">"))
    rows = []
    for ch in sorted(set(re.findall(r"[α-ωΓ-Ω]", layer))):
        name = unicodedata.name(ch).split()[-1].lower()
        name = {"lamda": "lambda"}.get(name, name)  # Unicode spells it LAMDA
        if ch.isupper():
            name = name.capitalize()
        # LaTeX variant forms (\varepsilon, \varphi, \vartheta, ...) count too
        n_layer = layer.count(ch)
        n_chunk = len(re.findall(rf"\\(?:var)?{name}(?![A-Za-z])", chunk)) + chunk.count(ch)
        if n_layer != n_chunk:
            rows.append(f"{ch}: layer {n_layer}, chunk {n_chunk}")
    if rows:
        print("   Greek-letter count differences: " + "; ".join(rows))


if __name__ == "__main__":
    main()
