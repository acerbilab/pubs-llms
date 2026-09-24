#!/usr/bin/env python3
"""Flag mechanical problems in transcribed Markdown (chunks or assembled files).

Usage: python lint.py FILE [FILE ...]

Checks: leftover HTML-comment markers, unbalanced $$ / $ math delimiters (per paragraph),
soft hyphens, ligatures, replacement characters, Unicode math left outside math mode
(Greek letters, sub/superscript digits, relation symbols), and image-description
blockquotes with lines missing their '>' prefix. Warnings are heuristics: read the
flagged line before changing anything.
"""
import re
import sys
from pathlib import Path

UNICODE_MATH = re.compile(r"[Ͱ-Ͽ₀-ₜ²³¹⁰-ⁿ"
                          r"∼≈≡∝∈∉⊂⊆∑∏∫√∞≤≥≠±×·→←↔∂∇]")
BAD_CHARS = {"­": "soft hyphen", "�": "replacement character",
             "ﬀ": "ligature ff", "ﬁ": "ligature fi", "ﬂ": "ligature fl",
             "ﬃ": "ligature ffi", "ﬄ": "ligature ffl"}


def strip_math(par):
    par = re.sub(r"\$\$.*?\$\$", " ", par, flags=re.S)
    return re.sub(r"(?<!\\)\$.*?(?<!\\)\$", " ", par, flags=re.S)


def lint(path):
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    out = []

    def warn(lineno, msg):
        out.append(f"{path}:{lineno}: {msg}")

    for i, line in enumerate(lines, 1):
        # START/END/FIGURE markers belong in chunks; assemble.py reports any it cannot resolve
        if "<!--" in line and not re.match(r"\s*<!-- (START|END|FIGURE):", line):
            warn(i, "leftover HTML comment / marker")
        for ch, name in BAD_CHARS.items():
            if ch in line:
                warn(i, name)

    # paragraph-level checks (a $$ display block may span lines)
    in_code = False
    par, par_start = [], 1
    for i, line in enumerate(lines + [""], 1):
        if line.startswith("```"):
            in_code = not in_code
        if in_code:
            continue
        if line.strip() == "" and par:
            p = "\n".join(par)
            if p.count("$$") % 2:
                warn(par_start, "odd number of $$")
            else:
                singles = re.sub(r"\$\$.*?\$\$", "", p, flags=re.S)
                if len(re.findall(r"(?<!\\)\$", singles)) % 2:
                    warn(par_start, "odd number of inline $")
                prose = strip_math(p)
                prose = re.sub(r"`[^`]*`", " ", prose)
                hits = set(UNICODE_MATH.findall(prose))
                # "mean ± SEM" is fine as prose; "1.40 ± 0.07" belongs in math mode
                if "±" in hits and not re.search(r"\d\s*±|±\s*\d", prose):
                    hits.discard("±")
                hits = sorted(hits)
                if hits and not p.lstrip().startswith(">"):
                    warn(par_start, f"Unicode math outside $...$: {' '.join(hits)}")
            if p.lstrip().startswith("> **Image description.**"):
                for j, l in enumerate(par):
                    if l.strip() and not l.startswith(">"):
                        warn(par_start + j, "image-description line without '>'")
            par = []
        elif line.strip():
            if not par:
                par_start = i
            par.append(line)
    return out


def main():
    problems = [w for f in sys.argv[1:] for w in lint(Path(f))]
    print("\n".join(problems) if problems else "lint: clean")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
