#!/usr/bin/env python3
"""Assemble chunk transcriptions and figure descriptions into pubs-llms split files.

Usage: python assemble.py WD/config.json [--out publications_dir]

config.json (paths relative to its directory):
{
  "base": "liu2026distilling",
  "title": "Distilling noise characteristics ...",
  "bibtex": "paper.bib",                        # file holding the @article{...} entry
  "parts": {
    "main":       {"chunks": ["M1", "M2"], "figs": "main"},
    "backmatter": {"chunks": ["M1_sidebar", "M9"], "figs": "main"},   # optional
    "appendix":   {"chunks": ["S1", "S2"], "figs": "supp"}            # optional
  }
}

Chunks are WD/chunks/<name>.md. A chunk's first/last lines are START/END markers
(see prompts/transcribe.md) telling whether it opens or closes mid-paragraph; a
chunk without markers (e.g. a sidebar snippet) is treated as whole paragraphs.
`<!-- FIGURE: Fig N -->` markers are replaced by WD/figdesc/<figs>_fig<L>.md, and every
label listed in WD/<figs>/figures.tsv (written by extract.py) must appear exactly once.
Writes {base}_main.md, _backmatter.md, _appendix.md and _full.md (main, appendix,
backmatter, like `inscriber join`).
"""
import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_PUB = Path(__file__).resolve().parents[4] / "publications"
FOOTER = ("*Transcribed from the PDF text layer and corrected with LLMs; text, equations, "
          "tables, and figure descriptions may contain mistakes.*")
START_RE = re.compile(r"\A\s*<!-- START: (new paragraph|continues previous paragraph) -->[ \t]*\n")
END_RE = re.compile(r"\n\s*<!-- END: (paragraph complete|paragraph continues) -->\s*\Z")
FIG_RE = re.compile(r"^<!-- FIGURE: (?:Fig\.?|Figure)\s*(S?\d+|[A-Z]\d*) -->$", re.M)

problems, notes = [], []


def label_key(label):
    return f"{int(label):02d}" if label.isdigit() else label


def load(wd, name):
    text = (wd / "chunks" / f"{name}.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    s, e = START_RE.search(text), END_RE.search(text)
    if not s and not e:
        notes.append(f"{name}: no START/END markers, treated as whole paragraphs")
        return text.strip(), "new paragraph", "paragraph complete"
    if not s or not e:
        problems.append(f"{name}: only one of the START/END markers present")
    body = text[s.end() if s else 0:e.start() if e else len(text)]
    return body.strip(), s.group(1) if s else "new paragraph", e.group(1) if e else "paragraph complete"


def join(wd, names):
    out, prev_end = "", None
    for name in names:
        body, start, end = load(wd, name)
        if prev_end is None:
            out = body
        else:
            cont_prev = prev_end == "paragraph continues"
            cont_next = start == "continues previous paragraph"
            if cont_prev != cont_next:
                problems.append(f"seam before {name}: previous END={prev_end!r}, START={start!r}")
            out, body = out.rstrip(), body.lstrip()
            # Display math keeps its blank lines even inside a continuing paragraph.
            at_display = out.endswith("$$") or body.startswith("$$")
            sep = " " if (cont_prev and cont_next and not at_display) else "\n\n"
            out = out + sep + body
        prev_end = end
    return out


def insert_figures(wd, text, prefix, seen):
    def repl(m):
        key = label_key(m.group(1))
        seen.setdefault(prefix, []).append(key)
        path = wd / "figdesc" / f"{prefix}_fig{key}.md"
        if not path.exists():
            problems.append(f"missing description {path.name}")
            return m.group(0)
        return path.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
    return FIG_RE.sub(repl, text)


def expected_labels(wd, prefix):
    tsv = wd / prefix / "figures.tsv"
    if not tsv.exists():
        return None
    rows = tsv.read_text(encoding="utf-8").splitlines()[1:]
    return sorted(label_key(r.split("\t")[0]) for r in rows if r.strip())


def write(pub, name, text):
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    (pub / name).write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {pub / name}: {len(text)} chars")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--out", type=Path, default=DEFAULT_PUB, help="publications directory")
    args = ap.parse_args()
    cfg_path = Path(args.config)
    wd = cfg_path.parent
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    base, title = cfg["base"], cfg["title"]
    bib = (wd / cfg["bibtex"]).read_text(encoding="utf-8").strip()
    bibtex = f"```\n{bib}\n```"
    footer = cfg.get("footer", FOOTER)

    seen, bodies = {}, {}
    for part, spec in cfg["parts"].items():
        bodies[part] = insert_figures(wd, join(wd, spec["chunks"]), spec["figs"], seen)
    for prefix in {spec["figs"] for spec in cfg["parts"].values()}:
        exp, got = expected_labels(wd, prefix), sorted(seen.get(prefix, []))
        if exp is not None and exp != got:
            missing = sorted(set(exp) - set(got))
            dup = sorted({k for k in got if got.count(k) > 1})
            extra = sorted(set(got) - set(exp))
            problems.append(f"{prefix} figure markers: missing {missing}, duplicated {dup}, unexpected {extra}")

    pub = args.out
    main_body = bodies["main"]
    write(pub, f"{base}_main.md", f"{bibtex}\n\n---\n\n{main_body}\n\n---\n\n{footer}")
    full = f"{bibtex}\n\n---\n\n{main_body}"
    for part, suffix in [("appendix", "Appendix"), ("backmatter", "Backmatter")]:
        if part in bodies:
            doc = f"# {title} - {suffix}\n\n---\n\n{bodies[part]}"
            write(pub, f"{base}_{part}.md", f"{doc}\n\n---\n\n{footer}")
            full += f"\n\n---\n\n{doc}"
    write(pub, f"{base}_full.md", f"{full}\n\n---\n\n{footer}")

    for n in notes:
        print("note:", n)
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
