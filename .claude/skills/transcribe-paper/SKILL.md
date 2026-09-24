---
name: transcribe-paper
description: Add a paper to pubs-llms by transcribing its PDF with Sonnet agents, verified by Opus agents — prose from the PDF text layer; equations, tables and figure descriptions reconstructed from page images (equations and tables copied from the LaTeX source when the paper is on arXiv) — then assembling the main/backmatter/appendix/full files and updating the README. Use when adding a paper whose PDF has a real text layer (publisher or LaTeX PDFs, plus optional supplement). For scanned PDFs without a text layer, use inscriber instead.
---

# Transcribe a paper into pubs-llms

The pipeline: extract each PDF page's text layer and a page render (script) → Sonnet agents
transcribe ~5-page chunks into Markdown, fixing math, tables and layout against the page
images → Sonnet agents describe the figures → 3–4 Opus agents verify the chunks and
descriptions → a script assembles the split files → the README is regenerated.

Paths below: `SKILL` = this skill's directory, `REPO` = the pubs-llms root, `WD` = a work
directory in the session scratchpad (e.g. `<scratchpad>/<base>`). The agent prompts in
`SKILL/prompts/` hold the formatting rules; `WD/PAPER.md` holds what is specific to one paper.

## Supervising the agents

Sonnet agents (transcription, figure descriptions) follow instructions to the letter. Where
the instructions leave a gap, they fill it with their own judgment calls, and some of those
calls are wrong. Treat every agent output
as a draft:

- **Concrete values, not placeholders.** Give absolute paths, exact page ranges and the exact
  figure list. Agents took the placeholder `WD\chunks\M2.md` literally and wrote to a new folder
  named `WD`.
- **Check the files, not the report.** When an agent finishes, confirm its file exists at the
  expected path and run `lint.py` on it. A report describes what the agent believes it did.
- **Mine the reports for gaps.** The judgment calls an agent lists (emphasis, degree signs,
  figure placement at a page break) show where the rules are ambiguous. Settle each one in
  `prompts/` or `WD/PAPER.md` so the remaining agents and the verifiers apply a single rule.
  When agents disagree about how a glyph is typeset, don't pick a side from the reports:
  `fonts.py --chars` settles it. In one run, two chunks disagreed on whether θ was bold; every
  θ in the PDF was set in `CMMIB10`, so it was.
- **Verification is required.** The verify pass (step 5) and your own review (step 6) are part
  of the pipeline. A chunk that only its author has checked is not finished.

## 0. Inputs

- The PDF(s): main paper, and supplementary material if separate. A combined PDF is fine;
  its appendix chunks just use the same part.
- Bibliographic data for the BibTeX entry (title, authors, venue, year, volume/number/pages,
  DOI or URL).
- Base name `<firstauthor><year><firstword>`, lowercase, as in `publications/` (e.g.
  `liu2026distilling`; a second paper with the same key gets a letter, like `huang2024bamortized`).
- Optional: a LaTeX source of the paper. It is either **exact** (the files the PDF was
  compiled from) or a **draft** (any other version). Agents copy equations and tables from an
  exact source; a draft is only a hint for how to write an equation.

### arXiv papers

For a paper on arXiv, take the PDF and the LaTeX source of the same version `vN` (the abstract
page lists the versions; the abstract page also gives title and authors for the BibTeX):

```bash
curl -sSL -o WD/paper.pdf https://arxiv.org/pdf/<id>vN
curl -sSL -o WD/src.tar.gz https://arxiv.org/e-print/<id>vN
mkdir WD/src && tar xzf WD/src.tar.gz -C WD/src
```

The e-print is usually a gzipped tar; a single-file source is a gzipped `.tex` instead, which
`gunzip` unpacks. Some authors withdraw the source, and then the pipeline runs without one.
The e-print is an exact source. It still differs from the PDF in ways PAPER.md must spell
out for the agents:

- **Macros**: list the preamble's `\newcommand`s that appear in the body with what they
  print (e.g. `\method` → POLAR in small caps; a `\vpmse{m}{s}` table macro → `$m \pm s$`).
- **Material not in the PDF**: `%` comments (whole abandoned paragraphs and table rows are
  common), `\iffalse` blocks, a commented-out `\input{checklist}`.
- **Resolved references**: citation numbers, equation numbers and `\Cref` targets exist only
  in the PDF, so they come from the page.

When the paper has since been published at a venue, the BibTeX names that venue and keeps
the arXiv URL, since the transcription is of the arXiv version.

Python with PyMuPDF is needed for the scripts (e.g. the inscriber venv,
`~/Documents/GitHub/inscriber/.venv/Scripts/python.exe`, or `pip install pymupdf`). On
Windows, set `PYTHONIOENCODING=utf-8`.

## 1. Extract

```bash
python SKILL/scripts/extract.py main.pdf WD --prefix main
python SKILL/scripts/extract.py supplement.pdf WD --prefix supp
```

Writes `WD/<prefix>/text/pNN.txt`, `WD/<prefix>/png/pNN.png`, `WD/figs/<prefix>_fig<L>.png`
and `WD/<prefix>/figures.tsv`. Check what it prints:

- **Running lines removed**: should be only headers/footers/page numbers. Add
  `--strip-regex` for any it missed.
- **chars/page**: prose pages should have thousands. If most pages are near-empty, the PDF
  has no usable text layer; use inscriber instead.
- **Figure captions**: the count must match the paper's figures. Adjust `--caption-regex`
  (default matches `Fig 1.`, `Figure 2:`, `Fig S3.`, `Fig C.`) if needed. Read a few
  crops, especially vector figures and multi-panel grids. A crop marked `page` fell back to
  the full page.

Also find the section layout (grep the text files for headings) to plan the chunks.

Run `python SKILL/scripts/fonts.py paper.pdf`: it lists the fonts with sample text. Math fonts
reveal notation that agents misjudge from page images. In LaTeX-derived math, `CMMIB` means
bold math italic (`\boldsymbol`), `CMR` upright (`\mathrm`), `CMMI` italic. Record the
conventions it settles in a Notation section of `WD/PAPER.md` before starting the agents.
`fonts.py paper.pdf --chars θ` shows the font of every occurrence of a glyph.

## 2. Write `WD/PAPER.md`

Copy the notes file from `SKILL/examples/` and adapt it:

- the topic and terminology;
- a table of parts, PDFs and pages;
- the LaTeX source, if any: exact or draft, its main file(s), and for an exact source the
  macros and the material that is not in the PDF (see "arXiv papers" in step 0);
- heading conventions for this venue;
- how to handle front matter (title/author block, metadata sidebars) and back matter
  (acknowledgments, references format);
- pages to omit (e.g. a supplement's title page or table of contents).

Every agent reads this file.

## 3. Plan chunks and figure batches

- **Chunk size:** about 5 pages each. Use 3–4 pages for pages dense in equations or tables;
  pages that are only figures are cheap.
- **Order and names:** chunks cover the pages in order with no gaps. Name them `M1…Mk` for the
  main paper and `S1…Sk` for the supplement.
- **Back matter:** give acknowledgments and references their own chunk(s).
- **Title-page chunk:** it also writes any front-matter snippet that belongs in the backmatter
  (e.g. PLOS's data availability / funding / competing interests → `M1_sidebar.md`).
- **Figure batches:** 3–5 figures per describe agent, fewer when they are dense.

## 4. Run the agents

Use the Agent tool with `subagent_type: general-purpose` and `model: sonnet`. Run **at most 5
agents at a time**, and start the next one as each finishes. Keep a ledger in `WD/ledger.md`
(job, kind, range, status: queued/running/done) and update it on every launch and completion.
Without one it is easy to go over the limit while other work is in progress. Transcription and description
are independent of each other, so interleave them. Keep prompts short, since the rules live
in the prompt files. Give every output path as an absolute path (`<WD>` below stands for the
real directory): agents shown `WD\chunks\M2.md` have created a folder literally named `WD`.

Transcribe:
```
Read the instructions `SKILL\prompts\transcribe.md` and follow them exactly.
- WD: `<WD>`
- Part: `main`
- Pages: p06–p10
- Output (absolute path): `<WD>\chunks\M2.md`
<one or two lines of context: the section it starts in, figures/tables in range, special handling>
```

Describe:
```
Read the instructions `SKILL\prompts\describe.md` and follow them exactly.
- WD: `<WD>`
- Figures (label, part, page → output):
  - Fig 1, main, p06 → `<WD>\figdesc\main_fig01.md`
  - Fig 2, main, p07 → `<WD>\figdesc\main_fig02.md`
```

Read each transcription agent's report as it arrives. When one reveals a convention the
prompts do not settle (e.g. emphasis), add the rule to `prompts/transcribe.md` (general) or
`WD/PAPER.md` (this paper), so later agents and the verifiers apply it.

## 5. Verify

Once all chunks and descriptions exist, run 3–4 verify agents with `model: opus`. Each one
covers a contiguous run of chunks within one part, about 15–25 pages, together with the
figures in those pages. A contiguous run also lets the verifier check the joins between its
chunks. Verification is where judgment matters most (spotting a wrong subscript, a
paraphrase or a misread axis), and Opus is worth the cost there.

```
Read the instructions `SKILL\prompts\verify.md` and follow them exactly.
- WD: `<WD>`
- Part: `main`, pages p01–p20
- Chunks, in order: `<WD>\chunks\M1.md` (p01–p05), `<WD>\chunks\M2.md` (p06–p10), ...
- Figures in range: `<WD>\figdesc\main_fig01.md` (`<WD>\figs\main_fig01.png`), ...
```

Before starting a verifier, copy its chunks and descriptions to `WD/snapshots/pre-verify/`.
Verifiers sometimes rewrite a whole file. With the snapshot you can `diff` their changes
instead of taking the report's word for them.

Expect the verifiers to fix a lot in the figure descriptions. In the first run, Opus found
factual errors in most Sonnet descriptions:

- a column's trend reversed;
- a legend placed in the wrong row;
- arrows missing from a diagram;
- a value range misread;
- a panel-letter mismatch between image and caption left unmentioned.

Verifier fixes to prose and math were fewer and smaller: soft-hyphen words, curly quotes,
italic vs upright subscripts, and figure placement at page breaks.

## 6. Assemble and lint

1. Write `WD/paper.bib`. List `title` before `author`, followed by `journal`/`booktitle` and
   `year`: that order hits the primary regex in `private/update_readme.py`.
2. Write `WD/config.json` (format in the `scripts/assemble.py` docstring; example in
   `SKILL/examples/`).
3. Run:

```bash
python SKILL/scripts/assemble.py WD/config.json      # writes REPO/publications/<base>_*.md
python SKILL/scripts/lint.py REPO/publications/<base>_*.md
```

`assemble.py` reports two kinds of problem:

- **Seam mismatches**: one chunk ends mid-paragraph but the next starts a new one, or the
  reverse. Read the two chunk edges against the page and fix the marker or the text.
- **Figure markers**: a figure is missing, duplicated or unexpected.

Lint warnings are heuristics: read each flagged line before changing it.

Then review the output yourself:

- **Text coverage:** run `python SKILL/scripts/textdiff.py <WD>/chunks/M2.md <WD>/main/text 6 10`
  for each chunk, giving the chunk's first and last page. It compares the chunk's prose with
  the text layer word by word, discounts relocated text such as captions moved to paragraph
  boundaries, and prints per-word count differences and Greek-letter count differences.
  Expected residue:
  - math tokens (`psame` in the text layer vs `\mathrm{same}` in the chunk, `\log`);
  - URLs broken across lines;
  - omitted metadata;
  - for vector figures, axis labels.

  Anything else is a candidate dropped or invented passage: read it on the page.
- **Seams:** grep the joined text around each chunk boundary.
- **Heading hierarchy:** `grep -n '^#'`.
- **Figures:** each description is followed by its caption.
- **References:** the count matches the PDF.
- **Math:** spot-check a few equations against the page images.
- **Glyph counts:** for a distinctive symbol, compare the per-page counts from
  `fonts.py --chars θ` with the occurrences in each chunk. Matching counts show that no math
  was dropped or duplicated.

## 7. README and commit

```bash
python private/update_readme.py
```

This regenerates `README.md` from `private/frontmatter.md` and the BibTeX blocks. Check that
the new entry has the right venue abbreviation, author initials and year. A new venue needs
a mapping in `update_readme.py`. On push, the GitHub Action reruns it. It also runs
`create_full_papers.py`, which skips `_full.md` files that already exist.

Two things to expect in the README diff:

- A local run can reorder entries that share a year and venue. Compare with
  `git diff README.md | grep '^[-+]- \*\*'`: an entry that shows as both removed and added
  has only moved.
- The regenerated list includes every paper whose files are in `publications/`, including
  another session's work in progress. Commit only this paper's entry, or leave the README to
  the Action.

Commit only when the user asks. The commit contains
`publications/<base>_{main,backmatter,appendix,full}.md` and `README.md`.

## Output conventions

These match the other 2026 papers in `publications/`:

- **`_main.md`**:
  - a fenced BibTeX block, then `---`;
  - then `# Title`, the author line, affiliations and the paper body;
  - then `---` and the footer line.
- **`_backmatter.md` / `_appendix.md`**: `# Title - Backmatter` / `# Title - Appendix`, then
  `---`, the content, `---` and the footer.
- **`_full.md`**: main, then appendix, then backmatter (the same order as `inscriber join`),
  with one footer at the end.
- **Footer**: `*Transcribed from the PDF text layer and corrected with LLMs; text, equations,
  tables, and figure descriptions may contain mistakes.*`
- **Figures**: a `> **Image description.** ...` blockquote, followed by the caption paragraph.
- **Math**: `$...$` inline, `$$...$$` for display, `\tag{n}` for equation numbers.

## Notes from past runs

- Header/footer and figure detection are heuristics:
  - Figure tick labels (`-20`) recur on figure pages. Letterless running-line candidates
    are therefore limited to page-number shapes.
  - Multi-panel grids are joined into one crop across the white space between rows by their
    short label text blocks.
  - A LaTeX `wrapfigure` is cropped at full text width, so its crop includes the paragraph
    beside it. The describe prompt tells agents to ignore it; the crop needs no fix.
- PLOS PDFs:
  - The metadata sidebar on p1–p2 is mixed into the text layer; data availability, funding
    and competing interests sit on p2.
  - Figure DOI lines (`.../journal.pcbi.NNNNNNN.g001`) follow each caption.
- arXiv PDFs:
  - p01 carries the arXiv identifier stamp (`arXiv:2606.25197v1 [cs.LG] 23 Jun 2026`) and, in
    the NeurIPS preprint style, a `Preprint.` footer. Neither recurs, so the running-line
    detection keeps them; strip them with `--strip-regex '^arXiv:\d{4}\.\d{4,5}v\d+\s+\['
    --strip-regex '^Preprint\.$'`.
  - With hyperref's `backref=page`, each reference ends with the pages that cite it
    (`... pages 748–756. PMLR. 1, 3, 6`). They point at PDF pages the transcription does not
    keep; PAPER.md tells the agents to drop them.
  - With the exact source, the transcribers copied every equation and table cell and the
    verifiers found only spacing differences in the math. Nearly all verification fixes were
    in the figure descriptions of dense multi-curve plots, so that is where to look hardest.
    A 26-page paper took 7 transcription, 3 describe and 3 verify agents.
- A paragraph can span a chunk boundary with figure-only pages in between. The prompt makes
  the continuation text come first in the next chunk and the figures after it. Still check
  the seams: `assemble.py` flags mismatched START/END markers.
- Mechanical fixes across chunks (e.g. `\theta` → `\boldsymbol{\theta}`): write the Python
  script to a file with the Write tool and run it. Inline heredocs through the Bash tool on
  Windows have collapsed `\\` to `\`, which breaks LaTeX-heavy regexes.
- Rough costs:
  - a 5-page Sonnet transcription chunk: about 6 minutes and 100k tokens;
  - a describe batch of 4 figures: about 4 minutes and 90k tokens;
  - an Opus verifier over about 20 pages: about 10 minutes and 300k tokens.

  A 46 + 33-page paper took 16 transcription, 9 describe and 4 verify agents.
