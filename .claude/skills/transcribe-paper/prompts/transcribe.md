# Transcription instructions

You are transcribing a range of pages of an academic paper (or its supplementary
material) into clean Markdown. The result becomes part of an LLM-friendly text-only
archive of the paper, so fidelity is what matters.

Your task message gives the work directory (`WD`), the part (e.g. `main`, `supp`), the page
range, and the output path. `WD\...` in these instructions means a path inside that
directory, never a folder literally named `WD`. Read `WD\PAPER.md` first: it has the paper-specific conventions
(heading styles, front matter, what to omit, source hints). Where PAPER.md and this file
differ, PAPER.md wins.

## Inputs

- `WD\<part>\text\pNN.txt`: the text layer of each PDF page (running headers/footers and
  soft hyphens already removed). Prose here is exact: it is your source for wording. Math,
  subscripts, superscripts, accents, and table layout are mangled in it (`xV` may be
  $x_\mathrm{V}$, `σ2(s)` may be $\sigma^2(s)$, `ˆs` may be $\hat{s}$, etc.).
- `WD\<part>\png\pNN.png`: renders of the same pages. Read every page image of your range.
  It is the authority for math, symbols, sub/superscripts, layout, headings, table
  structure, and what is bold/italic.
- Page numbers always mean the PDF page index (p01 = first page of the file), not
  printed page labels.
- PAPER.md may point to a LaTeX source of the paper (Grep it for nearby words to find a
  spot) and says which of two kinds it is:
  - **Exact source**: the files this PDF was compiled from (e.g. the arXiv e-print of the
    same version). Copy display equations, inline math, table cells and algorithms from it,
    expanding the paper's own macros into standard LaTeX (PAPER.md lists them) and dropping
    layout-only commands (`\label`, `\vspace`, `\looseness`, `\textcolor`, `\nonumber`).
    Equation numbers, citation numbers and cross-references (`Equation (3)`, `Figure 2`,
    `Appendix B.1`) come from the page, where they are already resolved. Prose wording still
    comes from the text layer. Commented-out lines (`%`) and disabled blocks are not in the
    PDF: skip them. Confirm every copied item against the page image; if they differ, the page
    image wins, and you report it.
  - **Draft**: use it only as a hint for how to write an equation or symbol in LaTeX. **The PDF
    wins on every symbol, subscript, number, and word.** Never copy prose, numbers, or table
    values from a draft; if the draft's equation differs from the page image, transcribe the
    page image.

Use only Read, Grep, and Write. Do not run scripts or other processes.

## Output

Write exactly one Markdown file to the path given in your task (plus any extra file PAPER.md
or the task asks for). No commentary inside the file beyond what is specified here.

- **Scope**: transcribe exactly your page range, in reading order. The first line of the
  file must be either `<!-- START: new paragraph -->` or
  `<!-- START: continues previous paragraph -->` (when your first page begins mid-paragraph).
  The last line must be `<!-- END: paragraph complete -->` or
  `<!-- END: paragraph continues -->` (when your last page ends mid-paragraph). Start and end
  with the text exactly where the pages start and end, even mid-sentence.
- **Fidelity**: keep the authors' wording verbatim, including their typos and grammatical
  slips. Fix only extraction artifacts: line breaks, broken hyphenation, lost
  sub/superscripts, mangled symbols, ligatures. Do not paraphrase, summarize, or improve.
- **Drop**: running headers/footers, page numbers, line numbers. No page markers.
  Paragraphs that continue across a page break inside your range are joined into one.
- **Headings**: the paper title is `#` (only in the chunk that contains it); top-level
  sections `##`; then `###`, `####`. Keep section numbers as printed (`## 3 Methods`,
  `### 3.1 Model fitting`). Run-in headings (bold or italic text opening a paragraph) stay
  inline as `**...**` at the paragraph start. PAPER.md may refine this for the venue.
- **Math**: all math in LaTeX. Inline: `$...$`. Display: `$$ ... $$` on its own lines with a
  blank line before and after. Numbered equations get `\tag{n}` inside the display math with
  the paper's number. Every mathematical symbol in running text goes in math mode, matching
  the typeset appearance: upright (roman) subscripts such as V, A, max, prior → `\mathrm{...}`;
  italic subscripts stay italic; bold vectors `\mathbf{x}` / `\boldsymbol{\theta}` as printed.
  Use standard amsmath (`\hat`, `\mathcal{N}`, `\propto`, `\mid`, `\left( \right)`,
  `\begin{aligned}`, `\begin{cases}`). Numeric expressions with an operator or a statistic
  go in math mode as a whole: `$1.40 \pm 0.07$`, `$1.1\times$`, `$p < 0.05$`,
  `$t(14) = 4.72$`, `$\rho_\mathrm{A} = 4/3$`; so do numeric intervals and sets
  (`$[-45^\circ, 45^\circ]$`, `$\{0^\circ, \pm 5^\circ\}$`). Plain numbers, percentages and
  phrases like "–20° to 20°" stay prose. Degrees: `°` in prose; `^\circ` inside math.
  Method/model names are plain text unless typeset as math.
- **Emphasis**: italic or bold emphasis in prose as printed, `*...*` / `**...**`. Keep the
  paper's quote marks (curly or straight, single or double) as printed.
- **Citations**: keep as printed (`[1,3,25]`, `[2–4]`, `(Smith et al., 2020)`).
- **Figures**: where a figure sits in the page flow (between paragraphs; if a paragraph is
  interrupted by a figure, place the figure after that paragraph ends — or, when that
  paragraph continues past your last page, before the paragraph starts; text continuing a
  paragraph from before your range always comes first in your file, before any figure; a
  figure set beside the text, with paragraphs wrapping around it, goes after the last
  paragraph it sits beside), write a line
  `<!-- FIGURE: <label> -->` using the label as printed in the caption (`Fig 3`, `Figure 3`,
  `Fig S2`, `Fig C`), a blank line, then the caption as one plain paragraph starting with that
  label as printed (`Fig 3. ...`, `Figure 3: ...`), with math converted and no bold
  (not the title sentence, not panel labels like `(A)`). Keep a figure DOI or
  URL line printed right after the caption as its own line. Do not describe the figure or
  transcribe text that is inside the figure (axis labels, legends, tick numbers); for vector
  figures such text appears as noise in the `.txt` files — ignore it.
- **Tables**: caption paragraph as printed (`Table 1. ...`, math converted), then a GitHub pipe
  table rebuilt from the page image, then any table footnote as a paragraph. Take digits from
  the text layer and structure from the image; check every cell. Math in cells as `$...$`;
  escape a literal `|` in a cell as `\|`; reproduce bold cells as `**...**`. Flatten multi-row
  headers into one header row with combined labels (e.g. `UV+UA: ΔAIC`). Keep a table
  DOI/URL line as its own line.
- **Algorithms / code**: fenced code block, verbatim.
- **Lists/other**: keep enumerations and bullet lists as Markdown lists. URLs as plain text.

## Report

After writing the file, reply in a few lines: the output path, anything you could not
determine with confidence (quote the spot), and any place where the LaTeX source and the PDF
disagreed on an equation or table cell (one line each). Nothing else.
