# Verification instructions

You are checking other agents' Markdown transcriptions of a run of pages of an academic
paper against the source, and fixing what is wrong in place. The transcriptions follow
`prompts/transcribe.md` in this directory (read it: its format rules are the spec) and
`WD\PAPER.md` (paper-specific conventions; read it too). Your task message gives `WD`, the
part, the page range, the chunk files in order (each with its own page range), and the
figure description files belonging to that range. `WD\...` means a path inside that
directory, never a folder literally named `WD`.

The transcribers were given the same rules but worked separately, so the chunks may apply
them inconsistently (emphasis, math mode for numbers, degree signs, caption formatting).
Make every chunk follow the spec as written.

## Inputs

- The chunk files (edit them in place with Edit).
- `WD\<part>\text\pNN.txt` (exact wording) and `WD\<part>\png\pNN.png` (authority for math,
  tables, layout) for every page in the range. Read every page image.
- For each figure in the range: its description `WD\figdesc\<part>_fig<L>.md` and image
  `WD\figs\<part>_fig<L>.png`.
- The LaTeX source, if PAPER.md points to one. When PAPER.md marks it as the exact source of
  the PDF, compare every equation and table cell with it as well as with the page image: it
  settles symbols that are hard to read in the render. A draft settles nothing.

Use only Read, Grep, Edit, and Write. Do not run scripts or other processes.

## Check, in this order

Work page by page, keeping the page image, the text layer and the chunk open side by side.

1. **Coverage**: every sentence of the page range appears exactly once and in order: nothing
   dropped or duplicated, especially at page breaks, around figures and tables, and in
   footnotes. Compare paragraph by paragraph against the text layer.
2. **Joins**: at each boundary between two of your chunks, the END marker of the first and the
   START marker of the next must agree (`paragraph continues` ↔ `continues previous paragraph`),
   and the two halves of a split paragraph must read as one sentence when joined with a space.
   Text continuing a paragraph from the previous chunk comes first in a chunk, before any figure.
3. **Wording**: prose matches the text layer verbatim (the authors' own typos stay). Revert any
   paraphrase or "correction".
4. **Math**: every equation and inline symbol against the page image: sub/superscripts,
   hats/tildes/bars, bold, Greek letters, operators, signs, fractions, limits, brackets,
   equation numbers. Also check the LaTeX is valid (balanced braces and `$`, no `\mathrm` around
   whole expressions, no Unicode math left in prose).
5. **Tables**: every cell, header, and bold marking against the image; pipe-table syntax valid.
6. **Captions**: verbatim, math converted, formatted as the spec says; figure markers
   `<!-- FIGURE: ... -->` present and placed at a paragraph boundary.
7. **Headings**: levels and text as the spec and PAPER.md say.
8. **Figure descriptions**: compare against the image for factual errors: wrong panel count or
   layout, axis labels, ranges, legend entries, colors, trends, values. Fix errors and add a
   missing salient element; do not rewrite for style.

Make minimal, targeted edits. Do not reformat text that is already correct.

## Report

Reply with a short list of the fixes you made (one line each, grouped by chunk and type; say
"none" if none) and anything you still could not resolve (quote the spot). Nothing else.
