# Paper notes: liu2026distilling

Liu S, Holland T, Ma WJ, Acerbi L (2026). Distilling noise characteristics and prior
expectations in multisensory causal inference. *PLoS Comput Biol* 22(5): e1014251.

Topic: Bayesian causal inference models of audiovisual localization. Tasks: UV and UA
(unisensory visual / auditory localization), BV and BA (bisensory visual / auditory
localization), BC (bisensory causal judgment, "same"/"different"). Semiparametric models of
the sensory noise function $\sigma(s)$ and the prior $p(s)$, distilled into parametric
families (e.g. Exp-GaussianLaplace); lifted-semiparametric models (e.g. LiftedSemiparam-PM);
causal inference strategies MA (model averaging), MS (model selection), PM (probability
matching); auditory range recalibration $\rho_\mathrm{A}$.

## Parts

| part | PDF | pages | notes |
| --- | --- | --- | --- |
| `main` | `C:\Users\luigi\Documents\GitHub\journal.pcbi.1014251.pdf` | 46 | PLOS typeset layout; figures are raster images; Figs 1–15, Table 1 |
| `supp` | `C:\Users\luigi\Documents\GitHub\pcbi.1014251.s001.pdf` | 33 | S1 Appendix, LaTeX; vector figures (Figs A–U), Tables A–J |

## Source hint

`C:\Users\luigi\Documents\GitHub\maintext.tex` is an October 2024 LaTeX draft: main text,
and from line ~1157 an appendix. The paper was substantially revised afterwards (received
June 2025, accepted April 2026). Equation hint only; the PDF wins.

## Headings

- Main: Introduction, Results, Discussion, Methods → `##`; subsection headings on their own
  line → `###`; a further level on its own line → `####`; run-in headings (bold text opening a
  paragraph, e.g. "Bisensory localization tasks (BV and BA).") → inline `**...**`.
- Supplement: "Section A  Lapse distribution selection" → `## Section A: Lapse distribution selection`;
  "Section B.1  ..." → `### Section B.1: ...`; "Section B.1.1  ..." → `#### Section B.1.1: ...`.

## Front matter and back matter

- Main p01–p02 (done): title, authors with `<sup>` markers, affiliations, notes, Abstract,
  Author summary. The metadata sidebar is omitted except Data availability / Funding /
  Competing interests, which go to the backmatter.
- Main p42–p46 (backmatter): `## Supporting information` (the "S1 Appendix. ... (PDF)" entry
  as a paragraph), `## Acknowledgments`, `## Author contributions` (each "Role: names." line as
  a `- ` bullet item), `## References` as a numbered Markdown list keeping each number and each
  reference verbatim, including its DOI URL and PMID; rejoin URLs that the layout broke across
  lines (`https://doi.` + `org/10...` → `https://doi.org/10...`).
- Supplement p01 (title page, repeating the author block) and p02 (table of contents) are
  omitted: supplement transcription starts at p03.

## Notation (resolved from the PDF's fonts)

- Every θ in the paper is bold (font CMMIB10): write `\boldsymbol{\theta}`, e.g.
  `$\boldsymbol{\theta}_\sigma$`, `$\boldsymbol{\theta}_\mathrm{V}$`, `$\boldsymbol{\theta}_\mathrm{prior}$`.
