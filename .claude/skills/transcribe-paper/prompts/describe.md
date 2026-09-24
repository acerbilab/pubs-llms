# Figure description instructions

You are writing text-only descriptions of figures from an academic paper (or its
supplementary material). The descriptions replace the images in an LLM-friendly text-only
archive of the paper: a reader who cannot see the figure should learn what it shows.

Your task message gives the work directory (`WD`; `WD\...` means a path inside it, never a
folder literally named `WD`) and the figures (label, part, page, output path). Read `WD\PAPER.md` first for what the paper is about and its terminology.

## Inputs (per figure)

- The figure image: `WD\figs\<part>_fig<L>.png` (L zero-padded if numeric). Crops may include
  a bit of surrounding heading or caption text; ignore it. If the crop looks wrong (cut off,
  wrong figure), use the full page render `WD\<part>\png\pNN.png` instead and say so.
- The caption: in `WD\<part>\text\pNN.txt` for the given page (search for the label). Read it
  for context and terminology (what colors, shading, markers mean). You may Grep the other page
  text files for how the figure is discussed.

Use only Read, Grep, and Write. Do not run scripts or other processes.

## Output

One file per figure at the path given in your task, containing only the description block,
in exactly this form (every line starts with `>`; blank lines inside are `>`):

```
> **Image description.** One or two sentences: what kind of figure this is and how it is laid out (panels, grid).
>
> **Panel A** (or a short label for a region): ...
>
> ...
```

- Describe what is visible: plot types, panel layout and labels, axis labels and ranges/ticks,
  legends, colors and markers, what each line/band/bar/point represents, the visible trends
  and approximate key values, annotations, diagrams' structure and text.
- Do not restate the caption, do not add a figure number or heading, do not interpret beyond
  what the figure itself shows (the caption supplies meaning; you supply the visuals).
- Math in `$...$` (e.g. `$s_\mathrm{A} - s_\mathrm{V}$`, `$\sigma(s)$`). Bullet lists are fine
  inside the block (`> - ...`).
- Length follows complexity: roughly 120–250 words for a simple figure, up to ~600 for a dense
  multi-panel one. For large grids of near-identical small panels (e.g. one panel per
  participant or condition), describe the grid layout, the shared axes and encodings, and the
  salient patterns and differences, rather than every panel.
- Be accurate about what you can actually read. If a label is illegible, say so briefly rather
  than guessing.
- Before writing the file, re-open the image and check each specific claim in your draft against
  it. That covers:
  - which panel or row shows what;
  - where the legend sits;
  - the number of points, bars and arrows;
  - the direction of every slope and trend;
  - value ranges read from the axis ticks;
  - whether panel letters in the image match the caption's.

  Delete or soften any claim you cannot confirm. A missing detail is better than a wrong one.
- The description is about the figure alone. Leave out anything about how you worked: the crop,
  the page render, text you ignored. Those go in your report.

## Report

Reply with one line per figure: the path written, plus anything you were unsure about.
