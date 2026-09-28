# Handoff: slide variant of Fig. 9 (per-signal readout) with larger labels

Not a slide (the Makefile only picks up `[0-9]*.md`). Request for the agent on the WSL machine, where
`~/DQML/analysis/phys-explain/out/readout_examples.csv` lives. Written 2026-09-29 by the Mac-side
readout fixer (round 3). Delete this file once the slide points at the new PNG.

## Why

Slide `49-readout.md` shows Fig. 9 of `projects/dqml/dqml-physics-results.md` §4.5
(`media/readout-trained-examples.png`, shown as `media/readout-trained-examples-crop.png`, blank rows
trimmed by `3. wiki/code/lm-260929-animations/fig_readout.py` `crop_examples()`), 1120 px wide on a
1280 × 720 slide. That is the widest that fits above the footer. At that size the per-QPU value labels
(0.52, 0.43, ...) are about 7 px tall and the tick digits about 9 px: readable on a laptop, not on a
projector. The limit is the figure's own font sizes. The Mac side has no `readout_examples.csv`, and it
must not redraw the figure from values read off the PNG (33 of its 48 numbers are not on the results
page), so the re-render has to come from WSL.

## What to render (same data, same layout)

- Same models, signals and values as Fig. 9: CC model, seed 0, test signals 1, 6, 5; per-QPU
  $P_0(c)$, $P_1(c)$, $P_2(c)$ on top, product $P(c\,\vert\,x)$ below; true class outlined; "true: 3"
  etc.; QPU colours `#3D8BE8`, `#22C08A`, `#8E7CF0`, product `#44403c`; column titles
  "test signal <idx>". Read the values from `readout_examples.csv`, not from the PNG.
- Size about **12 × 4.6 in**, 200 dpi, white background, tight bbox with small margins (it is shown
  about 1120 px wide, so 1 pt ≈ 1.3 px on the slide).
- **Fonts about 1.6× Fig. 9's:** per-QPU value labels and all tick labels ≥ 11 pt; product value labels
  ≥ 13 pt; panel titles ≥ 12 pt.
- To make room: y tick labels (0, 0.5, 1.0) only on the leftmost panel of each row of each signal
  (share y within a signal), and less blank space between the per-QPU row and the product row.
- Mathtext for every symbol (no literal Unicode Greek); no caption or title text beyond the labels;
  standard terminology only (`260929/GLOSSARY.md`).

## Return

- Save as `3. wiki/labmeeting/260929/media/readout-trained-examples-slide.png` (new file; leave
  `readout-trained-examples.png` and the results page's Fig. 9 unchanged).
- Note under this heading: script path, date, and a check that the 48 printed values equal Fig. 9's.
- Mac side then changes one line of `49-readout.md` (`![w:1120](media/readout-trained-examples-crop.png)`
  to the new file, width to be re-checked in a private build) and its src comment.
