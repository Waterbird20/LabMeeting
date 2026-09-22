# 260923 revision brief v3 (2026-09-22): addendum to BRIEF.md and BRIEF-v2.md

Not a slide. Read `BRIEF.md` (talk, writing rules, clip form) and `BRIEF-v2.md` (drive
convention, visual vocabulary, `_verify()` pattern) first; every rule there still holds
except the machine notes below. This revision acts on the speaker's 2026-09-22 review.

## Machine (changed again)

- We are on **macOS** (Apple Silicon, 10 cores), not WSL. The manim venv
  `3. wiki/code/env-manim/.venv` (manim community 0.21, python 3.13) works here;
  `latex`, `dvisvgm`, `ffmpeg`, `marp`, `pdftoppm` are on PATH. Verified 2026-09-22.
- Render from `3. wiki/code/lm-260923-animations/` exactly as in BRIEF-v2
  (`../env-manim/.venv/bin/python render.py <file> <Scene> -q l|h`). Eight agents share
  the machine: preview with `-q l`, and run **at most one `-q h` render at a time** per
  agent. If a `-q h` render would exceed ~25 min, use `-q m` (720p30) as the final.
  Run long renders in the background (`run_in_background`) and poll; a foreground shell
  is capped at 10 min. Two renders must not share one `--media_dir`; `render.py` uses
  `out/manim/` for everyone, so if you need a parallel preview, use manim directly with
  `--media_dir <your scratchpad>`.
- Private deck build from the deck folder, **never the un-tagged `make`**:
  `make THEME=serif BUILD=.build-<tag>.md BASE=<tag> pdf < /dev/null`, then
  `pdftoppm -png -r 60 -f A -l B <tag>.pdf <scratch>/pg`, Read the PNGs, and delete
  `.build-<tag>.md`, `<tag>.pdf`, `<tag>.html` when done. Find your page numbers with
  `grep -n "^# " .build-<tag>.md` (page = 1 + number of `---` separators before it) or just
  render a range around where you expect.
- Write each slide file **once**, consolidated, then `grep` it back to confirm (sync bridge).
  Write mp4/png outputs to your scratchpad first if you cut with ffmpeg by hand; `render.py`
  already installs into `media/` and has been reliable.

## Video embedding, updated form (mandatory for every `<video>` you touch)

```html
<figure class="figure">

<video src="media/CLIP.mp4" poster="media/CLIP.png" width="640" controls autoplay loop muted playsinline preload="none"></video>

*Caption as a markdown italic line with $math$.*

</figure>
```
`controls` is new (the speaker wants a seek bar and the fullscreen button). Keep
`preload="none"`, the blank lines, and the italic markdown caption.

## Bigger videos (speaker's request: "increase size of video, reduce the font for that page")

Every clip must be visibly larger than before. Rules of thumb on the 1280x720 slide with the
serif theme (`section` padding 62px 76px, so ~1128 px of usable width; `.columns` is flex
with a 1.8rem gap, equal columns):

- Two-column slide with the clip in one column: give the clip column more room with a
  scoped style, e.g.
  `<style scoped>.columns { gap: 1.2rem; } .columns .col:first-child { flex: 0 0 420px; } </style>`
  and set the video `width="640"`..`"680"`. Put `<!-- _class: tight -->` (21 px) on the
  slide and trim prose so nothing crosses the footer.
- Single-column slide (short text above, clip below): `width="820"`..`"900"` with
  `<!-- _class: tight -->` and at most ~4 lines of prose plus the formula.
- The clips are 16:9 (960x540); height = width x 9/16. Budget the vertical space: the
  content area is about 720 - 62 - 62 - footer ≈ 560 px tall. A 640-wide clip is 360 px
  tall, an 820-wide clip is 461 px tall.
- Always build and Read the rendered page; footer overflow is the #1 failure.

## Ownership for this revision (touch only your row; the integrator owns the rest)

| agent | slide files | clips / figures | code file (new unless noted) |
|---|---|---|---|
| `seq` | `17a-rabi.md`, `17b-ramsey.md`, `17c-odmr.md` | `media/pulse-rabi.png`, `media/pulse-ramsey.png`, `media/pulse-odmr.png` (static) | `fig_pulses.py` |
| `dd` | `18c-times.md`, `19-dd.md`, `19a-cpmg.md`, `19b-axes.md` | `dd-unit` (recut), `dd-cpmg` (recut), `media/dd-t2-vs-pulses.png` (given) | `scenes_dd.py` (existing) |
| `conv-identity` | `25-theorem.md`, `26-dd-product.md` | `conv-identity` (new clip), `media/taminiau-fig2a.png` | `scenes_conv_identity.py` |
| `conv-continuous` | new `24b-continuous.md` | `conv-continuous` (new clip) | `scenes_conv_continuous.py` (or a 3B1B cut, see prompt) |
| `conv-summary` | new `29-summary.md` | `conv-summary` (new clip) | `scenes_conv_summary.py` |
| `qsp-blocks` | `32-polynomial.md`, new `32a-pq-example.md` | `qsp-blocks`, `qsp-chebyshev` (new clips) | `scenes_qsp_blocks.py` |
| `qsp-conv` | `33-convolution.md`, `34-response.md` (+ `34a-` if split) | `qsp-coeff-conv` (recut) | `scenes_qsp.py` (existing) |
| `qsp-design` | `35-boxcar.md`, `39-phases.md` (+ `39a-` if split) | `media/boxcar-taps.png` etc. (static) | `fig_qsp_design.py` |

The integrator owns `00`, `01`, `10`-`16b`, `17`, `17d`, `18`-`18b`, `19z`, `20`-`24a`,
`27`, `30`, `31`, `36`-`38a`, `39z`, `80`, `90`, `REFS-*.md`, `README.md`, the Makefile,
`_index_wiki.md`, `log.md`. Chapter 3 is a draft the speaker will polish, so its agents
may rewrite prose freely; chapters 1 and 2 keep the speaker's sentences where they exist.

## Report back (final message)

Clip names, durations, poster fractions, pages rendered and checked, what changed in the
slide text, and anything the speaker still has to decide (as `<!-- EDIT-FORWARD: ... -->`
in the file plus one line in the report).
