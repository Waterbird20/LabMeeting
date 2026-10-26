# BRIEF — 261027 deck, Chapter 3 rebuild (2026-10-26)

**Request (human, 2026-10-26).** Update Chapter 3 ("Applications", the slides from PDF page 22 on) with the redone
notebooks `3. wiki/code/qsp-unot-sq/`, `3. wiki/code/qsp-dq-preparation/`, `3. wiki/code/qsp-dq-pi/`.
**Order: SQ universal NOT → DQ state preparation → DQ π pulse.** "Write draft first with sparing and planning a space
for animation": no clips now; leave dashed animation slots where clips will go.

The human's design workflow (their words, condensed): set the target on the bands where detunings actually occur
(¹⁴N lines, each broadened by ¹³C, rms 0.1 MHz) — **no stopband** any more, nothing is required between the lines;
Chebyshev decomposition to estimate the proper $K$; find $\tau$ by **no overlap** (lines with different targets must
get different signals, $\cos\Delta_1\tau\neq\cos\Delta_2\tau$); solve by **Newton iteration** (QSPPACK style, much
faster than gradient training); evaluate by **gate fidelity** over the detunings of interest; **hard pulses first**
($R_x(\phi)=e^{-i\phi X/2}$, no duration), **then finite pulses** ($e^{-iHt}$, $H=\Delta Z+\Omega X$ during the pulse) to
see how robust the design is as $\Delta$ grows. Try arbitrary and symmetric (palindromic) QSP sequences.

**Plots the human asked for, per application** (each must appear, presentation quality):
(a) target function in $\mathrm{Re},\mathrm{Im}$ of $P,Q$ against $\Delta$; (b) Chebyshev decomposition, $x$ = basis
index $k$, $y$ = coefficient; (c) varying $K$: how the response approaches the target, against $\Delta$ and $\Delta\tau$;
(d) the optimisation loss (Newton cost against step); (e) optimised response against the target, $\mathrm{Re},\mathrm{Im}$
$P,Q$ against $\Delta$; (f) finite-pulse comparison; (g) the resulting pulse sequence.
For the DQ preparation and the DQ π pulse the hard-pulse target is constant, so (b) is one coefficient ($K=0$) and the
design problem only appears with finite pulses: adapt (b)/(c) honestly (constant target ⇒ $K=0$; then the progression
in the number of pulses $N$ with finite pulses: one plain pulse → the small isolated solutions → the fixed-$\tau$
design; and the count of conditions against parameters).

## Files and ownership

Deck folder `3. wiki/labmeeting/261027/`. One file per slide, own frontmatter (`marp: true`, `theme: serif`,
`math: mathjax`), sorted order = slide order. Old Chapter 3 is in `_archive/2026-10-26-ch3-v1/` (read freely, reuse
wording; do not move files back). Stale copies of 261023 files are in `_archive/stale-261023-copies/`.

| section | slide files (yours only) | figure script | media prefix | notes file |
|---|---|---|---|---|
| U: universal NOT (SQ) | `32-unot-*.md` … `39-unot-*.md` (≤ 9 slides) | `3. wiki/code/lm-261027-figures/fig_unot.py` | `media/ch3-unot-*.png` | `NOTES-ch3-unot.md` |
| P: DQ state preparation | `40-prep-*.md` … `46-prep-*.md` (≤ 6 slides) | `…/fig_prep.py` | `media/ch3-prep-*.png` | `NOTES-ch3-prep.md` |
| Π: DQ π pulse | `50-pi-*.md` … `57-pi-*.md` (≤ 7 slides) | `…/fig_pi.py` | `media/ch3-pi-*.png` | `NOTES-ch3-pi.md` |

The lead writes `30-section-application.md`, `31-design-steps.md`, `60-take-home.md`, `80-references.md`,
`01-outline.md`, `NOTES.md`. Never touch another section's files, `00-title.md`, `Makefile`, `preview.sh`,
`figstyle.py`, `ch3common.py`, `dump_data.py`, `data/`, the notebooks, `1. raw/` or `2. notes/`. Do not run `make`.

## Sources and numbers (authority order)

1. The executed notebooks: `qsp-*/qsp-*.ipynb` (printed outputs), `qsp-*/out/designs.json`, `qsp-*/README.md`, and the
   markdown of the percent sources `qsp-*/qsp-*.py`. **Every number on a slide comes from these.**
2. `3. wiki/code/lm-261027-figures/data/{unot,prep,pi}.pkl` (via `ch3common.load_pkl`): a local seed-42 rerun of the
   same sources, for curves the notebooks do not store (Newton histories, the fixed-τ scans, isolated solutions, the
   $K$ ladder of the universal NOT). It reproduces every printed statistic; the DQ designs are identical, but the
   universal-NOT rerun kept the mirror-equivalent hard design (same τ, T, band error). **Draw designs from
   `designs.json` (`ch3common.load_designs`)**, never from the pickle; say in your NOTES which curves come from the rerun.
3. Conventions: `3. wiki/code/qsp-hf-core/qspcore.py`. Rabi frequencies 20 and 10 MHz (single-tone Ω); the DQ spin-½
   Rabi frequency is $\Omega_{\rm DQ}=\Omega/\sqrt2$; $A_\parallel/2\pi=-2.16$ MHz, $\Delta_m=A_\parallel m_I$, B = 440 G.

Figures: your own script, `from ch3common import *` (figstyle rcParams: cream background, font 20, mathtext; `save(fig,
name)` writes `out/name.png` and `media/name.png`; colours: `RF[20.0]` blue / `RF[10.0]` red, `HARD` ink, `SINGLE`
grey, `TARGET` green bars, `BAND` grey shading of $\Delta_m\pm3\sigma$, `MCOL` per line, palindromic solid /
arbitrary dashed; helpers `pq`, `gauge_fix`, `shade_bands`, `target_bar`, `draw_sequence`, `omega`). Run with
`3. wiki/code/env/.venv/bin/python fig_<tag>.py` from `3. wiki/code/lm-261027-figures/`. Symbols in mathtext only,
never literal Greek (no "π" in a label: `$0.740\pi$`). Large, legible, one idea per PNG; two or three panels only
when they are meant to be compared. Show progressions where a parameter varies. Arbitrary preparation designs are
compared with the target after the free common phase is fixed (`gauge_fix`), and the slide notes say so.

## Style (the human is strict; this deck's rules)

- Title: `# <span class="cat intro|method|strategy|results">…</span> <Application>: <one or two words>` in standard
  terminology (e.g. "Universal NOT: target function", "DQ π pulse: finite pulses").
- One opening sentence (a full sentence that says what the slide shows), then the equation / figure / table, at most
  3 short bullets, each a complete sentence ending with a period. Few words (≈ 50–90 visible words).
- **No em-dashes** anywhere (slides, notes). "as a function of", never "versus". Every symbol or Greek letter in prose
  in math (`$\tau$`, `$\Delta$`). Standard terminology, no wiki or code names on slides or in speaker notes (no
  "notebook", "qspcore", "pkl", "§5", "fig4"). Provenance goes to your NOTES file only.
- No figure captions; at most a small `<p class="src">` line for a literature source.
- Speaker notes (this deck keeps them): one HTML comment at the end of each slide file with what the speaker says,
  the derivation steps, the meaning of each figure panel. No provenance in them.
- Kit: `<!-- _class: tight -->` or `dense`, `<div class="columns"><div class="col">`, `<figure class="figure">`
  (blank line, `![w:…](media/x.png)`, blank line), `<style scoped>` for local tweaks, callouts only if essential
  (the deck has few). MathJax: no literal `|` inside math in a table (`\lvert…\rvert`); math is not parsed in raw
  HTML without blank lines around Markdown.
- **Animation slots** (no clips now): a dashed box where a clip will go, with Markdown inside so math renders:
  ```
  <div class="anim mid">

  Animation: <one line on what the clip will show, e.g. the three lines moving on the circle $e^{i\Delta\tau}$ as $\tau$ sweeps; red when two lines coincide>

  </div>
  ```
  Sizes `big` 520×360, `mid` 460×320, `side` 340×280, `half` 470×200 (or an inline `style="width:…;height:…"`).
  Plan 2–3 slots per section where a clip clearly helps (target building, $\tau$ sweep, $K$ or $N$ progression,
  Newton iterates converging, Bloch trajectories of the three manifolds, time-mirror construction). A slot may sit
  next to the static figure it will later replace or extend. List every slot in your NOTES (slide, what the clip
  shows, data it needs). The existing clip `media/real-pulse-bloch.mp4` (+ poster `.png`) may be embedded where
  finite pulses are introduced: `<video src="media/real-pulse-bloch.mp4" poster="media/real-pulse-bloch.png"
  width="…" controls autoplay loop muted playsinline preload="none"></video>` inside a `<figure class="figure">`.
- Preview your own slides only:
  `cd "3. wiki/labmeeting/261027" && ./preview.sh <tag> /private/tmp/claude-501/-Users-hun-Aleph/fd52bb76-d4a6-4e8b-be7c-adc6b7c4d166/scratchpad/<tag> <your files in order>`
  then Read every page PNG (`<tag>-NN.png`; page 1 is the title). Fix footer overflow (the #1 failure), red MathJax
  errors, tiny figures, broken tables. Iterate until clean.

## Chapter-wide story (so the three sections fit together)

Steps (slide 31, written by the lead): **1 target** on the bands $\Delta_m\pm3\sigma$, free in between ·
**2 degree $K$** (Chebyshev decomposition; counting conditions against parameters) · **3 wait $\tau$** (no overlap) ·
**4 solve** (damped Newton with exact Jacobian from random starts) · **5 evaluate** (gate fidelity averaged over the
¹³C band and the three manifolds; hard pulses, then finite pulses).

- **U (first)** introduces, once for the chapter: the band target; the average gate fidelity; the mirror symmetry
  $u(-\Delta)=Xu(\Delta)X$ of pulses about $\pm x$ (why x-only pulses fail for the universal NOT); symmetric
  (palindromic) QSP, $u^T=u\Rightarrow Q$ imaginary (Dong, Meng, Whaley, Lin, PRA 103, 042419 (2021); Wang, Dong, Lin,
  Quantum 6, 850 (2022)), and why it fails here (axes in the $x$–$z$ plane); the non-overlap rule for $\tau$; the
  Newton solver (Levenberg–Marquardt-damped Newton, exact Jacobian from prefix/suffix products; Newton's method is
  also the QSPPACK solver for symmetric QSP: Dong, Lin, Ni, Wang, SIAM J. Sci. Comput. 46, A2951 (2024)); and finite
  pulses: a rectangular pulse ≈ hard pulse between two free evolutions $t_e=\tan(\beta/2)/\Omega$ (Euler form; it
  fails at $\beta=\pi$).
- **P (second)** introduces the DQ-specific pieces: the target is a state, $f=4[\mathrm{Im}(P^*Q)]^2$, two conditions
  per line ($\lvert P\rvert^2=\tfrac12$, $\mathrm{Re}(P^*Q)=0$, the second is phase coherence); hard pulses give
  $K=0$; a single finite $\pi/2$ pulse leaves a DQ phase error $\simeq\Delta/\Omega_{\rm DQ}$; the three exact
  properties of finite $\pm x$ sequences (mirror symmetry ⇒ fit only $\Delta=0$ and $A$; $u(0)=R_x(\sum\beta)$;
  palindromic ⇒ $Q=-ir$); the count (3 conditions) ⇒ palindromic $N=5$ / arbitrary $N=3$ at fixed $\tau$; the choice
  $\tau=10$ ns from duration and band error.
- **Π (third)** refers back to P for the shared machinery and adds: target $P=0$, only $Q^2$ matters (sign of $Q$ free
  per manifold), spin-1 fidelity $F=[(4t^2-1)^2+3]/12$, $t=\lvert\mathrm{Im}\,Q\rvert$; one finite π pulse fails at first
  order by a $y$ rotation; 3 (palindromic) or 4 (arbitrary, coherence) conditions ⇒ $N=5$ / $N=4$ at $\tau=10$ ns;
  the **time-mirror construction** (a preparation followed by its mirror is a DQ π pulse; the 20 MHz symmetric π pulse
  is the 20 MHz arbitrary preparation mirrored); **repetition**: the same $U$ in every manifold ⇒ $US_zU^\dagger=-S_z$ ⇒
  a train of these pulses refocuses the hyperfine and static ¹³C shifts (decoupling).

## Report (your final message)

Files written, preview page count, figures made, animation slots, every number you quote with its source, anything
unsourced (`TODO (speaker to confirm)` in notes), and disagreements between notebook text and outputs.

## Status (2026-10-26, end of day)

Draft built (24 Chapter 3 slides, reviewed); the human approved seven clips, built and embedded (see `NOTES.md`, "Chapter 3 rebuild"). Deck: 53 pages.
