# 260929 Lab Meeting brief (shared contract for all section agents)

Not a slide (no leading digits, so the Makefile ignores it). Read it fully before writing.

## Changes since this brief was written (read first)

- **Round 1, 2026-09-29:** words cut and **every caption under a figure or clip removed** (the
  caption lines in "Deck conventions" below are superseded: essentials go into speaker-note
  comments). Objectives moved from `03-objectives.md` to `18-objectives.md`; new slides
  `15-fc-network.md` and `64a-fingerprint-protocol.md`; `99-backup.md` moved to `_removed/`.
- **Round 2, 2026-09-29:** the Manim shout-out comes first (`00a-manim.md`, right after the title and
  before the outline; it was `85-manim.md`). `56-shuffle.md` and `57-quantum-channel.md` were removed
  (moved to `_removed/`). `65-next.md` is now **"Take-home"**, one conclusion callout in place of the
  next-steps list (the list is in its speaker note).
- **Standing style (rounds 1 and 2):** few words, no captions, speaker-only material in comments; no
  mutual information, $D_\mathrm{add}$, "CE" or cross-entropy as a metric in visible text (see `GLOSSARY.md`).
- **Round 3, 2026-09-29:** the results page names the $0.978$ classical model (§0.1) and adds a
  per-signal product-of-experts readout of the trained seed-0 models (§4.5, Fig. 9;
  `media/readout-trained-examples.png`): slide 49 now shows it in place of the illustrative
  `poe-readout` clip (as `media/readout-trained-examples-crop.png`, blank rows trimmed), and a new
  `49-readout2.md` shows the per-QPU accuracies (state at 07:33 KST).

## The talk

Lab meeting, **2026-09-29**, Paulee Group (KIST). Speaker: Donghun Jung. Topic:
**Distributed Quantum Machine Learning (DQML)**, a research update. The audience is the
group: physicists, mostly not ML people. So the talk opens with a **very brief**, animated
machine-learning primer, then QML, then DQML, then the speaker's model and results, and
ends with honest doubts.

Storyline in one paragraph: *Machine learning needs nonlinearity and depth. A quantum
convolutional network gets them from mid-circuit measurement and pooling stages. Splitting
it over several small QPUs that exchange only classical bits (LOCC) is the distributed
version. On MNIST-1D, three 4-qubit QPUs reach about 0.9 test accuracy; most of the gain
came from a Fourier embedding that makes the encoded data almost linearly separable, and
communication adds a few points in every pattern. That is exactly why the interesting
questions (topology, classical vs quantum channel, trained logic) are not yet visible, and
the talk ends with what should change: a task where communication must matter, such as
quantum fingerprinting.*

**Do NOT use the 260922 deck** (`../260922/`): the speaker does not want it. Do not copy
its prose, structure, or its `media/` figures (they show an older 4 × 5-qubit, 8-round
model with 0.943 accuracy that is not the current model). Everything here is built from
the current wiki pages and fresh clips.

**Use only the 3 QPUs × 4 qubits (n = 4) results.** The speaker said: "We will not use
n = 6 data." Never quote a 6-qubit or 18-qubit number anywhere in the deck.

## Speaker's outline (verbatim intent; do not drift)

0. **Timeline** (why the update was late; "I was not holding with nothing"):
   2026 Feb, got the code from CY (who led the previous DQML project) and ran it.
   2026 Mar–May, busy with the Post-Selection and DDrf projects. 2026 Jun–Jul, Hidden
   Manifold model (rejected). 2026 Aug, Quantum Hidden Manifold (rejected). 2026 Aug–now,
   1D MNIST was chosen; "now I'm burning my Google Colab+ account".
   Objectives: multiple QPUs vs one big QPU; classical communication analysis; network topology.
1. What is machine learning. 2. Universal approximation theorem.
3. Quick example with the neuron: AND, OR (linear) and XOR (nonlinear). Animation.
4. CNN example (animation): how convolution is used for feature extraction; a CNN is a
   subset of a fully connected layer. Sources: 3Blue1Brown, *But what is a neural
   network?* (youtube.com/watch?v=aircAruvnKk) and *But what is a convolution?*
   (youtube.com/watch?v=KuXjwB4LzSA).
5. Introduce QML: nonlinearity and depth are what ML requires; nonlinearity comes from
   intermediate measurement and depth from a couple of pooling stages. Then **skepticism
   first**, then optimism.
6. DQML: add the distributed feature.
7. Intriguing questions: how much can classical communication catch up with a quantum
   channel? Are there efficient network topologies? How well do they perform under
   shuffle? Is there a lottery-ticket-like effect (if some communication channels are
   randomly removed, is there an accidental improvement)?
8. Dataset: 1D MNIST, with the classical analysis.
9. Model explanation: data slice (animation); $(a,b,c,d)$ are now trainable; Fourier
   embedding (animation); convolution (actually a PQC) and pooling (intermediate qubit
   measurement); how the prediction is determined (animation); data augmentation
   (animation); qubit reuse; cross entropy; how the model performs under shuffle.
10. Results: accuracy about 0.9 at most (no n = 6 data); most of the gain comes from the
    Fourier embedding (motivated by translational symmetry), and the embedded data was
    almost linear already (**show what a classical linear classifier looks like**); one
    large QPU (a plain QCNN) underperforms (still trying); with communication every
    pattern is already a good model and the differences are not dramatic; $(a,b,c,d)$ are
    trained but not dramatically, many links are not conditional (constant, or one-variable),
    unclear whether because of trainability; with $(a,b,c)$ fixed it is easy to find a
    non-communicating range of $d$ (XOR and OR tried), that range gives the
    no-communication test accuracy, and an optimal $d$ exists.
11. Skepticism: embedding a slice of the data vector seems not good; the Fourier embedding
    is too good to see a network-topology effect; the converged $(a,b,c,d)$ seem not to
    change from their initial values (check the others) no matter what they were.
    Suggestion: quantum fingerprinting.

## Sources of truth (read before writing; quote exact numbers)

- `3. wiki/projects/dqml/dqml-physics-results.md`: **the** results page (Phases 2–3B,
  2026-09-25/28). Sections are referenced as §N below.
- `3. wiki/projects/dqml/dqml-physics-plan.md`: the plan and the speaker's decisions.
- `3. wiki/projects/dqml/mnist1d-eda.md`, `mnist1d-repro.md`: dataset and classical analysis.
- `3. wiki/projects/dqml/dqml-quantum-design.md` (§2.3.1 translation-symmetric Fourier
  assignment, the motivation), `dqml-task-design.md` (LOCC ladder), `_index.md`.
- Papers: `3. wiki/papers/@hwang2024distributed.md`, `@greydanus2020scaling.md`,
  `@bowles2024better.md`, `@chinzei2024splitting.md`, `@belis2026spectral.md`,
  `@nguyen2024theory.md`, `@meyer2023exploiting.md`, `@das2024role.md`; raw folders in
  `1. raw/papers/` (e.g. `buhrman2001quantum`, `kempkes2026cautious`, `hwang2024distributed`).
  `1. raw/references.bib` for bibliographic data. Read-only.
- CY's original code (Feb 2026): `~/DQML/README.md` §7–8 (read-only): the old rule was
  `score = a m_i + b m_{i+1} + c m_i m_{i+1} >= d` on **single-shot outcome bits**
  $m\in\{0,1\}$, with $(a,b,c,d)$ fixed in `config.yaml` (`multi_fixed`, default AND
  $(1,1,-1,1.5)$) or tuned by SPSA (`multi`); 3 processors × 3 qubits; binary
  classification of a synthetic 9-dimensional cluster dataset; PennyLane.
- Figures already rendered from the real runs: `3. wiki/projects/dqml/figs/`
  (`dqml-physics-results-*.png`, `mnist1d-eda-*.png`, `mnist1d-repro-table1.png`, ...).
  Copy what you use into `media/` under the same name. Open each before using it.
- Web search is allowed for literature you cite (verify authors, venue, year).

Every number on a slide must come from one of these; put an HTML comment with the source
on the slide, e.g. `<!-- src: dqml-physics-results.md §4.3 -->`. Mark anything you could
not source `[unverified]` in an `<!-- EDIT-FORWARD: ... -->` comment, never on the slide.

### Key facts (n = 4 only; verify against the page before quoting)

- Model (§0.2): 3 QPUs × 4 qubits; MNIST-1D digits 0, 1, 3, 6; 1325 train / 264 validation
  / 411 test signals; each QPU sees a 14-feature window (contiguous: consecutive features;
  permuted: after a fixed random permutation of the 40 features); revised Fourier state
  preparation; $L=8$ brick-wall layers ($R_ZR_XR_Z$, IsingZZ on (0,1),(2,3) then (1,2),(3,0),
  $R_X$); two pooling rounds measure qubits 0 and 2 with feed-forward on a remaining qubit;
  links $g=a\,m_i+b\,m_j+c\,m_im_j+d$ on $K$-shot estimated probabilities, bit
  $s=1$ with probability $\pi=\Phi\big(g/\sqrt{\sigma_g^2+\epsilon^2}\big)$, receiver applies
  $U_+$ or $U_-$; readout: 2 qubits per QPU give $P_b(c)$, prediction
  $P(c\,|\,x)\propto\prod_bP_b(c)$ (product of experts), cross-entropy loss.
  434 parameters per QPU, 1302 in the whole model (Fig. 2 caption).
- Training (§0.3): Adam, lr 0.03 cosine, 200 epochs, batch 256; each epoch fresh
  augmentation: random shift up to ±2 samples plus correlated noise of scale 0.1; shot
  number annealed $K=10^2\to10^4$; model selected at the best validation epoch. About 900
  training runs over Phases 2–3B on Colab (A100 and CPU); exact density-matrix simulation.
- Headline accuracy (n = 4): two-input links 0.903 ± 0.007 (seeds 0–4, §4.3) and
  0.908 ± 0.002 (seeds 0–2, §2.4); on held-out seeds 3–7: 0.8895 (§3); best fixed test
  (OR, $d=-0.75$, $K=10^3$, 2 seeds): 0.917 (§5.2). No communication: 0.865 ± 0.014 (§2.4),
  0.872 ± 0.013 (§4.3). Classical references (§0.1): 0.978 best on all 40 features (kernel logistic regression with
  a translation-aware kernel, named on the page since 2026-09-29); RBF SVM 0.964; best additive per-window model 0.971 (contiguous) / 0.891 (permuted).
- Fourier gain (§2.4): original → revised encoding: no communication 0.738 → 0.865,
  with two-input links 0.848 → 0.908; linear classifier on the untrained encoded states
  0.773 → 0.943.
- One 12-qubit QCNN (§8): 0.805 ± 0.076 against 0.865 (no comm.) and 0.908 (comm.),
  with 1656 against 1128 / 1188 circuit parameters.
- Patterns (§4.3 table) and quantum channel (§8.1): see the page.
- Links (§5.0): 48 trained links at n = 4: constant 14, one variable 20, implication type 9,
  AND/NAND/NOR 5, XOR 0. Four of six links of seed 0 keep their initial Boolean function.
- $d$ scan (§5.2 table), Fig. 6, 7, 8.

## Deck conventions

- Theme **serif** (`make` default here), `math: mathjax`. Every slide file starts with
  ```
  ---
  marp: true
  theme: serif
  math: mathjax
  ---
  ```
  Deck-wide CSS is in `00-title.md`: `.dense` (23 px) and `.tight` (21 px) via
  `<!-- _class: dense -->`, `.small`, `.src` (small grey source line),
  `.columns`/`.col`, `figure.figure`, `.callout`, `.reqs` + `.chip done|wip|todo`.
- Category pill in every content H1: `# <span class="cat intro">Intro</span> Title`
  with intro | method | strategy | results | ongoing.
- Section divider:
  ```
  <!-- _class: section -->
  <!-- _paginate: false -->

  <div class="sec-num">02</div>

  # Title

  <div class="subtitle">One line</div>
  ```
- **Figures:** `<figure class="figure">` with blank lines, `![w:700](media/x.png)`, then a
  markdown italic caption line `*... $math$ ...*`. Never `<figcaption>`, never math in
  `<li>`. Keep figures large (single ~700–1100 px wide, two-column ~500–560 px).
- **Clips (mandatory form):**
  ```html
  <figure class="figure">

  <video src="media/CLIP.mp4" poster="media/CLIP.png" width="760" controls autoplay loop muted playsinline preload="none"></video>

  *Caption as a markdown italic line with $math$.*

  </figure>
  ```
  `preload="none"` is required (the PDF build hangs otherwise); the PDF prints the poster.
  Clips are 16:9, height = width × 9/16. The content area is about 1170 × 560 px after
  the 50 px / 54 px padding and the title, so a single-column slide can hold a ~820–900 px
  clip with at most ~3 lines of prose (`<!-- _class: tight -->`); a two-column slide a
  ~600–640 px clip with `<style scoped>.columns .col:first-child { flex: 0 0 420px; }</style>`.
- Citations on a slide: a `<div class="src">` line, e.g. *Hwang et al., Quantum Sci.
  Technol. 10, 015059 (2025)*. Put your full references in `REFS-<tag>.md` (numbered list
  with authors, title, venue, year, arXiv/DOI); the integrator merges them into
  `80-references.md`.
- Backup material that does not fit: `BACKUP-<tag>.md` (not numbered; the integrator decides).

## Writing rules (the speaker is strict)

- **Terminology is binding: see `GLOSSARY.md`** (standard QI / physics / ML terms, one name per thing; the speaker, 2026-09-28: "do not use our own slang").

- **Full sentences** ending in periods; no telegraphic fragments, including in bullets.
- **No em-dashes (`—`) anywhere**, also not in captions or comments that render. Use
  commas, colons, semicolons, periods, or words. En-dash only in compound names and ranges.
- **Every symbol and Greek letter in math mode** (`$\theta$`, `$m_i$`, `$K$`, `$n=4$`),
  including captions, tables and pills. Never bare unicode Greek.
- Explain the **physical meaning** of each equation and symbol.
- One idea per slide; ~12 lines at 26 px; **footer overflow is the #1 failure**.
- Callouts (`<div class="callout">`) are for genuine headlines only: at most one in your
  whole section (the integrator may remove it).
- Be honest where the data are weak: say "within noise", "two seeds", "not significant".

## Clips (manim community 0.21)

Code lives in `3. wiki/code/lm-260929-animations/` (Claude-owned). One scene file per
agent: `scenes_<tag>.py`. Import the shared helper:
```python
from manim import *
import numpy as np
from dqml_style import *          # colours, tex_template(), load_mnist1d(), windows(),
                                  # dft(), encode_window(), phase_slot(), link_prob(), ...
```
- Look: black background (manim default), LaTeX text (`Tex`, `MathTex` with
  `tex_template=tex_template()`), colours from `dqml_style` (`QPU_COLORS`,
  `DIGIT_COLORS`, `C_CLASSICAL` gold for classical bits/links, `C_QUANTUM` pink for
  quantum channels). Text large enough to read at 800 px wide (≥ 30 px equivalent;
  `font_size` ≥ 32 for body text in a 1080p frame).
- **Real data, not invented**: MNIST-1D traces come from `load_mnist1d()` (the standard
  dataset, 4-class filter, identical to the runs: test signal 1 is the digit 3 of the
  wiki's worked example). Anything schematic (a toy function, a cartoon cloud) is labelled
  "schematic" or "toy" in the clip.
- Deterministic: seeded (`np.random.default_rng(42)`), every curve computed, never
  hand-drawn. Each scene file has a `_verify()` that checks every formula or number the
  clip prints; `python scenes_<tag>.py` runs it.
- Scene classes carry `CLIP = "<clip-name>"`. Length 12–40 s, designed to loop (end on a
  clean summary frame; the poster is taken at `--poster` fraction, choose a frame that
  reads as a still: usually 0.9–0.97).
- Render (from the code folder, **always with your own media dir**):
  ```bash
  cd "/Users/hun/Aleph/3. wiki/code/lm-260929-animations"
  ../env-manim/.venv/bin/python render.py scenes_<tag>.py <Scene> -q l --media-dir out/manim-<tag>   # preview
  ../env-manim/.venv/bin/python render.py scenes_<tag>.py <Scene> -q h --media-dir out/manim-<tag> --poster 0.95  # final
  ```
  `render.py` installs `media/<clip>.mp4` and `media/<clip>.png` in the deck. Fifteen
  agents share a 10-core Mac: preview at `-q l`, check frames
  (`ffmpeg -ss <t> -i <mp4> -frames:v 1 <scratch>/f.png` then Read), run **at most one
  `-q h` render at a time**, run it in the background (`run_in_background`) and wait for
  it; if a `-q h` render would exceed ~20 min, use `-q m` as the final.
- Plots (matplotlib) for static figures: `fig_<tag>.py` in the same folder, run with
  `../env-manim/.venv/bin/python` (numpy, scipy, matplotlib, scikit-learn, mnist1d are
  installed). Large fonts (`font.size` ≈ 20), single-panel PNGs where possible, 200 dpi,
  **mathtext for every symbol** (`r"$\theta$"`), never literal Greek. Save into the deck's
  `media/`. When showing a fit or approximation, show the progression with the control
  parameter (a light-to-dark ladder plus an error-vs-parameter panel).

## Build and verify (every agent, before reporting)

From the deck folder, a private build (never the un-tagged `make`):
```bash
cd "/Users/hun/Aleph/3. wiki/labmeeting/260929"
make BUILD=.build-<tag>.md BASE=<tag> pdf < /dev/null
grep -n "^# " .build-<tag>.md        # page = 1 + number of '---' separators before the line
pdftoppm -png -r 60 -f A -l B <tag>.pdf <scratch>/pg
```
Read every one of your pages as an image. Check: nothing crosses the footer, math is
rendered (no raw `$`), figures and posters are large and legible, no leftover template
text. Fix and rebuild until clean. Then delete `.build-<tag>.md`, `<tag>.pdf`,
`<tag>.html`. Other agents' files may be half-written while you build; ignore their pages.

**One consolidated write per slide file.** The vault is mirrored by Obsidian LiveSync,
which can echo a stale snapshot over a file during a burst of rapid edits. Compose the
whole file, write it once, and at the end `grep` each of your files to confirm the content.

Scratch space: `/private/tmp/claude-501/-Users-hun-Aleph/ded9d751-38df-429f-b1b8-df9285d37eb3/scratchpad/<tag>/`.

## Ownership (touch only your row)

| tag | slide files | clips / figures | code |
|---|---|---|---|
| `timeline` | `02-timeline.md`, `03-objectives.md` (now `18-objectives.md`) | `media/timeline.png` | `fig_timeline.py` |
| `ml` | `10-section-ml.md`, `11-what-is-ml.md`, `12-uat.md` | `uat-bumps` | `scenes_ml.py` |
| `neuron` | `13-neuron.md`, `14-xor.md` (+`14a-` if needed) | `neuron-and-or`, `xor-hidden` | `scenes_neuron.py` |
| `cnn` | `16-conv-image.md`, `17-cnn-fc.md` (the 1D slide `15-conv-1d.md` was retired to `_removed/` on 2026-09-28 at the speaker's request) | `cnn-as-fc`; `image-conv-kirby`, `sobel-kirby` (3Blue1Brown's scenes rendered with a pixel-art Kirby, `code/conv-intro-animations`, `--deck 260929`) | `scenes_cnn.py` |
| `qml` | `20-section-qml.md`, `21-qcnn.md`, `22-nonlinearity.md`, `23-skepticism.md`, `24-optimism.md` | `qcnn-pooling` | `scenes_qml.py` |
| `dqml` | `25-dqml.md`, `26-channels.md`, `27-questions.md` | `dqml-split` | `scenes_dqml.py` |
| `data` | `30-section-data.md`, `31-…` to `35-…` | wiki `mnist1d-eda-*.png`, `mnist1d-repro-table1.png`, optional `fig_data.py` | `fig_data.py` |
| `embed` | `40-section-model.md`, `41-overview.md`, `42-slice.md`, `43-fourier.md`, `44-fourier-why.md` | `data-slice`, `fourier-embed-gates` (`fourier-embed` on no slide since 2026-09-29); `model-overview.png`, `embed-circuit.png`, `coherence-argand.png` | `scenes_embed.py`, `scenes_embed_gates.py`, `fig_model_overview.py`, `fig_embed_circuit.py`, `fig_coherence.py` |
| `circuit` | `45-pqc.md`, `46-pooling.md`, `47-reuse.md` | `pool-round` | `scenes_circuit.py` |
| `link` | `48-link.md`, `48a-trainable.md`, `48b-patterns.md` | `link-rule` | `scenes_link.py` |
| `readout` | `49-readout.md`, `49-readout2.md` (round 3), `49a-loss.md`, `49b-augmentation.md` | `augment`; `readout-trained-examples-crop.png`, `readout-trained.png` (`fig_readout.py`); `poe-readout` unused since round 3 | `scenes_readout.py`, `fig_readout.py` |
| `resacc` | `50-section-results.md`, `51-accuracy.md`, `52-fourier-gain.md`, `53-linear.md`, `54-one-qpu.md` | `media/linear-classifier*.png` | `fig_linear.py` |
| `rescomm` | `55-patterns.md` (`56-shuffle.md` and `57-quantum-channel.md` were removed to `_removed/` on 2026-09-29) | wiki figs; optional `fig_rescomm.py` | `fig_rescomm.py` |
| `resabcd` | `58-links-trained.md`, `58a-trajectory.md`, `59-d-scan.md`, `59a-clusters.md` | `d-sweep` | `scenes_resabcd.py` |
| `outlook` | `60-section-doubts.md`, `61-…` to `65-…` | `fingerprint` | `scenes_outlook.py` |

The integrator owns `00`, `01`, `80`, `90`, `99`, `BRIEF.md`, the Makefile, `README.md`
of the code folder, `dqml_style.py`, `render.py`, `_index_wiki.md`, `log.md`. If you need
something added to `dqml_style.py`, put it in your own scene file instead.

## Report back (final message)

Files written; clip names, durations, poster fractions; figures made; pages rendered and
checked (page numbers); every number used with its source section; anything the speaker
must decide (also as `<!-- EDIT-FORWARD: ... -->` in the file); references added to
`REFS-<tag>.md`.
