# 260923 Journal Meeting brief (shared contract for all section agents)

Not a slide (no leading digits, so the Makefile ignores it). Read fully before writing.

## The talk

Journal meeting, 2026-09-23, Paulee Group (KIST). Speaker: Donghun Jung. Topic: **QSP,
introduced through convolution**, preceded by an **introduction to quantum information**
for new members. Style: intuitive, animation-driven (manim clips embedded as `<video>`).
QSP was already covered on 2026-08-18 (`../260818/`), convolution on 2026-09-03
(`../260903/`); this deck re-tells both around one message:

> A quantum experiment is prepare -> unitary -> measure. What we design on purpose is the
> unitary. What we read out is a response, and that response is a *convolution*. QSP is
> the exact theory of designing that convolution.

## Speaker's outline (verbatim intent; do not drift)

### 1. Introduction to Quantum Information (files 10-19)
- Intro: three steps (prepare a state, apply a unitary, measure). The interesting part is
  the operation: we design unitaries on purpose.
- Quantum state: ket vector, complex components, norm 1, bra = Hermitian conjugate;
  everything observable is a bra-ket product so the first component can be taken real.
- Measurement: expectation value <psi|O|psi>; examples with famous qubit states and Pauli
  matrices; cyclic trace -> Tr(|psi><psi| O), density matrix. Single shots give an
  eigenvalue and collapse with the corresponding probability; repeated shots converge.
- Qubit and Bloch sphere: components are Pauli expectation values; example |+>; general
  form cos(theta/2)|0> + e^{i phi} sin(theta/2)|1> gives spherical coordinates.
  **Clip `bloch-plus`**: start with |+>, show the equations and the calculation, then the
  general form and its expectation values, then the Bloch sphere with the points.
- Our case: prepare |0>, read out in z basis via projective measurement on |0> (photon
  count is positive; effectively a z measurement).
- Quantum operation: unitary from the Schrodinger equation; time-independent H gives
  U = exp(-iHt), so U U^dag = I; time-dependent H is solved numerically, e.g. Euler
  steps; H = drift + control (intrinsic Hamiltonian + laser/microwave drive adds a term).
- Examples (Pham thesis Ch. 1, at the qubit level): ODMR, Rabi, Ramsey. Each is
  1 prepare, 2 pulse, 3 read out. Hamiltonian in the Pauli basis: the axis is the rotation
  axis, the coefficient times time is the rotation angle.
  **Clips `seq-rabi`, `seq-ramsey`, `seq-odmr`**: two panels, left Bloch sphere with the
  rotation axis and angle drawn, right the readout outcome building up as a graph.
- Entangled state: |00> -> H, CNOT -> Bell state; cannot be written as a product; partial
  trace gives (|0><0| + |1><1|)/2, rank 2, "mixed"; Bloch vector of length 0. Two
  messages: (i) entanglement looks like a mixed state in the subsystem, shorter Bloch
  vector; (ii) the source of entanglement is interaction (in NV, dipole-dipole coupling to
  nuclear spins), which acts as a conditional operation as time goes; the subsystem radius
  decays, hence the signal decays, with T1, T2, T2*.
  **Clip `bell-mixed`**: Bell-state construction and the shrinking reduced Bloch vector,
  then decay of the radius under interaction. (The speaker will supply the extended-T2 plot
  later; leave an EDIT-FORWARD placeholder, no plot.)
- Dynamical decoupling: convert conditional into unconditional. H = |0><0| x H0 + |1><1| x H1,
  U = |0><0| x U0 + |1><1| x U1. CPMG tau-pi-2tau-pi-tau: V0 = U0 U1 U1 U0, V1 = U1 U0 U0 U1.
  Both unitary, so compare rotation axis and angle. Signal ~ (1+M)/2 with
  M = Tr(V0^N V1^dag N)/2 after N repetitions. Unconditional -> no signal loss; exceptions
  at specific tau set by the nuclear configuration (Taminiau 2012). Radius of the Bloch
  vector = purity Tr rho^2 = 1 for a ket. T2 increases under DD (plot supplied later).
  **Clip `dd-cpmg`**: Bloch spheres of the nuclear spin for electron in 0 and in 1; draw
  the two rotation axes; show how the axes move as tau changes, coinciding (unconditional)
  except near the resonant tau where they open up (conditional). Source:
  taminiau2012detection.
- Summary: three steps; we design the unitary; every conclusion comes from analysing the
  response; the response also depends on the system configuration (bridge to section 2).

### 2. Convolution (files 20-29)
- Intro: the response is convoluted. Definition looks complex but is natural.
- From `../260903/`: definition first (`dice-slide` clip, `dice-grid`), dice example,
  skip image filters, pointwise product = convolution in polynomial coefficients
  (`polynomial` clip), the one identity (`conv-to-mult` clip).
- How this relates to us: DD example. Many nuclear spins -> product of responses ->
  the measured spectrum is a convolution (`../260903/51-steps.md`, `52-local.md`).

### 3. QSP (files 30-39) — DRAFT, speaker will refine later
- Backbone: `../260818/` (sequence, theorem, P and Q, reachable functions, gallery, fits).
- New emphasis: **how QSP is convolution**. Each step of the sequence (a wait/signal
  rotation followed by a phase) multiplies the 2x2 matrix polynomial by a degree-1 factor,
  i.e. convolves the coefficient list with a 2-tap kernel; the response over the signal
  is the pointwise product side (a trigonometric polynomial = Fourier series of the
  coefficients). Same shape as DD: the filter function is the Fourier transform of the
  modulation. Wiki: `3. wiki/concepts/qsp-phase-filter-design.md` (Laurent-polynomial
  form), `3. wiki/projects/qsp/conditional-gate.md` sections 2 and 4.
  **Clip `qsp-coeff-conv`**: build the sequence step by step; coefficient list on the left
  convolving with each new factor, response curve on the right sharpening with degree.

## File plan

```
00-title.md 01-outline.md            (integrator)
10-section-qi.md 11-... 19z-summary.md   agent slides-qi
20-section-conv.md 21-... 29-...         agent slides-conv
30-section-qsp.md 31-... 39-...          agent slides-qsp
80-references.md 90-thanks.md            (integrator; agents list their refs in a
                                          `REFS-<section>.md` file, not numbered)
media/                                   clips: <clip>.mp4 + <clip>.png
```
Use `NNa-`, `NNb-` to insert. Each file carries its own frontmatter:
```
---
marp: true
theme: serif
math: mathjax
---
```
Deck-wide CSS is in `00-title.md` (`.dense` 23px, `.tight` 21px via `<!-- _class: dense -->`).
Do NOT edit `00-title.md`, `01-outline.md`, the Makefile, or another agent's files.

## Embedding a clip (mandatory form)

```html
<figure class="figure">

<video src="media/CLIP.mp4" poster="media/CLIP.png" width="550" autoplay loop muted playsinline preload="none"></video>

*Caption as a markdown italic line with $math$ allowed.*

</figure>
```
`preload="none"` is required (PDF build hangs otherwise). Blank lines around the tag are
required (markdown must not be inside a raw HTML block). The PDF prints the poster PNG.

## Clip contract (names are fixed; slides reference them before they exist)

| clip | section | content |
|---|---|---|
| `bloch-plus` | 1 | |+> ket, <X>,<Y>,<Z> computed, point on sphere; then general (theta, phi) state, expectation values, sweeping point |
| `seq-rabi` | 1 | left Bloch sphere (rotation about x by Omega t), right P(|0>) vs pulse length |
| `seq-ramsey` | 1 | pi/2, free precession about z by delta tau, pi/2; right P vs tau |
| `seq-odmr` | 1 | pi pulse at detuning delta: tilted axis; right P vs delta (dip) |
| `bell-mixed` | 1 | |00> -> H -> CNOT; reduced Bloch vector shrinks to 0; then radius decay under interaction |
| `dd-cpmg` | 1 | nuclear-spin Bloch spheres for electron 0 / 1; CPMG axes vs tau (Taminiau) |
| `dice-grid`, `dice-slide`, `dice-weighted`, `polynomial`, `conv-to-mult`, `simple-example` | 2 | already in media/ (from 260903) |
| `qsp-coeff-conv` | 3 | coefficient list convolving step by step; response sharpening with degree |

Animation code lives in `3. wiki/code/lm-260923-animations/` (`scenes_bloch.py` for
section 1, `scenes_qsp.py` for section 3). Render with
`../env-manim/.venv/bin/python render.py scenes_bloch.py SceneName -q l` (preview) and
`-q h` (final); the driver installs `media/<clip>.mp4` + `.png`. Clip length 10-40 s,
looping. Use manim community 0.21 (`from manim import *`), seeded/deterministic. Use
`tex_template=TexTemplate(preamble=r"\usepackage{amsmath,amssymb,physics}")` for `\ket`.

## Writing rules (the speaker is strict)

- Full sentences ending in periods; no telegraphic fragments.
- **No em-dashes (`—`) anywhere.** Use commas, colons, periods, or words. En-dash only in
  compound names.
- Every symbol and Greek letter in math mode (`$\theta$`), including in captions.
- Explain the physical meaning of each equation and symbol.
- One idea per slide; ~12 lines max at 26px; footer overflow is the #1 failure.
- Callouts (`<div class="callout">`) at most 1 per section.
- Category pills in the H1: `# <span class="cat intro">Intro</span> Title` with
  intro | method | strategy | results | ongoing.
- Two columns: `<div class="columns"><div class="col">` ... `</div><div class="col">` ... `</div></div>`
  with blank lines around markdown inside.
- No math inside `<figcaption>` or `<li>`; captions are markdown italic lines.
- Cite: Pham thesis (1. raw/papers/Pham_gsas.harvard_0084L_10993.pdf, Ch. 1) for
  ODMR/Rabi/Ramsey/T1/T2/T2*; Taminiau et al., PRL 109, 137602 (2012) for DD; Martyn et
  al., PRX Quantum 2, 040203 (2021) and Motlagh & Wiebe, PRX Quantum 5, 020368 (2024) for QSP.
- Mark anything the speaker must confirm with `<!-- EDIT-FORWARD: ... -->`.

## Verify

From `260923/`: `make html` (fast) and `make` (PDF), then render pages with
`pdftoppm -png -r 60 -f A -l B 260923.pdf /Users/hun/.claude/jobs/6a5953d2/tmp/pg` and
Read the PNGs. The build concatenates ALL files, so build the whole deck; only check your
own pages. Fix overflow before reporting.

## Addendum: chapter 1 is split per section (one agent per section)

| agent | slide files | clip | scene file |
|---|---|---|---|
| qi-basics | `10-section-qi.md`, `11-three-steps.md`, `12-state.md`, `13-measurement.md`, `19z-summary.md` | none | none |
| qi-bloch | `14-bloch.md` (+`14a-`), `15-our-case.md` | `bloch-plus` | `scenes_bloch_plus.py` |
| qi-unitary | `16-unitary.md` (+`16a-`, `16b-`) | none (static figure allowed) | none |
| qi-sequences | `17-examples.md`, `17a-rabi.md`, `17b-ramsey.md`, `17c-odmr.md` | `seq-rabi`, `seq-ramsey`, `seq-odmr` | `scenes_seq.py` |
| qi-entanglement | `18-entanglement.md`, `18a-mixed.md`, `18b-decay.md` | `bell-mixed` | `scenes_bell.py` |
| qi-dd | `19-dd.md`, `19a-cpmg.md`, `19b-axes.md` | `dd-cpmg` | `scenes_dd.py` |

All Bloch-sphere scenes import the shared helper `bloch.py` in
`3. wiki/code/lm-260923-animations/` (`BlochSphere`, `StateArrow`, `AxisLine`,
`readout_axes`, `bloch_vector`, `rot`, `axis_angle`, `setup_camera`, `TEX`). Read its
docstring; do not edit it (if you need something, add it in your own scene file).
Clips are dark-background (matching the 260903 3Blue1Brown clips). Build privately:
`make BUILD=.build-<tag>.md BASE=<tag> pdf` from the deck folder, delete the outputs after.
