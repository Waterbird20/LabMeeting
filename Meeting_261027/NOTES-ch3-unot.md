# NOTES: Chapter 3, section U (universal NOT, SQ qubit)

Draft of 2026-10-26 (rebuild of the old Chapter 3 universal NOT; the archived `_archive/2026-10-26-ch3-v1/32-36, 99i, 99o`
supplied wording only; their numbers, the 8-pulse design and the stopband strategy are obsolete).

## Files

| slide file | title | figure(s) | animation slot |
|---|---|---|---|
| `32-unot-intro.md` | Intro: Universal NOT: goal | `media/ch3-unot-lines.png` | yes (Bloch rotations) |
| `33-unot-target.md` | Strategy: target function | `media/ch3-unot-target.png` (plot a) | no |
| `34-unot-phases.md` | Method: pulse phases | `media/ch3-unot-axes.png` | no |
| `35-unot-tau.md` | Strategy: proper τ | `media/ch3-unot-tau.png` | yes (τ sweep) |
| `36-unot-degree.md` | Strategy: proper K | `media/ch3-unot-kladder.png` (plot c) | yes (K ladder) |
| `37-unot-newton.md` | Method: Newton solver | `media/ch3-unot-newton-cost.png` (plot d), `media/ch3-unot-chebyshev.png` (plot b) | no |
| `38-unot-hard.md` | Results: hard pulses | `media/ch3-unot-response.png` (plot e), `media/ch3-unot-sigma.png` (part of f) | no |
| `39-unot-finite.md` | Results: finite pulses | existing clip `media/real-pulse-bloch.mp4` (+ poster), `media/ch3-unot-finite.png` (plot f) | existing clip embedded |
| `39a-unot-sequence.md` | Results: pulse sequence | `media/ch3-unot-sequence.png` (plot g) | no |

Figure script: `3. wiki/code/lm-261027-figures/fig_unot.py` (run from that folder with
`../env/.venv/bin/python fig_unot.py`; about 5 s; writes `out/ch3-unot-*.png` and copies to `media/`). Deterministic
(no random numbers drawn). Preview: `./preview.sh ch3u <scratch>/ch3u 32-unot-intro.md … 39a-unot-sequence.md`,
10 pages (title + 9), checked page by page on 2026-10-26: no footer overflow, no MathJax errors. Reviewed and
rebuilt as `ch3ur` on 2026-10-26 (see "Review fixes" below); every page re-read after the last change.

## Human's plots (a) to (g)

| plot | where | file |
|---|---|---|
| (a) target, Re/Im P, Q against Δ | slide 33 | `ch3-unot-target.png` |
| (b) Chebyshev decomposition (k on x, coefficient on y) | slide 37 (right) | `ch3-unot-chebyshev.png` |
| (c) varying K, against Δ and Δτ | slide 36 | `ch3-unot-kladder.png` |
| (d) Newton cost against step | slide 37 (left) | `ch3-unot-newton-cost.png` |
| (e) optimised response against target | slide 38 | `ch3-unot-response.png` |
| (f) finite-pulse comparison | slide 39 (1 − F against Δ, 3 implementations × 2 Rabi frequencies, + band table); σ scan on slide 38 | `ch3-unot-finite.png`, `ch3-unot-sigma.png` |
| (g) pulse sequence | slide 39a | `ch3-unot-sequence.png` |

## Figure provenance (function in `fig_unot.py` → data)

Source notebook: `3. wiki/code/qsp-unot-sq/` (`qsp-unot-sq.py`, executed `qsp-unot-sq.ipynb`, `out/designs.json`,
`README.md`). Rerun data: `lm-261027-figures/data/unot.pkl` (seed-42 local rerun via `dump_data.py`).

- `ch3-unot-lines.png` ← `fig_lines()`: closed form, axis of one rectangular π pulse
  n = (Ω, 0, Δ)/√(Ω² + Δ²) at Ω/2π = 20 (solid) and 10 MHz (dotted), Δ/2π ∈ [−4, 4] MHz; circles = target axes
  (x on m_I = −1 at +2.16 MHz, y on m_I = 0, z on m_I = +1 at −2.16 MHz). **Replaces** the archived
  `media/unot-lines-one-pulse.png`, which labels m_I = −1 at −A and +1 at +A (wrong for A_∥/2π = −2.16 MHz).
- `ch3-unot-target.png` ← `fig_target()`: targets from the notebook §2 table (TGT), bands Δ_m ± 3σ, σ/2π = 0.1 MHz;
  light bars = −u (as in the notebook's fig1).
- `ch3-unot-axes.png` ← `fig_axes()`: schematic (Bloch sphere, xz plane shaded, target axes x, y, z in the line colours).
- `ch3-unot-tau.png` ← `fig_tau()`: closed form min(|sin Aτ/2|, |sin Aτ|) (notebook §3 definition), τ_sep = 2π/3A;
  circle e^{iΔ_m τ} at τ = 105.89 ns (designs.json `hard.tau_ns`, filled) and τ_sep (open). The K = 3 marker line is
  ink (HARD), not violet, because violet is the m_I = −1 colour in the circle panel.
- `ch3-unot-kladder.png` ← `fig_kladder()`: **K = 1, 2 and K = 4 (τ_sep) from the rerun pickle** (`BEST[1]`, `BEST[2]`,
  `BEST['4sep']`); **K = 3 from designs.json `hard`** (the rerun kept the mirror-equivalent K = 3 design). Bottom row:
  each K at its own τ over Δτ ∈ [−π, π], with that K's target bars in its colour. x ticks at the line centres
  (top) and ±0.5 (bottom) so adjacent panels' edge labels do not collide. Note: the best K = 1 fit (notebook and
  rerun alike) sits at τ = 231.5 ns = π/A, where m_I = ±1 coincide; the rerun's K = 4 member has Re P ≡ 0.
- `ch3-unot-newton-cost.png` ← `fig_newton()`: **rerun pickle** `RUNS[3][*]['hist']` (Newton cost after every accepted
  step, 300 starts). Colour split by final cost: ≤ 1e-20 "converged" (290 starts in the rerun), > 1e-20 "stalled"
  (10 starts; 5 of them above 1e-6, 4 at cost 0.1 to 1). x range 0 to 110 steps (all runs shown). Legend moved to
  the empty right-middle region.
- `ch3-unot-chebyshev.png` ← `fig_chebyshev()`: FFT on 64 points of θ = Δτ/2 of the designs.json `hard` design,
  exactly the notebook §5 recipe; c_k (T_k) black, s_k (sin θ U_{k−1}) amber, k = 0…6. Values equal the notebook
  print: Re P: s1 = −0.4853, s3 = 0.3832; Im P: c1 = 0.3832, c3 = −0.3832, s1 = −0.1393, s3 = −0.4897;
  Re Q: c1 = −0.4231, c3 = −0.5769; Im Q: c1 = −0.3832, c3 = 0.3832, s1 = −0.1393, s3 = −0.4897; all others 0.
- `ch3-unot-response.png` ← `fig_response()`: designs.json `hard`, hard pulses, against the target bars (x ticks
  at the line centres). Line centres checked: P = +i on m_I = +1, Q = −1 on m_I = 0 (both −T), Q = −i on m_I = −1.
- `ch3-unot-sigma.png` ← `fig_sigma()`: band-averaged 1 − F (30-point Gauss-Hermite, notebook `band_infid`) for
  σ/2π ∈ [0.01, 0.5] MHz, designs.json `hard` and the two refined designs (legend "refined, 20/10 MHz"); dotted σ²
  reference through the σ = 0.1 MHz point.
- `ch3-unot-finite.png` ← `fig_finite()`: designs.json `hard` played with finite pulses (gap τ between pulse edges),
  with Euler gaps τ − t_e(β_k) − t_e(β_{k+1}), and the designs.json refined designs; 1 − F against the target of the
  nearest line, Ω/2π = 20 (blue) and 10 MHz (red). Recomputed band errors match the notebook print exactly
  (1.99e-2, 1.16e-2, 4.21e-3; 6.21e-2, 3.35e-2, 4.12e-3) and durations (358.9, 320.4, 333.6; 400.2, 323.1, 351.1 ns).
- `ch3-unot-sequence.png` ← `fig_sequence()` (+ `draw_hard()` for the hard row; `ch3common.draw_sequence` for the
  finite rows): designs.json `hard`, `20MHz_refined`, `10MHz_refined`; phases shown as the effective axis phase
  (φ + π for β < 0), as in the README table.

## Slide → sources and every number quoted

- **32 intro.** Gate assignment and Δ_m = A_∥ m_I, A_∥/2π = −2.16 MHz: notebook §1 and README; `qspcore.py`.
  r → −r/3, F = 2/3: Bužek, Hillery, Werner, PRA 60, R2626 (1999) (from the archived slide) and the algebra in the notes.
  n_z = 0.11 (20 MHz), 0.21 (10 MHz) at the outer lines: closed form (computed in `fig_unot.py`, 0.107 and 0.211).
  Equal weights 1/3 and the polarisation caveat: notebook §9 caveat 3.
- **33 target.** P, Q table: notebook §2 table. F = (|Tr T†u|² + 2)/6: notebook §2. σ/2π = 0.1 MHz, ±3σ bands:
  notebook (SIGMA, `shade`). "No stopband": brief (human's design workflow).
- **34 phases.** Mirror symmetry, u(0) = R_x(Σβ), symmetric sequences ⇒ n_y = 0: notebook §2 markdown.
  Symmetric QSP refs: Dong, Meng, Whaley, Lin, PRA 103, 042419 (2021); Wang, Dong, Lin, Quantum 6, 850 (2022)
  (as given in the brief).
- **35 τ.** Non-overlap rule Aτ/2, Aτ ∉ πℤ and τ_sep = 2π/(3A) = 154 ns (154.32): notebook §3 and README.
  Overlap at π/A = 231 ns and 2π/A = 463 ns: closed form (231.48, 462.96 ns). 106 ns: designs.json (105.89 ns).
- **36 K.** Counting 9 conditions against 2(K+1) (+1): notebook §4. K = 1, 2 band 1 − F 0.22 and 2.8e-2: notebook §4 print
  (2.25e-01, 2.85e-02; the unrounded value 0.028459 gives 2.8e-2). τ = 105.9 ns: notebook print / designs.json.
  Notes: recurring τ (±106, ±181 ns mod 463 ns): README and notebook §9; 3τ = 318 ns (317.7), 4τ_sep = 617 ns (617.3),
  K = 4 band 4.15e-3 (written 4.1e-3; unrounded 0.0041493): notebook §4 print.
- **37 Newton.** LM-damped Newton with exact Jacobian from prefix/suffix products: notebook §4 text and
  `qspcore.newton_lm`, `propagate_grads`. 245/300 exact (tolerance cost < 1e-30): notebook §4 print, now quoted
  in the speaker notes only (the slide figure is the rerun, where the split at 1e-20 gives 290/10, so the slide
  bullet says "most converge in about 20 steps; a few stall").
  "About 20 steps": rerun pickle (runs with final cost ≤ 1e-20: 10th / 50th / 90th percentile 10 / 14 / 20 accepted
  steps; 92 % within 20). Only odd k ≤ 3:
  notebook §5 and its print. QSPPACK/Newton ref: Dong, Lin, Ni, Wang, SIAM J. Sci. Comput. 46, A2951 (2024) (brief).
- **38 hard pulses.** τ = 105.9 ns, 3τ = 318 ns, band 1 − F = 4.3e-3 (4.28e-3), exact at centres (1 − F ≤ 2.2e-16),
  σ² growth and "intrinsic" argument: notebook §4/§7 prints, §9 "Band error", README. Flattening estimate (nine more
  conditions, about nine pulses, approaching 1 µs): notebook §9, marked there as an estimate.
- **39 finite pulses.** Euler form t_e = tan(β/2)/Ω: notebook §7 (and archived 99i-real-pulse.md for the clip
  explanation). Table 2.0e-2 / 6.2e-2, 1.2e-2 / 3.4e-2, 4.2e-3 / 4.1e-3: notebook §7 print and README. Notes:
  7 and 8 Newton steps, 334 / 351 ns: notebook §7 print; "outermost R_z(Δ t_e) do not cancel": notebook §9.
- **39a sequence.** Angles, phases, τ, T: README table and designs.json (rounded to 3 decimals). 1.5 ns second pulse
  (0.0596π at 20 MHz = 1.49 ns): designs.json, notebook §9 caveat 4. Notes: leakage ~1e-5 and light shift ~40 kHz are
  the notebook's own estimates (§9 caveat 1).

## Animation slots (no clips yet)

1. **Slide 32 (`.anim.side`, 340×300).** Three Bloch spheres (m_I = +1, 0, −1) rotate one input vector by R_z(π),
   R_y(π), R_x(π); a fourth marker shows the average −r/3; the input then sweeps over the sphere and the average
   always points opposite with length 1/3. Data: closed form only.
   Optional extension: the same three spheres under the actual K = 3 design (designs.json `hard`) at the three line
   centres, showing the trajectories pulse by pulse.
2. **Slide 35 (`.anim.side`, 330×260).** τ sweeps 0 → 500 ns: dots e^{iΔ_m τ} on the circle (m_I = 0 fixed at 1,
   m_I = ±1 counter-rotating at ±Aτ), the distance-from-overlap curve traced alongside, dots turn red when two
   coincide (231 ns: ±1; 463 ns: all three), markers at 106 ns (K = 3) and 154 ns (τ_sep). Data: closed form; the
   static version is the right panel of `ch3-unot-tau.png`. Model: the old `media/tau-sweep.mp4`.
3. **Slide 36 (`.anim.side`, 310×230).** The best response morphs K = 1 → 2 → 3 → 4 (top-row view of
   `ch3-unot-kladder.png`: Re/Im P, Q against Δ, target bars), with a counter of the band 1 − F (0.22, 2.8e-2,
   4.3e-3, 4.1e-3). Data: rerun pickle `BEST[1]`, `BEST[2]`, `BEST['4sep']`; designs.json `hard` for K = 3.
   (Interpolating between different K is not physical; cross-fade, or grow one pulse at a time.)
- **Slide 39:** the existing clip `media/real-pulse-bloch.mp4` (poster `real-pulse-bloch.png`) is embedded at
  width 410. The clip writes the Euler time as `e`; the slide writes `t_e` (same quantity).

## Disagreements and caveats found

1. **Notebook text against its output.** §5 markdown says "Its exact solutions all share τ = 106 ns", but the §4
   print lists exact K = 3 solutions at τ = 105.89, 180.67, 282.29, 357.07, 568.85, 745.25 ns (i.e. ±106 and ±181 ns
   modulo 463 ns, as §9 and the README say). The kept design (smallest band error) has τ = 105.89 ns. The slides
   follow the print/README.
2. **Rerun against notebook (counts).** The brief says the rerun reproduces every printed statistic; the converged
   counts differ slightly: K = 3: 245/300 (notebook) against 250/300 (rerun), best cost 1.80e-32 against 2.55e-32;
   K = 4 (τ_sep): 279 against 280. Of the rerun's 50 non-converged K = 3 starts, 40 stopped at cost 1.0e-30 to
   2.8e-21 (residual below 1e-10, exact for all purposes, just above the strict 1e-30 tolerance). The speaker notes quote the notebook's 245/300; plot (d) is drawn
   from the rerun and splits converged/stalled at 1e-20 (290/10), so the figure is not a picture of "245".
3. **Rerun against notebook (K ladder).** The rerun's K = 1 best fit is a different least-squares solution from the
   notebook's (same cost 0.5, same band 2.25e-1, but line-centre 1 − F [6.7e-1, 0, 6.4e-4] against notebook
   [6.4e-1, 0, 2.4e-2]); the rerun's K = 4 (τ_sep) is a different member of the one-parameter family (same band
   4.15e-3). So the K = 1 and K = 4 curves of plot (c) differ in shape from the notebook's fig3. K = 2 is the same
   solution up to a sign convention (same β magnitudes, τ, band).
4. **Notebook text "Euler-compensated gaps halve this".** Measured factors 0.58 (20 MHz) and 0.54 (10 MHz); the
   speaker notes say "about halves".
5. **designs.json phase convention.** `phi_over_pi` is stored with negative β for pulses 3 and 4 (0.813, 0.071);
   the README and the slides give the effective axis phase φ + π (1.813, 1.071) with positive angles.
6. Archived intro figure `media/unot-lines-one-pulse.png` has the m_I labels mirrored (see above); not used.

## TODO (speaker to confirm)

- The speaker notes of slide 37 say Newton is "much faster than gradient descent on a loss": this is the human's
  statement in the brief; no timing comparison is in the notebook.
- ¹⁴N polarisation at 440 G (equal weights 1/3 assumed): to be measured (old deck caveat).
- Rise time of the microwave chain against the 1.5 ns pulse (slide 39a): to be checked on the setup.
- The ¹³C model is a static Gaussian shift with rms 0.1 MHz per line (notebook caveat 2); the slides use it as given.

## Review fixes (independent review, 2026-10-26, preview tag ch3ur)

- Slide 32: title "Universal NOT" → "Universal NOT: goal" (title rule); first bullet made a full sentence.
- Slide 33: bullet "Light bars show −u" → "−T_m" (the bars are target values); speaker notes likewise.
- Slide 34: bullet adds "the same gate" for −R_z(π); speaker-note sentence "everything depends on x = cos(Δτ/2)"
  replaced (with x-only pulses Im P and Re Q are odd in Δ, so the response is not a function of cos(Δτ/2) alone; the
  correct statement is the mirror relation: Re P, Im Q even, Im P, Re Q odd). Figure enlarged to 400 px.
- Slide 35: "2π/3A" → "2π/(3A)"; figure: K = 3 marker ink, circle legend moved below the circle, red dots unclipped.
- Slide 36: counting equation rewritten so the "+1 (τ)" applies only to the τ-free case and the fixed-τ_sep count is
  9 ≤ 2(K+1); bullets shortened; figure tick collisions fixed and fonts enlarged.
- Slide 37: "245/300 starts converge" replaced (conflicted with the rerun figure's 290/10 split); 245/300 kept in the
  speaker notes with the tolerance; figure heights matched (h:300) so the two bullets align; legend moved.
- Slide 38: bullets made full sentences; σ-scan legend "finite" → "refined"; tick collisions fixed; σ figure resized
  to clear the footer.
- Slide 39a: opening "slightly larger angles" was false for the second pulse at 20 MHz (0.066π → 0.060π) → "larger main
  angles"; speaker note "a few hundredths of π" → "less than 0.07π in angle and 0.05π in phase".
- Slide 37, plot (d): renamed `media/ch3-unot-newton.png` → `media/ch3-unot-newton-cost.png` (fig_unot.py
  `save(fig, "ch3-unot-newton-cost")`). During the review a concurrent animation run (`lm-261027-animations/
  scenes_ch3_unot_b.py`, clip `ch3-unot-newton.mp4`) wrote its poster frame to `media/ch3-unot-newton.png` and so
  overwrote the static Newton-cost figure. That poster is now the clip's file and was left untouched; the lead should
  know the clip `ch3-unot-newton.mp4` exists for slide 37, which has no animation slot in this draft.
