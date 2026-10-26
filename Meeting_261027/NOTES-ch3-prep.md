# NOTES: Chapter 3, section P (DQ state preparation), slides 40-45

Draft of 2026-10-26, written against `BRIEF-ch3.md`, then reviewed the same day (see "Review" at the end). Six slides, three animation slots, no clips yet.
Preview: `./preview.sh ch3pr <scratchpad>/ch3pr 40-prep-target.md 41-prep-finite.md 42-prep-structure.md 43-prep-tau.md 44-prep-response.md 45-prep-sequence.md` (7 pages including the title; all pages checked after the review, nothing overflows the footer, no MathJax errors).

## Sources

- **N** = executed notebook `3. wiki/code/qsp-dq-preparation/qsp-dq-preparation.ipynb` (printed outputs) and its percent source `qsp-dq-preparation.py` (markdown §1-§10).
- **J** = `3. wiki/code/qsp-dq-preparation/out/designs.json` (angles, tau, T, band infidelity, Omega_DQ), read through `ch3common.load_designs("prep")`.
- **R** = `3. wiki/code/lm-261027-figures/data/prep.pkl`, the local seed-42 rerun (`dump_data.py prep`), used only for curves the notebook does not store: isolated solutions (`SOL`), fixed-tau scans (`CURVE`), Newton histories (`SOL[...]["hists"]`, `FIX_HISTS`).
- **Π** = `3. wiki/code/qsp-dq-pi/qsp-dq-pi.py` §1, §3, §4 (model, Euler form of a finite pulse, the three exact properties).
- **C** = closed-form 2x2 products evaluated in `fig_prep.py` (qspcore.propagate), no fitting.

## Slide by slide

| slide | content | sources |
|---|---|---|
| 40-prep-target | target state; $D^1(u)\lvert0\rangle$; $f=4[\mathrm{Im}(P^*Q)]^2$; two conditions; freedoms; plot (a) | N §2 (equation, conditions, freedoms, representative $R_x(\pi/2)$), N fig1 (target form) |
| 41-prep-finite | hard pulse exact, $K=0$ (plot b); Euler form of a finite $\pi/2$ pulse; DQ phase error; $1-f$ values; negative time | N §3 (text and printed 2.312e-2 / 9.004e-2, (Δ/Ω_DQ)² = 2.333e-2 / 9.331e-2); Π §3 (Euler form, 463 ns); J (Ω_DQ/2π = 14.14, 7.07 MHz in notes) |
| 42-prep-structure | three exact properties; 3 conditions for both classes; counting (isolated pal. N = 3, 4, arb. N = 2; fixed tau pal. N = 5, arb. N = 3) | N §4 (table and text), Π §4 (derivations) |
| 43-prep-tau | no overlap constraint; tau from duration and band error; plot (c) progression; plot (d) Newton cost; tau = 10 ns | N §5 printed outputs (isolated: shortest/best T and band; fixed tau: shortest T at tau = 2.0/2.4 ns); R (curves); J (designs circled) |
| 44-prep-response | plot (e) Re/Im P, Q of the four designs against the target (phase of each manifold fixed at its line centre); plot (f) 1 − f(Δ) and band 1 − f(σ) against one π/2 pulse; coherence check | J (designs), C (curves), N §8 printed arg(c_-1/c_+1) = 0 and band values, N §10 (∝ σ², caveats) |
| 45-prep-sequence | plot (g); table; arbitrary is better (half as long); 0.006π outer pulses = 0.2 ns | J (angles, T, band), N §10 table and text; N §6 and §10 (time-mirror bridge in notes) |

Numbers quoted (all from N or J unless marked):
- 2.3e-2 (20 MHz), 9.0e-2 (10 MHz): one finite π/2 pulse, 1 − f at Δ = A (N cell 6). 2.33e-2 and 9.33e-2: (Δ/Ω_DQ)² (N cell 6).
- 463 ns = 2π/|A_par| (N §3 text, Π §3).
- Ω_DQ/2π = 14.1 and 7.07 MHz (J `omega_dq_over_2pi_MHz`).
- Isolated solutions (N cell 10): shortest 139.3 ns (20 MHz, arb. N = 2, band 9.2e-6), 244.4 ns (10 MHz, arb. N = 2, band 2.1e-2); best band 5.38e-6 at 202.4 ns (20 MHz, pal. N = 4), 3.25e-4 at 388.4 ns (10 MHz, pal. N = 4).
- Fixed tau (N cell 12): shortest 74.9 ns (20 MHz arb.), 140.8 ns (20 MHz pal.), 145.2 ns (10 MHz arb.), 272.5 ns (10 MHz pal.).
- Designs at tau = 10 ns (J): 20 MHz arb. (−0.2029, 0.5462, −0.8433)π, 76.3 ns, 1.03e-6; 20 MHz pal. (−0.0062, 0.6057, −1.6990, 0.6057, −0.0062)π, 143.3 ns, 2.17e-6; 10 MHz arb. (0.2603, −0.6386, 0.8783)π, 145.7 ns, 4.57e-5; 10 MHz pal. (0.0656, −0.6994, 1.7676, −0.6994, 0.0656)π, 273.2 ns, 1.17e-4.
- arg(c_-1/c_+1) = 0 at the three line centres for all four designs (N cell 21).
- Derived, not printed in N (speaker notes only): single π/2 pulse duration (π/2)/Ω_DQ = 17.7 ns (20 MHz), 35.4 ns (10 MHz); single π/2 pulse band-averaged 1 − f = 1.5e-2 (20 MHz) and 6.0e-2 (10 MHz) (C, `band_infid`, same 40-node average as N; slide 43 and 45 notes); "each design within 3 ns of the shortest of its family" (143.3 − 140.8 = 2.5 ns is the largest gap, from N numbers); "the arbitrary designs are about four times longer" (76/18, 146/35; the palindromic ones are about eight times); "about four orders (20 MHz) and three (10 MHz)" of band error (1.5e-2 against 1.0e-6, 6.0e-2 against 4.6e-5).
- From R only: slide 43 bullet "at fixed τ, 6 to 20 steps" (fixed-τ Newton runs at τ = 10 ns: median 9-10 accepted steps, min 6-7, max 14-20; τ-free runs median 9-10, max 33-60); notes "from 1e-5 to 1e-30 in about five steps" (median 4-5 steps from cost < 1e-5 to < 1e-29; the convergence is superlinear but not quadratic, the cost exponent grows by about 1.5 per step); "for short τ" and "at long τ both families lose that advantage" (fixed-τ curves: the arbitrary family stays below the shortest isolated solution up to τ ≈ 52 ns at both Rabi frequencies, and both families keep a smaller band error than every isolated solution up to τ = 20 ns; beyond that the duration and band error grow).

## Figures (`3. wiki/code/lm-261027-figures/fig_prep.py`, run `../env/.venv/bin/python fig_prep.py [names]`)

| PNG (media/ and out/) | plot | function | data |
|---|---|---|---|
| ch3-prep-target.png | (a) target Re/Im P, Q on the bands Δ_m ± 3σ; dark = $R_x(\pi/2)$, light = other signs | `fig_target` | analytic |
| ch3-prep-hard-finite.png | (b) Chebyshev coefficients of the hard response (one coefficient, K = 0) + 1 − f(Δ) of one π/2 pulse, hard and finite, with (Δ/Ω_DQ)² | `fig_hard_finite` | C (`chebfit` of the constant hard response; finite pulse propagated) |
| ch3-prep-counting.png | counting: free parameters against N, τ free / τ fixed, 3 conditions | `fig_counting` | analytic (N §4 table) |
| ch3-prep-progression.png | (c) progression in N: band 1 − f against T (one π/2 pulse, isolated solutions, fixed-τ families, designs circled) and T against τ | `fig_progression` | R (`SOL`, `CURVE`), J (designs), C (single pulse band average) |
| ch3-prep-newton.png | (d) Newton cost against step, τ free (grey) and τ = 10 ns (colour; solid palindromic, dashed arbitrary) | `fig_newton` | R (`SOL[...]["hists"]`, `FIX_HISTS`) |
| ch3-prep-response.png | (e) Re/Im P, Q of the four designs against the target | `fig_response` | J + C; `gauge_fix(ref = Δ_m)` on the part of the axis nearest each line (split at ±A/2): P real and positive at each line centre |
| ch3-prep-finite.png | (f) 1 − f(Δ) of the designs and one π/2 pulse; band 1 − f against σ (∝ σ² guide) | `fig_finite` | J + C (Gauss-Hermite band average, 40 nodes, as in N) |
| ch3-prep-sequence.png | (g) the four pulse trains on one time axis | `fig_sequence` (`draw_sequence`) | J |

Every human plot is present: (a) slide 40, (b) slide 41 (left panel), (c) slide 43 (left figure, both panels), (d) slide 43 (right figure), (e) slide 44 (top), (f) slide 44 (bottom), (g) slide 45.

Rerun caveats (R): the converged counts of the isolated searches differ by a few from the Colab run (e.g. 20 MHz palindromic N = 3: 139/600 here, 141/600 in N; distinct 58 here, 59 in N), and some isolated solutions come out sign-flipped (all β → −β, e.g. arbitrary N = 2 at 20 MHz: (2.055, −0.555)π here, (−2.055, 0.555)π in N; equivalent, Q → −Q). Every printed design and statistic agrees. The fixed-τ curves have spikes where 120 random starts at that τ found only a longer branch (e.g. 20 MHz arbitrary at τ = 2.9 ns); the notes say so.

## Animation slots

1. **41-prep-finite** (`anim side`, 400×220): Bloch vectors $\mathbf n_m=u_m\hat z$ (= (2 Re P*Q, 2 Im P*Q, |P|² − |Q|²)) of the three manifolds (Δ = A_par m_I, A_par/2π = −2.16 MHz) during one finite π/2 pulse at 10 MHz, then during the 10 MHz arbitrary design (146 ns). Single pulse: end points on the equator around −ŷ, each turned about z by Δ t_e (t_e = 1/Ω_DQ); design: all three on −ŷ (the 10 MHz designs end on −ŷ, the 20 MHz ones on +ŷ). Data: J + time-sliced `qc.propagate` (split each pulse into short sub-pulses of the same phase; waits as R_z).
2. **42-prep-structure** (`anim side`, 310×250): the mirror symmetry; during the 20 MHz arbitrary design (or any ±x sequence) $\mathbf n_{+1}$ and $\mathbf n_{-1}$ are mirror images through the yz plane, $(n_x,n_y,n_z)\to(-n_x,n_y,n_z)$, and $\mathbf n_0$ stays in the plane. Data: J + time slicing. Reusable by the Π section.
3. **45-prep-sequence** (`anim side`, 360×270): Bloch trajectories of the three manifolds through the 76 ns 20 MHz arbitrary sequence; separate paths, common end point +ŷ. A timeline cursor on plot (g) can run in sync. Data: J + time slicing.

Slots 1 and 3 can share one Bloch-sphere scene (same camera, same colours `MCOL`).

## Decisions and disagreements (for the lead)

1. **Notebook text against outputs (N §5 "Choice")**: "both fixed-τ families are shorter than every isolated solution" is not true for the palindromic family: 20 MHz palindromic N = 5 is 140.8-143.3 ns against the isolated arbitrary N = 2 at 139.3 ns; 10 MHz palindromic 272.5-273.2 ns against 244.4 ns (isolated arbitrary N = 2, band 2.1e-2). True statement used on slide 43: for short τ the arbitrary family is shorter than every isolated solution, and both families have a smaller band error than every isolated solution (at τ = 10 ns, 20 MHz: ≤ 2.2e-6 against ≥ 5.4e-6; 10 MHz: ≤ 1.2e-4 against ≥ 3.3e-4). The qualifier "for short τ" is needed: at long τ the families are longer and worse (R curves; the N text itself says "for short τ" about the band error).
2. **Target signs**: N fig1 writes $P=\pm1/\sqrt2$, $Q=\mp i/\sqrt2$; the signs of P and Q are in fact independent ($R_x(-\pi/2)$, Q = +i/√2, is also exact, and the 20 MHz designs land there: Q = +i/√2 at all three centres). Slide 40 writes $P=\pm\tfrac1{\sqrt2}$, $Q=\pm\tfrac{i}{\sqrt2}$ with light bars for the other signs.
3. **Gauge of plot (e), one phase per manifold**: `gauge_fix` with ref Δ = 0 is a no-op here, since $u(0)=R_x(\sum\beta)$ makes P(0) real and positive for all four designs. The arbitrary designs differ from the palindromic representative on $m_I=\pm1$ by a phase of each manifold's whole state (arg P = −0.123π at Δ = +A, m_I = −1, and +0.123π at Δ = −A, m_I = +1, at 20 MHz; −0.228π and +0.228π at 10 MHz; computed with `qc.propagate`, not on a slide). That phase is unobservable: the three manifolds are orthogonal nuclear states, so the prepared mixture does not depend on it. The review therefore applies `gauge_fix(ref = Δ_m)` per manifold (on the part of the axis nearest each line, split at ±A/2, in the free regions): all four designs now sit on the bars on every band; the palindromic curves are unchanged and continuous; the dashed arbitrary curves break at ±A/2. Slide 44 says "The global phase of each manifold is fixed at its line centre", and the speaker notes explain the breaks. The first draft used a single phase and showed the dashed curves off the outer Im P, Re Q and Re P bars.
4. **Bloch picture** (slides 40-42 notes and the three slots): f = n_y², n_z = 0 is the population condition, n_x = 0 the coherence condition. This is algebra on N's equations (⟨σ_y⟩ = 2 Im P*Q), not in N; it is used only in speaker notes and slot texts.
5. **Lead's slides**: `31-design-steps.md` (current version) names "gate fidelity averaged over the ¹³C shift and the three lines" as the figure of merit. The preparation's target is a state, so this section uses the state fidelity $f=4[\mathrm{Im}(P^*Q)]^2$ averaged the same way (σ/2π = 0.1 MHz, three lines with weight 1/3), as N does. The lead may want "gate (or state) fidelity" on slide 31. Slide 31 also writes the pulse duration as $\lvert\beta\rvert/\Omega$; for the DQ applications it is $\lvert\beta\rvert/\Omega_{\rm DQ}$.

## TODO (speaker to confirm)

- Slide 40 notes: "the first step of a DQ Ramsey or echo measurement" is a general statement, not from N (softened in review from "every DQ magnetometry sequence").
- Whether to keep the 20 MHz palindromic design (outer pulses 0.2 ns, effectively three pulses) on the slide or move it to backup.
- The three clips (slots above).

## Review (2026-10-26, independent check)

Checked every slide and speaker-note number against N, J and README, re-derived the equations (Cayley-Klein middle column, f, the two conditions, the freedoms, the Euler form of a finite π/2 pulse and its DQ phase, the three exact properties, the count, the time-mirror bridge, the Bloch-vector picture) and verified with closed-form products: all four designs end on ±ŷ on all three lines with the same sign (20 MHz on +ŷ, 10 MHz on −ŷ); palindromic P = +1/√2 at all three centres; between the lines 1 − f stays below 5e-5 (20 MHz) and 2.5e-3 (10 MHz). Line positions: m_I = +1 at −2.16 MHz (A_par < 0) on every plot. Changes made:
- Slide 43: the bullet "the arbitrary family is shorter than every isolated solution, and both families have a smaller band error" was true only for short τ (R curves); now "For short τ ...". The Newton bullet now says "At fixed τ, ... in 6 to 20 steps" (τ-free runs take up to 60). The speaker notes no longer claim quadratic convergence "in two or three steps" (R: about five steps from 1e-5 to 1e-30), and add that the palindromic family is longer than the shortest isolated solution at 10 MHz.
- Plot (c): the right-panel legend crossed the 10 MHz curves; it now sits above both panels. Plot (d): the legend crossed the curves; it now sits above the axes. Both figures were re-proportioned so the fonts are larger at slide size, and slide 43 was re-fitted (the third bullet had hit the footer after the resize).
- Plot (e): one phase per manifold (decision 3 above).
- Plot (b) panel: smaller canvas, so the fonts are larger on slide 41. Plot (a): larger m_I labels.
- Slides 40, 41, 44: bullets rewritten as full sentences ("bars: ...", "Hard pulses: ...", "Finite pulse: ...", "Dashed: ..." were fragments).
- Slide 45: minus signs after commas in the angle column rendered as binary minus ("− 0.843"); grouped as `{-0.843}`. Speaker notes: "the designs are about four times longer" holds only for the arbitrary designs; the error comparison now uses band averages on both sides.
- Slide 40 notes: "every DQ magnetometry sequence" softened (TODO below).
