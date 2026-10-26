# NOTES: Chapter 3, section Π (DQ π pulse), slides 50-56

Draft of 2026-10-26, written against `BRIEF-ch3.md`. Seven slides, three animation slots, no clips yet.
Preview: `./preview.sh ch3pi <scratchpad>/ch3pi 50-pi-target.md 51-pi-finite.md 52-pi-mirror.md 53-pi-solve.md
54-pi-response.md 55-pi-finite-comparison.md 56-pi-sequence.md` (8 pages including the title; every page read; no
footer overflow, no MathJax errors, tables complete).

## Sources

- **N** = executed notebook `3. wiki/code/qsp-dq-pi/qsp-dq-pi.ipynb` (printed outputs, cells 1, 6, 10, 12, 14, 17, 21)
  and the markdown of its percent source `qsp-dq-pi.py` (§1-§10); `README.md`.
- **J** = `3. wiki/code/qsp-dq-pi/out/designs.json` (via `ch3common.load_designs("pi")`): angles, τ, T, band infidelity.
- **JP** = `3. wiki/code/qsp-dq-preparation/out/designs.json` (via `load_designs("prep")`): the preparation designs
  used in the time-mirror construction.
- **NP** = `3. wiki/code/qsp-dq-preparation/qsp-dq-preparation.py` §6 and §10 markdown (time-mirror construction,
  the N = 2 / N = 3 example at τ = 47.06 ns).
- **R** = `3. wiki/code/lm-261027-figures/data/pi.pkl`, the local seed-42 rerun (`dump_data.py pi`), used only for
  what N does not store: isolated solutions (`SOL[...]["uniq"]`), fixed-τ scans (`CURVE`), Newton histories
  (`SOL[...]["hists"]`, `FIX_HISTS`).
- **C** = closed-form 2×2 products in `fig_pi.py` (`qspcore.propagate`), with N's own formulas (spin-1 gate fidelity
  `infid1`, 40-node Gauss-Hermite band average). Check: C reproduces J's four band infidelities to all printed digits
  (2.061793e-6, 1.880882e-6, 9.136578e-5, 8.580482e-5) and N's single-pulse values (4.553e-2, 1.693e-1).
  `../env/.venv/bin/python fig_pi.py check` prints every C number quoted below.

## Slide by slide

| slide | content | sources |
|---|---|---|
| 50-pi-target | target $\lvert\pm1\rangle\to\lvert\mp1\rangle$; $D^1(u)$ at $P=0$; same in every manifold iff $Q_m^2=Q_0^2$; target $P=0$, $Q=\mp i$ on the bands; $F=[(4t^2-1)^2+3]/12$, $t=\lvert\mathrm{Im}\,Q\rvert$, $1-F\simeq2(\lvert P\rvert^2+(\mathrm{Re}\,Q)^2)$; repetition $US_zU^\dagger=-S_z$ ⇒ refocusing; plot (a) | N §1 ($D^1(u)$), §2 (target, fidelity derivation, repetition paragraph), N fig1 (target form); the expansion $1-F=2\epsilon-\tfrac43\epsilon^2$ in the speaker notes checked by hand |
| 51-pi-finite | hard $R_x(\pi)$ exact for all Δ, $K=0$ (plot b); first-order $F(\pi)=R_x(\pi)[1-i(\Delta/\Omega_{\rm DQ})Y]$; y rotation not imitable by free evolution; $1-F$ = 4.6e-2 / 0.17 on $m_I=\pm1$; conditions 3 / 4; counting ⇒ fixed-τ palindromic N = 5, arbitrary N = 4; isolated palindromic N = 3, 4, arbitrary N = 3 | N §3 (expansion, β = π statement, 463 ns), N cell 6 (\|P\| = 0.152 / 0.300, Δ/Ω_DQ = 0.153 / 0.305, 1 − F = 4.553e-2 / 1.693e-1), N §4 (three properties, conditions table, counting), N fig2 |
| 52-pi-mirror | reversing ±x sequences transposes $u$; $u=v^Tv$ has $P=p^2+q^2=0$ for $q=\pm ip$; 20 MHz prep (−0.203, 0.546, −0.843)π mirrored = symmetric π design, 153 ns, twice the preparation; 10 MHz mirror = design with all phases shifted by π, same gate | NP §6 (construction and proof), N §10 ("symmetric 20 MHz design is the arbitrary N = 3 preparation followed by its time mirror"), JP + J (angles, T), C `check` (see below) |
| 53-pi-solve | plot (c): band 1 − F against T for all solutions (N = 1 star, isolated N = 3, 4, fixed-τ families) and T against τ; plot (d): Newton cost against step; isolated palindromic N = 4 at 20 MHz 236 ns (τ = 20.1 ns); T flat below τ ≈ 20 ns; τ = 10 ns; "about ten steps" | N cell 10 (isolated: 236.1 ns at 20.08 ns, band 6.01e-6; 278.7 ns at 47.06 ns, 1.84e-5; 454.0 ns at 40.52 ns, 3.26e-4), N cell 12 (fixed-τ shortest T), N §5 "Choice" (τ ≲ 20 ns / 30 ns, composite pulses as τ → 0, τ = 10 ns), N §5 (LM-damped Newton, exact Jacobian, cost < 1e-30, milliseconds); R (curves); J (rings); C (N = 1 band values, plotted only); R median steps (below) |
| 54-pi-response | plot (e): Re/Im P, Q of the four designs against the target; symmetric Re Q ≡ 0; arbitrary Re Q = 0 at the centres only; P grows away from the lines, faster at 10 MHz | N §7 markdown (same statements), N fig5; J + C (curves) |
| 55-pi-finite-comparison | plot (f): spin-1 1 − F against Δ (designs and one π pulse) and band 1 − F against σ; unitaries coincide at the centres to 1e-15; band ∝ σ², 2.1e-6 / 9.1e-5 at σ/2π = 0.1 MHz | N §8 markdown (quadratic degradation, σ² scaling), N cell 21 (max \|D1(u_m) − D1(u_0)\| = 2.1e-15, 7.1e-16, 1.7e-15, 7.9e-16), N §10 ("coincide to 1e-15"), J (band values); C (curves) |
| 56-pi-sequence | plot (g); table of the four designs | J (angles, T, band), N §10 table (rounded values identical), N §10 caveats (speaker notes) |

The universal-NOT section (Euler form, band target, symmetric QSP, Newton solver) and the preparation section
(DQ model, K = 0, three exact properties, τ = 10 ns) are referred to in one phrase each, not re-derived.

## Numbers quoted (slides and speaker notes)

From N or J unless marked.
- Lines Δ_m/2π = ±2.16, 0 MHz; Ω_DQ/2π = 14.142 and 7.071 MHz (N cell 1, J).
- 2π/A = 463 ns (N §3).
- One finite π pulse at Δ = A: \|P\| = 0.152 (20 MHz), 0.300 (10 MHz); Δ/Ω_DQ = 0.153, 0.305; 1 − F = 4.553e-2, 1.693e-1 (N cell 6). Slides: 4.6e-2, 0.17.
- Conditions 3 (palindromic) / 4 (arbitrary); parameters ⌈N/2⌉ (+1 with τ) and N (+1) (N §4 table).
- Isolated (N cell 10): 20 MHz palindromic N = 3 best 278.7 ns, τ = 47.1 ns, 1.84e-5; palindromic N = 4 best 236.1 ns, τ = 20.1 ns, 6.01e-6; 10 MHz palindromic N = 4 best 454.0 ns, τ = 40.5 ns, 3.26e-4; 10 MHz N = 3 best 524.9 ns, τ = 83.7 ns, 7.45e-4.
- Fixed τ (N cell 12): shortest 149.8 ns at τ = 2.0 ns (20 MHz sym.), 151.4 ns at 3.6 ns (20 MHz arb.), 290.5 ns at 2.0 ns (10 MHz sym.), 293.8 ns at 7.7 ns (10 MHz arb.). Not quoted on slides; they support "T flat below 20 ns".
- Designs (J; N cell 14 and §10): 20 MHz sym. (−0.2029, 0.5462, −1.6866, 0.5462, −0.2029)π, 152.6 ns, 2.06e-6; 20 MHz arb. (0.4697, 1.6355, −1.2528, 0.1476)π, 153.9 ns, 1.88e-6; 10 MHz sym. (−0.2603, 0.6386, −1.7566, 0.6386, −0.2603)π, 291.3 ns, 9.14e-5; 10 MHz arb. (−0.4977, −1.6408, 1.3674, −0.2289)π, 294.1 ns, 8.58e-5.
- Preparation (JP): 20 MHz arb. (−0.2029, 0.5462, −0.8433)π, 76.30 ns; 10 MHz arb. (0.2603, −0.6386, 0.8783)π, 145.67 ns.
- Isolated N = 2 preparation (−2.055, 0.555)π at τ = 47.06 ns mirrored = palindromic N = 3 π pulse (−2.055, 1.110, −2.055)π (NP §6; N cell 17 prints it as (2.055, −1.1101, 2.055)π, the same gate with all phases shifted by π). Speaker notes of 52 only.
- Light shift of order 40 kHz at 20 MHz (N §10 caveat 1, marked "[estimate]" there). Speaker notes of 56 only.

Derived here (C, `fig_pi.py check`), speaker notes or slide wording only:
- Time mirror, 20 MHz: max \|mirror − design\| = 1.4e-14 π; max \|D1(mirror) − D1(design)\| at the line centres 1.9e-15; T(design)/T(preparation) = 2.000000 (no wait between the merged middle pulses). Slide 52: "twice the preparation".
- Time mirror, 10 MHz: mirror = −design to 7.1e-16 π (every pulse phase shifted by π); D1 equal at the line centres to 2.0e-15; \|ΔF\| < 2.9e-15 for \|Δ\|/2π ≤ 5 MHz; T ratio 2.000000. Slide 52: "an equivalent gate". Physical reason (speaker notes): a common phase π of the drive is conjugation by R_z(π), Q → −Q, D^1 unchanged where P = 0, t = \|Im Q\| unchanged everywhere.
- Total rotation Σ\|β\| = 3.185π, 3.506π (20 MHz), 3.554π, 3.735π (10 MHz). Notes of 56: "3.2π to 3.7π".
- Largest 1 − F between the lines: 4.8e-5 / 4.4e-5 (20 MHz), 2.2e-3 / 2.1e-3 (10 MHz). Notes of 55: "below about 5e-5 and 2e-3".
- 1 − F at 5 MHz: designs 2.7e-2 / 2.5e-2 vs one pulse 2.2e-1 (20 MHz, "about ten times better"); 5.4e-1 / 5.2e-1 vs 6.0e-1 (10 MHz, "approach the single pulse"). Notes of 55.
- Band 1 − F of one plain π pulse: 3.0e-2 (20 MHz), 1.1e-1 (10 MHz). Plotted (stars on slide 53, dotted lines on slide 55), not quoted.
- Single π pulse duration π/Ω_DQ = 35.4 ns (20 MHz), 70.7 ns (10 MHz): x positions of the stars.
- Newton steps (R): median accepted steps at τ = 10 ns 9-10 for all four families (max 24); isolated (τ free) median 9, 95th percentile 21, max 57 (733 converged histories). Slide 53 "about ten steps"; notes "median about the same, a few starts up to about 60". The notebook figure fig3 shows the same.

## Figures (`3. wiki/code/lm-261027-figures/fig_pi.py`; `../env/.venv/bin/python fig_pi.py [names]`)

| PNG (media/ and out/) | slide | plot | function | data |
|---|---|---|---|---|
| ch3-pi-target.png | 50 | (a) Re/Im P, Q targets on the bands Δ_m ± 3σ; Im Q = −1 (dark) or +1 (light, sign free) | `fig_target` | analytic (N §2) |
| ch3-pi-chebyshev.png | 51 | (b) Chebyshev coefficients of the hard target, $Q=-iT_0$: one coefficient, K = 0 | `fig_chebyshev` | analytic |
| ch3-pi-count.png | 51 | adapted (b)/(c): free parameters against N, angles only (fixed τ) and angles + τ, palindromic and arbitrary; red rings fixed-τ square systems, grey rings isolated | `fig_count` | analytic (N §4 table) |
| ch3-pi-mirror-sequence.png | 52 | preparation v (violet), its mirror v^T (orange), merged product = 20 MHz symmetric design (blue) | `fig_mirror` (`draw_sequence`) | JP + J |
| ch3-pi-solutions.png | 53 | (c) progression in N: band 1 − F against T (N = 1 stars, isolated N = 3 circles/triangles, N = 4 diamonds, fixed-τ N = 5 plus, N = 4 cross, designs ringed); T against τ | `fig_solutions` | R (`SOL`, `CURVE`), J (rings), C (stars) |
| ch3-pi-newton.png | 53 | (d) Newton cost ½\|r\|² against accepted step (0-60, so every converged history is shown): τ = 10 ns fixed (blue 20 MHz, red 10 MHz), isolated τ free (grey) | `fig_newton` | R (`FIX_HISTS`, `SOL[...]["hists"]`) |
| ch3-pi-response.png | 54 | (e) Re/Im P, Q of the four designs against the target (P and Re Q zoomed to ±0.35) | `fig_response` | J + C |
| ch3-pi-finite.png | 55 | (f) spin-1 1 − F against Δ (designs, one π pulse dotted); band 1 − F against σ with a σ² slope guide (grey dash-dot, offset by 0.2 below the 20 MHz symmetric curve so it stays visible) | `fig_finite` | J + C (Gauss-Hermite 40 nodes, 21 σ values 0.01-0.5 MHz) |
| ch3-pi-sequence.png | 56 | (g) the four pulse trains; columns share ns per inch | `fig_sequence` (`draw_sequence`) | J |

No gauge fixing is needed here: P and Q of a π pulse are compared directly (the free sign of Q is shown by the
light +1 target bar), unlike the arbitrary preparation designs.

## Plot coverage

(a) slide 50 · (b) slide 51 (left figure; count figure on the right is the adapted progression/condition count) ·
(c) slide 53 (left figure, both panels) · (d) slide 53 (right figure) · (e) slide 54 · (f) slide 55 (both panels) ·
(g) slide 56.

## Animation slots

1. **52-pi-mirror** (`anim side`, 340×280): the 20 MHz preparation v (76.3 ns) takes the Bloch vectors $\mathbf n_m=u_m\hat z$ of the three manifolds to the equator (for this design all on +ŷ, see NOTES-ch3-prep); then v^T is appended, the two adjacent −0.843π pulses merge into −1.687π, and all three vectors reach the south pole together at 152.6 ns. A timeline cursor on `ch3-pi-mirror-sequence.png` can run in sync. Data: JP + J, time-sliced `qc.propagate` (split pulses into sub-pulses of the same phase, waits as R_z). Can share the Bloch scene of the preparation slots.
2. **54-pi-response** (`anim side`, 340×280): Bloch trajectories of the three manifolds through the 20 MHz symmetric design (152.6 ns): different paths, common end point at the south pole with one phase; then one plain finite π pulse (35.4 ns) for comparison, $m_I=\pm1$ missing the pole by the y rotation (\|P\| = 0.152, about 17.5° from the pole). Data: J + time slicing; Δ_m = A_par m_I.
3. **55-pi-finite-comparison** (`anim side`, 340×280): a train of 20 MHz symmetric π pulses with free evolution between them (CPMG-like spacing t/2, π, t, π, t/2); phase of the DQ coherence ρ_{+1,−1} for $m_I=-1,0,+1$ and a few ¹³C-shifted copies (δ = ±σ, ±2σ) winding at 2(Δ_m + δ) and refocusing midway between consecutive pulses (each block t/2, π, t/2 equals the pulse U for every static Δ, since US_zU† = −S_z); after every pair of pulses the state is back to the start (u² = −1, D¹(u²) = 1). Data: J + propagation of D^1(u) and free evolution; the spacing t has to be chosen (TODO).

## TODO (speaker to confirm)

- Slot 3: choose the spacing t of the decoupling train, and decide whether the clip should also show the residual band error accumulating per pulse (about 2e-6 per pulse at 20 MHz). A train is not simulated in N; the refocusing statement itself is N §2 (exact algebra).
- 40 kHz light-shift estimate (N §10 caveat 1, "[estimate]" in N), speaker notes of 56.
- Whether to recommend the symmetric or the arbitrary design for the experiment (notes say "either"; arbitrary has one fewer pulse but its shortest pulse is 5 ns at 20 MHz, the most sensitive to rise time, N §10 caveat 4).
- The existing clip `media/real-pulse-bloch.mp4` is not embedded here (finite pulses are introduced in the universal-NOT section); slot 2 could reuse its Bloch-sphere style.

## Disagreements and caveats (notebook text against outputs, rerun against notebook)

- N §10 lists the isolated palindromic N = 3 solutions as "47.1 ns (279 ns) and 83.7 ns (525 ns)"; these are the best-band ones. At 10 MHz the shortest N = 3 solution is 488.9 ns at τ = 208.6 ns with band 4.0e-2 (N cell 10). Not an error, a selection; slides quote only the 20 MHz N = 4 case.
- NP §6 says doubling the arbitrary τ = 10 ns preparation gives a π pulse "of about twice its duration"; it is exactly twice (C: ratio 2.000000), because the merged middle pulses have no wait between them.
- N §10 states only the 20 MHz mirror relation; the 10 MHz pair (mirror = design with all phases shifted by π) is verified here (C) and is new in the slides.
- "Coincide to 1e-15" (N §10) against printed maxima 7.1e-16 to 2.1e-15 (N cell 21): order of magnitude; slide 55 keeps "to $10^{-15}$".
- Rerun R against N: converged counts of the isolated searches differ by a few (e.g. 20 MHz palindromic N = 3: 126/600 in R, 124/600 in N; distinct 28 vs 27), and some isolated solutions are sign-flipped (arbitrary N = 3 best at 20 MHz: (−2.055, 1.1101, −2.055)π in R, (2.055, −1.1101, 2.055)π in N; equivalent gates). Line-centre 1 − F at the 1e-15 level differs (rounding). Every design and printed statistic agrees; all designs on the slides come from J.
- The fixed-τ curves (R) are the shortest solution among 120 random starts at each τ; they have spikes where the starts missed the shortest branch (e.g. 20 MHz arbitrary at τ = 5.2 ns: 160.3 ns against about 151 ns at the neighbours). Plotted as markers so the spikes read as single points.

## Review of 2026-10-26 (independent check against the notebook outputs and designs.json)

Every slide number was re-derived from the notebook printed outputs, `designs.json` (both notebooks) and `fig_pi.py
check`; all agree. Equations re-derived by hand: $D^1(u)$ at $P=0$; $t=\lvert\mathrm{Im}\,Q\rvert$ from
$\mathrm{Tr}(iXu)=-2\,\mathrm{Im}\,Q$; $F=[(4t^2-1)^2+3]/12$ and $1-F=2\epsilon-\tfrac43\epsilon^2$; the first-order
finite-pulse expansion and $\lvert P\rvert\simeq\Delta/\Omega_{\rm DQ}$ at $\beta=\pi$; $R_z(a)R_x(\pi)R_z(b)=R_x(\pi)R_z(b-a)$;
$v^T$ for the reversed sequence, $P=p^2+q^2=0$, $Q=2i\,\mathrm{Im}(p^*q)=\pm i$; the $R_z(\pi)$ frame argument for the
10 MHz mirror ($P\to P$, $Q\to-Q$); $US_zU^\dagger=-S_z$ and the refocusing block. Signs: $m_I=+1$ at $-2.16$ MHz in
every figure. Checked here: the 20 MHz preparation puts all three lines on $+y$ (10 MHz: all on $-y$); the sign of
$Q$ at the line centres is $+i$ for both symmetric designs and the 10 MHz arbitrary one, $-i$ for the 20 MHz arbitrary
one; one plain π pulse leaves $m_I=\pm1$ at 17.5° (20 MHz) and 34.9° (10 MHz) from the pole.

Changes made in the review:
- 51: the finite-pulse propagator was written $F(\pi)$ on the same slide as the fidelity $1-F$; renamed
  $U_{\rm finite}(\pi)$ (slide and notes). Bullets rewritten as full sentences. Notes: operator order of the $y$
  rotation stated correctly (it acts before the ideal inversion).
- 52: bullets as full sentences; "the same gate" for 10 MHz softened to "an equivalent gate" (identical $D^1$ only where
  $P=0$, identical fidelity everywhere). Notes: "the shortest isolated solution" was wrong (the isolated palindromic
  $N=4$, 236 ns, is shorter than the $N=3$, 279 ns); now "the isolated $N=3$ solution".
- 53: title "proper $\tau$" (same as the other two sections); opening sentence qualified "at short $\tau$" (the
  notebook's statement holds for $\tau\lesssim20$-30 ns). Notes: "isolated problems take longer, up to about 50 steps"
  contradicted the rerun (median 9 for both); now "median about the same, longer tail up to about 60".
- 54: bullets as full sentences; notes: the 10 MHz arbitrary design also has $\mathrm{Im}\,Q=+1$.
- 55: bullet as a full sentence, units kept on one line; slot text and notes: phases refocus midway between
  consecutive pulses (not "after every pair"); "below about 2e-3" corrected to "at most about 2e-3" (max 2.2e-3).
- 56: title "pulse sequence" (as in the other sections); notes on why four arbitrary pulses suffice reworded.
- Figures: solutions (frequency labels moved off the rings), Newton (x axis to 60 so the two longest isolated
  histories are not cut; legend on an opaque background), finite (σ² guide no longer hidden under the 20 MHz curve;
  legends moved so nothing overlaps a curve).

- Name collision found during the review: the clip pipeline (`3. wiki/code/lm-261027-animations/scenes_ch3_pi.py`,
  `PiMirror`) writes `media/ch3-pi-mirror.mp4` with its poster `media/ch3-pi-mirror.png`, which overwrote the static
  figure of slide 52. The static figure is now `media/ch3-pi-mirror-sequence.png` (fig_pi.py `fig_mirror`); the clip's
  poster was left untouched. The same pipeline plans `media/ch3-pi-decoupling.mp4` for slot 3 (no collision).
