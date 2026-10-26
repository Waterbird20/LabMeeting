# NOTES — 261023 deck: provenance (kept out of the slides at the human's request, 2026-10-23)


## NOTES-A: provenance for Part I (agent A, revision 1, 2026-10-23)

The slides and speaker notes carry no internal source names (SPEC Revision 1, R1). This file records where every equation, number and figure comes from. Notebook = `3. wiki/code/dq-pulse-notebook/dq-pulses.ipynb` (state of 2026-10-22). Content contract = archived `3. wiki/code/lm-261023-animations/_removed/manim-slides/SPEC.md` §3, Part I.

| slide file | content | sources |
|---|---|---|
| `01-outline.md` | three parts | deck structure set by the human (SPEC §3, R3) |
| `10-section-dq.md` | section card "01" | none |
| `11-dq-transition.md` | level diagram; SQ phase $\gamma Bt$ and DQ phase $2\gamma Bt$; doubled sensitivity | notebook intro (cell 0) and §1.1 (cell 3); Mamin et al., PRL 113, 030803 (2014) |
| `12-rotating-frame.md` | lab Hamiltonian with two frequency components; rotating frame + RWA; $\tilde H$ 3×3 in $(\lvert{+1}\rangle,\lvert0\rangle,\lvert{-1}\rangle)$; $\Delta_{1,2}=D\pm\gamma B-\omega_{1,2}$; $\mathrm{diag}(\Delta_1,0,\Delta_2)=\Delta S_z+\Delta_cS_z^2$; physical origin of $\Delta$ and $\Delta_c$ | notebook §1.1 (cell 3), which reproduces Mamin et al. Eq. (S3); the split checked by hand ($S_z=\mathrm{diag}(1,0,-1)$, $S_z^2=\mathrm{diag}(1,0,1)$) |
| `12a-spin-half.md` | balanced, mirrored drive ⇒ $\tilde H=\Delta S_z+\Omega_{DQ}(\cos\varphi S_x+\sin\varphi S_y)$; $h$; $e^{-i\tilde Ht}=D^1(e^{-iht})$; triplet of two spin-½, $U=u\otimes u$; $D^1(u)$ in $P,Q$; DQ π pulse $P=0$; $D^1(-u)=D^1(u)$ | notebook §1.1 (cell 3, "derived here"), §2.1 (cell 9, the $D^1(u)$ matrix and populations). Off-diagonal check: $\gamma B_{\rm mw}/(2\sqrt2)\,e^{\mp i\delta}=\frac{\Omega_{DQ}}{\sqrt2}e^{-i\varphi}$ = matrix elements of $\Omega_{DQ}(\cos\varphi S_x+\sin\varphi S_y)$, verified by hand |
| `13-nitrogen.md` | full electron–¹⁴N Hamiltonian; table of term sizes ($A_\parallel$ −2.16 MHz, $A_\perp$ −2.70 MHz → 2.5 kHz shifts, $P$ −4.95 MHz, $\gamma_nB$ 74 kHz, nuclear drive ≈ 7 kHz); secular block form $H=\sum_mH_m\otimes\lvert m\rangle\langle m\rvert$, $U=\sum_mU_m\otimes\lvert m\rangle\langle m\rvert$; mixed-unitary channel; $\Delta_m=mA$, $p_m=1/3$ | notebook §2 (cell 7: Hamiltonian, size table, secular approximation, channel) and §2.3 (cell 14: weights 1/3, dephasing by a static distribution); Doherty et al., Phys. Rep. 528, 1 (2013). Block form written out here from cell 7's statement "block diagonal in $m_I$" |
| `14-one-pulse.md` | inversion profile; $q=1-\eta_m^2+O(\eta^4)$; $1-P_{-1}\approx\frac43\eta_A^2$; table at three drives; π/2 phase; common-$Q^2$ requirement | notebook §2.2 (cell 11 expansions), Table 2.1 (cell 12 output): $\eta_A$ 0.069 / 0.153 / 0.305; $1-P_{-1}$ 6.34e-3 / 3.05e-2 / 1.15e-1; process infidelity 8.45e-3 / 4.05e-2 / 1.50e-1; preparation $P_0$ 7.0e-7 / 1.7e-5 / 2.6e-4; phase ±7.9° / ±17.5° / ±34.9°. Cell 5: Mamin drive $\Omega_{DQ}/2\pi$ = 31.25 MHz, π pulse 16 ns. Common $Q^2$: archived SPEC §3 Part I and notebook §3.1 target ($Q=s_me^{i\chi}$); "populations alone are not sufficient for DQ sensing": notebook intro (cell 0) |

## Figures

| file | used on | provenance |
|---|---|---|
| `media/dq-levels.png` | 11 | `3. wiki/code/lm-261023-figures/fig_dq.py`, `fig_levels()`, 2026-10-23; schematic, not to scale |
| `media/dq-inversion.png` | 14 | `fig_dq.py`, `fig_inversion()`, 2026-10-23; closed form (Rabi's formula, $\Omega_{DQ}T=\pi$, transfer $q^2$), drives $\Omega_{DQ}/2\pi$ = 31.25, 14.14, 7.07 MHz. The script prints the nuclear averages 6.343e-3, 3.047e-2, 1.146e-1, which match Table 2.1 |

Dropped in revision 1: `dq-manifolds.png` (stem plot of the three detunings; slide 13 now carries the Hamiltonian instead) and the notebook figure `dq-pulses-c12.png` (was on the deleted `14a-phase.md`).

## Clips
None for Part I (R3).

## Rendering note
`\dyad{m}` (and `\op`, `\outerproduct`) are **not** defined in this Marp MathJax build: they render as red error text, even though the kit README says the physics package is available. `\ket`, `\bra` and `\ketbra{a}{b}` work. Slide 13 therefore uses `\ketbra{m}{m}`.

## NOTES-B: provenance for part II (slides 20–26), agent B

Revision 1 (2026-10-23). Provenance lives here, not in the slides. The speaker notes contain only what the speaker says.

## Sources
- Content authority: archived manim-slides `SPEC.md` §3 Part II (`3. wiki/code/lm-261023-animations/_removed/manim-slides/SPEC.md`), checked against `dq-qsp-prep.ipynb` §3.2 (markdown cell 7). That cell has the notation, the Laurent form, the Chebyshev form ($z^n\pm z^{-n}$), completion by Fejér–Riesz and layer stripping, including $\cot(\vartheta_K/2)=\lvert q_d/p_d\rvert$ and $\phi_K=\arg(q_d/p_d)-\pi/2$.
- Archived speaker-note wording reused from `_removed/manim-slides/part2a_qsp.py` and `part2b_qsp.py`.
- Literature (cited on the slides and in the notes): Low, Yoder, Chuang, PRX 6, 041067 (2016); Martyn, Rossi, Tan, Chuang, PRX Quantum 2, 040203 (2021), Thm. 1; Haah, Quantum 3, 190 (2019).

## Slide by slide
| file | content source | figure / clip |
|---|---|---|
| `20-section-qsp.md` | the human's framing (SPEC Revision 1, R3 row B) | none |
| `21-pulses-off.md` | SPEC §3 "Pulses off"; dq-qsp-prep cell 7 "Chebyshev form" | clip A1 `media/qsp-pulses-off.mp4` + poster `.png`, from `3. wiki/code/lm-261023-animations/scenes_qsp.py` (`PulsesOff`, 24.5 s, 1920×1080, rendered `-q h` 2026-10-23, poster at 0.9). Curves computed from $\cos d\theta$ at render time; `_verify()` checks $R_z(\Delta\tau)^d=R_z(d\Delta\tau)$, $P=z^{-d}$, $T_3$, $T_5$. |
| `22-chebyshev.md` | SPEC §3 "Chebyshev" | `media/qsp-chebyshev.png` by `3. wiki/code/lm-261023-figures/fig_qsp.py` (`fig_chebyshev`) |
| `23-sequence.md` | SPEC §3 notation; dq-qsp-prep cell 7 "Instantaneous pulses" | `media/qsp-timeline.png` by `fig_qsp.py` (`fig_timeline`), K = 5, illustrative bar heights |
| `24-wait-and-pulse.md` | SPEC §3 "Pauli form" (induction proof); the explicit recursions were derived for this slide: one wait gives $A\to xA-(1-x^2)C$, $B\to xB-(1-x^2)D$, $C\to xC+A$, $D\to xD+B$; one $x$ pulse gives $A\to cA-sB$, $B\to cB+sA$, $C\to cC+sD$, $D\to cD-sC$. Both are checked numerically in `fig_qsp.py` `verify()`. | none |
| `24a-what-is-reachable.md` | SPEC §3 "Pauli form", "Reachability" (theorem, count, corollary); dictionary $P=A-i\sin\theta\,C$, $Q=-iB+\sin\theta\,D$ | none |
| `25-back-to-pulses.md` | dq-qsp-prep cell 7 "QSP and layer stripping"; SPEC §3 "Reachability" | none |
| `26-populations.md` | SPEC §3 "Populations" | `media/qsp-populations.png` by `fig_qsp.py` (`fig_populations`): example K = 4 about x, angles (π/4, π/2, π/2, π/4), $c_n=(1/8,3/16,1/8,1/16)$. The amplitude is from 2×2 products; the coefficients were obtained by a cosine fit, then checked exactly in `verify()`. |

## Checks
- `fig_qsp.py` `verify()` (seed 42) passes 13975 checks against 2×2 matrix products at random θ, φ. It covers:
  - every formula on slides 21–26;
  - the Pauli-form recursions, degrees, parities and unitarity, and the $P,Q$ dictionary;
  - the layer-stripping formula for $G_K$ and the count;
  - the population cosine series and $P_0=(1-2q)^2$;
  - the populations example;
  - the convention map: in the ± basis our $u$ is the complex conjugate of Martyn et al.'s $U_\Phi$, with their phases $=\phi_k/2$.
- Run: `"3. wiki/code/env/.venv/bin/python" "3. wiki/code/lm-261023-figures/fig_qsp.py"`.
- Clip: `cd "3. wiki/code/lm-261023-animations" && ../env-manim/.venv/bin/python render.py scenes_qsp.py PulsesOff -q h --media-dir out/manim-B --poster 0.9`.

## Not from the sources
- The populations example sequence (π/4, π/2, π/2, π/4) was chosen for this slide. It is a closed-form illustration, not a design from the notebooks.
- The remark in the 25 notes that the choice of root in each pair changes only the phase of $P$ is standard. On the circle, $\lvert P\rvert$ is fixed by $1-\lvert Q\rvert^2$.

## Removed in revision 1
- Slides `21-signal`, `22-pulses-off`, `23-chebyshev`, `24-pulses-on`, `24a-examples`, `25-two-coordinates`, `26-reachable` and `27-populations` are deleted. The ± basis flip lattice and the Ramsey example are gone. The Laurent ↔ Chebyshev content moves to agent C's slide 36.
- Figures deleted from `media/` and `out/`: `qsp-flip-lattice`, `qsp-lattice-d1`, `qsp-lattice-d2`, `qsp-zpaths`, `qsp-ramsey`, and the static `qsp-pulses-off` (now the clip poster).

## NOTES-C: provenance for Part III, first half (slides 30 to 36a)

Agent C, revision 1 (2026-10-23). Provenance lives here, not in the slides. Content source throughout is
`3. wiki/code/dq-pulse-notebook/dq-pulses.ipynb` §3.1 to §3.3 (markdown cells 19 to 28) as summarised in the
archived `3. wiki/code/lm-261023-animations/_removed/manim-slides/SPEC.md` §3. Static figures come from
`3. wiki/code/lm-261023-figures/fig_target.py` and clips from `3. wiki/code/lm-261023-animations/scenes_target.py`.
Both are closed-form, and each has a self-check (`verify()` / `_verify()`) that runs before drawing.

| slide file | content source | figures / clips |
|---|---|---|
| `30-section-target.md` | none (section card) | none |
| `32-target.md` | dq-pulses §3.1 (cell 20, Table 3.1; cell 22: one rectangular π pulse at 10 MHz, \|P\| = 0.300 at ±A, E = 0.992) | clip `media/target-build.mp4` (+ poster `.png`), scene `TargetBuild`. The π-pulse response is the exact SU(2) rotation at Rabi frequency (10 MHz)/√2, the DQ spin-½ representation; `_verify` reproduces \|P(±A)\| = 0.300 and E = 0.992 and checks against a matrix exponential. |
| `34-tau-periodicity.md` | dq-pulses §3.3 (cell 26): compatibility rule and the p = A / 2π/3A / 2π/1.5A cases. The shift is 2π/τ, not the 4π/τ printed in cell 26 (W(Δ + 2π/τ) = −W(Δ); checked numerically in both scripts). | clip `media/tau-sweep.mp4`, scene `TauSweep`: circle coverage and window images m A + n p computed from the intervals at render time; `_verify` checks the compatibility rule against an exact interval test. |
| `34a-tau-table.md` | dq-pulses Table 3.3 (cell 27 output, cell 28 text). τ values and (K − 1)τ re-checked in `fig_target.py verify()`. | none |
| `35-easy-targets.md` | dq-pulses §3.3 (cell 26, point case: R_x(π/2) W(2π/A) R_x(π/2)), Appendix E.2 (windows-only: one π pulse) | clip `media/resonances-to-x.mp4`, scene `ResonancesToX`; figure `media/target-point-q2.png` (`fig_point_q2`, \|Q\|² = cos²(πΔ/A); max 0.345 in the stopband, checked). |
| `36-target-expansion.md` | Laurent ↔ Chebyshev: dq-qsp-prep §3.2 as summarised in archived SPEC §3 (Part II); step coefficients a_k = 4 sin(0.2πk)/(πk): dq-pulses cell 23, re-derived by quadrature in `verify()` for both sign patterns | figure `media/target-step-coeffs.png` (`fig_step_coeffs`) |
| `36a-gibbs.md` | dq-pulses cell 25 (Gibbs, r ≈ 0.92, E ≈ 0.2) | clip `media/gibbs-ladder.mp4`, scene `GibbsLadder` (K = 2, 4, 6, 8, 10, 14, 18, 26, 42; E of the scaled series printed per K); figure `media/target-step-error.png` (`fig_step_error`, θ grid of 20001 points) |

Clip durations (1080p60, poster at 0.9 of the clip): target-build 19.7 s, tau-sweep 24.7 s,
resonances-to-x 19.5 s, gibbs-ladder 25.8 s. Rendered 2026-10-23 with
`render.py scenes_target.py <Scene> -q h --poster 0.9 --media-dir out/manim-C`.

Numbers that differ from the earlier SPEC:
- The uniform error of the scaled truncated series for K = 12 to 26 is 0.163 (K = 23) to 0.316 (K = 15), not
  0.17 to 0.28. The notebook's grids give the same values. The slide says only E ≈ 0.2.
- The peak overshoot of the partial sum at K = 42 is 8.3 %. The asymptotic Gibbs value is 8.95 % of the jump, so
  the slide says "about 9 %" and the clip labels it "Gibbs overshoot" without a number.

Removed in revision 1: `31-five-choices.md` (deleted); `33-one-function.md` (moved to agent D as 38a);
`36-step-in-x.md` (split into 36 and 36a). The static figures `target-circles.png`, `target-point-case.png` and
`target-step-ladder.png` were replaced by clips and deleted.

## NOTES-D — provenance for agent D's slides (261023 deck, revision 1, 2026-10-23)

Sources: `3. wiki/code/dq-pulse-notebook/dq-pulses.ipynb` (abbreviated *nb*) and its stored output `dq-pulse-notebook/out/selective_qsp.json` (*json*). Numbers were checked against the archived manim SPEC §3 (`3. wiki/code/lm-261023-animations/_removed/manim-slides/SPEC.md`) and the notebook tables on 2026-10-22 and 2026-10-23. None of this provenance appears on the slides or in the speaker notes (SPEC revision 1, R1).

## Slides

| file | content sources | figures / clips |
|---|---|---|
| `37-upper-limit.md` | nb §3.2 (cell 23 text, cell 25 reading); Table 3.2 (E_K, K = 13, 14, 15, 16; 6019 ns) | clip **A6** `media/qsp-best-approx.mp4` + poster `.png` |
| `38-k14-sequence.md` | nb §3.2 decomposition, §3.3 (cells 26, 28); Table 3.4 angles and phases; E = 5.218e-4 | `media/fig_results_k14_train.png` (from Table 3.4, by `fig_results.py`); `media/dq-pulses-c27-a-nt.png` = cell 27 panel (a), title strip cropped |
| `38a-palindromic.md` | content of the old slide 33 (nb §3.2, cell 23; §3.4 cell 29); symmetry derivations from the archived SPEC §3 Part II, checked in the archived `backup_math.py` `_verify()`; cone-program check Table 3.2 (3e-4 at K = 12, 14); trained 5.148e-4 both (Table 3.5/3.6) | none |
| `39-training.md` | nb §3.4 (cells 29, 31); Table 3.5; Wang, Dong, Lin, Quantum 6, 850 (2022) in the notes | `media/fig_results_trained.png` (json H_runs min E per (mode, K), B.series E_K) |
| `40-real-pulse.md` | nb §3.5 (cell 32: finite pulse, first-order equivalence e_k = tan(β/2)/Ω_k, g_j); App. E.1 Table E.4 (1.4e-6, 5.3e-4, 6.3e-3, 0.11) | clip **A7** `media/real-pulse-bloch.mp4` + poster |
| `40a-not-retrained.md` | nb Table 3.7 and cell 34 reading (|Q| at 4A: 0.89 / 0.25 / 0.05; 8.0e-3 vs 3.2e-2) | `media/dq-pulses-c33-a-nt.png` = cell 33 panel (a), title strip cropped |
| `41-retrained.md` | nb Table 3.8, Table 3.10, cell 36 reading | `media/fig_results_retrained.png` (json F_runs min E per (pulse, Ω, sym, K), B.series) |
| `42-take-home.md` | draft from the archived SPEC §5 | none |
| `80-references.md` | `1. raw/references.bib`: mamin2014multipulse, doherty2013nitrogen, martyn2021grand, wang2022energy, motlagh2024generalized; unfiled: Low–Yoder–Chuang 2016, Low–Chuang 2017 (PRL), Low–Chuang 2017 (arXiv:1707.05391), Haah 2019 | none |
| `90-thanks.md` | closing slide as in the 260923 deck | none |
| `99-backup.md` | section card | none |
| `99b-pauli-form.md` | archived SPEC §3 Part II (Pauli form, dictionary, symmetries), verified in the archived `backup_math.py` | none |
| `99c-d2-both-bases.md` | archived SPEC §3 Part II (examples), verified in the archived `backup_math.py` | none |
| `99d-earlier-designs.md` | nb App. E.1, Tables E.1–E.3 (durations), E.6 (nine-level model) | `media/dq-pulses-c60-ab.png` = cell 60 panels (a), (b) + legend (crop of 2026-10-22) |
| `99e-wide-identity.md` | nb App. E.3, Table E.12 (36/68/104; 2701/5170/7948 ns), Tables E.13/E.14 (d_min 39, 5.9–6.8 µs), Table E.16 (48 vs 21, 109 vs 41) | `media/dq-pulses-c74-a-nt.png` = cell 74 panel (a), title cropped |
| `99f-continuation.md` | nb §3.5 cell 36 reading (continuation numbers, 39 of 524 runs), Table 3.10 | `media/dq-pulses-c35-1-a-nt.png` = cell 35 second figure panel (a), title cropped |
| `99g-degree-scaling.md` | Tables 3.2 and E.12 for the data; Low & Chuang 2017, Martyn et al. 2021 for the law; constants c ≈ 1.2, 1.6 derived here from the slopes | `media/fig_results_scaling.png` |

Dropped in revision 1 (embedded notebook legends with "upper limit", "decomposed", "Euler" wording): `dq-pulses-c24-a/-c`, `c30-a/-b`, `c35-a`, `c35-1-b`, `c74-c`; `fig_results_tilt.png` and `fig_results_euler.png` (replaced by clip A7; files removed). `99a-notation.md` deleted (vault conventions).

## Figures — `3. wiki/code/lm-261023-figures/fig_results.py`
Run with `3. wiki/code/env/.venv/bin/python fig_results.py`; deterministic, no random numbers; writes `out/` and the deck `media/`.
- `fig_results_k14_train.png`: Table 3.4 angles / π as bars, phase π → down (−x), phase 0 or 2π → up (+x); teal waits τ = 2π/A.
- `fig_results_trained.png`, `fig_results_retrained.png`: minima over stored runs in the json, no computation beyond `min`.
- `fig_results_scaling.png`: Tables 3.2 and E.12.
- `crop_titles()`: `dq-pulses-c27-a-nt`, `c33-a-nt`, `c35-1-a-nt`, `c74-a-nt` (top 30 px removed).

## Clips — `3. wiki/code/lm-261023-animations/scenes_results.py`
Rendered 2026-10-23 with `render.py ... -q h --media-dir out/manim-D --poster 0.95`; `_verify()` (run `../env-manim/.venv/bin/python scenes_results.py`) checks every plotted formula.
- **A6 `BestApprox` → `media/qsp-best-approx.mp4` (22.1 s).** r(θ) = Σ c_k cos kθ from json `B.series[d]` for d = 1, 3, …, 17 (K = 2 … 18), θ = πΔ/A at τ = 2π/A; E_K dots from the same entries. `_verify()` recomputes the uniform error of every plotted series on the target set and matches the stored E within 5 %.
- **A7 `RealPulse` → `media/real-pulse-bloch.mp4` (25.3 s).** Bloch trajectories by Rodrigues rotation about n = (1, 0, Δ/Ω)/√(1+(Δ/Ω)²) by Ωt√(1+(Δ/Ω)²), Ωt = π, Δ/Ω = 0, 0.23, 0.46; then β = π/2 at Δ/Ω = 0.46: real pulse against z-rotation(Δe) · x-rotation(β) · z-rotation(Δe), e = tan(β/2)/Ω. `_verify()` checks the Bloch map against the SU(2) action and the second-order agreement (‖U − W R W‖ = 1.1e-5, 4.3e-5, 2.3e-2 at Δ/Ω = 0.01, 0.02, 0.46; without the waits 7.1e-3, 1.4e-2, 0.32).

## TODO (speaker to confirm)
1. `42-take-home.md`: the take-home wording (archived SPEC §5 draft).
2. `80-references.md`: titles of the unfiled Low–Yoder–Chuang 2016, Low–Chuang 2017 (PRL), Low–Chuang 2017 (arXiv:1707.05391) and Haah 2019 were written from memory; file them with `aleph-ref` to verify.
3. `99g`: that Low & Chuang 2017 (arXiv:1707.05391) and Martyn et al. 2021 state the gapped sign-function degree bound d = O(ln(1/ε)/Δθ) in this form; the constants c ≈ 1.2 and 1.6 (speaker notes) are this deck's own estimate from three and four points.
4. `40-real-pulse.md`: the slide quotes the error 1.4e-6 "at the drive of Mamin et al." (Ω_DQ/2π = 31.25 MHz, from the notebook), which is a parameter choice, not a result of that paper.

## Notebook / archived SPEC discrepancies
- The archived SPEC and the notebook use W(Δ + 4π/τ) = −W(Δ); the period of Q² is 2π/τ (handled on agent C's slide 34; my slides use only "Q² repeats every 2π/τ = A").
- "At most 1.2e-13 discarded in layer stripping" (nb cell 25) is not visible in Table 3.2 (K ≤ 18: max 5.1e-14; K = 14: 6.9e-15); not quoted on the slides.

---

# Revision 2 (2026-10-23)

## NOTES-A: Part I, revision 2 (agent A, 2026-10-23), new and changed items only

The earlier Part I provenance is already merged into `NOTES.md`. This file lists only what revision 2 adds or changes. Notebook = `3. wiki/code/dq-pulse-notebook/dq-pulses.ipynb` (state of 2026-10-22). Figure script = `3. wiki/code/lm-261023-figures/fig_dq.py` (closed form, deterministic; run with `3. wiki/code/env/.venv/bin/python`).

## Slide files

| slide file | status | content | sources |
|---|---|---|---|
| `01-outline.md` | notes changed | Part II now described as K waits, N = K+1 pulses (R2.0) | SPEC R2.0 |
| `11-dq-transition.md` | figure and notes changed | level diagram now states $2\gamma B$ = 2.46 GHz (44 mT); notes give B = 44 mT, transitions at 1.64 and 4.10 GHz | field from the human (R2); $\gamma_e/2\pi$ = 28 GHz/T as in notebook cell 4 |
| `12a-spin-half.md` | rewritten | why $D^1$ is natural: (i) two spin-½ in the triplet, $\tilde H=h\otimes\mathbb 1+\mathbb 1\otimes h$ ⇒ $u\otimes u$; (ii) same commutation relations; (iii) classical precession of $\langle\mathbf S\rangle$ | notebook §1.1 (cell 3: "Interpretation and limits", $\frac12\otimes\frac12=1\oplus0$, $U=u\otimes u$). (ii) and (iii) are standard angular-momentum facts, not in the notebook |
| `12b-dq-pi.md` | new (split from 12a) | $u$ in $P,Q$; $D^1(u)$; $P=0$ form; $P_0=(1-2q)^2$, $P_{\pm1}=2q(1-q)$; $D^1(-u)=D^1(u)$ | notebook §2.1 (cell 9) and §1.1 (cell 3); the $P=0$ matrix checked by hand |
| `13-nitrogen.md` | changed | title now "block diagonal in $m_I$"; table at 44 mT: nuclear Zeeman 135 kHz (was 74 kHz at 24.1 mT), $A_\perp$ shifts "2 to 5 kHz" (was "about 2.5 kHz" at 24.1 mT); $H_m$ written in the lab frame | notebook §2 (cell 7) for the Hamiltonian and sizes at 24.1 mT; 44-mT values: closed-form estimates printed by `fig_dq.py` (`field_estimates`): $\gamma_nB$ = 3.077 MHz/T × 44 mT = 135 kHz; $A_\perp^2/(D\mp\gamma_eB)$ = 4.5 / 1.8 kHz (the same estimate gives 3.3 / 2.1 kHz at 24.1 mT, consistent with the notebook's "about 2.5 kHz") |
| `13a-manifolds.md` | new | $\tilde H_m$ as a 3×3 matrix in the rotating frame, $=\Delta_mS_z+\Omega_{DQ}(\cos\varphi S_x+\sin\varphi S_y)$, $\Delta_m=\Delta_0+mA$; $h_m$; $U=\sum_mD^1(u_m)\otimes\lvert m\rangle\langle m\rvert$; mixed-unitary channel | notebook §2 (cell 7) and §2.1 (cell 9: $\tilde H_m$ in operator form). The 3×3 matrix is written out here: its off-diagonal elements $\frac{\Omega_{DQ}}{\sqrt2}e^{\mp i\varphi}$ were checked against $S_x,S_y$ for spin 1. $\Delta_c$ does not depend on $m$, because the quadrupole and nuclear Zeeman terms are constant within a block (derived here) |
| `13b-nutation.md` | new | $P_0^{(m)}(t)$ closed form; bright/dark ladder $\lvert0\rangle\leftrightarrow\lvert B\rangle\leftrightarrow\lvert D\rangle$; on resonance $\cos^2\Omega_{DQ}t$; off resonance $q\le\Omega_{DQ}^2/\Omega'^2$, maximum at odd multiples $((\Omega_{DQ}^2-\Delta^2)/\Omega'^2)^2$ = 0.69 at 10 MHz; nuclear-average beat | notebook §1.2 (cell 6, bright/dark states, Morris–Shore), §2.1 (cell 9, closed form for $P_0^{(m)}$), cell 11 (0.69), Appendix B (cell 43: lines at $2\Omega_{DQ}$, $2\Omega'$, $\Omega'$; beat at $A^2/\Omega_{DQ}$). Bright/dark states written for $\varphi=0$. The script prints: DQ π at 70.7 ns; outer-manifold $P_0$ = 0.672 exactly at $\Omega_{DQ}t=\pi$ (the 0.69 is the local maximum at $\Omega't=\pi$, 67.6 ns); outer period $2\pi/\Omega'$ = 135.3 ns |
| `14a-field.md` | new | what depends on B at 44 mT (table: splitting; hyperfine error independent of B; rotating-wave error $\Omega_{DQ}^2/6(\gamma_eB)^2=x^2/3$; $A_\perp$ shift; nuclear Zeeman); cross terms stay inside $D^1(u)$ at about 1e-4 over 2–60 mT; excited-state anticrossing caveat | rotating-wave estimate: notebook §2.3 (cell 14, $\varepsilon\approx\frac{16}9A^2/\Omega_{DQ}^2+\Omega_{DQ}^2/6(\gamma_eB)^2$) and Appendix A (cell 38: $x^2/3$, $x=\Omega_{SQ}/2\gamma B$; the shift $\Omega_{DQ}^2/4\gamma B$). Field sweep (Appendix A, cells 40–42, Tables A.1–A.2): a spin-½ model with a linearly polarised drive matches the laboratory frame to 1.1–1.8e-4 over 2–60 mT at the drive of Mamin et al.; the closed-form rotating-wave infidelity follows $x^2/3$ within 0.85–1.40. The 44-mT numbers are evaluated by `fig_dq.py` (`field_estimates`, derived here, no simulation): 1.07e-4 / 2.20e-5 / 5.49e-6 for Mamin / 20 / 10 MHz; best single DQ π pulse at $\Omega_{DQ}/2\pi$ ≈ 93 MHz with 1.9e-3 (the same code gives 69 MHz and 3.5e-3 at 24.1 mT, as in notebook Table 2.3); shift $\Omega_{DQ}^2/4\gamma_eB$ = 10 kHz at 10 MHz. Anticrossing: Jacques et al., PRL 102, 057403 (2009), **not in `references.bib`, cited from memory: TODO** |
| `14-one-pulse.md` | unchanged | | |

## Figures

| file | used on | provenance |
|---|---|---|
| `media/dq-levels.png` | 11 | `fig_dq.py`, `fig_levels()`, 2026-10-23; now labels $2\gamma B$ = 2.46 GHz (44 mT); schematic, not to scale |
| `media/dq-nutation.png` | 13b (new) | `fig_dq.py`, `fig_nutation()`, 2026-10-23; $P_0^{(m)}(t)$ closed form, $\Omega/2\pi$ = 10 MHz ($\Omega_{DQ}/2\pi$ = 7.07 MHz), $A/2\pi$ = 2.16 MHz, t = 0–300 ns, m = 0, ±1 and the average $(P_0^{(0)}+2P_0^{(1)})/3$. This is the closed-form version of notebook cell 10, panel (a) |
| `media/dq-inversion.png` | 14 | unchanged |

## TODO (speaker to confirm)
- `14a-field.md`: 44 mT is close to the excited-state level anticrossing (≈ 51 mT), where optical pumping polarises the ¹⁴N (Jacques et al. 2009). If the nucleus is partly polarised in our set-up, $p_m\ne1/3$, and every Part I and Part III number assumes $p_m=1/3$. The weights should be measured, e.g. from the three hyperfine resonances in pulsed ODMR.

## Disagreements with the SPEC wording
- R2.1 asks to explain "why the outer manifolds never empty $\lvert0\rangle$". The closed form says they **do** empty it ($q=1/2$ is reached, at about the π/2 time). What they never do is return fully to $\lvert0\rangle$ at the DQ π time ($q<1$), so $P_0$ peaks at 0.69 there. The slide and notes say the latter.

## Rendering note (unchanged)
`\dyad` is undefined in this Marp MathJax build; use `\ketbra{m}{m}`.

## NOTES-B: provenance for part II, revision 2 (agent B, 2026-10-23)

Notation follows SPEC Revision 2: $K$ is the number of waits, which is also the degree, and $N=K+1$ is the number of pulses. Everything earlier is in NOTES.md (the merged NOTES-B of revision 1). This file records only what changed.

## Files
| file | content source | figure / clip |
|---|---|---|
| `20-section-qsp.md` | section card; notes in K / N | none |
| `21-pulses-off.md` | archived SPEC §3 "Pulses off"; dq-qsp-prep cell 7 "Chebyshev form". The sequence $R_z(\Delta\tau)^K=R_z(K\Delta\tau)$ is drawn explicitly, with $\mathrm{Re}\,P=T_K$ and $\mathrm{Im}\,P=-\sin\theta\,U_{K-1}$ | clip `media/qsp-pulses-off.mp4` (+ `.png` poster). Scene `PulsesOff`, 22.0 s, 1920×1080, `-q h`, 2026-10-23. It shows $K=3$ boxes merging into $R_z(3\Delta\tau)$, the circle, and Re and Im of $P$ traced as functions of $x$; then $K=5$. |
| `21a-pulses-on.md` (new) | archived SPEC §3 "Two bases / Pulses on". Degree and parity are checked in `_verify()` | clip `media/qsp-pulses-on.mp4`. Scene `PulsesOn`, 23.7 s. It has $K=3$ waits and $N=4$ pulses about $x$; the angles rise one by one to $(\pi/2, 2\pi/3, \pi/3, \pi/4)$, then $\phi_2$ goes down to $\pi/6$ and back. Re and Im of $P$ and $Q$ are plotted as functions of $x$, and every frame comes from the 2×2 products (`seq_entries`). |
| `22-chebyshev.md` | archived SPEC §3 "Chebyshev" | `media/qsp-chebyshev.png` (redrawn large: $T_0\dots T_5$, $U_0\dots U_4$) by `3. wiki/code/lm-261023-figures/fig_qsp.py` `fig_chebyshev`. Colours pair $U_{n-1}$ with $T_n$. |
| `23-sequence.md` | as before, notation $G_N\cdots G_1$, degree $K=N-1$ | `media/qsp-timeline.png` (unchanged) |
| `24-what-is-reachable.md` (replaces `24-wait-and-pulse` + `24a-what-is-reachable`) | archived SPEC §3 "Pauli form", "Reachability". The wait and pulse recursions on $(A,B,C,D)$ were derived for the deck and checked numerically. Theorem: Low, Yoder, Chuang, PRX 6, 041067 (2016); Martyn, Rossi, Tan, Chuang, PRX Quantum 2, 040203 (2021), Thm. 1. Decomposition remark in the notes: Haah, Quantum 3, 190 (2019) | clip `media/qsp-pauli-build.mp4`. Scene `PauliBuild`, 20.3 s. It uses the same angles $(\pi/2, 2\pi/3, \pi/3, \pi/4)$ and $K=3$. $A,B,C,D$ are computed as exact polynomials (`pauli_steps`) after each factor, with a degree counter. Pulses are shown with double arrows A↔B and C↔D. The final frame shows the degrees and parities and the unitarity identity. |
| `26-populations.md` | as before, in K notation (example: $N=4$ pulses, $K=3$) | `media/qsp-populations.png` (unchanged) |

Deleted: `24-wait-and-pulse.md`, `24a-what-is-reachable.md` (merged into 24), and `25-back-to-pulses.md`. No backup version was made; its content is summarised in one sentence in the speaker notes of 24.

## Code and checks
- Clips: `3. wiki/code/lm-261023-animations/scenes_qsp.py`.
  - `_verify()` checks: $R_z^K=R_z(K\Delta\tau)$; Re and Im of $P$ against $T_K$ and $U_{K-1}$ for $K=3,5$; the explicit $T_3,T_5,U_2,U_4$; the vectorised $P,Q$ against 2×2 products for random angles, plus their degree and parity; and the Pauli recursions against $\frac12\mathrm{Tr}$ projections of the 2×2 products, with degrees and unitarity.
  - Render: `../env-manim/.venv/bin/python render.py scenes_qsp.py <Scene> -q h --media-dir out/manim-B --poster 0.9`.
- Figures: `3. wiki/code/lm-261023-figures/fig_qsp.py`. `verify()` passes 13981 checks (seed 42), now including the plotted $U_k$ (from the recurrence) and $T_5$.

## Not from the sources
- The example angles $(\pi/2, 2\pi/3, \pi/3, \pi/4)$ in the two new clips and $(\pi/4,\pi/2,\pi/2,\pi/4)$ on 26 are illustrative choices, not designs from the notebooks.
- Notes on 24 say a real $f$ can be realised "as A, or with a pulse phase shift as B". The A part is from the archived SPEC corollary. The A↔B swap by shifting an end pulse is from BRIEF §5 ("shifting an end pulse by π swaps A and B (up to sign)").

## NOTES-C: provenance for Part III, first half (slides 30 to 36a), revision 2

Agent C, 2026-10-23. Notation (deck-wide): K = number of waits = degree; N = K + 1 pulses.

Content source throughout: `3. wiki/code/dq-pulse-notebook/dq-pulses.ipynb` §3.1 to §3.3 (markdown cells 19 to 28),
as summarised in the archived `3. wiki/code/lm-261023-animations/_removed/manim-slides/SPEC.md` §3. Static figures
come from `3. wiki/code/lm-261023-figures/fig_target.py` and clips from
`3. wiki/code/lm-261023-animations/scenes_target.py`. Both are closed-form, and each has a self-check
(`verify()` / `_verify()`) that runs before drawing.

| slide file | content source | figures / clips |
|---|---|---|
| `30-section-target.md` | none (section card) | none |
| `32-target.md` | dq-pulses §3.1 (cell 20, Table 3.1: windows, transition bands, stopband, ε, E, tolerance 1e-3; spin-1 error 2ε and (8/3)ε(1 − (2/3)ε); why not \|Q − Q_t\|²) | clip `media/target-build.mp4` (+ poster), scene `TargetBuild`: bands, \|Q_t\|, \|P_t\|, requirements, the E line. The single-π-pulse comparison was removed in revision 2 at the human's request. |
| `34-tau-periodicity.md` | dq-pulses §3.3 (cell 26): compatibility rule; cases p = A, 3A, 1.5A. Uses W(Δ + 2π/τ) = −W(Δ) (cell 26 prints 4π/τ, which is wrong; checked in both scripts). | clip `media/tau-sweep.mp4`, scene `TauSweep` (unchanged from revision 1; no K labels) |
| `34a-tau-table.md` | dq-pulses Table 3.3 (cell 27 output, cell 28). The "smallest K" column of the notebook counts pulses and is shown here as N. τ and Kτ re-checked in `fig_target.py verify()`. | none |
| `35-easy-targets.md` | dq-pulses §3.3 (cell 26: R_x(π/2) W(2π/A) R_x(π/2)); Appendix E.2 (one π pulse for the passband alone) | clip `media/resonances-to-x.mp4`, scene `ResonancesToX` (unchanged); figure `media/target-point-q2.png` (`fig_point_q2`; max \|Q\|² in the stopband = cos²(0.3π) = 0.345, checked) |
| `36-target-expansion.md` | Laurent to Chebyshev (dq-qsp-prep §3.2 via archived SPEC §3, Part II); a_k = 4 sin(0.2πk)/(πk): dq-pulses cell 23, re-derived by quadrature in `verify()` for both sign patterns | figure `media/target-step-coeffs.png` (`fig_step_coeffs`) |
| `36a-gibbs.md` | dq-pulses cell 25 (Gibbs; r ≈ 0.92 on the windows; E ≈ 0.2) | clip `media/gibbs-ladder.mp4`, scene `GibbsLadder` (degree K = 1, 3, 5, 7, 9, 13, 17, 25, 41, signs (−,+,−); E of the scaled series shown per K); figure `media/target-step-error.png` (`fig_step_error`, x axis = degree K; θ grid of 20001 points) |

Clips (1080p60, poster at 0.9 of the clip, `render.py scenes_target.py <Scene> -q h --poster 0.9 --media-dir out/manim-C`):

| clip | length | rendered |
|---|---|---|
| target-build | 19.9 s | revision 2 |
| tau-sweep | 24.7 s | revision 1 |
| resonances-to-x | 19.5 s | revision 1 |
| gibbs-ladder | 25.8 s | revision 2 |

Numbers that differ from the earlier SPEC:
- The uniform error of the scaled cut series for N = 12 to 26 pulses (K = 11 to 25) is 0.163 to 0.316, not 0.17
  to 0.28. The notebook's grids give the same values. The slide says only E ≈ 0.2.
- The peak overshoot at K = 41 is 8.3 %. The asymptotic Gibbs value is 8.95 % of the jump, so the slide says
  "about 9 %" and the clip labels it "Gibbs overshoot" without a number.

## NOTES-D — agent D, revision 2 (2026-10-23)

Provenance for the changes of revision 2 only. Revision 1 provenance is already merged into `NOTES.md`. Deck notation since revision 2: **K = number of waits = degree, N = K + 1 pulses**. The notebook (`3. wiki/code/dq-pulse-notebook/dq-pulses.ipynb`) and its json (`dq-pulse-notebook/out/selective_qsp.json`) index by "K" = number of pulses, which is the deck's N. So the notebook's "K = 14" design is "N = 14 (K = 13)", and its E_K is the deck's E_N.

## Slide changes

| file | change | sources |
|---|---|---|
| `37-upper-limit.md` | Conditions explained in a table (condition · meaning); N/K notation. | nb §3.2 cell 23 (the three inequalities, the LP, E_K root); Table 3.2 numbers as before |
| `38-k14-sequence.md` | Notation only (Kτ = 13τ; "N = 14 pulses, K = 13 waits" in the notes). | Table 3.4 |
| `38a-palindromic.md` | Title "Symmetric QSP leaves one real function". New paragraph on where symmetric QSP comes from; derivation kept; degree K. | Wang, Dong, Lin, Quantum 6, 850 (2022): `1. raw/papers/wang2022energy/source/main.tex` lines 244–266 (definition of symmetric QSP; "Ref. [DongMengWhaleyEtAl2021] suggests that the solution can be restricted to be symmetric"; fixed start Φ⁰ = (π/4, 0, …, 0, π/4); many global and local minima), Theorem 1 (existence and uniqueness of symmetric phase factors, l. 289), Theorems 3–4 (maximal solution within π/√3 ‖f‖∞ of Φ⁰ for ‖f‖∞ ≤ 1/2; Hessian between 1/4 and 25/4 if ‖f‖∞ ≤ √3/(20π d̃), l. 363–380). Dong, Meng, Whaley, Lin, Phys. Rev. A 103, 042419 (2021): bibitem in the same main.tex, l. 1904–1909. The wiki literature note `3. wiki/papers/@wang2022energy.md` named in SPEC does not exist, so I read the raw source instead. |
| `39-training.md` | Loss grid count renamed M (N is now pulses); notation N; two sentences moved to 39a. | as before |
| `39a-training-runs.md` (new) | Parametrisation, exact gradient, two stages, starting points, convergence fraction, with a learning-curve figure. | nb §3.4 cells 29, 31 (L-BFGS-B, minimax/SLSQP, dense grids, 12 random starts plus decomposition (arbitrary) or fit (symmetric), 7/12 and 6/12, 34 s vs 5 s, L 0.03–0.7 → 1.64e-4 with largest error 1.8e-3 → 5.09e-4, about 30 iterations from the good starts) |
| `40a-not-retrained.md`, `41-retrained.md` | Notation N. | as before |
| `42-take-home.md` | Rewritten with the human's four points (verbatim intent, SPEC revision 2); equation of the mixed-unitary channel added. | channel from the archived SPEC §3 Part I; numbers from Tables 3.2 and 3.8 |
| `80-references.md` | Added Dong, Meng, Whaley, Lin (2021), with title and journal from the Wang et al. bibitem; renumbered to 10 entries. | |
| `99b`, `99c` | d → K (degree); 99c title "The K = 2 example". | |
| `99e-wide-identity.md` | Table N and Kτ; smallest degree K = 39; figure replaced by `fig_results_wide.png` (the notebook panel's legend said "K" for pulses). | App. E.3 as before |
| `99f-continuation.md` | Table N; figure replaced by `fig_results_continuation.png` (the notebook panel's legend said "K = 14"). | §3.5 as before |
| `99g-degree-scaling.md` | K ≃ c ln(1/ε)/Δθ (degree); E_N. | as before |

## Figures (`3. wiki/code/lm-261023-figures/fig_results.py`, run 2026-10-23)
- Relabelled: `fig_results_trained.png`, `fig_results_retrained.png`, `fig_results_scaling.png` (axes now read "pulses N", legend $E_N$).
- **`fig_results_learning.png` (new):** json `H_runs`, entries with K (pulses) = 14, mode `sym`, the 12 random starts. Thin lines are `hist` (mean error L per L-BFGS-B iteration); thick lines are the running minimum of `mm` (largest error per minimax iteration). Green means E ≤ 1.01 × best, which gives 7 of 12 and matches the notebook's Table 3.5. Dashed line: B.series E for 14 pulses.
- **`fig_results_continuation.png` (new):** json `F_runs`, K = 14, `sym`, Ω = 10 MHz, `ladder` (E at 160 … 10 MHz). It reproduces the notebook text: 5.2e-4 → 1.5e-3 at full amplitude, → 2.0e-2 at fixed length.
- **`fig_results_wide.png` (new):** `dq-pulse-notebook/out/target_vs_detuning.json`, `coeff["-+-"][d]`, with r = Σ c_k cos(kθ) and θ = πΔ/6A (τ = 2π/6A), for N = 17, 39 and 104. Check: on dense grids the passband error matches the stored E exactly (for example 9.55e-4 against 9.42e-4 at N = 104). The stopband error evaluated densely is larger (2.0e-3 at N = 104) than the stored value, which was sampled at 41 points per interval. The slide quotes the stored values.
- Removed from the deck: `dq-pulses-c35-1-a-nt.png` and `dq-pulses-c74-a-nt.png`, whose legends printed "K" for pulses.

## Clips
- **`qsp-best-approx` (A6)** in `3. wiki/code/lm-261023-animations/scenes_results.py`. The lead renamed the file from `scenes_D.py`. I relabelled the counter to "N = …, E_N = …", the axis to "pulses N" and the y label to $E_N$. Re-rendered `-q h --media-dir out/manim-D --poster 0.95`: 22.1 s. `_verify()` passes.
- `real-pulse-bloch` (A7): unchanged, since it has no K or N labels.

## TODO (speaker to confirm)
1. `42-take-home.md`: the wording of the four points follows the human's notes; please confirm.
2. `80-references.md`: the titles of Low–Yoder–Chuang 2016, Low–Chuang 2017 (PRL), Low–Chuang 2017 (arXiv:1707.05391) and Haah 2019 are from memory (not filed). Dong et al. 2021 is taken from the Wang et al. bibliography but is not filed either.
3. `38a`: the slide says Wang, Dong and Lin "proved that every admissible polynomial pair has exactly one symmetric phase list". That is their Theorem 1, which holds in their convention (signal as an x-rotation, real Q) with the sign condition on the leading coefficient of Q for odd degree. The slide leaves out that caveat.
4. `99g`: the gapped sign-function bound and the constants c ≈ 1.2 and 1.6, as before.

## NOTES-E: provenance of slides 41a-41f (agent E, revision 2, 2026-10-23)

Sources: `3. wiki/code/dq-qsp-prep/dq-qsp-prep.ipynb` (executed 2026-10-22 build, outputs stored) and its README; `3. wiki/projects/qsp/dq-qsp-preparation.md`; `3. wiki/code/nv14n-universal-not/nv14n-universal-not.ipynb` (2026-10-22 build) and its README; `3. wiki/projects/qsp/universal-not.md`. Every number below was read in the notebook output (table / markdown cell) unless marked otherwise. Notation map: the notebooks' "K" (pulses) = deck N; notebooks' "d" (free-evolution periods) = deck K.

Figures: `fig_more.py` in `3. wiki/code/lm-261023-figures/` (run 2026-10-23 with `../env/.venv/bin/python fig_more.py`; deterministic, no simulation, no optimisation). No notebook PNG is used in the deck (all were extracted to the scratchpad and inspected; their legends use the notebooks' K for pulses, so own figures were drawn instead). No clips.

## 41a-prep-target.md
- Prepared state, $\varphi=\pi+2\arg(QP^*)$, $F_\varphi$: prep nb §1.2 (cell 2). DQ coherence $-\frac23\sum(P_mQ_m^*)^2$, phase-coherence condition: §2 (cell 3). Target $(1/\sqrt2,-i/\sqrt2)$, every phase from one design ($\varphi\to\varphi+2\phi_0$): §1.3. $\varphi_m\approx2A m_I/\Omega_{DQ}$, infidelity $\eta^2$, population $O(\eta^4)$: §2 and Table 2.1 (cell 4).
- Table: Table 3.3.1 (cell 9), single π/2 pulse (full amplitude): 1−F0 = 3.2e-3 / 1.5e-2 / 6.0e-2, |ρ₊₁,₋₁| = 0.4968 / 0.4846 / 0.4401 at Mamin et al. (Ω_DQ/2π = 31.25 MHz) / Ω/2π = 20 / 10 MHz.

## 41b-prep-constant.md
- Degree-0 target, one instantaneous π/2 pulse exact at every Δ, E = 0, T_min = 0: §3.3 "Hard-pulse limit" (cell 8), §3.4 "Consequence for the preparation" (cell 20), Table 3.4.4 (robust d_min = 0 at every τ). Waits only separate the manifolds, τ = 0 predicted best: §3.4 (cell 15), §3.5 (cell 26).
- Figure `media/more-prep-robust.png` (`fig_more.py: fig_prep`), Ω/2π = 10 MHz (Ω_max/2π = 7.07 MHz, t_p = 141.4 ns). Left: 1−F0 = 1 − 4[Im(QP*)]² as a function of Δ, from 2×2 products of closed-form rectangular-pulse propagators: single π/2 pulse (full amplitude, 35.4 ns) and the stored robust no-wait designs N = 3, 4, 5 (`dq-qsp-prep/out/flat_prep.json`, target "coherent", tau_name "0"). Checks printed by the script reproduce the notebook: single pulse nuclear average 6.0e-2 (Table 3.3.1); worst on windows 4.28e-3 / 8.79e-5 / 5.51e-6 = the json max_error and Table 3.4.2 (cell 16) 4.3e-3 / 8.8e-5 / 5.5e-6. Right: the json's max_error for N = 3–8 without waits and with τ = 2π/3A (= Table 3.4.2 rows "10 MHz, τ = 0" and "10 MHz, τ = 2π/3A").
- Notes: "about 0.09 at the outer resonances" = script output 0.0900 (notebook Table 2.1 gives 8.7e-2 at η = 0.3; η_A = 0.3055). "10 to 400 times smaller": ratios of the two Table 3.4.2 rows (10× at N = 3 to 380× at N = 6).

## 41c-prep-designs.md
- Exact: Table 3.3.1 (733 / 521 / 404.6 ns; composite no waits 424.3 / 212.1 / 96.0 ns; 1−F0 ≤ 1.5e-16); parameter counting and no solution with one pulse fewer: §3.3, Table 3.3.2.
- Robust fixed length: Table 3.4.1 (N = 4, 566 ns; N = 3, 212 ns; with τ = 2π/3A N = 5, 1324 ns; N = 4, 746 ns). Full amplitude: Table 3.5.2 (cell 29) τ = 0: N = 3, 145 ns (10 MHz), 134 ns (20 MHz).
- Laboratory frame: Table 3.5.4 (cell 33): robust τ = 0, 10 MHz 4.1e-5 / 9.5e-5 / 1.5e-4 for δ = 0 / 0.1 / 0.2 MHz; exact 5.3e-5 → 3.0e-2; single pulse 6.0e-2; 20 MHz robust 7.1–7.6e-4; p0 5.6–6.2e-4 (Table 3.5.5); coherence 0.5000.
- Readout $(u^\top u)_{00}=P^2+Q^2$: §1.4.
- Stopband (selective) case: Table 3.4.7 (cell 24) at τ = 2π/6A: N = 46 (K = 45), T = 3472 ns, E = 8.2e-4; E ≈ 0.5 up to N = 6; T_min = 3334 ns (64 steps), Table 3.4.4. Slide rounds to 3.5 µs and 3.3 µs.

## 41d-unot-target.md
- Channel, Choi bound, F = 2/3, entanglement breaking, measure-and-prepare: U-NOT nb §1 (cell 2) (Bužek, Hillery, Werner, PRA 60, R2626 (1999); Horodecki, Shor, Ruskai, Rev. Math. Phys. 15, 629 (2003)).
- Mixture condition ΣR_m = −1 ⇔ three π rotations about orthonormal axes; recovery by applying twice: §2.2 (cell 3).
- $P=-in_z$, $Q=n_y-in_x$, step target, heights √2 apart, "only target that differs": §3.1 (cell 4).
- Closed form π_{0°}–τ–β_{90°}–τ, cos β = 1/3 (70.53°), cube roots of unity at τ = 2π/3A = 154 ns: §3.3 (cell 6); I re-derived the circular average algebraically (agrees). 309 ns: Table 3.5.1.
- No palindromic solution: §3.3. SQ vs DQ qubit and the DQ frame (two DQ π pulses + z rotation): §3.4 (cell 7). Unpolarised nucleus = demonstration of control: §4 (cell 37).

## 41e-unot-approx.md
- Figure `media/more-unot-approx.png` (`fig_more.py: fig_unot`): stored best-approximation Laurent coefficients `nv14n-universal-not/out/unot_polyapprox.json`, part A, SQ, taudiv 3, eq False (final z rotation allowed, Pauli frame on the windows, as in the notebook figure of cell 13), d = 1, 3, 7 (N = 2, 4, 8), evaluated as P = Σ p_k e^{ikΔτ/2}; components n_x = −Im Q, n_y = Re Q, n_z = −Im P; green bars = stored targets P_t, Q_t. Bottom right: lower bound on the process infidelity 1 − (1 − E)² for d = 1–16; the script's printout equals Table 3.4.2 row "SQ qubit, τ = 2π/3A" (cell 11).
- Numbers: 0.34 / 3.9e-2 / 4.3e-4 (N = 2 / 4 / 8): Table 3.4.2 and cell 14; 1.3e-2 at N = 7, 7.4e-6 at N = 13, factor 30: cell 12. Fewer than 8 impossible, 1080 ns: Table 3.4.2. Constructed / refined 8-pulse 5.8e-4 (refined search design) and 8.0e-4 (built from the bound): Table 3.4.3 (cell 15). 886 ns (SQ), 905 ns (DQ): Table 3.5.5 (cell 30). τ trade-off 7 / 1080, 12 / 926, 48 / 889 and aliasing ranges 0.455–0.556, 0.833–1.25: Table 3.4.5 and cell 18. DQ ≥ 9 pulses: cell 12.

## 41f-unot-designs.md
- Exact finite pulses: Table 3.5.1 (cell 22): SQ 863 / 663 / 563 ns (10 / 20 / 40 MHz), N = 4; DQ N = 5, 1324 / 971 ns; with one pulse fewer residuals 1.8e-2 / 3.4e-2 / 8.0e-3 (Table 3.4.1).
- Robust per rotation: Table 3.5.3 (N = 8, 1880 / 1480 ns). Channel robust: Table 3.5.4 (N = 6, 1372 ns; N = 7, 1276 ns); per-rotation designs miss the channel by 1.4–1.5e-2 (cell 27). No waits: Table 3.5.6 (N = 10 and 20, 1000 ns; 13 % above 886 ns; full amplitude 1026 ns at 2π/4A, cell 33).
- Robustness: Table A.2 (cell 41): 0.1 MHz detuning 2.7e-2 / 7.7e-3 / 1.3e-4; T2* = 1 µs 5.1e-2 / 1.1e-2 / 2.7e-4; +2 % amplitude 2.1–7.4e-2 over all designs.
- Full Hamiltonian (nine levels, 24.1 mT): Table 3.5.7 and cell 36: per-input 0.664–0.669, recovery ≥ 0.9987 (smallest 0.99868), ≤ 2.2e-4 leaves the qubit, channel-robust 1.1e-3.
- Selective π pulses 1.6–5.5 µs, 7.8e-3–7.6e-2: Table B.1 (cell 44).
- Title range 0.7–1.9 µs = 663 ns to 1880 ns (the table rows).

## Unsourced / caveats
- Nothing on the slides is unsourced. The laboratory-frame U-NOT check was done at 24.1 mT, not at our 44 mT (the wiki page points to a separate 44 mT study, not read here); the speaker notes say 24.1 mT.
- "Ω/2π" means the single-component Rabi frequency for the DQ drive (Ω = √2 Ω_DQ, as on the single-pulse slide) and the Rabi frequency of the single field for the single-frequency qubit; said in the 41f notes.

---

# Revision 3 (2026-10-23)

## NOTES-A: Part I, revision 3 (agent A, 2026-10-23), new and changed items only

Earlier Part I provenance (revisions 1 and 2) is already in `NOTES.md`. Notebook = `3. wiki/code/dq-pulse-notebook/dq-pulses.ipynb`. Figure script = `3. wiki/code/lm-261023-figures/fig_dq.py`.

## Titles (R3.0)
| file | new title | message moved into the first body sentence |
|---|---|---|
| `01-outline.md` | Outline (unchanged; body as edited by the lead) | none |
| `11-dq-transition.md` | Double-quantum transition | "Two frequency components drive the Δm = 2 transition between \|+1⟩ and \|−1⟩." |
| `12-rotating-frame.md` | Rotating frame | "The rotating frame leaves a differential and a common-mode detuning." The Mamin et al. citation now sits inline next to "rotating-wave approximation", replacing the source line, which overflowed the footer |
| `12a-spin-half.md` | Spin-1 representation | "A balanced drive moves the spin-1 like a spin-½." |
| `12b-dq-pi.md` | DQ π pulse | "The DQ π pulse is P = 0 on the spin-½, and only Q² matters." |
| `13-nitrogen.md` | Hyperfine structure | "The ¹⁴N couples to the electron mainly through A∥SzIz, which is block diagonal in m_I." |
| `13a-manifolds.md` | Hyperfine manifolds | "Each hyperfine manifold is a spin-1 with its own detuning." |
| `13b-nutation.md` | DQ nutation | "The \|0⟩ population oscillates differently in each manifold: one line on resonance, two off resonance." |
| `14-one-pulse.md` | Single-pulse error | "One rectangular DQ π pulse misses the outer hyperfine manifolds." The two equations are merged into one line, and the second bullet now reads "The π/2 pulse fails through its phase" (the phase values remain in the table), so the callout clears the footer |

## Content changes
- `12a-spin-half.md`: the three-row table is replaced by two sentences with equations: (i) the symmetric triplet of two spin-½, $\tilde H=h\otimes\mathbb 1+\mathbb 1\otimes h\Rightarrow e^{-i\tilde Ht}=e^{-iht}\otimes e^{-iht}$; (ii) the same commutation relations, and the classical precession of $\langle\mathbf S\rangle$. Sources unchanged (notebook §1.1, cell 3; standard angular momentum).
- `13b-nutation.md`: adds the spectrum of the \|0⟩ population. Expanding the square gives $P_0^{(m)}=c_m+\frac{2\Delta_m^2\Omega_{DQ}^2}{\Omega_m'^4}\cos\Omega_m't+\frac{\Omega_{DQ}^4}{2\Omega_m'^4}\cos2\Omega_m't$ with $c_m=a^2+b^2/2$ (derived here). At Ω/2π = 10 MHz the lines and their weights in the nuclear average are 14.142 MHz (1/6 = 0.1667, m = 0), 14.787 MHz (0.2789) and 7.394 MHz (0.1041, both m = ±1). These agree with the notebook's two-frequency nutation appendix (Appendix B, cell 43 table: $1/6$ at $2\Omega_{DQ}$, $\frac13(\Omega_{DQ}/\Omega')^4$ at $2\Omega'$, $\frac43A^2\Omega_{DQ}^2/\Omega'^4$ at $\Omega'$; beat at about $A^2/\Omega_{DQ}$; doubling Ω halves the splitting and suppresses the $\Omega'$ line). Bullets shortened.
- `13-nitrogen.md`: speaker notes gain the excited-state level anticrossing caveat, moved from the deleted `14a-field.md`, as a TODO (below).
- `14a-field.md`: **deleted**. B = 44 mT stays on the level diagram (`dq-levels.png`: 2γB = 2.46 GHz, 44 mT) and in the notes of 11 and 13. The table of 44-mT values on 13 is unchanged; `fig_dq.py` (`field_estimates`) still prints them.

## Figures
| file | used on | provenance |
|---|---|---|
| `media/dq-spectrum.png` (new) | 13b | `fig_dq.py`, `fig_spectrum()`, 2026-10-23. Bars: closed-form lines, weights per manifold × 1/3 or 2/3, coloured by manifold (blue m = 0, red m = ±1). Black curve: Hann-windowed Fourier amplitude (×2/Σwindow, zero-padded to 2^18) of the averaged closed-form trace over 0–5 µs at 0.5 ns steps; it reproduces the bar heights. Closed form only, no simulation |
| `media/dq-nutation.png` | 13b | `fig_dq.py`, `fig_nutation()`; now 8 × 4.0 in (was 8 × 4.8) to sit next to the spectrum |

## TODO (speaker to confirm)
- `13-nitrogen.md` notes: 44 mT is close to the excited-state level anticrossing (about 51 mT), where optical pumping polarises the ¹⁴N (Jacques et al., PRL 102, 057403 (2009); **not in `references.bib`, cited from memory**). If the nucleus is partly polarised, $p_m\neq1/3$, and the weights should be measured.

## Rendering note (unchanged)
`\dyad` is undefined in this Marp MathJax build; use `\ketbra{m}{m}`.

## NOTES-B: provenance for part II, revision 3 (agent B, 2026-10-23)

Notation: $K$ waits (the degree), $N=K+1$ pulses. Earlier revisions are already merged into NOTES.md; this file records only the changes of revision 3.

## Titles (R3.0)
Each slide's message moved into its first sentence.

| file | title | first sentence |
|---|---|---|
| `20-section-qsp.md` | Quantum signal processing | (section card) |
| `21-pulses-off.md` | Free evolution | "Without pulses, $K$ waits make one rotation $R_z(K\Delta\tau)$." |
| `21a-pulses-on.md` | Pulse sequence | "Pulses between the waits tune $P$ and $Q$; the degree stays $K$." |
| `22-chebyshev.md` | Chebyshev polynomials | (figure) |
| `23-sequence.md` | Signal and processing | "The waits carry the signal; the pulses process it." |
| `24-what-is-reachable.md` | Reachable polynomials | "$K$ waits and $K+1$ pulses about $x$ make exactly these polynomials:" |

## Clip re-made
- **Clip** `media/qsp-pulses-off.mp4` (+ poster `.png`). Scene `PulsesOff` in `3. wiki/code/lm-261023-animations/scenes_qsp.py`; 19.8 s, 1920×1080, `-q h`, poster at 0.9, 2026-10-23.
- **Contents:** there is no circle panel and no plot in θ. First, $K=3$ blocks $R_z(\Delta\tau)$ merge into $R_z(3\Delta\tau)=\mathrm{diag}(z^{-3},z^3)$, which gives $P=z^{-3}$ and $Q=0$. Then $\mathrm{Re}\,P=T_3(x)$ and $\mathrm{Im}\,P=-\sin\theta\,U_2(x)$ are drawn as functions of $x=\cos\frac{\Delta\tau}{2}$. The same follows for $K=5$.
- **Curves:** the closed forms $\cos K\theta$ and $-\sin K\theta$, evaluated on an $x$ grid.
- **Checks:** `_verify()` checks $R_z^K=R_z(K\Delta\tau)$, $\mathrm{Re}/\mathrm{Im}\,P$ against $T_K$ and $U_{K-1}$, and the explicit $T_3$, $T_5$, $U_2$, $U_4$.

## Deleted
- **Slide** `26-populations.md`, the populations figure `media/qsp-populations.png` (and `out/` copy), and `fig_populations` with its example checks in `3. wiki/code/lm-261023-figures/fig_qsp.py`. `verify()` now passes 13580 checks, seed 42.

## Unchanged since revision 2
- `21a` (clip `qsp-pulses-on`, 23.7 s), `22` (`qsp-chebyshev.png`), `23` (`qsp-timeline.png`, now shown at w:800), and `24` (clip `qsp-pauli-build`, 20.3 s; citations Low, Yoder, Chuang 2016; Martyn et al. 2021, Thm. 1).
- Their sources and checks are as recorded in NOTES.md (revision 2 entry).

## NOTES-C: provenance for Part III, first half (slides 30 to 36), revision 3

Agent C, 2026-10-23. Notation: K = number of waits = degree; N = K + 1 pulses. Titles follow R3.0 (noun
phrases); each slide's message is now its first body sentence.

Content source throughout: `3. wiki/code/dq-pulse-notebook/dq-pulses.ipynb` §3.1 to §3.3 (markdown cells 19 to 28),
as summarised in the archived `3. wiki/code/lm-261023-animations/_removed/manim-slides/SPEC.md` §3. Static figures
come from `3. wiki/code/lm-261023-figures/fig_target.py` and clips from
`3. wiki/code/lm-261023-animations/scenes_target.py`. Both are closed-form, and each has a self-check
(`verify()` / `_verify()`) that runs before drawing.

| slide file | title | content source | figures / clips |
|---|---|---|---|
| `30-section-target.md` | QSP for the DQ π pulse (section) | none | none |
| `32-target.md` | Target function | dq-pulses §3.1 (cell 20, Table 3.1: passband/windows, transition bands, stopband, ε, E, tolerance 1e-3; spin-1 error 2ε and (8/3)ε(1 − (2/3)ε); why not \|Q − Q_t\|²) | clip `media/target-build.mp4` (19.9 s, revision 2), scene `TargetBuild` |
| `34-tau-periodicity.md` | Periodicity | dq-pulses §3.3 (cell 26): compatibility rule; p = A, 3A, 1.5A. Uses W(Δ + 2π/τ) = −W(Δ) (the notebook's 4π/τ is wrong; checked in both scripts) | clip `media/tau-sweep.mp4` (24.7 s, revision 1), scene `TauSweep` |
| `34a-tau-table.md` | Choice of τ | dq-pulses Table 3.3 (cell 27 output, cell 28); its pulse counts are shown as N. τ and Kτ re-checked in `fig_target.py verify()` | none |
| `35-easy-targets.md` | Simple targets | dq-pulses §3.3 (cell 26: R_x(π/2) W(2π/A) R_x(π/2)); Appendix E.2 (one π pulse for the passband alone) | clip `media/resonances-to-x.mp4` (19.5 s, revision 1), scene `ResonancesToX`; figure `media/target-point-q2.png` (`fig_point_q2`; max \|Q\|² in the stopband = 0.345, checked) |
| `36-target-expansion.md` | Chebyshev expansion | Laurent to Chebyshev (dq-qsp-prep §3.2 via archived SPEC §3, Part II); a_k = 4 sin(0.2πk)/(πk): dq-pulses cell 23, re-derived by quadrature in `verify()`. Notes: one paragraph on why cutting the series fails (Gibbs overshoot about 9 %, E ≈ 0.2 after scaling; dq-pulses cell 25) | figure `media/target-step-coeffs.png` (`fig_step_coeffs`) |

All clips are 1080p60 with the poster at 0.9 of the clip, rendered with
`render.py scenes_target.py <Scene> -q h --poster 0.9 --media-dir out/manim-C`.

Removed in revision 3: `36a-gibbs.md`, together with `media/gibbs-ladder.mp4/.png` and `media/target-step-error.png`.
The `GibbsLadder` scene and `fig_step_error` stay in the scripts but are no longer rendered.

Numbers that differ from the earlier SPEC (now only in the notes of 36):
- The uniform error of the scaled cut series for N = 12 to 26 is 0.163 to 0.316, not 0.17 to 0.28. The notes say
  only E ≈ 0.2.

## NOTES-D: agent D, revision 3 (2026-10-23)

This note covers only the changes of revision 3. The provenance of revisions 1 and 2 is in `NOTES.md`, or in the revision-2 NOTES-D if it has not been merged yet. No numbers, figures or clips changed in this revision.

## Changes

| file | new title (R3.0) | message moved into the first body sentence | other |
|---|---|---|---|
| `37-upper-limit.md` | Best uniform approximation | "The best polynomial gives the best achievable error $E_N$." | |
| `38-k14-sequence.md` | Pulse sequence ($N=14$) | "The best polynomial gives fourteen pulses that invert at every $\Delta=nA$:" | The chain $r\to(P,Q)\to\vartheta_1\dots\vartheta_{14}$ is now inline. Figures are slightly smaller and the last bullet shorter, so the slide stays clear of the footer. |
| `38a-palindromic.md` | Symmetric QSP | "A palindromic sequence leaves one real function." | |
| `39-training.md` | Loss function | "Training on the mean error, then on the largest error, reaches the best achievable error." | Takes over from the deleted 39a. The learning-curve figure `fig_results_learning.png` replaces `fig_results_trained.png` on the slide. The parameters, gradient, two stages, starting points and the 7-of-12 outcome are now in the bullets and speaker notes. |
| `39a-training-runs.md` | deleted | | |
| `40-real-pulse.md` | Finite pulses | "A real pulse depends on the detuning: its axis tilts out of the equator." | |
| `40a-not-retrained.md` | Finite pulses, not retrained | "Played with real pulses, the $N=14$ design loses its accuracy." | |
| `41-retrained.md` | Retraining | "Retrained at full amplitude, real pulses cost 0 to 2 extra pulses." | |
| `42-take-home.md`, `80-references.md`, `90-thanks.md`, `99-backup.md` | Take-home, References, (closing), Backup | unchanged | |
| `99b-pauli-form.md` | Pauli form | "The Pauli form of a $\pm x$ sequence holds in every basis." | |
| `99c-d2-both-bases.md` | Example ($K=2$) | "Multiplying out $u=W\,R_x(\phi)\,W$ in both bases checks unitarity." | |
| `99d-earlier-designs.md` | Exact designs | "Earlier designs are exact at the three resonances only." | |
| `99e-wide-identity.md` | Wide stopband | "A stopband over $\lvert\Delta\rvert\le3A$ needs 104 pulses for $E=10^{-3}$." | |
| `99f-continuation.md` | Continuation in the Rabi frequency | "The error grows smoothly as the drive weakens." | The figure is smaller, so the table stays clear of the footer. |
| `99g-degree-scaling.md` | Degree scaling | "A step across a gap needs degree $K\propto\ln(1/\varepsilon)/\Delta\theta$." | |

`fig_results_trained.png` is no longer used on any slide. It stays in `media/` and in `fig_results.py`, so the lead can remove it if wanted.

## TODO (speaker to confirm), carried over
1. Take-home wording.
2. Titles of the unfiled references: Low–Yoder–Chuang 2016, Low–Chuang 2017 (PRL and arXiv:1707.05391), Haah 2019, Dong–Meng–Whaley–Lin 2021.
3. On 38a, the Theorem-1 caveats (stated in Wang et al.'s own convention; sign of the leading coefficient for odd degree).
4. On 99g, the gapped-step bound and the constants c ≈ 1.2 and 1.6.

## NOTES-E: provenance of slides 41a-41f (agent E, revision 3, 2026-10-23)

Slides: `41a-prep-response.md` (State preparation, clip) · `41b-prep-expansion.md` (State preparation: Chebyshev expansion) · `41c-prep-sequence.md` (State preparation: pulse sequence) · `41d-unot-response.md` (Universal NOT, clip) · `41e-unot-expansion.md` (Universal NOT: Laurent expansion) · `41f-unot-sequence.md` (Universal NOT: pulse sequence). The revision-2 files 41a-prep-target … 41f-unot-designs and their figures `more-prep-robust.png`, `more-unot-approx.png` were deleted.

Sources: `3. wiki/code/dq-qsp-prep/dq-qsp-prep.ipynb` (prep nb) with README and `3. wiki/projects/qsp/dq-qsp-preparation.md`; `3. wiki/code/nv14n-universal-not/nv14n-universal-not.ipynb` (U-NOT nb) with README and `3. wiki/projects/qsp/universal-not.md`. Notation: the notebooks' and jsons' "K" (pulses) = deck N; their "d" (free-evolution periods) = deck K.

## Clips (`3. wiki/code/lm-261023-animations/scenes_more.py`, manim 0.21, rendered 2026-10-23, `-q h --poster 0.9`, `--media-dir out/manim-E-h`)
| scene | clip | length | slide |
|---|---|---|---|
| `PrepResponse` | `media/prep-response.mp4` + `.png` | 21.8 s | 41a |
| `UnotResponse` | `media/unot-response.mp4` + `.png` | 18.1 s | 41d |

- Curves are computed at render time from stored sequences with closed-form rectangular-pulse 2×2 propagators (same formula as `unot.useq` / `dqqsp`); no optimisation. `_verify()` asserts the re-evaluated errors against the stored ones.
- PrepResponse: target (1/√2, −i/√2) on windows ±0.1A; one rectangular π/2 pulse at full amplitude (Ω_DQ/2π = 7.07 MHz, 35.4 ns); robust design `dq-qsp-prep/out/flat_prep.json`, target "coherent", Ω = 10 MHz, tau_name "0", K = 4 (deck N = 4), fixed length 141.4 ns, 565.7 ns. Shown with the global phase of (P, Q) chosen so that P is real (the prepared state does not depend on it). Checks: robust worst 1−F0 on the windows 8.79e-5 = json max_error (prep nb Table 3.4.2: 8.8e-5); single pulse 1−F0 = 0.0900 at ±A, nuclear average 6.0e-2 (Table 3.3.1).
- UnotResponse: `nv14n-universal-not/out/unot_designs.json` "flat", SQ, 20 MHz, K = 8 (deck N = 8), taudiv 3; frame from the stored `frame_rotvec` (scipy Rotation), target sign per window chosen as the design's (u and −u are the same operation). Check: worst process infidelity on the windows 6.85e-4 = json max_infid (U-NOT nb Table 3.5.3: 6.9e-4); duration 1480.2 ns (Table 3.5.5: 1480 ns).

## Figures (`3. wiki/code/lm-261023-figures/fig_more.py`, run 2026-10-23; `verify()` prints the checks)
- `more-prep-expansion.png` (41b): `dq-qsp-prep/out/target_vs_detuning_prep.json`, τ = 2π/6A = 77.2 ns. (a) |P| = √((1+r_z)/2) from the stored cosine coefficients `rz_cos` for d = 6, 11, 22, 45 (N = 7, 12, 23, 46); re-evaluated E (window 1−r_y², stopband 1−r_z², stopband edge +0.01A) 0.377 / 0.238 / 0.0275 / 8.29e-4 against stored 0.377 / 0.238 / 0.0274 / 8.18e-4 (the residual difference is the solver's sampling of the band edge). (b) target Chebyshev coefficients from the closed form of prep nb §3.4 (cell 23): f_0 = 1 − 3ω'/π, f_k = −sin(kω')(1 + 2cos kAτ)/(πk), ω' = 0.2Aτ, c_k = 2f_k; best-approximation `rz_cos` for N = 12, 46. (c) stored `state[].E` and `pal[].E_pal` as functions of N = d + 1.
- `more-prep-sequence.png` (41c): exact design angles 1.2897π, 1.4120π, 1.7983π (all +x), 141.4 ns pulses, waits 154.3 ns, 732.9 ns: prep nb Table 3.5.3 (cell 31). Robust fixed length: flat_prep.json (angles 0.3354π x, 1.7486π −x, 1.0709π x, 1.1571π −x; 565.7 ns; = Table 3.5.3). Robust full amplitude: `design_space_prep.json` finite / R-coh / 10 MHz / conv b / τ 0: areas 0.3259π, 0.7741π (−x), 0.9480π, 144.8 ns, worst 1−F0 2.77e-4 (re-evaluated = stored; Table 3.5.2: K = 3, 145 ns).
- `more-unot-expansion.png` (41e): (a) `unot_polyapprox.json` part A, SQ, taudiv 3, eq False, d = 1, 3, 7 (N = 2, 4, 8), n_x = −Im Q, frame x, y, z on the windows −A, 0, +A (as in the U-NOT nb figure of cell 13). (b) target coefficient norms `unot_target_expansion.json` cases[0] `coeff_norm` (0.5774, 0.4775 = 3/2π, 0.2387, 0, …; U-NOT nb Appendix C, cell 46); best approximation K = 8 norms recomputed from unot_polyapprox d = 8: 0.738, 0.434, 0.175, 0.089, 0.023, identical to the stored `coeff_norm_minimax` (cell 48: constant term 0.74 against 0.58). (c) cut series `gibbs[].infid_scaled` (K = 2n ≤ 40), smoothed series `smooth[].infid` (K = 2n), best approximation 1 − (1 − E)² from unot_polyapprox (= Table 3.4.2 row SQ, τ = 2π/3A).
- `more-unot-sequence.png` (41f): SQ, 20 MHz, pulses 50 ns, waits 154.3 ns. Exact N = 4: `unot_designs.json` point SQ 20 MHz K = 4: 1.1262π (0°), 0.4844π (100.2°), 1.8506π (249.7°), 0.6986π (259.2°), 663.0 ns (= U-NOT nb Table 3.5.2). Channel robust N = 7: `unot_cflat.json` SQ 20 MHz K = 7, max_deficit 2.04e-4, 1275.9 ns (Table 3.5.4: K = 7, 1276 ns). Robust per rotation N = 8: unot_designs.json flat SQ 20 MHz K = 8, 1480.2 ns.

## Numbers in the slide text and speaker notes
- 41a: prepared state, φ = π + 2 arg(QP*), coherence −(2/3)Σ(P_mQ_m*)²: prep nb §1.2, §2. Every phase from one design: §1.3. Single pulse 6.0e-2 / 1.5e-2 / 3.2e-3, coherence 0.44 (0.4401): Table 3.3.1. φ_m ≈ 2Δ/Ω_DQ: §2, Table 2.1. Robust 566 / 212 ns: Table 3.4.1. Laboratory frame 4.1e-5 to 1.5e-4, coherence 0.5000, ~7e-4 at 20 MHz: Tables 3.5.4–3.5.5. Readout P² + Q² = 0: §1.4.
- 41b: degree-0 target, E = 0, T_min = 0: §3.3, §3.4 (cell 20). 1324 ns with waits: Table 3.4.1. 145 ns: Table 3.5.2. Stopband case: N = 46, 3.47 µs, 8.2e-4, ≈ 0.5 up to N = 6, symmetric 49, T_min 3.33 µs: Tables 3.4.4, 3.4.7, cell 25.
- 41c: three conditions, no solution with one pulse fewer: §3.3, Table 3.3.2. 521 / 405 ns: Table 3.3.1. 134 ns: Table 3.5.2.
- 41d: channel, F = 2/3, Bužek–Hillery–Werner PRA 60, R2626 (1999): U-NOT nb §1. ΣR_m = −1 ⇔ orthonormal π rotations: §2.2. P = −i n_z, Q = n_y − i n_x, √2 apart: §3.1. Cube roots of unity, closed form cos β = 1/3, no palindromic solution: §3.3. Entanglement breaking, recovery by repetition, control not coherence: §1, §2.2, §4.
- 41e: 0.34 / 3.9e-2 / 1.3e-2 / 4.3e-4 (K = 1, 3, 6, 7), factor 30: Table 3.4.2, cells 12, 14. 5.8e-4: Table 3.4.3. Smoothed series K = 22 (n = 11), cut series 0.05–0.33 for K ≤ 40: Table C.1 and the stored gibbs list. 886 / 905 ns: Table 3.5.5. 12 / 926, 48 / 889: Table 3.4.5. DQ ≥ 9 pulses: cell 12.
- 41f: 863 / 663 / 563 ns, residuals with one pulse fewer 8.0e-3–3.4e-2: Tables 3.5.1, 3.4.1. Per-rotation designs miss the channel by 1.4–1.5e-2: cell 27. 0.1 MHz detuning 2.7e-2 / 1.3e-4, T2* = 1 µs 5.1e-2 / 2.7e-4, amplitude 2.1–7.4e-2: Table A.2. 1000 ns composite, 13 %: Table 3.5.6, cell 33. DQ 1324 / 971 ns: Table 3.5.1. Full Hamiltonian at 24.1 mT: 0.664–0.669, ≥ 0.9987, 1.1e-3: Table 3.5.7, cell 36. Selective pulses 1.6–5.5 µs, 7.8e-3–7.6e-2: Table B.1.

## Caveats
- Nothing on the slides is unsourced.
- The universal-NOT full-Hamiltonian check was done at 24.1 mT, not at our 44 mT.
- The universal NOT uses the single-frequency qubit {|0⟩, |−1⟩}; the DQ qubit appears only as a comparison in the notes.
- The prep clip shows (P, Q) with P made real by a global phase; said on the clip and in the notes.


# Chapter 3 rebuild (2026-10-26)

Request: update Chapter 3 (from PDF page 22) with the redone notebooks `qsp-unot-sq/`, `qsp-dq-preparation/`, `qsp-dq-pi/`; order universal NOT → DQ preparation → DQ π pulse; draft with animation slots first, then (approved) the clips. Contract: `BRIEF-ch3.md`. Section provenance (slide → sources, figure → script and data origin, TODOs): `NOTES-ch3-unot.md`, `NOTES-ch3-prep.md`, `NOTES-ch3-pi.md`.

- **Archived, not deleted:** old Chapter 3 (32–43a, 48-take-home) and superseded backups 99i–99o → `_archive/2026-10-26-ch3-v1/` (with the pre-rebuild PDF); 23 stale 261023 copies that were not in the 18:16 build → `_archive/stale-261023-copies/`.
- **Lead's slides:** 30 (subtitle), 31 "Design steps" (five steps; gate or state fidelity; $\Omega_{\rm DQ}$), 60 take-home (three columns; numbers from the notebooks' README tables), 01 outline (three applications), 00 title subtitle, 80 references (+ Dong, Lin, Ni, Wang, SIAM J. Sci. Comput. 46, A2951 (2024), arXiv:2307.12468, verified by web search 2026-10-26).
- **Figures:** `3. wiki/code/lm-261027-figures/fig_{unot,prep,pi}.py` (designs from the notebooks' `designs.json`; Newton histories, fixed-τ scans, isolated solutions and the K = 1, 2, 4 ladder from the seed-42 rerun `data/*.pkl`, see that README).
- **Clips** (`3. wiki/code/lm-261027-animations/scenes_ch3_*.py`, manim 0.21, `_verify()` in each file; built and independently reviewed 2026-10-26):

| clip | slide | content | numbers not printed by a notebook |
|---|---|---|---|
| `ch3-unot-channel` | 32 | three Bloch spheres, $R_x(\pi),R_y(\pi),R_z(\pi)$, average $-\mathbf r/3$, $F=2/3$ | none (closed form) |
| `ch3-unot-tau-sweep` | 35 | lines on $e^{i\Delta_m\tau}$, distance from overlap, stops at 105.9 / 154.3 / 231.5 / 463.0 ns | $A\tau=0.457\pi$ (computed) |
| `ch3-unot-k-ladder` | 36 | best response K = 1→4 with band $1-F$ | K = 1, 2, 4 polished from the notebook's printed angles; **poster re-extracted at 15.6 s (K = 3)** with ffmpeg, a re-render with `render.py` would reset it to the K = 4 end frame (use `--poster 0.684`) |
| `ch3-unot-newton` | 38 | target bars, one Newton run (seed 3, same start distribution as the notebook) converging to the designs.json hard design | cost trajectory of that run (not one of the notebook's 300) |
| `ch3-prep-bloch` | 41 | axes $\mathbf n_m=R(u_m)\hat z$: one finite π/2 pulse at 10 MHz (±17.4°, 35 ns) then the 10 MHz arbitrary design (146 ns) | ±17.4°, 35 ns (computed) |
| `ch3-pi-mirror` | 52 | spin-½ picture from $\lvert{+1}\rangle$: preparation, mirror appended, 0.843π pulses fuse into 1.687π; one plain π pulse for contrast | none |
| `ch3-pi-decoupling` | 55 | train t, π, 2t, π, … (t = 0.4 µs) on $(\lvert{+1}\rangle+\lvert{-1}\rangle)/\sqrt2$; phasors of three lines × five ¹³C samples; coherence 1.000 (design) against 0.450 (plain) after six pulses | all train numbers (five-point quantile ¹³C sample); plain-pulse loss is a coherent CP-type error that depends on the DQ phase (≥ 0.96 for $(\lvert{+1}\rangle-\lvert{-1}\rangle)/\sqrt2$), stated in the slide notes |

- **Removed slots** (not in the approved clip list): 42 (mirror images of $\mathbf n_{\pm1}$), 45 (sequence trajectories; covered by the clip on 41), 54 (Bloch trajectories; covered by the clip on 52).
- **Fixed in passing:** `media/dq-spectrum.png` (slide 13b, Chapter 1) was missing from this deck's media; copied from `261023/media/`. `preview.sh` had lost its exec bit; restored. `make` must be run with stdin closed (`make < /dev/null`) from a non-interactive shell, otherwise marp waits on stdin.
- **Open (speaker to confirm):** ¹⁴N polarisation at 440 G (equal weights assumed); microwave rise time against the 1.5 ns pulse (UNOT, 20 MHz) and the 0.2 ns outer pulses (prep, 20 MHz palindromic); light shifts ≈ 40 kHz [estimate in the notebooks]; "Newton much faster than gradient training" is the human's statement, not timed here.
