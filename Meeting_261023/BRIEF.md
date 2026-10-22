# BRIEF — 261023 deck

## Amendments agreed with the human (2026-10-22, binding where they differ from the prompt below)

1. **Format (revised 2026-10-22, evening).** First built as a full manim-slides presentation; the human then wrote: "I don't think full-animation seems not good idea. Lets return to marp presenation, but add manim animation if necessary. First write draft. You can remove current version." The deck is now **Marp** (serif theme, this folder, `make pdf < /dev/null`), with Manim clips only where needed, proposed to the human after the draft. The manim-slides sources are archived in `3. wiki/code/lm-261023-animations/_removed/manim-slides/` (their `SPEC.md` §3 is still the verified content); its build outputs were deleted. Draft contract: `SPEC.md` in this folder.
2. **Figures.** Notebook PNGs from `dq-pulses.ipynb` (extracted 2026-10-22, `media/dq-pulses-cNN*.png`) and closed-form figures from `3. wiki/code/lm-261023-figures/` (serif-theme colours). The "dark background" choice applied to the manim-slides version.
3. **Notebook state.** The deck waited for the 2026-10-22 rework of `dq-pulses` §3 (selective step target, τ chosen from the periodicity, trained QSP against the polynomial bound, finite pulses; τ = 0 removed). Part III is built on the reworked §3; the windows-only and |Δ| ≤ 3A designs of the prompt's M6/M7 are now Appendix E and appear in backup only. Several numbers of §6 below no longer exist in the notebook (robust designs without free evolution: 990 / 354 / 491 / 255 ns; $n_{\rm eff}$ = 7.4 / 2.6).
4. **Structure set by the human** ("This presentation is for introduction to QSP and applying it into Double Quantum"):
   - I. Double quantum with unpolarised ¹⁴N.
   - II. QSP, conceptually: (1) the $R_zR_zR_z\cdots$ example, $R_z(d\Delta\tau)$, giving $P,Q$ as polynomials in $\cos\frac{\Delta\tau}2$; (2) with $R_xR_zR_xR_z\cdots$ we can design $P,Q$ up to degree $d$ and $d-1$ (written as "K, K−1" by the human; the deck uses $K$ pulses and $d=K-1$ waits).
   - III. How QSP does the DQ π pulse.
5. **Meeting date:** 2026-10-23 (`261023`).
6. The take-home of §3 below ("With real pulses, the problem is the pulse itself") no longer fits the reworked results; a revised draft is in the archived `SPEC.md` §5, marked for the speaker to confirm.

---

## The original prompt (as sent, 2026-10-22)

# Prompt — Marp deck: QSP as polynomial design (Laurent and Chebyshev pictures, setting the target, the DQ π pulse)

<!-- FILL IN before sending (defaults in brackets):
     DATE        = [YYMMDD of the meeting]            → deck folder 3. wiki/labmeeting/<DATE>/
     LENGTH      = [~30 min, 22–28 main slides + backup]
     AUDIENCE    = [Paulee group: physicists and NV experimentalists; most saw the 260923 deck]
     THEME       = [serif]
     LANGUAGE    = [English]
     CODE        = [pre-approved: local Python only to (a) extract the PNGs already embedded in the executed notebooks
                    and (b) plot closed-form didactic figures; no simulations, no notebook rebuilds, no Colab]
-->

You are working in the Aleph research vault (`/home/hun/Aleph`). Read `CLAUDE.md` first and obey it: write only inside `3. wiki/`, never edit `1. raw/` or `2. notes/`, and finish with the `_index_wiki.md` and `log.md` bookkeeping. Running code needs the human's permission; the CODE line above is the permission for this task, and nothing beyond it.

## 1. Goal

Build a Marp lab-meeting deck that teaches **quantum signal processing (QSP) as polynomial design**, using the vault's own double-quantum (DQ) NV–¹⁴N problem as the running example. Use the existing kit in `3. wiki/labmeeting/`: run `./new.sh <DATE>`, then follow `3. wiki/labmeeting/README.md`. After the outline is approved, copy this prompt into the deck folder as `BRIEF.md`.

The talk answers three questions:
1. Why is a sequence of pulses and equal waits a polynomial in the detuning, and why does the same unitary look like a **Laurent** polynomial in one basis and a **Chebyshev** polynomial in another?
2. How do you **set the target function** for a physical operation? Use the phase-coherent DQ π pulse on three hyperfine manifolds.
3. What changes with **real (finite) pulses**?

## 2. Process (message-driven; stop after each gate)

1. **Read** the sources in §5–§6.
2. **Outline gate.** Propose an outline organised around the main messages of §3 (one section per message; derivations go to backup). Send it to the human as a slide list (file name · title · one-line content · figure) and **wait for approval**. Do not write slides before that.
3. **Write** the slides, one file per slide (`NN-name.md`, each with its own `marp: true` frontmatter).
4. **Figures** (§7).
5. **Build** with `make pdf < /dev/null`. Render every page to PNG and look at it: check overflow, MathJax errors and table breakage. Fix and rebuild until every page is clean.
6. **Bookkeeping.** Add a `labmeeting/` entry to `3. wiki/_index_wiki.md`, in the style of the 260923 and 260929 entries. Append `## [YYYY-MM-DD] maintain | deck <DATE> — …` to `3. wiki/log.md`. Mark anything you could not confirm with a `<!-- TODO (speaker to confirm): … -->` comment on that slide, and list these items in your final report.

## 3. Main messages (the spine; keep this order)

- **M1. The sequence turns the detuning into a polynomial.** $u=G_K\,W\cdots W\,G_1$ with $K$ instantaneous pulses and $d=K-1$ equal waits, $W=R_z(\Delta\tau)=e^{-i\Delta\tau\sigma_z/2}$. The waits are the signal; the pulse areas (and phases) are the processing. The degree equals the number of waits.
- **M2. One unitary, two bases, two pictures.**
  - In the Z basis the waits are diagonal: every path is one monomial, giving a Laurent polynomial in $z=e^{i\theta}$, $\theta=\Delta\tau/2$.
  - In the eigenbasis $\{\lvert\pm\rangle\}$ of the pulse axis (pulses about $\pm x$ only), the pulses are diagonal and the waits flip the state. Counting flips gives Chebyshev polynomials in $x=\cos\theta$.
- **M3. Chebyshev = the cosine and sine halves of a Laurent polynomial.** $z^{\pm n}=T_n(x)\pm i\sin\theta\,U_{n-1}(x)$; both forms have the same $d+1$ coefficients. Amplitudes are polynomials in $\cos(\Delta\tau/2)$, and populations are polynomials of degree $d$ in $\cos\Delta\tau$.
- **M4. What is reachable, and how to get the pulses back.** The QSP theorem (degree, parity, unitarity), the parameter count, completion (Fejér–Riesz) and layer stripping. Keep this short and point to the 260923 deck for the recursion.
- **M5. Setting a target = five choices.**
  1. Operation → condition on the spin-½ $u$.
  2. Target set, tolerance and don't-care regions.
  3. Symmetry → one real function.
  4. τ places that function on the circle.
  5. Interpolate (point case) or best uniform approximation (robust case).
- **M6. The DQ π pulse as the worked example.** $P=0$ with a common $Q^2$. Palindromic ±x pulses reduce the target to one real Chebyshev series $B(x)=iQ$. Formulation R is trivial with instantaneous pulses; the $d=1$ Ramsey design follows from the target alone; formulation S is a genuine filter whose free-evolution time is set by the transition-band width.
- **M7. Real pulses.** The exact polynomial structure is lost (Euler form, band-limited response with $n_{\rm eff}=LT/2$). The robust DQ π pulses are shortest without free evolution. All the work goes into cancelling the pulse's own detuning dependence.
- **Take-home (one callout):** *With instantaneous pulses, QSP turns "what should happen at which detuning" into a polynomial-approximation problem in one real function. Choosing τ and the symmetry decides how hard that problem is. With real pulses, the problem is the pulse itself.*
- **Optional closing slide.** The same five-step template across the series (preparation, DQ π pulse, decoupling, single-frequency, sequential-frequency, universal NOT), as a table of targets (§6).

## 4. Notation contract (binding — this is where confusion happens)

- **Variables.** Follow the notation table of `dq-qsp-prep` §3.2: $K$ pulses, $d=K-1$ waits, $\theta=\Delta\tau/2$, $z=e^{i\theta}$, $x=\cos\theta$, detuning $\Delta$, rotations $R_a(\phi)=e^{-i\phi\sigma_a/2}$.
- **Three different conventions exist in the vault; convert everything to the one above:**
  - the `dq-pulses` §3.2 step-target coefficients use $w=e^{i\Delta\tau}$ (so $w^k=z^{2k}$ and $\cos k\Delta\tau=T_{2k}(x)$);
  - `projects/qsp/conditional-gate` §4.3 uses $w=e^{i\delta\tau/2}$ and $z=w^2$;
  - the 260923 deck uses $z=e^{i\delta\tau}$ with half powers and $\hat P=z^{-d/2}P(z)$.
  
  Put one "notation bridge" slide in the backup.
- **$(P,Q)$ on the slides always means the Z-basis Cayley–Klein pair of the notebooks**, $u=\begin{pmatrix}P&-Q^*\\Q&P^*\end{pmatrix}$ ($P$ = stay, $Q$ = flip; $\lvert\uparrow\rangle\leftrightarrow\lvert{+1}\rangle$ in the DQ spin-1 picture).
- **The textbook pair in the $\pm$ basis gets distinct symbols, $(P_\pm,Q_\pm)$.** Never write a bare $P$ for it.
- Use standard terminology, following the table in `3. wiki/code/dq-pulse-notebook/README.md`: hyperfine resonance (not "line"), tolerance window, exact design / robust design, palindromic sequence, DQ π pulse, phase-coherent, free evolution, and so on. No coined labels.

## 5. Source A — the derivations (verified; the authority for M1–M6)

All of the following holds for instantaneous pulses about $\pm x$ unless a line says otherwise.

**Building blocks in the two bases** ($c_\phi=\cos\frac\phi2$, $s_\phi=\sin\frac\phi2$):
- Z basis: $W=\mathrm{diag}(z^{-1},z)$ and $R_x(\phi)=\begin{pmatrix}c_\phi&-is_\phi\\-is_\phi&c_\phi\end{pmatrix}$. The waits add phases and the pulses flip.
- $\pm$ basis: $R_x(\phi)=\mathrm{diag}(e^{-i\phi/2},e^{i\phi/2})$ and $W=\begin{pmatrix}\cos\theta&-i\sin\theta\\-i\sin\theta&\cos\theta\end{pmatrix}$. The pulses add phases and the waits flip.
- Operators never change, only their matrices: $X=\lvert+\rangle\langle+\rvert-\lvert-\rangle\langle-\rvert$ is diagonal and $Z=\lvert+\rangle\langle-\rvert+\lvert-\rangle\langle+\rvert$ is off-diagonal in the $\pm$ basis.

**Path counting.**
- Z basis: a path that spends $n_\downarrow$ of the $d$ waits in $\lvert\downarrow\rangle$ carries $z^{2n_\downarrow-d}$. Hence $P,Q=\sum_{k=-d,-d+2,\dots,d}(\cdot)\,z^k$, a Laurent polynomial whose harmonics have the parity of $d$.
- $\pm$ basis: a path with $f$ flips carries $\cos^{d-f}\theta\,(-i\sin\theta)^f$. For even $f$, $\sin^f\theta=(1-x^2)^{f/2}$, so $P_\pm$ is a polynomial of degree ≤ d with the parity of d. For odd $f$, $\langle-\rvert u\lvert+\rangle=-i\sin\theta\,Q_\pm$ with $Q_\pm$ of degree ≤ d−1 and the parity of d−1.

**Pauli form (basis-free):**
$$u=A(x)\,\mathbb 1-iB(x)\,X-i\sin\theta\,[C(x)Z+D(x)Y]$$
- $A,B$ are real, of degree ≤ d, with the parity of d.
- $C,D$ are real, of degree ≤ d−1, with the parity of d−1.
- Unitarity: $A^2+B^2+(1-x^2)(C^2+D^2)=1$.

Proof by induction: a wait sends the $\{\mathbb 1,X\}$ part into itself (×cos θ) and into $\{Y,Z\}$ (×sin θ), and sends $\{Y,Z\}$ back into $\{\mathbb 1,X\}$ through $\sin^2\theta=1-x^2$; an x-pulse keeps the two blocks separate. Why only cosines appear: x-only pulses give $u(-\Delta)=Xu(\Delta)X$.

**Dictionary between the two bases:**

| | Z basis (notebooks) | $\pm$ basis (textbook) |
|---|---|---|
| stay / diagonal | $P=A-i\sin\theta\,C=\mathrm{Re}P_\pm-i\sin\theta\,\mathrm{Re}Q_\pm$ | $P_\pm=A-iB$ |
| flip / off-diagonal | $Q=-iB+\sin\theta\,D=i\,\mathrm{Im}P_\pm-\sin\theta\,\mathrm{Im}Q_\pm$ | $-i\sin\theta\,Q_\pm$, with $Q_\pm=C-iD$ |
| unitarity | $\lvert P\rvert^2+\lvert Q\rvert^2=1$ | $\lvert P_\pm\rvert^2+(1-x^2)\lvert Q_\pm\rvert^2=1$ |

(Inside table cells write $\lvert\cdot\rvert$, never a literal `|` inside math.)

**Populations.** Starting in $\lvert\uparrow\rangle$, the flip probability is $q=\lvert Q\rvert^2=B^2+(1-x^2)D^2$. It is even in $x$, so it is a polynomial in $x^2=(1+\cos\Delta\tau)/2$: $q=\sum_{n=0}^{d}c_n\cos n\Delta\tau$, symmetric in Δ. The half angle exists because each spinor component picks up $e^{\mp i\theta}$; a probability sees only the differences between paths, which are multiples of $\Delta\tau$. DQ readout: from $\lvert0\rangle$, $P_0=(1-2q)^2$.

**Chebyshev polynomials.**
- Definition: $T_n(\cos\theta)=\cos n\theta$ and $U_n(\cos\theta)=\sin(n+1)\theta/\sin\theta$.
- First terms: $T_{0..4}=1,\ x,\ 2x^2-1,\ 4x^3-3x,\ 8x^4-8x^2+1$; $U_{0..4}=1,\ 2x,\ 4x^2-1,\ 8x^3-4x,\ 16x^4-12x^2+1$.
- Recurrence: $T_{n+1}=2xT_n-T_{n-1}$ (the same for $U$), from $\cos(n{+}1)\theta+\cos(n{-}1)\theta=2\cos\theta\cos n\theta$. In words, multiplying by $x$ moves harmonic n to n±1, which is what one wait does.
- Properties: $\lvert T_n\rvert\le1$ on $[-1,1]$ with n+1 alternation points; $T_n^2+(1-x^2)U_{n-1}^2=1$.
- Range: $x\in[-1,1]$ covers $\Delta\tau\in[0,2\pi]$, and the sign of Δ is invisible in $x$.

**Laurent ↔ Chebyshev.**
$$L(z)=\sum_ka_kz^k=\sum_{n\ge0}e_nT_n(x)+i\sin\theta\sum_{n>0}o_nU_{n-1}(x),\qquad e_n=a_n+a_{-n}\ (e_0=a_0),\quad o_n=a_n-a_{-n}.$$
Both sides have the same number of coefficients ($d+1$), so this is a change of basis among functions. In the $\pm$ basis each entry is a single half; in the Z basis each entry contains both halves.

**Symmetries in the Z basis.**
- x-only pulses: $P$ has real Laurent coefficients and $Q$ imaginary ones (a path back to ↑ flips an even number of times, and each flip contributes $-i$). Equivalently, $P(-\Delta)=P(\Delta)^*$ and $Q(-\Delta)=-Q(\Delta)^*$ (`dq-qsp-prep` §3.1, `dq-pulses` §3.4).
- A palindromic sequence adds $u^T=u$, so $Q$ is purely imaginary at every Δ, $D=0$, and $Q=-iB(x)=-i\sum_kc_k\cos k\theta$.
- Pulses with general (equatorial) axes break both symmetries. That is the GQSP case: complex, non-symmetric Laurent pairs.

**Worked examples (use them on the slides):**
- **All pulses off:** $u=R_z(d\Delta\tau)$, so $P=z^{-d}$, $Q=0$; $P_\pm=T_d(x)$, $Q_\pm=U_{d-1}(x)$.
- **Ramsey, $d=1$,** $R_x(\frac\pi2)WR_x(\frac\pi2)$: $A=0$, $B=x$, $C=1$, $D=0$. So $P=-i\sin\theta$, $Q=-i\cos\theta$; $P_\pm=-ix$, $Q_\pm=1$; $q=\cos^2(\Delta\tau/2)=(1+\cos\Delta\tau)/2$.
- **$d=2$,** $u=W\,R_x(\phi)\,W$: in the Z basis there are two paths, each one monomial, $P=c_\phi z^{-2}$ and $Q=-is_\phi$. In the $\pm$ basis, $P_\pm=c_\phi T_2(x)-is_\phi$ and $Q_\pm=c_\phi U_1(x)$. Check: $c^2[T_2^2+(1-x^2)U_1^2]+s^2=1$. Here the Laurent form is the simpler one. Also, $q=s_\phi^2$ does not depend on Δ, because waits next to a Z preparation or readout are only phases.

**Reachability.**
- Z basis: every Laurent pair of degree d with the parity of d and $\lvert P\rvert^2+\lvert Q\rvert^2=1$ on the circle is produced by $d+1$ pulses about equatorial axes (Low–Yoder–Chuang 2016; Low–Chuang 2017; Haah 2019; GQSP: Motlagh–Wiebe 2024).
- $\pm$ basis, x-only pulses: the degree, parity and $\lvert P_\pm\rvert^2+(1-x^2)\lvert Q_\pm\rvert^2=1$ conditions (Martyn et al. 2021, Thm. 1).
- Count: $2(d+1)$ real coefficients minus $d+1$ unitarity equations leaves $d+1$ angles.
- Corollary: any real $f$ with the parity of d and $\lvert f\rvert\le1$ equals $A=\frac12\mathrm{Tr}\,u$ for some sequence; shifting an end pulse by π swaps $A$ and $B$ (up to sign).
- Recovering the pulses: complete the missing entry by Fejér–Riesz, then strip one wait and one pulse at a time.

**Setting the target: the five choices, instantiated for the DQ π pulse.**
1. **Spin-½ condition.** Balanced DQ driving acts as $D^1(u)$. The DQ π pulse is $P=0$, $\lvert Q\rvert=1$. Since $D^1(-u)=D^1(u)$, only $Q^2$ matters.
2. **Target set.** The ¹⁴N nucleus is a classical label: $\Delta_m=mA$ with $m\in\{-1,0,1\}$ and $A/2\pi=2.16$ MHz, and the electron undergoes $\frac13\sum_mU_m\rho U_m^\dagger$. Phase coherence needs $P(\Delta_m)=0$ and a common $Q(\Delta_m)^2$.
   - Point case: the three resonances.
   - Robust case: windows $\lvert\Delta-mA\rvert\le0.1A$ with $1-\lvert Q\rvert^4\le10^{-3}$, equivalent to $\lvert P\rvert\le0.022$.
   - Elsewhere: formulation R leaves it free; formulation S requires $Q=0$ on O (|Δ| ≤ 3A, outside the transition bands of width 0.2A).
3. **Symmetry.** With palindromic ±x pulses, $Q=-iB(x)$ and phase coherence is automatic. **The target is the single real function $B$**: $B=s_m=\pm1$ on the windows (signs free), $\lvert B\rvert\lesssim0.022$ on O, $\lvert B\rvert\le1$ everywhere, free in the bands. $P$ is not prescribed; it follows from completion.
4. **Choose τ; it sets $\theta_m=mA\tau/2$.**
   - Resonances that need different operations must land on separate arcs.
   - Periodic images of a window must not fall where you impose a constraint.
   - A constant target needs even d (odd d has no constant term: $E_1=0.59$ in formulation R).
   - τ = 2π/A deliberately puts all three resonances on one point, which is harmless when they need the same operation.
5. **Solve.** Points → interpolation; windows → best uniform approximation (linear program, `dq-pulses` §3.4) or the Gaussian-smoothed Fourier series of the step (§3.2).

**What the recipe gives for the DQ π pulse:**
- **Formulation R, instantaneous pulses:** $B\equiv\pm1=T_0$, so $d=0$ and one π pulse is exact at every Δ; the best τ is 0.
- **$d=1$ design from the target alone:** take signs $(-,+,-)$ and τ = 2π/A, so that $x_m=-1,1,-1$. Then $B=T_1(x)=x$, completion gives $A=0$ and $C=1$, and the sequence is $R_x(\frac\pi2)WR_x(\frac\pi2)$ with $P=-i\sin(\Delta\tau/2)$: the analytic design of `dq-pulses` §3.3. Its value is with real pulses: a real π/2 pulse is fixed by shortening the wait by $2/\Omega$ (Euler form), whereas a real π pulse cannot be fixed this way because $\tan\frac\pi2$ diverges.
- **Formulation S at τ = 2π/6A:** each A is 30° of θ. On θ ∈ [0°, 90°] (i.e. $x\in[0,1]$, |Δ| ≤ 3A; the rest follows from evenness and parity) the target is:

  | θ | region | target |
  |---|---|---|
  | 0–3° | window, m = 0 | $B=s_0$ |
  | 3–9° | band | free |
  | 9–21° | O | $B\approx0$ |
  | 21–27° | band | free |
  | 27–33° | windows, m = ±1 | $B=s_1$ |
  | 33–39° | band | free |
  | 39–90° | O | $B\approx0$ |

  A step across a band of width Δθ needs degree $d\propto\ln(1/\varepsilon)/\Delta\theta$. Present this as the standard scaling for the sign function with a gap (Low–Chuang; Martyn et al. 2021), not as derived here. It is consistent with the notebook's 36 / 68 / 104 pulses for ε = 0.1 / 1e-2 / 1e-3 (linear in ln(1/ε)). Because $\Delta\theta=(0.2A)\tau/2$, the total free evolution $d\tau$ is set by the band width in Δ, not by τ: 5.9–6.8 µs for every τ (Table 3.11).

## 6. Source B — the notebooks and pages (re-verify every number against the current text)

**Main source: `3. wiki/code/dq-pulse-notebook/dq-pulses.ipynb`** (read the markdown cells; the README gives the structure). Use these sections:
- §1.1: the $D^1(u)$ reduction.
- §2.1–2.3: the ¹⁴N model and the single-pulse error $\frac43\eta_A^2$.
- §3.1 target, §3.2 Laurent/Chebyshev and the step-target expansion, §3.3 instantaneous-pulse limit, §3.4 optimization, §3.5 finite pulses.

Numbers to quote (verify each one first):
- **Single DQ π pulse:** population error 6.3e-3 at the drive of Mamin et al. ($\eta_A=0.069$) and 0.11 at Ω/2π = 10 MHz.
- **Exact five-pulse palindromic designs:** 777 / 971 / 1324 ns at the drive of Mamin et al. / 20 MHz / 10 MHz.
- **Robust designs (formulation R), fixed-duration pulses:** 990 ns (K = 7) at 10 MHz and 354 ns (K = 5) at 20 MHz, without free evolution.
- **Robust designs (formulation R), full-amplitude pulses:** 491 ns at 10 MHz and 255 ns at 20 MHz.
- **Formulation S:** $d_{\min}=39$ at τ = 2π/3A and ≥ 5.9 µs for any pulse shape. At τ = 2π/6A it needs 36 / 68 / 104 pulses (2.7 / 5.2 / 7.9 µs of free evolution).
- **Smoothed series against best approximation:** degree 48 against 21 at 2π/3A, and 109 against 41 at 2π/6A.
- **Analytic design with real pulses:** wait $2\pi/A-2/\Omega_{\max}$; population error 1.4e-6 at the drive of Mamin et al. and 5.3e-4 at 10 MHz.
- **Laboratory frame:** 3.1e-4 at 10 MHz and 8.3e-4 at 20 MHz, against 0.11 and 3.0e-2 for a single π pulse.
- **Effective degree:** $n_{\rm eff}$ = 7.4 (10 MHz) and 2.6 (20 MHz) without free evolution.

**Polynomial exposition: `3. wiki/code/dq-qsp-prep/dq-qsp-prep.ipynb` §3.2.** This is the reference exposition for the series: notation table, two-pulse worked example, layer stripping. Its §3.1 gives the preparation target, $(P_m,Q_m)=(1/\sqrt2,-i/\sqrt2)$ on all windows, with $P=1$ outside in the selective formulation.

**Related notebooks; the same §3.1–3.5 outline, used for the optional template slide:**
- `dq-qsp-decoupling` (a decoupling sequence has a constant QSP response),
- `single-frequency-qsp` (one frequency, qubit {0, −1} at 44 mT; Appendix D is the universal NOT),
- `sequential-tone-qsp` (two frequencies one at a time).

Take each target statement from that notebook's §3.1.

**Wiki pages:**
- `3. wiki/projects/qsp/qsp-state.md`
- `3. wiki/projects/qsp/conditional-gate.md` §4.2–4.3 (the Hadamard bridge; Laurent or Chebyshev)
- `3. wiki/projects/qsp/universal-not.md` (the only target that differs between windows)
- `3. wiki/concepts/qsp-phase-filter-design.md`
- `3. wiki/concepts/double-quantum-driving.md`

**Previous deck: `3. wiki/labmeeting/260923/`, chapter 3 (slides 30–39z).** It already covers the sequence, the block recursion, the Chebyshev example, the theorem, reachability and the peel-off recursion.
- Do not repeat it at length. One recap slide plus pointers is enough.
- Reuse its clips (`media/qsp-blocks.mp4`, `media/qsp-chebyshev.mp4`, `media/chebyshev.png`) only where they fit the notation contract, and say which you reused.
- Note that its sequence slide uses a $y$ drive and $z=e^{i\delta\tau}$.

**References** (slide 80): take the bibliographic data from `1. raw/references.bib` and the literature notes in `3. wiki/papers/`, not from memory. The vault has `@motlagh2024generalized`, `@martyn2021grand`, `@wang2022energy` and `@mamin2014multipulse`. Low–Yoder–Chuang PRX 6, 041067 (2016), Low–Chuang PRL 118, 010501 (2017) and Haah, Quantum 3, 190 (2019) are cited in the notebooks but not filed; list them as plain references.

## 7. Figures

- **Canonical sources are the executed notebooks.** Their figures are embedded PNG outputs; `out/` holds only JSON. Extract the PNGs you need (cell index → `media/<notebook>-c<NN>.png`, decoded from `image/png` outputs) and record their provenance in a speaker-note comment.
- **Candidate cells in `dq-pulses.ipynb`:**
  - 22: step-target coefficients and the constructed sequence;
  - 33: robust designs, error as a function of detuning;
  - 35: target and best approximations for several K, error as a function of K;
  - 38: design space as a function of τ;
  - 46: $P$ and $Q$ in the complex plane;
  - 49: pulse waveforms;
  - 60: Chebyshev and Laurent coefficients and $n_{\rm eff}$.
  
  Look at each one before choosing; crop to the panels you need.
- **New didactic figures** (closed-form, no simulation) go into `3. wiki/code/lm-<DATE>-figures/`, following the code conventions in `CLAUDE.md` (a deterministic script, `README.md`, `out/`), and are copied into the deck's `media/`. Suggested figures:
  1. the two path lattices (Z basis: phase per wait; $\pm$ basis: stay or flip);
  2. $T_n$ and $\sin\theta\,U_{n-1}$ for n = 1–4;
  3. the unit circle with the three resonances for τ = 2π/A, 2π/3A and 2π/6A (aliasing, separation, periodic images);
  4. the formulation-S target on θ ∈ [0°, 90°], with windows, bands and O shaded.
  
  Ask the human whether they want Manim clips (earlier decks used them: `lm-260923-animations`); do not make any by default.

## 8. Marp kit mechanics and style

- **Kit.** One file per slide, numbered for order. Deck-wide frontmatter goes in `00-title.md` (`math: mathjax`, `paginate`, `footer`). Use the classes from the kit README: `title`, `section`, `closing`, `.cat` pills, `.columns`, `figure`, `.callout`, `.ref`.
- **MathJax.**
  - The physics and braket packages are loaded.
  - `\ketbra{a}{b}` takes **two** arguments; use `\dyad{+}` for $\lvert+\rangle\langle+\rvert$.
  - Use `\newcommand`, never `\gdef`.
  - Math is not parsed inside multi-line raw-HTML blocks.
  - No literal `|` inside math in a Markdown table; use `\lvert…\rvert` or `\vert`.
- **House style** (from the 260923 and 260929 reviews):
  - few words per slide and declarative titles;
  - no descriptive figure captions (source, parameters and provenance only);
  - explanations and derivation steps the speaker should say aloud go into `<!-- speaker notes -->`;
  - full derivations go to backup slides;
  - write "as a function of", never "versus";
  - standard terms only (§4).
- **Equations over words.** Every message slide should carry its one defining equation or table from §5. Each claim comes from §5 or §6; mark anything else `[unverified]` in a speaker note and raise it with the human.

## 9. Final report to the human

Report:
- the slide count and the path of the built PDF;
- which figures you extracted, newly plotted or reused;
- every `TODO (speaker to confirm)`;
- any number in §6 that no longer matches the current notebook.
