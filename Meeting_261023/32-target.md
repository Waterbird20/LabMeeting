---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
.columns .col:first-child { flex: 0 0 520px; }
</style>

# <span class="cat strategy">Strategy</span> Target function

The target is a step function of the detuning.

<div class="columns">
<div class="col">

<figure class="figure">

<video src="media/target-build.mp4" poster="media/target-build.png" width="520" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
<div class="col">

- **Passband** $\lvert\Delta-mA\rvert\le0.1A$: invert, $P=0$, $Q=s_m e^{i\chi}$ with one common $\chi$.
- **Stopband** $0.3A\le\lvert\Delta\rvert\le0.7A$ and $1.3A\le\lvert\Delta\rvert\le1.5A$: nothing happens, $Q=0$.
- **Transition bands** in between are free.

</div>
</div>

**Error at one detuning**, the process infidelity of the spin-½ operation:

$$
\varepsilon(\Delta)=1-\Big\lvert\tfrac12\mathrm{Tr}\,u_t^\dagger u\Big\rvert^2=
\begin{cases}
\lvert P\rvert^2+\big[\mathrm{Im}\,(e^{-i\chi}Q)\big]^2 & \text{passband: population left behind + phase error}\\[2pt]
\lvert Q\rvert^2 & \text{stopband: population transferred}
\end{cases}
$$

**Uniform error** $E=\max_{\Delta\,\in\,\text{passband}\,\cup\,\text{stopband}}\varepsilon(\Delta)$, the worst case; we ask for $E\le10^{-3}$.

<!--
What the DQ π pulse has to do, detuning by detuning. The nucleus picks one of three detunings, Δ = mA with m = −1, 0, 1. Around each of them we allow a tolerance window of ±0.1A for slow field drifts: this is the passband, where the electron must be inverted, P = 0. For phase coherence Q must be the same up to a sign in all three windows: Q = s_m e^{iχ} with one common phase χ. The signs are free because only Q² enters the spin-1 operation; χ is free because shifting the phase of every pulse multiplies Q by a phase.

Between the windows we want nothing to happen: the stopband, 0.3A to 0.7A and 1.3A to 1.5A in |Δ|, where Q = 0. The phase of P is left free there, because a detuning-dependent phase of P is just extra precession about z. Between passband and stopband there are transition bands of width 0.2A with no requirement: the response has to turn somewhere. Beyond 1.5A nothing is required. So the target is a step function of the detuning. The clip builds it: passband, transition bands, stopband, then the target |Q_t| (top) and |P_t| (bottom).

How we measure the error. At one detuning we compare the spin-½ operation u that the sequence produces with the target operation u_t: ε(Δ) = 1 − |½ Tr u_t† u|² is zero when they agree and one when they are orthogonal, the process infidelity. The free phases (χ, and the phase of P in the stopband) are chosen to make it as small as possible. In the passband this gives |P|², the population that is not transferred, plus [Im(e^{−iχ} Q)]², the part of Q that has the wrong phase. In the stopband it gives |Q|², the population that is transferred although it should not be.

Then E is simply the worst ε over the whole passband and stopband, the uniform error, and the design goal is E ≤ 10⁻³. For the spin-1 operation itself, the population error in the passband is about 2ε and the process infidelity (8/3)ε(1 − (2/3)ε). We do not use |Q − Q_t|² as the error: it is quadratic in the transfer error, so 10⁻³ would allow |P|² ≈ 0.06.
-->
