---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

<style scoped>
.columns { gap: 1.2rem; }
.columns .col:first-child { flex: 0 0 520px; }
table { font-size: 0.74em; }
mjx-container[display="true"] { margin: 0.35em 0 !important; }
.callout { margin: 0.4em 0; }
</style>

# <span class="cat intro">Intro</span> Single-pulse error

One rectangular DQ $\pi$ pulse misses the outer hyperfine manifolds.

<div class="columns">
<div class="col">

<figure class="figure">

![w:480](media/dq-inversion.png)

</figure>

</div>
<div class="col">

$$
q=1-\eta_m^2+O(\eta^4),\qquad 1-P_{-1}\ \approx\ \tfrac43\,\eta_A^2
$$

with $\eta_m=\Delta_m/\Omega_{DQ}$ and $\Omega=\sqrt2\,\Omega_{DQ}$.

| drive | $\eta_A$ | $1-P_{-1}$ | $\pi/2$ phase |
|---|---|---|---|
| Mamin et al. | 0.069 | $6.3\times10^{-3}$ | $\pm7.9^\circ$ |
| $\Omega/2\pi=20$ MHz | 0.153 | $3.0\times10^{-2}$ | $\pm17.5^\circ$ |
| $\Omega/2\pi=10$ MHz | 0.305 | $0.115$ | $\pm34.9^\circ$ |

- The $\pi$ pulse leaves population in $\ket{0}$.
- The $\pi/2$ pulse fails through its phase.

</div>
</div>

<div class="callout">

**Phase coherence needs one common $Q^2$:** $\;Q(\Delta_m)=s_m\,e^{i\chi}$ with signs $s_m=\pm1$ and one phase $\chi$.

</div>

<!--
One rectangular DQ pi pulse is not enough.

The figure: one rectangular DQ pi pulse, Omega_DQ T = pi, in a manifold with detuning Delta. Rabi's formula for the spin-1/2 gives the flip probability q = (Omega_DQ^2 / Omega'^2) sin^2(Omega' T / 2), with Omega' = sqrt(Omega_DQ^2 + Delta^2); the DQ transfer |+1> -> |-1> is q^2, both spins must flip. The dots are the three hyperfine resonances, Delta = 0 and +-A. Three drives: the drive of Mamin et al., Omega_DQ/2pi = 31.25 MHz with a 16 ns pi pulse, and Omega/2pi = 20 MHz and 10 MHz, where Omega = sqrt2 Omega_DQ is the single-quantum Rabi frequency (Omega_DQ/2pi = 14.1 and 7.07 MHz). As the drive weakens, the profile narrows and the outer manifolds fall off.

Expansion in eta_m = Delta_m / Omega_DQ: q = 1 - eta^2 + O(eta^4), so P_(-1) is about 1 - 2 eta^2, and the error is population left in |0>, P_0 about 2 eta^2. All errors are even in eta, so the average over the nucleus is 2/3 of the outer-manifold value: 1 - P_(-1) is about (4/3) eta_A^2, with eta_A = A / Omega_DQ.

Exact values averaged over the nucleus: 6.3e-3 at the drive of Mamin et al. (eta_A = 0.069), 3.0e-2 at 20 MHz, 0.115 at 10 MHz (eta_A = 0.305); the process infidelities are 8.5e-3, 4.1e-2 and 0.15. A factor of about 20 between the drive of Mamin et al. and 10 MHz.

The last column is the preparation, the pi/2 pulse that takes |0> into the +-1 superposition. It is insensitive in population: what stays in |0> is fourth order in eta, only 7e-7 at the drive of Mamin et al. and 2.6e-4 at 10 MHz. But the relative phase of the +-1 components is off by about 2 eta_m, with the sign of m_I: +-7.9 degrees, +-17.5 degrees and +-34.9 degrees in the outer manifolds. Averaged over the nucleus, a phase that depends on m shrinks the coherence. The preparation fails through its phase.

Hence the requirement for part three. A requirement on populations alone is not enough for DQ sensing: the three manifolds must also acquire the same phase. In D^1(u) after a DQ pi pulse, |+1> -> Q^2 |-1>, so the phase that matters is that of Q^2. Phase coherence across the manifolds needs one common Q^2: Q(Delta_m) = s_m e^{i chi} with one chi for all three m. The signs s_m are free, because only Q^2 enters.
-->
