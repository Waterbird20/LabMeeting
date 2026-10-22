---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

# Take-home

$$
\rho_e\mapsto\frac13\sum_{m}U_m\,\rho_e\,U_m^\dagger,\qquad Q=-i\,r(x),\qquad r(x)=\sum_{k\equiv K\,(2)}c_k\,T_k(x)
$$

<div class="callout">

1. The DQ problem we handle is mixed and detuned: the unpolarised ¹⁴N gives three detunings $\Delta_m=mA$, and the electron sees their average.
2. QSP tunes a Laurent or Chebyshev polynomial. The best polynomial approximation is the upper bound on what $N$ pulses can do, so setting the target function is what matters.
3. Setting $\tau$ is not free: it places the windows on the circle, and no image of a window may land in the stopband.
4. Real pulses break the periodicity and must be retrained; at full amplitude that costs 0 to 2 extra pulses.

</div>

<!--
Four points to take home.

First, the problem. The DQ pi pulse we need acts on a mixed, detuned spin: the nitrogen is unpolarised, so with probability one third each it shifts the transition by -A, 0 or +A, and the electron undergoes the average of three different unitaries. One pulse cannot be right for all three.

Second, the tool. QSP turns a sequence of waits and pulses into a Laurent polynomial in z, equivalently a Chebyshev polynomial in x = cos(Delta tau / 2), and the pulse angles tune its coefficients. The best polynomial approximation of the target is the upper bound on what N pulses can achieve, and any good sequence reaches it. So the real design work is in setting the target function: which detunings must be inverted, which must be left alone, and which are free.

Third, the free evolution tau is not a free parameter. It decides where the three windows sit on the circle and where their periodic images land; at tau = 2 pi / A all three windows map to one point, and 14 pulses meet the tolerance with 6.0 microseconds of free evolution. In the gaps between compatible values the target cannot be met at all.

Fourth, real pulses depend on the detuning, break the periodicity and must be retrained. With full-amplitude pulses that costs no extra pulse at 20 MHz and one or two at 10 MHz.
-->
