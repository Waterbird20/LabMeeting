---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat method">Method</span> What $P$ is with no pulses, and what it can be with them

<div class="columns">
<div class="col">

Switch every pulse off. The sequence is bare free precession, $U=W(\delta)^{d}$, whose tap list has a single nonzero entry, so its response is one pure harmonic,
$$
\langle 0|U|0\rangle=e^{-id\delta\tau/2},\qquad
\mathrm{Re}\,\langle 0|U|0\rangle=\cos (dx)=T_d(a),
$$
with $a=\cos x$ and $x=\delta\tau/2$. In the textbook labelling this is the trivial-phase case of [1], $P=T_d(a)$ and $Q=U_{d-1}(a)$, Chebyshev of the first and second kind, and the unitarity condition is just $T_d^{2}+(1-a^{2})U_{d-1}^{2}=\cos^{2}(dx)+\sin^{2}(dx)=1$. **Chebyshev polynomials are the pure harmonics of the wait**, and the pulses are what turns one harmonic into a designed sum of them.

</div>
<div class="col">

<figure class="figure">

![w:360](media/chebyshev.png)

*$P(a)=T_d(a)$ at trivial phases for $d=1,2,5$, from our evaluator (agrees to $10^{-12}$).*

</figure>

At $a=\pm1$ the wait is trivial, so the sequence collapses to one fixed rotation and $|P(\pm1)|=1$ is forced: a target such as $\tfrac12\,\mathrm{sign}(a)$ is unreachable as $P$ itself. Reading only the real part cures it, since $\mathrm{Im}\,P$ can absorb the unit-modulus burden. **$\mathrm{Re}\,P$ may then be any real polynomial with parity $d \bmod 2$ and $|\mathrm{Re}\,P|\le1$** [1].

</div>
</div>
