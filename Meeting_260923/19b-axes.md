---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
.columns { gap: 1.2rem; margin-top: 0.1em; }
.columns .col:first-child { flex: 0 0 420px; }
</style>

<!-- EDIT-FORWARD: $N$ counts repetitions of the unit here, so the clip runs $N=32$ units, that is $64$ pulses; the paper counts $\pi$ pulses ($N=32$ in its Fig. 2) and writes $\phi$ for half the Bloch angle, which makes the printed formula identical. Say which convention you prefer out loud. -->

# <span class="cat intro">Intro</span> Same angle, different axis: the dips that name one nucleus

<div class="columns">
<div class="col">

With $A=U_0(\tau)$ and $B=U_1(\tau)$, cyclicity of the trace gives $\operatorname{Tr}(ABBA)=\operatorname{Tr}(A^{2}B^{2})$, that is $\operatorname{Tr}V_0=\operatorname{Tr}V_1$, so both branches turn by the **same angle** $\phi$ and only the axes $\hat n_0,\hat n_1$ can differ:

$$
M=1-\big(1-\hat n_0\!\cdot\!\hat n_1\big)\sin^{2}\tfrac{N\phi}{2} .
$$

Each nucleus has its own $A_\parallel$, hence its own comb of spacings where the axes anti-align. That comb is a fingerprint: it names the nucleus.

$$
\tau_k=\frac{(2k-1)\pi}{2\omega_L+A_\parallel} .
$$

</div>
<div class="col">

<figure class="figure">

<video src="media/dd-cpmg.mp4" poster="media/dd-cpmg.png" width="680" controls autoplay loop muted playsinline preload="none"></video>

*Spin 3 of the paper's Table I ($B_0=401$ G, $N=32$ units): the axes coincide, open up as $\tau$ nears $\tau_7,\tau_8,\tau_9$, and anti-align there, where $M$ plunges. The nucleus starts in $\ket{0}$, and at the resonance the two branches run round opposite halves of one great circle, antipodal after $16$ units. The curve $P_x(\tau)$ is computed for $\rho_n=I/2$, so it does not depend on that choice.*

</figure>

</div>
</div>
