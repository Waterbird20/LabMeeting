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

<!-- TODO (speaker to confirm): $N$ counts repetitions of the unit here, so the clip runs $N=32$ units, that is $64$ pulses; the paper counts $\pi$ pulses ($N=32$ in its Fig. 2) and writes $\phi$ for half the Bloch angle, which makes the printed formula identical. Say which convention you prefer out loud. -->

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

<video src="media/dd-cpmg.mp4" poster="media/dd-cpmg.png" width="730" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
</div>
