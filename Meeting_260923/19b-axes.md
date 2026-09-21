---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<!-- EDIT-FORWARD: $N$ counts repetitions of the unit here, so the clip runs $N=32$ units, that is $64$ pulses; the paper counts $\pi$ pulses ($N=32$ in its Fig. 2) and writes $\phi$ for half the Bloch angle, which makes the printed formula identical. Say which convention you prefer out loud. -->

# <span class="cat intro">Intro</span> Same angle, different axis: the dips that name one nucleus

With $A=U_0(\tau)$, $B=U_1(\tau)$, cyclicity gives $\operatorname{Tr}V_0=\operatorname{Tr}(ABBA)=\operatorname{Tr}(A^{2}B^{2})=\operatorname{Tr}V_1$, so both branches turn by the **same angle** $\phi$ and only the axes $\hat n_0,\hat n_1$ can differ; each nucleus has its own $A_\parallel$, hence its own comb $\tau_k$:

$$
M=1-\big(1-\hat n_0\!\cdot\!\hat n_1\big)\sin^{2}\tfrac{N\phi}{2},
\qquad
\tau_k=\frac{(2k-1)\pi}{2\omega_L+A_\parallel} .
$$

<figure class="figure">

<video src="media/dd-cpmg.mp4" poster="media/dd-cpmg.png" width="580" autoplay loop muted playsinline preload="none"></video>

*Spin 3 of the paper's Table I ($B_0=401$ G, $N=32$ units): the axes coincide except at $\tau_7,\tau_8,\tau_9$, where they anti-align and $M$ plunges.*

</figure>
