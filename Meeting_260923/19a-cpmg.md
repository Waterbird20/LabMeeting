---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
.columns { gap: 1.2rem; }
.columns .col:first-child { flex: 0 0 400px; }
</style>

<!-- EDIT-FORWARD: confirm the source of the figure and what alpha counts (Bradley et al. 2019?) -->

# <span class="cat intro">Intro</span> The read-out is one number: how well the two branches agree

Prepare the electron in $\ket{+}$, leave the nucleus unpolarised, $\rho_n=I/2$, and run $N$ units. Tracing the nucleus out leaves the electron coherence $\bra{0}\rho_e\ket{1}=\tfrac12M$, so a read-out along $x$ returns

$$
P_x=\tfrac12+\operatorname{Re}\bra{0}\rho_e\ket{1}=\tfrac{1+M}{2},
\qquad
M=\tfrac12\operatorname{Tr}\big(V_0^{N}V_1^{\dagger N}\big),
\qquad |M|\le 1 .
$$

<div class="columns">
<div class="col">

That $\tfrac12$ is the trace normalisation of the unpolarised nucleus, so $M$ is an average of $V_1^{\dagger N}V_0^{N}$. If the operation is **unconditional**, $V_0=V_1$, then $M=1$ exactly and no signal is lost.

The electron Bloch vector has length $|M|$ and purity $\operatorname{Tr}\rho_e^{2}=\tfrac12(1+|M|^{2})$, so staying unconditional keeps the state pure. More pulses therefore buy a longer coherence time $T_2^{(N)}$.

</div>
<div class="col">

<figure class="figure">

![w:470](media/dd-t2-vs-pulses.png)

*More decoupling, longer coherence: the decay to the $0.5$ floor moves from about $1.5\,$s at $\alpha=1$ out to about $25\,$s at $\alpha=256$, as the unconditional picture predicts.*

</figure>

</div>
</div>
