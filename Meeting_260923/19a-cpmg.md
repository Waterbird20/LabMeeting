---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat intro">Intro</span> The read-out is one number: how well the two branches agree

Prepare the electron in $\ket{+}$, leave the nucleus unpolarised, $\rho_n=I/2$, and run $N$ units:

$$
\tfrac{1}{\sqrt2}\big(\ket{0}+\ket{1}\big)\otimes\rho_n
\ \longrightarrow\
\tfrac{1}{\sqrt2}\big(\ket{0}\otimes V_0^{N}+\ket{1}\otimes V_1^{N}\big)\ \text{acting on}\ \rho_n .
$$

Tracing the nucleus out leaves the electron coherence
$\bra{0}\rho_e\ket{1}=\tfrac12\operatorname{Tr}_n\!\big(V_0^{N}\rho_nV_1^{\dagger N}\big)=\tfrac12M$,
so a read-out along $x$ returns

$$
P_x=\tfrac12+\operatorname{Re}\bra{0}\rho_e\ket{1}=\frac{1+M}{2},
\qquad
M=\tfrac12\operatorname{Tr}\big(V_0^{N}V_1^{\dagger N}\big).
$$

<div class="columns">
<div class="col">

That $\tfrac12$ is the trace normalisation of the unpolarised nucleus, so $M$ is an average of $V_1^{\dagger N}V_0^{N}$ and $|M|\le1$. If the operation is **unconditional**, $V_0=V_1$, then $M=1$ exactly. 

</div>
<div class="col">

The electron Bloch vector has length $|M|$ and purity $\operatorname{Tr}\rho_e^{2}=\tfrac12(1+|M|^{2})$, so staying unconditional keeps the state pure and the signal full. That is why decoupling buys a coherence time $T_2^{(N)}$ far longer than the single-pulse echo $T_2$ (Pham thesis, Ch. 1).

</div>
</div>

<!-- EDIT-FORWARD: the measured $T_2$ versus number of pulses plot goes here; the speaker will supply it. -->
