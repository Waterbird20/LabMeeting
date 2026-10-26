---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
.columns .col:last-child { flex: 0 0 430px; }
mjx-container[display="true"] { margin: 0.3em 0 !important; }
</style>

# <span class="cat intro">Intro</span> Detuning problem

From the ideal Hamiltonian, with one microwave field of two frequency components,

$$
H(t)=DS_z^2+\gamma BS_z+\big[\gamma B_1\cos(\omega_1t+\varphi_1)+\gamma B_2\cos(\omega_2t+\varphi_2)\big]S_x ,
$$

in the double rotating frame, with the rotating-wave and secular approximations, $B_1=B_2$, $\varphi_1=-\varphi_2=\varphi$ and zero detuning, in the basis $(\ket{{+1}},\ket{0},\ket{{-1}})$:

<div class="columns">
<div class="col">

$$
\tilde H_0=\begin{pmatrix}0&\frac{\gamma B_1}{2\sqrt2}e^{-i\varphi}&0\\ \frac{\gamma B_1}{2\sqrt2}e^{i\varphi}&0&\frac{\gamma B_1}{2\sqrt2}e^{-i\varphi}\\ 0&\frac{\gamma B_1}{2\sqrt2}e^{i\varphi}&0\end{pmatrix}
=\Omega_{DQ}\,(\cos\varphi\,S_x+\sin\varphi\,S_y)
$$

- In the presence of interacting spins, including ¹⁴N and ¹³C, each spin induces a detuning that depends on its own spin state.
- Two drive frequencies give two detunings, $\delta_{+1}$ and $\delta_{-1}$. A coupling $\propto S_z$ shifts $\ket{\pm1}$ oppositely: $\delta_{+1}=-\delta_{-1}=\Delta$.

</div>
<div class="col">

<figure class="figure">

<video src="media/detuning-sources.mp4" poster="media/detuning-sources.png" width="450" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
</div>
