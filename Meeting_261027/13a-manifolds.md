---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat intro">Intro</span> Hyperfine manifolds

Each hyperfine manifold is a spin-1 with its own detuning. In the rotating frame, with the balanced drive, in the basis $(\ket{{+1}},\ket{0},\ket{{-1}})$:

$$
\tilde H_m=\begin{pmatrix}\Delta_m&\frac{\Omega_{DQ}}{\sqrt2}e^{-i\varphi}&0\\ \frac{\Omega_{DQ}}{\sqrt2}e^{i\varphi}&0&\frac{\Omega_{DQ}}{\sqrt2}e^{-i\varphi}\\ 0&\frac{\Omega_{DQ}}{\sqrt2}e^{i\varphi}&-\Delta_m\end{pmatrix}
=\Delta_mS_z+\Omega_{DQ}(\cos\varphi\,S_x+\sin\varphi\,S_y),
\qquad \Delta_m=\Delta_0+mA .
$$

$$
h_m=\tfrac{\Delta_m}{2}\sigma_z+\tfrac{\Omega_{DQ}}{2}(\cos\varphi\,\sigma_x+\sin\varphi\,\sigma_y),
\qquad
U=\sum_m D^1(u_m)\otimes\ketbra{m}{m}.
$$

Unpolarised nucleus, $p_m=\tfrac13$, $A/2\pi=2.16$ MHz:

$$
\rho_e\ \longmapsto\ \frac13\sum_{m=-1}^{1}D^1(u_m)\,\rho_e\,D^1(u_m)^\dagger .
$$

- The drive sits on the $m_I=0$ resonance, $\Delta_0=0$; the outer manifolds see $\pm A$.
- No entanglement is created: the nucleus is a classical label.
