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

<!--
H_m in the rotating frame.

Take one block H_m and go to the same rotating frame as before. The common-mode detuning D - (omega_1 + omega_2)/2 does not depend on m and is set to zero. The differential detuning becomes Delta_m = Delta_0 + A_par m, where Delta_0 is the field detuning of the carrier pair, which we tune to zero on the m_I = 0 resonance. So each manifold is the same spin-1 in an effective field, with its own static detuning Delta_m: the 3x3 matrix above, which is Delta_m S_z + Omega_DQ (cos phi S_x + sin phi S_y), and its spin-1/2 image h_m. The sign convention: A_par is negative; we write A = |A_par| and Delta_m = m A, which does not matter because everything is symmetric in m.

Because the blocks never mix, the whole evolution is block diagonal: U = sum_m D^1(u_m) (x) |m><m|, with u_m the spin-1/2 evolution of the sequence at detuning Delta_m.

At room temperature the 14N is unpolarised: p_m = 1/3 each, a nuclear state diagonal in m. Trace out the nucleus, and the electron undergoes a mixed-unitary channel: with probability 1/3 it evolves with D^1(u_m). No electron-nuclear entanglement is created: for the electron the nucleus is a classical random label that selects one of three detunings, 0 and +-2.16 MHz. This is dephasing by a static distribution, not decoherence. No single carrier frequency removes all three detunings.

So the design problem is: one sequence u, evaluated at three detunings, must do the right thing at all three.
-->
