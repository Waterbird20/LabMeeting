---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
mjx-container[display="true"] { margin: 0.3em 0 !important; }
</style>

# <span class="cat intro">Intro</span> Rotating frame

The rotating frame leaves a differential and a common-mode detuning. In the laboratory frame, with a field $B$ along the NV axis and one microwave field with two frequency components along $x$,

$$
H(t)=DS_z^2+\gamma BS_z+\big[\gamma B_1\cos(\omega_1t+\delta_1)+\gamma B_2\cos(\omega_2t+\delta_2)\big]S_x .
$$

Rotate at $\omega_1$ on $\ket{{+1}}$ and at $\omega_2$ on $\ket{{-1}}$, and drop the terms that oscillate at frequencies of order $D$ or $2\gamma B$ (rotating-wave approximation; Mamin et al., PRL 113, 030803 (2014)). In the basis $(\ket{{+1}},\ket{0},\ket{{-1}})$,

$$
\tilde H=\begin{pmatrix}\Delta_1&\frac{\gamma B_1}{2\sqrt2}e^{-i\delta_1}&0\\ \frac{\gamma B_1}{2\sqrt2}e^{i\delta_1}&0&\frac{\gamma B_2}{2\sqrt2}e^{i\delta_2}\\ 0&\frac{\gamma B_2}{2\sqrt2}e^{-i\delta_2}&\Delta_2\end{pmatrix},
\qquad \Delta_{1,2}=D\pm\gamma B-\omega_{1,2}.
$$

$$
\mathrm{diag}(\Delta_1,0,\Delta_2)=\Delta\,S_z+\Delta_c\,S_z^2,\qquad
\Delta=\tfrac{\Delta_1-\Delta_2}{2},\quad \Delta_c=\tfrac{\Delta_1+\Delta_2}{2}.
$$

- The differential detuning $\Delta$ comes from the magnetic field and the hyperfine shift.
- The common-mode detuning $\Delta_c$ comes from the carrier offset from $D$, temperature and strain.


<!--
The rotating frame.

Start in the laboratory frame. The NV ground state is a spin-1 with zero-field splitting D and Zeeman term gamma B S_z. The microwave field along x has two frequency components, with amplitudes B_1, B_2, frequencies omega_1, omega_2 and phases delta_1, delta_2.

Go to the frame that rotates at omega_1 on |+1> and at omega_2 on |-1>. Every term that still oscillates at frequencies of order D or 2 gamma B is dropped: that is the rotating-wave approximation. What is left is time independent. The off-diagonal elements are the matrix elements of S_x, 1/sqrt2, times half the drive amplitude, gamma B_j / 2, with the phase of each component. This is Eq. (S3) of Mamin et al.

The two diagonal entries are the detunings of each component from its own transition, Delta_1 = D + gamma B - omega_1 and Delta_2 = D - gamma B - omega_2. Their difference and their mean have different physics. S_z = diag(1, 0, -1) and S_z^2 = diag(1, 0, 1), so diag(Delta_1, 0, Delta_2) = Delta S_z + Delta_c S_z^2.

The differential detuning Delta = (Delta_1 - Delta_2)/2 moves +1 and -1 in opposite directions: this is what the magnetic field does, and what the secular hyperfine interaction with the nucleus does. It is the signal in magnetometry, and the error we fight in this talk.

The common-mode detuning Delta_c = (Delta_1 + Delta_2)/2 moves both levels together: a carrier away from D, temperature (D depends on temperature) and strain.
-->
