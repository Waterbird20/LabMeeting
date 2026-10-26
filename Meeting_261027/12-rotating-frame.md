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
