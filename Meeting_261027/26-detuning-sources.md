---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

<style scoped>
mjx-container[display="true"] { margin: 0.15em 0 !important; }
table { font-size: 0.74em; }
</style>

# <span class="cat intro">Intro</span> Detuning sources

**¹⁴N.** The secular hyperfine term is block diagonal in $m_I$, and in each block it acts as a detuning:

$$
\tilde H=\sum_{m}\tilde H_m\otimes\ketbra{m}{m},\qquad
\tilde H_m=\tilde H_0+A_N\,m\,S_z=
\begin{pmatrix}A_Nm&\frac{\gamma B_1}{2\sqrt2}e^{-i\varphi}&0\\ \frac{\gamma B_1}{2\sqrt2}e^{i\varphi}&0&\frac{\gamma B_1}{2\sqrt2}e^{-i\varphi}\\ 0&\frac{\gamma B_1}{2\sqrt2}e^{i\varphi}&-A_Nm\end{pmatrix}
$$

**Many sources.** Every secularly coupled spin adds one label and one term to the detuning:

$$
\tilde H=\sum_{m,c_1,c_2,\dots}\tilde H_{mc_1c_2\cdots}\otimes\ketbra{m}{m}\otimes\ketbra{c_1}{c_1}\otimes\ketbra{c_2}{c_2}\otimes\cdots,\qquad
\tilde H_{mc_1c_2\cdots}=
\begin{pmatrix}\Delta&\frac{\gamma B_1}{2\sqrt2}e^{-i\varphi}&0\\ \frac{\gamma B_1}{2\sqrt2}e^{i\varphi}&0&\frac{\gamma B_1}{2\sqrt2}e^{-i\varphi}\\ 0&\frac{\gamma B_1}{2\sqrt2}e^{i\varphi}&-\Delta\end{pmatrix}
$$

$$
\Delta=A_N\,m+A_{C_1}c_1+A_{C_2}c_2+\cdots,\qquad A_N/2\pi=2.16\ \text{MHz}\ (^{14}\text{N}),\quad A_{C_j}/2\pi\sim100\ \text{kHz}\ (^{13}\text{C})
$$

<p class="src">Doherty et al., Phys. Rep. 528, 1 (2013)</p>
