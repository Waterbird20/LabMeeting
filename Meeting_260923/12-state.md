---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

# <span class="cat intro">Intro</span> State preparation

<div class="columns">
<div class="col">

A state of a $d$-level system is a ket, a column of complex amplitudes in a basis $\{\ket{j}\}$:

$$
\ket{\psi}=\sum_{j=0}^{d-1}c_j\ket{j}=
\begin{pmatrix}
c_0 \\ c_1 \\ \vdots \\ c_{d-1}
\end{pmatrix}
\qquad c_j\in\mathbb{C}.
$$

The bra is its Hermitian conjugate, the row vector of conjugated amplitudes $\bra{\psi}=(c_0^{*},\dots,c_{d-1}^{*})$, and the norm is the bra taken against the ket,

$$
\braket{\psi|\psi}=\sum_j|c_j|^2=1,
$$

which is the normalization condition.

</div>
<div class="col">

Every quantity we can actually observe is a bra times a ket: an expectation value $\braket{\psi|O|\psi}$, or an overlap $|\braket{\phi|\psi}|^2$.

Multiply the ket by a phase, $\ket{\psi}\to e^{i\alpha}\ket{\psi}$, and the bra picks up the conjugate, $\bra{\psi}\to e^{-i\alpha}\bra{\psi}$, so every such product is unchanged:

$$
\bra{\psi}e^{-i\alpha}\,O\,e^{i\alpha}\ket{\psi}=\braket{\psi|O|\psi},
\qquad
\big|\bra{\phi}e^{i\alpha}\ket{\psi}\big|^2=|\braket{\phi|\psi}|^2 .
$$

Therefore we lose no generality by taking the first amplitude $c_0$ to be real and non-negative.

So a $d$-level state needs $2d-2$ real numbers, $2d$ from the amplitudes minus the norm and the phase. A qubit is left with **two**, which is why the next slides draw it as a point on a sphere.

</div>
</div>
