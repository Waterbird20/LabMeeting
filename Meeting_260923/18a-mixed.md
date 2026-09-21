---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat intro">Intro</span> Mixed state

<div class="columns">
<div class="col">

Suppose our interest is the first qubit. Every measurement on it is $\operatorname{Tr}(\rho_1 O)$, with the other qubits traced out,

$$
\rho_1=\operatorname{Tr}_2\ket{\Phi^{+}}\bra{\Phi^{+}}
=\tfrac{1}{2}\big(\ket{0}\bra{0}+\ket{1}\bra{1}\big)=\tfrac{1}{2}I .
$$

That matrix has rank two, while a ket gives $\rho=\ket{\psi}\bra{\psi}$ of rank one; no ket describes the first qubit alone, so we call it **mixed**.

Every Pauli matrix is traceless, so all three expectation values vanish, $\langle\sigma_i\rangle=\operatorname{Tr}\big(\tfrac{1}{2}I\,\sigma_i\big)=\tfrac{1}{2}\operatorname{Tr}\sigma_i=0$, and the arrow sits at the centre, $|\vec r_1|=0$. The length is the purity, $\operatorname{Tr}\rho^2=\tfrac{1}{2}(1+|\vec r|^2)$: $1$ for a ket on the surface, $1/2$ at the centre.

</div>
<div class="col">

<figure class="figure">

<video src="media/bell-mixed.mp4" poster="media/bell-mixed.png" width="550" autoplay loop muted playsinline preload="none"></video>

*The reduced Bloch vector of qubit $1$: from $\ket{0}$ to $\ket{+}$ under the Hadamard, then to the origin during the CNOT, with $\langle X\rangle$, $\langle Y\rangle$, $\langle Z\rangle$ read off the partial trace.*

</figure>

</div>
</div>

<div class="callout">

Entanglement with the outside looks like mixing inside our qubit: the more entangled, the shorter the Bloch vector.

</div>
