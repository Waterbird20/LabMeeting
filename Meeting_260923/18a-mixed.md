---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
section { font-size: 20px; }
.columns { gap: 1.0rem; }
.columns .col:first-child { flex: 0 0 450px; }
</style>

# <span class="cat intro">Intro</span> Mixed state

<div class="columns">
<div class="col">

Suppose our interest is the first qubit. Every measurement on it is $\operatorname{Tr}(\rho_1 O)$, with the other qubit traced out,

$$
\rho_1=\operatorname{Tr}_2\ket{\Phi^{+}}\bra{\Phi^{+}}
=\tfrac{1}{2}\big(\ket{0}\bra{0}+\ket{1}\bra{1}\big)=\tfrac{1}{2}I .
$$

This has rank two, while a ket gives $\rho=\ket{\psi}\bra{\psi}$ of rank one: no ket describes the first qubit alone, so we call it **mixed**. Every Pauli matrix is traceless, so $\langle\sigma_i\rangle=\tfrac{1}{2}\operatorname{Tr}\sigma_i=0$ and the arrow sits at the centre, $|\vec r_1|=0$. The length is the purity, $\operatorname{Tr}\rho^2=\tfrac{1}{2}(1+|\vec r|^2)$: $1$ on the surface, $1/2$ at the centre.

**Entanglement with the outside looks like mixing inside our qubit: the more entangled, the shorter the Bloch vector.**

</div>
<div class="col">

<figure class="figure">

<video src="media/bell-mixed.mp4" poster="media/bell-mixed.png" width="700" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
</div>
