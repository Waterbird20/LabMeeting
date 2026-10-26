---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
.columns { align-items: center; margin-top: 0.2em; }
.columns .col:last-child { flex: 0 0 360px; }
mjx-container[display="true"] { margin: 0.3em 0 !important; }
figure.figure { margin: 0; }
</style>

# <span class="cat intro">Intro</span> Universal NOT

The universal NOT flips every qubit state to its orthogonal state.

<div class="columns">
<div class="col">

**Ideal:** $\ket{\psi}\mapsto\ket{\psi^\perp}$ for every $\ket\psi$, i.e. $\mathbf r\mapsto-\mathbf r$ on the Bloch sphere.

- Inverting every Bloch vector is a point reflection: antiunitary, so no physical operation does it.

**Optimal:** the best channel for all inputs at once,

$$
\rho\ \mapsto\ \frac13\sum_{i=x,y,z}\sigma_i\,\rho\,\sigma_i=\frac{2\,\mathbb 1-\rho}{3},\qquad \mathbf r\mapsto-\frac{\mathbf r}{3},\qquad F=\frac23 .
$$

- Three $\pi$ rotations about orthogonal axes, applied at random with equal weights.
- The unpolarised ¹⁴N supplies that random choice: one rotation per hyperfine line.

</div>
<div class="col">

<figure class="figure">

![w:350](media/ch3-unot-definition.png)

</figure>

</div>
</div>

<p class="src">Bužek, Hillery, Werner, PRA 60, R2626 (1999); De Martini, Bužek, Sciarrino, Sias, Nature 419, 815 (2002)</p>
