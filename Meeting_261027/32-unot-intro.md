---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
.columns { margin-top: 0; align-items: center; }
.columns .col:first-child { flex: 0 0 575px; }
figure.figure { margin: 0; }
mjx-container[display="true"] { margin: 0.25em 0 !important; }
ul { margin: 0.2em 0 0; }
</style>

# <span class="cat intro">Intro</span> Universal NOT: goal

One pulse sequence on the qubit $\{\ket{0},\ket{-1}\}$ should apply a different $\pi$ rotation on each ¹⁴N line.

$$
m_I=-1:\ R_x(\pi),\qquad m_I=0:\ R_y(\pi),\qquad m_I=+1:\ R_z(\pi)\qquad\Longrightarrow\qquad \frac13\sum_{m_I}u_m\,\rho\,u_m^\dagger:\ \ \mathbf r\mapsto-\tfrac13\,\mathbf r
$$

<div class="columns">
<div class="col">

<figure class="figure">

<video src="media/ch3-unot-channel.mp4" poster="media/ch3-unot-channel.png" width="575" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
<div class="col">

- Every input ends at $F=2/3$ with its orthogonal state, the optimum.
- One $\pi$ pulse gives nearly the same rotation on all three lines.

</div>
</div>

<p class="src">Bužek, Hillery, Werner, PRA 60, R2626 (1999)</p>
