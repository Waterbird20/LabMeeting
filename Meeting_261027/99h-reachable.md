---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
.columns .col:first-child { flex: 0 0 640px; }
.columns { gap: 1.0rem; }
</style>

# <span class="cat method">Method</span> Reachable polynomials

$K$ waits and $K+1$ pulses about $x$ make exactly these polynomials:

$$
u=A\,\mathbb I-iB\,X-i\sin\theta\,(C\,Z+D\,Y),\qquad W=x\,\mathbb I-i\sin\theta\,Z,\qquad R_x(\phi)=\cos\tfrac\phi2\,\mathbb I-i\sin\tfrac\phi2\,X
$$

<div class="columns">
<div class="col">

<figure class="figure">

<video src="media/qsp-pauli-build.mp4" poster="media/qsp-pauli-build.png" width="630" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
<div class="col">

| real polynomials | degree | parity |
|---|---|---|
| $A,\ B$ | $\le K$ | of $K$ |
| $C,\ D$ | $\le K-1$ | of $K-1$ |

$$
A^2+B^2+(1-x^2)(C^2+D^2)=1
$$
$$
P=A-i\sin\theta\,C,\quad Q=-iB+\sin\theta\,D
$$

Conversely, every such $A,B,C,D$ is made by $K+1$ pulses about $x$:
$$
\underbrace{2(K+1)}_{\text{coefficients}}-\underbrace{(K+1)}_{\text{unitarity}}=\underbrace{K+1}_{\text{angles}}
$$

<p class="src">Low, Yoder, Chuang, PRX 6, 041067 (2016); Martyn, Rossi, Tan, Chuang, PRX Quantum 2, 040203 (2021), Thm. 1</p>

</div>
</div>
