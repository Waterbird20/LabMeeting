---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
.columns { margin-top: 0; align-items: flex-start; }
.columns .col:first-child { flex: 0 0 580px; }
figure.figure { margin: 0; }
mjx-container[display="true"] { margin: 0.2em 0 !important; }
ul { margin-top: 0.1em; }
</style>

# <span class="cat method">Method</span> Universal NOT: Newton solver

Damped Newton iteration solves the nine equations $r(\mathbf p)=0$ from random starts, with an exact Jacobian.

$$
\mathbf p\ \leftarrow\ \mathbf p-\big(J^TJ+\lambda\,\mathbb 1\big)^{-1}J^T r,\qquad\quad \frac{\partial u}{\partial p_k}=U_N\cdots U_{k+1}\,\frac{\partial U_k}{\partial p_k}\,U_{k-1}\cdots U_1
$$

<div class="columns">
<div class="col">

<figure class="figure">

![h:300](media/ch3-unot-newton-cost.png)

</figure>

- Most starts converge in about 20 steps; a few stall.

</div>
<div class="col">

<figure class="figure">

![h:300](media/ch3-unot-chebyshev.png)

</figure>

- Only odd $k\le3$ appear, as degree $3$ requires.

</div>
</div>

<p class="src">Newton iteration for symmetric QSP: Dong, Lin, Ni, Wang, SIAM J. Sci. Comput. 46, A2951 (2024)</p>
