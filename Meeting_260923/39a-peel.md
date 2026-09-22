---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
section p { margin: 0.42em 0; }
</style>

# <span class="cat strategy">Strategy</span> Peeling the pulses off is deconvolution

<div class="columns">
<div class="col">

Now strip $G_d$ away. In $G_d^{\dagger}\tilde U(z)=A(z)G_{d-1}\cdots A(z)G_0$ the top tap carries $\Pi_1$ on the left and the bottom tap carries $\Pi_0$, so the $\bra{0}$ row has no $z^{d}$ and the $\bra{1}$ row no $z^{0}$. Dividing that row by $z$ is legal, and
$$
A(z)^{-1}G_d^{\dagger}\tilde U(z)=G_{d-1}A(z)\cdots A(z)G_0
$$
is the same object with one wait fewer, so $d$ steps leave $G_0$. This is deconvolution: one two-tap kernel $G_k\Pi_0+zG_k\Pi_1$ per step built the sequence, and the recursion divides one out per step, in reverse order.

</div>
<div class="col">

**One wait, two pulses.** With $G_0=R_x(\pi/3)$ and $G_1=R_x(\pi/2)$, $\tilde U(z)=C_0+zC_1$ where
$$
C_0=\frac{1}{4}\begin{pmatrix}\sqrt6&-i\sqrt2\\-i\sqrt6&-\sqrt2\end{pmatrix},\ \
C_1=\frac{1}{4}\begin{pmatrix}-\sqrt2&-i\sqrt6\\-i\sqrt2&\sqrt6\end{pmatrix}.
$$
Both determinants vanish and $C_1C_0^{\dagger}=0$ exactly. The column directions $(1,-i)/\sqrt2$ and $(1,i)/\sqrt2$ are orthogonal as promised, and as the two columns of one $SU(2)$ matrix they rebuild $G_1=R_x(\pi/2)$, up to one phase per column that is a harmless virtual $z$-rotation. One division then returns $G_0$, and the same recursion peels a random $d=3$ sequence back to $10^{-15}$.

</div>
</div>

Optimization over the areas and phases skips the factorization altogether, the route of `pyqsp` and QSPPACK and of our own fits of 2026-08-18, at the price of local minima. For a **gate** target the design is instead a linear program in $P$ with a global optimum, whose pulses this same peel-off returns [3].
