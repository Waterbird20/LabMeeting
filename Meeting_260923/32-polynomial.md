---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat method">Method</span> The sequence is a polynomial

<style scoped>
section { font-size: 20px; }
.columns { gap: 1.0rem; }
.columns .col:first-child { flex: 0 0 452px; }
</style>

<div class="columns">
<div class="col">

Split the half-powers off the wait, $W(\delta)=z^{-1/2}A(z)$ with $A(z)=\mathrm{diag}(1,z)$, so the $d$ of them make one phase:
$$
\begin{aligned}
U(\delta)&=z^{-d/2}\,\tilde U(z),\\
\tilde U(z)&={\color{#1f6feb}G_dA(z)}\,{\color{#b91c1c}G_{d-1}A(z)}\cdots{\color{#15803d}G_1A(z)}\,{\color{#b45309}G_0}.
\end{aligned}
$$
Each coloured block is one wait and then one pulse: a $2\times2$ matrix of degree-one polynomials in $z$. So the **degree is the number of waits**. Every $G_k$ and every $W$ is in $SU(2)$, so
$$
U(\delta)=\begin{pmatrix}\hat P & -\hat Q^{*}\\[2pt] \hat Q & \hat P^{*}\end{pmatrix},\ \ |\hat P|^{2}+|\hat Q|^{2}=1 .
$$

</div>
<div class="col">

<figure class="figure">

<video src="media/qsp-blocks.mp4" poster="media/qsp-blocks.png" width="640" controls autoplay loop muted playsinline preload="none"></video>

*Four pulses $R_y(\pi/2)$ and three waits, one coloured block at a time: every wait raises the degree by one, and the left column of $\tilde U(z)$ is $P(z)$ over $Q(z)$.*

</figure>

</div>
</div>

Here $\hat P=z^{-d/2}P(z)$ and $\hat Q=z^{-d/2}Q(z)$, with $\deg P,\deg Q\le d$. We prepare and read $\ket{0}$, so the experiment measures exactly $|\hat P(\delta)|^{2}$. **Designing the sequence is designing one polynomial.**
