---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

<style scoped>
mjx-container[display="true"] { margin: 0.2em 0 !important; }
p { margin: 0.2em 0; }
.columns .col:first-child { flex: 0 0 450px; }
</style>

# <span class="cat intro">Intro</span> DQ $\pi$ pulse and state preparation

Both DQ operations are conditions on the spin-½ amplitudes $P$ (stay) and $Q$ (flip).

<div class="columns">
<div class="col">

Spin-½, $\lvert P\rvert^2+\lvert Q\rvert^2=1$:

$$
u=\begin{pmatrix}P&-Q^*\\Q&P^*\end{pmatrix}.
$$

Spin-1: the triplet block of $u\otimes u$,

$$
D^1(u)=\begin{pmatrix}P^2&-\sqrt2PQ^*&Q^{*2}\\ \sqrt2PQ&\lvert P\rvert^2-\lvert Q\rvert^2&-\sqrt2P^*Q^*\\ Q^2&\sqrt2P^*Q&P^{*2}\end{pmatrix}.
$$

From $\ket{0}$, with $q=\lvert Q\rvert^2$: $P_0=(1-2q)^2$, $P_{\pm1}=2q(1-q)$.

</div>
<div class="col">

**DQ $\pi$ pulse**, $\ket{\pm1}\to\ket{\mp1}$: $P=0$, e.g. $R_x(\pi)$ with $Q=-i$,

$$
D^1(u)=\begin{pmatrix}0&0&Q^{*2}\\ 0&-1&0\\ Q^2&0&0\end{pmatrix},\qquad \text{only } Q^2 \text{ matters}.
$$

**DQ preparation**, $\ket{0}\to\tfrac{1}{\sqrt2}\big(\ket{{+1}}+\ket{{-1}}\big)$:

$$
D^1(u)\ket{0}=\begin{pmatrix}-\sqrt2\,PQ^*\\ \lvert P\rvert^2-\lvert Q\rvert^2\\ \sqrt2\,P^*Q\end{pmatrix},\qquad f=4\big[\mathrm{Im}(P^*Q)\big]^2 .
$$

- $\lvert P\rvert^2=\lvert Q\rvert^2=\tfrac12$ empties $\ket0$; $\mathrm{Re}(P^*Q)=0$ fixes the relative phase.
- E.g. $R_x(\tfrac\pi2)$ prepares $-\tfrac{i}{\sqrt2}\big(\ket{{+1}}+\ket{{-1}}\big)$.

</div>
</div>
