---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat strategy">Strategy</span> From a target to the pulses: first the partner

<div class="columns">
<div class="col">

Everything so far designed a polynomial. The bench needs the $2(d+1)$ numbers $\beta_k$ and $\varphi_k$, so the design is finished only once $P$ is turned back into pulses. Complete the matrix first: unitarity on the circle asks for a partner with $|Q|^{2}=1-|P|^{2}$, and the bound $|P|\le1$ is the only obstruction, because no entry of a unitary matrix can exceed one in modulus.

**Fejér and Riesz:** a trigonometric polynomial that is non-negative everywhere on the circle is the squared modulus of an ordinary polynomial of the same degree.

Since $S=1-|P|^{2}\ge0$ is exactly such a polynomial, the partner exists. To build it, note that $z^{d}S(z)$ has $2d$ roots which pair up as $(r,1/\bar r)$ because $S$ is real on the circle, and keeping one root of each pair fixes $Q$ [2].

</div>
<div class="col">

With $P$ and $Q$ in hand every tap of $\tilde U(z)=\sum_nC_nz^{n}$ is known. Its two ends are single terms, since only one choice of projector per wait reaches an extreme power:
$$
\begin{aligned}
C_d&=G_d\,\Pi_1G_{d-1}\Pi_1\cdots\Pi_1G_0,\\
C_0&=G_d\,\Pi_0G_{d-1}\Pi_0\cdots\Pi_0G_0 .
\end{aligned}
$$
Each $\Pi_b=\ket{b}\bra{b}$ cuts its product into scalars,
$$
C_d=\Big(\textstyle\prod_{k=1}^{d-1}\bra{1}G_k\ket{1}\Big)\,G_d\ket{1}\bra{1}G_0 ,
$$
so both ends have rank one, with column spaces $G_d\ket{1}$ and $G_d\ket{0}$. Unitarity makes those orthogonal: on the circle $\tilde U^{\dagger}=\sum_nC_n^{\dagger}z^{-n}$, so $\tilde U\tilde U^{\dagger}=I$ reads coefficient by coefficient as $\sum_nC_{n+m}C_n^{\dagger}=\delta_{m0}I$, whose top power $m=d$ is the single term $C_dC_0^{\dagger}=0$. **The last pulse is read off the two ends of the list.**

</div>
</div>
