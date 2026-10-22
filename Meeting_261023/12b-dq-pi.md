---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

# <span class="cat intro">Intro</span> DQ $\pi$ pulse

The DQ $\pi$ pulse is $P=0$ on the spin-½, and only $Q^2$ matters.

<div class="columns">
<div class="col">

Spin-½: $P$ = stay, $Q$ = flip, $\lvert P\rvert^2+\lvert Q\rvert^2=1$,

$$
u=\begin{pmatrix}P&-Q^*\\Q&P^*\end{pmatrix}.
$$

Spin-1: the triplet block of $u\otimes u$,

$$
D^1(u)=\begin{pmatrix}P^2&-\sqrt2PQ^*&Q^{*2}\\ \sqrt2PQ&\lvert P\rvert^2-\lvert Q\rvert^2&-\sqrt2P^*Q^*\\ Q^2&\sqrt2P^*Q&P^{*2}\end{pmatrix}.
$$

</div>
<div class="col">

DQ $\pi$ pulse, $P=0$:

$$
D^1(u)=\begin{pmatrix}0&0&Q^{*2}\\ 0&-1&0\\ Q^2&0&0\end{pmatrix}.
$$

From $\ket{0}$, with $q=\lvert Q\rvert^2$:

$$
P_0=(1-2q)^2,\qquad P_{\pm1}=2q(1-q).
$$

- Both spins flip: $\ket{{+1}}\to Q^2\ket{{-1}}$.
- $D^1(-u)=D^1(u)$, so only $Q^2$ matters.

</div>
</div>

<!--
The DQ pi pulse in the spin-1/2 language.

Write the spin-1/2 evolution with its Cayley-Klein parameters: P is the amplitude for the spin-1/2 to stay, Q the amplitude to flip, and |P|^2 + |Q|^2 = 1.

D^1(u) is the triplet block of u (x) u, so every entry is a product of two spin-1/2 amplitudes. For example, the amplitude from |+1> = up-up to |-1> = down-down is Q times Q: both particles must flip. The amplitude to stay in |+1> is P times P. The middle entry, from |0> to |0>, is |P|^2 - |Q|^2.

The DQ pi pulse exchanges |+1> and |-1>. That needs P = 0: every entry containing P vanishes, |+1> -> Q^2 |-1>, |-1> -> Q*^2 |+1>, and |0> -> -|0>. So the condition on the spin-1/2 is simply P = 0.

All populations are polynomials in q = |Q|^2. From |0>: P_0 = (1 - 2q)^2 and P_(+-1) = 2q(1 - q). We will use this on the next slides: the |0> population tells us q.

Finally D^1(-u) = D^1(u): u and -u are the same rotation, so the overall sign of u is invisible. Only Q^2 matters for a DQ target, and in particular the sign of Q is free.
-->
