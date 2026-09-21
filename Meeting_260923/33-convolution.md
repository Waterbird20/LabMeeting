---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<!-- EDIT-FORWARD: the clip writes the two components of v_n as p_n and u_n with <1|U|0> = -i sum u_n z^n, so that an R_x(beta) pulse gives the real two-tap kernel (c, -s; s, c) with c = cos(beta/2), s = sin(beta/2). That is the same recursion as the boxed one; say which notation you want on the slide. -->

# <span class="cat method">Method</span> Each added step convolves the coefficient list

<div class="columns">
<div class="col">

The wait leaves the $\ket{0}$ component alone and advances the $\ket{1}$ component by one power of $z$, so $A(z)=\Pi_0+z\,\Pi_1$ with $\Pi_b=\ket{b}\bra{b}$, and one step is a **two-tap kernel**
$$
G_k A(z)=K^{(0)}_k+z\,K^{(1)}_k ,\qquad K^{(b)}_k=G_k\,\Pi_b .
$$
Write the state after $k$ steps as a list of $2$-vectors, $\tilde U_k\ket{0}=\sum_n v^{(k)}_n z^n$. Multiplying by the new factor gives
$$
v^{(k)}_n=K^{(0)}_k v^{(k-1)}_n+K^{(1)}_k v^{(k-1)}_{n-1}\equiv\big(K_k * v^{(k-1)}\big)_n .
$$
This is the identity of Section 2 with matrix-valued taps: a product of polynomials is a convolution of coefficients. Starting from one tap $v^{(0)}_0=G_0\ket{0}$, each step lengthens the list by exactly one, which is why **degree $=$ number of waits**.

</div>
<div class="col">

<figure class="figure">

<video src="media/qsp-coeff-conv.mp4" poster="media/qsp-coeff-conv.png" width="500" autoplay loop muted playsinline preload="none"></video>

*Left: the two coefficient lists of $\langle0|U|0\rangle$ and $\langle1|U|0\rangle$, each gaining one tap per wait and pulse. Right: the response $|\langle0|U|0\rangle|^{2}$ over $\delta\tau$, which sharpens with the degree $d$ and changes shape when the pulse areas are windowed.*

</figure>

</div>
</div>
