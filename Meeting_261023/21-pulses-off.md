---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

<style scoped>
.columns .col:first-child { flex: 0 0 470px; }
</style>

# <span class="cat method">Method</span> Free evolution

Without pulses, $K$ waits make one rotation $R_z(K\Delta\tau)$.

<div class="columns">
<div class="col">

$$
\underbrace{R_z(\Delta\tau)\cdots R_z(\Delta\tau)}_{K\ \text{waits}}=R_z(K\Delta\tau)=\begin{pmatrix}z^{-K}&0\\0&z^{K}\end{pmatrix}
$$
$$
P=z^{-K}=e^{-iK\theta},\qquad Q=0
$$
$$
\begin{aligned}
\mathrm{Re}\,P&=\cos K\theta=T_K(x)\\
\mathrm{Im}\,P&=-\sin K\theta=-\sin\theta\,U_{K-1}(x)
\end{aligned}
$$

- $\theta=\frac{\Delta\tau}{2}$, $z=e^{i\theta}$, $x=\cos\theta$.
- $T_K$ and $U_{K-1}$ are polynomials of degree $K$ and $K-1$ in $x$.

</div>
<div class="col">

<figure class="figure">

<video src="media/qsp-pulses-off.mp4" poster="media/qsp-pulses-off.png" width="680" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
</div>

<!--
Start with the simplest sequence: no pulses at all, only K waits of length tau.

During a wait only the detuning acts, so each wait is a rotation about z by the angle Delta tau: R_z(Delta tau) = exp(-i Delta tau sigma_z / 2). The clip draws them one after the other: R_z R_z R_z. Rotations about one axis add, so K waits are one rotation R_z(K Delta tau). In the basis up, down it is diagonal with entries z^{-K} and z^{K}, where z = e^{i theta} and theta = Delta tau / 2 is the half angle.

So the stay amplitude is P = z^{-K} = e^{-i K theta}, and the flip amplitude is Q = 0.

The animation: first the K blocks R_z(Delta tau) merge into one block R_z(K Delta tau). Once K is fixed, we plot the real and the imaginary part of P as functions of x = cos(Delta tau / 2). As Delta tau goes from 0 to 2 pi, x runs from 1 to -1. First K = 3, then K = 5.

The real part is cos(K theta). As a function of x = cos theta this is a polynomial, the Chebyshev polynomial T_K(x): for K = 3 it is 4x^3 - 3x, for K = 5 it is 16x^5 - 20x^3 + 5x.

The imaginary part is -sin(K theta) = -sin(theta) U_{K-1}(x), with the Chebyshev polynomial of the second kind U_{K-1}, of degree K - 1: for K = 3, U_2 = 4x^2 - 1.

So P is a polynomial of order K in x = cos(Delta tau / 2), plus sin(theta) times a polynomial of order K - 1. The order equals the number of waits. And x only sees cos(Delta tau / 2): a detuning and its negative give the same x.
-->
