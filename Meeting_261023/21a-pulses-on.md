---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

<style scoped>
.columns .col:first-child { flex: 0 0 400px; }
</style>

# <span class="cat method">Method</span> Pulse sequence

Pulses between the waits tune $P$ and $Q$; the degree stays $K$.

$$
u=R_x(\phi_N)\,R_z(\Delta\tau)\,R_x(\phi_{N-1})\cdots R_z(\Delta\tau)\,R_x(\phi_1)=\begin{pmatrix}P&-Q^*\\Q&P^*\end{pmatrix}
$$

<div class="columns">
<div class="col">

- $K$ waits and $N=K+1$ pulses about $x$.
- Each angle $\phi_k$ changes the polynomials, not their degree.
- All pulses off gives back $\mathrm{Re}\,P=T_K(x)$ and $Q=0$.

</div>
<div class="col">

<figure class="figure">

<video src="media/qsp-pulses-on.mp4" poster="media/qsp-pulses-on.png" width="660" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
</div>

<!--
Now insert pulses between the waits: N = K + 1 rotations about x, R_x(phi_1) to R_x(phi_N), with K = 3 waits in between. The clip starts with all angles at zero: that is the previous slide, Re P = T_3(x) and Q = 0.

Then we turn the angles up one after another, to pi/2, 2 pi/3, pi/3 and pi/4, and at the end we move phi_2 down and back. Every frame is computed from the product of the 2 x 2 matrices.

Watch two things. First, the curves change: the pulse angles choose what P and Q look like as functions of x. Second, they never get more wiggly than with three waits: Re P and Im Q stay polynomials of degree at most 3, Im P and Re Q are sin(theta) times polynomials of degree at most 2. The number of waits fixes the degree; the pulses only choose the coefficients.

Unitarity holds at every frame: |P|^2 + |Q|^2 = 1. So turning a pulse moves weight between P and Q.
-->
