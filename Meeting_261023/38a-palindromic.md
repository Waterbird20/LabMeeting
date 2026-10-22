---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

<style scoped>
mjx-container[display="true"] { margin: 0.2em 0 !important; }
</style>

# <span class="cat method">Method</span> Symmetric QSP

A palindromic sequence leaves one real function.

$$
u=G_1\,W\,G_2\cdots G_2\,W\,G_1,\qquad W^T=W,\quad R_x(\vartheta)^T=R_x(\vartheta)\quad\Rightarrow\quad u^T=u
$$

$$
\begin{pmatrix}P&-Q^*\\Q&P^*\end{pmatrix}=\begin{pmatrix}P&Q\\-Q^*&P^*\end{pmatrix}\ \Rightarrow\ Q=-Q^*=-i\,r,\qquad r(x)=\sum_{k\equiv K\,(2)}c_k\,T_k(x)\ \ \text{real}
$$

**Symmetric QSP**: the phase list is a palindrome, $(\phi_0,\phi_1,\dots,\phi_1,\phi_0)$. It was proposed for computing phase factors by Dong, Meng, Whaley and Lin (2021); Wang, Dong and Lin (2022) proved that every admissible polynomial pair has exactly one symmetric phase list, and that the cost is strongly convex near a fixed symmetric starting point when the target is small.

- Every $Q(\Delta_m)$ has the phase $\pm\pi/2$, so phase coherence is automatic.
- The target becomes three conditions on one real function $r$.
- With instantaneous pulses, symmetric and arbitrary sequences reach the same error.

<!--
Why can we work with one real function? The idea is symmetric quantum signal processing. In the QSP literature the pulses are written as a list of phase factors, and symmetric QSP restricts that list to a palindrome, phi_0, phi_1, ..., phi_1, phi_0. Dong, Meng, Whaley and Lin proposed it in 2021 as a way to compute phase factors by optimisation. Wang, Dong and Lin, in their 2022 paper on the energy landscape of symmetric QSP, showed two things. First, existence and uniqueness: for every admissible pair of polynomials, with the right degree, parity and normalisation, there is exactly one symmetric phase list that produces it. Second, the cost landscape: it has many global and local minima, yet optimisation from the fixed symmetric start (pi/4, 0, ..., 0, pi/4) robustly finds one particular solution, which they call the maximal solution; they prove that it lies close to that start and that the cost is strongly convex there when the target polynomial is small, of order one over the degree. Their convention has the signal as an x-rotation and the phases as z-rotations; ours is the same structure in the rotated basis, with waits about z and pulses about x, so a symmetric phase list is our palindromic pulse sequence.

Here is the mechanism in our variables. The free evolution W = diag(e^{-i theta}, e^{i theta}) is diagonal, so it equals its transpose; a rotation about x is cos(theta/2) - i sin(theta/2) X, symmetric because X is. The transpose of a product reverses the order, and for a palindromic sequence the reversed product is the same product: u^T = u. In Cayley–Klein form that says the two off-diagonal entries are equal, Q = -Q*, so Q is purely imaginary at every detuning: Q = -i r with r real. Pulses about x alone also give Q(-Delta) = -Q(Delta)*, so r is even in theta: a cosine series, a real Chebyshev series in x of degree K with the parity of K.

Two consequences. Phase coherence comes for free: every Q(Delta_m) is -i times a real number, so all three windows share the phase -pi/2 up to a sign, and only Q^2 matters for the DQ pi pulse. And the target becomes the three conditions on r we just saw.

Nothing is lost by the restriction. For every error level the set of admissible Q is convex and invariant under the two maps Q(theta) -> Q(-theta) and Q -> -conj(Q(-theta)); averaging an admissible Q over them gives an admissible palindromic one. A check against the general convex program for a complex Q agrees to 3e-4 relative at N = 12 and 14, and training confirms it: at N = 14 the best symmetric and the best arbitrary sequences both reach 5.15e-4.
-->
