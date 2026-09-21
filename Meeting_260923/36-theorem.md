---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

# <span class="cat method">Method</span> The QSP theorem: the reachable responses are known exactly

<div class="callout">

**Theorem.** With $d$ waits and $d+1$ pulses the sequence is always $U=\begin{pmatrix}\hat P&-\hat Q^{*}\\ \hat Q&\hat P^{*}\end{pmatrix}$ with $\deg P,\deg Q\le d$ and $|\hat P|^{2}+|\hat Q|^{2}=1$ on the circle, and conversely **every** such pair is realized by some choice of pulses [2, Thm. 3]. The textbook $W_x$ form is the restricted version, with real coefficients and definite parity: $\deg P\le d,\ \deg Q\le d-1$, parities $d$ and $d-1$, and $|P|^{2}+(1-a^{2})|Q|^{2}=1$ [1, Thm. 1].

</div>

The forward direction is nothing but the convolution recursion: each step appends one tap and raises the degree by one, which is what flips the parity in the $W_x$ form, while unitarity is inherited because every factor is unitary on the circle. The converse is the direction we use, because it says the design space is a **complete** list rather than a perturbative family: any bounded polynomial of degree $d$ is a sequence we can actually build, so a polynomial that vanishes exactly at three chosen detunings is a gate that is exactly blind to three spectral lines.
