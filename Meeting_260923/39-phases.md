---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

# <span class="cat strategy">Strategy</span> Finding the pulses is deconvolution

Given a target $P$ with $|P|\le1$ on the circle, a partner $Q$ with $|Q|^{2}=1-|P|^{2}$ exists by the Fejér–Riesz factorization, and the pulses are then peeled off one at a time [2]. The two end taps of the matrix list are
$$
C_d=G_d\,\Pi_1 G_{d-1}\Pi_1\cdots\Pi_1 G_0 ,\qquad
C_0=G_d\,\Pi_0 G_{d-1}\Pi_0\cdots\Pi_0 G_0 ,
$$
each containing a projector, hence each of rank one, and unitarity on the circle forces $C_dC_0^{\dagger}=0$. Their column directions are therefore the two orthogonal columns $G_d\ket{1}$ and $G_d\ket{0}$: **the last pulse is read directly off the two ends of the tap list.** Removing it, $A(z)^{-1}G_d^{\dagger}\tilde U$ has degree $d-1$, and $d$ repetitions give every pulse. Dividing out one two-tap factor per step is the exact inverse of the step-by-step convolution that built the sequence.

Two practical alternatives exist. Optimization over the phases is what `pyqsp` and QSPPACK do and what our own fits of 2026-08-18 did. For a **gate** target the whole design is even convex: maximizing the smallest projection of $P$ on the target phase subject to $|P|\le1$ is a linear program with a global optimum.
