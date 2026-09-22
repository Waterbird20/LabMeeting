---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<!-- TODO (speaker to confirm): page 41 of the first draft is now two slides, this one (the kernel,
     with the explicit R_x matrices) and 33a-recursion.md (the recursion, the k = 1 and
     k = 2 unrolling, and the clip). Merge them back if the section runs long. -->

# <span class="cat method">Method</span> One wait and one pulse make a kernel function

Let $\Pi_b=\ket{b}\bra{b}$ be the projector onto $\ket{b}$, so that $\Pi_0+\Pi_1=I$. During a wait only $\ket{1}$ collects the phase $\delta\tau$ relative to $\ket{0}$. So the wait leaves the $\ket{0}$ component where it is, and advances the $\ket{1}$ component by one power of $z$:
$$
A(z)=\mathrm{diag}(1,z)=\Pi_0+z\,\Pi_1 ,\qquad z=e^{i\delta\tau} .
$$
**One power of $z$ is one wait.** The pulse that follows is a fixed matrix while $z$ is a scalar, so it multiplies straight through both terms:
$$
G_k\,A(z)=K^{(0)}_k+z\,K^{(1)}_k ,\qquad K^{(0)}_k=G_k\,\Pi_0 ,\quad K^{(1)}_k=G_k\,\Pi_1 .
$$
For a pulse about $x$ of area $\beta_k$, writing $c=\cos\frac{\beta_k}{2}$ and $s=\sin\frac{\beta_k}{2}$,
$$
G_k=R_x(\beta_k)=\begin{pmatrix} c & -is\\ -is & c\end{pmatrix},\qquad
K^{(0)}_k=\begin{pmatrix} c & 0\\ -is & 0\end{pmatrix},\qquad
K^{(1)}_k=\begin{pmatrix} 0 & -is\\ 0 & c\end{pmatrix}.
$$
So each step of the sequence is a kernel function with two terms, and its coefficients are matrices. The term at $z^{0}$ sends the $\ket{0}$ amplitude through the pulse at the same power of $z$, and the term at $z^{1}$ sends the $\ket{1}$ amplitude through the pulse one power of $z$ later.
