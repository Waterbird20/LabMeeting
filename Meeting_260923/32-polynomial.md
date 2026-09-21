---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

# <span class="cat method">Method</span> Every factor is degree one, so the sequence is a polynomial

Split the half-powers off the wait, $W(\delta)=z^{-1/2}A(z)$ with $A(z)=\mathrm{diag}(1,z)$, and collect the $d$ of them into one overall phase:
$$
U(\delta)=z^{-d/2}\,\tilde U(z),\qquad
\tilde U(z)=G_d\,A(z)\,G_{d-1}\,A(z)\cdots A(z)\,G_0 .
$$
Each factor $G_k A(z)$ is a $2\times2$ matrix whose entries are **degree-one polynomials in $z$**, so the product is a matrix polynomial of degree $d$, and $d$ is simply the number of waits. Since every $G_k$ and every $W$ has unit determinant, $U(\delta)\in SU(2)$ and its first column is a unit vector:
$$
U(\delta)=\begin{pmatrix}\hat P(\delta) & -\hat Q(\delta)^{*}\\[2pt] \hat Q(\delta) & \hat P(\delta)^{*}\end{pmatrix},
\qquad |\hat P|^{2}+|\hat Q|^{2}=1 ,
$$
with $\hat P=z^{-d/2}P(z)$, $\hat Q=z^{-d/2}Q(z)$ and $\deg P,\deg Q\le d$. We prepare $\ket{0}$ and read $\ket{0}$, so the experiment measures exactly $|\hat P(\delta)|^{2}$, and the transfer to $\ket{1}$ is $|\hat Q(\delta)|^{2}$. **Designing the sequence means designing one polynomial.**
