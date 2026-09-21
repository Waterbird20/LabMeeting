---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

# <span class="cat intro">Intro</span> Expectation value

An observable is a Hermitian matrix $O=O^{\dagger}$, and the number the experiment reports is
$\langle O\rangle_{\psi}=\braket{\psi|O|\psi}=\sum_{j,k}c_j^{*}O_{jk}c_k$, a row vector times a
matrix times a column vector, which Hermiticity makes real.

For a single qubit the observables of interest are the Pauli matrices

$$
X=\begin{pmatrix}0&1\\1&0\end{pmatrix},\quad
Y=\begin{pmatrix}0&-i\\i&0\end{pmatrix},\quad
Z=\begin{pmatrix}1&0\\0&-1\end{pmatrix},
$$

and we write the row, the matrix and the column, then multiply.

With $\ket{+}=\tfrac{1}{\sqrt2} \begin{pmatrix} 1 \\ 1 \end{pmatrix}$ this gives
$\braket{+|X|+}=\tfrac12(1,1)\,\begin{pmatrix}0&1\\1&0\end{pmatrix}\,\begin{pmatrix} 1 \\ 1 \end{pmatrix}=1$.
