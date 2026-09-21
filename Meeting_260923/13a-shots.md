---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

# <span class="cat intro">Intro</span> One shot gives an eigenvalue, many shots give the average

<div class="columns">
<div class="col">

The trace is cyclic, so the same number can be written with the state moved to the outside:

$$
\braket{\psi|O|\psi}=\operatorname{Tr}\big(\ket{\psi}\bra{\psi}\,O\big)=\operatorname{Tr}(\rho\,O),
\qquad \rho=\ket{\psi}\bra{\psi}.
$$

The object $\rho$ is the **density matrix**, which is what one half of an entangled pair becomes later in this chapter.

</div>
<div class="col">

A single shot does not return the average. Diagonalise the observable, $O=\sum_k\lambda_k\ket{k}\bra{k}$. One shot returns **one eigenvalue** $\lambda_k$ with the Born probability $p_k=|\braket{k|\psi}|^2$, and it leaves the system in the corresponding eigenvector $\ket{k}$, which is the collapse.

Averaging those outcomes gives $\sum_k\lambda_k p_k=\braket{\psi|O|\psi}$: the expectation value **is** the shot-averaged outcome, reached with an error of order $1/\sqrt{N}$ after $N$ shots. For $Z$ the eigenvalues are $\pm1$ and $\langle Z\rangle=p_0-p_1$.

</div>
</div>
