---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

# <span class="cat intro">Intro</span> Schrodinger equation

The state obeys the Schrodinger equation (with $\hbar=1$),

$$
\frac{d}{dt}\ket{\psi(t)}=-i\,H\,\ket{\psi(t)},
$$

where $H$ is the Hamiltonian, the matrix whose expectation value is the energy, and it is Hermitian, $H^{\dagger}=H$, because energies are real. If $H$ does not depend on time this is a linear equation with constant coefficients, so an exponential solves it,

$$
\ket{\psi(t)}=U(t)\ket{\psi(0)},
\qquad
U(t)=e^{-iHt}=\sum_{n=0}^{\infty} \frac{(-iHt)^{n}}{n!}.
$$

It is easy to see that $U^{\dagger}U=e^{+iH^{\dagger}t}e^{-iHt}=e^{+iHt}e^{-iHt}=I$, and the same in the other order. So the Schrodinger equation generates a unitary operation, and the norm of the state is preserved.
