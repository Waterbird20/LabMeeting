---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<!-- TODO (speaker to confirm): $H_0$ is written in the spin-1 form ($S_z^2$, $S_z$) because that is the
     honest NV ground-state Hamiltonian, while the rest of chapter 1 works at the qubit level
     ($Y$, $Z$). If the speaker prefers one convention throughout, replace this block with the
     rotating-frame qubit form $H=\tfrac{\delta}{2}Z+\tfrac{\Omega(t)}{2}Y$. -->

# <span class="cat method">Method</span> Hamiltonian and unitary operation

To implement the gate operation we need, we have to engineer the Hamiltonian. The Hamiltonian is always drift plus control, $H(t)=H_0+H_c(t)$: what the system does on its own, plus the pulse we add.

For example, the NV centre has its own Hamiltonian, and we can apply a microwave pulse to it, which is effectively an oscillating magnetic field:

$$
\begin{aligned}
H_0&=D\,S_z^{2}+\gamma B_{\parallel}S_z,\\
H_c(t)&=\Omega(t)\cos(\omega t+\varphi)\,S_x .
\end{aligned}
$$

Here $D$ is the zero-field splitting, $\gamma B_\parallel$ the Zeeman shift, $\Omega(t)$ the drive amplitude, $\omega$ its carrier and $\varphi$ its phase, which in the rotating frame selects the rotation axis; the sequences later use $\varphi=\pi/2$, a $y$ drive.

Now $H(t_1)$ and $H(t_2)$ no longer commute, and the time-dependent Schrodinger equation has no closed form in general; solving it efficiently is a challenge of its own and out of scope for today. Briefly, one can step it forward like the Euler method,

$$
\ket{\psi(t+dt)} = \ket{\psi(t)} -i\, H(t) \ket{\psi(t)}\,dt ,
$$

and the product of many such small steps is the unitary $U$.
