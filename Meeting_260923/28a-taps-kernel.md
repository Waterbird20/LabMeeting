---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
section { font-size: 20px; }
p { margin: 0.25em 0; }
figure.figure { margin: 0.1em auto; }
</style>

# <span class="cat method">Method</span> Kernel function

A sequence is $N$ equal units of length $\tau$. Unit $k$ gives the system one small operation of amplitude $w_k$: a pulse area, or an RF amplitude. Between the operations the system precesses at the swept variable $\omega$, so the operation of unit $k$ arrives with the phase $e^{ik\omega\tau}$. To first order the operations add, so the total amplitude is the Fourier series of the amplitude list, the polynomial of the dice slides with $z=e^{i\omega\tau}$:

$$
A(\omega)=\sum_{k=0}^{N-1} w_k\,e^{ik\omega\tau},
\qquad
K(\omega)=\frac{|A(\omega)|^{2}}{|A(0)|^{2}} .
$$

Here $A(\omega)$ is the amplitude the whole sequence drives at an offset $\omega$ from resonance, and $|A|^{2}$ is the population it transfers there. Normalised by its resonant value, $K$ is the **kernel**, or the filter function: what the sequence transfers at an offset $\omega$ from a line. For equal amplitudes the sum is geometric, $\sum_k e^{ik\theta}=e^{i(N-1)\theta/2}\sin(N\theta/2)/\sin(\theta/2)$, so $K=|\mathcal{D}_N(\omega\tau)|^{2}$, the Dirichlet kernel.

<figure class="figure">

![w:660](media/taps-kernel.png)

</figure>
