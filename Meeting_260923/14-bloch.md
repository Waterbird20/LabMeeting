---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

# <span class="cat intro">Intro</span> Bloch sphere

<div class="columns">
<div class="col">

A qubit state is a vector in a two-dimensional complex space, $\ket{\psi}=c_0\ket{0}+c_1\ket{1}$ with $|c_0|^2+|c_1|^2=1$.

Take expectation value of pauli matrix, $\langle \sigma_i \rangle=\bra{\psi}\sigma_i\ket{\psi}$, so the three form the **Bloch vector**:

$$
\vec{r}=\big(\langle X \rangle,\langle Y\rangle,\langle Z\rangle\big).
$$

For example, $\ket{+}$, 

$$
\langle X\rangle=\tfrac12\begin{pmatrix}1&1\end{pmatrix}
\begin{pmatrix}0&1\\1&0\end{pmatrix}
\begin{pmatrix}1\\1\end{pmatrix}=1,
$$

with $\langle Y\rangle=\langle Z\rangle=0$, so $\ket{+}$ sits at $\vec{r}=(1,0,0)$.

</div>
<div class="col">

<figure class="figure">

<video src="media/bloch-plus.mp4" poster="media/bloch-plus.png" width="550" autoplay loop muted playsinline preload="none"></video>
<!-- TODO: Split the video. The $\ket{\psi}= \cos\frac{\theta}{2} \ket{0} + \sin\frac{\theta}{2}e^{i\phi}\ket{1} is for next slide -->
<!-- TODO: For the video, show full matrix multiplication. For both +-state and general form. -->
<!-- TODO: In representing Bloch sphere with vector, add elevation angle $\theta$ and azimutal angle $\phi$ in the plot.   -->

</figure>

</div>
</div>
