---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

<!-- EDIT-FORWARD: the reused ladder figure is drawn in the textbook W_x convention while the text is in the physical (detuning-as-signal) convention; redraw it in the physical convention if the aside feels like a detour. -->

# <span class="cat method">Method</span> The sequence

<div class="columns">
<div class="col">

While the drive is off, the only surviving term is the detuning. So a wait of length $\tau$ is a precession about the $z$ axis,
$$
W(\delta)=R_z(\delta\tau)=\begin{pmatrix} z^{-1/2} & 0\\ 0 & z^{1/2}\end{pmatrix},
\qquad z=e^{i\delta\tau}.
$$
This is the **signal**: the detuning $\delta$ is what the system hands us, and $W$ is the only place it enters. The microwave pulses $G_k$ are the **processing**, chosen freely through their area $\beta_k$ and carrier phase $\varphi_k$. With $d$ equal waits and $d+1$ pulses,
$$
U(\delta)=G_d\,W(\delta)\,G_{d-1}\cdots W(\delta)\,G_0 .
$$

</div>
<div class="col">

<figure class="figure">

![w:420](media/qsp_ladder.png)

*The textbook ladder of [1] puts the signal in an $x$-rotation $W_x(a)$ and the processing in $z$-rotations. It is the same sequence in a rotated basis, since $H\,W_x(a)\,H$ is our wait with $a=\cos(\delta\tau/2)$ and $H\,e^{i\phi_kZ}H=R_x(-2\phi_k)$ is our pulse.*

</figure>

</div>
</div>
