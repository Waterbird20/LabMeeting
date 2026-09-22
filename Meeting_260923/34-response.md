---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat method">Method</span> The response is the Fourier transform of the taps

<div class="columns">
<div class="col">

Keep only the $\ket{0}$ component of the list. Writing $p_n$ for the upper entry of $v_n$,
$$
P(z)=\bra{0}\tilde U\ket{0}=\sum_{n=0}^{d}p_n\,z^{n} .
$$
The experiment does not hand us a free $z$. The sweep puts $z$ on the unit circle, $z=e^{i\delta\tau}$, so restoring the phase $z^{-d/2}$ split off the waits,
$$
\hat P(\delta)=z^{-d/2}P(z)=\sum_{n=0}^{d}p_n\,e^{i\left(n-\frac{d}{2}\right)\delta\tau} ,
$$
a **Fourier series in the detuning** whose coefficients are the taps. Each tap is one harmonic of the wait: an amplitude that has waited $n$ times carries the phase $n\delta\tau$, so $p_n$ oscillates in $\delta$ at the frequency $n\tau$.

</div>
<div class="col">

The measurement squares it, and $|\hat P|^{2}=\hat P\hat P^{*}$ is a product of two such series. By the convolution theorem of chapter 2 a product of series is a convolution of lists, so
$$
|\hat P(\delta)|^{2}=\sum_{m=-d}^{d}r_m\,e^{im\delta\tau},
\qquad r_m=\sum_n p_{n+m}\,p_n^{*} ,
$$
and the measured curve is the transform of the **autocorrelation** of the tap list.

For example, take $d=1$, two taps. Then
$$
|\hat P|^{2}=|p_0|^{2}+|p_1|^{2}+2\,\mathrm{Re}\!\left(p_1p_0^{*}\,e^{i\delta\tau}\right) ,
$$
a constant plus one cosine. Two $\pi/2$ pulses of opposite sign around one wait give $p_0=p_1=\tfrac12$ and $\tfrac12\left(1+\cos\delta\tau\right)$, the Ramsey fringe of chapter 1.

</div>
</div>
