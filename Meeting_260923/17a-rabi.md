---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

<!-- EDIT-FORWARD: confirm the read-out convention. These slides plot $P(0)$, the population of $\ket{0}$, which rises with fluorescence; the thesis plots normalised fluorescence contrast, so its minima are the $\pi$ pulses just as they are here. -->

# <span class="cat intro">Intro</span> Rabi: calibrating the rotation angle

Setting pulse on resonance, $\delta=0$, so $H=\frac{\Omega}{2}X$ and the pulse is a rotation about $y$ by the angle $\Omega t$:

$$
P(0)=\big|\bra{0}e^{-i\Omega X t/2}\ket{0}\big|^{2} =\cos^{2}\!\Big(\frac{\Omega t}{2}\Big).
$$

Sweep the pulse length and the fluorescence oscillates at $\Omega$. 

<figure class="figure">

<video src="media/seq-rabi.mp4" poster="media/seq-rabi.png" width="520" autoplay loop muted playsinline preload="none"></video>

</figure>

<!-- TODO: Fix animation. x-axis should be lied on -y axis in current animation. As the other animation did, show measurement explicitly, which show z-projection. Then draw dot on the plot. -->
<!-- TODO: Draw angle $\Omega t$ in the plot -->