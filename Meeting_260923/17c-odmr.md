---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

<!-- TODO (speaker to confirm): this is pulsed ODMR, one fixed $\pi$ pulse per point. The thesis measures continuous-wave ESR, laser and microwaves on together, whose lineshape is a Lorentzian rather than the sinc-like curve above; say which one the lab runs. -->

# <span class="cat intro">Intro</span> ODMR

Fix the pulse at the calibrated length $t=\pi/\Omega$ and sweep the carrier instead. So on resonance, $\delta=0$, we have $\Omega t=\pi$, a $\pi$ pulse. Off resonance, the rotation axis $\hat n=(0,\Omega,\delta)/\Omega_R$ tilts toward $z$ and the rotation is faster:

$$
P(0)=1-\frac{\Omega^{2}}{\Omega^{2}+\delta^{2}}\,
\sin^{2}\!\Big(\frac{\Omega_R t}{2}\Big).
$$

The resulting signal is approximately a sinc function.

<figure class="figure">

<video src="media/seq-odmr.mp4" poster="media/seq-odmr.png" width="520" autoplay loop muted playsinline preload="none"></video>

</figure>
