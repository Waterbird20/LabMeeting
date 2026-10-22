---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

<style scoped>
.columns .col:first-child { flex: 0 0 640px; }
figure.figure { margin: 0; }
</style>

# <span class="cat method">Method</span> Finite pulses

A real pulse depends on the detuning: its axis tilts out of the equator.

$$
U=\exp\Big[-\tfrac{i}{2}\,t\,\big(\Delta\sigma_z+\Omega(\cos\varphi\,\sigma_x+\sin\varphi\,\sigma_y)\big)\Big],\qquad \text{axis tilted by }\arctan(\Delta/\Omega)
$$

<div class="columns">
<div class="col">

<figure class="figure">

<video src="media/real-pulse-bloch.mp4" poster="media/real-pulse-bloch.png" width="630" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
<div class="col">

A finite pulse acts like an instantaneous pulse between two short free evolutions:

$$
e_k=\frac{\tan(\beta_k/2)}{\Omega_k},\qquad g_j=\frac{2\pi}{A}-e_j-e_{j+1}
$$

- Inside $1.5A$, $\lvert\Delta\rvert/\Omega_{\max}$ reaches $0.46$ at $10\,\text{MHz}$.
- $\frac\pi2-\big(\frac{2\pi}A-\frac2{\Omega_{\max}}\big)-\frac\pi2$ is exact at the three resonances to $1.4\times10^{-6}$ (drive of Mamin et al.).

</div>
</div>

<!--
So far the pulses were instantaneous. A real rectangular pulse of amplitude Omega and phase phi lasting t is U = exp[-(i/2) t (Delta sigma_z + Omega(cos phi sigma_x + sin phi sigma_y))]: a rotation by t sqrt(Omega^2 + Delta^2) about an axis tilted out of the equator by arctan(Delta/Omega). It depends on the detuning, so the sequence is no longer a polynomial in z and no longer periodic in Delta.

The clip, part one: a pi pulse on resonance takes the spin from up to down along a meridian. At Delta/Omega = 0.23 and 0.46 the axis tilts, and the spin misses the south pole. Inside our range of 1.5A, |Delta|/Omega_max reaches 0.46 at Omega/2pi = 10 MHz and 0.23 at 20 MHz; Omega_max = Omega/sqrt(2) in the spin-1/2 picture of the DQ drive. Far from resonance a pulse barely rotates the spin at all, so the periodic images of the windows are suppressed.

Part two: to first order in Delta/Omega, a pulse of angle beta = Omega t equals the instantaneous pulse with a free evolution e = tan(beta/2)/Omega before and after it. The first-order terms match: in the frame of the pulse the term is (Delta/2 Omega)[sin beta sigma_z + (1 - cos beta) sigma_y]; for the instantaneous pulse between two free evolutions e it is (Delta e/2)[(1 + cos beta) sigma_z + sin beta sigma_y]; they agree for e = tan(beta/2)/Omega. In the clip, a pi/2 pulse at Delta/Omega = 0.46: the violet path is the real pulse; the teal path, instantaneous pulse then free evolution, ends next to it; the red point, the instantaneous pulse alone, is far off. For a small angle e is half the pulse duration; for beta -> pi it diverges, so a pi pulse cannot be replaced this way.

So between pulses j and j+1 the free evolution is shortened to g_j = 2 pi/A - e_j - e_{j+1}, which keeps the effective spacing at 2 pi/A to first order. What remains is of relative size (Delta/Omega)^2 and grows with Delta.

Example: the exact two-pulse design at tau = 2 pi/A, pi/2 - wait - pi/2, with real full-amplitude pi/2 pulses: tan(pi/4) = 1, so the wait is shortened by 2/Omega_max. Its population error is 1.4e-6 at the drive strength of Mamin et al. (Omega_DQ/2pi = 31.25 MHz) and 5.3e-4 at 10 MHz; without the shortening it is no better than one pi pulse (6.3e-3 and 0.11).
-->
