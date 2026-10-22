---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

<style scoped>
figure.figure { margin: 0.2em 0 0 0; }
p { margin: 0.2em 0; }
</style>

# <span class="cat strategy">Strategy</span> State preparation

The clip shows the target for preparing $\tfrac{1}{\sqrt2}(\ket{{+1}}+\ket{{-1}})$, a $\pi/2$ pulse about $x$ on all three windows, and then $P$ and $Q$ of one rectangular $\pi/2$ pulse and of a robust four-pulse sequence without waits.

<figure class="figure">

<video src="media/prep-response.mp4" poster="media/prep-response.png" width="770" controls autoplay loop muted playsinline preload="none"></video>

</figure>

<!--
The second operation a DQ magnetometer needs: prepare the superposition of plus one and minus one from zero, B_phi = (|+1> + e^{i phi} |-1>) / sqrt2. The sequence acts as D^1(u), and the prepared state is the middle column of D^1(u): minus sqrt2 P Q-star, |P|^2 - |Q|^2, sqrt2 Q P-star. So the populations of plus and minus one are always equal, and the relative phase is phi = pi + 2 arg(Q P-star).

The magnetometer reads the DQ coherence. With the nucleus mixed it is the average over the three manifolds, minus two thirds times the sum of (P_m Q_m-star)^2. It reaches its maximum 1/2 only if every manifold has |P_m|^2 = 1/2 and the same phase. Correct populations with phases that differ between the manifolds average the coherence away. So the requirement is phase coherence: the same state in all three manifolds.

The target: a pi/2 pulse about x, P = 1/sqrt2 and Q = -i/sqrt2, on every window of plus-minus 0.1A. That prepares B_0. Every other phase follows from the same design: advancing the phase of every pulse by phi_0 multiplies Q by e^{i phi_0}, so phi goes to phi + 2 phi_0, exactly, for pulses of any shape.

The state depends on P and Q only up to a common phase, so the clip shows them with that phase chosen to make P real. First the target bars on the three windows: P = 1/sqrt2, Re Q = 0, Im Q = -1/sqrt2.

Then one rectangular pi/2 pulse at Omega/2pi = 10 MHz, 35 ns. It moves the population correctly, P stays at 1/sqrt2, but Re Q tilts with the detuning: the prepared phase turns by about 2 Delta / Omega_DQ. A real pulse acts like an instantaneous pulse between two short free evolutions, and the free evolution after it turns the phase. Infidelity 0.09 at plus-minus A, 6.0e-2 averaged over the nucleus, and the coherence drops from 0.5 to 0.44. At the drive of Mamin et al. the same error is 3.2e-3, at 20 MHz 1.5e-2.

Then the robust design: four fixed-length pulses about plus and minus x, no waits, 566 ns at 10 MHz. P, Re Q and Im Q are flat across all three windows, infidelity at most 8.8e-5. At 20 MHz three pulses suffice, 212 ns.

The full Hamiltonian check: electron and nitrogen, counter-rotating terms, the cross terms of the two frequency components, non-secular couplings, nucleus mixed, all transitions detuned together by up to 0.2 MHz. At 10 MHz this design stays between 4.1e-5 and 1.5e-4, coherence 0.5000. At 20 MHz a detuning-independent leakage of about 7e-4 remains. The time-reversed sequence reads the state out exactly, because P^2 + Q^2 = 0 for a phase-coherent preparation.
-->
