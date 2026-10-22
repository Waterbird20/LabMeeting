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

# <span class="cat strategy">Strategy</span> Universal NOT

With an unpolarised ¹⁴N the electron undergoes $\tfrac13\sum_mU_m\rho\,U_m^\dagger$, the universal NOT $\mathbf r\mapsto-\mathbf r/3$ if the $U_m$ are $\pi$ rotations about orthogonal axes; the clip shows this target and a robust eight-pulse response.

<figure class="figure">

<video src="media/unot-response.mp4" poster="media/unot-response.png" width="770" controls autoplay loop muted playsinline preload="none"></video>

</figure>

<!--
A different kind of target: the universal NOT. The ideal universal NOT maps every pure state to the orthogonal one, it inverts the Bloch vector; that map is antiunitary, and no physical operation does it. The best channel that treats all inputs alike contracts and inverts, r goes to -r/3, with fidelity 2/3 for every input: one third of the sum of sigma_j rho sigma_j (Buzek, Hillery and Werner, Phys. Rev. A 60, R2626, 1999).

The unpolarised nucleus provides the average. Each manifold gets its own unitary U_m, and the electron sees one third of the sum of U_m rho U_m-dagger. For equal weights this is the universal NOT exactly when the three Bloch rotations sum to minus the identity: all three angles must be pi and the three axes orthonormal. The frame and a sign per manifold are free.

In P and Q, a pi rotation about n has P = -i n_z and Q = n_y - i n_x. So on window m the target is a different pair. This is the only target in this talk that differs between the windows: for orthogonal axes two window targets are sqrt2 apart, genuine steps. Here the single-frequency qubit, zero and minus one, where u is the qubit operation.

The clip: first the target bars, real part blue and imaginary part orange, in the frame the design uses. Re P = 0 on every window because every rotation is a pi rotation. Then the robust design: eight fixed-length pulses at Omega/2pi = 20 MHz with seven waits of 2 pi / 3A, 1480 ns. Each component follows its bar across the window and turns between the windows. Process infidelity at most 6.9e-4 on every window.

At tau = 2 pi / 3A the three resonances sit at the cube roots of unity on the circle of the waits, as far apart as possible. There is even a closed form with instantaneous pulses: a pi pulse about x, a wait, a pulse about y by beta with cos beta = 1/3, a wait. A palindromic sequence cannot do it: it has Q purely imaginary, n_y = 0 on every window, and three orthogonal axes cannot all lie in the x-z plane.

What it shows: 2/3 is also what measure-and-prepare reaches, and the channel is entanglement breaking. Each pi rotation squares to -1, so applying the sequence twice restores the input. With the nucleus unpolarised this demonstrates control, three different rotations from one sequence; a quantum test would need the nitrogen in a superposition of m_I.
-->
