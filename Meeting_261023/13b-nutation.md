---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
.columns { gap: 1.0rem; }
.columns .col { flex: 1 1 0; }
mjx-container[display="true"] { margin: 0.25em 0 !important; }
figure.figure { margin: 0.1em auto; }
ul { margin: 0.2em 0; }
</style>

# <span class="cat intro">Intro</span> DQ nutation

The $\ket{0}$ population oscillates differently in each manifold: one line on resonance, two off resonance.

<div class="columns">
<div class="col">

<figure class="figure">

![w:450](media/dq-nutation.png)

</figure>

$$
P_0^{(m)}(t)=\Big[\frac{\Delta_m^2}{\Omega_m'^2}+\frac{\Omega_{DQ}^2}{\Omega_m'^2}\cos\Omega_m't\Big]^2,\quad \Omega_m'=\sqrt{\Omega_{DQ}^2+\Delta_m^2}
$$

</div>
<div class="col">

<figure class="figure">

![w:450](media/dq-spectrum.png)

</figure>

$$
P_0^{(m)}=c_m+\frac{2\Delta_m^2\Omega_{DQ}^2}{\Omega_m'^4}\cos\Omega_m't+\frac{\Omega_{DQ}^4}{2\Omega_m'^4}\cos2\Omega_m't
$$

</div>
</div>

- On resonance $P_0=\cos^2\Omega_{DQ}t$: a single line at $2\Omega_{DQ}$.
- Off resonance $q<1$: $P_0$ returns only to $0.69$ at the DQ $\pi$ time, and a line at $\Omega_m'$ appears.

<!--
The |0> population under DQ driving and its spectrum. Left figure: Omega/2pi = 10 MHz, Omega_DQ/2pi = 7.07 MHz, the drive on the m_I = 0 resonance; blue m_I = 0, red m_I = +-1 (Delta = +-2.16 MHz), dashed the average over the unpolarised nucleus, which is what a measurement of the |0> population sees.

Why this shape. With phi = 0 the drive couples |0> only to one combination of +1 and -1, the bright state |B> = (|+1> + |-1>)/sqrt2, with strength Omega_DQ. The orthogonal combination, the dark state |D> = (|+1> - |-1>)/sqrt2, is not driven. The differential detuning couples bright and dark: S_z |B> = |D>, with strength Delta_m. So each manifold is a three-level ladder, |0> - |B> - |D>, a Morris-Shore reduction of the V system.

On resonance, Delta = 0, the dark state is never reached and |0> and |B> exchange their population completely: P_0 = cos^2(Omega_DQ t). After the pi/2 pulse (Omega_DQ t = pi/2, 35 ns) |0> is empty, which is the preparation of the +-1 superposition; after the DQ pi pulse (Omega_DQ t = pi, 71 ns) the bright transition has made a full 2 pi rotation, the population is back in |0>, and +1 and -1 have been exchanged.

Off resonance, the ladder has eigenvalues 0 and +-Omega', with Omega' = sqrt(Omega_DQ^2 + Delta^2). The state |0> has weight Delta^2/Omega'^2 on the zero eigenvector, which gives the closed form P_0 = [Delta^2/Omega'^2 + (Omega_DQ^2/Omega'^2) cos Omega' t]^2. It is the same as P_0 = (1 - 2q)^2 with Rabi's formula for the spin-1/2 flip probability, q = (Omega_DQ^2/Omega'^2) sin^2(Omega' t/2). The oscillation is faster (period 2 pi / Omega' = 135 ns for the outer manifolds, against 71 ns for m_I = 0 in P_0), and q never reaches 1: q is at most Omega_DQ^2/Omega'^2 = 0.91. So at the DQ pi time the outer manifolds return only to P_0 = ((Omega_DQ^2 - Delta^2)/Omega'^2)^2 = 0.69 (10 MHz drive): the rest is in |+-1>, partly in the dark state. Every second maximum is lower. |0> does empty (q = 1/2 is reached), which is why the preparation looks fine in population; its problem is the phase (next slide).

The average over the nucleus contains three frequencies: 2 Omega_DQ from m_I = 0, and 2 Omega' and Omega' from the outer manifolds; the two fast lines beat at about A^2 / Omega_DQ. This is what two-frequency nutation looks like with an unpolarised 14N.

The spectrum (right). Square the bracket: P_0 = a^2 + b^2/2 + 2ab cos(Omega' t) + (b^2/2) cos(2 Omega' t), with a = Delta^2/Omega'^2 and b = Omega_DQ^2/Omega'^2. So each manifold has at most two lines: one at 2 Omega' with weight Omega_DQ^4 / 2 Omega'^4, the bright-transition Rabi oscillation, and one at Omega' with weight 2 Delta^2 Omega_DQ^2 / Omega'^4, which exists only off resonance because it comes from the dark state. On resonance only the first survives, at 2 Omega_DQ, with weight 1/2. At Omega/2pi = 10 MHz: m_I = 0 gives a line at 14.14 MHz; m_I = +-1 give lines at 14.79 MHz and 7.39 MHz. Averaged over the nucleus (weights 1/3, 2/3) the amplitudes are 1/6 = 0.167 at 14.14 MHz, 0.279 at 14.79 MHz and 0.104 at 7.39 MHz. The coloured bars are these closed-form lines; the black curve is the Fourier amplitude of the averaged trace over 5 microseconds, which reproduces them. The two high lines are 0.65 MHz apart, about A^2/Omega_DQ, so a measurement must run for well over a microsecond to resolve them; doubling the drive halves that splitting and suppresses the line at Omega'.
-->
