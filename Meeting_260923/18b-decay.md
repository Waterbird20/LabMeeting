---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<!-- EDIT-FORWARD: speaker will supply the extended-T2 plot -->

# <span class="cat intro">Intro</span> Interaction is the source of entanglement.

<div class="columns">
<div class="col">

The CNOT came from an interaction, and in the NV centre the hyperfine coupling to a nearby nucleus has the same conditional form, with $\hat H_0=\omega_L\hat I_z$ and $\hat H_1=\hat H_0+A\hat I_z+B\hat I_x$:

$$
\begin{gathered}
\hat H=\ket{0}\bra{0}\otimes\hat H_0+\ket{1}\bra{1}\otimes\hat H_1 ,\\[2pt]
\ket{+}\ket{n}\;\to\;\tfrac{1}{\sqrt{2}}\big(\ket{0}U_0\ket{n}+\ket{1}U_1\ket{n}\big),\\[2pt]
|\vec r_e|=\big|\bra{n}U_1^{\dagger}U_0\ket{n}\big| .
\end{gathered}
$$

The nucleus precesses about a different axis in each branch, since $H_0 \neq H_1$, so the electron's Bloch radius is the overlap of the two branches. There are many independent nuclei, so multiply their overlaps, giving $M(t)\simeq e^{-(t/T_2^{*})^{2}}$.

</div>
<div class="col">

Three names for the same shrinking, NV at room temperature.

- **$T_2^{*}$, free precession.** The Ramsey decay set by the static spread of local fields, $\approx 180\,\mathrm{ns}$ for one NV and $\approx 500\,\mathrm{ns}$ for an ensemble.
- **$T_2$, spin echo.** A $\pi$ pulse refocuses whatever was static, so only the motion of the bath survives, giving $397\pm 5\,\mu\mathrm{s}$.
- **$T_1$, relaxation.** Phonons flip the spin and the populations decay, typically $\sim 6\,\mathrm{ms}$.

Natural-abundance $^{13}\mathrm{C}$ caps $T_2^{*}$ near $1\,\mu\mathrm{s}$; $99.7\%$
$^{12}\mathrm{C}$ material exceeds $10\,\mu\mathrm{s}$ (Pham thesis, Ch. 1; conditional form
from Taminiau et al., 2012).

<!-- TODO: split this column to next slide. -->
<!-- TODO: Instead insert splited video from previous slide. -->
<!-- TODO: In animation, both state rotates along z-axis. But let one of them rotate in tilted axis to x-axis. -->

</div>
</div>
