# References for the entanglement slides (`18-`, `18a-`, `18b-`)

Not a slide (no leading digits, so the Makefile ignores it). The integrator should fold these
into `80-references.md`.

1. L. M. Pham, *Magnetic Field Sensing with Nitrogen-Vacancy Color Centers in Diamond*,
   doctoral dissertation, Harvard University (2013), Chapter 1, section 1.6
   (`1. raw/papers/Pham_gsas.harvard_0084L_10993.pdf`). Source for every number on
   `18b-decay.md`: $T_2^{*}\approx 180$ ns from a single-NV Ramsey free-induction decay
   (Fig. 1.7), $T_2^{*}\approx 500$ ns for an ensemble from the zero-power extrapolation of the
   ESR linewidth (Fig. 1.8), $T_2^{*}\sim 1\ \mu$s set by natural-abundance $^{13}$C
   ($\Gamma_{^{13}\mathrm{C}}\sim 10^{6}\ \mathrm{s}^{-1}$), $T_2^{*}>10\ \mu$s in $99.7\%$
   $^{12}$C material, $T_2=397\pm 5\ \mu$s from an ensemble spin echo (Fig. 1.9), and
   $T_1\sim 6$ ms in bulk diamond at room temperature.
   <!-- EDIT-FORWARD: the thesis is a loose PDF in `1. raw/papers/` with no citekey folder,
        so `18b-decay.md` cites it in prose as "Pham thesis, Ch. 1" rather than by number. -->
2. T. H. Taminiau, J. J. T. Wagenaar, T. van der Sar, F. Jelezko, V. V. Dobrovitski and
   R. Hanson, *Detection and control of individual nuclear spins using a weakly coupled
   electron spin*, *Phys. Rev. Lett.* **109**, 137602 (2012), arXiv:1205.4128. Source for the
   conditional Hamiltonian of `18b-decay.md`, their Eq. (4) and Eq. (5),
   $\hat H=A\hat S_z\hat I_z+B\hat S_z\hat I_x+\omega_L\hat I_z
   =\ket{0}\bra{0}\hat H_0+\ket{1}\bra{1}\hat H_1$ with $\hat H_0=\omega_L\hat I_z$ and
   $\hat H_1=(A+\omega_L)\hat I_z+B\hat I_x$, and for the statement that a conditional rotation
   of the nucleus entangles it with the electron so that the electron is left in a mixture.
   Citekey `taminiau2012detection`.
3. Clip `bell-mixed`: `3. wiki/code/lm-260923-animations/scenes_bell.py`, class `BellMixed`,
   rendered with `render.py` on the shared helper `bloch.py`. Deterministic, seed `20260923`
   for the five hyperfine couplings of part 3. The reduced state is a genuine numerical partial
   trace, and the gate is interpolated as the matrix power
   $\mathrm{CNOT}^{s}=\ket{0}\bra{0}\otimes I+\ket{1}\bra{1}\otimes X^{s}$, which gives
   $|\vec r_1|=\cos(\pi s/2)$.
4. Figure `media/bell-circuit.png`: quantikz, compiled with `pdflatex` and cropped to $680$ px
   wide. It is a plain $H$ plus CNOT circuit, nothing to source.
