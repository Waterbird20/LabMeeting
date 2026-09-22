# References for the pulse-sequence slides (`17-examples`, `17a`, `17b`, `17c`)

Not a slide (no leading digits, so the Makefile ignores it). The integrator should fold these
into `80-references.md`.

1. L. M. Pham, *Magnetic Field Sensing with Nitrogen-Vacancy Color Centers in Diamond*,
   Ph.D. thesis, Harvard University (2013).
   `1. raw/papers/Pham_gsas.harvard_0084L_10993.pdf`.
   Source for every experimental statement on these four slides: Sec. 1.5.1 for CW-ESR,
   which is the ODMR spectrum, and Fig. 1.4 for the measured spectra; Sec. 1.5.2 and Fig. 1.5
   for Rabi nutations, with $\Omega=\gamma B_1$ and $\gamma=2.8$ MHz/G as Eq. 1.1, and Fig. 1.6
   for the beating and decay from off-resonant hyperfine driving; Sec. 1.6.1 and Fig. 1.7 for
   the Ramsey free induction decay and $T_2^{*}\approx 180$ ns on a single NV; Ch. 2, Eq. 2.1
   for the power-broadening trade-off between contrast and linewidth.
   Figures `pham-fig-1-4.png`, `pham-fig-1-5.png` and `pham-fig-1-7.png` in `media/` are crops
   of Fig. 1.4(b), Fig. 1.5(b) and Fig. 1.7(b).

2. The qubit-level algebra is standard two-level physics and is not taken from the thesis.
   The rotating-frame Hamiltonian $H=\tfrac{\delta}{2}Z+\tfrac{\Omega}{2}X$, the propagator
   $U=\exp(-\tfrac{i}{2}\Omega_R t\,\hat n\cdot\vec\sigma)$ and the three read-out formulas are
   derived and checked numerically in `3. wiki/code/lm-260923-animations/scenes_seq.py`
   (function `_verify`, which runs at import and compares each formula with `scipy.linalg.expm`
   of the Hamiltonian).

3. Clips `seq-rabi`, `seq-ramsey` and `seq-odmr` are rendered from that same file with
   `render.py`. Parameters: $\Omega=1$ in the clip's own time unit; Rabi sweeps
   $\Omega t\in[0,4\pi]$; Ramsey uses $\delta=1$ and ten values $\delta\tau=0.4\pi k$,
   $k=1\ldots10$, with no decoherence, so the clip shows the ideal fringe while the slide
   carries the $T_2^{*}$ envelope; ODMR uses the fixed pulse $t=\pi/\Omega$ and seventeen
   detunings $\delta/\Omega=-4,-3.5,\ldots,4$.

<!-- EDIT-FORWARD: the Gaussian envelope $e^{-(\tau/T_2^*)^2}$ on slide `17b` is the usual
     quasi-static-bath form. The thesis states only that $T_2^*$ is "the characteristic time of
     the decay envelope" (Sec. 1.6.1) and does not fix the exponent, so change it to
     $e^{-\tau/T_2^*}$ or to a stretched exponential if the lab fits a different form. -->

<!-- EDIT-FORWARD: Sec. 1.5.1 of the thesis is continuous-wave ESR, not the pulsed ODMR drawn
     on slide `17c`. If the group runs CW-ESR, the lineshape to quote is a Lorentzian of width
     set by the laser and microwave powers rather than the sinc-like curve of a fixed
     $\pi$ pulse. -->
