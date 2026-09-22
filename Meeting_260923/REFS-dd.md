# References for the dynamical-decoupling slides (`19-`, `19a-`, `19b-`)

Not a slide (no leading digits, so the Makefile ignores it). The integrator should fold these into
`80-references.md`.

1. T. H. Taminiau, J. J. T. Wagenaar, T. van der Sar, F. Jelezko, V. V. Dobrovitski and R. Hanson,
   *Detection and control of individual nuclear spins using a weakly coupled electron spin*,
   **Phys. Rev. Lett. 109, 137602 (2012)**, arXiv:1205.4128. Citekey `taminiau2012detection`,
   filed at `1. raw/papers/taminiau2012detection/`.
   Used for: the conditional Hamiltonian $H=\ket{0}\bra{0}H_0+\ket{1}\bra{1}H_1$ with
   $H_0=\omega_LI_z$ and $H_1=(A+\omega_L)I_z+BI_x$ (supplemental Eqs. (4), (5)); the CPMG unit and
   its two branch propagators $V_0=U_0U_1U_1U_0$, $V_1=U_1U_0U_0U_1$ (supplemental Eqs. (7), (8));
   the signal $P_x=(M+1)/2$ with $M=1-(1-\hat n_0\cdot\hat n_1)\sin^2(N\phi/2)$ (main-text Eqs. (1),
   (2)); the resonance positions $\tau_k=(2k-1)\pi/(2\omega_L+A)$ with $A=\omega_h\cos\theta$
   (main-text Eq. (3); the supplemental Eq. (15) writes the same family as $(2k+1)\pi/(2\omega_L+A)$);
   and the hyperfine parameters of nuclear spin 3, $\omega_h/2\pi=55(2)$ kHz and $\theta=54(2)^\circ$
   at $B_0=401$ G (Table I and Fig. 2), which drive the `dd-cpmg` clip.
   The reused figure is `media/taminiau-fig1.png`, a crop of the paper's Fig. 1.

2. T. M. Pham, *Magnetic Field Sensing with Nitrogen-Vacancy Color Centers in Diamond*, PhD thesis,
   Harvard University (2013), Sec. 1.6.2, at
   `1. raw/papers/Pham_gsas.harvard_0084L_10993.pdf`. Used for: the Hahn-echo definition of $T_2$,
   the collapses and revivals at multiples of the ${}^{13}$C Larmor period, and the convention that
   an $n$-pulse decoupling sequence measures a longer $T_2^{(n)}$.

3. The clip is generated code, `3. wiki/code/lm-260923-animations/scenes_dd.py`
   (class `DDCpmg`, `CLIP = "dd-cpmg"`). Its `_verify()` re-derives every formula above from
   `scipy.linalg.expm` at import time, so a wrong formula cannot be rendered.

<!-- EDIT-FORWARD: slide `26-dd-product.md` writes the same overlap as $\mathfrak{M}$ and already
     carries a note about switching to $M$. These slides use $M$, the speaker's own notation, so the
     switch can now be made. -->
