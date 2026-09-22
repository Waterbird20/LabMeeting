# References for the Bloch-sphere slides (`14-bloch.md`, `14a-angles.md`, `15-our-case.md`)

Not a slide (no leading digits, so the Makefile ignores it). The integrator should fold these
into `80-references.md`.

1. H. J. Pham, *Magnetic Field Sensing with Nitrogen-Vacancy Color Centers in Diamond*,
   PhD thesis, Harvard University (2013), Ch. 1.
   `1. raw/papers/Pham_gsas.harvard_0084L_10993.pdf`. Source for slide `15-our-case.md`:
   the NV electronic spin is initialised into $m_s=0$ by optical pumping and detected via
   spin-state-dependent fluorescence integrated over roughly $300$ ns, and the pulse
   sequences in that chapter are drawn as initialisation, microwave operation, detection.
2. Clip `bloch-plus`, rendered for this deck by
   `3. wiki/code/lm-260923-animations/scenes_bloch_plus.py` on the shared helper
   `bloch.py`. Every Bloch vector shown in the clip is computed from the actual ket with
   `bloch_vector`, so the drawn arrow and the printed formula cannot disagree.

The algebra on `14-bloch.md` and `14a-angles.md` (the Pauli expectation values of $\ket{+}$,
the spherical-coordinate form of $\vec{r}$, and $|\vec{r}|^2=2\operatorname{Tr}\rho^2-1$) is
standard textbook material and is derived in full on the slides, so it needs no citation
beyond a general reference if the speaker wants one.

<!-- EDIT-FORWARD: if the speaker prefers a textbook citation for the Bloch sphere, Nielsen
     and Chuang, *Quantum Computation and Quantum Information*, Section 1.2 and Section 2.4.2,
     is the usual one. It is not in `1. raw/` yet, so it is not cited on the slides. -->
