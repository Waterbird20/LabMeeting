# References for section 2 (convolution)

Not a slide (no leading digits, so the Makefile ignores it). The integrator should fold these
into `80-references.md`.

1. T. H. Taminiau, J. J. T. Wagenaar, T. van der Sar, F. Jelezko, V. V. Dobrovitski and
   R. Hanson, *Detection and control of individual nuclear spins using a weakly coupled
   electron spin*. *Phys. Rev. Lett.* **109**, 137602 (2012). Source for the conditional
   propagators $V_0$, $V_1$, the contrast $\mathfrak{M}=\tfrac12\operatorname{Tr}V_0V_1^\dagger$
   and the resonance condition of the CPMG comb. Citekey `taminiau2012detection`.
2. Internal DDrf analysis, lab meeting 2026-09-03,
   `3. wiki/labmeeting/260903/` slides `24-signal.md`, `51-steps.md`, `52-local.md`.
   Source for the factorization $\mathfrak{M}=\prod_j\mathfrak{M}_j$, the shallow-dip
   expansion, the three-factor lineshape and the boxed identity
   $1-|\mathfrak{M}|\simeq[W*|\mathcal{D}_K|^2]$.
   Figures `ddrf-fig2-factors.png` and `ddrf-fig5-bath.png` are reused from that deck.
3. Clips `dice-grid`, `dice-slide`, `dice-weighted`, `simple-example`, `polynomial` and
   `conv-to-mult` are the manim animations rendered for the 2026-09-03 deck and copied into
   `media/`.

<!-- EDIT-FORWARD: if the DDrf work is being written up as a manuscript or a preprint, the
     speaker should replace reference 2 with the manuscript citation. -->
