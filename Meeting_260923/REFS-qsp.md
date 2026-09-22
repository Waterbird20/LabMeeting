# References cited in Section 3 (QSP)

Not a slide (no leading digits, so the Makefile ignores it). The section's slides cite
these as `[1]`, `[2]`, `[3]`; the integrator should renumber them into `80-references.md`.

1. J. M. Martyn, Z. M. Rossi, A. K. Tan, I. L. Chuang, *Grand Unification of Quantum
   Algorithms*. PRX Quantum **2**, 040203 (2021). `arXiv:2105.02859`.
   Used for: Theorem 1 in the $W_x$ convention, the trivial-phase Chebyshev pair, the
   $\mathrm{Re}\,P$ reachability statement, and the Appendix-D phase lists behind the
   gallery figures (sign, threshold, cosine, sine).
2. D. Motlagh, N. Wiebe, *Generalized Quantum Signal Processing*. PRX Quantum **5**,
   020368 (2024). `arXiv:2308.01501`.
   Used for: the Laurent/GQSP form of the theorem (complex coefficients, no parity
   constraint) and the peel-off recursion that recovers the pulses.
3. Our own work on the $V_B^-$ conditional gate, wiki pages
   `3. wiki/projects/qsp/conditional-gate.md` (sections 1, 2 and 4) and
   `3. wiki/concepts/qsp-phase-filter-design.md`.
   Used for: the detuning-as-signal convention, the block structure
   $U_{\rm seq}=\sum_{m_I}\Pi_{m_I}\otimes U(\delta_{m_I})$, degree $=$ number of waits,
   the four-tap boxcar, the seven-pulse $d=6$ conditional gate, and the linear-programme
   formulation of the gate design.

Background works referred to in passing, kept here in case the speaker expands a slide:

4. J. Haah, *Product decomposition of periodic functions in quantum signal processing*.
   Quantum **3**, 190 (2019). The cleanest statement of the existence theorem in the
   Laurent form used on slides 32 and 39. Not filed in the vault.
5. Y. Dong, X. Meng, K. B. Whaley, L. Lin, *Efficient phase-factor evaluation in quantum
   signal processing*. Phys. Rev. A **103**, 042419 (2021); code QSPPACK,
   `github.com/qsppack/QSPPACK`.
6. pyqsp, `github.com/ichuang/pyqsp` (source of the Appendix-D phase lists of [1]).
7. The DDrf apodization figure on slide 35 is from our lab meeting of 2026-04-28, reused
   from the 2026-08-18 deck (`3. wiki/labmeeting/260818/media/`).
