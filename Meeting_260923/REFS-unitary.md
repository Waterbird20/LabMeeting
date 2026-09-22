# References for the quantum-operation slides (`16-`, `16a-`, `16b-`)

Not a slide (no leading digits, so the Makefile ignores it). The integrator should fold these
into `80-references.md`.

1. L. M. Pham, *Magnetic Field Sensing with Nitrogen-Vacancy Color Centers in Diamond*.
   Doctoral dissertation, Harvard University (2013), Chapter 1.
   Raw copy: `1. raw/papers/Pham_gsas.harvard_0084L_10993.pdf`.
   Source for the NV ground-state spin picture used on `16a-drive.md`: the $^{3}A_2$
   spin-triplet ground state with zero-field splitting $D_{\rm gs}\approx2.87$ GHz
   (Sec. 1.4), the Zeeman shift $\Delta=m_s\gamma B_{\parallel}$ with gyromagnetic ratio
   $\gamma=g\mu_B/h=2.8$ MHz/G (Sec. 1.4), and the resonant microwave drive with Rabi
   frequency $\Omega=\gamma B_1$, Eq. (1.1) in Sec. 1.5.2, where $B_1$ is the component of
   the microwave field perpendicular to the N-V symmetry axis.

2. M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum Information*,
   10th anniversary edition, Cambridge University Press (2010), Secs. 2.2.2 and 4.2.
   Source for the textbook material on `16-unitary.md` and `16b-rotation.md`: the
   Schrodinger equation and the propagator $U=e^{-iHt}$ for time-independent $H$, and the
   rotation-operator identity
   $e^{-i(\theta/2)\vec n\cdot\vec\sigma}=\cos(\theta/2)I-i\sin(\theta/2)\,\vec n\cdot\vec\sigma$.
   This is the same book already proposed in `REFS-basics.md`, so it should appear only once
   in `80-references.md`.

3. Figure `media/unitary-stepping.png` on `16a-drive.md` is our own computation, not a
   literature figure. Script: `3. wiki/code/lm-260923-animations/fig_unitary.py`, run with
   `3. wiki/code/env-manim/.venv/bin/python`, deterministic, seed $42$. Parameters: a
   Gaussian microwave envelope of peak Rabi frequency $\Omega_0/2\pi=10$ MHz and width
   $45$ ns, detuning $\delta/2\pi=2$ MHz, pulse duration $200$ ns, rotating-frame
   Hamiltonian $H(t)=\tfrac{\delta}{2}Z+\tfrac{\Omega(t)}{2}X$. Measured numbers, for the
   record: at $N=16$ steps the Euler propagator has
   $\|U^{\dagger}U-I\|_F=2.5$ against $1.6\times10^{-15}$ for the product form, and at
   $N=8192$ steps $3.0\times10^{-3}$ against $8.2\times10^{-15}$; the propagator errors
   $\|U_N-U_{\rm exact}\|_F$ fall as $1/N$ for both schemes, $9.7\times10^{-1}$ to
   $1.5\times10^{-3}$ for Euler and $1.4\times10^{-2}$ to $2.6\times10^{-5}$ for the
   product form.

<!-- EDIT-FORWARD: reference 2 is a convenience citation for standard textbook material and
     is not filed in `1. raw/` or `1. raw/references.bib`. Either drop it from
     `80-references.md`, keep only the single shared entry from `REFS-basics.md`, or have the
     book filed with `aleph-ref` first. -->

<!-- EDIT-FORWARD: the Pham thesis is a loose PDF in `1. raw/papers/` and has no citekey
     folder, so the slides cite it in prose as "Pham thesis, Ch. 1" rather than with a
     bracketed number. If the speaker prefers bracketed numbering throughout, the thesis
     should be filed with `aleph-ref` and the prose citation on `16a-drive.md` replaced. -->
