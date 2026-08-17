---
marp: true
theme: serif
paginate: true
math: mathjax
footer: 'Donghun Jung · Journal Meeting'
---

<!-- _class: title -->
<!-- _paginate: false -->
<!-- _footer: '' -->

<div class="kicker">Paulee Group · Center for Quantum Technology, KIST</div>

# Tutorial: Quantum Signal Processing

<div class="meta">
<span class="author">Donghun Jung</span>
<span class="affil">2026-08-18 · Journal Meeting</span>
</div>

<style>
/* deck-wide: trim margins and give figures more room */
section { padding: 50px 54px; }
.columns { gap: 1.0rem; }
figure.figure p { color: #6b6b6b; font-size: 0.74em; font-style: italic; margin-top: 0.35em; }
.reqs ul { list-style: none; padding-left: 0; }
</style>


---


# Outline

1. **Signal processing, classically.** 
2. **Quantum signal processing.** 
- The QSP sequence
- The polynomial theorem with concrete $(P, Q)$ examples
- The response examples.
3. **Fitting phases** 

<div class="legend">
<span class="cat intro">Intro</span>
<span class="cat method">Method</span>
<span class="cat strategy">Strategy</span>
<span class="cat results">Results</span>
<span class="cat ongoing">Ongoing</span>
</div>


---


# <span class="cat intro">Journal Club</span> The papers behind this tutorial

- J. M. Martyn, Z. M. Rossi, A. K. Tan, I. L. Chuang, *Grand Unification of Quantum Algorithms*, PRX Quantum **2**, 040203 (2021). A tutorial showing that search, phase estimation, and Hamiltonian simulation are all one algorithm, the quantum singular value transformation (QSVT). We use its QSP core: Sec. II A, the conventions of App. A, and the explicit phase lists of App. D.

- D. Motlagh, N. Wiebe, *Generalized Quantum Signal Processing*, PRX Quantum **5**, 020368 (2024). 




---


<!-- _class: section -->
<!-- _paginate: false -->

<div class="sec-num">01</div>

# Signal processing, classically

<div class="subtitle">From window coefficients to filters, including our own DDrf apodization</div>


---


# <span class="cat intro">Intro</span> Windowing, also called apodization, is filter design

Multiplying data by a window $f(k)$ in time convolves the spectrum with the window's own Fourier transform, so the FFT of the taps **is** the filter:
$$
\widetilde{S}(\omega) = (S * F)(\omega), \qquad F(\omega) = \sum_k f(k)\, e^{-i\omega k}.
$$

<div class="columns">
<div class="col">

<figure class="figure">

![w:430](./Meeting_260818/src/Presentation/media/window_taps.png)

*Time domain: the chosen coefficients $f(k)$ ($48$ taps).*

</figure>

</div>
<div class="col">

<figure class="figure">

![w:430](./Meeting_260818/src/Presentation/media/window_response.png)

*Frequency domain: rectangular sidelobes start at $-13$ dB; Blackman pushes them below $-58$ dB for a wider main lobe.*

</figure>

</div>
</div>


---


# <span class="cat results">Results</span> Recap: DDrf spectroscopy and its side peaks

<div class="columns">
<div class="col">

DDrf drives a nuclear spin with $N$ identical RF cells interleaved with electron $\pi$-pulses; sweeping $\omega_{\mathrm{RF}}$ maps out resonances. In April we found **side peaks** persisting even where the gate should be unconditional; the detuned-Rabi formula explains envelope and lobe period,
$$
\tfrac{1}{2}\,\mathrm{Tr}\,U_0 U_1^\dagger = 1 - \frac{2\Omega^2}{\Omega^2 + \delta_1^2}\, \sin^2\!\Big(\frac{\sqrt{\Omega^2 + \delta_1^2}\; N\tau}{2}\Big),
$$
with $\delta_1$ the detuning. These sidelobes are the Fourier structure of a **rectangular** drive envelope.

</div>
<div class="col">

<figure class="figure">

![w:500](./Meeting_260818/src/Presentation/media/sidepeak.png)

*The observation from the April meeting: side peaks in the spectroscopy signal at parameters where the response should be flat. (source: lab meeting 2026-04-28)*

</figure>

</div>
</div>


---


# <span class="cat results">Results</span> Suppression by apodized pulse envelopes

Replacing the constant amplitude by a windowed one, $\Omega_k = \Omega f(k)$, factorizes the normalized response into a $\mathrm{sinc}$ part and a kernel $G$ fixed by the window,
$$
\Big|\frac{F(\delta_1)}{F(0)}\Big|^2 = \mathrm{sinc}^2(u)\; |G(u)|^2, \qquad u = \frac{\delta_1}{2\Omega \bar f}.
$$

<div class="columns">
<div class="col">

<figure class="figure">

![w:460](./Meeting_260818/src/Presentation/media/DDrf_Apodization_N48_focused.png)

*Apodized envelopes flatten the detuned region, preserving the resonant peak ($N=48$).*

</figure>

</div>
<div class="col">

<figure class="figure">

![w:460](./Meeting_260818/src/Presentation/media/DDrf_Apodization_N136_focused.png)

*The suppression sharpens with the cell number ($N=136$). (source: lab meeting 2026-04-28)*

</figure>

</div>
</div>


---


<!-- _class: section -->
<!-- _paginate: false -->

<div class="sec-num">02</div>

# What is QSP doing?

<div class="subtitle">Interleaved rotations realize polynomial response functions</div>


---


# <span class="cat method">Method</span> The QSP sequence: one signal, many processing knobs

Two ingredients alternate [1, Sec. II A]. The **signal rotation** is a fixed $x$-rotation whose angle encodes the unknown signal $a \in [-1, 1]$,
$$
W(a) = \begin{bmatrix} a & i\sqrt{1-a^2} \\ i\sqrt{1-a^2} & a \end{bmatrix}, \qquad a = \cos\tfrac{\theta}{2},
$$
and it is applied $d$ times, always the same. The **signal-processing rotations** $S(\phi_k) = e^{i\phi_k Z}$ are $z$-rotations through angles that we choose freely. The full sequence is
$$
U_{\vec\phi} = e^{i\phi_0 Z}\, \prod_{k=1}^{d} W(a)\, e^{i\phi_k Z}, \qquad \vec\phi \in \mathbb{R}^{d+1}.
$$

The structure is a composite pulse in which all the design freedom sits in the $z$-phases: the signal enters only through the repeated $W(a)$, and the $d+1$ knobs $\phi_k$ decide what the sequence does with it.


---


# <span class="cat intro">Intro</span> Composite pulses: signal processing on the Bloch sphere

<div class="columns">
<div class="col">

This interleaved structure is older than quantum computing: it is an NMR **composite pulse**. There the rotation angle $\theta$ is set by the field, a miscalibrated pulse makes $\theta$ an unknown signal, and phased rotations are interleaved so that the transition probability responds to $\theta$ in a designed way.

BB1 uses five rotations with phases
$$
\vec\phi = \big(\tfrac{\pi}{2}, -\eta, 2\eta, 0, -2\eta, \eta\big), \quad \eta = \tfrac{1}{2}\cos^{-1}\!\big(\!-\!\tfrac{1}{4}\big),
$$
which flattens the response to $1 - \tfrac{5}{8}(\theta/2)^6$ near $\theta = 0$ and switches sharply near $\theta \approx 2\pi/3$.

</div>
<div class="col">

<figure class="figure">

![w:440](./Meeting_260818/src/Presentation/media/bb1_response.png)

*Reproduction of Fig. 1 of [1] with the phase vector below: the bare rotation responds as $\cos^2(\theta/2)$, the BB1 sequence holds the qubit unflipped over a wide band.*

</figure>

</div>
</div>


---


# <span class="cat method">Method</span> The QSP theorem: the response is a designed polynomial

<div class="callout">

**Theorem 1 of [1].** The sequence $U_{\vec\phi}$ always takes the form
$$
U_{\vec\phi} = \begin{bmatrix} P(a) & iQ(a)\sqrt{1-a^2} \\ iQ^*(a)\sqrt{1-a^2} & P^*(a) \end{bmatrix},
$$
and conversely a $\vec\phi$ exists for **any** polynomials $P, Q$ with (i) $\deg P \le d$, $\deg Q \le d-1$, (ii) parity $d \bmod 2$ and $(d-1) \bmod 2$, (iii) $|P|^2 + (1-a^2)|Q|^2 = 1$.

</div>

The forward direction is an induction: each extra $W(a)\,e^{i\phi_k Z}$ raises the degree by one and flips the parities, while unitarity enforces condition (iii). The converse is the useful direction: the reachable response family is known **exactly**, not perturbatively. The next slide makes $(P, Q)$ concrete.


---


# <span class="cat method">Method</span> What are $P$ and $Q$, concretely?

<div class="columns">
<div class="col">

Take no processing, $\vec\phi = (0, \dots, 0)$: the sequence is bare repetition, $U = W(a)^d$.

- $d = 1$: $P(a) = a$, $Q(a) = 1$.
- $d = 2$: squaring $W(a)$ gives $P(a) = 2a^2 - 1$, $Q(a) = 2a$.
- In general $P = T_d(a)$, $Q = U_{d-1}(a)$, Chebyshev of the first and second kind; condition (iii) is the identity $T_d^2 + (1-a^2)\,U_{d-1}^2 = 1$.

The phases $\phi_k$ reshape this family into any other admissible pair.

</div>
<div class="col">

<figure class="figure">

![w:500](./Meeting_260818/src/Presentation/media/chebyshev.png)

*$P(a) = T_d(a)$ at trivial phases for $d = 1, 2, 5$, from our evaluator (matches to $10^{-12}$).*

</figure>

</div>
</div>


---


# <span class="cat method">Method</span> Which functions can the response be?

**$P$ alone is boxed in at the endpoints.** At $a = \pm 1$ the signal rotation is trivial, $W(\pm 1) = \pm I$, so the whole sequence collapses to a single $z$-rotation and condition (iii) forces $|P(\pm 1)| = 1$. A modest target like $0.5\,\mathrm{sign}(a)$ is therefore impossible as $P$ itself.

**The fix is to read only the real part in the $\ket{+}$ basis** [1]:
$$
\langle +|U_{\vec\phi}|+\rangle = \mathrm{Re}\,P(a) + i\,\mathrm{Re}\,Q(a)\sqrt{1-a^2}.
$$
The target is encoded in $\mathrm{Re}\,P$, while $\mathrm{Im}\,P$ absorbs the unit-modulus burden: the published sign list has $\mathrm{Re}\,P(1) = 0.90$ with $|P(1)| = 1$ exactly, the rest being imaginary.

**Result.** $\mathrm{Re}\,P$ can be **any** real polynomial with parity $d \bmod 2$ and $|\mathrm{Poly}(a)| \le 1$. This is the design space used by the gallery and by our fits.

**Watch the convention:** papers and libraries differ by $\pm\pi/4$ phase shifts and signal rescalings [1, App. A]; a live example follows on the trigonometric gallery slide.


---


# <span class="cat method">Method</span> What can we do? Polynomial design is algorithm design.

<div class="columns">
<div class="col">

<figure class="figure">

![w:420](./Meeting_260818/src/Presentation/media/gallery_sign.png)

*Sign function ($d=19$): the decision primitive behind amplitude amplification and search.*

</figure>

</div>
<div class="col">

<figure class="figure">

![w:420](./Meeting_260818/src/Presentation/media/gallery_threshold.png)

*Threshold function ($d=18$): a band-pass response, used for eigenvalue thresholding and eigenstate filtering.*

</figure>

</div>
</div>

Appendix D of [1] also lists phases for $\cos$, $\sin$ (next slide), $1/a$, $e^{-\beta a}$, and ReLU; each response is drawn with its phase vector $\vec\phi$, the program that produces it.


---


# <span class="cat method">Method</span> Gallery: cosine and sine for Hamiltonian simulation

<style scoped>
.phivec p { font-size: 0.62em; color: #6b6b6b; text-align: center; margin: 0.05em 0; }
</style>

<div class="columns">
<div class="col">

<figure class="figure">

![w:385](./Meeting_260818/src/Presentation/media/gallery_cos.png)

*Cosine list [1, App. D4], $t = 5$, $d = 14$.*

</figure>

</div>
<div class="col">

<figure class="figure">

![w:385](./Meeting_260818/src/Presentation/media/gallery_sin.png)

*Sine list [1, App. D4], $t = 5$, $d = 15$.*

</figure>

</div>
</div>

<div class="phivec">

$\vec\phi_{\cos} = (-1.71, -0.05, 2.12, -0.83, -0.50, 0.41, 0.33, 0.91, -2.81, 0.41, -0.50, 2.31, -1.02, -0.05, 3.00)$

$\vec\phi_{\sin} = (-1.63, 0.21, -0.84, 0.40, -0.27, 2.41, 0.05, -2.03, 1.11, 0.05, -0.73, -0.27, 0.40, -0.84, 0.21, -0.06)$

</div>

Hamiltonian simulation uses the pair as $e^{-i\mathcal{H}t} = \cos(\mathcal{H}t) - i\sin(\mathcal{H}t)$; these two angle lists are the entire program (rendered as $\cos(2ta)/2$, $\sin(2ta)/2$ in our convention [1, App. A]).


---


# <span class="cat method">Method</span> Where do the phases come from?

Theorem 1 is existence; the numbers take an algorithm, as implemented in **pyqsp** (Chuang group; generated the App.-D lists) and **QSPPACK** (Lin group, MATLAB):

- **The target polynomial comes first**: a bounded polynomial approximant of the desired function, via $\mathrm{erf}$ and Taylor constructions, Chebyshev interpolation, or Remez exchange [1, App. D].
- **Exact factorization**: root-finding on the complementary polynomial (GSLW), or Laurent-polynomial division at machine precision (Haah; Chao *et al.*; `--method laurent`). Numerically delicate at large degree.
- **Iteration and optimization**: quasi-Newton on symmetric phase vectors (Dong *et al.*, QSPPACK; `--method sym_qsp`), fast and stable with degrees in the thousands routine; plain gradient descent on the response error also works (`--method tf`).

Our fits in Section 3 are the minimal version of the last route: same model, exact gradient.


---


# <span class="cat ongoing">Ongoing</span> Where this goes: from one number to a whole matrix

So far one qubit encoded one number $a$. **Qubitization** embeds a Hamiltonian $\mathcal{H} = \sum_\lambda \lambda \ket{\lambda}\bra{\lambda}$ as a block of a unitary,
$$
U = \begin{bmatrix} \mathcal{H} & \sqrt{1-\mathcal{H}^2} \\ \sqrt{1-\mathcal{H}^2} & -\mathcal{H} \end{bmatrix},
$$
and each eigenvalue $\lambda$ then lives in its own private Bloch sphere with signal $a = \lambda$. The **same** phase sequence acts on all of them at once [1, Sec. II C]:
$$
\mathrm{Poly}(\mathcal{H}) = \sum_\lambda \mathrm{Poly}(\lambda)\, \ket{\lambda}\bra{\lambda}.
$$

With the Appendix-D polynomials this one template becomes Grover search (sign), phase estimation (threshold), Hamiltonian simulation (cosine, sine), and matrix inversion ($1/a$): the "grand unification" of the title. Today we stop at the single-qubit layer.


---


<!-- _class: section -->
<!-- _paginate: false -->

<div class="sec-num">03</div>

# Fitting phases by our own hand

<div class="subtitle">A numpy evaluator, validated against the paper, then optimized</div>


---


# <span class="cat method">Method</span> Setup and validation gates

- **Model.** We implement $U_{\vec\phi}(a)$ as the literal Eq.-(3) product of $2 \times 2$ matrices and read out the response $\mathrm{Re}\,\langle +|U_{\vec\phi}|+\rangle$. Fitting minimizes the mean squared error to a target on a grid of $51$ points in $a \in [-1, 1]$.
- **Optimizer.** L-BFGS with the exact gradient (obtained by differentiating the product chain, $\partial U/\partial\phi_k = P_k\, (iZ)\, Q_k$ with prefix and suffix products $P_k, Q_k$), deterministic multi-start with seed $42$.
- **Validation before any fitting.** Trivial phases reproduce $T_d(a)$ below $10^{-12}$; BB1 matches its closed-form response to $10^{-15}$; the published $d=19$ sign list [1, App. D2] reproduces Fig. 21 of the paper, odd to machine precision with plateau values in $[0.881, 0.936]$.
- **Reference point.** The PennyLane demo *Function fitting using QSP* trains the same model with PyTorch SGD for up to $25{,}000$ iterations; with the exact gradient, every fit on the next slides converges in less than three seconds.


---


# <span class="cat results">Results</span> Warm-up target: exact once $d$ reaches the degree

<div class="columns">
<div class="col">

<figure class="figure">

![w:420](./Meeting_260818/src/Presentation/media/polyfit_response.png)

*Fits at $d = 1, 3, 5$ with the phases of the exact $d=5$ fit; the dashed target lies on the $d=5$ curve.*

</figure>

</div>
<div class="col">

<figure class="figure">

![w:410](./Meeting_260818/src/Presentation/media/polyfit_mse.png)

*The error collapses at $d = 5$, from $2.8 \times 10^{-2}$ to $5.9 \times 10^{-13}$, then stays at machine precision.*

</figure>

</div>
</div>

The response of the PennyLane demo target $4a^5 - 5a^3 + a$ improves with $d$ and becomes **exact** once the sequence can carry it: a degree-$5$ odd polynomial needs exactly $d = 5$.


---


# <span class="cat results">Results</span> Step target: systematic sharpening with $d$

<div class="columns">
<div class="col">

<figure class="figure">

![w:420](./Meeting_260818/src/Presentation/media/signfit_response.png)

*Fits of $\mathrm{sign}(a)$ at $d = 3, 9, 19, 31$ with the phases of the $d=31$ fit; the plateaus hug $\pm 1$ with ripples.*

</figure>

</div>
<div class="col">

<figure class="figure">

![w:410](./Meeting_260818/src/Presentation/media/signfit_mse.png)

*MSE decreases monotonically, $1.5 \times 10^{-1}$ at $d=3$ to $8.1 \times 10^{-3}$ at $d=31$; no finite degree is exact for a discontinuity.*

</figure>

</div>
</div>

Sharper transition, lower ripple, higher degree: the window-design trade in polynomial language. The paper instead smooths the target to $\mathrm{erf}(ka)$ [1, App. D2].


---


# <span class="cat results">Results</span> Delta-function target: resolution costs degree

<div class="columns">
<div class="col">

<figure class="figure">

![w:420](./Meeting_260818/src/Presentation/media/deltafit_response.png)

*Fits of a delta-like spike $e^{-a^2/2\sigma^2}$, $\sigma = 0.1$, at $d = 8, 16, 32$, with the phases of the $d=32$ fit.*

</figure>

</div>
<div class="col">

<figure class="figure">

![w:410](./Meeting_260818/src/Presentation/media/deltafit_mse.png)

*MSE falls from $3.8 \times 10^{-2}$ at $d=4$ to $1.3 \times 10^{-7}$ at $d=32$.*

</figure>

</div>
</div>

A spike of width $\sigma$ needs $d \sim 1/\sigma$: resolution is bought with sequence length, as the DDrf line width shrinks with $N$. This response is eigenstate filtering [1, App. D8].


---


# <span class="cat ongoing">Ongoing</span> Summary

**QSP is response engineering promoted to an exact design principle, and we have used its classical shadow before.**

- The sequence interleaves one fixed signal rotation $W(a)$ with chosen $z$-phases; the response is exactly a bounded polynomial of the signal, and every admissible polynomial is reachable (Theorem 1 of [1]).
- Polynomial design is algorithm design: sign, threshold, cosine, and $1/a$ responses become search, filtering, simulation, and inversion once the signal is an eigenvalue (QSVT).
- Our DDrf apodization solved the same shape of problem: per-cell amplitudes shape the response over detuning, and in both worlds more repetitions buy sharper features.
- Fitting phases is easy at tutorial scale: a numpy script validates against the published lists and fits polynomial, step, and delta targets in seconds, improving systematically with $d$. A possible follow-up: QSP-style exact synthesis for our pulse-envelope design.


---


# References

<div class="ref">

1. J. M. Martyn, Z. M. Rossi, A. K. Tan, I. L. Chuang, *Grand Unification of Quantum Algorithms*. PRX Quantum **2**, 040203 (2021).
2. D. Motlagh, N. Wiebe, *Generalized Quantum Signal Processing*. PRX Quantum **5**, 020368 (2024).
3. G. H. Low, T. J. Yoder, I. L. Chuang, *Methodology of Resonant Equiangular Composite Quantum Gates*. Phys. Rev. X **6**, 041067 (2016).
4. A. Gilyén, Y. Su, G. H. Low, N. Wiebe, *Quantum singular value transformation and beyond*. STOC 2019, `arXiv:1806.01838`.
5. M. H. Levitt, *Composite pulses*. Prog. Nucl. Magn. Reson. Spectrosc. **18**, 61 (1986); S. Wimperis, *Broadband, narrowband, and passband composite pulses*. J. Magn. Reson. A **109**, 221 (1994).
6. QSPPACK, `github.com/qsppack/QSPPACK` (phase-factor solvers; Y. Dong, X. Meng, K. B. Whaley, L. Lin, Phys. Rev. A **103**, 042419 (2021)).
7. pyqsp, `github.com/ichuang/pyqsp` (source of the Appendix-D phase lists of [1]; Laurent method of R. Chao *et al.*, `arXiv:2003.02831`).
8. PennyLane demo, *Function fitting using quantum signal processing*, `pennylane.ai/demos/function_fitting_qsp`.

</div>


---


<!-- _class: closing -->
<!-- _paginate: false -->
<!-- _footer: '' -->

# Thank you

<div class="subtitle">Questions &amp; discussion</div>

<div class="meta">
<span class="author">Donghun Jung</span>
<span class="affil">Paulee Group, Center for Quantum Technology, KIST</span>
</div>
