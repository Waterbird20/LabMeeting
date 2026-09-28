---
marp: true
theme: serif
math: mathjax
---

# <span class="cat method">Method</span> Nonlinearity from measurement, depth from pooling

<div class="columns">
<div class="col">

**Non-selective measurement: linear**

$$P(c)=\operatorname{Tr}\big[\Pi_c\,\mathcal E(\rho(x))\big]=\operatorname{Tr}\big[E_c\,\rho(x)\big]$$

- Gates, measurements and feed-forward, averaged over outcomes: one channel $\mathcal E$
- Nonlinear in $x$ only through the embedding $x\mapsto\rho(x)$

</div>
<div class="col">

**Selective measurement: nonlinear**

$$\rho_\mu=\frac{M_\mu\,\rho\,M_\mu^\dagger}{\operatorname{Tr}\big[M_\mu\,\rho\,M_\mu^\dagger\big]}$$

- Nonlinear in $\rho$ through the normalisation
- A threshold $\hat m>\tau$ of a $K$-shot estimate: nonlinear, needs many copies

</div>
</div>

**Depth:** each pooling layer builds new features from the previous outcomes.

<div class="src">Cong, Choi, Lukin, Nat. Phys. 15, 1273 (2019).</div>

<!-- Speaker note: machine learning needs a nonlinear function of the input and several layers of features; in a QCNN both come from the pooling layers, but the nonlinearity needs care. Left: unitaries, mid-circuit measurements with classical feed-forward and tracing out the measured qubits together form one quantum channel E; Pi_c is the readout projector, E_c = E^dagger(Pi_c) is the POVM element of the whole circuit (Pi_c in the Heisenberg picture), so P(c) is linear in rho(x). Right: for measurement operators M_mu the post-measurement state is nonlinear in rho through the normalisation; the non-selective state sum_mu M_mu rho M_mu^dagger is linear again. The bit s = 1 if m_hat > tau, with m_hat estimated from K shots, is a genuinely nonlinear function of rho. Cong et al.: "nonlinearities in QCNN arise from reducing the number of degrees of freedom". -->
<!-- src: linearity P(c) = Tr[Pi_c E(rho)] = Tr[E_c rho], E_c = E^dagger(Pi_c) a POVM: dqml-physics-results.md §1 and App. A ("Averaged over outcomes this is a completely positive trace-preserving map"). -->
<!-- src: "A threshold on an estimated probability is a nonlinear function of rho and needs many copies of the state to evaluate": dqml-physics-results.md §9; the links between QPUs use Phi(g/sigma), "the only nonlinearity between QPUs": App. A last paragraph. -->
<!-- src: conditional state rho_mu = K rho K^dag / Tr[K rho K^dag]: standard quantum-measurement postulate (Nielsen and Chuang, Eq. 2.92); not on a wiki page, textbook fact. -->
<!-- src: Cong quote "nonlinearities in QCNN arise from reducing the number of degrees of freedom": arXiv:1810.03787v2 p. 2 ("QCNN circuit model"), verified in the PDF text. -->
<!-- EDIT-FORWARD: the speaker's framing "the nonlinearity comes from intermediate measurement" is exact only for a kept single outcome or a many-shot decision; the QCNN's outcome-averaged Born probabilities are linear in rho(x) (App. A). Say it aloud: our QPUs' output is linear in rho, and the links' threshold on K-shot estimates is the many-copy nonlinearity (§9, App. A). -->
<!-- EDIT-FORWARD: optional spoken example from §9: with the original encoding a threshold on a QPU's own estimated probability helped (own - no test: +2.6 +- 2.7 with 2 rounds, +9.7 +- 4.7 with 4 rounds, 3/3 seeds each; note the §9 runs are the measure-and-reset variant with L = 5, not the reference model), with the revised Fourier encoding it added nothing measurable (+0.1 +- 1.0, 3/5 seeds). -->
