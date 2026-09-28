---
marp: true
theme: serif
math: mathjax
---

# <span class="cat intro">Intro</span> Optimism: reasons to keep going

1. **Deep circuits can generalize:** double descent, also on MNIST-1D, while trainable.
2. **QCNNs have no barren plateaus:** the gradient variance decays only polynomially.
3. **Provable separations,** for special problems such as discrete logarithms.
4. **Spectral methods suit quantum computers:** the quantum Fourier transform is efficient.

<div class="callout">

Our aim: understand what communication does, not a quantum advantage.

</div>

<div class="src">[1] Kempkes et al. (2026). [2] Pesah et al., PRX (2021). [3] Liu et al., Nat. Phys. (2021); Huang et al., Science (2022). [4] Belis et al. (2026).</div>
<!-- src: full entries (arXiv:2607.21409, arXiv:2603.24654; volumes and pages in the src comments below) are on the references slides; shortened here to cut words. -->

<!-- Speaker note [1]: gradient-trained parametrised circuits can show double descent, also on MNIST-1D: the test loss peaks at the interpolation threshold, where the number of parameters p matches the N K training targets, and falls again beyond it; the optimism is cautious, it holds only while the circuit stays trainable. [2]: the gradient variance of a randomly initialised QCNN decays only polynomially in the number of qubits, so it remains trainable. [3]: from classical data, a quantum kernel estimated on a fault-tolerant quantum computer learns a discrete-logarithm task on which no efficient classical learner beats random guessing, if that logarithm is hard (Liu et al.); a quantum memory learns some properties of quantum data from exponentially fewer experiments (Huang et al.). [4]: kernels and convolutions are filters in Fourier space, deep networks learn low frequencies first (spectral bias), and the QFT is efficient for every finite Abelian group; our discrete Fourier transform (DFT) encoding is such a spectral choice. Callout: for classical data like MNIST-1D the aim is to understand what communication does, not a quantum advantage. -->
<!-- src [1]: 1. raw/papers/kempkes2026cautious (arXiv:2607.21409v2) abstract: "gradient-based PQCs can exhibit improved performance on unseen data as model size increases, displaying the phenomenon of double descent ... our finding that deeper parameterized quantum circuits do not necessarily exhibit degraded performance provides reasons for cautious optimism." Fig. 1: MNIST-1D, Fashion MNIST and a synthetic regression; interpolation threshold p = N K (N training points, K outputs). Discussion: caution is "twofold. First, our analysis is restricted to trainable PQCs ... Second ... our results do not imply that overparameterized PQCs outperform underparameterized ones." -->
<!-- src [2]: Pesah, Cerezo, Wang, Volkoff, Sornborger, Coles, PRX 11, 041011 (2021), arXiv:2011.02966 abstract (web-verified): "the variance of the gradient vanishes no faster than polynomially, implying that QCNNs do not exhibit barren plateaus". -->
<!-- src [3]: Liu, Arunachalam, Temme, Nat. Phys. 17, 1013-1017 (2021), arXiv:2010.02174 (web-verified): "no classical learner can classify the data inverse-polynomially better than random guessing, assuming the widely-believed hardness of the discrete logarithm problem"; SVM with a kernel estimated on a fault-tolerant quantum computer, classical access to data. Huang et al., Science 376, 1182-1186 (2022), arXiv:2112.00778 (web-verified): "quantum machines can learn from exponentially fewer experiments than those required in conventional experiments"; experiments with up to 40 superconducting qubits. The visible line gives only Liu et al.'s result; Huang et al. is in the speaker note. -->
<!-- src [4]: 3. wiki/papers/@belis2026spectral.md §1, §3, §4 (QFT efficient for every finite Abelian group; kernels, CNNs and spectral bias as spectral filters); our embedding: dqml-quantum-design.md §2.3.1, @belis2026spectral.md "Relevance to the DQML project". -->
<!-- EDIT-FORWARD: Belis et al. also warn that the Born rule filters the amplitudes, not the model, and that some of their own filters dequantise (@belis2026spectral.md §2, §4); mention if asked. -->
<!-- EDIT-FORWARD: [2] and the skepticism slide's point 3 pull in opposite directions (Bermejo et al. use exactly this trainability to simulate QCNNs classically); say it aloud. -->
<!-- EDIT-FORWARD: the callout is one of two in the section (the other is on the communication slide); the integrator may remove it. -->
