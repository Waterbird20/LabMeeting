---
marp: true
theme: serif
math: mathjax
---

# <span class="cat intro">Intro</span> Skepticism: four reasons for doubt

1. **Classical baselines win overall** on $160$ small datasets; entanglement often does not help.
2. **Barren plateaus:** random circuits have gradients exponentially small in the qubit number.
3. **Trainable often means simulable:** classical surrogates match QCNNs up to $1024$ qubits.
4. **The fine print:** data loading and readout can erase the speed-up.

<div class="src">[1] Bowles et al., arXiv (2024). [2] McClean et al., Nat. Commun. (2018). [3] Cerezo et al., Nat. Commun. (2025); Bermejo et al., PRX Quantum (2026). [4] Aaronson, Nat. Phys. (2015).</div>
<!-- src: volumes and pages are in the src comments below and on the references slides; shortened here to cut words. -->

<!-- Speaker note [1]: Bowles, Ahmed and Schuld tested 12 popular quantum classifiers on 6 binary tasks (160 datasets); out-of-the-box classical models outperform them overall, and removing entanglement often gives as good or better accuracy, so on such tasks "quantumness" may not be the crucial ingredient. [2]: for a wide class of randomly initialised parametrised circuits the gradient is exponentially small in the number of qubits, so a large generic circuit started at random cannot be trained with a reasonable number of shots. [3]: in the cases collected so far, the structure that provably avoids barren plateaus confines the loss to a small subspace that a classical, or quantum-enhanced classical, algorithm can simulate; a randomly initialised QCNN only uses low-body observables, and a classical surrogate fed with classical shadows of the data matches or beats it on standard benchmarks. [4]: exponential speed-ups for learning assume the classical data can be loaded into a quantum state efficiently and that the output need not be read out in full. -->
<!-- src [1]: 1. raw/papers/bowles2024betterclassicalsubtleart (arXiv:2403.07059v2 abstract): "systematically tests 12 popular quantum machine learning models on 6 binary classification tasks used to create 160 individual datasets. We find that overall, out-of-the-box classical machine learning models outperform the quantum classifiers. Moreover, removing entanglement from a quantum model often results in as good or better performance, suggesting that 'quantumness' may not be the crucial ingredient for the small learning tasks considered here." Also 3. wiki/papers/@bowles2024better.md, Main results. "Entanglement often does not help" on the slide = "removing entanglement ... often results in as good or better performance". -->
<!-- src [2]: McClean, Boixo, Smelyanskiy, Babbush, Neven, Nat. Commun. 9, 4812 (2018), abstract (web-verified 2026-09-28): "for a wide class of reasonable parameterized quantum circuits, the probability that the gradient along any reasonable direction is non-zero to some fixed precision is exponentially small as a function of the number of qubits". -->
<!-- src [3]: Cerezo, Larocca, Garcia-Martin, Diaz, Braccia, Fontana, Rudolph, Bermejo, Ijaz, Thanasilp, Anschuetz, Holmes, "Does provable absence of barren plateaus imply classical simulability?", Nat. Commun. 16, 7907 (2025), doi:10.1038/s41467-025-63099-6 (web-verified): "a wide class of loss landscapes which provably do not exhibit barren plateaus can be simulated using either a classical algorithm or a quantum-enhanced classical algorithm"; their answer is "Yes and No" (hence "often" on the slide). Bermejo, Braccia, Rudolph, Holmes, Cincio, Cerezo, PRX Quantum 7, 020304 (2026), arXiv:2408.12739 (web-verified): randomly initialised QCNNs "can only operate on the information encoded in low-bodyness measurements"; classical surrogates matched or outperformed standard QCNNs on all tested benchmarks, up to 1024 qubits. -->
<!-- src [4]: Aaronson, "Read the fine print", Nat. Phys. 11, 291-293 (2015), doi:10.1038/nphys3272 (web-verified): caveats of HHL-type quantum ML speed-ups (loading the input vector, sparsity and conditioning of the matrix, the output is a quantum state). The slide names two of them. -->
<!-- EDIT-FORWARD: point 3 applies to our model too in spirit: 4-qubit QPUs are trivially simulable; say aloud that the project does not aim at an advantage (next slide). -->
