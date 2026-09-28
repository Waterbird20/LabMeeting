---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat intro">Intro</span> The quantum convolutional neural network (QCNN)

- **Convolution:** quasi-local parametrised unitaries $U$
- **Pooling:** measure qubits; the outcome $m$ controls a rotation $V_m$ on a neighbour
- **Output:** a final unitary $F$, then measure the class

<figure class="figure">

<video src="media/qcnn-pooling.mp4" poster="media/qcnn-pooling.png" width="640" controls autoplay loop muted playsinline preload="none"></video>

</figure>

<div class="src">Cong, Choi, Lukin, Nat. Phys. 15, 1273 (2019).</div>

<!-- Speaker note: schematic on 8 qubits; each pooling layer halves the register (8 -> 4 -> 2), so n qubits need O(log n) pooling layers (this line is also drawn at the bottom of the clip). -->
<!-- src: Cong, Choi, Lukin, arXiv:1810.03787, p. 1-2: "A convolution layer applies a single quasi-local unitary (U_i) in a translationally-invariant manner for finite depth. For pooling, a fraction of qubits are measured, and their outcomes determine unitary rotations (V_j) applied to nearby qubits. ... a fully connected layer is applied as a unitary F on the remaining qubits. Finally, the outcome of the circuit is obtained by measuring a fixed number of output qubits." and "A QCNN to classify N-qubit input states is thus characterized by O(log(N)) parameters." Venue verified: Nat. Phys. 15, 1273-1278 (2019), doi:10.1038/s41567-019-0648-8. -->
<!-- src: clip qcnn-pooling from code/lm-260929-animations/scenes_qml.py (QcnnPooling; _verify checks 8 -> 4 -> 2, pooling on every other alive wire, 2 = log2(8/2) pooling layers). Schematic; the outcome bits are drawn from default_rng(42) for illustration only. -->
<!-- EDIT-FORWARD: Cong et al. measure "a fraction" of the qubits per pooling layer (their SPT example pools by a factor 3); "halves" is the choice in this schematic and in our model (n = 4: two rounds measure qubits 0 and 2, qubits 1 and 3 are read out; dqml-physics-results.md §0.2). The O(log n) count holds for any fixed fraction. -->
<!-- EDIT-FORWARD: Cong et al. say the O(log N) count is of variational parameters (translation-invariant layers); the clip's bottom line states it for pooling layers. -->
