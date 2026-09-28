---
marp: true
theme: serif
math: mathjax
---

# <span class="cat results">Results</span> A monolithic QPU does worse (still trying)

<!-- src: dqml-physics-results.md §8, Phase 2g bullet (revised encoding; source phys-phase2g/README.md §7): single 12-qubit QCNN 0.805 ± 0.076 against 0.865 without and 0.908 with communication, 0 of 3 wins in both comparisons, circuit parameters 1656 against 1128 and 1188, 0.60 nats of entanglement entropy between the data blocks (22 % of the maximum after the first circuit layer), linear-classifier accuracy of its own trained encoding 0.864 against 0.946; Phase 2 bullet (original encoding: 1-6 points below three QPUs without communication for every window type); paragraph after the bullets (differs in circuit, pooling and readout; reference = quantum channel with the same crossing budget, decided 2026-09-26). Single-QPU model definition: §0.2 "Other variants" (one QPU holding all 3n qubits, same product-state encoding, one global brick-wall circuit, pooling down to one 2-qubit readout). -->
<!-- terminology pass 2026-09-28 (GLOSSARY.md): "one large QPU" -> monolithic QPU; "clean reference" -> matched baseline; "quantum channel with the same crossing budget" -> QC at the same communication cost and circuit location; "data blocks" -> feature blocks' registers; "linearly readable" -> linear-classifier accuracy. -->

<style scoped>table { font-size: 0.85em; margin: 0.4em 0 0.6em; } th, td { padding: 0.35em 0.8em; } li { margin-bottom: 0.25em; }</style>

| model (phase on $\lvert b(m)\rangle$, 4 digits, three seeds) | circuit parameters | test accuracy |
|---|---|---|
| monolithic $12$-qubit QCNN | $1656$ | $0.805\pm0.076$ |
| three $4$-qubit QPUs, no communication (seeds $0$–$2$) | $1128$ | $0.865$ |
| three $4$-qubit QPUs, classical communication (seeds $0$–$2$) | $1188$ | $0.908$ |

- **Loses on all three seeds,** despite more parameters and entanglement across the feature blocks.
- **Its trained encoded states are worse:** linear classifier $0.864$ vs $0.946$.
- **Not a matched baseline:** circuit, pooling and readout all differ.

<div class="src">QCNN (quantum convolutional neural network): I. Cong, S. Choi and M. D. Lukin, Nature Physics 15, 1273 (2019).</div>

<!-- Speaker note: "0.864 vs 0.946" = linear-classifier accuracy on the monolithic QCNN's own trained encoded states against that of the distributed model (§8). The monolithic QPU holds all 12 qubits with the same encoding, one brick-wall circuit over all of them and pooling layers down to one 2-qubit readout: a standard QCNN (§0.2). CC (2->1) = classical communication, 2 senders -> 1 receiver. "entanglement between the blocks" = 0.60 nats of entanglement entropy between the registers of the three feature blocks (22 % of the maximum after the first circuit layer). The matched baseline is quantum communication (QC): the pooled qubits are sent to the receiving QPU instead of being measured, at the same communication cost and circuit location as the classical message bits; the deck lists it as an open item (bits vs qubits), not as a result. With the original encoding the monolithic QCNN was also the least accurate model, 1-6 points below NC (§8 Phase 2). -->
<!-- 2026-09-29 integration (second round): visible "0.60 nats of entanglement" -> "entanglement across the feature blocks" (no nats on visible slides; the 0.60-nats entanglement entropy stays in the speaker note), and "Matched: quantum communication (QC)" dropped, because the QC comparison slide (57-quantum-channel.md) was removed and the deck no longer shows QC results. -->
<!-- EDIT-FORWARD: the speaker is still working on the one-large-QPU comparison ("still trying"); the numbers above are the Phase 2g runs of 2026-09-26/27 and may be superseded. The spread of the single QCNN (sd 0.076 over three seeds) is large; say that it is unstable rather than uniformly worse. -->
<!-- EDIT-FORWARD: 0.865 and 0.908 are the seeds 0-2 means of §2.4 (the seeds used to choose the design; the held-out seeds 3-7 give 0.878 and 0.890, accuracy slide), while 1128 and 1188 are the parameter counts of §4.3; §8 pairs them as quoted here. The wiki does not state the seeds of the single-QCNN runs beyond "0 of 3 wins". -->
<!-- 2026-09-29 integration: visible task label "4 classes" -> "4 digits" (speaker's standing style: label accuracies with the task, "4 digits"). -->
