---
marp: true
theme: serif
math: mathjax
---

# <span class="cat results">Results</span> A monolithic QPU does worse (still trying)

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
