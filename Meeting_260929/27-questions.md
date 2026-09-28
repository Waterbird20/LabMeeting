---
marp: true
theme: serif
math: mathjax
---

# <span class="cat ongoing">Ongoing</span> How we test the objectives

1. **Distributed vs monolithic:** one $12$-qubit QCNN with the same encoding.
2. **Classical communication:** accuracy against no communication (NC); which Boolean function each trained message bit computes.
3. **Topology:** accuracy per transmitted bit for $1\to1$, $2\to1$ (default), $2\to2$ (to senders), $2\to3$ (broadcast).
4. **Open, bits vs qubits:** send the pooled qubits (QC) or only their outcomes (CC), at the same cost and location.
5. **Open, lottery tickets:** remove random message edges, retrain: can a sparser graph do better? (not run yet)

<div class="src">Frankle and Carbin, ICLR (2019).</div>
