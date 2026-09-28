---
marp: true
theme: serif
math: mathjax
---

# <span class="cat method">Method</span> Communication between QPUs: $\mathrm{NC}\subset\mathrm{CC}\subset\mathrm{QC}$

<style scoped>table { width: 100%; } td, th { padding: 0.18em 0.5em; } .callout { margin: 0.3em 0; padding-top: 0.35em; padding-bottom: 0.35em; } p { margin: 0.35em 0; } ul { margin: 0.3em 0; }</style>

| communication | exchanged between QPUs | entanglement between QPUs |
|---|---|---|
| **NC**: no communication | nothing | none |
| **CC**: classical communication (LOCC), our model | message bits that control gates | none |
| **QC**: quantum communication | the pooled qubits | yes |
| monolithic QPU | (no partition) | yes |

- **Deferred measurement:** a qubit used only as a control is worth one bit, $\mathrm{Tr}_q\big[CU\,\rho\,CU^\dagger\big]=\sum_{\mu=0,1}U^\mu\langle\mu|\rho|\mu\rangle_q\,U^{\mu\dagger}$.
- **Holevo bound:** without shared entanglement, one qubit carries at most one bit.

<div class="callout">

QC helps only if the receiving QPU processes the received qubits coherently.

</div>

<div class="src">Nielsen and Chuang (2010), §4.4, §12.1. QC in Hwang et al. (2025): non-local two-qubit gates.</div>

<!-- Speaker note: NC is local operations only (LO); CC is local operations and classical communication (LOCC); with CC the joint state is a product state for every measurement record. The chain continues NC ⊂ CC ⊂ QC ⊂ monolithic QPU (global unitaries, no partition). A QPU's class probability P(c) = Tr[E_c rho] (E_c the POVM element of class c) is linear in its encoded state rho; in our CC model the only nonlinearity between QPUs is a received message bit, a threshold of other QPUs' estimated outcome probabilities. In Hwang et al., QC is realized by non-local two-qubit gates between QPUs; here the pooled qubits themselves are transmitted. Hence QC can help only if the receiving QPU does more with the received qubits than use them as controls. -->
<!-- src: ladder: 3. wiki/projects/dqml/dqml-task-design.md §0 (LO / LOCC / global; "the joint state across processors is always separable"); quantum channel rung: dqml-physics-results.md §8 and §8.1 (qubits sent to the neighbouring QPU at the measurement step); product state before hand-over: dqml-physics-results.md App. E; linearity P(c) = Tr[E_c rho] and "the only nonlinearity between QPUs": App. A; deferred measurement, Holevo bound and the conclusion (a quantum channel helps only through operations on the received qubit that do not commute with Z): App. D. -->
<!-- 2026-09-29 integrator: source line year set to the 10th-anniversary edition (2010), as on the references slide; the section numbers are the same in the 2000 edition. -->
<!-- src: the deferred-measurement identity is checked numerically in 3. wiki/code/lm-260929-animations/scenes_dqml.py _verify() (residual < 1e-12, seed 42). -->
<!-- EDIT-FORWARD: "a product state for every measurement record" (speaker note) is exact for the model (product encoding, local gates, bits only); averaged over the bits the joint state is separable, not product. Say "separable" if someone pushes. -->
