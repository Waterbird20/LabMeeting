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
