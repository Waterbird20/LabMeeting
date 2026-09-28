---
marp: true
theme: serif
math: mathjax
---

<style scoped>
ol { font-size: 1.12em; line-height: 1.6; margin-top: 0.6em; }
ol li { margin: 0.75em 0; }
ol li::marker { color: var(--primary); font-family: 'Spectral', serif; font-weight: 600; }
</style>

# <span class="cat strategy">Strategy</span> Three objectives

1. **Distributed vs monolithic:** three 4-qubit QPUs against one 12-qubit QPU.
2. **Classical communication:** what do the bits sent between QPUs add, and what do they compute?
3. **Topology:** does it matter which QPU sends to which?

<div class="src">Setting: Hwang et al., Quantum Sci. Technol. 10, 015059 (2025).</div>

<!-- src: objectives = the speaker's outline (BRIEF.md, item 0: "multiple QPUs vs one big QPU; classical communication analysis; network topology") and the PI discussion list, dqml-physics-plan.md §0 (items 1 and 4); N = 3 QPUs of n = 4 qubits against one 12-qubit monolithic QPU = dqml-physics-results.md §0.2 ("Single-QPU model") and §8; classical communication (objective 2) = the speaker's "classical communication analysis" (BRIEF.md item 0) and the PI list items 2-3 (the message decision function; find the converged a, b, c, d, then fix a, b, c and vary d), dqml-physics-plan.md §0; communication topologies = §0.2 table and §4.3. Moved on 2026-09-29 from 03-objectives.md to the end of the machine-learning introduction at the speaker's request, and cut to the three goals; the detailed questions (with the NC / CC / QC acronyms) are on 27-questions.md. -->
<!-- Speaker note: (1) the monolithic QPU runs the same kind of convolutional circuit on all 12 qubits; the question is what is lost when the qubits are partitioned across QPUs that cannot share entanglement. (2) Each message bit is the threshold of a trainable decision function of two QPUs' estimated outcome probabilities; we compare with no communication and ask which Boolean function each trained message bit ends up computing. Sending the pooled qubits themselves (quantum communication) is the natural next comparison; it stays an open question in this talk. (3) We vary who sends to whom, and the decision function that computes each message bit; if the class depends on features held by different QPUs, some topologies should carry that information better than others. -->
<!-- Map (checked by the integrator 2026-09-29; 27-questions.md uses the same three labels; updated 2026-09-29 by the structure editor): (1) is answered by 54-one-qpu.md; (2) by 51-accuracy.md and 55-patterns.md (what the bits add), 58-links-trained.md, 59-d-scan.md, 59a-clusters.md and 63-stuck.md (what they compute); (3) by 55-patterns.md and 62-too-good.md. -->
<!-- 2026-09-29 (speaker removed 56-shuffle.md and 57-quantum-channel.md, now in _removed/): objective 2 was "Bits vs qubits: how much of the gain from sending qubits can bits recover?", answered only by 57-quantum-channel.md. It now states the speaker's own objective from the outline, "classical communication analysis" (BRIEF.md item 0), which the remaining results slides answer; bits vs qubits (QC vs CC) is kept as an open question on 27-questions.md. -->
