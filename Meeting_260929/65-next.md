---
marp: true
theme: serif
math: mathjax
---

<style scoped>.callout { margin: auto 0; font-size: 1.3em; line-height: 1.55; padding: 0.9em 1.4em; } .callout p { margin: 0; }</style>

# Take-home

<div class="callout">

Three $4$-qubit QPUs reach about $0.9$ on four digits, but the encoding did most of the work: communication needs a task where it must matter.

</div>

<!-- 2026-09-29 (speaker: "remove todo list"): the numbered next-steps list was removed and the slide retitled from "Next steps" (Ongoing) to "Take-home"; only the conclusion callout remains, unchanged. File name kept (65-next.md) so the order and other agents' references are unaffected. -->
<!-- Speaker note: one message to take home. If asked what comes next (the removed list, speaker-only): (1) an inherently distributed task, equality by quantum fingerprinting, measuring worst-case error against message size for classical (CC) against quantum communication (QC); (2) classical messages of increasing precision in place of the qubit: one bit, <Z>, the Bloch vector, the full reduced state (the last reproduces QC for our product readout, so the question is the cost in classical bits and copies); (3) other networks: remove message edges at random and retrain (cf. lottery tickets, found by magnitude pruning, Frankle and Carbin), and 3 x 4 against 4 x 3 (QPUs x qubits). Also planned: more seeds for QC against single-shot CC on permuted features, where QC wins by 2.4 +- 0.9 percentage points on only five seeds. -->

<!-- terminology pass 2026-09-28 (GLOSSARY.md): "ladder" = classical messages of increasing precision; "link pruning" = removing edges of the communication graph (Frankle and Carbin 2019 prune by magnitude); "channel comparison" / "quantum channel vs measured bits" = QC vs CC with a single-shot outcome sent; "permuted windows" = permuted features; "truly distributed" = inherently distributed. -->
<!-- src: planned items: dqml-physics-results.md §13 "Planned" (communication on permuted windows, more seeds; number of QPUs at 12 qubits in total, 3 x 4 and 4 x 3; the ladder of classical messages of §12.3). Ladder levels: §12.3 table (one bit; <Z>; Bloch vector; full reduced state, 15 real numbers); level 4 "reproduces the quantum channel for this readout". Permuted windows, quantum channel vs measure once: +2.4 +- 0.9 points, five seeds (seeds 0-4): §8.1. Lottery tickets / pruning: speaker's question (BRIEF outline item 7), not run; Frankle and Carbin, ICLR 2019 (see REFS-dqml.md, 27-questions.md). Fingerprinting, worst-case error: 64-fingerprint.md and 64a-fingerprint-protocol.md (our proposal). -->
<!-- src: callout: about 0.9 = 0.903 +- 0.007 (§4.3), 0.908 +- 0.002 (§2.4), 0.8895 on held-out seeds (§3); encoding did most of the work: §2.4 (no communication 0.738 -> 0.865; linear classifier on the encoded states 0.773 -> 0.943) and 62-too-good.md. -->
<!-- EDIT-FORWARD: §13 also lists a confirmation with larger registers next to 3 x 4 against 4 x 3; omitted because the deck uses only n = 4 results. The five "Decisions for the human" of §13 (accuracy target, evaluation shots, more training signals, multi-amplitude encodings, amplitude encoding with QFT) are not on the slide. -->
