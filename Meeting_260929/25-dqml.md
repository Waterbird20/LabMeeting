---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat intro">Intro</span> Distributed QML: one QCNN partitioned across several QPUs

<style scoped>.columns .col:first-child { flex: 0 0 460px; }</style>

<div class="columns">
<div class="col">

- Entanglement between QPUs: proof-of-principle.
- Classical feed-forward: fast enough.
- A mid-circuit outcome on one QPU controls a gate on another.

</div>
<div class="col">

<figure class="figure">

<video src="media/dqml-split.mp4" poster="media/dqml-split.png" width="640" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
</div>

**Two 4-qubit QPUs, synthetic binary task:** classical communication beats no communication by $5$ to $11$ points and matches quantum communication.

<div class="src">Hwang et al., Quantum Sci. Technol. 10, 015059 (2025); three-QPU extension: CY (previous DQML project). See also Chinzei et al., PRR 6, 023042 (2024).</div>

<!-- Speaker note: today's QPUs are small; entanglement distribution between QPUs is still at the proof-of-principle stage, but classical feed-forward between QPUs fits within the coherence time. Hwang et al. partition a QCNN across QPUs that exchange only such classical bits. With two 4-qubit QPUs on a synthetic task, classical communication beat no communication by 5 to 11 percentage points and matched quantum communication (non-local two-qubit gates in their work) within one standard deviation. The previous DQML project, led by CY, extended this to three QPUs; my code started from it. Clip: a 12-qubit QCNN partitioned across three QPUs of n = 4 qubits; the inter-QPU gates are removed, and only measurement outcomes (gold classical bits) control gates on another QPU. -->
<!-- src: Hwang intro (p. 1-2 of the PDF): "achieving high-fidelity entanglement distribution remains a challenge and is currently limited to proof-of-principle experiments ... CC can be reliably implemented within the coherence time"; scheme: 3. wiki/papers/@hwang2024distributed.md §1; numbers: @hwang2024distributed.md §3 Table 1 (CC - NC = 5.4 at L=3 to 11.2 at L=20; CC vs QC within one std at every L), two 4-qubit QPUs, 8-dim synthetic binary clusters (App. B; "synthetic binary task" on the slide). CC = QC holds with the trained interpret function (§4 of the note). -->
<!-- src: CY's code: ~/DQML/README.md §1 (3 processors x 3 qubits, binary classification, synthetic 9-dim clusters); BRIEF.md timeline (Feb 2026). -->
<!-- src: Chinzei: 3. wiki/papers/@chinzei2024splitting.md summary and §3 (a QCNN split into parallel branches on translationally symmetric data to save measurement shots; branches exchange nothing after the split; no mid-circuit measurement). PRR = Phys. Rev. Research. -->
<!-- src: clip dqml-split from code/lm-260929-animations/scenes_dqml.py (DqmlSplit); schematic. -->
<!-- EDIT-FORWARD: the acronyms NC / CC / QC are defined on the next slide (the communication table); this slide says the words in full. -->
<!-- EDIT-FORWARD: Chinzei et al. is only in the source line; if asked: they split a QCNN into parallel branches on translationally symmetric data to save shots, with no communication between branches. -->
<!-- EDIT-FORWARD: the speaker's phrase "quantum links are slow and lossy" is not in a vault source; the slide uses Hwang's wording (entanglement distribution is proof-of-principle; classical feed-forward fits within coherence time). Replace if you have a hardware reference. -->
<!-- EDIT-FORWARD: Hwang's "matched quantum communication" holds with their trained interpret function on the joint outcomes; with a parity readout the order is NC < CC < QC (@hwang2024distributed.md §4, Table 2). Say this aloud if asked. -->
<!-- EDIT-FORWARD: CY's code used three QPUs of 3 qubits on a synthetic 9-dimensional binary task (~/DQML/README.md §1); say it aloud if useful. -->
<!-- EDIT-FORWARD: the clip routes the bits schematically (round 1 down the stack, round 2 back up); the model's own two-input links are on the link slides. -->
