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
