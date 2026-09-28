---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>figure.figure { margin: 0 -30px; }</style>

# <span class="cat method">Method</span> The whole model on one page

<figure class="figure">

![w:1110](media/model-overview.png)

</figure>

- $3$ QPUs $\times$ $n=4$ qubits, $14$ features each, $L=8$ trainable layers
- Two rounds: each QPU measures one qubit mid-circuit and receives one message bit (dashed: classical only).

<!-- Speaker note: dashed lines are the classical links; only classical bits are exchanged between the QPUs. Each message bit is computed by a decision function from the other two QPUs' estimated probabilities; the product of the three P_b(c) is the prediction. (Former caption: "Three QPUs of n = 4 qubits each. Dashed lines are the classical links: only classical bits are exchanged between the QPUs.") -->
<!-- src: figure = dqml-physics-results.md Fig. 1 panel (a) (3. wiki/projects/dqml/figs/dqml-physics-results-model.png, rendered by 3. wiki/code/dqml-phys-diagram/sim.py). Since 2026-09-28 (terminology pass) media/model-overview.png is rendered by the copy 3. wiki/code/lm-260929-animations/fig_model_overview.py with the glossary labels ("partitioned into feature blocks", "phase on |b(m)>", "decision function (i,j)->k", "message bit", "round 1: same decision functions"); drawing and numbers unchanged; panel (a) cropped below its title line. The wiki figure and sim.py are untouched. Shown at w:1110 with a -30 px side margin so it spans the slide. -->
<!-- src: dqml-physics-results.md §0.2 steps 1-5: 3 QPUs x 4 qubits; 14-feature windows (§0.1); X_m, m = 0..7; L = 8 brick-wall layers; two pooling rounds measure qubits 0 and 2, qubits 1 and 3 remain; links (i,i+1) -> i+2, each QPU receives one bit per round; two output qubits give P_b(c) over four classes (digits 0, 1, 3, 6); product of experts P(c|x) ∝ prod_b P_b(c); cross-entropy loss. -->
<!-- EDIT-FORWARD: the QPU boxes now say "phase on |b(m)>" (the phase-carrying basis state of slide 44; the wiki's own figure still says "phases paired"). -->
