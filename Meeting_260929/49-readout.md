---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat method">Method</span> How the prediction is made: a product of experts

<style scoped>.columns { align-items: center; } .columns .col:first-child { flex: 0 0 470px; } .columns ul { margin-top: 0.4em; } .columns li { margin: 0.35em 0; }</style>

<div class="columns">
<div class="col">

$$P(c\,|\,x)=\frac{\prod_bP_b(c)}{\sum_{c'}\prod_bP_b(c')}$$

- $P_b(c)$: two output qubits of QPU $b$, one outcome per digit
- Uniform $P_b$: no effect; confident $P_b$: veto
- No parameters; without communication: naive Bayes

</div>
<div class="col">

<video src="media/poe-readout.mp4" poster="media/poe-readout.png" width="640" controls autoplay loop muted playsinline preload="none"></video>

</div>
</div>

<div class="src">Product of experts: G. E. Hinton, Neural Comput. 14, 1771 (2002).</div>

<!-- Speaker note: the clip uses ILLUSTRATIVE numbers, not a trained model (it says so in red). First example: three mildly agreeing QPUs give a confident product. Second example: a uniform P_b changes nothing, and a confident expert vetoes class 0, which QPU 0 alone would predict. The next two slides show the trained model itself (Fig. 9 of dqml-physics-results.md, and each QPU alone against the product). -->
<!-- src: clip poe-readout from 3. wiki/code/lm-260929-animations/scenes_readout.py (PoEReadout; illustrative expert distributions, labelled in the clip; _verify() checks every printed product, Z, posterior and argmax, that the flat QPU leaves the posterior unchanged, and log P = sum_b log P_b - log Z). Restored 2026-09-29 from _removed/media/ at the speaker's request ("the animation which was previously in Page 38 is worth to stay"). -->
<!-- src: equation and readout: dqml-physics-results.md §0.2 item 5 (two output qubits per QPU give P_b(c) over the four classes; P(c|x) ∝ prod_b P_b(c), product of experts, cross-entropy loss); App. A (log P(c|x) = sum_b log Tr[E_c^(b) rho_b] + const). dqml-physics-plan.md §1 (fixed code {0,1,3,6} -> {00,01,10,11}, same for every QPU; normalised product; "It has no parameters"; "Why the product": with blocks conditionally independent given the class and balanced classes, the product of local posteriors is the global posterior, so the communication-free model is a quantum naive-Bayes classifier over blocks). Product of experts, "veto": G. E. Hinton, Neural Comput. 14, 1771 (2002) (visible source line under the equation). -->
<!-- EDIT-FORWARD: why not one-versus-rest (dqml-physics-plan.md §1): under one-versus-rest QPU c alone scores class c, so without messages the evidence about class c held by the other windows can reach the decision only through the argmax, and the number of QPUs is tied to the number of classes. The product readout lets every QPU answer the same question, so by design the communication gain would then measure synergy only; in practice, on contiguous blocks, it mainly compensates the bounded confidence (poor calibration) of Born-rule class probabilities, not cross-block synergy (dqml-physics-results.md §4.2: "The gain is not synergy between windows"; "It mainly compensates for the calibration limit of Born-rule outputs"). Say this aloud if asked why the readout changed. -->
<!-- EDIT-FORWARD: "exact" for naive Bayes assumes balanced classes (the four classes are near-balanced; mnist1d-eda.md §2). The classical analogue of this readout, the best classifier whose log-probability is a sum of per-block terms, reaches 0.971 (contiguous) / 0.891 (permuted) (dqml-physics-results.md §0.1); quote aloud if asked how good a product of per-block experts can be. -->
