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
