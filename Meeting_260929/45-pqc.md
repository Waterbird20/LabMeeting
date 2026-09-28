---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat method">Method</span> The "convolution" is a parametrised quantum circuit

<style scoped>li { margin-bottom: 0.3em; } figure.figure { margin: 0 auto 0.2em; }</style>

<figure class="figure">

![w:840](media/pqc-layer.png)

</figure>

- $\mathrm{IsingZZ}(\varphi)=e^{-i\varphi\,Z\otimes Z/2}$: the only entangling gate
- All angles trainable: $8\times36=288$ of the $434$ parameters of a QPU
- Same gate pattern on every neighbouring pair, angles not shared (unlike a CNN kernel or the translation-invariant QCNN layer)

<div class="src">Hardware-efficient ansatz: Kandala et al., Nature 549, 242 (2017).</div>
