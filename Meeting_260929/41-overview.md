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
