---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
p { margin: 0.25em 0; }
figure.figure { margin: 0.1em auto; }
.two { display: flex; gap: 1.4rem; margin-top: 0.2em; }
.two .col { flex: 1 1 0; min-width: 0; }
</style>

# <span class="cat method">Method</span> Two things are convolved

<figure class="figure">

![w:880](media/signal-flow-summary.png)

</figure>

<div class="two">
<div class="col">

**1. The sources.** Many spins each emit their own response $M_j(\omega)$. The signal is their product $S=\prod_j M_j$, which is a convolution of their coefficient lists, so what we measure is already a convolved signal.

</div>
<div class="col">

**2. The processing.** The sequence we run is a kernel $K(\omega)$, so the read-out is the convolution $S*K$ once more. This one we design. Its coefficients are our pulses, so it can exclude what we do not want, or make the response we intended.

</div>
</div>
