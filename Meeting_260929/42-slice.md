---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

<style scoped>.columns .col:first-child { flex: 0 0 440px; } li { margin-bottom: 0.5em; }</style>

# <span class="cat method">Method</span> Feature partition: each QPU encodes $14$ of the $40$ features

<div class="columns">
<div class="col">

- **Contiguous blocks** (cyclic):<br>$0$–$13$, $14$–$27$, $27$–$39$ + $0$
- **Permuted features:** same partition after a fixed random permutation
- One block alone: linear classifier on its encoded state $\le0.67$ (4 digits, chance $0.25$)

</div>
<div class="col">

<figure class="figure">

<video src="media/data-slice.mp4" poster="media/data-slice.png" width="700" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
</div>

<div class="src">MNIST-1D: Greydanus and Kobak, ICML 2024, arXiv:2011.14439.</div>
