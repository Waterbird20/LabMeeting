---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat method">Method</span> Online data augmentation: a freshly augmented sample every epoch

<style scoped>.columns .col:first-child { flex: 0 0 480px; }</style>

<div class="columns">
<div class="col">

- Each epoch: circular shift $|s|\le2$ + correlated noise $0.1$
- Generalization gap ($n=4$): $0.008$ without communication, $0.035$ with
- Random message bits (same mean) remove about half of the $0.035$

</div>
<div class="col">

<video src="media/augment.mp4" poster="media/augment.png" width="660" controls autoplay loop muted playsinline preload="none"></video>

</div>
</div>

<div class="src">Gaps: seeds 3–7, not used for model selection. Augmentation as in the MNIST-1D generator: Greydanus and Kobak, ICML 2024, arXiv:2011.14439.</div>
