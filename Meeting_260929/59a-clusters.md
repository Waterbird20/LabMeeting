---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>.columns { align-items: center; } .columns .col:first-child { flex: 0 0 620px; } ul { margin-top: 0.5em; } li { margin: 0.25em 0; }</style>

# <span class="cat results">Results</span> Only a decision boundary that intersects the data can learn

<div class="columns">
<div class="col">

<figure class="figure">

<video src="media/d-sweep.mp4" poster="media/d-sweep-xor.png" width="560" controls autoplay loop muted playsinline preload="none"></video>

</figure>

- XOR-type, $d=-0.875$: boundary outside the data, $s=0$ always; no gain (accuracy $0.878$, 4 digits).
- XOR-type, $d=-0.5$ (right): training forms clusters; the bit asks "0 or 6, not 1 or 3?" (accuracy $0.900$).
- Gradient only where $|g|\lesssim\sigma_g$ (shot-noise width): a boundary outside the data never learns.

</div>
<div class="col">

<figure class="figure">

![h:500](media/dqml-physics-results-clusters-b.png)

</figure>

</div>
</div>
