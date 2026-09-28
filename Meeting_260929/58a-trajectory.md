---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>.columns .col:first-child { flex: 0 0 850px; } .columns { align-items: center; } li { margin-bottom: 0.6em; }</style>

# <span class="cat results">Results</span> The coefficients move early, then settle

<div class="columns">
<div class="col">

<figure class="figure">

![w:840](media/dqml-physics-results-trajectory-round1.png)

</figure>

</div>
<div class="col">

Seed 0 (test accuracy $0.905$, 4 digits):

- Four of the six decision functions keep their initial Boolean function; shown are the two that change.
- Left: switches (grey lines), ends as $\lnot m_i$. Right: becomes NOR, then constant.
- Trained, but not dramatically (one seed).

</div>
</div>
