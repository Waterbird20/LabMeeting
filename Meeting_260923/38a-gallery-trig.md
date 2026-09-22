---
marp: true
theme: serif
math: mathjax
---

# <span class="cat method">Method</span> Cosine and sine

<div class="columns">
<div class="col">

<figure class="figure">

![w:355](media/gallery_cos.png)

*Cosine response at $t=5$, $d=14$, from the phase list of [1, App. D4].*

</figure>

</div>
<div class="col">

<figure class="figure">

![w:355](media/gallery_sin.png)

*Sine response at $t=5$, $d=15$, from the phase list of [1, App. D4].*

</figure>

</div>
</div>

Hamiltonian simulation uses the pair through $e^{-i\mathcal{H}t}=\cos(\mathcal{H}t)-i\sin(\mathcal{H}t)$. So these two angle lists are the entire program, and the degree grows linearly with the simulated time $t$.
