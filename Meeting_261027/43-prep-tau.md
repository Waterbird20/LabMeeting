---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
figure.figure { margin: 0.1em 0; }
.columns { gap: 0.6rem; align-items: flex-start; }
.columns .col:first-child { flex: 0 0 820px; }
ul { margin-top: 0.2em; }
</style>

# <span class="cat strategy">Strategy</span> DQ preparation: proper $\tau$

One common target means no overlap rule; $\tau$ is chosen for a short sequence with a small band error.

<div class="columns">
<div class="col">

<figure class="figure">

![w:820](media/ch3-prep-progression.png)

</figure>

</div>
<div class="col">

<figure class="figure">

![w:330](media/ch3-prep-newton.png)

</figure>

</div>
</div>

- For short $\tau$ the arbitrary family is shorter than every isolated solution, and both families have a smaller band error.
- Below $\tau\approx10$ ns the duration hardly depends on $\tau$ (composite pulses as $\tau\to0$), so we take $\tau=10$ ns.
- At fixed $\tau$, damped Newton converges from random starts in 6 to 20 steps.
