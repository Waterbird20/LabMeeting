---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
.columns { gap: 1.0rem; }
.columns .col:first-child { flex: 0 0 430px; }
</style>

# <span class="cat intro">Intro</span> Two lists

<div class="columns">
<div class="col">

Take two plain lists, $(1,2,3)$ and $(4,5,6)$, with no probabilities attached. Reverse one, slide it, multiply what faces each other, and add:

$$
(1,2,3)*(4,5,6)=(4,\;13,\;28,\;27,\;18).
$$

For example, at the displacement $n=3$ the window holds $2\cdot 6+3\cdot 5=27$. Two lists of length $3$ overlap at $5$ displacements, so the output has length $5$.

This is the discrete definition. The continuous one is the same sentence with an integral in place of the sum:

$$
(f*g)(x)=\int f(y)\,g(x-y)\,dy .
$$

</div>
<div class="col">

<figure class="figure">

<video src="media/simple-example.mp4" poster="media/simple-example.png" width="640" controls autoplay loop muted playsinline preload="none"></video>

*One list reversed and slid across the other; each frame is one output entry.*

</figure>

</div>
</div>
