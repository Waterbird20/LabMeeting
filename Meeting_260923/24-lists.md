---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

# <span class="cat intro">Intro</span> Forget the dice: any two lists convolve the same way

<div class="columns">
<div class="col">

Take two plain lists, $(1,2,3)$ and $(4,5,6)$, with no probabilities attached. Reverse one, slide it, multiply what faces each other, add:

$$
(1,2,3)*(4,5,6)=(4,\;13,\;28,\;27,\;18).
$$

At displacement $n=3$ the window holds $2\cdot 6+3\cdot 5=27$. Two lists of length $3$ overlap at $5$ displacements, so the output has length $5$.

This is the discrete definition, and the continuous one is the same sentence with an integral in place of the sum:

$$
(f*g)(x)=\int f(y)\,g(x-y)\,dy .
$$

</div>
<div class="col">

<figure class="figure">

<video src="media/simple-example.mp4" poster="media/simple-example.png" width="520" autoplay loop muted playsinline preload="none"></video>

*One list reversed and slid across the other; each frame is one output entry.*

</figure>

</div>
</div>
