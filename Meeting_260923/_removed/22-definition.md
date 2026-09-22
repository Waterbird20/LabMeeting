---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

# <span class="cat intro">Intro</span> Flip one list, slide it, add up the overlap

$$
(f*g)[n]=\sum_{k} f[k]\,g[n-k],
\qquad\qquad
(f*g)(x)=\int f(y)\,g(x-y)\,dy .
$$

<div class="columns">
<div class="col">

The minus sign is the **flip** and $n$ (or $x$) is the **slide**. At each displacement the two
lists are multiplied entry by entry and the products are added, which gives one number.

The only rule is that the two indices must add up to the output index, since $k+(n-k)=n$.
Everything else is bookkeeping.

The output is longer than either input, because two lists of length $3$ overlap at $5$
distinct displacements.

</div>
<div class="col">

<figure class="figure">

<video src="media/simple-example.mp4" poster="media/simple-example.png" width="500" autoplay loop muted playsinline preload="none"></video>

*One list is reversed and slid across the other, and each frame is one output entry. Here $(1,2,3)*(4,5,6)$ at displacement $n=3$ gives $2\cdot 6+3\cdot 5=27$.*

</figure>

</div>
</div>

<!-- TODO: remove this slide. -->