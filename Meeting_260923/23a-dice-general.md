---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat intro">Intro</span> Convolution of $(a_i)$ and $(b_i)$

Replace the weights by symbols, $P_X(i)=a_i$ and $P_Y(j)=b_j$. The sliding window collects the pairs whose indices add up to $n$, and that is the whole definition:

$$
(a*b)_n=\sum_{i+j=n}a_i\,b_j=\sum_{i}a_i\,b_{n-i}.
$$

**Up to now we have only added two independent random variables.** Convolution is what that addition looks like in the distributions.

<div class="columns">
<div class="col">

<figure class="figure">

<video src="media/dice-general.mp4" poster="media/dice-general.png" width="520" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
<div class="col">

<figure class="figure">

<video src="media/dice-formula.mp4" poster="media/dice-formula.png" width="520" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
</div>
