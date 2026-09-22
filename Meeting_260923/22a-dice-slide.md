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

# <span class="cat intro">Intro</span> Flip and slide

<div class="columns">
<div class="col">

Write the two distributions as lists. Reverse one of them, slide it along the other, and at each displacement multiply the entries that face each other and add the products. That single number is $P(Z=n)$ for that displacement.

$$
(f*g)[n]=\sum_{k} f[k]\,g[n-k].
$$

The minus sign is the **flip**, and $n$ is the **slide**. The only rule is that the two indices add up to the output index, $k+(n-k)=n$.

Two lists of length $6$ overlap at $11$ displacements, the sums $2$ to $12$.

</div>
<div class="col">

<figure class="figure">

<video src="media/dice-slide.mp4" poster="media/dice-slide.png" width="640" controls autoplay loop muted playsinline preload="none"></video>

*The second row is reversed and slid across the first. The pairs that line up are exactly the anti-diagonal of the table, now read as a moving window.*

</figure>

</div>
</div>
