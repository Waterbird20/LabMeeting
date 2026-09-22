---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat intro">Intro</span> Continuous case

<style scoped>
.columns { gap: 1.2rem; }
.columns .col:first-child { flex: 0 0 430px; }
</style>

<div class="columns">
<div class="col">

For two curves the definition reads the same, with the sum replaced by an integral:

$$
(f*g)(x)=\int f(y)\,g(x-y)\,dy .
$$

So hold $f$ still, reverse $g$, slide it by $x$, multiply point by point, and add up over a continuum.

If $f$ and $g$ are two probability densities, then $f*g$ is the density of their sum.

</div>
<div class="col">

<figure class="figure">

<video src="media/conv-continuous.mp4" poster="media/conv-continuous.png" width="720" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
</div>

<!-- TODO (speaker to confirm): this clip is our own manim rebuild of 3Blue1Brown's continuous picture, not a cut of his source (his manimgl build does not run on this machine, and his continuous scene is not in the file our cutter drives). If you want the credit on screen rather than in the reference list, add it to the caption. -->
