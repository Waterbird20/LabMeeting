---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
.columns { gap: 1.2rem; }
.columns .col:first-child { flex: 0 0 420px; line-height: 1.44; }
section p { margin: 0.35em 0; }
mjx-container[display] { margin: 0.2em 0 !important; }
</style>

# <span class="cat method">Method</span> Convolution theorem

$$
\widehat{f*g}=\hat f\cdot\hat g
\qquad\Longleftrightarrow\qquad
\widehat{f\cdot g}=\hat f*\hat g .
$$

<div class="columns">
<div class="col">

A convolution treats all positions alike, so it commutes with shifts, and the complex
exponentials $e^{i\omega x}$ are its eigenfunctions. In that basis the operation is
**diagonal**, and diagonal means multiply entry by entry.

Read the other way, a product of two records in time is a convolution of their spectra.

For example, a trigonometric polynomial $F(\theta)=\sum_n c_n e^{in\theta}$ is the polynomial slide with $x=e^{i\theta}$. So multiplying two such responses **convolves their coefficient lists**, and this is our case.

</div>
<div class="col">

<figure class="figure">

<video src="media/conv-identity.mp4" poster="media/conv-identity.png" width="660" controls autoplay loop muted playsinline preload="none"></video>

*Convolving $a=(1,2,3)$ with $b=(4,5,6)$ on the left, multiplying $|\hat a(\theta)|$ by $|\hat b(\theta)|$ on the right: the dashed curve, the transform of the convolution, lands exactly on the product. The panels plot moduli, since $|\hat a\hat b|=|\hat a|\,|\hat b|$.*

</figure>

</div>
</div>
