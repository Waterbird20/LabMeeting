---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
section { font-size: 19px; }
.columns { gap: 1.2rem; }
.columns .col:first-child { flex: 0 0 470px; line-height: 1.38; }
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

A convolution treats all positions alike, so it commutes with shifts, and the complex exponentials are its eigenfunctions. Convolve one with $g$ and it comes back unchanged, times a number:

$$
\big(e^{i\omega\,\cdot}*g\big)(x)=\int e^{i\omega(x-y)}g(y)\,dy=\hat g(\omega)\,e^{i\omega x}.
$$

The eigenvalue is $\hat g(\omega)$, one number per frequency, so convolving with $g$ is diagonal in that basis, and diagonal means multiply entry by entry.

Read the other way, a product of two records in time is a convolution of their spectra.

For example, a trigonometric polynomial $F(\theta)=\sum_n c_n e^{in\theta}$ is the polynomial slide with $x=e^{i\theta}$. So multiplying two such responses **convolves their coefficient lists**, and this is our case.

</div>
<div class="col">

<figure class="figure">

<video src="media/conv-identity.mp4" poster="media/conv-identity.png" width="670" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
</div>
