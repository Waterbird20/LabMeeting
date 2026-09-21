---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat method">Method</span> One identity: convolve here, multiply there

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

Read the other way, a product of two records in time is a convolution of their spectra, so a
finite measurement window smears every line by the transform of that window.

A trigonometric polynomial $F(\theta)=\sum_n c_n e^{in\theta}$ is the polynomial slide with $x=e^{i\theta}$: multiplying two such responses **convolves their coefficient lists** $c_n$. That is the case we live in, because our response is a product.

</div>
<div class="col">

<figure class="figure">

<video src="media/conv-to-mult.mp4" poster="media/conv-to-mult.png" width="480" autoplay loop muted playsinline preload="none"></video>

*The same two lists, convolved on the left and multiplied entry by entry on the right. A transform carries one picture into the other.*

</figure>

</div>
</div>
