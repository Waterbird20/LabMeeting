---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat method">Method</span> A product of responses is a convolution of coefficients

<div class="callout">

Our response is a product, $M=\prod_j M_j$. In Fourier space that product is a convolution: the coefficient lists of the single-nucleus responses are convolved, one more nucleus is one more convolution.

</div>

<div class="columns">
<div class="col">

Each $M_j(\theta)$ is a function of the swept variable $\theta$ and, for a sequence of $K$ equal units, a trigonometric polynomial $M_j(\theta)=\sum_n c^{(j)}_n e^{in\theta}$ of degree $K-1$. Multiplying the responses is the polynomial slide with $x=e^{i\theta}$:

$$
M_1M_2=\sum_n c_n e^{in\theta},\qquad c_n=\sum_k c^{(1)}_k\,c^{(2)}_{n-k}=\big(c^{(1)}*c^{(2)}\big)_n .
$$

The degrees add, the lists get longer, and the measured curve is the product of the dips. Chapter 3 turns this around: a QSP sequence multiplies degree-one factors on purpose, so it convolves its coefficient list one tap at a time.

</div>
<div class="col">

<figure class="figure">

<video src="media/response-product.mp4" poster="media/response-product.png" width="500" autoplay loop muted playsinline preload="none"></video>

*Two nuclei give two dips $M_1(\theta)$ and $M_2(\theta)$; the measured response is their pointwise product, and its Fourier list is the convolution of the two coefficient lists (degree $2+2=4$). One more nucleus is one more convolution.*

</figure>

</div>
</div>
