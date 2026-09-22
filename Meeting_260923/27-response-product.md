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

# <span class="cat method">Method</span> A product of responses is a convolution of coefficients

**Our response is a product, $M=\prod_j M_j$, and in Fourier space a product is a convolution. So one more nucleus is one more convolution of coefficient lists.**

<div class="columns">
<div class="col">

Each $M_j(\theta)$ is a function of the swept variable $\theta$. For $K$ equal units it is a trigonometric polynomial $M_j(\theta)=\sum_n c^{(j)}_n e^{in\theta}$ of degree $K-1$. Multiplying the responses is the polynomial slide with $x=e^{i\theta}$:

$$
M_1M_2=\sum_n c_n e^{in\theta},\qquad c_n=\sum_k c^{(1)}_k\,c^{(2)}_{n-k}=\big(c^{(1)}*c^{(2)}\big)_n .
$$

The degrees add, so the lists get longer. 

</div>
<div class="col">

<figure class="figure">

<video src="media/response-product.mp4" poster="media/response-product.png" width="720" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
</div>
