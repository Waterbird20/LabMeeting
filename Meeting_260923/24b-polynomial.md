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

# <span class="cat intro">Intro</span> A product of polynomials is a convolution of coefficients

<div class="columns">
<div class="col">

Use the same numbers as coefficients and multiply two polynomials, an ordinary **pointwise** product of two functions of $x$:

$$
(1+2x+3x^2)(4+5x+6x^2)
$$
$$
=4+13x+28x^2+27x^3+18x^4 .
$$

The coefficient of $x^n$ collects every pair of powers that adds up to $n$, which leads to the same result as the convolution:

$$
c_n=\sum_{i+j=n}a_i\,b_j=(a*b)[n].
$$


</div>
<div class="col">

<figure class="figure">

<video src="media/polynomial.mp4" poster="media/polynomial.png" width="720" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
</div>
