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

Use the same numbers as coefficients and multiply two polynomials, which is an ordinary **pointwise** product of two functions of $x$:

$$
(1+2x+3x^2)(4+5x+6x^2)
$$
$$
=4+13x+28x^2+27x^3+18x^4 .
$$

The coefficient of $x^n$ collects every pair of powers that adds up to $n$, and that is the flip-and-slide sum again:

$$
c_n=\sum_{i+j=n}a_i\,b_j=(a*b)[n].
$$

Multiplying by $x^j$ **shifts** the coefficient list by $j$, so a product of polynomials is a weighted sum of shifted copies, which is what a convolution is.

</div>
<div class="col">

<figure class="figure">

<video src="media/polynomial.mp4" poster="media/polynomial.png" width="640" controls autoplay loop muted playsinline preload="none"></video>

*Each power of $x$ collects one diagonal of the table of pairwise products, and the diagonals are exactly the sums $i+j=n$.*

</figure>

</div>
</div>
