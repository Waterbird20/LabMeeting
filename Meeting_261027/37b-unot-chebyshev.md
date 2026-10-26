---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
.columns { align-items: center; margin-top: 0.2em; }
.columns .col:first-child { flex: 0 0 710px; }
figure.figure { margin: 0; }
mjx-container[display="true"] { margin: 0.2em 0 !important; }
ul { margin-top: 0.3em; }
</style>

# <span class="cat results">Results</span> Universal NOT: Chebyshev decomposition

The $K=3$ design is a degree-3 Chebyshev series; it meets the target at the three line centres, not across the bands.

$$
\mathrm{Re}\,P,\ \mathrm{Im}\,P,\ \mathrm{Re}\,Q,\ \mathrm{Im}\,Q=\sum_k\big[c_k\,T_k(x)+s_k\sin\theta\,U_{k-1}(x)\big],\qquad x=\cos\theta,\ \ T_k(x)=\cos k\theta
$$

<div class="columns">
<div class="col">

<figure class="figure">

![w:700](media/ch3-unot-cheb-series.png)

</figure>

</div>
<div class="col">

<figure class="figure">

![w:400](media/ch3-unot-chebyshev.png)

</figure>

- Only odd $k\le3$: degree and parity $3$.
- Exact at the centres; across the bands $1-F=4.3\times10^{-3}$.

</div>
</div>
