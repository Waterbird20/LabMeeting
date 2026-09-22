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

# <span class="cat intro">Intro</span> Two dice

<div class="columns">
<div class="col">

Roll two independent dice $X$ and $Y$, and ask for the probability of their sum, $Z=X+Y$. The natural picture is a table of $36$ cells, one per pair $(X,Y)$, each with weight $\tfrac{1}{36}$.

The event $Z=n$ happens whenever $X=k$ **and** $Y=n-k$, for any $k$. So the cells with a fixed sum lie on one anti-diagonal of the table:

$$
P(Z=n)=\sum_k P(X=k)\,P(Y=n-k).
$$

For example, the anti-diagonal of $n=7$ holds six cells, so $P(Z=7)=6/36$, the largest of all.

</div>
<div class="col">

<figure class="figure">

<video src="media/dice-grid.mp4" poster="media/dice-grid.png" width="640" controls autoplay loop muted playsinline preload="none"></video>

*The $6\times6$ table of pairs, and the anti-diagonals $X+Y=n$ marched one by one. Each diagonal is one entry of the distribution of the sum.*

</figure>

</div>
</div>
