---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat intro">Intro</span> Universal approximation theorem

<style scoped>.columns .col:first-child { flex: 0 0 460px; }</style>

For continuous $f$ on a compact $K\subset\mathbb{R}^d$ and any $\varepsilon>0$, some $N$ and $v_k,\,w_k,\,b_k$ give

$$\sup_{x\in K}\Big|\,f(x)-\sum_{k=1}^{N} v_k\,\sigma(w_k\cdot x+b_k)\Big|<\varepsilon .$$

<div class="columns">
<div class="col">

- Two steep sigmoids make a bump; weighted bumps tile any curve.
- The theorem does not say how to find the weights or how large $N$ must be.

<div class="src">Cybenko, Math. Control Signals Syst. 2, 303 (1989); Hornik, Neural Netw. 4, 251 (1991).</div>

</div>
<div class="col">

<figure class="figure">

<video src="media/uat-bumps.mp4" poster="media/uat-bumps.png" width="660" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
</div>
