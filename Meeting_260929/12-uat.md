---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat intro">Intro</span> Universal approximation theorem

<style scoped>.columns .col:first-child { flex: 0 0 460px; }</style>

<!-- src: theorem statement from Cybenko 1989 (sigmoidal activation) and Hornik 1991 Theorem 2 (continuous, bounded, nonconstant activation), see REFS-ml.md; clip numbers from code/lm-260929-animations/scenes_ml.py (_verify: max error 0.739, 0.445, 0.232, 0.0547, 0.0237 for N = 2, 4, 8, 16, 32) -->
<!-- EDIT-FORWARD: Hornik 1991 needs a bounded activation; unbounded ones such as ReLU are covered by Leshno et al. 1993 (any non-polynomial activation). Say this aloud if someone asks about ReLU. -->

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

<!-- Speaker note: sigma is a sigmoid (a smoothed step); the bump on [a, b] is sigma(w(x-a)) - sigma(w(x-b)) with a large w, and v_k sets the height of each bump, like the bars of a histogram. In the clip only the output weights v_k are fitted (least squares) and the maximum error falls from 0.739 at N = 2 to 0.0237 at N = 32. Cybenko proved it for sigmoids, Hornik for any continuous, bounded, nonconstant activation. The theorem is silent on how to find the weights and how many neurons are needed; this is one reason structure and depth matter. -->
