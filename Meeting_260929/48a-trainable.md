---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat strategy">Strategy</span> $(a,b,c,d)$ are now trainable

<style scoped>.columns { align-items: center; } .columns .col:first-child { flex: 0 0 480px; } table { font-size: 0.95em; margin-bottom: 0.3em; } td, th { padding: 0.25em 0.7em; } p { margin: 0.35em 0; }</style>

| | CY's code (Feb 2026) | now |
|---|---|---|
| input $m$ | single-shot bits $\{0,1\}$ | $K$-shot estimates $[0,1]$ |
| message bit $s$ | deterministic threshold | stochastic: $P(s{=}1)=\pi(g)$ |
| $(a,b,c,d)$ | fixed, or SPSA | backpropagation (all six $g$) |

<div class="columns">
<div class="col">

- Exact gradient $\partial\pi/\partial g=\varphi(g/\sigma)/\sigma$: appreciable only for $|g|\lesssim\sigma\propto K^{-1/2}$
- Shot schedule $K=10^2\to10^4$: $+4.5$ percentage points over fixed $K=10^4$

</div>
<div class="col">

<figure class="figure">

![w:660](media/link-gradient.png)

</figure>

</div>
</div>

<div class="src">SPSA (simultaneous perturbation stochastic approximation): J. C. Spall, IEEE Trans. Autom. Control 37, 332 (1992).</div>
