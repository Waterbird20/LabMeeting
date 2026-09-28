---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

<style scoped>table { font-size: 17px; width: 100%; margin: 0.1em 0 0.15em; } table td, table th { padding: 2px 12px; white-space: nowrap; } table td:last-child { white-space: normal; } table td:first-child { font-weight: 600; } p { margin: 0.45em 0; } .columns { align-items: center; margin-top: 0; } .columns .col:first-child { flex: 0 0 500px; } figure.figure { margin: 0; }</style>

# <span class="cat strategy">Strategy</span> Suggestion: a distributed task, not a classification benchmark

| | now: MNIST-1D (4 digits) | proposal: inherently distributed task |
|---|---|---|
| each QPU holds | a contiguous block of one signal | senders: $x$ or $y$; receiver: none |
| label | mostly local (additive model: $0.971$) | a joint function $f(x,y)$: equality, Hamming distance, disjointness, ... |
| communication | adds $1.1$ to $3.1$ points | necessary |
| figure of merit | test accuracy | message size at a given worst-case error |
| quantum advantage | none observed: ours $\approx0.90$,<br>kernel logistic regression (shift-aware) $0.978$ | proven for some $f$, not for all |

<div class="columns">
<div class="col">

<figure class="figure">

<video src="media/fingerprint.mp4" poster="media/fingerprint.png" width="500" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
<div class="col">

**Goal:** an answer that depends jointly on inputs no single QPU holds.

**Example, equality:** each sender sends a quantum fingerprint, the referee runs a SWAP test.

<div class="src">Buhrman, Cleve, Watrous, de Wolf, Phys. Rev. Lett. 87, 167902 (2001).</div>

</div>
</div>
