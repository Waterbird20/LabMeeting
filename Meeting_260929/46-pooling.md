---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat method">Method</span> Pooling is a mid-circuit measurement

<style scoped>.columns .col:first-child { flex: 0 0 470px; } li { margin-bottom: 0.6em; }</style>

<div class="columns">
<div class="col">

- Each round: measure one qubit mid-circuit, trace it out, apply $T_\mu$ to a remaining qubit; $4\to3\to2$ qubits, as in CNN pooling
- $K$ shots give $\hat m$; the receiver gets only the message bit $s$ ($U_+$ or $U_-$)
- Averaged over $\mu$: still a quantum channel, linear in $\rho$; the nonlinearity is the threshold on $\hat m$

</div>
<div class="col">

<figure class="figure">

<video src="media/pool-round.mp4" poster="media/pool-round.png" width="640" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
</div>
