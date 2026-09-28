---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat strategy">Strategy</span> Qubit reuse: reset the measured qubit instead of tracing it out

<style scoped>.columns .col:first-child { flex: 0 0 440px; } table { font-size: 0.9em; } li { margin-bottom: 0.4em; }</style>

<div class="columns">
<div class="col">

<figure class="figure">

![w:420](media/reuse-schematic.png)

</figure>

</div>
<div class="col">

- Size stays $n$, so the number of rounds $R$ is free
- A threshold on the QPU's own $\hat m_0$ needs many copies: a nonlinearity no single-copy circuit has

| phase on, $R$ | local − no control | two-QPU (CC) − local |
|---|---|---|
| $\lvert m{+}1\rangle$, $2$ | $+2.6\pm2.7$ | $+9.7\pm3.0$ |
| $\lvert m{+}1\rangle$, $4$ | $+9.7\pm4.7$ | $+6.5\pm4.8$ |
| $\lvert b(m)\rangle$, $2$ | $+0.1\pm1.0$ | $+1.7\pm1.0$ |

- Phase on $\lvert b(m)\rangle$: gain within noise

<div class="src">

Test-accuracy differences (percentage points), 4 digits, $n=4$, contiguous blocks, $5$-layer circuit per round; $\pm$ Welch standard error, $3$ or $5$ seeds.

</div>

</div>
</div>
