---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>.columns .col:last-child { flex: 0 0 410px; } .columns { align-items: center; } table { font-size: 0.95em; margin: 0.3em 0 0.6em; } table th, table td { padding: 3px 14px; } li { margin-bottom: 0.4em; }</style>

# <span class="cat results">Results</span> What the trained decision functions compute

<div class="columns">
<div class="col">

$6$ decision functions per model $\times$ seeds 0–7 $=48$, read at the corners $\{0,1\}^2$:

| Boolean function | count |
|---|---|
| constant | 14 |
| dictator or its negation ($m_i$, $\lnot m_j$, …) | 20 |
| implication and its variants ($\lnot m_i\lor m_j$, …) | 9 |
| AND, NAND, NOR | 5 |
| XOR | 0 |

- Many ignore an input: constant, or one QPU's estimate only. Task or trainability? Open.
- The $34$ non-constant ones are mostly unbalanced: only $26\,\%$ have $P(s=1)$ in $0.3$–$0.7$; the bit flags a subgroup of signals.

</div>
<div class="col">

<figure class="figure">

![w:400](media/dqml-physics-results-response-n4.png)

</figure>

</div>
</div>
