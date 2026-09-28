---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat method">Method</span> Communication topologies compared: three messages per round

<figure class="figure">

![w:1100](media/link-patterns.png)

</figure>

- NC (no communication): local operations only (LO); the other four are LOCC, two rounds
- $1\to1$ reads one QPU: $g=a\,m_i+d$
- $2\to1$, $2\to2$, $2\to3$ compute the same three $g(m_i,m_{i+1})$; only the receivers differ
