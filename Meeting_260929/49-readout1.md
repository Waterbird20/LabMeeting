---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat results">Results</span> The trained model on three test signals

<style scoped>h1 { margin-bottom: 0.2em; } ul { margin: 0.1em 0 0.2em; font-size: 0.95em; } li { margin: 0.1em 0; } figure.figure { margin: 0.1em auto 0; }</style>

- Trained model with classical communication (CC), seed $0$; top: each QPU's $P_b(c)$, bottom: the product
- Signal 1: every QPU alone is wrong; the product gives digit 3 probability $0.91$
- Signal 6: QPU 0 vetoes 1 and 3; signal 5: one confident wrong QPU outvotes two

<figure class="figure">

![w:1120](media/readout-trained-examples-crop.png)

</figure>
