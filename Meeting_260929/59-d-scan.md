---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>figure.figure { margin: 0 auto 0.3em; } ul { margin-top: 0.2em; } li { margin: 0.1em 0; }</style>

# <span class="cat results">Results</span> Fixing $(a,b,c)$ and sweeping the bias $d$

<figure class="figure">

![w:1040](media/dqml-physics-results-A-of-d-crop.png)

</figure>

- Outside the input-dependent range $-1<d<0$ the bit is constant: accuracy equals no communication (NC).
- Training succeeds only near the balanced bias: $d=-\tfrac12$ (XOR-type), $d=-\tfrac34$ (OR-type).
- An optimal $d$ exists: OR-type at $d=-0.75$ gives $0.917$ (4 digits), equal to trained coefficients ($0.914$).
