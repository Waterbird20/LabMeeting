---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat method">Method</span> Our task: four digits, three feature blocks

<style scoped>.columns .col:first-child { flex: 0 0 370px; } table { font-size: 0.8em; } th, td { padding: 0.35em 0.6em; } th:last-child, td:last-child { white-space: nowrap; } td .sub { display: block; color: var(--muted); font-size: 0.86em; margin-top: 0.15em; }</style>

<div class="columns">
<div class="col">

<figure class="figure">

![w:360](media/data-classes.png)

</figure>

</div>
<div class="col">

- Digits $0, 1, 3, 6$ only: $1325$ training, $264$ validation, $411$ test signals.
- Each QPU encodes a block $x_b$ of $14$ features, contiguous or permuted.
- Additive model $\log P(c\,|\,x)=\sum_b f_b(c,x_b)-\log Z(x)$: the classical product of experts.

| classical model, **4 digits** | test accuracy |
|---|---|
| kernel logistic regression, all $40$ features <span class="sub">shift-aware kernel (sum of RBF kernels over cyclic width-5 sub-windows), chosen on validation data among $2330$ models</span> | $0.978$ |
| tuned RBF SVM, all $40$ features | $0.964$ |
| additive, contiguous blocks | $0.971$ |
| additive, permuted features | $0.891$ |

</div>
</div>
