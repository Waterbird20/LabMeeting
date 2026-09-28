---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat results">Results</span> A linear classifier on the encoded states already reaches $0.943$

<style scoped>figure.figure { margin-top: 0.1em; } mjx-container[display="true"] { margin: 0.1em 0 !important; } p { margin: 0.15em 0; } .columns { gap: 14px; margin-top: 0.2em; align-items: flex-end; } .columns .col:first-child { flex: 0 0 660px; } .columns .col:last-child { flex: 0 0 440px; } .columns p { margin: 0; }</style>

Logistic regression on the entries of the three encoded states $\rho_b(x)$:
$$\mathrm{score}_c(x)=\sum_{b=0}^{2}\mathrm{Tr}\big[W_c^{(b)}\rho_b(x)\big]+\beta_c,\qquad \hat c=\arg\max_c\,\mathrm{score}_c .$$
A QPU without incoming messages, $P_b(c)=\mathrm{Tr}[E_c\rho_b]$ (POVM elements $E_c$), is such a classifier: this accuracy bounds it.

<figure class="figure">

<div class="columns">
<div class="col">

![w:660](media/linear-classifier-planes.png)

</div>
<div class="col">

![w:440](media/linear-classifier-bars.png)

</div>
</div>

</figure>
