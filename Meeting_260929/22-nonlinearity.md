---
marp: true
theme: serif
math: mathjax
---

# <span class="cat method">Method</span> Nonlinearity from measurement, depth from pooling

<div class="columns">
<div class="col">

**Non-selective measurement: linear**

$$P(c)=\operatorname{Tr}\big[\Pi_c\,\mathcal E(\rho(x))\big]=\operatorname{Tr}\big[E_c\,\rho(x)\big]$$

- Gates, measurements and feed-forward, averaged over outcomes: one channel $\mathcal E$
- Nonlinear in $x$ only through the embedding $x\mapsto\rho(x)$

</div>
<div class="col">

**Selective measurement: nonlinear**

$$\rho_\mu=\frac{M_\mu\,\rho\,M_\mu^\dagger}{\operatorname{Tr}\big[M_\mu\,\rho\,M_\mu^\dagger\big]}$$

- Nonlinear in $\rho$ through the normalisation
- A threshold $\hat m>\tau$ of a $K$-shot estimate: nonlinear, needs many copies

</div>
</div>

**Depth:** each pooling layer builds new features from the previous outcomes.

<div class="src">Cong, Choi, Lukin, Nat. Phys. 15, 1273 (2019).</div>
