---
marp: true
theme: serif
math: mathjax
---

# <span class="cat results">Result</span> Structure helps generalization

<div class="columns">
<div class="col">

- Asymptotic error $\epsilon_g^{*}$ versus $\delta = D/N$, the intrinsic-dimension ratio (log-log axes).
- A lower intrinsic dimension gives better generalization: a more sharply folded, lower-dimensional manifold is easier to learn.
- Real datasets sit at small $\delta$; the markers indicate MNIST and CIFAR for context.

</div>
<div class="col">

<figure class="figure">

![w:620](media/fig5_eg_vs_delta.png)

<figcaption>Asymptotic test error vs. the dimension ratio D/N: a smaller latent dimension generalizes better. K=M=2, rate 0.2, sign/erf, orthogonalized teacher; D=25/50/100, mean over 3 seeds.</figcaption>
</figure>

</div>
</div>

---

<style scoped>
.columns .col { flex: 1; }
</style>

# <span class="cat results">Result</span> Specialization and model size

Larger students specialize many-to-one, and as complexity grows they separate to lower error.

<div class="columns">
<div class="col">

<figure class="figure">

![h:330](media/fig7_student_size.png)

<figcaption>Asymptotic error vs. student size K/M (teachers M=1,2,4; D=50,100; D/N=0.01, rate 0.2, sign/erf): student units specialize many-to-one, and gains saturate.</figcaption>
</figure>

</div>
<div class="col">

<figure class="figure">

![h:330](media/fig8_increasing_complexity.png)

<figcaption>Learning curves for growing student size K=1..8 (teacher M=10, N=500, D=25, rate 0.2, sign/erf): wider students break away later, fitting functions of increasing complexity.</figcaption>
</figure>

</div>
</div>
