---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

# <span class="cat method">Method</span> Training: cross-entropy loss and the training setup

<style scoped>.columns .col:first-child { flex: 0 0 470px; } .columns { margin-top: 1.2em; } table { font-size: 0.9em; } td, th { padding: 0.35em 0.7em; }</style>

<div class="columns">
<div class="col">

$$L=-\frac1B\sum_{i=1}^{B}\log P(y_i\,|\,x_i)$$

- Uniform guess (four digits): $\ln4\approx1.386$ nats
- Product-of-experts readout:

$$\log P(y\,|\,x)=\sum_b\log P_b(y)-\log Z(x)$$

</div>
<div class="col">

| step | choice |
|---|---|
| optimizer | Adam, learning rate $0.03$, cosine decay |
| length | $200$ epochs, batch $B=256$ |
| shots | $K$: $10^2\to10^4$, geometric |
| model selection | best epoch on $264$ validation signals; $411$ test signals |
| simulation | exact density matrices |
| compute | about $900$ runs, Colab (A100 GPU, CPU) |

</div>
</div>

<div class="src">Adam: D. P. Kingma and J. Ba, ICLR 2015, arXiv:1412.6980.</div>
