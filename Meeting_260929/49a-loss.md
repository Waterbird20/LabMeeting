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

<!-- Speaker note: on a batch of B signals x_i with true digits y_i, the loss is the cross-entropy. It vanishes when the true digit gets probability 1 and diverges when the model is sure and wrong; uniform guessing over four digits costs ln 4 = 1.386 nats. With the product-of-experts readout each QPU adds its log-probability of the true digit, and besides the message bits only the normaliser Z(x) = sum_c prod_b P_b(c) couples them. The shot schedule starts at small K because the shot noise of the decision function, sigma_g ∝ K^(-1/2), is what gives the message bits a non-vanishing gradient; training at a fixed large K from the start fails. -->
<!-- src: loss: dqml-physics-results.md §0.2 item 5 (cross-entropy); dqml-physics-plan.md §1 (L = -log P(y|x)); ln 4 = 1.386 checked in scenes_readout.py _verify(). Additive form: App. A (log P(c|x) = sum_b log Tr[E_c^(b) rho_b] + const). -->
<!-- src: recipe: dqml-physics-results.md §0.3 (Adam, lr 0.03 with cosine decay, 200 epochs, batch 256; K increases geometrically 10^2 -> 10^4 (or -> 10^3); test accuracy at the epoch with the best validation accuracy, evaluated at the final shot number; exact density-matrix simulation in complex128/float64, checked against a full-register simulator up to 9 qubits, agreement 1e-10; about 900 training runs over Phases 2-3B on Colab, A100 GPU and CPU). Splits: §0.1 (1325 train / 264 validation / 411 test). -->
<!-- src: shot noise: dqml-physics-results.md §6 (d pi / d g = phi(g/sigma)/sigma, only signals with |g| <~ sigma contribute; sigma_g ∝ K^{-1/2}; training at fixed K = 10^3 and 10^4 from the start loses 6.8 and 8.4 points against K = 100, Phase 2c, original encoding). -->
<!-- EDIT-FORWARD: some runs anneal K only to 10^3 (§0.3 "or -> 10^3"); the table gives the default. The fixed-K numbers (6.8 and 8.4 points) are from the original encoding; the slide now leaves the shot-schedule rationale to the speaker note (the numbers are on the "(a,b,c,d) are now trainable" slide); quote them aloud if useful. -->
<!-- EDIT-FORWARD: every simulation is exact (density matrices), but the link bits use K-shot estimated probabilities (the shot noise is simulated analytically through the probit, App. B), so "exact" means no sampling error beyond the modelled one. -->
