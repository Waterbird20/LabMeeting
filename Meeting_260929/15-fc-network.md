---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat intro">Intro</span> A fully connected network: layers of neurons

<style scoped>.columns .col:first-child { flex: 0 0 470px; }</style>

<div class="columns">
<div class="col">

$$\begin{aligned}h&=\sigma\big(W^{(1)}x+b^{(1)}\big),\quad \sigma(z)=1/(1+e^{-z})\\ P(c\,|\,x)&=\mathrm{softmax}_c\big(W^{(2)}h+b^{(2)}\big)\\ L&=-\tfrac1M\textstyle\sum_i\log P(y_i\,|\,x_i)\end{aligned}$$

- **Fully connected:** all $W_{ij}$ are free.
- **Training:** gradient descent on the cross-entropy $L$; backpropagation gives $\nabla_\theta L$.
- **Size:** the MNIST-1D reference MLP has $15\,210$ parameters.

<div class="src">Rumelhart, Hinton and Williams, Nature 323, 533 (1986); Goodfellow, Bengio and Courville, <i>Deep Learning</i> (2016), ch. 6.</div>

</div>
<div class="col">

<figure class="figure">

<video src="media/fc-network.mp4" poster="media/fc-network.png" width="680" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
</div>
