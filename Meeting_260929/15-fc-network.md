---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat intro">Intro</span> A fully connected network: layers of neurons

<style scoped>.columns .col:first-child { flex: 0 0 470px; }</style>

<!-- src: clip media/fc-network.mp4 from 3. wiki/code/lm-260929-animations/scenes_fc.py (FcNetwork, Manim Community v0.21, rendered 2026-09-29, 25.1 s, poster at 0.95): a toy 4 -> 5 -> 3 network with random weights (seed 42), input x = (0.8, 0.4, 0.3, 0.7), sigmoid hidden units, softmax output; true class y = 1; P(c|x) = (0.38, 0.51, 0.12) -> (0.74, 0.18, 0.07) and L = 0.97 -> 0.29 after one gradient step with eta = 0.5 (values shown during the step are forward passes of the linearly interpolated parameters, P(1|x) rising monotonically, checked in _verify()); 4*5 + 5 + 5*3 + 3 = 43 parameters. _verify() checks every printed number and the backpropagated gradients against finite differences. -->
<!-- src: MLP size: MLPBase of Greydanus and Kobak, 40 -> 100 -> 100 -> 10 with a residual connection (which adds no parameters), 15,210 parameters (mnist1d-repro.md §2; mnist1d-eda.md §4.4); 40*100+100 + 100*100+100 + 100*10+10 = 15210 checked in scenes_fc.py _verify(). This is the MLP whose published accuracy (about 68%) appears on the convolution slide. -->
<!-- src: notation: W_ij with row i = receiving unit and column j = input, as on 17-cnn-fc.md (y = W x); cross-entropy L = -(1/M) sum_i log P(y_i|x_i) with labels y_i as on 11-what-is-ml.md and 49a-loss.md. Backpropagation: Rumelhart, Hinton and Williams, Nature 323, 533 (1986); feedforward networks, softmax output and cross-entropy: Goodfellow, Bengio and Courville, Deep Learning, MIT Press (2016), ch. 6 (ch. 6.5 for back-propagation). The sigmoid sigma(z) = 1/(1+e^{-z}) is shown in the first equation because slide 13 defines sigma as the step, which has zero gradient almost everywhere; backpropagation needs the smooth version (Goodfellow et al. sec. 6.3). -->

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

<!-- Speaker note: stack the neurons of the last two slides in layers and connect every unit to every unit of the previous layer: a fully connected network, or multilayer perceptron (MLP). Unit i of a layer computes sigma(sum_j W_ij x_j + b_i); the output layer turns its scores z_c into class probabilities with the softmax, P(c|x) = e^{z_c} / sum_c' e^{z_c'}, a Boltzmann distribution with energies -z_c at unit temperature. Training minimises the cross-entropy (the same loss our quantum model uses) by gradient descent; backpropagation is the chain rule applied layer by layer from the output back, which gives dL/dW_ij for every edge at a cost comparable to one forward pass (Goodfellow et al. sec. 6.5). It needs a smooth sigma: the step of the previous slides has zero gradient almost everywhere, so trained networks use a sigmoid or ReLU. In the clip (toy weights): forward pass layer by layer, the loss for the true class, the backward pass, and one gradient step that moves P(1|x) from 0.38 to 0.74; the network has 43 trainable parameters. The MNIST-1D reference MLP is 40 -> 100 -> 100 -> 10 (15,210 parameters); its published accuracy, about 68%, is on the next slide. -->
