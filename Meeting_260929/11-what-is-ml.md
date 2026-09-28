---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat intro">Intro</span> What is machine learning?

<style scoped>.columns .col:last-child { flex: 0 0 490px; }</style>

<!-- src: figure from code/lm-260929-animations/fig_ml.py (_verify: test-loss minimum at t = 1032, then rising while the training loss keeps falling; training loss 0.620 -> 0.0093); toy data, seed 42. The in-figure labels "toy data" and "overfitting" replaced the slide caption on 2026-09-29. -->

<div class="columns">
<div class="col">

**Model.** A function $f_\theta$ with trainable parameters $\theta$.

**Loss** on $M$ training examples $(x_i,y_i)$:

$$L(\theta)=\frac{1}{M}\sum_{i=1}^{M}\big(f_\theta(x_i)-y_i\big)^2$$

**Training.** Gradient descent with step size $\eta$:

$$\theta\;\leftarrow\;\theta-\eta\,\nabla_\theta L(\theta)$$

**Success** is judged on unseen (test) data.

</div>
<div class="col">

<figure class="figure">

![w:460](media/ml-toy-fit.png)

</figure>

</div>
</div>

<!-- Speaker note: theta are, for example, the weights of a network. Physically, L is a potential over parameter space and gradient descent is overdamped motion in it; eta is the learning rate. Figure (toy data): top, f_theta after t gradient steps (light to dark); bottom, the test loss is lowest near t = 10^3 and then rises while the training loss L keeps falling. That rise is overfitting, which is why success is judged on data never seen in training (generalization), not on the training loss. -->
