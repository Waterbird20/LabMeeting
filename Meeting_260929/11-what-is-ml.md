---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat intro">Intro</span> What is machine learning?

<style scoped>.columns .col:last-child { flex: 0 0 490px; }</style>

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
