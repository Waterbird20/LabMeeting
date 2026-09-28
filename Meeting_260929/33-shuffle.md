---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat method">Method</span> The feature-permutation test: the value of the locality prior

- One fixed random permutation of the $40$ features ("shuffled MNIST-1D"); every model retrained.
- Logistic regression, $k$-nearest neighbours, RBF SVM unchanged: distances and inner products are preserved.
- On all $10$ digits the CNN falls from $0.855$ to $0.578$: its **locality prior** is worth $0.277$.

<div class="columns">
<div class="col">

<figure class="figure">

![h:370](media/data-shuffle-example.png)

</figure>

</div>
<div class="col">

<figure class="figure">

![h:370](media/data-shuffle.png)

</figure>

</div>
</div>
