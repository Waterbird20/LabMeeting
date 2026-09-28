---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat method">Method</span> Why it is hard: no clusters, only local structure

- All $10$ digits, raw features: no clusters (silhouette $-0.039$); two principal components hold $35.7\%$ of the variance.
- Test accuracy, $10$ digits: Gaussian-kernel (RBF) SVM $0.637$, MLP $0.632$, CNN $0.855$ ($0.938$, authors' setup).
- The CNN wins by locality: small filters, reused at every position.

<div class="columns">
<div class="col">

<figure class="figure">

![h:350](media/data-tsne.png)

</figure>

</div>
<div class="col">

<figure class="figure">

![h:350](media/data-baselines.png)

</figure>

</div>
</div>
