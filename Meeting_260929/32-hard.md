---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat method">Method</span> Why it is hard: no clusters, only local structure

<!-- src: all numbers on this slide are TEN-class (all ten digits, 1000 test signals). mnist1d-eda.md §4.2 (PCA 35.7 %), §4.3 (silhouette -0.039), §4.4 (baselines, our protocol: logistic 0.329, RBF SVM with C = 100 by CV 0.637, MLP 0.632, CNN 0.855); mnist1d-repro.md §3 (CNN 93.8 +- 0.4, authors' protocol, 3 seeds) -->

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

<!-- Speaker note: these are ten-class numbers. Our own task (next part) keeps only four digits and is much easier: there a tuned RBF SVM reaches 0.964 and a kernel logistic regression with a shift-aware kernel (the best joint model, chosen on validation data among 2330 classical models) 0.978 (dqml-physics-results.md §0.1; 1325 training signals instead of 4000, with its own tuning). Do not mix the two. Silhouette -0.039 means that on average a signal lies slightly closer to the nearest other class than to its own. The CNN's edge is locality plus weight sharing: its filters see neighbouring features and are reused at every position. t-SNE: the 4000 raw training signals; each digit label sits at its class median (the figure says "digit = class median"); crowded labels are nudged apart, with a thin line to the true median. -->
<!-- Figures: fig_data.py (fig_tsne: perplexity 30, PCA init, seed 42; fig_baselines: numbers of mnist1d-eda.md §4.4 and mnist1d-repro.md §3; hatched bar = authors' training setup). Both carry the title "MNIST-1D, all 10 digits" since 2026-09-29 (fig_tsne also: label repulsion with white outline so the 1 is no longer hidden under the 9, note "digit = class median"), after the ten-class SVM value was read as if it belonged to the four-class task. _verify() recomputes silhouette -0.0389, PC1-2 0.3572, 90 % of the variance at k = 15 and 1-NN 0.557 from the local data. -->
<!-- EDIT-FORWARD: the CNN gap (0.855 vs 0.938) is the training setup, not architecture (learning rate, weight decay, number of steps, best-epoch selection; mnist1d-eda.md §4.4). Say so if asked. -->
