---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat method">Method</span> MNIST-1D: a digit written as $40$ numbers

<!-- src: papers/@greydanus2020scaling.md §2; mnist1d-eda.md §2 (ten classes, 4000/1000 split), §2.1 (pipeline order verified against the installed mnist1d package) -->

- Each signal: one of ten $12$-point templates, randomly transformed to $40$ features.
- The shift puts the digit anywhere: the label lives in the shape of a local stroke.
- Standard set: all $10$ digits, $4000$ training and $1000$ test signals.

<figure class="figure">

![w:1100](media/data-pipeline.png)

</figure>

<div class="src">Greydanus and Kobak, ICML 2024 (arXiv:2011.14439).</div>

<!-- Speaker note: the transformation is pad, scale, circular shift, noise, shear, downsample to 40 features (the four panels). The figure is test signal 1 of our four-digit task (a digit 3), regenerated with the package's own random draws; orange marks the points that come from the template. The set trains in minutes on a CPU, yet it separates model families far more than MNIST does, where logistic regression already reaches about 94 % (mnist1d-eda.md §2). -->
<!-- Figure: 3. wiki/code/lm-260929-animations/fig_data.py (fig_pipeline): replay of mnist1d.make_dataset (seed 42), checked bit for bit against the stored dataset in _verify(). Shear = subtraction of a random linear ramp. Caption removed 2026-09-29 (speaker: no captions); the orange panel title "1. template" now carries the colour key. -->
