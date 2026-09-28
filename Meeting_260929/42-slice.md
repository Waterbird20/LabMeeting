---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

<style scoped>.columns .col:first-child { flex: 0 0 440px; } li { margin-bottom: 0.5em; }</style>

# <span class="cat method">Method</span> Feature partition: each QPU encodes $14$ of the $40$ features

<div class="columns">
<div class="col">

- **Contiguous blocks** (cyclic):<br>$0$–$13$, $14$–$27$, $27$–$39$ + $0$
- **Permuted features:** same partition after a fixed random permutation
- One block alone: linear classifier on its encoded state $\le0.67$ (4 digits, chance $0.25$)

</div>
<div class="col">

<figure class="figure">

<video src="media/data-slice.mp4" poster="media/data-slice.png" width="700" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
</div>

<div class="src">MNIST-1D: Greydanus and Kobak, ICML 2024, arXiv:2011.14439.</div>

<!-- Speaker note: neighbouring contiguous blocks share features 0 and 27. No single block identifies the digit; the message bits give a QPU a thresholded summary of another block. The clip shows MNIST-1D test signal 1 (a digit 3); its permutation (seed 42) is illustrative, not the one used in the runs (the clip title says "Illustrative permutation"). (Former caption: "MNIST-1D test signal 1 (a digit 3). The permutation in the clip is illustrative, not the one used in the runs.") -->
<!-- src: 14 of 40 features per QPU, contiguous vs permuted windows: dqml-physics-results.md §0.1 and §0.2. Windows 0-13, 14-27, 27-39 + 0: dqml-quantum-stage.md §2 "Slicings": "contiguous windows of width w (cyclic, the two 13-blocks widened on the right, so w = 14 shares features 0 and 27 between neighbours)". -->
<!-- note: the window definition comes from the 2026-09-05 slicing study (dqml-quantum-stage.md §2) and matches "QPU 1 (features 14-27)" of dqml-physics-results.md §2 (worked example). -->
<!-- src: per-window linear-classifier accuracy 0.64-0.67 (window 1, best), 0.53-0.54 (window 0), 0.34-0.36 (window 2): dqml-physics-results.md §4.2 last bullet. "Linear classifier" = logistic regression on vec(rho) of the encoded state (§1, §4.2), not on raw features (the best classical single-window classifier on raw features is 0.70, log.md Phase 2b; not quoted). Chance 0.25 = 1/4 classes. -->
<!-- src: clip data-slice from 3. wiki/code/lm-260929-animations/scenes_embed.py (DataSlice): real test signal 1 from load_mnist1d() (digit 3); windows = dqml_style.window_indices() (cyclic; window 2's wrap onto feature 0 drawn as a sliver band with "wraps" arrows); permutation np.random.default_rng(42).permutation(40), illustrative only. _verify() checks the digit, the three windows, the shared features 0 and 27, the band labels and the permutation. -->
<!-- src: "the bits deliver a thresholded summary of window 1" to QPUs 0 and 2: dqml-physics-results.md §4.2 last bullet. -->
<!-- EDIT-FORWARD: note (§4.2) that with contiguous windows the communication gain is mainly calibration, not cross-window synergy (D_add = 0.021 nats); the results section may say this. -->
<!-- 2026-09-29 integration: visible task label "4 classes" -> "4 digits" (speaker's standing style: label accuracies with the task, "4 digits"). -->
