---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat method">Method</span> Online data augmentation: a freshly augmented sample every epoch

<style scoped>.columns .col:first-child { flex: 0 0 480px; }</style>

<div class="columns">
<div class="col">

- Each epoch: circular shift $|s|\le2$ + correlated noise $0.1$
- Generalization gap ($n=4$): $0.008$ without communication, $0.035$ with
- Random message bits (same mean) remove about half of the $0.035$

</div>
<div class="col">

<video src="media/augment.mp4" poster="media/augment.png" width="660" controls autoplay loop muted playsinline preload="none"></video>

</div>
</div>

<div class="src">Gaps: seeds 3–7, not used for model selection. Augmentation as in the MNIST-1D generator: Greydanus and Kobak, ICML 2024, arXiv:2011.14439.</div>

<!-- Speaker note: a fixed training set invites memorising "a 3 at position 17"; a new shift every epoch leaves only the shape as a stable cue. Without communication the model's accuracy on freshly augmented training signals equals the test accuracy. With classical communication (CC), random message bits of the same mean roughly halve the gap, so about half of it sits in the message bits. Shift and noise are applied to the whole signal before it is split into feature blocks; the noise is correlated noise of scale 0.1, smoothed with a Gaussian of width 2 (the clip shows both). -->
<!-- Speaker note (former clip caption): a real MNIST-1D training signal (a digit 3): four epochs, four shifts and four noise draws, the same label; the noise is drawn 10x enlarged. The feature blocks are cyclic, so the third also takes feature 0. -->
<!-- src: recipe: dqml-physics-results.md §0.3 (each epoch a random shift of up to ±2 samples plus correlated noise of scale 0.1); dqml-quantum-stage.md §14 (fresh circular shift |s| <= 2 and correlated noise 0.1 smoothed with sigma = 2, on every training signal every epoch, before windowing; the generator's own invariances). -->
<!-- src: gap: dqml-physics-results.md §3 Stage 1 table (3 QPUs x 4 qubits, seeds 3-7: two-input links gap 0.035, no communication gap 0.008) and "Origin of the gap" (without communication: accuracy on newly augmented copies of the training signals equals test accuracy, differences -0.0006 and 0.0000; replacing every bit by an input-independent random bit with the same mean removes about 50 % of the gap at 4 qubits; "the thresholds are fitted to the finite set of training signals" is marked [the decomposition is measured, the mechanism is not tested]). 1325 training signals: §0.1. -->
<!-- src: clip augment from 3. wiki/code/lm-260929-animations/scenes_readout.py (Augment): load_mnist1d() four-class training array, index 5 (a digit 3); shifts and noise from np.random.default_rng(42) (s = -2, +1, -1, -2); noise = gaussian_filter(0.1 * randn(40), 2), the convention of mnist1d.transform.corr_noise_like; cyclic windows 0-13 / 14-27 / 27-39 plus feature 0 (dqml_style.window_indices(); dqml-quantum-stage.md §2 "Slicings": neighbours share features 0 and 27); window 2 is drawn as a band on 27-39 plus a small band on feature 0. -->
<!-- EDIT-FORWARD: the brief said "most of the gap is in the bits"; at n = 4 the results page says random bits remove about 50 % of it (80 % holds only for the 6-qubit model, which the deck does not quote). The slide says "about half". -->
<!-- EDIT-FORWARD: "position 17" is an illustrative example of memorisation, not a measured feature of a trained model. -->
<!-- EDIT-FORWARD: the exact noise implementation of the DQML code (smoothing mode, normalisation) is not in the vault; the clip uses the mnist1d package convention (scale times white noise, then a Gaussian filter of width 2), which gives a noise standard deviation of about 0.02 to 0.05, small next to the signal (standard deviation about 1.1). Confirm against ~/DQML src (aug_noise) if asked. -->
<!-- EDIT-FORWARD: the clip's training signal is index 5 of the four-class filter of the standard 4000-signal training split; whether it is in the run's 1325 training or 264 validation signals is not recorded. -->
<!-- EDIT-FORWARD: a stronger augmentation (noise 0.2, shifts ±3) changes where the gap appears but not its size (§3 Stage 2); mention if asked. -->
