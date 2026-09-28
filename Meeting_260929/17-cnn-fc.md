---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat intro">Intro</span> A CNN is a constrained fully connected network

<style scoped>.columns .col:first-child { flex: 0 0 460px; }</style>

<div class="columns">
<div class="col">

A convolution is a fully connected (FC) layer $y=Wx$ with two constraints:

- **Locality:** $W_{ij}=0$ outside the kernel, so $W$ is banded.
- **Weight sharing:** $W_{ij}=k_{j-i}$, so $W$ is Toeplitz.
- On $40$ inputs, $1600$ free weights become $5$ (a width-$5$ kernel).

</div>
<div class="col">

<figure class="figure">

<video src="media/cnn-as-fc.mp4" poster="media/cnn-as-fc.png" width="680" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
</div>

<!-- Speaker note: in the clip, an 8 -> 8 FC layer; the long-range weights are set to zero (banded), the rest are tied to k_{-1}, k_0, k_{+1} (Toeplitz); free weights 64 -> 22 -> 3. An FC network can therefore represent any CNN. The CNN wins because its constraint, a locality prior, matches the local structure of the data: under a fixed random permutation of the 40 features, which destroys locality, the CNN falls from 85.5% to 57.8% in our training setup (all ten MNIST-1D digits; the data section shows it in full). -->
<!-- src: counts 64 / 22 / 3 and 1600 / 5 checked in 3. wiki/code/lm-260929-animations/scenes_cnn.py _verify(); the kernel in the clip is a toy kernel k = (-0.6, 1.0, 0.4) and the dense W is random (seed 42). -->
<!-- src: permutation 0.855 -> 0.578 (speaker note only): mnist1d-eda.md §4.6 (ConvBase, one fixed permutation of the 40 features, our protocol, all 10 digits, single seed). Removed from the visible text on 2026-09-29 (word cut), which also settles the earlier EDIT-FORWARD about quoting ten-digit numbers here. -->
