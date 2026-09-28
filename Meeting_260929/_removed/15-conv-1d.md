---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat intro">Intro</span> Convolution extracts local features

<style scoped>.columns .col:first-child { flex: 0 0 490px; }</style>

<div class="columns">
<div class="col">

A short kernel $k$ slides along the signal $x$:

$$(x*k)[i]=\sum_{j=-1}^{1}k_j\,x_{i+j}.$$

This feature map is large where $x$ locally looks like $k$: $k_1=(-1,0,1)$ finds rising edges, $k_2=(-1,2,-1)$ finds peaks.

The same $k$ acts at every $i$, so a shifted input gives a shifted map (equivariance). A ReLU and a global max-pool then give a feature that ignores the position (invariance).

MNIST-1D places each digit at a random shift, which is why a CNN reaches about 94% on it (all ten digits) and an MLP only about 68%.

</div>
<div class="col">

<figure class="figure">

<video src="media/conv1d-features.mp4" poster="media/conv1d-features.png" width="600" controls autoplay loop muted playsinline preload="none"></video>

*MNIST-1D test signal 1 (the digit 3). Shifted by $4$ samples, both maps move by $4$; their maxima $1.75$ and $1.43$ stay.*

</figure>

</div>
</div>

<!-- src: MNIST-1D trace = load_mnist1d() test signal 1 (digit 3), code: 3. wiki/code/lm-260929-animations/scenes_cnn.py (_verify checks the formula against numpy.correlate / numpy.convolve, the shift equivariance, and the maxima 1.75 at i=8 -> 12 and 1.43 at i=10 -> 14). -->
<!-- src: random translation: mnist1d-eda.md §2.1 (circular translation by 0-47 positions before downsampling). CNN ~94% vs MLP ~68%: mnist1d-eda.md §2 (paper reference accuracies, all 10 digits; under our protocol ConvBase 0.855 vs MLPBase 0.632, §4.4). -->
<!-- Speaker note: deep learning calls this a convolution; strictly it is a cross-correlation (a true convolution flips k). The shift in the clip is a translation with the first sample repeated on the left, not the dataset's circular roll, so the boundary does not create a spurious maximum. -->
