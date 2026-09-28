---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat intro">Intro</span> Convolution extracts local features

<style scoped>.defn { align-items: center; } .defn .col:first-child { flex: 0 0 330px; } .defn p { margin: 0; } .clip-label p { text-align: center; font-weight: 600; margin: 0.1em 0 0.25em; }</style>

<div class="columns defn">
<div class="col">

$$(x*k)[i,j]=\sum_{m,n}k_{m,n}\,x_{i+m,j+n}$$

</div>
<div class="col">

Each output pixel weights the patch under the kernel $k$; the same $k$ at every $(i,j)$ gives translation equivariance.

</div>
</div>

<div class="columns">
<div class="col">

<div class="clip-label">

Box blur, $k_{m,n}=1/9$

</div>

<figure class="figure">

<video src="media/image-conv-kirby.mp4" poster="media/image-conv-kirby.png" width="540" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
<div class="col">

<div class="clip-label">

Sobel kernel: vertical edges

</div>

<figure class="figure">

<video src="media/sobel-kirby.mp4" poster="media/sobel-kirby.png" width="540" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
</div>

A CNN learns its kernels. MNIST-1D, ten digits (published): CNN $\approx0.94$, MLP $\approx0.68$.

<div class="src">Clips: 3Blue1Brown, <i>But what is a convolution?</i> (2022), re-rendered with Kirby; see also <i>But what is a neural network?</i> (2017).</div>
