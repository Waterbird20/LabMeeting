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
