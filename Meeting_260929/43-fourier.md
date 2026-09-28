---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>.columns .col:first-child { flex: 0 0 500px; } li { margin-bottom: 0.45em; } figure.figure { margin: 0 auto; } .circ { margin-top: 0.35em; text-align: center; }</style>

# <span class="cat method">Method</span> The Fourier (DFT) encoding

<div class="columns">
<div class="col">

<figure class="figure">

<video src="media/fourier-embed-gates.mp4" poster="media/fourier-embed-gates.png" width="500" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
<div class="col">

$$X_m=\sum_{j=0}^{13}x_j\,e^{-2\pi i jm/14},\quad m=0,\dots,7$$

- **Uniformly controlled** $R_y$: **blue** $a_m=\mathrm{clip}(s_m|X_m|+\beta_m,0,\pi)$; grey: trainable constants
- **Orange** $|b(m)\rangle$: $a_m$'s control values, then $1$, then all $0$; $D$ adds the phase
- Cyclic shift $s$: $X_m\to e^{-2\pi i sm/14}X_m$, one fixed diagonal $D_s$

</div>
</div>

<div class="circ">

![w:1040](media/embed-circuit.png)

</div>
