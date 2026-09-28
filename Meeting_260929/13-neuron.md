---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->
<style scoped>.columns .col:first-child { flex: 0 0 420px; }</style>

# <span class="cat intro">Intro</span> One neuron: a weighted sum and a threshold

$$y=\sigma(w_1x_1+w_2x_2+b),\qquad \sigma(z)=\begin{cases}1, & z>0\\ 0, & z\le 0\end{cases}$$

<div class="columns">
<div class="col">

- The **decision boundary** $w_1x_1+w_2x_2+b=0$ is a line.
- $w=(1,1)$: $b=-1.5$ gives AND, $b=-0.5$ gives OR.
- Both are **linearly separable**.

<div class="src">McCulloch and Pitts, Bull. Math. Biophys. 5, 115 (1943); Rosenblatt, Psychol. Rev. 65, 386 (1958).</div>

</div>
<div class="col">

<figure class="figure">

<video src="media/neuron-and-or.mp4" poster="media/neuron-and-or.png" width="700" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
</div>
