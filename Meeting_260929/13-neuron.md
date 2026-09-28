---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->
<style scoped>.columns .col:first-child { flex: 0 0 420px; }</style>

# <span class="cat intro">Intro</span> One neuron: a weighted sum and a threshold

<!-- src: speaker's outline item 3 (BRIEF.md); weights and truth tables checked by 3. wiki/code/lm-260929-animations/scenes_neuron.py _verify() -->

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

<!-- Speaker note: a neuron forms a weighted sum of its inputs and passes it through a threshold. The weight w_i says how much input x_i counts; the bias b shifts the decision boundary. The sigmoid is the smooth version of the step. In the clip: the neuron, then the corners of the unit square; gold corners output 1, hollow corners output 0. Raising b from -1.5 to -0.5 shifts the line and turns AND into OR. Linearly separable: one line splits the 1s from the 0s. -->
