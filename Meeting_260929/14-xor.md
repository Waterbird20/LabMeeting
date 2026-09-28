---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->
<style scoped>.columns .col:first-child { flex: 0 0 460px; }</style>

# <span class="cat intro">Intro</span> XOR needs a threshold and one more layer

<div class="columns">
<div class="col">

**No single line** separates XOR: its $0$ corners need $w_1+w_2+2b\le0$, its $1$ corners $w_1+w_2+2b>0$.

**One hidden layer** fixes it:

$$\begin{gathered}h_1=\mathrm{OR}(x),\quad h_2=\mathrm{NAND}(x)\\ y=\mathrm{AND}(h_1,h_2)\end{gathered}$$

Without the threshold, two linear layers collapse into one.

<div class="src">Minsky and Papert, <i>Perceptrons</i>, MIT Press (1969); Rumelhart, Hinton and Williams, <i>Parallel Distributed Processing</i>, Vol. 1, MIT Press (1986).</div>

</div>
<div class="col">

<figure class="figure">

<video src="media/xor-hidden.mp4" poster="media/xor-hidden.png" width="680" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
</div>
