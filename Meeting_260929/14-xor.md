---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->
<style scoped>.columns .col:first-child { flex: 0 0 460px; }</style>

# <span class="cat intro">Intro</span> XOR needs a threshold and one more layer

<!-- src: speaker's outline item 3 (BRIEF.md); non-separability, hidden-layer map and truth tables checked by 3. wiki/code/lm-260929-animations/scenes_neuron.py _verify() -->

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

<!-- Speaker note: XOR outputs 1 when exactly one input is 1. The proof. The 0 corners (0,0) and (1,1) need b <= 0 and w1 + w2 + b <= 0; the 1 corners need w1 + b > 0 and w2 + b > 0. Each pair adds to w1 + w2 + 2b, once <= 0 and once > 0: a contradiction. The hidden layer moves the corners to the (h1, h2) plane: the two 1 corners, x = (0,1) and x = (1,0), land on one point, and the AND line separates them. Nonlinearity (the threshold) plus depth (one more layer) is what lets a network compute XOR. -->
<!-- Speaker note (optional, looking ahead): in our model the message bit s is a threshold of the decision function g = a m_i + b m_j + c m_i m_j + d with bias d. Its bilinear term lets one unit compute XOR: at the corners m_i, m_j in {0,1}, (a,b,c,d) = (1,1,-2,-1/2) gives g > 0 exactly when one input is 1. Caveat if you say it: that is CY's original Feb 2026 rule on single-shot outcome bits (~/DQML/README.md sec. 8); the current model feeds the decision function K-shot estimated probabilities m in [0,1], and none of the 48 trained n = 4 decision functions ended as XOR at the corners (dqml-physics-results.md sec. 5.0). Removed from the slide on 2026-09-29 (word cut); the link and results slides carry it. -->
<!-- src (looking-ahead note): decision function g = a m_i + b m_j + c m_i m_j + d from dqml-physics-results.md §0.2 item 4; (1,1,-2,-1/2) is XOR at the corners, App. B and dqml_style.corner_function -->
