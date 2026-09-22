---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
.columns { gap: 1.0rem; }
.columns .col:first-child { flex: 0 0 430px; }
</style>

<!-- EDIT-FORWARD: the loaded-dice numbers are read off the dice-weighted animation; they are illustrative, not data. -->

# <span class="cat intro">Intro</span> Uneven dice: the same picture with weights

<div class="columns">
<div class="col">

Now load the dice, so the faces are no longer equally likely. Nothing changes in the bookkeeping: the same anti-diagonal is summed, only each cell now carries the product of two unequal weights,

$$
P(Z=n)=\sum_k P(X=k)\,P(Y=n-k)=\big[P_X*P_Y\big](n).
$$

For example, the clip reads off

$$
P(Z=4)=0.16\cdot 0.24+0.21\cdot 0.22+0.17\cdot 0.11 .
$$

Nobody chose to convolve anything. Adding two independent random variables **is** a convolution of their distributions.

</div>
<div class="col">

<figure class="figure">

<video src="media/dice-weighted.mp4" poster="media/dice-weighted.png" width="640" controls autoplay loop muted playsinline preload="none"></video>

*Loaded dice: the same table and the same anti-diagonals, now weighted cell by cell.*

</figure>

</div>
</div>
