---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

# <span class="cat intro">Intro</span> The response is already convolved

We prepare a state, apply a unitary we designed, and measure. The outcome can be read as how the source, in our case the quantum system, responds to our input, the signal. I prefer to call that output the **response**, spectrum included, and it is convolved: it carries the configuration of the system as well as the sequence we ran.

Sweep a knob $\omega$ and the curve on the screen is never the bare spectrum $S(\omega)$ of the sample. It is

$$
S_{\rm meas}(\omega)\;=\;\int d\omega'\; S(\omega')\,K(\omega-\omega') \;\equiv\; \big[S*K\big](\omega),
$$

which is the definition of a convolution. The definition looks complicated, so before using it let me show why convolution is natural, starting from two dice.
