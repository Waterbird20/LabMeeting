---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
p { margin: 0.25em 0; }
figure.figure { margin: 0.15em auto; }
</style>

# <span class="cat intro">Intro</span> The response is already convolved

The **sources** are the spins in the sample. They respond to our input, the **signal**, through the **processing** we chose, and we call the output the **response**, spectrum included. It is convolved: it carries the configuration of the sample as well as the sequence we ran.

$$
S_{\rm meas}(\omega)\;=\;\int d\omega'\; S(\omega')\,K(\omega-\omega') \;\equiv\; \big[S*K\big](\omega).
$$

The definition looks complicated, so we start from two dice, where the convolution appears on its own.

<figure class="figure">

![w:760](media/signal-flow.png)

*Three sources emit their own dips $M_j(\omega)$, together they are the signal $S(\omega)$, the sequence acts as a kernel $K(\omega)$, and the screen shows $S*K$.*

</figure>
