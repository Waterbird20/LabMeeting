---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
section { font-size: 20px; }
p { margin: 0.25em 0; }
figure.figure { margin: 0.1em auto; }
</style>

# <span class="cat method">Method</span> Why the spectrum takes that form: one kernel copy per line

The sample answers at its own frequencies. Write its spectrum as a set of lines, one per source, at position $\omega_j$ with strength $s_j$ (for a nucleus, its coupling):

$$
S(\omega')=\sum_j s_j\,\delta(\omega'-\omega_j)
\quad\Longrightarrow\quad
S_{\rm meas}(\omega)=\int d\omega'\,S(\omega')\,K(\omega-\omega')=\sum_j s_j\,K(\omega-\omega_j).
$$

Each line sees the kernel only at its own offset $\omega-\omega_j$, so the measured curve is a copy of $K$ placed on every line, scaled by $s_j$, and added. That is the convolution of the chapter opener, and it says what we can and cannot see: no line is sharper than the main lobe, and the side lobes of a strong line spill onto its neighbours.

<figure class="figure">

![w:860](media/taps-spectrum.png)

*Three lines read through the eight-tap kernel. The response is the sum of the three dotted copies; the ripples between the lines are side lobes, not sources.*

</figure>
