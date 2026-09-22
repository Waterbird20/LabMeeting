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

# <span class="cat method">Method</span> One kernel copy per line

The sample answers at its own resonant frequencies. So write its spectrum as a set of lines, one per source, at the position $\omega_j$ with the strength $s_j$:

$$
S(\omega')=\sum_j s_j\,\delta(\omega'-\omega_j)
\quad\Longrightarrow\quad
S_{\rm meas}(\omega)=\int d\omega'\,S(\omega')\,K(\omega-\omega')=\sum_j s_j\,K(\omega-\omega_j).
$$

Each line sees the kernel only at its own offset $\omega-\omega_j$. So the measured curve is a copy of $K$ placed on every line, scaled by $s_j$, and added. 

<figure class="figure">

![w:900](media/taps-spectrum.png)

</figure>
