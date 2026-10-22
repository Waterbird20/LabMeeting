---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

<style scoped>
.columns .col:first-child { flex: 0 0 560px; }
figure.figure { margin: 0; }
</style>

# <span class="cat results">Backup</span> Degree scaling

A step across a gap needs degree $K\propto\ln(1/\varepsilon)/\Delta\theta$.

<div class="columns">
<div class="col">

<figure class="figure">

![w:540](media/fig_results_scaling.png)

</figure>

</div>
<div class="col">

$$
K\simeq c\,\frac{\ln(1/\varepsilon)}{\Delta\theta}
$$

- The gap $\Delta\theta$ is the transition band, $0.2A$ in $\Delta$: $0.2\pi$ at $\tau=2\pi/A$, $0.2\pi/6$ at $\tau=2\pi/6A$.
- Both targets need a number of pulses linear in $\ln(1/E)$.
- $E_N$ falls by 7 to 8 per four pulses.

<p class="src">Low and Chuang (2017); Martyn et al. (2021)</p>

</div>
</div>

<!--
Backup: how many pulses a step with a gap needs. For the sign function with a gap of width Delta theta, a polynomial of degree K proportional to ln(1/epsilon)/Delta theta reaches uniform error epsilon (Low and Chuang 2017; Martyn et al. 2021). Here the gap is the transition band, 0.2A in Delta, which is Delta theta = 0.2 pi at tau = 2 pi / A and 0.2 pi / 6 at tau = 2 pi / 6A.

The plot shows the number of pulses N as a function of log10(1/E). Violet: the target of the main part at tau = 2 pi / A, E_N at N = 6, 10, 14, 18 (3.1e-2, 3.8e-3, 5.1e-4, 7.2e-5): a straight line, the error falls geometrically, a factor 7 to 8 per four pulses. Grey: the wider stopband over |Delta| <= 3A at tau = 2 pi / 6A, N = 36, 68, 104 at E = 8.7e-2, 8.8e-3, 9.4e-4: also linear in ln(1/E).

From the slopes, the constant c = K Delta theta / ln(1/epsilon) is about 1.2 at tau = 2 pi / A and about 1.6 at tau = 2 pi / 6A: the same order, although the second target has three windows and six bands per period.
-->
