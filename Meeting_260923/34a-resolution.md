---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<!-- EDIT-FORWARD: page 42 of the first draft is now two slides, 34-response.md (the
     series and its autocorrelation, with the d = 1 Ramsey case) and this one (the two
     consequences plus the figure). The figure is optional: drop it and fold these two
     paragraphs back onto 34 if the section runs long. -->

# <span class="cat method">Method</span> Two numbers follow: the period and the resolution

<style scoped>
.columns { gap: 1.2rem; }
.columns .col:last-child { flex: 0 0 430px; }
</style>

<div class="columns">
<div class="col">

**Periodicity.** The response depends on the detuning only through $z=e^{i\delta\tau}$, and $z$ returns to itself when $\delta\tau$ advances by $2\pi$, so the measured curve repeats with period $2\pi/\tau$ in $\delta$. The wait is a **sampling interval**: choosing $\tau$ chooses where the lines of the spectrum land on the circle, and two lines a multiple of $2\pi/\tau$ apart are aliased onto the same point and can never be told apart.

**Resolution.** The highest harmonic on the list is $e^{\pm id\,\delta\tau/2}$, so the fastest the curve can vary is set by $d\tau$ and the narrowest feature it can carry is
$$
\Delta\delta\sim\frac{2\pi}{d\tau}=\frac{2\pi}{T} .
$$
The total sequence time $T=d\tau$ is a **record length**, and this is the statement Section 2 made with the Dirichlet kernel: a record of finite length convolves every line with the transform of its window. **Resolution is bought with sequence length.**

</div>
<div class="col">

<figure class="figure">

![w:410](media/response-fourier.png)

*The same sequence as the clip, $G_k=R_x(\pi/5)$ with $d$ equal waits. Top: the $d=4$ curve over three periods, the shaded band being one period. Bottom: the notch at $d=1,2,4$, with dots at half its depth. Fig. by `fig_response.py`, 2026-09-22.*

</figure>

</div>
</div>
