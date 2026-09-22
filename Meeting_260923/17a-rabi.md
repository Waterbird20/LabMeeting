---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<!-- TODO (speaker to confirm): confirm the read-out convention. These slides plot $P(0)$, the population of $\ket{0}$, which rises with fluorescence; the thesis plots normalised fluorescence contrast, so its minima are the $\pi$ pulses just as they are here. -->

<style scoped>
.columns { gap: 1.2rem; align-items: center; margin-top: 0.15em; font-size: 0.95em; }
.columns .col:first-child { flex: 0 0 452px; }
mjx-container[display="true"] { margin: 0.3em 0 !important; }
figure.figure { margin: 0.25em auto 0; }
figure.figure p { margin-top: 0.2em; }
</style>

# <span class="cat intro">Intro</span> Rabi: calibrating the rotation angle

<div class="columns">
<div class="col">

Set the pulse on resonance, $\delta=0$, so $H=\frac{\Omega}{2}Y$ and the pulse is a rotation about $y$ by the angle $\Omega t$:

$$
\begin{aligned}
P(0)&=\big|\bra{0}e^{-i\Omega Y t/2}\ket{0}\big|^{2}\\
&=\cos^{2}\!\Big(\frac{\Omega t}{2}\Big).
\end{aligned}
$$

Sweep the pulse length and the fluorescence oscillates at $\Omega$.

<figure class="figure">

![w:430](media/pulse-rabi.png)

</figure>

</div>
<div class="col">

<figure class="figure">

<video src="media/seq-rabi.mp4" poster="media/seq-rabi.png" width="700" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
</div>
