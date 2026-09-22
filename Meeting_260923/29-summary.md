---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<!-- EDIT-FORWARD: the designed band in the clip (flat to $|\omega-\omega_0|\le0.8$, stop band beyond $2.0$) is my illustrative choice of target, not a lab specification; say the word if you would rather the third kernel place an exact null on an unwanted line, which is closer to the $V_B^-$ conditional gate of chapter 3. -->

# <span class="cat method">Method</span> What to carry out of chapter 2

<style scoped>
.columns { gap: 1.1rem; }
.columns .col:first-child { flex: 0 0 490px; }
</style>

<div class="columns">
<div class="col">

- **Convolution was never a choice.** It appeared on its own in the sum of two dice, in the product of two polynomials, and in the product of many nuclear responses.
- **One factor is the response, the other is the kernel.** In $(f*g)(x)=\int f(y)\,g(x-y)\,dy$ one factor is the response of the system, and the other is the kernel, also called the filter function. Which is which is our own choice.
- **So the kernel is ours to design.** A filter that excludes a part of the response we do not want is apodization. A filter that makes the response we intended is signal processing, and chapter 3 is the exact version of the second.

</div>
<div class="col">

<figure class="figure">

<video src="media/conv-summary.mp4" poster="media/conv-summary.png" width="640" controls autoplay loop muted playsinline preload="none"></video>

*One sharp line at $\omega_0$ read out through eight sequence units. Equal taps give the Dirichlet kernel $|\mathcal{D}_8(\omega)|^2$ with side lobes at $5.2\%$ of the peak. A Hann taper is apodization and drops them to $0.07\%$, at the cost of a line $1.43$ times wider. Taps designed for a band give the response we asked for.*

</figure>

</div>
</div>
