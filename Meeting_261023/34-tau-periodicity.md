---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
.columns .col:last-child { flex: 0 0 600px; }
</style>

# <span class="cat strategy">Strategy</span> Periodicity

<div class="columns">
<div class="col">

The wait $\tau$ follows from the periodicity of the response. A wait depends on $\Delta$ only through $e^{\pm i\Delta\tau/2}$. Shifting $\Delta$ by $2\pi/\tau$ flips the sign of every wait:

$$
\Delta\to\Delta+\tfrac{2\pi}{\tau}:\quad W\to-W,\quad u\to(-1)^{K}\,u,\quad Q^2\to Q^2 .
$$

- The response repeats every $p=2\pi/\tau$, and so do the windows.
- An image of a window must not land in the stopband.
- Compatible iff $0.9A<p<1.1A$, $1.8A<p<2.2A$ or $p>2.6A$.

</div>
<div class="col">

<figure class="figure">

<video src="media/tau-sweep.mp4" poster="media/tau-sweep.png" width="600" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
</div>

<!--
The waits carry the detuning, and a wait W = diag(e^{−iΔτ/2}, e^{iΔτ/2}) depends on Δ only through these two phases. If we shift Δ by 2π/τ, each phase moves by π, so every wait changes sign, W → −W. The sequence has K waits, so u → (−1)^K u, and Q² does not change at all. The spin-1 operation only sees Q², so the whole response is periodic in the detuning with period p = 2π/τ.

The target has to live with that. Every window reappears at mA + np, its periodic images. An image landing on another window or in a transition band does no harm. But if an image lands in the stopband, the same detuning would have to be inverted and left alone at once. A window point and a stopband point differ by 0.2A to 0.8A, 1.2A to 1.8A or 2.2A to 2.6A, so τ is compatible exactly when no multiple of p falls into these ranges: 0.9A < p < 1.1A, 1.8A < p < 2.2A, or p > 2.6A.

The clip: left, the circle z² = e^{iΔτ}, one turn per period of the response; green outer arcs are the windows, yellow inner arcs the stopband, red where a window lands on the stopband. Right, the detuning axis with the windows and their images. At p = A (τ = 2π/A = 463 ns) all three windows map to one point, z² = 1, and the stopband to the opposite side, around z² = −1. Sweeping to p = 3A, the three windows separate into three points with the stopband in between; red appears exactly in the gaps 1.1A < p < 1.8A and 2.2A < p < 2.6A. At p = 1.5A the image of the central window falls on the stopband at 1.5A, and the images of the outer windows on ∓0.5A: then 1 − r² ≤ E and r² ≤ E at one point, so E ≥ 1/2 for any number of pulses.
-->
