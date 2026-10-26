---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

<style scoped>
.columns .col:first-child { flex: 0 0 560px; }
</style>

# <span class="cat method">Method</span> Free evolution

Start with all pulses off, $\phi_k=0$: $K$ waits make one rotation.

<div class="columns">
<div class="col">

$$
u=R_z(K\Delta\tau)=\begin{pmatrix}z^{-K}&0\\0&z^{K}\end{pmatrix},\qquad z=e^{i\theta},\quad \theta=\tfrac{\Delta\tau}{2}
$$
$$
\mathrm{Re}\,P=\cos K\theta,\qquad \mathrm{Im}\,P=-\sin K\theta=-\sin\theta\,\frac{\sin K\theta}{\sin\theta},\qquad Q=0
$$

| $K$ | $\mathrm{Re}\,P$ | $\mathrm{Im}\,P$ |
|---|---|---|
| 1 | $\cos\theta$ | $-\sin\theta\cdot 1$ |
| 2 | $2\cos^2\theta-1$ | $-\sin\theta\cdot 2\cos\theta$ |
| 3 | $4\cos^3\theta-3\cos\theta$ | $-\sin\theta\,(4\cos^2\theta-1)$ |

- $\mathrm{Re}\,P$ and $\mathrm{Im}\,P/\sin\theta$ are polynomials in $\cos\theta$, of degree $K$ and $K-1$.

</div>
<div class="col">

<figure class="figure">

<video src="media/qsp-pulses-off-k5.mp4" poster="media/qsp-pulses-off-k5.png" width="540" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
</div>
