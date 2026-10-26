---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
.columns { margin-top: 0; align-items: center; }
.columns .col:first-child { flex: 0 0 510px; }
figure.figure { margin: 0; }
mjx-container[display="true"] { margin: 0.2em 0 !important; }
table { font-size: 0.78em; margin: 0.3em 0 0; }
th, td { padding: 0.22em 0.5em; }
</style>

# <span class="cat method">Method</span> Universal NOT: finite pulses

A rectangular pulse acts as a hard pulse between two short free evolutions, so the hard design no longer fits exactly.

$$
e^{-i\frac{\beta}{2\Omega}(\Delta Z+\Omega X)}\ \simeq\ R_z(\Delta t_e)\,R_x(\beta)\,R_z(\Delta t_e),\qquad t_e=\frac{\tan(\beta/2)}{\Omega}
$$

<div class="columns">
<div class="col">

<figure class="figure">

<video src="media/real-pulse-bloch.mp4" poster="media/real-pulse-bloch.png" width="500" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
<div class="col">

Two ways to play the $K=3$ design with pulses of length $\lvert\beta_k\rvert/\Omega$:

1. **Unchanged:** the same angles and phases, a wait $\tau$ between pulse edges.
2. **Re-solved:** the nine equations solved again by Newton with the finite-pulse propagator, starting from the hard design.

</div>
</div>
