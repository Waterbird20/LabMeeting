---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
figure.figure { margin: 0; }
.columns { align-items: center; margin-top: 0.2em; }
.columns .col:first-child { flex: 0 0 610px; }
mjx-container[display="true"] { margin: 0.25em 0 !important; }
</style>

# <span class="cat strategy">Strategy</span> Universal NOT: proper $\tau$

Different targets need different signals: no two lines may land on the same point $e^{i\Delta_m\tau}$.

$$
u(\Delta)\ \text{depends on }\Delta\text{ only through } e^{i\Delta\tau}\ (\text{up to sign})\qquad\Longrightarrow\qquad e^{iA\tau}\neq1,\ \ e^{2iA\tau}\neq1\ \iff\ A\tau\notin\pi\mathbb Z
$$

<div class="columns">
<div class="col">

<figure class="figure">

<video src="media/ch3-unot-tau-sweep.mp4" poster="media/ch3-unot-tau-sweep.png" width="600" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
<div class="col">

- Not allowed: $231.5$ ns ($m_I=\pm1$ meet) and $463.0$ ns (all three meet).
- Pulses about $x$ alone see only $\cos\tfrac{\Delta\tau}{2}$, equal for $\pm A$: hence the pulse phases.
- Any other $\tau$ works; the design uses $105.9$ ns.

</div>
</div>
