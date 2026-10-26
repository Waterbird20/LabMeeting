---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
.columns { gap: 1.0rem; align-items: center; }
.columns .col:first-child { flex: 0 0 630px; }
mjx-container[display="true"] { margin: 0.3em 0 !important; }
figure.figure { margin: 0; }
</style>

# <span class="cat method">Method</span> DQ $\pi$ pulse: time mirror

A preparation followed by its time mirror is a phase-coherent DQ $\pi$ pulse.

$$
v=\begin{pmatrix}p&-q^*\\q&p^*\end{pmatrix}\ \xrightarrow{\ \text{reversed order}\ }\ v^T=\begin{pmatrix}p&q\\-q^*&p^*\end{pmatrix},\qquad u=v^Tv:\quad P=p^2+q^2=0\ \ \text{since}\ q=\pm ip
$$

<div class="columns">
<div class="col">

<figure class="figure">

![w:625](media/ch3-pi-mirror-sequence.png)

</figure>

</div>
<div class="col">

<figure class="figure">

<video src="media/ch3-pi-mirror.mp4" poster="media/ch3-pi-mirror.png" width="510" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
</div>

- At 20 MHz the mirrored preparation is the symmetric $\pi$ design, $153$ ns, twice the preparation.
- At 10 MHz it is the symmetric design with every pulse phase shifted by $\pi$, an equivalent gate.
