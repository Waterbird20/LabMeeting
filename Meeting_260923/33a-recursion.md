---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<!-- EDIT-FORWARD: the clip uses equal pulse areas beta = pi/5 rather than the suggested
     pi/4, because five pulses of pi/5 add up to pi and close the notch exactly at the
     last beat, where five of pi/4 overshoot pi and reopen it (0.146 at d = 4). Say if
     you would rather have pi/4 or a windowed set; only BETAS in scenes_qsp.py changes. -->

# <span class="cat method">Method</span> Each added step convolves the coefficient list

<style scoped>
.columns { gap: 1.2rem; }
.columns .col:first-child { flex: 0 0 495px; font-size: 0.88em; }
</style>

<div class="columns">
<div class="col">

The state after $k$ steps is a list of $2$-vectors, one entry per number of waits:
$$
\tilde U_k\ket{0}=\sum_{n}v^{(k)}_n\,z^{n} .
$$
The coefficient of $z^{n}$ in $G_kA(z)\,\tilde U_{k-1}$ is
$$
v^{(k)}_n=K^{(0)}_k v^{(k-1)}_n+K^{(1)}_k v^{(k-1)}_{n-1} ,
$$
which is a flip-and-slide sum with matrix taps, $\big(K_k*v^{(k-1)}\big)_n$. Unrolled from $v^{(0)}_0=G_0\ket{0}$:
$$
\begin{aligned}
v^{(1)}_0&=K^{(0)}_1v^{(0)}_0 , & v^{(1)}_1&=K^{(1)}_1v^{(0)}_0 ,\\
v^{(2)}_0&=K^{(0)}_2v^{(1)}_0 , & v^{(2)}_2&=K^{(1)}_2v^{(1)}_1 ,\\
v^{(2)}_1&=K^{(0)}_2v^{(1)}_1 &&+\,K^{(1)}_2v^{(1)}_0 .
\end{aligned}
$$
At either end the kernel hangs off the list, so only one term survives and each step adds one entry. **The degree is the number of waits.**

</div>
<div class="col">

<figure class="figure">

<video src="media/qsp-coeff-conv.mp4" poster="media/qsp-coeff-conv.png" width="640" controls autoplay loop muted playsinline preload="none"></video>

*Five pulses $G_k=R_x(\pi/5)$, whose areas add to $\pi$, and four waits. The reversed kernel $\big(K^{(1)}_k,K^{(0)}_k\big)$ slides along the old list from $n=0$ to $n=k$, and each stop writes one entry of the new list; the printed numbers are the entries of $v^{(k)}_n$. On the right the response $|\langle0|U|0\rangle|^{2}$ over $\delta\tau$ gains one extremum per degree.*

</figure>

</div>
</div>
