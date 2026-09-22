---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<!-- TODO (speaker to confirm): this page is the concrete, numerical example; `37-pq-reachable.md`
     makes the same Chebyshev statement again inside the general theory. Keep the
     example here and trim the repeat there, or the other way round. -->

# <span class="cat method">Method</span> An example

<style scoped>
.columns { gap: 1.0rem; }
.columns .col:first-child { flex: 0 0 452px; }
</style>

<div class="columns">
<div class="col">

Switch every pulse off. The sequence is bare precession, $U(\delta)=W(\delta)^{d}$, and the amplitude we read is one pure harmonic:
$$
\begin{aligned}
\bra{0}U\ket{0}&=e^{-i\,d\,\delta\tau/2},\\
\mathrm{Re}\,\bra{0}U\ket{0}&=\cos\!\big(d\,\tfrac{\delta\tau}{2}\big)=T_d(a),
\end{aligned}
$$
with $a=\cos(\delta\tau/2)$. At $d=1$ that is $a$, at $d=2$ it is $2a^{2}-1$, and at $d=5$ it is $16a^{5}-20a^{3}+5a$. These are the Chebyshev polynomials, the textbook's $P=T_d(a)$ and $Q=U_{d-1}(a)$.

</div>
<div class="col">

<figure class="figure">

<video src="media/qsp-chebyshev.mp4" poster="media/qsp-chebyshev.png" width="700" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
</div>

Now switch three pulses on at $d=2$, with the areas $\beta=(\tfrac{\pi}{4},\tfrac{\pi}{2},\tfrac{\pi}{4})$ about $y$. The same read-out becomes $\mathrm{Re}\,\bra{0}U\ket{0}=a^{2}-1=\tfrac12T_2(a)-\tfrac12$, a different polynomial of the same degree. **The waits fix the degree, and the pulses choose which polynomial of that degree we measure.**
