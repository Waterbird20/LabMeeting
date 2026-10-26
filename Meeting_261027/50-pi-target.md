---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
.columns { gap: 1.2rem; }
.columns .col:first-child { flex: 0 0 570px; }
mjx-container[display="true"] { margin: 0.3em 0 !important; }
figure.figure { margin: 0; }
</style>

# <span class="cat strategy">Strategy</span> DQ $\pi$ pulse: target function

The DQ $\pi$ pulse swaps $\ket{{+1}}$ and $\ket{{-1}}$ with the same spin-1 unitary on all three lines.

<div class="columns">
<div class="col">

<figure class="figure">

![w:560](media/ch3-pi-target.png)

</figure>

</div>
<div class="col">

$$
P=0:\qquad D^1(u)=\begin{pmatrix}0&0&Q^{*2}\\0&-1&0\\Q^2&0&0\end{pmatrix}
$$

The same on every line iff $Q_m^2=Q_0^2$: target $P=0$, $Q=\mp i$ on the bands.

$$
F=\frac{(4t^2-1)^2+3}{12},\qquad t=\lvert\mathrm{Im}\,Q\rvert
$$
$$
1-F\simeq2\big(\lvert P\rvert^2+(\mathrm{Re}\,Q)^2\big)
$$

- A common $U$ has $US_zU^\dagger=-S_z$, so $e^{-i\Delta S_zt}\,U\,e^{-i\Delta S_zt}=U$ for every $\Delta$.
- A train of these pulses refocuses the hyperfine and static ¹³C shifts.

</div>
</div>
