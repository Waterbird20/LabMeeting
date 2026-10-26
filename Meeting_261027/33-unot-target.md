---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
figure.figure { margin: 0 0 0.2em; }
.columns { margin-top: 0; align-items: center; }
.columns .col:first-child { flex: 0 0 600px; }
table { font-size: 0.8em; margin: 0; }
th, td { padding: 0.25em 0.6em; }
mjx-container[display="true"] { margin: 0.2em 0 !important; }
</style>

# <span class="cat strategy">Strategy</span> Universal NOT: target function

The target fixes $P$ and $Q$ on the three bands $\Delta_m\pm3\sigma$ and leaves them free in between.

<figure class="figure">

![w:1150](media/ch3-unot-target.png)

</figure>

<div class="columns">
<div class="col">

| $m_I$ | $\Delta_m/2\pi$ | gate | $P$ | $Q$ |
|---|---|---|---|---|
| $+1$ | $-2.16$ MHz | $R_z(\pi)=-iZ$ | $-i$ | $0$ |
| $0$ | $0$ | $R_y(\pi)=-iY$ | $0$ | $1$ |
| $-1$ | $+2.16$ MHz | $R_x(\pi)=-iX$ | $0$ | $-i$ |

</div>
<div class="col">

$$
F_m=\frac{\lvert\mathrm{Tr}\,T_m^\dagger u\rvert^2+2}{6}
$$

- Light bars show $-T_m$, which is the same gate.
- Each band has the rms ¹³C width $\sigma/2\pi=0.1$ MHz.

</div>
</div>
