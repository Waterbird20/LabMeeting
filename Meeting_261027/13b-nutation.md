---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
.columns { gap: 1.0rem; }
.columns .col { flex: 1 1 0; }
mjx-container[display="true"] { margin: 0.25em 0 !important; }
figure.figure { margin: 0.1em auto; }
ul { margin: 0.2em 0; }
</style>

# <span class="cat intro">Intro</span> DQ nutation

The $\ket{0}$ population oscillates differently in each manifold: one line on resonance, two off resonance.

<div class="columns">
<div class="col">

<figure class="figure">

![w:450](media/dq-nutation.png)

</figure>

$$
P_0^{(m)}(t)=\Big[\frac{\Delta_m^2}{\Omega_m'^2}+\frac{\Omega_{DQ}^2}{\Omega_m'^2}\cos\Omega_m't\Big]^2,\quad \Omega_m'=\sqrt{\Omega_{DQ}^2+\Delta_m^2}
$$

</div>
<div class="col">

<figure class="figure">

![w:450](media/dq-spectrum.png)

</figure>

$$
P_0^{(m)}=c_m+\frac{2\Delta_m^2\Omega_{DQ}^2}{\Omega_m'^4}\cos\Omega_m't+\frac{\Omega_{DQ}^4}{2\Omega_m'^4}\cos2\Omega_m't
$$

</div>
</div>

- On resonance $P_0=\cos^2\Omega_{DQ}t$: a single line at $2\Omega_{DQ}$.
- Off resonance $q<1$: $P_0$ returns only to $0.69$ at the DQ $\pi$ time, and a line at $\Omega_m'$ appears.
