---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
section { font-size: 20px; }
p { margin: 0.35em 0; }
.columns { gap: 1.0rem; }
.columns .col:first-child { flex: 0 0 450px; }
</style>

# <span class="cat intro">Intro</span> The two angles are the sphere's own coordinates

<div class="columns">
<div class="col">

A ket has four real numbers. Normalisation removes one, and the global phase, which no expectation value can see, removes another. Only two real parameters are left, so by convention we write

$$
\ket{\psi}=\cos\tfrac{\theta}{2}\,\ket{0}+e^{i\phi}\sin\tfrac{\theta}{2}\,\ket{1}.
$$

With $\langle X\rangle=2\,\mathrm{Re}(c_0^{*}c_1)$, $\langle Y\rangle=2\,\mathrm{Im}(c_0^{*}c_1)$ and $\langle Z\rangle=|c_0|^2-|c_1|^2$ this gives

$$
\vec{r}=\big(\sin\theta\cos\phi,\;\sin\theta\sin\phi,\;\cos\theta\big),
$$

the spherical coordinates of a unit vector. $\theta$ is the polar angle from $\ket{0}$, $\phi$ the azimuth about $z$, and the half angle inside the ket is what makes the full angle $\theta$ appear on the sphere. Since $|\vec{r}|=1$, **every ket sits on the surface**; the radius is the purity, $|\vec{r}|^2=2\operatorname{Tr}\rho^2-1$.

</div>
<div class="col">

<figure class="figure">

<video src="media/bloch-angles.mp4" poster="media/bloch-angles.png" width="700" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
</div>
