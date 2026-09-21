---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
section { font-size: 20px; }
p { margin: 0.35em 0; }
</style>

# <span class="cat intro">Intro</span> The two angles are the sphere's own coordinates

<div class="columns">
<div class="col">

A ket has four real numbers. Normalisation removes one, and the global phase, which no expectation value can see, removes another. So only two real parameters are left, and by convention we represent a ket state as

$$
\ket{\psi}=\cos\tfrac{\theta}{2}\,\ket{0}+e^{i\phi}\sin\tfrac{\theta}{2}\,\ket{1}.
$$

Feeding this into $\langle X\rangle=2\,\mathrm{Re}(c_0^{*}c_1)$,
$\langle Y\rangle=2\,\mathrm{Im}(c_0^{*}c_1)$ and $\langle Z\rangle=|c_0|^2-|c_1|^2$ gives

$$
\vec{r}=\big(\sin\theta\cos\phi,\;\sin\theta\sin\phi,\;\cos\theta\big).
$$

Those are the spherical coordinates of a unit vector, which is why this parametrisation is used. $\theta$ is the polar angle from $\ket{0}$, $\phi$ the azimuth around $z$, and the half angle inside the ket is what makes the full angle $\theta$ appear on the sphere. Since $|\vec{r}|^2=\sin^2\theta+\cos^2\theta=1$, **every ket sits on the surface**. Note that the radius is the purity $|\vec{r}|^2=2\operatorname{Tr}\rho^2-1$.

</div>
<div class="col">

<figure class="figure">

<video src="media/bloch-angles.mp4" poster="media/bloch-angles.png" width="550" autoplay loop muted playsinline preload="none"></video>

*The same products for the general ket, then the arrow on the sphere with the polar angle $\theta$ (blue arc from the $z$ axis) and the azimuth $\phi$ (green arc from the $x$ axis) following it as the two angles are swept.*

</figure>

</div>
</div>
