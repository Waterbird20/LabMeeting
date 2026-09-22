---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
section { font-size: 20px; }
.columns { gap: 1.0rem; }
.columns .col:first-child { flex: 0 0 450px; }
</style>

# <span class="cat intro">Intro</span> Hamiltonian on Bloch Sphere

<div class="columns">
<div class="col">

A $2\times2$ Hermitian matrix has four real parameters, so it is always a combination of Pauli matrices,

$$
H=c_0 I+\frac{\omega}{2}\,\hat n\cdot\vec\sigma,
\qquad
\hat n=\frac{\vec c}{|\vec c\,|},
\quad
\omega=2|\vec c\,| ,
$$

with $\vec c=(c_x,c_y,c_z)$ the Pauli coefficients and $c_0 I$ a global phase. Then

$$
e^{-i\frac{\theta}{2}\hat n\cdot\vec\sigma}
=\cos\frac{\theta}{2}\,I-i\sin\frac{\theta}{2}\,\hat n\cdot\vec\sigma ,
\qquad \theta=\omega t ,
$$

**rotates the Bloch vector about $\hat n$ by $\theta$**: the Pauli coefficients give the axis, their magnitude times the duration gives the angle. A resonant pulse $H=\tfrac{\Omega}{2}Y$ turns the state about $y$ by $\Omega t$; free precession $H=\tfrac{\delta}{2}Z$ about $z$ by $\delta t$.

</div>
<div class="col">

<figure class="figure">

<video src="media/unitary-rotation.mp4" poster="media/unitary-rotation.png" width="620" controls autoplay loop muted playsinline preload="none"></video>

*$H=2X+Y+2Z$ has $\vec c=(2,1,2)$, so $\hat n=(2,1,2)/3$ and $\theta=6t$; the state $\ket{0}$ precesses on a cone about $\hat n$. Then $\tfrac{\Omega}{2}Y$ turns $\ket{0}$ toward $\ket{+}$ by $\Omega t$, and $\tfrac{\delta}{2}Z$ turns $\ket{+}$ about $z$ by $\delta t$.*

</figure>

</div>
</div>
