---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat intro">Intro</span> Hamiltonian on Bloch Sphere

<div class="columns">
<div class="col">

A $2\times2$ Hermitian matrix has only four real parameters, so it can always be written as a linear combination of Pauli matrices,

$$
H=c_0 I+\frac{\omega}{2}\,\hat n\cdot\vec\sigma,
\qquad
\hat n=\frac{\vec c}{|\vec c\,|},
\quad
\omega=2|\vec c\,| .
$$

Here $\vec c=(c_x,c_y,c_z)$ collects the Pauli coefficients and $c_0 I$ is only a global phase. It is useful to use

$$
e^{-i\frac{\theta}{2}\hat n\cdot\vec\sigma}
=\cos\frac{\theta}{2}\,I-i\sin\frac{\theta}{2}\,\hat n\cdot\vec\sigma ,
\qquad \theta=\omega t .
$$

**This rotates the Bloch vector about the axis $\hat n$ by the angle $\theta$.** The Pauli coefficients give the axis, their magnitude times the duration gives the angle. A resonant pulse, $H=\tfrac{\Omega}{2}Y$, turns the state about $y$ by $\Omega t$; free precession, $H=\tfrac{\delta}{2}Z$, turns it about $z$ by $\delta t$. A pulse sequence is now a list of axes and angles.

</div>
<div class="col">

<figure class="figure">

<video src="media/unitary-rotation.mp4" poster="media/unitary-rotation.png" width="550" autoplay loop muted playsinline preload="none"></video>

*The example $H=2X+Y+2Z$ has $\vec c=(2,1,2)$, so the axis is $\hat n=(2,1,2)/3$ and the angle grows as $\theta=6t$; the state $\ket{0}$ precesses on a cone about $\hat n$. Then the two cases of the next slides: $\tfrac{\Omega}{2}Y$ turns $\ket{0}$ toward $\ket{+}$ by $\Omega t$, and $\tfrac{\delta}{2}Z$ turns $\ket{+}$ about $z$ by $\delta t$.*

</figure>

</div>
</div>
