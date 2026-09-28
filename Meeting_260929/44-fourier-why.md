---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>.columns .col:first-child { flex: 0 0 545px; } li { margin-bottom: 0.4em; } .def { font-size: 0.9em; margin: -0.3em 0 0 1.55em; line-height: 1.5; } .def p { margin: 0; } .callout { margin: 0.5em 0 0; } figure.figure { margin: 0.35em auto 0; }</style>

# <span class="cat method">Method</span> Why the choice of phase-carrying basis state matters

<div class="columns">
<div class="col">

- One copy: $P=\mathrm{Tr}[E\rho]$ is linear in $\rho_{xy}=\psi_x\psi_y^*$
  $$\rho_{b(m),0}=C_0A_m\sin(a_m/2)\,e^{i(\arg X_m+c_m)}$$

<div class="def">

$C_0=\psi_{0000}=\prod_{k=0,1,3,7}\cos(a_k/2)\ge0$
$|\psi_{b(m)}|=A_m\sin(a_m/2)$, $A_m\ge0$:
the other $\cos$, $\sin$ factors on its path, free of $a_m$

</div>

</div>
<div class="col">

- **Phase on $|b(m)\rangle$:** a rescaled copy of $X_m$
- **Phase on $|m{+}1\rangle$:** same angle, radius without $|X_m|$

<div class="callout">

No communication, phase on $|m{+}1\rangle\to|b(m)\rangle$:
test accuracy $0.738\to0.865$ (4 digits, seeds 0–2)

</div>

</div>
</div>

<figure class="figure">

![w:940](media/coherence-argand.png)

</figure>
