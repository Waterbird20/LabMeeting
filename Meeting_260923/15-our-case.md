---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

<!-- EDIT-FORWARD: the middle lane is labelled "microwave" because the operation we design
     drives the NV electron spin. If the sequence in question drives a nuclear spin, relabel
     that lane "radio frequency". -->

# <span class="cat intro">Intro</span> Our case: prepare $\ket{0}$, read out along $z$

<figure class="figure">

<div class="seqfig"><div class="seqrow"><span class="seqname">laser</span><span class="seqbar"><span class="pulse p-init">initialise</span><span class="pulse p-read">read out</span></span></div><div class="seqrow"><span class="seqname">microwave</span><span class="seqbar"><span class="pulse p-op">operation</span></span></div><div class="seqrow"><span class="seqname"></span><span class="seqaxis"><span class="seqtime">time</span></span></div></div>

*A laser pulse pumps the spin into $\ket{0}$, the unitary we designed acts, and a second laser pulse returns a photon count.*

</figure>

Strictly, the last step is a projective measurement on $\ket{0}$ rather than a measurement of $Z$, because a photon count is intrinsically positive. The observable is the projector $\Pi_0=\ket{0}\bra{0}$, not an operator with eigenvalues $\pm 1$. The two are one affine step apart, since $\Pi_0=\tfrac12(I+Z)$:

$$
P(0)=\bra{\psi}\Pi_0\ket{\psi}=\frac{1+\langle Z\rangle}{2},
\qquad\quad
\langle Z\rangle=2P(0)-1 .
$$

The mean count is linear in $P(0)$ between a bright and a dark reference level, so dividing by that contrast turns photons into $P(0)$. **After contrast normalisation the readout is effectively a $Z$ measurement**, and $\langle Z\rangle$ is what every curve in this talk plots.

<style>
.seqfig { width: 86%; margin: 0.15em auto 0.15em; font-family: 'Inter', system-ui, sans-serif; }
.seqrow { display: flex; align-items: center; height: 38px; }
.seqname { width: 130px; flex: none; text-align: right; padding-right: 14px; color: #6b7280; font-size: 0.60em; }
.seqbar { position: relative; flex: 1 1 0; height: 32px; border-bottom: 2px solid #ddd8cc; }
.pulse { position: absolute; top: 0; height: 32px; line-height: 32px; text-align: center; color: #fff; font-size: 0.55em; font-weight: 600; letter-spacing: 0.02em; border-radius: 4px; }
.p-init { left: 1%; width: 18%; background: #15803d; }
.p-op { left: 34%; width: 24%; background: #b91c1c; }
.p-read { left: 68%; width: 22%; background: #15803d; }
.seqaxis { position: relative; flex: 1 1 0; height: 0; border-top: 1.5px solid #4b4b4b; margin-top: 6px; }
.seqaxis::after { content: ""; position: absolute; right: -10px; top: -6px; width: 0; height: 0; border-left: 10px solid #4b4b4b; border-top: 6px solid transparent; border-bottom: 6px solid transparent; }
.seqtime { position: absolute; right: -2px; top: 8px; color: #6b7280; font-size: 0.52em; }
</style>
