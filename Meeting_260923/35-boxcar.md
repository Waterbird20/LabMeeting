---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat results">Results</span> Weak pulses: the taps are the pulse amplitudes

<div class="columns">
<div class="col">

Expand each pulse to first order, $G_k\simeq I-\tfrac{i}{2}\beta_k(\cos\varphi_k X+\sin\varphi_k Y)$:
$$
\hat Q(\delta)\simeq-\frac{i}{2}\sum_{k=0}^{d}\beta_k e^{i\varphi_k}\,e^{i\left(\frac{d}{2}-k\right)\delta\tau}.
$$
The tap $q_k$ **is** the complex pulse amplitude $\beta_ke^{i\varphi_k}$. Equal pulses give the all-ones boxcar of length $N=d+1$, hence
$$
|\hat Q(\delta)|^{2}=\frac{(N\beta)^{2}}{4}\,\big|\mathcal{D}_N(\delta\tau)\big|^{2},
\quad
\mathcal{D}_N(\theta)=\frac{\sin(N\theta/2)}{N\sin(\theta/2)},
$$
the Dirichlet kernel that Section 2 met as the instrument function of a decoupling spectrum.

</div>
<div class="col">

<figure class="figure">

![w:470](media/DDrf_Apodization_N48_focused.png)

*The same trade in our own DDrf data: per-cell amplitudes $\Omega_k=\Omega f(k)$ flatten the detuned region at the cost of a wider resonance, at $N=48$ cells. (source: lab meeting 2026-04-28)*

</figure>

</div>
</div>

Shaping the $\beta_k$ is apodization, so in this limit QSP is classical window design. At large pulse areas the linear map from amplitudes to taps is replaced by the exact matrix convolution we started from, and that is where QSP goes further.
