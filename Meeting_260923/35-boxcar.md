---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat method">Method</span> Weak pulses

<div class="columns">
<div class="col">

Expand each pulse to first order, $G_k\simeq I-\tfrac{i}{2}\beta_k(\cos\varphi_k X+\sin\varphi_k Y)$, and keep the terms with one pulse in them. Each wait left of pulse $k$ contributes $e^{+i\delta\tau/2}$ and each wait to its right $e^{-i\delta\tau/2}$, so
$$
\hat Q(\delta)\simeq-\frac{i}{2}\sum_{k=0}^{d}\beta_k e^{i\varphi_k}\,e^{i\left(\frac{d}{2}-k\right)\delta\tau}.
$$
The tap $q_k$ **is** the complex amplitude $\beta_ke^{i\varphi_k}$, so the knobs on the bench are the coefficient list. Equal pulses are the all-ones boxcar of length $N=d+1$, and its transform is the Dirichlet kernel,
$$
|\hat Q(\delta)|^{2}=\frac{(N\beta)^{2}}{4}\,\big|\mathcal{D}_N(\delta\tau)\big|^{2},
\quad
\mathcal{D}_N(\theta)=\frac{\sin(N\theta/2)}{N\sin(\theta/2)},
$$
which is the instrument function of chapter 2.

</div>
<div class="col">

<figure class="figure">

![w:540](media/qsp-design-boxcar.png)

*Left: the transfer of $N=9$ equal pulses of area $\beta=0.15$, from the exact product $U=G_dW\cdots WG_0$ (solid) and from the formula above (dashed). Every null and every side lobe coincides, and only the peak differs, $\sin^{2}(N\beta/2)=0.391$ against $(N\beta)^{2}/4=0.456$. Right: that largest deviation is $17\%$ of the peak here, falls as $\beta^{2}$, and is $1.7\%$ at $\beta=0.05$. From our own evaluator, `fig_qsp_design.py`.*

</figure>

</div>
</div>
