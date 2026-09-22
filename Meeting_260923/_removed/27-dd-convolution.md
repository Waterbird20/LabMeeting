---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<!-- EDIT-FORWARD: confirm ddrf-fig5-bath.png is the figure you want here; ddrf-fig2-factors.png (the three factors and their product) is also copied into media/ as an alternative. -->

# <span class="cat method">Method</span> So the measured spectrum is a convolution

<div class="callout">

We never see the bath. We see the bath smeared by our own sequence, and the smearing is a convolution.

</div>

<div class="columns">
<div class="col">

Collect one spike per nucleus into a density, and the sum over nuclei becomes an integral
against one fixed shape:

$$
W(\omega')=\frac{(\Omega\ell)^2}{8}\sum_j\mathcal{R}(A_j)^2\,
\delta_{\rm D}\big(\omega'-\omega_1^{(j)}\big),
$$

$$
\boxed{\;1-|M(\omega)|\;\simeq\;\big[\,W*|\mathcal{D}_K|^2\,\big](\omega)\;}
$$

</div>
<div class="col">

<figure class="figure">

![h:195](media/ddrf-fig5-bath.png)

*Simulated bath. Panel (a) is the weight $\mathcal{R}(A_\parallel)$ that fixes each line's height, panel (b) the spectrum a sweep of $\omega_{\rm RF}$ returns.*

</figure>

</div>
</div>

$W$ is the **nuclear spectral density**, one line per nucleus at its precession frequency
$\omega_1^{(j)}$, weighted by how well the sequence tells the two electron branches apart;
$|\mathcal{D}_K|^2$ is the **instrument function** of width $1/(N\tau)$, set by how long we
drove, never by the sample.
