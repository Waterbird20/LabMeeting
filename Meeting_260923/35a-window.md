---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<!-- TODO (speaker to confirm): the DDrf panel is reused from the meeting of 2026-04-28 through the
     2026-08-18 deck. I read its axes as the RF sweep against the surviving population
     $P_x$, with $N=48$ cells of amplitude $\Omega_k=\Omega f(k)$, so that $1-P_x$ is the
     transfer. Confirm that reading, and say whether the $N=136$ panel should sit next
     to it as the resolution-versus-length statement. -->

# <span class="cat results">Results</span> Apodization in our data

<div class="columns">
<div class="col">

<figure class="figure">

![w:545](media/qsp-design-window.png)

*Nine pulses of the same total area $\sum_k\beta_k=1.35$. Hann pushes the largest side lobe from $-12.2$ dB to $-30.2$ dB and widens the main lobe by $1.46$. Fig. by `fig_qsp_design.py`.*

</figure>

</div>
<div class="col">

<figure class="figure">

![w:535](media/DDrf_Apodization_N48_focused.png)

*The same trade in our own DDrf data: $N=48$ RF cells of amplitude $\Omega_k=\Omega f(k)$, swept in $\omega_{\mathrm{RF}}$, with $1-P_x$ the transfer. (source: lab meeting 2026-04-28)*

</figure>

</div>
</div>

Shaping the $\beta_k$ is apodization, so in this limit QSP **is** classical window design. At large pulse areas the linear map from the amplitudes to the coefficients is replaced by the exact matrix convolution we started from, and that is where QSP goes further.
