---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
section { font-size: 20px; line-height: 1.46; }
section p { margin: 0.25em 0; }
mjx-container[display] { margin: 0.15em 0 !important; }
figure.figure { margin: 0.3em auto 0; }
</style>

# <span class="cat method">Method</span> Our case: many nuclei, and the responses multiply

The electron is prepared in $\ket{+}$, a dynamical decoupling sequence runs, and it is read out at azimuth $\varphi$, where $V_0$ and $V_1$ are the propagators the **nuclear** spin experiences with the electron in $\ket{0}$ and in $\ket{1}$.

$$
P(\varphi)=\tfrac12+\tfrac12|M|\cos(\varphi-\arg M),
\qquad
M=\tfrac12\operatorname{Tr}\,V_0V_1^{\dagger}
$$

For several nuclei that do not interact with each other the trace factorizes.

$$
M=\prod\nolimits_j M_j,\qquad M_j=\tfrac12\operatorname{Tr}\,V_0^{(j)}V_1^{(j)\dagger}
$$

Each dip below is one nucleus telling the branches apart, the several combs multiply, and the plotted $P_x=\tfrac12(1+M)$ is exactly this **pointwise product**; losses multiply, and for shallow dips they add, $1-|M|\simeq\sum_j(1-|M_j|)$.

<figure class="figure">

![w:800](media/taminiau-fig2a.png)

*Taminiau et al., PRL **109**, 137602 (2012), Fig. 2(a): $P_x$ against the interpulse delay $\tau$, $N=32$ pulses, $B_0=401$ G.*

</figure>

<!-- EDIT-FORWARD: the figure is at w:800 with the page font at 20 px, which is what the
     data panel and the two equations cost together. If you want more prose back (the
     chapter-1 remark that this M is the same M, or the per-nucleus dip discussion), say
     so and the figure goes down to w:700. -->
