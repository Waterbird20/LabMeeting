---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

# <span class="cat method">Method</span> Our case: many nuclei, and the responses multiply

The electron is prepared in $\ket{+}$, a dynamical decoupling sequence runs, and the electron
is read out at azimuth $\varphi$. The whole experiment is one number,

$$
P(\varphi)=\tfrac12+\tfrac12|M|\cos(\varphi-\arg M),
\qquad
M=\tfrac12\operatorname{Tr}\,V_0V_1^{\dagger},
$$

where $V_0$ and $V_1$ are the propagators the **nuclear** spin experiences with the electron
in $\ket{0}$ and in $\ket{1}$, exactly the $M$ of chapter 1. If the two agree, $M=1$ and no
contrast is lost; a dip means the nucleus can tell the branches apart.

For several nuclei that do not interact with each other the trace factorizes,

$$
M=\prod_j M_j,\qquad M_j=\tfrac12\operatorname{Tr}\,V_0^{(j)}V_1^{(j)\dagger},
$$

so each nucleus contributes its own response to the swept variable, and what we measure is their **pointwise product**. Contrast losses multiply, and for shallow dips they simply add, $1-|M|\simeq\sum_j\big(1-|M_j|\big)$, one comb of dips per nucleus as in chapter 1 (Taminiau et al., 2012).
