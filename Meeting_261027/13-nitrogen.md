---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
table { font-size: 0.82em; }
</style>

# <span class="cat intro">Intro</span> Hyperfine structure

The ¹⁴N couples to the electron mainly through $A_\parallel S_zI_z$, which is block diagonal in $m_I$.

$$
H=DS_z^2+\gamma_eBS_z+A_\parallel S_zI_z+A_\perp(S_xI_x+S_yI_y)+PI_z^2-\gamma_nBI_z+f(t)\Big(S_x-\tfrac{\gamma_n}{\gamma_e}I_x\Big)
$$

| term | magnitude ($/2\pi$), at 44 mT | effect on the electron transitions |
|---|---|---|
| $A_\parallel S_zI_z$ | $-2.16$ MHz | detuning $A_\parallel m_I$, comparable to the Rabi frequency |
| $A_\perp(S_xI_x+S_yI_y)$ | $-2.70$ MHz | second-order shifts of 2 to 5 kHz |
| $PI_z^2$, $-\gamma_nBI_z$ | $-4.95$ MHz, $-135$ kHz | same for every $m_s$: negligible |
| nuclear drive | 7 kHz | far off resonance |

Secular approximation: drop $A_\perp$ and the nuclear drive. Then $H$ commutes with $I_z$:

$$
H=\sum_{m=-1}^{1}H_m\otimes\ketbra{m}{m},\qquad
H_m=DS_z^2+(\gamma_eB+A_\parallel m)\,S_z+f(t)\,S_x+\text{const}.
$$

<p class="src">Doherty et al., Phys. Rep. 528, 1 (2013)</p>
