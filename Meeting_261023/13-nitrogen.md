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

| term | size ($/2\pi$), at 44 mT | effect on the electron transitions |
|---|---|---|
| $A_\parallel S_zI_z$ | $-2.16$ MHz | detuning $A_\parallel m_I$, comparable to the Rabi frequency |
| $A_\perp(S_xI_x+S_yI_y)$ | $-2.70$ MHz | second-order shifts of 2 to 5 kHz |
| $PI_z^2$, $-\gamma_nBI_z$ | $-4.95$ MHz, $-135$ kHz | same for every $m_s$: cancel |
| nuclear drive | 7 kHz | far off resonance |

Secular approximation: drop $A_\perp$ and the nuclear drive. Then $H$ commutes with $I_z$:

$$
H=\sum_{m=-1}^{1}H_m\otimes\ketbra{m}{m},\qquad
H_m=DS_z^2+(\gamma_eB+A_\parallel m)\,S_z+f(t)\,S_x+\text{const}.
$$

<p class="src">Doherty et al., Phys. Rep. 528, 1 (2013)</p>

<!--
The electron and the 14N nucleus.

Natural nitrogen is 14N, nuclear spin I = 1. The full Hamiltonian of the electron spin S and the nuclear spin I: zero-field splitting D S_z^2, electron Zeeman gamma_e B S_z, the hyperfine interaction with a secular part A_par S_z I_z and a transverse part A_perp (S_x I_x + S_y I_y), the nuclear quadrupole P I_z^2, the nuclear Zeeman term -gamma_n B I_z, and the drive f(t), which couples to the electron and, about 10^4 times more weakly, to the nucleus.

The sizes (Doherty et al., Phys. Rep. 528, 1 (2013)). A_par/2pi = -2.16 MHz: the only term comparable to the Rabi frequency; it shifts the electron transitions by A_par m_I. A_perp/2pi = -2.70 MHz, but it flips m_s and m_I together across gaps of order D +- gamma_e B, so it acts only in second order: shifts of about A_perp^2 / (D -+ gamma_e B), 4.5 kHz and 1.8 kHz at 44 mT, a thousand times smaller than A. The quadrupole, -4.95 MHz, and the nuclear Zeeman term, 135 kHz at 44 mT, are the same for every m_s, so they cancel in the electron transitions. The nuclear drive, about 7 kHz, is far off any nuclear resonance.

Secular approximation: drop A_perp and the nuclear drive. Then H commutes with I_z, and in the eigenbasis |m> of I_z it is block diagonal: H = sum_m H_m (x) |m><m|. Each block is the electron alone, with the Zeeman term shifted by A_par m. The constant is P m^2 - gamma_n B m, a phase within each block. Each m defines a hyperfine manifold. The next slide writes H_m in the rotating frame.

All of this assumes an unpolarised nucleus, p_m = 1/3. TODO (speaker to confirm): our field, 44 mT, is close to the excited-state level anticrossing (about 51 mT), where optical pumping polarises the 14N nucleus (Jacques et al., PRL 102, 057403 (2009)). If the nucleus is partly polarised in our set-up, the weights are not 1/3 and should be measured, for example from the three hyperfine resonances in pulsed ODMR.
-->
