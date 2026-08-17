---
marp: true
theme: serif
math: mathjax
---

<style scoped>
section { font-size: 22px; }
</style>

# <span class="cat intro">Recap</span> The paper in one question

**How should a post-selection filter be designed to maximize the signal-domain conditional QFI in spin-qutrit sensing under dephasing?**

The single-sensor part is settled (shown in the June 18 and July 2 meetings):

- The field writes a phase $\phi = \gamma_e B\, t_s$ into the sensing qubit while dephasing shrinks its coherence, $\eta = e^{-t_s/T_2}$.
- After encoding, a **filter** $K = D_\gamma V$ acts: a rotation $V$, then a partial-strength coupling of the sensing levels to the auxiliary $\ket{+1}$ level; only runs that stay in the sensing subspace are kept ("conditional" = per accepted event).
- The **matched** (state-aligned) filter and its operating point are closed-form, and they recover the QFI that dephasing had erased.

The storyline: **(1)** spin-qutrit post-selection under dephasing, **(2)** phase-QFI versus signal-QFI, **(3)** state-filter alignment and the optimal filter, **(4)** the NV proof-of-principle experiment, **(5)** the readout-limited sensitivity advantage, and **(6) the collective two-sensor extension (my part, today).**

<!-- EDIT-FORWARD: storyline follows Geonhee Kim's flow (2026-07 discussion); confirm wording of item 5 with GH before presenting. -->
