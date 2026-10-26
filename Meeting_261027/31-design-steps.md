---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat strategy">Strategy</span> Design steps

The same five steps for each application, first with instantaneous pulses and then with finite pulses.

| step | question | tool |
|---|---|---|
| 1. Target | what must $P$ and $Q$ be on the bands $\Delta_m\pm3\sigma$? | the physics of the operation; free between the bands |
| 2. Degree $K$ | how many waits does the target need? | Chebyshev decomposition; conditions against parameters |
| 3. Wait $\tau$ | where do the lines land on the circle $e^{i\Delta\tau}$? | no overlap: different targets need different signals |
| 4. Solve | which angles $\{\beta_k\}$ and phases $\{\varphi_k\}$? | damped Newton iteration with the exact Jacobian |
| 5. Evaluate | how good is it on the bands? | gate or state fidelity averaged over the ¹³C shift and the three lines |

- Each ¹⁴N line $\Delta_m=A_\parallel m_I$ is broadened by the ¹³C bath, rms $\sigma/2\pi=0.1$ MHz; no detuning source lies between the lines.
- A finite pulse lasts $\lvert\beta\rvert/\Omega$ ($\Omega_{\rm DQ}=\Omega/\sqrt2$ for the DQ qubit) and depends on $\Delta$ itself, so the design is solved again with finite pulses.
