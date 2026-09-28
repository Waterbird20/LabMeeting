---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat results">Results</span> Most of the gain came from the Fourier (DFT) encoding

<style scoped>table { font-size: 0.88em; margin: 0.45em 0 0.55em; } th, td { padding: 0.35em 0.6em; } td:nth-child(1), td:nth-child(2) { white-space: nowrap; } li { margin-bottom: 0.15em; } p { margin: 0.2em 0; }</style>

Block DFT $X_m=\lvert X_m\rvert e^{i\varphi_m}$; a cyclic shift only moves phases: $X_m\to e^{-2\pi i sm/14}X_m$.
NC / CC: no / classical communication ($2\to1$); 4 digits, seeds $0$–$2$, mean $\pm$ s.d.

| encoding | state ($a_m$: tree angle set by $\lvert X_m\rvert$) | linear classifier<br>(untrained) | trained, NC | trained, CC |
|---|---|---|---|---|
| (1) tree state,<br>phase on $\lvert m{+}1\rangle$ | $\psi_{m+1}\propto\sin\tfrac{\theta_t}{2}\,e^{i\varphi_m}$,<br>$\theta_t$: another node's angle | $0.773$ | $0.738\pm0.014$ | $0.848\pm0.027$ |
| **(2) tree state,<br>phase on $\lvert b(m)\rangle$** | $\psi_{b(m)}\propto\sin\tfrac{a_m}{2}\,e^{i\varphi_m}$ | $0.943$ | $\mathbf{0.865\pm0.014}$ | $\mathbf{0.908\pm0.002}$ |

- **(2) vs (1):** $+12.7\pm1.1$ points without, $+5.9\pm1.6$ with CC (three of three seeds).
- **Less underfitting:** training accuracy rises as much; NC gap $4.1\to1.6$ points.
