---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat results">Results</span> Backup: the full quantum-channel table

<style scoped>table { font-size: 0.74em; margin: 0.2em auto 0.4em; } td, th { padding: 0.12em 0.55em; }</style>

Three QPUs with $n=4$ qubits, revised encoding, seeds 0–4; accuracy and test cross-entropy (CE, nats).

| channel (what arrives at the receiver) | contiguous | CE | permuted | CE |
|---|---|---|---|---|
| none: fresh $\vert 00\rangle$, sender applies its feed-forward | $0.875\pm0.016$ | $0.495$ | $0.800\pm0.016$ | $0.502$ |
| none: fresh $\vert 00\rangle$, no feed-forward | $0.869$ | | $0.789$ | |
| none: the receiver keeps its own qubits unmeasured | $0.871\pm0.023$ | $0.502$ | $0.796\pm0.012$ | $0.504$ |
| measure once, send the outcome; sender also uses it | $0.877\pm0.013$ | $0.416$ | $0.795\pm0.012$ | $0.510$ |
| measure once, send the outcome; sender does not use it | $0.894\pm0.010$ | $0.398$ | $0.794\pm0.018$ | $0.508$ |
| **the qubits themselves (quantum channel)** | $0.903\pm0.008$ | $0.375$ | $0.818\pm0.009$ | $0.478$ |
| threshold test on a $10^3$-shot estimate | $0.883\pm0.009$ | $0.403$ | $0.838\pm0.039$ | $0.449$ |

| recovered fraction $R'$ | accuracy | cross-entropy | label information at the receiver |
|---|---|---|---|
| contiguous | $0.75\pm0.16$ | $0.81\pm0.15$ | $0.71\pm0.07$ |
| permuted | $0.17\pm0.35$ | $0.36\pm0.33$ | $0.33\pm0.10$ |

The originally pre-registered ratio, in which the sender also uses its outcome, is $0.07\pm0.33$ (contiguous) and $-0.31\pm0.64$ (permuted); reusing the outcome costs $1.7\pm0.7$ points on contiguous windows.

<!-- src: dqml-physics-results.md §8.1 (channel table, R' table, "The originally pre-registered ratio" bullet). Blank CE cells: not reported on the page. -->
