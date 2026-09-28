---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

<style scoped>table { font-size: 24px; margin: 0.3em auto 0.7em; } table td, table th { padding: 6px 18px; } table td:first-child { white-space: nowrap; } p { margin: 0.3em 0; }</style>

# <span class="cat ongoing">Ongoing</span> Doubt 1: a feature partition is not an inherently distributed task

| classical model, test accuracy, **4 digits** | contiguous blocks | permuted features |
|---|---|---|
| best additive (per-block) model | $0.971$ | $0.891$ |
| below the best joint model, kernel logistic regression (shift-aware):<br>$0.978$ on all $40$ features | $0.7$ points | $8.7$ points |

- **Contiguous:** class information lies almost entirely within blocks; still, communication adds $1.1$ to $3.1$ points to our model (by seed set).
- **Permuted:** $8.7$ points need combinations across blocks, but the Fourier (DFT) encoding loses its motivation.
