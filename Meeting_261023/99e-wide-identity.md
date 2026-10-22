---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

<style scoped>
figure.figure { margin: 0; }
.columns .col:last-child { flex: 0 0 340px; }
</style>

# <span class="cat results">Backup</span> Wide stopband

A stopband over $\lvert\Delta\rvert\le3A$ needs 104 pulses for $E=10^{-3}$.

<div class="columns">
<div class="col">

<figure class="figure">

![w:600](media/fig_results_wide.png)

</figure>

</div>
<div class="col">

$$
\tau=\frac{2\pi}{6A}
$$

| $E$ | $N$ | $K\tau$ |
|---|---|---|
| $0.1$ | 36 | 2.7 $\mu$s |
| $10^{-2}$ | 68 | 5.2 $\mu$s |
| $10^{-3}$ | 104 | 7.9 $\mu$s |

</div>
</div>

- At $\tau=2\pi/3A$ the smallest degree is $K=39$; $K\tau$ stays between $5.9$ and $6.8\,\mu\text{s}$ for every $\tau$.
- Smoothed step series in $w=e^{i\Delta\tau}$: degree 48 instead of 21 at $2\pi/3A$, 109 instead of 41 at $2\pi/6A$.

<!--
Backup: an earlier version of the target asked for the identity on the whole range |Delta| <= 3A outside the windows and transition bands, at tau = 2 pi / 6A = 77 ns, so that one period of Q^2 covers |Delta| <= 3A.

Best uniform approximations: an error of 0.1 needs 36 instantaneous pulses, 2.7 microseconds of free evolution; 1e-2 needs 68, 5.2 microseconds; 1e-3 needs 104, 7.9 microseconds. Compare 14 pulses and 6.0 microseconds for the narrower stopband of the main part.

The smallest degree K that meets the tolerance falls roughly as 1/tau, while the total free evolution K tau stays between 5.9 and 6.8 microseconds for every tau: at 2 pi / 3A the smallest degree is 39, at least 5.9 microseconds for any pulse shape.

The step target written as a Laurent series in w = e^{i Delta tau} and smoothed across the bands needs degree 48 instead of 21 at 2 pi / 3A, and 109 instead of 41 at 2 pi / 6A: it fixes the shape inside the bands, which the best approximation chooses freely.

The figure: the target, minus one, plus one, minus one on the windows and zero outside, with the best approximations for N = 17, 39 and 104 pulses.
-->
