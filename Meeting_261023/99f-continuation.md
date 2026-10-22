---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

<style scoped>
figure.figure { margin: 0; }
table { font-size: 0.78em; }
</style>

# <span class="cat results">Backup</span> Continuation in the Rabi frequency

The error grows smoothly as the drive weakens.

<figure class="figure">

![w:520](media/fig_results_continuation.png)

</figure>

| $10$ MHz, symmetric | $N=14$ ($2\pi/A$) | $N=27$ ($\pi/A$) | $N=40$ ($2\pi/3A$) |
|---|---|---|---|
| full amplitude | $1.5\times10^{-3}$ (5.8 $\mu$s) | $5.2\times10^{-4}$ (5.9 $\mu$s) | $5.2\times10^{-4}$ (5.9 $\mu$s) |
| fixed length | $1.4\times10^{-2}$ (5.7 $\mu$s) | $7.7\times10^{-4}$ (5.8 $\mu$s) | $2.6\times10^{-4}$ (6.4 $\mu$s) |

<!--
Backup: continuation in the Rabi frequency. The instantaneous sequence is first trained at Omega/2pi = 160 MHz, where the pulses are nearly instantaneous, and then retrained at 80, 40, 28, 20, 16, 13, 11.5 and 10 MHz in turn, each time from the previous optimum with the pulse centres kept in place.

The figure: for the N = 14 symmetric sequence the error rises smoothly from 5.2e-4 at 160 MHz to 1.5e-3 at 10 MHz with full-amplitude pulses, but to 2.0e-2 with fixed-length pulses, most of the rise between 40 and 16 MHz. Finite-pulse training needs several hundred iterations, more than with instantaneous pulses, and of 524 runs 39 stopped without any inversion.

The table: one or two extra pulses in every period of the N = 14 symmetric design, spacing pi/A (27 pulses) or 2 pi/3A (40 pulses), with everything retrained at 10 MHz. Full amplitude: 1.5e-3 becomes 5.2e-4 at the same duration of about 5.9 microseconds. Fixed length: 1.4e-2 becomes 7.7e-4 in 5.8 microseconds and 2.6e-4 in 6.4 microseconds, the shortest 10 MHz designs found. With instantaneous pulses the extra pulses would be useless, because the optimum is a function of z^n; with real pulses they give the training the freedom to cancel the detuning dependence of the pulses.
-->
