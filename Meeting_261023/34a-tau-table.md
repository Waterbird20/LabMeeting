---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
table { font-size: 0.72em; }
</style>

# <span class="cat results">Results</span> Choice of $\tau$

The wait $\tau=2\pi/A$ needs the fewest pulses.

| $\tau$ | $\tau$ (ns) | period of $Q^2$ | smallest $N$ with $E\le10^{-3}$ | free evolution $K\tau=(N-1)\tau$ |
|---|:-:|:-:|:-:|:-:|
| $2\pi/A$ | 463.0 | $A$ | **14** | 6019 ns |
| $\pi/A$ | 231.5 | $2A$ | 27 | 6019 ns |
| $2\pi/3A$ | 154.3 | $3A$ | 40 | 6019 ns |
| $2\pi/4A$ | 115.7 | $4A$ | 51 | 5787 ns |
| $2\pi/6A$ | 77.2 | $6A$ | 75 | 5710 ns |
| ${\color{#b91c1c}2\pi/1.5A}$ | 308.6 | $1.5A$ | ${\color{#b91c1c}E\ge0.50}$ for any $N$ | |
| ${\color{#b91c1c}2\pi/2.5A}$ | 185.2 | $2.5A$ | ${\color{#b91c1c}E\ge0.50}$ for any $N$ | |

- At $\pi/A$ and $2\pi/3A$ the optimum is the $2\pi/A$ sequence with zero-angle pulses in between.
- Shorter waits save 4 to 5 % of free evolution at four to five times the pulses.
- Incompatible $\tau$ ask one detuning to be inverted and unchanged: $E\ge1/2$.

<!--
Instantaneous pulses, A/2π = 2.16 MHz, so 2π/A = 463.0 ns. τ = 2π/A needs N = 14 pulses (K = 13 waits). At π/A and 2π/3A the smallest degree is exactly two and three times larger, 27 and 40 pulses, with the same 6019 ns of free evolution: the best sequence is the 2π/A sequence with zero-angle pulses in between, every second or third pulse doing something. At 2π/4A and 2π/6A, 51 and 75 pulses give 5787 and 5710 ns, 4 and 5 % less free evolution, because the three windows are then separate points of the circle and part of the circle lies beyond 1.5A, where nothing is required. The incompatible values 2π/1.5A and 2π/2.5A stay at E = 0.50 for any number of pulses.

So with instantaneous pulses τ = 2π/A is the natural choice: saving a few per cent of free evolution would cost four to five times as many pulses, and every real pulse adds its own duration and error.
-->
