---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

<style scoped>
.columns .col:first-child { flex: 0 0 560px; }
figure.figure { margin: 0; }
table { font-size: 0.8em; }
</style>

# <span class="cat results">Results</span> Best uniform approximation

The best polynomial gives the best achievable error $E_N$. Here $Q=-i\,r$ with one real series $r(x)=\sum_k c_k\,T_k(x)$ of degree $K$, made by $N=K+1$ pulses.

<div class="columns">
<div class="col">

<figure class="figure">

<video src="media/qsp-best-approx.mp4" poster="media/qsp-best-approx.png" width="550" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
<div class="col">

| condition | meaning |
|---|---|
| $s_m\,r\ge\sqrt{1-E}$ in the passband | at most $E$ of the population is left behind |
| $\lvert r\rvert\le\sqrt{E}$ in the stopband | at most $E$ is transferred |
| $\lvert r\rvert\le1$ everywhere | all that unitarity asks; $P$ follows |

- For a fixed $E$ the conditions are linear in the $c_k$: a linear program finds the smallest error $E_N$.
- $N=14$ ($K=13$) gives $E_{14}=5.1\times10^{-4}$ in $6.0\,\mu\text{s}$.

</div>
</div>

<!--
From now on the flip amplitude of our sequences is Q = -i r with r real; I explain why on the slide after next. r is a Chebyshev series in x = cos(Delta tau / 2) whose degree K is the number of waits; N = K + 1 pulses.

What does it mean to meet the target with error E? Read the three conditions one by one. In the passband we want full inversion with the right sign: the population left behind is |P|^2 = 1 - r^2, so asking s_m r >= sqrt(1 - E) means at most a fraction E stays behind, and since Q is purely imaginary the phase is automatically the same on all three windows. In the stopband we want nothing to happen: the transferred population is |Q|^2 = r^2, so |r| <= sqrt(E) means at most E is transferred. And everywhere |r| <= 1: that is all unitarity asks for, because given such an r the missing entry P can always be completed.

For a fixed E all three are linear inequalities in the coefficients c_k. So we can ask a linear program: what is the smallest stopband bound on |r| we can reach while the passband and circle conditions hold? The best error E_N is where that bound equals sqrt(E). This is the best uniform approximation of the step, and no sequence of N instantaneous pulses can beat it: it is the upper bound of what QSP can do here.

The clip shows the best approximation for N = 2, 4, up to 18 pulses at tau = 2 pi / A, plotted against Delta / A: green the passband, yellow the stopband, white the target. The curve flattens on the windows and on the stopband, and its error equioscillates. On the right, E_N on a log scale: 3.1e-2 at N = 6, 3.8e-3 at 10, 5.1e-4 at 14, 7.2e-5 at 18, a factor 7 to 8 per four pulses. One more pulse does not always help: N = 7 and 8 both give 1.9e-2, N = 15 and 16 give 2.3e-4 and 2.5e-4, because consecutive N alternate between the two sign patterns. The tolerance 1e-3 is first met at N = 14, that is K = 13 waits of 463 ns, about 6.0 microseconds; N = 13 gives 1.4e-3.

Compare the cut Chebyshev series of the previous slide: it insists on the jump and stays at an error of about 0.2. The best approximation uses the free transition bands to turn smoothly, so its error falls exponentially.
-->
