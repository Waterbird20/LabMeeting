---
marp: true
theme: serif
math: mathjax
---

# Outline

1. **Double quantum with an unpolarised ¹⁴N.** The nucleus gives three detunings, and one $\pi$ pulse cannot serve all three.
2. **Quantum signal processing.** Pulses and equal waits turn the detuning into a polynomial.
3. **QSP for the DQ $\pi$ pulse.** The target is a step function of the detuning, and a polynomial approximates it. The same recipe prepares the $\pm1$ superposition and makes a universal NOT.

<div class="legend">
<span class="cat intro">Intro</span>
<span class="cat method">Method</span>
<span class="cat strategy">Strategy</span>
<span class="cat results">Results</span>
<span class="cat ongoing">Ongoing</span>
</div>

<!--
Three parts.

Part one: double quantum. A microwave field with two frequency components moves the NV spin between m_s = +1 and -1, and the +-1 superposition picks up phase twice as fast. The unpolarised 14N nucleus shifts each hyperfine manifold by Delta_m = m A, with A/2pi = 2.16 MHz, and a single rectangular pi pulse cannot serve all three.

Part two: quantum signal processing. A sequence of K equal waits and N = K + 1 pulses is a polynomial of degree K in the detuning: the waits fix the degree, the pulse angles choose the polynomial.

Part three: the DQ pi pulse with QSP. We write the target as a step function of the detuning, choose the wait time from the periodicity of the response, find the best achievable error for each number of pulses, and then turn to real pulses of finite length.
-->
