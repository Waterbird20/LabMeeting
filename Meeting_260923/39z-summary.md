---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

<!-- EDIT-FORWARD: the last bullet points at our own unpublished V_B^- conditional-gate result; drop it or expand it depending on the audience. -->

# <span class="cat ongoing">Ongoing</span> Summary

**A quantum experiment prepares a state, applies a unitary and measures. The unitary is what we design, the measurement returns a response, and that response is a convolution that QSP lets us design exactly.**

- Chapter 1 fixed the vocabulary: a state, a unitary built from a drift term and a control term, and a projective read-out whose statistics are the response.
- Chapter 2 showed that responses multiply, and therefore convolve, from the product over nuclear spins to the Dirichlet instrument function of a decoupling spectrum.
- Chapter 3 made the convolution the design variable. With $d$ waits the sequence is a degree-$d$ polynomial in $z=e^{i\delta\tau}$ assembled by $d$ two-tap convolutions, the measured curve is the Fourier transform of that tap list, weak pulses reproduce classical windowing, and the theorem says every bounded polynomial is reachable with the pulses recovered by a peel-off recursion.
- The concrete payoff is the conditional gate on the $V_B^-$ centre, where a seven-pulse sequence is exactly the identity on three hyperfine lines and $R_x(\pi)$ on the fourth [3].
