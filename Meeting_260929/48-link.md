---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat method">Method</span> The decision function: a stochastic threshold (probit)

<style scoped>.columns .col:first-child { flex: 0 0 560px; } p { margin: 0.3em 0; } ul { margin-top: 0.4em; }</style>

<div class="columns">
<div class="col">

$$g=a\,m_i+b\,m_j+c\,m_im_j+d$$

$$P(s{=}1)=\pi=\Phi\Big(g\big/\sqrt{\sigma_g^2+\epsilon^2}\Big)$$

<figure class="figure">

![w:560](media/link-cdf.png)

</figure>

</div>
<div class="col">

<figure class="figure">

<video src="media/link-rule.mp4" poster="media/link-rule.png" width="580" controls autoplay loop muted playsinline preload="none"></video>

</figure>

- $m_i,m_j$: outcome-$1$ probabilities from $K$ shots; shot noise $\sigma_g\propto K^{-1/2}$
- $\Phi$: standard normal CDF; $s=1$ applies $U_+$, else $U_-$
- Corners $\{0,1\}^2$: the sign of $g$ is a Boolean function (AND, OR, XOR)

</div>
</div>

<!-- Speaker note: QPUs i and j each measure one qubit K times, giving estimated probabilities m_i and m_j of outcome 1. The estimate hat g scatters around g with the shot-noise standard deviation sigma_g (left panel: the shaded area hat g > 0 is the probability that the bit is 1, which is the standard normal CDF Phi(g / sigma_g)); epsilon = 1e-3 is a noise floor. More shots make the probit steeper, down to the hard step (right panel). The receiving QPU applies U_+ if s = 1, otherwise U_-. At the corners {0,1}^2 of the unit square the sign of g is a Boolean function of two bits (the clip: AND, OR, XOR); inside, the decision boundary g = 0 is a hyperbola centred on the saddle point (-b/c, -a/c). The bit depends on the input only if -d lies between the smallest and largest corner values of g - d. Physics: with z = 1 - 2m = <Z>, g = h_0 + h_i z_i + h_j z_j + J z_i z_j (J = c/4) is an Ising energy, and sigma_g ∝ K^(-1/2) acts as an effective temperature. -->
<!-- Speaker note (former clip caption): illustrative (a,b,c,d): AND, OR and XOR at the corners, then XOR with K increased from 10^2 to 10^4. -->
<!-- src: figure link-cdf.png: 3. wiki/code/lm-260929-animations/fig_link.py (fig_cdf, 2026-09-29, no randomness). Coefficients = one trained decision function, dqml-physics-results.md §5.0 table (seed 0, round 0, QPUs 1, 2 -> 0, normalised (a,b,c,d) = (0.81, -0.34, 0.29, -0.37), dictator m_i at the corners); operating point (m_i, m_j) = (0.58, 0.30) and the right-panel path m_j = 0.30 are illustrative. sigma_g = App. B formula (dqml_style.link_sigma): g = 0.048, sigma_g = 0.045 / 0.014 / 0.0045 at K = 1e2 / 1e3 / 1e4; pi(K = 1e2) = 0.858 (probit) vs 0.857 (exact binomial P(hat g > 0), checked in _verify_cdf together with sigma_g against the exact binomial standard deviation). -->
<!-- src: link rule, probit, sigma_g, U_+/U_-: 3. wiki/projects/dqml/dqml-physics-results.md §0.2 item 4 and App. B (epsilon = 1e-3 floor: App. B "Probit approximation"). -->
<!-- src: corner values g(0,0)=d, g(1,0)=a+d, g(0,1)=b+d, g(1,1)=a+b+c+d, input-dependent range d in (-max(0,a,b,a+b+c), -min(0,a,b,a+b+c)), saddle (-b/c,-a/c) with g* = d - ab/c: App. B. -->
<!-- src: Ising form z = 1-2m, J = c/4, h_i = -(2a+c)/4, h_j = -(2b+c)/4, h_0 = d + (2a+2b+c)/4: App. B and §12; "the shot number sets a temperature", effective temperature proportional to K^(-1/2), probit instead of Boltzmann (logistic) response: §12.1. -->
<!-- src: clip link-rule: 3. wiki/code/lm-260929-animations/scenes_link.py (heat map pi from dqml_style.link_prob, corners from dqml_style.corner_function); _verify() checks the three truth tables, the saddle points, the XOR boundary as the two lines m = 1/2, the sigma_g formula against exact binomial enumeration, the K^(-1/2) scaling, the Ising coefficients and the probit gradient. -->
<!-- EDIT-FORWARD: the clip's coefficients are illustrative and unnormalised: AND (1,1,1,-1.5), OR (1,1,-1,-0.75), XOR (1,1,-2,-0.5). The OR and XOR vectors are the two fixed directions of the d scan at their best d (dqml-physics-results.md §5.2); the model itself normalises (a,b,c,d) to unit length, which does not change pi apart from the epsilon floor. -->
<!-- EDIT-FORWARD: at XOR with d = -1/2 the saddle value is g* = 0, so the hyperbola degenerates into the two lines m_i = 1/2 and m_j = 1/2 (App. B); say "a degenerate hyperbola" if asked why the clip shows straight lines. -->
