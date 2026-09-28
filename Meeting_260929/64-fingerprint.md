---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

<style scoped>table { font-size: 17px; width: 100%; margin: 0.1em 0 0.15em; } table td, table th { padding: 2px 12px; white-space: nowrap; } table td:last-child { white-space: normal; } table td:first-child { font-weight: 600; } p { margin: 0.45em 0; } .columns { align-items: center; margin-top: 0; } .columns .col:first-child { flex: 0 0 500px; } figure.figure { margin: 0; }</style>

# <span class="cat strategy">Strategy</span> Suggestion: a distributed task, not a classification benchmark

| | now: MNIST-1D (4 digits) | proposal: inherently distributed task |
|---|---|---|
| each QPU holds | a contiguous block of one signal | senders: $x$ or $y$; receiver: none |
| label | mostly local (additive model: $0.971$) | a joint function $f(x,y)$: equality, Hamming distance, disjointness, ... |
| communication | adds $1.1$ to $3.1$ points | necessary |
| figure of merit | test accuracy | message size at a given worst-case error |
| quantum advantage | none observed: ours $\approx0.90$,<br>kernel logistic regression (shift-aware) $0.978$ | proven for some $f$, not for all |

<div class="columns">
<div class="col">

<figure class="figure">

<video src="media/fingerprint.mp4" poster="media/fingerprint.png" width="500" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
<div class="col">

**Goal:** an answer that depends jointly on inputs no single QPU holds.

**Example, equality:** each sender sends a quantum fingerprint, the referee runs a SWAP test.

<div class="src">Buhrman, Cleve, Watrous, de Wolf, Phys. Rev. Lett. 87, 167902 (2001).</div>

</div>
</div>

<!-- Speaker note: this answers "we want more than a classification task, right?": yes. On MNIST-1D the label is mostly local: a classical additive per-block model already reaches 0.971 with contiguous blocks (0.891 permuted), so communication has little to carry and adds only a few points, and a classical model on all 40 features, a kernel logistic regression with a shift-aware kernel chosen on validation data among 2330 classical models, reaches 0.978, above every quantum model. In an inherently distributed task each QPU holds its own input and the answer is a joint function f(x, y) of them, so without communication the referee can only guess; the figure of merit becomes the communication cost (message size) needed for a given worst-case error. Communication complexity offers a whole menu of such f with known quantum and classical costs: equality, Hamming distance, disjointness, hidden matching, vector in subspace, and inner product mod 2 as a counterexample with no advantage (next slide). A machine-learning flavoured variant: two QPUs each receive a whole signal and the receiving QPU decides "same digit?" (a verification task); no advantage is proven for that one. Equality is itself a binary classification of the pair (x, y): what changes is not "classification or not" but that the inputs sit on different QPUs and the figure of merit becomes message size at a given worst-case error instead of test accuracy. The "now" column is contiguous blocks; with permuted features communication adds 5.7 to 7.5 points (§4.3), but a block is then no longer a piece of the signal. -->
<!-- Speaker note (clip): Alice and Bob each send a fingerprint |h_x> = (1/sqrt m) sum_i (-1)^{E_i(x)} |i>, built from an error-correcting code E whose distinct code words differ in many positions, so |<h_x|h_y>| <= delta for x != y. The referee runs the SWAP test and answers "equal" with probability (1 + |<h_x|h_y>|^2)/2: 1 if x = y, at most (1 + delta^2)/2 otherwise; k copies push the error down. The clip uses a toy code with 16 amplitudes (4 qubits), not the paper's: one flipped bit gives <h_x|h_y> = 0.25 and P(0) = 0.53, where P(0) is the probability of "equal". -->
<!-- 2026-09-29 round 2 (user on the protocol slide: "there are more operations more than =="): the proposal column now names a family of joint functions f(x, y) instead of equality alone; the "quantum advantage" row says "proven for some f, not for all" (inner product mod 2 has none, Cleve et al. 1998, see 64a); "message size at a given error" became "... worst-case error"; the fingerprint clip moved here from 64a (as the worked example for one f) so that 64a can hold the table of functions. Fixer pass: "each QPU holds" in the proposal column changed from "its own input, x or y" to "senders: x or y; receiver: none", consistent with the SMP referee (no input) on 64a (verifier). -->
<!-- src: "none observed" = no quantum advantage measured on this benchmark, not a claim that none is possible (verifier 2026-09-29). Permuted features: communication +5.7 to +7.5 points over no communication, §4.3. -->
<!-- src: additive model 0.971 (contiguous) / 0.891 (permuted), best classical on all 40 features 0.978: dqml-physics-results.md §0.1 (four-digit task); the 0.978 model is named there since 2026-09-29: kernel logistic regression with a translation-aware kernel (sum of RBF kernels over the 40 cyclic width-5 sub-windows), chosen on validation data among 2330 candidate models; table cell uses the deck-wide short label "kernel logistic regression (shift-aware)" (was "classical 0.978"). Communication adds 1.1-3.1 points over no communication (integration 2026-09-29: was 1.5-3.1, the §4.3 topology range on seeds 0-4; widened to match the Doubt-1 slide 61-slices.md, whose 2->1 gain is 1.1 / 2.3 / 3.1 points on seeds 3-7 / 0-7 / 0-4, §3 Stage 1, §4.2, §4.3; every topology on seeds 0-4 gives 1.5-3.1, §4.3 table, fig_outlook.py _verify(), so 1.1-3.1 covers both). Ours about 0.90: 0.903 +- 0.007 (§4.3, seeds 0-4), 0.908 +- 0.002 (§2.4, seeds 0-2), 0.8895 on held-out seeds 3-7 (§3); all n = 4. -->
<!-- src: equality problem, simultaneous message passing (SMP) model, referee: 1. raw/papers/buhrman2001quantum/buhrman2001quantum.pdf p. 1, eq. (1); quantum O(log n) qubits vs classical Theta(sqrt n) bits without a shared key: Theorem 1 (p. 2). Hamming distance and disjointness in the proposal cell, and "proven for some f, not for all": see the verified table and src comments of 64a-fingerprint-protocol.md (Yao, STOC 2003; Buhrman, Cleve, Wigderson, STOC 1998; Aaronson and Ambainis 2003/2005; Razborov 2003; Cleve, van Dam, Nielsen, Tapp 1998). The mapping onto our QPUs is our proposal, not the papers'. -->
<!-- src: clip numbers from 3. wiki/code/lm-260929-animations/scenes_outlook.py _verify(): x = 10110010, y = 00110010, 6 of 16 signs differ, overlap (16 - 12)/16 = 0.25, P(0) = 17/32 = 0.531 (state-vector simulation of the SWAP test equals the formula); toy-code worst case P(0) <= 0.625. SWAP test: paper eq. (3), Fig. 1; fingerprint: eq. (5). -->
<!-- terminology (GLOSSARY.md): feature partition of one data vector vs inherently distributed task; quantum fingerprinting, simultaneous message passing, referee, SWAP test are standard (Keep list). "Sender" / "referee" as in the SMP model. -->
<!-- Layout 2026-09-29 (model named in the "quantum advantage" row, which now has two lines): table font 18px -> 17px, cell padding 3px -> 2px, table bottom margin 0.3em -> 0.15em, so that the clip still clears the table (checked in a private build; without it the clip overlapped the second line). Fixer 2026-09-29 (verifier nit): the cell was "none observed (ours ≈0.90; kernel logistic regression (shift-aware): 0.978)", reworded without nested parentheses; re-checked in a private build. -->
