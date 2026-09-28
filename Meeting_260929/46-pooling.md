---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat method">Method</span> Pooling is a mid-circuit measurement

<!-- src: dqml-physics-results.md §0.2 items 3 to 4 (one qubit per round rotated by trainable R_Z, R_X, measured and discarded; feed-forward rotation on a remaining qubit; at n = 4 two rounds measure qubits 0 and 2, qubits 1 and 3 remain; m_i = probability of outcome 1 estimated from K shots; receiver applies U_+ if s = 1 and U_- otherwise) and Fig. 2 (T_mu on q1 / q3, U(s) on q1 / q3, readout on q1, q3). Linearity of the averaged feed-forward: §1 and App. A; deferred measurement: App. D; the link threshold is the only nonlinearity between QPUs: App. A. Clip: 3. wiki/code/lm-260929-animations/scenes_circuit.py (PoolRound); histograms (K = 100, m = 0.30 and 0.65) and bits (s = 1, 0) are illustrative and labelled so in the clip. -->
<!-- EDIT-FORWARD: the clip draws the round-0 bit arriving from above and the round-1 bit from below only for layout; both come from two-input links of the other two QPUs (§0.2, Fig. 2). -->
<!-- Speaker note: in each round one qubit gets a trainable basis rotation and a mid-circuit measurement and is traced out; a rotation T_mu conditioned on its outcome mu acts on a remaining qubit. The estimate of m = P(mu = 1) from K shots enters the decision functions; the receiving QPU gets only the message bit s, which applies U_+ or U_-. T_mu acts differently for each outcome, but averaged over mu (non-selective measurement) the QPU is still a quantum channel, linear in rho; the nonlinearity between QPUs comes from the threshold of the decision function on the estimate. In the clip q0 and q2 are pooled, q1 and q3 are measured at the end; shot counts and message bits are illustrative (labelled in the clip). (Former caption: "One QPU: q0 and q2 are pooled (measured mid-circuit), q1 and q3 are measured at the end. Shot counts and message bits are illustrative.") -->

<style scoped>.columns .col:first-child { flex: 0 0 470px; } li { margin-bottom: 0.6em; }</style>

<div class="columns">
<div class="col">

- Each round: measure one qubit mid-circuit, trace it out, apply $T_\mu$ to a remaining qubit; $4\to3\to2$ qubits, as in CNN pooling
- $K$ shots give $\hat m$; the receiver gets only the message bit $s$ ($U_+$ or $U_-$)
- Averaged over $\mu$: still a quantum channel, linear in $\rho$; the nonlinearity is the threshold on $\hat m$

</div>
<div class="col">

<figure class="figure">

<video src="media/pool-round.mp4" poster="media/pool-round.png" width="640" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
</div>
