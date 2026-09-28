---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat results">Results</span> How much of the quantum-communication gain does classical communication recover?

<style scoped>.columns .col:first-child { flex: 0 0 580px; } section h1 { font-size: 1.29em; margin-bottom: 0.5em; } section li { margin: 0.2em 0 0.55em; } section ul { margin-top: 0.4em; }</style>

<div class="columns">
<div class="col">

<figure class="figure">

![w:570](media/channels.png)

</figure>

</div>
<div class="col">

$$R'=\frac{A_\text{CC}-A_\text{NC}}{A_\text{QC}-A_\text{NC}}$$

<div class="src">

$A$: test accuracy with no (NC), classical (CC) or quantum communication (QC), no sender feed-forward; four digits, $n=4$, five seeds.

</div>

- **Contiguous blocks:** $R'=0.75\pm0.16$; QC beats CC by only $0.8\pm0.6$ points.
- **Permuted features:** $R'=0.17\pm0.35$; QC gives the receiver $3\times$ the information about the class label.
- **Received qubits are processed coherently:** dephasing them costs $0.45$ / $0.23$ nats; they are entangled with each other in $96$–$97\%$ of test signals.

</div>
</div>

<!-- Speaker note (from the removed caption): top three rows are no communication (NC). Otherwise each QPU measures its two pooled qubits once and sends the outcomes (CC), or sends the pooled qubits to the next QPU (QC); the last row sends a message bit thresholded from 10^3 shots (CC). n = 4, seeds 0-4, mean +- sd. -->
<!-- Speaker note: caveat, the 10^3-shot threshold bit is the best model on permuted features (0.838 +- 0.039), but it uses many copies of the sender's state, not one transmitted qubit: an estimated expectation value is a useful message, not proof that classical beats quantum at equal resources. Next (proposed, not yet run): classical messages of increasing precision in place of the qubit, from one bit to the full reduced state, which reproduces QC for this readout. -->
<!-- src: dqml-physics-results.md §8.1 (70 runs; 3 QPUs x 4 qubits, revised encoding, seeds 0-4; 2026-09-28). Table of channels (contiguous / permuted accuracy): fresh |00>, sender feed-forward 0.875 +- 0.016 / 0.800 +- 0.016; fresh |00>, no feed-forward 0.869 / 0.789; receiver keeps own qubits 0.871 +- 0.023 / 0.796 +- 0.012; measure once, sender uses the outcome 0.877 +- 0.013 / 0.795 +- 0.012; measure once, sender does not use it 0.894 +- 0.010 / 0.794 +- 0.018; quantum channel 0.903 +- 0.008 / 0.818 +- 0.009; 10^3-shot threshold 0.883 +- 0.009 / 0.838 +- 0.039. Figure: fig_rescomm.py (fig_channels), numbers copied from the table. -->
<!-- src: R' definition (numerator: measure once, sender does not use the outcome; baseline: no communication, no feed-forward; a pre-registered addendum) and R' = 0.75 +- 0.16 / 0.17 +- 0.35: §8.1 R' table. From the table means R' = 0.735 / 0.172 (fig_rescomm.py _verify()); the quoted values are seed averages. Three times the information about the class label reaching the receiver on permuted windows (0.185 vs 0.061 bits; the wiki calls it "label information") and +2.4 +- 0.9 points: §8.1 bullets. Dephasing costs 0.45 / 0.23 nats (contiguous / permuted), entangled in 96-97 % of test windows: §8.1 "Why the quantum channel helps". Many-shot caveat and 0.838: §8.1 "Many copies change the comparison". Ladder of messages: §12.3 (proposed, not run); why a full reduced state reproduces the quantum channel for this readout: App. E. Deferred measurement (why coherent use is needed): App. D, slide 26. -->
<!-- EDIT-FORWARD: cut for space, say aloud: R' compares measure-once with the sender NOT using its outcome against no communication without feed-forward (A: test accuracy). Contiguous: QC - CC = +0.8 +- 0.6 points, within noise. Permuted: +2.4 +- 0.9 points and 0.185 vs 0.061 bits reaching the receiver; R' = 0.17 +- 0.35 is uncertain with five seeds. Ladder levels: one bit, <Z>, the Bloch vector, the full reduced state (15 numbers), proposed not run (§12.3). -->
<!-- EDIT-FORWARD: the originally pre-registered ratio (sender also uses its outcome) is 0.07 +- 0.33 (contiguous) and -0.31 +- 0.64 (permuted); letting the sender use its outcome costs 1.7 +- 0.7 points on contiguous windows (§8.1). R' was added before the runs as a pre-registered addendum. Say this if asked why R' and not R. -->
<!-- EDIT-FORWARD: five seeds only. On permuted windows the denominator of R is only 2.1 standard errors from zero (errors 0.3-1.8 on the ratios); on contiguous windows the quantum channel beats measuring once by +0.8 +- 0.6 points, not distinguishable from zero (§8.1 caveats). -->
<!-- EDIT-FORWARD: dephasing sensitivity alone is not evidence of a quantum advantage: QPUs that keep their own qubits unmeasured are just as sensitive (§8.1 caveat); the retrained comparison (quantum channel against measuring once) is the evidence. The test-time analyses were added after the runs and are exploratory. -->
