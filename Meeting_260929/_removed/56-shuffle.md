---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat results">Results</span> Permuted features: more synergy, but the message bits carry little of it

<style scoped>table { font-size: 0.9em; margin: 0.3em auto 0.7em; } td, th { padding: 0.25em 0.9em; } li { margin-bottom: 0.45em; }</style>

| 4 digits, $n=4$ | contiguous blocks | permuted features |
|---|---|---|
| best additive classical model | $0.971$ | $0.891$ |
| non-additivity $D_\mathrm{add}$ (synergy proxy) | $0.021$ nats | $0.224$ nats |
| our model, no communication (NC) | $0.875$ | $0.800$ |
| communication gain, accuracy | $1.5$–$3.1$ points | $5.7$–$7.5$ points |

- **Expected, the drop:** additive model $-8.0$ points, ours (NC) $-7.5$.
- **Surprising:** communication gains more accuracy here, yet the decision functions rarely become XOR.
- **Our reading:** the message bits partly regularise an overfitting NC model (generalization gap $0.075$).

<!-- Speaker note: permuted features = one fixed random permutation of the 40 features before partitioning them into feature blocks (the feature-permutation test of the data section). A block no longer holds neighbouring features, and the Fourier (DFT) encoding loses its motivation, translation symmetry. For comparison only: the ten-class CNN of the data section lost 27.7 points under the same kind of permutation (different task: ten classes, all 40 features). NC generalization gap: 0.016 contiguous vs 0.075 permuted; NC sd 0.016 in both columns. -->
<!-- src: permutation definition and D_add = CE(best additive) - CE(best joint), 0.021 / 0.224 nats; classical additive model 0.971 / 0.891: dqml-physics-results.md §0.1. No communication 0.875 +- 0.016 / 0.800 +- 0.016: §8.1 table, first row (fresh |00>, sender feed-forward; the channel experiment's no-communication model, seeds 0-4). Gap 0.016 (contiguous): §4.3 table, "none" row; gap 0.075 (permuted) and "every pattern gains 5.7-7.5 points and even reduces the gap": §4.3 last bullet. Contiguous accuracy gain 1.5-3.1 points and cross-entropy gain 0.092-0.192 nats: §4.3 table (differences of means, fig_rescomm.py _verify()). Permuted cross-entropy gain 0.05-0.08 nats against D_add 0.224: §4.2 ("The gain is not synergy between windows"). -->
<!-- src: additive classical loss 0.971 - 0.891 = 8.0 points (§0.1) and ours 0.875 - 0.800 = 7.5 points (§8.1), checked in fig_rescomm.py _verify(). CNN 0.855 -> 0.578, -27.7 points, ten classes: mnist1d-eda.md §4.6 (slide 33); now only in the speaker note because it is a ten-class number. Revised vs original encoding on permuted windows, -2.4 +- 2.5 / +2.9 +- 2.8 points, and "after a permutation the DFT modes are not physical frequencies" [interpretation]: §2.4, "Permuted windows barely change" (this compares the two encodings, not communication). Translation-symmetry motivation: dqml-quantum-design.md §2.3.1. XOR: §5.1 (XOR rare: 1 of 62 input-dependent links over the one-test-per-pair patterns, 4 of 386 in Phase 2e, 0 of 34 four-qubit two-input links; the plan's predicted XOR on synergistic window pairs was not observed) and Open questions. -->
<!-- EDIT-FORWARD: the no-communication row comes from the channel experiment (§8.1, extra idle qubits in the final layer; controls within +-0.6 points of the original model on contiguous windows). §4.3 gives the permuted pattern gains only as a range (5.7-7.5 points) and no absolute permuted accuracies; the wiki has no permuted per-pattern table. -->
<!-- EDIT-FORWARD: "Our reading: the bits partly regularise" is an interpretation, not a tested mechanism. It rests on the no-communication gap 0.075 and on the small cross-entropy gain (§4.2, §4.3). The wiki's Open questions say "with permuted windows communication helps little", which is true in nats but not in accuracy (5.7-7.5 points); be ready for that question. The provenance of the 0.05-0.08 nats range (which encoding, pattern and qubit number) is not stated on the page: the same §4.2 bullet quotes 0.14-0.25 nats for contiguous windows, which is not the §4.3 range and appears to pool the original encoding and possibly the 6-qubit runs of the §4.2 table. Resolved 2026-09-29: the cross-entropy row and the "lowers the cross-entropy far less than D_add" clause were removed from the visible slide (both ranges are now in the speaker note); confirm the n of 0.05-0.08 nats before quoting it. The 5.7-7.5 points (§4.3, revised encoding, n = 4) and the 0.05-0.08 nats (§4.2) may come from different runs, so the "more accuracy, less information" contrast is our synthesis, not a comparison the page makes. -->
<!-- EDIT-FORWARD: cut for space: revised against original encoding, permuted accuracy moves only -2.4 +- 2.5 points without and +2.9 +- 2.8 with communication, best 0.844 (§2.4; this compares the two encodings, not communication). Supports "the Fourier embedding loses its motivation" if asked. -->
<!-- Resolved 2026-09-29: the visible CNN comparison (ten classes next to four-class numbers) was moved to the speaker note. -->
<!-- Speaker note: if asked why "carry little": at n = 4 (§8.1 channel table, permuted features, seeds 0-4) the 10^3-shot threshold bit lowers the cross-entropy from 0.502 (no communication) to 0.449 nats, i.e. 0.053 nats, about a quarter of D_add = 0.224; measuring once and sending the outcome does not lower it at all (0.508 / 0.510). On contiguous blocks communication lowers it by 0.09-0.19 nats (n = 4, §4.3), more than D_add = 0.021. The cross-entropy row was removed from the slide (2026-09-29): §4.2's permuted range 0.05-0.08 nats states no qubit number or encoding (possibly pooled with 6-qubit runs), so do not quote it until confirmed. -->
<!-- src: "carry little of it" (title): dqml-physics-results.md §4.2 ("With permuted windows ... communication gains only 0.05-0.08 nats"), §5.1 / Open questions ("with permuted windows communication helps little"), §8.1 ("Permuted windows: classical bits recover little", R' = 0.17 +- 0.35) and the §8.1 table differences 0.502 - 0.449 = 0.053 nats (threshold bit) and 0.508 / 0.510 >= 0.502 (measure once), n = 4, revised encoding, seeds 0-4 (my arithmetic on the table values). -->
