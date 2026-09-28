---
marp: true
theme: serif
math: mathjax
---

<style scoped>table { font-size: 0.9em; margin: 0.3em 0 0.5em; } th, td { padding: 0.25em 0.9em; white-space: nowrap; } tbody tr:last-child td { color: var(--muted); border-bottom: none; font-size: 0.85em; } li { margin-bottom: 0.2em; }</style>

# <span class="cat results">Results</span> Accuracy: just below the $0.9$ target

<!-- src: dqml-physics-results.md §3 Stage 1 table (phys-phase2g README §1; 3 QPUs x 4 qubits, revised encoding = phase on |b(m)>, contiguous windows, pre-registered evaluation on seeds 3-7, which were not used for any design choice): no communication 0.8783 (sd 0.007), 1805/2055, 0/5 seeds above 0.9, gap 0.008; two-input links (CC, 2 senders -> 1 receiver) 0.8895 (sd 0.008) = 1828/2055 correct with 1850 needed, 0/5 seeds above 0.9, gap 0.035. Target (§3): mean test accuracy > 0.9 over five unused seeds and train-test gap <= 0.03. "0.890" = 1828/2055 = 0.88954 rounded. §0.3: mean ± sd over seeds, best-validation epoch, 411 test signals per seed. -->
<!-- src: "random bits of equal mean remove about half of the gap": §3 "Origin of the gap" (input-independent random bits with the same mean remove about 50 % of the gap at 4 qubits; the mechanism, overfitting of the decision functions to the 1325 training signals, is flagged there as not tested). -->
<!-- src: classical row: §0.1 (best classical classifier on all 40 features 0.978, cross-entropy 0.060; tuned RBF support-vector machine 0.964; four-class task 0, 1, 3, 6). Named 2026-09-29 from the updated §0.1: the 0.978 model is a kernel logistic regression with a translation-aware kernel (sum of RBF kernels over the 40 cyclic width-5 sub-windows, C = 300, g = 2), chosen on validation data (minimum validation cross-entropy) among 2330 candidate models; deck-wide short label "kernel logistic regression (shift-aware)", introduced in full on 34-our-task.md. A width-10 kernel reaches 0.985 but was not selected (§0.1). -->
<!-- src: seeds 0-2 line: App. G (the revised encoding was chosen on seeds 0-2); §2.4 (seeds 0-2: NC 0.865 ± 0.014, CC 0.908 ± 0.002). -->
<!-- terminology pass 2026-09-28 (GLOSSARY.md): "two-input links" = CC, 2 senders -> 1 receiver (the default topology); "held-out seeds" = seeds 3-7 not used for model selection; "target" = the pre-registered acceptance criterion of §3; "fixed test" = fixed coefficients. -->
<!-- Restructured 2026-09-29 (speaker: "why did you show seed 0-2, 0-4, and 3-7 separately?"): the three columns came from three different studies of the same configuration (§2.4 encoding comparison on seeds 0-2, §4.3 topology comparison on seeds 0-4, §3 pre-registered check on seeds 3-7). Only seeds 3-7 were never used to choose anything, and per-seed values are not in the vault, so the pooled seeds 0-7 means of §4.2 (NC 0.873, CC 0.896; exactly the 3:5 weighted means of §2.4 and §3) carry no s.d. and mix in the design seeds. The table now shows the held-out numbers only. -->

Four digits ($0,1,3,6$), $411$ test signals, three QPUs of $n=4$ qubits; mean $\pm$ s.d. over seeds $3$–$7$ (not used for model selection).

| model | test accuracy | generalization gap |
|---|---|---|
| no communication (NC) | $0.878\pm0.007$ | $0.008$ |
| classical communication (CC), 2 senders $\to$ 1 receiver | $0.890\pm0.008$ | $0.035$ |
| **pre-registered target** | $>0.9$ | $\le0.03$ |
| classical, all $40$ features:<br>tuned RBF SVM / kernel logistic regression (shift-aware) | $0.964$ / $0.978$ | |

- **Target not met:** $1828$ of $2055$ test predictions correct, $1850$ needed.
- **The gap comes with communication:** random message bits remove about half of it.
- Seeds $0$–$2$ were used to choose the encoding (CC there: $0.908$).

<!-- Speaker note: "random message bits" = every message bit replaced by an input-independent random bit with the same mean (§3 "Origin of the gap"). Say aloud that 0.890 is 22 test predictions short over five seeds (1850 - 1828), and that no single seed exceeds 0.9. The generalization gap is train minus test accuracy; without communication it is below 0.01. -->
<!-- Speaker note (cut from the slide 2026-09-29, quote if asked): the best fixed coefficients (OR-type (1,1,-1), d = -0.75, K = 10^3 shots) reach 0.917 on two seeds, equal to trained coefficients on those seeds (0.914) (§5.2 point 4). The wiki does not identify those two seeds as held-out (Figs. 7/8 of the same Phase 3B use seed 0), so do not compare 0.917 with the 0.9 target. -->
<!-- Speaker note (if asked about the design seeds): on seeds 0-2 CC scores 0.908 ± 0.002, 1.85 points above the held-out 0.8895, while NC scores 0.865 ± 0.014, i.e. 1.3 points BELOW its held-out 0.878 (differences of the §2.4 and §3 means). App. G's "seeds 0-2 beat seeds 3-7 by 2.3 ± 0.4 points with 4 qubits (A100)" refers to runs the wiki does not identify and does not follow from these means; do not quote it. The earlier seeds 0-4 column (§4.3: NC 0.872 ± 0.013, CC 0.903 ± 0.007) belongs to the topology comparison. -->
<!-- Layout 2026-09-29 (model named in the classical row): the row now wraps after "classical, all 40 features:" and is set at 0.85em like a muted reference row; th/td padding 0.3em -> 0.25em and nowrap, so that the table and the three bullets still fit above the footer (checked in a private build). -->
