---
marp: true
theme: serif
math: mathjax
---

<style scoped>table { font-size: 0.9em; margin: 0.3em 0 0.5em; } th, td { padding: 0.25em 0.9em; white-space: nowrap; } tbody tr:last-child td { color: var(--muted); border-bottom: none; font-size: 0.85em; } li { margin-bottom: 0.2em; }</style>

# <span class="cat results">Results</span> Accuracy: just below the $0.9$ target

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
