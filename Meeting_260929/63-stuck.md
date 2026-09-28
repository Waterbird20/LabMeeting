---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

<style scoped>ul { margin-top: 0.6em; } li { margin: 0.55em 0; } p { margin: 0.7em 0 0; }</style>

# <span class="cat ongoing">Ongoing</span> Doubt 3: do decision functions keep their initial Boolean function?

- **Observed.** Seed $0$: $4$ of $6$ keep their initial function. Seeds $0$–$7$: $14$ of $48$ constant, $20$ depend on one input.
- **Possible cause.** Only a boundary that intersects the data gets a gradient: training succeeds only for $d$ from $-0.625$ to $-0.25$ (XOR-type) or $-0.875$ to $-0.375$ (OR-type).
- **Tried.** Re-initializing $d$ at the data median (phase on $\lvert m{+}1\rangle$): $94$–$100\,\%$ become input-dependent; accuracy within noise.

**So trainability, not usefulness, may choose the Boolean function.**

<!-- Speaker note: in seed 0 the coefficients move during the first 25 to 50 epochs and then settle; one decision function becomes NOR after one epoch and ends constant. "Depend on one input only" = dictators or their negations. The bit can depend on the input for every d in (-1, 0) (sweep of the bias d with (a, b, c) fixed, K = 10^3), but training works only in the narrower ranges on the slide. The reset test ran with the phase on |m+1> (the original encoding): accuracy stayed within +-1.2 standard errors, so constant decision functions are a symptom, not the cause of the limited gain. Only seed 0 was logged every epoch; the initial distributions of (m_i, m_j) were not measured. Not yet checked: other seeds' trajectories, the initial distribution of (m_i, m_j). The trainable ranges of d are at K = 10^3. -->

<!-- terminology pass 2026-09-28 (GLOSSARY.md): "links stay where they started" = decision functions stay near their initial Boolean functions; "links" = decision functions; "depend on one QPU only" = dictators or their negations (as on 58), shown on the slide as "depend on one input only"; "Not yet tested" = "Not yet checked" (verifier 2026-09-28); "threshold that does not cut the data" = decision boundary outside the support of the data; "fixed-coefficient scan" = sweep of the bias d with (a,b,c) fixed; "logic a link ends with" = Boolean function it converges to; "resetting thresholds" = re-initializing the bias d; "original encoding" = phase on |m+1> (GLOSSARY l.66; was "earlier encoding" on the slide, changed 2026-09-29 per verifier); "re-centring the threshold" = re-initializing the bias d at the data median. -->
<!-- src: seed 0 (revised encoding, two-input links, contiguous windows, n = 4): "Four of the six links keep their initial Boolean function. One switches between NOT m_j, NAND and NOT m_i during epochs 6-29 and ends as NOT m_i. One becomes NOR after a single epoch ... so that link becomes constant": dqml-physics-results.md §5.0 (Fig. 4). Link census 3 x 4 qubits, 48 links, seeds 0-7: constant 14, one variable 20, implication type 9, AND/NAND/NOR 5, XOR 0: §5.0 table. -->
<!-- src: gradient only through |g| <~ sigma_g; "A threshold that does not intersect the data never receives a gradient"; trainable ranges XOR -0.625 to -0.25, OR -0.875 to -0.375 at K = 10^3; bit input-dependent only for d in (-1, 0): §5.2 items 1-3 and "How to read Fig. 8". -->
<!-- src: threshold reset to the data median (and an entropy penalty), Phase 2e: "94-100 % of links become input-dependent again, but test accuracy stays within +-1.2 standard errors: constant links are a symptom, not the cause": §11 table. Initial distributions not measured: §5.2 item 3 and Open questions. "Accuracy within noise" on the slide = test accuracy within +-1.2 standard errors. "94-100 %" is of links (decision functions), as §11 states. -->
<!-- EDIT-FORWARD: check the other seeds' trajectories. Only seed 0 was retrained with (a,b,c,d) logged every epoch (§5.0); the claim "they stay near their initial values whatever those were" rests on one seed plus the d scan. Also the planned test in §13 ("Trainability at initialisation"). -->
<!-- src (review 2026-09-28): Phase 2e ran at N = 3, n = 4 (dqml-physics-plan.md, decision 2026-09-26; log.md 2026-09-26 "Phase 2e at N = 3, n = 4 ... threshold re-centring"), before the revised (paired) encoding was introduced in Phase 2f, so the reset test is n = 4 with the original encoding. "The coefficients move substantially during the first 25-50 epochs and then settle": §5.0. -->
