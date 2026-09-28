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
