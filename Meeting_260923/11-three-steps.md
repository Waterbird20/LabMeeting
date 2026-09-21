---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

# <span class="cat intro">Intro</span> Everything we do is three steps

<style scoped>
.steps { display: flex; align-items: stretch; gap: 0.55rem; margin: 0.6em 0 0.5em; }
.steps .arrow { align-self: center; color: var(--primary); font-size: 1.25em; }
.steps .step { flex: 1 1 0; min-width: 0; border: 1px solid var(--rule); border-radius: 5px; padding: 0.45em 0.8em; }
.steps .step.one { border-top: 3px solid #1f6feb; }
.steps .step.two { border-top: 3px solid #b91c1c; }
.steps .step.three { border-top: 3px solid #15803d; }
.steps .n { font-size: 0.58em; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); }
.steps .t { font-size: 1.0em; font-weight: 600; color: var(--ink); line-height: 1.35; }
.steps .d { font-size: 0.66em; line-height: 1.42; }
</style>

Quantum information is an enormous field, but what we actually run in the lab generally has the same three steps.

<div class="steps">
<div class="step one">
<div class="n">Step one</div>
<div class="t">Prepare a state</div>
<div class="d">The system is initialised into one known starting state.</div>
</div>
<div class="arrow">&rarr;</div>
<div class="step two">
<div class="n">Step two</div>
<div class="t">Apply a unitary</div>
<div class="d">A quantum operation, which is to say an algorithm, evolves the state the way we want.</div>
</div>
<div class="arrow">&rarr;</div>
<div class="step three">
<div class="n">Step three</div>
<div class="t">Measure</div>
<div class="d">One number comes back, and every conclusion has to be extracted from it.</div>
</div>
</div>

$$
\ket{\psi_0}\;\xrightarrow{\ \ U\ \ }\;U\ket{\psi_0},
\qquad\qquad
\text{what we read out}\;=\;\braket{\psi_0|U^{\dagger}OU|\psi_0}.
$$

Each box has a long story behind it, and each is a field of its own. On the theory side the main interest sits in the unitary $U$: it is the part we choose, and we design it on purpose so that the number produced by the third box answers the question we asked.
