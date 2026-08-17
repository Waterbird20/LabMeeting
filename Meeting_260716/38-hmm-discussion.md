---
marp: true
theme: serif
math: mathjax
---

<style scoped>
section { font-size: 23px; }
</style>

# <span class="cat strategy">Discussion</span> Critique

Elegant and exact, but obtained under strong simplifications:

- **A deliberately simple model.** A single hidden layer and plain SGD, chosen so the equations of motion close. 
- **Online, not batch.** One-pass SGD with a fresh sample per step is atypical of modern (multi-epoch, mini-batch) training. Reusing samples might break the GEP.
- **A non-general task.** The target is a continuous, unbounded real number (MSE regression on a sum-of-sigmoids teacher with a linear, unsquashed readout), not the bounded or discrete classification that dominates real ML. (The two-colour manifold picture is for illustration.)
- **Not always a reduction.** The order parameters grow as $Q$ ($K\times K$), $R$ ($K\times M$), $T$ ($M\times M$), plus the spectral densities $r(\rho),\sigma(\rho)$. For a large student that is as many quantities as the model has parameters, so the "few macroscopic variables" framing is misleading. 

---

# <span class="cat ongoing">Outlook</span> Toward a quantum version

For the DQML project, the HMM is a starting point:

- We would need a complex-valued version of the dataset, with structure native to quantum states and circuits rather than real images.
- We would need to derive our own equations of motion for the quantum model. Our current setup (a CNN with classical communication) is heavy to simulate and not sure to have closed dynamics.
- The payoff is an analytical tool to argue where quantumness helps, which addresses the question that **"Better than classical?"** poses.

<div class="callout">

To demonstrate a real quantum advantage we need both data and theory that isolate the quantum resource. The HMM shows what solvable structured data can look like.

</div>

---

# Summary

- The **hidden manifold model** generates structured data: a low-dimensional latent variable, folded into high dimensions, labeled by a teacher.
- The **Gaussian Equivalence Property** reduces two-layer online-SGD learning to **closed ODEs** for a few order parameters, giving the exact learning curve.
- Results: smaller $\delta$ (more structure) helps; units specialize; wider students reach higher complexity; the curve is reproduced to $\approx 10^{-3}$.
- For us: a concrete example of "physics for AI", and a target to adapt to quantum machine learning.
