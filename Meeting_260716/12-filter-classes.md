---
marp: true
theme: serif
math: mathjax
---

<style scoped>
section { font-size: 22.5px; }
</style>

# <span class="cat method">Design</span> Two sensors: product versus collective filters

With two sensors, the single-sensor recipe has two natural extensions, and they are not equal.

- **Product filter** $K_1 \otimes K_2$: apply the single-sensor optimal filter to each sensor independently. The conjugator $V = V_1 \otimes V_2$ is local, so each sensor is asked its own question and the answers are combined classically.
- **Collective filter** $K_{\mathrm{coll}} = D_\gamma\, V$ with an entangling $V$: the success branch of the joint circuit reduces to one effective Kraus operator that does **not** factorize, $K_{\mathrm{coll}} \neq K_1 \otimes K_2$. The two sensors are asked a single joint question.

Both classes are optimized on equal footing: at every initial-state angle $\theta$, a structure-free numerical search over the conjugator, both post-selection strengths $\gamma_1, \gamma_2$, and the sensing time $t_s$ (up to 18 parameters, multi-seed annealing plus polish).

**Preview of the result.** The product class stays *additive*: its conditional QFI is the sum of the two single-sensor contributions. The collective class reaches about $4\times$ the single-sensor optimum, and it does so for separable initial states too.
