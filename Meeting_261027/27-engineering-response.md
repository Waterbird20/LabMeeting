---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
.columns .col:first-child { flex: 0 0 640px; }
mjx-container[display="true"] { margin: 0.3em 0 !important; }
figure.figure { margin: 0; }
</style>

# <span class="cat strategy">Strategy</span> Engineering the response

The idea of QSP is engineering the response $U_{mc_1c_2\cdots}=e^{-i\tilde H_{mc_1c_2\cdots}t}$ at every detuning the environment produces.

<div class="columns">
<div class="col">

<figure class="figure">

<video src="media/engineering-response.mp4" poster="media/engineering-response.png" width="630" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
<div class="col">

$$
\{\Delta\}\ \longrightarrow\ \{\Delta\tau\}\ \longrightarrow\ \{x=\cos\tfrac{\Delta\tau}{2}\}
$$

The electron sees a mixture over the configurations:

$$
\rho\ \longmapsto\ \sum_{m,c_1,c_2,\dots}p_{mc_1c_2\cdots}\;U_{mc_1c_2\cdots}\,\rho\,U_{mc_1c_2\cdots}^\dagger
$$

**Next steps:**

1. **Set the target function**, on purpose: what $P$ and $Q$ must be at each detuning.
2. **Find $\{\phi_k\}_{K+1}$**: the proper $K$, the proper $\tau$, and how to fit.

</div>
</div>
