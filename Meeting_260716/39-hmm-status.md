---
marp: true
theme: serif
math: mathjax
---

<style scoped>
section { font-size: 22px; }
</style>

# <span class="cat ongoing">Update</span> Since the journal club: the HMM enters DQML

The outlook slide asked for a quantum-native version of this program. Two of its three levels now exist (July 15):

- **Level 0, the dataset bridge (built).** The HMM generator now emits DQML's exact train/validation/test format, with labels binarized at the teacher median for the cross-entropy classifier. The intrinsic-dimension ratio $\delta = D/N$ becomes the data-structure knob, and the correlation the HMM injects across QPUs is exactly its input covariance, $\operatorname{Cov}(x_i, x_j) = b^2 (F F^\top / D)_{ij}$, so inter-QPU correlation is now analytic and designable.
- **Level 1, frozen circuit plus trained head (derived and verified).** Training only the readout head on a frozen circuit is a kernel / GLM problem on the quantum feature map: each Pauli feature decomposes exactly as $g_P(z) = 2^{-n} \sum_b \chi_P(b)\, e^{i \Delta_b(z)}$ with $\Delta_b$ quadratic in the inputs, so the feature means, the kernel, and the teacher alignment all have closed forms through one Gaussian integral. Verified against Monte Carlo; the feature identity holds to $6 \times 10^{-16}$.
- **Level 2, the trained deep circuit, stays open.** That is where the equations-of-motion question of the paper returns for the quantum model.

The practical consequence: the parameter sweeps can now target regimes the closed forms point at, instead of sweeping blind.
