---

title       : Post-Selected Quantum Metrology
author      : Donghun Jung
marp        : true
paginate    : true
theme       : KIST
math        : mathjax


header-includes:
- \usepackage{braket}
output:
  pdf_document:
    keep_tex: true
style: @import url('https://unpkg.com/tailwindcss@^2/dist/utilities.min.css');

---

<!-- _class: titlepage -->


<style>
.container{
   display: flex;
   align-items: center;
   width: 100%;
   height: 100%;
}
.col-left {
   flex: 0 0 65%;
   padding-right: 2rem;
   color: #666;
}
.col-left .title{
   color: #00356B;
   font-size: 32pt;
}

.col-right{
   flex: 0 0 30%;
   display: flex;
   align-items: center;
   justify-content: center;
}

/* === Category pills (used in every slide H1) === */
.cat {
   display: inline-block;
   font-size: 0.50em;
   font-weight: 600;
   padding: 0.20em 0.75em;
   border-radius: 0.85em;
   color: #ffffff;
   vertical-align: middle;
   margin-right: 0.55em;
   letter-spacing: 0.04em;
   text-transform: uppercase;
   font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
   line-height: 1.2;
}
.cat-intro    { background: #2563eb; }   /* blue   */
.cat-method   { background: #475569; }   /* slate  */
.cat-strategy { background: #7c3aed; }   /* violet */
.cat-results  { background: #059669; }   /* green  */
.cat-ongoing  { background: #d97706; }   /* amber  */

/* === Status chips for Required-Features list === */
.status {
   display: inline-block;
   font-size: 0.78em;
   font-weight: 600;
   padding: 0.10em 0.55em;
   border-radius: 0.4em;
   color: #ffffff;
   margin-right: 0.6em;
   min-width: 4.5em;
   text-align: center;
   letter-spacing: 0.03em;
   font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
}
.status-done       { background: #059669; }   /* green  */
.status-progress   { background: #d97706; }   /* amber  */
.status-tentative  { background: #6b7280; }   /* gray   */
.status-todo       { background: #94a3b8; }   /* light  */

ul.req-list { list-style: none; padding-left: 0; }
ul.req-list li { margin: 0.35em 0; }

/* === Outline legend === */
.legend { font-size: 0.85rem; margin: 0.4rem 0 1.0rem; }
.legend .cat { font-size: 0.65em; margin-right: 0.4em; }
.legend-row { margin: 0.15em 0; }
</style>

<div class="container">

<div class="col-left">

<div class="title">
Post-Selected Quantum Metrology
</div>

<div class="author">
Donghun Jung
</div>

<div class="date">
2026 May 14
</div>

<div class="organization">
Paulee Group, Center for Quantum Technology, Korea Institute of Science and Technology
</div>

</div>

<div class="col-right">
<img src="../../../media/images/PauleeLogo.png" style="max-width: 100%; height: 100%; object-fit: contain;">
</div>

</div>

---

<!-- backgroundColor: white -->

# Outline

<style scoped>
.outline-wrap { padding: 0 1.5rem; }
.legend-block { margin: 0.3rem 0 1.0rem; font-size: 0.85rem; color: #444; }
.legend-block .legend-row { margin: 0.18em 0; }
.projects { font-size: 0.95rem; }
.projects li { margin: 0.25em 0; }
.projects li ul li { font-size: 0.82rem; color: #555; }
</style>

<div class="outline-wrap">

<div class="legend-block">

<div class="legend-row"><span class="cat cat-intro">Intro</span> <span class="cat cat-method">Method</span> <span class="cat cat-strategy">Strategy</span> <span class="cat cat-results">Results</span> <span class="cat cat-ongoing">Ongoing</span> </div>

</div>

<div class="projects">

1. **Post-Selection** — analytical bound on the post-selected QFI and its saturating probes
2. **DDrf** — hybrid driving, side-peak suppression, Gaussian pulse shaping
3. **Mattermost–Outline** — lab tooling update

</div>

</div>

---

# <span class="cat cat-intro">Intro</span> Post-Selection: Sensing

<style scoped>
.figcard { text-align: center; margin: 0.4rem auto 0; }
.figcard img {
  max-width: 720px;
  max-height: 220px;
  border: 1px solid #d0d4da;
  border-radius: 6px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
  background: #fff;
}
</style>

Regardless of the quantity being sensed, quantum sensing requires an interaction between the system and the target. In our NV center system, we sense an external magnetic field, captured by the Hamiltonian $\mathcal{H}_B = \gamma B S_z$, where $\gamma$ is the gyromagnetic ratio. Through this interaction term and the time evolution $e^{-i\mathcal{H}_B t}$, information about the $B$-field is imprinted onto the state: $\ket{\psi} \rightarrow e^{-i\mathcal{H}_B t}\ket{\psi}$.

<div class="figcard">

![](./Meeting_260514/src/Presentation/media/sensing-diagram.png)

</div>

---

# <span class="cat cat-intro">Intro</span> Post-Selection: Quantum Fisher Information

The quantum Fisher information (QFI) quantifies how rapidly a quantum state — represented by a density matrix — changes with respect to the $B$-field. The faster the change, the greater the sensitivity:
$$
F_Q \;=\; 2\sum_{k,l} \frac{(\lambda_k - \lambda_l)^2}{\lambda_k + \lambda_l}\,\left|\bra{k}\mathcal{H}_{B}\ket{l}\right|^2,
$$
where $\rho = \sum_k \lambda_k \ket{k}\bra{k}$.

---

# <span class="cat cat-intro">Intro</span> Post-Selection: Hypothesis

<style scoped>
.figcard { text-align: center; margin: 0.1rem auto 0; }
.figcard img {
  max-width: 1560px;
  max-height: 280px;
  border: 1px solid #d0d4da;
  border-radius: 6px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
  background: #fff;
}
</style>

We hypothesize that adding a post-selection step can be advantageous. Intuitively, the $B$-field information is encoded in the phase, and post-selection lets us discard the unnecessary part of the state into an ancillary qubit (or the other energy level). For a probe coupled to an ancilla,
$$
\rho \;=\; p_0\,\ket{0}\bra{0} \otimes \rho_0 \;+\; p_1\,\ket{1}\bra{1} \otimes \rho_1 \;\;\longrightarrow\;\;
\begin{cases}
\rho_0 & \text{if the ancilla is measured in $\ket{0}$, with probability $p_0$,}\\
\rho_1 & \text{if the ancilla is measured in $\ket{1}$, with probability $p_1$.}
\end{cases}
$$

<div class="figcard">

![](./Meeting_260514/src/Presentation/media/Post-selection-pipeline.png)

</div>

---

# <span class="cat cat-strategy">Strategy</span> Post-Selection: Analytical Approach

<style scoped>
.split { display: flex; gap: 1.4rem; align-items: center; margin-top: 0.6rem; }
.split .text { flex: 1 1 55%; }
.split .papercard { flex: 0 0 42%; text-align: center; }
.split .papercard img {
  max-width: 100%;
  max-height: 320px;
  border: 1px solid #cfd3d8;
  border-radius: 6px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.10);
  background: #fff;
}
.split .papercard .caption {
  font-size: 0.7rem;
  color: #666;
  margin-top: 0.3rem;
  font-style: italic;
}
</style>

<div class="split">
<div class="text">

Last time I ran numerical simulations to identify which probe state and post-selection filter achieve the maximum QFI.

GH then shared a ChatGPT-generated report that gave useful insight into how to compute the bound.

Building on that, I reorganized the proof and analyzed the solution with Claude Opus 4.7.

</div>
<div class="papercard">

![](./Meeting_260514/src/Presentation/media/GH-paper.png)

<div class="caption">Shared by GH — analytic two-qubit derivation.</div>

</div>
</div>

---

# <span class="cat cat-method">Method</span> Post-Selection: Setup

The state evolves through four stages:
$$
\ket\psi \;\underbrace{\rightarrow}_{\text{Time evo}}\; \ket{\psi_0} = e^{-i H_B \tau}\ket\psi \;\underbrace{\rightarrow}_{\text{Phase Damping}}\; \rho = \mathcal{D}(\rho_0) \;\underbrace{\rightarrow}_{\text{Post-selection}}\; \frac{K\rho K^{\dagger}}{\mathrm{Tr}\,K\rho K^{\dagger}}.
$$

Without loss of generality, write the pure probe state in the computational basis:
$$
\ket{\psi} \;=\; \sum_x a_x \ket{x}, \qquad \sum_x |a_x|^2 = 1,
$$
where $x$ ranges over computational-basis bitstrings like $\ket{00\cdots 0}$.

**Time evolution.** Under the detuned magnetic field,
$$
\ket{\psi_0} \;=\; e^{-i H_B \tau}\ket\psi \;=\; \sum_x a_x e^{i\theta |x|} \ket{x},
$$
where $|x|$ is the Hamming weight.

---

# <span class="cat cat-method">Method</span> Post-Selection: Phase-Damping Channel

The density matrix before damping is
$$
\rho_0 \;=\; \sum_{x,y} a_x \bar{a}_y\, e^{i\theta(|x| - |y|)} \ket{x}\bra{y}.
$$

After phase damping, off-diagonal elements decay by $\eta^{d(x,y)}$, where $d(x,y) = |x\oplus y|$ is the Hamming distance:
$$
\begin{aligned}
\rho_0 \;\rightarrow\; \rho
&= \sum_{x,y} a_x \bar{a}_y\, e^{i\theta(|x| - |y|)}\, \eta^{d(x,y)} \ket{x}\bra{y} \\
&= U(\theta)\, D_a\, C_N\, D_a^{\dagger}\, U^\dagger(\theta),
\end{aligned}
$$
with $D_a = \mathrm{diag}(a_x)$ and $C_N = (I + \eta X)^{\otimes N}$.

Phase damping kills coherences across different Hamming weights but preserves populations $|a_x|^2$, so $\rho$ has the eigenvalue structure of a *mixed* state on $\mathrm{supp}\,\rho$.

---

# <span class="cat cat-method">Method</span> Post-Selection: Filter Step

A $\theta$-independent Kraus operator $K$ (with $K^\dagger K \preceq I$) is applied to $\rho$, producing the success branch
$$
\sigma \;=\; \frac{K\rho K^\dagger}{p_s}, \qquad p_s = \mathrm{Tr}\bigl[K\rho K^\dagger\bigr].
$$

The conditional phase QFI
$$
F^{\mathrm{ps}}_{\theta,Q} \;=\; \mathrm{Tr}(\sigma L_\sigma^2)
$$
is the quantity we bound and optimize in the remaining slides.

---

# <span class="cat cat-strategy">Strategy</span> Post-Selection: Quick Summary

For an $N$-qubit system, the phase QFI of the post-selected state obeys
$$
F^{\mathrm{ps}}_{\theta,Q} \;\le\; \frac{N^2 \eta^2}{1-\eta^2}, \qquad \eta = e^{-\tau_d},\quad \tau_d = (\tau/T_2)^p.
$$

**Strategy.** Chain the relevant inequalities and analyze when each saturates. From the saturation conditions we can answer:

- Which probe states satisfy equality? What is the corresponding filter basis?
- Is there an entanglement-related gain?
- What is the post-selection success probability?
- Which probe state maximizes that success probability?


---

# <span class="cat cat-results">Results</span> Post-Selection: Quick Summary — Answers

For an $N$-qubit system, the phase QFI of the post-selected state obeys
$$
F^{\mathrm{ps}}_{\theta,Q} \;\le\; \frac{N^2 \eta^2}{1-\eta^2}, \qquad \eta = e^{-\tau_d},\quad \tau_d = (\tau/T_2)^p.
$$

- **Which probe states satisfy equality? What is the corresponding filter basis?**
  Any $\ket{\psi} = \sum_x a_x \ket{x}$ with $a_x \neq 0$ for all $x$ attains the maximum; the corresponding filter basis is derived below.
- **Is there an entanglement-related gain?**
  No — the saturating class is dense and includes product states, so QFI alone shows no entanglement-related gain.
- **What is the post-selection success probability, and which state maximizes it?**
  Among saturating probes, the state that maximizes the success probability is symmetric under both qubit exchange and bit flip.

---

# <span class="cat cat-method">Method</span> Post-Selection: SLD and the $M$ Operator

The post-selected density matrix is
$$
\sigma \;=\; \frac{K\rho K^\dagger}{p_s}, \qquad p_s = \mathrm{Tr}\,K\rho K^\dagger.
$$
The QFI is obtained from the symmetric logarithmic derivative (SLD):
$$
F_Q \;=\; \mathrm{Tr}\,\sigma L^2, \qquad \dot\sigma \;=\; \tfrac{1}{2}(\sigma L + L\sigma).
$$

Computing the SLD directly is generally hard, so we work with another operator instead.

---

# <span class="cat cat-method">Method</span> Post-Selection: Bounding via $M$

Introduce $M = \sigma^{-1/2}\dot\sigma\,\sigma^{-1/2}$. Substituting the SLD definition,
$$
M \;=\; \tfrac{1}{2}\bigl(\sigma^{1/2} L \sigma^{-1/2} + \sigma^{-1/2} L \sigma^{1/2}\bigr).
$$
The key inequality is $\boxed{\mathrm{Tr}\,\sigma L^2 \le \mathrm{Tr}\,\sigma M^2}$.

In the eigenbasis $\sigma = \sum_k p_k \ket{p_k}\bra{p_k}$ with $p_k \neq 0$,
$$
L_{ab} = \frac{2\dot\sigma_{ab}}{p_a + p_b}, \qquad M_{ab} = \frac{\dot\sigma_{ab}}{\sqrt{p_a p_b}}.
$$

$$
\mathrm{Tr}\,\sigma L^2 = \sum_{a,b} \frac{2|\dot\sigma_{ab}|^2}{p_a + p_b}, \qquad
\mathrm{Tr}\,\sigma M^2 = \sum_{a,b} \frac{|\dot\sigma_{ab}|^2 (p_a + p_b)}{2 p_a p_b}.
$$

Term by term, $m_{ab}/l_{ab} = (p_a+p_b)^2/(4 p_a p_b) \ge 1$, with **saturation when $p_a = p_b$**.

---

# <span class="cat cat-method">Method</span> Post-Selection: Lifting to $\rho$

Here, $\sigma$ lives in the subspace of $\rho$ selected by $K$. Accordingly, $M$ lives in that same subspace, and its operator norm is bounded by that of $M_\rho$:
$$
\mathrm{Tr}\,\sigma M^2 \;\le\; \|M_\rho\|_\infty^2.
$$

On the other hand,
$$
M_\rho \;=\; \rho^{-1/2}\dot\rho\,\rho^{-1/2}, \qquad
\dot\rho \;=\; -i[H_B, \rho] \;=\; \sum_i -\tfrac{i}{2}[Z_i, \rho].
$$

Writing $M_\rho = \sum_i M_i$ with $M_i = \rho^{-1/2}\bigl(-\tfrac{i}{2}[Z_i, \rho]\bigr)\rho^{-1/2}$,
$$
\|M_\rho\|_\infty \;\le\; \sum_i \|M_i\|_\infty.
$$

Here, $C_N = C^{\otimes N}$ and $[H_B, D_a] = 0$ force equality for any full-support probe.

---

# <span class="cat cat-method">Method</span> Post-Selection: Chained Inequality

Putting it together,
$$
F_Q \;=\; \mathrm{Tr}\,\sigma L^2 \;\le\; \mathrm{Tr}\,\sigma M_\sigma^2 \;\le\; \|M_\rho\|_\infty^2 \;\le\; \Bigl(\sum_i \|M_i\|_\infty\Bigr)^2.
$$
When every step saturates simultaneously,
$$
F_Q \;\le\; N^2\,\|M_i\|_\infty^2.
$$

**Three saturation conditions:**
1. $\sigma$ is maximally mixed on the filter subspace.
2. The optimal filter is chosen.
3. (Automatically satisfied here — see the previous slide.)

Next, we evaluate $\|M_i\|_\infty$.

---

# <span class="cat cat-method">Method</span> Post-Selection: GEP for $M$

To extract the eigenvalues and eigenvectors of $M$, it helps to recast the problem as a generalized eigenvalue problem:
$$
\begin{aligned}
M\ket{v} &= \lambda\ket{v} \\
\rho^{-1/2}\dot\rho_j\,\rho^{-1/2}\ket{v} &= \lambda\ket{v} \\
\dot\rho_j\,\bigl(\rho^{-1/2}\ket{v}\bigr) &= \lambda\,\rho\,\bigl(\rho^{-1/2}\ket{v}\bigr) \\
\dot\rho_j\ket{w} &= \lambda\,\rho\,\ket{w},
\end{aligned}
$$
with $\ket{w} = \rho^{-1/2}\ket{v}$. The last line is the GEP we solve.

---

# <span class="cat cat-method">Method</span> Post-Selection: Generalized Eigenvalue Problem

The standard eigenvalue problem is $A\mathbf{x} = \lambda\mathbf{x}$, with eigenvalues from $\det(A - \lambda I) = 0$.

The **generalized eigenvalue problem (GEP)** extends this to $A\mathbf{x} = \lambda B\mathbf{x}$. This should be familiar from, e.g., coupled mechanical systems: the Lagrangian gives
$$
M \ddot{\mathbf{x}} + K\mathbf{x} = 0.
$$
A harmonic ansatz $\mathbf{x} = \mathbf{x}_0 e^{i\omega t}$ then yields
$$
K\mathbf{x} \;=\; \omega^2 M \mathbf{x},
$$
a GEP whose generalized eigenvalues are the squared normal-mode frequencies.


---

# <span class="cat cat-method">Method</span> Post-Selection: Block Form of the GEP

The relevant block-form is
$$
\dot\rho_j \;=\; \begin{pmatrix} 0 & -i\eta B_j \\ i\eta B^\dagger_j & 0 \end{pmatrix},
$$
so the GEP reduces to
$$
\begin{pmatrix} 0 & -i\eta B_j \\ i\eta B^\dagger_j & 0 \end{pmatrix}
\begin{pmatrix} w_1 \\ w_2 \end{pmatrix}
\;=\; \lambda
\begin{pmatrix} A_j & \eta B_j \\ \eta B^\dagger_j & D_j \end{pmatrix}
\begin{pmatrix} w_1 \\ w_2 \end{pmatrix}.
$$

---

# <span class="cat cat-method">Method</span> Post-Selection: Schur Decomposition

Applying a Schur-style decomposition with $B_j = A_j^{1/2} W_j D_j^{1/2}$,
$$
\begin{aligned}
&\begin{pmatrix} A_j^{1/2} & 0 \\ 0 & D_j^{1/2} \end{pmatrix}
\begin{pmatrix} 0 & -i\eta W_j \\ i\eta W_j^\dagger & 0 \end{pmatrix}
\begin{pmatrix} A_j^{1/2} & 0 \\ 0 & D_j^{1/2} \end{pmatrix}
\begin{pmatrix} w_1 \\ w_2 \end{pmatrix} \\
&\quad = \lambda
\begin{pmatrix} A_j^{1/2} & 0 \\ 0 & D_j^{1/2} \end{pmatrix}
\begin{pmatrix} I & \eta W_j \\ \eta W_j^\dagger & I \end{pmatrix}
\begin{pmatrix} A_j^{1/2} & 0 \\ 0 & D_j^{1/2} \end{pmatrix}
\begin{pmatrix} w_1 \\ w_2 \end{pmatrix}.
\end{aligned}
$$

Stripping the outer factors,
$$
\begin{pmatrix} 0 & -i\eta W_j \\ i\eta W_j^\dagger & 0 \end{pmatrix}
\begin{pmatrix} \tilde{w}_1 \\ \tilde{w}_2 \end{pmatrix}
\;=\; \lambda
\begin{pmatrix} I & \eta W_j \\ \eta W_j^\dagger & I \end{pmatrix}
\begin{pmatrix} \tilde{w}_1 \\ \tilde{w}_2 \end{pmatrix}.
$$

---

# <span class="cat cat-method">Method</span> Post-Selection: Reducing via SVD

Take the SVD $W_j = U \Sigma_j V^\dagger$ with $\Sigma_j = \mathrm{diag}(s_{jk})$:
$$
\begin{pmatrix} 0 & -i\eta s_{jk} \\ i\eta s_{jk} & 0 \end{pmatrix}
\begin{pmatrix} \tilde{u}_{jk1} \\ \tilde{u}_{jk2} \end{pmatrix}
\;=\; \lambda_{jk}
\begin{pmatrix} 1 & \eta s_{jk} \\ \eta s_{jk} & 1 \end{pmatrix}
\begin{pmatrix} \tilde{u}_{jk1} \\ \tilde{u}_{jk2} \end{pmatrix}.
$$

The determinant condition
$$
\det \begin{pmatrix} -\lambda & -\eta s_{jk}(\lambda + i) \\ \eta s_{jk}(-\lambda + i) & -\lambda \end{pmatrix} = 0
$$
gives
$$
\lambda_{jk} \;=\; \pm\sqrt{\frac{\eta^2 s_{jk}^2}{1 - \eta^2 s_{jk}^2}}.
$$

---

# <span class="cat cat-results">Results</span> Post-Selection: Single-Qubit Bound

Since $0 \le s_{jk} \le 1$, the magnitude of $\lambda_{jk}$ is maximized at $s_{jk} = 1$, giving
$$
\|M_j\|_\infty \;\le\; \frac{\eta}{\sqrt{1 - \eta^2}}.
$$

This is one more inequality, which must hold for every qubit. Generically $s_{jk} < 1$ for mixed states; however, for any initial state $\ket{\psi_0} = \sum_x a_x \ket{x}$ with $a_x \neq 0$ for all $x$, phase damping still lets us attain $s_{jk} = 1$.

At joint saturation,
$$
\|M\|_\infty \;=\; \frac{N\eta}{\sqrt{1-\eta^2}}, \qquad
F_Q \;\le\; \|M\|_\infty^2 \;=\; \frac{N^2 \eta^2}{1-\eta^2}.
$$

---

# <span class="cat cat-results">Results</span> Post-Selection: Saturating Probe States

Every pure state $\ket{\psi_0} = \sum_x a_x \ket{x}$ with $a_x \neq 0$ for **all** computational-basis strings $x$ saturates the global bound
$$
F^{\mathrm{ps}}_{\theta,Q} \;=\; \frac{N^2 \eta^2}{1-\eta^2}.
$$

The saturating class is **open** (the condition $|a_x| > 0$ for all $x$ is open) and **dense** (full-support probes are generic) in the projective Hilbert space $\mathbb{CP}^{2^N - 1}$.

---

# <span class="cat cat-method">Method</span> Post-Selection: Back to the GEP

As shown above, the phase-damped state factorizes as $\rho = D_a C_N D_a^\dagger$ with $C_N = (I + \eta X)^{\otimes N}$. $D_a$ is invertible iff $a_x \neq 0$ for all $x$. Return to the GEP:
$$
\dot\rho_j\ket{w} \;=\; \lambda\,\rho\,\ket{w}.
$$
Then
$$
\begin{aligned}
-i[H_B, D_a C_N D_a^\dagger]\ket{w} &= \lambda\,D_a C_N D_a^\dagger\ket{w} \\
-i D_a[H_B, C_N] D_a^\dagger\ket{w} &= \lambda\,D_a C_N D_a^\dagger\ket{w},
\end{aligned}
$$
since $H_B$ and $D_a$ commute. Letting $\ket{\epsilon} = D_a^\dagger\ket{w}$, we obtain another GEP:
$$
-i[H_B, C_N]\ket{\epsilon} \;=\; \lambda\,C_N\ket{\epsilon}.
$$

---

# <span class="cat cat-method">Method</span> Post-Selection: Single-Qubit GEP

With $C_N = (I + \eta X)^{\otimes N} = C^{\otimes N}$ and $H_B = \tfrac{1}{2}\sum_i Z_i$,
$$
-\tfrac{i}{2}\sum_i [Z_i, C^{\otimes N}]\ket{\epsilon} \;=\; \lambda\,C^{\otimes N}\ket{\epsilon}.
$$
For the product ansatz $\ket{\epsilon} = \bigotimes_i \ket{\epsilon_i}$, each factor satisfies a single-qubit GEP:
$$
-\tfrac{i}{2}[Z, C]\ket{\epsilon_i} \;=\; \lambda\,C\ket{\epsilon_i}.
$$

Since $-\tfrac{i}{2}[Z, C] = \eta Y$, this reduces to
$$
\begin{pmatrix} -\lambda & -\eta(\lambda + i) \\ -\eta(\lambda - i) & -\lambda \end{pmatrix}\ket{\epsilon_i} = 0,
$$
yielding $\lambda_\pm = \pm\eta/\sqrt{1 - \eta^2}$ and $\ket{\epsilon_i^\pm} = \tfrac{1}{\sqrt 2}(\ket{0} + e^{\pm i\alpha}\ket{1})$, where $\cos\alpha = -\eta$, $\sin\alpha = \sqrt{1 - \eta^2}$.

For the $N$-qubit system, $\ket{\epsilon^\pm}^{\otimes N}$ has eigenvalue $\pm N\eta/\sqrt{1 - \eta^2}$.

---

# <span class="cat cat-results">Results</span> Post-Selection: Filter Vectors $\ket{w^\pm}$

Returning to $\ket{w}$:
$$
\ket{w^\pm} \;=\; (D_a^{-1})^{\dagger}\ket{\epsilon^\pm},
$$
so
$$
\dot\rho\,\ket{w^\pm} \;=\; \pm\,\frac{N\eta}{\sqrt{1-\eta^2}}\,\rho\,\ket{w^\pm}.
$$

Therefore $M = \rho^{-1/2}\dot\rho\,\rho^{-1/2}$ contains the eigenvalues $\pm\, N\eta/\sqrt{1-\eta^2}$, giving $\|M\|_\infty = N\eta/\sqrt{1-\eta^2}$ and
$$
F^{\mathrm{ps}}_{\theta,Q} \;=\; \frac{N^2 \eta^2}{1-\eta^2},
$$
with filter vectors $\ket{w^\pm}$.

---

# <span class="cat cat-results">Results</span> Post-Selection: Filter Basis $\ket{\Psi^\pm}$

Since $\ket{w} = \rho^{-1/2}\ket{v}$, the filter basis (denoted $\ket{\Psi^\pm}$) is
$$
\boxed{\qquad\ket{\Psi^\pm} \;=\; \ket{v^\pm} \;=\; \frac{\rho^{1/2}\ket{w^\pm}}{(1-\eta^2)^{N/2}}.\qquad}
$$

The two basis vectors are mutually orthogonal:
$$
\braket{\Psi^+ | \Psi^-} \;=\; \frac{\bra{w^+}\rho\ket{w^-}}{(1-\eta^2)^N} \;=\; 0,
$$
inherited from $\rho$-biorthogonality (the $C$-biorthogonality lifted to $N$ qubits).

---

# <span class="cat cat-method">Method</span> Post-Selection: Matched Kraus Operator

The filter is a rank-2 projector onto $\mathrm{span}\{\ket{\Psi^+}, \ket{\Psi^-}\}$:
$$
P_\pm \;=\; \ket{\Psi^+}\bra{\Psi^+} + \ket{\Psi^-}\bra{\Psi^-}.
$$
The matched Kraus operator is
$$
K \;=\; \sqrt{\frac{p_s}{2}}\,P_\pm\,\rho^{-1/2}.
$$
Applying it to $\rho$:
$$
K\rho K^\dagger \;=\; \frac{p_s}{2}\,P_\pm\,\rho^{-1/2}\rho\,\rho^{-1/2}\,P_\pm \;=\; \frac{p_s}{2}\,P_\pm^2 \;=\; \frac{p_s}{2}\,P_\pm.
$$

---

# <span class="cat cat-results">Results</span> Post-Selection: Accepted State

The accepted state is
$$
\sigma \;=\; \frac{K\rho K^\dagger}{p_s} \;=\; \frac{P_\pm}{2} \;=\; \frac{\ket{\Psi^+}\bra{\Psi^+} + \ket{\Psi^-}\bra{\Psi^-}}{2}.
$$
That is, the accepted state is the *maximally mixed state on the filter subspace* — exactly the structural requirement for QFI saturation.

This form looks unfamiliar, so we re-express the filter as
$$
K \;=\; \ket{\Psi^+}\bra{w^+} + \ket{\Psi^-}\bra{w^-}.
$$

It can be implemented as $K = \sqrt{1 - \gamma}\,\ket{00\cdots 0}\bra{00\cdots 0} + \ket{00\cdots 0}\bra{10\cdots 0}$ by sandwiching with appropriate unitaries before and after the post-selection step, **setting one post-selection strength to zero and the others to one**.

---

# <span class="cat cat-method">Method</span> Post-Selection: POVM Constraint

The POVM constraint is $K^\dagger K \preceq I$. Explicitly,
$$
K^\dagger K \;=\; \ket{w^+}\bra{w^+} + \ket{w^-}\bra{w^-} \;=\; (D_a^{-1})^\dagger\,\Pi\,D_a^{-1},
$$
where
$$
\Pi \;=\; \ket{\epsilon^+}^{\otimes N}\bra{\epsilon^+}^{\otimes N} + \ket{\epsilon^-}^{\otimes N}\bra{\epsilon^-}^{\otimes N}.
$$
Its component form is
$$
\Pi_{xy} \;=\; \frac{1}{2^{N-1}}\cos\bigl(\alpha(|x| - |y|)\bigr),
$$
so $(K^\dagger K)_{xy} = \Pi_{xy}/(\bar{a}_x a_y)$.

---

# <span class="cat cat-method">Method</span> Post-Selection: Success Probability — Setup

The condition $\Pi\ket{u} = \mu D_p\ket{u}$ is equivalent to $D_p^{-1/2}\Pi D_p^{-1/2}\ket{w} = \mu\ket{w}$ with $\ket{w} = D_p^{1/2}\ket{u}$, so we want the eigenvalues of the symmetric operator $D_p^{-1/2}\Pi D_p^{-1/2}$.

In the 2D image of $\Pi$ (basis $\{\ket{v^+}^{\otimes N}, \ket{v^-}^{\otimes N}\}$), $\Pi$ acts as the rank-2 identity.

---

# <span class="cat cat-results">Results</span> Post-Selection: Success Probability — Result

The Gram matrix of $\{D_p^{-1/2}\ket{v^\pm}^{\otimes N}\}$ is
$$
G \;=\; \begin{pmatrix} \hat S & \hat T \\ \overline{\hat T} & \hat S \end{pmatrix},
$$
and the eigenvalues of $D_p^{-1/2}\Pi D_p^{-1/2}$ in this basis are those of $G$, namely $\hat S \pm |\hat T|$.

So $\lambda_{\max}(\Xi) = \hat S + |\hat T| = (S + |T|)/2^N$, giving
$$
\boxed{\;p_s^{*} \;=\; \frac{2(1-c^2)^N}{\lambda_{\max}(\Xi)} \;=\; \frac{2^{N+1}(1-c^2)^N}{S + |T|}.\;}
$$

---

# <span class="cat cat-results">Results</span> Post-Selection: Optimal Probe — Symmetries

The optimization is over all probe magnitudes $\{|a_x|^2\}_x$ with $\sum_x |a_x|^2 = 1$. The objective $S + |T|$ depends on the magnitudes through
$$
S \;=\; \sum_x \frac{1}{|a_x|^2}, \qquad T \;=\; \sum_x \frac{e^{-2i\beta|x|}}{|a_x|^2}, \qquad \beta = \pi - \arccos c.
$$

The objective is **invariant under permutations of the $N$ qubits**, since both $S$ and $|T|$ depend on $|a_x|^2$ only through the Hamming weight $|x|$. It is also **invariant under bit-flip** $x \to \bar x$: $e^{-2i\beta|\bar x|} = e^{-2i\beta(N - |x|)} = e^{-2i\beta N} e^{2i\beta|x|}$ merely conjugates $T$ — a global phase invisible to $|T|$.

By a standard symmetrization argument (averaging over the symmetry group does not increase $S + |T|$, by convexity), **the optimum lies in the permutation- and bit-flip-symmetric subclass**.

---

# <span class="cat cat-results">Results</span> Post-Selection: Optimal Probe — Dicke Parametrization

In the symmetric subclass, $|a_x|^2$ depends only on $|x| = k$. Parametrize by Dicke populations $p_k$ for $k = 0, 1, \ldots, N$:
$$
|a_x|^2 \;=\; \frac{p_k}{\binom{N}{k}} \text{ for } |x| = k, \quad \sum_k p_k = 1, \quad p_k = p_{N-k}.
$$
The corresponding probe, in the Dicke basis $\ket{D_k^N} = \binom{N}{k}^{-1/2}\sum_{|x| = k}\ket{x}$, is
$$
\ket{\psi_0} \;=\; \sum_{k=0}^N q_k\,\ket{D_k^N}, \qquad q_k = \sqrt{p_k} \;\;(\text{up to phase}).
$$
This collapses the optimization from $2^N$ variables to $\lfloor N/2 \rfloor + 1$.


---

# <span class="cat cat-intro">Intro</span> DDrf: Hamiltonian Engineering

DDrf = selective, phase-controlled RF driving of nuclear spins, interleaved with dynamical decoupling on the electron spin.

If the Hamiltonian is **block-diagonal** in the electron basis,
$$
H \;=\; \ket{0}\bra{0} \otimes H_0 \;+\; \ket{1}\bra{1} \otimes H_1,
$$
then sandwiching the evolution with an electron $\pi$-pulse swaps the two branches:
$$
\ket{+}\ket{0} \xrightarrow{\,t_1,\,\pi,\,t_2\,} \tfrac{1}{\sqrt{2}}\Bigl( \ket{0} \otimes \underbrace{e^{-iH_0 t_2}e^{-iH_1 t_1}}_{U_0}\ket{0} + \ket{1} \otimes \underbrace{e^{-iH_1 t_2}e^{-iH_0 t_1}}_{U_1}\ket{0} \Bigr).
$$
Generically $U_0 \neq U_1$ — a **conditional gate**.

---

# <span class="cat cat-method">Method</span> DDrf: Pulse Sequence

![](./Meeting_260514/src/Presentation/media/DDrf_pulse.png)

---

# <span class="cat cat-method">Method</span> DDrf: Spectroscopy — NV–${}^{13}$C Hamiltonian

For NV–${}^{13}$C with RF driving:
$$
\begin{aligned}
H &= \ket{0}\bra{0} \otimes H_0 + \ket{-1}\bra{-1} \otimes H_1, \\
H_0 &= \omega_0 I_z + 2\Omega_{\text{RF}}\cos(\omega_{\text{RF}} t + \phi)\,I_x, \\
H_1 &= \omega_1 \tilde{I}_z + 2\Omega_{\text{RF}}\cos\beta\cos(\omega_{\text{RF}} t + \phi)\,\tilde{I}_x + 2\Omega_{\text{RF}}\sin\beta\cos(\omega_{\text{RF}} t + \phi)\,\tilde{I}_z.
\end{aligned}
$$
The $\ket{1}$ branch has a **tilted** nuclear quantization axis (angle $\beta$, with $\sin\beta = A_\perp/\omega_1$). This small tilt is what enables hyperfine-mediated control of nuclei with vanishing $A_\perp$.

---

# <span class="cat cat-method">Method</span> DDrf: Spectroscopy — Rotating Frame

<style scoped>
.split { display: flex; gap: 1rem; align-items: center; }
.split .text { flex: 1 1 60%; }
.split .fig  { flex: 0 0 38%; text-align: center; }
.split .fig img { max-width: 100%; }
</style>

<div class="split">
<div class="text">

In two electron-conditioned rotating frames $R_s(t) = e^{i\omega_{\text{RF}} t\,\tilde{I}_z^{(s)}}$,
$$
\begin{aligned}
H_0' &= (\omega_0 - \omega_{\text{RF}})\,I_z + \Omega_{\text{RF}}(\cos\phi\,I_x + \sin\phi\,I_y), \\
H_1' &= (\omega_1 - \omega_{\text{RF}})\,\tilde{I}_z + \Omega_{\text{RF}}\cos\beta\,(\cos\phi\,\tilde{I}_x + \sin\phi\,\tilde{I}_y).
\end{aligned}
$$
At resonance ($\omega_{\text{RF}} = \omega_1$), with $\omega_0 - \omega_{\text{RF}} \gg \Omega_{\text{RF}}$ and $\beta \to 0$, $H_1'$ becomes a pure transverse drive while $H_0'$ is a pure $z$-rotation.

</div>
<div class="fig">

![width:430px](./Meeting_260514/src/Presentation/media/DDrf_rotating_axis.png)

</div>
</div>

---

# <span class="cat cat-method">Method</span> DDrf: Spectroscopy — Full Time Evolution

Each MW $\pi$-pulse acts as an **instantaneous frame swap** $\Lambda_{s,\bar s}(t) \equiv R_s(t) R_{\bar s}(t)^\dagger$ between the two rotating frames. The full unitary $U = \sum_s \ket{s}\bra{s} \otimes U_s$ is a chain of $H_s'$-segments separated by frame swaps:
$$
\begin{aligned}
U_s \;=\;& R_s(4N\tau)^\dagger \cdot e^{-iH_s'\tau} \cdot \Lambda_{s,\bar s}((2N{-}1)\tau) \cdot e^{-iH_{\bar s}' 2\tau} \cdot \Lambda_{\bar s, s}((2N{-}3)\tau) \cdot e^{-iH_s'\tau} \cdots \\
&\cdots e^{-iH_s'\tau} \cdot \Lambda_{s,\bar s}(3\tau) \cdot e^{-iH_{\bar s}' 2\tau} \cdot \Lambda_{\bar s, s}(\tau) \cdot e^{-iH_s'\tau} \cdot R_s(0).
\end{aligned}
$$
This is exact under the assumption of negligible MW pulse duration, and **much faster than direct Schrödinger integration** when $\Omega_{\text{RF}}$ is constant: each $e^{-iH_s'\tau}$ is a single matrix exponential of a time-independent Hamiltonian.

---

# <span class="cat cat-method">Method</span> DDrf: Spectroscopy — Procedure

<style scoped>
.split { display: flex; gap: 1rem; align-items: center; }
.split .text { flex: 1 1 62%; }
.split .fig  { flex: 0 0 36%; text-align: center; }
.split .fig img { max-width: 100%; }
</style>

<div class="split">
<div class="text">

Sequence: $\pi/2 \rightarrow \text{DDrf}(N, \tau) \rightarrow \pi/2_\phi$, projecting onto $\ket{+}$ ($\phi = \pi/2$).

For $N$ nuclear spins,
$$
P_x \;=\; \tfrac{1}{2} + \tfrac{1}{2^{N+1}}\,\mathfrak{Re}\,\mathrm{Tr}\,U_0 U_1^\dagger,
$$
$$
\mathrm{Tr}\,U_0 U_1^\dagger \;=\; \prod_i \mathrm{Tr}\,U_0^i\, {U_1^i}^\dagger.
$$
Peaks appear at $\omega_{\text{RF}} = \omega_1$.

</div>
<div class="fig">

![width:420px](./Meeting_260514/src/Presentation/media/spectroscopy_sequence.png)

</div>
</div>

---

# <span class="cat cat-results">Results</span> DDrf: Spectroscopy — Reproduced Taminiau

<style scoped>
img { display: block; margin: 0.15rem auto; }
.caption { font-size: 0.75rem; text-align: center; margin-top: 0.3rem; }
</style>

![width:800px](./Meeting_260514/src/Presentation/media/Taminiau_spectroscopy.png)
![width:900px](./Meeting_260514/src/Presentation/media/Reproduce_focused.png)



---

# <span class="cat cat-method">Method</span> DDrf: Per-Cell Decomposition

The DDrf sequence is built from identical $4\tau$ cells. Each cell is itself block-diagonal:
$$
V^{(k)} \;=\; \ket{0}\bra{0} \otimes V_0^{(k)} + \ket{1}\bra{1} \otimes V_1^{(k)}, \qquad
U \;=\; \sum_{s \in \{0,1\}} \ket{s}\bra{s} \otimes \prod_{k=1}^{N/2} V_s^{(k)}.
$$

![width:480px](./Meeting_260514/src/Presentation/media/DDrf_Pulse_cell.png)

---

# <span class="cat cat-method">Method</span> DDrf: Commutation Trick

A $z$-rotation can be commuted through a transverse rotation by shifting its azimuthal angle:
$$
e^{i\alpha I_z}\,(\cos\phi\,I_x + \sin\phi\,I_y)\,e^{-i\alpha I_z} \;=\; \cos(\phi - \alpha)\,I_x + \sin(\phi - \alpha)\,I_y.
$$
Applied per cell (Taminiau limit, $\beta = 0$, $\omega_{\text{RF}} = \omega_1$):
$$
V_0^{(k)} \;=\; e^{-iH_0'\tau}\,e^{-iH_1' 2\tau}\,e^{-iH_0'\tau} \;=\; e^{-i\,2\delta_0\tau\,I_z}\,e^{-i\,2\Omega\tau\,\hat\phi_k'\cdot \vec I}.
$$
Each cell splits cleanly into a $z$-piece plus a transverse rotation. Choosing $\phi_k$ to align successive transverse axes makes the product over $k$ contract into a single conditional rotation.

---

# <span class="cat cat-results">Results</span> DDrf: Side-Peak Problem — Observation

In the Taminiau limit, $U_s = R_z(N(\omega_L - \omega_1)\tau) \cdot R_\phi(\pm N\Omega_{\text{RF}}\tau)$.

**Observation.** When $N\Omega_{\text{RF}}\tau = 2\pi$, $U_0 = U_1$ — the gate becomes **unconditional**, so a flat spectroscopy signal is expected. In practice, however:

![height:400px width:1000px](./Meeting_260514/src/Presentation/media/sidepeak.png)

---

# <span class="cat cat-method">Method</span> DDrf: Side-Peak Problem — Detuned Rotating Frame

Restoring finite detuning $\delta_1 = \omega_1 - \omega_{\text{RF}}$ (with $\beta = 0$ for clarity):
$$
H_1' \;=\; \delta_1\,I_z + \Omega\,(\cos\phi\,I_x + \sin\phi\,I_y) \;=\; \Omega_{\text{eff}}\,\hat n(\phi) \cdot \vec I,
$$
$$
\Omega_{\text{eff}} = \sqrt{\Omega^2 + \delta_1^2}, \qquad \sin\gamma = \frac{\delta_1}{\Omega_{\text{eff}}}, \qquad \hat n(\phi) = (\cos\gamma\cos\phi,\,\cos\gamma\sin\phi,\,\sin\gamma).
$$

Conjugation still works because the tilt angle $\gamma$ is invariant under $z$-rotations. Result:
$$
V_s^{\text{tot}} \;=\; e^{-iN\delta_0\tau\,I_z}\,e^{-i\Omega_{\text{eff}} N\tau\,\hat n_s \cdot \vec I}, \qquad \hat n_{0,1} = (\pm\cos\gamma,\,0,\,\sin\gamma).
$$

---

# <span class="cat cat-results">Results</span> DDrf: Side-Peak Problem — Detuned-Rabi vs. Numerics

$$
\boxed{\;
\tfrac{1}{2}\,\mathrm{Tr}(U_0 U_1^\dagger) \;=\; 1 - \frac{2\Omega^2}{\Omega^2 + \delta_1^2}\,\sin^2\!\!\left(\frac{\sqrt{\Omega^2 + \delta_1^2}\,N\tau}{2}\right)
\;}
$$

The detuned-Rabi formula reproduces both the envelope and the side-lobe period:

<style scoped>
.row2 { display: flex; gap: 1rem; align-items: center; justify-content: center; }
.row2 > div { flex: 1; text-align: center; }
.row2 img { max-width: 100%; }
</style>

<div class="row2">
<div>

![width:560px](./Meeting_260514/src/Presentation/media/detuned_rabi_overlap_48.png)

</div>
<div>

![width:560px](./Meeting_260514/src/Presentation/media/detuned_rabi_overlap.png)

</div>
</div>

---

# <span class="cat cat-strategy">Strategy</span> DDrf: Suppression — Apodized Pulse Idea

Replace the constant RF amplitude with a per-cell envelope $\Omega_k = \Omega\,f(k)$, where $f$ is a discrete window function (Hanning, Hamming, Blackman, …):

![width:2000px](./Meeting_260514/src/Presentation/media/DDrf_pulse_circuit.png)

**Intuition.** The side-lobes are essentially the discrete Fourier transform of a rectangular window. Shaping the window suppresses its sidelobes — the same trick used in classical signal processing.

---

# <span class="cat cat-method">Method</span> DDrf: Suppression — Window Catalogue

<style scoped>
img { display: block; margin: 1rem auto; }
</style>

![width:800px](./Meeting_260514/src/Presentation/media/window_shapes.png)

The four windows we compare: rectangular (the baseline that produces the side-lobes), plus three classical apodization windows from signal processing.

---

# <span class="cat cat-results">Results</span> DDrf: Suppression — Numerical Result

<style scoped>
.row2 { display: flex; gap: 1rem; align-items: center; justify-content: center; }
.row2 > div { flex: 1; text-align: center; }
.row2 img { max-width: 100%; }
</style>

<div class="row2">
<div>

![width:560px](./Meeting_260514/src/Presentation/media/DDrf_Apodization_N48_focused.png)

</div>
<div>

![width:560px](./Meeting_260514/src/Presentation/media/DDrf_Apodization_N136_focused.png)

</div>
</div>

Apodized envelopes flatten the off-resonant region while preserving the on-resonant peak. The effect grows with $N$ (right): the rectangular window develops a clean side-lobe pair, while both apodized windows stay smooth.

---

# <span class="cat cat-results">Results</span> DDrf: Suppression — Window Comparison

The normalized spectral response factorizes into a window-independent `sinc` and a window-dependent kernel:
$$
\left|\frac{F(\delta_1)}{F(0)}\right|^2 \;=\; \mathrm{sinc}^2(u) \cdot |G(u)|^2, \qquad u = \frac{\delta_1}{2\Omega\bar f}.
$$

For a rectangular (constant-amplitude) pulse,
$$
\left|\frac{F}{F(0)}\right|^2_{\mathrm{rect}} \;=\; \mathrm{sinc}^2(u).
$$

For a Blackman-apodized pulse,
$$
\left|\frac{F}{F(0)}\right|^2_{\mathrm{Black}} \;=\; \mathrm{sinc}^2(u) \cdot \left(\frac{50 u^4 - 209 u^2 + 84}{21(u^2 - 1)(u^2 - 4)}\right)^2, \qquad u = \frac{\delta_1}{2\Omega\bar f}.
$$

---

# <span class="cat cat-results">Results</span> DDrf: Suppression — Build-up Mismatch

<style scoped>
.split { display: flex; gap: 1rem; align-items: center; }
.split .text { flex: 1 1 58%; }
.split .fig  { flex: 0 0 40%; text-align: center; }
.split .fig img { max-width: 100%; }
</style>

<div class="split">
<div class="text">

The mismatch between consecutive axes is
$$
\Delta\gamma_k \;\equiv\; \gamma_{k+1} - \gamma_k \;\approx\; -\,\frac{\delta_1}{\Omega}\,\Delta f_k,
$$
where $\Delta f_k = f(k+1) - f(k)$. So the non-commutativity enters at order $\delta_1 \cdot \Delta f_k$, which is small when either $\delta_1$ is small or the window varies slowly.

</div>
<div class="fig">

![width:450px](./Meeting_260514/src/Presentation/media/apodized_rotating_axis.png)

</div>
</div>

---

# <span class="cat cat-ongoing">Ongoing</span> DDrf: Outlook — Beyond Constant $\Omega_{\text{RF}}$

The full-evolution formalism above assumes $\Omega_{\text{RF}}$ is time-independent (compatible with the RWA). With a Gaussian envelope
$$
\Omega_{\text{RF}}(t) \;=\; \Omega_0\,e^{-(t - t_k)^2 / 2\sigma^2},
$$
the per-segment propagator $e^{-iH_{0,1}'\tau}$ is no longer exact, and direct Schrödinger integration takes tens of minutes per frequency point.

**Approach.** Magnus expansion, valid when $\Omega_{\text{RF}}$ varies slowly relative to $\omega_{\text{RF}}$.

---

# <span class="cat cat-ongoing">Ongoing</span> DDrf: Outlook — Magnus Expansion

For $\partial_t U = -iH(t)U$, write $U = e^{\Omega(t)}$ with $\Omega(t) = \sum_n \Omega_n(t)$:
$$
\Omega_1 = \int_0^t A_1\,dt_1, \qquad \Omega_2 = \tfrac{1}{2}\!\int\!\!\int [A_1, A_2]\,dt_1\,dt_2, \ldots
$$

Computed terms (with $f(t)$ the Gaussian envelope):
$$
\begin{aligned}
\Omega_1(T) &= -i\bigl(\delta_{(0,1)} T\,I_z + c_1\,(\cos\phi\,I_x + \sin\phi\,I_y)\bigr), \\
\Omega_3(T) &= \tfrac{\delta_{(0,1)}^2}{24}\,K_1\,(\cos\phi\,I_x + \sin\phi\,I_y) + \tfrac{\delta_{(0,1)}}{24}\,K_2\,I_z, \\
\Omega_2 &= \Omega_4 = 0,
\end{aligned}
$$
with $c_1 = \int_0^T f$ and $K_{1,2}$ triple integrals of $f$. Convergence is guaranteed when $\int_0^T \|A(s)\|_2\,ds < \pi$ — likely satisfied here; **simulation pending**.

---

# <span class="cat cat-ongoing">Ongoing</span> DDrf: Outlook — Gaussian Pulse Shaping

<style scoped>
.row2 { display: flex; gap: 1rem; align-items: center; justify-content: center; }
.row2 > div { flex: 1; text-align: center; }
.row2 img { max-width: 100%; }
</style>

<div class="row2">
<div>

![width:560px](./Meeting_260514/src/Presentation/media/gaussian_apod_spectroscopy_48.png)

</div>
<div>

![width:560px](./Meeting_260514/src/Presentation/media/gaussian_apod_spectroscopy_136.png)

</div>
</div>

---

# <span class="cat cat-intro">Intro</span> Mattermost–Outline: Update

Special thanks to WD Lee.

<style scoped>
img { display: block; margin: 0.6rem auto; max-height: 480px; }
</style>

![width:900px](./Meeting_260514/src/Presentation/media/outline-wiki.png)

---

# <span class="cat cat-intro">Intro</span> Mattermost–Outline: Required Features

<ul class="req-list">
<li><span class="status status-done">DONE</span> Mattermost: private channel</li>
<li><span class="status status-progress">WIP</span> Send alert from Outline changes to Mattermost</li>
<li><span class="status status-tentative">MAYBE</span> Code / PPT update</li>
<li><span class="status status-progress">WIP</span> Cool-paper channel update to Outline</li>
<li><span class="status status-tentative">MAYBE</span> Equipment directory / Diamond sample</li>
<li><span class="status status-done">DONE</span> Emergency contact list</li>
</ul>

---

# <span class="cat cat-results">Results</span> Mattermost–Outline: Cool-Paper Updates

<style scoped>
.row2 { display: flex; gap: 1rem; align-items: center; justify-content: center; }
.row2 > div { flex: 1; text-align: center; }
.row2 img { max-width: 100%; max-height: 480px; }
</style>

<div class="row2">
<div>

![width:520px](./Meeting_260514/src/Presentation/media/mattermost-example-cool-paper.png)

</div>
<div>

![width:520px](./Meeting_260514/src/Presentation/media/outline-example-cool-paper.png)

</div>
</div>

---

# <span class="cat cat-results">Results</span> Mattermost–Outline: Alerts for Outline Updates

<style scoped>
.row2 { display: flex; gap: 0.8rem; align-items: center; justify-content: center; margin-bottom: 0.5rem; }
.row2 > div { flex: 1; text-align: center; }
.row2 img { max-width: 100%; max-height: 320px; }
ul { font-size: 0.8rem; }
</style>

<div class="row2">
<div>

![width:480px](./Meeting_260514/src/Presentation/media/outline-example-alert.png)

</div>
<div>

![width:480px](./Meeting_260514/src/Presentation/media/mattermost-example-alert.png)

</div>
</div>

An alert is sent when:
- a new document is added, a comment is added, or a document is modified;
- and only if the *collection* has a corresponding Mattermost channel.

---

# <span class="cat cat-results">Results</span> Mattermost–Outline: Dataview Feature

<style scoped>
.row2 { display: flex; gap: 0.6rem; align-items: center; justify-content: center; margin-bottom: 0.3rem; }
.row2 > div { flex: 1; text-align: center; }
.row2 img { max-width: 100%; max-height: 200px; }
ul { font-size: 0.8rem; }
</style>

<div class="row2">
<div>

![width:420px](./Meeting_260514/src/Presentation/media/outline-example-table.png)

</div>
<div>

![width:420px](./Meeting_260514/src/Presentation/media/outline-example-kanban.png)

</div>
</div>

<div class="row2">
<div>

![width:420px](./Meeting_260514/src/Presentation/media/outline-example-calendar.png)

</div>
<div>

![width:420px](./Meeting_260514/src/Presentation/media/outline-example-timeline.png)

</div>
</div>

- Notion-like dataview (requires SQL-like embedding).
- Table, Kanban, Calendar, and timeline views are supported.

---

# <span class="cat cat-results">Results</span> Mattermost–Outline: Datafield Feature

<style scoped>
img { display: block; margin: 0.4rem auto; max-height: 420px; }
</style>

![width:720px](./Meeting_260514/src/Presentation/media/outline-example-datafield.png)

Datafield values are not searchable as keywords on their own — the dataview feature on the previous slide is the workaround.

---

# <span class="cat cat-ongoing">Ongoing</span> Others

<style scoped>
.split { display: flex; gap: 1.5rem; align-items: center; }
.split .text { flex: 1 1 65%; }
.split .text ul { font-size: 0.9rem; }
.split .idcard { flex: 0 0 30%; text-align: center; }
.split .idcard img {
  max-width: 100%;
  max-height: 340px;
  border: 1px solid #cfd3d8;
  border-radius: 6px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.10);
}
</style>

<div class="split">
<div class="text">

- Set up a desktop with an RTX 3090, accessible remotely from anywhere.
- Borrowing a server from Prof. Yosep Kim (Korea University).
- (personal) Promoted to principal candidate for the Fulbright STEM scholarship on Apr 16; Cornell Applied Physics initially responded positively but ultimately rejected my application, so I formally declined the Fulbright STEM scholarship.
- (personal) Now an official research intern here!

</div>
<div class="idcard">

![](./Meeting_260514/src/Presentation/media/ID_card.jpg)

</div>
</div>


---

<!-- _class: titlepage -->

<style scoped>
.container{
   display: flex;
   align-items: center;
   justify-content: center;
   width: 100%;
   height: 100%;
}
.center-content {
   text-align: center;
   color: #00356B;
}
.center-content .title {
   font-size: 48pt;
   margin-bottom: 40px;
}
.center-content .subtitle {
   font-size: 24pt;
   color: #666;
}
</style>

<div class="container">
<div class="center-content">

<div class="title">
Thank You!
</div>

<div class="subtitle">
Questions & Discussion
</div>

</div>
</div>
