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
<img src="./media/images/PauleeLogo.png" style="max-width: 100%; height: 100%; object-fit: contain;">
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

![](./Meeting_260605/src/Presentation/media/sensing-diagram.png)

</div>

---

# <span class="cat cat-intro">Intro</span> Post-Selection: Quantum Fisher Information

The quantum Fisher information (QFI) quantifies how rapidly a quantum state, represented by a density matrix, changes with respect to the $B$-field. The faster the change, the greater the sensitivity:
$$
F_Q =  \mathrm{Tr} \rho L^2 .
$$


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
\rho =  p_0 \ket{0}\bra{0} \otimes \rho_0 +  p_1 \ket{1}\bra{1} \otimes \rho_1  \longrightarrow  
\begin{cases}
\rho_0 & \text{if the ancilla is measured in $\ket{0}$, with probability $p_0$,}\\
\rho_1 & \text{if the ancilla is measured in $\ket{1}$, with probability $p_1$.}
\end{cases}
$$

<div class="figcard">

![](./Meeting_260605/src/Presentation/media/Post-selection-pipeline.png)

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


![width:1050px](Meeting_260604/src/Presentation/media/ent_ps.svg)

Last time I ran numerical simulations to identify which probe state and post-selection filter achieve the maximum QFI. I had tried numerical simulation:
$$
\max_{\theta ,U, V} F_C^B
$$

Recently, I got insight to analyze QFI enhancement.

---

# <span class="cat cat-method">Method</span> Post-Selection: Setup

The state evolves through four stages:
$$
\ket\psi \underbrace{\rightarrow}_{\text{Time evo}}  \ket{\psi_0} = e^{-i H_B \tau}\ket\psi \underbrace{\rightarrow}_{\text{Phase Damping}}  \rho = \mathcal{D}(\rho_0) \underbrace{\rightarrow}_{\text{Post-selection}} \sigma = \frac{K\rho K^{\dagger}}{\mathrm{Tr} K\rho K^{\dagger}}.
$$

Without loss of generality, write the pure probe state in the computational basis:
$$
\ket{\psi} =  \sum_x a_x \ket{x}, \qquad \sum_x |a_x|^2 = 1,
$$
where $x$ ranges over computational-basis bitstrings like $\ket{00\cdots 0}$.

**Time evolution.** Under the detuned magnetic field,
$$
\ket{\psi_0} =  e^{-i H_B \tau}\ket\psi =  \sum_x a_x e^{i\theta |x|} \ket{x},
$$
where $|x|$ is the Hamming weight.

---

# <span class="cat cat-method">Method</span> Post-Selection: Phase-Damping Channel

The density matrix before damping is
$$
\rho_0 =  \sum_{x,y} a_x \bar{a}_y  e^{i\theta(|x| - |y|)} \ket{x}\bra{y}.
$$

After phase damping, off-diagonal elements decay by $\eta^{d(x,y)}$, where $d(x,y) = |x\oplus y|$ is the Hamming distance:
$$
\begin{aligned}
\rho_0 \rightarrow  \rho
&= \sum_{x,y} a_x \bar{a}_y  e^{i\theta(|x| - |y|)}  \eta^{d(x,y)} \ket{x}\bra{y} \\
&= U(\theta)  D_a  C_N  D_a^{\dagger}  U^\dagger(\theta),
\end{aligned}
$$
with $D_a = \mathrm{diag}(a_x)$ and $C_N = (I + \eta X)^{\otimes N}$.

Phase damping kills coherences across different Hamming weights but preserves populations $|a_x|^2$, so $\rho$ has the eigenvalue structure of a *mixed* state on $\mathrm{supp} \rho$.

---

# <span class="cat cat-method">Method</span> Post-Selection: Filter Step

A $\theta$-independent Kraus operator $K$ (with $K^\dagger K \preceq I$) is applied to $\rho$, producing the success branch
$$
\sigma =  \frac{K\rho K^\dagger}{p_s}, \qquad p_s = \mathrm{Tr}\bigl[K\rho K^\dagger\bigr].
$$

The conditional phase QFI
$$
F^{\mathrm{ps}}_{\theta,Q} =  \mathrm{Tr}(\sigma L_\sigma^2)
$$
is the quantity we bound and optimize in the remaining slides.

---

# <span class="cat cat-strategy">Strategy</span> Post-Selection: Quick Summary

For an $N$-qubit system, the phase QFI of the post-selected state obeys
$$
F^{\mathrm{ps}}_{\theta,Q} \le  N^2 \frac{\eta^2}{1-\eta^2}, \qquad \eta = e^{-\tau_d},\quad \tau_d = (\tau/T_2)^p.
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
F^{\mathrm{ps}}_{\theta,Q} \le  N^2 \frac{\eta^2}{1-\eta^2}, \qquad \eta = e^{-\tau_d},\quad \tau_d = (\tau/T_2)^p.
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
\sigma =  \frac{K\rho K^\dagger}{p_s}, \qquad p_s = \mathrm{Tr} K\rho K^\dagger.
$$
The QFI is obtained from the symmetric logarithmic derivative (SLD):
$$
F_Q =  \mathrm{Tr} \sigma L^2, \qquad \dot\sigma =  \tfrac{1}{2}(\sigma L + L\sigma).
$$

Computing the SLD directly is generally hard, so we work with another operator instead.

---

# <span class="cat cat-method">Method</span> Post-Selection: Bounding via $M$

Introduce $M = \sigma^{-1/2}\dot\sigma \sigma^{-1/2}$. Substituting the SLD definition,
$$
M =  \tfrac{1}{2}\bigl(\sigma^{1/2} L \sigma^{-1/2} + \sigma^{-1/2} L \sigma^{1/2}\bigr).
$$
The key inequality is $\boxed{\mathrm{Tr} \sigma L^2 \le \mathrm{Tr} \sigma M^2}$.

In the eigenbasis $\sigma = \sum_k p_k \ket{p_k}\bra{p_k}$ with $p_k \neq 0$,
$$
L_{ab} = \frac{2\dot\sigma_{ab}}{p_a + p_b}, \qquad M_{ab} = \frac{\dot\sigma_{ab}}{\sqrt{p_a p_b}}.
$$

$$
\mathrm{Tr} \sigma L^2 = \sum_{a,b} \frac{2|\dot\sigma_{ab}|^2}{p_a + p_b}, \qquad
\mathrm{Tr} \sigma M^2 = \sum_{a,b} \frac{|\dot\sigma_{ab}|^2 (p_a + p_b)}{2 p_a p_b}.
$$

Term by term, $m_{ab}/l_{ab} = (p_a+p_b)^2/(4 p_a p_b) \ge 1$, with **saturation when $p_a = p_b$**.

---

# <span class="cat cat-method">Method</span> Post-Selection: Lifting to $\rho$

Here, $\sigma$ lives in the subspace of $\rho$ selected by $K$. Accordingly, $M$ lives in that same subspace, and its operator norm is bounded by that of $M_\rho$:
$$
\mathrm{Tr} \sigma M^2 \le  \|M_\rho\|_\infty^2.
$$

On the other hand,
$$
M_\rho =  \rho^{-1/2}\dot\rho \rho^{-1/2}, \qquad
\dot\rho =  -i[H_B, \rho] =  \sum_i -\tfrac{i}{2}[Z_i, \rho].
$$

Writing $M_\rho = \sum_i M_i$ with $M_i = \rho^{-1/2}\bigl(-\tfrac{i}{2}[Z_i, \rho]\bigr)\rho^{-1/2}$,
$$
\|M_\rho\|_\infty \le  \sum_i \|M_i\|_\infty.
$$

Here, $C_N = C^{\otimes N}$ and $[H_B, D_a] = 0$ force equality for any full-support probe.

---

# <span class="cat cat-method">Method</span> Post-Selection: Chained Inequality

Putting it together,
$$
F_Q =  \mathrm{Tr} \sigma L^2 \le  \mathrm{Tr} \sigma M_\sigma^2 \le  \|M_\rho\|_\infty^2 \le  \Bigl(\sum_i \|M_i\|_\infty\Bigr)^2.
$$
When every step saturates simultaneously,
$$
F_Q \le  N^2 \|M_i\|_\infty^2.
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
\rho^{-1/2}\dot\rho_j \rho^{-1/2}\ket{v} &= \lambda\ket{v} \\
\dot\rho_j \bigl(\rho^{-1/2}\ket{v}\bigr) &= \lambda \rho \bigl(\rho^{-1/2}\ket{v}\bigr) \\
\dot\rho_j\ket{w} &= \lambda \rho \ket{w},
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
K\mathbf{x} =  \omega^2 M \mathbf{x},
$$
a GEP whose generalized eigenvalues are the squared normal-mode frequencies.


---

# <span class="cat cat-method">Method</span> Post-Selection: Block Form of the GEP

The relevant block-form is
$$
\dot\rho_j =  \begin{pmatrix} 0 & -i\eta B_j \\ i\eta B^\dagger_j & 0 \end{pmatrix},
$$
so the GEP reduces to
$$
\begin{pmatrix} 0 & -i\eta B_j \\ i\eta B^\dagger_j & 0 \end{pmatrix}
\begin{pmatrix} w_1 \\ w_2 \end{pmatrix}
 =  \lambda
\begin{pmatrix} A_j & \eta B_j \\ \eta B^\dagger_j & D_j \end{pmatrix}
\begin{pmatrix} w_1 \\ w_2 \end{pmatrix}.
$$

---

# <span class="cat cat-method">Method</span> Post-Selection: Schur Decomposition

Applying a Schur decomposition with $B_j = A_j^{1/2} W_j D_j^{1/2}$,
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

Stripping the outer factors ($A_j^{1/2} = \tilde{w}_1 , D_j^{1/2} = \tilde{w}_2$),
$$
\begin{pmatrix} 0 & -i\eta W_j \\ i\eta W_j^\dagger & 0 \end{pmatrix}
\begin{pmatrix} \tilde{w}_1 \\ \tilde{w}_2 \end{pmatrix}
 =  \lambda
\begin{pmatrix} I & \eta W_j \\ \eta W_j^\dagger & I \end{pmatrix}
\begin{pmatrix} \tilde{w}_1 \\ \tilde{w}_2 \end{pmatrix}.
$$

---

# <span class="cat cat-method">Method</span> Post-Selection: Reducing via SVD

Take the SVD $W_j = U \Sigma_j V^\dagger$ with $\Sigma_j = \mathrm{diag}(s_{jk})$:
$$
\begin{pmatrix} 0 & -i\eta s_{jk} \\ i\eta s_{jk} & 0 \end{pmatrix}
\begin{pmatrix} \tilde{u}_{jk1} \\ \tilde{u}_{jk2} \end{pmatrix}
 =  \lambda_{jk}
\begin{pmatrix} 1 & \eta s_{jk} \\ \eta s_{jk} & 1 \end{pmatrix}
\begin{pmatrix} \tilde{u}_{jk1} \\ \tilde{u}_{jk2} \end{pmatrix}.
$$

The determinant condition
$$
\det \begin{pmatrix} -\lambda & -\eta s_{jk}(\lambda + i) \\ \eta s_{jk}(-\lambda + i) & -\lambda \end{pmatrix} = 0
$$
gives
$$
\lambda_{jk} =  \pm\sqrt{\frac{\eta^2 s_{jk}^2}{1 - \eta^2 s_{jk}^2}}.
$$

---

# <span class="cat cat-results">Results</span> Post-Selection: Single-Qubit Bound

Since $0 \le s_{jk} \le 1$, the magnitude of $\lambda_{jk}$ is maximized at $s_{jk} = 1$, giving
$$
\|M_j\|_\infty \le  \frac{\eta}{\sqrt{1 - \eta^2}}.
$$

This is one more inequality, which must hold for every qubit. Generically $s_{jk} < 1$ for mixed states; however, for any initial state $\ket{\psi_0} = \sum_x a_x \ket{x}$ with $a_x \neq 0$ for all $x$, phase damping still lets us attain $s_{jk} = 1$. (Precisely, this condition is satisfied when we set-up optimal filter.)

At saturation,
$$
\|M\|_\infty =  \frac{N\eta}{\sqrt{1-\eta^2}}, \qquad
F_Q \le  \|M\|_\infty^2 =  \frac{N^2 \eta^2}{1-\eta^2}.
$$

---

# <span class="cat cat-results">Results</span> Post-Selection: Saturating Probe States

Every pure state $\ket{\psi_0} = \sum_x a_x \ket{x}$ with $a_x \neq 0$ for **all** computational-basis strings $x$ saturates the global bound
$$
F^{\mathrm{ps}}_{\theta,Q} =  \frac{N^2 \eta^2}{1-\eta^2}.
$$

The saturating class is **open** (the condition $|a_x| > 0$ for all $x$ is open) and **dense** (full-support probes are generic) in the projective Hilbert space $\mathbb{CP}^{2^N - 1}$.

---

# <span class="cat cat-method">Method</span> Post-Selection: Back to the GEP

As shown above, the phase-damped state factorizes as $\rho = D_a C_N D_a^\dagger$ with $C_N = (I + \eta X)^{\otimes N}$. $D_a$ is invertible iff $a_x \neq 0$ for all $x$. Return to the GEP:
$$
\dot\rho_j\ket{w} =  \lambda \rho \ket{w}.
$$
Then
$$
\begin{aligned}
-i[H_B, D_a C_N D_a^\dagger]\ket{w} &= \lambda D_a C_N D_a^\dagger\ket{w} \\
-i D_a[H_B, C_N] D_a^\dagger\ket{w} &= \lambda D_a C_N D_a^\dagger\ket{w},
\end{aligned}
$$
since $H_B$ and $D_a$ commute. Letting $\ket{\epsilon} = D_a^\dagger\ket{w}$, we obtain another GEP:
$$
-i[H_B, C_N]\ket{\epsilon} =  \lambda C_N\ket{\epsilon}.
$$

---

# <span class="cat cat-method">Method</span> Post-Selection: Single-Qubit GEP

With $C_N = (I + \eta X)^{\otimes N} = C^{\otimes N}$ and $H_B = \tfrac{1}{2}\sum_i Z_i$,
$$
-\tfrac{i}{2}\sum_i [Z_i, C^{\otimes N}]\ket{\epsilon} =  \lambda C^{\otimes N}\ket{\epsilon}.
$$
For the product ansatz $\ket{\epsilon} = \bigotimes_i \ket{\epsilon_i}$, each factor satisfies a single-qubit GEP:
$$
-\tfrac{i}{2}[Z, C]\ket{\epsilon_i} =  \lambda C\ket{\epsilon_i}.
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
\ket{w^\pm} =  (D_a^{-1})^{\dagger}\ket{\epsilon^\pm},
$$
so
$$
\dot\rho \ket{w^\pm} =  \pm \frac{N\eta}{\sqrt{1-\eta^2}} \rho \ket{w^\pm}.
$$

Therefore $M = \rho^{-1/2}\dot\rho \rho^{-1/2}$ contains the eigenvalues $\pm  N\eta/\sqrt{1-\eta^2}$, giving $\|M\|_\infty = N\eta/\sqrt{1-\eta^2}$ and
$$
F^{\mathrm{ps}}_{\theta,Q} =  \frac{N^2 \eta^2}{1-\eta^2},
$$
with filter vectors $\ket{w^\pm}$.

---

# <span class="cat cat-results">Results</span> Post-Selection: Filter Basis $\ket{\Psi^\pm}$

Since $\ket{w} = \rho^{-1/2}\ket{v}$, the filter basis (denoted $\ket{\Psi^\pm}$) is
$$
\boxed{\qquad\ket{\Psi^\pm} =  \ket{v^\pm} =  \frac{\rho^{1/2}\ket{w^\pm}}{(1-\eta^2)^{N/2}}.\qquad}
$$

The two basis vectors are mutually orthogonal:
$$
\braket{\Psi^+ | \Psi^-} =  \frac{\bra{w^+}\rho\ket{w^-}}{(1-\eta^2)^N} =  0,
$$
inherited from $\rho$-biorthogonality.

---

# <span class="cat cat-method">Method</span> Post-Selection: Matched Kraus Operator

The filter is a rank-2 projector onto $\mathrm{span}\{\ket{\Psi^+}, \ket{\Psi^-}\}$:
$$
P_\pm =  \ket{\Psi^+}\bra{\Psi^+} + \ket{\Psi^-}\bra{\Psi^-}.
$$
The matched Kraus operator is
$$
K =  \sqrt{\frac{p_s}{2}} P_\pm \rho^{-1/2}.
$$
Applying it to $\rho$:
$$
K\rho K^\dagger =  \frac{p_s}{2} P_\pm \rho^{-1/2}\rho \rho^{-1/2} P_\pm =  \frac{p_s}{2} P_\pm^2 =  \frac{p_s}{2} P_\pm.
$$

---

# <span class="cat cat-results">Results</span> Post-Selection: Accepted State

The accepted state is
$$
\sigma =  \frac{K\rho K^\dagger}{p_s} =  \frac{P_\pm}{2} =  \frac{\ket{\Psi^+}\bra{\Psi^+} + \ket{\Psi^-}\bra{\Psi^-}}{2}.
$$
That is, the accepted state is the *maximally mixed state on the filter subspace* — exactly the structural requirement for QFI saturation.

This form looks unfamiliar, so we re-express the filter as
$$
K =  \ket{\Psi^+}\bra{w^+} + \ket{\Psi^-}\bra{w^-}.
$$

It can be implemented as $K = \sqrt{1 - \gamma} \ket{00\cdots 0}\bra{00\cdots 0} + \ket{00\cdots 0}\bra{10\cdots 0}$ by sandwiching with appropriate unitaries before and after the post-selection step, **setting one post-selection strength to zero and the others to one**.

---

# <span class="cat cat-method">Method</span> Post-Selection: POVM Constraint



The POVM constraint is $K^\dagger K \preceq I$. Explicitly,
$$
K^\dagger K = \frac{p_s}{2} \rho^{-1/2}P_\pm\rho^{-1/2} = \frac{p_s}{2(1-\eta^2)^N} \Xi
$$
where
$$
\begin{align}
\Xi &= \ket{w^+}\bra{w^+} + \ket{w^-}\bra{w^-} =  (D_a^{-1})^\dagger \Pi D_a^{-1} \\
\Pi &=  \ket{\epsilon^+}^{\otimes N}\bra{\epsilon^+}^{\otimes N} + \ket{\epsilon^-}^{\otimes N}\bra{\epsilon^-}^{\otimes N}.
\end{align}
$$
And its component form is
$$
\Pi_{xy} =  \frac{1}{2^{N-1}}\cos\bigl(\alpha(|x| - |y|)\bigr).
$$


---

# <span class="cat cat-method">Method</span> Post-Selection: Success Probability — Setup

The constraint $K^\dagger K \preceq I$ requires $\lambda_{\max}(K^\dagger K)\le 1$, hence
$$
p_s \;\le\; \frac{2(1-\eta^2)^N}{\lambda_{\max}(\Xi)}.
$$
The maximum $p_s$ is obtained when this is tight. 

<!-- The condition $\Pi\ket{u} = \mu D_p\ket{u}$ is equivalent to $D_p^{-1/2}\Pi D_p^{-1/2}\ket{w} = \mu\ket{w}$ with $\ket{w} = D_p^{1/2}\ket{u}$, so we want the eigenvalues of the symmetric operator $D_p^{-1/2}\Pi D_p^{-1/2}$.

In the 2D image of $\Pi$ (basis $\{\ket{v^+}^{\otimes N}, \ket{v^-}^{\otimes N}\}$), $\Pi$ acts as the rank-2 identity. -->

<!-- ---

# <span class="cat cat-results">Results</span> Post-Selection: Success Probability — Result

The Gram matrix of $\{D_p^{-1/2}\ket{v^\pm}^{\otimes N}\}$ is
$$
G =  \begin{pmatrix} \hat S & \hat T \\ \overline{\hat T} & \hat S \end{pmatrix},
$$
and the eigenvalues of $D_p^{-1/2}\Pi D_p^{-1/2}$ in this basis are those of $G$, namely $\hat S \pm |\hat T|$.

So $\lambda_{\max}(\Xi) = \hat S + |\hat T| = (S + |T|)/2^N$, giving
$$
\boxed{ p_s^{*} =  \frac{2(1-\eta^2)^N}{\lambda_{\max}(\Xi)} =  \frac{2^{N+1}(1-\eta^2)^N}{S + |T|}. }
$$

---

# <span class="cat cat-results">Results</span> Post-Selection: Optimal Probe — Symmetries

The optimization is over all probe magnitudes $\{|a_x|^2\}_x$ with $\sum_x |a_x|^2 = 1$. The objective $S + |T|$ depends on the magnitudes through
$$
S =  \sum_x \frac{1}{|a_x|^2}, \qquad T =  \sum_x \frac{e^{-2i\beta|x|}}{|a_x|^2}, \qquad \beta = \pi - \arccos c.
$$

The objective is **invariant under permutations of the $N$ qubits**, since both $S$ and $|T|$ depend on $|a_x|^2$ only through the Hamming weight $|x|$. It is also **invariant under bit-flip** $x \to \bar x$: $e^{-2i\beta|\bar x|} = e^{-2i\beta(N - |x|)} = e^{-2i\beta N} e^{2i\beta|x|}$ merely conjugates $T$ — a global phase invisible to $|T|$.

By a standard symmetrization argument (averaging over the symmetry group does not increase $S + |T|$, by convexity), **the optimum lies in the permutation- and bit-flip-symmetric subclass**.

---

# <span class="cat cat-results">Results</span> Post-Selection: Optimal Probe — Dicke Parametrization

In the symmetric subclass, $|a_x|^2$ depends only on $|x| = k$. Parametrize by Dicke populations $p_k$ for $k = 0, 1, \ldots, N$:
$$
|a_x|^2 =  \frac{p_k}{\binom{N}{k}} \text{ for } |x| = k, \quad \sum_k p_k = 1, \quad p_k = p_{N-k}.
$$
The corresponding probe, in the Dicke basis $\ket{D_k^N} = \binom{N}{k}^{-1/2}\sum_{|x| = k}\ket{x}$, is
$$
\ket{\psi_0} =  \sum_{k=0}^N q_k \ket{D_k^N}.
$$
This collapses the optimization from $2^N$ variables to $\lfloor N/2 \rfloor + 1$. -->

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
