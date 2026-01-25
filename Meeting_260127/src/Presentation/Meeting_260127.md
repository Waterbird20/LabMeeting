---

title       : Taming Multiparticle Entanglement
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
   font-size: 36pt;
}

.col-right{
   flex: 0 0 30%;
   display: flex;
   align-items: center;
   justify-content: center;
}
</style>

<div class="container">

<div class="col-left">

<div class="title">
Taming Multiparticle Entanglement
</div>

<div class="author">
Donghun Jung
</div>

<div class="date">
27 Jan 2026
</div>

<div class="organization">
Department of Physics, Sungkyunkwan University
<br>
QuiME Lab, Center for Quantum Technology, Korea Institute of Science Technology
</div>

</div>

<div class="col-right">
<img src="media/images/PauleeLogo.png" style="max-width: 100%; height: auto; object-fit: contain;">
</div>

</div>

---

<!-- backgroundColor: white -->

# Paper Information

<style scoped>
section h1 {
  font-size: 28pt;
}
p, li {
   font-size: 20pt;
   color: #000000;
}
</style>

## Taming Multiparticle Entanglement

**Authors:** Bastian Jungnitsch, Tobias Moroder1, and Otfried Gühne
**Journal:** Physical Review Letters 106, 190502 (2011)
**arXiv:** 1010.6049


### Key Contributions:
- Efficient criterion for **genuine multipartite entanglement (GME)** via PPT mixtures
- **Semidefinite programming (SDP)** formulation for practical detection
- Fully decomposable entanglement witnesses

---

# Outline

<style scoped>
.container{
   display: flex;
   align-items: center;
   width: 100%;
   height: 100%;
}
.col-left-content{
   margin-left: -100px;
   flex: 0 0 50%;
   padding-right: 0rem;
   padding-left: -3rem;
   padding-bottom: 5.5rem;
   color: #000000;
}

.col-right-content{
   margin-left: -150px;
   flex: 0 0 50%;
   display: flex;
   align-items: center;
   justify-content: center;
   padding-bottom: 10rem;
}

li {
   font-size: 0.85rem;
}

</style>

<div class="container">
<div class="col-left-content">

</div>

<div class="col-right-content">

</div>
</div>

---

# Motivation

<style scoped>
section h1 {
  font-size: 26pt;
}

p, li {
   font-size: 22pt;
   line-height: 1rem;
   color: #000000;
}

.highlight-box {
   background: blue;
   color: white;
   padding: 20px;
   border-radius: 10px;
   margin: 0px 50px;
}
</style>

**The Challenge:** How do we efficiently verify that a state has *genuine* multipartite entanglement that cannot be reduced to bipartite correlations?

### Separability and Entanglement

A bipartite state $\rho$ is **separable** if:
$$
\begin{align}
\ket{\psi} &= \sum_i \lambda_i \ket{u_i}_{A} \otimes \ket{v_i}_B\\
\rho &= \sum_{i} p_i \rho^{A}_{i} \otimes \rho^{B}_{i} \\
\end{align}
$$
where $\sum_i p_i = 1$ and $p_i > 0$.

**Problem:** Directly determining separability is **NP-hard**!
- How can we split the system into two subsystem? 
- It is costy operation!

---

# Examples of Multiparticle Entangled States

<style scoped>
section h1 {
  font-size: 26pt;
}
p, li {
   font-size: 18pt;
   line-height: 1rem;
   color: #000000;
}
</style>

#### GHZ State (Greenberger-Horne-Zeilinger)
$$
|GHZ_n\rangle = \frac{1}{\sqrt{2}}\left(|00\cdots0\rangle + |11\cdots1\rangle\right)
$$

#### W State
$$
|W_3\rangle = \frac{1}{\sqrt{3}}\left(|001\rangle + |010\rangle + |100\rangle\right)
$$

#### Cluster State (4-qubit linear)
$$
|Cl_4\rangle = \frac{1}{2}\left(|0000\rangle + |0011\rangle + |1100\rangle - |1111\rangle\right)
$$

---

<!-- TODO change Title -->

# Examples of Multiparticle Entangled States

For pure state, SVD is the most effective way to quantify entanglement. For example, consider two-qubit bell state (though it is trivial example).

**Step 1.** Write the state. $\ket{\Phi^{+}} = \frac{1}{\sqrt{2}} \ket{00} + 0\ket{01} + 0\ket{10} + \frac{1}{\sqrt{2}}\ket{11} = \sum_{i,j} c_{ij} \ket{i}_{A}\ket{j}_{B}$
**Step 2.** Form Coefficient Matrix. 
$$
C = \begin{pmatrix}
c_{00} & c_{01} \\
c_{10} & c_{11} 
\end{pmatrix}
= 
\frac{1}{\sqrt{2}}
\begin{pmatrix}
1 & 0 \\
0 & 1 
\end{pmatrix}
$$
**Step 3.** Find SVD of $C$, that is, $C=U\Sigma V^{\dagger}$.
$$
C = \frac{1}{\sqrt{2}}
\begin{pmatrix}
1 & 0 \\
0 & 1 
\end{pmatrix}
= 
\begin{pmatrix}
1 & 0 \\
0 & 1 
\end{pmatrix}
\begin{pmatrix}
\frac{1}{\sqrt{2}} & 0 \\
0 & \frac{1}{\sqrt{2}}
\end{pmatrix}
\begin{pmatrix}
1 & 0 \\
0 & 1 
\end{pmatrix}
$$
**Step 4.** So two Schmidt coefficient, suggesting it is entangled state. Try for remaining Bell states!
 

---

<!-- TODO change Title -->

# Challenges for Entanglement Detection and Quantification; pure state

Exponential cost is inevitalbe. For an $m \times n$ matrix:
| Resource | Complexity |
|----------|------------|
| **Time** | $\mathcal{O}(\min(mn^2, m^2n))$ $\simeq$ $\mathcal{O}(mn \cdot \min(m,n))$ |
| **Memory** | $\mathcal{O}(mn)$ for matrix + $\mathcal{O}(m^2 + n^2)$ for $U, V$ |

For $N$ qubits with bipartition $k | (N-k)$, the size of matrix becomes $2^k \times 2^{N-k}$. Thing goes worse if we have to try every possible combination of bipartition, $2^{N-1}$. 


---

<!-- TODO change Title -->

# Challenges for Entanglement Detection and Quantification; mixed state


Even worse, For mixed states, there's no "canonical form" like Schmidt decomposition. The same $\rho$ can be written as:
$$\rho = \sum_k p_k |\psi_k\rangle\langle\psi_k| = \sum_j q_j |\phi_j\rangle\langle\phi_j|$$
with completely different ensembles! One might use only product states (separable), another might not (entangled representation), while the **physical state is the same**.


---

# Biseparable States: Definition


To generalize, a state is separable with respect with to bipartition $M|\bar{M}$, if:
$$
\rho^{\text{sep}}_{M|\bar{M}} = \sum_i p_i \, \rho_M^{(i)} \otimes \rho_{\bar{M}}^{(i)}
$$
A state is **biseparable** if it can be written as a mixture:
$$
\rho^{\text{bs}} = \sum_{i} p_i \rho^{\text{sep}}_{M_{i}|\bar{M}_{i}}
$$
where $\sum_i p_i = 1$. Note that non-trivial form, (at least two $p_i\neq 0$), must be mixed sate.

---

# Main Idea: Checking whether PPT mixture exists

It is well-known that separable states are also Positive Partial Transpose (PPT). We denote such states by $\rho^{\text{ppt}}_{M|\bar{M}}$. Thus, we ask whether a state can be written as
$$
\rho^{\text{pmix}} = \sum_{i} p_i \rho^{\text{ppt}}_{M_{i}|\bar{M}_{i}}
$$
We call states of this form **PPT mixtures**.

Clearly, any biseparable state is a PPT mixture, so proving that a state is no PPT mixture implies **genuine multipartite entanglement**. 

### Genuine Multipartite Entanglement (GME)
A state is **genuinely multipartite entangled** if and only if it is **NOT biseparable**.

$$
\text{GME} = \text{NOT biseparable}
$$

---

# Recap: Positive Partial Transpose

For a bipartite state $\rho_{AB}$ with matrix elements $\rho_{ij,kl} = \langle i,j | \rho | k,l \rangle$:

$$
\rho^{T_A}_{ij,kl} = \rho_{kj,il}
$$

A state has **Positive Partial Transpose (PPT)** if $\rho^{T_A} \geq 0$.


**Quick Example: Bell state**
$$
\rho = |\phi^+\rangle\langle\phi^+| 
= \frac{1}{2}\begin{pmatrix} 
1 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
1 & 0 & 0 & 1 
\end{pmatrix}
\rightarrow 
\rho^{T_A} = 
\frac{1}{2}\begin{pmatrix} 
1 & 0 & 0 & 0 \\ 
0 & 0 & 1 & 0 \\ 
0 & 1 & 0 & 0 \\ 
0 & 0 & 0 & 1 
\end{pmatrix}
$$

**Eigenvalues of $\rho^{T_A}$:** $\left\{\frac{1}{2}, \frac{1}{2}, \frac{1}{2}, -\frac{1}{2}\right\}$
The **negative eigenvalue** proves the Bell state is entangled!


---

# Strategy: Characterization via entanglement witnesses

<style scoped>
section h1 {
  font-size: 26pt;
}
p, li {
   font-size: 17pt;
   color: #000000;
}
</style>

## Definition
An observable $W$ is an **entanglement witness** for GME if:
- $\text{Tr}(W\rho) \geq 0$ for all biseparable states $\rho$
- $\text{Tr}(W\rho) < 0$ for some GME state $\rho$

### Decomposable Witnesses
A witness is **decomposable** with respect to bipartition $M|\bar{M}$ if:
$$
W = P_M + Q_M^{T_M}
$$
where $P_M \geq 0$ and $Q_M \geq 0$.

A witness $W$ is **fully decomposable** if for **all** bipartitions $M$:
$$
W = P_M + Q_M^{T_M}, \quad P_M \geq 0, \quad Q_M \geq 0
$$



---

# Fully Decomposable Witnesses

<style scoped>

</style>

### Theorem (Main Result)
Fully decomposable witnesses detect **exactly** the states that are **not PPT mixtures**.
$$
\text{
If $\rho$ is not a PPT mixture, then there exists a fully decomposable witness $W$ that detects $\rho$.}
$$
$$
\rho \notin \mathcal{S}_{\text{PPT-mix}} \implies \exists \, W \in \mathcal{W}_{\text{fd}} : \operatorname{Tr}(W\rho) < 0
$$
where 
- $\mathcal{S}_{\text{PPT-mix}}$ = set of PPT mixtures
- $\mathcal{W}_{\text{fd}}$ = set of fully decomposable witnesses
- $\text{Tr}(W\rho) < 0$ = "W detects ρ"

This suggests that **if we find a fully decomposable $W$ with $\text{Tr}(W\rho) < 0$, the state is GME!**

---

# Example: Fully Decomposable Witnesses

Let's return to Bell state in two-qubit system. There's only one bipartition, $A|B$. The natural candidate for the Bell state is the **projector witness**: $W = \frac{1}{2}I - |\Phi^+\rangle\langle\Phi^+|$.

We need to check if $W = P + Q^{T_A}$ for some $P, Q \geq 0$. 

$W$ is decomposable if and only if $W^{T_A} \geq 0$. Computing the partial transpose:
$$(|\Phi^+\rangle\langle\Phi^+|)^{T_A} = 
\frac{1}{2}
\begin{pmatrix}
 1 & 0 & 0 & 0 \\
 0 & 0 & 1 & 0 \\ 
 0 & 1 & 0 & 0 \\ 
 0 & 0 & 0 & 1 
\end{pmatrix} 
\rightarrow 
W^{T_A} = 
\frac{1}{2}I - (\ket{\Phi^+}\bra{\Phi^+}|)^{T_A} 
= \frac{1}{2}\begin{pmatrix} 0 & 0 & 0 & 0 \\ 0 & 1 & -1 & 0 \\ 0 & -1 & 1 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}$$

The eigenvalues are $\{0, 0, 0, 1\}$. So $W^{T_A} \geq 0$. 

---

# Example: Fully Decomposable Witnesses

We can simply choose:
$$P = 0, \qquad Q = W^{T_A}$$

Trivially, we see $\text{Tr}W =1$, and $P,Q \geq 0$.

Unfortunately, its derivation is non-trivial. 


---

# Strategy: SDP Formulation

<style scoped>


</style>

## Semidefinite Program for GME Detection

$$
\begin{aligned}
\min_{W, P_M, Q_M} \quad & \text{Tr}(W\rho) \\
\text{subject to} \quad & \text{Tr}(W) = 1 \\
& W = P_M + Q_M^{T_M}, P_M \geq 0, Q_M \geq 0\quad \forall M 
\end{aligned}
$$

The free parameters are given by W and the operators $P_M$ for every subset $M$ . If the minimum of $\text{Tr}(W\rho)$ is negative, $\rho$ is not a PPT mixture and hence **GME**. 

The given eqution takes the form of a semidefinite program. The global optimality of an SDP can be certified and the solution can efficiently be computed via interior-point methods.

## Additional Strategy:
- fully PPT witnesses; Let $P_{M} = 0$ so that $W^{T_M}\geq0$.
- Let $O = \{O_1 , \cdots, O_k \}$ be such a set of observables. Then, let $W = \sum_i \lambda_i O_i$.

---

# Numerical Results: White Noise Tolerance

<style scoped>

</style>

Consider mixed states: $\rho(p) = p \cdot \frac{I}{2^N} + (1-p) \cdot |\psi\rangle\langle\psi|$

**Critical noise tolerance $p_{\text{tol}}$**: Maximum $p$ where GME is still detected.

| State $\ket{\psi}$ | PPT mixtures | Previous best | 
|-------|-------------|---------------|
| GHZ₃ | **0.571** | 0.571 | 
| W₃ | **0.521** | 0.421 | 
| Cluster₄ | **0.615** | 0.533 | 
| Dicke₄ | **0.539** | 0.471 | 
| GHZ₄ | **0.500** | 0.429 | 


---

# Strategy: Modified SDP for Quantification

<style scoped>

</style> 

This approach can also be used to quantify GME. If the trace normalization $\text{Tr}(W ) = 1$ is replaced by $0\leq P_M \leq 1$, and $0\leq Q_M \leq 1$, the negative witness expectation value is a multipartite entanglement monotone.
$$
\begin{aligned}
\min_{W, P_M, Q_M} \quad & \text{Tr}(W\rho) \\
\text{subject to} \quad & \text{Tr}(W) = 1 \\
& W = P_M + Q_M^{T_M}, 0 \leq P_M , Q_M \leq I  \quad \forall M
\end{aligned}
$$

### Properties of the Monotone $\mathcal{N}(\rho) = -\min \text{Tr}(W\rho)$

- **Vanishes on biseparable states:** $\mathcal{N}(\rho^{\text{bs}}) = 0$
- **Convex:** $\mathcal{N}(\sum_i p_i \rho_i) \leq \sum_i p_i \mathcal{N}(\rho_i)$
- **LOCC monotone:** Doesn't increase under local operations
- **Reduces to negativity** in the bipartite case


---

# Python Implementation: Overview

<style scoped>
section h1 {
  font-size: 26pt;
}
p, li {
   font-size: 16pt;
   color: #000000;
}
code {
   font-size: 14pt;
   margin: 10px 10px;
}
</style>

## PPTMixer: A Python Package for GME Detection

```python
import numpy as np
import cvxpy as cp

def partial_transpose(rho, dims, axis):
    """Compute partial transpose of density matrix."""
    # Reshape, transpose subsystem, reshape back
    ...

def fdecwit(rho, dims):
    """Find optimal fully decomposable witness via SDP."""
    n = rho.shape[0]
    W = cp.Variable((n, n), hermitian=True)
    # Constraints for all bipartitions M
    constraints = [cp.trace(W) == 1]
    for M in bipartitions:
        P_M = cp.Variable((n, n), PSD=True)
        Q_M = cp.Variable((n, n), PSD=True)
        constraints += [W == P_M + partial_transpose(Q_M, dims, M)]

    prob = cp.Problem(cp.Minimize(cp.trace(W @ rho)), constraints)
    return prob.solve()
```

---

# Python Demo: GHZ and W States

<style scoped>
section h1 {
  font-size: 26pt;
}
p, li {
   font-size: 16pt;
   color: #000000;
}
code {
   font-size: 14pt;
}
</style>

## State Definitions

```python
def ghz_state(n):
    """n-qubit GHZ state: (|00...0⟩ + |11...1⟩)/√2"""
    d = 2**n
    psi = np.zeros(d, dtype=complex)
    psi[0] = psi[-1] = 1/np.sqrt(2)
    return np.outer(psi, psi.conj())

def w_state():
    """3-qubit W state: (|001⟩ + |010⟩ + |100⟩)/√3"""
    psi = np.array([0, 1, 1, 0, 1, 0, 0, 0], dtype=complex) / np.sqrt(3)
    return np.outer(psi, psi.conj())
```


---

# Python Demo: White Noise Tolerance

<style scoped>
section h1 {
  font-size: 26pt;
}
p, li {
   font-size: 16pt;
   color: #000000;
}
</style>

## Noise Model

$$
\rho(p) = p \cdot \frac{\mathbb{1}}{d} + (1-p) \cdot |\psi\rangle\langle\psi|
$$

## Scanning for Critical $p_{\text{tol}}$

```python
def find_noise_tolerance(pure_state, dims, p_values):
    """Find critical noise level where GME is lost."""
    d = pure_state.shape[0]
    noise = np.eye(d) / d

    for p in p_values:
        rho = p * noise + (1-p) * pure_state
        min_val = fdecwit(rho, dims)
        if min_val >= 0:
            return p  # GME lost at this noise level
    return 1.0
```


---

# Python Demo: Entanglement Monotone

<style scoped>
section h1 {
  font-size: 26pt;
}
p, li {
   font-size: 17pt;
   color: #000000;
}
code {
   font-size: 14pt;
}
</style>

## Modified SDP with Bounded Operators

```python
def entmon(rho, dims):
    """Compute entanglement monotone via bounded SDP."""
    n = rho.shape[0]
    W = cp.Variable((n, n), hermitian=True)
    constraints = [cp.trace(W) == 1]

    for M in bipartitions:
        P_M = cp.Variable((n, n), hermitian=True)
        Q_M = cp.Variable((n, n), hermitian=True)
        # Bounded constraints: 0 ≤ P, Q ≤ I
        constraints += [P_M >> 0, P_M << np.eye(n)]
        constraints += [Q_M >> 0, Q_M << np.eye(n)]
        constraints += [W == P_M + partial_transpose(Q_M, dims, M)]

    prob = cp.Problem(cp.Minimize(cp.trace(W @ rho)), constraints)
    return -prob.solve()  # Return positive value as monotone
```

---

# Summary


---

# References

<style scoped>
section h1 {
  font-size: 28pt;
}
p, li {
   font-size: 16pt;
   color: #000000;
}
</style>

## Main Paper
- Jungnitsch, Bastian, Tobias Moroder, and Otfried Gühne. "Taming multiparticle entanglement." Physical review letters 106.19 (2011): 190502.

## Related Works
- Peres, Asher. "Separability criterion for density matrices." Physical Review Letters 77.8 (1996): 1413.
- Horodecki, Michał, Paweł Horodecki, and Ryszard Horodecki. "Separability of n-particle mixed states: necessary and sufficient conditions in terms of linear maps." Physics Letters A 283.1-2 (2001): 1-7.
- Gühne, Otfried, and Géza Tóth. "Entanglement detection." Physics Reports 474.1-6 (2009): 1-75.

## Software
- CVXPY: [https://www.cvxpy.org/](https://www.cvxpy.org/)

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
