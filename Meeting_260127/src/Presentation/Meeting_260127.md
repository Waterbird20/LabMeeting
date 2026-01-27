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
   font-size: 32pt;
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
- Efficient criterion for **genuine multipartite entanglement (GME)** via **PPT mixtures**
- **Semidefinite programming (SDP)** formulation for practical detection
- **Fully decomposable entanglement witnesses**
- **Entanglement Quantification**

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
}

.col-right-content{
   margin-left: -100px;
   flex: 0 0 50%;
   display: flex;
   align-items: center;
   justify-content: center;
   padding-bottom: 3rem;
}

li {
   font-size: 0.85rem;
}

</style>

<div class="container">
<div class="col-left-content">

1. **Motivation**
   - Separability & entanglement
   - Examples of multipartite states
   - Challenges in detection

2. **Definitions**
   - Biseparable states
   - Genuine multipartite entanglement (GME)

</div>

<div class="col-right-content">

3. **Main Idea**
   - PPT mixtures
   - Fully decomposable witnesses
   - SDP formulation

4. **Results**
   - White noise tolerance
   - Entanglement monotone

5. **Summary**

</div>
</div>

---

# Motivation

<style scoped>

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

**The Challenge:** How do we efficiently verify that a state has *genuine* multipartite entanglement that cannot be reduced to bipartite states?

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
- How should we split the system into subsystems?
- It is a costly operation!

---

# [Motivation] Examples of Multiparticle Entangled States

<style scoped>
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

# [Motivation] Verification of Entangled States

For pure states, SVD is the most effective way to quantify entanglement. For example, consider the two-qubit Bell state (though this is a trivial example).

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
**Step 4.** Two Schmidt coefficients $\Rightarrow$ entangled state. Try the other Bell states!
 

---

# [Motivation] Challenges for Entanglement Detection: Pure States

Exponential cost is inevitable. For an $m \times n$ matrix:
| Resource | Complexity |
|----------|------------|
| **Time** | $\mathcal{O}(\min(mn^2, m^2n))$ $\simeq$ $\mathcal{O}(mn \cdot \min(m,n))$ |
| **Memory** | $\mathcal{O}(mn)$ for matrix + $\mathcal{O}(m^2 + n^2)$ for $U, V$ |

For $N$ qubits with bipartition $k | (N-k)$, the matrix size becomes $2^k \times 2^{N-k}$. Things get worse if we must try every possible bipartition, of which there are $2^{N-1}$. 


---

# [Motivation] Challenges for Entanglement Detection: Mixed States


Even worse, for mixed states there is no "canonical form" like the Schmidt decomposition. The same $\rho$ can be written as:
$$\rho = \sum_k p_k |\psi_k\rangle\langle\psi_k| = \sum_j q_j |\phi_j\rangle\langle\phi_j|$$
with completely different ensembles! One might use only product states (separable), another might not (entangled representation), while the **physical state is the same**.


---

# [Definition] Biseparable States, Genuine Multipartite Entanglement

<style scoped>
p {
   font-size: 20pt;
}
</style>

A state is separable with respect to the bipartition $M|\bar{M}$ if:
$$
\rho^{\text{sep}}_{M|\bar{M}} = \sum_i p_i \, \rho_M^{(i)} \otimes \rho_{\bar{M}}^{(i)}
$$
A state is **biseparable** if it can be written as a mixture:
$$
\rho^{\text{bs}} = \sum_{i} p_i \rho^{\text{sep}}_{M_{i}|\bar{M}_{i}}
$$
where $\sum_i p_i = 1$. Note that a non-trivial form (with at least two $p_i\neq 0$) must be a mixed state.

### Genuine Multipartite Entanglement (GME)
A state is **genuinely multipartite entangled** if and only if it is **NOT biseparable**.

$$
\text{GME} = \text{NOT biseparable}
$$

---

# [Example] Biseparable States

Consider a three-qubit system (labeled A, B, C). There are three possible bipartitions: $A|BC$, $B|CA$, and $C|AB$, i.e., $M|\bar{M}\in\{A|BC , B|CA, C|AB\}$. A biseparable state can be written as:
$$
\rho^{\text{bs}} = p_{A|BC} \rho^{\text{sep}}_{A|BC} + p_{B|CA} \rho^{\text{sep}}_{B|CA} + p_{C|AB} \rho^{\text{sep}}_{C|AB}
$$

---

# [Main Idea] Checking whether PPT mixture exists

It is well-known that separable states have Positive Partial Transpose (PPT). We denote such PPT states by $\rho^{\text{ppt}}_{M|\bar{M}}$. Thus, we ask whether a state can be written as
$$
\rho^{\text{pmix}} = \sum_{i} p_i \rho^{\text{ppt}}_{M_{i}|\bar{M}_{i}}
$$
We call states of this form **PPT mixtures**.

Clearly, any biseparable state is a PPT mixture, so proving that a state is *not* a PPT mixture implies **genuine multipartite entanglement**. 



---

# [Recap] Positive Partial Transpose

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

# [Main Idea] Characterization via entanglement witnesses

<style scoped>

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

# [Main Idea] Fully Decomposable Witnesses

<style scoped>

</style>

### Theorem
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
- $\text{Tr}(W\rho) < 0$ = "W detects $\rho$"

This suggests that **if we find a fully decomposable $W$ with $\text{Tr}(W\rho) < 0$, the state is GME!**

---

# [Example] Fully Decomposable Witnesses

Let's return to the Bell state in a two-qubit system. There is only one bipartition, $A|B$. A natural candidate witness for the Bell state is the **projector witness**: $W = \frac{1}{2}I - \ket{\Phi^+}\bra{\Phi^+}$.

$$W = \frac{1}{2}I - \ket{\Phi^+}\bra{\Phi^+}= 
\frac{1}{2}
\begin{pmatrix}
 0 & 0 & 0 & -1 \\
 0 & 1 & 0 & 0 \\ 
 0 & 0 & 1 & 0 \\ 
-1 & 0 & 0 & 0 
\end{pmatrix} 
$$

This entanglement witness detects the Bell state: 

$$
\text{Tr} W\ket{\Phi^+}\bra{\Phi^+} = -\frac{1}{2}
$$

---

# [Example] Fully Decomposable Witnesses

We need to check if $W = P + Q^{T_A}$ for some $P, Q \geq 0$. 

We can simply choose:
$$P = 0, \qquad Q = W^{T_A} = \frac{1}{2}I - (\ket{\Phi^+}\bra{\Phi^+})^{T_A} 
= \frac{1}{2}\begin{pmatrix} 0 & 0 & 0 & 0 \\ 0 & 1 & -1 & 0 \\ 0 & -1 & 1 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}$$

One can verify that $\text{Tr}(W) = 1$, and $P, Q \geq 0$, confirming $W$ is fully decomposable.

In general, finding such decompositions is non-trivial. 

---

# [Strategy] SDP Formulation

<style scoped>
p, li {
   font-size: 18pt;

}
</style>

## Semidefinite Program for GME Detection

$$
\begin{aligned}
\min_{W, P_M, Q_M} \quad & \text{Tr}(W\rho) \\
\text{subject to} \quad & \text{Tr}(W) = 1 \\
& W = P_M + Q_M^{T_M}, P_M \geq 0, Q_M \geq 0\quad \forall M 
\end{aligned}
$$

The free parameters are $W$ and the operators $P_M$, $Q_M$ for every bipartition $M$. If the minimum of $\text{Tr}(W\rho)$ is negative, then $\rho$ is not a PPT mixture and hence is **GME**.

This optimization takes the form of a semidefinite program (SDP). The global optimality of an SDP can be certified and the solution can be efficiently computed via interior-point methods.

## Additional Strategies
- **Fully PPT witnesses:** Set $P_{M} = 0$ so that $W^{T_M}\geq0$ for all $M$.
- **Observable basis:** Let $O = \{O_1 , \cdots, O_k \}$ be a set of observables. Then parameterize $W = \sum_i \lambda_i O_i$.

---

# [Results] White Noise Tolerance

<style scoped>
table {
   align: center;
   margin: 0rem 15rem;
}
</style>

Consider mixed states(isotropic state): $\rho(p) = p \cdot \frac{I}{2^N} + (1-p) \cdot |\psi\rangle\langle\psi|$

**Critical noise tolerance $p_{\text{tol}}$**: Maximum $p$ for which GME is still detected. A higher $p_{\text{tol}}$ indicates stronger entanglement. 



| State $\ket{\psi}$ | PPT mixtures | 
|-------|-------------|
| GHZ₃ | **0.571** | 
| W₃ | **0.521** |
| Cluster₄ | **0.615** | 
| Dicke₄ | **0.539** | 
| GHZ₄ | **0.500** |

---

# [Numerical Simulation] White Noise Tolerance

![width:1500px](Meeting_260127/src/Presentation/media/images/Numerical_Result.svg)

---

# [Strategy] Modified SDP for Quantifying GME

<style scoped>
p {
   font-size: 18pt;
}
</style>

This approach can also be used to quantify GME. If the trace normalization $\text{Tr}(W) = 1$ is replaced by $0\leq P_M \leq I$ and $0\leq Q_M \leq I$, the negative witness expectation value becomes a multipartite entanglement monotone. Note that this quantity equals the **negativity** in the bipartite case. 
$$
\begin{aligned}
-\min_{W, P_M, Q_M} \quad & \text{Tr}(W\rho) \\
\text{subject to} \quad & \text{Tr}(W) = 1 \\
& W = P_M + Q_M^{T_M}, 0 \leq P_M , Q_M \leq I  \quad \forall M
\end{aligned}
$$

### Properties of the Monotone $\mathcal{N}(\rho) = -\min \text{Tr}(W\rho)$

- **Vanishes on biseparable states:** $\mathcal{N}(\rho^{\text{bs}}) = 0$
- **Convex:** $\mathcal{N}(\sum_i p_i \rho_i) \leq \sum_i p_i \mathcal{N}(\rho_i)$
- **LOCC monotone:** Does not increase under local operations and classical communication

---

# [Numerical Result] $\mathcal{N}(\rho)$ on the two-qubit isotropic state

![width:1500px](Meeting_260127/src/Presentation/media/images/Bell_Entanglement_vs_Noise.svg)

---

# [Numerical Result] Quantifying GME

![width:1500px](Meeting_260127/src/Presentation/media/images/Entanglement_Monotone_Scaling.svg)

---

# Summary

## Pros
- The framework is systematic and complete.
- It provides a unified SDP-based approach for detecting GME.
- No false negatives within the PPT mixture class.
- The entanglement monotone not only detects but also quantifies GME.

---

# Summary

## Cons

- Generally needs full density matrix $\rho$ — expensive quantum state tomography required
- No Closed-Form Solution. SDP optimization (convex optimization) is required for each state.
   - Time Complexity: $\mathcal{O}(\sqrt{m} D^3 (m+D^2))$
   - Space Complexity: $\mathcal{O}(mD^2 + D^4)$

   where $D$ is the Hilbert space dimension and $m$ is the number of constraints. 
- Not Complete for All Entanglement. Only detects states outside PPT mixtures; PPT entangled states are missed


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
