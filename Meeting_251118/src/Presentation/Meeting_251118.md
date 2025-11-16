---

title       : Entanglement Detection and Quantification of Quantum Systems
author      : Donghun Jung
# description : Variational methods for entanglement detection and quantification
# keywords    : Quantum Entanglement, VED, Logarithmic Negativity
marp        : true
paginate    : true
theme       : KIST


header-includes:
- \usepackage{braket}
output:
  pdf_document:
    keep_tex: true
style: @import url('https://unpkg.com/tailwindcss@^2/dist/utilities.min.css');

---

<!-- _class: titlepage -->
<!-- backgroundColor: #000000 -->


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
   color: #e2a147;
}
.col-left .title{
   color: white;
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
Entanglement Detection and Quantification of Quantum Systems
</div>

<div class="author">
Donghun Jung
</div>

<div class="date">
18 Nov 2025
</div>

<div class="organization">
Department of Physics, Sungkyunkwan University
<br>
Center for Quantum Technology QuiME Lab, Korea Institute of Science Technology
</div>

</div>

<div class="col-right">
<img src="media/images/KIST_CI.png" style="max-width: 100%; height: auto; object-fit: contain;">
</div>

</div>


---

<!-- backgroundColor: white -->


# Outline

- Motivation
- Problem Definition
  - Separability vs. Entanglement
  - Positive but Not Completely Positive (PNCP) Maps
  - Logarithmic Negativity
- Variational Entanglement Detection (VED)
- Examples of Positive Maps
- Variational Entanglement Quantification
- Outlook

---

# Motivation

<style>
.container{
   display: flex;
   align-items: center;
   width: 100%;
   height: 100%;
}
.col-left-content{
   flex: 0 0 100%;
   padding-right: 2rem;
   padding-bottom: 3rem;
   color: #000000;
}

.col-right-content{
   flex: 0 0 40%;
   display: flex;
   align-items: center;
   justify-content: center;
   padding-bottom: 5rem;
}

img[alt~="rightside"]{
   position: absolute;
   top: 6.5rem;
   right: 2.5rem;
   width: 12rem;
}

em {
   font-size: 0.7rem;
}

</style>


**Challenge:** Quantum entanglement is vital for quantum computing, yet its detection and quantification remains challenging.

**Existing Methods:**
- Bell inequalities, CGLMP tests require prior knowledge of target state
- Quantum State Tomography (QST) exponential scaling with system size
- Detection & quantification computationally expensive

**Suggested Approach:**
- Use **strategically sampled measurement bases** to avoid exponential cost
- Extract statistical correlations without complete state information
- **Hybrid variational method**: Quantum state preparation + Quantum Measurement + classical optimization


---

# Separability and Entanglement


A bipartite state $\rho$ is **separable** if:
$$
\rho = \sum_{i} p_i \rho^{A}_{i} \otimes \rho^{B}_{i}
$$
where $\sum_i p_i = 1$ and $p_i > 0$.

Otherwise, the state is **entangled**.

**Problem:** Directly determining separability is **NP-hard**!


**Key Insight:**
Instead of direct verification, use **positive but not completely positive (PNCP) maps** to detect entanglement through negative eigenvalues.


---

# Positive but Not Completely Positive Maps

**PNCP Map** $\mathcal{N}$ can produce negative eigenvalues when applied to entangled states:
$$
\mathcal{N} (\cdot) = \sum_{\mathcal{O}} r_{\mathcal{O}}\mathcal{O}(\cdot)
$$
where $\mathcal{O}$ are operator basis elements (e.g., Pauli operators) and some $r_{\mathcal{O}} < 0$.

---

# Example: Partial Transpose
Any bipartite system can be written as:
$$
\rho_{AB} = \sum_{ijkl} \alpha_{ijkl} \ket{i_{A}}\bra{j_{A}} \otimes \ket{k_{B}}\bra{l_{B}}
$$
Under partial transpose on subsystem B:
$$
\begin{align}
\rho_{AB}^{T_{B}} &= (I_{A} \otimes T_{B})\rho_{AB} \\
&= \sum_{ijkl} \alpha_{ijkl} \ket{i_{A}}\bra{j_{A}} \otimes (\ket{k_{B}}\bra{l_{B}})^{T} \\
&= \sum_{ijkl} \alpha_{ijkl} \ket{i_{A}}\bra{j_{A}} \otimes \ket{l_{B}}\bra{k_{B}}
\end{align}
$$
If subsystem is one-qubit system, this operation decomposes as:
$$
T(\rho) = \frac{1}{2}(\rho + X\rho X - Y \rho Y + Z \rho Z)
$$


---

# Separable vs. Entangled States

## Separable States
For **separable states**, the partial transpose preserves positivity:
$$
\begin{align}
\rho &= \sum_{i} p_i \rho^{A}_{i} \otimes \rho^{B}_{i} \\
&= \sum_{i} p_i
\left(\sum_a \lambda_i^a \ket{\lambda_i^a}\bra{\lambda_i^a}\right) \otimes
\left(\sum_b \mu_i^b \ket{\mu_i^b}\bra{\mu_i^b}\right) \\
&= \sum_{i,a,b} p_i \lambda_i^a \mu_i^b \ket{\lambda_i^a , \mu_i^b}\bra{\lambda_i^a , \mu_i^b}
\end{align}
$$
Since all coefficients $p_i \lambda_i^a \mu_i^b \geq 0$ and sum to unity, this represents a valid eigendecomposition. Under positive and trace-preserving operations, the eigenspectrum remains positive.

---

# Separable vs. Entangled States

## Entangled States
For **entangled states**, quasi-probability decomposition becomes necessary:
$$
\rho = \sum_i q_i \ket{a_i , b_i}\bra{a_i , b_i}
$$
where $\sum_i q_i = 1$ but some $q_i < 0$. 

The states $\{\ket{a_i ,b_i}\}$ form an overcomplete (non-orthonormal) basis. Under PNCP operations, while the quasi-probability coefficients remain unchanged, the transformation of the overcomplete basis induces negative eigenvalues.


---

# Example: Bell State Under Partial Transpose

Consider the two-qubit Bell state $\ket{\Phi^+} = (\ket{00} + \ket{11})/\sqrt{2}$.
**Density matrix:**
$$
\rho_{AB} = \frac{1}{2}
\begin{pmatrix}
1 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
1 & 0 & 0 & 1
\end{pmatrix} =  \frac{1}{2} ( \ket{0,0}\bra{0,0} +\ket{1,1}\bra{1,1} + \ket{+,+}\bra{+,+} +\ket{-,-}\bra{-,-} - \ket{i,i}\bra{i,i} - \ket{-i, -i}\bra{-i,-i} )
$$

**After partial transpose:**
$$
\rho_{AB}^{T_B} = \frac{1}{2}
\begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1
\end{pmatrix}
= \frac{1}{2}\Big( \ket{0,0}\bra{0,0} +\ket{1,1}\bra{1,1} + \ket{+,+}\bra{+,+} +\ket{-,-}\bra{-,-} - \ket{i,-i}\bra{i,-i} - \ket{-i, i}\bra{-i,i} \Big)
$$
The quasi-probability coefficients remain unchanged, but the transformation of the basis yields eigenvalues $\left\{\frac{1}{2},  \frac{1}{2},  \frac{1}{2},  -\frac{1}{2}\right\}$. The negative eigenvalue serves as a witness to entanglement.

---

# Important Remarks on Partial Transpose

1. **PPT Criterion:** Partial transpose detects entanglement for $2\times2$ and $2\times3$ systems (not all entangled states)

2. **Two-qubit systems:** At most **one** negative eigenvalue

3. **General $m\otimes n$ systems:** At most $(m-1)(n-1)$ negative eigenvalues

**Conclusion:** Finding $\lambda_{\min}$ of $\mathcal{N}(\rho)$ is crucial for entanglement detection

---

# Logarithmic Negativity

For pure states, von Neumann entropy is the standard measure. However, for mixed states, multiple measures exist (Rényi entropy, concurrence, logarithmic negativity), with no single universally accepted standard. This paper employs **logarithmic negativity**, defined as:
$$
E_N (\rho_{AB}) = \ln \left\| \rho_{AB}^{T_B} \right\|
$$
where trace norm: $\left\| A \right\| = \text{Tr} \sqrt{A^{\dagger}A}$ = sum of absolute eigenvalues.

**Properties:**
- Separable states: $\|\rho_{AB}^{T_B}\| = 1 \Rightarrow E_N = 0$
- Entangled states: $\|\rho_{AB}^{T_B}\| > 1 \Rightarrow E_N > 0$

---

# Example (Bell state)
For the Bell state with density matrix, the partial transpose yields:
$$
\rho_{AB}^{T_B} = \frac{1}{2}
\begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1
\end{pmatrix}
$$
with eigenvalues $\{\frac{1}{2}, \frac{1}{2}, \frac{1}{2}, -\frac{1}{2}\}$. The trace norm is $\|\rho_{AB}^{T_B}\| = 2$, yielding logarithmic negativity $E_N = \ln 2$.

---

# Variational Entanglement Detection (VED)

**Goal:** Find minimum eigenvalue of $\mathcal{N}(\rho)$
$$
\lambda_{\min} = \min_{\ket{\psi}} \bra{\psi}\mathcal{N}(\rho)\ket{\psi} = \min_{\ket{\psi}} \sum_{\mathcal{O}} r_{\mathcal{O}} \bra{\psi}\mathcal{O}(\rho)\ket{\psi}
$$

This problem is analogous to the Variational Quantum Eigensolver (VQE):
- **VQE**: Find the minimum eigenvalue of a Hamiltonian
- **VED**: Find the minimum eigenvalue of a transformed state $\mathcal{N}(\rho)$

Following the VQE approach, we prepare parameterized test states and measure expectation values. The minimum eigenvalue satisfies:
$$
\lambda_{\min} = \arg\min_{\ket{\psi(\alpha)}} \sum_{\mathcal{O}} r_{\mathcal{O}} \bra{\psi(\alpha)} \mathcal{O}(\rho) \ket{\psi(\alpha)}
$$

---

# VED: Quantum Implementation

Each expectation value can be computed as:
$$
\begin{align}
\bra{\psi(\alpha)} \mathcal{O} (\rho_{AB}) \ket{\psi(\alpha)} &= \bra{0^{2n}} U^{\dagger}_\alpha\mathcal{O}(\rho_{AB})U_{\alpha}\ket{0^{2n}}\\
&= \text{Tr} \left[U^{\dagger}_{\alpha}\mathcal{O}(\rho_{AB})U_{\alpha}\ket{0^{2n}}\bra{0^{2n}} \right]
\end{align}
$$
This can be implemented on a quantum circuit (see Fig. 1 in the original paper). The number of required circuits equals the number of non-zero coefficients $r_{\mathcal{O}}$ in the map decomposition. The implementation of $U_{\alpha}^{\dagger}$ depends on the chosen circuit ansatz.

This simplified quantum circuit estimates the overlap $\bra{\psi}\mathcal{O}(\rho_{AB})\bra{\psi}$ for a given implementable operation $\mathcal{O}$, where $\ket{\psi} = U_{\alpha} \ket{0^{2N}}$ is the parameterized input state.

**Remark:** Since separable states always have positive eigenspectra under PNCP maps, there are no false positives—if the algorithm detects entanglement, the state is genuinely entangled.

---

# VED Algorithm

1. **Input:** $2n$-qubit quantum state $\rho_{AB}$, decomposition $\mathcal{N} (\cdot) = \sum_{\mathcal{O}} r_{\mathcal{O}}\mathcal{O}(\cdot)$ of the PNCP map, parameterized quantum circuit $U_{\alpha}$ with initial parameters $\alpha$, and tolerance $\delta$
2. Initialize loss function $L(\alpha) = 0$
3. **for all** $\mathcal{O}$ with $r_{\mathcal{O}} \neq 0$ **do**
   - Apply $U(\alpha)$ to $\ket{0^{2n}}$ to obtain test state $\ket{\psi(\alpha)}$
   - Input $\rho_{AB}$ and compute the overlap using the quantum circuit
   - Update the loss function: $L(\alpha) \leftarrow L(\alpha) + r_{\mathcal{O}} \bra{\psi(\alpha)}\mathcal{O}(\rho_{AB})\ket{\psi(\alpha)}$
4. **end for**
5. Perform classical optimization to minimize $L(\alpha)$; terminate when $L(\alpha) < -\delta$
6. **Output** "Entangled" if the optimized $L(\alpha) < -\delta$

**Complexity:** This algorithm requires at most $4^N$ circuit copies. The authors also propose Probabilistic VED, which uses probabilistic sampling of observables with a cutoff to reduce measurement overhead while maintaining estimation accuracy within $\delta$ and success probability above $1-\epsilon$.

---

# Examples of Positive Maps
### 1. Partial Transpose (PPT)
$$
T_{B} = \bigotimes_{i=1}^{n} T_{B_i}
$$
### 2. Reduction Map
$$
\rho_{AB} \rightarrow \sigma_{AB} = I_{A} \otimes \rho_B - \rho_{AB}
$$
### 3. Enhanced Reduction Map
$$
\rho_{B} \rightarrow R_{B} (\rho_{B}) - V T_{B}(\rho_{B}) V^{\dagger}
$$
where $V= X\otimes X\otimes \cdots \otimes X \otimes iY$

---

# Reduction Map: Pauli Decomposition

The reduction map is defined as:
$$
\rho_{AB} \rightarrow \sigma_{AB} =  I_{A} \otimes \rho_B - \rho_{AB}
$$
For separable states, this map preserves positivity:
$$
\begin{align}
\rho &= \sum_{i} p_i \rho^{A}_{i} \otimes \rho^{B}_{i} \\
\rightarrow\sigma_{AB} &= \sum_{i} p_i (I - \rho^{A}_{i}) \otimes \rho^{B}_{i} \\
&= \sum_{i} p_i
\left(\sum_a (1-\lambda_i^a) \ket{\lambda_i^a}\bra{\lambda_i^a}\right) \otimes
\left(\sum_b \mu_i^b \ket{\mu_i^b}\bra{\mu_i^b}\right)
\end{align}
$$
Since $0 \leq \lambda_i^a \leq 1$, we have $1-\lambda_i^a \geq 0$, ensuring positivity.

---

# Reduction Map: Pauli Decomposition


The reduction map decomposes in the Pauli basis as:
$$
\begin{align}
R_{B} (\rho_{B}) &= \text{Tr}(\rho_{B}) I_{B} - \rho_{B} \\
&= \frac{1}{2^N} \sum_{P} P \rho_{B} P^{\dagger} I_B - \rho_{B} \\
&= \frac{1-2^N}{2^N}\rho_{B} + \frac{1}{2^N}\sum_{P\neq I^{\otimes N}} P \rho_{B} P^{\dagger}
\end{align}
$$
For a single qubit ($N=1$):
$$
\rho \rightarrow \frac{-\rho + X\rho X + Y \rho Y + Z\rho Z }{2}
$$

---

# Variational Entanglement Quantification

We can measure logarithmic negativity using ancillary qubit

$$
\begin{align}
\left\| \rho_{AB}^{T_{B}} \right\| &= 2\max_{U} \text{Tr}\left[ \ket{0}\bra{0}_{R} Q_R\right] - \text{Tr}(\rho_{AB}^{T_{B}}) \\
&= 2\max_{U} \text{Tr}\left[ \ket{0}\bra{0}_{R} Q_R\right] - 1
\end{align}
$$
where:
- $Q_R = \text{Tr}_{AB}[Q_{ABR}]$
- $Q_{ABR} = U (\rho_{AB}^{T_{B}} \otimes \ket{0}\bra{0}_{R}) U^{\dagger}$

This formulation enables variational optimization similar to VED. The authors note that this framework can be extended to other entanglement measures based on sandwiched Rényi relative entropy.

---

# Outlook



---

# Summary




---

# References & Further Reading

**Main Paper:** [To be filled with actual paper reference]

**Related Work:**
- PPT Criterion: Peres (1996), Horodecki et al. (1996)
- Logarithmic Negativity: Plenio (2005)
- VQE: Peruzzo et al. (2014)

**Suggested Reading:**
- Nielsen & Chuang, "Quantum Computation and Quantum Information"
- Horodecki et al., "Quantum entanglement" Rev. Mod. Phys. (2009)
