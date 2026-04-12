---

title       : "Entanglement: PT-Moments and Geometric Measure"
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
Entanglement: PT-Moments and Geometric Measure
</div>

<div class="author">
Donghun Jung
</div>

<div class="date">
14 Apr 2026
</div>

<div class="organization">
Department of Physics, Sungkyunkwan University
<br>
QuiME Lab, Center for Quantum Technology, Korea Institute of Science Technology
</div>

</div>

<div class="col-right">
<img src="media/images/PauleeLogo.png" style="max-width: 100%; height: 100%; object-fit: contain;">
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
   font-size: 18pt;
   color: #000000;
}
</style>

## Paper 1
**Optimal Entanglement Certification from Moments of the Partial Transpose**
**Authors:** Xiao-Dong Yu, Satoya Imai, and Otfried Gühne
**Journal:** Physical Review Letters **127**, 060504 (2021)

## Paper 2
**Quantifying Entanglement from the Geometric Perspective**
**Authors:** Lisa T. Weinbrenner and Otfried Gühne
**Journal:** EPL **151**, 68001 (2025)

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

1. **Review**
   - Separability & entanglement
   - Partial transpose (PPT)
   - Negativity
   - Entanglement witnesses
   - Multipartite entanglement

2. **Paper 1: PT-Moment Certification**
   - PT-moments & measurement
   - Classical moment problems
   - Hankel matrix criteria
   - Optimal $p_n$-OPPT criterion

</div>

<div class="col-right-content">

3. **Paper 2: Geometric Measure**
   - Definition & properties
   - Asymptotic behaviour
   - Connection to MBQC
   - Tensor eigenvalues

4. **Summary**

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
Review
</div>

<div class="subtitle">
Entanglement Detection & Quantification
</div>

</div>
</div>

---

# [Review] Separability & Entanglement

<style scoped>
p, li {
   font-size: 20pt;
   color: #000000;
}
</style>

A bipartite state $\rho$ is **separable** if it can be written as:
$$
\rho = \sum_{i} p_i \, \rho^{A}_{i} \otimes \rho^{B}_{i}, \quad \sum_i p_i = 1, \quad p_i > 0
$$

Otherwise, the state is **entangled**.

**Problem:** Directly determining separability is **NP-hard**.

### Approaches to detect entanglement
- **PNCP maps** (e.g., partial transpose): produce negative eigenvalues for entangled states
- **Entanglement witnesses**: observables with non-negative expectation on all separable states
- **Entanglement measures**: quantify the degree of entanglement (negativity, geometric measure, ...)

---

# [Review] Partial Transpose & PPT Criterion

<style scoped>
p, li {
   font-size: 18pt;
   color: #000000;
}
</style>

For a bipartite state $\rho_{AB} = \sum_{ijkl} \alpha_{ijkl} \ket{i_A}\bra{j_A} \otimes \ket{k_B}\bra{l_B}$, the **partial transpose** on $B$ is:
$$
\rho_{AB}^{T_{B}} = \sum_{ijkl} \alpha_{ijkl} \ket{i_{A}}\bra{j_{A}} \otimes \ket{l_{B}}\bra{k_{B}}
$$

For a single-qubit subsystem, this decomposes as:
$$
T(\rho) = \frac{1}{2}(\rho + X\rho X - Y \rho Y + Z \rho Z)
$$

### PPT Criterion (Peres–Horodecki)
- **Separable** $\Rightarrow$ $\rho^{T_B} \geq 0$ (positive partial transpose)
- **PPT is necessary and sufficient** for $2\times2$ and $2\times3$ systems
- In higher dimensions, PPT entangled (bound entangled) states exist

---

# [Review] Negativity & Logarithmic Negativity

<style scoped>
p, li {
   font-size: 20pt;
   color: #000000;
}
</style>

The **negativity** quantifies the violation of PPT:
$$
\mathcal{N}(\rho) = \frac{\left\| \rho^{T_B} \right\| - 1}{2} = \sum_{\lambda_i < 0} |\lambda_i|
$$
where $\left\| A \right\| = \text{Tr}\sqrt{A^{\dagger}A}$ is the trace norm.

The **logarithmic negativity** is defined as:
$$
E_N(\rho_{AB}) = \ln \left\| \rho_{AB}^{T_B} \right\|
$$

**Properties:**
- Separable states: $\left\| \rho^{T_B} \right\| = 1 \Rightarrow E_N = 0$
- Entangled states (detected by PPT): $\left\| \rho^{T_B} \right\| > 1 \Rightarrow E_N > 0$
- For the Bell state: eigenvalues of $\rho^{T_B}$ are $\{\frac{1}{2}, \frac{1}{2}, \frac{1}{2}, -\frac{1}{2}\}$, giving $E_N = \ln 2$

---

# [Review] Entanglement Witnesses

<style scoped>
p, li {
   font-size: 18pt;
   color: #000000;
}
</style>

An observable $W$ is an **entanglement witness** if:
- $\text{Tr}(W\rho) \geq 0$ for all separable states $\rho$
- $\text{Tr}(W\rho) < 0$ for some entangled state $\rho$

### Decomposable Witnesses
A witness $W$ is **decomposable** w.r.t. bipartition $M|\bar{M}$ if:
$$
W = P_M + Q_M^{T_M}, \quad P_M \geq 0, \quad Q_M \geq 0
$$

A witness is **fully decomposable** if this holds for **all** bipartitions $M$.

### Key theorem
Fully decomposable witnesses detect **exactly** the states that are **not PPT mixtures**:
$$
\rho \notin \mathcal{S}_{\text{PPT-mix}} \implies \exists \, W \in \mathcal{W}_{\text{fd}} : \text{Tr}(W\rho) < 0
$$

---

# [Review] Multipartite Entanglement

<style scoped>
p, li {
   font-size: 18pt;
   color: #000000;
}
</style>

A state is **biseparable** if it can be written as:
$$
\rho^{\text{bs}} = \sum_{i} p_i \, \rho^{\text{sep}}_{M_{i}|\bar{M}_{i}}, \quad \sum_i p_i = 1
$$
A state has **genuine multipartite entanglement (GME)** iff it is NOT biseparable.

### GME Detection via SDP
$$
\begin{aligned}
\min_{W, P_M, Q_M} \quad & \text{Tr}(W\rho) \\
\text{subject to} \quad & \text{Tr}(W) = 1 \\
& W = P_M + Q_M^{T_M}, \quad P_M \geq 0, \quad Q_M \geq 0 \quad \forall M
\end{aligned}
$$
If $\min \text{Tr}(W\rho) < 0$, the state is **GME**.

### Limitation
All these methods generally require **full density matrix** $\rho$ — expensive quantum state tomography.

---

# [Review] Summary

<style scoped>
table {
   font-size: 16pt;
   margin: 0 auto;
}
p {
   font-size: 18pt;
}
</style>

| Method | Detects | Requires | Limitation |
|--------|---------|----------|------------|
| **PPT** | Entanglement (NPT states) | $\rho^{T_B}$ spectrum | Misses PPT entangled states |
| **Negativity** | + Quantifies | $\rho^{T_B}$ eigenvalues | Not a full measure |
| **Witnesses** | Tailored detection | Observable $W$ | Must construct $W$ per state |
| **SDP (GME)** | Genuine multipartite | Full $\rho$ | Exponential scaling |

**Question:** Can we certify entanglement **without full state tomography**?

$\Rightarrow$ **Paper 1:** Use only the *moments* of $\rho^{T_A}$ — accessible via randomized measurements.

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
   font-size: 40pt;
   margin-bottom: 40px;
}
.center-content .subtitle {
   font-size: 22pt;
   color: #666;
}
</style>

<div class="container">
<div class="center-content">

<div class="title">
Optimal Entanglement Certification from Moments of the Partial Transpose
</div>

<div class="subtitle">
Yu, Imai, and Gühne (PRL 2021)
</div>

</div>
</div>

---

# Motivation: Why PT-Moments?

<style scoped>
p, li {
   font-size: 18pt;
   color: #000000;
}
</style>

**Challenge:** Full state tomography scales exponentially. Can we detect entanglement from **partial information**?

### PT-Moments
The $n$-th moment of the partially transposed state is:
$$
p_n = \text{Tr}\!\left[(\rho_{AB}^{T_A})^n\right]
$$

**Key advantage:** PT-moments can be efficiently measured via **randomized measurements** (classical shadows, $U$-statistics) without reconstructing $\rho$.

### Measurement cost
For $N$-qubit states with bipartition $A|B$, the number of copies needed scales as:
$$
M \sim \frac{n^2 \, 2^N \, p_2^{n-1}}{\varepsilon^2 \, \delta}
$$
Higher-order moments ($p_n$ for large $n$) require only marginally more measurements since $p_n \leq p_2^{n/2}$.

---

# PT-Moments: Definition and Properties

<style scoped>
p, li {
   font-size: 18pt;
   color: #000000;
}
</style>

Given a bipartite state $\rho_{AB}$ on $\mathcal{H}_A \otimes \mathcal{H}_B$ with $\dim(\mathcal{H}_A \otimes \mathcal{H}_B) = d$, the **PT-moments** are:
$$
p_n = \text{Tr}\!\left[(\rho_{AB}^{T_A})^n\right], \quad n = 0, 1, 2, \ldots
$$

### Basic properties
- $p_0 = d$ (dimension)
- $p_1 = \text{Tr}[\rho_{AB}^{T_A}] = \text{Tr}[\rho_{AB}] = 1$
- $p_2 = \text{Tr}[(\rho_{AB}^{T_A})^2] = \text{Tr}[\rho_{AB}^2]$ (purity), so $\frac{1}{d} \leq p_2 \leq 1$
- For **separable** states: $\rho_{AB}^{T_A} \geq 0$, so $p_n = \text{Tr}[(\rho_{AB}^{T_A})^n] \geq 0$ for all $n$, and the spectrum $(x_1, \ldots, x_d)$ satisfies $x_i \geq 0$

### The PT-Moment Problem
Given moments $(p_1, p_2, \ldots, p_n)$, is there a **separable state** $\rho_{AB}$ with these PT-moments?

If not $\Rightarrow$ the state **must be entangled**.

---

# Connection to Classical Moment Problems

<style scoped>
p, li {
   font-size: 18pt;
   color: #000000;
}
</style>

Writing the spectrum of $\rho_{AB}^{T_A}$ as $(x_1, x_2, \ldots, x_d)$, we have:
$$
p_n = \sum_{i=1}^{d} x_i^n
$$

This connects the PT-moment problem to **classical moment problems**:

### Hamburger Moment Problem
Given moments $m^{(n)} = (m_0, m_1, \ldots, m_n)$, does there exist a measure on $\mathbb{R}$ with these moments?

### Stieltjes Moment Problem
Same question, but the measure is supported on $[0, \infty)$.

For **separable** states, $x_i \geq 0$ $\Rightarrow$ Stieltjes problem.
For **general** (possibly entangled) states, $x_i \in \mathbb{R}$ $\Rightarrow$ Hamburger problem.

The key mathematical tool for solving both: **Hankel matrices**.

---

# Why Hankel Matrices?

<style scoped>
p, li {
   font-size: 17pt;
   color: #000000;
}
</style>

### Gram matrix construction (Appendix A)
Use the **Hilbert-Schmidt inner product** $(X, Y) := \text{Tr}(X^\dagger Y)$ in operator space.

Given $\rho \geq 0$ and $X \geq 0$ satisfying $m_k = \text{Tr}(\rho X^k) = \braket{\varphi | X^k | \varphi}$, construct the operator sequences:
$$
v = \left(\rho^{1/2},\; \rho^{1/2} X,\; \ldots,\; \rho^{1/2} X^{\lfloor n/2 \rfloor}\right), \quad u = \left(\rho^{1/2} X^{1/2},\; \rho^{1/2} X^{3/2},\; \ldots,\; \rho^{1/2} X^{\lfloor \frac{n-1}{2} \rfloor + 1/2}\right)
$$

Their **Gram matrices** are:
$$
(v_i, v_j) = \text{Tr}(X^i \rho \, X^j) = \text{Tr}(\rho \, X^{i+j}) = m_{i+j} \quad \Rightarrow \quad [H_k]_{ij} = m_{i+j}
$$
$$
(u_i, u_j) = \text{Tr}(X^{i+1/2} \rho \, X^{j+1/2}) = m_{i+j+1} \quad \Rightarrow \quad [B_k]_{ij} = m_{i+j+1}
$$

Since **Gram matrices are always positive semidefinite**, we get:
$$
\rho \geq 0 \;\Rightarrow\; H_{\lfloor n/2 \rfloor} \geq 0, \qquad \rho \geq 0 \text{ and } X \geq 0 \;\Rightarrow\; B_{\lfloor (n-1)/2 \rfloor} \geq 0
$$

---

# Hankel Matrix Criterion

<style scoped>
p, li {
   font-size: 17pt;
   color: #000000;
}
</style>

Define the **Hankel matrices** from a moment sequence $m^{(n)} = (m_0, m_1, \ldots, m_n)$:
$$
H_k = \begin{pmatrix} m_0 & m_1 & \cdots & m_k \\ m_1 & m_2 & \cdots & m_{k+1} \\ \vdots & \vdots & \ddots & \vdots \\ m_k & m_{k+1} & \cdots & m_{2k} \end{pmatrix}, \quad
B_k = \begin{pmatrix} m_1 & m_2 & \cdots & m_{k+1} \\ m_2 & m_3 & \cdots & m_{k+2} \\ \vdots & \vdots & \ddots & \vdots \\ m_{k+1} & m_{k+2} & \cdots & m_{2k+1} \end{pmatrix}
$$

### Lemma (Classical moment theory)
**(a)** $m^{(n)} \in \mathcal{M}_n$ (Hamburger, measure on $\mathbb{R}$) $\iff$ $H_{\lfloor n/2 \rfloor} \geq 0$

**(b)** $m^{(n)} \in \mathcal{M}_n^+$ (Stieltjes, measure on $[0,\infty)$) $\iff$ $H_{\lfloor n/2 \rfloor} \geq 0$ **and** $B_{\lfloor (n-1)/2 \rfloor} \geq 0$

Since separable states have $x_i \geq 0$, the Stieltjes condition applies. Violation of either Hankel condition certifies entanglement.

---

# $p_n$-PPT Criterion

<style scoped>
p, li {
   font-size: 18pt;
   color: #000000;
}
</style>

### Theorem (Entanglement criteria from Hankel matrices)
Let $p_k = \text{Tr}[(\rho_{AB}^{T_A})^k]$ for $k = 1, 2, \ldots, n$. A **necessary condition** for $\rho_{AB}$ being separable is:
$$
H_{\lfloor n/2 \rfloor} \geq 0 \quad \text{and} \quad B_{\lfloor (n-1)/2 \rfloor} \geq 0
$$

### Lowest order: $p_3$-PPT criterion
For $n = 3$, the Hankel condition $H_1 \geq 0$ gives:
$$
H_1 = \begin{pmatrix} p_1 & p_2 \\ p_2 & p_3 \end{pmatrix} = \begin{pmatrix} 1 & p_2 \\ p_2 & p_3 \end{pmatrix} \geq 0
$$
This requires $\det(H_1) \geq 0$, yielding the **$p_3$-PPT criterion**:
$$
\boxed{p_3 \geq p_2^2}
$$

If this is violated, the state is **entangled**. Higher orders ($n = 5, 7, \ldots$) give strictly stronger criteria.

---

# Optimal Solution to the PT-Moment Problem

<style scoped>
p, li {
   font-size: 17pt;
   color: #000000;
}
</style>

The Hankel criteria are **not optimal** — they only use the positive semidefiniteness of Hankel matrices, ignoring the constraint that the spectrum $\boldsymbol{x} = (x_1, \ldots, x_d)$ must be **finite-dimensional** ($d$ eigenvalues).

### Exact optimization formulation
For fixed $(p_1, p_2, \ldots, p_{n-1})$, find the range of $p_n$ compatible with a separable state:
$$
\begin{aligned}
\min_{x_i} / \max_{x_i} \quad & \hat{p}_n := \sum_{i=1}^{d} x_i^n \\
\text{subject to} \quad & \sum_{i=1}^{d} x_i^k = p_k, \quad k = 1, 2, \ldots, n-1 \\
& x_i \geq 0 \quad \text{for } i = 1, 2, \ldots, d
\end{aligned}
$$

The solutions are found **analytically** using Cramer's rule and the Vandermonde determinant. The extremal spectra have a specific structure with at most **three distinct values**.

---

# $p_3$-OPPT Criterion (Theorem 3)

<style scoped>
p, li {
   font-size: 16pt;
   color: #000000;
}
</style>

### Theorem (Optimal $p_3$-OPPT)
**(a)** There exists a $d$-dimensional **separable (PPT)** state $\rho_{AB}$ satisfying $p_k = \text{Tr}[(\rho_{AB}^{T_A})^k]$ for $k = 1, 2, 3$ if and only if:
$$
p_1 = 1, \quad \frac{1}{d} \leq p_2 \leq 1
$$
$$
\alpha x^3 + (1 - \alpha x)^3 \leq p_3 \leq [1 - (d-1)y]^3 + (d-1)y^3
$$

where:
$$
\alpha = \left\lfloor \frac{1}{p_2} \right\rfloor, \quad x = \frac{\alpha + \sqrt{\alpha[p_2(\alpha+1) - 1]}}{\alpha(\alpha + 1)}, \quad y = \frac{d - 1 - \sqrt{(d-1)(p_2 d - 1)}}{d(d-1)}
$$

**(b)** If the $p_k$ for $k = 1, 2, 3$ are PT-moments from a **quantum state**, they are compatible with a separable state if and only if:
$$
\boxed{p_3 \geq \alpha x^3 + (1 - \alpha x)^3}
$$

The $p_3$-OPPT criterion is **dimension-independent** and strictly stronger than $p_3$-PPT.

---

# Theorem 3(b): Quantum Refinement

<style scoped>
p, li {
   font-size: 18pt;
   color: #000000;
}
</style>

### Why is the lower bound sufficient?
For PT-moments from a quantum state, we always have:
- $p_1 = \text{Tr}[\rho_{AB}^{T_A}] = 1$
- $p_2 = \text{Tr}[\rho_{AB}^2]$ (purity), satisfying $1/d \leq p_2 \leq 1$

The **upper bound** on $p_3$ is automatically satisfied by any (separable or entangled) state. Thus only the **lower bound** matters for entanglement detection:
$$
p_3 \geq \alpha x^3 + (1 - \alpha x)^3
$$

### Remarks
- When $\alpha = 1$ (i.e., $p_2 \geq 1/2$): criterion reduces to $p_3 \geq p_2^2$, recovering $p_3$-PPT
- When $\alpha \geq 2$ (i.e., $p_2 < 1/2$, high-dimensional or mixed states): the optimal criterion is **strictly stronger**
- The criterion is **dimension-independent**: the same formula works for any $d$
- Violation up to **12.5% larger** than $p_3$-PPT

---

# Comparison: $p_3$-PPT vs $p_3$-OPPT

<style scoped>
p, li {
   font-size: 18pt;
   color: #000000;
}
</style>

### Hierarchical structure
The criteria form a hierarchy based on PT-moments:
- **$p_3$-PPT** ($n = 3$, Hankel): $p_3 \geq p_2^2$
- **$p_3$-OPPT** ($n = 3$, optimal): $p_3 \geq \alpha x^3 + (1-\alpha x)^3$
- **$p_n$-PPT** ($n = 3, 5, 7, \ldots$): higher-order Hankel conditions
- **$p_n$-OPPT** ($n = 3, 5, 7, \ldots$): higher-order optimal conditions

### Key distinctions
1. $p_3$-PPT and $p_3$-OPPT **coincide** when $\alpha = 1$ (i.e., $p_2 \geq 1/2$)
2. For $p_2 < 1/2$, the optimal criterion detects **more** entangled states
3. Higher-order Hankel criteria ($p_n^{n-2} \geq p_{n-1}^{n-1}$) are **strictly weaker** than the Hankel matrix conditions
4. The optimal criteria at each order are the **best possible** given the available moments

---

# Detection Power

<style scoped>
p, li {
   font-size: 17pt;
   color: #000000;
}
table {
   font-size: 15pt;
   margin: 0 auto;
}
</style>

### Fraction of entangled $D \times D$ states detected (Hilbert–Schmidt distribution)

| $D$ | NPT | NPT3 | NPT5 | ONPT3 | ONPT4 | NPT5 |
|-----|------|-------|-------|--------|--------|------|
| 3 | 25.68% | 15.55% | 39.97% | 75.08% | 64.28% | 75.08% |
| 4 | 99.93% | 23.33% | 39.40% | 99.55% | 97.51% | $\approx$ 100% |
| 5 | 100% | 21.80% | 34.94% | 99.99% | $\approx$ 100% | 100% |
| 10 | 100% | 18.54% | 31.25% | 100% | 100% | 100% |

- **NPT**: PPT criterion (full spectrum)
- **NPT$n$**: $p_n$-PPT criterion
- **ONPT$n$**: $p_n$-OPPT criterion (optimal)

The optimal criterion ONPT3 detects **significantly more** states than NPT3, and approaches full NPT detection power. ONPT4 and NPT5 achieve near-complete detection.

---

# Summary: PT-Moment Certification

<style scoped>
p, li {
   font-size: 18pt;
   color: #000000;
}
</style>

### Two systematic methods for entanglement detection from PT-moments:

1. **Hankel matrix method** ($p_n$-PPT): relaxation to classical moment problems
   - Lowest order gives $p_3 \geq p_2^2$
   - Higher orders provide strictly stronger criteria

2. **Optimal method** ($p_n$-OPPT): exact solution of the PT-moment problem
   - Gives necessary and sufficient conditions
   - Detects significantly more entangled states
   - Dimension-independent

### Advantages over conventional methods
- **No full tomography required** — only PT-moments from randomized measurements
- **No prior information** needed (unlike witness-based methods)
- **Efficient measurement scaling** — higher-order moments cost only marginally more

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
   font-size: 40pt;
   margin-bottom: 40px;
}
.center-content .subtitle {
   font-size: 22pt;
   color: #666;
}
</style>

<div class="container">
<div class="center-content">

<div class="title">
Quantifying Entanglement from the Geometric Perspective
</div>

<div class="subtitle">
Weinbrenner and Gühne (EPL 2025)
</div>

</div>
</div>

---

# Geometric Measure of Entanglement

<style scoped>
p, li {
   font-size: 18pt;
   color: #000000;
}
</style>

The **geometric measure** quantifies entanglement by the distance to the nearest separable (product) state.

### Definition
For a pure state $\ket{\psi}$ of $N$ particles, define the **maximal overlap** with product states:
$$
\Lambda^2(\psi) = \max_{\ket{\phi} = \ket{a_1} \otimes \ket{a_2} \otimes \cdots \otimes \ket{a_N}} \left|\braket{\phi | \psi}\right|^2
$$

The **geometric measure of entanglement** is:
$$
E_G(\psi) = 1 - \Lambda^2(\psi)
$$

### Properties
- $E_G = 0$ iff $\ket{\psi}$ is a product state
- $0 \leq E_G \leq 1 - 1/d$ for a $d$-dimensional system
- Monotone under LOCC (local operations and classical communication)

---

# Schmidt Decomposition and $\Lambda$

<style scoped>
p, li {
   font-size: 18pt;
   color: #000000;
}
</style>

### Bipartite case
Any bipartite state can be written in the **Schmidt decomposition**:
$$
\ket{\psi} = \sum_{i=1}^{r} s_i \ket{a_i} \ket{b_i}
$$
where $s_1 \geq s_2 \geq \cdots \geq s_r > 0$ are the **Schmidt coefficients**, $\sum_i s_i^2 = 1$, and $r$ is the **Schmidt rank**.

The maximal overlap is simply the **largest Schmidt coefficient**:
$$
\Lambda(\psi) = s_1
$$

### Connection to SVD
The Schmidt decomposition follows from the SVD of the coefficient matrix $\tau_{ij}$:
$$
\ket{\psi} = \sum_{i,j} \tau_{ij} \ket{i}_A \ket{j}_B
$$
A state is separable iff the Schmidt rank $r = 1$ (i.e., $\tau_{ij}$ has rank one).

---

# Basic Examples

<style scoped>
p, li {
   font-size: 18pt;
   color: #000000;
}
</style>

### GHZ state
$$
\ket{\text{GHZ}_N} = \frac{1}{\sqrt{2}}(\ket{0}^{\otimes N} + \ket{1}^{\otimes N})
$$
$$
\Lambda^2 = \frac{1}{2}, \quad E_G = \frac{1}{2}
$$
Nearest product state: $\ket{+}^{\otimes N}$ or $\ket{-}^{\otimes N}$, independent of $N$.

### W state
$$
\ket{W_N} = \frac{1}{\sqrt{N}}(\ket{10\cdots 0} + \ket{01\cdots 0} + \cdots + \ket{0\cdots 01})
$$
$$
\Lambda^2 = \left(\frac{N-1}{N}\right)^{N-1} \xrightarrow{N \to \infty} \frac{1}{e}, \quad E_G = 1 - \Lambda^2
$$

The W state has **larger** geometric entanglement than GHZ for $N \geq 3$: $E_G^W > E_G^{\text{GHZ}}$.

---

# Computation and the Injective Tensor Norm

<style scoped>
p, li {
   font-size: 18pt;
   color: #000000;
}
</style>

### Multipartite states
For an $N$-partite state $\ket{\psi} = \sum_{i_1, \ldots, i_N} T_{i_1 \cdots i_N} \ket{i_1 \cdots i_N}$, the maximal overlap is:
$$
\Lambda(\psi) = \max_{\ket{a_k}} \left| \sum_{i_1, \ldots, i_N} T_{i_1 \cdots i_N} \, (a_1)_{i_1} \cdots (a_N)_{i_N} \right|
$$

### Injective tensor norm
This is precisely the **injective tensor norm** of the coefficient tensor $T$:
$$
\|T\|_\sigma = \max_{\|a_k\| = 1} \left| T(a_1, a_2, \ldots, a_N) \right| = \Lambda(\psi)
$$

The injective tensor norm is also called the **spectral norm** of the tensor.

**Remark:** Computing $\Lambda(\psi)$ for multipartite states is in general **NP-hard**. However, analytical results exist for specific families (GHZ, W, Dicke, cluster states), and iterative algorithms can approximate it.

---

# Mixed State Extension: Convex Roof

<style scoped>
p, li {
   font-size: 18pt;
   color: #000000;
}
</style>

For mixed states $\rho$, the geometric measure is defined via the **convex roof construction**:
$$
E_G(\rho) = \min_{\{p_i, \ket{\psi_i}\}} \sum_i p_i \, E_G(\ket{\psi_i})
$$
where the minimization is over all pure-state decompositions $\rho = \sum_i p_i \ket{\psi_i}\bra{\psi_i}$.

### Equivalent formulation
$$
\Lambda^2(\rho) = \max_{\{p_i, \ket{\psi_i}\}} \sum_i p_i \, \Lambda^2(\ket{\psi_i})
$$

This can be reformulated as a **MAX-$N$ Hamiltonian problem**: finding the maximal overlap with all fully separable states is related to optimizing over product states in the geometric measure.

**Key difficulty:** The convex roof minimization is generally intractable. Analytical results exist only for special cases (e.g., permutation-symmetric states).

---

# Asymptotic Behaviour of Generic States

<style scoped>
p, li {
   font-size: 18pt;
   color: #000000;
}
</style>

### Key result
For Haar-random $N$-qubit pure states:
$$
\boxed{\text{Prob}\!\left(\Lambda^2(\psi) > 3N^2 \cdot 2^{-N}\right) < \exp(-N)}
$$

This means that for generic multi-qubit states, the maximal overlap with product states is **exponentially small**:
$$
\Lambda^2(\psi) \lesssim 3N^2 \cdot 2^{-N}
$$

### Physical interpretation
- The geometric entanglement $E_G = 1 - \Lambda^2 \to 1$ for almost all states
- Generic states are **maximally entangled** in the geometric sense
- The bound $\Lambda^2 \approx \mathcal{O}(N^2 / 2^N)$ matches the state space dimension scaling

This follows directly from the **Schur convexity** of the geometric measure and concentration of measure phenomena on the unit sphere in high-dimensional Hilbert spaces.

---

# "Too Entangled to Be Useful"

<style scoped>
p, li {
   font-size: 18pt;
   color: #000000;
}
</style>

### Connection to measurement-based quantum computation (MBQC)
In MBQC, computation proceeds by performing local measurements on a highly entangled **resource state** (e.g., cluster states).

However, if $\Lambda^2(\psi) \lesssim 3N^2 \cdot 2^{-N}$:
- Local measurement outcomes become **uncorrelated** with the global state
- Measurement results carry **negligible information** about the entangled state
- The state cannot serve as a useful resource for computation

### The paradox of entanglement
- Only states with **structured entanglement** (e.g., cluster, graph states) are useful for MBQC
- **Generic** multi-qubit states have near-maximal entanglement but are computationally **useless**
- Useful entanglement is **atypical** — it lives on a measure-zero subset of Hilbert space

This underscores that the **type** of entanglement, not just its **amount**, determines usefulness.

---

# Why Tensor Eigenvalues?

<style scoped>
p, li {
   font-size: 18pt;
   color: #000000;
}
</style>

### The computational problem
Computing the geometric measure requires solving:
$$
\Lambda(\psi) = \max_{\|a_k\|=1} \left| \sum_{i_1, \ldots, i_N} T_{i_1 \cdots i_N} \, (a_1)_{i_1} \cdots (a_N)_{i_N} \right|
$$
This is a **polynomial optimization on the product of unit spheres** — a non-convex problem.

### Why eigenvalues?
For **matrices** ($N = 2$), the analogous problem $\max_{\|x\|=1} |x^T A \, y|$ is solved by the **singular value decomposition**. The critical points satisfy $Ay = \sigma x$, $A^T x = \sigma y$.

For **tensors** ($N \geq 3$), no SVD exists in general. Instead, the critical points of the optimization satisfy a **tensor eigenvalue equation**:
$$
T \bar{x}^{N-1} = \lambda \, x, \quad \|x\| = 1
$$
The **largest eigenvalue** gives $\Lambda(\psi)$. Tensor eigenvalue theory thus provides the natural mathematical framework for characterizing the extrema of the geometric measure.

---

# Tensor Eigenvalues: Setup

<style scoped>
p, li {
   font-size: 18pt;
   color: #000000;
}
</style>

The coefficient tensor $T_{i_1 i_2 \cdots i_N}$ of a quantum state $\ket{\psi}$ defines a multilinear form. The geometric measure connects to the **eigenvalue theory of tensors**.

### Tensor notation
For a tensor $T$ of order $N$ with dimensions $d_1 \times d_2 \times \cdots \times d_N$, define the contraction:
$$
T \bar{x}^{N-1} := \sum_{i_2, \ldots, i_N} T_{i_1 i_2 \cdots i_N} \, \bar{x}_{i_2}^{(2)} \cdots \bar{x}_{i_N}^{(N)}
$$

### Eigenvalue equation
A scalar $\lambda$ and vector $x$ satisfy the **tensor eigenvalue equation** if:
$$
T \bar{x}^{N-1} = \lambda \, x
$$
with the normalization $x^T x = 1$ (or $x^\dagger x = 1$ in the complex case).

The **largest eigenvalue** of the tensor equals $\Lambda(\psi)$, the maximal overlap with product states.

---

# Z-Eigenvalues

<style scoped>
p, li {
   font-size: 18pt;
   color: #000000;
}
</style>

### Definition (for real symmetric tensors)
A scalar $\lambda \in \mathbb{R}$ is a **Z-eigenvalue** of a real symmetric tensor $T$ if there exists $x \in \mathbb{R}^d$ with $x^T x = 1$ such that:
$$
T \bar{x}^{N-1} = \lambda \, x
$$

### Properties
- Z-eigenvalues always exist for symmetric tensors
- Every Z-eigenvalue is a **critical value** of the polynomial $T(x, x, \ldots, x)$ on the unit sphere
- The **largest Z-eigenvalue** equals $\|T\|_\sigma$ (the injective tensor norm)

### Connection to entanglement
For a real state $\ket{\psi}$ with coefficient tensor $T$:
$$
\Lambda(\psi) = \max \{ |\lambda| : \lambda \text{ is a Z-eigenvalue of } T \}
$$

This reformulates entanglement quantification as a **polynomial optimization** on the unit sphere — a well-studied problem in multilinear algebra.

---

# US-Eigenvalues

<style scoped>
p, li {
   font-size: 18pt;
   color: #000000;
}
</style>

### Beyond real tensors: unitary eigenvalues
For complex tensors (general quantum states), Z-eigenvalues are insufficient. The appropriate generalization uses **US-eigenvalues** (unitary-similarity eigenvalues).

### Definition
A scalar $\lambda$ is a **US-eigenvalue** of tensor $T$ if there exist unitaries $U_2, \ldots, U_N$ and a vector $x$ with $x^\dagger x = 1$ such that:
$$
T(I, U_2, \ldots, U_N) \bar{x}^{N-1} = \lambda \, x
$$

### Key result
For a general $N$-partite state $\ket{\psi}$ with coefficient tensor $T$:
$$
\Lambda(\psi) = \max \{ |\lambda| : \lambda \text{ is a US-eigenvalue of } T \}
$$

### Significance
- For **real** symmetric tensors, US-eigenvalues reduce to Z-eigenvalues
- US-eigenvalues capture the full **complex structure** of the tensor
- Connects entanglement quantification to the rich mathematical theory of **tensor decompositions**

---

# Summary: The Geometric Perspective

<style scoped>
p, li {
   font-size: 18pt;
   color: #000000;
}
</style>

The geometric measure of entanglement provides a **bridge** between physics and mathematics:

### Physical insights
| Result | Implication |
|--------|------------|
| $\Lambda^2 \lesssim 3N^2 \cdot 2^{-N}$ | Generic states are "too entangled to be useful" |
| $E_G^W > E_G^{\text{GHZ}}$ | W states are more geometrically entangled than GHZ |
| Convex roof | Mixed state extension is NP-hard in general |

### Mathematical connections
| Concept | Entanglement interpretation |
|---------|---------------------------|
| Injective tensor norm | $= \Lambda(\psi)$, maximal product state overlap |
| Z-eigenvalues | Extrema on real unit sphere $\to$ $\Lambda$ for real states |
| US-eigenvalues | Complex generalization $\to$ $\Lambda$ for arbitrary states |

### Open directions
- Extending the geometric measure to multipartite **mixed states** efficiently
- Connecting tensor rank to operational entanglement properties
- Better algorithms for computing the injective tensor norm

---

# References

<style scoped>
section h1 {
  font-size: 28pt;
}
p, li {
   font-size: 13pt;
   color: #000000;
}
</style>

## Main Papers
- Yu, X.-D., Imai, S., and Gühne, O. "Optimal Entanglement Certification from Moments of the Partial Transpose." Physical Review Letters **127**, 060504 (2021).
- Weinbrenner, L. T. and Gühne, O. "Quantifying entanglement from the geometric perspective." EPL **151**, 68001 (2025).

## Related Works
- Elben, A. et al. "Statistical correlations between locally randomized measurements." Phys. Rev. Lett. **125**, 200501 (2020).
- Neven, A. et al. "Symmetry-resolved entanglement detection using partial transpose moments." npj Quantum Inf. **7**, 152 (2021).
- Shimony, A. "Degree of entanglement." Ann. N.Y. Acad. Sci. **755**, 675 (1995).
- Wei, T.-C. and Goldbart, P. M. "Geometric measure of entanglement and applications." Phys. Rev. A **68**, 042307 (2003).
- Barnum, H. and Linden, N. "Monotones and invariants for multi-particle quantum states." J. Phys. A **34**, 6787 (2001).
- Qi, L. "Eigenvalues of a real supersymmetric tensor." J. Symb. Comput. **40**, 1302 (2005).

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
