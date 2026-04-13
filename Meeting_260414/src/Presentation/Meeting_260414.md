---

title       : "Entanglement Detection and Quantification: PT-Moments and Geometric Measure"
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
   font-size: 24pt;
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
Entanglement Detection and Quantification: PT-Moments and Geometric Measure
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
Paulee Lab, Center for Quantum Technology, Korea Institute of Science and Technology
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
   padding-bottom: 12.5rem;
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

A bipartite state $\ket{\psi}$ is **separable** if it can be written as:
$$
\ket{\psi} = \sum_i \lambda_i \ket{\psi_A}\ket{\psi_B}.
$$

For density matrix, a bipartite state $\rho$ is **separable** if it can be written as:
$$
\rho = \sum_{i} p_i \, \rho^{A}_{i} \otimes \rho^{B}_{i}, \quad \sum_i p_i = 1, \quad p_i > 0
$$

Otherwise, the state is **entangled**.

### Approaches to detect entanglement
- **PNCP maps** (e.g., partial transpose): produce negative eigenvalues for entangled states
- **Entanglement witnesses**: observables with non-negative expectation on all separable states
<!-- - **Entanglement measures**: quantify the degree of entanglement (negativity, geometric measure, ...) -->

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
which does not change eigen-spectrum for separable state.
<!-- For a single-qubit subsystem, this decomposes as:
$$
T(\rho) = \frac{1}{2}(\rho + X\rho X - Y \rho Y + Z \rho Z)
$$ -->

### PPT Criterion (Peres–Horodecki)
- **Separable** $\Rightarrow$ $\rho^{T_B} \geq 0$ (positive partial transpose)
- **PPT is necessary and sufficient** for $2\times2$ and $2\times3$ systems
- In higher dimensions, PPT entangled (bound entangled) states exist

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

***Key theorem***:  Fully decomposable witnesses detect **exactly** the states that are **not PPT mixtures**:
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

***Limitation***: All these methods generally require **full density matrix** $\rho$ — expensive quantum state tomography, and even numerical search for entanglement witness $W$. 

---

# [Review] Summary

<style scoped>
table {
   font-size: 15pt;
   margin: 0 auto;
}
p, li {
   font-size: 16pt;
}
</style>

<!--
================================================================================
FIGURE 1 — STATE-SPACE HIERARCHY DIAGRAM
================================================================================
CONTEXT FOR IMAGE GENERATOR:
In quantum information, the set of all bipartite density matrices ρ on
H_A ⊗ H_B has a nested structure. The innermost set is "separable states"
(ρ = Σ p_i ρ^A_i ⊗ ρ^B_i). Containing it is the set of "PPT states" (states
whose partial transpose ρ^{T_B} is positive semidefinite). Surrounding that
is the set of all density matrices. The region between PPT and all-states is
"NPT entangled" (detectable by the partial transpose). The region between
separable and PPT is "bound entangled" / "PPT entangled" (NOT detectable by
PPT — requires witnesses). Different entanglement-detection techniques cover
different regions. This is one of the most iconic diagrams in the field.

WHAT TO DRAW:
- Three nested rounded rectangles (or ellipses) — NOT a Venn diagram, but
  strict containment (one fully inside the other).
- Outermost rectangle: light gray fill. Label inside near its top edge:
  "All density matrices ρ ∈ D(H_A ⊗ H_B)".
- Middle rectangle: light blue fill. Label near top: "PPT states
  (ρ^{T_B} ≥ 0)".
- Innermost rectangle: light green fill. Label inside center: "Separable
  states   ρ = Σ p_i ρ^A_i ⊗ ρ^B_i".
- In the gray "NPT entangled" annulus between outer and middle: italic label
  "NPT entangled — detected by PPT, negativity, p_n-PPT, p_n-OPPT".
- In the blue "bound entangled" annulus between middle and inner: italic
  label "PPT-entangled (bound) — requires entanglement witnesses".
- Four arrows drawn from OUTSIDE the outer rectangle pointing at the regions
  they detect, each with a small text label at its base:
    (1) "PPT / Negativity"  →  NPT annulus
    (2) "p_n-OPPT (Paper 1)"  →  NPT annulus (draw to a different spot)
    (3) "Witnesses"  →  a cut plane slicing across the separable boundary
        (draw as a straight dashed line tangent to the inner rectangle)
    (4) "SDP hierarchy"  →  bound-entangled annulus

STYLE:
- Flat academic diagram, sans-serif, pastel fills, thin black borders.
- No 3D, no shading, no gradients. Clean and publication-ready.
- Approximate aspect ratio 4:3. Text legible at 14pt minimum.
================================================================================
-->

![width:560px](media/images/fig1_state_hierarchy.png)

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

# PT-Moments: Eigen-spectrum

If we are aware of all the PT-moments $p = (p_1 , p_2 , \cdots , p_d)$ where $d=d_A d_B$, all the eigenvalues of $\rho_{AB}^{T}$ can be directly calculated. It is just solving  equation, employing Newton's identities.

$$
\begin{align}
p_1 &= \mathrm{Tr} \rho_{AB}^{T} = \sum_{i} \lambda_i \\
p_2 &= \mathrm{Tr}{\rho_{AB}^{T}}^2 = \sum_{i} \lambda_i^2 \\
p_3 &= \mathrm{Tr}{\rho_{AB}^{T}}^3 = \sum_{i} \lambda_i^3 \\
\vdots & ~~~~~~~~~~~ \vdots ~~~~~~~~~~~~~~~~ \vdots
\end{align}
$$

---

# PT-Moments: Eigen-spectrum

The eigenvalues $\{\lambda_i\}$ are roots of
$$P(x) = \prod_{i=1}^d (x - \lambda_i) = x^d - e_1 x^{d-1} + e_2 x^{d-2} - \cdots + (-1)^d e_d$$
where $e_k$ are the **elementary symmetric polynomials**:
$$e_1 = \sum_i \lambda_i, \quad e_2 = \sum_{i<j} \lambda_i \lambda_j, \quad e_3 = \sum_{i<j<k} \lambda_i \lambda_j \lambda_k, \quad \ldots$$
These provide a recursive relation between $s_k = \sum_i \lambda_i^k$ and $e_k$: $s_1 = e_1$, $s_2 = s_1 e_1 - 2e_2$, $s_3 = s_2 e_1 - s_1 e_2 + 3e_3$, $\cdots$, and in general:
$$s_k = \sum_{j=1}^{k-1} (-1)^{j-1} s_{k-j}\, e_j + (-1)^{k-1} k\, e_k$$

---

# PT-Moments: Eigen-spectrum

For example of two-qubit system, 

One measure $p_1 = 1$ (trivial), $p_2 = p_2$, $p_3 = p_3$, $p_4 = p_4$. Then:
$$
\begin{align}
e_1 &= 1 \\
e_2 &= \frac{p_1^2 - p_2}{2} = \frac{1 - p_2}{2} \\
e_3 &= \frac{p_1^3 - 3p_1 p_2 + 2p_3}{6} = \frac{1 - 3p_2 + 2p_3}{6} \\
e_4 &= \frac{p_1^4 - 6p_1^2 p_2 + 3p_2^2 + 8p_1 p_3 - 6p_4}{24} = \frac{1 - 6p_2 + 3p_2^2 + 8p_3 - 6p_4}{24}
\end{align}
$$

The eigenvalues of $\rho_{AB}^{T_A}$ are then the four roots of
$$
x^4 - e_1 x^3 + e_2 x^2 - e_3 x + e_4 = 0 .
$$  


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

***Hamburger Moment Problem***: Given moments $m^{(n)} = (m_0, m_1, \ldots, m_n)$, does there exist a measure on $\mathbb{R}$ with these moments?
***Stieltjes Moment Problem***: Same question, but the measure is supported on $[0, \infty)$.

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
   font-size: 15pt;
   color: #000000;
}
</style>

<!--
================================================================================
FIGURE 2 — HANKEL MATRIX ANATOMY
================================================================================
CONTEXT FOR IMAGE GENERATOR:
A Hankel matrix is a square matrix whose anti-diagonals are constant — that is,
the (i, j) entry depends only on the SUM i+j. When the entries are the moments
p_0, p_1, p_2, ... of a probability measure on the real line (in our case, the
moments of the partial transpose), the positive semidefiniteness (PSD) of this
matrix characterizes whether the sequence could have come from a legitimate
probability distribution. A separable quantum state always produces a PSD
Hankel matrix; if the Hankel matrix is NOT PSD, then the state must be
entangled. This diagram should make the "anti-diagonal structure" visually
obvious — that is the entire point of calling it a Hankel matrix.

WHAT TO DRAW:
- A 4×4 square matrix drawn as a grid with thin black borders.
- Entries from top-left, reading row by row:
    Row 0:  p_0   p_1   p_2   p_3
    Row 1:  p_1   p_2   p_3   p_4
    Row 2:  p_2   p_3   p_4   p_5
    Row 3:  p_3   p_4   p_5   p_6
- CRUCIAL: color each anti-diagonal (cells sharing the same i+j) with the
  SAME soft pastel color. There are 7 anti-diagonals (i+j = 0 through 6).
  Use 7 distinct pastel hues — e.g., pale yellow (i+j=0), peach, pink,
  lavender, mint, sky blue, pale teal (i+j=6). The visual repetition is
  the pedagogical point.
- Draw a curved arrow outside the matrix pointing to one anti-diagonal
  (say, the one filled with p_2 entries in positions (0,2),(1,1),(2,0))
  with a label: "Anti-diagonal: constant entry p_{i+j}".
- Below the matrix, centered, write in italic:
    "H ≥ 0  ⟺  moments come from a valid measure on ℝ"
- To the right of the matrix, a callout box:
    "Violation of H ≥ 0  ⟹  state is entangled"
- Top-left corner: small label "H_3 (order-3 Hankel matrix)"

STYLE:
- Clean academic schematic. Matrix cells roughly 60×60 px. Entry text in
  serif math font (LaTeX-like), at least 18pt. Annotations in italic
  sans-serif. No 3D, no shadows, no gradients on cells — just flat fills.
- Approximate canvas 4:3 aspect ratio.
================================================================================
-->

Define the **Hankel matrices** from a moment sequence $m^{(n)} = (m_0, m_1, \ldots, m_n)$:
$$
H_k = \begin{pmatrix} m_0 & m_1 & \cdots & m_k \\ m_1 & m_2 & \cdots & m_{k+1} \\ \vdots & \vdots & \ddots & \vdots \\ m_k & m_{k+1} & \cdots & m_{2k} \end{pmatrix}, \quad
B_k = \begin{pmatrix} m_1 & m_2 & \cdots & m_{k+1} \\ m_2 & m_3 & \cdots & m_{k+2} \\ \vdots & \vdots & \ddots & \vdots \\ m_{k+1} & m_{k+2} & \cdots & m_{2k+1} \end{pmatrix}
$$

### Lemma (Classical moment theory)
**(a)** $m^{(n)} \in \mathcal{M}_n$ (Hamburger, measure on $\mathbb{R}$) $\iff$ $H_{\lfloor n/2 \rfloor} \geq 0$ where
$$
\mathcal{M}_n = \left\{ m^{(n)} | \mathrm{Tr}(\sigma X^{k} ) = m_k , \sigma \ge 0 , X^{\dagger} = X \right\}
$$
**(b)** $m^{(n)} \in \mathcal{M}_n^+$ (Stieltjes, measure on $[0,\infty)$) $\iff$ $H_{\lfloor n/2 \rfloor} \geq 0$ **and** $B_{\lfloor (n-1)/2 \rfloor} \geq 0$
$$
\mathcal{M}_n = \left\{ m^{(n)} | \mathrm{Tr}(\sigma X^{k} ) = m_k , \sigma \ge 0 , X \ge 0 \right\}
$$
Since separable states have $x_i \geq 0$, the Stieltjes condition applies. Violation of either Hankel condition certifies entanglement. Let $\sigma = I , X = \rho^{T}_{AB}$.

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

The Hankel criteria are **not optimal** — we may assume $\sigma \neq I$. 

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

<!-- ---

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
- The criterion is **dimension-independent**: the same formula works for any $d$ -->

---

# Comparison: $p_3$-PPT vs $p_3$-OPPT

<style scoped>
p, li {
   font-size: 16pt;
   color: #000000;
}
</style>

![width:470px](./Meeting_260414/src/Presentation/media/OPPT_vs_PPT_Detection.png)
### Hierarchical structure
The criteria form a hierarchy based on PT-moments:
- **$p_3$-PPT** ($n = 3$, Hankel): $p_3 \geq p_2^2$
- **$p_3$-OPPT** ($n = 3$, optimal): $p_3 \geq \alpha x^3 + (1-\alpha x)^3$
- **$p_n$-PPT** ($n = 3, 5, 7, \ldots$): higher-order Hankel conditions
- **$p_n$-OPPT** ($n = 3, 5, 7, \ldots$): higher-order optimal conditions

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
| 2 | 75.68% | 25.53% | 39.97% | 75.68% | 64.78% | 75.68% |
| 3 | 99.99% | 25.32% | 39.46% | 91.63% | 97.51% | 98.97% |
| 4 | 100% | 23.29% | 33.69% | 98.68% | 100% | 100% |
| 5 | 100% | 21.80% | 34.54% | 99.95% | 100% | 100% |
| 6 | 100% | 20.93% | 31.20% | 100% | 100% | 100% |

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

1. **Hankel matrix method** ($p_n$-PPT): relaxation to classical moment problems
   - Lowest order gives $p_3 \geq p_2^2$
   - Higher orders provide strictly stronger criteria
2. **Optimal method** ($p_n$-OPPT): exact solution of the PT-moment problem
   - Gives necessary and sufficient conditions
   - Detects significantly more entangled states
   - Dimension-independent
3.  **Advantages** over conventional methods
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
   font-size: 16pt;
   color: #000000;
}
</style>

![width:400px](./Meeting_260414/src/Presentation/media/fig4_geometric_schematic.png)

The **geometric measure** quantifies entanglement by the distance to the nearest separable (product) state.

***Definition***: For a pure state $\ket{\psi}$ of $N$ particles, define the **maximal overlap** with product states:
$$
\Lambda^2(\psi) = \max_{\ket{\phi} = \ket{a_1} \otimes \ket{a_2} \otimes \cdots \otimes \ket{a_N}} \left|\braket{\phi | \psi}\right|^2
$$
The **geometric measure of entanglement** is $E_G(\psi) = 1 - \Lambda^2(\psi)$.


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

The maximal overlap is simply the **largest Schmidt coefficient**$\Lambda(\psi) = s_1$.

The Schmidt decomposition follows from the SVD of the coefficient matrix $\tau_{ij}$:
$$
\ket{\psi} = \sum_{i,j} \tau_{ij} \ket{i}_A \ket{j}_B
$$
A state is separable iff the Schmidt rank $r = 1$ (i.e., $\tau_{ij}$ has rank one and must be one).

---

# Basic Examples

<style scoped>
p, li {
   font-size: 18pt;
   color: #000000;
}
</style>

***GHZ state***
$$
\ket{\text{GHZ}_N} = \frac{1}{\sqrt{2}}(\ket{0}^{\otimes N} + \ket{1}^{\otimes N})
$$
$$
\Lambda^2 = \frac{1}{2}, \quad E_G = \frac{1}{2}
$$
Nearest product state: $\ket{0}^{\otimes N}$ or $\ket{1}^{\otimes N}$, independent of $N$.

***W state***
$$
\ket{W_N} = \frac{1}{\sqrt{N}}(\ket{10\cdots 0} + \ket{01\cdots 0} + \cdots + \ket{0\cdots 01})
$$
$$
\Lambda^2 = \left(\frac{N-1}{N}\right)^{N-1} \xrightarrow{N \to \infty} \frac{1}{e}, \quad E_G = 1 - \Lambda^2
$$

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
**Remark:** Computing $\Lambda(\psi)$ for multipartite states is in general **NP-hard**. 

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

---

# Why Tensor Eigenvalues?

<style scoped>
p, li {
   font-size: 18pt;
   color: #000000;
}
</style>

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


---

# Asymptotic Behaviour of Generic States

<style scoped>
p, li {
   font-size: 16pt;
   color: #000000;
}
</style>

### Key result
For Haar-random $N$-qubit pure states (N>11):
$$
\boxed{\text{Prob}\!\left(\Lambda^2(\psi) > 3N^2 \cdot 2^{-N}\right) < \exp(-N)}
$$

This means that for generic multi-qubit states, the maximal overlap with product states is **exponentially small**:
$$
\Lambda^2(\psi) \lesssim 3N^2 \cdot 2^{-N}
$$

### Physical interpretation
- The geometric entanglement $E_G = 1 - \Lambda^2 \to 1$ for almost all states
- Generic states are highly entangled in the geometric sense


---

# Mixed State Extension: Convex Roof

<style scoped>
p, li {
   font-size: 18pt;
   color: #000000;
}
</style>

For mixed states $\rho$, the geometric measure might be extended via the convex roof construction:
$$
\Lambda^2(\rho) = \max_{\{p_i, \ket{\psi_i}\}} \sum_i p_i \, \Lambda^2(\ket{\psi_i}) = \min_{\{p_i, \ket{\psi_i}\}} \sum_i p_i \, (1 - \Lambda^2(\ket{\psi_i}))
$$
where the minimization is over all pure-state decompositions $\rho = \sum_i p_i \ket{\psi_i}\bra{\psi_i}$.

The convex roof minimization is generally intractable. Analytical results exist only for special cases (e.g., permutation-symmetric states).


---

# Summary: The Geometric Perspective

<style scoped>
p, li {
   font-size: 18pt;
   color: #000000;
}
</style>

### Concepts

- Injective tensor norm: $= \Lambda(\psi)$, maximal product state overlap 
- Z-eigenvalues: Extrema on real unit sphere $\to$ $\Lambda$ for real states 
- US-eigenvalues: Complex generalization $\to$ $\Lambda$ for arbitrary states 

### Open directions
- Extending the geometric measure to multipartite mixed states efficiently
- Connecting tensor rank to operational entanglement properties
- Better algorithms for computing the injective tensor norm


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
