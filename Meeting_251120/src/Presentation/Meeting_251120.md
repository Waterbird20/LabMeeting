---

title       : Trial and Error with CPMG and DDrf
author      : Donghun Jung
# description : This is an example of how to use my themes.
# keywords    falserp, Slides, Themes.
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
RK7 Implementation and CPMG Analysis
</div>
 
<div class="author">
Donghun Jung
</div>

<div class="date">
20 Nov 2025
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

# TODO

1. **Analytical study:** Find better parameters

2. **Polish code:**
   - Memory management
   - Find better hyperparameters for faster simulation
   - Reduce simulation time to ~1min (for learning purposes)

3. **Run learning** after obtaining good parameters

4. **(Later) Develop density matrix time-evo solver in PyTorch**
   - NV/nuclear spins are not fully initialized to pure states
   - Need to consider mixed states → density matrix approach


---

# Progress Update

## Completed Tasks
- **Code optimization**
  - Memory management improvements
  - Hyperparameter tuning for faster simulation
  - Reduced simulation time to ~1 min (for learning purposes)

- **Density matrix/Unitary matrix solver development**
  - Implemented time-evolution solver in PyTorch
  - Supports mixed states for non-initialized NV/nuclear spins

## In Progress
- **Numerical/Analytical analysis of CPMG** 

---

# Progress Update


## Pending(DDrf)

- **Analytical study**: Determine better control parameters
- **Run learning**: After obtaining optimized parameters

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
   margin-left: -20px;
   flex: 0 0 50%;
   display: flex;
   align-items: center;
   justify-content: center;
   padding-bottom: 11rem;
}

li {
   font-size: 0.85rem;
}

</style>

<div class="container">
<div class="col-left-content">

- Progress Update
- Solving Differential Equations
  - Euler Method & Runge-Kutta Method
  - Key Features of RK7 Implementation
  - Why PyTorch Implementation Was Necessary
- CPMG
  - CPMG Sequence
  - Finding τ Analytically
  - Conditional Operation

</div>

<div class="col-right-content">

- CPMG Analysis
  - Effect of Non-negligible MW Pulse Time
  - Rotation Axis Analysis ($\vec{\sigma_{0}}\cdot\vec{\sigma_{1}}$)
  - Singular Value Analysis ($\lambda_i$)
- Future Work

</div>

</div>



---

# Solving Differential Equations

## Time-Dependent Schrödinger Equation
$$
\frac{\partial}{\partial t} \ket{\psi} = -i \mathcal{H} (t) \ket{\psi}
$$
## Naive Approach (Euler Method)

Consider a very simple model:
$$
\mathcal{H} = \begin{pmatrix}
0 & 1 \\
1 & 0
\end{pmatrix}
$$
and initial state $\ket{\psi} = \ket{0}$. We can do something like this:
$$
\ket{\psi(t + dt)} = \ket{\psi(t)} -i dt \mathcal{H}\ket{\psi(t)} 
$$

(General idea of the Euler method)



---

# (Explicit) Runge-Kutta Method

The generalization of explicit Runge--Kutta methods is given by
$$
y_{n+1} = y_n + dt \sum_{i=1}^{s} b_i k_i,
$$
where
$$
\begin{align*}
k_1 &= f(t_n, y_n), \\
k_2 &= f(t_n + c_2 dt, y_n + (a_{21} k_1)dt), \\
k_3 &= f(t_n + c_3 dt, y_n + (a_{31} k_1 + a_{32} k_2)dt), \\
&\vdots \\
k_s &= f(t_n + c_s dt, y_n + (a_{s1} k_1 + a_{s2} k_2 + \cdots + a_{s,s-1} k_{s-1})dt).
\end{align*}
$$
I ported code from `qutip/solver/integrator/qutip_integrator.py`. The specific coefficient values are shown [here](http://people.math.sfu.ca/~jverner/).

---

# Key Features of Implementation and Improvement

<style scoped>
table { font-size: 0.8rem; }
</style>

| Feature | Description |
|---------|-------------|
| **Modified Norm** | Error-controlled stepping with WRMSE norm |
| **Memory Pooling** | Pre-allocated buffers minimize additional GPU memory allocations |
| **GPU Acceleration** | Full PyTorch tensor operations with CUDA support |
| **Fixed Time Step** | Removed adaptive time stepping; `dt` is not re-estimated at every step |
| **Event-Adaptive Stepping** | The solver automatically reduces step size during critical intervals |
---

# Key Features of Implementation and Improvement

<style scoped>
p {
   font-size: 0.9rem
}
</style>

### Modified Norm

The solver uses weighted root mean square error norm:

$$\text{error} = \sqrt{\frac{1}{N}\sum_i \left(\frac{|\Delta y_i|}{\text{atol} + \text{rtol} \cdot |y_i|}\right)^2}$$

This norm does not explicitly test whether the state vector norm is preserved at 1. Nevertheless, the norm remains preserved if `dt` and total evolution time are chosen properly. This also allows us to extend the RK7 implementation beyond SESolver to other solvers.

### Fixed Time Step

With fixed time stepping, we must verify that the final state remains unchanged when `dt` is reduced (i.e., the solution has converged). However, this approach reduces execution time. 

---

# Key Features of Implementation and Improvement

Module Structure is shown below:
```
solver/
├── __init__.py          # Package exports
├── solver.py            # Base Integrator class
├── explicit_RK.py       # Core RK algorithm
├── rk7_coeff.py         # Verner 7 coefficients
├── sesolver.py          # Schrödinger equation solver
├── lvnesolver.py        # Lindblad von Neumann equation solver
└── upsolver.py          # Unitary propagator solver
```


---

# Key Features of Implementation and Improvement

**SESolver** - Schrödinger Equation: Solves the time-dependent Schrödinger equation:
$$i\hbar \frac{\partial}{\partial t}|\psi(t)\rangle = H(t)|\psi(t)\rangle$$

**LVNESolver** - Lindblad von Neumann Equation: Solves the master equation for density matrices:
$$\frac{\partial \rho}{\partial t} = -\frac{i}{\hbar}[H, \rho] + \mathcal{L}[\rho]$$

**UPSolver** - Unitary Propagator: Evolves the unitary propagator matrix:
$$i\hbar \frac{\partial U}{\partial t} = H(t)U(t)$$

---

# Why It Was Necessary

- **PyTorch enables GPU integration**
   - We plan to realize GHZ states for up to (1+8)-qubit systems.
   - QuTiP's C-implemented solver would face scaling issues at this size.
- **PyTorch enables optimization with convenient and efficient optimizers**
   - DDrf requires tuning: DDrf is too complex to selectively control each nuclear spin qubit for precise phase updates.
   - Pulse optimization: Implementing two-qutrit QST POVM operators in 2 $\mu$s requires smoother pulses, which can be derived from the GRAPE algorithm.

**Note:** QuTiP has supported GPU since May 2024 (v5.0.2, via TensorFlow)

---

# Dynamic Decoupling

<style scoped> 
.container{
   display: flex;
   align-items: center;
   width: 100%;
   height: 100%;
}
.col-left-content{
   flex: 0 0 50%;
   padding-right: 2rem;
   padding-bottom: 3rem;
   color: #000000;
}

.col-right-content{
   flex: 0 0 35%;
   display: flex;
   align-items: center;
   justify-content: center;
   padding-bottom: 5rem;
}

img[alt~="rightside"]{
   position: absolute;
   top: 10rem;
   right: 3rem;
   width: 15rem;
}

em {
   font-size: 0.7rem;
}

</style>

## Original Purpose: Extending Coherence Time
<div class="container">
<div class="col-left-content">

CPMG was originally used to increase coherence time, but can also be used for various purposes, including implementing conditional gates.


## Hamiltonian

$$
\begin{aligned}
\mathcal{H} =& \underbrace{\gamma_c B_z I_z^i + A_{||}^i S_z I_z^i + A_{\perp} S_z I_x^i}_{\text{Drift Hamiltonian}} + \underbrace{\frac{1}{\sqrt{2}} \Omega_{\text{MW}} S_x}_{\pi \text{-pulse}}  \\
\rightarrow \mathcal{H} =& \ket{0}\bra{0} \otimes \omega_0 I_z + 
\ket{-1}\bra{-1} \otimes \left( \omega_L I_z -  A_{||}I_z - A_{\perp}I_x  \right) \\
\rightarrow \mathcal{H} =& \ket{0}\bra{0} \otimes H_0 + 
\ket{-1}\bra{-1} \otimes H_1 
\end{aligned} 
$$

That is, nuclear spin evolves by $e^{-i H_{0(1)} t}$ when electron spin is in state $\ket{0}(\ket{1})$.


</div>
<div class="col-right-content">

![](Meeting_251024/src/Presentation/media/CPMG_sequence.png)
<br>
<em>
Figure. Diagram for CPMG sequence/ CPMG is a repetition of $(\tau - \pi - 2\tau - \pi - \tau)$ for $\frac{N}{2}$ times. The figure is extracted from `Phys. Rev. X 15, 021011`.
</em>

</div>



---

# CPMG Sequence

- Initial electron spin state 0(1): $e^{-i \phi \hat{I}\cdot\hat{\sigma^{i}}}$
   - if NV spin is 0-state: $e^{-i \phi \hat{I}\cdot\hat{\sigma^{0}}} = e^{-i H_{0} \tau}e^{-i H_{1} 2\tau}e^{-i H_{0} \tau}$
   - if NV spin is 1-state: $e^{-i \phi \hat{I}\cdot\hat{\sigma^{1}}} = e^{-i H_{1} \tau}e^{-i H_{0} 2\tau}e^{-i H_{1} \tau}$

- When vectors $\sigma^0$ and $\sigma^1$ are anti-parallel → conditional gate
- Directions are generally parallel, but become anti-parallel at specific $\tau$ values

**Note**: $\phi$ is the same regardless of whether the initial state is 0 or 1 in the CPMG train.

---

# Finding τ Analytically(?)

To enable a conditional gate via repetition of CPMG train, we should choose $\tau$ such that $\hat{\sigma^{0}}\cdot\hat{\sigma^{1}} = -1$.
$\tau$ can be found analytically under a strong magnetic field. [[Phys. Rev. Lett. 109, 137602]](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.109.137602)
$$
\tau \simeq \frac{(2k-1) \pi}{2\gamma_{c}B_{z} + A_{||}}
$$
Then, the number of repetitions of CPMG train is chosen meticulously to achieve $N\phi = \frac{\pi}{2}$.

---

# Conditional Operation

At a certain $\tau$, we observe a conditional operation. Red (Blue) arrow denotes the rotation axis when the initial electron spin state was $\ket{0}$ ($\ket{1}$).

<!-- TODO: add figure -->


---

# CPMG analysis

- Initial electron spin state 0(1): $e^{-i \phi \hat{I}\cdot\hat{\sigma^{i}}}$
   - if NV spin is 0-state: $U_0 = e^{-i \phi \hat{I}\cdot\hat{\sigma^{0}}} = e^{-i H_{0} \tau}e^{-i H_{1} 2\tau}e^{-i H_{0} \tau}$
   - if NV spin is 1-state: $U_1 = e^{-i \phi \hat{I}\cdot\hat{\sigma^{1}}} = e^{-i H_{1} \tau}e^{-i H_{0} 2\tau}e^{-i H_{1} \tau}$

If one CPMG train can be described this way, the unitary gate takes the form of a block matrix:
$$
U^{\otimes \frac{N}{2}} = \begin{pmatrix}
U_0^{\otimes \frac{N}{2}} & 0 \\
0 & U_1^{\otimes \frac{N}{2}}
\end{pmatrix}
$$
where off-diagonal components are zero.

However, if the time for MW implementing $\pi$-pulse gate is not short enough, it can cause additional effects beyond those described below. 


---

# CPMG analysis

However, if we calculate $U$ with non-negligible MW time, off-diagonal components become non-zero and each block matrix is no longer unitary.

$$
U^{\otimes\frac{N}{2}} =
\left(
e^{-i \tau H_{0}}
e^{-i \tau_{\text{MW}}(H_{0} + H_{\text{MW}})}
e^{-i 2\tau H_{0}}
e^{-i \tau_{\text{MW}}(H_{0} + H_{\text{MW}})}
e^{-i \tau H_{0}}
\right)^{\otimes \frac{N}{2}}
= \begin{pmatrix}
V_{00} & V_{01} \\
V_{10} & V_{11}
\end{pmatrix}
$$

Its operation on a state is described as: 
$$
\ket{\psi} = \frac{1}{\sqrt{2}}(\ket{0} + \ket{1}) \otimes \ket{\phi}
$$
then, we have
$$
\ket{\psi} \rightarrow \frac{1}{\sqrt{2}}(
\ket{0}\otimes \underbrace{(V_{00} + V_{01})}_{=V^0}\ket{\phi}
+ 
\ket{1}\otimes \underbrace{(V_{11} + V_{10})}_{=V^1}\ket{\phi}
)
$$

---

# CPMG analysis



Both $V_{0}$ and $V_{1}$ are not unitary. To analyze them, I used polar decomposition which is related to SVD.
$$
V = U P = (uv^{\dagger})(v \Sigma v^{\dagger})
$$
where $u, v$ are unitary matrices derived from SVD. Here $U$ is unitary, while $P$ is non-unitary and can be regarded as a quantum channel.
$$
P = \sum_i \lambda_{i} \ket{\lambda_{i}}\bra{\lambda_{i}}
$$
where $\lambda_{i}$ are the singular values. Thus, we have
$$
\ket{\psi} \rightarrow \frac{1}{\sqrt{2}}(
\ket{0}\otimes U_{0}P_{0}\ket{\phi}
+ 
\ket{1}\otimes U_{1}P_{1}\ket{\phi}
)
$$

--- 

# CPMG analysis
<style scoped>
   li{
      font-size: 0.8rem;
   }
   tr {
      font-size: 0.8rem;
   }
</style>
We can analyze from two perspectives:
- How $\vec{\sigma_{0}}\cdot\vec{\sigma_{1}}$ (with respect to $U_0$ and $U_1$) changes with $t_{\text{MW}}$ and $N$.
- How the singular values $\lambda_{0,1}$ change with $t_{\text{MW}}$ and $N$. 

For this tentative simulation, I used the following parameters:

| Param | Value | Description | Param | Value | Description
|-----------|-------|-------------|-----------|-------|-------------|
| $B_z$ | 440.1 G | Magnetic field strength |$\tau$ | 0.1 - 20 μs | Free evolution time (varied) |
| $A_{\|\|}$ | $(2\pi)$  130 kHz | Parallel hyperfine coupling |$\tau_{\text{MW}}$ | 0.1 - 50 ns | π-pulse duration (varied) |
| $A_{\perp}$ | $(2\pi)$  50 kHz | Perpendicular hyperfine coupling |$N$ | 2 - 40 | Number of CPMG repetitions |


---

# CPMG analysis: $\vec{\sigma_{0}}\cdot\vec{\sigma_{1}}$

The following colormap shows how the $\vec{\sigma_{0}}\cdot\vec{\sigma_{1}}$ value changes with $\tau$ and $\tau_{\text{MW}}$.

$\tau$ shifts to shorter values as $\tau_{MW}$ increases. However, the amount of left-shift keeps the total time for CPMG train constant. 

<!-- TODO: add figure -->

---

# CPMG analysis: $\vec{\sigma_{0}}\cdot\vec{\sigma_{1}}$

Varying $N$ and fixing $\tau_{\text{MW}}$, the value $\tau$ that makes total operation conditional, $\vec{\sigma_{0}}\cdot\vec{\sigma_{1}}=-1$, is unchanged. 

<!-- TODO: add figure -->


--- 

# CPMG analysis: $\lambda_{i}$

If $\lambda_{i} \simeq 1$, $P$ can be approximated as the identity matrix.
$$
P = \sum_i \lambda_{i} \ket{\lambda_{i}}\bra{\lambda_{i}}
$$
This operator amplifies or attenuates components based on $\lambda_{i}$. While the overall operation preserves norm, $P$ alone does not preserve the norm of the state vector. 

**Quick example:**
Suppose $\lambda_{i} = \left\{ \sqrt{\frac{3}{2}}, \sqrt{\frac{1}{2}} \right\}$, with $\ket{\lambda_{0}} = \ket{0}$ and $\ket{\lambda_{1}} = \ket{1}$.
If the prepared state is $\ket{+}$, it transforms to
$$
\ket{+} \rightarrow \frac{\sqrt{3}}{2}\ket{0} + \frac{1}{2}\ket{1}
$$


---

# CPMG analysis: $\lambda_{i}$
As expected, as $\tau_{\text{MW}}$ and $N$ increase, the singular value $\lambda_{i}$ deviates from 1. 

<!-- TODO: add figure -->

---

# [TODO] CPMG analysis: $\lambda_{i}$

In a CPMG sequence, we perform an $X$-basis measurement at the end of the sequence. Without considering the drift Hamiltonian during $\pi$-pulse time, we have
$$
\begin{align}
P_{x} &= \text{Tr} (\ket{+}\bra{+} \otimes I) \rho = \text{Tr} (\ket{+}\bra{+} \otimes I) U\rho_{0}U^{\dagger} \\
&= \frac{1}{4} \text{Tr} U_{0}\rho_{{}^{13}\text{C}} U_{0}^{\dagger} + \frac{1}{4} \text{Tr} U_{0}\rho_{{}^{13}\text{C}} U_{1}^{\dagger} + \frac{1}{4} \text{Tr} U_{1}\rho_{{}^{13}\text{C}} U_{0}^{\dagger} + \frac{1}{4} \text{Tr} U_{1}\rho_{{}^{13}\text{C}} U_{1}^{\dagger} \\
&= \frac{1}{2}\left( \Re (\text{Tr} U_{0} U_{1}^{\dagger} ) + 1  \right) 
\end{align}
$$
where 
$$
\rho_{0} = \ket{+}\bra{+} \otimes \rho_{{}^{13}\text{C}}
$$
and the last line follows from $\rho_{{}^{13}\text{C}} = I$.

---

# [TODO] CPMG analysis: $\lambda_{i}$

Now it becomes more complex...
$$
\begin{align}
P_{x} &= \text{Tr} (\ket{+}\bra{+} \otimes I) \rho = \text{Tr} (\ket{+}\bra{+} \otimes I) U\rho_{0}U^{\dagger} \\
&= \text{Tr} \frac{1}{4}
\begin{pmatrix}
I & I\\ I & I
\end{pmatrix}
\begin{pmatrix}
V_{00} & V_{01} \\
V_{10} & V_{11}
\end{pmatrix}
\begin{pmatrix}
\rho & \rho\\ \rho & \rho
\end{pmatrix}
\begin{pmatrix}
V_{00}^{\dagger} & V_{10}^{\dagger} \\
V_{01}^{\dagger} & V_{11}^{\dagger}
\end{pmatrix} \\
&= \frac{1}{4}\text{Tr}(\sigma_{00} + \sigma_{01} + \sigma_{10} + \sigma_{11})
\end{align}
$$
where
$$
\begin{align}
\sigma_{00} &= (V_{00} + V_{01})\rho_{{}^{13}\text{C}}(V_{00}^{\dagger} + V_{01}^{\dagger}) &= U_{0}P_{0} \rho_{{}^{13}\text{C}} P_{0}^{}U_{0}^{\dagger} \\
\sigma_{01} &= (V_{00} + V_{01})\rho_{{}^{13}\text{C}}(V_{10}^{\dagger} + V_{11}^{\dagger}) &= U_{0}P_{0} \rho_{{}^{13}\text{C}} P_{1}^{}U_{1}^{\dagger} \\
\sigma_{10} &= (V_{10} + V_{11})\rho_{{}^{13}\text{C}}(V_{00}^{\dagger} + V_{01}^{\dagger}) &= U_{1}P_{1} \rho_{{}^{13}\text{C}} P_{0}^{}U_{0}^{\dagger} \\
\sigma_{11} &= (V_{10} + V_{11})\rho_{{}^{13}\text{C}}(V_{10}^{\dagger} + V_{11}^{\dagger}) &= U_{1}P_{1} \rho_{{}^{13}\text{C}} P_{1}^{}U_{1}^{\dagger}
\end{align}
$$

---

# Future Work

## Analytical Study
- Derive analytical expression for $P_x$ with non-negligible MW time
- Determine optimal control parameters from analysis

## Machine Learning
- Run learning with optimized parameters
- Validate learned parameters against analytical predictions

## Numerical Analysis
- Complete singular value analysis for $P_x$ measurement
- Investigate effect of multiple $^{13}$C spins

