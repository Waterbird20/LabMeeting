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
RK7 implementation and CPMG analysis
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


# Outline

- TODO(Last time)
- [Quick Fix] CPMG axis analysis
- [Quick Recap] 
- Machine Learning Approach: Why it fails?
- Strategy
- Implementation of Runge-Kutta 7
    - Why Euler Method fails
    - Key features of RK7 implementation
- CPMG & DDrf explanation
- Numerical Simulations

---

# What supposed to be done and what I have done.

1. **Analytical study:** Find better parameters

2. **Polish code:** <- Done
   - Memory management
   - Find better hyperparameters for faster simulation
   - Reduce simulation time to ~1min (for learning purposes)

3. **Run learning** after obtaining good parameters

4. Develop density matrix time-evo solver in PyTorch <- Done
   - NV/nuclear spins are not fully initialized to pure states
   - Need to consider mixed states → density matrix approach

5. (New) Numerical Analysis of CPMG

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
I ported code from `qutip/solver/integrator/qutip_integrator.py` . The details of the specific value of this method is shown [here](http://people.math.sfu.ca/~jverner/).

---

# Key Features of Implementation and Improvement


---

# Dynamic Decoupling

## Original Purpose: Extending Coherence Time
<div class="container">
<div class="col-left-content">

CPMG was used to increase coherence time, but can be used for many purposes, including implementing conditional gates.


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

That is, nuclear spin evolves by $e^{-i H_{0(1)} t}$ when electron spin lies on spin state $\ket{0}(\ket{1})$.


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
- Directions are generally parallel, but at a specific $\tau$ they become anti-parallel

**Note**: $\phi$ is equal whatever initial state is 0 or 1 in CPMG train.

---

# Finding τ Analytically(?)

To enable conditional Gate via repeatation of CPMG train, we should choose $\tau$ such that $\hat{\sigma^{0}}\cdot\hat{\sigma^{1}} = -1$.
$\tau$ can be found analytically under strong magnetic field. [[Phys. Rev. Lett. 109, 137602]](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.109.137602)
$$
\tau \simeq \frac{(2k-1) \pi}{2\gamma_{c}B_{z} + A_{||}}
$$
Then, the number of repeatation number of CPMG train is chosen meticulously, to acheive $N\phi = \frac{\pi}{2}$.

---

# Conditional Operation

At a certain $\tau$, we observe a conditional operation. Red(Blue) arrow denotes rotating axis where initial eletron spin state was $\ket{0}(\ket{1})$.

<!-- GIF: Conditional operation visualization -->
![](Meeting_251024/src/simulation/CPMG.gif)

---

# CPMG analysis

However, if the time for MW implementing $\pi$-pulse gate is not short enough, it can make additional effect other than described below. 

- Initial electron spin state 0(1): $e^{-i \phi \hat{I}\cdot\hat{\sigma^{i}}}$
   - if NV spin is 0-state: $U_0 = e^{-i \phi \hat{I}\cdot\hat{\sigma^{0}}} = e^{-i H_{0} \tau}e^{-i H_{1} 2\tau}e^{-i H_{0} \tau}$
   - if NV spin is 1-state: $U_1 = e^{-i \phi \hat{I}\cdot\hat{\sigma^{1}}} = e^{-i H_{1} \tau}e^{-i H_{0} 2\tau}e^{-i H_{1} \tau}$

If it is possible to describe one CPMG train such description, the unitary gate would take form of Block matrix:
$$
U^{\otimes \frac{N}{2}} = \begin{pmatrix}
U_0^{\otimes \frac{N}{2}} & 0 \\
0 & U_1^{\otimes \frac{N}{2}}
\end{pmatrix}
$$
where off-diagonal components go zero.

---

# CPMG analysis

However, if we calculate $U$ under un-neglibile MW time, off-diagonal components are non-zero and each block matrix is not unitary.

$$
U^{\otimes N} = \begin{pmatrix}
V_{00} & V_{01} \\
V_{10} & V_{11}
\end{pmatrix}
$$

Its operation on a state, 
$$
\ket{\psi} = \frac{1}{\sqrt{2}}(\ket{0} + \ket{1}) \otimes \ket{\phi}
$$
we have
$$
\ket{\psi} \rightarrow \frac{1}{\sqrt{2}}(
\ket{0}\otimes \underbrace{(V_{00} + V_{01})}_{=V_0}\ket{\phi}
+ 
\ket{1}\otimes \underbrace{(V_{11} + V_{10})}_{=V_1}\ket{\phi}
)
$$

---

# CPMG analysis

Both $V_{0}$ and $V_{1}$ are not unitary, to analysis, I used polor decomposition which is related to SVD.
$$
V = U P = (uv^{\dagger})(v \Sigma v^{\dagger})
$$
where $u, v$ are unitary operation can be derived from SVD, then $u$ is unitary operation, while $P$ is non-unitary and be regarded as quantum channel.
$$
P = \sum_i \lambda_{i} \ket{\lambda_{i}}\bra{\lambda_{i}}
$$
where $\lambda_{i}$ is singular values. So, we have
$$
\ket{\psi} \rightarrow \frac{1}{\sqrt{2}}(
\ket{0}\otimes U_{0}P_{0}\ket{\phi}
+ 
\ket{1}\otimes U_{1}P_{1}\ket{\phi}
)
$$

--- 

# CPMG analysis

We can analysis in two perspectives:
- How $\vec{\sigma_{0}}\cdot\vec{\sigma_{1}}$ with respect to $U_0$ and $U_1$ changes for $t_{\text{MW}}$ and $N$.
- How the singular value $\lambda_{0,1}$ change for for $t_{\text{MW}}$ and $N$. 

---

# CPMG analysis: $\vec{\sigma_{0}}\cdot\vec{\sigma_{1}}$

The following colormap shows how $\vec{\sigma_{0}}\cdot\vec{\sigma_{1}}$ value changes as $\tau$ and $\tau_{\text{MW}}$

$\tau$ shifts shorter as $\tau_{MW}$ increases. However, the amount of left-shift is keeping the total time for CPMG train. That is,


---

# CPMG analysis: $\vec{\sigma_{0}}\cdot\vec{\sigma_{1}}$

Varing $N$ and fixing $\tau_{\text{MW}}$, the value $\tau$ making total operation conditional, $\vec{\sigma_{0}}\cdot\vec{\sigma_{1}}=-1$, is unchanged. 

--- 

# CPMG alaysis: $\lambda_{i}$

As expected, $\tau_{\text{MW}}$ increases, the number of 