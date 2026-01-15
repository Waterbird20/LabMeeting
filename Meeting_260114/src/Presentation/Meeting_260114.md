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
New Year Plan
</div>
 
<div class="author">
Donghun Jung
</div>

<div class="date">
14 Jan 2026
</div>

<div class="organization">
Department of Physics, Sungkyunkwan University
<br>
Center for Quantum Technology QuiME Lab, Korea Institute of Science Technology
</div>

</div>

<div class="col-right">
<img src="media/images/PauleeLogo.png" style="max-width: 100%; height: auto; object-fit: contain;">
</div>

</div>

---

<!-- backgroundColor: white -->

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

- What I did since last lab meeting (25/11/20)
- What I'm doing now
   - Projects
   - Progress Update
- Dynamic Decoupling
   - CPMG Sequence
   - Finding $\tau$ Analytically
   - Conditional Operation
   - CPMG analysis
</div>

<div class="col-right-content">

- DDrf
   - DDrf Milestone
   - DDrf: Analytical Approach
   - DDrf Spectroscopy

- Slack + Notion

</div>
</div>

---

# What I did since last lab meeting (25/11/20)

- Assignments (~2 weeks)
- Grad Admission (25/12/17)
- Final Exam (~1 week) $\rightarrow$ Computer Network (A), Programming Language (A+)
- Wrote abstract for conference (26/01/07)
- Interview from UC Davis (26/01/14)
    - If accepted, Open house (26/02/22)
- And many other personal stuffs ...

---

# Projects

1. [**Ongoing**] Hybrid DDrf: Enhanced Electron Nuclear control leveraging hyperfine coupling and radio-frequency drive
2. [Holding] Observable Expectation Value Extraction (with Geonhee Kim)
3. [Holding] Entanglement Generation / Quantification + AI? (with Seongpyo Hong)
4. [Pending] Compressive QST
5. [Ongoing] Slack + Notion

---

# Progress Update

## Completed Tasks

- Analytical CPMG study: Effect of $\tau_{\text{MW}}$
      - Not that meaningful result though ...

## In Progress

- DDrf Spectroscopy
- Hybrid DDrf conditional gate strategy
- Slack + Notion

## Pending

- Observable Expectation Value Extraction (Theoretical background work completed)
- Entanglement Generation / Quantification
- Compressive QST

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

- When vectors $\sigma^0$ and $\sigma^1$ are anti-parallel, a conditional gate is achieved
- The directions are generally parallel, but become anti-parallel at specific $\tau$ values

**Note**: $\phi$ is the same regardless of whether the initial state is 0 or 1 in the CPMG train.

---

# Finding $\tau$ Analytically

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
![width:1100px](Meeting_251120/src/Presentation/media/CPMG_axis.png)


---

# CPMG analysis

- Initial electron spin state 0(1): $e^{-i \phi \hat{I}\cdot\hat{\sigma^{i}}}$
   - if NV spin is 0-state: $U_0 = e^{-i \phi \hat{I}\cdot\hat{\sigma^{0}}} = e^{-i H_{0} \tau}e^{-i H_{1} 2\tau}e^{-i H_{0} \tau}$
   - if NV spin is 1-state: $U_1 = e^{-i \phi \hat{I}\cdot\hat{\sigma^{1}}} = e^{-i H_{1} \tau}e^{-i H_{0} 2\tau}e^{-i H_{1} \tau}$

If one CPMG train can be described this way, the unitary gate takes the form of a block matrix:
$$
U^{ \frac{N}{2}} = \begin{pmatrix}
U_0^{ \frac{N}{2}} & 0 \\
0 & U_1^{ \frac{N}{2}}
\end{pmatrix}
$$
where off-diagonal components are zero.

However, if the time for the MW pulse implementing the $\pi$-pulse gate is not short enough, it can cause additional effects beyond those described below. 


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
\right)^{ \frac{N}{2}}
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

$\tau$ shifts to shorter values as $\tau_{MW}$ increases. However, the amount of this shift keeps the total time for the CPMG train constant. 


![width:1200px](Meeting_251120/src/Presentation/media/CPMG_imshow.png)


---

# CPMG analysis: $\vec{\sigma_{0}}\cdot\vec{\sigma_{1}}$

When varying $N$ while fixing $\tau_{\text{MW}}$, the value of $\tau$ that makes the total operation conditional, $\vec{\sigma_{0}}\cdot\vec{\sigma_{1}}=-1$, remains unchanged. 


![width:1200px](Meeting_251120/src/Presentation/media/CPMG_N.png)


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
As expected, when $\tau_{\text{MW}}$ and $N$ increase, the singular value $\lambda_{i}$ deviates from 1. 

![width:1200px](Meeting_251120/src/Presentation/media/CPMG_singularV.png)

---

# [Update] CPMG analysis: $\lambda_{i}$

In a CPMG sequence, we perform an $X$-basis measurement at the end of the sequence. Without considering the drift Hamiltonian during $\pi$-pulse time, we have
$$
\begin{align}
P_{x} &= \text{Tr} (\ket{+}\bra{+} \otimes I) \rho = \text{Tr} (\ket{+}\bra{+} \otimes I) U\rho_{0}U^{\dagger} \\
&= \frac{1}{4} \text{Tr} U_{0}\rho_{{}^{13}\text{C}} U_{0}^{\dagger} + \frac{1}{4} \text{Tr} U_{0}\rho_{{}^{13}\text{C}} U_{1}^{\dagger} + \frac{1}{4} \text{Tr} U_{1}\rho_{{}^{13}\text{C}} U_{0}^{\dagger} + \frac{1}{4} \text{Tr} U_{1}\rho_{{}^{13}\text{C}} U_{1}^{\dagger} \\
&= \frac{1}{2}\left( \frac{1}{2}\Re (\text{Tr} U_{0} U_{1}^{\dagger} ) + 1  \right) 
\end{align}
$$
where 
$$
\rho_{0} = \ket{+}\bra{+} \otimes \rho_{{}^{13}\text{C}}
$$
and the last line follows from $\rho_{{}^{13}\text{C}} = I$.

---

# [Update] CPMG analysis: $\lambda_{i}$

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

# [Update] CPMG analysis: $\lambda_{i}$

However, its effect does not seem to be critical.
$$
\begin{align}
\sigma_{00} + \sigma_{11} &\simeq 2 \\
\sigma_{01} + \sigma_{10} &= \Re (\text{Tr} V_{0}V_{1}^{\dagger})
\end{align}
$$
Then, 
$$
\begin{align}
\Re (\text{Tr} V_{0}V_{1}^{\dagger}) &= 2p_0 p_1 (\cos^2\phi + \sin^2\phi(\hat{\sigma}_0 \cdot \hat{\sigma}_1))(1 + r_0 r_1 (\hat{\mu}_0 \cdot \hat{\mu}_1))
\end{align}
$$
where $p_0 , p_1$ are average of eigenvalues, $\phi$ is conditional rotation angle, $\hat{\sigma}_i$ is rotation-axis vector related to $U_{i}$, and $\mu_{i}$ is the vector related to quantum channel $P_{i}$. So $r_0 r_1$ term is responsible for the non-negligible $\tau_{\text{MW}}$, but its order is at most $\mathcal{O}(10^{-2})$. 


---

# CPMG Fitting
<!-- TODO: Add Figure -->

![](Meeting_260114/src/Presentation/media/CPMG_experiment.png)

---

# CPMG Fitting

We measure the probability, $P_x$, that the initial state of electron spin is preserved.
$$
\begin{align} 
M &= \Re (\text{Tr } U_{0}U_{1}^{\dagger}) = 1 - \left( 1 - \hat{n}_{0}\cdot\hat{n}_{1}  \right)\sin^{2} \frac{N\phi}{2}  && \cos \phi &= \cos\alpha \cos\beta - \cos\beta \sin\alpha \sin\beta \\
P_x &= \frac{M e^{-\frac{2N\tau}{T_\alpha}}}{2} + \frac{1}{3} + \frac{e^{-\frac{-2N\tau}{T_{\beta}}}}{6} && 1 - \hat{n}_{0}\cdot\hat{n}_{1} &= \sin^2 \beta \frac{(1-\cos\alpha)(1 - \cos\beta)}{1 + \cos\phi}
\end{align}
$$
where 
$$
\begin{align}
\cos\beta &= \frac{\omega_L - \omega_{\parallel}}{\sqrt{(\omega_L - \omega_{\parallel})^2 + A_{\perp}^2}} && \alpha = \tau\sqrt{(\omega_L - \omega_{\parallel})^2 + A_{\perp}^2} \\
\sin\beta &= \frac{A_\perp}{\sqrt{(\omega_L - \omega_{\parallel})^2 + A_{\perp}^2}} && \beta = \omega_L \tau
\end{align}
$$

---

# CPMG Fitting

## Strategy

1. Using three values of $N$ for fitting data, while the remaining are used for validation data.
2. The cost function is based on Mean Squared Error (MSE). The cost function can be adjusted so that peak data could be regarded more crucially. For example, $\frac{(y_{\text{data}} - y_{\text{fit}})^2}{y_{\text{data}}}$, where $0<y_{\text{data}}<1$.
3. By assuming the number of $^{13}$C spins, perform fitting and check validation. It was expected that training error would decrease while validation error increases. In this process, one might choose the desired number of $^{13}$C spins. 

## Problem $\rightarrow$ Failed!
However, every time I performed fitting, I obtained different hyperfine coupling parameters. 


---

# DDrf Milestone

1. DDrf Spectroscopy (Experiment) <- (Eunsang Lee/ Donghun Jung)
$\rightarrow$ Output: $\omega_1$ of 13C spins

2. DDrf Spectroscopy (Experiment) 
$\rightarrow$ Output: $\sin\beta$, $A_{\perp}$

3. DDrf/DD comparative study (Theory)

4. Hybrid DDrf Gate (Theory) <- (Jiwon Jeon/ Donghun Jung)
$\rightarrow$ Further faster conditional gate implementation

---

# DDrf: Analytical Approach
As the nuclear spin quantization axis is dependent on the electron spin state, the Hamiltonian becomes
$$
\begin{align}
H &= \ket{0}\bra{0}\otimes H_0 + \ket{1}\bra{1}\otimes H_1 \\
H_{0} &= \omega_{0} I_z + 2\Omega_{\text{RF}}\cos(\omega_{\text{RF}}t +\phi) I_{x} \\
H_{1} &= \omega_{1} \tilde{I}_z + 2\Omega_{\text{RF}}\cos\beta \cos(\omega_{\text{RF}}t +\phi) \tilde{I}_{x} + 2\Omega_{\text{RF}}\sin\beta\cos(\omega_{\text{RF}}t +\phi) \tilde{I}_{z} 
\end{align}
$$
where 
$$
\begin{align}
\cos\beta   &= \frac{\omega_0 - A_{\perp}}{\omega_1} & \tilde{I}_z &= \cos\beta I_z + \sin \beta I_x \\
\sin\beta   &= \frac{A_{\perp}}{\omega_1} & \tilde{I}_x &= \cos\beta I_x - \sin \beta I_z .
\end{align}
$$

---

# DDrf: Analytical Approach

Two rotating frames are used with respect to electron spin state;
$$
\begin{align}
R_{0} (t) &= e^{-i \omega_{\text{RF}} t I_z} &
R_{1} (t) &= e^{-i \omega_{\text{RF}} t \tilde{I}_z} 
\end{align}
$$
In rotating frame, each Hamiltonian becomes:
$$
\begin{align}
H_0 \rightarrow H_{0}^{\prime} &= R_{0}(t) (H_{0} - \omega_{\text{RF}}I_z) R_0 (t)^{\dagger} &\\ &= (\omega_0 - \omega_{text{RF}}) I_z + \Omega_{RF} (\cos\phi I_x + \sin\phi I_y ) &\\
H_1 \rightarrow H_{1}^{\prime} &= R_{1}(t) (H_{1} - \omega_{\text{RF}}I_z) R_1 (t)^{\dagger} &\\ &= (\omega_1 - \omega_{text{RF}}) \tilde{I}_z + \Omega_{RF} \cos\beta (\cos\phi \tilde{I}_x + \sin\phi \tilde{I}_y ) & \\ &= (\omega_1 - \omega_{\text{RF}})(\cos\beta I_z + \sin\beta I_x) + \Omega_{RF} \cos\beta (\cos\beta\cos\phi I_x + \sin\phi I_y - \sin\beta\cos\phi I_z ) 
\end{align}
$$
In each rotating frame, time evolution can be calculated readily; $e^{-i H_{0(1)}t}$.

--- 

# DDrf: Analytical Approach

Then, throughout full time evolution, the unitary operation can be calculated as:
$$
\begin{align}
U    =& \ket{0}\bra{0}\otimes U_{0} + \ket{1}\bra{1}\otimes U_{1}\\
U_{0}=& R_{0}(4N\tau)^{\dagger} e^{-i H_{0}^{\prime}\tau} R_{0}((2N-1)\tau)R_{1}((2N-1)\tau)^{\dagger}e^{-i H_{1}^{\prime}2\tau}R_{1}((2N-3)\tau)R_{0}((2N-3)\tau)^{\dagger}e^{-i H_{0}^{\prime}\tau}\cdots \\
& \cdots e^{-i H_{0}^{\prime}\tau}R_{0}(3\tau)R_{1}(3\tau)^{\dagger}e^{-i H_{1}^{\prime}2\tau}R_{1}(\tau)R_{0}(\tau)^{\dagger}e^{-i H_{0}^{\prime}\tau}R_{0}(0)\\
U_{1}=& R_{1}(4N\tau)^{\dagger} e^{-i H_{1}^{\prime}\tau} R_{1}((2N-1)\tau)R_{0}((2N-1)\tau)^{\dagger}e^{-i H_{0}^{\prime}2\tau}R_{0}((2N-3)\tau)R_{1}((2N-3)\tau)^{\dagger}e^{-i H_{1}^{\prime}\tau}\cdots \\
& \cdots e^{-i H_{1}^{\prime}\tau}R_{1}(3\tau)R_{0}(3\tau)^{\dagger}e^{-i H_{0}^{\prime}2\tau}R_{0}(\tau)R_{1}(\tau)^{\dagger}e^{-i H_{1}^{\prime}\tau}R_{1}(0)
\end{align}
$$

It is worth mentioning that
- $\Omega_{\text{RF}} \rightarrow 0 , \omega_{\text{RF}} = \omega_1$: CPMG at a certain $\tau \simeq \frac{(2k-1)\pi}{2\omega_0 + A_{\parallel}}$
- $\beta\rightarrow 0, \omega_{\text{RF}} = \omega_1 > \omega_0$: DDrf(2019)
- $\Omega_{\text{RF}} \rightarrow \frac{\Omega_{\text{RF}}}{\cos\beta}$: Jiwon's idea
- $\tau = \frac{(2k-1)\pi}{2\omega_0 + A_{\parallel}}$: Hybrid DDrf

---

# DDrf Spectroscopy

<!-- Main Idea: (M+1)/2 -->
The paper, `Physical Review X 9.3 (2019): 031045`, suggested that the DDrf gate provides the additional benefit that spins with small $A_{\perp}$ can also be detected. 

Procedure:
1. $\frac{\pi}{2}$-pulse rotates electron spin to $\ket{+}$.
2. DDrf Gate with fixed $N$ and $\tau$.
3. $\frac{\pi}{2}$-pulse is applied to electron spin with varying phase $\phi$.

Then, at the resonant frequency, peaks are observed. 
$$
\omega_{\text{RF}} = \omega_{1} + \frac{2\pi m}{\tau}
$$
where $m$ is an integer. 

---

# DDrf Spectroscopy

<style scoped> 
.container{
   display: flex;
   align-items: center;
   width: 100%;
   height: 100%;
}
.col-left-content{
   flex: 0 0 25%;
   padding-right: 2rem;
   padding-bottom: 3rem;
   color: #000000;
}

.col-right-content{
   flex: 0 0 70%;
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

<div class="container">
<div class="col-left-content">
<img src="Meeting_260114/src/Presentation/media/Taminiau.png" style="max-width: 100%; height: auto; object-fit: contain;">

</div>
<div class="col-right-content">
<img src="Meeting_260114/src/Presentation/media/DDrf_simulation.png" style="max-width: 100%; height: auto; object-fit: contain;">

<br>
<em>
</em>

</div>

---

# DDrf Spectroscopy

## Problems

1. Peaks were not observed for all resonant frequencies.
2. The peaks corresponding to $\omega_{+1}$ and $\omega_{-1}$ must be different.

## Potential Issues

1. Incorrect coding.
2. Additional implementation required for sweeping phase $\phi$ of the electron spin. 

---

# Slack + Notion

## Problems

1. Data stored in Slack chat is not retained for more than 90 days.
2. As we are not using Notion or other tools, the management of lab resources is contingent on each individual's ability.
$\rightarrow$ We have craved a systemic and integrated management system.
Based on the following clone coding, I'm working on building a Slack alternative and integrating Notion features. We will potentially be able to record lab notes here. Of course, before this, we must reach an agreement on writing lab notes. 

![](Meeting_260114/src/Presentation/media/CodeWithAntonio.png)

---

# Slack + Notion

![](Meeting_260114/src/Presentation/media/slack_demo.png)

---

# Future Work

1. [**Ongoing**] Hybrid DDrf: Enhanced Electron Nuclear control leveraging hyperfine coupling and radio-frequency drive
2. [Holding] Observable Expectation Value Extraction (with Geonhee Kim)
3. [Holding] Entanglement Generation / Quantification + AI? (with Seongpyo Hong)
4. [Pending] Compressive QST
5. [Ongoing] Slack + Notion


## I'm going to be a "FAT" person! 