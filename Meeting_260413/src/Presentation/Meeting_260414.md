---

title       : KIST Interview
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
   color: rgb(228, 65, 38);
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
KIST Interview
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
<img src="./media/KIST_CI.png" style="max-width: 100%; height: auto; object-fit: contain;">
</div>

</div>

---

<!-- backgroundColor: white -->

# Coarse-grained quantum state tomography with optimal POVM construction

<style scoped> 
.container{
   display: flex;
   align-items: center;
   width: 100%;
   height: 100%;
}
.col-left-content{
   flex: 0 0 35%;
   padding-right: 1rem;
   padding-bottom: 7rem;
   color: #000000;
}

.col-right-content{
   flex: 0 0 50%;
   padding-bottom: 7rem;

   color: #000000;
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

section h1 {
  font-size: 24pt;
}

p {
   font-size: 18pt
}

</style>

<div class="container">
<div class="col-left-content">


![width:400px](./media/Figure1.png)


</div>

<div class="col-right-content">


$\eta_0$, $\eta_1$:  the probability of **failing** to detect a signal from the $\ket{0}$ and $\ket{1}$ state.
<br>
$$
\begin{align}
\Pi_{0} &= \eta_{0} \ket{0}\bra{0} + \eta_{1}\ket{1}\bra{1} && \text{no detection} \\
\Pi_{1} &= (1-\eta_{0}) \ket{0}\bra{0} + (1-\eta_{1})\ket{1}\bra{1} && \text{a detection event}
\end{align}
$$
<br>

Extending this framework to $N$-qubit system, the signal detection probatility becomes:

$$
M_{\text{CG}} = I^{\otimes N} - \Pi_{0}^{\otimes N}
$$
resulting in two positive semidefinite observable operators $\{ M_{\text{CG}}, I^{\otimes N} - M_{\text{CG}} \}$. 

By implementing unitary gates $G^{k} (\hat{\theta})$ via parameterized quantum circuits, an extended set of CG POVM operators, $\Omega^{k} = {G^{k}}^{\dagger}(\hat{\theta})M_{\text{CG}}G^{k}(\hat{\theta})$ can be constructed. 


</div>


---

# Coarse-grained quantum state tomography with optimal POVM construction

<style scoped> 
.container{
   display: flex;
   align-items: center;
   width: 100%;
   height: 100%;
}
.col-left-content{
   flex: 0 0 35%;
   padding-right: 2rem;
   padding-bottom: 3rem;
   color: #000000;
}

.col-right-content{
   flex: 0 0 50%;
   padding-bottom: 3rem;

   color: #000000;
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

section h1 {
  font-size: 24pt;
}

p {
   font-size: 18pt
}
</style>

<div class="container">
<div class="col-left-content">


![width:400px](./media/Figure1.png)


</div>

<div class="col-right-content">


**Idea**: Uniformly distributed Measurement Operators across Hilbert space.

Uniformness of the set of measurement operators can be represented by von Neumann entropy of Gram matrix
$$
S = -\text{Tr} (\Pi^{\prime} \ln \Pi^{\prime}) = -\sum_{i}^{} \lambda_{i}^{\prime} \ln \lambda_{i}^{\prime}
$$
where Gram matrix is defined as each element is the inner product of the POVM bases $\Pi_{ij} = \text{Tr} \Omega^{i}\Omega^{j}$, and normalized  $\Pi^{\prime} = \Pi / \text{Tr} \Pi$ so that sum of eigenvalues becomes 1. 

To maximize von Neumann entropy $S$, we iteratively update the circuit parameters $\hat{\theta}$. That is, we solved $\arg\max_{\hat{\theta}} S$.

</div>





---

# Coarse-grained quantum state tomography with optimal POVM construction

<style scoped> 

section h1 {
  font-size: 24pt;
}

p {
   font-size: 18pt;
   color: #000000;
}
</style>

![](./media/Figure2.png)

(a) The CG POVM based QST process for a two-qubit system is illustrated, where quantum state tomography is performed after an arbitrary quantum operation $\mathcal{O}$ and subsequent projection basis gates $G^{k}(\theta)$. 
(b) Higher von Neumann entropy of the CG POVM set corresponds to reduced infidelity, demonstrating robustness against the statistical noise. 
(c) CG POVM sets with higher entropy provide more consistent reconstruction results, demonstrating their state-independent performance across a variety of quantum states.

---

# Enhanced electron nuclear control leveraging hyperfine coupling and radio-frequency drive

<style scoped> 

.container{
   align-items: center;
   width: 100%;
   height: 100%;
}
.col-left-content{
   flex: 0 0 45%;
   padding-right: 2rem;
   color: #000000;
}

.col-right-content{
   flex: 0 0 45%;

   color: #000000;
}

.temp {
   color: #000000;
   align-items: left;
   text-align: left;
   padding-bottom: 10rem;
}

img[alt~="rightside"]{
   position: absolute;
   top: 10rem;
   right: 2rem;
}

em {
   font-size: 0.7rem;
}

section h1 {
  font-size: 20pt;
}

p {
   font-size: 18pt
}
</style>

<div class="container">
   <div class="col-left-content">

   **Dynamic Decoupling (DD)** sequences, consisting of multi-pulse control on the NV spin, have been developed to achieve entangling gates with selected target nuclear spins while reducing decoherence from the broader spin bath. 

   An additional selective phase-controlled **radio-frequency (rf)** driving on nuclear spins, referred as DDrf gate, enables elaborated control of individual spins.

   </div>

   <div class="col-right-content">

   ![width:500px](./media/DDrf_pulse.png)

   </div>

</div>

<div class="temp">

$$
\begin{align}
\mathcal{H} =& \underbrace{\gamma_c B_z I_z^i + A_{||}^i S_z I_z^i + A_{\perp} S_z I_x^i}_{\text{Drift Hamiltonian}} + \underbrace{\frac{1}{\sqrt{2}} \Omega_{\text{MW}} S_x}_{\text{NV electron spin control}} + \underbrace{2\Omega_{\text{rf}}I_x}_{\text{$^{13}$C spin control}}  \\
\rightarrow \mathcal{H} =& \ket{0}\bra{0} \otimes \omega_0 I_z + 
\ket{-1}\bra{-1} \otimes \left( \omega_L I_z -  A_{||}I_z - A_{\perp}I_x  \right) \\
\rightarrow \mathcal{H} =& \ket{0}\bra{0} \otimes H_0 + 
\ket{-1}\bra{-1} \otimes H_1 
\end{align} 
$$

</div>

---

# Enhanced magnetic field sensing employing Post-Selection technique

We prepared the following circuit ansatz.

![width:1050px](./media/ent_ps.svg)

- State Preparation (Red): To prepare an arbitrary two-qubit state, we employ one CNOT gate.

- Sensing (Orange): The system interacts with the (perturbed) magnetic field $\delta B$ during sensing time $t_s$, and the noise channel is embedded. 

- Post-Selection and Measurement (Purple): Unitary operations $U$, $V$ change the post-selection basis, described by Kraus operation, and measurement basis. 

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
   color: rgb(228, 65, 38);
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
