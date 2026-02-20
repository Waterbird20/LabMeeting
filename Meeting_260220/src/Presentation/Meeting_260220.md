---

title       : Coarse-Grained Measurement
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
Coarse-Grained Measurement
</div>

<div class="author">
Donghun Jung
</div>

<div class="date">
20 Feb 2026
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

# Schematic of the NV${}^{-}$ electronic structure

![](Meeting_260220/src/Presentation/media/laser_curve.png)


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


![width:400px](Meeting_260220/src/Presentation/media/Figure1.png)


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


![width:400px](Meeting_260220/src/Presentation/media/Figure1.png)


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
