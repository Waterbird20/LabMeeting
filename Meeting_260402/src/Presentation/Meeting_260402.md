---

title       : Update
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
Update
</div>

<div class="author">
Donghun Jung
</div>

<div class="date">
02 Apr 2026
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
   flex: 0 0 45%;
   padding-right: 0.5rem;
   padding-left: 0.5rem;
   padding-bottom: 6.5rem;
}

.col-right-content{
   margin-left: 0px;
   flex: 0 0 55%;
   display: flex;
   align-items: center;
   justify-content: center;
   padding-bottom: 5rem;
}

li {
   font-size: 0.85rem;
}

</style>

<div class="container">
<div class="col-left-content">

1. **Entanglement**

2. **CGLMP Test**
   - Already (almost) done!

3. **Distributed Quantum Machine Learning**
   - Dataset: 9-dimensional clustered binary classification
   - Data Encoding: Ising-type feature map
   - Potential References

4. **Compressive QST**

</div>

<div class="col-right-content">

5. **Post-Selection**
   - 

6. **DDrf**
   - 

</div>
</div>

---

# Entanglement

---

# CGLMP Test

---

# Distributed Quantum Machine Learning


---

# Compressive QST

---

# Post-Selection

1. Breif introduction to Measurement Scenerio
2. Analytical Trial to find maximum QFI
3. Numerical optimization; separable filter
4. Numerical optimization; entangled filter

---

# Post-Selection

In estimating Quantum Fisher Information, 

$
\frac{\partial \rho}{\partial B} = \ket{\partial_B \psi}\bra{\psi} + \ket{\psi}\bra{\partial_B \psi}
$

$
\ket{\partial_B \psi} = \frac{\partial}{\partial B} e^{-i \mathcal{H} t}\ket{\psi_0} = -i G t\ket{\psi}$

where
$
G = \frac{\partial}{\partial B}\mathcal{H}
$

$
\mathcal{L}= 2i t_s \left[ \rho, G\right]
$

$
F_Q = \text{Tr} \rho L^2 = 4 t_s^2 ( \text{Tr} G^2  - (\text{Tr}G)^2 ) = 4\gamma^2 t_s^2 
$

---

# Post-Selection

For a certain density matrix, which only depends on $\ket{00} , \ket{11}$ basis,
$
\rho = \ket{\psi}\bra{\psi} =
\begin{pmatrix}
a^2 & ab\eta e^{i\phi}\\
ab\eta e^{-i\phi} & b^2
\end{pmatrix}
$
where 
$
\ket{\psi} = a\ket{00} + b\ket{11}
$
under dephasing channel, 
$
\eta = e^{-2\tau} = e^{\left(\frac{t_s}{T_2^*}\right)^p}
$
and phase accumulation 
$
\phi = \theta + 2\gamma B t_s
$
Under this subsystem of $\ket{00} , \ket{11}$, we can treat it as single qubit system so that density matrix can be decomposed into 
$
\vec{r} = (2ab\eta\cos\phi , 2ab\eta\sin\phi , a^2 - b^2)
$
$
F_Q^B = \left|\partial_B r \right|^2 = 16\gamma^2 t_s^2 e^{-4\tau} a^2 b^2 \leq 4\gamma^2 t_s^2 e^{-4\tau}
$

Note that in sinlge qubit system the maximum QFI is 4\gamma^2 t_s^2 e^{-2\tau}. 
Given Quantum Fisher Information is addictive, two-qubit separable state is (twice) better than preparing entangled state.

---

# Post-Selection

In this sense, we have thought that adding Post-selection procedure might be advantageous. Intuitively speaking, the information of $B$-field is encoded in phase and via post-selection we may discard unnecessary information to ancillary qubit(or the other energy level). 

Post-Selection process is not described by unitary operation but Kraus operation and it can change QFI. While analytical approach is challenging (although I keep trying), we tried numerical optimization to find optimal post-selection strength, state preparation, measurement basis and the corresponding maximum QFI. 

---

# Post-Selection

We prepared the following circuit ansatz.

In order to prepare arbitrary two-qubit state, we employed one CNOT gate.

The system interact with (purturbed) magnetic field $\delta B$ during sensing time $t_s$ and noise channel is imbedded. 

Then perform Post Selection and measurement. 


---

# Post-Selection

Given Learning curve, optimization seems saturated well. 
However, the value it saturated was twice of QFI of Post-selected single-qubit system. 

---

# Post-Selection

The prepared state turns out to be separable state, by investivating negativity of prepared state.

---

# Post-Selection

GH suggested to enable entangled filter and entangled basis measurement. So we modified circuit ansatz. Two-qubit gate is added as if we are performing Post-selection on entangled basis.
$
K_{\text{eff}} = U (K_1 \otimes K_2) U^{\dagger}
$
At here, we employed KAK decomposition(equivalent using 3 CNOT gate per two-qubit gate) to explore full SU(4) space.

---

# Post-Selection

Interestingly, optimized QFI exceeds one of Post-selected separable state with order twice. 

---

# Post-Selection

But the prepared state was separable state but entanglement was retrieved at the end of Post-selection. Note that the strength of entanglement was not the stongest where negativity of Bell state is 0.5. 

---

# Post-Selection

Further the filter strength also becomes asymmetric, and even one of post-selection strength $\gamma$ became almost 1, meaning that strong measurement. 

---

# [TODO] Post-Selection

- Message became clear that under dephasing nosie, entanglement was vulnerable to noise rather than resource(well-known Heisenberg limit). Post-selection onto entangled basis can retrieve entanglement and enable gain in sensing. 
- Further analysis required: prepared state, filter strength and prepared Post-selection/measurement basis. 
- I need to write draft. I presume Apr 09 is target deadline. 
- Based on obtained insight, I'm trying to build analytical analysis framework.  

---

# DDrf

1. Comparative Study between CPMG and DDrf (Theory/Numeric)
   - **[Completed] Poster work $\leftarrow$ (Hun)**
2. Enhanced (Hybrid) DDrf gate (Theory/Numeric)
   - [Completed] Tried many ideas... $\leftarrow$ (J. J)
   - [Pending?] Alternating $\Omega_{\text{RF}}$ for odd- and even-numbered pulses $\leftarrow$ (J. J)
   - [Ongoing] Draw phase diagram (J. J)
   - **[Ongoing] Analytical Study; Conditional Rotation Angle $\leftarrow$(Hun)**
3. DDrf Spectroscopy (Experiment)
   - [Completed] Numerical Simulation based on Taminiau Paper $\leftarrow$ (Hun)
   - [Ongoing] Experiemnt $\leftarrow$ (Dr. Leee)
   - **[Ongoing] Numerical Simulation; side-peak problem $\leftarrow$(Hun)**
4. Multi-qubit Control (Numeric/Experiment)
   - [Pending] **$\omega_0$ control $\leftarrow$(Hun)**

---

# DDrf



---

# Numerical Simulation based on Taminiau Paper
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

# Numerical Simulation based on Taminiau Paper

Two rotating frames are used with respect to the electron spin state:
$$
\begin{align}
R_{0} (t) &= e^{i \omega_{\text{RF}} t I_z} &
R_{1} (t) &= e^{i \omega_{\text{RF}} t \tilde{I}_z} 
\end{align}
$$
In the rotating frame, each Hamiltonian becomes:
$$
\begin{align}
H_0 \rightarrow H_{0}^{\prime} &= R_{0}(t) (H_{0} - \omega_{\text{RF}}I_z) R_0 (t)^{\dagger} &\\ &= (\omega_0 - \omega_{\text{RF}}) I_z + \Omega_{RF} (\cos\phi I_x + \sin\phi I_y ) &\\
H_1 \rightarrow H_{1}^{\prime} &= R_{1}(t) (H_{1} - \omega_{\text{RF}}I_z) R_1 (t)^{\dagger} &\\ &= (\omega_1 - \omega_{\text{RF}}) \tilde{I}_z + \Omega_{RF} \cos\beta (\cos\phi \tilde{I}_x + \sin\phi \tilde{I}_y ) & \\ &= (\omega_1 - \omega_{\text{RF}})(\cos\beta I_z + \sin\beta I_x) + \Omega_{RF} \cos\beta (\cos\beta\cos\phi I_x + \sin\phi I_y - \sin\beta\cos\phi I_z ) 
\end{align}
$$
In each rotating frame, the time evolution can be readily calculated as $e^{-i H_{0(1)}t}$.

--- 

# Numerical Simulation based on Taminiau Paper

Then, over the full time evolution, the unitary operation can be calculated as:
$$
\begin{align}
U    =& \ket{0}\bra{0}\otimes U_{0} + \ket{1}\bra{1}\otimes U_{1}\\
U_{0}=& \textcolor{red}{R_{0}(4N\tau)^{\dagger}} 
e^{-i H_{0}^{\prime}\tau} 
\textcolor{red}{R_{0}((2N-1)\tau)R_{1}((2N-1)\tau)^{\dagger}}
e^{-i H_{1}^{\prime}2\tau}
\textcolor{red}{R_{1}((2N-3)\tau)R_{0}((2N-3)\tau)^{\dagger}}
e^{-i H_{0}^{\prime}\tau}\cdots \\
& \cdots e^{-i H_{0}^{\prime}\tau}
\textcolor{red}{R_{0}(3\tau)R_{1}(3\tau)^{\dagger}}
e^{-i H_{1}^{\prime}2\tau}
\textcolor{red}{R_{1}(\tau)R_{0}(\tau)^{\dagger}}
e^{-i H_{0}^{\prime}\tau}
\textcolor{red}{R_{0}(0)}\\
U_{1}=& \textcolor{red}{R_{1}(4N\tau)^{\dagger}} 
e^{-i H_{1}^{\prime}\tau} 
\textcolor{red}{R_{1}((2N-1)\tau)R_{0}((2N-1)\tau)^{\dagger}}
e^{-i H_{0}^{\prime}2\tau}
\textcolor{red}{R_{0}((2N-3)\tau)R_{1}((2N-3)\tau)^{\dagger}}
e^{-i H_{1}^{\prime}\tau}\cdots \\
& \cdots e^{-i H_{1}^{\prime}\tau}
\textcolor{red}{R_{1}(3\tau)R_{0}(3\tau)^{\dagger}}
e^{-i H_{0}^{\prime}2\tau}
\textcolor{red}{R_{0}(\tau)R_{1}(\tau)^{\dagger}}
e^{-i H_{1}^{\prime}\tau}\textcolor{red}{R_{1}(0)}
\end{align}
$$

This approach is exact except an assumption that the time of MW pulse to change electron state is small. It is worth mentioning that
- $\Omega_{\text{RF}} \rightarrow 0 , \omega_{\text{RF}} = \omega_1$: CPMG at a certain $\tau \simeq \frac{(2k-1)\pi}{2\omega_0 + A_{\parallel}}$
- $\beta\rightarrow 0, \omega_{\text{RF}} = \omega_1 > \omega_0$: DDrf(2019)
- $\Omega_{\text{RF}} \rightarrow \frac{\Omega_{\text{RF}}}{\cos\beta}$: Jiwon's idea
- $\tau = \frac{(2k-1)\pi}{2\omega_0 + A_{\parallel}}$: Hybrid DDrf

---

# Numerical Simulation based on Taminiau Paper

The paper [Phys. Rev. X **9**, 031045 (2019)] showed that the DDrf gate offers the additional benefit of detecting spins with small $A_{\perp}$.

<!-- Todo: Add circuit figure -->

Procedure:
1. $\frac{\pi}{2}$-pulse rotates electron spin to $\ket{+}$.
2. DDrf Gate with fixed $N$ and $\tau$, resulting in $U= \ket{0}\bra{0}\otimes U_{0} + \ket{1}\bra{1}\otimes U_{1}$
3. $\frac{\pi}{2}$-pulse is applied to electron spin with varying phase $\phi$.
   - In our experiment, we measure $P_x$, the projection onto $\ket{+}$ ($\phi=\frac{\pi}{2}$).

![height:200px](Meeting_260219/src/Presentation/media/spectroscopy_sequence.png)

---

# Numerical Simulation based on Taminiau Paper

Results:
1. the expectation value becomes $P_x = \frac{1}{2} + \frac{1}{4}\Re(\text{Tr}U_0 U_1^{\dagger})$
2. Extended to an $N$-qubit simulation, $P_x = \frac{1}{2} + \frac{1}{2^{N+1}}\Re(\text{Tr}U_0 U_1^{\dagger})$ where
$$\text{Tr} U_0 U_1^{\dagger} = \prod_{i=1}^N \text{Tr}U_0^i {U_1^i}^{\dagger} .$$
3. In the Taminiau paper, the amplitude is $A=\frac{1}{2^{N+1}}\left|\text{Tr} U_0 U_1^{\dagger} \right|$.
4. At the resonant frequency ($\omega_{\text{RF}} = \omega_{1}$), peaks are observed.
5. Peaks also appear at off-resonant conditions ($\omega_{\text{RF}} = \omega_{1} + \frac{2\pi m}{\tau}$, $m\in\mathbb{Z}$) due to the same phase accumulation.

---

# Numerical Simulation based on Taminiau Paper


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
<!-- TODO: Edit Taminiau Figure -->
<div class="container">
<div class="col-left-content">
<img src="Meeting_260219/src/Presentation/media/Taminiau.png" style="max-width: 100%; height: 80%; object-fit: contain;">

</div>
<div class="col-right-content">
<img src="Meeting_260219/src/Presentation/media/DDrf_simulation_wrong.png" style="max-width: 100%; height: 80%; object-fit: contain;">

<br>
<em>
</em>

</div>

---

# Numerical Simulation based on Taminiau Paper


<img src="Meeting_260219/src/Presentation/media/Taminiau_spectroscopy.png" style="max-width: 100%; height: 25%; object-fit: contain;">


<img src="Meeting_260219/src/Presentation/media/Reproduce.png" style="max-width: 100%; height: 45%; object-fit: contain;">

---

# Numerical Simulation based on Taminiau Paper


<img src="Meeting_260219/src/Presentation/media/Taminiau_spectroscopy.png" style="max-width: 100%; height: 25%; object-fit: contain;">


<img src="Meeting_260219/src/Presentation/media/Reproduce_ext.png" style="max-width: 100%; height: 45%; object-fit: contain;">



---

--- 

# Numerical Simulation based on Taminiau Paper

Given the following time evolution

$$
\begin{align}
U    =& \ket{0}\bra{0}\otimes U_{0} + \ket{1}\bra{1}\otimes U_{1}\\
U_{0}=& \textcolor{red}{R_{0}(4N\tau)^{\dagger}} 
e^{-i H_{0}^{\prime}\tau} 
\textcolor{red}{R_{0}((2N-1)\tau)R_{1}((2N-1)\tau)^{\dagger}}
e^{-i H_{1}^{\prime}2\tau}
\textcolor{red}{R_{1}((2N-3)\tau)R_{0}((2N-3)\tau)^{\dagger}}
e^{-i H_{0}^{\prime}\tau}\cdots \\
& \cdots e^{-i H_{0}^{\prime}\tau}
\textcolor{red}{R_{0}(3\tau)R_{1}(3\tau)^{\dagger}}
e^{-i H_{1}^{\prime}2\tau}
\textcolor{red}{R_{1}(\tau)R_{0}(\tau)^{\dagger}}
e^{-i H_{0}^{\prime}\tau}
\textcolor{red}{R_{0}(0)}\\
U_{1}=& \textcolor{red}{R_{1}(4N\tau)^{\dagger}} 
e^{-i H_{1}^{\prime}\tau} 
\textcolor{red}{R_{1}((2N-1)\tau)R_{0}((2N-1)\tau)^{\dagger}}
e^{-i H_{0}^{\prime}2\tau}
\textcolor{red}{R_{0}((2N-3)\tau)R_{1}((2N-3)\tau)^{\dagger}}
e^{-i H_{1}^{\prime}\tau}\cdots \\
& \cdots e^{-i H_{1}^{\prime}\tau}
\textcolor{red}{R_{1}(3\tau)R_{0}(3\tau)^{\dagger}}
e^{-i H_{0}^{\prime}2\tau}
\textcolor{red}{R_{0}(\tau)R_{1}(\tau)^{\dagger}}
e^{-i H_{1}^{\prime}\tau}\textcolor{red}{R_{1}(0)}
\end{align}
$$

- $\Omega_{\text{RF}} \rightarrow 0 , \omega_{\text{RF}} = \omega_1$: CPMG at a certain $\tau \simeq \frac{(2k-1)\pi}{2\omega_0 + A_{\parallel}}$
- $\beta\rightarrow 0, \omega_{\text{RF}} = \omega_1 > \omega_0$: DDrf(2019)

Then, change of frame is responsible for CPMG effect at a certain $\tau$, additional rf driving resides in the $H_0$ and $H_1$. 

---

# DDrf

Note that DDrf sequence is recursive build up MW and RF pulses. Each build up(train), $V^{(k)}$ makes unitary operation. Simply speaking, $V^{(k)} = \mathcal{T}\left(\exp -i\int \mathcal{H}dt \right)$.
$$
V^{(k)} = \ket{0}\bra{0} \otimes V_0^{(k)} + \ket{1}\bra{1} \otimes V_1^{(k)}
$$
Then, the total unitary operation becomes: 
$$
U = \ket{0}\bra{0} \otimes \prod_{k=1}^{N/2} V_0^{(k)} + \ket{1}\bra{1} \otimes \prod_{k=1}^{N/2} V_1^{(k)}.
$$
This equation is itself equivalent to the previous one, but imply this strategy. The idea is that any unitary operation can be written as $\textcolor{}{e^{-iA}}$ for arbitrary Hermitian operator $A$. At the same time, it can be decomposed into $\textcolor{}{e^{-iA_1}e^{-iA_2}}$ but this decomposition is non-trivial and not unique. This means we can choose a favorable form to calculate the equation above without fully calculating previous equation. 

---

# DDrf

For example, in $\beta \rightarrow 0$, $\omega_{\text{RF}}=\omega_1$ case (Taminiau, 2019),
$$
\begin{align}
V_0^{k} &= e^{-iH_0 \tau}e^{-iH_1 2\tau}e^{-iH_0 \tau} \\
        &= e^{-i\delta_0 \tau I_z} e^{-i 2\Omega\tau \hat{\phi}_{2k}} e^{-i\delta_0 \tau I_z} \\
        &= \left[e^{-i\delta_0 \tau I_z} \textcolor{red}{e^{-i\delta_0 \tau I_z}}\right] \left[\textcolor{red}{e^{+i\delta_0 \tau I_z}}e^{-i 2\Omega\tau \hat{\phi}_{2k}} e^{-i\delta_0 \tau I_z}\right] \\
        &= e^{-i\delta_0 2\tau I_z} e^{-2\Omega\tau \hat{\phi}_{2k}^{\prime}}
\end{align}
$$
where $\hat{\phi} = \cos\phi I_x + \sin\phi I_y$ and $\hat\phi_{2k}^{\prime} = \cos(\phi - \delta\tau) I_x + \sin(\phi - \delta\tau) I_y$.
In a successive product, we can keep using such trick.
$$
\begin{align}
V_0^{k}V_0^{(k-1)} &= e^{-i\delta_0 2\tau I_z} e^{-2\Omega\tau \hat{\phi}_{2k}^{\prime}} e^{-i\delta_0 2\tau I_z} e^{-2\Omega\tau \hat{\phi}_{2k-2}^{\prime}} \\
&= \left[e^{-i\delta_0 2\tau I_z} \textcolor{red}{e^{-i\delta_0 2\tau I_z}}\right] \left[\textcolor{red}{e^{+i\delta_0 2\tau I_z}}e^{-2\Omega\tau \hat{\phi}_{2k}^{\prime}} e^{-i\delta_0 2\tau I_z}\right] e^{-2\Omega\tau \hat{\phi}_{2k-2}^{\prime}} \\
&= e^{-i\delta_0 4\tau I_z} e^{-2\Omega\tau \hat{\phi}_{2k}^{\prime\prime}}e^{-2\Omega\tau \hat{\phi}_{2k-2}^{\prime}}
\end{align}
$$
By setting, $\phi_{2k} = (2k-1)\delta\tau + \phi_0$, we can make $\phi_{2k}$ aligned to the same axis, enabling successive rotation build up. 

---

# DDrf

Now in the exact time evo equation, we can set $V_0$ and $V_1$ for each.
$$
\begin{align}
V_0^{(k)} &= e^{-iH_0\tau} R_0((4k-3)\tau)R_1^{\dagger}((4k-3)\tau)e^{-iH_1 2\tau}R_1((4k-1)\tau)R_0^{\dagger}e^{-iH_0\tau}\\
V_1^{(k)} &= e^{-iH_1\tau} R_1((4k-3)\tau)R_0^{\dagger}((4k-3)\tau)e^{-iH_0 2\tau}R_0((4k-1)\tau)R_1^{\dagger}e^{-iH_1\tau}
\end{align}
$$

To observe CPMG effect by choosing proper $\tau$, I employed Taylor-expansion-like approach. That is,
$$
\begin{align}
V_0^{(k)} &\simeq  \left. V_0^{(k)}\right|_{\beta=0} + \beta \frac{\partial}{\partial\beta}\left. V_0^{(k)}\right|_{\beta=0} + \mathcal{O}(\beta^2)\\
V_1^{(k)} &\simeq  \left. V_1^{(k)}\right|_{\beta=0} + \beta \frac{\partial}{\partial\beta}\left. V_1^{(k)}\right|_{\beta=0} + \mathcal{O}(\beta^2)\\
\end{align}
$$
Under such operation, we expect to find a form $e^{-i\alpha I_z} e^{-i \theta_{\beta}\hat\sigma_{\beta}}$ where $\alpha, \theta_{\beta}, \hat\sigma_{\beta}$ are arbitrary parameter and axis and $\beta$ denotes $\beta$ dependence. Further we can verify this approximation by testing its limit onto $\beta\rightarrow0$ and $\Omega\rightarrow0$, for each corresponds to Taminiau DDrf and CPMG. 

---

# DDrf

Without detailed calculation (but I assumed $\delta_1 \ll \Omega \ll \delta_0$), I obtained
$$
\begin{align}
V_0^{(k)} =& e^{-i2\delta_0\tau I_z} \left[
  \color{orange}\underbrace{\left(\color{black}\cos\Omega\tau +\frac{\beta}{2}\sin\Omega\tau\cos(\phi_{2k} + (4k-2)\omega_{\text{RF}}\tau)\color{orange}\right)}_{=\cos\theta}\color{black} I \right. \\
  &+\left. -i\color{orange}\left(\color{black}
  \sin\Omega\tau \hat{\phi}^{\prime}+\beta 
  \color{cyan}\left( \color{black}\cos\omega_{\text{RF}}\tau\sin\Omega\tau\cos(\phi_{2k} + (4k-2)\omega_{\text{RF}}\tau)I_z \right.\right.\right. \\
  & \left.\left.\left.
   -\sin\omega_{\text{RF}}\tau \cos\Omega\tau(\cos((4k-2)\omega_{\text{RF}}\tau)I_x - \sin((4k-2)\omega_{\text{RF}}\tau)I_y )\color{black} \color{cyan}\right)\color{orange} \right)\right]
\end{align}
$$
$$
\begin{align}
V_1^{(k)} =& e^{-i2\delta_0\tau I_z} \left[ e^{-i H_1(\phi_{2k+1}^{\prime})\tau} e^{-i H_1(\phi_{2k-1})\tau} + i\beta\sin\delta_0 \tau e^{-i H_1(\phi_{2k+1}^{\prime})\tau} \hat{\chi} e^{-i H_1(\phi_{2k-1})\tau}
  \right]
\end{align}
$$
where $\hat\chi =\cos((4k-2)\omega\tau +\delta_0 \tau)I_x -\sin((4k-2)\omega\tau +\delta_0 \tau)I_y$. Since $V_1^{(k)}$ term has two phase, I did not proceeds to calculate whole. Optimistically, we might choose appropriate $\tau$ and $\phi_{k}$ such that $\theta > \Omega\tau$, rotation axis antiparallel, and successive conditional rotation build up.

---

# Numerical Simulation; side-peak problem
In the Taminiau paper, given $\tau$, $\Omega_{\text{RF}}$, $N$, and $\omega_{\text{RF}}=\omega_1$, the DDrf gate operation is
$$
U = \ket{0}\bra{0} \otimes U_0 + \ket{1}\bra{1} \otimes U_1
$$
where:
$$
\begin{align}
U_0 &= R_z (N(\omega_L - \omega_1)\tau) \cdot R_{\phi} (N\Omega_{\text{RF}}\tau) \\
U_1 &= R_z (N(\omega_L - \omega_1)\tau) \cdot R_{\phi} (-N\Omega_{\text{RF}}\tau)
\end{align}
$$
where $R_{z}(\theta)=e^{-i\theta I_z}$ and $R_\phi (\theta) = e^{-i\theta (\cos\phi I_x + \sin\phi I_y)}$. 


**[Observation]** The operation becomes unconditional when $N\Omega_{\text{RF}}\tau = 2\pi$, suggesting a flat spectroscopy signal.

---

# Numerical Simulation; side-peak problem

<img src="Meeting_260219/src/Presentation/media/unconditional.png" style="max-width: 100%; height: 80%; object-fit: contain;">

---

# Numerical Simulation; side-peak problem

<img src="Meeting_260219/src/Presentation/media/unconditional_focus.png" style="max-width: 100%; height: 80%; object-fit: contain;">

---

# Numerical Simulation; side-peak problem

<!-- TODO: Side peak is also detected in large $\Omega_{\text{RF}}$ -->
<img src="Meeting_260219/src/Presentation/media/sidepeak.png" style="max-width: 100%; height: 80%; object-fit: contain;">

---

# Numerical Simulation; side-peak problem

**[Analysis]** DDrf spectroscopy can be understood from the perspective of Rabi oscillations. Although an analytical form of $\Omega_{\text{eff}}$ is difficult to derive (though $\Omega_{\text{eff}} \propto \Omega_{\text{RF}}$), when the RF frequency is detuned from resonance by $\delta = \omega_1 - \omega_{\text{RF}}$, the generalized Rabi frequency is
$$
\Omega_{\text{gen}} = \sqrt{\delta^2 + \Omega_{\text{eff}}^2}
$$
The signal might take form:
$$
P_x \simeq 1 - \underbrace{\frac{\Omega_{\text{eff}}}{\Omega_{\text{gen}}}}_{\text{Lorentzian envelope}}\underbrace{\sin^2 \frac{\Omega_{\text{gen}}  2N\tau}{2}}_{\text{oscillation}}
$$
According to this analysis, the bandwidth scales as $\sim \frac{1}{2N\tau}$.


---

# Numerical Simulation; side-peak problem

**[Suggestion]** Apply pulse shaping (e.g., a Gaussian envelope).
$\rightarrow$ $\Omega_{\text{RF}}$ becomes time-dependent. 

The time-evolution method used here ***assumes*** $\Omega_{\text{RF}}$ is time-independent, as required by the RWA.
$$
\begin{align}
U    =& \ket{0}\bra{0}\otimes U_{0} + \ket{1}\bra{1}\otimes U_{1}\\
U_{0}=& \textcolor{red}{R_{0}(4N\tau)^{\dagger}} 
e^{-i H_{0}^{\prime}\tau} 
\textcolor{red}{R_{0}((2N-1)\tau)R_{1}((2N-1)\tau)^{\dagger}}
e^{-i H_{1}^{\prime}2\tau}
\textcolor{red}{R_{1}((2N-3)\tau)R_{0}((2N-3)\tau)^{\dagger}}
e^{-i H_{0}^{\prime}\tau}\cdots \\
& \cdots e^{-i H_{0}^{\prime}\tau}
\textcolor{red}{R_{0}(3\tau)R_{1}(3\tau)^{\dagger}}
e^{-i H_{1}^{\prime}2\tau}
\textcolor{red}{R_{1}(\tau)R_{0}(\tau)^{\dagger}}
e^{-i H_{0}^{\prime}\tau}
\textcolor{red}{R_{0}(0)}\\
U_{1}=& \textcolor{red}{R_{1}(4N\tau)^{\dagger}} 
e^{-i H_{1}^{\prime}\tau} 
\textcolor{red}{R_{1}((2N-1)\tau)R_{0}((2N-1)\tau)^{\dagger}}
e^{-i H_{0}^{\prime}2\tau}
\textcolor{red}{R_{0}((2N-3)\tau)R_{1}((2N-3)\tau)^{\dagger}}
e^{-i H_{1}^{\prime}\tau}\cdots \\
& \cdots e^{-i H_{1}^{\prime}\tau}
\textcolor{red}{R_{1}(3\tau)R_{0}(3\tau)^{\dagger}}
e^{-i H_{0}^{\prime}2\tau}
\textcolor{red}{R_{0}(\tau)R_{1}(\tau)^{\dagger}}
e^{-i H_{1}^{\prime}\tau}R_{1}(0)
\end{align}
$$

Without this scheme, one would need to solve the Schrödinger equation directly, which takes tens of minutes per frequency point. For spectroscopy, I typically sweep hundreds to thousands of $\omega_{\text{RF}}$ values.

---

# Post-selection: Architecture / File Structure

```
CFIsimulation/
├── main.py                     # Entry point: parse config, build circuit, run optimization
├── config.yaml                 # User-facing configuration file (all tunable parameters)
├── pyproject.toml              # Project metadata and dependencies
├── requirements.txt            # Pip dependency list
├── core/
│   ├── circuit.py              # Circuit class: assembles layers into a PennyLane QNode
│   ├── layers.py               # Layer: 
│   │                             Initialization, Entangler, RamseyZ, PostSelection
│   ├── trainer.py              # Trainer class: 
│   │                             CFI cost function, optimization, QFI benchmark
│   └── utils/
│       ├── arguments.py        # Dataclass for typed configuration 
│       │                         (circuitarguments, optarguments, etc.)
│       ├── customparser.py     # YAML config parser → dataclass instances
│       └── utils.py            # Hamiltonians, dephasing, density matrix plotting, 
│                                 file I/O helpers
├── dmplots/                    # Output directory for density matrix visualizations
└── paramplots/                 # Output directory for optimized parameter plots
```

---

# Post-selection: Density Matrix Analysis

**[ToDo]** Entanglement investigation (two-qubit system) $\leftarrow$ Isn't it already done?

The density matrix is optimized via the following procedure.
1. Initialization
2. Entangler
3. Ramsey under dephasing noise
4. (Optional) Post-selction

After optimization, we obtain the optimal sensing time $t_s$.

The Entangler parameters are optimized to maximize the classical Fisher information (CFI). The quantum Fisher information (QFI) serves as the upper bound for the CFI.

---

# Post-selection: Density Matrix Analysis

The circuit structure looks like the following:

<img src="Meeting_260219/src/Presentation/media/circuit.png" style="max-width: 100%; height: 80%; object-fit: contain;">

---

# Post-selection: Density Matrix Analysis

Presumably, this scheme is based on [*Preparation of metrological states in dipolar-interacting spin systems*, npj Quantum Inf **8**, 150 (2022)].

<img src="Meeting_260219/src/Presentation/media/metrological_states.png" style="max-width: 100%; height: 60%; object-fit: contain;">


---

# Distributed Quantum Machine Learning: Architecture / File Structure

```
DQML/
├── main.py                    # Training entry point
├── autoscript.py              # Automated hyperparameter sweep runner
├── drawcircuit.py             # Circuit diagram generator
├── config.yaml                # Primary experiment configuration
├── dummy.yaml                 # Config template used by autoscript.py
│
├── core/                      # Core quantum circuit implementation
│   ├── qcnn.py                #   Top-level model + scheme builder
│   ├── blocks.py              #   Mid-level: groups of layers into blocks
│   ├── layers.py              #   Low-level: individual quantum operations
│   └── utils/
│       ├── arguments.py       #     Dataclass definitions for config
│       ├── customparser.py    #     YAML config file parser
│       └── utils.py           #     Training helpers, plotting, I/O
├── dataset/
│   ├── Dataset4               # Pre-generated dataset (pickle, ~655 KB)
│   ├── ClusterDataset.ipynb   # Notebook that generates the dataset
│   └── test.py                # Quick dataset loading sanity check
├── pyproject.toml             # Project metadata and dependencies
├── requirements.txt           # Pip-compatible dependency list
└── uv.lock                    # Locked dependency versions
```


---

# Distributed Quantum Machine Learning: Data Set

The dataset is a **synthetic 9-dimensional clustered binary classification** problem:

1. **Define the cluster centers:** From all `2^9 = 512` vertices of the hypercube `{-π/4, +π/4}^9`, randomly select 128 cluster centers. Half are assigned label 0, half label 1.

2. **Generate data points:** For each cluster, generate 64 data points (`8192 / 128 = 64`) by:
   - Sampling a point uniformly from a 9-dimensional ball of radius `π/4`
   - Shifting it to the cluster center (subtracting the vertex coordinates)

3. **Result:** 8192 data points, each a 9-dimensional vector with values roughly in `[-π/2, π/2]`, with binary labels.

---

# Distributed Quantum Machine Learning: Data Set

<img src="Meeting_260219/src/Presentation/media/Fig1.png" style="max-width: 100%; height: 60%; object-fit: contain;">

---

# Distributed Quantum Machine Learning: Data Set

<img src="Meeting_260219/src/Presentation/media/Fig2.png" style="max-width: 100%; height: 60%; object-fit: contain;">

---

# Distributed Quantum Machine Learning: Data Set

<img src="Meeting_260219/src/Presentation/media/Fig3.png" style="max-width: 100%; height: 60%; object-fit: contain;">

---

# Distributed Quantum Machine Learning: Data Encoding

The embedding uses an **Ising-type feature map**, which encodes data into both single-qubit rotations and two-qubit entangling interactions:

```
For 3 qubits (one processor), depth=1:

     ┌───┐ ┌────────┐
q0 ──┤ H ├─┤ RZ(x0) ├──■─────────■──
     └───┘ └────────┘  │         │
     ┌───┐ ┌────────┐  │         │
q1 ──┤ H ├─┤ RZ(x1) ├──■────■────■──
     └───┘ └────────┘       │
     ┌───┐ ┌────────┐       │
q2 ──┤ H ├─┤ RZ(x2) ├───────■───────
     └───┘ └────────┘

IsingZZ angle = 0.5 * (π - x_i) * (π - x_{i+1})
```

---

# Distributed Quantum Machine Learning: Potential References

<img src="Meeting_260219/src/Presentation/media/Distributed_quantum_machine_learning_via_classical_communication.png" style="max-width: 100%; max-height: 35%; object-fit: contain;">
<img src="Meeting_260219/src/Presentation/media/Quantum_machine_learning_for_image_classification.png" style="max-width: 100%; max-height: 35%; object-fit: contain;">




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
