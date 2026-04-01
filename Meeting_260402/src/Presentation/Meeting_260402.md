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
- $\beta\rightarrow 0$, (then $I_z = \tilde{I}_z$), $\omega_{\text{RF}} = \omega_1$: DDrf(2019)

Then, change of frame is responsible for CPMG effect at a certain $\tau$, additional rf driving resides in the $H_0$ and $H_1$. But to find $U_{(0,1)} = e^{-i\theta\hat\sigma_{(0,1)}}$ is not intriguing...

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


**[Observation]** The operation becomes unconditional when $N\Omega_{\text{RF}}\tau = 2\pi$, a flat spectroscopy signal was expected. 

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

---

# DDrf

Side peak problem arises near resonant peak $\omega_{\text{RF}}\simeq \omega_1$. Let's retrun to rotating frame Hamiltonian. 

$$
\begin{align}
H_0 \rightarrow H_{0}^{\prime} &= R_{0}(t) (H_{0} - \omega_{\text{RF}}I_z) R_0 (t)^{\dagger} &\\ &= \delta_0 I_z + \Omega_{RF} (\cos\phi I_x + \sin\phi I_y ) &\\
H_1 \rightarrow H_{1}^{\prime} &= R_{1}(t) (H_{1} - \omega_{\text{RF}}I_z) R_1 (t)^{\dagger} &\\ &= \delta_1 \tilde{I}_z + \Omega_{RF} \cos\beta (\cos\phi \tilde{I}_x + \sin\phi \tilde{I}_y ) 
\end{align}
$$
where $(\omega_{0(1)} - \omega_{\text{RF}})$. At here lets assume $\hat{\tilde{I}} = \hat{I}$ for simplicity, that is $A_\perp \simeq 0$ and $\beta\rightarrow 0$.

Previously, we have assumed $\delta_1 \ll \Omega \ll \delta_0$. To see dutuned effect, lets make $\delta_1$ small but not negligible.

---

# DDrf

This is a great question. The nice thing is that the algebraic structure of the DDrf sequence survives intact when $\delta_1 \neq 0$ — the corrections enter in a clean, geometrically transparent way.

## Setup with $\delta_1 \neq 0$

With $H_1(\phi) = \delta_1 I_z + \Omega\hat\phi\cdot\vec{I}$, the propagator $e^{-iH_1 t}$ is now a rotation about a **tilted** axis. Define

$$\Omega_{\mathrm{eff}} = \sqrt{\Omega^2 + \delta_1^2}, \qquad \sin\beta = \frac{\delta_1}{\Omega_{\mathrm{eff}}}, \qquad \cos\beta = \frac{\Omega}{\Omega_{\mathrm{eff}}}$$

so that

$$H_1(\phi) = \Omega_{\mathrm{eff}}\,\hat{n}(\phi)\cdot\vec{I}, \qquad \hat{n}(\phi) = (\cos\beta\cos\phi,\;\cos\beta\sin\phi,\;\sin\beta).$$

The axis $\hat{n}(\phi)$ is tilted out of the $xy$-plane by angle $\beta$.

---

# DDrf


## Per-cell propagators

The conjugation trick still works, since $e^{i\alpha I_z}(\hat{n}\cdot\vec{I})e^{-i\alpha I_z}$ only rotates the azimuthal angle by $-\alpha$ and leaves the $z$-component ($\sin\beta$) invariant. So exactly the same manipulations you wrote give:

$$V_0^{(k)} = e^{-i2\delta_0\tau I_z}\;e^{-i\Omega_{\mathrm{eff}}\,2\tau\;\hat{n}(\phi_{2k}')\cdot\vec{I}}$$

with $\phi_{2k}' = \phi_{2k} - \delta_0\tau = (2k-2)\delta_0\tau$, and

$$V_1^{(k)} = e^{-i2\delta_0\tau I_z}\;e^{-i\Omega_{\mathrm{eff}}\,2\tau\;\hat{n}(\phi_{2k-1})\cdot\vec{I}}$$

with $\phi_{2k-1} = (2k-2)\delta_0\tau + \pi$. The phase protocol ensures the two RF pulses within each $V_1$ cell collapse onto the same tilted axis, just as before.

---

# DDrf


## Telescoping still works

Composing over $N/2$ cells, each factor of $e^{-i2\delta_0\tau I_z}$ shifts subsequent azimuthal angles by $2\delta_0\tau$, which exactly compensates the $k$-dependent phase stepping. The tilt angle $\beta$ is $k$-independent, so it passes through the telescoping unchanged:

$$V_0^{\mathrm{tot}} = e^{-iN\delta_0\tau I_z}\;e^{-i\Omega_{\mathrm{eff}}\,N\tau\;\hat{n}_0\cdot\vec{I}}, \qquad \hat{n}_0 = (\cos\beta,\;0,\;\sin\beta)$$

$$V_1^{\mathrm{tot}} = e^{-iN\delta_0\tau I_z}\;e^{-i\Omega_{\mathrm{eff}}\,N\tau\;\hat{m}_0\cdot\vec{I}}, \qquad \hat{m}_0 = (-\cos\beta,\;0,\;\sin\beta)$$

---

# DDrf


## What changes: the conditional gate structure

The common factor $e^{-iN\delta_0\tau I_z}$ is the same as the $\delta_1=0$ case, so the interesting physics is in the rotations. Comparing the two axes:

$$\hat{n}_0 = \cos\beta\,\hat{x} + \sin\beta\,\hat{z}, \qquad \hat{m}_0 = -\cos\beta\,\hat{x} + \sin\beta\,\hat{z}$$

In the ideal case $\beta=0$, these are antiparallel ($\hat{n}_0 = -\hat{m}_0$), giving a perfect conditional rotation: the nuclear spin rotates in opposite senses depending on the electron state. With $\beta\neq 0$, the axes share a common $\hat{z}$ projection but differ only in the transverse part. This means the total gate decomposes (conceptually) into:

- **Conditional part** (electron-state-dependent): rotation about $\pm\hat{x}$, with effective angle $\Omega_{\mathrm{eff}}N\tau\cos\beta = \Omega N\tau$. This is actually unchanged from the $\delta_1=0$ case in terms of the transverse rotation angle.
- **Unconditional part** (same for both electron states): rotation about $\hat{z}$, with effective angle $\Omega_{\mathrm{eff}}N\tau\sin\beta = \delta_1 N\tau$. This is a parasitic $I_z$ rotation that does not create entanglement.


---

# DDrf

## Telescoping the product

Define shorthand:

$$\mathcal{Z} \equiv e^{-i2\delta_0\tau I_z}, \qquad R_\alpha^{(k)} \equiv e^{-i\Omega_{\mathrm{eff}}\,2\tau\;\hat{n}(\varphi_k)\cdot\vec{I}}$$

where $\hat{n}(\varphi) = (\cos\beta\cos\varphi,\;\cos\beta\sin\varphi,\;\sin\beta)$ with $\sin\beta = \delta_1/\Omega_{\mathrm{eff}}$.

With the standard phase protocol, the per-cell propagators reduce to:

$$V_0^{(k)} = \mathcal{Z}\cdot R_0^{(k)}, \qquad \varphi_k^{(0)} = (2k-2)\delta_0\tau$$

$$V_1^{(k)} = \mathcal{Z}\cdot R_1^{(k)}, \qquad \varphi_k^{(1)} = (2k-2)\delta_0\tau + \pi$$

---

# DDrf


### Key identity for telescoping

Since $e^{i\alpha I_z}(\hat{n}(\varphi)\cdot\vec{I})e^{-i\alpha I_z} = \hat{n}(\varphi - \alpha)\cdot\vec{I}$ (the tilt angle $\beta$ is invariant under $z$-rotations), we have

$$R^{(k)}\cdot\mathcal{Z} = \mathcal{Z}\cdot R^{(k-1)}$$

because commuting $\mathcal{Z}$ through shifts the azimuthal phase by $-2\delta_0\tau$, which is exactly the step between consecutive $k$ labels.

---

# DDrf


### Inductive product

For $V_0$, writing the time-ordered product (cell 1 rightmost):

$$\prod_{k=1}^{N/2} V_0^{(k)} = \bigl[\mathcal{Z}\,R_0^{(N/2)}\bigr]\cdots\bigl[\mathcal{Z}\,R_0^{(2)}\bigr]\bigl[\mathcal{Z}\,R_0^{(1)}\bigr]$$

**By induction:**

$$\boxed{\prod_{k=1}^{N/2} V_0^{(k)} = e^{-iN\delta_0\tau\,I_z}\;\cdot\;e^{-i\Omega_{\mathrm{eff}}\,N\tau\;\hat{n}(0)\cdot\vec{I}}}$$

$$\boxed{\prod_{k=1}^{N/2} V_1^{(k)} = e^{-iN\delta_0\tau\,I_z}\;\cdot\;e^{-i\Omega_{\mathrm{eff}}\,N\tau\;\hat{n}(\pi)\cdot\vec{I}}}$$

where

$$\hat{n}(0) = (\cos\beta,\;0,\;\sin\beta), \qquad \hat{n}(\pi) = (-\cos\beta,\;0,\;\sin\beta)$$

---

# DDrf


## Full unitary

Defining $\alpha \equiv N\delta_0\tau$ and $\theta \equiv \Omega_{\mathrm{eff}}\,N\tau$:

$$U = \bigl(\mathbb{1}_e \otimes e^{-i\alpha I_z}\bigr)\cdot\Bigl[\lvert 0\rangle\langle 0\rvert \otimes e^{-i\theta\,\hat{n}(0)\cdot\vec{I}} \;+\; \lvert 1\rangle\langle 1\rvert \otimes e^{-i\theta\,\hat{n}(\pi)\cdot\vec{I}}\Bigr]$$

The first factor is an **unconditional** nuclear $z$-rotation (identical for both electron states). The second factor is the **conditional gate**.

---

# DDrf


Substituting $\Omega_{\mathrm{eff}}^2 = \Omega^2 + \delta_1^2$ and $\sin^2\beta = \delta_1^2/(\Omega^2 + \delta_1^2)$:

$$
\mathrm{tr}(U_0 U_1^\dagger) = 2\cos\!\Big(\!\sqrt{\Omega^2 + \delta_1^2}\;N\tau\Big) + \frac{4\delta_1^2}{\Omega^2 + \delta_1^2}\,\sin^2\!\!\left(\frac{\sqrt{\Omega^2+\delta_1^2}\;N\tau}{2}\right)
$$

---

# DDrf

Using $2\cos\theta = 2 - 4\sin^2(\theta/2)$:

$$\frac{1}{2}\mathrm{tr}(U_0 U_1^\dagger) = 1 - \frac{2\Omega^2}{\Omega^2 + \delta_1^2}\,\sin^2\!\!\left(\frac{\sqrt{\Omega^2+\delta_1^2}\;N\tau}{2}\right)$$

This is exactly the **detuned Rabi formula**. 


---

# DDrf

## Step 1: Per-cell propagators with $\Omega_k = \Omega f(k)$

The phase protocol $\phi_{2k} = (2k-1)\delta_0\tau$, $\phi_{2k-1} = (2k-2)\delta_0\tau + \pi$ depends only on $\delta_0\tau$ and $k$, **not on $\Omega$**. So it is unchanged by apodization. The conjugation trick $e^{i\alpha I_z}(\hat{n}\cdot\vec{I})e^{-i\alpha I_z}$ rotates the azimuthal angle while leaving the tilt angle invariant. Since apodization only modifies the tilt angle $\beta_k$ and the effective frequency $\Omega_{\mathrm{eff},k}$, **the phase protocol still collapses all azimuthal angles to a common value**. Verified.

The per-cell propagators are:

$$V_0^{(k)} = e^{-i2\delta_0\tau I_z}\;e^{-i\theta_k\,\hat{n}_k(0)\cdot\vec{I}}, \qquad V_1^{(k)} = e^{-i2\delta_0\tau I_z}\;e^{-i\theta_k\,\hat{n}_k(\pi)\cdot\vec{I}}$$

with $k$-dependent quantities:

$$\theta_k = 2\Omega_{\mathrm{eff},k}\,\tau, \qquad \Omega_{\mathrm{eff},k} = \sqrt{\Omega_k^2 + \delta_1^2}, \qquad \sin\beta_k = \frac{\delta_1}{\Omega_{\mathrm{eff},k}}, \qquad \cos\beta_k = \frac{\Omega_k}{\Omega_{\mathrm{eff},k}}$$

$$\hat{n}_k(0) = (\cos\beta_k,\;0,\;\sin\beta_k), \qquad \hat{n}_k(\pi) = (-\cos\beta_k,\;0,\;\sin\beta_k)$$

Note: the tilt angle $\beta_k$ now **varies with $k$** because $\Omega_k$ does.

---

## Step 2: Telescoped product

The same telescoping identity holds: commuting $\mathcal{Z} = e^{-i2\delta_0\tau I_z}$ through $R_k$ shifts the azimuthal angle by $-2\delta_0\tau$ while preserving $\beta_k$ and $\theta_k$. By induction (exactly as before):

$$\prod_{k=1}^{N/2} V_0^{(k)} = e^{-iN\delta_0\tau I_z}\;\prod_{k=N/2}^{1} e^{-i\theta_k\,\hat{n}_k(0)\cdot\vec{I}}$$

$$\prod_{k=1}^{N/2} V_1^{(k)} = e^{-iN\delta_0\tau I_z}\;\prod_{k=N/2}^{1} e^{-i\theta_k\,\hat{n}_k(\pi)\cdot\vec{I}}$$

**Crucial difference from the constant-$\Omega$ case**: the ordered product does **not** collapse into a single rotation, because $\hat{n}_k(0)$ has $k$-dependent tilt $\beta_k$, and rotations about different axes in the $xz$-plane do not commute.

**Exception — $\delta_1 = 0$**: all $\beta_k = 0$, all axes align ($\hat{x}$ or $-\hat{x}$), and the products do collapse:

$$\prod_{k=N/2}^{1} e^{-i\cdot 2\Omega_k\tau\,I_x} = e^{-i\Theta\,I_x}, \qquad \Theta \equiv 2\Omega\tau\sum_{k=1}^{N/2} f(k)$$

---

## Step 3: Full unitary

$$\boxed{U = \bigl(\mathbb{1}_e\otimes e^{-iN\delta_0\tau I_z}\bigr)\cdot\left[\lvert 0\rangle\langle 0\rvert\otimes\prod_{k=N/2}^{1}e^{-i\theta_k\hat{n}_k(0)\cdot\vec{I}}\;+\;\lvert 1\rangle\langle 1\rvert\otimes\prod_{k=N/2}^{1}e^{-i\theta_k\hat{n}_k(\pi)\cdot\vec{I}}\right]}$$

For $\delta_1 = 0$, this simplifies to:

$$U\big|_{\delta_1=0} = \bigl(\mathbb{1}_e\otimes e^{-iN\delta_0\tau I_z}\bigr)\cdot\left[\lvert 0\rangle\langle 0\rvert\otimes e^{-i\Theta I_x}\;+\;\lvert 1\rangle\langle 1\rvert\otimes e^{+i\Theta I_x}\right]$$

---

## Step 4: $\mathrm{tr}(U_0 U_1^\dagger)$

The common $e^{-iN\delta_0\tau I_z}$ cancels as before:

$$\mathrm{tr}(U_0 U_1^\dagger) = \mathrm{tr}\!\left(\prod_{k=N/2}^{1}e^{-i\theta_k\hat{n}_k(0)\cdot\vec{I}}\;\cdot\;\prod_{k=1}^{N/2}e^{+i\theta_k\hat{n}_k(\pi)\cdot\vec{I}}\right)$$


---


### Case $\delta_1 = 0$: clean windowed result

All $V_0$ rotations are about $+\hat{x}$, all $V_1$ rotations about $-\hat{x}$:

$$\mathrm{tr}(U_0 U_1^\dagger)\big|_{\delta_1=0} = \mathrm{tr}\!\left(e^{-i\Theta I_x}\cdot e^{-i\Theta I_x}\right) = \mathrm{tr}\!\left(e^{-i2\Theta I_x}\right) = 2\cos\Theta$$

$$\boxed{\mathrm{tr}(U_0 U_1^\dagger)\big|_{\delta_1=0} = 2\cos\!\left(2\Omega\tau\sum_{k=1}^{N/2}f(k)\right)}$$

The $\pi$-gate condition becomes $2\Omega\tau\sum_k f(k) = \pi$, which just rescales the required $\Omega$ to compensate for the reduced total pulse area of the window.

---


### Case $\delta_1 \neq 0$: perturbative result

For small $\delta_1$, define the zeroth-order (unperturbed, $\delta_1=0$) rotation and the perturbation:

$$e^{-i\theta_k\hat{n}_k(0)\cdot\vec{I}} = e^{-i(2\Omega_k\tau\,I_x\;+\;2\delta_1\tau\,I_z)\;+\;\mathcal{O}(\delta_1^2)}$$

where I used $\theta_k\cos\beta_k = 2\Omega_k\tau + \mathcal{O}(\delta_1^2)$ and $\theta_k\sin\beta_k = 2\delta_1\tau$ (exactly, independent of $k$).

Going to the **interaction picture** with respect to the $I_x$ rotations, define the cumulative angle after cell $j$:

$$\Phi_j \equiv 2\Omega\tau\sum_{m=1}^{j}f(m)$$

The $I_z$ perturbation in cell $k$ is rotated into the frame where cells $1,\ldots,k-1$ have already been applied:

$$I_z \;\longrightarrow\; \cos\Phi_{k-1}\;I_z \;-\; \sin\Phi_{k-1}\;I_y$$

---


Summing the first-order Magnus contribution over all cells:

$$\prod_{k=N/2}^{1}e^{-i\theta_k\hat{n}_k(0)\cdot\vec{I}} \;\approx\; e^{-i\Theta I_x}\;\exp\!\left(-i2\delta_1\tau\sum_{k=1}^{N/2}\bigl[\cos\Phi_{k-1}\;I_z - \sin\Phi_{k-1}\;I_y\bigr]\right)$$

Similarly for the $V_1$ branch (with $I_x \to -I_x$). When we form $U_0 U_1^\dagger$, the leading $e^{-i\Theta I_x}$ terms combine into $e^{-i2\Theta I_x}$, and the first-order corrections produce:

$$\boxed{\frac{1}{2}\mathrm{tr}(U_0 U_1^\dagger) \;\approx\; \cos\Theta \;-\; \frac{2\delta_1^2\tau^2}{\sin^2(\Theta/2)}\;\sin^2(\Theta/2)\;\left|\sum_{k=1}^{N/2}f_k\,e^{i\Phi_{k-1}}\right|^2\;\cdot(\ldots)}$$

This is getting notationally heavy, so let me give the cleaner **physical result**. 

---


Defining the normalized window:

$$F(\delta_1) \equiv \sum_{k=1}^{N/2} f(k)\,e^{i\Phi_{k-1}(\delta_1)}$$

the spectral response near resonance takes the form of a **windowed discrete Fourier transform**:

$$\frac{1}{2}\mathrm{tr}(U_0 U_1^\dagger) \approx -1 + \text{const}\times\left|\frac{F(\delta_1)}{F(0)}\right|^2 \cdot \delta_1^2$$

where $F(0) = \sum_k f(k)$ is just the normalization. The key result:

- **Rectangular window** $f(k) = 1$: $|F|^2$ gives the **Fejér kernel** (sinc-squared), recovering the detuned Rabi lineshape with prominent sidelobes.
- **Hanning/Hamming/Blackman**: the sidelobes are suppressed at the cost of a broader main lobe, exactly as in classical spectral analysis.

---


The FWHM broadens by a factor that depends on the window (roughly $\times 1.5$ for Hanning, $\times 1.7$ for Blackman), but the sidelobe rejection improves dramatically (from $-13$ dB for rectangular to $-31$ dB for Hanning to $-58$ dB for Blackman). This is the standard **resolution–sidelobe tradeoff** from signal processing, now appearing in the DDrf gate's spectral selectivity.

---

The root cause is clear: the rotation axis $\hat{n}_k(0) = (\cos\beta_k, 0, \sin\beta_k)$ lives in the $xz$-plane, and $\beta_k = \arctan(\delta_1/\Omega_k)$. Whenever both $\delta_1$ and the $k$-dependence of $\Omega_k$ are present simultaneously, $\beta_k$ wanders from cell to cell. Rotations about different axes in the $xz$-plane don't commute, so the product $\prod_k e^{-i\theta_k \hat{n}_k \cdot \vec{I}}$ cannot be collapsed into a single exponential.

## How badly is it broken?

The mismatch between consecutive axes is

$$\Delta\beta_k \equiv \beta_{k+1} - \beta_k \approx -\frac{\delta_1}{\Omega^2}\,\Omega\,\Delta f_k$$

where $\Delta f_k = f(k+1) - f(k)$. So the non-commutativity enters at order $\delta_1 \cdot \Delta f_k$, which is small when either $\delta_1$ is small or the window varies slowly. This is reassuring: smooth windows (Hanning, Blackman) have small $\Delta f_k$ everywhere except near the edges where $f(k) \approx 0$ anyway, so the damage is doubly suppressed.


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

# Numerical Simulation; side-peak problem

Nevertheless, if we assume $\Omega_{\text{RF}}$ changes slow enough compared to $\omega_{\text{rf}}$, the error comes from RWA. Next problem is to evaluate, $e^{-i H_{0,1}^{\prime} \tau}$ where 
$$
\Omega_{\text{RF}} = \Omega_0 e^{-\frac{(t-t_k)^2}{2\sigma^2}}.
$$

To avoid solving Schrodinger equation everytime, I employed Magnus expansion!

---

# Numerical Simulation; side-peak problem

## Magnus expansion

The solution of the differentical equation $Y^\prime = A(t)Y$ with initial condition $Y(0)=Y_0$ can be written as $Y(t) = \exp(\Omega(t))$ with $\Omega(t)$ defined by
$$
\Omega^{\prime} = d\exp_{\Omega}^{-1}A(t), \Omega(0) = 0
$$
where 
$$
d\exp_{\Omega}^{-1}(A) = \sum_{k=0}^{\inf} \frac{B_k}{k!}\text{ad}_\Omega^k A. 
$$

In our case, we have differential equation
$$
\frac{\partial}{\partial t}U(t) = -i H(t) U(t). 
$$

---

# Numerical Simulation; side-peak problem

Generally, we try to find solution in the form of series
$$
\Omega(t) = \sum_{n=1}^\inf \Omega_n (t)
$$
In such form, there is well known solution.
$$
\begin{align}
\Omega_1 (t) &= \int_0^t dt_1 A_1 \\
\Omega_2 (t) &= \frac{1}{2} \int_0^t dt_1 \int_0^{t_1} dt_2 \left[A_1 , A_2 \right] \\
\Omega_3 (t) &= \frac{1}{6} \int_0^t dt_1 \int_0^{t_1} dt_2 \int_0^{t_2} dt_3 \left[ \left[A_1, \left[A_2 , A_3 \right]\right] + \left[\left[A_1 , A_2 \right], A_3 \right] \right]\\
\Omega_4 (t) &= \frac{1}{12} \int_0^t dt_1 \int_0^{t_1} dt_2 \int_0^{t_2} dt_3 \int_0^{t_3} dt_4 \left[ \left[[[A_1 , A_2], A_3], A_4 \right] + \left[A_1 , [[A_2, A_3], A_4] \right] + \left[A_1 , [A_2, [A_3, A_4]] \right] +\left[A_2 , [A_3, [A_4, A_1]] \right] \right]\\
\end{align}
$$

---

# Numerical Simulation; side-peak problem

Til now, an important question arises, the solution $e^{\Omega(t)}$ derived from the series $\Omega(t) = \sum_{n=1}^\inf \Omega_n (t)$ can be exact or approximated solution. It is challenging to verify but at least we can and should verify whether
- $e^{\Omega(t)}$ is in the Lie algebra $\mathfrak{g}$. 
- The series converges.

---

# Numerical Simulation; side-peak problem

Here are my derived solutions:
$$
\begin{align}
\Omega_1 (T) &= -i (\delta_{(0,1)} T I_z^{(0,1)} + c_1 (\cos\phi I_x^{(0,1)} + \sin\phi I_y^{(0,1)}))\\
\Omega_2 (T) &= 0\\
\Omega_3 (T) &= -i \frac{i\delta_{(0,1)}^2}{24} K_1 (\cos\phi I_x^{(0,1)} + \sin\phi I_y^{(0,1)}) -i \frac{i\delta_{0,1}}{24} K_2 I_z^{(0,1)} \\
\Omega_4 (T) &= 0\\
\end{align}
$$

where
$$
\begin{align}
c_1 &= \int_0^T dt f(t) \\
K_1 &= \int_0^T dt_1 \int_0^{t_1} dt_2 \int_0^{t_2} dt_3 (2f(t_2 ) - f(t_1) - f(t_3)) \\
K_1 &= \int_0^T dt_1 \int_0^{t_1} dt_2 \int_0^{t_2} dt_3 (2f(t_1 )f(t_3) - f(t_1)f(t_2) - f(t_2)f(t_3)) 
\end{align}
$$

---

# DDrf

Trivially, $e^{\Omega(T)}$ is in Lie algebra $\mathfrak{g}$. And it is known that if 
$$
\int_0^T \left|| A(s) \right||_2 ds < \pi .
$$

I think it might be(?)... I should've run simulation but not yet!


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
