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

# Entanglement: Reference

![](./media/Ryu.png)

The entanglement spectrum of partial transpose in conformal field theory (CFT) was also recently studied. Among other useful numerical methods, tree tensor network, Monte Carlo simulations, and rational interpolations are notable.

---

# Entanglement: Partial transpose as partial time reversal in spin systems

For spin systems at fixed spatial locations, partial transposition of the density matrix is unitarily equivalent to partial time reversal, differing only by a local rotation $U_T = e^{-i\pi \sigma_y}$ on the transposed subsystem.

---

# CGLMP Test

---

# Distributed Quantum Machine Learning

Agenda:
1. Which data are we going to use? $\rightarrow$ Controversial
2. How to investigate correlation between data(data imbedding)? $\rightarrow$ Need some studies
3. Inter-Processor Communication $\rightarrow$ Doable/Straightforward

---

# Distributed Quantum Machine Learning

Our current dataset, actually constructed to implement nonlinear classification problem, is not clear, seemingly (physically) might not be that meaningful, not widely used. And these dataset has well-known solutions, from perceptron to CNN. 

![width:1200px](./media/BetterThanClassical.png)


---

<style scoped>
li{
   font-size: 18pt;
}
</style>

# Distributed Quantum Machine Learning

![width:1200px](./media/whyMNIST.png)

* Due to the input size limitations of quantum simulators, preprocessing MNIST via PCA or resolution reduction fundamentally changes the nature of the learning problem, rendering the well-established baselines set by classical benchmarks meaningless.
* Most studies simplify the task to binary classification between two digits rather than the full 10-class problem, effectively reporting high accuracy on trivially easy tasks that a linear classifier could nearly solve perfectly.
* As a result, reported accuracies vary widely from 70% to 99.6% across studies with no consistent interpretation, failing to provide any reliable evidence for the genuine performance of quantum models.

---

# Distributed Quantum Machine Learning

**One-way cyclic** (`cyclic`):
```
    P₀ ──measure──► P₁ ──measure──► P₂
     ▲                                │
     └────────────measure─────────────┘
```
Each processor measures a qubit and sends the classical outcome to the next processor in the ring. The receiving processor applies conditional gates based on the single measurement outcome.

---

# Distributed Quantum Machine Learning


**Joint cyclic feedback** (`multi_fixed`, current default):
```
         P₀
        ▲ ▲ 
  m₀,m₂/   \m₀,m₁
      /     \
     ▼       ▼
    P₂ ◄───► P₁
       m₁,m₂
```
Each pair of processors sends their joint measurement outcomes to the third. The decision function `score = a·m_i + b·m_{i+1} + c·m_i·m_{i+1}` combines both measurements to determine which gate to apply on the target processor.

---

# Distributed Quantum Machine Learning

the cross-processor pooling layer uses a **classical decision function** to determine which quantum gate to apply on a target qubit, based on mid-circuit measurement outcomes from two source processors:

```
score = a · m_i  +  b · m_{i+1}  +  c · m_i · m_{i+1}

if score >= d:
    apply RX(w₀), RZ(w₁) to target qubit on processor (i+2)
else:
    apply RX(w₂), RZ(w₃) to target qubit on processor (i+2)
```

where `m_i ∈ {0, 1}` is the measurement outcome from processor *i*, and `m_{i+1} ∈ {0, 1}` from processor *(i+1)*.


---

# Distributed Quantum Machine Learning

If `m=(m_i, m_{i+1}) ∈ {0, 1}`, the score function defines a Boolean logic gate. If different $(a, b, c, d)$ settings satisfies the following table, we might implement classical gates:

|m=(0,0)|m=(1,0)|m=(0,1)| =m=(1,1)| Gate |
|-------|-------|-------|-------|------|
| 0 < d | a < d | b < d | a+b+c ≥ d | AND |
| 0 ≥ d | a < d | b < d | a+b+c ≥ d | XNOR |
| 0 < d | a ≥ d | b ≥ d | a+b+c ≥ d | OR |
| 0 < d | a ≥ d | b < d | a+b+c ≥ d | COPY |


---

# Compressive QST

![](./media/CQST.png)


---

# Post-Selection

1. Breif introduction to Measurement Scenerio
2. Analytical Trial to find maximum QFI
3. Numerical optimization; separable filter
4. Numerical optimization; entangled filter

---

# Post-Selection

Whatever we are sensing on, it requires interaction between the system and the something we are measuring on. In our NV-center system, we can perform quantum sensing on the external magnetic field. In the quantum area, such interaction lives in the Hamiltonian.

$$
\mathcal{H}_B = \gamma B S_z
$$
where $\gamma$ is gyromagnetic ratio, related to magnetic sensitivity. Through this interaction term and time evolution, $e^{-i\mathcal{H}}$, the information on $B$-field is imbedded on the state. 
$$
\ket{\psi} \rightarrow e^{-i\mathcal{H}_B} \ket\psi .
$$

---

# Post-Selection

In this sense, the quantum fisher information(QFI) is how quickly a quantum state, represented by a density matrix, changes with respect to $B$-field. The faster, the more sensitivity. 
$$
F_Q = 2\sum_{k,l} \frac{(\lambda_k - \lambda_l)^2}{\lambda_k + \lambda_l} \left| \bra{k}\mathcal{H}_{B}\ket{l}\right|
$$
where $\rho = \sum_k \lambda_k \ket{k}\bra{k}$. 

---

# Post-Selection

The information on quantum system can be obtained through actual measurement. That is, the quantum system is sampled into probability distribution $\{p_i\}$. 

In statistics, log-likelihood shows how plausible is it that $B$ is the true parameter given the data observed. 
$$
L(B) = \sum_i \ln p_i
$$
we can see how the log-likelihood is to a small change in the parameter $\delta B$. The variance of this score across different data samples tells us how wildly the slope fluctuates around that peak. 

---

# Post-Selection

- If the variance is large, it means the slope changes rapidly around the maximum. A sharp peak means that even a tiny shift away from the true parameter $\theta$ causes a massive drop in the likelihood of observing the data.
- Otherwise, the log-likelihood function is flat and wide. A flat log-likelihood means that many different values of $\theta$ are roughly equally likely to have produced the data you observed. 

In this sense, the classical fisher information (CFI) is a measure of how quickly a probability distribution changes with respect to $B$. 

$$
F_C = \sum_i p_i \left(\frac{\partial}{\partial B} \ln p_i\right)^2
$$

Note that $F_C < F_Q$.

---

# Post-Selection

Here is how to calculate QFI $F_Q$ in two-qubit system. 

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

At here, $G = \gamma (S_z \otimes I + I \otimes S_z )$, which has non-zero eigenvalue on the eigenbasis $\ket{00}, \ket{11}$.

---

# Post-Selection

So, consider density matrix under the subsystem of $\ket{00} , \ket{11}$ basis,
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
Under this subsystem, we can treat it as single qubit system so that density matrix can be decomposed into 
$
\vec{r} = (2ab\eta\cos\phi , 2ab\eta\sin\phi , a^2 - b^2)
$
$
F_Q^B = \left|\partial_B r \right|^2 = 16\gamma^2 t_s^2 e^{-4\tau} a^2 b^2 \leq 4\gamma^2 t_s^2 e^{-4\tau}
$

Note that in sinlge qubit system the maximum QFI is $4\gamma^2 t_s^2 e^{-2\tau}$. 
Given Quantum Fisher Information is addictive, two-qubit separable state is (twice) better than preparing entangled state.

---

# Post-Selection

In this sense, we have thought that adding Post-selection procedure might be advantageous. Intuitively speaking, the information of $B$-field is encoded in phase and through post-selection we may discard unnecessary information to ancillary qubit(or the other energy level). 

<!-- Post-Selection process is not described by unitary operation but Kraus operation and it can change QFI. While analytical approach is challenging (although I keep trying), we tried numerical optimization to find optimal post-selection strength, state preparation, measurement basis and the corresponding maximum QFI.  -->

---

# Post-Selection

We prepared the following circuit ansatz.

![width:1050px](./media/sep_ps.svg)

- State Preparation (RED): In order to prepare arbitrary two-qubit state, we employed one CNOT gate.

- Sensing (Orange): The system interact with (purturbed) magnetic field $\delta B$ during sensing time $t_s$ and noise channel is imbedded. 

- Then perform Post Selection and measurement (Purple): Unitary operation $U$, $V$ changes post-selection basis and measurement basis. 


---

# Post-Selection

Given Learning curve, optimization seems saturated well. However, the value it saturated was twice of QFI of Post-selected single-qubit system.

![](./media/learning_curves_sep_ps.png)

---

# Post-Selection

The prepared state turns out to be separable state, by investivating negativity of prepared state.

![](./media/negativity_trajectory.png)

---

# Post-Selection

GH suggested to enable entangled filter and entangled basis measurement. So we modified circuit ansatz. Two-qubit gate is added as if we are performing Post-selection on entangled basis.
$$
K_{\text{eff}} = U (K_1 \otimes K_2) U^{\dagger}
$$
At here, we employed KAK decomposition(equivalent using 3 CNOT gate per two-qubit gate) for $U$, $V$, in order to explore full SU(4) space.

![width:1050px](./media/ent_ps.svg)

---

# Post-Selection

Interestingly, optimized QFI(CFI) exceeds one of Post-selected separable state with order twice. 

![](./media/learning_curves_ent_ps_b.png)

---

# Post-Selection

But the prepared state was separable state but entanglement was retrieved at the end of Post-selection. Note that the strength of entanglement was not the stongest where negativity of Bell state is 0.5. 

![](./media/negativity_trajectory.png)


---

# Post-Selection

Further the filter strength also becomes asymmetric, and even one of post-selection strength $\gamma$ became almost 1, meaning that strong measurement. 

![](./media/ps_gamma_params.png)

---

# Post-Selection

I'm currently investigating optimal filter design. Under the guidance, Dr. Lee, I fixed Post-selection filter as $K=\sqrt{1-\gamma}\ket{00}\bra{00} + \ket{11}\bra{11}$. The poor performance is thought to be $\left[ \mathcal{H}_B , K \right] =0$.

![](./media/learning_curves_fixed_gamma.png)

---

# Post-Selection

Next, I tried to maximize frobenius norm of commutator relation. $\left[ \mathcal{H}_B , K \right] =0$, $K=\sqrt{1-\gamma}\ket{++}\bra{++} + \ket{--}\bra{--}$ .

![](./media/learning_curves_xbasis_ps.png)

---

# [TODO] Post-Selection

- Message became clear that under dephasing nosie, entanglement was vulnerable to noise rather than resource(well-known Heisenberg limit). Post-selection onto entangled basis can retrieve entanglement and enable gain in sensing. 
- Further analysis required: prepared state, filter strength and prepared Post-selection/measurement basis. 
- I need to write draft. I presume Apr 09 is target deadline. 
- Based on obtained insight, I'm trying to build analytical analysis framework.  

---

<style scoped>
   li{
      font-size: 18pt;
   }
</style>

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

I managed to finish QISK Poster Presentation (26/02/25). ✌️

![bg right:45%](./media/QISK.jpg)

---

# DDrf

DDrf refers selective phase-controlled radio-frequency(rf) driving of nuclear spins interleaved with dynamic decoupling(DD) sequences on an electron spin.

I would like explain DDrf in terms of Hamiltonian engineering. 
If we can write Hamiltonian separately, 
$$
H = \ket{0}\bra{0} \otimes H_0 + \ket{1}\bra{1} \otimes H_1
$$
then, the time evolution of second qubit becomes contingent on the first qubit. For example, if we prepare first qubit into $\ket{+}$ state and perform time evolution for $t_1$, we have
$$
\ket{\psi} = \frac{1}{\sqrt{2}} \left( \ket{00} + \ket{10} \right) \underbrace{\rightarrow}_{t_1}\frac{1}{\sqrt{2}} \left( \ket{0} \otimes e^{-iH_0 t_1} \ket{0} + \ket{1} \otimes e^{-iH_1 t_1} \ket{0} \right)
$$

---

# DDrf

If we change first qubit state using $\pi$-pulse ($\ket{0(1)} \rightarrow \ket{1(0)}$) and time-evolution $t_2$, then
$$
\begin{align}
\ket{\psi} = \frac{1}{\sqrt{2}} \left( \ket{00} + \ket{10} \right) &\underbrace{\rightarrow}_{t_1} \frac{1}{\sqrt{2}} \left( \ket{0} \otimes e^{-iH_0 t_1} \ket{0} + \ket{1} \otimes e^{-iH_1 t_1} \ket{0} \right) \\
&\underbrace{\rightarrow}_{\pi}\frac{1}{\sqrt{2}} \left( \ket{0} \otimes e^{-iH_1 t_1} \ket{0} + \ket{1} \otimes e^{-iH_0 t_1} \ket{0} \right) \\
&\underbrace{\rightarrow}_{t_2}\frac{1}{\sqrt{2}} \left( \ket{0} \otimes \underbrace{e^{-iH_0 t_2}e^{-iH_1 t_1}}_{=U_0} \ket{0} + \ket{1} \otimes \underbrace{e^{-iH_1 t_2}e^{-iH_0 t_1}}_{=U_1} \ket{0} \right)
\end{align}
$$
Generally, $U_0 \neq U_1$, hence, it is conditional gate!


---

# DDrf

![](./media/DDrf_pulse.png)

---

# Numerical Simulation based on Taminiau Paper

Given NV-${}^{13}$C Hamiltonian, we have
$$
\begin{aligned}
\mathcal{H} =& \gamma_c B_z I_z^i + A_{||}^i S_z I_z^i + A_{\perp} S_z I_x^i \\
\rightarrow \mathcal{H} =& \ket{0}\bra{0} \otimes \omega_0 I_z + 
\ket{-1}\bra{-1} \otimes \left( \omega_0 I_z -  A_{||}I_z - A_{\perp}I_x  \right) 
\end{aligned} 
$$
Then, with additional rf driving, we have
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
\omega_0 &= \gamma_c B_z                                    & \cos\beta   &= \frac{\omega_0 - A_{\perp}}{\omega_1} & \tilde{I}_z &= \cos\beta I_z + \sin \beta I_x \\
\omega_1 &= \sqrt{(\omega_0 - A_{\parallel}) + A_\perp^2}   & \sin\beta   &= \frac{A_{\perp}}{\omega_1} & \tilde{I}_x &= \cos\beta I_x - \sin \beta I_z
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

![height:200px](./media/spectroscopy_sequence.png)

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


<img src="./media/Taminiau_spectroscopy.png" style="max-width: 100%; height: 25%; object-fit: contain;">



---

# Numerical Simulation based on Taminiau Paper



<img src="./media/Reproduce.png" style="max-width: 100%; height: 45%; object-fit: contain;">

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

![width:500px](./media/DDrf_pulse_cell.png)

---

# DDrf



Then, the total unitary operation becomes: 
$$
U = \ket{0}\bra{0} \otimes \prod_{k=1}^{N/2} V_0^{(k)} + \ket{1}\bra{1} \otimes \prod_{k=1}^{N/2} V_1^{(k)}.
$$
This equation is itself equivalent to the previous one, but imply the following strategy. The idea is that any unitary operation can be written as $\textcolor{red}{e^{-iA}}$ for arbitrary Hermitian operator $A$. At the same time, it can be decomposed into $\textcolor{orange}{e^{-iA_1}e^{-iA_2}}$ but this decomposition is non-trivial and not unique. This means we can choose a favorable form to calculate the equation above without fully calculating previous equation. 

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

---

# DDrf

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
Under such operation, we expect to find a form $e^{-i\alpha I_z} e^{-i \theta_{\beta}\hat\sigma_{\beta}}$ where $\alpha, \theta_{\beta}, \hat\sigma_{\beta}$ are arbitrary parameter and axis and $\beta$ denotes $\beta$ dependence. Further we can **verify this approximation by testing its limit onto $\beta\rightarrow0$ and $\Omega\rightarrow0$**, for each corresponds to Taminiau DDrf and CPMG. 

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

<img src="./media/unconditional.png" style="max-width: 100%; height: 80%; object-fit: contain;">

---

# Numerical Simulation; side-peak problem

<img src="./media/unconditional_focus.png" style="max-width: 100%; height: 80%; object-fit: contain;">

---

# Numerical Simulation; side-peak problem

<!-- TODO: Side peak is also detected in large $\Omega_{\text{RF}}$ -->
<img src="./media/sidepeak.png" style="max-width: 100%; height: 80%; object-fit: contain;">

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

We have observed that side peak problem arises near resonant peak $\omega_{\text{RF}}\simeq \omega_1$. Let's retrun to rotating frame Hamiltonian. 

$$
\begin{align}
H_0 \rightarrow H_{0}^{\prime} &= R_{0}(t) (H_{0} - \omega_{\text{RF}}I_z) R_0 (t)^{\dagger} &\\ &= \delta_0 I_z + \Omega_{RF} (\cos\phi I_x + \sin\phi I_y ) &\\
H_1 \rightarrow H_{1}^{\prime} &= R_{1}(t) (H_{1} - \omega_{\text{RF}}I_z) R_1 (t)^{\dagger} &\\ &= \delta_1 \tilde{I}_z + \Omega_{RF} \cos\beta (\cos\phi \tilde{I}_x + \sin\phi \tilde{I}_y ) 
\end{align}
$$
where $\delta_{(0,1)} = \omega_{0(1)} - \omega_{\text{RF}}$. At here lets assume $\hat{\tilde{I}} = \hat{I}$ for simplicity, that is $A_\perp \simeq 0$ and $\beta\rightarrow 0$.

Previously, we have assumed $\delta_1 \ll \Omega \ll \delta_0$. To see dutuned effect, lets make $\delta_1$ small but not negligible.

---

# DDrf

With $H_1(\phi) = \delta_1 I_z + \Omega\hat\phi\cdot\vec{I}$, the propagator $e^{-iH_1 t}$ is now a rotation about a **tilted** axis. Define

$$\Omega_{\mathrm{eff}} = \sqrt{\Omega^2 + \delta_1^2}, \qquad \sin\gamma = \frac{\delta_1}{\Omega_{\mathrm{eff}}}, \qquad \cos\gamma = \frac{\Omega}{\Omega_{\mathrm{eff}}}$$

so that

$$H_1(\phi) = \Omega_{\mathrm{eff}}\,\hat{n}(\phi)\cdot\vec{I}, \qquad \hat{n}(\phi) = (\cos\gamma\cos\phi,\;\cos\gamma\sin\phi,\;\sin\gamma).$$

The axis $\hat{n}(\phi)$ is tilted out of the $xy$-plane by angle $\gamma$.

---

# DDrf



The conjugation trick still works, since $e^{i\alpha I_z}(\hat{n}\cdot\vec{I})e^{-i\alpha I_z}$ only rotates the azimuthal angle by $-\alpha$ and leaves the $z$-component ($\sin\gamma$) invariant. So exactly the same manipulations you wrote give:

$$V_0^{(k)} = e^{-i2\delta_0\tau I_z}\;e^{-i\Omega_{\mathrm{eff}}\,2\tau\;\hat{n}(\phi_{2k}')\cdot\vec{I}}$$

with $\phi_{2k}' = \phi_{2k} - \delta_0\tau = (2k-2)\delta_0\tau$, and

$$V_1^{(k)} = e^{-i2\delta_0\tau I_z}\;e^{-i\Omega_{\mathrm{eff}}\,2\tau\;\hat{n}(\phi_{2k-1})\cdot\vec{I}}$$

with $\phi_{2k-1} = (2k-2)\delta_0\tau + \pi$. The phase protocol ensures the two RF pulses within each $V_1$ cell collapse onto the same tilted axis, just as before.

---

# DDrf


Composing over $N/2$ cells, each factor of $e^{-i2\delta_0\tau I_z}$ shifts subsequent azimuthal angles by $2\delta_0\tau$, which exactly compensates the $k$-dependent phase stepping. The tilt angle $\gamma$ is $k$-independent, so it passes through the telescoping unchanged:

$$V_0^{\mathrm{tot}} = e^{-iN\delta_0\tau I_z}\;e^{-i\Omega_{\mathrm{eff}}\,N\tau\;\hat{n}_0\cdot\vec{I}}, \qquad \hat{n}_0 = (\cos\gamma,\;0,\;\sin\gamma)$$

$$V_1^{\mathrm{tot}} = e^{-iN\delta_0\tau I_z}\;e^{-i\Omega_{\mathrm{eff}}\,N\tau\;\hat{m}_0\cdot\vec{I}}, \qquad \hat{m}_0 = (-\cos\gamma,\;0,\;\sin\gamma)$$

The RZ factor $e^{-iN\delta_0\tau I_z}$ is the same as the $\delta_1=0$ case, so the interesting physics is in the rotations. Comparing the two axes:

$$\hat{n}_0 = \cos\gamma\,\hat{x} + \sin\gamma\,\hat{z}, \qquad \hat{m}_0 = -\cos\gamma\,\hat{x} + \sin\gamma\,\hat{z} .$$

---

# DDrf

## Full unitary

Defining $\alpha \equiv N\delta_0\tau$ and $\theta \equiv \Omega_{\mathrm{eff}}\,N\tau$:

$$U = \bigl(\mathbb{1}_e \otimes e^{-i\alpha I_z}\bigr)\cdot\Bigl[\lvert 0\rangle\langle 0\rvert \otimes e^{-i\theta\,\hat{n}(0)\cdot\vec{I}} \;+\; \lvert 1\rangle\langle 1\rvert \otimes e^{-i\theta\,\hat{n}(\pi)\cdot\vec{I}}\Bigr]$$

The first factor is an **unconditional** nuclear $z$-rotation (identical for both electron states). The second factor is the **conditional gate**.

---

# DDrf


Substituting $\Omega_{\mathrm{eff}}^2 = \Omega^2 + \delta_1^2$ and $\sin^2\gamma = \delta_1^2/(\Omega^2 + \delta_1^2)$:

$$
\mathrm{Tr}(U_0 U_1^\dagger) = 2\cos\!\Big(\!\sqrt{\Omega^2 + \delta_1^2}\;N\tau\Big) + \frac{4\delta_1^2}{\Omega^2 + \delta_1^2}\,\sin^2\!\!\left(\frac{\sqrt{\Omega^2+\delta_1^2}\;N\tau}{2}\right)
$$


Using $2\cos\theta = 2 - 4\sin^2(\theta/2)$:

$$\frac{1}{2}\mathrm{Tr}(U_0 U_1^\dagger) = 1 - \frac{2\Omega^2}{\Omega^2 + \delta_1^2}\,\sin^2\!\!\left(\frac{\sqrt{\Omega^2+\delta_1^2}\;N\tau}{2}\right)$$

This is exactly the **detuned Rabi formula**. Yay!

---

# DDrf 

Given $\frac{\delta_0}{\Omega} \ll 1$, applying Taylor expansion upto first order, we have,
$$
\frac{1}{2}\mathrm{Tr}(U_0 U_1^\dagger) = 1 - \underbrace{2\Omega^2 \left( 1 - \frac{\delta_0^2}{\Omega^2} \right)}_{\text{Lorentzian envelope}} \underbrace{\sin^2 \left(N\tau \delta_0^2 + \frac{N\Omega\tau}{2}  \right)}_{\text{sidelobe}}
$$
We can see:
- Lorentzian envelope takes quadratic form. 
- The osillation makes sidelobe in a period $\frac{2\pi}{N\tau}$.


---

# DDrf 

It seems that this equation explains this curve very well!
$$
\boxed{
   \frac{1}{2}\mathrm{Tr}(U_0 U_1^\dagger) = 1 - 2\Omega^2 \left( 1 - \frac{\delta_0^2}{\Omega^2} \right)\sin^2 \left(N\Omega \delta_0^2 + \frac{N\Omega\tau}{2}  \right)
}
$$

![width:800px](./media/unconditional_focus.png)

---

# DDrf

## Per-cell propagators with $\Omega_k = \Omega f(k)$

Although it might be unrealistic, the idea is using apodizied pulse RF driving like this...

![](./media/Adodizied.jpeg)

---

# DDrf

The per-cell propagators are:

$$V_0^{(k)} = e^{-i2\delta_0\tau I_z}\;e^{-i\theta_k\,\hat{n}_k(0)\cdot\vec{I}}, \qquad V_1^{(k)} = e^{-i2\delta_0\tau I_z}\;e^{-i\theta_k\,\hat{n}_k(\pi)\cdot\vec{I}}$$

with $k$-dependent quantities:

$$\theta_k = 2\Omega_{\mathrm{eff},k}\,\tau, \qquad \Omega_{\mathrm{eff},k} = \sqrt{\Omega_k^2 + \delta_1^2}, \qquad \sin\gamma_k = \frac{\delta_1}{\Omega_{\mathrm{eff},k}}, \qquad \cos\gamma_k = \frac{\Omega_k}{\Omega_{\mathrm{eff},k}}$$

$$\hat{n}_k(0) = (\cos\gamma_k,\;0,\;\sin\gamma_k), \qquad \hat{n}_k(\pi) = (-\cos\gamma_k,\;0,\;\sin\gamma_k)$$

Note: the tilt angle $\gamma_k$ now **varies with $k$** because $\Omega_k$ does.

---

# DDrf

Commuting $e^{-i2\delta_0\tau I_z}$ through $R_k$ shifts the azimuthal angle by $-2\delta_0\tau$ while preserving $\gamma_k$ and $\theta_k$. By induction (exactly as before):

$$\prod_{k=1}^{N/2} V_0^{(k)} = e^{-iN\delta_0\tau I_z}\;\prod_{k=N/2}^{1} e^{-i\theta_k\,\hat{n}_k(0)\cdot\vec{I}}, ~~~~~~~~~~~~~~ \prod_{k=1}^{N/2} V_1^{(k)} = e^{-iN\delta_0\tau I_z}\;\prod_{k=N/2}^{1} e^{-i\theta_k\,\hat{n}_k(\pi)\cdot\vec{I}}$$

**Crucial difference from the constant-$\Omega$ case**: the ordered product does **not** collapse into a single rotation, because $\hat{n}_k(0)$ has $k$-dependent tilt $\gamma_k$, and rotations about different axes in the $xz$-plane do not commute and are not aligned anymore. 

**Exception — $\delta_1 = 0$**: all $\gamma_k = 0$, all axes align ($\hat{x}$ or $-\hat{x}$), and the products do collapse:

$$\prod_{k=N/2}^{1} e^{-i\cdot 2\Omega_k\tau\,I_x} = e^{-i\Theta\,I_x}, \qquad \Theta \equiv 2\Omega\tau\sum_{k=1}^{N/2} f(k)$$


---

# DDrf

### Case $\delta_1 = 0$: clean windowed result

All $V_0$ rotations are about $+\hat{x}$, all $V_1$ rotations about $-\hat{x}$:

$$\mathrm{tr}(U_0 U_1^\dagger)\big|_{\delta_1=0} = \mathrm{tr}\!\left(e^{-i\Theta I_x}\cdot e^{-i\Theta I_x}\right) = \mathrm{tr}\!\left(e^{-i2\Theta I_x}\right) = 2\cos\Theta$$

$$\boxed{\mathrm{tr}(U_0 U_1^\dagger)\big|_{\delta_1=0} = 2\cos\!\left(2\Omega\tau\sum_{k=1}^{N/2}f(k)\right)}$$



---

# DDrf


### Case $\delta_1 \neq 0$: perturbative result

For small $\delta_1$, define the zeroth-order (unperturbed, $\delta_1=0$) rotation and the perturbation:

$$e^{-i\theta_k\hat{n}_k(0)\cdot\vec{I}} = e^{-i(2\Omega_k\tau\,I_x\;+\;2\delta_1\tau\,I_z)\;+\;\mathcal{O}(\delta_1^2)}$$

where I used $\theta_k\cos\gamma_k = 2\Omega_k\tau + \mathcal{O}(\delta_1^2)$ and $\theta_k\sin\gamma_k = 2\delta_1\tau$ (exactly, independent of $k$).

Going to the interaction picture with respect to the $I_x$ rotations, define the cumulative angle after cell $j$:

$$\Phi_j \equiv 2\Omega\tau\sum_{m=1}^{j}f(m)$$


---

# DDrf

The $I_z$ perturbation in cell $k$ is rotated into the frame where cells $1,\ldots,k-1$ have already been applied:

$$I_z \;\longrightarrow\; \cos\Phi_{k-1}\;I_z \;-\; \sin\Phi_{k-1}\;I_y$$

Summing the first-order contribution over all cells:

$$\prod_{k=N/2}^{1}e^{-i\theta_k\hat{n}_k(0)\cdot\vec{I}} \;\approx\; e^{-i\Theta I_x}\;\exp\!\left(-i2\delta_1\tau\sum_{k=1}^{N/2}\bigl[\cos\Phi_{k-1}\;I_z - \sin\Phi_{k-1}\;I_y\bigr]\right)$$

Similarly for the $V_1$ branch (with $I_x \to -I_x$). When we form $U_0 U_1^\dagger$, the leading $e^{-i\Theta I_x}$ terms combine into $e^{-i2\Theta I_x}$, and the first-order corrections produce:

$$\boxed{\frac{1}{2}\mathrm{tr}(U_0 U_1^\dagger) \;\approx\; \cos\Theta \;-\; \frac{2\delta_1^2\tau^2}{\sin^2(\Theta/2)}\;\sin^2(\Theta/2)\;\left|\sum_{k=1}^{N/2}f_k\,e^{i\Phi_{k-1}}\right|^2\;\cdot(\ldots)}$$

---

# DDrf

Defining the normalized window:

$$F(\delta_1) \equiv \sum_{k=1}^{N/2} f(k)\,e^{i\Phi_{k-1}(\delta_1)}$$

the spectral response near resonance takes the form of a (windowed) discrete Fourier transform:

$$\frac{1}{2}\mathrm{tr}(U_0 U_1^\dagger) \approx -1 + \text{const}\times\left|\frac{F(\delta_1)}{F(0)}\right|^2 \cdot \delta_1^2$$

where $F(0) = \sum_k f(k)$ is just the normalization. 

---

# DDrf

Each window $f(k)$ has a continuous counterpart $w(t)$, and the spectral response is the normalized Fourier transform squared. All windows in the cosine family can be written as $w(t) = a_0 - a_1\cos(2\pi t/T) + a_2\cos(4\pi t/T)$. The normalized spectral response has the universal form:

$$\left|\frac{F(\delta_1)}{F(0)}\right|^2 = \mathrm{sinc}^2(u)\cdot\left|G(u)\right|^2$$

where $\mathrm{sinc}(u) = \sin(\pi u)/(\pi u)$ and the **window kernel** $G(u)$ depends on the coefficients.

**Rectangular** ($a_0 = 1$, $a_1 = a_2 = 0$):$\left|\frac{F}{F(0)}\right|^2_{\mathrm{rect}} = \mathrm{sinc}^2(u)$

---

# DDrf

- **Hanning** ($a_0 = a_1 = 1/2$, $a_2 = 0$):$\left|\frac{F}{F(0)}\right|^2_{\mathrm{Hann}} = \frac{\mathrm{sinc}^2(u)}{(1-u^2)^2}$
The factor $(1-u^2)^{-2}$ kills the sinc sidelobes: each sinc zero at integer $u$ is now cancelled by the denominator zero at $u = \pm 1$, leaving only the zero structure at $u = \pm 2, \pm 3, \ldots$

- **Hamming** ($a_0 = 0.54$, $a_1 = 0.46$, $a_2 = 0$):$\left|\frac{F}{F(0)}\right|^2_{\mathrm{Hamm}} = \mathrm{sinc}^2(u)\cdot\left(\frac{50u^2 - 27}{27(u^2 - 1)}\right)^2$
The numerator $50u^2 - 27$ has a zero at $u = \sqrt{27/50} \approx 0.735$, which partially cancels the first sinc sidelobe. The imperfect cancellation at $u = 1$ (unlike Hann) gives a small residual, but the first sidelobe is pushed down to $-42.7$ dB.

- **Blackman** ($a_0 = 0.42$, $a_1 = 0.5$, $a_2 = 0.08$):$\left|\frac{F}{F(0)}\right|^2_{\mathrm{Black}} = \mathrm{sinc}^2(u)\cdot\left(\frac{50u^4 - 209u^2 + 84}{21(u^2-1)(u^2 - 4)}\right)^2$ 
The extra $\cos(4\pi t/T)$ harmonic introduces the $(u^2 - 4)$ denominator, cancelling the sinc zeros at both $u = \pm 1$ and $u = \pm 2$, pushing the first nonzero sidelobe out to $u \approx 3$.

---

# DDrf 

Using the $\pi$-gate condition $2\Omega\tau\sum_k f(k) = \pi$ with $\sum_k f(k) = M\bar{f}$:

$$u = \frac{\delta_1}{2\Omega\bar{f}}$$

where $\bar{f}$ is the mean of the window. The FWHM in physical units:


| Window | $\bar{f}$ | FWHM ($u$) | FWHM ($\delta_1$) |
|---|---|---|---|
| Rectangular | 1.00 | 0.89 | $1.77\,\Omega$ |
| Hanning | 0.50 | 1.44 | $1.44\,\Omega$ |
| Hamming | 0.54 | 1.30 | $1.41\,\Omega$ |
| Blackman | 0.42 | 1.68 | $1.41\,\Omega$ |


---


# DDrf

## How badly is it broken?

The mismatch between consecutive axes is

$$\Delta\gamma_k \equiv \gamma_{k+1} - \gamma_k \approx -\frac{\delta_1}{\Omega^2}\,\Omega\,\Delta f_k$$

where $\Delta f_k = f(k+1) - f(k)$. So the non-commutativity enters at order $\delta_1 \cdot \Delta f_k$, which is small when either $\delta_1$ is small or the window varies slowly. 

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
\Omega(t) = \sum_{n=1}^{\inf} \Omega_n (t)
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
