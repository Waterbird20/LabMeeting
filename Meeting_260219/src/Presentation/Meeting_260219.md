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
19 Feb 2026
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
   flex: 0 0 70%;
   padding-right: 5rem;
   padding-left: 1rem;
   padding-bottom: 8rem;
}

.col-right-content{
   margin-left: -100px;
   flex: 0 0 20%;
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

1. **DDrf**
   - Work Flow
   - Numerical Simulation based on Taminiau Paper
   - Numerical Simulation: Side-peak Problem
2. **Post-Selection**
   - Architecture / File Structure
   - Density Matrix Analysis
3. **Distributed Quantum Machine Learning**
   - Architecture / File Structure
   - Dataset: 9-dimensional clustered binary classification
   - Data Encoding: Ising-type feature map
   - Potential References



</div>

<div class="col-right-content">


</div>
</div>

---

# Work Flow

1. Comparative Study between CPMG and DDrf (Theory/Numeric)
   - **[Pending] Poster work $\leftarrow$ (Hun)**
2. Enhanced (Hybrid) DDrf gate (Theory/Numeric)
   - [Completed] Tried many ideas... $\leftarrow$ (J. J)
   - [Ongoing] Alternating $\Omega_{\text{RF}}$ for odd- and even-numbered pulses $\leftarrow$ (J. J)
3. DDrf Spectroscopy (Experiment)
   - **[Completed] Numerical Simulation based on Taminiau Paper $\leftarrow$ (Hun)**
   - [Ongoing] Experiemnt $\leftarrow$ (Dr. Lee)
   - **[Ongoing] Numerical Simulation; side-peak problem $\leftarrow$(Hun)**
4. Multi-qubit Control (Numeric/Experiment)
   - [Pending] 

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
e^{-i H_{1}^{\prime}\tau}R_{1}(0)
\end{align}
$$

It is worth mentioning that
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

## Problems

1. **[Resolved]** Peaks were not observed for all resonant frequencies (arbitrary $m$).
*Note: The strengths of the off-resonant interactions are set by the Rabi frequency in combination with the detuning of the RF field $\omega_{\text{RF}}$ from both $\omega_0$ and $\omega_1$.*

2. **[Resolved]** The peaks corresponding to $\omega_{+1}$ and $\omega_{-1}$ must be different. 
*Already discussed in a previous meeting* (Dr. Lee)

3. **[Ongoing]** Side-peak problem.


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

<img src="Meeting_260219/src/Presentation/media/fig1.png" style="max-width: 100%; height: 60%; object-fit: contain;">

---

# Distributed Quantum Machine Learning: Data Set

<img src="Meeting_260219/src/Presentation/media/fig2.png" style="max-width: 100%; height: 60%; object-fit: contain;">

---

# Distributed Quantum Machine Learning: Data Set

<img src="Meeting_260219/src/Presentation/media/fig3.png" style="max-width: 100%; height: 60%; object-fit: contain;">

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
