---
marp: true
theme: serif
paginate: true
math: mathjax
footer: 'Donghun Jung · Journal Club'
---

<!-- _class: title -->
<!-- _paginate: false -->
<!-- _footer: '' -->

<div class="kicker">Paulee Group · Center for Quantum Technology, KIST</div>

# The Hidden Manifold Model

<div class="subtitle">Analytic learning dynamics of two-layer networks on structured data, and its implications for quantum machine learning</div>

**Goldt, Mézard, Krzakala & Zdeborová** · *Phys. Rev. X* **10**, 041044 (2020)

<div class="meta">
<span class="author">Donghun Jung</span>
<span class="affil">2026-06-29 · Journal Club</span>
</div>

<style>
/* title slide: frame the paper-cover screenshot */
.cover { margin: 0.5rem auto 0.2rem; line-height: 0; }
.cover img { border: 1px solid #c9c9c9; border-radius: 4px; box-shadow: 0 2px 10px rgba(0,0,0,0.12); }
/* deck-wide: trim margins and give figures more room (figure column wider) */
section { padding: 50px 54px; }
.columns { gap: 1.0rem; }
.columns .col:first-child { flex: 0.92; }
.columns .col:last-child  { flex: 1.08; }
.columns figure.figure img { max-height: 420px; }
</style>


---


# The paper

<div class="cover">

![w:840](./Meeting_260629/src/Presentation/media/paper_cover.png)

</div>

<div class="paper-meta">
Open access · <em>Phys. Rev. X</em> <strong>10</strong>, 041044 (2020) · arXiv:1909.11500
</div>

<style scoped>
.cover { margin: 0.6rem auto 0.4rem; }
.paper-meta { text-align: center; color: #555; font-size: 20px; }
</style>


---


# Outline

1. **Why a physicist cares about neural networks**: universal approximation, "physics for AI", and the quantum-ML motivation.
2. **The hidden manifold model**: structured data from a GAN-like generator, with a teacher and a student.
3. **Learning setup**: a two-layer student trained by online SGD.
4. **Gaussian Equivalence**: the local fields, the order parameters, and why the problem becomes solvable.
5. **Equations of motion**: the closed system, and the learning curve.
6. **Results**: structure, specialization, and increasing complexity.
7. **Discussion and outlook**: limitations, and the path to quantum ML.

<div class="legend">
<span class="cat intro">Intro</span>
<span class="cat method">Model &amp; theory</span>
<span class="cat results">Results</span>
<span class="cat strategy">Discussion</span>
<span class="cat ongoing">Outlook</span>
</div>


---


<style scoped>
section { font-size: 23px; }
figure.figure img { max-height: 300px; }
</style>

# <span class="cat intro">Intro</span> Neural networks won data science

- **Malleable and scalable.** Stacking and widening layers lets the *same* architecture fit vision, language, and control.
- **One universal building block.** A layer is $x \mapsto g(Wx)$: a linear map followed by a pointwise nonlinearity.
- **Mature tooling.** Autodiff frameworks (**PyTorch**, **TensorFlow**) made training networks the default in data science and modern AI.

<figure class="figure">

![w:560](./Meeting_260629/src/Presentation/media/fig_e_feedforward_net.png)

<figcaption>Schematic feed-forward net (layers 4-6-5-1, illustrative): each layer is a linear map followed by a pointwise activation.</figcaption>
</figure>

---

# <span class="cat intro">Intro</span> Why they work, in principle

**Universal Approximation Theorem.** A feed-forward network with

- at least one hidden layer,
- enough hidden units, and
- a nonlinear activation $g$ (sigmoid, $\tanh$, ReLU),

can approximate any continuous function on a compact subset of $\mathbb{R}^n$ to arbitrary accuracy.

Training also works in practice: given a well-posed dataset and a convex surrogate loss (squared error or cross-entropy), SGD reliably finds a good fit.

<div class="callout">

**The open question.** Existence and trainability tell us *that* a network can fit, not *how* it learns or *why* it generalizes.

</div>

---

<style scoped>
section { font-size: 21px; }
figure.figure img { max-height: 298px; }
</style>

# <span class="cat intro">Intro</span> We do not understand the behavior

- For small models, trial-and-error was enough: tune width and hyperparameters by hand.
- As models became large and expensive, that approach stopped scaling, which motivated a "physics for AI": treat the network as a many-body system and look for laws.
- A leading example is **LLM scaling laws**: the test loss falls as a power law in compute, dataset size, and model size, a regularity we can measure but not yet derive. (Kaplan et al. 2020)

<figure class="figure">

![w:1080](./Meeting_260629/src/Presentation/media/kaplan_scaling_fig1.png)

<figcaption>Kaplan et al. 2020, Fig. 1: test loss falls as a power law in compute, dataset size, and parameter count (WebText2 language models; their published figure).</figcaption>
</figure>

---

# <span class="cat intro">Intro</span> This paper: statistical physics of the dynamics

- Statistical physics usually describes stationary states through a few **order parameters**, macroscopic summaries of a many-body system.
- This paper instead tracks order parameters **and their time evolution**: equations of motion for learning.
- Here "time" is training progress (the number of samples seen). The claim is that a few macroscopic overlaps evolve by **closed ODEs** that can be written down and integrated.

---

<style scoped>
section { font-size: 22px; }
</style>

# <span class="cat intro">Intro</span> Why this is possible: solvable data

<div class="columns">
<div class="col">

- Real data such as images is natural and, in ML benchmarks, deliberately decorrelated, which makes the learning dynamics analytically opaque.
- The hidden manifold model replaces this with data that has a controlled statistical structure (the **Gaussian Equivalence Property**, GEP), which makes the dynamics solvable.
- The data is synthetic, but it lets us follow exactly what happens during training, and in principle in more complex models.

</div>
<div class="col">

<figure class="figure">

![w:600](./Meeting_260629/src/Presentation/media/fig_d_low_to_high.png)

<figcaption>A flat latent plane folded into a curved manifold; colour = continuous label y*. Illustration: latent D=2 to ambient N=3, teacher M=3, inputs bounded to [-1,1], seed 8.</figcaption>
</figure>

</div>
</div>

---

# <span class="cat intro">Intro</span> Why I care: quantum machine learning

- Our **DQML** project has no native dataset and no large quantum hardware, so we emulate small circuits on classical GPUs.
- A frequent claim for QML is that it needs far fewer parameters than classical models.
- The benchmark study **"Better than classical?"** (Bowles, Ahmed & Schuld 2024) questions this claim:

<div class="callout">

**Findings.** Out-of-the-box **classical models outperform all 12 quantum models** across 160 benchmark datasets, and **removing entanglement rarely changes the result**, so "quantumness" may not be the deciding factor. The tasks are easy binary classification, and the quantum models often reproduce classical ones.

</div>

- We need data and analyses that identify where quantum methods actually help. An analytic handle of the HMM type is one such tool.


---


<style scoped>
section { font-size: 21px; }
figure.figure img { max-height: 290px; }
.columns .col { flex: 1; }
</style>

# <span class="cat method">Model</span> Structured data from a generator

- A generative model maps low-dimensional latent noise to high-dimensional, realistic-looking data, as a GAN generator does. The HMM uses the simplest such map, a single nonlinear transformation, with a **teacher** that assigns the labels.
- This is a **teacher-student** setup: the teacher fixes the ground-truth labels, and the student is trained to match them. The two need not be the same size.

<div class="columns">
<div class="col">

<figure class="figure">

![h:290](./Meeting_260629/src/Presentation/media/fig_b_teacher.png)

<figcaption>Teacher (frozen): maps the latent c to the continuous label y*. Schematic; M=2 units drawn.</figcaption>
</figure>

</div>
<div class="col">

<figure class="figure">

![h:290](./Meeting_260629/src/Presentation/media/fig_b_student.png)

<figcaption>Student (trained): maps the folded input x to a prediction; the prediction error (prediction minus target) drives SGD on the student only. Schematic; K=4 units drawn.</figcaption>
</figure>

</div>
</div>

---

<style scoped>
section { font-size: 22px; }
</style>

# <span class="cat method">Model</span> The hidden manifold

<div class="columns">
<div class="col">

- Latent coordinates $c$ of dimension $D$, drawn i.i.d. Gaussian.
- These are folded into a high-dimensional ambient space ($N \gg D$) through a fixed feature matrix $F$ and a pointwise nonlinearity $f$:

$$ x = f\!\left(\frac{cF}{\sqrt D}\right) \in \mathbb{R}^{N}. $$

- The labels depend only on $c$ through the teacher. The student sees only $x$, so it must recover the manifold structure. In this sense the manifold is hidden.

</div>
<div class="col">

<figure class="figure">

![w:560](./Meeting_260629/src/Presentation/media/fig2_hidden_manifold.png)

<figcaption>Folded inputs concentrate on a 2D sheet in 3D space; colour = teacher label. Illustration: latent D=2, ambient N=3, teacher M=2, seed 2 (the real target is continuous).</figcaption>
</figure>

</div>
</div>

---

# <span class="cat method">Model</span> Why this is close to real data

<div class="columns">
<div class="col">

- **Manifold hypothesis.** Realistic data (CIFAR-10, MNIST) does not fill its pixel space; it concentrates on a low-dimensional manifold.
- A few features, for example **PCA** components, already separate the classes.
- The HMM is a minimal instance of this picture: recognizable images lie on the surface, and off-manifold points are noise.

</div>
<div class="col">

<figure class="figure">

![w:580](./Meeting_260629/src/Presentation/media/paper_fig1_manifold.png)

<figcaption>Goldt et al. Fig. 1 (conceptual illustration): recognizable CIFAR images lie on the manifold; off-surface points are random noise.</figcaption>
</figure>

</div>
</div>


---


<style scoped>
section { font-size: 22px; }
</style>

# <span class="cat method">Setup</span> Two-layer student, online SGD

<div class="columns">
<div class="col">

**Student** ($K$ hidden units):

$$ \phi(x) = \sum_{k=1}^{K} v_k\, g\!\left(\frac{w_k\cdot x}{\sqrt N}\right). $$

- A **teacher** ($M$ units) on the latent $c$ sets the target $y^{*}$; the student is trained by one-pass **online SGD** (rate $\eta$), with $\alpha=P/N$ and $\delta=D/N$ held fixed.
- Bounded hidden units ($g=\mathrm{erf}\in(-1,1)$), but a **linear readout**, so the output is an *unbounded continuous real number*: the task is **regression** (MSE), not classification.

</div>
<div class="col">

<figure class="figure">

![w:600](./Meeting_260629/src/Presentation/media/fig_a_two_layer_net.png)

<figcaption>The two-layer (soft-committee) student: each unit applies the activation to a weighted sum, and a linear readout combines them. Schematic; K=2 units, N=3 inputs drawn.</figcaption>
</figure>

</div>
</div>

---

<style scoped>
section { font-size: 22px; }
</style>

# <span class="cat method">Setup</span> The learning step (online SGD)

The student learns **one sample at a time**. At step $\mu$:

1. Draw a **fresh** pair $(x_\mu, y^{*}_\mu)$, with the label $y^{*}_\mu=\phi(c_\mu;\tilde\theta)$ set by the teacher.
2. Predict $\hat y=\phi(x_\mu;\theta)$ and measure the error $\Delta_\mu=\hat y-y^{*}_\mu$.
3. Take one gradient step on the squared error $\tfrac12\Delta_\mu^2$ (learning rate $\eta$):

$$ \theta_{\mu+1}=\theta_\mu-\eta\,\nabla_\theta\,\tfrac12\Delta_\mu^2, \qquad \theta=(W,v). $$

- **One pass:** each sample is used once, then discarded, so the weights stay independent of the next sample. (We write this step out weight-by-weight later, to turn it into the equations of motion.)
- We grade the student by the **test (generalization) error** on unseen data, $\epsilon_g=\tfrac12\,\mathbb{E}[(\hat y-y^{*})^2]$, not the training loss.
- This generalizes the classic **teacher-student** setup (i.i.d. Gaussian inputs); the HMM keeps that solvability but adds the manifold structure real data has.


---


# <span class="cat method">Theory</span> Local fields

A neuron's output depends on its input only through its **pre-activation**, the scalar that enters the activation function. These pre-activations are the **local fields**:

$$ \lambda^k = \frac{1}{\sqrt N}\sum_i w^k_i\, f(u_i), \qquad \nu^m = \frac{1}{\sqrt D}\sum_r c_r\, \tilde w^m_r, \qquad u_i = \frac{1}{\sqrt D}\sum_r c_r F_{ir}. $$

- $u_i$: the pre-activation of input feature $i$, the latent vector projected onto row $i$ of $F$, before folding by $f$.
- $\lambda^k$: the pre-activation of **student** hidden unit $k$, a weighted sum of the folded inputs $f(u_i)$.
- $\nu^m$: the pre-activation of **teacher** hidden unit $m$, a weighted sum of the latent coordinates.

The factors $1/\sqrt N$ and $1/\sqrt D$ keep each field of order one. The test error depends on the data only through these $K+M$ scalar fields.

---

# <span class="cat method">Theory</span> The Gaussian Equivalence Property

**Property (GEP).** In the limit $N, P, D \to \infty$ (with $\alpha, \delta$ fixed), the $K+M$ local fields $\{\lambda^k\}, \{\nu^m\}$ are **jointly Gaussian**, so their joint distribution is fixed entirely by their first two moments.

The means are $\mathcal{O}(1/\sqrt N)$ for the student fields and $0$ for the teacher fields; the covariances are the three overlap matrices on the next slide.

<div class="callout">

**Consequence.** Everything the test error can depend on is contained in a few **covariances**, the order parameters; the individual microscopic weights drop out.

</div>

---

# <span class="cat method">Theory</span> Order parameters: the overlaps

Each order parameter is a covariance of the (centred) local fields:

$$ Q^{k\ell} = \mathbb{E}[\bar\lambda^k\bar\lambda^\ell], \qquad R^{km} = \mathbb{E}[\bar\lambda^k \nu^m], \qquad T^{mn} = \mathbb{E}[\nu^m\nu^n]. $$

- $Q^{k\ell}$ (**student-student**): how aligned student units $k$ and $\ell$ are. The diagonal $Q^{kk}$ is the variance, i.e. the size, of unit $k$'s field.
- $R^{km}$ (**student-teacher**): how much student unit $k$ has aligned with teacher unit $m$. Learning is the growth of these overlaps; **specialization** means each $k$ locks onto a single $m$.
- $T^{mn}$ (**teacher-teacher**): fixed by the frozen teacher, it encodes the structure of the target.

The generalization error is a function of $Q$, $R$, $T$ and the second-layer weights $v$ alone.

---

# <span class="cat method">Theory</span> Order parameters: the reduction

The student-side overlaps come from two underlying matrices:

$$ W^{k\ell} = \frac1N\sum_i w^k_i w^\ell_i \ \ (\text{ambient weight overlap}), \qquad \Sigma^{k\ell} = \frac1D\sum_r S^k_r S^\ell_r, \quad S^k_r = \frac1{\sqrt N}\sum_i w^k_i F_{ir}. $$

The folding function $f$ enters **only** through three scalars (for $u\sim\mathcal N(0,1)$):

$$ a = \langle f(u)\rangle, \qquad b = \langle u\,f(u)\rangle, \qquad c = \langle f(u)^2\rangle. $$

These combine into the student-side overlaps:

$$ Q^{k\ell} = (c-a^2-b^2)\,W^{k\ell} + b^2\,\Sigma^{k\ell}, \qquad R^{km} = b\,\frac1D\sum_r S^k_r\,\tilde w^m_r. $$

Any two folding functions with the same $(a,b,c)$ produce identical learning curves. (For an odd folding $a=\langle f\rangle=0$, so the dynamics below use $Q=(c-b^2)W+b^2\Sigma$.)



---


<style scoped>
section { font-size: 21px; }
</style>

# <span class="cat method">Theory</span> From weight updates to equations of motion

Every equation of motion descends from the **SGD updates** of the weights (one fresh sample per step):

$$ (w^k_i)_{\mu+1}-(w^k_i)_\mu=-\frac{\eta}{\sqrt N}\,v^k\,\Delta\,g'(\lambda^k)\,f(u_i), \qquad v^k_{\mu+1}-v^k_\mu=-\frac{\eta}{N}\,\Delta\,g(\lambda^k), $$

with the prediction error $\Delta=\sum_j v^j g(\lambda^j)-\sum_m\tilde v^m\tilde g(\nu^m)$.

The test error sees the data **only through the local fields**, so by the GEP it collapses to a function of the order parameters alone:

$$ \epsilon_g=\tfrac12\,\mathbb{E}\big[(\phi(x;\theta)-y^{*})^2\big] =\tfrac12\,\mathbb{E}\Big[\big(\textstyle\sum_k v^k g(\lambda^k)-\sum_m\tilde v^m\tilde g(\nu^m)\big)^2\Big] \;\xrightarrow{\,N,D\to\infty\,}\;\epsilon_g\big(Q,R,T,v,\tilde v\big). $$

So we recast each weight update as the induced change in $Q,R,T$. Because a fresh sample is independent of the current weights, the per-step change **self-averages**; in the limit it becomes a deterministic ODE in the time $t=\mu/N$. (We set $a=\langle f\rangle=0$, so $Q=(c-b^2)W+b^2\Sigma$.)

---

<style scoped>
section { font-size: 20px; }
</style>

# <span class="cat method">Theory</span> Equations of motion (1): direct overlaps

The dynamics are driven by Gaussian averages over the jointly-Gaussian fields (indices $j,k,\ell$ run over student units, $m,n$ over teacher units; $g_a$ is the activation of the third unit):

$$ I_2(k,j)=\mathbb{E}[g(\lambda^k)g(\lambda^j)], \quad I_3(k,j,a)=\mathbb{E}[g'(\lambda^k)\,\lambda^j\, g_a], \quad I_4(k,\ell,j,\iota)=\mathbb{E}[g'(\lambda^k)g'(\lambda^\ell)g(\lambda^j)g(\lambda^\iota)]. $$

**Second-layer weights** $v^k$:
$$ \frac{dv^k}{dt} = \eta\Big[\sum_n \tilde v_n\, I_2(k,n) - \sum_j v^j\, I_2(k,j)\Big]. $$

**Ambient student-student overlap** $W^{k\ell}=\tfrac1N\sum_i w^k_i w^\ell_i$:
$$ \frac{dW^{k\ell}}{dt} = -\eta v^k\!\Big(\sum_j v^j I_3(k,\ell,j) - \sum_n \tilde v^n I_3(k,\ell,n)\Big) - \eta v^\ell\!\Big(\sum_j v^j I_3(\ell,k,j) - \sum_n \tilde v^n I_3(\ell,k,n)\Big) $$
$$ +\, c\,\eta^2 v^k v^\ell\Big(\sum_{j,\iota} v^j v^\iota I_4(k,\ell,j,\iota) - 2\sum_{j,m} v^j \tilde v^m I_4(k,\ell,j,m) + \sum_{n,m} \tilde v^n \tilde v^m I_4(k,\ell,n,m)\Big). $$

---

# <span class="cat method">Theory</span> Equations of motion (2): spectral densities

The remaining overlaps become densities over the spectrum of $\Omega=\tfrac1N FF^\top$ (eigenvalues $\rho$), with $d(\rho)=(c-b^2)\delta+b^2\rho$ and $R^{km}=b\!\int\! d\rho\,p_\Omega(\rho)\,r^{km}(\rho,t)$:

$$ \frac{\partial r^{km}}{\partial t} = -\frac{\eta}{\delta}\,v^k\Bigg[\, d(\rho)\,r^{km}\!\sum_{j\neq k} v^j\frac{Q^{jj}I_3(k,k,j)-Q^{kj}I_3(k,j,j)}{Q^{jj}Q^{kk}-(Q^{kj})^2} \;+\; d(\rho)\!\sum_{j\neq k} v^j r^{jm}\frac{Q^{kk}I_3(k,j,j)-Q^{kj}I_3(k,k,j)}{Q^{jj}Q^{kk}-(Q^{kj})^2} $$

$$ +\;\frac{v^k\,d(\rho)\,r^{km}}{Q^{kk}}\,I_3(k,k,k) \;-\; d(\rho)\,r^{km}\!\sum_n \tilde v^n\frac{T^{nn}I_3(k,k,n)-R^{kn}I_3(k,n,n)}{Q^{kk}T^{nn}-(R^{kn})^2} \;-\; b\rho\!\sum_n \tilde v^n \tilde T^{nm}\frac{Q^{kk}I_3(k,n,n)-R^{kn}I_3(k,k,n)}{Q^{kk}T^{nn}-(R^{kn})^2}\,\Bigg]. $$

The latent overlap $\Sigma^{k\ell}=\!\int\! d\rho\,p_\Omega(\rho)\,\sigma^{k\ell}(\rho,t)$ obeys a structurally identical equation. Together with $Q^{k\ell}=(c-b^2)W^{k\ell}+b^2\Sigma^{k\ell}$ and the $v^k$, $W^{k\ell}$ equations above, the system is closed (no free parameters beyond $a,b,c,\eta,\delta$).

---

# <span class="cat method">Theory</span> Closing the equations, and the test error

Three steps make the system finite and explicit:

1. **Eigenbasis.** Projecting onto the eigenvectors of $\Omega=\tfrac1N FF^\top$ replaces an infinite hierarchy of order parameters with densities over the eigenvalues $\rho$ (Marchenko-Pastur for Gaussian $F$).
2. **Densities.** The overlaps become $r^{km}(\rho,t)$ and $\sigma^{k\ell}(\rho,t)$, integrated against the spectrum.
3. **Gaussian integrals.** $I_2, I_3, I_4$ have closed forms (arcsin, arctan) for erf activations.

Integrating the system and substituting back gives the closed-form generalization error:

$$ \epsilon_g = \frac1\pi\sum_{k,\ell} v^k v^\ell \arcsin\frac{Q^{k\ell}}{\sqrt{1+Q^{kk}}\sqrt{1+Q^{\ell\ell}}} \;-\; \sum_{k,n} v^k\tilde v^n \frac{R^{kn}}{\sqrt{2\pi}\sqrt{1+Q^{kk}}} \;+\; (\text{teacher term}). $$

---

# <span class="cat results">Result</span> The learning curve

<div class="columns">
<div class="col">

- Online dynamics versus $\alpha = P/N$ (which plays the role of training time): the order parameters and $\epsilon_g$ trace three phases.
- A plateau, then **specialization** (each student unit locks onto a teacher unit), then an asymptote.
- The **ODE prediction** (lines) sits on top of the finite-size **simulation** (crosses), with no fitting.

</div>
<div class="col">

<figure class="figure">

![w:620](./Meeting_260629/src/Presentation/media/paper_fig3_dynamics.png)

<figcaption>Goldt et al. Fig. 3: test error and order parameters vs. time t=steps/N; theory (lines) vs. simulation (crosses). N=10,000, D=100, K=M=2, rate 0.2, sign folding / erf activation.</figcaption>
</figure>

</div>
</div>

---

# <span class="cat results">Result</span> Reproduction, from scratch

<div class="columns">
<div class="col">

- A from-scratch **PyTorch** port of both the simulator and the ODE integrator.
- It tracks a single simulation to $\max|\Delta\epsilon_g| \approx 9\times10^{-4}$ across the whole trajectory, reproducing the paper's central claim.
- The small residual is only the difference between Python and C++ random number generation, not a modeling gap; it is cross-checked against the authors' C++.

</div>
<div class="col">

<figure class="figure">

![w:620](./Meeting_260629/src/Presentation/media/fig3_dynamics.png)

<figcaption>Our PyTorch reproduction at the same settings: ODE (lines) vs. one simulation (crosses), agreeing to about 1e-3. N=10,000, D=100, K=M=2, rate 0.2, sign/erf, seed 1.</figcaption>
</figure>

</div>
</div>


---


# <span class="cat results">Result</span> Structure helps generalization

<div class="columns">
<div class="col">

- Asymptotic error $\epsilon_g^{*}$ versus $\delta = D/N$, the intrinsic-dimension ratio (log-log axes).
- A lower intrinsic dimension gives better generalization: a more sharply folded, lower-dimensional manifold is easier to learn.
- Real datasets sit at small $\delta$; the markers indicate MNIST and CIFAR for context.

</div>
<div class="col">

<figure class="figure">

![w:620](./Meeting_260629/src/Presentation/media/fig5_eg_vs_delta.png)

<figcaption>Asymptotic test error vs. the dimension ratio D/N: a smaller latent dimension generalizes better. K=M=2, rate 0.2, sign/erf, orthogonalized teacher; D=25/50/100, mean over 3 seeds.</figcaption>
</figure>

</div>
</div>

---

<style scoped>
.columns .col { flex: 1; }
</style>

# <span class="cat results">Result</span> Specialization and model size

Larger students specialize many-to-one, and as complexity grows they separate to lower error.

<div class="columns">
<div class="col">

<figure class="figure">

![h:330](./Meeting_260629/src/Presentation/media/fig7_student_size.png)

<figcaption>Asymptotic error vs. student size K/M (teachers M=1,2,4; D=50,100; D/N=0.01, rate 0.2, sign/erf): student units specialize many-to-one, and gains saturate.</figcaption>
</figure>

</div>
<div class="col">

<figure class="figure">

![h:330](./Meeting_260629/src/Presentation/media/fig8_increasing_complexity.png)

<figcaption>Learning curves for growing student size K=1..8 (teacher M=10, N=500, D=25, rate 0.2, sign/erf): wider students break away later, fitting functions of increasing complexity.</figcaption>
</figure>

</div>
</div>


---


<style scoped>
section { font-size: 23px; }
</style>

# <span class="cat strategy">Discussion</span> Critique

Elegant and exact, but obtained under strong simplifications:

- **A deliberately simple model.** A single hidden layer and plain SGD, chosen so the equations of motion close. 
- **Online, not batch.** One-pass SGD with a fresh sample per step is atypical of modern (multi-epoch, mini-batch) training. Reusing samples might break the GEP.
- **A non-general task.** The target is a continuous, unbounded real number (MSE regression on a sum-of-sigmoids teacher with a linear, unsquashed readout), not the bounded or discrete classification that dominates real ML. (The two-colour manifold picture is for illustration.)
- **Not always a reduction.** The order parameters grow as $Q$ ($K\times K$), $R$ ($K\times M$), $T$ ($M\times M$), plus the spectral densities $r(\rho),\sigma(\rho)$. For a large student that is as many quantities as the model has parameters, so the "few macroscopic variables" framing is misleading. 

---

# <span class="cat ongoing">Outlook</span> Toward a quantum version

For the DQML project, the HMM is a starting point:

- We would need a complex-valued version of the dataset, with structure native to quantum states and circuits rather than real images.
- We would need to derive our own equations of motion for the quantum model. Our current setup (a CNN with classical communication) is heavy to simulate and not sure to have closed dynamics.
- The payoff is an analytical tool to argue where quantumness helps, which addresses the question that **"Better than classical?"** poses.

<div class="callout">

To demonstrate a real quantum advantage we need both data and theory that isolate the quantum resource. The HMM shows what solvable structured data can look like.

</div>

---

# Summary

- The **hidden manifold model** generates structured data: a low-dimensional latent variable, folded into high dimensions, labeled by a teacher.
- The **Gaussian Equivalence Property** reduces two-layer online-SGD learning to **closed ODEs** for a few order parameters, giving the exact learning curve.
- Results: smaller $\delta$ (more structure) helps; units specialize; wider students reach higher complexity; the curve is reproduced to $\approx 10^{-3}$.
- For us: a concrete example of "physics for AI", and a target to adapt to quantum machine learning.


---


# References

<div class="ref">

1. S. Goldt, M. Mézard, F. Krzakala, L. Zdeborová, **"Modeling the influence of data structure on learning in neural networks: The hidden manifold model,"** *Phys. Rev. X* **10**, 041044 (2020). arXiv:1909.11500.
2. J. Bowles, S. Ahmed, M. Schuld, **"Better than classical? The subtle art of benchmarking quantum machine learning models,"** arXiv:2403.07059 (2024).
3. G. Cybenko, *Math. Control Signals Systems* (1989); K. Hornik, *Neural Networks* (1991): the universal approximation theorems.
4. J. Kaplan et al., **"Scaling laws for neural language models,"** arXiv:2001.08361 (2020).

</div>



---


<!-- _class: closing -->
<!-- _paginate: false -->
<!-- _footer: '' -->

# Thank you

<div class="subtitle">Questions &amp; discussion</div>

<div class="meta">
<span class="author">Donghun Jung</span>
<span class="affil">Hidden Manifold Model · Journal Club · 2026-06-29</span>
</div>
