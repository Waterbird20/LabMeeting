---
marp: true
theme: serif
math: mathjax
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

