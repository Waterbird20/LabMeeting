---
marp: true
theme: serif
math: mathjax
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

![h:290](media/fig_b_teacher.png)

<figcaption>Teacher (frozen): maps the latent c to the continuous label y*. Schematic; M=2 units drawn.</figcaption>
</figure>

</div>
<div class="col">

<figure class="figure">

![h:290](media/fig_b_student.png)

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

![w:560](media/fig2_hidden_manifold.png)

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

![w:580](media/paper_fig1_manifold.png)

<figcaption>Goldt et al. Fig. 1 (conceptual illustration): recognizable CIFAR images lie on the manifold; off-surface points are random noise.</figcaption>
</figure>

</div>
</div>
