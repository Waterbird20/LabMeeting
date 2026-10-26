---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat method">Method</span> QSP sequence

A product of pulses and waits is one SU(2) matrix.

<figure class="figure">

![w:700](media/qsp-timeline.png)

</figure>

$$
u=R_x(\phi_{K+1})\,R_z(\Delta\tau)\,R_x(\phi_K)\cdots R_z(\Delta\tau)\,R_x(\phi_1)\ \in\ \mathrm{SU}(2)
$$

Any SU(2) matrix is a rotation, $u=e^{-i\alpha\,\mathbf n\cdot\boldsymbol\sigma}=\cos\alpha\,\mathbb 1-i\sin\alpha\,(n_xX+n_yY+n_zZ)$:

$$
u=\begin{pmatrix}\cos\alpha-in_z\sin\alpha&-(n_y+in_x)\sin\alpha\\ (n_y-in_x)\sin\alpha&\cos\alpha+in_z\sin\alpha\end{pmatrix}
=\begin{pmatrix}P&-Q^*\\ Q&P^*\end{pmatrix},\qquad \lvert P\rvert^2+\lvert Q\rvert^2=1 .
$$

- Only the waits depend on the detuning, so $P$ and $Q$ are functions of $\theta=\Delta\tau/2$.
