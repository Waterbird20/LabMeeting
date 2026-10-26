---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

# <span class="cat method">Backup</span> Example ($K=2$)

Multiplying out $u=W\,R_x(\phi)\,W$ in both bases checks unitarity; $c=\cos\frac\phi2$, $s=\sin\frac\phi2$.

**Z basis.**

$$
u=\begin{pmatrix}z^{-1}&0\\0&z\end{pmatrix}\begin{pmatrix}c&-is\\-is&c\end{pmatrix}\begin{pmatrix}z^{-1}&0\\0&z\end{pmatrix}=\begin{pmatrix}c\,z^{-2}&-is\\-is&c\,z^{2}\end{pmatrix},\qquad \lvert P\rvert^2+\lvert Q\rvert^2=c^2+s^2=1
$$

**$\pm$ basis.** $R_x(\phi)=\mathrm{diag}(e^{-i\phi/2},e^{i\phi/2})$, $W=\begin{pmatrix}x&-i\sin\theta\\-i\sin\theta&x\end{pmatrix}$:

$$
P_\pm=\cos^2\theta\,e^{-i\phi/2}+(-i\sin\theta)^2e^{i\phi/2}=c\,T_2(x)-is,\qquad Q_\pm=c\,U_1(x)
$$

$$
\lvert P_\pm\rvert^2+(1-x^2)\lvert Q_\pm\rvert^2=c^2\big[T_2^2+(1-x^2)U_1^2\big]+s^2=c^2+s^2=1
$$

- The bracket is $T_n^2+(1-x^2)U_{n-1}^2=\cos^2n\theta+\sin^2n\theta=1$.
- $q=\lvert Q\rvert^2=s^2$ at every $\Delta$.
