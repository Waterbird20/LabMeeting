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

<!--
Backup: u = W R_x(phi) W multiplied out in both bases, with c = cos(phi/2), s = sin(phi/2), x = cos theta.

Z basis: diag(z^{-1}, z) times [[c, -is], [-is, c]] times diag(z^{-1}, z) = [[c z^{-2}, -is], [-is, c z^2]]. So P = c z^{-2}, Q = -is, and unitarity is c^2 + s^2 = 1 at once.

± basis: R_x(phi) = diag(e^{-i phi/2}, e^{i phi/2}) and W = [[x, -i sin theta], [-i sin theta, x]]. P_± is the sum of two paths, stay-stay, cos^2 theta e^{-i phi/2}, and flip-flip, (-i sin theta)^2 e^{i phi/2}: P_± = (cos^2 theta - sin^2 theta) cos(phi/2) - i (cos^2 theta + sin^2 theta) sin(phi/2) = c T_2(x) - is. The one-flip paths give <-|u|+> = -i sin theta cos theta (e^{-i phi/2} + e^{i phi/2}) = -i sin theta · 2cx, so Q_± = c U_1(x).

Unitarity: |P_±|^2 + (1 - x^2)|Q_±|^2 = c^2 T_2^2 + s^2 + (1 - x^2) c^2 U_1^2 = c^2 [T_2^2 + (1 - x^2) U_1^2] + s^2 = c^2 + s^2 = 1, using T_n^2 + (1 - x^2) U_{n-1}^2 = cos^2 n theta + sin^2 n theta = 1.

In the Pauli form: A = c T_2, B = s, C = c U_1, D = 0, as it must be for a palindromic sequence. And q = |Q|^2 = s^2 at every Delta: waits next to a Z preparation or readout only add phases. Here the Laurent form is the simpler one.
-->
