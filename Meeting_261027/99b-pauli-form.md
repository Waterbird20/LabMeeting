---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

# <span class="cat method">Backup</span> Pauli form

The Pauli form of a $\pm x$ sequence holds in every basis.

$$
u=A(x)\,\mathbb{I}-iB(x)\,X-i\sin\theta\,\big[C(x)\,Z+D(x)\,Y\big],\qquad A^2+B^2+(1-x^2)(C^2+D^2)=1
$$

$A,B$ real, degree $\le K$, parity $K$; $C,D$ real, degree $\le K-1$, parity $K-1$ ($K$ waits).

| | Z basis | $\pm$ basis |
|---|---|---|
| stay | $P=A-i\sin\theta\,C$ | $P_\pm=A-iB$ |
| flip | $Q=-iB+\sin\theta\,D$ | $\bra{-}u\ket{+}=-i\sin\theta\,Q_\pm,\ \ Q_\pm=C-iD$ |
| unitarity | $\lvert P\rvert^2+\lvert Q\rvert^2=1$ | $\lvert P_\pm\rvert^2+(1-x^2)\lvert Q_\pm\rvert^2=1$ |

- A wait maps $(A,B,C,D)\to\big(xA-(1-x^2)C,\ xB-(1-x^2)D,\ A+xC,\ B+xD\big)$.
- $u(-\Delta)=X\,u(\Delta)\,X$, so $P(-\Delta)=P(\Delta)^*$ and $Q(-\Delta)=-Q(\Delta)^*$.
- A palindromic sequence has $u^T=u$, so $D=0$ and $Q=-iB(x)$.
