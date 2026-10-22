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

<!--
Backup: the Pauli form of a sequence of instantaneous pulses about plus or minus x, written without choosing a basis: u = A 1 - i B X - i sin theta (C Z + D Y), with A and B real polynomials in x of degree at most K, the number of waits, and the parity of K, C and D of degree at most K - 1 and the parity of K - 1, and A^2 + B^2 + (1 - x^2)(C^2 + D^2) = 1, which is det u = 1.

Dictionary: reading the same operator in the two bases. In the Z basis, P = <up|u|up> = A - i sin theta C and Q = <down|u|up> = -iB + sin theta D. In the ± basis of X, P_± = <+|u|+> = A - iB, and <-|u|+> = -i sin theta Q_± with Q_± = C - iD. The operators never change, only their matrices: X is diagonal and Z off-diagonal in the ± basis.

Induction. One pulse R_x(phi) = c 1 - i s X is K = 0: A = c, B = s, C = D = 0. A wait W = x 1 - i sin theta Z maps (A, B, C, D) to (xA - (1 - x^2)C, xB - (1 - x^2)D, A + xC, B + xD): the {1, X} block goes into itself times cos theta and into {Z, Y} times sin theta, and {Z, Y} comes back through sin^2 theta = 1 - x^2. Degrees rise by one and parities flip. A pulse R_x(phi) maps (A, B, C, D) to (cA - sB, cB + sA, cC + sD, cD - sC): it keeps the two blocks apart.

Symmetries. W(-Delta) = X W(Delta) X and X R_x X = R_x, so u(-Delta) = X u(Delta) X; in the Z basis P(-Delta) = P(Delta)* and Q(-Delta) = -Q(Delta)*, i.e. P has real Laurent coefficients and Q imaginary ones. For a palindromic sequence u^T = u, so Q is purely imaginary, D = 0, and Q = -i B(x), a real Chebyshev series. Pulses about general equatorial axes break both symmetries; that is generalized QSP (Motlagh and Wiebe, PRX Quantum 2024).
-->
