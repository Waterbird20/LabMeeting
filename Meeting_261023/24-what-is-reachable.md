---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
.columns .col:first-child { flex: 0 0 640px; }
.columns { gap: 1.0rem; }
</style>

# <span class="cat method">Method</span> Reachable polynomials

$K$ waits and $K+1$ pulses about $x$ make exactly these polynomials:

$$
u=A\,\mathbb I-iB\,X-i\sin\theta\,(C\,Z+D\,Y),\qquad W=x\,\mathbb I-i\sin\theta\,Z,\qquad R_x(\phi)=\cos\tfrac\phi2\,\mathbb I-i\sin\tfrac\phi2\,X
$$

<div class="columns">
<div class="col">

<figure class="figure">

<video src="media/qsp-pauli-build.mp4" poster="media/qsp-pauli-build.png" width="630" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
<div class="col">

| real polynomials | degree | parity |
|---|---|---|
| $A,\ B$ | $\le K$ | of $K$ |
| $C,\ D$ | $\le K-1$ | of $K-1$ |

$$
A^2+B^2+(1-x^2)(C^2+D^2)=1
$$
$$
P=A-i\sin\theta\,C,\quad Q=-iB+\sin\theta\,D
$$

Conversely, every such $A,B,C,D$ is made by $K+1$ pulses about $x$:
$$
\underbrace{2(K+1)}_{\text{coefficients}}-\underbrace{(K+1)}_{\text{unitarity}}=\underbrace{K+1}_{\text{angles}}
$$

<p class="src">Low, Yoder, Chuang, PRX 6, 041067 (2016); Martyn, Rossi, Tan, Chuang, PRX Quantum 2, 040203 (2021), Thm. 1</p>

</div>
</div>

<!--
Now we want to know exactly what K waits and K + 1 pulses can produce. The cleanest way is to write the unitary in the Pauli basis and follow it through the sequence, one factor at a time.

We take pulses about +x or -x. Any 2 x 2 unitary of determinant one can be written as u = A times the identity, minus i B X, minus i sin(theta) times (C Z + D Y). The factor sin(theta) is pulled out of the Z and Y parts on purpose.

The wait is W = e^{-i theta Z} = x I - i sin(theta) Z, with x = cos(theta). A pulse about x is R_x(phi) = cos(phi/2) I - i sin(phi/2) X.

The clip follows four pulses, angles pi/2, 2 pi/3, pi/3, pi/4, with three waits, and plots A, B, C, D as functions of x after every factor.

- It starts with the first pulse alone: A = cos(phi_1 / 2) and B = sin(phi_1 / 2) are constants, C = D = 0.
- One wait: the identity part of W multiplies everything by x. The Z part turns the identity into Z and X into Y, each with one factor sin(theta); and it turns Z and Y back into the identity and X with a factor sin^2(theta) = 1 - x^2. Written out: A goes to xA - (1 - x^2) C, B to xB - (1 - x^2) D, C to xC + A, D to xD + B. Every coefficient becomes a polynomial of one degree more, and every parity flips. The counter at the top goes up by one.
- One pulse: since X commutes with I and X, and turns Z into Y, the pulse only rotates the pair (A, B) into itself and the pair (C, D) into itself, with the constant coefficients cos(phi/2) and sin(phi/2). The curves change shape, but the degrees do not.

After three waits: A and B have degree at most 3 and are odd, C and D have degree at most 2 and are even. And the unitarity identity A^2 + B^2 + (1 - x^2)(C^2 + D^2) = 1 holds for every x; it is det u = 1.

In the up/down basis the two entries are P = A - i sin(theta) C, the stay amplitude, and Q = -i B + sin(theta) D, the flip amplitude.

The important part is the converse, the theorem of quantum signal processing: every set of real polynomials A, B, C, D with these degrees and parities that satisfies the unitarity identity is produced by some choice of K + 1 pulse angles about x. It was proven by Low, Yoder and Chuang (2016); the form with polynomials in x and the two parity classes is Theorem 1 of Martyn, Rossi, Tan and Chuang (2021), who write the pair as A - iB and C - iD.

The count shows that nothing is left over. A, B, C, D together have 2(K + 1) real coefficients. The unitarity identity is one even polynomial of degree 2K, so it imposes K + 1 equations. That leaves K + 1 free numbers: exactly one angle per pulse.

A useful consequence for design: any real function of x with the parity of K and modulus at most one can be realised as A, or with a pulse phase shift as B, of some sequence. So we pick one real polynomial and the rest follows. The pulses are then obtained from the polynomial by a finite algebraic procedure: complete the unitary, then strip one pulse and one wait at a time (Haah, Quantum 3, 190 (2019)).

The waits fix the degree; the pulse angles choose the polynomials.
-->
