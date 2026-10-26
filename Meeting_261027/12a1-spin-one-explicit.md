---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
mjx-container[display="true"] { margin: 0.25em 0 !important; }
p { margin: 0.25em 0; }
</style>

# <span class="cat intro">Intro</span> Spin-1 representation

Two spin-½ in the same field evolve as $u\otimes u$; in the triplet and singlet basis it splits into $3\times3$ and $1\times1$ blocks.

$$
u=\begin{pmatrix}P&-Q^*\\Q&P^*\end{pmatrix}
\ \Longrightarrow\
u\otimes u=\begin{pmatrix}
P^2&-PQ^*&-PQ^*&Q^{*2}\\
PQ&\lvert P\rvert^2&-\lvert Q\rvert^2&-P^*Q^*\\
PQ&-\lvert Q\rvert^2&\lvert P\rvert^2&-P^*Q^*\\
Q^2&P^*Q&P^*Q&P^{*2}
\end{pmatrix}
\quad\text{in }(\ket{\uparrow\uparrow},\ket{\uparrow\downarrow},\ket{\downarrow\uparrow},\ket{\downarrow\downarrow})
$$

$$
V=\begin{pmatrix}
1&0&0&0\\
0&\tfrac{1}{\sqrt2}&0&\tfrac{1}{\sqrt2}\\
0&\tfrac{1}{\sqrt2}&0&-\tfrac{1}{\sqrt2}\\
0&0&1&0
\end{pmatrix}
\ \ \text{columns }\ket{{+1}},\ket{0},\ket{{-1}},\ket{S}
\qquad\Longrightarrow\qquad
V^\dagger(u\otimes u)V=\left(\begin{array}{ccc|c}
P^2&-\sqrt2\,PQ^*&Q^{*2}&0\\
\sqrt2\,PQ&\lvert P\rvert^2-\lvert Q\rvert^2&-\sqrt2\,P^*Q^*&0\\
Q^2&\sqrt2\,P^*Q&P^{*2}&0\\ \hline
0&0&0&1
\end{array}\right)
$$

- The singlet block is $\lvert P\rvert^2+\lvert Q\rvert^2=\det u=1$: the triplet never mixes with it.
- Cutting the $3\times3$ block gives $D^1(u)$, the NV evolution for every $\Delta$.
