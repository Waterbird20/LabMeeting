---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>table { font-size: 18px; width: 100%; margin: 0.2em 0 0.5em; } table td, table th { padding: 4px 8px; white-space: nowrap; } table td:first-child { font-weight: 600; } table td:last-child { white-space: normal; } ul { margin-top: 0.3em; } li { margin: 0.3em 0; } .noadv { color: var(--accent); } .src { font-size: 0.6em; line-height: 1.35; }</style>

# <span class="cat strategy">Strategy</span> Joint functions $f(x,y)$: quantum vs classical communication

| $f(x,y)$ | model (bounded error) | qubits | bits | reference |
|---|---|---|---|---|
| equality $[x=y]$ | simultaneous (SMP), no shared randomness | $O(\log n)$ | $\Theta(\sqrt n)$ | Buhrman et al. 2001 |
| Hamming distance $\le d$ ($d$ fixed) | SMP, no shared randomness | $O(\log n)$ | $\Theta(\sqrt n)$ | Yao 2003 |
| disjointness $[x\cap y=\emptyset]$ | two-way | $\Theta(\sqrt n)$ | $\Theta(n)$ | Aaronson, Ambainis 2003 |
| Boolean hidden matching, variant (promise) | one-way | $O(\log n)$ | $\Theta(\sqrt n)$ | Gavinsky et al. 2007 |
| vector in subspace (promise) | quantum one-way, classical two-way | $O(\log n)$ | $\Omega(n^{1/3})$ | Klartag, Regev 2011 |
| inner product mod 2: <span class="noadv">no advantage</span> | two-way, even with entanglement | $\Theta(n)$ | $\Theta(n)$ | Cleve et al. 1998 |

- **Our architecture:** the senders hold $x$ and $y$; the receiving QPU, given no input, is the SMP referee.
- **First target, equality:** SMP matches our $2$ senders $\to$ $1$ receiver, and the referee needs only a SWAP test.
- **Measure** worst-case error vs message size, CC vs QC (trained parameters act as a shared key).

<div class="src">Buhrman, Cleve, Watrous, de Wolf, PRL 87, 167902 (2001); Yao, STOC 2003; Buhrman, Cleve, Wigderson, STOC 1998; Aaronson, Ambainis, Theory Comput. 1, 47 (2005); Razborov, Izv. Math. 67, 145 (2003); Bar-Yossef, Jayram, Kerenidis, STOC 2004; Gavinsky, Kempe, Kerenidis, Raz, de Wolf, STOC 2007; Klartag, Regev, STOC 2011; Cleve, van Dam, Nielsen, Tapp, LNCS 1509, 61 (1998).</div>
