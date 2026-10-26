---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat intro">Intro</span> Spin-1 representation

A balanced drive moves the spin-1 like a spin-½. Equal, mirrored components ($B_1=B_2$, $\delta_1=-\delta_2=\varphi$, $\Delta_c=0$, $\Omega_{DQ}=\gamma B_{\rm mw}/2$) give a spin-1 and a spin-½ in the same effective field:

$$
\tilde H=\Delta S_z+\Omega_{DQ}(\cos\varphi\,S_x+\sin\varphi\,S_y)
\quad\longleftrightarrow\quad
h=\tfrac{\Delta}{2}\sigma_z+\tfrac{\Omega_{DQ}}{2}(\cos\varphi\,\sigma_x+\sin\varphi\,\sigma_y).
$$

The three levels are the symmetric triplet of two spin-½, and $\mathbf S=\mathbf s_1+\mathbf s_2$ puts the same field on both:

$$
\ket{{+1}}=\ket{\uparrow\uparrow},\quad
\ket{0}=\tfrac{1}{\sqrt2}\big(\ket{\uparrow\downarrow}+\ket{\downarrow\uparrow}\big),\quad
\ket{{-1}}=\ket{\downarrow\downarrow},\qquad
\tilde H=h\otimes\mathbb 1+\mathbb 1\otimes h\ \Rightarrow\ e^{-i\tilde Ht}=e^{-iht}\otimes e^{-iht}.
$$

Equivalently, $S_i$ and $\sigma_i/2$ obey the same commutation relations, so one generator gives one rotation, as a $3\times3$ or a $2\times2$ matrix; classically, $\langle\mathbf S\rangle$ precesses like a Bloch vector. Hence

$$
e^{-i\tilde Ht}=D^1\big(e^{-iht}\big)\ \text{ for every }\Delta
\quad\Longrightarrow\quad
\text{a sequence of pulses and waits acts as } D^1(u).
$$
