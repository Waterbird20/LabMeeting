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

<!--
Why the spin-1 representation is natural.

Start from the rotating-frame Hamiltonian of the previous slide and choose the two frequency components with equal amplitude, B_1 = B_2 = B_mw, mirrored phases delta_1 = phi and delta_2 = -phi, and no common-mode detuning. The off-diagonal elements become Omega_DQ / sqrt2 times e^{-i phi}, with Omega_DQ = gamma B_mw / 2, and these are exactly the matrix elements of Omega_DQ (cos phi S_x + sin phi S_y). So H-tilde is a spin-1 in an effective field: Delta along z and Omega_DQ in the equatorial plane at azimuth phi. Replace S by sigma/2 and you have h, a spin-1/2 in the same field.

Three ways to see that the two evolve "the same way".

First, two spin-1/2. Take two spin-1/2 particles and put them in the same field. Their total spin is S = s_1 + s_2, and one-half times one-half is one plus zero: the three symmetric states up-up, the symmetric combination, and down-down form a spin-1, the singlet is a spin-0. Our three levels are exactly that triplet: |+1> is up-up, |0> is the symmetric combination, |-1> is down-down. A Hamiltonian linear in S is the sum of the same h acting on each particle, h (x) 1 + 1 (x) h, so the evolution is u (x) u, each particle rotated by the same u, and it never leaves the triplet. The spin-1 evolution is u (x) u restricted to the triplet; this matrix is called D^1(u). The DQ transition, +1 to -1, flips both particles.

Second, algebra. The spin-1 matrices S_x, S_y, S_z obey the same commutation relations as the Pauli matrices divided by two. A rotation is the exponential of a combination of generators, and the result of composing rotations depends only on these commutation relations. So the same generator exponentiates to "the same rotation", represented once by 2x2 and once by 3x3 matrices: e^{-i H-tilde t} = D^1(e^{-i h t}) for every Delta. D^1 maps products to products, so a whole sequence of pulses and free evolution acts on the NV as D^1(u), with u the evolution of the same sequence on a spin-1/2.

Third, the classical picture. The expectation value of S obeys the same precession equation as a spin-1/2 Bloch vector, d<S>/dt = field x <S>. A spin-1 is a vector that rotates exactly like the spin-1/2 Bloch vector.

Consequence: we can design pulses for a spin-1/2 and read off what they do to the NV.

The correspondence needs one common field on both spin-1/2. It fails for a common-mode detuning (S_z^2 is an Ising coupling between the two particles), unequal amplitudes, and the transverse zero-field splitting E. With equal field per component the single-quantum Rabi frequency is Omega_SQ = sqrt2 Omega_DQ.
-->
