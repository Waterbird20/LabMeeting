---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>.columns .col:first-child { flex: 0 0 545px; } li { margin-bottom: 0.4em; } .def { font-size: 0.9em; margin: -0.3em 0 0 1.55em; line-height: 1.5; } .def p { margin: 0; } .callout { margin: 0.5em 0 0; } figure.figure { margin: 0.35em auto 0; }</style>

# <span class="cat method">Method</span> Why the choice of phase-carrying basis state matters

<div class="columns">
<div class="col">

- One copy: $P=\mathrm{Tr}[E\rho]$ is linear in $\rho_{xy}=\psi_x\psi_y^*$
  $$\rho_{b(m),0}=C_0A_m\sin(a_m/2)\,e^{i(\arg X_m+c_m)}$$

<div class="def">

$C_0=\psi_{0000}=\prod_{k=0,1,3,7}\cos(a_k/2)\ge0$
$|\psi_{b(m)}|=A_m\sin(a_m/2)$, $A_m\ge0$:
the other $\cos$, $\sin$ factors on its path, free of $a_m$

</div>

</div>
<div class="col">

- **Phase on $|b(m)\rangle$:** a rescaled copy of $X_m$
- **Phase on $|m{+}1\rangle$:** same angle, radius without $|X_m|$

<div class="callout">

No communication, phase on $|m{+}1\rangle\to|b(m)\rangle$:
test accuracy $0.738\to0.865$ (4 digits, seeds 0–2)

</div>

</div>
</div>

<figure class="figure">

![w:940](media/coherence-argand.png)

</figure>

<!-- Speaker note: a single-copy measurement gives P = Tr[E rho], linear in the entries rho_xy = psi_x psi_y^*, so what matters is which entry carries X_m. Amplitudes of the tree (slide before): a 0-branch multiplies by cos(theta/2), a 1-branch by sin(theta/2). The all-zero path runs through the nodes of modes 0, 1, 3, 7, so psi_0000 = C_0 = cos(a_0/2) cos(a_1/2) cos(a_3/2) cos(a_7/2): real, never carries a phase (phi_0 = 0). |b(m)> takes a_m's 1-branch, so psi_{b(m)} = A_m sin(a_m/2) e^{i(arg X_m + c_m)}, where A_m collects the cosines and sines of the other nodes on its path (the ancestors, then the cosines below node m) and does not contain a_m. Example b(2) = |1100>: A_2 = sin(a_0/2) cos(a_6/2) cos(t/2), t the constant angle below (node (4,6)); the slide's line |psi_{b(m)}| = A_m sin(a_m/2) says exactly this: A_m is every other cos / sin factor on the path of |b(m)>. Hence rho_{b(m),0} = psi_{b(m)} psi_0^* = C_0 A_m sin(a_m/2) e^{i(arg X_m + c_m)}. (If a trained constant angle leaves [0, pi], A_m picks up a fixed sign; that is a constant pi absorbed into c_m.) The figure: mode 2 of QPU 1's feature block, all 411 test signals, untrained encoding (c_m = 0). Left: X_2 itself; orange = the quarter of signals with the smallest |X_2|, whose phase is mostly noise. Middle, phase on |b(2)>: every point keeps the angle of X_2 and its radius grows with |X_2| (through sin(a_2/2); across signals the rank correlation is 0.53, because C_0 A_2 also varies with modes 0, 1, 3, 6, 7), so the orange points stay near 0: a linear readout weights each phase by its own magnitude. Right, phase on |3> = |0011> (the earlier choice): the angle is still exactly arg X_2 (psi_0 is real and >= 0 in both placements), but the radius is cos^2(a_0/2) cos^2(a_1/2) sin(a_3/2) cos(a_3/2) cos(a_7/2) sin(t'/2), t' another constant angle (node (4,1)): no factor of |X_2| at all (the middle radius has modes 0, 1, 3, 7 too, through C_0, but also sin(a_2/2)). Titles give both states in binary, |1100> and |0011>, as on the circuit slide. The noisy small-|X_2| phases get full weight. So the old placement did not scramble the phase; it paired each phase with the wrong magnitude. Fixing that alone: 0.738 -> 0.865 without communication (seeds 0-2). -->
<!-- src: P = Tr[E rho], entries rho_xy = psi_x psi_y^*: dqml-physics-results.md §2.0 first bullet. Phase placement: revised on b(m), original on m+1, "all remaining angles and phases are trainable constants": §2.1, §2.2 (b(t) = (2p+1) 2^{n-k}; b(2) = 12 = |1100>, original slot of mode 2 = 3 = |0011>; table column "amplitude of the original state is set by"). -->
<!-- src: C_0 = prod_k cos(theta_{t_k(0)}/2) > 0 "(modes 0, 1, 3, 7 at n = 4)"; amplitude of |b(m)> = A_m sin(a_m/2) e^{i varphi_m}, A_m "a product of cosines and sines along the path to node m and of cosines below it"; "The coherence carries the phase of X_m exactly ... arg rho_{b(m),0} = arg X_m + c_m"; "In the original encoding the phase multiplies an amplitude set by a different mode": dqml-physics-results.md §2.3 (lines ~220-231). The slide writes rho_{b(m),0} = psi_{b(m)} psi_0^* (argument +arg X_m + c_m), the entry the figure plots; §2.3 writes its complex conjugate rho_{0,b(m)} = C_0 A_m sin(a_m/2) e^{-i varphi_m}. "Both >= 0": the data angles are clipped to [0, pi], so their half-angle cosines and sines are >= 0 (C_0 = 0 only when a path angle is clipped to pi); A_m is defined on the slide by |psi_{b(m)}| = A_m sin(a_m/2), i.e. the modulus of the other path factors, so it is >= 0 by construction (the vault does not state a sign for A_m; 43-fourier.md src: >= 0 when the constant angles lie in [0, pi]). "A rescaled copy of X_m": "a copy of the complex Fourier coefficient, with its modulus rescaled" (§2.0). The modulus grows monotonically with |X_m| only for modes 2, 4, 5, 6; modes 0, 1, 3, 7 fold back beyond a_m = pi/2 (§2.3), hence "rescaled copy", not "grows with". -->
<!-- src: 0.738 ± 0.014 (original) -> 0.865 ± 0.014 (revised), trained, no communication, 3 QPUs x 4 qubits, contiguous windows, seeds 0-2: dqml-physics-results.md §2.4 table (which says "three seeds"); the three seeds are 0-2: App. G ("Seeds 0–2 were used for design choices. The revised encoding was chosen on them") and the §2.4 per-window bullet. -->
<!-- src: figure media/coherence-argand.png from 3. wiki/code/lm-260929-animations/fig_coherence.py (2026-09-29; deterministic, point order rng seed 42): untrained encoding (percentile_scale_offset scale/offset, c_m = 0, constant tree angles pi/2, as fig_linear.py), QPU 1 feature block (features 14-27), mode 2, all 411 test signals of the four digits; panels X_2, rho_{b(2),0} (phase on |1100>), rho_{3,0} (phase on |0011>); orange = lowest 25 % of |X_2|; each panel scaled to 1.06 x its own 99th-percentile radius, no ticks; the few points outside a square panel (2, 4, 2 of 411; the two in the right panel are orange and move in < 2 %) are pulled in radially to its edge, angle kept, so all 411 are drawn; titles "phase on |b(2)> = |1100>", "phase on |3> = |0011>". Its _verify() (run 2026-09-29, all checks passed): arg rho_{b(2),0} = arg X_2 to 2.9e-16 on the 391 signals with a nonzero coherence (20 are 0 because an angle is clipped to 0 or pi); |rho_{b(2),0}| = C_0 A_2 sin(a_2/2) (closed form of §2.3) to 5.6e-17, with C_0 = cos(a_0/2) cos(a_1/2) cos(a_3/2) cos(a_7/2) and A_2 = sin(a_0/2) cos(a_6/2) cos(pi/4); original placement: arg rho_{3,0} = arg X_2 to 3.4e-16 (the angle is kept), |rho_{3,0}| = cos^2(a_0/2) cos^2(a_1/2) sin(a_3/2) cos(a_3/2) cos(a_7/2) sin(t'/2) (t' = constant angle of node (4,1); t in A_2 is node (4,6)), no a_2, hence the panel note "radius without |X_2|"; median radius of the smallest-|X_2| quartile / 99th-percentile radius: X_2 0.150, phase on |b(2)> 0.066, phase on |3> 0.435; Spearman |rho| vs |X_2|: +0.533 (b(2)) vs -0.130 (|3>). Replaces the trained-model scatter media/embedding-coherence.png (wiki Fig. 3f; speaker 2026-09-29: "the plot is not intuitive"). -->
<!-- EDIT-FORWARD: the modulus is monotone in |X_m| only for modes 2, 4, 5, 6; for modes 0, 1, 3, 7 (nodes on the path to |0000>) it goes as sin(a_m)/2 and folds back beyond a_m = pi/2, and 19-30 % of test signals are beyond the fold for modes 0, 1, 3 (§2.3). The figure shows mode 2, which is monotone. Say so if asked. -->
<!-- EDIT-FORWARD: with communication the same change gives 0.848 -> 0.908 (§2.4); the results section (resacc, 52-fourier-gain) owns the full comparison. -->
<!-- 2026-09-29 integration: visible task label "4 classes" -> "4 digits" (speaker's standing style: label accuracies with the task, "4 digits"). -->
<!-- 2026-09-29 revision (speaker: "What is A_m and C_0? Also, the plot is not intuitive."): C_0 and A_m defined on the slide, consistent with the circuit on 43-fourier.md (0-branch cos, 1-branch sin; all-zero path through a_0, a_1, a_3, a_7); formula switched to rho_{b(m),0} (angle +arg X_m, as plotted); old "mode 0's phase times mode 7's magnitude: no meaning for the class" replaced by "same angle, radius from other modes": the old placement keeps the angle and breaks the pairing of phase and magnitude; figure replaced; "3 seeds" -> "seeds 0-2"; figure label "trained model, mode 2" removed (the figure states its own setup). -->
<!-- 2026-09-29 fix pass (verifier): A_m line now defines A_m through |psi_{b(m)}| = A_m sin(a_m/2), A_m >= 0 "the other cos, sin factors on its path, free of a_m" (path as on the circuit of 43), ">= 0" moved onto each line; right bullet "radius from other modes" -> "radius without |X_m|" (the middle radius also holds modes 0, 1, 3, 7 via C_0; the real difference is the missing |X_m| factor), figure note changed the same way; figure titles add the binary states, Re label cleared of the legend, Im label on a white backing, off-panel points pulled to the edge; speaker note: second constant angle renamed t'. -->
