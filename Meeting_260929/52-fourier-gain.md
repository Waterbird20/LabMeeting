---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat results">Results</span> Most of the gain came from the Fourier (DFT) encoding

<!-- src: dqml-physics-results.md §2.4 table and bullets (3 QPUs x 4 qubits, contiguous windows, seeds 0-2; untrained linear-classifier column from phys-embed-layout, trained columns from phys-phase2f); revised - original +12.7 ± 1.1 (no comm.) and +5.9 ± 1.6 (links), 3/3 seeds each; gap without communication 4.1 -> 1.6 points; one-qubit-per-mode encoding loses 12 of 12 seed comparisons. Motivation: dqml-quantum-design.md §2.3.1 (random circular shift of the digit; the Fourier assignment makes the state equivariant under shifts); dqml-physics-results.md §2.3 last bullet (a cyclic shift s multiplies X_m by e^{-2 pi i s m/14}). -->
<!-- src (state column, rows 1 and 2): §2.1 (binary tree of amplitudes; the magnitude of mode m sets the angle a_m of tree node m; phi_m = arg X_m + c_m), §2.2 (node t opens basis state b(t) = (2p+1) 2^(n-k), whose amplitude is proportional to sin(theta_t/2); original: phi_m on basis state m+1; revised: phi_m on b(m); b(m) = 8, 4, 12, 2, 6, 10, 14, 1 at n = 4, never equal to m+1, so in the original the phase multiplies the half-angle sine of a different node: another mode's angle for m = 0, 1, 3, 5, 7, a trainable constant for m = 2, 4, 6), §2.3 (psi_b(m) = A_m sin(a_m/2) e^{i phi_m}, so rho_{0,b(m)} = C_0 A_m sin(a_m/2) e^{-i phi_m}: a copy of X_m). -->
<!-- Row (3) "one qubit per Fourier mode, modes in uniform superposition" (0.961 untrained / 0.846 ± 0.006 NC / 0.891 ± 0.018 CC; it trains worse than (2) in 12 of 12 seed comparisons, §2.4) and its bullet removed from the slide 2026-09-29 at the speaker's request; kept here for questions. Former src (state column, row 3): the vault gives only the name, §2.4 "one qubit per Fourier mode, modes in uniform superposition" (code layout="flat"; log.md 2026-09-26 "embedding option 4" calls it the direct-sum layout). With 8 modes in 16 amplitudes (n = 4) the name fixes the structure written here: 3 qubits index the mode in uniform superposition (amplitude 1/sqrt 8 each) and one qubit per mode carries X_m, i.e. C^16 = C^8 (x) C^2 = direct sum of 8 two-dimensional blocks. How |X_m| and arg X_m set the qubit |q_m> is NOT in the vault [unverified]; the definition is in ~/DQML/src/dqml/phys/SPEC.md and analysis/phys-embed-layout (WSL box). -->
<!-- terminology pass 2026-09-28 (GLOSSARY.md): "revised" = phase on |b(m)>, "original" = phase on |m+1>; "two-input links" = CC, 2 senders -> 1 receiver (short label 2->1). -->
<!-- Rows relabelled 2026-09-29 (speaker: "what is the difference between the 2nd row and the 3rd row?"): rows 1 and 2 are the same binary-tree state and differ only in which basis state carries arg X_m; row 3 is a different state: every mode gets its own qubit (a two-dimensional block), with the 8 blocks in equal superposition. Verifier pass 2026-09-29: row labels now say "tree state" for (1)/(2) and give the §2.4 name of (3) in full (GLOSSARY: describe it exactly as the source does); theta_t is defined in the cell (the angle of another tree node, i.e. another mode's a_{m'} or a trainable constant); the subscripts mode/qubit name the two registers of (3). The second setup line now names seeds 0-2 and the four classes, so 0.908 here is not confused with the held-out 0.890 on the accuracy slide. -->

<style scoped>table { font-size: 0.88em; margin: 0.45em 0 0.55em; } th, td { padding: 0.35em 0.6em; } td:nth-child(1), td:nth-child(2) { white-space: nowrap; } li { margin-bottom: 0.15em; } p { margin: 0.2em 0; }</style>

Block DFT $X_m=\lvert X_m\rvert e^{i\varphi_m}$; a cyclic shift only moves phases: $X_m\to e^{-2\pi i sm/14}X_m$.
NC / CC: no / classical communication ($2\to1$); 4 digits, seeds $0$–$2$, mean $\pm$ s.d.

| encoding | state ($a_m$: tree angle set by $\lvert X_m\rvert$) | linear classifier<br>(untrained) | trained, NC | trained, CC |
|---|---|---|---|---|
| (1) tree state,<br>phase on $\lvert m{+}1\rangle$ | $\psi_{m+1}\propto\sin\tfrac{\theta_t}{2}\,e^{i\varphi_m}$,<br>$\theta_t$: another node's angle | $0.773$ | $0.738\pm0.014$ | $0.848\pm0.027$ |
| **(2) tree state,<br>phase on $\lvert b(m)\rangle$** | $\psi_{b(m)}\propto\sin\tfrac{a_m}{2}\,e^{i\varphi_m}$ | $0.943$ | $\mathbf{0.865\pm0.014}$ | $\mathbf{0.908\pm0.002}$ |

- **(2) vs (1):** $+12.7\pm1.1$ points without, $+5.9\pm1.6$ with CC (three of three seeds).
- **Less underfitting:** training accuracy rises as much; NC gap $4.1\to1.6$ points.

<!-- Speaker note: (1) and (2) are the same binary-tree state of 16 amplitudes; |X_m| sets 8 of its 15 rotation angles, so every amplitude is a product of several modes' factors. They differ only in which basis state gets arg X_m: in (1) one whose amplitude another mode (or a constant) sets, so the phase of X_m is multiplied by the size of a different frequency; in (2) the state |b(m)> that X_m's own magnitude opens, so the coherence rho_{0,b(m)} is a copy of X_m and one linear measurement reads Re X_m and Im X_m (§2.0, §2.3). (Removed row, if asked:) (3) is a different state, called in the wiki "one qubit per Fourier mode, modes in uniform superposition" (§2.4): each mode gets its own qubit, and the eight mode blocks sit in equal superposition (weight 1/8 each; this reading of the name is not spelled out in the vault, see the src comment). -->
<!-- Speaker note (quote if asked): in the state, phi_m = arg X_m + c_m with a trainable offset c_m (§2.1); the DFT line on the slide shows only arg X_m. Mean ± s.d. over three seeds (0-2), 3 QPUs x 4 qubits, contiguous blocks. Training accuracy rises by about as much as test accuracy from (1) to (2), so the change removes underfitting rather than trading accuracy for variance (§2.4). The 12 of 12 are seed comparisons of (2) against (3). CC adds 11.0 points with the phase on |m+1> but only 4.3 with it on |b(m)> (differences of the §2.4 means: 0.848 - 0.738 and 0.908 - 0.865). In (3) the individual QPUs predict only 0.30-0.42 and succeed only in combination, and dephasing its encoded states removes 92 % of the model's information E, against 62 % for (2) (§2.4, §7). -->
<!-- EDIT-FORWARD: seeds 0-2 were also the seeds on which the revised encoding was chosen (App. G), so these three-seed differences favour it slightly; the held-out numbers are on the accuracy slide. -->
<!-- 2026-09-29 integration: visible task label "4 classes" -> "4 digits" (speaker's standing style: label accuracies with the task, "4 digits"). -->
