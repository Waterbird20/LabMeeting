# 260929 deck glossary: standard terminology (binding for every slide, caption, table, figure label and clip text)

Not a slide. The speaker (2026-09-28): "Overall, do not use our own slang. Use standard terminology in
quantum information, physics and machine learning." Built from a sweep of 214 candidate terms, a
consolidated glossary and an adversarial critique, adjudicated by the integrator. One name per thing,
everywhere. Code names may remain only inside HTML comments. Where no standard term exists, describe
the thing in plain words at first use (the "describe" column), then use the short form.

Updated 2026-09-29 (round 3): mutual information, $I(S;Y)$ and $D_\mathrm{add}$ marked obsolete (not in visible
text), slide 57 removed, and the name of the $0.978$ classical model added (results page §0.1).

## Communication and architecture

| in-house term (do not use) | use instead | anchor |
|---|---|---|
| none, isolated QPUs, local only, "fresh \|00⟩" as the baseline | **no communication (NC)**, i.e. local operations only (LO) | Hwang et al. 2025 (NC/CC/QC); Chitambar et al. 2014 |
| the bits, link bits, exchange classical bits, measured bits that condition gates | **classical communication (CC)**; one transmitted bit is the **message bit $s$** (the receiving QPU gets a "received message bit"); its effect is a **classically controlled gate** | Hwang et al. 2025; Nielsen and Chuang §4.4 |
| quantum channel / quantum link / "the qubits themselves" (meaning: the pooled qubits are sent) | **quantum communication (QC): the pooled qubits are sent to the receiving QPU**. At first use (26; `57-quantum-channel.md` was removed on 2026-09-29) add: "in Hwang et al., QC is realized by non-local two-qubit gates between QPUs; here the qubits themselves are transmitted" (26's source line says so). Keep **"quantum channel" only for a CPTP map** (e.g. $\mathcal E$ on 22, "the QPU is still a quantum channel" on 46) | Wilde, *Quantum Information Theory*; Nielsen and Chuang ch. 8 |
| one big / one large QPU, single-QPU model, "several small QPUs" | **monolithic QPU** (one 12-qubit processor) vs **distributed QPUs** (three modules of $n=4$ qubits) | Caleffi et al., Computer Networks 254, 110672 (2024) |
| the cut, cross the cut, what crosses between QPUs | **partition** of the qubits across QPUs; **inter-QPU (non-local) gates**; "exchanged between QPUs" | Caleffi et al. 2024; Eisert et al. 2000 |
| crossing budget, crossing point | **the same communication cost at the same circuit location** | Kushilevitz and Nisan 1997 |
| ladder, rung | **communication resources NC ⊂ CC ⊂ QC**, compared with a monolithic QPU (column header "communication"). Cite Chitambar et al. only for LO ⊂ LOCC | Hwang et al. 2025 |
| communication patterns, pattern(s), who sends to whom | **communication topology** | networking / DQC |
| none / one-input / two-input (default) / fed back / broadcast (labels) | **NC**; **1 sender → 1 receiver** ($i\to i{+}1$); **2 senders → 1 receiver** ($(i,i{+}1)\to i{+}2$, default); **2 senders → both senders** ($(i,i{+}1)\to i, i{+}1$); **2 senders → all 3 QPUs (broadcast)**. Short figure labels: "NC", "1→1", "2→1 (default)", "2→2 (to senders)", "2→3 (broadcast)", with the legend "senders → receivers" | plain description |
| link (as the trainable rule), trainable links, what the links learned, link input | the trainable rule is the **decision function** $g(m_i,m_j)=a\,m_i+b\,m_j+c\,m_im_j+d$ (bit $s=1$ when $g>0$ up to shot noise); "What the trained decision functions compute"; there are six (2 rounds × 3 messages). "Classical link" may be used only for the communication edge itself | ML: decision function / decision boundary $g=0$ (Bishop §4.1; Hastie et al. §4) |
| test, threshold test, fixed test, XOR test, "the logic a link ends with" | **threshold** of $g$; "fixed coefficients $(a,b,c)$"; "XOR-type coefficients $(1,1,-2)$", "OR-type coefficients $(1,1,-1)$"; "the Boolean function it converges to". Never "test" (clashes with test set and SWAP test) | O'Donnell 2014 |
| soft threshold, random bit | **stochastic threshold unit (probit)**: $s=1$ with probability $\pi=\Phi(g/\sqrt{\sigma_g^2+\epsilon^2})$, $\Phi$ the standard normal CDF | Bishop §4.3.5; Bengio et al. 2013 |
| two-input unit, product term, sigma-pi | a unit with a **bilinear (interaction) term** $c\,m_im_j$ | |
| "degree-2 PTF" of $g$ on $[0,1]^2$ | "the threshold of the bilinear polynomial $g$; restricted to $\{0,1\}^2$ it is one of the 16 Boolean functions of two bits" | O'Donnell 2014 |
| floor $\epsilon$ | **noise floor** $\epsilon$ | |
| colder bit | "a lower effective temperature, i.e. a more deterministic bit" (keep "temperature") | Hinton and Sejnowski 1986 |
| gradient window | **region of non-vanishing gradient** | |
| K rises geometrically / starts small / annealing of K | **shot schedule**: $K$ increased geometrically from $10^2$ to $10^4$ (no "annealing") | Kübler et al., Quantum 4, 263 (2020) |
| measured probabilities, "this number is sent to the other QPUs" | **estimated outcome probability** $\hat m$ from $K$ shots. Correct statement: "the estimates are sent over the classical network to evaluate the decision function; the receiving QPU gets only the message bit $s$" | |
| bit rate, activation rate | $P(s=1)$ | |
| label information, bit-label information | **Obsolete (2026-09-29): not in visible text.** The speaker: "remove mutual information stuff"; no slide, figure or clip shows it (the $I(S;Y)$ row of `patterns.png` and the $I(S;Y)$ panel of the slide-58 crop were removed). Values may stay in source comments as provenance. Former rule: mutual information $I(S;Y)$ between the message bits and the class label, never $I(s;Y)$ | Cover and Thomas ch. 2 |
| measure once, send the outcome; sender also uses it / does not | "**CC, single-shot outcome sent** (measure-and-prepare)"; "sender also applies local feed-forward" / "no sender feed-forward" | Horodecki, Shor and Ruskai 2003 |
| $A_\text{bits}$, $A_\text{none}$, $A_\text{qubits}$ | $A_\text{CC}$, $A_\text{NC}$, $A_\text{QC}$ | Hwang et al. notation |
| used coherently | "the receiver processes the received qubits coherently" (26's callout). The former "completely dephasing them costs 0.45 nats" is **obsolete**: it was on the removed slide 57, and cross-entropy is not quoted as a metric (2026-09-29) | Wilde (dephasing channel) |
| the two received qubits are entangled (bipartition unclear) | "the two received qubits are entangled **with each other** in 96–97% of test signals" (§8.1; not with the receiver's register). Was used on the removed slide 57 only; keep the wording if it is asked | source |
| own test, "own", two QPUs (reuse slide) | "**local classical control** from the QPU's own $K$-shot estimate" vs "**control from two other QPUs' estimates**" (CC) | |
| feed-forward, for the model's message bit | reserve "classical feed-forward" for single-shot conditioning (e.g. $T_\mu$, the single-shot CC rows). For the message bit: "classically controlled gate conditioned on a $K$-shot estimate (uses many copies)" | Córcoles et al., PRL 127, 100501 (2021) |
| efficient network topologies | "topologies with the best accuracy per transmitted bit" | |
| link pruning, remove communication links | "**remove message edges of the communication graph** (at random) and retrain"; note that lottery-ticket pruning (Frankle and Carbin 2019) is magnitude-based | |
| receiver (fingerprinting) | "referee" only inside the simultaneous-message-passing model; in our model "a QPU plays the referee" | Buhrman et al. 2001 |

## Circuit and measurement

| in-house term | use instead | anchor |
|---|---|---|
| pooling stage | **pooling layer** | Cong, Choi, Lukin 2019 |
| intermediate measurement, measured and discarded, measured vs read out | **mid-circuit measurement** (the qubit is then traced out); "pooled (mid-circuit-measured) qubits" vs "output qubits (measured at the end)" | Córcoles et al. 2021 |
| measured-position qubits | **pooled qubits** | |
| half-layer | **sublayer** | Hwang et al. |
| brick-wall | keep **brick-wall** (Hwang et al.), consistently spelled | |
| the mixing spreads layer by layer | "the **causal light cone** grows with depth" | Cerezo et al. 2021 |
| convolution with unshared angles | "a **hardware-efficient brick-wall ansatz** with local connectivity; the same gate pattern on every neighbouring pair, but the angles are not shared, unlike a CNN kernel or the translation-invariant QCNN layer" (say once on 45) | Kandala et al. 2017 |
| measurement operator (for $E_c$) | **POVM element** $E_c$ | Nielsen and Chuang §2.2.6 |
| record of bits | **measurement record** | Wiseman and Milburn |
| one outcome kept / outcomes averaged | **selective** / **non-selective** measurement | Breuer and Petruccione §2.4 |
| the whole circuit read backwards | **Heisenberg picture**: $E_c=\mathcal E^\dagger(\Pi_c)$ | Wilde |
| single-copy circuit / needs many copies | "single-copy measurement statistics are linear in $\rho$; a threshold of an estimated probability needs **many copies** (multi-copy access)" | Huang et al. 2021 |

## Data encoding

| in-house term | use instead | anchor |
|---|---|---|
| Fourier embedding / Fourier encoding | **Fourier (DFT) encoding**: "the window's DFT coefficients loaded by binary-tree state preparation: $\lvert X_m\rvert$ sets rotation angles, $\arg X_m$ sets basis-state phases". Never "amplitude encoding" for it. "Embedding" alone (data embedding, quantum embedding) is standard and may stay | Schuld and Petruccione 2021; Lloyd et al. 2020 |
| revised / original (encoding, placement, $\rho$), paired vs sequential | "**phase on $\lvert b(m)\rangle$**" (the basis state whose amplitude carries $\lvert X_m\rvert$; our choice) vs "**phase on $\lvert m{+}1\rangle$**" (the earlier choice). Slide 44 title: "Why the choice of phase-carrying basis state matters". In tables: "phase on $\lvert b(m)\rangle$" / "phase on $\lvert m{+}1\rangle$" | describe (no literature name) |
| probability tree, tree, nodes, hollow nodes, leaves, tree factors | **binary-tree state preparation** (uniformly controlled $R_y$ rotations); nodes → "rotation angles"; hollow nodes → "trainable constant angles"; leaves → "computational-basis states $\lvert b\rangle$" | Grover and Rudolph 2002; Möttönen et al., QIC 5, 467 (2005) |
| one qubit per Fourier mode | describe exactly as the source (§2.4) does ("one qubit per Fourier mode, modes in uniform superposition"); call it "angle encoding" only if the source confirms it | source |
| slicing, slice the signal, "sees 14 features" | **feature partition** (vertically partitioned data): "each QPU encodes a block of 14 of the 40 features" | Yang et al., ACM TIST 10, 12 (2019) |
| window(s) for the per-QPU feature subset, contiguous/permuted windows, data blocks | **feature block** $x_b$; "contiguous blocks (cyclic; neighbouring blocks share features 0 and 27)" and "blocks of permuted features". "Window" stays only in signal-processing contexts (DFT window, kernel window) | |
| shuffle, shuffled features, under shuffle | **fixed random feature permutation**; condition label "**permuted features**"; test name "**feature-permutation test**" (Greydanus and Kobak call it "shuffled" MNIST-1D: say so once) | Greydanus and Kobak 2024; permuted MNIST |
| price of locality, locality premium | "accuracy drop under feature permutation (the value of the **locality prior**)" | Goodfellow et al. §9.4 |
| exactly unchanged (logistic, kNN, SVM) | "their accuracy is unchanged: these learners depend only on inner products and distances, which a fixed coordinate permutation preserves" | |
| equivariance (unqualified) | **translation equivariance** | Goodfellow et al. §9.2 |
| artificial distributed task / truly distributed task | "a feature partition of one data vector" vs "an **inherently distributed task**: inputs held by different parties, label depending on them jointly" | |

## Learning, evaluation, results

| in-house term | use instead | anchor |
|---|---|---|
| our protocol / the authors' protocol / the recipe | **training setup** (optimizer, learning rate, epochs, model selection) | Goodfellow et al. ch. 7, §11.4 |
| product readout | **product-of-experts (PoE) readout** | Hinton 2002 |
| flat $P_b$, flat QPU, overrules, leans | "a **uniform** $P_b$ changes nothing; a confident expert can **veto** a class (Hinton's word)" | Hinton 2002 |
| the gap, train–test gap, "gap" row | **generalization gap** (train − test accuracy) | Goodfellow et al. §5.2 |
| a new copy of every signal each epoch, augmented copies | **online data augmentation** (a freshly augmented sample every epoch); avoid "copies" (collides with multi-copy access) | Shorten and Khoshgoftaar 2019 |
| points (accuracy difference) | **percentage points** (pp) at first use; "points" after | |
| almost linear (data), reads, readable, linearly readable | "**a linear classifier on the encoded states already reaches 0.943**" (linear separability is yes/no, so not "nearly linearly separable"); "accessible to a single-copy measurement $\operatorname{Tr}[E\rho]$" | Bishop §4.1 |
| linear rule, leading discriminants | **linear classifier**, "the two leading Fisher discriminant directions (LDA)", "decision regions" | Bishop §4.1.4 |
| the $0.978$ classical model on all $40$ features (short labels "best classical", "best joint model tried" may stay) | when the model is named: **kernel logistic regression with a shift-aware (translation-aware) kernel**, i.e. a sum of RBF kernels over the $40$ cyclic width-$5$ sub-windows of the signal ($C=300$, $g=2$), chosen by validation cross-entropy among 2330 candidate classifiers on all $40$ features. Short label: "best classical". Next to it: "tuned RBF SVM" ($0.964$). The results page says "translation-aware"; "shift-aware" is the plain-words form (MNIST-1D digits sit at random cyclic shifts) | `dqml-physics-results.md` §0.1 (model named 2026-09-29, WSL) |
| local reproduction, (local) vs (wiki), grey: wiki | "**this reproduction**" vs "**reported**" | |
| clean reference, matched reference | **matched baseline** | |
| held-out seeds | "seeds 3–7, **not used for model selection**" (keep "pre-registered": the criteria were fixed in advance, App. G) | |
| allowed gap, 1850 needed, target | state the **pre-registered acceptance criterion** explicitly ("mean test accuracy > 0.9 and generalization gap ≤ 0.03") | |
| conditional / not conditional / input-dependent vs constant links | "**constant**", "**dictator** (or its negation): depends on one input only", "depends on both inputs"; "non-constant decision functions" | O'Donnell 2014 |
| implication type ... and its mirrors | "implication and its variants ($\lnot m_i\lor m_j$, $m_i\lor\lnot m_j$, $m_i\land\lnot m_j$, $\lnot m_i\land m_j$)" | |
| corners, at the corners | keep "**corners of the unit square**, $\{0,1\}^2$" (plain geometry), the same on 13, 14, 48, 58 | |
| scanning $d$, $d$ scan, fixed-coefficient scan | "**sweep of the bias $d$** with $(a,b,c)$ fixed"; title "Fixing $(a,b,c)$ and sweeping the bias $d$" | |
| active interval | "**input-dependent range of $d$**": $-1<d<0$ for both coefficient sets (the bit can depend on the input only there) | |
| trainable range, range where training works | "**the range of $d$ in which training succeeds**" (narrower than $(-1,0)$ and $K$-dependent; not the same as the input-dependent range) | |
| balanced value of $d$ | "**balanced bias**: $P(s=1)=1/2$ at $m_i=m_j=1/2$" ($d=-1/2$ XOR-type, $-3/4$ OR-type) | |
| cloud, the boundary misses / cuts the data, threshold through the data | "**distribution** of $(\hat m_i,\hat m_j)$"; "a decision boundary **outside** the support of the data: the unit saturates and receives no gradient" vs "**intersects** the data distribution" | Goodfellow et al. §6.3 |
| the thresholds fit the training signals | "the decision functions **overfit** the 1325 training signals" | |
| resetting thresholds of constant links to the data median | "re-initializing the bias $d$ of constant decision functions so that the decision boundary passes through the data median" | |
| XOR vs parity | **XOR** everywhere; for figures labelled "parity" add "(parity = XOR)" in the caption | |
| $D_\mathrm{add}$, per-window evidence added up, best additive model | "**best additive (per-block) model**" with its test accuracy ($0.971$ contiguous, $0.891$ permuted, §0.1). **$D_\mathrm{add}$ is obsolete (2026-09-29): not in visible text**, since it is a cross-entropy difference and the deck quotes no cross-entropy as a metric; source comments may keep it. Former rule: non-additivity $D_\mathrm{add}$, the excess cross-entropy of the best additive model over the best joint model | Hastie and Tibshirani 1990; Williams and Beer 2010 |
| $R'$, recovered fraction | keep, defined as $R'=(A_\text{CC}-A_\text{NC})/(A_\text{QC}-A_\text{NC})$ | |
| the $10^3$-shot bit | "a message bit from a threshold of a probability estimated from $10^3$ shots (many copies, not one transmitted qubit)" | |
| ladder of classical messages | "classical messages of increasing precision sent in place of the qubit: one bit, $\langle Z\rangle$, the Bloch vector, the full reduced state" | |
| calibration limit of Born-rule outputs | "**bounded confidence** of Born-rule class probabilities (poor calibration)" | Guo et al. 2017 |
| odd cycle of three QPUs | "no measurable **frustration** on the three-QPU cycle ($A(d)=A(-1-d)$)" | Toulouse 1977 |
| knobs | **trainable parameters** | |
| weighted vote, sets / lowers the bar | **weighted sum** $w\cdot x+b$; "the bias shifts the decision boundary" | |
| far links (FC slide) | **long-range weights** (outside the kernel width) | |
| active ingredient | "crucial ingredient" | |
| HMM (as abbreviation) | "hidden manifold model" (HMM usually means hidden Markov model) | Goldt et al., PRX 10, 041044 (2020) |
| Phases 2 to 3B, `~/DQML` paths on visible slides | describe what was done and when ("the experiments of 2026-09-25 to 09-28"); paths only in comments | |
| mutual information, $I(S;Y)$, $D_\mathrm{add}$, "CE", cross-entropy or nats as a figure of merit | **not in visible text** (speaker, 2026-09-29): quote accuracies ("4 digits"). Cross-entropy stays only as the **training loss** (49a); write it out, never "CE". Source comments may keep these values as provenance | speaker's standing style |

## Keep (standard; do not "correct")

LOCC, LO, POVM (element), Born rule, shots, shot noise, mid-circuit measurement, reset, qubit reuse,
classical feed-forward (single-shot), classically controlled gate, monolithic / distributed QPU, QPU,
brick-wall, PQC, light cone, QCNN convolution and pooling layers, Ising energy, IsingZZ, data
embedding, binary-tree state preparation, Boolean function, XOR / OR / NOR / implication / dictator,
probit, effective temperature, saddle, entropy, nats (for the training loss only), synergy (concept), naive
Bayes, kernel logistic regression, RBF SVM, product of experts, cross-entropy, logistic regression, LDA, generalization gap, early
stopping, data augmentation, translation equivariance, locality prior, lottery ticket hypothesis,
double descent, t-SNE, MNIST-1D, hidden manifold model, selective / non-selective measurement,
Heisenberg picture, measurement record, single- / multi-copy access, entanglement entropy,
dephasing, measure-and-prepare channel, quantum fingerprinting, SWAP test, simultaneous message
passing, referee, communication round / cost, broadcast, ring, percentage points, "pre-registered",
"Colab", "CY", project names (Post-Selection, DDrf), slide-title style ("Where the months went",
"Doubt 1"), kit category pills. ("Mutual information" left this list on 2026-09-29: it is standard, but the speaker wants it out of the deck.)

## Figures that cannot be regenerated (from `~/DQML/analysis`, not on this machine)

Crop off in-figure titles that carry project labels where possible; otherwise translate the labels in
the caption ("two-source ring" = "2 senders → 1 receiver", "parity" = XOR, "active interval" =
input-dependent range of $d$, "bit rate" = $P(s=1)$, "protocol range" = pre-registered shot range).
