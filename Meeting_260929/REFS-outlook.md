# References for the outlook section (slides 60-65)

Updated 2026-09-29 (round 3) to match `64-fingerprint.md`, `64a-fingerprint-protocol.md` (table of joint
functions since round 2) and `80-references.md` page 3, "communication complexity" (items 31-39 there).
Bibliographic details of items 2-9 are as verified by the round-2 functions group against the arXiv
full texts, recorded in the src comments of `64a-fingerprint-protocol.md` (read 2026-09-29); none of
items 2-9 is in `1. raw/references.bib` or `1. raw/papers/` (checked 2026-09-29). Candidates for
`aleph-ref add` on the human's request.

## Cited on the slides

1. H. Buhrman, R. Cleve, J. Watrous and R. de Wolf, "Quantum Fingerprinting," Phys. Rev. Lett. 87, 167902 (2001). DOI: 10.1103/PhysRevLett.87.167902. Citekey `buhrman2001quantum` (filed in `1. raw/papers/buhrman2001quantum/`; `references.bib` not rebuilt since). Slide 64 (source line; the `fingerprint` clip: eqs. 3-5, Fig. 1, Theorems 1-2) and slide 64a (equality row). References slide item 31.
2. H. Buhrman, R. Cleve and A. Wigderson, "Quantum vs. classical communication and computation," STOC 1998. arXiv:quant-ph/9802040. Slide 64a source line (disjointness: first quantum protocol, Theorem 1.6; also the distributed Deutsch-Jozsa separation, Theorem 1.7, in the speaker note only). Item 32.
3. R. Cleve, W. van Dam, M. Nielsen and A. Tapp, "Quantum entanglement and the communication complexity of the inner product function," QCQC 1998, Lecture Notes in Computer Science 1509, 61-74 (1998). arXiv:quant-ph/9708019. Slide 64a (inner product mod 2 row, no advantage). Item 33.
4. A. C.-C. Yao, "On the power of quantum fingerprinting," STOC 2003, pp. 77-81. Slide 64a (Hamming distance row; Lemma 1, the SWAP-test overlap estimate, in the speaker note). Item 34.
5. A. A. Razborov, "Quantum communication complexity of symmetric predicates," Izvestiya: Mathematics 67, 145-159 (2003). arXiv:quant-ph/0204025. Slide 64a source line (the $\Omega(\sqrt n)$ lower bound for disjointness). Item 35.
6. Z. Bar-Yossef, T. S. Jayram and I. Kerenidis, "Exponential separation of quantum and classical one-way communication complexity," STOC 2004; SIAM J. Comput. 38, 366-384 (2008). Slide 64a source line (hidden matching, introduced there). The paper itself was not accessible to the functions group; its content is cross-checked in items 8 and 9. Item 36.
7. S. Aaronson and A. Ambainis, "Quantum search of spatial regions," FOCS 2003; Theory of Computing 1, 47-79 (2005). arXiv:quant-ph/0303041. Slide 64a (disjointness row, $O(\sqrt n)$ qubits). Item 37.
8. D. Gavinsky, J. Kempe, I. Kerenidis, R. Raz and R. de Wolf, "Exponential separations for one-way quantum communication complexity, with applications to cryptography," STOC 2007, pp. 516-525; SIAM J. Comput. 38(5), 1695-1708 (2008), DOI: 10.1137/070706550. arXiv:quant-ph/0611209. Slide 64a (Boolean hidden matching variant row, Theorem 2). Item 38.
9. B. Klartag and O. Regev, "Quantum one-way communication can be exponentially stronger than classical communication," STOC 2011, pp. 31-40. arXiv:1009.3640. Slide 64a (vector in subspace row). Item 39.

## Cited only in source comments or speaker notes (not on the references slide)

10. A. Ambainis, "Communication complexity in a 3-computer model," Algorithmica 16, 298-301 (1996). DOI: 10.1007/BF01955678. Ref. [4] of Buhrman et al. for the $O(\sqrt n)$-bit classical fingerprints without a shared key. 64a src comment (equality row). Not in `references.bib` (verified by web search, 2026-09-28: link.springer.com/article/10.1007/BF01955678). Dropped from the references slide in round 2.
11. I. Newman and M. Szegedy, "Public vs. private coin flips in one round communication games," Proc. 28th ACM STOC (1996), pp. 561-570. DOI: 10.1145/237814.238004. Ref. [7] of Buhrman et al. for the matching $\Omega(\sqrt n)$ lower bound. 64a src comments (equality and Hamming distance rows). Not in `references.bib` (verified by web search, 2026-09-28: dl.acm.org/doi/10.1145/237814.238004). Dropped from the references slide in round 2.
12. E. Kushilevitz and N. Nisan, *Communication Complexity* (Cambridge University Press, 1997). 64a speaker note (Yao's minimax principle, the shared-key caveat).
13. D. Gavinsky, J. Kempe and R. de Wolf, "Strengths and weaknesses of quantum fingerprinting," CCC 2006, pp. 288-295. arXiv:quant-ph/0603173. 64a speaker note and src comment.
14. Classical lower bounds named in the 64a src comments only, as cited there: B. Kalyanasundaram and G. Schnitger, SIAM J. Discrete Math. 5, 545-557 (1992), and A. A. Razborov, Theor. Comput. Sci. 106, 385-390 (1992) (disjointness); B. Chor and O. Goldreich, SIAM J. Comput. 17, 230 (1988) (inner product); I. Kremer, master's thesis, Hebrew University (1995), and R. Raz, STOC 1999 (vector in subspace).
15. J. Frankle and M. Carbin, "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks," ICLR 2019. arXiv:1803.03635. Cited on slide 27 (REFS-dqml.md item 4; references slide item 11); in this section only in the speaker note of the Take-home slide, `65-next.md` (the removed next-steps list).
