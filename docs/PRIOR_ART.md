# Prior art · what Dervish borrows and what, if anything, is new

The point of this note is to avoid
rediscovering a known method under a new name. This container has no access
to arXiv, OpenReview or Semantic Scholar (every probe returned no route), so
the citations below are from memory and are marked `[unverified]` where a
year, venue or exact title could not be checked. Each should be verified
against the primary source before any of this is cited in a paper.

## The ingredients, and who had them first

| Ingredient in Dervish | Prior work | What it already did |
| --- | --- | --- |
| Orthogonalised momentum update via quintic Newton-Schulz | Muon, Keller Jordan et al., 2024 (blog post and `KellerJordan/Muon`); the coefficients and the `(bf16, 5 steps)` recipe are copied verbatim from the reference trainer `[unverified: exact post date]` | The update direction. Dervish uses it unchanged for the frame direction. |
| Steepest descent under a spectral norm, duality view | Bernstein and Newhouse, "Old optimizer, new norm" and "Modular duality in deep learning", 2024 `[unverified]` | The theory that explains why Muon's update is the spectral-norm steepest descent. Dervish inherits it on the tangent space only informally; no claim is made that the tangent-projected quintic output is the Riemannian steepest descent. |
| Weight decay on the Muon group, update RMS matching | Moonshot AI, "Muon is scalable for LLM training", 2025 `[unverified]` | The published repair for weight-norm growth. It is the `muon_wd` baseline in E1. |
| QK-Clip | Kimi K2 technical report, Moonshot AI, 2025 `[unverified]` | Rescales query and key projections when the largest attention logit exceeds a threshold. A controller; Dervish's Proposition 2 makes the threshold a consequence of the cap. |
| Enforced Lipschitz constraints with Muon | Newhouse, Bernstein et al., "Training transformers with enforced Lipschitz constants", 2025 `[unverified]` | Caps spectral norms by spectral normalisation or a "spectral soft cap" after each Muon step and trains transformers without activation norms. The closest relative: same goal, same optimizer family. The difference is representational: they project `W` back under a norm after the step; Dervish never leaves the factored form, learns the spectrum as a separate variable, and rotates the frame instead of adding to it. |
| Weight matrices kept unitary or orthogonal during training | Unitary evolution RNNs, Arjovsky, Shah and Bengio, ICML 2016; Wisdom et al., "Full-capacity unitary RNNs", NeurIPS 2016 `[unverified: venues]` | The reason this is called quantum-inspired at all: unitary matrices cannot explode or vanish gradients. Dervish applies the idea to transformer hidden matrices, with an explicit spectrum so the matrices are *scaled* isometries, not isometries. |
| Riemannian optimisation on the Stiefel manifold with cheap retractions | Cayley SGD/Adam, Li, Li and Todorovic, ICLR 2020; Absil, Mahony and Sepulchre, "Optimization algorithms on matrix manifolds", 2008 `[unverified]` | Tangent projection and retraction are standard. Dervish's retraction (two cubic Newton-Schulz steps) is a polar retraction; the polar retraction is classical. What is particular here is pairing it with Muon's orthogonalised direction, so that the retraction input has singular values in `[1, sqrt(1 + lr^2)]` and two steps suffice. |
| Orthogonal reparameterisation by Newton iteration on the weight | ONI, Huang et al., "Controllable orthogonalization in training DNNs", CVPR 2020 `[unverified]`; Parseval networks, Cisse et al., ICML 2017 `[unverified]` | Keep `W` (near) orthogonal by reparameterising `W = NS(V)` or by a regulariser. Closest to Dervish's invariant; different mechanism (reparameterise through autograd versus move on the manifold with a dedicated optimizer) and no separate learned spectrum. |
| Polar decomposition and the nearest unitary | Standard numerical linear algebra (Higham, "Functions of matrices", 2008 `[unverified]`); Newton-Schulz iteration for the polar factor is classical | Used in `examples/nearest_unitary.py` and in `docs/QUANTUM_CONNECTION.md`. Nothing new. |
| Fast polynomial polar iterations | "Polar Express" (Amsel et al., 2025) `[unverified]` and related work on optimised Newton-Schulz coefficients | Faster or more accurate replacements for the quintic. Dervish keeps Muon's coefficients so the comparison with the standard Muon recipe changes one thing; a faster polynomial is a drop-in improvement later. |

## What is, as far as this note can tell, new in the composition

1. The explicit factorisation `W = diag(s) Q` as the *optimizer's* state, with
   the spectrum trained by Adam under a hard interval and the frame trained by
   Muon-style rotations. Lipschitz-constrained Muon caps the spectrum; it does
   not learn it as a variable. Orthogonal reparameterisations fix the spectrum
   at one.
2. Per-head frames for the fused attention projection, so that the attention
   logit bound (Proposition 2) is per head and computable from live
   parameters, and reported next to the measured maximum at every evaluation.
3. The use of the tangent structure to make the polar retraction two-step
   exact at Muon's step sizes (section 3 of `docs/DESIGN.md`).
4. The declared matricisation rule (`FrameView`) that makes mixture-of-experts
   banks and convolutions first-class frames in the same optimizer, which is
   what a mixture-of-experts model needs.

None of these is deep. Each is a small, falsifiable engineering claim, which is
the kind the lab keeps.

## Risk of novelty theatre

Two honest risks. First, something very close to item 1 may exist under
another name (a 2025 "spectral Muon" or "Stiefel Muon" preprint would not be
surprising); a literature search should look for exactly that before
this leaves the repository. Second, the quantum framing is an analogy
(`docs/QUANTUM_CONNECTION.md` section 4), and it should never be described as
more than that.
