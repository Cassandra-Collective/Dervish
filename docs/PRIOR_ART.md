# Prior art · what Dervish borrows and what, if anything, is new

The point of this note is to avoid rediscovering a known method under a new
name. The first version (2026-10-09) was written without network access and
marked every citation `[unverified]`. On 2026-10-14 each citation was checked
against its primary source, and a search for closer prior work was run. That
search found work that pre-empts most of what the first version called new.
The section "What is new" below is the corrected statement.

## Verified citations

| Ingredient in Dervish | Prior work (verified 2026-10-14) | What it already did |
| --- | --- | --- |
| Orthogonalised momentum update via quintic Newton-Schulz | K. Jordan, Y. Jin, V. Boza, J. You, F. Cesista, L. Newhouse, J. Bernstein, "Muon: An optimizer for hidden layers in neural networks", blog post, 8 December 2024, <https://kellerjordan.github.io/posts/muon/>; code at `KellerJordan/Muon` (MIT) | The update direction. Dervish uses it unchanged for the frame direction, coefficients `3.4445, -4.7750, 2.0315`, five steps. |
| Steepest descent under a spectral norm, duality view | J. Bernstein and L. Newhouse, "Old Optimizer, New Norm: An Anthology", arXiv:2409.20325 (2024); "Modular Duality in Deep Learning", arXiv:2410.21265 (2024) | The theory that explains why Muon's update is the spectral-norm steepest descent. Dervish uses it on the tangent space only informally; no claim is made that the tangent-projected quintic output is the Riemannian steepest descent. |
| Weight decay on the Muon group, update RMS matching | J. Liu, J. Su et al., "Muon is Scalable for LLM Training", arXiv:2502.16982 (2025) | The published repair for weight-norm growth; the `muon_wd` baseline of Stage D1. |
| QK-Clip / MuonClip | Kimi Team, "Kimi K2: Open Agentic Intelligence", arXiv:2507.20534 (2025) | Rescales each head's query and key projections, `gamma_h = min(1, tau / S_max^h)`, when the *measured* largest logit of that head exceeds `tau`. A data-driven controller; Dervish's Proposition 2 is a parameter-driven bound. |
| Enforced Lipschitz constants with Muon | L. Newhouse, R. P. Hess, F. Cesista, A. Zahorodnii, J. Bernstein, P. Isola, "Training Transformers with Enforced Lipschitz Constants", arXiv:2507.13338 (2025) | Caps spectral norms after each Muon step ("spectral soft cap", "spectral hard cap", spectral normalisation) and proves that Muon plus the soft cap bounds the spectral norm. Same goal, same optimizer family; they project `W` after the step, Dervish never leaves the factored form. |
| Weight matrices kept unitary during training | M. Arjovsky, A. Shah, Y. Bengio, "Unitary Evolution Recurrent Neural Networks", ICML 2016 (PMLR 48:1120-1128), arXiv:1511.06464; S. Wisdom, T. Powers, J. Hershey, J. Le Roux, L. Atlas, "Full-Capacity Unitary Recurrent Neural Networks", NIPS 2016, arXiv:1611.00035 | Why the quantum analogy exists at all: a unitary matrix cannot explode or vanish along any direction. |
| Riemannian optimisation on the Stiefel manifold | J. Li, L. Fuxin, S. Todorovic, "Efficient Riemannian Optimization on the Stiefel Manifold via the Cayley Transform", ICLR 2020, arXiv:2002.01113; P.-A. Absil, R. Mahony, R. Sepulchre, *Optimization Algorithms on Matrix Manifolds*, Princeton University Press, 2008 | Tangent projection and retraction are standard; the polar retraction is classical. |
| Orthogonal reparameterisation | L. Huang, L. Liu, F. Zhu, D. Wan, Z. Yuan, B. Li, L. Shao, "Controllable Orthogonalization in Training DNNs", CVPR 2020, arXiv:2004.00917; M. Cisse, P. Bojanowski, E. Grave, Y. Dauphin, N. Usunier, "Parseval Networks: Improving Robustness to Adversarial Examples", ICML 2017 (PMLR 70:854-863) | Keep `W` (near) orthogonal by reparameterisation or a regulariser; no separately learned spectrum. |
| Polar decomposition and the nearest unitary | N. J. Higham, *Functions of Matrices: Theory and Computation*, SIAM, 2008, doi:10.1137/1.9780898717778 | Used in `examples/nearest_unitary.py` and `docs/QUANTUM_CONNECTION.md`. Nothing new. |
| Fast polynomial polar iterations | N. Amsel, D. Persson, C. Musco, R. M. Gower, "The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm", arXiv:2505.16932, ICLR 2026 | A faster or more accurate replacement for the quintic; a drop-in improvement Dervish does not use, so that the comparison with Muon changes one thing. |
| Acceleration in front of Muon's momentum | J. Li, J. Tan, H. Xu, J. Zhang, Y. Lu, Y. Sun, Y. Xie, X. Cai, "MONA: Muon Optimizer with Nesterov Acceleration for Scalable Language Model Training", arXiv:2605.26842 v2 (2026), Findings of EMNLP 2026 | The baseline of Hypothesis D002. With the paper's tie `alpha = -1/(2(1-beta_a))` its term equals the mean of the gradient and its slow EMA (`dervish/optimizer.py: mona_accelerate`), a two-timescale momentum in the family of quasi-hyperbolic momentum (Ma and Yarats, ICLR 2019). |

## Closer prior work found on 2026-10-14

Ranked by how much of Dervish each one already contains. Every item predates
2026-10-09.

1. **Muown.** K. Lion, F. Hübler, B. Li, A. Orvieto, N. He, "Muown: Row-Norm
   Control for Muon Optimization", arXiv:2605.10797 (May 2026). It writes each
   matrix as row magnitudes times row directions, `W = Diag(g) D` with unit-norm
   rows. The magnitude gradient is `rowsum(grad W * D)`, which is Dervish's
   spectrum gradient exactly, and Adam trains the magnitudes in the practical
   algorithm. The direction gets a radially projected gradient and Nesterov
   Muon. What Dervish adds: orthonormal rows rather than unit rows (so `s` is
   the spectrum, not row norms), a hard clamp that makes `||W||_2 <= cap` a
   theorem, an explicit retraction, and the factored state kept by the
   optimizer. Muown has no bound and no clamp.
2. **Manifold Muon.** J. Bernstein, "Modular Manifolds", Thinking Machines Lab
   blog, 26 September 2025. It keeps weights on the Stiefel manifold and
   steps by spectral-norm steepest descent in the tangent space with an
   msign retraction. Singular values are fixed at one; no spectrum is learned.
   A follow-up with a diagonal-Gram constraint (Tilde Research, "Gram-space
   manifold Muon", October 2025) amounts to an orthogonal frame times free
   scales, which is Dervish's geometry with the scales moved inside the Muon
   step. That follow-up could only be read through secondary sources.
3. **Spectral Sphere Optimizer (SSO).** T. Xie et al., "Controlled LLM
   Training on Spectral Sphere", arXiv:2601.08393 (January 2026). It
   constrains both weights and updates under module-wise spectral
   constraints, is implemented in Megatron, and reports beating AdamW and
   Muon at 1.7B dense and 8B mixture-of-experts scale. It has Dervish's goal,
   with evidence at a scale Dervish has never reached.
4. **SVD parameterisation with an interval-bounded spectrum.** E. Vorontsov,
   C. Trabelsi, S. Kadoury, C. Pal, "On Orthogonality and Learning Recurrent
   Networks with Long Term Dependencies", ICML 2017, arXiv:1702.00071. It
   writes `W = U S V^T`, updates `U` and `V` with geodesic steps on the Stiefel
   manifold, and trains a spectrum held to `[1 - m, 1 + m]`. This is the direct
   ancestor of "frame on the Stiefel manifold plus a spectrum in a hard
   interval". Dervish's difference is the Muon-style frame update and the
   one-sided `diag(s) Q` form.
5. **Logit bounds from spectral norms.** K. Yoshihara, "Different Layers,
   Different Manifolds: Module-Wise Weight-Space Geometry in Transformer
   Optimization", arXiv:2606.13276 (June 2026), traces the instability of
   diagonal-Gram manifold Muon in attention to singular-value growth that
   amplifies attention logits. S. Zhai et al., "Stabilizing Transformer Training
   by Preventing Attention Entropy Collapse" (sigma-Reparam), ICML 2023,
   arXiv:2303.06296, bounds attention entropy through the spectral norm of the
   logit map. Neither enforces and reports a live per-head bound as Dervish
   does.

## What is new, corrected

None of the four ingredients the first version listed is new on its own:

- The spectrum/frame split with Adam on the magnitudes is Muown's.
- A Stiefel frame with a spectrum held to an interval is Vorontsov et al.'s.
- Weights held to a spectral bound with Muon-family updates are in
  Newhouse et al. and SSO.
- Matricising 4D kernels is in Muon's own code (`update.view(len(update), -1)`).
  Per-expert views of 3D banks are a convention, so the old claim 4 is dropped.

What remains is a specific composition, which the evidence in
`docs/RESULTS.md` has to justify on its own:

1. A frame with orthonormal rows, moved by a tangent-projected Muon direction
   and a two-step cubic Newton-Schulz retraction. The retraction is exact to
   float32 at Muon step sizes because the tangent step keeps the singular
   values in `[1, sqrt(1 + lr^2)]`. A blog post by F. Cesista ("Heuristic
   Solutions for Steepest Descent on the Stiefel Manifold", July 2025) uses
   the same project-then-orthogonalise loop. The search found no statement of
   the two-step exactness, which is not the same as showing it is new.
2. An Adam-trained spectrum under a hard clamp. This makes `||W||_2 <= cap` hold
   at every step, and with per-head frames gives an attention-logit bound
   computed from live parameters at every evaluation.

Whether this composition is worth anything beyond its parts is an empirical
question. Hypothesis D002 answers it at this repository's scale.
