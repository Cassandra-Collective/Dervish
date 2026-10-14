# Dervish · design

Dervish is an optimizer for the hidden matrices of a transformer. It is built
on Muon and changes one thing about it: Muon constrains the *update* (every
step is an orthogonalised matrix) and leaves the *weights* free; Dervish
constrains the weights and lets the update be whatever rotation descends
fastest. The name is the Mevlevi dervish: the optimizer whirls instead of
pushing.

The file is organised as a hypothesis document is: the problem,
the mechanism, the claims that follow from it with their proofs, the knobs, the
cost, and the limits. Code references point into `dervish/`.

## 1 · The problem Muon leaves open

Muon (Jordan et al., 2024) replaces each gradient-momentum matrix `M` by its
orthogonalised version `U = polar(M) ~ U_M V_M^T`, computed with a quintic
Newton-Schulz iteration, and steps `W <- W - lr * U`. Every update therefore
has all singular values near one. This is the source of Muon's speed (no
direction is starved) and of its best-known failure: nothing bounds the
singular values of `W` itself. Under a constant learning rate without weight
decay, the usual Muon recipe (for example lr `0.01` for `50,000` steps, no
decay), the spectral norms of the hidden matrices can grow
without limit, and with them the attention logits `q . k / sqrt(d)` and the
output logits. Two published repairs exist: decoupled weight decay on the Muon
group (Moonshot, "Muon is scalable for LLM training", 2025) and MuonClip
(Kimi K2, 2025), which rescales the query and key projections whenever the
largest attention logit crosses a threshold. Both are controllers around a
missing invariant.

## 2 · The parameterisation

Every hidden matrix `W` of shape `(m, n)` is kept, at all times, as

    W = diag(s) Q,      Q in R^{m x n},  Q Q^T = I_m,  s in R^m     (m <= n)

and as `W = Q diag(s)` with orthonormal columns when `m > n`. Internally every
frame is handled wide (`dervish/matricize.py` transposes tall matrices), so
the rest of this document assumes `m <= n`.

Three facts make the form useful.

- The singular values of `W` are exactly `|s_1|, ..., |s_m|`, because
  `W W^T = diag(s) Q Q^T diag(s) = diag(s^2)`. The spectrum is a *variable*,
  not an emergent property.
- `Q` lives on the Stiefel manifold `St(m, n)`, whose tangent space at `Q` is
  `{D : D Q^T + Q D^T = 0}`. Moving `Q` within the manifold cannot change any
  singular value of `W`.
- The optimizer keeps `s >= 0` (a floor of at least `0` is always applied), so
  `s_i` *is* the `i`-th singular value and a cap on `s` is a cap on the
  spectral norm.

At initialisation (`Dervish._init_frame`), `W_0 = diag(||rows of W_0||) polar(W_0)`:
the rows are made orthonormal by the exact SVD polar factor and their norms are
kept as the initial spectrum. For `N(0, 0.02)` initialisation this changes the
function slightly (step-`0` validation loss `3.579` versus `3.625` in the
smoke test); the curves in `docs/RESULTS.md` start at step `0` so the offset
is visible.

## 3 · One step

`Dervish._frame_step`, for one frame with gradient `G = dL/dW`:

1. **Split the gradient.** By the chain rule on `W = diag(s) Q`,

       dL/ds_i = sum_j G_ij Q_ij          (`scale_grad = (G * Q).sum(-1)`)
       dL/dQ   = diag(s) G                (`frame_grad`)

2. **Momentum and tangent projection.** Muon's momentum (`lerp` by `0.95`,
   Nesterov form) is applied to `dL/dQ`. The result is projected onto the
   tangent space at `Q`:

       P_Q(D) = D - sym(D Q^T) Q,    sym(A) = (A + A^T) / 2

   This removes the component of the direction that would only rescale rows,
   which is the spectrum's job. (`dervish/polar.py: tangent_project`.)

3. **Orthogonalise the direction.** The quintic Newton-Schulz iteration of
   Muon (same coefficients `3.4445, -4.7750, 2.0315`, same five steps) turns
   the projected direction into `U` with all singular values in about
   `[0.68, 1.20]` (measured over `120` Gaussian matrices: minimum `0.0058`
   for near-singular square inputs, maximum `1.2023`). `U` is projected to
   the tangent space once more; the projection is cheap and keeps the next
   step a near-rotation.

4. **Rotate the frame (retraction).**

       Q <- polar_cubic(Q - lr * U)

   Because `U` is tangent, `(Q - lr U)(Q - lr U)^T = I + lr^2 U U^T`, so the
   singular values of `Q - lr U` lie in `[1, sqrt(1 + lr^2)]` and the cubic
   Newton-Schulz iteration `X <- 1.5 X - 0.5 (X X^T) X` converges
   quadratically: two steps reach float32 precision (`tests/test_polar.py`).
   The new frame equals the old one multiplied by an orthogonal matrix
   `R in O(n)` on the right (for a square frame, `R = Q_new Q_old^T` is itself
   orthogonal: `tests/test_optimizer.py::test_rotation_interpretation...`).
   The rotation angle is at most about `1.2 * lr` radians per step, the
   quintic's spectral spread times the learning rate, so `lr` is an angle.

5. **Move the spectrum.** Adam (bias-corrected, betas `(0.9, 0.95)`) on
   `dL/ds` with its own learning rate `scale_lr`, optional decoupled decay,
   then a hard clamp to `[scale_floor, scale_cap]`.

6. **Write back** `W = diag(s) Q` into the parameter. The model never sees
   anything but a plain weight tensor, which is why no model code changes.

Everything that is not a frame (embeddings, output head unless opted in,
LayerNorm gains and biases, linear biases) is trained by the same auxiliary
Adam that Muon uses (`dervish/optimizer.py: _adam_group_step`, a line-for-line
mirror of a reference Muon trainer's `adam_update`). A Dervish-versus-Muon
comparison therefore changes exactly one thing.

## 4 · What follows, with proofs

**Proposition 1 (bounded spectral norm).** For every frame and every step,
`||W||_2 = max_i s_i <= scale_cap` when a cap is set. *Proof.* Section 2:
the singular values are `s_i`, and step 5 clamps `s_i <= scale_cap` before the
write-back. The retraction leaves `Q` orthonormal to float32 precision, which
is the only approximation; `tests/test_optimizer.py::test_spectral_cap_bounds_the_spectral_norm_under_pressure`
checks it after `40` deliberately violent steps.

**Proposition 2 (bounded attention logits).** Let `x` be a token's residual
vector and `u = LN(x)` its pre-attention LayerNorm output with gain vector
`gamma` and bias `beta`, so `||u|| <= sqrt(n_embd) * max|gamma| + ||beta|| =: r`.
With per-head frames (`FrameView.heads`), the query of head `h` is
`q_h = W_q^h u + b_q^h` with `||W_q^h||_2 <= cap`, and likewise `k_h`. Then for
every pair of positions,

    |q_h . k_h'| / sqrt(d_head) <= (cap * r + ||b_q^h||) (cap * r + ||b_k^h||) / sqrt(d_head)

RoPE rotates `q` and `k` and changes no norm. `TinyTransformer.attention_logit_bound`
evaluates the right-hand side from the current LayerNorm parameters and biases
and reports the maximum over blocks; the lab logs it beside the measured
maximum logit at every evaluation. *Proof.* Cauchy-Schwarz and the operator
norm bound `||W u|| <= ||W||_2 ||u||`.

The bound is a theorem about the implementation, not an empirical claim; the
empirical content of claim 1 in `docs/HYPOTHESIS-D001.md` is whether the
*control* grows, which no theorem decides.

**Proposition 3 (bounded output logits, optional).** With `--dervish-head`,
`lm_head.weight` (`vocab x n_embd`, wide) is a frame too, so every logit
satisfies `|z_v| <= cap * ||LN_f(x)|| + |b_v|`. The default leaves the head on
Adam, as Muon does, so that the embedding and head treatment matches the
usual Muon recipe.

**Proposition 4 (no rank collapse).** If `scale_floor = f > 0` and
`scale_cap = c`, the stable rank `||W||_F^2 / ||W||_2^2 >= m (f / c)^2` for every
frame. *Proof.* `||W||_F^2 = sum s_i^2 >= m f^2` and `||W||_2^2 <= c^2`.
Without a floor the inequality is vacuous; the matrices in E1 run with
`floor = 0` and their stable rank is *measured*, not guaranteed, which is why
the rank question is exploratory in the hypothesis.

## 5 · Architecture flexibility

Muon acts on 2D tensors and treats a 4D convolution kernel as
`(out, rest)`. Dervish makes the matricisation an explicit, declared rule
(`dervish/matricize.py: FrameView`):

| Parameter shape | Default rule | Frames |
| --- | --- | --- |
| `(m, n)` | one matrix | `1` |
| `(E, m, n)` | a batch, one frame per expert (mixture-of-experts banks) | `E` |
| `(out, in, kh, kw)` | `(out, in*kh*kw)` | `1` |
| `(k*r, n)` with `FrameView.heads(k, r, n)` | `k` frames of `(r, n)` (a fused projection, one frame per head and role) | `k` |

All frames in a batch are processed by one batched Newton-Schulz call. The 3D
rule is a choice, not a fact about tensors: a `Conv1d` kernel `(out, in, k)` is
also 3D, and for it the right view is one frame `FrameView((out, in*k))`,
passed explicitly (`examples/experts_demo.py` does so). The lab
views every `c_attn.weight` (`3*n_embd, n_embd`) as `3*n_head` frames of
`(d_head, n_embd)`, which is what makes Proposition 2 per-head. Anything that
is not a frame (1D parameters) goes to Adam; a parameter whose shape has no
automatic rule must be given an explicit `FrameView` or the optimizer refuses
it at construction (`tests/test_matricize.py`).

## 6 · Knobs and defaults

| Knob | Default | Meaning |
| --- | --- | --- |
| `lr` | `0.05` | rotation angle per step, radians (Muon's `0.02` moves `W` by spectral norm `0.02`; Dervish moves `W` by about `s * 1.2 * lr`, comparable at `s ~ 0.3`) |
| `momentum`, `nesterov`, `ns_steps` | `0.95`, `True`, `5` | Muon's, unchanged |
| `ns_dtype` | `float32` | `bfloat16` on CUDA reproduces the usual GPU path; the retraction always runs in float32 |
| `retraction_steps` | `2` | cubic Newton-Schulz steps; `tests/test_polar.py` shows `2` is enough for `lr <= 0.2` |
| `tangent` | `True` | the two projections of section 3 |
| `scale_lr` | `1e-3` | Adam rate for the spectrum; the lab uses the same rate as for LayerNorm gains |
| `scale_cap` | `None` (lab: `1.0`) | upper bound on every singular value |
| `scale_floor` | `0.0` | lower bound; `> 0` buys Proposition 4 |
| `scale_decay` | `0.0` | decoupled decay of the spectrum towards `0` |
| `init_polar` | `svd` | exact polar factor at initialisation; `iterative` for devices without a usable SVD |

Why `1.0` for the cap in the lab: with LayerNorm in front of every sublayer,
`r ~ sqrt(n_embd) = 11.3` at unit gain, so Proposition 2 gives attention
logits at most about `n_embd / sqrt(d_head) = 22.6` for `n_embd 128`,
`d_head 32`, which still allows a near-one-hot attention pattern (the copy
behaviour needs sharp attention) while ruling out explosion.

## 7 · Cost

Per frame and step, relative to Muon: the same quintic iteration on the
direction, plus two tangent projections (one `m x m` matmul and one
`m x m x n` product each) and two cubic steps (one `m x n x m` and one
`m x m x n` product each). For `m <= n` this is roughly `1.6x` Muon's
optimizer arithmetic; on the lab model the whole step (forward, backward,
optimizer) measured `0.21 s` versus `0.17 s` for Muon on four CPU threads
(smoke test, `40` steps). Memory: one extra matrix (`Q`) and three vectors per
frame versus Muon's single momentum buffer; Adam keeps two matrices.

## 8 · Limits and failure modes

- The initial projection changes the initial function. At `N(0, 0.02)`
  initialisation the rows of a `128 x 512` matrix are already nearly
  orthogonal, so the change is small, but it is not zero.
- A cap that binds caps capacity. If a task needs a singular value above the
  cap, the spectrum saturates and the loss pays for it; the `dervish_nocap`
  arm in E1 exists to show the price of the cap separately from the price of
  the manifold.
- The spectrum learning rate is a second knob. Too slow and the spectrum
  cannot follow the frame; too fast and `s` oscillates against the clamp.
- Biases are not frames. Proposition 2 carries them as additive terms; they
  are trained by Adam and could in principle grow, which is why the bound is
  evaluated from the live parameters rather than assumed constant.
- LayerNorm gains are not capped either, and Proposition 2 scales with
  `max|gamma|`. The bound is therefore a bound relative to parameters that
  Adam can grow, not a constant: on the fresh seeds of Stage D8b it rose from
  `22.6` at step `0` to `32.7` (linear decay) and `38.4` (constant rate) on
  average at step `1000`, while the measured logits peaked near `16`.
  A constant bound needs the gains capped too (or removed, as in
  Lipschitz-constrained transformers).
- Nothing here is tested beyond `0.8M` parameters and `1000` steps on one CPU
  (see `docs/RESULTS.md` for exactly what was run), and the bfloat16
  Newton-Schulz path is untested on hardware.

## 9 · Pseudocode

    for each frame parameter W (wide view, state Q, s, M):
        G      = grad(W)
        g_s    = rowsum(G * Q)
        G_Q    = diag(s) G
        M      = lerp(M, G_Q, 1 - beta);   D = lerp(G_Q, M, beta)      # Nesterov
        D      = D - sym(D Q^T) Q                                        # tangent
        U      = NewtonSchulz5(D);   U = U - sym(U Q^T) Q                 # orthogonalise, re-tangent
        Q      = NewtonSchulz3(Q - lr * U, steps=2)                      # rotate
        s      = clamp(Adam(s, g_s, scale_lr), floor, cap)              # spectrum
        W      = diag(s) Q
    for every other parameter: Adam (Muon's auxiliary Adam, unchanged)
