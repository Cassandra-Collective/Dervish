# Hypothesis D001 · A rotation-based successor to Muon keeps hidden spectra bounded without losing next-character loss

- Status: RESOLVED on 2026-10-09 by `python lab/check_claims.py` over
  `runs/`: claim 1 PASS, claim 2 KILL, claim 3 KILL (`docs/RESULTS.md`
  Stages D1 and D2). The KILLs trace to two knobs fixed wrongly before the
  run (spectrum rate about `10x` too slow, rotation angle about `4x` too
  large); the post-hoc Stages D4 to D7 measure the corrected setting and
  are the basis for the next registration, not a verdict. The text below
  is as written before any matrix ran; the thresholds were fixed from a
  40-step smoke test only. Nothing else in this file has been edited.
- Date: 2026-10-09
- Edited 2026-10-14, for publication only: references to a private project
  (its name, stage numbers and file paths) and a personal dedication were
  replaced by neutral wording when the repository history was rewritten. No
  hypothesis, arm, seed, threshold or pass line was changed.
- Author: Mert Efe Şensoy
- Builds on: the common Muon recipe (Muon on hidden matrices, Adam on
  embeddings and head, constant learning rate, no weight decay), a copy-task
  probe for long-context behaviour, and a design brief: an optimizer that
  fixes logit explosions, resists the simplicity bias, and is flexible about
  architecture.

## Context

A common Muon recipe trains the hidden matrices with Muon and the rest with
Adam, at a constant learning rate and without weight decay. Muon orthogonalises each update, so every step
adds a matrix with all singular values near one. Nothing in that recipe bounds
the singular values of the weights themselves. In the literature the known
consequences are growing attention logits (the Kimi K2 report's reason for
MuonClip) and a reliance on weight decay (the Moonshot "Muon is scalable"
recipe); both fixes are reactive patches around the same missing invariant.

Dervish (`dervish/optimizer.py`) replaces the patch with a parameterisation:
every hidden matrix is kept as `W = diag(s) Q` with `Q` orthonormal, the frame
`Q` moves only by rotations found with the same Newton-Schulz machinery Muon
already uses, and the spectrum `s` is a small vector trained by Adam under a
hard cap. The singular values of `W` are `|s|` exactly, at every step, so the
spectral norm of every hidden matrix and therefore every attention logit has a
bound that can be computed from the model and checked against measurement.
The mechanism is described in `docs/DESIGN.md`; what it borrows from unitary
RNNs, Stiefel optimisation and orthogonal reparameterisations is listed in
`docs/PRIOR_ART.md`.

## Hypothesis

At the lab scale that fits this machine (a `4`-layer, `128`-wide character
transformer, about `0.8M` parameters, `1000` steps), Dervish with a spectral
cap of `1.0`:

1. keeps every hidden spectral norm at or below the cap and every measured
   attention logit at or below the analytic bound, while the uncapped Muon
   control's spectral norms and attention logits grow with training;
2. reaches a next-character validation loss within a small margin of the best
   Muon arm at the same step budget;
3. tolerates a wider range of learning rates than Muon.

Two further quantities are measured but carry no pass line, because this
hypothesis has no established metric for them: whether the copy
behaviour forms earlier under Dervish (the "simplicity bias"
concern), and whether the hidden matrices keep a higher stable rank.

## Arms

All arms share the model, the corpus, the seeds `7 11 19`, the batch sampler
and the evaluation. Only the optimizer changes.

| Arm | What it is | Why it is here |
| --- | --- | --- |
| `adamw` | AdamW, lr `1e-3`, betas `(0.9, 0.95)`, no decay | the generic control |
| `muon` | Muon lr `0.02`, momentum `0.95`, 5 Newton-Schulz steps, no decay; Adam lr `1e-3` elsewhere | the common regime (constant lr, no decay) |
| `muon_wd` | `muon` with decoupled weight decay `0.1` on the Muon group | the published fix for norm growth; the honest baseline for claim 1 |
| `dervish` | rotation lr `0.05`, spectrum lr `1e-3`, cap `1.0`, floor `0`, per-head frames for `c_attn`; Adam lr `1e-3` elsewhere | the proposal |
| `dervish_nocap` | `dervish` without the cap | separates the manifold structure from the cap |

Fixed before running: `dervish_lr=0.05` and `dervish_scale_lr=1e-3` are not
tuned on the matrix; only the rotation learning rate is swept, in E2, and
Muon's learning rate is swept over the same number of grid points.

## Experiments and command shapes

E1, next-character loss and spectra on natural text (Tiny Shakespeare,
`1,115,394` raw characters, lowercased and restricted to the `36`-symbol lab
alphabet; `10%` validation; chunked deterministic evaluation):

```bash
python lab/compare.py --corpus lab/corpus/tiny_shakespeare_raw.txt --normalize --steps 1000 --eval-interval 100 --eval-mode full --block-size 128 --batch-size 32 --n-layer 4 --n-head 4 --n-embd 128 --seeds 7 11 19 --configs adamw muon muon_wd dervish dervish_nocap --out runs/e1_text.jsonl --summary runs/e1_text.md --title "E1 Text Summary"
```

E2, learning-rate robustness, seed `7`, `600` steps, six grid points per
optimizer at ratio `2`:

```bash
python lab/compare.py --corpus lab/corpus/tiny_shakespeare_raw.txt --normalize --steps 600 --eval-interval 100 --eval-mode full --seeds 7 --configs muon --sweep-key muon_lr --sweep-values 0.005 0.01 0.02 0.04 0.08 0.16 --out runs/e2_muon.jsonl --summary runs/e2_muon.md --title "E2 Muon LR Sweep"
python lab/compare.py --corpus lab/corpus/tiny_shakespeare_raw.txt --normalize --steps 600 --eval-interval 100 --eval-mode full --seeds 7 --configs dervish --sweep-key dervish_lr --sweep-values 0.0125 0.025 0.05 0.1 0.2 0.4 --out runs/e2_dervish.jsonl --summary runs/e2_dervish.md --title "E2 Dervish LR Sweep"
python lab/compare.py --corpus lab/corpus/tiny_shakespeare_raw.txt --normalize --steps 600 --eval-interval 100 --eval-mode full --seeds 7 --configs adamw --sweep-key lr --sweep-values 0.00025 0.0005 0.001 0.002 0.004 0.008 --out runs/e2_adamw.jsonl --summary runs/e2_adamw.md --title "E2 AdamW LR Sweep"
```

E3, copy behaviour on the long-context copy corpus
(`512` lines, generator seed `20260617`, `38,921`
characters, `8` keys, chance `0.125`), plain and with a
copy-position weight `50`:

```bash
python lab/compare.py --corpus lab/corpus/long_context_seed.txt --steps 1000 --eval-interval 100 --eval-mode full --block-size 96 --batch-size 32 --n-layer 4 --n-head 4 --n-embd 128 --seeds 7 11 19 --copy-probe-marker "answer=" --configs adamw muon muon_wd dervish adamw_copyw muon_copyw muon_wd_copyw dervish_copyw --out runs/e3_copy.jsonl --summary runs/e3_copy.md --title "E3 Copy Behaviour Summary"
```

## Primary decision metrics and pass or fail lines

Claim 1, bounded spectra and logits (E1, all `dervish` runs, every evaluation
of every seed):

- PASS: `hidden_spectral_norm_max <= 1.0 + 1e-4` and
  `att_logit_max <= att_logit_bound` at every evaluation, and the `muon` arm's
  mean `hidden_spectral_norm_max` at step `1000` exceeds its step-`0` value by
  at least `2x`.
- PARTIAL: the Dervish inequalities hold but the Muon control does not grow
  `2x` in `1000` steps (the fix is real but the disease is not visible at this
  scale).
- KILL: any Dervish evaluation violates either inequality (an implementation
  defect, since the bound is a theorem).

Claim 2, quality parity (E1, mean of the final validation NLL over seeds
`7 11 19` at step `1000`, `dervish` versus the better of `muon` and `muon_wd`):

- PASS: `dervish` mean is at most `+0.030` NLL worse (or better).
- PARTIAL: between `+0.030` and `+0.100`.
- KILL: more than `+0.100` worse. The structure costs too much at this scale
  to be a drop-in replacement.

Claim 3, learning-rate robustness (E2, seed `7`, step `600`). For each
optimizer, the stable basin is the number of grid points whose final
validation NLL is within `0.100` of that optimizer's best grid point; a run
that diverges (non-finite loss) or ends above `3.0` NLL is outside the basin.

- PASS: Dervish's basin has at least one more grid point than Muon's.
- PARTIAL: the basins are equal in size.
- KILL: Dervish's basin is smaller.

Exploratory, no pass line (E3 and E1): first evaluation step at which the copy
probe accuracy reaches `0.25` (twice chance) per arm; final copy accuracy;
final mean stable rank of the hidden matrices per arm. These are reported as
measured and interpreted cautiously.

## Risks and confounds

- The Dervish initialisation projects each hidden matrix onto the manifold
  (`W <- diag(row norms) polar(W)`), so its step-`0` loss differs slightly from
  the controls' (`3.579` versus `3.625` in the smoke test). Every curve is
  reported from step `0`, so the offset is visible rather than hidden.
- The spectrum learning rate `1e-3` was fixed by analogy with LayerNorm gains
  and is not swept. A worse Dervish NLL may be a spectrum-speed problem, not a
  structure problem; the `dervish_nocap` arm and the per-evaluation
  `dervish_scale_max` curve are there to tell those apart.
- Newton-Schulz runs in float32 on this CPU for every arm. GPU trainers
  usually run it in bfloat16 on CUDA. The comparison between optimizers is
  fair; the absolute numbers are not the GPU family's.
- One seed in E2. The basin sizes are coarse (grid ratio `2`) on purpose.
- Scale. `0.8M` parameters and `1000` steps on `1.1M` characters say nothing
  about `200M` parameters on `494M` characters. A PASS here earns a GPU-scale
  stage, not a claim about large models.
- Tiny Shakespeare is one small corpus. The copy corpus is synthetic.

## What result would change the plan

- PASS on claims 1 to 3: register a stage that reruns E1 on a GPU trainer
  (through the integration note in `examples/trainer_integration.md`) at
  about `25M` parameters on CUDA, with the bfloat16 Newton-Schulz path.
- PASS on 1, PARTIAL on 2: keep Dervish as the candidate for runs where
  stability matters more than the last `0.05` NLL (long constant-lr runs), and
  open a spectrum-learning-rate sweep before any larger stage.
- KILL on 2 or 3: the parameterisation is not a drop-in replacement at this
  scale. Keep the measurement harness (the logit and spectrum curves are
  useful on their own) and retire the optimizer branch unless the GPU-era lab
  shows Muon logit growth that this scale could not.
