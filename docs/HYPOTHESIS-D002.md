# Hypothesis D002 · Dervish against MONA on fresh seeds

- Status: REGISTERED 2026-10-14, before any run in this file. The selection
  rules, arms, seeds and pass lines below are fixed. Verdicts will be computed
  by `lab/check_d002.py` from the run files under `runs/d8/` and recorded in
  `docs/RESULTS.md` Stage D8. This file will not be edited after the
  runs, except to add the status line at its top.
- Date: 2026-10-14
- Builds on: `docs/HYPOTHESIS-D001.md` (claim 2 KILL on its own lines) and the
  post-hoc Stages D5 to D7 of `docs/RESULTS.md`, which found that Dervish with
  a rotation angle of `0.0125`, a spectrum rate of `0.01` and cap `1.0` finishes
  `+0.016` NLL behind Muon when both decay linearly to `10%`.
- The new baseline: MONA, "Muon Optimizer with Nesterov Acceleration for
  Scalable Language Model Training" (Li, Tan, Xu, Zhang, Lu, Sun, Xie and Cai,
  Meituan; arXiv:2605.26842 v2, 22 Sep 2026; Findings of EMNLP 2026).

## Why this test

Stages D5 to D7 have two weaknesses that a reader would find first.

1. **Tuned and scored on the same seed.** Dervish's angle and spectrum rate
   were chosen by looking at seed `7`, then reported as means over seeds
   `7 11 19`. Seed `7` is in both sets.
2. **The baseline is the weakest Muon in the lab.** Muon at `0.02` without
   tuning; the record itself says Muon at `0.01` with decay was never run.

MONA is a Muon variant that reports lower loss than Muon at 1B to 68B
parameters. If Dervish is a successor to Muon, it should survive a stronger
member of the Muon family, scored on seeds nobody tuned on.

## The MONA arm

The repo's Muon (`dervish/optimizer.py: MuonWithAuxAdam`) with one change: the
gradient of every hidden matrix is replaced, before the momentum, by

    D_k = G_k - G_{k-1}            (G_0 = 0)
    A_k = beta_a A_{k-1} + (1 - beta_a) D_k
    G~_k = G_k + alpha A_k,   alpha = -1 / (2 (1 - beta_a))     (the paper's tie)

(paper equations 7 to 9 and Algorithm 1; `dervish/optimizer.py: mona_accelerate`).
Everything else is the repo's Muon: the Nesterov `lerp` momentum `0.95`, the
`max(1, m/n)^0.5` scale and five quintic Newton-Schulz steps. The paper's
Algorithm 1 instead uses heavy-ball momentum `M = mu M + G~` and the
`0.2 sqrt(max(m, n))` scale of Moonshot's Muon. Those are left alone on
purpose, so that MONA versus Muon here changes exactly one thing. `dervish_mona`
puts the same term in front of Dervish's frame momentum (on `diag(s) G`); the
spectrum's Adam is untouched, as MONA leaves Adam's vectors untouched.

## Stage D8a · selection, seed 7 only

All runs: the Stage D1 model and corpus (4 layers, width 128, block 128,
batch 32, Tiny Shakespeare normalised, chunked full evaluation every `100`
steps), `1000` steps, CPU, float32 Newton-Schulz, two threads per run. Each
grid is run twice: once at constant rate and once with `--lr-decay-to 0.1`.

| Family | Grid | Points per schedule |
| --- | --- | ---: |
| Muon | `muon_lr` in `{0.01, 0.02}` | `2` |
| MONA | `muon_lr` in `{0.01, 0.02}` x `beta_a` in `{0.9, 0.95, 0.98, 0.99}` | `8` |
| Dervish | the Stage D5 setting, not retuned (angle `0.0125`, spectrum `0.01`, cap `1.0`) | `1` |
| Dervish + MONA | the Dervish setting x `beta_a` in `{0.9, 0.95, 0.98, 0.99}` | `4` |

Selection rule, per family and per schedule: the grid point with the lowest
seed-`7` validation NLL at step `1000`. Dervish's knobs were already chosen on
seed `7` in Stage D5 with a comparable budget, so every family has had one
look at seed `7`.

The Muon and Dervish seed-`7` constant-rate runs also check this machine
against the record: `1.538644` (Muon `0.02`), `1.527197` (Muon `0.01`) and
`1.590136` (Dervish tuned) in `docs/RESULTS.md`. Agreement to `0.01` is
expected; bitwise agreement is not (another CPU, another PyTorch build,
Windows). A larger gap is reported and stops the stage.

## Stage D8b · evaluation, fresh seeds

Seeds `23 29 31 37 41`, none used before by anyone. Eight arms: the selected
point of each family (Muon, MONA, Dervish, Dervish + MONA) under each schedule
(constant, decay to `10%`). Forty runs, same settings as D8a.

## Pass lines

All differences are paired by seed (same initialisation and batch order) at
step `1000`: `delta(X, Y)` is the mean over the five seeds of
`NLL_X - NLL_Y`, with a 95% paired-t interval (`t = 2.776`, four degrees of
freedom). The reference `R` for a schedule is whichever of the selected Muon
and the selected MONA has the lower mean NLL on the fresh seeds. Choosing `R`
after the fact favours the baseline, which is the conservative direction for
a claim about Dervish.

**Q1 · Does MONA beat Muon at this scale?** (per schedule)

- REPLICATED: the interval for `delta(MONA, Muon)` lies below `0`.
- NOT REPLICATED: the interval contains `0`.
- REVERSED: the interval lies above `0`.

**Q2 · Is Dervish as good as the best of the Muon family?** Primary schedule:
decay (where Stage D7 placed Dervish closest). The constant schedule is
reported with the same lines as secondary.

- PASS: `delta(Dervish, R) <= +0.030`, and the bound of Q3 holds.
- PARTIAL: `+0.030 < delta(Dervish, R) <= +0.100`.
- KILL: `delta(Dervish, R) > +0.100`, or the bound fails.

**Q3 · Does the bound still hold, and does MONA grow like Muon?**

- Bound, pass or fail: at every evaluation of every Dervish and Dervish +
  MONA run, the largest hidden spectral norm is at most `1.0 + 1e-4` and the
  largest attention logit is at most the analytic bound of Proposition 2.
- Growth, measured with no line: the ratio of MONA's hidden spectral norm at
  step `1000` to its value at step `0`, and its peak attention logit, beside
  Muon's.

**Q4 · Does MONA's term compose with Dervish?** (per schedule)

- COMPOSES: the interval for `delta(Dervish + MONA, Dervish)` lies below `0`.
- NEUTRAL: it contains `0`.
- HURTS: it lies above `0`.

## What the answer decides

- **Q2 PASS on the primary schedule:** Dervish keeps parity with the best
  Muon-family optimizer measured here while holding its bound. The repository
  is prepared for open source (licence, packaging, README led by the honest
  record, citations checked).
- **Q2 PARTIAL or KILL:** Dervish is not, at this scale, as good as the
  Muon family it claims to succeed. The repository gets an audit of why
  instead: which claims stand, which fall, what the mechanism costs, and
  what would have to be true for a larger test to be worth running.

## What this test cannot say

- Nothing about MONA's own regime. The paper trains mixture-of-experts models
  of 1B to 68B parameters on 400B to 1T tokens with batch sizes this machine
  cannot hold. Here the model has `0.8M` parameters, the batch is `32 x 128`
  characters and the budget is `1000` steps. A NOT REPLICATED on Q1 says
  MONA's term does not help *here*, not that the paper is wrong.
- `beta_a` was chosen in the paper for runs `10^5` times longer. The grid
  above (`0.9` to `0.99`) moves the term's memory from about `10` to about
  `100` steps so that it is not unfairly long for a `1000`-step run.
- Anything beyond Tiny Shakespeare, CPU float32 and one machine.
