# Dervish · a rotation-based successor to Muon

A research artefact built and measured on 2026-10-09: an optimizer for the
hidden matrices of a transformer that bounds spectral norms and attention
logits by construction, delivered with a pre-registered hypothesis, a
comparison matrix over three seeds, raw run files, figures regenerated from
them, a prior-art note, and a results record that says what it does not
prove.

## What it is, in one paragraph

Muon orthogonalises every update and leaves the weights unbounded. Dervish
keeps every hidden weight matrix in the factored form `W = diag(s) Q` with `Q`
orthonormal, moves `Q` only by rotations (one small orthogonal gate per step,
found with the same Newton-Schulz machinery Muon already uses) and trains the
spectrum `s` as a separate, capped vector. The singular values of `W` are
`|s|` exactly, at every step, so spectral norms and attention logits carry
bounds that are computed from the live model and checked against measurement
at every evaluation. The name is the Mevlevi dervish: the optimizer whirls
instead of pushing. The quantum half is the mathematics it reuses: unitary
evolution of the frame, and the polar factor as the nearest unitary to a
drifted gate (`docs/QUANTUM_CONNECTION.md`, `examples/nearest_unitary.py`).

## Run it

Requirements: Python 3.11 or newer, PyTorch 2.1 or newer, matplotlib, pytest.
No accounts, no keys, no network after `pip install`. CPU is enough; CUDA is
optional (`--device cuda`).

```bash
pip install -r requirements.txt
./verify.sh            # Windows: .\verify.ps1   (tests, examples, 20-step smoke per optimizer; about two minutes)
./run_matrices.sh e1   # the matrices behind docs/RESULTS.md; e2, e3, or all (about three hours on four CPU threads)
python lab/figures.py  # figures/*.png from runs/*.jsonl
python lab/check_claims.py   # the pre-registered verdicts, computed from the run files
```

Use the optimizer in your own code:

```python
from dervish import FrameView, build_dervish

optimizer, report = build_dervish(
    model.named_parameters(),
    is_frame=lambda name, p: p.ndim >= 2 and name.startswith("blocks."),
    views={"blocks.0.attn.c_attn.weight": FrameView.heads(3 * n_head, n_embd // n_head, n_embd)},
    frame_kwargs={"lr": 0.05, "scale_lr": 1e-3, "scale_cap": 1.0},
    adam_kwargs={"lr": 1e-3},
)
optimizer.initialize()   # project the hidden matrices onto the manifold before step 0
```

`examples/trainer_integration.md` shows the branch to add to an existing
trainer's `build_optimizer`.

## Layout

| Path | What it is |
| --- | --- |
| `dervish/` | the optimizer package: `polar.py` (two Newton-Schulz iterations), `matricize.py` (frame views for 2D, expert banks, convolutions, per-head projections), `optimizer.py` (Dervish, and a Muon reference that mirrors a reference Muon trainer line for line), `diagnostics.py` |
| `lab/` | a self-contained, CPU-sized character-level language-model lab: instrumented `TinyTransformer`, deterministic data, `train.py`, `compare.py` (matrices), `figures.py`, `check_claims.py` |
| `lab/corpus/` | Tiny Shakespeare (raw, `1,115,394` bytes) and a synthetic long-context copy corpus (`512` lines, generator seed `20260617`) |
| `tests/` | 24 pytest tests: polar factors against SVD, the manifold invariant for every shape rule, the spectral cap under pressure, the rank floor, determinism, the rotation-gate interpretation, and the lab harness itself (causality, split alignment, loss mask, bound consistency) |
| `docs/DESIGN.md` | the algorithm, the four propositions with proofs, knobs, cost, limits |
| `docs/HYPOTHESIS-D001.md` | the pre-registered claims and pass or fail lines, written before any matrix ran |
| `docs/RESULTS.md` | the staged evidence record: command shapes, seeds, parameter counts, NLL and bits/char, logit and spectrum curves, wall-clock, what each stage does not prove |
| `docs/PRIOR_ART.md` | what Dervish borrows (Muon, unitary RNNs, Stiefel optimisation, Lipschitz-constrained Muon, MuonClip) and the small composition that is new |
| `docs/QUANTUM_CONNECTION.md` | exactly how far the quantum analogy goes |
| `examples/` | `nearest_unitary.py` (a drifted CNOT repaired with the optimizer's own routine), `experts_demo.py` (one optimizer over a dense layer, a mixture-of-experts bank and a convolution), `trainer_integration.md` |
| `runs/` | the raw `*.jsonl` run files and `*.md` summaries the results rest on, plus the launcher logs |
| `figures/` | PNGs regenerated from `runs/` by `lab/figures.py` |

## Results

All numbers below are means over seeds `7 11 19` at step `1000` unless
stated, from `runs/*.jsonl`; the staged record with command shapes,
per-seed values and what each stage does not prove is `docs/RESULTS.md`,
and `python lab/check_claims.py` recomputes every verdict.

**Pre-registered verdicts** (`docs/HYPOTHESIS-D001.md`, lines fixed before
any run):

| Claim | Verdict | The number |
| --- | --- | --- |
| 1 · bounded spectra and logits | **PASS** | Dervish's largest hidden spectral norm over every evaluation of every seed: `0.802` (cap `1.0`); its attention logits stayed below the analytic bound by at least `22.3`. The Muon control grew its spectral norm `15x` in `1000` steps (`0.68` to `10.4`) and its largest attention logit to `52` to `64`; AdamW and weight-decayed Muon sat near `23`. |
| 2 · loss parity with Muon | **KILL** | `1.8226` versus `1.5519` NLL (`+0.271`); the line was `+0.030`. `dervish_nocap` was identical in every seed, so the cap was not the cause: the spectrum learning rate fixed before the run (`1e-3`) was about `10x` too slow and the hidden layers stayed at gain `0.75`. |
| 3 · wider learning-rate basin | **KILL** | On the pre-registered six-point grids, within `0.1` of each optimizer's best: Muon `3`, Dervish `2`, AdamW `4`. Dervish's best was the smallest angle on its grid (`0.0125`), four times below the default fixed before the run. Beyond their basins Muon's logits reached `21,100` and AdamW's `891`; Dervish's never exceeded `10.4` at any angle. |

**Post-hoc diagnosis and repair** (Stages D4 to D7, not pre-registered,
reported as the basis for the next registration):

- A faster spectrum (`0.01`) and a smaller angle (`0.0125`) take Dervish
  from `1.815` to `1.590` on seed `7`; over three seeds the capped tuned
  arm averages `1.5929`, a tie with weight-decayed Muon (`1.5837`,
  `+0.009`) and `+0.041` behind undecayed Muon, at a peak attention logit
  of `16.3` against Muon's `56.6`, with its hidden spectral norm held at
  exactly `1.000` and a stable rank of `60` against Muon's `35`.
- The mechanism behind the wrong angle is the study's most useful finding:
  a Muon step has fixed spectral norm, so as the weights grow each step
  becomes a smaller rotation. Muon's norm growth is an implicit
  learning-rate decay, and Dervish, which forbids the growth, loses the
  free decay with it. Give both optimizers the same explicit linear decay and
  the pair finishes `0.016` apart over three seeds (`1.5112` versus
  `1.4952`), inside the margin the hypothesis set for parity, at a peak
  attention logit of `16.6` against `30.7` and with Dervish's hidden
  spectral norm still pinned at `1.000`.
- The copy behaviour (Stage D3, the long-context copy corpus): under plain training
  the behaviour formed first under Muon (`100` steps), then AdamW, then
  Dervish last; under a copy-position loss weight every arm, Dervish
  included, passed twice chance at step `100`, and Dervish reached `0.987`
  with its logits bounded at `11`. The simplicity-bias hope is not
  supported at this scale; the bound does not forbid the behaviour.

Figures: `figures/e1_text_curves.png` (the four curves of Stage D1),
`figures/e2_lr_basins.png`, `figures/e3_copy_plain_curves.png` and
`figures/e3_copy_weighted_curves.png`, `figures/e4_scale_lr.png`,
`figures/e5_tuned_curves.png`, `figures/e6_e7_replication.png`.

## What was verified, on what, and what was not

Verified on 2026-10-09, in a Linux x86-64 cloud container (kernel `6.18`,
`4` CPU threads of an Intel Xeon at `2.10 GHz`, `15 GB` RAM), Python
`3.13.16`, PyTorch `2.14.1` on CPU, matplotlib `3.11.2`, pytest `9.1.1`:

- `python -m pytest -q tests`: `24` passed (polar factors against an SVD
  reference, the factored form `W = diag(s) Q` after optimisation for 2D,
  tall, square, per-head, convolution and expert-bank frames, the spectral
  cap under deliberately violent steps, the rank floor, determinism of two
  identical runs, the rotation-gate interpretation, the lab's causality,
  split alignment, loss mask and bound consistency).
- `examples/nearest_unitary.py` and `examples/experts_demo.py` run and print
  what their docstrings say.
- `./verify.sh` end to end: `24` tests passed, both examples printed their tables, the three `20`-step smokes gave validation NLL `2.8201` (AdamW), `2.5592` (Muon) and `2.5726` (Dervish) with largest hidden spectral norms `1.024`, `1.081` and `0.502`, and the verdict line read `claim 1: PASS   claim 2: KILL   claim 3: KILL`.
- Every matrix in `docs/RESULTS.md` was run here by `run_matrices.sh`
  (`79` training runs, `936 KB` of run files); the figures were regenerated
  from those files by `lab/figures.py`; the verdicts were computed by
  `lab/check_claims.py`. Determinism was checked across launches: the E6
  seed-`7` run reproduced the E5 seed-`7` run to every printed digit.

Not verified: anything on Windows or PowerShell (`verify.ps1` and
`run_matrices.ps1` were written to mirror the bash scripts and never
executed; the `--threads`, `--device` and path handling should behave, but
check `python -m pytest -q tests` first); the CUDA path and the bfloat16
Newton-Schulz option (`--ns-dtype bfloat16`, the usual GPU
setting) on a GPU; any model larger than `0.8M` parameters or any budget
longer than `1000` steps; the integration note in
`examples/trainer_integration.md` against the real trainer (it was
written from reading `build_optimizer`, not by running it); the citations
in `docs/PRIOR_ART.md`, which could not be checked against their sources
from this container and are marked as such.

## Reading order

`docs/DESIGN.md`, then `docs/HYPOTHESIS-D001.md`, then `docs/RESULTS.md`,
then `docs/PRIOR_ART.md` before anyone cites this, then
`docs/QUANTUM_CONNECTION.md` and `examples/trainer_integration.md` for the
next step.
