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

RESULTS_PLACEHOLDER

## What was verified, on what, and what was not

VERIFICATION_PLACEHOLDER

## Reading order

`docs/DESIGN.md`, then `docs/HYPOTHESIS-D001.md`, then `docs/RESULTS.md`,
then `docs/PRIOR_ART.md` before anyone cites this, then
`docs/QUANTUM_CONNECTION.md` and `examples/trainer_integration.md` for the
next step.
