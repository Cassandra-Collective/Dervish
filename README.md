<p><a href="https://github.com/Cassandra-Collective"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Cassandra-Collective/.github/main/brand/svg/cassandra-mark-small-on-dark.svg"><img src="https://raw.githubusercontent.com/Cassandra-Collective/.github/main/brand/svg/cassandra-mark-small-on-light.svg" alt="Cassandra Collective mark" width="20" height="20" align="absmiddle"></picture></a>&nbsp; Part of the <a href="https://github.com/Cassandra-Collective">Cassandra Collective</a></p>

# Dervish · a Muon variant with a provably bounded spectrum

Dervish is a PyTorch optimizer for the hidden matrices of a transformer. It
keeps every hidden matrix as a capped spectrum times an orthonormal frame, so
the spectral norm of every hidden matrix, and with it every attention logit,
has a bound you can compute from the live model. It ships with a
pre-registered evaluation record: hypotheses written before the runs, raw run
files, verdicts computed from them by script, and a statement of what each
result does not prove.

## The evidence in five lines

At `0.8M` parameters, `1000` steps, character-level Tiny Shakespeare, on CPU:

1. **The bound works.** Every hidden spectral norm stays at or below the cap,
   and attention logits stay near `16`. The Muon family reaches `24` to `57`,
   and up to `21,100` at a bad learning rate.
2. **It costs loss.** With a linear decay schedule, Dervish finishes `+0.026`
   NLL behind the best Muon-family optimizer on five fresh seeds (95% interval
   `[+0.015, +0.038]`). At a constant learning rate it is `+0.057` behind.
3. **Against MONA** (arXiv:2605.26842), Dervish is `+0.018` behind with decay
   and `+0.054` behind at a constant rate. MONA itself does not beat Muon at
   this scale, and it makes weight and logit growth worse.
4. **It is not a successor to Muon.** It trades a few hundredths of a nat for
   a hard spectral bound. Whether that trade pays off at scale is untested.
5. **Most of its parts have prior art** (`docs/PRIOR_ART.md`). What is new is
   the composition.

## What it is, in one paragraph

Muon orthogonalises every update and leaves the weights unbounded. Dervish
keeps every hidden weight matrix in the factored form `W = diag(s) Q`, with `Q`
orthonormal. It moves `Q` only by rotations, one small orthogonal step per
update found with the Newton-Schulz machinery Muon already uses, and it trains
the spectrum `s` as a separate vector with Adam under a hard clamp. The
singular values of `W` are `|s|` exactly, at every step. Spectral norms are
therefore capped by construction, and attention logits have a bound that is
computed from the live parameters and checked against measurement at every
evaluation.

The name is the Mevlevi dervish: the optimizer whirls instead of pushing. The
quantum connection is limited to the mathematics it reuses
(`docs/QUANTUM_CONNECTION.md`).

## Install

Requirements: Python 3.11 or newer and PyTorch 2.1 or newer. The lab and tests
also need matplotlib and pytest. CPU is enough; CUDA is optional.

```bash
pip install .              # the optimizer package only (import dervish)
pip install ".[test]"      # plus pytest and matplotlib for the tests and the lab
```

## Use it

```python
from dervish import FrameView, build_dervish

optimizer, report = build_dervish(
    model.named_parameters(),
    is_frame=lambda name, p: p.ndim >= 2 and name.startswith("blocks."),
    views={"blocks.0.attn.c_attn.weight": FrameView.heads(3 * n_head, n_embd // n_head, n_embd)},
    frame_kwargs={"lr": 0.0125, "scale_lr": 0.01, "scale_cap": 1.0},
    adam_kwargs={"lr": 1e-3},
)
optimizer.initialize()   # project the hidden matrices onto the manifold before step 0
```

- The frame settings above are the ones measured in `docs/RESULTS.md`
  (Stages D5 to D8). `lr` is a rotation angle in radians.
- Use them with a learning-rate schedule. Dervish forbids the weight growth
  that acts as Muon's implicit decay, so a constant rate costs it twice as
  much loss (Stage D8).
- The package also contains the reference Muon (`build_muon`) and the MONA term
  (`mona_beta=...`) used in the comparison.
- `examples/trainer_integration.md` shows the branch to add to an existing
  trainer's `build_optimizer`.

## Reproduce the record

```bash
./verify.sh                  # Windows: .\verify.ps1 (34 tests, two examples, a 20-step smoke per optimizer; about two minutes)
./run_matrices.sh e1         # e1, e2, e3 or all: the pre-registered matrices of Hypothesis D001 (about three hours on four CPU threads)
./run_matrices.sh d8         # Hypothesis D002, Dervish against MONA (70 runs; about four hours with 8 parallel jobs)
python lab/check_claims.py   # D001 verdicts, computed from runs/
python lab/check_d002.py     # D002 verdicts, computed from runs/d8/
python lab/figures.py        # figures/*.png from the run files
```

`run_matrices.ps1` covers `e1`, `e2`, `e3` and `d8`. The post-hoc stages E2b
and E4 to E7 are in `run_matrices.sh` only.

## Results

The staged record, with command shapes, per-seed values and what each stage
does not prove, is `docs/RESULTS.md`.

**Hypothesis D001** (`docs/HYPOTHESIS-D001.md`; seeds `7 11 19`, lines fixed
before any run):

| Claim | Verdict | The number |
| --- | --- | --- |
| 1 · bounded spectra and logits | **PASS** | Dervish's largest hidden spectral norm over every evaluation: `0.802` (cap `1.0`). Uncapped Muon grew its spectral norm `15x` in `1000` steps and its largest attention logit to `52` to `64`. |
| 2 · loss parity with Muon | **KILL** | `+0.271` NLL; the line was `+0.030`. The spectrum learning rate fixed before the run was about `10x` too slow. |
| 3 · wider learning-rate basin | **KILL** | Basins of `2` grid points for Dervish against `3` for Muon. The default angle was about `4x` too large. Dervish's logits never exceeded `10.4` at any angle; Muon's reached `21,100`. |

**Post-hoc diagnosis** (Stages D4 to D7). A faster spectrum and a smaller
angle repair most of the gap. The most useful finding is the mechanism:

- A Muon step has fixed spectral norm, so as Muon's weights grow, each step
  becomes a smaller rotation. Weight growth acts as an implicit learning-rate
  decay.
- Dervish forbids the growth and loses that free decay with it.
- With the same explicit decay for both, the pair finished `0.016` apart over
  seeds `7`, `11` and `19`.

**Hypothesis D002** (`docs/HYPOTHESIS-D002.md`) was written to test that
post-hoc result fairly. It uses MONA as a stronger Muon-family baseline, gives
every family one tuning look at seed `7`, and evaluates on five fresh seeds
(`23 29 31 37 41`):

| Schedule | Muon | MONA | Dervish | Dervish − best | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| linear decay (primary) | `1.4848` | `1.4930` | `1.5109` | `+0.026` `[+0.015, +0.038]` | **PASS** on the registered `+0.030` line |
| constant | `1.5328` | `1.5357` | `1.5893` | `+0.057` `[+0.046, +0.067]` | PARTIAL |

- **The decay PASS is narrow.** It holds on the mean, by `0.004`. The interval
  lies above zero, so Dervish is reliably a little worse than Muon, and a
  stricter non-inferiority rule would not have passed.
- **The bound held** in all `20` Dervish runs. Peak attention logits were
  `15.5` to `16.9`, against `24` to `39` for Muon and MONA.
- **MONA's term did not help Muon** (constant: NOT REPLICATED; decay:
  REVERSED). Under the paper's tie, the term equals the mean of the gradient
  and its slow EMA, and on seed `7` it grew spectral norms faster than Muon at every matched
  learning rate.
- **MONA's term does not compose with Dervish:** HURTS at a constant rate,
  NEUTRAL with decay.

Figures: `figures/e1_text_curves.png`, `figures/e2_lr_basins.png`,
`figures/e3_copy_plain_curves.png`, `figures/e3_copy_weighted_curves.png`,
`figures/e4_scale_lr.png`, `figures/e5_tuned_curves.png`,
`figures/e6_e7_replication.png`, `figures/d8_mona_comparison.png`.

## Limits worth knowing before you use it

- **Scale.** Nothing here goes beyond `0.8M` parameters, `1000` steps, one
  character-level corpus and CPU float32. The CUDA and bfloat16 Newton-Schulz
  paths are untested on a GPU.
- **The bound is not a constant.** It scales with the LayerNorm gains and the
  q/k biases, which Adam trains without limit. It rose from `22.6` to `33`
  to `38` over `1000` steps while measured logits stayed near `16`. A constant
  bound needs the gains capped too (`docs/DESIGN.md` section 8).
- **At this scale the problem it solves is mild once a schedule is used.**
  Muon with decay peaks at an attention logit of about `30`.
- **Copy behaviour.** With a slow spectrum, Dervish was the last optimizer to
  form the copy behaviour (Stage D3). The simplicity-bias hope is not
  supported.

## Layout

| Path | What it is |
| --- | --- |
| `dervish/` | The optimizer package. `polar.py`: Newton-Schulz iterations. `matricize.py`: frame views for 2D matrices, expert banks, convolutions and per-head projections. `optimizer.py`: Dervish, the reference Muon and the optional MONA term. `diagnostics.py`. |
| `lab/` | A self-contained, CPU-sized character-level language-model lab: an instrumented `TinyTransformer`, deterministic data, `train.py`, `compare.py`, `run_d8.py`, `figures.py`, `check_claims.py`, `check_d002.py`. |
| `lab/corpus/` | Tiny Shakespeare (raw, `1,115,394` bytes) and a synthetic long-context copy corpus (`512` lines, generator seed `20260617`). |
| `tests/` | 34 pytest tests: polar factors against SVD, the manifold invariant for every shape rule, the spectral cap under pressure, the rank floor, determinism, the rotation-gate interpretation, MONA (`alpha = 0` is Muon bit for bit; the tied-term identity), and the lab harness itself. |
| `docs/DESIGN.md` | The algorithm, the four propositions with proofs, knobs, cost and limits. |
| `docs/HYPOTHESIS-D001.md`, `docs/HYPOTHESIS-D002.md` | The pre-registered claims and pass or fail lines. |
| `docs/RESULTS.md` | The staged evidence record, Stages D1 to D8. |
| `docs/PRIOR_ART.md` | Verified citations, the closer prior work found later, and what is new once that is accounted for. |
| `docs/QUANTUM_CONNECTION.md` | Exactly how far the quantum analogy goes. |
| `docs/implementations/` | Engineering notes on how the repository was changed, and why. |
| `examples/` | `nearest_unitary.py` (a drifted CNOT repaired with the optimizer's own routine), `experts_demo.py` (one optimizer over a dense layer, a mixture-of-experts bank and a convolution), `trainer_integration.md`. |
| `runs/` | The raw `*.jsonl` run files and `*.md` summaries the results rest on, with the launcher logs. Stage D8 is in `runs/d8/`. |
| `figures/` | PNGs drawn from `runs/` by `lab/figures.py`. |

## What was verified, on what, and what was not

**2026-10-09: Stages D1 to D7.** Linux x86-64 cloud container, `4` CPU
threads, Python `3.13.16`, PyTorch `2.14.1` on CPU.

- `24` tests passed, and both examples ran.
- `79` training runs (every matrix of Stages D1 to D7), with figures and
  verdicts computed from the run files.
- Determinism was checked across launches.

**2026-10-14: Stage D8 and this release.** Windows 11, Python `3.13.14`,
PyTorch `2.12.1+cu126` on CPU.

- `.\verify.ps1` end to end, after fixing a quoting bug in its smoke step:
  - `29` tests passed;
  - both examples ran;
  - the three `20`-step smokes reproduced the Linux values to every printed
    digit (`2.8201`, `2.5592`, `2.5726`);
  - the D001 verdict line was recomputed.
- `70` training runs for Stage D8, with seed-`7` reproductions of the record
  within `0.003`, one of them exact.
- The `d8` stage of both `run_matrices.sh` and `run_matrices.ps1` was checked
  on the finished run files.
- The wheel builds and installs, with only `dervish/` inside.
- Every citation in `docs/PRIOR_ART.md` was checked against its source.

**Not verified:**

- the PowerShell runs of `e1` to `e3` end to end;
- CUDA, or bfloat16 Newton-Schulz on a GPU;
- anything larger than `0.8M` parameters or longer than `1000` steps;
- `examples/trainer_integration.md` against a real external trainer.

## Citing the work this builds on

- K. Jordan et al., "Muon: An optimizer for hidden layers in neural networks",
  2024, <https://kellerjordan.github.io/posts/muon/>.
- J. Li et al., "MONA: Muon Optimizer with Nesterov Acceleration for Scalable
  Language Model Training", arXiv:2605.26842, Findings of EMNLP 2026.
- K. Lion et al., "Muown: Row-Norm Control for Muon Optimization",
  arXiv:2605.10797, 2026. It has the same magnitude/direction gradient split,
  without a bound.

The rest are in `docs/PRIOR_ART.md`.

## Reading order

`docs/DESIGN.md`, `docs/HYPOTHESIS-D001.md`, `docs/RESULTS.md` (Stages D1 to
D7), `docs/HYPOTHESIS-D002.md`, `docs/RESULTS.md` Stage D8, then
`docs/PRIOR_ART.md` before anyone cites this.

## Contributing

Contributions are welcome, including experiments that show Dervish losing.
`CONTRIBUTING.md` covers setup, where to find work, the pre-registration
workflow for experiments, and pull requests. Everyone taking part follows the
[Cassandra Collective code of conduct](https://github.com/Cassandra-Collective/.github/blob/main/CODE_OF_CONDUCT.md).
Please report security issues privately, as the
[security policy](https://github.com/Cassandra-Collective/.github/blob/main/SECURITY.md)
describes.

## License

MIT, see `LICENSE`.
