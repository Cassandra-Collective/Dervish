# Dervish against MONA, and preparation for an open-source release

- Date: 2026-10-14
- Author: Mert Efe Şensoy
- Commits: `0493768` (MONA term and the D002 pre-registration, committed
  before any evaluation result was read), `a3cff0a` (Stage D8 run files,
  verdict script, figure), and the release-preparation commit that adds this
  note.

## 1 · Problem / motivation

The question was whether Dervish is really good. The test was against MONA
(arXiv:2605.26842, Findings of EMNLP 2026), a Muon variant that reports beating
Muon and AdamW at 1B to 68B parameters. The decision rule was fixed in
advance: prepare the repository for open source if Dervish holds up, otherwise
audit why it does not.

The existing record had two weaknesses that any reader would find first:

- **Tuned and scored on the same seed.** Dervish's knobs were tuned on seed `7`
  and then reported as means over seeds `7 11 19`.
- **A weak baseline.** The comparison was against an untuned Muon at lr `0.02`.

A test against a stronger Muon-family baseline, scored on seeds nobody had
tuned on, answers both.

## 2 · What changed

| File | Change |
| --- | --- |
| `dervish/optimizer.py` | Adds `mona_accelerate` and `mona_tied_alpha`, plus optional `mona_beta` and `mona_alpha` keys on the Muon group and the Dervish frame group. `mona_beta=None` reproduces plain Muon bit for bit. Docstrings reframed from "successor" to "variant". |
| `dervish/__init__.py` | Docstring reframed. |
| `lab/train.py` | `--mona-beta` and `--mona-alpha` flags; `mona_beta` and `mona_alpha` recorded in every report. |
| `lab/compare.py` | Stage D8 configs: `muon_lr01`, `mona`, `mona_lr01`, `dervish_tuned`, `dervish_tuned_mona`, each also with `_decay`. |
| `lab/run_d8.py` | Parallel launcher, one process per (config, seed, beta_a). Skips finished jobs, and can run from a frozen copy of the code. |
| `lab/check_d002.py` | Re-derives the selection, checks this machine against the record, and applies the Q1 to Q4 lines mechanically. |
| `lab/figures.py` | `fig_d8`: curves per schedule and per-seed paired differences. |
| `tests/test_mona.py` | Five tests. They check the paper's alpha tie, that `alpha = 0` is plain Muon bit for bit, that the term changes the trajectory, the closed-form identity of section 4, and that Dervish + MONA keeps the manifold and the cap. |
| `docs/HYPOTHESIS-D002.md` | The pre-registration. After the runs, a resolution line was added on top; nothing else changed. |
| `docs/RESULTS.md` | Stage D8. The header now names Stage D8's platform. A Stage D3 note discloses the corpus header change. |
| `docs/PRIOR_ART.md` | Every citation checked against its source. Adds the closer prior work (Muown, manifold Muon, SSO, Vorontsov et al., Yoshihara, sigma-Reparam). The novelty claims are restated as a composition, and the old fourth claim is dropped. |
| `docs/DESIGN.md` | New limitation: the attention-logit bound scales with LayerNorm gains that Adam can grow. |
| `README.md` | Rewritten for a public reader. It leads with the evidence, including where Dervish loses, and drops "successor to Muon" from the title. |
| `run_matrices.sh`, `run_matrices.ps1` | Each gains a `d8` stage. The PowerShell runner now stops on a failing native command. |
| `verify.ps1` | Fixes a quoting bug that broke the smoke step on its first Windows run, and makes a failing command stop the script. |
| `pyproject.toml`, `LICENSE`, `.gitattributes`, `.gitignore` | Packaging that installs `dervish/` only, under the MIT licence. Corpora are kept byte-exact on Windows checkouts, and build artefacts are ignored. |
| `runs/d8/` | `70` run reports, summaries, sanitised logs and the verdict output. |
| `figures/d8_mona_comparison.png` | New figure, drawn from `runs/d8/`. |

## 3 · Implementation approach

**MONA as one change.** MONA was implemented as a single change to the repo's
existing Muon: the gradient is replaced by MONA's accelerated gradient before
the momentum, and nothing else moves. The repo's Nesterov momentum,
`max(1, m/n)^0.5` scale and quintic Newton-Schulz are left as they are, so
MONA against Muon here differs in exactly one thing. The same term can be put
in front of Dervish's frame momentum (on `diag(s) G`, the frame gradient). The
spectrum's Adam is left alone, as MONA leaves Adam's vectors alone.

**Pre-registration.** The hypothesis was written and committed before any
evaluation run was read. It has two stages:

- **D8a (selection, seed 7).** Each family gets a small grid: Muon's lr; MONA's
  lr and `beta_a`; Dervish fixed as in Stage D5; Dervish + MONA's `beta_a`. The
  grid runs once at a constant rate and once with linear decay. The lowest
  seed-7 NLL wins.
- **D8b (evaluation).** The selected arm of each family runs on five seeds
  never used before (`23 29 31 37 41`).

**Running it.** All runs were CPU float32, because CPU and CUDA runs of the
same seed disagree and mixed-device arms are not comparable. The jobs ran
`8` in parallel at `2` threads each, from a frozen snapshot of the code, so
that editing the working tree could not change a running job.

**Release preparation.**

- Licence (MIT, the owner's choice) and packaging (`pip install .`).
- A README that answers the question first.
- A prior-art note checked against primary sources.
- The two PowerShell scripts actually run on Windows, which exposed a real bug.

## 4 · Mathematical / statistical details

**MONA's term.** Let `G_k` be the gradient at step `k`, with `G_0 = 0`, and let
`beta` be the EMA rate (`beta_a` in the paper). The term is:

    A_k = beta A_{k-1} + (1 - beta) (G_k - G_{k-1})
    G~_k = G_k + alpha A_k

Write `E_{k-1}` for the zero-initialised beta-EMA of the earlier gradients:

    E_{k-1} = (1 - beta) sum_{j<k} beta^(k-1-j) G_j

Summing the first line by parts gives an exact identity:

    A_k = (1 - beta) (G_k - E_{k-1})

The paper ties `alpha = -1 / (2 (1 - beta))` (section 8). With that tie:

    G~_k = G_k - (G_k - E_{k-1}) / 2 = (G_k + E_{k-1}) / 2

So the accelerated gradient is the mean of the current gradient and a slow
average of the past ones. In front of Muon's own momentum, that is a second,
slower momentum (the quasi-hyperbolic family of Ma and Yarats, 2019), not a
look-ahead. A longer memory makes consecutive orthogonalised updates more
aligned, so the weights grow faster. Stage D8 measures that growth: MONA's
spectral norm grew `x13.0` against Muon's `x7.8`. `tests/test_mona.py`
checks the identity to `1e-12` in float64.

**Paired differences.** For arms `X` and `Y` on seeds `s = 1..n` (`n = 5`):

    d_s = NLL_X(s) - NLL_Y(s)

The same seed means the same initialisation and the same batch order, so
pairing removes the seed-to-seed variation the two arms share. The reported
quantities are:

    mean   d_bar = (1/n) sum_s d_s
    spread s_d   = the sample standard deviation of the d_s
    95% CI       = d_bar ± t_{0.975, n-1} s_d / sqrt(n),  with t_{0.975, 4} = 2.776

**Q2 line.** With `R` the better (lower mean) of the selected Muon and MONA
arms, the verdict is:

- PASS: `d_bar(Dervish, R) <= 0.030` and the bound held;
- PARTIAL: `d_bar <= 0.100`;
- KILL: otherwise.

Choosing `R` after the fact favours the baseline. The line is on the mean, not
on the interval. A non-inferiority test, PASS only if the upper end of the
interval is `<= 0.030`, would be stricter; Stage D8 would have failed it at
`+0.038`. The next registration should use it.

**Bound check.** At every evaluation, the largest hidden spectral norm must be
`<= 1 + 1e-4`, and the measured largest attention logit must be at most the
Proposition 2 bound:

    (cap r + ||b_q||)(cap r + ||b_k||) / sqrt(d_head),  r = sqrt(n_embd) max|gamma| + ||beta||

Here `gamma` and `beta` are the attention LayerNorm's gain and bias. `gamma`
is not capped, so the bound grows during training: from `22.6` at step `0` to
`33` to `38` at step `1000`.

## 5 · Design decisions

- **One-change MONA against the paper's full Algorithm 1.** The paper uses
  heavy-ball momentum and Moonshot's `0.2 sqrt(max(m, n))` scale. Porting both
  would have made the MONA/Muon comparison a three-variable change. The cost is
  that "MONA" here is "MONA's idea on this Muon"; `docs/HYPOTHESIS-D002.md`
  says so.
- **A `beta_a` grid of `0.9` to `0.99`.** The paper's values (`0.975` to `0.99`)
  were set for runs `10^5` times longer. The grid gives MONA's memory a fair
  range for `1000` steps.
- **Fresh seeds for evaluation.** These fix the tuning-on-the-test-seed flaw of
  Stages D5 to D7.
- **CPU-only runs.** Bitwise comparability came before speed. The GPU would
  have been about five times faster.
- **Following the registered rule.** The primary verdict passed by `0.004`
  while the interval excludes zero. The tempting move was to reinterpret the
  verdict after seeing it. The registered rule was applied as written, and both
  readings are reported side by side in `docs/RESULTS.md` and the README.
- **Release preparation without publishing.** The repository stays private.
  Making it public is the owner's decision.

## 6 · Verification

```bash
python -m pytest -q tests                  # 29 passed
python lab/check_d002.py                   # re-derives the selection and prints the D002 verdicts
python lab/check_claims.py                 # D001 verdicts unchanged: PASS, KILL, KILL
./verify.sh                                # or .\verify.ps1 on Windows
pip wheel . --no-deps -w dist              # one wheel containing dervish/ and LICENSE only
```

Reproducing Stage D8 from scratch: `./run_matrices.sh d8`. That is `70` runs,
about four hours with `8` parallel jobs on a 14-core CPU. Check that the
selection `check_d002.py` prints matches the `--select` list in
`run_matrices.sh` before trusting D8b.

## 7 · Related docs

- `docs/HYPOTHESIS-D002.md`, `docs/RESULTS.md` Stage D8, `docs/PRIOR_ART.md`,
  `docs/DESIGN.md` section 8
- `docs/implementations/2026-10-14-history-rewrite.md`
