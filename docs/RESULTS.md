# Results · Dervish lab, staged record

Every number here traces to a file under `runs/`. Platform for every stage:
a Linux x86-64 cloud container (kernel `6.18`, glibc `2.39`), `4` CPU threads of an Intel Xeon at `2.10 GHz`, `15 GB` RAM, Python `3.13.16`, PyTorch `2.14.1` (a CUDA build with no GPU present, so every run is CPU float32 and every Newton-Schulz iteration runs in float32), matplotlib `3.11.2`. Two or three matrices usually shared the four cores, so the wall-clock columns are upper bounds for a quiet machine. Seeds `7 11 19` unless stated. Validation NLL is the
chunked deterministic evaluation over the whole validation split
(`--eval-mode full`), in nats per character; bits/char is NLL divided by
`ln 2`. Verdicts are computed by `lab/check_claims.py` against the lines fixed
in `docs/HYPOTHESIS-D001.md` before any run.

Stages:

- D1 · E1, next-character loss and spectra on natural text (claims 1 and 2)
- D2 · E2, learning-rate robustness (claim 3), with the post-hoc E2b extension
- D3 · E3, copy behaviour on the long-context copy corpus (exploratory)
- D4 · E4, the spectrum learning-rate sweep (post-hoc)
- D5 to D7 · tuning what the pre-registered run got wrong (post-hoc, E5 to E7)

Verdict summary, from `python lab/check_claims.py`: claim 1 PASS, claim 2 KILL,
claim 3 KILL. The post-hoc stages explain both KILLs and measure the corrected
setting; they carry no verdict of their own.

## Stage D1 · E1, next-character loss and spectra on natural text

Date: 2026-10-09 · pre-registered in `docs/HYPOTHESIS-D001.md` (claims 1 and 2).

Artifacts: `runs/e1_text.jsonl` (`15` reports with per-evaluation history),
`runs/e1_text.md` (summary tables and per-evaluation curves),
`figures/e1_text_curves.png`, `runs/logs/e1.log`.

Command shape:

```bash
python lab/compare.py --corpus lab/corpus/tiny_shakespeare_raw.txt --normalize --eval-interval 100 --eval-mode full --block-size 128 --batch-size 32 --n-layer 4 --n-head 4 --n-embd 128 --threads 2 --quiet --steps 1000 --seeds 7 11 19 --configs adamw muon muon_wd dervish dervish_nocap --out runs/e1_text.jsonl --summary runs/e1_text.md --title "E1 Text Summary"
```

Model: `4` layers, `4` heads, width `128`, RoPE, block `128`, `802,596`
trainable parameters (`786,432` of them hidden matrices, under Muon or Dervish;
the rest under Adam). Corpus: Tiny Shakespeare normalised to the `36`-symbol
lab alphabet, `1,115,345` characters, `10%` validation (`111,535` characters,
`871` chunked windows). Three seeds per arm, `1000` steps each.

| Arm | Mean val NLL | Min · max | Mean bits/char | Peak attention logit (mean) | Spectral max at step `1000` (mean) | Stable rank at `1000` (mean) | Train s (mean) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `muon` | `1.551929` | `1.538644` · `1.562646` | `2.238960` | `56.59` | `10.363` | `35.40` | `308.8` |
| `muon_wd` | `1.583726` | `1.579973` · `1.588875` | `2.284833` | `23.31` | `4.938` | `35.44` | `294.5` |
| `adamw` | `1.645718` | `1.638606` · `1.658257` | `2.374269` | `23.57` | `3.406` | `15.17` | `320.5` |
| `dervish` | `1.822598` | `1.815227` · `1.836123` | `2.629453` | `11.67` | `0.752` | `53.09` | `326.7` |
| `dervish_nocap` | `1.822598` | `1.815227` · `1.836123` | `2.629453` | `11.67` | `0.752` | `53.09` | `301.8` |

The spectrum and logit curves, means over seeds (`runs/e1_text.md` has every
evaluation; `step 0` is before any update):

| Step | `muon` spectral max | `muon` attention max | `muon_wd` spectral max | `adamw` spectral max | `dervish` spectral max | `dervish` attention max | `dervish` analytic bound |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `0.683` | `0.34` | `0.683` | `0.683` | `0.499` | `0.29` | `22.6` |
| `200` | `5.079` | `25.40` | `3.968` | `1.409` | `0.547` | `3.83` | `33.9` |
| `500` | `7.317` | `40.09` | `4.701` | `2.327` | `0.595` | `7.65` | `45.1` |
| `1000` | `10.363` | `56.59` | `4.938` | `3.406` | `0.752` | `11.67` | `58.3` |

Verdicts (`python lab/check_claims.py`):

- **Claim 1, bounded spectra and logits: PASS.** Over every evaluation of
  every Dervish seed the largest hidden spectral norm was `0.802195` (cap
  `1.0`) and the measured attention logit stayed below the analytic bound of
  Proposition 2 by at least `22.33`. The uncapped Muon control grew its
  largest hidden spectral norm by `x15.12`, `x15.12` and `x15.28` over
  `1000` steps (`0.68` to `10.4`) and its largest attention logit from `0.34`
  to `63.73`, `52.45` and `53.60`; the pass line asked for `2x`. Decoupled
  weight decay `0.1` (`muon_wd`) holds the spectral norm near `4.9` and the
  attention logits near `23`, which is the published repair doing its job.
- **Claim 2, quality parity: KILL.** `dervish` finishes `+0.270669` NLL
  worse than the best Muon arm (`1.822598` versus `1.551929`); the pass line
  was `+0.030` and the partial line `+0.100`. The structure, as configured
  here, is not a drop-in replacement at this scale.

Interpretation:

The disease the optimizer was built for is real and visible at `0.8M`
parameters: Muon without decay has no mechanism that stops its weights
growing, and in `1000` steps its hidden spectral norms reach `10` and its
attention logits `57` on average, more than twice the AdamW and `muon_wd`
levels, while its loss is the best of the five arms. Dervish removes the
growth entirely, by construction, and the measured curves agree with the
theorem to the last evaluation.

The KILL on parity has a specific, pre-named cause. `dervish_nocap` is
identical to `dervish` in every seed (`1.815227`, `1.816444`, `1.836123`
both), so the cap never bound; the spectrum simply did not move. Dervish's
largest singular value crawled from `0.499` to `0.752` in `1000` steps
(about `0.00025` per step at `scale_lr 1e-3`), while Muon's climbed about
`0.01` per step and `muon_wd`'s settled near `4.9`. With every hidden layer
held to a gain below `0.8`, the network is under-powered: attention logits
top out at `11.7` against a bound of `58`, and the loss follows. This is
the "spectrum-speed problem, not a structure problem" risk listed before
the run, and the hypothesis's own plan for it is a spectrum learning-rate
sweep, run as Stage D4 below.

Two measured side facts, both exploratory. First, rank: at step `1000` the
mean stable rank of the hidden matrices is `15.2` under AdamW, `35.4` under
Muon (with or without decay) and `53.1` under Dervish, whose smallest
hidden singular values (`0.012` to `0.036`) are an order of magnitude above
Muon's (`0.0001` to `0.0018`); the orthonormality error of every frame is
`3.4e-6`. Second, cost: Dervish's training time is within `6%` of Muon's
(`326.7` versus `308.8` seconds; both measured with two threads while a
second matrix shared the CPU).

What this stage does not prove: that Dervish can match Muon's loss at any
spectrum learning rate (Stage D4 asks that), that the rank difference means
anything for behaviour (Stage D3 is the only behaviour measurement here),
that any of it holds beyond `0.8M` parameters and `1000` steps on one CPU
in float32, or that Muon's logit growth at this scale resembles a
`200M`-parameter model's. The Dervish initialisation projects the weights onto
the manifold, which moved its step-`0` loss from `3.6148` to `3.5876`; the
curves start there and the offset is in the arm's favour, not against it.

## Stage D2 · E2, learning-rate robustness

Date: 2026-10-09 · pre-registered in `docs/HYPOTHESIS-D001.md` (claim 3);
the E2b extension below is post-hoc.

Artifacts: `runs/e2_muon.jsonl`, `runs/e2_dervish.jsonl`,
`runs/e2_adamw.jsonl` (`6` reports each), the matching `runs/e2_*.md`,
`runs/e2b_dervish_low.jsonl` (post-hoc, `2` reports),
`figures/e2_lr_basins.png`, `runs/logs/e2.log`, `runs/logs/e2b.log`.

Command shape (one invocation per optimizer; `run_matrices.sh e2`):

```bash
python lab/compare.py --corpus lab/corpus/tiny_shakespeare_raw.txt --normalize --eval-interval 100 --eval-mode full --block-size 128 --batch-size 32 --n-layer 4 --n-head 4 --n-embd 128 --threads 2 --quiet --steps 600 --seeds 7 --configs muon --sweep-key muon_lr --sweep-values 0.005 0.01 0.02 0.04 0.08 0.16 --out runs/e2_muon.jsonl --summary runs/e2_muon.md --title "E2 Muon LR Sweep"
```

Seed `7`, `600` steps, six grid points at ratio `2` per optimizer. The
swept rate is the hidden-matrix rate for Muon (`muon_lr`) and Dervish
(`dervish_lr`, the rotation angle), and the single Adam rate for AdamW. All
other knobs as in Stage D1 (Dervish's spectrum rate `1e-3`).

| Optimizer | Grid point | Val NLL at `600` | Peak attention logit | Spectral max at `600` | In basin |
| --- | ---: | ---: | ---: | ---: | :---: |
| `muon` | `0.005` | `1.593666` | `15.27` | `2.28` | yes |
| `muon` | `0.01` | `1.577231` | `23.14` | `4.12` | yes (best) |
| `muon` | `0.02` | `1.597937` | `43.03` | `7.97` | yes |
| `muon` | `0.04` | `1.683983` | `389.35` | `17.04` | no |
| `muon` | `0.08` | `1.872226` | `3127.81` | `37.78` | no |
| `muon` | `0.16` | `2.025102` | `21100.23` | `85.28` | no |
| `dervish` | `0.0125` | `1.704168` | `10.39` | `0.66` | yes (best) |
| `dervish` | `0.025` | `1.760363` | `10.20` | `0.65` | yes |
| `dervish` | `0.05` | `1.856089` | `8.36` | `0.66` | no |
| `dervish` | `0.1` | `2.037375` | `8.46` | `0.62` | no |
| `dervish` | `0.2` | `2.193886` | `10.04` | `0.66` | no |
| `dervish` | `0.4` | `2.363965` | `8.96` | `0.74` | no |
| `adamw` | `0.00025` | `1.921047` | `14.26` | `1.28` | no |
| `adamw` | `0.0005` | `1.829015` | `18.95` | `1.53` | yes |
| `adamw` | `0.001` | `1.738072` | `23.37` | `2.52` | yes (best) |
| `adamw` | `0.002` | `1.772961` | `33.57` | `5.76` | yes |
| `adamw` | `0.004` | `1.835623` | `240.17` | `9.87` | yes |
| `adamw` | `0.008` | `2.004097` | `890.80` | `23.34` | no |

Verdict (`python lab/check_claims.py`): **claim 3, KILL.** Basin sizes on
the pre-registered rule (grid points within `0.100` NLL of the optimizer's
best): Muon `3/6`, Dervish `2/6`, AdamW `4/6`. Dervish's basin is smaller,
not larger. No run diverged to a non-finite loss.

Interpretation:

The grid was centred on the wrong default. Dervish's loss falls
monotonically as the angle shrinks, from `2.364` at `0.4` to `1.704` at
`0.0125`, and its best point is the smallest on the grid, so the basin is
truncated by the grid edge rather than by the optimizer. The angle `0.05`
fixed before the runs is at least `4x` too large. The verdict is kept as
computed, because that is what pre-registration is for; the post-hoc E2b
extension (`0.003125` and `0.00625`) is reported below so that the next
registration can centre the grid.

Why the angle matters more for Dervish than the rate matters for Muon is
the stage's real finding. A Muon step has a fixed spectral norm, so as the
weights grow (Stage D1: `0.68` to `10.4`) each step becomes a smaller
rotation of the weight; Muon's norm growth is an implicit learning-rate
decay. Dervish removes the growth and with it the free decay: a `0.05`
radian rotation at step `1000` is as large as at step `1`. The lab's answer
is either a smaller constant angle or an explicit schedule, which is what
Stage D5 tries.

The logit columns say what the basin rule does not. Past its basin, Muon
does not merely get worse, its attention logits reach `389`, `3128` and
`21100`; AdamW's reach `240` and `891`. Dervish's never exceed `10.4` at any
angle on the grid, by construction (Proposition 2), so its failure mode at
a bad rate is a slow model, not an exploding one. That is the property the
optimizer was built for, and it survives a `32x` range of rates.

What this stage does not prove: that Dervish's true basin is wider (E2b
and D5 address the low end); anything with more than one seed; that the
implicit-decay reading is the whole explanation of the slope (the spectrum
rate was still `1e-3` here, Stage D4's slow setting).

E2b, post-hoc extension of the Dervish grid downward (`./run_matrices.sh e2b`,
same seed, budget and spectrum rate as E2):

| Grid point | Val NLL at `600` | Peak attention logit | Spectral max at `600` |
| ---: | ---: | ---: | ---: |
| `0.003125` | `1.747590` | `8.82` | `0.655` |
| `0.00625` | `1.698934` | `9.09` | `0.631` |

The optimum at this spectrum rate is near `0.00625` (`1.698934`), and on the
extended eight-point grid four Dervish points lie within `0.100` of it
(`0.003125` to `0.025`, an `8x` range) against Muon's three (`0.005` to
`0.02`, a `4x` range) on its six. That is the comparison the next
registration should make, with a pre-committed grid; it is not a verdict on
claim 3, whose grid was fixed in advance and gave KILL.

## Stage D3 · E3, copy behaviour on the long-context copy corpus

Date: 2026-10-09 · exploratory in `docs/HYPOTHESIS-D001.md` (no pass line).

Artifacts: `runs/e3_copy.jsonl` (`24` reports), `runs/e3_copy.md`,
`figures/e3_copy_plain_curves.png`, `figures/e3_copy_weighted_curves.png`,
`runs/logs/e3.log`.

Command shape:

```bash
python lab/compare.py --corpus lab/corpus/long_context_seed.txt --eval-interval 100 --eval-mode full --block-size 96 --batch-size 32 --n-layer 4 --n-head 4 --n-embd 128 --threads 2 --quiet --steps 1000 --seeds 7 11 19 --copy-probe-marker "answer=" --configs adamw muon muon_wd dervish adamw_copyw muon_copyw muon_wd_copyw dervish_copyw --out runs/e3_copy.jsonl --summary runs/e3_copy.md --title "E3 Copy Behaviour Summary"
```

Corpus: `lab/corpus/long_context_seed.txt`, a synthetic copy task
(`512` lines, generator seed `20260617`; `38,921`
characters, `43` symbols, `8` keys, chance `0.125`); `51` probe cases in the
validation split. Model as in Stage D1 with block `96` (`804,395`
parameters). The `_copyw` arms add a copy-position loss weight `50` on
the character after `answer=`; the probe is the same for every arm.

| Arm | Mean val NLL | Copy accuracy at `1000` (mean) | Per seed | First evaluation at or above `0.25` (seeds `7 11 19`) | Peak attention logit (mean) | Spectral max at `1000` (mean) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adamw` | `0.251985` | `0.993464` | `1.000` `0.980` `1.000` | `200` `300` `300` | `25.29` | `3.082` |
| `muon` | `0.252694` | `1.000000` | `1.000` `1.000` `1.000` | `100` `100` `100` | `247.68` | `13.619` |
| `muon_wd` | `0.281399` | `0.993464` | `1.000` `0.980` `1.000` | `100` `100` `100` | `60.27` | `5.814` |
| `dervish` | `0.267591` | `0.888889` | `0.824` `0.843` `1.000` | `600` `400` `200` | `10.77` | `0.778` |
| `adamw_copyw` | `0.303530` | `1.000000` | `1.000` `1.000` `1.000` | `100` `100` `200` | `23.06` | `3.839` |
| `muon_copyw` | `0.242133` | `1.000000` | `1.000` `1.000` `1.000` | `100` `100` `100` | `433.02` | `15.906` |
| `muon_wd_copyw` | `0.247307` | `1.000000` | `1.000` `1.000` `1.000` | `100` `100` `100` | `104.70` | `7.322` |
| `dervish_copyw` | `0.327848` | `0.986928` | `1.000` `0.961` `1.000` | `100` `100` `100` | `11.32` | `0.888` |

Copy accuracy per evaluation, mean over seeds (full table in `runs/e3_copy.md`):

| Step | `adamw` | `muon` | `muon_wd` | `dervish` | `adamw_copyw` | `muon_copyw` | `dervish_copyw` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `100` | `0.124` | `0.497` | `0.562` | `0.163` | `0.366` | `0.993` | `0.856` |
| `200` | `0.216` | `0.993` | `0.987` | `0.222` | `0.987` | `0.967` | `0.908` |
| `300` | `0.608` | `1.000` | `1.000` | `0.255` | `0.980` | `1.000` | `0.941` |
| `500` | `0.987` | `0.967` | `0.967` | `0.137` | `1.000` | `1.000` | `1.000` |
| `700` | `0.987` | `0.987` | `1.000` | `0.902` | `1.000` | `1.000` | `0.895` |
| `1000` | `0.993` | `1.000` | `0.993` | `0.889` | `1.000` | `1.000` | `0.987` |

Interpretation:

Three things are measured here, and the first goes against the proposal.
Under plain next-character training, the copy behaviour forms in the order
Muon (`100` steps, with or without decay), AdamW (`200` to `300`), Dervish
(`200` to `600`, and only `0.89` at the end). Dervish with the slow
spectrum of Stage D1 is the *slowest* to form the behaviour, not the
fastest. The simplicity-bias intuition that keeping every direction alive
would make the copy circuit easier to find is not supported by this run;
the simpler explanation is that a sharp copy needs sharp attention, and the
starved spectrum keeps the attention logits at `10.8` when Muon's reach
`248`.

Second, under the copy-position weight, the ordering disappears:
every arm, Dervish included, passes twice chance at the first evaluation,
and Dervish ends at `0.987` (`1.000`, `0.961`, `1.000`) with its attention
logits bounded at `11.3`. Logits of that size are enough for a near-perfect
copy when the objective gives the behaviour direct credit. That is a useful
fact for the bound-first design: the cap does not forbid the behaviour, it
only stops being found for free.

Third, this corpus is where Muon's growth is most extreme: peak attention
logits of `247.68` without and `433.02` with the copy weight, hidden
spectral norms of `13.6` and `15.9`, with no divergence in `1000` steps.
Dervish's plain-training loss (`0.267591`) is within `+0.015` of Muon's
here, far closer than on natural text, because this corpus is nearly
deterministic and the loss floor is reached by every arm.

What this stage does not prove: anything about the simplicity bias beyond
one synthetic task at one model size; that Dervish would still be last with
a faster spectrum (Stage D4 changes exactly that knob, on natural text, and
the copy corpus was not rerun); that Muon's `433` logits would have
diverged with more steps. The `_copyw` validation NLLs are not comparable
to the plain ones, since the weighted objective spends capacity on the
answer positions (the expected cost of the weighting).

## Stage D4 · E4, the spectrum learning-rate sweep (post-hoc)

Date: 2026-10-09 · post-hoc: launched after Stage D1 was read, as the
follow-up the hypothesis named for a KILL on claim 2. Not pre-registered;
one seed; read as a diagnosis, not a result.

Artifacts: `runs/e4_scale_lr.jsonl` (`8` reports), `runs/e4_scale_lr.md`,
`runs/logs/e4.log`.

Command shape:

```bash
python lab/compare.py --corpus lab/corpus/tiny_shakespeare_raw.txt --normalize --eval-interval 100 --eval-mode full --block-size 128 --batch-size 32 --n-layer 4 --n-head 4 --n-embd 128 --threads 2 --quiet --steps 1000 --seeds 7 --configs dervish dervish_nocap --sweep-key dervish_scale_lr --sweep-values 0.003 0.01 0.03 0.1 --out runs/e4_scale_lr.jsonl --summary runs/e4_scale_lr.md --title "E4 Dervish Spectrum LR Sweep (post-hoc)"
```

Same model, corpus, budget and seed `7` as Stage D1; rotation angle `0.05`
as in Stage D1; only the spectrum learning rate changes.

| Arm | Spectrum lr | Val NLL | Spectral max at `1000` | Peak attention logit | Stable rank at `1000` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dervish` (cap `1.0`) | `0.001` (Stage D1) | `1.815227` | `0.802` | `12.01` | `51.9` |
| `dervish` | `0.003` | `1.792909` | `1.000` | `15.95` | `37.1` |
| `dervish` | `0.01` | `1.765969` | `1.000` | `20.89` | `41.2` |
| `dervish` | `0.03` | `1.777600` | `1.000` | `18.01` | `47.0` |
| `dervish` | `0.1` | `1.792898` | `1.000` | `18.14` | `49.3` |
| `dervish_nocap` | `0.003` | `1.781425` | `1.363` | `16.30` | `37.2` |
| `dervish_nocap` | `0.01` | `1.775106` | `2.975` | `25.65` | `25.3` |
| `dervish_nocap` | `0.03` | `1.758380` | `6.247` | `39.89` | `20.2` |
| `dervish_nocap` | `0.1` | `1.802167` | `12.973` | `87.51` | `17.8` |
| `muon` (Stage D1, seed `7`) | · | `1.538644` | `10.320` | `63.73` | `35.7` |

Interpretation:

A faster spectrum helps, by at most `0.057` NLL, and then stops helping.
With the cap, every spectrum learning rate from `0.003` up saturates the
cap by step `300` or earlier (`spectral max = 1.000` for the rest of the
run), so the cap becomes the binding constraint and the best capped loss is
`1.765969`, still `+0.227` above Muon. Without the cap the spectrum grows
freely, to `13.0` at the fastest rate, and the best uncapped loss is
`1.758380`, still `+0.220` above Muon. The uncapped arm also shows what the
cap buys: at spectrum lr `0.1`, the attention logit peak is `87.5` and the
stable rank has fallen to `17.8`, Muon's regime, while the capped arm at the
same rate holds `18.1` and `49.3`.

So the slow spectrum explains only a quarter of the Stage D1 gap. The rest
is not the cap either, since removing it changes the loss by `0.008`. The
remaining suspect is the rotation angle `0.05`, which Stage D2's sweep
tests directly; the reading there (the smallest angle on the grid is the
best, by a wide margin) is what sent the lab to Stage D5.

What this stage does not prove: anything with more than one seed; that the
`0.01` to `0.03` optimum is stable; that the cap's benefit on rank carries
any loss benefit at this scale (it did not here).

## Stages D5 to D7 · tuning what the pre-registered run got wrong (post-hoc)

Date: 2026-10-09 · post-hoc, launched after Stages D2 and D4 were read.
Seed `7` for D5; seeds `7 11 19` for D6 and D7. Not pre-registered: these
stages are the diagnosis the hypothesis asked for, and the numbers below
are what the next registration should be centred on, not a verdict.

Artifacts: `runs/e5_tuned.jsonl` and `.md` (`7` reports),
`runs/e6_replicate.jsonl` and `.md` (`3`), `runs/e7_decay_pair.jsonl` and
`.md` (`6`), `figures/e5_tuned_curves.png`,
`figures/e6_e7_replication.png`, `runs/logs/e5.log`, `e6.log`, `e7.log`.

Command shapes: `./run_matrices.sh e5`, then
`E6_ARGS="--optimizer dervish --dervish-lr 0.0125 --dervish-scale-lr 0.01 --dervish-cap 1.0" E6_LABEL=dervish_tuned ./run_matrices.sh e6`,
then `./run_matrices.sh e7`. Every run is one `lab/train.py` invocation
with the Stage D1 model, corpus and budget (`1000` steps), changing only
the knobs in the table. "Decay" is `--lr-decay-to 0.1`: every learning rate
of the run (hidden, spectrum, Adam) falls linearly to `10%` of its start by
the last step; it is applied to both optimizers in a pair.

### D5 · the knobs, one seed

| Arm | Rotation angle | Spectrum lr | Cap | Decay | Val NLL | Peak attention logit | Spectral max at `1000` | Stable rank |
| --- | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: |
| `dervish` (Stage D1) | `0.05` | `0.001` | `1.0` | no | `1.815227` | `12.01` | `0.802` | `51.9` |
| `dervish_a0125_s01_cap` | `0.0125` | `0.01` | `1.0` | no | `1.590136` | `15.66` | `1.000` | `60.6` |
| `dervish_a0125_s01_nocap` | `0.0125` | `0.01` | none | no | `1.589086` | `20.79` | `3.346` | `43.1` |
| `dervish_a00625_s01_cap` | `0.00625` | `0.01` | `1.0` | no | `1.571974` | `17.08` | `1.000` | `59.6` |
| `dervish_a00625_s01_nocap` | `0.00625` | `0.01` | none | no | `1.557936` | `21.51` | `2.747` | `44.3` |
| `dervish_a0125_s01_cap_decay` | `0.0125` | `0.01` | `1.0` | yes | `1.511704` | `15.64` | `1.000` | `55.9` |
| `muon` (Stage D1) | lr `0.02` | · | · | no | `1.538644` | `63.73` | `10.320` | `35.7` |
| `muon_wd` (Stage D1) | lr `0.02`, wd `0.1` | · | · | no | `1.582329` | `22.80` | `4.913` | `35.9` |
| `muon_lr01` | lr `0.01` | · | · | no | `1.527197` | `25.15` | `5.288` | `36.7` |
| `muon_lr02_decay` | lr `0.02` | · | · | yes | `1.491569` | `30.14` | `6.613` | `37.1` |

### D6 · the capped tuned arm, three seeds, beside Stage D1

| Arm | Seeds `7` · `11` · `19` | Mean val NLL | Mean bits/char | Peak attention logit (mean) | Spectral max at `1000` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `muon` (D1) | `1.538644` · `1.562646` · `1.554498` | `1.551929` | `2.238960` | `56.59` | `10.363` |
| `muon_wd` (D1) | `1.582329` · `1.579973` · `1.588875` | `1.583726` | `2.284833` | `23.31` | `4.938` |
| `dervish_tuned` (E6: angle `0.0125`, spectrum `0.01`, cap `1.0`) | `1.590136` · `1.603042` · `1.585442` | `1.592873` | `2.298031` | `16.34` | `1.000` |

The E6 seed-`7` row reproduces the E5 row to every printed digit
(`1.590136`), which is the lab's determinism check passing across two
separate launches.

### D7 · the decayed pair, three seeds

| Arm | Seeds `7` · `11` · `19` | Mean val NLL | Mean bits/char | Peak attention logit (mean) | Spectral max at `1000` (mean) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `muon_decay` (lr `0.02`, linear decay to `10%`) | `1.491569` · `1.500971` · `1.493113` | `1.495217` | `2.157143` | `30.72` | `6.583` |
| `dervish_tuned_decay` (angle `0.0125`, spectrum `0.01`, cap `1.0`, same decay) | `1.511704` · `1.514465` · `1.507320` | `1.511163` | `2.180148` | `16.58` | `1.000` |

Interpretation:

Two knobs fixed before the run were wrong, by about `10x` and `4x`, and
fixing them moves Dervish from `+0.271` behind Muon to a different regime:

- A faster spectrum (`0.01`) and a smaller angle (`0.0125`) together take
  seed `7` from `1.815` to `1.590`; halving the angle again gives `1.572`
  (`1.558` uncapped). Over three seeds the capped tuned arm (D6) averages
  `1.592873`: `+0.040944` behind undecayed Muon (inside the hypothesis's
  partial band) and `+0.009147` behind weight-decayed Muon (inside its pass
  band), with its peak attention logits at `16.3` against Muon's `56.6` and
  `muon_wd`'s `23.3`. Dervish with its cap holds its hidden spectral norm
  at exactly `1.000` for the whole run and keeps a stable rank of `60`
  against Muon's `35`.
- An explicit decay is worth about `0.08` to both optimizers at this
  budget, which supports the Stage D2 reading that Muon's norm growth was
  acting as a schedule: once both get a real one, the pair sits
  `+0.015946` apart over three seeds (`1.511163` versus `1.495217`),
  inside the `+0.030` line the hypothesis set for parity, with Dervish's
  peak attention logits at `16.6` against Muon's `30.7` and its stable rank
  at `54.6` against `36.8`. The E7 seed-`7` runs reproduce the E5 runs to
  every printed digit for both arms. The one reference not run is Muon at
  lr `0.01` with the same decay; at constant rate `0.01` was Muon's best
  (`1.527197` on seed `7`, against `1.538644` at `0.02`), so the gap to the
  best decayed Muon may be a little larger than `0.016`.
- The cap costs almost nothing in loss once the angle is right
  (`1.590` versus `1.589`, `1.572` versus `1.558`) and buys the bounded
  logits and the rank; the uncapped arm's spectral norm climbs to `2.7` to
  `3.3` and its stable rank falls into the forties.

What these stages do not prove: the pre-registered verdicts (Stage D1 and
D2 stand as KILL on their own lines; the tuned numbers are post-hoc and a
fresh registration with the grid centred here is the honest next step);
anything at larger scale, on CUDA, with bfloat16 Newton-Schulz, or on a
corpus other than Tiny Shakespeare; that the decay schedule chosen here
(linear to `10%`) is the right one for either optimizer; whether the
remaining gap to undecayed Muon closes with a longer budget.
