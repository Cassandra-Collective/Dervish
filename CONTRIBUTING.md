# Contributing to Dervish

Thank you for considering a contribution. Dervish is a small PyTorch optimizer
with a pre-registered evaluation record, and this guide explains how to work
on both without weakening the record.

- **Code of conduct:** this project follows the
  [Cassandra Collective code of conduct](https://github.com/Cassandra-Collective/.github/blob/main/CODE_OF_CONDUCT.md).
- **Security:** please do not open a public issue for a vulnerability. Report
  it privately through
  [this repository's advisory form](https://github.com/Cassandra-Collective/Dervish/security/advisories/new),
  as described in the
  [security policy](https://github.com/Cassandra-Collective/.github/blob/main/SECURITY.md).
- **Help and questions:** see the
  [support page](https://github.com/Cassandra-Collective/.github/blob/main/SUPPORT.md),
  or open an issue with the "Question" form.

## 1 · What Dervish is, and the evidence discipline

Dervish keeps every hidden weight matrix of a transformer in the factored form
`W = diag(s) Q`, with `Q` orthonormal and the spectrum `s` clamped, so the
spectral norm of every hidden matrix and every attention logit has a bound
computed from the live model (`docs/DESIGN.md`). The README states what it
costs: at `0.8M` parameters it finishes a few hundredths of a nat behind the
best Muon-family optimizer.

The repository is as much an evidence record as a package, and the record is
kept to four rules:

1. **Claims are registered before the runs that test them.** A hypothesis file
   fixes the arms, seeds and numeric pass lines, and is merged before any run
   (`docs/HYPOTHESIS-D001.md`, `docs/HYPOTHESIS-D002.md`).
2. **Raw run files are kept.** Every number in `docs/RESULTS.md` traces to a
   file under `runs/`.
3. **Verdicts are computed by script** from those files
   (`lab/check_claims.py`, `lab/check_d002.py`), and the registered rule is
   applied as written, even when the result is inconvenient.
4. **Every result says what it does not prove.** Each stage of
   `docs/RESULTS.md` ends with that statement.

Contributions that keep these rules are welcome, including the ones that
show Dervish losing.

## 2 · Getting set up

Requirements, from `pyproject.toml`: Python `3.11` or newer and PyTorch `2.1`
or newer. The tests and the lab also need pytest and matplotlib. CPU is
enough; CUDA is optional. The declared floor has not been run yet: the record
was produced on Python `3.13` with PyTorch `2.12` and `2.14` (README, "What
was verified"), and issue #2 asks for a CI matrix that covers the floor.

```bash
git clone https://github.com/Cassandra-Collective/Dervish.git
cd Dervish
python -m pip install -r requirements.txt   # torch, matplotlib, pytest
```

A virtual environment is a good idea. Create it outside the repository folder,
because `.gitignore` does not list one.

You do not need to install the package to work on it. `tests/conftest.py`, the
examples and the `lab/` scripts that import it put the repository root on
`sys.path`, so everything runs from a clone. To test the packaged form,
`pip install ".[test]"` installs `dervish/` with the test extras (README,
"Install").

Check that everything works:

```bash
python -m pytest -q tests     # the unit tests, under 20 seconds on a laptop CPU
./verify.sh                   # Windows, PowerShell 7.3 or newer: .\verify.ps1
```

`verify.sh` and `verify.ps1` run the tests, both examples, a 20-step training
smoke per optimizer and the D001 verdict line, in about two minutes on a
laptop CPU. The smokes print `val_nll` values of `2.8201` (AdamW), `2.5592`
(Muon) and `2.5726` (Dervish); those values were reproduced to every printed
digit on Linux and on Windows. The last two lines should read
`VERDICTS  claim 1: PASS   claim 2: KILL   claim 3: KILL` and `verify: ok`.

To recompute the recorded verdicts from the committed run files:

```bash
python lab/check_claims.py    # Hypothesis D001, from runs/
python lab/check_d002.py      # Hypothesis D002, from runs/d8/
```

`python lab/figures.py` redraws the figures from `runs/` into `figures/` and
overwrites the committed PNGs. Use `--out <folder>` to draw them elsewhere.

The full matrices (`./run_matrices.sh e1`, `d8` and so on) take hours. You
only need them to reproduce a stage, not to contribute.

## 3 · Finding something to work on

Every piece of planned work is an issue. Filter the
[open issues](https://github.com/Cassandra-Collective/Dervish/issues) by label:

| Label | What it marks |
| --- | --- |
| [`good first issue`](https://github.com/Cassandra-Collective/Dervish/labels/good%20first%20issue) | Small, well-scoped tasks with pointers into the code |
| [`help wanted`](https://github.com/Cassandra-Collective/Dervish/labels/help%20wanted) | Work the owner would welcome help with |
| [`experiment`](https://github.com/Cassandra-Collective/Dervish/labels/experiment) | A pre-registered experiment: the hypothesis and pass lines are merged before any run (section 4) |
| [`area: optimizer`](https://github.com/Cassandra-Collective/Dervish/labels/area%3A%20optimizer) | The `dervish/` package: optimizer, polar iterations, frame views |
| [`area: evaluation`](https://github.com/Cassandra-Collective/Dervish/labels/area%3A%20evaluation) | `lab/`, `runs/`, verdict scripts, hypotheses and results |
| [`area: packaging`](https://github.com/Cassandra-Collective/Dervish/labels/area%3A%20packaging) | Packaging, CI and the verify and run scripts |

Two milestones group the experiments:

- [Next lab-scale registration](https://github.com/Cassandra-Collective/Dervish/milestone/1):
  the tests the record says should come next at the current scale (for
  example a non-inferiority test with more seeds).
- [GPU-scale stage](https://github.com/Cassandra-Collective/Dervish/milestone/2):
  the larger stage that Hypothesis D001 names, and the GPU checks it needs.

Before you start, comment on the issue to say you are taking it, so that two
people do not do the same work. For anything larger than a typo, open an issue
first if none exists.

## 4 · Experiments: the pre-registration workflow

An experiment changes the evidence record, so it follows a fixed order. Issue
#14 is a worked example of a task written this way.

### 4.1 · Propose

Open an issue with the "Experiment proposal" form. State the claim, the arms,
the seeds, the pass lines, the baselines and the compute, and what the result
would not prove. Discuss it there before writing the hypothesis file.

### 4.2 · Register

Write `docs/HYPOTHESIS-D00N.md`, taking the next free number (D003 at the time
of writing; if two pull requests collide, the later one renumbers). Follow the
structure of `docs/HYPOTHESIS-D001.md` and `docs/HYPOTHESIS-D002.md`:

- a title of the form `Hypothesis D00N · <claim>`;
- header lines: a status line (`REGISTERED <date>, before any run in this
  file`), the date, the author, and what it builds on;
- why the test is worth running, and the hypothesis itself;
- the arms, as a table, with every setting that differs between them;
- the command shapes that will be run;
- the selection rule, if any arm is tuned: every family gets the same number
  of looks, on a selection seed that is not an evaluation seed (Stage D8a);
- the evaluation seeds, which must be fresh. Already used: `7 11 19`
  (D001 and the post-hoc Stages D4 to D7) and `7` plus `23 29 31 37 41` (D002);
- numeric pass lines for every question: the metric, the step, how seeds are
  aggregated, the interval used, and the outcome names (PASS, PARTIAL and
  KILL, or REPLICATED, NOT REPLICATED and REVERSED). Write the line the way it
  will be applied: on the mean, or on the upper end of the paired interval.
  Stage D8 recommends the stricter non-inferiority form (the upper end of the
  interval below the margin) for the next registration;
- risks and confounds, what each outcome would change, and what the test
  cannot say.

The registration pull request contains the hypothesis file and any code the
arms need (new options, launchers, their tests). It contains no run files. It
is merged before the first run.

After registration, the file is not edited. When the runs are done, one
resolution line is added at the top. If anything else ever has to change, add a
dated note at the top that says what changed and confirms that no arm, seed or
line did (see the publication note in `docs/HYPOTHESIS-D001.md`).

### 4.3 · Run

- Use the lab (`lab/train.py`, `lab/compare.py`, or a launcher in the style of
  `lab/run_d8.py`) and add the stage to `run_matrices.sh` and, where possible,
  `run_matrices.ps1`.
- Keep one device per comparison. As Stage D8 notes, CPU and CUDA runs of the
  same seed do not agree, so mixed-device arms are not comparable. The lab-scale
  record is CPU float32; keep it so unless the hypothesis says otherwise.
- On a machine that has not produced part of the record before, first
  reproduce a recorded seed and compare it with `docs/RESULTS.md`. Stage D8a
  agreed within `0.003` against a registered tolerance of `0.01`.
- Run every registered run, and report runs that diverged, failed or were
  restarted. Do not drop or rerun a run because of its result.

### 4.4 · Record

- **Raw run files:** the `*.jsonl` reports (one JSON report per line, as
  `lab/train.py --out` writes them), their `*.md` summaries and the launcher
  logs go under `runs/`, as `runs/d8/` does for Stage D8. They are committed as
  produced and never edited by hand.
- **Verdict script:** `lab/check_<id>.py` reads the run files, prints every
  number it uses and applies the registered lines mechanically, as
  `lab/check_d002.py` does. It may not encode anything the hypothesis file
  does not say. Writing it before the runs, in the registration pull request,
  is best. Save its output beside the runs (for example `runs/d8/verdicts.txt`).
- **A `docs/RESULTS.md` stage,** `## Stage D9 · <title>` for the next one,
  with: the date and the hypothesis it was registered in; the artifacts; the
  command shape; the platform (OS, CPU, Python and PyTorch versions, threads,
  device); the tables; the verdicts as printed by the script; the
  interpretation; and a closing list, **What this stage does not prove**.
- Update the README's evidence summary and the hypothesis file's resolution
  line, and add a figure with `lab/figures.py` if it helps.

Analyses that were not registered are welcome, but they are labelled post-hoc
and carry no verdict, as Stages D4 to D7 do. They can motivate the next
registration; they cannot confirm a claim.

## 5 · Code changes

- **Keep the invariant.** Every frame is `W = diag(s) Q` (transposed for tall
  matrices) with `Q` orthonormal and `s` clamped to
  `[scale_floor, scale_cap]`, so the singular values of `W` are `s` and the
  cap is a cap on the spectral norm (`docs/DESIGN.md`, sections 2 and 4). The tests that guard it are
  `test_dervish_keeps_the_factored_form_for_every_shape` and
  `test_spectral_cap_bounds_the_spectral_norm_under_pressure` in
  `tests/test_optimizer.py`, and `test_dervish_with_mona_keeps_the_manifold_and_the_cap`
  in `tests/test_mona.py`. A change to `dervish/optimizer.py`, `polar.py` or
  `matricize.py` must keep them passing, and a new frame option or shape rule
  needs its own check of the invariant.
- **Keep the comparison fair.** Dervish and the reference Muon train every
  non-frame parameter with the same auxiliary Adam (`_adam_group_step`), so
  that comparing them changes one thing. The MONA term is off when
  `mona_beta` is `None`, and with `mona_alpha = 0` it reproduces plain Muon
  bit for bit (`test_alpha_zero_is_plain_muon_bit_for_bit`). Keep these
  properties.
- **Test at the right level.** The polar iterations are tested against an
  exact reference, the SVD (`tests/test_polar.py`); the frame views for their
  shape rules, round trips and loud failures (`tests/test_matricize.py`);
  optimizer behaviour on small seeded models (`tests/test_optimizer.py`,
  `tests/test_mona.py`); the lab harness itself in `tests/test_lab.py`. Tests
  run on CPU, seed any random draw with `torch.manual_seed`, and finish in
  seconds. A test that needs a GPU should skip when CUDA is not available.
- **Type hints and contracts.** Functions in `dervish/` carry type hints; new
  code should too, with a docstring that states inputs, outputs, side effects
  and the invariants it relies on.
- **Dependencies.** The package depends on PyTorch only. A new runtime
  dependency needs an issue first.
- **Trajectory changes.** If a change alters the numbers that `verify.sh`
  prints for the 20-step smokes, say so in the pull request and explain why.
  The recorded runs are not rerun or edited to match; they describe the code
  that produced them.
- **Defaults and documented numbers are decisions.** Changing a default (for
  example `FRAME_DEFAULTS`, issue #6) or a number stated in the documents
  needs an issue first. Keep the README, `docs/DESIGN.md`, the docstrings and
  `examples/trainer_integration.md` consistent with each other.
- **Scripts come in pairs.** `verify.sh` and `verify.ps1`, and
  `run_matrices.sh` and `run_matrices.ps1`, should change together where the
  stage exists in both.

## 6 · Implementation documents

Every meaningful change, including an experiment stage, adds one note:
`docs/implementations/YYYY-MM-DD-<slug>.md`, copied from
`docs/implementations/_TEMPLATE.md`. It has a header (date, author, commits)
and seven numbered sections:

1. Problem / motivation
2. What changed (a table with one sentence per file)
3. Implementation approach
4. Mathematical / statistical details (write "None" only for structural changes)
5. Design decisions (the alternatives, and why this one won)
6. Verification (commands someone else can run)
7. Related docs

A typo fix or a one-line correction does not need one.

## 7 · Writing style

- **Middle dot as the separator** in titles and headings, and between listed
  values: `Stage D8 · Dervish against MONA on fresh seeds`.
- **No em dashes or en dashes,** in documents, code comments, commit messages
  and pull request text. Use a comma, a colon, parentheses or a new sentence,
  and "to" for ranges (`Stages D5 to D7`). The plain hyphen is fine in
  compound words and negative numbers. To check before you push:

  ```bash
  git grep --untracked -nP "\x{2013}|\x{2014}"   # prints nothing when the tree is clean
  ```

- **Numbers, seeds, paths and identifiers** in backticks, as in
  `docs/RESULTS.md`.
- **State what was measured, on what,** and label inference as inference.
  Explain why, not only what.

## 8 · Pull requests

1. **An issue first,** for anything beyond a typo. Comment on it before you
   start.
2. **A branch from `main`,** named `<kind>/<issue number>-<slug>`, for example
   `test/4-checkpoint-resume`. Contributors without write access work on a
   fork.
3. **Commits** with a short imperative summary line and the reason in the body.
4. **A pull request** that says `Closes #<issue>` and fills in the template,
   including the evidence: test or verify output, and for an experiment the
   link to the merged hypothesis.
5. **Review and merge** are done by the owner.

## 9 · Licence

Dervish is released under the MIT licence (`LICENSE`). By opening a pull
request, you agree that your contribution is released under the same licence.
