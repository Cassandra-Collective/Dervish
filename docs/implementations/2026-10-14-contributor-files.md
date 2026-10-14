# Contributor files: a contributing guide, issue forms and a pull request template

- Date: 2026-10-14
- Author: Mert Efe Şensoy
- Commits: see the pull request `docs/contributor-files`

## 1 · Problem / motivation

Dervish is public and has a backlog of 19 issues (#2 to #20) labelled for outside help, with two milestones ("Next lab-scale registration" and "GPU-scale stage"). On 2026-10-14 the owner asked for a contributing guide, a code of conduct and templates in every repository of the organisation, to attract open-source contributors.

The repository had none of them. Its evidence discipline existed only implicitly, spread over the two hypothesis files, `docs/RESULTS.md`, the implementation notes and the text of issue #14. That is the main risk for Dervish in particular: its value is the pre-registered record, and a well-meant contribution that runs an experiment before registering it, edits a run file, drops an inconvenient run or reports a verdict that no script computed would damage the record in a way that is hard to undo. The guide writes the discipline down where a new contributor will find it, and the forms ask for it at the moment a contribution is proposed.

## 2 · What changed

| File | Change |
| --- | --- |
| `CONTRIBUTING.md` | New guide: what Dervish is and its four evidence rules, setup and checks, where to find work, the pre-registration workflow for experiments, rules for code changes, implementation documents, writing style, the pull request flow, the licence, and links to the organisation's code of conduct, security policy and support page. |
| `.github/ISSUE_TEMPLATE/bug_report.yml` | Bug report form, label `bug`: separate required fields for the Dervish version or commit, Python, PyTorch, operating system, device and a minimal reproduction. |
| `.github/ISSUE_TEMPLATE/feature_request.yml` | Feature or improvement form, label `enhancement`: problem, proposal, area, and whether the change moves training trajectories or recorded numbers. |
| `.github/ISSUE_TEMPLATE/experiment_proposal.yml` | Experiment proposal form, labels `experiment` and `area: evaluation`: hypothesis, arms, baselines and selection rule, pass lines, seeds, compute, what it would not prove, and three required evidence-discipline checkboxes. |
| `.github/ISSUE_TEMPLATE/question.yml` | Question form, label `question`, pointing at the documents and the support page first. |
| `.github/ISSUE_TEMPLATE/config.yml` | Turns blank issues off and adds four contact links: the private advisory form, the code of conduct, the support page and the contributing guide. |
| `.github/pull_request_template.md` | Pull request template: the issue closed, what changes and why, evidence, a checklist (tests, invariant, documents, implementation document, no em or en dashes) and an experiment checklist (hypothesis merged before the runs, raw files, scripted verdicts, a RESULTS stage that says what it does not prove). |
| `README.md` | A short "Contributing" section between "Reading order" and "License". The header at the top is unchanged. |
| `docs/implementations/2026-10-14-contributor-files.md` | This document. |

## 3 · Implementation approach

**Rules drawn from the record, not invented.** Each rule in the guide's sections 1 and 4 restates something the repository already does, so that the guide describes the existing practice rather than a new one:

| Rule in `CONTRIBUTING.md` | Where the repository already does it |
| --- | --- |
| A hypothesis file with numeric lines, merged before any run; a status line on top; only a resolution line added afterwards | `docs/HYPOTHESIS-D002.md` header ("REGISTERED 2026-10-14, before any run in this file", "will not be edited after the runs, except to add the status line") |
| A dated note for any later edit, confirming no arm, seed or line changed | The publication note in `docs/HYPOTHESIS-D001.md` |
| Equal tuning looks on a selection seed that is not an evaluation seed | Stage D8a in `docs/HYPOTHESIS-D002.md` and `docs/RESULTS.md` |
| Fresh evaluation seeds; the list of used seeds | D001 (`7 11 19`), D002 (`7`, then `23 29 31 37 41`) |
| Reproduce a recorded seed on a new machine first | The Stage D8a machine check (within `0.003`, line `0.01`) |
| One device per comparison | The Stage D8 platform paragraph of `docs/RESULTS.md` |
| A verdict script that prints every number it uses and applies the lines mechanically | `lab/check_d002.py`, and the acceptance criteria of issue #14 |
| A RESULTS stage that ends with what it does not prove | Every stage of `docs/RESULTS.md` |
| Post-hoc analyses carry no verdict | Stages D4 to D7 |
| The registered rule is applied as written | Section 5 of `docs/implementations/2026-10-14-mona-comparison-and-release-prep.md` |
| The non-inferiority form for the next registration | The Stage D8 interpretation, and issue #14 |

**Code rules from the code.** The invariant and the tests that guard it were read from `dervish/optimizer.py` and `tests/`: `W = diag(s) Q` with `s` clamped to `[scale_floor, scale_cap]`; `test_dervish_keeps_the_factored_form_for_every_shape`, `test_spectral_cap_bounds_the_spectral_norm_under_pressure` and `test_dervish_with_mona_keeps_the_manifold_and_the_cap`; the shared auxiliary Adam (`_adam_group_step`, called by both `MuonWithAuxAdam` and `Dervish`); the MONA gate (`mona_beta=None` is off; `test_alpha_zero_is_plain_muon_bit_for_bit` checks that `alpha = 0` is plain Muon); and the single runtime dependency in `pyproject.toml`.

**Commands checked against the files and run.** Every command in the guide's setup section was run on this branch's base (section 6). Claims that could not be run here are attributed to the README instead of being restated as checked: the `pip install ".[test]"` route, and the Linux reproduction of the smoke values.

**Issue forms, not Markdown templates.** GitHub's YAML issue forms enforce required fields, which is the point for the experiment form: a proposal cannot be filed without pass lines, seeds and a statement of what it would not prove. Each form applies only labels that exist in this repository.

**Organisation files linked, not copied.** The code of conduct, security policy and support page are linked by absolute URL into `Cassandra-Collective/.github`.

## 4 · Mathematical / statistical details

None: documentation and repository metadata only. The guide restates two statistical rules from Stage D8 in words (paired differences by seed with a 95% interval, and the non-inferiority form that puts the upper end of the interval below the margin), but adds no formula. The formulas are in section 4 of `docs/implementations/2026-10-14-mona-comparison-and-release-prep.md`.

## 5 · Design decisions

**The owner's decisions of 2026-10-14.**

- The code of conduct, security policy and support page are organisation defaults in `Cassandra-Collective/.github`. This repository links to them and does not copy them: one source of truth, and a copy here would override the default and drift from it.
- This repository gets its own `CONTRIBUTING.md`, issue forms and pull request template, because those are specific to Dervish: the pre-registration workflow, the spectral invariant and this repository's labels.
- Private vulnerability reporting is enabled on this repository, so the security contact link and the guide point at its advisory form.
- Consequence: the three organisation links return 404 until the `.github` community-health pull request is merged. That pull request should be merged first.

**Choices made here, which the owner may change.**

- **`blank_issues_enabled: false`.** It follows the precedent of the CASSANDRA-Chorus repository and makes every outside issue go through a form. Issues created with `gh issue create --body` or through the API do not go through the forms, so the owner's own backlog work is unaffected. Setting it to `true` would lower the barrier for one-line reports at the cost of structure.
- **Labels.** The forms apply `bug`, `enhancement`, `question`, and `experiment` with `area: evaluation` (every open experiment issue carries both). Area labels for bugs and features are not applied automatically, because a form's dropdown cannot set a label; triage sets them.
- **The experiment form requires three discipline checkboxes.** They make the order (register, run, record) explicit at the moment of proposing, which is when it is cheapest to follow.
- **The feature form asks whether trajectories move.** A change that alters the 20-step smoke values silently desynchronises `verify.sh` from the README and the record, so the question is asked up front.
- **The verdict script "best" written before the runs, not required.** That is stricter than the D002 precedent, where `lab/check_d002.py` arrived with the run files. Making it a requirement would retroactively fault D002, so the guide recommends it and requires only that the script encode nothing the hypothesis does not say.
- **The non-inferiority form is recommended, not mandated.** Stage D8 recommends it for the next registration; whether a given hypothesis uses it is that hypothesis's decision (issue #14 proposes it).
- **Branch names `<kind>/<issue number>-<slug>`,** borrowed from the CASSANDRA-Chorus guide for consistency across the organisation.
- **Licence: inbound equals outbound, no contributor licence agreement.** The lowest-friction option compatible with MIT.
- **Next numbers stated (D003, Stage D9), with renumbering on collision,** so that two concurrent proposals do not both claim D003.
- **README kept to one new section.** The layout table and the organisation header are untouched.

## 6 · Verification

Platform: Windows 11 (build `26300`), Git Bash, PowerShell `7.6.6`, Python `3.13.14` with PyTorch `2.12.1+cu126` on CPU, PyYAML `6.0.3`. The interpreter at `AppData/Local/Programs/Python/Python312` has no torch, pytest or PyYAML, so the checks used `python`, which resolves to the `3.13.14` install; `verify.sh` and `verify.ps1` call the same `python`.

Run on the base commit `3ef1e04` before any file was changed:

- `./verify.sh`: exit `0` in `2 min 19 s`; `29 passed`; smokes `2.8201`, `2.5592`, `2.5726`; `VERDICTS  claim 1: PASS   claim 2: KILL   claim 3: KILL`; `verify: ok`. The package was not installed (`importlib.util.find_spec("dervish")` is `None` outside the clone), which confirms the guide's statement that everything runs from a clone.
- `pwsh -File ./verify.ps1`: exit `0` in `127 s`, with the same test count, smoke values and verdict line.
- `python lab/check_claims.py` and `python lab/check_d002.py`: exit `0`, with the recorded verdicts.
- `python lab/figures.py --out <scratch folder>`: wrote the eight PNGs outside the repository; the default `--out` is `figures` (read in `lab/figures.py`).
- `python -m pip install -r requirements.txt --dry-run`: resolves (every requirement already satisfied on this machine).
- `git grep -nP "\x{2013}|\x{2014}"` over the whole tree prints nothing, which confirms that `git grep -P` works with Git for Windows. The guide uses `git grep --untracked`, because plain `git grep` skips files that are not yet added.

On the new files:

- Each issue form and `config.yml` was parsed with PyYAML and checked by a script against the issue-form rules: allowed top-level keys; `name`, `description` and `body` present; every label a string that exists in `gh label list`; a known `type` for every element; `markdown` elements with only `attributes.value`; a unique `id` matching `[A-Za-z0-9_-]+` on every other element; a non-empty `label`; `validations` only as `{required: bool}` and never on checkboxes; dropdown options as unique non-empty strings (so an unquoted `Yes` or `No`, which YAML reads as a boolean, would fail); at least one input element per form; contact links with exactly `name`, `url` and `about`; no byte-order mark, no CRLF, and no U+2012 to U+2015 dashes. All five files passed. The same script flagged every fault in a deliberately broken form (an unquoted `area: evaluation` label, an unknown label, an `id` on a markdown element, unquoted `Yes`/`No` options, a duplicated `id`, `validations` on checkboxes).
- Links: the issue list, the six label pages and both milestone pages return HTTP `200`. The three `Cassandra-Collective/.github` links return `404` until that repository's pull request is merged. Private vulnerability reporting reads `enabled: true` through the API.
- Every file this change adds or edits is free of dashes U+2012 to U+2015 (`git grep --untracked -nP` over the working tree before committing), has LF line endings and no byte-order mark.

Not verified here: how the forms render. GitHub serves issue forms and the pull request template from the default branch only. After merging, open "New issue" and check that the chooser shows four forms and four contact links, and open a draft pull request to see the template.

To repeat the checks:

```bash
git grep --untracked -nP "\x{2013}|\x{2014}"     # prints nothing
python -c "import yaml,glob; [yaml.safe_load(open(f, encoding='utf-8')) for f in glob.glob('.github/ISSUE_TEMPLATE/*.yml')]; print('parsed')"
gh label list -R Cassandra-Collective/Dervish     # every label used by the forms is listed
./verify.sh                                       # unchanged by this documentation-only change
```

## 7 · Related docs

- `CONTRIBUTING.md`
- `docs/HYPOTHESIS-D001.md`, `docs/HYPOTHESIS-D002.md`, `docs/RESULTS.md` (Stage D8)
- `docs/implementations/2026-10-14-mona-comparison-and-release-prep.md`
- Organisation files (live once the `.github` pull request is merged): [code of conduct](https://github.com/Cassandra-Collective/.github/blob/main/CODE_OF_CONDUCT.md), [security policy](https://github.com/Cassandra-Collective/.github/blob/main/SECURITY.md), [support](https://github.com/Cassandra-Collective/.github/blob/main/SUPPORT.md)
- GitHub's issue form syntax: <https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/syntax-for-issue-forms>
