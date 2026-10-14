Closes #

## What changes, and why

<!-- One paragraph: what this pull request changes and the reason. Link the implementation document. -->

## Evidence

<!--
Paste what you ran and what it printed. For example:
- `python -m pytest -q tests` (the summary line)
- `./verify.sh` or `.\verify.ps1` (the three smoke lines, the VERDICTS line and "verify: ok")
- for a trajectory change, the old and new smoke values and the reason they moved
State the platform: OS, Python and PyTorch versions, device.
-->

## Checklist

- [ ] The linked issue exists and its acceptance criteria are met, or the gaps are listed under "Not covered".
- [ ] Tests pass, and new behaviour has a test at the right level. The invariant `W = diag(s) Q` with `s` in `[scale_floor, scale_cap]` still holds.
- [ ] The documents that describe the change are updated (README, `docs/DESIGN.md`, docstrings, `examples/`), and an implementation document is added in `docs/implementations/` from `_TEMPLATE.md`.
- [ ] No em dashes or en dashes in any file, commit message or this description (`git grep --untracked -nP "\x{2013}|\x{2014}"` prints nothing).
- [ ] No tokens, keys or personal data.

### For an experiment (delete if not one)

- [ ] The hypothesis file was merged before the first run: <!-- link the merged pull request or commit -->
- [ ] Every registered run is reported, and the raw run files are committed as produced under `runs/`.
- [ ] The verdicts come from a script that applies the registered lines as written; its output is committed.
- [ ] The `docs/RESULTS.md` stage states the platform and what the result does not prove, and the hypothesis file gained only its resolution line.

## Not covered

<!-- What this pull request deliberately leaves out. -->
