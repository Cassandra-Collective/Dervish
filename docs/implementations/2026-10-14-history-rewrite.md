# History rewrite: one author, no personal or private-project content

- Date: 2026-10-14
- Author: Mert Efe Şensoy
- Commits: every commit up to `0f93e1f` was rewritten; the result was
  force-pushed over the previous `main` (`95aecd1`).

## 1 · Problem / motivation

The repository is meant to become public. Its history had four problems.

1. **Authorship.** Eight of the nine commits had an AI coding assistant as
   author and committer, with a co-author trailer. The first commit had a
   web-UI committer. The owner wanted every commit authored by them alone.
2. **Personal information.** A personal dedication, a folder name and a
   top-level heading together revealed personal information about the owner.
3. **Session links.** Every commit message linked to a private assistant
   session.
4. **Private-project references.** Documentation and comments named the
   owner's private projects, their stage numbers, file paths and model sizes.
   One reference concerned a collaborator's project. These mean nothing to an
   outside reader and were not the owner's to publish.

Editing only the latest commit would not fix any of this: every old version
stays readable in history. The history itself had to change.

## 2 · What changed

| Scope | Change |
| --- | --- |
| Every commit | Author and committer set to `Mert Efe Şensoy <143845647+mertefesensoy@users.noreply.github.com>`. Co-author trailers and session links removed from messages; two subject lines reworded. |
| Every tree | The project folder moved to the repository root; the integration note renamed to `examples/trainer_integration.md`. |
| `README.md`, `docs/*.md`, code comments and docstrings, `verify.*` | In every historical version: the dedication and second-person framing removed, and private-project names, stage numbers, paths and sizes replaced by neutral wording ("a reference Muon trainer", "the usual Muon recipe"). |
| `examples/trainer_integration.md` | Rewritten as a generic guide for adding Dervish to an existing trainer. |
| `lab/corpus/long_context_seed.txt` | One word in the header line replaced by `synthetic`, a word of the same length from the same alphabet. |
| `tests/test_lab.py` | One test function renamed. |
| `docs/HYPOTHESIS-D001.md` | Its author line now names the owner. A note added on 2026-10-14 says the registration's wording was neutralised and that no hypothesis, arm, seed, threshold or pass line changed. |
| `docs/RESULTS.md` Stage D3 | A note added on 2026-10-14 discloses the corpus header change. |

The Tiny Shakespeare corpus, the figures (binary) and the run files (`*.jsonl`)
were not modified.

## 3 · Implementation approach

The rewrite used `git-filter-repo` (2.47, run through its Python API) on a fresh
clone, with four callbacks:

- **`commit_callback`** sets one identity on every commit and maps each file
  path: it strips the old folder prefix and renames the integration note.
- **`blob_callback`** applies an ordered list of regular-expression rules to
  every UTF-8 blob, specific multi-line rules first and generic fallbacks last.
  It skips binary blobs and the Tiny Shakespeare corpus, which is recognised by
  its first line.
- **`message_callback`** removes the trailer and session-link lines and
  rewrites two subjects.
- The integration note is replaced whole, because nothing in it was generic.

The rules were tested before the rewrite. A script applied them to all `122`
distinct historical blobs and scanned the output for a list of markers: names,
private terms, dedication words, session identifiers and paths. The only
remaining hits were Shakespeare's own uses of common words. The rewritten tree
then passed the full test suite (`24` tests) before the push.

The push used `git push --force-with-lease=main:<old tip>`, so it could not
overwrite a commit made by anyone else after the backup was taken.

The rules script is kept outside the repository. Committing it would publish
the strings it removes.

## 4 · Mathematical / statistical details

The only numeric concern is the corpus edit. The header word is replaced by a
word of exactly the same length, so the corpus length `N` is unchanged and
every character keeps its position. The train/validation split is at
`max(block + 2, floor(0.9 N))` (`lab/data.py: split_at_index`), so it is
unchanged too. The vocabulary is the set of distinct characters; every letter
of the new word already occurs elsewhere in the corpus, so the vocabulary and
its index order are unchanged.

What does change is nine token ids in the first line of the training split. A
training window covers those positions only when its random start falls in the
first line. The E3 numbers were produced with the original line, so a rerun can
differ in the last digits. Stage D3 says so.

## 5 · Design decisions

- **`git-filter-repo` over `git filter-branch` or a squash.** `filter-branch`
  is deprecated and slow. A squash would have destroyed the staged history
  that the evidence record refers to ("the E6 run reproduced the E5 run"). The
  `filter-repo` callbacks keep every commit, date and structure, and change
  only what was asked.
- **GitHub noreply address over the personal email.** Commits still link to the
  owner's GitHub profile, and the personal address never enters a public
  history.
- **Rewriting contents, not only metadata.** With metadata alone, the
  dedication and the private references would stay one `git checkout` away.
  The owner chose the full clean.
- **The same rules for every version.** Rules keyed on text, not on commits,
  give every historical version of a file the same treatment. The rewritten
  `HEAD` is the rules applied to the old `HEAD`, which keeps later work easy to
  port.
- **Disclosure notes.** Two documents that make claims about what was or was
  not edited (the D001 registration and the E3 corpus) got dated notes rather
  than silent changes.

## 6 · Verification

```bash
git log --all --format='%an <%ae> | %cn <%ce>' | sort | uniq -c   # one identity on every commit
git log --all --format=%B | grep -ciE 'co-authored|session'        # 0
python -m pytest -q tests                                           # all tests pass
```

To check for residual private terms, run `git log --all -p` through a grep for
the terms you care about. Expect hits only inside
`lab/corpus/tiny_shakespeare_raw.txt`.

GitHub serves unreachable commits by their old hash until it garbage-collects
them. The repository was private throughout. Before it is made public, ask
GitHub Support to purge the cached objects, or publish from a freshly created
repository.

## 7 · Related docs

- `docs/HYPOTHESIS-D001.md` (edit note), `docs/RESULTS.md` Stage D3 (corpus note)
- `docs/implementations/2026-10-14-mona-comparison-and-release-prep.md`
