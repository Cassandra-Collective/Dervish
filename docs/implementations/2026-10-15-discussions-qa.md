# Questions go to Discussions Q&A

- Date: 2026-10-15
- Author: Mert Efe Şensoy
- Commits: see the pull request `docs/discussions-qa`

## 1 · Problem / motivation

On 2026-10-15 the owner turned on GitHub Discussions in every repository of the organisation. This repository now has the default categories, including **Q&A**, the one category whose threads can have an answer marked as accepted: <https://github.com/Cassandra-Collective/Dervish/discussions/categories/q-a>.

Nothing pointed there. The "Support" link on the new-issue page led to the organisation support page, and the top of `CONTRIBUTING.md` offered the support page or the "Question" issue form. A question that only needs an answer therefore became an issue that someone had to label and close, and its answer was harder to find again.

## 2 · What changed

| File | Change |
| --- | --- |
| `.github/ISSUE_TEMPLATE/config.yml` | The "Support" contact link becomes "Ask a question", pointing at this repository's Discussions Q&A, with an `about` line saying open questions and help go there. The security, code of conduct and contributing guide links and `blank_issues_enabled: false` are unchanged. |
| `CONTRIBUTING.md` | The "Help and questions" item at the top points first at Discussions Q&A, keeps the "Question" form for a question that should be tracked as an issue, and still links the support page. |
| `docs/implementations/2026-10-15-discussions-qa.md` | This note. |

## 3 · Implementation approach

Two text edits, no code. In `config.yml` the third of the four contact links is replaced in place, so the order on the new-issue page stays: security report, code of conduct, questions, contributing guide. In `CONTRIBUTING.md` only the questions item changes.

The guide now gives two routes with one rule for choosing: an open question goes to Discussions Q&A; a question that should be tracked, for example one whose answer will need a change to the code, the documents or the record, goes through the "Question" form, which applies the `question` label. The form itself (`.github/ISSUE_TEMPLATE/question.yml`) is not changed.

## 4 · Mathematical / statistical details

None: links and wording only.

## 5 · Design decisions

- **Replace the support link rather than add a fifth one.** The support page is organisation-wide and mostly says where to ask; with Q&A on, the direct route is more useful on the new-issue page. `CONTRIBUTING.md` still links the support page for what to expect.
- **Keep the "Question" form.** Some questions are bug reports, documentation gaps or doubts about a recorded number in disguise, and those belong in the tracker with a label. With blank issues off, the form is also the only way to turn a question into an issue.
- **Link the Q&A category, not the Discussions front page.** The category URL opens the list of Q&A threads, and a discussion started there can have an answer marked.
- **No issue.** The change applies an owner decision of 2026-10-15 to repository metadata, as `docs/contributor-files` (#21) did for the owner's decisions of 2026-10-14. Section 8 of `CONTRIBUTING.md` asks contributors for an issue first; the owner may want one for this change too.

## 6 · Verification

Platform: Windows 11, Git Bash, Python `3.13.14` with PyYAML `6.0.3`. Run from the repository root on 2026-10-15:

- `config.yml` parses and keeps its keys:

  ```bash
  python -c "import yaml; d = yaml.safe_load(open('.github/ISSUE_TEMPLATE/config.yml', encoding='utf-8')); print(sorted(d), d['blank_issues_enabled'], [(l['name'], sorted(l)) for l in d['contact_links']])"
  ```

  prints the keys `blank_issues_enabled` and `contact_links`, `False`, and four links (`Report a security vulnerability`, `Code of conduct`, `Ask a question`, `Contributing guide`), each with exactly `about`, `name` and `url`.
- The Q&A link resolves: `curl -s -o /dev/null -w "%{http_code}" https://github.com/Cassandra-Collective/Dervish/discussions/categories/q-a` prints `200` signed out.
- `git grep --untracked -nP "\x{2013}|\x{2014}"` prints nothing over the whole working tree. The commit message and the pull request text were checked for U+2013 and U+2014 too.
- `git diff --stat main` lists only the three files in the table above.
- No code, test, script or recorded result changed, so `./verify.sh` was not rerun.

Not verified here: how the new-issue page renders. GitHub reads `config.yml` from the default branch only. After merging, "New issue" should show the four forms and four contact links, with "Ask a question" third.

## 7 · Related docs

- `CONTRIBUTING.md`
- `docs/implementations/2026-10-14-contributor-files.md` (the issue forms and `config.yml`)
- Discussions Q&A: <https://github.com/Cassandra-Collective/Dervish/discussions/categories/q-a>
- Organisation support page: <https://github.com/Cassandra-Collective/.github/blob/main/SUPPORT.md>
