# The move to the Cassandra-Collective organisation, and the org header

- Date: 2026-10-14
- Author: Mert Efe Şensoy
- Commits: see the pull request `docs/org-header`

## 1 · Problem / motivation

On 2026-10-14 this repository moved from the personal account `mertefesensoy` to the organisation `Cassandra-Collective`. GitHub redirects the old URL only until a repository is created at the old path, and `pyproject.toml` publishes the repository URL as package metadata, so it should name the real location. The organisation also gained an identity the same day (the Forecast mark, in `Cassandra-Collective/.github`), and the owner asked for a small "part of the collective" header on each project README.

## 2 · What changed

| File | Change |
| --- | --- |
| `pyproject.toml` | `[project.urls] Repository` now points at `https://github.com/Cassandra-Collective/Dervish`. |
| `README.md` | A one-line header above the title: the small Forecast mark (light and dark variants) and "Part of the Cassandra Collective", both linking to the organisation. |
| `docs/implementations/2026-10-14-org-move-and-header.md` | This document. |

## 3 · Implementation approach

One scripted replacement over every git-tracked text file: `github.com/mertefesensoy/(CASSANDRA|CASSANDRA-Chorus|Ephesus|Dervish)` becomes `github.com/Cassandra-Collective/<same name>`, followed by a re-grep. It found one link. The author e-mail in `docs/implementations/2026-10-14-history-rewrite.md` contains the account name but is not a repository link and is untouched.

The header loads the mark from the organisation's brand folder by absolute URL, so a later change to the mark updates every README at once. It uses the small mark, which the brand guide prescribes for 24 px and below, at 20 px, with a `<picture>` element choosing the light or dark variant.

## 4 · Mathematical / statistical details

None: metadata and markup only.

## 5 · Design decisions

- **Absolute URL to the brand folder rather than a copy of the SVG here:** one source of truth for the mark, at the cost of depending on `Cassandra-Collective/.github` staying public.
- **Same markup as the CASSANDRA and CASSANDRA-Chorus READMEs**, which was checked rendering on GitHub in light and dark mode before it was copied here.

## 6 · Verification

- `git grep -i -E "github\.com/mertefesensoy/(CASSANDRA|Ephesus|Dervish)"` returns nothing after the change.
- The header renders on GitHub on this branch (checked after pushing; see the pull request).
- No code, test or result file changed, so the D001 and D002 records and `verify.sh` are unaffected.

## 7 · Related docs

- Organisation brand guide: <https://github.com/Cassandra-Collective/.github/blob/main/brand/README.md>
