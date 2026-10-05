# Changelog fragments

Each user-facing PR adds **one new file here** instead of editing `CHANGELOG.md`
(#1545). Two PRs adding two different files cannot conflict, so concurrent green PRs
merge back-to-back without a rebase round-trip.

**Name:** `<N>.<category>.md`

- `<N>` — the issue or PR number. For a second entry on the same number append
  `-2`, `-3`, … (`1541-2.fixed.md`).
- `<category>` — one of `added`, `changed`, `deprecated`, `removed`, `fixed`,
  `security`, `performance`.

**Body:** exactly one Markdown bullet, as it should read in `CHANGELOG.md` —
user-facing language, with the `#NN` reference. The first line starts with `- `;
every further line is blank or indented (a continuation of that bullet, tables
included).

```markdown
- **`simulate()` now honours `block_sigma`** (#672): paired rows are drawn from
  the dense residual covariance instead of independent normals.
```

Not user-facing (refactor, tests, CI)? No fragment — put the `no-changelog` label
on the PR instead.

```bash
tools/changelog.sh check      # validate the fragments (also: tools/preflight.sh changelog)
tools/changelog.sh preview    # what the next release section will look like
tools/changelog.sh assemble 0.5.0   # release time only — see docs/development/sdlc.qmd
```

This README is not a fragment.
