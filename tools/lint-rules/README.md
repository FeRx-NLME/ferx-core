# `tools/lint-rules/` — source-pattern lints for test code

Rules enforced by `crates/patterns-lint` (#2634), run by `tools/preflight.sh
patterns` and the `Patterns lint` CI job.

| Rule | What it catches | Rule file |
|---|---|---|
| `nan-fold` | a `max`/`min` fold in test code, which discards `NaN` | [`nan-fold.md`](nan-fold.md) |

[`baseline.txt`](baseline.txt) is the shrink-only ratchet of findings that
predate a rule. A new finding fails, a function whose count grows fails, and a
function whose count falls fails until its line is lowered
(`cargo run -p patterns-lint -- --update-baseline`).

```bash
tools/preflight.sh patterns                       # the gate CI runs
cargo run -p patterns-lint                        # same report from the CLI
cargo run -p patterns-lint -- --no-baseline       # every finding, baselined or not
cargo run -p patterns-lint -- --explain nan-fold  # print a rule file
```

The rules work on the `syn` AST, not on lines. A regex cannot see a fold that
rustfmt wrapped over three lines, cannot tell a `#[cfg(test)]` module from the
production code around it, and cannot look inside an `assert!(…)`. That is also
why this is a small crate and not an ast-grep config: tree-sitter-rust makes
attributes siblings of the items they annotate, which makes "inside a
`#[cfg(test)]` module" awkward to express, and the `is_finite()` escape hatch
compares identifiers across statements.

To add a rule, add a module next to `crates/patterns-lint/src/nan_fold.rs`,
call it from `scan()`, and add its rule file and a row in the table above. If
you need to record its existing sites, use `--record-baseline` in that PR only.
