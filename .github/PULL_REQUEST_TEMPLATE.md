<!-- Title format: type(scope): short description  [closes #N] -->
<!--
  type  : feat | fix | perf | refactor | docs | test | chore
  scope : parser | estimation | pk | ode | stats | io | api | ad
  e.g.  : feat(estimation): add BHHH Gauss-Newton optimizer with LM damping [closes #14]
-->

## Why
<!-- What problem does this solve? What was wrong or missing before?
     Include any relevant issue, user report, or NONMEM/nlmixr2 comparison
     that motivated the change. -->

## What changed
<!-- Brief description of the approach taken. For estimation changes:
     which approximation, which optimizer, why this over alternatives. -->

## Alternatives considered
<!-- What else was tried or evaluated, and why this approach won. Omit if obvious. -->

## Cross-repo dependency
| Repo | PR | Status | Must merge |
|------|----|--------|------------|
| ferx (R) | FeRx-NLME/ferx-r#___ | open / merged / not needed | before / after / together / — |

<!-- If a ferx PR is required but not yet open, mark this PR as Draft. -->

## Breaking changes
- [ ] `.ferx` model file format (add migration note below)
- [ ] `FitResult` / sdtab fields changed (ferx parses these — link ferx PR above)
- [ ] Public Rust API (`api.rs` / `types.rs`)
- [ ] None

<details>
<summary>Migration notes (if breaking)</summary>

<!-- What do users need to change in their .ferx files or calling code? -->

</details>

## Numerical validation
<!-- For estimation / PK / stats changes: show that results are correct. -->
- [ ] Warfarin example converges and estimates are within tolerance of prior run
- [ ] Compared against: NONMEM / nlmixr2 / prior ferx commit `______`
- [ ] Reference values (paste key theta/omega/OFV, or link to output file):

```
# before / reference

# after
```

- [ ] Not applicable (docs / refactor / CI only)

## Performance
- [ ] No significant impact expected
- [ ] Faster — benchmark: `______` (before) → `______` (after)
- [ ] Slower — justified because: `______`

## Tests
- [ ] Unit test(s) added in the same module (`cargo test --lib` passes)
- [ ] Regression test added that fails without this fix
- [ ] No new tests needed (why: `______`)

## Example (user-facing features only)
<!-- If this adds or changes any user-visible behaviour (new fit option, new DSL syntax,
     new output field, new estimator), add a minimal working .ferx snippet and the
     expected output. This becomes the basis for docs and can be copy-pasted by users. -->
- [ ] Example added to `examples/` or inline below
- [ ] Not applicable (internal / refactor / fix with no new API surface)

<details>
<summary>Example (if not a standalone file)</summary>

```toml
# .ferx snippet demonstrating the new feature

```

```
# expected key output (theta / omega / OFV or sdtab excerpt)

```

</details>

## Docs
- [ ] `docs/src/` updated for user-visible changes
- [ ] New pages linked in `docs/src/SUMMARY.md` (do **not** commit `docs/book/` — it is git-ignored; CI builds & deploys it)
- [ ] `CHANGELOG.md` `[Unreleased]` entry added (user-facing change), or N/A (internal/refactor/CI)
- [ ] No user-visible change

## Checklist
- [ ] `cargo clippy` clean
- [ ] `cargo check --features autodiff` still compiles (if touching AD-reachable code)
- [ ] `Cargo.toml` version bumped if breaking

## Docs & examples

### ferx-site
- [ ] New `.ferx` DSL syntax or `[fit_options]` key → `ferx-site/model-dsl/` page updated or PR opened
- [ ] New example `.ferx` file added to `examples/` → matching entry in ferx-site examples and ferx-book
- [ ] New data file added to `data/` → mirrored to `ferx-r/inst/examples/data/` and referenced in site/book

### ferx-book
- [ ] New estimator, DSL feature, or fit option → relevant book chapter updated or PR opened

### Example execution (run locally before marking ready for review)
- [ ] Rebuilt ferx-core: `cargo build --release`
- [ ] Rebuilt ferx-r against updated ferx-core: `cd ../ferx-r && FERX_NO_AUTODIFF=1 R CMD INSTALL .`
- [ ] All affected `examples/*.ferx` run cleanly via CLI: `cargo run --release -- examples/<model>.ferx --data data/<data>.csv`
- [ ] All affected ferx-site example `.qmd` pages render cleanly: `quarto render examples/<page>.qmd`
- [ ] All affected ferx-book chapters render cleanly: `quarto render chapters/<chapter>.qmd`
- [ ] No example execution step needed (internal refactor / docs-only / no user-visible change)

## Reviewer hints
<!-- Where to focus. What's subtle. What can be skimmed. -->

## Open questions
<!-- Things you're uncertain about and want input on. -->
