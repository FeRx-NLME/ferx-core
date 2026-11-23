# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, Codex, Copilot, and any other tool that reads the [AGENTS.md](https://agents.md) convention) when working with code in this repository.

## Project Overview

ferx-core is a Rust-based Nonlinear Mixed Effects (NLME) modeling engine for population pharmacokinetics/pharmacodynamics, similar to NONMEM. It offers several estimation methods — FOCE/FOCEI (the default family), Laplace/AGQ, SAEM, importance sampling (IMP/IMPMAP), full Bayesian MCMC, and others (`EstimationMethod` in `types.rs`; one file per method under `src/estimation/`) — with analytical PK solutions and an ODE solver.

## Code style: parsimony

Prefer the simplest solution that solves the problem.

- Write the least code that does the job correctly. No speculative generality, extra options, or layers that nothing asks for yet.
- Factor shared logic into functions rather than duplicating it. If two code paths compute the same thing, they should call the same function — a second copy will drift (see "two implementations disagree" under Tests).
- Keep functions small and single-purpose; extract a helper when a block has a nameable job.
- Reuse existing helpers before writing new ones — search the codebase first.
- Delete dead code rather than leaving it commented out or behind a flag.

## Sibling repositories

The R wrapper package lives at `../ferx-r`. Its `src/rust/.cargo/config.toml` carries a `[patch]` that swaps in `../../../ferx-core` (this repo) when the sibling checkout exists, so **local** R-package builds pick up changes here automatically. When a `pub` API change lands here, expect a matching PR in `ferx-r`.

ferx-r's **CI** builds from the ferx-core commit pinned in `ferx-r/src/rust/Cargo.lock` (the patch is local only). A ferx-r PR that needs a new ferx-core commit must bump that lock via `ferx-r/tools/update-ferx-core-lock.sh` (never a bare `cargo update`, which the patch will unpin); otherwise CI fails with `error[E0603]: ... is private`.

**Downstream repos pin ferx-core transitively.** `ferxtranslate` (and anything built on the R package) pins **ferx-r**, whose `Cargo.lock` pins ferx-core. A merge here reaches them only after two bumps, in order: ferx-r's `Cargo.lock`, then the downstream ferx-r pin. Never tell a downstream maintainer to "pin ferx-core at `<sha>`". `ferx-r/src/rust/Cargo.toml` says `branch = "main"`, but **the lock is the pin**:

```bash
git -C ../ferx-r show origin/main:src/rust/Cargo.lock | grep -A2 'name = "ferx-core"'
```

## Worktree isolation

When working on any branch other than `main`, use `EnterWorktree` at the start of the session, so uncommitted WIP from one session cannot contaminate another sharing the same checkout.

## Workspace layout and the public-API boundary

```
ferx-core/                       # repo root = ferx-core package + workspace root
├─ Cargo.toml                    # [package] ferx-core + [workspace] members
├─ src/                          # the engine
├─ api/ferx-core-public-api.txt  # public-API baseline, diffed in CI
└─ crates/
   ├─ ferx-tools/                # depends on ferx-core
   ├─ ferx-cli/                  # depends on ferx-core; [[bin]] name = "ferx"
   ├─ docs-lint/                 # dev tooling; depends on NOTHING
   └─ patterns-lint/             # dev tooling; test-code lints, depends only on `syn`
```

- **The root package must stay `ferx-core`.** ferx-r's local patch points at the repo root; moving the package under `crates/` silently makes ferx-r build `main` instead of your branch.
- **Boundary rule.** `ferx-core` = one model, one dataset, one fit. `ferx-tools` = many fits, resampling, model-space search — if it calls `fit()` more than once, it is a tool. Statistical kernels (npde, CWRES, shrinkage, covariance/SE, SIR, FREM, VPC statistics) stay in core. `ferx-cli` is thin: argument parsing, printing, exit codes. `ferx-core` never depends on a member (pinned by `tests/public_api_boundary.rs`). `docs-lint` and `patterns-lint` (source-pattern lints over test code; rules and baseline in `tools/lint-rules/`) depend on no workspace crate and nothing depends on them, which keeps their CI jobs fast.
- **Widening the public API is a design step.** Any new `pub` item fails the `Public API baseline` CI job until the baseline is regenerated in the same PR (`tools/update-public-api.sh`, `--check` is what CI runs). Say in the PR description which tool needs each added item and why the existing surface does not suffice.
  - **No `#[doc(hidden)] pub`.** `cargo public-api` omits such items, so it bypasses the gate; `tests/public_api_boundary.rs` bans it. An item is either public API or `pub(crate)`.
  - Anything `ferx-tools` needs must also make sense from ferx-r, which consumes `ferx-core` as an ordinary external crate.
- **Feature flags belong to `ferx-core`**, so workspace-wide or member-scoped commands must package-qualify them: `--features ferx-core/ci`, not `--features ci`.

## First-time setup

```bash
git config core.hooksPath .githooks   # pre-commit hook: blocks commits that fail rustfmt
```

## Before you push: `tools/preflight.sh`

**A green test suite is not a green build.** CI feature sets are not nested — `cargo test --features ci,survival` compiles neither the `nn`- nor the `slow-tests`-gated source — so run the fast gates before pushing:

```bash
tools/preflight.sh            # fmt, the cargo check feature matrix, clippy, rustdoc, docs, patterns, public-API, changelog
tools/preflight.sh check      # just the check matrix, for a tight edit loop
tools/preflight.sh --list     # show the commands without running them
```

CI's `Check`, `Clippy`, `Format`, `Rustdoc` and `Docs lint` jobs and the pre-commit hook call this same script, so they cannot drift. **Add a new gate to the script (`ALL_GROUPS`), not to the workflow.** It gates compilation and lint, not behaviour — the test jobs can still go red after a green preflight. `tests/preflight_owns_the_fast_gates.rs` pins the arrangement; when editing the script, keep `run` **exiting** (not returning) on failure, since bash suspends `errexit` inside AND-OR lists.

Things to know:

- Clippy runs `--all-targets -- -Dunused`. `cargo clippy` exits 0 on warnings, so `-Dunused` is what turns a left-behind import/item into a failure. It is deliberately not `-Dwarnings` (hundreds of style findings, and CI uses a moving nightly). Fix an unused-item error by deleting the item. "Used" is per feature set: an item read only from a `#[cfg(feature = "survival")]` body is dead in the default build ferx-r ships, so the group also lints `--lib` alone under `ci` — gate such an item on the same feature as its reader.
- `rustdoc` (doc comments in `src/**`) and `docs` (the `.qmd` linter) are different gates. Fix a broken intra-doc link at the link — inline code for an internal target, a resolvable path for a public one — never with a crate-level `#![allow(rustdoc::...)]`.

## Build & Run Commands

```bash
cargo build                                   # library (debug)
cargo build --release --workspace             # everything incl. the `ferx` binary (or: -p ferx-cli)
cargo run --release -p ferx-cli -- examples/warfarin.ferx --data data/warfarin.csv
cargo run --release -p ferx-cli -- examples/warfarin.ferx --simulate
```

The binary is called `ferx` and outputs `{model}-fit.yaml` (estimates) and `{model}-sdtab.csv` (per-subject diagnostics).

## Tests

Put a new test in the lowest tier whose constraints it fits.

**Tier 1 — Fast unit tests** (`#[cfg(test)]` modules in `src/**/*.rs`). Test the smallest helper that isolates the behaviour; avoid calling `fit()`. Run with `cargo test --lib`; must stay fast. In the largest files the test module lives in a sibling file (`#[cfg(test)] #[path = "<file>_tests.rs"] mod tests;`) — **add new tests to the sibling when one exists.**

**Tier 2 — Integration tests** (`tests/*.rs`). Call the public API but return immediately (`Ok` after a handful of outer iterations, or `Err`). No convergence loops. Compile-checked on every PR, run nightly.

**Tier 3 — Slow convergence tests.** Full fits to convergence, gated:

```rust
#[test]
#[cfg_attr(not(feature = "slow-tests"), ignore = "slow: opt in with --features slow-tests")]
fn test_my_new_estimator() { ... }
```

The gate marks a *fit to convergence*. An anchor that evaluates once (a NONMEM objective at `maxiter = 0`, a frozen mrgsolve table) is Tier 2 and is not gated, however slow its oracle was to produce; `tests/anchor_gating.rs` enforces this for the listed anchors.

**A reconverged-FD case table is a shard floor.** Each `fd_oracle` / `parity` cell can cost 60–120 s and libtest runs a test on one thread, so wrap such tables in `oracle_cells(cases, per_pr)` (`sens/ode_adjoint_tests.rs`): all cells under `slow-tests`, per PR only the cells that catch the named defect. Time new oracles with `-- -Zunstable-options --report-time`. Smoke fits that assert only a finite objective need a few outer iterations, not 60.

**Every new feature requires a test**, and bug fixes require a regression test that fails without the fix.

**Every new feature that produces numerical results requires a comparison with NONMEM output** (estimates, OFV, residuals, diagnostics), in the feature's docs page or the PR description. A control stream committed under `nonmem_anchor/` gets a row in `nonmem_anchor/README.md`'s catalogue table in the same PR (`tests/nonmem_anchor_index.rs` enforces it). Exceptions, only when there is genuinely no NONMEM spelling of the quantity (not when the construction is merely hard):

- *Adaptive / feedback dosing (#391):* NONMEM has no feedback dosing. Use a degenerate oracle (a controller re-emitting a fixed regimen equals `simulate()` bit-for-bit), the frozen-schedule replay verifier, and reproduction of a published **mrgsolve** dynamic-dosing example.
- *Survival terms with no NONMEM/survreg estimator* (e.g. recurrent left truncation under gap-time RTTE, #740) and *hazard times before a subject's first record* (#1223): use exact hand-computed closed forms, reduction to an already-anchored term, agreement across every production caller, and a differential pair straddling the boundary.

### Oracle and assertion rules

- **Keep every side of the object under test non-degenerate.** A single-dose dataset cannot test a dose event's incoming side (state is zero before the first dose). Pair dose-event anchors with a multi-dose case, and make sure the quantity under test actually varies across the boundary. A fixture asserted against ferx's own predictor, or two ferx engines sharing upstream code, can agree on a wrong answer.
- **The oracle must be more accurate than the difference you call a defect.** An ODE twin at default tolerances is not automatically an oracle (relative accuracy degrades as a state approaches `abstol`). When unsure which side is wrong, compute the quantity a third way outside both engines.
- **Measure tolerances; never justify them with a story.** Print the worst realised error, put it in the comment, choose the bound from it with stated headroom. A bound far looser than the realised error is a test that has stopped testing.
- **Linux is the reference platform for fit numbers.** macOS fits differ via the OS math library. Never compare a macOS number to a Linux one or widen a band to cover macOS. Get Linux numbers with `tools/linux-test.sh -- <cargo args>` or `gh workflow run slow-tests.yml --ref <branch> -f nocapture=true`.
- **When two implementations disagree, the fix is one implementation**, shared by both callers.
- **A green test is not evidence it can fail.** Common holes:
  - A differential pair must straddle the gate under the *old* behaviour.
  - A fast path (`if xs.is_empty()`) can make the mutation unreachable.
  - `f64::max`/`f64::min` discard `NaN`, as do `filter_map(…ok())` and `unwrap_or(0.0)`; assert `is_finite()` before folding. This is linted in test code (`nan-fold`, `tools/preflight.sh patterns`): use `// nan-ok: <reason>` for a justified exception, and never add to the shrink-only `tools/lint-rules/baseline.txt`.
  - `is_finite()` cannot see a sentinel (e.g. `1e20` for a repelled subject); bound the magnitude instead.
  - Two redundant gates rejecting the same inputs hide each other's deletion; collapse to one. A predicate that cannot change the answer for any input that reaches it is the degenerate case — delete it.
  - A shared branch serving several inputs needs an assertion on what makes them different (a keyword, a name) **and its negative**, e.g. `contains("`block_kappa`") && !contains("block_omega")`.
- **Name the regression each assertion catches, and check it can reach it** — by mutation for a regression test. Ask what the *smallest* edit removing the fix is and which test dies; mutate each side of a twin separately; and treat user-facing messages as code: delete each sentence in turn and name the test that dies. The `Nightly mutants` workflow (`mutants.yml`) lists surviving mutants from each day's merged `src/` diff — a backstop for the hand check, not a substitute (`--lib`-only, never mutates string literals).

**Every change to an analytic sensitivity, gradient, marginal, or likelihood path requires a dual-vs-FD parity test** (Tier 1), against central finite differences of the `T = f64` production objective (EBE re-found at every point), to a measured tolerance. Follow the reconverged-FD oracles `fd_oracle` / `fd_oracle_agq` (outer gradient), `hessian_gaps_for` (R-matrix) and `objective_gaps` in `sens/ode_adjoint_tests.rs`, plus the per-kernel `*_g_dual_matches_fd` checks (`sens/propagate.rs`). Parity pins the derivative against the value path, not the value path itself. Models or subjects outside the recorders' scope must route to FD via the scope predicates (`adjoint::model_in_scope` / `recorder_in_scope`, `analytic_inner_grad_supported_model`, `cov_width_declines`); unit-test that routing. Check which engine each fixture actually ran on — the `FitResult` FD-fallback warning tells you; a fixture that routes to FD says nothing about the plan path.

**Coverage is gated per PR**: Codecov `patch` ≥90% on the diff. Slow tests don't run on PRs, so unit / Tier-2 tests provide coverage. Exclude code from coverage by role (dev tooling, generated code, test scaffolding), never because it reads red.

## Documentation

Docs live in `docs/` as a [Quarto](https://quarto.org/) website. Edit and commit only `docs/**/*.qmd` (and `docs/_quarto.yml` to add pages to navigation). `docs/_site/` is generated, git-ignored, and deployed by CI — never commit it. Styling stays aligned with `../ferx-nlme.github.io`.

`docs/llms.txt` is the curated index for language models (built into `llms-full.txt` by `docs/_llms.py`); a new user-facing page worth an agent's attention gets a line there. The user-facing counterpart of this file is `docs/agents.qmd`.

Any user-visible change must update the relevant page — typically `docs/model-file/fit-options.qmd`, `docs/model-file/individual-parameters.qmd`, `docs/estimation/*.qmd`, or `docs/faq.qmd`.

**Docs lint** (`crates/docs-lint`, CI job `Docs lint`): run `cargo run -p docs-lint -- --check` after editing a page. Rules: R1 a leaf section stays under 5,000 characters; R2 no skipped heading levels; R3 no duplicate heading ids; R4 internal links resolve, including `#anchor`. Pandoc's slugs are rarely what a hand-written anchor assumes (`## 3. Communication` → `#communication`, `## A — b` → `#a-b`); see `docs/development/docs-lint.qmd`. The R1 baseline (`docs/.lint-baseline`) is a ratchet — never add entries; split the section instead.

## Architecture

### Model Pipeline

```
.ferx file → parser/model_parser.rs → CompiledModel
NONMEM CSV  → io/datareader.rs       → Population
(CompiledModel, Population) → api/fit.rs:fit() → FitResult
FitResult → io/output.rs → sdtab CSV + fit YAML
```

### Estimation

FOCE/FOCEI use nested optimization: an **outer loop** (`estimation/outer_optimizer.rs`) over population parameters (theta, omega, sigma) and an **inner loop** (`estimation/inner_optimizer.rs`) finding each subject's EBEs. The optimizer works in a transformed space — log theta/sigma, Cholesky omega (`estimation/parameterization.rs`). Other methods (SAEM, IMP, Bayes, Laplace, …) live in their own files under `estimation/` and reuse the same model, likelihood, and sensitivity machinery.

### Key Modules

| Module | Purpose |
|--------|---------|
| `types.rs` | Core structs: `CompiledModel`, `Population`, `Subject`, `FitResult`, `FitOptions`, `EstimationMethod` |
| `api/` | Public API: `fit.rs`, `run.rs`, `simulate.rs`, `predict.rs`, `postfit.rs`, `validation.rs`, …; `mod.rs` re-exports |
| `pool.rs`, `entry_refusals.rs`, `levels.rs`, `covariate_stats.rs`, `occasions.rs` | Crate-level (not in `api/`) so code below `api` can use them without depending on it |
| `parser/model_parser.rs` | Parses `.ferx` model DSL into `CompiledModel` |
| `pk/` | Analytical PK solutions with superposition |
| `ode/` | ODE solvers (`solver.rs` RK45 default, `rosenbrock.rs` stiff) and predictions |
| `dosing.rs` | Dose resolution and steady-state equilibration, shared by pk/ode/sens |
| `estimation/` | Estimators, covariance/SE step (`covariance.rs`), parameterization |
| `stats/likelihood.rs` | Individual and population negative log-likelihoods |
| `stats/residual_error.rs` | Residual error models and IWRES/CWRES; sole owner of residual-variance math |
| `sens/` | `PkNum` trait, generic `*_g<T>` kernels, and the adjoint that records them into compiled per-subject plans: exact gradients, `H` and R-matrix for FOCE/FOCEI/Laplace/AGQ |
| `io/datareader.rs` | NONMEM-format CSV reader |

### Model File Format (.ferx)

Custom DSL with blocks: `[parameters]`, `[individual_parameters]`, `[structural_model]`, `[error_model]`, `[fit_options]`, `[odes]`, `[simulation]`. See `examples/`.

### Conventions

- **Warnings** go into `FitResult.warnings`, never directly to stderr; the CLI handles display.
- **Analytic sensitivities**: closed-form PK solutions, event-driven propagators and ODE solvers are written once as generic `*_g<T: PkNum>` functions (`sens/`, `pk/event_driven.rs`, `ode/`). `T = f64` gives predictions; the adjoint (`sens/ode_adjoint.rs`) records the same functions into a Cranelift-compiled per-subject plan, replayed by forward and reverse passes for exact derivatives. `Dual2` / `jet::DualT` remain as forward types used as test oracles. Edit the generic version — there is no second copy.

## Changelog

**In the same PR as any user-facing change, add a fragment** `changelog.d/<N>.<category>.md` — never edit `CHANGELOG.md` directly. `<N>` is the issue/PR number (`-2`, `-3` suffix for extra entries), `<category>` one of `added`, `changed`, `deprecated`, `removed`, `fixed`, `security`, `performance`. Body: exactly one user-facing bullet with the `#NN` reference (see `changelog.d/README.md`). PRs touching only internal refactors, tests, or CI take the **`no-changelog`** label instead. Never delete another PR's fragment. `tools/changelog.sh check` validates. A cross-repo change may also need an entry in ferx-r's `NEWS.md`.

## Priority labels

Issues and PRs carry GitHub labels `P0`–`P3` (correctness severity × how common the trigger): `P0` silent wrong results in common use; `P1` wrong results in realistic use, or a loud failure blocking common workflows; `P2` edge-case correctness, important features, UX/diagnostics; `P3` performance, refactors, nice-to-haves, research. The applied label is the source of truth, not a priority written in the title.

## Pull Requests

When creating a PR, read `.github/PULL_REQUEST_TEMPLATE.md` and fill every section before calling `gh pr create`. Its **Tests** section asks for *Same pattern elsewhere* and *Consumers checked*; use rust-analyzer find-references for the counts.

After reviewing a PR, always post your comments on the PR.
