# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ferx-core is a Rust-based Nonlinear Mixed Effects (NLME) modeling engine for population pharmacokinetics. It implements FOCE/FOCEI estimation methods, similar to NONMEM, with analytical PK solutions and an optional ODE solver.

## Sibling repositories

The R wrapper package lives at `../ferx-r` (sibling directory). The R package's Rust glue depends on `ferx-core` via git, but its `src/rust/.cargo/config.toml` carries a `[patch]` that auto-swaps in `../../../ferx-core` (this repo) when the sibling checkout exists. So **local** R-package builds pick up changes here automatically — no Cargo.toml edits needed on either side. When a change to a `pub` API in `ferx-core` lands, expect to follow up with a matching PR in `ferx-r`.

Note that ferx-r's **CI** does not get the change automatically: it builds from the ferx-core commit pinned in `ferx-r/src/rust/Cargo.lock` (the patch only applies locally). A ferx-r PR that needs a new ferx-core commit — e.g. a newly-`pub` API — must bump that lock, via `ferx-r/tools/update-ferx-core-lock.sh` (never a bare `cargo update`, which the patch will unpin). Otherwise CI fails with `error[E0603]: ... is private`.

## Worktree isolation

When working on a feature branch or any branch other than `main`, always use `EnterWorktree` at the start of the session. This prevents uncommitted WIP from one session contaminating another session on a different branch (a real problem when two chats share the same checkout directory).

## First-time setup

After cloning, activate the shared pre-commit hook (blocks commits that fail `rustfmt`):

```bash
git config core.hooksPath .githooks
```

## Build & Run Commands

```bash
# Build (debug)
cargo build

# Build (release, with fat LTO)
cargo build --release

# Build with autodiff feature
cargo build --release --features autodiff

# Run CLI with data file
cargo run --release -- examples/warfarin.ferx --data data/warfarin.csv

# Run CLI with simulated data
cargo run --release -- examples/warfarin.ferx --simulate

# Check compilation without building
cargo check

# Run clippy lints
cargo clippy
```

The binary is called `ferx` and outputs `{model}-fit.yaml` (estimates) and `{model}-sdtab.csv` (per-subject diagnostics).

## Tests

There are three tiers of tests. Put a new test in the lowest tier whose constraints it fits.

**Tier 1 — Fast unit tests** (inline `#[cfg(test)] mod tests { ... }` blocks in `src/**/*.rs`)
Test the smallest helper that isolates the behaviour; avoid calling `fit()`. Run with `cargo test --lib`. These run on every PR and must stay fast (seconds total).

**Tier 2 — Integration tests** (`tests/*.rs`)
Call the public API (`fit()`, `predict()`, etc.) but must return immediately — either with an `Ok` after a handful of outer iterations or with an `Err`. No convergence loops. These files are compile-checked on every PR (`cargo check --tests`) and run nightly in `slow-tests.yml`. Put tests here when you need to exercise a public-API boundary that can't be reached from a `src/` unit test.

**Tier 3 — Slow convergence tests** (`tests/*.rs` or `src/` with `slow-tests` gate)
Full population fits that run to convergence. Gate them so they are skipped in the default PR job:

```rust
#[test]
#[cfg_attr(not(feature = "slow-tests"), ignore = "slow: opt in with --features slow-tests")]
fn test_my_new_estimator() { ... }
```

These run nightly via `slow-tests.yml` and on any push to `main` that touches estimation code. Fast-failing tests (those that call `fit()` but expect an immediate `Err`) do not need gating.

**Every new feature requires a test** at the appropriate tier. When adding a new parser pattern, fit option, estimator, or any public behaviour, add a corresponding test before considering the change done. Bug fixes should add a regression test that fails without the fix.

**Every new feature requires a comparison with NONMEM output.** When adding an estimator, error model, structural model, or any behaviour that produces numerical results (estimates, OFV, residuals, diagnostics), validate it against equivalent NONMEM output and include the comparison — either in the feature's docs page (e.g. `docs/src/faq.md` or the relevant `docs/src/estimation/*.md`) or in the PR description.

**Coverage is gated per PR.** A PR's changed lines must carry their own tests — the Codecov `patch` status enforces ≥90% coverage on the diff, and a 90% project floor is enforced on the weekly `main` run (see `codecov.yml`). This is the automated backstop to the rules above; slow-tests never run on PRs, so unit / Tier-2 tests are what register coverage. When excluding code from coverage, **scope `ignore`s by role, not by coverage %**: leave code out for *what it is* — dev-only tooling (e.g. `src/bin/generate_data.rs`), generated code (`build.rs`), or test scaffolding (`tests/`) — never because it reads red. (Feature-gated code that the coverage build doesn't compile reads as "missed" but is a measurement gap, not an ignore target — see #293.)

## Documentation

Docs live in `docs/` as an [mdBook](https://rust-lang.github.io/mdBook/):

- `docs/src/` — Markdown sources. **These are the only docs files you edit or commit.**
- `docs/src/SUMMARY.md` — table of contents; new pages must be added here to show up in the book.
- `docs/book/` — built HTML. **Generated, git-ignored, and never committed.** CI builds and deploys it: the `Deploy Docs` workflow (`.github/workflows/docs.yml`) runs `mdbook build` and publishes to the `gh-pages` branch on every push to `main` that touches `docs/**`. Do **not** run `mdbook build` for the purpose of committing output (you may still build locally to preview — the result stays untracked).

Any user-visible feature (new fit option, new estimator, new file-format directive, behavioural change) must update the relevant page — typically one of:

- `docs/src/model-file/fit-options.md` for `[fit_options]` keys.
- `docs/src/model-file/individual-parameters.md` for DSL syntax.
- `docs/src/estimation/*.md` for estimator-specific behaviour.
- `docs/src/faq.md` for user-facing explanations / comparisons to NONMEM / nlmixr2.

## Architecture

### Two-Level Optimization (FOCE/FOCEI)

The estimation engine uses a nested optimization structure:

- **Outer loop** (`estimation/outer_optimizer.rs`): Optimizes population parameters (theta, omega, sigma) using NLopt BOBYQA (default), SLSQP, L-BFGS, MMA, or built-in BFGS. Parameters are log-transformed for theta/sigma, Cholesky-factored for omega.
- **Inner loop** (`estimation/inner_optimizer.rs`): For each subject, finds empirical Bayes estimates (EBEs) of random effects (eta) by minimizing individual negative log-likelihood. Uses BFGS with warm-start from prior iteration; falls back to Nelder-Mead on failure.

### Gauss-Newton (BHHH) Optimizer

An alternative estimation method using the BHHH (Berndt-Hall-Hall-Hausman) approximation to the Hessian is available in `estimation/gauss_newton.rs`. It uses the outer product of per-subject gradients (`H ≈ Σ gᵢgᵢᵀ`) with Levenberg-Marquardt damping and backtracking line search. Two variants are available:

- **`method = gn`** (pure Gauss-Newton): Fast convergence for well-conditioned problems.
- **`method = gn_hybrid`**: Runs GN first, then polishes with FOCEI via `outer_optimizer.rs` for robustness.

Set via `[fit_options]` in the model file or `EstimationMethod::FoceGn` / `FoceGnHybrid` in code.

### Model Pipeline

```
.ferx file → parser/model_parser.rs → CompiledModel
NONMEM CSV  → io/datareader.rs       → Population
(CompiledModel, Population) → api.rs:fit() → FitResult
FitResult → io/output.rs → sdtab CSV + fit YAML
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `types.rs` | Core structs: `CompiledModel`, `Population`, `Subject`, `FitResult`, `FitOptions` |
| `api.rs` | Public API: `fit()`, `simulate()`, `predict()`, `fit_from_files()` |
| `parser/model_parser.rs` | Parses `.ferx` model DSL into `CompiledModel` with closures |
| `pk/` | Analytical 1-cpt and 2-cpt PK solutions (IV, oral, infusion) with superposition |
| `ode/solver.rs` | Dormand-Prince RK45 adaptive ODE solver |
| `ode/predictions.rs` | ODE-based predictions with dose event handling |
| `estimation/gauss_newton.rs` | Gauss-Newton (BHHH) optimizer with LM damping; pure GN and GN+FOCEI hybrid |
| `estimation/trust_region.rs` | Newton trust-region outer optimizer (argmin + Steihaug CG); FD gradient & Hessian with fixed EBEs |
| `estimation/parameterization.rs` | Pack/unpack optimizer vector (log-theta, Cholesky-omega, log-sigma) |
| `stats/likelihood.rs` | Individual, FOCE, and FOCEI negative log-likelihood computations |
| `stats/residual_error.rs` | Additive, proportional, combined error models; IWRES/CWRES |
| `ad/` | Forward-mode automatic differentiation via dual numbers (behind `autodiff` feature) |
| `io/datareader.rs` | NONMEM-format CSV reader (ID, TIME, DV, EVID, AMT, CMT, RATE, MDV, II, SS) |

### Model File Format (.ferx)

Models are defined in a custom DSL with blocks: `[parameters]`, `[individual_parameters]`, `[structural_model]`, `[error_model]`, `[fit_options]`, `[odes]`, `[simulation]`. See `examples/` for reference models. Omega can be diagonal (`omega NAME ~ variance`) or block (`block_omega (NAME1, NAME2) = [lower_triangle]`) for correlated random effects.

### PK Parameter Convention

PK parameters use a fixed-size array `[f64; 8]` with indices: CL=0, V/V1=1, Q=2, V2=3, KA=4, F=5. This fixed layout enables automatic differentiation without dynamic allocation.

### Parameterization

The optimizer works in a transformed space: theta and sigma are log-transformed, omega uses Cholesky factorization. `estimation/parameterization.rs` handles packing/unpacking between the optimizer vector and model parameters.

### Warning and Error Conventions

Warnings and non-fatal issues should be collected into `FitResult.warnings` (a `Vec<String>`), not printed directly to stderr. The CLI layer (`output::print_results`) handles display. This keeps the library quiet for non-verbose callers and ensures warnings appear in both console and YAML output.

### Autodiff-Safe Code in `ad/` Module

Any function that is autodiff-instrumented (i.e., called from code under `#[autodiff_forward]` / `#[autodiff_reverse]` macros, or reachable from `single_dose_ad` / `individual_nll_ad` / `predict_all_ad`) **must not use `f64::max()` or `f64::min()`**.

Recent rustc (2025+) lowers these methods to the LLVM intrinsics `llvm.maximumnum.f64` and `llvm.minimumnum.f64`. Enzyme does not yet have differentiation rules for these intrinsics and will fail at compile time with:

```
error: Enzyme: cannot handle (forward) unknown intrinsic llvm.maximumnum
```

**Do this instead** — use explicit comparisons:

```rust
// Bad (in AD-instrumented code):
let alpha = lambda0.max(lambda1).max(lambda2);
let disc = (s * s - 4.0 * d).max(0.0).sqrt();

// Good:
let alpha = if lambda0 >= lambda1 && lambda0 >= lambda2 {
    lambda0
} else if lambda1 >= lambda2 {
    lambda1
} else {
    lambda2
};
let disc = { let x = s * s - 4.0 * d; if x > 0.0 { x.sqrt() } else { 0.0 } };
```

The same restriction applies to any helper the AD code calls transitively — `macro_rates`, `macro_rates_three_cpt_ad`, etc. The analytical PK functions in `pk/` are fine to use `.max()`/`.min()` because they're called from the non-AD path; only the inlined AD duplicates (in `ad/ad_gradients.rs`) need this care.

This restriction will go away once Enzyme upstream adds rules for the newer intrinsics — track at https://github.com/EnzymeAD/Enzyme/issues. When removing the workaround, re-enable a representative test under CI with the `autodiff` feature to catch regressions.

## Changelog

User-facing changes are tracked in `CHANGELOG.md` at the repo root, in
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format with
[semantic versioning](https://semver.org/).

**In the same PR as any user-facing change, add a one-line entry under the
`## [Unreleased]` heading** in the correct category (`Added`, `Changed`,
`Deprecated`, `Removed`, `Fixed`, `Security`, or `Performance`). Write it in
user-facing language and reference the issue/PR number (`#NN`). A PR that only
touches internal refactors, tests, or CI does not need an entry.

At release time (not per-PR), `## [Unreleased]` is renamed to the new version
with a date, a fresh empty `## [Unreleased]` is started, and the compare links
at the bottom are updated. The R wrapper (`../ferx-r`) tracks its own
user-facing changes in `NEWS.md`, so a cross-repo change may need an entry in
both.

## Pull Requests

When creating a PR in this repo, always read `.github/PULL_REQUEST_TEMPLATE.md` and fill every section before calling `gh pr create`.
