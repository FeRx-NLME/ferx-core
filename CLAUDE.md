# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ferx-core is a Rust-based Nonlinear Mixed Effects (NLME) modeling engine for population pharmacokinetics. It implements FOCE/FOCEI estimation methods, similar to NONMEM, with analytical PK solutions and an optional ODE solver.

## Sibling repositories

The R wrapper package lives at `../ferx-r` (sibling directory). The R package's Rust glue depends on `ferx-core` via git, but its `src/rust/.cargo/config.toml` carries a `[patch]` that auto-swaps in `../../../ferx-core` (this repo) when the sibling checkout exists. So changes here are picked up by an R-package build automatically — no Cargo.toml edits needed on either side. When a change to a `pub` API in `ferx-core` lands, expect to follow up with a matching PR in `ferx-r`.

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

Inline `#[cfg(test)] mod tests` blocks at the bottom of each module (e.g. `src/parser/model_parser.rs`, `src/estimation/parameterization.rs`). Run with `cargo test --lib`.

**Every new feature requires a test.** When adding a new parser pattern, fit option, estimator, or any public behaviour, add a corresponding unit test in the same file's `tests` module before considering the change done. Bug fixes should add a regression test that fails without the fix.

Prefer unit tests of the smallest helper that isolates the new behaviour (e.g. test `detect_mu_refs` directly, not just through a full `fit()` call) — end-to-end fits are too slow and flaky for the default test suite.

**Any test that calls `fit()` and runs to convergence must be marked `#[ignore]`** to keep `cargo test --lib` fast. Add the attribute immediately after `#[test]`:

```rust
#[test]
#[ignore = "slow: runs a full optimisation to convergence"]
fn test_my_new_estimator() { ... }
```

Ignored tests still run in CI on a weekly schedule and can be triggered manually via `workflow_dispatch`. Fast-failing tests (those that call `fit()` but expect an immediate `Err` without running the optimiser) do not need `#[ignore]`.

## Documentation

Docs live in `docs/` as an [mdBook](https://rust-lang.github.io/mdBook/):

- `docs/src/` — Markdown sources (edit these).
- `docs/book/` — built HTML, **committed to the repo** (GitHub Pages deploys from it). Run `cd docs && mdbook build` after editing any `docs/src/*.md` and commit both the source and the built output in the same commit.
- `docs/src/SUMMARY.md` — table of contents; new pages must be added here to show up in the book.

Any user-visible feature (new fit option, new estimator, new file-format directive, behavioural change) must update the relevant page — typically one of:

- `docs/src/model-file/fit-options.md` for `[fit_options]` keys.
- `docs/src/model-file/individual-parameters.md` for DSL syntax.
- `docs/src/estimation/*.md` for estimator-specific behaviour.
- `docs/src/faq.md` for user-facing explanations / comparisons to NONMEM / nlmixr2.

## Architecture

### Two-Level Optimization (FOCE/FOCEI)

The estimation engine uses a nested optimization structure:

- **Outer loop** (`estimation/outer_optimizer.rs`): Optimizes population parameters (theta, omega, sigma) using NLopt SLSQP (default), L-BFGS, MMA, or built-in BFGS. Parameters are log-transformed for theta/sigma, Cholesky-factored for omega.
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

## Pull Requests

When creating a PR in this repo, always read `.github/PULL_REQUEST_TEMPLATE.md` and fill every section before calling `gh pr create`.
