# Software Development Life Cycle — ferx-core

## 1. Project Overview

**ferx-core** is a Rust-based Nonlinear Mixed Effects (NLME) modeling engine for population pharmacokinetics. It implements FOCE/FOCEI estimation methods with analytical PK solutions and an optional ODE solver.

| Attribute | Value |
|-----------|-------|
| Language | Rust |
| Current version | 0.1.0 (active development) |
| Binary name | `ferx` |
| Repository | GitHub (FeRx-NLME/ferx-core) |
| Toolchain | Stock Rust nightly (pinned in `rust-toolchain.toml`) |

## 2. Development Environment Setup

### Prerequisites

- Stock Rust nightly (pinned in `rust-toolchain.toml`)
- Standard system C linker

### Build commands

```bash
# Debug build (library only)
cargo build

# Release build (with fat LTO). The repo is a workspace and the root package is
# the ferx-core LIBRARY, so `--workspace` (or `-p ferx-cli`) is what produces
# the `ferx` binary — a bare `cargo build --release` does not.
cargo build --release --workspace

# Compilation check (no artifact output)
cargo check

# Lint
cargo clippy
```

### Running

```bash
# Fit a model to data
cargo run --release -p ferx-cli -- examples/warfarin.ferx --data data/warfarin.csv

# Simulate from a model
cargo run --release -p ferx-cli -- examples/warfarin.ferx --simulate
```

## 3. Branching Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Primary development branch; must always build cleanly |
| `feature/*` | Feature branches for new estimation methods, model types, or major changes |
| `gh-pages` | Auto-generated documentation site (Quarto output) |

### Workflow

1. Create a feature branch from `main` (e.g., `feature/gauss-newton`).
2. Develop and validate locally using example models.
3. Open a pull request targeting `main`.
4. Merge after code review and successful validation.

## 4. Code Quality

### Current practices

- **Linting**: `cargo clippy` for Rust idiom and correctness checks.
- **Compilation checks**: `tools/preflight.sh check` — the five `cargo check` feature
  sets CI runs, in one command (#1157). The sets are not nested, so a single
  `cargo check` proves much less than it looks like it does.
- **Formatting**: Rust default formatting via `rustfmt` (no custom configuration).
- **Code review**: Pull requests on GitHub before merging to `main`.

### Recommendations

- Add `rustfmt.toml` if the team wants to codify style preferences beyond defaults.
- Consider `cargo-audit` for dependency vulnerability scanning.

## 5. Testing & Validation

### Unit tests

The project has 57 unit tests covering core computational modules. Tests use the `approx` crate for floating-point comparisons.

```bash
# Run all unit tests
cargo test --lib
```

**Tested modules:**

| Module | Tests | Coverage |
|--------|-------|----------|
| `stats/residual_error.rs` | 11 | Additive, proportional, combined error models; IWRES; MIN_VARIANCE floor |
| `pk/one_compartment.rs` | 13 | IV bolus, infusion, oral; singularity handling; guard clauses; predict dispatcher |
| `pk/two_compartment.rs` | 15 | IV bolus, infusion, oral; macro rates; bioavailability; guard clauses |
| `pk/mod.rs` | 4 | Superposition with multiple doses; future dose exclusion; compute_predictions |
| `estimation/parameterization.rs` | 7 | Pack/unpack round-trip; log-transform correctness; bounds; clamping |
| `ode/solver.rs` | 5 | Exponential decay, linear growth, two-state system; parameter passing |

### Validation models

End-to-end validation uses example models against known datasets (in `examples/` and `data/`):

| Model | Description |
|-------|-------------|
| `warfarin.ferx` | 1-compartment oral (warfarin PK) |
| `warfarin_saem.ferx` | 1-compartment oral with SAEM estimation |
| `two_cpt_iv.ferx` | 2-compartment IV bolus |
| `two_cpt_oral_cov.ferx` | 2-compartment oral with covariates (WT, CRCL) |
| `mm_oral.ferx` | Michaelis-Menten elimination (ODE-based) |

### Future testing roadmap

1. **Integration tests**: Add a `tests/` directory with end-to-end model fitting tests asserting parameter estimates within tolerance.
2. **Regression tests**: Automate comparison of fit results against stored baseline outputs.
3. **Property-based tests**: Consider `proptest` for numerical edge cases in PK solvers and likelihood computations.

## 6. Build & Release

### Build profiles

- **Debug**: Standard Rust debug build for development.
- **Release**: Fat LTO enabled (`lto = "fat"` in `Cargo.toml`) for maximum optimization.

### Versioning

The project uses [Semantic Versioning](https://semver.org/) (currently `0.1.0`). During the `0.x` phase, breaking changes may occur in minor releases.

### Release process (recommended)

1. Update version in `Cargo.toml`.
2. Update `CHANGELOG.md` with notable changes.
3. Create a git tag: `git tag -a v0.2.0 -m "Release v0.2.0"`.
4. Push tag: `git push origin v0.2.0`.
5. CI builds release binaries and creates a GitHub Release (when CI is implemented).

### Changelog

A `CHANGELOG.md` file should be maintained following [Keep a Changelog](https://keepachangelog.com/) format, documenting additions, changes, fixes, and breaking changes per release.

## 7. Documentation

### Sources

| Resource | Location | Purpose |
|----------|----------|---------|
| Quarto site | `docs/**/*.qmd` | User-facing documentation, model DSL reference, estimation methods |
| README.md | Project root | Quick start, overview, model syntax examples |
| CLAUDE.md | Project root | Developer guidance, architecture, build commands |

### Building documentation

```bash
quarto render docs    # Output to docs/_site/ (git-ignored, never committed)
quarto preview docs   # Local preview with live reload
```

### Deployment

The documentation site is deployed to the `gh-pages` branch and served via GitHub Pages.

## 8. CI/CD

A GitHub Actions CI pipeline runs on every push to `main` and on pull requests (`.github/workflows/ci.yml`).

### Pipeline stages

| Job | Command | Purpose |
|-----|---------|---------|
| **Check** | `tools/preflight.sh check` | The five `cargo check` feature sets — `ci`, `ci,survival,slow-tests`, `ci,markov`, `ci,nn,slow-tests`, and the workspace members under `ferx-core/ci`. Covers every feature combo, including the ones no per-PR test job builds (#1157) |
| **Tests + coverage (core)** | `cargo llvm-cov --workspace --tests --features ferx-core/ci` | Runs the Tier-1/2 suite — core *and* the `ferx-tools` / `ferx-cli` members — and produces the per-PR patch-coverage report (`fast` flag) |
| **Tests + coverage (TTE/CTMM endpoints)** | `cargo llvm-cov --lib --test <each endpoint test file> --features ci,markov` | Runs and measures the feature-gated TTE / categorical / CTMM code the base build compiles out (`survival` + `markov` flags) |
| **Clippy** | `tools/preflight.sh clippy` | Lint `ferx-core` under `ci,markov,nn` with `--all-targets` (without it a third of the repo — every `#[cfg(test)]` module and all of `tests/` — goes unlinted, #1023), then the workspace members |
| **Format** | `tools/preflight.sh fmt` | Enforce consistent formatting (`--all`: the workspace members too). `.githooks/pre-commit` runs this same group |
| **Public API baseline** | `tools/update-public-api.sh --check` | Diff `ferx-core`'s public surface against `api/ferx-core-public-api.txt`; any widening fails until the baseline is regenerated in the same PR (#1114) |

There is deliberately **no** separate uninstrumented `Test` job, and no separate
`Survival` job: the former was a strict subset of `Tests + coverage (core)`
(same features, `--tests` ⊇ `--lib`), and `markov = ["survival"]` makes the
latter a strict subset of the endpoints job. Both were removed rather than left
re-paying an ~11 min lib compile for coverage that Codecov already merges in
from another flag.

The endpoints job's `--test` list is not free-form: `tests/ci_workflow_endpoint_coverage.rs`
asserts it equals exactly the set of `tests/*.rs` that mention a `survival` or
`markov` cfg, so the file count above is deliberately left unstated — read the
list off `ci.yml`, and add to it when you add an endpoint test.

Feature flags belong to the `ferx-core` package, so any command scoped to a
member or to the whole workspace has to write them package-qualified
(`--features ferx-core/ci`); a bare `--features ci` fails with *"none of the
selected packages contains this feature"*. Qualifying them also keeps the
members linking the `ferx-core` rlib the previous step just built rather than
forcing a second compile under a different feature hash.

`rust-toolchain.toml` pins a stock nightly, and `Check`, `Clippy` and `Format`
use it. `Public API baseline` overrides to a *dated* nightly (via the same
`RUSTUP_TOOLCHAIN` mechanism) because its output is rustdoc-derived and would
otherwise drift nightly-to-nightly; the pin lives in
`tools/update-public-api.sh` and the workflow reads it from there. The three coverage jobs deliberately override to **stable** via
`RUSTUP_TOOLCHAIN` — the crate has no nightly-only code, and stable's rustc
version holds for ~6 weeks against nightly's daily bump, which keeps their
instrumented build cache (keyed on rustc version) warm instead of cold-rebuilding
the dependency graph under coverage `RUSTFLAGS` on every run.

### Future CI additions

- **Validation**: Run example models and compare output to baselines
- **Docs**: Build the Quarto site and deploy to gh-pages on main branch pushes
- **Security**: `cargo audit` for dependency vulnerabilities
- **Release**: Tag push (`v*`) builds release binaries and creates a GitHub Release

## 9. Deployment & Packaging

### CLI binary

The `ferx` binary is the primary distribution artifact. It reads `.ferx` model files and NONMEM-format CSV data, and outputs:
- `{model}-fit.yaml` — parameter estimates
- `{model}-sdtab.csv` — per-subject diagnostics

### R package integration

An R package (`ferx`) wraps the Rust engine via the [extendr](https://extendr.github.io/) framework, providing an R interface for pharmacometricians who work in R.

### Future packaging considerations

- Pre-built binaries for Linux, macOS, and Windows via GitHub Releases
- Docker image for reproducible environments
- Homebrew formula or cargo-binstall support for easier installation

## 10. Security & Compliance

### Dependency management

- Dependencies are specified in `Cargo.toml` with version constraints.
- Run `cargo audit` periodically to check for known vulnerabilities.
- Run `cargo update` to keep dependencies current within semver bounds.

### License

A license should be added to the repository root (`LICENSE` file) to clarify usage terms.

### Contributing

A `CONTRIBUTING.md` file should be created to document:
- How to set up the development environment (stock Rust nightly)
- Code style expectations
- Pull request process
- How to run validation models

## 11. Development Workflow Summary

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Feature     │     │  Pull        │     │  Main       │
│  Branch      │────▶│  Request     │────▶│  Branch     │
│              │     │  + Review    │     │             │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
      ┌─────────────────────────────────────────┤
      │                                         │
      ▼                                         ▼
┌─────────────┐                          ┌─────────────┐
│  Validate   │                          │  Tag +      │
│  Examples   │                          │  Release    │
└─────────────┘                          └─────────────┘
```

1. **Plan**: Identify the feature or fix, create an issue if applicable.
2. **Branch**: Create `feature/<name>` from `main`.
3. **Develop**: Write code, run `cargo check` and `cargo clippy` iteratively.
4. **Validate**: Run example models against known datasets, compare results.
5. **Review**: Open PR, get code review, address feedback.
6. **Merge**: Squash-merge or merge into `main`.
7. **Release** (when ready): Tag, update changelog, build release artifacts.
8. **Document**: Update Quarto docs and README as needed.
