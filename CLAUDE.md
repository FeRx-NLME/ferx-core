# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ferx-core is a Rust-based Nonlinear Mixed Effects (NLME) modeling engine for population pharmacokinetics. It implements FOCE/FOCEI estimation methods, similar to NONMEM, with analytical PK solutions and an optional ODE solver.

## Sibling repositories

The R wrapper package lives at `../ferx-r` (sibling directory). The R package's Rust glue depends on `ferx-core` via git, but its `src/rust/.cargo/config.toml` carries a `[patch]` that auto-swaps in `../../../ferx-core` (this repo) when the sibling checkout exists. So **local** R-package builds pick up changes here automatically — no Cargo.toml edits needed on either side. When a change to a `pub` API in `ferx-core` lands, expect to follow up with a matching PR in `ferx-r`.

Note that ferx-r's **CI** does not get the change automatically: it builds from the ferx-core commit pinned in `ferx-r/src/rust/Cargo.lock` (the patch only applies locally). A ferx-r PR that needs a new ferx-core commit — e.g. a newly-`pub` API — must bump that lock, via `ferx-r/tools/update-ferx-core-lock.sh` (never a bare `cargo update`, which the patch will unpin). Otherwise CI fails with `error[E0603]: ... is private`.

### Downstream repos pin ferx-core *transitively* — there is no ferx-core SHA to hand out

`ferxtranslate` (and anything else built on the R package) does not depend on ferx-core directly: it pins **ferx-r** at a commit, and that ferx-r commit's `src/rust/Cargo.lock` pins ferx-core. So a merge here is **not** reachable downstream until two bumps land, in order — ferx-r's `Cargo.lock`, then the downstream repo's ferx-r pin. Do not tell a downstream maintainer to "pin ferx-core at `<sha>`"; there is no such knob on their side.

The trap that makes this easy to get wrong: reading `ferx-r/src/rust/Cargo.toml` alone shows `branch = "main"` and reads as *unpinned*. **The lock is the pin.** Check it, don't infer it:

```bash
git -C ../ferx-r show origin/main:src/rust/Cargo.lock | grep -A2 'name = "ferx-core"'
```

When a change here matters to a downstream consumer, tell them the two-step and the current lock rev — not a ferx-core SHA.

## Worktree isolation

When working on a feature branch or any branch other than `main`, always use `EnterWorktree` at the start of the session. This prevents uncommitted WIP from one session contaminating another session on a different branch (a real problem when two chats share the same checkout directory).

## Workspace layout and the public-API boundary

The repo is a cargo **workspace** whose root package is `ferx-core` itself:

```
ferx-core/                       # repo root = ferx-core package + workspace root
├─ Cargo.toml                    # [package] ferx-core + [workspace] members
├─ src/                          # the engine
├─ api/ferx-core-public-api.txt  # public-API baseline, diffed in CI
└─ crates/
   ├─ ferx-tools/                # depends on ferx-core
   ├─ ferx-cli/                  # depends on ferx-core; [[bin]] name = "ferx"
   └─ docs-lint/                 # dev tooling; depends on NOTHING (see Documentation)
```

`docs-lint` sits outside the layering above: it is a structural linter for
`docs/**/*.qmd`, it depends on no crate in the workspace (not even `ferx-core`),
and nothing may depend on it. Keeping it dependency-free is what lets its CI job
be seconds of compile instead of landing on the `ferx-core` compile path (#969).

**The root package must stay `ferx-core`.** `ferx-r/src/rust/.cargo/config.toml`
patches `ferx-core = { path = "../../../ferx-core" }` — the repo *root*. Moving
the package under `crates/` breaks every local ferx-r build, and breaks it
*silently*: the patch stops applying and ferx-r quietly builds `main` instead of
your branch.

**Boundary rule.** `ferx-core` = one model, one dataset, one fit. `ferx-tools` =
many fits, resampling, model-space search. If it calls `fit()` more than once, it
is a tool. Statistical kernels (npde, CWRES, shrinkage, covariance/SE, SIR, FREM,
VPC statistics) stay in core; orchestration of repeated fits goes to `ferx-tools`.
`ferx-cli` is thin — argument parsing, printing, exit codes, no numerical or
orchestration logic. The dependency is strictly one-way: `ferx-core` never
depends on either member (pinned by `tests/public_api_boundary.rs`).

**Widening the public API is a design step, not paperwork.** `api/ferx-core-public-api.txt`
is a committed `cargo public-api` snapshot; the `Public API baseline` CI job
regenerates and diffs it, so any new `pub` item fails CI until the baseline is
updated **in the same PR**. When a change genuinely needs a wider surface:

```bash
tools/update-public-api.sh          # regenerate the baseline
tools/update-public-api.sh --check  # what CI runs
```

and say in the PR description which tool needs each added item and why the
existing surface does not suffice. Two rules make this stick:

- **No `#[doc(hidden)] pub` escape hatch.** `cargo public-api` omits
  `#[doc(hidden)]` items entirely, so the attribute is a *working bypass* of the
  gate — which is why `tests/public_api_boundary.rs` bans it outright. An item is
  either public API (documented, in the baseline, usable from ferx-r) or it stays
  `pub(crate)` and the caller does without.
- **The ferx-r reachability test.** Anything `ferx-tools` needs must also be
  reachable from ferx-r, which consumes `ferx-core` as an ordinary external
  crate. An API that only makes sense "because we're in the same workspace" is
  the wrong API.

**Feature flags belong to `ferx-core`, not to the members**, so a workspace-wide
or member-scoped command has to write them package-qualified — `--features
ferx-core/ci`, not `--features ci` (which fails with "none of the selected
packages contains this feature"). Qualifying them also keeps the members linking
the `ferx-core` rlib the other CI steps just built, instead of forcing a second
compile under a different feature hash.

## First-time setup

After cloning, activate the shared pre-commit hook (blocks commits that fail `rustfmt`):

```bash
git config core.hooksPath .githooks
```

## Before you push: `tools/preflight.sh`

**A green test suite is not a green build.** The CI feature sets are *not nested*:
`cargo test --features ci,survival` compiles neither the `nn`-gated nor the
`slow-tests`-gated source, so a file behind those cfgs can be broken while the
entire suite passes. On #1133 exactly that happened — one struct literal in
`src/estimation/nn_theta_gradient_tests.rs` turned three CI jobs red after a
local run of 158 binaries / 4634 tests reported clean.

So run the fast gates before pushing:

```bash
tools/preflight.sh            # fmt, the 5 cargo check feature sets, clippy, rustdoc, docs, public-API
tools/preflight.sh check      # just the check matrix, for a tight edit loop
tools/preflight.sh --list     # show the commands without running them
```

`.github/workflows/ci.yml` invokes this same script for its `Check`, `Clippy`,
`Format`, `Rustdoc` and `Docs lint` jobs, and `.githooks/pre-commit` calls its `fmt` group, so the
local gate, the hook and the CI gate cannot drift — the same contract
`tools/update-public-api.sh` uses for the API baseline. **Add a new gate to the
script, not to the workflow**; groups are enumerated once, in the `ALL_GROUPS`
array. On failure it names the CI job that would have gone red.

It does **not** cover the test jobs. `cargo check --tests` compiles the test
targets and runs nothing, so `Tests + coverage (core)` can still go red after a
green preflight. This gates compilation and lint, not behaviour. The one
exception is the `docs` group, which *runs* `cargo test -p docs-lint`: that
crate's gate is its test run — a filesystem walk over `docs/`, not a fit — and
it also carries `cargo clippy -p docs-lint`, since the `clippy` group is scoped
to `ferx-core` and its two library members and would leave it unlinted.

`rustdoc` and `docs` are two different gates and the names are easy to swap.
`rustdoc` denies rustdoc's own warnings on `src/**` doc comments — an intra-doc
link that does not resolve, or that resolves to a `pub(crate)` item, renders in
the HTML with its markdown brackets intact (`[<code>name</code>]`) and in
rust-analyzer hover the same way. Fix those at the link: inline code for an
internal target, a resolvable path (`[`crate::run_covariance`]`) for a public
one. Never with a crate-level `#![allow(rustdoc::...)]` in `src/lib.rs` — that
switches the lint off for every future link too, and
`tests/preflight_owns_the_fast_gates.rs` fails if one appears. `docs` is the
structural linter on `docs/**/*.qmd` described below.

Two traps it exists to cover: clippy runs `--all-targets` (without it, every
`#[cfg(test)]` module and all of `tests/` goes unlinted — #1023), and the
workspace members need **package-qualified** features (`ferx-core/ci`, not `ci`,
which fails outright — #1114).

`tests/preflight_owns_the_fast_gates.rs` pins the whole arrangement, and the
invariant worth knowing when editing the script is that **`run` exits rather
than returns** on failure. That is not style. The first version returned a
status through a group function and a `case … esac || { …; exit 1; }`, and bash
suspends `errexit` for every command of an AND-OR list but the last — then
carries that suspension into the called function's whole body. Four of the five
`cargo check` lines could fail with the script printing `preflight OK` and
exiting 0. The guard now fails each command position in turn behind a fake
`cargo` and asserts a non-zero exit, so a gate that stops gating is a red test.

## Build & Run Commands

```bash
# Build the library (debug)
cargo build

# Build everything, including the `ferx` binary (release, with fat LTO).
# The repo is a workspace: the root package is the `ferx-core` LIBRARY and no
# longer carries a binary, so a bare `cargo build --release` produces no `ferx`.
cargo build --release --workspace          # or: -p ferx-cli

# Run CLI with data file
cargo run --release -p ferx-cli -- examples/warfarin.ferx --data data/warfarin.csv

# Run CLI with simulated data
cargo run --release -p ferx-cli -- examples/warfarin.ferx --simulate

# Check compilation without building
cargo check

# Run clippy lints
cargo clippy
```

The binary is called `ferx` and outputs `{model}-fit.yaml` (estimates) and `{model}-sdtab.csv` (per-subject diagnostics).

## Tests

There are three tiers of tests. Put a new test in the lowest tier whose constraints it fits.

**Tier 1 — Fast unit tests** (`#[cfg(test)] mod tests { ... }` in `src/**/*.rs`)
Test the smallest helper that isolates the behaviour; avoid calling `fit()`. Run with `cargo test --lib`. These run on every PR and must stay fast (seconds total).

The test module is inline in most files, but in the largest ones it lives in a **sibling `#[path]` file** so the production source stays navigable: the parent declares `#[cfg(test)] #[path = "<file>_tests.rs"] mod tests;` and the body sits in `src/.../<file>_tests.rs` (or `<file>_<modname>.rs` when a file has several test modules; api's siblings live under `src/api/tests/`). The module is still a child of the parent, so `super::…` and cross-module `crate::<mod>::test_helpers` paths resolve unchanged, and the test's fully-qualified name is identical to the inline form. **Add new tests to the sibling when one exists.** A bare module-scope `#[cfg(test)]` helper (a `fn`, `thread_local!`, or `use`) stays in the parent — only the `mod` blocks move — so `super::` in the sibling still finds it.

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

**Every new feature requires a comparison with NONMEM output.** When adding an estimator, error model, structural model, or any behaviour that produces numerical results (estimates, OFV, residuals, diagnostics), validate it against equivalent NONMEM output and include the comparison — either in the feature's docs page (e.g. `docs/faq.qmd` or the relevant `docs/estimation/*.qmd`) or in the PR description.

> **Exception — adaptive / feedback dosing simulations (epic #391) do not use NONMEM.** NONMEM has no native feedback dosing (a controller reading simulated state to choose the next dose), so there is no equivalent NONMEM run to anchor against. Validate these features instead with: (c) a **degenerate oracle** — a controller that re-emits a fixed regimen must equal `simulate()` on that regimen bit-for-bit; the **frozen-schedule replay verifier** — rebuild a static subject from the realized dose ledger and assert bit-equality with the static engine (the exact internal analogue of a NONMEM dose-bookkeeping anchor); and, for genuinely reactive behaviour, (b) reproduction of a published **mrgsolve** dynamic-dosing example (e.g. vancomycin AUC-target TDM, the platelet ladder). mrgsolve — not NONMEM — is the external comparator for this family, since it actually does feedback dosing. See issue #391's Validation section.

> **Exception — survival-likelihood terms with no equivalent NONMEM/survreg estimator.** A few
> recurrent-event objects have no standard tool to anchor against as a single object — e.g.
> recurrent **left truncation** under gap-time (clock-reset) RTTE (#740), which NONMEM / survreg
> / flexsurv have no native estimator for. Validate these instead with: (a) **exact closed
> forms** — Tier-1 unit tests pinning the −log L (or the simulated stream) on hand-computed
> values; (b) **reduction to an already-anchored term** — the delayed-entry first sojourn
> `H(t₁) − H(entry)` *is* the single-event left-truncation contribution that single-event / clock-
> forward TTE already validate against NONMEM, so no new formula is anchored from scratch; and,
> for the simulate side, (c) a **round-trip / degenerate oracle** — `entry = 0` must be
> bit-identical to the non-truncated draw, and a simulated left-truncated stream must refit under
> the same convention it was drawn from. A committed external reference dataset (e.g. flexsurv) is
> added when that tool is available in the environment; its absence does not block the closed-form
> + reduction validation above.

**An oracle — NONMEM anchor, ODE twin, or FD parity — must keep every side of the object under test non-degenerate.** A
single-dose dataset cannot test a dose event's *incoming* side: the state is zero
before a first arrival, so `g(x⁻) = 0` and any error there cancels rather than
showing up. Pair every dose-event anchor with a **multi-dose** case whose later dose
lands with residual drug present, and check that the quantity under test is actually
live on both sides — a covariate that genuinely differs across the boundary, a
covariate on the compartment the dose *lands in* rather than only on downstream ones,
an `init(...)` baseline where a first dose would otherwise start from zero. The same
applies to a fixture asserted against ferx's own predictor: it agrees with a wrong
answer by construction whenever both paths share the convention under test, so the
external reference is what has to see both sides. Two engines are not automatically
two references either — a cross-engine oracle only sees a defect *downstream* of the
point where the engines part. #1079's κ = 0 readout was catchable that way because
each engine applies the readout itself, but the per-occasion snapshot feeding
`ALAG`/`F`/`D{n}`/`R{n}` is built in `predict_iov` *before* the `ode_spec` branch, so
both arms inherit it and an analytic-vs-ODE twin would agree on a wrong one. #1060
shipped a green single-dose anchor next to a 14.9-OFV multi-dose divergence (#1073)
that no fixture could see.
When an anchor does fail, vary **one input at a time** — the pair that differs by a
single number is what localises the defect (`nonmem_anchor/tvcov_lag_saltation*`).

**A green test is not evidence that it can fail.** The rule above is about a fixture that
cannot expose the defect; these are three ways the *assertion* cannot observe it, all three
found on #1166 in tests that had been written, run green and believed:

- **A differential pair must straddle the gate under the OLD behaviour.** A routing oracle
  compared `hazard = h` against `h * (1.0 + 0.0*TIME)` — the exact IEEE identity, so any
  difference is a difference of engine — and asserted bit-identity. Both spellings read `TIME`,
  so both took the same arm before *and* after the fix and agreed either way. Put the twins on
  opposite sides of the predicate, and assert the straddle itself so it cannot silently become
  a tautology again.
- **A fast path can make the mutation unreachable.** "Filter by slot, not by name" was tested
  on a model with no `[event_model]` — but with no injected slots the code short-circuits
  before the filter runs, so a name-prefix mutation sailed through. If the code has an
  `if xs.is_empty()` arm, a fixture with `xs` empty tests that arm, not the logic behind it.
- **`f64::max` / `f64::min` discard `NaN`** (`a.max(NaN) == a`), so `worst = worst.max((got -
  want).abs())` leaves `worst` at `0.0` when `got` is `NaN` and every bound passes. All three
  assertions in a fresh NONMEM anchor had that shape: a regression making the solver return
  `NaN` — the likeliest way to break the thing being anchored — would have gone green. Assert
  `is_finite()` before folding. A *direct* `(got - want).abs() < tol` does catch `NaN`; only
  the fold hides it. `filter_map(…ok())` and `unwrap_or(0.0)` absorb failures the same way.

For each new assertion, name the regression it exists to catch and check that regression can
actually reach it. For a regression test that means mutation; for an anchor, feeding it the
failure it exists to catch.

**Every change to an analytic sensitivity, gradient, marginal, or likelihood path requires a `Dual2`-vs-FD parity test.** The closed-form PK solutions and event-driven propagators are written once as generic `*_g<T: PkNum>` functions; instantiating `T = Dual2<M>` yields the exact `∂f/∂η` / `∂f/∂θ` that FOCE/FOCEI/HMC consume (`sens/`). A wrong sensitivity compiles and runs silently — there is no second copy of the formula to disagree with it — so when you add or modify one of these kernels, or the provider that assembles them, assert it against central finite differences of the `T = f64` production predictor, to tolerance, in a Tier-1 unit test. That parity pins the *derivative* against the value path, not the value path itself — when both share a wrong convention it passes against the exact derivative of the wrong function, which is how #1079 survived. It is an oracle for the gradient only; see the non-degeneracy rule above for what it cannot see. Follow the existing pattern: per-kernel `*_g_dual_matches_fd` checks (`sens/propagate.rs`, `sens/dual2.rs`) and the end-to-end `check_full_provider_vs_fd` harness (`sens/provider_tests.rs`). If a model is outside the analytic scope it must route to FD via the support predicates (`sens_supported` / `analytic_inner_grad_supported_model`); unit-test that routing so a scope gap fails loudly to FD instead of silently returning a wrong gradient. (This is the post-Enzyme successor to the retired `AD↔FD` parity rule — see #285 / #281.)

**Coverage is gated per PR.** A PR's changed lines must carry their own tests — the Codecov `patch` status enforces ≥90% coverage on the diff, and a 90% project floor is enforced on the weekly `main` run (see `codecov.yml`). This is the automated backstop to the rules above; slow-tests never run on PRs, so unit / Tier-2 tests are what register coverage. When excluding code from coverage, **scope `ignore`s by role, not by coverage %**: leave code out for *what it is* — dev-only tooling (e.g. `src/bin/generate_data.rs`), generated code (`build.rs`), or test scaffolding (`tests/`) — never because it reads red. (Feature-gated code that the coverage build doesn't compile reads as "missed" but is a measurement gap, not an ignore target — see #293.)

## Documentation

Docs live in `docs/` as a [Quarto](https://quarto.org/) website:

- `docs/**/*.qmd` — Quarto source pages. **These are the only docs files you edit or commit.**
- `docs/_quarto.yml` — Quarto website configuration and sidebar; new pages must be added here to show up in the site navigation.
- `docs/_site/` — built HTML. **Generated, git-ignored, and never committed.** CI builds and deploys it: the `Deploy Docs` workflow (`.github/workflows/docs.yml`) runs `quarto render docs` and publishes to the `gh-pages` branch on every push to `main` that touches `docs/**`. Do **not** commit rendered output (you may still build locally to preview — the result stays untracked).
- Styling and branding should stay aligned with the main Quarto site in `../ferx-nlme.github.io` (shared `assets/ferx.scss`, `assets/ferx-dark.scss`, `_brand.yml`, logo assets, and `styles.css`).

Any user-visible feature (new fit option, new estimator, new file-format directive, behavioural change) must update the relevant page — typically one of:

- `docs/model-file/fit-options.qmd` for `[fit_options]` keys.
- `docs/model-file/individual-parameters.qmd` for DSL syntax.
- `docs/estimation/*.qmd` for estimator-specific behaviour.
- `docs/faq.qmd` for user-facing explanations / comparisons to NONMEM / nlmixr2.

### The docs are linted at PR time — `crates/docs-lint` (#1163)

`docs/**/*.qmd` has a structural gate that runs on every PR (the `Docs lint` CI
job, via `tools/preflight.sh docs`). Run it yourself after editing a page:

```bash
cargo run -p docs-lint -- --check          # what CI runs
cargo run -p docs-lint -- --update-baseline
```

Four rules, all structural — it checks *shape*, never prose:

| | Rule |
|---|---|
| R1 | a section with no subsections stays under 5,000 characters (~2,000 tokens), so a retrieval hit returns a usable chunk |
| R2 | never skip a heading level (an `h1` followed by an `h3` attaches to the wrong parent) |
| R3 | no two headings on a page may generate the same id |
| R4 | internal links must resolve, **including the `#anchor`** |

Two things to know before you fight it:

- **R4 knows what pandoc actually generates**, which is rarely what a hand-written
  anchor assumes. `## 3. Communication` is `#communication` — the number is
  dropped. `## A — b` is `#a-b`, a *single* hyphen, because the em-dash is
  deleted and the spaces around it collapse. `` ## Modeled duration (`D{n}`) ``
  is `#modeled-duration-dn`. The algorithm and its traps live in
  `docs/development/docs-lint.qmd` and in `crates/docs-lint/src/slug.rs`; read
  one of them before hand-writing an anchor.
- **R1 has a baseline** (`docs/.lint-baseline`) holding the pages that were
  already too long when the gate landed. It is a ratchet: an entry may shrink,
  never grow, and an entry naming a section that no longer exists is itself a
  failure. Do not add new entries — split the section instead. For a chunk that
  genuinely cannot be split, put `<!-- lint-disable R1 -->` on the line above its
  heading **with a reason**, and keep that rare.

`docs-lint` depends on nothing, so it compiles in seconds and never touches the
`ferx-core` build cache. Keep it that way.

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
(CompiledModel, Population) → api/fit.rs:fit() → FitResult
FitResult → io/output.rs → sdtab CSV + fit YAML
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `types.rs` | Core structs: `CompiledModel`, `Population`, `Subject`, `FitResult`, `FitOptions` |
| `api/` | Public API split into a thin `mod.rs` facade + domain submodules: `fit.rs` (`fit`/`fit_inner`), `run.rs` (file/data entrypoints), `simulate.rs`, `adaptive.rs`, `predict.rs`, `postfit.rs` (covariance/SE/shrinkage/warnings), `output_columns.rs`, `pool.rs` (thread pool), `validation.rs` (model/data checks — `check_model_data`, `validate_model_file`, absorption/dosing/survival asserts). `mod.rs` re-exports everything so `crate::api::*` paths and the crate-root `pub use api::{..}` are unchanged |
| `parser/model_parser.rs` | Parses `.ferx` model DSL into `CompiledModel` with closures |
| `pk/` | Analytical 1-cpt and 2-cpt PK solutions (IV, oral, infusion) with superposition |
| `ode/solver.rs` | `Stepper` abstraction + shared integration drivers; the Dormand-Prince RK45 stepper (default method) |
| `ode/rosenbrock.rs` | Linearly implicit Rosenbrock steppers (`rosenbrock23`/`rodas4`/`rodas5p`) for stiff systems — same `Stepper` trait, so every feature works with every method |
| `ode/predictions.rs` | ODE-based predictions with dose event handling |
| `dosing.rs` | Neutral dose-resolution (`resolve_subject_doses`, #324) + SS-equilibration policy (`SS_EQUILIBRATION_CYCLES`, `SsStopTracker`) — the single home shared by pk/ode/sens |
| `estimation/gauss_newton.rs` | Gauss-Newton (BHHH) optimizer with LM damping; pure GN and GN+FOCEI hybrid |
| `estimation/trust_region.rs` | Newton trust-region outer optimizer (argmin + Steihaug CG); FD gradient & Hessian with fixed EBEs |
| `estimation/parameterization.rs` | Pack/unpack optimizer vector (log-theta, Cholesky-omega, log-sigma) |
| `estimation/covariance.rs` | Covariance/SE step: FD-of-OFV Hessian, eigen-floor inverse, score cross-product, non-PD SIR fallback (`compute_covariance`, `run_covariance_step`) |
| `stats/likelihood.rs` | Individual, FOCE, and FOCEI negative log-likelihood computations |
| `stats/residual_error.rs` | Additive, proportional, combined error models; IWRES/CWRES. **Sole owner of the residual-variance-and-derivatives math**: the `residual_variance` primitive, the `ErrorSpec` dispatch layer (`variance_at{,_scaled,_with_correlations}`, `dvar_df{,_scaled}`, `d2var_df2{,_scaled}`, `sigma_loadings`/`_slopes`, …), the `residual_rd`/`residual_rd2` scalar accessors, and the dense-`R`/IWRES consumers. `types.rs` keeps only the `ErrorSpec` data definition + `obs_key`/`obs_keys`. Two association traps: `variance_at` (legacy `(f·σ)·(f·σ)`) is deliberately **not** collapsed into `variance_at_scaled` (`((f·f)·σ)·σ`) — they differ by ~1 ULP and every bare-σ R/OFV/CWRES is pinned to the legacy form; and `ruv_scale` is applied by each **caller**, never folded into `residual_rd` |
| `sens/` | Hand-rolled `Dual2` analytic sensitivities (`∂f/∂η`, `∂f/∂θ`) over the `PkNum` trait — the exact gradients FOCE/FOCEI/HMC use |
| `io/datareader.rs` | NONMEM-format CSV reader (ID, TIME, DV, EVID, AMT, CMT, RATE, MDV, II, SS) |

### Model File Format (.ferx)

Models are defined in a custom DSL with blocks: `[parameters]`, `[individual_parameters]`, `[structural_model]`, `[error_model]`, `[fit_options]`, `[odes]`, `[simulation]`. See `examples/` for reference models. Omega can be diagonal (`omega NAME ~ variance`) or block (`block_omega (NAME1, NAME2) = [lower_triangle]`) for correlated random effects.

### PK Parameter Convention

PK parameters use a fixed-size array `[f64; 8]` with indices: CL=0, V/V1=1, Q=2, V2=3, KA=4, F=5. The fixed layout keeps the closed forms allocation-free, including under the `Dual2` sensitivity type.

### Parameterization

The optimizer works in a transformed space: theta and sigma are log-transformed, omega uses Cholesky factorization. `estimation/parameterization.rs` handles packing/unpacking between the optimizer vector and model parameters.

### Warning and Error Conventions

Warnings and non-fatal issues should be collected into `FitResult.warnings` (a `Vec<String>`), not printed directly to stderr. The CLI layer (`output::print_results`) handles display. This keeps the library quiet for non-verbose callers and ensures warnings appear in both console and YAML output.

### Analytic Sensitivities (`sens/` over `PkNum`)

The exact `∂f/∂η` and `∂f/∂θ` gradients used by FOCE/FOCEI (outer + inner) and the
SAEM/Bayes HMC sampler come from hand-rolled forward sensitivities. The
closed-form PK solutions and event-driven propagators are written **once** as
generic `*_g<T: PkNum>` functions (`sens/`, `pk/event_driven.rs`); instantiating
`T = f64` gives gradient-less predictions and `T = Dual2<M>` gives the
sensitivities. There is no second copy of any formula to keep in sync — edit the
generic version.

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

After reviewing a PR, always post your comments on the PR.
