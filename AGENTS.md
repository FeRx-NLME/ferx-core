# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, Codex, Copilot, and any other tool that reads the [AGENTS.md](https://agents.md) convention) when working with code in this repository.

## Project Overview

ferx-core is a Rust-based Nonlinear Mixed Effects (NLME) modeling engine for population pharmacokinetics/pharmacodynamics, similar to NONMEM. It offers several estimation methods — FOCE/FOCEI (the default family), Laplace/AGQ, SAEM, importance sampling (IMP/IMPMAP), full Bayesian MCMC, and others (`EstimationMethod` in `types.rs`; implementations under `src/estimation/`) — with analytical PK solutions and an ODE solver.

## Code style: parsimony

Prefer the simplest solution that solves the problem.

- Write the least code that does the job correctly. No speculative generality, extra options, or layers that nothing asks for yet.
- Factor shared logic into functions rather than duplicating it. If two code paths compute the same thing, they should call the same function — a second copy will drift (see "two implementations disagree" under Tests). Exception: equal in exact arithmetic is not the same function where float association is pinned (e.g. `variance_at` vs `variance_at_scaled`); read the site comment before merging such a pair.
- Keep functions small and single-purpose; extract a helper when a block *you are already changing* has a nameable job.
- A PR fixes its issue, not the area around it; a neighbouring defect gets its own issue.
- Reuse existing helpers before writing new ones — search the codebase first.
- Delete dead code rather than leaving it commented out or behind a flag nothing enables (feature-gated `nn`/`survival` code is not dead).
- Parsimony governs production code. It is not a reason to drop test assertions, or comments recording a measured number.

## Sibling repositories

The R wrapper package lives at `../ferx-r`. When a sibling `ferx-core` checkout exists, ferx-r's `src/Makevars` patches `ferx-core`, `ferx-tools` and `ferx-search` in from it for that one build, via `tools/sibling-cargo-build.sh` (ferx-r #353). Never hand-write a `[patch]` into ferx-r's `src/rust/.cargo/config.toml`; Makevars truncates that file on every build. When a `pub` API change lands here, expect a matching PR in `ferx-r`.

A "local" ferx-r build may not contain your change:

- It builds whatever is checked out at `../ferx-core`: the main checkout, **not** your `.claude/worktrees/` worktree. (By hand, `FERX_CORE_SIBLING=<abs path>` points the wrapper elsewhere.)
- Run from a ferx-r worktree, it builds the pinned revision unless the `.claude/worktrees/ferx-core` symlink exists there.
- Cargo applies the patch only when the sibling's version **equals** the locked one, so a PR that bumps `version` here makes local ferx-r builds refuse (only one crate matches) or fall back to the pin (neither does) until the lock is bumped.

Read the wrapper's last line: `ferx: the sibling supplied BOTH …` or `ferx: WARNING the sibling was NOT used …`.

ferx-r's **CI** builds from the ferx-core commit pinned in `ferx-r/src/rust/Cargo.lock`. A ferx-r PR that needs a new ferx-core commit must bump that lock via `ferx-r/tools/update-ferx-core-lock.sh` (never a bare `cargo update`, which moves the pin to today's `main` and every registry crate with it); otherwise CI fails with `error[E0603]: ... is private`.

**Downstream repos pin ferx-core transitively.** `ferxtranslate` (and anything built on the R package) pins **ferx-r**, whose `Cargo.lock` pins ferx-core. A merge here reaches them only after two bumps, in order: ferx-r's `Cargo.lock`, then the downstream ferx-r pin. Tell a downstream maintainer that two-step and the current lock rev, never "pin ferx-core at `<sha>`". `ferx-r/src/rust/Cargo.toml` says `branch = "main"`, but **the lock is the pin**:

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
   ├─ ferx-search/               # depends on ferx-tools and ferx-core
   ├─ ferx-cli/                  # depends on all three; [[bin]] name = "ferx"
   ├─ docs-lint/                 # dev tooling; depends on NOTHING
   └─ patterns-lint/             # dev tooling; test-code lints, depends only on `syn`
```

- **The root package must stay `ferx-core`, and `ferx-tools` and `ferx-search` must stay at `crates/ferx-tools` and `crates/ferx-search`.** ferx-r's Makevars looks for all three there; moving any of them breaks its local builds.
- **Boundary rule.** `ferx-core` = one model, one dataset, one fit. `ferx-tools` = many fits of a given model (bootstrap, GAM screening, the `run_in_lanes` fit queue). `ferx-search` = many fits over a space of models (the `.ferxsearch` layer and the search tools). If it calls `fit()` more than once, it is not core. Statistical kernels (npde, CWRES, shrinkage, covariance/SE, SIR, FREM, VPC statistics) stay in core. `ferx-cli` is thin: argument parsing, printing, exit codes. The dependency is strictly one-way, `ferx-search` → `ferx-tools` → `ferx-core`: neither `ferx-tools` nor `ferx-core` ever depends on `ferx-search`, not even as a dev-dependency or through a re-export (#3185), and `ferx-core` never depends on a member (both pinned by `tests/public_api_boundary.rs`). `docs-lint` and `patterns-lint` (source-pattern lints over test code, #2634; rules and baseline in `tools/lint-rules/`) depend on no workspace crate and nothing depends on them, which keeps their CI jobs fast (#969).
- **Licenses follow the boundary** (#3183). `ferx-search` is LGPL-3.0-or-later (it follows Pharmpy's behaviour; see its `NOTICE`); `ferx-core`, `ferx-tools` and the `ferx-cli` source are MIT. This is the second reason the arrow points one way: MIT crates must never depend on `ferx-search`, and code moved from `ferx-search` into an MIT crate needs its authors' agreement. Every `.rs` file under `crates/ferx-search` starts with `// SPDX-License-Identifier: LGPL-3.0-or-later`. A `Cargo.lock` change that adds or bumps a dependency of `ferx-cli` means regenerating `crates/ferx-cli/THIRD-PARTY-NOTICES.txt` with `tools/third-party-notices.py` (`ferx --licenses` prints it). `tools/preflight.sh licenses` checks all of this.
- **Widening the public API is a design step.** Any new `pub` item fails the `Public API baseline` CI job until the baseline is regenerated in the same PR (`tools/update-public-api.sh`, `--check` is what CI runs; #1114). Say in the PR description which tool needs each added item and why the existing surface does not suffice.
  - **No `#[doc(hidden)] pub`.** `cargo public-api` omits such items, so it bypasses the gate; `tests/public_api_boundary.rs` bans it. An item is either public API or `pub(crate)`.
  - Anything `ferx-tools` or `ferx-search` needs must also make sense from ferx-r, which consumes `ferx-core` as an ordinary external crate.
- **Feature flags belong to `ferx-core`**, so workspace-wide or member-scoped commands must package-qualify them: `--features ferx-core/ci`, not `--features ci` (which fails with `none of the selected packages contains this feature`). The one member feature is `ferx-cli/search` (model search, default-on, #3187): a `--no-default-features` command covering `ferx-cli` must name it, or the search subcommands drop out (`tests/preflight_owns_the_fast_gates.rs` enforces it).

## First-time setup

```bash
git config core.hooksPath .githooks   # pre-commit hook: blocks commits that fail rustfmt
```

## Before you push: `tools/preflight.sh`

**A green test suite is not a green build.** CI feature sets are not nested — `cargo test --features ci,survival` compiles neither the `nn`- nor the `slow-tests`-gated source — so run the fast gates before pushing (#1133, #1157):

```bash
tools/preflight.sh            # fmt, the cargo check feature matrix, clippy, rustdoc, docs, patterns, public-API, changelog, licenses
tools/preflight.sh check      # just the check matrix, for a tight edit loop
tools/preflight.sh --list     # show the commands without running them
```

The default run leaves out two opt-in groups that run the unit suite: `tools/preflight.sh debug-assertions` (no CI job of its own; the coverage jobs' `ci-cov` profile covers it, #1248) and `release-semantics` (#1293).

CI's `Check`, `Clippy`, `Format`, `Rustdoc`, `Changelog`, `Licenses`, `Docs lint`, `Patterns lint` and `Tests (release semantics)` jobs and the pre-commit hook call this same script, so they cannot drift. **Add a new gate to the script (`ALL_GROUPS`), not to the workflow.** The default run gates compilation and lint, not behaviour (except `docs` and `patterns`, which run their lint crate's tests) — the test jobs can still go red after a green preflight. `tests/preflight_owns_the_fast_gates.rs` pins the arrangement; when editing the script, keep `run` **exiting** (not returning) on failure, since bash suspends `errexit` inside AND-OR lists.

Things to know:

- Clippy runs `--all-targets -- -Dunused`. `cargo clippy` exits 0 on warnings, so `-Dunused` is what turns a left-behind import/item into a failure (#1470). It is deliberately not `-Dwarnings` (hundreds of style findings, and CI uses a moving nightly). Fix an unused-item error by deleting the item. The gate's limits: `dead_code` never fires on a `pub` item; under `--all-targets` a test-only reader counts as a use; an unused item inside a `slow-tests` body is not compiled. `-Dunused` also promotes `unused_macro_rules`. "Used" is per feature set: an item read only from a `#[cfg(feature = "survival")]` body is dead in the default build ferx-r ships, so the group also lints `--lib` alone under `ci`, where a test-only reader does not count (#2075) — gate such an item on the same feature as its reader.
- `rustdoc` (doc comments in `src/**`) and `docs` (the `.qmd` linter) are different gates. Fix an intra-doc link that is broken **or resolves to a `pub(crate)` item** at the link — inline code for an internal target, a resolvable path for a public one — never with a crate-level `#![allow(rustdoc::...)]`.

## Build & Run Commands

```bash
cargo build                                   # library (debug)
cargo build --release --workspace             # or: -p ferx-cli. A bare `cargo build --release` builds no `ferx`
cargo run --release -p ferx-cli -- examples/warfarin.ferx --data data/warfarin.csv
cargo run --release -p ferx-cli -- examples/warfarin.ferx --simulate
```

The binary is called `ferx` and outputs `{model}-fit.yaml` (estimates) and `{model}-sdtab.csv` (per-subject diagnostics).

## Tests

Put a new test in the lowest tier whose constraints it fits.

**Tier 1 — Fast unit tests** (`#[cfg(test)]` modules in `src/**/*.rs`). Test the smallest helper that isolates the behaviour; avoid calling `fit()`. Run with `cargo test --lib`; must stay fast. In the largest files the test module lives in a sibling file (`#[cfg(test)] #[path = "<file>_tests.rs"] mod tests;`) — **add new tests to the sibling when one exists.** api's siblings live under `src/api/tests/`. A module-scope `#[cfg(test)]` helper stays in the parent.

**Tier 2 — Integration tests** (`tests/*.rs`). Call the public API but return immediately (`Ok` after a handful of outer iterations, or `Err`). No convergence loops. Run on every PR (coverage shards) and nightly.

**Tier 3 — Slow convergence tests.** Full fits to convergence, gated:

```rust
#[test]
#[cfg_attr(not(feature = "slow-tests"), ignore = "slow: opt in with --features slow-tests")]
fn test_my_new_estimator() { ... }
```

A test that calls `fit()` and expects an immediate `Err` needs no gate.

**The gate marks a fit to convergence** (#1132). An anchor that evaluates once (a NONMEM objective at `maxiter = 0`, a frozen mrgsolve table) is Tier 2 and is not gated by default, however slow its oracle was to produce; `tests/anchor_gating.rs` enforces this for the listed anchors. The one reason to gate it anyway is measured CI cost **plus** a named per-PR test that dies on the same mutation; `anchor_gating.rs` records the anchors kept nightly this way. Measure in `Tests + coverage (core)`: local `ci-cov` times ran 6–22× off.

**A reconverged-FD case table is a shard floor** (#2786). Each `fd_oracle` / `parity` cell can cost 60–120 s and libtest runs a test on one thread, so wrap such tables in `oracle_cells(cases, per_pr)` (`sens/ode_adjoint_tests.rs`): all cells under `slow-tests`, per PR only the cells that catch the named defect. Time new oracles with `-- -Zunstable-options --report-time`. Smoke fits that assert only a finite objective need a few outer iterations, not 60.

**Every new feature requires a test**, and bug fixes require a regression test that fails without the fix.

**Every new feature that produces numerical results requires a comparison with NONMEM output** (estimates, OFV, residuals, diagnostics), in the feature's docs page or the PR description. A control stream committed under `nonmem_anchor/` gets a row in `nonmem_anchor/README.md`'s catalogue table in the same PR (`tests/nonmem_anchor_index.rs` enforces it, #1024). Exceptions, only when there is genuinely no NONMEM spelling of the quantity (not when the construction is merely hard):

- *Adaptive / feedback dosing (#391):* NONMEM has no feedback dosing. Use a degenerate oracle (a controller re-emitting a fixed regimen equals `simulate()` bit-for-bit), the frozen-schedule replay verifier, and reproduction of a published **mrgsolve** dynamic-dosing example.
- *Survival terms with no NONMEM/survreg estimator* (e.g. recurrent left truncation under gap-time RTTE, #740): use exact hand-computed closed forms and reduction to an already-anchored term. On the simulate side, `entry = 0` must be bit-identical to the non-truncated draw, and a simulated left-truncated stream must refit under the convention it was drawn from.
- *Hazard times at or before a subject's first record* (#1223): use exact hand-computed values, agreement across every production caller (asserted per caller), and a differential pair straddling the boundary. Callers routed to the same engine are not independent geometries; the PR says which is which. A merely awkward anchor is still required (`ss_chz_r5_drug_lag`, #1220).

### Oracle and assertion rules

- **Keep every side of the object under test non-degenerate.** A single-dose dataset cannot test a dose event's incoming side (state is zero before the first dose; #1060, #1073). Pair dose-event anchors with a multi-dose case, and make sure the quantity under test actually varies across the boundary: put a covariate on the compartment the dose *lands in*, and use an `init(...)` baseline where a first dose would start from zero. A fixture asserted against ferx's own predictor, or two ferx engines sharing upstream code, can agree on a wrong answer (#1079), so the *external* reference must see both sides.
- **When an anchor fails, vary one input at a time.** The pair differing by a single number localises the defect (`nonmem_anchor/tvcov_lag_saltation*`).
- **The oracle has to be more accurate than the difference you are about to call a defect** (#1124, #1130). An ODE twin at default tolerances is not automatically an oracle (relative accuracy degrades as a state approaches `abstol`). Gate the comparison on the condition for the oracle to outrank the disagreement, and remember that swapping oracles changes the experiment (#1171). When unsure which side is wrong, compute the quantity a third way outside both engines.
- **Measure tolerances; never justify them with a story** (PR #1174). Print the worst realised error, put it in the comment, choose the bound from it with stated headroom. A bound far looser than the realised error is a test that has stopped testing.
- **Linux is the reference platform for fit numbers.** macOS fits differ via the OS math library, not the CPU (#1688). Never compare a macOS number to a Linux one or widen a band to cover macOS; a macOS-only red is a question about the test's identifiability (#1551, #1671). Get Linux numbers with `tools/linux-test.sh -- <cargo args>` (prints a `ferx-platform:` marker line) or `gh workflow run slow-tests.yml --ref <branch> -f nocapture=true` (prints the output of *passing* tests too).
- **When two implementations disagree, the fix is one implementation**, shared by both callers (#1223). If two callers genuinely cannot share, record the asymmetry at the call site.
- **A green test is not evidence that it can fail.** Common holes:
  - A differential pair must straddle the gate under the *old* behaviour (#1166); assert the straddle itself.
  - A fast path (`if xs.is_empty()`) can make the mutation unreachable.
  - `f64::max`/`f64::min` discard `NaN`, as do `filter_map(…ok())` and `unwrap_or(0.0)`; assert `is_finite()` before folding. This is linted in test code (`nan-fold`, `tools/preflight.sh patterns`, #2634): use `// nan-ok: <reason>` for a justified exception, and never add to the shrink-only `tools/lint-rules/baseline.txt`.
  - `is_finite()` cannot see a sentinel (e.g. `1e20` for a repelled subject); bound the magnitude instead (#1223).
  - Two redundant gates cover for each other: when both reject the same inputs, deleting either stays green (#1229, #1311). Collapse to one. A predicate that cannot change the answer for any input that reaches it is the degenerate case — delete it.
  - A shared branch serving several inputs needs an assertion on what makes them different (a keyword, a name) **and its negative**, e.g. ``contains("`block_kappa`") && !contains("block_omega")`` (#1229). Then mutate: swapping the discriminator must redden that input's test *and only* that one.
- **Name the regression each assertion catches, and check it can reach it** — by mutation for a regression test; for an anchor, by feeding it the failure it exists to catch. Ask what the *smallest* edit removing the fix is and which test dies (#1171); mutate each side of a twin separately (#1223). The `Nightly mutants` workflow (`mutants.yml`, #2633) lists surviving mutants from each day's merged `src/` diff — a backstop for the hand check, not a substitute (`--lib`-only, never mutates string literals).
- **A user-facing message is code** (#1405, PR #1456). Enumerate the message's input space *before* writing its prose; delete each sentence in turn and name the test that dies; and where a claim is conditional, assert both sides of its gate in one test, so a gate stuck on either branch reddens it.

**Every change to an analytic sensitivity, gradient, marginal, or likelihood path requires a dual-vs-FD parity test** (Tier 1), against central finite differences of the `T = f64` production objective (EBE re-found at every point), to a measured tolerance. Follow the reconverged-FD oracles `fd_oracle` / `fd_oracle_agq` (outer gradient), `hessian_gaps_for` (R-matrix) and `objective_gaps` in `sens/ode_adjoint_tests.rs`, plus the per-kernel `*_g_dual_matches_fd` checks (`sens/propagate.rs`). Parity pins the derivative against the value path, not the value path itself. Models or subjects outside the recorders' scope must route to FD via the scope predicates (`adjoint::model_in_scope` / `recorder_in_scope`, `analytic_inner_grad_supported_model`, `cov_width_declines`); unit-test that routing. Check which engine each fixture actually ran on — the `FitResult` FD-fallback warning tells you; a fixture that routes to FD says nothing about the plan path (#1210).

**Coverage is gated per PR**: Codecov `patch` ≥90% on the diff. Slow tests don't run on PRs, so unit / Tier-2 tests provide coverage. Exclude code from coverage by role (dev tooling, generated code, test scaffolding), never because it reads red (#293).

### Third-party output, text and data

Never commit another tool's text: its listings, logs or documentation (#3182). Numbers it produced are fine.

- **Fixtures:** commit our own inputs (control stream, config, data) and the extracted numbers, with the tool version and the command to regenerate them. No NONMEM listings (`.lst`) or PsN logs: they carry license-holder details and the tool's own text. `tools/nonmem-extract.py run.lst` writes the numbers of a listing to `run.nm.json`. Numeric tables (`.ext`, `.phi`, `.tab`, `.cov`) are fine. `tools/preflight.sh patterns` fails on a committed listing.
- **Comments and docs:** cite a paper or name the tool; do not quote or closely paraphrase its manual or source. Say "compatible with", not "verbatim", and only after checking against a named version.
- **Source:** do not read PsN (GPL) source to write code here. Pharmpy (LGPL) source may inform model search only (`crates/ferx-search`, to be relicensed LGPL under #3181). Record the source used in the PR.
- **Data:** record the source and license beside every third-party dataset.

## Documentation

Docs live in `docs/` as a [Quarto](https://quarto.org/) website. Edit and commit only `docs/**/*.qmd` (and `docs/_quarto.yml` to add pages to navigation). `docs/_site/` is generated, git-ignored, and deployed by CI — never commit it. Styling stays aligned with `../ferx-nlme.github.io`.

`docs/llms.txt` is the curated index for language models (built into `llms-full.txt` by `docs/_llms.py`); a new user-facing page worth an agent's attention gets a line there, as an absolute site URL (checked by docs-lint R4, #3073). The user-facing counterpart of this file is `docs/agents.qmd`.

Any user-visible change must update the relevant page — typically `docs/model-file/fit-options.qmd`, `docs/model-file/individual-parameters.qmd`, `docs/estimation/*.qmd`, or `docs/faq.qmd`.

**Docs lint** (`crates/docs-lint`, CI job `Docs lint`, #1163): run `cargo run -p docs-lint -- --check` after editing a page. Rules: R1 a leaf section stays under 5,000 characters; R2 no skipped heading levels; R3 no duplicate heading ids; R4 internal links resolve, including `#anchor`. Pandoc's slugs are rarely what a hand-written anchor assumes (`## 3. Communication` → `#communication`, `## A — b` → `#a-b`); see `docs/development/docs-lint.qmd`. The R1 baseline (`docs/.lint-baseline`) is a ratchet — never add entries; split the section instead. After splitting a baselined section run `--update-baseline`, since a stale entry is itself a failure; use it only to record a shrink. For a chunk that genuinely cannot be split, put `<!-- lint-disable R1 -->` on the line above its heading **with a reason**, rarely.

## Architecture

### Model Pipeline

```
.ferx file → parser/model_parser.rs → CompiledModel
NONMEM CSV  → io/datareader.rs       → Population
(CompiledModel, Population) → api/fit.rs:fit() → FitResult
FitResult → io/output.rs → sdtab CSV + fit YAML
```

### Estimation

FOCE/FOCEI use nested optimization: an **outer loop** (`estimation/outer_optimizer.rs`) over population parameters (theta, omega, sigma) and an **inner loop** (`estimation/inner_optimizer.rs`) finding each subject's EBEs. The optimizer works in a transformed space — log theta/sigma, Cholesky omega (`estimation/parameterization.rs`). Other methods (SAEM, IMP, Bayes, Laplace, …) live in their own files under `estimation/` and reuse the same model, likelihood, and sensitivity machinery. Optimizer selection (`optimizer = auto`) and Gauss-Newton are described in `docs/estimation/optimizers.qmd` and `docs/estimation/gauss-newton.qmd`.

### Key Modules

| Module | Purpose |
|--------|---------|
| `types.rs` | Core structs: `CompiledModel`, `Population`, `Subject`, `FitResult`, `FitOptions`, `EstimationMethod` |
| `api/` | Public API: `fit.rs`, `run.rs`, `simulate.rs`, `predict.rs`, `postfit.rs`, `validation.rs`, …; `mod.rs` re-exports |
| `pool.rs`, `entry_refusals.rs`, `levels.rs`, `covariate_stats.rs`, `occasions.rs` | Crate-level (not in `api/`) so code below `api` can use them without depending on it (#3041) |
| `parser/model_parser.rs` | Parses `.ferx` model DSL into `CompiledModel` |
| `pk/` | Analytical PK solutions with superposition |
| `ode/` | ODE solvers (`solver.rs` RK45 default, `rosenbrock.rs` stiff) and predictions. Every method implements the same `Stepper` trait, so every feature must work with every method |
| `dosing.rs` | Dose resolution and steady-state equilibration, shared by pk/ode/sens |
| `estimation/` | Estimators, covariance/SE step (`covariance.rs`), parameterization |
| `stats/likelihood.rs` | Individual and population negative log-likelihoods |
| `stats/residual_error.rs` | Residual error models and IWRES/CWRES; sole owner of residual-variance math. `variance_at` (`(f·σ)·(f·σ)`) is deliberately not collapsed into `variance_at_scaled` (`((f·f)·σ)·σ`): they differ by ~1 ULP and every bare-σ R/OFV/CWRES is pinned to the former. `ruv_scale` is applied by each **caller**, never inside `residual_rd` |
| `sens/` | `PkNum` trait, generic `*_g<T>` kernels, and the adjoint that records them into compiled per-subject plans: exact gradients, `H` and R-matrix for FOCE/FOCEI/Laplace/AGQ |
| `io/datareader.rs` | NONMEM-format CSV reader |

### Model File Format (.ferx)

Custom DSL with blocks, e.g. `[parameters]`, `[individual_parameters]`, `[structural_model]`, `[error_model]`, `[fit_options]`, `[odes]`, `[simulation]`. The list is closed-world: `BLOCK_REGISTRY` in `parser/model_parser.rs` is the full set, and an unlisted block fails with `E_UNKNOWN_BLOCK` (#1040). See `examples/`.

### Warning and Error Conventions

Warnings and non-fatal issues go into `FitResult.warnings`, never directly to stderr; the CLI handles display.

### Analytic Sensitivities (`sens/` over `PkNum`)

Closed-form PK solutions, event-driven propagators and ODE solvers are written once as generic `*_g<T: PkNum>` functions (`sens/`, `pk/event_driven.rs`, `ode/`). `T = f64` gives predictions; the adjoint (`sens/ode_adjoint.rs`) records the same functions into a Cranelift-compiled per-subject plan, replayed by forward and reverse passes for exact derivatives. `Dual2` and `jet::DualT` remain as plain forward types (test oracles, and local derivatives recorded into a plan; `DualT` is production code). Edit the generic version — there is no second copy.

## Changelog

**In the same PR as any user-facing change, add a fragment** `changelog.d/<N>.<category>.md` — never edit `CHANGELOG.md` directly (#1545). `<N>` is the issue/PR number (`-2`, `-3` suffix for extra entries), `<category>` one of `added`, `changed`, `deprecated`, `removed`, `fixed`, `security`, `performance`. Body: exactly one user-facing bullet with the `#NN` reference (see `changelog.d/README.md`). The gate is path-triggered: any PR touching `src/`, `crates/ferx-cli/`, `crates/ferx-tools/` or `crates/ferx-search/` (test files excluded) fails without a fragment, so a non-user-facing one, such as an internal refactor, takes the **`no-changelog`** label instead. Never delete or rename another PR's fragment. `tools/changelog.sh check` validates. A cross-repo change may also need a ferx-r `changelog.d/` fragment, which has its own categories (`breaking`, `feature`, `fix`, `docs`, `internal`); never edit ferx-r's `NEWS.md`.

## Priority labels

Issues and PRs carry GitHub labels `P0`–`P3` (correctness severity × how common the trigger): `P0` silent wrong results in common use; `P1` wrong results in realistic use, or a loud failure blocking common workflows; `P2` edge-case correctness, important features, UX/diagnostics; `P3` performance, refactors, nice-to-haves, research. The applied label is the source of truth, not a priority written in the title.

## Pull Requests

When creating a PR, read `.github/PULL_REQUEST_TEMPLATE.md` and fill every section before calling `gh pr create`. Its **Tests** section asks for *Same pattern elsewhere* and *Consumers checked* (#2637); use rust-analyzer find-references for the counts.

After reviewing a PR, always post your comments on the PR.
