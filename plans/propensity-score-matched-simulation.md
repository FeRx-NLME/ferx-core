# Plan: Simulate with propensity-score matching (`match` flag)

## Goal

Add a `match = true|false` (default `false`) argument to the simulation API. When
`true`, each replicate's freshly drawn etas are **reassigned to subjects by
propensity-score matching against the subjects' fitted (posthoc) etas**, instead
of each subject simply using its own freshly drawn eta. This corrects VPC bias
from treatment adaptation (interval/dose/sampling adaptation) in real-world data.

Scope is intentionally narrow: **just the simulation numerics.** No VPC binning,
no plotting, no new output dataset format — the caller (ferx-r / user) handles
the VPC from the returned `Vec<SimulationResult>`.

## Background / what "matched" means

Reference: `~/projects/insightrx/vpc_and_rwd/README.md` and
`R/example_poster_nonmem.R` (PAGE poster, Keizer/Bergstrand/Hughes).

Per replicate `j` (current behaviour draws one eta per subject and simulates on
that subject's own design):

1. Draw `N` etas from `N(0, Ω)` (`N` = number of observed subjects) — the
   "simulated subjects" pool.
2. Match the `N` drawn etas (type 0) 1:1 to the `N` fitted etas (type 1) by
   **optimal matching without replacement** using **Mahalanobis distance** in
   eta-space (R script: `MatchIt::matchit(method="optimal", distance="mahalanobis")`).
3. Each subject `i` (keeping its observed dosing + sampling design) is simulated
   with the drawn eta that matched to subject `i`'s fitted eta — so e.g. a
   high-CL draw lands on a subject whose adaptive design (longer interval)
   reflects high CL.
4. Simulate PK/(PD) + residual error as today.

In ferx, `eta` is already the random-effect deviation (covariate effects live in
the structural model, not in eta), so `fit_result.subjects[i].eta` /
`find_ebe()` output is exactly the "ETA, not EBE" quantity the README wants.

## Design decisions (defaults; flag-able later if needed)

- **Matching metric: Ω-based Mahalanobis**, `d²(a,b) = (a−b)ᵀ Ω⁻¹ (a−b)`.
  Principled (both pools are ~`N(0,Ω)`), deterministic, and free —
  `params.omega.inv` is already precomputed (`OmegaMatrix.inv`). This differs
  slightly from MatchIt's empirical-pooled-covariance default; we use the model
  Ω. (Empirical pooled covariance could be added behind an option later.)
- **Assignment: optimal (global min total cost)** via an in-repo Kuhn–Munkres
  (Hungarian) solver — no new dependency. `N×N`, O(N³) per replicate; fine for
  typical N (a few hundred). Replicates run independently (parallelizable).
- **Match on BSV etas only** (length `n_eta`); IOV kappas excluded from matching.
- **Fitted etas computed once** (they depend only on observed data + params, not
  on the replicate) via the existing batch inner loop; only the drawn pool and
  assignment change per replicate.
- **`match` requires observed data** (subjects must carry `observations` to get
  posthoc etas). The synthetic `[simulation]` block has no observed data, so the
  flag lives on the **programmatic simulate API only**, not the DSL block. Calling
  with `match=true` on a population without observations → `Err`.
- **`match=false` path is left byte-for-byte unchanged** so existing seeded
  simulations / regression tests reproduce exactly (same RNG draw order).

## Implementation

### 1. New module: `src/propensity_match.rs`
- `fn mahalanobis_sq(a: &[f64], b: &[f64], omega_inv: &DMatrix<f64>) -> f64`.
- `fn optimal_assignment(cost: &DMatrix<f64>) -> Vec<usize>` — Kuhn–Munkres on a
  square cost matrix; returns, for each subject (column), the matched drawn-eta
  row (or row→col; pick one orientation and document it).
- `fn match_draws_to_fitted(pool: &[DVector<f64>], fitted: &[DVector<f64>], omega_inv) -> Vec<usize>`
  — builds the cost matrix and returns subject→pool-index assignment.
- Register `mod propensity_match;` in `src/lib.rs`.

### 2. API surface in `src/api.rs`
Introduce `SimulateOptions { seed: Option<u64>, propensity_match: bool }` and a
single entry point, keeping existing functions as non-breaking wrappers:

```rust
pub struct SimulateOptions { pub seed: Option<u64>, pub propensity_match: bool }

pub fn simulate_with_options(
    model: &CompiledModel, population: &Population, params: &ModelParameters,
    n_sim: usize, opts: &SimulateOptions,
) -> Result<Vec<SimulationResult>, String>;

// unchanged behaviour, delegate with propensity_match=false:
pub fn simulate(...) -> Vec<SimulationResult>           // thread_rng
pub fn simulate_with_seed(...) -> Vec<SimulationResult> // seeded
```

(`match` is a Rust keyword, so the field is `propensity_match`; ferx-r exposes it
to R as `match`.)

### 3. Matched path in the simulate core
- When `propensity_match`:
  - Compute fitted etas once: run the existing batch posthoc inner loop
    (`estimation::inner_optimizer::run_inner_loop_warm`) over `population` with
    `params` → `Vec<DVector<f64>>` of length `N` (BSV etas).
  - Error if any subject has zero observations (no posthoc eta) or `N == 0`.
  - Per replicate: draw `N` pool etas (`L·z`), call `match_draws_to_fitted`,
    then for each subject `i` simulate with `pool[assignment[i]]` instead of a
    per-subject fresh draw. Residual-error sampling unchanged.
- When `!propensity_match`: call the existing `simulate_inner_with_draw`
  untouched.

### 4. ferx-r follow-up (separate PR, per CLAUDE.md)
- `simulate_with_options` is a new `pub` API → bump `ferx-r` Cargo.lock via
  `tools/update-ferx-core-lock.sh` and surface a `match` argument on the R
  simulate wrapper. Note in PR.

## Tests (per CLAUDE.md tiers)

- **Tier 1 (unit, `src/propensity_match.rs`)**
  - `optimal_assignment` vs brute-force min-cost permutation for small N
    (e.g. N≤7), including ties and asymmetric costs.
  - `mahalanobis_sq` against hand-computed values; identity-Ω reduces to squared
    Euclidean.
  - Degenerate check: if drawn pool == fitted etas, optimal assignment is the
    identity permutation.
- **Tier 2 (integration, `tests/`)** — `simulate_with_options(match=true)` on a
  tiny 2–3 subject population returns `Ok` with the expected row count and finite
  DVs; `match=true` on a population with no observations returns `Err`.
- **Regression** — `simulate_with_seed` output is unchanged (matched path not
  taken) for a fixed seed.

## NONMEM / external comparison (per CLAUDE.md)

The new numeric kernel is the *matching*, not an estimator. Validate the Rust
optimal-Mahalanobis assignment against the R reference: dump a fixed set of
observed etas + drawn etas + Ω, run `MatchIt::matchit(method="optimal",
distance="mahalanobis")` (and the poster's NONMEM workflow), and confirm the Rust
assignment reproduces the same matched pairs (or same total cost on ties). Record
in the PR description / a short `docs` note.

## Docs & changelog

- `docs` — document the `match` simulate argument (a simulation/FAQ page; add
  to `_quarto.yml` if a new page). Explain the pmVPC use case and that matching
  requires observed designs + fitted etas.
- `CHANGELOG.md` `## [Unreleased]` → `Added`: propensity-score-matched simulation
  (`match`) for VPCs on adaptively-dosed real-world data (`#NN`).

## Files touched

- `src/propensity_match.rs` (new) — Mahalanobis + Hungarian + match orchestration.
- `src/lib.rs` — register module.
- `src/api.rs` — `SimulateOptions`, `simulate_with_options`, matched branch.
- `tests/*.rs` — integration test.
- `docs/...` + `_quarto.yml`, `CHANGELOG.md`.
- (Follow-up) `ferx-r` — lock bump + R `match` argument.

## Open questions for reviewer

- Metric default: Ω-based Mahalanobis (proposed) vs MatchIt-style empirical
  pooled covariance — OK to default to Ω and add empirical later?
- Should `match=true` reuse `fit_result` etas when a `FitResult` is in hand
  (faster) instead of always recomputing posthoc? Current plan recomputes for a
  self-contained signature.
