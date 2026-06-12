# Simulation

## `simulate()`

Generate simulated observations from a model with random effects and residual error.

```rust
pub fn simulate(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
) -> Vec<SimulationResult>
```

**Parameters:**
- `model`: Compiled model
- `population`: Template population (dose events and observation times are used; DV values are ignored)
- `params`: True parameter values for simulation
- `n_sim`: Number of simulation replicates

**Returns:** Vector of `SimulationResult`, one per observation per subject per replicate.

**Example:**
```rust
let model = parse_model_file(Path::new("model.ferx"))?;
let population = read_nonmem_csv(Path::new("data.csv"), None)?;

// Simulate 1000 replicates
let sims = simulate(&model, &population, &model.default_params, 1000);

for sim in &sims[..5] {
    println!("Sim {}, ID {}, TIME {}, IPRED {:.3}, DV {:.3}",
             sim.sim, sim.id, sim.time, sim.ipred, sim.dv_sim);
}
```

## `simulate_with_seed()`

Same as `simulate()` but with a fixed random seed for reproducibility.

```rust
pub fn simulate_with_seed(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
    seed: u64,
) -> Vec<SimulationResult>
```

## `simulate_with_options()` — propensity-score matching

Same outputs as `simulate()`, with options for a seed and for **propensity-score
matching** of the simulated random effects.

```rust
pub fn simulate_with_options(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
    opts: &SimulateOptions,
) -> Result<Vec<SimulationResult>, String>

pub struct SimulateOptions {
    pub seed: Option<u64>,        // None draws from entropy
    pub propensity_match: bool,   // default false
}
```

With `propensity_match = false` this is identical to `simulate_with_seed()` (or
`simulate()` when `seed` is `None`).

### Why match?

In real-world data, therapy is often **adapted in response to a patient's own
PK** — e.g. high-clearance patients are given longer dosing intervals or more
frequent sampling. A standard VPC draws each subject's `eta` *independently* of
its dosing/sampling design, so this design↔eta association is lost and the VPC
can show spurious model misspecification even when the model is correct.

Propensity-score matching restores the association. For each replicate:

1. Draw a pool of `N` etas from \\( N(0, \Omega) \\) (`N` = number of subjects).
2. **Optimally** match the `N` drawn etas 1:1 (without replacement) to the `N`
   subjects' **fitted (posthoc) etas**, minimising the total **Mahalanobis**
   distance under the model \\( \Omega \\):
   \\( d^2(a,b) = (a-b)^\top \Omega^{-1} (a-b) \\)
   (a linear assignment problem; mirrors `MatchIt(method = "optimal",
   distance = "mahalanobis")`).
3. Each subject keeps its own observed design but is simulated with the drawn
   eta that matched its fitted eta — so a high-clearance draw lands on a subject
   whose adaptive design reflects high clearance.

The fitted etas are computed once via a posthoc (MAP) inner-loop pass over the
observed `population` at `params`; only the drawn pool and the matching change
across replicates.

**Requires observed data**: every subject must carry observations (so its
posthoc eta is defined). The function returns `Err` if the population is empty
or any subject has no observations. The matching is the only correction applied;
binning and plotting the VPC is left to the caller (e.g. the `vpc` R package).

### Validation against R

The matching numerics were cross-checked against R on a fixed 8-subject,
2-eta scenario:

- the Mahalanobis cost matrix matches R's `mahalanobis()` under Ω to `2e-11`;
- ferx-core's optimal assignment is **identical** to two independent optimal
  solvers — `clue::solve_LSAP` and `optmatch::pairmatch` (the engine
  `MatchIt(method = "optimal")` uses) — with the same total cost to 10 dp.

A direct `MatchIt(method = "optimal", distance = "mahalanobis")` call (as in the
PAGE-poster workflow) returns a slightly different *pairing*, because MatchIt
scales the Mahalanobis metric by the *empirical* covariance of the combined
etas rather than the model Ω. The optimal-matching algorithm is the same
(optmatch); only the metric differs, which is the documented modelling choice
here (Ω is exact for draws from `N(0, Ω)` and needs no extra estimation).

**Example:**
```rust
let pop = read_nonmem_csv(Path::new("rwd.csv"), None)?;   // observed data
let fit = fit(&model, &pop, &model.default_params, &opts)?;

// Recover the fitted ModelParameters (theta/Omega/sigma) from the result.
let params = fitted_params_from_result(&fit, &model);
let sims = simulate_with_options(
    &model, &pop, &params, 200,
    &SimulateOptions { seed: Some(42), propensity_match: true },
)?;
// → 200 replicates; build the pmVPC from `sims` as usual.
```

## `predict()`

Population predictions without random effects (eta = 0). No simulation noise is added.

```rust
pub fn predict(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
) -> Vec<PredictionResult>
```

**Returns:** Vector of `PredictionResult` with population-level predictions.

**Example:**
```rust
let preds = predict(&model, &population, &model.default_params);
for p in &preds {
    println!("ID {}, TIME {}, PRED {:.3}", p.id, p.time, p.pred);
}
```

## `simulate_with_uncertainty()`

Simulate observations while propagating **population parameter uncertainty**.
For each parameter draw from the uncertainty distribution, etas (from the
drawn Omega) and epsilons (from the drawn Sigma) are sampled — giving
simulation bands that include both individual variability *and* uncertainty
in the population estimates.

```rust
pub fn simulate_with_uncertainty(
    model: &CompiledModel,
    population: &Population,
    fit_result: &FitResult,
    opts: &SimulateUncertaintyOptions,
) -> Result<Vec<SimulationResult>, String>
```

```rust
pub struct SimulateUncertaintyOptions {
    pub n_uncertainty_draws: usize,      // parameter sets drawn
    pub n_sim_per_draw: usize,           // eta/eps replicates per draw
    pub method: UncertaintyMethod,       // Asymptotic | Sir
    pub seed: Option<u64>,
}

pub enum UncertaintyMethod {
    Asymptotic,   // MVN around ML estimate using FitResult.covariance_matrix
    Sir,          // resamples from FitResult.sir_resamples_packed
}
```

**Parameters:**
- `fit_result`: Result of a previous `fit()` call. Must include
  `covariance_matrix` (Asymptotic) or `sir_resamples_packed` (SIR).
- `opts.n_uncertainty_draws`: number of parameter sets drawn.
- `opts.n_sim_per_draw`: number of subject-level replicates per draw.
- `opts.method`: how to draw the parameter sets.

**Returns:** `Vec<SimulationResult>` of length `n_uncertainty_draws *
n_sim_per_draw * n_subjects * n_obs`. Each row carries a `draw` index
(1..=n_uncertainty_draws) and a `sim` index (1..=n_sim_per_draw).

**Prerequisites:**
- For `UncertaintyMethod::Asymptotic`: run `fit()` with
  `run_covariance_step = true` (or `covariance = true` in `[fit_options]`).
- For `UncertaintyMethod::Sir`: run with `sir = true` *and*
  `sir_keep_samples = true` so the resampled vectors are kept on the
  `FitResult`.

**Example:**
```rust
let fit_opts = FitOptions {
    run_covariance_step: true,
    ..FitOptions::default()
};
let fit = fit(&model, &population, &model.default_params, &fit_opts)?;

let sims = simulate_with_uncertainty(
    &model,
    &population,
    &fit,
    &SimulateUncertaintyOptions {
        n_uncertainty_draws: 200,
        n_sim_per_draw: 10,
        method: UncertaintyMethod::Asymptotic,
        seed: Some(42),
    },
)?;
```

## Result Types

```rust
pub struct SimulationResult {
    pub draw: usize,    // Uncertainty draw index (1-indexed; always 1 for simulate())
    pub sim: usize,     // Replicate number within a draw (1-indexed)
    pub id: String,     // Subject ID
    pub time: f64,      // Observation time
    pub ipred: f64,     // Individual prediction (no residual error)
    pub dv_sim: f64,    // Simulated observation (with residual error)
}

pub struct PredictionResult {
    pub id: String,
    pub time: f64,
    pub pred: f64,      // Population prediction (eta = 0)
}
```

## Simulation Process

For each replicate and each subject:

1. Sample random effects: \\( \eta_i \sim N(0, \Omega) \\) using the Cholesky factor \\( L \\): \\( \eta = L \cdot z \\), where \\( z \sim N(0, I) \\)
2. Compute individual PK parameters via `pk_param_fn(theta, eta, covariates)`
3. Generate predictions using the structural model
4. Add residual error: \\( DV = IPRED + \sqrt{V} \cdot \epsilon \\), where \\( \epsilon \sim N(0, 1) \\) and \\( V \\) is the residual variance from the error model

## Simulation with Uncertainty Process

`simulate_with_uncertainty()` wraps the steps above in an outer loop over
parameter draws:

1. **Outer loop** — for each of `n_uncertainty_draws`, draw a population
   parameter set \\( (\theta_k, \Omega_k, \Sigma_k) \\) from the uncertainty
   distribution.
   - **Asymptotic**: \\( x_k = \hat{x} + L_\text{cov} z \\) in packed log-space
     (theta, Cholesky-omega, sigma share one packed vector), then unpack —
     theta, Omega, and Sigma are perturbed coherently from a single MVN draw.
   - **SIR**: pick a parameter vector at random from
     `FitResult.sir_resamples_packed` (the resampled pool retained from the
     SIR step).
   Draws that fall outside the parameter bounds or yield non-positive
   theta/omega/sigma are rejected and resampled (up to `10 ×
   n_uncertainty_draws` attempts).

2. **Inner loop** — for the drawn \\( (\theta_k, \Omega_k, \Sigma_k) \\), run
   `n_sim_per_draw` replicates of the standard simulation process above.
   Etas are drawn from \\( N(0, \Omega_k) \\) and epsilons from the drawn
   \\( \Sigma_k \\).

The resulting `SimulationResult` rows are tagged with both `draw` and `sim`
so downstream code can compute either marginal bands (over all draws and
sims) or hierarchical bands (e.g. median across sims within each draw, then
percentiles across draws).
