# `TIME` event-time built-in in a structural parameter — analytic sensitivities — #486 / #610

## Goal

PR #610 added the `TIME`/`time` built-in inside `[individual_parameters]` expressions and direct
`pk(...=TIME)` mappings (NONMEM-style time-dependent PK-parameter switching, e.g.
`CL = TVCL * (TIME < 45 ? 1 : THETA_RB)`). A model that **uses** `TIME` in a structural parameter
routes its FOCE/FOCEI **sensitivity gradient** to finite differences. This threads the per-event
time into the analytic `Dual1`/`Dual2` kernels so those models get analytic gradients.

`TIME` is **data** (an event time): `∂param/∂TIME` is not a parameter sensitivity. Like a
time-varying covariate, it makes the parameter piecewise/time-varying and is threaded as a
**per-event constant**. So the fix mirrors the existing TV-covariate constant-threading.

## Current state (what already exists)

- **The FD gate**: `compiled_model_uses_time_builtin(model)` (`model_parser.rs:9404`) returns
  `indiv_param_program.uses_time_builtin` (flag computed `8983`, stored `8993`, declared `11395`;
  covers both `[individual_parameters]` use via `stmts_use_time_builtin:9394` and the
  `pk(...=TIME)` direct mapping via `pk_time_mapping`). Consulted at the three provider gates —
  `analytical_supported` (`provider.rs:209-212`), `iov_analytical_supported` (`provider.rs:616-622`),
  `ode_analytical_supported` (`ode_provider.rs:242-245`) — which feed `sens_supported`
  (`provider.rs:334`), `iov_sens_supported` (`700`), `analytic_outer_gradient_available` (`360`),
  and the inner gates (`ode_inner_grad_supported:390`, `inner_optimizer.rs:1426`). Gate test
  `time_builtin_indiv_params_force_fd_fallback` (`provider.rs:3493`).
- **The f64 mechanism to mirror** (a thread-local + RAII guard):
  - `MODEL_TIME` thread-local (`model_parser.rs:9152`); `ModelTimeGuard::enter(time)` (panic-safe,
    `9161-9176`); `with_model_time` (`9178`); reader `current_model_time()` (`9183`).
  - The f64 `pk_param_fn` closure sets the guard per evaluation:
    `let _time_guard = uses_time_builtin.then(|| ModelTimeGuard::enter(time));` (`9011`, `time: f64`
    param at `9004`), invoked per-event with each event's own time via `pk_params_at_time`
    (`pk/mod.rs:21→30`) from `compute_event_pk_params_into` (`pk/mod.rs:407`, TV/TIME arm
    `418-445`; non-TIME arm collapses to one `t=0` snapshot `446`).
  - Bytecode `Op::PushTime` (`model_parser.rs:9989`, emitted from `Expression::Time` `10239` and
    the `pk_time_mapping` slot `8847`) resolves via `current_model_time()` in **both** the f64
    evaluator (`10396`) and the generic-`T` evaluator (`10618`, `push!(k(current_model_time()))` —
    it seeds a **constant** dual). So the dual evaluator already *reads* the thread-local; it is
    just never *set* on the dual code paths.
- **The dual eval entry points that must learn `time`** (no `time` arg today):
  `IndivParamProgram::eval_param_duals::<M>` (`model_parser.rs:11434`),
  `eval_param_eta_grad::<M>` (`11542`), cov-static fold `eval_cov_static_f64` (`11506`). Provider
  callers: `pd_from_program` (`ode_provider.rs:1051`), `param_derivatives_at_cov` (`2623`),
  `param_eta_derivatives_from_prog` (`999`), `seed_pk_dual1` (`2647`), `seed_pk_dual2` (`1681`),
  IOV `iov_combined_derivs` (`provider.rs:571→580`).
- **TV-covariate constant-threading to mirror** (per-event `cov` seeds): closed-form non-IOV
  `subject_sensitivities_tvcov` (`provider.rs:1346`) → `run_obs_tvcov` (`1432`), per-event seed
  `mk(cov)` via `dose_cov(k)`/`obs_cov(j)`/`pk_only_cov(m)` (`1504-1512`); inner `1569`/`1653`. ODE
  `run_subject_tvcov` (`ode_provider.rs:1908`), inner `2678`, dedup seed `seed_tvcov_snapshots`
  (`1867`, maps per-event cov over a `seed(cov)` closure `1892-1900`). IOV per-event sources
  (`provider.rs:893-920`, note the hardcoded `0.0` time at `899`/`909`/`916`).
- **Precedent**: the ODE-RHS clock (`rhs_program.uses_time_vars()`, `model_parser.rs:11191`)
  already sets `ModelTimeGuard::enter(...)` inside dual integration (`ode_provider.rs:2690`, `2704`,
  `7227`; `model_parser.rs:11247`). This is the same pattern applied to per-event PK-param seeding
  rather than RHS integration.

## Gap

The dual seeding paths evaluate the indiv-param program with **no `time`** (and the IOV path
hardcodes `0.0`), so `Op::PushTime` reads the default `0.0`. The gate declines analytic to avoid a
wrong gradient.

## Design

Thread a per-event `time` alongside each per-event `cov`, wrapping the dual seed in a
`ModelTimeGuard` (gated on `uses_time_builtin`, exactly like the f64 closure at `9011`):

1. Give `eval_param_duals` / `eval_param_eta_grad` an optional `time` (or set the guard at the
   provider seam). The simplest, lowest-risk approach: pass `time` down and have these set
   `ModelTimeGuard::enter(time)` internally when `uses_time_builtin`, paralleling `9011`.
2. **TV-cov walks** (closed-form `run_obs_tvcov`, ODE `seed_tvcov_snapshots`): at each per-event
   seed, pass that event's time (`subject.obs_times[j]` / `dose times` / `pk_only_times[m]`)
   alongside its `cov`.
3. **IOV walk** (`provider.rs:893-920`): replace the hardcoded `0.0` with the event time — and do
   it for **both** the f64 `pk` value and the duals together (`iov_combined_derivs:580`), or the
   value and derivative will disagree.
4. **Static / non-TV fast paths** keep a single `t=0` seed (mirror the f64 split at
   `pk/mod.rs:446`) so non-TIME models pay nothing. A TIME-using model is, by definition, dynamic —
   it must route through a per-event seed even with no TV covariates (the f64 path already does this
   via `compute_event_pk_params_into`).
5. Relax the gate: drop the `compiled_model_uses_time_builtin` early-returns at `provider.rs:210`,
   `617`, `ode_provider.rs:243` **only** once the per-event seeding is in place and tested. Keep the
   gate as a fallback for any path not yet converted (e.g. if IOV is done in a later PR, keep
   `iov_analytical_supported` declining TIME until then).

## Tests

FD-comparison harnesses (analytic dual provider vs central FD of the production predictor):
- Closest analogue `tvcov_provider_matches_fd_of_production` (`provider.rs:5582`, subject
  `tvcov_subject` `5543`) and `check_full_provider_vs_fd` (`4697`, `df_deta` tol `2e-4` `4728`) — add
  a `TIME`-switch model (e.g. `CL = TVCL*(TIME<45 ? 1 : RB)`) with the same harness.
- ODE: mirror `ode_provider_form_c_per_obs_covariate_matches_production` (`ode_provider.rs:4261`)
  and the TV-cov walk tests (`6453-6534`).
- IOV: `check_iov_provider_vs_fd` (`provider.rs:5962`), `iov_tvcov_provider_matches_fd_of_predict_iov`
  (`7362`).
- Keep `time_builtin_indiv_params_force_fd_fallback` (`provider.rs:3493`) updated — narrow it to the
  paths still on FD, and add a direct-`pk(...=TIME)` mapping twin.
- f64 oracle already exists: `test_event_pk_params_time_builtin_uses_event_times*` (`pk/mod.rs:2067`).

## NONMEM comparison (required by CLAUDE.md)

Fit a model with a NONMEM-style time-dependent θ switch (`IF (TIME.GE.45) CL=...`) in ferx vs
NONMEM `METHOD=1 INTER`; record OFV + parameter + SE agreement in the PR.

## Docs + changelog

- Matrix issue **#486**: flip the **`TIME`/`time` event-time built-in** row's cells and update the
  "Both paths" gap bullet (`143`).
- `docs/`: flip the #610 FD-fallback note. `CHANGELOG.md` `[Unreleased]` → `Added`. ferx-r: no
  follow-up unless a `pub` signature changes.

## Risk / watch-outs

- **Dedup cache key collision** (`seed_tvcov_snapshots:1867`, `snapshot_key:1876`): the cache keys
  only on covariate bits. Two events with identical covariates at different times would wrongly
  share a seed. Incorporate `time` into the key (or bypass the cache) when `uses_time_builtin`.
- **IOV hardcoded `0.0`** (`provider.rs:899/909/916/926`): move the `pk` value AND the duals to the
  event time together.
- **Cov-static fold** (`eval_cov_static_f64:11506`, `cov_static_mask:11378`): a slot reading TIME is
  **not** cov-static — verify the #485 static-mask builder treats `Expression::Time` as dynamic, or
  it gets frozen at one time.
- **TIME contributes zero gradient** (constant dual, `10618`) — correct (no TIME axis); the
  piecewise switch's threshold is a measure-zero kink the analytic path is exact *within* a segment
  but cannot represent across the jump (same as TV-covariate steps — acceptable).
- **Guard nesting**: don't let the indiv-param `ModelTimeGuard` clobber the ODE-RHS clock guard
  already active during integration (`ode_provider.rs:2690`). The guard restores on drop, so correct
  scoping (seed *outside* the integration step) is fine; seeding inside the integration loop would
  shadow the RHS clock.
- **Direct `pk(...=TIME)` mapping** (`8847`) shares the flag — cover it, not just
  `[individual_parameters]`; extend the gate test (`3493`) with a direct-mapping twin.
- **Axis caps** unchanged (TIME adds no axis); confirm test models stay within existing axis counts
  so they exercise the analytic path rather than declining for an unrelated reason.
