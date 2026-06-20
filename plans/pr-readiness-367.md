# Plan: get the analytic FOCE/FOCEI epic ready for review (#367 / PR #381)

PR #381 began as the *scoped* analytical FOCE/FOCEI outer gradient and has since
grown into the epic: it now also carries IOV (κ), EVID 3/4 resets, time-varying
covariates, steady-state + TV-cov, constant `ScalarScale` + TV-cov, and (compiled
but gated off) the ODE sensitivity provider. Ron's 10-finding review
(2026-06-17) is **9/10 addressed**. This plan closes the remaining gaps and makes
the branch a clean, reviewable PR.

## State of Ron's review (verified against current code)

| # | Finding | Status |
|---|---------|--------|
| 1 | `two_cpt` `ka≈α` L'Hôpital wrong (sens **and** production) | ✅ fixed in both copies + independent-truth test (8.309) |
| 2 | FOCEI no M3/BLOQ guard | ✅ M3 implemented for FOCE **and** FOCEI |
| 3 | `acos` NaN, no fallback | ✅ `Dual2::acos` floor + finiteness backstop |
| 4 | eta not checked log-normal | ✅ general program-Dual2 chain + `LogNormal` fallback gate |
| 5 | missing `conc.max(0)` clamp | ✅ clamp parity in every provider path |
| 6/7 | `h_matrix`/`reconverge` override, no opt-out | ✅ `reconverge_gradient_interval` honored as escape hatch |
| 8 | predictor step, no clamp | ✅ step clamped to bounds |
| 10 | FD/AD Jacobian computed then discarded | ✅ skipped when analytic Jacobian available |
| 9 | **cleanup: hoist `pk/*` generic, delete `sens/*_cpt.rs`** | ✅ deferred to tracked follow-up **#408** (correctness already fixed in both copies + test) |

Plus Ron's headline ask — a runtime analytic-vs-FD cross-check outside
`#[cfg(test)]` — exists as `FERX_SENS_CHECK=1`.

## A. Blockers (must land before "ready for review")

### A1. Validate the IOV + TV-covariate merge ✅ DONE
Landed: `iov_tvcov_*` provider tests (1/2/3-cpt + EVID=2 breakpoint, vs FD of
`predict_iov`) and `iov_tvcov_packed_gradient_matches_reconverged_fd` (FOCEI +
FOCE outer, vs Richardson reconverged-FD). All green.

The per-event seeding refactor in `subject_sensitivities_iov` / `run_obs_iov`
landed this session (sources carry per-event `(pk, cd, group)`; `pk_only`/EVID=2
events seed κ=0; non-TV path still caches one source per occasion group). All 60
existing IOV tests pass, but the **new combination has no test**. Add, mirroring
the existing IOV harnesses:
- **Provider vs FD of `predict_iov`** (`check_iov_provider_vs_fd`): 1-/2-/3-cpt
  IOV with an allometric WT-on-CL covariate that varies across occasions; one
  case with an EVID=2 covariate breakpoint (exercises the κ=0 `pk_only` source).
- **Outer packed gradient** FOCEI + FOCE vs reconverged-FD on an IOV+TV-cov
  subject (reuse `check`/`precise_ebe`-style harness with stacked η).
- These also cover the refactor's new `pk_only` handling on the **non-TV** IOV
  path (previously a hard bail).

### A2. Changelog + docs consolidation (scope drift) ✅ DONE
Landed: CHANGELOG TV-cov entry now lists SS / constant `ScalarScale` / IOV as
supported and narrows the fallback to lagtime + expression scaling; covariates.md
fallback list updated; iov.md rewritten for the analytic IOV gradient (inner
Jacobian, outer gradient, limitations); stale "1-cpt" comments fixed in
`inner_optimizer.rs` + `outer_optimizer.rs`; consolidated FD-fallback matrix added
above for follow-up PRs.

- **CHANGELOG**: the TV-cov entry now also covers **SS** and constant
  **`ScalarScale`** — drop them from its fallback list (remaining fallbacks:
  dose lagtime, `ExpressionScale`/`PerCmt` scaling). Add an **IOV + TV-cov**
  line once A1 is green. The IOV entry's "fall back" note should drop TV-cov.
- **docs/model-file/covariates.md**: the TV-cov section's fallback list →
  (lagtime, expression scaling) only; IOV no longer excluded.
- **docs/model-file/iov.md**: stale — still says "gradient is computed by finite
  differences (no AD path for IOV)". Update to the analytic IOV gradient (κ,
  block-Ω, gradient-based optimizer requirement).
- **Stale "1-cpt" prose**: fixed in `provider.rs`; still in
  `inner_optimizer.rs:~828` and `outer_optimizer.rs:~1985` ("analytical 1-cpt
  scope") — change to "analytical PK scope".

### A3. Decision on Ron #9 (pk/* dedup) ✅ DONE — filed as #408
The duplicate closed forms (`sens/{one,two,three}_cpt.rs` ↔
`pk/{one,two,three}_compartment.rs`) remain. The #1 bug is the drift risk
realized; it is now fixed in **both** copies with an independent test, so the
*correctness* exposure is closed — only the structural duplication stands.
- **Recommended:** file a tracked follow-up issue ("hoist `pk/*` to
  generic-over-`PkNum`, delete `sens/*_cpt.rs`") and keep this PR focused. The
  hoist touches the hot path and the explicit-kernel relationship; it deserves
  its own PR + benchmark.
- If reviewers insist, do it here — but it is cleanup, not a correctness blocker.

## B. PR hygiene

- **PR description** from `.github/PULL_REQUEST_TEMPLATE.md`: state the (grown)
  scope; a validation table (provider-vs-FD, outer-packed-vs-reconverged-FD,
  Tier-3 convergence, NONMEM cross-checks); a "Ron review resolution" summary;
  and the deferred follow-ups (#9 dedup; TV-cov + lagtime / ExpressionScale;
  ODE re-arm).
- **NONMEM comparisons** (CLAUDE.md requires one per numerical feature): warfarin
  FOCEI/FOCE and IOV (307.8 vs 308.8) exist. TV-cov is currently
  simulated-recovery only — add a klebsiella NONMEM cross-check for a WT-on-CL
  model (FOCEI + FOCE, OFV + θ/Ω/σ) and commit the `.lst`. `NM_PASS` stays in
  env/memory only, never committed.
- **Test sweep**: `cargo test --lib` green; `cargo check --tests` green; run the
  Tier-3 gated tests once locally (`--features slow-tests`: `reset_convergence`,
  `tvcov_convergence`). Confirm `FERX_SENS_CHECK=1` is clean at the optimum on
  warfarin + IOV + TV-cov.
- **Coverage**: changed lines ≥90% (Codecov patch) — the new provider/outer code
  is covered by the unit + outer tests; verify no diff line reads red.
- **rustfmt**: pre-commit hook checks staged files; keep formatting churn scoped
  to touched files (don't whole-crate-fmt unrelated drift).

## Remaining finite-difference fallbacks (follow-up PR surface)

Canonical list of analytic-PK feature combinations that still route to the
finite-difference outer gradient, so follow-up PRs have one place to pick from.
Each is a *transparent* fallback: the fit still runs (correct estimates), only the
gradient is numeric, so a gradient-based optimizer is slower / may stall on
weakly-identified variance components. Mirrored in CHANGELOG (TV-cov + IOV entries)
and `docs/src/model-file/covariates.md` / `iov.md`.

| Combination | Status | Notes / where to start |
|---|---|---|
| TV-covariate **+ dose lagtime** | FD fallback | genuine gap; lagtime seeds a `−1` elapsed-time axis — needs per-event lag duals in the walk |
| TV-covariate **+ expression `obs_scale`** (`obs_scale = f(θ,cov)`) | FD fallback | constant `ScalarScale` IS analytic; only the covariate/parameter-dependent expression scale on the TV-cov walk falls back |
| **IOV + steady-state** doses | FD fallback | SS pre-equilibration not yet wired into the IOV stacked-κ walk |
| **ODE** `[odes]` models (all) | FD fallback (gated off) | analytic infra built but `ODE_SENS_ENABLED = false`; see section C |
| ODE + SS / lagtime / input-rate / SDE / `obs_scale`/LTBS / IOV / TV-cov | FD fallback | out of the `ode_analytical_supported` scope even once ODE is re-armed |
| Any model under **derivative-free BOBYQA** (default) | no gradient used | analytic path only runs under `lbfgs`/`bfgs`/`slsqp` |

Now analytic (no longer fallbacks, this PR): TV-cov alone, TV-cov + SS, TV-cov +
constant `ScalarScale`, TV-cov + EVID 3/4 resets, TV-cov + IOV, IOV alone (FOCEI
**and** FOCE), IOV + resets, lagtime alone, expression `obs_scale` alone, LTBS,
overlapping SS infusions.

## C. Stretch (separate decision): ODE eta sensitivities

Largely **already built and tested**, gated by `const ODE_SENS_ENABLED = false`:
`ode_provider.rs` has `run_subject<N>` (augmented `Dual2<N>` RK45, right-sized
dual-width dispatch 1–16), `pd_from_program`, `dual_init_state`, F-as-dual,
`ode_analytical_supported` (ObsCmt / simple Form-C readout; no
input-rate/SDE/IOV/scaling/LTBS/lagtime), and 6 tests. See
`plans/ode-sensitivities.md`.

Re-arming = flip the flag + (a) broaden FD-parity tests across ODE shapes
(infusion, F, `init(...)`, Form-C readout), (b) a light `Dual1` inner provider
(ODE inner is FD/AD today), (c) a **perf benchmark** — `Dual2<N>` RK45 is O(N²)
per RHS eval × every step, the one real risk.

**Recommendation:** land ODE re-arm as the **next** epic PR, not this one. This
PR is already large and ODE perf needs its own gate. Keep the infra compiled +
tested (gated off) so it doesn't bit-rot.

## D. Definition of done (this PR)

1. A1 IOV+TV-cov tests green (provider + outer, 1/2/3-cpt).
2. A2 changelog + docs accurate to the shipped scope; stale prose fixed.
3. A3 decided (follow-up issue filed, or hoist done).
4. B: PR template filled, NONMEM evidence attached, full local sweep green,
   coverage ≥90% on the diff.
5. ODE stays gated off (C deferred), infra still compiles + tests run.

See [[iov-tvcov-fold-into-ode-367]] for the IOV/TV-cov implementation notes and
[[focei-analytic-outer-gradient-367]] / [[sens-dual2-clean-slate-367]] for the
gradient architecture.
