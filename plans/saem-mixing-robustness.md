# Plan: SAEM robustness — prevent Ω collapse and slow mixing on sparse data

## Goal

Make SAEM converge to sensible estimates on **sparse, multi-η models** without
hand-tuning MH settings. Today, on the Jasmine vanco-peds model (5937 subjects,
**2.93 obs/subject**, 2-cpt, 4 random effects) SAEM badly under-fits Ω and dumps
between-subject variability into residual error:

| | NONMEM (eval) | SAEM (default) |
|---|---|---|
| ω(CL/V1/Q/V2) | 0.071 / 0.010 / 0.073 / 1.19 | 0.021 / 0.016 / 0.014 / 0.015 |
| ADD_ERR | 0.72 | 3.11 |
| OFV | 67513 | 79328 |

The estimator is **not wrong** — it is mixing too slowly and has not converged.
This plan fixes the three mechanisms behind that, verified empirically below.

## Diagnosis (confirmed by instrumenting the Ω/σ trajectory)

Two compounding mechanisms, both triggered by sparse data + zero-initialized ηs:

1. **Ω collapses on iteration 1.** ηs initialize at 0
   (`get_eta_init(n_eta, None, None)`, `saem.rs:921`). During exploration γ=1, so
   the M-step *replaces* Ω wholesale with `(1/N)Σηηᵀ` (`saem.rs:1283`, `:1310`).
   After only `n_mh_steps=3` small MH steps from zero, that statistic is ~0.016,
   so Ω crashes 0.1 → 0.016 on the first iteration, discarding the user's initial
   estimate. There is also **no positive-definite floor on the BSV Ω** — the IOV
   Ω gets one (`saem.rs:1362`), the BSV Ω does not.

2. **Recovery is far too slow.** The MH proposal scale is tied to the (now tiny)
   Ω; measured acceptance is **0.76** vs the 0.40 target (proposals ~2× too small),
   and step-scale adaptation only fires every `adapt_interval=50` iters — ~8 times
   in the default 400-iter run. The poorly-identified V1/Q/V2 ηs barely move.

Instrumented trajectory, default settings (`n_mh_steps=3`, `adapt_interval=50`):

```
iter  1: Ω=[0.016,0.017,0.017,0.016]  σ_add=5.12   (started at 0.1)
iter 40: Ω=[0.076,0.020,0.018,0.016]  σ_add=0.97   (still climbing, not converged)
```

Same data/seed/iterations with `n_mh_steps=15`, `adapt_interval=5`:

```
iter  1: Ω=[0.070,0.060,0.057,0.060]  σ_add=2.46   (no collapse)
iter 40: Ω=[0.096,0.065,0.082,0.087]  σ_add=0.555  → OFV 66944
```

OFV **66944**, beating both the SAEM bench (79328) and the NONMEM eval (67513),
with Ω/σ in the right ballpark. So mixing — not the algorithm — is the problem.

## Non-goals

- Replacing random-walk MH with a fundamentally different sampler (HMC already
  exists behind `autodiff` for analytical-PK models; this plan targets the
  default MH path that everyone hits).
- Changing the γ schedule formula (Monolix two-phase `γ=1` then `1/(k−K1)`),
  except where Phase 2 protects Ω during the earliest exploration steps.
- Auto-tuning `n_exploration` / `n_convergence` (separate convergence-detection
  work). This plan only makes each iteration mix better and protects early Ω.

## Current state (relevant code)

- Defaults: `saem_n_exploration=150`, `saem_n_convergence=250`, `saem_n_mh_steps=3`,
  `saem_adapt_interval=50` (`types.rs:1743`).
- η init at 0, `step_scales=0.3` (`saem.rs:921-924`).
- γ schedule: `saem.rs:1109`.
- SA stat update `s2 = (1-γ)s2 + γ·eta_outer` (`saem.rs:1283`); `omega_mat = s2`
  then zero structural off-diagonals + restore FIX rows (`saem.rs:1310-1334`).
  **No diagonal floor.**
- IOV Ω diagonal floored to 1e-8 (`saem.rs:1360-1365`) — the pattern to mirror.
- Step-size adaptation, target 0.40 (MH) / 0.65 (HMC) (`saem.rs:1540-1568`).
- fit_option keys `n_mh_steps`, `adapt_interval`, `n_exploration`, `n_convergence`
  parsed at `model_parser.rs:1719-1724`.

## Status

- **Phase 1 — DONE.** `floor_omega_diagonal` helper + `SAEM_OMEGA_DIAG_FLOOR`
  (1e-6) in `saem.rs`; unit tests pass.
- **Phase 2 — DONE.** Hard burn-in chosen (not the damped update). New
  `saem_omega_burnin` fit-option (default 20), clamped to `n_exploration`; gates
  both the BSV and IOV Ω M-steps. Parser key `omega_burnin` added.
  - **Empirical result on Jasmine (default `n_mh_steps=3`, burn-in 20):** Ω held
    at 0.1 through iter 20, first update lands [0.097, 0.093, 0.101, 0.096] — no
    collapse — and the run reaches **OFV 67013** (NONMEM eval 67513; old SAEM
    bench 79328). The burn-in alone resolves the reported failure.
- **Phase 3 — DESCOPED.** Timed runs (full Jasmine data, 30 iters, 8 threads)
  showed raising `n_mh_steps` costs ~9% wall at 10 and ~22% at 15, for no
  correctness benefit now that the burn-in fixes the collapse. Decision: keep the
  `n_mh_steps` (3) and `adapt_interval` (50) defaults and document both as
  tunables instead of changing them.
- **Phase 5 — SKIPPED** by request (no standalone NONMEM-validation writeup;
  the in-line comparison above suffices).
- **Phase 6 — DONE.** Docs: `omega_burnin` documented in
  `docs/model-file/fit-options.qmd` and the SAEM page (M-step Ω section,
  config block, new "Ω Collapses / Residual Error Inflates" troubleshooting
  entry). Tests: Tier-1 floor unit tests + parser test (above) and a new
  slow-tests-gated convergence test `tests/saem_omega_burnin.rs` — a fully
  synthetic, sparse (1 obs/subject) 2-η model where the burn-in run recovers
  trace(Ω) ≈ 0.147 (truth 0.18) while the no-burn-in run collapses to ≈ 0.044.
  The recovery assertion fails on the pre-fix code. Auto-discovered by
  `slow-tests.yml` (no registration needed); no proprietary data used.

## Phase 1 — BSV Ω diagonal floor (smallest, highest-safety)

Mirror the IOV floor for the BSV Ω so the diagonal can never reach a degenerate
value that starves the MH proposal. After the structural-zero / FIX-restore block
(`saem.rs:1334`), floor each free diagonal:

```rust
for i in 0..n_eta {
    let fixed = init_params.omega_fixed.get(i).copied().unwrap_or(false);
    if !fixed && state.omega_mat[(i, i)] < 1e-6 {
        state.omega_mat[(i, i)] = 1e-6;
    }
}
```

(1e-6 not 1e-8: the BSV proposal scale is `step_scale·chol(Ω)`, and we want it to
stay large enough to move. Tune against the benchmark.)

Test: unit test that a near-zero free diagonal entering the M-step is floored,
and FIX-ed entries are left untouched.

## Phase 2 — Protect Ω during the earliest exploration iterations

The collapse is specifically the iteration-1 wholesale replacement before the
chain has moved. Add a short burn-in `K0` (e.g. `min(20, K1)`) during which Ω is
**not** updated from the SA statistic — the chain mixes at the user's initial Ω
first. ηs still move, θ/σ still update; only the Ω M-step (`saem.rs:1310-1334`)
is skipped while `k <= K0`. Keep accumulating `s2` so the first real Ω update
uses a warmed-up statistic.

Decision to make (ask reviewer): a hard burn-in `K0` vs. a damped early update
`Ω ← max(Ω_init·floor_frac, s2)` — the burn-in is simpler and matches the
empirical fix. Add `saem_omega_burnin` (default ~20) as a fit_option so it is
tunable and testable.

Test: synthetic sparse model — assert Ω after iteration 1 stays ≥ a fraction of
Ω_init rather than collapsing to the 3-step statistic.

## Phase 3 — Better default mixing

Two cheap default changes, justified by the trajectory above:

- Raise `saem_n_mh_steps` default `3 → 10`. Cost is linear in MH steps but the
  MH inner loop reuses `pk_scratch` and is cheap relative to the M-step; the
  benchmark timing should be re-measured (Phase 5).
- Lower `saem_adapt_interval` default `50 → 10` so step-scale tracking actually
  engages within the run.

Both stay overridable via fit_options. Update `types.rs:1743` defaults and the
docs (Phase 6). Note the verbose-trace "acc=0.00" artifact on adapt iterations
(rate printed right after the counter reset) — fix the reset/print ordering in
`saem.rs:1540-1588` while here, or document it.

## Phase 4 — (optional) seed ηs off a cheap per-subject mode

Instead of η=0, warm-start ηs from a couple of inner-loop steps (or the
mu-reference values already computed for FOCE) so iteration 1 starts the chain
near the conditional mode. Lower priority than Phases 1–3 and interacts with the
burn-in; gate behind its own decision. Skip if Phases 1–3 close the gap.

## Phase 5 — Validation against NONMEM (required by CLAUDE.md)

Re-run the Jasmine vanco-peds model (`ferx-testdata/jasmine_vanco_peds/run60.ferx`,
`train_0416.csv`) with default settings after Phases 1–3 and compare Ω, σ, OFV to
the NONMEM eval (`run60_nm_eval-fit.yaml`, OFV 67513). Target: default SAEM lands
Ω/σ within the same ballpark and OFV ≤ ~67600 *without* user tuning. Record the
comparison in the PR description and add a row to `docs/estimation/saem.qmd`.
Re-measure wall time (default SAEM bench was 736 s) so the `n_mh_steps` bump is
accounted for.

## Phase 6 — Tests & docs

- Tier-1 unit tests for the Ω floor (Phase 1) and burn-in guard (Phase 2).
- Tier-3 `slow-tests` gated convergence test: sparse multi-η synthetic model whose
  default-settings SAEM recovers Ω within tolerance (would fail pre-fix).
- Docs: document `saem_omega_burnin` and the changed `n_mh_steps`/`adapt_interval`
  defaults in `docs/model-file/fit-options.qmd` and the SAEM page; note the
  sparse-data guidance and the `method = [saem, focei]` polish.

## Risks / open questions

- New defaults must not regress the existing SAEM benchmarks (cefepime, warfarin)
  — richer data where the collapse doesn't occur. Run those before merging.
- Floor value (Phase 1) and burn-in length (Phase 2) are tunables; pick against
  the benchmark, don't hard-code blindly.
- `ferx-r` exposes these settings — if defaults change, confirm no R-side
  assertions pin the old values (follow-up PR in `../ferx-r` if any `pub`
  default-bearing API shifts).
