//! End-to-end coverage for the `MAX_PK_PARAMS` slot-count ceiling.
//!
//! Exercises `parse_model_string` → `simulate_with_seed` → `fit()` on
//! models whose `[individual_parameters]` slot count exceeds
//! `MAX_PK_PARAMS = 16`. Under that older ceiling the parser bails with
//! `ODE model has too many individual parameters for the 16-slot PK
//! layout`; under the current ceiling the same source parses and the
//! whole pipeline runs to a handful of outer iterations with finite
//! results.
//!
//! Three tests, covering the two families of `MAX_PK_PARAMS`-sized stack
//! arrays plus a correctness check on the analytic-gradient path. Every
//! outer iteration of `fit()` dispatches per-subject work through rayon
//! `par_iter`, so all three variants exercise the extended-slot arrays
//! on rayon worker threads (~2 MB stacks, versus the main thread's 8 MB)
//! — a stack overflow would surface as a test panic.
//!
//! - **ODE path** (`bloated_slot_ode_model_parses_simulates_and_fits`):
//!   exercises the plain `f64` stack arrays used everywhere along the
//!   value path — `PkParams::values` (`[f64; MAX_PK_PARAMS]`) and the
//!   `ode/predictions.rs` extended-params buffers (`[f64; MAX_PK_PARAMS
//!   + 2]`). At the slot counts this test uses the ODE analytic
//!   sensitivity provider declines the model (its own independent cap
//!   `MAX_ODE_SENS_DIM = 12` on `pk_indices.len()` in
//!   `src/sens/ode_provider.rs`, orthogonal to `MAX_PK_PARAMS`), so the
//!   run takes the finite-difference outer-gradient path — no `Dual2`
//!   stack arrays constructed.
//!
//! - **Closed-form analytic path**
//!   (`bloated_slot_closed_form_model_engages_dual2_stack_arrays`):
//!   exercises the `[Dual2<M>; N_PK]` stack arrays that
//!   `src/sens/provider.rs` builds inside the per-observation loop, sized
//!   `N_PK = MAX_PK_PARAMS`. Uses the closed-form `pk one_cpt_oral(...)`
//!   provider (`analytical_supported` has no `MAX_ODE_SENS_DIM`-style
//!   cap) with unused-structural fillers to bloat the slot layout past
//!   the pre-bump ceiling; the analytic dispatch stays engaged and the
//!   dual arrays are actually constructed per observation on rayon
//!   workers. A `sens_supported`/`analytical_supported` assertion before
//!   the fit guards against a silent FD fallback that would degenerate
//!   the test into the `f64`-only case.
//!
//! - **Bloated-vs-unbloated analytic parity**
//!   (`bloated_slot_filler_positions_dont_shift_analytic_fit`): fits the
//!   bloated model (`N_FILLER = 15`) and the unbloated version
//!   (`N_FILLER = 0`) under the *same* analytic outer gradient, on the
//!   same synthetic data, with the same options — and asserts the two
//!   fits produce bit-identical θ, Ω, and OFV. Filler slots are dead
//!   weight (no theta/eta/observation reference) and must contribute
//!   exactly zero to the derivative sum; if the extended `[Dual2<M>;
//!   MAX_PK_PARAMS]` array leaks non-zero content from filler positions,
//!   the two fits will diverge. Sharper than an analytic-vs-FD
//!   comparison because optimizer-trajectory noise cancels out — both
//!   branches take the same code path and differ only in the size of
//!   the underlying stack arrays.
//!
//! ## Model shape (shared)
//!
//! Warfarin 1-cpt oral, plus `N_FILLER` unused structural individual
//! parameters that force the slot layout past `MAX_PK_PARAMS -
//! RESERVED_PK_SLOTS.len()` usable slots. Filler parameters are declared
//! in `[individual_parameters]` but never read by `[odes]` (or, for the
//! closed-form variant, never mapped into the `pk one_cpt_oral(...)`
//! call): `ode_param_slots` still routes each to a real PK slot in the
//! ODE variant, so the layout genuinely exceeds the pre-bump cap; the
//! parser emits an "unused" warning rather than rejecting them. Filler
//! expressions reference `TVCL` so they don't get optimised away as dead
//! theta references.
//!
//! Slot budget with `N_FILLER = 15`:
//!
//! - `CL`, `V`, `KA`: 3 canonical PK slots
//! - `FILLER_1 … FILLER_15`: 15 structural slots
//! - `F`, `LAGTIME`: 2 reserved
//!
//! → 20 slots. `MAX_PK_PARAMS = 16` (14 usable) would reject this model
//! at parse time; the current, higher ceiling admits it.
//!
//! ## Analytic path scope
//!
//! Bloated-slot ODE models decline the analytic ODE sensitivity provider
//! because of `MAX_ODE_SENS_DIM = 12` in `src/sens/ode_provider.rs` — a
//! monomorphisation cap on `pk_indices.len()` that routes wider models to
//! FD via a `1..=MAX_ODE_SENS_DIM` dispatch table with a silent `_ =>
//! None` arm. So the analytic ODE path is not reachable at bloated slot
//! counts today; widening it to admit wider models is a separate change
//! (widening `MAX_ODE_SENS_DIM` and its dispatch tables in lockstep).
//! The closed-form analytic path (`analytical_supported`) has no such
//! cap and *is* exercised here.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::{DoseEvent, GradientMethod, OmegaMatrix, Population, Subject};
use ferx_core::{fit, simulate_with_seed, EstimationMethod, FitOptions, MAX_PK_PARAMS};
use std::collections::HashMap;

const N_FILLER: usize = 15;
/// Count of engine-reserved PK slots (`F`, `LAGTIME`). Mirrors
/// `crate::types::RESERVED_PK_SLOTS.len()`, which is `pub(crate)` and so
/// not visible from an integration test.
const RESERVED_PK_SLOTS_LEN: usize = 2;

fn fillers() -> String {
    (1..=N_FILLER)
        .map(|i| {
            format!(
                "  FILLER_{i} = TVCL * {factor:.3}\n",
                factor = 1.0 + (i as f64) * 0.01
            )
        })
        .collect()
}

/// Bloated-slot ODE variant: `ode(...)` structural model, Form-C readout,
/// N_FILLER unused structural params routed into real PK slots by
/// `ode_param_slots` (Pass 2).
fn model_src_ode() -> String {
    format!(
        "\
[parameters]
  theta TVCL(0.15, 0.001, 10.0)
  theta TVV(8.0, 0.1, 500.0)
  theta TVKA(1.2, 0.01, 50.0)
  omega ETA_CL ~ 0.07
  omega ETA_V  ~ 0.02
  omega ETA_KA ~ 0.10
  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
{fillers}
[structural_model]
  ode(states=[depot, central])

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  ode_reltol = 1e-8
  ode_abstol = 1e-10
",
        fillers = fillers(),
    )
}

/// Bloated-slot closed-form variant: `pk one_cpt_oral(cl=CL, v=V, ka=KA)`
/// structural model. `analytical_supported` (`src/sens/provider.rs`) has
/// no `MAX_ODE_SENS_DIM`-style cap, so the analytic sensitivity provider
/// stays engaged even at N_FILLER = 15; every observation constructs
/// `[Dual2<M>; MAX_PK_PARAMS]` stack arrays on the rayon worker threads.
fn model_src_closed_form() -> String {
    format!(
        "\
[parameters]
  theta TVCL(0.15, 0.001, 10.0)
  theta TVV(8.0, 0.1, 500.0)
  theta TVKA(1.2, 0.01, 50.0)
  omega ETA_CL ~ 0.07
  omega ETA_V  ~ 0.02
  omega ETA_KA ~ 0.10
  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
{fillers}
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
",
        fillers = fillers(),
    )
}

fn template_population(n: usize) -> Population {
    let obs_times = [1.0_f64, 4.0, 12.0];
    let subjects: Vec<Subject> = (1..=n)
        .map(|i| Subject {
            id: format!("{i}"),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: obs_times.to_vec(),
            obs_raw_times: vec![],
            observations: vec![0.0; obs_times.len()],
            obs_cmts: vec![2; obs_times.len()],
            covariates: HashMap::new(),
            dose_covariates: vec![],
            obs_covariates: vec![],
            pk_only_times: vec![],
            pk_only_covariates: vec![],
            reset_times: vec![],
            cens: vec![0; obs_times.len()],
            occasions: vec![],
            dose_occasions: vec![],
            fremtype: vec![],
            #[cfg(feature = "survival")]
            obs_records: vec![],
        })
        .collect();

    Population {
        subjects,
        covariate_names: vec![],
        dv_column: "dv".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

fn simulate_into(model: &ferx_core::types::CompiledModel, template: &Population) -> Population {
    let mut truth = model.default_params.clone();
    truth.theta = vec![0.15, 8.0, 1.2];
    truth.omega = OmegaMatrix::from_diagonal(
        &[0.07, 0.02, 0.10],
        vec!["ETA_CL".into(), "ETA_V".into(), "ETA_KA".into()],
    );
    truth.sigma.values = vec![0.01];

    let sim = simulate_with_seed(model, template, &truth, 1, 20260706);
    let mut pop = template.clone();
    for subj in pop.subjects.iter_mut() {
        let dv: Vec<f64> = sim
            .iter()
            .filter(|s| s.id == subj.id)
            .map(|s| s.outcome.continuous_value().max(1e-6))
            .collect();
        subj.observations = dv;
    }
    pop
}

fn fit_short_with(
    model: &mut ferx_core::types::CompiledModel,
    pop: &Population,
    gradient: GradientMethod,
) -> ferx_core::FitResult {
    // Both the inner and outer loops key off the gradient method: `opts`
    // controls the inner loop, but `model.gradient_method` is what
    // `analytic_outer_gradient_available` reads to decide whether the
    // outer loop uses the analytic sensitivity provider or FD. Callers
    // built via `parse_model_string` get `Auto` by default; set both so
    // the two loops agree and `Fd` actually disables the analytic outer
    // path.
    model.gradient_method = gradient;
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.interaction = true;
    opts.outer_maxiter = 2;
    opts.run_covariance_step = false;
    opts.gradient_method = gradient;
    opts.verbose = false;
    fit(model, pop, &model.default_params, &opts)
        .expect("fit() must complete on a bloated-slot model")
}

fn fit_short(
    model: &mut ferx_core::types::CompiledModel,
    pop: &Population,
) -> ferx_core::FitResult {
    fit_short_with(model, pop, GradientMethod::Auto)
}

fn assert_finite_fit_result(res: &ferx_core::FitResult, ctx: &str) {
    assert!(
        res.ofv.is_finite(),
        "{ctx}: fit() returned non-finite OFV = {}; possible NaN/Inf from \
         the extended MAX_PK_PARAMS slot layout",
        res.ofv
    );
    for (i, t) in res.theta.iter().enumerate() {
        assert!(t.is_finite(), "{ctx}: θ[{i}] = {t} is non-finite");
    }
    for k in 0..res.omega.nrows() {
        let v = res.omega[(k, k)];
        assert!(v.is_finite(), "{ctx}: ω²[{k}] = {v} is non-finite");
    }
    for (i, s) in res.sigma.iter().enumerate() {
        assert!(s.is_finite(), "{ctx}: σ[{i}] = {s} is non-finite");
    }
}

/// Static arithmetic guard shared by both tests — makes the failure mode
/// explicit if a future revert of MAX_PK_PARAMS would otherwise only fail
/// at the less obvious parser layer.
fn check_slot_budget() {
    let required = 3 /* CL, V, KA */
        + N_FILLER   /* filler structural slots */
        + RESERVED_PK_SLOTS_LEN /* F, LAGTIME reserved */;
    assert!(
        MAX_PK_PARAMS >= required,
        "MAX_PK_PARAMS must be at least {required} for this test's model \
         to fit the slot layout (got {MAX_PK_PARAMS})"
    );
    assert!(
        3 + N_FILLER > 16 - RESERVED_PK_SLOTS_LEN,
        "this test is only meaningful when the slot count exceeds the old \
         MAX_PK_PARAMS = 16 usable ceiling"
    );
}

/// Parse → simulate → fit end-to-end on a bloated-slot ODE model.
/// Exercises the `f64` `[f64; MAX_PK_PARAMS]` and `[f64; MAX_PK_PARAMS +
/// 2]` stack arrays on rayon worker threads. Kept fast: two subjects with
/// sparse observations, a couple of outer iterations, no covariance step
/// — well short of convergence. The goal is pipeline completion with
/// finite results across the extended slot layout, not tight parameter
/// estimates.
#[test]
fn bloated_slot_ode_model_parses_simulates_and_fits() {
    check_slot_budget();

    // Parse — the older MAX_PK_PARAMS = 16 ceiling would bail here.
    let mut model = parse_model_string(&model_src_ode())
        .expect("bloated-slot ODE model must parse under the current MAX_PK_PARAMS ceiling");

    // Simulate — exercises the ODE prediction path over the extended
    // `[f64; MAX_PK_PARAMS]` and `[f64; MAX_PK_PARAMS + 2]` buffers on
    // the main thread.
    let template = template_population(2);
    let pop = simulate_into(&model, &template);
    for (i, subj) in pop.subjects.iter().enumerate() {
        assert_eq!(
            subj.observations.len(),
            subj.obs_times.len(),
            "subject {i} simulate() did not populate observations",
        );
        assert!(
            subj.observations.iter().all(|v| v.is_finite() && *v > 0.0),
            "subject {i} simulate() produced non-finite/non-positive obs: {:?}",
            subj.observations
        );
    }

    // Fit — dispatches per-subject work through rayon `par_iter`, so
    // every outer iteration crosses the worker-thread pool. A stack
    // overflow from the extended `f64` slot arrays on the smaller worker
    // stacks would surface as a panic here.
    let res = fit_short(&mut model, &pop);
    assert_finite_fit_result(&res, "ODE variant");
}

/// Parse → simulate → fit end-to-end on a bloated-slot **closed-form**
/// model. Guaranteed to engage the analytic sensitivity provider (asserted
/// via `sens_supported`/`analytical_supported`), which constructs
/// `[Dual2<M>; MAX_PK_PARAMS]` stack arrays inside the per-observation
/// loop. Those arrays are the largest stack allocations affected by the
/// ceiling bump — up to hundreds of KB per array at the maximum dual
/// width — and this test exercises them across the rayon worker-thread
/// pool during a real fit.
#[test]
fn bloated_slot_closed_form_model_engages_dual2_stack_arrays() {
    check_slot_budget();

    let mut model = parse_model_string(&model_src_closed_form()).expect(
        "bloated-slot closed-form model must parse under the current MAX_PK_PARAMS ceiling",
    );

    // Hard guard: this test's whole reason for existing is to exercise
    // the `[Dual2<M>; N_PK]` stack arrays inside the analytic sensitivity
    // provider. A silent FD fallback would build no dual arrays and
    // reduce this to a duplicate of the ODE test above.
    assert!(
        ferx_core::sens::provider::analytical_supported(&model),
        "closed-form analytic path must be armed on the bloated-slot model; \
         otherwise this test degenerates to the FD-only case already covered \
         by the ODE variant"
    );
    assert!(
        ferx_core::sens::provider::sens_supported(&model),
        "sens_supported must be true so the fit dispatches through the analytic \
         outer-gradient path (which is what constructs the Dual2 stack arrays)"
    );

    let template = template_population(2);
    let pop = simulate_into(&model, &template);
    for (i, subj) in pop.subjects.iter().enumerate() {
        assert!(
            subj.observations.iter().all(|v| v.is_finite() && *v > 0.0),
            "subject {i} simulate() produced non-finite/non-positive obs: {:?}",
            subj.observations
        );
    }

    let res = fit_short(&mut model, &pop);
    assert_finite_fit_result(&res, "closed-form variant");
}

/// Bloated-vs-unbloated model source: identical apart from the fillers.
/// Both build the same 1-cpt-oral fit; the only difference is how many
/// PK slots the layout occupies.
fn model_src_closed_form_unbloated() -> String {
    format!(
        "\
[parameters]
  theta TVCL(0.15, 0.001, 10.0)
  theta TVV(8.0, 0.1, 500.0)
  theta TVKA(1.2, 0.01, 50.0)
  omega ETA_CL ~ 0.07
  omega ETA_V  ~ 0.02
  omega ETA_KA ~ 0.10
  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
"
    )
}

/// Correctness check for the analytic sensitivity path on the extended
/// slot layout. Fits the *bloated* closed-form model (`N_FILLER` unused
/// structural params, exercising `[Dual2<M>; MAX_PK_PARAMS]` arrays with
/// most slots as `Dual2::constant(0.0)` filler) and the *unbloated*
/// version (`N_FILLER = 0`, same three PK slots, no filler) under the
/// analytic outer gradient, on the same synthetic data with the same
/// options. The two fits must land on identical estimates — filler slots
/// carry no theta, no eta, no observation contribution, so they are dead
/// weight on the derivative accumulation and must not shift the optimum.
///
/// Any drift is direct evidence that the enlarged-array derivative
/// arithmetic is leaking non-zero contributions from filler positions,
/// which was the review's concern about the extended `Dual2` slot
/// layout.
///
/// This is a sharper check than analytic-vs-FD parity: the two branches
/// use the *same* code path with the *same* optimizer, differing only in
/// the size of the underlying stack arrays. Optimizer-trajectory
/// sensitivity cancels out.
#[test]
fn bloated_slot_filler_positions_dont_shift_analytic_fit() {
    check_slot_budget();

    let mut m_bloat =
        parse_model_string(&model_src_closed_form()).expect("bloated-slot model must parse");
    let mut m_plain =
        parse_model_string(&model_src_closed_form_unbloated()).expect("unbloated model must parse");
    assert!(
        ferx_core::sens::provider::analytical_supported(&m_bloat),
        "analytic path must be armed on the bloated model"
    );
    assert!(
        ferx_core::sens::provider::analytical_supported(&m_plain),
        "analytic path must be armed on the unbloated model too"
    );

    // Same synthetic data for both fits — simulate from the unbloated
    // model to avoid any incidental contribution of the fillers to the
    // simulation itself (they should have none by construction, but
    // simulating from the smaller model makes the invariant obvious).
    let template = template_population(2);
    let pop = simulate_into(&m_plain, &template);

    let res_bloat = fit_short_with(&mut m_bloat, &pop, GradientMethod::Auto);
    let res_plain = fit_short_with(&mut m_plain, &pop, GradientMethod::Auto);
    assert_finite_fit_result(&res_bloat, "bloated analytic fit");
    assert_finite_fit_result(&res_plain, "unbloated analytic fit");

    // The two runs must land on bit-identical estimates. Filler slots
    // are `Dual2::constant(0.0)` — with no theta/eta/observation
    // contribution they contribute exactly zero to every derivative sum
    // and every OFV term, so the optimizer trajectory must be
    // pointwise-identical.
    assert_eq!(
        res_bloat.theta.len(),
        res_plain.theta.len(),
        "θ dimensions must match; both models declare the same 3 θs"
    );
    for i in 0..res_bloat.theta.len() {
        assert_eq!(
            res_bloat.theta[i], res_plain.theta[i],
            "θ[{i}] differs between bloated (with {N_FILLER} fillers) and \
             unbloated (0 fillers) analytic fits — {} vs {}. Filler slots \
             at positions > n_indiv must contribute zero to the derivative \
             sum but appear to be leaking non-zero content.",
            res_bloat.theta[i], res_plain.theta[i],
        );
    }
    for k in 0..res_bloat.omega.nrows() {
        assert_eq!(
            res_bloat.omega[(k, k)],
            res_plain.omega[(k, k)],
            "ω²[{k}] differs between bloated and unbloated analytic fits",
        );
    }
    assert_eq!(
        res_bloat.ofv, res_plain.ofv,
        "OFV differs between bloated and unbloated analytic fits — \
         filler slots must contribute nothing to the objective"
    );
}
