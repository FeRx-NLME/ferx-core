//! #426 T9 / T10: `run_covariance` with default options re-scores the objective under the
//! settings the fit recorded (`FitResult::scoring_settings`), so it reproduces the fit's
//! inline covariance step bit for bit for a non-default fit too.
//!
//! T9 is one test per recorded field, each failing under its field's name. Every row first
//! asserts its premise — the fit's covariance, recomputed with that one field back at its
//! default, is *different* bits — so a row whose field does not move the covariance on its
//! fixture cannot pass by accident. The premise runs on a copy of the fit with the record
//! cleared, since with the record present a default field resolves back to the record.
//! Mutation: delete any one line of `with_scoring_record` → that field's row dies, naming it.
//!
//! Two fields have no row, because no affordable fixture moves the covariance with them; their
//! resolve lines are pinned at the resolver instead (`fit_inputs::tests::scoring_record_*`):
//! - `ode_stiff_abort_after` only acts on a segment the stiffness probe flags, and no example
//!   model is stiff.
//! - `inner_restarts` re-solves a cold EBE from ±k·sd seeds and keeps a seed only when it beats
//!   the mode by more than 1e-9, so it acts only on a multimodal subject. Measured inert on
//!   warfarin (default solver; Nelder–Mead at `inner_maxiter = 2`) and on the EVID=4 washout
//!   model, whose resets make every coordinate scanned (default solver at `inner_maxiter` 200
//!   and 1; Nelder–Mead at 2).
//!
//! Engines (measured, and asserted per fit through `gradient_method_inner`): every fixture
//! runs analytic (`Dual2`) inner gradients — warfarin, warfarin_ode, warfarin_iov — under
//! FOCEI, or Laplace for the quadrature rows. Three outer iterations: the identity holds at whatever
//! point the fit stops.

use ferx_core::{
    fit, prepare_run, run_covariance, run_sir, CompiledModel, EstimationMethod, FitOptions,
    FitResult, InnerOptimizer, Population,
};

fn quiet(method: EstimationMethod) -> FitOptions {
    FitOptions {
        verbose: false,
        method,
        run_covariance_step: true,
        sir: false,
        outer_maxiter: 3,
        ..FitOptions::default()
    }
}

/// The caller of every claim: `FitOptions::default()`, quietened (`verbose` is not recorded).
fn defaults() -> FitOptions {
    FitOptions {
        verbose: false,
        ..FitOptions::default()
    }
}

fn bits(fit: &FitResult, what: &str) -> Vec<u64> {
    let cov = fit
        .covariance_matrix
        .as_ref()
        .unwrap_or_else(|| panic!("{what}: covariance step produced no matrix"));
    assert!(
        cov.iter().all(|x| x.is_finite()),
        "{what}: non-finite covariance"
    );
    cov.iter().map(|x| x.to_bits()).collect()
}

/// A model file and its dataset.
struct Fixture(&'static str, &'static str);

const WARFARIN: Fixture = Fixture("examples/warfarin.ferx", "data/warfarin.csv");
const WARFARIN_ODE: Fixture = Fixture("examples/warfarin_ode.ferx", "data/warfarin.csv");
const WARFARIN_IOV: Fixture = Fixture("examples/warfarin_iov.ferx", "data/warfarin_iov.csv");

struct Fitted {
    model: CompiledModel,
    population: Population,
    fit: FitResult,
}

impl Fitted {
    fn new(fixture: &Fixture, options: &FitOptions, what: &str) -> Self {
        let prep = prepare_run(fixture.0, Some(fixture.1)).expect("prepare");
        let (model, population, init) = (prep.parsed.model, prep.population, prep.init_params);
        let fit =
            fit(&model, &population, &init, options).unwrap_or_else(|e| panic!("{what}: fit: {e}"));
        assert!(
            fit.gradient_method_inner.starts_with("analytic"),
            "{what}: the engine this file names moved: {}",
            fit.gradient_method_inner
        );
        assert!(
            fit.scoring_settings.is_some(),
            "{what}: fit() recorded nothing"
        );
        Fitted {
            model,
            population,
            fit,
        }
    }

    fn cov(&self, fit: &FitResult, options: &FitOptions, what: &str) -> Vec<u64> {
        let out = run_covariance(fit, Some(&self.model), Some(&self.population), options)
            .unwrap_or_else(|e| panic!("{what}: run_covariance: {e}"));
        bits(&out, what)
    }

    /// The fit with no record, so `options` is scored as given.
    fn unrecorded(&self) -> FitResult {
        let mut f = self.fit.clone();
        f.scoring_settings = None;
        f.sir_settings = None;
        f
    }
}

/// One T9 row: fit under `base` with one field moved; (premise) the record cleared and the
/// field back at its default moves the bits; (claim) default options reproduce the inline step.
fn row(field: &str, fixture: Fixture, with: FitOptions, base: FitOptions) {
    let f = Fitted::new(&fixture, &with, field);
    let inline = bits(&f.fit, field);
    assert!(
        f.cov(&f.unrecorded(), &base, field) != inline,
        "{field}: premise — this field at its default must move the covariance on this fixture"
    );
    assert!(
        f.cov(&f.fit, &defaults(), field) == inline,
        "{field}: run_covariance with default options must take the recorded {field}"
    );
}

macro_rules! field_row {
    ($name:ident, $fixture:expr, $base:expr, $field:ident = $value:expr) => {
        #[test]
        fn $name() {
            let base: FitOptions = $base;
            let with = FitOptions {
                $field: $value,
                ..base.clone()
            };
            row(stringify!($field), $fixture, with, base);
        }
    };
}

/// A base whose EBE solves fail (`inner_optimizer = nelder_mead`, `inner_maxiter = 2`, held
/// on both sides of the premise), so every solve takes the Nelder–Mead fallback (measured:
/// 40 of 40). The settings that only act on a solve that did not converge — the fallback's
/// warm start, the restart seeds — are inert at the default solver, with `inner_maxiter` at
/// 200, 3 or 1 (measured: the premise fails).
fn failing_inner() -> FitOptions {
    FitOptions {
        inner_optimizer: InnerOptimizer::NelderMead,
        inner_maxiter: 2,
        ..quiet(EstimationMethod::FoceI)
    }
}

field_row!(
    inner_optimizer,
    WARFARIN,
    quiet(EstimationMethod::FoceI),
    inner_optimizer = InnerOptimizer::Lbfgs
);
field_row!(
    inner_maxiter,
    WARFARIN,
    quiet(EstimationMethod::FoceI),
    inner_maxiter = 3
);
field_row!(
    ebe_warm_start,
    WARFARIN,
    failing_inner(),
    ebe_warm_start = true
);
// The non-IOV EBE solve ignores the μ shift; the IOV solve optimises in ψ = η + μ.
field_row!(
    mu_referencing,
    WARFARIN_IOV,
    quiet(EstimationMethod::FoceI),
    mu_referencing = false
);
field_row!(n_agq, WARFARIN, quiet(EstimationMethod::Laplace), n_agq = 3);
field_row!(
    ode_reltol,
    WARFARIN_ODE,
    quiet(EstimationMethod::FoceI),
    ode_reltol = 1e-7
);
field_row!(
    ode_abstol,
    WARFARIN_ODE,
    quiet(EstimationMethod::FoceI),
    ode_abstol = 1e-10
);
field_row!(
    ode_max_steps,
    WARFARIN_ODE,
    quiet(EstimationMethod::FoceI),
    ode_max_steps = 60
);
field_row!(
    ode_method,
    WARFARIN_ODE,
    quiet(EstimationMethod::FoceI),
    ode_method = ferx_core::ode::OdeMethod::Rodas5P
);
field_row!(
    ode_auto_switch,
    WARFARIN_ODE,
    quiet(EstimationMethod::FoceI),
    ode_auto_switch = false
);

/// `inner_tol`, on a two-stage chain, unset by the caller: the Laplace stage tightens its own
/// `inner_tol` to 1e-8, so the record must come from that stage's options, not the
/// top-level ones (1e-5). Mutations: delete the `inner_tol` resolve line, or record from
/// `options` instead of `stage_opts` → dies.
#[test]
fn inner_tol() {
    let with = FitOptions {
        methods: vec![EstimationMethod::FoceI, EstimationMethod::Laplace],
        ..quiet(EstimationMethod::FoceI)
    };
    let f = Fitted::new(&WARFARIN, &with, "inner_tol");
    assert_eq!(
        f.fit.scoring_settings.as_ref().map(|s| s.inner_tol),
        Some(1e-8),
        "inner_tol: the record is the producing stage's"
    );
    let inline = bits(&f.fit, "inner_tol");
    assert!(
        f.cov(&f.unrecorded(), &defaults(), "inner_tol") != inline,
        "inner_tol: premise — the top-level inner_tol must move the covariance"
    );
    assert!(
        f.cov(&f.fit, &defaults(), "inner_tol") == inline,
        "inner_tol: run_covariance with default options must take the stage's recorded inner_tol"
    );
}

/// T10: the caller's non-default value wins over the record, and only it. Both sides of the
/// "caller left the default?" gate on one fit: `bfgs` (non-default) scores exactly as on an
/// unrecorded fit — and not as the inline `lbfgs` step — while `auto` (the default) yields to
/// the recorded `lbfgs`. Mutations: the record always wins → the first half dies; the caller
/// always wins → the second.
#[test]
fn a_callers_non_default_setting_wins_over_the_record() {
    let with = FitOptions {
        inner_optimizer: InnerOptimizer::Lbfgs,
        ..quiet(EstimationMethod::FoceI)
    };
    let f = Fitted::new(&WARFARIN, &with, "T10");
    let inline = bits(&f.fit, "T10");
    let bfgs = FitOptions {
        inner_optimizer: InnerOptimizer::Bfgs,
        ..defaults()
    };
    let caller_wins = f.cov(&f.fit, &bfgs, "T10 bfgs");
    assert!(
        caller_wins == f.cov(&f.unrecorded(), &bfgs, "T10 bfgs, unrecorded"),
        "T10: a non-default caller value must be scored as given"
    );
    assert!(
        caller_wins != inline,
        "T10: premise — bfgs must move the covariance away from the lbfgs fit's"
    );
    let auto = FitOptions {
        inner_optimizer: InnerOptimizer::Auto,
        ..defaults()
    };
    assert!(
        f.cov(&f.fit, &auto, "T10 auto") == inline,
        "T10: a default caller value must yield to the record"
    );
}

/// `run_sir` takes its scoring settings from the SIR record, not from
/// `fit.scoring_settings`: the in-fit SIR scores with the fit's top-level options, and on a
/// Laplace fit those differ from the producing stage's (`inner_tol` 1e-5 against 1e-8). The
/// premise is the difference itself, and that it moves the draws; then default options
/// repeat the in-fit SIR. Mutation: resolve `run_sir` through the stage record first → dies.
#[test]
fn run_sir_scores_with_the_sir_record_not_the_stage_record() {
    let with = FitOptions {
        sir: true,
        sir_samples: 200,
        sir_resamples: 100,
        sir_seed: Some(7),
        ..quiet(EstimationMethod::Laplace)
    };
    let f = Fitted::new(&WARFARIN, &with, "run_sir");
    let ess = f.fit.sir_ess.expect("in-fit SIR ran");
    assert!(ess.is_finite(), "in-fit ESS {ess}");
    let stage = f.fit.scoring_settings.clone().expect("stage record");
    let sir = f.fit.sir_settings.clone().expect("SIR record");
    assert_ne!(
        stage.inner_tol, sir.scoring.inner_tol,
        "premise: the stage and SIR records differ on this fixture"
    );
    let rerun = |fit: &FitResult| {
        let mut bare = fit.clone();
        bare.sir_ess = None;
        bare.sir_ci_theta = None;
        run_sir(&bare, Some(&f.model), Some(&f.population), &defaults())
            .expect("run_sir")
            .sir_ess
            .map(f64::to_bits)
    };
    let mut stage_scored = f.fit.clone();
    stage_scored.sir_settings.as_mut().unwrap().scoring = stage;
    assert_ne!(
        rerun(&stage_scored),
        Some(ess.to_bits()),
        "premise: scoring the draws with the stage record must move the ESS"
    );
    assert_eq!(
        rerun(&f.fit),
        Some(ess.to_bits()),
        "run_sir with default options must repeat the in-fit SIR"
    );
}
