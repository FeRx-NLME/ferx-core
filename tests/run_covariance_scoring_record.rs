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
//! #1806 T1–T3 (the `in_fit_sir_*` and `run_sir_without_a_sir_record_*` tests): the in-fit
//! SIR scores under the stage record, so the two records are equal and `run_sir` repeats it;
//! a fit with no SIR record re-scores under the stage record. The non-PD SIR fallback has no
//! row: no fixture makes a quadrature Hessian non-PD. It shares the one `sir_opts` binding
//! with the normal SIR in `fit()`.
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

/// `inner_tol` again, on the same chain with the Laplace stage an **evaluator**
/// (`agq_eval_only`): the FOCEI stage produces the estimates and runs the inline covariance
/// step at 1e-5, and the trailing readout's 1e-8 must not overwrite the record (#1805 r1
/// finding 1). Premise: the readout's 1e-8 moves the covariance, so a record holding it
/// could not reproduce the inline step. Mutation: drop `!eval_only_methods.contains(&method)`
/// from the record site in `fit_inner` → the record is 1e-8 and the row dies.
/// `imp_eval_only` needs no row: an IMP readout does not tighten `inner_tol`, so its record
/// equals the estimating stage's and the defect cannot show there.
#[test]
fn inner_tol_ignores_a_trailing_evaluator() {
    let with = FitOptions {
        methods: vec![EstimationMethod::FoceI, EstimationMethod::Laplace],
        agq_eval_only: true,
        ..quiet(EstimationMethod::FoceI)
    };
    let what = "inner_tol (trailing evaluator)";
    let f = Fitted::new(&WARFARIN, &with, what);
    assert_eq!(
        f.fit.scoring_settings.as_ref().map(|s| s.inner_tol),
        Some(FitOptions::default().inner_tol),
        "{what}: the record is the estimating FOCEI stage's, not the Laplace readout's"
    );
    let inline = bits(&f.fit, what);
    let readout = FitOptions {
        inner_tol: 1e-8,
        ..defaults()
    };
    assert!(
        f.cov(&f.unrecorded(), &readout, what) != inline,
        "{what}: premise — the readout's inner_tol must move the covariance"
    );
    assert!(
        f.cov(&f.fit, &defaults(), what) == inline,
        "{what}: run_covariance with default options must reproduce the inline step"
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

/// Every SIR output a `run_sir` must repeat: the ESS and each θ/Ω/σ interval, as bits.
/// `is_finite` first, so a `NaN` ESS cannot match a `NaN` ESS.
fn sir_bits(fit: &FitResult, what: &str) -> Vec<u64> {
    let ess = fit
        .sir_ess
        .unwrap_or_else(|| panic!("{what}: SIR produced no ESS"));
    assert!(ess.is_finite(), "{what}: ESS {ess}");
    let mut out = vec![ess.to_bits()];
    for ci in [&fit.sir_ci_theta, &fit.sir_ci_omega, &fit.sir_ci_sigma] {
        let ci = ci
            .as_ref()
            .unwrap_or_else(|| panic!("{what}: SIR produced no interval"));
        for (lo, hi) in ci {
            assert!(lo.is_finite() && hi.is_finite(), "{what}: CI ({lo}, {hi})");
            out.extend([lo.to_bits(), hi.to_bits()]);
        }
    }
    out
}

/// `fit` with its SIR outputs cleared, the records left as they are.
fn sir_cleared(fit: &FitResult) -> FitResult {
    let mut bare = fit.clone();
    bare.sir_ess = None;
    bare.sir_ci_theta = None;
    bare.sir_ci_omega = None;
    bare.sir_ci_sigma = None;
    bare
}

impl Fitted {
    fn sir(&self, fit: &FitResult, options: &FitOptions, what: &str) -> Vec<u64> {
        let out = run_sir(fit, Some(&self.model), Some(&self.population), options)
            .unwrap_or_else(|e| panic!("{what}: run_sir: {e}"));
        sir_bits(&out, what)
    }
}

fn with_sir(base: FitOptions) -> FitOptions {
    FitOptions {
        sir: true,
        sir_samples: 200,
        sir_resamples: 100,
        sir_seed: Some(7),
        ..base
    }
}

/// #1806 T1 / T2: the in-fit SIR scores under the record of the stage that produced the
/// estimates, so `sir_settings.scoring == scoring_settings`, and `run_sir` with default
/// options repeats it bit for bit. The quadrature rows straddle the old behaviour: each
/// premise asserts the stage ran at a different `inner_tol` from the top-level options, which
/// is what the in-fit SIR scored at before #1806.
///
/// Mutations: build `sir_opts` from the top-level `options` again → every quadrature row's
/// record equality dies (1e-5 or 1e-6 against 1e-8). Overlay the record with the value-based
/// `with_scoring_record` instead of `ScoringSettings::overwrite` → only
/// `laplace_explicit_tol` dies (the caller's non-default 1e-6 would win). `saem_focei` is the
/// non-quadrature control: there the stage, SIR and top-level settings are equal before and
/// after.
fn sir_row(row: &str, options: FitOptions, quadrature: bool) {
    let f = Fitted::new(&WARFARIN, &options, row);
    let stage = f.fit.scoring_settings.clone().expect("stage record");
    if quadrature {
        assert_ne!(
            stage.inner_tol, options.inner_tol,
            "{row}: premise — the quadrature stage must run at a tighter inner_tol than the \
             top-level options"
        );
        assert_eq!(
            stage.inner_tol, 1e-8,
            "{row}: premise — the stage tolerance"
        );
    } else {
        assert_eq!(
            stage.inner_tol, options.inner_tol,
            "{row}: control — no stage tightens on this chain"
        );
    }
    let sir = f.fit.sir_settings.clone().expect("SIR record");
    assert_eq!(
        sir.scoring, stage,
        "{row}: the in-fit SIR must score under the producing stage's record"
    );
    let inline = sir_bits(&f.fit, row);
    assert_eq!(
        f.sir(&sir_cleared(&f.fit), &defaults(), row),
        inline,
        "{row}: run_sir with default options must repeat the in-fit SIR"
    );
}

#[test]
fn in_fit_sir_scores_under_the_stage_record_laplace() {
    sir_row("laplace", with_sir(quiet(EstimationMethod::Laplace)), true);
}

#[test]
fn in_fit_sir_scores_under_the_stage_record_focei_nagq3() {
    sir_row(
        "focei_nagq3",
        with_sir(FitOptions {
            n_agq: 3,
            // The analytic AGQ covariance Hessian trips a debug-build symmetry assert
            // (`agq_cov_hessian.rs`, `S_kl must be symmetric`) on this fit with or without
            // SIR, measured at `d43afca9`; the FD stencil scores the same objective.
            analytic_cov_hessian: false,
            ..quiet(EstimationMethod::FoceI)
        }),
        true,
    );
}

#[test]
fn in_fit_sir_scores_under_the_stage_record_chain_focei_laplace() {
    sir_row(
        "chain_focei_laplace",
        with_sir(FitOptions {
            methods: vec![EstimationMethod::FoceI, EstimationMethod::Laplace],
            ..quiet(EstimationMethod::FoceI)
        }),
        true,
    );
}

#[test]
fn in_fit_sir_scores_under_the_stage_record_laplace_explicit_tol() {
    sir_row(
        "laplace_explicit_tol",
        with_sir(FitOptions {
            inner_tol: 1e-6,
            ..quiet(EstimationMethod::Laplace)
        }),
        true,
    );
}

#[test]
fn in_fit_sir_scores_under_the_stage_record_saem_focei_control() {
    sir_row(
        "saem_focei",
        with_sir(FitOptions {
            methods: vec![EstimationMethod::Saem, EstimationMethod::FoceI],
            inner_tol: 1e-7,
            saem_n_exploration: 2,
            saem_n_convergence: 2,
            ..quiet(EstimationMethod::Saem)
        }),
        false,
    );
}

/// #1806 T3: `run_sir` on a fit with no SIR record takes the stage record, so it repeats the
/// SIR the same fit would have run with `sir = true` (ferx-r#511). Each row's premise is the
/// straddle: with the stage record cleared too, the same call scores at the defaults and
/// lands on different bits. Mutation: drop the stage-record fallback in `resolve_sir_options`
/// → every row's claim dies.
fn no_sir_record_row(row: &str, fixture: Fixture, base: FitOptions) {
    let sir_args = FitOptions {
        sir_samples: 300,
        sir_resamples: 100,
        sir_seed: Some(1),
        ..defaults()
    };
    let twin_opts = FitOptions {
        sir: true,
        sir_samples: sir_args.sir_samples,
        sir_resamples: sir_args.sir_resamples,
        sir_seed: sir_args.sir_seed,
        ..base.clone()
    };
    let twin = Fitted::new(&fixture, &twin_opts, row);
    let f = Fitted::new(&fixture, &base, row);
    assert!(f.fit.sir_settings.is_none(), "{row}: premise — no SIR ran");
    let want = sir_bits(&twin.fit, row);
    assert_ne!(
        f.sir(&f.unrecorded(), &sir_args, row),
        want,
        "{row}: premise — with neither record, run_sir must score elsewhere"
    );
    assert_eq!(
        f.sir(&f.fit, &sir_args, row),
        want,
        "{row}: run_sir on a fit with no SIR record must score under the stage record"
    );
}

#[test]
fn run_sir_without_a_sir_record_takes_the_stage_record_inner_maxiter() {
    no_sir_record_row(
        "focei_inner_maxiter_5",
        WARFARIN,
        FitOptions {
            inner_maxiter: 5,
            ..quiet(EstimationMethod::FoceI)
        },
    );
}

#[test]
fn run_sir_without_a_sir_record_takes_the_stage_record_laplace() {
    no_sir_record_row("laplace", WARFARIN, quiet(EstimationMethod::Laplace));
}

#[test]
fn run_sir_without_a_sir_record_takes_the_stage_record_ltbs() {
    no_sir_record_row(
        "ltbs",
        Fixture("examples/warfarin_ltbs.ferx", "data/warfarin.csv"),
        quiet(EstimationMethod::FoceI),
    );
}
