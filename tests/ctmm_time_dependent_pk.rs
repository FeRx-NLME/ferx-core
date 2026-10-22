//! #1918 — `fit()` and `ferx check` refuse a `TIME`-dependent disposition on a
//! time-inhomogeneous (drug-driven) CTMM generator. The `[markov_model]` arm of #1853.
//!
//! `markov::endpoint`'s inhomogeneous scorer solves the `[odes]` system under one `$PK`
//! snapshot, `(model.pk_param_fn)(…, 0.0)`, and samples the states the transition
//! intensities read from that solve. Dose-record quantities are read per dose record
//! (#1575); the disposition the RHS integrates, and `init()`, are not.
//!
//! ## What was measured, on this file's model
//!
//! `CL = TVCL·(1 + KT·TIME)`, `TVCL = 1`, `V = 10`, a 100 mg IV bolus at `t = 0`, both
//! intensities scaled by `C = central/V` (`A·C`, `B·C`, `A = 0.5`, `B = 0.3`), so the
//! generator commutes across time and `P(Δt) = expm(Q₀·∫C)` exactly. States observed at
//! 0.5, 1.3, 2.6, 5, 12 and 20. `ctmm_subject_nll` before the refusal below was added,
//! against a Simpson computation in Python, outside both engines:
//!
//! | | `−log L` | `central(20)` |
//! |---|---|---|
//! | ferx, `KT = 0`    | 3.374745938459441 | |
//! | ferx, `KT = 0.05` | 3.374745938459441 (**bit-identical**) | |
//! | closed form, `CL` frozen at `TIME = 0` | 3.374745987743 | 13.534 |
//! | closed form, `$PK` per record (the Gaussian walk's convention) | 3.377830924039 | 3.676 |
//!
//! So the CTMM pass is the frozen closed form to its sub-grid error (4.9e-8), the `TIME`
//! coefficient has no influence on the transition term at all, and the generator reads a
//! concentration 3.7× the one the Gaussian predictions use at `t = 20`.
//!
//! ## Why there is no NONMEM anchor here
//!
//! The change refuses a model class and produces no numerical result to anchor. NONMEM
//! has no CTMM likelihood either (an occupancy reset per observation is not expressible
//! without zeroing the PK, see `tests/markov_smoke.rs`); the measurement above uses a
//! closed form for the same reason #1853's did.

#![cfg(feature = "markov")]

use ferx_core::api::read_population_for;
use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::{CompiledModel, Population};
use ferx_core::{
    fit, predict, run_covariance, run_sir, simulate, validate_model_file, EstimationMethod,
    FitOptions,
};
use std::io::Write;

/// `CL_EXPR` is spliced into `CL`; everything else is fixed.
const MODEL: &str = r"
[parameters]
  theta TVCL(1.0, FIX)
  theta TVV(10.0, FIX)
  theta TVKT(0.05, FIX)
  theta A(0.5, 1e-4, 100.0)
  theta B(0.3, 1e-4, 100.0)
  sigma PROP ~ 0.05 (sd)

[individual_parameters]
  CL = CL_EXPR
  V  = TVV

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -(CL/V) * central

[error_model]
  DV ~ proportional(PROP)

[markov_model]
  type   = ctmm
  cmt    = 5
  states = [s0=0, s1=1]
  transition s0 -> s1 = A * (central / V)
  transition s1 -> s0 = B * (central / V)
";

/// A 100 mg bolus at 0, one PK observation, and the CTMM states.
const DATA: &str = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,SS,II\n\
                    1,0,.,1,100,1,0,1,0,0\n\
                    1,0.5,0,0,.,5,0,0,0,0\n\
                    1,1,9,0,.,1,0,0,0,0\n\
                    1,1.3,1,0,.,5,0,0,0,0\n\
                    1,2.6,0,0,.,5,0,0,0,0\n\
                    1,5,1,0,.,5,0,0,0,0\n\
                    1,12,0,0,.,5,0,0,0,0\n\
                    1,20,1,0,.,5,0,0,0,0\n";

const TIME_CL: &str = "TVCL * (1.0 + TVKT * TIME)";
const FREE_CL: &str = "TVCL";

fn model(cl: &str) -> CompiledModel {
    parse_model_string(&MODEL.replace("CL_EXPR", cl)).expect("the drug-driven CTMM parses")
}

fn population(model: &CompiledModel) -> Population {
    let mut f = tempfile::NamedTempFile::new().expect("create temp csv");
    f.write_all(DATA.as_bytes()).expect("write temp csv");
    let path = f.path().to_str().expect("temp path is utf-8");
    let (pop, _) = read_population_for(model, &None, path, None, None, None, &[])
        .expect("endpoint-routed load must succeed");
    pop
}

fn fit_opts() -> FitOptions {
    FitOptions {
        method: EstimationMethod::FoceI,
        outer_maxiter: 0,
        run_covariance_step: false,
        threads: Some(2),
        ..FitOptions::default()
    }
}

fn assert_is_the_1918_refusal(entry: &str, msg: &str) {
    for want in [
        "[markov_model] on CMT=5",
        "`CL` reads the `TIME` built-in and the [odes] RHS reads it",
        "#1918",
        "Hold `CL` constant in TIME",
    ] {
        assert!(msg.contains(want), "{entry}: missing {want:?} in: {msg}");
    }
}

/// `fit()` refuses before any iteration, with the code and block `ferx check` reports.
#[test]
fn fit_refuses_a_time_dependent_disposition_on_a_drug_driven_ctmm() {
    let m = model(TIME_CL);
    let err = fit(&m, &population(&m), &m.default_params, &fit_opts())
        .expect_err("fit must refuse a TIME-dependent disposition on a drug-driven CTMM");
    assert_is_the_1918_refusal("fit", &err.to_string());
    assert_eq!(err.code(), Some("E_ENDPOINT_TIME_DEPENDENT_PK"), "{err}");
    assert_eq!(err.block(), Some("markov_model"), "{err}");
}

/// **The straddle**: one line of `$PK` apart, the time-free control fits to a real
/// objective, so the refusal above is the `TIME` read and not the fixture.
#[test]
fn the_time_free_control_fits() {
    let m = model(FREE_CL);
    let res = fit(&m, &population(&m), &m.default_params, &fit_opts())
        .expect("the time-free control must fit");
    assert!(
        res.ofv.is_finite() && res.ofv.abs() < 1e6,
        "control objective {} must be real, not the 1e20 repel sentinel",
        res.ofv
    );
}

/// `ferx check` without `--data`: the check is a function of the model alone. The control
/// stays valid, so a check that refused every drug-driven CTMM would die here too.
#[test]
fn ferx_check_reports_it_without_data() {
    for (cl, refused) in [(TIME_CL, true), (FREE_CL, false)] {
        let mut f = tempfile::Builder::new()
            .suffix(".ferx")
            .tempfile()
            .expect("create temp model");
        f.write_all(MODEL.replace("CL_EXPR", cl).as_bytes())
            .expect("write temp model");
        let report = validate_model_file(f.path().to_str().expect("utf-8 path"), None);
        let hits: Vec<_> = report
            .diagnostics
            .iter()
            .filter(|d| d.code == "E_ENDPOINT_TIME_DEPENDENT_PK")
            .collect();
        if refused {
            assert!(!report.valid, "{cl}: {:?}", report.diagnostics);
            assert_eq!(hits.len(), 1, "{cl}: {:?}", report.diagnostics);
            assert_eq!(hits[0].block.as_deref(), Some("markov_model"));
            assert!(
                hits[0].line.is_some(),
                "the [markov_model] header line is attached"
            );
            assert_is_the_1918_refusal("ferx check", &hits[0].message);
        } else {
            assert!(hits.is_empty(), "{cl}: {hits:?}");
        }
    }
}

/// **Only the entry points that score the CTMM refuse** (round-1 review). `predict()` drops
/// CTMM rows when continuous observations exist and its per-record walk never runs the
/// frozen solve, so the Gaussian prediction of this mixed model stays available; and
/// `simulate()` keeps its own `E_CTMM_UNSUPPORTED` refusal rather than this one.
#[test]
fn predict_and_simulate_are_not_pre_empted() {
    let m = model(TIME_CL);
    let pop = population(&m);
    let preds = predict(&m, &pop, &m.default_params)
        .expect("predict() of the Gaussian half must not be refused by the CTMM check");
    assert!(
        !preds.is_empty() && preds.iter().all(|p| p.pred.is_finite()),
        "{preds:?}"
    );
    let err = simulate(&m, &pop, &m.default_params, 1).expect_err("simulate refuses a CTMM");
    assert_eq!(err.code(), Some("E_CTMM_UNSUPPORTED"), "{err}");
}

/// **The post-fit re-scorers refuse too** (round-2 review). `run_covariance()` and
/// `run_sir()` re-run the CTMM likelihood from a `FitResult`, which may predate the refusal
/// or be caller-supplied. The fit here is the time-free control's (so it exists); handed
/// the `TIME`-dependent model, both must return the coded refusal — and handed the control
/// model, neither may, so the `Err` is the `TIME` read and not the inputs.
#[test]
fn run_covariance_and_run_sir_refuse_it() {
    let control = model(FREE_CL);
    let pop = population(&control);
    let res = fit(&control, &pop, &control.default_params, &fit_opts()).expect("control fits");
    let refusing = model(TIME_CL);
    let opts = fit_opts();

    let err = run_covariance(&res, Some(&refusing), Some(&pop), &opts)
        .expect_err("run_covariance must refuse");
    assert_eq!(err.code(), Some("E_ENDPOINT_TIME_DEPENDENT_PK"), "{err}");
    assert_is_the_1918_refusal("run_covariance", &err.to_string());

    let err = run_sir(&res, Some(&refusing), Some(&pop), &opts).expect_err("run_sir must refuse");
    assert_eq!(err.code(), Some("E_ENDPOINT_TIME_DEPENDENT_PK"), "{err}");
    assert_is_the_1918_refusal("run_sir", &err.to_string());

    // The straddle: on the control model whatever these return is not this refusal.
    for (entry, out) in [
        (
            "run_covariance",
            run_covariance(&res, Some(&control), Some(&pop), &opts),
        ),
        ("run_sir", run_sir(&res, Some(&control), Some(&pop), &opts)),
    ] {
        if let Err(e) = out {
            assert_ne!(
                e.code(),
                Some("E_ENDPOINT_TIME_DEPENDENT_PK"),
                "{entry}: {e}"
            );
        }
    }
}

/// #1854: the covariate axis of the same freeze reaches the post-fit re-scorers too.
/// `WT` steps from 70 to 90 at `t = 5` and `CL` reads it, so the frozen `[odes]` solve
/// would read `CL` at the baseline `WT`. `fit()` refuses it; a `FitResult` of the
/// `WT`-free control handed the `WT` model must be refused by `run_covariance()` and
/// `run_sir()` as well — and handed the control, neither may return this code. The step
/// is carried by the `t = 8` PK record: the reader keeps no covariates on discrete-state
/// rows, so a change seen only there is time-varying to no engine.
#[test]
fn run_covariance_and_run_sir_refuse_a_tv_covariate_on_the_disposition() {
    const DATA_WT: &str = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,SS,II,WT\n\
                           1,0,.,1,100,1,0,1,0,0,70\n\
                           1,0.5,0,0,.,5,0,0,0,0,70\n\
                           1,1,9,0,.,1,0,0,0,0,70\n\
                           1,1.3,1,0,.,5,0,0,0,0,70\n\
                           1,2.6,0,0,.,5,0,0,0,0,70\n\
                           1,5,1,0,.,5,0,0,0,0,90\n\
                           1,8,4,0,.,1,0,0,0,0,90\n\
                           1,12,0,0,.,5,0,0,0,0,90\n\
                           1,20,1,0,.,5,0,0,0,0,90\n";
    let refusing = model("TVCL * WT / 70");
    let control = model(FREE_CL);
    let mut f = tempfile::NamedTempFile::new().expect("create temp csv");
    f.write_all(DATA_WT.as_bytes()).expect("write temp csv");
    let path = f.path().to_str().expect("temp path is utf-8");
    let (pop, _) = read_population_for(&refusing, &None, path, None, None, None, &[])
        .expect("endpoint-routed load must succeed");
    let opts = fit_opts();

    let err = fit(&refusing, &pop, &refusing.default_params, &opts)
        .expect_err("fit must refuse a TV covariate on CL");
    assert_eq!(err.code(), Some("E_ENDPOINT_TV_COVARIATE"), "{err}");

    let res = fit(&control, &pop, &control.default_params, &opts).expect("control fits");
    for (entry, out) in [
        (
            "run_covariance",
            run_covariance(&res, Some(&refusing), Some(&pop), &opts),
        ),
        ("run_sir", run_sir(&res, Some(&refusing), Some(&pop), &opts)),
    ] {
        let err = out.expect_err(entry);
        assert_eq!(
            err.code(),
            Some("E_ENDPOINT_TV_COVARIATE"),
            "{entry}: {err}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("[markov_model] on CMT=5") && msg.contains("`CL` reads covariate `WT`"),
            "{entry}: {msg}"
        );
    }
    for (entry, out) in [
        (
            "run_covariance",
            run_covariance(&res, Some(&control), Some(&pop), &opts),
        ),
        ("run_sir", run_sir(&res, Some(&control), Some(&pop), &opts)),
    ] {
        if let Err(e) = out {
            assert_ne!(e.code(), Some("E_ENDPOINT_TV_COVARIATE"), "{entry}: {e}");
        }
    }
}
