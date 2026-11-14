//! A subject whose compiled plan fails takes finite differences for the quantity at hand, the
//! fit completes with the exact path's results, and `W_PLAN_FD_FALLBACK` reports it (#2786).
//!
//! Every test fits one fixture twice — plans working, then every plan failing
//! ([`FORCE_PLAN_FAIL`] on a one-worker pool) — and compares the two to a measured bound.
use super::FORCE_PLAN_FAIL;
use crate::types::{
    CompiledModel, EstimationMethod, FitOptions, FitResult, Optimizer, Population, WarningCode,
};
use std::path::Path;

const TOKEN: &str = "W_PLAN_FD_FALLBACK";

/// `f` on the worker of a one-thread pool with every plan failing.
fn plans_failing<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    let pool = rayon::ThreadPoolBuilder::new()
        .stack_size(crate::api::FIT_RAYON_STACK_SIZE)
        .num_threads(1)
        .build()
        .expect("one-thread pool");
    pool.install(|| {
        FORCE_PLAN_FAIL.set(true);
        let out = f();
        FORCE_PLAN_FAIL.set(false);
        out
    })
}

fn example(model: &str, data: &str) -> (CompiledModel, Population) {
    let parsed = crate::parser::model_parser::parse_full_model_file(Path::new(model))
        .unwrap_or_else(|e| panic!("parse {model}: {e}"));
    let (population, _) = crate::api::read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        data,
        None,
        parsed.fit_options.iov_column.as_deref(),
        None,
        &parsed.column_map,
    )
    .unwrap_or_else(|e| panic!("read {data}: {e}"));
    (parsed.model, population)
}

fn opts(method: EstimationMethod) -> FitOptions {
    FitOptions {
        method,
        outer_maxiter: 3,
        threads: Some(1),
        verbose: false,
        ..FitOptions::default()
    }
}

/// The fit with plans working and with every plan failing.
fn pair(model: &CompiledModel, pop: &Population, o: &FitOptions) -> (FitResult, FitResult) {
    let fit = || crate::api::fit(model, pop, &model.default_params, o).expect("fit");
    let exact = fit();
    let forced = plans_failing(fit);
    (exact, forced)
}

/// The largest relative gap between two parameter vectors, every entry finite.
fn worst(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut w = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        assert!(x.is_finite() && y.is_finite(), "{x} vs {y}");
        w = w.max((x - y).abs() / x.abs().max(1e-3));
    }
    w
}

/// The forced fit matches the exact one — OFV within `ofv_tol` absolute, θ / Ω / σ (and the
/// standard errors, when `se_tol` is given) within their relative bounds — and only the forced
/// fit reports the fallback, naming every subject and the plan layer's reason.
fn check(label: &str, exact: &FitResult, forced: &FitResult, tol: [f64; 3]) {
    let [ofv_tol, par_tol, se_tol] = tol;
    assert!(
        exact.ofv.is_finite() && forced.ofv.is_finite(),
        "{label}: OFV"
    );
    assert!(
        forced.ofv < 1e10,
        "{label}: forced OFV {} repelled",
        forced.ofv
    );
    let d_ofv = (exact.ofv - forced.ofv).abs();
    let pars = |r: &FitResult| -> Vec<f64> {
        r.theta
            .iter()
            .chain(r.omega.iter())
            .chain(&r.sigma)
            .copied()
            .collect()
    };
    let d_par = worst(&pars(exact), &pars(forced));
    let d_se = match (&exact.se_theta, &forced.se_theta) {
        (Some(a), Some(b)) => worst(a, b),
        (None, None) => 0.0,
        (a, b) => panic!("{label}: SE present {} vs {}", a.is_some(), b.is_some()),
    };
    eprintln!(
        "{label}: |ΔOFV| = {d_ofv:.3e}, max rel Δparam = {d_par:.3e}, max rel ΔSE = {d_se:.3e}"
    );
    assert!(
        d_ofv <= ofv_tol,
        "{label}: |ΔOFV| = {d_ofv:e} > {ofv_tol:e}"
    );
    assert!(
        d_par <= par_tol,
        "{label}: Δparam = {d_par:e} > {par_tol:e}"
    );
    assert!(
        d_se <= se_tol,
        "{label}: ΔSE = {d_se:e} > {se_tol:e}: {:?} vs {:?}",
        exact.se_theta,
        forced.se_theta
    );
    assert!(
        !exact.warnings.iter().any(|w| w.contains(TOKEN)),
        "{label}: plans held, yet {:?}",
        exact.warnings
    );
    let n = forced.subjects.len();
    let w: Vec<&String> = forced
        .warnings
        .iter()
        .filter(|w| w.contains(TOKEN))
        .collect();
    assert_eq!(
        w.len(),
        1,
        "{label}: one warning, got {:?}",
        forced.warnings
    );
    let want = format!(
        "W_PLAN_FD_FALLBACK: {n} of {n} subjects fell back to finite-difference derivatives \
         because their compiled plan failed ({}); results are correct but slower — please \
         report this as a bug.",
        super::FORCED_FAILURE
    );
    assert_eq!(*w[0], want, "{label}");
    let entry = forced
        .warnings_structured
        .iter()
        .find(|e| e.message.contains(TOKEN))
        .expect("typed entry");
    assert_eq!(entry.category, WarningCode::GradientFallback);
    // Re-classified from its text (a `.fitrx` round trip) it lands in the same category.
    assert_eq!(
        crate::types::classify_warning(w[0]).category,
        WarningCode::GradientFallback
    );
}

#[test]
fn focei_falls_back_to_fd_inner_outer_and_covariance() {
    let (m, p) = example("examples/warfarin.ferx", "data/warfarin.csv");
    let (e, f) = pair(&m, &p, &opts(EstimationMethod::FoceI));
    // Measured 5.8e-4 / 8.2e-5 / 1.2e-4 (Linux x86_64).
    check("focei", &e, &f, [3e-3, 5e-4, 1e-3]);
}

#[test]
fn foce_falls_back_to_fd() {
    let (m, p) = example("examples/warfarin.ferx", "data/warfarin.csv");
    let (e, f) = pair(&m, &p, &opts(EstimationMethod::Foce));
    // Measured 1.5e-5 / 6.1e-6 / 2.3e-4.
    check("foce", &e, &f, [1e-4, 5e-5, 2e-3]);
}

#[test]
fn laplace_and_agq_fall_back_to_fd() {
    let (m, p) = example("examples/warfarin.ferx", "data/warfarin.csv");
    let (e, f) = pair(&m, &p, &opts(EstimationMethod::Laplace));
    // Measured 2.9e-4 / 6.3e-5 / 3.3e-5.
    check("laplace", &e, &f, [2e-3, 5e-4, 3e-4]);
    let o = FitOptions {
        n_agq: 3,
        ..opts(EstimationMethod::Laplace)
    };
    let (e, f) = pair(&m, &p, &o);
    // Measured 3.0e-4 / 4.7e-5 / 2.4e-5.
    check("agq3", &e, &f, [2e-3, 5e-4, 3e-4]);
}

#[test]
fn gauss_newton_and_trust_region_fall_back_to_fd() {
    let (m, p) = example("examples/warfarin.ferx", "data/warfarin.csv");
    let (e, f) = pair(&m, &p, &opts(EstimationMethod::FoceGn));
    // Measured 2.2e-6 / 1.1e-8 / 2.7e-4.
    check("gn", &e, &f, [2e-5, 1e-6, 2e-3]);
    let o = FitOptions {
        optimizer: Optimizer::TrustRegion,
        ..opts(EstimationMethod::FoceI)
    };
    let (e, f) = pair(&m, &p, &o);
    // Measured 1.8e-5 / 3.7e-7 / 4.0e-4.
    check("trust-region", &e, &f, [2e-4, 4e-6, 3e-3]);
}

#[test]
fn sir_falls_back_to_fd() {
    let (m, p) = example("examples/warfarin.ferx", "data/warfarin.csv");
    let o = FitOptions {
        sir: true,
        sir_samples: 200,
        sir_resamples: 50,
        sir_seed: Some(7),
        ..opts(EstimationMethod::FoceI)
    };
    let (e, f) = pair(&m, &p, &o);
    check("sir", &e, &f, [3e-3, 5e-4, 1e-3]);
    let (a, b) = (e.sir_ess.expect("ess"), f.sir_ess.expect("ess"));
    // Measured 41.983 vs 42.006: the log-Jacobian gradient and the proposal both held.
    eprintln!("sir: ESS {a} vs {b}");
    assert!((a - b).abs() < 0.2, "ESS {a} vs {b}");
}

#[test]
fn saem_falls_back_to_fd() {
    let (m, p) = example("examples/warfarin.ferx", "data/warfarin.csv");
    let o = FitOptions {
        saem_n_exploration: 5,
        saem_n_convergence: 5,
        saem_seed: Some(3),
        // Every θ through the M-step's numerical score, not the mu-referenced closed form.
        mu_referencing: false,
        ..opts(EstimationMethod::Saem)
    };
    let (e, f) = pair(&m, &p, &o);
    // Measured 5.2e-10 / 0 / 1.8e-8: the M-step's per-θ score lands on the same draws.
    check("saem", &e, &f, [1e-8, 1e-9, 1e-6]);
}

#[test]
fn iov_falls_back_to_fd() {
    let (m, p) = example("examples/warfarin_iov.ferx", "data/warfarin_iov.csv");
    let (e, f) = pair(&m, &p, &opts(EstimationMethod::FoceI));
    // Measured 2.5e-3 / 2.1e-3 / 4.3e-4.
    check("iov", &e, &f, [2e-2, 2e-2, 3e-3]);
}

/// The SDE (EKF) subject: the fit at its start — inner loop, objective and covariance — and,
/// since a trajectory of a few outer steps amplifies any difference, the outer gradient
/// itself: the reconverged central difference a declined subject takes against the exact
/// adjoint, per subject.
#[test]
fn sde_falls_back_to_fd() {
    let (m, mut p) = example("examples/warfarin_sde.ferx", "data/warfarin_sde.csv");
    // Its finite differences re-run the filter per coordinate: a few subjects keep it short.
    p.subjects.truncate(4);
    let o = FitOptions {
        outer_maxiter: 0,
        ..opts(EstimationMethod::FoceI)
    };
    let (e, f) = pair(&m, &p, &o);
    // Measured 8.6e-6 / 0 / 6.3e-4.
    check("sde", &e, &f, [1e-4, 1e-8, 5e-3]);
    // The difference quotient is only as good as the trajectory it differences: the
    // gradients are compared where the filter is integrated tightly.
    let mut m = m;
    if let Some(ode) = m.ode_spec.as_mut() {
        ode.solver_opts.reltol = 1e-10;
        ode.solver_opts.abstol = 1e-12;
    }
    let o = FitOptions {
        inner_tol: 1e-10,
        ..o
    };
    let worst = outer_gradient_gap(&m, &p, &o);
    eprintln!("sde: worst outer-gradient gap {worst:.3e}");
    // Measured 6.6e-5 at these tolerances (8.1e-2 at the defaults, the quotient's noise).
    assert!(worst < 5e-4, "{worst:e}");
}

/// The largest gap (relative to the gradient's largest entry), over `pop`'s subjects at the
/// initial estimates, between the exact adjoint outer gradient and the reconverged central
/// difference (`subject_reconverged_fd_gradient`) a subject whose plan failed takes instead.
fn outer_gradient_gap(m: &CompiledModel, pop: &Population, o: &FitOptions) -> f64 {
    let params = &m.default_params;
    let x = crate::estimation::parameterization::pack_params(params);
    let bounds = crate::estimation::parameterization::compute_bounds(params);
    let pt = crate::sens::ode_provider::adjoint::PackedPoint::new(params, &x).expect("point");
    // The fit's per-subject outer gradient.
    let adjoint = |s: &crate::types::Subject, eta: &[f64]| {
        crate::estimation::sens_outer_gradient::subject_gradient_at(m, s, &pt, eta, o.interaction)
    };
    let mut worst = 0.0f64;
    for s in &pop.subjects {
        let ebe = crate::estimation::inner_optimizer::find_ebe_for_stage(
            m,
            s,
            params,
            o.inner_maxiter,
            1e-10,
            None,
            None,
            0,
            o,
        );
        let eta = ebe.eta.as_slice();
        let exact = adjoint(s, eta).expect("exact gradient");
        let fd = plans_failing(|| {
            assert!(
                adjoint(s, eta).is_none(),
                "the failed plan declines the gradient"
            );
            crate::estimation::outer_optimizer::subject_reconverged_fd_gradient(
                &x, params, m, s, &ebe.eta, &bounds, o,
            )
        });
        assert!(exact.iter().all(|v| v.is_finite()), "{exact:?}");
        let scale = exact.iter().fold(1.0f64, |a, v| a.max(v.abs()));
        for (k, (a, b)) in exact.iter().zip(&fd).enumerate() {
            assert!(a.is_finite() && b.is_finite(), "{exact:?} vs {fd:?}");
            // A FIX coordinate is pinned (`lower == upper`): the optimizer's FD never moves it.
            if bounds.lower[k] == bounds.upper[k] {
                continue;
            }
            worst = worst.max((a - b).abs() / scale); // nan-ok: a, b, scale finite asserted above; scale >= 1
        }
    }
    worst
}

#[test]
#[cfg(feature = "survival")]
fn non_gaussian_curvature_falls_back_to_fd() {
    // The CTMM example needs `markov`; the base `ci,survival` set does not compile it in.
    #[allow(unused_mut)]
    let mut cases = vec![
        ("examples/tte_exponential.ferx", "data/tte_exponential.csv"),
        ("examples/binary_logistic.ferx", "data/binary_logistic.csv"),
    ];
    #[cfg(feature = "markov")]
    cases.push(("examples/ctmm_2state.ferx", "data/ctmm_2state.csv"));
    for (model, data) in cases {
        let (m, p) = example(model, data);
        let (e, f) = pair(&m, &p, &opts(EstimationMethod::Laplace));
        // Measured, worst of the three (TTE): 2.1e-4 / 5.3e-3 / 5.9e-4.
        check(model, &e, &f, [2e-3, 5e-2, 5e-3]);
    }
}

const MIXTURE: &str = r"
[parameters]
  theta TVCL1(1.2, 0.01, 100.0)
  theta TVCL2(2.5, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  theta BWT(0.0, -5.0, 5.0)
  omega ETA_CL ~ 0.05
  sigma EPS ~ 0.01

[mixture]
  nsub = 2
  logit(1) = MIXL + BWT*(WT - 75)

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

#[test]
fn mixture_falls_back_to_fd() {
    use std::io::Write;
    let model = crate::parser::model_parser::parse_model_string(MIXTURE).expect("parse");
    let mut csv = String::from("ID,TIME,DV,AMT,EVID,CMT,WT\n");
    let mut sid = 0usize;
    for (cl, wt) in [(1.0f64, 60.0), (3.0, 90.0)] {
        for _ in 0..6 {
            sid += 1;
            csv.push_str(&format!("{sid},0,0,100,1,1,{wt}\n"));
            for (i, t) in [0.5f64, 1.0, 2.0, 4.0, 8.0].iter().enumerate() {
                let c =
                    10.0 * (-(cl / 10.0) * t).exp() * (1.0 + 0.03 * ((sid + i) as f64 * 1.3).sin());
                csv.push_str(&format!("{sid},{t},{c:.5},0,0,1,{wt}\n"));
            }
        }
    }
    let mut file = tempfile::NamedTempFile::new().expect("tmp");
    file.write_all(csv.as_bytes()).expect("write");
    let pop =
        crate::io::datareader::read_nonmem_csv(file.path(), Some(&["WT"]), None).expect("read");
    // No covariance step: a mixture's exact R is all or nothing, so a failed plan sends the
    // whole step to the existing FD stencil — a model-level route this test does not add.
    let o = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        ..opts(EstimationMethod::FoceI)
    };
    let (e, f) = pair(&model, &pop, &o);
    // Measured 1.8e-10 / 0 (EBEs and objective at the start).
    check("mixture", &e, &f, [1e-8, 1e-9, 0.0]);
    // The outer gradient: a declined subject takes its own reconverged central difference in
    // each class, so the mixture gradient stays the posterior-weighted one.
    let params = &model.default_params;
    let grad = || {
        let eval = crate::estimation::mixture::mixture_ofv(&model, &pop, params, &o, None);
        let log = crate::estimation::outer_optimizer::OuterFdDeclineLog::new(pop.subjects.len());
        crate::estimation::mixture::mixture_gradient_salvaged(
            &model, &pop, params, &o, &eval, None, &log,
        )
        .expect("a mixture gradient")
    };
    let exact = grad();
    let fd = plans_failing(grad);
    assert!(exact.iter().all(|v| v.is_finite()), "{exact:?}");
    let scale = exact.iter().fold(1.0f64, |a, v| a.max(v.abs()));
    let mut worst = 0.0f64;
    for (a, b) in exact.iter().zip(&fd) {
        assert!(a.is_finite() && b.is_finite(), "{exact:?} vs {fd:?}");
        worst = worst.max((a - b).abs() / scale); // nan-ok: a, b, scale finite asserted above; scale >= 1
    }
    eprintln!("mixture: worst outer-gradient gap {worst:.3e}");
    // Measured 1.3e-6.
    assert!(worst < 1e-5, "{worst:e}");
}

/// With the plans working nothing falls back: every example model with a `[simulation]` block,
/// fitted on its own simulated data (shortened: 6 subjects, 2 outer iterations, one start),
/// reports no `W_PLAN_FD_FALLBACK` — the switch-off control for every test above.
#[test]
fn no_example_model_falls_back_with_plans_working() {
    let mut paths: Vec<_> = std::fs::read_dir("examples")
        .expect("examples/")
        .map(|e| e.expect("entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "ferx"))
        .collect();
    paths.sort();
    let dir = tempfile::tempdir().expect("tmp");
    let mut n = 0;
    for path in paths {
        let text = std::fs::read_to_string(&path).expect("read");
        if !text.lines().any(|l| l.trim() == "[simulation]") {
            continue;
        }
        let short = |key: &str, v: &str, l: &str| {
            let t = l.trim_start();
            (t.starts_with(key) && t[key.len()..].trim_start().starts_with('='))
                .then(|| format!("  {key} = {v}"))
        };
        let mut out: Vec<String> = text
            .lines()
            .map(|l| {
                short("maxiter", "2", l)
                    .or_else(|| short("n_subjects", "6", l))
                    .or_else(|| short("n_starts", "1", l))
                    .unwrap_or_else(|| l.to_string())
            })
            .collect();
        if !out.iter().any(|l| l.trim_start().starts_with("maxiter")) {
            match out.iter().position(|l| l.trim() == "[fit_options]") {
                Some(i) => out.insert(i + 1, "  maxiter = 2".into()),
                None => out.extend(["[fit_options]".into(), "  maxiter = 2".into()]),
            }
        }
        let file = dir.path().join(path.file_name().expect("name"));
        std::fs::write(&file, out.join("\n")).expect("write");
        let overrides = crate::api::RunOverrides {
            threads: Some(1),
            ..Default::default()
        };
        let fit = match crate::api::run_model_simulate_with_overrides(
            file.to_str().expect("utf8"),
            &overrides,
        ) {
            Ok((fit, _)) => fit,
            // A `[simulation]` block `--simulate` itself refuses (a covariate the design does
            // not state) never reaches a plan.
            Err(e) => {
                eprintln!("{}: not simulated: {e}", path.display());
                continue;
            }
        };
        assert!(
            !fit.warnings.iter().any(|w| w.contains(TOKEN)),
            "{}: {:?}",
            path.display(),
            fit.warnings
        );
        eprintln!("{}: no fallback", path.display());
        n += 1;
    }
    // Measured: 20 example models simulate and fit with `survival`, 19 without it (an
    // `[event_model]` example does not parse in a build without the feature).
    let want = if cfg!(feature = "survival") { 20 } else { 19 };
    assert!(
        n >= want,
        "only {n} example models were simulated and fitted"
    );
}

/// The standalone post-fit steps report a failed plan too (#2786 review #11): `run_covariance`
/// and `run_sir` register their own scope and fall back per subject exactly as `fit()` does, but
/// used to drop the failures with the scope without saying so. Dies when either entry point stops
/// calling `refresh_plan_fd_fallback`; the exact run, on a fit that already carries a stale
/// report, shows the incoming fit's line is replaced rather than inherited or duplicated.
#[test]
fn standalone_covariance_and_sir_report_the_fallback() {
    let (m, p) = example("examples/warfarin.ferx", "data/warfarin.csv");
    let base = FitOptions {
        sir_samples: 20,
        sir_resamples: 5,
        sir_seed: Some(1),
        ..opts(EstimationMethod::FoceI)
    };
    let mut fit = crate::api::fit(&m, &p, &m.default_params, &base).expect("fit");
    assert!(
        fit.covariance_matrix.is_some(),
        "the fit carries a covariance"
    );
    let n = p.subjects.len();
    // A stale report from the fit's own run must not survive into a step that ran clean.
    fit.warnings
        .push(format!("{TOKEN}: 99 of {n} subjects fell back (stale)"));
    let count = |r: &FitResult| r.warnings.iter().filter(|w| w.contains(TOKEN)).count();
    let typed = |r: &FitResult| {
        r.warnings_structured
            .iter()
            .filter(|e| e.message.contains(TOKEN))
            .count()
    };
    type Step = fn(&FitResult, &CompiledModel, &Population, &FitOptions) -> FitResult;
    let steps: [(&str, Step); 2] = [
        ("run_covariance", |f, m, p, o| {
            crate::run_covariance(f, Some(m), Some(p), o).expect("run_covariance")
        }),
        ("run_sir", |f, m, p, o| {
            crate::run_sir(f, Some(m), Some(p), o).expect("run_sir")
        }),
    ];
    for (label, step) in steps {
        let exact = step(&fit, &m, &p, &base);
        assert_eq!(
            count(&exact),
            0,
            "{label}: plans held: {:?}",
            exact.warnings
        );
        assert_eq!(typed(&exact), 0, "{label}: no typed entry either");
        let forced = plans_failing(|| step(&fit, &m, &p, &base));
        assert_eq!(count(&forced), 1, "{label}: {:?}", forced.warnings);
        assert_eq!(typed(&forced), 1, "{label}: one typed entry");
        let w = forced.warnings.iter().find(|w| w.contains(TOKEN)).unwrap();
        assert!(
            w.contains(super::FORCED_FAILURE),
            "{label}: the reason: {w}"
        );
        assert!(!w.contains("(stale)"), "{label}: replaced, not inherited");
        let e = forced
            .warnings_structured
            .iter()
            .find(|e| e.message.contains(TOKEN))
            .unwrap();
        assert_eq!(e.category, WarningCode::GradientFallback, "{label}");
    }
}

// ---- #2949: the plan's value reads ----------------------------------------------------------

/// Parse `model` (its `[data]` path rewritten to a file holding `data`) and read the population.
fn from_text(model: &str, data: &str) -> (CompiledModel, Population) {
    let dir = tempfile::tempdir().expect("tmp");
    let csv = dir.path().join("data.csv");
    std::fs::write(&csv, data).expect("write csv");
    let text = model.replace("path=data.csv", &format!("path={}", csv.display()));
    let ferx = dir.path().join("run.ferx");
    std::fs::write(&ferx, text).expect("write model");
    example(ferx.to_str().expect("utf8"), csv.to_str().expect("utf8"))
}

/// A FOCEI fit (3 outer iterations, one thread) reports no fallback and its plan outer
/// gradient matches the reconverged central difference; returns the gradient gap.
fn exact_and_matches_fd(label: &str, model: &str, data: &str) -> f64 {
    let (m, p) = from_text(model, data);
    let o = opts(EstimationMethod::FoceI);
    let fit = crate::api::fit(&m, &p, &m.default_params, &o).expect("fit");
    assert!(
        !fit.warnings.iter().any(|w| w.contains(TOKEN)),
        "{label}: {:?}",
        fit.warnings
    );
    let gap = outer_gradient_gap(&m, &p, &o);
    eprintln!("{label}: worst outer-gradient gap {gap:.3e}");
    gap
}

/// Bound on the plan-vs-reconverged-FD gap (relative to the gradient's largest entry).
const GAP_TOL: f64 = 1e-3;

const TWO_CPT_TV: &str = r#"[data]
  path=data.csv
[parameters]
  theta TVCL(3.0, 0.1, 150.0)
  theta TVV(50.0, 1.0, 2000.0)
  theta TVQ(2.0, 0.1, 50.0)
  theta TVV2(100.0, 1.0, 2500.0)
  theta TVKA(1.0, 0.01, 10.0)
  theta TH_WT(0.75, 0.01, 3.0)
  omega ETA_CL ~ 0.2
  omega ETA_V ~ 0.2
  sigma PROP_ERR ~ 0.2
[covariates]
  WT continuous
[individual_parameters]
  CL = TVCL * (WT / 70.0)^TH_WT * exp(ETA_CL)
  V  = TVV * exp(ETA_V)
  Q  = TVQ
  V2 = TVV2
  KA = TVKA
[structural_model]
  pk two_cpt_iv(cl=CL, v1=V, q=Q, v2=V2)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  maxiter    = 0
  covariance = false
"#;
const TWO_CPT_TV_DATA: &str = "ID,TIME,EVID,MDV,DV,AMT,RATE,CMT,WT
1,0,1,1,0,1000,0,1,70
1,12,0,0,5.0,0,0,1,90
";
const SS_RESET_F_LOGADD: &str = r#"[data]
  path=data.csv
[parameters]
  theta TVKA(8.0, FIX)
  theta TVV(1000.0, 1e-6, 10000.0)
  theta TVCL(100.0, 1e-6, 1000.0)
  omega ETA_KA ~ 1.0 FIX
  block_omega (ETA_V, ETA_CL) = [0.3, 0.1, 0.3]
  omega ETA_RUV ~ 0.1
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  F = 1e6 / 167.19
  KA = TVKA * exp(ETA_KA)
  V = TVV * 5.0 * exp(ETA_V)
  CL = TVCL * exp(ETA_CL)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, f=F)
[error_model]
  DV ~ log_additive(PROP_ERR)
  iiv_on_ruv = ETA_RUV
[fit_options]
  method     = focei
  maxiter    = 0
  covariance = false
"#;
const SS_RESET_F_LOGADD_DATA: &str = "ID,TIME,EVID,MDV,DV,AMT,CMT,II,SS
1,0,4,1,0,20,1,1,1
1,0.17,0,0,5.8,0,2,0,0
1,0.17,0,0,5.83,0,2,0,0
1,0.5,0,0,5.81,0,2,0,0
1,0.5,0,0,5.83,0,2,0,0
1,1,0,0,5.81,0,2,0,0
1,1,0,0,5.82,0,2,0,0
";

/// A two-compartment closed form whose covariate changes between records: the event walk reads
/// `CL`/`V` through the branch API (`cl_or_v_degenerate_g`), not `.val()`. The twin with a
/// constant covariate takes the superposition path (the straddle: only the time-varying
/// subjects reach the walk). Mutation: restore `cl_or_v_degenerate(cl.val(), v1.val())` in
/// `two_cpt_eigen_g` -> the time-varying cases fail on the token.
#[test]
fn a_two_compartment_time_varying_covariate_plans_without_fallback() {
    let constant = TWO_CPT_TV_DATA.replace("12,0,0,5.0,0,0,1,90", "12,0,0,5.0,0,0,1,70");
    assert_ne!(constant, TWO_CPT_TV_DATA);
    let infusion = TWO_CPT_TV_DATA.replace("1000,0,1,70", "1000,1000,1,70");
    assert_ne!(infusion, TWO_CPT_TV_DATA);
    let oral = TWO_CPT_TV.replace(
        "two_cpt_iv(cl=CL, v1=V, q=Q, v2=V2)",
        "two_cpt_oral(cl=CL, v1=V, q=Q, v2=V2, ka=KA)",
    );
    assert_ne!(oral, TWO_CPT_TV);
    let gaps = [
        exact_and_matches_fd("two_cpt const", TWO_CPT_TV, &constant),
        exact_and_matches_fd("two_cpt tv", TWO_CPT_TV, TWO_CPT_TV_DATA),
        exact_and_matches_fd("two_cpt tv infusion", TWO_CPT_TV, &infusion),
        exact_and_matches_fd("two_cpt oral tv", &oral, TWO_CPT_TV_DATA),
    ];
    for g in gaps {
        assert!(g.is_finite() && g < GAP_TOL, "{g:e}");
    }
}

/// `EVID=4` + `SS=1` + `F` + `log_additive` + `iiv_on_ruv`, and its `EVID=1` twin (which
/// planned before the fix: the straddle).
#[test]
fn a_reset_steady_state_dose_with_f_and_ruv_eta_plans_without_fallback() {
    let evid1 = SS_RESET_F_LOGADD_DATA.replace("1,0,4,1,0,20", "1,0,1,1,0,20");
    assert_ne!(evid1, SS_RESET_F_LOGADD_DATA);
    let gaps = [
        exact_and_matches_fd(
            "ss_reset_f_logadd",
            SS_RESET_F_LOGADD,
            SS_RESET_F_LOGADD_DATA,
        ),
        exact_and_matches_fd("ss_evid1_f_logadd", SS_RESET_F_LOGADD, &evid1),
    ];
    for g in gaps {
        assert!(g.is_finite() && g < GAP_TOL, "{g:e}");
    }
}

const TWO_CPT_TIME: &str = r#"[data]
  path=data.csv
[parameters]
  theta TVCL(3.0, 0.1, 150.0)
  theta TVV(50.0, 1.0, 2000.0)
  theta TVQ(2.0, 0.1, 50.0)
  theta TVV2(100.0, 1.0, 2500.0)
  theta IMAX(0.4, 0.01, 3.0)
  theta TI50(20.0, 1.0, 500.0)
  theta HILL(1.5, 0.5, 5.0)
  omega ETA_CL ~ 0.2
  omega ETA_V ~ 0.2
  sigma PROP_ERR ~ 0.2
[individual_parameters]
  CL = TVCL * exp(IMAX*TIME^HILL/(TI50^HILL+TIME^HILL)) * exp(ETA_CL)
  V  = TVV * exp(ETA_V)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  pk two_cpt_iv(cl=CL, v1=V, q=Q, v2=V2)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  maxiter    = 0
  covariance = false
"#;
const TWO_CPT_TIME_DATA: &str = "ID,TIME,EVID,MDV,DV,AMT,RATE,CMT
1,0,1,1,0,1000,0,1
1,6,0,0,9.0,0,0,1
1,24,0,0,5.0,0,0,1
1,48,0,0,3.5,0,0,1
";

/// The pembrolizumab shape: `TIME` inside an individual parameter of a two-compartment closed
/// form (no covariate column at all). Control: the same model with the `TIME` term removed
/// takes the superposition path.
#[test]
fn a_two_compartment_time_in_parameter_plans_without_fallback() {
    let g = exact_and_matches_fd("two_cpt TIME in CL", TWO_CPT_TIME, TWO_CPT_TIME_DATA);
    let no_time = TWO_CPT_TIME.replace("exp(IMAX*TIME^HILL/(TI50^HILL+TIME^HILL)) * ", "");
    assert_ne!(no_time, TWO_CPT_TIME);
    let g0 = exact_and_matches_fd("two_cpt no TIME", &no_time, TWO_CPT_TIME_DATA);
    for g in [g, g0] {
        assert!(g.is_finite() && g < GAP_TOL, "{g:e}");
    }
}

/// The realised-route labels name the plan fallback: with every plan failing the fit's inner
/// and outer labels say finite differences, never `analytic`; with the plans working they
/// say `analytic`. The split form is pinned on `plan_fallback_label` directly.
#[test]
fn the_route_labels_name_a_plan_fallback() {
    let (m, p) = from_text(TWO_CPT_TV, TWO_CPT_TV_DATA);
    let (exact, forced) = pair(&m, &p, &opts(EstimationMethod::FoceI));
    assert_eq!(exact.gradient_method_inner, "analytic");
    assert_eq!(exact.gradient_method_outer, "analytic");
    for l in [&forced.gradient_method_inner, &forced.gradient_method_outer] {
        assert_eq!(l, "finite differences (compiled plan failed)");
    }
    use crate::api::postfit::plan_fallback_label as lab;
    assert_eq!(lab("analytic", 0, 4), "analytic");
    assert_eq!(lab("analytic", 1, 4), "analytic ×3, FD ×1");
    assert_eq!(
        lab("analytic", 4, 4),
        "finite differences (compiled plan failed)"
    );
    assert_eq!(lab("N/A", 4, 4), "N/A");
}

const ONE_CPT_SS_INFUSION: &str = r#"[data]
  path=data.csv
[parameters]
  theta TVCL(3.0, 0.1, 150.0)
  theta TVV(50.0, 1.0, 2000.0)
  omega ETA_CL ~ 0.2
  omega ETA_V ~ 0.2
  sigma PROP_ERR ~ 0.2
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  maxiter    = 0
  covariance = false
"#;

/// An overlapping steady-state infusion (`dur = AMT/RATE = 10 > II = 6`) sums its still-running
/// pulses by the phase's value; a recording guards that count instead of reading the value
/// (`mode_infusion_ss`). Control: the non-overlapping twin (`II = 12`).
#[test]
fn an_overlapping_steady_state_infusion_plans_without_fallback() {
    let data = "ID,TIME,EVID,MDV,DV,AMT,RATE,CMT,II,SS
1,0,1,1,0,1000,100,1,6,1
1,2,0,0,9.0,0,0,1,0,0
1,5,0,0,12.0,0,0,1,0,0
1,8,0,0,8.0,0,0,1,0,0
";
    let plain = data.replace("100,1,6,1", "100,1,12,1");
    assert_ne!(plain, data);
    let gaps = [
        exact_and_matches_fd("ss overlap", ONE_CPT_SS_INFUSION, data),
        exact_and_matches_fd("ss no overlap", ONE_CPT_SS_INFUSION, &plain),
    ];
    for g in gaps {
        assert!(g.is_finite() && g < GAP_TOL, "{g:e}");
    }
}
