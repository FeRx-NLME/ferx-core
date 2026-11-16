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
/// used to drop the failures with the scope without saying so. The step's line names the step
/// (`W_PLAN_FD_FALLBACK (the covariance step):`) and is *added*: the incoming fit's own report
/// of its estimation is kept, never rewritten from the step's counts.
/// Mutation: drop the `refresh_plan_fd_fallback` call of either step -> its forced run has no step
/// line; strip the incoming `W_PLAN_FD_FALLBACK` again -> the "(from the fit)" line is gone.
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
    // The estimation's own report, which a step did not rerun.
    let own = format!("{TOKEN}: 99 of {n} subjects fell back (from the fit)");
    fit.warnings.push(own.clone());
    type Step = fn(&FitResult, &CompiledModel, &Population, &FitOptions) -> FitResult;
    let steps: [(&str, &str, Step); 2] = [
        ("run_covariance", "the covariance step", |f, m, p, o| {
            crate::run_covariance(f, Some(m), Some(p), o).expect("run_covariance")
        }),
        ("run_sir", "the SIR step", |f, m, p, o| {
            crate::run_sir(f, Some(m), Some(p), o).expect("run_sir")
        }),
    ];
    for (label, name, step) in steps {
        let tag = format!("{TOKEN} ({name}):");
        let of_step = |r: &FitResult| r.warnings.iter().filter(|w| w.starts_with(&tag)).count();
        let exact = step(&fit, &m, &p, &base);
        assert_eq!(
            of_step(&exact),
            0,
            "{label}: plans held: {:?}",
            exact.warnings
        );
        assert!(
            exact.warnings.contains(&own),
            "{label}: the fit's report is kept"
        );
        let forced = plans_failing(|| step(&fit, &m, &p, &base));
        assert_eq!(of_step(&forced), 1, "{label}: {:?}", forced.warnings);
        assert!(
            forced.warnings.contains(&own),
            "{label}: the fit's report is kept"
        );
        let w = forced
            .warnings
            .iter()
            .find(|w| w.starts_with(&tag))
            .unwrap();
        assert!(
            w.contains(super::FORCED_FAILURE),
            "{label}: the reason: {w}"
        );
        assert!(w.contains(name), "{label}: names the step: {w}");
        let e = forced
            .warnings_structured
            .iter()
            .find(|e| e.message.starts_with(&tag))
            .expect("typed entry");
        assert_eq!(e.category, WarningCode::GradientFallback, "{label}");
        // A step run over its own result replaces its line rather than doubling it.
        let again = plans_failing(|| step(&forced, &m, &p, &base));
        assert_eq!(of_step(&again), 1, "{label}: not doubled");
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

/// An amount-state ODE whose `[scaling]` divisor reads an individual parameter (`obs_scale = V1`),
/// the idiom of every amount model. Its first fit overshoots: the inner search steps to an extreme
/// η, where `V1 = TVV1·exp(η)` underflows to 0 and the recording's scale check failed — the
/// whole subject went to finite differences for the rest of the fit ("the recorder returned
/// None", all 6 subjects, and a converged fit that is not converged under it). The fit reports
/// no fallback and its plan outer gradient matches the reconverged central difference.
/// Mutation: restore `return None` at the divisor check in `record_ode` -> the token appears.
#[test]
fn an_ode_expression_scale_on_an_individual_parameter_plans_without_fallback() {
    let model = TWO_CPT_AMOUNTS;
    let data = "ID,TIME,EVID,MDV,DV,AMT,RATE,CMT\n1,0,1,1,.,100,100,1\n1,0.5,0,0,4.2931,0,0,1\n1,1.0,0,0,4.2024,0,0,1\n1,2.0,0,0,3.2577,0,0,1\n1,4.0,0,0,2.9236,0,0,1\n1,8.0,0,0,1.9541,0,0,1\n1,24.0,0,0,0.5111,0,0,1\n2,0,1,1,.,100,100,1\n2,0.5,0,0,3.9917,0,0,1\n2,1.0,0,0,3.9186,0,0,1\n2,2.0,0,0,3.6982,0,0,1\n2,4.0,0,0,2.7256,0,0,1\n2,8.0,0,0,1.8316,0,0,1\n2,24.0,0,0,0.5425,0,0,1\n3,0,1,1,.,100,100,1\n3,0.5,0,0,3.6904,0,0,1\n3,1.0,0,0,3.6348,0,0,1\n3,2.0,0,0,3.4465,0,0,1\n3,4.0,0,0,2.5276,0,0,1\n3,8.0,0,0,1.7090,0,0,1\n3,24.0,0,0,0.5245,0,0,1\n4,0,1,1,.,100,100,1\n4,0.5,0,0,4.2177,0,0,1\n4,1.0,0,0,4.1315,0,0,1\n4,2.0,0,0,3.1948,0,0,1\n4,4.0,0,0,2.8741,0,0,1\n4,8.0,0,0,1.9235,0,0,1\n4,24.0,0,0,0.5066,0,0,1\n5,0,1,1,.,100,100,1\n5,0.5,0,0,3.9164,0,0,1\n5,1.0,0,0,3.8477,0,0,1\n5,2.0,0,0,3.6353,0,0,1\n5,4.0,0,0,2.6761,0,0,1\n5,8.0,0,0,1.8009,0,0,1\n5,24.0,0,0,0.5380,0,0,1\n6,0,1,1,.,100,100,1\n6,0.5,0,0,4.4438,0,0,1\n6,1.0,0,0,3.5639,0,0,1\n6,2.0,0,0,3.3836,0,0,1\n6,4.0,0,0,3.0226,0,0,1\n6,8.0,0,0,1.6784,0,0,1\n6,24.0,0,0,0.5200,0,0,1";
    let (m, p) = from_text(model, data);
    let o = FitOptions {
        outer_maxiter: 25,
        ..opts(EstimationMethod::FoceI)
    };
    let fit = crate::api::fit(&m, &p, &m.default_params, &o).expect("fit");
    assert!(
        !fit.warnings.iter().any(|w| w.contains(TOKEN)),
        "{:?}",
        fit.warnings
    );
    assert!(fit.converged, "converged");
    let gap = exact_and_matches_fd("ode obs_scale = V1", model, data);
    assert!(gap.is_finite() && gap < GAP_TOL, "{gap:e}");
}

const TWO_CPT_AMOUNTS: &str = r#"[data]
  path=data.csv
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV1(10.0, 0.5, 500.0)
  theta TVQ(2.0, 0.05, 50.0)
  theta TVV2(20.0, 0.5, 1000.0)
  omega ETA_CL ~ 0.2
  omega ETA_V1 ~ 0.2
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  ode(obs_cmt=central, states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  obs_scale = V1
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  maxiter    = 0
  covariance = false
"#;

// ---- Code-review findings on the failure report (#2786) -------------------------------------

use super::{
    current_fit, in_fit, note_failure, note_model_failure, plan_failure_counts, register, state_of,
    SubjectKey,
};

/// A closed-form transit model whose subjects reroute to its ODE twin (a time-varying `WT`).
const TRANSIT_TV: &str = r#"[data]
  path=data.csv
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVMTT(1.0, 0.05, 24.0)
  theta TVN(3.0, 0.0, 30.0)
  theta TH_WT(0.75, 0.01, 3.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[covariates]
  WT continuous
[individual_parameters]
  CL  = TVCL * (WT / 70.0)^TH_WT * exp(ETA_CL)
  V   = TVV * exp(ETA_V)
  MTT = TVMTT
  NTR = TVN
[structural_model]
  pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  maxiter    = 0
  covariance = false
"#;
const TRANSIT_TV_DATA: &str = "ID,TIME,EVID,MDV,DV,AMT,RATE,CMT,WT
1,0,1,1,0,100,0,1,70
1,1,0,0,1.1,0,0,2,70
1,4,0,0,1.4,0,0,2,90
1,12,0,0,0.9,0,0,2,90
2,0,1,1,0,100,0,1,60
2,1,0,0,1.3,0,0,2,60
2,4,0,0,1.6,0,0,2,80
2,12,0,0,1.0,0,0,2,80
3,0,1,1,0,100,0,1,80
3,1,0,0,0.9,0,0,2,80
3,4,0,0,1.2,0,0,2,100
3,12,0,0,0.8,0,0,2,100
";

fn one_fit(m: &CompiledModel, p: &Population) -> FitResult {
    crate::api::fit(m, p, &m.default_params, &opts(EstimationMethod::FoceI)).expect("fit")
}

/// Subjects rerouted to a closed-form model's ODE twin plan under the twin's own id: a failure
/// there is the fit's — warning, label — and leaves with the fit, so a later fit records fresh
/// instead of skipping to finite differences silently.
/// Mutation: drop the twin's id from `plan_failure_counts` -> no warning in the forced fit;
/// stop `drop_scope` clearing `failures`/`bad_models` -> the twin's state outlives the fit.
#[test]
fn a_twin_plan_failure_is_reported_and_leaves_with_the_fit() {
    let (m, p) = from_text(TRANSIT_TV, TRANSIT_TV_DATA);
    let twin = m
        .absorption_ode_equivalent
        .as_ref()
        .expect("a transit model carries its ODE twin")
        .built();
    let (twin_id, id) = (twin.plan_id.get(), m.plan_id.get());
    let forced = plans_failing(|| one_fit(&m, &p));
    let n = p.subjects.len();
    assert!(
        forced
            .warnings
            .iter()
            .any(|w| w.contains(TOKEN) && w.contains(&format!("{n} of {n} subjects"))),
        "{:?}",
        forced.warnings
    );
    assert_eq!(
        forced.gradient_method_inner,
        "finite differences (compiled plan failed)"
    );
    // Nothing of the failed fit is left: not the twin's marks, not the model's.
    assert_eq!(state_of(twin_id), 0, "the twin's state outlived the fit");
    assert_eq!(state_of(id), 0, "the model's state outlived the fit");
    // A later fit, plans unforced, records fresh and reports nothing.
    let later = one_fit(&m, &p);
    assert!(
        !later.warnings.iter().any(|w| w.contains(TOKEN)),
        "{:?}",
        later.warnings
    );
    assert_eq!(later.gradient_method_inner, "analytic");
    // A failure filed under the twin's own id alone is the fit's too (the fit above may also
    // have failed plans of the model itself, which would hide a twin that is not counted).
    let _g = register(&m, &p.subjects);
    assert_eq!(plan_failure_counts(&m, &p.subjects), None);
    note_failure(&SubjectKey::new(twin, &p.subjects[1], 2, 6), "the twin's");
    let c = plan_failure_counts(&m, &p.subjects).expect("the twin's failure is the fit's");
    assert_eq!((c.any, c.inner), (1, 1));
}

/// Fits of one model do not share failures: one whose plans fail reports them while another,
/// registered over the same model at the same time, reports nothing — and a failure is counted
/// for the subjects it was noted for, not guessed from the failure set's size.
/// Mutation: key a registered subject's plans under the model's scope -> the second fit's
/// subjects (clones: the same designs) are counted; restore the `min(n)` guess -> the stranger's
/// failure below counts one subject.
#[test]
fn overlapping_fits_of_one_model_keep_their_failures_apart() {
    let (m, p) = from_text(TWO_CPT_TV, TWO_CPT_TV_DATA);
    let (pa, pb) = (p.clone(), p.clone());
    // Each fit's registration on a thread that runs no fit (a registration inside a fit of the
    // model joins it), and each fit's calls under its own id, as its pool's tasks run them.
    fn reg<'a>(m: &'a CompiledModel, pop: &'a Population) -> (super::Prints<'a>, u64) {
        in_fit(0, || {
            let g = register(m, &pop.subjects);
            (g, current_fit())
        })
    }
    let ((_a, id_a), (_b, id_b)) = (reg(&m, &pa), reg(&m, &pb));
    let key = |s: &crate::types::Subject| SubjectKey::new(&m, s, 2, 6);
    in_fit(id_a, || note_failure(&key(&pa.subjects[0]), "forced"));
    let a = in_fit(id_a, || plan_failure_counts(&m, &pa.subjects)).expect("A's failure");
    assert_eq!((a.any, a.inner, a.outer), (1, 1, 1));
    assert_eq!(
        in_fit(id_b, || plan_failure_counts(&m, &pb.subjects)),
        None,
        "B's subjects share A's designs and nothing else"
    );
    // A failure of a subject the population does not hold counts for none of its subjects.
    let mut other = pa.subjects[0].clone();
    other.doses[0].amt += 1.0;
    in_fit(id_a, || note_failure(&key(&other), "a cut copy"));
    let a = in_fit(id_a, || plan_failure_counts(&m, &pa.subjects)).expect("A's failure");
    assert_eq!(a.any, 1, "a stranger's failure is not a guess at A's");
    // The same through whole fits, concurrently: only the one whose plans fail reports.
    let (m2, p2) = from_text(TWO_CPT_TV, TWO_CPT_TV_DATA);
    let (q2, q3) = (p2.clone(), p2.clone());
    let barrier = std::sync::Barrier::new(2);
    let (fa, fb) = std::thread::scope(|sc| {
        let a = sc.spawn(|| {
            barrier.wait();
            plans_failing(|| one_fit(&m2, &q2))
        });
        let b = sc.spawn(|| {
            barrier.wait();
            one_fit(&m2, &q3)
        });
        (a.join().expect("A"), b.join().expect("B"))
    });
    assert!(fa.warnings.iter().any(|w| w.contains(TOKEN)));
    assert!(
        !fb.warnings.iter().any(|w| w.contains(TOKEN)),
        "{:?}",
        fb.warnings
    );
    assert_eq!(fb.gradient_method_inner, "analytic");
}

/// Fits over the very same model and population objects (a multi-start) keep their plan
/// failures apart: a registration is a fit of its own, and every lookup reads the fit its thread
/// runs, across the pool's tasks too. One fit's plans are forced to fail while another's, at the
/// same time over the same objects and with two workers, are not: only the first reports.
/// Mutation: key a registered address by address alone (`registered` ignoring `CUR_FIT`) -> the
/// second fit reports the first's failures.
#[test]
fn fits_over_the_same_objects_keep_their_plan_failures_apart() {
    let (m, p) = from_text(TWO_CPT_TV, TWO_CPT_TV_DATA);
    let o = FitOptions {
        threads: Some(2),
        ..opts(EstimationMethod::FoceI)
    };
    let fit = || crate::api::fit(&m, &p, &m.default_params, &o).expect("fit");
    let barrier = std::sync::Barrier::new(2);
    let (fa, fb) = std::thread::scope(|sc| {
        let a = sc.spawn(|| {
            barrier.wait();
            plans_failing(fit)
        });
        let b = sc.spawn(|| {
            barrier.wait();
            fit()
        });
        (a.join().expect("A"), b.join().expect("B"))
    });
    let n = p.subjects.len();
    assert!(
        fa.warnings
            .iter()
            .any(|w| w.contains(TOKEN) && w.contains(&format!("{n} of {n} subjects"))),
        "{:?}",
        fa.warnings
    );
    assert!(
        !fb.warnings.iter().any(|w| w.contains(TOKEN)),
        "{:?}",
        fb.warnings
    );
    assert_eq!(fb.gradient_method_inner, "analytic");
}

/// Registrations of one object are fits of their own: distinct scopes, each read under the
/// thread's fit; a registration made inside a fit of the model joins it (one scope, so its plans
/// are not recorded again), and the thread's fit is restored as they leave.
/// Mutation: let a nested registration open a scope -> the joined assertion; let a guard leave
/// its id set -> the restored assertions.
#[test]
fn registration_scopes_are_per_fit_and_nested_ones_join() {
    use super::registered;
    let (m, p) = from_text(TWO_CPT_TV, TWO_CPT_TV_DATA);
    let addr = p.subjects.as_ptr() as usize;
    assert_eq!(current_fit(), 0);
    let a = register(&m, &p.subjects);
    let fit_a = current_fit();
    assert_ne!(fit_a, 0);
    let nested = register(&m, &p.subjects);
    assert_eq!(current_fit(), fit_a, "joined");
    drop(nested);
    assert_eq!(current_fit(), fit_a);
    let b = in_fit(0, || {
        let b = register(&m, &p.subjects);
        let fit_b = current_fit();
        assert_ne!(fit_b, fit_a);
        assert_eq!(registered(addr).expect("b").1, fit_b);
        in_fit(fit_a, || assert_eq!(registered(addr).expect("a").1, fit_a));
        b
    });
    drop(b);
    assert_eq!(current_fit(), fit_a, "the other fit's registration left it");
    assert_eq!(registered(addr).expect("still a").1, fit_a);
    drop(a);
    assert_eq!(current_fit(), 0);
    assert!(registered(addr).is_none());
}

/// A θ-only recording that fell back (`note_model_failure`) is the outer gradient's, so the
/// inner route keeps its label: the warning is there, `gradient_method_outer` says finite
/// differences, `gradient_method_inner` still says analytic. A covariance-step plan failure
/// (kind 9) is neither's: reported, no label moves.
/// Mutation: label both routes from the total count -> the inner label reads FD.
#[test]
fn an_outer_only_failure_keeps_the_inner_label() {
    let (m, p) = from_text(TWO_CPT_TV, TWO_CPT_TV_DATA);
    let o = opts(EstimationMethod::FoceI);
    // Registered over the model first, so the fit's own registration (over the scored copy of the
    // population, which is no registered object) joins it.
    let guard = register(&m, &p.subjects);
    note_model_failure(&m, "a θ-only recording: test");
    let fit = crate::api::fit(&m, &p, &m.default_params, &o).expect("fit");
    assert!(
        fit.warnings.iter().any(|w| w.contains(TOKEN)),
        "{:?}",
        fit.warnings
    );
    assert_eq!(fit.gradient_method_inner, "analytic");
    assert_eq!(
        fit.gradient_method_outer,
        "finite differences (compiled plan failed)"
    );
    drop(guard);
    // The counts, per route.
    let _g = register(&m, &p.subjects);
    let key = SubjectKey::new(&m, &p.subjects[0], 2, 6);
    note_failure(&key.with_kind(9), "covariance");
    let c = plan_failure_counts(&m, &p.subjects).expect("counted");
    assert_eq!((c.any, c.inner, c.outer), (1, 0, 0));
    note_model_failure(&m, "θ");
    let c = plan_failure_counts(&m, &p.subjects).expect("counted");
    let n = p.subjects.len();
    assert_eq!((c.any, c.inner, c.outer), (n, 0, n));
}

/// Failures are keyed as their plans are: a likelihood kernel bakes the observed values in, so
/// two subjects of one design with other data are not one failure.
/// Mutation: key a failure by design alone -> both subjects count.
#[test]
fn a_kernel_failure_counts_the_subject_whose_data_it_is() {
    let data = "ID,TIME,EVID,MDV,DV,AMT,RATE,CMT,WT
1,0,1,1,0,1000,0,1,70
1,12,0,0,5.0,0,0,1,90
2,0,1,1,0,1000,0,1,70
2,12,0,0,6.5,0,0,1,90
";
    let (m, p) = from_text(TWO_CPT_TV, data);
    let _g = register(&m, &p.subjects);
    let key =
        |s: &crate::types::Subject| SubjectKey::new(&m, s, 0, 3).with_kind(8).with_observed(s);
    assert_ne!(
        key(&p.subjects[0]),
        key(&p.subjects[1]),
        "fixture: data differ"
    );
    note_failure(&key(&p.subjects[1]), "kernel");
    let c = plan_failure_counts(&m, &p.subjects).expect("counted");
    assert_eq!(c.any, 1);
    // A design-keyed plan (the predictions) is shared by both, and both fell back.
    note_failure(&SubjectKey::new(&m, &p.subjects[0], 2, 6), "predictions");
    let c = plan_failure_counts(&m, &p.subjects).expect("counted");
    assert_eq!(c.any, 2);
}

/// A point failure is the subject's that failed, not every subject of its design (#2786 review
/// round 4): two registered subjects with identical dosing, times and covariates share one plan,
/// but only the one whose recording failed at its point fell back. A plan condemned for its
/// design (`note_failure`, the structural and model scopes) is every sharer's.
/// Mutation: key a point failure by design ([`note_failure`]) -> both subjects count.
#[test]
fn a_point_failure_counts_the_subject_that_failed_not_its_design_twins() {
    let data = "ID,TIME,EVID,MDV,DV,AMT,RATE,CMT,WT
1,0,1,1,0,1000,0,1,70
1,12,0,0,5.0,0,0,1,90
2,0,1,1,0,1000,0,1,70
2,12,0,0,5.0,0,0,1,90
";
    let (m, p) = from_text(TWO_CPT_TV, data);
    let _g = register(&m, &p.subjects);
    let key = |s: &crate::types::Subject| SubjectKey::new(&m, s, 2, 6);
    assert_eq!(
        key(&p.subjects[0]),
        key(&p.subjects[1]),
        "fixture: one design, one shared plan"
    );
    super::note_point_failure(&key(&p.subjects[1]), "point");
    let c = plan_failure_counts(&m, &p.subjects).expect("counted");
    assert_eq!((c.any, c.inner, c.outer), (1, 1, 1));
    // A condemned design is both subjects'.
    note_failure(&key(&p.subjects[0]), "design");
    let c = plan_failure_counts(&m, &p.subjects).expect("counted");
    assert_eq!(c.any, 2);
}

/// A fit whose outer route is already finite differences (Gauss-Newton: its outer step reads no
/// analytic gradient) keeps that label when plans fail: no plan fed it, so none failed it, and
/// "(compiled plan failed)" there would blame the plans for a route they never took (#2786
/// round 2, finding 3). The inner route is the plans', and is rewritten. Mutation: let
/// `plan_fallback_label` rewrite any label but `N/A` -> the outer label changes.
#[test]
fn a_finite_difference_outer_route_keeps_its_label_when_plans_fail() {
    let (m, p) = example("examples/warfarin.ferx", "data/warfarin.csv");
    let (exact, forced) = pair(&m, &p, &opts(EstimationMethod::FoceGn));
    assert_eq!(exact.gradient_method_outer, "finite differences");
    assert_eq!(forced.gradient_method_outer, "finite differences");
    assert_eq!(exact.gradient_method_inner, "analytic");
    assert_eq!(
        forced.gradient_method_inner,
        "finite differences (compiled plan failed)"
    );
    use crate::api::postfit::plan_fallback_label as lab;
    assert_eq!(lab("finite differences", 1, 4), "finite differences");
    assert_eq!(lab("finite differences", 4, 4), "finite differences");
}

/// A standalone post-fit step does not rewrite the incoming fit's record of its estimation
/// (#2786 review round 4): over a fit that carries FD labels and the warning, a clean
/// `run_covariance` keeps both — the estimation's gradients were computed that way whatever the
/// step did — and a step whose own plans fail adds its own named line beside them.
/// Mutation: let `refresh_plan_fd_fallback` call `apply_plan_failures` with the step's counts ->
/// the labels are restored to `exact`'s and the warning is stripped.
#[test]
fn a_standalone_step_keeps_the_fits_route_labels_and_warning() {
    let (m, p) = example("examples/warfarin.ferx", "data/warfarin.csv");
    let o = opts(EstimationMethod::FoceI);
    let (exact, forced) = pair(&m, &p, &o);
    assert!(forced.warnings.iter().any(|w| w.starts_with(TOKEN)));
    assert_ne!(forced.gradient_method_inner, exact.gradient_method_inner);
    let out = crate::run_covariance(&forced, Some(&m), Some(&p), &o).expect("run_covariance");
    assert_eq!(out.gradient_method_inner, forced.gradient_method_inner);
    assert_eq!(out.gradient_method_outer, forced.gradient_method_outer);
    let kept = |r: &FitResult| {
        r.warnings
            .iter()
            .filter(|w| w.starts_with(&format!("{TOKEN}:")))
            .count()
    };
    assert_eq!(kept(&out), kept(&forced), "the fit's warning is kept");
    assert!(
        !out.warnings
            .iter()
            .any(|w| w.contains("(the covariance step)")),
        "a clean step adds none: {:?}",
        out.warnings
    );
    // Its own failure is a separate line naming the step; the labels are still the fit's.
    let failed = plans_failing(|| {
        crate::run_covariance(&exact, Some(&m), Some(&p), &o).expect("run_covariance")
    });
    let lines = failed
        .warnings
        .iter()
        .filter(|w| w.contains("(the covariance step)"))
        .count();
    assert_eq!(lines, 1, "{:?}", failed.warnings);
    assert_eq!(failed.gradient_method_inner, exact.gradient_method_inner);
    assert_eq!(failed.gradient_method_outer, exact.gradient_method_outer);
    // And a split label is undone by the fit's own `apply_plan_failures`: `analytic ×a, FD ×k`
    // restarts from `analytic`.
    let mut split = exact.clone();
    split.gradient_method_inner = "analytic ×3, FD ×1".into();
    crate::api::postfit::apply_plan_failures(&mut split, None, 4);
    assert_eq!(split.gradient_method_inner, "analytic");
}

/// The inner solve decides which model it actually evaluated once per point, not once per
/// consumer (#2786 round 2, finding 8): the stall stop (per iterate) and the fused-value checks
/// (per gradient and per fused trial) used to each call `effective_model_for_eval` at the same
/// point. Both routes are exercised — a transit model with a time-varying covariate (its ODE
/// twin: stall on) and a closed form (fused values memoised) — and no two consecutive decisions
/// are for the same η. Mutation: make `route_at` recompute without its cache -> consecutive
/// repeats appear.
#[test]
fn the_inner_solve_decides_the_evaluated_model_once_per_point() {
    use crate::estimation::inner_optimizer::{find_ebe, EVAL_ROUTE_LOG};
    let fixtures = [
        (
            "transit with a time-varying covariate",
            from_text(TRANSIT_TV, TRANSIT_TV_DATA),
        ),
        (
            "closed form",
            example("examples/warfarin.ferx", "data/warfarin.csv"),
        ),
    ];
    for (label, (m, p)) in fixtures {
        EVAL_ROUTE_LOG.with(|l| l.borrow_mut().clear());
        for s in p.subjects.iter().take(3) {
            let _ = find_ebe(&m, s, &m.default_params, 200, 1e-6, None, None, 0);
        }
        let log = EVAL_ROUTE_LOG.with(|l| l.borrow().clone());
        assert!(
            log.len() >= 3,
            "{label}: the solve decided {} points",
            log.len()
        );
        let repeats = log.windows(2).filter(|w| w[0] == w[1]).count();
        assert_eq!(
            repeats,
            0,
            "{label}: {repeats} of {} decisions repeat",
            log.len()
        );
    }
}

/// The model language's `floor` / `ceil` / `round` of an input-dependent value (an η-dependent
/// expression) record as primitives: they used to read the value (`val` fails a recording), so
/// every model that used one on an η fell back to finite differences. The terms are multiplied
/// by zero, so the fit is warfarin's; what is pinned is that the plan serves it. Dies when
/// `apply_unary` reads a value again, or `Tr::{floor,ceil,round}_c` stop recording.
#[test]
fn floor_ceil_round_of_a_dependent_value_do_not_fall_back() {
    let text = std::fs::read_to_string("examples/warfarin.ferx").expect("read");
    let patched = text.replace(
        "CL = TVCL * exp(ETA_CL)",
        "CL = TVCL * exp(ETA_CL) + 0.0 * (floor(ETA_CL * 40.0 + 0.3) + ceil(ETA_V * 40.0) \
         + round(ETA_KA * 40.0))",
    );
    assert_ne!(patched, text, "the fixture line moved");
    let dir = tempfile::tempdir().expect("tmp");
    let file = dir.path().join("w.ferx");
    std::fs::write(&file, patched).expect("write");
    let parsed = crate::parser::model_parser::parse_full_model_file(&file).expect("parse");
    let (pop, _) = crate::api::read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        "data/warfarin.csv",
        None,
        parsed.fit_options.iov_column.as_deref(),
        None,
        &parsed.column_map,
    )
    .expect("read data");
    let o = opts(EstimationMethod::FoceI);
    let fit = crate::api::fit(&parsed.model, &pop, &parsed.model.default_params, &o).expect("fit");
    assert!(
        !fit.warnings.iter().any(|w| w.contains(TOKEN)),
        "{:?}",
        fit.warnings
    );
    assert!(fit.ofv.is_finite());
    // Straddle: the same fit with every plan failing does report the fallback, so the absence
    // above is the plan serving the subject, not the warning being unreachable.
    let forced = plans_failing(|| {
        crate::api::fit(&parsed.model, &pop, &parsed.model.default_params, &o).expect("fit")
    });
    assert!(forced.warnings.iter().any(|w| w.contains(TOKEN)));
}

// ---- #2786 G12: steady-state ODE fixtures that once looked like point failures ---------------

/// A three-subject steady-state bolus dose (`II = ii`) into the two-compartment amounts ODE.
fn ss_bolus_data(ii: f64) -> String {
    let mut s = String::from("ID,TIME,EVID,MDV,DV,AMT,RATE,CMT,II,SS\n");
    for (id, dv) in [
        (1, [4.0, 3.5, 3.0]),
        (2, [4.2, 3.7, 3.1]),
        (3, [3.8, 3.4, 2.9]),
    ] {
        s += &format!("{id},0,1,1,.,100,0,1,{ii},1\n");
        for (t, y) in [0.5, 1.0, 2.0].iter().zip(dv) {
            s += &format!("{id},{t},0,0,{y},0,0,1,0,0\n");
        }
    }
    s
}

/// The audit's O2 (`fail_here` Point failures in `record_fixed` / the ODE segment's record-time
/// run) named a short `II` against a slow elimination and a step-limited stiff ODE as the shapes
/// that might reach them. Measured: neither does. A cycle count past `SS_EQUILIBRATION_CYCLES`
/// serves as `Found::Iterated`, and a step-limited segment's freeze-padded tail is served through
/// `LaneHook::tail`, so both fits report no `W_PLAN_FD_FALLBACK`. Pins that route: a regression
/// that makes either decline at its record-time run puts the token back. (No mutation check
/// was run: both cases are measured green on the unmodified tree, the sites are not reached.)
#[test]
fn short_interval_and_step_limited_steady_states_plan_without_fallback() {
    for (label, cl, q, ii, extra) in [
        ("tiny II", "5.0", "2.0", 0.001, ""),
        (
            "fast Q, step-limited",
            "5.0",
            "50000.0",
            1.0,
            "\n  ode_max_steps = 40",
        ),
    ] {
        let model = TWO_CPT_AMOUNTS
            .replace(
                "theta TVCL(5.0, 0.1, 100.0)",
                &format!("theta TVCL({cl}, 1e-6, 100.0)"),
            )
            .replace(
                "theta TVQ(2.0, 0.05, 50.0)",
                &format!("theta TVQ({q}, 0.05, 1e6)"),
            )
            .replace("covariance = false", &format!("covariance = false{extra}"));
        let (m, p) = from_text(&model, &ss_bolus_data(ii));
        let o = FitOptions {
            outer_maxiter: 3,
            ..opts(EstimationMethod::FoceI)
        };
        let fit = crate::api::fit(&m, &p, &m.default_params, &o).expect("fit");
        assert!(
            !fit.warnings.iter().any(|w| w.contains(TOKEN)),
            "{label}: {:?}",
            fit.warnings
        );
        assert!(fit.ofv.is_finite(), "{label}: {}", fit.ofv);
    }
}

const GROWTH: &str = r#"[data]
  path=data.csv
[parameters]
  theta TVK(3.5e-6, 1e-8, 1e-3)
  omega ETA_K ~ 0.01
  sigma PROP_ERR ~ 0.2
[individual_parameters]
  K = TVK * exp(ETA_K)
[structural_model]
  ode(obs_cmt=A, states=[A])
[odes]
  d/dt(A) = (K * 1000000.0) * A
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  maxiter    = 0
  covariance = false
"#;

/// The audit's O3: a state of ~1e304 (the #1204 repro's scale) is a point the VALUE path repels
/// before any derivative exists: a Gaussian residual variance `(σ·f)²` overflows from `f ≈ 1e154`,
/// so the subject scores the `2e20` sentinel and the exact gradient is declined. No
/// `W_PLAN_FD_FALLBACK`, and the outer step backtracks (the fit stays at its start with
/// `W_NO_FEASIBLE_EVAL`). Pins that the plan path agrees with the value path at that point.
#[test]
fn an_overflowing_growth_state_is_repelled_not_sent_to_finite_differences() {
    let mut data = String::from("ID,TIME,EVID,MDV,DV,AMT,RATE,CMT\n");
    for id in 1..=3 {
        data +=
            &format!("{id},0,1,1,.,1,0,1\n{id},100,0,0,1e150,0,0,1\n{id},200,0,0,1e300,0,0,1\n");
    }
    let (m, p) = from_text(GROWTH, &data);
    let o = FitOptions {
        outer_maxiter: 3,
        ..opts(EstimationMethod::FoceI)
    };
    let fit = crate::api::fit(&m, &p, &m.default_params, &o).expect("fit");
    assert!(
        !fit.warnings.iter().any(|w| w.contains(TOKEN)),
        "{:?}",
        fit.warnings
    );
    assert!(
        fit.warnings
            .iter()
            .any(|w| w.contains("W_NO_FEASIBLE_EVAL")),
        "every trial is repelled: {:?}",
        fit.warnings
    );
}
