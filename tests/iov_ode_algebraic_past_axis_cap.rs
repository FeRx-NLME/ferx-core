//! #2031 end to end: an IOV model on the ODE engine or the compartment-free readout keeps
//! the exact analytic gradient, inner and outer, past 24 `n_theta + n_eta + n_kappa` axes.
//!
//! Before #2093 both engines carried a 24-axis clause of their own in the forward
//! sensitivity provider (`ode_iov_supported`, `algebraic::supported_iov`), on top of the
//! per-occasion program source `iov_combined_derivs_dyn` that #1907 chunked over θ. One
//! axis past it, an ODE IOV fit ran its EBE search on finite differences (28–38k FD calls
//! in two outer iterations on `warfarin_iov`) and `optimizer = auto` resolved to BOBYQA with
//! `gradient_method_outer` reading `N/A`; a compartment-free IOV fit ran both loops on
//! finite differences under `analytic (Dual2)` labels. A FIX θ did not help: `n_theta`
//! counts it. #2093 retired that provider, and the adjoint recorders that replaced it
//! seed θ on a reverse tape and only `[η, κ]` forward, so no θ-dependent width remains.
//! These tests pin that the axis count no longer matters on either engine.
//!
//! The fixtures are the issue's: each model is padded with `n` θ that have no effect —
//! every pad enters one individual parameter as `+ 0.0 * PADi` — so the likelihood is the
//! same at every pad count. Each pair straddles the old cap (24 vs 25 axes), estimated and
//! FIX, and must agree on route and objective.
//!
//! No NONMEM anchor: no new numeric object is created — the 25-axis twin is the 24-axis
//! model with one more inert θ, and the objective comparison below is that oracle.
//!
//! Tier 2: the route is read from fits with `maxiter = 1` (one outer budget, so the outer
//! gradient is computed and a fallback would warn) and the objective from evaluation-only
//! fits (`maxiter = 0`).

use std::path::Path;

/// Compartment-free IOV data from the issue: 8 subjects × 2 occasions × 4 observations.
const ALGEBRAIC_DATA: &str = "\
ID,TIME,DV,EVID,AMT,CMT,MDV,OCC
1,1,12.0849,0,.,1,0,1
1,2,10.2993,0,.,1,0,1
1,4,8.8606,0,.,1,0,1
1,8,8.3164,0,.,1,0,1
1,11,9.8192,0,.,1,0,2
1,12,10.6304,0,.,1,0,2
1,14,10.5078,0,.,1,0,2
1,18,10.5978,0,.,1,0,2
2,1,5.5817,0,.,1,0,1
2,2,4.2526,0,.,1,0,1
2,4,3.6946,0,.,1,0,1
2,8,2.8419,0,.,1,0,1
2,11,-0.5909,0,.,1,0,2
2,12,-0.6863,0,.,1,0,2
2,14,-0.9173,0,.,1,0,2
2,18,-1.1169,0,.,1,0,2
3,1,11.3992,0,.,1,0,1
3,2,10.8766,0,.,1,0,1
3,4,9.7450,0,.,1,0,1
3,8,8.7293,0,.,1,0,1
3,11,9.2478,0,.,1,0,2
3,12,9.0279,0,.,1,0,2
3,14,9.2279,0,.,1,0,2
3,18,8.8812,0,.,1,0,2
4,1,12.0010,0,.,1,0,1
4,2,11.2543,0,.,1,0,1
4,4,9.3634,0,.,1,0,1
4,8,8.9268,0,.,1,0,1
4,11,10.1844,0,.,1,0,2
4,12,9.1643,0,.,1,0,2
4,14,9.4072,0,.,1,0,2
4,18,9.2523,0,.,1,0,2
5,1,8.0979,0,.,1,0,1
5,2,6.8496,0,.,1,0,1
5,4,6.4833,0,.,1,0,1
5,8,5.3018,0,.,1,0,1
5,11,4.8261,0,.,1,0,2
5,12,4.8009,0,.,1,0,2
5,14,4.0968,0,.,1,0,2
5,18,3.9619,0,.,1,0,2
6,1,7.6151,0,.,1,0,1
6,2,6.4948,0,.,1,0,1
6,4,4.9818,0,.,1,0,1
6,8,4.4421,0,.,1,0,1
6,11,3.4438,0,.,1,0,2
6,12,3.1916,0,.,1,0,2
6,14,3.1787,0,.,1,0,2
6,18,3.2352,0,.,1,0,2
7,1,3.7837,0,.,1,0,1
7,2,2.9795,0,.,1,0,1
7,4,1.9927,0,.,1,0,1
7,8,1.2329,0,.,1,0,1
7,11,-0.9294,0,.,1,0,2
7,12,-1.0659,0,.,1,0,2
7,14,-1.1358,0,.,1,0,2
7,18,-1.3142,0,.,1,0,2
8,1,10.0492,0,.,1,0,1
8,2,9.5782,0,.,1,0,1
8,4,8.3832,0,.,1,0,1
8,8,8.6782,0,.,1,0,1
8,11,5.8445,0,.,1,0,2
8,12,6.5344,0,.,1,0,2
8,14,6.1905,0,.,1,0,2
8,18,6.2090,0,.,1,0,2
";

/// `n` pad θ lines, estimated or FIX, and the `* (1.0 + 0.0 * PAD0 + …)` factor that reads
/// every one of them.
fn pads(n: usize, fix: bool) -> (String, String) {
    let fix = if fix { " FIX" } else { "" };
    let thetas = (0..n)
        .map(|i| format!("  theta PAD{i}(0.5, 0.01, 10.0){fix}\n"))
        .collect();
    let terms: String = (0..n).map(|i| format!(" + 0.0 * PAD{i}")).collect();
    (thetas, format!(" * (1.0{terms})"))
}

/// The issue's ODE fixture: 3 θ + 3 η + 1 κ, so 17 pads is 24 axes and 18 is 25.
fn ode_model(n: usize, fix: bool) -> String {
    let (thetas, factor) = pads(n, fix);
    format!(
        "[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
{thetas}  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.1 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL){factor}
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot / V - (CL/V) * central

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = focei
  iov_column = OCC
  covariance = false
  checkpoint = false
"
    )
}

/// The issue's compartment-free fixture: 3 θ + 1 η + 1 κ, so 19 pads is 24 axes and 20 is 25.
fn algebraic_model(n: usize, fix: bool) -> String {
    let (thetas, factor) = pads(n, fix);
    format!(
        "[parameters]
  theta TVE0(10.0, 0.1, 100.0)
  theta TVEMAX(6.0, 0.1, 100.0)
  theta TVET50(2.0, 0.01, 100.0)
{thetas}  omega ETA_E0 ~ 0.09
  kappa KAPPA_EMAX ~ 0.04
  sigma PROP ~ 0.04 (sd)

[individual_parameters]
  E0   = TVE0 * exp(ETA_E0){factor}
  EMAX = TVEMAX * exp(KAPPA_EMAX)
  ET50 = TVET50

[structural_model]
  y = E0 - EMAX * TIME / (ET50 + TIME)

[error_model]
  DV ~ proportional(PROP)

[fit_options]
  iov_column = OCC
  covariance = false
  checkpoint = false
"
    )
}

/// Fit `model` with an outer budget of `maxiter`: `1` runs outer gradient evaluations,
/// `0` is the evaluation-only path (inner loop at the start values, no outer gradient).
fn run(dir: &Path, name: &str, model: &str, maxiter: usize, data: &Path) -> ferx_core::FitResult {
    let model_path = dir.join(format!("{name}_{maxiter}.ferx"));
    std::fs::write(&model_path, format!("{model}  maxiter    = {maxiter}\n")).unwrap();
    ferx_core::run_model_with_data(model_path.to_str().unwrap(), Some(data.to_str().unwrap()))
        .map(|(fit, _)| fit)
        .unwrap_or_else(|e| panic!("{name}: fit failed: {e}"))
}

/// Fit the 24-axis twin and the estimated and FIX 25-axis twins. With an outer budget all
/// three must take the analytic route on both loops, resolve `auto` to L-BFGS and carry no
/// finite-difference warning; evaluated at the start values they must agree subject by
/// subject.
fn assert_straddle(engine: &str, model: fn(usize, bool) -> String, n_at_cap: usize, data: &Path) {
    let dir = tempfile::tempdir().unwrap();
    let twins = [
        ("24 axes", model(n_at_cap, false)),
        ("25 axes, pads estimated", model(n_at_cap + 1, false)),
        ("25 axes, pads FIX", model(n_at_cap + 1, true)),
    ];
    for (i, (label, text)) in twins.iter().enumerate() {
        let r = run(dir.path(), &format!("twin{i}"), text, 1, data);
        assert!(r.ofv.is_finite(), "{engine}, {label}: OFV {}", r.ofv);
        // Fixture: every pad is read by the program, so the program really is that wide.
        assert!(
            !r.warnings.iter().any(|w| w.contains("but not referenced")),
            "{engine}, {label}: a pad is not in the program: {:?}",
            r.warnings
        );
        assert_eq!(
            r.gradient_method_inner, "analytic (Dual2)",
            "{engine}, {label}: inner route"
        );
        assert_eq!(
            r.gradient_method_outer, "analytic (Dual2)",
            "{engine}, {label}: outer route"
        );
        assert_eq!(
            r.optimizer, "auto (nlopt_lbfgs)",
            "{engine}, {label}: `auto` takes L-BFGS on an analytic outer gradient"
        );
        let fd: Vec<&String> = r
            .warnings
            .iter()
            .filter(|w| {
                w.contains("finite-difference")
                    || w.contains("finite difference")
                    || w.contains("could not be given the exact analytic outer gradient")
            })
            .collect();
        assert!(
            fd.is_empty(),
            "{engine}, {label}: a subject fell back to finite differences: {fd:?}"
        );
    }

    // The objective oracle runs at the start values: a run with an outer budget is not
    // comparable across twins, because the budget `maxiter = 1` buys scales with the
    // parameter count (26 vs 27 objective evaluations here).
    let evals: Vec<ferx_core::FitResult> = twins
        .iter()
        .enumerate()
        .map(|(i, (_, text))| run(dir.path(), &format!("twin{i}"), text, 0, data))
        .collect();
    eprintln!(
        "#2031 {engine}: start-value OFV 24 axes {:.12}, 25 estimated {:.12}, 25 FIX {:.12}",
        evals[0].ofv, evals[1].ofv, evals[2].ofv
    );
    let at_cap = &evals[0];
    assert!(at_cap.ofv.is_finite(), "{engine}: OFV {}", at_cap.ofv);
    for ((label, _), r) in twins.iter().zip(&evals).skip(1) {
        assert_eq!(at_cap.subjects.len(), r.subjects.len());
        for (a, b) in at_cap.subjects.iter().zip(&r.subjects) {
            assert_eq!(a.id, b.id);
            assert_eq!(
                a.ofv_contribution, b.ofv_contribution,
                "{engine}, {label}, subject {}: per-subject OFV vs 24 axes",
                a.id
            );
            assert_eq!(
                a.eta, b.eta,
                "{engine}, {label}, subject {}: η̂ vs 24 axes",
                a.id
            );
        }
        assert_eq!(at_cap.ofv, r.ofv, "{engine}, {label}: OFV vs 24 axes");
    }
}

#[test]
fn ode_iov_keeps_the_analytic_gradient_past_24_axes() {
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/warfarin_iov.csv");
    assert_straddle("ODE", ode_model, 17, &data);
}

#[test]
fn compartment_free_iov_keeps_the_analytic_gradient_past_24_axes() {
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("algebraic_iov.csv");
    std::fs::write(&data, ALGEBRAIC_DATA).unwrap();
    assert_straddle("compartment-free", algebraic_model, 19, &data);
}
