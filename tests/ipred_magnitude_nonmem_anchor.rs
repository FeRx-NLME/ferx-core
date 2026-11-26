//! NONMEM 7.6.0 anchor for a residual magnitude that reads the prediction (#3121):
//! natural-scale naproxen arm means (`nonmem_anchor/ipred_magnitude.csv`, `DV·WPSE` of
//! `data/mbma_naproxen.csv`), residual SD `A·exp(B·IPRED)/sqrt(NIND)`, `$SIGMA 1 FIX`, FOCEI.
//! The SD is non-affine in `f` (`∂²SD/∂f² ≠ 0`), so no `combined` / `power` form expresses
//! it. Control streams `nonmem_anchor/ipred_magnitude{,_init,_foce_init,_nore,_post}.ctl`,
//! outputs under `nonmem_anchor/results/` (the `nonmem:7.6.0-anchor` build; the arm64
//! builds agree to 4e-9 on both fits). Realised differences, macOS arm64:
//!
//! | quantity | NONMEM | \|ferx − NONMEM\| |
//! |---|---:|---:|
//! | OBJ at the initial estimates (`_init`, `MAXEVAL=0`) | −143.04872703055159 | 3.5e-8 |
//! | FOCE (no `INTER`), `EMAX` log-normal, initial estimates (`_foce_init`) | −71.564006005561183 | 3.3e-8 |
//! | OBJ at NONMEM's final estimates | −171.18523552116284 | 5.6e-8 |
//! | no-RE OBJ at NONMEM's final estimates (`_nore`) | −36.321931791661434 | 2.1e-14 |
//! | POSTHOC η̂ / IPRED / IWRES / CWRES (`_post`) | table | 3.0e-7 / 2.2e-7 / 1.1e-6 / 1.9e-8 |
//! | fitted OBJ (slow) | −171.18523552116284 | 3.7e-9 (ferx lower) |
//! | fitted θ (slow), worst relative | `A` 4.47106, `B` −0.170729 | 3.0e-5 |

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

const DATA: &str = "nonmem_anchor/ipred_magnitude.csv";

/// `results/ipred_magnitude_init.ext`, the `-1000000000` row.
const NM_INIT_OBJ: f64 = -143.04872703055159;
/// `results/ipred_magnitude_foce_init.ext` (FOCE without `INTER`), the `-1000000000` row.
const NM_FOCE_INIT_OBJ: f64 = -71.564006005561183;
/// `results/ipred_magnitude.ext`, the `-1000000000` row.
const NM_FIT_OBJ: f64 = -171.18523552116284;
const NM_FIT_THETA: [f64; 9] = [
    5.1988064127263920,
    1.1354060587995800,
    0.64725298191215153,
    0.96431318073983263,
    0.84221518026902764,
    0.78991800220748587,
    -1.1507574976032338,
    4.4710616420372471,
    -0.17072918146769034,
];
const NM_FIT_OMEGA: [f64; 2] = [0.38514057087519554, 0.53922461802591781];
/// `results/ipred_magnitude_nore.ext`, the `-1000000000` row.
const NM_NORE_OBJ: f64 = -36.321931791661434;
const NM_NORE_THETA: [f64; 9] = [
    5.0427992957272432,
    0.80697257828715330,
    0.49717386932826974,
    1.2269992046185010,
    0.89439980392173624,
    0.80991746165150325,
    -2.7067626219174699,
    2.1322940654998619,
    0.22975805358365431,
];
const INIT_THETA: [f64; 9] = [5.19, 0.88, 0.70, 0.97, 1.10, 0.79, -1.18, 1.0, 0.1];
const INIT_OMEGA: [f64; 2] = [0.393129, 0.582169];

/// The OFV bound at every evaluation point. Realised errors (macOS arm64): 3.5e-8 at the
/// initial estimates, 5.6e-8 at NONMEM's optimum, 2.1e-14 without random effects — the
/// engines' summation order, not the model — so ~18× headroom, and a regression in how
/// `h(f)` enters `R` (thousandths of OFV) cannot pass.
const OFV_TOL: f64 = 1e-6;

/// The ferx twin of `ipred_magnitude.ctl`; `omega = None` is the no-RE variant.
fn model(theta: [f64; 9], omega: Option<[f64; 2]>) -> ferx_core::types::CompiledModel {
    parse_model_string(&model_src(theta, omega)).expect("model must parse")
}

fn model_src(theta: [f64; 9], omega: Option<[f64; 2]>) -> String {
    let t = theta;
    let (omegas, e0_eta, emax_eta) = match omega {
        Some([o1, o2]) => {
            (format!("omega ETA_E0 ~ {o1}\n  omega ETA_EMAX ~ {o2}"), " + ETA_E0", " + ETA_EMAX")
        }
        None => (String::new(), "", ""),
    };
    format!(
        "[parameters]
  theta TVE0({}, 1.0, 20.0)
  theta TVEMAX({}, -5.0, 20.0)
  theta TVET50({}, 0.01, 20.0)
  theta B_E0_FLARE({}, -10.0, 10.0)
  theta B_EMAX_FLARE({}, -10.0, 10.0)
  theta B_EMAX_TRT({}, -10.0, 10.0)
  theta B_LET50_TRT({}, -10.0, 10.0)
  theta A({}, 0.01, 100.0)
  theta B({}, -5.0, 5.0)
  {omegas}
  sigma EPS ~ 1.0 (variance) FIX

[covariates]
  TRT   categorical
  FLARE categorical
  NIND  continuous

[individual_parameters]
  E0   = TVE0 + B_E0_FLARE * FLARE{e0_eta}
  EMAX = TVEMAX + B_EMAX_FLARE * FLARE + B_EMAX_TRT * TRT{emax_eta}
  ET50 = TVET50 * exp(B_LET50_TRT * TRT)

[structural_model]
  y = E0 - EMAX * TIME / (TIME + ET50)

[error_model]
  DV ~ additive(EPS * A * exp(B * IPRED) / sqrt(NIND))

[fit_options]
  method = focei
",
        t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8]
    )
}

fn opts(method: EstimationMethod, outer_maxiter: usize) -> FitOptions {
    FitOptions {
        method,
        outer_maxiter,
        run_covariance_step: false,
        verbose: false,
        checkpoint: false,
        threads: Some(2),
        ..FitOptions::default()
    }
}

fn objective(m: &ferx_core::types::CompiledModel, method: EstimationMethod) -> f64 {
    let pop = read_nonmem_csv(Path::new(DATA), None, None).expect("data");
    fit(m, &pop, &m.default_params, &opts(method, 0)).expect("evaluation").ofv
}

fn assert_ofv(label: &str, m: &ferx_core::types::CompiledModel, want: f64) {
    assert_ofv_by(label, m, EstimationMethod::FoceI, want);
}

fn assert_ofv_by(
    label: &str, m: &ferx_core::types::CompiledModel, method: EstimationMethod, want: f64,
) {
    let ofv = objective(m, method);
    eprintln!("{label}: ferx {ofv:.12} vs NONMEM {want:.12}, diff {:e}", ofv - want);
    assert!((ofv - want).abs() < OFV_TOL, "{label}: ferx {ofv} vs NONMEM {want}");
}

/// The same model at the same point in both engines, before any optimizer moves.
#[test]
fn ipred_magnitude_objective_matches_nonmem_at_the_initial_estimates() {
    assert_ofv("initial estimates", &model(INIT_THETA, Some(INIT_OMEGA)), NM_INIT_OBJ);
}

/// FOCE without interaction (`METHOD=1`, no `INTER`): the residual variance is held at
/// the population prediction, so `h` must read `f(η = 0)`. `EMAX` is log-normal here: in
/// the main model `f` is linear in η, so the linearised `f(η̂) − Hη̂` *equals* `f(η = 0)`
/// and the check could not tell them apart. Without the magnitude in the f-dependence gate
/// (`CompiledModel::has_f_dependent_variance`) an additive model scores `h` at the
/// linearised point instead.
#[test]
fn ipred_magnitude_foce_objective_matches_nonmem() {
    let src = model_src(INIT_THETA, Some([INIT_OMEGA[0], 0.3]))
        .replace("B_EMAX_TRT * TRT + ETA_EMAX", "B_EMAX_TRT * TRT) * exp(ETA_EMAX)")
        .replace("EMAX = TVEMAX", "EMAX = (TVEMAX");
    let m = parse_model_string(&src).expect("model must parse");
    assert_ofv_by("FOCE, initial estimates", &m, EstimationMethod::Foce, NM_FOCE_INIT_OBJ);
}

/// A second point, with `B` of the opposite sign (the SD now falls with the prediction),
/// so an error in `h(f)` cannot cancel across the two.
#[test]
fn ipred_magnitude_objective_matches_nonmem_at_its_final_estimates() {
    assert_ofv("NONMEM optimum", &model(NM_FIT_THETA, Some(NM_FIT_OMEGA)), NM_FIT_OBJ);
}

/// No random effects (`IPRED = PRED`): the arm-level form an MBMA summary-data model
/// needs, where the SD moves with θ only through the prediction and the magnitude.
#[test]
fn ipred_magnitude_without_random_effects_matches_nonmem() {
    assert_ofv("no random effects", &model(NM_NORE_THETA, None), NM_NORE_OBJ);
}

/// The whitespace table `results/{name}`, header rows skipped, as numbers.
fn nm_table(name: &str) -> Vec<Vec<f64>> {
    let text = std::fs::read_to_string(format!("nonmem_anchor/results/{name}")).unwrap();
    text.lines()
        .skip(2)
        .map(|l| l.split_whitespace().map(|v| v.parse().unwrap()).collect())
        .collect()
}

/// Worst `|got − want|`, refusing a non-finite difference (a `NaN` would vanish in a fold).
fn worst(pairs: impl Iterator<Item = (f64, f64)>) -> f64 {
    pairs.fold(0.0, |w, (got, want)| {
        let d = (got - want).abs();
        assert!(d.is_finite(), "{got} vs {want}");
        w.max(d)
    })
}

/// POSTHOC at NONMEM's optimum (`ipred_magnitude_post.ctl`): η̂ (the EBE search reads
/// `h(f(η))`), IPRED, IWRES (`(DV − IPRED)/W`, the conditional SD) and CWRES (the
/// linearised `R̃` built on `R⁰` and its `f`-slope), row by row.
#[test]
fn ipred_magnitude_diagnostics_match_nonmem_posthoc() {
    let m = model(NM_FIT_THETA, Some(NM_FIT_OMEGA));
    let pop = read_nonmem_csv(Path::new(DATA), None, None).expect("data");
    let r = fit(&m, &pop, &m.default_params, &opts(EstimationMethod::FoceI, 0)).expect("fit");
    let ebes = nm_table("ipred_magnitude_post.tab");
    assert_eq!(ebes.len(), r.subjects.len());
    let d_eta = worst(r.subjects.iter().zip(&ebes).flat_map(|(s, row)| {
        assert_eq!(s.id, format!("{}", row[0] as u32), "subject order");
        (0..2).map(move |k| (s.eta[k], row[2 + k]))
    }));
    let rows = nm_table("ipred_magnitude_post.res");
    let col = |k: usize| rows.iter().map(move |r| r[k]);
    let ferx = |f: fn(&ferx_core::types::SubjectResult) -> &Vec<f64>| -> Vec<f64> {
        r.subjects.iter().flat_map(|s| f(s).clone()).collect()
    };
    let (ipred, iwres, cwres) = (ferx(|s| &s.ipred), ferx(|s| &s.iwres), ferx(|s| &s.cwres));
    assert_eq!(ipred.len(), rows.len());
    let d_ipred = worst(ipred.into_iter().zip(col(4)));
    let d_iwres = worst(iwres.into_iter().zip(col(5)));
    let d_cwres = worst(cwres.into_iter().zip(col(6)));
    eprintln!(
        "|Δη̂| {d_eta:.2e} |ΔIPRED| {d_ipred:.2e} |ΔIWRES| {d_iwres:.2e} |ΔCWRES| {d_cwres:.2e}"
    );
    // Realised (macOS arm64): η̂ 3.0e-7, IPRED 2.2e-7 (the two inner tolerances), IWRES
    // 1.1e-6, CWRES 1.9e-8; each bound ~10× its measurement.
    assert!(d_eta < 3e-6, "η̂ {d_eta:.2e}");
    assert!(d_ipred < 3e-6, "IPRED {d_ipred:.2e}");
    assert!(d_iwres < 1e-5, "IWRES {d_iwres:.2e}");
    assert!(d_cwres < 2e-7, "CWRES {d_cwres:.2e}");
}

/// The FOCEI fit lands where NONMEM's did.
#[test]
#[cfg_attr(not(feature = "slow-tests"), ignore = "slow: opt in with --features slow-tests")]
fn ipred_magnitude_fit_matches_nonmem() {
    let m = model(INIT_THETA, Some(INIT_OMEGA));
    let pop = read_nonmem_csv(Path::new(DATA), None, None).expect("data");
    let r = fit(&m, &pop, &m.default_params, &opts(EstimationMethod::FoceI, 500)).expect("fit");
    eprintln!("fit: OFV {} theta {:?}", r.ofv, r.theta);
    // The fit ran on the dual path, so this anchor sees the adjoint's `h(f)` too.
    let analytic = ferx_core::build_info::GradientMethodKind::Analytic.as_str();
    assert_eq!(r.gradient_method_inner, analytic);
    assert_eq!(r.gradient_method_outer, analytic);
    // Realised (macOS arm64): ferx 3.7e-9 *below* NONMEM's OBJ, worst θ 3.0e-5 relative
    // (TVEMAX). Bounds sit above the optimizer's stopping noise, not at the measurement.
    assert!((r.ofv - NM_FIT_OBJ).abs() < 1e-4, "OFV {} vs NONMEM {NM_FIT_OBJ}", r.ofv);
    for (k, (&got, &want)) in r.theta.iter().zip(&NM_FIT_THETA).enumerate() {
        assert!((got - want).abs() < 1e-3 * want.abs(), "θ{k}: {got} vs {want}");
    }
}
