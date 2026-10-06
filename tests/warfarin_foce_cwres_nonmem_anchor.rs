//! NONMEM anchor: sdtab CWRES of a FOCE (no interaction) fit (#1710).
//!
//! `nonmem_anchor/warfarin_foce_cwres.ctl` is `examples/warfarin.ferx` in NONMEM 7.6.0
//! (`ADVAN2 TRANS2`, `$EST METHOD=1` without `INTER`, `POSTHOC`, `FORMAT=s1PE23.16`), run
//! with `nm3`; the committed `results/warfarin_foce_cwres.{lst,ext,phi,tab}` are the
//! `nonmem:7.6.0-anchor` outputs (OFV −280.35950111582304; arm64 and arm64-nofma agree to
//! 2e-9). ferx is evaluated at NONMEM's estimates with `outer_maxiter = 0`, so both
//! engines score the same point and the residual comparison is not an optimizer race.
//!
//! The object under test is which marginal the post-fit residual pass scores. A FOCE fit
//! used to hand `compute_subject_results` the top-level `interaction` flag, `true` by
//! default, so its CWRES were the FOCEI ones at FOCE estimates. Measured against this
//! table on macOS arm64 before the fix: worst |ferx − NONMEM| CWRES 0.679 (median 5.0e-2).
//!
//! Two input classes, two gates (see the #1710 plan, §4). Arm (b) builds
//! `FitOptions { method: Foce, ..default() }` through the API, so `interaction` arrives
//! `true` and only the post-fit resolution in `fit_inner` can clear it — reverting that
//! reddens it. Arm (a) runs the model file's own `[fit_options]`, which the parser now
//! resolves too; it survives either single revert by design and pins the file path end
//! to end.
//!
//! Engine: analytic `Dual2` (warfarin one-compartment oral is in scope; asserted below by
//! the absence of the finite-difference fallback warning).

use ferx_core::{fit, prepare_run, EstimationMethod, FitOptions, FitResult, OmegaMatrix};

/// NONMEM's final `.ext` row (`-1000000000`), anchor build. Column order there is THETA,
/// SIGMA(1,1), then the OMEGA lower triangle.
const NM_THETA: [f64; 3] = [
    1.3296190773543662E-01,
    7.7305172240926252E+00,
    7.2537506677117802E-01,
];
const NM_SIGMA_VAR: f64 = 1.1548425869877831E-04;
const NM_OMEGA: [f64; 3] = [
    2.8596520575084699E-02,
    9.5781911176013226E-03,
    3.4881225592887283E-01,
];
const NM_OFV: f64 = -280.35950111582304;

/// Bounds, from the realised errors (macOS arm64 and Linux aarch64 agree in every printed digit):
/// - CWRES: worst |Δ| 6.057e-3 under the fix, 0.6787 with the post-fit resolution
///   reverted. 2e-2 is 3.3× above the first and 34× below the second. The residual is
///   NONMEM's POSTHOC η̂ against ferx's, each from its own inner optimizer, and CWRES
///   carries the η̂-dependent linearisation, which is where the 6e-3 lives.
/// - OFV: |Δ| 4.454e-3 (ferx −280.3639550 vs NONMEM −280.3595011), with 2.2× headroom.
///   The leak does not move it — the estimation stage always set its own flag — so this
///   bound pins the point, not the fix.
/// - PRED: θ-only population predictions, worst relative |Δ| 3.96e-16. A miss here means
///   the parameter decoding or the row order is wrong, not the residual pass.
const CWRES_TOL: f64 = 2e-2;
const OFV_TOL: f64 = 1e-2;
const PRED_TOL: f64 = 1e-12;

/// `(ID, CWRES, PRED)` for every observation row (`EVID = 0`), in file order.
fn nonmem_obs() -> Vec<(String, f64, f64)> {
    let text = std::fs::read_to_string("nonmem_anchor/results/warfarin_foce_cwres.tab")
        .expect("NONMEM table");
    text.lines()
        .skip(2)
        .map(|l| {
            let c: Vec<f64> = l
                .split_whitespace()
                .map(|x| x.parse().expect("number"))
                .collect();
            assert_eq!(c.len(), 5, "ID TIME EVID CWRES PRED");
            c
        })
        .filter(|c| c[2] == 0.0)
        .map(|c| (format!("{}", c[0] as i64), c[3], c[4]))
        .collect()
}

fn run_at_nonmem_point(opts: FitOptions) -> FitResult {
    let prep = prepare_run("examples/warfarin.ferx", Some("data/warfarin.csv"))
        .expect("warfarin must prepare");
    let mut init = prep.init_params.clone();
    init.theta = NM_THETA.to_vec();
    init.omega = OmegaMatrix::from_diagonal(&NM_OMEGA, init.omega.eta_names.clone());
    // ferx's proportional sigma is an SD; NONMEM's SIGMA is the variance.
    init.sigma.values = vec![NM_SIGMA_VAR.sqrt()];
    let opts = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        ..opts
    };
    fit(&prep.parsed.model, &prep.population, &init, &opts).expect("fit must run")
}

fn assert_matches_nonmem(label: &str, r: &FitResult) {
    assert!(
        !r.warnings
            .iter()
            .any(|w| w.contains("finite-difference inner gradients")),
        "{label}: fixture left the analytic path: {:?}",
        r.warnings
    );
    assert_eq!(r.method, EstimationMethod::Foce, "{label}: method");
    let want = nonmem_obs();
    let got: Vec<(String, f64, f64)> = r
        .subjects
        .iter()
        .flat_map(|s| {
            s.cwres
                .iter()
                .zip(&s.pred)
                .map(move |(&c, &p)| (s.id.clone(), c, p))
        })
        .collect();
    assert_eq!(got.len(), want.len(), "{label}: observation count");
    let (mut worst_c, mut worst_p) = (0.0_f64, 0.0_f64);
    for (k, ((gid, gc, gp), (wid, wc, wp))) in got.iter().zip(&want).enumerate() {
        assert_eq!(gid, wid, "{label}: row {k} subject order");
        // f64::max discards NaN, so finiteness is asserted before the fold.
        assert!(
            gc.is_finite() && gp.is_finite(),
            "{label}: row {k} ferx CWRES {gc} / PRED {gp} not finite"
        );
        worst_c = worst_c.max((gc - wc).abs());
        worst_p = worst_p.max((gp - wp).abs() / wp.abs().max(1e-12));
    }
    let d_ofv = (r.ofv - NM_OFV).abs();
    eprintln!(
        "MEASURE {label}: worst |dCWRES| {worst_c:.3e}, worst rel dPRED {worst_p:.3e}, \
         |dOFV| {d_ofv:.3e} (ferx {:.10}), interaction {}",
        r.ofv, r.interaction
    );
    assert!(
        worst_p < PRED_TOL,
        "{label}: PRED off NONMEM by {worst_p:e} — the point or row order is wrong"
    );
    assert!(
        r.ofv.is_finite() && d_ofv < OFV_TOL,
        "{label}: OFV {} vs NONMEM {NM_OFV}",
        r.ofv
    );
    assert!(
        worst_c < CWRES_TOL,
        "{label}: CWRES off NONMEM FOCE by {worst_c:e} (> {CWRES_TOL}): the residual pass is \
         not scoring the FOCE marginal"
    );
    assert!(
        !r.interaction,
        "{label}: a FOCE fit must record interaction = false"
    );
}

/// Arm (b): API-built FOCE options, `interaction` left at the `true` default.
#[test]
fn foce_cwres_matches_nonmem_api_options() {
    let opts = FitOptions {
        method: EstimationMethod::Foce,
        ..FitOptions::default()
    };
    assert!(
        opts.interaction,
        "fixture premise: API default interaction = true"
    );
    assert_matches_nonmem("api", &run_at_nonmem_point(opts));
}

/// Arm (a): the model file's own `[fit_options]` (`method = foce`).
#[test]
fn foce_cwres_matches_nonmem_file_options() {
    let prep = prepare_run("examples/warfarin.ferx", Some("data/warfarin.csv"))
        .expect("warfarin must prepare");
    assert_matches_nonmem("file", &run_at_nonmem_point(prep.parsed.fit_options));
}
