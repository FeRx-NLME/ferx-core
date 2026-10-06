//! NONMEM anchor: the residual variance a non-interaction FOCE EBE search scores (#1722).
//!
//! NONMEM `METHOD=1` without `INTER` finds each subject's EBEs with the residual variance
//! held at the population prediction `f(η = 0)` — the same `R⁰` its Sheiner–Beal marginal
//! uses. ferx scored the variance at the conditional `f(η)` during the search, so it
//! linearised the marginal around a different mode: every subject's η̂ moved, and the OFV
//! came out below NONMEM's at NONMEM's own estimates.
//!
//! Two committed NONMEM 7.6.0 runs (`nm3`, `nonmem:7.6.0-anchor`; arm64 and arm64-nofma
//! agree to ≤ 2e-6), both on `data/warfarin_iov.csv` with proportional error at σ ≈ 20%,
//! each evaluated by ferx at NONMEM's estimates with `outer_maxiter = 0`:
//!
//! - **non-IOV** — `nonmem_anchor/foce_ebe_freeze_noiov.ctl` /
//!   `foce_ebe_freeze_noiov_fit.ferx`, OFV 390.91305651859278. Before the fix ferx gave
//!   390.173257 (Δ −0.740).
//! - **IOV** — `nonmem_anchor/foce_ebe_freeze_iov.ctl` / `examples/warfarin_iov.ferx`,
//!   OFV 205.09053792965955. Before the fix ferx gave 203.806423 (Δ −1.284). Not
//!   degenerate on the incoming side of a dose: occasion 2's dose lands on residual drug.
//!
//! An independent Python reimplementation (piecewise-exact one-compartment oral, scipy
//! BFGS mode, central-FD `H`, Sheiner–Beal marginal with `R⁰` at `f(b = 0)`) reproduces
//! NONMEM with the variance frozen (390.913057 / 205.090543) and the old ferx without it
//! (390.173258 / 203.806424), so the two conventions are the whole difference.
//!
//! `warfarin.ferx` (σ ≈ 1%, `warfarin_foce_cwres_nonmem_anchor.rs`) cannot see this: at
//! that σ both conventions give the same mode to the printed digits.
//!
//! Engine: analytic `Dual2` inner gradients on both (one-compartment oral, and its IOV
//! twin, are in scope) — asserted by the absence of the finite-difference fallback warning.

use ferx_core::{fit, prepare_run, EstimationMethod, FitOptions, FitResult, OmegaMatrix};

/// Bounds, from the realised errors after the fix (macOS arm64; see the README row for
/// Linux). Each is ~10× above what the fix leaves and far below what the old conditional
/// search produced at the same point:
/// - OFV: |Δ| 1.4e-8 (non-IOV) / 2.3e-7 (IOV); before the fix 0.740 / 1.284.
/// - η̂ vs `.phi`: 6.2e-7 / 4.9e-6; before the fix up to 0.73 (IOV subject 10's ETA_KA:
///   ferx −0.542, NONMEM −1.272).
/// - κ̂ vs `.phi`: 4.9e-7.
/// - per-subject OBJ vs `.phi`: 2.4e-5 / 7.3e-5; before the fix 0.03–0.81, one-signed.
/// - CWRES vs the NONMEM table (non-IOV): 3.1e-6.
///
/// NONMEM's estimates enter at the six significant digits `.ext` prints, which is why the
/// OFV agrees to 1e-8 rather than to the last digit; the point is still NONMEM's optimum to
/// first order, so the objective is flat there.
const OFV_TOL: f64 = 2e-6;
const ETA_TOL: f64 = 5e-5;
const KAPPA_TOL: f64 = 5e-6;
const OBJ_TOL: f64 = 5e-4;
const CWRES_TOL: f64 = 3e-5;

/// One NONMEM `.phi` row: `(ID, [ETA…], OBJ)`.
fn read_phi(path: &str, n_eta: usize) -> Vec<(String, Vec<f64>, f64)> {
    let text = std::fs::read_to_string(path).expect("NONMEM .phi");
    text.lines()
        .skip(2)
        .map(|l| {
            let c: Vec<&str> = l.split_whitespace().collect();
            let etas: Vec<f64> = c[2..2 + n_eta]
                .iter()
                .map(|x| x.parse().expect("eta"))
                .collect();
            let obj: f64 = c.last().unwrap().parse().expect("obj");
            (c[1].to_string(), etas, obj)
        })
        .collect()
}

fn run_at(
    model_file: &str,
    theta: &[f64],
    omega: &[f64],
    kappa: Option<f64>,
    sigma_var: f64,
) -> FitResult {
    let prep = prepare_run(model_file, Some("data/warfarin_iov.csv")).expect("prepare");
    let mut init = prep.init_params.clone();
    init.theta = theta.to_vec();
    init.omega = OmegaMatrix::from_diagonal(omega, init.omega.eta_names.clone());
    if let Some(k) = kappa {
        let iov = init.omega_iov.as_ref().expect("IOV model");
        init.omega_iov = Some(OmegaMatrix::from_diagonal(&[k], iov.eta_names.clone()));
    }
    // ferx's proportional sigma is an SD; NONMEM's SIGMA is the variance.
    init.sigma.values = vec![sigma_var.sqrt()];
    let opts = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        ..prep.parsed.fit_options.clone()
    };
    assert_eq!(opts.method, EstimationMethod::Foce, "fixture is a FOCE fit");
    fit(&prep.parsed.model, &prep.population, &init, &opts).expect("fit must run")
}

fn assert_analytic(label: &str, r: &FitResult) {
    assert!(
        !r.warnings
            .iter()
            .any(|w| w.contains("finite-difference inner gradients")),
        "{label}: fixture left the analytic inner path: {:?}",
        r.warnings
    );
    assert!(!r.interaction, "{label}: FOCE records interaction = false");
}

/// Worst |a − b| over pairs, asserting finiteness first (`f64::max` discards NaN).
fn worst_abs(label: &str, pairs: impl Iterator<Item = (f64, f64)>) -> f64 {
    let mut worst = 0.0_f64;
    for (k, (a, b)) in pairs.enumerate() {
        assert!(
            a.is_finite() && b.is_finite(),
            "{label}: pair {k} not finite ({a}, {b})"
        );
        worst = worst.max((a - b).abs());
    }
    worst
}

// ── non-IOV ─────────────────────────────────────────────────────────────────

const NOIOV_THETA: [f64; 3] = [1.43657E-01, 8.83827E+00, 1.22791E+00];
const NOIOV_OMEGA: [f64; 3] = [8.31907E-02, 8.12141E-03, 2.33209E-02];
const NOIOV_SIGMA_VAR: f64 = 4.50339E-02;
const NOIOV_OFV: f64 = 390.91305651859278;

#[test]
fn foce_ebe_variance_matches_nonmem_noiov() {
    let r = run_at(
        "nonmem_anchor/foce_ebe_freeze_noiov_fit.ferx",
        &NOIOV_THETA,
        &NOIOV_OMEGA,
        None,
        NOIOV_SIGMA_VAR,
    );
    assert_analytic("noiov", &r);
    let phi = read_phi("nonmem_anchor/results/foce_ebe_freeze_noiov.phi", 3);
    assert_eq!(phi.len(), r.subjects.len(), "subject count");
    let eta = worst_abs(
        "noiov eta",
        r.subjects.iter().zip(&phi).flat_map(|(s, (id, e, _))| {
            assert_eq!(&s.id, id, "subject order");
            (0..3).map(move |k| (s.eta[k], e[k]))
        }),
    );
    let obj = worst_abs(
        "noiov per-subject OBJ",
        r.subjects
            .iter()
            .zip(&phi)
            .map(|(s, (_, _, o))| (s.ofv_contribution, *o)),
    );
    // CWRES: NONMEM's table carries every record; observation rows are the MDV = 0 rows
    // of the data file, in file order.
    let data = std::fs::read_to_string("data/warfarin_iov.csv").expect("data");
    let mdv: Vec<bool> = data
        .lines()
        .skip(1)
        .map(|l| l.split(',').nth(7).unwrap().trim() == "1")
        .collect();
    let tab =
        std::fs::read_to_string("nonmem_anchor/results/foce_ebe_freeze_noiov.tab").expect("table");
    let nm_cwres: Vec<f64> = tab
        .lines()
        .skip(2)
        .zip(&mdv)
        .filter(|(_, &m)| !m)
        .map(|(l, _)| l.split_whitespace().nth(2).unwrap().parse().unwrap())
        .collect();
    let got_cwres: Vec<f64> = r.subjects.iter().flat_map(|s| s.cwres.clone()).collect();
    assert_eq!(got_cwres.len(), nm_cwres.len(), "observation count");
    let cwres = worst_abs("noiov CWRES", got_cwres.into_iter().zip(nm_cwres));
    let d_ofv = r.ofv - NOIOV_OFV;
    eprintln!(
        "MEASURE noiov: ferx OFV {:.9}, dOFV {d_ofv:+.3e}, worst |d eta| {eta:.3e}, \
         worst |d OBJ_i| {obj:.3e}, worst |d CWRES| {cwres:.3e}",
        r.ofv
    );
    assert!(
        r.ofv.is_finite() && d_ofv.abs() < OFV_TOL,
        "non-IOV FOCE OFV {} vs NONMEM {NOIOV_OFV}: the EBE search is not holding the \
         residual variance at f(eta = 0)",
        r.ofv
    );
    assert!(eta < ETA_TOL, "non-IOV eta-hat off NONMEM .phi by {eta:e}");
    assert!(
        obj < OBJ_TOL,
        "non-IOV per-subject OBJ off NONMEM by {obj:e}"
    );
    assert!(cwres < CWRES_TOL, "non-IOV CWRES off NONMEM by {cwres:e}");
}

// ── IOV ─────────────────────────────────────────────────────────────────────

const IOV_THETA: [f64; 3] = [3.15410E-01, 8.38248E+00, 2.64431E+00];
const IOV_OMEGA: [f64; 3] = [4.46654E-01, 1.25169E-02, 1.07769E+00];
const IOV_KAPPA: f64 = 4.25800E-02;
const IOV_SIGMA_VAR: f64 = 3.70648E-02;
const IOV_OFV: f64 = 205.09053792965955;

#[test]
fn foce_ebe_variance_matches_nonmem_iov() {
    let r = run_at(
        "examples/warfarin_iov.ferx",
        &IOV_THETA,
        &IOV_OMEGA,
        Some(IOV_KAPPA),
        IOV_SIGMA_VAR,
    );
    assert_analytic("iov", &r);
    // `.phi` ETA(1..3) are the BSV etas, ETA(4..5) the occasion-1/2 kappas.
    let phi = read_phi("nonmem_anchor/results/foce_ebe_freeze_iov.phi", 5);
    assert_eq!(phi.len(), r.subjects.len(), "subject count");
    assert_eq!(r.ebe_kappas.len(), r.subjects.len(), "kappa rows");
    let eta = worst_abs(
        "iov eta",
        r.subjects.iter().zip(&phi).flat_map(|(s, (id, e, _))| {
            assert_eq!(&s.id, id, "subject order");
            (0..3).map(move |k| (s.eta[k], e[k]))
        }),
    );
    let kappa = worst_abs(
        "iov kappa",
        r.ebe_kappas.iter().zip(&phi).flat_map(|(ks, (_, e, _))| {
            assert_eq!(ks.len(), 2, "two occasions per subject");
            (0..2).map(move |o| (ks[o][0], e[3 + o]))
        }),
    );
    let obj = worst_abs(
        "iov per-subject OBJ",
        r.subjects
            .iter()
            .zip(&phi)
            .map(|(s, (_, _, o))| (s.ofv_contribution, *o)),
    );
    let d_ofv = r.ofv - IOV_OFV;
    eprintln!(
        "MEASURE iov: ferx OFV {:.9}, dOFV {d_ofv:+.3e}, worst |d eta| {eta:.3e}, \
         worst |d kappa| {kappa:.3e}, worst |d OBJ_i| {obj:.3e}",
        r.ofv
    );
    assert!(
        r.ofv.is_finite() && d_ofv.abs() < OFV_TOL,
        "IOV FOCE OFV {} vs NONMEM {IOV_OFV}: the IOV EBE search is not holding the \
         residual variance at f(eta = 0, kappa = 0)",
        r.ofv
    );
    assert!(eta < ETA_TOL, "IOV eta-hat off NONMEM .phi by {eta:e}");
    assert!(
        kappa < KAPPA_TOL,
        "IOV kappa-hat off NONMEM .phi by {kappa:e}"
    );
    assert!(obj < OBJ_TOL, "IOV per-subject OBJ off NONMEM by {obj:e}");
}
