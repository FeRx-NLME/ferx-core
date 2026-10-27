//! CWRES under inter-occasion variability (#1948), against a hand-built recipe.
//!
//! One subject, 1-cpt IV, two occasions with a dose each, IIV and IOV on CL. The closed form
//! is piecewise in the elimination rate: occasion 1 decays at `k₁ = (CL/V)·e^{η+κ₁}` up to
//! its last record, and from there occasion 1's residual and occasion 2's dose decay at `k₂`.
//! So the
//! κ₁ column of `H` is **dense** — it is live on occasion 2's rows through the carryover —
//! and every column of `H_b = [∂f/∂η │ ∂f/∂κ₁ │ ∂f/∂κ₂]` is written down in closed form
//! here, outside the engine.
//!
//! The recipe is NONMEM's (Hooker 2007 over all three ETAs): `f0 = f(b̂) − H_b·b̂`,
//! `R̃ = H_b·Σ_b·H_bᵀ + R`, `Σ_b = diag(ω², ω²_IOV, ω²_IOV)`, `CWRES = R̃^{-1/2}(y − f0)` with
//! the symmetric inverse square root; `R` at IPRED (FOCEI) or at `f(η = 0, κ = 0)` (FOCE).
//!
//! The fix has two halves, and each has a test that only it can fail:
//! - `f0_subtracts_the_kappa_part_of_the_linearization` dies when `H_κ·κ̂` is dropped from
//!   `f0`. `Ω_IOV = 1e-12` there, so the `R̃` half moves CWRES by ~1e-12 and cannot.
//! - `rtilde_carries_the_iov_variance` dies when `H_κ·Ω_IOV·H_κᵀ` is dropped from `R̃`.
//!   `κ̂ = 0` there, so `H_κ·κ̂ = 0` and the `f0` half is a no-op.
//!
//! `matches_the_stacked_recipe` checks the whole object, both halves live, FOCEI and FOCE.
//! `a_kappa_shape_mismatch_is_refused` pins what the post-fit pass does when it cannot build
//! the linearization at all.

use super::postfit::compute_subject_results;
use crate::parser::model_parser::parse_model_string;
use crate::types::{CompiledModel, ModelParameters, OmegaMatrix, Population};
use nalgebra::{DMatrix, DVector};

const MODEL: &str = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.05 FIX
  kappa KAPPA_CL ~ 0.04 FIX
  sigma EPS ~ 0.04 FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

const DOSE: f64 = 100.0;
/// Occasion 2's dose time.
const T2: f64 = 4.0;
/// Where the rate switches from `k₁` to `k₂`: occasion 1's last record. The interval
/// `(2, 4]` belongs to the record that ends it — occasion 2's dose — which is NONMEM's
/// convention (a record's `$PK` values govern the advance *to* it), and the one the
/// IPRED premise in `run` pins. About 80% of occasion 1's dose is still present at the
/// switch, so the carryover is live.
const T_SWITCH: f64 = 2.0;
/// `(time, occasion, DV)`; DVs are chosen off the curve so every residual is O(1).
const OBS: [(f64, usize, f64); 4] = [(1.0, 1, 7.1), (2.0, 1, 6.4), (5.0, 2, 9.8), (6.0, 2, 7.2)];

fn population() -> Population {
    use std::io::Write;
    let mut csv = String::from("ID,TIME,DV,AMT,EVID,CMT,OCC\n");
    csv.push_str(&format!("1,0,0,{DOSE},1,1,1\n"));
    for &(t, occ, dv) in &OBS[..2] {
        csv.push_str(&format!("1,{t},{dv},0,0,1,{occ}\n"));
    }
    csv.push_str(&format!("1,{T2},0,{DOSE},1,1,2\n"));
    for &(t, occ, dv) in &OBS[2..] {
        csv.push_str(&format!("1,{t},{dv},0,0,1,{occ}\n"));
    }
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(csv.as_bytes()).unwrap();
    crate::read_nonmem_csv(f.path(), None, Some("OCC")).unwrap()
}

/// `f(b)` and `H_b = ∂f/∂b` in closed form, `b = [η, κ₁, κ₂]`.
fn closed_form(theta: &[f64], b: [f64; 3]) -> (Vec<f64>, DMatrix<f64>) {
    let (cl, v) = (theta[0], theta[1]);
    let k1 = cl / v * (b[0] + b[1]).exp();
    let k2 = cl / v * (b[0] + b[2]).exp();
    let mut f = vec![0.0; OBS.len()];
    let mut h = DMatrix::zeros(OBS.len(), 3);
    for (j, &(t, occ, _)) in OBS.iter().enumerate() {
        if occ == 1 {
            f[j] = DOSE / v * (-k1 * t).exp();
            h[(j, 1)] = -k1 * t * f[j];
        } else {
            // Occasion 1's residual at the switch, then both amounts decay at k₂.
            let a_s = DOSE * (-k1 * T_SWITCH).exp();
            let carry = a_s * (-k2 * (t - T_SWITCH)).exp() / v;
            let fresh = DOSE * (-k2 * (t - T2)).exp() / v;
            f[j] = carry + fresh;
            // κ₁ reaches occasion 2 only through the carried-over amount.
            h[(j, 1)] = -k1 * T_SWITCH * carry;
            h[(j, 2)] = -k2 * ((t - T_SWITCH) * carry + (t - T2) * fresh);
        }
        // η enters both rates.
        h[(j, 0)] = h[(j, 1)] + h[(j, 2)];
    }
    (f, h)
}

/// Hooker CWRES over the given linearization (`R = (σ·r_pred)²`, proportional error).
fn recipe(
    f: &[f64],
    h: &DMatrix<f64>,
    b: &DVector<f64>,
    w: &DMatrix<f64>,
    sigma_sd: f64,
    r_pred: &[f64],
) -> Vec<f64> {
    let n = f.len();
    let hb = h * b;
    let res = DVector::from_iterator(n, (0..n).map(|j| OBS[j].2 - (f[j] - hb[j])));
    let mut rt = h * w * h.transpose();
    for j in 0..n {
        rt[(j, j)] += (sigma_sd * r_pred[j]).powi(2);
    }
    let eig = rt.symmetric_eigen();
    let inv_sqrt = DMatrix::from_diagonal(&eig.eigenvalues.map(|l| 1.0 / l.sqrt()));
    let w_half = &eig.eigenvectors * inv_sqrt * eig.eigenvectors.transpose();
    (w_half * res).iter().copied().collect()
}

struct Case {
    model: CompiledModel,
    pop: Population,
    params: ModelParameters,
}

fn case(omega_iov: f64) -> Case {
    let model = parse_model_string(MODEL).unwrap();
    let mut params = model.default_params.clone();
    let names = params.omega_iov.as_ref().unwrap().eta_names.clone();
    params.omega_iov = Some(OmegaMatrix::from_diagonal(&[omega_iov], names));
    Case {
        model,
        pop: population(),
        params,
    }
}

/// ferx's sdtab CWRES at `b̂`, alongside the recipe's.
fn run(c: &Case, b: [f64; 3], interaction: bool) -> (Vec<f64>, Vec<f64>) {
    let theta = &c.params.theta;
    let (f, h) = closed_form(theta, b);
    let eta_hats = vec![DVector::from_element(1, b[0])];
    let h_eta = vec![h.columns(0, 1).into_owned()];
    let kappas = vec![vec![
        DVector::from_element(1, b[1]),
        DVector::from_element(1, b[2]),
    ]];
    let (subjects, _) = compute_subject_results(
        &c.model,
        &c.pop,
        &c.params,
        &eta_hats,
        &h_eta,
        &kappas,
        interaction,
        None,
        false,
    );
    let s = &subjects[0];
    // Premise: ferx predicts the closed form, so the recipe is about the same curve.
    for (j, (got, want)) in s.ipred.iter().zip(&f).enumerate() {
        assert!(
            (got - want).abs() < 1e-12 * want.abs(),
            "IPRED row {j}: ferx {got} vs closed form {want}"
        );
    }
    let sigma_sd = c.params.sigma.values[0];
    let omega = c.params.omega.matrix[(0, 0)];
    let omega_iov = c.params.omega_iov.as_ref().unwrap().matrix[(0, 0)];
    let w = DMatrix::from_diagonal(&DVector::from_vec(vec![omega, omega_iov, omega_iov]));
    let r_pred = if interaction {
        f.clone()
    } else {
        closed_form(theta, [0.0; 3]).0
    };
    let want = recipe(&f, &h, &DVector::from_row_slice(&b), &w, sigma_sd, &r_pred);
    (s.cwres.clone(), want)
}

/// The η-only recipe ferx used before #1948, for the straddle premises.
fn eta_only(c: &Case, b: [f64; 3], interaction: bool) -> Vec<f64> {
    let theta = &c.params.theta;
    let (f, h) = closed_form(theta, b);
    let w = DMatrix::from_element(1, 1, c.params.omega.matrix[(0, 0)]);
    let r_pred = if interaction {
        f.clone()
    } else {
        closed_form(theta, [0.0; 3]).0
    };
    recipe(
        &f,
        &h.columns(0, 1).into_owned(),
        &DVector::from_element(1, b[0]),
        &w,
        c.params.sigma.values[0],
        &r_pred,
    )
}

fn worst(label: &str, got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len(), "{label}: row count");
    let mut d = 0.0_f64;
    for (j, (g, w)) in got.iter().zip(want).enumerate() {
        // f64::max discards NaN, so finiteness is asserted before the fold.
        assert!(g.is_finite() && w.is_finite(), "{label}: row {j} {g} / {w}");
        d = d.max((g - w).abs());
    }
    d
}

/// The κ columns of `H` are the builder's central differences (step ~1e-6) against the
/// closed form here. Realised worst |Δ| on macOS arm64, FOCEI / FOCE: stacked 7.5e-11 /
/// 8.3e-11, f0 2.1e-10 / 1.9e-10, rtilde 1.0e-10 / 9.0e-11. `1e-8` is 48× above the
/// largest; the η-only gaps the three premises measure are 0.11–0.38, 1e7× above it.
const TOL: f64 = 1e-8;

#[test]
fn matches_the_stacked_recipe() {
    let c = case(0.04);
    let b = [0.15, -0.2, 0.25];
    for interaction in [true, false] {
        let (got, want) = run(&c, b, interaction);
        let d = worst("stacked", &got, &want);
        let gap_old = worst("eta-only premise", &eta_only(&c, b, interaction), &want);
        eprintln!(
            "MEASURE stacked interaction={interaction}: |d| {d:.3e}, eta-only gap {gap_old:.3e}"
        );
        assert!(
            gap_old > 1e-2,
            "premise: the eta-only CWRES must differ from the stacked one ({gap_old:e})"
        );
        assert!(
            d < TOL,
            "interaction={interaction}: CWRES {got:?} vs the [eta, k1, k2] recipe {want:?} ({d:e})"
        );
    }
}

#[test]
fn f0_subtracts_the_kappa_part_of_the_linearization() {
    // Ω_IOV → 1e-12: `H_κ·Ω_IOV·H_κᵀ` is ~1e-12 of `R`, so only the `f0` half is live.
    let c = case(1e-12);
    let b = [0.15, -0.2, 0.25];
    for interaction in [true, false] {
        let (got, want) = run(&c, b, interaction);
        let d = worst("f0", &got, &want);
        let gap_old = worst("eta-only premise", &eta_only(&c, b, interaction), &want);
        eprintln!("MEASURE f0 interaction={interaction}: |d| {d:.3e}, eta-only gap {gap_old:.3e}");
        assert!(
            gap_old > 1e-2,
            "premise: H_k*k_hat must be live ({gap_old:e})"
        );
        assert!(
            d < TOL,
            "interaction={interaction}: CWRES off the recipe by {d:e}: f0 is not subtracting \
             H_kappa * kappa_hat"
        );
    }
}

#[test]
fn rtilde_carries_the_iov_variance() {
    // κ̂ = 0: `H_κ·κ̂ = 0`, so only the `R̃` half is live.
    let c = case(0.04);
    let b = [0.15, 0.0, 0.0];
    for interaction in [true, false] {
        let (got, want) = run(&c, b, interaction);
        let d = worst("rtilde", &got, &want);
        let gap_old = worst("eta-only premise", &eta_only(&c, b, interaction), &want);
        eprintln!(
            "MEASURE rtilde interaction={interaction}: |d| {d:.3e}, eta-only gap {gap_old:.3e}"
        );
        assert!(
            gap_old > 1e-2,
            "premise: H_k*W_iov*H_k' must be live ({gap_old:e})"
        );
        assert!(
            d < TOL,
            "interaction={interaction}: CWRES off the recipe by {d:e}: R-tilde is missing \
             H_kappa * Omega_IOV * H_kappa'"
        );
    }
}

/// One κ for the two-occasion subject. The EBE pipeline never produces this, so
/// `iov_linearization` returns `None` and each consumer refuses on its own terms:
/// - CWRES is `NaN` on every row, since there is no `[η, κ₁, κ₂]` system to linearize over;
/// - the OFV contribution is the marginal's sentinel, `2 × 1e20`;
/// - IPRED still comes from `predict_iov` at `(η̂, κ̂₁)`. Occasion 1's rows depend on κ₁ alone,
///   so they must equal the closed form.
#[test]
fn a_kappa_shape_mismatch_is_refused() {
    let c = case(0.04);
    let (eta, kappa1) = (0.15, -0.2);
    let eta_hats = vec![DVector::from_element(1, eta)];
    let h_eta = vec![DMatrix::zeros(OBS.len(), 1)];
    let kappas = vec![vec![DVector::from_element(1, kappa1)]];
    // Occasion 2's κ is irrelevant to occasion 1's rows; 0.0 is a placeholder.
    let (f, _) = closed_form(&c.params.theta, [eta, kappa1, 0.0]);
    for interaction in [true, false] {
        let (subjects, _) = compute_subject_results(
            &c.model,
            &c.pop,
            &c.params,
            &eta_hats,
            &h_eta,
            &kappas,
            interaction,
            None,
            false,
        );
        let s = &subjects[0];
        assert_eq!(
            s.cwres.len(),
            OBS.len(),
            "interaction={interaction}: CWRES rows"
        );
        assert!(
            s.cwres.iter().all(|x| x.is_nan()),
            "interaction={interaction}: CWRES {:?} must be NaN without a linearization",
            s.cwres
        );
        assert_eq!(
            s.ofv_contribution, 2e20,
            "interaction={interaction}: OFV contribution must be the marginal's sentinel"
        );
        assert!(
            s.ipred.iter().all(|x| x.is_finite()),
            "interaction={interaction}: IPRED {:?}",
            s.ipred
        );
        for (j, &(_, occ, _)) in OBS.iter().enumerate().filter(|(_, o)| o.1 == 1) {
            let (got, want) = (s.ipred[j], f[j]);
            assert!(
                (got - want).abs() < 1e-12 * want.abs(),
                "interaction={interaction}: occasion-{occ} IPRED row {j}: ferx {got} vs closed \
                 form {want}"
            );
        }
    }
}
