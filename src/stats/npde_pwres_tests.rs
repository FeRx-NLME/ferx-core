//! EPRED and PWRES / EWRES (#2871): the Monte-Carlo population-weighted residual
//! `Σ_sim^(-1/2) (y − E_sim[y])` against closed forms.
//!
//! Two oracles, both exact moments rather than a second engine:
//!
//! - **Kernel**: replicates drawn straight from a known multivariate normal, so
//!   `μ` and `Σ` are the generating values and the analytic residual is
//!   `Σ^(-1/2) (y − μ)`.
//! - **End to end** through [`compute_npde_npd`]: a compartment-free model,
//!   `y_j = B·exp(η_B) + η_S·t_j + ε_j`, whose first two moments are closed form
//!   (`E[y] = B·e^{ω_B/2}`, `Cov = B²(e^{2ω_B} − e^{ω_B}) + t_i t_j ω_S + δ_ij σ`).
//!   The log-normal term puts `E[y]` a factor `e^{ω_B/2}` above `PRED`, so a residual
//!   centred on `PRED` instead of `EPRED` is visible, and the `η_S·t` term makes
//!   the within-subject correlation unequal across pairs, so the decorrelation is
//!   not a scalar.

use super::*;
use approx::assert_relative_eq;

/// `Σ^(-1/2) r` (symmetric root) and `L⁻¹ r` (Cholesky) from an exact `Σ`.
fn analytic(sigma: &DMatrix<f64>, r: &DVector<f64>) -> (DVector<f64>, DVector<f64>) {
    let eig = nalgebra::SymmetricEigen::new(sigma.clone());
    let inv_sqrt = &eig.eigenvectors
        * DMatrix::from_diagonal(&eig.eigenvalues.map(|l| 1.0 / l.sqrt()))
        * eig.eigenvectors.transpose();
    let l = nalgebra::Cholesky::new(sigma.clone())
        .expect("the oracle Σ is PD")
        .l();
    let chol = l.solve_lower_triangular(r).expect("PD factor");
    (inv_sqrt * r, chol)
}

/// Worst |Δ| between two vectors, asserting each operand finite first (a `NaN`
/// would be discarded by `f64::max` and pass on the strength of the finite rows).
fn worst(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let diffs: Vec<f64> = a.iter().zip(b).map(|(x, y)| (x - y).abs()).collect();
    assert!(
        diffs.iter().all(|e| e.is_finite()),
        "non-finite operand in {a:?} / {b:?}"
    );
    diffs.into_iter().fold(0.0, f64::max)
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Worst |PWRES − Σ^(-1/2)(y − μ)| allowed at K = 50 000 (measured 0.01260).
const KERNEL_PWRES_TOL: f64 = 0.025;

/// A 3-dimensional Gaussian with unequal, strong correlations (0.80, 0.15, 0.60),
/// and an observation whose decorrelated coordinates stay within |z| < 2 under
/// both factors, away from the tails where an empirical-CDF rank is noisy.
fn kernel_oracle() -> (DVector<f64>, DMatrix<f64>, Vec<f64>) {
    let mu = DVector::from_vec(vec![10.0, 6.0, 2.0]);
    #[rustfmt::skip]
    let sigma = DMatrix::from_row_slice(3, 3, &[
        4.0, 2.4, 0.6,
        2.4, 2.25, 0.9,
        0.6, 0.9, 1.0,
    ]);
    let observed = vec![12.0, 5.5, 1.5];
    (mu, sigma, observed)
}

/// `K` replicates of `N(μ, Σ)`, one per column.
fn mvn_sims(mu: &DVector<f64>, sigma: &DMatrix<f64>, k: usize, seed: u64) -> DMatrix<f64> {
    let l = nalgebra::Cholesky::new(sigma.clone()).expect("PD").l();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let n = mu.len();
    let mut sims = DMatrix::<f64>::zeros(n, k);
    for c in 0..k {
        let z = DVector::from_iterator(n, (0..n).map(|_| normal.sample(&mut rng)));
        sims.set_column(c, &(mu + &l * z));
    }
    sims
}

/// **Kernel closed form.** PWRES tends to `Σ^(-1/2)(y − μ)` as `K` grows, EPRED to
/// `μ`, and the two decorrelation conventions disagree per observation while
/// agreeing in norm.
///
/// Measured (seed 2871): worst |PWRES − analytic| 0.04540 at K = 2 000 and 0.01260
/// at K = 50 000 (ratio 3.6 against the `1/√K` rate's 5); worst |EPRED − μ|
/// 0.00767 at K = 50 000. [`KERNEL_PWRES_TOL`] carries 2× headroom over the
/// K = 50 000 figure. The mutations it catches:
///
/// - **Cholesky instead of the symmetric root**: |PWRES − L⁻¹r| measured 0.686,
///   27× the bound (analytic |sym − chol| 0.674 at row 1).
/// - **diagonal-only covariance** (`r_j / √Σ_jj`): measured 1.063 off, 42× the
///   bound.
#[test]
fn pwres_kernel_converges_to_the_symmetric_root_of_the_exact_covariance() {
    let (mu, sigma, observed) = kernel_oracle();
    let r = DVector::from_iterator(3, observed.iter().zip(mu.iter()).map(|(y, m)| y - m));
    let (sym, chol) = analytic(&sigma, &r);

    let gaps: Vec<f64> = [2_000usize, 50_000]
        .iter()
        .map(|&k| {
            let sims = mvn_sims(&mu, &sigma, k, 2871);
            let out = npde_scores(&observed, &[0, 0, 0], &sims, BloqMethod::Drop);
            let e = epred(&sims);
            let gap = worst(&out.pwres, sym.as_slice());
            eprintln!(
                "K = {k}: worst |PWRES − Σ^(-1/2) r| = {gap:.5}, worst |EPRED − μ| = {:.5}",
                worst(&e, mu.as_slice())
            );
            if k == 50_000 {
                // Measured 0.00767; 2× headroom.
                assert!(
                    worst(&e, mu.as_slice()) < 0.016,
                    "EPRED must be the replicate mean, got {e:?} against μ = {mu:?}"
                );
                // The norm is convention-free: √(rᵀΣ⁻¹r) under either factor.
                let quad = r.dot(&(sigma.clone().try_inverse().unwrap() * &r)).sqrt();
                eprintln!("‖PWRES‖ − √(rᵀΣ⁻¹r) = {:.5}", norm(&out.pwres) - quad);
                assert!(
                    (norm(&out.pwres) - quad).abs() < 0.025, // measured 0.01261
                    "‖PWRES‖ = {} must approach √(rᵀΣ⁻¹r) = {quad}",
                    norm(&out.pwres)
                );
            }
            gap
        })
        .collect();
    assert!(
        gaps[1] < KERNEL_PWRES_TOL,
        "worst |PWRES − Σ^(-1/2)(y − μ)| = {:.5} at K = 50 000 exceeds \
         {KERNEL_PWRES_TOL} (measured 0.01260)",
        gaps[1]
    );
    assert!(
        gaps[1] < gaps[0],
        "PWRES must converge as K grows: {:.5} (K = 2 000) → {:.5} (K = 50 000)",
        gaps[0],
        gaps[1]
    );

    // The convention straddle, on the analytic values themselves: same norm,
    // different coordinates. If these agreed the oracle could not tell the two
    // factors apart, and the PWRES bound above would not pin the symmetric one.
    assert!(
        (norm(sym.as_slice()) - norm(chol.as_slice())).abs() < 1e-12,
        "both factors give √(rᵀΣ⁻¹r)"
    );
    let sym_vs_chol = worst(sym.as_slice(), chol.as_slice());
    assert!(
        sym_vs_chol > 0.5,
        "the oracle Σ must separate the conventions per observation, got {sym_vs_chol}"
    );
    let sims = mvn_sims(&mu, &sigma, 50_000, 2871);
    let pwres = npde_scores(&observed, &[0, 0, 0], &sims, BloqMethod::Drop).pwres;
    assert!(
        worst(&pwres, chol.as_slice()) > 10.0 * KERNEL_PWRES_TOL,
        "PWRES {pwres:?} sits on the Cholesky-decorrelated residual {chol:?}: the \
         symmetric root (NONMEM's EWRES default) was swapped for WRESCHOL's factor"
    );
    // Diagonal-only scaling is a third, wrong answer.
    let diag: Vec<f64> = (0..3).map(|j| r[j] / sigma[(j, j)].sqrt()).collect();
    assert!(
        worst(&pwres, &diag) > 10.0 * KERNEL_PWRES_TOL,
        "PWRES {pwres:?} matches the per-row standardisation {diag:?}: the \
         off-diagonal covariance was dropped"
    );
}

/// NPDE keeps the Cholesky (Brendel/Comets) convention while PWRES takes the
/// symmetric root — the two choices are independent. On the first observation
/// the two factors coincide in direction (`L` is lower triangular, so
/// `(L⁻¹ r)_1 = r_1/√Σ_11`), so NPDE row 1 is `Φ⁻¹(Φ(r_1/√Σ_11))` exactly under a
/// Gaussian reference: `1.0` here. A symmetric NPDE would put it at
/// `(Σ^(-1/2) r)_1 = 1.674` instead.
#[test]
fn npde_stays_cholesky_while_pwres_is_symmetric() {
    let (mu, sigma, observed) = kernel_oracle();
    let r = DVector::from_iterator(3, observed.iter().zip(mu.iter()).map(|(y, m)| y - m));
    let (sym, chol) = analytic(&sigma, &r);
    assert_relative_eq!(chol[0], 1.0, epsilon = 1e-12);
    let sims = mvn_sims(&mu, &sigma, 50_000, 2871);
    let out = npde_scores(&observed, &[0, 0, 0], &sims, BloqMethod::Drop);
    // Measured |NPDE − L⁻¹r| worst 0.0165 (rank noise at K = 50 000); 2× headroom.
    // A symmetric NPDE would sit 0.67 away.
    let npde_gap = worst(&out.npde, chol.as_slice());
    assert!(
        npde_gap < 0.035,
        "NPDE {:?} must be the Cholesky-decorrelated score {chol:?} (gap {npde_gap:.5})",
        out.npde
    );
    assert!(
        worst(&out.npde, sym.as_slice()) > 0.5,
        "NPDE {:?} moved to the symmetric root {sym:?}",
        out.npde
    );
}

/// A compartment-free model with closed-form first two moments; see the module
/// docs. One subject, observations at `t = 0, 2, 5`.
const B: f64 = 10.0;
const OMEGA_B: f64 = 0.25;
const OMEGA_S: f64 = 0.04;
const SIGMA_ADD: f64 = 0.5;
const TIMES: [f64; 3] = [0.0, 2.0, 5.0];

fn end_to_end_model_and_population(observed: &[f64]) -> (CompiledModel, Population) {
    let text = format!(
        "[parameters]\n  theta TVB({B}, 0.1, 100.0)\n  omega ETA_B ~ {OMEGA_B}\n  \
         omega ETA_S ~ {OMEGA_S}\n  sigma ADD ~ {SIGMA_ADD} (variance)\n\
         [individual_parameters]\n  BASE = TVB * exp(ETA_B)\n  SL = ETA_S\n\
         [structural_model]\n  y = BASE + SL * TIME\n\
         [error_model]\n  DV ~ additive(ADD)\n"
    );
    let model = crate::parser::model_parser::parse_full_model(&text)
        .expect("parse")
        .model;
    let pop = Population {
        subjects: vec![crate::types::Subject {
            id: "1".into(),
            obs_times: TIMES.to_vec(),
            observations: observed.to_vec(),
            obs_cmts: vec![1; TIMES.len()],
            cens: vec![0; TIMES.len()],
            ..Default::default()
        }],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    (model, pop)
}

fn end_to_end_moments() -> (DVector<f64>, DMatrix<f64>) {
    let mean = B * (OMEGA_B / 2.0).exp();
    let var_b = B * B * ((2.0 * OMEGA_B).exp() - OMEGA_B.exp());
    let n = TIMES.len();
    let mu = DVector::from_element(n, mean);
    let sigma = DMatrix::from_fn(n, n, |i, j| {
        var_b + TIMES[i] * TIMES[j] * OMEGA_S + if i == j { SIGMA_ADD } else { 0.0 }
    });
    (mu, sigma)
}

/// **End to end.** `compute_npde_npd`'s EPRED and PWRES approach the closed-form
/// `E[y]` and `Σ^(-1/2)(y − E[y])` of the fitted population model.
///
/// Measured (seed 2871): worst |EPRED − E[y]| 0.01267 and worst |PWRES − analytic|
/// 0.00855 at K = 40 000. Bounds carry ~2.3× headroom. The mutations:
///
/// - **PRED (η = 0) instead of EPRED**: `E[y] − PRED = B(e^{ω_B/2} − 1) = 1.33`
///   (measured |EPRED − PRED| 1.344), 45× the EPRED bound; the residual moves
///   with it.
/// - **Cholesky**: measured 2.160 off, 108× the PWRES bound; **diagonal-only**:
///   2.590 off, 130×. The η_B term makes the correlations ≈ 0.98, so the factors
///   are far apart, and the η_S·t term makes them unequal across pairs.
#[test]
fn epred_and_pwres_match_the_closed_form_moments_end_to_end() {
    let observed = [14.0, 10.0, 12.0];
    let (model, pop) = end_to_end_model_and_population(&observed);
    let (mu, sigma) = end_to_end_moments();
    let r = DVector::from_iterator(3, observed.iter().zip(mu.iter()).map(|(y, m)| y - m));
    let (sym, chol) = analytic(&sigma, &r);

    let out = compute_npde_npd(
        &model,
        &pop,
        &model.default_params,
        40_000,
        Some(2871),
        None,
    )
    .expect("npde");
    let s = &out[0];
    let epred_gap = worst(&s.epred, mu.as_slice());
    let pwres_gap = worst(&s.pwres, sym.as_slice());
    eprintln!("#2871 end-to-end: worst |ΔEPRED| = {epred_gap:.5}, worst |ΔPWRES| = {pwres_gap:.5}");
    assert!(
        epred_gap < 0.03,
        "EPRED {:?} must be E[y] = {mu:?} (gap {epred_gap:.5}, measured 0.01267)",
        s.epred
    );
    // PRED is the η = 0 prediction, B, for every row.
    assert!(
        worst(&s.epred, &[B; 3]) > 10.0 * 0.03,
        "EPRED {:?} sits on PRED = {B}: the Monte-Carlo mean was replaced by the \
         typical-value prediction",
        s.epred
    );
    assert!(
        pwres_gap < 0.02,
        "PWRES {:?} must be Σ^(-1/2)(y − E[y]) = {sym:?} (gap {pwres_gap:.5}, \
         measured 0.00855)",
        s.pwres
    );
    assert!(
        worst(&s.pwres, chol.as_slice()) > 10.0 * 0.02,
        "PWRES {:?} matches the Cholesky residual {chol:?}",
        s.pwres
    );
    let diag: Vec<f64> = (0..3).map(|j| r[j] / sigma[(j, j)].sqrt()).collect();
    assert!(
        worst(&s.pwres, &diag) > 10.0 * 0.02,
        "PWRES {:?} matches the per-row standardisation {diag:?}",
        s.pwres
    );
}

/// EPRED survives where the decorrelated scores do not: on a subject with an M3
/// censored row the NPDE and PWRES are `NaN` (the LLOQ would mix into the
/// decorrelation), while EPRED — a prediction, like `PRED` — stays finite on every
/// row, censored or not. And with `K <= n_obs` the covariance is singular, so the
/// same split holds.
#[test]
fn epred_is_emitted_where_pwres_is_masked() {
    let observed = [14.0, 10.0, 12.0];
    let (mut model, mut pop) = end_to_end_model_and_population(&observed);
    pop.subjects[0].cens = vec![0, 1, 0];
    model.bloq_method = BloqMethod::M3;
    assert!(model.bloq_in_force().is_censored_row(1));
    let out =
        compute_npde_npd(&model, &pop, &model.default_params, 500, Some(1), None).expect("npde");
    let s = &out[0];
    assert!(s.pwres.iter().chain(&s.npde).all(|v| v.is_nan()), "{s:?}");
    assert!(s.epred.iter().all(|v| v.is_finite()), "{s:?}");

    let (model, pop) = end_to_end_model_and_population(&observed);
    let out =
        compute_npde_npd(&model, &pop, &model.default_params, 3, Some(1), None).expect("npde");
    let s = &out[0];
    assert!(s.pwres.iter().all(|v| v.is_nan()), "K = n_obs: {s:?}");
    assert!(s.epred.iter().all(|v| v.is_finite()), "K = n_obs: {s:?}");
}
