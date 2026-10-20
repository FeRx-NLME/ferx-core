//! The covariance step's finite-difference steps on a small identity-packed θ (#2147).
//!
//! A θ whose lower bound is negative is packed as itself, and every FD step on a packed
//! coordinate was `eps·(1 + |x|)`: about `eps` in the θ's own units once `|θ| < 1`. On a θ of
//! `0.0077` the R-matrix stencil's `eps = 1e-2` stepped it through zero to a negative volume,
//! whose objective is huge but finite, so the step probe kept it and every SE collapsed. The
//! per-subject score of `S` (`eps = 1e-4` / `1e-5`) did not cross zero but stepped `θ` by
//! 1–10 % of itself, a truncation error that grows as `θ` shrinks. The step's floor is now
//! `|θ₀|` for such a θ, its declared initial value.
//!
//! These fits evaluate once (`outer_maxiter = 0`) and compare each rescaled file with the
//! same model at `k = 1`, where `|θ₀| > 1` keeps the old step exactly. The last test reduces
//! the fix to an already-anchored object: the #432 NONMEM 7.5.1 `MATRIX=R` standard errors
//! (`tests/covariance_se_at_nonmem_minimum.rs`), with `TVV` written as an identity-packed
//! `TVV_S = 1e-3·TVV` and the R matrix built by FD.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::omega_se_at;
use ferx_core::{fit, read_nonmem_csv, CovarianceMethod, EstimationMethod, FitOptions, FitResult};
use std::path::Path;

/// Warfarin at its FOCEI estimates, with `V` written as the identity-packed `TVV_S = k·V`
/// (bounds `±100·TVV_S`, so the lower bound is negative and the θ is packed as itself).
/// `ruv_eta` adds IIV on the residual error, which no closed-form score carries, so the
/// score is the central FD of the whole subject objective.
fn warfarin_scaled_v(k: f64, ruv_eta: bool) -> String {
    let tv = 7.737706 * k;
    let (ruv_omega, ruv_line) = if ruv_eta {
        ("omega ETA_RUV ~ 0.05", "iiv_on_ruv = ETA_RUV")
    } else {
        ("", "")
    };
    format!(
        "
[parameters]
  theta TVCL(0.132695, 0.001, 10.0)
  theta TVV_S({tv:e}, {lo:e}, {hi:e})
  theta TVKA(0.810796, 0.01, 50.0)
  omega ETA_CL ~ 0.028590
  omega ETA_V  ~ 0.009592
  omega ETA_KA ~ 0.335871
  {ruv_omega}
  sigma PROP_ERR ~ 0.010565 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV_S / {k:e} * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
  {ruv_line}
",
        lo = -100.0 * tv,
        hi = 100.0 * tv,
    )
}

/// One route to the standard errors.
#[derive(Clone, Copy, Debug)]
struct Route {
    cov: CovarianceMethod,
    interaction: bool,
    /// IIV on the residual error ([`warfarin_scaled_v`]).
    ruv_eta: bool,
    /// `method = laplace` with `reconverge_gradient_interval = 1`: the quadrature score's
    /// reconverged central FD, the route an estimated `block_sigma` correlation also takes.
    laplace_fd: bool,
}

impl Route {
    fn n_eta(&self) -> usize {
        if self.ruv_eta {
            4
        } else {
            3
        }
    }
}

fn fit_scaled(k: f64, route: Route) -> FitResult {
    let model =
        parse_model_string(&warfarin_scaled_v(k, route.ruv_eta)).expect("scaled-V model parses");
    let pop = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).expect("data loads");
    let mut opts = FitOptions::default();
    opts.method = if route.laplace_fd {
        EstimationMethod::Laplace
    } else if route.interaction {
        EstimationMethod::FoceI
    } else {
        EstimationMethod::Foce
    };
    opts.interaction = route.interaction;
    opts.reconverge_gradient_interval = usize::from(route.laplace_fd);
    opts.outer_maxiter = 0;
    opts.run_covariance_step = true;
    opts.covariance_method = route.cov;
    // The FD stencil, the route every model outside the analytic R's scope takes.
    opts.analytic_cov_hessian = false;
    opts.verbose = false;
    fit(&model, &pop, &model.default_params, &opts).expect("pinned fit runs")
}

/// Every SE on the natural scale, `TVV_S`'s divided by `k`: TVCL, TVV, TVKA, each ω, σ.
fn natural_ses(r: &FitResult, k: f64, n_eta: usize, label: &str) -> Vec<f64> {
    assert!(
        r.covariance_matrix.is_some(),
        "{label}: the covariance step failed: {:?}",
        r.warnings
    );
    let regularized: Vec<&String> = r
        .warnings
        .iter()
        .filter(|w| w.contains("Covariance step regularized"))
        .collect();
    assert!(
        regularized.is_empty(),
        "{label}: the matrix was regularized: {regularized:?}"
    );
    let th = r.se_theta.as_ref().expect("theta SEs");
    let mut v = vec![th[0], th[1] / k, th[2]];
    for i in 0..n_eta {
        v.push(omega_se_at(&r.se_omega, n_eta, i, i).expect("omega SE"));
    }
    v.push(r.se_sigma.as_ref().expect("sigma SEs")[0]);
    v
}

/// The SEs of [`natural_ses`], `ETA_RUV`'s present only on a `ruv_eta` route.
const NAMES: [&str; 8] = [
    "TVCL", "TVV", "TVKA", "ETA_CL", "ETA_V", "ETA_KA", "ETA_RUV", "PROP_ERR",
];

/// Worst relative difference over every SE, failing on a non-finite one.
fn worst_rel(got: &[f64], want: &[f64], n_eta: usize, label: &str) -> f64 {
    assert_eq!(got.len(), 4 + n_eta, "{label}");
    let mut worst: f64 = 0.0;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let name = if i == got.len() - 1 {
            "PROP_ERR"
        } else {
            NAMES[i]
        };
        assert!(g.is_finite() && g > 0.0, "{label}: SE({name}) = {g}");
        let rel = ((g - w) / w).abs();
        eprintln!("{label}: SE({name}) {g:.6e} vs {w:.6e}, rel {rel:.3e}");
        if rel > worst {
            worst = rel;
        }
    }
    worst
}

/// Each `k` in `ks` against `k = 1`, every SE within `tol`.
fn check(route: Route, ks: &[f64], tol: f64) {
    let n_eta = route.n_eta();
    let reference = natural_ses(
        &fit_scaled(1.0, route),
        1.0,
        n_eta,
        &format!("{route:?}, k = 1"),
    );
    for &k in ks {
        let label = format!("{route:?}, k = {k:e}");
        let got = natural_ses(&fit_scaled(k, route), k, n_eta, &label);
        let worst = worst_rel(&got, &reference, n_eta, &label);
        eprintln!("{label}: WORST {worst:.3e}");
        assert!(
            worst < tol,
            "{label}: worst relative SE difference {worst:e}"
        );
    }
}

/// Worst SE at every `k` 1.206e-4 (SE(TVV)), Linux x86_64 and macOS arm64 alike, so `1e-3`
/// is 8× headroom. Before the fix:
/// 1.01e-2 at `k = 1e-2`, and at `k ≤ 1e-3` the matrix was regularized and every SE ~1000×
/// small.
#[test]
fn the_fd_r_matrix_does_not_step_a_small_identity_theta_through_zero() {
    check(
        Route {
            cov: CovarianceMethod::Hessian,
            interaction: true,
            ruv_eta: false,
            laplace_fd: false,
        },
        &[1e-2, 1e-3, 1e-4],
        1e-3,
    );
}

/// The four ways the per-subject score of `S` steps θ: the Almquist Laplace closed form
/// (FOCEI), the Sheiner–Beal one (FOCE, both its `f(η̂)` and its `f(η = 0)` sensitivities),
/// the central FD of the whole subject objective (IIV on the residual error), and the
/// quadrature score's reconverged central FD (`method = laplace`).
///
/// Worst SE on Linux x86_64, macOS arm64 in brackets where it differs, then before the fix
/// (macOS arm64): Laplace 9.344e-10 [4.258e-7] (5.35e-4), Sheiner–Beal 3.515e-6 (4.97e-3),
/// FD 1.489e-4 (0.27, SE(TVV)), quadrature FD 1.393e-5 [5.805e-6] (1.876e-2, SE(TVV)). Each
/// bound sits 7–70× above the worst on either platform and 18–270× below the value before
/// the fix. The FD route runs at
/// `k = 1e-3`: at `1e-4` its `S` is rejected as singular with or without this fix, by the
/// scale-dependent floor of #2145.
#[test]
fn the_score_steps_a_small_identity_theta_relative_to_its_size() {
    for (interaction, ruv_eta, laplace_fd, k, tol) in [
        (true, false, false, 1e-4, 3e-5),
        (false, false, false, 1e-4, 3e-5),
        (true, true, false, 1e-3, 1e-3),
        (true, false, true, 1e-4, 1e-4),
    ] {
        check(
            Route {
                cov: CovarianceMethod::CrossProduct,
                interaction,
                ruv_eta,
                laplace_fd,
            },
            &[k],
            tol,
        );
    }
}

/// The #432 anchor's model at NONMEM's FOCEI minimum, `TVV` written as `TVV_S = 1e-3·TVV`
/// with a negative lower bound.
const NM_MIN_SCALED_SRC: &str = r"
[parameters]
  theta TVCL(0.132696, 0.001, 10.0)
  theta TVV_S(0.00773752, -0.773752, 0.773752)
  theta TVKA(0.810794, 0.01, 50.0)
  block_omega (ETA_CL, ETA_V) = [0.0285938, 0.00190087, 0.00959304]
  omega ETA_KA ~ 0.335846
  sigma PROP_ERR ~ 0.010565 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV_S / 1e-3 * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
";

#[test]
fn an_identity_packed_v_in_small_units_reproduces_nonmem_matrix_r_ses_by_fd() {
    let model = parse_model_string(NM_MIN_SCALED_SRC).expect("scaled #432 model parses");
    let pop = read_nonmem_csv(Path::new("data/warfarin_block_omega.csv"), None, None)
        .expect("block-omega data loads");
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.interaction = true;
    opts.outer_maxiter = 0;
    opts.run_covariance_step = true;
    opts.covariance_method = CovarianceMethod::Hessian;
    opts.analytic_cov_hessian = false;
    opts.inner_tol = 1e-8;
    opts.inner_maxiter = 500;
    opts.verbose = false;
    let r = fit(&model, &pop, &model.default_params, &opts).expect("pinned fit runs");
    assert!(
        r.covariance_matrix.is_some(),
        "the covariance step failed: {:?}",
        r.warnings
    );
    let th = r.se_theta.as_ref().expect("theta SEs");
    let omega_se = |i: usize| omega_se_at(&r.se_omega, 3, i, i).expect("omega diag SE");
    // NONMEM 7.5.1 FOCEI `$COV MATRIX=R`, `.ext` row ITERATION = -1000000001, as in #432.
    let checks: [(&str, f64, f64); 7] = [
        ("TVCL", th[0], 7.09762e-3),
        ("TVV", th[1] / 1e-3, 2.40059e-1),
        ("TVKA", th[2], 1.48631e-1),
        (
            "PROP_ERR",
            r.se_sigma.as_ref().expect("sigma SEs")[0],
            8.35394e-4,
        ),
        ("omega_CL", omega_se(0), 1.27953e-2),
        ("omega_V", omega_se(1), 4.30584e-3),
        ("omega_KA", omega_se(2), 1.50360e-1),
    ];
    let mut worst: f64 = 0.0;
    for (name, got, nm) in checks {
        let rel = ((got - nm) / nm).abs();
        eprintln!("#432 twin: SE({name}) {got:.6e} vs NONMEM {nm:.6e}, rel {rel:.3e}");
        assert!(got.is_finite(), "SE({name}) = {got:e}");
        if rel > worst {
            worst = rel;
        }
    }
    // Worst 5.175e-4 (SE(PROP_ERR); SE(TVV) 1.80e-4), Linux x86_64 and macOS arm64 alike,
    // so `2e-3` is 3.9× headroom. Before
    // the fix every SE was ~1000× small (worst 1.0).
    eprintln!("#432 twin: WORST {worst:.3e}");
    assert!(
        worst < 2e-3,
        "worst relative SE difference vs NONMEM {worst:e}"
    );
}
