//! The covariance step's eigenvalue floor does not see the units of an identity-packed θ
//! (#2145).
//!
//! A θ whose lower bound is negative is packed as itself, so its row of `R` and `S` carries
//! the θ's units: writing it `k` times larger scales that row by `1/k`. The floor
//! `max(1e-10·λ_max, 1e-12)` was applied to that raw spectrum, so at `k = 1e5` it clipped a
//! positive-definite `R` and understated `SE(θ)` by up to 2000×, failed `S⁻¹` outright, and at
//! `k = 3e-6` clipped the *other* coordinates instead. These fits evaluate once
//! (`outer_maxiter = 0`) and compare each rescaled file with the same model at `k = 1`.
//!
//! The second test reduces the fix to an already-anchored object: the #432 NONMEM 7.5.1
//! `MATRIX=R` standard errors (`tests/covariance_se_at_nonmem_minimum.rs`), with `TVV`
//! written as an identity-packed `TVV_S = 1e5·TVV`.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::omega_se_at;
use ferx_core::{fit, read_nonmem_csv, CovarianceMethod, EstimationMethod, FitOptions, FitResult};
use std::path::Path;

/// Warfarin at its FOCEI estimates, with `V` written as the identity-packed `TVV_S = k·V`
/// (bounds `±100·TVV_S`, so the lower bound is negative and the θ is packed as itself).
fn warfarin_scaled_v(k: f64) -> String {
    let tv = 7.737706 * k;
    format!(
        "
[parameters]
  theta TVCL(0.132695, 0.001, 10.0)
  theta TVV_S({tv:e}, {lo:e}, {hi:e})
  theta TVKA(0.810796, 0.01, 50.0)
  omega ETA_CL ~ 0.028590
  omega ETA_V  ~ 0.009592
  omega ETA_KA ~ 0.335871
  sigma PROP_ERR ~ 0.010565 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV_S / {k:e} * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
",
        lo = -100.0 * tv,
        hi = 100.0 * tv,
    )
}

fn pinned_opts(method: CovarianceMethod) -> FitOptions {
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.interaction = true;
    opts.outer_maxiter = 0;
    opts.run_covariance_step = true;
    opts.covariance_method = method;
    opts.verbose = false;
    opts
}

fn fit_scaled(k: f64, method: CovarianceMethod) -> FitResult {
    let model = parse_model_string(&warfarin_scaled_v(k)).expect("scaled-V model parses");
    let pop = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).expect("data loads");
    fit(&model, &pop, &model.default_params, &pinned_opts(method)).expect("pinned fit runs")
}

/// Every SE on the natural scale, `TVV_S`'s divided by `k`: TVCL, TVV, TVKA, the three ω, σ.
fn natural_ses(r: &FitResult, k: f64) -> Vec<f64> {
    let label = format!("k = {k:e}");
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
        "{label}: a positive-definite matrix was regularized: {regularized:?}"
    );
    let th = r.se_theta.as_ref().expect("theta SEs");
    let mut v = vec![th[0], th[1] / k, th[2]];
    for i in 0..3 {
        v.push(omega_se_at(&r.se_omega, 3, i, i).expect("omega SE"));
    }
    v.push(r.se_sigma.as_ref().expect("sigma SEs")[0]);
    v
}

const NAMES: [&str; 7] = [
    "TVCL", "TVV", "TVKA", "ETA_CL", "ETA_V", "ETA_KA", "PROP_ERR",
];

/// Worst relative difference over the coordinates in `which`, failing on a non-finite SE.
fn worst_rel(got: &[f64], want: &[f64], which: &[usize], label: &str) -> f64 {
    let mut worst: f64 = 0.0;
    for &i in which {
        assert!(
            got[i].is_finite() && got[i] > 0.0,
            "{label}: SE({}) = {}",
            NAMES[i],
            got[i]
        );
        let rel = ((got[i] - want[i]) / want[i]).abs();
        eprintln!(
            "{label}: SE({}) {:.6e} vs {:.6e}, rel {rel:.3e}",
            NAMES[i], got[i], want[i]
        );
        if rel > worst {
            worst = rel;
        }
    }
    worst
}

#[test]
fn rescaling_an_identity_packed_theta_leaves_every_standard_error_where_it_was() {
    let all: Vec<usize> = (0..7).collect();
    for method in [
        CovarianceMethod::CrossProduct,
        CovarianceMethod::Hessian,
        CovarianceMethod::Sandwich,
    ] {
        let reference = natural_ses(&fit_scaled(1.0, method), 1.0);
        // Before #2145 every cell failed: `s` could not invert `S` at any of the three k, and
        // `r` / `rsr` were regularized. In large units that understated SE(TVV) by 4.5×
        // (r, k = 1e5) up to 2000× (rsr, k = 1e6); in small units the floor clipped the
        // *other* coordinates, and SE(TVKA) came out 2.5× (r) and 6.5× (rsr) too small at
        // k = 3e-6.
        //
        // Since #2239 steps an identity-packed θ relative to its θ₀, the FD stencils are
        // equivariant under the rescaling, so what is left is rounding. Measured worst relative
        // difference over all seven SEs, macOS arm64: 8.2e-10 (`s`, k = 3e-6), 7.8e-11 (`rsr`,
        // k = 1e6), 7.8e-16 (`r`, k = 1e6). The bound is 12× the worst.
        for k in [1e5, 1e6, 3e-6] {
            let label = format!("{method:?}, k = {k:e}");
            let got = natural_ses(&fit_scaled(k, method), k);
            let worst = worst_rel(&got, &reference, &all, &label);
            eprintln!("WORST {label}: {worst:.3e}");
            assert!(
                worst < 1e-8,
                "{label}: worst relative SE difference {worst:e}"
            );
        }
    }
}

/// The #432 anchor's model at NONMEM's FOCEI minimum, `TVV` written as `TVV_S = 1e5·TVV`
/// with a negative lower bound.
const NM_MIN_SCALED_SRC: &str = r"
[parameters]
  theta TVCL(0.132696, 0.001, 10.0)
  theta TVV_S(773752.0, -77375200.0, 77375200.0)
  theta TVKA(0.810794, 0.01, 50.0)
  block_omega (ETA_CL, ETA_V) = [0.0285938, 0.00190087, 0.00959304]
  omega ETA_KA ~ 0.335846
  sigma PROP_ERR ~ 0.010565 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV_S / 1e5 * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
";

#[test]
fn an_identity_packed_v_in_large_units_reproduces_nonmem_matrix_r_ses() {
    let model = parse_model_string(NM_MIN_SCALED_SRC).expect("scaled #432 model parses");
    let pop = read_nonmem_csv(Path::new("data/warfarin_block_omega.csv"), None, None)
        .expect("block-omega data loads");
    let mut opts = pinned_opts(CovarianceMethod::Hessian);
    opts.inner_tol = 1e-8;
    opts.inner_maxiter = 500;
    let r = fit(&model, &pop, &model.default_params, &opts).expect("pinned fit runs");
    assert!(
        r.covariance_matrix.is_some(),
        "the covariance step failed: {:?}",
        r.warnings
    );
    let th = r.se_theta.as_ref().expect("theta SEs");
    let omega_se = |i: usize| omega_se_at(&r.se_omega, 3, i, i).expect("omega diag SE");
    // NONMEM 7.5.1 FOCEI `$COV MATRIX=R`, `.ext` row ITERATION = -1000000001, as in #432.
    // Measured worst relative difference 1.8e-5 (TVKA; the others 3.5e-7 to 6.3e-6), Linux
    // x86_64 (glibc 2.28), and the same worst on macOS arm64; the bound is 5.5× that. Before #2145 the floor clipped this R: SE(TVV) came out
    // 0.0308, 7.8× too small, and SE(TVCL) 0.65 % low. NONMEM's own values are printed to six
    // significant figures, so the bound cannot usefully go much below 1e-5.
    let checks: [(&str, f64, f64); 7] = [
        ("TVCL", th[0], 7.09762e-3),
        ("TVV", th[1] / 1e5, 2.40059e-1),
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
    for (name, got, nm) in checks {
        let rel = ((got - nm) / nm).abs();
        eprintln!("#432 twin: SE({name}) {got:.6e} vs NONMEM {nm:.6e}, rel {rel:.3e}");
        assert!(
            got.is_finite() && rel < 1e-4,
            "SE({name}) = {got:e} vs NONMEM {nm:e}"
        );
    }
}
