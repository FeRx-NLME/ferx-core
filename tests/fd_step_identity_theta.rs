//! #2147: every covariance-step finite difference steps a packed coordinate by
//! `eps·(sᵢ + |xᵢ|)`. Before the fix `sᵢ = 1` everywhere, so an **identity-packed** θ (a
//! negative lower bound) below about `eps` in magnitude was stepped through zero — on the
//! warfarin volume written in units of 1000 L the minus point evaluated a negative volume,
//! and every SE came back ~1000× too small.
//!
//! The same model is fitted at its FOCEI estimates (`outer_maxiter = 0`) with the volume as
//! `TVV_S = TVV·k`, `V = TVV_S / k`, bounds `±100·TVV_S` (so identity-packed). Rescaling a
//! parameter by `k` must leave every other SE unchanged and scale `SE(TVV_S)` by exactly `k`;
//! `k = 1` (`|θ₀| ≥ 1`, floor 1, the pre-#2147 step) is the reference.
//!
//! One test per FD route the fix touches, so a failure names the site:
//! - `r` with `analytic_cov_hessian = false` — the R stencil and its step probe;
//! - `s` under FOCEI — the Almquist-Laplace score
//!   (`subject_nll_pop_grad_analytical_laplace_cached`);
//! - `s` under FOCE — the Sheiner–Beal score (`subject_nll_pop_grad_analytical`);
//! - `s` under FOCE on an ODE model — the central-FD fallback of `subject_nll_pop_grad`;
//! - `s` under `method = laplace` with `reconverge_gradient_interval = 1` — the quadrature
//!   score's reconverged FD (`SubjectScoreContext::finite_difference_score`).
//!
//! Evaluate-once fits: Tier 2, not gated.
use ferx_core::types::{CovarianceMethod, EstimationMethod, FitOptions, FitResult};
use ferx_core::{fit, read_nonmem_csv};
use std::path::Path;

const TVV: f64 = 7.737706;

fn model_src(k: f64, ode: bool) -> String {
    let tvv_s = TVV * k;
    let structural = if ode {
        "[structural_model]\n  ode(obs_cmt=central, states=[depot, central])\n\
         [odes]\n  d/dt(depot) = -KA * depot\n  d/dt(central) = KA * depot / V - CL / V * central\n"
    } else {
        "[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n"
    };
    format!(
        "[parameters]
  theta TVCL(0.132695, 0.001, 10.0)
  theta TVV_S({tvv_s:e}, {lo:e}, {hi:e})
  theta TVKA(0.810796, 0.01, 50.0)
  omega ETA_CL ~ 0.028590
  omega ETA_V  ~ 0.009592
  omega ETA_KA ~ 0.335871
  sigma PROP_ERR ~ 0.010565 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV_S / {k:e} * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
{structural}[error_model]
  DV ~ proportional(PROP_ERR)
",
        lo = -100.0 * tvv_s,
        hi = 100.0 * tvv_s,
    )
}

/// SEs in a fixed order: TVCL, TVV_S / k, TVKA, then Ω diagonal and σ.
fn ses(k: f64, ode: bool, opts: &FitOptions) -> Vec<f64> {
    let model = ferx_core::parser::model_parser::parse_model_string(&model_src(k, ode)).unwrap();
    let pop = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).unwrap();
    let r: FitResult = fit(&model, &pop, &model.default_params, opts).unwrap();
    let th = r
        .se_theta
        .clone()
        .unwrap_or_else(|| panic!("k = {k}: no SEs; warnings: {:?}", r.warnings));
    let mut v = vec![th[0], th[1] / k, th[2]];
    for i in 0..3 {
        v.push(ferx_core::types::omega_se_at(&r.se_omega, 3, i, i).unwrap());
    }
    v.extend(r.se_sigma.as_ref().unwrap());
    v
}

/// Assert every SE at each small `k` matches `k = 1` to `tol` relative.
fn assert_scale_invariant(route: &str, ode: bool, opts: FitOptions, ks: &[f64], tol: f64) {
    let names = [
        "TVCL", "TVV_S/k", "TVKA", "ETA_CL", "ETA_V", "ETA_KA", "PROP_ERR",
    ];
    let reference = ses(1.0, ode, &opts);
    assert!(
        reference.iter().all(|s| s.is_finite() && *s > 0.0),
        "{route}: reference SEs {reference:?}"
    );
    for &k in ks {
        let got = ses(k, ode, &opts);
        let mut worst = 0.0_f64;
        for (i, (g, r)) in got.iter().zip(&reference).enumerate() {
            assert!(g.is_finite(), "{route} k = {k}: SE({}) = {g}", names[i]);
            let rel = (g - r).abs() / r;
            eprintln!(
                "{route} k = {k:e}: SE({}) = {g:.6e} vs k = 1 {r:.6e}, rel {rel:.2e}",
                names[i]
            );
            worst = worst.max(rel);
        }
        eprintln!("{route} k = {k:e}: worst rel {worst:.3e}");
        for (i, (g, r)) in got.iter().zip(&reference).enumerate() {
            assert!(
                (g - r).abs() / r < tol,
                "{route} k = {k}: SE({}) = {g} vs k = 1 {r}",
                names[i]
            );
        }
    }
}

fn base(method: EstimationMethod, cov: CovarianceMethod) -> FitOptions {
    FitOptions {
        method,
        interaction: method == EstimationMethod::FoceI,
        outer_maxiter: 0,
        run_covariance_step: true,
        covariance_method: cov,
        covariance_method_set: true,
        analytic_cov_hessian: false,
        ..FitOptions::default()
    }
}

#[test]
fn r_fd_stencil_steps_a_small_identity_theta_relatively() {
    let opts = base(EstimationMethod::FoceI, CovarianceMethod::Hessian);
    // Measured on Linux x86_64 and macOS arm64 (same to 2 figures): worst 1.21e-4 at every k with
    // the fix — the two steps are 2 % (floor |θ₀|) and 1.1 % (floor 1) of TVV, so they do not agree
    // exactly. Before it: 1.01e-2 at k = 1e-2, 1.0 (every SE collapsed) at k ≤ 1e-3.
    assert_scale_invariant("r/fd", false, opts, &[1e-2, 1e-3, 1e-4], 1e-3);
}

#[test]
fn s_focei_laplace_score_steps_a_small_identity_theta_relatively() {
    let opts = base(EstimationMethod::FoceI, CovarianceMethod::CrossProduct);
    // Measured (Linux x86_64, macOS arm64): worst 1.2e-6 with the fix, 5.4e-4 before it (k = 1e-4).
    assert_scale_invariant("s/focei", false, opts, &[1e-3, 1e-4], 1e-5);
}

#[test]
fn s_foce_sheiner_beal_score_steps_a_small_identity_theta_relatively() {
    let opts = base(EstimationMethod::Foce, CovarianceMethod::CrossProduct);
    // Measured (Linux x86_64, macOS arm64): worst 3.5e-6 with the fix; 5.2e-4 (k = 1e-3) and 5.0e-3
    // (k = 1e-4) before it.
    assert_scale_invariant("s/foce", false, opts, &[1e-3, 1e-4], 3e-5);
}

#[test]
fn s_fd_fallback_score_steps_a_small_identity_theta_relatively() {
    let opts = base(EstimationMethod::Foce, CovarianceMethod::CrossProduct);
    // Measured (Linux x86_64, macOS arm64): worst 1.5e-7 with the fix; 9.4e-4 (k = 1e-3) and 6.2e-2
    // (k = 1e-4) before it.
    assert_scale_invariant("s/fd-fallback", true, opts, &[1e-3, 1e-4], 2e-6);
}

#[test]
fn s_quadrature_reconverged_score_steps_a_small_identity_theta_relatively() {
    let opts = FitOptions {
        reconverge_gradient_interval: 1,
        ..base(EstimationMethod::Laplace, CovarianceMethod::CrossProduct)
    };
    // Measured (Linux x86_64, macOS arm64): worst 2.7e-6 with the fix; 5.7e-5 (k = 1e-3) and 1.9e-2
    // (k = 1e-4) before it.
    assert_scale_invariant("s/agq-fd", false, opts, &[1e-3, 1e-4], 2e-5);
}

/// The profiled-levels covariance (#2130) differences its own objective — the profile
/// objective's stencil and the per-subject `B` block, both in `estimation/profile.rs`. F1
/// of the profile tests with `ED50` written as `ED50_S = ED50·k`, identity-packed.
fn profiled_ses(k: f64) -> Vec<f64> {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let text =
        std::fs::read_to_string(root.join("tests/data/profile_mbma/f1_profile.ferx")).unwrap();
    let decl = "theta ED50(20.0, 0.1, 1000.0)";
    let use_ = "EMAX * DOSE / (ED50 + DOSE)";
    // Evaluated at F1's own FOCEI optimum (OFV 97.0700, the profiled fit from the file's
    // inits): at those inits the profile objective's Hessian is not positive definite.
    let at_optimum = [
        ("theta TVE0(45.0,", "theta TVE0(48.576341,"),
        ("theta EMAX(8.0,", "theta EMAX(13.605569,"),
        ("theta ET50(2.0,", "theta ET50(4.293364,"),
        ("omega ETA_E0 ~ 4.0 (sd)", "omega ETA_E0 ~ 2.193935 (sd)"),
        (
            "kappa KAPPA_ARM ~ 4.0 (sd)",
            "kappa KAPPA_ARM ~ 12.490309 (sd)",
        ),
    ];
    assert!(text.contains(decl) && text.contains(use_), "fixture moved");
    let text = at_optimum.iter().fold(text, |t, (from, to)| {
        assert!(t.contains(from), "fixture moved: {from}");
        t.replace(from, to)
    });
    let ed50_s = 33.253532 * k;
    let text = text
        .replace(
            decl,
            &format!(
                "theta ED50_S({ed50_s:e}, {:e}, {:e})",
                -100.0 * ed50_s,
                100.0 * ed50_s
            ),
        )
        .replace(use_, &format!("EMAX * DOSE / (ED50_S / {k:e} + DOSE)"));
    let path = std::env::temp_dir().join(format!(
        "ferx_fd_step_identity_theta_{}_{k:e}.ferx",
        std::process::id()
    ));
    std::fs::write(&path, text).unwrap();
    let p = ferx_core::prepare_run(
        path.to_str().unwrap(),
        Some(
            root.join("tests/data/mbma_placebo/mbma_placebo.csv")
                .to_str()
                .unwrap(),
        ),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let opts = FitOptions {
        verbose: false,
        outer_maxiter: 0,
        run_covariance_step: true,
        ..p.parsed.fit_options.clone()
    };
    let r = fit(&p.parsed.model, &p.population, &p.init_params, &opts).unwrap();
    let mut th = r
        .se_theta
        .clone()
        .unwrap_or_else(|| panic!("k = {k}: no SEs; warnings: {:?}", r.warnings));
    th[2] /= k;
    th
}

#[test]
fn profiled_levels_covariance_steps_a_small_identity_theta_relatively() {
    let reference = profiled_ses(1.0);
    assert!(
        reference.iter().all(|s| s.is_finite() && *s > 0.0),
        "reference SEs {reference:?}"
    );
    for k in [1e-3, 1e-4] {
        let got = profiled_ses(k);
        assert_eq!(got.len(), reference.len());
        let mut worst = 0.0_f64;
        for (i, (g, r)) in got.iter().zip(&reference).enumerate() {
            assert!(g.is_finite(), "k = {k}: SE[{i}] = {g}");
            let rel = (g - r).abs() / r;
            eprintln!("profile k = {k:e}: SE[{i}] = {g:.6e} vs k = 1 {r:.6e}, rel {rel:.2e}");
            worst = worst.max(rel);
        }
        eprintln!("profile k = {k:e}: worst rel {worst:.3e}");
        // Measured on Linux x86_64 and macOS arm64 (same to 2 figures): worst 1.03e-4
        // (SE(ED50_S)/k) at both k with the fix. Reverting the floor in the profile stencil alone
        // gives 3.3e-2, in the `B` block alone 7.2e-3.
        assert!(worst < 1e-3, "profile k = {k}: worst rel {worst}");
    }
}

/// NONMEM, by reduction to the #432 anchor (`tests/covariance_se_at_nonmem_minimum.rs`):
/// the block-Ω warfarin model at NONMEM 7.5.1's own FOCEI minimum, against its
/// `$COVARIANCE MATRIX=R` SEs — with `TVV` written as the identity-packed
/// `TVV_S = 1e-3·TVV` and the R matrix built by FD. Rescaling a θ multiplies its SE by the
/// same factor and leaves every other SE alone, in NONMEM as in any parameterisation-
/// invariant R, so the anchor's numbers apply with `SE(TVV) = SE(TVV_S)·1000`. Before #2147
/// this evaluation stepped `TVV_S` through zero and reported every SE ~1000× too small.
#[test]
fn fd_r_on_a_rescaled_volume_matches_nonmem_matrix_r() {
    const K: f64 = 1e-3;
    let tvv_s = 7.73752 * K;
    let src = format!(
        "[parameters]
  theta TVCL(0.132696, 0.001, 10.0)
  theta TVV_S({tvv_s:e}, {lo:e}, {hi:e})
  theta TVKA(0.810794, 0.01, 50.0)
  block_omega (ETA_CL, ETA_V) = [0.0285938, 0.00190087, 0.00959304]
  omega ETA_KA ~ 0.335846
  sigma PROP_ERR ~ 0.010565 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV_S / {K:e} * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
",
        lo = -100.0 * tvv_s,
        hi = 100.0 * tvv_s,
    );
    let model = ferx_core::parser::model_parser::parse_model_string(&src).unwrap();
    let pop = read_nonmem_csv(Path::new("data/warfarin_block_omega.csv"), None, None).unwrap();
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        interaction: true,
        mu_referencing: true,
        outer_maxiter: 0,
        run_covariance_step: true,
        analytic_cov_hessian: false,
        inner_tol: 1e-8,
        inner_maxiter: 500,
        verbose: false,
        ..FitOptions::default()
    };
    let r = fit(&model, &pop, &model.default_params, &opts).unwrap();
    let se_theta = r
        .se_theta
        .as_ref()
        .unwrap_or_else(|| panic!("no SEs; warnings: {:?}", r.warnings));
    let se_sigma = r.se_sigma.as_ref().unwrap();
    let omega_se = |i: usize| ferx_core::types::omega_se_at(&r.se_omega, 3, i, i).unwrap();
    // NONMEM 7.5.1 FOCEI MATRIX=R SE row at this minimum, as in the #432 test.
    let checks: [(&str, f64, f64); 7] = [
        ("TVCL", se_theta[0], 7.09762e-3),
        ("TVV", se_theta[1] / K, 2.40059e-1),
        ("TVKA", se_theta[2], 1.48631e-1),
        ("PROP_ERR", se_sigma[0], 8.35394e-4),
        ("omega_CL", omega_se(0), 1.27953e-2),
        ("omega_V", omega_se(1), 4.30584e-3),
        ("omega_KA", omega_se(2), 1.50360e-1),
    ];
    let mut worst = 0.0_f64;
    for (name, se, nm) in checks {
        assert!(se.is_finite(), "SE({name}) = {se}");
        let rel = (se - nm).abs() / nm;
        eprintln!("NONMEM reduction: SE({name}) = {se:.6e} vs {nm:.6e}, rel {rel:.2e}");
        worst = worst.max(rel);
    }
    eprintln!("NONMEM reduction: worst rel {worst:.3e}");
    // Measured on Linux x86_64 and macOS arm64 (same to 2 figures): worst 5.2e-4 (PROP_ERR); TVV
    // 1.8e-4. The #432 test's own 6 % band covers ferx's analytic route; this one is the FD stencil
    // alone, so it can be tight.
    assert!(worst < 3e-3, "worst rel {worst}");
}

/// A mixture subject's score adds `−∂ ln pᵢₖ/∂θ` from the mixing map, a finite difference
/// in θ's own units (`mixing_logp_grad`, `mixing_logp_grad_fd` for the `p` form). It stepped
/// `1e-6·max(|θ|, 1)`, so a mixing θ near `1e-6` was stepped through zero. The
/// `tests/mixture_cov_s_nonmem.rs` model at the NONMEM finals, with its mixing parameter
/// rewritten nonlinearly through a rescaled θ `M_S` (`M = sqrt(M_S / k)`, so the mixing map is
/// the same at every `k`), under `covariance_method = s`: every SE must match `k = 1`, and
/// `SE(M_S)` must scale as the derivative `dM_S/dM = 2·M·k` says.
fn mixture_ses(k: f64, logit: bool) -> Vec<f64> {
    // NONMEM 7.5.1 finals (`mixture_iv_cov.ext`); M = P1 on the `p` form, −logit(P1) on
    // the logit form, so M > 0 and `sqrt` is real either way.
    let p1: f64 = 0.471970;
    let m = if logit { -(p1 / (1.0 - p1)).ln() } else { p1 };
    let m_s = m * m * k;
    let mixing = if logit {
        format!("logit(1) = -sqrt(M_S / {k:e})")
    } else {
        format!("p(1) = sqrt(M_S / {k:e})")
    };
    let src = format!(
        "[parameters]
  theta TVCL1(0.980833, 0.01, 100.0)
  theta TVCL2(2.84198, 0.01, 100.0)
  theta TVV(9.97859, 0.1, 1000.0)
  theta M_S({m_s:e}, {lo:e}, {hi:e})
  omega ETA_CL ~ 0.09 FIX
  sigma EPS ~ 0.04 FIX
[mixture]
  nsub = 2
  {mixing}
[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(EPS)
",
        lo = -100.0 * m_s,
        hi = 100.0 * m_s,
    );
    let model = ferx_core::parser::model_parser::parse_model_string(&src).unwrap();
    let pop = read_nonmem_csv(
        Path::new("tests/nonmem/mixture_iv.csv"),
        Some(&["WT"]),
        None,
    )
    .unwrap();
    let opts = FitOptions {
        interaction: true,
        outer_maxiter: 0,
        run_covariance_step: true,
        covariance_method: CovarianceMethod::CrossProduct,
        covariance_method_set: true,
        ..FitOptions::default()
    };
    let r = fit(&model, &pop, &model.default_params, &opts).unwrap();
    let mut se = r
        .se_theta
        .clone()
        .unwrap_or_else(|| panic!("k = {k}: no SEs; warnings: {:?}", r.warnings));
    // SE(M) by the delta method, M = sqrt(M_S / k): dM/dM_S = 1 / (2·M·k).
    se[3] /= 2.0 * m * k;
    se
}

/// Down to `k = 1e-6`. At `k ≤ 1e-5` the `M_S` score column is ~1e5× the others, and until
/// #2145 the eigenvalue floor on the raw `S` refused it as rank-deficient; the floor now
/// rescales identity-packed θ coordinates first, so those `k` are reachable (measured on macOS
/// arm64: worst 2.9e-15 at `k = 1e-5`, 1.1e-15 at `1e-6`). The mixing derivative at θ = 7.7e-8
/// itself is pinned in `estimation::mixture`.
fn assert_mixture_scale_invariant(logit: bool, tol: f64) {
    let form = if logit { "logit" } else { "p" };
    let reference = mixture_ses(1.0, logit);
    assert!(
        reference.iter().all(|s| s.is_finite() && *s > 0.0),
        "{form}: reference SEs {reference:?}"
    );
    for k in [1e-4, 1e-5, 1e-6] {
        let got = mixture_ses(k, logit);
        let mut worst = 0.0_f64;
        for (i, (g, r)) in got.iter().zip(&reference).enumerate() {
            assert!(g.is_finite(), "{form} k = {k}: SE[{i}] = {g}");
            let rel = (g - r).abs() / r;
            eprintln!(
                "mixture {form} k = {k:e}: SE[{i}] = {g:.6e} vs k = 1 {r:.6e}, rel {rel:.2e}"
            );
            worst = worst.max(rel);
        }
        eprintln!("mixture {form} k = {k:e}: worst rel {worst:.3e}");
        assert!(worst < tol, "mixture {form} k = {k}: worst rel {worst}");
    }
}

#[test]
fn s_mixture_logit_mixing_score_steps_a_small_theta_relatively() {
    // Measured: worst 1.1e-9 (macOS arm64) / 4.6e-9 (Linux x86_64) with the fix; 1.0e-1 with
    // `mixing_logp_grad`'s step floor at 1.
    assert_mixture_scale_invariant(true, 1e-6);
}

#[test]
fn s_mixture_p_mixing_score_steps_a_small_theta_relatively() {
    // Measured (macOS arm64, Linux x86_64): worst 6.4e-9 with the fix; 4.9e-4 with
    // `mixing_logp_grad_fd`'s step floor at 1.
    assert_mixture_scale_invariant(false, 1e-6);
}
