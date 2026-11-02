//! Nested levels composed with a residual `correlation … by` line (#1082 PR5): the IOV
//! marginal's `R̃ = H Σ_b Hᵀ + R`, the level-name grouping vocabulary
//! (`E_RESIDUAL_CORRELATION_GROUPING_AMBIGUOUS`) and the confounding gate
//! (`E_RESIDUAL_CORRELATION_CONFOUNDED_KAPPA`).
//!
//! The fixture is the #1082 design note's `exact3` (`nonmem_anchor/nested_levels_3.*`):
//! 12 sponsors × 2 studies × 2 arms × 5 visits, `ARM` numbered 1, 2 inside each study and
//! `ARMID` numbering the same arms across the file. The NONMEM anchor of the composed
//! model is `tests/nested_levels_correlation_nonmem_anchor.rs`.

use crate::api::validation::check_model_data;
use crate::build_info::GradientMethodKind;
use crate::diagnostics::Diagnostic;
use crate::parser::model_parser::parse_full_model;
use crate::types::{
    CompiledModel, EstimationMethod, FitOptions, IovOccasionRule, Population, Subject,
};
use nalgebra::{DMatrix, DVector};

const EXACT3: &str = include_str!("../../../nonmem_anchor/nested_levels_3.ferx");
const DATA: &str = "nonmem_anchor/nested_levels_3.csv";

/// `exact3` with `extra` appended to `[parameters]` and `line` under the error model.
fn exact3_src(extra: &str, line: &str) -> String {
    EXACT3
        .replace(
            "  sigma ADD ~ 0.255164\n",
            &format!("  sigma ADD ~ 0.255164\n{extra}"),
        )
        .replace(
            "  DV ~ additive(ADD)\n",
            &format!("  DV ~ additive(ADD)\n{line}"),
        )
}

/// `exact3` with a free θ `RHO` at `rho` and `correlation <kind>(RHO) by <by>`.
fn with_corr(kind: &str, rho: f64, by: &str) -> String {
    // `cs` over a ten-row study needs ρ > −1/9; `car1` needs ρ ≥ 0.
    let lo = match kind {
        "cs" => -0.1,
        "car1" => 0.0,
        _ => -0.9,
    };
    exact3_src(
        &format!("  theta RHO({rho}, {lo}, 0.9)\n"),
        &format!("  correlation {kind}(RHO) by {by}\n"),
    )
}

fn load(src: &str) -> (CompiledModel, Population, FitOptions) {
    let parsed = parse_full_model(src).expect("model parses");
    let (pop, _) = crate::api::read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        DATA,
        None,
        parsed.fit_options.iov_column.as_deref(),
        None,
        &parsed.column_map,
    )
    .expect("data loads");
    (parsed.model, pop, parsed.fit_options)
}

/// The population as `fit()` prepares it: level cells, then residual blocks.
fn prepared(model: &CompiledModel, pop: &Population) -> Population {
    crate::api::fitted_population(
        model,
        &IovOccasionRule::Column,
        false,
        std::borrow::Cow::Borrowed(pop),
        &mut Vec::new(),
    )
    .into_owned()
}

fn eval(
    model: &CompiledModel,
    pop: &Population,
    opts: &FitOptions,
    method: EstimationMethod,
) -> crate::types::FitResult {
    let opts = FitOptions {
        method,
        interaction: method == EstimationMethod::FoceI,
        outer_maxiter: 0,
        run_covariance_step: false,
        ..opts.clone()
    };
    crate::api::fit(model, pop, &model.default_params, &opts).expect("evaluates")
}

/// Central-difference `∂f/∂η` of the per-cell prediction at `(eta, kappas)`.
fn h_eta(model: &CompiledModel, s: &Subject, eta: &[f64], kappas: &[Vec<f64>]) -> DMatrix<f64> {
    let theta = &model.default_params.theta;
    let mut h = DMatrix::zeros(s.obs_times.len(), eta.len());
    for c in 0..eta.len() {
        let step = 1e-6 * (1.0 + eta[c].abs());
        let (mut p, mut m) = (eta.to_vec(), eta.to_vec());
        p[c] += step;
        m[c] -= step;
        let fp = crate::pk::predict_iov(model, s, theta, &p, kappas);
        let fm = crate::pk::predict_iov(model, s, theta, &m, kappas);
        for j in 0..s.obs_times.len() {
            h[(j, c)] = (fp[j] - fm[j]) / (2.0 * step);
        }
    }
    h
}

/// The subject's linearization at a non-degenerate point: `η = 0.05`, and a κ per group
/// that differs between groups (so each level's column is live and distinct).
fn linearize(model: &CompiledModel, s: &Subject) -> crate::stats::likelihood::IovLinearization {
    let layout = crate::stats::nested_levels::NestedLayout::of(model, s).expect("prepared");
    let b: Vec<f64> = (0..layout.n_compact())
        .map(|i| 0.03 * (i as f64) - 0.05)
        .collect();
    let eta = b[..model.n_eta].to_vec();
    let kappas = layout.expand(&b);
    let h = h_eta(model, s, &eta, &kappas);
    let p = &model.default_params;
    let kappas: Vec<DVector<f64>> = kappas.into_iter().map(DVector::from_vec).collect();
    crate::stats::likelihood::iov_linearization(
        model,
        s,
        &p.theta,
        &DVector::from_vec(eta),
        &h,
        &p.omega,
        &kappas,
        p.omega_iov.as_ref().expect("omega_iov"),
    )
    .expect("linearizes")
}

/// `½ (rᵀ R̃⁻¹ r + log|R̃|)`, the FOCE (Sheiner–Beal) marginal over `R̃`.
fn sb_marginal(r_tilde: &DMatrix<f64>, resid: &DVector<f64>) -> f64 {
    let chol = r_tilde.clone().cholesky().expect("R̃ is PD");
    let quad = resid.dot(&chol.solve(resid));
    let log_det: f64 = 2.0 * chol.l().diagonal().iter().map(|d| d.ln()).sum::<f64>();
    0.5 * (quad + log_det)
}

/// V3. Each level's term of `R̃ = H Σ_b Hᵀ + R`, rebuilt by hand from the model's
/// declared variances and the `ar1` closed form, and checked one at a time in the FOCE
/// marginal of an `exact3` subject (two studies × two arms × five visits).
///
/// The marginal the engine returns equals the hand marginal over all four terms; and for
/// each term `T`, it is **not** the hand marginal over the other three. So deleting one
/// term from the engine reddens the assertion named after that term, and only that one:
/// (a) the sponsor η block of `Σ_b`, (b) the STUDY κ blocks, (c) the ARM κ blocks, (d) the
/// off-diagonal of the serial `R` (a diagonal `R` in the IOV marginal).
#[test]
fn each_level_term_of_r_tilde_enters_the_iov_foce_marginal() {
    let (m, pop, _) = load(&with_corr("ar1", 0.5, "ARM"));
    let pop = prepared(&m, &pop);
    let s = &pop.subjects[0];
    let p = &m.default_params;
    let lin = linearize(&m, s);
    let engine = crate::stats::likelihood::foce_subject_nll_iov_linearized(
        &m,
        s,
        &p.theta,
        &p.sigma.values,
        m.residual_cov_at(p),
        false,
        &lin,
    );
    assert!(engine.is_finite() && engine.abs() < 1e10, "{engine}");

    // The compact columns of each level: every cell's index for that level's κ.
    let layout = crate::stats::nested_levels::NestedLayout::of(&m, s).unwrap();
    let cols_of = |k: usize| {
        let mut c: Vec<usize> = (0..layout.n_cells())
            .map(|g| layout.compact_index(g, k))
            .collect();
        c.sort_unstable();
        c.dedup();
        c
    };
    let omega_iov = &p.omega_iov.as_ref().unwrap().matrix;
    let n = s.obs_times.len();
    let h = &lin.h_full;
    let term = |cols: &[usize], var: f64| {
        let mut t = DMatrix::zeros(n, n);
        for &c in cols {
            t += var * h.column(c) * h.column(c).transpose();
        }
        t
    };
    assert_eq!(cols_of(0).len(), 2, "two studies");
    assert_eq!(cols_of(1).len(), 4, "four arms, nested by key");
    let t_eta = term(&[0], p.omega.matrix[(0, 0)]);
    let t_study = term(&cols_of(0), omega_iov[(0, 0)]);
    let t_arm = term(&cols_of(1), omega_iov[(1, 1)]);
    // ar1 within each (STUDY, ARM): σ² ρ^{lag in visits}; additive error, so R does not
    // depend on the prediction.
    let rho = p.theta[m.residual_serial.as_ref().unwrap().rho_theta];
    let var = m.residual_variance_at(1, 1.0, &p.sigma.values);
    let key = |j: usize| {
        let c = s.obs_cov(j);
        (c["STUDY"].to_bits(), c["ARM"].to_bits())
    };
    let mut r_off = DMatrix::zeros(n, n);
    for j in 0..n {
        for k in 0..n {
            if j != k && key(j) == key(k) {
                let lag = (0..n)
                    .filter(|&i| {
                        key(i) == key(j)
                            && s.obs_times[i] > s.obs_times[j].min(s.obs_times[k])
                            && s.obs_times[i] <= s.obs_times[j].max(s.obs_times[k])
                    })
                    .count();
                r_off[(j, k)] = var * rho.powi(lag as i32);
            }
        }
    }
    let r_diag = DMatrix::from_diagonal_element(n, n, var);
    let b_hat = &lin.b_hat;
    let f0 = DVector::from_vec(lin.ipreds.clone()) - h * b_hat;
    let resid = DVector::from_vec(s.observations.clone()) - f0;

    let all = &t_eta + &t_study + &t_arm + &r_diag + &r_off;
    let hand = sb_marginal(&all, &resid);
    eprintln!(
        "engine {engine:.12}  hand {hand:.12}  |Δ| {:.3e}",
        (engine - hand).abs()
    );
    // Measured |Δ| 1.4e-14 (macOS arm64 and Linux aarch64; the two sums associate
    // differently). Each term moves the marginal by 2.1 (η), 11.4 (STUDY), 33.7 (ARM), 4.6
    // (serial R) — so 1e-3 is three orders below the smallest term, and 1e-9 five above the
    // agreement.
    //
    // One named check per term, all evaluated before any assertion, then the list of the
    // terms the engine does not carry asserted empty: deleting one term from the engine
    // reports that term's name and no other (the equality is checked last, so it cannot
    // fire first and hide which term went missing).
    let missing: Vec<&str> = [
        ("(a) the sponsor η block of Σ_b", &t_eta),
        ("(b) the STUDY κ blocks of Σ_b", &t_study),
        ("(c) the ARM κ blocks of Σ_b", &t_arm),
        ("(d) the off-diagonal of the serial R", &r_off),
    ]
    .into_iter()
    .filter_map(|(name, t)| {
        let without = sb_marginal(&(&all - t), &resid);
        eprintln!(
            "without {name}: {without:.12}  |Δ| {:.3e}",
            (without - engine).abs()
        );
        ((without - engine).abs() <= 1e-3).then_some(name)
    })
    .collect();
    assert!(
        missing.is_empty(),
        "the IOV FOCE marginal does not carry {missing:?}: engine {engine} equals the \
         marginal without it"
    );
    assert!(
        (engine - hand).abs() < 1e-9,
        "engine {engine} vs hand {hand}"
    );
}

/// V1 + V2, and the branch the FOCEI marginal takes.
///
/// V1: at ρ = 0 the composed model is the nested model without the line. ρ is a **free**
/// θ at its initial value 0, not `RHO FIX 0` (which the parser drops, so the comparison
/// would be a model against itself). With `outer_maxiter = 0` both evaluate one point.
/// They are not bit-equal, and are not expected to be: the line moves the EBE search onto
/// the dense objective's exact gradient and FOCEI onto the dense-`R` marginal, whose sums
/// associate differently. The bound is measured.
///
/// V2: at ρ = 0.5 the OFV moves by whole units.
///
/// The discriminator, asserted per subject on the same models: the FOCEI IOV marginal is
/// `foce_subject_nll_interaction_dense_cov` (to the bit) when the line is present and the
/// diagonal `foce_subject_nll_interaction` when it is not — so V1 compares the dense
/// branch against the diagonal one, not the diagonal one twice.
#[test]
fn rho_zero_composes_with_the_nested_fit_and_rho_moves_it() {
    let (base, pop, opts) = load(&exact3_src("", ""));
    let (zero, _, _) = load(&with_corr("ar1", 0.0, "ARM"));
    let (half, _, _) = load(&with_corr("ar1", 0.5, "ARM"));
    assert!(
        zero.residual_serial.is_some(),
        "a free ρ at 0 keeps the line"
    );

    // The discriminator.
    for (m, dense) in [(&base, false), (&zero, true), (&half, true)] {
        let prepared = prepared(m, &pop);
        let p = &m.default_params;
        for s in prepared.subjects.iter().take(3) {
            let lin = linearize(m, s);
            let got = crate::stats::likelihood::foce_subject_nll_iov_linearized(
                m,
                s,
                &p.theta,
                &p.sigma.values,
                m.residual_cov_at(p),
                true,
                &lin,
            );
            let want = if dense {
                crate::stats::likelihood::foce_subject_nll_interaction_dense_cov(
                    s,
                    &lin.ipreds,
                    &lin.b_hat,
                    &lin.h_full,
                    &lin.sigma_b,
                    &p.sigma.values,
                    &m.error_spec,
                    m.residual_cov_theta(&p.theta),
                    &[],
                    None,
                    m.residual_error_eta,
                )
            } else {
                crate::stats::likelihood::foce_subject_nll_interaction(
                    s,
                    &lin.ipreds,
                    &lin.b_hat,
                    &lin.h_full,
                    &lin.sigma_b,
                    &p.sigma.values,
                    &m.error_spec,
                    m.bloq_in_force(),
                    &[],
                    None,
                    None,
                    None,
                )
            };
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "subject {}: dense = {dense}",
                s.id
            );
        }
    }

    let r_base = eval(&base, &pop, &opts, EstimationMethod::FoceI);
    let r_zero = eval(&zero, &pop, &opts, EstimationMethod::FoceI);
    let r_half = eval(&half, &pop, &opts, EstimationMethod::FoceI);
    let d_ofv = (r_zero.ofv - r_base.ofv).abs();
    let mut d_ebe = 0.0f64;
    for i in 0..r_base.subjects.len() {
        for (a, b) in r_base.subjects[i]
            .eta
            .iter()
            .zip(r_zero.subjects[i].eta.iter())
        {
            assert!(a.is_finite() && b.is_finite());
            d_ebe = d_ebe.max((a - b).abs());
        }
        for (ka, kb) in r_base.ebe_kappas[i].iter().zip(&r_zero.ebe_kappas[i]) {
            for (a, b) in ka.iter().zip(kb.iter()) {
                assert!(a.is_finite() && b.is_finite());
                d_ebe = d_ebe.max((a - b).abs());
            }
        }
    }
    let mut d_cwres = 0.0f64;
    for (sa, sb) in r_base.subjects.iter().zip(&r_zero.subjects) {
        for (a, b) in sa.cwres.iter().zip(&sb.cwres) {
            assert!(a.is_finite() && b.is_finite());
            d_cwres = d_cwres.max((a - b).abs());
        }
    }
    eprintln!(
        "ρ = 0 vs no line: |ΔOFV| {d_ofv:.3e}, worst |ΔEBE| {d_ebe:.3e}, worst |ΔCWRES| \
         {d_cwres:.3e}; ρ = 0.5 OFV {:.6} vs {:.6}",
        r_half.ofv, r_base.ofv
    );
    // Measured, Linux aarch64 (macOS arm64): |ΔOFV| 5.560e-9 (8.975e-9), worst |ΔEBE|
    // 1.908e-7 (1.492e-7), worst |ΔCWRES| 8.586e-10 (1.687e-9) — the FD inner gradient's
    // stopping point against the analytic one's. The bounds are ≥ 10× above the worse of
    // each pair. ρ = 0.5 moves the OFV by 44.6.
    assert!(r_zero.ofv.is_finite() && d_ofv < 1e-7, "ΔOFV {d_ofv}");
    assert!(d_ebe < 2e-6, "ΔEBE {d_ebe}");
    assert!(d_cwres < 2e-8, "ΔCWRES {d_cwres}");
    // V2.
    assert!(
        r_half.ofv.is_finite() && (r_half.ofv - r_base.ofv).abs() > 1.0,
        "ρ = 0.5 OFV {} vs {}",
        r_half.ofv,
        r_base.ofv
    );
    // The route the line takes (#2101): the same as without it — every subject's EBE
    // search and the outer gradient analytic, so nothing warns about finite differences.
    assert!(
        !r_zero
            .warnings
            .iter()
            .any(|w| w.contains("finite-difference")),
        "{:?}",
        r_zero.warnings
    );
    assert_eq!(
        r_zero.gradient_method_outer,
        GradientMethodKind::Analytic.as_str()
    );
    assert!(!r_base
        .warnings
        .iter()
        .any(|w| w.contains("finite-difference")));
    assert_eq!(
        r_base.gradient_method_outer,
        GradientMethodKind::Analytic.as_str()
    );
}

fn codes(d: &[Diagnostic], code: &str) -> Vec<String> {
    d.iter()
        .filter(|d| d.code == code)
        .map(|d| d.message.clone())
        .collect()
}

const CONFOUNDED: &str = "E_RESIDUAL_CORRELATION_CONFOUNDED_KAPPA";

/// V4, gate B: `cs` over the groups of a level that carries a κ is refused; every other
/// combination is accepted. One table over both sides of the gate, on an explicit nested
/// level and on the implicit one-level model (`iov_column`):
///
/// | correlation | κ at | refused |
/// |---|---|---|
/// | `cs by ARM` | ARM (multiplicative, exact3's) | yes |
/// | `cs by ARM` | ARM (additive, `E0 + K_ARM`) | yes |
/// | `cs by ARMID` (a column numbering the arms across the file) | ARM | yes — by partition |
/// | `ar1 by ARM`, `car1 by ARM` | ARM | no |
/// | `cs by STUDY` | ARM only | no |
/// | `cs by OCC` | the occasions (`iov_column = OCC`) | yes |
/// | `ar1 by OCC` | the occasions | no |
///
/// Mutations it dies on: the refusal deleted (every "yes" row); widened to `ar1`/`car1`
/// (their rows); a level matched by name instead of by partition (`cs by ARMID`, and `cs
/// by STUDY` with only an ARM κ, which a "level with a κ named in `by`" test gets wrong
/// the other way round once the STUDY κ is dropped).
#[test]
fn cs_over_a_kappa_level_is_refused_and_everything_else_is_not() {
    let check = |src: &str| {
        let (m, pop, _) = load(src);
        check_model_data(&m, &pop)
    };
    // Exact3 with K_ST dropped: the only κ is at ARM.
    let arm_only = |kind: &str, by: &str| {
        with_corr(kind, 0.3, by)
            .replace("  kappa K_ST ~ 0.0904362 at STUDY\n", "")
            .replace("ETA_SP + K_ST + K_ARM", "ETA_SP + K_ARM")
    };
    let additive = with_corr("cs", 0.3, "ARM")
        .replace("y = TVE0 + EMAX", "y = E0 + EMAX")
        .replace(
            "  EMAX = TVEMAX * exp(ETA_SP + K_ST + K_ARM)\n",
            "  EMAX = TVEMAX * exp(ETA_SP + K_ST)\n  E0 = TVE0 + 0.5 * K_ARM\n",
        );
    assert!(additive.contains("E0 = TVE0 + 0.5 * K_ARM"));
    let cells: Vec<(&str, String, bool)> = vec![
        (
            "cs by ARM, multiplicative κ at ARM",
            with_corr("cs", 0.3, "ARM"),
            true,
        ),
        ("cs by ARM, additive κ at ARM", additive, true),
        ("cs by ARMID, κ at ARM", with_corr("cs", 0.3, "ARMID"), true),
        ("ar1 by ARM, κ at ARM", with_corr("ar1", 0.3, "ARM"), false),
        (
            "car1 by ARM, κ at ARM",
            with_corr("car1", 0.3, "ARM"),
            false,
        ),
        ("cs by STUDY, κ only at ARM", arm_only("cs", "STUDY"), false),
        ("cs by ARM, κ only at ARM", arm_only("cs", "ARM"), true),
    ];
    for (label, src, refused) in &cells {
        let d = check(src);
        let hits = codes(&d, CONFOUNDED);
        assert_eq!(!hits.is_empty(), *refused, "{label}: {hits:?}");
        assert!(
            d.iter()
                .filter(|x| x.code != CONFOUNDED)
                .all(|x| x.severity != crate::diagnostics::Severity::Error),
            "{label}: {d:?}"
        );
    }

    // The message, one sentence at a time (each assertion dies when its sentence is cut).
    let msg = codes(&check(&with_corr("cs", 0.3, "ARM")), CONFOUNDED).remove(0);
    for sentence in [
        "`correlation cs(RHO) by ARM` correlates the rows in exactly the groups of the level \
         `ARM`, where the kappa K_ARM is drawn.",
        "A kappa gives every row of a group one shared random effect, which is the correlation \
         `cs` estimates, so ρ and the kappa variance are not identified together: the same \
         covariance for an additive kappa, nearly the same for a multiplicative one.",
        "Use `ar1` or `car1`, whose correlation decays with the lag, or drop the kappa.",
    ] {
        assert!(msg.contains(sentence), "missing {sentence:?} in {msg}");
    }
    let by_id = codes(&check(&with_corr("cs", 0.3, "ARMID")), CONFOUNDED).remove(0);
    assert!(by_id.contains("the groups of the level `ARM`"), "{by_id}");

    // The implicit level: one `iov_column`, both sides.
    let one_level = |kind: &str| {
        format!(
            "[parameters]\n  theta TVE0(10.0)\n  theta TVEMAX(20.0)\n  theta RHO(0.3, 0.0, 0.9)\n  \
             omega ETA_SP ~ 0.09\n  kappa K_OC ~ 0.04\n  sigma ADD ~ 0.25\n\
             [individual_parameters]\n  EMAX = TVEMAX * exp(ETA_SP + K_OC)\n\
             [structural_model]\n  y = TVE0 + EMAX * TIME / (TIME + 3)\n\
             [error_model]\n  DV ~ additive(ADD)\n  correlation {kind}(RHO) by ARMID\n\
             [fit_options]\n  iov_column = ARMID\n"
        )
    };
    let d = check(&one_level("cs"));
    let msg = codes(&d, CONFOUNDED);
    assert_eq!(msg.len(), 1, "{d:?}");
    assert!(
        msg[0].contains("in exactly the groups of the occasion grouping, where the kappa K_OC"),
        "{}",
        msg[0]
    );
    assert!(codes(&check(&one_level("ar1")), CONFOUNDED).is_empty());
}

const AMBIGUOUS: &str = "E_RESIDUAL_CORRELATION_GROUPING_AMBIGUOUS";

/// V5, the grouping vocabulary. `by ARM` naming the level groups by the level's groups,
/// nested by key: the same blocks as `by ARMID` and as a level-free model's `by (STUDY,
/// ARM)`, and **not** the blocks of a level-free `by ARM`, which reads the column within
/// `ID` and merges arm 1 of study 1 with arm 1 of study 2 (the straddle, asserted). The two
/// nested spellings then score the same bits.
///
/// Mutation: the level key replaced by the bare label (`path[l..=l]`) → the `by ARMID`
/// equality and the straddle both redden.
#[test]
fn a_level_name_in_by_groups_by_the_level_nested_by_key() {
    let groups = |src: &str| {
        let (m, pop, _) = load(src);
        prepared(&m, &pop)
            .subjects
            .iter()
            .map(|s| s.obs_rgroup.clone())
            .collect::<Vec<_>>()
    };
    let level = groups(&with_corr("ar1", 0.3, "ARM"));
    let arm_id = groups(&with_corr("ar1", 0.3, "ARMID"));
    // A level-free model (η only) reading the columns.
    let flat = |by: &str| {
        exact3_src(
            "  theta RHO(0.3, -0.9, 0.9)\n",
            &format!("  correlation ar1(RHO) by {by}\n"),
        )
        .replace(
            "  level STUDY by STUDY in ID\n  level ARM by ARM in STUDY\n",
            "",
        )
        .replace("  kappa K_ST ~ 0.0904362 at STUDY\n", "")
        .replace("  kappa K_ARM ~ 3.75478 at ARM weight = NARM\n", "")
        .replace("ETA_SP + K_ST + K_ARM", "ETA_SP")
    };
    let tuple = groups(&flat("STUDY, ARM"));
    let column = groups(&flat("ARM"));
    assert_eq!(level, arm_id, "by ARM (level) vs by ARMID");
    assert_eq!(
        level, tuple,
        "by ARM (level) vs a level-free by (STUDY, ARM)"
    );
    assert_ne!(
        level, column,
        "the straddle: the column within ID merges the arms"
    );
    assert!(level[0].iter().max() == Some(&4) && column[0].iter().max() == Some(&2));

    let (m_level, pop, opts) = load(&with_corr("ar1", 0.4, "ARM"));
    let (m_id, _, _) = load(&with_corr("ar1", 0.4, "ARMID"));
    let a = eval(&m_level, &pop, &opts, EstimationMethod::FoceI);
    let b = eval(&m_id, &pop, &opts, EstimationMethod::FoceI);
    assert_eq!(a.ofv.to_bits(), b.ofv.to_bits(), "{} vs {}", a.ofv, b.ofv);
}

/// V5, the three refused spellings and the accepted ones, in one table. Each refusal names
/// the level and says to write it.
///
/// Mutations: the ambiguity check deleted (every refused row); the `parent ≠ ID` clause
/// dropped (the level-in-`ID` row starts refusing); the tuple clause dropped (its row).
#[test]
fn ambiguous_groupings_are_refused_and_the_rest_accepted() {
    // Levels whose names differ from their columns, so each spelling can be written.
    let renamed = |by: &str| {
        with_corr("ar1", 0.3, by)
            .replace("level STUDY by STUDY in ID", "level ST by STUDY in ID")
            .replace("level ARM by ARM in STUDY", "level AR by ARM in ST")
            .replace("at STUDY", "at ST")
            .replace("at ARM", "at AR")
    };
    let check = |src: &str| {
        let (m, pop, _) = load(src);
        check_model_data(&m, &pop)
    };
    // Each refused cell lists every sentence of its message, so cutting any one sentence
    // reddens that cell (AGENTS.md: a message is code).
    let cells: Vec<(&str, String, &[&str])> = vec![
        (
            "a level in a tuple",
            with_corr("ar1", 0.3, "STUDY, ARM"),
            &[
                "`correlation ar1(RHO) by STUDY, ARM` names the levels `STUDY`, `ARM` in a list \
                 of columns.",
                "A level groups rows by its whole key path, its parent levels included, so it \
                 does not combine with other columns.",
                "Write `by ARM` alone.",
            ],
        ),
        (
            "a nested level's column under another name",
            renamed("ARM"),
            &[
                "`correlation ar1(RHO) by ARM`: `ARM` is the column of the level `AR`, which is \
                 nested in `ST`.",
                "As a column, `ARM` groups rows by its value within ID, so rows with the same \
                 value in two groups of `ST` share one block, while the level keeps them apart.",
                "Write `by AR` to group by the level.",
            ],
        ),
        (
            "a level name that is also another data column",
            renamed("ARMID")
                .replace("level AR by ARM in ST", "level ARMID by ARM in ST")
                .replace("at AR ", "at ARMID "),
            &[
                "`correlation ar1(RHO) by ARMID`: `ARMID` is the level `ARMID`, grouped by the \
                 column `ARM`, and also a data column of its own, so `by ARMID` could mean \
                 either grouping.",
                "Rename the data column or the level.",
            ],
        ),
        ("the level by its name", renamed("AR"), &[]),
        // `cs`, since `ar1` over a study ties its arms' visit times; no κ at the study.
        (
            "a level in ID by its column",
            renamed("STUDY")
                .replace("ar1(RHO)", "cs(RHO)")
                .replace("theta RHO(0.3, -0.9, 0.9)", "theta RHO(0.3, -0.05, 0.9)")
                .replace("  kappa K_ST ~ 0.0904362 at ST\n", "")
                .replace("ETA_SP + K_ST + K_ARM", "ETA_SP + K_ARM"),
            &[],
        ),
        (
            "a column no level reads",
            with_corr("ar1", 0.3, "ARMID"),
            &[],
        ),
    ];
    for (label, src, sentences) in &cells {
        let d = check(src);
        let hits = codes(&d, AMBIGUOUS);
        if sentences.is_empty() {
            assert!(
                d.iter()
                    .all(|x| x.severity != crate::diagnostics::Severity::Error),
                "{label}: {d:?}"
            );
            continue;
        }
        assert_eq!(hits.len(), 1, "{label}: {d:?}");
        for sentence in *sentences {
            assert!(
                hits[0].contains(sentence),
                "{label}: missing {sentence:?} in {}",
                hits[0]
            );
        }
    }
}

/// Finding 3 of review round 1 on #2102: gate B on a one-level model whose occasions an
/// `iov_occasion` rule derives (here `time(5)`). The rule labels the rows only when the
/// population is prepared, so the check applies it as the fit does: `cs by WIN`, a column
/// that groups the rows exactly as the derived windows do, is refused, and `ar1` is not.
/// Mutation: skip applying the rule (the occasions stay as read, empty) → the `cs` cell is
/// no longer refused.
#[test]
fn gate_b_applies_a_derived_occasion_rule() {
    let src = |kind: &str| {
        format!(
            "[parameters]\n  theta TVE0(10.0)\n  theta RHO(0.3, 0.0, 0.9)\n  \
             omega ETA ~ 0.09\n  kappa K ~ 0.04\n  sigma ADD ~ 0.25\n\
             [individual_parameters]\n  E0 = TVE0 * exp(ETA + K)\n\
             [structural_model]\n  y = E0\n[error_model]\n  DV ~ additive(ADD)\n  \
             correlation {kind}(RHO) by WIN\n[fit_options]\n  iov_occasion = time(5)\n"
        )
    };
    let times = [1.0, 2.0, 4.0, 8.0, 12.0];
    let win = |t: f64| {
        std::collections::HashMap::from([("WIN".to_string(), if t < 5.0 { 1.0 } else { 2.0 })])
    };
    let pop = Population {
        subjects: (1..=2)
            .map(|i| Subject {
                id: i.to_string(),
                obs_times: times.to_vec(),
                observations: vec![10.0; 5],
                obs_cmts: vec![1; 5],
                cens: vec![0; 5],
                covariates: win(1.0),
                obs_covariates: times.iter().map(|&t| win(t)).collect(),
                ..Default::default()
            })
            .collect(),
        covariate_names: vec!["WIN".into()],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };
    for (kind, refused) in [("cs", true), ("ar1", false)] {
        let parsed = parse_full_model(&src(kind)).expect("parses");
        let rule = parsed.fit_options.iov_occasion.clone();
        assert_ne!(rule, IovOccasionRule::Column, "the rule is derived");
        let d = crate::api::validation::check_model_data_rule(&parsed.model, &pop, &rule);
        let hits = codes(&d, CONFOUNDED);
        assert_eq!(!hits.is_empty(), refused, "{kind}: {d:?}");
        if refused {
            assert!(
                hits[0].contains(
                    "in exactly the groups of the occasion grouping, where the kappa K is drawn."
                ),
                "{}",
                hits[0]
            );
        }
    }
    // The warning half on the same derived rule: subject 2's `WIN` no longer follows the
    // windows, so 1 of 2 subjects matches. `fit()` runs the warnings walk on the population
    // it prepared (occasions already derived); `ferx check` reads the data as written, so
    // it must hand the walk the rule — given `column`, the occasions stay empty and nothing
    // matches (mutation W4').
    let mut partial = pop.clone();
    for c in partial.subjects[1].obs_covariates.iter_mut() {
        c.insert("WIN".to_string(), 1.0);
    }
    let parsed = parse_full_model(&src("cs")).unwrap();
    let r = crate::api::fit(
        &parsed.model,
        &partial,
        &parsed.model.default_params,
        &FitOptions {
            outer_maxiter: 0,
            run_covariance_step: false,
            ..parsed.fit_options.clone()
        },
    )
    .expect("a partial match fits");
    assert!(
        r.warnings.iter().any(|w| w.contains(
            "in 1 of 2 subjects the blocks are exactly the groups of the \
                                 occasion grouping"
        )),
        "{:?}",
        r.warnings
    );

    let dir = tempfile::tempdir().unwrap();
    let model_path = dir.path().join("m.ferx");
    let data_path = dir.path().join("d.csv");
    std::fs::write(&model_path, src("cs")).unwrap();
    let mut csv = String::from("ID,TIME,DV,WIN\n");
    for (id, s) in partial.subjects.iter().enumerate() {
        for (j, t) in s.obs_times.iter().enumerate() {
            csv += &format!("{},{t},10,{}\n", id + 1, s.obs_cov(j)["WIN"]);
        }
    }
    std::fs::write(&data_path, csv).unwrap();
    let report = crate::api::validation::validate_model_file(
        model_path.to_str().unwrap(),
        Some(data_path.to_str().unwrap()),
    );
    let warns = codes(
        &report.diagnostics,
        "W_RESIDUAL_CORRELATION_PARTLY_CONFOUNDED_KAPPA",
    );
    assert_eq!(warns.len(), 1, "{:?}", report.diagnostics);
    assert!(warns[0].contains("in 1 of 2 subjects"), "{}", warns[0]);

    // Finding 9: two kappas at the grouping are named in the plural.
    let two = src("cs")
        .replace(
            "  kappa K ~ 0.04\n",
            "  kappa K ~ 0.04\n  kappa K2 ~ 0.04\n",
        )
        .replace(
            "  theta TVE0(10.0)\n",
            "  theta TVE0(10.0)\n  theta TVS(1.0)\n",
        )
        .replace("  y = E0\n", "  y = E0 * S\n")
        .replace(
            "  E0 = TVE0 * exp(ETA + K)\n",
            "  E0 = TVE0 * exp(ETA + K)\n  S = TVS * exp(K2)\n",
        );
    let parsed = parse_full_model(&two).expect("parses");
    let d = crate::api::validation::check_model_data_rule(
        &parsed.model,
        &pop,
        &parsed.fit_options.iov_occasion,
    );
    let hits = codes(&d, CONFOUNDED);
    assert_eq!(hits.len(), 1, "{d:?}");
    assert!(
        hits[0].contains("where the kappas K, K2 are drawn."),
        "{}",
        hits[0]
    );
}

/// Finding 4 of review round 1 on #2102: a block a level-named `by` defines is named by its
/// key path in the data checks' messages (`E_RESIDUAL_CORRELATION_TIES` here), since the
/// level's value alone names a group in every parent. Mutation: drop the level arm of
/// `residual_block_key` → the message names the bare `ARM=1`.
#[test]
fn a_level_block_is_named_by_its_key_path() {
    let (m, mut pop, _) = load(&with_corr("ar1", 0.3, "ARM"));
    // Two rows of subject 1's first arm (study 1, arm 1) at one TIME.
    let s = &mut pop.subjects[0];
    s.obs_times[1] = s.obs_times[0];
    if !s.obs_raw_times.is_empty() {
        s.obs_raw_times[1] = s.obs_raw_times[0];
    }
    let d = check_model_data(&m, &pop);
    let ties = codes(&d, "E_RESIDUAL_CORRELATION_TIES");
    assert_eq!(ties.len(), 1, "{d:?}");
    assert!(
        ties[0].contains("in the block STUDY=1, ARM=1."),
        "{}",
        ties[0]
    );
}

/// The conditional IOV objective (`conditional_obs_nll_from_preds(.., iov = true)`, the
/// AGQ data term) takes the dense `R` as the non-IOV one does — it used to skip it by
/// design. Mutation: restore the `ConditionalIov` exclusion → the IOV value is the
/// diagonal one and parts from the non-IOV value.
#[test]
fn the_conditional_iov_data_term_takes_the_dense_r() {
    let (m, pop, _) = load(&with_corr("ar1", 0.5, "ARM"));
    let pop = prepared(&m, &pop);
    let s = &pop.subjects[0];
    let p = &m.default_params;
    let preds: Vec<f64> = s.observations.iter().map(|y| y - 0.3).collect();
    let at = |iov: bool| {
        crate::stats::likelihood::conditional_obs_nll_from_preds(
            &m,
            s,
            &preds,
            &p.theta,
            &p.sigma.values,
            m.residual_cov_theta(&p.theta),
            &[0.0],
            iov,
        )
    };
    let (iov, plain) = (at(true), at(false));
    assert_eq!(iov.to_bits(), plain.to_bits(), "{iov} vs {plain}");
    // And the dense value is not the diagonal one: ρ is live here.
    let mut diag = 0.0;
    for (&y, &f) in s.observations.iter().zip(&preds) {
        let v = m.residual_variance_at(1, f, &p.sigma.values);
        diag += 0.5 * (v.ln() + (y - f).powi(2) / v);
    }
    assert!((iov - diag).abs() > 1e-3, "{iov} vs diagonal {diag}");
}

/// V10: the CWRES a nested fit with a correlation reports whiten by the dense
/// `R̃ = H Σ_b Hᵀ + R`: on every subject they equal `compute_cwres_cov` at the fit's EBEs
/// with the line's `R`, and are not the ones a diagonal `R` gives. Mutation: hand the
/// post-fit CWRES a diagonal `R` → the first assertion reddens.
#[test]
fn the_reported_cwres_whiten_by_the_dense_r_tilde() {
    let (m, pop, opts) = load(&with_corr("ar1", 0.5, "ARM"));
    let r = eval(&m, &pop, &opts, EstimationMethod::FoceI);
    let pop = prepared(&m, &pop);
    let p = &m.default_params;
    let (mut worst, mut gap) = (0.0f64, f64::INFINITY);
    for (i, s) in pop.subjects.iter().enumerate() {
        let eta: Vec<f64> = r.subjects[i].eta.iter().copied().collect();
        let kappas: Vec<Vec<f64>> = r.ebe_kappas[i]
            .iter()
            .map(|k| k.as_slice().to_vec())
            .collect();
        let h = h_eta(&m, s, &eta, &kappas);
        let kv: Vec<DVector<f64>> = kappas.into_iter().map(DVector::from_vec).collect();
        let lin = crate::stats::likelihood::iov_linearization(
            &m,
            s,
            &p.theta,
            &DVector::from_vec(eta),
            &h,
            &p.omega,
            &kv,
            p.omega_iov.as_ref().unwrap(),
        )
        .unwrap();
        let cwres = |cov| {
            crate::stats::likelihood::compute_cwres_cov(
                s,
                &lin.ipreds,
                &lin.b_hat,
                &lin.h_full,
                &lin.sigma_b,
                &p.sigma.values,
                &m.error_spec,
                cov,
                None,
                None,
                None,
                None,
                m.bloq_in_force(),
            )
        };
        let dense = cwres(m.residual_cov_theta(&p.theta));
        let diag = cwres(crate::stats::residual_error::ResidualCov::block_sigma(&[]));
        let got = &r.subjects[i].cwres;
        for j in 0..got.len() {
            assert!(got[j].is_finite() && dense[j].is_finite() && diag[j].is_finite());
            worst = worst.max((got[j] - dense[j]).abs());
        }
        let d: f64 = got
            .iter()
            .zip(&diag)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        gap = gap.min(d);
    }
    eprintln!("worst |CWRES − dense| {worst:.3e}; smallest per-subject gap to diag {gap:.3e}");
    // The reported H comes from the fit's own Jacobian, this one from central
    // differences: measured 3.294e-9 on Linux aarch64 (1.669e-9 on macOS arm64), against a
    // smallest per-subject gap of 0.543 to the diagonal-R CWRES. 4e-8 is 12× above the
    // agreement; 0.05 is 10× below the gap.
    assert!(worst < 4e-8, "reported CWRES vs dense: {worst}");
    assert!(
        gap > 0.05,
        "a subject's CWRES are the diagonal-R ones: {gap}"
    );
}

/// Gate B's warning half, every side in one test. `cs by STUDY` next to a kappa only at
/// `ARM` matches in exactly the subjects whose studies each hold one arm. On exact3 none
/// do, so nothing is said. Relabel three sponsors' arms to one per study and it matches in
/// 3 of 12: `W_RESIDUAL_CORRELATION_PARTLY_CONFOUNDED_KAPPA`, not the refusal, also in
/// the fit's warnings. Relabel all twelve and it is the refusal, without the warning.
///
/// Mutations: the warning deleted, or raised beside the refusal, or counted over every
/// subject instead of the matching ones; each reddens its own cell. Every sentence of the
/// message is asserted.
#[test]
fn a_partial_kappa_partition_match_warns_and_a_full_one_refuses() {
    const WARN: &str = "W_RESIDUAL_CORRELATION_PARTLY_CONFOUNDED_KAPPA";
    let src = with_corr("cs", 0.3, "STUDY")
        .replace("  kappa K_ST ~ 0.0904362 at STUDY\n", "")
        .replace("ETA_SP + K_ST + K_ARM", "ETA_SP + K_ARM");
    let (m, pop, opts) = load(&src);
    // One arm per study in the first `n` sponsors.
    let single_armed = |n: usize| {
        let mut p = pop.clone();
        for s in p.subjects.iter_mut().take(n) {
            for c in s.obs_covariates.iter_mut() {
                c.insert("ARM".to_string(), 1.0);
                c.insert("NARM".to_string(), 50.0);
            }
            s.covariates.insert("ARM".to_string(), 1.0);
        }
        p
    };
    let at = |p: &Population| {
        let errors = codes(&check_model_data(&m, p), CONFOUNDED);
        let warns = codes(
            &crate::api::validation::check_model_data_warnings(&m, p, &m.default_params),
            WARN,
        );
        (errors, warns)
    };

    let (e, w) = at(&pop);
    assert!(
        e.is_empty() && w.is_empty(),
        "no sponsor matches: {e:?} {w:?}"
    );

    let partial = single_armed(3);
    let (e, w) = at(&partial);
    assert!(e.is_empty(), "3 of 12 is not the refusal: {e:?}");
    assert_eq!(w.len(), 1, "{w:?}");
    for sentence in [
        "`correlation cs(RHO) by STUDY`: in 3 of 12 subjects the blocks are exactly the \
         groups of the level `ARM`, where the kappa K_ARM is drawn.",
        "In those subjects `cs` and the kappa are the same covariance for an additive kappa, \
         nearly the same for a multiplicative one, so ρ and the kappa variance are separated \
         only by the other 9; expect wide standard errors on both.",
        "If the blocks should differ in every subject, check the `by` column.",
    ] {
        assert!(w[0].contains(sentence), "missing {sentence:?} in {}", w[0]);
    }
    // The fit says it too.
    let r = eval(&m, &partial, &opts, EstimationMethod::FoceI);
    assert!(
        r.warnings
            .iter()
            .any(|x| x.contains("in 3 of 12 subjects the blocks")),
        "{:?}",
        r.warnings
    );

    let (e, w) = at(&single_armed(12));
    assert_eq!(e.len(), 1, "12 of 12 is the refusal");
    assert!(w.is_empty(), "the refusal is not also a warning: {w:?}");

    // 5′ of review round 2: a subject whose partitions agree only trivially — one
    // observation, so every row its own class — carries no `cs` pair and is in neither
    // count. Mutation: count any equal partition as a match → the first cell warns "1 of
    // 12", the third refuses.
    let one_obs = |p: &Population, n: usize| {
        let mut p = p.clone();
        for s in p.subjects.iter_mut().take(n) {
            s.obs_times.truncate(1);
            s.observations.truncate(1);
            s.obs_cmts.truncate(1);
            s.cens.truncate(1);
            s.obs_covariates.truncate(1);
            if !s.obs_raw_times.is_empty() {
                s.obs_raw_times.truncate(1);
            }
        }
        p
    };
    let (e, w) = at(&one_obs(&pop, 1));
    assert!(
        e.is_empty() && w.is_empty(),
        "a one-observation sponsor matches nothing: {e:?} {w:?}"
    );
    // Among real matches it changes the count, not the verdict: 11 of 11 is still the
    // refusal.
    let (e, w) = at(&one_obs(&single_armed(12), 1));
    assert_eq!(e.len(), 1, "{e:?}");
    assert!(w.is_empty(), "{w:?}");
    // Every subject trivial: `cs` couples nothing, so there is nothing to refuse.
    let (e, w) = at(&one_obs(&pop, 12));
    assert!(
        e.is_empty() && w.is_empty(),
        "every sponsor trivial: {e:?} {w:?}"
    );
}
