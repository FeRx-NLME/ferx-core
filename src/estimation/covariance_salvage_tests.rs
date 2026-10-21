//! The #1514 salvage route and the FD covariance stencil's invariants, on the adjoint's analytic
//! R-matrix (#2093 review round 2, finding 7). Ported from the forward provider's
//! `sens_cov_hessian` tests, whose subjects of interest — `compute_covariance`'s `Partial`
//! assembly, the salvage note's gate (#1516/#1803), the stencil's per-subject sum rule and its
//! thread-count determinism (#703/#1935) — are all still here.
//!
//! A subject is declined through [`super::forced_cov_decline`] by id, so the salvage is reached
//! without changing any prediction: the forward provider's tests routed a subject onto the
//! event walk with an unused time-varying covariate, a shape the adjoint now serves.
//!
//! Also here (#2048): an ODE model past the forward provider's old 24-axis `θ + η` cap keeps
//! the analytic outer route and assembles the analytic R-matrix for every subject, measured
//! by the FD objectives the step evaluates.
//!
//! And (#1933): no `[scaling]` form declines the R-matrix on either engine, and every
//! remediable decline clause is checked by applying its remedy.
use crate::estimation::inner_optimizer::find_ebe;
use crate::estimation::parameterization::pack_params;
use crate::parser::model_parser::parse_model_string;
use crate::types::{CompiledModel, DoseEvent, ModelParameters, Subject};
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

const WARFARIN: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

fn warfarin_subject(model: &CompiledModel, theta: &[f64], times: &[f64]) -> Subject {
    let n = times.len();
    let mut subject = Subject {
        id: "1".to_string(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: times.to_vec(),
        observations: vec![0.0; n],
        obs_cmts: vec![1; n],
        covariates: HashMap::new(),
        cens: vec![0; n],
        occasions: vec![1; n],
        ..Subject::default()
    };
    let eta_ref = [0.12, -0.08, 0.2];
    let preds = crate::pk::compute_predictions_with_tv(model, &subject, theta, &eta_ref);
    subject.observations = preds.iter().map(|p| p * 0.85).collect();
    subject
}

/// The EBE to the inner search's tightest tolerance on the adjoint's exact inner gradient, so a
/// reconverged-FD reference is free of inner-solver noise.
fn precise_ebe(model: &CompiledModel, subject: &Subject, params: &ModelParameters) -> Vec<f64> {
    find_ebe(model, subject, params, 500, 1e-12, None, None, 0)
        .eta
        .iter()
        .copied()
        .collect()
}

/// `n` warfarin subjects whose observations differ (so no subject's term copies another's),
/// with ids `{tag}{k}` — test-unique, since [`super::forced_cov_decline`] is keyed by id.
fn salvage_population(
    model: &CompiledModel,
    params: &ModelParameters,
    n: usize,
    tag: &str,
) -> crate::types::Population {
    let times = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0];
    let subjects: Vec<Subject> = (0..n)
        .map(|k| {
            let mut s = warfarin_subject(model, &params.theta, &times);
            s.id = format!("{tag}{k}");
            for (j, y) in s.observations.iter_mut().enumerate() {
                *y *= 1.0 + 0.03 * k as f64 - 0.01 * (j % 3) as f64;
            }
            s
        })
        .collect();
    crate::types::Population {
        subjects,
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

/// Decline `subject`'s analytic covariance term (by id).
fn decline(subject: &Subject) {
    super::forced_cov_decline::decline(&subject.id);
}

fn iov_cov_fixture(occasions: usize, block: bool) -> (CompiledModel, Subject) {
    let mut source = WARFARIN
        .replace(
            "sigma PROP_ERR ~ 0.04",
            "kappa KAPPA_CL ~ 0.02\n  sigma PROP_ERR ~ 0.15",
        )
        .replace(
            "CL = TVCL * exp(ETA_CL)",
            "CL = TVCL * exp(ETA_CL + KAPPA_CL)",
        );
    if block {
        source = source
            .replace(
                "omega ETA_CL ~ 0.09\n  omega ETA_V  ~ 0.04",
                "block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04]",
            )
            .replace("omega ETA_KA ~ 0.30", "")
            .replace("KA = TVKA * exp(ETA_KA)", "KA = TVKA")
            .replace(
                "kappa KAPPA_CL ~ 0.02",
                "block_kappa (KAPPA_CL, KAPPA_V) = [0.02, 0.004, 0.03]",
            )
            .replace("V  = TVV  * exp(ETA_V)", "V  = TVV  * exp(ETA_V + KAPPA_V)");
    }
    let model = parse_model_string(&source).unwrap();
    let times: Vec<_> = (0..occasions)
        .flat_map(|k| [0.5, 2.0, 6.0, 12.0].map(|t| t + 24.0 * k as f64))
        .collect();
    let mut subject = warfarin_subject(&model, &model.default_params.theta, &times);
    subject.doses = (0..occasions)
        .map(|k| DoseEvent::new(24.0 * k as f64, 100.0, 1, 0.0, false, 0.0))
        .collect();
    subject.occasions = (0..occasions)
        .flat_map(|k| vec![(k + 1) as u32; 4])
        .collect();
    subject.dose_occasions = (1..=occasions).map(|k| k as u32).collect();
    let eta = vec![0.1; model.n_eta];
    let kappas: Vec<_> = (0..occasions)
        .map(|k| vec![if k % 2 == 0 { 0.12 } else { -0.08 }; model.n_kappa])
        .collect();
    let pred = crate::pk::predict_iov(&model, &subject, &model.default_params.theta, &eta, &kappas);
    subject.observations = pred
        .iter()
        .enumerate()
        .map(|(i, p)| p * (0.91 + 0.02 * (i % 3) as f64))
        .collect();
    (model, subject)
}

/// Standard errors from a packed covariance, as a user reads them.
fn se_of(cov: &DMatrix<f64>) -> Vec<f64> {
    (0..cov.nrows())
        .map(|i| cov[(i, i)].max(0.0).sqrt())
        .collect()
}

/// Worst relative difference between two SE vectors over the coordinates that carry one,
/// with finiteness asserted before the fold (`f64::max` discards `NaN`).
fn worst_relative_se_diff(a: &[f64], b: &[f64], label: &str) -> f64 {
    assert_eq!(a.len(), b.len(), "{label}: SE vectors must be comparable");
    assert!(
        a.iter().chain(b.iter()).all(|v| v.is_finite()),
        "{label}: every SE must be finite before any of them are compared\na = {a:?}\nb = {b:?}"
    );
    let mut worst = 0.0f64;
    let mut compared = 0usize;
    for (x, y) in a.iter().zip(b.iter()) {
        if *x > 1e-12 {
            compared += 1;
            worst = worst.max(((y - x) / x).abs());
        }
    }
    assert!(
        compared >= 3,
        "{label}: only {compared} coordinate(s) carried a comparable SE"
    );
    worst
}

/// End to end through `compute_covariance`: one subject in ten declining reproduces the
/// covariance the pure-analytic route gives on the same data.
///
/// The decline is created by a constant, unreferenced per-observation covariate, which
/// changes the subject's *route* and not its predictions — asserted, not assumed, by
/// requiring the two populations' covariance OFVs to agree at the same point. Without
/// that check the reference would be a different fit and the bound would be measuring
/// two things at once.
///
/// Both objectives that reach the salvage with a different marginal are run: the FOCEI
/// Almquist–Laplace one and the AGQ quadrature one (`n_agq = 3`). The salvage evaluates
/// its subset through `pop_nll_opts`, the same dispatcher the whole-population stencil
/// uses, so the AGQ marginal *should* dispatch identically — this is the fixture that
/// says it does.
#[test]
fn one_declining_subject_reproduces_the_pure_analytic_covariance() {
    use crate::estimation::covariance::{
        analytic_cov_assembly, compute_covariance, AnalyticCovAssembly, CovarianceStepResult,
    };
    use crate::types::{EstimationMethod, FitOptions};

    let model = parse_model_string(WARFARIN).expect("parse");
    let mut params = model.default_params.clone();
    params.theta = vec![0.2, 10.0, 1.5];

    let plain_pop = salvage_population(&model, &params, 10, "salv-one-");
    let mut tv_pop = plain_pop.clone();
    tv_pop.subjects[3].id = "salv-one-declined".into();
    decline(&tv_pop.subjects[3]);

    let x_hat = pack_params(&params);
    let eta_hats: Vec<DVector<f64>> = plain_pop
        .subjects
        .iter()
        .map(|s| DVector::from_vec(precise_ebe(&model, s, &params)))
        .collect();
    let h_mats = vec![DMatrix::zeros(model.n_eta, model.n_eta); plain_pop.subjects.len()];
    let kappas = vec![vec![]; plain_pop.subjects.len()];

    for (label, n_agq) in [("FOCEI", 1), ("AGQ-FOCEI", 3)] {
        let opts = FitOptions {
            method: EstimationMethod::FoceI,
            interaction: true,
            n_agq,
            analytic_cov_hessian: true,
            verbose: false,
            ..FitOptions::default()
        };

        // Premise 2: the two runs really are the two routes under test — a complete
        // analytic assembly on one population and a one-subject salvage on the other.
        assert!(
            matches!(
                analytic_cov_assembly(&model, &plain_pop, &params, &x_hat, &eta_hats, &[], &opts),
                AnalyticCovAssembly::Full(_)
            ),
            "{label}: the reference must be the pure analytic route"
        );
        match analytic_cov_assembly(&model, &tv_pop, &params, &x_hat, &eta_hats, &[], &opts) {
            AnalyticCovAssembly::Partial { declined, .. } => {
                assert_eq!(declined, vec![3], "{label}: exactly subject 3 is salvaged")
            }
            _ => panic!("{label}: the salvage route must be taken, not Full or Unavailable"),
        }

        let run_with = |pop: &crate::types::Population, o: &FitOptions| match compute_covariance(
            &x_hat, &params, &model, pop, &eta_hats, &h_mats, &kappas, o,
        ) {
            CovarianceStepResult::Success(out) => out,
            other => panic!(
                "{label}: covariance step must succeed; got {}",
                match other {
                    CovarianceStepResult::Unusable(m) => m,
                    CovarianceStepResult::FailedNonPd { reason, .. } => reason,
                    _ => unreachable!(),
                }
            ),
        };
        let reference = run_with(&plain_pop, &opts);
        let hybrid = run_with(&tv_pop, &opts);

        // The yardstick, computed rather than quoted: the same population run through the
        // *whole-population* FD stencil. ferx already ships that and the pure analytic
        // R-matrix as interchangeable estimators of one matrix, so the gap between them is
        // the difference a user is already expected to tolerate — and routing one subject
        // in ten through the FD estimator *for its own term* has to be a fraction of it.
        let population_fd = run_with(
            &plain_pop,
            &FitOptions {
                analytic_cov_hessian: false,
                ..opts.clone()
            },
        );
        let reference_se = se_of(&reference.matrix);
        let estimator_gap = worst_relative_se_diff(
            &reference_se,
            &se_of(&population_fd.matrix),
            &format!("{label} (analytic vs population FD)"),
        );

        // Re-measured on the adjoint's R-matrix 2026-10-19 (Linux): 3.695e-4 (FOCEI), 2.597e-4
        // (AGQ-FOCEI), against estimator gaps of 3.363e-3 and 2.510e-3 — the forward provider's
        // numbers to three figures. The original measurement:
        // Realised worst relative SE difference, measured 2026-09-21: 3.699e-4 (FOCEI),
        // 2.599e-4 (AGQ-FOCEI), against estimator gaps of 3.364e-3 and 2.510e-3 — the
        // salvage costs 11.00 % and 10.36 % of swapping the whole population's estimator,
        // which is what one subject in ten should cost. Both bounds are asserted: the
        // absolute one so a regression that inflates the error while also inflating the
        // FD gap cannot hide behind the ratio, and the relative one so the number stays
        // anchored to something with a meaning rather than to a round constant.
        let worst = worst_relative_se_diff(&reference_se, &se_of(&hybrid.matrix), label);
        eprintln!("SALVAGE {label}: salvage Δ {worst:.3e}, analytic-vs-FD Δ {estimator_gap:.3e}");
        assert!(
            worst < 1e-3,
            "{label}: salvaging one subject in ten must reproduce the pure-analytic \
             standard errors; worst relative Δ = {worst:.3e}"
        );
        assert!(
            estimator_gap > 0.0 && worst < 0.25 * estimator_gap,
            "{label}: salvaging one subject of ten must cost a fraction of swapping the \
             whole population's estimator; salvage Δ = {worst:.3e}, analytic-vs-FD \
             Δ = {estimator_gap:.3e}"
        );

        // The informational note is emitted on the salvage route and on neither pure one.
        let note = |o: &crate::estimation::covariance::CovarianceOutput| {
            o.warnings
                .iter()
                .any(|w| w.contains("W_COV_ANALYTIC_SALVAGE"))
        };
        assert!(note(&hybrid), "{label}: the salvage must be reported");
        assert!(
            !note(&reference),
            "{label}: a fully analytic run must say nothing about salvaging"
        );
        let msg = hybrid
            .warnings
            .iter()
            .find(|w| w.contains("W_COV_ANALYTIC_SALVAGE"))
            .unwrap();
        assert!(
            msg.contains("1 of 10 subjects") && msg.contains("salv-one-declined"),
            "{label}: the note must name the count and the subject: {msg}"
        );
    }
}

/// The premise the whole salvage rests on: the population's covariance stencil **is** the
/// sum of the per-subject stencils. Only if that holds is a declining subject's term
/// something that can be computed separately without changing any other subject's.
///
/// Run through the production pieces — `subset_population`, `reconverge_population`,
/// `fd_ofv_stencil` — and not a re-derivation of the difference formulas, so this compares
/// one formula on two inputs rather than two copies of a formula.
///
/// Four objectives, because `pop_nll_opts` dispatches and the salvage rides that same
/// dispatcher: FOCE's Sheiner–Beal marginal, FOCEI's Almquist–Laplace one, the AGQ
/// quadrature marginal, and the IOV joint (η, κ) marginal. If any of them were not a
/// per-subject sum, a declined subject's stencil would not be its own term.
#[test]
fn the_population_cov_stencil_is_the_sum_of_the_per_subject_stencils() {
    use crate::estimation::covariance::{fd_ofv_stencil, reconverge_population, subset_population};
    use crate::estimation::outer_optimizer::pop_nll_opts;
    use crate::estimation::parameterization::pack_with_bounds;
    use crate::types::{EstimationMethod, FitOptions, Population};

    let plain = parse_model_string(WARFARIN).expect("parse");
    let mut plain_params = plain.default_params.clone();
    plain_params.theta = vec![0.2, 10.0, 1.5];
    let plain_pop = salvage_population(&plain, &plain_params, 4, "salv-sum-");

    // IOV: three subjects with different occasion counts, so the κ block is genuinely
    // ragged across subjects and a per-occasion bookkeeping error cannot cancel.
    let (iov_model, s3) = iov_cov_fixture(3, false);
    let (_, mut s2) = iov_cov_fixture(2, false);
    s2.id = "two-occasions".into();
    let (_, mut s1) = iov_cov_fixture(1, false);
    s1.id = "one-occasion".into();
    let iov_params = iov_model.default_params.clone();
    let iov_pop = Population {
        subjects: vec![s1, s2, s3],
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };

    let cases: [(
        &str,
        &CompiledModel,
        &Population,
        &ModelParameters,
        FitOptions,
    ); 4] = [
        (
            "FOCE",
            &plain,
            &plain_pop,
            &plain_params,
            FitOptions {
                method: EstimationMethod::Foce,
                interaction: false,
                ..FitOptions::default()
            },
        ),
        (
            "FOCEI",
            &plain,
            &plain_pop,
            &plain_params,
            FitOptions {
                method: EstimationMethod::FoceI,
                interaction: true,
                ..FitOptions::default()
            },
        ),
        (
            "AGQ-FOCEI",
            &plain,
            &plain_pop,
            &plain_params,
            FitOptions {
                method: EstimationMethod::FoceI,
                interaction: true,
                n_agq: 3,
                ..FitOptions::default()
            },
        ),
        (
            "IOV-FOCEI",
            &iov_model,
            &iov_pop,
            &iov_params,
            FitOptions {
                method: EstimationMethod::FoceI,
                interaction: true,
                ..FitOptions::default()
            },
        ),
    ];

    for (label, model, pop, params, opts) in cases {
        let x_hat = pack_params(params);
        let n = x_hat.len();
        let etas: Vec<DVector<f64>> = pop
            .subjects
            .iter()
            .map(|s| {
                DVector::from_iterator(
                    model.n_eta,
                    find_ebe(model, s, params, 200, 1e-10, None, None, 0)
                        .eta
                        .iter()
                        .copied(),
                )
            })
            .collect();
        let free_idx: Vec<usize> = {
            let packed = pack_with_bounds(params);
            (0..n).filter(|&i| !packed.fixed[i]).collect()
        };
        let tol = opts.effective_cov_inner_tol(model.uses_closed_form_ltbs_inner());

        let stencil_over = |idx: &[usize]| -> DMatrix<f64> {
            let (sub, warm) = subset_population(pop, &etas, idx);
            let ofv = |xv: &[f64]| -> f64 {
                let (p, e, h, k) =
                    reconverge_population(xv, model, &sub, params, &warm, &opts, tol, false);
                2.0 * pop_nll_opts(model, &sub, &p, &e, &h, &k, &opts)
            };
            let f0 = ofv(&x_hat);
            assert!(
                f0.is_finite(),
                "{label}: subset {idx:?} has a non-finite base OFV"
            );
            fd_ofv_stencil(
                n,
                &x_hat,
                &free_idx,
                &crate::estimation::parameterization::fd_step_floors(params, &model.default_params),
                opts.fd_hessian_step,
                f0,
                None,
                &ofv,
                &opts,
            )
            .expect("nothing cancels in a test")
            .hess
        };

        let all: Vec<usize> = (0..pop.subjects.len()).collect();
        let whole = stencil_over(&all);
        let per_subject: Vec<DMatrix<f64>> = all.iter().map(|&i| stencil_over(&[i])).collect();
        let mut summed = DMatrix::<f64>::zeros(n, n);
        for h in &per_subject {
            summed += h;
        }
        assert!(
            whole.iter().chain(summed.iter()).all(|v| v.is_finite()),
            "{label}: both sides must be finite before they are differenced"
        );

        // Realised worst entry difference, measured 2026-09-21: 3.461e-11 (FOCE),
        // 1.799e-11 (FOCEI), 3.597e-11 (AGQ-FOCEI), 3.461e-11 (IOV-FOCEI), against matrix
        // scales of 6.8e1–1.5e2 — i.e. at the ULP of the entries themselves. The bound is
        // four orders above the worst of those. It is not a statement about the
        // derivative's accuracy (both sides are the *same* stencil), only about the
        // reassociation of a floating-point sum.
        const SUM_RULE_TOL: f64 = 1e-6;
        let worst = (&whole - &summed).amax();
        assert!(
            worst < SUM_RULE_TOL,
            "{label}: the population stencil must equal the sum of the per-subject \
             stencils; worst entry Δ = {worst:.3e}"
        );

        // Two mutations, each reaching something the other cannot.
        //
        // (a) Drop subject 0 from the summed side. If the fixture's subjects contributed
        //     nothing, the agreement above would hold for any implementation at all, so
        //     this has to move the comparison by orders of magnitude. Realised
        //     2026-09-21: 3.209e1 (FOCE), 3.752e1 (FOCEI), 3.709e1 (AGQ-FOCEI), 1.904e1
        //     (IOV-FOCEI) — nine orders over `SUM_RULE_TOL`.
        //
        // (b) Re-run the *population* stencil over subjects 1.. and require it to equal
        //     the sum over subjects 1.. to the same tight bound. This is the leg that
        //     kills a `stencil_over` which ignored `idx` and always evaluated the whole
        //     population: such an implementation passes (a) — it would still differ from
        //     the mutilated sum — and returns `whole` here, failing by exactly the margin
        //     (a) measures. Checking only `stencil_over(&all[1..]) != summed` would be the
        //     same assertion as (a) numerically (both reduce to ±subject 0's term), which
        //     is a redundant gate rather than a second one.
        let sum_without_0 = &summed - &per_subject[0];
        let contribution = (&whole - &sum_without_0).amax();
        assert!(
            contribution.is_finite() && contribution > 1e3 * SUM_RULE_TOL,
            "{label}: subject 0 must contribute something, else the sum rule above is a \
             tautology; got {contribution:.3e}"
        );
        let whole_without_0 = stencil_over(&all[1..]);
        assert!(
            whole_without_0.iter().all(|v| v.is_finite()),
            "{label}: the subset stencil must be finite before it is differenced"
        );
        let subset_worst = (&whole_without_0 - &sum_without_0).amax();
        assert!(
            subset_worst < SUM_RULE_TOL,
            "{label}: the sum rule must hold on a proper subset too — a stencil that \
             ignored its subject list would fail here by ~{contribution:.3e}; \
             got {subset_worst:.3e}"
        );
    }
}

/// The full-decline short-circuit. When every subject is outside the analytic scope there
/// is no analytic majority to carry the salvage's cost, so the whole-population stencil is
/// taken — the same matrix, one route fewer, and no note claiming a hybrid that did not
/// happen. Measured on clofarabine (56/56 declining): the salvage rebuilt it as 56
/// per-subject stencils for +4 % wall and cells identical to 7 significant figures.
#[test]
fn a_fully_declining_population_takes_the_population_stencil() {
    use crate::estimation::covariance::{
        analytic_cov_assembly, compute_covariance, AnalyticCovAssembly, CovarianceStepResult,
    };
    use crate::types::{EstimationMethod, FitOptions};

    let model = parse_model_string(WARFARIN).expect("parse");
    let mut params = model.default_params.clone();
    params.theta = vec![0.2, 10.0, 1.5];
    let pop = salvage_population(&model, &params, 4, "salv-all-");
    for s in &pop.subjects {
        decline(s);
    }
    let x_hat = pack_params(&params);
    let eta_hats: Vec<DVector<f64>> = pop
        .subjects
        .iter()
        .map(|s| DVector::from_vec(precise_ebe(&model, s, &params)))
        .collect();
    let h_mats = vec![DMatrix::zeros(model.n_eta, model.n_eta); pop.subjects.len()];
    let kappas = vec![vec![]; pop.subjects.len()];
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        interaction: true,
        analytic_cov_hessian: true,
        verbose: false,
        ..FitOptions::default()
    };

    assert!(
        matches!(
            analytic_cov_assembly(&model, &pop, &params, &x_hat, &eta_hats, &[], &opts),
            AnalyticCovAssembly::Unavailable
        ),
        "a fully declining population must short-circuit, not return a Partial with every \
         subject salvaged"
    );

    let run = |o: &FitOptions| match compute_covariance(
        &x_hat, &params, &model, &pop, &eta_hats, &h_mats, &kappas, o,
    ) {
        CovarianceStepResult::Success(out) => out,
        other => panic!(
            "covariance step must succeed; got {}",
            match other {
                CovarianceStepResult::Unusable(m) => m,
                CovarianceStepResult::FailedNonPd { reason, .. } => reason,
                _ => unreachable!(),
            }
        ),
    };
    let with_analytic = run(&opts);
    let forced_fd = run(&FitOptions {
        analytic_cov_hessian: false,
        ..opts.clone()
    });

    // Bit-identical, not merely close: the short-circuit is supposed to take the *same*
    // route `analytic_cov_hessian = false` takes, so any difference at all means it took
    // some third thing.
    assert_eq!(
        with_analytic.matrix, forced_fd.matrix,
        "the short-circuit must be the population stencil, bit for bit"
    );
    assert!(
        !with_analytic
            .warnings
            .iter()
            .any(|w| w.contains("W_COV_ANALYTIC_SALVAGE")),
        "nothing was salvaged, so nothing may be reported as salvaged: {:?}",
        with_analytic.warnings
    );
}

/// The salvage note describes the numbers on the page, so it is gated on the estimator
/// that produced them (#1516 review §1).
///
/// Under `covariance_method = s` the returned covariance is `S⁻¹` from the score
/// cross-product alone and no `R` is built at all (#1803) — hybrid or not — so nothing
/// about the salvage route reaches the reported standard errors, and the note's closing
/// claim ("only the named subjects' terms use a different estimator") would be a statement
/// about a matrix this step never assembled. `rsr` is the other side of that gate and must keep the note: it
/// returns `R⁻¹ S R⁻¹`, so the salvaged terms are in the numbers. Asserted in one test, so
/// a gate stuck on either branch reddens rather than passing half.
#[test]
fn the_salvage_note_is_gated_on_the_estimator_that_uses_the_r_matrix() {
    use crate::estimation::covariance::{
        analytic_cov_assembly, compute_covariance, AnalyticCovAssembly, CovarianceStepResult,
    };
    use crate::types::{CovarianceMethod, EstimationMethod, FitOptions};

    let model = parse_model_string(WARFARIN).expect("parse");
    let mut params = model.default_params.clone();
    params.theta = vec![0.2, 10.0, 1.5];

    let mut pop = salvage_population(&model, &params, 30, "salv-note-");
    // `salvage_population` scales every subject's curve by one factor, which makes the
    // per-subject scores collinear and `S = Σᵢ gᵢgᵢᵀ` rank-deficient — the cross-product
    // estimator then refuses outright and the arm under test never runs. A deterministic
    // per-(subject, observation) wobble breaks that collinearity without touching the
    // scope gate, which reads route and structure, not values.
    for (k, s) in pop.subjects.iter_mut().enumerate() {
        for (j, y) in s.observations.iter_mut().enumerate() {
            *y *= 1.0 + 0.12 * (1.7 * k as f64 + 0.9 * j as f64).sin();
        }
    }
    decline(&pop.subjects[3]);

    let x_hat = pack_params(&params);
    let eta_hats: Vec<DVector<f64>> = pop
        .subjects
        .iter()
        .map(|s| DVector::from_vec(precise_ebe(&model, s, &params)))
        .collect();
    // Real `a = ∂f/∂η` matrices (`n_obs × n_eta`), not the zero placeholders the
    // Hessian-only fixtures above can get away with: `s` and `rsr` route through
    // `assemble_score_cross_product`, which reads `h_matrix.row(j)` for every observation.
    let h_mats: Vec<DMatrix<f64>> = pop
        .subjects
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let eta = eta_hats[i].as_slice().to_vec();
            let n_obs = s.observations.len();
            let mut a = DMatrix::zeros(n_obs, model.n_eta);
            for m in 0..model.n_eta {
                let step = 1e-5;
                let (mut up, mut down) = (eta.clone(), eta.clone());
                up[m] += step;
                down[m] -= step;
                let f_up = crate::pk::compute_predictions_with_tv(&model, s, &params.theta, &up);
                let f_down =
                    crate::pk::compute_predictions_with_tv(&model, s, &params.theta, &down);
                for j in 0..n_obs {
                    a[(j, m)] = (f_up[j] - f_down[j]) / (2.0 * step);
                }
            }
            a
        })
        .collect();
    let kappas = vec![vec![]; pop.subjects.len()];

    let base = FitOptions {
        method: EstimationMethod::FoceI,
        interaction: true,
        analytic_cov_hessian: true,
        verbose: false,
        ..FitOptions::default()
    };

    // Premise: the hybrid route is actually taken. Without this the "silent" arm below is
    // satisfied by a population that never salvaged anything.
    match analytic_cov_assembly(&model, &pop, &params, &x_hat, &eta_hats, &[], &base) {
        AnalyticCovAssembly::Partial { declined, .. } => {
            assert_eq!(declined, vec![3], "exactly subject 3 is salvaged")
        }
        _ => panic!("the salvage route must be taken, not Full or Unavailable"),
    }

    let run = |method: CovarianceMethod| {
        let opts = FitOptions {
            covariance_method: method,
            ..base.clone()
        };
        match compute_covariance(
            &x_hat, &params, &model, &pop, &eta_hats, &h_mats, &kappas, &opts,
        ) {
            CovarianceStepResult::Success(out) => out,
            other => panic!(
                "{method:?}: covariance step must succeed; got {}",
                match other {
                    CovarianceStepResult::Unusable(m) => m,
                    CovarianceStepResult::FailedNonPd { reason, .. } => reason,
                    _ => unreachable!(),
                }
            ),
        }
    };
    let note = |o: &crate::estimation::covariance::CovarianceOutput| {
        o.warnings
            .iter()
            .any(|w| w.contains("W_COV_ANALYTIC_SALVAGE"))
    };

    for method in [CovarianceMethod::Hessian, CovarianceMethod::Sandwich] {
        let out = run(method);
        assert!(
            note(&out),
            "{method:?} reports R⁻¹, so the salvaged terms are in the numbers and the note \
             must be emitted: {:?}",
            out.warnings
        );
    }
    let cross = run(CovarianceMethod::CrossProduct);
    assert!(
        !note(&cross),
        "covariance_method = s returns S⁻¹ and discards R, so nothing may claim the \
         salvage moved these standard errors: {:?}",
        cross.warnings
    );
}

/// #1935 moved three things off one thread — the base reconvergence at `x̂`, the step
/// probe's passes, and the salvage route's subset base — and made the stencil take the
/// probe's diagonal instead of recomputing it. None of that may move a number: `find_ebe`
/// is deterministic per subject and `collect()` keeps subject order, so the FD covariance
/// has to be bit-identical at 1 and at 4 threads (#703's reproducibility guarantee).
///
/// One cell per route that reaches the changed code: the whole-population stencil on a
/// non-IOV, an AGQ and an IOV model (`analytic_cov_hessian = false` forces the stencil),
/// and the #1514 salvage route, whose subset base `f0_fd` is the third parallelised call.
/// Mixture fits are not covered: their base point goes through `mixture_ofv`, which this
/// PR does not touch, and only their probe pass became parallel. Each run
/// asserts its route, so a fixture that drifted onto the analytic path cannot pass by
/// never reaching the stencil.
#[test]
fn the_fd_covariance_is_bit_identical_across_thread_counts() {
    use crate::estimation::covariance::{
        analytic_cov_assembly, compute_covariance, AnalyticCovAssembly, CovarianceStepResult,
    };
    use crate::types::{CovarianceMethod, EstimationMethod, FitOptions, Population};

    let plain = parse_model_string(WARFARIN).expect("parse");
    let mut plain_params = plain.default_params.clone();
    plain_params.theta = vec![0.2, 10.0, 1.5];
    let plain_pop = salvage_population(&plain, &plain_params, 6, "salv-threads-");
    let mut salvage_pop = plain_pop.clone();
    salvage_pop.subjects[2].id = "salv-threads-declined".into();
    decline(&salvage_pop.subjects[2]);

    let (iov_model, s3) = iov_cov_fixture(3, false);
    let (_, mut s2) = iov_cov_fixture(2, false);
    s2.id = "two-occasions".into();
    let iov_params = iov_model.default_params.clone();
    let iov_pop = Population {
        subjects: vec![s2, s3],
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };

    let opts = |analytic: bool| FitOptions {
        method: EstimationMethod::FoceI,
        interaction: true,
        analytic_cov_hessian: analytic,
        covariance_method: CovarianceMethod::Hessian,
        verbose: false,
        ..FitOptions::default()
    };
    let cases: [(
        &str,
        &CompiledModel,
        &Population,
        &ModelParameters,
        FitOptions,
    ); 4] = [
        (
            "non-IOV stencil",
            &plain,
            &plain_pop,
            &plain_params,
            opts(false),
        ),
        (
            // AGQ scores through `agq_population_nll`, whose subject `par_iter` is not
            // size-gated, so this is the cell where the probe's point loop always nests a
            // second parallel loop.
            "AGQ stencil",
            &plain,
            &plain_pop,
            &plain_params,
            FitOptions {
                n_agq: 3,
                ..opts(false)
            },
        ),
        (
            "IOV stencil",
            &iov_model,
            &iov_pop,
            &iov_params,
            opts(false),
        ),
        (
            "salvage route",
            &plain,
            &salvage_pop,
            &plain_params,
            opts(true),
        ),
    ];

    for (label, model, pop, params, o) in cases {
        let x_hat = pack_params(params);
        let etas: Vec<DVector<f64>> = pop
            .subjects
            .iter()
            .map(|s| {
                DVector::from_iterator(
                    model.n_eta,
                    find_ebe(model, s, params, 200, 1e-10, None, None, 0)
                        .eta
                        .iter()
                        .copied(),
                )
            })
            .collect();
        let n_subj = pop.subjects.len();
        let h_mats = vec![DMatrix::zeros(model.n_eta, model.n_eta); n_subj];
        let kappas = vec![vec![]; n_subj];

        // The route under test, asserted rather than assumed.
        if o.analytic_cov_hessian {
            match analytic_cov_assembly(model, pop, params, &x_hat, &etas, &[], &o) {
                AnalyticCovAssembly::Partial { declined, .. } => {
                    assert_eq!(declined, vec![2], "{label}: exactly subject 2 is salvaged")
                }
                _ => panic!("{label}: the salvage route must be taken"),
            }
        }

        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    match compute_covariance(
                        &x_hat, params, model, pop, &etas, &h_mats, &kappas, &o,
                    ) {
                        CovarianceStepResult::Success(out) => out.matrix,
                        CovarianceStepResult::Unusable(m) => panic!("{label}: {m}"),
                        CovarianceStepResult::FailedNonPd { reason, .. } => {
                            panic!("{label}: {reason}")
                        }
                    }
                })
        };
        let one = run(1);
        let four = run(4);
        assert!(
            one.iter().all(|v| v.is_finite()) && one.amax() > 0.0,
            "{label}: the matrix must be finite and live before it is compared:\n{one}"
        );
        assert_eq!(one, four, "{label}: 1 and 4 threads must agree bit for bit");
    }
}

/// #1935 at the production call sites, not just in `fd_ofv_hessian`: the population
/// objectives one `compute_covariance` runs, and which of them re-solve their subjects in
/// parallel, read back from the `cfg(test)` log that `reconverge_population` writes.
///
/// Whole-population stencil: one parallel base reconvergence at `x̂`, then `2·n_free²`
/// serial points — the probe's pass and the off-diagonal stencil, the diagonal not
/// evaluated a second time. Salvage route (#1514): the whole-population base (parallel),
/// the declined subjects' own base `f0_fd` (parallel), then `2·n_free²` serial points over
/// those subjects alone.
///
/// The log is written by `map_subjects` from the branch it takes, and
/// `map_subjects_runs_concurrently_exactly_when_asked` witnesses that each branch runs the
/// way it says, so together they pin the actual dispatch, not just the requested flag.
///
/// Each production edit this exists to catch moves a count here: calling `select_fd_step`
/// + `fd_ofv_stencil(…, None, …)` instead of `fd_ofv_hessian` adds `2·n_free` serial
/// points; calling `fd_ofv_stencil` at `initial_eps` with no probe at all leaves the counts
/// alone but empties the `fd_ofv_hessian` entry log; flipping the base point's or `f0_fd`'s `parallel_subjects` to `false` (the
/// pre-#1935 behaviour) moves its entry to the serial column; flipping the stencil's to
/// `true` reintroduces #256's per-point fork/join and empties the serial column.
#[test]
fn the_covariance_step_runs_one_parallel_base_and_2nf2_serial_points() {
    use crate::estimation::covariance::{compute_covariance, reconverge_log, CovarianceStepResult};
    use crate::estimation::parameterization::pack_with_bounds;
    use crate::types::{CovarianceMethod, EstimationMethod, FitOptions, Population};

    let model = parse_model_string(WARFARIN).expect("parse");
    let mut params = model.default_params.clone();
    params.theta = vec![0.2, 10.0, 1.5];
    let x_hat = pack_params(&params);
    let nf = {
        let packed = pack_with_bounds(&params);
        (0..x_hat.len()).filter(|&i| !packed.fixed[i]).count()
    };
    assert!(
        nf >= 3,
        "the fixture must have a real off-diagonal block: nf = {nf}"
    );

    // Unique id prefixes: the log is process-global and tests run concurrently.
    let tagged =
        |n: usize, tag: &str| -> Population { salvage_population(&model, &params, n, tag) };
    let whole_pop = tagged(4, "t1935-whole-");
    let mut salvage_pop = tagged(6, "t1935-salvage-");
    // Two declined subjects (of six, under the salvage cut of half), so the subset base
    // point `f0_fd` has more than one subject to re-solve in parallel. The subset starts
    // with subject 2's id, so its calls carry a prefix the whole-population calls (first
    // id `…-0`) do not share.
    for (k, tag) in [(2, "a"), (4, "b")] {
        salvage_pop.subjects[k].id = format!("t1935-salvage-declined-{tag}");
        decline(&salvage_pop.subjects[k]);
    }

    // Each route gets its own `fd_hessian_step`, a hair off the `1e-2` default and used by
    // no other test, so the `fd_ofv_hessian` entry log can be read back per route: the
    // no-halving counts below cannot tell the probe-then-stencil route from a stencil run
    // straight at `initial_eps` (same points, same matrix), but the entry log can.
    const WHOLE_STEP: f64 = 0.010_193_5;
    const SALVAGE_STEP: f64 = 0.010_193_7;
    let opts = |analytic: bool| FitOptions {
        method: EstimationMethod::FoceI,
        interaction: true,
        analytic_cov_hessian: analytic,
        covariance_method: CovarianceMethod::Hessian,
        fd_hessian_step: if analytic { SALVAGE_STEP } else { WHOLE_STEP },
        verbose: false,
        ..FitOptions::default()
    };
    let run = |pop: &Population, o: &FitOptions| {
        let etas: Vec<DVector<f64>> = pop
            .subjects
            .iter()
            .map(|s| DVector::from_vec(precise_ebe(&model, s, &params)))
            .collect();
        let n = pop.subjects.len();
        let h = vec![DMatrix::zeros(model.n_eta, model.n_eta); n];
        match compute_covariance(&x_hat, &params, &model, pop, &etas, &h, &vec![vec![]; n], o) {
            CovarianceStepResult::Success(_) => {}
            CovarianceStepResult::Unusable(m) => panic!("{m}"),
            CovarianceStepResult::FailedNonPd { reason, .. } => panic!("{reason}"),
        }
    };
    let tally = |calls: &[(usize, bool)], n: usize, parallel: bool| {
        calls.iter().filter(|&&c| c == (n, parallel)).count()
    };

    run(&whole_pop, &opts(false));
    assert_eq!(
        reconverge_log::take_fd_hessian(WHOLE_STEP),
        1,
        "whole: the step must be sized by the probe, through `fd_ofv_hessian`"
    );
    let calls = reconverge_log::take("t1935-whole-");
    assert_eq!(
        tally(&calls, 4, true),
        1,
        "whole: one parallel base point: {calls:?}"
    );
    assert_eq!(
        tally(&calls, 4, false),
        2 * nf * nf,
        "whole: 2·n_free² serial stencil + probe points (n_free = {nf})"
    );
    assert_eq!(
        calls.len(),
        1 + 2 * nf * nf,
        "whole: nothing else: {calls:?}"
    );

    run(&salvage_pop, &opts(true));
    assert_eq!(
        reconverge_log::take_fd_hessian(SALVAGE_STEP),
        1,
        "salvage: the step must be sized by the probe, through `fd_ofv_hessian`"
    );
    let whole_calls = reconverge_log::take("t1935-salvage-0");
    let subset_calls = reconverge_log::take("t1935-salvage-declined");
    assert_eq!(
        whole_calls,
        vec![(6, true)],
        "salvage: the only whole-population objective is the parallel base point"
    );
    assert_eq!(
        tally(&subset_calls, 2, true),
        1,
        "salvage: one parallel f0_fd over both declined subjects: {subset_calls:?}"
    );
    assert_eq!(
        tally(&subset_calls, 2, false),
        2 * nf * nf,
        "salvage: 2·n_free² serial points over the declined subjects (n_free = {nf})"
    );
    assert_eq!(
        subset_calls.len(),
        1 + 2 * nf * nf,
        "salvage: {subset_calls:?}"
    );
}

/// The premise the off-diagonal warning describes (#1514 review §1): `fd_ofv_stencil`
/// never *stores* a non-finite result, so that entry keeps whatever the caller already had
/// there — the zero initialisation on the whole-population route, the in-scope subjects'
/// analytic cross-partial on the hybrid one. "Cross-partial correlation set to 0" is
/// therefore true on the first and false on the second, which is why the message is now
/// built from the route.
///
/// Driven through the stencil directly with a synthetic objective, because no end-to-end
/// fixture overflows a *cross-partial* without also overflowing a diagonal — and a
/// non-finite diagonal is caught earlier and reported differently (`select_fd_step` halves
/// first, then `problem_params` makes it a hard `Unusable`). The objective here is
/// separately quadratic in each coordinate and `NaN` only when both are perturbed at once,
/// which is exactly that cell and nothing else.
#[test]
fn a_non_finite_cross_partial_is_recorded_and_leaves_its_entry_untouched() {
    use crate::estimation::covariance::fd_ofv_stencil;
    use crate::types::FitOptions;

    let ofv = |xv: &[f64]| -> f64 {
        if xv[0] != 0.0 && xv[1] != 0.0 {
            f64::NAN
        } else {
            xv[0] * xv[0] + 2.0 * xv[1] * xv[1]
        }
    };
    let x = [0.0, 0.0];
    let opts = FitOptions::default();
    let stencil = fd_ofv_stencil(2, &x, &[0, 1], &[1.0, 1.0], 1e-2, 0.0, None, &ofv, &opts)
        .expect("nothing cancels in a test");

    // The diagonals are unaffected — the fixture must fail the cross-partial *only*, or it
    // would be testing the `Unusable` path instead of this one.
    assert!(
        stencil.diag_nan.is_empty(),
        "the fixture must keep both diagonals finite: {:?}",
        stencil.diag_nan
    );
    assert!(
        (stencil.hess[(0, 0)] - 2.0).abs() < 1e-6,
        "{}",
        stencil.hess[(0, 0)]
    );
    assert!(
        (stencil.hess[(1, 1)] - 4.0).abs() < 1e-6,
        "{}",
        stencil.hess[(1, 1)]
    );

    // Both coordinates are named, and the entry is left at its initial value rather than
    // at some `NaN` that would poison the whole inverse.
    let named: std::collections::HashSet<usize> = [0, 1].into_iter().collect();
    assert_eq!(stencil.offdiag_nan, named);
    assert_eq!(stencil.hess[(0, 1)], 0.0);
    assert_eq!(stencil.hess[(1, 0)], 0.0);
}

// ── #2048: an ODE model past the retired forward provider's 24-axis `θ + η` cap ──────────

/// Runs `f` with [`super::FD_OBJECTIVE_EVALS`] reset, returning its result and the objectives
/// the covariance step's FD probe and stencil evaluated meanwhile.
fn fd_objective_evals_during<T>(f: impl FnOnce() -> T) -> (T, usize) {
    super::FD_OBJECTIVE_EVALS.with(|c| c.set(0));
    let out = f();
    (out, super::FD_OBJECTIVE_EVALS.with(|c| c.get()))
}

/// #2048's fixture: a one-state ODE model with 2 θ + 2 η plus `n_fix` FIX θ, each a
/// factor of exactly 1 on CL, so the likelihood does not depend on `n_fix` while the
/// `θ + η` count the retired forward provider capped at 24 does.
fn padded_ode_model(n_fix: usize) -> crate::types::CompiledModel {
    let thetas: String = (0..n_fix)
        .map(|k| format!("  theta TH_{k}(0.0, -3.0, 3.0) FIX\n"))
        .collect();
    let factors: String = (0..n_fix).map(|k| format!(" * (1 + TH_{k})")).collect();
    crate::parser::model_parser::parse_model_string(&format!(
        "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  theta TVV(40.0, 1.0, 500.0)\n\
         {thetas}  omega ETA_CL ~ 0.15\n  omega ETA_V ~ 0.15\n  sigma PROP_ERR ~ 0.04\n\
         [individual_parameters]\n  CL = TVCL{factors} * exp(ETA_CL)\n\
         V = TVV * exp(ETA_V)\n\
         [structural_model]\n  ode(obs_cmt=central, states=[central])\n\
         [odes]\n  d/dt(central) = -(CL/V) * central\n\
         [error_model]\n  DV ~ proportional(PROP_ERR)\n"
    ))
    .unwrap()
}

/// #2048: an ODE model past the retired forward provider's 24-axis `θ + η` cap keeps every
/// analytic route the fit reports. Before #2093 the 25th axis (here a FIX θ, which counts)
/// dropped the outer gradient to FD, sent `auto` to BOBYQA and the covariance step to the
/// stencil, with nothing in the fit saying why. The adjoint has no θ cap, so the three
/// predicates the fit routes on must read the same at 24 and past it — asserted on both
/// sides of the old cap so a reinstated cap reddens the far side by name, and on a
/// `gradient = fd` control so the predicates are shown able to decline at all.
#[test]
fn an_ode_model_past_the_old_axis_cap_keeps_the_analytic_outer_and_r_routes() {
    use crate::estimation::cov_diagnostics::CovScopeDecline;
    use crate::sens::provider::{
        analytic_outer_gradient_for_interaction, analytic_outer_gradient_in_scope,
    };
    use crate::types::{FitOptions, GradientMethod, Optimizer, Population, Subject};
    let model = padded_ode_model;
    let population = Population {
        subjects: vec![Subject::default()],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let options = FitOptions::default();
    for n_fix in [20, 21, 40] {
        let m = model(n_fix);
        let axes = m.n_theta + m.n_eta;
        assert_eq!(axes, 4 + n_fix, "the FIX θ are counted as θ");
        assert!(m.ode_spec.is_some(), "{axes} axes: an ODE model");
        assert!(
            analytic_outer_gradient_in_scope(&m),
            "{axes} axes: the outer gradient must stay analytic"
        );
        for interaction in [true, false] {
            assert_eq!(
                Optimizer::Auto.resolve_auto(&m, interaction),
                Optimizer::NloptLbfgs,
                "{axes} axes, interaction = {interaction}: `auto` must take L-BFGS"
            );
        }
        assert_eq!(
            super::analytic_cov_declines(&m, &population, &options, false),
            vec![CovScopeDecline::PerSubjectBail],
            "{axes} axes: no model-level clause may decline the analytic R-matrix"
        );
        assert!(super::analytic_r_expected(&m, &population, &options, false));

        let mut fd = model(n_fix);
        fd.gradient_method = GradientMethod::Fd;
        assert!(
            !analytic_outer_gradient_for_interaction(&fd, true),
            "{axes} axes, FD"
        );
        assert_eq!(
            Optimizer::Auto.resolve_auto(&fd, true),
            Optimizer::Bobyqa,
            "{axes} axes, FD"
        );
        assert!(!super::analytic_r_expected(
            &fd,
            &population,
            &options,
            false
        ));
    }
}

/// #2048, the covariance step as it **runs**, not as it is predicted. `analytic_r_expected`
/// above is a model-level walk; a cap reinstated inside the profile-Hessian recorder
/// (`subject_adjoint_cov_hessian` returning `None` per subject) would leave it green while
/// every subject fell to the stencil — the issue's 50× covariance step. So on real
/// subjects at 24 and 25 axes: the assembly returns `Full` (no subject declined), and the
/// whole step evaluates **zero** FD objectives. The controls run the same step with the
/// analytic R-matrix switched off and with `gradient = fd`, and must pay the full `2·n²`
/// stencil, so the zero is a count that can move.
#[test]
fn an_ode_model_past_the_old_axis_cap_assembles_r_analytically_for_every_subject() {
    use crate::types::{FitOptions, GradientMethod};
    let mut population = crate::io::datareader::read_nonmem_csv(
        std::path::Path::new("data/two_cpt_oral_cov.csv"),
        None,
        None,
    )
    .expect("two_cpt_oral_cov.csv loads");
    population.subjects.truncate(6);
    let options = FitOptions::default();
    let off = FitOptions {
        analytic_cov_hessian: false,
        ..FitOptions::default()
    };
    for n_fix in [20, 21] {
        let model = padded_ode_model(n_fix);
        let axes = model.n_theta + model.n_eta;
        let p = model.default_params.clone();
        let x = crate::estimation::parameterization::pack_params(&p);
        let (etas, hs): (Vec<_>, Vec<_>) = population
            .subjects
            .iter()
            .map(|s| {
                let e = crate::estimation::inner_optimizer::find_ebe_for_stage(
                    &model, s, &p, 200, 1e-8, None, None, 0, &options,
                );
                (e.eta, e.h_matrix)
            })
            .unzip();
        let kappas = vec![Vec::new(); population.subjects.len()];
        let n_free = model.free_packed_dim();

        let assembly =
            super::analytic_cov_assembly(&model, &population, &p, &x, &etas, &kappas, &options);
        assert!(
            matches!(assembly, super::AnalyticCovAssembly::Full(_)),
            "{axes} axes: every subject must take the analytic R-matrix"
        );
        let step = |m: &crate::types::CompiledModel, o: &FitOptions| {
            fd_objective_evals_during(|| {
                super::compute_covariance(&x, &p, m, &population, &etas, &hs, &kappas, o)
            })
        };
        let (out, evals) = step(&model, &options);
        assert!(
            matches!(out, super::CovarianceStepResult::Success(_)),
            "{axes} axes: analytic step"
        );
        assert_eq!(
            evals, 0,
            "{axes} axes: the analytic step differences no objective"
        );

        let (out, evals) = step(&model, &off);
        assert!(matches!(out, super::CovarianceStepResult::Success(_)));
        assert_eq!(
            evals,
            super::fd_r_objective_evals(n_free).unwrap(),
            "{axes} axes, analytic_cov_hessian = false: the stencil"
        );
        let mut fd = padded_ode_model(n_fix);
        fd.gradient_method = GradientMethod::Fd;
        assert!(matches!(
            super::analytic_cov_assembly(&fd, &population, &p, &x, &etas, &kappas, &options),
            super::AnalyticCovAssembly::Unavailable
        ));
        let (_, evals) = step(&fd, &options);
        assert_eq!(
            evals,
            super::fd_r_objective_evals(n_free).unwrap(),
            "{axes} axes, gradient = fd: the stencil"
        );
    }
}

/// #2054: the pre-fit cost prediction (`predicted_r_route`) follows the covariance step's own
/// route, salvage included, and the assembly agrees with it on the same population. Laplace on
/// the exact anchor serves at most 8 stacked `[η, κ]` axes, so on the 3-η / 1-κ IOV fixture a
/// subject with 6 occasions (9 axes) declines and one with 1 occasion (4 axes) does not —
/// the population's mix of the two is the only input that changes between cells:
///
/// - none wide → `Analytic`, no cost note, no count;
/// - 1 of 3 wide → `Salvage { 1, 3 }`, the `2·n²` count over that one subject — and
///   `analytic_cov_assembly` returns `Partial` declining exactly that subject;
/// - 1 of 2 wide → `Stencil`, the salvage cut is `>=` one half;
/// - the 1-of-3 population with `analytic_cov_hessian = false` → `Stencil`, the model-level
///   gate overrides the per-subject walk.
#[test]
fn the_pre_fit_cost_prediction_follows_the_salvage_route() {
    use super::{
        analytic_cov_assembly, pre_fit_covariance_cost, pre_fit_covariance_cost_note,
        predicted_r_route, AnalyticCovAssembly, PredictedRRoute,
    };
    use crate::types::{EstimationMethod, FitOptions, Population};
    let (model, narrow) = iov_cov_fixture(1, false);
    let (_, wide) = iov_cov_fixture(6, false);
    let subject = |s: &Subject, id: &str| Subject {
        id: id.into(),
        ..s.clone()
    };
    let pop = |subjects: Vec<Subject>| Population {
        subjects,
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let options = FitOptions {
        method: EstimationMethod::Laplace,
        run_covariance_step: true,
        ..FitOptions::default()
    };
    let none_wide = pop(vec![subject(&narrow, "n1"), subject(&narrow, "n2")]);
    let one_of_three = pop(vec![
        subject(&narrow, "n1"),
        subject(&narrow, "n2"),
        subject(&wide, "w1"),
    ]);
    let one_of_two = pop(vec![subject(&narrow, "n1"), subject(&wide, "w1")]);

    let route = |p: &Population, o: &FitOptions| predicted_r_route(&model, p, o, false);
    assert_eq!(route(&none_wide, &options), PredictedRRoute::Analytic);
    let salvage = PredictedRRoute::Salvage {
        declined: 1,
        n_subjects: 3,
    };
    assert_eq!(route(&one_of_three, &options), salvage);
    assert_eq!(route(&one_of_two, &options), PredictedRRoute::Stencil);
    let disabled = FitOptions {
        analytic_cov_hessian: false,
        ..options.clone()
    };
    assert_eq!(route(&one_of_three, &disabled), PredictedRRoute::Stencil);
    assert!(
        route(
            &one_of_three,
            &FitOptions {
                method: EstimationMethod::FoceI,
                interaction: true,
                ..options.clone()
            }
        ) == PredictedRRoute::Analytic,
        "control: under FOCEI's wider ladder the 9-axis subject is in scope"
    );

    // The cost and the note, at n = 31, on each route.
    assert_eq!(
        pre_fit_covariance_cost(31, &options, false, PredictedRRoute::Analytic),
        None
    );
    assert_eq!(
        pre_fit_covariance_cost(31, &options, false, salvage),
        Some(Some(1922))
    );
    assert_eq!(
        pre_fit_covariance_cost(31, &options, false, PredictedRRoute::Stencil),
        Some(Some(1922))
    );
    let note = pre_fit_covariance_cost_note(31, Some(Some(1922)), salvage).expect("a note");
    assert!(note.contains("over 1 of 3 subjects"), "{note}");

    // The assembly declines exactly the subject the prediction named.
    let params = model.default_params.clone();
    let x_hat = pack_params(&params);
    let eta_hats: Vec<DVector<f64>> = one_of_three
        .subjects
        .iter()
        .map(|_| DVector::from_element(model.n_eta, 0.1))
        .collect();
    let kappas: Vec<Vec<DVector<f64>>> = one_of_three
        .subjects
        .iter()
        .map(|s| {
            (0..crate::stats::stacked_layout::group_count(s))
                .map(|k| {
                    DVector::from_element(model.n_kappa, if k % 2 == 0 { 0.12 } else { -0.08 })
                })
                .collect()
        })
        .collect();
    match analytic_cov_assembly(
        &model,
        &one_of_three,
        &params,
        &x_hat,
        &eta_hats,
        &kappas,
        &options,
    ) {
        AnalyticCovAssembly::Partial { declined, .. } => assert_eq!(declined, vec![2]),
        AnalyticCovAssembly::Full(_) => panic!("expected Partial, got Full"),
        AnalyticCovAssembly::Unavailable => panic!("expected Partial, got Unavailable"),
    }
}

/// #1933: the warning once told a closed-form `obs_scale` user that rewriting it as
/// `[scaling] y = central / V` "moves the fit onto the analytic route", and on a `pk`
/// model the rewrite only traded the scaling clause for an analytic-readout one. Since
/// #2093 no `[scaling]` form declines the R-matrix on either engine, so the regression to
/// catch is a scaling- or readout-keyed clause coming back on one engine only. Every form
/// the issue tabulated, on both engines in one test, and each fixture checked to carry the
/// representation it is named for, so a parse that dropped `[scaling]` cannot pass as the
/// unscaled control.
#[test]
fn no_scaling_form_declines_the_r_matrix_on_either_engine() {
    use crate::types::{FitOptions, Population, ScalingSpec, Subject};
    let population = Population {
        subjects: vec![Subject::default()],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let options = FitOptions::default();
    // `examples/warfarin_iov.ferx`, the issue's closed-form fixture (κ on CL).
    let closed = |scaling: &str| {
        crate::parser::model_parser::parse_model_string(&format!(
            "[parameters]\n  theta TVCL(0.2, 0.001, 10.0)\n  theta TVV(10.0, 0.1, 500.0)\n\
             theta TVKA(1.5, 0.01, 50.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n\
             omega ETA_KA ~ 0.3\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n\
             [individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n\
             V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n\
             [structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n\
             [error_model]\n  DV ~ proportional(PROP_ERR)\n{scaling}"
        ))
        .unwrap()
    };
    // `examples/one_cpt_iv_ode.ferx`, the issue's `[odes]` control.
    let ode = |obs_cmt: &str, scaling: &str| {
        crate::parser::model_parser::parse_model_string(&format!(
            "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  theta TVV(40.0, 1.0, 500.0)\n\
             omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  sigma PROP_ERR ~ 0.02 (sd)\n\
             [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n\
             [structural_model]\n  ode({obs_cmt}states=[central])\n\
             [odes]\n  d/dt(central) = -(CL/V) * central\n\
             [scaling]\n{scaling}\n[error_model]\n  DV ~ proportional(PROP_ERR)\n"
        ))
        .unwrap()
    };

    let none = closed("");
    assert!(matches!(none.scaling, ScalingSpec::None) && none.analytic_readout.is_none());
    let mut cases = vec![("pk, no [scaling]", none)];
    for (label, body) in [
        ("pk, obs_scale = 1.0", "  obs_scale = 1.0"),
        ("pk, obs_scale = 0.5", "  obs_scale = 0.5"),
        ("pk, obs_scale = V", "  obs_scale = V"),
        ("pk, obs_scale = 1000 / V", "  obs_scale = 1000 / V"),
    ] {
        let m = closed(&format!("[scaling]\n{body}\n"));
        assert!(!matches!(m.scaling, ScalingSpec::None), "{label}");
        cases.push((label, m));
    }
    for (label, body) in [
        ("pk, y = central / V", "  y = central / V"),
        ("pk, y = central / V * 2", "  y = central / V * 2"),
    ] {
        let m = closed(&format!("[scaling]\n{body}\n"));
        assert!(
            m.analytic_readout.is_some(),
            "{label}: Form-C on a pk model"
        );
        cases.push((label, m));
    }
    let ode_y = ode("", "  y = central / V");
    assert!(matches!(
        ode_y.ode_spec.as_ref().unwrap().readout,
        crate::ode::predictions::OdeReadout::Single(_)
    ));
    assert!(ode_y.analytic_readout.is_none());
    cases.push(("odes, y = central / V", ode_y));
    let ode_os = ode("obs_cmt=central, ", "  obs_scale = V");
    assert!(!matches!(ode_os.scaling, ScalingSpec::None));
    cases.push(("odes, obs_scale = V", ode_os));

    for (label, model) in &cases {
        assert_eq!(
            super::analytic_cov_scope_declines(model, &population, &options, false),
            Vec::new(),
            "{label}"
        );
        assert!(
            super::analytic_r_expected(model, &population, &options, false),
            "{label}"
        );
    }
}

/// #1933 proposed fix (2), in its general form: a remedy is checked by **applying** it,
/// not by reading it. For every clause whose `remedy()` is `Some`, a model that hits only
/// that clause declines with exactly it, and performing the action leaves the walk empty —
/// so a remedy whose action introduces a clause of its own (the closed-form `obs_scale` →
/// `y = …` rewrite trading one decline for another) reddens here. A new remediable clause
/// without an arm below fails the `unreachable!`.
#[test]
fn applying_each_remedy_to_a_model_that_hits_only_its_clause_clears_the_walk() {
    use crate::estimation::cov_diagnostics::CovScopeDecline;
    use crate::types::{FitOptions, GradientMethod, Population, Subject};
    let population = Population {
        subjects: vec![Subject::default()],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let base = || {
        crate::parser::model_parser::parse_model_string(
            "[parameters]\n  theta TVCL(0.2, 0.001, 10.0)\n  theta TVV(10.0, 0.1, 500.0)\n\
             omega ETA_CL ~ 0.09\n  sigma PROP_ERR ~ 0.04\n[individual_parameters]\n\
             CL = TVCL * exp(ETA_CL)\n  V = TVV\n[structural_model]\n\
             pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n",
        )
        .unwrap()
    };
    let mut remediable = 0;
    for decline in CovScopeDecline::ALL {
        if decline.remedy().is_none() {
            continue;
        }
        remediable += 1;
        let mut model = base();
        let mut options = FitOptions::default();
        match decline {
            CovScopeDecline::Disabled => options.analytic_cov_hessian = false,
            CovScopeDecline::GradientFd => model.gradient_method = GradientMethod::Fd,
            other => unreachable!("{other:?} has a remedy but no fixture applying it"),
        }
        assert_eq!(
            super::analytic_cov_scope_declines(&model, &population, &options, false),
            vec![decline],
            "{decline:?}: the fixture hits only its own clause"
        );
        match decline {
            CovScopeDecline::Disabled => options.analytic_cov_hessian = true,
            CovScopeDecline::GradientFd => model.gradient_method = GradientMethod::Auto,
            _ => unreachable!(),
        }
        assert_eq!(
            super::analytic_cov_scope_declines(&model, &population, &options, false),
            Vec::new(),
            "{decline:?}: {} must clear the walk",
            decline.remedy().unwrap()
        );
    }
    assert_eq!(remediable, 2, "a remediable clause was added or lost");
}
