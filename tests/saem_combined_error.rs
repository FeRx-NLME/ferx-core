//! Slow regression tests for issues #267 and #1445: SAEM must not collapse the
//! additive component of a combined residual-error model.
//!
//! ## Two fixtures, because the first one cannot fail the way #1445 does
//!
//! [`MODEL`] (#267) is 1-cpt IV, 32 subjects × 8 dense samples, truth
//! `combined(prop = 0.05, add = 3.0)` on predictions that run from 3.33 down to
//! 0.09. The additive SD is **larger than every prediction in the dataset**, so
//! it carries essentially the whole residual variance at every time point and is
//! pinned by all 256 observations at once. That fixture can catch a σ_add that
//! is dropped on the floor by a coding error; it cannot catch #1445, because
//! #1445 is a property of a *minority* variance component — one identified only
//! by the part of the curve where `σ_add` and `σ_prop·f` are comparable — whose
//! single-draw M-step maximiser has a boundary-heavy sampling distribution. Give
//! σ_add the whole variance and that distribution has no boundary mass to sit
//! on, so the assertion below it (SAEM within 35% of FOCEI) passed green through
//! the entire history of the bug.
//!
//! [`SPARSE_MODEL`] (#1445) is the regime the report is actually about, and is
//! shaped on `ferx-testdata/cefepime_jordan`: 2-cpt IV infusion, block Ω(3), 300
//! subjects with a **median of one observation each** (1–3, trough-heavy), truth
//! `combined(prop = 0.13, add = 1.8)` on troughs around 15–120. There the
//! additive term is a minority component, sparse per-subject data lets the
//! frozen η track each observation, and — before #1445 — the reported σ_add was
//! simply whichever value the last M-step's maximiser happened to draw.
//!
//! What changed about the assertion: the #267 test compares SAEM to FOCEI on a
//! fixture where neither can move, and asserts a *relative* bound. The #1445
//! test asserts against the **simulation truth** on a fixture where the
//! per-M-step maximiser is measured to swing over two orders of magnitude, and
//! states its distance from the σ floor explicitly, so "collapsed" is a number
//! rather than a comparison against another estimator that might collapse too.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::{DoseEvent, OmegaMatrix, Population};
use ferx_core::{fit, simulate_with_seed, EstimationMethod, FitOptions};

mod common;

const MODEL: &str = r#"
[parameters]
  theta TVCL(3.0, 0.1, 20.0)
  theta TVV(30.0, 1.0, 200.0)
  omega ETA_CL ~ 0.08
  omega ETA_V  ~ 0.08
  sigma PROP_ERR ~ 0.05 (sd)
  sigma ADD_ERR  ~ 2.00 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)
"#;

fn template_population(n: usize) -> Population {
    let times = [0.5_f64, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 36.0];
    let subjects = (1..=n)
        .map(|i| {
            common::subject(
                &format!("{i}"),
                vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                times.to_vec(),
                vec![0.0; times.len()],
                vec![1; times.len()],
            )
        })
        .collect();

    Population {
        subjects,
        covariate_names: vec![],
        dv_column: "dv".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

fn simulated_population(model: &ferx_core::types::CompiledModel) -> Population {
    let template = template_population(32);
    let mut truth = model.default_params.clone();
    truth.theta = vec![3.0, 30.0];
    truth.omega = OmegaMatrix::from_diagonal(&[0.08, 0.08], vec!["ETA_CL".into(), "ETA_V".into()]);
    truth.sigma.values = vec![0.05, 3.00];

    let sim = simulate_with_seed(model, &template, &truth, 1, 20260619);
    let mut pop = template;
    for subj in pop.subjects.iter_mut() {
        subj.observations = sim
            .iter()
            .filter(|row| row.id == subj.id)
            .map(|row| row.outcome.continuous_value())
            .collect();
    }
    pop
}

fn fit_with(
    method: EstimationMethod,
    model: &ferx_core::types::CompiledModel,
    pop: &Population,
) -> f64 {
    let mut opts = FitOptions::default();
    opts.method = method;
    opts.run_covariance_step = false;
    opts.verbose = false;
    opts.outer_maxiter = 300;
    opts.saem_n_exploration = 80;
    opts.saem_n_convergence = 80;
    opts.saem_seed = Some(267);

    let result = fit(model, pop, &model.default_params, &opts).expect("fit must succeed");
    result.sigma[1]
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn saem_combined_error_additive_sigma_matches_focei() {
    let model = parse_model_string(MODEL).expect("combined model parses");
    let pop = simulated_population(&model);

    let focei_add = fit_with(EstimationMethod::FoceI, &model, &pop);
    let saem_add = fit_with(EstimationMethod::Saem, &model, &pop);

    assert!(
        focei_add > 0.75,
        "fixture must identify a non-trivial additive sigma; FOCEI ADD={focei_add}"
    );
    let rel = (saem_add - focei_add).abs() / focei_add;
    assert!(
        rel < 0.35,
        "SAEM ADD should stay close to FOCEI, not collapse: SAEM={saem_add}, FOCEI={focei_add}, rel={rel:.3}"
    );
}

// ---------------------------------------------------------------------------
// #1445: a *minority* additive component, sparse data
// ---------------------------------------------------------------------------

/// Shaped on `ferx-testdata/cefepime_jordan/ferx/run64.ferx` — 2-cpt IV
/// infusion, block Ω(3) on CL/V1/V2, `combined()` residual error — with the
/// covariate model dropped, since #1445 is a property of the residual channel
/// and not of the structural model.
const SPARSE_MODEL: &str = r#"
[parameters]
  theta TVCL(4.0, 0.2, 40.0)
  theta TVV1(22.0, 2.0, 200.0)
  theta TVV2(13.0, 1.0, 200.0)
  block_omega (ETA_CL, ETA_V1, ETA_V2) = [0.17, 0.02, 0.15, 0.02, 0.02, 0.69]
  sigma PROP_ERR ~ 0.13 (sd)
  sigma ADD_ERR  ~ 1.80 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = 12.0
  V2 = TVV2 * exp(ETA_V2)

[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)

[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)
"#;

/// Simulation truth for [`SPARSE_MODEL`].
const SPARSE_TRUE_ADD: f64 = 1.8;
const SPARSE_TRUE_PROP: f64 = 0.13;

/// The optimizer's hard lower bound for a log-packed σ (`exp(-8)`), quoted from
/// `run_saem`'s `log_sigma_lower`. "Collapsed" in #1445 means sitting on this;
/// the three reference seeds returned 0.011, 0.0025 and 0.011 against a truth of
/// 1.8, and the real datasets in the report reached 4.3e-4 and 4.5e-4.
const SIGMA_FLOOR: f64 = 3.354_626_279_025_118e-4; // exp(-8)

/// Sparse, cefepime-shaped design: q8h 30-minute infusions for three days, then
/// 1–3 samples in the last interval, trough-heavy. Deterministic (a fixed LCG),
/// so the fixture is the same on every platform and run.
fn sparse_template_population(n: usize) -> Population {
    let mut rng_state: u64 = 0x1445_0000_beef;
    let mut next = || {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rng_state >> 33) as f64) / ((1u64 << 31) as f64)
    };
    let subjects = (1..=n)
        .map(|i| {
            let doses: Vec<DoseEvent> = (0..9)
                .map(|d| DoseEvent::new(d as f64 * 8.0, 2000.0, 1, 4000.0, false, 0.0))
                .collect();
            let u = next();
            let n_obs = if u < 0.62 {
                1
            } else if u < 0.88 {
                2
            } else {
                3
            };
            // 64..72 h is the last dosing interval; 60% of samples land in its
            // last two hours.
            let mut times: Vec<f64> = (0..n_obs)
                .map(|_| {
                    let v = next();
                    if next() < 0.6 {
                        70.0 + 2.0 * v
                    } else {
                        64.6 + 5.0 * v
                    }
                })
                .collect();
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let m = times.len();
            common::subject(&format!("{i}"), doses, times, vec![0.0; m], vec![1; m])
        })
        .collect();
    Population {
        subjects,
        covariate_names: vec![],
        dv_column: "dv".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

fn sparse_simulated_population(model: &ferx_core::types::CompiledModel, n: usize) -> Population {
    let template = sparse_template_population(n);
    let mut truth = model.default_params.clone();
    truth.theta = vec![4.0, 22.0, 13.0];
    truth.sigma.values = vec![SPARSE_TRUE_PROP, SPARSE_TRUE_ADD];
    let sim = simulate_with_seed(model, &template, &truth, 1, 20_260_918);
    let mut pop = template;
    for subj in pop.subjects.iter_mut() {
        subj.observations = sim
            .iter()
            .filter(|row| row.id == subj.id)
            .map(|row| row.outcome.continuous_value())
            .collect();
    }
    pop
}

fn sparse_fit(
    method: EstimationMethod,
    model: &ferx_core::types::CompiledModel,
    pop: &Population,
    seed: u64,
) -> ferx_core::types::FitResult {
    let mut opts = FitOptions::default();
    opts.method = method;
    opts.run_covariance_step = false;
    opts.verbose = false;
    opts.outer_maxiter = 400;
    opts.saem_n_exploration = 150;
    opts.saem_n_convergence = 250;
    opts.saem_seed = Some(seed);
    fit(model, pop, &model.default_params, &opts).expect("fit must succeed")
}

/// Tier-3, #1445. SAEM's `ADD_ERR` must stay within a factor of two of the
/// simulation truth on a dataset where the additive term is a genuinely
/// identified *minority* variance component, instead of reporting whichever
/// value the last M-step's single-draw maximiser produced.
///
/// **Reference measurements** (release-equivalent `ci-test` profile, this
/// fixture, seeds 1/2/3 at 150/250):
///
/// | | `ADD_ERR` (truth 1.8) | `PROP_ERR` (truth 0.13) |
/// |---|---|---|
/// | FOCEI | 3.220 | 0.120 |
/// | SAEM before #1445 | 0.011 / 0.0025 / 0.011 | 0.133 / 0.137 / 0.135 |
/// | SAEM after #1445 | 1.697 / 1.161 / 1.746 | 0.137 / 0.143 / 0.138 |
///
/// **Bounds, measured.** Over 8 seeds × 2 schedules (150/250 and 300/700) after
/// the fix, `ADD_ERR` realises [1.112, 1.776] — worst |Δ| from truth 0.688 — and
/// `PROP_ERR` realises [0.1338, 0.1446], worst |Δ| 0.0146. The gates below are a
/// factor of two either side of truth for `ADD` (0.9, 3.6; the worst realised
/// value clears the lower gate by 1.24×) and 0.05 for `PROP` (3.4× the worst
/// realised error). The *before* column is not near those bounds in any sense:
/// 0.0025 is 7.4 σ-floors up where truth is 5400 floors up, so there is no risk
/// of the collapsed answer sneaking through a loose bound.
///
/// **Mutation results.** Each edit applied alone; every one reddens this test,
/// and the last column names the Tier-1 unit test in `src/estimation/saem.rs`
/// that also dies, so no side of the fix rests on this slow-gated test alone:
///
/// | edit | `ADD_ERR`, seed 1 | unit test that dies |
/// |---|---|---|
/// | full pre-#1445 revert (θ blend **and** θ's γ) | 0.0205 | `the_sigma_mstep_result_is_blended_not_assigned` |
/// | σ blended by `damp_mstep` (θ's log blend), γ_σ kept | 0.277 | same |
/// | `sigma_mstep_sa_step` returns `gamma_mstep` (schedule off) | 0.0205 | `sigma_mstep_sa_step_never_assigns_and_keeps_the_decay` |
/// | blend `log σ` instead of `σ²` (scale changed) | 0.277 | `damp_mstep_sigma_variance_blends_on_the_variance_scale` |
/// | `SIGMA_SA_MAX_STEP = 1.0` (cap off, blend kept) | 0.346 | `sigma_mstep_sa_step_never_assigns_and_keeps_the_decay` |
///
/// Three properties this test needs, and how each is kept live:
///
/// * **The additive term must really be identified**, or SAEM agreeing with a
///   collapsed FOCEI would pass. Asserted directly against the *truth* rather
///   than against FOCEI, plus a FOCEI sanity bound — FOCEI over-shoots here
///   (3.22 against 1.8) because the design has no low-concentration tail, and
///   that is exactly why FOCEI is not the anchor.
/// * **It must not be one lucky seed.** All three seeds are asserted; before the
///   fix all three collapsed, so a single-seed test would also have gone red,
///   but the seed spread (0.0025 vs 0.011, a factor of 4.4) is the visible
///   signature of a single-draw readout and is worth keeping in the assertion
///   set.
/// * **`is_finite` is not the assertion.** A collapsed σ_add is a perfectly
///   finite `f64` near `exp(-8)`; the bound has to be on the magnitude, and both
///   distances are stated above.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn saem_sparse_combined_additive_sigma_is_not_a_single_draw() {
    let model = parse_model_string(SPARSE_MODEL).expect("sparse combined model parses");
    let pop = sparse_simulated_population(&model, 300);

    // Fixture non-degeneracy: sparse per-subject data (median one observation
    // against three ETAs) is what makes the frozen-η residuals informative about
    // σ_add at all, and it is the half of the regime the #267 fixture lacks.
    let n_obs: usize = pop.subjects.iter().map(|s| s.observations.len()).sum();
    assert!(
        n_obs < 2 * pop.subjects.len(),
        "fixture must stay sparse (median ~1 observation/subject): {n_obs} obs / {} subjects",
        pop.subjects.len()
    );

    let focei = sparse_fit(EstimationMethod::FoceI, &model, &pop, 1);
    let focei_add = focei.sigma[1];
    assert!(
        focei_add.is_finite() && focei_add > 0.9,
        "fixture must identify a non-trivial additive sigma under FOCEI: ADD={focei_add}"
    );

    for seed in [1_u64, 2, 3] {
        let saem = sparse_fit(EstimationMethod::Saem, &model, &pop, seed);
        let add = saem.sigma[1];
        let prop = saem.sigma[0];
        assert!(
            add.is_finite() && prop.is_finite(),
            "seed {seed}: sigma must be finite, got PROP={prop} ADD={add}"
        );
        assert!(
            add > SIGMA_FLOOR * 1000.0,
            "seed {seed}: ADD={add:.6} is {:.1}× the optimizer floor {SIGMA_FLOOR:.3e} — \
             the #1445 collapse (reference seeds: 0.011, 0.0025, 0.011)",
            add / SIGMA_FLOOR
        );
        assert!(
            add > SPARSE_TRUE_ADD / 2.0 && add < SPARSE_TRUE_ADD * 2.0,
            "seed {seed}: ADD={add:.4} must stay within a factor of two of the simulation \
             truth {SPARSE_TRUE_ADD} (worst realised over 8 seeds × 2 schedules: 1.112)"
        );
        // The proportional half must not absorb the additive term's variance
        // either — a σ_prop that ran away would be the same defect wearing the
        // other component's name.
        assert!(
            (prop - SPARSE_TRUE_PROP).abs() < 0.05,
            "seed {seed}: PROP={prop:.4} must stay near the simulation truth \
             {SPARSE_TRUE_PROP} (worst realised |Δ| over 8 seeds × 2 schedules: 0.0146)"
        );
    }
}
