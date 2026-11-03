//! #1464 — the σ exploration cap of #1445, released per component.
//!
//! The cap (`SIGMA_SA_MAX_STEP`) exists for a residual variance component the
//! frozen-η objective barely identifies: its per-draw maximiser has a
//! boundary-heavy sampling distribution, so assigning it lets one draw put the
//! component on its floor (#1445, cefepime). A component the data pin down has
//! no such distribution, and capping it only slows σ's exploration trajectory —
//! which reaches every no-ETA θ through the same objective (busulfan `TVQ`,
//! #1464). These tests pin the statistic that tells the two apart, the schedule
//! that acts on it, and that a production fit actually reaches both.

use super::*;
use crate::parser::model_parser::parse_model_string;
use crate::types::{DoseEvent, Population, Subject};

fn population(subjects: Vec<Subject>) -> Population {
    Population {
        subjects,
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

fn subject(id: usize, doses: Vec<DoseEvent>, times: &[f64]) -> Subject {
    Subject {
        id: format!("{id}"),
        doses,
        obs_times: times.to_vec(),
        observations: vec![0.0; times.len()],
        obs_cmts: vec![1; times.len()],
        cens: vec![0; times.len()],
        ..Default::default()
    }
}

/// Population predictions (η = 0) at the model's initial θ, per subject.
fn predictions(model: &CompiledModel, pop: &Population) -> Vec<Vec<f64>> {
    let mut scratch = EventPkParams::default();
    pop.subjects
        .iter()
        .map(|s| {
            crate::pk::compute_predictions_with_tv_into(
                model,
                s,
                &model.default_params.theta,
                &vec![0.0; model.n_eta],
                &mut scratch,
            )
            .to_vec()
        })
        .collect()
}

fn zero_etas(model: &CompiledModel, pop: &Population) -> Vec<Vec<f64>> {
    vec![vec![0.0; model.n_eta]; pop.subjects.len()]
}

const ONE_CPT_ADDITIVE: &str = r"
[parameters]
  theta TVCL(2.0, 0.1, 20.0)
  theta TVV(20.0, 1.0, 200.0)
  omega ETA_CL ~ 0.09
  sigma ADD_ERR ~ 0.7 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ additive(ADD_ERR)
";

const ONE_CPT_COMBINED: &str = r"
[parameters]
  theta TVCL(2.0, 0.1, 20.0)
  theta TVV(20.0, 1.0, 200.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.2 (sd)
  sigma ADD_ERR  ~ 0.3 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)
";

fn dense_one_cpt_population(n: usize) -> Population {
    let times = [0.5_f64, 2.0, 6.0, 12.0, 24.0];
    population(
        (1..=n)
            .map(|i| {
                subject(
                    i,
                    vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                    &times,
                )
            })
            .collect(),
    )
}

/// One additive σ: `V = σ²`, `∂V/∂log σ = 2σ²`, so every observation carries
/// information `½·(2σ²)²/σ⁴ = 2` and `SE(log σ) = 1/√(2N)` — whatever σ is and
/// whatever the predictions are. An exact closed form, so the bound is
/// floating-point only.
///
/// Regression this exists to catch: a dropped `½`, a missing square on `V`, or
/// the information summed over subjects but not observations — each moves the
/// SE by a factor of at least √2.
#[test]
fn sigma_log_se_of_a_single_additive_sigma_is_one_over_root_2n() {
    let model = parse_model_string(ONE_CPT_ADDITIVE).expect("fixture parses");
    let pop = dense_one_cpt_population(7);
    let n_obs = 7.0 * 5.0;
    for sigma in [0.05_f64, 0.7, 30.0] {
        let se = sigma_log_se(
            &model,
            &pop,
            &model.default_params.theta,
            &[sigma],
            &[true],
            &zero_etas(&model, &pop),
            &[],
        )
        .expect("one free additive σ has a positive-definite block");
        let want = 1.0 / (2.0_f64 * n_obs).sqrt();
        assert!(
            (se[0] - want).abs() < 1e-12,
            "σ = {sigma}: SE(log σ) = {} against the closed form 1/√(2N) = {want}",
            se[0]
        );
    }
    // No free component is no block at all — `None`, which every caller reads
    // as "keep the cap".
    assert!(sigma_log_se(
        &model,
        &pop,
        &model.default_params.theta,
        &[0.7],
        &[false],
        &zero_etas(&model, &pop),
        &[],
    )
    .is_none());
}

/// Combined error: the 2×2 σ block assembled by hand from the two variance
/// components — `V = σ_a² + (σ_p·f)²`, `∂V/∂log σ_a = 2σ_a²`,
/// `∂V/∂log σ_p = 2(σ_p·f)²` — written here per component rather than through
/// `dvar_dlogsigma`, so the production assembly is checked against an
/// independent spelling of the same formula.
///
/// The second half is the reason the block is **inverted** rather than read off
/// its diagonal: the two components are positively confounded (both raise `V`),
/// so each one's SE once the other is free is strictly larger than `1/√I_kk`.
/// A diagonal-only gate would call a confounded component identified. With the
/// proportional σ FIXed the free one's SE is exactly `1/√I_aa`, and the fixed
/// slot reads `+∞` (never released — it is not a coordinate).
#[test]
fn sigma_log_se_inverts_the_combined_block() {
    let model = parse_model_string(ONE_CPT_COMBINED).expect("fixture parses");
    let pop = dense_one_cpt_population(6);
    let (sp, sa) = (0.2_f64, 0.3_f64);
    let mut i_aa = 0.0;
    let mut i_pp = 0.0;
    let mut i_ap = 0.0;
    for preds in predictions(&model, &pop) {
        for f in preds {
            let (va, vp) = (sa * sa, (sp * f) * (sp * f));
            let v = va + vp;
            i_aa += 0.5 * (2.0 * va) * (2.0 * va) / (v * v);
            i_pp += 0.5 * (2.0 * vp) * (2.0 * vp) / (v * v);
            i_ap += 0.5 * (2.0 * va) * (2.0 * vp) / (v * v);
        }
    }
    let det = i_aa * i_pp - i_ap * i_ap;
    let (want_p, want_a) = ((i_aa / det).sqrt(), (i_pp / det).sqrt());

    let theta = &model.default_params.theta;
    let etas = zero_etas(&model, &pop);
    // σ order follows the declaration: [PROP_ERR, ADD_ERR].
    let se = sigma_log_se(&model, &pop, theta, &[sp, sa], &[true, true], &etas, &[])
        .expect("a dense combined design identifies both components");
    for (k, (got, want)) in [(se[0], want_p), (se[1], want_a)].into_iter().enumerate() {
        assert!(
            (got - want).abs() < 1e-9 * want,
            "component {k}: SE {got} against the hand-assembled inverse {want}"
        );
    }
    // The confounding is real on this fixture, so the inverse is doing work.
    for (k, (got, diag_only)) in [(se[0], 1.0 / i_pp.sqrt()), (se[1], 1.0 / i_aa.sqrt())]
        .into_iter()
        .enumerate()
    {
        assert!(
            got > 1.05 * diag_only,
            "component {k}: inverse SE {got} is not visibly above the diagonal-only \
             {diag_only} — the fixture no longer confounds the two components"
        );
    }

    let se_fix = sigma_log_se(&model, &pop, theta, &[sp, sa], &[false, true], &etas, &[])
        .expect("one free component");
    assert_eq!(se_fix[0], f64::INFINITY, "a FIXed σ is not a coordinate");
    let want_marginal = 1.0 / i_aa.sqrt();
    assert!(
        (se_fix[1] - want_marginal).abs() < 1e-9 * want_marginal,
        "with PROP FIXed, ADD's SE is 1/√I_aa = {want_marginal}, got {}",
        se_fix[1]
    );
}

/// The schedule. Exploration releases exactly the components at or under the
/// threshold, to `min(γ, γ_mstep)` — never faster than θ; convergence and an
/// absent SE vector keep #1445's cap on every component.
///
/// Regression this exists to catch: the release dropped (every component back
/// on `SIGMA_SA_MAX_STEP`, which is `main` before #1464 and the busulfan `TVQ`
/// bias), the release leaking into convergence (the #1445 hand-off overwrite),
/// or the comparison flipped (releasing the *weak* component, which is #1445).
#[test]
fn sigma_mstep_sa_steps_releases_only_identified_components_in_exploration() {
    let thr = SIGMA_CAP_RELEASE_MAX_LOG_SE;
    let se = [0.5 * thr, thr, 1.5 * thr, f64::INFINITY];
    let capped = sigma_mstep_sa_step(1.0, 1.0);
    assert_eq!(capped, SIGMA_SA_MAX_STEP);

    // Exploration, damping off: the identified components are assigned.
    assert_eq!(
        sigma_mstep_sa_steps(1.0, 1.0, true, Some(&se), 4),
        vec![1.0, 1.0, capped, capped]
    );
    // Never faster than θ: with `mstep_damping` binding, released == γ_mstep.
    assert_eq!(
        sigma_mstep_sa_steps(1.0, 0.5, true, Some(&se), 4),
        vec![0.5, 0.5, SIGMA_SA_MAX_STEP, SIGMA_SA_MAX_STEP]
    );
    assert_eq!(
        sigma_mstep_sa_steps(1.0, MSTEP_SA_MAX_STEP_IIV_ON_RUV, true, Some(&se), 4),
        vec![MSTEP_SA_MAX_STEP_IIV_ON_RUV; 4]
    );
    // No SE (out of scope, singular, not computed): the #1445 schedule exactly.
    assert_eq!(
        sigma_mstep_sa_steps(1.0, 1.0, true, None, 3),
        vec![capped; 3]
    );
    // Convergence: the hand-off cap stays on for every component, and the
    // decaying schedule is untouched past it.
    assert_eq!(
        sigma_mstep_sa_steps(1.0, 1.0, false, Some(&se), 4),
        vec![capped; 4]
    );
    assert_eq!(
        sigma_mstep_sa_steps(1.0 / 6.0, 1.0, false, Some(&se), 4),
        vec![1.0 / 6.0; 4]
    );
}

/// One-compartment IV with a **no-ETA** `TVV` — so the numerical θ/σ M-step
/// runs and σ rides it rather than the scalar-statistic channel — and a single
/// proportional σ started at ten times its truth.
const RELEASE_FIT_MODEL: &str = r"
[parameters]
  theta TVCL(2.0, 0.1, 20.0)
  theta TVV(20.0, 1.0, 200.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 1.0 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
";

/// Truth for [`RELEASE_FIT_MODEL`]'s σ; the start is 1.0.
const RELEASE_FIT_TRUE_PROP: f64 = 0.1;

/// 30 subjects × 6 samples with deterministic proportional noise of SD σ, so
/// `SE(log σ) = 1/√(2·180) = 0.053` — under the release threshold.
fn release_fit_population(model: &CompiledModel) -> Population {
    let times = [0.5_f64, 1.0, 2.0, 6.0, 12.0, 24.0];
    let mut pop = population(
        (1..=30)
            .map(|i| {
                subject(
                    i,
                    vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                    &times,
                )
            })
            .collect(),
    );
    let mut scratch = EventPkParams::default();
    let mut st = 0x1464_u64;
    let mut next = || {
        st = st
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((st >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    for s in pop.subjects.iter_mut() {
        let eta = 0.3 * next();
        let preds = crate::pk::compute_predictions_with_tv_into(
            model,
            s,
            &model.default_params.theta,
            &[eta],
            &mut scratch,
        );
        // Uniform on ±√3·σ has SD σ exactly.
        s.observations = preds
            .iter()
            .map(|&f| f * (1.0 + RELEASE_FIT_TRUE_PROP * 3.0_f64.sqrt() * next()))
            .collect();
    }
    pop
}

/// The release reaches a production fit: `run_saem` computes the SE at the
/// M-step and hands the released step to the blend.
///
/// Regression this exists to catch: the gate wired to nothing — `sigma_se`
/// never computed (`None`), every helper test above still green, and every fit
/// back on #1445's cap. That is the smallest edit that undoes #1464 in
/// production, and no helper-level test can see it.
///
/// **Closed forms for both arms.** Three exploration M-steps then one
/// convergence step, σ starting at 1.0 against a maximiser near 0.1. Capped,
/// σ² moves 0.2 of the way per step, so after four steps it still carries
/// `0.8⁴ = 0.41` of the starting variance: σ ≥ √0.41 = 0.64. Released, the
/// first exploration step assigns the maximiser and σ is near 0.1 from there
/// on. The bound sits between them with a factor of ~2 to each side.
#[test]
fn a_well_identified_sigma_is_released_in_a_production_fit() {
    let model = parse_model_string(RELEASE_FIT_MODEL).expect("fixture parses");
    let pop = release_fit_population(&model);
    let mut opts = FitOptions::default();
    opts.method = crate::types::EstimationMethod::Saem;
    opts.saem_n_exploration = 3;
    opts.saem_n_convergence = 1;
    opts.saem_seed = Some(1464);
    opts.run_covariance_step = false;
    let sigma = crate::api::fit(&model, &pop, &model.default_params, &opts)
        .expect("SAEM Ok")
        .sigma[0];
    assert!(sigma.is_finite(), "σ = {sigma}");
    assert!(
        sigma < 0.3,
        "σ = {sigma} after three exploration M-steps from 1.0 against a truth of \
         {RELEASE_FIT_TRUE_PROP}: the capped schedule leaves σ ≥ 0.64, so the \
         release did not reach this fit"
    );
    assert!(
        sigma > 0.5 * RELEASE_FIT_TRUE_PROP,
        "σ = {sigma} overshot to below half its truth"
    );
}

/// The release is gated to the scope the information is defined on, and an
/// `iiv_on_ruv` model — whose σ shares a ridge with ω_RUV (#895) and has its own
/// open σ-side drift (#1421) — keeps the cap.
#[test]
fn the_release_scope_excludes_iiv_on_ruv_and_every_numerical_mstep_gap() {
    let plain = parse_model_string(ONE_CPT_COMBINED).expect("fixture parses");
    assert_eq!(sigma_cap_release_scope_gap(&plain, None), None);
    assert_eq!(
        sigma_cap_release_scope_gap(&plain, Some("the model has IOV")),
        Some("the model has IOV")
    );
    let ruv = parse_model_string(
        r"
[parameters]
  theta TVCL(2.0, 0.1, 20.0)
  theta TVV(20.0, 1.0, 200.0)
  omega ETA_CL ~ 0.09
  omega ETA_RUV ~ 0.04
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
  iiv_on_ruv = ETA_RUV
",
    )
    .expect("iiv_on_ruv fixture parses");
    assert!(
        ruv.residual_error_eta.is_some(),
        "fixture must carry iiv_on_ruv"
    );
    assert!(sigma_cap_release_scope_gap(&ruv, None).is_some());
}

/// #1445's sparse, trough-heavy cefepime shape (`tests/saem_combined_error.rs`'s
/// `SPARSE_MODEL`): 2-cpt q8h infusions, 1–3 samples per subject in the last
/// interval, `combined(0.13, 1.8)` on troughs of 15–120.
const SPARSE_1445_MODEL: &str = r"
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
";

fn sparse_1445_population(n: usize) -> Population {
    let mut st: u64 = 0x1445_0000_beef;
    let mut next = || {
        st = st
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((st >> 33) as f64) / ((1u64 << 31) as f64)
    };
    population(
        (1..=n)
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
                subject(i, doses, &times)
            })
            .collect(),
    )
}

/// Busulfan's shape (`ferx-testdata/busulfan_dominic`, the #1464 B6 benchmark),
/// shrunk: dense sampling over a 3-hour infusion and its decline, readout in
/// µg/L so concentrations run to the thousands, `combined(0.10, 28)` — the
/// values B6 converges to.
const DENSE_BUSULFAN_MODEL: &str = r"
[parameters]
  theta TVCL(12.0, 0.1, 500.0)
  theta TVV1(40.0, 1.0, 500.0)
  theta TVQ(10.0, 0.1, 500.0)
  theta TVV2(12.0, 1.0, 500.0)
  omega ETA_CL ~ 0.1
  sigma PROP_ERR ~ 0.10 (sd)
  sigma ADD_ERR  ~ 28.0 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1
  Q  = TVQ
  V2 = TVV2
[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)
[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)
";

fn dense_busulfan_population(n: usize) -> Population {
    let times = [3.0_f64, 3.25, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0];
    population(
        (1..=n)
            .map(|i| {
                // 55 g·10⁻³ over 3 h, in µg so the readout is µg/L as on B6.
                subject(
                    i,
                    vec![DoseEvent::new(0.0, 55_000.0, 1, 55_000.0 / 3.0, false, 0.0)],
                    &times,
                )
            })
            .collect(),
    )
}

/// The threshold separates the two shapes #1464 is about, at their own sizes.
///
/// Realised when written (η = 0, σ at the values each fit converges to):
///
/// | | `SE(log σ_prop)` | `SE(log σ_add)` |
/// |---|---|---|
/// | #1445 sparse fixture, 300 subjects | 0.0924 | **1.061** |
/// | busulfan shape, 600 subjects | 0.0179 | **0.0296** |
///
/// against a threshold of 0.10. On the real data the additive SE along 6-seed
/// fits was 0.15–0.24 for cefepime and fell to 0.06 for busulfan — the same
/// ordering, with the sparse fixture's additive term an order of magnitude
/// weaker still. The sparse proportional term is released, with 8 % to spare:
/// it is the majority component there, and `saem_combined_error.rs`'s #1445
/// tests are what show that releasing it does not bring the additive collapse
/// back. Pinned so a change to [`SIGMA_CAP_RELEASE_MAX_LOG_SE`] has to revisit
/// this measurement.
///
/// Regression this exists to catch: a threshold loose enough to release
/// #1445's component (the collapse comes back — 2 of 6 cefepime seeds reached
/// `ADD_ERR` 0.05 / 0.10 with the cap off entirely), or tight enough to keep
/// busulfan's capped (the `TVQ` bias stays).
#[test]
fn the_release_threshold_separates_the_1445_and_busulfan_shapes() {
    let sparse = parse_model_string(SPARSE_1445_MODEL).expect("sparse fixture parses");
    let sp = sparse_1445_population(300);
    let se_sparse = sigma_log_se(
        &sparse,
        &sp,
        &sparse.default_params.theta,
        &[0.13, 1.8],
        &[true, true],
        &zero_etas(&sparse, &sp),
        &[],
    )
    .expect("sparse block is positive definite");

    let dense = parse_model_string(DENSE_BUSULFAN_MODEL).expect("dense fixture parses");
    let dp = dense_busulfan_population(600);
    let se_dense = sigma_log_se(
        &dense,
        &dp,
        &dense.default_params.theta,
        &[0.10, 28.0],
        &[true, true],
        &zero_etas(&dense, &dp),
        &[],
    )
    .expect("dense block is positive definite");

    // Pinned: the value is the measured one (see the constant's table), not a
    // free parameter this fixture happens to bracket.
    assert_eq!(SIGMA_CAP_RELEASE_MAX_LOG_SE, 0.1);
    let thr = SIGMA_CAP_RELEASE_MAX_LOG_SE;
    assert!(
        se_sparse[1] > thr,
        "#1445's minority additive σ reads SE {} ≤ {thr}: it would be released",
        se_sparse[1]
    );
    assert!(
        se_dense[1] <= thr,
        "busulfan's additive σ reads SE {} > {thr}: it would stay capped",
        se_dense[1]
    );
    for (name, se) in [("sparse", se_sparse[0]), ("dense", se_dense[0])] {
        assert!(
            se <= thr,
            "the {name} proportional σ reads SE {se} > {thr}: a majority component \
             would stay capped"
        );
    }
}
