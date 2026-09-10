//! `categorical2` ≡ `categorical` at the matching θ (#1312).
//!
//! The two forms differ only in the non-reference branch — `1 + θ_k` against
//! `θ_k` — so they are an exact reparameterization of each other by
//! `θ_cat2 = 1 + θ_cat`. That identity is the oracle this feature needs no
//! external tool for: `categorical` is already the NONMEM-anchored PsN state 2
//! (`tests/covariate_model_equivalence.rs` pins the block against the classical
//! expression it desugars to), so pinning `cat2` onto `cat` at the matching θ
//! anchors it transitively.
//!
//! Two things the identity would not catch on its own, and which are asserted
//! here for that reason:
//!
//! - **The covariate has to be live.** A `SEX` column that never changes the
//!   prediction makes both arms equal to a no-covariate model, and the test
//!   passes on a factor of 1. So the fixture asserts the covariate moves the
//!   objective before comparing the two arms.
//! - **Both levels have to be reached.** With every subject at the reference
//!   level the non-reference branch — the only branch that differs — is never
//!   evaluated. The fixture asserts both levels are present in the data.
//!
//! `θ_cat = 0.25` / `θ_cat2 = 1.25` is chosen so `1 + θ_cat` is exact in IEEE
//! double, which is what lets this be a bit-for-bit comparison rather than a
//! tolerance.
//!
//! ## Why the objective is compared at `outer_maxiter = 0`
//!
//! The identity is between the two *models*, not between two optimizer runs.
//! The outer optimizer works on a packed vector, and the packing of θ is not
//! equivariant under `θ ↦ 1 + θ`: from the corresponding starts the two arms
//! take different steps and, after a bounded number of iterations, sit at
//! different points — measured at 2.47 OFV apart at `outer_maxiter = 2`, and
//! still 1.78 apart with the covariate θ declared `FIX`. That is a difference of
//! *trajectory*, not of objective. `outer_maxiter = 0` is an evaluation: it runs
//! the inner loop and scores the objective at the parameters as given, which is
//! the quantity the identity is about, and it can then be compared exactly.

use std::path::Path;

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::FitOptions;
use ferx_core::{fit, predict, read_nonmem_csv, Population};

const DATA: &str = "data/two_cpt_oral_cov.csv";

/// The non-reference factor, written both ways.
const THETA_CAT: f64 = 0.25;
const THETA_CAT2: f64 = 1.0 + THETA_CAT;

/// A two-compartment oral model whose `[covariate_model]` block is supplied by
/// the caller. `CL` carries an η so the factor has a non-η group to sit in.
fn model(covariate_model: &str) -> String {
    format!(
        r"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV1(40.0, 1.0, 500.0)
  theta TVQ(8.0, 0.1, 100.0)
  theta TVV2(80.0, 1.0, 500.0)
  theta TVKA(1.0, 0.01, 10.0)

  omega ETA_CL ~ 0.15
  sigma PROP_ERR ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1
  Q  = TVQ
  V2 = TVV2
  KA = TVKA

[structural_model]
  pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)

[covariates]
  SEX categorical(levels = [0, 1])

[covariate_model]
{covariate_model}

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method   = focei
  maxiter  = 2
  gradient = fd
"
    )
}

/// `data/two_cpt_oral_cov.csv` with a two-level `SEX` derived from the subject
/// index, so the fixture needs no new committed dataset and the split is
/// deterministic. Alternating by position guarantees both levels are populated
/// (asserted by the caller).
fn population_with_sex() -> Population {
    let mut pop = read_nonmem_csv(Path::new(DATA), None, None).expect("covariate dataset loads");
    for (i, subject) in pop.subjects.iter_mut().enumerate() {
        let level = (i % 2) as f64;
        subject.covariates.insert("SEX".to_string(), level);
        // The per-event snapshots are empty for this dataset (no time-varying
        // covariates), but write into them if they ever are not: a consumer
        // that reads a snapshot must not see a subject without `SEX`.
        let snapshots = [
            &mut subject.dose_covariates,
            &mut subject.obs_covariates,
            &mut subject.pk_only_covariates,
            &mut subject.reset_covariates,
        ];
        for rows in snapshots {
            for row in rows.iter_mut() {
                row.insert("SEX".to_string(), level);
            }
        }
    }
    pop
}

/// Which levels the population actually carries.
fn levels(pop: &Population) -> Vec<f64> {
    let mut seen: Vec<f64> = Vec::new();
    for s in &pop.subjects {
        let v = *s.covariates.get("SEX").expect("every subject carries SEX");
        if !seen.contains(&v) {
            seen.push(v);
        }
    }
    seen.sort_by(f64::total_cmp);
    seen
}

/// An evaluation, not a fit: the inner loop runs and the objective is scored at
/// the parameters as given. Tier-2 by construction — there is no outer loop to
/// converge.
fn eval_only_opts() -> FitOptions {
    FitOptions {
        outer_maxiter: 0,
        ..FitOptions::default()
    }
}

#[test]
fn categorical2_is_categorical_at_the_matching_theta() {
    let pop = population_with_sex();

    // Non-degeneracy 1: both branches of the chain are reached.
    assert_eq!(
        levels(&pop),
        vec![0.0, 1.0],
        "the fixture must populate the reference *and* the contrast level, or the \
         branch the two forms differ in is never evaluated"
    );

    let cat = parse_full_model(&model(&format!(
        "  CL ~ SEX categorical(ref = 0) => T_SEX({THETA_CAT}, -1.0, 5.0)"
    )))
    .expect("cat model parses");
    let cat2 = parse_full_model(&model(&format!(
        "  CL ~ SEX categorical2(ref = 0) => T_SEX({THETA_CAT2}, 0.0, 6.0)"
    )))
    .expect("cat2 model parses");
    let plain = parse_full_model(&model("  CL ~ SEX none")).expect("covariate-free model parses");

    // Both arms spend exactly one θ: this is a reparameterization, not a
    // cheaper test.
    assert_eq!(cat.model.theta_names, cat2.model.theta_names);
    assert_eq!(
        cat.model.theta_names.len(),
        plain.model.theta_names.len() + 1
    );

    let pred_cat = predict(&cat.model, &pop, &cat.model.default_params);
    let pred_cat2 = predict(&cat2.model, &pop, &cat2.model.default_params);
    let pred_plain = predict(&plain.model, &pop, &plain.model.default_params);

    // Non-degeneracy 2: the covariate is live — a factor of 1 would make the
    // identity below hold against a model that ignores SEX entirely.
    assert!(
        pred_cat
            .iter()
            .zip(&pred_plain)
            .any(|(a, b)| a.pred != b.pred),
        "SEX must change the prediction, or this test compares two null models"
    );

    // The identity itself, bit for bit: `1 + 0.25` and `1.25` are the same
    // double, and the desugar emits the same multiplication order either way.
    assert_eq!(pred_cat.len(), pred_cat2.len());
    for (a, b) in pred_cat.iter().zip(&pred_cat2) {
        assert_eq!(a.id, b.id);
        assert_eq!(a.time, b.time);
        assert_eq!(
            a.pred, b.pred,
            "PRED differs for subject {} at t = {}",
            a.id, a.time
        );
    }
}

#[test]
fn categorical2_is_the_same_objective_function_at_the_matching_theta() {
    let pop = population_with_sex();
    let opts = eval_only_opts();

    // The objective at the parameters as written — see the module header on why
    // this is an evaluation and not a short fit.
    let eval_block = |block: &str| {
        let parsed = parse_full_model(&model(block)).expect("model parses");
        fit(&parsed.model, &pop, &parsed.model.default_params, &opts).expect("evaluation")
    };

    let f_cat = eval_block(&format!(
        "  CL ~ SEX categorical(ref = 0) => T_SEX({THETA_CAT}, -1.0, 5.0)"
    ));
    let f_cat2 = eval_block(&format!(
        "  CL ~ SEX categorical2(ref = 0) => T_SEX({THETA_CAT2}, 0.0, 6.0)"
    ));
    let f_plain = eval_block("  CL ~ SEX none");

    // The objective actually ran: a sentinel or a diverged solve would satisfy
    // an equality between two equally broken arms.
    assert!(
        f_cat.ofv.is_finite() && f_cat2.ofv.is_finite() && f_cat.ofv.abs() < 1e6,
        "cat {} / cat2 {}",
        f_cat.ofv,
        f_cat2.ofv
    );
    // Non-degeneracy: the covariate moves the objective. Without this the
    // equality below is satisfied by two copies of the null model.
    assert_ne!(
        f_cat.ofv, f_plain.ofv,
        "the covariate must move the objective, or the identity is vacuous"
    );
    assert_eq!(
        f_cat.ofv, f_cat2.ofv,
        "cat at θ and cat2 at 1 + θ are the same objective function"
    );
    // The two θ vectors are the same but for the covariate coordinate, which
    // is the reparameterization itself — and it is *live*, not a coincidence
    // of two identical vectors.
    let n = f_cat.theta.len();
    assert_eq!(f_cat.theta[..n - 1], f_cat2.theta[..n - 1]);
    assert_eq!(f_cat.theta[n - 1] + 1.0, f_cat2.theta[n - 1]);
    assert_ne!(f_cat.theta[n - 1], f_cat2.theta[n - 1]);
}

/// `fix = v` pins the θ as written and nothing reinterprets it per form, so the
/// **null** is `fix = 0` for `categorical` and `fix = 1` for `categorical2`.
///
/// Stated as a differential pair that straddles the change: `fix = 1` must be
/// the covariate-free model for `cat2` and must *not* be for `cat`. Either half
/// alone passes against an implementation that silently re-centres the fixed
/// value. Compared on `predict()` rather than `fit()` so the covariate-free arm,
/// which carries one θ fewer, is not being compared through a different
/// optimizer trajectory.
#[test]
fn fix_is_the_theta_as_written_so_the_null_moves_with_the_form() {
    let pop = population_with_sex();

    let preds = |block: &str| {
        let parsed = parse_full_model(&model(block)).expect("model parses");
        predict(&parsed.model, &pop, &parsed.model.default_params)
            .into_iter()
            .map(|p| p.pred)
            .collect::<Vec<f64>>()
    };

    let plain = preds("  CL ~ SEX none");
    assert!(
        plain.iter().all(|p| p.is_finite()),
        "the covariate-free reference must be a real prediction vector"
    );

    assert_eq!(
        preds("  CL ~ SEX categorical2(ref = 0, fix = 1)"),
        plain,
        "`categorical2(fix = 1)` is a factor of 1 — the covariate-free model"
    );
    assert_eq!(
        preds("  CL ~ SEX categorical(ref = 0, fix = 0)"),
        plain,
        "`categorical(fix = 0)` is a factor of 1 — the covariate-free model"
    );
    // The straddle: the same `fix = 1` is a factor of *2* under `categorical`.
    // Without this half, an implementation that re-centres the fixed value per
    // form would pass.
    assert_ne!(
        preds("  CL ~ SEX categorical(ref = 0, fix = 1)"),
        plain,
        "`categorical(fix = 1)` is a factor of 2, not the null: the block pins the θ \
         as written and does not reinterpret it per form"
    );
    // …and symmetrically, `categorical2(fix = 0)` is a factor of *zero*, which
    // is emphatically not the null either.
    assert_ne!(
        preds("  CL ~ SEX categorical2(ref = 0, fix = 0)"),
        plain,
        "`categorical2(fix = 0)` zeroes the parameter at the contrast level"
    );
}
