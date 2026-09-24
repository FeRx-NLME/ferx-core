//! #1499: the residual diagnostics must ask the same question about a `CENS`
//! row that the likelihood asks — "is this row censored **for this fit**",
//! [`BloqMethod::is_censored_row`] — not "is the flag nonzero".
//!
//! Under the default `bloq_method = drop` the fit scores a `CENS != 0` row as an
//! ordinary observation at its `DV` (the documented meaning of `Drop`), while
//! IWRES / CWRES / NPDE used to blank it on the raw flag alone. For CWRES that
//! also moved every *other* row of the subject, since the decorrelation
//! `R̃^{-1/2}(y − f0)` mixes rows.
//!
//! Each test below straddles the gate in one body — the `drop` leg and the `m3`
//! leg on the same fixture — so a predicate stuck on either branch reddens it.

use super::postfit::compute_subject_results;
use crate::parser::model_parser::parse_model_string;
use crate::types::{BloqMethod, CompiledModel, DoseEvent, Population, Subject, SubjectResult};
use nalgebra::{DMatrix, DVector};

/// 1-cpt IV, one η on CL, proportional error. Parsed rather than hand-built so
/// the fixture stays close to a real model.
fn model(bloq: BloqMethod) -> CompiledModel {
    let mut m = parse_model_string(
        "[parameters]\n  theta TVCL(5.0)\n  theta TVV(50.0)\n  omega ETA_CL ~ 0.09\n  \
         sigma PROP_ERR ~ 0.04\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  \
         V  = TVV\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  \
         DV ~ proportional(PROP_ERR)\n",
    )
    .expect("parse");
    m.bloq_method = bloq;
    m
}

/// One subject, a 100 mg bolus and three post-dose samples; `cens` as given.
fn population(cens: Vec<i8>) -> Population {
    let subject = Subject {
        id: "1".to_string(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![1.0, 4.0, 12.0],
        observations: vec![1.7, 1.2, 0.35],
        obs_cmts: vec![1, 1, 1],
        cens,
        ..Default::default()
    };
    Population {
        subjects: vec![subject],
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

/// Post-fit diagnostics for that subject at a nonzero η̂ and a **nonzero** `H`.
///
/// `H` has to be nonzero or the fixture is degenerate for the CWRES half: with
/// `H = 0` the marginal `R̃ = R` is diagonal, the decorrelation stops mixing
/// rows, and dropping one row could not move the others — which is exactly the
/// effect under test.
fn diagnostics(bloq: BloqMethod, cens: Vec<i8>) -> SubjectResult {
    let m = model(bloq);
    let pop = population(cens);
    let params = m.default_params.clone();
    let eta = DVector::from_vec(vec![0.3]);
    let h = DMatrix::from_column_slice(3, 1, &[-0.9, -0.6, -0.15]);
    let (mut subjects, _) = compute_subject_results(
        &m,
        &pop,
        &params,
        std::slice::from_ref(&eta),
        std::slice::from_ref(&h),
        &[Vec::new()],
        true,
        None,
        false,
    );
    subjects.pop().expect("one subject")
}

/// The `H` above must actually correlate the rows, or every assertion in this
/// file about "the other rows move" is satisfied by a diagonal `R̃`. Pinning it
/// here keeps a later fixture edit from quietly making the straddles vacuous.
#[test]
fn the_fixture_is_non_degenerate_for_the_decorrelation() {
    // Two CWRES recipes that agree only when `H·Ω·Hᵀ` has no off-diagonals: the
    // decorrelated vector must not be the marginal standardisation. Compare the
    // all-live CWRES against the same subject's IWRES, which *is* per-row.
    let r = diagnostics(BloqMethod::Drop, vec![0, 0, 0]);
    let differs = r
        .cwres
        .iter()
        .zip(r.iwres.iter())
        .any(|(c, w)| (c - w).abs() > 1e-6);
    assert!(
        differs,
        "fixture is degenerate: CWRES equals the per-row IWRES, so the \
         decorrelation mixes nothing. cwres={:?} iwres={:?}",
        r.cwres, r.iwres
    );
}

/// IWRES (`src/api/postfit.rs`): blanked only when the fit scored the row as
/// censored.
#[test]
fn iwres_blanks_a_flagged_row_only_under_m3() {
    let flagged = vec![0, 1, 0];

    let m3 = diagnostics(BloqMethod::M3, flagged.clone());
    assert!(m3.iwres[1].is_nan(), "m3 must blank the censored row");
    assert!(m3.iwres[0].is_finite() && m3.iwres[2].is_finite());

    let drop = diagnostics(BloqMethod::Drop, flagged);
    assert!(
        drop.iwres.iter().all(|w| w.is_finite()),
        "under `drop` the flagged row is an ordinary observation: {:?}",
        drop.iwres
    );
    // ... and identical to the same data with no flag at all.
    let unflagged = diagnostics(BloqMethod::Drop, vec![0, 0, 0]);
    assert_eq!(drop.iwres, unflagged.iwres);
}

/// CWRES (`compute_cwres`): the flagged row leaves the decorrelation only under
/// `m3`. The `drop` leg asserts the **other** rows against the unflagged twin —
/// the half a per-row NaN check cannot see, and the half that was measured
/// wrong on the issue (subject 1's other CWRES moved by 6e-4).
#[test]
fn cwres_excludes_a_flagged_row_from_the_decorrelation_only_under_m3() {
    let flagged = vec![0, 1, 0];
    let unflagged = diagnostics(BloqMethod::Drop, vec![0, 0, 0]);

    let drop = diagnostics(BloqMethod::Drop, flagged.clone());
    assert!(
        drop.cwres.iter().all(|c| c.is_finite()),
        "under `drop` no row is censored for the fit: {:?}",
        drop.cwres
    );
    assert_eq!(
        drop.cwres, unflagged.cwres,
        "under `drop` the flag must not touch any row's CWRES"
    );

    let m3 = diagnostics(BloqMethod::M3, flagged);
    assert!(m3.cwres[1].is_nan(), "m3 must blank the censored row");
    assert!(m3.cwres[0].is_finite() && m3.cwres[2].is_finite());
    // Removing a row from the system moves the survivors — this is why the two
    // halves of the fit had to agree in the first place.
    assert!(
        (m3.cwres[0] - unflagged.cwres[0]).abs() > 1e-9,
        "dropping a row from the decorrelation must move the others: \
         m3={:?} all-live={:?}",
        m3.cwres,
        unflagged.cwres
    );
}
