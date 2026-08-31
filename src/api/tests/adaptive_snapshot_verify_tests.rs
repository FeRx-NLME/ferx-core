use super::*;
use crate::parser::model_parser::parse_model_string;
use crate::sim::adaptive::{
    kappa_standard_normal, subject_kappa_base_seed, ControllerCtx, DoseAction,
};
use std::collections::HashMap;

// IOV (per-occasion κ on CL) × a time-varying covariate (CRCL) on CL — the exact
// composition the "decision covariate frozen at t=0" bug lived in (#732 / #739).
// CL depends on CRCL, so resolving a decision's covariate at the wrong time gives
// a different `decision_pk`, which is what the check must catch.
const IOV_TVCOV: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  kappa KAPPA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * (CRCL / 100.0) * exp(ETA_CL + KAPPA_CL)
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[error_model]
  DV ~ proportional(PROP)
"#;

// Same structural model without IOV — exercises the TV-only branch (event_pk
// present, eta_occ / decision_pk absent).
const TVCOV_NO_IOV: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 1e-10
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * (CRCL / 100.0) * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[error_model]
  DV ~ proportional(PROP)
"#;

const SEED: u64 = 20_260_708;
const SIM: usize = 1;

// A dose-free base subject whose covariate (CRCL) drops from the t=0 baseline
// (100) to 50 at the first record (t=6). An off-grid decision at t=12 then has a
// LOCF covariate (50, from the t=6 record) that differs from the frozen t=0
// baseline (100) — the setup that makes the historical decision_pk bug observable.
fn tv_subject() -> Subject {
    let base = HashMap::from([("CRCL".to_string(), 100.0)]);
    let lo = HashMap::from([("CRCL".to_string(), 50.0)]);
    Subject {
        id: "S1".into(),
        doses: Vec::new(),
        obs_times: vec![6.0, 24.0],
        obs_raw_times: Vec::new(),
        observations: vec![0.0, 0.0],
        obs_cmts: vec![1, 1],
        covariates: base,
        dose_covariates: Vec::new(),
        obs_covariates: vec![lo.clone(), lo],
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0, 0],
        occasions: vec![1, 1],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    }
}

// `tv_subject` plus an EVID=2 (pk-only) record at t=12 (CRCL=75) — so `event_pk`
// carries a `pk_only` snapshot and the inline pk-only re-derivation branch is
// exercised.
fn tv_subject_with_pk_only() -> Subject {
    let mut s = tv_subject();
    s.pk_only_times = vec![12.0];
    s.pk_only_covariates = vec![HashMap::from([("CRCL".to_string(), 75.0)])];
    s
}

// `tv_subject` plus a pre-scheduled base dose at t=0 (its own covariate snapshot,
// CRCL=100) — so `event_pk.dose` carries a per-dose snapshot (#931) and the inline
// dose re-derivation branch of `check_event_pk_records` is exercised.
fn tv_subject_with_dose() -> Subject {
    let mut s = tv_subject();
    s.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    s.dose_covariates = vec![HashMap::from([("CRCL".to_string(), 100.0)])];
    s.dose_occasions = vec![0];
    s
}

/// Build the canonical IOV snapshots exactly as `run_adaptive_population` does —
/// a third, independent orchestration used only to hand the check a *correct*
/// build (accepted) and known-corrupt copies (rejected).
fn canonical_iov(
    model: &CompiledModel,
    params: &ModelParameters,
    subject: &Subject,
    eta_slice: &[f64],
    decision_times: &[f64],
) -> (Vec<Vec<f64>>, Vec<PkParams>, crate::pk::EventPkParams) {
    let n_eta = model.n_eta;
    let omega_iov = params.omega_iov.as_ref().unwrap();
    let kappa_base = subject_kappa_base_seed(SEED, &subject.id, SIM);
    let n_occ = decision_times.len();
    let mut eta_occ = Vec::with_capacity(n_occ);
    let mut decision_pk = Vec::with_capacity(n_occ);
    for g in 0..n_occ {
        let zk: Vec<f64> = (0..model.n_kappa)
            .map(|k| kappa_standard_normal(kappa_base, g, k))
            .collect();
        let kappa_g = &omega_iov.chol * DVector::from_column_slice(&zk);
        let mut e: Vec<f64> = eta_slice[..n_eta].to_vec();
        e.extend(kappa_g.iter().copied());
        let dcov = crate::ode::predictions::locf_decision_cov(
            decision_times[g],
            subject,
            &subject.covariates,
        );
        decision_pk.push((model.pk_param_fn)(
            &params.theta,
            &e,
            dcov,
            decision_times[g],
        ));
        eta_occ.push(e);
    }
    let event_pk = crate::pk::compute_event_pk_params_iov(
        model,
        subject,
        &params.theta,
        eta_slice,
        &eta_occ,
        decision_times,
    );
    (eta_occ, decision_pk, event_pk)
}

#[test]
fn accepts_a_canonical_iov_build() {
    let model = parse_model_string(IOV_TVCOV).expect("parse");
    let params = model.default_params.clone();
    let subject = tv_subject();
    let eta_slice = vec![0.0; model.n_eta + model.n_kappa];
    let decisions = vec![0.0, 12.0, 24.0];
    let (eta_occ, decision_pk, event_pk) =
        canonical_iov(&model, &params, &subject, &eta_slice, &decisions);
    verify_adaptive_snapshots(
        &model,
        &params,
        &subject,
        &eta_slice,
        &decisions,
        SEED,
        SIM,
        Some(&eta_occ),
        Some(&decision_pk),
        Some(&event_pk),
    )
    .expect("a canonical build must pass the snapshot check");
}

#[test]
fn catches_decision_pk_frozen_at_t0_covariate() {
    // The #732 / #739 defect: build decision_pk[1] (off-grid decision at t=12)
    // with the frozen t=0 covariate instead of LOCF. The check must reject it —
    // the frozen-replay verifier reuses the same array and cannot.
    let model = parse_model_string(IOV_TVCOV).expect("parse");
    let params = model.default_params.clone();
    let subject = tv_subject();
    let eta_slice = vec![0.0; model.n_eta + model.n_kappa];
    let decisions = vec![0.0, 12.0, 24.0];
    let (eta_occ, mut decision_pk, event_pk) =
        canonical_iov(&model, &params, &subject, &eta_slice, &decisions);

    // Off-grid decision g=1 (t=12): LOCF covariate is the t=6 record (CRCL=50);
    // the frozen baseline is CRCL=100 → a genuinely different snapshot.
    let frozen = (model.pk_param_fn)(&params.theta, &eta_occ[1], &subject.covariates, 12.0);
    assert!(
        !pk_bits_eq(&frozen, &decision_pk[1]),
        "test setup must make the frozen-t0 snapshot differ from the LOCF one, \
             else the corruption is vacuous"
    );
    decision_pk[1] = frozen;

    let err = verify_adaptive_snapshots(
        &model,
        &params,
        &subject,
        &eta_slice,
        &decisions,
        SEED,
        SIM,
        Some(&eta_occ),
        Some(&decision_pk),
        Some(&event_pk),
    )
    .expect_err("a frozen-t0 decision_pk must be rejected");
    assert!(err.contains("decision_pk[1]"), "names the slot: {err}");
    assert!(err.contains("#748"), "references the issue: {err}");
}

#[test]
fn catches_corrupted_occasion_kappa() {
    let model = parse_model_string(IOV_TVCOV).expect("parse");
    let params = model.default_params.clone();
    let subject = tv_subject();
    let eta_slice = vec![0.0; model.n_eta + model.n_kappa];
    let decisions = vec![0.0, 12.0, 24.0];
    let (mut eta_occ, decision_pk, event_pk) =
        canonical_iov(&model, &params, &subject, &eta_slice, &decisions);
    // Perturb window 1's κ by a clearly-nonzero amount (independent of its draw).
    eta_occ[1][model.n_eta] += 1.0;
    let err = verify_adaptive_snapshots(
        &model,
        &params,
        &subject,
        &eta_slice,
        &decisions,
        SEED,
        SIM,
        Some(&eta_occ),
        Some(&decision_pk),
        Some(&event_pk),
    )
    .expect_err("a corrupted occasion κ must be rejected");
    assert!(err.contains("eta_occ[1]"), "names the slot: {err}");
}

#[test]
fn catches_corrupted_event_pk_snapshot() {
    let model = parse_model_string(IOV_TVCOV).expect("parse");
    let params = model.default_params.clone();
    let subject = tv_subject();
    let eta_slice = vec![0.0; model.n_eta + model.n_kappa];
    let decisions = vec![0.0, 12.0, 24.0];
    let (eta_occ, decision_pk, mut event_pk) =
        canonical_iov(&model, &params, &subject, &eta_slice, &decisions);
    // Corrupt the first observation's CL snapshot.
    event_pk.obs[0].values[0] += 1.0;
    let err = verify_adaptive_snapshots(
        &model,
        &params,
        &subject,
        &eta_slice,
        &decisions,
        SEED,
        SIM,
        Some(&eta_occ),
        Some(&decision_pk),
        Some(&event_pk),
    )
    .expect_err("a corrupted event_pk snapshot must be rejected");
    assert!(err.contains("event_pk.obs[0]"), "names the slot: {err}");
}

#[test]
fn catches_corrupted_pk_only_snapshot() {
    // A subject with an EVID=2 (pk-only) record exercises the inline pk-only
    // re-derivation branch; a mis-built pk_only snapshot must be rejected.
    let model = parse_model_string(IOV_TVCOV).expect("parse");
    let params = model.default_params.clone();
    let subject = tv_subject_with_pk_only();
    let eta_slice = vec![0.0; model.n_eta + model.n_kappa];
    let decisions = vec![0.0, 12.0, 24.0];
    let (eta_occ, decision_pk, mut event_pk) =
        canonical_iov(&model, &params, &subject, &eta_slice, &decisions);
    assert_eq!(
        event_pk.pk_only.len(),
        1,
        "the EVID=2 record must yield a pk_only snapshot"
    );
    event_pk.pk_only[0].values[0] += 1.0; // corrupt CL at the pk-only record
    let err = verify_adaptive_snapshots(
        &model,
        &params,
        &subject,
        &eta_slice,
        &decisions,
        SEED,
        SIM,
        Some(&eta_occ),
        Some(&decision_pk),
        Some(&event_pk),
    )
    .expect_err("a corrupted pk_only snapshot must be rejected");
    assert!(err.contains("event_pk.pk_only[0]"), "names the slot: {err}");
}

#[test]
fn catches_corrupted_dose_snapshot() {
    // #931: a subject carrying a pre-scheduled BASE dose exercises the inline dose
    // re-derivation branch. The driver AND the frozen-replay verifier both read
    // `event_pk.dose[k]` for that dose's F, so a mis-built dose snapshot is applied by
    // both and the replay cannot catch it — this #748 check is the backstop. A canonical
    // build is accepted; a corrupted dose snapshot must be rejected by slot.
    let model = parse_model_string(IOV_TVCOV).expect("parse");
    let params = model.default_params.clone();
    let subject = tv_subject_with_dose();
    let eta_slice = vec![0.0; model.n_eta + model.n_kappa];
    let decisions = vec![0.0, 12.0, 24.0];
    let (eta_occ, decision_pk, mut event_pk) =
        canonical_iov(&model, &params, &subject, &eta_slice, &decisions);
    assert_eq!(
        event_pk.dose.len(),
        1,
        "the base dose must yield a per-dose snapshot (#931)"
    );
    // Sanity: the canonical (correct) build passes before we corrupt it.
    verify_adaptive_snapshots(
        &model,
        &params,
        &subject,
        &eta_slice,
        &decisions,
        SEED,
        SIM,
        Some(&eta_occ),
        Some(&decision_pk),
        Some(&event_pk),
    )
    .expect("a canonical base-dose build must pass");

    event_pk.dose[0].values[0] += 1.0; // corrupt CL at the base dose
    let err = verify_adaptive_snapshots(
        &model,
        &params,
        &subject,
        &eta_slice,
        &decisions,
        SEED,
        SIM,
        Some(&eta_occ),
        Some(&decision_pk),
        Some(&event_pk),
    )
    .expect_err("a corrupted dose snapshot must be rejected");
    assert!(err.contains("event_pk.dose[0]"), "names the slot: {err}");
}

#[test]
fn catches_missized_occasion_arrays() {
    let model = parse_model_string(IOV_TVCOV).expect("parse");
    let params = model.default_params.clone();
    let subject = tv_subject();
    let eta_slice = vec![0.0; model.n_eta + model.n_kappa];
    let decisions = vec![0.0, 12.0, 24.0];
    let (mut eta_occ, decision_pk, event_pk) =
        canonical_iov(&model, &params, &subject, &eta_slice, &decisions);
    eta_occ.pop(); // length 2, not 3
    let err = verify_adaptive_snapshots(
        &model,
        &params,
        &subject,
        &eta_slice,
        &decisions,
        SEED,
        SIM,
        Some(&eta_occ),
        Some(&decision_pk),
        Some(&event_pk),
    )
    .expect_err("a mis-sized occasion array must be rejected");
    assert!(
        err.contains("mis-sized"),
        "explains the size mismatch: {err}"
    );
}

#[test]
fn accepts_and_checks_tv_only_path() {
    // Non-IOV TV-covariate path: event_pk present, eta_occ / decision_pk absent.
    let model = parse_model_string(TVCOV_NO_IOV).expect("parse");
    let params = model.default_params.clone();
    let subject = tv_subject();
    let eta_slice = vec![0.0; model.n_eta + model.n_kappa]; // n_kappa == 0
    let decisions = vec![0.0, 12.0, 24.0];
    let mut event_pk =
        crate::pk::compute_event_pk_params(&model, &subject, &params.theta, &eta_slice);
    verify_adaptive_snapshots(
        &model,
        &params,
        &subject,
        &eta_slice,
        &decisions,
        SEED,
        SIM,
        None,
        None,
        Some(&event_pk),
    )
    .expect("a canonical TV-only build must pass");

    event_pk.obs[1].values[1] += 1.0;
    let err = verify_adaptive_snapshots(
        &model,
        &params,
        &subject,
        &eta_slice,
        &decisions,
        SEED,
        SIM,
        None,
        None,
        Some(&event_pk),
    )
    .expect_err("a corrupted TV-only event_pk must be rejected");
    assert!(err.contains("event_pk.obs[1]"), "names the slot: {err}");
}

#[test]
fn constant_path_is_a_noop() {
    // No precomputed snapshots (constant-covariate / non-IOV): nothing to check.
    let model = parse_model_string(TVCOV_NO_IOV).expect("parse");
    let params = model.default_params.clone();
    let subject = tv_subject();
    let eta_slice = vec![0.0; model.n_eta + model.n_kappa];
    let decisions = vec![0.0, 12.0, 24.0];
    verify_adaptive_snapshots(
        &model, &params, &subject, &eta_slice, &decisions, SEED, SIM, None, None, None,
    )
    .expect("the constant path performs no snapshot checks");
}

#[test]
fn wired_into_a_real_reactive_tvcov_iov_run() {
    // End-to-end: a genuine reactive IOV × declining-covariate run with the
    // default-on verifier must pass — the snapshot check is wired at the
    // chokepoint and does not false-positive on a correct production build.
    let model = parse_model_string(IOV_TVCOV).expect("parse");
    let base = HashMap::from([("CRCL".to_string(), 100.0)]);
    let decisions = vec![0.0, 24.0, 48.0, 72.0];
    // Obs coincide with decisions; CRCL declines across the horizon.
    let crcls = [100.0, 80.0, 60.0, 40.0];
    let obs_covariates: Vec<HashMap<String, f64>> = crcls
        .iter()
        .map(|&c| HashMap::from([("CRCL".to_string(), c)]))
        .collect();
    let subject = Subject {
        id: "S1".into(),
        doses: Vec::new(),
        obs_times: decisions.clone(),
        obs_raw_times: Vec::new(),
        observations: vec![0.0; decisions.len()],
        obs_cmts: vec![1; decisions.len()],
        covariates: base,
        dose_covariates: Vec::new(),
        obs_covariates,
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; decisions.len()],
        occasions: vec![1; decisions.len()],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    let pop = Population {
        subjects: vec![subject],
        covariate_names: vec!["CRCL".into()],
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let opts = AdaptiveSimulateOptions {
        seed: Some(SEED),
        decision_times: decisions,
        ..Default::default() // verify = true
    };
    let controller = || |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }];
    let res = simulate_adaptive(&model, &pop, &model.default_params, 1, controller, &opts)
        .expect("a correct reactive IOV × TV-cov run passes the wired snapshot + replay checks");
    assert_eq!(res.ledger.len(), 4, "a dose at every decision");
}

#[test]
fn rejects_iov_arrays_without_omega_iov() {
    // eta_occ / decision_pk present (the IOV path) but the params carry no
    // omega_iov — a build-loop invariant violation the check must flag with a
    // typed error rather than unwrap-panic.
    let model = parse_model_string(TVCOV_NO_IOV).expect("parse"); // omega_iov = None
    let params = model.default_params.clone();
    assert!(
        params.omega_iov.is_none(),
        "TVCOV_NO_IOV must have no omega_iov"
    );
    let subject = tv_subject();
    let eta_slice = vec![0.0; model.n_eta + model.n_kappa];
    let decisions = vec![0.0, 12.0, 24.0];
    let eta_occ = vec![vec![0.0; model.n_eta]; decisions.len()];
    let decision_pk = vec![PkParams::default(); decisions.len()];
    let event_pk = crate::pk::compute_event_pk_params(&model, &subject, &params.theta, &eta_slice);
    let err = verify_adaptive_snapshots(
        &model,
        &params,
        &subject,
        &eta_slice,
        &decisions,
        SEED,
        SIM,
        Some(&eta_occ),
        Some(&decision_pk),
        Some(&event_pk),
    )
    .expect_err("IOV arrays without omega_iov must be rejected");
    assert!(err.contains("omega_iov"), "names the missing matrix: {err}");
}

#[test]
fn rejects_iov_run_missing_event_pk() {
    // On the IOV path event_pk must be present (κ makes PK per-occasion, so obs /
    // pk-only records each carry a snapshot). A None here is a build invariant
    // violation, caught after the per-occasion checks pass.
    let model = parse_model_string(IOV_TVCOV).expect("parse");
    let params = model.default_params.clone();
    let subject = tv_subject();
    let eta_slice = vec![0.0; model.n_eta + model.n_kappa];
    let decisions = vec![0.0, 12.0, 24.0];
    let (eta_occ, decision_pk, _event_pk) =
        canonical_iov(&model, &params, &subject, &eta_slice, &decisions);
    let err = verify_adaptive_snapshots(
        &model,
        &params,
        &subject,
        &eta_slice,
        &decisions,
        SEED,
        SIM,
        Some(&eta_occ),
        Some(&decision_pk),
        None,
    )
    .expect_err("an IOV run without event_pk must be rejected");
    assert!(err.contains("event_pk"), "names the missing array: {err}");
}

#[test]
fn catches_event_pk_length_mismatch() {
    // An event_pk array longer than the subject's record grid (a mis-built array
    // the frozen-replay verifier would reuse) must be rejected on length alone.
    let model = parse_model_string(IOV_TVCOV).expect("parse");
    let params = model.default_params.clone();
    let subject = tv_subject();
    let eta_slice = vec![0.0; model.n_eta + model.n_kappa];
    let decisions = vec![0.0, 12.0, 24.0];
    let (eta_occ, decision_pk, mut event_pk) =
        canonical_iov(&model, &params, &subject, &eta_slice, &decisions);
    event_pk.obs.push(PkParams::default()); // one more obs snapshot than records
    let err = verify_adaptive_snapshots(
        &model,
        &params,
        &subject,
        &eta_slice,
        &decisions,
        SEED,
        SIM,
        Some(&eta_occ),
        Some(&decision_pk),
        Some(&event_pk),
    )
    .expect_err("an over-long event_pk array must be rejected");
    assert!(err.contains("entr"), "explains the length mismatch: {err}");
}
