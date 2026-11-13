use super::*;
use crate::parser::model_parser::parse_model_string;
use std::collections::HashMap;

fn disc(time: f64, state: usize, cmt: usize) -> ObsRecord {
    ObsRecord::DiscreteState {
        time,
        raw_time: time,
        state,
        cmt,
    }
}

fn subject(states: &[(f64, usize)]) -> Subject {
    Subject {
        id: "1".into(),
        obs_records: states.iter().map(|&(t, s)| disc(t, s, 5)).collect(),
        ..Default::default()
    }
}

fn inv_logit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Two states, a random effect on awake → asleep only.
const TWO_STATE: &str = r"
[parameters]
  theta A_AS(-0.8, -10.0, 10.0)
  theta A_SA(-1.3, -10.0, 10.0)
  omega ETA_AS ~ 0.3

[markov_model]
  type   = dtmm
  cmt    = 5
  states = [awake=0, asleep=1]
  transition awake  -> asleep = inv_logit(A_AS + ETA_AS)
  transition asleep -> awake  = inv_logit(A_SA)
";

/// Three ordered states written as a cumulative logit: `P(s ≤ k) = inv_logit(α_k + η)`
/// with monotone cut-points `α_1 = A1`, `α_2 = A1 + exp(D2)`, the same from every
/// previous state except for a previous-state shift `G_j` — so every row is a valid
/// ordinal distribution and the stay probabilities are the auto-filled diagonal.
const THREE_STATE: &str = r"
[parameters]
  theta A1(-0.4, -10.0, 10.0)
  theta D2(0.2, -10.0, 10.0)
  theta G1(0.9, -10.0, 10.0)
  theta G2(1.7, -10.0, 10.0)
  omega ETA ~ 0.4
  omega ETA_B ~ 0.2

[markov_model]
  type   = dtmm
  cmt    = 5
  states = [none=0, mild=1, severe=2]
  transition none   -> mild   = inv_logit(A1 + exp(D2) + ETA) - inv_logit(A1 + ETA)
  transition none   -> severe = 1 - inv_logit(A1 + exp(D2) + ETA)
  transition mild   -> none   = inv_logit(A1 - G1 + ETA + ETA_B)
  transition mild   -> severe = 1 - inv_logit(A1 + exp(D2) - G1 + ETA + ETA_B)
  transition severe -> none   = inv_logit(A1 - G2 + ETA)
  transition severe -> mild   = inv_logit(A1 + exp(D2) - G2 + ETA) - inv_logit(A1 - G2 + ETA)
";

/// Hand-computed −log L of a short two-state sequence. Every step kind appears — stay in
/// 0, 0 → 1, stay in 1, 1 → 0 — and the first record is conditioned on.
#[test]
fn two_state_nll_matches_hand_computation() {
    let model = parse_model_string(TWO_STATE).expect("parse");
    let theta = [-0.8_f64, -1.3];
    let eta = [0.35_f64];
    let s = subject(&[(0.0, 0), (1.0, 0), (2.0, 1), (3.0, 1), (4.0, 0)]);
    let p01 = inv_logit(-0.8 + 0.35);
    let p10 = inv_logit(-1.3);
    let want = -((1.0 - p01).ln() + p01.ln() + (1.0 - p10).ln() + p10.ln());
    let got = dtmm_subject_nll(&model, &s, &theta, &eta);
    assert!((got - want).abs() < 1e-13, "got {got}, want {want}");

    // It is the discrete seam's value too: that is what every estimator scores.
    let seam = crate::categorical::discrete_subject_nll(&model, &s, &theta, &eta);
    assert_eq!(seam.to_bits(), got.to_bits());
}

/// The first record carries no previous state, so it contributes nothing: one record
/// scores 0, and changing only the first state moves the objective solely through the
/// first step it starts.
#[test]
fn first_record_is_conditioned_on() {
    let model = parse_model_string(TWO_STATE).expect("parse");
    let (theta, eta) = ([-0.8_f64, -1.3], [0.0_f64]);
    assert_eq!(
        dtmm_subject_nll(&model, &subject(&[(0.0, 1)]), &theta, &eta),
        0.0
    );
    let p01 = inv_logit(-0.8);
    let p10 = inv_logit(-1.3);
    let from0 = dtmm_subject_nll(&model, &subject(&[(0.0, 0), (1.0, 0)]), &theta, &eta);
    let from1 = dtmm_subject_nll(&model, &subject(&[(0.0, 1), (1.0, 0)]), &theta, &eta);
    assert!((from0 + (1.0 - p01).ln()).abs() < 1e-14, "{from0}");
    assert!((from1 + p10.ln()).abs() < 1e-14, "{from1}");
}

/// A step is a record, not a time unit: with no `TIME` term the spacing of the records
/// does not enter the likelihood.
#[test]
fn spacing_does_not_enter_without_time() {
    let model = parse_model_string(TWO_STATE).expect("parse");
    let (theta, eta) = ([-0.8_f64, -1.3], [0.2_f64]);
    let even = subject(&[(0.0, 0), (1.0, 1), (2.0, 1), (3.0, 0)]);
    let uneven = subject(&[(0.0, 0), (0.1, 1), (7.5, 1), (7.5, 0)]);
    assert_eq!(
        dtmm_subject_nll(&model, &even, &theta, &eta).to_bits(),
        dtmm_subject_nll(&model, &uneven, &theta, &eta).to_bits()
    );
}

/// The probabilities of a step are those of the **current** record: a `TIME` term reads
/// the time of the record being scored, not the previous one. The pair straddles the
/// choice — the two readings differ by far more than the tolerance.
#[test]
fn time_term_reads_the_current_record() {
    let src = TWO_STATE.replace("inv_logit(A_SA)", "inv_logit(A_SA + 0.5 * TIME)");
    let model = parse_model_string(&src).expect("parse");
    let (theta, eta) = ([-0.8_f64, -1.3], [0.0_f64]);
    // one step 1 → 0, from TIME 1 to TIME 3
    let got = dtmm_subject_nll(&model, &subject(&[(1.0, 1), (3.0, 0)]), &theta, &eta);
    let current = -inv_logit(-1.3 + 0.5 * 3.0).ln();
    let previous = -inv_logit(-1.3 + 0.5 * 1.0).ln();
    assert!(
        (current - previous).abs() > 0.1,
        "fixture does not straddle"
    );
    assert!(
        (got - current).abs() < 1e-14,
        "got {got}, current-record {current}"
    );
}

/// Three ordered states via the cumulative-logit spelling: the auto-filled diagonal is
/// the middle of each ordinal distribution, checked against a hand computation.
#[test]
fn three_state_cumulative_logit_matches_hand_computation() {
    let model = parse_model_string(THREE_STATE).expect("parse");
    let theta = [-0.4_f64, 0.2, 0.9, 1.7];
    let eta = [0.3_f64, -0.25];
    let (a1, a2) = (-0.4, -0.4 + 0.2_f64.exp());
    let cut = |shift: f64| (inv_logit(a1 + shift), inv_logit(a2 + shift));
    let (c0_0, c1_0) = cut(0.3);
    let (c0_1, c1_1) = cut(0.3 - 0.25 - 0.9);
    let (c0_2, c1_2) = cut(0.3 - 1.7);
    // 0 → 0 → 2 → 1 → 1 → 0 → 1 → 2 → 2
    let s = subject(&[
        (0.0, 0),
        (1.0, 0),
        (2.0, 2),
        (3.0, 1),
        (4.0, 1),
        (5.0, 0),
        (6.0, 1),
        (7.0, 2),
        (8.0, 2),
    ]);
    let probs = [
        c0_0,        // 0 → 0 (stay = P(s ≤ 0 | 0))
        1.0 - c1_0,  // 0 → 2
        c1_2 - c0_2, // 2 → 1
        c1_1 - c0_1, // 1 → 1 (stay)
        c0_1,        // 1 → 0
        c1_0 - c0_0, // 0 → 1
        1.0 - c1_1,  // 1 → 2
        1.0 - c1_2,  // 2 → 2 (stay)
    ];
    let want: f64 = -probs.iter().map(|p| p.ln()).sum::<f64>();
    let got = dtmm_subject_nll(&model, &s, &theta, &eta);
    assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
}

/// A matrix whose declared probabilities leave `[0, 1]`, or whose row sums pass 1 — in
/// **any** row, not only the one the step reads — is the penalized regime: the subject repels
/// with the sentinel. Each fixture is paired with a valid control on the same data so the
/// sentinel cannot be the answer for every input.
#[test]
fn invalid_matrix_repels() {
    let probe = |p01: &str, p02: &str, p10: &str, states: &[(f64, usize)]| {
        let src = format!(
            "
[parameters]
  theta A(0.2, -10.0, 10.0)

[markov_model]
  type   = dtmm
  cmt    = 5
  states = [0, 1, 2]
  transition 0 -> 1 = {p01}
  transition 0 -> 2 = {p02}
  transition 1 -> 0 = {p10}
  transition 1 -> 2 = 0.3
"
        );
        let model = parse_model_string(&src).expect("parse");
        let s = subject(states);
        let v = dtmm_subject_nll(&model, &s, &[0.2], &[]);
        // the analytic path declines exactly where the value is the sentinel
        let g = dtmm_subject_eta_grad(&model, &s, &[0.2], &[]);
        assert_eq!(
            g.is_none(),
            v == SUBJECT_SENTINEL_NLL,
            "{p01} {p02} {p10}: {v}"
        );
        v
    };
    let chain = [(0.0, 0), (1.0, 1)];
    // the row being stepped from: P01 > 1
    assert_eq!(probe("A + 1", "0.1", "A", &chain), SUBJECT_SENTINEL_NLL);
    // ... P02 < 0, though the observed step 0 → 1 has a fine probability
    assert_eq!(probe("A", "A - 0.5", "A", &chain), SUBJECT_SENTINEL_NLL);
    // ... off-diagonals sum past 1 (a negative stay probability)
    assert_eq!(probe("0.6", "0.5", "A", &chain), SUBJECT_SENTINEL_NLL);
    // a stay with an off-diagonal sum of exactly 1 has probability 0
    assert_eq!(
        probe("0.5", "0.5", "A", &[(0.0, 0), (1.0, 0)]),
        SUBJECT_SENTINEL_NLL
    );
    // a row the step does NOT read: P10 > 1, the chain only ever leaves state 0
    assert_eq!(probe("A", "0.1", "A + 1", &chain), SUBJECT_SENTINEL_NLL);
    // ... row 1's off-diagonals 0.8 + 0.3 sum past 1
    assert_eq!(probe("A", "0.1", "0.8", &chain), SUBJECT_SENTINEL_NLL);
    // ... a NaN in row 1
    assert_eq!(
        probe("A", "0.1", "log(A - 1)", &chain),
        SUBJECT_SENTINEL_NLL
    );
    // control: the same data under a valid matrix
    let ok = probe("A", "0.1", "A", &chain);
    assert!((ok + 0.2_f64.ln()).abs() < 1e-14, "control {ok}");
}

/// The exact η-gradient (dual probability program chained through `∂(−log P)/∂P`) equals
/// a central FD of the very NLL the fit minimizes — on off-diagonal *and* stay steps, on
/// both η axes, away from η = 0.
#[test]
fn eta_grad_matches_fd() {
    for (src, theta, s) in [
        (
            TWO_STATE,
            vec![-0.8_f64, -1.3],
            subject(&[(0.0, 0), (1.0, 0), (2.0, 1), (3.0, 1), (4.0, 0), (5.0, 0)]),
        ),
        (
            THREE_STATE,
            vec![-0.4_f64, 0.2, 0.9, 1.7],
            subject(&[
                (0.0, 0),
                (1.0, 0),
                (2.0, 2),
                (3.0, 1),
                (4.0, 1),
                (5.0, 0),
                (6.0, 1),
                (7.0, 2),
                (8.0, 2),
            ]),
        ),
    ] {
        let model = parse_model_string(src).expect("parse");
        let n_eta = model.n_eta;
        for scale in [-1.0_f64, 0.5, 1.3] {
            let eta: Vec<f64> = (0..n_eta).map(|i| scale * (0.3 + 0.2 * i as f64)).collect();
            let (v, g) = dtmm_subject_eta_grad(&model, &s, &theta, &eta)
                .expect("a dual-evaluable DTMM is served analytically");
            let v_ref = dtmm_subject_nll(&model, &s, &theta, &eta);
            assert!((v - v_ref).abs() < 1e-12, "value drift {v} vs {v_ref}");
            for i in 0..n_eta {
                let h = 1e-6;
                let mut ep = eta.clone();
                let mut em = eta.clone();
                ep[i] += h;
                em[i] -= h;
                let fd = (dtmm_subject_nll(&model, &s, &theta, &ep)
                    - dtmm_subject_nll(&model, &s, &theta, &em))
                    / (2.0 * h);
                assert!(
                    (g[i] - fd).abs() < 1e-7 * fd.abs().max(1.0),
                    "η{i} = {}: analytic {}, FD {fd}",
                    eta[i],
                    g[i]
                );
                assert!(g[i].abs() > 1e-3, "η{i}: gradient suspiciously flat");
            }
            // The seam returns the same block.
            let (vs, gs) =
                crate::categorical::discrete_subject_eta_grad(&model, &s, &theta, &eta).unwrap();
            assert_eq!(vs.to_bits(), v.to_bits());
            assert_eq!(gs, g);
        }
    }
}

/// The penalized regime has no derivative: the analytic path declines (→ FD) instead of
/// returning a gradient of the flat sentinel.
#[test]
fn eta_grad_declines_on_an_invalid_row() {
    let model = parse_model_string(TWO_STATE).expect("parse");
    let s = subject(&[(0.0, 1), (1.0, 0)]);
    // A_SA so large that inv_logit rounds to exactly 1: the 1 → 1 stay is impossible…
    let stay = subject(&[(0.0, 1), (1.0, 1)]);
    assert!(dtmm_subject_eta_grad(&model, &stay, &[-0.8, 40.0], &[0.0]).is_none());
    // …while the observed step on the same parameters is served.
    assert!(dtmm_subject_eta_grad(&model, &s, &[-0.8, 40.0], &[0.0]).is_some());
}

/// The DTMM model is routed to the analytic inner gradient (its program exists), and a
/// time-reading probability still is (the program carries `TIME`).
#[test]
fn dtmm_routes_to_the_analytic_inner_gradient() {
    for src in [
        TWO_STATE.to_string(),
        TWO_STATE.replace("inv_logit(A_SA)", "inv_logit(A_SA + 0.5 * TIME)"),
    ] {
        let model = parse_model_string(&src).expect("parse");
        assert!(model.has_dtmm() && model.has_discrete() && !model.has_ctmm());
        assert!(all_dtmm_endpoints_analytic(&model));
        assert!(crate::estimation::inner_optimizer::analytic_inner_grad_supported_model(&model));
    }
}

/// The TIME-reading program agrees with the value path record by record (the same
/// model-time scope on both).
#[test]
fn time_reading_eta_grad_matches_fd() {
    let src = TWO_STATE.replace(
        "inv_logit(A_AS + ETA_AS)",
        "inv_logit(A_AS + ETA_AS - 0.3 * TIME)",
    );
    let model = parse_model_string(&src).expect("parse");
    let s = subject(&[(0.0, 0), (1.0, 0), (2.5, 1), (3.0, 0), (6.0, 0)]);
    let theta = [-0.8_f64, -1.3];
    let eta = [0.4_f64];
    let (_, g) = dtmm_subject_eta_grad(&model, &s, &theta, &eta).unwrap();
    let h = 1e-6;
    let fd = (dtmm_subject_nll(&model, &s, &theta, &[eta[0] + h])
        - dtmm_subject_nll(&model, &s, &theta, &[eta[0] - h]))
        / (2.0 * h);
    assert!((g[0] - fd).abs() < 1e-7, "analytic {}, FD {fd}", g[0]);
}

#[test]
fn validate_rejects_an_undeclared_step_and_names_it() {
    // only 0 → 1 is declared
    let transitions = [(0, 1)];
    let codes = [3, 7];
    let ok = [
        disc(0.0, 3, 5),
        disc(1.0, 3, 5),
        disc(2.0, 7, 5),
        disc(3.0, 7, 5),
    ];
    assert!(validate_dtmm_records(5, &codes, &transitions, &ok).is_ok());
    let bad = [disc(0.0, 3, 5), disc(1.0, 7, 5), disc(2.0, 3, 5)];
    let err = validate_dtmm_records(5, &codes, &transitions, &bad).unwrap_err();
    assert!(err.contains("from state 7 to state 3"), "{err}");
    assert!(err.contains("TIME 2"), "{err}");
    assert!(err.contains("cmt = 5"), "{err}");
    // an undeclared code and decreasing times keep the CTMM checks' messages
    let err = validate_dtmm_records(5, &codes, &transitions, &[disc(0.0, 4, 5)]).unwrap_err();
    assert!(err.contains("DV 4"), "{err}");
    let err = validate_dtmm_records(5, &codes, &transitions, &[disc(1.0, 3, 5), disc(0.0, 3, 5)])
        .unwrap_err();
    assert!(err.contains("non-decreasing"), "{err}");
    // rows on another CMT are not part of this chain
    assert!(
        validate_dtmm_records(5, &codes, &transitions, &[disc(0.0, 7, 5), disc(1.0, 3, 6)]).is_ok()
    );
}

#[test]
fn parser_builds_a_dtmm_endpoint() {
    let model = parse_model_string(THREE_STATE).expect("parse");
    let ep = model.endpoints.get(&5).expect("cmt 5");
    let EndpointLikelihood::Dtmm {
        n_states,
        state_codes,
        transitions,
        prob_program,
        prob_covariates,
        ..
    } = ep
    else {
        panic!("not a DTMM: {ep:?}");
    };
    assert_eq!(*n_states, 3);
    assert_eq!(state_codes, &[0, 1, 2]);
    assert_eq!(
        transitions,
        &[(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    );
    assert!(prob_program.is_some());
    assert!(prob_covariates.is_empty());
    // A block with no `type` line is still a CTMM.
    let ctmm = parse_model_string(&TWO_STATE.replace("  type   = dtmm\n", "")).expect("parse");
    assert!(ctmm.has_ctmm() && !ctmm.has_dtmm());
}

/// One-compartment ODE PK with a drug-driven DTMM: the awake → asleep probability reads the
/// concentration `central / V`. Refused at parse (see below).
const DRUG_DRIVEN: &str = r"
[parameters]
  theta TVCL(2.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta A(-1.5, -6.0, 3.0)
  theta S(0.4, -5.0, 5.0)
  theta B(-0.5, -6.0, 3.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -CL/V * central

[error_model]
  DV ~ proportional(PROP)

[markov_model]
  type   = dtmm
  cmt    = 5
  states = [awake=0, asleep=1]
  transition awake  -> asleep = inv_logit(A + S * central / V)
  transition asleep -> awake  = inv_logit(B)
";

#[test]
fn parser_refuses_a_drug_driven_dtmm_and_mctmm() {
    // a probability reading an ODE state — directly, or through an [individual_parameters]
    // value — is refused by name: a drug-driven DTMM needs same-TIME record order (#2473)
    let via_indiv = DRUG_DRIVEN
        .replace("  V  = TVV\n", "  V  = TVV\n  CC = central / V\n")
        .replace("S * central / V", "S * CC");
    // a state name that is also a θ or an η parses to the parameter leaf; refused rather than
    // silently read as the constant parameter
    let theta_named_like_the_state = DRUG_DRIVEN
        .replace("theta A(", "theta central(")
        .replace("A + S * central / V", "central + S");
    let eta_named_like_the_state = DRUG_DRIVEN
        .replace("ETA_CL", "central")
        .replace("A + S * central / V", "A + S * central");
    for src in [
        DRUG_DRIVEN.to_string(),
        via_indiv,
        theta_named_like_the_state,
        eta_named_like_the_state,
    ] {
        let err = parse_model_string(&src).unwrap_err();
        for needle in [
            "transition probability reads the model state `central`",
            "A drug-driven DTMM is not yet supported",
            "same-TIME record order against doses",
            "may read θ, η, covariates and TIME",
            "Use a continuous-time `type = ctmm`",
            "(tracked in #2510)",
        ] {
            assert!(err.contains(needle), "missing {needle:?} in: {err}");
        }
    }
    // the twin reading no state parses, ODE system and all
    let stateless = DRUG_DRIVEN.replace("inv_logit(A + S * central / V)", "inv_logit(A + S)");
    assert!(parse_model_string(&stateless).is_ok());

    let err =
        parse_model_string(&TWO_STATE.replace("type   = dtmm", "type   = mctmm")).unwrap_err();
    assert!(err.contains("remove the `transition` lines"), "{err}");
    // an mCTMM key (#2474) on a DTMM is refused, not ignored
    let err = parse_model_string(&TWO_STATE.replace("  cmt    = 5\n", "  cmt    = 5\n  tau = 2\n"))
        .unwrap_err();
    assert!(err.contains("`tau` is a minimal-CTMM key"), "{err}");
    let err = parse_model_string(&TWO_STATE.replace("type   = dtmm", "type   = hmm")).unwrap_err();
    assert!(err.contains("expected `ctmm`, `dtmm` or `mctmm`"), "{err}");
}

/// The DOSE model of the covariate tests: the asleep → awake probability reads `DOSE`.
fn dose_model() -> CompiledModel {
    parse_model_string(&TWO_STATE.replace("inv_logit(A_SA)", "inv_logit(A_SA + 0.6 * DOSE)"))
        .expect("parse")
}

fn dose_cov(v: f64) -> HashMap<String, f64> {
    HashMap::from([("DOSE".to_string(), v)])
}

fn population_of(s: Subject) -> crate::types::Population {
    crate::types::Population {
        subjects: vec![s],
        covariate_names: vec!["DOSE".to_string()],
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

/// A covariate that changes within a subject is read at the **record being scored**: the
/// step 1 → 0 lands on the record where `DOSE` changes, so the current-record and the
/// previous-record readings straddle it. Fit setup accepts it — and refuses the same subject
/// once its snapshot cannot be aligned with its records, rather than fall back to the
/// baseline.
#[test]
fn time_varying_covariate_reads_the_current_record() {
    let model = dose_model();
    let (theta, eta) = ([-0.8_f64, -1.3], [0.1_f64]);
    let mut s = subject(&[(0.0, 0), (1.0, 1), (2.0, 0), (3.0, 0)]);
    s.covariates = dose_cov(1.0);
    s.endpoint_covariates = vec![dose_cov(1.0), dose_cov(1.0), dose_cov(3.0), dose_cov(3.0)];
    let p01 = inv_logit(-0.8 + 0.1);
    let p10 = |dose: f64| inv_logit(-1.3 + 0.6 * dose);
    let want = -(p01.ln() + p10(3.0).ln() + (1.0 - p01).ln());
    let previous = -(p01.ln() + p10(1.0).ln() + (1.0 - p01).ln());
    assert!((want - previous).abs() > 0.1, "fixture does not straddle");
    let got = dtmm_subject_nll(&model, &s, &theta, &eta);
    assert!(
        (got - want).abs() < 1e-14,
        "got {got}, current record {want}"
    );
    // the analytic gradient reads the same record covariates
    let (v, _) = dtmm_subject_eta_grad(&model, &s, &theta, &eta).unwrap();
    assert_eq!(v.to_bits(), got.to_bits());

    assert!(crate::api::check_survival_tv_covariates(&model, &population_of(s.clone())).is_empty());

    // misaligned: a snapshot one short of the records
    let mut short = s.clone();
    short.endpoint_covariates.pop();
    assert!(short.record_covariate_snapshots().is_none());
    // misaligned: a time-to-event record on the subject
    let mut with_tte = s.clone();
    with_tte.obs_records.push(ObsRecord::Event {
        time: 4.0,
        event_type: crate::types::EventType::RightCensored,
        entry_time: 0.0,
        cmt: 7,
    });
    with_tte.endpoint_covariates.push(dose_cov(3.0));
    assert!(with_tte.record_covariate_snapshots().is_none());
    for bad in [short, with_tte] {
        let refused = crate::api::check_survival_tv_covariates(&model, &population_of(bad));
        assert_eq!(refused.len(), 1, "{refused:?}");
        assert_eq!(refused[0].code, "E_ENDPOINT_TV_COVARIATE");
        assert!(
            refused[0].message.contains("type = dtmm"),
            "{}",
            refused[0].message
        );
        assert!(
            refused[0].message.contains("`DOSE`"),
            "{}",
            refused[0].message
        );
    }
    // the same covariate held constant needs no snapshot
    let mut constant = subject(&[(0.0, 0), (1.0, 1), (2.0, 0)]);
    constant.covariates = dose_cov(1.0);
    assert!(crate::api::check_survival_tv_covariates(&model, &population_of(constant)).is_empty());
}

/// The data reader's per-row snapshot is what aligns: read through the real CSV reader, a
/// `DOSE` that changes **on a state row** (and a missing-DV row inside the chain) reaches
/// the probability of the step that lands there.
#[test]
fn reader_snapshots_align_with_state_records() {
    let dir = std::env::temp_dir().join(format!("ferx_dtmm_tvcov_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let model_path = dir.join("m.ferx");
    let data_path = dir.join("d.csv");
    std::fs::write(
        &model_path,
        TWO_STATE.replace("inv_logit(A_SA)", "inv_logit(A_SA + 0.6 * DOSE)"),
    )
    .unwrap();
    std::fs::write(
        &data_path,
        "ID,TIME,DV,MDV,CMT,DOSE\n\
         1,0,0,0,5,1\n\
         1,1,1,0,5,1\n\
         1,1.5,.,1,5,2\n\
         1,2,0,0,5,3\n\
         1,3,0,0,5,3\n",
    )
    .unwrap();
    let run = crate::api::prepare_run(
        model_path.to_str().unwrap(),
        Some(data_path.to_str().unwrap()),
    )
    .expect("load");
    let _ = std::fs::remove_dir_all(&dir);
    let s = &run.population.subjects[0];
    let snaps = s
        .record_covariate_snapshots()
        .expect("a time-varying DOSE yields aligned snapshots");
    let doses: Vec<f64> = snaps.iter().map(|c| c["DOSE"]).collect();
    assert_eq!(doses, [1.0, 1.0, 3.0, 3.0]);
    let (theta, eta) = ([-0.8_f64, -1.3], [0.0_f64]);
    let p01 = inv_logit(-0.8);
    let want = -(p01.ln() + inv_logit(-1.3 + 0.6 * 3.0).ln() + (1.0 - p01).ln());
    let got = dtmm_subject_nll(&run.parsed.model, s, &theta, &eta);
    assert!((got - want).abs() < 1e-14, "got {got}, want {want}");
    assert!(
        crate::api::check_survival_tv_covariates(&run.parsed.model, &run.population).is_empty()
    );
}

/// A valid mode on the stochastic boundary — `P01 = A + η²` at `A = 1`, `η = 0`: every
/// perturbation in either direction leaves `[0, 1]`, so no stencil could measure it. The
/// plan reads the curvature at the mode alone, so the Laplace fold scores it:
/// `∂²/∂η²[−2·log(1 + η²)] = −4` at `η = 0`, and the objective is ordinary. A twin whose
/// transition has no program cannot record its term and is repelled rather than scored
/// with no curvature. The bounds straddle: sentinel `1e20` vs an objective below `1e3`.
#[test]
fn laplace_fold_takes_the_exact_curvature_at_a_boundary_mode() {
    let src = r"
[parameters]
  theta A(1.0, -10.0, 10.0)
  theta B(0.3, -10.0, 10.0)
  omega ETA ~ 0.1

[markov_model]
  type   = dtmm
  cmt    = 5
  states = [0, 1]
  transition 0 -> 1 = A + ETA * ETA
  transition 1 -> 0 = B
";
    let model = parse_model_string(src).expect("parse");
    let mut unrecorded = parse_model_string(src).expect("parse");
    for ep in unrecorded.endpoints.values_mut() {
        if let EndpointLikelihood::Dtmm { prob_program, .. } = ep {
            *prob_program = None;
        }
    }
    let s = subject(&[(0.0, 0), (1.0, 1), (2.0, 0), (3.0, 1)]);
    let eta = nalgebra::DVector::from_element(1, 0.0);
    let p = &model.default_params;
    let theta = [1.0_f64, 0.3];
    let fold = |m: &CompiledModel| {
        crate::stats::likelihood::foce_subject_nll(
            m,
            &s,
            &theta,
            &eta,
            &nalgebra::DMatrix::zeros(0, 1),
            &p.omega,
            &p.sigma.values,
            &p.residual_correlations,
            true,
        )
    };
    assert!(mode_is_valid(&model, &s, 1.0), "the mode itself is valid");
    let f = |e: &[f64]| dtmm_subject_nll(&model, &s, &theta, e);
    assert!(
        f(&[1e-3]) >= 1e20 && f(&[-1e-3]) >= 1e20,
        "the fixture's mode sits on the boundary, both sides repelled"
    );

    let (_, h) = crate::categorical::discrete_subject_eta_hessian(&model, &s, &theta, &[0.0])
        .expect("the exact jet serves a valid mode");
    assert!((h[(0, 0)] + 4.0).abs() < 1e-12, "exact −4, got {h}");
    let exact = fold(&model);
    assert!(
        exact < 1e3,
        "the exact curvature gives an ordinary objective: {exact}"
    );

    let repelled = fold(&unrecorded);
    assert!(
        repelled >= 1e20,
        "without a program the term does not record and repels: {repelled}"
    );
}

fn mode_is_valid(model: &CompiledModel, s: &Subject, a: f64) -> bool {
    dtmm_subject_nll(model, s, &[a, 0.3], &[0.0]) < 1e3
}

/// The discrete seam's value is exactly its binary and DTMM halves — which is what lets the
/// Laplace fold score them separately (guarding only the DTMM block) without changing the
/// objective at the mode.
#[test]
fn discrete_seam_is_the_sum_of_its_binary_and_dtmm_halves() {
    let src = format!("{TWO_STATE}\n[binary_model]\n  cmt   = 6\n  logit = A_AS + ETA_AS\n");
    let model = parse_model_string(&src).expect("parse");
    assert!(model.has_binary() && model.has_dtmm());
    let mut s = subject(&[(0.0, 0), (1.0, 1), (2.0, 1)]);
    s.obs_records.push(disc(0.5, 1, 6));
    s.obs_records.push(disc(1.5, 0, 6));
    let (theta, eta) = ([-0.8_f64, -1.3], [0.2_f64]);
    let b = crate::categorical::binary_subject_nll(&model, &s, &theta, &eta);
    let d = dtmm_subject_nll(&model, &s, &theta, &eta);
    assert!(b > 0.0 && d > 0.0);
    assert_eq!(
        crate::categorical::discrete_subject_nll(&model, &s, &theta, &eta).to_bits(),
        (d + b).to_bits()
    );
}

/// Load `model_src` and `csv` through the real reader (a temp directory per call).
fn read_through_reader(tag: &str, model_src: &str, csv: &str) -> crate::api::PreparedRun {
    let dir = std::env::temp_dir().join(format!("ferx_dtmm_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (model_path, data_path) = (dir.join("m.ferx"), dir.join("d.csv"));
    std::fs::write(&model_path, model_src).unwrap();
    std::fs::write(&data_path, csv).unwrap();
    let run = crate::api::prepare_run(
        model_path.to_str().unwrap(),
        Some(data_path.to_str().unwrap()),
    )
    .expect("load");
    let _ = std::fs::remove_dir_all(&dir);
    run
}

/// A probability's `TIME` is the record's TIME **as written**, as a `$PRED` reads it — not
/// the engine clock, which the reader stacks across a reset occasion that restarts `TIME`.
/// The step after the reset lands at written TIME 1 but engine time ≈ 4, and the two
/// readings of `exp(0.5·TIME)` straddle any tolerance.
#[test]
fn time_reads_the_written_clock_across_a_reset() {
    let model = TWO_STATE.replace("inv_logit(A_SA)", "inv_logit(A_SA + 0.5 * TIME)");
    let csv = "ID,TIME,DV,AMT,EVID,MDV,CMT\n\
               1,0,0,.,0,0,5\n\
               1,1,1,.,0,0,5\n\
               1,2,1,.,0,0,5\n\
               1,0,.,.,3,1,5\n\
               1,1,0,.,0,0,5\n";
    let run = read_through_reader("reset", &model, csv);
    let s = &run.population.subjects[0];
    let times: Vec<(f64, f64)> = s
        .obs_records
        .iter()
        .filter_map(|r| match r {
            ObsRecord::DiscreteState { time, raw_time, .. } => Some((*time, *raw_time)),
            _ => None,
        })
        .collect();
    let (engine, written) = times[3];
    assert_eq!(written, 1.0);
    assert!(
        engine > 2.5,
        "the reader stacks the occasion: engine time {engine}"
    );

    let (theta, eta) = ([-0.8_f64, -1.3], [0.0_f64]);
    let p01 = inv_logit(-0.8);
    let p10 = |t: f64| inv_logit(-1.3 + 0.5 * t);
    // steps: 0→1 at 1, 1→1 at 2, 1→0 at written 1
    let at = |t_last: f64| -(p01.ln() + (1.0 - p10(2.0)).ln() + p10(t_last).ln());
    let (want, engine_clock) = (at(written), at(engine));
    assert!(
        (want - engine_clock).abs() > 0.1,
        "fixture does not straddle"
    );
    let got = dtmm_subject_nll(&run.parsed.model, s, &theta, &eta);
    assert!(
        (got - want).abs() < 1e-14,
        "got {got}, written clock {want}"
    );
    let (v, _) = dtmm_subject_eta_grad(&run.parsed.model, s, &theta, &eta).unwrap();
    assert_eq!(
        v.to_bits(),
        got.to_bits(),
        "the analytic path reads the same clock"
    );
}
