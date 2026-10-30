use super::*;
use crate::parser::model_parser::parse_model_string;

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

/// A row whose declared probabilities leave `[0, 1]`, or sum past 1, is the penalized
/// regime: the subject repels with the sentinel. Each fixture is paired with a valid
/// control on the same data so the sentinel cannot be the answer for every input.
#[test]
fn invalid_row_repels() {
    let probe = |rhs_as: &str, rhs_sa: &str, states: &[(f64, usize)]| {
        let src = format!(
            "
[parameters]
  theta A(0.2, -10.0, 10.0)

[markov_model]
  type   = dtmm
  cmt    = 5
  states = [0, 1, 2]
  transition 0 -> 1 = {rhs_as}
  transition 0 -> 2 = {rhs_sa}
  transition 1 -> 0 = A
"
        );
        let model = parse_model_string(&src).expect("parse");
        dtmm_subject_nll(&model, &subject(states), &[0.2], &[])
    };
    let chain = [(0.0, 0), (1.0, 1)];
    // P01 > 1
    assert_eq!(probe("A + 1", "0.1", &chain), SUBJECT_SENTINEL_NLL);
    // P02 < 0, though the observed step 0 → 1 has a fine probability
    assert_eq!(probe("A", "A - 0.5", &chain), SUBJECT_SENTINEL_NLL);
    // off-diagonals sum past 1 (a negative stay probability)
    assert_eq!(probe("0.6", "0.5", &chain), SUBJECT_SENTINEL_NLL);
    // a stay with an off-diagonal sum of exactly 1 has probability 0
    assert_eq!(
        probe("0.5", "0.5", &[(0.0, 0), (1.0, 0)]),
        SUBJECT_SENTINEL_NLL
    );
    // control: the same data under a valid row
    let ok = probe("A", "0.1", &chain);
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

#[test]
fn parser_refuses_a_drug_driven_dtmm_and_mctmm() {
    let src = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta A(-0.7, -6.0, 3.0)
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
  states = [s0=0, s1=1]
  transition s0 -> s1 = inv_logit(A + central / V)
";
    let err = parse_model_string(src).unwrap_err();
    assert!(err.contains("model state `central`"), "{err}");
    assert!(err.contains("type = ctmm"), "{err}");

    let err =
        parse_model_string(&TWO_STATE.replace("type   = dtmm", "type   = mctmm")).unwrap_err();
    assert!(err.contains("`mctmm` is not yet supported"), "{err}");
    let err = parse_model_string(&TWO_STATE.replace("type   = dtmm", "type   = hmm")).unwrap_err();
    assert!(err.contains("expected `ctmm` or `dtmm`"), "{err}");
}

/// A covariate that changes within a subject is refused on a DTMM transition probability
/// (it would be read at the baseline, so a step after the change would use the wrong
/// value), while the same covariate held constant is accepted. The pair straddles the gate:
/// the two subjects differ only in whether `DOSE` steps between records.
#[test]
fn time_varying_covariate_on_a_probability_is_refused() {
    let src = TWO_STATE.replace("inv_logit(A_SA)", "inv_logit(A_SA + 0.4 * DOSE)");
    let model = parse_model_string(&src).expect("parse");
    let dose = |v: f64| std::collections::HashMap::from([("DOSE".to_string(), v)]);
    let pop = |second: f64| {
        let mut s = subject(&[(0.0, 0), (1.0, 1), (2.0, 0)]);
        s.covariates = dose(1.0);
        s.endpoint_covariates = vec![dose(1.0), dose(second), dose(second)];
        crate::types::Population {
            subjects: vec![s],
            covariate_names: vec!["DOSE".to_string()],
            dv_column: "DV".to_string(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        }
    };
    let refused = crate::api::check_survival_tv_covariates(&model, &pop(2.0));
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
    assert!(crate::api::check_survival_tv_covariates(&model, &pop(1.0)).is_empty());
}
