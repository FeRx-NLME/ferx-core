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
/// concentration `central / V`.
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
fn parser_accepts_a_drug_driven_dtmm_and_refuses_mctmm() {
    let model = parse_model_string(DRUG_DRIVEN).expect("parse");
    let Some(EndpointLikelihood::Dtmm {
        prob_states,
        prob_covariates,
        prob_program,
        ..
    }) = model.endpoints.get(&5)
    else {
        panic!("not a DTMM");
    };
    assert_eq!(prob_states, &[("central".to_string(), 0)]);
    // the state is not a data column
    assert!(prob_covariates.is_empty(), "{prob_covariates:?}");
    assert!(!model.referenced_covariates.contains(&"central".to_string()));
    // its η-dependence runs through the ODE trajectory, which a program does not see
    assert!(prob_program.is_none());
    assert!(!all_dtmm_endpoints_analytic(&model));

    // a state reached only through an [individual_parameters] value is refused
    let via_indiv = DRUG_DRIVEN
        .replace("  V  = TVV\n", "  V  = TVV\n  CC = central / V\n")
        .replace("S * central / V", "S * CC");
    let err = parse_model_string(&via_indiv).unwrap_err();
    assert!(err.contains("reads the model state `central`"), "{err}");

    let err =
        parse_model_string(&TWO_STATE.replace("type   = dtmm", "type   = mctmm")).unwrap_err();
    assert!(err.contains("`mctmm` is not yet supported"), "{err}");
    let err = parse_model_string(&TWO_STATE.replace("type   = dtmm", "type   = hmm")).unwrap_err();
    assert!(err.contains("expected `ctmm` or `dtmm`"), "{err}");
}

/// A drug-driven probability reads the ODE state **at the record being scored**. Two IV
/// boluses, so the step at `t = 10` lands on drug from both (the incoming side of the second
/// dose is live). The oracle is the closed-form one-compartment superposition, outside the
/// ODE engine; the differential twin reads the concentration at the *previous* record and
/// must be far away.
#[test]
fn drug_driven_probability_reads_the_state_at_the_current_record() {
    use crate::types::DoseEvent;
    let model = parse_model_string(DRUG_DRIVEN).expect("parse");
    let theta = [2.0_f64, 10.0, -1.5, 0.4, -0.5];
    let eta = [0.2_f64];
    let (cl, v) = (2.0 * 0.2_f64.exp(), 10.0);
    let conc = |t: f64| {
        [(0.0, 100.0), (6.0, 100.0)]
            .iter()
            .filter(|(td, _)| *td <= t)
            .map(|(td, amt)| amt / v * (-(cl / v) * (t - td)).exp())
            .sum::<f64>()
    };
    let mut s = subject(&[(1.0, 0), (3.0, 1), (7.0, 0), (10.0, 0), (12.0, 1)]);
    s.doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(6.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let p01 = |c: f64| inv_logit(-1.5 + 0.4 * c);
    let p10 = inv_logit(-0.5);
    // steps: 0→1 at 3, 1→0 at 7, 0→0 at 10, 0→1 at 12
    let at = |t3: f64, t10: f64, t12: f64| {
        -(p01(conc(t3)).ln() + p10.ln() + (1.0 - p01(conc(t10))).ln() + p01(conc(t12)).ln())
    };
    let want = at(3.0, 10.0, 12.0);
    let previous_record = at(1.0, 7.0, 10.0);
    let got = dtmm_subject_nll(&model, &s, &theta, &eta);
    eprintln!(
        "drug-driven DTMM vs closed form: |Δ| = {:.3e}",
        (got - want).abs()
    );
    // measured |got − want| = 4.4e-16 (macOS arm64): a linear `[odes]` system is solved by
    // the exact matrix-exponential path, so the bound is round-off, not solver tolerance.
    assert!((got - want).abs() < 1e-12, "got {got}, closed form {want}");
    assert!(
        (previous_record - want).abs() > 0.1,
        "the fixture does not straddle the record choice: {previous_record} vs {want}"
    );
    // the drug must move the objective, or the comparison would hold for a model that
    // ignored `central`
    let drug_free = dtmm_subject_nll(
        &model,
        &subject(&[(1.0, 0), (3.0, 1), (7.0, 0), (10.0, 0), (12.0, 1)]),
        &theta,
        &eta,
    );
    assert!(
        (drug_free - want).abs() > 0.1,
        "drug-free {drug_free} vs {want}"
    );
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
    assert!(record_covariate_snapshots(&short).is_none());
    // misaligned: a time-to-event record on the subject
    let mut with_tte = s.clone();
    with_tte.obs_records.push(ObsRecord::Event {
        time: 4.0,
        event_type: crate::types::EventType::RightCensored,
        entry_time: 0.0,
        cmt: 7,
    });
    with_tte.endpoint_covariates.push(dose_cov(3.0));
    assert!(record_covariate_snapshots(&with_tte).is_none());
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
    let snaps =
        record_covariate_snapshots(s).expect("a time-varying DOSE yields aligned snapshots");
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

/// A drug-driven DTMM reads an `[odes]` solve run under the baseline individual parameters,
/// so — as for the drug-driven CTMM — a disposition that reads `TIME`, or a covariate that
/// varies within the subject, and reaches the referenced state is refused. Each refusal is
/// paired with its control (the same model without the dependence) so neither can pass by
/// refusing everything.
#[test]
fn drug_driven_dtmm_refuses_a_frozen_disposition() {
    // TIME on CL
    let timed = DRUG_DRIVEN.replace(
        "CL = TVCL * exp(ETA_CL)",
        "CL = TVCL * exp(ETA_CL) * (1 + 0.01 * TIME)",
    );
    let m = parse_model_string(&timed).expect("parse");
    let msg = crate::api::check_ctmm_time_dependent_pk(&m).expect("refused");
    assert!(msg.contains("DTMM transition model"), "{msg}");
    assert!(msg.contains("`CL`"), "{msg}");
    let control = parse_model_string(DRUG_DRIVEN).expect("parse");
    assert!(crate::api::check_ctmm_time_dependent_pk(&control).is_none());

    // a time-varying WT on CL
    let wt = DRUG_DRIVEN.replace(
        "CL = TVCL * exp(ETA_CL)",
        "CL = TVCL * exp(ETA_CL) * WT / 70",
    );
    let m = parse_model_string(&wt).expect("parse");
    let pop = |second: f64| {
        let mut s = subject(&[(1.0, 0), (3.0, 1), (7.0, 0)]);
        let w = |v: f64| HashMap::from([("WT".to_string(), v)]);
        s.covariates = w(70.0);
        s.endpoint_covariates = vec![w(70.0), w(second), w(second)];
        let mut p = population_of(s);
        p.covariate_names = vec!["WT".to_string()];
        p
    };
    let refused = crate::api::check_survival_tv_covariates(&m, &pop(80.0));
    assert_eq!(refused.len(), 1, "{refused:?}");
    assert!(
        refused[0].message.contains("drug-driven"),
        "{}",
        refused[0].message
    );
    assert!(
        refused[0].message.contains("`CL`"),
        "{}",
        refused[0].message
    );
    assert!(crate::api::check_survival_tv_covariates(&m, &pop(70.0)).is_empty());
}

/// A valid DTMM mode next to the stochastic boundary puts a Laplace stencil point outside
/// it, where the subject scores the `1e20` sentinel. Folded unguarded, that is a curvature
/// of order `1e20 / h²` in `log|H̃|`; the shared Markov guard drops the block instead. The
/// pair straddles the boundary: the same model at η = 0 is folded, bit-identical to the
/// unguarded Hessian.
#[test]
fn laplace_curvature_is_guarded_against_the_sentinel() {
    use crate::survival::{data_term_hessian_fd, shi_step_sizes};
    let model = parse_model_string(
        r"
[parameters]
  theta A(0.5, -10.0, 10.0)
  theta B(0.3, -10.0, 10.0)
  omega ETA ~ 0.1

[markov_model]
  type   = dtmm
  cmt    = 5
  states = [0, 1]
  transition 0 -> 1 = A + ETA
  transition 1 -> 0 = B
",
    )
    .expect("parse");
    let s = subject(&[(0.0, 0), (1.0, 1), (2.0, 0), (3.0, 1)]);
    let theta = [0.5_f64, 0.3];
    let f = |e: &[f64]| dtmm_subject_nll(&model, &s, &theta, e);
    let unguarded = |eta: &[f64]| {
        let steps = shi_step_sizes(f, eta);
        data_term_hessian_fd(f, eta, &steps)
    };

    // P01 = 1 − 1e-9 at the mode: valid, but one stencil point leaves [0, 1]
    let edge = [0.5 - 1e-9];
    let mode = f(&edge);
    assert!(
        mode < 1e3,
        "the mode itself is valid, not the sentinel: {mode}"
    );
    assert!(
        unguarded(&edge)[(0, 0)].abs() > 1e18,
        "the fixture must put a stencil point on the sentinel, or the guard is untested"
    );
    assert!(crate::stats::likelihood::guarded_markov_fd_hessian(&f, &edge, mode).is_none());

    // control: away from the boundary the block is folded unchanged
    let inner = [0.0];
    let h = crate::stats::likelihood::guarded_markov_fd_hessian(&f, &inner, f(&inner))
        .expect("a healthy block is folded");
    assert_eq!(h[(0, 0)].to_bits(), unguarded(&inner)[(0, 0)].to_bits());
    assert!(h[(0, 0)] > 0.0);
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

/// A drug-driven state row listed **before** a same-TIME dose reads the pre-dose state, and
/// one listed after it the post-dose state — NONMEM's record order, through the CSV reader.
/// At TIME 6 the file holds state / dose / state; the closed-form two-dose superposition is
/// the oracle, and the twin in which both rows read the post-dose concentration is far away.
#[test]
fn drug_driven_state_rows_honour_record_order_at_a_co_timed_dose() {
    let dir = std::env::temp_dir().join(format!("ferx_dtmm_codose_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (model_path, data_path) = (dir.join("m.ferx"), dir.join("d.csv"));
    std::fs::write(&model_path, DRUG_DRIVEN).unwrap();
    std::fs::write(
        &data_path,
        "ID,TIME,DV,AMT,EVID,MDV,CMT\n\
         1,0,.,100,1,1,1\n\
         1,1,0,.,0,0,5\n\
         1,6,0,.,0,0,5\n\
         1,6,.,100,1,1,1\n\
         1,6,1,.,0,0,5\n\
         1,8,0,.,0,0,5\n",
    )
    .unwrap();
    let run = crate::api::prepare_run(
        model_path.to_str().unwrap(),
        Some(data_path.to_str().unwrap()),
    )
    .expect("load");
    let _ = std::fs::remove_dir_all(&dir);
    let s = &run.population.subjects[0];

    let theta = [2.0_f64, 10.0, -1.5, 0.4, -0.5];
    let eta = [0.0_f64];
    let (k, v) = (0.2_f64, 10.0);
    let pre = 100.0 / v * (-k * 6.0).exp();
    let post = pre + 100.0 / v;
    let p01 = |c: f64| inv_logit(-1.5 + 0.4 * c);
    let p10 = inv_logit(-0.5);
    // steps: 0→0 at 6 (pre-dose row), 0→1 at 6 (post-dose row), 1→0 at 8
    let want = -((1.0 - p01(pre)).ln() + p01(post).ln() + p10.ln());
    let all_post = -((1.0 - p01(post)).ln() + p01(post).ln() + p10.ln());
    assert!(
        (want - all_post).abs() > 0.1,
        "fixture does not straddle the dose"
    );
    let got = dtmm_subject_nll(&run.parsed.model, s, &theta, &eta);
    eprintln!("co-timed dose: got {got}, record order {want}, all post-dose {all_post}");
    assert!(
        (got - want).abs() < 1e-12,
        "got {got}, record-order closed form {want}"
    );
    // the user clock is untouched
    let raw: Vec<f64> = s
        .obs_records
        .iter()
        .filter_map(|r| match r {
            ObsRecord::DiscreteState { raw_time, .. } => Some(*raw_time),
            _ => None,
        })
        .collect();
    assert_eq!(raw, [1.0, 6.0, 6.0, 8.0]);
}

/// The same record order at the subject's **first** dose: a state row listed before a dose
/// at TIME 0 reads the pre-dose (empty) state, one listed after it the post-dose state. The
/// pre-dose row's internal time is one ULP below the first break, which the dense solve
/// used to leave unread — `NaN`, so the subject scored the sentinel.
#[test]
fn drug_driven_state_row_before_the_first_dose_reads_the_initial_state() {
    let dir = std::env::temp_dir().join(format!("ferx_dtmm_first_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (model_path, data_path) = (dir.join("m.ferx"), dir.join("d.csv"));
    std::fs::write(&model_path, DRUG_DRIVEN).unwrap();
    std::fs::write(
        &data_path,
        "ID,TIME,DV,AMT,EVID,MDV,CMT\n\
         1,0,0,.,0,0,5\n\
         1,0,.,100,1,1,1\n\
         1,0,1,.,0,0,5\n\
         1,2,0,.,0,0,5\n",
    )
    .unwrap();
    let run = crate::api::prepare_run(
        model_path.to_str().unwrap(),
        Some(data_path.to_str().unwrap()),
    )
    .expect("load");
    let _ = std::fs::remove_dir_all(&dir);
    let s = &run.population.subjects[0];
    let theta = [2.0_f64, 10.0, -1.5, 0.4, -0.5];
    let p01 = |c: f64| inv_logit(-1.5 + 0.4 * c);
    // The pre-dose row is the conditioned-on first record, so no step reads its state;
    // check the sampled states directly, then the NLL (0 → 1 at C = 10, then 1 → 0).
    let chain = collect_states(5, &[0, 1], &s.obs_records).unwrap();
    let states = chain_ode_states(&run.parsed.model, s, &theta, &[0.0], &chain)
        .expect("every record, including the pre-dose one, has a finite state");
    assert_eq!(states[0], [0.0], "pre-dose row reads the initial state");
    assert_eq!(states[1], [100.0], "post-dose row reads the dose");
    let want = -(p01(10.0).ln() + inv_logit(-0.5).ln());
    let got = dtmm_subject_nll(&run.parsed.model, s, &theta, &[0.0]);
    assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
}
