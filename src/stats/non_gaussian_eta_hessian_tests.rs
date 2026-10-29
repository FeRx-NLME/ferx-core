//! #2437: the FOCEI Laplace term's non-Gaussian data-term η-Hessians. Every endpoint with a
//! dual program runs its one generic likelihood on η-jets (`sens::eta_jets`); the parity
//! matrix below pins, per cell, that it takes the exact route, that the route's value is
//! the production `f64` value path's, and that its Hessian matches a Richardson stencil of
//! that production value — the one oracle that shares nothing with the dual. The cells
//! span the endpoint kinds, the record kinds and the boundary values where a formula
//! switches arm. The finite-difference fallback for the endpoints the duals cannot serve
//! is pinned last.

use super::*;
use crate::parser::model_parser::parse_model_string;
use crate::survival::data_term_hessian_fd;
use crate::types::{DoseEvent, EndpointLikelihood, EventType, ObsRecord};

fn parse(src: &str) -> CompiledModel {
    parse_model_string(src).expect("fixture model parses")
}

/// Richardson-extrapolated 4-point stencil on a uniform step: `(4·H(h/2) − H(h)) / 3`
/// cancels the `h²` truncation term, leaving `O(h⁴)` plus roundoff `~ε|f|/h²`.
fn richardson_hessian(f: impl Fn(&[f64]) -> f64, eta: &[f64], h: f64) -> DMatrix<f64> {
    let n = eta.len();
    let coarse = data_term_hessian_fd(&f, eta, &vec![h; n]);
    let fine = data_term_hessian_fd(&f, eta, &vec![h / 2.0; n]);
    (fine * 4.0 - coarse) / 3.0
}

/// Worst entry error relative to the largest entry of `want`. Asserts every entry of both
/// matrices finite first, so a `NaN` cannot hide behind `f64::max`.
fn rel_err(got: &DMatrix<f64>, want: &DMatrix<f64>) -> f64 {
    assert!(
        got.iter().chain(want.iter()).all(|x| x.is_finite()),
        "non-finite Hessian entry:\ngot {got}\nwant {want}"
    );
    let scale = want.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    assert!(scale > 0.0, "degenerate reference Hessian {want}");
    got.iter()
        .zip(want.iter())
        .fold(0.0_f64, |m, (g, w)| m.max((g - w).abs()))
        / scale
}

/// Every off-diagonal entry of `h` is live, so a kernel that drops the cross terms fails.
fn assert_coupled(label: &str, h: &DMatrix<f64>) {
    let scale = h.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    for i in 0..h.nrows() {
        for j in 0..i {
            assert!(
                h[(i, j)].abs() > 1e-3 * scale,
                "{label}: fixture precondition: η{i}–η{j} coupling live, got {h}"
            );
        }
    }
}

fn discrete(time: f64, state: usize, cmt: usize) -> ObsRecord {
    ObsRecord::DiscreteState {
        time,
        raw_time: time,
        state,
        cmt,
    }
}

fn event(time: f64, event_type: EventType, entry_time: f64) -> ObsRecord {
    ObsRecord::Event {
        time,
        event_type,
        entry_time,
        cmt: 2,
    }
}

/// One cell of the parity matrix: its exact `(value, H)` and route, and the production
/// value path at any η.
struct Cell<'a> {
    label: String,
    eta: Vec<f64>,
    route: EtaHessianRoute,
    exact: Option<(f64, DMatrix<f64>)>,
    hessian: DMatrix<f64>,
    value: Box<dyn Fn(&[f64]) -> f64 + 'a>,
}

/// Check one cell; `Err` names what failed, so the matrix reports every bad cell at once.
fn check(c: &Cell<'_>) -> Result<f64, String> {
    let label = &c.label;
    if c.route != EtaHessianRoute::Exact {
        return Err(format!("{label}: took {:?}", c.route));
    }
    let (v, h) = c
        .exact
        .as_ref()
        .ok_or(format!("{label}: exact kernel declined"))?;
    if (&c.hessian - h).abs().max() != 0.0 {
        return Err(format!(
            "{label}: the routed Hessian is not the exact kernel's"
        ));
    }
    let f0 = (c.value)(&c.eta);
    if !(f0.is_finite() && f0.abs() < 1e3) {
        return Err(format!("{label}: fixture value {f0}"));
    }
    // One generic kernel, so the values agree to round-off. Not to the bit: a dual power
    // with a variable exponent is `exp(e·ln x)`, not `powf` (Weibull measured 1 ULP apart).
    if !((v - f0).abs() <= 1e-14 * f0.abs()) {
        return Err(format!("{label}: dual value {v} vs value path {f0}"));
    }
    let want = richardson_hessian(&c.value, &c.eta, 2e-3);
    assert_coupled(label, &want);
    let err = rel_err(h, &want);
    eprintln!("{label}: exact vs Richardson {err:.3e}");
    // Measured worst 2.0e-9 (ctmm slow rates; Linux x86_64 and macOS arm64): the Richardson
    // reference's own roundoff on a near-flat log P.
    if !(err < 1e-7) {
        return Err(format!("{label}: {err:.3e}\n{h}\nvs\n{want}"));
    }
    Ok(err)
}

fn binary_cell<'a>(label: &str, model: &'a CompiledModel, s: &'a Subject, eta: &[f64]) -> Cell<'a> {
    let theta = model.default_params.theta.clone();
    let omega = model.default_params.omega.matrix.clone();
    let mode = crate::categorical::discrete_subject_nll(model, s, &theta, eta);
    let (hessian, route) = discrete_data_term_eta_hessian(model, s, &theta, eta, mode, &omega);
    let exact = crate::categorical::discrete_subject_eta_hessian(model, s, &theta, eta);
    Cell {
        label: label.into(),
        eta: eta.to_vec(),
        route,
        exact,
        hessian,
        value: Box::new(move |e| crate::categorical::discrete_subject_nll(model, s, &theta, e)),
    }
}

#[cfg(feature = "markov")]
fn ctmm_cell<'a>(label: &str, model: &'a CompiledModel, s: &'a Subject, eta: &[f64]) -> Cell<'a> {
    let theta = model.default_params.theta.clone();
    let omega = model.default_params.omega.matrix.clone();
    let mode = crate::markov::endpoint::ctmm_subject_nll(model, s, &theta, eta);
    let (hessian, route) = ctmm_data_term_eta_hessian(model, s, &theta, eta, mode, &omega)
        .unwrap_or((DMatrix::zeros(0, 0), EtaHessianRoute::FiniteDifference));
    let exact = crate::markov::endpoint::ctmm_subject_eta_hessian(model, s, &theta, eta);
    Cell {
        label: label.into(),
        eta: eta.to_vec(),
        route,
        exact,
        hessian,
        value: Box::new(move |e| crate::markov::endpoint::ctmm_subject_nll(model, s, &theta, e)),
    }
}

fn tte_cell<'a>(
    label: &str,
    model: &'a CompiledModel,
    records: &'a [ObsRecord],
    cov: &'a std::collections::HashMap<String, f64>,
    eta: &[f64],
) -> Cell<'a> {
    let Some(EndpointLikelihood::Tte {
        hazard,
        recurrence,
        hazard_program,
        ..
    }) = model.endpoints.get(&2)
    else {
        panic!("{label}: fixture precondition: a TTE endpoint on CMT 2");
    };
    let HazardSpec::Analytic { family, .. } = hazard else {
        panic!("{label}: fixture precondition: an analytic hazard");
    };
    let theta = model.default_params.theta.clone();
    let omega = model.default_params.omega.matrix.clone();
    let value = move |e: &[f64]| {
        crate::survival::tte_data_term(records, hazard, *recurrence, &theta, e, cov)
    };
    let theta = model.default_params.theta.clone();
    let (hessian, route) = tte_data_term_eta_hessian(
        hazard,
        *recurrence,
        hazard_program.as_ref(),
        records,
        cov,
        &theta,
        eta,
        &value,
        || value(eta),
        &omega,
    );
    let exact = hazard_program.as_ref().and_then(|prog| {
        crate::survival::analytic_tte_eta_hessian(
            records,
            *family,
            *recurrence,
            prog,
            &theta,
            eta,
            cov,
        )
    });
    Cell {
        label: label.into(),
        eta: eta.to_vec(),
        route,
        exact,
        hessian,
        value: Box::new(value),
    }
}

/// The parity matrix: every endpoint kind with a dual program, each record kind its
/// likelihood scores, and the boundary values where a formula switches arm —
///
/// * binary: both outcomes, a `TIME` term, a covariate, η meeting through an `exp`; and a
///   saturated predictor (`p` within 1e-5 of 1, curvature `p(1 − p)` near zero);
/// * homogeneous CTMM: three states, η on two rates that also meet in a third, a
///   zero-length same-state gap, stays and transitions both ways, and a path that ends in
///   a different state from where it starts — this birth–death chain is reversible, so on
///   a closed path `P[b, a]` and `P[a, b]` give the same likelihood and a transposed
///   generator would pass; and slow rates
///   (`P[s, s]` within ~1e-4 of 1, near the `p ≤ 1` clamp);
/// * analytic TTE, every family and both RTTE clocks, with an exact event, a
///   right-censored one, an interval-censored one, and left truncation, η in the hazard
///   parameters and in `loghr`; and Gompertz at `γ̂ = 0` exactly and with `γt` on both
///   sides of the series boundary, where a value-only limit once dropped the
///   γ-curvature.
#[test]
fn every_endpoint_cell_is_exact_and_matches_its_value_path() {
    let binary = parse(
        "[parameters]\n  theta TH0(0.2, -10.0, 10.0)\n  theta THX(0.5, -10.0, 10.0)\n  theta \
         THT(0.3, -10.0, 10.0)\n  omega ETA_A ~ 0.25\n  omega ETA_B ~ 0.16\n[binary_model]\n  \
         cmt   = 3\n  logit = TH0 + THX * X + ETA_A + THT * exp(ETA_B + 0.5 * ETA_A) * TIME\n",
    );
    let mut bs = Subject {
        id: "1".into(),
        ..Default::default()
    };
    bs.covariates.insert("X".into(), 0.8);
    bs.obs_records = [(0.5, 1), (1.0, 0), (2.0, 1), (3.5, 0)]
        .into_iter()
        .map(|(t, y)| discrete(t, y, 3))
        .collect();
    let mut saturated = bs.clone();
    saturated.covariates.insert("X".into(), 18.0);

    #[cfg(feature = "markov")]
    let ctmm = |rate: f64| {
        parse(&format!(
            "[parameters]\n  theta L01({}, -12.0, 3.0)\n  theta L10({}, -12.0, 3.0)\n  theta \
             L12({}, -12.0, 3.0)\n  theta L21({}, -12.0, 3.0)\n  omega ETA_A ~ 0.25\n  omega \
             ETA_B ~ 0.16\n[markov_model]\n  type   = ctmm\n  cmt    = 5\n  states = [s0=0, \
             s1=1, s2=2]\n  transition s0 -> s1 = exp(L01 + ETA_A)\n  transition s1 -> s0 = \
             exp(L10 + 0.5 * ETA_B)\n  transition s1 -> s2 = exp(L12 + ETA_A * ETA_B)\n  \
             transition s2 -> s1 = exp(L21 - ETA_B)\n",
            -0.7 + rate,
            -1.2 + rate,
            -0.4 + rate,
            -1.0 + rate
        ))
    };
    #[cfg(feature = "markov")]
    let (ctmm_fast, ctmm_slow) = (ctmm(0.0), ctmm(-9.0));
    #[cfg(feature = "markov")]
    let cs = {
        let mut s = Subject {
            id: "1".into(),
            ..Default::default()
        };
        s.obs_records = [
            (0.0, 0),
            (1.0, 1),
            (1.0, 1),
            (2.5, 2),
            (4.0, 1),
            (6.0, 0),
            (7.0, 0),
            (9.0, 2),
        ]
        .into_iter()
        .map(|(t, st)| discrete(t, st, 5))
        .collect();
        s
    };

    let params = "[parameters]\n  theta TA(0.08, 0.001, 10.0)\n  theta TB(1.4, 0.01, 10.0)\n  \
                  theta TX(0.3, -10.0, 10.0)\n  omega ETA_S ~ 0.25\n  omega ETA_K ~ 0.09\n";
    let single = [
        event(3.0, EventType::Exact, 0.5),
        event(8.0, EventType::RightCensored, 0.0),
        event(
            0.0,
            EventType::IntervalCensored {
                left: 2.0,
                right: 5.0,
            },
            0.0,
        ),
    ];
    let recurrent = [
        event(1.5, EventType::Exact, 0.5),
        event(4.0, EventType::Exact, 0.0),
        event(9.0, EventType::RightCensored, 0.0),
    ];
    let gompertz = |gamma: &str| {
        format!(
            "family = gompertz\n  alpha  = TA * exp(ETA_S)\n  gamma  = {gamma}\n  loghr  = TX \
             * X + 0.3 * ETA_S"
        )
    };
    let tte: Vec<(&str, String, &[ObsRecord], [f64; 2])> =
        vec![
        (
            "exponential",
            "family = exponential\n  scale  = TA * exp(ETA_S)\n  loghr  = TX * X + 0.4 * ETA_K \
             * ETA_S + ETA_K"
                .into(),
            &single,
            [0.3, -0.2],
        ),
        (
            "weibull",
            "family = weibull\n  scale  = (1 / TA) * exp(ETA_S)\n  shape  = TB * exp(ETA_K)\n  \
             loghr  = TX * X"
                .into(),
            &single,
            [0.3, -0.2],
        ),
        ("gompertz", gompertz("0.1 * TB + 0.1 * ETA_K"), &single, [0.3, -0.2]),
        // γ̂ = 0 exactly: the series arm's own derivatives are the whole γ-curvature.
        ("gompertz at γ̂ = 0", gompertz("0.1 * ETA_K"), &single, [0.3, 0.0]),
        // γ̂ = 0.015: γt runs 0.0075–0.12 over the records, both arms in one subject.
        (
            "gompertz across the series boundary",
            gompertz("0.02 + 0.1 * ETA_K"),
            &single,
            [0.3, -0.05],
        ),
        (
            "weibull rtte forward",
            "type   = rtte\n  family = weibull\n  scale  = (1 / TA) * exp(ETA_S)\n  shape  = TB \
             * exp(ETA_K)"
                .into(),
            &recurrent,
            [0.3, -0.2],
        ),
        (
            "weibull rtte reset",
            "type   = rtte\n  clock  = reset\n  family = weibull\n  scale  = (1 / TA) * \
             exp(ETA_S)\n  shape  = TB * exp(ETA_K)"
                .into(),
            &recurrent,
            [0.3, -0.2],
        ),
    ];
    let mut cov = std::collections::HashMap::new();
    cov.insert("X".to_string(), 0.8);
    let tte_models: Vec<(&str, CompiledModel, &[ObsRecord], [f64; 2])> = tte
        .into_iter()
        .map(|(label, body, records, eta)| {
            let model = parse(&format!("{params}[event_model]\n  cmt    = 2\n  {body}\n"));
            (label, model, records, eta)
        })
        .collect();

    // Built last: every cell borrows the fixtures above.
    let mut cells = vec![
        binary_cell("binary", &binary, &bs, &[0.3, -0.2]),
        binary_cell("binary saturated", &binary, &saturated, &[0.3, -0.2]),
    ];
    #[cfg(feature = "markov")]
    {
        cells.push(ctmm_cell("ctmm", &ctmm_fast, &cs, &[0.3, -0.2]));
        cells.push(ctmm_cell("ctmm slow rates", &ctmm_slow, &cs, &[0.3, -0.2]));
    }
    for (label, model, records, eta) in &tte_models {
        cells.push(tte_cell(label, model, records, &cov, eta));
    }

    let failures: Vec<String> = cells.iter().filter_map(|c| check(c).err()).collect();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The routes the duals cannot serve fall back to the FD stencil, and the dispatch cap
/// straddles: 48 η (the widest bucket) is exact, 49 is FD. A drug-driven (ODE-accumulated) hazard and a
/// time-inhomogeneous CTMM generator have no program. Each FD fallback must still agree
/// with the Richardson reference to the stencil's own accuracy.
#[test]
fn endpoints_without_a_dual_program_fall_back_to_finite_differences() {
    // The cap: η count 48 (exact) and 49 (FD) on one binary predictor. A 48-wide
    // second-order dual needs a fit's stack (`FIT_RAYON_STACK_SIZE`), not a test thread's.
    let wide = |n: usize, want_route: EtaHessianRoute| {
        let omegas: String = (0..n).map(|k| format!("  omega E{k} ~ 0.1\n")).collect();
        let lp: Vec<String> = (0..n)
            .map(|k| format!("{} * E{k}", 0.02 * (k + 1) as f64))
            .collect();
        let model = parse(&format!(
            "[parameters]\n  theta TH0(0.2, -10.0, 10.0)\n{omegas}[binary_model]\n  cmt   = 3\n  \
             logit = TH0 + {}\n",
            lp.join(" + ")
        ));
        let mut s = Subject {
            id: "1".into(),
            ..Default::default()
        };
        s.obs_records = [(0.0, 1), (1.0, 0), (2.0, 1)]
            .into_iter()
            .map(|(t, y)| discrete(t, y, 3))
            .collect();
        let theta = model.default_params.theta.clone();
        let eta = vec![0.1; n];
        let omega = model.default_params.omega.matrix.clone();
        let mode = crate::categorical::discrete_subject_nll(&model, &s, &theta, &eta);
        let (got, route) = discrete_data_term_eta_hessian(&model, &s, &theta, &eta, mode, &omega);
        assert_eq!(route, want_route, "{n} η");
        let want = richardson_hessian(
            |e| crate::categorical::discrete_subject_nll(&model, &s, &theta, e),
            &eta,
            2e-3,
        );
        let err = rel_err(&got, &want);
        // Measured (Linux x86_64 and macOS arm64): 4.0e-9 at 48 η (exact), 1.1e-7 at 49 η (FD) — the
        // Richardson reference's roundoff over 48² entries, and the FD stencil's own error.
        eprintln!("binary, {n} η ({route:?}): {err:.3e}");
        assert!(err < 1e-6, "{n} η: {err:.3e}");
    };
    for (n, want_route) in [
        (48, EtaHessianRoute::Exact),
        (49, EtaHessianRoute::FiniteDifference),
    ] {
        std::thread::Builder::new()
            .stack_size(crate::FIT_RAYON_STACK_SIZE)
            .spawn(move || wide(n, want_route))
            .expect("spawn")
            .join()
            .expect("width straddle");
    }

    // A drug-driven hazard: `examples/pktte_joint.ferx`'s, one dose, one exact event.
    let model = crate::parser::model_parser::parse_model_file(std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/pktte_joint.ferx"
    )))
    .expect("example parses");
    let (cmt, hazard, recurrence, hazard_program) = model
        .endpoints
        .iter()
        .find_map(|(cmt, ep)| match ep {
            EndpointLikelihood::Tte {
                hazard,
                recurrence,
                hazard_program,
                ..
            } => Some((*cmt, hazard, *recurrence, hazard_program)),
            _ => None,
        })
        .expect("a TTE endpoint");
    assert!(matches!(hazard, HazardSpec::OdeAccumulated { .. }));
    assert!(hazard_program.is_none());
    let mut s = Subject {
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        ..Default::default()
    };
    s.obs_records = vec![ObsRecord::Event {
        time: 12.0,
        event_type: EventType::Exact,
        entry_time: 0.0,
        cmt,
    }];
    let theta = model.default_params.theta.clone();
    let eta = [0.2];
    let omega = model.default_params.omega.matrix.clone();
    let records = s.obs_records.clone();
    let tte_fn = |e: &[f64]| tte_endpoint_nll(&model, &s, hazard, recurrence, &records, &theta, e);
    let f0 = tte_fn(&eta);
    let (got, route) = tte_data_term_eta_hessian(
        hazard,
        recurrence,
        None,
        &records,
        &s.covariates,
        &theta,
        &eta,
        tte_fn,
        || f0,
        &omega,
    );
    assert_eq!(route, EtaHessianRoute::FiniteDifference);
    let want = richardson_hessian(tte_fn, &eta, 2e-3);
    let err = rel_err(&got, &want);
    // Measured 2.4e-7 (Linux x86_64 and macOS arm64).
    eprintln!("drug-driven TTE (FD): {err:.3e}");
    assert!(err < 1e-5, "drug-driven TTE FD fallback off by {err:.3e}");

    // A time-inhomogeneous CTMM: `examples/ctmm_pd_2state.ferx`'s generator reads `central`.
    #[cfg(feature = "markov")]
    {
        let model = crate::parser::model_parser::parse_model_file(std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/ctmm_pd_2state.ferx"
        )))
        .expect("example parses");
        assert!(model.endpoints.values().any(|e| matches!(
            e,
            EndpointLikelihood::Ctmm { generator_program: None, generator_states, .. }
                if !generator_states.is_empty()
        )));
        let mut s = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            ..Default::default()
        };
        s.obs_records = [(0.0, 0), (2.0, 1), (5.0, 0), (9.0, 0)]
            .into_iter()
            .map(|(t, st)| discrete(t, st, 5))
            .collect();
        let theta = model.default_params.theta.clone();
        let eta = [0.2];
        let omega = model.default_params.omega.matrix.clone();
        let mode = crate::markov::endpoint::ctmm_subject_nll(&model, &s, &theta, &eta);
        let (_, route) =
            ctmm_data_term_eta_hessian(&model, &s, &theta, &eta, mode, &omega).expect("served");
        assert_eq!(route, EtaHessianRoute::FiniteDifference);
    }
}

/// A subject the value path repels (`1e20`) has no curvature to give: each exact kernel
/// declines it rather than differentiate the flat sentinel to a zero Hessian. One cell per
/// endpoint guard — a TTE exact event at zero hazard, a binary predictor that overflows,
/// a CTMM generator with a negative rate — each asserted to be the sentinel first.
#[test]
fn a_repelled_subject_declines_the_exact_hessian() {
    let tte = parse(
        "[parameters]\n  theta TA(0.5, -10.0, 10.0)\n  omega ETA_S ~ 0.25\n[event_model]\n  \
         cmt    = 2\n  family = exponential\n  scale  = TA * ETA_S\n",
    );
    let Some(EndpointLikelihood::Tte {
        hazard,
        recurrence,
        hazard_program,
        ..
    }) = tte.endpoints.get(&2)
    else {
        panic!("fixture precondition: a TTE endpoint on CMT 2");
    };
    let HazardSpec::Analytic { family, .. } = hazard else {
        panic!("fixture precondition: an analytic hazard");
    };
    let records = [event(3.0, EventType::Exact, 0.0)];
    let cov = std::collections::HashMap::new();
    let theta = tte.default_params.theta.clone();
    let v = crate::survival::tte_data_term(&records, hazard, *recurrence, &theta, &[0.0], &cov);
    assert_eq!(
        v, 1e20,
        "fixture precondition: h = 0 at an exact event is repelled"
    );
    let prog = hazard_program.as_ref().expect("program");
    assert!(crate::survival::analytic_tte_eta_hessian(
        &records,
        *family,
        *recurrence,
        prog,
        &theta,
        &[0.0],
        &cov
    )
    .is_none());

    let binary = parse(
        "[parameters]\n  theta TH0(0.2, -10.0, 10.0)\n  omega ETA_A ~ 0.25\n[binary_model]\n  \
         cmt   = 3\n  logit = TH0 + exp(1000 * ETA_A)\n",
    );
    let mut bs = Subject {
        id: "1".into(),
        ..Default::default()
    };
    bs.obs_records = vec![discrete(1.0, 1, 3)];
    let theta = binary.default_params.theta.clone();
    let v = crate::categorical::discrete_subject_nll(&binary, &bs, &theta, &[1.0]);
    assert_eq!(
        v, 1e20,
        "fixture precondition: an overflowed predictor is repelled"
    );
    assert!(
        crate::categorical::discrete_subject_eta_hessian(&binary, &bs, &theta, &[1.0]).is_none()
    );

    #[cfg(feature = "markov")]
    {
        let ctmm = parse(
            "[parameters]\n  theta L01(0.5, -6.0, 3.0)\n  omega ETA_A ~ 0.25\n[markov_model]\n  \
             type   = ctmm\n  cmt    = 5\n  states = [s0=0, s1=1]\n  transition s0 -> s1 = \
             L01 + ETA_A\n  transition s1 -> s0 = exp(L01)\n",
        );
        let mut cs = Subject {
            id: "1".into(),
            ..Default::default()
        };
        // A stay in s0: with `q01 < 0` its `P[0, 0] > 1` clamps to a finite constant, so
        // only the negative-rate guard makes this subject the sentinel.
        cs.obs_records = vec![discrete(0.0, 0, 5), discrete(1.0, 0, 5)];
        let theta = ctmm.default_params.theta.clone();
        let v = crate::markov::endpoint::ctmm_subject_nll(&ctmm, &cs, &theta, &[-1.0]);
        assert_eq!(v, 1e20, "fixture precondition: a negative rate is repelled");
        assert!(
            crate::markov::endpoint::ctmm_subject_eta_hessian(&ctmm, &cs, &theta, &[-1.0])
                .is_none()
        );
    }
}
