//! #2437: the FOCEI Laplace term's non-Gaussian data-term η-Hessians. Each endpoint family
//! with a dual program takes the exact route (`EtaHessianRoute::Exact`); these tests pin
//! that route, its value against a reference computed outside it, and the finite-difference
//! fallback for the endpoints the duals cannot serve.

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

/// Central differences of an exact η-gradient, symmetrized: `O(h²)` truncation on the
/// third derivative and roundoff `~ε|g|/h`, so ~1e-10 at `h = 1e-5`.
fn fd_of_gradient(g: impl Fn(&[f64]) -> Vec<f64>, eta: &[f64], h: f64) -> DMatrix<f64> {
    let n = eta.len();
    let mut out = DMatrix::zeros(n, n);
    for j in 0..n {
        let mut up = eta.to_vec();
        let mut dn = eta.to_vec();
        up[j] += h;
        dn[j] -= h;
        let (gu, gd) = (g(&up), g(&dn));
        for i in 0..n {
            out[(i, j)] = (gu[i] - gd[i]) / (2.0 * h);
        }
    }
    (&out + out.transpose()) * 0.5
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

/// Binary: two η that meet in one record through an `exp`, with a `TIME` term, both
/// outcomes, and a covariate. The reference differences the exact first-order gradient
/// (`discrete_subject_eta_grad`, `Dual1`), a separate path from the `Dual2` Hessian.
#[test]
fn binary_eta_hessian_is_exact() {
    let model = parse(
        "[parameters]\n  theta TH0(0.2, -10.0, 10.0)\n  theta THX(0.5, -10.0, 10.0)\n  theta \
         THT(0.3, -10.0, 10.0)\n  omega ETA_A ~ 0.25\n  omega ETA_B ~ 0.16\n[binary_model]\n  \
         cmt   = 3\n  logit = TH0 + THX * X + ETA_A + THT * exp(ETA_B + 0.5 * ETA_A) * TIME\n",
    );
    let mut s = Subject {
        id: "1".into(),
        ..Default::default()
    };
    s.covariates.insert("X".into(), 0.8);
    s.obs_records = [(0.5, 1), (1.0, 0), (2.0, 1), (3.5, 0)]
        .into_iter()
        .map(|(t, y)| discrete(t, y, 3))
        .collect();
    let theta = model.default_params.theta.clone();
    let eta = [0.3, -0.2];
    let omega = model.default_params.omega.matrix.clone();
    let mode = crate::categorical::discrete_subject_nll(&model, &s, &theta, &eta);

    let (got, route) = discrete_data_term_eta_hessian(&model, &s, &theta, &eta, mode, &omega);
    assert_eq!(route, EtaHessianRoute::Exact);
    let want = fd_of_gradient(
        |e| {
            crate::categorical::discrete_subject_eta_grad(&model, &s, &theta, e)
                .expect("exact gradient")
                .1
        },
        &eta,
        1e-5,
    );
    assert_coupled("binary", &want);
    let err = rel_err(&got, &want);
    // Measured 2.1e-11 (Linux x86_64 and macOS arm64): the reference's own accuracy.
    eprintln!("binary: exact vs FD-of-gradient {err:.3e}");
    assert!(
        err < 1e-8,
        "binary η-Hessian off by {err:.3e}:\n{got}\nvs\n{want}"
    );
}

/// Homogeneous CTMM: three states, η on two rates that also meet in a third, a zero-length
/// same-state gap, stays and transitions both ways. The reference differences the exact
/// Van Loan gradient (`ctmm_subject_eta_grad`).
#[cfg(feature = "markov")]
#[test]
fn ctmm_eta_hessian_is_exact() {
    let model = parse(
        "[parameters]\n  theta L01(-0.7, -6.0, 3.0)\n  theta L10(-1.2, -6.0, 3.0)\n  theta \
         L12(-0.4, -6.0, 3.0)\n  theta L21(-1.0, -6.0, 3.0)\n  omega ETA_A ~ 0.25\n  omega \
         ETA_B ~ 0.16\n[markov_model]\n  type   = ctmm\n  cmt    = 5\n  states = [s0=0, s1=1, \
         s2=2]\n  transition s0 -> s1 = exp(L01 + ETA_A)\n  transition s1 -> s0 = exp(L10 + \
         0.5 * ETA_B)\n  transition s1 -> s2 = exp(L12 + ETA_A * ETA_B)\n  transition s2 -> s1 \
         = exp(L21 - ETA_B)\n",
    );
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
    ]
    .into_iter()
    .map(|(t, st)| discrete(t, st, 5))
    .collect();
    let theta = model.default_params.theta.clone();
    let eta = [0.3, -0.2];
    let omega = model.default_params.omega.matrix.clone();
    let mode = crate::markov::endpoint::ctmm_subject_nll(&model, &s, &theta, &eta);

    let (got, route) =
        ctmm_data_term_eta_hessian(&model, &s, &theta, &eta, mode, &omega).expect("served");
    assert_eq!(route, EtaHessianRoute::Exact);
    let want = fd_of_gradient(
        |e| {
            crate::markov::endpoint::ctmm_subject_eta_grad(&model, &s, &theta, e)
                .expect("exact gradient")
                .1
        },
        &eta,
        1e-5,
    );
    assert_coupled("ctmm", &want);
    let err = rel_err(&got, &want);
    // Measured 2.6e-11 (Linux x86_64 and macOS arm64): the reference's own accuracy.
    eprintln!("ctmm: exact vs FD-of-gradient {err:.3e}");
    assert!(
        err < 1e-8,
        "CTMM η-Hessian off by {err:.3e}:\n{got}\nvs\n{want}"
    );
}

/// Analytic TTE, every family and recurrence, with every record kind the likelihood scores:
/// an exact event, a right-censored one, an interval-censored one, and left truncation.
/// Two η in each, one of them through `loghr`, so the parameter chain, the `loghr` slot and
/// the cross terms are all live. TTE has no exact gradient, so the reference is the
/// Richardson-extrapolated value stencil; the exact route's value must also be the value
/// path's to round-off, since both are instances of one generic kernel.
#[test]
fn analytic_tte_eta_hessian_is_exact_for_every_family_and_recurrence() {
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
    let cells: Vec<(&str, String, &[ObsRecord])> = vec![
        (
            "exponential",
            "family = exponential\n  scale  = TA * exp(ETA_S)\n  loghr  = TX * X + 0.4 * ETA_K \
             * ETA_S + ETA_K"
                .into(),
            &single,
        ),
        (
            "weibull",
            "family = weibull\n  scale  = (1 / TA) * exp(ETA_S)\n  shape  = TB * exp(ETA_K)\n  \
             loghr  = TX * X"
                .into(),
            &single,
        ),
        (
            "gompertz",
            "family = gompertz\n  alpha  = TA * exp(ETA_S)\n  gamma  = 0.1 * TB + 0.1 * ETA_K\n  \
             loghr  = TX * X + 0.3 * ETA_S"
                .into(),
            &single,
        ),
        (
            "weibull rtte forward",
            "type   = rtte\n  family = weibull\n  scale  = (1 / TA) * exp(ETA_S)\n  shape  = TB \
             * exp(ETA_K)"
                .into(),
            &recurrent,
        ),
        (
            "weibull rtte reset",
            "type   = rtte\n  clock  = reset\n  family = weibull\n  scale  = (1 / TA) * \
             exp(ETA_S)\n  shape  = TB * exp(ETA_K)"
                .into(),
            &recurrent,
        ),
    ];
    let mut failures = Vec::new();
    for (label, body, records) in cells {
        let model = parse(&format!("{params}[event_model]\n  cmt    = 2\n  {body}\n"));
        let Some(EndpointLikelihood::Tte {
            hazard,
            recurrence,
            hazard_program,
            ..
        }) = model.endpoints.get(&2)
        else {
            panic!("{label}: fixture precondition: a TTE endpoint on CMT 2");
        };
        let theta = model.default_params.theta.clone();
        let eta = [0.3, -0.2];
        let omega = model.default_params.omega.matrix.clone();
        let mut cov = std::collections::HashMap::new();
        cov.insert("X".to_string(), 0.8);
        let tte_fn = |e: &[f64]| {
            crate::survival::tte_data_term(records, hazard, *recurrence, &theta, e, &cov)
        };
        let f0 = tte_fn(&eta);
        assert!(f0.is_finite() && f0 < 1e3, "{label}: fixture value {f0}");

        let (got, route) = tte_data_term_eta_hessian(
            hazard,
            *recurrence,
            hazard_program.as_ref(),
            records,
            &cov,
            &theta,
            &eta,
            tte_fn,
            || f0,
            &omega,
        );
        assert_eq!(route, EtaHessianRoute::Exact, "{label}");
        let HazardSpec::Analytic { family, .. } = hazard else {
            unreachable!()
        };
        let (value, _) = crate::survival::analytic_tte_eta_hessian(
            records,
            *family,
            *recurrence,
            hazard_program.as_ref().expect("program"),
            &theta,
            &eta,
            &cov,
        )
        .expect("served");
        // One generic kernel, so the values agree to round-off. Not to the bit: a dual
        // power with a variable exponent is `exp(e·ln x)`, not `powf` (Weibull measured
        // 1 ULP apart). The Laplace term reads only the Hessian off this route.
        assert!(
            (value - f0).abs() <= 1e-14 * f0.abs(),
            "{label}: dual value {value} vs value path {f0}"
        );
        let want = richardson_hessian(tte_fn, &eta, 2e-3);
        assert_coupled(label, &want);
        let err = rel_err(&got, &want);
        eprintln!("{label}: exact vs Richardson {err:.3e}");
        // Measured worst 2.5e-10 (exponential, Linux x86_64 and macOS arm64): the Richardson reference's own
        // roundoff.
        if !(err < 1e-7) {
            failures.push(format!("{label}: {err:.3e}\n{got}\nvs\n{want}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The routes the duals cannot serve fall back to the FD stencil, and the dispatch cap
/// straddles: eight η is exact, nine is FD. A drug-driven (ODE-accumulated) hazard and a
/// time-inhomogeneous CTMM generator have no program. Each FD fallback must still agree
/// with the Richardson reference to the stencil's own accuracy.
#[test]
fn endpoints_without_a_dual_program_fall_back_to_finite_differences() {
    // The cap: η count 8 (exact) and 9 (FD) on one binary predictor.
    for (n, want_route) in [
        (8, EtaHessianRoute::Exact),
        (9, EtaHessianRoute::FiniteDifference),
    ] {
        let omegas: String = (0..n).map(|k| format!("  omega E{k} ~ 0.1\n")).collect();
        let lp: Vec<String> = (0..n).map(|k| format!("0.{} * E{k}", k + 1)).collect();
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
        // Measured (Linux x86_64 / macOS arm64): 7.6e-10 / 7.4e-10 at 8 η (exact), 5.1e-8 at
        // 9 η (FD) on both.
        eprintln!("binary, {n} η ({route:?}): {err:.3e}");
        assert!(err < 1e-6, "{n} η: {err:.3e}");
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
