//! #2437 / #2786: the FOCEI Laplace term's non-Gaussian data-term η-Hessians, from the
//! subject's compiled term plan. The parity matrix below pins, per cell, that the marginal's
//! whole-term Hessian is the endpoint's own plan's, that the plan's value is the production
//! `f64` value path's, and that its Hessian matches a Richardson stencil of that production
//! value — the one oracle that shares nothing with the recording. The cells span the
//! endpoint kinds, the record kinds and the boundary values where a formula switches arm.
//! The widths and endpoints the retired duals could not serve are pinned next, and the
//! repelled subjects last.

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

/// One cell of the parity matrix: the endpoint's own exact `(value, H)`, the marginal's
/// whole-term Hessian, and the production value path at any η.
struct Cell<'a> {
    label: String,
    eta: Vec<f64>,
    exact: Option<(f64, DMatrix<f64>)>,
    hessian: Option<DMatrix<f64>>,
    value: Box<dyn Fn(&[f64]) -> f64 + 'a>,
}

/// The η-Hessian of a one-η subject term on the tests' forward dual (`Dual2`): an
/// oracle that shares no code with the plans and, unlike a stencil of an adaptively
/// integrated objective, has no `tol/h²` roundoff — the solver's steps are differentiated
/// as taken, exactly as the recording does.
fn dual_hessian1<S: crate::survival::SubjectTerm>(
    term: &S,
    theta: &[f64],
    eta: f64,
) -> DMatrix<f64> {
    use crate::sens::dual2::Dual2;
    let th: Vec<Dual2<1>> = theta.iter().map(|&v| Dual2::constant(v)).collect();
    let e = [Dual2::var(eta, 0)];
    let y = term.eval(&th, &e).expect("the dual evaluates the term");
    DMatrix::from_element(1, 1, y.hess[0][0])
}

/// The marginal's curvature of `s`'s whole non-Gaussian term at `eta`.
fn whole_term(model: &CompiledModel, s: &Subject, eta: &[f64]) -> Option<DMatrix<f64>> {
    let theta = &model.default_params.theta;
    crate::survival::non_gaussian_eta_jets2(model, s, theta, eta).map(|(_, _, h)| h)
}

/// Check one cell; `Err` names what failed, so the matrix reports every bad cell at once.
fn check(c: &Cell<'_>) -> Result<f64, String> {
    let label = &c.label;
    let (v, h) = c
        .exact
        .as_ref()
        .ok_or(format!("{label}: the endpoint's plan declined"))?;
    let whole = c
        .hessian
        .as_ref()
        .ok_or(format!("{label}: the whole term's plan declined"))?;
    if (whole - h).abs().max() != 0.0 {
        return Err(format!(
            "{label}: the marginal's Hessian is not the endpoint's plan's"
        ));
    }
    let f0 = (c.value)(&c.eta);
    if !(f0.is_finite() && f0.abs() < 1e3) {
        return Err(format!("{label}: fixture value {f0}"));
    }
    // One generic kernel, so the values agree to round-off. Not to the bit: a recorded power
    // with a variable exponent is `exp(e·ln x)`, not `powf` (Weibull measured 1 ULP apart).
    if !((v - f0).abs() <= 1e-14 * f0.abs()) {
        return Err(format!("{label}: dual value {v} vs value path {f0}"));
    }
    let want = richardson_hessian(&c.value, &c.eta, 2e-3);
    assert_coupled(label, &want);
    let err = rel_err(h, &want);
    eprintln!("{label}: exact vs Richardson {err:.3e}");
    // Measured worst 1.5e-9 (ctmm slow rates; Linux x86_64 and macOS arm64): the Richardson
    // reference's own roundoff on a near-flat log P.
    if !(err < 1e-7) {
        return Err(format!("{label}: {err:.3e}\n{h}\nvs\n{want}"));
    }
    Ok(err)
}

fn binary_cell<'a>(label: &str, model: &'a CompiledModel, s: &'a Subject, eta: &[f64]) -> Cell<'a> {
    let theta = model.default_params.theta.clone();
    let exact = crate::categorical::discrete_subject_eta_hessian(model, s, &theta, eta);
    Cell {
        label: label.into(),
        eta: eta.to_vec(),
        exact,
        hessian: whole_term(model, s, eta),
        value: Box::new(move |e| crate::categorical::discrete_subject_nll(model, s, &theta, e)),
    }
}

#[cfg(feature = "markov")]
fn ctmm_cell<'a>(label: &str, model: &'a CompiledModel, s: &'a Subject, eta: &[f64]) -> Cell<'a> {
    let theta = model.default_params.theta.clone();
    let exact = crate::markov::endpoint::ctmm_subject_eta_hessian(model, s, &theta, eta);
    Cell {
        label: label.into(),
        eta: eta.to_vec(),
        exact,
        hessian: whole_term(model, s, eta),
        value: Box::new(move |e| crate::markov::endpoint::ctmm_subject_nll(model, s, &theta, e)),
    }
}

/// The TTE subject's exact `(value, H)` from its TTE term's plan.
fn tte_exact(model: &CompiledModel, s: &Subject, eta: &[f64]) -> Option<(f64, DMatrix<f64>)> {
    use crate::sens::ode_provider::adjoint::terms::TermKind;
    let term = crate::survival::TteTerm::new(model, s);
    let theta = &model.default_params.theta;
    crate::survival::term_eta_hessian(model, s, TermKind::Tte, &term, theta, eta)
}

fn tte_cell<'a>(label: &str, model: &'a CompiledModel, s: &'a Subject, eta: &[f64]) -> Cell<'a> {
    let Some(EndpointLikelihood::Tte {
        hazard, recurrence, ..
    }) = model.endpoints.get(&2)
    else {
        panic!("{label}: fixture precondition: a TTE endpoint on CMT 2");
    };
    assert!(
        matches!(hazard, HazardSpec::Analytic { .. }),
        "{label}: fixture precondition: an analytic hazard"
    );
    let theta = model.default_params.theta.clone();
    let value = move |e: &[f64]| {
        crate::survival::tte_data_term(
            &s.obs_records,
            hazard,
            *recurrence,
            &theta,
            e,
            &s.covariates,
        )
    };
    Cell {
        label: label.into(),
        eta: eta.to_vec(),
        exact: tte_exact(model, s, eta),
        hessian: whole_term(model, s, eta),
        value: Box::new(value),
    }
}

/// The parity matrix: every endpoint kind with a dual program, each record kind its
/// likelihood scores, and the boundary values where a formula switches arm —
///
/// * binary: both outcomes, a `TIME` term, a covariate, η meeting through an `exp`; a
///   saturated predictor (`p` within 1e-5 of 1, curvature `p(1 − p)` near zero); and two
///   binary endpoints on one subject, each scoring only its own CMT's records;
/// * homogeneous CTMM: three states, a covariate on one rate, η on two rates that also
///   meet in a third, a
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
    let binary_src = "[parameters]\n  theta TH0(0.2, -10.0, 10.0)\n  theta THX(0.5, -10.0, \
                      10.0)\n  theta THT(0.3, -10.0, 10.0)\n  omega ETA_A ~ 0.25\n  omega ETA_B \
                      ~ 0.16\n[binary_model]\n  cmt   = 3\n  logit = TH0 + THX * X + ETA_A + \
                      THT * exp(ETA_B + 0.5 * ETA_A) * TIME\n";
    let binary = parse(binary_src);
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
    // Two binary endpoints on one subject, each with its own records: each predictor must
    // score only its own CMT's rows, as the value path's walk does. The parser builds one
    // `[binary_model]`, so the second endpoint comes from a twin parse on CMT 4.
    let two_binary = {
        let mut m = parse(binary_src);
        let second = parse(
            "[parameters]\n  theta TH0(0.2, -10.0, 10.0)\n  theta THX(0.5, -10.0, 10.0)\n  \
             theta THT(0.3, -10.0, 10.0)\n  omega ETA_A ~ 0.25\n  omega ETA_B ~ 0.16\n\
             [binary_model]\n  cmt   = 4\n  logit = -TH0 + ETA_B * exp(ETA_A)\n",
        );
        let mut second = second;
        let ep = second.endpoints.remove(&4).expect("second binary endpoint");
        m.endpoints.insert(4, ep);
        m
    };
    let mut two_bs = bs.clone();
    two_bs
        .obs_records
        .extend([(0.7, 0), (1.5, 1), (3.0, 1)].map(|(t, y)| discrete(t, y, 4)));

    #[cfg(feature = "markov")]
    let ctmm = |rate: f64| {
        parse(&format!(
            "[parameters]\n  theta L01({}, -12.0, 3.0)\n  theta L10({}, -12.0, 3.0)\n  theta \
             L12({}, -12.0, 3.0)\n  theta L21({}, -12.0, 3.0)\n  omega ETA_A ~ 0.25\n  omega \
             ETA_B ~ 0.16\n[markov_model]\n  type   = ctmm\n  cmt    = 5\n  states = [s0=0, \
             s1=1, s2=2]\n  transition s0 -> s1 = exp(L01 + ETA_A + 0.4 * X)\n  transition s1 -> s0 = \
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
        // A live covariate on an asymmetric rate, so the generator must be built from the
        // subject's covariates on both paths.
        s.covariates.insert("X".into(), 0.7);
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

    // DTMM (#2473): three states, every step kind — stays, a declared off-diagonal from each
    // row — with coupled η, a covariate and a `TIME` term; then beside a binary endpoint, so
    // the discrete seam's two families are summed on both paths.
    #[cfg(feature = "markov")]
    let dtmm_body = "[markov_model]\n  type   = dtmm\n  cmt    = 5\n  states = [s0=0, s1=1, \
                     s2=2]\n  transition s0 -> s1 = inv_logit(TH0 - 1.0 + ETA_A + 0.4 * X)\n  \
                     transition s0 -> s2 = 0.2 * inv_logit(THX + ETA_A * ETA_B)\n  \
                     transition s1 -> s0 = inv_logit(-0.8 + 0.5 * ETA_B + THT * TIME)\n  \
                     transition s2 -> s1 = inv_logit(-1.0 - ETA_B + ETA_A)\n";
    #[cfg(feature = "markov")]
    let dtmm = parse(&format!(
        "[parameters]\n  theta TH0(0.2, -10.0, 10.0)\n  theta THX(0.5, -10.0, 10.0)\n  \
         theta THT(0.3, -10.0, 10.0)\n  omega ETA_A ~ 0.25\n  omega ETA_B ~ 0.16\n{dtmm_body}"
    ));
    #[cfg(feature = "markov")]
    let binary_and_dtmm = parse(&format!("{binary_src}{dtmm_body}"));
    #[cfg(feature = "markov")]
    let ds = {
        let mut s = Subject {
            id: "1".into(),
            ..Default::default()
        };
        s.covariates.insert("X".into(), 0.8);
        s.obs_records = [0, 0, 1, 1, 0, 2, 2, 1, 0]
            .into_iter()
            .enumerate()
            .map(|(i, st)| discrete(i as f64, st, 5))
            .collect();
        s
    };
    #[cfg(feature = "markov")]
    let bds = {
        let mut s = ds.clone();
        s.obs_records.extend(bs.obs_records.iter().cloned());
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
    let tte_models: Vec<(&str, CompiledModel, Subject, [f64; 2])> = tte
        .into_iter()
        .map(|(label, body, records, eta)| {
            let model = parse(&format!("{params}[event_model]\n  cmt    = 2\n  {body}\n"));
            let mut s = Subject {
                id: "1".into(),
                obs_records: records.to_vec(),
                ..Default::default()
            };
            s.covariates.insert("X".to_string(), 0.8);
            (label, model, s, eta)
        })
        .collect();

    // Built last: every cell borrows the fixtures above.
    let mut cells = vec![
        binary_cell("binary", &binary, &bs, &[0.3, -0.2]),
        binary_cell("binary saturated", &binary, &saturated, &[0.3, -0.2]),
        binary_cell("two binary endpoints", &two_binary, &two_bs, &[0.3, -0.2]),
    ];
    #[cfg(feature = "markov")]
    {
        cells.push(ctmm_cell("ctmm", &ctmm_fast, &cs, &[0.3, -0.2]));
        cells.push(ctmm_cell("ctmm slow rates", &ctmm_slow, &cs, &[0.3, -0.2]));
        cells.push(binary_cell("dtmm", &dtmm, &ds, &[0.3, -0.2]));
        cells.push(binary_cell(
            "binary + dtmm",
            &binary_and_dtmm,
            &bds,
            &[0.3, -0.2],
        ));
    }
    for (label, model, s, eta) in &tte_models {
        cells.push(tte_cell(label, model, s, eta));
    }

    let failures: Vec<String> = cells.iter().filter_map(|c| check(c).err()).collect();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// What the retired duals could not serve, each now exact from its plan: a binary predictor
/// 49 η wide (past the old 48-η dispatch cap, where the marginal fell back to a stencil),
/// against the Richardson reference of its closed-form value path; and a drug-driven
/// (ODE-accumulated) hazard and a time-inhomogeneous CTMM generator reading a PK state,
/// against the forward dual ([`dual_hessian1`]) — a stencil of their adaptively integrated
/// value is off by `~tol/h²` (measured 0.295 relative on the CTMM at `h = 2e-3`), far
/// coarser than the disagreement it would have to detect.
#[test]
fn endpoints_the_duals_could_not_serve_are_exact_from_their_plans() {
    let n = 49;
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
    let got = whole_term(&model, &s, &eta).expect("a 49-η binary term records");
    let want = richardson_hessian(
        |e| crate::categorical::discrete_subject_nll(&model, &s, &theta, e),
        &eta,
        2e-3,
    );
    let err = rel_err(&got, &want);
    // Measured 4.0e-9 at 48 η on the retired duals: the Richardson reference's roundoff over
    // 48² entries.
    eprintln!("binary, {n} η: {err:.3e}");
    assert!(err < 1e-6, "{n} η: {err:.3e}");

    // A drug-driven hazard: `examples/pktte_joint.ferx`'s, one dose, one exact event.
    let model = crate::parser::model_parser::parse_model_file(std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/pktte_joint.ferx"
    )))
    .expect("example parses");
    let (cmt, hazard, recurrence) = model
        .endpoints
        .iter()
        .find_map(|(cmt, ep)| match ep {
            EndpointLikelihood::Tte {
                hazard, recurrence, ..
            } => Some((*cmt, hazard, *recurrence)),
            _ => None,
        })
        .expect("a TTE endpoint");
    assert!(matches!(hazard, HazardSpec::OdeAccumulated { .. }));
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
    let records = s.obs_records.clone();
    let got = whole_term(&model, &s, &eta).expect("a drug-driven hazard records");
    // The value is the production path's.
    let (v, _, _) =
        crate::survival::non_gaussian_eta_jets2(&model, &s, &theta, &eta).expect("records");
    let want_v = tte_endpoint_nll(&model, &s, hazard, recurrence, &records, &theta, &eta);
    assert!(
        (v - want_v).abs() <= 1e-12 * want_v.abs(),
        "value {v} vs {want_v}"
    );
    let want = dual_hessian1(&crate::survival::TteTerm::new(&model, &s), &theta, eta[0]);
    let err = rel_err(&got, &want);
    eprintln!("drug-driven TTE: {err:.3e}");
    // Measured 0 on Linux x86_64: the dual takes the same operations; 1e-12 leaves room for a
    // reassociation, and an error in a derivative rule misses by far more.
    assert!(err < 1e-12, "drug-driven TTE off by {err:.3e}");

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
            EndpointLikelihood::Ctmm { generator_program: Some(p), generator_states, .. }
                if !generator_states.is_empty() && p.reads_states()
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
        let got = whole_term(&model, &s, &eta).expect("a drug-driven generator records");
        let (v, _, _) =
            crate::survival::non_gaussian_eta_jets2(&model, &s, &theta, &eta).expect("records");
        let want_v = crate::markov::endpoint::ctmm_subject_nll(&model, &s, &theta, &eta);
        assert!(
            (v - want_v).abs() <= 1e-12 * want_v.abs(),
            "value {v} vs {want_v}"
        );
        let term = crate::markov::endpoint::CtmmTerm {
            model: &model,
            subject: &s,
        };
        let want = dual_hessian1(&term, &theta, eta[0]);
        let err = rel_err(&got, &want);
        eprintln!("drug-driven CTMM: {err:.3e}");
        // Measured 0 on Linux x86_64 (see the TTE cell).
        assert!(err < 1e-12, "drug-driven CTMM off by {err:.3e}");
    }
}

/// A subject the value path repels (`1e20`) has no curvature to give: each plan
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
        hazard, recurrence, ..
    }) = tte.endpoints.get(&2)
    else {
        panic!("fixture precondition: a TTE endpoint on CMT 2");
    };
    let ts = Subject {
        id: "1".into(),
        obs_records: vec![event(3.0, EventType::Exact, 0.0)],
        ..Default::default()
    };
    let theta = tte.default_params.theta.clone();
    let v = crate::survival::tte_data_term(
        &ts.obs_records,
        hazard,
        *recurrence,
        &theta,
        &[0.0],
        &ts.covariates,
    );
    assert_eq!(
        v, 1e20,
        "fixture precondition: h = 0 at an exact event is repelled"
    );
    assert!(tte_exact(&tte, &ts, &[0.0]).is_none());
    assert!(whole_term(&tte, &ts, &[0.0]).is_none());

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
    assert!(whole_term(&binary, &bs, &[1.0]).is_none());

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
        assert!(whole_term(&ctmm, &cs, &[-1.0]).is_none());
    }
}
