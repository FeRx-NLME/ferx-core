//! #2706: the thetas the mu-scale channels carry are recorded as constants and their
//! gradient read off the η-jets (`ChannelPlan`). The oracle is the same adjoint with every
//! theta taped (`THETA_CHANNELS_OFF`), whose θ-derivatives come from the reverse sweep through
//! the solve — a route the channel identity does not share. Each parity test also asserts
//! that the reduced path actually ran (`THETA_CHANNEL_HITS`), so a fixture that silently falls
//! back to the full tape cannot pass as a check of the reduction.

use super::tests::{ebe_for, moved_params, subject_shell};
use super::*;
use crate::estimation::parameterization::pack_params;
use crate::parser::model_parser::parse_model_string;
use crate::types::{DoseEvent, Subject};
use std::collections::HashMap;

fn carried(src: &str) -> Option<Vec<bool>> {
    let model = parse_model_string(src).unwrap();
    model
        .indiv_param_partials
        .theta_channels
        .as_ref()
        .map(|c| c.carried.clone())
}

const HEAD: &str = "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  theta TVV(30.0, 1.0, 500.0)\n  \
    theta TVKA(1.2, 0.05, 20.0)\n  theta TH_WT(0.75, -3.0, 5.0)\n  omega ETA_CL ~ 0.09\n  \
    omega ETA_V ~ 0.06\n  omega ETA_KA ~ 0.2\n  sigma PROP_ERR ~ 0.1 (sd)\n";

/// The three channels every exclusion fixture starts from: CL with a covariate exponent.
const CHANNELS: &str = "  CL = TVCL * (WT / 70)^TH_WT * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n  \
    KA = TVKA * exp(ETA_KA)\n";

/// `[individual_parameters]` lines into a one-compartment oral model; `F1`, when assigned, is
/// its bioavailability — a route into the predictions outside the channels.
fn src(indiv: &str) -> String {
    let f = if indiv.contains("F1 =") { ", f=F1" } else { "" };
    format!(
        "{HEAD}[individual_parameters]\n{indiv}[structural_model]\n  \
         pk one_cpt_oral(cl=CL, v=V, ka=KA{f})\n[error_model]\n  DV ~ proportional(PROP_ERR)\n"
    )
}

/// The exclusion fixtures, each one route away from the served base: `(name, lines, served)`.
/// A wrongly admitted theta has a route the channel identity does not see, so these double as
/// the gradient fixtures of `every_exclusion_fixture_keeps_the_taped_gradient`.
fn exclusion_fixtures() -> Vec<(&'static str, String, bool)> {
    vec![
        ("base", CHANNELS.to_string(), true),
        // A local typical value is inlined, as the mu-reference scans do.
        (
            "local",
            CHANNELS.replace(
                "  CL = TVCL * (WT / 70)^TH_WT",
                "  TVC = TVCL * (WT / 70)^TH_WT\n  CL = TVC",
            ),
            true,
        ),
        // A theta another parameter reads (`F1`) is not carried.
        (
            "shared theta",
            format!("{CHANNELS}  F1 = 0.5 + 0.1 * TH_WT\n"),
            false,
        ),
        // A theta reaching the predictions through a local another parameter also reads.
        (
            "shared local",
            CHANNELS.replace("  CL = TVCL", "  TVC = TVCL * 2\n  CL = TVC")
                + "  F1 = 0.5 + 0.001 * TVC\n",
            false,
        ),
        // An η a second parameter reads: CL is no channel.
        (
            "shared eta",
            format!("{CHANNELS}  F1 = exp(0.2 * ETA_CL) / 2\n"),
            false,
        ),
        // A parameter assigned twice is no channel.
        (
            "assigned twice",
            format!("{CHANNELS}  CL = CL * 1.1\n"),
            false,
        ),
        // `TIME` in the typical value: no channel.
        (
            "time",
            CHANNELS.replace("(WT / 70)^TH_WT", "(WT / 70)^TH_WT * (1 + 0 * TIME)"),
            false,
        ),
        // A theta with no random effect stays a tape leaf, and so does the solve.
        (
            "theta without eta",
            CHANNELS.replace("TVKA * exp(ETA_KA)", "TVKA"),
            false,
        ),
    ]
}

/// Each exclusion of `detect_theta_channels`, against the same model with the exclusion
/// removed, so every case straddles its rule.
#[test]
fn a_theta_is_carried_only_through_exclusive_mu_scale_channels() {
    for (name, indiv, served) in exclusion_fixtures() {
        let want = served.then(|| vec![true; 4]);
        assert_eq!(carried(&src(&indiv)), want, "{name}");
    }
    // A logit channel carries its theta too.
    let logit = "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  theta TVV(30.0, 1.0, 500.0)\n  \
        theta TVKA(1.2, 0.05, 20.0)\n  theta LGF(0.5, -5.0, 5.0)\n  omega ETA_CL ~ 0.09\n  \
        omega ETA_V ~ 0.06\n  omega ETA_KA ~ 0.2\n  omega ETA_F ~ 0.2\n  sigma PROP_ERR ~ 0.1 (sd)\n\
        [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n  \
        KA = TVKA * exp(ETA_KA)\n  F = inv_logit(LGF + ETA_F)\n\
        [structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA, f=F)\n\
        [error_model]\n  DV ~ proportional(PROP_ERR)\n";
    assert_eq!(carried(logit), Some(vec![true; 4]));
    // A theta whose taint another block reads (here `[odes]`, through the local `TVC`) is
    // taped; the twin whose `[odes]` does not read `TVC` carries it.
    let ode = |rhs: &str| {
        format!(
            "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  theta TVV(30.0, 1.0, 500.0)\n  \
             omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.06\n  sigma PROP_ERR ~ 0.1 (sd)\n\
             [individual_parameters]\n  TVC = TVCL * 1\n  CL = TVC * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n\
             [structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = {rhs}\n\
             [scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n"
        )
    };
    assert_eq!(carried(&ode("-(CL/V) * central")), Some(vec![true, true]));
    assert_eq!(carried(&ode("-(CL/V) * central * TVC / TVC")), None);
}

/// A two-dose subject, covariates `covs` constant, observations from `eta_ref` with residuals
/// of both signs.
fn static_subject(model: &CompiledModel, covs: &[(&str, f64)], eta_ref: &[f64]) -> Subject {
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.5, 14.0, 18.0, 24.0];
    let mut s = subject_shell(doses, &times);
    s.covariates = covs.iter().map(|(n, v)| (n.to_string(), *v)).collect();
    if model.n_kappa > 0 {
        s.occasions = times
            .iter()
            .map(|&t| if t < 12.0 { 1 } else { 2 })
            .collect();
        s.dose_occasions = vec![1, 2];
    }
    let preds =
        crate::pk::compute_predictions_with_tv(model, &s, &model.default_params.theta, eta_ref);
    s.observations = preds
        .iter()
        .enumerate()
        .map(|(j, p)| p * (1.0 + 0.15 * (1.3 * j as f64).sin()))
        .collect();
    s
}

/// The packed gradient (and, with `dx`, the mode response) with the channels on and off,
/// and how many recordings took the channels.
fn both_ways(
    model: &CompiledModel,
    s: &Subject,
    objective: Objective,
    dx: bool,
) -> (
    (Vec<f64>, Option<Vec<DVector<f64>>>),
    (Vec<f64>, Option<Vec<DVector<f64>>>),
    usize,
) {
    let params = moved_params(model);
    let x = pack_params(&params);
    let eta = ebe_for(model, s, &params, 1e-12, objective != Objective::Foce);
    let pt = PackedPoint::new(&params, &x).unwrap();
    let go = || {
        if dx {
            let (g, d) =
                subject_adjoint_gradient_and_eta_dx_at(model, s, &pt, eta.as_slice(), objective);
            (g.expect("in scope"), d)
        } else {
            (
                subject_adjoint_gradient(model, s, &params, &x, eta.as_slice(), objective)
                    .expect("in scope"),
                None,
            )
        }
    };
    THETA_CHANNEL_HITS.with(|c| c.set(0));
    let on = go();
    let hits = THETA_CHANNEL_HITS.with(|c| c.get());
    THETA_CHANNELS_OFF.with(|c| c.set(true));
    let off = go();
    THETA_CHANNELS_OFF.with(|c| c.set(false));
    (on, off, hits)
}

fn worst_rel(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let gaps: Vec<f64> = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / (1.0 + y.abs()))
        .collect();
    assert!(gaps.iter().all(|e| e.is_finite()), "{a:?} vs {b:?}");
    gaps.into_iter().fold(0.0, f64::max)
}

/// Closed form, covariate exponent, block Ω, combined error; and its ODE twin.
const COV_ORAL: &str =
    "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  theta TVV(30.0, 1.0, 500.0)\n  \
    theta TVKA(1.2, 0.05, 20.0)\n  theta TH_WT(0.75, -3.0, 5.0)\n  theta TH_CR(0.3, -3.0, 5.0)\n  \
    block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.06]\n  omega ETA_KA ~ 0.2\n  \
    sigma PROP_ERR ~ 0.1 (sd)\n  sigma ADD_ERR ~ 0.05 (sd)\n[individual_parameters]\n  \
    CL = TVCL * (WT / 70)^TH_WT * (CRCL / 100)^TH_CR * exp(ETA_CL)\n  \
    V = TVV * (WT / 70) * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n";
const ORAL_CF: &str = "[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n\
    [error_model]\n  DV ~ combined(PROP_ERR, ADD_ERR)\n";
const ORAL_ODE: &str = "[structural_model]\n  ode(states=[depot, central])\n[odes]\n  \
    d/dt(depot) = -KA * depot\n  d/dt(central) = KA * depot - (CL/V) * central\n\
    [scaling]\n  y = central / V\n[error_model]\n  DV ~ combined(PROP_ERR, ADD_ERR)\n";

/// The reduced gradient equals the fully taped one on closed-form and ODE models, FOCEI and
/// FOCE, with the mode response's lanes, for every theta the covariate exponents included.
#[test]
fn carried_thetas_read_off_the_eta_jets_match_the_taped_gradient() {
    for structural in [ORAL_CF, ORAL_ODE] {
        let model =
            super::tests::tight(parse_model_string(&format!("{COV_ORAL}{structural}")).unwrap());
        assert_eq!(
            model
                .indiv_param_partials
                .theta_channels
                .as_ref()
                .map(|c| c.carried.clone()),
            Some(vec![true; 5]),
            "premise: every theta is carried"
        );
        let s = static_subject(&model, &[("WT", 82.0), ("CRCL", 74.0)], &[0.1, -0.05, 0.2]);
        // FOCE with a combined error keeps `f(η = 0)`, so FOCE runs on an additive twin below.
        for (objective, dx) in [(Objective::Focei, false), (Objective::Focei, true)] {
            let ((g_on, d_on), (g_off, d_off), hits) = both_ways(&model, &s, objective, dx);
            assert!(hits > 0, "the reduction must have served the recording");
            let w = worst_rel(&g_on, &g_off);
            // Same derivatives by two routes, round-off only: measured worst 2.4e-15 here and
            // 2.8e-14 across this file (FOCE), so 1e-12 keeps ~35× headroom.
            assert!(
                w < 1e-12,
                "{objective:?} dx={dx}: worst gradient gap {w:.2e}"
            );
            // The carried thetas' entries are live, so the parity is not on zeros.
            assert!(g_off[..5].iter().all(|v| v.abs() > 1e-6), "{g_off:?}");
            if dx {
                let (a, b) = (d_on.expect("response"), d_off.expect("response"));
                for (ca, cb) in a.iter().zip(&b) {
                    let w = worst_rel(ca.as_slice(), cb.as_slice());
                    assert!(w < 1e-12, "mode response gap {w:.2e}");
                }
            }
        }
    }
}

/// FOCE, with and without `f(η = 0)` in the variance: both served, the population
/// prediction carrying its η-tangents so the carried thetas' derivative through it is read
/// off them too.
#[test]
fn foce_reads_the_population_prediction_off_its_eta_tangents() {
    for error in ["additive(ADD_ERR)", "combined(PROP_ERR, ADD_ERR)"] {
        let mut src = format!(
            "{COV_ORAL}{}",
            ORAL_CF.replace("combined(PROP_ERR, ADD_ERR)", error)
        );
        if error.starts_with("additive") {
            src = src.replace("  sigma PROP_ERR ~ 0.1 (sd)\n", "");
        }
        let model = parse_model_string(&src).unwrap();
        let s = static_subject(&model, &[("WT", 82.0), ("CRCL", 74.0)], &[0.1, -0.05, 0.2]);
        let ((g_on, _), (g_off, _), hits) = both_ways(&model, &s, Objective::Foce, false);
        assert!(hits > 0, "{error}");
        let w = worst_rel(&g_on, &g_off);
        assert!(w < 1e-12, "{error}: worst gradient gap {w:.2e}");
    }
}

/// A channel covariate that varies within the subject moves `A_j` with time: the subject is
/// declined, and the gradient is the taped one.
#[test]
fn a_time_varying_channel_covariate_declines_the_subject() {
    let model = parse_model_string(&format!("{COV_ORAL}{ORAL_CF}")).unwrap();
    let mut s = static_subject(&model, &[("WT", 82.0), ("CRCL", 74.0)], &[0.1, -0.05, 0.2]);
    let ((_, _), (_, _), hits) = both_ways(&model, &s, Objective::Focei, false);
    assert!(hits > 0, "premise: the static twin is served");
    s.obs_covariates = s
        .obs_times
        .iter()
        .map(|&t| {
            HashMap::from([
                ("WT".to_string(), if t < 12.0 { 82.0 } else { 90.0 }),
                ("CRCL".to_string(), 74.0),
            ])
        })
        .collect();
    let ((g_on, _), (g_off, _), hits) = both_ways(&model, &s, Objective::Focei, false);
    assert_eq!(hits, 0);
    assert_eq!(g_on, g_off);
}

/// A logit channel (bioavailability) and an IOV channel `exp(ETA_CL + KAPPA_CL)`, plain and
/// weighted: the κ shifts each occasion's `u` with η, so `∂/∂η` still carries the theta.
#[test]
fn logit_and_iov_channels_match_the_taped_gradient() {
    let logit = "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  theta TVV(30.0, 1.0, 500.0)\n  \
        theta TVKA(1.2, 0.05, 20.0)\n  theta LGF(0.5, -5.0, 5.0)\n  omega ETA_CL ~ 0.09\n  \
        omega ETA_V ~ 0.06\n  omega ETA_KA ~ 0.2\n  omega ETA_F ~ 0.2\n  sigma PROP_ERR ~ 0.1 (sd)\n\
        [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n  \
        KA = TVKA * exp(ETA_KA)\n  F = inv_logit(LGF + ETA_F)\n\
        [structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA, f=F)\n\
        [error_model]\n  DV ~ proportional(PROP_ERR)\n";
    let iov = "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  theta TVV(30.0, 1.0, 500.0)\n  \
        theta TVKA(1.2, 0.05, 20.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.06\n  \
        omega ETA_KA ~ 0.2\n  kappa KAPPA_CL ~ 0.04\n  sigma PROP_ERR ~ 0.1 (sd)\n\
        [individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n  \
        KA = TVKA * exp(ETA_KA)\n\
        [structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n\
        [error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  iov_column = OCC\n";
    // A weighted κ (#1031) is rewritten to `KAPPA_CL / sqrt(W)` inside the link: a link
    // offset the channel detection looks through (`strip_link_offsets`).
    let weighted = iov.replace("kappa KAPPA_CL ~ 0.04", "kappa KAPPA_CL ~ 0.04 weight = 4");
    for (name, src, eta_ref) in [
        ("logit", logit.to_string(), vec![0.1, -0.05, 0.2, 0.3]),
        ("iov", iov.to_string(), vec![0.1, -0.05, 0.2, 0.05]),
        ("weighted iov", weighted, vec![0.1, -0.05, 0.2, 0.05]),
    ] {
        let model = parse_model_string(&src).unwrap();
        let s = static_subject(&model, &[], &eta_ref);
        let ((g_on, _), (g_off, _), hits) = both_ways(&model, &s, Objective::Focei, false);
        assert!(
            hits > 0,
            "{name}: the reduction must have served the recording"
        );
        let w = worst_rel(&g_on, &g_off);
        assert!(w < 1e-12, "{name}: worst gradient gap {w:.2e}");
    }
}

/// The predicate's exclusions as gradients: on each fixture of
/// `a_theta_is_carried_only_through_exclusive_mu_scale_channels` the gradient equals the
/// taped one, and only the served fixtures take the reduction. A theta wrongly admitted past
/// an exclusion has a route the channel identity does not see (`F1`, a second `CL`), so its
/// entry breaks parity here rather than only relabelling the predicate.
#[test]
fn every_exclusion_fixture_keeps_the_taped_gradient() {
    for (name, indiv, served) in exclusion_fixtures() {
        let model = parse_model_string(&src(&indiv)).unwrap();
        let s = static_subject(&model, &[("WT", 82.0)], &[0.1, -0.05, 0.2]);
        let ((g_on, _), (g_off, _), hits) = both_ways(&model, &s, Objective::Focei, false);
        // Parity first, so a wrongly admitted theta reports the wrong gradient it causes.
        let w = worst_rel(&g_on, &g_off);
        assert!(w < 1e-12, "{name}: worst gradient gap {w:.2e}");
        assert_eq!(hits > 0, served, "{name}");
    }
}

/// Nested levels (#1082): the shared fixture with a BSV η on KA too and its arm κ
/// unweighted, so every theta is carried and the reduction runs (asserted), against the taped gradient — closed form and
/// ODE, FOCEI and FOCE — at the compact EBE `find_ebe_for_stage` returns.
#[test]
fn nested_levels_take_the_channels_and_match_the_taped_gradient() {
    use crate::estimation::inner_optimizer::find_ebe_for_stage;
    use crate::stats::nested_levels::test_helpers as fx;
    // The arm κ unweighted: the weighted-κ rewrite (#1031) takes `V` out of the mu shape,
    // so a weighted fixture carries no `TVV` and takes the taped path (it still nests).
    let with_eta_ka = |src: String| {
        src.replace("KA = TVKA * exp(K_B)", "KA = TVKA * exp(ETA_KA + K_B)")
            .replace(" at ARM weight = NARM", " at ARM")
            .replace(
                "omega ETA_V ~ 0.04\n",
                "omega ETA_V ~ 0.04\n  omega ETA_KA ~ 0.2\n",
            )
    };
    for (label, src) in [("closed form", fx::closed_form()), ("ode", fx::ode())] {
        let (model, pop) = fx::prepared(&with_eta_ka(src), 1);
        assert!(model.has_nested_levels(), "{label}: the fixture must nest");
        let s = &pop.subjects[0];
        for interaction in [true, false] {
            let p = moved_params(&model);
            let x = pack_params(&p);
            let stage = crate::types::FitOptions {
                method: if interaction {
                    crate::types::EstimationMethod::FoceI
                } else {
                    crate::types::EstimationMethod::Foce
                },
                interaction,
                inner_tol: 1e-10,
                inner_maxiter: 2000,
                ..crate::types::FitOptions::default()
            };
            let ebe = find_ebe_for_stage(&model, s, &p, 2000, 1e-10, None, None, 0, &stage);
            let mut stacked = ebe.eta.as_slice().to_vec();
            for k in &ebe.kappas {
                stacked.extend_from_slice(k.as_slice());
            }
            let objective = if interaction {
                Objective::Focei
            } else {
                Objective::Foce
            };
            let go = || subject_adjoint_gradient(&model, s, &p, &x, &stacked, objective);
            THETA_CHANNEL_HITS.with(|c| c.set(0));
            let on = go().expect("in scope");
            let hits = THETA_CHANNEL_HITS.with(|c| c.get());
            THETA_CHANNELS_OFF.with(|c| c.set(true));
            let off = go().expect("in scope");
            THETA_CHANNELS_OFF.with(|c| c.set(false));
            let case = format!("{label} interaction={interaction}");
            assert!(
                hits > 0,
                "{case}: the reduction must have served the recording"
            );
            let w = worst_rel(&on, &off);
            assert!(w < 1e-12, "{case}: worst gradient gap {w:.2e}");
        }
    }
}

/// A dose attribute the ODE engine reads by name (`ALAG1`, `F1`), with no other block
/// spelling it, is a route to the data (#2711 review): a theta it reads is not carried. The
/// twin whose attribute reads no theta is served, and the gradient is the taped one on both
/// — a wrongly carried TVKA would drop the lag's (or the fraction's) θ-derivative.
#[test]
fn an_engine_read_dose_attribute_is_a_route_to_the_data() {
    for (attr, reads, free) in [
        ("ALAG1", "ALAG1 = 0.1 * TVKA", "ALAG1 = 0.1"),
        ("F1", "F1 = 0.4 + 0.1 * TVKA", "F1 = 0.6"),
    ] {
        for (line, served) in [(reads, false), (free, true)] {
            let src = format!("{COV_ORAL}  {line}\n{ORAL_ODE}");
            let model = parse_model_string(&src).unwrap();
            let s = static_subject(&model, &[("WT", 82.0), ("CRCL", 74.0)], &[0.1, -0.05, 0.2]);
            let ((g_on, _), (g_off, _), hits) = both_ways(&model, &s, Objective::Focei, false);
            // Parity first, so a wrongly carried theta reports the gradient it breaks.
            let w = worst_rel(&g_on, &g_off);
            assert!(w < 1e-12, "{attr} `{line}`: worst gradient gap {w:.2e}");
            assert_eq!(hits > 0, served, "{attr}: `{line}`");
            assert_eq!(
                model.indiv_param_partials.theta_channels.is_some(),
                served,
                "{attr}: `{line}`"
            );
        }
    }
}
