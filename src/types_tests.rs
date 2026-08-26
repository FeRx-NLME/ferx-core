use super::*;
use approx::assert_relative_eq;

/// Phase 4.0: the `Category` / `Count` `SimOutcome` variants compile
/// unconditionally (shared categorical/count/Markov draw shapes) and derive
/// Debug/Clone. The Gaussian path is unchanged.
#[test]
fn sim_outcome_category_and_count_variants_construct() {
    let c = SimOutcome::Category { state: 2 };
    let n = SimOutcome::Count { count: 5 };
    assert!(format!("{c:?}").contains("Category"));
    assert!(format!("{n:?}").contains("Count"));
    let _ = c.clone();
    let _ = n.clone();
    assert_eq!(
        SimOutcome::Continuous { value: 4.25 }.continuous_value(),
        4.25
    );
}

/// `continuous_value()` returns NAN for the non-Gaussian outcomes. The
/// misuse `debug_assert` fires in debug builds, so this is asserted only when
/// debug-assertions are off — the profile CI and coverage use (`ci-test`,
/// which inherits `release`), where the NAN branch is actually taken.
#[test]
#[cfg(not(debug_assertions))]
fn sim_outcome_category_and_count_continuous_value_is_nan() {
    assert!(SimOutcome::Category { state: 3 }
        .continuous_value()
        .is_nan());
    assert!(SimOutcome::Count { count: 9 }.continuous_value().is_nan());
}

/// The TTE `Event` outcome likewise has no continuous value. Same profile
/// caveat as above (the debug-assert fires in debug builds).
#[test]
#[cfg(all(feature = "survival", not(debug_assertions)))]
fn sim_outcome_event_continuous_value_is_nan() {
    assert!(SimOutcome::Event {
        time: 1.0,
        observed: true,
    }
    .continuous_value()
    .is_nan());
}

/// `effective_cov_inner_tol` resolves the covariance-step reconvergence tolerance:
/// an explicit `cov_inner_tol` wins; otherwise the fit's `inner_tol`, tightened to
/// at most `LTBS_COV_INNER_TOL` (1e-8) for LTBS models only (non-LTBS unchanged).
#[test]
fn effective_cov_inner_tol_resolution() {
    let mut o = FitOptions::default();
    o.cov_inner_tol = None;
    o.inner_tol = 1e-5;
    // Non-LTBS: unchanged (byte-identical to before this option).
    assert_relative_eq!(o.effective_cov_inner_tol(false), 1e-5);
    // LTBS: tightened to the 1e-8 default.
    assert_relative_eq!(o.effective_cov_inner_tol(true), 1e-8);
    // A fit already tighter than 1e-8 keeps its own tolerance (LTBS).
    o.inner_tol = 1e-10;
    assert_relative_eq!(o.effective_cov_inner_tol(true), 1e-10);
    // Explicit override always wins for either model kind.
    o.inner_tol = 1e-5;
    o.cov_inner_tol = Some(1e-5);
    assert_relative_eq!(o.effective_cov_inner_tol(false), 1e-5);
    assert_relative_eq!(o.effective_cov_inner_tol(true), 1e-5);
}

/// `effective_inner_tol` tightens the fit's inner tolerance to at most 1e-6 for the
/// closed-form LTBS path (reproducible flat-direction SEs), leaves other
/// models untouched, never loosens an already-tighter setting, and — crucially —
/// honours an explicit user `inner_tol` (`user_set_keys`).
#[test]
fn effective_inner_tol_ltbs() {
    let mut o = FitOptions::default();
    o.inner_tol = 1e-5;
    // Non-(closed-form LTBS): unchanged.
    assert_relative_eq!(o.effective_inner_tol(false), 1e-5);
    // Closed-form LTBS with the default inner_tol: tightened to the 1e-6 ceiling.
    assert_relative_eq!(o.effective_inner_tol(true), 1e-6);
    // A fit already tighter than 1e-6 is kept.
    o.inner_tol = 1e-8;
    assert_relative_eq!(o.effective_inner_tol(true), 1e-8);
    // An EXPLICIT user inner_tol (even looser) is honoured, not overridden (#665 review).
    o.inner_tol = 1e-4;
    o.user_set_keys.push("inner_tol".to_string());
    assert_relative_eq!(o.effective_inner_tol(true), 1e-4);
}

/// Transit (`OneCptTransit`) descriptors: canonical name, the analytic 2-state
/// `[depot, central]` layout, and no modeled-duration infusions (#386).
#[test]
fn one_cpt_transit_descriptors() {
    assert_eq!(PkModel::OneCptTransit.canonical_name(), "one_cpt_transit");
    assert!(PkModel::OneCptTransit.infusable_compartments().is_empty());
}

/// Transit (`TwoCptTransit`) descriptors: canonical name + alias round-trip, the
/// analytic 3-state `[depot, central, peripheral]` layout, required params
/// (cl/v1/q/v2/n/mtt), oral routing, and no modeled-duration infusions (#386 PR D).
#[test]
fn two_cpt_transit_descriptors() {
    assert_eq!(PkModel::TwoCptTransit.canonical_name(), "two_cpt_transit");
    assert_eq!(
        PkModel::from_name("two_cpt_transit"),
        Some(PkModel::TwoCptTransit)
    );
    assert_eq!(
        PkModel::from_name("two_compartment_transit"),
        Some(PkModel::TwoCptTransit)
    );
    assert!(PkModel::TwoCptTransit.is_oral());
    assert!(PkModel::TwoCptTransit.infusable_compartments().is_empty());
    let req: Vec<&str> = PkModel::TwoCptTransit
        .required_pk_params()
        .iter()
        .map(|(_, n)| *n)
        .collect();
    assert_eq!(req, vec!["cl", "v1", "q", "v2", "n", "mtt"]);
}

/// `sim_residual_variance` must split FREM covariate pseudo-observations
/// (FREMTYPE>0) off the PK error model: they use the additive covariate
/// sigma (EPSCOV) and ignore the IIV-on-RUV scaling, while ordinary PK rows
/// use the PK error model scaled by `ruv_scale`. Regression for the FREM
/// simulation corruption introduced when sim/NPDE moved onto the TV-cov
/// dispatcher (which writes the θ+η override into FREM rows).
#[test]
fn sim_residual_variance_splits_frem_rows_from_pk_error() {
    let mut model = test_helpers::analytical_model(GradientMethod::Fd);
    // Proportional PK error: V_pk = (f·σ)².
    model.error_model = ErrorModel::Proportional;
    model.error_spec = ErrorSpec::Single(ErrorModel::Proportional);
    // FREMTYPE 100 maps to (θ[0], η[0]); the covariate sigma sits at index 1.
    model.frem_config = Some(FremConfig {
        fremtype_to_indices: std::collections::HashMap::from([(100u16, (0usize, 0usize))]),
        covariate_sigma_index: 1,
    });

    let subject = Subject {
        id: "S".into(),
        doses: vec![],
        obs_times: vec![1.0, 2.0],
        obs_raw_times: vec![],
        observations: vec![0.0, 0.0],
        obs_cmts: vec![1, 1],
        covariates: std::collections::HashMap::new(),
        dose_covariates: vec![],
        obs_covariates: vec![],
        pk_only_times: vec![],
        pk_only_covariates: vec![],
        reset_times: vec![],
        cens: vec![0, 0],
        occasions: vec![],
        obs_l2: Vec::new(),
        dose_occasions: vec![],
        // Row 0 = PK observation, row 1 = covariate pseudo-observation.
        fremtype: vec![0, 100],
        obs_records: vec![],
    };

    let sigma = vec![0.3, 0.5]; // [pk_prop_sigma, frem_cov_sigma]
    let ruv = 4.0; // exercise the IIV-on-RUV scaling on the PK row
    let f = 2.0;

    // PK row: (f·σ_pk)² · ruv_scale.
    let pk_var = model.sim_residual_variance(&subject, 0, f, &sigma, ruv, None);
    assert!((pk_var - (f * 0.3) * (f * 0.3) * ruv).abs() < 1e-12);

    // FREM row: covariate σ², independent of `f_pred` and `ruv_scale`.
    let frem_var = model.sim_residual_variance(&subject, 1, 999.0, &sigma, ruv, None);
    assert!((frem_var - 0.5 * 0.5).abs() < 1e-12);

    // Without a FREM config every row uses the PK error model.
    model.frem_config = None;
    let pk_only = model.sim_residual_variance(&subject, 1, f, &sigma, 1.0, None);
    assert!((pk_only - (f * 0.3) * (f * 0.3)).abs() < 1e-12);
}

#[test]
fn sim_residual_variance_applies_custom_magnitude() {
    // #484 review #3: simulate() must draw residual error with the custom
    // magnitude, not a constant SD. A multiplier m on the proportional slot
    // scales the variance by m²; an all-ones / None multiplier is the legacy
    // variance exactly.
    let mut model = test_helpers::analytical_model(GradientMethod::Fd);
    model.error_model = ErrorModel::Proportional;
    model.error_spec = ErrorSpec::Single(ErrorModel::Proportional);
    model.frem_config = None;

    let subject = Subject {
        id: "S".into(),
        doses: vec![],
        obs_times: vec![1.0],
        obs_raw_times: vec![],
        observations: vec![0.0],
        obs_cmts: vec![1],
        covariates: std::collections::HashMap::new(),
        dose_covariates: vec![],
        obs_covariates: vec![],
        pk_only_times: vec![],
        pk_only_covariates: vec![],
        reset_times: vec![],
        cens: vec![0],
        occasions: vec![],
        obs_l2: Vec::new(),
        dose_occasions: vec![],
        fremtype: vec![0],
        obs_records: vec![],
    };
    let sigma = vec![0.3];
    let f = 2.0;

    // Bare row: (f·σ)².
    let bare = model.sim_residual_variance(&subject, 0, f, &sigma, 1.0, None);
    assert!((bare - (f * 0.3) * (f * 0.3)).abs() < 1e-12);

    // Magnitude 2 on the proportional slot: (f·2·σ)² = 4·bare.
    let scaled = model.sim_residual_variance(&subject, 0, f, &sigma, 1.0, Some(&[2.0]));
    assert!((scaled - 4.0 * (f * 0.3) * (f * 0.3)).abs() < 1e-12);

    // An all-ones multiplier reproduces the bare variance.
    let unit = model.sim_residual_variance(&subject, 0, f, &sigma, 1.0, Some(&[1.0]));
    assert!((unit - bare).abs() < 1e-12);
}

#[test]
fn resolve_auto_picks_nlopt_lbfgs_when_analytic() {
    // Analytical PK model with `gradient = auto` → exact FOCE gradient is
    // available, so `auto` resolves to the gradient-based NLopt L-BFGS (#490).
    let m = test_helpers::analytical_model(GradientMethod::Auto);
    assert_eq!(
        Optimizer::Auto.resolve_auto(&m, true),
        Optimizer::NloptLbfgs,
        "analytic-gradient model should resolve auto → nlopt_lbfgs"
    );
}

#[test]
fn resolve_auto_picks_bobyqa_when_fd_only() {
    // ODE model (no analytic scope) and an analytical model with
    // `gradient = fd` both fall back to finite differences → bobyqa.
    let ode = test_helpers::ode_model(GradientMethod::Auto);
    let forced_fd = test_helpers::analytical_model(GradientMethod::Fd);
    for m in [&ode, &forced_fd] {
        assert_eq!(
            Optimizer::Auto.resolve_auto(m, true),
            Optimizer::Bobyqa,
            "FD-only model should resolve auto → bobyqa"
        );
    }
}

/// #486 σ-magnitude FOCE port: a custom residual-magnitude model is analytic on
/// **both** loops now (the Sheiner–Beal FOCE marginal threads `mult(θ)` through
/// its `R⁰`), so `auto` resolves to `NloptLbfgs` under plain `method = foce`
/// (`interaction = false`) as well as under FOCEI (`interaction = true`).
#[test]
fn resolve_auto_analytic_for_custom_magnitude_both_loops() {
    let content = "[parameters]\n  theta TVCL(0.2)\n  theta TVV(10.0)\n  theta RUV_LATE(1.5, 0.0, 10.0)\n  omega ETA_CL ~ 0.09\n  sigma PROP_ERR ~ 0.04\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(PROP_ERR * (1.0 + RUV_LATE * TIME / 48.0))\n";
    let m = crate::parser::model_parser::parse_model_string(content).expect("parse");
    assert!(m.has_custom_ruv_magnitude());
    assert!(crate::sens::provider::analytic_outer_gradient_available(&m));
    assert_eq!(
        Optimizer::Auto.resolve_auto(&m, false),
        Optimizer::NloptLbfgs,
        "FOCE + custom magnitude now has the analytic gradient"
    );
    assert_eq!(
        Optimizer::Auto.resolve_auto(&m, true),
        Optimizer::NloptLbfgs,
        "FOCEI + custom magnitude has the analytic gradient"
    );
}

#[test]
fn resolve_auto_is_identity_for_concrete_optimizers() {
    // Every non-`Auto` optimizer is returned unchanged, regardless of model.
    let m = test_helpers::analytical_model(GradientMethod::Auto);
    for opt in [
        Optimizer::Bfgs,
        Optimizer::Lbfgs,
        Optimizer::Slsqp,
        Optimizer::NloptLbfgs,
        Optimizer::Mma,
        Optimizer::Bobyqa,
        Optimizer::TrustRegion,
    ] {
        assert_eq!(opt.resolve_auto(&m, true), opt);
    }
}

#[test]
fn required_pk_params_match_docs_table() {
    // Locks the per-model required slots to the "Required Parameters" table
    // in docs/model-file/structural-model.qmd (issue #309).
    use PkModel::*;
    let cases: &[(PkModel, &[&str])] = &[
        (OneCptIv, &["cl", "v"]),
        (OneCptOral, &["cl", "v", "ka"]),
        (OneCptTransit, &["cl", "v", "n", "mtt"]),
        (OneCptIg, &["cl", "v", "mat", "cv2"]),
        (TwoCptIv, &["cl", "v1", "q", "v2"]),
        (TwoCptOral, &["cl", "v1", "q", "v2", "ka"]),
        (TwoCptTransit, &["cl", "v1", "q", "v2", "n", "mtt"]),
        (TwoCptIg, &["cl", "v1", "q", "v2", "mat", "cv2"]),
        (ThreeCptIv, &["cl", "v1", "q2", "v2", "q3", "v3"]),
        (ThreeCptOral, &["cl", "v1", "q2", "v2", "q3", "v3", "ka"]),
    ];
    for (model, expected_names) in cases {
        let req = model.required_pk_params();
        let names: Vec<&str> = req.iter().map(|(_, n)| *n).collect();
        assert_eq!(&names, expected_names, "wrong required names for {model:?}");
        // Every (slot, name) pair must be self-consistent with name_to_index,
        // so the parser's key→slot canonicalisation lines up with this table.
        for (slot, name) in req {
            assert_eq!(
                PkParams::name_to_index(name),
                Some(*slot),
                "slot/name mismatch for `{name}` in {model:?}"
            );
        }
        // f / lagtime are optional and must never appear as required.
        assert!(
            !req.iter()
                .any(|(s, _)| *s == PK_IDX_F || *s == PK_IDX_LAGTIME),
            "{model:?} must not require f/lagtime"
        );
    }
}

#[test]
fn canonical_name_covers_all_variants() {
    use PkModel::*;
    assert_eq!(OneCptIv.canonical_name(), "one_cpt_iv");
    assert_eq!(OneCptOral.canonical_name(), "one_cpt_oral");
    assert_eq!(OneCptTransit.canonical_name(), "one_cpt_transit");
    assert_eq!(OneCptIg.canonical_name(), "one_cpt_ig");
    assert_eq!(TwoCptIv.canonical_name(), "two_cpt_iv");
    assert_eq!(TwoCptOral.canonical_name(), "two_cpt_oral");
    assert_eq!(TwoCptTransit.canonical_name(), "two_cpt_transit");
    assert_eq!(TwoCptIg.canonical_name(), "two_cpt_ig");
    assert_eq!(ThreeCptIv.canonical_name(), "three_cpt_iv");
    assert_eq!(ThreeCptOral.canonical_name(), "three_cpt_oral");
}

#[test]
fn from_name_round_trips_and_accepts_aliases() {
    use PkModel::*;
    // `from_name` is the single source of name → model for both the `pk` parser
    // and the `ode_template` desugarer (Ron #363). It must be the inverse of
    // `canonical_name` on the canonical spelling and accept every long-form
    // alias the parser historically accepted.
    let cases: &[(PkModel, &str)] = &[
        (OneCptIv, "one_compartment_iv"),
        (OneCptOral, "one_compartment_oral"),
        (TwoCptIv, "two_compartment_iv"),
        (TwoCptOral, "two_compartment_oral"),
        (ThreeCptIv, "three_compartment_iv"),
        (ThreeCptOral, "three_compartment_oral"),
    ];
    for (model, alias) in cases {
        assert_eq!(
            PkModel::from_name(model.canonical_name()),
            Some(*model),
            "canonical name of {model:?} must round-trip"
        );
        assert_eq!(
            PkModel::from_name(alias),
            Some(*model),
            "alias `{alias}` must resolve to {model:?}"
        );
    }
    // Unknown names and the retired `*_bolus` / `*_infusion` spellings do NOT
    // resolve — the parser maps the retired ones to a migration error itself,
    // so `from_name` must return `None` for them (not a wrong variant).
    for none in [
        "four_cpt_oral",
        "one_cpt_iv_bolus",
        "two_cpt_infusion",
        "",
        "pk",
    ] {
        assert_eq!(PkModel::from_name(none), None, "`{none}` must not resolve");
    }
}

#[test]
fn error_spec_has_f_dependent_variance() {
    use std::collections::HashMap;
    // Single endpoint: only additive is f-independent.
    assert!(!ErrorSpec::Single(ErrorModel::Additive).has_f_dependent_variance());
    assert!(ErrorSpec::Single(ErrorModel::Proportional).has_f_dependent_variance());
    assert!(ErrorSpec::Single(ErrorModel::Combined).has_f_dependent_variance());

    // PerCmt: f-dependent if ANY endpoint is non-additive.
    let ep = |em: ErrorModel| EndpointError {
        error_model: em,
        sigma_idx: vec![0],
    };
    let mut all_additive = HashMap::new();
    all_additive.insert(1usize, ep(ErrorModel::Additive));
    all_additive.insert(2usize, ep(ErrorModel::Additive));
    assert!(!ErrorSpec::PerCmt(all_additive).has_f_dependent_variance());

    let mut mixed = HashMap::new();
    mixed.insert(1usize, ep(ErrorModel::Additive));
    mixed.insert(2usize, ep(ErrorModel::Proportional));
    assert!(ErrorSpec::PerCmt(mixed).has_f_dependent_variance());
}

#[test]
fn combined_additive_sigma_indices_picks_second_slot() {
    use std::collections::HashMap;
    // Single combined: additive component is sigma index 1.
    assert_eq!(
        ErrorSpec::Single(ErrorModel::Combined).combined_additive_sigma_indices(),
        vec![1]
    );
    // Non-combined single specs have no additive-combined slot.
    assert!(ErrorSpec::Single(ErrorModel::Additive)
        .combined_additive_sigma_indices()
        .is_empty());
    assert!(ErrorSpec::Single(ErrorModel::Proportional)
        .combined_additive_sigma_indices()
        .is_empty());

    let ep = |em: ErrorModel, idx: Vec<usize>| EndpointError {
        error_model: em,
        sigma_idx: idx,
    };

    // PerCmt: returns the global index of each combined endpoint's
    // second sigma slot, de-duplicated; non-combined endpoints contribute
    // nothing.
    let mut map = HashMap::new();
    map.insert(1usize, ep(ErrorModel::Proportional, vec![0]));
    map.insert(2usize, ep(ErrorModel::Combined, vec![1, 3]));
    let mut got = ErrorSpec::PerCmt(map).combined_additive_sigma_indices();
    got.sort_unstable();
    assert_eq!(got, vec![3]);

    // Two combined endpoints that share the same additive sigma index
    // collapse to a single entry.
    let mut shared = HashMap::new();
    shared.insert(1usize, ep(ErrorModel::Combined, vec![0, 2]));
    shared.insert(2usize, ep(ErrorModel::Combined, vec![1, 2]));
    assert_eq!(
        ErrorSpec::PerCmt(shared).combined_additive_sigma_indices(),
        vec![2]
    );
}

#[test]
fn classify_warning_convergence_is_critical() {
    let w = classify_warning("Outer optimization did not converge");
    assert_eq!(w.severity, WarningSeverity::Critical);
    assert_eq!(w.category.as_str(), "convergence");
    assert!(w.source_method.is_none());
}

#[test]
fn classify_warning_covariance_is_critical() {
    let w = classify_warning("Covariance step failed");
    assert_eq!(w.severity, WarningSeverity::Critical);
    assert_eq!(w.category.as_str(), "covariance_failed");
}

#[test]
fn classify_warning_flat_parameter_is_warning() {
    let w = classify_warning(
        "[parameters] `TVFLAT` has no effect on the objective (gradient ≈ 0 at the \
             initial estimate) — it is likely computed but never used. Freezing it at its \
             initial value (1) so the remaining parameters can be estimated.",
    );
    assert_eq!(w.severity, WarningSeverity::Warning);
    assert_eq!(w.category.as_str(), "flat_parameter");
    assert_eq!(w.source_method.as_deref(), Some("parameters"));
}

/// #1008: a declined absorption ODE twin gets its own code rather than the `general`
/// fallback, so a consumer (the R wrapper, an agent) can branch on "this model kept no
/// ODE fallback" instead of matching prose. Built through the real emitter rather than a
/// pasted copy, so a reworded message cannot drift away from the token it is keyed on.
#[test]
fn classify_warning_absorption_twin_declined_is_warning() {
    let w = classify_warning(&super::absorption_twin_declined_warning(
        "[odes]: name `CENTRAL` collides with a previously-declared state (case-insensitive); \
         state, individual-parameter, and ODE-block intermediate names must all be distinct.",
    ));
    assert_eq!(w.severity, WarningSeverity::Warning);
    assert_eq!(w.category.as_str(), "absorption_twin_declined");
}

/// The decline arm sits ahead of *every* prose arm because the message carries the twin
/// parser's own error text verbatim (#1027 review). A twin error that happens to contain a
/// substring a later arm matches — here `"ill-conditioned"`, which would otherwise classify
/// `Critical` / `condition_number` — must still read as a twin decline.
#[test]
fn classify_warning_twin_decline_beats_prose_arms_in_the_reason() {
    for reason in [
        "[odes]: the Jacobian is ill-conditioned at the initial state",
        "the fit did not converge in the re-emitted [fit_options]",
        "covariance failed to parse",
        "condition number check rejected the block",
    ] {
        let w = classify_warning(&super::absorption_twin_declined_warning(reason));
        assert_eq!(
            w.category.as_str(),
            "absorption_twin_declined",
            "a twin decline whose reason says {reason:?} must not be claimed by a prose arm"
        );
        assert_eq!(w.severity, WarningSeverity::Warning);
    }
}

/// The reason round-trips out of the warning verbatim, so the up-front rejection messages
/// in `api::validation` can quote it (#1027 review). Also pins the terminator normalisation:
/// a reason with no sentence-ending punctuation gets one, so it cannot run into the fixed
/// text around it.
#[test]
fn absorption_twin_decline_reason_round_trips() {
    let mut model = test_helpers::analytical_model(GradientMethod::Fd);
    assert_eq!(
        super::absorption_twin_decline_reason(&model),
        None,
        "a model with no decline warning has no reason"
    );
    model
        .parse_warnings
        .push(super::absorption_twin_declined_warning(
            "two parameters map to the V slot",
        ));
    assert_eq!(
        super::absorption_twin_decline_reason(&model),
        Some("two parameters map to the V slot."),
        "the reason comes back verbatim, with a terminator added"
    );
    // An already-terminated reason is not double-punctuated.
    let terminated = super::absorption_twin_declined_warning("names must all be distinct.");
    assert!(
        terminated.ends_with("names must all be distinct."),
        "got: {terminated}"
    );
}

#[test]
fn classify_warning_dw_is_warning() {
    let w = classify_warning("Positive IWRES autocorrelation detected (Durbin-Watson = 1.20).");
    assert_eq!(w.severity, WarningSeverity::Warning);
    assert_eq!(w.category.as_str(), "dw_autocorrelation");
}

#[test]
fn classify_warning_sir_proposal_diagnostics_route_to_sir() {
    // #1021: the proposal-conditioning notes are prefixed `SIR:` / `SIR fallback:`
    // and match none of the older "sir failed" / "sir requested" phrasings. The
    // shrinkage one also name-drops "the covariance step", so it must not be
    // claimed by a covariance arm.
    let w = classify_warning(
        "SIR: proposal was shrunk in 3 direction(s) so draws stay inside the parameter \
         bounds [TVKA -0.74 (sd 6.15e3 → 2.89e0)]. Those directions come from \
         eigenvalue-floored (non-identified) curvature in the covariance step; the SIR \
         CIs along them understate the true uncertainty.",
    );
    assert_eq!(w.category.as_str(), "sir", "message: {}", w.message);
    assert_eq!(w.severity, WarningSeverity::Warning);

    let w = classify_warning(
        "SIR: proposal covariance is rank-deficient beyond the FIX-ed parameters: 1 \
         direction(s) carry no uncertainty [TVCL +0.71, TVV -0.70].",
    );
    assert_eq!(w.category.as_str(), "sir", "message: {}", w.message);

    let w = classify_warning("SIR fallback: proposal was shrunk in 1 direction(s).");
    assert_eq!(w.category.as_str(), "sir", "message: {}", w.message);
}

#[test]
fn classify_warning_mu_ref_is_info() {
    let w = classify_warning("mu-ref: CL, V");
    assert_eq!(w.severity, WarningSeverity::Info);
    assert_eq!(w.category.as_str(), "mu_referencing");
}

#[test]
fn classify_warning_saem_non_mu_ref_is_warning() {
    let w = classify_warning(
        "SAEM: individual parameter(s) not mu-referenced: V. This can strongly \
             affect convergence; prefer forms such as `CL = TVCL * exp(ETA_CL)` when possible.",
    );
    assert_eq!(w.severity, WarningSeverity::Warning);
    assert_eq!(w.category.as_str(), "mu_referencing");
}

#[test]
fn classify_warning_strips_chain_prefix() {
    let w = classify_warning("[FOCEI] Covariance step failed");
    assert_eq!(w.source_method.as_deref(), Some("FOCEI"));
    assert_eq!(w.message, "Covariance step failed");
    assert_eq!(w.severity, WarningSeverity::Critical);
    assert_eq!(w.category.as_str(), "covariance_failed");
}

#[test]
fn classify_warning_unknown_falls_back_to_general() {
    let w = classify_warning("some entirely novel message");
    assert_eq!(w.severity, WarningSeverity::Warning);
    assert_eq!(w.category.as_str(), "general");
}

#[test]
fn classify_warning_simulation_beats_optimizer_health_degenerate() {
    // #762/#763 simulation diagnostics contain the word "degenerate", which the
    // generic optimizer-health branch also keys on. The `w_tte/​w_rtte` prefix
    // branch must precede it so a simulated-subject diagnostic is not mislabelled
    // as an estimation/optimizer problem.
    for msg in [
        "W_TTE_DEGENERATE_HAZARD: subject '1' (CMT=1) drew a non-positive / \
             non-finite effective hazard rate (#763).",
        "W_RTTE_DEGENERATE: subject '1' (CMT=2) has a degenerate recurrent hazard (#762).",
    ] {
        let w = classify_warning(msg);
        assert_eq!(w.category.as_str(), "simulation", "misclassified: {msg:?}");
        assert_eq!(w.severity, WarningSeverity::Warning);
        assert_ne!(w.category, WarningCode::OptimizerHealth);
    }
}

#[test]
fn classify_warning_flip_flop_beats_optimizer_health() {
    // Both absorption flip-flop warnings — the twin-carrying heads-up (#776) and
    // the twin-less EBE warning (#785) — say "flip-flop regime". The EBE message
    // also contains "degenerating", which the generic optimizer-health branch keys
    // on ("degenerate"), so the flip-flop branch must precede it to type these
    // `flip_flop`. These are representative proxies of the real emitters
    // (src/api.rs ~2596 and ~6613); the drift-proof end-to-end check on the actual
    // emitted string lives in `flip_flop_ebe_warning_fires_for_twinless_transit_crosser`.
    for msg in [
        "one_cpt_transit disposition rate (0.5000) ≥ transit rate KTR = (n+1)/mtt \
             (0.2000) at typical values (subject 3): the flip-flop regime, outside the \
             analytic absorption closed form's convergence domain. ferx automatically \
             evaluates the equivalent ODE transit model for such parameters.",
        "one_cpt_transit subject(s) [3] enter the analytic absorption closed form's \
             clamp region — the flip-flop regime (disposition rate ≥ transit rate KTR = \
             (n+1)/mtt), or (for a 2-cpt model) coincident disposition eigenvalues — at \
             their fitted empirical-Bayes estimates, silently degenerating those \
             subjects' likelihood contributions.",
    ] {
        let w = classify_warning(msg);
        assert_eq!(w.category, WarningCode::FlipFlop, "misclassified: {msg:?}");
        assert_eq!(w.category.as_str(), "flip_flop");
        assert_eq!(w.severity, WarningSeverity::Warning);
        assert_ne!(w.category, WarningCode::OptimizerHealth);
    }
}

#[test]
fn classify_warning_eta_shrinkage_is_word_bounded() {
    // The real message classifies to eta_shrinkage.
    assert_eq!(
        classify_warning("High ETA shrinkage (>= 30%): eta_V (42%)")
            .category
            .as_str(),
        "eta_shrinkage"
    );
    // A substring like "beta shrinkage" must NOT collide.
    assert_ne!(
        classify_warning("beta shrinkage looks odd")
            .category
            .as_str(),
        "eta_shrinkage"
    );
}

/// #778: the `WarningCode` serde token is a public API an agent / the R
/// wrapper pins against. This snapshot fails loudly if a variant's token
/// drifts, and asserts `as_str()` and the serde representation agree.
#[test]
fn warning_code_tokens_are_stable() {
    use WarningCode::*;
    let expected: &[(WarningCode, &str)] = &[
        (Convergence, "convergence"),
        (CovarianceStep, "covariance_step"),
        (CovarianceFailed, "covariance_failed"),
        (CovarianceRegularized, "covariance_regularized"),
        (ConditionNumber, "condition_number"),
        (OptimizerHealth, "optimizer_health"),
        (DwAutocorrelation, "dw_autocorrelation"),
        (EtaNormality, "eta_normality"),
        (Experimental, "experimental"),
        (BloqMethod, "bloq_method"),
        (Sir, "sir"),
        (ImportanceSampling, "importance_sampling"),
        (EpsShrinkage, "eps_shrinkage"),
        (EtaShrinkage, "eta_shrinkage"),
        (BoundaryEstimate, "boundary_estimate"),
        (InflatedRse, "inflated_rse"),
        (HighCorrelation, "high_correlation"),
        (DataQuality, "data_quality"),
        (OmegaStructure, "omega_structure"),
        (GradientFallback, "gradient_fallback"),
        (MuReferencing, "mu_referencing"),
        (OptimizerConfig, "optimizer_config"),
        (MultiStart, "multi_start"),
        (Cancelled, "cancelled"),
        (Threads, "threads"),
        (Simulation, "simulation"),
        (General, "general"),
    ];
    for (code, token) in expected {
        assert_eq!(code.as_str(), *token, "as_str drift for {code:?}");
        // serde token == as_str token (rename_all = "snake_case").
        assert_eq!(
            serde_json::to_value(code).unwrap(),
            serde_json::json!(token),
            "serde token drift for {code:?}"
        );
        // and it round-trips back to the same variant.
        let back: WarningCode = serde_json::from_value(serde_json::json!(token)).unwrap();
        assert_eq!(back, *code);
    }
}

/// #778: the optional `details` payload serializes when present (the
/// at-source numeric channel) and is omitted when `None`.
#[test]
fn warning_entry_details_serialize_when_present() {
    let with = WarningEntry {
        severity: WarningSeverity::Warning,
        category: WarningCode::DwAutocorrelation,
        message: "Positive IWRES autocorrelation".to_string(),
        source_method: None,
        details: Some(serde_json::json!({ "durbin_watson": 1.2 })),
    };
    let v = serde_json::to_value(&with).unwrap();
    assert_eq!(v["category"], "dw_autocorrelation");
    assert_eq!(v["details"]["durbin_watson"], 1.2);
    // round-trips
    let back: WarningEntry = serde_json::from_value(v).unwrap();
    assert_eq!(back.details, with.details);
    assert_eq!(back.category, WarningCode::DwAutocorrelation);

    // None -> key omitted entirely.
    let without = classify_warning("some novel message");
    let vo = serde_json::to_value(&without).unwrap();
    assert!(vo.get("details").is_none());
}

/// Round-trip table covering every literal warning message emitted by the
/// engine, with the (severity, category) it should classify to.
///
/// This protects the classifier contract from quiet regressions when a
/// message gets a typo fix or a wording change. If you edit a message
/// string in `outer_optimizer.rs`, `saem.rs`, `gauss_newton.rs`,
/// `trust_region.rs`, or `api.rs`, mirror the edit here so the test
/// keeps reflecting the engine's actual output.
///
/// Messages built by `format!(..)` are exercised with a representative
/// instantiation: the substrings the classifier matches on must remain
/// present after interpolation.
#[test]
fn classify_warning_roundtrips_every_engine_message() {
    use WarningSeverity::*;
    let table: &[(&str, WarningSeverity, &str)] = &[
        // -- outer_optimizer.rs ----------------------------------------
        (
            "Outer optimization did not converge",
            Critical,
            "convergence",
        ),
        ("Covariance step failed", Critical, "covariance_failed"),
        // global_search has two arms: explicit "disabled" is a runtime
        // failure (Warning); a bare mention without "disabled" is
        // informational.
        (
            "global_search disabled: bad seed",
            Warning,
            "optimizer_config",
        ),
        ("global_search reached eval cap", Info, "optimizer_config"),
        ("cancelled by user", Info, "cancelled"),
        // -- gauss_newton.rs -------------------------------------------
        (
            "Gauss-Newton: trust radius collapsed",
            Warning,
            "optimizer_health",
        ),
        (
            "Gauss-Newton: degenerate BHHH Hessian, trust radius collapsed",
            Warning,
            "optimizer_health",
        ),
        (
            "Gauss-Newton: max iterations reached without convergence",
            Critical,
            "convergence",
        ),
        // -- saem.rs ---------------------------------------------------
        (
            "Covariance step failed \u{2014} SEs not available",
            Critical,
            "covariance_failed",
        ),
        (
            "saem_n_leapfrog > 0 but HMC is unavailable (requires an analytical PK model \
                 the Dual2 gradient supports); falling back to Metropolis-Hastings",
            Info,
            "gradient_fallback",
        ),
        // -- trust_region.rs ------------------------------------------
        (
            "Trust-region did not converge: line search stalled",
            Critical,
            "convergence",
        ),
        // -- api.rs ----------------------------------------------------
        (
            "Positive IWRES autocorrelation detected (Durbin-Watson = 1.20).",
            Warning,
            "dw_autocorrelation",
        ),
        (
            "Negative IWRES autocorrelation detected (Durbin-Watson = 2.80).",
            Warning,
            "dw_autocorrelation",
        ),
        (
            "M3 censoring under FOCE uses non-interaction (Sheiner–Beal) semantics",
            Warning,
            "bloq_method",
        ),
        (
            "SIR failed: covariance not positive definite",
            Warning,
            "sir",
        ),
        (
            "SIR requested but covariance matrix is not available",
            Warning,
            "sir",
        ),
        // #972: the stranded-SIR-after-a-failed-covariance-step message names
        // the covariance step and the Hessian, both of which earlier arms in the
        // chain sniff for. It must still route to "sir", not to
        // "covariance_failed" / "optimizer_health".
        (
            "SIR requested but the covariance step did not succeed and no usable SIR \
             proposal could be built from it, so SIR could not run — see the \
             covariance warning above for the cause.",
            Warning,
            "sir",
        ),
        (
            "SIR requested but not run: Bayesian estimation reports posterior \
             credible intervals instead of a Hessian-based covariance, which SIR \
             would have to draw from.",
            Warning,
            "sir",
        ),
        // #1021: proposal-conditioning diagnostics. The shrinkage message names
        // "the covariance step", which earlier arms sniff for — it must still
        // route to "sir".
        (
            "SIR: proposal was shrunk in 3 direction(s) so draws stay inside the \
             parameter bounds [TVKA -0.74 (sd 6.15e3 → 2.89e0)]. Those directions \
             come from eigenvalue-floored (non-identified) curvature in the \
             covariance step; the SIR CIs along them understate the true uncertainty.",
            Warning,
            "sir",
        ),
        (
            "SIR: proposal covariance is rank-deficient beyond the FIX-ed \
             parameters: 1 direction(s) carry no uncertainty [TVCL +0.71, TVV -0.70]. \
             SIR holds those parameter combinations at their ML values, so their CIs \
             are not explored — they are not identified by the data.",
            Warning,
            "sir",
        ),
        (
            "SIR fallback: proposal was shrunk in 1 direction(s) so draws stay inside \
             the parameter bounds [ADD_ERR +0.99 (sd 4.36e3 → 2.17e0)].",
            Warning,
            "sir",
        ),
        (
            "IMP: 2 subject(s) had ESS = 0 (proposal collapse)",
            Warning,
            "importance_sampling",
        ),
        (
            "High ETA shrinkage (\u{2265} 30%): eta_CL (42%). EBE-based diagnostics \
                 for these random effects are unreliable.",
            Warning,
            "eta_shrinkage",
        ),
        (
            "Parameter estimate(s) pinned to an optimizer bound: CL (0.0010 at lower bound).",
            Warning,
            "boundary_estimate",
        ),
        (
            "High relative standard error (RSE > 50%): TVCL (72%).",
            Warning,
            "inflated_rse",
        ),
        (
            "Highly correlated parameter pair(s) (|r| >= 0.95): TVCL ~ TVV (0.98).",
            Warning,
            "high_correlation",
        ),
        (
            "LTBS (log(DV) ~ ...): 3 observation(s) with non-positive DV",
            Warning,
            "data_quality",
        ),
        (
            "W_MISSING_DV: 2 observation row(s) (EVID=0) had a missing DV",
            Warning,
            "data_quality",
        ),
        (
            "Stochastic differential equations ([diffusion] / Extended Kalman \
                 Filter) are an EXPERIMENTAL feature: validated only on a small set",
            Warning,
            "experimental",
        ),
        (
            "Neural-network model components ([covariate_nn] / deep compartment \
                 models) are an EXPERIMENTAL feature: validated only on a small set",
            Warning,
            "experimental",
        ),
        (
            "block omega: ETA_CL x ETA_V have mixed lognormal / additive parameterisations",
            Warning,
            "omega_structure",
        ),
        ("mu-ref: CL, V, KA", Info, "mu_referencing"),
        (
            "SAEM: individual parameter(s) not mu-referenced: V. This can strongly \
                 affect convergence; prefer forms such as `CL = TVCL * exp(ETA_CL)` when possible.",
            Warning,
            "mu_referencing",
        ),
        (
            "Multi-start: best result from start 3/8",
            Info,
            "multi_start",
        ),
        (
            "12 threads configured but only 10 subject(s)",
            Info,
            "threads",
        ),
        ("SAEM with more threads than subjects/2", Info, "threads"),
        (
            "Covariance step: 35 parameters \u{2192} n\u{00b2} OFV evaluations",
            Info,
            "covariance_step",
        ),
        (
            "Covariance step: 35 parameters -> n^2 OFV evaluations",
            Info,
            "covariance_step",
        ),
        // format_non_pd_warning path (NonPdHessian variant).
        (
            "Covariance step: Hessian is not positive definite. \
                 Eigenvalues: [8.4000, 2.1000, -0.0100]. SE estimates not available.",
            Critical,
            "covariance_failed",
        ),
        // Covariance step regularised — present in all three severity tiers.
        (
            "Covariance step regularized: eigenvalue floor applied to FD Hessian \
                 (1 of 3 free-block eigenvalues clipped; min eig = 1.2e-6, floor = 8.4e-14; \
                 severity: minor). Standard errors are likely reliable.",
            Warning,
            "covariance_regularized",
        ),
        (
            "Covariance step regularized: eigenvalue floor applied to FD Hessian \
                 (3 of 5 free-block eigenvalues clipped; min eig = 1.2e-6, floor = 8.4e-14; \
                 severity: severe). Standard errors are likely unreliable; \
                 SIR-based confidence intervals are recommended.",
            Warning,
            "covariance_regularized",
        ),
        // Off-diagonal NaN soft warning — Success result, but correlation missing.
        (
            "Covariance step: off-diagonal FD stencil(s) non-finite for theta[CL], sigma[1]. \
                 Cross-partial correlation set to 0; SE for these parameter(s) \
                 may be over-optimistic. Try tuning fd_hessian_step.",
            Warning,
            "covariance_regularized",
        ),
        // Chain-prefixed off-diagonal warning.
        (
            "[FOCEI] Covariance step: off-diagonal FD stencil(s) non-finite for theta[CL]. \
                 Cross-partial correlation set to 0; SE for these parameter(s) \
                 may be over-optimistic. Try tuning fd_hessian_step.",
            Warning,
            "covariance_regularized",
        ),
        // Unusable messages introduced in commit 2.
        (
            "Covariance step failed: base OFV is non-finite at convergence \
                 (likely numerical overflow or underflow in model evaluation). \
                 SE estimates not available.",
            Critical,
            "covariance_failed",
        ),
        (
            "Covariance step failed: Hessian has ill-conditioned entries for the \
                 following parameter(s) — theta[CL] (non-finite diagonal); \
                 sigma[1] (non-finite off-diagonal). SE estimates not available.",
            Critical,
            "covariance_failed",
        ),
        // Omega near-singular (tiny positive eigenvalue) — "near-singular" descriptor.
        (
            "Covariance step failed: Omega matrix is near-singular at \
                 convergence (min eigenvalue = 1.2e-10; eigenvalues: [0.5000, 1.2e-10]). \
                 SE estimates not available.",
            Critical,
            "covariance_failed",
        ),
        // Omega truly non-PD (negative eigenvalue) — "not positive definite" descriptor.
        (
            "Covariance step failed: Omega matrix is not positive definite at \
                 convergence (min eigenvalue = -1.0e-3; eigenvalues: [0.5000, -1.0e-3]). \
                 SE estimates not available.",
            Critical,
            "covariance_failed",
        ),
        // SIR message that also contains "not positive definite" — must
        // still route to "sir", NOT to covariance_step.
        (
            "SIR failed: covariance not positive definite",
            Warning,
            "sir",
        ),
        // -- chain-prefixed (multi-stage) -----------------------------
        (
            "[FOCEI] Covariance step failed",
            Critical,
            "covariance_failed",
        ),
        (
            "[FOCEI] Covariance step: Hessian is not positive definite. \
                 Eigenvalues: [2.1000, -0.0100]. SE estimates not available.",
            Critical,
            "covariance_failed",
        ),
        (
            "[SAEM] individual parameter(s) not mu-referenced: V",
            Warning,
            "mu_referencing",
        ),
        (
            "[FOCEI] Outer optimization did not converge",
            Critical,
            "convergence",
        ),
        // -- survival/mod.rs simulation diagnostics (#762/#763) --------
        // The `w_tte_degenerate` / `w_rtte_degenerate` prefixes must route to
        // `simulation`, winning over the generic "degenerate" -> optimizer_health
        // branch that their message body would otherwise match.
        (
            "W_TTE_DEGENERATE_HAZARD: subject '7' (CMT=1) drew a non-positive / \
                 non-finite effective hazard rate; no event was generated and the \
                 subject is censored at the observation window (#763). Check the \
                 hazard parameters / covariate values.",
            Warning,
            "simulation",
        ),
        (
            "W_RTTE_DEGENERATE: subject '7' (CMT=2) has a degenerate recurrent \
                 hazard (~1.234e7 events expected before horizon 30); its event stream \
                 was skipped and the subject censored at the horizon (#762). Check the \
                 hazard parameters / covariate values.",
            Warning,
            "simulation",
        ),
        (
            "W_RTTE_DEGENERATE: subject '7' (CMT=2) produced over 1000000 events \
                 before horizon 30 (vanishing inter-event gaps); its stream was \
                 truncated and the subject censored at the horizon (#762).",
            Warning,
            "simulation",
        ),
    ];

    let mut failures: Vec<String> = Vec::new();
    for (msg, want_sev, want_cat) in table {
        let got = classify_warning(msg);
        if got.severity != *want_sev || got.category.as_str() != *want_cat {
            failures.push(format!(
                "  {msg:?} -> expected ({:?}, {:?}), got ({:?}, {:?})",
                want_sev, want_cat, got.severity, got.category
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "classify_warning round-trip failures:\n{}",
        failures.join("\n")
    );
}

#[test]
fn is_ode_based_false_for_analytical() {
    let m = test_helpers::analytical_model(GradientMethod::Auto);
    assert!(!m.is_ode_based());
}

#[test]
fn is_ode_based_true_for_ode() {
    let m = test_helpers::ode_model(GradientMethod::Auto);
    assert!(m.is_ode_based());
}

#[test]
fn has_bioavailability_detects_f_on_either_engine() {
    // Baseline test models declare no F → false on both engines.
    assert!(!test_helpers::analytical_model(GradientMethod::Auto).has_bioavailability());
    assert!(!test_helpers::ode_model(GradientMethod::Auto).has_bioavailability());

    // Analytical route: `f=` on `[structural_model]` puts PK_IDX_F in pk_indices.
    let mut m = test_helpers::analytical_model(GradientMethod::Auto);
    m.pk_indices = vec![PK_IDX_F];
    assert!(m.has_bioavailability());

    // Either engine: a bare `F` (any case) in `[individual_parameters]`.
    let mut m = test_helpers::analytical_model(GradientMethod::Auto);
    m.indiv_param_names = vec!["CL".into(), "f".into()];
    assert!(m.has_bioavailability());

    // ODE engine only: a compartment-indexed `Fn` routes via the DoseAttrMap.
    let mut m = test_helpers::ode_model(GradientMethod::Auto);
    m.indiv_param_names = vec!["CL".into(), "F1".into()];
    assert!(m.has_bioavailability());

    // The same `F1` on the analytical engine is not bioavailability (no ode_spec).
    let mut m = test_helpers::analytical_model(GradientMethod::Auto);
    m.indiv_param_names = vec!["CL".into(), "F1".into()];
    assert!(!m.has_bioavailability());
}

#[test]
fn error_spec_single_ignores_cmt() {
    let spec = ErrorSpec::Single(ErrorModel::Proportional);
    // Proportional: V = (f * sigma)^2 = (10 * 0.1)^2 = 1.0, regardless of CMT.
    assert!((spec.variance_at(1, 10.0, &[0.1]) - 1.0).abs() < 1e-12);
    assert!((spec.variance_at(7, 10.0, &[0.1]) - 1.0).abs() < 1e-12);
}

#[test]
fn error_spec_per_cmt_dispatches_model_and_sigma_slice() {
    // Flat sigma vector [prop_pk, add_pd]; CMT=2 uses idx 0 (proportional),
    // CMT=3 uses idx 1 (additive).
    let mut map = HashMap::new();
    map.insert(
        2,
        EndpointError {
            error_model: ErrorModel::Proportional,
            sigma_idx: vec![0],
        },
    );
    map.insert(
        3,
        EndpointError {
            error_model: ErrorModel::Additive,
            sigma_idx: vec![1],
        },
    );
    let spec = ErrorSpec::PerCmt(map);
    let sigma = [0.1, 2.0];

    // CMT=2 proportional: (10 * 0.1)^2 = 1.0 (independent of additive sigma).
    assert!((spec.variance_at(2, 10.0, &sigma) - 1.0).abs() < 1e-12);
    // CMT=3 additive: 2.0^2 = 4.0 (independent of prediction).
    assert!((spec.variance_at(3, 10.0, &sigma) - 4.0).abs() < 1e-12);
    assert!((spec.variance_at(3, 999.0, &sigma) - 4.0).abs() < 1e-12);

    // Unregistered CMT → NaN guard.
    assert!(spec.variance_at(1, 10.0, &sigma).is_nan());
}

#[test]
fn error_spec_single_sigma_types_padded_to_n_sigma() {
    // Single proportional with extra declared sigmas: leading slot is
    // proportional, the rest default to additive, length == n_sigma.
    let spec = ErrorSpec::Single(ErrorModel::Proportional);
    assert_eq!(
        spec.sigma_types(3),
        vec![
            SigmaType::Proportional,
            SigmaType::Additive,
            SigmaType::Additive
        ]
    );
    // Combined stamps both leading slots.
    let combined = ErrorSpec::Single(ErrorModel::Combined);
    assert_eq!(
        combined.sigma_types(2),
        vec![SigmaType::Proportional, SigmaType::Additive]
    );
}

#[test]
fn variance_at_scaled_unit_mult_matches_legacy() {
    // A unit multiplier must reproduce the plain variance bit-for-bit.
    let combined = ErrorSpec::Single(ErrorModel::Combined);
    let sigma = [0.3, 2.0];
    let f = 5.0;
    let base = combined.variance_at_with_correlations(1, f, &sigma, &[]);
    let scaled = combined.variance_at_scaled(1, f, &sigma, &[], &[1.0, 1.0]);
    assert_eq!(base, scaled);
}

#[test]
fn variance_at_scaled_scales_proportional_loading() {
    // mult = [2, 1] on combined → (2·f·σ_p)² + σ_a².
    let combined = ErrorSpec::Single(ErrorModel::Combined);
    let sigma = [0.3, 2.0];
    let f = 5.0;
    let v = combined.variance_at_scaled(1, f, &sigma, &[], &[2.0, 1.0]);
    let expected = (2.0 * f * 0.3).powi(2) + 2.0_f64.powi(2);
    assert!((v - expected).abs() < 1e-12, "got {v}, want {expected}");
}

#[test]
fn dvar_df_scaled_scales_by_mprop_squared() {
    // dvar_df scales by the proportional multiplier squared.
    let combined = ErrorSpec::Single(ErrorModel::Combined);
    let sigma = [0.3, 2.0];
    let f = 7.0;
    let base = combined.dvar_df(1, f, &sigma);
    let scaled = combined.dvar_df_scaled(1, f, &sigma, &[3.0, 1.0]);
    assert!((scaled - base * 9.0).abs() < 1e-12);
}

#[test]
fn d2var_df2_scaled_scales_by_mprop_squared() {
    // d2var_df2 takes the same m² proportional-loading factor as dvar_df_scaled,
    // so the M3 censored curvature is built from a consistent v(f). A unit mult
    // reproduces the unscaled value; additive stays 0 at any mult.
    let combined = ErrorSpec::Single(ErrorModel::Combined);
    let sigma = [0.3, 2.0];
    // A prediction well away from 0: the combined variance `(f·σ₁)² + σ₂²`
    // stays far above the `MIN_VARIANCE` floor, so `d2var_df2` returns the raw
    // `2·σ₁²` and the m² scaling below is unaffected by the floor gate (#958).
    let f = 5.0;
    let base = combined.d2var_df2(1, f, &sigma);
    assert!((combined.d2var_df2_scaled(1, f, &sigma, &[3.0, 1.0]) - base * 9.0).abs() < 1e-12);
    assert!((combined.d2var_df2_scaled(1, f, &sigma, &[1.0, 1.0]) - base).abs() < 1e-12);
    assert_eq!(
        ErrorSpec::Single(ErrorModel::Additive).d2var_df2_scaled(1, f, &[0.5], &[4.0]),
        0.0
    );
}

#[test]
fn error_spec_per_cmt_variance_at_out_of_range_idx_is_nan() {
    // Hand-constructed endpoint whose sigma_idx points past the sigma slice
    // must yield NaN, not panic.
    let mut map = HashMap::new();
    map.insert(
        2,
        EndpointError {
            error_model: ErrorModel::Proportional,
            sigma_idx: vec![5], // out of range for a length-1 sigma vector
        },
    );
    let spec = ErrorSpec::PerCmt(map);
    assert!(spec.variance_at(2, 10.0, &[0.1]).is_nan());
}

#[test]
fn error_spec_dvar_df_matches_legacy_single_formulas() {
    let f = 7.0;
    // Additive: 0. Proportional/Combined: 2*f*sigma_prop^2.
    assert_eq!(
        ErrorSpec::Single(ErrorModel::Additive).dvar_df(1, f, &[0.5]),
        0.0
    );
    assert!(
        (ErrorSpec::Single(ErrorModel::Proportional).dvar_df(1, f, &[0.3]) - 2.0 * f * 0.09).abs()
            < 1e-12
    );
    // Combined uses the first (proportional) sigma.
    assert!(
        (ErrorSpec::Single(ErrorModel::Combined).dvar_df(1, f, &[0.3, 2.0]) - 2.0 * f * 0.09).abs()
            < 1e-12
    );
}

#[test]
fn error_spec_per_cmt_dvar_df_and_dlogsigma_dispatch_by_endpoint() {
    // CMT=2 proportional on sigma idx 0; CMT=3 additive on sigma idx 1.
    let mut map = HashMap::new();
    map.insert(
        2,
        EndpointError {
            error_model: ErrorModel::Proportional,
            sigma_idx: vec![0],
        },
    );
    map.insert(
        3,
        EndpointError {
            error_model: ErrorModel::Additive,
            sigma_idx: vec![1],
        },
    );
    let spec = ErrorSpec::PerCmt(map);
    let sigma = [0.3, 2.0];
    let f = 7.0;

    // dvar_df: proportional endpoint -> 2*f*0.3^2; additive -> 0.
    assert!((spec.dvar_df(2, f, &sigma) - 2.0 * f * 0.09).abs() < 1e-12);
    assert_eq!(spec.dvar_df(3, f, &sigma), 0.0);

    // dvar_dlogsigma: each sigma only contributes through its own endpoint.
    // sigma 0 (proportional, CMT=2): 2*0.3^2*f^2 at CMT=2, 0 at CMT=3.
    assert!((spec.dvar_dlogsigma(2, 0, f, &sigma) - 2.0 * 0.09 * f * f).abs() < 1e-12);
    assert_eq!(spec.dvar_dlogsigma(3, 0, f, &sigma), 0.0);
    // sigma 1 (additive, CMT=3): 2*2.0^2 at CMT=3, 0 at CMT=2.
    assert!((spec.dvar_dlogsigma(3, 1, f, &sigma) - 2.0 * 4.0).abs() < 1e-12);
    assert_eq!(spec.dvar_dlogsigma(2, 1, f, &sigma), 0.0);
}

#[test]
fn error_spec_per_cmt_sigma_types_maps_each_slot() {
    let mut map = HashMap::new();
    map.insert(
        2,
        EndpointError {
            error_model: ErrorModel::Proportional,
            sigma_idx: vec![0],
        },
    );
    map.insert(
        3,
        EndpointError {
            error_model: ErrorModel::Additive,
            sigma_idx: vec![1],
        },
    );
    let spec = ErrorSpec::PerCmt(map);
    assert_eq!(
        spec.sigma_types(2),
        vec![SigmaType::Proportional, SigmaType::Additive]
    );
}

#[test]
fn test_lagtime_name_to_index_and_default() {
    assert_eq!(PkParams::name_to_index("lagtime"), Some(PK_IDX_LAGTIME));
    // NONMEM-style alias maps to the same slot.
    assert_eq!(PkParams::name_to_index("alag"), Some(PK_IDX_LAGTIME));
    assert_eq!(PK_IDX_LAGTIME, 8);
    // Slots 0–8 are named PK params; the rest are ODE structural-param
    // headroom (see MAX_PK_PARAMS docs / issue #122).
    assert!(MAX_PK_PARAMS > PK_IDX_LAGTIME);

    let default = PkParams::default();
    assert_eq!(default.lagtime(), 0.0);
    // F still defaults to 1.0 (unchanged).
    assert_eq!(default.f_bio(), 1.0);
}

#[test]
fn dose_attr_map_resolves_indexed_then_bare_then_default() {
    // params: slot 5 = bare F, slot 8 = bare lag, slots 9/10 = F2/ALAG2.
    let mut params = [0.0f64; MAX_PK_PARAMS];
    params[PK_IDX_F] = 0.8; // bare F
    params[PK_IDX_LAGTIME] = 0.5; // bare lag
    params[9] = 0.3; // F for compartment 2
    params[10] = 1.25; // ALAG for compartment 2

    let mut map = DoseAttrMap::default();
    map.insert(DoseAttr::F, 2, 9);
    map.insert(DoseAttr::Lag, 2, 10);

    // Compartment 1 has no indexed entry -> falls through to the bare slot.
    assert_eq!(map.f_bio(1, &params), 0.8);
    assert_eq!(map.lagtime(1, &params), 0.5);
    // Compartment 2 is overridden by its indexed slot.
    assert_eq!(map.f_bio(2, &params), 0.3);
    assert_eq!(map.lagtime(2, &params), 1.25);
    // indexed_slot exposes the raw mapping (used by upstream validation).
    assert_eq!(map.indexed_slot(DoseAttr::F, 2), Some(9));
    assert_eq!(map.indexed_slot(DoseAttr::F, 1), None);
}

/// `CMT=0` is NONMEM's *default dose compartment* — compartment 1 — so it must
/// resolve to the `F1` / `ALAG1` entry, not miss the indexed map (#899).
///
/// The silent failure this pins: a model declaring `F1 = 0.6` and no bare `F`
/// leaves `PK_IDX_F` at `PkParams::default()`'s **1.0**, so a dose written
/// `CMT=0` would be delivered at full amount — a 40% dose error with no error
/// and no warning. It was the last site where `CMT=0` still meant something
/// other than compartment 1 after the rest of #899 unified the dose routing.
#[test]
fn cmt_zero_resolves_dose_attributes_as_the_default_compartment() {
    let mut params = [0.0f64; MAX_PK_PARAMS];
    // No bare F declared: the slot keeps the 1.0 default, which is exactly what
    // a missed lookup would silently return.
    params[PK_IDX_F] = 1.0;
    params[PK_IDX_LAGTIME] = 0.0;
    params[9] = 0.6; // F1
    params[10] = 0.75; // ALAG1

    let mut map = DoseAttrMap::default();
    map.insert(DoseAttr::F, 1, 9);
    map.insert(DoseAttr::Lag, 1, 10);

    // The bug: without the normalisation these returned the bare 1.0 / 0.0.
    assert_eq!(map.f_bio(0, &params), 0.6, "CMT=0 must read F1, not bare F");
    assert_eq!(
        map.lagtime(0, &params),
        0.75,
        "CMT=0 must read ALAG1, not bare ALAG"
    );
    // …and identically to the same dose written CMT=1.
    assert_eq!(map.f_bio(0, &params), map.f_bio(1, &params));
    assert_eq!(map.lagtime(0, &params), map.lagtime(1, &params));

    // The slot accessors agree, so `lag_slot` (which the sens walk differentiates
    // through) picks the same parameter — otherwise the gradient would be taken
    // with respect to the wrong slot.
    assert_eq!(
        map.indexed_slot(DoseAttr::F, 0),
        map.indexed_slot(DoseAttr::F, 1)
    );
    assert_eq!(map.lag_slot(0), map.lag_slot(1));

    // Not over-broad: a model with no compartment-1 entry still falls through to
    // the bare slot for CMT=0, exactly as CMT=1 does.
    let mut only_cmt2 = DoseAttrMap::default();
    only_cmt2.insert(DoseAttr::F, 2, 9);
    assert_eq!(only_cmt2.f_bio(0, &params), 1.0);
    assert_eq!(only_cmt2.indexed_slot(DoseAttr::F, 0), None);
}

/// The canonical CMT-conversion accessors (#899, #912): `CMT=0` is the default dose
/// compartment == compartment 1, so `cmt_idx()` (0-based state index) maps both 0 and 1 to
/// 0, and `cmt_1based()` maps both to 1. Every dose-compartment comparison in the engines
/// routes through these instead of open-coding `cmt - 1` (underflows on 0) or `cmt >= 1`
/// (drops 0) — the scattering that produced the #913-review gaps.
#[test]
fn dose_cmt_accessors_resolve_cmt_zero_to_the_default_compartment() {
    let d0 = DoseEvent::new(0.0, 100.0, 0, 0.0, false, 0.0);
    let d1 = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
    let d2 = DoseEvent::new(0.0, 100.0, 2, 0.0, false, 0.0);

    // CMT=0 and CMT=1 are the same compartment on both bases.
    assert_eq!(d0.cmt_idx(), 0);
    assert_eq!(d1.cmt_idx(), 0);
    assert_eq!(d0.cmt_1based(), 1);
    assert_eq!(d1.cmt_1based(), 1);

    // A genuine second compartment is unaffected (no clamping past 1).
    assert_eq!(d2.cmt_idx(), 1);
    assert_eq!(d2.cmt_1based(), 2);

    // `cmt_raw()` is the literal authored value, `0` included — the escape hatch
    // that lets a site distinguish an authored `CMT=0` from `CMT=1` (the resolved
    // accessors cannot).
    assert_eq!(d0.cmt_raw(), 0);
    assert_eq!(d1.cmt_raw(), 1);
    assert_eq!(d2.cmt_raw(), 2);

    // The two resolved accessors are exact complements, and both are pure
    // functions of the raw value (#912: the field is private, these are the only
    // ways to read it).
    for d in [&d0, &d1, &d2] {
        assert_eq!(d.cmt_idx() + 1, d.cmt_1based());
        assert_eq!(d.cmt_idx(), d.cmt_raw().saturating_sub(1));
        assert_eq!(d.cmt_1based(), d.cmt_raw().max(1));
    }
}

/// `synthetic_cycle_dose()` (#912/#419) repositions a dose to a single SS cycle's
/// origin — `time = 0`, `ss`/`ii` cleared — while preserving the compartment and
/// the bioavailability-reshaped `(rate, duration)` that a bare `DoseEvent::new`
/// would recompute and drop. It is the replacement for the `..dose.clone()`
/// struct-update the now-private `cmt` field forbids outside this module.
#[test]
fn synthetic_cycle_dose_preserves_shape_and_clears_steady_state() {
    // An SS infusion into compartment 2, then F-reshaped so (rate, duration) no
    // longer equal the raw amt/rate pair `::new` would derive.
    let dose = DoseEvent::new(5.0, 100.0, 2, 25.0, true, 12.0).with_bioavailable_infusion(0.5);
    let cyc = dose.synthetic_cycle_dose();

    // Repositioned to the cycle origin.
    assert_eq!(cyc.time, 0.0);
    assert!(!cyc.ss);
    assert_eq!(cyc.ii, 0.0);

    // Compartment and the reshaped infusion shape survive verbatim.
    assert_eq!(cyc.cmt_raw(), dose.cmt_raw());
    assert_eq!(cyc.amt, dose.amt);
    assert_eq!(cyc.rate, dose.rate);
    assert_eq!(cyc.duration, dose.duration);
    assert_eq!(cyc.is_infusion(), dose.is_infusion());
    assert_eq!(cyc.rate_mode, dose.rate_mode);
    assert_eq!(cyc.infusion_def, dose.infusion_def);

    // `rate_mode` / `infusion_def` are carried through directly, not inferred from
    // `is_infusion()` — assert on a *modeled* dose so the check is non-vacuous (the
    // reshaped `Fixed` case above would pass even if they were hard-coded to
    // `Fixed`/`RateDefined`, which is exactly what `DoseEvent::new` would do).
    let modeled = DoseEvent::modeled(5.0, 100.0, 2, true, 12.0, RateMode::ModeledDuration);
    let mcyc = modeled.synthetic_cycle_dose();
    assert_eq!(mcyc.rate_mode, RateMode::ModeledDuration);
    assert_eq!(mcyc.infusion_def, modeled.infusion_def);
}

#[test]
fn dose_attr_from_indexed_name_recognizes_f_lag_duration_and_rate() {
    use DoseAttr::*;
    // Bioavailability and both lag spellings, case-insensitive.
    assert_eq!(DoseAttr::from_indexed_name("F1"), Some((F, 1)));
    assert_eq!(DoseAttr::from_indexed_name("f2"), Some((F, 2)));
    assert_eq!(DoseAttr::from_indexed_name("ALAG1"), Some((Lag, 1)));
    assert_eq!(DoseAttr::from_indexed_name("alag3"), Some((Lag, 3)));
    assert_eq!(DoseAttr::from_indexed_name("LAGTIME2"), Some((Lag, 2)));
    // Modeled infusion duration D{n} (RATE=-2; #324), case-insensitive.
    assert_eq!(DoseAttr::from_indexed_name("D1"), Some((Duration, 1)));
    assert_eq!(DoseAttr::from_indexed_name("d2"), Some((Duration, 2)));
    // Modeled infusion rate R{n} (RATE=-1; #324), case-insensitive.
    assert_eq!(DoseAttr::from_indexed_name("R1"), Some((Rate, 1)));
    assert_eq!(DoseAttr::from_indexed_name("r2"), Some((Rate, 2)));

    // Bare forms and zero index are not compartment-indexed attributes.
    assert_eq!(DoseAttr::from_indexed_name("F"), None);
    assert_eq!(DoseAttr::from_indexed_name("lagtime"), None);
    assert_eq!(DoseAttr::from_indexed_name("alag"), None);
    assert_eq!(DoseAttr::from_indexed_name("F0"), None);
    assert_eq!(DoseAttr::from_indexed_name("D0"), None);
    assert_eq!(DoseAttr::from_indexed_name("D"), None);
    assert_eq!(DoseAttr::from_indexed_name("R0"), None);
    assert_eq!(DoseAttr::from_indexed_name("R"), None);

    // Must not capture canonical PK names, the [scaling] `S{n}` names, or
    // non-numeric suffixes — including `D`/`R`-prefixed words (`delta`,
    // `rate`, `RUV`, `R2D2`).
    for n in [
        "CL", "V1", "V2", "Q2", "Q3", "KA", "S1", "S2", "f_bio", "rate", "delta", "decay", "RUV",
        "R2D2",
    ] {
        assert_eq!(DoseAttr::from_indexed_name(n), None, "{n} must not match");
    }

    // An all-digit suffix that overflows usize fails to parse -> not an
    // attribute (exercises the parse-error guard, not just the success path).
    assert_eq!(
        DoseAttr::from_indexed_name(&format!("R{}", "9".repeat(40))),
        None
    );
}

#[test]
fn dose_attr_map_empty_yields_engine_defaults() {
    // An empty map (the common bare-/single-route model) must reproduce the
    // pre-existing defaults: F = 1.0, lag = 0.0, even for a params slice too
    // short to hold the reserved slots (cannot panic).
    let map = DoseAttrMap::default();
    let full = PkParams::default().values; // F slot already 1.0
    assert_eq!(map.f_bio(1, &full), 1.0);
    assert_eq!(map.lagtime(1, &full), 0.0);

    let short: [f64; 3] = [2.0, 3.0, 4.0];
    assert_eq!(map.f_bio(1, &short), 1.0);
    assert_eq!(map.lagtime(1, &short), 0.0);
    // Duration/Rate have no bare fallback -> no indexed slot means None.
    assert_eq!(map.indexed_slot(DoseAttr::Duration, 1), None);
    assert_eq!(map.indexed_slot(DoseAttr::Rate, 1), None);
}

#[test]
fn dose_event_resolve_rate_modeled_duration_matches_explicit_infusion() {
    // RATE=-2 with D{1} in slot 9: a 100-unit dose over D = 5 must resolve to
    // the same (rate, duration) as an explicit RATE = 100/5 = 20 infusion.
    let mut map = DoseAttrMap::default();
    map.insert(DoseAttr::Duration, 1, 9);
    let mut params = [0.0; MAX_PK_PARAMS];
    params[9] = 5.0; // D1

    let modeled = DoseEvent::modeled(0.0, 100.0, 1, false, 0.0, RateMode::ModeledDuration);
    assert!(
        modeled.is_infusion(),
        "modeled dose is an infusion pre-resolve"
    );
    let resolved = modeled.resolve_rate(&map, &params);

    assert_eq!(resolved.rate_mode, RateMode::Fixed);
    assert_eq!(resolved.duration, 5.0);
    assert_eq!(resolved.rate, 20.0);
    // Bit-equal to the hand-written explicit infusion (the #324 invariant).
    let explicit = DoseEvent::new(0.0, 100.0, 1, 20.0, false, 0.0);
    assert_eq!(resolved.rate, explicit.rate);
    assert_eq!(resolved.duration, explicit.duration);
    // cmt/time/amt/ss/ii are preserved through resolution.
    assert_eq!(resolved.cmt, 1);
    assert_eq!(resolved.amt, 100.0);

    // A Fixed dose is returned unchanged (the common, allocation-cheap path).
    let same = explicit.resolve_rate(&map, &params);
    assert_eq!(
        (same.rate, same.duration, same.rate_mode),
        (20.0, 5.0, RateMode::Fixed)
    );
}

#[test]
fn dose_event_resolve_rate_clamps_nonpositive_duration() {
    // A transient D <= 0 (or NaN) mid-search clamps to DURATION_FLOOR so
    // rate = amt / D stays finite (mirrors PreparedInputRate::MIN_PARAM).
    let mut map = DoseAttrMap::default();
    map.insert(DoseAttr::Duration, 1, 9);
    let modeled = DoseEvent::modeled(0.0, 100.0, 1, false, 0.0, RateMode::ModeledDuration);

    for bad in [0.0, -3.0, f64::NAN] {
        let mut params = [0.0; MAX_PK_PARAMS];
        params[9] = bad;
        let r = modeled.resolve_rate(&map, &params);
        assert_eq!(r.duration, DoseEvent::DURATION_FLOOR, "clamp at floor");
        assert!(r.rate.is_finite() && r.rate > 0.0, "rate finite");
    }
}

#[test]
fn dose_event_resolve_rate_modeled_rate_matches_explicit_infusion() {
    // RATE=-1 with R{1} in slot 9: a 100-unit dose at R = 20 must resolve to
    // the same (rate, duration) as an explicit RATE = 20 infusion (the
    // #324 invariant, mirror of the modeled-duration case).
    let mut map = DoseAttrMap::default();
    map.insert(DoseAttr::Rate, 1, 9);
    let mut params = [0.0; MAX_PK_PARAMS];
    params[9] = 20.0; // R1

    let modeled = DoseEvent::modeled(0.0, 100.0, 1, false, 0.0, RateMode::ModeledRate);
    assert!(
        modeled.is_infusion(),
        "modeled-rate dose is an infusion pre-resolve"
    );
    let resolved = modeled.resolve_rate(&map, &params);

    assert_eq!(resolved.rate_mode, RateMode::Fixed);
    assert_eq!(resolved.rate, 20.0);
    // amt / rate = 100 / 20; bit-equal to the hand-written explicit infusion.
    assert_eq!(resolved.duration, 5.0);
    let explicit = DoseEvent::new(0.0, 100.0, 1, 20.0, false, 0.0);
    assert_eq!(resolved.rate, explicit.rate);
    assert_eq!(resolved.duration, explicit.duration);
    assert_eq!(resolved.cmt, 1);
    assert_eq!(resolved.amt, 100.0);
}

#[test]
fn dose_event_resolve_rate_clamps_nonpositive_rate() {
    // A transient R <= 0 (or NaN) mid-search clamps to RATE_FLOOR so the
    // implied duration = amt / R stays finite (mirror of the duration clamp).
    let mut map = DoseAttrMap::default();
    map.insert(DoseAttr::Rate, 1, 9);
    let modeled = DoseEvent::modeled(0.0, 100.0, 1, false, 0.0, RateMode::ModeledRate);

    for bad in [0.0, -3.0, f64::NAN] {
        let mut params = [0.0; MAX_PK_PARAMS];
        params[9] = bad;
        let r = modeled.resolve_rate(&map, &params);
        assert_eq!(r.rate, DoseEvent::RATE_FLOOR, "clamp at floor");
        assert!(
            r.duration.is_finite() && r.duration > 0.0,
            "duration finite"
        );
    }
}

#[test]
fn clamp_above_floor_passes_through_or_clamps() {
    // The shared clamp behind DURATION_FLOOR / MIN_PARAM: > floor passes through,
    // <= floor (and NaN — every `>` is false for NaN) returns the floor.
    let floor = 1e-8;
    assert_eq!(clamp_above_floor(5.0, floor), 5.0, "above floor passes");
    assert_eq!(
        clamp_above_floor(2e-8, floor),
        2e-8,
        "just above floor passes"
    );
    assert_eq!(
        clamp_above_floor(floor, floor),
        floor,
        "at floor → floor (not >)"
    );
    assert_eq!(clamp_above_floor(0.0, floor), floor, "zero clamps to floor");
    assert_eq!(
        clamp_above_floor(-3.0, floor),
        floor,
        "negative clamps to floor"
    );
    assert_eq!(
        clamp_above_floor(f64::NAN, floor),
        floor,
        "NaN clamps to floor"
    );
}

#[test]
fn test_lagtime_from_hashmap_primary_and_alias() {
    let mut m = HashMap::new();
    m.insert("lagtime".to_string(), 1.5);
    let p = PkParams::from_hashmap(&m);
    assert_eq!(p.lagtime(), 1.5);

    let mut m_alias = HashMap::new();
    m_alias.insert("alag".to_string(), 2.0);
    let p_alias = PkParams::from_hashmap(&m_alias);
    assert_eq!(p_alias.lagtime(), 2.0);
}

/// Guard the SAEM MH-step default. An early value (3) was too low for hard
/// cold-start surfaces — the chain didn't decorrelate between SAEM outer
/// iterations, so the single-draw stochastic M-step received sticky
/// correlated ETAs and locked the population-θ M-step into a degenerate
/// basin (observed on Emax PKPD: PD-curve thetas pinned to boundary, ~150
/// OFV units worse than the correct basin). The default was raised to 10,
/// then to 20 alongside the componentwise eta kernel and the damped Ω
/// stochastic-approximation step — both added to stop a block (correlated)
/// Ω collapsing to a near rank-1 correlation matrix (UVM 2-cpt: every
/// off-diagonal correlation → ~0.99, one variance → 0). The larger default
/// also sizes the componentwise sweep count (`max(2, n_mh_steps / n_eta)`).
///
/// If a future change drops the default below ~5, re-run both the Emax PKPD
/// basin regression and the UVM block-Ω collapse regression in the
/// experiment repo before merging — both fail silently (OFV looks fine;
/// parameters wrong).
#[test]
fn saem_n_mh_steps_default_is_20() {
    let opts = FitOptions::default();
    assert_eq!(
        opts.saem_n_mh_steps, 20,
        "saem_n_mh_steps default changed — see comment above this test \
             for the basin-trap and block-Ω-collapse regression rationale \
             before adjusting."
    );
}

#[test]
fn impmap_proposal_df_default_is_finite_t() {
    // IMPMAP defaults to a Student-t proposal (df = 4), not a Gaussian: the
    // MVN tails are too light for weakly-identified posteriors and bias the
    // M-step moments (absorption-param drift, #411). Do not revert to ∞
    // without re-checking that regression.
    let opts = FitOptions::default();
    assert_eq!(opts.impmap_proposal_df, 4.0);
    assert!(opts.impmap_proposal_df.is_finite());
}

// ── small pure helpers ───────────────────────────────────────────────────

/// Bare subject with no doses/observations; tests mutate the fields they
/// exercise.
fn bare_subject(id: &str) -> Subject {
    Subject {
        id: id.to_string(),
        doses: Vec::new(),
        obs_times: Vec::new(),
        obs_raw_times: Vec::new(),
        observations: Vec::new(),
        obs_cmts: Vec::new(),
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: Vec::new(),
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: Vec::new(),
    }
}

fn dose(ss: bool) -> DoseEvent {
    DoseEvent::new(0.0, 100.0, 1, 0.0, ss, 0.0)
}

#[test]
fn subject_has_censored_observation_reflects_cens_flags() {
    let mut s = bare_subject("1");
    s.cens = vec![0, 0];
    assert!(!s.has_censored_observation());
    s.cens = vec![0, 1];
    assert!(s.has_censored_observation());
    s.cens = vec![-1, 0];
    assert!(s.has_censored_observation());
}

#[test]
fn time_varying_covariate_names_detects_only_varying_covariates() {
    let mut s = bare_subject("1");
    // WT varies across obs snapshots; SEX is constant → only WT is reported.
    s.obs_covariates = vec![
        HashMap::from([("WT".to_string(), 70.0), ("SEX".to_string(), 1.0)]),
        HashMap::from([("WT".to_string(), 80.0), ("SEX".to_string(), 1.0)]),
    ];
    assert_eq!(s.time_varying_covariate_names(), vec!["WT".to_string()]);
}

#[test]
fn time_varying_covariate_names_empty_when_constant_or_absent() {
    // No snapshots at all → empty.
    let s = bare_subject("1");
    assert!(s.time_varying_covariate_names().is_empty());
    // Present but constant across snapshots → empty.
    let mut s2 = bare_subject("2");
    s2.obs_covariates = vec![
        HashMap::from([("WT".to_string(), 70.0)]),
        HashMap::from([("WT".to_string(), 70.0)]),
    ];
    assert!(s2.time_varying_covariate_names().is_empty());
}

#[test]
fn time_varying_covariate_names_inspects_pk_only_rows() {
    // A covariate that changes only across EVID=2 (pk-only) rows is time-varying —
    // this is the case `has_tv_covariates()` misses (it inspects only obs/dose
    // snapshots), so with obs/dose empty it reports no TV covariates at all.
    let mut s = bare_subject("1");
    s.pk_only_covariates = vec![
        HashMap::from([("CRCL".to_string(), 100.0)]),
        HashMap::from([("CRCL".to_string(), 60.0)]),
    ];
    assert!(
        !s.has_tv_covariates(),
        "has_tv_covariates inspects only obs/dose rows, so it is blind to pk_only variation"
    );
    assert_eq!(s.time_varying_covariate_names(), vec!["CRCL".to_string()]);
}

#[test]
fn subject_has_ss_doses_detects_steady_state_flag() {
    let mut s = bare_subject("1");
    s.doses = vec![dose(false), dose(false)];
    assert!(!s.has_ss_doses());
    s.doses.push(dose(true));
    assert!(s.has_ss_doses());
}

#[test]
fn subject_has_periodic_ss_dose_requires_ss_flag_and_positive_ii() {
    let mut s = bare_subject("1");
    // No doses → false.
    assert!(!s.has_periodic_ss_dose());
    // SS flag but `II = 0` (not a periodic SS dose) → false, unlike
    // `has_ss_doses`, which keys on the flag alone.
    s.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 0.0)];
    assert!(s.has_ss_doses());
    assert!(!s.has_periodic_ss_dose());
    // Periodic SS dose (`SS=1`, `II > 0`) → true.
    s.doses.push(DoseEvent::new(0.0, 100.0, 1, 0.0, true, 24.0));
    assert!(s.has_periodic_ss_dose());
    // `II > 0` without the SS flag → false.
    s.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 24.0)];
    assert!(!s.has_periodic_ss_dose());
}

#[test]
fn subject_has_rate_defined_infusion_distinguishes_infusion_modes() {
    let mut s = bare_subject("1");
    // No doses → false.
    assert!(!s.has_rate_defined_infusion());

    // Bolus (RATE=0) is not an infusion → false.
    s.doses = vec![dose(false)];
    assert!(!s.has_rate_defined_infusion());

    // Duration-defined infusion (RATE=-2 → D{cmt}): it *is* an infusion, but F
    // scales its rate, not its window, so it must not count (#419).
    s.doses = vec![DoseEvent::modeled(
        0.0,
        100.0,
        1,
        false,
        0.0,
        RateMode::ModeledDuration,
    )];
    assert!(s.doses[0].is_infusion());
    assert!(!s.has_rate_defined_infusion());

    // Rate-defined infusion (RATE>0 data) → true.
    s.doses = vec![DoseEvent::new(0.0, 100.0, 1, 10.0, false, 0.0)];
    assert!(s.has_rate_defined_infusion());

    // A rate-defined infusion alongside a bolus still trips it.
    s.doses = vec![dose(false), DoseEvent::new(0.0, 100.0, 1, 10.0, false, 0.0)];
    assert!(s.has_rate_defined_infusion());
}

#[test]
fn subject_pk_only_cov_falls_back_to_static_map() {
    let mut s = bare_subject("1");
    s.covariates.insert("WT".to_string(), 70.0);
    // No per-EVID-2 snapshots → static map.
    assert_eq!(s.pk_only_cov(0).get("WT"), Some(&70.0));
    // With a snapshot at index 0 → that snapshot.
    let mut snap = HashMap::new();
    snap.insert("WT".to_string(), 80.0);
    s.pk_only_covariates = vec![snap];
    assert_eq!(s.pk_only_cov(0).get("WT"), Some(&80.0));
    // Out-of-range index → fall back to static map.
    assert_eq!(s.pk_only_cov(5).get("WT"), Some(&70.0));
}

#[test]
fn prune_irrelevant_tv_covariates_clears_only_constant_referenced_covs() {
    // Subject A: the referenced covariate (WT) is constant across snapshots
    // but an unreferenced one (DAY) varies → snapshots should be pruned.
    let mut a = bare_subject("A");
    a.covariates = HashMap::from([("WT".to_string(), 70.0), ("DAY".to_string(), 1.0)]);
    a.obs_covariates = vec![
        HashMap::from([("WT".to_string(), 70.0), ("DAY".to_string(), 1.0)]),
        HashMap::from([("WT".to_string(), 70.0), ("DAY".to_string(), 2.0)]),
    ];
    // Subject B: the referenced covariate (WT) genuinely varies → keep.
    let mut b = bare_subject("B");
    b.covariates = HashMap::from([("WT".to_string(), 70.0)]);
    b.obs_covariates = vec![
        HashMap::from([("WT".to_string(), 70.0)]),
        HashMap::from([("WT".to_string(), 90.0)]),
    ];

    let mut pop = Population {
        subjects: vec![a, b],
        covariate_names: vec!["WT".to_string(), "DAY".to_string()],
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let pruned = pop.prune_irrelevant_tv_covariates(&["WT".to_string()]);
    assert_eq!(pruned, 1, "only subject A should be pruned");
    assert!(
        pop.subjects[0].obs_covariates.is_empty(),
        "A pruned to fast path"
    );
    assert!(
        !pop.subjects[1].obs_covariates.is_empty(),
        "B keeps TV snapshots"
    );
}

#[test]
fn covariate_kind_label_strings() {
    assert_eq!(CovariateKind::Continuous.label(), "continuous");
    assert_eq!(CovariateKind::Categorical.label(), "categorical");
}

#[test]
fn scaling_spec_requires_fd_is_always_false() {
    assert!(!ScalingSpec::None.requires_fd());
    assert!(!ScalingSpec::ScalarScale(1000.0).requires_fd());
    assert!(!ScalingSpec::PerCmt(HashMap::new()).requires_fd());
}

#[test]
fn estimation_method_label_covers_all_variants() {
    assert_eq!(EstimationMethod::Foce.label(), "FOCE");
    assert_eq!(EstimationMethod::FoceI.label(), "FOCEI");
    assert_eq!(EstimationMethod::FoceGn.label(), "FOCE-GN");
    assert_eq!(EstimationMethod::FoceGnHybrid.label(), "FOCE-GN-Hybrid");
    assert_eq!(EstimationMethod::Saem.label(), "SAEM");
    assert_eq!(EstimationMethod::Imp.label(), "IMP");
}

#[test]
fn model_parameters_has_any_fixed_tracks_every_flag_vec() {
    let mut mp = test_helpers::analytical_model(GradientMethod::Fd).default_params;
    mp.theta_fixed = vec![false];
    mp.omega_fixed = vec![false];
    mp.sigma_fixed = vec![false];
    mp.kappa_fixed = Vec::new();
    assert!(!mp.has_any_fixed());
    mp.sigma_fixed = vec![true];
    assert!(mp.has_any_fixed());
}

#[test]
fn pk_params_from_hashmap_maps_named_fields_and_aliases() {
    let map = HashMap::from([
        ("cl".to_string(), 5.0),
        ("v1".to_string(), 30.0), // v1 alias → V index
        ("q".to_string(), 2.0),
        ("v2".to_string(), 50.0),
        ("ka".to_string(), 1.2),
        ("f".to_string(), 0.8),
        ("q3".to_string(), 0.5),
        ("v3".to_string(), 100.0),
    ]);
    let p = PkParams::from_hashmap(&map);
    assert_eq!(p.values[PK_IDX_CL], 5.0);
    assert_eq!(p.values[PK_IDX_V], 30.0);
    assert_eq!(p.values[PK_IDX_Q], 2.0);
    assert_eq!(p.values[PK_IDX_V2], 50.0);
    assert_eq!(p.values[PK_IDX_KA], 1.2);
    assert_eq!(p.values[PK_IDX_F], 0.8);
    assert_eq!(p.values[PK_IDX_Q3], 0.5);
    assert_eq!(p.values[PK_IDX_V3], 100.0);
}

// ── block_sigma / correlated residual error (PR #545) ──────────────────

#[test]
fn sigma_loadings_single_matches_error_model_shape() {
    // Additive loads only the additive sigma (coeff 1); proportional loads
    // the proportional sigma (coeff f); combined loads both, in the
    // (prop, add) order that `residual_variance` assumes positionally.
    let add = ErrorSpec::Single(ErrorModel::Additive);
    assert_eq!(add.sigma_loadings(0, 5.0, 1), vec![(0, 1.0)]);

    let prop = ErrorSpec::Single(ErrorModel::Proportional);
    assert_eq!(prop.sigma_loadings(0, 5.0, 1), vec![(0, 5.0)]);

    let comb = ErrorSpec::Single(ErrorModel::Combined);
    assert_eq!(comb.sigma_loadings(0, 5.0, 2), vec![(0, 5.0), (1, 1.0)]);
}

#[test]
fn sigma_loadings_single_respects_n_sigma_guards() {
    // With no sigmas available the loadings are empty; a 1-sigma combined
    // model exposes only the proportional loading.
    let add = ErrorSpec::Single(ErrorModel::Additive);
    assert!(add.sigma_loadings(0, 5.0, 0).is_empty());
    let prop = ErrorSpec::Single(ErrorModel::Proportional);
    assert!(prop.sigma_loadings(0, 5.0, 0).is_empty());
    let comb = ErrorSpec::Single(ErrorModel::Combined);
    assert_eq!(comb.sigma_loadings(0, 5.0, 1), vec![(0, 5.0)]);
}

#[test]
fn sigma_loadings_per_cmt_dispatches_by_endpoint() {
    // Endpoint in cmt 1 is combined over global sigma indices (2, 3);
    // cmt 2 is additive over index 0. Loadings must use those global
    // indices, and an unmapped compartment yields no loadings.
    let mut map = std::collections::HashMap::new();
    map.insert(
        1usize,
        EndpointError {
            error_model: ErrorModel::Combined,
            sigma_idx: vec![2, 3],
        },
    );
    map.insert(
        2usize,
        EndpointError {
            error_model: ErrorModel::Additive,
            sigma_idx: vec![0],
        },
    );
    let spec = ErrorSpec::PerCmt(map);
    assert_eq!(spec.sigma_loadings(1, 7.0, 4), vec![(2, 7.0), (3, 1.0)]);
    assert_eq!(spec.sigma_loadings(2, 7.0, 4), vec![(0, 1.0)]);
    assert!(spec.sigma_loadings(9, 7.0, 4).is_empty());
}

#[test]
fn sigma_loading_slopes_match_loading_shape_with_f_derivatives() {
    // Slopes have the SAME slot presence as `sigma_loadings` but carry
    // ∂coeff/∂f: 0 for the additive (constant) slot, 1 for the proportional
    // slot. This keeps the dense-R ∂R/∂f cross-covariance assembly consistent.
    let add = ErrorSpec::Single(ErrorModel::Additive);
    assert_eq!(add.sigma_loading_slopes(0, 1), vec![(0, 0.0)]);
    assert!(add.sigma_loading_slopes(0, 0).is_empty());

    let prop = ErrorSpec::Single(ErrorModel::Proportional);
    assert_eq!(prop.sigma_loading_slopes(0, 1), vec![(0, 1.0)]);
    assert!(prop.sigma_loading_slopes(0, 0).is_empty());

    let comb = ErrorSpec::Single(ErrorModel::Combined);
    assert_eq!(comb.sigma_loading_slopes(0, 2), vec![(0, 1.0), (1, 0.0)]);
    assert_eq!(comb.sigma_loading_slopes(0, 1), vec![(0, 1.0)]);

    // PerCmt dispatch over global sigma indices, mirroring sigma_loadings.
    let mut map = std::collections::HashMap::new();
    map.insert(
        1usize,
        EndpointError {
            error_model: ErrorModel::Combined,
            sigma_idx: vec![2, 3],
        },
    );
    map.insert(
        2usize,
        EndpointError {
            error_model: ErrorModel::Proportional,
            sigma_idx: vec![0],
        },
    );
    map.insert(
        3usize,
        EndpointError {
            error_model: ErrorModel::Additive,
            sigma_idx: vec![1],
        },
    );
    let spec = ErrorSpec::PerCmt(map);
    assert_eq!(spec.sigma_loading_slopes(1, 4), vec![(2, 1.0), (3, 0.0)]);
    assert_eq!(spec.sigma_loading_slopes(2, 4), vec![(0, 1.0)]);
    assert_eq!(spec.sigma_loading_slopes(3, 4), vec![(1, 0.0)]);
    assert!(spec.sigma_loading_slopes(9, 4).is_empty());
}

#[test]
fn variance_at_with_correlations_empty_delegates_to_variance_at() {
    // No correlations ⇒ identical to the plain diagonal variance.
    let spec = ErrorSpec::Single(ErrorModel::Combined);
    let sigma = [0.2, 1.0];
    let plain = spec.variance_at(0, 10.0, &sigma);
    let corr = spec.variance_at_with_correlations(0, 10.0, &sigma, &[]);
    assert_relative_eq!(plain, corr, epsilon = 1e-12);
}

#[test]
fn variance_at_with_correlations_combined_adds_cross_term() {
    // V = (10·0.2)² + 1² + 2·10·0.5·0.2·1 = 4 + 1 + 2 = 7.
    let spec = ErrorSpec::Single(ErrorModel::Combined);
    let corr = ResidualCorrelation {
        sigma_i: 0,
        sigma_j: 1,
        rho: 0.5,
    };
    let v = spec.variance_at_with_correlations(1, 10.0, &[0.2, 1.0], &[corr]);
    assert_relative_eq!(v, 7.0, epsilon = 1e-12);
}

#[test]
fn variance_at_with_correlations_per_cmt_combined() {
    // Same combined cross term, but reached through a PerCmt endpoint whose
    // sigmas live at global indices (0, 1).
    let mut map = std::collections::HashMap::new();
    map.insert(
        1usize,
        EndpointError {
            error_model: ErrorModel::Combined,
            sigma_idx: vec![0, 1],
        },
    );
    let spec = ErrorSpec::PerCmt(map);
    let corr = ResidualCorrelation {
        sigma_i: 0,
        sigma_j: 1,
        rho: 0.5,
    };
    let v = spec.variance_at_with_correlations(1, 10.0, &[0.2, 1.0], &[corr]);
    assert_relative_eq!(v, 7.0, epsilon = 1e-12);
}

#[test]
fn variance_at_with_correlations_nan_when_no_loadings() {
    // An observation in an unmapped compartment has no loadings ⇒ NaN.
    let map = std::collections::HashMap::new();
    let spec = ErrorSpec::PerCmt(map);
    let corr = ResidualCorrelation {
        sigma_i: 0,
        sigma_j: 1,
        rho: 0.5,
    };
    let v = spec.variance_at_with_correlations(1, 10.0, &[0.2, 1.0], &[corr]);
    assert!(v.is_nan());
}

#[test]
fn variance_at_with_correlations_nan_when_sigma_index_missing() {
    // A PerCmt combined endpoint whose sigma_idx (0, 1) points past a
    // length-1 sigma slice: the diagonal accumulation hits a missing sigma
    // and returns NaN rather than silently dropping a term.
    let mut map = std::collections::HashMap::new();
    map.insert(
        1usize,
        EndpointError {
            error_model: ErrorModel::Combined,
            sigma_idx: vec![0, 1],
        },
    );
    let spec = ErrorSpec::PerCmt(map);
    let corr = ResidualCorrelation {
        sigma_i: 0,
        sigma_j: 1,
        rho: 0.5,
    };
    let v = spec.variance_at_with_correlations(1, 10.0, &[0.2], &[corr]);
    assert!(v.is_nan());
}

#[test]
fn variance_at_with_correlations_skips_unloaded_correlation() {
    // The correlation names sigma index 2, which the combined endpoint
    // never loads, so the cross term is skipped and V is the plain
    // diagonal (10·0.2)² + 1² = 5.
    let spec = ErrorSpec::Single(ErrorModel::Combined);
    let corr = ResidualCorrelation {
        sigma_i: 0,
        sigma_j: 2,
        rho: 0.5,
    };
    let v = spec.variance_at_with_correlations(0, 10.0, &[0.2, 1.0, 0.3], &[corr]);
    assert_relative_eq!(v, 5.0, epsilon = 1e-12);
}

#[test]
fn residual_variance_at_threads_correlations_from_model() {
    // The `CompiledModel` convenience wrapper must apply its own
    // `residual_correlations`, not just the diagonal.
    let mut model = test_helpers::analytical_model(GradientMethod::Fd);
    model.error_spec = ErrorSpec::Single(ErrorModel::Combined);
    model.error_model = ErrorModel::Combined;
    model.residual_correlations = vec![ResidualCorrelation {
        sigma_i: 0,
        sigma_j: 1,
        rho: 0.5,
    }];
    let v = model.residual_variance_at(0, 10.0, &[0.2, 1.0]);
    assert_relative_eq!(v, 7.0, epsilon = 1e-12);
}

#[test]
fn has_residual_error_for_cmt_covers_single_and_per_cmt() {
    let mut model = test_helpers::analytical_model(GradientMethod::Fd);

    // Single error model: defined for every cmt iff sigma covers its n_sigma.
    model.error_spec = ErrorSpec::Single(ErrorModel::Proportional);
    assert!(model.has_residual_error_for_cmt(1, &[0.1]));
    assert!(model.has_residual_error_for_cmt(99, &[0.1]));
    assert!(
        !model.has_residual_error_for_cmt(1, &[]),
        "no sigma ⇒ no error model"
    );
    // A Combined model needs two sigmas; one is too short — `residual_variance`
    // would otherwise index `sigma[1]` and PANIC (symmetric to the PerCmt
    // out-of-range guard below, never a fabricated σ).
    model.error_spec = ErrorSpec::Single(ErrorModel::Combined);
    assert!(model.has_residual_error_for_cmt(1, &[0.1, 0.2]));
    assert!(
        !model.has_residual_error_for_cmt(1, &[0.1]),
        "Combined needs 2 sigmas; 1 is too short to draw a DV"
    );

    // PerCmt: defined only for compartments present in the map.
    let mut map = std::collections::HashMap::new();
    map.insert(
        2usize,
        EndpointError {
            error_model: ErrorModel::Proportional,
            sigma_idx: vec![0],
        },
    );
    model.error_spec = ErrorSpec::PerCmt(map);
    assert!(
        model.has_residual_error_for_cmt(2, &[0.1]),
        "cmt 2 has an endpoint"
    );
    assert!(
        !model.has_residual_error_for_cmt(1, &[0.1]),
        "cmt 1 is not in the map"
    );
    // An endpoint whose sigma_idx falls outside `sigma` is not drawable —
    // `residual_variance_at` would otherwise return NaN (symmetric to the
    // Single empty-sigma guard, never a fabricated σ).
    assert!(
        !model.has_residual_error_for_cmt(2, &[]),
        "cmt 2's sigma_idx is out of range for an empty sigma"
    );
    // A Combined endpoint needs two sigma indices; one is too few — `variance_at`
    // would read `sigma_idx[1]` (absent) and return NaN even though sigma is long.
    let mut combined_map = std::collections::HashMap::new();
    combined_map.insert(
        2usize,
        EndpointError {
            error_model: ErrorModel::Combined,
            sigma_idx: vec![0], // only one index for a two-sigma model
        },
    );
    model.error_spec = ErrorSpec::PerCmt(combined_map);
    assert!(
        !model.has_residual_error_for_cmt(2, &[0.1, 0.2]),
        "a Combined endpoint with too few sigma indices is not drawable"
    );
}

/// The transit/IG source used by the #814 twin-tolerance tests. A closed-form
/// `one_cpt_transit` + IOV model: analytic primary (`ode_spec == None`), carries an ODE
/// twin, and every subject reroutes to it (`n_kappa > 0`). Its `[fit_options]` sets a
/// distinctive `ode_reltol = 1e-8` (≠ `OdeSolverOptions::default` 1e-4) so a source-default
/// read is distinguishable from a call-time override.
const TRANSIT_IOV_TOL_SRC: &str = "\
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVMTT(1.0, 0.01, 50.0)
  theta TVN(3.0, 0.5, 20.0)
  omega ETA_V ~ 0.09
  kappa KAPPA_CL ~ 0.02
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL  = TVCL * exp(KAPPA_CL)
  V   = TVV  * exp(ETA_V)
  MTT = TVMTT
  NTR = TVN
[structural_model]
  pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method        = focei
  iov_column    = OCC
  ode_reltol    = 1e-8
  ode_abstol    = 1e-8
  ode_max_steps = 5000
";

#[test]
fn call_time_ode_tolerances_reach_the_absorption_twin() {
    // #814: a closed-form transit/IG model is analytic (`ode_spec == None`), but its IOV /
    // TV-cov subjects integrate on the ODE twin. A call-time `ode_reltol`/`ode_abstol`/
    // `ode_max_steps` — merged into the runtime FitOptions and applied via
    // `sync_ode_solver_opts`, exactly as ferx-r's `ferx_fit` does — must reach that twin
    // rather than be silently dropped at the twin source's parse default.
    use crate::parser::model_parser::parse_model_string;

    // The override must reach the twin (built at parse time since #1008).
    let mut model = parse_model_string(TRANSIT_IOV_TOL_SRC).expect("parse transit+IOV");
    assert!(
        model.absorption_ode_equivalent.is_some(),
        "one_cpt_transit carries an ODE twin"
    );
    assert!(model.n_kappa > 0, "model has IOV");
    assert!(
        model.ode_spec.is_none(),
        "the closed-form transit primary is analytic"
    );

    let mut opts = FitOptions::default();
    opts.ode_reltol = 1e-10;
    opts.ode_abstol = 1e-11;
    opts.ode_max_steps = 77_777;
    model.sync_ode_solver_opts(&opts);

    // IOV ⇒ every subject reroutes to the twin; its OdeSpec must carry the *call-time*
    // tolerances, not the source's 1e-8 / 5000.
    let twin = model.effective_for(&bare_subject("1"));
    let so = twin
        .ode_spec
        .as_ref()
        .expect("IOV transit subject is served by the ODE twin")
        .solver_opts;
    assert_eq!(so.reltol, 1e-10, "call-time reltol reaches the twin");
    assert_eq!(so.abstol, 1e-11, "call-time abstol reaches the twin");
    assert_eq!(so.max_steps, 77_777, "call-time max_steps reaches the twin");

    // Default-path regression: with no call-time sync, the twin keeps its source
    // `[fit_options]` (1e-8 / 5000), NOT `OdeSolverOptions::default()` and NOT a leaked
    // override from another model.
    let untouched = parse_model_string(TRANSIT_IOV_TOL_SRC).expect("parse transit+IOV");
    let dso = untouched
        .effective_for(&bare_subject("2"))
        .ode_spec
        .as_ref()
        .unwrap()
        .solver_opts;
    assert_eq!(dso.reltol, 1e-8, "un-synced twin keeps its source reltol");
    assert_eq!(dso.abstol, 1e-8, "un-synced twin keeps its source abstol");
    assert_eq!(
        dso.max_steps, 5000,
        "un-synced twin keeps its source max_steps"
    );

    // Re-sync: a second fit reusing the same owned model with *different* tolerances must
    // overwrite the first sync's stamp, not keep it. (Before #1008 this block tested a
    // sync-before-build vs sync-after-build ordering; the twin is now always built by the
    // time anyone can sync, so the only distinction left worth pinning is second-sync-wins.)
    let mut tight = FitOptions::default();
    tight.ode_reltol = 1e-12;
    tight.ode_abstol = 1e-13;
    tight.ode_max_steps = 4242;
    model.sync_ode_solver_opts(&tight);
    let rso = model
        .effective_for(&bare_subject("3"))
        .ode_spec
        .as_ref()
        .unwrap()
        .solver_opts;
    assert_eq!(rso.reltol, 1e-12, "a second sync overwrites the first");
    assert_eq!(rso.abstol, 1e-13);
    assert_eq!(rso.max_steps, 4242);
}

// ── #1064: scale guardrails for models with hundreds of θ ──────────────────
mod theta_block_scale_guards {
    use crate::io::output::compact_theta_blocks;
    use crate::types::{Optimizer, BOBYQA_MAX_DIM, COV_HESSIAN_MAX_DIM};

    /// A gather model with `len` levels — an FD-gradient model by construction
    /// (the analytic provider caps at `MAX_SCALE_AXES` axes).
    fn wide_model(len: usize) -> crate::types::CompiledModel {
        let content = format!(
            r#"
[parameters]
  theta TVCL(2.0, 0.001, 10.0)
  theta PLACEBO[{len}](0.5, -10.0, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02
[individual_parameters]
  CL = TVCL + PLACEBO[PLA_IDX]
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#
        );
        crate::parser::model_parser::parse_full_model(&content)
            .unwrap()
            .model
    }

    #[test]
    fn free_packed_dim_counts_theta_omega_and_sigma() {
        let model = wide_model(10);
        // 12 θ + 1 Ω element + 1 σ.
        assert_eq!(model.free_packed_dim(), 14);
    }

    #[test]
    fn free_packed_dim_matches_separate_packed_masks() {
        let content = r#"
[parameters]
  theta TVCL(2.0, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0) FIX
  omega ETA_CL ~ 0.09
  block_omega (ETA_V, ETA_KA) = [0.04, 0.01, 0.16]
  kappa KAPPA_CL ~ 0.02
  kappa KAPPA_V ~ 0.03 FIX
  sigma ADD_ERR ~ 0.1 FIX
  sigma PROP_ERR ~ 0.02
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ combined(ADD_ERR, PROP_ERR)
"#;
        let model = crate::parser::model_parser::parse_full_model(content)
            .unwrap()
            .model;
        let p = &model.default_params;
        let fixed = crate::estimation::parameterization::packed_fixed_mask(p);
        let structural = crate::estimation::parameterization::omega_structural_zero_mask(p);
        let exact = fixed
            .iter()
            .zip(&structural)
            .filter(|(f, z)| !**f && !**z)
            .count();
        assert_eq!(model.free_packed_dim(), exact);
        assert_eq!(exact, 7);
    }

    #[test]
    fn free_packed_dim_ignores_fixed_omega_coordinates() {
        // The review case for this guard (#1095): 14 fixed diagonal etas plus
        // two free thetas and one free sigma is *three* free coordinates. The
        // original formula counted a dense triangular Omega regardless of FIX
        // flags and returned 108 — enough on its own to flip `optimizer = auto`
        // and the default covariance estimator on a model that has nothing to do
        // with level blocks. This is the assertion that would have caught it.
        let etas: String = (1..=14)
            .map(|i| format!("  omega ETA_{i} ~ 0.09 FIX\n"))
            .collect();
        let sum: Vec<String> = (1..=14).map(|i| format!("ETA_{i}")).collect();
        let content = format!(
            r#"
[parameters]
  theta TVCL(2.0, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
{etas}  sigma PROP_ERR ~ 0.02
[individual_parameters]
  CL = TVCL * exp({})
  V = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#,
            sum.join(" + ")
        );
        let model = crate::parser::model_parser::parse_full_model(&content)
            .unwrap()
            .model;
        assert_eq!(model.n_eta, 14);
        assert_eq!(
            model.free_packed_dim(),
            3,
            "two free thetas and one free sigma; every omega coordinate is fixed"
        );
        // And therefore neither scale guard fires on it.
        assert!(model.free_packed_dim() <= BOBYQA_MAX_DIM);
        assert!(model.free_packed_dim() <= COV_HESSIAN_MAX_DIM);
    }

    #[test]
    fn auto_keeps_bobyqa_at_ordinary_dimension() {
        // The guard must not disturb the historical `auto` behaviour on the
        // models everything else in this codebase is: `gradient = fd` removes
        // the analytic gradient, so BOBYQA is what `auto` has always picked.
        let mut model = wide_model(4);
        model.gradient_method = crate::types::GradientMethod::Fd;
        assert!(model.free_packed_dim() <= BOBYQA_MAX_DIM);
        assert_eq!(
            Optimizer::Auto.resolve_auto(&model, true),
            Optimizer::Bobyqa,
            "small models must be unaffected by the scale guard"
        );
    }

    #[test]
    fn auto_still_prefers_the_analytic_gradient_at_ordinary_dimension() {
        // A small gather model keeps its exact analytic gradient, so `auto`
        // resolves to the gradient-based optimizer exactly as it would without
        // the block.
        let model = wide_model(4);
        assert_eq!(
            Optimizer::Auto.resolve_auto(&model, true),
            Optimizer::NloptLbfgs
        );
    }

    #[test]
    fn auto_refuses_bobyqa_above_the_dimension_threshold() {
        // BOBYQA interpolates a quadratic over the whole space; at several
        // hundred θ the interpolation set alone is the whole budget.
        let model = wide_model(BOBYQA_MAX_DIM + 10);
        assert!(model.free_packed_dim() > BOBYQA_MAX_DIM);
        assert_eq!(
            Optimizer::Auto.resolve_auto(&model, true),
            Optimizer::NloptLbfgs
        );
    }

    #[test]
    fn an_explicit_optimizer_is_never_overridden_by_the_guard() {
        let model = wide_model(BOBYQA_MAX_DIM + 10);
        for opt in [
            Optimizer::Bobyqa,
            Optimizer::Slsqp,
            Optimizer::Mma,
            Optimizer::TrustRegion,
        ] {
            assert_eq!(opt.resolve_auto(&model, true), opt);
        }
    }

    #[test]
    fn a_wide_gather_model_bails_out_of_the_analytic_dual_ladder() {
        // The `Dual2<M>` dispatch runs `1..=MAX_SCALE_AXES` axes. A
        // several-hundred-θ block is past the table, and the contract there is
        // a clean decline — the caller then finite-differences that subject —
        // never a truncation to the first 24 axes, which would be a wrong
        // gradient nothing else in the tree could contradict.
        let model = wide_model(400);
        let prog = model
            .indiv_param_partials
            .indiv_param_program
            .as_ref()
            .expect("program");
        let theta = vec![1.0; model.n_theta];
        let cov = std::collections::HashMap::from([("PLA_IDX".to_string(), 1.0)]);
        assert!(crate::sens::ode_provider::param_derivatives_at_cov(
            prog,
            &model,
            &cov,
            &theta,
            &[0.0]
        )
        .is_none());
    }

    #[test]
    fn a_narrow_gather_model_stays_in_analytic_sens_scope() {
        // The gather itself must not knock a model out of scope — `∂f/∂θ_k`
        // through it is exact under `Dual2` (pinned against finite differences
        // in `sens::provider::tests::theta_gather_sens`).
        let model = wide_model(3);
        assert!(
            crate::sens::provider::sens_supported(&model),
            "a gather is analytically differentiable; only the axis count gates dispatch"
        );
    }

    #[test]
    fn cov_hessian_threshold_is_above_the_bobyqa_one() {
        // The covariance step is quadratic in n where the optimizer is only
        // linear, but it runs once — so it tolerates more, not less.
        assert!(COV_HESSIAN_MAX_DIM > BOBYQA_MAX_DIM);
    }

    #[test]
    fn compact_theta_blocks_finds_only_large_contiguous_blocks() {
        let mut names = vec!["TVCL".to_string()];
        names.extend((1..=30).map(|i| format!("PLACEBO[{i}]")));
        names.push("TVV".to_string());
        // A small block stays inline.
        names.extend((1..=3).map(|i| format!("SMALL[{i}]")));
        let blocks = compact_theta_blocks(&names);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].0, "PLACEBO");
        assert_eq!(blocks[0].1, 1..31);
    }

    #[test]
    fn compact_theta_blocks_ignores_a_model_with_no_blocks() {
        let names = vec!["TVCL".to_string(), "TVV".to_string()];
        assert!(compact_theta_blocks(&names).is_empty());
    }
}
