use super::*;

/// #486: the residual-magnitude direct-θ program dispatches its `Dual1<M>`
/// evaluator for every axis count up to [`MAX_RUV_MAG_AXES`] (raised past 16),
/// and declines (`None`, caller → FD) only beyond it. `mult = θ_last · σ̂` (σ̂
/// pinned to 1.0), so `∂mult/∂θ_last = 1` and all other axes are 0 — a cheap
/// exact check that each newly-added dispatch arm is wired.
#[test]
fn ruv_mag_theta_grad_dispatches_up_to_axis_cap() {
    use std::collections::HashMap;
    assert!(MAX_RUV_MAG_AXES >= 17, "cap must be raised past the old 16");
    let cov = HashMap::new();
    // Every axis count up to the cap — exercises each `Dual1<M>` dispatch arm.
    for n in 1..=MAX_RUV_MAG_AXES {
        let expr = Expression::BinOp(
            Box::new(Expression::Theta(n - 1)),
            BinOp::Mul,
            Box::new(Expression::Variable("PROP".to_string())),
        );
        let prog = compile_ruv_mag_deriv_program(&expr, Some("PROP"), n);
        let theta = vec![0.5f64; n];
        let grad = prog
            .theta_grad(&theta, &cov, 0.0, f64::NAN)
            .unwrap_or_else(|| panic!("theta_grad must dispatch for n_theta = {n}"));
        assert_eq!(grad.len(), n);
        assert!(
            (grad[n - 1] - 1.0).abs() < 1e-9,
            "∂mult/∂θ_last = 1 (n={n})"
        );
        for (i, g) in grad.iter().enumerate() {
            if i != n - 1 {
                assert!(
                    g.abs() < 1e-9,
                    "off-axis derivative must be 0 (n={n}, i={i})"
                );
            }
        }
    }
    // Beyond the cap → None, so the outer gradient falls back to FD.
    let over = MAX_RUV_MAG_AXES + 1;
    let expr = Expression::BinOp(
        Box::new(Expression::Theta(0)),
        BinOp::Mul,
        Box::new(Expression::Variable("PROP".to_string())),
    );
    let prog = compile_ruv_mag_deriv_program(&expr, Some("PROP"), over);
    assert!(
        prog.theta_grad(&vec![0.5; over], &HashMap::new(), 0.0, f64::NAN)
            .is_none(),
        "n_theta beyond MAX_RUV_MAG_AXES must decline to FD"
    );
}

// ── ode_template desugaring + error rule (#322 Phase 0b) ─────────────────

/// Parse `src`, asserting it fails; returns the error message. (`ParsedModel`
/// is not `Debug`, so `unwrap_err()` can't be used directly.)
fn parse_err(src: &str) -> String {
    match parse_full_model(src) {
        Ok(_) => panic!("expected a parse error, but the model parsed"),
        Err(e) => e,
    }
}

/// Two-cpt-oral parameter header (CL/V1/Q/V2/KA) plus optional extra
/// individual params, used to build `ode_template` test models.
fn two_cpt_oral_model(structural: &str, extra_indiv: &str, odes: &str) -> String {
    format!(
        "[parameters]\n\
             \x20 theta TVCL(3.0, 0.01, 100.0)\n\
             \x20 theta TVV1(15.0, 1.0, 500.0)\n\
             \x20 theta TVQ(3.0, 0.01, 100.0)\n\
             \x20 theta TVV2(30.0, 1.0, 500.0)\n\
             \x20 theta TVKA(1.1, 0.01, 50.0)\n\
             \x20 omega ETA_CL ~ 0.09\n\
             \x20 sigma PROP ~ 0.01 (sd)\n\n\
             [individual_parameters]\n\
             \x20 CL = TVCL * exp(ETA_CL)\n\
             \x20 V1 = TVV1\n\
             \x20 Q  = TVQ\n\
             \x20 V2 = TVV2\n\
             \x20 KA = TVKA\n{extra_indiv}\n\
             [structural_model]\n  {structural}\n\n{odes}\
             [error_model]\n  DV ~ proportional(PROP)\n"
    )
}

#[test]
fn ode_template_desugars_to_ode_model() {
    let src = two_cpt_oral_model(
        "ode_template two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)",
        "",
        "",
    );
    let model = parse_full_model(&src)
        .unwrap_or_else(|e| panic!("ode_template should parse: {e}"))
        .model;
    let ode = model
        .ode_spec
        .expect("ode_template must produce an ODE model");
    assert_eq!(ode.state_names, vec!["depot", "central", "periph"]);
    // Pure disposition — no built-in input-rate forcing without an override.
    assert!(ode.input_rate.is_empty());
    // Observed compartment is central (state index 1).
    match ode.readout {
        crate::ode::OdeReadout::ObsCmt(idx) => assert_eq!(idx, 1),
        _ => panic!("expected an ObsCmt readout for ode_template"),
    }
}

#[test]
fn ode_template_override_replaces_only_named_compartment() {
    // Override the generated depot with a transit input; central & periph
    // keep their generated equations (so the 3-state structure is intact and
    // exactly one transit forcing lands on the depot, compartment 0).
    let src = two_cpt_oral_model(
        "ode_template two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)",
        "  NTR = 3.0\n  MTT = 1.0\n",
        "[odes]\n  d/dt(depot) = transit(n=NTR, mtt=MTT) - KA*depot\n\n",
    );
    let model = parse_full_model(&src)
        .unwrap_or_else(|e| panic!("override should parse: {e}"))
        .model;
    let ode = model.ode_spec.expect("ODE model");
    assert_eq!(ode.state_names, vec!["depot", "central", "periph"]);
    assert_eq!(ode.input_rate.len(), 1, "exactly one transit forcing");
    assert_eq!(ode.input_rate[0].cmt, 0, "transit forces the depot (cmt 0)");
}

#[test]
fn ode_template_override_unknown_compartment_errors() {
    let src = two_cpt_oral_model(
        "ode_template two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)",
        "",
        "[odes]\n  d/dt(gut) = -KA*gut\n\n",
    );
    let err = parse_err(&src);
    assert!(
        err.contains("d/dt(gut)") && err.contains("generated states"),
        "got: {err}"
    );
}

#[test]
fn ode_template_missing_required_role_errors() {
    // Surfaced through the full-model parse, not just the generator.
    let src = two_cpt_oral_model(
        "ode_template two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2)",
        "",
        "",
    );
    let err = parse_err(&src);
    assert!(err.contains("requires `ka`"), "got: {err}");
}

#[test]
fn ode_template_rejects_mixing_with_pk_or_ode() {
    let src = two_cpt_oral_model(
        "ode_template two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)\n  pk one_cpt_iv(cl=CL, v=V1)",
        "",
        "",
    );
    let err = parse_err(&src);
    assert!(err.contains("cannot be combined"), "got: {err}");
}

#[test]
fn transit_time_desugars_to_ode_equivalent() {
    // Plain `pk one_cpt_transit(...)` + a TIME switch is rewritten to the exact ODE
    // transit() twin (which carries per-event TIME analytically, #664) instead of being
    // rejected — the closed form can't honour a mid-profile parameter switch (#486).
    const TRANSIT_TIME: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVCL_LATE(7.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVMTT(1.0, 0.05, 24.0)
  theta TVN(3.0, 0.0, 30.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  if (TIME > 12.0) {
    CL = TVCL_LATE * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V   = TVV
  MTT = TVMTT
  NTR = TVN
[structural_model]
  pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
"#;
    let model = parse_model_string(TRANSIT_TIME).expect("parse transit+TIME");
    // The primary model stays the fast closed-form transit; it CARRIES an exact ODE
    // `transit()` equivalent that the runtime dispatch uses only for the subjects the
    // closed form can't serve (here, every subject — the model reads TIME).
    assert_eq!(
        model.pk_model,
        PkModel::OneCptTransit,
        "primary model is still the closed-form transit"
    );
    assert!(
        model.ode_spec.is_none(),
        "the primary transit model is not itself an ODE model"
    );
    let eq = model
        .absorption_ode_equivalent
        .as_ref()
        .expect("transit + TIME must carry an ODE equivalent")
        .built();
    let ode = eq
        .ode_spec
        .as_ref()
        .expect("the equivalent is an ODE model");
    assert_eq!(ode.state_names, vec!["central"]);
    assert_eq!(ode.input_rate.len(), 1, "exactly one transit forcing");
    assert!(
        compiled_model_uses_time_builtin(&model),
        "the TIME switch is on the model"
    );

    // A transit model WITHOUT a TIME switch keeps the fast, exact closed form as its
    // primary representation (the equivalent is still built, but the dispatch never uses
    // it unless a subject carries time-varying covariates).
    const TRANSIT_PLAIN: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVMTT(1.0, 0.05, 24.0)
  theta TVN(3.0, 0.0, 30.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  MTT = TVMTT
  NTR = TVN
[structural_model]
  pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
"#;
    let plain = parse_model_string(TRANSIT_PLAIN).expect("parse plain transit");
    assert_eq!(
        plain.pk_model,
        PkModel::OneCptTransit,
        "no TIME switch → keep the closed-form transit"
    );
    assert!(plain.ode_spec.is_none());
}

#[test]
fn ode_template_malformed_arg_errors() {
    let src = two_cpt_oral_model("ode_template one_cpt_iv(cl)", "", "");
    let err = parse_err(&src);
    assert!(err.contains("malformed parameter"), "got: {err}");
}

#[test]
fn ode_template_missing_parens_errors_clearly() {
    // A line that *looks* like an ode_template directive but doesn't match
    // `NAME(...)` must produce a clear "malformed ode_template" error — not
    // fall through to a confusing "No PK model found". Pair it with a
    // transit() in [odes] (the worst case: the error rule would otherwise
    // tell the user to "use ode_template" when they already are).
    let src = two_cpt_oral_model(
        "ode_template two_cpt_oral",
        "  NTR = 3.0\n  MTT = 1.0\n",
        "[odes]\n  d/dt(depot) = transit(n=NTR, mtt=MTT) - KA*depot\n\n",
    );
    let err = parse_err(&src);
    assert!(
        err.contains("malformed `ode_template`"),
        "expected a clear malformed-ode_template error, got: {err}"
    );
}

#[test]
fn ode_template_duplicate_arg_errors() {
    let src = two_cpt_oral_model("ode_template one_cpt_iv(cl=CL, cl=V1)", "", "");
    let err = parse_err(&src);
    assert!(err.contains("duplicate parameter"), "got: {err}");
}

#[test]
fn ode_template_multiple_lines_errors() {
    let src = two_cpt_oral_model(
        "ode_template one_cpt_iv(cl=CL, v=V1)\n  ode_template two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)",
        "",
        "",
    );
    let err = parse_err(&src);
    assert!(err.contains("more than one"), "got: {err}");
}

#[test]
fn error_rule_analytical_pk_plus_transit() {
    // Analytical disposition + an ODE-only absorption function → hard error
    // pointing at ode_template, never a silent analytical→ODE swap.
    let src = two_cpt_oral_model(
        "pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)",
        "  NTR = 3.0\n  MTT = 1.0\n",
        "[odes]\n  d/dt(depot) = transit(n=NTR, mtt=MTT) - KA*depot\n\n",
    );
    let err = parse_err(&src);
    assert!(
        err.contains("ode_template"),
        "should point at ode_template: {err}"
    );
    assert!(err.contains("transit"), "should name the function: {err}");
}

#[test]
fn error_rule_analytical_pk_plus_weibull() {
    // Weibull has no closed form, so analytical `pk` + `weibull()` is a hard
    // error pointing at ode_template — and stays one permanently (unlike
    // transit/igd, it can never route to the analytical path).
    let src = two_cpt_oral_model(
        "pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)",
        "  TD = 2.0\n  BETA = 1.5\n",
        "[odes]\n  d/dt(depot) = weibull(td=TD, beta=BETA) - KA*depot\n\n",
    );
    let err = parse_err(&src);
    assert!(
        err.contains("ode_template"),
        "should point at ode_template: {err}"
    );
    assert!(err.contains("weibull"), "should name the function: {err}");
}

#[test]
fn ode_only_absorption_fn_detection() {
    let transit = vec!["d/dt(depot) = transit(n=N, mtt=M) - KA*depot".to_string()];
    assert_eq!(
        ode_only_absorption_fn_in_odes(Some(&transit)),
        Some("transit")
    );
    // `igd` is implemented as of #347, so the error rule now recognises it
    // (routing an analytical `pk` + `igd()` to `ode_template`).
    let igd = vec!["d/dt(central) = igd(mat=MAT, cv2=CV2) - (CL/V)*central".to_string()];
    assert_eq!(ode_only_absorption_fn_in_odes(Some(&igd)), Some("igd"));
    // `weibull` is implemented as of Phase 2, so the error rule now recognises
    // it. Unlike `transit`/`igd`, it has no closed form, so it stays on the
    // ODE-only list permanently (it can never route to an analytical `pk`).
    let weibull = vec!["d/dt(central) = weibull(td=TD, beta=B) - (CL/V)*central".to_string()];
    assert_eq!(
        ode_only_absorption_fn_in_odes(Some(&weibull)),
        Some("weibull")
    );
    // `zero_order` is implemented as of #504. It has a closed form (an
    // infusion), but #504 ships it ODE-only, so the error rule lists it until
    // the analytical acceleration lands (keeps `pk ... + zero_order(...)` from
    // silently dropping the absorption).
    let zero_order = vec!["d/dt(central) = zero_order(dur=DUR) - (CL/V)*central".to_string()];
    assert_eq!(
        ode_only_absorption_fn_in_odes(Some(&zero_order)),
        Some("zero_order")
    );
    // A plain disposition ODE has no ODE-only absorption call.
    let plain = vec!["d/dt(central) = -(CL/V)*central".to_string()];
    assert_eq!(ode_only_absorption_fn_in_odes(Some(&plain)), None);
    assert_eq!(ode_only_absorption_fn_in_odes(None), None);
}

#[test]
fn ode_template_rejects_mixing_with_ode() {
    // The `ode(...)` arm of the mixing guard (the `pk` arm is covered above).
    let src = two_cpt_oral_model(
        "ode_template two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)\n  \
             ode(obs_cmt=central, states=[depot, central, periph])",
        "",
        "",
    );
    let err = parse_err(&src);
    assert!(err.contains("cannot be combined"), "got: {err}");
}

#[test]
fn ode_template_trailing_comma_ok() {
    // A trailing comma yields an empty `role=VAR` pair, which is skipped.
    let src = two_cpt_oral_model(
        "ode_template two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA, )",
        "",
        "",
    );
    assert!(
        parse_full_model(&src).is_ok(),
        "a trailing comma in the parameter list should be tolerated"
    );
}

#[test]
fn ode_template_override_tolerates_whitespace_in_d_dt() {
    // `build_ode_spec` recognises `d/dt(NAME)` at the token level, so spacing
    // like `d/dt (central)` / `d / dt(central)` is a valid state equation
    // there. Override detection must agree — otherwise the override is missed,
    // the generated equation also survives, and the user gets a misleading
    // "duplicate d/dt(central)" error instead of their override taking effect.
    for lhs in ["d/dt (central)", "d / dt(central)", "d/dt(  central  )"] {
        let src = two_cpt_oral_model(
            "ode_template two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)",
            "",
            &format!("[odes]\n  {lhs} = KA*depot - (CL/V1 + Q/V1)*central + (Q/V2)*periph\n\n"),
        );
        let model = parse_full_model(&src)
            .unwrap_or_else(|e| panic!("spaced override `{lhs}` should parse: {e}"))
            .model;
        // Override consumed the generated central → still exactly 3 states,
        // and no duplicate-d/dt error fired.
        assert_eq!(
            model.ode_spec.expect("ODE model").state_names,
            vec!["depot", "central", "periph"],
            "override `{lhs}`"
        );
    }
}

#[test]
fn ode_template_rejected_with_diffusion_block() {
    // ode_template injects obs_scale (a concentration readout), which the SDE
    // path can't carry. Reject with a message that names the real cause
    // (`ode_template` + `[diffusion]`), not the injected `[scaling]` block the
    // user never wrote.
    let src = two_cpt_oral_model(
        "ode_template two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)",
        "",
        "[diffusion]\n  central ~ 0.1\n\n",
    );
    let err = parse_err(&src);
    assert!(
        err.contains("ode_template") && err.contains("[diffusion]"),
        "diffusion error should name ode_template + [diffusion]: {err}"
    );
}

#[test]
fn ode_template_transit_on_non_ddt_line_errors_with_pointer() {
    // Under ode_template, a `transit(...)` that is not the input rate of a
    // `d/dt(...)` equation (here a bare helper assignment) must not be silently
    // retained as a dead term: the input-rate extractor catches it and points
    // at the correct `d/dt(...)` form (Ron #363, finding 5 — confirmed already
    // guarded, pinned here so it can't regress).
    let src = two_cpt_oral_model(
        "ode_template two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)",
        "  NTR = 3.0\n  MTT = 1.0\n",
        "[odes]\n  rate = transit(n=NTR, mtt=MTT)\n\n",
    );
    let err = parse_err(&src);
    assert!(
        err.contains("d/dt"),
        "a transit() off a d/dt line should point at the d/dt form, got: {err}"
    );
}

// ── transit() input-rate parse-split (#322, design A) ────────────────────
#[test]
fn extract_transit_input_rate_basic() {
    let lines = vec![
        "d/dt(depot) = transit(n=NTR, mtt=MTT) - KA*depot".to_string(),
        "d/dt(central) = KA*depot - (CL/V)*central".to_string(),
    ];
    let states = vec!["depot".to_string(), "central".to_string()];
    let names = vec![
        "NTR".to_string(),
        "MTT".to_string(),
        "KA".to_string(),
        "CL".to_string(),
        "V".to_string(),
    ];
    let slots = vec![10, 11, 4, 0, 1];
    let (cleaned, forcings) = extract_input_rate_terms(&lines, &states, &names, &slots).unwrap();
    // transit(...) replaced by 0; the rest of the RHS is untouched.
    assert_eq!(cleaned[0], "d/dt(depot) = 0 - KA*depot");
    assert_eq!(cleaned[1], "d/dt(central) = KA*depot - (CL/V)*central");
    assert_eq!(forcings.len(), 1);
    assert_eq!(forcings[0].cmt, 0); // depot
    assert!(matches!(
        forcings[0].kind,
        crate::pk::absorption::InputRateKind::Transit
    ));
    assert_eq!(forcings[0].arg_slots, vec![10, 11]); // [n=NTR slot, mtt=MTT slot]
}

#[test]
fn extract_transit_input_rate_rejects_bad_specs() {
    let states = vec!["depot".to_string()];
    let names = vec!["NTR".to_string(), "MTT".to_string()];
    let slots = vec![0, 1];
    let go = |line: &str| extract_input_rate_terms(&[line.to_string()], &states, &names, &slots);

    assert!(go("d/dt(depot) = transit(n=NTR, mtt=ZZZ)")
        .unwrap_err()
        .contains("not a declared individual parameter"));
    assert!(go("d/dt(depot) = transit(n=NTR, foo=MTT)")
        .unwrap_err()
        .contains("no argument `foo`"));
    assert!(go("d/dt(depot) = transit(n=NTR)")
        .unwrap_err()
        .contains("missing required argument `mtt`"));
    assert!(go("x = transit(n=NTR, mtt=MTT)")
        .unwrap_err()
        .contains("d/dt"));
    // A leading multiplier that is not a declared individual parameter is not a
    // valid pathway fraction (#388): `FR` is undeclared here.
    assert!(go("d/dt(depot) = FR*transit(n=NTR, mtt=MTT)")
        .unwrap_err()
        .contains("pathway-fraction multiplier"));
    // Leading unary minus: the forcing is injected `+R_in`, so a negated call
    // would silently flip the sign of the input rate. Must be rejected.
    assert!(go("d/dt(depot) = -transit(n=NTR, mtt=MTT) - KA*depot")
        .unwrap_err()
        .contains("bare additive"));
    // Subtracted term (wrong sign) even without `*`/`/` scaling.
    assert!(go("d/dt(depot) = KA*depot - transit(n=NTR, mtt=MTT)")
        .unwrap_err()
        .contains("bare additive"));
    // Parenthesised + scaled outside the group: the old adjacency check saw
    // only the flanking `(`/`)`, so the `/V` was silently dropped — now rejected.
    assert!(go("d/dt(depot) = (transit(n=NTR, mtt=MTT))/V")
        .unwrap_err()
        .contains("bare additive"));
    assert!(go("d/dt(depot) = transit(n=NTR, mtt=MTT")
        .unwrap_err()
        .contains("unbalanced"));
    assert!(go("d/dt(depot) = transit(NTR, mtt=MTT)")
        .unwrap_err()
        .contains("name=parameter"));
    // Two bare transit terms now PARSE — parallel/biphasic absorption is allowed
    // at the parser layer (#388); the "must be fractioned" rule is a fit-init
    // check (see `check_absorption_dosing` / `E_ABSORPTION_FRACTION`).
    let (_c, two) = go("d/dt(depot) = transit(n=NTR, mtt=MTT) + transit(n=NTR, mtt=MTT)").unwrap();
    assert_eq!(two.len(), 2);
    assert!(two.iter().all(|f| f.frac_slot.is_none()));
    // Nested parens in an arg value are split correctly, then rejected as a
    // non-parameter (exercises the comma-splitter's paren-depth tracking).
    assert!(go("d/dt(depot) = transit(n=foo(a,b), mtt=MTT)")
        .unwrap_err()
        .contains("not a declared individual parameter"));
    // A word *ending* in `transit` (e.g. `xtransit(`) is not a transit() call:
    // left unchanged, no forcing recorded.
    let (kept, none) = go("d/dt(depot) = xtransit(n=NTR, mtt=MTT) - depot").unwrap();
    assert_eq!(kept[0], "d/dt(depot) = xtransit(n=NTR, mtt=MTT) - depot");
    assert!(none.is_empty());
    // No transit() → unchanged, no forcings.
    let (cleaned, f) = go("d/dt(depot) = -KA*depot").unwrap();
    assert_eq!(cleaned[0], "d/dt(depot) = -KA*depot");
    assert!(f.is_empty());

    // Positive controls for the tightened sign/scale guard: a bare additive
    // `transit(...)` is accepted whether it is the only term, the first term
    // before a `-` disposition term, or a later `+` term.
    assert!(go("d/dt(depot) = transit(n=NTR, mtt=MTT)").is_ok());
    assert!(go("d/dt(depot) = transit(n=NTR, mtt=MTT) - KA*depot").is_ok());
    assert!(go("d/dt(depot) = -KA*depot + transit(n=NTR, mtt=MTT)").is_ok());
}

// ── igd() input-rate parse-split (#347, design A) ─────────────────────────
#[test]
fn extract_igd_input_rate_basic() {
    // igd(mat, cv2) straight into central: forcing on the central cmt, args
    // resolved to the [mat, cv2] slots in order; the rest of the RHS untouched.
    let lines = vec!["d/dt(central) = igd(mat=MAT, cv2=CV2) - CL/V*central".to_string()];
    let states = vec!["depot".to_string(), "central".to_string()];
    let names = vec![
        "MAT".to_string(),
        "CV2".to_string(),
        "CL".to_string(),
        "V".to_string(),
    ];
    let slots = vec![4, 5, 0, 1];
    let (cleaned, forcings) = extract_input_rate_terms(&lines, &states, &names, &slots).unwrap();
    assert_eq!(cleaned[0], "d/dt(central) = 0 - CL/V*central");
    assert_eq!(forcings.len(), 1);
    assert_eq!(forcings[0].cmt, 1); // central
    assert!(matches!(
        forcings[0].kind,
        crate::pk::absorption::InputRateKind::InverseGaussian
    ));
    assert_eq!(forcings[0].arg_slots, vec![4, 5]); // [mat=MAT slot, cv2=CV2 slot]
}

#[test]
fn extract_biphasic_igd_with_fractions() {
    // Freijer biphasic IG (#388): FR1*igd(...) + FR2*igd(...) on one compartment.
    // Two forcings, each with its declared-parameter fraction slot; the `FR*`
    // prefix is swallowed in the cleaned RHS (each term → `0`).
    let lines = vec!["d/dt(central) = FR1*igd(mat=MAT1, cv2=CV2_1) + FR2*igd(mat=MAT2, cv2=CV2_2) - CL/V*central".to_string()];
    let states = vec!["central".to_string()];
    let names = vec![
        "FR1".to_string(),
        "MAT1".to_string(),
        "CV2_1".to_string(),
        "FR2".to_string(),
        "MAT2".to_string(),
        "CV2_2".to_string(),
        "CL".to_string(),
        "V".to_string(),
    ];
    let slots = vec![0, 1, 2, 3, 4, 5, 6, 7];
    let (cleaned, forcings) = extract_input_rate_terms(&lines, &states, &names, &slots).unwrap();
    assert_eq!(cleaned[0], "d/dt(central) = 0 + 0 - CL/V*central");
    assert_eq!(forcings.len(), 2);
    // Pathway 1: FR1 (slot 0) × igd(MAT1@1, CV2_1@2).
    assert_eq!(forcings[0].cmt, 0);
    assert_eq!(forcings[0].frac_slot, Some(0));
    assert_eq!(forcings[0].arg_slots, vec![1, 2]);
    // Pathway 2: FR2 (slot 3) × igd(MAT2@4, CV2_2@5).
    assert_eq!(forcings[1].frac_slot, Some(3));
    assert_eq!(forcings[1].arg_slots, vec![4, 5]);
}

#[test]
fn extract_fraction_must_be_a_declared_parameter() {
    let states = vec!["central".to_string()];
    let names = vec!["FR".to_string(), "MAT".to_string(), "CV2".to_string()];
    let slots = vec![0, 4, 5];
    let go = |line: &str| extract_input_rate_terms(&[line.to_string()], &states, &names, &slots);

    // A bare declared-parameter multiplier IS accepted and resolves to its slot.
    let (_c, f) = go("d/dt(central) = FR*igd(mat=MAT, cv2=CV2)").unwrap();
    assert_eq!(f[0].frac_slot, Some(0));

    // Whitespace around the `*` is tolerated (the prefix scanner skips spaces on
    // both sides of the multiplier): `FR *igd`, `FR* igd`, `FR * igd` all resolve.
    for line in [
        "d/dt(central) = FR *igd(mat=MAT, cv2=CV2)",
        "d/dt(central) = FR* igd(mat=MAT, cv2=CV2)",
        "d/dt(central) = FR * igd(mat=MAT, cv2=CV2)",
    ] {
        let (_c, f) = go(line).unwrap();
        assert_eq!(f[0].frac_slot, Some(0), "spaced fraction: {line}");
    }

    // `(1-FR)` is an expression, not a bare declared parameter — rejected (#388);
    // the user must declare a complementary parameter.
    let err =
        go("d/dt(central) = FR*igd(mat=MAT, cv2=CV2) + (1-FR)*igd(mat=MAT, cv2=CV2)").unwrap_err();
    assert!(
        err.contains("parentheses") || err.contains("pathway"),
        "got: {err}"
    );

    // An undeclared multiplier names the offender and points at the fix.
    let err2 = go("d/dt(central) = ZZ*igd(mat=MAT, cv2=CV2)").unwrap_err();
    assert!(err2.contains("pathway-fraction multiplier"), "got: {err2}");

    // A *declared* fraction in a negated/subtracted position is a dropped-sign
    // error, not a "not a declared parameter" one: `FR` here IS declared, the
    // defect is the leading `-`. Report it as the sign/scale rejection so the
    // message is not misleading (regression for the misattributed diagnostic).
    let err3 = go("d/dt(central) = MAT*central - FR*igd(mat=MAT, cv2=CV2)").unwrap_err();
    assert!(
        err3.contains("bare additive") && err3.contains("negated"),
        "got: {err3}"
    );
    // A compound multiplier (`K*FR*`) where both names are declared is likewise a
    // scale rejection, not a non-parameter one.
    let err4 = go("d/dt(central) = MAT*FR*igd(mat=MAT, cv2=CV2)").unwrap_err();
    assert!(err4.contains("bare additive"), "got: {err4}");
    // A numeric-literal multiplier is a dropped scale, not a "declared parameter"
    // problem — report it as the sign/scale rejection (review finding #2).
    let err5 = go("d/dt(central) = 5*igd(mat=MAT, cv2=CV2)").unwrap_err();
    assert!(
        err5.contains("bare additive") && !err5.contains("pathway-fraction multiplier"),
        "got: {err5}"
    );
    let err6 = go("d/dt(central) = 2.5*igd(mat=MAT, cv2=CV2)").unwrap_err();
    assert!(err6.contains("bare additive"), "got: {err6}");
}

#[test]
fn extract_input_rate_non_ascii_does_not_panic() {
    // The input-rate extractor byte-indexes the RHS, but every slice point is an
    // ASCII anchor (`(` `)`, the fn name, the identifier/`*` of a fraction), so a
    // multi-byte char *between* anchors must surface as a clean error or pass
    // through untouched — never a non-char-boundary slice panic (review finding #4).
    let states = vec!["central".to_string()];
    let names = vec!["MAT".to_string(), "CV2".to_string()];
    let slots = vec![4, 5];
    let go = |line: &str| extract_input_rate_terms(&[line.to_string()], &states, &names, &slots);

    // Unicode inside an arg value (param name): clean "not declared" error.
    assert!(go("d/dt(central) = igd(mat=MÄT, cv2=CV2)").is_err());
    // Unicode minus (U+2212) and an accented word outside the call: the igd term is
    // still extracted and the non-ASCII tail is preserved verbatim.
    let (cleaned, f) = go("d/dt(central) = igd(mat=MAT, cv2=CV2) \u{2212} café").unwrap();
    assert_eq!(f.len(), 1);
    assert_eq!(cleaned[0], "d/dt(central) = 0 \u{2212} café");
    // Unicode immediately before a `+`-led bare term.
    let (_c, f2) = go("d/dt(central) = café + igd(mat=MAT, cv2=CV2)").unwrap();
    assert_eq!(f2.len(), 1);
}

#[test]
fn extract_igd_input_rate_rejects_bad_specs() {
    let states = vec!["central".to_string()];
    let names = vec!["MAT".to_string(), "CV2".to_string()];
    let slots = vec![4, 5];
    let go = |line: &str| extract_input_rate_terms(&[line.to_string()], &states, &names, &slots);

    assert!(go("d/dt(central) = igd(mat=MAT, cv2=ZZZ)")
        .unwrap_err()
        .contains("not a declared individual parameter"));
    assert!(go("d/dt(central) = igd(mat=MAT, foo=CV2)")
        .unwrap_err()
        .contains("no argument `foo`"));
    // Missing arg names the offender and lists the expected names generically.
    let err = go("d/dt(central) = igd(mat=MAT)").unwrap_err();
    assert!(
        err.contains("missing required argument `cv2`"),
        "got: {err}"
    );
    // `FR*igd(...)` is a pathway-fraction multiplier (#388), but `FR` is not a
    // declared individual parameter here, so it is rejected as a non-parameter
    // fraction; the message names `igd`, not `transit`.
    let scaled = go("d/dt(central) = FR*igd(mat=MAT, cv2=CV2)").unwrap_err();
    assert!(
        scaled.contains("pathway-fraction multiplier") && scaled.contains("igd"),
        "got: {scaled}"
    );
    assert!(go("d/dt(central) = igd(mat=MAT, cv2=CV2")
        .unwrap_err()
        .contains("unbalanced"));
    assert!(go("d/dt(central) = igd(MAT, cv2=CV2)")
        .unwrap_err()
        .contains("name=parameter"));
    // Two bare igd calls on one equation now PARSE (biphasic sum, #388) into two
    // unfractioned forcings; the "each term needs a fraction" rule is enforced at
    // fit-init (`check_absorption_dosing`), not here.
    let (_c, two) = go("d/dt(central) = igd(mat=MAT, cv2=CV2) + igd(mat=MAT, cv2=CV2)").unwrap();
    assert_eq!(two.len(), 2);
    assert!(two.iter().all(|f| f.frac_slot.is_none()));
    // A word *ending* in `igd` (e.g. `xigd(`) is not an igd() call.
    let (kept, none) = go("d/dt(central) = xigd(mat=MAT, cv2=CV2) - central").unwrap();
    assert_eq!(kept[0], "d/dt(central) = xigd(mat=MAT, cv2=CV2) - central");
    assert!(none.is_empty());
    // Positive control: a bare additive igd() is accepted.
    assert!(go("d/dt(central) = igd(mat=MAT, cv2=CV2) - CL/V*central").is_ok());
}

// ── weibull() input-rate parse-split (#322 Phase 2, design A) ─────────────
#[test]
fn extract_weibull_input_rate_basic() {
    // weibull(td, beta) straight into central: forcing on the central cmt,
    // args resolved to the [td, beta] slots in order; the rest of the RHS
    // untouched.
    let lines = vec!["d/dt(central) = weibull(td=TD, beta=BETA) - CL/V*central".to_string()];
    let states = vec!["depot".to_string(), "central".to_string()];
    let names = vec![
        "TD".to_string(),
        "BETA".to_string(),
        "CL".to_string(),
        "V".to_string(),
    ];
    let slots = vec![4, 5, 0, 1];
    let (cleaned, forcings) = extract_input_rate_terms(&lines, &states, &names, &slots).unwrap();
    assert_eq!(cleaned[0], "d/dt(central) = 0 - CL/V*central");
    assert_eq!(forcings.len(), 1);
    assert_eq!(forcings[0].cmt, 1); // central
    assert!(matches!(
        forcings[0].kind,
        crate::pk::absorption::InputRateKind::Weibull
    ));
    assert_eq!(forcings[0].arg_slots, vec![4, 5]); // [td=TD slot, beta=BETA slot]
}

#[test]
fn extract_weibull_input_rate_rejects_bad_specs() {
    let states = vec!["central".to_string()];
    let names = vec!["TD".to_string(), "BETA".to_string()];
    let slots = vec![4, 5];
    let go = |line: &str| extract_input_rate_terms(&[line.to_string()], &states, &names, &slots);

    assert!(go("d/dt(central) = weibull(td=TD, beta=ZZZ)")
        .unwrap_err()
        .contains("not a declared individual parameter"));
    assert!(go("d/dt(central) = weibull(td=TD, foo=BETA)")
        .unwrap_err()
        .contains("no argument `foo`"));
    let err = go("d/dt(central) = weibull(td=TD)").unwrap_err();
    assert!(
        err.contains("missing required argument `beta`"),
        "got: {err}"
    );
    // `FR*weibull(...)` is a pathway-fraction multiplier (#388) but `FR` is
    // undeclared, so it is rejected as a non-parameter fraction naming `weibull`.
    let scaled = go("d/dt(central) = FR*weibull(td=TD, beta=BETA)").unwrap_err();
    assert!(
        scaled.contains("pathway-fraction multiplier") && scaled.contains("weibull"),
        "got: {scaled}"
    );
    // Positive control: a bare additive weibull() is accepted.
    assert!(go("d/dt(central) = weibull(td=TD, beta=BETA) - CL/V*central").is_ok());
}

// ── zero_order() input-rate parse-split (#504) ────────────────────────────
#[test]
fn extract_zero_order_input_rate_basic() {
    // zero_order(dur) straight into central: forcing on the central cmt, the
    // single `dur` arg resolved to its slot; the rest of the RHS untouched.
    let lines = vec!["d/dt(central) = zero_order(dur=DUR) - CL/V*central".to_string()];
    let states = vec!["depot".to_string(), "central".to_string()];
    let names = vec!["DUR".to_string(), "CL".to_string(), "V".to_string()];
    let slots = vec![4, 0, 1];
    let (cleaned, forcings) = extract_input_rate_terms(&lines, &states, &names, &slots).unwrap();
    assert_eq!(cleaned[0], "d/dt(central) = 0 - CL/V*central");
    assert_eq!(forcings.len(), 1);
    assert_eq!(forcings[0].cmt, 1); // central
    assert!(matches!(
        forcings[0].kind,
        crate::pk::absorption::InputRateKind::ZeroOrder
    ));
    assert_eq!(forcings[0].arg_slots, vec![4]); // single [dur=DUR slot]
}

#[test]
fn extract_zero_order_into_depot_for_sequential() {
    // `sequential` is zero_order into a depot, emptied to central by a
    // hand-written `- KA*depot` (no new intrinsic): the forcing lands on the
    // depot, the transfer term is preserved.
    let lines = vec![
        "d/dt(depot) = zero_order(dur=DUR) - KA*depot".to_string(),
        "d/dt(central) = KA*depot - CL/V*central".to_string(),
    ];
    let states = vec!["depot".to_string(), "central".to_string()];
    let names = vec![
        "DUR".to_string(),
        "KA".to_string(),
        "CL".to_string(),
        "V".to_string(),
    ];
    let slots = vec![4, 5, 0, 1];
    let (cleaned, forcings) = extract_input_rate_terms(&lines, &states, &names, &slots).unwrap();
    assert_eq!(cleaned[0], "d/dt(depot) = 0 - KA*depot");
    assert_eq!(cleaned[1], "d/dt(central) = KA*depot - CL/V*central");
    assert_eq!(forcings.len(), 1);
    assert_eq!(forcings[0].cmt, 0); // depot
    assert_eq!(forcings[0].arg_slots, vec![4]);
}

#[test]
fn extract_zero_order_input_rate_rejects_bad_specs() {
    let states = vec!["central".to_string()];
    let names = vec!["DUR".to_string(), "FZO".to_string()];
    let slots = vec![4, 6];
    let go = |line: &str| extract_input_rate_terms(&[line.to_string()], &states, &names, &slots);

    assert!(go("d/dt(central) = zero_order(dur=ZZZ)")
        .unwrap_err()
        .contains("not a declared individual parameter"));
    assert!(go("d/dt(central) = zero_order(foo=DUR)")
        .unwrap_err()
        .contains("no argument `foo`"));
    // Empty parens: the single (empty) argument is rejected as not a
    // `name=parameter` pair — `dur` is required, so the call cannot be bare.
    assert!(go("d/dt(central) = zero_order()")
        .unwrap_err()
        .contains("name=parameter"));
    // A pathway fraction on zero_order with a *declared* `FZO` now parses (#505):
    // the zero-order channel carries the fraction (`mixed` model). The
    // `FZO` multiplier resolves to its slot and the term is a ZeroOrder forcing.
    let (_, frac_forcings) = go("d/dt(central) = FZO*zero_order(dur=DUR)").unwrap();
    assert_eq!(frac_forcings.len(), 1);
    assert_eq!(
        frac_forcings[0].kind,
        crate::pk::absorption::InputRateKind::ZeroOrder
    );
    assert_eq!(frac_forcings[0].frac_slot, Some(6)); // FZO @ slot 6
    assert!(go("x = zero_order(dur=DUR)").unwrap_err().contains("d/dt"));
    // Positive control: a bare additive zero_order() is accepted.
    assert!(go("d/dt(central) = zero_order(dur=DUR) - CL/V*central").is_ok());
}

#[test]
fn extract_accepts_two_different_input_rate_fns_on_one_equation() {
    // transit + igd on a single d/dt now parses (#388 lifts the one-call guard):
    // two forcings on the same compartment, table order (transit before igd). The
    // "each must be fractioned" rule is a fit-init check, not a parser one.
    let states = vec!["central".to_string()];
    let names = vec![
        "NTR".to_string(),
        "MTT".to_string(),
        "MAT".to_string(),
        "CV2".to_string(),
    ];
    let slots = vec![10, 11, 4, 5];
    let line = "d/dt(central) = transit(n=NTR, mtt=MTT) + igd(mat=MAT, cv2=CV2)".to_string();
    let (cleaned, forcings) = extract_input_rate_terms(&[line], &states, &names, &slots).unwrap();
    assert_eq!(cleaned[0], "d/dt(central) = 0 + 0");
    assert_eq!(forcings.len(), 2);
    assert!(matches!(
        forcings[0].kind,
        crate::pk::absorption::InputRateKind::Transit
    ));
    assert!(matches!(
        forcings[1].kind,
        crate::pk::absorption::InputRateKind::InverseGaussian
    ));
    assert!(forcings.iter().all(|f| f.cmt == 0 && f.frac_slot.is_none()));
}

// ── first_order() composition: parallel / mixed (#505) ────────────────────

#[test]
fn extract_first_order_single_input_rate() {
    // first_order(ka) straight into central: one FirstOrder forcing reading `ka`,
    // no fraction (single pathway), rewritten to `0`.
    let states = vec!["central".to_string()];
    let names = vec!["KA".to_string(), "CL".to_string(), "V".to_string()];
    let slots = vec![4, 0, 1];
    let line = "d/dt(central) = first_order(ka=KA) - CL/V*central".to_string();
    let (cleaned, forcings) = extract_input_rate_terms(&[line], &states, &names, &slots).unwrap();
    assert_eq!(cleaned[0], "d/dt(central) = 0 - CL/V*central");
    assert_eq!(forcings.len(), 1);
    assert_eq!(
        forcings[0].kind,
        crate::pk::absorption::InputRateKind::FirstOrder
    );
    assert_eq!(forcings[0].arg_slots, vec![4]); // ka @ 4
    assert_eq!(forcings[0].frac_slot, None);
}

#[test]
fn extract_parallel_dual_first_order() {
    // parallel: FR1*first_order(ka=KA1) + FR2*first_order(ka=KA2) — two FirstOrder
    // forcings on one compartment, each carrying a declared pathway fraction.
    let states = vec!["central".to_string()];
    let names = vec![
        "FR1".to_string(),
        "FR2".to_string(),
        "KA1".to_string(),
        "KA2".to_string(),
    ];
    let slots = vec![2, 3, 4, 5];
    let line = "d/dt(central) = FR1*first_order(ka=KA1) + FR2*first_order(ka=KA2) - CL/V*central"
        .to_string();
    let (_, forcings) = extract_input_rate_terms(&[line], &states, &names, &slots).unwrap();
    assert_eq!(forcings.len(), 2);
    assert!(forcings
        .iter()
        .all(|f| f.kind == crate::pk::absorption::InputRateKind::FirstOrder && f.cmt == 0));
    assert_eq!(forcings[0].frac_slot, Some(2)); // FR1
    assert_eq!(forcings[1].frac_slot, Some(3)); // FR2
    assert_eq!(forcings[0].arg_slots, vec![4]); // KA1
    assert_eq!(forcings[1].arg_slots, vec![5]); // KA2
}

#[test]
fn extract_mixed_first_plus_zero_order() {
    // mixed: FZO1*first_order(ka=KA) + FZO*zero_order(dur=DUR) — a Channel-A
    // first-order term and a Channel-B zero-order term on one compartment, both
    // fractioned. Table order puts zero_order before first_order.
    let states = vec!["central".to_string()];
    let names = vec![
        "FZO1".to_string(),
        "FZO".to_string(),
        "KA".to_string(),
        "DUR".to_string(),
    ];
    let slots = vec![2, 3, 4, 6];
    let line = "d/dt(central) = FZO1*first_order(ka=KA) + FZO*zero_order(dur=DUR) - CL/V*central"
        .to_string();
    let (_, forcings) = extract_input_rate_terms(&[line], &states, &names, &slots).unwrap();
    assert_eq!(forcings.len(), 2);
    // Every forcing carries a fraction and feeds central.
    assert!(forcings.iter().all(|f| f.frac_slot.is_some() && f.cmt == 0));
    let zo = forcings
        .iter()
        .find(|f| f.kind == crate::pk::absorption::InputRateKind::ZeroOrder)
        .expect("a zero-order forcing");
    let fo = forcings
        .iter()
        .find(|f| f.kind == crate::pk::absorption::InputRateKind::FirstOrder)
        .expect("a first-order forcing");
    assert_eq!(zo.frac_slot, Some(3)); // FZO
    assert_eq!(fo.frac_slot, Some(2)); // FZO1
}

#[test]
fn extract_first_order_with_per_route_lag() {
    // first_order(ka=KA, lag=LAG): the optional `lag` arg is captured into
    // `lag_slot`, leaving `arg_slots` = [ka] (lag is not a required intrinsic arg).
    let states = vec!["central".to_string()];
    let names = vec!["KA".to_string(), "LAG".to_string()];
    let slots = vec![4, 7];
    let line = "d/dt(central) = first_order(ka=KA, lag=LAG) - CL/V*central".to_string();
    let (_, forcings) = extract_input_rate_terms(&[line], &states, &names, &slots).unwrap();
    assert_eq!(forcings.len(), 1);
    assert_eq!(forcings[0].arg_slots, vec![4]); // ka @ 4 (lag NOT in arg_slots)
    assert_eq!(forcings[0].lag_slot, Some(7)); // LAG @ 7
    assert_eq!(forcings[0].frac_slot, None);
}

#[test]
fn extract_parallel_distinct_per_route_lags() {
    // Each parallel route may carry its OWN lag: FR1*first_order(ka=KA1, lag=L1)
    // + FR2*first_order(ka=KA2, lag=L2) → two forcings with distinct lag_slots.
    let states = vec!["central".to_string()];
    let names = vec![
        "FR1".to_string(),
        "FR2".to_string(),
        "KA1".to_string(),
        "KA2".to_string(),
        "L1".to_string(),
        "L2".to_string(),
    ];
    let slots = vec![2, 3, 4, 5, 6, 7];
    let line = "d/dt(central) = FR1*first_order(ka=KA1, lag=L1) \
                    + FR2*first_order(ka=KA2, lag=L2) - CL/V*central"
        .to_string();
    let (_, forcings) = extract_input_rate_terms(&[line], &states, &names, &slots).unwrap();
    assert_eq!(forcings.len(), 2);
    assert_eq!(forcings[0].lag_slot, Some(6)); // L1
    assert_eq!(forcings[1].lag_slot, Some(7)); // L2
    assert_eq!(forcings[0].frac_slot, Some(2)); // FR1
    assert_eq!(forcings[1].frac_slot, Some(3)); // FR2
}

#[test]
fn extract_zero_order_with_per_route_lag() {
    // The `lag` arg is universal — a zero-order route takes it too (its window
    // start/end shift by the route lag).
    let states = vec!["central".to_string()];
    let names = vec!["DUR".to_string(), "LAG".to_string()];
    let slots = vec![4, 7];
    let line = "d/dt(central) = zero_order(dur=DUR, lag=LAG) - CL/V*central".to_string();
    let (_, forcings) = extract_input_rate_terms(&[line], &states, &names, &slots).unwrap();
    assert_eq!(forcings.len(), 1);
    assert_eq!(
        forcings[0].kind,
        crate::pk::absorption::InputRateKind::ZeroOrder
    );
    assert_eq!(forcings[0].arg_slots, vec![4]); // dur @ 4
    assert_eq!(forcings[0].lag_slot, Some(7)); // LAG @ 7
}

#[test]
fn extract_rejects_duplicate_lag_arg() {
    let states = vec!["central".to_string()];
    let names = vec!["KA".to_string(), "L1".to_string(), "L2".to_string()];
    let slots = vec![4, 6, 7];
    let line = "d/dt(central) = first_order(ka=KA, lag=L1, lag=L2) - CL/V*central".to_string();
    let err = extract_input_rate_terms(&[line], &states, &names, &slots).unwrap_err();
    assert!(
        err.contains("duplicate") && err.contains("lag"),
        "got: {err}"
    );
}

#[test]
fn extract_unknown_arg_message_mentions_lag() {
    // An unknown arg's error names the intrinsic args AND the universal `lag`, so
    // the user learns `lag` is accepted.
    let states = vec!["central".to_string()];
    let names = vec!["KA".to_string(), "X".to_string()];
    let slots = vec![4, 6];
    let line = "d/dt(central) = first_order(ka=KA, foo=X) - CL/V*central".to_string();
    let err = extract_input_rate_terms(&[line], &states, &names, &slots).unwrap_err();
    assert!(
        err.contains("no argument `foo`") && err.contains("lag"),
        "got: {err}"
    );
}

#[test]
fn extract_first_order_rejects_bad_specs() {
    let states = vec!["central".to_string()];
    let names = vec!["KA".to_string(), "FR".to_string()];
    let slots = vec![4, 2];
    let go = |line: &str| extract_input_rate_terms(&[line.to_string()], &states, &names, &slots);
    // Unknown / wrong argument name.
    assert!(go("d/dt(central) = first_order(mtt=KA)")
        .unwrap_err()
        .contains("no argument `mtt`"));
    // Undeclared parameter as the argument.
    assert!(go("d/dt(central) = first_order(ka=ZZZ)")
        .unwrap_err()
        .contains("not a declared individual parameter"));
    // A non-fraction scaling (division) is rejected by the shared prefix parser.
    assert!(go("d/dt(central) = first_order(ka=KA)/V").is_err());
    // Positive control: a bare additive first_order() parses.
    assert!(go("d/dt(central) = first_order(ka=KA) - CL/V*central").is_ok());
}

#[test]
fn pk_param_parser_is_strict_about_duplicates_and_malformed_pairs() {
    // The analytical `pk NAME(...)` parser now shares the strict
    // `parse_role_pairs` helper with `ode_template` (Ron #363, finding 2). It
    // previously silently dropped malformed pairs and let a duplicate role
    // last-win; both are now hard errors, single-sourced so the two paths can't
    // drift in strictness again.

    // Duplicate role → rejected (was: silent last-win, here `cl=V` would have
    // shadowed `cl=CL` with no warning).
    let dup = vec!["pk one_cpt_iv(cl=CL, cl=V)".to_string()];
    let err = parse_structural_model(&dup).unwrap_err();
    assert!(
        err.contains("duplicate parameter") && err.contains("cl"),
        "duplicate pk param should error: {err}"
    );

    // Malformed pairs (no `=`, double `=`, empty side) → rejected (was: silently
    // dropped, then surfaced only as a confusing missing-required error later).
    for bad in [
        "pk one_cpt_iv(cl, v=V)",
        "pk one_cpt_iv(cl=CL=X, v=V)",
        "pk one_cpt_iv(=CL, v=V)",
        "pk one_cpt_iv(cl=, v=V)",
    ] {
        let lines = vec![bad.to_string()];
        let err = parse_structural_model(&lines).unwrap_err();
        assert!(
            err.contains("malformed parameter"),
            "`{bad}` should be a malformed-parameter error, got: {err}"
        );
    }

    // A well-formed list (incl. a tolerated trailing comma) still parses.
    let ok = vec!["pk one_cpt_iv(cl=CL, v=V, )".to_string()];
    let (model, params) = parse_structural_model(&ok).expect("well-formed pk must parse");
    assert_eq!(model, PkModel::OneCptIv);
    assert_eq!(params.get("cl").map(String::as_str), Some("CL"));
    assert_eq!(params.get("v").map(String::as_str), Some("V"));
}

#[test]
fn unknown_pk_model_name_errors() {
    // A model name that is neither a valid name/alias (`PkModel::from_name`)
    // nor a retired #176 spelling falls through to the generic "Unknown PK
    // model" error — the `None`/`other` arm of the name resolution.
    let lines = vec!["pk four_cpt_iv(cl=CL, v=V)".to_string()];
    let err = parse_structural_model(&lines).unwrap_err();
    assert!(
        err.contains("Unknown PK model") && err.contains("four_cpt_iv"),
        "got: {err}"
    );
}

// Issue #176 retired the split `*_iv_bolus` / `*_infusion` model names
// in favour of a single `*_iv` per compartment count. The parser must
// reject the old names with a migration message rather than silently
// accept or emit a generic "unknown model" error.
#[test]
fn test_retired_iv_bolus_and_infusion_names_emit_migration_error() {
    let retired = [
        ("one_cpt_iv_bolus", "one"),
        ("one_compartment_iv_bolus", "one"),
        ("one_cpt_infusion", "one"),
        ("two_cpt_iv_bolus", "two"),
        ("two_compartment_infusion", "two"),
        ("three_cpt_iv_bolus", "three"),
        ("three_cpt_infusion", "three"),
    ];
    for (name, n) in retired {
        let lines = vec![format!("pk {}(cl=CL, v=V)", name)];
        let err = parse_structural_model(&lines)
            .expect_err(&format!("expected retired name `{name}` to error"));
        assert!(
            err.contains("#176") && err.contains(&format!("{n}_cpt_iv")),
            "missing migration hint for `{name}`: {err}"
        );
    }
}

#[test]
fn test_unified_iv_names_parse_to_iv_variant() {
    // The new spelling must compile to the unified IV variant.
    let cases = [
        ("one_cpt_iv", PkModel::OneCptIv),
        ("one_compartment_iv", PkModel::OneCptIv),
        ("two_cpt_iv", PkModel::TwoCptIv),
        ("three_cpt_iv", PkModel::ThreeCptIv),
    ];
    for (name, expected) in cases {
        // `parse_structural_model` only resolves the model name → variant;
        // required-parameter completeness is checked later in `parse_full_model`.
        let lines = vec![format!("pk {}(cl=CL, v=V, q=Q, v2=V2, q2=Q2, v3=V3)", name)];
        let (pk_model, _) = parse_structural_model(&lines)
            .unwrap_or_else(|e| panic!("`{name}` failed to parse: {e}"));
        assert_eq!(pk_model, expected, "wrong variant for `{name}`");
    }
}

#[test]
fn canonical_name_round_trips_through_parser() {
    // Every `PkModel::canonical_name()` must be a model name the parser
    // accepts and maps back to the same variant — guards `canonical_name`
    // (types.rs) against drifting from this name table.
    for model in [
        PkModel::OneCptIv,
        PkModel::OneCptOral,
        PkModel::TwoCptIv,
        PkModel::TwoCptOral,
        PkModel::ThreeCptIv,
        PkModel::ThreeCptOral,
    ] {
        let lines = vec![format!("pk {}(cl=CL, v=V)", model.canonical_name())];
        let (parsed, _) = parse_structural_model(&lines).unwrap_or_else(|e| {
            panic!(
                "canonical_name `{}` did not parse: {e}",
                model.canonical_name()
            )
        });
        assert_eq!(
            parsed,
            model,
            "canonical_name `{}` did not round-trip",
            model.canonical_name()
        );
    }
}

/// A complete analytical model wrapping `pk_line`, with every commonly
/// referenced individual parameter defined so `build_pk_param_fn`'s per-key
/// value validation passes and only the structural-mapping checks (required /
/// unused, run in `parse_full_model`) are exercised. Surplus declared params
/// just produce the usual declared-but-unused warnings, which don't affect
/// whether parsing succeeds.
fn structural_model_with(pk_line: &str) -> String {
    format!(
        "
[parameters]
  theta TVX(1.0, 0.001, 100.0)
  omega ETA ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL   = TVX * exp(ETA)
  V    = TVX
  V1   = TVX
  Q    = TVX
  Q2   = TVX
  V2   = TVX
  Q3   = TVX
  V3   = TVX
  KA   = TVX
  F    = TVX
  TLAG = TVX

[structural_model]
  {pk_line}

[error_model]
  DV ~ proportional(EPS)
"
    )
}

#[test]
fn test_missing_required_pk_param_errors() {
    // Omitting a required structural parameter must be a hard parse error,
    // not a silent default-to-0.0 broken fit (issue #309).

    // Single missing param: exact message (matches the #309 acceptance text).
    let err = expect_parse_err(&structural_model_with("pk one_cpt_oral(cl=CL, v=V)"));
    assert_eq!(
        err,
        "[structural_model] one_cpt_oral requires `ka`, which is not mapped. \
             Add it, e.g. `ka=KA`."
    );

    // Multiple missing params: plural grammar, names all of them.
    let err = expect_parse_err(&structural_model_with("pk two_cpt_iv(cl=CL, v1=V1)"));
    assert!(err.contains("`q`"), "should name q: {err}");
    assert!(err.contains("`v2`"), "should name v2: {err}");
    assert!(
        err.contains("are not mapped") && err.contains("Add them"),
        "plural grammar expected: {err}"
    );

    // `ka` omitted on a three-cpt oral model (every other slot present).
    let err = expect_parse_err(&structural_model_with(
        "pk three_cpt_oral(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3)",
    ));
    assert!(err.contains("`ka`"), "should name ka: {err}");

    // Supplying an *optional* parameter (`f`) must not mask a *missing*
    // required one: `ka` is still reported even though `f` is present.
    let err = expect_parse_err(&structural_model_with("pk one_cpt_oral(cl=CL, v=V, f=F)"));
    assert!(err.contains("`ka`"), "should still name ka: {err}");
}

#[test]
fn test_required_params_present_parses_ok() {
    // A model that maps every required slot parses without a structural error
    // — including via the `v`/`v1` and `q`/`q2` aliases (canonical-slot check)
    // and with optional `f`/`lagtime`/`alag` present. The required-
    // completeness check runs after `build_pk_param_fn` (issue #309).
    let ok_lines = [
        "pk one_cpt_iv(cl=CL, v=V)",
        "pk one_cpt_oral(cl=CL, v=V, ka=KA)",
        // optional f + lagtime present alongside the required params:
        "pk one_cpt_oral(cl=CL, v=V, ka=KA, f=F, lagtime=TLAG)",
        // `alag` alias for lagtime:
        "pk one_cpt_oral(cl=CL, v=V, ka=KA, alag=TLAG)",
        "pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)",
        // `q2` alias satisfies the `q` slot; `v` alias satisfies `v1`:
        "pk two_cpt_iv(cl=CL, v=V1, q2=Q, v2=V2)",
        "pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)",
        "pk three_cpt_iv(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3)",
        "pk three_cpt_oral(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3, ka=KA)",
    ];
    for line in ok_lines {
        assert!(
            super::parse_full_model(&structural_model_with(line)).is_ok(),
            "`{line}` should parse without a structural error"
        );
    }
}

#[test]
fn test_unused_pk_param_warns() {
    // PK parameters mapped but not used by the chosen model parse Ok but warn
    // (#309): on an IV model `ka` (no absorption) is flagged. `f`
    // (bioavailability — applied to IV bolus/infusion since #327) and
    // `lagtime` (applied to every dose) are both used, so neither is flagged.
    let model_str = "
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta TVKA(1.0, 0.01, 100.0)
  theta TVF(1.0, 0.01, 10.0)
  theta TVLAG(0.5)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  KA  = TVKA
  F   = TVF
  LAG = TVLAG

[structural_model]
  pk one_cpt_iv(cl=CL, v=V, ka=KA, f=F, lagtime=LAG)

[error_model]
  DV ~ proportional(EPS)
";
    let parsed =
        super::parse_full_model(model_str).expect("unused params are a warning, not a parse error");
    let warn = parsed
        .model
        .parse_warnings
        .iter()
        .find(|w| w.contains("does not use"))
        .unwrap_or_else(|| {
            panic!(
                "expected an unused-param warning, got: {:?}",
                parsed.model.parse_warnings
            )
        });
    assert!(
        warn.contains("`ka`"),
        "ka should be flagged unused on IV: {warn}"
    );
    assert!(
        !warn.contains("`f`"),
        "f is applied to IV bolus/infusion (#327) and must not be flagged: {warn}"
    );
    assert!(
        !warn.contains("`lagtime`"),
        "lagtime is applied to every dose and must not be flagged: {warn}"
    );
}

#[test]
fn test_f_lagtime_warning_matrix() {
    // Pins the f/lagtime warning matrix (#309). Two distinct checks fire:
    //  - "does not use" (`consumes_pk_slot`): a param mapped in `pk(...)` but
    //    not consumed — `f` and `lagtime` are consumed by every model (#327),
    //    so neither warns; an unused structural slot (e.g. `ka` on IV) does;
    //  - "computed but never used": a param declared in [individual_parameters]
    //    but never mapped or referenced anywhere.
    // KA/F/LAG are literals so the helper declares no surplus thetas (which
    // would otherwise add unrelated unused-theta warnings).
    let model = |indiv: &str, pk: &str| -> String {
        format!(
            "
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
{indiv}

[structural_model]
  {pk}

[error_model]
  DV ~ proportional(EPS)
"
        )
    };
    let warns = |indiv: &str, pk: &str| -> Vec<String> {
        super::parse_full_model(&model(indiv, pk))
            .unwrap_or_else(|e| panic!("`{pk}` should parse: {e}"))
            .model
            .parse_warnings
    };
    let has = |ws: &[String], needle: &str| ws.iter().any(|w| w.contains(needle));
    let clv = "  CL = TVCL * exp(ETA_CL)\n  V = TVV";

    // (1) IV + `f` mapped → NOT flagged. F scales the bioavailable amount
    // on every route — IV bolus and infusion included (#327) — so it is
    // used, not inert, on IV models.
    let ws = warns(
        &format!("{clv}\n  F = 0.8"),
        "pk one_cpt_iv(cl=CL, v=V, f=F)",
    );
    assert!(
        !has(&ws, "does not use"),
        "IV + mapped f must not warn now that F applies to IV doses (#327): {ws:?}"
    );

    // (2) IV + `lagtime` mapped → NOT flagged (every model applies lagtime).
    let ws = warns(
        &format!("{clv}\n  LAG = 0.5"),
        "pk one_cpt_iv(cl=CL, v=V, lagtime=LAG)",
    );
    assert!(
        !has(&ws, "does not use"),
        "IV + lagtime must not warn: {ws:?}"
    );

    // (3) Oral + `f` mapped (defined) → no warning.
    let ws = warns(
        &format!("{clv}\n  KA = 1.0\n  F = 0.8"),
        "pk one_cpt_oral(cl=CL, v=V, ka=KA, f=F)",
    );
    assert!(
        !has(&ws, "does not use") && !has(&ws, "computed but never used"),
        "oral + used f must not warn: {ws:?}"
    );

    // (4) Oral + `f` mapped to an UNDEFINED variable → parse error (#308).
    assert!(
        super::parse_full_model(&model(
            "  CL = TVCL * exp(ETA_CL)\n  V = TVV\n  KA = 1.0",
            "pk one_cpt_oral(cl=CL, v=V, ka=KA, f=FNOPE)",
        ))
        .is_err(),
        "oral + undefined `f` reference must error"
    );

    // (5) Oral + `F` declared but NOT mapped → "computed but never used `F`".
    let ws = warns(
        &format!("{clv}\n  KA = 1.0\n  F = 0.8"),
        "pk one_cpt_oral(cl=CL, v=V, ka=KA)",
    );
    assert!(
        has(&ws, "computed but never used") && has(&ws, "`F`"),
        "oral + declared-not-mapped F should warn dead: {ws:?}"
    );

    // (6) IV + `F` declared but NOT mapped → "computed but never used `F`".
    let ws = warns(&format!("{clv}\n  F = 0.8"), "pk one_cpt_iv(cl=CL, v=V)");
    assert!(
        has(&ws, "computed but never used") && has(&ws, "`F`"),
        "IV + declared-not-mapped F should warn dead: {ws:?}"
    );
}

#[test]
fn test_param_used_only_in_derived_not_flagged_dead() {
    // An individual parameter referenced only in [derived] — not mapped into
    // pk(...) and not used by another parameter — must NOT be flagged
    // "computed but never used": the census tokenizes every block (#309).
    let model = |derived: &str| -> String {
        format!(
            "
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  KA  = 1.0
  KEL = CL / V

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
{derived}
[error_model]
  DV ~ proportional(EPS)
"
        )
    };
    // KEL is referenced in [derived] → used → no dead warning.
    let p = super::parse_full_model(&model("\n[derived]\n  RATE = KEL * 2\n")).expect("parse ok");
    assert!(
        !p.model
            .parse_warnings
            .iter()
            .any(|w| w.contains("computed but never used")),
        "KEL is used in [derived] and must not be flagged dead: {:?}",
        p.model.parse_warnings
    );
    // Negative control: with no [derived] reference, KEL IS dead.
    let p2 = super::parse_full_model(&model("")).expect("parse ok");
    assert!(
        p2.model
            .parse_warnings
            .iter()
            .any(|w| w.contains("computed but never used") && w.contains("`KEL`")),
        "without the [derived] use KEL must be flagged dead: {:?}",
        p2.model.parse_warnings
    );
}

/// #486 regression: the synthetic `__ferx_pktime_*` parameter a direct
/// `pk(...=TIME)` mapping desugars into must be exempt from the "computed but never
/// used" census (like the `__ferx_ro_*` readout synthetics). The census tokenizes the
/// raw block text, which still reads `q=TIME` (never `q=__ferx_pktime_q`), so the
/// synthetic name looks unreferenced — only the `is_synthetic_readout_param` prefix
/// match keeps it from being flagged. If that exemption is ever dropped, every direct
/// `pk(...=TIME)` model would tell users to remove a load-bearing parameter.
#[test]
fn direct_pk_time_synth_param_not_flagged_unused() {
    let src = "
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV1(10.0, 0.1, 1000.0)
  theta TVV2(20.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1
  V2 = TVV2

[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=TIME, v2=V2)

[error_model]
  DV ~ proportional(EPS)
";
    let p = super::parse_full_model(src).expect("parse ok");
    assert!(
        !p.model
            .parse_warnings
            .iter()
            .any(|w| w.contains("computed but never used")),
        "direct pk(q=TIME) must not flag the synthetic __ferx_pktime_q as dead: {:?}",
        p.model.parse_warnings
    );
}

/// #486: a user parameter using the reserved `__ferx_pktime_` prefix is rejected, and
/// the message names the *correct* prefix and feature (`pk(...=TIME)`), not the Form-C
/// readout prefix — even though the shared guard also matches the readout prefix.
#[test]
fn reserved_pktime_prefix_rejected_with_correct_message() {
    let src = "
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  __ferx_pktime_cl = TVCL * exp(ETA_CL)

[structural_model]
  pk one_cpt_iv(cl=__ferx_pktime_cl, v=10.0)

[error_model]
  DV ~ proportional(EPS)
";
    let err = super::parse_full_model(src)
        .err()
        .expect("reserved prefix must be rejected");
    assert!(
        err.contains(PKTIME_SYNTH_PREFIX),
        "message must name the pktime prefix: {err}"
    );
    assert!(
        err.contains("pk(...=TIME)"),
        "message must name the pk(...=TIME) feature, not Form-C readout: {err}"
    );
}

#[test]
fn test_ode_dead_indiv_param_warns() {
    // #315: an ODE [individual_parameters] entry never referenced in the
    // [odes] RHS (and not engine-applied f/lagtime) is routed to a free slot,
    // computed every evaluation, but never read — silently inert. The #310
    // "computed but never used" census skipped ODE models; it now covers them.
    let content = r#"
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta TVKE(0.5, 0.001, 10.0)
  omega ETA_CL ~ 0.1
  omega ETA_KE ~ 0.09
  sigma EPS ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KE = TVKE * exp(ETA_KE)   # never referenced in [odes]

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -(CL/V) * central

[error_model]
  DV ~ proportional(EPS)
"#;
    let ws = super::parse_full_model(content)
        .expect("a dead ODE param is a warning, not an error")
        .model
        .parse_warnings;
    let dead: Vec<&String> = ws
        .iter()
        .filter(|w| w.contains("computed but never used"))
        .collect();
    assert_eq!(
        dead.len(),
        1,
        "exactly one dead-param warning expected, got: {ws:?}"
    );
    let w = dead[0];
    assert!(w.contains("`KE`"), "warning must name KE: {w}");
    // The params actually used in the RHS must not be flagged.
    assert!(
        !w.contains("`CL`") && !w.contains("`V`"),
        "used params must not be flagged dead: {w}"
    );
    // ODE-flavored guidance points at [odes], not the analytical pk(...) map.
    assert!(
        w.contains("[odes]") && !w.contains("pk(...)"),
        "ODE warning should reference [odes], not pk(...): {w}"
    );
}

#[test]
fn test_ode_engine_applied_f_lagtime_not_flagged_dead() {
    // #315 carve-out: F and lagtime on an ODE model are routed to the
    // engine-reserved PK_IDX_F / PK_IDX_LAGTIME slots (`ode_param_slots`) and
    // applied to the dose by the engine without ever appearing in the [odes]
    // RHS, so their textual absence must NOT flag them dead. Mirrors
    // examples/bioavailability_ode.ferx and examples/warfarin_ode_lagtime.ferx.
    let content = r#"
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta TVKA(1.0, 0.001, 10.0)
  theta THETA_F(0.0, -10.0, 10.0)
  theta TVLAG(0.5, 0.001, 10.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL      = TVCL * exp(ETA_CL)
  V       = TVV
  KA      = TVKA
  F       = inv_logit(THETA_F)
  LAGTIME = TVLAG

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central

[error_model]
  DV ~ proportional(EPS)
"#;
    let ws = super::parse_full_model(content)
        .expect("engine-applied F/lagtime on an ODE model must parse")
        .model
        .parse_warnings;
    assert!(
        !ws.iter().any(|w| w.contains("computed but never used")),
        "engine-applied F/LAGTIME on an ODE model must not be flagged dead: {ws:?}"
    );
}

// ── #993: a dose attribute applied by the engine AND read by the model ───────
//
// The mirror image of the carve-out above. `F` / `LAGTIME` / `ALAG` (and the
// compartment-indexed `F{n}` / `ALAG{n}` / `LAGTIME{n}`) are load-bearing while
// textually absent, so the dead-param census exempts them — but a model that
// *does* read one on the prediction path gets it applied twice, silently: once by
// the engine at the dose event, once where it is read. Measured on the engine at
// exactly `F` on every prediction, and `exp(LAGTIME·ke)` for lag.

/// Build a 1-compartment ODE model whose only individual parameter besides CL/V is
/// `name`, read in the `[odes]` RHS as the elimination rate constant. This is the
/// exact shape of the #993 repro: rename `name` and the fit moves by a level block.
fn dose_attr_rhs_model(name: &str) -> String {
    format!(
        "
[parameters]
  theta TVCL(5.0, 0.0, 1e15)
  theta TVV(50.0, 0.0, 1e15)
  omega ETA_CL ~ 0.09
  sigma EPS1 ~ 0.1 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  {name} = 0.1

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -{name} * central

[error_model]
  DV ~ proportional(EPS1)
"
    )
}

#[test]
fn dose_attr_read_in_odes_rhs_is_rejected() {
    // Every spelling the engine applies to *every* dose: the two bare reserved-slot
    // names plus the `alag` alias, and the compartment-indexed forms. Each must be
    // a hard parse error naming the parameter, both readings, and the rename.
    for (name, noun) in [
        ("F", "bioavailability"),
        ("LAGTIME", "absorption lag"),
        ("ALAG", "absorption lag"),
        ("F1", "bioavailability"),
        ("ALAG1", "absorption lag"),
        ("LAGTIME1", "absorption lag"),
    ] {
        let err = expect_parse_err(&dose_attr_rhs_model(name));
        assert!(
            err.contains("[odes]:") && err.contains(&format!("`{name}`")),
            "must name the block and the parameter, got: {err}"
        );
        assert!(
            err.contains(noun),
            "must say what `{name}` means to the engine ({noun}), got: {err}"
        );
        // The remediation is the half a user acts on, and the sentinel
        // `parse_error_to_diagnostic` keys `E_DOSE_ATTR_DOUBLE_USE` off.
        assert!(
            err.contains("reserved dose-attribute name"),
            "must offer the rename, got: {err}"
        );
    }
}

#[test]
fn dose_attr_read_in_odes_rhs_is_matched_case_insensitively() {
    // `build_ode_spec` aliases each parameter's case variants onto one var slot, so
    // a lower-case `f` in the RHS reads the very same value as `F` and must
    // diagnose identically. Guards against a check that compares raw spellings.
    let src = dose_attr_rhs_model("F").replace("-F * central", "-f * central");
    let err = expect_parse_err(&src);
    assert!(
        err.contains("reserved dose-attribute name"),
        "a lower-case read of `F` is the same double use: {err}"
    );
}

/// Build a 1-compartment ODE model that declares dose attribute `name` and reads it
/// in an `init(...)` seed — never in the RHS. Companion to [`dose_attr_rhs_model`],
/// which is the rejected shape; this is the accepted one (#1046).
fn dose_attr_init_model(name: &str) -> String {
    format!(
        "
[parameters]
  theta TVCL(5.0, 0.0, 1e15)
  theta TVV(50.0, 0.0, 1e15)
  theta TVA(0.5, 0.0, 1.0)
  omega ETA_CL ~ 0.09
  sigma EPS1 ~ 0.1 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  {name} = TVA

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  init(central) = {name} * 100.0
  d/dt(central) = -(CL/V) * central

[error_model]
  DV ~ proportional(EPS1)
"
    )
}

#[test]
fn dose_attr_read_in_odes_init_is_accepted() {
    // The scope floor of the #993 rule, and the one surface it must NOT claim
    // (#1046). An `init(...)` seed is not a dose: `OdeSpec::initial_state` writes the
    // raw expression value into the state vector, and `F`/lag are resolved through
    // `DoseAttrMap` at dose events only — a path the seed never takes. So
    // `init(central) = F * 100` (the bioavailable residue of a pre-study 100 mg dose)
    // applies `F` exactly once, to a quantity the engine never scales.
    //
    // Anchored on NONMEM 7.6.0, which agrees: `A_0(1) = F1*100` with `F1 = 0.5` seeds
    // 50, not 25, and its table is byte-identical to the twin seeding from an ordinary
    // parameter of the same value (`nonmem_anchor/odes_init_dose_attr_f_{A,B}.ctl`;
    // same for lag). Rejecting this made a correct model unwritable and advised the
    // wrong repair — renaming the parameter whose meaning *is* bioavailability.
    //
    // Contrast the RHS, which IS a doubling and stays rejected:
    // `dose_attr_read_in_odes_rhs_is_rejected`.
    for name in ["F", "LAGTIME", "ALAG", "F1", "ALAG1", "LAGTIME1"] {
        let src = dose_attr_init_model(name);
        if let Err(e) = parse_full_model(&src) {
            panic!("`init(central) = {name} * 100` applies `{name}` once: {e}");
        }
    }
}

#[test]
fn an_init_read_counts_as_a_use_for_the_never_used_census() {
    // Companion to the acceptance test above, and deliberately *not* folded into it:
    // asserting "no never-used warning" for `F`/`LAGTIME`/`F1`/… cannot fail, because
    // the census exempts every dose-attribute name before it looks at block text
    // (`RESERVED_PK_SLOTS` for the bare names, `DoseAttr::from_indexed_name` for the
    // indexed ones). Measured: with the `init(...)` line deleted entirely, all six
    // still produce zero "never used" warnings.
    //
    // So the census claim has to be made with a name the exemption does not cover. An
    // ordinary parameter read *only* from the init is that name — and it is the shape
    // that would regress if the seed ever stopped being counted as a use.
    let src = "
[parameters]
  theta TVCL(5.0, 0.0, 1e15)
  theta TVV(50.0, 0.0, 1e15)
  theta TVA(0.5, 0.0, 1.0)
  omega ETA_CL ~ 0.09
  sigma EPS1 ~ 0.1 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  FSEED = TVA

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  init(central) = FSEED * 100.0
  d/dt(central) = -(CL/V) * central

[error_model]
  DV ~ proportional(EPS1)
";
    let never_used = |s: &str| -> Vec<String> {
        parse_full_model(s)
            .expect("parses")
            .model
            .parse_warnings
            .iter()
            .filter(|w| w.contains("never used"))
            .cloned()
            .collect()
    };
    assert!(
        never_used(src).is_empty(),
        "`FSEED` is read by the init seed, so it is used: {:?}",
        never_used(src)
    );
    // Non-vacuity: delete the init line and the very same parameter *is* reported —
    // so the assertion above is a statement about the init read, not about the census
    // being silent in general.
    let without = src.replace("  init(central) = FSEED * 100.0\n", "");
    assert_ne!(without, src, "the init line must actually be removed");
    assert!(
        never_used(&without).iter().any(|w| w.contains("`FSEED`")),
        "with the init read gone `FSEED` must be reported unused: {:?}",
        never_used(&without)
    );
}

#[test]
fn coded_rate_param_read_in_odes_init_is_not_recorded() {
    // The `D{n}`/`R{n}` half of #1046, and the reason the fix had to drop the init
    // walk from `rhs_reads` rather than skip the check: that same set drives
    // `mark_prediction_path_read`, whose marks `check_modeled_dose_rates` turns into
    // a data-gated error once a `RATE=-2` dose lands on the compartment. An init
    // amount consults no dose attribute at all — modeled duration included — so a
    // `D1` read there must record nothing, or a correct model would be rejected as
    // soon as its dataset happened to carry a coded RATE.
    use crate::types::DoseAttr;
    let parsed = parse_full_model(&dose_attr_init_model("D1")).expect("`D1` in an init parses");
    assert_eq!(
        parsed
            .model
            .active_dose_attr_map()
            .prediction_path_read(DoseAttr::Duration, 1),
        None,
        "an init-only `D1` read is not a prediction-path read"
    );

    // Control: the same parameter read in the RHS *is* recorded, so the assertion
    // above cannot pass for a map that simply never marks anything.
    let rhs = dose_attr_init_model("D1").replace(
        "d/dt(central) = -(CL/V) * central",
        "d/dt(central) = -(CL/V) * D1 * central",
    );
    assert_eq!(
        parse_full_model(&rhs)
            .expect("`D1` in the RHS parses — it is data-gated, not rejected")
            .model
            .active_dose_attr_map()
            .prediction_path_read(DoseAttr::Duration, 1),
        Some("D1"),
        "a `D1` read by the [odes] RHS must still be recorded for the data check"
    );
}

#[test]
fn dose_attr_read_in_both_odes_init_and_rhs_is_still_rejected() {
    // Overshoot guard for #1046. Accepting the init read must not disarm the RHS
    // walk: a model that seeds from `F` *and* folds `F` into the flux still applies
    // it twice on the dose-driven term, which is exactly the #993 defect. Pins that
    // the fix dropped the init walk only.
    let src = dose_attr_init_model("F").replace(
        "d/dt(central) = -(CL/V) * central",
        "d/dt(central) = -(CL/V) * F * central",
    );
    let err = expect_parse_err(&src);
    assert!(
        err.contains("[odes]:") && err.contains("reserved dose-attribute name"),
        "the RHS read is still a double use even when an init also reads it: {err}"
    );
    // And the message must say *which* read to remove. This is the exact model where
    // the un-narrowed wording misled: "remove it from [odes]" invites deleting the
    // `init(...)` line, which is legal (#1046) and does not clear the error. The
    // sibling data-gated half already says "[odes] RHS or [scaling]"
    // (`api/validation.rs`), so the two halves of `E_DOSE_ATTR_DOUBLE_USE` agree.
    assert!(
        err.contains("read in the [odes] RHS") && err.contains("remove it from the [odes] RHS"),
        "the remedy must name the RHS, not the whole block: {err}"
    );
}

#[test]
fn out_of_range_indexed_dose_attr_reports_the_compartment_error() {
    // Scope boundary on the compartment index. `F5` on a one-state model is a
    // *different* defect, and `dose_attr_map`'s build reports it precisely. This
    // check runs first, so it must decline out-of-range indices — otherwise the
    // user is told to rename `F5` when the real problem is that compartment 5 does
    // not exist.
    let err = expect_parse_err(&dose_attr_rhs_model("F5"));
    assert!(
        err.contains("only 1 compartment"),
        "the compartment-index error must win over the double-use rename advice: {err}"
    );
    assert!(
        !err.contains("reserved dose-attribute name"),
        "out-of-range `F5` is not the #993 diagnostic: {err}"
    );
}

#[test]
fn ordinary_names_read_in_odes_rhs_are_accepted() {
    // The other half of the contract, and the one that would break real models if
    // the predicate over-matched. `N`/`MTT`/`MAT`/`CV2` hold canonical PK slots but
    // are only ever read through an explicit `transit(n=…)`/`igd(…)` arg mapping;
    // `S{n}` is not routed at all; `D{n}`/`R{n}` are dose attributes but inert
    // unless the *data* codes RATE, so they are not a parse error (see
    // `tests/modeled_rate.rs` for the data-gated half).
    for name in [
        "KA", "Q", "V2", "Q3", "V3", "N", "MTT", "MAT", "CV2", "S1", "S2", "D1", "R1", "KEL", "ZQ",
    ] {
        assert!(
            parse_full_model(&dose_attr_rhs_model(name)).is_ok(),
            "`{name}` in the [odes] RHS is not a dose-attribute double use and must parse"
        );
    }
}

/// A 1-compartment ODE model with a declarative reactive controller whose
/// `observe` signal is `signal_expr`. `[adaptive_dosing] observe` is compiled by
/// the same `build_y_output_fn` a Form-C `y` readout uses, so it reaches
/// individual parameters — including the reserved dose-attribute names.
fn adaptive_observe_model(extra_param: &str, signal_expr: &str) -> String {
    format!(
        "
[parameters]
  theta TVCL(5.0, 0.0, 1e15)
  theta TVV(50.0, 0.0, 1e15)
  theta TVX(0.5, 0.0, 1.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.1 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  {extra_param} = TVX

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = -(CL/V) * central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP)

[adaptive_dosing]
  observe = {signal_expr}
  at = [24, 48]
  start_dose = 100
  route = bolus(cmt = 1)
  dose_bounds = [0, 400]
  when signal < 10 : increase 25%
"
    )
}

#[test]
fn dose_attr_read_in_adaptive_observe_is_rejected() {
    // The controller's signal is a prediction path too, and the worst one to get
    // wrong: `observe` does not merely report a number, it is what the `when` rules
    // compare against, so a double-applied `F` biases the titration decision and
    // every dose the controller then emits.
    for (name, noun) in [
        ("F", "bioavailability"),
        ("LAGTIME", "absorption lag"),
        ("F1", "bioavailability"),
    ] {
        let err = expect_parse_err(&adaptive_observe_model(
            name,
            &format!("central / (V * {name})"),
        ));
        assert!(
            err.contains("[adaptive_dosing]:") && err.contains(&format!("`{name}`")),
            "must name the block and the parameter, got: {err}"
        );
        assert!(
            err.contains(noun) && err.contains("reserved dose-attribute name"),
            "must say what `{name}` means and offer the rename, got: {err}"
        );
    }
}

#[test]
fn ordinary_name_read_in_adaptive_observe_is_accepted() {
    // Over-match guard: an ordinary parameter in the controller signal is the
    // normal case and must keep parsing. `D1`/`R1` are dose attributes but inert
    // until the data codes a RATE, so they are not a parse error either.
    for name in ["KEL", "ZQ", "D1", "R1"] {
        let src = adaptive_observe_model(name, &format!("central / (V * {name})"));
        assert!(
            parse_full_model(&src).is_ok(),
            "`{name}` in [adaptive_dosing] observe must parse"
        );
    }
}

#[test]
fn malformed_adaptive_observe_now_fails_at_parse_time() {
    // Consequence of the #993 walk, worth pinning because it is a behaviour change:
    // `parse_adaptive_dosing_block` only stores `observe` as a string (the real
    // compile happens in `compile_observe`, at simulate time), so this is the FIRST
    // parse of that expression. A syntactically broken signal now surfaces here
    // rather than at the first `simulate()` — earlier, and attributed to the block
    // it came from.
    let err = expect_parse_err(&adaptive_observe_model("KEL", "central / (V * KEL"));
    assert!(
        err.contains("[adaptive_dosing] observe"),
        "the error must name the block and key it came from, got: {err}"
    );
}

#[test]
fn coded_rate_param_read_in_adaptive_observe_is_recorded() {
    // The data-gated half of the same path: `R1` in the controller signal is legal
    // on ordinary data, so nothing is rejected — but the read must be *recorded* so
    // `check_modeled_dose_rates` can report it once a `RATE=-1` dose lands on it.
    // Silence here and a diagnostic there is the whole contract.
    use crate::types::DoseAttr;
    let parsed = parse_full_model(&adaptive_observe_model("R1", "central / (V * R1)"))
        .expect("an R1 controller signal parses");
    assert_eq!(
        parsed
            .model
            .active_dose_attr_map()
            .prediction_path_read(DoseAttr::Rate, 1),
        Some("R1"),
        "an `R1` read by [adaptive_dosing] observe must be recorded for the data check"
    );
    // The same model without the read records nothing — otherwise the assertion
    // above would pass for a map that marks everything.
    let clean = parse_full_model(&adaptive_observe_model("R1", "central / V"))
        .expect("the control model parses");
    assert_eq!(
        clean
            .model
            .active_dose_attr_map()
            .prediction_path_read(DoseAttr::Rate, 1),
        None,
        "an unread `R1` must not be marked"
    );
}

#[test]
fn dose_attr_param_reused_as_a_disposition_role_declines_the_twin() {
    // Second door into the same panic, found probing the first. `F` here is the `f=`
    // mapping — so the #735 shadow guard allows it — *and* the `v=` role, so the
    // generated twin emits `d/dt(central) = … − (CL/F) * central` with
    // `obs_scale = F`, reading a name `ode_param_slots` routes to the F slot. The
    // twin's own parse rejects that as a #993 double use, which (before #1008 made
    // the build a parse-time decline) `get_or_build` `.expect()`ed — so a model the
    // analytical primary accepts crashed the moment a TV-covariate / `TIME` / IOV
    // subject rerouted to the twin.
    //
    // The specific guard is kept even though #1008's attach-site decline would now
    // catch this generically: it declines *before* reconstructing a twin known to be
    // unusable, so the model needs no `W_ABSORPTION_TWIN_DECLINED` warning for a case
    // the desugar can name exactly. (`state_named_parameter_declines_the_absorption_
    // twin_with_a_warning` covers the generic backstop.)
    //
    // The model is pharmacological nonsense (bioavailability used as a volume), but
    // nonsense must not panic. Declining keeps it closed-form — exactly what it was
    // before the twin existed.
    let src = "
[parameters]
  theta TVCL(5.0, 0.0, 1e15)
  theta TVN(3.0, 1.0, 20.0)
  theta TVMTT(1.5, 0.01, 100.0)
  theta THETA_F(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma EPS1 ~ 0.1 (sd)

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  N   = TVN
  MTT = TVMTT
  F   = inv_logit(THETA_F)

[structural_model]
  pk one_cpt_transit(cl=CL, v=F, n=N, mtt=MTT, f=F)

[error_model]
  DV ~ proportional(EPS1)
";
    let parsed = parse_full_model(src).expect("the analytical primary still parses");
    assert!(
        parsed.model.absorption_ode_equivalent.is_none(),
        "a dose-attribute parameter reused as a disposition role must decline the twin, \
         not build one that panics"
    );
    // The desugar names this case, so it declines *before* reconstructing a source —
    // no generic build-failure warning is raised.
    assert!(
        !parsed
            .model
            .parse_warnings
            .iter()
            .any(|w| w.contains("W_ABSORPTION_TWIN_DECLINED")),
        "a guard-recognised decline must not fall through to the generic build-failure \
         warning; warnings: {:?}",
        parsed.model.parse_warnings
    );
}

/// #1008: the third live door into the `.expect()` panic the first two (#1003) armed —
/// found by probing the class rather than the instances, and the reason the fix is
/// structural instead of a fourth point guard.
///
/// The twin's ODE states are named `central` (and `periph` for 2-cpt). An individual
/// parameter named `CENTRAL` is meaningless to the analytical primary — which has no state
/// namespace at all — so nothing rejects it there; the twin then re-emits it into
/// `[individual_parameters]` beside `states=[central]`, and the twin's own parse rejects the
/// case-insensitive name collision. None of the three existing guards look at state names,
/// so this reached `get_or_build` and panicked mid-fit.
///
/// Three shapes are covered: a stray parameter that merely *exists* (never referenced by the
/// `pk()` mapping), one that fills a disposition role, and the 2-cpt `periph` analogue —
/// whose colliding name set differs because the twin's state list is topology-dependent.
#[test]
fn state_named_parameter_declines_the_absorption_twin_with_a_warning() {
    let stray = "
[parameters]
  theta TVCL(5.0, 0.0, 1e15)
  theta TVV(50.0, 0.0, 1e15)
  theta TVN(3.0, 1.0, 20.0)
  theta TVMTT(1.5, 0.01, 100.0)
  theta TVX(1.0, 0.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma EPS1 ~ 0.1 (sd)

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  N   = TVN
  MTT = TVMTT
  CENTRAL = TVX

[structural_model]
  pk one_cpt_transit(cl=CL, v=V, n=N, mtt=MTT)

[error_model]
  DV ~ proportional(EPS1)
";
    // `v=CENTRAL` — the parameter is load-bearing, so the twin's `d/dt(central)` RHS reads
    // the very name it declares as a state.
    let disposition_role = "
[parameters]
  theta TVCL(5.0, 0.0, 1e15)
  theta TVV(50.0, 0.0, 1e15)
  theta TVN(3.0, 1.0, 20.0)
  theta TVMTT(1.5, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  sigma EPS1 ~ 0.1 (sd)

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  CENTRAL = TVV
  N   = TVN
  MTT = TVMTT

[structural_model]
  pk one_cpt_transit(cl=CL, v=CENTRAL, n=N, mtt=MTT)

[error_model]
  DV ~ proportional(EPS1)
";
    // The 2-cpt twin declares a second state, `periph`, so its state namespace — and hence
    // the set of parameter names that collide — depends on the topology. Cover both.
    let two_cpt_periph = "
[parameters]
  theta TVCL(5.0, 0.0, 1e15)
  theta TVV1(50.0, 0.0, 1e15)
  theta TVQ(10.0, 0.0, 1e15)
  theta TVV2(80.0, 0.0, 1e15)
  theta TVN(3.0, 1.0, 20.0)
  theta TVMTT(1.5, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  sigma EPS1 ~ 0.1 (sd)

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V1  = TVV1
  Q   = TVQ
  PERIPH = TVV2
  N   = TVN
  MTT = TVMTT

[structural_model]
  pk two_cpt_transit(cl=CL, v1=V1, q=Q, v2=PERIPH, n=N, mtt=MTT)

[error_model]
  DV ~ proportional(EPS1)
";
    for (label, src, colliding) in [
        ("stray", stray, "CENTRAL"),
        ("disposition role", disposition_role, "CENTRAL"),
        ("2-cpt periph", two_cpt_periph, "PERIPH"),
    ] {
        let parsed = parse_full_model(src)
            .unwrap_or_else(|e| panic!("the analytical primary ({label}) must still parse: {e}"));
        assert!(
            parsed.model.absorption_ode_equivalent.is_none(),
            "a parameter named after a twin state ({label}) must decline the twin, not build \
             one that panics"
        );
        let warning = parsed
            .model
            .parse_warnings
            .iter()
            .find(|w| w.contains("W_ABSORPTION_TWIN_DECLINED"))
            .unwrap_or_else(|| {
                panic!(
                    "the decline ({label}) must be reported; warnings: {:?}",
                    parsed.model.parse_warnings
                )
            });
        // The twin parser's own message is carried through, so the reason is readable
        // without a debugger — the specific complaint here is the state-name collision.
        assert!(
            warning.contains(colliding) && warning.contains("collides"),
            "the warning must carry the twin parser's reason (naming `{colliding}`), \
             got: {warning}"
        );
        // …and it must classify to its own code rather than the `general` bucket, so a
        // consumer can branch on "this model kept no ODE fallback". Asserted here on the
        // *produced* message, so the code cannot drift away from the text that carries it.
        assert_eq!(
            crate::types::classify_warning(warning).category.as_str(),
            "absorption_twin_declined",
            "the decline ({label}) must classify to its own warning code"
        );

        // The load-bearing half of the trade (#1027 review): declining is only safe because a
        // subject that would have rerouted is *rejected*, not silently served by the closed
        // form. Reach the guard directly rather than through `fit()` (Tier 1: no convergence
        // loop) — this is the exact call `fit()` makes at its `check_absorption_closed_form_
        // support` gate, and the same `Option<String>` `predict()`/`simulate()` panic on.
        let rejection =
            crate::api::check_absorption_closed_form_support(&parsed.model, &tv_cov_population())
                .unwrap_or_else(|| {
                    panic!(
                        "a declined-twin model ({label}) must reject a TV-covariate subject, \
                         not silently serve it from the closed form"
                    )
                });
        // …and the rejection must name the real cause. Before #1027 it told the author their
        // model was "an unrecognised closed form" and pointed at a rewrite they did not need,
        // while the reason lived in a parse warning `fit()` never reaches on the `Err` path.
        // Asserted on the *content* (the colliding name) rather than the wrapper wording, so a
        // reworded clause does not silently pass a rejection that dropped the reason.
        assert!(
            rejection.contains(colliding),
            "the rejection ({label}) must quote the twin's decline reason (naming \
             `{colliding}`), got: {rejection}"
        );
        assert!(
            !rejection.contains("an unrecognised closed form"),
            "a recognised closed form whose twin was declined must not be called unrecognised \
             ({label}), got: {rejection}"
        );

        // The other two reroute-needing features the twin serves (#719) take their own message
        // arms, so assert each carries the cause and the reason too — a declined twin loses SS
        // and infusion exactly like it loses the TV-covariate reroute, and an author who hits
        // one of those first must not be told the form is unrecognised either.
        for (feature, marker, population) in [
            (
                "steady-state",
                "steady-state (SS) doses",
                ss_dose_population(),
            ),
            ("infusion", "infusion doses", infusion_population()),
        ] {
            let msg = crate::api::check_absorption_closed_form_support(&parsed.model, &population)
                .unwrap_or_else(|| {
                    panic!(
                        "a declined-twin model ({label}) must reject a {feature} subject, not \
                         silently serve it from the closed form"
                    )
                });
            // Pin *which* arm fired: these populations are built by mutating the TV-covariate
            // one, so without this the assertions below would pass just as happily on the
            // TV-covariate message and prove nothing about the SS / infusion arms.
            assert!(
                msg.contains(marker),
                "the {feature} subject ({label}) must hit the {feature} arm, got: {msg}"
            );
            assert!(
                msg.contains(colliding) && !msg.contains("an unrecognised closed form"),
                "the {feature} rejection ({label}) must name the decline cause and quote the \
                 reason (naming `{colliding}`), got: {msg}"
            );
        }
    }
}

/// A one-subject population whose subject takes a **steady-state** dose, and one whose subject
/// takes an **infusion** — the other two features that reroute to the ODE twin (#719) and are
/// therefore rejected when there is no twin. Each hits its own arm of
/// `check_absorption_closed_form_support`, so both are exercised alongside the TV-covariate one.
fn ss_dose_population() -> crate::types::Population {
    let mut pop = tv_cov_population();
    let s = &mut pop.subjects[0];
    s.id = "SS1".to_string();
    s.obs_covariates = Vec::new(); // drop the TV covariate so the SS arm is what fires
    s.doses = vec![crate::types::DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)];
    pop
}

fn infusion_population() -> crate::types::Population {
    let mut pop = tv_cov_population();
    let s = &mut pop.subjects[0];
    s.id = "INF1".to_string();
    s.obs_covariates = Vec::new(); // drop the TV covariate so the infusion arm is what fires
    s.doses = vec![crate::types::DoseEvent::new(
        0.0, 100.0, 1, 50.0, false, 0.0,
    )];
    pop
}

/// A one-subject population whose subject carries a time-varying covariate row — the cheapest
/// thing that makes `Subject::has_tv_covariates()` true, which is what routes a closed-form
/// absorption subject to its twin (`CompiledModel::effective_for`) and, twin-less, is what
/// `check_absorption_closed_form_support` rejects on.
fn tv_cov_population() -> crate::types::Population {
    crate::types::Population {
        subjects: vec![crate::types::Subject {
            id: "TV1".to_string(),
            doses: vec![crate::types::DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0],
            obs_raw_times: Vec::new(),
            observations: vec![10.0],
            obs_cmts: vec![1],
            covariates: std::collections::HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: vec![std::collections::HashMap::from([("WT".to_string(), 80.0)])],
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            reset_covariates: Vec::new(),
            cens: vec![0],
            occasions: vec![1],
            obs_l2: Vec::new(),
            dose_occasions: vec![1],
            reset_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
        }],
        covariate_names: vec!["WT".to_string()],
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

/// The positive control for the test above: an ordinary transit model — same shape, no
/// state-named parameter — keeps its twin and is warned about nothing. Without this, the
/// decline path could pass by declining *everything*.
#[test]
fn an_ordinary_transit_model_keeps_its_twin_and_warns_about_nothing() {
    let src = "
[parameters]
  theta TVCL(5.0, 0.0, 1e15)
  theta TVV(50.0, 0.0, 1e15)
  theta TVN(3.0, 1.0, 20.0)
  theta TVMTT(1.5, 0.01, 100.0)
  theta TVX(1.0, 0.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma EPS1 ~ 0.1 (sd)

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  N   = TVN
  MTT = TVMTT
  CENTRAL_AMT = TVX

[structural_model]
  pk one_cpt_transit(cl=CL, v=V, n=N, mtt=MTT)

[error_model]
  DV ~ proportional(EPS1)
";
    let parsed = parse_full_model(src).expect("an ordinary transit model parses");
    assert!(
        parsed
            .model
            .absorption_ode_equivalent
            .as_ref()
            .is_some_and(|eq| eq.built().ode_spec.is_some()),
        "an ordinary transit model must keep a working ODE twin; warnings: {:?}",
        parsed.model.parse_warnings
    );
    assert!(
        !parsed
            .model
            .parse_warnings
            .iter()
            .any(|w| w.contains("W_ABSORPTION_TWIN_DECLINED")),
        "nothing to decline here; warnings: {:?}",
        parsed.model.parse_warnings
    );
    // The negative control for the rejection assertion in the test above: a twin-carrying
    // model must *accept* the very subject a declined one rejects — otherwise that assertion
    // could pass by rejecting every transit model with a TV covariate.
    assert_eq!(
        crate::api::check_absorption_closed_form_support(&parsed.model, &tv_cov_population()),
        None,
        "a twin-carrying transit model must reroute a TV-covariate subject, not reject it"
    );
}

#[test]
fn adaptive_observe_does_not_reach_the_absorption_twin() {
    // Regression guard on the interaction between the #993 `[adaptive_dosing]`
    // rejection and the absorption ODE twin, found reviewing that commit.
    //
    // `absorption_ode_equivalent_source` re-emits the model's blocks into a source
    // whose `[structural_model]` is `ode(...)`. `[odes]` and `[scaling]` can never
    // reach a new ODE-only check from there — their presence makes the twin decline
    // outright — but `[adaptive_dosing]` does not decline it. So an analytical model
    // that the parser deliberately accepts (the rejection is ODE-scoped, see #1004)
    // produced a twin that the parser *rejects* — which, before #1008 made the build
    // a parse-time decline, `get_or_build` turned into a `.expect()` panic mid-fit, on
    // the plain predict path that reroutes TV-covariate / `TIME` / IOV subjects.
    //
    // The block is now dropped from the twin, so the model keeps a working fallback.
    //
    // #1004 update: the primary no longer accepts a dose-attribute read in `observe`
    // — that is now the analytical rejection, asserted below — so no *reachable*
    // controller carries something the twin's own parse would reject. That makes
    // "the twin exists, therefore the block was dropped" vacuous: with the drop
    // removed this test still passed. The second half therefore asserts on the
    // re-emitted source itself, which holds whatever the controller says.
    let src = "
[parameters]
  theta TVCL(5.0, 0.0, 1e15)
  theta TVV(50.0, 0.0, 1e15)
  theta TVN(3.0, 1.0, 20.0)
  theta TVMTT(1.5, 0.01, 100.0)
  theta THETA_F(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma EPS1 ~ 0.1 (sd)

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  N   = TVN
  MTT = TVMTT
  F   = inv_logit(THETA_F)

[structural_model]
  pk one_cpt_transit(cl=CL, v=V, n=N, mtt=MTT, f=F)

[adaptive_dosing]
  observe = central / (V * F)
  at = [24, 48]
  start_dose = 100
  route = bolus(cmt = 1)
  dose_bounds = [0, 400]
  when signal < 10 : increase 25%

[error_model]
  DV ~ proportional(EPS1)
";
    // Half one: the primary rejects it itself (#1004), rather than accepting it and
    // handing the twin a source whose own parse fails — which since #1008 costs the
    // model its fallback silently instead of panicking.
    let err = expect_parse_err(src);
    assert!(
        err.contains("[adaptive_dosing]:") && err.contains("remove the `f=F` mapping"),
        "the primary must reject the dose-attribute observe, got: {err}"
    );

    // Half two: the block-drop itself, asserted on the reconstructed source rather
    // than inferred from the twin parsing. "The twin builds, therefore the block
    // was dropped" only holds while the controller still carries something the ODE
    // parse rejects — and after #1004 the primary rejects that shape first, so any
    // controller reaching this point is one the twin's own parse would accept.
    // Reading the emitted source keeps the guard honest whatever `observe` says.
    let ok_src = src.replace("observe = central / (V * F)", "observe = central / V");
    assert_ne!(ok_src, src, "the accepted variant must actually differ");
    let parsed = parse_full_model(&ok_src).expect("a controller without a dose attribute parses");
    // The primary keeps its controller — only the twin drops it.
    assert!(
        parsed.adaptive_dosing.is_some(),
        "the primary must still carry the [adaptive_dosing] block"
    );
    let eq = parsed
        .model
        .absorption_ode_equivalent
        .as_ref()
        .expect("a plain transit model carries an ODE twin");
    // The drop, asserted where it happens. `AbsorptionOdeEquivalent` keeps only the
    // built model since #1008, so the reconstruction is re-run here rather than read
    // off the twin — same function, same blocks, and it fails if the block comes back.
    let extracted = super::extract_blocks(&ok_src).expect("the model file extracts");
    let twin_src = super::absorption_ode_equivalent_source(&extracted)
        .expect("a plain transit model reconstructs a twin source");
    assert!(
        !twin_src.contains("[adaptive_dosing]"),
        "the twin source must not re-emit the controller block, got:\n{twin_src}"
    );
    let twin = eq.built();
    assert!(
        twin.ode_spec.is_some(),
        "the twin must be a working ODE model"
    );
}

#[test]
fn dose_attr_read_in_scaling_is_rejected() {
    // The readout is a prediction path too: dividing by `F` applies bioavailability
    // a second time, measured at exactly `F`. Both readout forms — Form-C `y` and
    // `obs_scale` — reach the same check.
    for readout in ["y = central / (V * F)", "obs_scale = V * F"] {
        let src = format!(
            "
[parameters]
  theta TVCL(5.0, 0.0, 1e15)
  theta TVV(50.0, 0.0, 1e15)
  theta TVF(0.5, 0.0, 1.0)
  omega ETA_CL ~ 0.09
  sigma EPS1 ~ 0.1 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  F  = TVF

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -(CL/V) * central

[scaling]
  {readout}

[error_model]
  DV ~ proportional(EPS1)
"
        );
        let err = expect_parse_err(&src);
        assert!(
            err.contains("[scaling]:") && err.contains("reserved dose-attribute name"),
            "`{readout}` double-applies F and must be rejected, got: {err}"
        );
    }
}

#[test]
fn dose_attr_read_only_for_reporting_is_accepted() {
    // `[derived]` / `[output]` are post-solve reporting, not the prediction path: a
    // model that tabulates its own bioavailability (and an exposure derived from
    // it) is correct and must stay silent. Rejecting here would also panic the
    // absorption ODE twin, which re-emits `[derived]` verbatim into a source it
    // `.expect()`s to re-parse.
    let src = "
[parameters]
  theta TVCL(5.0, 0.0, 1e15)
  theta TVV(50.0, 0.0, 1e15)
  theta TVF(0.5, 0.0, 1.0)
  omega ETA_CL ~ 0.09
  sigma EPS1 ~ 0.1 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  F  = TVF

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -(CL/V) * central

[derived]
  AUCF = F * 100.0 / CL

[output]
  F

[error_model]
  DV ~ proportional(EPS1)
";
    let parsed = parse_full_model(src)
        .expect("reading F only for reporting is not a double use and must parse");
    assert!(
        !parsed
            .model
            .parse_warnings
            .iter()
            .any(|w| w.contains("computed but never used")),
        "a reported F is used, not dead: {:?}",
        parsed.model.parse_warnings
    );
}

/// An analytical model whose `pk(...)` call carries the extra argument `mapping`
/// (empty for none) and whose `[scaling]`/`[initial_conditions]`/
/// `[adaptive_dosing]` block is `block`. `decl` is the individual-parameter line
/// for the parameter under test.
fn analytical_dose_attr_src(mapping: &str, decl: &str, block: &str) -> String {
    let args = if mapping.is_empty() {
        "cl=CL, v=V, ka=KA".to_string()
    } else {
        format!("cl=CL, v=V, ka=KA, {mapping}")
    };
    format!(
        "
[parameters]
  theta TVCL(5.0, 0.0, 1e15)
  theta TVV(50.0, 0.0, 1e15)
  theta TVKA(1.5, 0.0, 1e15)
  theta TVF(0.5, 0.0, 1.0)
  theta TVLAG(0.3, 0.0, 5.0)
  omega ETA_CL ~ 0.09
  sigma EPS1 ~ 0.1 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  {decl}

[structural_model]
  pk one_cpt_oral({args})

{block}

[error_model]
  DV ~ proportional(EPS1)
"
    )
}

#[test]
fn analytical_dose_attr_read_in_scaling_is_rejected() {
    // #1004, the analytical half of #993. `pk(..., f=F)` applies `F` at the dose;
    // an `obs_scale`/`y` that reads `F` applies it a second time — measured at
    // exactly `F` on the prediction (`tests/dose_attr_double_use_nonmem_anchor.rs`,
    // which also pins NONMEM's own ADVAN2 doing the same silently under
    // `S2 = V/F1`). #1003 left this accepted, arguing the explicit mapping made it
    // "stated rather than silent"; nothing in the model states the value is applied
    // twice, which is exactly the silence #993 closed on the ODE engine.
    //
    // The readouts deliberately do NOT mention `V`: `obs_scale = V * F` (what this
    // test pinned before) trips the pre-existing volume double-divide warning in
    // `build_obs_scale_spec`, which masks how quiet the general case is.
    for readout in [
        "obs_scale = 50.0 * F",
        "obs_scale = 1.0 / F",
        "y = central * F",
    ] {
        let src = analytical_dose_attr_src("f=F", "F  = TVF", &format!("[scaling]\n  {readout}"));
        let err = expect_parse_err(&src);
        assert!(
            err.contains("[scaling]:") && err.contains("remove the `f=F` mapping"),
            "`{readout}` double-applies F and must be rejected, got: {err}"
        );
        // The remediation is engine-correct: renaming does nothing here, because
        // the mapping follows the parameter.
        assert!(
            !err.contains("reserved dose-attribute name"),
            "the analytical wording must not tell the user to rename: {err}"
        );
    }
}

#[test]
fn analytical_dose_attr_read_through_scaling_intermediate_is_rejected() {
    // #1030 must not open a hole in #1004: naming the doubled read does not make it
    // stop happening, so a readout that reaches `F` only through a named
    // intermediate is rejected exactly as the inline spelling is. This is the
    // read-scan half of the inlining contract — the scan runs on the expanded AST,
    // not on the raw line.
    let src = analytical_dose_attr_src(
        "f=F",
        "F  = TVF",
        "[scaling]\n  SCALE = 50.0 * F\n  obs_scale = SCALE",
    );
    let err = expect_parse_err(&src);
    assert!(
        err.contains("[scaling]:") && err.contains("remove the `f=F` mapping"),
        "a dose attribute reached through an intermediate must still be rejected, got: {err}"
    );
}

#[test]
fn analytical_lag_mapping_read_in_scaling_is_rejected() {
    // The lag half, and both spellings of the mapping (`lagtime=` and the NONMEM
    // alias `alag=`, which `PkParams::name_to_index` routes to the same slot). A
    // readout that multiplies by the lag applies it once as a time shift at the
    // dose and once as a number here.
    for mapping in ["lagtime=TLAG", "alag=TLAG"] {
        let src = analytical_dose_attr_src(
            mapping,
            "TLAG = TVLAG",
            "[scaling]\n  obs_scale = 2.0 * TLAG",
        );
        let err = expect_parse_err(&src);
        // The message must quote the role the model file actually uses: telling an
        // `alag=` user to remove a `lagtime=` argument names text that is not there.
        assert!(
            err.contains("[scaling]:")
                && err.contains("absorption lag")
                && err.contains(&format!("remove the `{mapping}` mapping")),
            "`{mapping}` + a TLAG readout must be rejected and quote its own spelling, got: {err}"
        );
    }
}

#[test]
fn analytical_dose_attr_read_in_initial_conditions_is_accepted() {
    // Scope floor, and the one surface the #1004 rule must NOT claim. An analytical
    // init amount is not a dose: `pk::analytical_init_concentration_g` propagates it
    // with `F = 1` and no lag ("an initial condition is not an absorbed dose, so
    // bioavailability does not apply"), so `init(depot) = F * 500` — the
    // bioavailable residue of a pre-study 500 mg dose — applies `F` exactly once,
    // to a term the engine never scales. That is a legitimate model.
    //
    // Contrast `[scaling]`, which IS a doubling: `obs_scale` multiplies every
    // prediction, the F-scaled dose contribution included. An init sits beside that
    // contribution, not on top of it.
    for block in [
        "[initial_conditions]\n  init(central) = F * 100.0",
        "[initial_conditions]\n  init(depot) = F * 500.0",
    ] {
        let src = analytical_dose_attr_src("f=F", "F  = TVF", block);
        assert!(
            parse_full_model(&src).is_ok(),
            "an init amount that reads a mapped `F` applies it once, not twice: {block}"
        );
    }
    // Same for lag, whose init impulse is likewise deposited at t=0 unshifted.
    let lag = analytical_dose_attr_src(
        "lagtime=TLAG",
        "TLAG = TVLAG",
        "[initial_conditions]\n  init(central) = TLAG * 100.0",
    );
    assert!(
        parse_full_model(&lag).is_ok(),
        "the init impulse carries no lag either"
    );
}

#[test]
fn analytical_dose_attr_read_only_for_reporting_is_accepted() {
    // The analytical counterpart of `dose_attr_read_only_for_reporting_is_accepted`
    // (which pins the ODE engine). `[derived]`/`[output]` are post-solve reporting,
    // not the prediction path, so tabulating a mapped `F` is correct and must stay
    // silent on the default engine too — this is the scope floor `ferxtranslate`'s
    // reporting columns sit on.
    let src = "
[parameters]
  theta TVCL(5.0, 0.0, 1e15)
  theta TVV(50.0, 0.0, 1e15)
  theta TVF(0.5, 0.0, 1.0)
  omega ETA_CL ~ 0.09
  sigma EPS1 ~ 0.1 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  F  = TVF

[structural_model]
  pk one_cpt_iv(cl=CL, v=V, f=F)

[derived]
  AUCF = F * 100.0 / CL

[output]
  F

[error_model]
  DV ~ proportional(EPS1)
";
    assert!(
        parse_full_model(src).is_ok(),
        "reading a mapped F only for reporting is not a double use"
    );
}

#[test]
fn analytical_dose_attr_remedy_quotes_the_mapping_as_written() {
    // The remediation clause has to name text the model file actually contains.
    // `analytical_dose_attr_slot_map` binds through `build_pk_param_fn`'s lowercase
    // compat lookup as well as an exact match, so `alag=TLAG` legitimately binds a
    // parameter declared `tlag` — and the message must say `alag=TLAG`, not the
    // canonical `lagtime=` (an argument the file never wrote) nor `tlag` (the
    // declaration's casing, which is not what the `pk(...)` call says).
    let src = analytical_dose_attr_src(
        "alag=TLAG",
        "tlag = TVLAG",
        "[scaling]\n  obs_scale = 2.0 * tlag",
    );
    let err = expect_parse_err(&src);
    assert!(
        err.contains("remove the `alag=TLAG` mapping"),
        "the message must quote the mapping as written, got: {err}"
    );
    assert!(
        !err.contains("lagtime="),
        "naming a `lagtime=` argument the file does not contain misdirects: {err}"
    );
}

#[test]
fn analytical_dose_attr_diagnostic_is_deterministic_across_parses() {
    // `pk_param_map` is a `HashMap`, and one parameter can fill both dose-attribute
    // roles. With arbitrary iteration order the surviving slot — and therefore the
    // noun and the quoted role — differed between parses of the identical source,
    // and the two halves disagreed with each other ("bioavailability … remove the
    // `lagtime=X` mapping"). Sorted iteration in `analytical_dose_attr_slot_map`
    // plus an `attr`-filtered role lookup makes both stable and mutually
    // consistent. Several parses in one process, because `HashMap`'s ordering is
    // per-instance.
    let src = analytical_dose_attr_src(
        "f=X, lagtime=X",
        "X  = TVF",
        "[scaling]\n  obs_scale = 2.0 * X",
    );
    let first = expect_parse_err(&src);
    for _ in 0..16 {
        assert_eq!(
            expect_parse_err(&src),
            first,
            "the diagnostic must not depend on HashMap iteration order"
        );
    }
    // And the noun must agree with the role the clause tells the user to remove.
    assert!(
        first.contains("absorption lag") && first.contains("remove the `lagtime=X` mapping"),
        "noun and quoted role must describe the same slot, got: {first}"
    );
}

#[test]
fn analytical_lag_alias_spellings_each_name_their_own_mapping() {
    // `lagtime=` and `alag=` are two spellings of the SAME slot, so a `pk(...)`
    // call may legally carry both. Two cases, both of which the remediation clause
    // has to get right on its own — the role lookup cannot just take the first map
    // entry that routes to the lag slot.
    //
    // (a) Two different parameters on the one slot. Whichever is read, the message
    //     must quote *that* parameter's own spelling.
    let two = |readout: &str| {
        format!(
            "
[parameters]
  theta TVCL(5.0, 0.0, 1e15)
  theta TVV(50.0, 0.0, 1e15)
  theta TVKA(1.5, 0.0, 1e15)
  theta TVLAG(0.3, 0.0, 5.0)
  omega ETA_CL ~ 0.09
  sigma EPS1 ~ 0.1 (sd)

[individual_parameters]
  CL    = TVCL * exp(ETA_CL)
  V     = TVV
  KA    = TVKA
  TLAGA = TVLAG
  TLAGB = TVLAG

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, lagtime=TLAGA, alag=TLAGB)

[scaling]
  obs_scale = 2.0 * {readout}

[error_model]
  DV ~ proportional(EPS1)
"
        )
    };
    for (readout, want) in [
        ("TLAGA", "remove the `lagtime=TLAGA` mapping"),
        ("TLAGB", "remove the `alag=TLAGB` mapping"),
    ] {
        let err = expect_parse_err(&two(readout));
        assert!(
            err.contains(want),
            "reading `{readout}` must quote its own mapping, got: {err}"
        );
    }

    // (b) One parameter under both spellings. Both `analytical_dose_attr_slot_map`
    //     and `build_pk_param_fn` iterate ascending and let the last write win, so
    //     `lagtime=` is the mapping that actually reaches the slot — quoting
    //     `alag=` would name one whose removal changes nothing.
    let both = analytical_dose_attr_src(
        "lagtime=X, alag=X",
        "X = TVLAG",
        "[scaling]\n  obs_scale = 2.0 * X",
    );
    let err = expect_parse_err(&both);
    assert!(
        err.contains("remove the `lagtime=X` mapping"),
        "must quote the binding mapping, not the shadowed alias, got: {err}"
    );
}

#[test]
fn analytical_dose_attr_read_in_adaptive_observe_is_rejected() {
    // The controller signal. A dose attribute read here biases every dose the
    // controller then emits by exactly that attribute.
    let src = analytical_dose_attr_src(
        "f=F",
        "F  = TVF",
        "[adaptive_dosing]\n  observe = central / (50.0 * F)\n  at = [24, 48]\n  \
         start_dose = 100\n  route = bolus(cmt = 1)\n  dose_bounds = [0, 400]\n  \
         when signal < 10 : increase 25%",
    );
    let err = expect_parse_err(&src);
    assert!(
        err.contains("[adaptive_dosing]:") && err.contains("remove the `f=F` mapping"),
        "an observe signal that reads F double-applies it, got: {err}"
    );
}

#[test]
fn analytical_unmapped_f_named_parameter_is_not_a_double_use() {
    // Scope floor for #1004. On the analytical engine the *name* is inert — only
    // the `pk(...)` mapping binds a parameter to the dose route. A model with an
    // ordinary parameter that happens to be called `F`, not mapped, is correct and
    // must keep parsing (this is what ferxtranslate's nlmixr2 sources produce: a
    // plain parameter whose name has dose-attribute shape).
    let src = analytical_dose_attr_src("", "F  = TVF", "[scaling]\n  obs_scale = 50.0 * F");
    assert!(
        parse_full_model(&src).is_ok(),
        "an unmapped `F` is an ordinary parameter on the analytical engine"
    );

    // And the mapped-but-unread case: `f=F` with a readout that does not mention
    // `F` is the ordinary bioavailability model, which must stay silent.
    let ok = analytical_dose_attr_src("f=F", "F  = TVF", "[scaling]\n  obs_scale = 50.0");
    assert!(
        parse_full_model(&ok).is_ok(),
        "a mapped-but-unread F is the ordinary model"
    );
}

#[test]
fn analytical_dose_attr_mapped_to_another_parameter_only_rejects_that_one() {
    // The mapping, not the name, is what the check follows: `f=FBIO` makes `FBIO`
    // the dose attribute even though its name has no dose-attribute shape, while a
    // separate parameter literally named `F` stays ordinary. Reading `F` is fine;
    // reading `FBIO` is the double use.
    let ok = "
[parameters]
  theta TVCL(5.0, 0.0, 1e15)
  theta TVV(50.0, 0.0, 1e15)
  theta TVF(0.5, 0.0, 1.0)
  omega ETA_CL ~ 0.09
  sigma EPS1 ~ 0.1 (sd)

[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  F    = TVF
  FBIO = TVF

[structural_model]
  pk one_cpt_iv(cl=CL, v=V, f=FBIO)

[scaling]
  obs_scale = 50.0 * F

[error_model]
  DV ~ proportional(EPS1)
";
    assert!(
        parse_full_model(ok).is_ok(),
        "the unmapped `F` is ordinary; only the mapped `FBIO` is the dose attribute"
    );
    let bad = ok.replace("obs_scale = 50.0 * F\n", "obs_scale = 50.0 * FBIO\n");
    assert_ne!(bad, ok, "the bad variant must actually differ");
    let err = expect_parse_err(&bad);
    assert!(
        err.contains("`FBIO`") && err.contains("remove the `f=FBIO` mapping"),
        "reading the mapped `FBIO` is the double use, got: {err}"
    );
}

#[test]
fn transit_twin_with_reserved_f_name_still_builds() {
    // Regression guard on the absorption ODE twin: any new parse error the twin can
    // trip costs the model its fallback (and, before #1008, panicked at fit time), so
    // the twin must keep re-parsing for the shapes the desugar accepts. The twin
    // re-emits `[individual_parameters]` verbatim and, for an `f=` role whose
    // parameter does not already self-route, appends an `f = <param>` alias — so
    // `f` appears as a declaration but never as a read. Cover both: a parameter
    // literally named `F` (self-routing, no alias) and one named `FBIO` (alias
    // emitted).
    for (fname, mapping) in [("F", "f=F"), ("FBIO", "f=FBIO")] {
        let src = format!(
            "
[parameters]
  theta TVCL(5.0, 0.0, 1e15)
  theta TVV(50.0, 0.0, 1e15)
  theta TVN(3.0, 1.0, 20.0)
  theta TVMTT(1.5, 0.01, 100.0)
  theta THETA_F(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma EPS1 ~ 0.1 (sd)

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  N   = TVN
  MTT = TVMTT
  {fname}   = inv_logit(THETA_F)

[structural_model]
  pk one_cpt_transit(cl=CL, v=V, n=N, mtt=MTT, {mapping})

[error_model]
  DV ~ proportional(EPS1)
"
        );
        let parsed = parse_full_model(&src)
            .unwrap_or_else(|e| panic!("transit model with {mapping} must parse: {e}"));
        // The twin is built during that parse (#1008). Assert it is *present*: a twin
        // whose source fails to parse is now declined rather than panicking, so
        // "didn't panic" alone would pass vacuously for a broken reconstruction.
        let eq = parsed
            .model
            .absorption_ode_equivalent
            .as_ref()
            .unwrap_or_else(|| {
                panic!(
                    "transit model with {mapping} must keep its ODE twin; warnings: {:?}",
                    parsed.model.parse_warnings
                )
            });
        assert!(
            eq.built().ode_spec.is_some(),
            "the twin for {mapping} must be a working ODE model"
        );
    }
}

#[test]
fn test_ode_multiple_dead_params_use_plural_message() {
    // #315: two+ dead ODE params share one warning and use the plural grammar
    // ("are … they … them") in the ODE-flavored message. Locks the plural
    // branch + name list for ODE (the singular case is covered above).
    let content = r#"
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta TVKE(0.5, 0.001, 10.0)
  theta TVKE2(0.2, 0.001, 10.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  KE  = TVKE      # never referenced in [odes]
  KE2 = TVKE2     # never referenced in [odes]

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -(CL/V) * central

[error_model]
  DV ~ proportional(EPS)
"#;
    let ws = super::parse_full_model(content)
        .expect("dead ODE params are a warning, not an error")
        .model
        .parse_warnings;
    let dead: Vec<&String> = ws
        .iter()
        .filter(|w| w.contains("computed but never used"))
        .collect();
    assert_eq!(
        dead.len(),
        1,
        "both dead params share a single warning, got: {ws:?}"
    );
    let w = dead[0];
    // Both names listed (sorted, comma-joined), neither used param flagged.
    assert!(
        w.contains("`KE`") && w.contains("`KE2`"),
        "both dead params must be named: {w}"
    );
    assert!(
        !w.contains("`CL`") && !w.contains("`V`"),
        "used params not flagged: {w}"
    );
    // Plural grammar + ODE-flavored guidance.
    assert!(
        w.contains("are computed but never used") && w.contains("remove them"),
        "plural ODE message expected: {w}"
    );
    assert!(
        w.contains("[odes]") && !w.contains("pk(...)"),
        "ODE plural warning should reference [odes], not pk(...): {w}"
    );
}

#[test]
fn test_undeclared_name_in_derived_is_accepted_silently() {
    // [derived] uses `fallback_covariate = false`: an unknown identifier
    // becomes a Variable resolved at output time, not a covariate. So an
    // undeclared name used only in [derived] parses without error and without
    // any warning — unlike an ODE RHS (which rejects covariate references), or
    // [individual_parameters] when a [covariates] block is present (strict
    // mode, which warns on an undeclared covariate).
    let src = "
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = 1.0

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[derived]
  RATIO = MYSTERY_NAME / CL

[error_model]
  DV ~ proportional(EPS)
";
    let parsed = super::parse_full_model(src).expect("undeclared name in [derived] should parse");
    assert!(
        !parsed
            .model
            .parse_warnings
            .iter()
            .any(|w| w.contains("MYSTERY_NAME")),
        "undeclared name in [derived] is a silent Variable, no warning: {:?}",
        parsed.model.parse_warnings
    );
}

#[test]
fn test_pk_key_relevance_matrix() {
    // Exhaustively pins every (model, PK key) cell, classified
    // R(equired) / O(ptional, used) / U(nused) — hardcoded independently of
    // `consumes_pk_slot`/`required_pk_params` so the test can't co-drift with
    // the code it checks:
    //   R → omitting the key is an error naming it;
    //   O → mapping the key is silently accepted (no "does not use" warning);
    //   U → mapping the key warns "does not use `key`".
    // Keys use each model's canonical slot name (`v`/`v1`, `q`/`q2`), so the
    // required-omission error (which echoes the canonical name) matches.
    let matrix: &[(&str, [(&str, char); 9])] = &[
        (
            "one_cpt_iv",
            [
                ("cl", 'R'),
                ("v", 'R'),
                ("q", 'U'),
                ("v2", 'U'),
                ("ka", 'U'),
                ("f", 'O'),
                ("q3", 'U'),
                ("v3", 'U'),
                ("lagtime", 'O'),
            ],
        ),
        (
            "one_cpt_oral",
            [
                ("cl", 'R'),
                ("v", 'R'),
                ("q", 'U'),
                ("v2", 'U'),
                ("ka", 'R'),
                ("f", 'O'),
                ("q3", 'U'),
                ("v3", 'U'),
                ("lagtime", 'O'),
            ],
        ),
        (
            "two_cpt_iv",
            [
                ("cl", 'R'),
                ("v1", 'R'),
                ("q", 'R'),
                ("v2", 'R'),
                ("ka", 'U'),
                ("f", 'O'),
                ("q3", 'U'),
                ("v3", 'U'),
                ("lagtime", 'O'),
            ],
        ),
        (
            "two_cpt_oral",
            [
                ("cl", 'R'),
                ("v1", 'R'),
                ("q", 'R'),
                ("v2", 'R'),
                ("ka", 'R'),
                ("f", 'O'),
                ("q3", 'U'),
                ("v3", 'U'),
                ("lagtime", 'O'),
            ],
        ),
        (
            "three_cpt_iv",
            [
                ("cl", 'R'),
                ("v1", 'R'),
                ("q2", 'R'),
                ("v2", 'R'),
                ("ka", 'U'),
                ("f", 'O'),
                ("q3", 'R'),
                ("v3", 'R'),
                ("lagtime", 'O'),
            ],
        ),
        (
            "three_cpt_oral",
            [
                ("cl", 'R'),
                ("v1", 'R'),
                ("q2", 'R'),
                ("v2", 'R'),
                ("ka", 'R'),
                ("f", 'O'),
                ("q3", 'R'),
                ("v3", 'R'),
                ("lagtime", 'O'),
            ],
        ),
    ];
    // `cl` carries the theta/eta so [parameters] has no unused declarations;
    // every other mapped key is a literal so there are no surplus declared
    // individual parameters to flag as dead.
    let build = |model: &str, keys: &[&str]| -> String {
        let indiv: String = keys
            .iter()
            .map(|k| {
                let var = k.to_uppercase();
                if *k == "cl" {
                    format!("  {var} = TVX * exp(ETA)\n")
                } else {
                    format!("  {var} = 1.0\n")
                }
            })
            .collect();
        let pk = keys
            .iter()
            .map(|k| format!("{k}={}", k.to_uppercase()))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "
[parameters]
  theta TVX(1.0, 0.001, 100.0)
  omega ETA ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
{indiv}
[structural_model]
  pk {model}({pk})

[error_model]
  DV ~ proportional(EPS)
"
        )
    };
    for entry in matrix {
        let model = entry.0;
        let cells = &entry.1;
        let required: Vec<&str> = cells.iter().filter(|c| c.1 == 'R').map(|c| c.0).collect();
        for cell in cells {
            let key = cell.0;
            match cell.1 {
                'R' => {
                    // Map every required key EXCEPT this one → error naming it.
                    let keys: Vec<&str> = required.iter().copied().filter(|k| *k != key).collect();
                    let err = super::parse_full_model(&build(model, &keys))
                        .err()
                        .unwrap_or_else(|| panic!("{model}: omitting required `{key}` must error"));
                    assert!(
                        err.contains(&format!("`{key}`")),
                        "{model}: omitting `{key}` should name it, got: {err}"
                    );
                }
                class @ ('O' | 'U') => {
                    // Map every required key PLUS this one → warn iff unused.
                    let mut keys: Vec<&str> = required.clone();
                    keys.push(key);
                    let parsed = super::parse_full_model(&build(model, &keys))
                        .unwrap_or_else(|e| panic!("{model} + `{key}` should parse: {e}"));
                    let warned = parsed
                        .model
                        .parse_warnings
                        .iter()
                        .any(|w| w.contains("does not use") && w.contains(&format!("`{key}`")));
                    assert_eq!(
                        warned,
                        class == 'U',
                        "{model} `{key}` ({class}): unexpected warning state, got: {:?}",
                        parsed.model.parse_warnings
                    );
                }
                other => panic!("bad classification `{other}`"),
            }
        }
    }
}

#[test]
fn test_parse_method_single() {
    let opts = parse_fit_options(&["method = focei".to_string()]).unwrap();
    assert_eq!(opts.method, EstimationMethod::FoceI);
    assert!(opts.methods.is_empty());
    assert!(opts.interaction);
}

#[test]
fn test_parse_inits_from_nca() {
    use crate::suggest_start::NcaInit;
    // Default is off.
    assert_eq!(FitOptions::default().inits_from_nca, None);
    // Boolean true maps to the sweep strategy.
    let opts = parse_fit_options(&["inits_from_nca = true".to_string()]).unwrap();
    assert_eq!(opts.inits_from_nca, Some(NcaInit::Sweep));
    // Boolean false disables.
    let opts = parse_fit_options(&["inits_from_nca = false".to_string()]).unwrap();
    assert_eq!(opts.inits_from_nca, None);
    // Explicit strategy names.
    let opts = parse_fit_options(&["inits_from_nca = nca".to_string()]).unwrap();
    assert_eq!(opts.inits_from_nca, Some(NcaInit::Nca));
    let opts = parse_fit_options(&["inits_from_nca = nca_sweep".to_string()]).unwrap();
    assert_eq!(opts.inits_from_nca, Some(NcaInit::Sweep));
    let opts = parse_fit_options(&["inits_from_nca = nca_ebe".to_string()]).unwrap();
    assert_eq!(opts.inits_from_nca, Some(NcaInit::Ebe));
    // Unknown value is rejected.
    assert!(parse_fit_options(&["inits_from_nca = bogus".to_string()]).is_err());
}

#[test]
fn test_parse_nn_regularization_lambdas() {
    // Default: both off (0.0), a strict no-op for existing fits.
    let d = FitOptions::default();
    assert_eq!(d.nn_l2_lambda, 0.0);
    assert_eq!(d.nn_smooth_lambda, 0.0);

    // Settable from the [fit_options] block (same path as ferx_fit settings).
    let opts =
        parse_fit_options(&["nn_l2 = 1e-2".to_string(), "nn_smooth = 0.25".to_string()]).unwrap();
    assert_eq!(opts.nn_l2_lambda, 1e-2);
    assert_eq!(opts.nn_smooth_lambda, 0.25);

    // Non-negativity guard: negative strengths are rejected.
    assert!(parse_fit_options(&["nn_l2 = -1.0".to_string()]).is_err());
    assert!(parse_fit_options(&["nn_smooth = -0.5".to_string()]).is_err());
    // Non-numeric values are rejected.
    assert!(parse_fit_options(&["nn_l2 = high".to_string()]).is_err());
}

#[test]
fn test_parse_method_chain() {
    let opts = parse_fit_options(&["method = [saem, focei]".to_string()]).unwrap();
    assert_eq!(
        opts.methods,
        vec![EstimationMethod::Saem, EstimationMethod::FoceI]
    );
    assert_eq!(opts.method, EstimationMethod::FoceI);
    assert!(opts.interaction);
}

#[test]
fn test_parse_method_chain_final_foce() {
    let opts = parse_fit_options(&["method = [saem, foce]".to_string()]).unwrap();
    assert_eq!(opts.method, EstimationMethod::Foce);
    assert!(!opts.interaction);
}

#[test]
fn test_method_default_warning() {
    // A model file with no `method` key leaves `user_set_keys` without
    // "method", so the engine warns it defaulted to FOCEI.
    let opts = parse_fit_options(&["covariance = false".to_string()]).unwrap();
    let w = opts.method_default_warning();
    assert!(w.is_some());
    assert!(w.unwrap().contains("FOCEI"));
    // An explicit `method` key suppresses the warning.
    let opts = parse_fit_options(&["method = saem".to_string()]).unwrap();
    assert!(opts.method_default_warning().is_none());
    // Bare default (constructed directly) also warns.
    assert!(FitOptions::default().method_default_warning().is_some());
}

#[test]
fn test_parse_method_chain_empty_rejected() {
    assert!(parse_fit_options(&["method = []".to_string()]).is_err());
}

#[test]
fn test_parse_method_unknown_rejected() {
    assert!(parse_fit_options(&["method = [foce, wibble]".to_string()]).is_err());
}

#[test]
fn test_method_chain_helper_default() {
    let opts = FitOptions::default();
    assert_eq!(opts.method_chain(), vec![EstimationMethod::FoceI]);
}

#[test]
fn test_method_chain_helper_populated() {
    let mut opts = FitOptions::default();
    opts.methods = vec![EstimationMethod::Saem, EstimationMethod::FoceI];
    assert_eq!(
        opts.method_chain(),
        vec![EstimationMethod::Saem, EstimationMethod::FoceI]
    );
}

#[test]
fn test_parse_threads_positive() {
    let opts = parse_fit_options(&["threads = 4".to_string()]).unwrap();
    assert_eq!(opts.threads, Some(4));
}

#[test]
fn test_parse_threads_auto() {
    let opts = parse_fit_options(&["threads = auto".to_string()]).unwrap();
    assert_eq!(opts.threads, None);
    // Case-insensitive.
    let opts = parse_fit_options(&["threads = AUTO".to_string()]).unwrap();
    assert_eq!(opts.threads, None);
}

#[test]
fn test_parse_threads_zero_means_auto() {
    // `threads = 0` is treated as "use the engine's default thread count"
    // (available cores - 1, floored at 1, capped at 8 — #707), matching the
    // R binding's `threads <= 0` sentinel.
    let opts = parse_fit_options(&["threads = 0".to_string()]).unwrap();
    assert_eq!(opts.threads, None);
}

#[test]
fn test_parse_threads_invalid_errors() {
    // Strict parsing: malformed threads values raise a parse error
    // rather than silently falling back to `None` (the pre-refactor
    // `.parse().ok().filter(...)` behavior was a typo trap).
    assert!(parse_fit_options(&["threads = -1".to_string()]).is_err());
    assert!(parse_fit_options(&["threads = wibble".to_string()]).is_err());
}

#[test]
fn test_parse_threads_default_is_none() {
    // No `threads` line → None (engine picks its own default thread count: #707).
    let opts = parse_fit_options(&["method = focei".to_string()]).unwrap();
    assert_eq!(opts.threads, None);
}

// ── mu_referencing fit option ────────────────────────────────────────

#[test]
fn test_parse_mu_referencing_default_true() {
    let opts = parse_fit_options(&["method = foce".to_string()]).unwrap();
    assert!(opts.mu_referencing);
}

#[test]
fn test_parse_mu_referencing_false() {
    let opts = parse_fit_options(&["mu_referencing = false".to_string()]).unwrap();
    assert!(!opts.mu_referencing);
}

#[test]
fn test_parse_mu_referencing_accepts_synonyms() {
    for raw in &["true", "TRUE", "1", "yes", "on"] {
        let opts = parse_fit_options(&[format!("mu_referencing = {}", raw)]).unwrap();
        assert!(opts.mu_referencing, "{} should enable", raw);
    }
    for raw in &["false", "FALSE", "0", "no", "off"] {
        let opts = parse_fit_options(&[format!("mu_referencing = {}", raw)]).unwrap();
        assert!(!opts.mu_referencing, "{} should disable", raw);
    }
}

#[test]
fn test_parse_mu_referencing_invalid_rejected() {
    assert!(parse_fit_options(&["mu_referencing = wibble".to_string()]).is_err());
}

// ── apply_fit_option (shared dispatch used by the R wrapper's `settings`
//    argument and by parse_fit_options) ────────────────────────────────

#[test]
fn test_apply_fit_option_known_applies() {
    let mut opts = FitOptions::default();
    assert_eq!(
        apply_fit_option(&mut opts, "n_exploration", "200"),
        Ok(true)
    );
    assert_eq!(opts.saem_n_exploration, 200);

    assert_eq!(
        apply_fit_option(&mut opts, "n_convergence", "400"),
        Ok(true)
    );
    assert_eq!(opts.saem_n_convergence, 400);

    assert_eq!(apply_fit_option(&mut opts, "omega_burnin", "30"), Ok(true));
    assert_eq!(opts.saem_omega_burnin, 30);
}

/// `mstep_damping` (#1011) round-trips under both spellings, defaults to `None`
/// so the calibrated constant applies, and rejects anything outside `(0, 1]` —
/// `1.0` is the documented "off" value and must stay accepted.
#[test]
fn test_mstep_damping_round_trips_and_validates() {
    let mut opts = FitOptions::default();
    assert_eq!(
        opts.saem_mstep_damping, None,
        "unset must stay None so the MSTEP_SA_MAX_STEP default applies"
    );

    assert_eq!(
        apply_fit_option(&mut opts, "mstep_damping", "0.01"),
        Ok(true)
    );
    assert_eq!(opts.saem_mstep_damping, Some(0.01));

    // The `saem_`-prefixed spelling writes the same field.
    assert_eq!(
        apply_fit_option(&mut opts, "saem_mstep_damping", "0.5"),
        Ok(true)
    );
    assert_eq!(opts.saem_mstep_damping, Some(0.5));

    // 1.0 disables the damping and is in range.
    assert_eq!(
        apply_fit_option(&mut opts, "mstep_damping", "1.0"),
        Ok(true)
    );
    assert_eq!(opts.saem_mstep_damping, Some(1.0));

    for bad in ["0", "-0.1", "1.5"] {
        let err = apply_fit_option(&mut opts, "mstep_damping", bad)
            .expect_err("out-of-range mstep_damping must be rejected");
        assert!(err.contains("(0, 1]"), "got: {err}");
        // The offending value is echoed, as the neighbouring range validators do.
        assert!(err.contains(bad), "value not echoed, got: {err}");
    }
    // The error names the spelling the user actually wrote.
    let err = apply_fit_option(&mut opts, "saem_mstep_damping", "-1")
        .expect_err("out-of-range saem_mstep_damping must be rejected");
    assert!(err.contains("`saem_mstep_damping`"), "got: {err}");
    // A rejected value must not have clobbered the last good one.
    assert_eq!(opts.saem_mstep_damping, Some(1.0));
}

/// The key is advertised as SAEM-specific, so using it under FOCEI warns rather
/// than silently doing nothing.
#[test]
fn test_mstep_damping_under_focei_warns() {
    let opts = parse_fit_options(&[
        "method = focei".to_string(),
        "mstep_damping = 0.01".to_string(),
    ])
    .unwrap();
    let warnings = opts.unsupported_keys_warnings();
    assert!(
        warnings.iter().any(|w| w.contains("mstep_damping")),
        "got: {warnings:?}"
    );
}

/// Conditional-distribution keys (#257) round-trip into FitOptions, both the
/// bare and `saem_`-prefixed spellings of the master switch.
#[test]
fn test_apply_fit_option_conddist_keys() {
    let mut opts = FitOptions::default();
    assert!(!opts.saem_conddist);

    assert_eq!(apply_fit_option(&mut opts, "conddist", "true"), Ok(true));
    assert!(opts.saem_conddist);

    // The `saem_conddist` alias sets the same field.
    opts.saem_conddist = false;
    assert_eq!(
        apply_fit_option(&mut opts, "saem_conddist", "true"),
        Ok(true)
    );
    assert!(opts.saem_conddist);

    assert_eq!(
        apply_fit_option(&mut opts, "conddist_nsamp", "500"),
        Ok(true)
    );
    assert_eq!(opts.saem_conddist_nsamp, 500);

    assert_eq!(
        apply_fit_option(&mut opts, "conddist_burnin", "75"),
        Ok(true)
    );
    assert_eq!(opts.saem_conddist_burnin, 75);

    assert_eq!(
        apply_fit_option(&mut opts, "conddist_keep_samples", "true"),
        Ok(true)
    );
    assert!(opts.saem_conddist_keep_samples);
}

#[test]
fn test_apply_fit_option_unknown_key_returns_false() {
    let mut opts = FitOptions::default();
    // Typo / unknown → Ok(false). Caller decides whether to error out.
    assert_eq!(
        apply_fit_option(&mut opts, "n_exploraton", "200"),
        Ok(false)
    );
    // `method` is deliberately excluded (list-chain syntax is handled
    // in the block parser); treat it as unknown here.
    assert_eq!(apply_fit_option(&mut opts, "method", "focei"), Ok(false));
}

#[test]
fn test_apply_fit_option_malformed_value_errors() {
    let mut opts = FitOptions::default();
    assert!(apply_fit_option(&mut opts, "n_exploration", "oops").is_err());
    assert!(apply_fit_option(&mut opts, "covariance", "maybe").is_err());
    assert!(apply_fit_option(&mut opts, "gn_lambda", "x").is_err());
    assert!(apply_fit_option(&mut opts, "optimizer", "does_not_exist").is_err());
    assert!(apply_fit_option(&mut opts, "bloq_method", "nope").is_err());
    assert!(apply_fit_option(&mut opts, "threads", "-1").is_err());
    assert!(apply_fit_option(&mut opts, "sir_df", "0.0").is_err());
    assert!(apply_fit_option(&mut opts, "sir_df", "0.5").is_err());
    // Failed apply must not mutate — default preserved.
    assert_eq!(opts.saem_n_exploration, 150);
}

#[test]
fn test_apply_fit_option_ode_solver_tolerances() {
    let mut opts = FitOptions::default();
    // Defaults match OdeSolverOptions::default() (engine default unchanged).
    assert_eq!(opts.ode_reltol, 1e-4);
    assert_eq!(opts.ode_abstol, 1e-6);
    assert_eq!(opts.ode_max_steps, 10_000);

    assert_eq!(apply_fit_option(&mut opts, "ode_reltol", "1e-10"), Ok(true));
    assert_eq!(apply_fit_option(&mut opts, "ode_abstol", "1e-12"), Ok(true));
    assert_eq!(
        apply_fit_option(&mut opts, "ode_max_steps", "200000"),
        Ok(true)
    );
    assert_eq!(opts.ode_reltol, 1e-10);
    assert_eq!(opts.ode_abstol, 1e-12);
    assert_eq!(opts.ode_max_steps, 200_000);

    // Non-positive / non-finite / zero are rejected, and a failed apply
    // must not mutate the previously-set value.
    assert!(apply_fit_option(&mut opts, "ode_reltol", "0").is_err());
    assert!(apply_fit_option(&mut opts, "ode_reltol", "-1e-9").is_err());
    assert!(apply_fit_option(&mut opts, "ode_abstol", "nan").is_err());
    assert!(apply_fit_option(&mut opts, "ode_max_steps", "0").is_err());
    assert!(apply_fit_option(&mut opts, "ode_max_steps", "x").is_err());
    assert_eq!(opts.ode_reltol, 1e-10);
    assert_eq!(opts.ode_max_steps, 200_000);
}

#[test]
fn test_apply_fit_option_ode_method() {
    use crate::ode::OdeMethod;
    let mut opts = FitOptions::default();
    // The engine default is the probe (#978): a model that names no stepper gets `auto`, which
    // keeps the explicit method on everything that is not stability-limited.
    assert_eq!(opts.ode_method, OdeMethod::Auto);

    assert_eq!(apply_fit_option(&mut opts, "ode_method", "rk45"), Ok(true));
    assert_eq!(opts.ode_method, OdeMethod::Rk45);
    assert_eq!(apply_fit_option(&mut opts, "ode_method", "auto"), Ok(true));
    assert_eq!(opts.ode_method, OdeMethod::Auto);

    assert_eq!(
        apply_fit_option(&mut opts, "ode_method", "rodas5p"),
        Ok(true)
    );
    assert_eq!(opts.ode_method, OdeMethod::Rodas5P);
    assert_eq!(
        apply_fit_option(&mut opts, "ode_method", "ode23s"),
        Ok(true)
    );
    assert_eq!(opts.ode_method, OdeMethod::Rosenbrock23);
    assert_eq!(
        apply_fit_option(&mut opts, "ode_method", "Rodas4"),
        Ok(true)
    );
    assert_eq!(opts.ode_method, OdeMethod::Rodas4);

    // An unknown stepper is an error, not a silent fallback to RK45 (which would run the
    // stiff model on the method the user was trying to get away from).
    let err = apply_fit_option(&mut opts, "ode_method", "lsoda").unwrap_err();
    assert!(err.contains("ode_method"), "unexpected message: {err}");
    assert_eq!(
        opts.ode_method,
        OdeMethod::Rodas4,
        "failed apply must not mutate"
    );
}

/// `ode_stiff_abort_after` (#708 / #1080 Part B): opt-in, disable-able without deleting the
/// key, and framework-level so it is not reported as ignored.
#[test]
fn test_apply_fit_option_ode_stiff_abort_after() {
    let mut opts = FitOptions::default();
    assert_eq!(opts.ode_stiff_abort_after, None, "must be opt-in");

    assert_eq!(
        apply_fit_option(&mut opts, "ode_stiff_abort_after", "25"),
        Ok(true)
    );
    assert_eq!(opts.ode_stiff_abort_after, Some(25));

    // `0` / `off` / `none` turn the budget back off, so a settings list can disable it
    // without removing the key.
    for off in ["0", "off", "none", "false"] {
        opts.ode_stiff_abort_after = Some(25);
        assert_eq!(
            apply_fit_option(&mut opts, "ode_stiff_abort_after", off),
            Ok(true)
        );
        assert_eq!(opts.ode_stiff_abort_after, None, "{off}");
    }

    assert!(apply_fit_option(&mut opts, "ode_stiff_abort_after", "x").is_err());
}

/// The budget reaches the integrator the same way the tolerances do — through
/// `sync_ode_solver_opts` at parse time — so `predict()`, which gets no fit options, honours
/// it too.
#[test]
fn test_ode_stiff_abort_after_reaches_ode_spec() {
    let src = r#"
[parameters]
  theta TVCL(1.0, 0.01, 10.0)
  theta TVV(10.0, 0.1, 100.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_stiff_abort_after = 40
"#;
    let p = parse_full_model(src).unwrap();
    assert_eq!(
        p.model
            .ode_spec
            .as_ref()
            .unwrap()
            .solver_opts
            .stiff_abort_after,
        Some(40)
    );
    assert!(!p
        .fit_options
        .unsupported_keys_warnings()
        .iter()
        .any(|w| w.contains("ode_stiff_abort_after")));
}

/// `ode_auto_switch` (#1080 Part C): on by default, turn-off-able without deleting the key,
/// framework-level, and carried onto the solver options the integrator actually reads.
#[test]
fn test_apply_fit_option_ode_auto_switch() {
    let mut opts = FitOptions::default();
    assert!(opts.ode_auto_switch, "mid-segment switching is the default");

    assert_eq!(
        apply_fit_option(&mut opts, "ode_auto_switch", "false"),
        Ok(true)
    );
    assert!(!opts.ode_auto_switch);
    assert_eq!(
        apply_fit_option(&mut opts, "ode_auto_switch", "true"),
        Ok(true)
    );
    assert!(opts.ode_auto_switch);
    assert!(apply_fit_option(&mut opts, "ode_auto_switch", "sometimes").is_err());
}

#[test]
fn test_ode_auto_switch_reaches_ode_spec() {
    let src = r#"
[parameters]
  theta TVCL(1.0, 0.01, 10.0)
  theta TVV(10.0, 0.1, 100.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_auto_switch = false
"#;
    let p = parse_full_model(src).unwrap();
    assert!(!p.model.ode_spec.as_ref().unwrap().solver_opts.auto_switch);
    assert!(!p
        .fit_options
        .unsupported_keys_warnings()
        .iter()
        .any(|w| w.contains("ode_auto_switch")));
}

#[test]
fn test_ode_reltol_from_fit_options_reaches_ode_spec() {
    // [fit_options] ODE solver tolerances must be baked onto
    // OdeSpec.solver_opts by the parser (via sync_ode_solver_opts) so the
    // integrator - including predict(), which receives no fit options -
    // uses the requested accuracy.
    let base = r#"
[parameters]
  theta TVCL(1.0, 0.01, 10.0)
  theta TVV(10.0, 0.1, 100.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    // Default: OdeSpec inherits OdeSolverOptions::default().
    let def = parse_full_model(base).unwrap();
    let s = def.model.ode_spec.as_ref().unwrap().solver_opts;
    assert_eq!(s.reltol, 1e-4);
    assert_eq!(s.abstol, 1e-6);
    assert_eq!(s.max_steps, 10_000);

    assert_eq!(s.method, crate::ode::OdeMethod::Auto);

    // Override via [fit_options].
    let with_opts = format!(
            "{base}\n[fit_options]\n  ode_reltol = 1e-9\n  ode_abstol = 1e-11\n  ode_max_steps = 50000\n  ode_method = rodas4\n"
        );
    let p = parse_full_model(&with_opts).unwrap();
    let s2 = p.model.ode_spec.as_ref().unwrap().solver_opts;
    assert_eq!(s2.reltol, 1e-9);
    assert_eq!(s2.abstol, 1e-11);
    assert_eq!(s2.max_steps, 50_000);
    // `ode_method` rides the same sync, so `predict()` — which gets no fit options — also
    // integrates with the requested stepper.
    assert_eq!(s2.method, crate::ode::OdeMethod::Rodas4);
    // …and it is framework-level, so it must not be flagged as an ignored key.
    assert!(!p
        .fit_options
        .unsupported_keys_warnings()
        .iter()
        .any(|w| w.contains("ode_method")));
}

#[test]
fn test_imp_method_and_imp_options_parse() {
    // `methods = [focei, imp]` plus the `imp_*` keys must apply
    // cleanly and produce no `unsupported_keys_warnings`.
    let opts = parse_fit_options(&[
        "method = [focei, imp]".to_string(),
        "imp_samples = 500".to_string(),
        "imp_proposal_df = 4.0".to_string(),
        "imp_seed = 99".to_string(),
        "imp_low_ess_threshold = 0.2".to_string(),
        "covariance = false".to_string(),
        "verbose = false".to_string(),
    ])
    .expect("parse must succeed");
    assert_eq!(
        opts.methods,
        vec![EstimationMethod::FoceI, EstimationMethod::Imp]
    );
    assert_eq!(opts.imp_samples, 500);
    assert_eq!(opts.imp_proposal_df, 4.0);
    assert_eq!(opts.imp_seed, Some(99));
    assert_eq!(opts.imp_low_ess_threshold, 0.2);
    // No "ignored option" warnings — keys are method-specific to Imp,
    // and Imp is in the chain.
    assert!(opts.unsupported_keys_warnings().is_empty());
}

#[test]
fn test_imp_options_validate_ranges() {
    let mut opts = FitOptions::default();
    assert!(apply_fit_option(&mut opts, "imp_samples", "1").is_err()); // < 2
    assert!(apply_fit_option(&mut opts, "imp_proposal_df", "0.5").is_err()); // < 1
    assert!(apply_fit_option(&mut opts, "imp_low_ess_threshold", "1.5").is_err()); // > 1
    assert!(apply_fit_option(&mut opts, "imp_low_ess_threshold", "-0.1").is_err()); // < 0
    assert!(apply_fit_option(&mut opts, "imp_iterations", "0").is_err()); // < 1
                                                                          // Defaults preserved after a failed apply.
    assert_eq!(opts.imp_samples, 1000);
}

#[test]
fn test_imp_estimator_options_parse() {
    // The estimating-IMP controls and the eval-only switch apply cleanly.
    let opts = parse_fit_options(&[
        "method = imp".to_string(),
        "imp_iterations = 80".to_string(),
        "imp_averaging = 20".to_string(),
        "imp_eval_only = true".to_string(),
        "imp_proposal_df = normal".to_string(),
    ])
    .expect("parse must succeed");
    assert_eq!(opts.method, EstimationMethod::Imp);
    assert_eq!(opts.imp_iterations, 80);
    assert_eq!(opts.imp_averaging, 20);
    assert!(opts.imp_eval_only);
    assert!(opts.imp_proposal_df.is_infinite());
    assert!(opts.unsupported_keys_warnings().is_empty());
}

#[test]
fn test_imp_proposal_df_accepts_normal_token() {
    for kw in ["normal", "mvn", "NORMAL"] {
        let mut o = FitOptions::default();
        assert!(apply_fit_option(&mut o, "imp_proposal_df", kw).is_ok());
        assert!(o.imp_proposal_df.is_infinite(), "`{kw}` must select MVN");
    }
}

#[test]
fn test_imp_method_token_defaults_to_estimator() {
    // `imp` alone must not flip the eval-only switch — the default is the
    // NONMEM METHOD=IMP estimator.
    let opts = parse_fit_options(&["method = imp".to_string()]).expect("parse must succeed");
    assert_eq!(opts.method, EstimationMethod::Imp);
    assert!(!opts.imp_eval_only);
}

#[test]
fn test_legacy_is_prefixed_imp_options_are_not_accepted() {
    for key in [
        "is_samples",
        "is_proposal_df",
        "is_seed",
        "is_low_ess_threshold",
        "is_iterations",
        "is_averaging",
        "is_eval_only",
        "is_auto",
    ] {
        let mut opts = FitOptions::default();
        assert_eq!(
            apply_fit_option(&mut opts, key, "1"),
            Ok(false),
            "`{key}` must not remain a parser alias for the renamed `imp_*` options"
        );
    }
}

#[test]
fn test_impmap_method_tokens_parse() {
    for tok in [
        "impmap",
        "importance_sampling_map",
        "importance-sampling-map",
    ] {
        let opts = parse_fit_options(&[format!("method = {tok}")]).expect("parse must succeed");
        assert_eq!(
            opts.method,
            EstimationMethod::Impmap,
            "token `{tok}` must map to Impmap"
        );
    }
    // `imp` must still resolve to the evaluation-only stage, not Impmap.
    let opts = parse_fit_options(&["method = imp".to_string()]).expect("parse");
    assert_eq!(opts.method, EstimationMethod::Imp);
}

#[test]
fn test_impmap_options_parse_and_validate() {
    let opts = parse_fit_options(&[
        "method = importance_sampling_map".to_string(),
        "impmap_iterations = 80".to_string(),
        "impmap_samples = 250".to_string(),
        "impmap_proposal_df = 8.0".to_string(),
        "impmap_averaging = 30".to_string(),
        "impmap_seed = 77".to_string(),
        "impmap_low_ess_threshold = 0.2".to_string(),
        "impmap_mceta = 3".to_string(),
        "covariance = false".to_string(),
    ])
    .expect("parse must succeed");
    assert_eq!(opts.method, EstimationMethod::Impmap);
    assert_eq!(opts.impmap_iterations, 80);
    assert_eq!(opts.impmap_samples, 250);
    assert_eq!(opts.impmap_proposal_df, 8.0);
    assert_eq!(opts.impmap_averaging, 30);
    assert_eq!(opts.impmap_seed, Some(77));
    assert_eq!(opts.impmap_low_ess_threshold, 0.2);
    assert_eq!(opts.impmap_mceta, 3);
    // All keys are method-specific to Impmap and Impmap is selected.
    assert!(opts.unsupported_keys_warnings().is_empty());

    // `normal` / `mvn` select the multivariate-normal proposal (df = +inf).
    for kw in ["normal", "mvn", "NORMAL"] {
        let mut o = FitOptions::default();
        assert!(apply_fit_option(&mut o, "impmap_proposal_df", kw).is_ok());
        assert!(o.impmap_proposal_df.is_infinite());
    }

    // Range validation.
    let mut o = FitOptions::default();
    assert!(apply_fit_option(&mut o, "impmap_samples", "1").is_err()); // < 2
    assert!(apply_fit_option(&mut o, "impmap_iterations", "0").is_err()); // < 1
    assert!(apply_fit_option(&mut o, "impmap_proposal_df", "0.5").is_err()); // < 1
    assert!(apply_fit_option(&mut o, "impmap_low_ess_threshold", "1.5").is_err()); // > 1
                                                                                   // Default (Student-t, df=4) preserved after the failed applies.
    assert_eq!(o.impmap_proposal_df, 4.0);
}

#[test]
fn test_sir_df_valid_and_invalid() {
    let mut opts = FitOptions::default();
    assert!(apply_fit_option(&mut opts, "sir_df", "5.0").is_ok());
    assert_eq!(opts.sir_df, 5.0);
    assert!(apply_fit_option(&mut opts, "sir_df", "1.0").is_ok());
    assert!(apply_fit_option(&mut opts, "sir_df", "0.9").is_err());
    assert!(apply_fit_option(&mut opts, "sir_df", "0.0").is_err());
}

#[test]
fn test_apply_fit_option_bool_variants() {
    let mut opts = FitOptions::default();
    for v in ["true", "True", "TRUE", "yes", "1", "t"] {
        opts.sir = false;
        assert_eq!(apply_fit_option(&mut opts, "sir", v), Ok(true));
        assert!(opts.sir, "value `{v}` should parse as true");
    }
    for v in ["false", "False", "no", "0", "f"] {
        opts.sir = true;
        assert_eq!(apply_fit_option(&mut opts, "sir", v), Ok(true));
        assert!(!opts.sir, "value `{v}` should parse as false");
    }
}

#[test]
fn test_apply_fit_option_adaptive_sampling_flags() {
    let mut opts = FitOptions::default();
    assert!(opts.imp_auto && opts.impmap_auto, "default on");
    assert_eq!(apply_fit_option(&mut opts, "imp_auto", "false"), Ok(true));
    assert!(!opts.imp_auto);
    assert_eq!(apply_fit_option(&mut opts, "impmap_auto", "no"), Ok(true));
    assert!(!opts.impmap_auto);
}

#[test]
fn test_apply_fit_option_seed_null_clears() {
    let mut opts = FitOptions::default();
    opts.saem_seed = Some(7);
    // R sends NULL/NA through as the literal "null" / "na".
    assert_eq!(apply_fit_option(&mut opts, "seed", "null"), Ok(true));
    assert_eq!(opts.saem_seed, None);

    assert_eq!(apply_fit_option(&mut opts, "seed", "42"), Ok(true));
    assert_eq!(opts.saem_seed, Some(42));

    // `saem_seed` is accepted as an alias so R users can use either spelling.
    assert_eq!(apply_fit_option(&mut opts, "saem_seed", "99"), Ok(true));
    assert_eq!(opts.saem_seed, Some(99));
}

#[test]
fn test_apply_fit_option_npde() {
    let mut opts = FitOptions::default();
    assert_eq!(opts.npde_nsim, 0, "NPDE is off by default");
    assert_eq!(apply_fit_option(&mut opts, "npde_nsim", "1000"), Ok(true));
    assert_eq!(opts.npde_nsim, 1000);
    assert_eq!(apply_fit_option(&mut opts, "npde_seed", "12345"), Ok(true));
    assert_eq!(opts.npde_seed, Some(12345));
    // NULL/NA from R clears the seed back to the default.
    assert_eq!(apply_fit_option(&mut opts, "npde_seed", "null"), Ok(true));
    assert_eq!(opts.npde_seed, None);
}

#[test]
fn test_apply_fit_option_threads_variants() {
    let mut opts = FitOptions::default();
    assert_eq!(apply_fit_option(&mut opts, "threads", "4"), Ok(true));
    assert_eq!(opts.threads, Some(4));

    assert_eq!(apply_fit_option(&mut opts, "threads", "auto"), Ok(true));
    assert_eq!(opts.threads, None);

    opts.threads = Some(4);
    assert_eq!(apply_fit_option(&mut opts, "threads", "0"), Ok(true));
    assert_eq!(opts.threads, None);
}

#[test]
fn test_apply_fit_option_optimizer_and_bloq() {
    let mut opts = FitOptions::default();
    // `lbfgs` and `bfgs` are deprecated aliases for `nlopt_lbfgs` — all three
    // resolve to the NLopt L-BFGS (+ SLSQP polish) path. See #483.
    assert_eq!(apply_fit_option(&mut opts, "optimizer", "lbfgs"), Ok(true));
    assert_eq!(opts.optimizer, Optimizer::NloptLbfgs);
    assert_eq!(apply_fit_option(&mut opts, "optimizer", "bfgs"), Ok(true));
    assert_eq!(opts.optimizer, Optimizer::NloptLbfgs);
    assert_eq!(
        apply_fit_option(&mut opts, "optimizer", "nlopt_lbfgs"),
        Ok(true)
    );
    assert_eq!(opts.optimizer, Optimizer::NloptLbfgs);

    assert_eq!(apply_fit_option(&mut opts, "bloq", "m3"), Ok(true));
    assert_eq!(opts.bloq_method, BloqMethod::M3);
}

#[test]
fn test_apply_fit_option_inner_optimizer() {
    use crate::types::InnerOptimizer;
    let mut opts = FitOptions::default();
    // Default is Auto (size-based dispatch).
    assert_eq!(opts.inner_optimizer, InnerOptimizer::Auto);
    for (s, want) in [
        ("auto", InnerOptimizer::Auto),
        ("bfgs", InnerOptimizer::Bfgs),
        ("lbfgs", InnerOptimizer::Lbfgs),
        ("nelder_mead", InnerOptimizer::NelderMead),
        ("neldermead", InnerOptimizer::NelderMead),
    ] {
        assert_eq!(apply_fit_option(&mut opts, "inner_optimizer", s), Ok(true));
        assert_eq!(opts.inner_optimizer, want, "inner_optimizer = {s}");
    }
    assert!(apply_fit_option(&mut opts, "inner_optimizer", "nope").is_err());
}

// ── Warn on options that don't apply to the selected estimation method.
//    These fire from inside fit() via FitOptions::unsupported_keys_warnings,
//    so we check the raw mechanism here without running a full fit. ────

#[test]
fn test_unsupported_saem_key_under_focei_warns() {
    let opts = parse_fit_options(&[
        "method = focei".to_string(),
        "n_convergence = 300".to_string(),
    ])
    .unwrap();
    let warnings = opts.unsupported_keys_warnings();
    assert_eq!(warnings.len(), 1, "got: {:?}", warnings);
    let w = &warnings[0];
    assert!(w.contains("n_convergence"), "got: {w}");
    assert!(w.contains("FOCEI"), "got: {w}");
    assert!(w.contains("will be ignored"), "got: {w}");
    // Mentions a FOCE-applicable key so the user can see what's available.
    assert!(w.contains("optimizer"), "got: {w}");
    // Does NOT suggest SAEM-specific keys as available.
    assert!(!w.contains("n_mh_steps"), "got: {w}");
}

#[test]
fn test_unsupported_focei_key_under_saem_warns() {
    let opts =
        parse_fit_options(&["method = saem".to_string(), "optimizer = lbfgs".to_string()]).unwrap();
    let warnings = opts.unsupported_keys_warnings();
    assert_eq!(warnings.len(), 1, "got: {:?}", warnings);
    let w = &warnings[0];
    assert!(w.contains("optimizer"), "got: {w}");
    assert!(w.contains("SAEM"), "got: {w}");
    assert!(w.contains("n_exploration"), "got: {w}");
}

#[test]
fn test_applicable_key_in_chain_no_warning() {
    // methods = [saem, focei]: n_convergence applies to SAEM, optimizer
    // applies to FOCEI, so neither should warn.
    let opts = parse_fit_options(&[
        "method = [saem, focei]".to_string(),
        "n_convergence = 300".to_string(),
        "optimizer = lbfgs".to_string(),
    ])
    .unwrap();
    assert!(opts.unsupported_keys_warnings().is_empty());
}

#[test]
fn test_common_keys_never_warn() {
    // Covariance/verbose/sir/bloq/threads/mu_referencing apply to every
    // method — they must not produce a warning regardless of method.
    for method in ["foce", "focei", "gn", "gn_hybrid", "saem"] {
        let opts = parse_fit_options(&[
            format!("method = {method}"),
            "covariance = true".to_string(),
            // The whole covariance family is framework-wide — none of these
            // are method-specific, so none must warn under any method.
            "covariance_method = s".to_string(),
            "covariance_fallback = sir".to_string(),
            "verbose = false".to_string(),
            "sir = true".to_string(),
            "bloq_method = m3".to_string(),
            "threads = 2".to_string(),
            "mu_referencing = false".to_string(),
        ])
        .unwrap();
        let w = opts.unsupported_keys_warnings();
        assert!(
            w.is_empty(),
            "method={method} produced unexpected warnings: {:?}",
            w
        );
    }
}

#[test]
fn test_unsupported_warning_omits_framework_keys() {
    // Framework-wide keys (covariance/verbose/sir/bloq/threads/mu_referencing)
    // are exposed as top-level wrapper args, not as method-specific settings.
    // The warning's "Method-specific options" list must not include them —
    // listing `covariance` next to `optimizer` would conflate the layers.
    let opts = parse_fit_options(&[
        "method = focei".to_string(),
        "n_convergence = 300".to_string(),
    ])
    .unwrap();
    let w = &opts.unsupported_keys_warnings()[0];
    for framework in [
        "covariance",
        "verbose",
        "sir",
        "sir_samples",
        "sir_resamples",
        "sir_seed",
        "sir_keep_samples",
        "bloq_method",
        "bloq",
        "threads",
        "mu_referencing",
    ] {
        assert!(
            !w.contains(framework),
            "framework key `{framework}` leaked into method-specific list: {w}"
        );
    }
    // And it uses the new phrasing, not the old "Available options".
    assert!(w.contains("Method-specific options"), "got: {w}");
}

/// Every advertised fit-option key must actually be reachable from a `.ferx` file.
///
/// `framework_keys()` / `method_specific_keys()` are what the "unsupported option" warning
/// and the wrapper docs advertise; `apply_fit_option` is what the block parser dispatches
/// on, and its `_ => Ok(false)` arm turns anything it does not recognise into a hard
/// ``unknown key`` parse error. Nothing tied the two together, so `analytic_cov_hessian`
/// shipped in the first list and not the second: the documented escape hatch back to the
/// finite-difference covariance aborted the fit at parse time and was reachable only from
/// Rust (PR #953 review finding 7).
///
/// Keys are string-typed, so the test tries a small battery of values and only requires
/// that *some* value is accepted — enough to prove the arm exists without duplicating each
/// key's value grammar here.
#[test]
fn every_advertised_fit_option_key_has_an_apply_fit_option_arm() {
    use crate::types::{framework_keys, method_specific_keys, EstimationMethod};

    // Values chosen to cover the parser's shapes: bool, integer, float, enum-ish, free text.
    const CANDIDATES: &[&str] = &[
        "true", "1", "0.5", "1e-3", "auto", "column", "", "r", "none", "focei",
    ];

    let mut keys: Vec<&'static str> = framework_keys().to_vec();
    for m in [
        EstimationMethod::Foce,
        EstimationMethod::FoceI,
        EstimationMethod::FoceGn,
        EstimationMethod::FoceGnHybrid,
        EstimationMethod::Saem,
        EstimationMethod::Imp,
        EstimationMethod::Impmap,
        EstimationMethod::Bayes,
        EstimationMethod::Laplace,
    ] {
        keys.extend(method_specific_keys(m).iter().copied());
    }
    keys.sort_unstable();
    keys.dedup();

    let mut unreachable: Vec<&str> = Vec::new();
    for key in keys {
        let recognised = CANDIDATES.iter().any(|v| {
            let mut opts = FitOptions::default();
            // `Ok(true)` = dispatched. `Err` also proves the arm exists (the value was
            // rejected by *that key's* parser); only `Ok(false)` means no arm at all.
            !matches!(apply_fit_option(&mut opts, key, v), Ok(false))
        });
        if !recognised {
            unreachable.push(key);
        }
    }
    assert!(
        unreachable.is_empty(),
        "advertised fit-option key(s) with no `apply_fit_option` arm — setting these in a \
         `.ferx` file is a hard parse error: {unreachable:?}"
    );
}

#[test]
fn test_ode_solver_keys_do_not_warn() {
    // ode_reltol/ode_abstol/ode_max_steps configure the RK45 integrator (any
    // ODE model, any estimation method) — they are framework-level, not
    // method-specific. Regression for the spurious "is not used by method
    // FOCEI and will be ignored" warning these once produced even though
    // `sync_ode_solver_opts` does apply them.
    for method in ["focei", "foce", "saem", "imp", "bayes"] {
        let opts = parse_fit_options(&[
            format!("method = {method}"),
            "ode_reltol = 1e-9".to_string(),
            "ode_abstol = 1e-11".to_string(),
            "ode_max_steps = 1000000".to_string(),
        ])
        .unwrap();
        assert!(
            opts.unsupported_keys_warnings().is_empty(),
            "method={method} spuriously warned on ODE-solver keys: {:?}",
            opts.unsupported_keys_warnings()
        );
    }
}

#[test]
fn test_checkpoint_keys_parse_and_do_not_warn() {
    // checkpoint / checkpoint_interval_secs drive the resume-point writer,
    // which is method-independent (#755) — framework-level, so they must
    // parse and never be flagged "not used by method".
    for method in ["focei", "foce", "saem", "imp", "gn"] {
        let opts = parse_fit_options(&[
            format!("method = {method}"),
            "checkpoint = false".to_string(),
            "checkpoint_interval_secs = 60".to_string(),
        ])
        .unwrap();
        assert!(!opts.checkpoint, "method={method}: checkpoint should parse");
        assert_eq!(opts.checkpoint_interval_secs, 60);
        assert!(
            opts.unsupported_keys_warnings().is_empty(),
            "method={method} spuriously warned on checkpoint keys: {:?}",
            opts.unsupported_keys_warnings()
        );
    }
}

#[test]
fn test_nn_regularization_keys_do_not_warn_on_foce_family() {
    // nn_l2 / nn_smooth are applied by every FOCE-family outer optimizer
    // (foce, focei, laplace, gn, gn_hybrid — NLopt, built-in BFGS, trust-region
    // and Gauss–Newton all add the penalty), so they must never be flagged "not
    // used by method" there. Regression: they were parsed but listed in no
    // `method_specific_keys` arm, so every regularized fit — including one from
    // ferx-r's `settings = list(nn_l2 = ...)`, which routes through the same
    // `apply_fit_option` — reported them as ignored.
    for method in ["focei", "foce", "laplace", "gn", "gn_hybrid"] {
        let opts = parse_fit_options(&[
            format!("method = {method}"),
            "nn_l2 = 1e-2".to_string(),
            "nn_smooth = 1e-1".to_string(),
        ])
        .unwrap();
        assert_eq!(opts.nn_l2_lambda, 1e-2);
        assert_eq!(opts.nn_smooth_lambda, 1e-1);
        assert!(
            opts.unsupported_keys_warnings().is_empty(),
            "method={method} spuriously warned on nn_l2/nn_smooth: {:?}",
            opts.unsupported_keys_warnings()
        );
    }
    // ...and they *are* flagged where nothing applies them, so a user asking
    // for a regularized SAEM fit hears about it rather than getting an
    // unregularized one silently.
    let saem =
        parse_fit_options(&["method = saem".to_string(), "nn_l2 = 1e-2".to_string()]).unwrap();
    let w = saem.unsupported_keys_warnings();
    assert_eq!(w.len(), 1, "saem must warn once on nn_l2: {w:?}");
    assert!(
        w[0].contains("`nn_l2` is not used by method `SAEM`"),
        "{w:?}"
    );
}

#[test]
fn test_inner_optimizer_under_focei_does_not_warn() {
    // `inner_optimizer` drives the per-subject EBE loop, which FOCEI uses —
    // it must be in the method's recognized keys and not flagged "ignored".
    let opts = parse_fit_options(&[
        "method = focei".to_string(),
        "inner_optimizer = lbfgs".to_string(),
    ])
    .unwrap();
    assert_eq!(opts.inner_optimizer, crate::types::InnerOptimizer::Lbfgs);
    let warnings = opts.unsupported_keys_warnings();
    assert!(
        !warnings.iter().any(|w| w.contains("inner_optimizer")),
        "inner_optimizer should not warn under FOCEI, got: {warnings:?}"
    );
}

#[test]
fn test_gn_lambda_under_focei_warns() {
    let opts =
        parse_fit_options(&["method = focei".to_string(), "gn_lambda = 0.05".to_string()]).unwrap();
    let warnings = opts.unsupported_keys_warnings();
    assert_eq!(warnings.len(), 1, "got: {:?}", warnings);
    assert!(warnings[0].contains("gn_lambda"));
}

#[test]
fn test_no_warning_when_no_keys_set() {
    // Bare default FitOptions (no parser path) must not conjure warnings.
    let opts = FitOptions::default();
    assert!(opts.unsupported_keys_warnings().is_empty());
}

#[test]
fn test_stagnation_guard_recognized_by_nlopt_methods() {
    // `stagnation_guard` lives in the NLopt outer-loop code, so the
    // unused-key guard must accept it under FOCE/FOCEI and the FoceGn
    // hybrid (which has a FOCEI polish phase) — and must flag it under
    // pure GN and SAEM, which don't touch the NLopt path.
    for method in ["foce", "focei", "gn_hybrid"] {
        let opts = parse_fit_options(&[
            format!("method = {}", method),
            "stagnation_guard = false".to_string(),
        ])
        .unwrap();
        let warnings = opts.unsupported_keys_warnings();
        assert!(
            warnings.is_empty(),
            "method={method} should not warn for stagnation_guard: {:?}",
            warnings
        );
    }
    for method in ["gn", "saem"] {
        let opts = parse_fit_options(&[
            format!("method = {}", method),
            "stagnation_guard = false".to_string(),
        ])
        .unwrap();
        let warnings = opts.unsupported_keys_warnings();
        assert_eq!(
            warnings.len(),
            1,
            "method={method}: expected exactly one warning, got: {:?}",
            warnings
        );
        assert!(warnings[0].contains("stagnation_guard"));
    }
}

#[test]
fn test_ebe_warm_start_parses_and_is_framework_key() {
    // Parses to the bool, defaults to false (opt-in).
    let opts = parse_fit_options(&[
        "method = focei".to_string(),
        "ebe_warm_start = true".to_string(),
    ])
    .unwrap();
    assert!(opts.ebe_warm_start);
    assert!(!FitOptions::default().ebe_warm_start, "default is off");
    // Framework key: the inner NM fallback exists under every method, so it
    // must not warn as unsupported for any of them.
    for method in ["foce", "focei", "gn", "saem"] {
        let opts = parse_fit_options(&[
            format!("method = {method}"),
            "ebe_warm_start = true".to_string(),
        ])
        .unwrap();
        assert!(
            opts.unsupported_keys_warnings().is_empty(),
            "method={method} should accept the framework key ebe_warm_start"
        );
    }
}

// ── parse_fit_options: strict parsing at the .ferx layer. Unknown
//    keys and malformed values both raise an error — a typo like
//    `covariance = maybe` or `bloq_method = nope` now fails loudly
//    instead of silently landing on an unexpected default. ───────────

#[test]
fn test_parse_fit_options_unknown_key_errors() {
    let err = parse_fit_options(&["n_exploraton = 200".to_string()]).unwrap_err();
    assert!(err.contains("unknown key"), "got: {err}");
    assert!(err.contains("n_exploraton"), "got: {err}");
}

#[test]
fn test_parse_fit_options_covariance_ofv_hessian_removed() {
    // The `covariance_ofv_hessian` option was removed in #639 (the analytical-
    // gradient covariance R stencil it selected is gone; the reconverged-OFV
    // second difference is the sole stencil). An old model file that still sets
    // it must now fail loudly as an unknown key, not silently be accepted.
    let err = parse_fit_options(&["covariance_ofv_hessian = false".to_string()]).unwrap_err();
    assert!(err.contains("unknown key"), "got: {err}");
    assert!(err.contains("covariance_ofv_hessian"), "got: {err}");
}

#[test]
fn test_parse_fit_options_malformed_numeric_errors() {
    assert!(parse_fit_options(&["n_exploration = oops".to_string()]).is_err());
}

#[test]
fn test_parse_fit_options_malformed_bool_errors() {
    // Pre-refactor, `covariance = maybe` silently coerced to `false`
    // via `== "true"`, flipping the default. Now it errors.
    assert!(parse_fit_options(&["covariance = maybe".to_string()]).is_err());
}

#[test]
fn test_parse_fit_options_uppercase_bool_accepted() {
    // Pre-refactor, `covariance = TRUE` silently became `false`
    // because the inline check only matched lowercase "true". The
    // strict parser accepts common casing variants.
    let opts = parse_fit_options(&["covariance = TRUE".to_string()]).unwrap();
    assert!(opts.run_covariance_step);
}

#[test]
fn test_parse_fit_options_bloq_method_typo_errors() {
    // `bloq_method` was already strict in the old inline parser; the
    // new strict dispatch must preserve that (not silently default).
    assert!(parse_fit_options(&["bloq_method = nope".to_string()]).is_err());
}

#[test]
fn test_parse_fit_options_gradient_method() {
    // Accepted aliases resolve to the expected GradientMethod variant.
    for (input, expected) in [
        ("gradient = auto", GradientMethod::Auto),
        ("gradient = ad", GradientMethod::Ad),
        ("gradient = autodiff", GradientMethod::Ad),
        ("gradient = fd", GradientMethod::Fd),
        ("gradient = finite", GradientMethod::Fd),
        ("gradient_method = ad", GradientMethod::Ad),
    ] {
        let opts = parse_fit_options(&[input.to_string()]).unwrap();
        assert_eq!(opts.gradient_method, expected, "input: {input}");
    }

    // Unknown values must fail loudly — silently defaulting would hide
    // typos like `gradient = auo` that a user probably intended as `auto`.
    assert!(parse_fit_options(&["gradient = nope".to_string()]).is_err());
}

#[test]
fn test_parse_fit_options_reconverge_gradient_interval() {
    // Defaults to 0 (off) so the cheap fixed-EBE path stays the non-IOV
    // default.
    assert_eq!(
        parse_fit_options(&[]).unwrap().reconverge_gradient_interval,
        0
    );

    // Parses as a non-negative integer and is recorded as user-set (so it
    // isn't flagged as ignored downstream).
    let opts = parse_fit_options(&["reconverge_gradient_interval = 5".to_string()]).unwrap();
    assert_eq!(opts.reconverge_gradient_interval, 5);
    assert!(opts
        .user_set_keys
        .iter()
        .any(|k| k == "reconverge_gradient_interval"));

    // 0 is a valid explicit value (disables reconverging), not an error.
    assert_eq!(
        parse_fit_options(&["reconverge_gradient_interval = 0".to_string()])
            .unwrap()
            .reconverge_gradient_interval,
        0
    );

    // Non-integer values fail loudly.
    assert!(parse_fit_options(&["reconverge_gradient_interval = lots".to_string()]).is_err());
}

#[test]
fn test_parse_all_example_ferx_files() {
    // Smoke test: every checked-in example must parse under the strict
    // [fit_options] rules. Guards against accidentally tightening a key
    // in apply_fit_option in a way that breaks a shipped example.
    //
    // When `--features nn` is off, files that declare a `[covariate_nn]`
    // or `[dynamics_nn]` block are skipped — they're only valid under
    // the feature gate. A cheap pre-scan of file contents handles this
    // without splitting the example directory.
    let examples_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
    let mut seen = 0;
    for entry in std::fs::read_dir(&examples_dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|s| s.to_str()) != Some("ferx") {
            continue;
        }
        if !cfg!(feature = "nn") {
            let src = std::fs::read_to_string(&path).unwrap_or_default();
            if src.contains("[covariate_nn") || src.contains("[dynamics_nn") {
                continue;
            }
        }
        // A file declaring a feature-gated endpoint block — TTE `[event_model]` /
        // `[binary_model]` (`survival`), or the CTMM `[markov_model]` (`markov`,
        // which implies `survival`) — cannot parse in a build without that feature:
        // since #1040 the parser rejects the block outright rather than reading the
        // file as a Gaussian-only model. Skip those files here; the rejection itself
        // is pinned by `test_feature_gated_block_rejected_without_its_feature`.
        // Line-start checks so a comment like "# Note: [structural_model] ..." in an
        // example header does not falsely count as a block.
        let block_at_line_start = |src: &str, block: &str| {
            src.lines()
                .any(|l| l.trim_start().starts_with(&format!("[{block}")))
        };
        if !cfg!(feature = "survival") {
            let src = std::fs::read_to_string(&path).unwrap_or_default();
            if block_at_line_start(&src, "event_model") || block_at_line_start(&src, "binary_model")
            {
                continue;
            }
        }
        if !cfg!(feature = "markov") {
            let src = std::fs::read_to_string(&path).unwrap_or_default();
            if block_at_line_start(&src, "markov_model") {
                continue;
            }
        }
        seen += 1;
        if let Err(e) = parse_full_model_file(&path) {
            panic!("failed to parse {}: {}", path.display(), e);
        }
    }
    assert!(
        seen > 0,
        "no .ferx files found in {}",
        examples_dir.display()
    );
}

#[test]
fn test_parse_fit_options_applies_known_keys() {
    let lines = vec![
        "method = saem".to_string(),
        "n_exploration = 200".to_string(),
        "n_convergence = 400".to_string(),
        "sir = true".to_string(),
        "sir_samples = 2000".to_string(),
    ];
    let opts = parse_fit_options(&lines).unwrap();
    assert_eq!(opts.method, EstimationMethod::Saem);
    assert_eq!(opts.saem_n_exploration, 200);
    assert_eq!(opts.saem_n_convergence, 400);
    assert!(opts.sir);
    assert_eq!(opts.sir_samples, 2000);
}

#[test]
fn test_parse_data_block_path() {
    let (path, map) = parse_data_block(&["path = warfarin.csv".to_string()]).unwrap();
    assert_eq!(path, "warfarin.csv");
    assert!(map.is_empty());
    // Quoted paths (e.g. containing spaces) are accepted, quotes stripped.
    let (path, _) = parse_data_block(&["path = \"my data.csv\"".to_string()]).unwrap();
    assert_eq!(path, "my data.csv");
}

#[test]
fn test_parse_data_block_missing_path_is_error() {
    let err = parse_data_block(&[]).unwrap_err();
    assert!(err.contains("missing required key `path`"), "{err}");
}

#[test]
fn test_parse_data_block_arbitrary_target_is_accepted() {
    // #742: a non-canonical target renames an arbitrary column. Its case is
    // preserved (covariate lookups are case-sensitive), unlike canonical
    // roles which are upper-cased.
    let (_, map) =
        parse_data_block(&["path = d.csv".to_string(), "WT = weight".to_string()]).unwrap();
    assert_eq!(map, vec![("WT".to_string(), "weight".to_string())]);
    // Mixed-case arbitrary target kept verbatim.
    let (_, map) =
        parse_data_block(&["path = d.csv".to_string(), "MyCov = raw".to_string()]).unwrap();
    assert_eq!(map, vec![("MyCov".to_string(), "raw".to_string())]);
}

#[test]
fn test_parse_data_block_column_mapping() {
    // `target = actual` remappings. Canonical roles are upper-cased to the
    // standard spelling; the actual header is preserved verbatim. `path`
    // still parses alongside.
    let (path, map) = parse_data_block(&[
        "path = mydata.csv".to_string(),
        "time = TAFD".to_string(),
        "DV = CONC".to_string(),
    ])
    .unwrap();
    assert_eq!(path, "mydata.csv");
    assert_eq!(
        map,
        vec![
            ("TIME".to_string(), "TAFD".to_string()),
            ("DV".to_string(), "CONC".to_string()),
        ]
    );
}

#[test]
fn test_parse_data_block_empty_path_is_error() {
    // A bare `path =` fails loudly here rather than surfacing later as an
    // opaque CSV-open error.
    let err = parse_data_block(&["path =".to_string()]).unwrap_err();
    assert!(err.contains("empty `path`"), "{err}");
}

#[test]
fn test_parse_data_block_empty_target_is_error() {
    let err = parse_data_block(&["path = d.csv".to_string(), "time =".to_string()]).unwrap_err();
    assert!(err.contains("empty column name for `time`"), "{err}");
}

#[test]
fn test_parse_data_block_target_mapped_twice_is_error() {
    let err = parse_data_block(&[
        "path = d.csv".to_string(),
        "TIME = TAFD".to_string(),
        "time = T2".to_string(),
    ])
    .unwrap_err();
    assert!(err.contains("target `TIME` mapped more than once"), "{err}");
}

#[test]
fn test_parse_data_block_duplicate_target_is_error() {
    // Two targets pointing at the same actual header (case-insensitive) is
    // rejected: the reader would rename one header to two targets.
    let err = parse_data_block(&[
        "path = d.csv".to_string(),
        "TIME = X".to_string(),
        "DV = x".to_string(),
    ])
    .unwrap_err();
    assert!(err.contains("both map to column `X`"), "{err}");
}

#[test]
fn test_data_block_populates_parsed_model_column_map() {
    let content =
        format!("{MINIMAL_MODEL}[data]\n  path = warfarin.csv\n  TIME = TAFD\n  DV = CONC\n");
    let parsed = parse_full_model(&content).unwrap();
    assert_eq!(parsed.data_path.as_deref(), Some("warfarin.csv"));
    assert_eq!(
        parsed.column_map,
        vec![
            ("TIME".to_string(), "TAFD".to_string()),
            ("DV".to_string(), "CONC".to_string()),
        ]
    );
}

#[test]
fn test_model_without_data_mappings_has_empty_column_map() {
    let content = format!("{MINIMAL_MODEL}[data]\n  path = warfarin.csv\n");
    let parsed = parse_full_model(&content).unwrap();
    assert!(parsed.column_map.is_empty());
}

/// Minimal valid one-compartment IV model, used as a base by the `[data]`
/// block tests below (block content varies per test).
const MINIMAL_MODEL: &str = "\
[parameters]
  theta TVCL(1.0)
  theta TVV(1.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.1
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
";

/// `MINIMAL_MODEL` with every `omega` declaration and every `exp(ETA_*)` term
/// removed — a continuous (residual-error) endpoint with no random effects. This
/// is the shape #989 unblocked; before it, `parse_full_model` returned
/// `Err("No omega parameters defined")`.
const NO_OMEGA_MODEL: &str = "\
[parameters]
  theta TVCL(1.0)
  theta TVV(1.0)
  sigma PROP_ERR ~ 0.1
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
";

#[test]
fn test_parse_continuous_model_with_no_omega_declarations() {
    let parsed = parse_full_model(NO_OMEGA_MODEL)
        .expect("a continuous model with no random effects must parse (#989)");
    assert_eq!(parsed.model.n_eta, 0);
    assert_eq!(parsed.model.default_params.omega.matrix.nrows(), 0);
    assert!(parsed.model.default_params.omega.eta_names.is_empty());
    assert!(parsed.model.default_params.omega_fixed.is_empty());
    // Only Ω went away — the residual error is untouched, and its sigma is still
    // the thing that defines the likelihood.
    assert_eq!(parsed.model.default_params.sigma.values.len(), 1);
    // No IOV was declared, so the kappa matrix must stay absent rather than
    // becoming a second empty matrix (`omega_iov.is_some()` reads as "IOV active").
    assert!(parsed.model.default_params.omega_iov.is_none());
}

#[test]
fn test_no_omega_model_still_requires_sigma() {
    // Ω is optional; σ is not. The two capabilities are separate, and the error
    // must say so — a user who deleted the wrong line needs to see which.
    let no_sigma = NO_OMEGA_MODEL.replace("  sigma PROP_ERR ~ 0.1\n", "");
    // `ParsedModel` is not `Debug`, so `expect_err` is unavailable here.
    let err = match parse_full_model(&no_sigma) {
        Ok(_) => panic!("a continuous endpoint with no sigma must still be rejected"),
        Err(e) => e,
    };
    assert!(
        err.contains("sigma"),
        "the sigma error must name sigma, not Omega: {err}"
    );
    assert!(
        !err.contains("omega parameters"),
        "must not report this as an Omega problem: {err}"
    );
}

#[test]
fn test_no_omega_model_with_error_model_removed_is_rejected() {
    let no_error_model =
        NO_OMEGA_MODEL.replace("[error_model]\n  DV ~ proportional(PROP_ERR)\n", "");
    let err = match parse_full_model(&no_error_model) {
        Ok(_) => panic!("a continuous model with no [error_model] must still be rejected"),
        Err(e) => e,
    };
    assert!(err.contains("[error_model]"), "{err}");
}

#[test]
fn test_parse_no_bsv_omega_with_kappa_is_iov_only() {
    // Kappas with no BSV etas: Ω is 0x0 while Ω_IOV is 1x1. This combination was
    // also blocked by the old rejection (it keys on the BSV eta list), so it needs
    // its own pin — the IOV matrix must not be collapsed along with Ω.
    let src = "\
[parameters]
  theta TVCL(1.0)
  theta TVV(1.0)
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.1
[individual_parameters]
  CL = TVCL * exp(KAPPA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
";
    let parsed = parse_full_model(src).expect("kappa without BSV omega must parse (#989)");
    assert_eq!(parsed.model.n_eta, 0);
    assert_eq!(parsed.model.default_params.omega.matrix.nrows(), 0);
    let iov = parsed
        .model
        .default_params
        .omega_iov
        .as_ref()
        .expect("declaring a kappa must still build an IOV matrix");
    assert_eq!(iov.matrix.nrows(), 1);
    assert!((iov.matrix[(0, 0)] - 0.01).abs() < 1e-12);
}

#[test]
fn test_data_block_populates_parsed_model_data_path() {
    let content = format!("{MINIMAL_MODEL}[data]\n  path = warfarin.csv\n");
    let parsed = parse_full_model(&content).unwrap();
    assert_eq!(parsed.data_path.as_deref(), Some("warfarin.csv"));
}

#[test]
fn test_model_without_data_block_has_no_data_path() {
    let parsed = parse_full_model(MINIMAL_MODEL).unwrap();
    assert_eq!(parsed.data_path, None);
}

#[test]
fn test_parse_full_model_file_resolves_relative_data_path_against_model_dir() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("warfarin.csv"), "ID,TIME,DV,EVID,AMT,CMT\n").unwrap();
    let model_path = dir.path().join("m.ferx");
    let content = format!("{MINIMAL_MODEL}[data]\n  path = warfarin.csv\n");
    std::fs::write(&model_path, content).unwrap();
    let parsed = parse_full_model_file(&model_path).unwrap();
    assert_eq!(
        parsed.data_path.as_deref(),
        Some(dir.path().join("warfarin.csv").to_str().unwrap())
    );
}

#[test]
fn test_parse_imp_defensive_alpha() {
    // Default is opt-out (legacy single-proposal sampler); the mixture is
    // opt-in (issue #528).
    assert_eq!(parse_fit_options(&[]).unwrap().imp_defensive_alpha, 0.0);
    // In-range values are accepted.
    let opts = parse_fit_options(&["imp_defensive_alpha = 0.25".to_string()]).unwrap();
    assert_eq!(opts.imp_defensive_alpha, 0.25);
    assert_eq!(
        parse_fit_options(&["imp_defensive_alpha = 0.0".to_string()])
            .unwrap()
            .imp_defensive_alpha,
        0.0
    );
    // `impmap_defensive_alpha` is an accepted alias for the same field.
    assert_eq!(
        parse_fit_options(&["impmap_defensive_alpha = 0.1".to_string()])
            .unwrap()
            .imp_defensive_alpha,
        0.1
    );
    // Out of [0, 1) is rejected at parse time, under either spelling.
    assert!(parse_fit_options(&["imp_defensive_alpha = 1.0".to_string()]).is_err());
    assert!(parse_fit_options(&["imp_defensive_alpha = -0.1".to_string()]).is_err());
    assert!(parse_fit_options(&["impmap_defensive_alpha = 1.0".to_string()]).is_err());
}

#[test]
fn test_parse_covariance_fallback_none_and_sir() {
    use crate::types::CovarianceFallback;
    let opts = parse_fit_options(&["covariance_fallback = sir".to_string()]).unwrap();
    assert_eq!(opts.covariance_fallback, CovarianceFallback::Sir);

    let opts2 = parse_fit_options(&["covariance_fallback = none".to_string()]).unwrap();
    assert_eq!(opts2.covariance_fallback, CovarianceFallback::None);

    // Case-insensitive
    let opts3 = parse_fit_options(&["covariance_fallback = SIR".to_string()]).unwrap();
    assert_eq!(opts3.covariance_fallback, CovarianceFallback::Sir);

    // Invalid value returns error
    let err = parse_fit_options(&["covariance_fallback = hmc".to_string()]);
    assert!(err.is_err());
}

#[test]
fn test_parse_covariance_method() {
    use crate::types::CovarianceMethod;
    // default
    let def = parse_fit_options(&[]).unwrap();
    assert_eq!(def.covariance_method, CovarianceMethod::Hessian);
    // r / s / rsr (and the long-form aliases), case-insensitive
    for (input, expected) in [
        ("r", CovarianceMethod::Hessian),
        ("hessian", CovarianceMethod::Hessian),
        ("S", CovarianceMethod::CrossProduct),
        ("cross_product", CovarianceMethod::CrossProduct),
        ("RSR", CovarianceMethod::Sandwich),
        ("sandwich", CovarianceMethod::Sandwich),
    ] {
        let opts = parse_fit_options(&[format!("covariance_method = {input}")]).unwrap();
        assert_eq!(opts.covariance_method, expected, "input `{input}`");
    }
    // invalid
    assert!(parse_fit_options(&["covariance_method = bhhh".to_string()]).is_err());
}

#[test]
fn test_parse_parameter_scaling() {
    use crate::types::ParameterScaling;
    // Default: parameter_scaling = Auto.
    let def = parse_fit_options(&[]).unwrap();
    assert_eq!(def.parameter_scaling, ParameterScaling::Auto);
    // parameter_scaling keywords, case-insensitive.
    for (input, expected) in [
        ("auto", ParameterScaling::Auto),
        ("none", ParameterScaling::None),
        ("ABS", ParameterScaling::Abs),
        ("rescale2", ParameterScaling::Rescale2),
    ] {
        let opts = parse_fit_options(&[format!("parameter_scaling = {input}")]).unwrap();
        assert_eq!(opts.parameter_scaling, expected, "input `{input}`");
    }
    assert!(parse_fit_options(&["parameter_scaling = bogus".to_string()]).is_err());
}

// ── mu-referencing pattern detection ─────────────────────────────────

fn detect_one(line: &str, theta_names: &[&str], eta_names: &[&str]) -> Option<MuRef> {
    let tn: Vec<String> = theta_names.iter().map(|s| s.to_string()).collect();
    let en: Vec<String> = eta_names.iter().map(|s| s.to_string()).collect();
    let ctx = ParseCtx::new(&tn, &en, &[]);
    let stmts = parse_block_statements(line, ctx, StatementMode::Plain).ok()?;
    let refs = detect_mu_refs(&stmts, &tn, &en, &[]);
    // Return the one detected mu-ref (if any). Tests assume a single line.
    refs.into_iter().next().map(|(_, v)| v)
}

#[test]
fn test_detect_mu_ref_multiplicative_exp() {
    // Classic NONMEM pattern: CL = TVCL * exp(ETA_CL)
    let m = detect_one("CL = TVCL * exp(ETA_CL)", &["TVCL"], &["ETA_CL"])
        .expect("should detect mu-ref");
    assert_eq!(m.theta_name, "TVCL");
    assert!(m.log_transformed);
}

#[test]
fn test_detect_mu_ref_exp_of_log_sum() {
    // Canonical mu-reference form: exp(log(THETA) + ETA)
    let m = detect_one("CL = exp(log(TVCL) + ETA_CL)", &["TVCL"], &["ETA_CL"])
        .expect("should detect mu-ref");
    assert_eq!(m.theta_name, "TVCL");
    assert!(m.log_transformed);
}

#[test]
fn test_detect_mu_ref_exp_of_log_sum_reversed() {
    // ETA on the left: exp(ETA + log(THETA))
    let m = detect_one("CL = exp(ETA_CL + log(TVCL))", &["TVCL"], &["ETA_CL"])
        .expect("should detect mu-ref");
    assert_eq!(m.theta_name, "TVCL");
    assert!(m.log_transformed);
}

#[test]
fn test_detect_mu_ref_additive() {
    // Additive eta: CL = TVCL + ETA_CL → mu = TVCL (not log-transformed)
    let m = detect_one("CL = TVCL + ETA_CL", &["TVCL"], &["ETA_CL"]).expect("should detect mu-ref");
    assert_eq!(m.theta_name, "TVCL");
    assert!(!m.log_transformed);
}

#[test]
fn test_detect_mu_ref_additive_reversed() {
    // ETA first: CL = ETA_CL + TVCL
    let m = detect_one("CL = ETA_CL + TVCL", &["TVCL"], &["ETA_CL"]).expect("should detect mu-ref");
    assert_eq!(m.theta_name, "TVCL");
    assert!(!m.log_transformed);
}

#[test]
fn test_detect_mu_ref_product_chain_with_covariate() {
    // Real covariate model: CL = TVCL * (WT/70)^0.75 * exp(ETA_CL).
    // The detector walks the Mul chain for the anchor theta and the
    // exp(eta) factor; the Power sub-expression is opaque (neither a
    // Theta nor an exp(Eta)), so it is simply skipped. As long as there
    // is exactly one bare Theta factor, detection still succeeds.
    let m = detect_one(
        "CL = TVCL * (WT/70)^0.75 * exp(ETA_CL)",
        &["TVCL"],
        &["ETA_CL"],
    )
    .expect("should still detect mu-ref through opaque covariate term");
    assert_eq!(m.theta_name, "TVCL");
    assert!(m.log_transformed);
}

#[test]
fn test_detect_mu_ref_rejects_two_thetas() {
    // Two thetas in the product → ambiguous anchor, pattern rejected.
    let m = detect_one(
        "CL = TVCL * TVCL2 * exp(ETA_CL)",
        &["TVCL", "TVCL2"],
        &["ETA_CL"],
    );
    assert!(m.is_none());
}

#[test]
fn test_detect_mu_ref_rejects_constant_only() {
    // No theta in the product → not a mu-ref.
    let m = detect_one("CL = 2.0 * exp(ETA_CL)", &["TVCL"], &["ETA_CL"]);
    assert!(m.is_none());
}

#[test]
fn test_detect_mu_ref_rejects_compound_eta_expression() {
    // exp(ETA_CL + ETA_OCC) — IIV+IOV combined pattern. Since Fix 1,
    // find_exp_eta_in_mul recognises this and returns the min-index eta
    // (ETA_CL, index 0), so a mu-ref IS detected for ETA_CL → TVCL.
    let m = detect_one(
        "CL = TVCL * exp(ETA_CL + ETA_OCC)",
        &["TVCL"],
        &["ETA_CL", "ETA_OCC"],
    );
    let m = m.expect("IIV+IOV combined pattern should detect mu-ref for ETA_CL");
    assert_eq!(m.theta_name, "TVCL");
    assert!(m.log_transformed);
}

#[test]
fn test_detect_mu_ref_rejects_no_eta() {
    // KM = TVKM — no eta, no mu-ref recorded.
    let m = detect_one("KM = TVKM", &["TVKM"], &[]);
    assert!(m.is_none());
}

#[test]
fn test_detect_mu_ref_multiple_parameters() {
    // Detect across several lines; each eta maps to its own theta.
    let block = "CL = TVCL * exp(ETA_CL)\nV  = TVV  * exp(ETA_V)\nKA = TVKA * exp(ETA_KA)";
    let tn = vec!["TVCL".to_string(), "TVV".to_string(), "TVKA".to_string()];
    let en = vec![
        "ETA_CL".to_string(),
        "ETA_V".to_string(),
        "ETA_KA".to_string(),
    ];
    let ctx = ParseCtx::new(&tn, &en, &[]);
    let stmts = parse_block_statements(block, ctx, StatementMode::Plain).unwrap();
    let refs = detect_mu_refs(&stmts, &tn, &en, &[]);
    assert_eq!(refs.len(), 3);
    assert_eq!(refs["ETA_CL"].theta_name, "TVCL");
    assert_eq!(refs["ETA_V"].theta_name, "TVV");
    assert_eq!(refs["ETA_KA"].theta_name, "TVKA");
    assert!(refs.values().all(|m| m.log_transformed));
}

#[test]
fn test_detect_mu_ref_full_model_parse() {
    // End-to-end: parse a minimal .ferx and verify mu_refs is populated.
    let content = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)

  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04

  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let parsed = parse_full_model(content).expect("model should parse");
    assert_eq!(parsed.model.mu_refs.len(), 2);
    let cl = parsed.model.mu_refs.get("ETA_CL").unwrap();
    assert_eq!(cl.theta_name, "TVCL");
    assert!(cl.log_transformed);
}

#[test]
fn test_parse_diagonal_omega() {
    let lines = vec![
        "omega ETA_CL ~ 0.07".to_string(),
        "omega ETA_V  ~ 0.02".to_string(),
    ];
    let (_, omegas, block_omegas, _, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert_eq!(omegas.len(), 2);
    assert_eq!(block_omegas.len(), 0);
    assert_eq!(omegas[0].name, "ETA_CL");
    assert!((omegas[0].variance - 0.07).abs() < 1e-10);
}

#[test]
fn test_parse_block_omega() {
    let lines = vec!["block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04]".to_string()];
    let (_, omegas, block_omegas, _, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert_eq!(omegas.len(), 0);
    assert_eq!(block_omegas.len(), 1);
    assert_eq!(block_omegas[0].names, vec!["ETA_CL", "ETA_V"]);
    assert_eq!(block_omegas[0].lower_triangle, vec![0.09, 0.02, 0.04]);
}

#[test]
fn test_parse_block_omega_multiline() {
    // Lower triangle spread across several lines, as extract_blocks would
    // hand them to parse_parameters (one trimmed string per source line).
    let lines = vec![
        "block_omega (ETA_CL, ETA_V) = [".to_string(),
        "0.09,".to_string(),
        "0.02, 0.04".to_string(),
        "]".to_string(),
    ];
    let (_, omegas, block_omegas, _, _, eta_names, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert_eq!(omegas.len(), 0);
    assert_eq!(block_omegas.len(), 1);
    assert_eq!(block_omegas[0].names, vec!["ETA_CL", "ETA_V"]);
    assert_eq!(block_omegas[0].lower_triangle, vec![0.09, 0.02, 0.04]);
    assert!(!block_omegas[0].fixed);
    assert_eq!(eta_names, vec!["ETA_CL", "ETA_V"]);
}

#[test]
fn test_parse_block_omega_multiline_fix() {
    // FIX keyword on the closing-bracket line must still be honored.
    let lines = vec![
        "block_omega (ETA_CL, ETA_V) = [0.09,".to_string(),
        "0.02, 0.04] FIX".to_string(),
    ];
    let (_, _, block_omegas, _, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert_eq!(block_omegas.len(), 1);
    assert!(block_omegas[0].fixed);
}

#[test]
fn test_parse_block_omega_multiline_fix_own_line() {
    // FIX on its own line after the closing bracket must still be honored.
    let lines = vec![
        "block_omega (ETA_CL, ETA_V) = [".to_string(),
        "0.09,".to_string(),
        "0.02, 0.04".to_string(),
        "]".to_string(),
        "FIX".to_string(),
    ];
    let (_, _, block_omegas, _, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert_eq!(block_omegas.len(), 1);
    assert!(block_omegas[0].fixed);
}

#[test]
fn test_parse_block_kappa_multiline() {
    let lines = vec![
        "block_kappa (KAPPA_CL, KAPPA_V) = [".to_string(),
        "0.05, 0.01, 0.03".to_string(),
        "]".to_string(),
    ];
    let (_, _, _, _, _, _, kappas, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert_eq!(kappas.block.len(), 1);
    assert_eq!(kappas.block[0].names, vec!["KAPPA_CL", "KAPPA_V"]);
    assert_eq!(kappas.block[0].lower_triangle, vec![0.05, 0.01, 0.03]);
}

#[test]
fn test_parse_block_kappa_multiline_fix_own_line() {
    // FIX on its own line after the closing bracket must be honored for
    // IOV blocks too (shared fold logic in join_bracketed_lines).
    let lines = vec![
        "block_kappa (KAPPA_CL, KAPPA_V) = [".to_string(),
        "0.05, 0.01, 0.03".to_string(),
        "]".to_string(),
        "FIX".to_string(),
    ];
    let (_, _, _, _, _, _, kappas, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert_eq!(kappas.block.len(), 1);
    assert!(kappas.block[0].fixed);
}

#[test]
fn test_parse_block_omega_3x3() {
    let lines = vec![
        "block_omega (ETA_CL, ETA_V, ETA_KA) = [0.09, 0.01, 0.04, 0.005, 0.002, 0.16]".to_string(),
    ];
    let (_, _, block_omegas, _, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert_eq!(block_omegas[0].names.len(), 3);
    assert_eq!(block_omegas[0].lower_triangle.len(), 6); // 3*(3+1)/2
}

#[test]
fn test_parse_block_omega_wrong_count() {
    let lines = vec![
        "block_omega (ETA_CL, ETA_V) = [0.09, 0.02]".to_string(), // needs 3, got 2
    ];
    let result = parse_parameters(&lines, &Default::default());
    assert!(result.is_err());
}

#[test]
fn test_parse_mixed_diagonal_and_block() {
    let lines = vec![
        "omega ETA_KA ~ 0.40".to_string(),
        "block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04]".to_string(),
    ];
    let (_, omegas, block_omegas, _, _, eta_names, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert_eq!(omegas.len(), 1);
    assert_eq!(block_omegas.len(), 1);
    // Declaration order preserved: ETA_KA first, then block (ETA_CL, ETA_V)
    assert_eq!(eta_names, vec!["ETA_KA", "ETA_CL", "ETA_V"]);
}

#[test]
fn test_declaration_order_block_before_diagonal() {
    let lines = vec![
        "block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04]".to_string(),
        "omega ETA_KA ~ 0.40".to_string(),
    ];
    let (_, _, _, _, _, eta_names, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    // block_omega declared first, so ETA_CL, ETA_V come before ETA_KA
    assert_eq!(eta_names, vec!["ETA_CL", "ETA_V", "ETA_KA"]);
}

/// #989: an empty eta list is a fixed-effects-only (naive-pooled) model, not an
/// error. This used to return `Err("No omega parameters defined")`, which is what
/// blocked a continuous model from omitting its `omega` declarations.
#[test]
fn test_build_omega_matrix_empty_is_naive_pooled_not_an_error() {
    let omega = build_omega_matrix(&[], &[], &[])
        .expect("an empty eta list must yield a 0x0 Omega, not an error");
    assert_eq!(omega.matrix.nrows(), 0);
    assert_eq!(omega.matrix.ncols(), 0);
    assert!(omega.eta_names.is_empty());
    // The 0x0 matrix must take the diagonal packing path: `omega_packed_len(0, ..)`
    // is 0 either way, but the flag is what `parameterization.rs` branches on.
    assert!(omega.diagonal);
    // `log|Ω| = 0` is what makes the FOCE/FOCEI objective collapse to plain ML —
    // the marginal loses its `½log|Ω|` penalty rather than picking up a NaN.
    assert_eq!(omega.log_det, 0.0);
    assert_eq!(
        crate::estimation::parameterization::omega_packed_len(0, false),
        0,
        "a 0x0 Omega must contribute no packed optimizer coordinates"
    );
}

#[test]
fn test_build_omega_matrix_diagonal_only() {
    let diag = vec![
        OmegaSpec {
            name: "ETA_CL".into(),
            variance: 0.09,
            fixed: false,
            init_as_sd: false,
        },
        OmegaSpec {
            name: "ETA_V".into(),
            variance: 0.04,
            fixed: false,
            init_as_sd: false,
        },
    ];
    let names = vec!["ETA_CL".into(), "ETA_V".into()];
    let omega = build_omega_matrix(&diag, &[], &names).unwrap();
    assert!(omega.diagonal);
    assert!((omega.matrix[(0, 0)] - 0.09).abs() < 1e-10);
    assert!((omega.matrix[(1, 1)] - 0.04).abs() < 1e-10);
    assert!((omega.matrix[(0, 1)]).abs() < 1e-10);
}

#[test]
fn test_build_omega_matrix_block() {
    let block = vec![BlockOmegaSpec {
        names: vec!["ETA_CL".into(), "ETA_V".into()],
        lower_triangle: vec![0.09, 0.02, 0.04],
        fixed: false,
    }];
    let names = vec!["ETA_CL".into(), "ETA_V".into()];
    let omega = build_omega_matrix(&[], &block, &names).unwrap();
    assert!(!omega.diagonal);
    assert!((omega.matrix[(0, 0)] - 0.09).abs() < 1e-10);
    assert!((omega.matrix[(1, 1)] - 0.04).abs() < 1e-10);
    assert!((omega.matrix[(0, 1)] - 0.02).abs() < 1e-10);
    assert!((omega.matrix[(1, 0)] - 0.02).abs() < 1e-10);
}

#[test]
fn test_build_omega_matrix_mixed() {
    let diag = vec![OmegaSpec {
        name: "ETA_KA".into(),
        variance: 0.16,
        fixed: false,
        init_as_sd: false,
    }];
    let block = vec![BlockOmegaSpec {
        names: vec!["ETA_CL".into(), "ETA_V".into()],
        lower_triangle: vec![0.09, 0.02, 0.04],
        fixed: false,
    }];
    let names = vec!["ETA_KA".into(), "ETA_CL".into(), "ETA_V".into()];
    let omega = build_omega_matrix(&diag, &block, &names).unwrap();
    assert!(!omega.diagonal);
    assert!((omega.matrix[(0, 0)] - 0.16).abs() < 1e-10); // ETA_KA
    assert!((omega.matrix[(1, 1)] - 0.09).abs() < 1e-10); // ETA_CL
    assert!((omega.matrix[(2, 2)] - 0.04).abs() < 1e-10); // ETA_V
    assert!((omega.matrix[(1, 2)] - 0.02).abs() < 1e-10); // cov(CL, V)
    assert!((omega.matrix[(0, 1)]).abs() < 1e-10); // no cov(KA, CL)
}

// ── FIX keyword ────────────────────────────────────────────────────────

#[test]
fn test_parse_theta_fix_without_bounds() {
    let lines = vec!["theta TVCL(0.1, FIX)".to_string()];
    let (thetas, _, _, _, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert_eq!(thetas.len(), 1);
    assert!(thetas[0].fixed);
    assert!((thetas[0].init - 0.1).abs() < 1e-12);
}

#[test]
fn test_parse_theta_fix_with_bounds() {
    let lines = vec!["theta TVCL(0.1, 0.01, 1.0, FIX)".to_string()];
    let (thetas, _, _, _, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert!(thetas[0].fixed);
    assert!((thetas[0].lower - 0.01).abs() < 1e-12);
    assert!((thetas[0].upper - 1.0).abs() < 1e-12);
}

#[test]
fn test_parse_theta_fix_no_comma_inside_parens() {
    // theta NAME(init FIX) — no comma before FIX
    let lines = vec!["theta TVCL(0.75 FIX)".to_string()];
    let (thetas, _, _, _, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert_eq!(thetas.len(), 1);
    assert!(thetas[0].fixed);
    assert!((thetas[0].init - 0.75).abs() < 1e-12);
}

#[test]
fn test_parse_theta_fix_after_paren() {
    // theta NAME(init) FIX — FIX outside closing paren
    let lines = vec!["theta TVCL(0.75) FIX".to_string()];
    let (thetas, _, _, _, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert_eq!(thetas.len(), 1);
    assert!(thetas[0].fixed);
    assert!((thetas[0].init - 0.75).abs() < 1e-12);
}

#[test]
fn test_parse_theta_fix_after_paren_with_bounds() {
    // theta NAME(init, lower, upper) FIX — bounds + FIX outside paren
    let lines = vec!["theta TVKA(1.0, 0.01, 10.0) FIX".to_string()];
    let (thetas, _, _, _, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert_eq!(thetas.len(), 1);
    assert!(thetas[0].fixed);
    assert!((thetas[0].init - 1.0).abs() < 1e-12);
    assert!((thetas[0].lower - 0.01).abs() < 1e-12);
    assert!((thetas[0].upper - 10.0).abs() < 1e-12);
}

#[test]
fn test_parse_theta_lower_bound_only() {
    // theta NAME(init, lower) — upper defaults to 1e9
    let lines = vec!["theta TVCL(1.0, 0.01)".to_string()];
    let (thetas, _, _, _, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert_eq!(thetas.len(), 1);
    assert!(!thetas[0].fixed);
    assert!((thetas[0].init - 1.0).abs() < 1e-12);
    assert!((thetas[0].lower - 0.01).abs() < 1e-12);
    assert!((thetas[0].upper - 1e9).abs() < 1.0);
}

#[test]
fn test_parse_theta_lower_bound_fix_inside() {
    // theta NAME(init, lower, FIX) — lower only + FIX inside parens
    let lines = vec!["theta TVCL(1.0, 0.01, FIX)".to_string()];
    let (thetas, _, _, _, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert_eq!(thetas.len(), 1);
    assert!(thetas[0].fixed);
    assert!((thetas[0].lower - 0.01).abs() < 1e-12);
    assert!((thetas[0].upper - 1e9).abs() < 1.0);
}

#[test]
fn test_parse_theta_lower_bound_fix_outside() {
    // theta NAME(init, lower) FIX — lower only + FIX after paren
    let lines = vec!["theta TVCL(1.0, 0.01) FIX".to_string()];
    let (thetas, _, _, _, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert_eq!(thetas.len(), 1);
    assert!(thetas[0].fixed);
    assert!((thetas[0].lower - 0.01).abs() < 1e-12);
    assert!((thetas[0].upper - 1e9).abs() < 1.0);
}

#[test]
fn test_parse_theta_unfixed_by_default() {
    let lines = vec!["theta TVCL(0.1, 0.01, 1.0)".to_string()];
    let (thetas, _, _, _, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert!(!thetas[0].fixed);
}

#[test]
fn test_parse_theta_allows_space_before_paren() {
    // Regression: `theta TVCL (5, ...)` (with whitespace before the paren)
    // used to silently fail to match, causing TVCL to be misclassified as
    // a covariate downstream.
    let lines = vec![
        "theta TVCL (5, 0.001, 100.0)".to_string(),
        "theta TVV  ( 10 )".to_string(),
        "theta TVKA\t(0.5, FIX)".to_string(),
    ];
    let (thetas, _, _, _, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert_eq!(thetas.len(), 3);
    assert_eq!(thetas[0].name, "TVCL");
    assert!((thetas[0].init - 5.0).abs() < 1e-12);
    assert!((thetas[0].lower - 0.001).abs() < 1e-12);
    assert!((thetas[0].upper - 100.0).abs() < 1e-12);
    assert!(!thetas[0].fixed);
    assert_eq!(thetas[1].name, "TVV");
    assert!((thetas[1].init - 10.0).abs() < 1e-12);
    assert_eq!(thetas[2].name, "TVKA");
    assert!(thetas[2].fixed);
}

#[test]
fn test_parse_omega_fix() {
    let lines = vec!["omega ETA_CL ~ 0.09 FIX".to_string()];
    let (_, omegas, _, _, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert!(omegas[0].fixed);
}

#[test]
fn test_omega_unfixed_no_annotation() {
    // Baseline: plain omega with no FIX and no annotation — confirms the
    // group-numbering shift (annotation moved 3→4) didn't regress the
    // common case.
    let lines = vec!["omega ETA_CL ~ 0.09".to_string()];
    let (_, omegas, _, _, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert!(!omegas[0].fixed);
    assert!(!omegas[0].init_as_sd);
    assert!((omegas[0].variance - 0.09).abs() < 1e-12);
}

#[test]
fn test_omega_double_fix_is_harmless() {
    // `FIX (sd) FIX` — both FIX groups fire; result must still be fixed
    // with SD squaring applied.
    let lines = vec!["omega ETA_CL ~ 0.30 FIX (sd) FIX".to_string()];
    let (_, omegas, _, _, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    let expected = 0.30 * 0.30;
    assert!((omegas[0].variance - expected).abs() < 1e-12);
    assert!(omegas[0].fixed);
    assert!(omegas[0].init_as_sd);
}

#[test]
fn test_parse_sigma_fix() {
    let lines = vec!["sigma PROP ~ 0.05 FIX".to_string()];
    let (_, _, _, sigmas, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert!(sigmas[0].fixed);
}

#[test]
fn test_parse_block_sigma_builds_sigmas_and_correlation() {
    let lines = vec!["block_sigma (PROP, ADD) = [0.04, 0.10, 1.0]".to_string()];
    let (_, _, _, sigmas, block_sigmas, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert_eq!(sigmas.len(), 2);
    assert_eq!(sigmas[0].name, "PROP");
    assert_eq!(sigmas[1].name, "ADD");
    assert!((sigmas[0].value - 0.2).abs() < 1e-12);
    assert!((sigmas[1].value - 1.0).abs() < 1e-12);

    let sigma_names: Vec<String> = sigmas.iter().map(|s| s.name.clone()).collect();
    let (corrs, fixed) = build_residual_correlations(&block_sigmas, &sigma_names).unwrap();
    assert_eq!(corrs.len(), 1);
    assert_eq!(corrs[0].sigma_i, 1);
    assert_eq!(corrs[0].sigma_j, 0);
    assert!((corrs[0].rho - 0.5).abs() < 1e-12);
    // #847: a bare `block_sigma` estimates its off-diagonal.
    assert_eq!(fixed, vec![false]);
}

#[test]
fn test_parse_block_sigma_fix_marks_sigmas_fixed() {
    let lines = vec!["block_sigma (PROP, ADD) = [0.04, 0.10, 1.0] FIX".to_string()];
    let (_, _, _, sigmas, block_sigmas, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert!(sigmas.iter().all(|s| s.fixed));

    // #847: `FIX` pins the off-diagonal too, which is what distinguishes it from
    // a bare `block_sigma` (that one estimates ρ, NONMEM `$SIGMA BLOCK(n)`).
    let sigma_names: Vec<String> = sigmas.iter().map(|s| s.name.clone()).collect();
    let (corrs, fixed) = build_residual_correlations(&block_sigmas, &sigma_names).unwrap();
    assert_eq!(corrs.len(), 1);
    assert_eq!(fixed, vec![true]);
}

#[test]
fn test_parse_block_sigma_wrong_triangle_count_errs() {
    // (PROP, ADD) needs 3 lower-triangle values; only 2 supplied.
    let lines = vec!["block_sigma (PROP, ADD) = [0.04, 1.0]".to_string()];
    let Err(err) = parse_parameters(&lines, &Default::default()) else {
        panic!("expected an error for a short lower triangle");
    };
    assert!(err.contains("lower-triangle"), "got: {err}");
}

#[test]
fn test_parse_block_sigma_negative_variance_errs() {
    let lines = vec!["block_sigma (PROP, ADD) = [-0.04, 0.10, 1.0]".to_string()];
    let Err(err) = parse_parameters(&lines, &Default::default()) else {
        panic!("expected an error for a negative variance");
    };
    assert!(err.contains("negative initial variance"), "got: {err}");
}

#[test]
fn test_parse_block_sigma_non_finite_covariance_errs() {
    let lines = vec!["block_sigma (PROP, ADD) = [0.04, inf, 1.0]".to_string()];
    let Err(err) = parse_parameters(&lines, &Default::default()) else {
        panic!("expected an error for a non-finite covariance");
    };
    assert!(err.contains("non-finite"), "got: {err}");
}

#[test]
fn test_build_residual_correlations_zero_diagonal_errs() {
    // A non-zero covariance against a zero diagonal variance is rejected.
    let block = BlockSigmaSpec {
        names: vec!["A".to_string(), "B".to_string()],
        lower_triangle: vec![0.0, 0.1, 1.0],
        fixed: false,
    };
    let names = vec!["A".to_string(), "B".to_string()];
    let err = build_residual_correlations(&[block], &names).unwrap_err();
    assert!(err.contains("positive diagonal variances"), "got: {err}");
}

#[test]
fn test_build_residual_correlations_invalid_rho_errs() {
    // cov 0.10 with both variances 0.04 implies ρ = 0.10/(0.2·0.2) = 2.5 > 1.
    let block = BlockSigmaSpec {
        names: vec!["A".to_string(), "B".to_string()],
        lower_triangle: vec![0.04, 0.10, 0.04],
        fixed: false,
    };
    let names = vec!["A".to_string(), "B".to_string()];
    let err = build_residual_correlations(&[block], &names).unwrap_err();
    assert!(err.contains("invalid correlation"), "got: {err}");
}

#[test]
fn test_build_residual_correlations_unit_rho_errs() {
    // cov 0.04 with both variances 0.04 implies rho = 1 exactly, which makes
    // the residual covariance matrix singular: reject it at parse time rather
    // than let it surface as a NaN/Inf OFV mid-fit.
    for cov in [0.04_f64, -0.04_f64] {
        let block = BlockSigmaSpec {
            names: vec!["A".to_string(), "B".to_string()],
            lower_triangle: vec![0.04, cov, 0.04],
            fixed: false,
        };
        let names = vec!["A".to_string(), "B".to_string()];
        let err = build_residual_correlations(&[block], &names).unwrap_err();
        assert!(err.contains("|rho| < 1"), "got: {err}");
    }
}

#[test]
fn test_build_residual_correlations_unknown_name_errs() {
    // Valid covariance, but the block names are absent from `sigma_names`.
    let block = BlockSigmaSpec {
        names: vec!["X".to_string(), "Y".to_string()],
        lower_triangle: vec![1.0, 0.5, 1.0],
        fixed: false,
    };
    let names = vec!["A".to_string(), "B".to_string()];
    let err = build_residual_correlations(&[block], &names).unwrap_err();
    assert!(err.contains("unknown sigma"), "got: {err}");
}

#[test]
fn test_build_residual_correlations_zero_covariance_kept_when_estimated() {
    // #847: a zero off-diagonal on a non-`FIX` block still declares an estimated
    // correlation. `$SIGMA BLOCK(2)` with a zero covariance init is a routine
    // NONMEM starting point; dropping it would silently fit a diagonal residual
    // with no coordinate to move and no diagnostic.
    let block = BlockSigmaSpec {
        names: vec!["A".to_string(), "B".to_string()],
        lower_triangle: vec![0.04, 0.0, 1.0],
        fixed: false,
    };
    let names = vec!["A".to_string(), "B".to_string()];
    let (corrs, fixed) = build_residual_correlations(&[block], &names).unwrap();
    assert_eq!(corrs.len(), 1);
    assert_eq!(corrs[0].rho, 0.0);
    assert_eq!(fixed, vec![false]);
}

#[test]
fn test_build_residual_correlations_zero_covariance_omitted_when_fixed() {
    // A `FIX`ed zero correlation is the same object as no correlation, so it
    // still carries nothing — and costs no packed coordinate.
    let block = BlockSigmaSpec {
        names: vec!["A".to_string(), "B".to_string()],
        lower_triangle: vec![0.04, 0.0, 1.0],
        fixed: true,
    };
    let names = vec!["A".to_string(), "B".to_string()];
    let (corrs, fixed) = build_residual_correlations(&[block], &names).unwrap();
    assert!(corrs.is_empty());
    assert!(fixed.is_empty());
}

#[test]
fn test_parse_block_omega_fix() {
    let lines = vec!["block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04] FIX".to_string()];
    let (_, _, blocks, _, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert!(blocks[0].fixed);
}

#[test]
fn test_fix_keyword_case_insensitive() {
    let lines = vec![
        "theta TVCL(0.1, fix)".to_string(),
        "omega ETA ~ 0.05 Fix".to_string(),
        "sigma S ~ 0.02 FIX".to_string(),
    ];
    let (thetas, omegas, _, sigmas, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert!(thetas[0].fixed);
    assert!(omegas[0].fixed);
    assert!(sigmas[0].fixed);
}

// ── SD-init annotation (issue #5 + #56) ─────────────────────────────

#[test]
fn test_omega_default_is_variance() {
    // No annotation: value is stored verbatim as variance.
    let lines = vec!["omega ETA_CL ~ 0.07".to_string()];
    let (_, omegas, _, _, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert!((omegas[0].variance - 0.07).abs() < 1e-12);
    assert!(!omegas[0].init_as_sd);
}

#[test]
fn test_omega_sd_annotation_squares_value() {
    // `(sd)` → variance is the square of the raw value.
    let lines = vec!["omega ETA_CL ~ 0.265 (sd)".to_string()];
    let (_, omegas, _, _, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    let expected = 0.265 * 0.265;
    assert!((omegas[0].variance - expected).abs() < 1e-12);
    assert!(omegas[0].init_as_sd);
}

#[test]
fn test_omega_variance_annotation_is_noop() {
    // `(variance)` and `(var)` are explicit no-ops.
    let lines = vec![
        "omega ETA_CL ~ 0.07 (variance)".to_string(),
        "omega ETA_V  ~ 0.04 (var)".to_string(),
    ];
    let (_, omegas, _, _, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert!((omegas[0].variance - 0.07).abs() < 1e-12);
    assert!(!omegas[0].init_as_sd);
    assert!((omegas[1].variance - 0.04).abs() < 1e-12);
    assert!(!omegas[1].init_as_sd);
}

#[test]
fn test_omega_sd_annotation_with_fix() {
    // `(sd) FIX` — both annotations must be honored together.
    let lines = vec!["omega ETA_CL ~ 0.30 (sd) FIX".to_string()];
    let (_, omegas, _, _, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    let expected = 0.30 * 0.30;
    assert!((omegas[0].variance - expected).abs() < 1e-12);
    assert!(omegas[0].fixed);
    assert!(omegas[0].init_as_sd);
}

#[test]
fn test_omega_fix_before_sd_annotation() {
    // `FIX (sd)` — FIX before the scale annotation.
    let lines = vec!["omega ETA_CL ~ 0.30 FIX (sd)".to_string()];
    let (_, omegas, _, _, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    let expected = 0.30 * 0.30;
    assert!((omegas[0].variance - expected).abs() < 1e-12);
    assert!(omegas[0].fixed);
    assert!(omegas[0].init_as_sd);
}

#[test]
fn test_omega_fix_before_annotation_no_sd() {
    // `FIX` before a no-op annotation — fixed and variance-scale.
    let lines = vec!["omega ETA_CL ~ 0.09 FIX (variance)".to_string()];
    let (_, omegas, _, _, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert!((omegas[0].variance - 0.09).abs() < 1e-12);
    assert!(omegas[0].fixed);
    assert!(!omegas[0].init_as_sd);
}

#[test]
fn test_sigma_fix_before_sd_annotation() {
    // `FIX (sd)` — FIX before the scale annotation for sigma.
    let lines = vec!["sigma PROP ~ 0.30 FIX (sd)".to_string()];
    let (_, _, _, sigmas, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert!(sigmas[0].fixed);
    assert!(sigmas[0].init_as_sd);
    assert!((sigmas[0].value - 0.30).abs() < 1e-12);
}

#[test]
fn test_sigma_fix_after_sd_annotation() {
    // `(sd) FIX` — existing form still works.
    let lines = vec!["sigma PROP ~ 0.30 (sd) FIX".to_string()];
    let (_, _, _, sigmas, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert!(sigmas[0].fixed);
    assert!(sigmas[0].init_as_sd);
}

#[test]
fn test_sigma_unfixed_no_annotation() {
    // Baseline: plain sigma with no FIX and no annotation — confirms the
    // group-numbering shift didn't regress the common case.
    let lines = vec!["sigma PROP ~ 0.04".to_string()];
    let (_, _, _, sigmas, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert!(!sigmas[0].fixed);
    assert!(!sigmas[0].init_as_sd);
    // Stored as SD internally: sqrt(0.04) = 0.2
    assert!((sigmas[0].value - 0.2).abs() < 1e-12);
}

#[test]
fn test_sigma_default_is_variance() {
    // Since #56, the default sigma input is variance — the parser sqrt's
    // it into the internal SD representation that the likelihood uses.
    let lines = vec!["sigma PROP ~ 0.04".to_string()];
    let (_, _, _, sigmas, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    // Stored value is SD = sqrt(variance) = sqrt(0.04) = 0.2.
    assert!((sigmas[0].value - 0.2).abs() < 1e-12);
    assert!(!sigmas[0].init_as_sd);
}

#[test]
fn test_sigma_sd_annotation_stores_value_as_is() {
    // `(sd)` → the value is already on the SD scale, no transform.
    let lines = vec!["sigma PROP ~ 0.2 (sd)".to_string()];
    let (_, _, _, sigmas, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert!((sigmas[0].value - 0.2).abs() < 1e-12);
    assert!(sigmas[0].init_as_sd);
}

#[test]
fn test_sigma_default_and_sd_equivalent_initial_value() {
    // `sigma X ~ v²` (default variance) must produce the same internal
    // SD as `sigma X ~ v (sd)` (SD).
    let lines = vec![
        "sigma A ~ 0.0004".to_string(),    // variance 0.0004
        "sigma B ~ 0.02 (sd)".to_string(), // SD 0.02
    ];
    let (_, _, _, sigmas, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert!((sigmas[0].value - sigmas[1].value).abs() < 1e-12);
}

#[test]
fn test_sigma_negative_variance_rejected() {
    // A negative value on the (default) variance scale is meaningless —
    // sqrt would yield NaN and silently corrupt the fit. Reject up-front
    // with a clear error.
    let lines = vec!["sigma PROP ~ -0.1".to_string()];
    let res = parse_parameters(&lines, &Default::default());
    match res {
        Err(msg) => assert!(msg.contains("negative initial variance"), "got: {msg}"),
        Ok(_) => panic!("expected error for negative sigma variance"),
    }
}

#[test]
fn test_sigma_negative_sd_rejected() {
    // Negative SD is just as nonsensical as negative variance, and the
    // optimizer's `s.max(1e-10).ln()` packing would silently clamp the
    // bad input rather than surface it. Reject at parse time, symmetric
    // with the negative-variance case.
    let lines = vec!["sigma PROP ~ -0.5 (sd)".to_string()];
    let res = parse_parameters(&lines, &Default::default());
    match res {
        Err(msg) => assert!(msg.contains("negative initial SD"), "got: {msg}"),
        Ok(_) => panic!("expected error for negative sigma SD"),
    }
}

#[test]
fn test_omega_negative_value_rejected() {
    // Same rule applies to omega — variance must be ≥ 0, and SD ≥ 0.
    for line in [
        "omega ETA_CL ~ -0.04",
        "omega ETA_CL ~ -0.2 (sd)",
        "kappa KAPPA_CL ~ -0.03",
        "kappa KAPPA_CL ~ -0.1 (sd)",
    ] {
        let res = parse_parameters(&[line.to_string()], &Default::default());
        assert!(res.is_err(), "expected negative `{line}` to be rejected");
    }
}

#[test]
fn test_kappa_sd_annotation_squares_value() {
    let lines = vec!["kappa KAPPA_CL ~ 0.25 (sd)".to_string()];
    let (_, _, _, _, _, _, kappas, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    let k = &kappas.diagonal[0];
    let expected = 0.25 * 0.25;
    assert!((k.variance - expected).abs() < 1e-12);
    assert!(k.init_as_sd);
}

#[test]
fn test_sd_annotation_case_insensitive() {
    // `(SD)`, `(Sd)`, `(sd)` must all be accepted.
    let lines = vec![
        "omega ETA_A ~ 0.1 (SD)".to_string(),
        "omega ETA_B ~ 0.2 (Sd)".to_string(),
        "omega ETA_C ~ 0.3 (sd)".to_string(),
    ];
    let (_, omegas, _, _, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert!(omegas.iter().all(|o| o.init_as_sd));
}

#[test]
fn test_unknown_scale_tag_is_ignored_as_trailing_garbage() {
    // The omega regex is intentionally unanchored — it matches the
    // leading `omega NAME ~ value` and lets trailing tokens fall through.
    // An unrecognized tag like `(foo)` therefore doesn't fail the parse;
    // the value is taken as variance and `init_as_sd` stays `false`, just
    // as if the tag weren't there. (This matches how the `FIX` keyword's
    // prefix-match check works — only the exact, recognized tag changes
    // behavior; anything else is silently ignored, consistent with the
    // parser's existing FIXED-vs-FIX handling.)
    let lines = vec!["omega ETA_CL ~ 0.07 (foo)".to_string()];
    let (_, omegas, _, _, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert_eq!(omegas.len(), 1);
    assert!((omegas[0].variance - 0.07).abs() < 1e-12);
    assert!(!omegas[0].init_as_sd);
}

#[test]
fn test_parse_full_model_threads_init_as_sd_to_compiled_model() {
    // End-to-end: a `(sd)` annotation in the .ferx text must surface as
    // `true` in the matching CompiledModel.{omega,sigma}_init_as_sd slot.
    let content = r#"
[parameters]
  theta TVCL(0.2)
  theta TVV(10.0)
  theta TVKA(1.5)
  omega ETA_CL ~ 0.30 (sd)
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.50 (sd)
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let parsed = parse_full_model(content).unwrap();
    assert_eq!(
        parsed.model.omega_init_as_sd,
        vec![true, false, true],
        "omega_init_as_sd flags must thread through to CompiledModel"
    );
    assert_eq!(parsed.model.sigma_init_as_sd, vec![true]);
    // Verify the SD-coded omega was squared: 0.30² = 0.09.
    let omega = &parsed.model.default_params.omega.matrix;
    assert!((omega[(0, 0)] - 0.09).abs() < 1e-12);
    // And the variance-coded omega is stored verbatim.
    assert!((omega[(1, 1)] - 0.04).abs() < 1e-12);
    // Sigma stored as SD (input was already SD, no transform).
    let sigma = &parsed.model.default_params.sigma.values;
    assert!((sigma[0] - 0.02).abs() < 1e-12);
}

#[test]
fn test_fix_keyword_rejects_prefix_match() {
    // `FIXED` must not be silently accepted as `FIX`. Any non-exact token
    // should leave the parameter as free (or fail to parse the line),
    // never flip `fixed = true`.
    let lines = vec![
        "omega ETA_CL ~ 0.09 FIXED".to_string(),
        "sigma PROP ~ 0.02 FIXED".to_string(),
        "block_omega (A, B) = [1.0, 0.0, 1.0] FIXED".to_string(),
    ];
    let (_, omegas, blocks, sigmas, _, _, _, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    // omega/sigma still parse (trailing `FIXED` is ignored) but must NOT
    // be marked fixed.
    assert!(!omegas[0].fixed);
    assert!(!sigmas[0].fixed);
    assert!(!blocks[0].fixed);
}

#[test]
fn test_parse_full_model_block_sigma_combined_builds_correlation() {
    // End-to-end: a combined-error model whose two sigmas are declared via
    // block_sigma must carry one residual correlation into CompiledModel.
    let content = r#"
[parameters]
  theta TVCL(0.2)
  theta TVV(10.0)
  omega ETA_CL ~ 0.09
  block_sigma (PROP_ERR, ADD_ERR) = [0.04, 0.10, 1.0] FIX
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)
"#;
    let parsed = parse_full_model(content).unwrap();
    assert_eq!(parsed.model.residual_correlations.len(), 1);
    let c = parsed.model.residual_correlations[0];
    // cov 0.10 / (sqrt(0.04)·sqrt(1.0)) = 0.10 / 0.2 = 0.5.
    assert!((c.rho - 0.5).abs() < 1e-12);
}

#[test]
fn test_ruv_magnitude_bare_sigmas_is_none() {
    // A plain combined model carries no custom magnitude (legacy path).
    let content = r#"
[parameters]
  theta TVCL(0.2)
  theta TVV(10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04
  sigma ADD_ERR ~ 1.0
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)
"#;
    let model = parse_model_string(content).unwrap();
    assert!(model.ruv_magnitude.is_none());
    assert!(!model.has_custom_ruv_magnitude());
}

#[test]
fn test_ruv_magnitude_time_varying_prop_parses() {
    // The proportional sigma magnitude inflates for TIME > 24 via a theta.
    let content = r#"
[parameters]
  theta TVCL(0.2)
  theta TVV(10.0)
  theta RUV_LATE(1.5, 0.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04
  sigma ADD_ERR ~ 1.0
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ combined(PROP_ERR * (if (TIME > 24.0) RUV_LATE else 1.0), ADD_ERR)
"#;
    let model = parse_model_string(content).unwrap();
    let rm = model.ruv_magnitude.as_ref().expect("magnitude present");
    assert!(rm.is_active());
    assert_eq!(rm.per_sigma.len(), 2);
    assert!(rm.per_sigma[0].is_some(), "prop slot has a multiplier");
    assert!(rm.per_sigma[1].is_none(), "add slot is bare");

    // theta = [.., RUV_LATE = 1.5]; index 2 is RUV_LATE.
    let theta = vec![0.2, 10.0, 1.5];
    let cov = std::collections::HashMap::new();
    // Before 24 h the multiplier is 1; after, it is RUV_LATE.
    let m_early = rm.eval_obs(&theta, &cov, 10.0, 10.0);
    let m_late = rm.eval_obs(&theta, &cov, 30.0, 30.0);
    assert!((m_early[0] - 1.0).abs() < 1e-12);
    assert!((m_early[1] - 1.0).abs() < 1e-12);
    assert!((m_late[0] - 1.5).abs() < 1e-12);
    assert!((m_late[1] - 1.0).abs() < 1e-12);
}

/// #710: an `[individual_parameters]` expression that references a name
/// declared *later* in the same block must be a hard parse error, not a
/// silent `0.0` substitution (the pre-fix behaviour collapsed the formula).
#[test]
fn indiv_params_forward_reference_is_error() {
    let content = r#"
[parameters]
  theta TVCL(0.2)
  theta TVV(10.0)
  theta TVIMAX(-0.3)
  omega ETA_CL ~ 0.09
  sigma ADD_ERR ~ 1.0
[individual_parameters]
  CL   = TVCL * exp(IMAX) * exp(ETA_CL)
  V    = TVV
  IMAX = TVIMAX
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ additive(ADD_ERR)
"#;
    let err = parse_err(content);
    assert!(
        err.contains("forward reference") && err.contains("IMAX"),
        "expected forward-reference error naming IMAX, got: {err}"
    );
}

/// #710: reordering so the referenced name is declared first parses cleanly —
/// the guard keys on declaration *order*, not mere presence.
#[test]
fn indiv_params_declaration_before_use_parses() {
    let content = r#"
[parameters]
  theta TVCL(0.2)
  theta TVV(10.0)
  theta TVIMAX(-0.3)
  omega ETA_CL ~ 0.09
  sigma ADD_ERR ~ 1.0
[individual_parameters]
  IMAX = TVIMAX
  CL   = TVCL * exp(IMAX) * exp(ETA_CL)
  V    = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ additive(ADD_ERR)
"#;
    parse_model_string(content).expect("declaration-before-use must parse");
}

/// #710: the order guard must not false-positive on a name assigned in an
/// earlier statement and read later — the normal backward-reference case.
#[test]
fn indiv_params_backward_reference_parses() {
    let content = r#"
[parameters]
  theta TVCL(0.2)
  theta TVV(10.0)
  omega ETA_CL ~ 0.09
  sigma ADD_ERR ~ 1.0
[individual_parameters]
  KE = TVCL / TVV
  CL = TVCL * KE * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ additive(ADD_ERR)
"#;
    parse_model_string(content).expect("backward reference must parse");
}

/// #710 (unit): `collect_forward_refs` walks statement order and flags only a
/// read of a not-yet-assigned in-block name. A name assigned across both arms
/// of an `if` is available afterwards (no false positive).
#[test]
fn collect_forward_refs_detects_order_and_respects_branches() {
    use BinOp::Mul;
    let in_block: HashSet<String> = ["A", "B", "C"].iter().map(|s| s.to_string()).collect();

    // B = A * 2 (A not yet assigned) — forward reference.
    let bad = vec![
        Statement::Assign(
            "B".to_string(),
            Expression::BinOp(
                Box::new(Expression::Variable("A".to_string())),
                Mul,
                Box::new(Expression::Literal(2.0)),
            ),
        ),
        Statement::Assign("A".to_string(), Expression::Literal(1.0)),
    ];
    let refs = collect_forward_refs(&bad, &in_block);
    assert_eq!(refs, vec![("A".to_string(), "B".to_string())]);

    // A assigned on both arms of an if, then read — no violation.
    let ok = vec![
        Statement::If {
            branches: vec![(
                Condition::Compare(Expression::Time, CmpOp::Gt, Expression::Literal(0.0)),
                vec![Statement::Assign("A".to_string(), Expression::Literal(1.0))],
            )],
            else_body: Some(vec![Statement::Assign(
                "A".to_string(),
                Expression::Literal(2.0),
            )]),
        },
        Statement::Assign("C".to_string(), Expression::Variable("A".to_string())),
    ];
    assert!(collect_forward_refs(&ok, &in_block).is_empty());
}

/// A theta referenced only from an `[error_model]` magnitude expression
/// (e.g. `RUV_LATE`, never used in `[individual_parameters]`) must not trip
/// the "declared but not referenced" warning — it *is* meaningfully
/// estimated, via the magnitude's direct-θ channel (#484/#576/#486).
#[test]
fn test_ruv_magnitude_theta_not_flagged_as_unused() {
    let content = r#"
[parameters]
  theta TVCL(0.2)
  theta TVV(10.0)
  theta RUV_LATE(1.5, 0.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR * (1.0 + RUV_LATE * TIME / 48.0))
"#;
    let model = parse_model_string(content).unwrap();
    assert!(model.has_custom_ruv_magnitude());
    let warns: Vec<_> = model
        .parse_warnings
        .iter()
        .filter(|w| w.contains("RUV_LATE"))
        .collect();
    assert!(
        warns.is_empty(),
        "RUV_LATE is used by the magnitude expression, got: {:?}",
        warns
    );
}

/// xhigh review: `MACHEPS` inside a magnitude expression must resolve to
/// `f64::EPSILON` in the `Dual1` θ-derivative program too (not just the
/// value-only runtime closure) — `RuvMagDerivProgram` reserves a var slot
/// for it now. Regression case: `WT + MACHEPS` guards a denominator; at
/// `WT = 0.0` the pre-fix bug (`MACHEPS` silently resolving to `0.0` in the
/// bytecode path) would divide by exactly zero and produce a non-finite
/// `∂mult/∂θ`.
#[test]
fn test_ruv_magnitude_macheps_resolves_in_theta_grad() {
    let content = r#"
[parameters]
  theta TVCL(0.2)
  theta TVV(10.0)
  theta RUV_LATE(1.5, 0.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR * (1.0 + RUV_LATE / (WT + MACHEPS)))
[covariates]
  WT continuous
"#;
    let model = parse_model_string(content).unwrap();
    let rm = model.ruv_magnitude.as_ref().expect("magnitude present");
    let deriv = rm.per_sigma_deriv[0]
        .as_ref()
        .expect("prop slot has a deriv program");

    let theta = vec![0.2, 10.0, 1.5];
    let cov_zero = std::collections::HashMap::from([("WT".to_string(), 0.0)]);
    let grad = deriv
        .theta_grad(&theta, &cov_zero, 10.0, 10.0)
        .expect("theta_grad supported");
    assert!(
        grad.iter().all(|g| g.is_finite()),
        "MACHEPS must guard the WT=0 denominator, got {:?}",
        grad
    );
    // ∂mult/∂RUV_LATE = 1/(WT+MACHEPS) = 1/EPSILON — a large but finite value.
    approx::assert_relative_eq!(grad[2], 1.0 / f64::EPSILON, max_relative = 1e-9);
}

/// xhigh review: a *legacy* pre-built magnitude expression that still
/// represents `TIME` as an `Expression::Covariate("TIME")` node (rather than
/// the `Expression::Time` built-in every parsed `.ferx` model produces) must
/// still resolve `TIME` to the observation time in `theta_grad`, mirroring the
/// runtime `RuvMagFn` closure's `cov.insert("TIME", time)` injection. Without
/// it, `cov_vec` would look up "TIME" in the (never-populated) observation
/// covariate map and silently default to `0.0`.
#[test]
fn test_ruv_magnitude_deriv_program_resolves_time_as_covariate_node() {
    // RUV_LATE * TIME, built directly as an AST (bypassing the parser, which
    // would emit `Expression::Time` instead) to exercise the legacy branch.
    let expr = Expression::BinOp(
        Box::new(Expression::Theta(0)),
        BinOp::Mul,
        Box::new(Expression::Covariate("TIME".to_string())),
    );
    let deriv = compile_ruv_mag_deriv_program(&expr, Some("UNUSED_SIGMA"), 1);
    let empty_cov = std::collections::HashMap::new();
    let grad = deriv
        .theta_grad(&[2.0], &empty_cov, 10.0, 10.0)
        .expect("theta_grad supported");
    // d(RUV_LATE * TIME)/d(RUV_LATE) = TIME = 10.0, not 0.0.
    approx::assert_relative_eq!(grad[0], 10.0, max_relative = 1e-12);
}

#[test]
fn test_ruv_magnitude_rejects_eta_dependence() {
    // A magnitude may not depend on a random effect (it must be η-independent).
    let content = r#"
[parameters]
  theta TVCL(0.2)
  theta TVV(10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR * exp(ETA_CL))
"#;
    let err = expect_parse_err(content);
    assert!(err.contains("random effect"), "got: {err}");
}

#[test]
fn test_ruv_magnitude_rejects_undeclared_covariate() {
    // With a [covariates] block, a magnitude covariate must be declared.
    let content = r#"
[parameters]
  theta TVCL(0.2)
  theta TVV(10.0)
  theta RUV_K(1.2)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR * (1.0 + RUV_K * NOTACOV))
[covariates]
  WT continuous
"#;
    let err = expect_parse_err(content);
    assert!(err.contains("undeclared covariate"), "got: {err}");
}

#[test]
fn test_ruv_magnitude_rejects_covariate_without_covariates_block() {
    // #484 review #5: with NO [covariates] block, a covariate reference in a
    // magnitude expression must still be rejected — otherwise an undeclared
    // name (or a typo) silently evaluates to 0 and collapses the multiplier
    // to a constant, running a fit that does nothing of what was written.
    let content = r#"
[parameters]
  theta TVCL(0.2)
  theta TVV(10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR * (1.0 + 0.5 * WT))
"#;
    let err = expect_parse_err(content);
    assert!(err.contains("undeclared covariate"), "got: {err}");
}

#[test]
fn test_parse_full_model_block_sigma_single_endpoint_order_mismatch_errs() {
    // Single-endpoint error models use the leading sigma slots positionally, so
    // a name order that would bind the proportional/additive components
    // backwards is rejected. `block_sigma` declares its entries into the same
    // flat sigma vector, so it reaches the same rule as plain `sigma` lines —
    // this was the one case the order was enforced in before #1001 generalised
    // it to every single-endpoint model.
    let content = r#"
[parameters]
  theta TVCL(0.2)
  theta TVV(10.0)
  omega ETA_CL ~ 0.09
  block_sigma (ADD_ERR, PROP_ERR) = [1.0, 0.10, 0.04] FIX
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)
"#;
    let err = expect_parse_err(content);
    // Pin the argument index, not just a name: `PROP_ERR` appears in the
    // argument-1 message (`named`) *and* in the argument-2 message (`slot`), so
    // asserting the name alone would still pass if the enumerate loop reported
    // the wrong argument.
    assert!(
        err.contains("consumed positionally")
            && err.contains("argument 1 names sigma 'PROP_ERR'")
            && err.contains("supplies 'ADD_ERR'"),
        "got: {err}"
    );
    // The block_sigma-specific half of the remedy: reordering the names without
    // permuting the lower triangle parses clean and silently swaps the two
    // variances, so the message must say so.
    assert!(err.contains("lower triangle"), "got: {err}");
}

// ── #1001: single-endpoint sigma names must match declaration order ──────────
//
// A single-endpoint `[error_model]` consumes its sigmas positionally from the
// `[parameters]` declaration order, so the names written in the arguments are
// only meaningful if they *are* that order. Before #1001 those names were
// validated (the sigma had to exist) and then discarded, so a model naming them
// in any other order fitted silently against different sigmas than it reads as
// describing. These pin the parse-time rejection that replaced the silence.

/// A one-compartment analytical model with the given `[parameters]` sigma
/// declarations and `[error_model]` statement.
fn sigma_order_model(sigma_decls: &str, error_stmt: &str) -> String {
    format!(
        r"
[parameters]
  theta TVCL(0.2)
  theta TVV(10.0)
  omega ETA_CL ~ 0.09
{sigma_decls}
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  {error_stmt}
"
    )
}

#[test]
fn single_error_model_wrong_sigma_name_errs() {
    // The #1001 reproducer, in the shape a parse test can hold: before the fix
    // `proportional(S_SMALL)` bound `S_BIG` because `S_BIG` was declared first.
    // The objective that misbinding cost is measured — on the anchor dataset,
    // against NONMEM — in `tests/sigma_order_nonmem_anchor.rs` (103.828964 vs
    // −6.838931, a 110.67-unit span); this test only pins the rejection, so it
    // deliberately quotes no number of its own.
    let content = sigma_order_model(
        "  sigma S_BIG ~ 2.0 (sd)\n  sigma S_SMALL ~ 0.05 (sd)",
        "DV ~ proportional(S_SMALL)",
    );
    let err = expect_parse_err(&content);
    assert!(
        err.contains("consumed positionally") && err.contains("S_SMALL") && err.contains("S_BIG"),
        "got: {err}"
    );
}

#[test]
fn single_error_model_declaration_order_match_parses() {
    // The control for the test above: the same file with the declarations in
    // the order the `[error_model]` names them is accepted unchanged.
    let content = sigma_order_model(
        "  sigma S_SMALL ~ 0.05 (sd)\n  sigma S_BIG ~ 2.0 (sd)",
        "DV ~ proportional(S_SMALL)",
    );
    parse_full_model(&content).expect("declaration order matches the argument order");
}

#[test]
fn single_error_model_combined_transposed_args_err() {
    // `combined(prop, add)` is positional in both directions: transposing the
    // two arguments used to be a no-op that silently swapped which sigma played
    // the proportional role.
    let content = sigma_order_model(
        "  sigma ADD_ERR ~ 1.0 (sd)\n  sigma PROP_ERR ~ 0.04 (sd)",
        "DV ~ combined(PROP_ERR, ADD_ERR)",
    );
    let err = expect_parse_err(&content);
    assert!(
        err.contains("consumed positionally") && err.contains("PROP_ERR"),
        "got: {err}"
    );
}

#[test]
fn single_error_model_repeated_sigma_arg_errs() {
    // `combined(S_A, S_A)` parsed and then bound slot 1 to whatever was
    // declared second — here `S_B`, which the model never names.
    //
    // Reported as a repeat, *not* as an order mismatch: duplicate `sigma`
    // declarations are rejected upstream, so no permutation of `[S_A, S_B]`
    // puts `S_A` in both slots and transposing two identical arguments is a
    // no-op. Telling the user to reorder here — and handing a consumer the
    // reorder-shaped `E_SIGMA_ORDER_MISMATCH` — would prescribe a fix that
    // cannot work, which is exactly what the count arm below avoids for the
    // one-sigma spelling of the same defect.
    let content = sigma_order_model(
        "  sigma S_A ~ 0.05 (sd)\n  sigma S_B ~ 1.0 (sd)",
        "DV ~ combined(S_A, S_A)",
    );
    let err = expect_parse_err(&content);
    assert!(
        err.contains("argument 2 names sigma 'S_A'")
            && err.contains("argument 1 already claims")
            && !err.contains("consumed positionally")
            && !err.contains("Reorder"),
        "got: {err}"
    );
}

#[test]
fn single_error_model_repeated_sigma_arg_without_a_second_sigma_errs() {
    // The same repetition with only one sigma declared reaches the same arm and
    // reports the same defect — one message, one code, for both spellings — and
    // still carries the counts, which are the actionable part here: there is no
    // second sigma to name yet.
    let content = sigma_order_model("  sigma S_A ~ 0.05 (sd)", "DV ~ combined(S_A, S_A)");
    let err = expect_parse_err(&content);
    assert!(
        err.contains("argument 1 already claims")
            && err.contains("needs 2, [parameters] declares 1")
            && !err.contains("consumed positionally"),
        "got: {err}"
    );
}

#[test]
fn single_error_model_trailing_unused_sigma_parses() {
    // A sigma declared *after* the ones the error model names occupies a slot the
    // error model never loads, so the order rule does not reach it. It still earns
    // the pre-existing "not referenced in [error_model]" warning.
    let content = sigma_order_model(
        "  sigma PROP_ERR ~ 0.04 (sd)\n  sigma EPSCOV ~ 1.0 (sd)",
        "DV ~ proportional(PROP_ERR)",
    );
    let parsed = parse_full_model(&content).expect("a trailing unused sigma is not an order error");
    assert!(
        parsed
            .model
            .parse_warnings
            .iter()
            .any(|w| w.contains("EPSCOV") && w.contains("not referenced")),
        "got: {:?}",
        parsed.model.parse_warnings
    );
}

#[test]
fn single_error_model_frem_shape_parses() {
    // The real FREM shape, which the test above only approximates: `prepare_frem`
    // (`src/frem/mod.rs`) copies the base model's sigmas in order and then appends
    // `sigma EPSCOV`, wiring it up through `[fit_options] frem_sigma` rather than
    // `[error_model]`. This pins that the *spelling* survives the #1001 order
    // rule — a trailing sigma consumed by a `[fit_options]` key rather than by
    // `[error_model]` is still accepted.
    //
    // It does NOT pin `generate_frem_model`'s emission order, because it
    // hand-writes the text rather than generating it. The end-to-end pin is
    // `frem::tests::test_generate_frem_model_preserves_scaling_block`
    // (`src/frem/mod.rs`), which runs `generate_frem_model` on a base model with
    // `sigma PROP_ERR` + `DV ~ proportional(PROP_ERR)` and parses the result: if
    // EPSCOV were ever emitted before the copied base sigmas, that test — not
    // this one — is what fails.
    let content = r"
[parameters]
  theta TVCL(0.2)
  theta TVV(10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02
  sigma EPSCOV ~ 1e-6 FIX
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  frem_sigma = EPSCOV
";
    parse_full_model(content).expect("a generated FREM model keeps its trailing covariate sigma");
}

#[test]
fn single_error_model_magnitude_expression_respects_declaration_order() {
    // #484 magnitude expressions inherit the positional binding: the multiplier
    // is stored per argument slot and applied to that slot's sigma, so an
    // expression scaling `S_SMALL` in argument 1 was applied to `S_BIG`.
    let content = sigma_order_model(
        "  sigma S_BIG ~ 2.0 (sd)\n  sigma S_SMALL ~ 0.05 (sd)",
        "DV ~ proportional(S_SMALL * 2.0)",
    );
    let err = expect_parse_err(&content);
    assert!(
        err.contains("consumed positionally") && err.contains("S_SMALL"),
        "got: {err}"
    );
}

#[test]
fn single_error_model_ltbs_respects_declaration_order() {
    // The LTBS spelling reaches the same `ParsedErrorModel::Single`, so it must
    // be held to the same rule rather than slipping past on its own regex.
    let content = sigma_order_model(
        "  sigma S_BIG ~ 2.0 (sd)\n  sigma ADD_LOG ~ 0.1 (sd)",
        "log(DV) ~ additive(ADD_LOG)",
    );
    let err = expect_parse_err(&content);
    assert!(
        err.contains("consumed positionally") && err.contains("ADD_LOG"),
        "got: {err}"
    );
}

#[test]
fn test_parse_full_model_block_sigma_cross_endpoint_covariance_builds_correlation() {
    // Cross-endpoint covariances are carried by the subject-level residual
    // covariance matrix for paired observations.
    let content = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  block_sigma (S_PROP, S_PD) = [0.04, 0.10, 1.0] FIX
  sigma S_ADD ~ 1.0 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(states=[central, effect])

[odes]
  d/dt(central) = -CL/V * central
  d/dt(effect)  =  central/V - effect

[scaling]
  y[CMT=1] = central / V
  y[CMT=2] = effect

[error_model]
  CMT=1: DV ~ combined(S_PROP, S_ADD)
  CMT=2: DV ~ additive(S_PD)
"#;
    let parsed = parse_full_model(content).expect("cross-endpoint block_sigma should parse");
    assert_eq!(parsed.model.residual_correlations.len(), 1);
}

#[test]
fn test_parse_full_model_duplicate_sigma_name_errs() {
    // A plain sigma colliding with a block_sigma entry is rejected.
    let content = r#"
[parameters]
  theta TVCL(0.2)
  theta TVV(10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02
  block_sigma (PROP_ERR, ADD_ERR) = [0.04, 0.10, 1.0] FIX
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)
"#;
    let Err(err) = parse_full_model(content) else {
        panic!("expected a duplicate-sigma-name error");
    };
    assert!(err.contains("more than once"), "got: {err}");
}

#[test]
fn test_build_omega_fixed_diagonal() {
    let diag = vec![
        OmegaSpec {
            name: "ETA_CL".into(),
            variance: 0.09,
            fixed: true,
            init_as_sd: false,
        },
        OmegaSpec {
            name: "ETA_V".into(),
            variance: 0.04,
            fixed: false,
            init_as_sd: false,
        },
    ];
    let names = vec!["ETA_CL".into(), "ETA_V".into()];
    let flags = build_omega_fixed(&diag, &[], &names).unwrap();
    assert_eq!(flags, vec![true, false]);
}

#[test]
fn test_build_omega_fixed_block() {
    let block = vec![BlockOmegaSpec {
        names: vec!["ETA_CL".into(), "ETA_V".into()],
        lower_triangle: vec![0.09, 0.02, 0.04],
        fixed: true,
    }];
    let names = vec!["ETA_CL".into(), "ETA_V".into()];
    let flags = build_omega_fixed(&[], &block, &names).unwrap();
    assert_eq!(flags, vec![true, true]);
}

#[test]
fn test_build_omega_fixed_rejects_diag_fix_inside_free_block() {
    // ETA_CL is in a non-FIX block but also declared FIX as a diagonal —
    // the parser must reject this as ambiguous.
    let diag = vec![OmegaSpec {
        name: "ETA_CL".into(),
        variance: 0.09,
        fixed: true,
        init_as_sd: false,
    }];
    let block = vec![BlockOmegaSpec {
        names: vec!["ETA_CL".into(), "ETA_V".into()],
        lower_triangle: vec![0.09, 0.02, 0.04],
        fixed: false,
    }];
    let names = vec!["ETA_CL".into(), "ETA_V".into()];
    let res = build_omega_fixed(&diag, &block, &names);
    assert!(res.is_err());
}

#[test]
fn test_parse_full_model_with_fix() {
    let content = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, FIX)
  theta TVKA(1.5, 0.01, 50.0)

  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04 FIX
  omega ETA_KA ~ 0.30

  sigma PROP_ERR ~ 0.02 FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let parsed = parse_full_model(content).unwrap();
    let p = &parsed.model.default_params;
    assert_eq!(p.theta_fixed, vec![false, true, false]);
    assert_eq!(p.omega_fixed, vec![false, true, false]);
    assert_eq!(p.sigma_fixed, vec![true]);
    assert!(p.has_any_fixed());
}

/// Build a minimal one-cpt model, optionally with a `[covariates]` block and
/// an optional covariate reference in the CL expression.
fn model_with_covariates(cov_block: Option<&str>, cl_uses_wt: bool) -> String {
    let cl_expr = if cl_uses_wt {
        "CL = TVCL * (WT / 70.0) * exp(ETA_CL)"
    } else {
        "CL = TVCL * exp(ETA_CL)"
    };
    let cov = cov_block
        .map(|b| format!("\n[covariates]\n{}\n", b))
        .unwrap_or_default();
    format!(
        r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02
{cov}
[individual_parameters]
  {cl_expr}
  V  = TVV
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
"#
    )
}

#[test]
fn test_covariates_absent_is_none() {
    let parsed = parse_full_model(&model_with_covariates(None, false)).unwrap();
    assert!(parsed.covariate_decls.is_none());
}

#[test]
fn test_covariates_per_line_form() {
    let parsed = parse_full_model(&model_with_covariates(
        Some("WT continuous\nSEX categorical"),
        true,
    ))
    .unwrap();
    let decls = parsed.covariate_decls.expect("declared");
    assert_eq!(decls.len(), 2);
    assert_eq!(decls[0].name, "WT");
    assert_eq!(decls[0].kind, CovariateKind::Continuous);
    assert_eq!(decls[1].name, "SEX");
    assert_eq!(decls[1].kind, CovariateKind::Categorical);
}

#[test]
fn test_covariates_colon_form_and_order() {
    let parsed = parse_full_model(&model_with_covariates(
        Some("continuous: WT, HT, CRCL\ncategorical: SEX"),
        true,
    ))
    .unwrap();
    let decls = parsed.covariate_decls.expect("declared");
    let names: Vec<&str> = decls.iter().map(|d| d.name.as_str()).collect();
    // Declaration order preserved across the two lines.
    assert_eq!(names, vec!["WT", "HT", "CRCL", "SEX"]);
    assert_eq!(decls[3].kind, CovariateKind::Categorical);
}

#[test]
fn test_covariates_type_aliases() {
    let parsed = parse_full_model(&model_with_covariates(
        Some("WT cont\nSEX cat\ncont: HT"),
        true,
    ))
    .unwrap();
    let decls = parsed.covariate_decls.expect("declared");
    assert_eq!(decls[0].kind, CovariateKind::Continuous);
    assert_eq!(decls[1].kind, CovariateKind::Categorical);
    assert_eq!(decls[2].kind, CovariateKind::Continuous);
}

#[test]
fn test_covariates_duplicate_errors() {
    let err = parse_full_model(&model_with_covariates(
        Some("WT continuous\nWT categorical"),
        true,
    ))
    .err()
    .unwrap();
    assert!(err.contains("more than once"), "got: {err}");
}

#[test]
fn test_covariates_unknown_type_errors() {
    let err = parse_full_model(&model_with_covariates(Some("WT numeric"), true))
        .err()
        .unwrap();
    assert!(err.contains("unknown covariate type"), "got: {err}");
}

#[test]
fn test_time_builtin_in_conditional_rhs_sets_uses_time_flag() {
    // Regression: the `TIME` built-in used inside a conditional-expression
    // RHS (`STEP = if (TIME > 5) 1 else 0`) must flag the model as using
    // TIME, so a time-dependent individual parameter routes through the
    // per-event evaluation path. The flag was computed AFTER
    // `resolve_variable_indices` bytecode-compiled the statements, which
    // erased the `Expression::Time` node the scan looks for — so it read
    // false and the analytical path evaluated PK params once at t=0 (the
    // parameter never switched; e.g. infliximab's maintenance-phase CL
    // multiplier became unidentifiable).
    let model = "[parameters]\n  \
                       theta TVCL (1.0, 0.01, 100.0)\n  \
                       theta TVV (10.0, 0.1, 500.0)\n  \
                       omega ETA_CL ~ 0.1\n  \
                       sigma PROP ~ 0.1\n\
                     [individual_parameters]\n  \
                       STEP = if (TIME > 5) 1 else 0\n  \
                       CL = TVCL * (1 + STEP) * exp(ETA_CL)\n  \
                       V = TVV\n\
                     [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n\
                     [error_model]\n  DV ~ proportional(PROP)\n";
    let compiled = parse_model_string(model).unwrap();
    assert!(
        compiled_model_uses_time_builtin(&compiled),
        "TIME inside a conditional-expression RHS must set uses_time_builtin"
    );
}

#[test]
fn test_no_time_builtin_leaves_uses_time_flag_unset() {
    // Counterpart: a model with no TIME reference must not be flagged (it
    // stays on the cheap single-snapshot analytical path).
    let model = "[parameters]\n  \
                       theta TVCL (1.0, 0.01, 100.0)\n  \
                       theta TVV (10.0, 0.1, 500.0)\n  \
                       omega ETA_CL ~ 0.1\n  \
                       sigma PROP ~ 0.1\n\
                     [individual_parameters]\n  \
                       CL = TVCL * exp(ETA_CL)\n  \
                       V = TVV\n\
                     [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n\
                     [error_model]\n  DV ~ proportional(PROP)\n";
    let compiled = parse_model_string(model).unwrap();
    assert!(
        !compiled_model_uses_time_builtin(&compiled),
        "a model with no TIME reference must not set uses_time_builtin"
    );
}

#[test]
fn test_covariates_referenced_but_undeclared_warns_not_errors() {
    // CL uses WT, but only SEX is declared. This is allowed (WT is still
    // usable) — the parser warns rather than erroring.
    let parsed = parse_full_model(&model_with_covariates(Some("SEX categorical"), true)).unwrap();
    // WT is still declared as known? No — only SEX is. But the parse succeeds.
    let decls = parsed.covariate_decls.expect("declared");
    assert_eq!(decls.len(), 1);
    assert_eq!(decls[0].name, "SEX");
    // A warning names the undeclared covariate.
    assert!(
        parsed
            .model
            .parse_warnings
            .iter()
            .any(|w| w.contains("WT") && w.contains("not declared")),
        "expected a warning about undeclared WT, got: {:?}",
        parsed.model.parse_warnings
    );
}

#[test]
fn test_covariates_declared_unreferenced_is_ok() {
    // Declaring a covariate the model doesn't use is the whole point
    // ("potentially available") — must not error.
    let parsed = parse_full_model(&model_with_covariates(Some("WT continuous"), false)).unwrap();
    assert_eq!(parsed.covariate_decls.expect("declared").len(), 1);
}

#[test]
fn test_parse_full_model_with_block_omega() {
    let content = r#"
# Test model with block omega

[parameters]
  theta TVCL(0.134, 0.001, 10.0)
  theta TVV(8.1, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)

  block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04]
  omega ETA_KA ~ 0.40

  sigma PROP_ERR ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let parsed = parse_full_model(content).unwrap();
    let omega = &parsed.model.default_params.omega;
    assert_eq!(omega.dim(), 3);
    assert!(!omega.diagonal);
    // Eta names preserve declaration order from the model file
    assert_eq!(omega.eta_names, vec!["ETA_CL", "ETA_V", "ETA_KA"]);
    // ETA_CL = index 0, ETA_V = index 1, ETA_KA = index 2
    assert!((omega.matrix[(0, 0)] - 0.09).abs() < 1e-10); // ETA_CL
    assert!((omega.matrix[(1, 1)] - 0.04).abs() < 1e-10); // ETA_V
    assert!((omega.matrix[(2, 2)] - 0.40).abs() < 1e-10); // ETA_KA
    assert!((omega.matrix[(0, 1)] - 0.02).abs() < 1e-10); // cov(CL, V)
}

// ── fit_options parsing: new optimizer choices ──────────────────────────

fn minimal_model_with_fit_options(fit_opts: &str) -> String {
    format!(
        r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
{}
"#,
        fit_opts
    )
}

/// Build a 1-cpt oral model with a custom `[error_model]` block (and one
/// extra sigma so combined/per-CMT cases have enough sigmas to reference).
fn model_with_error_block(error_block: &str) -> Result<crate::types::CompiledModel, String> {
    let content = format!(
        r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  sigma SIG1 ~ 0.02
  sigma SIG2 ~ 0.1

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
{error_block}
"#
    );
    parse_model_string(&content)
}

#[test]
fn test_ltbs_log_dv_additive_case2() {
    // `log(DV) ~ additive(...)`: engine logs DV (natural-scale data).
    let m = model_with_error_block("  log(DV) ~ additive(SIG1)").unwrap();
    assert!(
        m.log_transform,
        "log(DV) ~ additive should set log_transform"
    );
    assert!(
        !m.dv_pre_logged,
        "log(DV) ~ additive: engine logs DV, so dv_pre_logged must be false"
    );
    assert_eq!(m.error_model, ErrorModel::Additive);
}

#[test]
fn test_ltbs_log_additive_case1() {
    // `DV ~ log_additive(...)`: DV already log; only the prediction is logged.
    let m = model_with_error_block("  DV ~ log_additive(SIG1)").unwrap();
    assert!(m.log_transform, "log_additive should set log_transform");
    assert!(
        m.dv_pre_logged,
        "log_additive: DV is already log, so dv_pre_logged must be true"
    );
    assert_eq!(m.error_model, ErrorModel::Additive);
}

#[test]
fn test_non_ltbs_additive_unaffected() {
    let m = model_with_error_block("  DV ~ additive(SIG1)").unwrap();
    assert!(!m.log_transform);
    assert!(!m.dv_pre_logged);
}

#[test]
fn test_ltbs_rejects_proportional() {
    let err = model_with_error_block("  log(DV) ~ proportional(SIG1)").unwrap_err();
    assert!(
        err.contains("additive"),
        "expected additive-only message, got: {err}"
    );
}

#[test]
fn test_ltbs_rejects_double_log() {
    let err = model_with_error_block("  log(DV) ~ log_additive(SIG1)").unwrap_err();
    assert!(
        err.contains("double-log"),
        "expected double-log rejection, got: {err}"
    );
}

#[test]
fn test_ltbs_rejects_non_dv_lhs() {
    // `log(<not DV>) ~ additive(...)` would parse silently but the engine
    // always log-transforms the `DV` column, so the LHS is misleading.
    let err = model_with_error_block("  log(CONC) ~ additive(SIG1)").unwrap_err();
    assert!(
        err.contains("DV"),
        "expected DV-required rejection, got: {err}"
    );
}

#[test]
fn test_ltbs_rejects_per_cmt() {
    // LTBS is single-endpoint only.
    let err = model_with_error_block("  CMT=1: log(DV) ~ additive(SIG1)").unwrap_err();
    assert!(
        err.contains("per-CMT") || err.contains("multi-endpoint"),
        "expected per-CMT rejection, got: {err}"
    );
}

#[test]
fn test_parse_optimizer_bobyqa() {
    let content = minimal_model_with_fit_options("  optimizer = bobyqa");
    let parsed = parse_full_model(&content).unwrap();
    assert_eq!(parsed.fit_options.optimizer, Optimizer::Bobyqa);
}

#[test]
fn test_parse_optimizer_auto() {
    let content = minimal_model_with_fit_options("  optimizer = auto");
    let parsed = parse_full_model(&content).unwrap();
    assert_eq!(parsed.fit_options.optimizer, Optimizer::Auto);
}

#[test]
fn test_parse_optimizer_trust_region() {
    let content = minimal_model_with_fit_options("  optimizer = trust_region");
    let parsed = parse_full_model(&content).unwrap();
    assert_eq!(parsed.fit_options.optimizer, Optimizer::TrustRegion);
}

#[test]
fn test_parse_optimizer_newton_tr_alias() {
    // newton_tr is an accepted alias for trust_region
    let content = minimal_model_with_fit_options("  optimizer = newton_tr");
    let parsed = parse_full_model(&content).unwrap();
    assert_eq!(parsed.fit_options.optimizer, Optimizer::TrustRegion);
}

#[test]
fn test_parse_optimizer_lbfgs_bfgs_alias_nlopt() {
    // `lbfgs` and `bfgs` are deprecated aliases for `nlopt_lbfgs`: all three
    // select the NLopt L-BFGS path. The hand-rolled built-in is no longer
    // reachable by keyword. See #483.
    for key in ["lbfgs", "bfgs", "nlopt_lbfgs"] {
        let content = minimal_model_with_fit_options(&format!("  optimizer = {key}"));
        let parsed = parse_full_model(&content).unwrap();
        assert_eq!(
            parsed.fit_options.optimizer,
            Optimizer::NloptLbfgs,
            "optimizer = {key} should resolve to NLopt L-BFGS"
        );
    }
}

#[test]
fn test_parse_optimizer_case_insensitive() {
    // Parser lowercases the value, so mixed-case should map the same way.
    let content = minimal_model_with_fit_options("  optimizer = BOBYQA");
    let parsed = parse_full_model(&content).unwrap();
    assert_eq!(parsed.fit_options.optimizer, Optimizer::Bobyqa);

    let content2 = minimal_model_with_fit_options("  optimizer = Trust_Region");
    let parsed2 = parse_full_model(&content2).unwrap();
    assert_eq!(parsed2.fit_options.optimizer, Optimizer::TrustRegion);
}

#[test]
fn test_parse_optimizer_defaults_to_auto() {
    // `[fit_options]` block present but `optimizer` omitted → default is
    // `auto`, which resolves per-model to nlopt_lbfgs (analytic gradient) or
    // bobyqa (FD only). See `FitOptions::default` and #490.
    let content = minimal_model_with_fit_options("  maxiter = 100");
    let parsed = parse_full_model(&content).unwrap();
    assert_eq!(parsed.fit_options.optimizer, Optimizer::Auto);
}

#[test]
fn test_parse_steihaug_max_iters() {
    let content =
        minimal_model_with_fit_options("  optimizer = trust_region\n  steihaug_max_iters = 30");
    let parsed = parse_full_model(&content).unwrap();
    assert_eq!(parsed.fit_options.optimizer, Optimizer::TrustRegion);
    assert_eq!(parsed.fit_options.steihaug_max_iters, Some(30));
}

#[test]
fn test_steihaug_max_iters_default() {
    // Default is None (size-adaptive budget).
    let content = minimal_model_with_fit_options("  optimizer = trust_region");
    let parsed = parse_full_model(&content).unwrap();
    assert_eq!(parsed.fit_options.steihaug_max_iters, None);
}

#[test]
fn test_parse_inner_maxiter_and_tol() {
    let content = minimal_model_with_fit_options("  inner_maxiter = 75\n  inner_tol = 1e-5");
    let parsed = parse_full_model(&content).unwrap();
    assert_eq!(parsed.fit_options.inner_maxiter, 75);
    assert!((parsed.fit_options.inner_tol - 1e-5).abs() < 1e-15);
}

#[test]
fn test_parse_inner_restarts() {
    // Default is 1 (guarded multi-start on); the key overrides the count.
    let default = parse_full_model(&minimal_model_with_fit_options("  method = focei"))
        .unwrap()
        .fit_options
        .inner_restarts;
    assert_eq!(default, 1);
    let parsed = parse_full_model(&minimal_model_with_fit_options("  inner_restarts = 2")).unwrap();
    assert_eq!(parsed.fit_options.inner_restarts, 2);
    // `0` disables it.
    let off = parse_full_model(&minimal_model_with_fit_options("  inner_restarts = 0")).unwrap();
    assert_eq!(off.fit_options.inner_restarts, 0);
}

#[test]
fn test_fit_options_defaults() {
    // Guard against accidental drift in defaults — documented as:
    //   optimizer = auto, inner_maxiter = 200, inner_tol = 1e-5,
    //   steihaug_max_iters = None (adaptive).
    let opts = FitOptions::default();
    assert_eq!(opts.optimizer, Optimizer::Auto);
    assert_eq!(opts.inner_maxiter, 200);
    assert!((opts.inner_tol - 1e-5).abs() < 1e-20);
    assert_eq!(opts.inner_restarts, 1);
    assert_eq!(opts.steihaug_max_iters, None);
}

#[test]
fn test_parse_example_warfarin_bobyqa_file() {
    // The example file is part of the user-visible surface; parsing it is
    // a lightweight smoke test that the key names match what the docs
    // and examples advertise.
    let content = include_str!("../../examples/warfarin_bobyqa.ferx");
    let parsed = parse_full_model(content).unwrap();
    assert_eq!(parsed.fit_options.optimizer, Optimizer::Bobyqa);
    assert_eq!(parsed.fit_options.outer_maxiter, 300);
    assert_eq!(parsed.fit_options.inner_maxiter, 100);
}

#[test]
fn test_parse_example_warfarin_trust_region_file() {
    let content = include_str!("../../examples/warfarin_trust_region.ferx");
    let parsed = parse_full_model(content).unwrap();
    assert_eq!(parsed.fit_options.optimizer, Optimizer::TrustRegion);
    assert_eq!(parsed.fit_options.steihaug_max_iters, Some(30));
}

// ── apply_fit_option: coverage of the newly-added optimizer keys.
//    The generic apply_fit_option tests (known/unknown/malformed/bool
//    variants/threads/seed) live in the earlier test block — these
//    only add the keys that are new on this branch.

#[test]
fn test_apply_fit_option_optimizer_bobyqa() {
    let mut opts = FitOptions::default();
    assert_eq!(apply_fit_option(&mut opts, "optimizer", "bobyqa"), Ok(true));
    assert_eq!(opts.optimizer, Optimizer::Bobyqa);
}

#[test]
fn test_apply_fit_option_optimizer_trust_region() {
    let mut opts = FitOptions::default();
    assert_eq!(
        apply_fit_option(&mut opts, "optimizer", "trust_region"),
        Ok(true)
    );
    assert_eq!(opts.optimizer, Optimizer::TrustRegion);
}

#[test]
fn test_apply_fit_option_steihaug_max_iters() {
    let mut opts = FitOptions::default();
    assert_eq!(
        apply_fit_option(&mut opts, "steihaug_max_iters", "30"),
        Ok(true)
    );
    assert_eq!(opts.steihaug_max_iters, Some(30));
    // Reject malformed (e.g. negative) value.
    assert!(apply_fit_option(&mut opts, "steihaug_max_iters", "-1").is_err());
}

#[test]
fn test_parse_method_token_bayes() {
    assert_eq!(parse_method_token("bayes"), Ok(EstimationMethod::Bayes));
    assert_eq!(parse_method_token("BAYES"), Ok(EstimationMethod::Bayes));
    assert_eq!(parse_method_token("mcmc"), Ok(EstimationMethod::Bayes));
    assert_eq!(EstimationMethod::Bayes.label(), "BAYES");
}

#[test]
fn test_apply_fit_option_bayes_keys() {
    let mut opts = FitOptions::default();
    assert_eq!(apply_fit_option(&mut opts, "bayes_warmup", "500"), Ok(true));
    assert_eq!(opts.bayes_warmup, 500);
    assert_eq!(apply_fit_option(&mut opts, "bayes_iters", "2000"), Ok(true));
    assert_eq!(opts.bayes_iters, 2000);
    assert_eq!(apply_fit_option(&mut opts, "bayes_chains", "2"), Ok(true));
    assert_eq!(opts.bayes_chains, 2);
    assert_eq!(apply_fit_option(&mut opts, "bayes_thin", "5"), Ok(true));
    assert_eq!(opts.bayes_thin, 5);
    assert_eq!(apply_fit_option(&mut opts, "bayes_seed", "42"), Ok(true));
    assert_eq!(opts.bayes_seed, Some(42));
    assert!(apply_fit_option(&mut opts, "bayes_warmup", "oops").is_err());

    // All Bayes keys are recognised by method_specific_keys (no spurious
    // "unsupported key" warning when method = bayes).
    opts.method = EstimationMethod::Bayes;
    for k in [
        "bayes_warmup",
        "bayes_iters",
        "bayes_chains",
        "bayes_thin",
        "bayes_seed",
    ] {
        assert!(
            crate::types::method_specific_keys(EstimationMethod::Bayes).contains(&k),
            "method_specific_keys(Bayes) missing `{k}`"
        );
    }
}

#[test]
fn test_apply_fit_option_inner_maxiter_and_tol() {
    let mut opts = FitOptions::default();
    assert_eq!(apply_fit_option(&mut opts, "inner_maxiter", "75"), Ok(true));
    assert_eq!(opts.inner_maxiter, 75);

    assert_eq!(apply_fit_option(&mut opts, "inner_tol", "1e-5"), Ok(true));
    assert!((opts.inner_tol - 1e-5).abs() < 1e-15);

    assert!(apply_fit_option(&mut opts, "inner_maxiter", "oops").is_err());
    assert!(apply_fit_option(&mut opts, "inner_tol", "not_a_num").is_err());
}

#[test]
fn test_apply_fit_option_cov_inner_tol() {
    let mut opts = FitOptions::default();
    // Default is None (falls back to inner_tol via effective_cov_inner_tol).
    assert_eq!(FitOptions::default().cov_inner_tol, None);
    assert_eq!(
        apply_fit_option(&mut opts, "cov_inner_tol", "1e-9"),
        Ok(true)
    );
    assert_eq!(opts.cov_inner_tol, Some(1e-9));
    assert!(opts.user_set_keys.iter().any(|k| k == "cov_inner_tol"));
    assert!(apply_fit_option(&mut opts, "cov_inner_tol", "not_a_num").is_err());
}

#[test]
fn test_apply_fit_option_outer_xtol_ftol() {
    let mut opts = FitOptions::default();
    assert_eq!(apply_fit_option(&mut opts, "outer_xtol", "1e-5"), Ok(true));
    assert!((opts.outer_xtol - 1e-5).abs() < 1e-20);
    assert_eq!(apply_fit_option(&mut opts, "outer_ftol", "1e-9"), Ok(true));
    assert!((opts.outer_ftol.unwrap() - 1e-9).abs() < 1e-24);
    // Default is None (auto: 1e-8 for pure-TTE, 1e-6 otherwise).
    assert_eq!(FitOptions::default().outer_ftol, None);

    // Non-numeric, non-positive, and non-finite values are rejected.
    assert!(apply_fit_option(&mut opts, "outer_xtol", "not_a_num").is_err());
    assert!(apply_fit_option(&mut opts, "outer_ftol", "0").is_err());
    assert!(apply_fit_option(&mut opts, "outer_ftol", "-1e-8").is_err());
    assert!(apply_fit_option(&mut opts, "outer_xtol", "inf").is_err());
}

#[test]
fn test_apply_fit_option_fd_hessian_step_valid() {
    let mut opts = FitOptions::default();
    assert_eq!(
        apply_fit_option(&mut opts, "fd_hessian_step", "0.05"),
        Ok(true)
    );
    assert!((opts.fd_hessian_step - 0.05).abs() < 1e-15);
}

#[test]
fn test_apply_fit_option_fd_hessian_step_zero_rejected() {
    let mut opts = FitOptions::default();
    let err = apply_fit_option(&mut opts, "fd_hessian_step", "0").unwrap_err();
    assert!(
        err.contains("positive"),
        "error must mention 'positive': {err}"
    );
}

#[test]
fn test_apply_fit_option_fd_hessian_step_negative_rejected() {
    let mut opts = FitOptions::default();
    let err = apply_fit_option(&mut opts, "fd_hessian_step", "-0.01").unwrap_err();
    assert!(
        err.contains("positive"),
        "error must mention 'positive': {err}"
    );
}

// ── IOV: kappa keyword and iov_column ──────────────────────────────────

#[test]
fn test_parse_kappa_keyword() {
    let lines = vec!["kappa KAPPA_CL ~ 0.01".to_string()];
    let (_, _, _, _, _, _, ki, _, _, _) = parse_parameters(&lines, &Default::default()).unwrap();
    assert_eq!(ki.diagonal.len(), 1);
    assert_eq!(ki.diagonal[0].name, "KAPPA_CL");
    assert!((ki.diagonal[0].variance - 0.01).abs() < 1e-12);
    assert!(!ki.diagonal[0].fixed);
}

#[test]
fn test_parse_kappa_fix() {
    let lines = vec!["kappa KAPPA_V ~ 0.05 FIX".to_string()];
    let (_, _, _, _, _, _, ki, _, _, _) = parse_parameters(&lines, &Default::default()).unwrap();
    assert!(ki.diagonal[0].fixed);
}

#[test]
fn test_kappa_unfixed_no_annotation() {
    // Baseline: plain kappa with no FIX and no annotation — confirms the
    // group-numbering shift didn't regress the common case.
    let lines = vec!["kappa KAPPA_V ~ 0.05".to_string()];
    let (_, _, _, _, _, _, ki, _, _, _) = parse_parameters(&lines, &Default::default()).unwrap();
    assert!(!ki.diagonal[0].fixed);
    assert!(!ki.diagonal[0].init_as_sd);
    assert!((ki.diagonal[0].variance - 0.05).abs() < 1e-12);
}

#[test]
fn test_kappa_fix_before_sd_annotation() {
    // `FIX (sd)` — FIX before the scale annotation for kappa.
    let lines = vec!["kappa KAPPA_V ~ 0.30 FIX (sd)".to_string()];
    let (_, _, _, _, _, _, ki, _, _, _) = parse_parameters(&lines, &Default::default()).unwrap();
    let expected = 0.30 * 0.30;
    assert!((ki.diagonal[0].variance - expected).abs() < 1e-12);
    assert!(ki.diagonal[0].fixed);
    assert!(ki.diagonal[0].init_as_sd);
}

#[test]
fn test_kappa_appended_after_bsv_etas() {
    // kappa names must NOT appear in the BSV eta_names list returned
    // as the 5th element; they only appear in the 6th (kappas) element.
    let lines = vec![
        "omega ETA_CL ~ 0.09".to_string(),
        "kappa KAPPA_CL ~ 0.01".to_string(),
    ];
    let (_, _, _, _, _, bsv_etas, ki, _, _, _) =
        parse_parameters(&lines, &Default::default()).unwrap();
    assert_eq!(bsv_etas, vec!["ETA_CL"]);
    assert_eq!(ki.diagonal.len(), 1);
    assert_eq!(ki.diagonal[0].name, "KAPPA_CL");
}

#[test]
fn test_parse_full_model_with_kappa() {
    let content = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let parsed = parse_full_model(content).unwrap();
    let m = &parsed.model;
    assert_eq!(m.n_eta, 2); // BSV only
    assert_eq!(m.n_kappa, 1);
    assert_eq!(m.kappa_names, vec!["KAPPA_CL"]);
    assert!(m.default_params.omega_iov.is_some());
    let iov = m.default_params.omega_iov.as_ref().unwrap();
    assert_eq!(iov.dim(), 1);
    assert!((iov.matrix[(0, 0)] - 0.01).abs() < 1e-12);
}

#[test]
fn test_iov_column_fit_option() {
    let mut opts = FitOptions::default();
    assert_eq!(apply_fit_option(&mut opts, "iov_column", "OCC"), Ok(true));
    assert_eq!(opts.iov_column, Some("OCC".to_string()));
}

#[test]
fn test_iov_column_none_values() {
    let mut opts = FitOptions::default();
    apply_fit_option(&mut opts, "iov_column", "OCC").unwrap();
    apply_fit_option(&mut opts, "iov_column", "none").unwrap();
    assert!(opts.iov_column.is_none());
}

#[test]
fn test_iov_column_parsed_from_fit_options_block() {
    let content = minimal_model_with_fit_options("  iov_column = PERIOD");
    let parsed = parse_full_model(&content).unwrap();
    assert_eq!(parsed.fit_options.iov_column, Some("PERIOD".to_string()));
}

#[test]
fn test_iov_occasion_dose_and_column() {
    let mut opts = FitOptions::default();
    assert_eq!(
        apply_fit_option(&mut opts, "iov_occasion", "dose"),
        Ok(true)
    );
    assert_eq!(opts.iov_occasion, IovOccasionRule::PerDose);
    apply_fit_option(&mut opts, "iov_occasion", "column").unwrap();
    assert_eq!(opts.iov_occasion, IovOccasionRule::Column);
}

#[test]
fn test_iov_occasion_time_windows() {
    let mut opts = FitOptions::default();
    apply_fit_option(&mut opts, "iov_occasion", "time(24, 48)").unwrap();
    assert_eq!(
        opts.iov_occasion,
        IovOccasionRule::TimeWindows(vec![24.0, 48.0])
    );
}

#[test]
fn test_iov_occasion_rejects_bad_values() {
    let mut opts = FitOptions::default();
    // non-increasing breakpoints
    assert!(apply_fit_option(&mut opts, "iov_occasion", "time(48, 24)").is_err());
    // empty breakpoint list
    assert!(apply_fit_option(&mut opts, "iov_occasion", "time()").is_err());
    // unparseable breakpoint
    assert!(apply_fit_option(&mut opts, "iov_occasion", "time(a)").is_err());
    // unknown mode
    assert!(apply_fit_option(&mut opts, "iov_occasion", "weekly").is_err());
    // malformed time() (missing parens)
    assert!(apply_fit_option(&mut opts, "iov_occasion", "time 24").is_err());
}

// A leading `0` breakpoint (the `c(0, 24, 48)` habit) is redundant — the
// first occasion already spans `[-inf, edge1)` — and shifts every occasion
// up by one, so it is rejected with a message that names the fix.
#[test]
fn test_iov_occasion_rejects_zero_breakpoint() {
    let mut opts = FitOptions::default();
    let err = apply_fit_option(&mut opts, "iov_occasion", "time(0, 24, 48)")
        .expect_err("a 0 breakpoint must be rejected");
    assert!(
        err.contains("must not include `0`") && err.contains("implied"),
        "got: {err}"
    );
    // Also rejected as the sole breakpoint.
    assert!(apply_fit_option(&mut opts, "iov_occasion", "time(0)").is_err());
    // The non-zero form is still accepted.
    apply_fit_option(&mut opts, "iov_occasion", "time(24, 48)").unwrap();
}

#[test]
fn test_iov_occasion_parsed_from_fit_options_block() {
    let content = minimal_model_with_fit_options("  iov_occasion = time(24)");
    let parsed = parse_full_model(&content).unwrap();
    assert_eq!(
        parsed.fit_options.iov_occasion,
        IovOccasionRule::TimeWindows(vec![24.0])
    );
}

// ── block_kappa (Option B) ─────────────────────────────────────────────

#[test]
fn test_parse_block_kappa_syntax() {
    let lines = vec!["block_kappa (KAPPA_CL, KAPPA_V) = [0.01, 0.002, 0.005]".to_string()];
    let (_, _, _, _, _, _, ki, _, _, _) = parse_parameters(&lines, &Default::default()).unwrap();
    assert_eq!(ki.diagonal.len(), 0);
    assert_eq!(ki.block.len(), 1);
    assert_eq!(ki.block[0].names, vec!["KAPPA_CL", "KAPPA_V"]);
    assert_eq!(ki.block[0].lower_triangle, vec![0.01, 0.002, 0.005]);
    assert!(!ki.block[0].fixed);
    assert_eq!(ki.names_ordered, vec!["KAPPA_CL", "KAPPA_V"]);
}

#[test]
fn test_parse_block_kappa_fix() {
    let lines = vec!["block_kappa (KAPPA_CL, KAPPA_V) = [0.01, 0.002, 0.005] FIX".to_string()];
    let (_, _, _, _, _, _, ki, _, _, _) = parse_parameters(&lines, &Default::default()).unwrap();
    assert!(ki.block[0].fixed);
}

#[test]
fn test_parse_block_kappa_wrong_count_errors() {
    // 2 names → need 3 values, only 2 given
    let lines = vec!["block_kappa (KAPPA_CL, KAPPA_V) = [0.01, 0.002]".to_string()];
    assert!(parse_parameters(&lines, &Default::default()).is_err());
}

#[test]
fn test_parse_block_kappa_name_overlap_errors() {
    let lines = vec![
        "kappa KAPPA_CL ~ 0.01".to_string(),
        "block_kappa (KAPPA_CL, KAPPA_V) = [0.01, 0.002, 0.005]".to_string(),
    ];
    assert!(parse_parameters(&lines, &Default::default()).is_err());
}

#[test]
fn test_parse_full_model_with_block_kappa() {
    let content = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  block_kappa (KAPPA_CL, KAPPA_V) = [0.01, 0.002, 0.005]
  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V + KAPPA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let parsed = parse_full_model(content).unwrap();
    let m = &parsed.model;
    assert_eq!(m.n_eta, 2);
    assert_eq!(m.n_kappa, 2);
    assert_eq!(m.kappa_names, vec!["KAPPA_CL", "KAPPA_V"]);
    let iov = m.default_params.omega_iov.as_ref().unwrap();
    assert_eq!(iov.dim(), 2);
    assert!(
        !iov.diagonal,
        "block_kappa should produce non-diagonal omega_iov"
    );
    assert!((iov.matrix[(0, 0)] - 0.01).abs() < 1e-12);
    assert!((iov.matrix[(0, 1)] - 0.002).abs() < 1e-12);
    assert!((iov.matrix[(1, 0)] - 0.002).abs() < 1e-12);
    assert!((iov.matrix[(1, 1)] - 0.005).abs() < 1e-12);
}

// ── if-statement support ───────────────────────────────────────────────
//
// Tests the multi-line `if (cond) { ... } else if (...) { ... } else { ... }`
// syntax and the inline `if (cond) expr else expr` ternary. Coverage:
// tokenizer, statement parser, evaluator, integration with mu-ref
// detection, and ODE bodies.

fn empty_ctx() -> ParseCtx<'static> {
    const TN: &[String] = &[];
    const EN: &[String] = &[];
    const DV: &[String] = &[];
    ParseCtx::new(TN, EN, DV)
}

#[test]
fn test_tokenize_comparison_and_logical_operators() {
    let toks = tokenize("a >= 1 && !(b == 2 || c <= 3)").unwrap();
    // We only assert that the two-character ops landed as single tokens —
    // the rest of the stream is asserted indirectly via parser tests.
    assert!(toks.contains(&Token::Ge));
    assert!(toks.contains(&Token::AndAnd));
    assert!(toks.contains(&Token::Bang));
    assert!(toks.contains(&Token::EqEq));
    assert!(toks.contains(&Token::OrOr));
    assert!(toks.contains(&Token::Le));
}

#[test]
fn test_tokenize_newlines_and_braces() {
    let toks = tokenize("if (x > 0) {\n  y = 1\n}").unwrap();
    assert!(toks.contains(&Token::LBrace));
    assert!(toks.contains(&Token::RBrace));
    assert!(toks.contains(&Token::Newline));
    assert!(toks.contains(&Token::Eq));
    assert!(toks.contains(&Token::Gt));
}

#[test]
fn test_strip_newlines_keeps_brace_separators() {
    let toks = tokenize("if (a >\n  1) {\n  y = 2\n}").unwrap();
    let stripped = strip_newlines_in_groups(toks);
    // Newline inside the parens is gone; newlines inside the braces stay
    // (statement separators). Two newlines are typed inside the brace body
    // (one after `{` and one after `y = 2`); both survive.
    let n_newlines = stripped.iter().filter(|t| **t == Token::Newline).count();
    assert_eq!(n_newlines, 2);
}

#[test]
fn test_inline_if_expression_parses_and_evaluates() {
    let tn = vec!["TVCL".to_string()];
    let en: Vec<String> = vec![];
    let ctx = ParseCtx::new(&tn, &en, &[]);
    let stmts = parse_block_statements(
        "CL = if (WT > 70) TVCL * 1.2 else TVCL",
        ctx,
        StatementMode::Plain,
    )
    .unwrap();
    assert_eq!(stmts.len(), 1);

    let mut vars = HashMap::new();
    let theta = vec![5.0];
    let mut covs = HashMap::new();
    covs.insert("WT".to_string(), 80.0);
    eval_statements(&stmts, &theta, &[], &covs, &mut vars, None, None, &[]);
    assert!(
        (vars["CL"] - 6.0).abs() < 1e-12,
        "CL should pick the then-branch"
    );

    covs.insert("WT".to_string(), 60.0);
    let mut vars2 = HashMap::new();
    eval_statements(&stmts, &theta, &[], &covs, &mut vars2, None, None, &[]);
    assert!(
        (vars2["CL"] - 5.0).abs() < 1e-12,
        "CL should pick the else-branch"
    );
}

#[test]
fn test_multiline_if_else_block_evaluates() {
    let tn = vec!["TVCL".to_string()];
    let block = "
if (WT > 70) {
  CL = TVCL * (WT/70)
} else {
  CL = TVCL
}
";
    let ctx = ParseCtx::new(&tn, &[], &[]);
    let stmts = parse_block_statements(block, ctx, StatementMode::Plain).unwrap();
    assert_eq!(stmts.len(), 1, "single top-level if-statement");

    let theta = vec![10.0];
    for (wt, expected) in [(80.0, 10.0 * (80.0 / 70.0)), (50.0, 10.0)] {
        let mut covs = HashMap::new();
        covs.insert("WT".to_string(), wt);
        let mut vars = HashMap::new();
        eval_statements(&stmts, &theta, &[], &covs, &mut vars, None, None, &[]);
        assert!(
            (vars["CL"] - expected).abs() < 1e-12,
            "WT={} → expected CL={}",
            wt,
            expected
        );
    }
}

#[test]
fn test_else_if_chain_picks_first_match() {
    let tn = vec!["TVCL".to_string()];
    let block = "
if (X < 10) {
  CL = TVCL * 0.5
} else if (X < 20) {
  CL = TVCL
} else if (X < 30) {
  CL = TVCL * 1.5
} else {
  CL = TVCL * 2.0
}
";
    let ctx = ParseCtx::new(&tn, &[], &[]);
    let stmts = parse_block_statements(block, ctx, StatementMode::Plain).unwrap();
    let theta = vec![10.0];
    for (x, expected) in [(5.0, 5.0), (15.0, 10.0), (25.0, 15.0), (40.0, 20.0)] {
        let mut covs = HashMap::new();
        covs.insert("X".to_string(), x);
        let mut vars = HashMap::new();
        eval_statements(&stmts, &theta, &[], &covs, &mut vars, None, None, &[]);
        assert!(
            (vars["CL"] - expected).abs() < 1e-12,
            "X={x} should pick branch giving CL={expected}, got {}",
            vars["CL"]
        );
    }
}

#[test]
fn test_logical_operators_in_condition() {
    let tn = vec!["TVCL".to_string()];
    let block = "CL = if ((SEX == 1 && WT > 70) || AGE >= 65) TVCL * 1.5 else TVCL";
    let ctx = ParseCtx::new(&tn, &[], &[]);
    let stmts = parse_block_statements(block, ctx, StatementMode::Plain).unwrap();
    let theta = vec![10.0];
    let cases = [
        // (sex, wt, age, expected)
        (1.0, 80.0, 30.0, 15.0), // && true → boost
        (1.0, 60.0, 30.0, 10.0), // && false, || false → no boost
        (0.0, 50.0, 70.0, 15.0), // age >= 65 → boost
        (0.0, 50.0, 64.999, 10.0),
    ];
    for (sex, wt, age, expected) in cases {
        let mut covs = HashMap::new();
        covs.insert("SEX".to_string(), sex);
        covs.insert("WT".to_string(), wt);
        covs.insert("AGE".to_string(), age);
        let mut vars = HashMap::new();
        eval_statements(&stmts, &theta, &[], &covs, &mut vars, None, None, &[]);
        assert!(
            (vars["CL"] - expected).abs() < 1e-12,
            "sex={sex} wt={wt} age={age} expected CL={expected}, got {}",
            vars["CL"]
        );
    }
}

#[test]
fn test_if_statement_disables_mu_ref_for_var() {
    // CL is assigned only inside an if-block — it must NOT participate in
    // mu-referencing because the (eta, theta) relationship is conditional.
    // V is unconditional and SHOULD still mu-ref.
    let content = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)

  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04

  sigma PROP_ERR ~ 0.02

[individual_parameters]
  if (WT > 70) {
    CL = TVCL * (WT/70) * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let parsed = parse_full_model(content).expect("model should parse");
    // Only ETA_V participates in mu-referencing.
    assert!(parsed.model.mu_refs.contains_key("ETA_V"));
    assert!(!parsed.model.mu_refs.contains_key("ETA_CL"));
}

#[test]
fn test_if_statement_collects_covariates_from_all_branches() {
    // Covariates referenced inside any branch (including the condition)
    // must show up in the model's referenced_covariates list.
    let content = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)

  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04

  sigma PROP_ERR ~ 0.02

[individual_parameters]
  if (SEX == 1) {
    CL = TVCL * (WT/70) * exp(ETA_CL)
  } else {
    CL = TVCL * (CRCL/100) * exp(ETA_CL)
  }
  V = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let parsed = parse_full_model(content).expect("model should parse");
    let covs = &parsed.model.referenced_covariates;
    assert!(covs.contains(&"SEX".to_string()));
    assert!(covs.contains(&"WT".to_string()));
    assert!(covs.contains(&"CRCL".to_string()));
}

#[test]
fn test_pk_param_fn_runs_branch_at_runtime() {
    // End-to-end: parse a model with an if-block, run pk_param_fn at two
    // different covariate values, and confirm CL takes different values.
    let content = r#"
[parameters]
  theta TVCL(2.0, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)

  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04

  sigma PROP_ERR ~ 0.02

[individual_parameters]
  if (WT > 70) {
    CL = TVCL * 2.0
  } else {
    CL = TVCL
  }
  V = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let parsed = parse_full_model(content).unwrap();
    let theta = vec![2.0, 10.0];
    let eta = vec![0.0, 0.0];

    let mut covs_heavy = HashMap::new();
    covs_heavy.insert("WT".to_string(), 100.0);
    let p_heavy = (parsed.model.pk_param_fn)(&theta, &eta, &covs_heavy, 0.0);
    assert!((p_heavy.values[0] - 4.0).abs() < 1e-12, "WT=100 → CL=4");

    let mut covs_light = HashMap::new();
    covs_light.insert("WT".to_string(), 50.0);
    let p_light = (parsed.model.pk_param_fn)(&theta, &eta, &covs_light, 0.0);
    assert!((p_light.values[0] - 2.0).abs() < 1e-12, "WT=50 → CL=2");
}

#[test]
fn test_ode_block_supports_if_statements() {
    // ODE block with an if-statement that switches the elimination term
    // depending on whether central is above a threshold (Michaelis-Menten
    // approximation toggle).
    let content = r#"
[parameters]
  theta TVCL(2.0, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)

  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04

  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot) = 0
  if (central > 0) {
    d/dt(central) = -CL/V * central
  } else {
    d/dt(central) = 0
  }

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let parsed = parse_full_model(content).expect("ODE model should parse");
    let ode = parsed.model.ode_spec.as_ref().expect("ode_spec present");
    // States are [depot, central]; params are [CL, V] (declaration order in
    // [individual_parameters]). du must match n_states.
    let params = vec![2.0, 10.0];

    // central > 0 → if-branch fires, du[central] = -CL/V * central
    let u_pos = vec![0.0, 5.0];
    let mut du_pos = vec![0.0, 0.0];
    (ode.rhs)(&u_pos, &params, 0.0, &mut du_pos);
    assert!((du_pos[1] - (-2.0 / 10.0 * 5.0)).abs() < 1e-12);

    // central == 0 → else-branch fires, du[central] = 0. Pre-seed with junk
    // so the test would fail if the branch silently no-op'd instead of
    // assigning zero.
    let u_zero = vec![0.0, 0.0];
    let mut du_zero = vec![999.0, 999.0];
    (ode.rhs)(&u_zero, &params, 0.0, &mut du_zero);
    assert!(
        du_zero[1].abs() < 1e-12,
        "else-branch should emit 0, got {}",
        du_zero[1]
    );
}

#[test]
fn test_inline_if_in_parameter_assignment() {
    // Inline ternary form used directly as the RHS of a [individual_parameters]
    // assignment — this is the most concise form and should produce a
    // working pk_param_fn.
    let content = r#"
[parameters]
  theta TVCL(2.0, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)

  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04

  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = if (SEX == 1) TVCL * 1.5 else TVCL
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let parsed = parse_full_model(content).unwrap();
    let theta = vec![2.0, 10.0];
    let eta = vec![0.0, 0.0];

    let mut male = HashMap::new();
    male.insert("SEX".to_string(), 1.0);
    let p_male = (parsed.model.pk_param_fn)(&theta, &eta, &male, 0.0);
    assert!((p_male.values[0] - 3.0).abs() < 1e-12);

    let mut female = HashMap::new();
    female.insert("SEX".to_string(), 0.0);
    let p_female = (parsed.model.pk_param_fn)(&theta, &eta, &female, 0.0);
    assert!((p_female.values[0] - 2.0).abs() < 1e-12);
}

#[test]
fn test_missing_else_branch_in_inline_if_errors() {
    let ctx = empty_ctx();
    let res = parse_block_statements("CL = if (1 > 0) 5", ctx, StatementMode::Plain);
    assert!(res.is_err(), "inline if without else must error");
}

#[test]
fn test_missing_brace_in_if_block_errors() {
    let ctx = empty_ctx();
    let res = parse_block_statements("if (1 > 0) CL = 5", ctx, StatementMode::Plain);
    assert!(res.is_err(), "block-form if without `{{` must error");
}

#[test]
fn test_assigned_vars_in_order_includes_nested() {
    // Vars assigned inside if-bodies still count toward the ordered list
    // (so they're reachable as Variables in later expressions and end up
    // in the per-tv tables).
    let block = "
A = 1
if (1 > 0) {
  B = 2
  C = A + B
} else {
  D = 99
}
E = C + 1
";
    let ctx = empty_ctx();
    let stmts = parse_block_statements(block, ctx, StatementMode::Plain).unwrap();
    let names = assigned_vars_in_order(&stmts);
    assert_eq!(names, vec!["A", "B", "C", "D", "E"]);
}

// ── Regression tests for review-identified bugs ──────────────────────────

#[test]
fn test_parse_cond_atom_nested_parens_around_simple_compare() {
    // ((SEX == 1)) — double-parens around a comparison must parse correctly.
    // Previously the lookahead heuristic (only enter sub-condition path when
    // || or && is present) caused this to fail with "Missing closing )".
    let ctx = empty_ctx();
    let block = "X = if ((SEX == 1)) 1.0 else 2.0";
    assert!(
        parse_block_statements(block, ctx, StatementMode::Plain).is_ok(),
        "double-paren condition should parse"
    );
}

#[test]
fn test_parse_cond_atom_nested_parens_around_negation() {
    // (!(SEX == 1)) — parens around a negation must parse.
    let ctx = empty_ctx();
    let block = "X = if (!(SEX == 1)) 1.0 else 2.0";
    assert!(
        parse_block_statements(block, ctx, StatementMode::Plain).is_ok(),
        "parenthesised negation should parse"
    );
}

#[test]
fn test_top_level_assigned_vars_excludes_branch_locals() {
    // top_level_assigned_vars must NOT include variables assigned only
    // inside if-branches — those would corrupt the AD PK slot layout.
    let block = "
CL = 1.0
if (1 > 0) {
  SCALE = 2.0
  V = SCALE * 3.0
} else {
  V = 4.0
}
";
    let ctx = empty_ctx();
    let stmts = parse_block_statements(block, ctx, StatementMode::Plain).unwrap();
    let top = top_level_assigned_vars(&stmts);
    // SCALE is only assigned inside the if-body — must not appear.
    assert_eq!(top, vec!["CL"], "top-level vars should only contain CL");
    // But all_assigned_vars still sees SCALE and V.
    let all = assigned_vars_in_order(&stmts);
    assert!(all.contains(&"SCALE".to_string()));
    assert!(all.contains(&"V".to_string()));
}

#[test]
fn test_unconditionally_assigned_vars_includes_all_branch_assignments() {
    // V is assigned on BOTH branches → unconditionally defined → promoted.
    // SCALE is assigned in only one branch → branch-local → excluded.
    // (Issue #357: a param written on every branch must earn a PK slot.)
    let block = "
CL = 1.0
if (1 > 0) {
  SCALE = 2.0
  V = SCALE * 3.0
} else {
  V = 4.0
}
";
    let ctx = empty_ctx();
    let stmts = parse_block_statements(block, ctx, StatementMode::Plain).unwrap();
    let uncond = unconditionally_assigned_vars(&stmts);
    assert_eq!(
        uncond,
        vec!["CL", "V"],
        "V is assigned on every branch and must be promoted; SCALE must not"
    );
    // top_level still excludes V (its only assignments are inside branches).
    let top = top_level_assigned_vars(&stmts);
    assert_eq!(top, vec!["CL"]);
}

#[test]
fn test_unconditionally_assigned_vars_branch_only_param() {
    // The exact issue #357 shape: CL assigned only inside if/else, V at
    // top level. CL must come first (leading-branch order), then V.
    let block = "
if (WT > 70) {
  CL = TVCL * 1.5
} else {
  CL = TVCL
}
V = TVV
";
    let ctx = empty_ctx();
    let stmts = parse_block_statements(block, ctx, StatementMode::Plain).unwrap();
    let uncond = unconditionally_assigned_vars(&stmts);
    assert_eq!(uncond, vec!["CL", "V"]);
}

#[test]
fn test_unconditionally_assigned_vars_if_without_else_excludes() {
    // No `else` → the name could be undefined on the fall-through path, so
    // it stays branch-local (not promoted).
    let block = "
CL = 1.0
if (WT > 70) {
  V = 2.0
}
";
    let ctx = empty_ctx();
    let stmts = parse_block_statements(block, ctx, StatementMode::Plain).unwrap();
    let uncond = unconditionally_assigned_vars(&stmts);
    assert_eq!(
        uncond,
        vec!["CL"],
        "V lacks an else branch — not unconditional"
    );
}

#[test]
fn test_duplicate_diffeq_in_same_scope_errors() {
    // Two d/dt(central) at top level must be rejected.
    let block_text = "d/dt(central) = -0.1 * central\nd/dt(central) = -0.2 * central";
    let state_names = vec!["central".to_string()];
    let ctx = ParseCtx::ode(&state_names);
    let stmts = parse_block_statements(block_text, ctx, StatementMode::Ode).unwrap();
    let result = (|| -> Result<(), String> {
        fn check(stmts: &[Statement]) -> Result<(), String> {
            let mut seen = std::collections::HashSet::new();
            for s in stmts {
                match s {
                    Statement::DiffEq(name, _) => {
                        if !seen.insert(name.clone()) {
                            return Err(format!("duplicate d/dt({})", name));
                        }
                    }
                    Statement::If {
                        branches,
                        else_body,
                    } => {
                        for (_, b) in branches {
                            check(b)?;
                        }
                        if let Some(eb) = else_body {
                            check(eb)?;
                        }
                    }
                    Statement::Assign(_, _)
                    | Statement::AssignIdx(_, _)
                    | Statement::DiffEqIdx(_, _)
                    | Statement::AssignBc(_, _)
                    | Statement::DiffEqBc(_, _) => {}
                }
            }
            Ok(())
        }
        check(&stmts)
    })();
    assert!(result.is_err(), "duplicate d/dt in same scope must error");
}

#[test]
fn test_duplicate_diffeq_in_different_branches_allowed() {
    // d/dt(central) in if-branch AND else-branch is legitimate.
    // build_ode_spec must accept it.
    let ode_lines: Vec<String> = vec![
        "if (1 > 0) {".into(),
        "  d/dt(central) = -0.1 * central".into(),
        "} else {".into(),
        "  d/dt(central) = -0.2 * central".into(),
        "}".into(),
    ];
    let state_names = vec!["central".to_string()];
    let result = build_ode_spec(&ode_lines, &state_names, Some("central"), &[], &[], &[]);
    assert!(
        result.is_ok(),
        "same state in different branches must be allowed"
    );
}

#[test]
fn test_ode_rhs_undefined_name_errors() {
    // #314: a name in an ODE RHS that is not a state, individual parameter,
    // intermediate, or reserved time var must error — it would otherwise
    // resolve to the `usize::MAX` sentinel and silently read 0.0, producing
    // a structurally-broken fit. Here `central` is a state and `CL` is an
    // individual parameter, but `V` is undeclared.
    let ode_lines: Vec<String> = vec!["d/dt(central) = -(CL/V) * central".into()];
    let state_names = vec!["central".to_string()];
    let result = build_ode_spec(
        &ode_lines,
        &state_names,
        Some("central"),
        &["CL".to_string()],
        &[0],
        &[],
    );
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("undefined RHS name `V` must error"),
    };
    assert!(
        err.contains("undefined name(s): V."),
        "error should name the undefined `V`, got: {err}"
    );
    // The defined names are listed to make the fix obvious.
    assert!(
        err.contains("CL") && err.contains("central"),
        "error should list the defined names, got: {err}"
    );
}

#[test]
fn test_ode_rhs_undefined_name_walks_all_nodes() {
    // Exercises every arm of the undefined-name walker: an intermediate
    // assignment RHS, a block `if` condition, `&&`/`||`/`!` boolean
    // operators, an inline conditional, `exp(...)`, and `^`. Every BAD*
    // name is undefined; `k` (intermediate), `central` (state), `CL`
    // (param), and `TIME` (reserved) must NOT be flagged.
    let ode_lines: Vec<String> = vec![
            "k = CL / V1".into(),
            "if (BADIF > 0) {".into(),
            "  d/dt(central) = exp(BADEXP) + BADPOW^2 - k * central + (if (BADAND1 > 0 && BADAND2 > 0) 1.0 else 0.0) + (if (BADOR1 > 0 || BADOR2 > 0) 1.0 else 0.0) + (if (!(BADNOT > 0)) 1.0 else 0.0)".into(),
            "} else {".into(),
            "  d/dt(central) = -k * central".into(),
            "}".into(),
        ];
    let state_names = vec!["central".to_string()];
    let result = build_ode_spec(
        &ode_lines,
        &state_names,
        Some("central"),
        &["CL".to_string()],
        &[0],
        &[],
    );
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("undefined names in nested expressions must error"),
    };
    for bad in [
        "V1", "BADIF", "BADEXP", "BADPOW", "BADAND1", "BADAND2", "BADOR1", "BADOR2", "BADNOT",
    ] {
        assert!(err.contains(bad), "error should name `{bad}`, got: {err}");
    }
}

#[test]
fn test_ode_rhs_defined_names_ok() {
    // Regression guard against false positives: an intermediate (`k`),
    // individual parameters (`CL`, `V`), a state (`central`), and the
    // reserved `TIME` variable must all resolve and parse cleanly.
    let ode_lines: Vec<String> = vec![
        "k = CL / V".into(),
        "d/dt(central) = if (TIME < 24.0) -k * central else 0.0".into(),
    ];
    let state_names = vec!["central".to_string()];
    let result = build_ode_spec(
        &ode_lines,
        &state_names,
        Some("central"),
        &["CL".to_string(), "V".to_string()],
        &[0, 1],
        &[],
    );
    assert!(
        result.is_ok(),
        "intermediate + params + TIME must parse, got: {:?}",
        result.err()
    );
}

#[test]
fn test_ode_rhs_macheps_resolves_to_epsilon() {
    // MACHEPS is a reserved ODE builtin (machine epsilon): it must resolve
    // to f64::EPSILON in an ODE RHS — not be flagged as undefined, and not
    // silently read 0.0.
    let ode_lines: Vec<String> = vec!["d/dt(central) = MACHEPS".into()];
    let state_names = vec!["central".to_string()];
    let spec = match build_ode_spec(&ode_lines, &state_names, Some("central"), &[], &[], &[]) {
        Ok(s) => s,
        Err(e) => panic!("MACHEPS in an ODE RHS must parse, got: {e}"),
    };
    let params = vec![0.0; crate::types::MAX_PK_PARAMS + 2];
    let mut du = vec![0.0_f64];
    (spec.rhs)(&[0.0], &params, 0.0, &mut du);
    assert_eq!(
        du[0],
        f64::EPSILON,
        "d/dt(central) = MACHEPS must evaluate to machine epsilon"
    );
}

#[test]
fn test_ode_reserved_builtin_name_collision_errors() {
    // A state / individual parameter / intermediate may not reuse a reserved
    // builtin name — `MACHEPS` is now reserved alongside TIME/TAFD/TAD.
    let ode_lines: Vec<String> = vec!["d/dt(MACHEPS) = -MACHEPS".into()];
    let state_names = vec!["MACHEPS".to_string()];
    let result = build_ode_spec(&ode_lines, &state_names, Some("MACHEPS"), &[], &[], &[]);
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("a state named MACHEPS must collide with the reserved builtin"),
    };
    assert!(
        err.contains("MACHEPS") && err.contains("reserved"),
        "expected a reserved-name collision error, got: {err}"
    );
}

#[test]
fn test_mu_ref_warning_for_conditional_param() {
    // A model where CL is assigned only inside an if-block should emit a
    // parse_warning about mu-referencing being disabled.
    let model_str = "
[parameters]
  theta TVCL(1.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  if (1 > 0) {
    CL = TVCL * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }

[structural_model]
  pk one_cpt_oral(cl=CL, v=1.0, ka=1.0)

[error_model]
  DV ~ proportional(EPS)
";
    let parsed = super::parse_full_model(model_str).unwrap();
    assert!(
        !parsed.model.parse_warnings.is_empty(),
        "expected a parse warning about mu-ref disabled; got: {:?}",
        parsed.model.parse_warnings
    );
    let w = &parsed.model.parse_warnings[0];
    assert!(w.contains("CL"), "warning should mention CL");
    assert!(
        w.contains("Mu-referencing disabled"),
        "warning should mention mu-referencing"
    );
}

#[test]
fn nested_if_eta_param_sets_conditional_flag() {
    // Regression for #278/#280: an eta-bearing parameter assigned inside a
    // *nested* if-branch must still set `has_conditional_eta_params`, so the
    // inner loop routes to FD instead of the analytical AD kernel (which
    // cannot represent the branch). Detection previously looked only one
    // level deep and silently missed nested conditionals, leaving the model
    // on a wrong AD gradient — the exact failure class this gate prevents.
    let plain = minimal_model_with_indiv(
        "  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)",
    );
    assert!(
        !plain.has_conditional_eta_params,
        "an unconditional model must not be flagged conditional"
    );

    let nested = minimal_model_with_indiv(
        "  CL = TVCL
  V  = TVV * exp(ETA_V)
  if (1 > 0) {
    if (1 > 0) {
      CL = TVCL * exp(ETA_CL)
    } else {
      CL = TVCL * exp(ETA_CL)
    }
  }",
    );
    assert!(
        nested.has_conditional_eta_params,
        "eta-bearing param assigned only inside a nested if must set \
             has_conditional_eta_params"
    );
}

#[test]
fn test_indiv_param_names_populated_in_declaration_order() {
    // CompiledModel.indiv_param_names must hold every top-level
    // [individual_parameters] assignment, in source-declaration order.
    // Downstream consumers (the R FFI's per-subject EBE table) rely on
    // this list to label the columns of `individual_estimates`, and on
    // its alignment with `pk_indices` to read each value out of the
    // PkParams slot.
    let model_str = "
[parameters]
  theta TVCL(1.0)
  theta TVV(10.0)
  theta TVKA(2.0)
  omega ETA_CL ~ 0.1
  omega ETA_V  ~ 0.1
  omega ETA_KA ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(EPS)
";
    let parsed = super::parse_full_model(model_str).unwrap();
    assert_eq!(
        parsed.model.indiv_param_names,
        vec!["CL".to_string(), "V".to_string(), "KA".to_string()]
    );
    // The list must be parallel to pk_indices so the FFI can route each
    // name to its PkParams slot for analytical models.
    assert_eq!(
        parsed.model.indiv_param_names.len(),
        parsed.model.pk_indices.len()
    );
}

fn minimal_model_with_indiv(indiv_block: &str) -> crate::types::CompiledModel {
    let model_str = format!(
        r"
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.1
  omega ETA_V  ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
{}

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
",
        indiv_block
    );
    super::parse_full_model(&model_str).unwrap().model
}

fn minimal_logit_model() -> crate::types::CompiledModel {
    let model_str = r"
[parameters]
  theta THETA_F(0.0, -10.0, 10.0)
  sigma EPS ~ 0.01
  omega ETA_F  ~ 0.1

[individual_parameters]
  F = inv_logit(THETA_F + ETA_F)

[structural_model]
  pk one_cpt_iv(cl=1, v=1)

[error_model]
  DV ~ proportional(EPS)
";
    super::parse_full_model(model_str).unwrap().model
}

#[test]
fn test_inv_logit_evaluates() {
    let model = minimal_logit_model();
    let tv = model.tv_fn.as_ref().unwrap()(&[0.0], &Default::default());
    let expected = 0.5_f64; // inv_logit(0) = 0.5
    assert!(
        (tv[0] - expected).abs() < 1e-10,
        "inv_logit(0) should be 0.5, got {}",
        tv[0]
    );
}

#[test]
fn test_classify_lognormal_multiplicative() {
    use crate::types::{EtaParamType, ThetaTransform};
    // CL = TVCL * exp(ETA_CL)
    let model = minimal_model_with_indiv("  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)");
    assert_eq!(model.eta_param_info.len(), 2);
    let cl_info = model
        .eta_param_info
        .iter()
        .find(|i| i.eta_name == "ETA_CL")
        .unwrap();
    assert_eq!(cl_info.param_type, EtaParamType::LogNormal);
    // Anchor is the theta TVCL — linked_theta is now populated.
    assert_eq!(cl_info.linked_theta, Some("TVCL".to_string()));
    // theta_transform for TVCL (theta index 0) stays Identity — the
    // product pattern does not imply the theta is on the log scale.
    assert_eq!(model.theta_transform[0], ThetaTransform::Identity);
}

/// `TVCL * exp(ETA_CL + KAPPA_CL)` — IIV and IOV on the same parameter.
/// ETA_CL (BSV) should be LogNormal with linked_theta TVCL.
/// Also verifies that detect_mu_refs sets mu_ref for ETA_CL → TVCL.
#[test]
fn test_classify_iov_combined() {
    use crate::types::EtaParamType;
    let src = r"
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma EPS ~ 0.01
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(EPS)
[fit_options]
  iov_column = OCC
";
    let model = super::parse_full_model(src).unwrap().model;
    let cl_info = model
        .eta_param_info
        .iter()
        .find(|i| i.eta_name == "ETA_CL")
        .expect("ETA_CL must be classified");
    assert_eq!(cl_info.param_type, EtaParamType::LogNormal);
    assert_eq!(cl_info.linked_theta, Some("TVCL".to_string()));
    // mu_ref must be detected so SAEM can initialise ETA_CL at ln(TVCL).
    let mu = model.mu_refs.get("ETA_CL").expect("mu_ref for ETA_CL");
    assert_eq!(mu.theta_name, "TVCL");
    assert!(mu.log_transformed);
}

/// `KTR = 4.0 / TVMTT; KA = KTR * exp(ETA_KA)` — the base is a derived
/// intermediate, not a raw theta. ETA_KA should still be LogNormal (CV%
/// can be computed from the omega). linked_theta is None because no direct
/// theta anchor is visible in the KA expression.
#[test]
fn test_classify_lognormal_derived_base() {
    use crate::types::EtaParamType;
    // Reuse minimal_model_with_indiv which has TVCL and TVV as thetas.
    let model = minimal_model_with_indiv(
        "  KTR = 4.0 / TVCL\n  V = TVV * exp(ETA_V)\n  CL = KTR * exp(ETA_CL)",
    );
    let cl_info = model
        .eta_param_info
        .iter()
        .find(|i| i.eta_name == "ETA_CL")
        .expect("ETA_CL must be classified");
    assert_eq!(
        cl_info.param_type,
        EtaParamType::LogNormal,
        "derived base should still yield LogNormal, not Custom"
    );
    // No direct theta is visible in `KTR * exp(ETA_CL)`, so linked_theta is None.
    assert!(cl_info.linked_theta.is_none());
}

/// Confirms that the covariate-product form `TVCL * (WT/70)^0.75 * exp(ETA_CL)`
/// (ferx-r #54) is already classified as LogNormal with linked_theta = TVCL.
#[test]
fn test_classify_covariate_product() {
    use crate::types::EtaParamType;
    let src = r"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma EPS ~ 0.01
[individual_parameters]
  CL = TVCL * (WT / 70)^0.75 * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(EPS)
";
    let model = super::parse_full_model(src).unwrap().model;
    let cl_info = model
        .eta_param_info
        .iter()
        .find(|i| i.eta_name == "ETA_CL")
        .expect("ETA_CL must be classified");
    assert_eq!(cl_info.param_type, EtaParamType::LogNormal);
    assert_eq!(cl_info.linked_theta, Some("TVCL".to_string()));
}

#[test]
fn test_classify_lognormal_log_scale() {
    use crate::types::{EtaParamType, ThetaTransform};
    // CL = exp(TVCL + ETA_CL)  — theta is on log scale
    let model = minimal_model_with_indiv("  CL = exp(TVCL + ETA_CL)\n  V = TVV * exp(ETA_V)");
    let cl_info = model
        .eta_param_info
        .iter()
        .find(|i| i.eta_name == "ETA_CL")
        .unwrap();
    assert_eq!(cl_info.param_type, EtaParamType::LogNormal);
    assert_eq!(cl_info.linked_theta, Some("TVCL".to_string()));
    assert_eq!(model.theta_transform[0], ThetaTransform::Log);
}

#[test]
fn test_classify_additive() {
    use crate::types::{EtaParamType, ThetaTransform};
    // CL = TVCL + ETA_CL  — additive
    let model = minimal_model_with_indiv("  CL = TVCL + ETA_CL\n  V = TVV * exp(ETA_V)");
    let cl_info = model
        .eta_param_info
        .iter()
        .find(|i| i.eta_name == "ETA_CL")
        .unwrap();
    assert_eq!(cl_info.param_type, EtaParamType::Additive);
    assert_eq!(model.theta_transform[0], ThetaTransform::Identity);
}

#[test]
fn test_classify_logit_scale() {
    // inv_logit(THETA_F + ETA_F) — THETA_F on logit scale
    use crate::types::{EtaParamType, ThetaTransform};
    let model = minimal_logit_model();
    let f_info = model
        .eta_param_info
        .iter()
        .find(|i| i.eta_name == "ETA_F")
        .unwrap();
    assert_eq!(f_info.param_type, EtaParamType::Logit);
    assert_eq!(f_info.linked_theta, Some("THETA_F".to_string()));
    assert_eq!(model.theta_transform[0], ThetaTransform::Logit);
}

#[test]
fn test_classify_logit_probability_scale() {
    // inv_logit(logit(THETA_F) + ETA_F) — THETA_F on probability scale (0,1)
    use crate::types::{EtaParamType, ThetaTransform};
    let model_str = r"
[parameters]
  theta THETA_F(0.70, 0.001, 0.999)
  sigma EPS ~ 0.01
  omega ETA_F  ~ 0.1

[individual_parameters]
  F = inv_logit(logit(THETA_F) + ETA_F)

[structural_model]
  pk one_cpt_iv(cl=1, v=1)

[error_model]
  DV ~ proportional(EPS)
";
    let model = super::parse_full_model(model_str).unwrap().model;
    let f_info = model
        .eta_param_info
        .iter()
        .find(|i| i.eta_name == "ETA_F")
        .unwrap();
    assert_eq!(f_info.param_type, EtaParamType::LogitProbability);
    assert_eq!(f_info.linked_theta, Some("THETA_F".to_string()));
    assert_eq!(model.theta_transform[0], ThetaTransform::LogitProbability);
}

#[test]
fn test_inv_logit_logit_theta_evaluates() {
    // inv_logit(logit(0.70) + 0) should equal 0.70
    let model_str = r"
[parameters]
  theta THETA_F(0.70, 0.001, 0.999)
  sigma EPS ~ 0.01
  omega ETA_F  ~ 0.1

[individual_parameters]
  F = inv_logit(logit(THETA_F) + ETA_F)

[structural_model]
  pk one_cpt_iv(cl=1, v=1)

[error_model]
  DV ~ proportional(EPS)
";
    let model = super::parse_full_model(model_str).unwrap().model;
    let tv = model.tv_fn.as_ref().unwrap()(&[0.70], &Default::default());
    assert!(
        (tv[0] - 0.70).abs() < 1e-10,
        "inv_logit(logit(0.70)) should be 0.70, got {}",
        tv[0]
    );
}

#[test]
fn test_sigma_types_proportional() {
    use crate::types::SigmaType;
    let model = minimal_logit_model();
    // sigma_types is on FitResult, not CompiledModel — verify via ErrorModel::sigma_types().
    assert_eq!(
        model.error_model.sigma_types(),
        vec![SigmaType::Proportional]
    );
}

// ── Per-CMT (multi-endpoint) error models (issue #14) ───────────────────

/// Minimal 2-endpoint ODE PK/PD model: PK readout at CMT=1 (proportional),
/// PD readout at CMT=2 (additive). `error_block` overrides the
/// `[error_model]` body so negative tests can vary just that block.
/// Parse, expecting failure; returns the error string (ParsedModel isn't Debug).
fn expect_parse_err(model_str: &str) -> String {
    match parse_full_model(model_str) {
        Ok(_) => panic!("expected parse error, got Ok"),
        Err(e) => e,
    }
}

fn pkpd_model_str(error_block: &str) -> String {
    format!(
        r"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR_PK ~ 0.10 (sd)
  sigma ADD_ERR_PD  ~ 1.00 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(states=[central, effect])

[odes]
  d/dt(central) = -CL/V * central
  d/dt(effect)  =  central/V - effect

[scaling]
  y[CMT=1] = central / V
  y[CMT=2] = effect

[error_model]
{}
",
        error_block
    )
}

#[test]
fn test_per_cmt_error_model_parses() {
    let model = parse_full_model(&pkpd_model_str(
        "  CMT=1: DV ~ proportional(PROP_ERR_PK)\n  CMT=2: DV ~ additive(ADD_ERR_PD)",
    ))
    .unwrap()
    .model;

    match &model.error_spec {
        ErrorSpec::PerCmt(map) => {
            assert_eq!(map.len(), 2);
            // CMT=1 → proportional on the first declared sigma (idx 0).
            let pk = map.get(&1).expect("CMT=1 endpoint present");
            assert_eq!(pk.error_model, ErrorModel::Proportional);
            assert_eq!(pk.sigma_idx, vec![0]);
            // CMT=2 → additive on the second declared sigma (idx 1).
            let pd = map.get(&2).expect("CMT=2 endpoint present");
            assert_eq!(pd.error_model, ErrorModel::Additive);
            assert_eq!(pd.sigma_idx, vec![1]);
        }
        other => panic!("expected PerCmt, got {:?}", other),
    }
}

#[test]
fn test_single_error_model_stays_single() {
    // A plain (unprefixed) line must still yield ErrorSpec::Single.
    let model = parse_full_model(&pkpd_model_str("  DV ~ proportional(PROP_ERR_PK)"))
        .unwrap()
        .model;
    assert!(matches!(
        model.error_spec,
        ErrorSpec::Single(ErrorModel::Proportional)
    ));
}

// ── Covariate-selected error models (`if/else`, issue #658) ─────────────

/// Minimal **analytical** 1-cpt model whose `[error_model]` is overridden by
/// `error_block`. Two proportional sigmas + a `FREE` covariate let a
/// free-vs-total split-error `if/else` be expressed on an analytical PK model.
fn selected_err_model_str(error_block: &str) -> String {
    format!(
        r"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_TOTAL   ~ 0.10 (sd)
  sigma PROP_UNBOUND ~ 0.30 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
{}

[covariates]
  FREE continuous
",
        error_block
    )
}

#[test]
fn test_selected_error_model_parses() {
    let model = parse_full_model(&selected_err_model_str(
            "  if (FREE == 0) {\n    DV ~ proportional(PROP_TOTAL)\n  } else {\n    DV ~ proportional(PROP_UNBOUND)\n  }",
        ))
        .unwrap()
        .model;

    match &model.error_spec {
        ErrorSpec::Selected {
            selector,
            endpoints,
        } => {
            assert_eq!(endpoints.len(), 2);
            // Branch 0 (FREE == 0) → PROP_TOTAL (sigma idx 0).
            let total = endpoints.get(&0).expect("branch 0 endpoint");
            assert_eq!(total.error_model, ErrorModel::Proportional);
            assert_eq!(total.sigma_idx, vec![0]);
            // Else branch (key 1) → PROP_UNBOUND (sigma idx 1).
            let unbound = endpoints.get(&1).expect("else endpoint");
            assert_eq!(unbound.error_model, ErrorModel::Proportional);
            assert_eq!(unbound.sigma_idx, vec![1]);
            // Selector resolves FREE=0 → branch 0, FREE=1 → else (key 1).
            let free0: std::collections::HashMap<String, f64> =
                [("FREE".to_string(), 0.0)].into_iter().collect();
            let free1: std::collections::HashMap<String, f64> =
                [("FREE".to_string(), 1.0)].into_iter().collect();
            assert_eq!((selector.eval)(&free0), 0);
            assert_eq!((selector.eval)(&free1), 1);
        }
        other => panic!("expected Selected, got {:?}", other),
    }

    // The selector covariate becomes a required data column.
    assert!(model.referenced_covariates.iter().any(|c| c == "FREE"));
}

#[test]
fn test_selected_error_model_else_if_chain() {
    // Three-way split: FREE==0, FREE==1, else — keys 0, 1, 2 (else).
    let model = parse_full_model(&selected_err_model_str(
            "  if (FREE == 0) {\n    DV ~ proportional(PROP_TOTAL)\n  } else if (FREE == 1) {\n    DV ~ proportional(PROP_UNBOUND)\n  } else {\n    DV ~ additive(PROP_TOTAL)\n  }",
        ))
        .unwrap()
        .model;
    match &model.error_spec {
        ErrorSpec::Selected {
            selector,
            endpoints,
        } => {
            assert_eq!(endpoints.len(), 3);
            let m = |v: f64| -> usize {
                let cov: std::collections::HashMap<String, f64> =
                    [("FREE".to_string(), v)].into_iter().collect();
                (selector.eval)(&cov)
            };
            assert_eq!(m(0.0), 0);
            assert_eq!(m(1.0), 1);
            assert_eq!(m(2.0), 2); // no branch matches → else key
        }
        other => panic!("expected Selected, got {:?}", other),
    }
}

#[test]
fn test_selected_error_model_requires_else() {
    let err = expect_parse_err(&selected_err_model_str(
        "  if (FREE == 0) {\n    DV ~ proportional(PROP_TOTAL)\n  }",
    ));
    assert!(
        err.contains("must end with a final"),
        "expected a missing-else error, got: {err}"
    );
}

#[test]
fn test_selected_error_model_rejects_ltbs_branch() {
    let err = expect_parse_err(&selected_err_model_str(
            "  if (FREE == 0) {\n    log(DV) ~ additive(PROP_TOTAL)\n  } else {\n    DV ~ proportional(PROP_UNBOUND)\n  }",
        ));
    assert!(
        err.contains("log-transform-both-sides"),
        "expected an LTBS-rejected error, got: {err}"
    );
}

#[test]
fn test_selected_error_unknown_sigma_rejected() {
    let err = expect_parse_err(&selected_err_model_str(
            "  if (FREE == 0) {\n    DV ~ proportional(NOPE)\n  } else {\n    DV ~ proportional(PROP_UNBOUND)\n  }",
        ));
    assert!(
        err.contains("unknown sigma"),
        "expected an unknown-sigma error, got: {err}"
    );
}

/// Shared free/total covariate-selected model with the two branch sigmas in
/// one `block_sigma` (#669). Used by the accept-parse and the cross-branch
/// R-matrix tests so the two can't drift.
const SELECTED_BLOCK_SIGMA_MODEL: &str = r"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  block_sigma (PROP_TOTAL, PROP_UNBOUND) = [0.01, 0.005, 0.09]

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  if (FREE == 0) {
    DV ~ proportional(PROP_TOTAL)
  } else {
    DV ~ proportional(PROP_UNBOUND)
  }

[covariates]
  FREE continuous
";

#[test]
fn test_selected_error_model_accepts_block_sigma() {
    // #669: block_sigma correlated residuals now parse together with a
    // covariate-selected [error_model]. The off-diagonal couples the two
    // branches' sigmas, and `build_residual_correlations` records it.
    let model = parse_full_model(SELECTED_BLOCK_SIGMA_MODEL).unwrap().model;
    assert!(matches!(model.error_spec, ErrorSpec::Selected { .. }));
    // One off-diagonal correlation between the two branch sigmas (idx 0,1).
    assert_eq!(model.residual_correlations.len(), 1);
    let corr = model.residual_correlations[0];
    let pair = (
        corr.sigma_i.min(corr.sigma_j),
        corr.sigma_i.max(corr.sigma_j),
    );
    assert_eq!(pair, (0, 1));
    // rho = cov / (sd_i · sd_j) = 0.005 / sqrt(0.01·0.09).
    let expected_rho = 0.005 / (0.01f64 * 0.09).sqrt();
    assert!((corr.rho - expected_rho).abs() < 1e-12);
}

#[test]
fn test_selected_error_model_rejects_block_sigma_unused_sigma() {
    // The reference-validity check still runs for Selected: a block_sigma
    // off-diagonal touching a sigma no branch loads is rejected.
    let src = r"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  block_sigma (PROP_TOTAL, PROP_UNUSED) = [0.01, 0.005, 0.09]
  sigma PROP_UNBOUND ~ 0.30 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  if (FREE == 0) {
    DV ~ proportional(PROP_TOTAL)
  } else {
    DV ~ proportional(PROP_UNBOUND)
  }

[covariates]
  FREE continuous
";
    let err = expect_parse_err(src);
    assert!(
        err.contains("not used by [error_model]"),
        "expected an unused-sigma rejection, got: {err}"
    );
}

#[test]
fn test_selected_error_block_sigma_cross_branch_covariance() {
    // #669 end-to-end: two rows at the same subject time+occasion but
    // different `FREE` flags resolve to different branches (total vs.
    // unbound). The dense R off-diagonal must carry the block_sigma
    // cross-branch covariance rho·σ_i·σ_j scaled by each row's proportional
    // loading (f·σ_i)(f·σ_j) — identical to the total/unbound PerCmt case.
    let model = parse_full_model(SELECTED_BLOCK_SIGMA_MODEL).unwrap().model;

    let subject = crate::types::Subject {
        id: "S1".to_string(),
        doses: vec![crate::types::DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![1.0, 1.0],
        obs_raw_times: Vec::new(),
        observations: vec![10.0, 10.0],
        obs_cmts: vec![1, 1],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: vec![
            HashMap::from([("FREE".to_string(), 0.0)]),
            HashMap::from([("FREE".to_string(), 1.0)]),
        ],
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0, 0],
        occasions: vec![1, 1],
        obs_l2: Vec::new(),
        dose_occasions: vec![1],
        reset_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };

    // sigma = [SD_total, SD_unbound] from the block diagonal.
    let sd_i = 0.01f64.sqrt();
    let sd_j = 0.09f64.sqrt();
    let sigma = [sd_i, sd_j];
    let f = 10.0;
    let ipreds = [f, f];
    let keys = model.error_spec.obs_keys(&subject);

    let r = crate::stats::residual_error::compute_r_matrix_with_correlations(
        &model.error_spec,
        &ipreds,
        keys.as_ref(),
        &subject.obs_times,
        &subject.obs_raw_times,
        &subject.occasions,
        &subject.obs_l2,
        &sigma,
        &model.residual_correlations,
    );

    // Diagonal: each row's own proportional variance from its branch sigma.
    assert!((r[(0, 0)] - (f * sd_i).powi(2)).abs() < 1e-12);
    assert!((r[(1, 1)] - (f * sd_j).powi(2)).abs() < 1e-12);
    // Off-diagonal: cross-branch covariance (f·σ_i)(f·σ_j)·rho.
    let rho = 0.005 / (sd_i * sd_j);
    let expected_off = f * sd_i * f * sd_j * rho;
    assert!((r[(0, 1)] - expected_off).abs() < 1e-12);
    assert!((r[(1, 0)] - expected_off).abs() < 1e-12);
    // Sanity: symmetric and equals the raw covariance times f².
    assert!((r[(0, 1)] - f * f * 0.005).abs() < 1e-12);
}

/// End-to-end dispatch: `obs_keys` resolves each observation's covariate
/// snapshot to the right endpoint key, and `variance_at` then uses the
/// per-endpoint sigma. A total row (FREE=0) and an unbound row (FREE=1) at
/// the same prediction must get *different* proportional variances.
#[test]
fn test_selected_error_obs_keys_dispatch() {
    let model = parse_full_model(&selected_err_model_str(
            "  if (FREE == 0) {\n    DV ~ proportional(PROP_TOTAL)\n  } else {\n    DV ~ proportional(PROP_UNBOUND)\n  }",
        ))
        .unwrap()
        .model;

    // Two rows in the *same* CMT (=1) but different `FREE` flags.
    let subject = crate::types::Subject {
        id: "S1".to_string(),
        doses: vec![crate::types::DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![1.0, 1.0],
        obs_raw_times: Vec::new(),
        observations: vec![10.0, 10.0],
        obs_cmts: vec![1, 1],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: vec![
            HashMap::from([("FREE".to_string(), 0.0)]),
            HashMap::from([("FREE".to_string(), 1.0)]),
        ],
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0, 0],
        occasions: vec![1, 1],
        obs_l2: Vec::new(),
        dose_occasions: vec![1],
        reset_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };

    let keys = model.error_spec.obs_keys(&subject);
    assert_eq!(keys.as_ref(), &[0usize, 1usize]);
    assert_eq!(model.error_spec.obs_key(&subject, 0), 0);
    assert_eq!(model.error_spec.obs_key(&subject, 1), 1);

    // sigma = [SD_total, SD_unbound]. Proportional variance = (f·σ)².
    let sigma = [0.10, 0.30];
    let f = 10.0;
    let v_total = model.error_spec.variance_at(keys[0], f, &sigma);
    let v_unbound = model.error_spec.variance_at(keys[1], f, &sigma);
    assert!((v_total - (f * 0.10).powi(2)).abs() < 1e-9);
    assert!((v_unbound - (f * 0.30).powi(2)).abs() < 1e-9);
    assert!(v_unbound > v_total);
}

#[test]
fn test_selected_error_model_parses_on_ode_model() {
    // Acceptance: covariate selection works on ODE models too (unlike
    // per-CMT error, which is ODE-only for the opposite reason).
    let src = r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.04
  sigma PROP_TOTAL   ~ 0.05 (sd)
  sigma PROP_UNBOUND ~ 0.30 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = -CL/V * central

[scaling]
  y = central / V

[error_model]
  if (FREE == 0) {
    DV ~ proportional(PROP_TOTAL)
  } else {
    DV ~ proportional(PROP_UNBOUND)
  }

[covariates]
  FREE continuous
";
    let model = parse_full_model(src).unwrap().model;
    assert!(model.ode_spec.is_some(), "should be an ODE model");
    match &model.error_spec {
        ErrorSpec::Selected { endpoints, .. } => assert_eq!(endpoints.len(), 2),
        other => panic!("expected Selected on ODE model, got {:?}", other),
    }
    assert!(model.referenced_covariates.iter().any(|c| c == "FREE"));
}

// ── IIV on residual error (`iiv_on_ruv`, #409) ──────────────────────────
fn iiv_ruv_model_str(error_block: &str) -> String {
    format!(
        r"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_RUV ~ 0.05
  sigma PROP_ERR ~ 0.10 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
{}
",
        error_block
    )
}

#[test]
fn test_iiv_on_ruv_resolves_eta_index() {
    let model = parse_full_model(&iiv_ruv_model_str(
        "  DV ~ proportional(PROP_ERR)\n  iiv_on_ruv = ETA_RUV",
    ))
    .unwrap()
    .model;
    // ETA_RUV is the 2nd declared omega → eta index 1.
    assert_eq!(model.residual_error_eta, Some(1));
    // The residual-error eta is NOT a structural/individual-parameter eta.
    assert!(
        !model.eta_param_info.iter().any(|e| e.eta_name == "ETA_RUV"),
        "ETA_RUV must not carry an EtaParamInfo entry"
    );
    // The scale factor is exp(2·η) at that index, 1.0 elsewhere.
    assert!((model.residual_var_scale(&[0.0, 0.0]) - 1.0).abs() < 1e-12);
    assert!((model.residual_var_scale(&[0.3, 0.5]) - (2.0_f64 * 0.5).exp()).abs() < 1e-12);
}

#[test]
fn test_iiv_on_ruv_eta_not_flagged_unused() {
    // The residual-error eta is referenced from [error_model] (not any
    // individual-parameter expression), but it scales the residual variance and
    // is estimated — it must NOT trigger the "not referenced" unused warning.
    let model = parse_full_model(&iiv_ruv_model_str(
        "  DV ~ proportional(PROP_ERR)\n  iiv_on_ruv = ETA_RUV",
    ))
    .unwrap()
    .model;
    assert!(
        !model
            .parse_warnings
            .iter()
            .any(|w| w.contains("ETA_RUV") && w.contains("not referenced")),
        "iiv_on_ruv eta must not be flagged unused; got: {:?}",
        model.parse_warnings
    );
}

#[test]
fn test_iiv_on_ruv_absent_is_none() {
    let model = parse_full_model(&iiv_ruv_model_str("  DV ~ proportional(PROP_ERR)"))
        .unwrap()
        .model;
    assert_eq!(model.residual_error_eta, None);
    assert!((model.residual_var_scale(&[0.7, 0.9]) - 1.0).abs() < 1e-12);
}

#[test]
fn test_iiv_on_ruv_unknown_eta_rejected() {
    let err = expect_parse_err(&iiv_ruv_model_str(
        "  DV ~ proportional(PROP_ERR)\n  iiv_on_ruv = NOPE",
    ));
    assert!(
        err.contains("iiv_on_ruv") && err.contains("NOPE"),
        "got: {err}"
    );
}

#[test]
fn test_iiv_on_ruv_duplicate_rejected() {
    let err = expect_parse_err(&iiv_ruv_model_str(
        "  DV ~ proportional(PROP_ERR)\n  iiv_on_ruv = ETA_RUV\n  iiv_on_ruv = ETA_CL",
    ));
    assert!(err.contains("more than one"), "got: {err}");
}

#[test]
fn test_iiv_on_ruv_rejected_with_per_cmt() {
    let err = expect_parse_err(&pkpd_model_str(
            "  CMT=1: DV ~ proportional(PROP_ERR_PK)\n  CMT=2: DV ~ additive(ADD_ERR_PD)\n  iiv_on_ruv = ETA_CL",
        ));
    assert!(err.contains("per-CMT"), "got: {err}");
}

#[test]
fn test_per_cmt_error_unknown_sigma_rejected() {
    let err = expect_parse_err(&pkpd_model_str(
        "  CMT=1: DV ~ proportional(NOPE)\n  CMT=2: DV ~ additive(ADD_ERR_PD)",
    ));
    assert!(err.contains("unknown sigma"), "got: {err}");
}

#[test]
fn test_per_cmt_error_duplicate_cmt_rejected() {
    let err = expect_parse_err(&pkpd_model_str(
        "  CMT=1: DV ~ proportional(PROP_ERR_PK)\n  CMT=1: DV ~ additive(ADD_ERR_PD)",
    ));
    assert!(err.contains("CMT=1"), "got: {err}");
}

#[test]
fn test_per_cmt_error_mixed_styles_rejected() {
    let err = expect_parse_err(&pkpd_model_str(
        "  DV ~ proportional(PROP_ERR_PK)\n  CMT=2: DV ~ additive(ADD_ERR_PD)",
    ));
    assert!(err.contains("mixes"), "got: {err}");
}

/// A `combined` (two-sigma) endpoint resolves both sigma indices, and they
/// can interleave with other endpoints' sigmas in [parameters] order.
fn pkpd_3sigma_model_str(error_block: &str) -> String {
    format!(
        r"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma S_PROP ~ 0.10 (sd)
  sigma S_ADD  ~ 1.00 (sd)
  sigma S_PD   ~ 0.50 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(states=[central, effect])

[odes]
  d/dt(central) = -CL/V * central
  d/dt(effect)  =  central/V - effect

[scaling]
  y[CMT=1] = central / V
  y[CMT=2] = effect

[error_model]
{}
",
        error_block
    )
}

#[test]
fn test_per_cmt_combined_endpoint_resolves_two_sigma_indices() {
    let model = parse_full_model(&pkpd_3sigma_model_str(
        "  CMT=1: DV ~ combined(S_PROP, S_ADD)\n  CMT=2: DV ~ additive(S_PD)",
    ))
    .unwrap()
    .model;

    match &model.error_spec {
        ErrorSpec::PerCmt(map) => {
            let c1 = map.get(&1).expect("CMT=1 present");
            assert_eq!(c1.error_model, ErrorModel::Combined);
            assert_eq!(c1.sigma_idx, vec![0, 1]); // S_PROP, S_ADD
            let c2 = map.get(&2).expect("CMT=2 present");
            assert_eq!(c2.error_model, ErrorModel::Additive);
            assert_eq!(c2.sigma_idx, vec![2]); // S_PD
        }
        other => panic!("expected PerCmt, got {other:?}"),
    }
}

#[test]
fn test_per_cmt_endpoint_sigma_count_mismatch_rejected() {
    // `combined` needs two sigmas; giving one must error at parse, not
    // silently propagate NaN into the likelihood.
    let err = expect_parse_err(&pkpd_3sigma_model_str(
        "  CMT=1: DV ~ combined(S_PROP)\n  CMT=2: DV ~ additive(S_PD)",
    ));
    assert!(
        err.contains("expects 2 sigma") || err.contains("2 sigma(s)"),
        "got: {err}"
    );
}

#[test]
fn test_single_error_unknown_sigma_rejected() {
    // A single (unprefixed) error line that references a sigma not declared
    // in [parameters] is a typo, not a silent bind to sigma[0].
    let model_str = r"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.10 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(NO_SUCH_SIGMA)
";
    let err = expect_parse_err(model_str);
    assert!(err.contains("unknown sigma"), "got: {err}");
}

#[test]
fn test_per_cmt_error_on_analytical_model_rejected() {
    let model_str = r"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR_PK ~ 0.10 (sd)
  sigma ADD_ERR_PD  ~ 1.00 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  CMT=1: DV ~ proportional(PROP_ERR_PK)
  CMT=2: DV ~ additive(ADD_ERR_PD)
";
    let err = expect_parse_err(model_str);
    assert!(err.contains("ODE"), "got: {err}");
}

// ── Issue 3: if/else classification ─────────────────────────────────────

#[test]
fn test_classify_if_else_unanimous_lognormal() {
    // Both branches use TVCL * exp(ETA_CL) — should classify as LogNormal.
    use crate::types::EtaParamType;
    let model = minimal_model_with_indiv(
            "  if (TVCL > 1) {\n    CL = TVCL * exp(ETA_CL)\n  } else {\n    CL = TVCL * exp(ETA_CL)\n  }\n  V = TVV * exp(ETA_V)",
        );
    let cl_info = model
        .eta_param_info
        .iter()
        .find(|i| i.eta_name == "ETA_CL")
        .unwrap();
    assert_eq!(cl_info.param_type, EtaParamType::LogNormal);
    assert_eq!(cl_info.individual_param_name, "CL");
}

#[test]
fn test_classify_if_else_disagreement_custom() {
    // Branches use different patterns — should fall back to Custom.
    use crate::types::EtaParamType;
    let model = minimal_model_with_indiv(
            "  if (TVCL > 1) {\n    CL = TVCL * exp(ETA_CL)\n  } else {\n    CL = TVCL + ETA_CL\n  }\n  V = TVV * exp(ETA_V)",
        );
    let cl_info = model
        .eta_param_info
        .iter()
        .find(|i| i.eta_name == "ETA_CL")
        .unwrap();
    assert_eq!(cl_info.param_type, EtaParamType::Custom);
}

#[test]
fn test_classify_if_no_else_skipped() {
    // No else arm → partially defined → classification skipped entirely for V.
    let model = minimal_model_with_indiv(
        "  CL = TVCL * exp(ETA_CL)\n  if (TVV > 1) {\n    V = TVV * exp(ETA_V)\n  }",
    );
    // CL should be classified; V (if-only, no else) should be absent.
    assert!(model.eta_param_info.iter().any(|i| i.eta_name == "ETA_CL"));
    assert!(!model.eta_param_info.iter().any(|i| i.eta_name == "ETA_V"));
}

#[test]
fn test_classify_multi_eta_custom() {
    // Expression with two ETAs (unusual) — both get their own Custom entry.
    use crate::types::EtaParamType;
    let model = minimal_model_with_indiv("  CL = TVCL + ETA_CL + ETA_V\n  V = 10.0");
    let customs: Vec<_> = model
        .eta_param_info
        .iter()
        .filter(|i| i.param_type == EtaParamType::Custom)
        .collect();
    assert_eq!(
        customs.len(),
        2,
        "both ETAs in the expression should be Custom"
    );
}

#[test]
fn test_lagtime_in_structural_model_block() {
    // The DSL key `lagtime=LAGTIME` on the structural_model line must
    // route LAGTIME's value into PK_IDX_LAGTIME (8). Verifies the
    // parser → name_to_index → PkParams pipeline end-to-end.
    let model_str = "
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta TVKA(2.0)
  theta TVLAGTIME(0.5)
  omega ETA_CL ~ 0.1
  omega ETA_V  ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL      = TVCL * exp(ETA_CL)
  V       = TVV  * exp(ETA_V)
  KA      = TVKA
  LAGTIME = TVLAGTIME

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, lagtime=LAGTIME)

[error_model]
  DV ~ proportional(EPS)
";
    let parsed = super::parse_full_model(model_str).unwrap();
    let pk_indices = &parsed.model.pk_indices;
    // LAGTIME should map to slot 8.
    assert!(
        pk_indices.contains(&crate::types::PK_IDX_LAGTIME),
        "pk_indices missing PK_IDX_LAGTIME: {:?}",
        pk_indices
    );

    // Evaluate pk_param_fn with default theta to confirm the value
    // flows through to the slot.
    let theta: Vec<f64> = parsed.model.default_params.theta.clone();
    let eta: Vec<f64> = vec![0.0; parsed.model.n_eta];
    let pk = (parsed.model.pk_param_fn)(&theta, &eta, &std::collections::HashMap::new(), 0.0);
    assert_eq!(pk.lagtime(), 0.5);
}

#[test]
fn test_alag_alias_in_structural_model_block() {
    // For NONMEM-user familiarity, `alag=` is accepted as an alias
    // for `lagtime=`. Same target slot (PK_IDX_LAGTIME).
    let model_str = "
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta TVKA(2.0)
  theta TVALAG(0.75)
  omega ETA_CL ~ 0.1
  omega ETA_V  ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV  * exp(ETA_V)
  KA   = TVKA
  ALAG = TVALAG

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, alag=ALAG)

[error_model]
  DV ~ proportional(EPS)
";
    let parsed = super::parse_full_model(model_str).unwrap();
    assert!(parsed
        .model
        .pk_indices
        .contains(&crate::types::PK_IDX_LAGTIME));

    let theta: Vec<f64> = parsed.model.default_params.theta.clone();
    let eta: Vec<f64> = vec![0.0; parsed.model.n_eta];
    let pk = (parsed.model.pk_param_fn)(&theta, &eta, &std::collections::HashMap::new(), 0.0);
    assert_eq!(pk.lagtime(), 0.75);
}

#[test]
fn test_undefined_structural_param_errors() {
    // #261: a [structural_model] PK value that names a variable not defined
    // in [individual_parameters] must be a hard parse error, not silently
    // defaulted to 0.0 (which yields a "converged" but structurally broken
    // fit). Here `cl=CL` but only V, KE, KA are defined.
    let model_str = "
[parameters]
  theta TVV(10.0, 0.1, 1000.0)
  theta TVKE(0.1, 0.001, 10.0)
  theta TVKA(1.0, 0.01, 100.0)
  omega ETA_V ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  V  = TVV * exp(ETA_V)
  KE = TVKE
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(EPS)
";
    let err = expect_parse_err(model_str);
    assert!(
        err.contains("CL") && err.contains("not defined"),
        "expected an undefined-variable error naming CL, got: {err}"
    );
    // The message should list the defined parameters to make the fix obvious.
    assert!(
        err.contains("KE"),
        "error should list defined params: {err}"
    );
}

#[test]
fn test_unknown_pk_param_key_errors() {
    // Audit of the same silent-drop pattern on the key side: an unrecognized
    // PK-parameter name (`clx` here, a typo for `cl`) previously dropped the
    // binding, leaving cl at its 0.0 default. It must now error.
    let model_str = "
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(clx=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";
    let err = expect_parse_err(model_str);
    assert!(
        err.contains("clx") && err.contains("unknown PK parameter"),
        "expected an unknown-PK-parameter error naming clx, got: {err}"
    );
}

#[test]
fn test_literal_pk_param_binds_constant() {
    // A numeric literal value (`ka=2.5`) is bound as a constant rather than
    // silently dropped to 0.0. Companion to #261: the same filter_map that
    // dropped undefined references also dropped literals. `cl=CL` (a defined
    // variable) must still bind, confirming the two paths coexist.
    let model_str = "
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=2.5)

[error_model]
  DV ~ proportional(EPS)
";
    let parsed = super::parse_full_model(model_str).unwrap();
    let theta: Vec<f64> = parsed.model.default_params.theta.clone();
    let eta: Vec<f64> = vec![0.0; parsed.model.n_eta];
    let pk = (parsed.model.pk_param_fn)(&theta, &eta, &std::collections::HashMap::new(), 0.0);
    assert_eq!(pk.ka(), 2.5, "literal ka should bind as the constant 2.5");
    assert!(
        pk.cl() > 0.0,
        "cl should still bind to the defined variable, got {}",
        pk.cl()
    );
}

#[test]
fn test_valid_structural_param_still_parses() {
    // #261 acceptance: the repro fixed by defining CL (= KE * V) parses
    // without error and binds cl to a positive value.
    let model_str = "
[parameters]
  theta TVV(10.0, 0.1, 1000.0)
  theta TVKE(0.1, 0.001, 10.0)
  theta TVKA(1.0, 0.01, 100.0)
  omega ETA_V ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  V  = TVV * exp(ETA_V)
  KE = TVKE
  KA = TVKA
  CL = KE * V

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(EPS)
";
    let parsed = super::parse_full_model(model_str).expect("model with CL defined should parse");
    let theta: Vec<f64> = parsed.model.default_params.theta.clone();
    let eta: Vec<f64> = vec![0.0; parsed.model.n_eta];
    let pk = (parsed.model.pk_param_fn)(&theta, &eta, &std::collections::HashMap::new(), 0.0);
    assert!(pk.cl() > 0.0, "cl should bind to KE * V, got {}", pk.cl());
}

#[test]
fn test_non_finite_literal_pk_param_errors() {
    // `f64::from_str` accepts "inf"/"nan", so a non-finite literal value
    // must be rejected rather than silently bound as a degenerate constant
    // (the same silent-wrong default #261 removes for undefined references).
    let model_str = "
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=inf)

[error_model]
  DV ~ proportional(EPS)
";
    let err = expect_parse_err(model_str);
    assert!(
        err.contains("non-finite") && err.contains("ka"),
        "expected a non-finite-constant error naming ka, got: {err}"
    );
}

#[test]
fn test_undefined_optional_pk_param_errors() {
    // The undefined-reference guard (#261/#308) applies uniformly to the
    // *optional* params too — `f`, `lagtime`, and the `alag` alias — not just
    // the required ones. A typo'd / undefined optional reference must error,
    // never silently default the slot. (#309 keeps `f`/`lagtime` *optional*,
    // i.e. omitting them is fine; this is the orthogonal value-validation:
    // if you DO map them, the referenced variable must exist.) All required
    // params (cl/v/ka) are defined here so the model reaches value resolution.
    let header = "
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta TVKA(1.0, 0.01, 100.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, ";
    let footer = ")

[error_model]
  DV ~ proportional(EPS)
";
    // (pk key, undefined variable) — one per optional slot, incl. the alias.
    for (key, badvar) in [("f", "BADF"), ("lagtime", "BADLAG"), ("alag", "BADALAG")] {
        let model_str = format!("{header}{key}={badvar}{footer}");
        let err = expect_parse_err(&model_str);
        assert!(
            err.contains(badvar) && err.contains("not defined"),
            "optional `{key}={badvar}` must error as an undefined reference, got: {err}"
        );
    }
}

#[test]
fn test_unknown_key_precedes_missing_required() {
    // Precedence: when a key is unrecognized (a typo like `vx` for `v`) the
    // error must name that bad key (#308), not report the now-unmapped slot
    // as a missing required param (#309). The required-param check defers to
    // the unknown-key check so the message points at the actual mistake.
    let model_str = "
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta TVKA(1.0, 0.01, 100.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, vx=V, ka=KA)

[error_model]
  DV ~ proportional(EPS)
";
    let err = expect_parse_err(model_str);
    assert!(
        err.contains("vx") && err.contains("unknown PK parameter"),
        "unknown key `vx` must be reported, not a missing-required error, got: {err}"
    );
}

#[test]
fn test_lagtime_in_ode_model_routes_to_canonical_slot() {
    // Regression for the ODE-with-lagtime path. For ODE models there is
    // no [structural_model] pk= line, so pk_param_map is empty and
    // pk_param_fn's ODE branch writes individual parameters by
    // declaration order. LAGTIME (and ALAG) must also land at the
    // canonical PK_IDX_LAGTIME slot so `ode_predictions` (which reads
    // `pk_params_flat[PK_IDX_LAGTIME]`) sees it. `has_lagtime()` must
    // likewise return true via the indiv_param_names fallback so the
    // SS/negative-lagtime warning gating fires for ODE users.
    let model_str = "
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta TVLAGTIME(0.5)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL      = TVCL * exp(ETA_CL)
  V       = TVV
  LAGTIME = TVLAGTIME

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -CL/V * central

[error_model]
  DV ~ proportional(EPS)
";
    let parsed = super::parse_full_model(model_str).unwrap();
    // ODE models must report has_lagtime() via the indiv_param_names
    // fallback even when pk_indices doesn't contain PK_IDX_LAGTIME.
    assert!(
        parsed.model.has_lagtime(),
        "has_lagtime() must return true for an ODE model declaring LAGTIME"
    );

    let theta: Vec<f64> = parsed.model.default_params.theta.clone();
    let eta: Vec<f64> = vec![0.0; parsed.model.n_eta];
    let pk = (parsed.model.pk_param_fn)(&theta, &eta, &std::collections::HashMap::new(), 0.0);
    assert_eq!(
        pk.lagtime(),
        0.5,
        "LAGTIME must be routed to PK_IDX_LAGTIME for ODE models"
    );
}

#[test]
fn test_bioavailability_in_ode_model_routes_to_canonical_slot() {
    // Issue #122: an ODE model that declares an `F` individual parameter
    // must route its value to the canonical PK_IDX_F slot so the ODE
    // engine (`ode_predictions`, which reads `pk_params_flat[PK_IDX_F]`)
    // loads the dosing compartment with F·AMT — NONMEM's convention.
    // F is declared third here; `ode_param_slots` maps the name `F` to
    // PK_IDX_F (5) regardless of declaration position.
    let model_str = "
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta TVF(0.7, 0.001, 0.999)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  F  = TVF

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -1.0 * depot
  d/dt(central) = depot - CL/V * central

[error_model]
  DV ~ proportional(EPS)
";
    let parsed = super::parse_full_model(model_str).unwrap();
    let theta: Vec<f64> = parsed.model.default_params.theta.clone();
    let eta: Vec<f64> = vec![0.0; parsed.model.n_eta];
    let pk = (parsed.model.pk_param_fn)(&theta, &eta, &std::collections::HashMap::new(), 0.0);
    assert_eq!(
        pk.f_bio(),
        0.7,
        "F must be routed to PK_IDX_F for ODE models"
    );
}

#[test]
fn test_ode_structural_param_does_not_alias_bioavailability_slot() {
    // Issue #122 regression: an ODE model with ≥6 individual parameters and
    // NO F declared must NOT let a structural parameter land in PK_IDX_F
    // (slot 5) and be silently read as bioavailability. Before the
    // `ode_param_slots` fix, the 6th-declared param (here KE0=7.0) was
    // written positionally into slot 5, so `f_bio()` returned 7.0 and the
    // engine scaled every dose by 7×. With the canonical slot map, F is
    // reserved and undeclared → defaults to 1.0.
    let model_str = "
[parameters]
  theta TCL(1.0)
  theta TV(10.0)
  theta TQ(2.0)
  theta TV2(20.0)
  theta TKA(1.5)
  theta TKE0(7.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL  = TCL * exp(ETA_CL)
  V   = TV
  Q   = TQ
  V2  = TV2
  KA  = TKA
  KE0 = TKE0

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) = KA * depot - CL/V * central

[error_model]
  DV ~ proportional(EPS)
";
    let parsed = super::parse_full_model(model_str).unwrap();
    let theta: Vec<f64> = parsed.model.default_params.theta.clone();
    let eta: Vec<f64> = vec![0.0; parsed.model.n_eta];
    let pk = (parsed.model.pk_param_fn)(&theta, &eta, &std::collections::HashMap::new(), 0.0);
    assert_eq!(
        pk.f_bio(),
        1.0,
        "undeclared F must default to 1.0, not alias a structural parameter"
    );
    assert_eq!(pk.lagtime(), 0.0, "undeclared lagtime must default to 0.0");

    // The structural parameter KE0 must still round-trip: ode_param_slots
    // assigns it a free, non-reserved slot, pk_param_fn writes its value
    // there, and the RHS reads it back from the same slot. Assert it lands
    // off the engine-reserved F/lagtime slots and carries its value (7.0).
    let ke0_pos = parsed
        .model
        .indiv_param_names
        .iter()
        .position(|n| n == "KE0")
        .expect("KE0 declared");
    let ke0_slot = parsed.model.pk_indices[ke0_pos];
    assert_ne!(
        ke0_slot,
        crate::types::PK_IDX_F,
        "KE0 must not alias F slot"
    );
    assert_ne!(
        ke0_slot,
        crate::types::PK_IDX_LAGTIME,
        "KE0 must not alias lagtime slot"
    );
    assert_eq!(
        pk.values[ke0_slot], 7.0,
        "KE0 value must round-trip to its assigned slot"
    );
}

// ── [diffusion] block parsing ─────────────────────────────────────────

fn minimal_ode_model_with_diffusion(diffusion_lines: &str) -> String {
    format!(
        r#"
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma ADD ~ 0.1

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -(CL/V) * central

[diffusion]
{diffusion_lines}

[error_model]
  DV ~ additive(ADD)

[fit_options]
  method = foce
"#,
        diffusion_lines = diffusion_lines
    )
}

// ── [odes] init(state) = expr ─────────────────────────────────────────

/// Turnover model `d/dt(response) = KIN - KOUT*response` with a baseline
/// initial condition. `init_lines` is spliced into the `[odes]` block.
fn turnover_ode_model(init_lines: &str) -> String {
    format!(
        r#"
[parameters]
  theta TVKIN(10.0, 0.001, 100.0)
  theta TVKOUT(2.0, 0.001, 100.0)
  omega ETA_KIN ~ 0.09
  sigma ADD ~ 0.1

[individual_parameters]
  KIN  = TVKIN * exp(ETA_KIN)
  KOUT = TVKOUT

[structural_model]
  ode(obs_cmt=response, states=[response])

[odes]
{init_lines}
  d/dt(response) = KIN - KOUT * response

[error_model]
  DV ~ additive(ADD)

[fit_options]
  method = foce
"#,
        init_lines = init_lines
    )
}

#[test]
fn test_init_directive_builds_init_fn() {
    let src = turnover_ode_model("  init(response) = KIN / KOUT");
    let parsed = parse_full_model(&src).unwrap();
    let ode = parsed.model.ode_spec.as_ref().expect("ODE spec");
    assert!(ode.init_fn.is_some(), "init_fn should be populated");

    // Evaluate at typical values (eta = 0): KIN = 10, KOUT = 2 → 5.0.
    // For an ODE model, individual params occupy PkParams slots in
    // declaration order: KIN @ 0, KOUT @ 1.
    let mut params = [0.0; crate::types::MAX_PK_PARAMS];
    params[0] = 10.0;
    params[1] = 2.0;
    let u0 = ode.initial_state(&params);
    assert_eq!(u0.len(), 1);
    assert!(
        (u0[0] - 5.0).abs() < 1e-9,
        "init(response) = KIN/KOUT = 5, got {}",
        u0[0]
    );
}

// ── [initial_conditions] on analytical models (issue #521) ──

/// A minimal analytical 1-cpt oral model, with an optional
/// `[initial_conditions]` body spliced in.
fn analytical_oral_with_init(init_block: &str) -> String {
    format!(
        "[parameters]
  theta TVCL(3.0, 0.01, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
{init_block}
[error_model]
  DV ~ proportional(PROP_ERR)
"
    )
}

#[test]
fn initial_conditions_populates_analytical_init_central() {
    let src = analytical_oral_with_init(
        "
[initial_conditions]
  init(central) = CONC0 * V
",
    );
    let model = parse_full_model(&src).expect("parses").model;
    assert_eq!(model.analytical_init.len(), 1);
    let init = &model.analytical_init[0];
    // `central` is cmt 2 for an oral model.
    assert_eq!(init.cmt, 2);

    // amount_fn = CONC0 * V. With CONC0=7 (covariate) and V=20 (PK slot 1),
    // the amount is 140. eta=0, theta unused by the expression.
    let mut pk = crate::types::PkParams::default();
    pk.values[crate::types::PK_IDX_V] = 20.0;
    let mut cov = std::collections::HashMap::new();
    cov.insert("CONC0".to_string(), 7.0);
    let a0 = (init.amount_fn)(&[3.0, 20.0, 1.0], &[0.0], &cov, &pk);
    assert!((a0 - 140.0).abs() < 1e-9, "CONC0*V = 140, got {a0}");
}

#[test]
fn initial_conditions_depot_maps_to_cmt1() {
    let src = analytical_oral_with_init(
        "
[initial_conditions]
  init(depot) = 50
",
    );
    let model = parse_full_model(&src).expect("parses").model;
    assert_eq!(model.analytical_init.len(), 1);
    assert_eq!(model.analytical_init[0].cmt, 1);
}

#[test]
fn initial_conditions_rejected_on_ode_model() {
    // An ODE model must use `init(...)` inside [odes], not this block.
    let src = "[parameters]
  theta TVCL(3.0, 0.01, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -CL / V * central

[scaling]
  obs_scale = V

[initial_conditions]
  init(central) = 10

[error_model]
  DV ~ proportional(PROP_ERR)
";
    let err = match parse_full_model(src) {
        Ok(_) => panic!("expected an error"),
        Err(e) => e,
    };
    assert!(
        err.contains("[initial_conditions]") && err.contains("[odes]"),
        "error should point ODE users to [odes]: {err}"
    );
}

#[test]
fn initial_conditions_peripheral_rejected() {
    // two_cpt_oral: cmt 3 is the peripheral — not supported on the analytical
    // path (needs the cross-compartment Green's function).
    let src = "[parameters]
  theta TVCL(3.0, 0.01, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  theta TVQ(2.0, 0.01, 50.0)
  theta TVV2(40.0, 1.0, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  Q  = TVQ
  V2 = TVV2
  KA = TVKA

[structural_model]
  pk two_cpt_oral(cl=CL, v=V, q=Q, v2=V2, ka=KA)

[initial_conditions]
  init(3) = 10

[error_model]
  DV ~ proportional(PROP_ERR)
";
    let err = match parse_full_model(src) {
        Ok(_) => panic!("expected an error"),
        Err(e) => e,
    };
    assert!(
        err.contains("peripheral") || err.contains("central"),
        "error should explain the central/depot restriction: {err}"
    );
}

#[test]
fn initial_conditions_unknown_compartment_errors() {
    let src = analytical_oral_with_init(
        "
[initial_conditions]
  init(gut) = 10
",
    );
    let err = match parse_full_model(&src) {
        Ok(_) => panic!("expected an error"),
        Err(e) => e,
    };
    assert!(
        err.contains("unknown compartment"),
        "error should name the unknown compartment: {err}"
    );
}

#[test]
fn test_no_init_directive_seeds_zero() {
    let src = turnover_ode_model("");
    let parsed = parse_full_model(&src).unwrap();
    let ode = parsed.model.ode_spec.as_ref().unwrap();
    assert!(ode.init_fn.is_none());
    // initial_state falls back to zeros.
    assert_eq!(ode.initial_state(&[10.0, 2.0]), vec![0.0]);
}

#[test]
fn test_init_unknown_state_errors() {
    let src = turnover_ode_model("  init(nonexistent) = 1.0");
    let err = match parse_full_model(&src) {
        Err(e) => e,
        Ok(_) => panic!("expected unknown-state error"),
    };
    assert!(
        err.contains("init(nonexistent)") && err.contains("unknown state"),
        "expected unknown-state error, got: {}",
        err
    );
}

#[test]
fn test_duplicate_init_errors() {
    let src = turnover_ode_model("  init(response) = KIN / KOUT\n  init(response) = 0.0");
    let err = match parse_full_model(&src) {
        Err(e) => e,
        Ok(_) => panic!("expected duplicate-init error"),
    };
    assert!(
        err.contains("duplicate init(response)"),
        "expected duplicate-init error, got: {}",
        err
    );
}

#[test]
fn test_init_literal_value() {
    let src = turnover_ode_model("  init(response) = 7.5");
    let parsed = parse_full_model(&src).unwrap();
    let ode = parsed.model.ode_spec.as_ref().unwrap();
    assert_eq!(ode.initial_state(&[10.0, 2.0]), vec![7.5]);
}

#[test]
fn test_init_undefined_name_errors() {
    // #314: an init expression referencing a name that is neither a state
    // nor an individual parameter must error rather than silently reading
    // 0.0 via `eval_expression`. `KIN` is a declared parameter (must not be
    // flagged); `BASE` is undeclared.
    let src = turnover_ode_model("  init(response) = BASE * KIN");
    let err = match parse_full_model(&src) {
        Err(e) => e,
        Ok(_) => panic!("expected undefined-name error for init(response)"),
    };
    assert!(
        err.contains("init(response)"),
        "error should reference init(response), got: {err}"
    );
    // Only BASE is flagged — KIN resolves to the declared parameter.
    assert!(
        err.contains("undefined name(s): BASE."),
        "error should flag only BASE as undefined, got: {err}"
    );
}

#[test]
fn test_init_macheps_resolves_to_epsilon() {
    // MACHEPS is available in init expressions (eval_expression resolves it)
    // — the undefined-name check must accept it, and it evaluates to EPSILON.
    let src = turnover_ode_model("  init(response) = MACHEPS");
    let parsed = parse_full_model(&src).unwrap();
    let ode = parsed.model.ode_spec.as_ref().unwrap();
    assert_eq!(ode.initial_state(&[10.0, 2.0]), vec![f64::EPSILON]);
}

#[test]
fn test_init_macheps_is_case_insensitive() {
    // `eval_expression` resolves MACHEPS case-insensitively, so a mixed-case
    // spelling must not be rejected by the undefined-name check (regression:
    // exact-key matching against init_defined would have flagged it).
    let src = turnover_ode_model("  init(response) = MachEps");
    let parsed = parse_full_model(&src).unwrap();
    let ode = parsed.model.ode_spec.as_ref().unwrap();
    assert_eq!(ode.initial_state(&[10.0, 2.0]), vec![f64::EPSILON]);
}

#[test]
fn test_init_time_builtin_rejected() {
    // #994: a bare `TIME` parses to `Expression::Time`, not to a `Variable`, so
    // the undefined-name check could not see it and it was silently accepted —
    // while `TAFD`/`TAD`, its siblings in the same reserved list, were rejected.
    // Model time at an initial condition is 0 by definition, so the reference
    // could only ever flatten the expression (`init = TIME * X` → 0).
    for rhs in ["TIME", "TIME + 50", "time * KIN", "2 * (TIME + KIN)"] {
        let src = turnover_ode_model(&format!("  init(response) = {rhs}"));
        let err = parse_full_model(&src)
            .err()
            .unwrap_or_else(|| panic!("`init(response) = {rhs}` must be rejected"));
        assert!(
            err.contains("init(response)")
                && err.contains("time built-in(s): TIME")
                && err.contains("evaluated at the time origin"),
            "error should name init(response) and the TIME built-in, got: {err}"
        );
        assert!(
            err.contains("#994"),
            "error should cite the issue, got: {err}"
        );
    }
}

#[test]
fn test_init_time_clocks_rejected_in_every_casing() {
    // Every clock, every spelling, one diagnostic (#994). Before this, the four
    // names split three ways on representation alone: `TIME`/`time` were
    // accepted (an `Expression::Time` node the undefined-name walker cannot
    // see), `Time` was reported as a plain undefined name, and `T`/`TAFD`/`TAD`
    // were reported as undefined names. The casing asymmetry was the tell that
    // none of it was a scope decision.
    //
    // The assertion is on the *clock* diagnostic specifically, not merely on
    // "some error": the node guard feeds the same message as the undefined-name
    // split, so a regression that reported one as the other would otherwise
    // pass unnoticed.
    for base in ODE_INIT_REJECTED_BUILTINS {
        for rhs in [
            base.to_string(),
            base.to_lowercase(),
            // `Tafd` / `Time` / `T` — mixed case, the spelling that used to be
            // handled differently from the other two.
            base[..1].to_string() + &base[1..].to_lowercase(),
        ] {
            let src = turnover_ode_model(&format!("  init(response) = {rhs}"));
            let err = parse_full_model(&src)
                .err()
                .unwrap_or_else(|| panic!("`init(response) = {rhs}` must be rejected"));
            assert!(
                err.contains("init(response)") && err.contains("time built-in(s)"),
                "`{rhs}` should get the clock diagnostic, not a bare undefined-name \
                 error, got: {err}"
            );
        }
    }
}

#[test]
fn test_init_reports_clock_and_undefined_name_together() {
    // One parse, both problems (#994 review): an init RHS can carry a clock and
    // an undefined name at once, and reporting them one per parse costs the user
    // two round-trips for a single line.
    let src = turnover_ode_model("  init(response) = TIME + BASE");
    let err = parse_full_model(&src)
        .err()
        .expect("`init(response) = TIME + BASE` must be rejected");
    assert!(
        err.contains("time built-in(s): TIME"),
        "error should name the clock, got: {err}"
    );
    assert!(
        err.contains("undefined name(s): BASE"),
        "error should name the undefined BASE in the same message, got: {err}"
    );
}

#[test]
fn test_init_undefined_name_message_lists_the_real_scope() {
    // The diagnostic is the only place the init scope rule is written down
    // (#994), so it must name every accepted built-in — including `MIXNUM`,
    // which is in scope and was absent from the message — and say that the time
    // built-ins are not.
    let src = turnover_ode_model("  init(response) = BASE");
    let err = parse_full_model(&src)
        .err()
        .expect("undefined name must be rejected");
    for builtin in ODE_INIT_SCOPE_BUILTINS {
        assert!(
            err.contains(builtin),
            "message should list the accepted built-in {builtin}, got: {err}"
        );
    }
    for rejected in ODE_INIT_REJECTED_BUILTINS {
        assert!(
            err.contains(rejected),
            "message should list the out-of-scope built-in {rejected}, got: {err}"
        );
    }
}

#[test]
fn test_init_mixnum_accepted_in_mixture_model() {
    // The other payload-free built-in node, `Expression::MixNum`, is invisible to
    // the undefined-name check for the same structural reason `TIME` was — but
    // unlike `TIME` it does real work at init: it resolves to the subject's class
    // and discriminates the classes. It stays in scope (#994 comment); the class-1
    // default gives 100 here.
    let src = r"
[parameters]
  theta TVKIN(10.0, 0.001, 100.0)
  theta TVKOUT(2.0, 0.001, 100.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_KIN ~ 0.09
  sigma ADD ~ 0.1

[mixture]
  nsub = 2
  logit(1) = MIXL

[individual_parameters]
  KIN  = TVKIN * exp(ETA_KIN)
  KOUT = TVKOUT

[structural_model]
  ode(obs_cmt=response, states=[response])

[odes]
  init(response) = MIXNUM * 100
  d/dt(response) = KIN - KOUT * response

[error_model]
  DV ~ additive(ADD)
";
    let parsed = parse_full_model(src).expect("MIXNUM is in scope in an init expression");
    let ode = parsed.model.ode_spec.as_ref().expect("ODE spec");
    assert_eq!(ode.initial_state(&[10.0, 2.0]), vec![100.0]);
}

#[test]
fn test_analytical_init_time_builtin_rejected() {
    // #994: the `[initial_conditions]` surface took the same leak. The baseline
    // amount is evaluated once per subject at t = 0, so `TIME` reads the
    // model-time thread-local's 0.0 and the reference contributes nothing.
    for rhs in ["TIME", "TIME + 50", "time * V"] {
        let src = analytical_oral_with_init(&format!(
            "
[initial_conditions]
  init(central) = {rhs}
"
        ));
        let err = parse_full_model(&src)
            .err()
            .unwrap_or_else(|| panic!("`init(central) = {rhs}` must be rejected"));
        assert!(
            err.contains("[initial_conditions] init") && err.contains("`TIME` built-in"),
            "error should name the block and the TIME built-in, got: {err}"
        );
        assert!(
            err.contains("#994"),
            "error should cite the issue, got: {err}"
        );
    }
}

#[test]
fn test_analytical_init_clock_names_are_ordinary_covariates() {
    // The `[odes]`-only built-ins are ordinary covariates everywhere else — the
    // same deliberate rule `[scaling]` follows so a dataset that really carries a
    // `TAD` column can use it (#1028, `ODE_ONLY_BUILTINS` in `api::validation`).
    // `[initial_conditions]` is "everywhere else", so only `TIME` is rejected
    // there; `T`/`TAFD`/`TAD`/`MACHEPS` parse as covariate leaves and become
    // required data columns. Pinned because `ODE_INIT_REJECTED_BUILTINS` is
    // public and a consumer mirroring it on an analytical model would otherwise
    // reject a legal reference (#994 review).
    for name in ["T", "TAFD", "TAD", "MACHEPS"] {
        let src = analytical_oral_with_init(&format!(
            "
[initial_conditions]
  init(central) = {name} * 10
"
        ));
        let model = parse_model_string(&src)
            .unwrap_or_else(|e| panic!("`init(central) = {name} * 10` must parse: {e}"));
        assert_eq!(model.analytical_init.len(), 1);
        assert!(
            model.referenced_covariates.iter().any(|c| c == name),
            "`{name}` must be registered as a required data column, got: {:?}",
            model.referenced_covariates
        );
    }
}

#[test]
fn test_diffusion_block_parsed_into_theta() {
    let src = minimal_ode_model_with_diffusion("  central ~ 0.05");
    let parsed = parse_full_model(&src).unwrap();
    let m = &parsed.model;
    // DIFF_CENTRAL should be appended as the last theta
    assert!(
        m.theta_names.iter().any(|n| n == "DIFF_CENTRAL"),
        "expected DIFF_CENTRAL in theta_names, got {:?}",
        m.theta_names
    );
    let idx = m
        .theta_names
        .iter()
        .position(|n| n == "DIFF_CENTRAL")
        .unwrap();
    assert!(
        (m.default_params.theta[idx] - 0.05).abs() < 1e-9,
        "initial diffusion variance should be 0.05"
    );
}

#[test]
fn test_diffusion_block_sets_diffusion_theta_start() {
    let src = minimal_ode_model_with_diffusion("  central ~ 0.01");
    let parsed = parse_full_model(&src).unwrap();
    let m = &parsed.model;
    assert!(
        m.diffusion_theta_start.is_some(),
        "diffusion_theta_start must be set"
    );
    assert_eq!(m.diffusion_state_indices.len(), 1);
    assert!(m.is_sde());
}

#[test]
fn test_diffusion_block_fix_flag() {
    let src = minimal_ode_model_with_diffusion("  central ~ 0.02 FIX");
    let parsed = parse_full_model(&src).unwrap();
    let m = &parsed.model;
    let idx = m
        .theta_names
        .iter()
        .position(|n| n == "DIFF_CENTRAL")
        .unwrap();
    assert!(
        m.default_params.theta_fixed[idx],
        "DIFF_CENTRAL should be FIX"
    );
}

#[test]
fn test_diffusion_block_unknown_state_error() {
    let src = minimal_ode_model_with_diffusion("  depot ~ 0.01");
    assert!(
        parse_full_model(&src).is_err(),
        "unknown state in [diffusion] must be an error"
    );
}

#[test]
fn test_diffusion_on_analytical_model_is_error() {
    let src = r#"
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma ADD ~ 0.1
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[diffusion]
  central ~ 0.01
[error_model]
  additive
"#;
    assert!(
        parse_full_model(src).is_err(),
        "[diffusion] on an analytical model must be an error"
    );
}

// ─── [covariate_nn NAME] block parsing ──────────────────────────────

/// Minimal full model with a `[covariate_nn]` block. Re-used across tests.
/// The base [parameters]/[individual_parameters] still use the analytical
/// form — this PR only registers the NN-weight thetas and stores the
/// mapper handle; it does not yet route PK params through the NN.
fn covariate_nn_model_src(nn_block: &str) -> String {
    format!(
        r#"
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma ADD ~ 0.1

{nn_block}

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ additive(ADD)
"#
    )
}

#[cfg(feature = "nn")]
#[test]
fn test_covariate_nn_block_parses_and_appends_thetas() {
    let src = covariate_nn_model_src(
            "[covariate_nn TYPICAL_PK]\n  inputs = [WT, CRCL]\n  outputs = [CL, V]\n  layers = [4]\n  activation = tanh\n  output = softplus\n",
        );
    let model = parse_model_string(&src).expect("model must parse with --features nn");

    // Mapper present and named correctly.
    assert_eq!(model.covariate_nns.len(), 1);
    let nn = &model.covariate_nns[0];
    assert_eq!(nn.name, "TYPICAL_PK");
    // 2 inputs -> 4 hidden -> 2 outputs:
    // W_1: 4*2 = 8, b_1: 4, W_2: 2*4 = 8, b_2: 2 → total 22 weights.
    let n_weights = 4 * 2 + 4 + 2 * 4 + 2;
    use crate::nn::CovariateMapper;
    assert_eq!(nn.mapper.n_weights(), n_weights);

    // Auto-generated thetas land at the tail of the theta vector.
    // Base thetas: TVCL, TVV. So weights_offset == 2.
    assert_eq!(nn.weights_offset, 2);
    assert_eq!(model.n_theta, 2 + n_weights);

    // Theta-name convention is W_<NAME>_<l>_<i>_<j> / B_<NAME>_<l>_<i>.
    let names: &[String] = &model.theta_names;
    assert_eq!(names[0], "TVCL");
    assert_eq!(names[1], "TVV");
    assert_eq!(names[2], "W_TYPICAL_PK_1_1_1");
    // Last name is the last bias of layer 2 (output layer, 2 outputs).
    assert_eq!(names[names.len() - 1], "B_TYPICAL_PK_2_2");
}

#[cfg(feature = "nn")]
#[test]
fn test_covariate_nn_block_rejects_unknown_pk_output() {
    let src = covariate_nn_model_src(
            "[covariate_nn FOO]\n  inputs = [WT]\n  outputs = [NOT_A_PK_PARAM]\n  layers = [3]\n  activation = relu\n",
        );
    let err = parse_model_string(&src).expect_err("unknown PK output must error");
    assert!(
        err.contains("NOT_A_PK_PARAM"),
        "error should name the bad output, got: {err}"
    );
}

#[cfg(feature = "nn")]
#[test]
fn test_covariate_nn_block_rejects_unknown_activation() {
    let src = covariate_nn_model_src(
            "[covariate_nn FOO]\n  inputs = [WT]\n  outputs = [CL]\n  layers = [3]\n  activation = quack\n",
        );
    let err = parse_model_string(&src).expect_err("bad activation must error");
    assert!(
        err.contains("quack"),
        "error should name the bad activation, got: {err}"
    );
}

#[cfg(feature = "nn")]
#[test]
fn test_covariate_nn_block_rejects_unknown_key() {
    let src = covariate_nn_model_src(
            "[covariate_nn FOO]\n  inputs = [WT]\n  outputs = [CL]\n  layers = [3]\n  activation = relu\n  not_a_real_key = 42\n",
        );
    let err = parse_model_string(&src).expect_err("unknown key must error");
    assert!(
        err.contains("not_a_real_key"),
        "error should name the bad key, got: {err}"
    );
}

#[cfg(feature = "nn")]
#[test]
fn test_covariate_nn_block_missing_required_field() {
    // Missing `layers`.
    let src = covariate_nn_model_src(
        "[covariate_nn FOO]\n  inputs = [WT]\n  outputs = [CL]\n  activation = relu\n",
    );
    let err = parse_model_string(&src).expect_err("missing layers must error");
    assert!(
        err.contains("layers"),
        "error should mention `layers`, got: {err}"
    );
}

#[cfg(feature = "nn")]
#[test]
fn test_multiple_covariate_nn_blocks_appear_in_sorted_order() {
    // Two blocks; the second-declared one should still come first by
    // alphabetical sort to keep theta-ordering reproducible.
    let src = covariate_nn_model_src(
            "[covariate_nn ZETA]\n  inputs = [WT]\n  outputs = [CL]\n  layers = [2]\n  activation = tanh\n\n\
             [covariate_nn ALPHA]\n  inputs = [WT]\n  outputs = [V]\n  layers = [2]\n  activation = tanh\n",
        );
    let model = parse_model_string(&src).expect("two NN blocks must parse");
    assert_eq!(model.covariate_nns.len(), 2);
    assert_eq!(model.covariate_nns[0].name, "ALPHA");
    assert_eq!(model.covariate_nns[1].name, "ZETA");
    // ALPHA's weights are first; ZETA's start where ALPHA's end.
    use crate::nn::CovariateMapper;
    let alpha_weights = model.covariate_nns[0].mapper.n_weights();
    assert_eq!(model.covariate_nns[1].weights_offset, 2 + alpha_weights);
}

/// When ferx-core is built without `--features nn`, a `[covariate_nn]`
/// block must produce a clear feature-gate error rather than being
/// silently ignored.
#[cfg(not(feature = "nn"))]
#[test]
fn test_covariate_nn_block_without_nn_feature_errors() {
    let src = covariate_nn_model_src(
            "[covariate_nn FOO]\n  inputs = [WT]\n  outputs = [CL]\n  layers = [2]\n  activation = tanh\n",
        );
    let err = parse_model_string(&src).expect_err("must error without --features nn");
    assert!(
        err.contains("nn"),
        "error should mention the nn feature, got: {err}"
    );
}

/// Sanity for the named-block parser extension itself (independent of the
/// NN feature). `[block_type NAME]` should be recognised and parsed. Uses a
/// registry block (`covariate_nn`, which is named-only) rather than an invented
/// name — since #1040 `extract_blocks` rejects unknown block names.
#[test]
fn test_extract_blocks_recognizes_named_block_form() {
    let src = "
[parameters]
  theta T1(1.0, 0.001, 10.0)

[covariate_nn FOO]
  key = value
";
    let extracted = extract_blocks(src).unwrap();
    // Unnamed block intact.
    assert!(extracted.unnamed.contains_key("parameters"));
    // Named block captured by type + instance.
    let by_inst = extracted
        .named
        .get("covariate_nn")
        .expect("named block extracted");
    let lines = by_inst.get("FOO").expect("instance FOO present");
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0], "key = value");
}

// ─── Closed-world `[block]` names (#1040) ────────────────────────────────────

/// A minimal but complete model, with `{extra}` spliced in after the required
/// blocks so a test can add one offending header.
fn model_with_extra_block(extra: &str) -> String {
    format!(
        "\
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)

{extra}"
    )
}

/// The headline regression: a misspelled optional block used to be dropped on
/// the floor — `[fit_option]` left the fit running with the default method and
/// no covariance step while `ferx check` still said `valid: true`.
#[test]
fn test_unknown_block_name_is_rejected_with_line_and_valid_set() {
    let src = model_with_extra_block("[fit_option]\n  method = focei\n");
    let err = parse_model_string(&src).expect_err("unknown block must be an error");
    assert!(
        err.starts_with("Unknown block `[fit_option]` (line 16)"),
        "message should name the block and its header line, got: {err}"
    );
    assert!(
        err.contains("did you mean `[fit_options]`"),
        "near miss should be suggested, got: {err}"
    );
    // The valid set is enumerated, in the `[event_model]: unknown key ...` style.
    // Assert on names present in every build — the enumeration is filtered by
    // the features this binary carries (see `known_block_names`).
    assert!(err.contains("Valid blocks: adaptive_dosing, "), "{err}");
    assert!(err.contains("fit_options, "), "{err}");
    assert!(err.contains("structural_model."), "{err}");
}

/// Every offender is reported in one pass, so a batch of typos does not turn
/// into a fix-one-reparse loop.
#[test]
fn test_unknown_block_names_are_all_reported_together() {
    let src = model_with_extra_block("[scalings]\n  V = 1.0\n\n[outputs]\n  CL\n");
    let err = parse_model_string(&src).expect_err("unknown blocks must be an error");
    assert!(err.contains("`[scalings]`"), "{err}");
    assert!(err.contains("`[outputs]`"), "{err}");
    assert!(err.contains("did you mean `[scaling]`"), "{err}");
    assert!(err.contains("did you mean `[output]`"), "{err}");
}

/// An invented name with no near miss still errors — just without a hint.
#[test]
fn test_unknown_block_without_near_match_has_no_suggestion() {
    let src = model_with_extra_block("[qqqqqqqq]\n  key = value\n");
    let err = parse_model_string(&src).expect_err("unknown block must be an error");
    assert!(err.contains("`[qqqqqqqq]`"), "{err}");
    assert!(!err.contains("did you mean"), "{err}");
}

/// A repeated unknown header is reported once, not once per occurrence.
#[test]
fn test_unknown_block_reported_once_per_name() {
    let src =
        model_with_extra_block("[fit_option]\n  method = focei\n\n[fit_option]\n  maxiter = 5\n");
    let err = parse_model_string(&src).expect_err("unknown block must be an error");
    assert_eq!(err.matches("`[fit_option]`").count(), 1, "{err}");
}

/// Block names are case-folded before the lookup, as they always have been —
/// `[FIT_OPTIONS]` is valid, not unknown.
#[test]
fn test_uppercase_block_name_is_not_unknown() {
    let src = model_with_extra_block("[FIT_OPTIONS]\n  maxiter = 5\n");
    assert!(parse_model_string(&src).is_ok());
}

/// An instance name on a block that takes none is the same silent drop by
/// another route: it lands in the `named` map that nothing reads.
#[test]
fn test_instance_name_on_unnamed_only_block_is_rejected() {
    let src = model_with_extra_block("[fit_options DOSE]\n  maxiter = 5\n");
    let err = parse_model_string(&src).expect_err("instance name must be an error");
    assert!(
        err.contains("`[fit_options DOSE]` (line 16) does not take an instance name"),
        "{err}"
    );
    assert!(err.contains("write `[fit_options]`"), "{err}");
}

/// The mirror case: `[covariate_nn]` is read only out of the named map, so an
/// unnamed one was silently ignored.
#[test]
fn test_named_only_block_without_instance_name_is_rejected() {
    let src = model_with_extra_block("[covariate_nn]\n  inputs = [WT]\n");
    let err = parse_model_string(&src).expect_err("missing instance name must be an error");
    assert!(
        err.contains("`[covariate_nn]` (line 16) requires an instance name"),
        "{err}"
    );
    assert!(err.contains("write `[covariate_nn NAME]`"), "{err}");
}

/// A block whose cargo feature is off is *known* but unusable — reject it
/// rather than parse the file as if the endpoint were not there.
#[cfg(not(feature = "survival"))]
#[test]
fn test_feature_gated_block_rejected_without_its_feature() {
    let src = model_with_extra_block("[event_model]\n  family = exponential\n");
    let err = parse_model_string(&src).expect_err("feature-gated block must be an error");
    assert!(
        err.contains(
            "`[event_model]` (line 16) requires building ferx-core with `--features survival`"
        ),
        "{err}"
    );
}

/// With the feature on, the same file parses — the gate is the only reason the
/// block is refused above.
#[cfg(feature = "survival")]
#[test]
fn test_feature_gated_block_accepted_with_its_feature() {
    let src = model_with_extra_block(
        "[event_model]\n  cmt = 2\n  family = exponential\n  scale = TVCL\n",
    );
    assert!(parse_model_string(&src).is_ok());
}

#[test]
fn test_nearest_block_name_thresholds() {
    // Prefix relationship, either direction.
    assert_eq!(nearest_block_name("fit_option"), Some("fit_options"));
    assert_eq!(nearest_block_name("parameters_pk"), Some("parameters"));
    // Within edit distance 2.
    assert_eq!(nearest_block_name("odez"), Some("odes"));
    // Far enough away that a suggestion would be noise.
    assert_eq!(nearest_block_name("qqqqqqqq"), None);
}

/// The registry is what the rest of the parser looks up, so nothing may be in
/// one and not the other. `known_block_names` is also the list `ferx-r` and
/// `ferxtranslate` are meant to read instead of keeping their own copy.
#[test]
fn test_known_block_names_are_sorted_and_unique() {
    let names = known_block_names();
    let mut sorted = names.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(names, sorted, "BLOCK_REGISTRY must be sorted and unique");
    assert!(names.contains(&"fit_options"));
    // Deprecated names are not valid, so they must never be advertised.
    assert!(!names.contains(&"initial_values"));
}

/// The advertised list is what *this* binary accepts. A build without a
/// feature must not name the blocks it would then refuse — a consumer told to
/// source the list from the engine (`ferx-r`) would otherwise call
/// `[markov_model]` valid against a core that rejects it.
#[test]
fn test_known_block_names_track_the_build_features() {
    let names = known_block_names();
    assert_eq!(
        names.contains(&"event_model"),
        cfg!(feature = "survival"),
        "event_model must be advertised iff `survival` is on"
    );
    assert_eq!(
        names.contains(&"markov_model"),
        cfg!(feature = "markov"),
        "markov_model must be advertised iff `markov` is on"
    );
    // Ungated blocks are there whatever the build.
    assert!(names.contains(&"parameters") && names.contains(&"odes"));
    // A registry entry naming a feature this function does not know can only be
    // a typo in the table; treat it as enabled so the mistake never rejects a
    // valid model.
    assert!(block_feature_enabled("not-a-real-feature"));
}

/// A block that *was* ferx syntax gets its own code and remediation rather
/// than a bare "unknown block": `initial_values` predates inline inits, the
/// parser stopped reading it silently, and `nearest_block_name` finds nothing
/// close enough to suggest (`initial_conditions` is far past edit distance 2),
/// so the generic path would leave the user with no explanation at all.
#[test]
fn test_deprecated_block_is_rejected_with_its_remediation() {
    assert_eq!(nearest_block_name("initial_values"), None);
    let src = model_with_extra_block("[initial_values]\n  theta = [0.2, 10.0]\n  sigma = [0.02]\n");
    let err = parse_model_string(&src).expect_err("deprecated block must be an error");
    assert!(
        err.starts_with("Deprecated block `[initial_values]` (line 16) is no longer read:"),
        "{err}"
    );
    assert!(err.contains("declared inline in `[parameters]`"), "{err}");
    assert!(err.contains("delete it"), "{err}");
    // Not the generic bucket: no "valid blocks" dump, no did-you-mean.
    assert!(!err.contains("Valid blocks"), "{err}");
}

/// An unknown header written with an instance name is echoed as written, so
/// the message can be grepped for in the user's own file — and two instances
/// of the same unknown type are two findings, not one entry that names only
/// the first.
#[test]
fn test_unknown_named_block_reports_its_instance_name() {
    let src = model_with_extra_block("[foo BAR]\n  k = 1\n\n[foo BAZ]\n  k = 2\n");
    let err = parse_model_string(&src).expect_err("unknown block must be an error");
    assert!(err.contains("`[foo BAR]` (line 16)"), "{err}");
    assert!(err.contains("`[foo BAZ]` (line 19)"), "{err}");
}

// ─── [covariate_nn] dot-access + pk_param_fn dispatch (Phase A M1 step 3) ────

#[cfg(feature = "nn")]
fn covariate_nn_dotted_model_src() -> String {
    // Two-output NN feeding both CL and V. Etas on the final PK params,
    // matching the Phase A M2 mu-ref form. The fit-objective surface
    // (method = nn_mse) isn't wired yet, so we only exercise the
    // pk_param_fn closure here — that's the simulate path.
    r#"
[parameters]
  theta TVKA(1.0, 0.001, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma ADD ~ 0.1

[covariate_nn TYPICAL_PK]
  inputs = [WT, CRCL]
  outputs = [CL, V]
  layers = [4]
  activation = tanh
  output = softplus

[individual_parameters]
  CL = TYPICAL_PK.CL * exp(ETA_CL)
  V  = TYPICAL_PK.V  * exp(ETA_V)
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ additive(ADD)
"#
    .to_string()
}

#[cfg(feature = "nn")]
#[test]
fn test_dot_access_parses_into_nn_output_node() {
    let src = covariate_nn_dotted_model_src();
    let model = parse_model_string(&src).expect("dot-access must parse");
    // 1 base theta (TVKA) + NN weights for a 2->4->2 net:
    //   W_1: 4*2 = 8, b_1: 4, W_2: 2*4 = 8, b_2: 2 → 22 weights.
    assert_eq!(model.n_theta, 1 + 22);
    assert_eq!(model.covariate_nns.len(), 1);
    assert_eq!(model.covariate_nns[0].name, "TYPICAL_PK");
    // Weights start right after the user-declared base thetas.
    assert_eq!(model.covariate_nns[0].weights_offset, 1);
}

/// Round-trip: call the closure with known theta values (including known
/// NN weights) and confirm the PK params it produces match what
/// `NamedMlpMapper::forward_raw` returns directly.
#[cfg(feature = "nn")]
#[test]
fn test_pk_param_fn_dispatches_through_nn() {
    use crate::nn::CovariateMapper;
    use crate::types::{PK_IDX_CL, PK_IDX_V};

    let src = covariate_nn_dotted_model_src();
    let model = parse_model_string(&src).expect("model parses");
    let theta: Vec<f64> = model.default_params.theta.clone();
    let eta = vec![0.0_f64, 0.0_f64]; // zero etas → PK params == NN typical values.
    let mut cov = HashMap::new();
    cov.insert("WT".to_string(), 70.0);
    cov.insert("CRCL".to_string(), 95.0);

    let pk = (model.pk_param_fn)(&theta, &eta, &cov, 0.0);

    // What the NN itself would emit, sliced from the same theta vector.
    let nn = &model.covariate_nns[0];
    let weights = &theta[nn.weights_offset..nn.weights_offset + nn.mapper.n_weights()];
    let nn_outputs = nn.mapper.forward_raw(weights, &cov).unwrap();

    // pk_param_fn writes to PkParams by name: output[0] = CL, output[1] = V.
    assert!((pk.values[PK_IDX_CL] - nn_outputs[0]).abs() < 1e-12);
    assert!((pk.values[PK_IDX_V] - nn_outputs[1]).abs() < 1e-12);
    // KA is a plain theta-only path.
    assert!((pk.values[crate::types::PK_IDX_KA] - theta[0]).abs() < 1e-12);
}

/// Non-zero etas: the mu-ref composition `TYPICAL_PK.CL * exp(ETA_CL)`
/// must apply log-normal IIV on top of the NN typical value.
#[cfg(feature = "nn")]
#[test]
fn test_pk_param_fn_composes_eta_on_top_of_nn_output() {
    use crate::nn::CovariateMapper;
    use crate::types::PK_IDX_CL;

    let src = covariate_nn_dotted_model_src();
    let model = parse_model_string(&src).expect("model parses");
    let theta = model.default_params.theta.clone();
    let mut cov = HashMap::new();
    cov.insert("WT".to_string(), 70.0);
    cov.insert("CRCL".to_string(), 95.0);

    let nn = &model.covariate_nns[0];
    let weights = &theta[nn.weights_offset..nn.weights_offset + nn.mapper.n_weights()];
    let nn_outputs = nn.mapper.forward_raw(weights, &cov).unwrap();
    let tv_cl = nn_outputs[0];

    // eta = +0.3 → CL should be tv_cl * exp(0.3).
    let pk = (model.pk_param_fn)(&theta, &[0.3_f64, 0.0_f64], &cov, 0.0);
    let expected = tv_cl * 0.3_f64.exp();
    assert!(
        (pk.values[PK_IDX_CL] - expected).abs() < 1e-10,
        "expected {expected}, got {}",
        pk.values[PK_IDX_CL]
    );
}

#[cfg(feature = "nn")]
#[test]
fn test_dot_access_rejects_unknown_output() {
    // GARBAGE is not in `outputs = [CL, V]`.
    let src = r#"
[parameters]
  theta TVKA(1.0, 0.001, 100.0)
  omega ETA_CL ~ 0.09
  sigma ADD ~ 0.1

[covariate_nn TYPICAL_PK]
  inputs = [WT]
  outputs = [CL, V]
  layers = [3]
  activation = tanh

[individual_parameters]
  CL = TYPICAL_PK.GARBAGE * exp(ETA_CL)
  V  = 50
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ additive(ADD)
"#;
    let err = parse_model_string(src).expect_err("unknown output must error");
    assert!(
        err.contains("GARBAGE"),
        "error should name the bad output, got: {err}"
    );
}

#[cfg(feature = "nn")]
#[test]
fn test_dot_access_on_unknown_nn_name_falls_back_to_parse_error() {
    // FOO is not a declared [covariate_nn] block. `FOO.CL` must error
    // (the parser stops at the `.` it can't classify).
    let src = r#"
[parameters]
  theta TVKA(1.0, 0.001, 100.0)
  omega ETA_CL ~ 0.09
  sigma ADD ~ 0.1

[individual_parameters]
  CL = FOO.CL
  V  = 50
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ additive(ADD)
"#;
    assert!(parse_model_string(src).is_err());
}

// ─── Fix 1 + fix 2: NN-anchored mu-ref detection + tv_fn parity ─────

/// Fix 1: `TYPICAL_PK.CL * exp(ETA_CL)` should be recognised as a
/// lognormal mu-ref, with `MuRef.theta_name = "TYPICAL_PK.CL"` (the
/// structured form). Downstream consumers that only know about plain
/// thetas silently fall back to FD; the M2 AD-aware consumer is a
/// follow-up.
#[cfg(feature = "nn")]
#[test]
fn test_detect_mu_refs_recognises_nn_anchored_lognormal() {
    let src = covariate_nn_dotted_model_src();
    let model = parse_model_string(&src).expect("model parses");
    let cl_ref = model
        .mu_refs
        .get("ETA_CL")
        .expect("ETA_CL must be detected as mu-referenced");
    assert_eq!(cl_ref.theta_name, "TYPICAL_PK.CL");
    assert!(
        cl_ref.log_transformed,
        "TYPICAL_PK.CL * exp(ETA_CL) is lognormal"
    );
    let v_ref = model.mu_refs.get("ETA_V").expect("ETA_V mu-referenced");
    assert_eq!(v_ref.theta_name, "TYPICAL_PK.V");
    assert!(v_ref.log_transformed);
}

/// Fix 1 follow-on: `eta_param_info` still classifies NN-anchored
/// patterns as `LogNormal` — the eta's statistical shape is unchanged;
/// only the anchor differs.
#[cfg(feature = "nn")]
#[test]
fn test_eta_param_info_classifies_nn_anchored_lognormal() {
    use crate::types::EtaParamType;
    let src = covariate_nn_dotted_model_src();
    let model = parse_model_string(&src).expect("model parses");
    let cl = model
        .eta_param_info
        .iter()
        .find(|i| i.eta_name == "ETA_CL")
        .expect("ETA_CL classification present");
    assert_eq!(cl.param_type, EtaParamType::LogNormal);
    let v = model
        .eta_param_info
        .iter()
        .find(|i| i.eta_name == "ETA_V")
        .expect("ETA_V classification present");
    assert_eq!(v.param_type, EtaParamType::LogNormal);
}

/// Fix 2: `tv_fn` (the eta=0 typical-value closure) must produce the
/// same values as the NN forward pass. Previously the unindexed
/// evaluator returned 0.0 for `NnOutput`, so `tv_fn` would have
/// silently produced zero TVs for NN-bearing models — a footgun for
/// the AD fast path the M2 PR will lean on.
#[cfg(feature = "nn")]
#[test]
fn test_tv_fn_dispatches_through_nn() {
    use crate::nn::CovariateMapper;
    let src = covariate_nn_dotted_model_src();
    let model = parse_model_string(&src).expect("model parses");
    let theta = model.default_params.theta.clone();
    let mut cov = HashMap::new();
    cov.insert("WT".to_string(), 70.0);
    cov.insert("CRCL".to_string(), 95.0);

    let tv_fn = model
        .tv_fn
        .as_ref()
        .expect("analytical model -> Some(tv_fn)");
    let tvs = tv_fn(&theta, &cov);

    // tv_fn returns values in `indiv_param_names` declaration order:
    // CL, V, KA. At eta=0 the lognormal mu-ref `tv * exp(0)` collapses
    // to tv, which equals the NN's raw output for CL and V.
    let nn = &model.covariate_nns[0];
    let weights = &theta[nn.weights_offset..nn.weights_offset + nn.mapper.n_weights()];
    let nn_outputs = nn.mapper.forward_raw(weights, &cov).unwrap();

    assert!(
        (tvs[0] - nn_outputs[0]).abs() < 1e-12,
        "TV[CL] from tv_fn = {} vs NN output = {}",
        tvs[0],
        nn_outputs[0]
    );
    assert!(
        (tvs[1] - nn_outputs[1]).abs() < 1e-12,
        "TV[V] from tv_fn = {} vs NN output = {}",
        tvs[1],
        nn_outputs[1]
    );
    // KA = TVKA (plain theta path, unchanged).
    assert!((tvs[2] - theta[0]).abs() < 1e-12);
}

// ── [scaling] block parser tests ────────────────────────────────────────

/// Analytical 1-cpt IV bolus model template — small enough to compose
/// scaling-block variants on top.
fn analytical_model_with_scaling(scaling_block: Option<&str>) -> String {
    let mut s = String::from(
        "\
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(50.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.05 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method  = focei
  maxiter = 10
  gradient = fd
",
    );
    if let Some(block) = scaling_block {
        s.push_str("\n[scaling]\n");
        s.push_str(block);
    }
    s
}

/// ODE 1-cpt oral template — small enough to compose Form C variants.
fn ode_model_with_scaling(struct_line: &str, scaling_block: Option<&str>) -> String {
    let mut s = format!(
        "\
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(50.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.05 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA

[structural_model]
  {struct_line}

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) = KA * depot - CL/V * central

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method  = focei
  maxiter = 10
  gradient = fd
",
        struct_line = struct_line
    );
    if let Some(block) = scaling_block {
        s.push_str("\n[scaling]\n");
        s.push_str(block);
    }
    s
}

#[test]
fn test_parse_scaling_none() {
    let src = analytical_model_with_scaling(None);
    let model = parse_model_string(&src).expect("base model parses");
    assert!(matches!(model.scaling, ScalingSpec::None));
}

#[test]
fn test_parse_scaling_scalar() {
    let src = analytical_model_with_scaling(Some("  obs_scale = 1000\n"));
    let model = parse_model_string(&src).expect("scalar scaling parses");
    match model.scaling {
        ScalingSpec::ScalarScale(k) => assert!((k - 1000.0).abs() < 1e-12),
        other => panic!("expected ScalarScale, got {:?}", other),
    }
}

#[test]
fn test_parse_scaling_scalar_rejects_zero() {
    let src = analytical_model_with_scaling(Some("  obs_scale = 0\n"));
    let err = parse_model_string(&src).expect_err("obs_scale = 0 must be rejected");
    assert!(
        err.to_lowercase().contains("strictly positive"),
        "expected strictly-positive error, got: {}",
        err
    );
}

#[test]
fn test_parse_scaling_scalar_rejects_negative() {
    let src = analytical_model_with_scaling(Some("  obs_scale = -1000\n"));
    let err = parse_model_string(&src).expect_err("negative obs_scale must be rejected");
    assert!(
        err.to_lowercase().contains("strictly positive"),
        "expected strictly-positive error, got: {}",
        err
    );
}

#[test]
fn test_parse_scaling_expression_with_theta_evaluates() {
    // `obs_scale = TVV / 10` — references a theta, not an indiv param.
    let src = analytical_model_with_scaling(Some("  obs_scale = TVV / 10\n"));
    let model = parse_model_string(&src).expect("expression scaling parses");
    match model.scaling {
        ScalingSpec::ExpressionScale { ref scale_fn, .. } => {
            // TVV = 50 (the parsed default), so scale = 50/10 = 5.
            let theta = vec![1.0, 50.0]; // [TVCL, TVV]
            let eta = vec![0.0];
            let cov = HashMap::new();
            let pk = PkParams::default();
            let s = scale_fn(&theta, &eta, &cov, &pk);
            assert!((s - 5.0).abs() < 1e-12, "expected 5.0, got {}", s);
        }
        other => panic!("expected ExpressionScale, got {:?}", other),
    }
}

#[test]
fn scale_deriv_program_matches_fd() {
    // The differentiable scale program's `∂scale/∂(θ,η)` must match FD of the
    // scale closure composed with `pk_param_fn` (issue #367). `obs_scale =
    // 1000 / V`, V = TVV·exp(ETA_V) — so the scale depends on θ (via TVV) and
    // η (via ETA_V) through the individual parameter V.
    use crate::sens::dual2::Dual2;
    let src = "\
[parameters]
  theta TVCL(0.13, 0.001, 10.0)
  theta TVV(8.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_KA ~ 0.20
  sigma PROP ~ 0.05
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP)
[scaling]
  obs_scale = 1000 / V
";
    let model = parse_model_string(src).expect("scaling model parses");
    let (scale_fn, prog) = match &model.scaling {
        ScalingSpec::ExpressionScale {
            scale_fn,
            deriv: Some(p),
        } => (scale_fn, p),
        other => panic!("expected ExpressionScale with deriv, got {:?}", other),
    };
    assert_eq!(prog.n_theta_axis(), 3);
    assert_eq!(prog.n_eta_axis(), 3);
    assert_eq!(prog.n_axes(), 6);

    const M: usize = 6; // n_theta(3) + n_eta(3)
    let theta = vec![0.13, 8.0, 1.0];
    let eta = vec![0.1, -0.05, 0.2];
    let cov: HashMap<String, f64> = HashMap::new();

    // Individual-parameter duals over (θ, η) from the same program the
    // provider uses, mapped to the scale program's var slots.
    let ipp = model
        .indiv_param_partials
        .indiv_param_program
        .as_ref()
        .expect("indiv param program");
    let pk_duals = ipp.eval_param_duals::<M>(&theta, &eta, &cov);
    let pk_slots = ipp.pk_slots();
    let mut slot_dual: HashMap<usize, Dual2<M>> = HashMap::new();
    for (i, &s) in pk_slots.iter().enumerate() {
        slot_dual.insert(s, pk_duals[i]);
    }
    let pk_vals = (model.pk_param_fn)(&theta, &eta, &cov, 0.0);
    let var_duals: Vec<Dual2<M>> = prog
        .var_to_pk_slot()
        .iter()
        .map(|&s| {
            slot_dual
                .get(&s)
                .copied()
                .unwrap_or_else(|| Dual2::constant(pk_vals.values[s]))
        })
        .collect();
    let scale = prog.eval_scale_dual::<M>(&theta, &eta, &cov, &var_duals);

    // Reference: scale as a function of (θ, η) through pk_param_fn.
    let g = |th: &[f64], et: &[f64]| -> f64 {
        let pk = (model.pk_param_fn)(th, et, &cov, 0.0);
        scale_fn(th, et, &cov, &pk)
    };
    approx::assert_relative_eq!(scale.value, g(&theta, &eta), max_relative = 1e-12);
    let h = 1e-6;
    for m in 0..3 {
        let mut tp = theta.clone();
        tp[m] += h;
        let mut tm = theta.clone();
        tm[m] -= h;
        let fd = (g(&tp, &eta) - g(&tm, &eta)) / (2.0 * h);
        approx::assert_relative_eq!(scale.grad[m], fd, max_relative = 1e-5, epsilon = 1e-9);
    }
    for k in 0..3 {
        let mut ep = eta.clone();
        ep[k] += h;
        let mut em = eta.clone();
        em[k] -= h;
        let fd = (g(&theta, &ep) - g(&theta, &em)) / (2.0 * h);
        approx::assert_relative_eq!(scale.grad[3 + k], fd, max_relative = 1e-5, epsilon = 1e-9);
    }
}

#[test]
fn test_parse_scaling_expression_uses_covariate() {
    let src = analytical_model_with_scaling(Some("  obs_scale = WT / 70\n"));
    let model = parse_model_string(&src).expect("covariate scaling parses");
    match model.scaling {
        ScalingSpec::ExpressionScale { ref scale_fn, .. } => {
            let theta = vec![1.0, 50.0];
            let eta = vec![0.0];
            let mut cov = HashMap::new();
            cov.insert("WT".to_string(), 84.0);
            let pk = PkParams::default();
            let s = scale_fn(&theta, &eta, &cov, &pk);
            assert!((s - 1.2).abs() < 1e-12, "expected 1.2, got {}", s);
        }
        other => panic!("expected ExpressionScale, got {:?}", other),
    }
}

#[test]
fn test_parse_scaling_expression_uses_indiv_param() {
    // Phase 1.5: `V` is an individual parameter — must resolve via the
    // PK slot in the pk_params snapshot passed to scale_fn. The
    // `analytical_model_with_scaling` template defines `V = TVV` (an
    // unattenuated theta passthrough), so V's runtime value equals
    // theta[1] = 50, and `obs_scale = 1000 / V` evaluates to 20.
    let src = analytical_model_with_scaling(Some("  obs_scale = 1000 / V\n"));
    let model = parse_model_string(&src).expect("indiv-param ref in obs_scale parses");
    match model.scaling {
        ScalingSpec::ExpressionScale { ref scale_fn, .. } => {
            let theta = vec![1.0, 50.0]; // [TVCL, TVV]
            let eta = vec![0.0];
            let cov = HashMap::new();
            // Mimic what apply_scaling does at runtime: evaluate pk_param_fn
            // with the subject's theta/eta/covariates to materialize V.
            let pk = (model.pk_param_fn)(&theta, &eta, &cov, 0.0);
            let s = scale_fn(&theta, &eta, &cov, &pk);
            assert!(
                (s - 20.0).abs() < 1e-12,
                "expected 20.0 (= 1000/50), got {}",
                s
            );
        }
        other => panic!("expected ExpressionScale, got {:?}", other),
    }
}

/// #712: `obs_scale` referencing the same individual parameter bound to the
/// `pk <model>(...)` block's `v` role is a supported feature (see the test
/// above) but also the signature of a common mistake (a leftover `obs_scale
/// = V` copied from an `ode(...)` translation, where it's required but on a
/// `pk` block double-divides). It must still parse — warn, not error.
#[test]
fn test_obs_scale_referencing_pk_volume_warns_not_errors() {
    let src = analytical_model_with_scaling(Some("  obs_scale = V\n"));
    let model = parse_model_string(&src).expect("must still parse (warning, not error)");
    assert!(
        model
            .parse_warnings
            .iter()
            .any(|w| w.contains("obs_scale") && w.contains("volume parameter")),
        "expected a volume-collision warning, got {:?}",
        model.parse_warnings
    );
}

/// A pure unit-conversion constant (Form A) never references any individual
/// parameter, so it must never trigger the #712 volume-collision warning.
#[test]
fn test_obs_scale_scalar_constant_on_pk_block_does_not_warn() {
    let src = analytical_model_with_scaling(Some("  obs_scale = 1000\n"));
    let model = parse_model_string(&src).expect("scalar obs_scale parses");
    assert!(
        !model
            .parse_warnings
            .iter()
            .any(|w| w.contains("volume parameter")),
        "scalar obs_scale must not warn, got {:?}",
        model.parse_warnings
    );
}

/// An expression referencing an individual parameter that is *not* the
/// `pk` block's volume role (here `CL`, not `V`) is not the double-scaling
/// pattern #712 targets — no warning.
#[test]
fn test_obs_scale_referencing_non_volume_indiv_param_does_not_warn() {
    let src = analytical_model_with_scaling(Some("  obs_scale = CL / 10\n"));
    let model = parse_model_string(&src).expect("CL-referencing obs_scale parses");
    assert!(
        !model
            .parse_warnings
            .iter()
            .any(|w| w.contains("volume parameter")),
        "obs_scale referencing a non-volume indiv param must not warn, got {:?}",
        model.parse_warnings
    );
}

/// On an `ode(...)` model there's no built-in concentration divisor to
/// collide with — `obs_scale = V` is the documented, required way to
/// convert raw compartment amount to concentration, not a mistake pattern.
/// Must never warn.
#[test]
fn test_obs_scale_referencing_volume_on_ode_model_does_not_warn() {
    let src = ode_model_with_scaling(
        "ode(obs_cmt=central, states=[depot, central])",
        Some("  obs_scale = V\n"),
    );
    let model = parse_model_string(&src).expect("ODE obs_scale = V parses");
    assert!(
        !model
            .parse_warnings
            .iter()
            .any(|w| w.contains("volume parameter")),
        "ODE obs_scale = V must not warn, got {:?}",
        model.parse_warnings
    );
}

#[test]
fn test_parse_scaling_y_form_c_on_analytical_accepted() {
    // #650: a full Form C output readout is now accepted on analytical PK
    // models; it replaces the built-in concentration and leaves `scaling` at
    // `None` (the readout, not a divisor, carries the output map).
    let src = analytical_model_with_scaling(Some("  y = central / V\n"));
    let model = parse_model_string(&src).expect("analytic Form C readout must parse");
    assert!(
        model.analytic_readout.is_some(),
        "analytic Form C must populate analytic_readout"
    );
    assert!(matches!(model.scaling, ScalingSpec::None));
}

#[test]
fn test_parse_scaling_y_form_c_analytical_rejects_peripheral() {
    // A peripheral (or other no-closed-form) compartment amount is out of
    // scope for an analytic readout — reject with a pointer to an ODE model.
    let src = analytical_model_with_scaling(Some("  y = central / V + periph\n"));
    let err = parse_model_string(&src).expect_err("peripheral ref must be rejected");
    assert!(
        err.contains("periph") && err.contains("ODE model"),
        "expected peripheral-out-of-scope error, got: {}",
        err
    );
}

#[test]
fn test_parse_scaling_y_and_obs_scale_mix_rejected_on_analytical() {
    let src = analytical_model_with_scaling(Some("  y = central / V\n  obs_scale = 1000\n"));
    let err = parse_model_string(&src).expect_err("y + obs_scale mix must be rejected");
    assert!(
        err.contains("cannot combine") && err.contains("obs_scale"),
        "expected mix-rejection error, got: {}",
        err
    );
}

#[test]
fn test_parse_scaling_unknown_key_errors() {
    // Since #1030 any key other than `obs_scale`/`y` is a *named intermediate*, so
    // the old blanket "unknown key" rejection is gone. The protection it provided —
    // catching a misspelt `obs_scale`, which would otherwise silently disable
    // scaling — is re-established from the other side: an intermediate no entry
    // reads is rejected, and the error names both readings.
    let src = analytical_model_with_scaling(Some("  foo = 1000\n"));
    let err = parse_model_string(&src).expect_err("an unread scaling key must be rejected");
    assert!(
        err.contains("foo") && err.contains("never used") && err.contains("misspelled"),
        "expected unused-intermediate error, got: {}",
        err
    );
}

#[test]
fn test_parse_scaling_unknown_key_with_cmt_subscript_errors() {
    // `[CMT=N]` belongs to `obs_scale`/`y` only — an intermediate takes no
    // subscript, so this stays an unknown-key error rather than becoming a
    // legal-but-unread binding (#1030).
    let src = analytical_model_with_scaling(Some("  foo[CMT=1] = 1000\n"));
    let err = parse_model_string(&src).expect_err("subscripted unknown key must be rejected");
    assert!(
        err.contains("unknown key") && err.contains("foo"),
        "expected unknown-key error, got: {}",
        err
    );
}

// ── [scaling] named intermediates (#1030) ───────────────────────────────────

/// The headline case: a readout written with named intermediates must compile to
/// the *same* function as the hand-expanded one-liner it replaces. That is the
/// whole contract — intermediates are inlined into the AST, so nothing downstream
/// can tell the two apart.
#[test]
fn test_scaling_intermediates_match_hand_expanded_readout() {
    let with_names = ode_model_with_scaling(
        "ode(states=[depot, central])",
        Some(
            "  ACR20    = central / V\n\
             \x20 ACR20SAT = max(ACR20, 0.01)\n\
             \x20 y        = log(ACR20SAT / (1 - ACR20SAT))\n",
        ),
    );
    // What the block had to say before #1030: the same sub-expression four times,
    // guarded with the inline conditional that `max` desugars to.
    let hand = ode_model_with_scaling(
        "ode(states=[depot, central])",
        Some(
            "  y = log((if (central / V > 0.01) central / V else 0.01) \
             / (1 - (if (central / V > 0.01) central / V else 0.01)))\n",
        ),
    );

    let readout = |src: &str| -> crate::ode::OdeOutputFn {
        let model = parse_model_string(src).expect("model parses");
        let ode = model.ode_spec.expect("ODE spec present");
        match ode.readout {
            crate::ode::OdeReadout::Single(f) => f,
            _ => panic!("Form C must set OdeReadout::Single"),
        }
    };
    let f_named = readout(&with_names);
    let f_hand = readout(&hand);

    let cov = HashMap::new();
    let mut pk = vec![0.0f64; crate::types::MAX_PK_PARAMS];
    pk[0] = 1.0; // CL
    pk[1] = 50.0; // V
    pk[2] = 1.0; // KA
                 // Above the clamp, below it, and far above — `max` uses `>=` where the hand
                 // form uses `>`, so the two agree everywhere except exactly at the bound.
    for central in [0.0, 0.05, 20.0, 100.0] {
        let state = vec![0.0, central];
        let a = f_named(&state, &pk, &[], &[], &cov);
        let b = f_hand(&state, &pk, &[], &[], &cov);
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "named-intermediate readout must equal the hand-expanded one at \
             central={central} (named={a}, hand={b})"
        );
    }
}

/// `min(a, b)` / `max(a, b)` compose into the published `[0.01, 0.99]` clamp, and
/// the clamp actually bites on both sides.
#[test]
fn test_scaling_intermediates_two_sided_clamp() {
    let src = ode_model_with_scaling(
        "ode(states=[depot, central])",
        Some(
            "  RAW = central / V\n\
             \x20 SAT = min(max(RAW, 0.01), 0.99)\n\
             \x20 y   = SAT\n",
        ),
    );
    let model = parse_model_string(&src).expect("model parses");
    let ode = model.ode_spec.expect("ODE spec present");
    let f = match ode.readout {
        crate::ode::OdeReadout::Single(f) => f,
        _ => panic!("Form C must set OdeReadout::Single"),
    };
    let cov = HashMap::new();
    let mut pk = vec![0.0f64; crate::types::MAX_PK_PARAMS];
    pk[1] = 100.0; // V — so y = central / 100
    for (central, want) in [(0.0, 0.01), (5.0, 0.05), (500.0, 0.99)] {
        let y = f(&vec![0.0, central], &pk, &[], &[], &cov);
        assert!(
            (y - want).abs() < 1e-12,
            "central={central} → expected {want}, got {y}"
        );
    }
}

/// A covariate reachable only through an intermediate is still a required data
/// column — intermediates participate in the #1028 "defined" set, as the issue
/// asked, because they are inlined before the covariate scan runs.
#[test]
fn test_scaling_intermediate_covariate_is_registered() {
    let src = ode_model_with_scaling(
        "ode(states=[depot, central])",
        Some("  W = sqrt(NARM)\n  y = central / V * W\n"),
    );
    let model = parse_model_string(&src).expect("model parses");
    assert!(
        model.referenced_covariates.iter().any(|c| c == "NARM"),
        "a covariate reached through an intermediate must be a required column, got {:?}",
        model.referenced_covariates
    );
}

/// Intermediates work on the `obs_scale` (Form B) half of the block too.
#[test]
fn test_scaling_intermediate_in_obs_scale() {
    let src = analytical_model_with_scaling(Some("  K = MW * 1000\n  obs_scale = K\n"));
    let model = parse_model_string(&src).expect("model parses");
    assert!(
        matches!(
            model.scaling,
            crate::types::ScalingSpec::ExpressionScale { .. }
        ),
        "obs_scale built from an intermediate must be a Form B expression scale"
    );
    assert!(
        model.referenced_covariates.iter().any(|c| c == "MW"),
        "obs_scale covariate reached through an intermediate must be registered, got {:?}",
        model.referenced_covariates
    );
}

/// Order is the same rule `[individual_parameters]` already enforces: define
/// above, use below. That makes a reference cycle unrepresentable rather than a
/// recursion guard to get right.
#[test]
fn test_scaling_intermediate_forward_reference_rejected() {
    let src = analytical_model_with_scaling(Some("  A = B * 2\n  B = 3\n  y = A\n"));
    let err = parse_model_string(&src).expect_err("forward reference must be rejected");
    assert!(
        err.contains("declared on or below"),
        "expected forward-reference error, got: {}",
        err
    );
}

#[test]
fn test_scaling_intermediate_self_reference_rejected() {
    let src = analytical_model_with_scaling(Some("  A = A + 1\n  y = A\n"));
    let err = parse_model_string(&src).expect_err("self reference must be rejected");
    assert!(
        err.contains("declared on or below"),
        "expected self-reference error, got: {}",
        err
    );
}

#[test]
fn test_scaling_intermediate_duplicate_rejected() {
    let src = analytical_model_with_scaling(Some("  A = 2\n  A = 3\n  y = A\n"));
    let err = parse_model_string(&src).expect_err("duplicate intermediate must be rejected");
    assert!(
        err.contains("duplicate named intermediate"),
        "expected duplicate error, got: {}",
        err
    );
}

/// A binding named after something already in scope would never be read — the
/// expression parser binds θ / η / individual parameters / states first — so it is
/// rejected rather than left silently dead.
#[test]
fn test_scaling_intermediate_shadowing_rejected() {
    for (name, what) in [
        ("V", "an individual parameter"),
        ("TVCL", "a theta"),
        ("ETA_CL", "an eta"),
    ] {
        let src = analytical_model_with_scaling(Some(&format!("  {name} = 2\n  y = {name}\n")));
        let err = parse_model_string(&src)
            .expect_err("a name already in scope must be rejected as an intermediate");
        assert!(
            err.contains(what) && err.contains(name),
            "expected shadowing error naming {what}, got: {}",
            err
        );
    }
}

/// The three pre-scans that run *before* `parse_scaling_block` — the #486 direct-θ/η
/// desugaring and the #650 readout slot allocator — must see through intermediates
/// too, or a readout that was analytic when written inline would silently drop to
/// finite-difference sensitivities once it was named.
#[test]
fn test_scaling_intermediate_keeps_the_analytic_readout_path() {
    let model_src = |scaling: &str| {
        format!(
            r#"
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(50.0, 0.1, 500.0)
  theta TVBMAX(2.0, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.05 (sd)

[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  BMAX = TVBMAX

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)

[scaling]
{scaling}
"#
        )
    };
    // A non-structural individual parameter (`BMAX`) reached only through an
    // intermediate still earns a differentiable PK slot; and a bare theta inside an
    // intermediate is still desugared into a synthetic readout parameter.
    for scaling in [
        "  C   = central / V\n  SIG = BMAX * C\n  y   = SIG\n",
        "  G = TVCL * 2\n  y = central / V + G\n",
    ] {
        let model = parse_model_string(&model_src(scaling)).expect("model parses");
        assert!(
            !model
                .parse_warnings
                .iter()
                .any(|w| w.contains("finite-difference")),
            "readout `{scaling}` must stay on the analytic path, warnings: {:?}",
            model.parse_warnings
        );
    }
}

/// `obs_scale` is a subject-static divisor, so a `TIME` reference in it is
/// rejected (#1028). Routing that reference through an intermediate must not
/// launder it — the inlining happens before the check.
#[test]
fn test_scaling_intermediate_cannot_smuggle_time_into_obs_scale() {
    let src = analytical_model_with_scaling(Some("  K = 1 + TIME\n  obs_scale = K\n"));
    let err = parse_model_string(&src).expect_err("TIME in obs_scale must be rejected");
    assert!(
        err.contains("subject-static divisor"),
        "expected the obs_scale TIME rejection, got: {}",
        err
    );
}

#[test]
fn test_scaling_intermediate_reserved_name_rejected() {
    let src = analytical_model_with_scaling(Some("  TIME = 2\n  y = TIME\n"));
    let err = parse_model_string(&src).expect_err("`TIME` must be rejected as an intermediate");
    assert!(
        err.contains("model-time built-in"),
        "expected reserved-name error, got: {}",
        err
    );
}

/// `TAD` / `TAFD` / `MACHEPS` are supplied by the evaluator, so a binding under one
/// of those names is either dead or a silent override. Rejected like `TIME`.
#[test]
fn test_scaling_intermediate_eval_builtin_name_rejected() {
    for name in ["TAD", "TAFD", "MACHEPS"] {
        let src =
            analytical_model_with_scaling(Some(&format!("  {name} = 2\n  obs_scale = {name}\n")));
        let err = parse_model_string(&src)
            .expect_err("an eval-time built-in must be rejected as an intermediate");
        assert!(
            err.contains("eval-time built-in") && err.contains(name),
            "expected built-in shadowing error for {name}, got: {}",
            err
        );
    }
}

/// A declared covariate clashes the *other* way round from a θ/η/state: an
/// unresolved `[scaling]` identifier parses as `Covariate(name)`, which the
/// substituter does rewrite — so an intermediate named after a data column would
/// silently shadow it and drop it from the required-column set. Rejected too.
#[test]
fn test_scaling_intermediate_covariate_shadowing_rejected() {
    let src = "\
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(50.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.05 (sd)

[covariates]
  WT continuous

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)

[scaling]
  WT = 70
  obs_scale = WT
";
    let err = parse_model_string(src)
        .expect_err("an intermediate named after a declared covariate must be rejected");
    assert!(
        err.contains("a declared covariate") && err.contains("WT"),
        "expected covariate shadowing error, got: {}",
        err
    );
}

/// Define-above-use-below is enforced on both sides: an *entry* may not read an
/// intermediate declared below it either, even though the substitution itself is
/// order-independent.
#[test]
fn test_scaling_entry_forward_reference_rejected() {
    let src = analytical_model_with_scaling(Some("  obs_scale = FOO * 2\n  FOO = 3\n"));
    let err = parse_model_string(&src)
        .expect_err("an entry reading an intermediate declared below it must be rejected");
    assert!(
        err.contains("declared on or below this line") && err.contains("FOO"),
        "expected entry forward-reference error, got: {}",
        err
    );
}

/// The dead-binding guard scans tokens while the substitution walks the AST, so the
/// two must agree on what counts as a *read*. An identifier followed by `(` parses
/// as a function call, never a `Variable`/`Covariate` leaf — so the intermediate is
/// not inlined, and must be reported rather than silently vanishing.
#[test]
fn test_scaling_intermediate_in_unreachable_position_is_reported() {
    let src = analytical_model_with_scaling(Some("  FOO = 2\n  obs_scale = FOO(V)\n"));
    let err = parse_model_string(&src)
        .expect_err("an intermediate the substituter cannot reach must be reported unused");
    assert!(
        err.contains("is never used") && err.contains("FOO"),
        "expected unused-intermediate error, got: {}",
        err
    );
}

// ── Two-argument min / max (#1030) ──────────────────────────────────────────

#[test]
fn test_min_max_two_args_in_individual_parameters() {
    let content = r#"
[parameters]
  theta TVCL(2.0, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = max(TVCL * WT / 70, 1.0)
  V  = min(TVV, 20.0)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let parsed = parse_full_model(content).unwrap();
    let theta = vec![2.0, 10.0];
    let eta = vec![0.0];
    let mut covs = HashMap::new();

    covs.insert("WT".to_string(), 70.0);
    let p = (parsed.model.pk_param_fn)(&theta, &eta, &covs, 0.0);
    assert!(
        (p.values[0] - 2.0).abs() < 1e-12,
        "WT=70 → CL = max(2, 1) = 2"
    );
    assert!((p.values[1] - 10.0).abs() < 1e-12, "V = min(10, 20) = 10");

    // Below the floor: the clamp bites.
    covs.insert("WT".to_string(), 7.0);
    let p = (parsed.model.pk_param_fn)(&theta, &eta, &covs, 0.0);
    assert!(
        (p.values[0] - 1.0).abs() < 1e-12,
        "WT=7 → CL = max(0.2, 1) = 1, got {}",
        p.values[0]
    );
}

/// The arity diagnostic the issue called out: a one-argument `max` used to be
/// silently the identity, and a stray comma in a one-argument function reported a
/// missing `)`, sending the reader after a bracket bug that does not exist.
#[test]
fn test_min_max_arity_diagnostics() {
    let one_arg = analytical_model_with_scaling(Some("  y = max(central)\n"));
    let err = parse_model_string(&one_arg).expect_err("one-argument max must be rejected");
    assert!(
        err.contains("two arguments") && err.contains("max(a, b)"),
        "expected an arity error, got: {}",
        err
    );

    let extra_arg = analytical_model_with_scaling(Some("  y = exp(central, 2)\n"));
    let err = parse_model_string(&extra_arg).expect_err("two-argument exp must be rejected");
    assert!(
        err.contains("takes 1 argument") && err.contains("min(a, b)"),
        "expected an arity error naming the two-argument functions, got: {}",
        err
    );
}

// ── Three-argument clamp (#1092) ────────────────────────────────────────────

/// Parse a single expression with `X` / `Y` bound to variable slots 0 and 1, so
/// the bytecode and `Dual2` helpers above can drive whatever the parser built.
fn parse_indexed_expr(src: &str) -> Expression {
    let toks = tokenize(src).expect("tokenizes");
    let defined = vec!["X".to_string(), "Y".to_string()];
    let ctx = ParseCtx::new(&[], &[], &defined);
    let (mut expr, p) = parse_add_sub(&toks, 0, ctx).expect("parses");
    assert_eq!(p, toks.len(), "trailing tokens in `{src}`");
    let var_idx: HashMap<String, usize> = [("X".to_string(), 0), ("Y".to_string(), 1)]
        .into_iter()
        .collect();
    resolve_expr_indices(&mut expr, &var_idx, &HashMap::new());
    expr
}

/// `clamp(x, lo, hi)` is defined as the nested form it replaces, so on every
/// finite `x` the two must agree — including *on* both bounds, where the branch
/// taken decides which of two equal values is returned. (`NaN` is the one input
/// where they deliberately part; see
/// `clamp_propagates_nan_where_nested_pins_to_lo`.)
#[test]
fn clamp_matches_nested_min_max_on_every_finite_x() {
    let clamped = parse_indexed_expr("clamp(X, 0.01, 0.99)");
    let nested = parse_indexed_expr("min(max(X, 0.01), 0.99)");
    let nn: Vec<Vec<f64>> = Vec::new();
    for x in [-1.0, 0.0, 0.005, 0.01, 0.5, 0.99, 1.0, 2.0] {
        let vars = [x, 0.0];
        let a = eval_expression_indexed(&clamped, &[], &[], &[], &vars, &nn);
        let b = eval_expression_indexed(&nested, &[], &[], &[], &vars, &nn);
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "clamp != min(max(..)) at x = {x}: {a} vs {b}"
        );
        assert_eq!(a, x.clamp(0.01, 0.99), "at x = {x}");
    }
}

/// The one input where `clamp` and `min(max(..))` part company, pinned so the
/// divergence stays a decision rather than an accident. `NaN` fails both
/// `x <= lo` and `x >= hi`, so `clamp` falls through to `x` and propagates it;
/// the nested form's `max` sees `x >= lo` false and returns the bound, and the
/// outer `min` keeps it — so a `NaN` readout comes back silently pinned to `lo`.
/// Propagating is what we want: a `NaN` reaches the OFV, where it is caught.
#[test]
fn clamp_propagates_nan_where_nested_pins_to_lo() {
    let clamped = parse_indexed_expr("clamp(X, 0.01, 0.99)");
    let nested = parse_indexed_expr("min(max(X, 0.01), 0.99)");
    let nn: Vec<Vec<f64>> = Vec::new();
    let vars = [f64::NAN, 0.0];
    assert!(
        eval_expression_indexed(&clamped, &[], &[], &[], &vars, &nn).is_nan(),
        "clamp must propagate a NaN input"
    );
    assert_eq!(
        eval_expression_indexed(&nested, &[], &[], &[], &vars, &nn),
        0.01,
        "the nested form pins a NaN to `lo` — the behaviour clamp does not copy"
    );
    // The bytecode VM takes the same branches as the AST evaluator.
    let bc = compile_bytecode(&clamped);
    let mut stack: Vec<f64> = Vec::new();
    assert!(
        eval_bytecode_g::<f64>(&bc, &[], &[], &[], &vars, &nn, &mut stack).is_nan(),
        "the compiled clamp must propagate a NaN too"
    );
}

/// The desugared tree compiles: the bytecode VM and the AST evaluator must agree
/// bit-for-bit in each of the three regimes and on both bounds.
#[test]
fn clamp_bytecode_matches_ast() {
    let clamped = parse_indexed_expr("clamp(X, 0.01, 0.99)");
    for x in [-1.0, 0.005, 0.01, 0.5, 0.99, 2.0] {
        bc_vs_ast(clamped.clone(), &[x, 0.0], &[], &[], &[]);
    }
    // Variable bounds are ordinary expressions, not literals-only.
    let var_bound = parse_indexed_expr("clamp(X, Y, 2 * Y)");
    for x in [-1.0, 0.5, 1.0, 5.0] {
        bc_vs_ast(var_bound.clone(), &[x, 0.75], &[], &[], &[]);
    }
}

/// Derivatives across both bounds: flat (∂/∂x = 0) outside, pass-through
/// (∂/∂x = 1) inside, and the bound's own derivative when the bound is a
/// variable. Checked against central FD of the `f64` evaluator, with the probe
/// steps kept clear of the kinks.
#[test]
fn clamp_dual2_matches_fd_in_every_regime() {
    let expr = parse_indexed_expr("clamp(X, 0.2, 0.8) * Y");
    for x in [0.05, 0.5, 0.95] {
        bc_g_dual2_fd(&expr, x, 1.5, 1e-6, 1e-3);
    }
    // A variable lower bound: below it, the value *is* `Y`, so the gradient
    // moves off `X` and onto `Y`.
    let var_lo = parse_indexed_expr("clamp(X, Y, 0.9)");
    bc_g_dual2_fd(&var_lo, 0.1, 0.3, 1e-6, 1e-3);
    bc_g_dual2_fd(&var_lo, 0.5, 0.3, 1e-6, 1e-3);
}

/// The convention at the bounds themselves, where FD cannot speak: `x <= lo` and
/// `x >= hi` take the *bound* branch, so the derivative is 0 on both bounds and 1
/// strictly between them. Pinned so a later `<`/`<=` edit is a test failure.
#[test]
fn clamp_derivative_convention_on_the_bounds() {
    use crate::sens::dual2::Dual2;
    let bc = compile_bytecode(&parse_indexed_expr("clamp(X, 0.2, 0.8)"));
    let nn: Vec<Vec<f64>> = Vec::new();
    let grad_at = |x: f64| {
        let vd = [Dual2::<2>::var(x, 0), Dual2::<2>::var(0.0, 1)];
        let mut s: Vec<Dual2<2>> = Vec::new();
        eval_bytecode_g::<Dual2<2>>(&bc, &[], &[], &[], &vd, &nn, &mut s).grad[0]
    };
    assert_eq!(
        grad_at(0.2),
        0.0,
        "on the lower bound the `lo` branch is taken"
    );
    assert_eq!(
        grad_at(0.8),
        0.0,
        "on the upper bound the `hi` branch is taken"
    );
    assert_eq!(grad_at(0.5), 1.0, "strictly inside, clamp is the identity");
    assert_eq!(grad_at(0.1), 0.0, "below the lower bound");
    assert_eq!(grad_at(0.9), 0.0, "above the upper bound");
}

/// End to end through a real model: the clamp bites at both bounds and passes
/// through in between, after index resolution and bytecode compilation.
#[test]
fn test_clamp_in_individual_parameters() {
    let content = r#"
[parameters]
  theta TVCL(2.0, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = clamp(TVCL * WT / 70, 1.0, 3.0)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let parsed = parse_full_model(content).unwrap();
    let theta = vec![2.0, 10.0];
    let eta = vec![0.0];
    let mut covs = HashMap::new();

    let mut cl_at = |wt: f64, covs: &mut HashMap<String, f64>| {
        covs.insert("WT".to_string(), wt);
        (parsed.model.pk_param_fn)(&theta, &eta, covs, 0.0).values[0]
    };
    assert!(
        (cl_at(70.0, &mut covs) - 2.0).abs() < 1e-12,
        "inside: CL = 2"
    );
    assert!((cl_at(7.0, &mut covs) - 1.0).abs() < 1e-12, "floor bites");
    assert!(
        (cl_at(700.0, &mut covs) - 3.0).abs() < 1e-12,
        "ceiling bites"
    );
}

/// `clamp` with one, two or four arguments must be a hard error naming the
/// three-argument form. The one-argument case is the trap the feature invites:
/// an unknown `name(arg)` used to become `UnaryFn`, which every consumer
/// evaluates as the identity — a no-op that fits, converges and gives wrong
/// numbers.
#[test]
fn test_clamp_arity_diagnostics() {
    for bad in [
        "  y = clamp(central)\n",
        "  y = clamp(central, 0.1)\n",
        "  y = clamp(central, 0.1, 0.9, 2)\n",
    ] {
        let src = analytical_model_with_scaling(Some(bad));
        let err = parse_model_string(&src)
            .err()
            .unwrap_or_else(|| panic!("`{}` must be rejected", bad.trim()));
        assert!(
            err.contains("three arguments") && err.contains("clamp(x, lo, hi)"),
            "expected a clamp arity error for `{}`, got: {}",
            bad.trim(),
            err
        );
    }

    // The reverse mistake: a third argument to `min`/`max` points at `clamp`
    // rather than at a bracket bug.
    let src = analytical_model_with_scaling(Some("  y = min(central, 0.1, 0.9)\n"));
    let err = parse_model_string(&src).expect_err("three-argument min must be rejected");
    assert!(
        err.contains("clamp(x, lo, hi)"),
        "expected the three-argument min error to name clamp, got: {}",
        err
    );
}

/// `clamp(x, 0.9, 0.1)` would never return `x` at all — quietly handing back
/// `0.9` below the crossing and `0.1` above it. With both bounds literal that
/// can only be a typo, so it is a parse error; bounds that
/// are not both literals are left to run rather than paying an ordering test on
/// every evaluation.
#[test]
fn test_clamp_rejects_inverted_literal_bounds() {
    let src = analytical_model_with_scaling(Some("  y = clamp(central, 0.9, 0.1)\n"));
    let err = parse_model_string(&src).expect_err("inverted literal bounds must be rejected");
    assert!(
        err.contains("bounds inverted"),
        "expected the inverted-bounds error, got: {}",
        err
    );

    // Equal bounds are a degenerate but well-defined constant, not an error.
    let src = analytical_model_with_scaling(Some("  y = clamp(central, 0.5, 0.5)\n"));
    parse_model_string(&src).expect("equal bounds are legal");

    // A non-literal bound is not checked — no runtime ordering test.
    let src = analytical_model_with_scaling(Some("  y = clamp(central, V, 0.1)\n"));
    parse_model_string(&src).expect("non-literal bounds parse");
}

// ── Operator continuation lines (#1030) ─────────────────────────────────────

#[test]
fn test_join_continuation_lines() {
    let join = |lines: &[&str]| -> Vec<String> {
        super::join_continuation_lines(&lines.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    };
    // Leading operator continues the previous line...
    assert_eq!(
        join(&["A = B", "+ C", "- D", "E = F"]),
        vec!["A = B + C - D".to_string(), "E = F".to_string()]
    );
    // ...as does a trailing one.
    assert_eq!(
        join(&["A = B +", "C", "E = F"]),
        vec!["A = B + C".to_string(), "E = F".to_string()]
    );
    // A trailing `=` continues too, so `LEMAX =` on its own line works.
    assert_eq!(join(&["A =", "B * 2"]), vec!["A = B * 2".to_string()]);
    // A leading operator on the very first line has nothing to join to; leave it
    // for the statement parser to reject with its own message.
    assert_eq!(join(&["+ C"]), vec!["+ C".to_string()]);
    // Ordinary lines are untouched.
    assert_eq!(
        join(&["A = B", "C = D"]),
        vec!["A = B".to_string(), "C = D".to_string()]
    );
}

/// The additive-on-logit covariate model from the issue: one covariate per line,
/// which used to fail with ``Expected an assignment, … got Plus``.
#[test]
fn test_individual_parameters_operator_continuation_lines() {
    let content = r#"
[parameters]
  theta LEMAX0(0.5, -5.0, 5.0)
  theta OR_ABATA(1.5, 0.1, 10.0)
  theta OR_ADALI(2.0, 0.1, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_EMAX ~ 0.09
  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = LEMAX0
       + log(OR_ABATA) * ABATA
       + log(OR_ADALI) * ADALI
       + ETA_EMAX
  V  = TVV *
       2.0

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let parsed = parse_full_model(content).expect("continuation lines parse");
    let theta = vec![0.5, 1.5, 2.0, 10.0];
    let eta = vec![0.25];
    let mut covs = HashMap::new();
    covs.insert("ABATA".to_string(), 1.0);
    covs.insert("ADALI".to_string(), 0.0);
    let p = (parsed.model.pk_param_fn)(&theta, &eta, &covs, 0.0);
    let want = 0.5 + 1.5f64.ln() + 0.25;
    assert!(
        (p.values[0] - want).abs() < 1e-12,
        "expected {want}, got {}",
        p.values[0]
    );
    assert!((p.values[1] - 20.0).abs() < 1e-12, "trailing `*` continues");
}

/// `[scaling]` is scanned line by line rather than through the token stream, so
/// its continuation support is worth pinning separately.
#[test]
fn test_scaling_continuation_lines() {
    let src = ode_model_with_scaling(
        "ode(states=[depot, central])",
        Some("  y = central / V\n      * 2.0\n"),
    );
    let model = parse_model_string(&src).expect("continuation line parses in [scaling]");
    let ode = model.ode_spec.expect("ODE spec present");
    let f = match ode.readout {
        crate::ode::OdeReadout::Single(f) => f,
        _ => panic!("Form C must set OdeReadout::Single"),
    };
    let mut pk = vec![0.0f64; crate::types::MAX_PK_PARAMS];
    pk[1] = 50.0;
    let y = f(&vec![0.0, 100.0], &pk, &[], &[], &HashMap::new());
    assert!((y - 4.0).abs() < 1e-12, "expected 100/50*2 = 4, got {y}");
}

/// `[initial_conditions]` is line-oriented too (`init(NAME) = <expr>`, one per
/// line), so its continuation support is pinned on its own path — and the joined
/// expression must actually reach the init amount, not just parse.
#[test]
fn test_initial_conditions_continuation_lines() {
    let content = r#"
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[initial_conditions]
  init(central) = CONC0 * V
                  + 5.0

[error_model]
  DV ~ proportional(EPS)
"#;
    let model = parse_model_string(content).expect("continuation line parses in the init block");
    assert_eq!(model.analytical_init.len(), 1, "init block parsed");
    assert!(
        model.referenced_covariates.iter().any(|c| c == "CONC0"),
        "the continued line's covariate must still be a required column, got: {:?}",
        model.referenced_covariates
    );
    // CONC0 = 2, V = 10 → 2*10 + 5 = 25. The `+ 5.0` line has to be part of the
    // expression, not silently dropped.
    let mut covs = HashMap::new();
    covs.insert("CONC0".to_string(), 2.0);
    let theta = vec![1.0, 10.0];
    let eta = vec![0.0];
    let pk = (model.pk_param_fn)(&theta, &eta, &covs, 0.0);
    let amount = (model.analytical_init[0].amount_fn)(&theta, &eta, &covs, &pk);
    assert!(
        (amount - 25.0).abs() < 1e-12,
        "expected 2*10 + 5 = 25, got {amount}"
    );
}

#[test]
fn test_parse_scaling_y_form_c_on_ode() {
    // ODE form C: ode(states=...) without obs_cmt, and `y = central / V`
    // replaces the readout. Verifies the parse path completes and
    // output_fn evaluates correctly.
    let src = ode_model_with_scaling("ode(states=[depot, central])", Some("  y = central / V\n"));
    let model = parse_model_string(&src).expect("Form C parses");
    let ode = model.ode_spec.as_ref().expect("ODE spec present");
    let out_fn = match &ode.readout {
        crate::ode::OdeReadout::Single(f) => f,
        other => panic!(
            "Form C must set OdeReadout::Single, got {:?}",
            match other {
                crate::ode::OdeReadout::ObsCmt(i) => format!("ObsCmt({})", i),
                crate::ode::OdeReadout::Single(_) => "Single(..)".into(),
                crate::ode::OdeReadout::PerCmt(_) => "PerCmt(..)".into(),
            }
        ),
    };

    // ODE writes indiv params sequentially into pk_params_flat[0..n] in
    // declaration order: [CL, V, KA] -> pk[0..3]. State order: [depot,
    // central] -> state[0..2]. So y = central / V = state[1] / pk[1].
    let state = vec![0.0, 100.0]; // depot=0, central=100
    let mut pk = vec![0.0f64; crate::types::MAX_PK_PARAMS];
    pk[0] = 1.0; // CL
    pk[1] = 50.0; // V
    pk[2] = 1.0; // KA
    let cov = HashMap::new();
    let y = out_fn(&state, &pk, &[], &[], &cov);
    assert!((y - 2.0).abs() < 1e-12, "expected 100/50 = 2, got {}", y);
}

/// A covariate referenced only inside an `[initial_conditions]` expression
/// must be registered in `referenced_covariates` so it becomes a required
/// data column. Without this, a `[covariates]`-declared model never reads
/// the column and a miscased/absent init covariate silently evaluates to 0
/// (issue #765).
#[test]
fn test_initial_conditions_covariate_is_registered() {
    let content = r#"
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta TVKA(1.0, 0.01, 10.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  F  = 1.0

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, f=F)

[initial_conditions]
  init(central) = CONC0 * V

[error_model]
  DV ~ proportional(EPS)
"#;
    let model = parse_model_string(content).expect("model parses");
    assert_eq!(model.analytical_init.len(), 1, "init block parsed");
    assert!(
        model.referenced_covariates.iter().any(|c| c == "CONC0"),
        "init covariate CONC0 must be registered, got: {:?}",
        model.referenced_covariates
    );
}

/// An `[initial_conditions]` expression built only from parameters (no free
/// identifier) registers no covariate — `V` is an individual parameter, not
/// a data column, so it must not leak into `referenced_covariates` (#765).
#[test]
fn test_initial_conditions_param_only_registers_no_covariate() {
    let content = r#"
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta TVKA(1.0, 0.01, 10.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  F  = 1.0

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, f=F)

[initial_conditions]
  init(central) = 0.5 * V

[error_model]
  DV ~ proportional(EPS)
"#;
    let model = parse_model_string(content).expect("model parses");
    assert_eq!(model.analytical_init.len(), 1, "init block parsed");
    assert!(
        model.referenced_covariates.is_empty(),
        "a param-only init must register no covariate, got: {:?}",
        model.referenced_covariates
    );
}

#[test]
fn test_parse_scaling_y_requires_scaling_block_for_missing_obs_cmt() {
    // ODE without obs_cmt and no [scaling] y = ... must error.
    let src = ode_model_with_scaling("ode(states=[depot, central])", None);
    let err =
        parse_model_string(&src).expect_err("ODE without obs_cmt and without Form C must error");
    assert!(
        err.contains("obs_cmt") || err.contains("Form C"),
        "expected validation error, got: {}",
        err
    );
}

#[test]
fn test_parse_scaling_scalar_on_ode_keeps_obs_cmt() {
    // Form A on an ODE model should preserve obs_cmt and apply post-multiply.
    let src = ode_model_with_scaling(
        "ode(obs_cmt=central, states=[depot, central])",
        Some("  obs_scale = 1000\n"),
    );
    let model = parse_model_string(&src).expect("Form A on ODE parses");
    let ode = model.ode_spec.as_ref().expect("ODE spec present");
    assert!(matches!(ode.readout, crate::ode::OdeReadout::ObsCmt(1)));
    assert!(matches!(model.scaling, ScalingSpec::ScalarScale(k) if (k - 1000.0).abs() < 1e-12));
}

#[test]
fn test_parse_scaling_y_form_c_resolves_covariates() {
    // Regression for Copilot review: Form C parsed with ParseCtx::ode
    // silently turned `WT` into a `Variable` and returned 0.0. With the
    // fix (ParseCtx::new), unknown identifiers resolve as Covariate and
    // get looked up in the per-call covariate map.
    let src = ode_model_with_scaling(
        "ode(states=[depot, central])",
        Some("  y = central / V * WT\n"),
    );
    let model = parse_model_string(&src).expect("Form C with covariate parses");
    assert!(
        model.referenced_covariates.contains(&"WT".to_string()),
        "Form C covariates must be retained for data loading and TV pruning"
    );
    let ode = model.ode_spec.as_ref().unwrap();
    let out_fn = match &ode.readout {
        crate::ode::OdeReadout::Single(f) => f,
        _ => panic!("Form C must set OdeReadout::Single"),
    };

    let state = vec![0.0, 50.0]; // depot=0, central=50
    let mut pk = vec![0.0f64; crate::types::MAX_PK_PARAMS];
    pk[0] = 1.0; // CL
    pk[1] = 50.0; // V
    pk[2] = 1.0; // KA
    let mut cov = HashMap::new();
    cov.insert("WT".to_string(), 70.0);
    let y = out_fn(&state, &pk, &[], &[], &cov);
    // central/V * WT = 50/50 * 70 = 70. With the bug, WT was treated as
    // a Variable and read 0 from the empty vars map → y = 0.
    assert!(
        (y - 70.0).abs() < 1e-12,
        "expected 70.0 (covariate WT must resolve), got {}",
        y
    );
}

#[test]
fn test_parse_scaling_y_form_c_resolves_theta() {
    // Follow-up after Copilot's Fix #1: switching Form C's ParseCtx to
    // ParseCtx::ode silently zeroed covariate refs; Copilot suggested
    // ParseCtx::new(EMPTY, EMPTY, ...) which fixed covariates but
    // introduced the symmetric bug for theta refs (TVCL became
    // Covariate("TVCL") → 0.0). Phase 1.5 fix: pass theta_names/eta_names
    // into ParseCtx so identifiers resolve as Theta(i) / Eta(i) and
    // thread theta/eta through OdeOutputFn at runtime.
    let src = ode_model_with_scaling(
        "ode(states=[depot, central])",
        Some("  y = central / V * TVCL\n"),
    );
    let model = parse_model_string(&src).expect("Form C with theta ref parses");
    let ode = model.ode_spec.as_ref().unwrap();
    let out_fn = match &ode.readout {
        crate::ode::OdeReadout::Single(f) => f,
        _ => panic!("Form C must set OdeReadout::Single"),
    };

    let state = vec![0.0, 50.0];
    let theta = vec![3.0, 50.0, 1.0]; // [TVCL=3, TVV=50, TVKA=1]
    let eta = vec![0.0];
    let cov = HashMap::new();
    // The bare `TVCL` in the readout is desugared (#486) into a synthetic
    // individual parameter, so its value flows through the PK-parameter vector
    // `pk_param_fn` fills (the production readout call path) — not the bare
    // `theta` slice. Build `pk` the same way production does.
    let pk = (model.pk_param_fn)(&theta, &eta, &cov, 0.0);
    let y = out_fn(&state, &pk.values, &theta, &eta, &cov);
    // central/V * TVCL = 50/50 * 3 = 3.
    assert!(
        (y - 3.0).abs() < 1e-12,
        "expected 3.0 (TVCL must resolve to theta[0]=3), got {}",
        y
    );
}

/// #486: a Form-C readout referencing θ/η directly is desugared into synthetic
/// individual parameters (`__ferx_ro_*`), which makes the readout analytic
/// (`is_dual_evaluable`) without adding any new random effect, and the synthetic
/// names are suppressed from user-facing EBE output.
#[test]
fn test_form_c_direct_theta_eta_desugars_to_indiv_params() {
    let src = ode_model_with_scaling(
        "ode(states=[depot, central])",
        // Bare θ (`TVCL`) and bare η (`ETA_CL`) directly in the readout.
        Some("  y = central / V * TVCL + ETA_CL\n"),
    );
    let model = parse_model_string(&src).expect("direct θ/η readout parses");

    // Two synthetic individual parameters were appended (θ and η source).
    let synth: Vec<&String> = model
        .indiv_param_names
        .iter()
        .filter(|n| is_synthetic_readout_param(n))
        .collect();
    assert_eq!(synth.len(), 2, "one synthetic param per distinct θ/η ref");
    assert_eq!(model.indiv_param_names.len(), model.pk_indices.len());

    // The desugared readout carries no bare θ/η ops, so it is dual-evaluable.
    let ode = model.ode_spec.as_ref().unwrap();
    let prog = ode
        .readout_program
        .as_ref()
        .expect("uniform Form-C readout has a sensitivity program");
    assert!(
        prog.is_dual_evaluable(),
        "a θ/η readout must be analytic after desugaring (#486)"
    );

    // The `ETA_CL` reference reuses the existing random effect — no new omega.
    assert_eq!(
        model.n_eta, 1,
        "desugaring an η reference adds no random effect"
    );

    // The synthetic parameters are load-bearing (read by the readout), so the
    // "computed but never used" census must not flag them.
    assert!(
        !model
            .parse_warnings
            .iter()
            .any(|w| w.contains("computed but never used")),
        "synthetic readout params must not trip the unused-parameter census, got: {:?}",
        model.parse_warnings
    );
}

/// #486 headroom helper: `ode_free_slot_count` reports how many non-reserved PK
/// slots remain, reusing the real `ode_param_slots` assignment so it can never
/// drift from it.
#[test]
fn ode_free_slot_count_reports_headroom() {
    // MAX_PK_PARAMS slots minus the reserved ones (F, lagtime) are usable.
    let usable = MAX_PK_PARAMS - RESERVED_PK_SLOTS.len();
    assert_eq!(ode_free_slot_count(&[]), usable);
    // CL→0, V→1 are non-reserved canonical slots, leaving `usable - 2`.
    assert_eq!(
        ode_free_slot_count(&["CL".to_string(), "V".to_string()]),
        usable - 2
    );
    // `usable` non-canonical params exactly fill the layout → 0 free; one more
    // still 0 (`ode_param_slots` errors, and the headroom is reported as full).
    let full: Vec<String> = (0..usable).map(|i| format!("EXTRA{i}")).collect();
    assert_eq!(ode_free_slot_count(&full), 0);
    let over: Vec<String> = (0..usable + 1).map(|i| format!("EXTRA{i}")).collect();
    assert_eq!(ode_free_slot_count(&over), 0);
}

/// #486: the `__ferx_ro_` prefix is reserved for synthetic readout params, so a
/// user/generated parameter carrying it is rejected at parse time rather than
/// being silently misclassified as internal (and dropped from output).
#[test]
fn test_form_c_reserved_prefix_param_rejected() {
    let src = ode_model_with_scaling("ode(states=[depot, central])", Some("  y = central / V\n"))
        .replace("  KA = TVKA\n", "  KA = TVKA\n  __ferx_ro_x = TVKA\n");
    let err = parse_model_string(&src)
        .expect_err("a user parameter using the reserved __ferx_ro_ prefix must be rejected");
    assert!(
        err.contains("reserved") && err.contains("__ferx_ro_"),
        "expected a reserved-prefix error, got: {err}"
    );
}

/// #486 regression: a model that previously parsed (direct-θ readout served by the
/// FD fallback) must not start failing the PK-slot layout once desugaring is
/// available. When the synthetic param has no free slot, the readout falls back to
/// FD (its prior behaviour) with a warning, instead of erroring.
#[test]
fn test_form_c_direct_readout_falls_back_to_fd_when_slots_full() {
    // CL, V, KA take 3 of the (MAX_PK_PARAMS − reserved) usable slots; the
    // remaining are filled by EXTRA params → zero headroom. A direct-θ readout
    // would need one more.
    let usable = MAX_PK_PARAMS - RESERVED_PK_SLOTS.len();
    let n_extras = usable - 3;
    let extras: String = (0..n_extras)
        .map(|i| format!("  EXTRA{i} = TVV\n"))
        .collect();
    let src = format!(
        "\
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(50.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.05 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
{extras}
[structural_model]
  ode(states=[depot, central])

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) = KA * depot - CL/V * central

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method  = focei
  maxiter = 10
  gradient = fd

[scaling]
  y = central / V * TVCL
"
    );
    let model = parse_model_string(&src)
        .expect("a slot-full model with a direct-θ readout must still parse (FD fallback)");

    // No synthetic param was injected — desugaring stood down for lack of a slot.
    assert!(
        !model
            .indiv_param_names
            .iter()
            .any(|n| is_synthetic_readout_param(n)),
        "desugaring must not consume a slot it does not have"
    );
    // The fallback is surfaced, not silent.
    assert!(
        model
            .parse_warnings
            .iter()
            .any(|w| w.contains("finite-difference")),
        "the FD fallback must be warned about, got: {:?}",
        model.parse_warnings
    );
}

/// Issue #107: a Form C ODE output expression that references KAPPA_*
/// directly would be evaluated per-observation with kappa=0, so it is
/// rejected at parse time for IOV models.
fn iov_ode_model_with_y(y_expr: &str) -> String {
    format!(
        "\
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(50.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.05 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV
  KA = TVKA

[structural_model]
  ode(states=[depot, central])

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) = KA * depot - CL/V * central

[error_model]
  DV ~ proportional(PROP_ERR)

[scaling]
  {y_expr}

[fit_options]
  method = focei
  iov_column = OCC
  gradient = fd
"
    )
}

#[test]
fn ode_form_c_output_referencing_kappa_is_rejected_under_iov() {
    let src = iov_ode_model_with_y("y = central / V * exp(KAPPA_CL)");
    let err = parse_model_string(&src)
        .expect_err("Form C output referencing KAPPA_* must be rejected under IOV");
    assert!(
        err.contains("KAPPA") && err.contains("#107"),
        "unexpected error: {err}"
    );
}

#[test]
fn ode_form_c_output_without_kappa_ref_is_allowed_under_iov() {
    // The readout reads only the structural state/param; the occasion
    // dependence enters through CL in the ODE rhs, so this must parse.
    let src = iov_ode_model_with_y("y = central / V");
    parse_model_string(&src)
        .expect("Form C output without a direct KAPPA_* reference must parse under IOV");
}

#[test]
fn ode_form_c_output_referencing_kappa_in_condition_is_rejected_under_iov() {
    // KAPPA_* inside a conditional *condition* (not just a branch) must also
    // be rejected — the guard walks the condition tree (Copilot review #108).
    let src = iov_ode_model_with_y("y = if (KAPPA_CL > 0) central / V else central / V");
    let err = parse_model_string(&src)
        .expect_err("KAPPA_* in a Form C output condition must be rejected under IOV");
    assert!(
        err.contains("KAPPA") && err.contains("#107"),
        "unexpected error: {err}"
    );
}

#[test]
fn init_amount_referencing_kappa_is_rejected_under_iov() {
    // [initial_conditions] mirrors the Form C guard (#107/#521 review): the
    // init expression's eta scope is BSV-only, so a KAPPA_* reference would
    // silently evaluate to 0 and give a wrong baseline. Reject it instead.
    let src = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVC0(5.0, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL   = TVCL * exp(ETA_CL + KAPPA_CL)
  V    = TVV
  CONC0 = TVC0

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[initial_conditions]
  init(central) = CONC0 * V * exp(KAPPA_CL)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let err = parse_model_string(src)
        .expect_err("init expression referencing KAPPA_* must be rejected under IOV");
    assert!(
        err.contains("KAPPA") && err.contains("IOV"),
        "unexpected error: {err}"
    );
}

#[test]
fn init_amount_without_kappa_ref_is_allowed_under_iov() {
    // The baseline reads only BSV/structural params; it must parse even when
    // the model carries IOV elsewhere.
    let src = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVC0(5.0, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL   = TVCL * exp(ETA_CL + KAPPA_CL)
  V    = TVV
  CONC0 = TVC0

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[initial_conditions]
  init(central) = CONC0 * V

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let m = parse_model_string(src)
        .expect("init expression without a KAPPA_* reference must parse under IOV");
    assert_eq!(m.analytical_init.len(), 1);
}

#[test]
fn test_parse_scaling_rejected_on_sde_model() {
    // Regression for Copilot review: EKF p_obs / r_obs run in unscaled
    // observation space, so Forms A/B Phase 1 scaling on SDE models
    // would produce mis-scaled variance. Parser must reject the
    // combination entirely (Form C was already rejected).
    let src = format!(
        "\
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(50.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.05 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -CL/V * central

[diffusion]
  central ~ 0.01

[error_model]
  DV ~ proportional(PROP_ERR)

[scaling]
  obs_scale = 1000

[fit_options]
  method  = focei
  maxiter = 5
  gradient = fd
"
    );
    let err = parse_model_string(&src).expect_err("SDE + [scaling] must be rejected in Phase 1");
    assert!(
        err.to_lowercase().contains("sde") || err.contains("[diffusion]"),
        "expected SDE rejection error, got: {}",
        err
    );
}

#[test]
fn test_parse_scaling_expression_accepts_ad_gradient() {
    // Form B (`obs_scale = <expr>`) + `gradient = ad` parses (the retired `ad`
    // request is normalised away by `check_model_options`, not rejected at parse).
    // The live FOCE/FOCEI gradient applies the exact η/θ quotient rule to a
    // differentiable `obs_scale` (#486), so it is exact for both η-independent and
    // η-dependent scales within analytic scope — not the old frozen-scale approximation.
    let base = analytical_model_with_scaling(Some("  obs_scale = TVV / 10\n"));
    let src = base.replace("gradient = fd", "gradient = ad");
    parse_model_string(&src).expect("ExpressionScale + gradient = ad now parses");
}

// ── Phase 2: multi-analyte / per-CMT scaling ────────────────────────────

#[test]
fn test_parse_scaling_key_uniform() {
    let (base, cmt) = parse_scaling_key("obs_scale").unwrap();
    assert_eq!(base, "obs_scale");
    assert_eq!(cmt, None);
}

#[test]
fn test_parse_scaling_key_per_cmt() {
    let (base, cmt) = parse_scaling_key("obs_scale[CMT=2]").unwrap();
    assert_eq!(base, "obs_scale");
    assert_eq!(cmt, Some(2));

    let (base, cmt) = parse_scaling_key("y[CMT=1]").unwrap();
    assert_eq!(base, "y");
    assert_eq!(cmt, Some(1));
}

#[test]
fn test_parse_scaling_key_rejects_malformed() {
    assert!(parse_scaling_key("obs_scale[").is_err());
    assert!(parse_scaling_key("obs_scale[CMT=]").is_err());
    assert!(parse_scaling_key("obs_scale[CMT=abc]").is_err());
    assert!(parse_scaling_key("obs_scale[FOO=1]").is_err());
    assert!(parse_scaling_key("obs_scale[CMT=0]").is_err()); // 1-based
    assert!(parse_scaling_key("obs_scale[CMT=1]extra").is_err());
}

#[test]
fn test_parse_scaling_per_cmt_scalar() {
    // Use the analytical template but layer a per-CMT scaling block on
    // top. Even though one_cpt_iv only emits CMT=1 observations,
    // the parser doesn't validate coverage (that happens at fit time),
    // so this exercises the parse path cleanly.
    let mut src = String::from(
        "\
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(50.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.05 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method  = focei
  maxiter = 5
  gradient = fd

[scaling]
  obs_scale[CMT=1] = 1000
  obs_scale[CMT=2] = 1
",
    );
    let _ = &mut src; // silence unused-mut on platforms without mods
    let model = parse_model_string(&src).expect("per-CMT obs_scale parses");
    match &model.scaling {
        ScalingSpec::PerCmt(map) => {
            assert_eq!(map.len(), 2);
            assert!(
                matches!(map.get(&1), Some(ScalingSpec::ScalarScale(k)) if (*k - 1000.0).abs() < 1e-12)
            );
            assert!(
                matches!(map.get(&2), Some(ScalingSpec::ScalarScale(k)) if (*k - 1.0).abs() < 1e-12)
            );
        }
        other => panic!("expected PerCmt, got {:?}", other),
    }
}

#[test]
fn test_parse_scaling_per_cmt_mixed_forms() {
    // CMT=1 uses scalar, CMT=2 uses expression. Both are valid within
    // the same PerCmt map.
    let src = analytical_model_with_scaling(Some(
        "  obs_scale[CMT=1] = 1000\n  obs_scale[CMT=2] = TVV / 10\n",
    ));
    let model = parse_model_string(&src).expect("mixed PerCmt parses");
    match &model.scaling {
        ScalingSpec::PerCmt(map) => {
            assert_eq!(map.len(), 2);
            assert!(matches!(map.get(&1), Some(ScalingSpec::ScalarScale(_))));
            assert!(matches!(
                map.get(&2),
                Some(ScalingSpec::ExpressionScale { .. })
            ));
        }
        other => panic!("expected PerCmt, got {:?}", other),
    }
}

#[test]
fn test_parse_scaling_rejects_mixing_uniform_and_per_cmt() {
    let src = analytical_model_with_scaling(Some("  obs_scale = 1000\n  obs_scale[CMT=2] = 1\n"));
    let err =
        parse_model_string(&src).expect_err("mixing uniform + per-CMT obs_scale must be rejected");
    assert!(
        err.to_lowercase().contains("cannot mix"),
        "expected cannot-mix error, got: {}",
        err
    );
}

#[test]
fn test_parse_scaling_rejects_duplicate_cmt() {
    let src = analytical_model_with_scaling(Some(
        "  obs_scale[CMT=1] = 1000\n  obs_scale[CMT=1] = 2000\n",
    ));
    let err = parse_model_string(&src).expect_err("duplicate CMT key must be rejected");
    assert!(
        err.contains("duplicate") && err.contains("CMT=1"),
        "expected duplicate-CMT-key error, got: {}",
        err
    );
}

#[test]
fn test_parse_scaling_per_cmt_accepts_ad() {
    // Phase 2.5: per-CMT `obs_scale[CMT=N]` + `gradient = ad` is now
    // allowed. The AD path builds a per-observation scale array
    // dispatching the right per-CMT inner spec to each observation
    // via `inner_optimizer::build_scale_array_for_ad`.
    let base =
        analytical_model_with_scaling(Some("  obs_scale[CMT=1] = 1000\n  obs_scale[CMT=2] = 1\n"));
    let src = base.replace("gradient = fd", "gradient = ad");
    parse_model_string(&src).expect("per-CMT obs_scale + gradient = ad now parses");
}

// ── [scaling] undefined-identifier guard + TIME built-in (issue #1028) ──

/// Every identifier a `[scaling]` expression cannot bind to a theta / eta /
/// individual parameter / state is a **covariate**, and therefore a required data
/// column. The Form C `y` half has registered them since #540; the Form B
/// `obs_scale` half silently dropped them, so a typo (or a real covariate the data
/// didn't carry) resolved to the covariate map's `0.0` default and the divisive
/// scale turned every prediction into the `apply_scaling` NaN path — with nothing
/// in `referenced_covariates` for `check_covariates` to catch. Registering the name
/// is what makes `fit()` / `simulate()` / `predict()` refuse the model instead.
#[test]
fn scaling_obs_scale_registers_covariate_references() {
    let src = analytical_model_with_scaling(Some("  obs_scale = V / TOTALLY_UNDEFINED_NAME\n"));
    let model = parse_model_string(&src).expect("obs_scale expression parses");
    assert!(
        model
            .referenced_covariates
            .iter()
            .any(|c| c == "TOTALLY_UNDEFINED_NAME"),
        "an unbindable `obs_scale` identifier must become a required data column, \
         got {:?}",
        model.referenced_covariates
    );
}

/// The per-CMT `obs_scale[CMT=N]` form registers its covariates too — the
/// collection sits in `build_obs_scale_spec`, which both key forms route through.
#[test]
fn scaling_obs_scale_per_cmt_registers_covariate_references() {
    let src = analytical_model_with_scaling(Some(
        "  obs_scale[CMT=1] = SCALE_A\n  obs_scale[CMT=2] = SCALE_B\n",
    ));
    let model = parse_model_string(&src).expect("per-CMT obs_scale expression parses");
    for name in ["SCALE_A", "SCALE_B"] {
        assert!(
            model.referenced_covariates.iter().any(|c| c == name),
            "`{name}` must be registered, got {:?}",
            model.referenced_covariates
        );
    }
}

/// A bound name is *not* a covariate: `obs_scale = V` reads the individual
/// parameter, so nothing is added to the required-column set. Guards the new
/// collection against over-registering.
#[test]
fn scaling_obs_scale_indiv_param_is_not_a_covariate() {
    let src = analytical_model_with_scaling(Some("  obs_scale = V / 10\n"));
    let model = parse_model_string(&src).expect("obs_scale expression parses");
    assert!(
        model.referenced_covariates.is_empty(),
        "an individual-parameter reference must not register as a data column, got {:?}",
        model.referenced_covariates
    );
}

/// `obs_scale` is a subject-static divisor — `apply_scaling` evaluates it once per
/// subject at t = 0 and divides the whole prediction vector by the result — so a
/// `TIME` reference in it would always read the t=0 value. Reject it (naming Form C
/// as the place for a time-dependent readout) rather than serve the silent
/// collapse. Both spellings of the built-in are caught.
#[test]
fn scaling_obs_scale_rejects_the_time_builtin() {
    for expr in ["TIME", "V * TIME", "T", "V + T"] {
        let src = analytical_model_with_scaling(Some(&format!("  obs_scale = {expr}\n")));
        let err =
            parse_model_string(&src).expect_err("obs_scale referencing TIME must be rejected");
        assert!(
            err.contains("subject-static") && err.contains("y = <expr>"),
            "error for `{expr}` must explain the subject-static divisor and point at \
             Form C, got: {err}"
        );
    }
}

/// A Form C `y = <expr>` readout may reference `TIME` — it is evaluated per
/// observation — so the parser must leave it alone and, crucially, must not
/// register it as a data column.
#[test]
fn scaling_y_readout_accepts_the_time_builtin() {
    let src = ode_model_with_scaling(
        "ode(states=[depot, central])",
        Some("  y = central / V * TIME\n"),
    );
    let model = parse_model_string(&src).expect("Form C readout may reference TIME");
    assert!(
        model.referenced_covariates.is_empty(),
        "`TIME` is a built-in, not a data column, got {:?}",
        model.referenced_covariates
    );
}

/// Does the model's uniform Form C readout compile to a `PushTime` read? The
/// direct discriminator for the `[scaling]` time built-in: `referenced_covariates`
/// alone cannot tell "folded into `TIME`" from "bound to something else", since
/// both leave it empty.
fn readout_reads_time(model: &CompiledModel) -> bool {
    let program = model
        .ode_spec
        .as_ref()
        .and_then(|o| o.readout_program.as_ref())
        .or_else(|| {
            model
                .analytic_readout
                .as_ref()
                .and_then(|a| a.program.as_ref())
        })
        .expect("uniform Form C readout carries a program");
    program.bc.ops.iter().any(|op| matches!(op, Op::PushTime))
}

/// `T` / `t` is the `[odes]` spelling of the same built-in (reserved there against
/// every state / indiv-param / intermediate name). `[scaling]` parses with
/// `fallback_covariate = true`, so before #1028 a bare `T` landed as a covariate —
/// a required column named `T`, or a silent zero. It now folds into the same
/// `Expression::Time` node `TIME` produces, so the two blocks agree on the name.
#[test]
fn scaling_y_readout_accepts_the_t_time_alias() {
    for alias in ["T", "t", "TIME", "time"] {
        let src = ode_model_with_scaling(
            "ode(states=[depot, central])",
            Some(&format!("  y = central / V * {alias}\n")),
        );
        let model =
            parse_model_string(&src).expect("Form C readout may reference the T time alias");
        assert!(
            model.referenced_covariates.is_empty(),
            "`{alias}` must fold into the TIME built-in, not register as a data \
             column; got {:?}",
            model.referenced_covariates
        );
        assert!(
            readout_reads_time(&model),
            "`{alias}` must compile to the model-time read"
        );
    }
}

/// The `T` fold targets `Covariate` leaves only, so a name the parse *did* bind
/// keeps winning — an individual parameter named `T` still resolves to itself, and
/// the readout compiles to a plain variable read rather than a model-time read.
/// (Unreachable on ODE models, where `[odes]` rejects that name outright.)
#[test]
fn scaling_y_readout_t_alias_does_not_shadow_a_bound_name() {
    let src = analytical_model_with_scaling(Some("  y = central / T\n"))
        .replace("  V  = TVV\n", "  V  = TVV\n  T  = TVV / 2\n");
    let model = parse_model_string(&src).expect("indiv param named T parses");
    assert!(
        model.referenced_covariates.is_empty(),
        "an individual parameter named `T` must win over the time alias, got {:?}",
        model.referenced_covariates
    );
    assert!(
        !readout_reads_time(&model),
        "an individual parameter named `T` must not be folded into the time built-in"
    );
}

/// The `T` fold walks the whole expression tree, not just its top-level operands:
/// `visit_expr_nodes_mut` has to recurse through every node kind that carries a
/// child. This readout puts a `T` under each one — a compound `if` condition
/// (`Conditional` over `Condition::And` over `Condition::Compare`) and both of its
/// arms, a unary function argument (`UnaryFn`), and both sides of a power
/// (`Power`) — so a missed arm leaves a stray `Covariate("T")` behind and shows up
/// as a spurious required data column.
#[test]
fn scaling_y_readout_folds_the_t_alias_under_every_node_kind() {
    let src = ode_model_with_scaling(
        "ode(states=[depot, central])",
        Some("  y = if (T > 1 && T < 100) exp(-T) * (T ^ T) else central / V + sqrt(T)\n"),
    );
    let model = parse_model_string(&src).expect("nested T references parse");
    assert!(
        model.referenced_covariates.is_empty(),
        "every `T` must fold into the time built-in, whatever node it sits under; \
         got {:?}",
        model.referenced_covariates
    );
    assert!(
        readout_reads_time(&model),
        "the folded readout must compile to the model-time read"
    );
}

/// A `[scaling]` `y` covariate has been a required data column since #540, so a
/// dataset with a real column named `T` and a readout like `y = central/V * T`
/// worked before #1028 — and the alias fold must not silently repoint it at the
/// clock. Declaring `T` in `[covariates]` is the explicit escape hatch: the name
/// then keeps its data-column meaning, exactly as a bound state or individual
/// parameter does.
#[test]
fn scaling_y_readout_t_alias_loses_to_a_declared_covariate() {
    for alias in ["T", "t"] {
        let src = ode_model_with_scaling(
            "ode(states=[depot, central])",
            Some(&format!("  y = central / V * {alias}\n")),
        ) + &format!("\n[covariates]\n  {alias} continuous\n");
        let model = parse_model_string(&src).expect("a declared T covariate parses");
        assert_eq!(
            model.referenced_covariates,
            vec![alias.to_string()],
            "a `[covariates]`-declared `{alias}` must stay a required data column"
        );
        assert!(
            !readout_reads_time(&model),
            "a declared `{alias}` must not be folded into the model-time built-in"
        );
    }
}

/// The alias collapses `T` and `t` into one built-in, so the declaration that
/// protects it has to be matched the same way. A case-exact guard let
/// `[covariates] T` fail to protect a `y = ... t ...` reference — silently folding
/// it to the clock even though the user took the documented escape hatch — and, in
/// `obs_scale`, raised the `TIME`-rejection error against a legitimately declared
/// column. Both spellings of the declaration must cover both spellings of the
/// reference (#1042 review).
#[test]
fn scaling_time_alias_declaration_is_case_insensitive() {
    for declared in ["T", "t"] {
        for used in ["T", "t"] {
            let src = ode_model_with_scaling(
                "ode(states=[depot, central])",
                Some(&format!("  y = central / V * {used}\n")),
            ) + &format!("\n[covariates]\n  {declared} continuous\n");
            let model = parse_model_string(&src)
                .unwrap_or_else(|e| panic!("declared `{declared}`, used `{used}`: {e}"));
            assert_eq!(
                model.referenced_covariates,
                vec![used.to_string()],
                "declared `{declared}` must protect the `{used}` reference"
            );
            assert!(
                !readout_reads_time(&model),
                "declared `{declared}`, used `{used}`: must not fold to the clock"
            );
        }
    }
}

/// The same case-insensitivity on the `obs_scale` side, where the mismatch was not
/// a silent fold but a hard error thrown at a column the model legitimately
/// declared.
#[test]
fn obs_scale_time_rejection_respects_a_case_mismatched_declaration() {
    let src = ode_model_with_scaling(
        "ode(states=[depot, central])",
        Some("  obs_scale = 1000 / t\n  y = central\n"),
    ) + "\n[covariates]\n  T continuous\n";
    let model = parse_model_string(&src).expect("a declared `T` column is not the TIME built-in");
    assert_eq!(model.referenced_covariates, vec!["t".to_string()]);
}

/// …and when the fold *does* fire (no declaration), it says so. The undeclared
/// case is the one the parser cannot disambiguate — a dataset column named `T` is
/// only a warning elsewhere — so the note names both escapes: spell it `TIME`, or
/// declare the column.
#[test]
fn scaling_y_readout_t_alias_fold_warns() {
    let src = ode_model_with_scaling(
        "ode(states=[depot, central])",
        Some("  y = central / V * T\n"),
    );
    let model = parse_model_string(&src).expect("the T alias parses");
    let note = model
        .parse_warnings
        .iter()
        .find(|w| w.contains("model-time built-in"))
        .unwrap_or_else(|| panic!("expected a fold warning, got {:?}", model.parse_warnings));
    assert!(
        note.contains("[covariates]"),
        "warning must name the escape hatch: {note}"
    );

    // Spelling it `TIME` is unambiguous and must not warn.
    let src_time = ode_model_with_scaling(
        "ode(states=[depot, central])",
        Some("  y = central / V * TIME\n"),
    );
    let model_time = parse_model_string(&src_time).expect("TIME parses");
    assert!(
        !model_time
            .parse_warnings
            .iter()
            .any(|w| w.contains("model-time built-in")),
        "the unambiguous `TIME` spelling must not warn, got {:?}",
        model_time.parse_warnings
    );
}

/// `[odes]` reserves `TIME`/`T`/`TAFD`/`TAD`/`MACHEPS`, but only `TIME` (and its
/// `T` alias) is a `[scaling]` built-in. `TAFD`/`TAD`/`MACHEPS` stay **ordinary
/// covariates** here — documented behaviour (`docs/model-file/scaling.qmd`), so a
/// dataset that really carries a `TAD` column can use it. This pins that: they
/// register as required data columns and are not folded into anything.
#[test]
fn scaling_odes_only_builtins_stay_covariates() {
    for builtin in ["TAFD", "TAD", "MACHEPS"] {
        let src = ode_model_with_scaling(
            "ode(states=[depot, central])",
            Some(&format!("  y = central / V + {builtin}\n")),
        );
        let model = parse_model_string(&src).expect("an [odes]-only name parses as a covariate");
        assert_eq!(
            model.referenced_covariates,
            vec![builtin.to_string()],
            "`{builtin}` must register as a required data column in [scaling]"
        );
        assert!(
            !readout_reads_time(&model),
            "`{builtin}` must not resolve to the model-time built-in"
        );
    }
}

#[test]
fn test_parse_scaling_y_per_cmt_form_c() {
    // Per-CMT Form C on an ODE model. Build_y_output_fn picks up each
    // entry; the parser assembles them into OdeReadout::PerCmt.
    let src = ode_model_with_scaling(
        "ode(states=[depot, central])",
        Some("  y[CMT=1] = central / V\n  y[CMT=2] = central / V * 1000\n"),
    );
    let model = parse_model_string(&src).expect("per-CMT y Form C parses");
    let ode = model.ode_spec.as_ref().expect("ODE spec present");
    match &ode.readout {
        crate::ode::OdeReadout::PerCmt(map) => {
            assert_eq!(map.len(), 2);
            assert!(map.contains_key(&1));
            assert!(map.contains_key(&2));
        }
        _ => panic!("expected OdeReadout::PerCmt"),
    }
}

#[test]
fn test_parse_scaling_y_form_c_with_ad_parses() {
    // `gradient = ad` is retired and now rejected uniformly by
    // `check_model_options` (`E_AD_RETIRED`), so the old Form-C-specific
    // *parse-time* guard was removed. A Form C model with `gradient = ad`
    // therefore parses; the rejection happens at model-options validation
    // (covered by `api::ad_requested_errors_now_that_ad_is_retired`).
    let src_per_cmt = ode_model_with_scaling(
        "ode(states=[depot, central])",
        Some("  y[CMT=1] = central / V\n  y[CMT=2] = central / V * 1000\n"),
    )
    .replace("gradient = fd", "gradient = ad");
    assert!(
        parse_model_string(&src_per_cmt).is_ok(),
        "Form C + gradient = ad should parse (rejected later at check_model_options)"
    );

    let src_single =
        ode_model_with_scaling("ode(states=[depot, central])", Some("  y = central / V\n"))
            .replace("gradient = fd", "gradient = ad");
    assert!(
        parse_model_string(&src_single).is_ok(),
        "single Form C + gradient = ad should parse (rejected later)"
    );
}

// ── Bytecode ↔ AST evaluator equivalence ────────────────────────────────
//
// The bytecode interpreter is a second evaluator for every
// Expression/Condition shape ferx supports. These tests pin its
// results to `eval_expression_indexed` so future opcode changes can't
// silently diverge — Copilot review on #137 flagged the gap.

fn bc_vs_ast(expr: Expression, vars: &[f64], theta: &[f64], eta: &[f64], covs: &[f64]) {
    let nn: Vec<Vec<f64>> = Vec::new();
    let ast = eval_expression_indexed(&expr, theta, eta, covs, vars, &nn);
    let bc = compile_bytecode(&expr);
    let mut stack: Vec<f64> = Vec::new();
    let by = eval_bytecode(&bc, theta, eta, covs, vars, &nn, &mut stack);
    assert_eq!(
        ast.to_bits(),
        by.to_bits(),
        "bit-identical mismatch: AST = {ast:?}, bytecode = {by:?}, expr = {expr:?}",
    );
}

fn lit(v: f64) -> Expression {
    Expression::Literal(v)
}
fn binop(op: BinOp, l: Expression, r: Expression) -> Expression {
    Expression::BinOp(Box::new(l), op, Box::new(r))
}
fn unary(name: &str, arg: Expression) -> Expression {
    Expression::UnaryFn(name.into(), Box::new(arg))
}
fn cond(c: Condition, t: Expression, e: Expression) -> Expression {
    Expression::Conditional(Box::new(c), Box::new(t), Box::new(e))
}
fn cmp(l: Expression, op: CmpOp, r: Expression) -> Condition {
    Condition::Compare(l, op, r)
}

// ── Generic bytecode VM (`eval_bytecode_g`) ──────────────────────────────
//
// The Dual2 analytic-sensitivity path reuses the same `Bytecode` through
// `eval_bytecode_g`. These tests pin it to the scalar evaluator: the `f64`
// monomorphization must be bit-identical to `eval_bytecode`, and the
// `Dual2<2>` value/grad/Hessian must match the f64 path and its finite
// differences — so the second evaluator cannot silently drift.

fn varidx(i: usize) -> Expression {
    Expression::VariableIdx(i)
}
fn power_expr(b: Expression, e: Expression) -> Expression {
    Expression::Power(Box::new(b), Box::new(e))
}
fn close(a: f64, b: f64, rel: f64, eps: f64) -> bool {
    let d = (a - b).abs();
    d <= eps || d <= rel * a.abs().max(b.abs())
}

fn bc_g_f64_matches(expr: &Expression, vars: &[f64]) {
    let nn: Vec<Vec<f64>> = Vec::new();
    let bc = compile_bytecode(expr);
    let mut s1: Vec<f64> = Vec::new();
    let f = eval_bytecode(&bc, &[], &[], &[], vars, &nn, &mut s1);
    let mut s2: Vec<f64> = Vec::new();
    let g = eval_bytecode_g::<f64>(&bc, &[], &[], &[], vars, &nn, &mut s2);
    assert_eq!(
        f.to_bits(),
        g.to_bits(),
        "f64 generic VM drift: expr={expr:?} f={f} g={g}"
    );
}

/// Vars 0 and 1 are seeded as `Dual2<2>` variables; check value/grad/Hessian
/// against the f64 evaluator and its central finite differences.
fn bc_g_dual2_fd(expr: &Expression, x: f64, y: f64, gtol: f64, htol: f64) {
    use crate::sens::dual2::Dual2;
    let nn: Vec<Vec<f64>> = Vec::new();
    let bc = compile_bytecode(expr);
    let vd = [Dual2::<2>::var(x, 0), Dual2::<2>::var(y, 1)];
    let mut sd: Vec<Dual2<2>> = Vec::new();
    let d = eval_bytecode_g::<Dual2<2>>(&bc, &[], &[], &[], &vd, &nn, &mut sd);
    let v = |a: f64, b: f64| {
        let mut s: Vec<f64> = Vec::new();
        eval_bytecode(&bc, &[], &[], &[], &[a, b], &nn, &mut s)
    };
    assert!(close(d.value, v(x, y), 1e-12, 1e-12), "value: {expr:?}");
    let h = 1e-5;
    let gx = (v(x + h, y) - v(x - h, y)) / (2.0 * h);
    let gy = (v(x, y + h) - v(x, y - h)) / (2.0 * h);
    assert!(
        close(d.grad[0], gx, gtol, 1e-7),
        "grad0 {:?}: {} vs {}",
        expr,
        d.grad[0],
        gx
    );
    assert!(
        close(d.grad[1], gy, gtol, 1e-7),
        "grad1 {:?}: {} vs {}",
        expr,
        d.grad[1],
        gy
    );
    let hh = 1e-4;
    let hxx = (v(x + hh, y) - 2.0 * v(x, y) + v(x - hh, y)) / (hh * hh);
    let hyy = (v(x, y + hh) - 2.0 * v(x, y) + v(x, y - hh)) / (hh * hh);
    let hxy = (v(x + hh, y + hh) - v(x + hh, y - hh) - v(x - hh, y + hh) + v(x - hh, y - hh))
        / (4.0 * hh * hh);
    assert!(
        close(d.hess[0][0], hxx, htol, 1e-4),
        "hxx {:?}: {} vs {}",
        expr,
        d.hess[0][0],
        hxx
    );
    assert!(
        close(d.hess[1][1], hyy, htol, 1e-4),
        "hyy {:?}: {} vs {}",
        expr,
        d.hess[1][1],
        hyy
    );
    assert!(
        close(d.hess[0][1], hxy, htol, 1e-4),
        "hxy {:?}: {} vs {}",
        expr,
        d.hess[0][1],
        hxy
    );
    assert!(
        close(d.hess[0][1], d.hess[1][0], 1e-12, 1e-12),
        "hess symmetry"
    );
}

#[test]
fn bytecode_g_f64_bit_identical() {
    let vars = [1.7_f64, 0.8];
    let exprs = [
        binop(BinOp::Add, varidx(0), varidx(1)),
        binop(BinOp::Sub, varidx(0), varidx(1)),
        binop(BinOp::Mul, varidx(0), varidx(1)),
        binop(BinOp::Div, varidx(0), varidx(1)),
        binop(BinOp::Mod, varidx(0), lit(0.5)),
        power_expr(varidx(0), lit(2.5)),
        power_expr(varidx(0), varidx(1)),
        unary("exp", binop(BinOp::Sub, lit(0.0), varidx(0))),
        unary("ln", binop(BinOp::Mul, varidx(0), varidx(1))),
        unary("sqrt", binop(BinOp::Add, varidx(0), varidx(1))),
        unary("abs", binop(BinOp::Sub, varidx(0), varidx(1))),
        unary("inv_logit", binop(BinOp::Sub, varidx(0), varidx(1))),
        unary("logit", binop(BinOp::Mul, lit(0.3), varidx(1))),
        unary("floor", varidx(0)),
        unary("ceil", varidx(0)),
        unary("round", varidx(0)),
        cond(cmp(varidx(0), CmpOp::Gt, varidx(1)), lit(1.0), lit(2.0)),
    ];
    for e in &exprs {
        bc_g_f64_matches(e, &vars);
    }
    // Guard branches: zero divisor, ln/sqrt of a non-positive argument.
    bc_g_f64_matches(&binop(BinOp::Div, varidx(0), lit(0.0)), &vars);
    bc_g_f64_matches(&unary("ln", binop(BinOp::Sub, lit(0.0), varidx(0))), &vars);
    bc_g_f64_matches(
        &unary("sqrt", binop(BinOp::Sub, lit(0.0), varidx(0))),
        &vars,
    );
}

#[test]
fn eval_statements_g_matches_f64_and_fd() {
    use crate::sens::dual2::Dual2;
    // vars: slot0=u (state), slot1=k (param), slot2=tmp (intermediate).
    //   tmp = k·u ;  d/dt(u) = −tmp.
    let stmts = vec![
        Statement::AssignBc(
            2,
            compile_bytecode(&binop(BinOp::Mul, varidx(1), varidx(0))),
        ),
        Statement::DiffEqBc(0, compile_bytecode(&binop(BinOp::Sub, lit(0.0), varidx(2)))),
    ];

    let mut vf = vec![2.0_f64, 0.3, 0.0];
    let mut duf = vec![0.0_f64];
    let mut sf: Vec<f64> = Vec::new();
    eval_statements_indexed_with_stack(
        &stmts,
        &[],
        &[],
        &[],
        &mut vf,
        Some(&mut duf),
        &[],
        &mut sf,
    );

    // Seed k (slot 1) as the differentiated variable.
    let mut vd = vec![
        Dual2::<1>::constant(2.0),
        Dual2::<1>::var(0.3, 0),
        Dual2::<1>::constant(0.0),
    ];
    let mut dud = vec![Dual2::<1>::constant(0.0)];
    let mut sd: Vec<Dual2<1>> = Vec::new();
    eval_statements_g::<Dual2<1>>(&stmts, &[], &[], &[], &mut vd, Some(&mut dud), &mut sd, &[]);

    assert!(close(dud[0].value, duf[0], 1e-12, 1e-12), "value vs f64");
    // du[0] = −k·u = −0.6; ∂/∂k = −u = −2; ∂²/∂k² = 0.
    assert!(close(dud[0].value, -0.6, 1e-12, 1e-12));
    assert!(close(dud[0].grad[0], -2.0, 1e-9, 1e-9));
    assert!(close(dud[0].hess[0][0], 0.0, 1e-9, 1e-9));
}

#[test]
fn bytecode_g_dual2_matches_fd() {
    bc_g_dual2_fd(
        &unary("ln", binop(BinOp::Mul, varidx(0), varidx(1))),
        1.7,
        0.8,
        1e-6,
        1e-3,
    );
    bc_g_dual2_fd(&power_expr(varidx(0), lit(2.5)), 1.7, 0.8, 1e-6, 1e-3);
    bc_g_dual2_fd(&power_expr(varidx(0), varidx(1)), 1.7, 0.8, 1e-6, 2e-3);
    bc_g_dual2_fd(
        &unary("sqrt", binop(BinOp::Add, varidx(0), varidx(1))),
        1.7,
        0.8,
        1e-6,
        1e-3,
    );
    bc_g_dual2_fd(
        &unary("inv_logit", binop(BinOp::Sub, varidx(0), varidx(1))),
        0.6,
        0.2,
        1e-6,
        1e-3,
    );
    bc_g_dual2_fd(
        &unary(
            "abs",
            binop(
                BinOp::Sub,
                binop(BinOp::Mul, varidx(0), varidx(0)),
                varidx(1),
            ),
        ),
        2.0,
        1.0,
        1e-6,
        1e-3,
    );
    // 1-cpt IV-bolus shape: (1/V)·exp(−(CL/V)). vars: 0=CL, 1=V.
    bc_g_dual2_fd(
        &binop(
            BinOp::Mul,
            binop(BinOp::Div, lit(1.0), varidx(1)),
            unary(
                "exp",
                binop(
                    BinOp::Sub,
                    lit(0.0),
                    binop(BinOp::Div, varidx(0), varidx(1)),
                ),
            ),
        ),
        3.0,
        5.0,
        1e-6,
        1e-3,
    );
}

#[test]
fn compute_max_stack_allows_branching_bytecode() {
    // Regression: a `Conditional` compiles to
    //   cond; JumpIfFalse(else); then; Jump(end); else
    // so `compute_max_stack`'s linear scan walks BOTH arms and ends above
    // depth 1 (one extra per branch). The end-depth assertion must be
    // skipped when jumps are present; before the fix, `compile_bytecode`
    // panicked here under debug-assertions ("bytecode ends at depth 2").
    let expr = cond(cmp(lit(2.0), CmpOp::Lt, lit(3.0)), lit(1.0), lit(-1.0));
    let bc = compile_bytecode(&expr); // would panic pre-fix
    assert!(
        bc.max_stack >= 1,
        "max_stack must be >= 1, got {}",
        bc.max_stack
    );

    // The reserved size must actually be sufficient to evaluate correctly:
    // 2 < 3 ⇒ the `then` arm (1.0).
    let nn: Vec<Vec<f64>> = Vec::new();
    let mut stack: Vec<f64> = Vec::new();
    let by = eval_bytecode(&bc, &[], &[], &[], &[], &nn, &mut stack);
    assert_eq!(by.to_bits(), 1.0_f64.to_bits());

    // Nested conditionals (deeper branch nesting) must also not trip it,
    // and a deeper `peak` must still be returned.
    let nested = cond(
        cmp(lit(1.0), CmpOp::Gt, lit(0.0)),
        cond(cmp(lit(2.0), CmpOp::Lt, lit(5.0)), lit(10.0), lit(20.0)),
        lit(-1.0),
    );
    assert!(compute_max_stack(&compile_bytecode(&nested).ops) >= 1);

    // The jump-free path is unchanged: `1 + 2` pushes two operands before
    // `Add`, so the peak (the returned size) is 2 — and its end-depth of 1
    // still satisfies the (retained) jump-free assertion.
    assert_eq!(
        compute_max_stack(&compile_bytecode(&binop(BinOp::Add, lit(1.0), lit(2.0))).ops),
        2
    );

    // `bytecode_has_branch` (which gates the assertion) — true for
    // conditional bytecode, false for a jump-free arithmetic expression.
    assert!(bytecode_has_branch(&compile_bytecode(&expr).ops));
    assert!(!bytecode_has_branch(
        &compile_bytecode(&binop(BinOp::Add, lit(1.0), lit(2.0))).ops
    ));

    // `ends_at_expected_depth` — the returnable predicate the `debug_assert!`
    // consumes. Exercised directly (with hard-coded depths) so its branching
    // logic is covered even under the `ci-test` profile (debug-assertions
    // off), where the assertion is compiled out and never runs. Branchy
    // bytecode is exempt at any end depth; jump-free bytecode must end at
    // depth 1. The depth real bytecode actually compiles to is pinned
    // separately in `compute_max_stack_jumpfree_bytecode_ends_at_depth_one`.
    let cond_ops = compile_bytecode(&expr).ops;
    let addops = compile_bytecode(&binop(BinOp::Add, lit(1.0), lit(2.0))).ops;
    assert!(ends_at_expected_depth(&cond_ops, 2)); // branchy: exempt even at depth 2
    assert!(ends_at_expected_depth(&addops, 1)); // jump-free: depth 1 ok
    assert!(!ends_at_expected_depth(&addops, 2)); // jump-free: depth 2 rejected
    assert!(ends_at_expected_depth(&[], 0)); // empty: ok
}

#[test]
fn compute_max_stack_jumpfree_bytecode_ends_at_depth_one() {
    // Closes the gap the sibling test leaves open. That one feeds
    // `ends_at_expected_depth` HARD-CODED depths, so it pins the predicate's
    // branching logic but never the depth a real expression compiles to — a
    // future `compile_expr_into` off-by-one would slip past it. Here we scan
    // REAL bytecode and assert the end-depth invariant directly.
    //
    // `scan_stack_depth` *returns* the end depth (rather than only feeding it
    // to the `debug_assert!` in `compute_max_stack`), so these are plain
    // value assertions that hold under every profile — including `ci-test`,
    // where debug-assertions are off. That is what makes such a regression
    // catchable in CI, not only under the local dev profile.
    for expr in [
        lit(3.0),
        binop(BinOp::Add, lit(1.0), lit(2.0)),
        binop(BinOp::Sub, binop(BinOp::Mul, lit(2.0), lit(3.0)), lit(4.0)),
        unary("exp", lit(0.5)),
        unary("ln", binop(BinOp::Div, lit(6.0), lit(2.0))),
    ] {
        let ops = compile_bytecode(&expr).ops;
        assert!(
            !bytecode_has_branch(&ops),
            "expr must compile jump-free for this invariant: {ops:?}"
        );
        let (peak, end_depth) = scan_stack_depth(&ops);
        // A well-formed expression leaves exactly one result on the stack.
        assert_eq!(
            end_depth, 1,
            "jump-free expr must end at depth 1, got {end_depth}: {ops:?}"
        );
        // `peak` (the reserved stack size) never under-counts the end depth,
        // and `compute_max_stack` returns exactly `peak.max(1)`.
        assert!(
            peak >= end_depth,
            "peak {peak} < end_depth {end_depth}: {ops:?}"
        );
        assert_eq!(compute_max_stack(&ops), peak.max(1) as usize);
    }

    // Counterpart to the exemption: branchy bytecode genuinely ends ABOVE
    // depth 1 (the linear scan walks both arms), which is exactly why
    // `compute_max_stack` skips the end-depth assert when jumps are present.
    let cond_ops = compile_bytecode(&cond(
        cmp(lit(2.0), CmpOp::Lt, lit(3.0)),
        lit(1.0),
        lit(-1.0),
    ))
    .ops;
    assert!(bytecode_has_branch(&cond_ops));
    assert!(
        scan_stack_depth(&cond_ops).1 > 1,
        "conditional should end above depth 1 (both arms walked): {cond_ops:?}"
    );
}

#[test]
fn bytecode_matches_ast_on_arithmetic_and_literals() {
    let v = &[3.5, -2.0, 0.0];
    let t = &[10.0, 0.1];
    let e = &[0.5];
    let c = &[];
    // Lits and slot pushes
    bc_vs_ast(lit(7.5), v, t, e, c);
    bc_vs_ast(Expression::Theta(0), v, t, e, c);
    bc_vs_ast(Expression::Eta(0), v, t, e, c);
    bc_vs_ast(Expression::VariableIdx(1), v, t, e, c);
    // Binary arithmetic — left-to-right evaluation
    bc_vs_ast(binop(BinOp::Add, lit(2.0), lit(3.0)), v, t, e, c);
    bc_vs_ast(binop(BinOp::Sub, lit(2.0), lit(3.0)), v, t, e, c);
    bc_vs_ast(binop(BinOp::Mul, lit(2.0), lit(3.0)), v, t, e, c);
    bc_vs_ast(binop(BinOp::Div, lit(7.0), lit(2.0)), v, t, e, c);
    // Div-by-tiny guard (both paths clamp `r.abs() < 1e-30 -> 0.0`)
    bc_vs_ast(binop(BinOp::Div, lit(1.0), lit(1e-40)), v, t, e, c);
    bc_vs_ast(binop(BinOp::Div, lit(1.0), lit(0.0)), v, t, e, c);
    // Power
    bc_vs_ast(
        Expression::Power(Box::new(lit(2.0)), Box::new(lit(3.0))),
        v,
        t,
        e,
        c,
    );
    bc_vs_ast(
        Expression::Power(Box::new(lit(4.0)), Box::new(lit(0.5))),
        v,
        t,
        e,
        c,
    );
}

#[test]
fn bytecode_matches_ast_on_unary_guards() {
    let v = &[];
    let t = &[];
    let e = &[];
    let c = &[];
    // exp / abs — no guards
    bc_vs_ast(unary("exp", lit(0.5)), v, t, e, c);
    bc_vs_ast(unary("abs", lit(-3.0)), v, t, e, c);
    // ln / log clamp `v.max(1e-30).ln()` — exercise negative and zero
    bc_vs_ast(unary("ln", lit(2.0)), v, t, e, c);
    bc_vs_ast(unary("ln", lit(0.0)), v, t, e, c);
    bc_vs_ast(unary("ln", lit(-1.0)), v, t, e, c);
    bc_vs_ast(unary("log", lit(1e-40)), v, t, e, c);
    // sqrt clamp `v.max(0.0).sqrt()`
    bc_vs_ast(unary("sqrt", lit(9.0)), v, t, e, c);
    bc_vs_ast(unary("sqrt", lit(-4.0)), v, t, e, c);
    // inv_logit numerically-stable branch (v ≥ 0 vs v < 0)
    bc_vs_ast(unary("inv_logit", lit(2.5)), v, t, e, c);
    bc_vs_ast(unary("inv_logit", lit(-2.5)), v, t, e, c);
    bc_vs_ast(unary("expit", lit(0.0)), v, t, e, c);
    // logit clamp `v.clamp(1e-15, 1.0 - 1e-15)`
    bc_vs_ast(unary("logit", lit(0.3)), v, t, e, c);
    bc_vs_ast(unary("logit", lit(0.0)), v, t, e, c);
    bc_vs_ast(unary("logit", lit(1.0)), v, t, e, c);
    // Unknown unary name — slow path returns the argument unchanged;
    // the bytecode no-op'd UnaryFn arm must too.
    bc_vs_ast(unary("expn", lit(42.0)), v, t, e, c);
}

#[test]
fn bytecode_matches_ast_on_conditional_and_logic() {
    let v = &[];
    let t = &[];
    let e = &[];
    let c = &[];
    // Simple compare arms
    for op in [
        CmpOp::Lt,
        CmpOp::Le,
        CmpOp::Gt,
        CmpOp::Ge,
        CmpOp::Eq,
        CmpOp::Ne,
    ] {
        bc_vs_ast(
            cond(cmp(lit(2.0), op, lit(3.0)), lit(1.0), lit(-1.0)),
            v,
            t,
            e,
            c,
        );
        bc_vs_ast(
            cond(cmp(lit(3.0), op, lit(3.0)), lit(1.0), lit(-1.0)),
            v,
            t,
            e,
            c,
        );
    }
    // NaN-comparison sanity (NaN compares as false everywhere)
    bc_vs_ast(
        cond(
            cmp(lit(f64::NAN), CmpOp::Eq, lit(f64::NAN)),
            lit(1.0),
            lit(-1.0),
        ),
        v,
        t,
        e,
        c,
    );
    // And / Or / Not over compound conditions
    bc_vs_ast(
        cond(
            Condition::And(
                Box::new(cmp(lit(1.0), CmpOp::Lt, lit(2.0))),
                Box::new(cmp(lit(3.0), CmpOp::Lt, lit(4.0))),
            ),
            lit(42.0),
            lit(0.0),
        ),
        v,
        t,
        e,
        c,
    );
    bc_vs_ast(
        cond(
            Condition::Or(
                Box::new(cmp(lit(1.0), CmpOp::Gt, lit(2.0))),
                Box::new(cmp(lit(3.0), CmpOp::Lt, lit(4.0))),
            ),
            lit(7.0),
            lit(0.0),
        ),
        v,
        t,
        e,
        c,
    );
    bc_vs_ast(
        cond(
            Condition::Not(Box::new(cmp(lit(1.0), CmpOp::Lt, lit(2.0)))),
            lit(7.0),
            lit(0.0),
        ),
        v,
        t,
        e,
        c,
    );
    // Nested Conditional — exercises both then- and else-branch bytecode
    // bookkeeping (the compute_max_stack-over-counts case).
    let inner = cond(cmp(lit(1.0), CmpOp::Lt, lit(0.5)), lit(11.0), lit(22.0));
    let outer = cond(
        cmp(lit(0.0), CmpOp::Lt, lit(1.0)),
        inner.clone(),
        binop(BinOp::Add, lit(1.0), inner),
    );
    bc_vs_ast(outer, v, t, e, c);
}

#[test]
fn bytecode_matches_ast_on_nn_output_fallback() {
    // PushNnOutput out-of-bounds falls back to 0.0 in both paths.
    let nn_outputs: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0]];
    let theta = &[];
    let eta = &[];
    let covs = &[];
    let vars = &[];
    // Valid index
    let ast_ok = eval_expression_indexed(
        &Expression::NnOutput {
            nn_idx: 0,
            output_idx: 1,
        },
        theta,
        eta,
        covs,
        vars,
        &nn_outputs,
    );
    let bc_ok = {
        let bc = compile_bytecode(&Expression::NnOutput {
            nn_idx: 0,
            output_idx: 1,
        });
        let mut stack = Vec::new();
        eval_bytecode(&bc, theta, eta, covs, vars, &nn_outputs, &mut stack)
    };
    assert_eq!(ast_ok, bc_ok);
    assert_eq!(bc_ok, 2.0);
}

// ── Symbolic AST differentiator ─────────────────────────────────────────
//
// Verifies `differentiate(expr, axis)` against central finite differences
// on `eval_expression_indexed`. The tolerance reflects FD's noise floor:
// central FD with h = 1e-5 on a smooth expression gives ~h² ≈ 1e-10
// truncation error, so any disagreement larger than ~1e-7 relative
// (multiplied by max(|f(x+h)|, |f(x-h)|, 1) for scale) is a real bug.

/// Evaluate an Expression at the given (theta, eta, vars, covs) point.
fn eval_at(expr: &Expression, theta: &[f64], eta: &[f64], vars: &[f64], covs: &[f64]) -> f64 {
    let nn: Vec<Vec<f64>> = Vec::new();
    eval_expression_indexed(expr, theta, eta, covs, vars, &nn)
}

/// Central finite-difference of `expr` along `axis` at the given point.
/// Caller mutates the point's slot for the axis to compute f(x±h); we
/// take a vec for each slice and indirect the mutation.
fn fd_along(
    expr: &Expression,
    axis: DiffAxis,
    theta: &[f64],
    eta: &[f64],
    vars: &[f64],
    covs: &[f64],
) -> f64 {
    let h = 1e-5;
    let plus = |slot: usize, base: &[f64]| -> Vec<f64> {
        let mut v = base.to_vec();
        v[slot] += h;
        v
    };
    let minus = |slot: usize, base: &[f64]| -> Vec<f64> {
        let mut v = base.to_vec();
        v[slot] -= h;
        v
    };
    let (fp, fm) = match axis {
        DiffAxis::Theta(k) => {
            let tp = plus(k, theta);
            let tm = minus(k, theta);
            (
                eval_at(expr, &tp, eta, vars, covs),
                eval_at(expr, &tm, eta, vars, covs),
            )
        }
        DiffAxis::Eta(k) => {
            let ep = plus(k, eta);
            let em = minus(k, eta);
            (
                eval_at(expr, theta, &ep, vars, covs),
                eval_at(expr, theta, &em, vars, covs),
            )
        }
        DiffAxis::Variable(k) => {
            let vp = plus(k, vars);
            let vm = minus(k, vars);
            (
                eval_at(expr, theta, eta, &vp, covs),
                eval_at(expr, theta, eta, &vm, covs),
            )
        }
    };
    (fp - fm) / (2.0 * h)
}

/// Assert that the symbolic derivative matches central FD to ~FD-noise
/// tolerance.
fn assert_diff_matches_fd(
    expr: Expression,
    axis: DiffAxis,
    theta: &[f64],
    eta: &[f64],
    vars: &[f64],
    covs: &[f64],
) {
    let dexpr = differentiate(&expr, axis);
    let sym = eval_at(&dexpr, theta, eta, vars, covs);
    let num = fd_along(&expr, axis, theta, eta, vars, covs);
    let scale = sym.abs().max(num.abs()).max(1.0);
    let rel = (sym - num).abs() / scale;
    assert!(
        rel < 1e-6,
        "symbolic ≠ FD: sym = {sym}, fd = {num}, axis = {axis:?}, \n  expr = {expr:?}",
    );
}

// `lit`, `binop`, `unary` are already defined above in the bytecode
// equivalence tests; reuse those. Add only `power` (not needed there).
fn power(b: Expression, e: Expression) -> Expression {
    Expression::Power(Box::new(b), Box::new(e))
}

#[test]
fn differentiate_constants_zero_against_every_axis() {
    let theta = &[1.0, 2.0];
    let eta = &[0.5];
    let vars = &[3.0];
    let covs = &[];
    for axis in [DiffAxis::Theta(0), DiffAxis::Eta(0), DiffAxis::Variable(0)] {
        let d = differentiate(&lit(7.5), axis);
        assert_eq!(eval_at(&d, theta, eta, vars, covs), 0.0);
        let d2 = differentiate(&Expression::CovariateIdx(0), axis);
        assert_eq!(eval_at(&d2, theta, eta, vars, covs), 0.0);
        let d3 = differentiate(
            &Expression::NnOutput {
                nn_idx: 0,
                output_idx: 0,
            },
            axis,
        );
        assert_eq!(eval_at(&d3, theta, eta, vars, covs), 0.0);
    }
}

#[test]
fn differentiate_slot_pushes_kronecker_delta() {
    let theta = &[1.5, 2.5, 3.5];
    let eta = &[0.1, 0.2];
    let vars = &[4.0, 5.0];
    let covs = &[];
    // ∂θ_1 / ∂θ_j = δ_{1,j}
    for j in 0..3 {
        let d = differentiate(&Expression::Theta(1), DiffAxis::Theta(j));
        let want = if j == 1 { 1.0 } else { 0.0 };
        assert_eq!(eval_at(&d, theta, eta, vars, covs), want, "θ axis j={j}");
    }
    // Cross-axis: ∂θ_1 / ∂η_0 = 0
    let d = differentiate(&Expression::Theta(1), DiffAxis::Eta(0));
    assert_eq!(eval_at(&d, theta, eta, vars, covs), 0.0);
    // ∂η_1 / ∂η_j = δ_{1,j}
    for j in 0..2 {
        let d = differentiate(&Expression::Eta(1), DiffAxis::Eta(j));
        let want = if j == 1 { 1.0 } else { 0.0 };
        assert_eq!(eval_at(&d, theta, eta, vars, covs), want, "η axis j={j}");
    }
    // ∂v_0 / ∂v_0 = 1, ∂v_0 / ∂v_1 = 0
    for j in 0..2 {
        let d = differentiate(&Expression::VariableIdx(0), DiffAxis::Variable(j));
        let want = if j == 0 { 1.0 } else { 0.0 };
        assert_eq!(eval_at(&d, theta, eta, vars, covs), want, "var axis j={j}");
    }
}

#[test]
fn differentiate_arithmetic_matches_fd() {
    // expr = θ_0 + θ_1*η_0 − v_0/θ_0
    let theta = &[1.5, 2.5];
    let eta = &[0.7];
    let vars = &[3.0];
    let covs = &[];
    let expr = binop(
        BinOp::Sub,
        binop(
            BinOp::Add,
            Expression::Theta(0),
            binop(BinOp::Mul, Expression::Theta(1), Expression::Eta(0)),
        ),
        binop(BinOp::Div, Expression::VariableIdx(0), Expression::Theta(0)),
    );
    for axis in [
        DiffAxis::Theta(0),
        DiffAxis::Theta(1),
        DiffAxis::Eta(0),
        DiffAxis::Variable(0),
    ] {
        assert_diff_matches_fd(expr.clone(), axis, theta, eta, vars, covs);
    }
}

#[test]
fn differentiate_unary_fn_matches_fd() {
    let theta = &[1.5, 0.8];
    let eta = &[0.3];
    let vars = &[2.0];
    let covs = &[];
    // exp(θ_0)
    assert_diff_matches_fd(
        unary("exp", Expression::Theta(0)),
        DiffAxis::Theta(0),
        theta,
        eta,
        vars,
        covs,
    );
    // ln(θ_0 * v_0)
    assert_diff_matches_fd(
        unary(
            "ln",
            binop(BinOp::Mul, Expression::Theta(0), Expression::VariableIdx(0)),
        ),
        DiffAxis::Theta(0),
        theta,
        eta,
        vars,
        covs,
    );
    assert_diff_matches_fd(
        unary(
            "ln",
            binop(BinOp::Mul, Expression::Theta(0), Expression::VariableIdx(0)),
        ),
        DiffAxis::Variable(0),
        theta,
        eta,
        vars,
        covs,
    );
    // sqrt(θ_0² + v_0²)  — exercises non-trivial chain
    assert_diff_matches_fd(
        unary(
            "sqrt",
            binop(
                BinOp::Add,
                power(Expression::Theta(0), lit(2.0)),
                power(Expression::VariableIdx(0), lit(2.0)),
            ),
        ),
        DiffAxis::Theta(0),
        theta,
        eta,
        vars,
        covs,
    );
    // inv_logit(η_0 + θ_1)
    assert_diff_matches_fd(
        unary(
            "inv_logit",
            binop(BinOp::Add, Expression::Eta(0), Expression::Theta(1)),
        ),
        DiffAxis::Eta(0),
        theta,
        eta,
        vars,
        covs,
    );
    assert_diff_matches_fd(
        unary(
            "expit",
            binop(BinOp::Add, Expression::Eta(0), Expression::Theta(1)),
        ),
        DiffAxis::Theta(1),
        theta,
        eta,
        vars,
        covs,
    );
    // logit needs a point strictly in (0, 1)
    assert_diff_matches_fd(
        unary("logit", binop(BinOp::Mul, lit(0.3), Expression::Theta(0))),
        DiffAxis::Theta(0),
        &[1.0, 0.0],
        eta,
        vars,
        covs,
    );
}

#[test]
fn differentiate_abs_branches_on_sign() {
    // abs(θ_0 − 1) at θ_0 = 1.7  → derivative is +1 (positive branch)
    // abs(θ_0 − 1) at θ_0 = 0.3  → derivative is −1 (negative branch)
    let eta = &[];
    let vars = &[];
    let covs = &[];
    let expr = unary("abs", binop(BinOp::Sub, Expression::Theta(0), lit(1.0)));
    assert_diff_matches_fd(expr.clone(), DiffAxis::Theta(0), &[1.7], eta, vars, covs);
    assert_diff_matches_fd(expr, DiffAxis::Theta(0), &[0.3], eta, vars, covs);
}

#[test]
fn differentiate_unknown_unary_passes_through() {
    // The slow path returns the argument unchanged for unknown names;
    // the differentiator must return the argument's derivative.
    let theta = &[2.0];
    let eta = &[];
    let vars = &[];
    let covs = &[];
    let expr = unary("expn", binop(BinOp::Mul, Expression::Theta(0), lit(3.0)));
    assert_diff_matches_fd(expr, DiffAxis::Theta(0), theta, eta, vars, covs);
}

#[test]
fn differentiate_power_matches_fd_for_constant_and_variable_exponents() {
    let theta = &[2.5, 1.5];
    let eta = &[];
    let vars = &[];
    let covs = &[];
    // Constant exponent: θ_0² (the canonical case)
    assert_diff_matches_fd(
        power(Expression::Theta(0), lit(2.0)),
        DiffAxis::Theta(0),
        theta,
        eta,
        vars,
        covs,
    );
    // Constant exponent, non-integer: θ_0^0.5
    assert_diff_matches_fd(
        power(Expression::Theta(0), lit(0.5)),
        DiffAxis::Theta(0),
        theta,
        eta,
        vars,
        covs,
    );
    // Variable exponent (rare in PK/PD, but common via the Hill term):
    // θ_0^θ_1 — exercises the e' · ln(b) path
    assert_diff_matches_fd(
        power(Expression::Theta(0), Expression::Theta(1)),
        DiffAxis::Theta(0),
        theta,
        eta,
        vars,
        covs,
    );
    assert_diff_matches_fd(
        power(Expression::Theta(0), Expression::Theta(1)),
        DiffAxis::Theta(1),
        theta,
        eta,
        vars,
        covs,
    );
}

#[test]
fn differentiate_conditional_picks_taken_branch() {
    // Conditional(θ_0 > 0, 2·θ_0, 3·θ_0)
    // At θ_0 = 1.5: derivative is 2; at θ_0 = -1.5: derivative is 3
    let eta = &[];
    let vars = &[];
    let covs = &[];
    let expr = Expression::Conditional(
        Box::new(Condition::Compare(
            Expression::Theta(0),
            CmpOp::Gt,
            lit(0.0),
        )),
        Box::new(binop(BinOp::Mul, lit(2.0), Expression::Theta(0))),
        Box::new(binop(BinOp::Mul, lit(3.0), Expression::Theta(0))),
    );
    assert_diff_matches_fd(expr.clone(), DiffAxis::Theta(0), &[1.5], eta, vars, covs);
    assert_diff_matches_fd(expr, DiffAxis::Theta(0), &[-1.5], eta, vars, covs);
}

#[test]
fn differentiate_emax_pkpd_readout_matches_fd() {
    // y = E0 + EMAX · effect^γ / (EC50^γ + effect^γ)
    // Mapped to slots: θ_0=E0, θ_1=EMAX, θ_2=EC50, θ_3=γ; var_0=effect.
    // This is the Emax PK/PD readout shape the Tier 4a Form C codegen
    // (milestone 4) would have chain-ruled through. The codegen was
    // reverted in #145, but the differentiator end-to-end behaviour
    // on this shape is still worth pinning for any future consumer.
    let theta = &[3.0, 60.0, 7.5, 1.5];
    let eta = &[];
    let vars = &[2.0];
    let covs = &[];
    let eff = Expression::VariableIdx(0);
    let e0 = Expression::Theta(0);
    let emax = Expression::Theta(1);
    let ec50 = Expression::Theta(2);
    let gamma = Expression::Theta(3);
    let eff_g = power(eff.clone(), gamma.clone());
    let ec_g = power(ec50.clone(), gamma);
    let denom = binop(BinOp::Add, ec_g, eff_g.clone());
    let frac = binop(BinOp::Div, eff_g, denom);
    let expr = binop(BinOp::Add, e0, binop(BinOp::Mul, emax, frac));
    for axis in [
        DiffAxis::Theta(0),    // ∂y/∂E0 = 1
        DiffAxis::Theta(1),    // ∂y/∂EMAX = frac
        DiffAxis::Theta(2),    // ∂y/∂EC50 — the awkward one
        DiffAxis::Theta(3),    // ∂y/∂γ
        DiffAxis::Variable(0), // ∂y/∂effect — the sensitivity-ODE input
    ] {
        assert_diff_matches_fd(expr.clone(), axis, theta, eta, vars, covs);
    }
}

#[test]
fn simplify_collapses_zero_and_one() {
    // 0 + (1 * x) → x
    let raw = binop(
        BinOp::Add,
        lit(0.0),
        binop(BinOp::Mul, lit(1.0), Expression::Theta(0)),
    );
    let simp = simplify_expr(&raw);
    assert!(matches!(simp, Expression::Theta(0)));
    // x * 0 → 0
    let raw = binop(BinOp::Mul, Expression::VariableIdx(2), lit(0.0));
    assert!(matches!(simplify_expr(&raw), Expression::Literal(v) if v == 0.0));
    // x - 0 → x
    let raw = binop(BinOp::Sub, Expression::Eta(0), lit(0.0));
    assert!(matches!(simplify_expr(&raw), Expression::Eta(0)));
    // 0 / x → 0
    let raw = binop(BinOp::Div, lit(0.0), Expression::Theta(1));
    assert!(matches!(simplify_expr(&raw), Expression::Literal(v) if v == 0.0));
}

// --- Milestone 2: `[individual_parameters]` partials ---
//
// Each test parses a small model and asserts the precomputed
// `IndivParamPartials` rows agree with central FD on the pk_param_fn
// output. We use the parser's `parse_full_model` entry point so the test
// exercises the full pipeline: parsing → resolve → differentiate →
// simplify → store on CompiledModel.

/// Numerically evaluate a stored partial Expression with the same inputs
/// the pk_param_fn closure sees. Returns the partial's value at
/// (theta, eta, covariates), with `vars` cleared to zero (intermediates
/// aren't realized — they show up in the partial expression through
/// chain-rule substitution at differentiation time, so the eval here
/// doesn't need to recompute them).
fn eval_partial(
    partial: &Expression,
    theta: &[f64],
    eta: &[f64],
    cov: &HashMap<String, f64>,
    var_count: usize,
) -> f64 {
    let mut cov_vec = vec![0.0_f64; cov.len()];
    // Build a deterministic cov ordering matching the partial's
    // CovariateIdx references. For these tests we use NO covariates.
    let mut keys: Vec<&String> = cov.keys().collect();
    keys.sort();
    for (i, k) in keys.iter().enumerate() {
        cov_vec[i] = cov[k.as_str()];
    }
    let mut vars = vec![0.0_f64; var_count];
    let nn_outputs: Vec<Vec<f64>> = Vec::new();
    eval_expression_indexed(partial, theta, eta, &cov_vec, &mut vars, &nn_outputs)
}

/// Central-FD reference for ∂(pk_param_fn output at slot i)/∂θ_k.
fn fd_d_theta(
    model: &crate::types::CompiledModel,
    i: usize,
    k: usize,
    theta: &[f64],
    eta: &[f64],
    cov: &HashMap<String, f64>,
) -> f64 {
    let h = 1e-5 * theta[k].abs().max(1.0);
    let mut tp = theta.to_vec();
    let mut tm = theta.to_vec();
    tp[k] += h;
    tm[k] -= h;
    let pk_plus = (model.pk_param_fn)(&tp, eta, cov, 0.0);
    let pk_minus = (model.pk_param_fn)(&tm, eta, cov, 0.0);
    let slot = model.pk_indices[i];
    (pk_plus.values[slot] - pk_minus.values[slot]) / (2.0 * h)
}

/// Central-FD reference for ∂(pk_param_fn output at slot i)/∂η_k.
fn fd_d_eta(
    model: &crate::types::CompiledModel,
    i: usize,
    k: usize,
    theta: &[f64],
    eta: &[f64],
    cov: &HashMap<String, f64>,
) -> f64 {
    let h = 1e-5_f64.max(eta[k].abs() * 1e-5);
    let mut ep = eta.to_vec();
    let mut em = eta.to_vec();
    ep[k] += h;
    em[k] -= h;
    let pk_plus = (model.pk_param_fn)(theta, &ep, cov, 0.0);
    let pk_minus = (model.pk_param_fn)(theta, &em, cov, 0.0);
    let slot = model.pk_indices[i];
    (pk_plus.values[slot] - pk_minus.values[slot]) / (2.0 * h)
}

#[test]
fn indiv_partials_populated_for_1cpt_oral() {
    // CL = TVCL * exp(ETA_CL), V = TVV * exp(ETA_V), KA = TVKA * exp(ETA_KA)
    // Three indiv params, three θ, three η, all flat (no chain).
    let model_str = "
[parameters]
  theta TVCL(7.5)
  theta TVV(75.0)
  theta TVKA(1.05)
  omega ETA_CL ~ 0.135
  omega ETA_V  ~ 0.135
  omega ETA_KA ~ 0.225
  sigma EPS ~ 0.245

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(EPS)
";
    let parsed = super::parse_full_model(model_str).unwrap();
    let m = &parsed.model;
    let partials = &m.indiv_param_partials;
    assert_eq!(partials.names, vec!["CL", "V", "KA"]);
    assert_eq!(partials.d_d_theta.len(), 3);
    assert_eq!(partials.d_d_eta.len(), 3);
    // Each row has length n_theta_base (3) and n_eta_extended (3).
    for row in &partials.d_d_theta {
        assert_eq!(row.len(), 3);
    }
    for row in &partials.d_d_eta {
        assert_eq!(row.len(), 3);
    }

    // FD verification at off-zero η so exp(ETA) ≠ 1 and we exercise
    // the chain through exp.
    let theta = &[7.5, 75.0, 1.05];
    let eta = &[0.1, -0.2, 0.05];
    let cov: HashMap<String, f64> = HashMap::new();
    for i in 0..3 {
        for k in 0..3 {
            let sym = eval_partial(&partials.d_d_theta[i][k], theta, eta, &cov, 8);
            let fd = fd_d_theta(m, i, k, theta, eta, &cov);
            assert!(
                (sym - fd).abs() < 1e-6 * sym.abs().max(fd.abs()).max(1e-8),
                "∂{}/∂θ_{}: sym={sym}, fd={fd}",
                partials.names[i],
                k,
            );
            let sym = eval_partial(&partials.d_d_eta[i][k], theta, eta, &cov, 8);
            let fd = fd_d_eta(m, i, k, theta, eta, &cov);
            assert!(
                (sym - fd).abs() < 1e-6 * sym.abs().max(fd.abs()).max(1e-8),
                "∂{}/∂η_{}: sym={sym}, fd={fd}",
                partials.names[i],
                k,
            );
        }
    }
}

#[test]
fn indiv_partials_zero_for_cross_axis() {
    // ∂CL/∂η_V = 0, ∂V/∂η_CL = 0. The simplified expressions should be
    // Literal(0.0) (or a tree that evaluates to 0).
    let model_str = "
[parameters]
  theta TVCL(7.5)
  theta TVV(75.0)
  theta TVKA(1.05)
  omega ETA_CL ~ 0.135
  omega ETA_V  ~ 0.135
  omega ETA_KA ~ 0.225
  sigma EPS ~ 0.245

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(EPS)
";
    let parsed = super::parse_full_model(model_str).unwrap();
    let partials = &parsed.model.indiv_param_partials;
    // After simplify, ∂CL/∂η_V is exactly Literal(0). Even if a future
    // simplifier rule leaves it as a tree, eval must produce 0.0.
    let theta = &[7.5, 75.0, 1.05];
    let eta = &[0.3, -0.1, 0.2];
    let cov: HashMap<String, f64> = HashMap::new();
    // Cross-η partials (indiv i, η k ≠ i).
    for (i, k) in [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)] {
        let v = eval_partial(&partials.d_d_eta[i][k], theta, eta, &cov, 8);
        assert!(
            v.abs() < 1e-12,
            "∂{}/∂η_{} should be 0, got {v}",
            partials.names[i],
            k
        );
    }
    // Cross-θ partials.
    for (i, k) in [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)] {
        let v = eval_partial(&partials.d_d_theta[i][k], theta, eta, &cov, 8);
        assert!(
            v.abs() < 1e-12,
            "∂{}/∂θ_{} should be 0, got {v}",
            partials.names[i],
            k
        );
    }
}

#[test]
fn indiv_partials_logit_inv_logit_chain() {
    // F = inv_logit(logit(THETA_F) + ETA_F)
    // Exercises the inv_logit/logit derivative rules end-to-end.
    let model_str = "
[parameters]
  theta TVCL(7.5)
  theta TVV(75.0)
  theta TVKA(1.05)
  theta THETA_F(0.6)
  omega ETA_CL ~ 0.135
  omega ETA_V  ~ 0.135
  omega ETA_KA ~ 0.225
  omega ETA_F  ~ 0.1
  sigma EPS ~ 0.245

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
  F  = inv_logit(logit(THETA_F) + ETA_F)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, f=F)

[error_model]
  DV ~ proportional(EPS)
";
    let parsed = super::parse_full_model(model_str).unwrap();
    let m = &parsed.model;
    let partials = &m.indiv_param_partials;
    assert_eq!(partials.names, vec!["CL", "V", "KA", "F"]);
    let theta = &[7.5, 75.0, 1.05, 0.6];
    let eta = &[0.05, -0.1, 0.15, 0.2];
    let cov: HashMap<String, f64> = HashMap::new();
    // F's row: position 3. Verify ∂F/∂η_F (index 3) and ∂F/∂THETA_F (index 3).
    let sym = eval_partial(&partials.d_d_eta[3][3], theta, eta, &cov, 8);
    let fd = fd_d_eta(m, 3, 3, theta, eta, &cov);
    assert!(
        (sym - fd).abs() < 1e-6 * sym.abs().max(fd.abs()).max(1e-8),
        "∂F/∂η_F: sym={sym}, fd={fd}",
    );
    let sym = eval_partial(&partials.d_d_theta[3][3], theta, eta, &cov, 8);
    let fd = fd_d_theta(m, 3, 3, theta, eta, &cov);
    assert!(
        (sym - fd).abs() < 1e-6 * sym.abs().max(fd.abs()).max(1e-8),
        "∂F/∂THETA_F: sym={sym}, fd={fd}",
    );
    // F doesn't depend on ETA_CL/V/KA — those partials must evaluate to 0.
    for k in 0..3 {
        let v = eval_partial(&partials.d_d_eta[3][k], theta, eta, &cov, 8);
        assert!(v.abs() < 1e-12, "∂F/∂η_{k} should be 0, got {v}");
    }
}

#[test]
fn indiv_partials_chain_rule_through_intermediate() {
    // Synthetic model exercising the chain-rule substitution at a
    // VariableIdx leaf — `KA` references `ka_mult` (an earlier
    // intermediate) by name. The differentiator must chain-rule through
    // ka_mult's η_KAM dependence into KA's row.
    //
    // ka_mult = TVKAM * exp(ETA_KAM)
    // KA      = TVKA  * ka_mult     (depends on η_KAM via ka_mult)
    //
    // ∂KA/∂η_KAM = TVKA * ∂ka_mult/∂η_KAM = TVKA * TVKAM * exp(ETA_KAM)
    // ∂KA/∂η_KA  = 0
    // ∂KA/∂TVKAM = TVKA * exp(ETA_KAM)
    let model_str = "
[parameters]
  theta TVCL(7.5)
  theta TVV(75.0)
  theta TVKA(1.05)
  theta TVKAM(2.0)
  omega ETA_CL  ~ 0.135
  omega ETA_V   ~ 0.135
  omega ETA_KAM ~ 0.1
  sigma EPS ~ 0.245

[individual_parameters]
  CL      = TVCL * exp(ETA_CL)
  V       = TVV  * exp(ETA_V)
  ka_mult = TVKAM * exp(ETA_KAM)
  KA      = TVKA * ka_mult

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(EPS)
";
    let parsed = super::parse_full_model(model_str).unwrap();
    let m = &parsed.model;
    let partials = &m.indiv_param_partials;
    // ka_mult IS a top-level Assign so it has a row; ordering must match
    // declaration order.
    assert_eq!(partials.names, vec!["CL", "V", "ka_mult", "KA"]);
    let theta = &[7.5, 75.0, 1.05, 2.0];
    let eta = &[0.0, 0.0, 0.3]; // η_CL=0, η_V=0, η_KAM=0.3
    let cov: HashMap<String, f64> = HashMap::new();
    // KA is at indiv-param index 3, η_KAM is at eta index 2.
    let sym = eval_partial(&partials.d_d_eta[3][2], theta, eta, &cov, 8);
    let expected = theta[2] * theta[3] * eta[2].exp(); // TVKA * TVKAM * exp(ETA_KAM)
    assert!(
        (sym - expected).abs() < 1e-12 * expected.abs().max(1.0),
        "∂KA/∂η_KAM (chain rule): sym={sym}, expected={expected}",
    );
    // FD cross-check via pk_param_fn:
    let fd = fd_d_eta(m, 3, 2, theta, eta, &cov);
    assert!(
        (sym - fd).abs() < 1e-6 * sym.abs().max(fd.abs()).max(1e-8),
        "∂KA/∂η_KAM: sym={sym}, fd={fd}",
    );
    // ∂KA/∂η_CL = 0 (no chain to ETA_CL).
    let v = eval_partial(&partials.d_d_eta[3][0], theta, eta, &cov, 8);
    assert!(v.abs() < 1e-12, "∂KA/∂η_CL should be 0, got {v}");
    // ∂KA/∂TVKAM (θ index 3): TVKA * exp(ETA_KAM).
    let sym = eval_partial(&partials.d_d_theta[3][3], theta, eta, &cov, 8);
    let expected = theta[2] * eta[2].exp();
    assert!(
        (sym - expected).abs() < 1e-12 * expected.abs().max(1.0),
        "∂KA/∂TVKAM: sym={sym}, expected={expected}",
    );
}

// ── unused-parameter warning tests ──────────────────────────────────────

fn minimal_model(parameters: &str, individual_parameters: &str, error_model: &str) -> String {
    format!(
        r#"
[parameters]
{parameters}

[individual_parameters]
{individual_parameters}

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
DV ~ proportional({error_model})

[fit_options]
method = foce
"#
    )
}

#[test]
fn test_all_used_no_warnings() {
    let src = minimal_model(
        "theta TVCL(0.1)\nomega ETA_CL ~ 0.09\nsigma PROP ~ 0.01",
        "CL = TVCL * exp(ETA_CL)\nV = 10.0\nKA = 1.0",
        "PROP",
    );
    let parsed = parse_full_model(&src).expect("parse ok");
    assert!(
        parsed.model.parse_warnings.is_empty(),
        "unexpected warnings: {:?}",
        parsed.model.parse_warnings
    );
}

#[test]
fn test_unused_theta_warns() {
    let src = minimal_model(
        "theta TVCL(0.1)\ntheta UNUSED(0.5)\nomega ETA_CL ~ 0.09\nsigma PROP ~ 0.01",
        "CL = TVCL * exp(ETA_CL)\nV = 10.0\nKA = 1.0",
        "PROP",
    );
    let parsed = parse_full_model(&src).expect("parse ok");
    let warns: Vec<_> = parsed
        .model
        .parse_warnings
        .iter()
        .filter(|w| w.contains("UNUSED"))
        .collect();
    assert_eq!(
        warns.len(),
        1,
        "expected exactly one warning for UNUSED theta"
    );
    assert!(warns[0].contains("theta"), "warning should mention 'theta'");
}

#[test]
fn test_unused_omega_warns() {
    let src = minimal_model(
        "theta TVCL(0.1)\nomega ETA_CL ~ 0.09\nomega ETA_UNUSED ~ 0.04\nsigma PROP ~ 0.01",
        "CL = TVCL * exp(ETA_CL)\nV = 10.0\nKA = 1.0",
        "PROP",
    );
    let parsed = parse_full_model(&src).expect("parse ok");
    let warns: Vec<_> = parsed
        .model
        .parse_warnings
        .iter()
        .filter(|w| w.contains("ETA_UNUSED"))
        .collect();
    assert_eq!(
        warns.len(),
        1,
        "expected exactly one warning for ETA_UNUSED omega"
    );
    assert!(warns[0].contains("omega"), "warning should mention 'omega'");
}

#[test]
fn test_unused_sigma_warns() {
    let src = minimal_model(
        "theta TVCL(0.1)\nomega ETA_CL ~ 0.09\nsigma PROP ~ 0.01\nsigma ADD_UNUSED ~ 0.01",
        "CL = TVCL * exp(ETA_CL)\nV = 10.0\nKA = 1.0",
        "PROP",
    );
    let parsed = parse_full_model(&src).expect("parse ok");
    let warns: Vec<_> = parsed
        .model
        .parse_warnings
        .iter()
        .filter(|w| w.contains("ADD_UNUSED"))
        .collect();
    assert_eq!(
        warns.len(),
        1,
        "expected exactly one warning for ADD_UNUSED sigma"
    );
    assert!(warns[0].contains("sigma"), "warning should mention 'sigma'");
}

#[test]
fn test_commented_out_usage_warns() {
    // Simulates: CL = TVCL #* exp(ETA_CL) — ETA_CL is commented away
    let src = minimal_model(
        "theta TVCL(0.1)\nomega ETA_CL ~ 0.09\nsigma PROP ~ 0.01",
        "CL = TVCL\nV = 10.0\nKA = 1.0",
        "PROP",
    );
    let parsed = parse_full_model(&src).expect("parse ok");
    let warns: Vec<_> = parsed
        .model
        .parse_warnings
        .iter()
        .filter(|w| w.contains("ETA_CL"))
        .collect();
    assert_eq!(warns.len(), 1, "ETA_CL not used → should warn");
}

#[test]
fn test_multiple_unused_all_reported() {
    let src = minimal_model(
            "theta TVCL(0.1)\ntheta TH_UNUSED(0.5)\nomega ETA_CL ~ 0.09\nomega ETA_UNUSED ~ 0.04\nsigma PROP ~ 0.01\nsigma SIG_UNUSED ~ 0.01",
            "CL = TVCL * exp(ETA_CL)\nV = 10.0\nKA = 1.0",
            "PROP",
        );
    let parsed = parse_full_model(&src).expect("parse ok");
    let unused: Vec<_> = parsed
        .model
        .parse_warnings
        .iter()
        .filter(|w| w.contains("UNUSED"))
        .collect();
    assert_eq!(
        unused.len(),
        3,
        "expected warnings for TH_UNUSED, ETA_UNUSED, SIG_UNUSED; got: {:?}",
        unused
    );
}

#[test]
fn test_unused_kappa_warns() {
    // KAPPA_CL declared but not used in any expression (index arithmetic:
    // Eta(n_eta + i), distinct from BSV etas).
    let src = format!(
        r#"
[parameters]
  theta TVCL(0.1)
  omega ETA_CL ~ 0.09
  kappa KAPPA_CL ~ 0.01
  sigma PROP ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = 10.0
  KA = 1.0

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP)

[fit_options]
  method = foce
  iov_column = OCC
"#
    );
    let parsed = parse_full_model(&src).expect("parse ok");
    let warns: Vec<_> = parsed
        .model
        .parse_warnings
        .iter()
        .filter(|w| w.contains("KAPPA_CL"))
        .collect();
    assert_eq!(
        warns.len(),
        1,
        "expected one warning for unused KAPPA_CL; got: {:?}",
        warns
    );
    assert!(warns[0].contains("kappa"), "warning should mention 'kappa'");
}

#[test]
fn test_derived_variable_no_false_positive() {
    // ke = CL / V is a derived variable — no theta/eta directly.
    // But CL = TVCL * exp(ETA_CL) and V = TVV * exp(ETA_V) ARE in
    // indiv_stmts and ARE walked, so TVCL/ETA_CL/TVV/ETA_V are found
    // through those statements. No spurious "unused" warnings expected.
    let src = format!(
        r#"
[parameters]
  theta TVCL(0.1)
  theta TVV(10.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.04
  sigma PROP ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  ke = CL / V
  KA = 1.0

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP)

[fit_options]
  method = foce
"#
    );
    let parsed = parse_full_model(&src).expect("parse ok");
    // `ke = CL/V` is itself an unused intermediate, now flagged by the
    // computed-but-unused check (#309). This test's point is narrower: the
    // thetas/etas it references (TVCL, TVV, ETA_CL, ETA_V) must NOT be falsely
    // reported as unused.
    assert!(
        !parsed
            .model
            .parse_warnings
            .iter()
            .any(|w| w.contains("not referenced in any model expression")),
        "thetas/etas used via the derived variable ke=CL/V must not be flagged \
             unused; got: {:?}",
        parsed.model.parse_warnings
    );
}

#[test]
fn test_theta_used_only_in_conditional_branch_no_warn() {
    // WT_POW is only referenced inside an if-branch — it must still be
    // found because collect_theta_eta_in_stmts recurses into if-bodies.
    let src = format!(
        r#"
[parameters]
  theta TVCL(0.1)
  theta WT_POW(0.75)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.01

[individual_parameters]
  if (WT > 0) {{
    CL = TVCL * (WT / 70)^WT_POW * exp(ETA_CL)
  }} else {{
    CL = TVCL * exp(ETA_CL)
  }}
  V  = 10.0
  KA = 1.0

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP)

[fit_options]
  method = foce
"#
    );
    let parsed = parse_full_model(&src).expect("parse ok");
    // The conditional model legitimately triggers a mu-referencing warning
    // for CL — that is unrelated to our check. Assert only that no
    // unused-parameter warning is emitted for WT_POW or any other parameter.
    let unused: Vec<_> = parsed
        .model
        .parse_warnings
        .iter()
        .filter(|w| w.contains("declared in [parameters] but not referenced"))
        .collect();
    assert!(
        unused.is_empty(),
        "WT_POW used inside if-branch must not trigger unused-param warning; got: {:?}",
        unused
    );
}

#[test]
fn test_block_omega_one_unused_warns() {
    // block_omega declares ETA_CL and ETA_V together; only ETA_CL is used.
    // ETA_V should produce a warning even though it's part of a block.
    let src = format!(
        r#"
[parameters]
  theta TVCL(0.1)
  block_omega (ETA_CL, ETA_V) = [0.09, 0.01, 0.04]
  sigma PROP ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = 10.0
  KA = 1.0

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP)

[fit_options]
  method = foce
"#
    );
    let parsed = parse_full_model(&src).expect("parse ok");
    let warns: Vec<_> = parsed
        .model
        .parse_warnings
        .iter()
        .filter(|w| w.contains("ETA_V"))
        .collect();
    assert_eq!(
        warns.len(),
        1,
        "ETA_V unused in block_omega should warn; got: {:?}",
        warns
    );
    assert!(warns[0].contains("omega"), "warning should mention 'omega'");
    // ETA_CL IS used — must not appear in warnings
    assert!(
        !parsed
            .model
            .parse_warnings
            .iter()
            .any(|w| w.contains("ETA_CL")),
        "ETA_CL is used and must not warn"
    );
}

#[test]
fn test_block_omega_all_used_no_warn() {
    // Both etas in a block_omega are used — no warnings expected.
    let src = format!(
        r#"
[parameters]
  theta TVCL(0.1)
  theta TVV(10.0)
  block_omega (ETA_CL, ETA_V) = [0.09, 0.01, 0.04]
  sigma PROP ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = 1.0

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP)

[fit_options]
  method = foce
"#
    );
    let parsed = parse_full_model(&src).expect("parse ok");
    assert!(
        parsed.model.parse_warnings.is_empty(),
        "all block_omega etas used — no warnings expected; got: {:?}",
        parsed.model.parse_warnings
    );
}

// ── [derived] block parser unit tests ────────────────────────────────────

fn minimal_model_with_derived(derived_block: &str) -> String {
    format!(
        r#"
[parameters]
  theta CL(1.0, 0, 100)
  theta V(10.0, 0, 1000)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP   ~ 0.01

[individual_parameters]
  CL = exp(log(CL) + ETA_CL)
  V  = exp(log(V)  + ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)

[derived]
{derived_block}
"#
    )
}

fn make_derived_ctx_simple() -> (
    Vec<f64>,
    Vec<f64>,
    std::collections::HashMap<String, f64>,
    std::collections::HashMap<String, f64>,
    std::collections::HashMap<String, f64>,
) {
    let theta = vec![1.0, 10.0];
    let eta = vec![0.0, 0.0];
    let mut indiv_params = std::collections::HashMap::new();
    indiv_params.insert("CL".to_string(), 1.0);
    indiv_params.insert("V".to_string(), 10.0);
    let covariates = std::collections::HashMap::new();
    let prev_derived = std::collections::HashMap::new();
    (theta, eta, indiv_params, covariates, prev_derived)
}

#[test]
fn parse_derived_per_row() {
    let src = minimal_model_with_derived("KE = CL / V");
    let parsed = parse_full_model(&src).expect("parse ok");
    assert_eq!(parsed.model.derived_exprs.len(), 1);
    assert_eq!(parsed.model.derived_exprs[0].name, "KE");
    assert!(matches!(
        parsed.model.derived_exprs[0].kind,
        DerivedKind::PerRow { .. }
    ));

    // Evaluate the closure
    let (theta, eta, indiv_params, covariates, prev_derived) = make_derived_ctx_simple();
    if let DerivedKind::PerRow { eval } = &parsed.model.derived_exprs[0].kind {
        let ctx = DerivedContext {
            theta: &theta,
            eta: &eta,
            indiv_params: &indiv_params,
            covariates: &covariates,
            ipred: 0.0,
            pred: 0.0,
            dv: 0.0,
            time: 1.0,
            tafd: 1.0,
            tad: 1.0,
            prev_derived: &prev_derived,
            compartments: &[],
            compartment_names: &[],
        };
        let ke = eval(&ctx);
        assert!((ke - 0.1).abs() < 1e-10, "KE = CL/V = 1/10 = 0.1, got {ke}");
    } else {
        panic!("expected PerRow");
    }
}

/// `min`/`max` at a `[derived]` statement's top level used to be read *only* as the
/// row aggregate, so the two-argument numeric form (#1030) meant one thing nested
/// (`1 * max(CL, 2)`) and failed outright unnested. A second argument with no
/// comparison operator cannot be a row filter, so both spellings now agree.
#[test]
fn parse_derived_two_arg_max_at_statement_top_level() {
    let (theta, eta, indiv_params, covariates, prev_derived) = make_derived_ctx_simple();
    let eval_one = |derived: &str| {
        let src = minimal_model_with_derived(derived);
        let parsed = parse_full_model(&src).expect("two-arg max at top level parses");
        let DerivedKind::PerRow { eval } = &parsed.model.derived_exprs[0].kind else {
            panic!("expected PerRow, got the row aggregate");
        };
        let ctx = DerivedContext {
            theta: &theta,
            eta: &eta,
            indiv_params: &indiv_params,
            covariates: &covariates,
            ipred: 0.0,
            pred: 0.0,
            dv: 0.0,
            time: 1.0,
            tafd: 1.0,
            tad: 1.0,
            prev_derived: &prev_derived,
            compartments: &[],
            compartment_names: &[],
        };
        eval(&ctx)
    };
    // CL = 1: the floor bites at top level exactly as it does nested.
    assert_eq!(eval_one("X = max(CL, 2.0)"), 2.0);
    assert_eq!(eval_one("X = 1 * max(CL, 2.0)"), 2.0);
    assert_eq!(eval_one("X = min(CL, 2.0)"), 1.0);
    // Still an expression, not a call: trailing operators are honoured.
    assert_eq!(eval_one("X = max(CL, 2.0) * 3"), 6.0);
}

/// `clamp` has no row-aggregate spelling, so it is unambiguous in `[derived]`:
/// it means the same thing at statement top level as it does nested one level
/// in, and it stays an expression that trailing operators continue.
#[test]
fn parse_derived_clamp_is_unambiguous() {
    let (theta, eta, indiv_params, covariates, prev_derived) = make_derived_ctx_simple();
    let eval_one = |derived: &str| {
        let src = minimal_model_with_derived(derived);
        let parsed = parse_full_model(&src).expect("clamp in [derived] parses");
        let DerivedKind::PerRow { eval } = &parsed.model.derived_exprs[0].kind else {
            panic!("expected PerRow, got the row aggregate");
        };
        let ctx = DerivedContext {
            theta: &theta,
            eta: &eta,
            indiv_params: &indiv_params,
            covariates: &covariates,
            ipred: 0.0,
            pred: 0.0,
            dv: 0.0,
            time: 1.0,
            tafd: 1.0,
            tad: 1.0,
            prev_derived: &prev_derived,
            compartments: &[],
            compartment_names: &[],
        };
        eval(&ctx)
    };
    // CL = 1, V = 10.
    assert_eq!(eval_one("X = clamp(CL, 2.0, 5.0)"), 2.0);
    assert_eq!(eval_one("X = 1 * clamp(CL, 2.0, 5.0)"), 2.0);
    assert_eq!(eval_one("X = clamp(V, 2.0, 5.0)"), 5.0);
    assert_eq!(eval_one("X = clamp(V, 2.0, 50.0)"), 10.0);
    // Still an expression, not a call: trailing operators are honoured.
    assert_eq!(eval_one("X = clamp(CL, 2.0, 5.0) * 3"), 6.0);
}

/// A third argument to a statement-top-level `[derived]` aggregate used to be
/// dropped in silence — `args[2..]` is never read — so `max(IPRED, TIME > 0, Q)`
/// computed `max(IPRED, TIME > 0)`. Reject it, and name `clamp` for `min`/`max`,
/// which is what the extra argument usually means (#1092).
#[test]
fn parse_derived_aggregate_rejects_a_third_argument() {
    for (derived, names_clamp) in [
        ("CMAX = max(IPRED, TIME > 0, 99)", true),
        ("CMIN = min(IPRED, TIME > 0, 99)", true),
        ("TP = tmax(IPRED, TIME > 0, 99)", false),
    ] {
        let src = minimal_model_with_derived(derived);
        let err = parse_full_model(&src)
            .err()
            .unwrap_or_else(|| panic!("a third argument must be rejected: `{derived}`"));
        assert!(
            err.contains("at most two arguments"),
            "expected the aggregate arity error for `{derived}`, got: {err}"
        );
        assert_eq!(
            err.contains("clamp(x, lo, hi)"),
            names_clamp,
            "only `min`/`max` should point at clamp: `{derived}` gave: {err}"
        );
    }
}

/// A second argument that *is* a comparison keeps meaning the row filter.
#[test]
fn parse_derived_aggregate_filter_still_parses() {
    let src = minimal_model_with_derived("CMAX = max(IPRED, TIME > 0)");
    let parsed = parse_full_model(&src).expect("aggregate with a row filter parses");
    assert!(matches!(
        parsed.model.derived_exprs[0].kind,
        DerivedKind::Aggregate { .. }
    ));
}

/// Trailing tokens after a row aggregate's `)` used to be dropped in silence, so
/// `max(IPRED) * 2` computed `max(IPRED)`. No aggregate form continues into a
/// larger expression, so it is now an error rather than a wrong number.
#[test]
fn parse_derived_aggregate_rejects_trailing_tokens() {
    let src = minimal_model_with_derived("CMAX = max(IPRED) * 2");
    let err = match parse_full_model(&src) {
        Ok(_) => panic!("trailing tokens after an aggregate must error"),
        Err(e) => e,
    };
    assert!(
        err.contains("unexpected token(s) after"),
        "expected the trailing-token error, got: {}",
        err
    );
}

#[test]
fn parse_derived_sequential_reference() {
    let src = minimal_model_with_derived("KE = CL / V\nT_HALF = 0.693 / KE");
    let parsed = parse_full_model(&src).expect("parse ok");
    assert_eq!(parsed.model.derived_exprs.len(), 2);

    let (theta, eta, indiv_params, covariates, _) = make_derived_ctx_simple();
    let mut prev_derived = std::collections::HashMap::new();

    // Evaluate KE first
    if let DerivedKind::PerRow { eval } = &parsed.model.derived_exprs[0].kind {
        let ctx = DerivedContext {
            theta: &theta,
            eta: &eta,
            indiv_params: &indiv_params,
            covariates: &covariates,
            ipred: 0.0,
            pred: 0.0,
            dv: 0.0,
            time: 1.0,
            tafd: 1.0,
            tad: 1.0,
            prev_derived: &prev_derived,
            compartments: &[],
            compartment_names: &[],
        };
        let ke = eval(&ctx);
        prev_derived.insert("KE".to_string(), ke);
    }

    // Now evaluate T_HALF using prev_derived
    if let DerivedKind::PerRow { eval } = &parsed.model.derived_exprs[1].kind {
        let ctx = DerivedContext {
            theta: &theta,
            eta: &eta,
            indiv_params: &indiv_params,
            covariates: &covariates,
            ipred: 0.0,
            pred: 0.0,
            dv: 0.0,
            time: 1.0,
            tafd: 1.0,
            tad: 1.0,
            prev_derived: &prev_derived,
            compartments: &[],
            compartment_names: &[],
        };
        let t_half = eval(&ctx);
        let expected = 0.693 / 0.1;
        assert!(
            (t_half - expected).abs() < 1e-8,
            "T_HALF = 0.693/KE = {expected}, got {t_half}"
        );
    } else {
        panic!("expected PerRow for T_HALF");
    }
}

#[test]
fn parse_derived_max_with_filter() {
    let src = minimal_model_with_derived("CMAX = max(IPRED, TIME < 24)");
    let parsed = parse_full_model(&src).expect("parse ok");
    assert!(matches!(
        parsed.model.derived_exprs[0].kind,
        DerivedKind::Aggregate {
            func: AggFunction::Max,
            filter: Some(_),
            ..
        }
    ));
}

#[test]
fn parse_derived_min_no_filter() {
    let src = minimal_model_with_derived("CMIN = min(IPRED)");
    let parsed = parse_full_model(&src).expect("parse ok");
    assert!(matches!(
        parsed.model.derived_exprs[0].kind,
        DerivedKind::Aggregate {
            func: AggFunction::Min,
            filter: None,
            ..
        }
    ));
}

#[test]
fn parse_derived_tmax() {
    let src = minimal_model_with_derived("TMAX = tmax(IPRED)");
    let parsed = parse_full_model(&src).expect("parse ok");
    assert!(matches!(
        parsed.model.derived_exprs[0].kind,
        DerivedKind::Aggregate {
            func: AggFunction::Tmax,
            filter: None,
            ..
        }
    ));
}

#[test]
fn parse_derived_integral_explicit() {
    let src = minimal_model_with_derived("AUC = integral(IPRED, from=0, to=24)");
    let parsed = parse_full_model(&src).expect("parse ok");
    if let DerivedKind::Integral {
        data_based,
        window: IntegralWindow::Explicit { from, to },
        step: IntegralStep::Auto,
        ..
    } = &parsed.model.derived_exprs[0].kind
    {
        assert!(!data_based, "IPRED is not DV-based");
        assert!((from - 0.0).abs() < 1e-10);
        assert!((to - 24.0).abs() < 1e-10);
    } else {
        panic!(
            "expected Integral(Explicit, Auto), got {:?} kind",
            parsed.model.derived_exprs[0].name
        );
    }
}

#[test]
fn parse_derived_integral_dv() {
    let src = minimal_model_with_derived("AUC_DV = integral(DV, from=0, to=24)");
    let parsed = parse_full_model(&src).expect("parse ok");
    if let DerivedKind::Integral {
        data_based,
        step: IntegralStep::ObsTimes,
        ..
    } = &parsed.model.derived_exprs[0].kind
    {
        assert!(*data_based, "DV is data_based");
    } else {
        panic!("expected Integral with ObsTimes step for DV integrand");
    }
}

#[test]
fn parse_derived_integral_periodic() {
    let src = minimal_model_with_derived("AUC_TAU = integral(IPRED, window=24, anchor=0)");
    let parsed = parse_full_model(&src).expect("parse ok");
    if let DerivedKind::Integral {
        window: IntegralWindow::Periodic { period, anchor },
        ..
    } = &parsed.model.derived_exprs[0].kind
    {
        assert!((period - 24.0).abs() < 1e-10);
        assert!((anchor - 0.0).abs() < 1e-10);
    } else {
        panic!("expected Integral(Periodic)");
    }
}

#[test]
fn parse_derived_name_conflict_error() {
    // IPRED is a built-in sdtab column — should error
    let src = minimal_model_with_derived("IPRED = CL / V");
    let result = parse_full_model(&src);
    assert!(
        result.is_err(),
        "expected parse error for IPRED name conflict"
    );
    let msg = match result {
        Err(e) => e,
        Ok(_) => panic!("expected Err"),
    };
    assert!(
        msg.contains("E_DERIVED_NAME_CONFLICT"),
        "expected E_DERIVED_NAME_CONFLICT in error, got: {msg}"
    );
}

#[test]
fn parse_output_block() {
    let src = format!(
        r#"{}
[output]
CL V KA WT
"#,
        minimal_model_with_derived("")
    );
    let parsed = parse_full_model(&src).expect("parse ok");
    assert_eq!(
        parsed.model.output_columns,
        vec!["CL", "V", "KA", "WT"],
        "output_columns mismatch"
    );
}

#[test]
fn parse_output_empty_block_ok() {
    let src = format!(
        r#"{}
[output]
"#,
        minimal_model_with_derived("")
    );
    let parsed = parse_full_model(&src).expect("parse ok");
    assert!(
        parsed.model.output_columns.is_empty(),
        "empty [output] block should produce empty output_columns"
    );
}

#[test]
fn sci_notation_negative_exp() {
    let src = minimal_model_with_derived("FLAG = if (TAD < 1e-10) 1 else 0");
    let parsed = parse_full_model(&src).expect("parse ok — 1e-10 must tokenise correctly");
    assert_eq!(parsed.model.derived_exprs.len(), 1);
}

#[test]
fn sci_notation_positive_exp() {
    let src = minimal_model_with_derived("FLAG = if (TAFD > 1.5E+3) 1 else 0");
    let parsed = parse_full_model(&src).expect("parse ok — 1.5E+3 must tokenise correctly");
    assert_eq!(parsed.model.derived_exprs.len(), 1);
}

#[test]
fn mod_operator_euclidean() {
    // 5 mod 2 == 1, 7 mod 3 == 1, -1 mod 24 == 23
    let tests = [("5 mod 2", 1.0), ("7 mod 3", 1.0), ("-1 mod 24", 23.0)];
    for (expr_str, expected) in &tests {
        let expr_src = format!("VAL = {expr_str}");
        let src = minimal_model_with_derived(&expr_src);
        let parsed = parse_full_model(&src).expect("parse ok");
        let (theta, eta, indiv, cov, prev) = make_derived_ctx_simple();
        if let DerivedKind::PerRow { eval } = &parsed.model.derived_exprs[0].kind {
            let ctx = DerivedContext {
                theta: &theta,
                eta: &eta,
                indiv_params: &indiv,
                covariates: &cov,
                ipred: 0.0,
                pred: 0.0,
                dv: 0.0,
                time: 0.0,
                tafd: 0.0,
                tad: 0.0,
                prev_derived: &prev,
                compartments: &[],
                compartment_names: &[],
            };
            let result = eval(&ctx);
            assert!(
                (result - expected).abs() < 1e-10,
                "{expr_str}: expected {expected}, got {result}"
            );
        }
    }
}

#[test]
fn floor_ceil_round_functions() {
    let tests = [
        ("floor(-2.3)", -3.0),
        ("ceil(-2.3)", -2.0),
        ("round(2.5)", 3.0),
    ];
    for (expr_str, expected) in &tests {
        let src = minimal_model_with_derived(&format!("VAL = {expr_str}"));
        let parsed = parse_full_model(&src).expect("parse ok");
        let (theta, eta, indiv, cov, prev) = make_derived_ctx_simple();
        if let DerivedKind::PerRow { eval } = &parsed.model.derived_exprs[0].kind {
            let ctx = DerivedContext {
                theta: &theta,
                eta: &eta,
                indiv_params: &indiv,
                covariates: &cov,
                ipred: 0.0,
                pred: 0.0,
                dv: 0.0,
                time: 0.0,
                tafd: 0.0,
                tad: 0.0,
                prev_derived: &prev,
                compartments: &[],
                compartment_names: &[],
            };
            let result = eval(&ctx);
            assert!(
                (result - expected).abs() < 1e-10,
                "{expr_str}: expected {expected}, got {result}"
            );
        }
    }
}

#[test]
fn macheps_available_in_derived() {
    let src = minimal_model_with_derived("FLAG = if (TAD < MACHEPS) 1 else 0");
    let parsed = parse_full_model(&src).expect("parse ok");
    let (theta, eta, indiv, cov, _prev) = make_derived_ctx_simple();
    let prev = std::collections::HashMap::new();
    if let DerivedKind::PerRow { eval } = &parsed.model.derived_exprs[0].kind {
        // TAD = 0.0 < MACHEPS → flag = 1
        let ctx = DerivedContext {
            theta: &theta,
            eta: &eta,
            indiv_params: &indiv,
            covariates: &cov,
            ipred: 0.0,
            pred: 0.0,
            dv: 0.0,
            time: 1.0,
            tafd: 1.0,
            tad: 0.0, // zero
            prev_derived: &prev,
            compartments: &[],
            compartment_names: &[],
        };
        assert_eq!(eval(&ctx), 1.0, "TAD=0 should be < MACHEPS");
        // TAD = 1.0 >> MACHEPS → flag = 0
        let ctx2 = DerivedContext { tad: 1.0, ..ctx };
        assert_eq!(eval(&ctx2), 0.0, "TAD=1 should not be < MACHEPS");
    }
}

#[test]
fn ode_compartment_indexed_dose_attrs_populate_map() {
    // `F2` / `ALAG2` in a 2-compartment ODE model must (a) parse without
    // tripping the dead-parameter census — they are engine-applied dose
    // attributes, not dead structural params — and (b) populate the
    // `dose_attr_map` and enable `has_lagtime()` (#369).
    let src = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TVF2(0.6, 0.01, 1.0)
  theta TVLAG2(0.4, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  F2 = TVF2
  ALAG2 = TVLAG2

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -CL/V * depot
  d/dt(central) =  CL/V * depot - CL/V * central

[error_model]
  DV ~ proportional(PROP)
"#;
    let parsed = parse_full_model(src).expect("parse ok");
    let map = &parsed
        .model
        .ode_spec
        .as_ref()
        .expect("ode spec")
        .dose_attr_map;
    assert!(
        map.indexed_slot(crate::types::DoseAttr::F, 2).is_some(),
        "F2 must map for compartment 2"
    );
    assert!(
        map.indexed_slot(crate::types::DoseAttr::Lag, 2).is_some(),
        "ALAG2 must map for compartment 2"
    );
    // No compartment-1 override was declared.
    assert!(map.indexed_slot(crate::types::DoseAttr::F, 1).is_none());
    assert!(
        parsed.model.has_lagtime(),
        "ALAG2 must enable has_lagtime() so downstream lag handling runs"
    );
}

#[test]
fn ode_dose_attr_compartment_out_of_range_errors() {
    // `F3` references compartment 3, but the model has only 2 states — a loud
    // parse error, never a silently-ignored spare slot (#369).
    let src = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TVF3(0.6, 0.01, 1.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  F3 = TVF3

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -CL/V * depot
  d/dt(central) =  CL/V * depot - CL/V * central

[error_model]
  DV ~ proportional(PROP)
"#;
    let err = parse_full_model(src)
        .err()
        .expect("F3 with 2 compartments must error");
    assert!(
        err.contains("compartment 3") && err.contains("F3"),
        "error must name the attribute and compartment, got: {err}"
    );
}

#[test]
fn ode_modeled_duration_param_populates_map() {
    // A `D{n}` parameter (modeled infusion duration, RATE=-2; #324) in an ODE
    // model must (a) parse without tripping the dead-parameter census — it is
    // an engine-applied dose attribute, not a dead structural param — and
    // (b) populate `dose_attr_map` as Duration for compartment 2.
    let src = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TVD2(2.0, 0.1, 24.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  D2 = TVD2

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -CL/V * depot
  d/dt(central) =  CL/V * depot - CL/V * central

[error_model]
  DV ~ proportional(PROP)
"#;
    let parsed = parse_full_model(src).expect("D2 parses (engine-applied, not dead)");
    let map = &parsed
        .model
        .ode_spec
        .as_ref()
        .expect("ode spec")
        .dose_attr_map;
    assert!(
        map.indexed_slot(crate::types::DoseAttr::Duration, 2)
            .is_some(),
        "D2 must map as modeled duration for compartment 2"
    );
    // It bound nothing else (no D1, no F2).
    assert!(map
        .indexed_slot(crate::types::DoseAttr::Duration, 1)
        .is_none());
    assert!(map.indexed_slot(crate::types::DoseAttr::F, 2).is_none());
}

#[test]
fn ode_modeled_duration_out_of_range_compartment_errors() {
    // `D5` references compartment 5 but the model has 2 states -> the same
    // loud n_states guard that rejects `F3` (#324/#369), never a silently
    // ignored spare slot.
    let src = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TVD5(2.0, 0.1, 24.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  D5 = TVD5

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -CL/V * depot
  d/dt(central) =  CL/V * depot - CL/V * central

[error_model]
  DV ~ proportional(PROP)
"#;
    let err = parse_full_model(src)
        .err()
        .expect("D5 on a 2-state model must error");
    assert!(
        err.contains("compartment 5") && err.contains("D5"),
        "error must name the attribute and compartment, got: {err}"
    );
}

#[test]
fn analytical_modeled_duration_param_populates_map() {
    // A `D1` parameter (modeled infusion duration, RATE=-2) in an ANALYTICAL
    // model (#394) must (a) parse, and (b) populate `CompiledModel.dose_attr_map`
    // (NOT an `ode_spec` — analytical models have none) as Duration for
    // compartment 1, routed to a spare PkParams slot above the canonical slots.
    let src = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TVD1(5.0, 0.1, 24.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  D1 = TVD1

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)
"#;
    let parsed = parse_full_model(src).expect("D1 parses on analytical model");
    assert!(
        parsed.model.ode_spec.is_none(),
        "model must be analytical (no ode_spec)"
    );
    let slot = parsed
        .model
        .dose_attr_map
        .indexed_slot(crate::types::DoseAttr::Duration, 1)
        .expect("D1 must map as modeled duration for compartment 1");
    // Routed to the spare region above the canonical PK slots, never aliasing
    // a canonical slot or the engine-reserved F / lagtime slots.
    assert!(
        slot > crate::types::PK_IDX_LAGTIME && slot < crate::types::MAX_PK_PARAMS,
        "D1 must land in a spare slot ({} < slot < {}), got {slot}",
        crate::types::PK_IDX_LAGTIME,
        crate::types::MAX_PK_PARAMS
    );
    // Nothing else bound.
    assert!(parsed
        .model
        .dose_attr_map
        .indexed_slot(crate::types::DoseAttr::Duration, 2)
        .is_none());
}

#[test]
fn analytical_modeled_duration_out_of_range_compartment_errors() {
    // `D2` on a 1-cpt IV analytical model is not an infusable compartment
    // (infusable = {1} for one_cpt_iv) — a loud parse error, never a
    // silently-ignored slot (#394).
    let src = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TVD2(5.0, 0.1, 24.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  D2 = TVD2

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)
"#;
    let err = parse_full_model(src)
        .err()
        .expect("D2 on a 1-compartment analytical model must error");
    assert!(
        err.contains("compartment 2") && err.contains("D2"),
        "error must name the attribute and compartment, got: {err}"
    );
}

#[test]
fn analytical_modeled_duration_into_oral_depot_parses() {
    // `D1` on a `one_cpt_oral` model targets the DEPOT (cmt 1): a zero-order
    // release into the depot, then first-order `ka` absorption into central
    // (#400). Since the analytical oral propagators gained the depot
    // forced response, the depot is now an infusable compartment — this must
    // parse and bind `D1` as a modeled Duration for compartment 1.
    let src = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TVKA(1.0, 0.01, 10.0)
  theta TVD1(5.0, 0.1, 24.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  D1 = TVD1

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP)
"#;
    let parsed = parse_full_model(src).expect("D1 into the oral depot must parse (#400)");
    assert!(
        parsed.model.ode_spec.is_none(),
        "model must stay analytical (no ode_spec)"
    );
    parsed
        .model
        .dose_attr_map
        .indexed_slot(crate::types::DoseAttr::Duration, 1)
        .expect("D1 must map as modeled duration for the depot (compartment 1)");
}

#[test]
fn analytical_modeled_duration_into_oral_peripheral_parses() {
    // `D3` on a `two_cpt_oral` model targets a PERIPHERAL (cmt 3). Since #375 the
    // analytical oral closed forms CAN infuse a peripheral — the oral propagators
    // reuse the 2-cpt IV forced response, since nothing flows back into the depot
    // — so `infusable = {1 depot, 2 central, 3 peripheral}` and this must parse.
    // (It was a loud parse error before #375, when the oral propagators had no
    // peripheral forcing term; see #400 for the depot case.)
    let src = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV1(50.0, 1.0, 500.0)
  theta TVQ(5.0, 0.1, 100.0)
  theta TVV2(80.0, 1.0, 500.0)
  theta TVKA(1.0, 0.01, 10.0)
  theta TVD3(5.0, 0.1, 24.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1
  Q  = TVQ
  V2 = TVV2
  KA = TVKA
  D3 = TVD3

[structural_model]
  pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)

[error_model]
  DV ~ proportional(PROP)
"#;
    parse_full_model(src).expect("D3 (oral peripheral) is infusable since #375");
}

/// …but a `D{cmt}` past the end of the state vector is still a loud parse error.
#[test]
fn analytical_modeled_duration_beyond_oral_state_count_is_rejected() {
    let src = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV1(50.0, 1.0, 500.0)
  theta TVQ(5.0, 0.1, 100.0)
  theta TVV2(80.0, 1.0, 500.0)
  theta TVKA(1.0, 0.01, 10.0)
  theta TVD4(5.0, 0.1, 24.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1
  Q  = TVQ
  V2 = TVV2
  KA = TVKA
  D4 = TVD4

[structural_model]
  pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)

[error_model]
  DV ~ proportional(PROP)
"#;
    let err = parse_full_model(src)
        .err()
        .expect("D4 on a 3-state oral model must error");
    assert!(
        err.contains("compartment 4") && err.contains("D4"),
        "error should name the offending slot: {err}"
    );
}

/// Assemble a minimal analytical one-cpt oral model from its
/// `[individual_parameters]` dose-attribute line and its `pk(...)` structural
/// line, so the `F{cmt}`/`ALAG{cmt}` tests below vary only the two lines that
/// matter. `TVATTR` (0.7) is a valid typical value for either bioavailability
/// or lag.
fn indexed_attr_model(indiv_attr_line: &str, pk_line: &str) -> String {
    format!(
        "[parameters]\n\
             theta TVCL(5.0, 0.1, 100.0)\n\
             theta TVV(50.0, 1.0, 500.0)\n\
             theta TVKA(1.0, 0.01, 10.0)\n\
             theta TVATTR(0.7, 0.01, 10.0)\n\
             omega ETA_CL ~ 0.09\n\
             sigma PROP ~ 0.04 (sd)\n\
             \n\
             [individual_parameters]\n\
             CL = TVCL * exp(ETA_CL)\n\
             V  = TVV\n\
             KA = TVKA\n\
             {indiv_attr_line}\n\
             \n\
             [structural_model]\n\
             {pk_line}\n\
             \n\
             [error_model]\n\
             DV ~ proportional(PROP)\n"
    )
}

#[test]
fn analytical_indexed_f_and_alag_unmapped_are_rejected() {
    // Per-compartment `F{cmt}` / `ALAG{cmt}` are ODE-only. On an analytical
    // `pk ...` model, a name of that form that is NOT bound to the single dose
    // route used to be silently dropped into an unused slot (effective F = 1 /
    // lag = 0, no effect). That silent no-op must now be a loud, reason-specific
    // parse error that names the param, says the value would be discarded, and
    // points at the bare `f=` / `lagtime=` mapping. (The correctly-*mapped*
    // case is accepted — see `analytical_indexed_f_and_alag_mapped_are_accepted`.)

    // Unmapped `F1`.
    let err = parse_full_model(&indexed_attr_model(
        "F1 = TVATTR",
        "pk one_cpt_oral(cl=CL, v=V, ka=KA)",
    ))
    .err()
    .expect("unmapped F1 on an analytical oral model must error, not be silently ignored");
    assert!(
        err.contains("F1")
            && err.contains("reads as a per-compartment")
            && err.contains("bioavailability")
            && err.contains("silently discarded")
            && err.contains("f=")
            && err.contains("ode("),
        "F error must be reason-specific: name the param, the noun, that it is discarded, \
             the bare `f=` mapping, and ode(...): {err}"
    );

    // Unmapped per-compartment lag `ALAG1`.
    let err = parse_full_model(&indexed_attr_model(
        "ALAG1 = TVATTR",
        "pk one_cpt_oral(cl=CL, v=V, ka=KA)",
    ))
    .err()
    .expect("unmapped ALAG1 on an analytical oral model must error");
    assert!(
        err.contains("ALAG1")
            && err.contains("reads as a per-compartment")
            && err.contains("lag time")
            && err.contains("silently discarded")
            && err.contains("lagtime="),
        "lag error must be reason-specific and name the bare `lagtime=` mapping: {err}"
    );

    // The `LAGTIME{n}` alias hits the same guard as `ALAG{n}`.
    let err = parse_full_model(&indexed_attr_model(
        "LAGTIME1 = TVATTR",
        "pk one_cpt_oral(cl=CL, v=V, ka=KA)",
    ))
    .err()
    .expect("unmapped LAGTIME1 (ALAG alias) on an analytical oral model must error");
    assert!(
        err.contains("LAGTIME1") && err.contains("lag time") && err.contains("lagtime="),
        "LAGTIME1 alias must be rejected like ALAG1, naming the actual param: {err}"
    );

    // A higher index (`F2` — a compartment the 1-cpt model doesn't even have)
    // is rejected the same way: an unmapped per-compartment F with no effect.
    let err = parse_full_model(&indexed_attr_model(
        "F2 = TVATTR",
        "pk one_cpt_oral(cl=CL, v=V, ka=KA)",
    ))
    .err()
    .expect("unmapped F2 on an analytical oral model must error");
    assert!(
        err.contains("F2") && err.contains("bioavailability"),
        "F2 must be rejected and named: {err}"
    );
}

#[test]
fn analytical_indexed_f_and_alag_mapped_are_accepted() {
    // Regression guard (PR #725 review): a user may NAME their bioavailability
    // / lag parameter `F1` / `ALAG1` / `LAGTIME1` and bind it to the single
    // analytical dose route via the bare `f=` / `lagtime=` / `alag=` mapping.
    // The value IS applied (routed to `PK_IDX_F` / `PK_IDX_LAGTIME`),
    // bit-identical to the canonical `f=F` idiom, so the guard must NOT reject
    // it. Before the narrowing fix these all errored — a breaking regression on
    // previously-working, numerically-correct models.
    for (indiv, pk) in [
        ("F1 = TVATTR", "pk one_cpt_oral(cl=CL, v=V, ka=KA, f=F1)"),
        (
            "ALAG1 = TVATTR",
            "pk one_cpt_oral(cl=CL, v=V, ka=KA, lagtime=ALAG1)",
        ),
        // `alag=` alias for the mapping key, and the `LAGTIME{n}` param spelling.
        (
            "ALAG1 = TVATTR",
            "pk one_cpt_oral(cl=CL, v=V, ka=KA, alag=ALAG1)",
        ),
        (
            "LAGTIME1 = TVATTR",
            "pk one_cpt_oral(cl=CL, v=V, ka=KA, lagtime=LAGTIME1)",
        ),
    ] {
        let res = parse_full_model(&indexed_attr_model(indiv, pk));
        assert!(
            res.is_ok(),
            "correctly-mapped `{pk}` (with `{indiv}`) must be accepted, got: {:?}",
            res.err()
        );
    }
}

#[test]
fn analytical_modeled_rate_param_populates_map() {
    // Mirror of `analytical_modeled_duration_param_populates_map` for the
    // modeled-*rate* form: an `R1` parameter (RATE=-1) in an ANALYTICAL model
    // (#324) must parse and bind `(Rate, 1)` in `CompiledModel.dose_attr_map`,
    // routed to a spare PkParams slot above the canonical slots.
    let src = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TVR1(10.0, 0.1, 100.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  R1 = TVR1

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)
"#;
    let parsed = parse_full_model(src).expect("R1 parses on analytical model");
    assert!(
        parsed.model.ode_spec.is_none(),
        "model must be analytical (no ode_spec)"
    );
    let slot = parsed
        .model
        .dose_attr_map
        .indexed_slot(crate::types::DoseAttr::Rate, 1)
        .expect("R1 must map as modeled rate for compartment 1");
    assert!(
        slot > crate::types::PK_IDX_LAGTIME && slot < crate::types::MAX_PK_PARAMS,
        "R1 must land in a spare slot ({} < slot < {}), got {slot}",
        crate::types::PK_IDX_LAGTIME,
        crate::types::MAX_PK_PARAMS
    );
    // It is a Rate, not a Duration.
    assert!(parsed
        .model
        .dose_attr_map
        .indexed_slot(crate::types::DoseAttr::Duration, 1)
        .is_none());
}

#[test]
fn analytical_modeled_rate_out_of_range_compartment_errors() {
    // `R2` on a 1-cpt IV analytical model is not an infusable compartment
    // (infusable = {1}) — a loud parse error naming the rate attribute and
    // RATE=-1, never a silently-ignored slot (#324). Mirrors the duration case.
    let src = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TVR2(10.0, 0.1, 100.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  R2 = TVR2

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)
"#;
    let err = parse_full_model(src)
        .err()
        .expect("R2 on a 1-compartment analytical model must error");
    assert!(
        err.contains("compartment 2")
            && err.contains("R2")
            && err.contains("rate")
            && err.contains("-1"),
        "error must name the attribute, compartment, and RATE=-1, got: {err}"
    );
}

#[test]
fn ode_modeled_rate_param_populates_map() {
    // The ODE engine routes `R{cmt}` through the same `from_indexed_name`
    // gate as `D{cmt}` (no engine-specific code), so an `R1` parameter on an
    // ODE model must bind `(Rate, 1)` in the `ode_spec`'s `dose_attr_map`.
    let src = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TVR1(10.0, 0.1, 100.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  R1 = TVR1

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -(CL/V)*central

[error_model]
  DV ~ proportional(PROP)
"#;
    let parsed = parse_full_model(src).expect("R1 parses on ODE model");
    let ode = parsed
        .model
        .ode_spec
        .as_ref()
        .expect("model must be an ODE model");
    ode.dose_attr_map
        .indexed_slot(crate::types::DoseAttr::Rate, 1)
        .expect("R1 must map as modeled rate for compartment 1 in the ode_spec");
}

#[test]
fn analytical_modeled_dose_param_not_flagged_as_dead() {
    // Regression: an ANALYTICAL `R{n}` (RATE=-1) / `D{n}` (RATE=-2) parameter
    // is routed to a spare `PkParams` slot and consulted only via coded-`RATE`
    // *data*, so it has no textual reference. The dead-parameter census must
    // NOT flag it as "computed but never used" — that message tells the user
    // to delete a load-bearing modeled-infusion parameter. (R1 was newly
    // recognised in #324; D1 was the same latent miss from #395.)
    let rate_src = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVR1(20.0, 0.1, 100.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  R1 = TVR1

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)
"#;
    let parsed = parse_full_model(rate_src).expect("R1 parses");
    assert!(
        !parsed
            .model
            .parse_warnings
            .iter()
            .any(|w| w.contains("never used") && w.contains("R1")),
        "analytical R1 must not be flagged as dead: {:?}",
        parsed.model.parse_warnings
    );

    // Same for the duration form `D1`.
    let dur_src = rate_src.replace("TVR1", "TVD1").replace("R1 =", "D1 =");
    let parsed = parse_full_model(&dur_src).expect("D1 parses");
    assert!(
        !parsed
            .model
            .parse_warnings
            .iter()
            .any(|w| w.contains("never used") && w.contains("D1")),
        "analytical D1 must not be flagged as dead: {:?}",
        parsed.model.parse_warnings
    );

    // Guard against over-exemption: a genuinely unused analytical parameter
    // (here `JUNK`, neither mapped into `pk(...)` nor referenced elsewhere)
    // must STILL be flagged. This proves the new analytical exemption is
    // scoped to `Dn`/`Rn` and did not silence the census wholesale.
    let dead_src = rate_src.replace("  R1 = TVR1\n", "  R1 = TVR1\n  JUNK = TVCL * 2.0\n");
    let parsed = parse_full_model(&dead_src).expect("parses");
    assert!(
        parsed
            .model
            .parse_warnings
            .iter()
            .any(|w| w.contains("never used") && w.contains("JUNK")),
        "a genuinely unused analytical param must still be flagged: {:?}",
        parsed.model.parse_warnings
    );
}

#[test]
fn derived_resolves_covariate_case_insensitively() {
    // Regression: a covariate carried in the dataset under an uppercase
    // header (`WT`) referenced as lowercase `wt` in a [derived] expression
    // must resolve to the covariate value, not silently evaluate to 0.
    // build_derived_vars must insert a lowercase alias because
    // `eval_expression` looks the name up verbatim.
    let src = minimal_model_with_derived("WT_DERIVED = wt * 2");
    let parsed = parse_full_model(&src).expect("parse ok");
    let (theta, eta, indiv, _cov, prev) = make_derived_ctx_simple();
    // Covariate stored under uppercase `WT`, as a NONMEM header would be.
    let mut cov = std::collections::HashMap::new();
    cov.insert("WT".to_string(), 70.0);
    if let DerivedKind::PerRow { eval } = &parsed.model.derived_exprs[0].kind {
        let ctx = DerivedContext {
            theta: &theta,
            eta: &eta,
            indiv_params: &indiv,
            covariates: &cov,
            ipred: 0.0,
            pred: 0.0,
            dv: 0.0,
            time: 0.0,
            tafd: 0.0,
            tad: 0.0,
            prev_derived: &prev,
            compartments: &[],
            compartment_names: &[],
        };
        assert_eq!(
            eval(&ctx),
            140.0,
            "lowercase `wt` must resolve to covariate header `WT` (=70)"
        );
    } else {
        panic!("expected PerRow derived kind");
    }
}

#[test]
fn test_apply_fit_option_frem_predictions() {
    let mut opts = FitOptions::default();
    assert_eq!(
        apply_fit_option(
            &mut opts,
            "frem_predictions",
            "TV_WT/ETA_WT_FREM:100, TV_AGE/ETA_AGE_FREM:200"
        ),
        Ok(true)
    );
    assert_eq!(
        opts.frem_predictions.as_deref(),
        Some("TV_WT/ETA_WT_FREM:100, TV_AGE/ETA_AGE_FREM:200")
    );
}

#[test]
fn test_apply_fit_option_frem_sigma() {
    let mut opts = FitOptions::default();
    assert_eq!(
        apply_fit_option(&mut opts, "frem_sigma", "EPSCOV"),
        Ok(true)
    );
    assert_eq!(opts.frem_sigma.as_deref(), Some("EPSCOV"));
}

/// #485: the cov-static constant fold in `eval_param_duals` /
/// `eval_param_eta_grad` must be bit-identical to the unfolded path. The
/// model below has a covariate + FIXED-θ kernel (`FMAT`, `FAC` — foldable)
/// and a dynamic tail (`CL = free-θ × … × exp(η)` — not foldable).
const COV_STATIC_FOLD_MODEL: &str = r#"
[parameters]
  theta TVCL (5, 0.001, 100.0)
  theta GAM (2.0, FIX)
  omega ETA_CL ~ 0.1
  sigma PROP ~ 0.2 (sd)
[individual_parameters]
  FMAT = WT ^ GAM / (WT ^ GAM + 3 ^ GAM)
  if (AGE < 13) {
    FAC = AGE * 0.1
  } else {
    FAC = 1.0
  }
  CL = TVCL * FMAT * FAC * exp(ETA_CL)
  V1 = 10
[structural_model]
  pk one_cpt_iv(cl=CL, v1=V1)
[error_model]
  DV ~ proportional(PROP)
"#;

fn cov_static_fold_fixture() -> (CompiledModel, Vec<f64>, Vec<f64>, HashMap<String, f64>) {
    let model = parse_model_string(COV_STATIC_FOLD_MODEL).expect("model compiles");
    let theta = vec![5.0, 2.0];
    let eta = vec![0.3_f64];
    let cov: HashMap<String, f64> = [("WT".to_string(), 70.0), ("AGE".to_string(), 5.0)]
        .into_iter()
        .collect();
    (model, theta, eta, cov)
}

#[test]
fn cov_static_mask_classifies_kernel_and_dynamic_tail() {
    let (model, ..) = cov_static_fold_fixture();
    let prog = model
        .indiv_param_partials
        .indiv_param_program
        .as_ref()
        .expect("indiv param program present");
    // Some slots fold (the covariate/FIXED-θ kernel) …
    assert!(
        prog.cov_static_mask.iter().any(|&b| b),
        "expected some cov-static slots"
    );
    // … and at least one does not (the free-θ × exp(η) CL slot).
    assert!(
        prog.cov_static_mask.iter().any(|&b| !b),
        "expected the dynamic CL slot to stay unfolded"
    );
}

#[test]
fn cov_static_fold_matches_unfolded_dual2() {
    let (model, theta, eta, cov) = cov_static_fold_fixture();
    let prog = model
        .indiv_param_partials
        .indiv_param_program
        .as_ref()
        .expect("indiv param program present");
    // n_theta + n_eta = 2 + 1 = 3.
    let folded = prog.eval_param_duals::<3>(&theta, &eta, &cov);
    let mut unfolded_prog = prog.clone();
    unfolded_prog.cov_static_mask = Vec::new();
    let unfolded = unfolded_prog.eval_param_duals::<3>(&theta, &eta, &cov);

    assert_eq!(folded.len(), unfolded.len());
    for (i, (a, b)) in folded.iter().zip(unfolded.iter()).enumerate() {
        assert_eq!(a.value, b.value, "value mismatch at row {i}");
        assert_eq!(a.grad, b.grad, "grad mismatch at row {i}");
        assert_eq!(a.hess, b.hess, "hess mismatch at row {i}");
    }
}

/// A covariate-heavy individual-parameters block modelled on the jasmine
/// vancomycin-pediatrics run60 kernel: a large covariate-only prefix
/// (CKD-EPI-/FFM-style pow/exp/log + sex/age branches — all cov-static) feeding
/// a small free-θ × exp(η) tail. Used by the fold microbenchmark.
const COV_STATIC_BENCH_MODEL: &str = r#"
[parameters]
  theta TVCL (5, 0.001, 100.0)
  theta TVV1 (30, 0.001, 500.0)
  theta THCR (0.2, -5.0, 5.0)
  omega ETA_CL ~ 0.1
  omega ETA_V1 ~ 0.1
  sigma PROP ~ 0.2 (sd)
[individual_parameters]
  BMI = WEIGHT / ((HEIGHT / 100.0) ^ 2)
  if (SEX == 0) {
    FFM = 9270.0 * WEIGHT / (8780.0 + 244.0 * BMI)
    A1  = -0.241
    KK  = 0.7
  } else {
    FFM = 9270.0 * WEIGHT / (6680.0 + 216.0 * BMI)
    A1  = -0.302
    KK  = 0.9
  }
  CKDEPI = 142 * (CREAT / KK) ^ A1 * 0.9938 ^ AGE
  SCH    = 0.413 * HEIGHT / CREAT
  EGFR   = if (AGE < 13) SCH else CKDEPI
  FCOV   = (EGFR / 100) ^ 0.75 * exp(-log(2) / 0.6 * AGE) * (FFM / 34) ^ 0.75
  CL = TVCL * FCOV * exp(ETA_CL)
  V1 = TVV1 * (FFM / 34) * exp(ETA_V1)
[structural_model]
  pk one_cpt_iv(cl=CL, v1=V1)
[error_model]
  DV ~ proportional(PROP)
"#;

/// #485: bit-identity of the cov-static fold on a covariate-heavy kernel
/// (the jasmine-style [`COV_STATIC_BENCH_MODEL`]: a `SEX` branch, an inline
/// `if (AGE < 13) … else …` conditional, FFM forward-references, and
/// pow/exp/log throughout). A far richer block than `cov_static_fold_fixture`:
/// it asserts the fold is bit-for-bit identical to the unfolded walk on every
/// dual axis, for both branches of the `SEX` split.
///
/// The per-call timing microbenchmark these numbers came from is dev-only
/// scaffolding (its figures live in PR #489), so it is not committed as a test.
#[test]
fn cov_static_fold_matches_unfolded_bench_kernel() {
    let model = parse_model_string(COV_STATIC_BENCH_MODEL).expect("bench model compiles");
    let prog = model
        .indiv_param_partials
        .indiv_param_program
        .as_ref()
        .expect("indiv param program");
    // The covariate kernel (BMI/FFM/CKDEPI/SCH/EGFR/FCOV/A1/KK) folds; the
    // free-θ × exp(η) CL/V1 tail does not.
    assert!(
        prog.cov_static_mask.iter().any(|&b| b),
        "expected cov-static slots in the kernel"
    );
    assert!(
        prog.cov_static_mask.iter().any(|&b| !b),
        "expected the CL/V1 tail to stay dynamic"
    );
    let theta: Vec<f64> = vec![5.0, 30.0, 0.2];
    assert_eq!(theta.len(), model.n_theta);
    let eta = vec![0.05_f64, -0.1_f64];
    assert_eq!(eta.len(), model.n_eta);
    let base_cov: HashMap<String, f64> = [
        ("AGE", 4.0),
        ("CREAT", 0.4),
        ("SEX", 1.0),
        ("HEIGHT", 100.0),
        ("WEIGHT", 16.0),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();

    let mut unfolded_prog = prog.clone();
    unfolded_prog.cov_static_mask = Vec::new();

    // Both arms of the `if (SEX == 0)` split, to fold each FFM branch.
    for sex in [1.0_f64, 0.0_f64] {
        let mut cov = base_cov.clone();
        cov.insert("SEX".to_string(), sex);

        // Dual2 (outer θ,η): n_theta + n_eta = 3 + 2 = 5.
        let folded2 = prog.eval_param_duals::<5>(&theta, &eta, &cov);
        let unfolded2 = unfolded_prog.eval_param_duals::<5>(&theta, &eta, &cov);
        assert_eq!(folded2.len(), unfolded2.len());
        for (i, (a, b)) in folded2.iter().zip(unfolded2.iter()).enumerate() {
            assert_eq!(
                a.value, b.value,
                "SEX={sex} dual2 value mismatch at row {i}"
            );
            assert_eq!(a.grad, b.grad, "SEX={sex} dual2 grad mismatch at row {i}");
            assert_eq!(a.hess, b.hess, "SEX={sex} dual2 hess mismatch at row {i}");
        }

        // Dual1 (inner η): n_eta = 2.
        let folded1 = prog.eval_param_eta_grad::<2>(&theta, &eta, &cov);
        let unfolded1 = unfolded_prog.eval_param_eta_grad::<2>(&theta, &eta, &cov);
        assert_eq!(folded1.len(), unfolded1.len());
        for (i, (a, b)) in folded1.iter().zip(unfolded1.iter()).enumerate() {
            assert_eq!(
                a.value, b.value,
                "SEX={sex} dual1 value mismatch at row {i}"
            );
            assert_eq!(a.grad, b.grad, "SEX={sex} dual1 grad mismatch at row {i}");
        }
    }
}

#[test]
fn cov_static_fold_matches_unfolded_dual1() {
    let (model, theta, eta, cov) = cov_static_fold_fixture();
    let prog = model
        .indiv_param_partials
        .indiv_param_program
        .as_ref()
        .expect("indiv param program present");
    // n_eta = 1.
    let folded = prog.eval_param_eta_grad::<1>(&theta, &eta, &cov);
    let mut unfolded_prog = prog.clone();
    unfolded_prog.cov_static_mask = Vec::new();
    let unfolded = unfolded_prog.eval_param_eta_grad::<1>(&theta, &eta, &cov);

    assert_eq!(folded.len(), unfolded.len());
    for (i, (a, b)) in folded.iter().zip(unfolded.iter()).enumerate() {
        assert_eq!(a.value, b.value, "value mismatch at row {i}");
        assert_eq!(a.grad, b.grad, "grad mismatch at row {i}");
    }
}

/// #485: exhaustively exercise the cov-static classifier helpers on every
/// AST / bytecode arm. The model-driven tests only reach the arms a parsed
/// `[individual_parameters]` block happens to emit; hand-built nodes hit the
/// rest (NN outputs, `&&`/`||`/`!`, inline conditionals, dynamic var refs).
#[test]
fn cov_static_classifier_helpers_cover_all_arms() {
    // dual-axis state: slot 0 dynamic, slot 1 cov-static.
    let dv = [true, false];

    // ── bytecode_is_dynamic ──
    let bc = |ops: Vec<Op>| Bytecode {
        ops,
        constants: vec![0.0],
        max_stack: 2,
        gathers: Vec::new(),
    };
    assert!(bytecode_is_dynamic(&bc(vec![Op::PushEta(0)]), &dv));
    assert!(bytecode_is_dynamic(&bc(vec![Op::PushTheta(0)]), &dv));
    assert!(bytecode_is_dynamic(&bc(vec![Op::PushNnOutput(0, 0)]), &dv));
    assert!(bytecode_is_dynamic(&bc(vec![Op::PushVar(0)]), &dv)); // slot 0 dynamic
    assert!(!bytecode_is_dynamic(&bc(vec![Op::PushVar(1)]), &dv)); // slot 1 static
    assert!(!bytecode_is_dynamic(
        &bc(vec![Op::PushCov(0), Op::PushConst(0), Op::Add]),
        &dv
    ));

    // ── expr_is_dynamic ──
    let lit = || Expression::Literal(1.0);
    assert!(expr_is_dynamic(&Expression::Eta(0), &dv));
    assert!(expr_is_dynamic(&Expression::Theta(0), &dv));
    assert!(expr_is_dynamic(
        &Expression::NnOutput {
            nn_idx: 0,
            output_idx: 0
        },
        &dv
    ));
    assert!(expr_is_dynamic(&Expression::Variable("x".into()), &dv));
    assert!(expr_is_dynamic(&Expression::VariableIdx(0), &dv)); // dynamic slot
    assert!(!expr_is_dynamic(&Expression::VariableIdx(1), &dv)); // static slot
    assert!(!expr_is_dynamic(&lit(), &dv));
    assert!(!expr_is_dynamic(&Expression::Covariate("WT".into()), &dv));
    assert!(!expr_is_dynamic(&Expression::CovariateIdx(0), &dv));
    // BinOp / Power recurse into both operands.
    assert!(expr_is_dynamic(
        &Expression::BinOp(Box::new(Expression::Theta(0)), BinOp::Add, Box::new(lit())),
        &dv
    ));
    assert!(!expr_is_dynamic(
        &Expression::Power(
            Box::new(Expression::Covariate("WT".into())),
            Box::new(lit())
        ),
        &dv
    ));
    // UnaryFn recurses into its argument.
    assert!(expr_is_dynamic(
        &Expression::UnaryFn("exp".into(), Box::new(Expression::Eta(0))),
        &dv
    ));
    assert!(!expr_is_dynamic(
        &Expression::UnaryFn("log".into(), Box::new(lit())),
        &dv
    ));

    // ── cond_is_dynamic ──
    let static_cmp = Condition::Compare(Expression::Covariate("AGE".into()), CmpOp::Lt, lit());
    let dyn_cmp = Condition::Compare(Expression::Theta(0), CmpOp::Gt, lit());
    assert!(!cond_is_dynamic(&static_cmp, &dv));
    assert!(cond_is_dynamic(&dyn_cmp, &dv));
    assert!(cond_is_dynamic(
        &Condition::And(Box::new(static_cmp.clone()), Box::new(dyn_cmp.clone())),
        &dv
    ));
    assert!(!cond_is_dynamic(
        &Condition::Or(Box::new(static_cmp.clone()), Box::new(static_cmp.clone())),
        &dv
    ));
    assert!(cond_is_dynamic(
        &Condition::Not(Box::new(dyn_cmp.clone())),
        &dv
    ));
    assert!(!cond_is_dynamic(
        &Condition::Not(Box::new(static_cmp.clone())),
        &dv
    ));

    // ── inline conditional (expr_is_dynamic::Conditional) ──
    assert!(expr_is_dynamic(
        &Expression::Conditional(
            Box::new(static_cmp.clone()),
            Box::new(Expression::Eta(0)), // dynamic `then`
            Box::new(lit())
        ),
        &dv
    ));
    assert!(expr_is_dynamic(
        &Expression::Conditional(
            Box::new(dyn_cmp.clone()), // dynamic condition
            Box::new(Expression::Covariate("WT".into())),
            Box::new(lit())
        ),
        &dv
    ));
    assert!(!expr_is_dynamic(
        &Expression::Conditional(
            Box::new(static_cmp),
            Box::new(Expression::Covariate("WT".into())),
            Box::new(lit())
        ),
        &dv
    ));
}

/// #485: `compute_cov_static_mask` corner cases the parsed fixtures don't
/// reach — a forward reference (a single pass is insufficient, so the
/// monotone fixpoint must re-iterate) and a slot assigned under a dynamic
/// `if` condition (covariate-only body, θ-dependent governing condition).
#[test]
fn cov_static_mask_fixpoint_and_dynamic_if_context() {
    // slot0 = slot1 + 1   (forward ref — slot1 is assigned *after* slot0)
    // slot1 = θ0          (dynamic)
    // A single forward pass marks slot0 static; the fixpoint re-pass flips it.
    let fwd = vec![
        // A non-`AssignBc`/`If` statement is ignored by the classifier (the
        // `_ => {}` arm) — `[individual_parameters]` never emits one, so cover
        // it here.
        Statement::Assign("ignored".into(), Expression::Literal(0.0)),
        Statement::AssignBc(
            0,
            Bytecode {
                ops: vec![Op::PushVar(1), Op::PushConst(0), Op::Add],
                constants: vec![1.0],
                max_stack: 2,
                gathers: Vec::new(),
            },
        ),
        Statement::AssignBc(
            1,
            Bytecode {
                ops: vec![Op::PushTheta(0)],
                constants: vec![],
                max_stack: 1,
                gathers: Vec::new(),
            },
        ),
    ];
    assert_eq!(compute_cov_static_mask(&fwd, 2), vec![false, false]);

    // if (θ0 > 5) { X = WT } else { X = 0 } — covariate/literal bodies, but the
    // governing condition is dynamic, so X must not fold.
    let dyn_if = vec![Statement::If {
        branches: vec![(
            Condition::Compare(Expression::Theta(0), CmpOp::Gt, Expression::Literal(5.0)),
            vec![Statement::AssignBc(
                0,
                Bytecode {
                    ops: vec![Op::PushCov(0)],
                    constants: vec![],
                    max_stack: 1,
                    gathers: Vec::new(),
                },
            )],
        )],
        else_body: Some(vec![Statement::AssignBc(
            0,
            Bytecode {
                ops: vec![Op::PushConst(0)],
                constants: vec![0.0],
                max_stack: 1,
                gathers: Vec::new(),
            },
        )]),
    }];
    assert_eq!(compute_cov_static_mask(&dyn_if, 1), vec![false]);

    // Pure-covariate `if`: the slot stays static (static branch of the If arm
    // + the else_body walk under a non-dynamic governing condition).
    let stat_if = vec![Statement::If {
        branches: vec![(
            Condition::Compare(
                Expression::Covariate("AGE".into()),
                CmpOp::Lt,
                Expression::Literal(13.0),
            ),
            vec![Statement::AssignBc(
                0,
                Bytecode {
                    ops: vec![Op::PushCov(0)],
                    constants: vec![],
                    max_stack: 1,
                    gathers: Vec::new(),
                },
            )],
        )],
        else_body: Some(vec![Statement::AssignBc(
            0,
            Bytecode {
                ops: vec![Op::PushConst(0)],
                constants: vec![1.0],
                max_stack: 1,
                gathers: Vec::new(),
            },
        )]),
    }];
    assert_eq!(compute_cov_static_mask(&stat_if, 1), vec![true]);
}

// ── [simulation] block key handling ──────────────────────────────────────
fn sim_lines(body: &[&str]) -> Vec<String> {
    body.iter().map(|s| s.to_string()).collect()
}

/// [`parse_simulation_block`] with no declared covariates — the shape every key
/// test below wants. The declared-name list only gates `covariate NAME = ...`
/// (#1083); an empty list means "no `[covariates]` block", which accepts any name.
fn parse_sim_block(lines: &[String]) -> Result<SimulationSpec, String> {
    parse_simulation_block(lines, &[])
}

#[test]
fn simulation_long_form_keys_apply() {
    // The canonical (and example-file) spellings must actually be honored —
    // the bug was that `n_subjects`/`dose_amt`/`dose_cmt` were silently ignored
    // and fell back to the defaults (10 / 100 / 1).
    let spec = parse_sim_block(&sim_lines(&[
        "n_subjects = 7",
        "dose_amt = 50",
        "dose_cmt = 2",
        "seed = 99",
        "times = [0.5, 1.0, 2.0]",
    ]))
    .expect("long-form keys parse");
    assert_eq!(spec.n_subjects, 7);
    assert_eq!(spec.dose_amt, 50.0);
    assert_eq!(spec.dose_cmt, 2);
    assert_eq!(spec.seed, 99);
    assert_eq!(spec.obs_times, vec![0.5, 1.0, 2.0]);
}

#[test]
fn simulation_short_form_aliases_apply() {
    // Back-compat: the short `subjects`/`dose`/`cmt` forms (the previously
    // documented spelling) remain valid aliases for the same fields, and the
    // defaults hold for the keys we omit (seed = 42).
    let spec = parse_sim_block(&sim_lines(&[
        "subjects = 3",
        "dose = 25",
        "cmt = 4",
        "times = [1.0]",
    ]))
    .expect("short-form aliases parse");
    assert_eq!(spec.n_subjects, 3);
    assert_eq!(spec.dose_amt, 25.0);
    assert_eq!(spec.dose_cmt, 4);
    assert_eq!(spec.seed, 42, "untouched seed keeps its default");
}

#[test]
fn simulation_unknown_key_errors() {
    // A typo (e.g. `n_subject`) must be a hard error, not a silent default —
    // this is the silent-failure class the fix closes.
    let err = parse_sim_block(&sim_lines(&[
        "n_subject = 5", // typo: missing the trailing 's'
        "times = [1.0]",
    ]))
    .unwrap_err();
    assert!(
        err.starts_with("[simulation]:")
            && err.contains("unknown key")
            && err.contains("n_subject"),
        "got: {err}"
    );
}

#[test]
fn simulation_malformed_line_errors() {
    // A non-blank line with no `=` (e.g. a forgotten `=`) is malformed and must
    // error rather than being silently skipped into the default.
    let err = parse_sim_block(&sim_lines(&["n_subjects 5", "times = [1.0]"])).unwrap_err();
    assert!(
        err.starts_with("[simulation]:") && err.contains("malformed line"),
        "got: {err}"
    );
}

// Every value-parsing arm must report a clear, prefixed error on a bad value —
// not just `n_subjects`. This pins the error branch of each `map_err`/`?` so the
// diff isn't covered by happy paths alone.
#[test]
fn simulation_bad_value_errors_per_key() {
    for (line, key) in [
        ("n_subjects = abc", "n_subjects"),
        ("dose_amt = abc", "dose_amt"),
        ("dose_cmt = 1.5", "dose_cmt"), // non-integer compartment
        ("seed = -1", "seed"),          // seed is u64
        ("times = [1.0, oops]", "times"),
    ] {
        let err = parse_sim_block(&sim_lines(&[line, "times = [1.0]"])).unwrap_err();
        assert!(
            err.starts_with("[simulation]:") && err.contains(key),
            "key `{key}` on `{line}` gave: {err}"
        );
    }
}

#[test]
fn simulation_requires_times() {
    let err = parse_sim_block(&sim_lines(&["n_subjects = 5"])).unwrap_err();
    assert!(err.contains("times"), "got: {err}");
}

#[test]
fn simulation_horizon_parses() {
    // `horizon = <t>` is captured as the administrative censoring window (#522).
    let spec =
        parse_sim_block(&sim_lines(&["horizon = 14", "times = [1.0]"])).expect("horizon parses");
    assert_eq!(spec.horizon, Some(14.0));
    // Absent ⇒ None (the per-record window path).
    let spec = parse_sim_block(&sim_lines(&["times = [1.0]"])).expect("no horizon");
    assert_eq!(spec.horizon, None);
}

#[test]
fn simulation_horizon_satisfies_observe_requirement() {
    // A TTE-only design has no continuous `times`; a `horizon` alone is enough
    // to make the block valid (the relaxed times-OR-horizon rule).
    let spec = parse_sim_block(&sim_lines(&["n_subjects = 5", "horizon = 14"]))
        .expect("horizon alone is a valid design");
    assert_eq!(spec.horizon, Some(14.0));
    assert!(spec.obs_times.is_empty());
}

#[test]
fn simulation_horizon_rejects_nonpositive_and_nonfinite() {
    for bad in [
        "horizon = 0",
        "horizon = -3",
        "horizon = inf",
        "horizon = nan",
    ] {
        let err = parse_sim_block(&sim_lines(&[bad, "times = [1.0]"])).unwrap_err();
        assert!(
            err.starts_with("[simulation]:") && err.contains("horizon"),
            "`{bad}` gave: {err}"
        );
    }
}

#[test]
fn simulation_horizon_bad_value_errors() {
    let err = parse_sim_block(&sim_lines(&["horizon = abc", "times = [1.0]"])).unwrap_err();
    assert!(
        err.starts_with("[simulation]:") && err.contains("horizon"),
        "got: {err}"
    );
}

// ── [simulation] per-subject covariates (#1083) ──────────────────────────
//
// A simulated trial invents arms that are in no dataset, so the covariates its
// design turns on — an MBMA arm size behind `weight = NARM`, a reported standard
// error behind `weight = WPSE` — have no row to be read from and must be stated
// here. Before #1083 `SimulationSpec::covariates` was a field the parser always
// wrote empty and nothing ever read.

#[test]
fn simulation_covariate_list_is_read_per_subject() {
    let spec = parse_sim_block(&sim_lines(&[
        "n_subjects = 3",
        "times = [1.0]",
        "covariate NARM = [400, 200, 50]",
    ]))
    .expect("per-subject covariate list parses");
    assert_eq!(
        spec.covariates,
        vec![("NARM".to_string(), vec![400.0, 200.0, 50.0])]
    );
}

#[test]
fn simulation_covariate_scalar_broadcasts_to_every_subject() {
    let spec = parse_sim_block(&sim_lines(&[
        "n_subjects = 4",
        "times = [1.0]",
        "covariate WT = 70",
    ]))
    .expect("scalar covariate parses");
    assert_eq!(spec.covariates, vec![("WT".to_string(), vec![70.0; 4])]);
}

#[test]
fn simulation_covariate_length_is_checked_regardless_of_key_order() {
    // `n_subjects` deliberately *follows* the covariate line: the length rule runs
    // after the whole block is read, so the keys stay order-independent like every
    // other key in this block.
    let err = parse_sim_block(&sim_lines(&[
        "covariate NARM = [400, 200]",
        "n_subjects = 3",
        "times = [1.0]",
    ]))
    .unwrap_err();
    assert!(
        err.starts_with("[simulation]:")
            && err.contains("NARM")
            && err.contains("2 value(s)")
            && err.contains("3 subject(s)"),
        "got: {err}"
    );
}

#[test]
fn simulation_covariate_must_be_declared_when_a_covariates_block_exists() {
    // The `[covariates]` block is authoritative when present, exactly as it is for
    // the data readers — a name it never declared is a typo, and one that reached
    // `Subject` silently would be read by nothing.
    let declared = vec!["NARM".to_string()];
    let err = parse_simulation_block(
        &sim_lines(&["n_subjects = 2", "times = [1.0]", "covariate NARN = 100"]),
        &declared,
    )
    .unwrap_err();
    assert!(
        err.contains("NARN") && err.contains("not declared in [covariates]"),
        "got: {err}"
    );
    // …and the correctly-spelled one is accepted against the same list.
    parse_simulation_block(
        &sim_lines(&["n_subjects = 2", "times = [1.0]", "covariate NARM = 100"]),
        &declared,
    )
    .expect("a declared covariate is accepted");
}

#[test]
fn simulation_covariate_rejects_duplicates_and_bad_values() {
    let dup = parse_sim_block(&sim_lines(&[
        "n_subjects = 2",
        "times = [1.0]",
        "covariate NARM = 100",
        "covariate NARM = 50",
    ]))
    .unwrap_err();
    assert!(dup.contains("declared twice"), "got: {dup}");

    let bad = parse_sim_block(&sim_lines(&[
        "n_subjects = 1",
        "times = [1.0]",
        "covariate NARM = many",
    ]))
    .unwrap_err();
    assert!(
        bad.starts_with("[simulation]:") && bad.contains("NARM"),
        "got: {bad}"
    );

    let nameless = parse_sim_block(&sim_lines(&["times = [1.0]", "covariate = 100"])).unwrap_err();
    assert!(nameless.contains("needs a name"), "got: {nameless}");
}

#[test]
fn simulation_covariate_matches_the_word_not_the_prefix() {
    // `covariates = ...` must stay an unknown key rather than being read as a
    // covariate named `s` — the arm is keyed on the word, not on `starts_with`.
    let err = parse_sim_block(&sim_lines(&["times = [1.0]", "covariates = [1, 2]"])).unwrap_err();
    assert!(
        err.contains("unknown key `covariates`"),
        "a prefix match would have accepted this: {err}"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Mixture models — `[mixture]` block + reserved `MIXNUM` index (#977, Phase 1)
// ─────────────────────────────────────────────────────────────────────────

/// A valid 2-class mixture model: class-specific CL via a `MIXNUM` branch, a
/// covariate-dependent mixing logit, and per-class Ω/Σ overrides.
const MIXTURE_2CLASS: &str = r"
[parameters]
  theta TVCL1(1.0, 0.001, 100.0)
  theta TVCL2(3.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  theta BWT(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[mixture]
  nsub = 2
  logit(1) = MIXL + BWT*WT
  omega(2) ETA_CL ~ 0.4
  sigma(2) EPS ~ 0.12

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

#[test]
fn mixture_block_parses_into_spec() {
    let model = parse_model_string(MIXTURE_2CLASS).expect("mixture model parses");
    let spec = model.mixture.as_ref().expect("mixture spec present");
    assert_eq!(spec.n_classes, 2);
    assert_eq!(spec.mixing.len(), 1, "K-1 mixing exprs for K=2");
    assert_eq!(spec.mixing[0].class, 1);
    assert!(spec.mixing[0].is_logit);
    assert_eq!(spec.logit_covariates, vec!["WT".to_string()]);
    assert_eq!(spec.omega_overrides.len(), 1);
    assert_eq!(spec.omega_overrides[0].class, 2);
    assert_eq!(spec.omega_overrides[0].name, "ETA_CL");
    assert!((spec.omega_overrides[0].init - 0.4).abs() < 1e-12);
    assert_eq!(spec.sigma_overrides.len(), 1);
    assert_eq!(spec.sigma_overrides[0].class, 2);
    assert_eq!(spec.sigma_overrides[0].name, "EPS");
}

// ─────────────────────────────────────────────────────────────────────────
// Class-aware mu-referencing for MIXNUM-switched typical values (#996)
// ─────────────────────────────────────────────────────────────────────────

/// Build a mixture model with `nsub = k` and the given `[individual_parameters]`
/// body, using thetas TVCL1/TVCL2/TVCL3/TVV/MIXL1/MIXL2 and eta ETA_CL/ETA_V.
fn mixture_model_with_indiv(n_classes: usize, indiv: &str) -> crate::types::CompiledModel {
    let mixing: String = (1..n_classes)
        .map(|c| format!("  logit({c}) = MIXL{c}\n"))
        .collect();
    let src = format!(
        r"
[parameters]
  theta TVCL1(1.0, 0.001, 100.0)
  theta TVCL2(3.0, 0.001, 100.0)
  theta TVCL3(5.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL1(0.0, -10.0, 10.0)
  theta MIXL2(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.1
  omega ETA_V ~ 0.1
  sigma EPS ~ 0.01

[mixture]
  nsub = {n_classes}
{mixing}
[individual_parameters]
{indiv}

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
"
    );
    parse_model_string(&src).expect("mixture model parses")
}

/// Names of the class anchors detected for `eta`, or `None` when the eta has no
/// class-aware mu-ref.
fn class_anchors(model: &crate::types::CompiledModel, eta: &str) -> Option<Vec<String>> {
    model
        .mixture
        .as_ref()
        .expect("mixture spec")
        .mu_refs
        .iter()
        .find(|m| m.eta_name == eta)
        .map(|m| m.theta_names.clone())
}

#[test]
fn class_aware_mu_ref_detects_canonical_two_class_ternary() {
    // The canonical ferx mixture form: one eta, one theta per class.
    let model = mixture_model_with_indiv(
        2,
        "  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)\n  V = TVV",
    );
    assert_eq!(
        class_anchors(&model, "ETA_CL"),
        Some(vec!["TVCL1".to_string(), "TVCL2".to_string()])
    );
    // The unswitched V carries no eta at all, so no entry.
    assert_eq!(class_anchors(&model, "ETA_V"), None);
    // The classical (non-class-aware) map is untouched — a mixture must not
    // start reporting a single anchor for a class-switched parameter.
    assert!(!model.mu_refs.contains_key("ETA_CL"));
}

#[test]
fn class_aware_mu_ref_detects_three_class_chain() {
    let model = mixture_model_with_indiv(
        3,
        "  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) \
             else if (MIXNUM == 2) TVCL2 * exp(ETA_CL) \
             else TVCL3 * exp(ETA_CL)\n  V = TVV",
    );
    assert_eq!(
        class_anchors(&model, "ETA_CL"),
        Some(vec![
            "TVCL1".to_string(),
            "TVCL2".to_string(),
            "TVCL3".to_string()
        ])
    );
}

#[test]
fn class_aware_mu_ref_trailing_else_covers_every_unnamed_class() {
    // K = 3 but only class 1 is named: classes 2 and 3 share the else arm's
    // theta, which is well-defined (and updates from their pooled members).
    let model = mixture_model_with_indiv(
        3,
        "  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)\n  V = TVV",
    );
    assert_eq!(
        class_anchors(&model, "ETA_CL"),
        Some(vec![
            "TVCL1".to_string(),
            "TVCL2".to_string(),
            "TVCL2".to_string()
        ])
    );
}

#[test]
fn class_aware_mu_ref_degenerate_same_theta_in_both_arms() {
    // The degenerate oracle: both classes anchor on the *same* theta, so the
    // per-class update must collapse to the classical pooled one.
    let model = mixture_model_with_indiv(
        2,
        "  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL1 * exp(ETA_CL)\n  V = TVV",
    );
    assert_eq!(
        class_anchors(&model, "ETA_CL"),
        Some(vec!["TVCL1".to_string(), "TVCL1".to_string()])
    );
}

#[test]
fn class_aware_mu_ref_accepts_log_sum_form_in_every_arm() {
    // Pattern 2 (`exp(log(THETA) + ETA)`) is a log mu-ref just like Pattern 1.
    let model = mixture_model_with_indiv(
        2,
        "  CL = if (MIXNUM == 1) exp(log(TVCL1) + ETA_CL) else exp(log(TVCL2) + ETA_CL)\n  V = TVV",
    );
    assert_eq!(
        class_anchors(&model, "ETA_CL"),
        Some(vec!["TVCL1".to_string(), "TVCL2".to_string()])
    );
}

#[test]
fn class_aware_mu_ref_rejects_differing_eta_per_arm() {
    let model = mixture_model_with_indiv(
        2,
        "  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_V)\n  V = TVV",
    );
    assert_eq!(class_anchors(&model, "ETA_CL"), None);
    assert_eq!(class_anchors(&model, "ETA_V"), None);
}

#[test]
fn class_aware_mu_ref_rejects_mixed_log_and_additive_arms() {
    let model = mixture_model_with_indiv(
        2,
        "  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 + ETA_CL\n  V = TVV",
    );
    assert_eq!(class_anchors(&model, "ETA_CL"), None);
}

#[test]
fn class_aware_mu_ref_rejects_all_additive_arms() {
    // Additive mu-refs are excluded on purpose: the closed-form shift is only
    // the EM optimum on the log scale.
    let model = mixture_model_with_indiv(
        2,
        "  CL = if (MIXNUM == 1) TVCL1 + ETA_CL else TVCL2 + ETA_CL\n  V = TVV",
    );
    assert_eq!(class_anchors(&model, "ETA_CL"), None);
}

#[test]
fn class_aware_mu_ref_rejects_non_mixnum_condition() {
    let model = mixture_model_with_indiv(
        2,
        "  CL = if (TVV > 5) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)\n  V = TVV",
    );
    assert_eq!(class_anchors(&model, "ETA_CL"), None);
}

#[test]
fn class_aware_mu_ref_rejects_out_of_range_class() {
    // `MIXNUM == 3` is unreachable in a 2-class mixture; reject loudly rather
    // than silently mapping a dead branch onto a class.
    let model = mixture_model_with_indiv(
        2,
        "  CL = if (MIXNUM == 3) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)\n  V = TVV",
    );
    assert_eq!(class_anchors(&model, "ETA_CL"), None);
}

#[test]
fn class_aware_mu_ref_rejects_non_mu_ref_arm() {
    // A class arm with no eta at all is not a mu-ref pattern.
    let model = mixture_model_with_indiv(
        2,
        "  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2\n  V = TVV",
    );
    assert_eq!(class_anchors(&model, "ETA_CL"), None);
}

#[test]
fn class_aware_mu_ref_class_shared_param_stays_in_plain_mu_refs() {
    // A non-switched typical value in a mixture model is an ordinary mu-ref;
    // it must keep appearing in `mu_refs` (the estimators broadcast it across
    // classes themselves).
    let model = mixture_model_with_indiv(
        2,
        "  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)\n  \
         V = TVV * exp(ETA_V)",
    );
    assert_eq!(class_anchors(&model, "ETA_V"), None);
    assert_eq!(
        model.mu_refs.get("ETA_V").map(|m| m.theta_name.as_str()),
        Some("TVV")
    );
}

#[test]
fn class_aware_mu_ref_failure_is_warned_not_silent() {
    let model = mixture_model_with_indiv(
        2,
        "  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_V)\n  V = TVV",
    );
    assert!(
        model
            .parse_warnings
            .iter()
            .any(|w| w.contains("Class-aware mu-referencing not applied") && w.contains("CL")),
        "expected a #996 fallback warning, got {:?}",
        model.parse_warnings
    );
}

#[test]
fn class_aware_mu_ref_success_emits_no_warning() {
    let model = mixture_model_with_indiv(
        2,
        "  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)\n  V = TVV",
    );
    assert!(
        !model
            .parse_warnings
            .iter()
            .any(|w| w.contains("Class-aware mu-referencing not applied")),
        "unexpected fallback warning: {:?}",
        model.parse_warnings
    );
}

#[test]
fn class_aware_mu_ref_no_warning_for_a_class_switched_value_without_iiv() {
    // A class-switched typical value the user deliberately gave no IIV is not a
    // missed mu-ref: there is no eta to anchor, and advising them to add one
    // would be wrong (#996 review).
    let model = mixture_model_with_indiv(
        2,
        "  CL = TVCL1 * exp(ETA_CL)\n  V = if (MIXNUM == 1) TVV else TVV * 2.0",
    );
    assert!(
        !model
            .parse_warnings
            .iter()
            .any(|w| w.contains("Class-aware mu-referencing not applied")),
        "an eta-free MIXNUM expression must not be flagged: {:?}",
        model.parse_warnings
    );
}

#[test]
fn class_aware_mu_ref_no_warning_for_an_intermediate_class_flag() {
    // Same for an intermediate flag variable that merely reads MIXNUM.
    let model = mixture_model_with_indiv(
        2,
        "  FLAG = if (MIXNUM == 1) 1.0 else 2.0\n  \
         CL = TVCL1 * exp(ETA_CL)\n  V = TVV",
    );
    assert!(
        !model
            .parse_warnings
            .iter()
            .any(|w| w.contains("Class-aware mu-referencing not applied")),
        "a MIXNUM flag variable must not be flagged: {:?}",
        model.parse_warnings
    );
}

#[test]
fn class_aware_mu_ref_still_warns_when_an_eta_bearing_switch_fails_to_match() {
    // The guard above must not swallow the real case: the expression carries an
    // eta but is not a mu-ref chain, so the fallback warning still fires.
    let model = mixture_model_with_indiv(
        2,
        "  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) + 1.0 else TVCL2 * exp(ETA_CL)\n  V = TVV",
    );
    assert!(
        model
            .parse_warnings
            .iter()
            .any(|w| w.contains("Class-aware mu-referencing not applied") && w.contains("CL")),
        "expected the #996 fallback warning, got {:?}",
        model.parse_warnings
    );
}

#[test]
fn class_aware_mu_ref_duplicate_eta_keeps_the_last_like_detect_mu_refs() {
    // Two class-aware assignments on the same eta: the mixture map must keep the
    // *last*, matching `detect_mu_refs`' `insert`, so `MixtureSpec::mu_refs` and
    // `CompiledModel::mu_refs` never name different anchors for one eta
    // (#996 review).
    let model = mixture_model_with_indiv(
        2,
        "  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)\n  \
         CL = if (MIXNUM == 1) TVCL2 * exp(ETA_CL) else TVCL3 * exp(ETA_CL)\n  V = TVV",
    );
    assert_eq!(
        class_anchors(&model, "ETA_CL"),
        Some(vec!["TVCL2".to_string(), "TVCL3".to_string()]),
        "the last assignment wins"
    );
}

#[test]
fn class_aware_mu_ref_yields_to_a_later_ordinary_mu_ref_on_the_same_eta() {
    // An ordinary mu-ref written *after* a class-aware one is what
    // `detect_mu_refs` keeps, so the mixture spec must drop its entry — otherwise
    // `get_mixture_mu_ref_pairs` (which prefers the spec) and every
    // `CompiledModel::mu_refs` consumer would use different anchors for one eta.
    let model = mixture_model_with_indiv(
        2,
        "  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)\n  \
         CL = TVCL3 * exp(ETA_CL)\n  V = TVV",
    );
    assert_eq!(
        class_anchors(&model, "ETA_CL"),
        None,
        "the later ordinary mu-ref wins, so no class-aware entry survives"
    );
    assert_eq!(
        model.mu_refs.get("ETA_CL").map(|m| m.theta_name.clone()),
        Some("TVCL3".to_string())
    );
}

#[test]
fn class_aware_mu_ref_survives_an_earlier_ordinary_mu_ref_on_the_same_eta() {
    // The other order: the class-aware assignment is last, so it is the anchor
    // both maps agree on (`get_mixture_mu_ref_pairs` prefers the spec).
    let model = mixture_model_with_indiv(
        2,
        "  CL = TVCL3 * exp(ETA_CL)\n  \
         CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)\n  V = TVV",
    );
    assert_eq!(
        class_anchors(&model, "ETA_CL"),
        Some(vec!["TVCL1".to_string(), "TVCL2".to_string()])
    );
}

#[test]
fn non_mixture_model_has_no_spec() {
    let model = minimal_model_with_indiv("  CL = TVCL * exp(ETA_CL)\n  V = TVV");
    assert!(model.mixture.is_none());
}

#[test]
fn mixnum_resolves_to_class_1_by_default() {
    // With no mixture-class set (Phase 1), `MIXNUM` reads the class-1 default, so
    // the typical-value closure evaluates the class-1 branch: CL = TVCL1 = 1.0.
    let model = parse_model_string(MIXTURE_2CLASS).expect("mixture model parses");
    let theta = &model.default_params.theta;
    let mut cov = std::collections::HashMap::new();
    cov.insert("WT".to_string(), 70.0);
    let tv = model.tv_fn.as_ref().unwrap()(theta, &cov);
    // indiv_param_names order: CL then V.
    let cl_idx = model
        .indiv_param_names
        .iter()
        .position(|n| n == "CL")
        .unwrap();
    assert!(
        (tv[cl_idx] - 1.0).abs() < 1e-9,
        "class-1 branch → CL = TVCL1 = 1.0, got {}",
        tv[cl_idx]
    );
}

#[test]
fn mixnum_in_non_mixture_model_rejected() {
    let err = minimal_model_str_result("  CL = if (MIXNUM == 1) TVCL else TVCL\n  V = TVV")
        .expect_err("MIXNUM without [mixture] must be rejected");
    assert!(
        err.contains("MIXNUM") && err.contains("mixture"),
        "got: {err}"
    );
}

#[test]
fn mixnum_outside_indiv_params_rejected() {
    // #980: MIXNUM anywhere in a model with no [mixture] block must be rejected,
    // not just in [individual_parameters]. The AST guard only scans indiv
    // statements; a raw-token scan of the other blocks catches this [derived] use.
    let src = r"
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)

[derived]
  FOO = MIXNUM
";
    let err = parse_model_string(src)
        .expect_err("MIXNUM in [derived] without [mixture] must be rejected");
    assert!(
        err.contains("MIXNUM") && err.contains("mixture"),
        "got: {err}"
    );
}

#[test]
fn mixing_expression_covariate_registered() {
    // #980: a covariate used only in the mixing expression (logit) must land in
    // referenced_covariates so the data reader loads its column — else it silently
    // reads 0 and the covariate-dependent mixing degrades to intercept-only (the
    // #765 trap, already fixed for the scaling / error-selector / init blocks).
    let model = parse_model_string(MIXTURE_2CLASS).expect("mixture model parses");
    assert!(
        model.referenced_covariates.iter().any(|c| c == "WT"),
        "mixing-expr covariate WT must be registered, got {:?}",
        model.referenced_covariates
    );
}

#[test]
fn mixnum_assignment_rejected() {
    // `MIXNUM = …` in [individual_parameters] of an otherwise-valid mixture model.
    let src = MIXTURE_2CLASS.replace("  V = TVV", "  V = TVV\n  MIXNUM = 2");
    let err = parse_model_string(&src).expect_err("assigning MIXNUM must be rejected");
    assert!(
        err.contains("MIXNUM") && err.contains("reserved"),
        "got: {err}"
    );
}

#[test]
fn mixture_nsub_below_two_rejected() {
    let src = MIXTURE_2CLASS.replace("nsub = 2", "nsub = 1");
    let err = parse_model_string(&src).expect_err("nsub < 2 must be rejected");
    assert!(err.contains("nsub") && err.contains(">= 2"), "got: {err}");
}

#[test]
fn mixture_missing_mixing_expr_rejected() {
    // Declare 3 classes but only supply logit(1); logit(2) is missing.
    let src = MIXTURE_2CLASS.replace("nsub = 2", "nsub = 3");
    let err = parse_model_string(&src).expect_err("missing class-2 mixing expr must be rejected");
    assert!(err.contains("missing mixing expression"), "got: {err}");
}

#[test]
fn mixture_mixing_expr_with_eta_rejected() {
    let src = MIXTURE_2CLASS.replace("logit(1) = MIXL + BWT*WT", "logit(1) = MIXL + ETA_CL");
    let err = parse_model_string(&src).expect_err("eta in mixing expr must be rejected");
    assert!(err.contains("eta"), "got: {err}");
}

#[test]
fn mixture_override_unknown_name_rejected() {
    let src = MIXTURE_2CLASS.replace("omega(2) ETA_CL ~ 0.4", "omega(2) ETA_NOPE ~ 0.4");
    let err = parse_model_string(&src).expect_err("override of unknown eta must be rejected");
    assert!(err.contains("ETA_NOPE"), "got: {err}");
}

#[test]
fn mixture_override_of_base_class_rejected() {
    let src = MIXTURE_2CLASS.replace("omega(2) ETA_CL ~ 0.4", "omega(1) ETA_CL ~ 0.4");
    let err = parse_model_string(&src).expect_err("class-1 override must be rejected");
    assert!(err.contains("class 1 is the base"), "got: {err}");
}

#[test]
fn mixture_mixed_logit_and_prob_forms_rejected() {
    // K=3 with logit(1) and p(2) — forms must not be mixed.
    let src = MIXTURE_2CLASS
        .replace("nsub = 2", "nsub = 3")
        .replace("logit(1) = MIXL + BWT*WT", "logit(1) = MIXL\n  p(2) = 0.3");
    let err = parse_model_string(&src).expect_err("mixed forms must be rejected");
    assert!(err.contains("cannot mix"), "got: {err}");
}

#[test]
fn fit_on_mixture_model_rejects_missing_mixture_params() {
    // Custom init_params with mixture: None (e.g. rebuilt from a FitResult) must
    // be rejected rather than silently running the single-population objective.
    let model = parse_model_string(MIXTURE_2CLASS).expect("mixture model parses");
    let pop = crate::types::Population {
        subjects: vec![],
        covariate_names: vec![],
        dv_column: "DV".to_string(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };
    let mut params = model.default_params.clone();
    params.mixture = None;
    let opts = crate::types::FitOptions::default();
    let err = crate::api::fit(&model, &pop, &params, &opts)
        .expect_err("mixture model with mixture: None params must be rejected");
    assert!(
        err.contains("per-class Omega/Sigma") && err.contains("init_params.mixture"),
        "got: {err}"
    );
}

#[test]
fn fit_on_mixture_model_rejects_unsupported_method() {
    // FOCE/FOCEI, SAEM, Bayes, and IMP objective-evaluation estimate/evaluate a
    // mixture (#985); the remaining estimators (here Gauss-Newton) must still
    // error clearly rather than silently ignoring the mixture structure. The
    // method guard fires before any subject is touched, so an empty population
    // suffices.
    let model = parse_model_string(MIXTURE_2CLASS).expect("mixture model parses");
    let pop = crate::types::Population {
        subjects: vec![],
        covariate_names: vec![],
        dv_column: "DV".to_string(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };
    let params = model.default_params.clone();
    let opts = crate::types::FitOptions {
        method: crate::types::EstimationMethod::FoceGn,
        ..crate::types::FitOptions::default()
    };
    let err = crate::api::fit(&model, &pop, &params, &opts)
        .expect_err("Gauss-Newton on a mixture model must be rejected");
    assert!(
        err.contains("mixture") && err.contains("FOCE"),
        "got: {err}"
    );
}

/// Helper: parse a full model whose `[individual_parameters]` body is `indiv`,
/// returning the raw `Result` so rejection tests can inspect the error. Mirrors
/// [`minimal_model_with_indiv`] but does not unwrap.
fn minimal_model_str_result(indiv: &str) -> Result<crate::types::CompiledModel, String> {
    let model_str = format!(
        r"
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.01

[individual_parameters]
{indiv}

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
"
    );
    parse_model_string(&model_str)
}

// ── Phase 2: per-class Omega/Sigma numeric params + packing round-trip (#977) ──

/// 2-class, 2-eta model. Class 2 overrides only `omega(2) ETA_CL` and
/// `sigma(2) EPS` — `ETA_V` is *not* overridden, so it must track the base.
const MIXTURE_2ETA: &str = r"
[parameters]
  theta TVCL1(1.0, 0.001, 100.0)
  theta TVCL2(3.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.1
  omega ETA_V  ~ 0.2
  sigma EPS ~ 0.01

[mixture]
  nsub = 2
  logit(1) = MIXL
  omega(2) ETA_CL ~ 0.4
  sigma(2) EPS ~ 0.09

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

#[test]
fn mixture_params_built_from_overrides() {
    let model = parse_model_string(MIXTURE_2ETA).expect("parses");
    let mp = model
        .default_params
        .mixture
        .as_ref()
        .expect("mixture params");
    assert_eq!(mp.omega.len(), 2);
    assert_eq!(mp.sigma.len(), 2);
    // Class 1 == base.
    assert!((mp.omega[0].matrix[(0, 0)] - 0.1).abs() < 1e-12);
    assert!((mp.omega[0].matrix[(1, 1)] - 0.2).abs() < 1e-12);
    // Class 2: ETA_CL overridden to 0.4, ETA_V tracks base 0.2.
    assert!((mp.omega[1].matrix[(0, 0)] - 0.4).abs() < 1e-12);
    assert!((mp.omega[1].matrix[(1, 1)] - 0.2).abs() < 1e-12);
    // Sigma stored on SD scale: base EPS = sqrt(0.01) = 0.1; class-2 = sqrt(0.09) = 0.3.
    assert!((mp.sigma[0].values[0] - 0.1).abs() < 1e-12);
    assert!((mp.sigma[1].values[0] - 0.3).abs() < 1e-12);
    // Override addresses: class index 1 (0-based), eta/sigma index 0.
    assert_eq!(mp.omega_override_addr, vec![(1, 0)]);
    assert_eq!(mp.sigma_override_addr, vec![(1, 0)]);
    assert_eq!(mp.omega_override_fixed, vec![false]);
    assert_eq!(mp.sigma_override_fixed, vec![false]);
}

#[test]
fn mixture_packed_len_counts_overrides() {
    use crate::estimation::parameterization::{packed_fixed_mask, packed_len};
    let model = parse_model_string(MIXTURE_2ETA).expect("parses");
    let p = &model.default_params;
    // 4 theta + 2 omega diag + 1 sigma + 2 mixture overrides = 9.
    assert_eq!(packed_len(p), 9);
    assert_eq!(packed_fixed_mask(p).len(), packed_len(p));
}

#[test]
fn mixture_pack_unpack_roundtrip() {
    use crate::estimation::parameterization::{pack_params, unpack_params};
    let model = parse_model_string(MIXTURE_2ETA).expect("parses");
    let p = &model.default_params;
    let v = pack_params(p);
    assert_eq!(v.len(), 9);
    let rt = unpack_params(&v, p);
    let v2 = pack_params(&rt);
    assert_eq!(v.len(), v2.len());
    for (a, b) in v.iter().zip(&v2) {
        assert!(
            (a - b).abs() < 1e-12,
            "pack∘unpack∘pack not identity: {a} vs {b}"
        );
    }
    // Numeric per-class values survive the round trip.
    let mp = rt.mixture.as_ref().expect("mixture survives unpack");
    assert!(
        (mp.omega[1].matrix[(0, 0)] - 0.4).abs() < 1e-10,
        "class-2 ETA_CL override"
    );
    assert!(
        (mp.omega[1].matrix[(1, 1)] - 0.2).abs() < 1e-10,
        "class-2 ETA_V tracks base"
    );
    assert!(
        (mp.sigma[1].values[0] - 0.3).abs() < 1e-10,
        "class-2 EPS override"
    );
    assert!(
        (mp.omega[0].matrix[(0, 0)] - 0.1).abs() < 1e-10,
        "class-1 == base"
    );
}

#[test]
fn mixture_unpack_tracks_perturbed_base() {
    // Perturbing the *base* ETA_V variance in the packed vector must flow into
    // class 2 (which does not override ETA_V), while class 2's overridden ETA_CL
    // stays put. This is the core Phase-2 semantic: non-overridden entries track
    // the base.
    use crate::estimation::parameterization::{coordinate_names, pack_params, unpack_params};
    let model = parse_model_string(MIXTURE_2ETA).expect("parses");
    let p = &model.default_params;
    let names = coordinate_names(p);
    let etav_idx = names
        .iter()
        .position(|n| n == "ETA_V")
        .expect("ETA_V coord");
    let mut v = pack_params(p);
    // Base ETA_V is packed as ln(sqrt(var)); set var := 0.5 → chol_diag = sqrt(0.5).
    v[etav_idx] = 0.5_f64.sqrt().ln();
    let rt = unpack_params(&v, p);
    // Base updated.
    assert!(
        (rt.omega.matrix[(1, 1)] - 0.5).abs() < 1e-10,
        "base ETA_V perturbed"
    );
    let mp = rt.mixture.as_ref().unwrap();
    // Class 2 ETA_V tracks the new base; ETA_CL override unchanged.
    assert!(
        (mp.omega[1].matrix[(1, 1)] - 0.5).abs() < 1e-10,
        "class-2 ETA_V tracks base"
    );
    assert!(
        (mp.omega[1].matrix[(0, 0)] - 0.4).abs() < 1e-10,
        "class-2 ETA_CL override held"
    );
}

#[test]
fn mixture_fixed_override_pinned_in_mask() {
    use crate::estimation::parameterization::{coordinate_names, packed_fixed_mask};
    let src = MIXTURE_2ETA.replace("omega(2) ETA_CL ~ 0.4", "omega(2) ETA_CL ~ 0.4 FIX");
    let model = parse_model_string(&src).expect("parses");
    let p = &model.default_params;
    let mp = p.mixture.as_ref().unwrap();
    assert_eq!(mp.omega_override_fixed, vec![true]);
    // The FIX flag must land at the override's packed coordinate.
    let names = coordinate_names(p);
    let mask = packed_fixed_mask(p);
    let mix_idx = names
        .iter()
        .position(|n| n == "ETA_CL_MIX2")
        .expect("mixture coord name");
    assert!(
        mask[mix_idx],
        "fixed omega override must be pinned in the mask"
    );
}

#[test]
fn mixture_negative_override_rejected() {
    let src = MIXTURE_2ETA.replace("omega(2) ETA_CL ~ 0.4", "omega(2) ETA_CL ~ -0.4");
    let err = parse_model_string(&src).expect_err("negative override must be rejected");
    assert!(err.contains("non-negative"), "got: {err}");
}

#[test]
fn mixture_duplicate_override_rejected() {
    let src = MIXTURE_2ETA.replace(
        "omega(2) ETA_CL ~ 0.4",
        "omega(2) ETA_CL ~ 0.4\n  omega(2) ETA_CL ~ 0.6",
    );
    let err = parse_model_string(&src).expect_err("duplicate override must be rejected");
    assert!(err.contains("duplicate"), "got: {err}");
}

#[test]
fn mixture_block_omega_override_rejected() {
    // Per-class omega override requires a diagonal base omega.
    let src = r"
[parameters]
  theta TVCL1(1.0, 0.001, 100.0)
  theta TVCL2(3.0, 0.001, 100.0)
  theta MIXL(0.0, -10.0, 10.0)
  block_omega (ETA_CL, ETA_V) = [0.1, 0.01, 0.2]
  sigma EPS ~ 0.01

[mixture]
  nsub = 2
  logit(1) = MIXL
  omega(2) ETA_CL ~ 0.4

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = 10 * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";
    let err = parse_model_string(src).expect_err("block-omega override must be rejected");
    assert!(err.contains("diagonal base omega"), "got: {err}");
}

/// #958 (second reported item): the parse-time error for an unknown `gradient` value
/// advertised `'ad'`, but no fit accepts it — the Enzyme AD path was retired in #428 and
/// the engine rejects `gradient = ad` at fit time. So a user who mistyped the option was
/// pointed at a setting that cannot work. The message now lists only what a fit will take.
/// `ad` itself still parses, so requesting it deliberately still reaches the engine's
/// specific "no longer supported, use auto or fd" error rather than a generic one.
#[test]
fn unknown_gradient_value_does_not_advertise_the_retired_ad_route() {
    let err = parse_model_string(
        "[parameters]\n  theta CL(1.0,0.1,10)\n  omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.1\n[individual_parameters]\n  CL = CL * exp(ETA_CL)\n  V = 10\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(PROP)\n[fit_options]\n  gradient = enzyme\n",
    )
    .expect_err("an unknown `gradient` value must be rejected at parse time");
    assert!(
        err.contains("expected 'auto' or 'fd'"),
        "message must list only the values a fit accepts, got: {err}"
    );
    assert!(
        !err.contains("'ad'"),
        "the retired AD route must not be advertised, got: {err}"
    );

    // `ad` still parses — the engine, not the parser, owns its retirement message.
    assert!(
        parse_model_string(
            "[parameters]\n  theta CL(1.0,0.1,10)\n  omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.1\n[individual_parameters]\n  CL = CL * exp(ETA_CL)\n  V = 10\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(PROP)\n[fit_options]\n  gradient = ad\n",
        )
        .is_ok(),
        "`gradient = ad` must still parse so the engine can emit its own retirement error"
    );
}

// ── Residual weighting: `weight = <expr>` (issue #1029) ──────────────────────
//
// The modifier desugars into a #484 per-observation residual magnitude on the
// *additive* sigma slot: weighting is defined as "divide DV and the prediction
// by the weight", so the additive loading picks up a level block `W` and the
// proportional loading is untouched (a common scale factor cancels out of a
// constant-CV error).

/// Study-as-subject MBMA model: one trial-level mean per row, weighted by its
/// reported standard error. `error_block` overrides just the `[error_model]`
/// body so negative tests can vary one line.
fn weighted_model_str(sigma_block: &str, error_block: &str) -> String {
    format!(
        r"
[parameters]
  theta TVCL(0.2)
  theta TVV(10.0)
  omega ETA_CL ~ 0.09
{}
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
{}

[covariates]
  WPSE continuous
  NARM continuous
",
        sigma_block, error_block
    )
}

/// Single sigma, so `additive(ADD_ERR)` lands on slot 0. A single-endpoint
/// `[error_model]` consumes its sigmas positionally from the `[parameters]`
/// declaration order (#1001), so the fixture has to declare exactly the sigmas
/// the error block names, in the order it names them.
const W_SIGMA_ADD: &str = "  sigma ADD_ERR ~ 1.0 (variance) FIX\n";
/// Proportional first, additive second — `combined`'s argument order.
const W_SIGMA_COMBINED: &str = "  sigma PROP_ERR ~ 0.04\n  sigma ADD_ERR ~ 1.0 (variance) FIX\n";
/// Proportional only, for the `weight =`-on-proportional rejection.
const W_SIGMA_PROP: &str = "  sigma PROP_ERR ~ 0.04\n";

/// The additive-slot fixture, the shape most of these tests want.
fn weighted_add_model_str(error_block: &str) -> String {
    weighted_model_str(W_SIGMA_ADD, error_block)
}

#[test]
fn test_weight_modifier_scales_the_additive_slot() {
    let model = parse_model_string(&weighted_add_model_str(
        "  DV ~ additive(ADD_ERR) weight = WPSE",
    ))
    .unwrap();
    let rm = model
        .ruv_magnitude
        .as_ref()
        .expect("`weight =` compiles to a residual magnitude");
    assert!(rm.is_active());
    assert_eq!(rm.per_sigma.len(), 1);
    assert!(
        rm.per_sigma[0].is_some(),
        "additive slot carries the weight"
    );

    let theta = vec![0.2, 10.0];
    let cov: std::collections::HashMap<String, f64> =
        [("WPSE".to_string(), 2.5)].into_iter().collect();
    let m = rm.eval_obs(&theta, &cov, 0.0, 0.0);
    assert!((m[0] - 2.5).abs() < 1e-12, "got {m:?}");
}

#[test]
fn test_weight_modifier_on_combined_leaves_the_proportional_slot_bare() {
    // `W * (f/W) = f`: a common scale factor cancels out of the proportional
    // (constant-CV) loading, so only the additive slot may be weighted.
    let model = parse_model_string(&weighted_model_str(
        W_SIGMA_COMBINED,
        "  DV ~ combined(PROP_ERR, ADD_ERR) weight = WPSE",
    ))
    .unwrap();
    let rm = model.ruv_magnitude.as_ref().expect("magnitude present");
    assert_eq!(rm.per_sigma.len(), 2);
    assert!(rm.per_sigma[0].is_none(), "proportional slot stays bare");
    assert!(
        rm.per_sigma[1].is_some(),
        "additive slot carries the weight"
    );

    let theta = vec![0.2, 10.0];
    let cov: std::collections::HashMap<String, f64> =
        [("WPSE".to_string(), 4.0)].into_iter().collect();
    let m = rm.eval_obs(&theta, &cov, 0.0, 0.0);
    assert!((m[0] - 1.0).abs() < 1e-12, "got {m:?}");
    assert!((m[1] - 4.0).abs() < 1e-12, "got {m:?}");
}

#[test]
fn test_weight_modifier_accepts_an_expression_with_its_own_parens() {
    // The weighted-logit MBMA transform: `weight = 1/sqrt(N)`. The peel must not
    // stop at the expression's own closing paren.
    let model = parse_model_string(&weighted_add_model_str(
        "  DV ~ additive(ADD_ERR) weight = 1.0 / sqrt(NARM)",
    ))
    .unwrap();
    let rm = model.ruv_magnitude.as_ref().expect("magnitude present");
    let theta = vec![0.2, 10.0];
    let cov: std::collections::HashMap<String, f64> =
        [("NARM".to_string(), 16.0)].into_iter().collect();
    let m = rm.eval_obs(&theta, &cov, 0.0, 0.0);
    assert!((m[0] - 0.25).abs() < 1e-12, "got {m:?}");
}

#[test]
fn test_weight_modifier_composes_with_a_custom_magnitude() {
    // `weight =` multiplies whatever magnitude expression the slot already
    // carries (#484) rather than replacing it.
    let model = parse_model_string(&weighted_add_model_str(
        "  DV ~ additive(ADD_ERR * 3.0) weight = WPSE",
    ))
    .unwrap();
    let rm = model.ruv_magnitude.as_ref().expect("magnitude present");
    let theta = vec![0.2, 10.0];
    let cov: std::collections::HashMap<String, f64> =
        [("WPSE".to_string(), 2.0)].into_iter().collect();
    let m = rm.eval_obs(&theta, &cov, 0.0, 0.0);
    assert!((m[0] - 6.0).abs() < 1e-12, "got {m:?}");
}

#[test]
fn test_weight_modifier_rejected_on_proportional_error() {
    let err = expect_parse_err(&weighted_model_str(
        W_SIGMA_PROP,
        "  DV ~ proportional(PROP_ERR) weight = WPSE",
    ));
    assert!(
        err.contains("no effect on a purely proportional"),
        "got: {err}"
    );
}

#[test]
fn test_weight_modifier_rejected_with_per_cmt_error_models() {
    let err = expect_parse_err(&pkpd_model_str(
        "  CMT=1: DV ~ proportional(PROP_ERR_PK)\n  CMT=2: DV ~ additive(ADD_ERR_PD) weight = WT",
    ));
    assert!(err.contains("not supported with per-CMT"), "got: {err}");
}

#[test]
fn test_weight_modifier_rejected_inside_a_covariate_selected_block() {
    let err = expect_parse_err(&weighted_add_model_str(
        "  if (WPSE > 1.0) { DV ~ additive(ADD_ERR) weight = WPSE }\n  else { DV ~ additive(ADD_ERR) }",
    ));
    assert!(
        err.contains("not supported inside a covariate-selected"),
        "got: {err}"
    );
}

#[test]
fn test_weight_modifier_rejects_an_undeclared_covariate() {
    // Inherits #484's guard: an undeclared name would silently evaluate to 0
    // and collapse every observation's weight to zero.
    let err = expect_parse_err(&weighted_add_model_str(
        "  DV ~ additive(ADD_ERR) weight = NOTACOV",
    ));
    assert!(err.contains("undeclared covariate"), "got: {err}");
}

#[test]
fn test_weight_modifier_rejects_an_empty_right_hand_side() {
    let err = expect_parse_err(&weighted_add_model_str("  DV ~ additive(ADD_ERR) weight ="));
    assert!(
        err.contains("no expression on its right-hand side"),
        "got: {err}"
    );
}

#[test]
fn test_weight_modifier_rejects_eta_dependence() {
    // A weight is a property of the datum, never of the individual.
    let err = expect_parse_err(&weighted_add_model_str(
        "  DV ~ additive(ADD_ERR) weight = exp(ETA_CL)",
    ));
    assert!(!err.is_empty(), "an eta-dependent weight must not parse");
}

#[test]
fn test_split_weight_modifier_ignores_a_weight_inside_the_sigma_parens() {
    // A covariate literally named `WEIGHT` inside a magnitude expression is at
    // paren depth 1 and must not be mistaken for the modifier.
    let (stmt, w) = super::split_weight_modifier(
        "DV ~ additive(ADD * WEIGHT)",
        "error_model",
        "a `DV ~ TYPE(...)` statement",
    )
    .unwrap();
    assert_eq!(stmt, "DV ~ additive(ADD * WEIGHT)");
    assert!(w.is_none());
}

#[test]
fn test_split_weight_modifier_ignores_a_comparison() {
    let (stmt, w) = super::split_weight_modifier(
        "DV ~ additive(ADD) weight == 2",
        "error_model",
        "a `DV ~ TYPE(...)` statement",
    )
    .unwrap();
    assert_eq!(stmt, "DV ~ additive(ADD) weight == 2");
    assert!(w.is_none());
}

#[test]
fn test_split_weight_modifier_ignores_a_longer_identifier() {
    let (stmt, w) = super::split_weight_modifier(
        "DV ~ additive(ADD) weighted = 2",
        "error_model",
        "a `DV ~ TYPE(...)` statement",
    )
    .unwrap();
    assert_eq!(stmt, "DV ~ additive(ADD) weighted = 2");
    assert!(w.is_none());
}

#[test]
fn test_split_weight_modifier_requires_a_statement_before_it() {
    let err = super::split_weight_modifier(
        "weight = WPSE",
        "error_model",
        "a `DV ~ TYPE(...)` statement",
    )
    .unwrap_err();
    assert!(
        err.contains("must follow a `DV ~ TYPE(...)` statement"),
        "got: {err}"
    );
}

/// A covariate that appears *only* in a `weight =` / magnitude expression is
/// still model-referenced. Without it in `referenced_covariates`,
/// `Population::prune_irrelevant_tv_covariates` drops the per-observation
/// snapshots for a subject whose only time-varying covariate is that one, and
/// every row is silently scored with the subject's *first* weight — the whole
/// point of the feature, lost with no diagnostic (#484 / #1029).
#[test]
fn test_weight_covariate_is_registered_as_model_referenced() {
    let model = parse_model_string(&weighted_add_model_str(
        "  DV ~ additive(ADD_ERR) weight = WPSE",
    ))
    .unwrap();
    assert!(
        model.referenced_covariates.contains(&"WPSE".to_string()),
        "got: {:?}",
        model.referenced_covariates
    );
}

#[test]
fn test_weight_covariate_survives_tv_snapshot_pruning() {
    use crate::types::{Population, Subject};

    let model = parse_model_string(&weighted_add_model_str(
        "  DV ~ additive(ADD_ERR) weight = WPSE",
    ))
    .unwrap();

    // One subject whose only time-varying column is the weight.
    let snap = |w: f64| -> std::collections::HashMap<String, f64> {
        [("WPSE".to_string(), w)].into_iter().collect()
    };
    let subject = Subject {
        id: "1".to_string(),
        obs_times: vec![1.0, 2.0],
        observations: vec![9.0, 6.0],
        obs_cmts: vec![1, 1],
        cens: vec![0, 0],
        covariates: snap(0.5),
        obs_covariates: vec![snap(0.5), snap(1.5)],
        ..Default::default()
    };
    let mut pop = Population {
        subjects: vec![subject],
        covariate_names: vec!["WPSE".to_string()],
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };

    let pruned = pop.prune_irrelevant_tv_covariates(&model.referenced_covariates);
    assert_eq!(pruned, 0, "the weight covariate is model-referenced");
    assert_eq!(pop.subjects[0].obs_cov(1).get("WPSE"), Some(&1.5));
}

/// `weight =` desugars into the sigma argument (`ADD_ERR` becomes
/// `(ADD_ERR) * (WPSE)`), and #1001's positional sigma-order check reads the
/// sigma name back *out* of that rewritten argument. The two must not blind each
/// other: an order mismatch under a weighted model still has to be reported as
/// an order mismatch, naming the sigma the user wrote, rather than degrading
/// into "references multiple sigmas" — or, worse, passing.
#[test]
fn test_weight_modifier_still_reports_a_sigma_order_mismatch() {
    let err = expect_parse_err(&weighted_model_str(
        "  sigma PROP_ERR ~ 0.04\n  sigma ADD_ERR ~ 1.0 (variance) FIX\n",
        "  DV ~ additive(ADD_ERR) weight = WPSE",
    ));
    assert!(
        err.contains("consumed positionally") && err.contains("ADD_ERR"),
        "got: {err}"
    );
}

/// The weight expression itself must not name a sigma — that would make the
/// desugared argument reference two, which no positional slot can express.
#[test]
fn test_weight_modifier_rejects_a_sigma_in_the_weight_expression() {
    let err = expect_parse_err(&weighted_model_str(
        "  sigma ADD_ERR ~ 1.0 (variance) FIX\n  sigma PROP_ERR ~ 0.04\n",
        "  DV ~ additive(ADD_ERR) weight = PROP_ERR",
    ));
    assert!(err.contains("multiple sigmas"), "got: {err}");
}

// ── #1031: sample-size-weighted IOV (`kappa K ~ γ² weight = W`) ─────────────
//
// `κ_ik ~ N(0, Ω_IOV / W_ik)` — the between-treatment-arm variability of a
// longitudinal MBMA, whose arm-level effect scales with the number of subjects
// behind the arm. The engine applies it by rewriting every reference to the
// kappa as `K / sqrt(W)`, which is the same distribution and the same model the
// modifier replaces.

/// One arm-level random effect on CL, weighted by the arm size `NARM`.
/// `kappa_line` and `extra_block` are varied by the negative tests.
fn weighted_kappa_model_str(kappa_line: &str, extra_block: &str) -> String {
    format!(
        r"
[parameters]
  theta TVCL(0.2)
  theta TVV(10.0)
  omega ETA_CL ~ 0.09
{kappa_line}
  sigma PROP_ERR ~ 0.04

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)

[covariates]
  NARM continuous
{extra_block}
[fit_options]
  iov_column = OCC
"
    )
}

/// The individual parameter a weighted kappa feeds must come out identical to
/// the hand-written `KAPPA / sqrt(NARM)` form — that equivalence is the whole
/// implementation, and it is what lets every estimator downstream stay unaware
/// of the feature.
#[test]
fn test_weighted_kappa_divides_the_kappa_by_sqrt_of_the_weight() {
    let weighted = parse_model_string(&weighted_kappa_model_str(
        "  kappa KAPPA_CL ~ 0.09 weight = NARM",
        "",
    ))
    .unwrap();
    let plain =
        parse_model_string(&weighted_kappa_model_str("  kappa KAPPA_CL ~ 0.09", "")).unwrap();
    let theta = weighted.default_params.theta.clone();
    let cov: std::collections::HashMap<String, f64> =
        [("NARM".to_string(), 4.0)].into_iter().collect();
    // Extended η = [ETA_CL, KAPPA_CL]. κ = 0.8 at NARM = 4 must act as κ = 0.4.
    let got = (weighted.pk_param_fn)(&theta, &[0.1, 0.8], &cov, 0.0);
    let want = (plain.pk_param_fn)(&theta, &[0.1, 0.4], &cov, 0.0);
    approx::assert_relative_eq!(got.values[0], want.values[0], max_relative = 1e-12);
    // ... and it is genuinely scaled, not passed through.
    let unscaled = (plain.pk_param_fn)(&theta, &[0.1, 0.8], &cov, 0.0);
    assert!(
        (got.values[0] - unscaled.values[0]).abs() > 1e-6,
        "weight had no effect: {got:?} vs {unscaled:?}"
    );
}

/// The weight is per *record*, so a covariate that differs between occasions
/// scales each occasion by its own arm size.
#[test]
fn test_weighted_kappa_reads_the_weight_at_the_record() {
    let model = parse_model_string(&weighted_kappa_model_str(
        "  kappa KAPPA_CL ~ 0.09 weight = NARM",
        "",
    ))
    .unwrap();
    let theta = model.default_params.theta.clone();
    let big: std::collections::HashMap<String, f64> =
        [("NARM".to_string(), 100.0)].into_iter().collect();
    let small: std::collections::HashMap<String, f64> =
        [("NARM".to_string(), 1.0)].into_iter().collect();
    let cl_big = (model.pk_param_fn)(&theta, &[0.0, 1.0], &big, 0.0).values[0];
    let cl_small = (model.pk_param_fn)(&theta, &[0.0, 1.0], &small, 0.0).values[0];
    let tvcl = theta[0];
    approx::assert_relative_eq!(cl_big, tvcl * (1.0f64 / 10.0).exp(), max_relative = 1e-12);
    approx::assert_relative_eq!(cl_small, tvcl * 1.0f64.exp(), max_relative = 1e-12);
}

/// Ω_IOV keeps its declared (unweighted) scale — the γ² a published MBMA
/// reports. Only the *effect* is divided.
#[test]
fn test_weighted_kappa_leaves_omega_iov_on_the_unweighted_scale() {
    let model = parse_model_string(&weighted_kappa_model_str(
        "  kappa KAPPA_CL ~ 2.0 (sd) weight = NARM",
        "",
    ))
    .unwrap();
    let iov = model
        .default_params
        .omega_iov
        .as_ref()
        .expect("IOV present");
    approx::assert_relative_eq!(iov.matrix[(0, 0)], 4.0, max_relative = 1e-12);
}

/// The model carries the weight for reporting: the source text plus an
/// evaluator, so a fit can print the effective SD `γ/√W` at a typical arm size.
#[test]
fn test_weighted_kappa_is_reported_on_the_compiled_model() {
    let model = parse_model_string(&weighted_kappa_model_str(
        "  kappa KAPPA_CL ~ 0.09 weight = NARM",
        "",
    ))
    .unwrap();
    assert!(model.has_weighted_kappa());
    let w = model.kappa_weights[0].as_ref().expect("weight recorded");
    assert_eq!(w.expr, "NARM");
    let cov: std::collections::HashMap<String, f64> =
        [("NARM".to_string(), 200.0)].into_iter().collect();
    approx::assert_relative_eq!(
        (w.eval)(&model.default_params.theta, &cov, 0.0),
        200.0,
        max_relative = 1e-12
    );
}

/// An unweighted IOV model must not start carrying weight metadata — the
/// `kappa_weights` slots exist but stay empty.
#[test]
fn test_unweighted_kappa_carries_no_weight() {
    let model =
        parse_model_string(&weighted_kappa_model_str("  kappa KAPPA_CL ~ 0.09", "")).unwrap();
    assert!(!model.has_weighted_kappa());
    assert!(model.kappa_weights.iter().all(|w| w.is_none()));
}

/// A covariate named only by a weight expression is still model-referenced:
/// without it in `referenced_covariates` the per-observation snapshots are
/// pruned and every occasion silently reads the subject's first arm size
/// (the #484 / #1029 trap).
#[test]
fn test_weighted_kappa_registers_its_covariate() {
    let model = parse_model_string(&weighted_kappa_model_str(
        "  kappa KAPPA_CL ~ 0.09 weight = NARM",
        "",
    ))
    .unwrap();
    assert!(
        model.referenced_covariates.contains(&"NARM".to_string()),
        "got: {:?}",
        model.referenced_covariates
    );
}

/// Weighting a kappa must not cost the *BSV* eta sharing its `exp(...)` its
/// mu-reference — that would silently drop SAEM/IMP to the general θ step.
#[test]
fn test_weighted_kappa_preserves_the_bsv_mu_reference() {
    let weighted = parse_model_string(&weighted_kappa_model_str(
        "  kappa KAPPA_CL ~ 0.09 weight = NARM",
        "",
    ))
    .unwrap();
    let plain =
        parse_model_string(&weighted_kappa_model_str("  kappa KAPPA_CL ~ 0.09", "")).unwrap();
    let plain_ref = plain.mu_refs.get("ETA_CL").expect("baseline mu-ref");
    let weighted_ref = weighted
        .mu_refs
        .get("ETA_CL")
        .expect("BSV mu-ref survives the kappa rewrite");
    assert_eq!(weighted_ref.theta_name, plain_ref.theta_name);
    assert_eq!(weighted_ref.log_transformed, plain_ref.log_transformed);
}

#[test]
fn test_weighted_kappa_rejects_a_random_effect_in_the_weight() {
    let err = expect_parse_err(&weighted_kappa_model_str(
        "  kappa KAPPA_CL ~ 0.09 weight = exp(ETA_CL)",
        "",
    ));
    assert!(err.contains("references a random effect"), "got: {err}");
}

#[test]
fn test_weighted_kappa_rejects_an_empty_right_hand_side() {
    let err = expect_parse_err(&weighted_kappa_model_str(
        "  kappa KAPPA_CL ~ 0.09 weight =",
        "",
    ));
    assert!(
        err.contains("no expression on its right-hand side"),
        "got: {err}"
    );
}

/// A weight on anything but a `kappa` is rejected rather than ignored: the
/// unanchored declaration regexes would otherwise drop it silently, which is
/// the failure mode the feature exists to remove.
#[test]
fn test_weight_modifier_rejected_on_a_non_kappa_declaration() {
    let err = expect_parse_err(&weighted_kappa_model_str(
        "  kappa KAPPA_CL ~ 0.09\n  omega ETA_V ~ 0.04 weight = NARM",
        "",
    ));
    assert!(
        err.contains("only supported on a `kappa NAME ~ VALUE` declaration"),
        "got: {err}"
    );
}

/// A `block_kappa` cannot carry a weight: the divisor would have to apply to
/// the block's covariances too for the matrix to stay a valid one.
#[test]
fn test_weight_modifier_rejected_on_block_kappa() {
    let err = expect_parse_err(&weighted_kappa_model_str(
        "  block_kappa (KAPPA_CL, KAPPA_V) = [0.09, 0.01, 0.04] weight = NARM",
        "",
    ));
    assert!(err.contains("`block_kappa` cannot carry one"), "got: {err}");
}

/// Only `[individual_parameters]` applies the scaling, so a weighted kappa
/// named anywhere else would read as the *unweighted* κ — a plausible wrong
/// answer. Reject it instead.
#[test]
fn test_weighted_kappa_rejected_outside_individual_parameters() {
    let err = expect_parse_err(&weighted_kappa_model_str(
        "  kappa KAPPA_CL ~ 0.09 weight = NARM",
        "\n[derived]\n  KOUT = KAPPA_CL\n",
    ));
    assert!(
        err.contains("references `KAPPA_CL`, a kappa declared with `weight = ...`"),
        "got: {err}"
    );
}

/// The unweighted kappa keeps its freedom to appear elsewhere.
#[test]
fn test_unweighted_kappa_is_not_restricted_to_individual_parameters() {
    parse_model_string(&weighted_kappa_model_str(
        "  kappa KAPPA_CL ~ 0.09",
        "\n[derived]\n  KOUT = KAPPA_CL\n",
    ))
    .expect("an unweighted kappa is unaffected by the #1031 guard");
}

/// A weighted kappa that no individual parameter references scales nothing —
/// but the fit still reports `→ SD = γ/√N`, and a covariate named *only* by the
/// weight never reaches `referenced_covariates`, so the datareader skips the
/// column and the data check then blames a present column for reading 0.
#[test]
fn test_weighted_kappa_must_be_referenced_in_individual_parameters() {
    let model_str = weighted_kappa_model_str(
        "  kappa KAPPA_CL ~ 0.09 weight = NARM\n  kappa KAPPA_V ~ 0.04 weight = NARM",
        "",
    );
    let err = expect_parse_err(&model_str);
    assert!(
        err.contains("`KAPPA_V`") && err.contains("never referenced in [individual_parameters]"),
        "got: {err}"
    );
}

/// The same guard must not fire on a kappa that *is* referenced, including one
/// reached only from inside a conditional branch.
#[test]
fn test_weighted_kappa_referenced_in_a_branch_is_not_inert() {
    let model_str = r"
[parameters]
  theta TVCL(0.2)
  theta TVV(10.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA_CL ~ 0.09 weight = NARM
  sigma PROP_ERR ~ 0.04

[individual_parameters]
  if (NARM > 1) {
    CL = TVCL * exp(ETA_CL + KAPPA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)

[covariates]
  NARM continuous

[fit_options]
  iov_column = OCC
";
    parse_model_string(model_str).expect("a kappa referenced inside a branch is not inert");
}

/// An *estimated* theta in the weight moves the divisor during the outer
/// optimisation, so the up-front `E_KAPPA_WEIGHT_NONPOSITIVE` check — evaluated
/// at the initial θ — certifies nothing: a weight that starts positive and
/// crosses zero mid-fit divides an individual parameter by zero silently.
#[test]
fn test_weighted_kappa_rejects_an_estimated_theta_in_the_weight() {
    let err = expect_parse_err(&weighted_kappa_model_str(
        "  kappa KAPPA_CL ~ 0.09 weight = TVCL * NARM",
        "",
    ));
    assert!(
        err.contains("references estimated theta(s) `TVCL`"),
        "got: {err}"
    );
}

/// A FIXed theta is a known constant, so it is as safe as a covariate column
/// and stays allowed.
#[test]
fn test_weighted_kappa_allows_a_fixed_theta_in_the_weight() {
    let model_str = weighted_kappa_model_str(
        "  theta NSCALE(2.0) FIX\n  kappa KAPPA_CL ~ 0.09 weight = NSCALE * NARM",
        "",
    );
    let model = parse_model_string(&model_str).expect("a FIXed theta in the weight is allowed");
    let w = model.kappa_weights[0].as_ref().expect("weight recorded");
    // 2.0 (NSCALE, FIXed) × 100 (NARM) = 200.
    let cov: std::collections::HashMap<String, f64> =
        [("NARM".to_string(), 100.0)].into_iter().collect();
    assert_eq!(
        (w.eval)(&model.default_params.theta, &cov, 0.0),
        200.0,
        "the fixed theta enters the weight"
    );
}

/// The "outside `[individual_parameters]`" error names the block the author
/// wrote (`event_model armA`), not the bare instance label (`armA`), which on
/// its own reads as a block name that does not exist. Every *named* block type
/// is feature-gated, so this is the survival build's case.
#[cfg(feature = "survival")]
#[test]
fn test_weighted_kappa_outside_error_names_the_block_type() {
    let model_str = r"
[parameters]
  theta TVLAMBDA(0.1, 0.001, 10.0)
  omega ETA ~ 0.09
  kappa KAPPA_CL ~ 0.09 weight = NARM

[event_model armA]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA * exp(KAPPA_CL)

[covariates]
  NARM continuous

[fit_options]
  iov_column = OCC
";
    let err = expect_parse_err(model_str);
    assert!(
        err.contains("[event_model armA] references `KAPPA_CL`"),
        "got: {err}"
    );
}

/// Mu-reference detection runs on the *rewritten* statements, so the declared
/// `weight = W` form and the hand-written `K / sqrt(W)` agree on their mu-refs
/// exactly as they agree on the objective. Detecting on a pre-rewrite snapshot
/// registered a kappa mu-ref the rewritten model does not satisfy — which
/// `method = bayes` then rejects while accepting the hand-written twin.
#[test]
fn test_weighted_kappa_mu_refs_match_the_hand_written_form() {
    let declared = parse_model_string(&weighted_kappa_model_str(
        "  kappa KAPPA_CL ~ 0.09 weight = NARM",
        "",
    ))
    .unwrap();
    let hand_written = parse_model_string(
        &weighted_kappa_model_str("  kappa KAPPA_CL ~ 0.09", "")
            .replace("ETA_CL + KAPPA_CL", "ETA_CL + KAPPA_CL / sqrt(NARM)"),
    )
    .unwrap();
    assert_eq!(
        declared.kappa_mu_refs.contains_key("KAPPA_CL"),
        hand_written.kappa_mu_refs.contains_key("KAPPA_CL"),
        "declared and hand-written forms must agree on kappa mu-refs"
    );
    assert!(
        !declared.kappa_mu_refs.contains_key("KAPPA_CL"),
        "a weighted kappa is not a bare Eta after the rewrite, so it is not a mu-ref"
    );
}

/// The order-dependent case the pre-rewrite snapshot got backwards: in
/// `TVCL * exp(KAPPA_CL) * exp(ETA_CL)` the left-to-right `or_else` returns the
/// *kappa* index on the snapshot, losing the BSV eta's mu-ref that survives the
/// rewrite — and with it SAEM/IMP's closed-form θ step.
#[test]
fn test_weighted_kappa_left_of_the_bsv_eta_keeps_the_bsv_mu_reference() {
    let model_str = weighted_kappa_model_str("  kappa KAPPA_CL ~ 0.09 weight = NARM", "").replace(
        "CL = TVCL * exp(ETA_CL + KAPPA_CL)",
        "CL = TVCL * exp(KAPPA_CL) * exp(ETA_CL)",
    );
    let model = parse_model_string(&model_str).unwrap();
    let mu = model
        .mu_refs
        .get("ETA_CL")
        .expect("BSV mu-ref survives a kappa written to its left");
    assert_eq!(mu.theta_name, "TVCL");
    assert!(mu.log_transformed);
}

#[test]
fn test_split_weight_modifier_peels_a_kappa_declaration() {
    let (stmt, w) = super::split_weight_modifier(
        "kappa KAPPA_EMAX ~ 2.0 (sd) weight = NARM",
        "parameters",
        "a `kappa NAME ~ VALUE` declaration",
    )
    .unwrap();
    assert_eq!(stmt, "kappa KAPPA_EMAX ~ 2.0 (sd)");
    assert_eq!(w.as_deref(), Some("NARM"));
}

/// `[covariate_nn]` inputs must count as referenced covariates.
///
/// They are named in the block, not in any expression, so no statement walker sees them.
/// If they are not registered here the model does not treat them as required data
/// columns, `Population::prune_irrelevant_tv_covariates` discards their trajectories, and
/// the network silently reads each subject's baseline value for the whole record.
#[cfg(feature = "nn")]
#[test]
fn covariate_nn_inputs_are_registered_as_referenced_covariates() {
    let model = parse_model_string(
        r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)

[covariate_nn CLNN]
  inputs     = [ZCOV, WT]
  outputs    = [CL]
  layers     = [3]
  activation = tanh
  output     = softplus

[individual_parameters]
  CL = TVCL * CLNN.CL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
",
    )
    .expect("DCM fixture parses");

    for name in ["ZCOV", "WT"] {
        assert!(
            model.referenced_covariates.iter().any(|c| c == name),
            "NN input {name} must be a referenced covariate, got {:?}",
            model.referenced_covariates
        );
    }
}

/// The regression proper: a network whose input is time-varying must still produce
/// time-varying predictions **after the fit pipeline has pruned covariates**.
///
/// Routing through `prune_irrelevant_tv_covariates` is the whole point. That is the call
/// `api::fit` makes, and it is where the bug lived: a subject constructed by hand keeps
/// its `obs_covariates` and predicts correctly whether or not the fix is present, so a
/// test that skips the prune passes either way and proves nothing. (Verified: without the
/// fix this test fails only when the prune is included.)
#[cfg(feature = "nn")]
#[test]
fn covariate_nn_reads_time_varying_covariate_values() {
    use crate::types::{DoseEvent, Subject};
    use std::collections::HashMap;

    let model = parse_model_string(
        r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)

[covariate_nn CLNN]
  inputs     = [ZCOV]
  outputs    = [CL]
  layers     = [3]
  activation = tanh
  output     = softplus

[individual_parameters]
  CL = TVCL * CLNN.CL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
",
    )
    .expect("DCM fixture parses");

    let times = vec![1.0, 4.0, 8.0, 16.0, 24.0];
    // `obs_covariates` carries the per-observation snapshot the engine reads.
    let make = |zcov: &[f64]| -> Subject {
        let per_obs: Vec<HashMap<String, f64>> = zcov
            .iter()
            .map(|&z| HashMap::from([("ZCOV".to_string(), z)]))
            .collect();
        Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: times.clone(),
            obs_raw_times: Vec::new(),
            observations: vec![1.0; times.len()],
            obs_cmts: vec![1; times.len()],
            covariates: HashMap::from([("ZCOV".to_string(), zcov[0])]),
            dose_covariates: vec![HashMap::from([("ZCOV".to_string(), zcov[0])])],
            obs_covariates: per_obs,
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            reset_covariates: Vec::new(),
            cens: vec![0; times.len()],
            occasions: Vec::new(),
            obs_l2: Vec::new(),
            dose_occasions: Vec::new(),
            reset_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
        }
    };

    // Same baseline, but one subject's covariate swings hard after the second sample.
    let mut pop = crate::types::Population {
        subjects: vec![
            make(&[-1.0, -1.0, -1.0, -1.0, -1.0]),
            make(&[-1.0, -1.0, 1.5, 1.5, 1.5]),
        ],
        covariate_names: vec!["ZCOV".to_string()],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };
    assert!(
        pop.subjects[1].has_tv_covariates(),
        "fixture must present time-varying covariates before pruning"
    );

    // The step that broke it: covariates the model does not reference are dropped here.
    pop.prune_irrelevant_tv_covariates(&model.referenced_covariates);

    let constant = &pop.subjects[0];
    let varying = &pop.subjects[1];
    let theta = &model.default_params.theta;
    let mut scratch = crate::pk::EventPkParams::default();
    let p_const =
        crate::pk::compute_predictions_with_tv_into(&model, constant, theta, &[0.0], &mut scratch);
    let p_vary =
        crate::pk::compute_predictions_with_tv_into(&model, varying, theta, &[0.0], &mut scratch);

    // The first two samples share a covariate value, so they must agree exactly; the
    // later ones must not, or the network never saw the change.
    for j in 0..2 {
        assert!(
            (p_const[j] - p_vary[j]).abs() < 1e-12,
            "obs {j} precedes the covariate change and must be identical"
        );
    }
    let diverged = (2..times.len()).any(|j| (p_const[j] - p_vary[j]).abs() > 1e-9);
    assert!(
        diverged,
        "predictions after the covariate change are identical ({p_const:?} vs {p_vary:?}); \
         the NN is reading a frozen baseline covariate instead of the trajectory"
    );
}

/// A `[covariate_nn]` input that the data does not carry must be a hard error, not a
/// silent zero.
///
/// `NamedMlpMapper::forward_raw` zero-fills any input it cannot find, matching the
/// expression evaluator's `unwrap_or(0.0)`. On the hot path that is the right shape --
/// it runs per prediction and must not allocate an error -- but it means a typo'd or
/// unavailable input degenerates the network to a constant with nothing to show for it.
/// The guard is `check_covariates` at fit time, which only sees NN inputs because they
/// are registered as referenced covariates.
///
/// `TIME` is the case a user is most likely to reach for, wanting a time-varying
/// parameter. It is a reserved column rather than a covariate, so it is not in the
/// covariate map and must be rejected like any other absent input.
#[cfg(feature = "nn")]
#[test]
fn covariate_nn_input_missing_from_data_is_rejected() {
    use crate::types::{DoseEvent, Population, Subject};
    use std::collections::HashMap;

    let parse = |inputs: &str| {
        parse_model_string(&format!(
            r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)

[covariate_nn CLNN]
  inputs     = [{inputs}]
  outputs    = [CL]
  layers     = [3]
  activation = tanh
  output     = softplus

[individual_parameters]
  CL = TVCL * CLNN.CL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"
        ))
        .expect("DCM fixture parses")
    };

    let population = Population {
        subjects: vec![Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0],
            obs_raw_times: Vec::new(),
            observations: vec![1.0],
            obs_cmts: vec![1],
            covariates: HashMap::from([("WT".to_string(), 70.0)]),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            reset_covariates: Vec::new(),
            cens: vec![0],
            occasions: Vec::new(),
            obs_l2: Vec::new(),
            dose_occasions: Vec::new(),
            reset_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
        }],
        covariate_names: vec!["WT".to_string()],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };

    // A typo'd covariate name.
    let diags = crate::api::check_covariates(&parse("ZCOV"), &population);
    assert!(
        diags
            .iter()
            .any(|d| d.code == "E_MISSING_COVARIATE" && d.message.contains("ZCOV")),
        "an NN input absent from the data must be rejected, got {diags:?}"
    );

    // `TIME` is not a covariate column, so it must be rejected too rather than
    // silently zero-filling the network's only input.
    let diags = crate::api::check_covariates(&parse("TIME"), &population);
    assert!(
        diags
            .iter()
            .any(|d| d.code == "E_MISSING_COVARIATE" && d.message.contains("TIME")),
        "`inputs = [TIME]` must be rejected, got {diags:?}"
    );

    // A present covariate must not be flagged.
    let diags = crate::api::check_covariates(&parse("WT"), &population);
    assert!(
        diags.is_empty(),
        "a present covariate must pass, got {diags:?}"
    );
}

/// `center` / `scale` must reach the network: the forward pass sees `(x - center)/scale`.
///
/// Asserted by equivalence rather than by inspecting the mapper's fields — a model
/// declaring `center`/`scale` must predict identically to one fed the already-normalised
/// covariate, for the same weights. That pins the transform's direction and its
/// application point, which reading the stored vectors back would not.
#[cfg(feature = "nn")]
#[test]
fn covariate_nn_center_and_scale_normalize_the_inputs() {
    use std::collections::HashMap;

    let build = |extra: &str| {
        parse_model_string(&format!(
            r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)

[covariate_nn CLNN]
  inputs     = [WT]
{extra}  outputs    = [CL]
  layers     = [3]
  activation = tanh
  output     = softplus

[individual_parameters]
  CL = TVCL * CLNN.CL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"
        ))
        .expect("DCM fixture parses")
    };

    let normalized = build("  center     = [70.0]\n  scale      = [12.0]\n");
    let raw = build("");

    // Same auto-generated Glorot weights in both (the init is deterministic).
    let w_norm = &normalized.default_params.theta[2..];
    let w_raw = &raw.default_params.theta[2..];
    assert_eq!(
        w_norm, w_raw,
        "weight init must be identical across the two fixtures"
    );

    let nn_norm = &normalized.covariate_nns[0];
    let nn_raw = &raw.covariate_nns[0];

    // WT = 94 under center 70 / scale 12 is z = 2.0; the un-normalised network must
    // reproduce it exactly when handed 2.0 directly.
    let out_norm = nn_norm
        .mapper
        .forward_raw(w_norm, &HashMap::from([("WT".to_string(), 94.0)]))
        .expect("forward");
    let out_raw = nn_raw
        .mapper
        .forward_raw(w_raw, &HashMap::from([("WT".to_string(), 2.0)]))
        .expect("forward");
    assert!(
        (out_norm[0] - out_raw[0]).abs() < 1e-12,
        "normalised input must equal the pre-normalised one: {out_norm:?} vs {out_raw:?}"
    );

    // And it must NOT equal the raw network fed the raw value, or nothing happened.
    let out_unnormalized = nn_raw
        .mapper
        .forward_raw(w_raw, &HashMap::from([("WT".to_string(), 94.0)]))
        .expect("forward");
    assert!(
        (out_norm[0] - out_unnormalized[0]).abs() > 1e-9,
        "declaring center/scale must change the forward pass"
    );
}

/// Normalisation is what keeps a `tanh` layer out of saturation.
///
/// The motivation for the feature, as a test rather than a claim: at Glorot
/// initialisation a raw `WT ≈ 70` drives every hidden unit to ±1, so the layer's
/// derivative collapses and the network is nearly blind to its input. Standardised
/// inputs leave it responsive. Measured as the spread of the output across the covariate
/// range — a saturated network barely moves.
#[cfg(feature = "nn")]
#[test]
fn normalization_keeps_the_hidden_layer_responsive() {
    use std::collections::HashMap;

    let build = |extra: &str| {
        parse_model_string(&format!(
            r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)

[covariate_nn CLNN]
  inputs     = [WT]
{extra}  outputs    = [CL]
  layers     = [8]
  activation = tanh
  output     = softplus

[individual_parameters]
  CL = TVCL * CLNN.CL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"
        ))
        .expect("DCM fixture parses")
    };

    // Observed weight range, roughly 45-95 kg.
    let wts = [45.0, 55.0, 65.0, 75.0, 85.0, 95.0];
    let spread = |model: &crate::types::CompiledModel| -> f64 {
        let w = &model.default_params.theta[2..];
        let outs: Vec<f64> = wts
            .iter()
            .map(|&x| {
                model.covariate_nns[0]
                    .mapper
                    .forward_raw(w, &HashMap::from([("WT".to_string(), x)]))
                    .expect("forward")[0]
            })
            .collect();
        outs.iter().cloned().fold(f64::MIN, f64::max)
            - outs.iter().cloned().fold(f64::MAX, f64::min)
    };

    let raw_spread = spread(&build(""));
    let norm_spread = spread(&build("  center     = [70.0]\n  scale      = [15.0]\n"));

    assert!(
        norm_spread > 10.0 * raw_spread,
        "standardising the input must leave the network far more responsive across the \
         covariate range: raw spread {raw_spread:.3e}, normalised {norm_spread:.3e}"
    );
}

/// `scale = 0` is a division by zero and must be rejected at parse time, as must a
/// length that does not match `inputs`.
#[cfg(feature = "nn")]
#[test]
fn covariate_nn_rejects_invalid_normalization() {
    let build = |extra: &str| {
        parse_model_string(&format!(
            r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)

[covariate_nn CLNN]
  inputs     = [WT, CRCL]
{extra}  outputs    = [CL]
  layers     = [3]
  activation = tanh
  output     = softplus

[individual_parameters]
  CL = TVCL * CLNN.CL * exp(ETA_CL)
  V  = 10.0

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"
        ))
    };

    let err = build("  scale      = [12.0, 0.0]\n").expect_err("zero scale must be rejected");
    assert!(
        err.contains("scale") && err.contains("CRCL"),
        "error must name the offending input: {err}"
    );

    let err = build("  center     = [70.0]\n").expect_err("short center must be rejected");
    assert!(
        err.contains("center") && err.contains("2"),
        "error must report the expected length: {err}"
    );

    // The identity is always acceptable and matches an undeclared block.
    build("  center     = [0.0, 0.0]\n  scale      = [1.0, 1.0]\n")
        .expect("identity normalisation parses");
}

// ---------------------------------------------------------------------------
// [covariate_nn] `init`
// ---------------------------------------------------------------------------

#[cfg(feature = "nn")]
fn covariate_nn_init_model_src(init_line: &str) -> String {
    format!(
        r#"
[parameters]
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma ADD ~ 0.1

[covariate_nn TYPICAL_PK]
  inputs = [WT, CRCL]
  outputs = [CL, V]
  layers = [4]
  activation = tanh
  output = softplus
{init_line}
[individual_parameters]
  CL = TYPICAL_PK.CL * exp(ETA_CL)
  V  = TYPICAL_PK.V  * exp(ETA_V)
  KA = 1.0

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ additive(ADD)
"#
    )
}

/// The behaviour `init` exists for: at the model's default parameters the
/// network emits exactly the declared values, for every subject's covariates.
///
/// Checked at two very different covariate vectors, because the whole point of
/// zeroing the output-layer weight block is that the starting value does not
/// depend on the inputs. Setting the bias alone would leave a covariate-dependent
/// `W_L · a_{L-1}` on top, of the same order as the value being set.
#[cfg(feature = "nn")]
#[test]
fn covariate_nn_init_sets_the_starting_outputs_exactly() {
    use crate::nn::CovariateMapper;

    let model = parse_model_string(&covariate_nn_init_model_src("  init = [1.25, 18.0]\n"))
        .expect("model with init parses");
    let theta = model.default_params.theta.clone();
    let nn = &model.covariate_nns[0];
    let w = &theta[nn.weights_offset..nn.weights_offset + nn.mapper.n_weights()];

    for (wt, crcl) in [(70.0, 95.0), (45.0, 150.0)] {
        let cov = HashMap::from([("WT".to_string(), wt), ("CRCL".to_string(), crcl)]);
        let out = nn.mapper.forward_raw(w, &cov).expect("forward");
        assert!(
            (out[0] - 1.25).abs() < 1e-12,
            "CL init at WT={wt}: got {}, want 1.25",
            out[0]
        );
        assert!(
            (out[1] - 18.0).abs() < 1e-12,
            "V init at WT={wt}: got {}, want 18.0",
            out[1]
        );
    }
}

/// Without `init` the head starts at `softplus(0) = 0.693` for *every* output —
/// the behaviour that made a DCM's initial volume 0.69 L. Pinned so the
/// difference `init` makes is visible in the test suite rather than only in a
/// changelog entry.
#[cfg(feature = "nn")]
#[test]
fn covariate_nn_without_init_starts_every_output_at_softplus_zero() {
    use crate::nn::CovariateMapper;

    let model = parse_model_string(&covariate_nn_init_model_src("")).expect("model parses");
    let theta = model.default_params.theta.clone();
    let nn = &model.covariate_nns[0];
    let w = &theta[nn.weights_offset..nn.weights_offset + nn.mapper.n_weights()];
    let cov = HashMap::from([("WT".to_string(), 70.0), ("CRCL".to_string(), 95.0)]);
    let out = nn.mapper.forward_raw(w, &cov).expect("forward");
    // Biases are 0 and W_L is Glorot (not zeroed), so the output is near but not
    // exactly softplus(0); the point is the *scale* — every output lands around
    // 0.7 no matter what the parameter means.
    for &v in &out {
        assert!(
            (0.2..2.0).contains(&v),
            "expected an un-initialised head near softplus(0)=0.693, got {v}"
        );
    }
}

/// `init` must not silently accept a value the head cannot emit — a negative
/// target under `softplus` would otherwise become a NaN bias and poison every
/// downstream evaluation.
#[cfg(feature = "nn")]
#[test]
fn covariate_nn_init_rejects_values_outside_the_activation_range() {
    let err = parse_model_string(&covariate_nn_init_model_src("  init = [1.0, -5.0]\n"))
        .expect_err("a negative softplus target must be rejected");
    assert!(
        err.contains("init") && err.contains("softplus"),
        "error should name the key and the activation: {err}"
    );
}

#[cfg(feature = "nn")]
#[test]
fn covariate_nn_init_requires_one_entry_per_output() {
    let err = parse_model_string(&covariate_nn_init_model_src("  init = [1.0]\n"))
        .expect_err("a short init list must be rejected");
    assert!(
        err.contains("one entry per output"),
        "error should explain the arity: {err}"
    );
}

/// `ModelNnGuard::enter_for` must return `None` for a model with no
/// `[covariate_nn]` blocks — the ambient-outputs slot is left untouched, so the
/// generic evaluator's `Op::PushNnOutput` arm stays unreachable rather than
/// reading an empty block.
///
/// Deliberately **not** `#[cfg(feature = "nn")]`. There are two `enter_for`
/// bodies — the real one and a `#[cfg(not(feature = "nn"))]` stub that returns
/// `None` unconditionally — and only one of them is compiled in any given build.
/// An ungated test exercises whichever one this build has, so the contract
/// "a model with no networks declines the guard" is pinned in the base `ci` build
/// and the `nn` build alike. Gating it would leave the stub untested and let a
/// future edit give the two bodies different answers.
#[test]
fn model_nn_guard_declines_a_model_with_no_networks() {
    use crate::parser::model_parser::ModelNnGuard;
    use std::collections::HashMap;

    let src = r#"
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma ADD ~ 0.1

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ additive(ADD)
"#;
    let model = parse_model_string(src).expect("plain model parses");
    // `covariate_nns` is itself an `nn`-gated field, so the premise can only be
    // stated in the `nn` build. In the base build there is no such field and the
    // premise holds by construction.
    #[cfg(feature = "nn")]
    assert!(
        model.covariate_nns.is_empty(),
        "the premise is a model with no networks"
    );

    let cov = HashMap::from([("WT".to_string(), 72.0)]);
    let guard = ModelNnGuard::enter_for(&model, &model.default_params.theta, &cov);
    assert!(
        guard.is_none(),
        "a model with no [covariate_nn] blocks must not install a guard"
    );
}

// ── #811: compartment-free (`$PRED`-equivalent) structural model ─────────────

/// A compartment-free model: `[structural_model]` holds the equation itself, with
/// no `pk` / `ode` line anywhere. `equations` is the block body.
fn algebraic_model_str(equations: &str) -> String {
    format!(
        "[parameters]\n\
        \x20 theta TVE0(10.0, 0.1, 100.0)\n\
        \x20 theta TVEMAX(5.0, 0.1, 100.0)\n\
        \x20 theta TVET50(2.0, 0.01, 100.0)\n\
        \x20 omega ETA_E0 ~ 0.09\n\
        \x20 sigma PROP ~ 0.02 (sd)\n\n\
        [individual_parameters]\n\
        \x20 E0   = TVE0 * exp(ETA_E0)\n\
        \x20 EMAX = TVEMAX\n\
        \x20 ET50 = TVET50\n\n\
        [structural_model]\n\
        {equations}\n\
        [error_model]\n\
        \x20 DV ~ proportional(PROP)\n"
    )
}

/// The base case: an Emax time-course with no compartments at all parses, and is
/// recognised as compartment-free — no ODE spec, a Form C readout whose state
/// layout is empty.
#[test]
fn algebraic_structural_model_parses_without_pk_or_ode() {
    let model = parse_model_string(&algebraic_model_str(
        "  y = E0 - EMAX * TIME / (ET50 + TIME)\n",
    ))
    .expect("a compartment-free [structural_model] parses");
    assert!(
        model.is_algebraic(),
        "must be recognised as compartment-free"
    );
    assert!(model.ode_spec.is_none(), "no ODE spec");
    let ar = model
        .analytic_readout
        .as_ref()
        .expect("the equation compiles to a Form C readout");
    assert!(
        ar.state_names.is_empty(),
        "a compartment-free readout reconstructs no amounts, got {:?}",
        ar.state_names
    );
}

/// Named intermediates (#1030) work here for free: the equations are moved into
/// the Form C pipeline verbatim, so a `y` built from bindings above it compiles to
/// the same readout as the inlined form.
#[test]
fn algebraic_structural_model_accepts_named_intermediates() {
    let model = parse_model_string(&algebraic_model_str(
        "  EFF = EMAX * TIME / (ET50 + TIME)\n  y   = E0 - EFF\n",
    ))
    .expect("intermediates are usable in a compartment-free model");
    assert!(model.is_algebraic());
    assert!(
        model
            .analytic_readout
            .as_ref()
            .is_some_and(|ar| ar.program.is_some()),
        "the readout carries a sensitivity program"
    );
}

/// The block must produce a prediction. Intermediates alone are not a model, and
/// the diagnostic says so rather than surfacing the Form C pipeline's downstream
/// "a binding no entry reads".
#[test]
fn algebraic_structural_model_requires_a_y_entry() {
    let err = expect_parse_err(&algebraic_model_str("  EFF = EMAX * TIME\n"));
    assert!(
        err.contains("[structural_model]") && err.contains("`y = <expr>`"),
        "got: {err}"
    );
}

/// `obs_scale` divides a *built-in* prediction; a compartment-free model has none.
#[test]
fn algebraic_structural_model_rejects_obs_scale() {
    let err = expect_parse_err(&algebraic_model_str("  y = E0\n  obs_scale = 1000\n"));
    assert!(
        err.contains("obs_scale") && err.contains("compartment-free"),
        "got: {err}"
    );
}

/// A compartment amount cannot be referenced when there are no compartments. The
/// message names the real problem (no compartment model declared) instead of
/// falling through to "undefined identifier" or a silent covariate lookup.
#[test]
fn algebraic_structural_model_rejects_a_compartment_reference() {
    for name in ["central", "depot", "peripheral"] {
        let err = expect_parse_err(&algebraic_model_str(&format!("  y = {name} / E0\n")));
        assert!(
            err.contains(name) && err.contains("no compartments"),
            "for `{name}`, got: {err}"
        );
    }
}

/// The equation lives in `[structural_model]`, so its diagnostics must say so —
/// the `[scaling]` pipeline that parses it after the desugaring must not leak the
/// block name the user never wrote.
#[test]
fn algebraic_structural_model_diagnostics_name_the_structural_block() {
    let err = expect_parse_err(&algebraic_model_str("  y = E0\n  y = EMAX\n"));
    assert!(
        err.contains("[structural_model]"),
        "diagnostic must name [structural_model], got: {err}"
    );
    assert!(
        !err.contains("[scaling]"),
        "diagnostic must not name a block the user never wrote, got: {err}"
    );
}

/// A name the model does not define is a **data column** — that is how an MBMA
/// equation reaches the per-row covariates it regresses on (arm size, dose,
/// study-level flags). It must land in `referenced_covariates` so the data check
/// requires it, exactly as in `[scaling]`.
#[test]
fn algebraic_structural_model_registers_free_names_as_covariates() {
    let model = parse_model_string(&algebraic_model_str("  y = E0 + EMAX * DOSE\n"))
        .expect("an undeclared name in the equation is a data column");
    assert!(
        model.referenced_covariates.iter().any(|c| c == "DOSE"),
        "`DOSE` must become a required data column, got {:?}",
        model.referenced_covariates
    );
}

/// `[scaling]` alongside a compartment-free model would either declare a second
/// readout or divide the equation's own output.
#[test]
fn algebraic_structural_model_rejects_a_scaling_block() {
    let src = format!(
        "{}\n[scaling]\n  y = E0\n",
        algebraic_model_str("  y = E0 - EMAX\n")
    );
    let err = expect_parse_err(&src);
    assert!(
        err.contains("[scaling]") && err.contains("compartment-free"),
        "got: {err}"
    );
}

/// Blocks that only mean something with compartments underneath are rejected by
/// name, rather than silently ignored (`[odes]` is only read when
/// `[structural_model]` declares `ode(...)`).
#[test]
fn algebraic_structural_model_rejects_compartment_only_blocks() {
    for (block, body) in [
        ("odes", "  d/dt(central) = -0.1 * central\n"),
        ("initial_conditions", "  init(central) = 1.0\n"),
        ("diffusion", "  central ~ 0.1\n"),
    ] {
        let src = format!(
            "{}\n[{block}]\n{body}",
            algebraic_model_str("  y = E0 - EMAX\n")
        );
        let err = expect_parse_err(&src);
        assert!(
            err.contains(&format!("[{block}]")) && err.contains("compartment-free"),
            "for [{block}], got: {err}"
        );
    }
}

/// A mistyped disposition used to parse: the `pk` matcher was unanchored, so
/// `zpk one_cpt_iv(...)` matched `pk one_cpt_iv(...)` inside it and the model
/// silently became a one-compartment IV fit (#811).
#[test]
fn structural_model_rejects_a_mistyped_pk_line() {
    let err = expect_parse_err(&algebraic_model_str(
        "  zpk one_cpt_iv(cl=E0, v=EMAX)\n  y = E0\n",
    ));
    assert!(
        err.contains("unrecognized line") && err.contains("zpk"),
        "got: {err}"
    );
}

/// Every line in the block is now classified, so a stray one errors instead of
/// being dropped — before #811 the scan returned on the first `pk` match and
/// discarded the rest of the block without a word.
#[test]
fn structural_model_rejects_a_stray_line_next_to_a_pk_model() {
    let src = "[parameters]\n  theta TVCL(0.2, 0.001, 10.0)\n  theta TVV(10.0, 0.1, 500.0)\n\
        \n[individual_parameters]\n  CL = TVCL\n  V = TVV\n\
        \n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n  this is not a directive\n\
        \n[error_model]\n  DV ~ additive(1.0)\n";
    let err = expect_parse_err(src);
    assert!(err.contains("unrecognized line"), "got: {err}");
}

/// A readout on top of a compartment model belongs in `[scaling]`; mixing the two
/// forms in one block is ambiguous about which one is the model.
#[test]
fn structural_model_rejects_mixing_a_compartment_model_with_an_equation() {
    let src = "[parameters]\n  theta TVCL(0.2, 0.001, 10.0)\n  theta TVV(10.0, 0.1, 500.0)\n\
        \n[individual_parameters]\n  CL = TVCL\n  V = TVV\n\
        \n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n  y = central / V\n\
        \n[error_model]\n  DV ~ additive(1.0)\n";
    let err = expect_parse_err(src);
    assert!(
        err.contains("cannot mix") && err.contains("[scaling]"),
        "got: {err}"
    );
}

/// Two dispositions in one block used to be silently resolved as "first one wins".
#[test]
fn structural_model_rejects_two_disposition_lines() {
    let src = "[parameters]\n  theta TVCL(0.2, 0.001, 10.0)\n  theta TVV(10.0, 0.1, 500.0)\n\
        \n[individual_parameters]\n  CL = TVCL\n  V = TVV\n\
        \n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n  pk one_cpt_oral(cl=CL, v=V, ka=CL)\n\
        \n[error_model]\n  DV ~ additive(1.0)\n";
    let err = expect_parse_err(src);
    assert!(err.contains("more than one structural model"), "got: {err}");
}

/// An empty `[structural_model]` is a model that predicts nothing. Block
/// extraction records a block only once it has a content line, so an empty one is
/// indistinguishable from an absent one and reports as missing — which is the
/// same instruction to the user. Pinned so the #811 classifier, which has its own
/// "is empty" arm for a block that reaches it with no classifiable line, cannot
/// quietly take over this case with a worse message. (A TTE-only model omits the
/// block entirely, which is a different, valid case.)
#[test]
fn structural_model_rejects_an_empty_block() {
    let src = "[parameters]\n  theta TVCL(0.2, 0.001, 10.0)\n\
        \n[individual_parameters]\n  CL = TVCL\n\
        \n[structural_model]\n\
        \n[error_model]\n  DV ~ additive(1.0)\n";
    let err = expect_parse_err(src);
    assert!(
        err.contains("Missing [structural_model] block"),
        "got: {err}"
    );
}

/// A compartment-free model lays its parameters out like an ODE model — every
/// individual parameter gets a slot from the full `MAX_PK_PARAMS` layout — so it
/// is not bound by the handful of spare slots an analytical Form C readout draws
/// from. This is what lets an MBMA-style model carry dozens of fixed effects.
#[test]
fn algebraic_structural_model_allows_many_readout_parameters() {
    // Well past the `0..=PK_IDX_MTT` spare region an analytical readout is
    // confined to (at most 11 slots, and fewer once a closed form takes its own).
    const N: usize = 40;
    let mut params = String::new();
    let mut indiv = String::new();
    let mut terms: Vec<String> = Vec::new();
    for i in 0..N {
        params.push_str(&format!("  theta TV{i}(1.0, 0.01, 100.0)\n"));
        indiv.push_str(&format!("  P{i} = TV{i}\n"));
        terms.push(format!("P{i}"));
    }
    let src = format!(
        "[parameters]\n{params}  sigma PROP ~ 0.02 (sd)\n\n\
        [individual_parameters]\n{indiv}\n\
        [structural_model]\n  y = {}\n\n\
        [error_model]\n  DV ~ proportional(PROP)\n",
        terms.join(" + ")
    );
    let model = parse_model_string(&src)
        .expect("a compartment-free model is not bound by the analytical spare-slot region");
    assert!(model.is_algebraic());
}

/// A bare `THETA(i)` / `ETA(k)` in the equation must ride the #486 desugaring:
/// the parser appends a synthetic individual parameter and rewrites the reference
/// to it, so the readout stays dual-evaluable and keeps the analytic gradient.
///
/// The synthetics are appended on the ODE-layout path (which a compartment-free
/// model takes), but the *acceptance list* was then overwritten with the analytical
/// allocator's empty result — so the reference was never rewritten, the readout
/// reported `dual_evaluable = false`, and the model silently dropped to finite
/// differences while carrying a warning blaming the readout's shape.
#[test]
fn algebraic_structural_model_desugars_a_bare_theta_and_eta() {
    let src = "[parameters]\n\
        \x20 theta TVE0(10.0, 0.1, 100.0)\n\
        \x20 theta TVSLOPE(0.5, -5.0, 5.0)\n\
        \x20 omega ETA_E0 ~ 0.09\n\
        \x20 sigma PROP ~ 0.02 (sd)\n\n\
        [individual_parameters]\n\
        \x20 E0 = TVE0 * exp(ETA_E0)\n\n\
        [structural_model]\n\
        \x20 y = E0 + TVSLOPE * TIME + ETA_E0\n\n\
        [error_model]\n\
        \x20 DV ~ proportional(PROP)\n";
    let model = parse_model_string(src).expect("a bare theta/eta in the equation parses");
    assert!(model.is_algebraic());
    let program = model
        .analytic_readout
        .as_ref()
        .and_then(|ar| ar.program.as_ref())
        .expect("the equation compiles to a readout program");
    assert!(
        program.is_dual_evaluable(),
        "a desugared THETA/ETA must leave the readout dual-evaluable — otherwise the \
         model loses its analytic gradient"
    );
    assert!(
        !model
            .parse_warnings
            .iter()
            .any(|w| w.contains("finite-difference")),
        "no FD-fallback warning should be emitted, got {:?}",
        model.parse_warnings
    );
    assert!(
        crate::sens::algebraic::supported(&model),
        "and the analytic walks must actually serve it"
    );
    // Exactly one synthetic per referenced θ/η. The append runs on the ODE-layout
    // path a compartment-free model takes; a second append from the analytical
    // allocator would double them and silently consume two PK slots each.
    let synths: Vec<&String> = model
        .indiv_param_names
        .iter()
        .filter(|n| n.starts_with(READOUT_SYNTH_PREFIX))
        .collect();
    assert_eq!(
        synths.len(),
        2,
        "one synthetic for TVSLOPE and one for ETA_E0, got {synths:?}"
    );
}

/// A compartment name is rejected in the equation because the model has no
/// compartments — but a parameter the user *declared* with that name is not a
/// compartment reference at all, and the rejection's own advice ("define it in
/// [individual_parameters]") is what they already did.
#[test]
fn algebraic_structural_model_accepts_a_declared_parameter_named_like_a_compartment() {
    for name in ["central", "depot", "periph"] {
        let src = format!(
            "[parameters]\n\
            \x20 theta TVE0(10.0, 0.1, 100.0)\n\
            \x20 sigma PROP ~ 0.02 (sd)\n\n\
            [individual_parameters]\n\
            \x20 {name} = TVE0\n\n\
            [structural_model]\n\
            \x20 y = {name} * 2\n\n\
            [error_model]\n\
            \x20 DV ~ proportional(PROP)\n"
        );
        let model = parse_model_string(&src).unwrap_or_else(|e| {
            panic!("`{name}` is a declared individual parameter here, not a compartment: {e}")
        });
        assert!(model.is_algebraic());
    }
}

// ── #1064: θ level blocks ────────────────────────────────────────
//
// An unstructured placebo effect in an MBMA model gives every (study ×
// timepoint) cell its own fixed effect — hundreds of θ. What makes that
// tractable is that the block is read by a *gather*: one `PkParams` slot, one
// contiguous θ block, one index per row.
mod theta_vector_blocks {
    use super::*;

    /// A one-compartment IV model reading `PLACEBO[PLA_IDX]` into `CL`.
    fn gather_model(len: usize) -> String {
        format!(
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
        )
    }

    fn cov(name: &str, value: f64) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert(name.to_string(), value);
        m
    }

    #[test]
    fn vector_theta_expands_to_one_named_theta_per_level() {
        let parsed = parse_full_model(&gather_model(4)).unwrap();
        assert_eq!(parsed.model.n_theta, 6, "TVCL + 4 levels + TVV");
        assert_eq!(
            parsed.model.theta_names,
            vec![
                "TVCL".to_string(),
                "PLACEBO[1]".to_string(),
                "PLACEBO[2]".to_string(),
                "PLACEBO[3]".to_string(),
                "PLACEBO[4]".to_string(),
                "TVV".to_string(),
            ]
        );
    }

    #[test]
    fn vector_theta_broadcasts_init_and_bounds_to_every_level() {
        let parsed = parse_full_model(&gather_model(3)).unwrap();
        let p = &parsed.model.default_params;
        for k in 1..=3 {
            assert_eq!(p.theta[k], 0.5, "init broadcast to level {k}");
            assert_eq!(p.theta_lower[k], -10.0, "lower broadcast to level {k}");
            assert_eq!(p.theta_upper[k], 10.0, "upper broadcast to level {k}");
            assert!(!p.theta_fixed[k]);
        }
    }

    #[test]
    fn vector_theta_broadcasts_fix_to_every_level() {
        let content = r#"
[parameters]
  theta TVCL(2.0, 0.001, 10.0)
  theta PLACEBO[3](0.5, -10.0, 10.0, FIX)
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
"#;
        let parsed = parse_full_model(content).unwrap();
        assert_eq!(
            parsed.model.default_params.theta_fixed,
            vec![false, true, true, true, false]
        );
    }

    #[test]
    fn gather_reads_the_level_the_index_column_selects() {
        let parsed = parse_full_model(&gather_model(4)).unwrap();
        // θ = [TVCL, PLACEBO[1..4], TVV]
        let theta = vec![2.0, 0.1, 0.2, 0.3, 0.4, 10.0];
        let eta = vec![0.0];
        for level in 1..=4usize {
            let p = (parsed.model.pk_param_fn)(&theta, &eta, &cov("PLA_IDX", level as f64), 0.0);
            let expected = 2.0 + theta[level];
            assert!(
                (p.values[0] - expected).abs() < 1e-12,
                "PLA_IDX={level} → CL={expected}, got {}",
                p.values[0]
            );
        }
    }

    #[test]
    fn out_of_range_index_is_nan_not_a_silent_zero() {
        // Division by zero already underflows to 0.0 in this evaluator, so a
        // gather that returned 0.0 for a bad index would be indistinguishable
        // from a legitimately-estimated level. It must be NaN.
        let parsed = parse_full_model(&gather_model(4)).unwrap();
        let theta = vec![2.0, 0.1, 0.2, 0.3, 0.4, 10.0];
        let eta = vec![0.0];
        for bad in [0.0, 5.0, -1.0, 2.5] {
            let p = (parsed.model.pk_param_fn)(&theta, &eta, &cov("PLA_IDX", bad), 0.0);
            assert!(p.values[0].is_nan(), "PLA_IDX={bad} must give NaN");
        }
    }

    #[test]
    fn literal_index_folds_to_a_plain_theta_read() {
        let content = r#"
[parameters]
  theta TVCL(2.0, 0.001, 10.0)
  theta PLACEBO[4](0.5, -10.0, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02
[individual_parameters]
  CL = TVCL + PLACEBO[3]
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let parsed = parse_full_model(content).unwrap();
        let theta = vec![2.0, 0.1, 0.2, 0.3, 0.4, 10.0];
        let p = (parsed.model.pk_param_fn)(&theta, &[0.0], &HashMap::new(), 0.0);
        assert!((p.values[0] - 2.3).abs() < 1e-12, "PLACEBO[3] = 0.3");
        // No index column is read, so nothing is recorded for the data check.
        assert_eq!(parsed.model.theta_blocks().index_columns().count(), 0);
    }

    #[test]
    fn out_of_range_literal_index_is_a_parse_error() {
        for bad in ["0", "5", "1.5"] {
            let content = format!(
                r#"
[parameters]
  theta TVCL(2.0, 0.001, 10.0)
  theta PLACEBO[4](0.5, -10.0, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02
[individual_parameters]
  CL = TVCL + PLACEBO[{bad}]
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#
            );
            let err = parse_full_model(&content).err().unwrap();
            assert!(
                err.contains("level index must be an integer in 1..=4"),
                "index {bad} should be rejected, got: {err}"
            );
        }
    }

    #[test]
    fn bare_reference_to_an_explicit_vector_is_an_error() {
        // Without an index there is nothing to gather on; silently treating
        // `PLACEBO` as a covariate would read 0.0 forever.
        let content = r#"
[parameters]
  theta TVCL(2.0, 0.001, 10.0)
  theta PLACEBO[4](0.5, -10.0, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02
[individual_parameters]
  CL = TVCL + PLACEBO
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let err = parse_full_model(content).err().unwrap();
        assert!(
            err.contains("is a vector of 4 θ levels"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn zero_length_vector_is_rejected() {
        let err = parse_full_model(&gather_model(0)).err().unwrap();
        assert!(err.contains("at least one level"), "got: {err}");
    }

    #[test]
    fn gather_index_column_is_recorded_for_the_pre_fit_check() {
        let parsed = parse_full_model(&gather_model(4)).unwrap();
        let uses: Vec<_> = parsed.model.theta_blocks().index_columns().collect();
        assert_eq!(uses, vec![("PLACEBO", "PLA_IDX", 4usize)]);
    }

    #[test]
    fn a_gather_occupies_one_pk_param_slot_not_one_per_level() {
        // The whole point: 800 levels must not need 800 `PkParams` slots.
        let parsed = parse_full_model(&gather_model(800)).unwrap();
        assert_eq!(parsed.model.n_theta, 802);
        assert_eq!(
            parsed.model.indiv_param_names,
            vec!["CL".to_string(), "V".to_string()],
            "the block is a gather — CL and V are still the only individual parameters"
        );
    }
}

// ── #1064: the level-block primitives, branch by branch ────────────────────
//
// The declaration surface and the end-to-end binding are covered by
// `theta_vector_blocks` above and `api::levels`. What is left here are the
// error and edge branches of the primitives themselves — reachable directly,
// and several of them not reachable at all through a well-formed model file.
mod theta_level_primitives {
    use super::*;

    fn spec(levels: Vec<LevelRule>) -> GatherSpec {
        GatherSpec {
            name: "PLACEBO".to_string(),
            levels,
        }
    }

    fn binding(labels: &[&str], groups: &[usize], contrast: LevelContrast) -> LevelBinding {
        LevelBinding {
            labels: labels.iter().map(|s| s.to_string()).collect(),
            groups: groups.to_vec(),
            contrast,
        }
    }

    // ── parse_theta_block_spec ──────────────────────────────────────────────

    #[test]
    fn a_digits_only_bracket_is_a_level_count() {
        assert_eq!(
            parse_theta_block_spec("P", " 12 ").unwrap(),
            ThetaBlockSpec::Count(12)
        );
    }

    #[test]
    fn a_zero_level_count_is_rejected() {
        let err = parse_theta_block_spec("P", "0").unwrap_err();
        assert!(err.contains("at least one level"), "{err}");
    }

    #[test]
    fn a_level_count_too_large_for_usize_is_rejected_not_wrapped() {
        // All-digits but unparseable — the branch that would otherwise panic on
        // `unwrap` or silently wrap.
        let huge = "9".repeat(40);
        let err = parse_theta_block_spec("P", &huge).unwrap_err();
        assert!(err.contains("bad level count"), "{err}");
    }

    #[test]
    fn columns_after_the_contrast_modifier_are_rejected() {
        let err = parse_theta_block_spec("P", "STUDY, contrast = ref, TIME").unwrap_err();
        assert!(err.contains("must come before `contrast = ...`"), "{err}");
    }

    #[test]
    fn a_repeated_contrast_modifier_is_rejected() {
        let err =
            parse_theta_block_spec("P", "STUDY, contrast = ref, contrast = none").unwrap_err();
        assert!(err.contains("`contrast` given twice"), "{err}");
    }

    #[test]
    fn a_column_name_with_punctuation_is_rejected() {
        let err = parse_theta_block_spec("P", "STUDY-ARM, TIME").unwrap_err();
        assert!(err.contains("not a valid data column name"), "{err}");
    }

    #[test]
    fn empty_entries_between_commas_are_skipped() {
        // `[STUDY, , TIME]` is sloppy but unambiguous.
        assert_eq!(
            parse_theta_block_spec("P", "STUDY, , TIME").unwrap(),
            ThetaBlockSpec::Columns {
                columns: vec!["STUDY".to_string(), "TIME".to_string()],
                contrast: LevelContrast::Auto,
            }
        );
    }

    #[test]
    fn every_contrast_spelling_parses() {
        for (token, expected) in [
            ("auto", LevelContrast::Auto),
            ("sum_to_zero", LevelContrast::SumToZero),
            ("sum_to_zero_within", LevelContrast::SumToZeroWithin),
            ("sum_to_zero_within_group", LevelContrast::SumToZeroWithin),
            ("ref", LevelContrast::Ref),
            ("reference", LevelContrast::Ref),
            ("first", LevelContrast::Ref),
            ("none", LevelContrast::Unconstrained),
            ("unconstrained", LevelContrast::Unconstrained),
            ("  REF  ", LevelContrast::Ref),
        ] {
            let parsed =
                parse_theta_block_spec("P", &format!("STUDY, contrast = {token}")).unwrap();
            match parsed {
                ThetaBlockSpec::Columns { contrast, .. } => {
                    assert_eq!(contrast, expected, "token {token}")
                }
                other => panic!("expected a column block for {token}, got {other:?}"),
            }
        }
    }

    // ── build_level_rules ───────────────────────────────────────────────────

    #[test]
    fn a_binding_whose_labels_and_groups_disagree_is_rejected() {
        let b = binding(&["a", "b"], &[0], LevelContrast::SumToZero);
        let err = build_level_rules("P", &b, 0).unwrap_err();
        assert!(err.contains("2 labels but 1 group ids"), "{err}");
    }

    #[test]
    fn a_binding_with_no_levels_is_rejected() {
        let b = binding(&[], &[], LevelContrast::SumToZero);
        let err = build_level_rules("P", &b, 0).unwrap_err();
        assert!(err.contains("no observed level combinations"), "{err}");
    }

    #[test]
    fn a_non_contiguous_contrast_group_is_rejected() {
        // The `NegSum` range is a single slice, so a group that re-opens after
        // another would silently sum the wrong levels. The binder orders them
        // correctly; this is the tripwire for any future caller that does not.
        let b = binding(&["a", "b", "c"], &[0, 1, 0], LevelContrast::SumToZero);
        let err = build_level_rules("P", &b, 0).unwrap_err();
        assert!(err.contains("is not contiguous"), "{err}");
    }

    #[test]
    fn unconstrained_bindings_estimate_every_level() {
        let b = binding(&["a", "b", "c"], &[0, 0, 0], LevelContrast::Unconstrained);
        let (rules, names) = build_level_rules("P", &b, 5).unwrap();
        assert_eq!(
            rules,
            vec![LevelRule::Free(5), LevelRule::Free(6), LevelRule::Free(7)]
        );
        assert_eq!(names, vec!["P[a]", "P[b]", "P[c]"]);
    }

    #[test]
    fn a_reference_binding_pins_the_first_level_with_an_empty_negsum() {
        let b = binding(&["a", "b"], &[0, 0], LevelContrast::Ref);
        let (rules, names) = build_level_rules("P", &b, 3).unwrap();
        // An empty `NegSum` range evaluates to 0.0 — that is how a reference
        // level is encoded, rather than as a special rule variant.
        assert_eq!(rules[0], LevelRule::NegSum(3, 3));
        assert_eq!(rules[1], LevelRule::Free(3));
        assert_eq!(names, vec!["P[b]"]);
    }

    // ── the gather itself ───────────────────────────────────────────────────

    #[test]
    fn eval_gather_rejects_every_ill_formed_index() {
        let s = spec(vec![LevelRule::Free(0), LevelRule::Free(1)]);
        let theta = [1.0, 2.0];
        for bad in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            3.0,
            1.5,
            -1.0,
        ] {
            assert!(
                eval_gather(&s, &theta, bad).is_nan(),
                "index {bad} must be NaN, never a silent value"
            );
        }
        assert_eq!(eval_gather(&s, &theta, 1.0), 1.0);
        assert_eq!(eval_gather(&s, &theta, 2.0), 2.0);
    }

    #[test]
    fn eval_gather_rejects_a_rule_pointing_past_the_theta_vector() {
        // Defensive: a spec built against a longer θ vector than the caller
        // supplies must not read out of bounds or return a neighbouring value.
        let s = spec(vec![LevelRule::Free(7)]);
        assert!(eval_gather(&s, &[1.0], 1.0).is_nan());
        let s = spec(vec![LevelRule::NegSum(0, 9)]);
        assert!(eval_gather(&s, &[1.0, 2.0], 1.0).is_nan());
        let s = spec(vec![LevelRule::NegSum(3, 1)]);
        assert!(eval_gather(&s, &[1.0, 2.0, 3.0, 4.0], 1.0).is_nan());
    }

    #[test]
    fn eval_gather_sums_the_dependent_level() {
        let s = spec(vec![
            LevelRule::Free(0),
            LevelRule::Free(1),
            LevelRule::NegSum(0, 2),
        ]);
        let theta = [0.25, -0.75];
        assert_eq!(eval_gather(&s, &theta, 3.0), 0.5);
    }

    #[test]
    fn the_pknum_gather_matches_the_f64_one_on_every_branch() {
        // `gather_g` is the copy the analytic-sensitivity path runs. The two
        // must agree bit for bit or a model's gradient and its prediction come
        // from different formulas. `f64` is itself a `PkNum`, so this compares
        // them directly.
        let cases = vec![
            spec(vec![LevelRule::Free(0), LevelRule::Free(1)]),
            spec(vec![LevelRule::Free(0), LevelRule::NegSum(0, 1)]),
            spec(vec![LevelRule::NegSum(0, 0), LevelRule::Free(0)]),
            spec(vec![LevelRule::Free(9)]),
            spec(vec![LevelRule::NegSum(0, 9)]),
            spec(vec![LevelRule::NegSum(3, 1)]),
        ];
        let theta = [0.25, -0.75, 1.5];
        for s in &cases {
            for raw in [f64::NAN, f64::INFINITY, -1.0, 0.0, 1.0, 2.0, 3.0, 1.5] {
                let a = eval_gather(s, &theta, raw);
                let b = gather_g::<f64>(s, &theta, raw);
                assert_eq!(
                    a.is_nan(),
                    b.is_nan(),
                    "NaN disagreement on {s:?} at index {raw}"
                );
                if !a.is_nan() {
                    assert_eq!(a, b, "value disagreement on {s:?} at index {raw}");
                }
            }
        }
    }

    // ── the θ-axis reverse map the differentiator uses ──────────────────────

    #[test]
    fn axes_for_theta_finds_the_free_and_dependent_levels() {
        let s = spec(vec![
            LevelRule::Free(0),
            LevelRule::Free(1),
            LevelRule::NegSum(0, 2),
        ]);
        // θ_0 is read directly by level 1 and through the contrast by level 3.
        assert_eq!(s.axes_for_theta(0), (Some(1), Some(3)));
        assert_eq!(s.axes_for_theta(1), (Some(2), Some(3)));
        // A θ the block does not touch at all.
        assert_eq!(s.axes_for_theta(7), (None, None));
    }

    #[test]
    fn a_reference_level_contributes_no_dependent_axis() {
        // Its `NegSum` range is empty, so no θ is read through it.
        let s = spec(vec![LevelRule::NegSum(0, 0), LevelRule::Free(0)]);
        assert_eq!(s.axes_for_theta(0), (Some(2), None));
    }
}

/// `vi_grad_clip` parses, accepts `0` as "disabled", and rejects a negative threshold.
///
/// `0` has to stay legal: it is how a user asks for the unclipped Adam trajectory, which
/// is the pre-#1097 behaviour and occasionally worth reproducing for comparison. A
/// negative clip has no meaning — `grad_clip_scale` would treat it as disabled — so it is
/// rejected at parse time rather than silently reinterpreted.
#[test]
fn test_apply_fit_option_vi_grad_clip() {
    let mut opts = FitOptions::default();
    assert_eq!(opts.vi_grad_clip, 1e4, "default clip");

    assert_eq!(apply_fit_option(&mut opts, "vi_grad_clip", "250"), Ok(true));
    assert_eq!(opts.vi_grad_clip, 250.0);

    // Explicitly disabled.
    assert_eq!(apply_fit_option(&mut opts, "vi_grad_clip", "0"), Ok(true));
    assert_eq!(opts.vi_grad_clip, 0.0);

    // Rejected, and a failed apply leaves the previous value alone.
    assert!(apply_fit_option(&mut opts, "vi_grad_clip", "-1").is_err());
    assert!(apply_fit_option(&mut opts, "vi_grad_clip", "nope").is_err());
    assert_eq!(opts.vi_grad_clip, 0.0);
}

/// Every *reachable* arm of the [`stmts_read_time_builtin`] walk, driven through
/// the parser.
///
/// A wrong `false` from any arm is a **silently dropped term with no
/// diagnostic** — the exact #1124 failure mode — because `reads_model_time` is
/// what keeps a non-autonomous RHS off the closed form and off the SS
/// equilibration. `mr_scope_declines_every_model_time_spelling` exercises two
/// arms (`Op::PushTime` in a bytecode statement, and `Condition::Compare`); the
/// rest were reachable only by inspection until this test.
///
/// **Where the `Expression::*` arms live.** `resolve_variable_indices` lowers
/// *every* `Assign` / `AssignIdx` / `DiffEq` / `DiffEqIdx` in a resolved RHS to
/// `AssignBc` / `DiffEqBc`, so in an `OdeRhsProgram` those four statement arms
/// are unreachable and a statement's expression is only ever walked as bytecode
/// (the `Op::PushTime` arm). `expr_reads_time_builtin` is therefore reached
/// **only through `if` conditions**, which keep their `Condition`/`Expression`
/// trees. Putting `exp(…TIME…)` in a `d/dt(...)` looks like it covers
/// `Expression::UnaryFn` and does not — it compiles to bytecode. Every
/// expression case below is inside a condition for that reason; a first draft of
/// this test had them in the derivative and the `UnaryFn` mutation survived.
///
/// `Expression::ThetaGather` is not covered: it is produced only by a levelled
/// theta reference (`NAME[idx]`), which has no route into an `[odes]` condition,
/// so there is no parseable model that reaches that arm here. Its recursion into
/// `idx` is written for the shared shape of the walk.
///
/// Each case is asserted in both directions against a `TIME`-free twin of the
/// same shape, so an arm that returns a blanket `true` fails just as loudly as
/// one that returns a blanket `false`.
#[test]
fn every_arm_of_the_time_builtin_walk_sees_time() {
    fn prog_reads_time(rhs: &str) -> bool {
        let src = format!(
            r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVKA(1.0, 0.05, 24.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  KA = TVKA
[structural_model]
  ode(states=[central])
[odes]
{rhs}
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#
        );
        parse_model_string(&src)
            .unwrap_or_else(|e| panic!("model must parse: {e}\n{src}"))
            .ode_spec
            .as_ref()
            .and_then(|o| o.rhs_program.as_ref())
            .expect("rhs program")
            .reads_time_builtin()
    }

    // (arm exercised, RHS that reads TIME, the same RHS with TIME removed)
    let cases: &[(&str, &str, &str)] = &[
        (
            "Expression::Time via a plain assignment",
            "  KE = TIME\n  d/dt(central) = first_order(ka=KA) - (CL/V)*central*(1.0 + 0.01*KE)",
            "  KE = 6.0\n  d/dt(central) = first_order(ka=KA) - (CL/V)*central*(1.0 + 0.01*KE)",
        ),
        (
            "Expression::Time directly in the DiffEq",
            "  d/dt(central) = first_order(ka=KA) - (CL/V)*central*(1.0 + 0.01*TIME)",
            "  d/dt(central) = first_order(ka=KA) - (CL/V)*central*(1.0 + 0.01*6.0)",
        ),
        (
            "Op::PushTime under a call in a bytecode statement",
            "  d/dt(central) = first_order(ka=KA) - (CL/V)*central*exp(-0.01*TIME)",
            "  d/dt(central) = first_order(ka=KA) - (CL/V)*central*exp(-0.01*6.0)",
        ),
        (
            "Expression::UnaryFn (in a condition)",
            "  KE = CL/V\n  if (exp(0.1*TIME) > 2.0) { KE = 3.0*CL/V }\n  d/dt(central) = first_order(ka=KA) - KE*central",
            "  KE = CL/V\n  if (exp(0.1*KA) > 2.0) { KE = 3.0*CL/V }\n  d/dt(central) = first_order(ka=KA) - KE*central",
        ),
        (
            "Expression::Power, base (in a condition)",
            "  KE = CL/V\n  if (TIME^2.0 > 4.0) { KE = 3.0*CL/V }\n  d/dt(central) = first_order(ka=KA) - KE*central",
            "  KE = CL/V\n  if (KA^2.0 > 4.0) { KE = 3.0*CL/V }\n  d/dt(central) = first_order(ka=KA) - KE*central",
        ),
        (
            "Expression::Power, exponent (in a condition)",
            "  KE = CL/V\n  if (2.0^TIME > 4.0) { KE = 3.0*CL/V }\n  d/dt(central) = first_order(ka=KA) - KE*central",
            "  KE = CL/V\n  if (2.0^KA > 4.0) { KE = 3.0*CL/V }\n  d/dt(central) = first_order(ka=KA) - KE*central",
        ),
        (
            "Expression::Binary under a UnaryFn (in a condition)",
            "  KE = CL/V\n  if (log(TIME + 1.0) > 1.0) { KE = 3.0*CL/V }\n  d/dt(central) = first_order(ka=KA) - KE*central",
            "  KE = CL/V\n  if (log(KA + 1.0) > 1.0) { KE = 3.0*CL/V }\n  d/dt(central) = first_order(ka=KA) - KE*central",
        ),
        (
            "Condition::Compare",
            "  KE = CL/V\n  if (TIME > 6.0) { KE = 3.0*CL/V }\n  d/dt(central) = first_order(ka=KA) - KE*central",
            "  KE = CL/V\n  if (KA > 6.0) { KE = 3.0*CL/V }\n  d/dt(central) = first_order(ka=KA) - KE*central",
        ),
        (
            "Condition::And",
            "  KE = CL/V\n  if (TIME > 6.0 && KA > 0.5) { KE = 3.0*CL/V }\n  d/dt(central) = first_order(ka=KA) - KE*central",
            "  KE = CL/V\n  if (CL > 6.0 && KA > 0.5) { KE = 3.0*CL/V }\n  d/dt(central) = first_order(ka=KA) - KE*central",
        ),
        (
            "Condition::Or",
            "  KE = CL/V\n  if (KA > 99.0 || TIME > 6.0) { KE = 3.0*CL/V }\n  d/dt(central) = first_order(ka=KA) - KE*central",
            "  KE = CL/V\n  if (KA > 99.0 || CL > 6.0) { KE = 3.0*CL/V }\n  d/dt(central) = first_order(ka=KA) - KE*central",
        ),
        (
            "Condition::Not",
            "  KE = CL/V\n  if (!(TIME > 6.0)) { KE = 3.0*CL/V }\n  d/dt(central) = first_order(ka=KA) - KE*central",
            "  KE = CL/V\n  if (!(KA > 6.0)) { KE = 3.0*CL/V }\n  d/dt(central) = first_order(ka=KA) - KE*central",
        ),
        (
            "statements inside a taken branch body",
            "  KE = CL/V\n  if (KA > 0.5) { KE = CL/V*(1.0 + 0.01*TIME) }\n  d/dt(central) = first_order(ka=KA) - KE*central",
            "  KE = CL/V\n  if (KA > 0.5) { KE = CL/V*(1.0 + 0.01*6.0) }\n  d/dt(central) = first_order(ka=KA) - KE*central",
        ),
        (
            "statements inside an else body",
            "  KE = CL/V\n  if (KA > 99.0) { KE = 2.0*CL/V } else { KE = CL/V*(1.0 + 0.01*TIME) }\n  d/dt(central) = first_order(ka=KA) - KE*central",
            "  KE = CL/V\n  if (KA > 99.0) { KE = 2.0*CL/V } else { KE = CL/V*(1.0 + 0.01*6.0) }\n  d/dt(central) = first_order(ka=KA) - KE*central",
        ),
    ];

    for (arm, with_time, without_time) in cases {
        assert!(
            prog_reads_time(with_time),
            "`{arm}`: the walk missed a `TIME` read — this RHS would be admitted as \
             time-invariant and its term silently dropped (#1124)"
        );
        assert!(
            !prog_reads_time(without_time),
            "`{arm}`: the walk reported a `TIME` read in an RHS that has none — the \
             positive case above proves nothing if the arm answers `true` blindly"
        );
    }
}

// ── `present(COV)` — the data-presence predicate (#1111) ────────────────────
//
// A missing covariate is `NaN`, and `x/NaN` underflows to `0.0` in this engine
// rather than blowing up, so a model that reads a covariate which may be
// missing needs a guard or it silently zeroes a parameter. `COV == COV` says
// that (NaN compares false against itself) but reads like a typo; `present(COV)`
// is the same test, spelled. It is the guard `[covariate_model]` generates, so
// it appears in text users audit against NONMEM.

/// A one-compartment model whose `CL` is guarded by `cond`.
fn present_model(cond: &str) -> String {
    format!(
        r#"
[parameters]
  theta TVCL(2.0, 0.1, 100.0)
  theta TVV(10.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL * (if ({cond}) (WT / 70) else 1.0)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[covariates]
  WT continuous

[error_model]
  DV ~ proportional(PROP_ERR)
"#
    )
}

/// `CL` at the given `WT`, through the production parameter closure — so this
/// exercises the compiled bytecode path the fit uses, not just the AST walker.
fn cl_at(cond: &str, wt: f64) -> f64 {
    let parsed = parse_full_model(&present_model(cond)).expect("model should parse");
    let mut covariates = HashMap::new();
    covariates.insert("WT".to_string(), wt);
    (parsed.model.pk_param_fn)(&[2.0, 10.0], &[0.0], &covariates, 0.0).values[0]
}

#[test]
fn present_is_true_for_a_value_and_false_for_a_missing_one() {
    // WT = 140 → factor 2; WT missing → the neutral branch, NOT 0.
    assert_eq!(cl_at("present(WT)", 140.0), 4.0);
    assert_eq!(cl_at("present(WT)", f64::NAN), 2.0);
}

#[test]
fn present_means_exactly_what_the_self_comparison_idiom_meant() {
    // `present(COV)` replaced `COV == COV` in generated model text, so the two
    // must agree everywhere — including on infinity, which is a value, not a
    // gap in the data. (`is_finite` would have differed here.)
    for wt in [140.0, 0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            cl_at("present(WT)", wt),
            cl_at("WT == WT", wt),
            "present(WT) and WT == WT disagree at WT = {wt}"
        );
    }
}

/// `TIME` and the slot-backed spellings are covered by **disjoint** flags, and
/// only [`OdeRhsProgram::reads_model_time`] sees both. Pinned structurally so a
/// caller pairing them by hand cannot drift back into #1124's gap.
#[test]
fn reads_model_time_covers_every_spelling_and_the_flags_are_disjoint() {
    fn flags(term: &str) -> (bool, bool, bool) {
        let src = format!(
            r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V)*central{term}
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#
        );
        let m = parse_model_string(&src).expect("parse");
        let p = m
            .ode_spec
            .as_ref()
            .and_then(|o| o.rhs_program.as_ref())
            .expect("rhs program");
        (
            p.uses_time_vars(),
            p.reads_time_builtin(),
            p.reads_model_time(),
        )
    }

    // `TIME` sets ONLY the built-in flag; the slot-backed names set ONLY the
    // broad flag. Neither flag alone covers all four — that disjointness is the
    // whole reason `reads_model_time` exists.
    assert_eq!(flags("*(1.0 + 0.01*TIME)"), (false, true, true), "TIME");
    assert_eq!(flags("*(1.0 + 0.01*T)"), (true, false, true), "T");
    assert_eq!(flags("*(1.0 + 0.01*TAFD)"), (true, false, true), "TAFD");
    assert_eq!(flags("*(1.0 + 0.01*TAD)"), (true, false, true), "TAD");
    assert_eq!(flags(""), (false, false, false), "autonomous control");
}

/// A joint PK-TTE model, with `{PK}` appended to the central derivative, `{PRE}`
/// inserted as an `[odes]` intermediate line and `{HAZ}` as the hazard.
#[cfg(feature = "survival")]
fn joint_src(pk: &str, pre: &str, haz: &str) -> String {
    format!(
        r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  theta TVH0(0.02, 1e-5, 10.0)
  theta TVBETA(0.5, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  KA   = TVKA
  H0   = TVH0
  BETA = TVBETA
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  {pre}
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central{pk}
[event_model]
  cmt    = 3
  hazard = {haz}
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
"
    )
}

/// `(uses_time_vars, reads_time_builtin, reads_model_time, pk_reads_model_time)`.
#[cfg(feature = "survival")]
fn joint_flags(pk: &str, pre: &str, haz: &str) -> (bool, bool, bool, bool) {
    let m = parse_model_string(&joint_src(pk, pre, haz)).expect("joint model must parse");
    let p = m
        .ode_spec
        .as_ref()
        .and_then(|o| o.rhs_program.as_ref())
        .expect("rhs program");
    (
        p.uses_time_vars(),
        p.reads_time_builtin(),
        p.reads_model_time(),
        p.pk_reads_model_time(),
    )
}

/// #1166: the injected `d/dt(__chz_<cmt>)` hazard line does not make the **PK
/// block** non-autonomous, under any spelling of model time.
///
/// The three wide flags keep seeing the whole augmented program — the SS gates
/// need them to, since steady-state equilibration integrates `__chz` too — and
/// only the fourth narrows. Every row is one variable away from its neighbour.
#[test]
#[cfg(feature = "survival")]
fn a_time_dependent_hazard_does_not_make_the_pk_block_non_autonomous() {
    const AUTO: &str = "H0 * exp(BETA * (central / V))";

    // Control: nothing anywhere reads time.
    assert_eq!(
        joint_flags("", "", AUTO),
        (false, false, false, false),
        "autonomous hazard, autonomous PK"
    );

    // The four spellings, in the hazard: wide flags fire, the narrow one does not.
    for (term, wide) in [
        ("TIME", (false, true, true)),
        ("T", (true, false, true)),
        ("TAD", (true, false, true)),
        ("TAFD", (true, false, true)),
    ] {
        let haz = format!("H0 * exp(0.05*{term}) * exp(BETA * (central / V))");
        assert_eq!(
            joint_flags("", "", &haz),
            (wide.0, wide.1, wide.2, false),
            "`{term}` in the hazard must leave the PK block autonomous"
        );
    }

    // The same term in the PK RHS, with an autonomous hazard: the narrow flag
    // must stay wide. This is the row that fails if the exclusion is written as
    // "drop every top-level derivative" rather than "drop the injected ones".
    assert_eq!(
        joint_flags("*(1.0 + 0.01*TIME)", "", AUTO),
        (false, true, true, true),
        "`TIME` in the PK RHS is a genuinely non-autonomous PK block"
    );
    assert_eq!(
        joint_flags("*(1.0 + 0.01*TAD)", "", AUTO),
        (true, false, true, true),
        "`TAD` in the PK RHS likewise"
    );

    // Documented over-decline: an intermediate that reads time and is consumed
    // *only* by the hazard is a top-level `AssignBc`, not one of the excluded
    // derivative lines, so it still reads wide. Conservative in the safe
    // direction — pinned so a later dataflow cut is a deliberate change.
    assert_eq!(
        joint_flags(
            "",
            "TT = TIME",
            "H0 * exp(0.05*TT) * exp(BETA * (central / V))"
        ),
        (false, true, true, true),
        "a time-reading intermediate used only by the hazard over-declines, by design"
    );
}

/// The narrowing is scoped to joint models by construction: with no injected
/// lines the filter is empty and the new flag *is* the old one. Asserted for
/// both a time-reading and an autonomous RHS, and in a build without the
/// `survival` feature this is the only shape that exists.
#[test]
fn without_an_event_model_the_narrow_flag_equals_the_wide_one() {
    fn flags(term: &str) -> (bool, bool) {
        let src = format!(
            r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -1.0 * depot
  d/dt(central) =  1.0 * depot - (CL/V) * central{term}
[error_model]
  DV ~ proportional(PROP_ERR)
"
        );
        let m = parse_model_string(&src).expect("parse");
        let p = m
            .ode_spec
            .as_ref()
            .and_then(|o| o.rhs_program.as_ref())
            .expect("rhs program");
        (p.reads_model_time(), p.pk_reads_model_time())
    }
    for term in ["", "*(1.0 + 0.01*TIME)", "*(1.0 + 0.01*TAD)"] {
        let (wide, narrow) = flags(term);
        assert_eq!(
            wide, narrow,
            "with no [event_model] the two predicates must agree (term `{term}`)"
        );
    }
}

/// The exclusion is keyed on the **slot** the parser injected, never on the name.
///
/// Without an `[event_model]` there is no injection and no reserved-name guard, so
/// `__chz_1` is an ordinary, legal user state — and in a build without the
/// `survival` feature that is the *only* way the name can occur. A name-prefix
/// filter would silently drop this user's own derivative from the predicate and
/// route a genuinely non-autonomous model as if it were time-invariant.
#[test]
fn a_user_state_named_like_the_accumulator_is_not_excluded() {
    let src = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central, __chz_1])
[odes]
  d/dt(central) = -(CL/V) * central
  d/dt(__chz_1) =  0.01 * TIME
[error_model]
  DV ~ proportional(PROP_ERR)
";
    let m = parse_model_string(src).expect("`__chz_1` is a legal user state with no [event_model]");
    let p = m
        .ode_spec
        .as_ref()
        .and_then(|o| o.rhs_program.as_ref())
        .expect("rhs program");
    assert!(
        p.reads_model_time() && p.pk_reads_model_time(),
        "a user's own `__chz_1` derivative reading TIME must still read as non-autonomous"
    );
}

/// The same rule where a name filter and a slot filter actually diverge *with*
/// `survival` on: the parser injects `__chz_3` for the CMT-3 endpoint, and the
/// collision guard only rejects a user state of **that** name — so a user state
/// called `__chz_9` is legal, is not injected, and must be walked.
///
/// The sibling above cannot catch a name-prefix filter, because with no
/// `[event_model]` the slot list is empty and the filter never runs at all.
#[test]
#[cfg(feature = "survival")]
fn a_second_chz_named_user_state_is_walked_because_it_was_not_injected() {
    let src = joint_src("", "", "H0 * exp(BETA * (central / V))")
        .replace(
            "  ode(obs_cmt=central, states=[depot, central])",
            "  ode(obs_cmt=central, states=[depot, central, __chz_9])",
        )
        .replace(
            "  d/dt(depot)   = -KA * depot",
            "  d/dt(depot)   = -KA * depot\n  d/dt(__chz_9) = 0.01 * TIME",
        );
    let m = parse_model_string(&src).expect("`__chz_9` is not the injected accumulator");
    let p = m
        .ode_spec
        .as_ref()
        .and_then(|o| o.rhs_program.as_ref())
        .expect("rhs program");
    assert!(
        p.pk_reads_model_time(),
        "a user state whose name merely looks like the accumulator is still a PK \
         equation — only the injected slot may be excluded"
    );
}

/// A Form C `[scaling] y = …` readout that reads `TIME` must **not** widen the
/// predicate (#1166 §2).
///
/// Both engines evaluate the readout under a `ModelTimeGuard`, so a time-reading
/// readout is already correct on the dense driver; folding
/// `readout_program.reads_time_builtin()` into this predicate — which
/// `pk/modified_release.rs` does hand-pair, for a closed-form-only reason — would
/// decline the shared solve for nothing. Pinned so that folding is a red test
/// rather than a silent cost.
#[test]
#[cfg(feature = "survival")]
fn a_time_reading_form_c_readout_does_not_widen_the_predicate() {
    let src = joint_src("", "", "H0 * exp(BETA * (central / V))")
        .replace(
            "  ode(obs_cmt=central, states=[depot, central])",
            "  ode(states=[depot, central])",
        )
        .replace(
            "[event_model]",
            "[scaling]\n  y = (central / V) * (1.0 + 0.0*TIME)\n\n[event_model]",
        );
    let m = parse_model_string(&src).expect("a TIME-reading Form C readout must parse");
    let spec = m.ode_spec.as_ref().expect("ode spec");
    // The precondition, asserted rather than assumed: the readout really does read
    // time. Without this the two assertions below would also hold for a model where
    // the `[scaling]` substitution silently did nothing.
    assert!(
        spec.readout_program
            .as_ref()
            .expect("Form C readout")
            .reads_time_builtin(),
        "the readout must actually read the TIME built-in"
    );
    let prog = spec.rhs_program.as_ref().expect("rhs program");
    assert!(
        !prog.reads_model_time(),
        "the RHS itself is autonomous — only the readout reads time"
    );
    assert!(
        !crate::pk::model_uses_time_anywhere(&m),
        "a time-reading readout must not route the model off the dense driver"
    );
}

/// #1166: `__chz_*` is write-only. The narrow flag is sound only because the
/// injected line cannot influence the PK dynamics, and that is a property of
/// every *normal* model rather than a structurally enforced one — the collision
/// guard rejects a user state of the same name, but a user *reference* to the
/// accumulator compiled fine. Establish the precondition by rejecting.
#[test]
#[cfg(feature = "survival")]
fn reading_the_hazard_accumulator_is_rejected_everywhere_it_could_feed_back() {
    let cases = [
        // From a PK derivative — the read that would actually couple the systems.
        (
            joint_src(" + 0.0*__chz_3", "", "H0 * exp(BETA * (central / V))"),
            "PK derivative",
        ),
        // Case-insensitively: the aliases resolve to the same slot.
        (
            joint_src(" + 0.0*__CHZ_3", "", "H0 * exp(BETA * (central / V))"),
            "PK derivative, upper case",
        ),
        // From an intermediate.
        (
            joint_src("", "Z = __chz_3", "H0 * exp(BETA * (central / V))"),
            "intermediate",
        ),
        // From inside an `if` condition — `stmts_read_slots` recurses into both
        // the condition and the body, and a gate that only walked bodies would
        // miss this.
        (
            joint_src(
                "",
                "if (__chz_3 > 1.0) { Z = 1.0 } else { Z = 2.0 }",
                "H0 * exp(BETA * (central / V))",
            ),
            "if condition",
        ),
        // From the hazard itself — a self-exciting hazard is the same coupling.
        (
            joint_src("", "", "H0 * exp(BETA * (central / V)) * exp(0.1*__chz_3)"),
            "hazard expression",
        ),
    ];
    for (src, what) in cases {
        let err = parse_model_string(&src)
            .err()
            .unwrap_or_else(|| panic!("reading __chz_3 from the {what} must be rejected"));
        assert!(
            err.contains("__chz") && err.contains("reserved"),
            "the {what} error must name the accumulator and say it is reserved: {err}"
        );
    }
}

/// `init(__chz_n)` is the same hole through a different door: `parse_init_line`
/// accepts any declared state, and the accumulator is a declared state by then,
/// so a user could seed `H(0) ≠ 0` and shift every survival probability.
#[test]
#[cfg(feature = "survival")]
fn seeding_the_hazard_accumulator_with_init_is_rejected() {
    let src = joint_src("", "init(__chz_3) = 0.5", "H0 * exp(BETA * (central / V))");
    let err = parse_model_string(&src)
        .err()
        .expect("init(__chz_3) must be rejected");
    assert!(
        err.contains("init(__chz_3)") && err.contains("H(0) = 0"),
        "error must name the init and the definition it violates: {err}"
    );
}

/// The one read that is **allowed**, pinned as allowed: a `[scaling]` readout
/// cannot feed back into the dynamics, so `y = __chz_3` is legal — and it
/// reads the cumulative hazard, which is what makes it worth a test rather than
/// an accident of where the rejection was placed.
#[test]
#[cfg(feature = "survival")]
fn a_scaling_readout_may_read_the_hazard_accumulator() {
    let src = joint_src("", "", "H0 * exp(BETA * (central / V))")
        .replace(
            "  ode(obs_cmt=central, states=[depot, central])",
            "  ode(states=[depot, central])",
        )
        .replace("[event_model]", "[scaling]\n  y = __chz_3\n\n[event_model]");
    let m = parse_model_string(&src).expect("a readout reading __chz_3 must parse");
    let p = m
        .ode_spec
        .as_ref()
        .and_then(|o| o.rhs_program.as_ref())
        .expect("rhs program");
    assert!(
        !p.pk_reads_model_time(),
        "a readout does not make the PK block non-autonomous"
    );
}

/// As [`cl_at`], but with a *constant* then-branch — so a branch taken while
/// `WT` is missing yields a number rather than propagating the `NaN` the test
/// is not about.
fn cl_at_const(cond: &str, wt: f64) -> f64 {
    let src = present_model(cond).replace("(WT / 70)", "3.0");
    let parsed = parse_full_model(&src).expect("model should parse");
    let mut covariates = HashMap::new();
    covariates.insert("WT".to_string(), wt);
    (parsed.model.pk_param_fn)(&[2.0, 10.0], &[0.0], &covariates, 0.0).values[0]
}

#[test]
fn present_negates_and_combines_like_any_other_condition() {
    // `!present(...)` selects the other branch …
    assert_eq!(cl_at_const("!present(WT)", 140.0), 2.0);
    assert_eq!(cl_at_const("!present(WT)", f64::NAN), 6.0);
    // … and it composes with `&&`, so a guard can carry a real test alongside
    // the presence check — and short-circuits, so the comparison against a
    // missing value never decides the branch on its own.
    assert_eq!(cl_at_const("present(WT) && WT > 100", 140.0), 6.0);
    assert_eq!(cl_at_const("present(WT) && WT > 100", 50.0), 2.0);
    assert_eq!(cl_at_const("present(WT) && WT > 100", f64::NAN), 2.0);
}

#[test]
fn present_accepts_an_expression_not_only_a_bare_name() {
    // The argument is an ordinary expression, so a derived quantity can be
    // tested for missingness too — `WT * 2` is NaN exactly when `WT` is.
    assert_eq!(cl_at("present(WT * 2)", 140.0), 4.0);
    assert_eq!(cl_at("present(WT * 2)", f64::NAN), 2.0);
}

#[test]
fn present_is_only_special_in_condition_position() {
    // A covariate that happens to be named `present` is not shadowed: the
    // predicate is recognised only as `present(` at the head of a condition.
    let src = r#"
[parameters]
  theta TVCL(2.0, 0.1, 100.0)
  theta TVV(10.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL * (if (present > 0) 2.0 else 1.0)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let parsed = parse_full_model(src).expect("a covariate called `present` still parses");
    let mut covariates = HashMap::new();
    covariates.insert("present".to_string(), 1.0);
    let cl = (parsed.model.pk_param_fn)(&[2.0, 10.0], &[0.0], &covariates, 0.0).values[0];
    assert_eq!(cl, 4.0);
}

#[test]
fn an_unclosed_present_is_a_parse_error() {
    let src = present_model("present(WT").replace("else 1.0)", "else 1.0)");
    let e = parse_full_model(&src)
        .map(|_| ())
        .expect_err("unbalanced `present(` must not parse");
    assert!(
        e.to_lowercase().contains("present") || e.contains(')'),
        "{e}"
    );
}

// ── #1182: `power(SIGMA, P)` and the `TAD` magnitude built-in ───────────────

const POWER_MODEL: &str = r#"
[parameters]
  theta TVCL(0.2)
  theta TVV(10.0)
  theta RUV_POW(1.3, 0.01, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ power(PROP_ERR, RUV_POW)
"#;

/// `power(σ, P)` is the proportional model plus an exponent program on its
/// slot: the compiled error model is `Proportional`, the magnitude row carries
/// `[m, p]`, and `∂p/∂θ` is the unit vector on the exponent θ.
#[test]
fn power_form_parses_as_proportional_with_an_exponent_slot() {
    let model = parse_model_string(POWER_MODEL).unwrap();
    assert_eq!(model.error_model, ErrorModel::Proportional);
    assert!(model.has_custom_ruv_magnitude());
    assert!(model.has_ruv_exponent());
    assert!(model.has_theta_dependent_ruv_magnitude());
    let rm = model.ruv_magnitude.as_ref().unwrap();
    assert!(rm.per_sigma[0].is_none(), "the sigma itself is bare");
    assert!(rm.per_sigma_exponent[0].is_some());
    let theta = vec![0.2, 10.0, 1.3];
    let cov = std::collections::HashMap::new();
    let row = rm.eval_obs(&theta, &cov, 1.0, 1.0);
    assert_eq!(row, vec![1.0, 1.3], "multiplier half then exponent half");
    let grad = rm.eval_obs_theta_grad(&theta, &cov, 1.0, 1.0).unwrap();
    assert_eq!(grad.len(), 2);
    assert_eq!(
        grad[0],
        vec![0.0; 3],
        "the bare multiplier has no θ derivative"
    );
    assert_eq!(
        grad[1],
        vec![0.0, 0.0, 1.0],
        "∂p/∂θ is the unit vector on RUV_POW"
    );
    // The exponent θ is referenced from [error_model], so it is not "unused".
    assert!(
        !model.parse_warnings.iter().any(|w| w.contains("RUV_POW")),
        "{:?}",
        model.parse_warnings
    );
    // The row reaches the variance: σ²·|f|^{2p} at f = 2.
    let sigma = &model.default_params.sigma.values;
    let v = model
        .error_spec
        .variance_at_scaled(1, 2.0, sigma, &[], &row);
    approx::assert_relative_eq!(
        v,
        sigma[0] * sigma[0] * 2.0f64.powf(2.6),
        max_relative = 1e-12
    );
}

/// An exponent of exactly `1` is the proportional model, on the legacy
/// arithmetic bit for bit — the search's neutral start must not move the
/// base fit.
#[test]
fn power_form_with_unit_exponent_is_the_proportional_variance_bit_for_bit() {
    let model = parse_model_string(&POWER_MODEL.replace("RUV_POW(1.3,", "RUV_POW(1.0,")).unwrap();
    let rm = model.ruv_magnitude.as_ref().unwrap();
    let theta = vec![0.2, 10.0, 1.0];
    let row = rm.eval_obs(&theta, &std::collections::HashMap::new(), 1.0, 1.0);
    assert_eq!(row, vec![1.0, 1.0]);
    let sigma = &model.default_params.sigma.values;
    for f in [0.0, 0.37, 2.0, 15.5, -3.0] {
        assert_eq!(
            model.error_spec.variance_at_scaled(1, f, sigma, &[], &row),
            model
                .error_spec
                .variance_at_scaled(1, f, sigma, &[], &row[..1]),
            "f = {f}"
        );
        assert_eq!(
            model.error_spec.dvar_df_scaled(1, f, sigma, &row),
            model.error_spec.dvar_df_scaled(1, f, sigma, &row[..1]),
        );
    }
}

/// `TAD` inside a magnitude expression is the record's data-derived time after
/// dose, in both the runtime closure and the `Dual1` θ-program.
#[test]
fn magnitude_expression_reads_tad() {
    let content = r#"
[parameters]
  theta TVCL(0.2)
  theta TVV(10.0)
  theta RUV_TV(1.5, 0.01, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR * (if (TAD < 12.0) RUV_TV else 1.0))
"#;
    let model = parse_model_string(content).unwrap();
    let rm = model.ruv_magnitude.as_ref().unwrap();
    assert!(!rm.has_exponent());
    let theta = vec![0.2, 10.0, 1.5];
    let cov = std::collections::HashMap::new();
    // TIME is 100 on both records; only TAD tells them apart.
    assert_eq!(rm.eval_obs(&theta, &cov, 100.0, 5.0), vec![1.5]);
    assert_eq!(rm.eval_obs(&theta, &cov, 100.0, 20.0), vec![1.0]);
    let early = rm.eval_obs_theta_grad(&theta, &cov, 100.0, 5.0).unwrap();
    let late = rm.eval_obs_theta_grad(&theta, &cov, 100.0, 20.0).unwrap();
    assert_eq!(early[0], vec![0.0, 0.0, 1.0]);
    assert_eq!(late[0], vec![0.0, 0.0, 0.0]);
}

/// `Subject::time_after_dose` is the data TAD: last dose at or before the
/// record, steady-state aware, `NaN` before the first dose.
#[test]
fn subject_time_after_dose_is_the_data_tad() {
    use crate::types::{DoseEvent, Subject};
    let subject = Subject {
        id: "1".into(),
        doses: vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
        ],
        obs_times: vec![-1.0, 2.0, 23.9, 24.0, 30.0],
        ..Default::default()
    };
    let tad: Vec<f64> = (0..5).map(|j| subject.time_after_dose(j)).collect();
    assert!(tad[0].is_nan());
    assert_eq!(tad[1..], [2.0, 23.9, 0.0, 6.0]);
    let ss = Subject {
        id: "2".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)],
        obs_times: vec![26.0],
        ..Default::default()
    };
    assert_eq!(
        ss.time_after_dose(0),
        2.0,
        "SS with II=12: last implied dose at 24"
    );
}

#[test]
fn power_form_rejects_the_wrong_argument_count() {
    let err = expect_parse_err(&POWER_MODEL.replace("power(PROP_ERR, RUV_POW)", "power(PROP_ERR)"));
    assert!(err.contains("power(SIGMA, EXPONENT)"), "got: {err}");
    let err = expect_parse_err(
        &POWER_MODEL.replace("power(PROP_ERR, RUV_POW)", "power(PROP_ERR, RUV_POW, TVV)"),
    );
    assert!(err.contains("3 argument(s)"), "got: {err}");
}

#[test]
fn power_form_rejects_a_sigma_in_the_exponent() {
    let err = expect_parse_err(&POWER_MODEL.replace(
        "power(PROP_ERR, RUV_POW)",
        "power(PROP_ERR, RUV_POW * PROP_ERR)",
    ));
    assert!(err.contains("references sigma `PROP_ERR`"), "got: {err}");
}

#[test]
fn power_form_rejects_an_eta_in_the_exponent() {
    let err = expect_parse_err(&POWER_MODEL.replace(
        "power(PROP_ERR, RUV_POW)",
        "power(PROP_ERR, RUV_POW * exp(ETA_CL))",
    ));
    assert!(
        err.contains("power exponent may not depend on a random effect"),
        "got: {err}"
    );
}

#[test]
fn power_form_rejects_the_forms_it_cannot_combine_with() {
    let err = expect_parse_err(&POWER_MODEL.replace(
        "DV ~ power(PROP_ERR, RUV_POW)",
        "log(DV) ~ power(PROP_ERR, RUV_POW)",
    ));
    assert!(
        err.contains("log-transform-both-sides supports only additive"),
        "got: {err}"
    );
    let err = expect_parse_err(&POWER_MODEL.replace(
        "DV ~ power(PROP_ERR, RUV_POW)",
        "DV ~ power(PROP_ERR, RUV_POW) weight = TVV",
    ));
    assert!(
        err.contains("`weight = …` has no effect on a `power(...)`"),
        "got: {err}"
    );
    let err = expect_parse_err(&POWER_MODEL.replace(
        "DV ~ power(PROP_ERR, RUV_POW)",
        "if (TVV > 1.0) { DV ~ power(PROP_ERR, RUV_POW) } else { DV ~ proportional(PROP_ERR) }",
    ));
    assert!(
        err.contains("not supported inside a covariate-selected"),
        "got: {err}"
    );
}

#[test]
fn power_form_rejects_per_cmt() {
    let content = r#"
[parameters]
  theta TVCL(0.2)
  theta TVV(10.0)
  theta RUV_POW(1.3, 0.01, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[error_model]
  CMT=1: DV ~ power(PROP_ERR, RUV_POW)
"#;
    let err = expect_parse_err(content);
    assert!(err.contains("not supported with per-CMT"), "got: {err}");
}
