use super::*;

fn lines(body: &[&str]) -> Vec<String> {
    body.iter().map(|s| s.to_string()).collect()
}

/// Block-parse with no `[covariates]` declarations, which is what every test here
/// wants: these exercise the block's own grammar and validation, not the `T`/`t`
/// name precedence the declarations feed (#1028 — that has its own tests against
/// the whole model). Shadows the parser's two-argument function so the 37 call
/// sites below stay on the shape they were written in.
fn parse_adaptive_dosing_block(lines: &[String]) -> Result<AdaptiveDosingSpec, String> {
    super::parse_adaptive_dosing_block(lines, &[])
}

/// A minimal block that parses — the base every "one thing wrong" error test
/// mutates. Continuous (percentage) titration, bare-endpoint observe.
fn valid_min() -> Vec<&'static str> {
    vec![
        "observe = central",
        "at = [24, 48]",
        "start_dose = 100",
        "route = bolus(cmt = 1)",
        "dose_bounds = [0, 400]",
        "when signal < 10 : increase 25%",
    ]
}

/// `valid_min` with every line starting with `prefix` removed.
fn without(prefix: &str) -> Vec<String> {
    valid_min()
        .into_iter()
        .filter(|l| !l.starts_with(prefix))
        .map(String::from)
        .collect()
}

fn perr(body: &[&str]) -> String {
    parse_adaptive_dosing_block(&lines(body)).expect_err("expected a typed parse error")
}

// ── Happy paths ──

#[test]
fn full_continuous_block_parses() {
    let spec = parse_adaptive_dosing_block(&lines(&[
        "observe = central / V",
        "at = every 24 from 24 to 96",
        "start_dose = 100",
        "route = bolus(cmt = 1)",
        "dose_bounds = [0, 400]",
        "confirm = 2",
        "when signal < 10 : increase 25%",
        "when signal > 20 : decrease 25%",
        "when signal > 30 : hold",
        "when signal > 40 : stop",
    ]))
    .expect("canonical block parses");
    assert_eq!(spec.observe.as_deref(), Some("central / V"));
    assert!(!spec.with_assay_error);
    assert_eq!(spec.assay_cmt, None);
    assert_eq!(spec.at, vec![24.0, 48.0, 72.0, 96.0]);
    assert_eq!(spec.start_dose, 100.0);
    assert_eq!(spec.route, AdaptiveRoute::Bolus { cmt: 1 });
    assert_eq!(spec.dose_bounds, (0.0, 400.0));
    assert_eq!(spec.confirm, 2);
    assert_eq!(spec.levels, None);
    assert_eq!(spec.rules.len(), 4);
    assert_eq!(
        spec.rules[0],
        AdaptiveRule {
            op: Comparison::Lt,
            threshold: 10.0,
            action: AdaptiveAction::Increase(DoseStep::Percent(25.0)),
            interval: None,
        }
    );
    assert_eq!(
        spec.rules[1].action,
        AdaptiveAction::Decrease(DoseStep::Percent(25.0))
    );
    assert_eq!(spec.rules[1].op, Comparison::Gt);
    assert_eq!(spec.rules[2].action, AdaptiveAction::Hold);
    assert_eq!(spec.rules[3].action, AdaptiveAction::Stop);
}

#[test]
fn minimal_block_parses_with_defaults() {
    let spec = parse_adaptive_dosing_block(&without("nothing")).expect("valid_min parses");
    // Optional keys take their documented defaults.
    assert_eq!(spec.confirm, 1);
    assert!(!spec.with_assay_error);
    assert_eq!(spec.assay_cmt, None);
    assert_eq!(spec.levels, None);
    // No `target_window` declared ⇒ `None` (metrics leave `pct_time_in_window`
    // unreported rather than guessing a band).
    assert_eq!(spec.target_window, None);
}

#[test]
fn target_window_parses_band_and_one_sided() {
    let band = parse_adaptive_dosing_block(&{
        let mut b = without("nothing");
        b.push("target_window = [10, 20]".into());
        b
    })
    .expect("two-sided target_window parses");
    assert_eq!(band.target_window, Some((10.0, 20.0)));

    // A one-sided "at or above 10" target: `high = inf` is allowed.
    let one_sided = parse_adaptive_dosing_block(&{
        let mut b = without("nothing");
        b.push("target_window = [10, inf]".into());
        b
    })
    .expect("one-sided target_window parses");
    let (lo, hi) = one_sided.target_window.expect("Some");
    assert_eq!(lo, 10.0);
    assert!(hi.is_infinite() && hi > 0.0);
}

#[test]
fn target_window_bad_band_errors() {
    // Inverted, non-finite low, and wrong arity each reject loudly, naming the key.
    for bad in [
        "target_window = [20, 10]", // high < low
        "target_window = [nan, 20]",
        "target_window = [inf, 20]", // low must be finite
        "target_window = [10]",      // not [low, high]
    ] {
        let mut b = without("nothing");
        b.push(bad.into());
        let err = parse_adaptive_dosing_block(&b).expect_err("expected a typed parse error");
        assert!(
            err.starts_with("[adaptive_dosing]:") && err.contains("target_window"),
            "`{bad}` gave: {err}"
        );
    }
}

#[test]
fn auc_target_parses_band_and_one_sided() {
    // Default: no `auc_target` ⇒ `None` (the window AUC is skipped and
    // `auc_target_attainment` is unreported).
    let none = parse_adaptive_dosing_block(&without("nothing")).expect("valid_min parses");
    assert_eq!(none.auc_target, None);

    // Two-sided AUC₂₄ band (the vancomycin 400–600 mg·h/L target).
    let band = parse_adaptive_dosing_block(&{
        let mut b = without("nothing");
        b.push("auc_target = [400, 600]".into());
        b
    })
    .expect("two-sided auc_target parses");
    assert_eq!(band.auc_target, Some((400.0, 600.0)));

    // One-sided "at or above 400": `high = inf` is allowed.
    let one_sided = parse_adaptive_dosing_block(&{
        let mut b = without("nothing");
        b.push("auc_target = [400, inf]".into());
        b
    })
    .expect("one-sided auc_target parses");
    let (lo, hi) = one_sided.auc_target.expect("Some");
    assert_eq!(lo, 400.0);
    assert!(hi.is_infinite() && hi > 0.0);
}

#[test]
fn auc_target_bad_band_errors() {
    // Inverted, negative/non-finite low, and wrong arity each reject loudly,
    // naming the key. Unlike `target_window`, a *negative* low is also rejected
    // (an AUC of a non-negative signal cannot be < 0).
    for bad in [
        "auc_target = [600, 400]", // high < low
        "auc_target = [-1, 600]",  // low must be >= 0
        "auc_target = [nan, 600]",
        "auc_target = [inf, 600]", // low must be finite
        "auc_target = [400]",      // not [low, high]
    ] {
        let mut b = without("nothing");
        b.push(bad.into());
        let err = parse_adaptive_dosing_block(&b).expect_err("expected a typed parse error");
        assert!(
            err.starts_with("[adaptive_dosing]:") && err.contains("auc_target"),
            "`{bad}` gave: {err}"
        );
    }
}

#[test]
fn at_explicit_list_is_honored() {
    let spec = parse_adaptive_dosing_block(&{
        let mut b = without("at");
        b.push("at = [24, 48, 72]".into());
        b
    })
    .expect("explicit at list parses");
    assert_eq!(spec.at, vec![24.0, 48.0, 72.0]);
}

#[test]
fn at_every_from_to_expands_inclusive() {
    let spec = parse_adaptive_dosing_block(&{
        let mut b = without("at");
        b.push("at = every 12 from 12 to 48".into());
        b
    })
    .expect("every-from-to parses");
    assert_eq!(spec.at, vec![12.0, 24.0, 36.0, 48.0]);
}

#[test]
fn all_comparison_operators_parse() {
    let spec = parse_adaptive_dosing_block(&lines(&[
        "observe = central",
        "at = [24]",
        "start_dose = 100",
        "route = bolus(cmt = 1)",
        "dose_bounds = [0, 400]",
        "when signal < 1 : hold",
        "when signal <= 2 : hold",
        "when signal > 3 : hold",
        "when signal >= 4 : hold",
        "when signal == 5 : hold",
    ]))
    .expect("all operators parse");
    let ops: Vec<Comparison> = spec.rules.iter().map(|r| r.op).collect();
    assert_eq!(
        ops,
        vec![
            Comparison::Lt,
            Comparison::Le,
            Comparison::Gt,
            Comparison::Ge,
            Comparison::Eq,
        ]
    );
}

#[test]
fn discrete_levels_block_parses() {
    let spec = parse_adaptive_dosing_block(&lines(&[
        "observe = neutrophils",
        "at = [24, 48]",
        "start_dose = 100",
        "route = bolus(cmt = 1)",
        "dose_bounds = [0, 200]",
        "levels = [50, 100, 150, 200]",
        "when signal < 1 : increase",
        "when signal > 3 : decrease",
    ]))
    .expect("discrete levels block parses");
    assert_eq!(spec.levels, Some(vec![50.0, 100.0, 150.0, 200.0]));
    assert_eq!(
        spec.rules[0].action,
        AdaptiveAction::Increase(DoseStep::Level)
    );
    assert_eq!(
        spec.rules[1].action,
        AdaptiveAction::Decrease(DoseStep::Level)
    );
}

#[test]
fn infuse_route_parses() {
    let spec = parse_adaptive_dosing_block(&{
        let mut b = without("route");
        b.push("route = infuse(cmt = 2, over = 1.5)".into());
        b
    })
    .expect("infuse route parses");
    assert_eq!(spec.route, AdaptiveRoute::Infuse { cmt: 2, over: 1.5 });
}

#[test]
fn assay_error_names_output_parses() {
    // With assay error the signal is the noised model output named by
    // `assay_cmt` — there is no `observe` expression to re-type.
    let spec = parse_adaptive_dosing_block(&lines(&[
        "with_assay_error = true",
        "assay_cmt = 2",
        "at = [24]",
        "start_dose = 100",
        "route = bolus(cmt = 1)",
        "dose_bounds = [0, 400]",
        "when signal < 10 : increase 25%",
    ]))
    .expect("assay error naming the output parses");
    assert!(spec.with_assay_error);
    assert_eq!(spec.assay_cmt, Some(2));
    assert_eq!(spec.observe, None);
}

// ── Typed-error matrix (no happy paths) ──

fn assert_adaptive_err(err: &str, needle: &str) {
    assert!(
        err.starts_with("[adaptive_dosing]:") && err.contains(needle),
        "expected an [adaptive_dosing] error containing {needle:?}, got: {err}"
    );
}

#[test]
fn unknown_key_errors() {
    assert_adaptive_err(&perr(&["foo = 1"]), "unknown key");
}

#[test]
fn malformed_line_errors() {
    assert_adaptive_err(&perr(&["this is not a key value pair"]), "malformed line");
}

#[test]
fn missing_required_keys_error() {
    // `observe` is conditionally required (it is the latent signal), so it is
    // covered by the signal-source matrix below, not here.
    for (prefix, needle) in [
        ("at", "missing required `at`"),
        ("start_dose", "missing required `start_dose`"),
        ("route", "missing required `route`"),
        ("dose_bounds", "missing required `dose_bounds`"),
    ] {
        let err = parse_adaptive_dosing_block(&without(prefix)).unwrap_err();
        assert_adaptive_err(&err, needle);
    }
}

#[test]
fn signal_source_matrix_errors() {
    // No signal (drop `observe`, no assay error).
    assert_adaptive_err(
        &parse_adaptive_dosing_block(&without("observe")).unwrap_err(),
        "a signal is required",
    );
    // Both an observe expression and assay error.
    let mut both = without("nothing"); // full valid block (observe present)…
    both.push("with_assay_error = true".into());
    both.push("assay_cmt = 1".into());
    assert_adaptive_err(
        &parse_adaptive_dosing_block(&both).unwrap_err(),
        "remove `observe`",
    );
    // Assay error without naming the measured output.
    let mut no_cmt = without("observe");
    no_cmt.push("with_assay_error = true".into());
    assert_adaptive_err(
        &parse_adaptive_dosing_block(&no_cmt).unwrap_err(),
        "requires `assay_cmt",
    );
}

#[test]
fn no_rules_errors() {
    assert_adaptive_err(
        &parse_adaptive_dosing_block(&without("when")).unwrap_err(),
        "rule",
    );
}

#[test]
fn dose_bounds_problems_error() {
    let mk = |bounds: &str| {
        let mut b = without("dose_bounds");
        b.push(format!("dose_bounds = {bounds}"));
        parse_adaptive_dosing_block(&b).unwrap_err()
    };
    assert_adaptive_err(&mk("[400, 0]"), "0 <= low <= high"); // inverted
    assert_adaptive_err(&mk("[-1, 400]"), "0 <= low <= high"); // negative low
    assert_adaptive_err(&mk("[0, 100, 200]"), "[low, high]"); // wrong length
}

#[test]
fn start_dose_outside_bounds_errors() {
    let mut b = without("start_dose");
    b.push("start_dose = 500".into()); // bounds are [0, 400]
    assert_adaptive_err(
        &parse_adaptive_dosing_block(&b).unwrap_err(),
        "outside dose_bounds",
    );
}

#[test]
fn levels_outside_dose_bounds_error() {
    // start_dose is in both `levels` and `dose_bounds`, but a rung exceeds the
    // high bound — `step_dose` would clamp the dose while the rung index keeps
    // advancing, decoupling the decision log from the realized dose.
    let err = parse_adaptive_dosing_block(&lines(&[
        "observe = central",
        "at = [24]",
        "start_dose = 100",
        "route = bolus(cmt = 1)",
        "dose_bounds = [0, 120]",
        "levels = [100, 150, 200]",
        "when signal < 10 : increase",
    ]))
    .unwrap_err();
    assert_adaptive_err(&err, "outside dose_bounds");
}

#[test]
fn schedule_negative_times_error() {
    // `start_dose` / `dose_bounds` reject < 0; `at` must too, on both forms.
    let mk = |at: &str| {
        let mut b = without("at");
        b.push(format!("at = {at}"));
        parse_adaptive_dosing_block(&b).unwrap_err()
    };
    assert_adaptive_err(&mk("[-24, 0, 24]"), ">= 0");
    assert_adaptive_err(&mk("every 24 from -24 to 24"), "0 <= start <= end");
}

#[test]
fn schedule_unclosed_bracket_errors() {
    // `[24, 48` (no closing `]`) was silently accepted as [24, 48].
    let mut b = without("at");
    b.push("at = [24, 48".into());
    assert_adaptive_err(&parse_adaptive_dosing_block(&b).unwrap_err(), "closing `]`");
}

#[test]
fn schedule_every_keeps_inclusive_endpoint_for_decimal_step() {
    // 0.3 / 0.1 = 2.999…6 in f64; a bare floor() drops the inclusive endpoint.
    let mk = |at: &str| {
        let mut b = without("at");
        b.push(format!("at = {at}"));
        parse_adaptive_dosing_block(&b).expect("schedule parses").at
    };
    let decimal = mk("every 0.1 from 0 to 0.3");
    assert_eq!(
        decimal.len(),
        4,
        "expected the inclusive 0.3, got {decimal:?}"
    );
    assert!((decimal.last().unwrap() - 0.3).abs() < 1e-12);
    // A non-multiple endpoint stays excluded (last point < t1, no spurious add).
    assert_eq!(
        mk("every 24 from 0 to 100"),
        vec![0.0, 24.0, 48.0, 72.0, 96.0]
    );
}

#[test]
fn confirm_zero_errors() {
    let mut b = without("nothing");
    b.push("confirm = 0".into());
    assert_adaptive_err(
        &parse_adaptive_dosing_block(&b).unwrap_err(),
        "confirm must be >= 1",
    );
}

#[test]
fn assay_cmt_without_assay_error_errors() {
    let mut b = without("nothing");
    b.push("assay_cmt = 2".into());
    assert_adaptive_err(
        &parse_adaptive_dosing_block(&b).unwrap_err(),
        "with_assay_error = false",
    );
}

#[test]
fn mixed_percent_and_levels_errors() {
    let err = perr(&[
        "observe = central",
        "at = [24]",
        "start_dose = 100",
        "route = bolus(cmt = 1)",
        "dose_bounds = [0, 200]",
        "levels = [50, 100, 150]",
        "when signal < 10 : increase 25%", // percentage with levels declared
    ]);
    assert_adaptive_err(&err, "not");
}

#[test]
fn bare_level_step_without_levels_errors() {
    let err = perr(&[
        "observe = central",
        "at = [24]",
        "start_dose = 100",
        "route = bolus(cmt = 1)",
        "dose_bounds = [0, 400]",
        "when signal < 10 : increase", // level step, no levels ladder
    ]);
    assert_adaptive_err(&err, "requires a `levels");
}

#[test]
fn start_dose_not_a_level_errors() {
    let err = perr(&[
        "observe = central",
        "at = [24]",
        "start_dose = 120",
        "route = bolus(cmt = 1)",
        "dose_bounds = [0, 200]",
        "levels = [50, 100, 150, 200]",
        "when signal < 10 : increase",
    ]);
    assert_adaptive_err(&err, "one of the declared levels");
}

#[test]
fn levels_not_strictly_increasing_errors() {
    let mut b = without("nothing");
    b.push("levels = [50, 50, 100]".into());
    b.push("when signal < 10 : increase".into()); // make it a level block
                                                  // valid_min already has a percentage rule; drop it so only level steps remain.
    let b: Vec<String> = b.into_iter().filter(|l| !l.contains("25%")).collect();
    assert_adaptive_err(
        &parse_adaptive_dosing_block(&b).unwrap_err(),
        "strictly increasing",
    );
}

#[test]
fn rule_condition_must_compare_signal() {
    let mut b = without("when");
    b.push("when conc < 10 : hold".into());
    assert_adaptive_err(
        &parse_adaptive_dosing_block(&b).unwrap_err(),
        "compare `signal`",
    );
}

#[test]
fn rule_missing_operator_errors() {
    let mut b = without("when");
    b.push("when signal 10 : hold".into());
    assert_adaptive_err(
        &parse_adaptive_dosing_block(&b).unwrap_err(),
        "comparison operator",
    );
}

#[test]
fn rule_missing_action_colon_errors() {
    let mut b = without("when");
    b.push("when signal < 10".into());
    assert_adaptive_err(
        &parse_adaptive_dosing_block(&b).unwrap_err(),
        "when <signal>",
    );
}

#[test]
fn unknown_action_errors() {
    let mut b = without("when");
    b.push("when signal < 10 : obliterate".into());
    assert_adaptive_err(
        &parse_adaptive_dosing_block(&b).unwrap_err(),
        "unknown action",
    );
}

#[test]
fn percentage_missing_percent_sign_errors() {
    let mut b = without("when");
    b.push("when signal < 10 : increase 25".into()); // no % and not bare → not a level
                                                     // `increase 25` has an operand `25` that is not a percentage.
    assert_adaptive_err(&parse_adaptive_dosing_block(&b).unwrap_err(), "percentage");
}

#[test]
fn route_problems_error() {
    let mk = |route: &str| {
        let mut b = without("route");
        b.push(format!("route = {route}"));
        parse_adaptive_dosing_block(&b).unwrap_err()
    };
    assert_adaptive_err(&mk("squirt(cmt = 1)"), "unknown route");
    assert_adaptive_err(&mk("bolus(cmt = 1, over = 2)"), "bolus route does not take");
    assert_adaptive_err(&mk("infuse(cmt = 1)"), "infuse route requires over");
    assert_adaptive_err(&mk("bolus()"), "requires cmt");
    assert_adaptive_err(&mk("bolus(cmt = 0)"), "1-based");
    assert_adaptive_err(&mk("bolus(cmt = 1, dose = 5)"), "unknown route argument");
    assert_adaptive_err(&mk("bolus cmt 1"), "bolus(cmt=N)");
}

#[test]
fn at_problems_error() {
    let mk = |at: &str| {
        let mut b = without("at");
        b.push(format!("at = {at}"));
        parse_adaptive_dosing_block(&b).unwrap_err()
    };
    assert_adaptive_err(&mk("every 24 from 24"), "every <"); // missing `to`
    assert_adaptive_err(&mk("[48, 24]"), "strictly increasing"); // out of order
    assert_adaptive_err(
        &mk("every 0 from 24 to 96"),
        "interval must be finite and > 0",
    );
}

// ── End-to-end wiring: the block reaches ParsedModel ──

const MODEL_HEAD: &str = r#"
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
  DV ~ proportional(PROP_ERR)
"#;

#[test]
fn parsed_model_carries_adaptive_dosing_block() {
    let with_block = format!(
            "{MODEL_HEAD}\n[adaptive_dosing]\n  observe = central / V\n  at = every 24 from 24 to 96\n  start_dose = 100\n  route = bolus(cmt = 1)\n  dose_bounds = [0, 400]\n  when signal < 10 : increase 25%\n"
        );
    let parsed = parse_full_model(&with_block).expect("model with [adaptive_dosing] parses");
    let spec = parsed
        .adaptive_dosing
        .expect("the block must attach to ParsedModel");
    assert_eq!(spec.start_dose, 100.0);
    assert_eq!(spec.at.len(), 4);
}

#[test]
fn parsed_model_without_block_is_none() {
    let parsed = parse_full_model(MODEL_HEAD).expect("plain model parses");
    assert!(parsed.adaptive_dosing.is_none());
}

// ── #945: the dosing-interval ladder ──

/// `valid_min` on a q12 grid, with `extra` lines appended and its rule replaced by
/// `rules`.
fn interval_block(extra: &[&str], rules: &[&str]) -> Vec<String> {
    let mut b: Vec<String> = without("when")
        .into_iter()
        .filter(|l| !l.starts_with("at "))
        .collect();
    b.push("at = every 12 from 0 to 96".into());
    b.extend(extra.iter().map(|s| s.to_string()));
    b.extend(rules.iter().map(|s| s.to_string()));
    b
}

const LADDER: [&str; 2] = ["intervals = [12, 24, 36]", "start_interval = 24"];

#[test]
fn interval_ladder_block_parses() {
    let spec = parse_adaptive_dosing_block(&interval_block(
        &LADDER,
        &[
            "when signal > 40 : hold, lengthen",
            "when signal > 20 : decrease 25%, lengthen",
            "when signal > 15 : lengthen",
            "when signal < 5 : shorten, increase 10%",
        ],
    ))
    .expect("interval ladder parses");
    assert_eq!(spec.intervals, Some(vec![12.0, 24.0, 36.0]));
    assert_eq!(spec.start_interval, Some(24.0));
    let effects: Vec<_> = spec
        .rules
        .iter()
        .map(|r| (r.action.clone(), r.interval))
        .collect();
    assert_eq!(
        effects,
        vec![
            (AdaptiveAction::Hold, Some(IntervalStep::Lengthen)),
            (
                AdaptiveAction::Decrease(DoseStep::Percent(25.0)),
                Some(IntervalStep::Lengthen)
            ),
            // An interval-only rule re-issues the running dose.
            (AdaptiveAction::Maintain, Some(IntervalStep::Lengthen)),
            // Either order.
            (
                AdaptiveAction::Increase(DoseStep::Percent(10.0)),
                Some(IntervalStep::Shorten)
            ),
        ]
    );
    // A block without the keys is unchanged.
    let plain = parse_adaptive_dosing_block(&without("nothing")).expect("valid_min parses");
    assert_eq!((plain.intervals, plain.start_interval), (None, None));
    assert_eq!(plain.rules[0].interval, None);
}

/// One rule, one dose effect and one interval effect: each excess is refused with
/// the sentence naming what a rule may carry. `stop` with an interval effect is
/// refused by the shared validator.
#[test]
fn interval_rule_effect_errors() {
    let e = perr_owned(interval_block(
        &LADDER,
        &["when signal > 20 : lengthen, shorten"],
    ));
    assert_adaptive_err(
        &e,
        "a rule takes at most one interval effect (`lengthen` or `shorten`): lengthen, shorten",
    );
    let e = perr_owned(interval_block(
        &LADDER,
        &["when signal > 20 : hold, decrease 10%"],
    ));
    assert_adaptive_err(
        &e,
        "a rule takes at most one dose effect (`increase`, `decrease`, `hold` or `stop`): \
         hold, decrease 10%",
    );
    let e = perr_owned(interval_block(
        &LADDER,
        &["when signal > 20 : stop, lengthen"],
    ));
    assert_adaptive_err(
        &e,
        "`stop` cannot be combined with `lengthen` / `shorten` (a stopped course has no next \
         dose)",
    );
    let e = perr_owned(interval_block(&LADDER, &["when signal > 20 : lengthn"]));
    assert_adaptive_err(&e, "unknown action `lengthn`");
    assert!(e.contains("lengthen/shorten"), "{e}");
}

/// A stray comma names itself, on either side of an effect; a rule with nothing after the
/// `:` keeps its own message. An operand on an interval effect is refused as such, not
/// read as a malformed dose percentage.
#[test]
fn interval_rule_malformed_effect_errors() {
    for rule in ["when signal > 20 : hold,", "when signal > 20 : , lengthen"] {
        let e = perr_owned(interval_block(&LADDER, &[rule]));
        assert_adaptive_err(&e, "a rule has an empty effect (a stray comma?)");
        assert!(!e.contains("no action after"), "{e}");
    }
    let e = perr_owned(interval_block(&LADDER, &["when signal > 20 :"]));
    assert_adaptive_err(&e, "rule has no action after `:`");
    let e = perr_owned(interval_block(&LADDER, &["when signal > 20 : lengthen 2"]));
    assert_adaptive_err(
        &e,
        "`lengthen` steps the `intervals` ladder one rung and takes no operand: lengthen 2",
    );
    assert!(!e.contains("percentage"), "{e}");
    let e = perr_owned(interval_block(
        &LADDER,
        &["when signal > 20 : hold, shorten 12"],
    ));
    assert_adaptive_err(&e, "`shorten` steps the `intervals` ladder");
}

#[test]
fn interval_keys_must_come_together() {
    let rule = ["when signal > 20 : lengthen"];
    let e = perr_owned(interval_block(&["intervals = [12, 24]"], &rule));
    assert_adaptive_err(&e, "`intervals` requires a `start_interval`");
    let e = perr_owned(interval_block(
        &["start_interval = 12"],
        &["when signal > 20 : hold"],
    ));
    assert_adaptive_err(
        &e,
        "`start_interval` requires an `intervals = [...]` ladder",
    );
    let e = perr_owned(interval_block(&[], &rule));
    assert_adaptive_err(
        &e,
        "`lengthen` / `shorten` require an `intervals = [...]` ladder and a `start_interval`",
    );
    let e = perr_owned(interval_block(
        &["intervals = [12, q24]", "start_interval = 12"],
        &rule,
    ));
    assert_adaptive_err(&e, "bad intervals:");
    let e = perr_owned(interval_block(
        &["intervals = [24, 12]", "start_interval = 12"],
        &rule,
    ));
    assert_adaptive_err(&e, "intervals must be strictly increasing");
    let e = perr_owned(interval_block(
        &["intervals = [12, 24]", "start_interval = q12"],
        &rule,
    ));
    assert_adaptive_err(&e, "bad start_interval: q12");
}

/// On an `every S` grid each interval must be a whole multiple of `S`; the error
/// names the value and the step. A decimal step needs the tolerance (0.3 is not
/// 3 × 0.1 in floating point), and an explicit `at` list leaves the check to the
/// driver.
#[test]
fn intervals_must_be_multiples_of_the_every_step() {
    let rule = ["when signal > 20 : lengthen"];
    let e = perr_owned(interval_block(
        &["intervals = [12, 18]", "start_interval = 12"],
        &rule,
    ));
    assert_adaptive_err(
        &e,
        "interval 18 is not a whole multiple of the `at` step 12, so its due times would \
         fall between decision times",
    );
    // Half the step rounds to one step and fails on the residual.
    let e = perr_owned(interval_block(
        &["intervals = [6, 12]", "start_interval = 12"],
        &rule,
    ));
    assert_adaptive_err(&e, "interval 6 is not a whole multiple of the `at` step 12");
    // Far below the step the residual is inside the tolerance and only `k < 1` refuses it
    // (the driver would refuse it too, at the first decision).
    let e = perr_owned(interval_block(
        &["intervals = [1e-10, 12]", "start_interval = 12"],
        &rule,
    ));
    assert_adaptive_err(
        &e,
        "interval 0.0000000001 is not a whole multiple of the `at` step 12",
    );

    let mut decimal = interval_block(&["intervals = [0.3, 0.6]", "start_interval = 0.3"], &rule);
    decimal.retain(|l| !l.starts_with("at "));
    decimal.push("at = every 0.1 from 0 to 1.2".into());
    assert!(0.3 != 3.0 * 0.1, "the fixture must need the tolerance");
    parse_adaptive_dosing_block(&decimal).expect("0.3 is a multiple of 0.1 within tolerance");

    let mut list = interval_block(&["intervals = [12, 18]", "start_interval = 12"], &rule);
    list.retain(|l| !l.starts_with("at "));
    list.push("at = [0, 12, 24, 30]".into());
    let spec = parse_adaptive_dosing_block(&list).expect("an explicit list defers to the driver");
    assert_eq!(spec.intervals, Some(vec![12.0, 18.0]));
}

fn perr_owned(body: Vec<String>) -> String {
    parse_adaptive_dosing_block(&body).expect_err("expected a typed parse error")
}

/// The rule words reach a whole-model parse: nothing in the model-level checks reads
/// `lengthen` / `shorten` as an identifier.
#[test]
fn parsed_model_carries_an_interval_ladder() {
    let with_block = format!(
        "{MODEL_HEAD}\n[adaptive_dosing]\n  observe = central / V\n  at = every 12 from 0 to 96\n  \
         start_dose = 100\n  route = bolus(cmt = 1)\n  dose_bounds = [0, 400]\n  \
         intervals = [12, 24]\n  start_interval = 12\n  \
         when signal > 20 : decrease 25%, lengthen\n  when signal < 5 : shorten\n"
    );
    let spec = parse_full_model(&with_block)
        .expect("model with an interval ladder parses")
        .adaptive_dosing
        .expect("block attached");
    assert_eq!(spec.start_interval, Some(12.0));
    assert_eq!(spec.rules[1].interval, Some(IntervalStep::Shorten));
}
