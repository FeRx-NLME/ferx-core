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
    assert_eq!(spec.dose_bounds, (0.0.into(), 400.0.into()));
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
    assert_eq!(
        spec.levels,
        crate::sim::adaptive::literal_levels(&[50.0, 100.0, 150.0, 200.0])
    );
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

// ── Dose expressions (#944) ──

/// Each accepted form: a number stays a literal; an expression, a mixed pair, a `min`
/// whose own comma sits inside parentheses, and an expression rung are kept as source.
///
/// Regressions: `[0, min(30 * WT, 4000)]` split on every comma (three items, refused as
/// not a pair); a number kept as an expression (the literal path, and its parse-time
/// checks, skipped).
#[test]
fn dose_values_parse_as_numbers_or_expressions() {
    let spec = parse_adaptive_dosing_block(&lines(&[
        "observe = central",
        "at = [24, 48]",
        "start_dose = 15 * WT",
        "route = bolus(cmt = 1)",
        "dose_bounds = [0, min(30 * WT, 4000)]",
        "levels = [500, 10 * WT, 4000]",
        "when signal < 10 : increase",
    ]))
    .expect("expressions parse");
    assert_eq!(spec.start_dose, DoseValue::Expr("15 * WT".to_string()));
    assert_eq!(
        spec.dose_bounds,
        (
            DoseValue::Literal(0.0),
            DoseValue::Expr("min(30 * WT, 4000)".to_string())
        )
    );
    assert_eq!(
        spec.levels,
        Some(vec![
            DoseValue::Literal(500.0),
            DoseValue::Expr("10 * WT".to_string()),
            DoseValue::Literal(4000.0),
        ])
    );
    let lit = parse_adaptive_dosing_block(&lines(&valid_min())).expect("literal block");
    assert_eq!(lit.start_dose, DoseValue::Literal(100.0));
    assert_eq!(
        lit.dose_bounds,
        (DoseValue::Literal(0.0), DoseValue::Literal(400.0))
    );
}

/// The shape errors a dose key can still have: no value, a list that is not a pair, a
/// list without brackets.
///
/// Regressions: an empty `start_dose` kept as an empty expression; the pair count taken
/// before the top-level split; a bracket-less list accepted.
#[test]
fn dose_value_shape_errors() {
    let mut b = without("start_dose");
    b.push("start_dose =".to_string());
    let refs: Vec<&str> = b.iter().map(String::as_str).collect();
    assert_eq!(
        perr(&refs),
        "[adaptive_dosing]: start_dose requires a number or an expression"
    );

    let mut b = without("dose_bounds");
    b.push("dose_bounds = [0, 30 * WT, 4000]".to_string());
    let refs: Vec<&str> = b.iter().map(String::as_str).collect();
    assert_eq!(
        perr(&refs),
        "[adaptive_dosing]: dose_bounds must be `[low, high]`: [0, 30 * WT, 4000]"
    );

    let mut b = without("dose_bounds");
    b.push("dose_bounds = 0, 30 * WT".to_string());
    let refs: Vec<&str> = b.iter().map(String::as_str).collect();
    assert_eq!(
        perr(&refs),
        "[adaptive_dosing]: dose_bounds must be a list `[...]`: 0, 30 * WT"
    );
}

/// A 1-cpt model with a WT covariate, an η on CL, and — when `kappa` — a κ on CL.
fn dose_model(kappa: bool, start_dose: &str) -> String {
    let (k_decl, k_use) = if kappa {
        ("  kappa KAPPA_CL ~ 0.09\n", " + KAPPA_CL")
    } else {
        ("", "")
    };
    format!(
        "[parameters]
  theta TVCL(6.0, 0.1, 50.0)
  theta TVV(0.7, 0.05, 5.0)
  omega ETA_CL ~ 0.09
{k_decl}  sigma ADD ~ 1.0
[individual_parameters]
  CL = TVCL * (WT / 70) * exp(ETA_CL{k_use})
  V  = TVV * WT
[structural_model]
  ode(states=[CENT])
[odes]
  d/dt(CENT) = -(CL / V) * CENT
[scaling]
  y = CENT / V
[error_model]
  DV ~ additive(ADD)
[adaptive_dosing]
  observe = CENT / V
  at = every 12 from 12 to 48
  start_dose = {start_dose}
  route = infuse(cmt=1, over=1)
  dose_bounds = [0, 4000]
  when signal < 15 : increase 25%
"
    )
}

fn full_err(src: &str) -> String {
    match crate::parser::model_parser::parse_full_model(src) {
        Ok(_) => panic!("expected a parse error"),
        Err(e) => e.to_string(),
    }
}

/// Every read a dose expression may not make is refused when the model is parsed, by
/// name, with the whole message asserted (so deleting any sentence fails here). A
/// θ-and-covariate expression on the same model parses, so the refusals are not a
/// blanket rejection of expressions.
///
/// Regressions, per row: the state check dropped (`2 * CENT` reads a zero state); the
/// `TIME` check dropped (it evaluates to 0.0 outside the driver), or the `T` alias
/// missed; κ reported with the shared compiler's `Form C … per observation` text; an
/// individual parameter or η read let through to a zero PK / η slice; `MIXNUM` let
/// through; a syntax error reported without the key.
#[test]
fn dose_expression_reads_are_refused_at_parse_time() {
    let cases: [(bool, &str, &str); 7] = [
        (
            false,
            "2 * CENT",
            "[adaptive_dosing] start_dose reads CENT, a state of the model. A dose \
             expression is evaluated once per subject, at the first decision; it may read \
             data covariates and θ. To respond to the drug level, write a `when signal …` \
             rule.",
        ),
        (
            false,
            "100 * TIME",
            "[adaptive_dosing] start_dose reads TIME. A dose expression is evaluated once, \
             at the first decision (t = 12), so TIME is always 12; write the value instead.",
        ),
        (
            false,
            "100 * T",
            "[adaptive_dosing] start_dose reads TIME. A dose expression is evaluated once, \
             at the first decision (t = 12), so TIME is always 12; write the value instead.",
        ),
        (
            true,
            "100 * exp(KAPPA_CL)",
            "[adaptive_dosing] start_dose reads KAPPA_CL, an inter-occasion random effect. \
             It is drawn for each occasion, so it is never known at dosing time, even for a \
             subject whose random effects are supplied.",
        ),
        (
            false,
            "400 * CL",
            "[adaptive_dosing] start_dose reads CL, an individual parameter. A dose \
             expression may read data covariates and θ; it cannot read a subject's own \
             parameter values yet. To dose from a known value, carry it as a data column \
             (e.g. CL_KNOWN) and read that.",
        ),
        (
            false,
            "1000 * exp(ETA_CL)",
            "[adaptive_dosing] start_dose reads ETA_CL, a random effect. A dose expression \
             may read data covariates and θ; it cannot read a subject's own parameter values \
             yet. To dose from a known value, carry it as a data column (e.g. CL_KNOWN) and \
             read that.",
        ),
        (false, "15 * (WT", "[adaptive_dosing] start_dose: "),
    ];
    for (kappa, expr, want) in cases {
        let got = full_err(&dose_model(kappa, expr));
        if expr == "15 * (WT" {
            assert!(got.starts_with(want), "{expr}: {got}");
        } else {
            assert!(got.ends_with(want), "{expr}:\n got {got}\nwant {want}");
            assert!(
                !got.contains("Form C") && !got.contains("per observation"),
                "{got}"
            );
        }
    }
    assert!(
        crate::parser::model_parser::parse_full_model(&dose_model(true, "15 * WT * TVCL")).is_ok()
    );
}

/// `MIXNUM` is drawn with the subject's class, so a dose expression cannot read it.
///
/// Regression: `MIXNUM` let through, evaluating to the class-1 default for every subject.
#[test]
fn a_dose_expression_reading_mixnum_is_refused() {
    let src = "[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  theta MIXL(0.0, -5.0, 5.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[mixture]
  nsub = 2
  logit(1) = MIXL
[individual_parameters]
  CL = if (MIXNUM == 1) TVCL * exp(ETA_CL) else 4 * TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP)
[adaptive_dosing]
  observe = central / V
  at = [0, 24]
  start_dose = 100 * MIXNUM
  route = bolus(cmt=1)
  dose_bounds = [0, 400]
  when signal < 1 : increase 25%
";
    assert!(
        full_err(src).ends_with(
            "[adaptive_dosing] start_dose reads MIXNUM, the subject's mixture class, which \
             the simulation draws; no prescriber knows it at dosing time."
        ),
        "{}",
        full_err(src)
    );
}

/// The key named in a refusal is the key the expression is under.
///
/// Regression: every refusal hard-coding `start_dose` (the shared arm's discriminator).
#[test]
fn a_dose_expression_refusal_names_its_own_key() {
    let src =
        dose_model(false, "500").replace("dose_bounds = [0, 4000]", "dose_bounds = [0, 2 * CENT]");
    let got = full_err(&src);
    assert!(
        got.contains("[adaptive_dosing] dose_bounds reads CENT"),
        "{got}"
    );
    assert!(!got.contains("start_dose"), "{got}");
    let src = dose_model(false, "500")
        .replace(
            "dose_bounds = [0, 4000]",
            "dose_bounds = [0, 4000]\n  levels = [500, 400 * CL]",
        )
        .replace("increase 25%", "increase");
    let got = full_err(&src);
    assert!(got.contains("[adaptive_dosing] levels reads CL"), "{got}");
    assert!(!got.contains("start_dose"), "{got}");
}

/// A block that mixes numbers and expressions is checked among its numbers when it is
/// parsed (#2667 review r1 row 1): every invariant whose values are all numbers is a
/// parse error, so a typo in the block is not reported as subject 1's dose value. Each
/// row has both sides of its gate: the refused block, and the same pair with one side
/// an expression, which parses (it is checked per subject).
///
/// Regressions: the dose checks gated on an all-numbers block again (every refused row
/// parses); a check that runs with an unknown operand (a parsing row refused); the
/// ladder order checked only between neighbours (`[500, 10 * WT, 400]` parses);
/// start-on-the-ladder checked with an expression rung present (`12 * WT` refused).
#[test]
fn a_mixed_block_is_checked_among_its_numbers_at_parse_time() {
    let block = |start: &str, bounds: &str, levels: Option<&str>| {
        let mut b = vec![
            "observe = central".to_string(),
            "at = [24, 48]".to_string(),
            format!("start_dose = {start}"),
            "route = bolus(cmt = 1)".to_string(),
            format!("dose_bounds = {bounds}"),
        ];
        match levels {
            Some(l) => {
                b.push(format!("levels = {l}"));
                b.push("when signal < 10 : increase".to_string());
            }
            None => b.push("when signal < 10 : increase 25%".to_string()),
        }
        b
    };
    let refused: [(&str, &str, Option<&str>, &str); 10] = [
        // A start no expression rung can reach (#2667 review r2 N3): the only expression
        // rung is above 200, or between 100 and 200 while the start is 250 (only its
        // upper neighbour rules it out) or 50 (only its lower neighbour does).
        (
            "50",
            "[0, 30 * WT]",
            Some("[100, 10 * WT, 200]"),
            "[adaptive_dosing]: start_dose 50 must be one of the declared levels [100, 10 * \
             WT, 200]. It is none of the numbers, and the ladder increases, so no expression \
             rung can equal it: none lies between numbers on either side of 50.",
        ),
        (
            "150",
            "[0, 30 * WT]",
            Some("[100, 200, 20 * WT]"),
            "[adaptive_dosing]: start_dose 150 must be one of the declared levels [100, 200, \
             20 * WT]. It is none of the numbers, and the ladder increases, so no expression \
             rung can equal it: none lies between numbers on either side of 150.",
        ),
        (
            "250",
            "[0, 30 * WT]",
            Some("[100, 10 * WT, 200]"),
            "[adaptive_dosing]: start_dose 250 must be one of the declared levels [100, 10 * \
             WT, 200]. It is none of the numbers, and the ladder increases, so no expression \
             rung can equal it: none lies between numbers on either side of 250.",
        ),
        (
            "500",
            "[-1, 30 * WT]",
            None,
            "[adaptive_dosing]: dose_bounds must be finite with 0 <= low <= high: [-1, 30 * WT]",
        ),
        // A negative numeric high beside an expression low: invalid on its own, since
        // `0 <= low <= high` (#2667 review r2 N1). Without that check it would be reported
        // as the start being outside the bounds.
        (
            "500",
            "[5 * WT, -1]",
            None,
            "[adaptive_dosing]: dose_bounds must be finite with 0 <= low <= high: [5 * WT, -1]",
        ),
        (
            "5",
            "[10, 30 * WT]",
            None,
            "[adaptive_dosing]: start_dose 5 is outside dose_bounds [10, 30 * WT]",
        ),
        (
            "500",
            "[0, 30 * WT]",
            Some("[500, 400, 20 * WT]"),
            "[adaptive_dosing]: levels must be strictly increasing",
        ),
        (
            "500",
            "[0, 30 * WT]",
            Some("[500, 10 * WT, 400]"),
            "[adaptive_dosing]: levels must be strictly increasing",
        ),
        (
            "600",
            "[0, 30 * WT]",
            Some("[500, 1000]"),
            "[adaptive_dosing]: start_dose 600 must be one of the declared levels [500.0, 1000.0]",
        ),
        (
            "10 * WT",
            "[0, 450]",
            Some("[100, 500, 20 * WT]"),
            "[adaptive_dosing]: level 500 is outside dose_bounds [0, 450]",
        ),
    ];
    for (start, bounds, levels, want) in refused {
        let lines = block(start, bounds, levels);
        let got = parse_adaptive_dosing_block(&lines).expect_err("refused at parse time");
        assert_eq!(got, want, "{lines:?}");
    }
    // The same pairs with one side an expression: accepted here, checked per subject.
    let parses: [(&str, &str, Option<&str>); 9] = [
        // An expression rung that can take the start dose: between its numeric
        // neighbours, below the first number, above the last (#2667 review r2 N3).
        ("150", "[0, 30 * WT]", Some("[100, 20 * WT, 200]")),
        ("50", "[0, 30 * WT]", Some("[10 * WT, 100, 200]")),
        ("250", "[0, 30 * WT]", Some("[100, 200, 20 * WT]")),
        ("500", "[0, 30 * WT]", None),
        ("5 * WT", "[10, 30 * WT]", None),
        ("500", "[0, 30 * WT]", Some("[400, 500, 20 * WT]")),
        ("500", "[0, 30 * WT]", Some("[400, 10 * WT, 500]")),
        ("600", "[0, 30 * WT]", Some("[500, 12 * WT]")),
        ("10 * WT", "[0, 30 * WT]", Some("[100, 500, 20 * WT]")),
    ];
    for (start, bounds, levels) in parses {
        let lines = block(start, bounds, levels);
        if let Err(e) = parse_adaptive_dosing_block(&lines) {
            panic!("{lines:?} should parse: {e}");
        }
    }
}
