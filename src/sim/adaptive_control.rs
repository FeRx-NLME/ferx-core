//! Compile a declarative [`AdaptiveDosingSpec`] into a runnable controller
//! (#391 S2.2).
//!
//! This module turns the parsed `[adaptive_dosing]` AST into a **controller
//! factory** of the exact shape [`crate::api::simulate_adaptive`] consumes —
//! `Fn() -> impl FnMut(&ControllerCtx) -> Vec<DoseAction>` — so the declarative
//! block reuses the S1 reactive engine, ledger, decision log, verifier, and DV
//! substreams unchanged.
//!
//! The control *logic* here (the first-matching-rule ladder, the `confirm`
//! debounce, continuous vs. discrete-level titration, `dose_bounds` clamping, the
//! running dose) is **model-independent**: it reads its signal by name from
//! [`ControllerCtx::signal`] and returns [`DoseAction`]s. Resolving that signal
//! from the model (compiling the `observe` expression and feeding it through the
//! engine's monitor mechanism, so `Dv` keeps the S1.5 assay substream) is the
//! separate engine-side half of S2.2.
//!
//! Wired to the public file-driven entry point [`crate::simulate_adaptive_from_spec`]
//! in S2.3 (#391).

use crate::ode::OdeOutputFn;
use crate::sim::adaptive::{
    AdaptiveAction, AdaptiveDosingSpec, AdaptiveRoute, AdaptiveRule, Comparison, ControllerCtx,
    ControllerDecision, DoseAction, DoseStep, MonitorSpec, ObserveMode,
};
use crate::types::CompiledModel;

/// The monitor name the declarative controller reads — the `signal` keyword the
/// `when signal <op> <value>` rules compare. The engine-side half declares its
/// `observe` monitor under this same name so the two always agree.
pub(crate) const ADAPTIVE_SIGNAL: &str = "signal";

impl AdaptiveRule {
    /// Does the observed `signal` satisfy this rule's comparison?
    ///
    /// `Eq` uses exact `f64` equality deliberately — it is the user-written
    /// `signal == <value>` semantics; titrating on an exact equality is unusual
    /// but it is what the rule says, not a tolerance we get to invent.
    #[allow(clippy::float_cmp)]
    fn matches(&self, signal: f64) -> bool {
        match self.op {
            Comparison::Lt => signal < self.threshold,
            Comparison::Le => signal <= self.threshold,
            Comparison::Gt => signal > self.threshold,
            Comparison::Ge => signal >= self.threshold,
            Comparison::Eq => signal == self.threshold,
        }
    }
}

/// A compiled, stateful controller for one `(subject, replicate)`.
///
/// Cloned fresh by the factory for every run so per-subject state (the running
/// dose, the `confirm` streak, the titration rung) never leaks across subjects —
/// the structural isolation [`crate::api::simulate_adaptive`] requires.
#[derive(Clone)]
struct AdaptiveController {
    // ── plan (immutable) ──
    route: AdaptiveRoute,
    dose_bounds: (f64, f64),
    confirm: u32,
    rules: Vec<AdaptiveRule>,
    levels: Option<Vec<f64>>,
    // ── running state ──
    /// The dose carried forward between decisions (re-issued when no rule fires).
    dose: f64,
    /// Index into `levels` (discrete titration only).
    level: usize,
    /// The rule index matched at the previous decision (for the `confirm` streak).
    last_rule: Option<usize>,
    /// Consecutive matches of `last_rule` so far.
    streak: u32,
    /// Set once a `stop` rule fires; the controller emits nothing thereafter.
    stopped: bool,
}

impl AdaptiveController {
    #[allow(clippy::float_cmp)] // start_dose ∈ levels is guaranteed by the parser
    fn new(spec: &AdaptiveDosingSpec) -> Self {
        let level = match &spec.levels {
            Some(l) => l.iter().position(|&x| x == spec.start_dose).unwrap_or(0),
            None => 0,
        };
        Self {
            route: spec.route.clone(),
            dose_bounds: spec.dose_bounds,
            confirm: spec.confirm,
            rules: spec.rules.clone(),
            levels: spec.levels.clone(),
            dose: spec.start_dose,
            level,
            last_rule: None,
            streak: 0,
            stopped: false,
        }
    }

    /// One decision: read the signal, find the first matching rule, apply the
    /// `confirm` debounce, and return the dose action(s).
    fn decide(&mut self, ctx: &ControllerCtx) -> ControllerDecision {
        if self.stopped {
            return ControllerDecision {
                actions: vec![],
                rule: None,
            };
        }
        // The engine-side half always declares the `signal` monitor this
        // controller reads, so `None` is an internal wiring bug, not user input;
        // re-issue the current dose rather than fabricate a decision on a missing
        // signal.
        let signal = match ctx.signal(ADAPTIVE_SIGNAL) {
            Some(v) => v,
            None => {
                return ControllerDecision {
                    actions: self.emit_dose(self.dose),
                    rule: None,
                }
            }
        };

        let matched = self.rules.iter().position(|r| r.matches(signal));
        match matched {
            Some(idx) if self.last_rule == Some(idx) => self.streak += 1,
            Some(idx) => {
                self.last_rule = Some(idx);
                self.streak = 1;
            }
            None => {
                self.last_rule = None;
                self.streak = 0;
            }
        }

        // Act only once a rule has matched `confirm` times in a row; until then
        // (and when nothing matches) re-issue the current dose — the regimen
        // continues unchanged, an explicit no-op rather than a silent skip.
        if let Some(idx) = matched {
            if self.streak >= self.confirm {
                self.streak = 0; // a fresh streak must build before acting again
                                 // Name the rule that fired (for the ledger's `rule_fired`); compute
                                 // the label before `apply` takes `&mut self`.
                let rule = self.rules[idx].label();
                let action = self.rules[idx].action.clone();
                return ControllerDecision {
                    actions: self.apply(action),
                    rule: Some(rule),
                };
            }
        }
        ControllerDecision {
            actions: self.emit_dose(self.dose),
            rule: None,
        }
    }

    fn apply(&mut self, action: AdaptiveAction) -> Vec<DoseAction> {
        match action {
            AdaptiveAction::Hold => vec![], // skip this decision's dose
            AdaptiveAction::Stop => {
                self.stopped = true;
                vec![DoseAction::Stop]
            }
            AdaptiveAction::Increase(step) => {
                self.step_dose(step, true);
                self.emit_dose(self.dose)
            }
            AdaptiveAction::Decrease(step) => {
                self.step_dose(step, false);
                self.emit_dose(self.dose)
            }
        }
    }

    /// Move the running dose one step up/down, clamped to `dose_bounds`.
    fn step_dose(&mut self, step: DoseStep, up: bool) {
        let (lo, hi) = self.dose_bounds;
        match (step, &self.levels) {
            (DoseStep::Percent(p), _) => {
                let factor = if up { 1.0 + p / 100.0 } else { 1.0 - p / 100.0 };
                self.dose = (self.dose * factor).clamp(lo, hi);
            }
            (DoseStep::Level, Some(levels)) => {
                self.level = if up {
                    (self.level + 1).min(levels.len() - 1)
                } else {
                    self.level.saturating_sub(1)
                };
                self.dose = levels[self.level].clamp(lo, hi);
            }
            // Parser guarantees a Level step only appears with a `levels` ladder.
            (DoseStep::Level, None) => {}
        }
    }

    fn emit_dose(&self, dose: f64) -> Vec<DoseAction> {
        // A dose titrated to exactly 0 is a hold, not a degenerate zero-amount
        // dose. An `Infuse { amt: 0 }` would carry `rate = 0 / over = 0`, which the
        // driver's up-front action validation rejects (`rate > 0`), erroring the
        // whole run instead of holding — whereas a zero bolus is silently skipped.
        // Emitting `Hold` keeps the bolus and infusion routes symmetric.
        if dose == 0.0 {
            return vec![DoseAction::Hold];
        }
        match self.route {
            AdaptiveRoute::Bolus { cmt } => vec![DoseAction::Bolus { amt: dose, cmt }],
            AdaptiveRoute::Infuse { cmt, over } => vec![DoseAction::Infuse {
                amt: dose,
                cmt,
                rate: dose / over,
            }],
        }
    }
}

/// Compile a declarative spec into a per-subject controller **factory**.
///
/// Each call to the returned factory mints a *fresh* controller with its own
/// running state (dose / `confirm` streak / titration rung), so a controller's
/// state never leaks across `(subject, replicate)` runs — the isolation
/// [`crate::api::simulate_adaptive`] makes structural via a factory rather than a
/// shared closure. The controller reads its signal under [`ADAPTIVE_SIGNAL`].
#[allow(clippy::type_complexity)] // a controller factory is inherently nested Fn/FnMut
pub(crate) fn build_adaptive_controller(
    spec: &AdaptiveDosingSpec,
) -> impl Fn() -> Box<dyn FnMut(&ControllerCtx) -> ControllerDecision> {
    let template = AdaptiveController::new(spec);
    move || {
        let mut controller = template.clone();
        Box::new(move |ctx: &ControllerCtx| controller.decide(ctx))
    }
}

/// Compile a latent (`Ipred`) `observe` expression into the engine's
/// readout-closure shape ([`OdeOutputFn`]) by reusing the model's own
/// output-expression compiler
/// ([`crate::parser::model_parser::build_y_output_fn`]). Used only for the latent
/// signal; the assay-noised (`Dv`) path reads the model's own output instead, so
/// no expression is compiled there.
///
/// The closure resolves states + individual parameters + covariates exactly as a
/// model readout does, so the controller observes the same quantity the model
/// would predict — `observe = central / V` yields the concentration, not the raw
/// amount. ODE-only (the adaptive engine runs on the ODE path); an unknown name
/// or an IOV reference in `observe` is rejected by the shared compiler.
pub(crate) fn compile_observe(
    model: &CompiledModel,
    observe: &str,
) -> Result<(OdeOutputFn, Vec<String>), String> {
    let ode = model.ode_spec.as_ref().ok_or_else(|| {
        "[adaptive_dosing] requires an ODE model (the analytical engine is a follow-up)".to_string()
    })?;
    // `cov_names` are the covariates the `observe` expression references; the
    // caller validates them against the data so a misspelt name fails loudly
    // instead of silently reading 0.0 (the readout leaves an absent covariate at
    // 0.0, which would drive the controller off a wrong signal).
    let (out_fn, _program, cov_names) = crate::parser::model_parser::build_y_output_fn(
        observe,
        "[adaptive_dosing] observe",
        &model.theta_names,
        &model.eta_names,
        &model.indiv_param_names,
        &model.pk_indices,
        &ode.state_names,
        &model.kappa_names,
    )?;
    Ok((out_fn, cov_names))
}

/// Everything needed to drive a declarative `[adaptive_dosing]` block through the
/// S1 reactive engine: the monitor(s) the engine resolves into `ControllerCtx`,
/// the compiled `observe` readout the engine evaluates for the signal's latent
/// value, and the per-subject controller factory.
pub(crate) struct CompiledAdaptive {
    /// Single monitor named [`ADAPTIVE_SIGNAL`]. Under `Dv` its `cmt` is the
    /// measured model output (value *and* σ); under `Ipred` the value comes from
    /// `observe` and the `cmt` is unused.
    pub monitors: Vec<MonitorSpec>,
    /// Compiled latent (`Ipred`) `observe` expression, or `None` under
    /// `with_assay_error` (`Dv`) — where the signal is the model's own readout for
    /// the monitor's `cmt` plus its assay noise, not a re-typed expression.
    pub observe: Option<OdeOutputFn>,
    /// Covariate names the `observe` expression references (empty under `Dv`) — the
    /// file-driven entry validates these against the data's covariate columns so a
    /// misspelt covariate fails loudly instead of silently reading 0.0.
    pub observe_covariates: Vec<String>,
    /// Mints a fresh controller per `(subject, replicate)` (state isolation).
    #[allow(clippy::type_complexity)]
    pub make_controller: Box<dyn Fn() -> Box<dyn FnMut(&ControllerCtx) -> ControllerDecision>>,
}

/// Compile a declarative spec against its model into the engine-ready
/// [`CompiledAdaptive`] bundle (#391 S2.2).
pub(crate) fn compile_adaptive(
    model: &CompiledModel,
    spec: &AdaptiveDosingSpec,
) -> Result<CompiledAdaptive, String> {
    // Re-check the spec's cross-field invariants here too: the struct is `pub`
    // with `pub` fields, so a programmatically-built spec may not have passed
    // through the block parser (which validates on its way out).
    spec.validate()?;
    // The reactive engine is ODE-only; reject an analytical model on both paths.
    let ode = model.ode_spec.as_ref().ok_or_else(|| {
        "[adaptive_dosing] requires an ODE model (the analytical engine is a follow-up)".to_string()
    })?;
    let (mode, observe, observe_covariates, cmt) = if spec.with_assay_error {
        // Dv: the signal is the *noised measurement* of the model output named by
        // `assay_cmt` (validate() requires it). Its value comes from the model's
        // own readout — `observe = None` makes the driver read `read_observable(cmt)`
        // — and its σ from that cmt's `[error_model]`, so the two share one source
        // and can never be on different scales.
        let cmt = spec
            .assay_cmt
            .expect("validate() requires assay_cmt under with_assay_error");
        if cmt == 0 || cmt > ode.n_states {
            return Err(format!(
                "[adaptive_dosing]: assay_cmt = {cmt} is out of range (model has {} compartment(s))",
                ode.n_states
            ));
        }
        (ObserveMode::Dv, None, Vec::new(), cmt)
    } else {
        // Ipred: titrate on the compiled `observe` expression (no measurement
        // noise); the monitor `cmt` is unused (placeholder 1).
        let observe_src = spec
            .observe
            .as_deref()
            .expect("validate() requires observe without with_assay_error");
        let (out_fn, cov) = compile_observe(model, observe_src)?;
        (ObserveMode::Ipred, Some(out_fn), cov, 1)
    };
    let monitors = vec![MonitorSpec::new(ADAPTIVE_SIGNAL, cmt, mode)];
    let factory = build_adaptive_controller(spec);
    Ok(CompiledAdaptive {
        monitors,
        observe,
        observe_covariates,
        make_controller: Box::new(factory),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::adaptive::DoseStep;
    use std::collections::HashMap;

    fn spec(
        start_dose: f64,
        route: AdaptiveRoute,
        dose_bounds: (f64, f64),
        confirm: u32,
        levels: Option<Vec<f64>>,
        rules: Vec<AdaptiveRule>,
    ) -> AdaptiveDosingSpec {
        AdaptiveDosingSpec {
            observe: Some("central".to_string()),
            with_assay_error: false,
            assay_cmt: None,
            at: vec![24.0, 48.0],
            start_dose,
            route,
            dose_bounds,
            confirm,
            levels,
            rules,
        }
    }

    fn rule(op: Comparison, threshold: f64, action: AdaptiveAction) -> AdaptiveRule {
        AdaptiveRule {
            op,
            threshold,
            action,
        }
    }

    /// Drive a fresh controller through a sequence of signal values, returning the
    /// action list each decision produced.
    fn run(spec: &AdaptiveDosingSpec, signals: &[f64]) -> Vec<Vec<DoseAction>> {
        let factory = build_adaptive_controller(spec);
        let mut controller = factory();
        let empty_cov: HashMap<String, f64> = HashMap::new();
        signals
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let mut sig = HashMap::new();
                sig.insert(ADAPTIVE_SIGNAL.to_string(), s);
                let ctx = ControllerCtx {
                    t: i as f64,
                    state: &[],
                    covariates: &empty_cov,
                    history: &[],
                    decision_index: i,
                    signals: &sig,
                };
                controller(&ctx).actions
            })
            .collect()
    }

    fn bolus(amt: f64, cmt: usize) -> Vec<DoseAction> {
        vec![DoseAction::Bolus { amt, cmt }]
    }

    #[test]
    fn reissues_start_dose_when_no_rule_matches() {
        let s = spec(
            100.0,
            AdaptiveRoute::Bolus { cmt: 1 },
            (0.0, 400.0),
            1,
            None,
            vec![rule(
                Comparison::Lt,
                10.0,
                AdaptiveAction::Increase(DoseStep::Percent(25.0)),
            )],
        );
        let out = run(&s, &[50.0, 50.0]); // signal never < 10
        assert_eq!(out[0], bolus(100.0, 1));
        assert_eq!(out[1], bolus(100.0, 1));
    }

    #[test]
    fn confirm_one_acts_immediately() {
        let s = spec(
            100.0,
            AdaptiveRoute::Bolus { cmt: 1 },
            (0.0, 400.0),
            1,
            None,
            vec![rule(
                Comparison::Lt,
                10.0,
                AdaptiveAction::Increase(DoseStep::Percent(25.0)),
            )],
        );
        let out = run(&s, &[5.0]);
        assert_eq!(out[0], bolus(125.0, 1));
    }

    #[test]
    fn confirm_debounces_then_acts_then_rebuilds_streak() {
        let s = spec(
            100.0,
            AdaptiveRoute::Bolus { cmt: 1 },
            (0.0, 1000.0),
            2,
            None,
            vec![rule(
                Comparison::Lt,
                10.0,
                AdaptiveAction::Increase(DoseStep::Percent(25.0)),
            )],
        );
        // signal < 10 at every decision; confirm = 2 ⇒ act on every 2nd consecutive match.
        let out = run(&s, &[5.0, 5.0, 5.0, 5.0]);
        assert_eq!(out[0], bolus(100.0, 1)); // streak 1 → re-issue
        assert_eq!(out[1], bolus(125.0, 1)); // streak 2 → +25%, reset
        assert_eq!(out[2], bolus(125.0, 1)); // streak 1 → re-issue
        assert_eq!(out[3], bolus(156.25, 1)); // streak 2 → +25% again
    }

    #[test]
    fn decrease_clamps_at_lower_bound() {
        let s = spec(
            100.0,
            AdaptiveRoute::Bolus { cmt: 1 },
            (10.0, 400.0),
            1,
            None,
            vec![rule(
                Comparison::Gt,
                20.0,
                AdaptiveAction::Decrease(DoseStep::Percent(50.0)),
            )],
        );
        let out = run(&s, &[30.0, 30.0, 30.0]); // 100 → 50 → 25 → 12.5? clamp at 10
        assert_eq!(out[0], bolus(50.0, 1));
        assert_eq!(out[1], bolus(25.0, 1));
        assert_eq!(out[2], bolus(12.5, 1)); // 12.5 still ≥ 10
    }

    #[test]
    fn decrease_to_zero_holds_for_both_routes() {
        // low bound 0 ⇒ a 100% decrease floors the dose at 0. A 0 dose is a *hold*,
        // not a zero-amount dose: an `Infuse { amt: 0 }` would carry `rate = 0`,
        // which the driver's up-front action validation rejects (`rate > 0`),
        // erroring the whole run. Both routes must emit `Hold` instead.
        let decrease_100 = || {
            vec![rule(
                Comparison::Gt,
                1.0,
                AdaptiveAction::Decrease(DoseStep::Percent(100.0)),
            )]
        };
        // Bolus route (previously a zero bolus the driver silently skipped).
        let s = spec(
            100.0,
            AdaptiveRoute::Bolus { cmt: 1 },
            (0.0, 400.0),
            1,
            None,
            decrease_100(),
        );
        assert_eq!(run(&s, &[50.0])[0], vec![DoseAction::Hold]);

        // Infuse route: previously emitted `Infuse { amt: 0, rate: 0 }`, the
        // route-asymmetric crash this guards against.
        let s = spec(
            100.0,
            AdaptiveRoute::Infuse { cmt: 1, over: 2.0 },
            (0.0, 400.0),
            1,
            None,
            decrease_100(),
        );
        assert_eq!(run(&s, &[50.0])[0], vec![DoseAction::Hold]);
    }

    #[test]
    fn discrete_levels_step_up_and_saturate_then_down() {
        let s = spec(
            100.0,
            AdaptiveRoute::Bolus { cmt: 1 },
            (0.0, 200.0),
            1,
            Some(vec![50.0, 100.0, 150.0, 200.0]),
            vec![
                rule(
                    Comparison::Lt,
                    1.0,
                    AdaptiveAction::Increase(DoseStep::Level),
                ),
                rule(
                    Comparison::Gt,
                    3.0,
                    AdaptiveAction::Decrease(DoseStep::Level),
                ),
            ],
        );
        // start at level idx 1 (=100). up, up (saturate at 200), then down.
        let out = run(&s, &[0.0, 0.0, 0.0, 5.0]);
        assert_eq!(out[0], bolus(150.0, 1));
        assert_eq!(out[1], bolus(200.0, 1));
        assert_eq!(out[2], bolus(200.0, 1)); // saturates at top level
        assert_eq!(out[3], bolus(150.0, 1)); // signal > 3 → step down
    }

    #[test]
    fn hold_skips_then_regimen_continues() {
        let s = spec(
            100.0,
            AdaptiveRoute::Bolus { cmt: 1 },
            (0.0, 400.0),
            1,
            None,
            vec![rule(Comparison::Gt, 20.0, AdaptiveAction::Hold)],
        );
        let out = run(&s, &[30.0, 5.0]);
        assert!(out[0].is_empty()); // hold → no dose this cycle
        assert_eq!(out[1], bolus(100.0, 1)); // no rule matches → re-issue
    }

    #[test]
    fn stop_emits_stop_then_silence() {
        let s = spec(
            100.0,
            AdaptiveRoute::Bolus { cmt: 1 },
            (0.0, 400.0),
            1,
            None,
            vec![rule(Comparison::Gt, 40.0, AdaptiveAction::Stop)],
        );
        let out = run(&s, &[50.0, 50.0]);
        assert_eq!(out[0], vec![DoseAction::Stop]);
        assert!(out[1].is_empty()); // nothing after Stop
    }

    #[test]
    fn infuse_route_sets_rate_from_duration() {
        let s = spec(
            100.0,
            AdaptiveRoute::Infuse { cmt: 1, over: 2.0 },
            (0.0, 400.0),
            1,
            None,
            vec![rule(Comparison::Lt, 1.0, AdaptiveAction::Hold)],
        );
        let out = run(&s, &[50.0]); // no match → re-issue start_dose as an infusion
        assert_eq!(
            out[0],
            vec![DoseAction::Infuse {
                amt: 100.0,
                cmt: 1,
                rate: 50.0, // 100 / over(2)
            }]
        );
    }

    #[test]
    fn first_matching_rule_wins() {
        let s = spec(
            100.0,
            AdaptiveRoute::Bolus { cmt: 1 },
            (0.0, 400.0),
            1,
            None,
            vec![
                rule(Comparison::Gt, 10.0, AdaptiveAction::Hold), // matches first for signal 50
                rule(
                    Comparison::Gt,
                    20.0,
                    AdaptiveAction::Decrease(DoseStep::Percent(50.0)),
                ),
            ],
        );
        let out = run(&s, &[50.0]);
        assert!(out[0].is_empty()); // first rule (Hold) wins over the later Decrease
    }

    #[test]
    fn factory_mints_independent_controllers() {
        let s = spec(
            100.0,
            AdaptiveRoute::Bolus { cmt: 1 },
            (0.0, 1000.0),
            1,
            None,
            vec![rule(
                Comparison::Lt,
                10.0,
                AdaptiveAction::Increase(DoseStep::Percent(50.0)),
            )],
        );
        let factory = build_adaptive_controller(&s);
        let empty_cov: HashMap<String, f64> = HashMap::new();
        let mut sig = HashMap::new();
        sig.insert(ADAPTIVE_SIGNAL.to_string(), 5.0);
        let ctx = ControllerCtx {
            t: 0.0,
            state: &[],
            covariates: &empty_cov,
            history: &[],
            decision_index: 0,
            signals: &sig,
        };
        // Drive controller A twice (100 → 150 → 225); a fresh controller B must
        // start from 100 again, proving no shared state.
        let mut a = factory();
        let _ = a(&ctx);
        let a2 = a(&ctx).actions;
        assert_eq!(a2, bolus(225.0, 1));
        let mut b = factory();
        let b1 = b(&ctx).actions;
        assert_eq!(b1, bolus(150.0, 1));
    }

    // ── Engine-side compile (observe expression → readout closure) ──

    use crate::parser::model_parser::parse_full_model;

    const ODE_MODEL: &str = r#"
[parameters]
  theta TVCL(1.0)
  theta TVV(50.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP)
"#;

    const ANALYTICAL_MODEL: &str = r#"
[parameters]
  theta TVCL(1.0)
  theta TVV(50.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP)
"#;

    fn mk_spec(
        observe: &str,
        with_assay_error: bool,
        assay_cmt: Option<usize>,
    ) -> AdaptiveDosingSpec {
        AdaptiveDosingSpec {
            // `observe` is the latent (Ipred) signal; under assay error there is no
            // expression — the signal is the named model output (`assay_cmt`).
            observe: (!with_assay_error).then(|| observe.to_string()),
            with_assay_error,
            assay_cmt,
            at: vec![24.0, 48.0],
            start_dose: 100.0,
            route: AdaptiveRoute::Bolus { cmt: 1 },
            dose_bounds: (0.0, 400.0),
            confirm: 1,
            levels: None,
            rules: vec![rule(Comparison::Gt, 1000.0, AdaptiveAction::Hold)],
        }
    }

    #[test]
    fn compile_observe_yields_concentration_not_amount() {
        let model = parse_full_model(ODE_MODEL).expect("ODE model parses").model;
        let (observe, _cov) = compile_observe(&model, "central / V").expect("observe compiles");
        let theta = model.default_params.theta.clone();
        let eta = vec![0.0; model.n_eta + model.n_kappa];
        let cov = HashMap::new();
        let pk = (model.pk_param_fn)(&theta, &eta, &cov);
        // central amount = 200, V = 50 ⇒ concentration 4.0 — NOT the raw amount 200.
        let v = observe(&[200.0], &pk.values, &theta, &eta, &cov);
        assert!(
            (v - 4.0).abs() < 1e-9,
            "expected concentration 4.0, got {v}"
        );
    }

    #[test]
    fn compile_observe_requires_ode_model() {
        let model = parse_full_model(ANALYTICAL_MODEL).unwrap().model;
        let err = compile_observe(&model, "central / V")
            .err()
            .expect("analytical model must error");
        assert!(err.contains("requires an ODE model"), "got: {err}");
    }

    #[test]
    fn compile_adaptive_sets_monitor_mode_and_cmt() {
        let model = parse_full_model(ODE_MODEL).unwrap().model;
        let ipred = compile_adaptive(&model, &mk_spec("central / V", false, None)).unwrap();
        assert_eq!(ipred.monitors.len(), 1);
        assert_eq!(ipred.monitors[0].name, ADAPTIVE_SIGNAL);
        assert_eq!(ipred.monitors[0].mode, ObserveMode::Ipred);
        assert!(
            ipred.observe.is_some(),
            "Ipred compiles the observe expression"
        );

        // Dv: the signal is the noised model output named by `assay_cmt`; its value
        // comes from the model's own readout, so no `observe` expression is compiled.
        let dv = compile_adaptive(&model, &mk_spec("", true, Some(1))).unwrap();
        assert_eq!(dv.monitors[0].mode, ObserveMode::Dv);
        assert_eq!(dv.monitors[0].cmt, 1);
        assert!(
            dv.observe.is_none(),
            "Dv reads the model output, not a re-typed expression"
        );
    }

    #[test]
    fn compile_adaptive_rejects_signal_misconfiguration() {
        let model = parse_full_model(ODE_MODEL).unwrap().model; // single compartment
                                                                // assay error without naming which output is measured
        let err = compile_adaptive(&model, &mk_spec("", true, None))
            .err()
            .expect("with_assay_error needs assay_cmt");
        assert!(
            err.contains("requires `assay_cmt"),
            "needs assay_cmt: {err}"
        );
        // a measured output beyond the model's compartments
        let err = compile_adaptive(&model, &mk_spec("", true, Some(3)))
            .err()
            .expect("out-of-range assay_cmt must error");
        assert!(err.contains("out of range"), "out of range: {err}");
    }
}
