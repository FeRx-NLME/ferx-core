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

use crate::sim::adaptive::{
    AdaptiveAction, AdaptiveDosingSpec, AdaptiveRoute, AdaptiveRule, Comparison, ControllerCtx,
    DoseAction, DoseStep,
};

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
    fn decide(&mut self, ctx: &ControllerCtx) -> Vec<DoseAction> {
        if self.stopped {
            return vec![];
        }
        // The engine-side half always declares the `signal` monitor this
        // controller reads, so `None` is an internal wiring bug, not user input;
        // re-issue the current dose rather than fabricate a decision on a missing
        // signal.
        let signal = match ctx.signal(ADAPTIVE_SIGNAL) {
            Some(v) => v,
            None => return self.emit_dose(self.dose),
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
                let action = self.rules[idx].action.clone();
                return self.apply(action);
            }
        }
        self.emit_dose(self.dose)
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
pub(crate) fn build_adaptive_controller(
    spec: &AdaptiveDosingSpec,
) -> impl Fn() -> Box<dyn FnMut(&ControllerCtx) -> Vec<DoseAction>> {
    let template = AdaptiveController::new(spec);
    move || {
        let mut controller = template.clone();
        Box::new(move |ctx: &ControllerCtx| controller.decide(ctx))
    }
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
            observe: "central".to_string(),
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
                controller(&ctx)
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
    fn decrease_to_zero_emits_zero_bolus() {
        // low bound 0 ⇒ a 100% decrease floors at 0; the driver later normalises
        // amt==0 to Hold, but the controller's job is just to emit the clamped dose.
        let s = spec(
            100.0,
            AdaptiveRoute::Bolus { cmt: 1 },
            (0.0, 400.0),
            1,
            None,
            vec![rule(
                Comparison::Gt,
                1.0,
                AdaptiveAction::Decrease(DoseStep::Percent(100.0)),
            )],
        );
        let out = run(&s, &[50.0]);
        assert_eq!(out[0], bolus(0.0, 1));
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
        let a2 = a(&ctx);
        assert_eq!(a2, bolus(225.0, 1));
        let mut b = factory();
        let b1 = b(&ctx);
        assert_eq!(b1, bolus(150.0, 1));
    }
}
