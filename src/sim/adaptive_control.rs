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

use crate::diagnostics::Diagnostic;
use crate::ode::OdeOutputFn;
use crate::sim::adaptive::{
    dose_plan_violation, AdaptiveAction, AdaptiveDosingSpec, AdaptiveRoute, AdaptiveRule,
    Comparison, ControllerCtx, ControllerDecision, DoseAction, DoseGrid, DosePlanViolation,
    DoseStep, IntervalStep, MonitorSpec, ObserveMode, SubjectDosePlan,
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
    /// The dose grid (`round_to`, #944): every dose this controller sets lands on it.
    /// `None` without `round_to`, where doses are only clamped to `dose_bounds`.
    grid: Option<DoseGrid>,
    confirm: u32,
    rules: Vec<AdaptiveRule>,
    levels: Option<Vec<f64>>,
    /// The dosing-interval ladder (#945); `None` when the block declares none.
    intervals: Option<Vec<f64>>,
    // ── running state ──
    /// The dose carried forward between decisions (re-issued when no rule fires).
    dose: f64,
    /// Index into `levels` (discrete titration only).
    level: usize,
    /// Index into `intervals`: the interval in force. The driver holds the same
    /// value, since only this controller sets it, from `start_interval` on.
    rung: usize,
    /// The rule index matched at the previous decision (for the `confirm` streak).
    last_rule: Option<usize>,
    /// Consecutive matches of `last_rule` so far.
    streak: u32,
    /// Set once a `stop` rule fires; the controller emits nothing thereafter.
    stopped: bool,
}

impl AdaptiveController {
    /// The controller for one subject: the block's rules and route, run from that
    /// subject's dose plan (#944). `plan` passed `dose_plan_violation`, so `start_dose`
    /// is a rung of `levels` when there is a ladder.
    #[allow(clippy::float_cmp)] // start_dose ∈ levels is checked before any subject runs
    fn new(spec: &AdaptiveDosingSpec, plan: &SubjectDosePlan) -> Self {
        let level = match &plan.levels {
            Some(l) => l.iter().position(|&x| x == plan.start_dose).unwrap_or(0),
            None => 0,
        };
        let rung = match (&spec.intervals, spec.start_interval) {
            (Some(l), Some(d)) => l.iter().position(|&x| x == d).unwrap_or(0),
            _ => 0,
        };
        // The start dose goes onto the grid by the same path as every step (#944). A
        // plan that passed `dose_plan_violation` has a grid whenever `round_to` is set.
        let grid = spec.dose_grid(plan.dose_bounds);
        let dose = grid.map_or(plan.start_dose, |g| g.place(plan.start_dose));
        Self {
            route: spec.route.clone(),
            dose_bounds: plan.dose_bounds,
            grid,
            confirm: spec.confirm,
            rules: spec.rules.clone(),
            levels: plan.levels.clone(),
            intervals: spec.intervals.clone(),
            dose,
            level,
            rung,
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
                let step = self.rules[idx].interval;
                let mut actions = self.apply(action);
                actions.extend(self.step_interval(step));
                return ControllerDecision {
                    actions,
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
            AdaptiveAction::Maintain => self.emit_dose(self.dose),
        }
    }

    /// Move the interval one rung (#945), saturating at both ends, and name the new
    /// interval to the driver. A step that does not move the rung (already at that
    /// end) emits nothing: the interval in force is unchanged.
    fn step_interval(&mut self, step: Option<IntervalStep>) -> Option<DoseAction> {
        let (step, ladder) = (step?, self.intervals.as_ref()?);
        let rung = match step {
            IntervalStep::Lengthen => (self.rung + 1).min(ladder.len() - 1),
            IntervalStep::Shorten => self.rung.saturating_sub(1),
        };
        if rung == self.rung {
            return None;
        }
        self.rung = rung;
        Some(DoseAction::SetInterval {
            interval: ladder[rung],
        })
    }

    /// Move the running dose one step up/down, clamped to `dose_bounds` — or, with
    /// `round_to`, placed on the dose grid (rounded, then clamped to the aligned bounds,
    /// #944). The placed dose is what is carried, so the next step starts from the dose
    /// actually given, and a re-issue (`emit_dose`) gives it as is, never rounding it again. A step
    /// the grid absorbs leaves the dose unchanged, as a step clamped at a bound does.
    fn step_dose(&mut self, step: DoseStep, up: bool) {
        let (lo, hi) = self.dose_bounds;
        match (step, &self.levels) {
            (DoseStep::Percent(p), _) => {
                let factor = if up { 1.0 + p / 100.0 } else { 1.0 - p / 100.0 };
                let x = self.dose * factor;
                self.dose = match &self.grid {
                    Some(g) => g.place(x),
                    None => x.clamp(lo, hi),
                };
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
/// shared closure. The controller reads its signal under [`ADAPTIVE_SIGNAL`]. Each
/// call takes the subject's dose plan (#944), so no dose value is shared across
/// subjects either.
#[allow(clippy::type_complexity)] // a controller factory is inherently nested Fn/FnMut
pub(crate) fn build_adaptive_controller(
    spec: &AdaptiveDosingSpec,
) -> impl Fn(&SubjectDosePlan) -> Box<dyn FnMut(&ControllerCtx) -> ControllerDecision> {
    let spec = spec.clone();
    move |plan: &SubjectDosePlan| {
        let mut controller = AdaptiveController::new(&spec, plan);
        Box::new(move |ctx: &ControllerCtx| controller.decide(ctx))
    }
}

/// The `[adaptive_dosing]` dose expressions compiled against the model (#944), one per
/// [`AdaptiveDosingSpec::dose_expressions`] entry, in that order.
pub(crate) struct CompiledDoseExprs {
    exprs: Vec<CompiledDoseExpr>,
}

struct CompiledDoseExpr {
    key: &'static str,
    eval: OdeOutputFn,
    covariates: Vec<String>,
    /// What it reads of the subject (`check_adaptive_dose_expression`'s return).
    reads: crate::parser::model_parser::DoseExprReads,
}

/// A subject's own values at the first decision, for the dose expressions that read
/// them (#944): its `[η | κ]` and the PK they give there. A supplied subject passes its
/// supplied η with κ = 0; an oracle subject its drawn η and first-occasion κ.
pub(crate) struct SubjectValues<'a> {
    pub eta: &'a [f64],
    pub pk: &'a crate::types::PkParams,
}

/// One subject's dose plan, or the first invariant it breaks.
pub(crate) struct ResolvedDosePlan {
    pub plan: SubjectDosePlan,
    /// The covariates the expressions read, with the values read, in first-read order.
    pub reads: Vec<(String, f64)>,
    pub violation: Option<DosePlanViolation>,
}

impl CompiledDoseExprs {
    /// The data covariates the expressions read, each once, with the first key that
    /// reads it.
    pub(crate) fn covariates(&self) -> Vec<(&'static str, &str)> {
        let mut out: Vec<(&'static str, &str)> = Vec::new();
        for e in &self.exprs {
            for c in &e.covariates {
                if !out.iter().any(|(_, n)| n == c) {
                    out.push((e.key, c));
                }
            }
        }
        out
    }

    /// The subject values the expressions read (individual parameters a draw reaches,
    /// random effects, `MIXNUM`), each once, with the first key that reads it.
    pub(crate) fn subject_value_reads(&self) -> Vec<(&'static str, &str)> {
        first_reads(&self.exprs, |r| &r.subject_values)
    }

    /// Whether any expression reads a subject value, so that a subject may be dosed from
    /// it only when that value is known (or under `dose_knowledge = oracle`).
    pub(crate) fn reads_subject_values(&self) -> bool {
        self.exprs
            .iter()
            .any(|e| !e.reads.subject_values.is_empty())
    }

    /// Whether any expression reads an individual parameter, a draw-free one included:
    /// every such read needs the subject's PK at the first decision.
    pub(crate) fn reads_indiv_params(&self) -> bool {
        self.exprs.iter().any(|e| !e.reads.indiv_params.is_empty())
    }

    /// Evaluate every dose value of `spec` for one subject, from `theta`, the
    /// covariates `covs` in force at the first decision, and — when an expression reads
    /// them — the subject's own values `subject` there (#944).
    ///
    /// The parser refuses a state and `TIME` (`check_adaptive_dose_expression`), and
    /// `compile_adaptive` refuses them again on a spec built in code, so the state slice
    /// is zeros that no expression reads. With `subject` absent the PK and η slices are
    /// zeros too; the caller passes `None` only when [`Self::reads_indiv_params`] and
    /// [`Self::reads_subject_values`] are both false.
    pub(crate) fn resolve(
        &self,
        spec: &AdaptiveDosingSpec,
        model: &CompiledModel,
        theta: &[f64],
        covs: &std::collections::HashMap<String, f64>,
        subject: Option<&SubjectValues<'_>>,
    ) -> ResolvedDosePlan {
        let n_states = model.ode_spec.as_ref().map_or(0, |o| o.n_states);
        let states = vec![0.0; n_states];
        let zero_pk = crate::types::PkParams::default();
        let zero_eta = vec![0.0; model.n_eta + model.n_kappa];
        let (pk, eta) = match subject {
            Some(v) => (v.pk, v.eta),
            None => (&zero_pk, zero_eta.as_slice()),
        };
        let mut values = self
            .exprs
            .iter()
            .map(|e| (e.eval)(&states, &pk.values, theta, eta, covs));
        let mut value = |v: &crate::sim::adaptive::DoseValue| match v.as_literal() {
            Some(x) => x,
            None => values
                .next()
                .expect("one compiled expression per dose expression, in order"),
        };
        // `dose_expressions` order: start_dose, the two bounds, then the levels.
        let start_dose = value(&spec.start_dose);
        let dose_bounds = (value(&spec.dose_bounds.0), value(&spec.dose_bounds.1));
        let levels = spec
            .levels
            .as_ref()
            .map(|l| l.iter().map(&mut value).collect::<Vec<f64>>());
        let plan = SubjectDosePlan {
            start_dose,
            dose_bounds,
            levels,
        };
        let mut reads: Vec<(String, f64)> = self
            .covariates()
            .into_iter()
            .map(|(_, c)| {
                (
                    c.to_string(),
                    crate::parser::model_parser::output_covariate_read(covs, c),
                )
            })
            .collect();
        // Every individual parameter read, then the random effects and `MIXNUM`.
        let indiv = first_reads(&self.exprs, |r| &r.indiv_params);
        let others = self.subject_value_reads();
        let mut names: Vec<&str> = indiv.into_iter().map(|(_, n)| n).collect();
        for (_, n) in others {
            if !names.contains(&n) {
                names.push(n);
            }
        }
        reads.extend(
            names
                .into_iter()
                .map(|n| (n.to_string(), subject_value_read(model, pk, eta, n))),
        );
        ResolvedDosePlan {
            violation: dose_plan_violation(spec, &plan),
            plan,
            reads,
        }
    }
}

/// Each name `pick` lists across `exprs`, once, with the first key that reads it.
fn first_reads<'a>(
    exprs: &'a [CompiledDoseExpr],
    pick: impl Fn(&'a crate::parser::model_parser::DoseExprReads) -> &'a Vec<String>,
) -> Vec<(&'static str, &'a str)> {
    let mut out: Vec<(&'static str, &str)> = Vec::new();
    for e in exprs {
        for n in pick(&e.reads) {
            if !out.iter().any(|(_, m)| m == n) {
                out.push((e.key, n));
            }
        }
    }
    out
}

/// The value a dose expression reads for subject value `name` — the slot
/// `build_y_output_fn` reads it from: an individual parameter's PK slot, a random
/// effect's η, or the mixture class. For the "reads" sentence of a refusal, so it
/// reports the value the expression used.
fn subject_value_read(
    model: &CompiledModel,
    pk: &crate::types::PkParams,
    eta: &[f64],
    name: &str,
) -> f64 {
    if name == "MIXNUM" {
        return crate::parser::model_parser::current_mixture_class() as f64;
    }
    if let Some(i) = model.indiv_param_names.iter().position(|n| n == name) {
        return pk.values[model.pk_indices.get(i).copied().unwrap_or(i)];
    }
    model
        .eta_names
        .iter()
        .position(|n| n == name)
        .map_or(f64::NAN, |i| eta[i])
}

/// Compile every dose expression of `spec` (#944) through the same
/// `build_y_output_fn` an `observe` expression goes through, after refusing the reads
/// a dose expression may not make (the parser's check, again, for a spec built in
/// code).
pub(crate) fn compile_dose_exprs(
    model: &CompiledModel,
    spec: &AdaptiveDosingSpec,
) -> Result<CompiledDoseExprs, String> {
    let state_names: &[String] = model
        .ode_spec
        .as_ref()
        .map_or(&[], |o| o.state_names.as_slice());
    let mut exprs = Vec::new();
    for (key, src) in spec.dose_expressions() {
        let reads = crate::parser::model_parser::check_adaptive_dose_expression(
            src,
            key,
            model,
            &spec.observe_declared_covariates,
            spec.at[0],
            spec.dose_knowledge,
        )?;
        let context = format!("[adaptive_dosing] {key}");
        let mut warnings: Vec<String> = Vec::new();
        let (eval, _program, covariates) = crate::parser::model_parser::build_y_output_fn(
            src,
            &context,
            &model.theta_names,
            &model.eta_names,
            &model.indiv_param_names,
            &model.pk_indices,
            state_names,
            &model.kappa_names,
            &[],
            &[],
            &spec.observe_declared_covariates,
            &[],
            &mut warnings,
            &mut Vec::new(),
        )?;
        exprs.push(CompiledDoseExpr {
            key,
            eval,
            covariates,
            reads,
        });
    }
    let compiled = CompiledDoseExprs { exprs };
    crate::parser::model_parser::check_dose_knowledge_used(
        spec.dose_knowledge,
        compiled.reads_subject_values(),
    )?;
    Ok(compiled)
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
/// `declared_covariates` is the model's `[covariates]` declarations, carried on the
/// spec as [`AdaptiveDosingSpec::observe_declared_covariates`]: `observe` must apply
/// the same `T` / `t` name precedence a `[scaling]` expression gets, or the same
/// model reads a declared `T` as the data column in `[scaling]` and as the
/// model-time built-in here (#1028).
pub(crate) fn compile_observe(
    model: &CompiledModel,
    observe: &str,
    declared_covariates: &[String],
) -> Result<(OdeOutputFn, Vec<String>), String> {
    let ode = model.ode_spec.as_ref().ok_or_else(|| {
        "[adaptive_dosing] requires an ODE model (the analytical engine is a follow-up)".to_string()
    })?;
    // `cov_names` are the covariates the `observe` expression references; the
    // caller validates them against the data so a misspelt name fails loudly
    // instead of silently reading 0.0 (the readout leaves an absent covariate at
    // 0.0, which would drive the controller off a wrong signal).
    //
    // The shared compiler's `T`-alias note is discarded *here* on purpose: this runs at
    // simulate time, long after `model.parse_warnings` was sealed, and the adaptive
    // result has no warnings channel. The parser emits the identical note for this same
    // `observe` string at parse time, from the same helper
    // (`warn_if_time_alias_folds`), so the fold is not silent — it is just reported
    // earlier (#1028).
    let mut observe_warnings: Vec<String> = Vec::new();
    let (out_fn, _program, cov_names) = crate::parser::model_parser::build_y_output_fn(
        observe,
        "[adaptive_dosing] observe",
        &model.theta_names,
        &model.eta_names,
        &model.indiv_param_names,
        &model.pk_indices,
        &ode.state_names,
        &model.kappa_names,
        // The latent `observe` readout is only evaluated for the simulation
        // controller (not the analytic-sensitivity path), so no θ/η desugaring (#486).
        &[],
        // ODE-only path: any integrated state is valid, no forbidden names (#650).
        &[],
        declared_covariates,
        // `[scaling]` named intermediates (#1030) are local to that block; the
        // `observe` expression is its own one-line readout with nothing to inline.
        &[],
        &mut observe_warnings,
        // No synthetics were passed in, so none can be read.
        &mut Vec::new(),
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
    /// The dose expressions (#944), evaluated per subject before any subject runs.
    pub dose_exprs: CompiledDoseExprs,
    /// Mints a fresh controller per `(subject, replicate)` (state isolation), from that
    /// subject's dose plan (#944).
    #[allow(clippy::type_complexity)]
    pub make_controller:
        Box<dyn Fn(&SubjectDosePlan) -> Box<dyn FnMut(&ControllerCtx) -> ControllerDecision>>,
}

/// Compile a declarative spec against its model into the engine-ready
/// [`CompiledAdaptive`] bundle (#391 S2.2).
pub(crate) fn compile_adaptive(
    model: &CompiledModel,
    spec: &AdaptiveDosingSpec,
) -> Result<CompiledAdaptive, Diagnostic> {
    // Re-check the spec's cross-field invariants here too: the struct is `pub`
    // with `pub` fields, so a programmatically-built spec may not have passed
    // through the block parser (which validates on its way out).
    spec.validate().map_err(crate::api::adaptive_spec)?;
    // The reactive engine is ODE-only; reject an analytical model on both paths. A model
    // the driver cannot run, not a malformed spec, so `E_ADAPTIVE_UNSUPPORTED` (#1774).
    let ode = model.ode_spec.as_ref().ok_or_else(|| {
        crate::api::adaptive_unsupported_diag(
            None,
            "[adaptive_dosing] requires an ODE model (the analytical engine is a follow-up)",
        )
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
            return Err(crate::api::adaptive_spec(format!(
                "[adaptive_dosing]: assay_cmt = {cmt} is out of range (model has {} compartment(s))",
                ode.n_states
            )));
        }
        (ObserveMode::Dv, None, Vec::new(), cmt)
    } else {
        // Ipred: titrate on the compiled `observe` expression (no measurement
        // noise); the monitor `cmt` is unused (placeholder 1).
        let observe_src = spec
            .observe
            .as_deref()
            .expect("validate() requires observe without with_assay_error");
        let (out_fn, cov) = compile_observe(model, observe_src, &spec.observe_declared_covariates)
            .map_err(crate::api::adaptive_spec)?;
        (ObserveMode::Ipred, Some(out_fn), cov, 1)
    };
    let monitors = vec![MonitorSpec::new(ADAPTIVE_SIGNAL, cmt, mode)];
    let dose_exprs = compile_dose_exprs(model, spec).map_err(crate::api::adaptive_spec)?;
    let factory = build_adaptive_controller(spec);
    Ok(CompiledAdaptive {
        monitors,
        observe,
        observe_covariates,
        dose_exprs,
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
            observe_declared_covariates: Vec::new(),
            with_assay_error: false,
            assay_cmt: None,
            at: vec![24.0, 48.0],
            start_dose: start_dose.into(),
            route,
            dose_bounds: (dose_bounds.0.into(), dose_bounds.1.into()),
            confirm,
            levels: levels.map(|l| l.into_iter().map(Into::into).collect()),
            target_window: None,
            auc_target: None,
            round_to: None,
            dose_knowledge: Default::default(),
            rules,
            intervals: None,
            start_interval: None,
        }
    }

    fn rule(op: Comparison, threshold: f64, action: AdaptiveAction) -> AdaptiveRule {
        AdaptiveRule {
            op,
            threshold,
            action,
            interval: None,
        }
    }

    /// Drive a fresh controller through a sequence of signal values, returning the
    /// action list each decision produced.
    fn run(spec: &AdaptiveDosingSpec, signals: &[f64]) -> Vec<Vec<DoseAction>> {
        let factory = build_adaptive_controller(spec);
        let mut controller = factory(&spec.literal_dose_plan().expect("a block of numbers"));
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
                    interval: None,
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
            interval: None,
        };
        // Drive controller A twice (100 → 150 → 225); a fresh controller B must
        // start from 100 again, proving no shared state.
        let plan = s.literal_dose_plan().expect("a block of numbers");
        let mut a = factory(&plan);
        let _ = a(&ctx);
        let a2 = a(&ctx).actions;
        assert_eq!(a2, bolus(225.0, 1));
        let mut b = factory(&plan);
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
            observe_declared_covariates: Vec::new(),
            with_assay_error,
            assay_cmt,
            at: vec![24.0, 48.0],
            start_dose: (100.0).into(),
            route: AdaptiveRoute::Bolus { cmt: 1 },
            dose_bounds: ((0.0).into(), (400.0).into()),
            confirm: 1,
            levels: None,
            target_window: None,
            auc_target: None,
            round_to: None,
            dose_knowledge: Default::default(),
            rules: vec![rule(Comparison::Gt, 1000.0, AdaptiveAction::Hold)],
            intervals: None,
            start_interval: None,
        }
    }

    #[test]
    fn compile_observe_yields_concentration_not_amount() {
        let model = parse_full_model(ODE_MODEL).expect("ODE model parses").model;
        let (observe, _cov) =
            compile_observe(&model, "central / V", &[]).expect("observe compiles");
        let theta = model.default_params.theta.clone();
        let eta = vec![0.0; model.n_eta + model.n_kappa];
        let cov = HashMap::new();
        let pk = (model.pk_param_fn)(&theta, &eta, &cov, 0.0);
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
        let err = compile_observe(&model, "central / V", &[])
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
            err.message.contains("requires `assay_cmt"),
            "needs assay_cmt: {err:?}"
        );
        assert_eq!(err.code, "E_ADAPTIVE_SPEC", "{err:?}");
        // a measured output beyond the model's compartments
        let err = compile_adaptive(&model, &mk_spec("", true, Some(3)))
            .err()
            .expect("out-of-range assay_cmt must error");
        assert!(
            err.message.contains("out of range"),
            "out of range: {err:?}"
        );
        assert_eq!(err.code, "E_ADAPTIVE_SPEC", "{err:?}");
    }

    // ── #945: the dosing-interval ladder ─────────────────────────────────────────

    /// `spec` with an interval ladder and `start_interval`; the rules carry their own
    /// interval effects.
    fn interval_spec(
        confirm: u32,
        intervals: &[f64],
        start: f64,
        rules: Vec<AdaptiveRule>,
    ) -> AdaptiveDosingSpec {
        let mut s = spec(
            100.0,
            AdaptiveRoute::Bolus { cmt: 1 },
            (0.0, 400.0),
            confirm,
            None,
            rules,
        );
        s.intervals = Some(intervals.to_vec());
        s.start_interval = Some(start);
        s
    }

    fn with_interval(mut r: AdaptiveRule, step: IntervalStep) -> AdaptiveRule {
        r.interval = Some(step);
        r
    }

    fn set_interval(interval: f64) -> DoseAction {
        DoseAction::SetInterval { interval }
    }

    /// An interval-only rule re-issues the running dose and steps one rung per firing;
    /// at the top rung it emits no `SetInterval`, since the interval in force does not
    /// change. Kills the `.min(len - 1)` clamp (index out of bounds) and a step that
    /// re-emits the unchanged top interval.
    #[test]
    fn interval_ladder_lengthens_and_saturates_at_top() {
        let s = interval_spec(
            1,
            &[12.0, 24.0, 48.0],
            12.0,
            vec![with_interval(
                rule(Comparison::Gt, 20.0, AdaptiveAction::Maintain),
                IntervalStep::Lengthen,
            )],
        );
        let out = run(&s, &[30.0, 30.0, 30.0, 30.0]);
        let b = DoseAction::Bolus { amt: 100.0, cmt: 1 };
        assert_eq!(out[0], vec![b.clone(), set_interval(24.0)]);
        assert_eq!(out[1], vec![b.clone(), set_interval(48.0)]);
        assert_eq!(out[2], vec![b.clone()]);
        assert_eq!(out[3], vec![b]);
    }

    /// The rung starts at `start_interval` (24, not the ladder's first rung), so the
    /// first `shorten` moves to 12 and the next one saturates. Kills a rung seeded at
    /// index 0 (the first `shorten` would emit nothing) and the `saturating_sub`.
    #[test]
    fn interval_ladder_starts_at_start_interval_and_saturates_at_bottom() {
        let s = interval_spec(
            1,
            &[12.0, 24.0, 48.0],
            24.0,
            vec![with_interval(
                rule(Comparison::Lt, 5.0, AdaptiveAction::Maintain),
                IntervalStep::Shorten,
            )],
        );
        let out = run(&s, &[1.0, 1.0, 1.0]);
        assert_eq!(out[0], vec![bolus(100.0, 1)[0].clone(), set_interval(12.0)]);
        assert_eq!(out[1], bolus(100.0, 1));
        assert_eq!(out[2], bolus(100.0, 1));
    }

    /// A dose effect and an interval effect compose in one rule: `hold, lengthen`
    /// emits only the interval (the driver logs a hold that consumes its slot), and
    /// `decrease 50%, lengthen` emits the lowered dose and the interval.
    #[test]
    fn dose_and_interval_effects_compose() {
        let s = interval_spec(
            1,
            &[12.0, 24.0, 48.0],
            12.0,
            vec![
                with_interval(
                    rule(Comparison::Gt, 40.0, AdaptiveAction::Hold),
                    IntervalStep::Lengthen,
                ),
                with_interval(
                    rule(
                        Comparison::Gt,
                        20.0,
                        AdaptiveAction::Decrease(DoseStep::Percent(50.0)),
                    ),
                    IntervalStep::Lengthen,
                ),
            ],
        );
        let out = run(&s, &[50.0, 30.0]);
        assert_eq!(out[0], vec![set_interval(24.0)]);
        assert_eq!(out[1], vec![bolus(50.0, 1)[0].clone(), set_interval(48.0)]);
    }

    /// `confirm` gates the interval effect exactly as it gates the dose effect: with
    /// `confirm = 2` the rung moves on every second consecutive match. Kills an
    /// interval step applied on the first match, outside the streak.
    #[test]
    fn confirm_gates_the_interval_effect() {
        let s = interval_spec(
            2,
            &[12.0, 24.0, 48.0],
            12.0,
            vec![with_interval(
                rule(Comparison::Gt, 20.0, AdaptiveAction::Maintain),
                IntervalStep::Lengthen,
            )],
        );
        let out = run(&s, &[30.0, 30.0, 30.0, 30.0]);
        assert_eq!(out[0], bolus(100.0, 1));
        assert_eq!(out[1], vec![bolus(100.0, 1)[0].clone(), set_interval(24.0)]);
        assert_eq!(out[2], bolus(100.0, 1));
        assert_eq!(out[3], vec![bolus(100.0, 1)[0].clone(), set_interval(48.0)]);
    }

    /// The rule label names both effects, and an interval-only rule names just the
    /// interval: it is the ledger's `rule_fired`.
    #[test]
    fn rule_label_names_both_effects() {
        let r = with_interval(
            rule(
                Comparison::Gt,
                20.0,
                AdaptiveAction::Decrease(DoseStep::Percent(25.0)),
            ),
            IntervalStep::Lengthen,
        );
        assert_eq!(r.label(), "signal > 20 : decrease 25%, lengthen");
        let r = with_interval(
            rule(Comparison::Lt, 10.0, AdaptiveAction::Maintain),
            IntervalStep::Shorten,
        );
        assert_eq!(r.label(), "signal < 10 : shorten");
    }

    // ── round_to: the dose grid (#944 PR3) ─────────────────────────────────────────
    //
    // The oracles are hand-computed: with `s = 250` every grid dose is an exact integer,
    // so each expected amount is compared by bits.

    /// `spec` with `round_to = 250`, bolus, confirm 1, and the one rule `rule_action`
    /// firing whenever the signal is below 10.
    fn grid_spec(start: f64, bounds: (f64, f64), action: AdaptiveAction) -> AdaptiveDosingSpec {
        let mut s = spec(
            start,
            AdaptiveRoute::Bolus { cmt: 1 },
            bounds,
            1,
            None,
            vec![rule(Comparison::Lt, 10.0, action)],
        );
        s.round_to = Some(250.0);
        s
    }

    /// Like [`run`], keeping the rule each decision names.
    fn decisions(spec: &AdaptiveDosingSpec, signals: &[f64]) -> Vec<ControllerDecision> {
        let factory = build_adaptive_controller(spec);
        let mut controller = factory(&spec.literal_dose_plan().expect("a block of numbers"));
        let empty_cov: HashMap<String, f64> = HashMap::new();
        signals
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let mut sig = HashMap::new();
                sig.insert(ADAPTIVE_SIGNAL.to_string(), s);
                controller(&ControllerCtx {
                    t: i as f64,
                    state: &[],
                    covariates: &empty_cov,
                    history: &[],
                    decision_index: i,
                    signals: &sig,
                    interval: None,
                })
            })
            .collect()
    }

    fn amt(actions: &[DoseAction]) -> f64 {
        match actions {
            [DoseAction::Bolus { amt, .. }] => *amt,
            other => panic!("expected one bolus, got {other:?}"),
        }
    }

    /// 500 → +30% → 750 → +30% → 1000: each step starts from the dose actually given.
    /// Carrying the unrounded 650 would give 845 → 750 at the second step (#944 audit 1
    /// row 15). The third decision fires no rule and re-issues 1000 as is.
    ///
    /// Mutation: `step_dose` carries `x` and only the emitted amount is rounded.
    #[test]
    fn round_to_carries_the_rounded_dose() {
        let s = grid_spec(
            500.0,
            (250.0, 4000.0),
            AdaptiveAction::Increase(DoseStep::Percent(30.0)),
        );
        let out = run(&s, &[0.0, 0.0, 50.0]);
        assert_eq!(amt(&out[0]).to_bits(), 750.0f64.to_bits(), "{out:?}");
        assert_eq!(amt(&out[1]).to_bits(), 1000.0f64.to_bits(), "{out:?}");
        assert_eq!(amt(&out[2]).to_bits(), 1000.0f64.to_bits(), "{out:?}");
    }

    /// `hi = 1150, s = 250`: a step to 1300 rounds to 1250, then clamps to the aligned
    /// 1000. Clamping first gives 1150, which rounds to 1250, above the written bound
    /// (#944 audit 1 row 13).
    ///
    /// Mutation: clamp to the raw bounds, then round.
    #[test]
    fn round_to_rounds_then_clamps_to_the_aligned_bound() {
        let s = grid_spec(
            1000.0,
            (250.0, 1150.0),
            AdaptiveAction::Increase(DoseStep::Percent(30.0)),
        );
        let out = run(&s, &[0.0]);
        assert_eq!(amt(&out[0]).to_bits(), 1000.0f64.to_bits(), "{out:?}");
    }

    /// The start dose takes the same path as a step: 375 is a tie between 250 and 500
    /// and goes up; 100 with a lower bound of `1e-6` rounds to 0, and the aligned lower
    /// bound, 250, lifts it — a dose, not a hold (#944 audit 1 row 14).
    ///
    /// Mutations: the start dose not placed on the grid; ties sent down; the aligned
    /// lower bound taken as the raw `lo`.
    #[test]
    fn round_to_places_the_start_dose_on_the_grid() {
        let maintain = AdaptiveAction::Maintain;
        let tie = grid_spec(375.0, (1e-6, 4000.0), maintain.clone());
        assert_eq!(amt(&run(&tie, &[50.0])[0]).to_bits(), 500.0f64.to_bits());
        let lifted = grid_spec(100.0, (1e-6, 4000.0), maintain);
        assert_eq!(amt(&run(&lifted, &[50.0])[0]).to_bits(), 250.0f64.to_bits());
    }

    /// 500 +10% is 550, which rounds back to 500: the rule fires and the dose stays —
    /// the same record a step clamped at a bound produces (`rule_fired`, unchanged
    /// `amt`). The next step from the carried 500 is again absorbed, not 605 → 500.
    ///
    /// Mutation: an absorbed step reported as no rule.
    #[test]
    fn round_to_an_absorbed_step_names_the_rule_and_keeps_the_dose() {
        let s = grid_spec(
            500.0,
            (250.0, 4000.0),
            AdaptiveAction::Increase(DoseStep::Percent(10.0)),
        );
        let out = decisions(&s, &[0.0, 0.0]);
        for (i, d) in out.iter().enumerate() {
            assert_eq!(
                amt(&d.actions).to_bits(),
                500.0f64.to_bits(),
                "decision {i}"
            );
            assert!(
                d.rule.is_some(),
                "decision {i}: the rule that fired is named"
            );
        }
    }

    /// A grid that already holds every dose the controller would set leaves the run
    /// unchanged: 1000 → +50% 1500 → −50% 750 → +100% 1500 on `s = 250` with the bounds
    /// on the grid and never reached. The actions equal those without `round_to`, by
    /// bits (#944 D2, second half).
    ///
    /// Mutation: rounding that moves an on-grid value (e.g. a tie bias of a whole step).
    #[test]
    fn round_to_on_a_grid_holding_every_dose_equals_no_round_to() {
        let mut s = spec(
            1000.0,
            AdaptiveRoute::Bolus { cmt: 1 },
            (250.0, 4000.0),
            1,
            None,
            vec![
                rule(
                    Comparison::Lt,
                    10.0,
                    AdaptiveAction::Increase(DoseStep::Percent(50.0)),
                ),
                rule(
                    Comparison::Gt,
                    90.0,
                    AdaptiveAction::Decrease(DoseStep::Percent(50.0)),
                ),
                rule(
                    Comparison::Lt,
                    20.0,
                    AdaptiveAction::Increase(DoseStep::Percent(100.0)),
                ),
            ],
        );
        let signals = [0.0, 95.0, 15.0, 50.0];
        let plain = run(&s, &signals);
        s.round_to = Some(250.0);
        let gridded = run(&s, &signals);
        let bits = |o: &[Vec<DoseAction>]| o.iter().map(|a| amt(a).to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&plain), bits(&gridded), "{plain:?} vs {gridded:?}");
        assert_eq!(
            bits(&plain),
            [1500.0f64, 750.0, 1500.0, 1500.0].map(f64::to_bits),
            "the doses are the ones named above: {plain:?}"
        );
    }
}
