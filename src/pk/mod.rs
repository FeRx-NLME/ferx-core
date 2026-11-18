pub mod absorption;
pub(crate) mod absorption_walk;
pub mod analytical_absorption;
pub mod event_driven;
pub(crate) mod modified_release;
pub mod ode_template;
pub mod one_compartment;
pub(crate) mod sharing_scope;
pub mod three_compartment;
pub(crate) mod topology;
pub mod two_compartment;

use crate::types::{CompiledModel, DoseEvent, PkModel, PkParams, ScalingSpec, Subject};

pub use one_compartment::*;
pub use three_compartment::*;
pub use two_compartment::*;

#[inline]
pub(crate) fn model_uses_time_builtin(model: &CompiledModel) -> bool {
    crate::parser::model_parser::compiled_model_uses_time_builtin(model)
}

/// Does this model depend on model time *anywhere*: the individual-parameter program (as
/// [`model_uses_time_builtin`] alone reports) **or** the `[odes]` right-hand side itself?
///
/// A non-autonomous RHS (`TAD`/`TAFD`/`T`/`TIME`) cannot go through the closed form (its
/// two-time probe pins `tad = 0`, so a `TAD` term was dropped: 8× wrong at 12 h, #1124) and
/// is routed to the event-driven predictor, which places doses at `d.time + lag` (validated
/// by `nonmem_anchor/tad_lag_{A,B}`; dense path: #1110/#1073).
///
/// Measured dense vs event-driven worst relative difference for models moved off the dense
/// path: bare `TAFD` RHS `6e-13`; `init(central)` + `TAFD` `4.8e-11`; `TAD` under lagtime
/// `5.8e-12` (`lag = 3.0`: `5.3e-12`). Not comparable: an SS dose whose RHS reads an absolute clock (`T`/`TIME` 67% off,
/// `TAFD` `NaN`; #1139). `TAD` under `SS=1` reproduces its explicit train to 3.8e-13 and
/// NONMEM to 7.4e-9. Pinned by `rerouting_a_bare_tafd_rhs_does_not_move_the_predictions` and
/// `rerouting_an_init_seeded_rhs_does_not_move_the_predictions`.
///
/// Deliberately whole-class (not narrowed to `!spec.input_rate.is_empty()`) so one engine
/// answers a non-autonomous RHS for value, gradient and `[derived]`. Cost (`ci-fast`, 2-state,
/// 2 doses, `reltol = 1e-8`, event-driven/dense): 8 obs 1.04×, 16 obs 1.15×, 32 obs 1.38×
/// (the event-driven walk restarts the step history per record).
///
/// The `[odes]` half asks the narrow [`OdeRhsProgram::pk_reads_model_time`], which excludes
/// the injected joint-PK-TTE `__chz_<cmt>` hazard lines (#1166): a time-reading baseline
/// hazard must not decline #570's shared solve (measured 1.82×). The SS gradient gates ask
/// `pk_reads_absolute_time()` (#2093), safe because a joint PK-TTE subject with an SS dose is
/// declined first (#1210). The Form C `readout_program` is not folded in (both engines
/// evaluate it under a `ModelTimeGuard`). A joint model is kept out of the closed form by
/// `identify_disposition` (triangular disposition), not by this predicate; pinned by
/// `pk::modified_release::tests::a_joint_pk_tte_model_never_reaches_the_closed_form` (#1209).
#[inline]
pub(crate) fn model_uses_time_anywhere(model: &CompiledModel) -> bool {
    model_uses_time_builtin(model)
        || model.ode_spec.as_ref().is_some_and(|s| {
            s.rhs_program
                .as_ref()
                .is_some_and(|p| p.pk_reads_model_time())
        })
}

#[inline]
fn pk_params_at_time(
    model: &CompiledModel,
    theta: &[f64],
    eta: &[f64],
    covariates: &std::collections::HashMap<String, f64>,
    time: f64,
) -> PkParams {
    // `pk_param_fn` sets the model-time thread-local from `time` itself, so the
    // `TIME` built-in resolves to this event time without a separate wrap.
    (model.pk_param_fn)(theta, eta, covariates, time)
}

/// Whether a closed-form absorption model (transit or inverse-Gaussian) is in the
/// **flip-flop** regime at this subject's individual parameters, where the `*_amt_g`
/// closed form would clamp to `0`: domain `ke < KTR = (n+1)/mtt` (transit) /
/// `ke < 1/(2·MAT·CV²)` (IG) for 1-cpt, the analogous `α < …` for 2-cpt.
///
/// Mirrors those exact guards (2-cpt via `*_2cpt_domain_ok`, which also covers the
/// confluent `α = β` clamp) so prediction and sensitivities route identically. `false`
/// for other models, for invalid parameters (the `0` is a deliberate optimiser
/// penalty) and for `NaN` ones (served as `NaN`, #2134).
pub(crate) fn absorption_flip_flop_at(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> bool {
    let pk = pk_params_at_time(model, theta, eta, &subject.covariates, 0.0);
    let (cl, v1) = (pk.cl(), pk.v());
    if !(cl > 0.0 && v1 > 0.0) {
        return false;
    }
    match model.pk_model {
        PkModel::OneCptTransit => {
            let (n, mtt) = (pk.n_transit(), pk.mtt());
            if !(mtt > 0.0 && n >= 0.0) {
                return false;
            }
            // 1-cpt clamps inline (`one_cpt_transit_amt_g`) at `!(ke < KTR)`; mirror
            // that single-exponential guard directly (no confluent-eigenvalue case).
            let ktr = (n + 1.0) / mtt;
            !(cl / v1 < ktr)
        }
        PkModel::TwoCptTransit => {
            let (n, mtt) = (pk.n_transit(), pk.mtt());
            if !(mtt > 0.0 && n >= 0.0 && pk.q() >= 0.0 && pk.v2() > 0.0) {
                return false;
            }
            // Defer to the closed form's own guard: `None` (it clamps to `0`) covers both
            // flip-flop and confluent `α = β`. Params are valid here, so `None` means "clamps".
            crate::sens::two_cpt::transit_2cpt_domain_ok::<f64>(cl, v1, pk.q(), pk.v2(), n, mtt)
                .is_none()
        }
        PkModel::OneCptIg => {
            let (mat, cv2) = (pk.mat(), pk.cv2());
            if !(mat > 0.0 && cv2 > 0.0) {
                return false;
            }
            // 1-cpt IG clamps inline (`one_cpt_ig_amt_g`) at `!(ke < 1/(2·MAT·CV²))`.
            let abscissa = 1.0 / (2.0 * mat * cv2);
            !(cl / v1 < abscissa)
        }
        PkModel::TwoCptIg => {
            let (mat, cv2) = (pk.mat(), pk.cv2());
            if !(mat > 0.0 && cv2 > 0.0 && pk.q() >= 0.0 && pk.v2() > 0.0) {
                return false;
            }
            // Defer to the IG closed form's own domain guard (flip-flop + confluent),
            // mirroring the transit arm.
            crate::sens::two_cpt::ig_2cpt_domain_ok::<f64>(cl, v1, pk.q(), pk.v2(), mat, cv2)
                .is_none()
        }
        _ => false,
    }
}

/// `subject`'s dose, observation, EVID=2 and EVID=3/4 records whose `CL` or `V`, at
/// `(theta, eta_bsv, kappas)`, is non-positive or non-finite, as `(time, nan)` in time
/// order (#2104). `nan` is `false` where the parameters are degenerate per
/// [`crate::sens::propagate::cl_or_v_degenerate`] (served 0), `true` where one is `NaN`
/// (served `NaN` once a dose has arrived, 0 before). Empty for a model the rule does not
/// govern ([`crate::sens::propagate::cl_v_rule_governs`]).
///
/// Each record is evaluated as the prediction evaluates it (own covariates, `TIME`, and
/// IOV occasion κ). `kappas` is ignored when `model.n_kappa == 0`.
pub(crate) fn invalid_cl_v_records<K: AsRef<[f64]>>(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta_bsv: &[f64],
    kappas: &[K],
) -> Vec<(f64, bool)> {
    if !crate::sens::propagate::cl_v_rule_governs(model.pk_model)
        || model.ode_spec.is_some()
        || model.is_algebraic()
    {
        return Vec::new();
    }
    let params = if model.n_kappa > 0 {
        let layout =
            crate::stats::stacked_layout::StackedLayout::new(eta_bsv.len(), model.n_kappa, subject);
        let combined_for = iov_occasion_eta(&layout, eta_bsv, kappas);
        let mut out = EventPkParams::default();
        fill_iov_record_params(
            model,
            subject,
            theta,
            &combined_for,
            &mut out.dose,
            &mut out.obs,
            &mut out.pk_only,
        );
        // EVID=3/4 rows under their own covariates and occasion — `predict_iov`'s reset
        // snapshot, with its BSV-only fallback for an unlabelled row.
        out.reset = (0..subject.reset_times.len())
            .map(|r| {
                let combined = combined_for(reset_row_occasion(subject, r).unwrap_or(u32::MAX));
                pk_params_at_time(
                    model,
                    theta,
                    &combined,
                    subject.reset_cov(r),
                    subject.param_time_at(subject.reset_times[r]),
                )
            })
            .collect();
        out
    } else {
        compute_event_pk_params(model, subject, theta, eta_bsv)
    };
    // `!(x > 0 && finite)` is true for a `NaN` too — which `cl_or_v_degenerate`, the
    // engines' rule, deliberately is not.
    let valid = |x: f64| x > 0.0 && x.is_finite();
    // The reader stores an EVID=4 row twice (in `doses`, flagged `evid4`, and in
    // `reset_times`), so its reset entry is skipped. A separate EVID=3 row co-timed with a
    // dose is a second record and is kept.
    let evid4_twin = |t: f64| subject.doses.iter().any(|d| d.evid4 && d.time == t);
    let mut records: Vec<(f64, bool)> = subject
        .doses
        .iter()
        .map(|d| d.time)
        .zip(&params.dose)
        .chain(subject.obs_times.iter().copied().zip(&params.obs))
        .chain(subject.pk_only_times.iter().copied().zip(&params.pk_only))
        .chain(
            subject
                .reset_times
                .iter()
                .copied()
                .zip(&params.reset)
                .filter(|&(t, _)| !evid4_twin(t)),
        )
        .filter(|(_, p)| !(valid(p.cl()) && valid(p.v())))
        .map(|(t, p)| {
            let nan = !crate::sens::propagate::cl_or_v_degenerate(p.cl(), p.v());
            (t, nan)
        })
        .collect();
    records.sort_by(|a, b| a.0.total_cmp(&b.0));
    records
}

/// The model to evaluate this subject with at `(theta, eta)`: the structural
/// [`CompiledModel::effective_for`] reroute **plus** a parameter-dependent flip-flop
/// reroute to the ODE twin when a transit / IG closed form would clamp to `0`. Prediction
/// and sensitivity dispatchers both route through here at the same `(theta, eta)`.
///
/// A flip-flop model with no twin is rejected up front by
/// [`crate::api::check_absorption_flip_flop_no_twin`]; see
/// [`crate::types::CompiledModel::absorption_ode_equivalent`] for what "no twin" covers
/// (an open-ended set, #1008).
pub(crate) fn effective_model_for_eval<'a>(
    model: &'a CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> &'a CompiledModel {
    // A TV-covariate subject of a closed-form transit/IG model (#1560): the absorption
    // walk serves it unless a pair leaves the tilting domain (`walk_domain`).
    if absorption_walk::walk_eligible(model, subject) {
        let mut ev = EventPkParams::default();
        compute_event_pk_params_into(model, subject, theta, eta, &mut ev);
        return absorption_walk_or_twin(model, subject, &ev);
    }
    let structural = model.effective_for(subject);
    // No twin to route to (a non-absorption model, or a transit/IG that has none — declined by
    // the desugar, or built and rejected by its own parse, #1008): nothing parameter-dependent
    // to decide.
    let Some(twin) = &model.absorption_ode_equivalent else {
        return structural;
    };
    // Keep the structural model when already rerouted to a twin, or when not flip-flop.
    if structural.ode_spec.is_some() || !absorption_flip_flop_at(model, subject, theta, eta) {
        return structural;
    }
    twin.built()
}

/// The IOV peer of [`effective_model_for_eval`] at `(θ, [η_bsv, κ₁ … κ_K])` (occasions in
/// [`crate::stats::likelihood::iov_occasion_groups`] order): a walk-eligible transit/IG
/// subject stays on the closed form when the walk serves it, as [`predict_iov`] decides;
/// everything else is [`CompiledModel::effective_for`] (#1560).
pub(crate) fn effective_model_for_eval_iov<'a>(
    model: &'a CompiledModel,
    subject: &Subject,
    theta: &[f64],
    stacked_eta: &[f64],
) -> &'a CompiledModel {
    if !absorption_walk::walk_eligible(model, subject) {
        return model.effective_for(subject);
    }
    let n_eta = model.n_eta.min(stacked_eta.len());
    let layout = crate::stats::stacked_layout::StackedLayout::of(model, subject);
    let kappas = layout.kappa_chunks(stacked_eta);
    let combined_for = iov_occasion_eta(&layout, &stacked_eta[..n_eta], &kappas);
    let mut ev = EventPkParams::default();
    fill_iov_record_params(
        model,
        subject,
        theta,
        &combined_for,
        &mut ev.dose,
        &mut ev.obs,
        &mut ev.pk_only,
    );
    absorption_walk_or_twin(model, subject, &ev)
}

/// `model` when the absorption walk serves `subject` at these per-event parameters, else
/// the twin. The caller has established [`absorption_walk::walk_eligible`], which includes
/// a twin being present.
fn absorption_walk_or_twin<'a>(
    model: &'a CompiledModel,
    subject: &Subject,
    ev: &EventPkParams,
) -> &'a CompiledModel {
    if absorption_walk::walk_serves(model.pk_model, subject, &ev.dose, &ev.obs, &ev.pk_only) {
        return model;
    }
    model
        .absorption_ode_equivalent
        .as_ref()
        .map_or(model, |twin| twin.built())
}

/// `η` per record for an IOV subject: `[η_bsv, κ_g]` for occasion `occ` in group `g`
/// ([`crate::stats::likelihood::iov_occasion_groups`] order), `κ = 0` for an occasion
/// with no group — an occasion-less caller asks for `u32::MAX` and gets that.
fn iov_occasion_eta<'a, K: AsRef<[f64]>>(
    layout: &'a crate::stats::stacked_layout::StackedLayout,
    eta_bsv: &'a [f64],
    kappas: &'a [K],
) -> impl Fn(u32) -> Vec<f64> + 'a {
    let n_kappa = layout.n_kappa();
    move |occ_id: u32| -> Vec<f64> {
        let mut c = Vec::with_capacity(eta_bsv.len() + n_kappa);
        c.extend_from_slice(eta_bsv);
        match layout.group_of(occ_id) {
            Some(k) if k < kappas.len() => c.extend_from_slice(kappas[k].as_ref()),
            _ => c.extend(std::iter::repeat_n(0.0, n_kappa)),
        }
        c
    }
}

/// The per-record parameters of an IOV subject (pk-only rows under their own occasion,
/// [`pk_only_row_occasion`], #1876) as [`predict_iov`] evaluates them; shared with
/// [`effective_model_for_eval_iov`] so the route uses the same parameters.
fn fill_iov_record_params(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    combined_for: &dyn Fn(u32) -> Vec<f64>,
    dose: &mut Vec<PkParams>,
    obs: &mut Vec<PkParams>,
    pk_only: &mut Vec<PkParams>,
) {
    // Consecutive records of one occasion usually share every input of the (pure)
    // parameter program, so the last result is reused bit for bit (#1564).
    let mut memo = RecordParamMemo::new(model);
    // Each record's parameters are cloned straight from the memo into its slot: one copy
    // of the (1 KB) `PkParams` per record, not a copy out of the memo and another into
    // the vector (#2549 W1).
    let mut push = |out: &mut Vec<PkParams>,
                    occ: u32,
                    cov: &std::collections::HashMap<String, f64>,
                    time: f64| {
        let p = memo.get_ref(occ, cov, time, || {
            pk_params_at_time(model, theta, &combined_for(occ), cov, time)
        });
        out.extend_from_slice(std::slice::from_ref(p));
    };
    dose.reserve_exact(subject.doses.len());
    for d in 0..subject.doses.len() {
        // `u32::MAX` names no group, so an unlabelled dose gets κ = 0 (#1153).
        let occ = dose_row_occasion(subject, d).unwrap_or(u32::MAX);
        push(dose, occ, subject.dose_cov(d), subject.dose_param_time(d));
    }
    inherit_parent_dose_attrs(model, subject, dose);
    obs.reserve_exact(subject.obs_times.len());
    for j in 0..subject.obs_times.len() {
        let occ = subject.occasions.get(j).copied().unwrap_or(0);
        push(obs, occ, subject.obs_cov(j), subject.param_time_obs(j));
    }
    pk_only.reserve_exact(subject.pk_only_times.len());
    for m in 0..subject.pk_only_times.len() {
        // `u32::MAX` names no group, so an unresolvable row gets κ = 0 (#1876).
        let occ = pk_only_row_occasion(subject, m).unwrap_or(u32::MAX);
        push(
            pk_only,
            occ,
            subject.pk_only_cov(m),
            subject.param_time_pk_only(m),
        );
    }
}

#[cfg(test)]
thread_local! {
    /// Test switch: predict IOV subjects from per-record copies even where the record table
    /// applies, for the parity test that pins the two against each other.
    pub(crate) static IOV_TABLE_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[inline]
fn iov_table_off() -> bool {
    #[cfg(test)]
    {
        IOV_TABLE_OFF.with(|c| c.get())
    }
    #[cfg(not(test))]
    {
        false
    }
}

/// [`fill_iov_record_params`] into a table of distinct snapshots and one index per record
/// (#2549). An `ADDL`-implied dose that inherits its parent's attributes gets its own
/// entry first (copy-on-write).
fn fill_iov_record_table(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    combined_for: &dyn Fn(u32) -> Vec<f64>,
    table: &mut Vec<PkParams>,
    dose_ix: &mut Vec<u32>,
    obs_ix: &mut Vec<u32>,
    pk_only_ix: &mut Vec<u32>,
) {
    let mut memo = RecordParamMemo::new(model);
    // The table entry holding the memo's evaluation: never one the parent-dose edits below
    // write (they edit copies), so every record the memo matches reads it.
    let mut held: Option<u32> = None;
    let mut push = |table: &mut Vec<PkParams>,
                    held: &mut Option<u32>,
                    out: &mut Vec<u32>,
                    occ: u32,
                    cov: &std::collections::HashMap<String, f64>,
                    time: f64| {
        let hit = memo.same(occ, cov, time);
        let i = match *held {
            Some(i) if hit => i,
            _ => {
                table.push(pk_params_at_time(
                    model,
                    theta,
                    &combined_for(occ),
                    cov,
                    time,
                ));
                (table.len() - 1) as u32
            }
        };
        *held = Some(i);
        out.push(i);
    };
    for d in 0..subject.doses.len() {
        let occ = dose_row_occasion(subject, d).unwrap_or(u32::MAX);
        push(
            table,
            &mut held,
            dose_ix,
            occ,
            subject.dose_cov(d),
            subject.dose_param_time(d),
        );
    }
    let mut own = vec![false; subject.doses.len()];
    for (k, p, slot) in parent_dose_attr_copies(model, subject) {
        if !own[k] {
            let copy = table[dose_ix[k] as usize];
            table.push(copy);
            dose_ix[k] = (table.len() - 1) as u32;
            own[k] = true;
        }
        let v = table[dose_ix[p] as usize].values[slot];
        table[dose_ix[k] as usize].values[slot] = v;
    }
    for j in 0..subject.obs_times.len() {
        let occ = subject.occasions.get(j).copied().unwrap_or(0);
        push(
            table,
            &mut held,
            obs_ix,
            occ,
            subject.obs_cov(j),
            subject.param_time_obs(j),
        );
    }
    for m in 0..subject.pk_only_times.len() {
        let occ = pk_only_row_occasion(subject, m).unwrap_or(u32::MAX);
        push(
            table,
            &mut held,
            pk_only_ix,
            occ,
            subject.pk_only_cov(m),
            subject.param_time_pk_only(m),
        );
    }
}

/// The last per-record parameter evaluation of [`fill_iov_record_params`] and its inputs:
/// the occasion (which fixes `[η, κ]` within one call), the covariate map by address, and the
/// time unless the parameter program provably does not read `TIME`.
struct RecordParamMemo {
    time_free: bool,
    last: Option<(u32, *const std::collections::HashMap<String, f64>, u64)>,
    value: PkParams,
}

impl RecordParamMemo {
    /// `time_free` trusts the parsed program's `TIME` flag ([`model_uses_time_builtin`]),
    /// as routing does. A model with no parsed program never shares a time-free entry.
    fn new(model: &CompiledModel) -> Self {
        RecordParamMemo {
            time_free: model.indiv_param_partials.indiv_param_program.is_some()
                && !model_uses_time_builtin(model),
            last: None,
            value: PkParams::default(),
        }
    }

    /// The parameters for these inputs: the last evaluation's when every input matches,
    /// else a fresh `eval()`, kept for the next record.
    fn get_ref(
        &mut self,
        occ: u32,
        cov: &std::collections::HashMap<String, f64>,
        time: f64,
        eval: impl FnOnce() -> PkParams,
    ) -> &PkParams {
        if !self.same(occ, cov, time) {
            self.value = eval();
        }
        &self.value
    }

    /// Whether these inputs are the last call's, which they become.
    fn same(&mut self, occ: u32, cov: &std::collections::HashMap<String, f64>, time: f64) -> bool {
        let key = (
            occ,
            cov as *const _,
            if self.time_free { 0 } else { time.to_bits() },
        );
        self.last.replace(key) == Some(key)
    }
}

/// Divide each prediction in-place by the scale derived from `model.scaling`
/// (**divisive**: `obs_scale = 1000` maps mg/L → mg/mL). `ScalingSpec::None` is a no-op.
/// `ExpressionScale` is evaluated on a subject-static `pk_param_fn` snapshot. Form C
/// (`y = <expr>`) does not come through here.
#[inline]
pub fn apply_scaling(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    preds: &mut [f64],
) {
    // Fast path: no scaling skips both the pk evaluation and the per-obs
    // array allocation. Keeps the hot loop allocation-free for the
    // overwhelming majority of models.
    if matches!(model.scaling, ScalingSpec::None) {
        return;
    }
    let scales = obs_scale_divisors(model, subject, theta, eta, &subject.obs_cmts);
    for (i, pred) in preds.iter_mut().enumerate() {
        *pred = divide_by_obs_scale(*pred, scales.get(i).copied().unwrap_or(f64::NAN));
    }
}

/// The `[scaling]` Form A/B divisor for each entry of `cmts`, under the subject-static
/// snapshot [`apply_scaling`] divides by (`pk_param_fn` at baseline covariates, `t = 0`).
/// Shared with the adaptive driver (#2129). Invalid scales come back `NaN` (see
/// [`ScalingSpec::build_obs_scale_array`]).
pub(crate) fn obs_scale_divisors(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    cmts: &[usize],
) -> Vec<f64> {
    // Shared `ScalingSpec::build_obs_scale_array` keeps FD and AD (`inner_optimizer.rs`)
    // identical. Only evaluate pk when an ExpressionScale actually needs it.
    let pk_owned;
    let pk_ref: &PkParams = if model.scaling.needs_pk_eval() {
        // Subject-static scale snapshot (the divisive scale is applied to the
        // whole prediction vector): typical values at t=0.
        pk_owned = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);
        &pk_owned
    } else {
        static DEFAULT_PK: PkParams = PkParams {
            values: [0.0; crate::types::MAX_PK_PARAMS],
        };
        &DEFAULT_PK
    };
    model
        .scaling
        .build_obs_scale_array(theta, eta, &subject.covariates, pk_ref, cmts)
}

/// Divide one prediction by its `[scaling]` divisor, or `NaN` when the divisor is
/// not a finite positive number. [`ScalingSpec::build_obs_scale_array`] already
/// encodes invalid scales as `NaN`; the check here also covers a missing entry.
#[inline]
pub(crate) fn divide_by_obs_scale(pred: f64, scale: f64) -> f64 {
    if scale.is_finite() && scale > 0.0 {
        pred / scale
    } else {
        f64::NAN
    }
}

/// Apply an analytic Form C output readout (`[scaling] y = <expr>`, #650), **replacing**
/// the central concentration in `preds` with the readout, per observation.
///
/// The readout sees compartment **amounts**: central = `concentration × V`; the oral depot
/// amount (only if referenced) comes from [`analytical_state_at_times`] (slot 0).
/// Covariates use the per-observation snapshot; on a TV-covariate or `TIME` model the
/// individual parameters are re-evaluated per observation (NONMEM's per-record `$ERROR`,
/// matching `record_analytic_events`). No-op unless `model.analytic_readout` is set.
pub fn apply_analytic_readout(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    preds: &mut [f64],
) {
    let Some(ar) = &model.analytic_readout else {
        return;
    };
    let n_states = ar.state_names.len();
    // Canonical order is `["central"]` (IV) or `["depot", "central"]` (oral): central is
    // the last slot. A compartment-free model (#811) has no amounts at all.
    let central_slot = n_states.checked_sub(1);
    let need_depot = n_states > 1 && ar.state_names[0] == "depot";

    // Re-evaluate the readout's individual parameters per observation whenever a
    // covariate (or the `TIME` builtin) can move them between rows; otherwise a
    // single subject-static snapshot suffices.
    let per_obs = subject.has_tv_covariates()
        || crate::parser::model_parser::compiled_model_uses_time_builtin(model);
    let static_pk = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);

    // Depot amounts via superposition state reconstruction (slot 0). Reset subjects with
    // a depot-referencing readout are rejected up front (`reject_depot_readout_with_resets`).
    let depot_amts: Option<Vec<f64>> = if need_depot {
        let states =
            analytical_state_at_times(model.pk_model, subject, &static_pk, &subject.obs_times);
        Some(
            states
                .iter()
                .map(|s| s.first().copied().unwrap_or(0.0))
                .collect(),
        )
    } else {
        None
    };
    let mut state = vec![0.0_f64; n_states];
    for (i, pred) in preds.iter_mut().enumerate() {
        let obs_pk;
        let pk_i: &PkParams = if per_obs {
            obs_pk = (model.pk_param_fn)(theta, eta, subject.obs_cov(i), subject.param_time_obs(i));
            &obs_pk
        } else {
            &static_pk
        };
        // Central AMOUNT = concentration × V (concentration already carries the
        // init impulse and per-event/SS dynamics). No-op for a compartment-free
        // model, whose `state[]` is empty (#811).
        if let Some(c) = central_slot {
            state[c] = *pred * pk_i.v();
        }
        if let Some(d) = &depot_amts {
            state[0] = d.get(i).copied().unwrap_or(0.0);
        }
        let cov = subject.obs_cov(i);
        let obs_cmt = subject.obs_cmts.get(i).copied().unwrap_or(0);
        // `OdeReadout::eval` enters this observation's time as the `TIME` guard (#1028).
        // `readout_time` is the raw data-file clock (the `$ERROR` convention).
        let t = subject.readout_time(i);
        *pred = ar
            .readout
            .eval(&state, &pk_i.values, theta, eta, cov, obs_cmt, t);
    }
}

/// Layer analytical initial-compartment amounts onto `preds` (#521): for a linear
/// disposition, `A₀` in compartment `c` at t=0 is the impulse response of an F-bypassed
/// bolus. `A₀` is evaluated once per subject. Called **before** `[scaling]` and the
/// log-transform, matching the ODE path. No-op for ODE models and without
/// `[initial_conditions]`. `preds` is parallel to `subject.obs_times`.
pub fn add_analytical_init(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    preds: &mut [f64],
) {
    if model.analytical_init.is_empty() || model.ode_spec.is_some() {
        return;
    }
    // Non-IOV path: one subject-static PK snapshot drives both the baseline
    // amount and its decay kernel. The baseline is seeded at the record start
    // (t=0), so the `TIME` built-in resolves there.
    let pk = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);
    add_analytical_init_with(model, subject, theta, eta, &pk, None, preds);
}

/// Core of [`add_analytical_init`]. `amount_pk` evaluates `A₀` (BSV snapshot);
/// `obs_decay_pk`, when `Some`, supplies per-observation params for the decay kernel (IOV,
/// #521 review), else `amount_pk` drives it. An EVID=3/4 reset zeros the baseline, so it
/// contributes only to observations strictly before the first reset (internal clock).
fn add_analytical_init_with(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    amount_pk: &PkParams,
    obs_decay_pk: Option<&[PkParams]>,
    preds: &mut [f64],
) {
    let first_reset = subject
        .reset_times
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    for init in &model.analytical_init {
        let a0 = (init.amount_fn)(theta, eta, &subject.covariates, amount_pk);
        if a0 == 0.0 || !a0.is_finite() {
            continue;
        }
        for (i, &t) in subject.obs_times.iter().enumerate() {
            // The initial amount is laid down at the subject's time origin
            // (t=0); a pre-origin observation sees none of it, and a reset
            // wipes it for every observation at/after the reset time.
            if t < 0.0 || t >= first_reset {
                continue;
            }
            let decay_pk = match obs_decay_pk {
                Some(ps) => ps.get(i).unwrap_or(amount_pk),
                None => amount_pk,
            };
            if let Some(pred) = preds.get_mut(i) {
                *pred += analytical_init_concentration(model.pk_model, init.cmt, a0, t, decay_pk);
            }
        }
    }
}

/// Closed-form central concentration at `t` from an initial amount `a0` in compartment
/// `cmt` (1-based) at t=0, with `F = 1`. Supports central and (oral) depot; peripheral
/// inits are rejected at parse time, so any other `cmt` returns 0.
fn analytical_init_concentration(
    pk_model: PkModel,
    cmt: usize,
    a0: f64,
    t: f64,
    p: &PkParams,
) -> f64 {
    // Delegate to the generic form at `T = f64`; the single formula dispatch
    // lives there so the Dual2/Dual2 sensitivity path (#524) cannot drift from
    // the f64 prediction.
    analytical_init_concentration_g::<f64>(
        pk_model,
        cmt,
        a0,
        t,
        p.cl(),
        p.v(),
        p.q(),
        p.v2(),
        p.ka(),
        p.q3(),
        p.v3(),
    )
}

/// Generic form of [`analytical_init_concentration`] over [`PkNum`]: in `Dual2` it gives
/// the exact `∂C/∂A₀`, `∂C/∂(CL,V,…)` for the analytic provider (#524). `F = 1`; returns 0
/// for unsupported compartments.
#[allow(clippy::too_many_arguments)]
pub(crate) fn analytical_init_concentration_g<T: crate::sens::num::PkNum>(
    pk_model: PkModel,
    cmt: usize,
    a0: T,
    t: T,
    cl: T,
    v: T,
    q: T,
    v2: T,
    ka: T,
    q3: T,
    v3: T,
) -> T {
    use crate::sens::one_cpt::{one_cpt_iv_bolus_amt_g, one_cpt_oral_amt_g};
    use crate::sens::three_cpt::{three_cpt_iv_bolus_amt_g, three_cpt_oral_amt_g};
    use crate::sens::two_cpt::{two_cpt_iv_bolus_amt_g, two_cpt_oral_amt_g};

    let one = T::from_f64(1.0);
    // Central is cmt 2 for oral models (depot is cmt 1), cmt 1 for IV models.
    let central = if pk_model.is_oral() { 2 } else { 1 };

    if cmt == central {
        // IV-bolus impulse directly into the central compartment.
        match pk_model {
            PkModel::OneCptIv
            | PkModel::OneCptOral
            | PkModel::OneCptTransit
            | PkModel::OneCptIg => one_cpt_iv_bolus_amt_g(a0, t, cl, v),
            PkModel::TwoCptIv
            | PkModel::TwoCptOral
            | PkModel::TwoCptTransit
            | PkModel::TwoCptIg => two_cpt_iv_bolus_amt_g(a0, t, cl, v, q, v2),
            PkModel::ThreeCptIv | PkModel::ThreeCptOral => {
                three_cpt_iv_bolus_amt_g(a0, t, cl, v, q, v2, q3, v3)
            }
        }
    } else if pk_model.is_oral() && cmt == 1 {
        // Pre-loaded depot amount: first-order absorption into central, F=1.
        match pk_model {
            PkModel::OneCptOral => one_cpt_oral_amt_g(a0, t, cl, v, ka, one),
            PkModel::TwoCptOral => two_cpt_oral_amt_g(a0, t, cl, v, q, v2, ka, one),
            PkModel::ThreeCptOral => three_cpt_oral_amt_g(a0, t, cl, v, q, v2, q3, v3, ka, one),
            _ => T::from_f64(0.0),
        }
    } else {
        T::from_f64(0.0)
    }
}

/// Validate that every observed CMT has an entry in the model's `ScalingSpec::PerCmt` map
/// (or nested `OdeReadout::PerCmt`). Called once at the top of `fit()`; the error names
/// every missing CMT. `Ok(())` for non-PerCmt scaling.
pub fn validate_per_cmt_scaling(model: &CompiledModel, subjects: &[Subject]) -> Result<(), String> {
    use std::collections::BTreeSet;

    // Collect every CMT that has at least one observation in the population.
    let mut observed_cmts: BTreeSet<usize> = BTreeSet::new();
    for subj in subjects {
        for &cmt in &subj.obs_cmts {
            observed_cmts.insert(cmt);
        }
    }

    // Check ScalingSpec::PerCmt coverage (Forms A/B).
    if let ScalingSpec::PerCmt(map) = &model.scaling {
        let missing: Vec<usize> = observed_cmts
            .iter()
            .copied()
            .filter(|c| !map.contains_key(c))
            .collect();
        if !missing.is_empty() {
            return Err(format!(
                "[scaling]: per-CMT scaling is missing entries for observed CMTs {:?}. \
                 Every observed CMT must have an `obs_scale[CMT=N]` (or `y[CMT=N]` for ODE) entry.",
                missing
            ));
        }
    }

    // Check OdeReadout::PerCmt coverage (Form C per-CMT).
    if let Some(ref ode) = model.ode_spec {
        if let crate::ode::OdeReadout::PerCmt(map) = &ode.readout {
            let missing: Vec<usize> = observed_cmts
                .iter()
                .copied()
                .filter(|c| !map.contains_key(c))
                .collect();
            if !missing.is_empty() {
                return Err(format!(
                    "[scaling]: per-CMT `y[CMT=N]` Form C is missing entries for observed CMTs {:?}. \
                     Every observed CMT must have a `y[CMT=N]` entry.",
                    missing
                ));
            }
        }
    }

    Ok(())
}

/// Per-event PK parameter snapshots for one subject. Without time-varying covariates all
/// entries are equal (one subject-static evaluation).
///
/// **All four vectors are filled on both arms**, and each field doc is an invariant: a
/// short `pk_only` broke the walkers' length assert (#1124 review) and an empty `reset`
/// lost the `init(...)` re-seed (#1133). When adding a field, fill it on both arms.
///
/// Hot loops reuse one buffer per subject ([`Self::with_capacity_for`] +
/// [`compute_event_pk_params_into`]).
#[derive(Debug, Clone, Default)]
pub struct EventPkParams {
    /// PK params at each dose event time, parallel to `subject.doses`.
    pub dose: Vec<PkParams>,
    /// PK params at each observation event time, parallel to `subject.obs_times`.
    pub obs: Vec<PkParams>,
    /// PK params at each EVID=2 event time, parallel to `subject.pk_only_times`. Empty
    /// **only** when the subject has no pk-only events — never short.
    pub pk_only: Vec<PkParams>,
    /// PK params at each EVID=3/4 reset time, parallel to `subject.reset_times`; re-seeds
    /// `[odes] init(...)` on reset (#1133). Empty when the subject has no resets.
    pub reset: Vec<PkParams>,
}

impl EventPkParams {
    /// Allocate empty `Vec`s with capacity for `subject`'s event timeline;
    /// [`compute_event_pk_params_into`] resizes and overwrites them.
    pub fn with_capacity_for(subject: &Subject) -> Self {
        Self {
            dose: Vec::with_capacity(subject.doses.len()),
            obs: Vec::with_capacity(subject.obs_times.len()),
            pk_only: Vec::with_capacity(subject.pk_only_times.len()),
            reset: Vec::with_capacity(subject.reset_times.len()),
        }
    }
}

/// Materialise per-event PK parameters by evaluating `model.pk_param_fn` at each event's
/// LOCF covariate snapshot (once, cloned, for a parameter-static subject). Allocates; hot
/// loops use [`compute_event_pk_params_into`] with a reused [`EventPkParams`].
pub fn compute_event_pk_params(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> EventPkParams {
    let mut out = EventPkParams::with_capacity_for(subject);
    compute_event_pk_params_into(model, subject, theta, eta, &mut out);
    out
}

/// Whether a subject needs **per-event** PK rather than one t=0 snapshot: a covariate
/// changes on a dose/obs row (`has_tv_covariates`) or only on an EVID=2 row
/// (`pk_only_covariates`), or the model reads `TIME`. Single source of truth (#700) for
/// [`compute_event_pk_params_into`] and the adaptive driver's `event_pk` / `auc_target`.
pub(crate) fn subject_needs_per_event_pk(model: &CompiledModel, subject: &Subject) -> bool {
    subject.has_tv_covariates()
        || !subject.pk_only_covariates.is_empty()
        // EVID=3/4 rows carry their own snapshot (#1133); a hand-assembled subject (FREM,
        // R glue, tests) can populate `reset_covariates` alone.
        || !subject.reset_covariates.is_empty()
        || model_uses_time_builtin(model)
}

/// Whether [`compute_predictions_with_states`] reports **approximate** compartment states
/// for this subject on an ODE model (#1855): the event-driven state pass holds the
/// disposition at the first observation's snapshot, which is inexact under a time-varying
/// covariate or a `TIME`-reading parameter program (resets and a time-reading RHS are
/// exact). `model` is the **effective** model ([`effective_model_for_eval`]).
pub(crate) fn ode_cmt_states_approximate(model: &CompiledModel, subject: &Subject) -> bool {
    model.ode_spec.is_some() && (subject.has_tv_covariates() || model_uses_time_builtin(model))
}

/// The occasion label for an EVID=3/4 **reset row's** `$PK` on the `OCC`-column IOV path
/// (#1133, #1153): the row's own `OCC`, or `None` (κ = 0) when the subject stores no reset
/// labels. Measured in `nonmem_anchor/reset_init_snapshot_J.ctl` (`A_0` seeds under the
/// reset row's occasion: 42.0, not 14.0). No guess from a neighbouring record: a subject
/// read from data has every label or none. Takes the reset's **index**, not its time, so
/// close resets cannot alias.
pub(crate) fn reset_row_occasion(subject: &Subject, r: usize) -> Option<u32> {
    subject.reset_occasions.get(r).copied()
}

/// The occasion label for an EVID=2 or EVID=0/MDV=1 **pk-only row's** `$PK` on the
/// `OCC`-column IOV path (#1876): the row's own `OCC` (measured in
/// `nonmem_anchor/iov_pk_only_*_X.ctl`), or `None` (κ = 0) when the subject stores no
/// pk-only labels, as in [`reset_row_occasion`].
pub(crate) fn pk_only_row_occasion(subject: &Subject, m: usize) -> Option<u32> {
    subject.pk_only_occasions.get(m).copied()
}

/// The occasion label for dose `d`'s `$PK` on the IOV path (#2065), read by
/// [`predict_iov`] and both `sens/` walks: the dose's own label, or `None` (κ = 0) when
/// `dose_occasions` is short (#1153).
pub(crate) fn dose_row_occasion(subject: &Subject, d: usize) -> Option<u32> {
    subject.dose_occasions.get(d).copied()
}

/// The κ group of dose `d` ([`dose_row_occasion`] through `occ_to_k`), as the `sens/`
/// walks seed it. `None` is the κ = 0 dose, where production's `combined_for` lands on
/// a miss — the treatment [`pk_only_row_group`] gives an ungrouped pk-only row.
pub(crate) fn dose_row_group(
    subject: &Subject,
    d: usize,
    occ_to_k: &std::collections::HashMap<u32, usize>,
) -> Option<usize> {
    dose_row_occasion(subject, d).and_then(|occ| occ_to_k.get(&occ).copied())
}

/// The κ group of pk-only row `m` ([`pk_only_row_occasion`] through `occ_to_k`); `None`
/// is the κ = 0 row.
pub(crate) fn pk_only_row_group(
    subject: &Subject,
    m: usize,
    occ_to_k: &std::collections::HashMap<u32, usize>,
) -> Option<usize> {
    pk_only_row_occasion(subject, m).and_then(|occ| occ_to_k.get(&occ).copied())
}

/// The κ group of reset row `r` ([`reset_row_occasion`] through `occ_to_k`), as the
/// `sens/` providers seed it — [`pk_only_row_group`]'s rule for a reset row.
pub(crate) fn reset_row_group(
    subject: &Subject,
    r: usize,
    occ_to_k: &std::collections::HashMap<u32, usize>,
) -> Option<usize> {
    reset_row_occasion(subject, r).and_then(|occ| occ_to_k.get(&occ).copied())
}

/// Fill `out` with one `$PK` snapshot per dose — the single spelling of where the engine
/// resolves a dose attribute, shared by [`compute_event_pk_params_into`] and
/// [`compute_dose_pk_params_into`]. `constant` is `Some(p)` for a parameter-static
/// subject so the caller's snapshot is reused (one `$PK` call less per evaluation).
#[inline]
fn fill_dose_pk_params(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    constant: Option<PkParams>,
    out: &mut Vec<PkParams>,
) {
    out.clear();
    match constant {
        Some(p) => out.resize(subject.doses.len(), p),
        None => {
            let doses: Vec<_> = dose_pk_inputs(subject).collect();
            fill_per_record_pk_params(
                model,
                subject,
                sharing_scope::ShareSite::Dose,
                theta,
                eta,
                &[&doses],
                &mut [out],
            );
            inherit_parent_dose_attrs(model, subject, out);
        }
    }
}

/// One record's `$PK` inputs: its covariate snapshot and its parameter time.
type PkInputs<'a> = (&'a std::collections::HashMap<String, f64>, f64);

/// The `$PK` inputs of each dose record, in order. This and its three siblings are the
/// one spelling of which snapshot and time a record's `$PK` reads, for the f64
/// materializer and for [`record_pk_inputs`].
fn dose_pk_inputs(subject: &Subject) -> impl Iterator<Item = PkInputs<'_>> {
    (0..subject.doses.len()).map(|k| (subject.dose_cov(k), subject.dose_param_time(k)))
}

/// The `$PK` inputs of each observation record, in order.
fn obs_pk_inputs(subject: &Subject) -> impl Iterator<Item = PkInputs<'_>> {
    (0..subject.obs_times.len()).map(|j| (subject.obs_cov(j), subject.param_time_obs(j)))
}

/// The `$PK` inputs of each PK-only (`EVID=2`) record, in order.
fn pk_only_pk_inputs(subject: &Subject) -> impl Iterator<Item = PkInputs<'_>> {
    (0..subject.pk_only_times.len())
        .map(|m| (subject.pk_only_cov(m), subject.param_time_pk_only(m)))
}

/// The `$PK` inputs of each reset (`EVID=3/4`) record, in order.
fn reset_pk_inputs(subject: &Subject) -> impl Iterator<Item = PkInputs<'_>> {
    (0..subject.reset_times.len()).map(|r| {
        (
            subject.reset_cov(r),
            subject.param_time_at(subject.reset_times[r]),
        )
    })
}

/// Every dose, observation and PK-only record of `subject`, in that order, as
/// `(occasion label, covariate snapshot, parameter time)` — the records the closed-form
/// recorder evaluates `$PK` over. An unlabelled dose/PK-only row is `u32::MAX`, an
/// unlabelled observation 0.
pub(crate) fn record_pk_inputs(
    subject: &Subject,
) -> Vec<(u32, &std::collections::HashMap<String, f64>, f64)> {
    let dose_occ =
        (0..subject.doses.len()).map(|k| dose_row_occasion(subject, k).unwrap_or(u32::MAX));
    let obs_occ =
        (0..subject.obs_times.len()).map(|j| subject.occasions.get(j).copied().unwrap_or(0));
    let pk_only_occ = (0..subject.pk_only_times.len())
        .map(|m| pk_only_row_occasion(subject, m).unwrap_or(u32::MAX));
    dose_occ
        .zip(dose_pk_inputs(subject))
        .chain(obs_occ.zip(obs_pk_inputs(subject)))
        .chain(pk_only_occ.zip(pk_only_pk_inputs(subject)))
        .map(|(occ, (cov, t))| (occ, cov, t))
        .collect()
}

/// For each record, the earliest record whose `$PK` inputs are bit-identical to its own
/// (or itself); `None` when the model shares nothing. Inputs are the κ `group` and every
/// referenced covariate, compared by bits (as the prefix cache does: `NaN` matches itself,
/// `-0.0 ≠ 0.0`).
///
/// A `TIME`-reading program shares nothing (sharing found almost nothing and was 9–13%
/// slower, #2543 review r1). A model without a parsed program shares nothing either; a
/// parsed model is trusted to have the `pk_param_fn` its program describes.
pub(crate) fn first_record_with_same_pk_inputs<'a>(
    model: &CompiledModel,
    records: impl IntoIterator<Item = (Option<usize>, &'a std::collections::HashMap<String, f64>)>,
) -> Option<Vec<usize>> {
    if model.indiv_param_partials.indiv_param_program.is_none()
        || crate::parser::model_parser::compiled_model_uses_time_builtin(model)
    {
        return None;
    }
    // The program looks covariates up by the names `referenced_covariates` held when it
    // was built, and later registrations only add to that list.
    let names = &model.referenced_covariates;
    // Key layout: group, a presence mask, then the covariate bits. The mask tells a
    // missing covariate from one holding 0.0.
    let n_mask = names.len().div_ceil(64);
    let stride = 1 + n_mask + names.len();
    let cov_part = 1..stride;
    let mut keys: Vec<u64> = Vec::new();
    let mut prev: Option<&std::collections::HashMap<String, f64>> = None;
    for (group, cov) in records {
        let start = keys.len();
        keys.resize(start + stride, 0);
        let (done, key) = keys.split_at_mut(start);
        key[0] = group.map_or(0, |g| g as u64 + 1);
        // Consecutive records often borrow one map: every record of a static subject
        // does, and so does each run of records falling back to `subject.covariates`.
        if prev.is_some_and(|p| std::ptr::eq(p, cov)) {
            key[cov_part.clone()].copy_from_slice(&done[start - stride..][cov_part.clone()]);
        } else {
            let (mask, bits) = key[cov_part.clone()].split_at_mut(n_mask);
            for (c, name) in names.iter().enumerate() {
                if let Some(v) = cov.get(name) {
                    mask[c / 64] |= 1 << (c % 64);
                    bits[c] = v.to_bits();
                }
            }
        }
        prev = Some(cov);
    }
    let n = keys.len() / stride;
    let key = |i: usize| &keys[i * stride..(i + 1) * stride];
    // Sorting, not hashing: a few short integer comparisons per record, no allocation per
    // key. The index breaks ties, so each run of equal keys starts at its earliest record.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| key(a).cmp(key(b)).then(a.cmp(&b)));
    let mut first = vec![0; n];
    let mut run = 0;
    for (pos, &i) in order.iter().enumerate() {
        if pos == 0 || key(i) != key(order[pos - 1]) {
            run = i;
        }
        first[i] = run;
    }
    Some(first)
}

/// Fill each of the empty `outs` with one `$PK` snapshot per entry of `records[l]`.
/// Records with the same inputs share one evaluation
/// ([`first_record_with_same_pk_inputs`]); the rest are bit-identical copies. `records`
/// are `subject`'s records for `site`, indexed by an open [`sharing_scope::SharingScope`].
#[allow(clippy::too_many_arguments)]
fn fill_per_record_pk_params(
    model: &CompiledModel,
    subject: &Subject,
    site: sharing_scope::ShareSite,
    theta: &[f64],
    eta: &[f64],
    records: &[&[(&std::collections::HashMap<String, f64>, f64)]],
    outs: &mut [&mut Vec<PkParams>],
) {
    debug_assert_eq!(records.len(), outs.len());
    // `locate` indexes each list from 0; both callers clear `outs` first.
    debug_assert!(outs.iter().all(|o| o.is_empty()));
    let first = sharing_scope::first_record_with_same_pk_inputs(
        model,
        subject,
        site,
        records
            .iter()
            .flat_map(|r| r.iter().map(|&(cov, _)| (None, cov))),
    );
    // Record `i` of the concatenation sits at `(list, i - offset[list])`.
    let locate = |i: usize| {
        let mut off = 0;
        for (l, r) in records.iter().enumerate() {
            if i < off + r.len() {
                return (l, i - off);
            }
            off += r.len();
        }
        unreachable!("record {i} past the {off} listed")
    };
    let mut i = 0;
    for (l, list) in records.iter().enumerate() {
        for &(cov, t) in list.iter() {
            match first.as_ref().map(|f| f[i]).filter(|&f| f != i) {
                None => push_pk_in_place(outs[l], || pk_params_at_time(model, theta, eta, cov, t)),
                // `f < i`, so its snapshot is already in place.
                Some(f) => {
                    let (fl, fj) = locate(f);
                    let p = outs[fl][fj];
                    push_pk_in_place(outs[l], || p);
                }
            }
            i += 1;
        }
    }
}

/// The dose attributes an `ADDL`-implied dose takes from its parent row's `$PK` rather
/// than its governing record's (#1909, [`crate::types::ImpliedDose`]): the lag, as NONMEM
/// 7.5.1 fixes it at the parent row (`nonmem_anchor/addl_*`). A route's own lag (#2367) is
/// a per-forcing slot, added by [`parent_dose_attr_copies`].
const PARENT_DOSE_ATTRS: [crate::types::DoseAttr; 1] = [crate::types::DoseAttr::Lag];

/// `(k, parent, slot)` for every `ADDL`-implied dose `k` and every slot of
/// [`PARENT_DOSE_ATTRS`] its compartment reads, then the route lag of every input-rate
/// forcing its dose feeds. The slot is the one the engine resolves for that dose
/// (`ALAG{cmt}`, else the bare lag; the forcing's `lag_slot`). A route lag is a depot
/// `ALAG`, which NONMEM fixes at the parent row (`nonmem_anchor/addl_lag`).
pub(crate) fn parent_dose_attr_copies<'a>(
    model: &'a CompiledModel,
    subject: &'a Subject,
) -> impl Iterator<Item = (usize, usize, usize)> + 'a {
    let map = model.active_dose_attr_map();
    let forcings = model
        .ode_spec
        .as_ref()
        .map_or(&[][..], |o| &o.input_rate[..]);
    (0..subject.doses.len()).flat_map(move |k| {
        let parent = subject.implied_dose_parent(k);
        let dose = &subject.doses[k];
        let cmt = dose.cmt_1based();
        let attrs = PARENT_DOSE_ATTRS
            .iter()
            .filter_map(move |&attr| match attr {
                crate::types::DoseAttr::Lag => Some(map.lag_slot(cmt)),
                other => map.indexed_slot(other, cmt),
            });
        let route_lags = forcings
            .iter()
            .filter(move |f| f.cmt == dose.cmt_idx())
            .filter_map(|f| f.lag_slot);
        parent
            .map(move |p| attrs.chain(route_lags).map(move |s| (k, p, s)))
            .into_iter()
            .flatten()
    })
}

/// Give each `ADDL`-implied dose its parent's [`PARENT_DOSE_ATTRS`] (#1909) and route lags
/// (#2367), as listed by [`parent_dose_attr_copies`]: the post-pass every per-dose builder
/// runs, so no second `$PK` call is needed.
pub(crate) fn inherit_parent_dose_attrs(
    model: &CompiledModel,
    subject: &Subject,
    dose: &mut [PkParams],
) {
    for (k, p, slot) in parent_dose_attr_copies(model, subject) {
        dose[k].values[slot] = dose[p].values[slot];
    }
}

/// [`inherit_parent_dose_attrs`] over per-dose seed vectors of any scalar, for the
/// `sens/` walks whose dose snapshots are `Vec<T>` of duals.
pub(crate) fn inherit_parent_dose_attrs_g<T: Clone>(
    model: &CompiledModel,
    subject: &Subject,
    dose: &mut [Vec<T>],
) {
    for (k, p, slot) in parent_dose_attr_copies(model, subject) {
        // Every seed vector spans the `ode_param_slots` layout, which reserves the lag
        // slot; a shorter one would leave the implied dose on the governing record's lag.
        debug_assert!(
            slot < dose[p].len() && slot < dose[k].len(),
            "seed vector too short for parent-dose slot {slot}: dose {k} has {}, parent {p} has {}",
            dose[k].len(),
            dose[p].len()
        );
        if slot < dose[p].len() && slot < dose[k].len() {
            dose[k][slot] = dose[p][slot].clone();
        }
    }
}

/// `out.push(f())` without the intermediate copy: `PkParams` is 1 KB and returned by
/// value, so writing straight into the reserved slot saves a 1 KB `memcpy` per event in
/// hot loops (#1477).
#[inline]
fn push_pk_in_place(out: &mut Vec<PkParams>, f: impl FnOnce() -> PkParams) {
    out.reserve(1);
    let len = out.len();
    // SAFETY: `reserve(1)` guarantees `capacity() > len`, so `ptr.add(len)` is a
    // valid, unaliased, uninitialised slot inside the allocation; it is fully
    // written before `set_len` publishes it, and `f` cannot observe `out`.
    unsafe {
        out.as_mut_ptr().add(len).write(f());
        out.set_len(len + 1);
    }
}

/// Just the **dose** snapshots of [`compute_event_pk_params_into`], for the fit-init
/// dose-attribute check (`E_DOSE_ATTR_NONFINITE`). Identical to
/// `compute_event_pk_params(..).dose` by construction (same [`subject_needs_per_event_pk`]
/// gate, same [`fill_dose_pk_params`]); pinned by
/// `tests::dose_only_snapshots_match_the_full_event_builder`. Exists for cost: 1000
/// subjects × (2 doses + 100 obs), TV covariate: 9.9 ms vs 111.7 ms (#1235).
pub(crate) fn compute_dose_pk_params_into(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    out: &mut Vec<PkParams>,
) {
    let constant = (!subject_needs_per_event_pk(model, subject))
        .then(|| (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0));
    fill_dose_pk_params(model, subject, theta, eta, constant, out);
}

/// Same as [`compute_event_pk_params`] but writes into a caller-owned buffer, reused
/// across `eta` evaluations (SAEM MH, FOCE objective); no allocation if capacity fits.
/// The TV-cov path still builds the per-record input lists and sharing index each call
/// (an in-crate EBE search keeps the index, #2558).
pub fn compute_event_pk_params_into(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    out: &mut EventPkParams,
) {
    out.dose.clear();
    out.obs.clear();
    out.pk_only.clear();
    out.reset.clear();

    // Reserve only when this materializer is actually reached. Static prediction
    // paths can keep an empty scratch buffer without allocating event snapshots.
    // Exact reservations avoid geometric growth of these large PkParams values.
    out.dose.reserve_exact(subject.doses.len());
    out.obs.reserve_exact(subject.obs_times.len());
    out.pk_only.reserve_exact(subject.pk_only_times.len());
    out.reset.reserve_exact(subject.reset_times.len());

    if subject_needs_per_event_pk(model, subject) {
        // Shared with `compute_dose_pk_params_into` so the doses-only caller cannot
        // drift from this one (#1235).
        fill_dose_pk_params(model, subject, theta, eta, None, &mut out.dose);
        let obs: Vec<_> = obs_pk_inputs(subject).collect();
        let pk_only: Vec<_> = pk_only_pk_inputs(subject).collect();
        // EVID=3/4 rows are data records too (#1133): `$PK` runs at them, and the
        // resulting snapshot is what re-seeds `[odes] init(...)` when the reset
        // restarts the episode.
        let reset: Vec<_> = reset_pk_inputs(subject).collect();
        fill_per_record_pk_params(
            model,
            subject,
            sharing_scope::ShareSite::ObsPkOnlyReset,
            theta,
            eta,
            &[&obs, &pk_only, &reset],
            &mut [&mut out.obs, &mut out.pk_only, &mut out.reset],
        );
    } else {
        // Reached only when the subject carries no time-varying covariates (on
        // dose/obs *or* pk-only rows) and the model uses no `TIME` built-in, so a
        // single snapshot (t=0) is exact for every event.
        let p = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);
        // Every event row gets the snapshot — `pk_only` included: the event-driven walkers
        // assert every length unconditionally, and `prune_irrelevant_tv_covariates` can
        // leave `pk_only_times` non-empty on a parameter-static subject (#1124 review).
        // `p` is handed to the shared filler so this arm costs one `$PK` call (#1235).
        fill_dose_pk_params(model, subject, theta, eta, Some(p), &mut out.dose);
        for _ in 0..subject.obs_times.len() {
            out.obs.push(p);
        }
        for _ in 0..subject.pk_only_times.len() {
            out.pk_only.push(p);
        }
        // Filled even on this branch: a reset-carrying subject with constant
        // covariates still needs a seed snapshot, and `p` is exact for it.
        for _ in 0..subject.reset_times.len() {
            out.reset.push(p);
        }
    }
}

/// The decision window (occasion) active at time `t` for adaptive-dosing IOV (#701): the
/// index of the latest decision at or before `t` (within 1e-9), or `None` before the first
/// (baseline window, κ = 0). Requires `decision_times` strictly ascending.
///
/// The 1e-9 tolerance is looser than the exact-bits lookups and the #700 guard (1e-15),
/// but driver, frozen-replay verifier and `compute_event_pk_params_iov` all resolve
/// occasions through this one function, so they cannot disagree.
pub(crate) fn occasion_of(decision_times: &[f64], t: f64) -> Option<usize> {
    let mut occ = None;
    for (g, &d) in decision_times.iter().enumerate() {
        if d <= t + 1e-9 {
            occ = Some(g);
        } else {
            break;
        }
    }
    occ
}

/// Occasion-aware per-event PK for the adaptive IOV path (#701): like
/// [`compute_event_pk_params`] but each record uses the eta of the decision window active at
/// its time ([`occasion_of`]), or `eta_baseline` (κ = 0) before the first decision.
/// Covariate snapshots are per record as usual (#700).
///
/// - **Doses** (#931): a base dose takes the occasion active at administration (baseline F
///   before the first decision).
/// - **pk-only (EVID=2)** rows take the decision window, not their `OCC`
///   ([`predict_iov`] reads `OCC`, [`pk_only_row_occasion`], #1876), consistent with the
///   driver's `governing_segment_occ_at`.
/// - **Base dose at an occasion boundary** agrees with [`predict_iov`] (end-of-interval rule,
///   #1073/#1148): 2e-16 relative on #2771's fixture.
/// - **Controller dose / hold**: every decision is a driver record (#2895), so the run equals
///   [`predict_iov`] with an `EVID=2` row per decision
///   (`the_run_equals_predict_iov_on_the_realized_doses_plus_a_record_per_decision`).
/// - **`ADDL`-implied dose** (#2371) takes its governing record's occasion, resolved at
///   [`Subject::dose_occasion_time`] ([`crate::types::ImpliedDose`]); the dose's own time
///   was 5.6 % off `predict_iov` on the regression fixture.
pub(crate) fn compute_event_pk_params_iov(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta_baseline: &[f64],
    eta_occ: &[Vec<f64>],
    decision_times: &[f64],
) -> EventPkParams {
    let mut out = EventPkParams::with_capacity_for(subject);
    let eta_at = |t: f64| -> &[f64] {
        match occasion_of(decision_times, t) {
            Some(g) => eta_occ[g].as_slice(),
            None => eta_baseline,
        }
    };
    // Base doses (#702 × #931) carry the occasion active at administration (an implied
    // dose its governing record's, #2371), so `f_bio` reads that occasion's κ.
    for k in 0..subject.doses.len() {
        out.dose.push(pk_params_at_time(
            model,
            theta,
            eta_at(subject.dose_occasion_time(k)),
            subject.dose_cov(k),
            subject.dose_param_time(k),
        ));
    }
    inherit_parent_dose_attrs(model, subject, &mut out.dose);
    for j in 0..subject.obs_times.len() {
        let t = subject.obs_times[j];
        out.obs.push(pk_params_at_time(
            model,
            theta,
            eta_at(t),
            subject.obs_cov(j),
            subject.param_time_obs(j),
        ));
    }
    for m in 0..subject.pk_only_times.len() {
        let t = subject.pk_only_times[m];
        out.pk_only.push(pk_params_at_time(
            model,
            theta,
            eta_at(t),
            subject.pk_only_cov(m),
            subject.param_time_pk_only(m),
        ));
    }
    // EVID=3/4 rows are records (#1133): the reset re-seeds `[odes] init(...)` from the
    // reset row's own `$PK`, which here means its own covariate snapshot under the κ of
    // the decision window active at the reset — the same rule every other record follows.
    for r in 0..subject.reset_times.len() {
        let t = subject.reset_times[r];
        out.reset.push(pk_params_at_time(
            model,
            theta,
            eta_at(t),
            subject.reset_cov(r),
            subject.param_time_at(t),
        ));
    }
    out
}

/// Positivity floor applied before log-transforming a prediction under LTBS, so
/// a prediction the optimizer drives toward zero yields a large-but-finite log
/// value instead of `-inf`/`NaN`.
pub(crate) const LTBS_FLOOR: f64 = 1e-12;

/// The post-engine tail of the `pk` sink: `[scaling]`, then
/// [`finish_on_observation_scale`]. Every non-IOV prediction sink ends here, so a new
/// tail step reaches all of them (#2129, #2156, #2259 were incomplete copies).
pub(crate) fn to_observation_scale(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    preds: &mut [f64],
) {
    apply_scaling(model, subject, theta, eta, preds);
    finish_on_observation_scale(model, subject, theta, eta, preds);
}

/// The part of the `pk` sink's tail after `[scaling]`: the LTBS log-wrap, then the FREM
/// override. Sinks with a per-occasion/per-window `[scaling]` step (`predict_iov`,
/// `api::adaptive::adaptive_rows_to_model_scale`) call this directly. The FREM override
/// comes last (pseudo-observations are `θ_k + η_m` on the raw scale); it reads only BSV
/// slots of `eta_bsv`.
pub(crate) fn finish_on_observation_scale(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta_bsv: &[f64],
    preds: &mut [f64],
) {
    apply_log_transform(model, preds);
    apply_frem_prediction_override(model, subject, theta, eta_bsv, preds);
}

/// Log-transform predictions in place under log-transform-both-sides (LTBS); a no-op
/// otherwise. Applied at every prediction sink (NONMEM's `Y = LOG(F) + EPS`). The analytic
/// sensitivity paths (`sens::provider`, `sens::ode_provider`) must stay in sync.
#[inline]
pub(crate) fn apply_log_transform(model: &CompiledModel, preds: &mut [f64]) {
    if model.log_transform {
        for p in preds.iter_mut() {
            *p = ltbs_log_g(*p);
        }
    }
}

/// The LTBS log transform, generic over `T: PkNum`, shared by the `f64` predictor and the
/// dual walks (#438 review #4 / #451): `p.max(LTBS_FLOOR).ln()` for `f64`. `guard_floor`
/// floors `NaN` too (like `f64::max`); dual callers that must keep a `NaN` visible
/// pre-check it.
#[inline]
pub(crate) fn ltbs_log_g<T: crate::sens::num::PkNum>(p: T) -> T {
    p.guard_floor(LTBS_FLOOR).ln()
}

pub fn predict_iov(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta_bsv: &[f64],
    kappas: &[Vec<f64>],
) -> Vec<f64> {
    predict_iov_with_scratch(
        model,
        subject,
        theta,
        eta_bsv,
        kappas,
        &mut EventPkParams::default(),
    )
}

/// IOV predictions with caller-owned per-event parameter storage. Every event
/// is still evaluated at its current theta, eta, kappa, covariates and time;
/// only allocation capacity is reused, never numerical values.
pub(crate) fn predict_iov_with_scratch<K: AsRef<[f64]>>(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta_bsv: &[f64],
    kappas: &[K],
    scratch: &mut EventPkParams,
) -> Vec<f64> {
    predict_iov_with_scratch_and_schedule(model, subject, theta, eta_bsv, kappas, scratch, None)
}

/// [`predict_iov_with_scratch`] with the subject's cached
/// [`EventSchedule`](event_driven::EventSchedule) (#1452 review). A cached schedule is
/// valid only when nothing needed resolving (`Cow::Borrowed`), as on the non-IOV arm.
/// `None` rebuilds it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn predict_iov_with_scratch_and_schedule<K: AsRef<[f64]>>(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta_bsv: &[f64],
    kappas: &[K],
    scratch: &mut EventPkParams,
    schedule: Option<&event_driven::EventSchedule>,
) -> Vec<f64> {
    let layout = crate::stats::stacked_layout::StackedLayout::new_cached(
        eta_bsv.len(),
        model.n_kappa,
        subject,
    );
    predict_iov_in(
        model, subject, &layout, theta, eta_bsv, kappas, scratch, schedule,
    )
}

/// [`predict_iov_with_scratch_and_schedule`] over a caller-built occasion → group
/// `layout`, built once per subject by callers predicting it many times (#2037).
#[allow(clippy::too_many_arguments)]
pub(crate) fn predict_iov_in<K: AsRef<[f64]>>(
    model: &CompiledModel,
    subject: &Subject,
    layout: &crate::stats::stacked_layout::StackedLayout,
    theta: &[f64],
    eta_bsv: &[f64],
    kappas: &[K],
    scratch: &mut EventPkParams,
    schedule: Option<&event_driven::EventSchedule>,
) -> Vec<f64> {
    debug_assert_eq!(layout.n_eta(), eta_bsv.len());
    debug_assert_eq!(layout.n_kappa(), model.n_kappa);
    let EventPkParams {
        dose: dose_params,
        obs: obs_params,
        pk_only: pk_only_params,
        reset: reset_params,
    } = scratch;
    dose_params.clear();
    obs_params.clear();
    pk_only_params.clear();
    reset_params.clear();
    // #905: an analytical subject with no Gaussian observation feeds the predictor
    // nothing (and its dose compartment is unchecked), so decline as the non-IOV gate
    // does: NaN (empty when `obs_times` is empty).
    if model.ode_spec.is_none() && !subject_feeds_analytical_pk(model, subject) {
        return vec![f64::NAN; subject.obs_times.len()];
    }
    // Closed-form transit/IG IOV (#1560): served by the exact absorption walk when every
    // (interval, open dose) pair is in the tilting domain, else by the ODE twin (#719).
    let primary = model;
    let walk = absorption_walk::walk_eligible(model, subject);
    let model = if walk {
        model
    } else {
        model.effective_for(subject)
    };
    let n_kappa = model.n_kappa;

    // occasion id -> kappa-group index (iov_occasion_groups order).
    let combined_for = iov_occasion_eta(layout, eta_bsv, kappas);

    // The closed-form event walk with a cached schedule borrows each record's snapshot
    // from a table of distinct ones (#2549); every other engine reads one per record.
    let indexed = !iov_table_off()
        && schedule.is_some()
        && model.ode_spec.is_none()
        && !model.is_algebraic()
        && !walk
        && event_driven::supports_event_driven(model.pk_model)
        && model.analytical_init.is_empty();
    let mut dose_ix: Vec<u32> = crate::scratch::take();
    let mut obs_ix: Vec<u32> = crate::scratch::take();
    let mut pk_only_ix: Vec<u32> = crate::scratch::take();
    if indexed {
        // `dose_params` holds the table; the per-record views are the index vectors.
        fill_iov_record_table(
            model,
            subject,
            theta,
            &combined_for,
            dose_params,
            &mut dose_ix,
            &mut obs_ix,
            &mut pk_only_ix,
        );
    } else {
        // Exact initial reservations preserve the convenience API's old footprint
        // for short vectors; later probes keep these capacities without allocating.
        fill_iov_record_params(
            model,
            subject,
            theta,
            &combined_for,
            dose_params,
            obs_params,
            pk_only_params,
        );
    }
    // The BSV-only vector `[η_bsv, 0…]` (pk-only rows read their own occasion, #1876).
    let bsv_only_combined = combined_for(u32::MAX);

    let mut preds = if model.is_algebraic() {
        // Compartment-free model (#811): κ enters only via the per-occasion readout pass.
        vec![0.0; subject.obs_times.len()]
    } else if let Some(ref ode) = model.ode_spec {
        // EVID=3/4 rows: the reset row's own covariates (#1133) under its own `OCC`
        // ([`reset_row_occasion`], #1153). Only the ODE engine re-seeds `init(...)`.
        reset_params.reserve_exact(subject.reset_times.len());
        reset_params.extend((0..subject.reset_times.len()).map(|r| {
            let t = subject.param_time_at(subject.reset_times[r]);
            let combined = match reset_row_occasion(subject, r) {
                Some(occ) => combined_for(occ),
                None => bsv_only_combined.clone(),
            };
            pk_params_at_time(model, theta, &combined, subject.reset_cov(r), t)
        }));
        crate::ode::ode_predictions_event_driven(
            ode,
            subject,
            theta,
            &bsv_only_combined,
            dose_params.as_slice(),
            obs_params.as_slice(),
            pk_only_params.as_slice(),
            reset_params.as_slice(),
        )
    } else if walk {
        match absorption_walk::absorption_walk_predictions(
            model.pk_model,
            subject,
            dose_params.as_slice(),
            obs_params.as_slice(),
            pk_only_params.as_slice(),
            None,
        ) {
            Some(preds) => preds,
            // A pair outside the tilting domain (the flip-flop regime): the twin, which is
            // valid in both regimes, serves the whole subject.
            None => {
                return predict_iov_with_scratch_and_schedule(
                    primary.effective_for(subject),
                    subject,
                    theta,
                    eta_bsv,
                    kappas,
                    &mut EventPkParams::default(),
                    None,
                )
            }
        }
    } else if event_driven::supports_event_driven(model.pk_model) {
        // Resolve modeled-`RATE` doses (#324/#394) with each dose's per-occasion PK
        // snapshot before the analytical walker. Borrowed no-op when all are `Fixed`.
        let resolved =
            crate::dosing::resolve_subject_doses_with(subject, model.active_dose_attr_map(), |k| {
                if indexed {
                    &dose_params[dose_ix[k] as usize].values
                } else {
                    &dose_params[k].values
                }
            });
        if indexed {
            let at = |idx: &[u32]| -> Vec<PkParams> {
                idx.iter().map(|&i| dose_params[i as usize]).collect()
            };
            match (&resolved, schedule) {
                (std::borrow::Cow::Borrowed(_), Some(sched)) => {
                    let view = |idx| event_driven::Indexed {
                        table: dose_params.as_slice(),
                        idx,
                    };
                    event_driven::event_driven_predictions_with_schedule_at(
                        model.pk_model,
                        &resolved,
                        sched,
                        &view(dose_ix.as_slice()),
                        &view(obs_ix.as_slice()),
                        &view(pk_only_ix.as_slice()),
                    )
                }
                // A resolved modeled dose rebuilds the schedule from per-record slices:
                // the snapshots each record reads, written out.
                _ => event_driven::event_driven_predictions(
                    model.pk_model,
                    &resolved,
                    &at(&dose_ix),
                    &at(&obs_ix),
                    &at(&pk_only_ix),
                ),
            }
        } else if let (std::borrow::Cow::Borrowed(_), Some(sched)) = (&resolved, schedule) {
            event_driven::event_driven_predictions_with_schedule(
                model.pk_model,
                &resolved,
                sched,
                dose_params.as_slice(),
                obs_params.as_slice(),
                pk_only_params.as_slice(),
            )
        } else {
            event_driven::event_driven_predictions(
                model.pk_model,
                &resolved,
                dose_params.as_slice(),
                obs_params.as_slice(),
                pk_only_params.as_slice(),
            )
        }
    } else {
        // Unreachable: every `PkModel` variant has event-driven support; a new variant not
        // wired into `supports_event_driven` fails loud here (#327).
        unreachable!(
            "predict_iov: model {:?} has neither an ODE spec nor event-driven \
             analytical support; wire it into `supports_event_driven`",
            model.pk_model
        )
    };

    // Analytical initial amounts (#521), before scaling: `eta_bsv` drives the baseline
    // amount, the per-occasion `obs_params` drive its decay (#521 review).
    if !model.analytical_init.is_empty() && model.ode_spec.is_none() {
        // Baseline amount is seeded at the record start (internal t=0 under the
        // subject-relative time origin), so the `TIME` built-in resolves there.
        let amount_pk = (model.pk_param_fn)(theta, eta_bsv, &subject.covariates, 0.0);
        add_analytical_init_with(
            model,
            subject,
            theta,
            eta_bsv,
            &amount_pk,
            Some(obs_params.as_slice()),
            &mut preds,
        );
    }

    // `[scaling]` applied **per occasion** so a κ-dependent scale uses that occasion's κ.
    if !matches!(model.scaling, ScalingSpec::None) {
        // Guard on `subject.occasions`, NOT `occ_groups`: dose-only occasions are groups
        // with no observations, so keying on `occ_groups` would leave observations
        // unscaled (#769 review).
        if subject.occasions.is_empty() {
            // No observation carries an occasion label: scale with `[η_bsv, 0…0]`, as the
            // non-IOV path does (#723 review).
            apply_scaling(model, subject, theta, &bsv_only_combined, &mut preds);
        } else {
            // Every observation is owned by exactly one group, so no κ=0 base pass is
            // needed (#769).
            let raw = preds.clone();
            for (occ_id, obs_indices) in layout.groups() {
                // Dose-only occasions own no observations, so their scaling pass
                // would write nothing — skip the redundant full `apply_scaling`
                // (+ clone) rather than compute and discard it.
                if obs_indices.is_empty() {
                    continue;
                }
                let combined = combined_for(*occ_id);
                let mut scaled = raw.clone();
                apply_scaling(model, subject, theta, &combined, &mut scaled);
                for &j in obs_indices {
                    preds[j] = scaled[j];
                }
            }
        }
    }
    // Analytic Form C readout (#650) under IOV: evaluated **per occasion** with that
    // occasion's `[η_bsv, κ]`, like NONMEM's per-record `$ERROR` (#1079). `central / V`
    // still cancels exactly since `V` is `obs_params[j]`. Compartment-free models (#811)
    // take the same loop.
    if model.analytic_readout.is_some() && n_kappa > 0 && !subject.occasions.is_empty() {
        // Guarded on `subject.occasions` for the scaling pass's reason; unlabelled
        // subjects fall through to the shared κ = 0 call below.
        let raw = preds.clone();
        for (occ_id, obs_indices) in layout.groups() {
            if obs_indices.is_empty() {
                continue;
            }
            let combined = combined_for(*occ_id);
            let mut occ_preds = raw.clone();
            apply_analytic_readout(model, subject, theta, &combined, &mut occ_preds);
            for &j in obs_indices {
                preds[j] = occ_preds[j];
            }
        }
    } else {
        // No readout (the no-op call), no κ (the per-occasion loop would rebuild the
        // identical `[η_bsv]` vector), or no observation carries an occasion label.
        apply_analytic_readout(model, subject, theta, eta_bsv, &mut preds);
    }
    // LTBS log-wrap, then the FREM override — the sink's shared tail after the
    // per-occasion `[scaling]` pass above. The IOV dispatch above is total (ODE or
    // event-driven), so predictions are logged exactly once.
    crate::scratch::give(dose_ix);
    crate::scratch::give(obs_ix);
    crate::scratch::give(pk_only_ix);
    finish_on_observation_scale(model, subject, theta, eta_bsv, &mut preds);
    preds
}

/// The superposition engine's [`times_have_non_finite`](crate::ode::predictions::times_have_non_finite)
/// (#1189, #1284): true when any dose's effective arrival `dose.time + lagtime` is
/// non-finite (a `NaN` lag, or out-of-domain `D{n}`/`R{n}`,
/// [`DoseEvent::non_finite`](crate::types::DoseEvent)), so the whole subject must come
/// back non-finite rather than silently dropping the dose. The single predicate for
/// [`predict_concentration`] and [`analytical_state_at_times`] (#1399 review); a third
/// superposition walk must use it too.
#[inline]
pub(crate) fn superposition_arrival_non_finite(doses: &[DoseEvent], lagtime: f64) -> bool {
    doses.iter().any(|d| !(d.time + lagtime).is_finite())
}

/// Predict concentration at a given time for a subject by superposition over all prior
/// doses. `pk_params.lagtime()` shifts every dose's start (infusion duration preserved).
///
/// # Steady state under lagtime
///
/// An observation between an SS dose's record and its lagged arrival still sees the
/// previous interval's tail (last pulse at `dose.time − crate::dosing::ss_seed_phase`).
/// Matches NONMEM `ALAG1` + `SS=1` (7.5 to 5 significant figures; 7.6.0 for the
/// `lagtime ≥ II` clamp, `nonmem_anchor/results/ss_lag_ge_ii`).
pub fn predict_concentration(
    pk_model: PkModel,
    doses: &[DoseEvent],
    t: f64,
    pk_params: &PkParams,
) -> f64 {
    let lagtime = pk_params.lagtime();
    // A non-finite arrival makes the whole subject non-finite — see
    // `superposition_arrival_non_finite`, which the state twin
    // (`analytical_state_at_times`) reads too.
    if superposition_arrival_non_finite(doses, lagtime) {
        return f64::NAN;
    }
    // An `SS=1` record resets the system (#1576): a dose that precedes the latest
    // reached record in (time, row order) contributes nothing here.
    let cutoff = crate::dosing::ss_reset_cutoff(doses, t);
    let mut conc = 0.0;
    for (k, dose) in doses.iter().enumerate() {
        if !crate::dosing::ss_reset_live(doses, cutoff, k) {
            continue;
        }
        let t_eff = dose.time + lagtime;
        if t_eff <= t {
            let tau = t - t_eff;
            if dose.ss && dose.ii > 0.0 && !crate::dosing::ss_arrival_is_trough(dose, lagtime) {
                // Past `lag = II` the collapsed `C_ss(t − t_eff)` reads ~4 % high (#1121),
                // so split into the tail (last pulse at `dose.time − phase`) plus the one
                // real dose arriving at `t_eff`. Kept as a branch: the forms are equal for
                // `lag ≤ II` only algebraically, and SS results are pinned to the collapsed one.
                conc += single_dose_concentration(
                    pk_model,
                    dose,
                    t - dose.time + crate::dosing::ss_seed_phase(dose, lagtime),
                    pk_params,
                );
                let mut arrival = dose.clone();
                arrival.ss = false;
                conc += single_dose_concentration(pk_model, &arrival, tau, pk_params);
            } else {
                conc += single_dose_concentration(pk_model, dose, tau, pk_params);
            }
        } else if dose.ss && dose.ii > 0.0 && t >= dose.time {
            // Pre-arrival steady-state tail, only at/after the record time. Elapsed time
            // from the preceding pulse is `t − dose.time + p`, `p = max(II − lagtime, 0)`:
            // it CLAMPS for `lagtime ≥ II` (NONMEM 7.6.0, `crate::dosing::ss_seed_phase`).
            let tau = t - dose.time + crate::dosing::ss_seed_phase(dose, lagtime);
            if tau >= 0.0 {
                conc += single_dose_concentration(pk_model, dose, tau, pk_params);
            }
        }
    }
    // Floor at zero, but preserve `NaN` (`f64::max` would discard it, #1284, #1399
    // review finding 2): a non-finite `F` reaches `conc` through the closed form.
    if conc.is_nan() {
        conc
    } else {
        conc.max(0.0)
    }
}

/// External bioavailability post-multiplier for the superposition closed forms. Only an
/// **IV bolus** uses it; oral-depot forms (`*_oral_f`) bake `F` in, and infusions carry it
/// in `(rate, duration)` via [`DoseEvent::with_bioavailable_infusion`] (#419). Matches
/// NONMEM's `F1` (#327).
fn route_f_scale(pk_model: PkModel, infusion: bool, p: &PkParams) -> f64 {
    if infusion || pk_model.is_oral() {
        1.0
    } else {
        p.f_bio()
    }
}

/// Concentration contribution from a single dose at elapsed time tau. IV variants pick
/// bolus vs infusion per dose (`dose.is_infusion()`, #176); oral routes always use the
/// oral form. `dose.ss && dose.ii > 0` dispatches the SS form; malformed SS (`ii <= 0`,
/// `T_inf > ii`) falls through to single-dose and is warned about in `api.rs`.
fn single_dose_concentration(pk_model: PkModel, dose: &DoseEvent, tau: f64, p: &PkParams) -> f64 {
    let cl = p.cl();
    let v = p.v();
    // A degenerate `CL` or `V` drops the dose (0), a `NaN` one makes it `NaN` — the rule
    // the event-driven walk applies too, so the two engines serve the same subject the
    // same prediction (#2104). Transit / IG keep their own convention.
    if crate::sens::propagate::cl_v_rule_governs(pk_model) {
        if let Some(c) = crate::sens::propagate::superposed_dose_override(cl, v) {
            return c;
        }
    }
    let infusion = dose.is_infusion();
    let f_scale = route_f_scale(pk_model, infusion, p);

    // Bake `F` into the infusion `(rate, duration)` (#419); shadowing routes every call
    // below through the reshaped copy.
    let dose = &dose.with_bioavailable_infusion(p.f_bio());

    f_scale * central_raw(pk_model, dose, tau, p, infusion)
}

/// The raw (pre-`f_scale`) central closed form of one dose at elapsed time `tau`: the
/// value [`single_dose_concentration`] scales and the central entry of
/// [`single_dose_states`]. `dose` is the bioavailability-reshaped copy and `infusion`
/// its route. At steady state the central forms have separate `_ss` variants taking the
/// same arguments. Infusions bypass an oral model's depot and use the IV infusion form.
fn central_raw(pk_model: PkModel, dose: &DoseEvent, tau: f64, p: &PkParams, infusion: bool) -> f64 {
    let (cl, v) = (p.cl(), p.v());
    let ss = dose.ss && dose.ii > 0.0;
    macro_rules! pick {
        ($f:ident, $f_ss:ident ($($a:expr),*)) => {
            if ss {
                $f_ss(dose, tau, cl, v, $($a),*)
            } else {
                $f(dose, tau, cl, v, $($a),*)
            }
        };
    }
    match pk_model {
        // Transit / IG SS subjects are rerouted to the ODE twin by `effective_for` (#719).
        PkModel::OneCptTransit | PkModel::TwoCptTransit | PkModel::OneCptIg | PkModel::TwoCptIg
            if ss =>
        {
            unreachable!(
                "{} SS doses reroute to the ODE twin (effective_for)",
                pk_model.canonical_name()
            )
        }
        // Transit / IG reject infusions at parse, so only the absorbed bolus exists.
        PkModel::OneCptTransit => {
            one_cpt_transit_f(dose, tau, cl, v, p.n_transit(), p.mtt(), p.f_bio())
        }
        PkModel::TwoCptTransit => {
            let (q, v2, n, mtt, f) = (p.q(), p.v2(), p.n_transit(), p.mtt(), p.f_bio());
            two_cpt_transit_f(dose, tau, cl, v, q, v2, n, mtt, f)
        }
        PkModel::OneCptIg => one_cpt_ig_f(dose, tau, cl, v, p.mat(), p.cv2(), p.f_bio()),
        PkModel::TwoCptIg => {
            two_cpt_ig_f(dose, tau, cl, v, p.q(), p.v2(), p.mat(), p.cv2(), p.f_bio())
        }
        PkModel::OneCptIv | PkModel::OneCptOral if infusion => {
            pick!(one_cpt_infusion, one_cpt_infusion_ss())
        }
        PkModel::OneCptIv => pick!(one_cpt_iv_bolus, one_cpt_iv_bolus_ss()),
        PkModel::OneCptOral => pick!(one_cpt_oral_f, one_cpt_oral_f_ss(p.ka(), p.f_bio())),
        PkModel::TwoCptIv | PkModel::TwoCptOral if infusion => {
            pick!(two_cpt_infusion, two_cpt_infusion_ss(p.q(), p.v2()))
        }
        PkModel::TwoCptIv => pick!(two_cpt_iv_bolus, two_cpt_iv_bolus_ss(p.q(), p.v2())),
        PkModel::TwoCptOral => {
            let (q, v2, ka, f) = (p.q(), p.v2(), p.ka(), p.f_bio());
            pick!(two_cpt_oral_f, two_cpt_oral_f_ss(q, v2, ka, f))
        }
        PkModel::ThreeCptIv | PkModel::ThreeCptOral if infusion => {
            let (q, v2, q3, v3) = (p.q(), p.v2(), p.q3(), p.v3());
            pick!(three_cpt_infusion, three_cpt_infusion_ss(q, v2, q3, v3))
        }
        PkModel::ThreeCptIv => {
            let (q, v2, q3, v3) = (p.q(), p.v2(), p.q3(), p.v3());
            pick!(three_cpt_iv_bolus, three_cpt_iv_bolus_ss(q, v2, q3, v3))
        }
        PkModel::ThreeCptOral => {
            let (q, v2, q3, v3, ka, f) = (p.q(), p.v2(), p.q3(), p.v3(), p.ka(), p.f_bio());
            pick!(three_cpt_oral_f, three_cpt_oral_f_ss(q, v2, q3, v3, ka, f))
        }
    }
}

// --- Compartment-state helpers for `compartment_states` in SubjectResult ---

/// Returns the full state vector contribution from one dose at elapsed time `tau`.
///
/// Layout per PkModel:
///   OneCptIv:     [C_central]
///   OneCptOral:   [A_depot, C_central]
///   TwoCptIv:     [C_central, C_periph]
///   TwoCptOral:   [A_depot, C_central, C_periph]
///   ThreeCptIv:   [C_central, C_periph1, C_periph2]
///   ThreeCptOral: [A_depot, C_central, C_periph1, C_periph2]
///
/// Scaling is NOT applied — these are the raw analytical PK states.
fn single_dose_states(pk_model: PkModel, dose: &DoseEvent, tau: f64, p: &PkParams) -> Vec<f64> {
    let cl = p.cl();
    let v = p.v();
    // The state twin of `single_dose_concentration`'s degenerate-parameter rule (#2104).
    if crate::sens::propagate::cl_v_rule_governs(pk_model) {
        if let Some(c) = crate::sens::propagate::superposed_dose_override(cl, v) {
            return vec![c; pk_model.topology().n_states];
        }
    }
    let infusion = dose.is_infusion();

    // Bioavailability as in `single_dose_concentration` (#419): IV bolus post-multiplies
    // via `f_scale`; oral-depot and infusion bake `F` in (`f_scale == 1.0`).
    let f_scale = route_f_scale(pk_model, infusion, p);
    let dose = &dose.with_bioavailable_infusion(p.f_bio());

    // The central entry is `central_raw`'s (SS dispatch); peripheral/depot helpers handle
    // SS internally. Infusions bypass an oral model's depot.
    let central = central_raw(pk_model, dose, tau, p, infusion);
    let mut state = match pk_model {
        PkModel::OneCptIv => vec![central],
        PkModel::OneCptOral if infusion => vec![0.0, central],
        PkModel::OneCptOral => vec![one_cpt_oral_depot(dose, tau, p.ka(), p.f_bio()), central],
        // [depot = lumped in-transit mass, central(, peripheral)].
        PkModel::OneCptTransit => vec![
            one_cpt_transit_depot(dose, tau, p.n_transit(), p.mtt(), p.f_bio()),
            central,
        ],
        PkModel::TwoCptTransit => {
            let (q, v2, n, mtt, f) = (p.q(), p.v2(), p.n_transit(), p.mtt(), p.f_bio());
            vec![
                two_cpt_transit_depot(dose, tau, cl, v, q, v2, n, mtt, f),
                central,
                two_cpt_transit_peripheral(dose, tau, cl, v, q, v2, n, mtt, f),
            ]
        }
        // [depot = unabsorbed IG mass, central(, peripheral)].
        PkModel::OneCptIg => vec![
            one_cpt_ig_depot(dose, tau, p.mat(), p.cv2(), p.f_bio()),
            central,
        ],
        PkModel::TwoCptIg => {
            let (q, v2, mat, cv2, f) = (p.q(), p.v2(), p.mat(), p.cv2(), p.f_bio());
            vec![
                two_cpt_ig_depot(dose, tau, cl, v, q, v2, mat, cv2, f),
                central,
                two_cpt_ig_peripheral(dose, tau, cl, v, q, v2, mat, cv2, f),
            ]
        }
        PkModel::TwoCptIv => vec![
            central,
            two_cpt_iv_peripheral(dose, tau, cl, v, p.q(), p.v2()),
        ],
        PkModel::TwoCptOral if infusion => {
            vec![
                0.0,
                central,
                two_cpt_iv_peripheral(dose, tau, cl, v, p.q(), p.v2()),
            ]
        }
        PkModel::TwoCptOral => {
            let (q, v2, ka, f) = (p.q(), p.v2(), p.ka(), p.f_bio());
            vec![
                one_cpt_oral_depot(dose, tau, ka, f),
                central,
                two_cpt_oral_peripheral(dose, tau, cl, v, q, v2, ka, f),
            ]
        }
        PkModel::ThreeCptIv => {
            let [p1, p2] =
                three_cpt_iv_peripherals(dose, tau, cl, v, p.q(), p.v2(), p.q3(), p.v3());
            vec![central, p1, p2]
        }
        PkModel::ThreeCptOral if infusion => {
            let [p1, p2] =
                three_cpt_iv_peripherals(dose, tau, cl, v, p.q(), p.v2(), p.q3(), p.v3());
            vec![0.0, central, p1, p2]
        }
        PkModel::ThreeCptOral => {
            let (q, v2, q3, v3, ka, f) = (p.q(), p.v2(), p.q3(), p.v3(), p.ka(), p.f_bio());
            let [p1, p2] = three_cpt_oral_peripherals(dose, tau, cl, v, q, v2, q3, v3, ka, f);
            vec![one_cpt_oral_depot(dose, tau, ka, f), central, p1, p2]
        }
    };

    // Post-multiply bioavailability for the IV-bolus route; a no-op
    // (`f_scale == 1.0`) for oral-depot and for infusions (whose `F` is already
    // baked into the reshaped `(rate, duration)`; #419).
    for s in &mut state {
        *s *= f_scale;
    }
    state
}

/// Compute the full compartment state vector at arbitrary `times` for an analytical model
/// (`[k]` is the state at `times[k]`), with `predict_concentration`'s lagtime/SS logic.
/// Used by `compute_extra_output_columns` and [`predict_all_states`].
pub fn analytical_state_at_times(
    pk_model: PkModel,
    subject: &Subject,
    pk_params: &PkParams,
    times: &[f64],
) -> Vec<Vec<f64>> {
    debug_assert!(
        subject.reset_times.is_empty(),
        "analytical_state_at_times called on a subject with EVID=3/4 resets — \
         superposition is invalid across reset boundaries; the caller should \
         return empty states for analytical+reset subjects instead"
    );
    let lagtime = pk_params.lagtime();
    let n_states = pk_model.topology().n_states;
    // The value twin's guard (#1284, #1399 review finding 1): without it a `NaN` arrival
    // yields a drug-free zero state next to `NaN` predictions.
    if superposition_arrival_non_finite(&subject.doses, lagtime) {
        return vec![vec![f64::NAN; n_states]; times.len()];
    }
    times
        .iter()
        .map(|&t| {
            let mut state = vec![0.0_f64; n_states];
            // The value twin's `SS=1` reset (#1576), on the same predicate.
            let cutoff = crate::dosing::ss_reset_cutoff(&subject.doses, t);
            for (k, dose) in subject.doses.iter().enumerate() {
                if !crate::dosing::ss_reset_live(&subject.doses, cutoff, k) {
                    continue;
                }
                let t_eff = dose.time + lagtime;
                let tau = if t_eff <= t {
                    t - t_eff
                } else if dose.ss && dose.ii > 0.0 && t >= dose.time {
                    let raw = t - t_eff;
                    let n = (-raw / dose.ii).ceil();
                    let wrapped = raw + n * dose.ii;
                    if wrapped >= 0.0 {
                        wrapped
                    } else {
                        continue;
                    }
                } else {
                    continue;
                };
                let contrib = single_dose_states(pk_model, dose, tau, pk_params);
                debug_assert_eq!(
                    contrib.len(),
                    n_states,
                    "single_dose_states returned {} values but n_states={}; \
                     update both match arms together",
                    contrib.len(),
                    n_states
                );
                for (s, v) in state.iter_mut().zip(contrib.iter()) {
                    *s += v;
                }
            }
            // Floor: states cannot be negative (amounts/concentrations).
            for s in &mut state {
                if *s < 0.0 {
                    *s = 0.0;
                }
            }
            state
        })
        .collect()
}

/// Superposition of compartment states at each `subject.obs_times`
/// ([`analytical_state_at_times`]).
pub fn predict_all_states(
    pk_model: PkModel,
    subject: &Subject,
    pk_params: &PkParams,
) -> Vec<Vec<f64>> {
    analytical_state_at_times(pk_model, subject, pk_params, &subject.obs_times)
}

/// Compute predictions AND full compartment states for all observations:
/// `(ipred_vec, compartment_states_vec)`. ODE states come from the same integration;
/// analytical states from superposition. Analytical subjects with resets, and IOV models,
/// get empty states (→ NaN in `[derived]`, `W_DERIVED_CMT_RESET_ANALYTICAL`).
pub fn compute_predictions_with_states(
    model: &crate::types::CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> (Vec<f64>, Vec<Vec<f64>>) {
    // A transit subject the closed form can't serve routes to its ODE twin, so states
    // come from the ODE integration (matches `compute_predictions_with_tv`, #486).
    let model = effective_model_for_eval(model, subject, theta, eta);
    let uses_time = model_uses_time_anywhere(model);
    if let Some(ref ode) = model.ode_spec {
        // ODE path: TV-covariate and reset subjects use the event-driven pass; its states
        // are approximate where `ode_cmt_states_approximate` (`W_DERIVED_CMT_TV_ODE`, #1855).
        let pk = pk_params_at_time(model, theta, eta, &subject.covariates, 0.0);
        let (mut ipred, states) =
            if subject.has_tv_covariates() || subject.has_state_events() || uses_time {
                let mut scratch = EventPkParams::with_capacity_for(subject);
                compute_event_pk_params_into(model, subject, theta, eta, &mut scratch);
                crate::ode::ode_predictions_event_driven_with_states(
                    ode,
                    subject,
                    theta,
                    eta,
                    &scratch.dose,
                    &scratch.obs,
                    &scratch.pk_only,
                    &scratch.reset,
                )
            } else {
                // Single-pass: one ODE integration yields both ipred and states.
                crate::ode::ode_predictions_with_states(ode, &pk.values, theta, eta, subject)
            };
        // The sink's tail ([scaling], LTBS, FREM override), as in
        // compute_predictions_with_tv_into_with_schedule (#2259).
        to_observation_scale(model, subject, theta, eta, &mut ipred);
        (ipred, states)
    } else {
        // Analytical path: ipred via compute_predictions_with_tv (handles SS, resets,
        // TV covariates); states via predict_all_states (superposition only — valid
        // for the no-reset, no-TV case).
        let ipred = compute_predictions_with_tv(model, subject, theta, eta);
        let states = if model.is_algebraic() {
            // Compartment-free model (#811): no compartments → NaN, as for reset/TV/IOV.
            vec![]
        } else if !model.analytical_init.is_empty() {
            // [initial_conditions] baseline (#521): superposition states don't seed it, so
            // return NaN rather than disagree with ipred (W_DERIVED_INIT_ANALYTICAL).
            vec![]
        } else if subject.has_resets() {
            // Superposition is invalid across resets; return outer-empty vec so
            // compartments[i] → NaN via the .unwrap_or(&[]) fallback. Consistent
            // with the IOV convention. W_DERIVED_CMT_RESET_ANALYTICAL explains why.
            vec![]
        } else if subject.has_tv_covariates() || uses_time {
            // Superposition would use baseline pk_params while ipred honours per-event
            // TV parameters — silently wrong states. Outer-empty → NaN, consistent
            // with IOV and reset. W_DERIVED_CMT_TV_ANALYTICAL explains why.
            vec![]
        } else {
            // Reached only when !uses_time (guarded above), so t=0 is exact.
            let pk = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);
            // Resolve modeled-`RATE` doses (#394) before the superposition states.
            let resolved = crate::dosing::resolve_subject_doses(
                subject,
                model.active_dose_attr_map(),
                &pk.values,
            );
            if dose_needs_event_walk(model.pk_model, &resolved) {
                // `single_dose_states` is cmt-blind (no depot infusion, #400; bolus always
                // into cmt 1), and the event walk has no states variant: NaN rather than
                // wrong (#375).
                vec![]
            } else {
                predict_all_states(model.pk_model, &resolved, &pk)
            }
        };
        (ipred, states)
    }
}

/// Compute predictions for all observation times of a subject.
/// Uses analytical equations for standard PK models, or delegates to ODE solver
/// when an OdeSpec is provided.
pub fn compute_predictions(pk_model: PkModel, subject: &Subject, pk_params: &PkParams) -> Vec<f64> {
    compute_predictions_recycle(pk_model, subject, pk_params, Vec::new())
}

/// Allocation-reusing counterpart of [`compute_predictions`]: the static superposition
/// path overwrites `out` in place.
fn compute_predictions_recycle(
    pk_model: PkModel,
    subject: &Subject,
    pk_params: &PkParams,
    mut out: Vec<f64>,
) -> Vec<f64> {
    // Modeled-RATE doses (RATE=-2) must be resolved to `Fixed` before this closed form
    // (#324/#394, #898); unresolved they would silently be 0-rate "infusions". A real
    // `assert!` so release builds fail loudly too.
    assert!(
        subject.all_doses_fixed(),
        "modeled-RATE dose reached the analytical predictor unresolved \
         (resolve via dose_attr_map, or validate with check_model_data, before predicting)"
    );
    // Superposition cannot express a reset (EVID=3/4) or a dose into a compartment the
    // closed forms cannot target (`dose_needs_event_walk`, #400/#375): route those through
    // the event-driven walk with the constant `pk_params` in every slot.
    if (subject.has_resets() || dose_needs_event_walk(pk_model, subject))
        && event_driven::supports_event_driven(pk_model)
    {
        let pk_dose = vec![*pk_params; subject.doses.len()];
        let pk_obs = vec![*pk_params; subject.obs_times.len()];
        let pk_pk_only = vec![*pk_params; subject.pk_only_times.len()];
        return event_driven::event_driven_predictions(
            pk_model,
            subject,
            &pk_dose,
            &pk_obs,
            &pk_pk_only,
        );
    }
    out.clear();
    out.extend(
        subject
            .obs_times
            .iter()
            .map(|&t| predict_concentration(pk_model, &subject.doses, t, pk_params)),
    );
    out
}

/// True when any dose targets a compartment **dose superposition** cannot express, so the
/// subject must use the event-driven walk. [`single_dose_concentration`] ignores
/// `dose.cmt`: a bolus lands in compartment 1 (depot or IV central), an infusion in central
/// (cmt 2 oral, 1 IV, #350). Anything else (depot infusion #400, any peripheral) is wrong
/// there. `CMT=0` resolves to 1.
///
/// The walk matches NONMEM to <1e-8 on these cases
/// (`tests/nonmem_dose_compartment_anchor.rs`) where superposition was off by up to 100×
/// (#375). **Single source of truth** for every consumer (prediction, states, `[derived]`
/// grid, Form-C gate, `provider::subject_routes_to_event_walk`,
/// `W_DERIVED_CMT_ORAL_DEPOT_INFUSION_ANALYTICAL`).
pub(crate) fn dose_needs_event_walk(pk_model: PkModel, subject: &Subject) -> bool {
    // Transit/IG cannot be state-propagated, so there is no walk to route to.
    // They reject non-depot doses up front instead (`check_absorption_dosing` /
    // `check_absorption_closed_form_support`).
    if !event_driven::supports_event_driven(pk_model) {
        return false;
    }
    let central_cmt = if pk_model.is_oral() { 2 } else { 1 };
    subject.doses.iter().any(|d| {
        // `CMT=0` == the default dose compartment == 1.
        let cmt = d.cmt_1based();
        if d.is_infusion() {
            cmt != central_cmt
        } else {
            cmt != 1
        }
    })
}

/// True when the analytical PK predictor must produce a value for this subject: it has at
/// least one Gaussian observation (CMT not a registered non-Gaussian endpoint). pk-only
/// records do not count (#1809).
///
/// When `false` on an analytical model nothing consumes the walk's output: an analytical
/// TTE hazard is closed-form (concentration hazards become ODE `__chz` states), and
/// binary/categorical predictors ([`crate::types::LinearPredictorFn`]) cannot read the
/// concentration. `validation::population_uses_analytical_pk` is `any` of this predicate;
/// the validator still range-checks every dose, this gate skips the walk (#905).
pub(crate) fn subject_feeds_analytical_pk(model: &CompiledModel, subject: &Subject) -> bool {
    #[cfg(feature = "survival")]
    {
        // No non-Gaussian endpoints ⇒ every observation is Gaussian by construction.
        if model.endpoints.is_empty() {
            return true;
        }
        subject
            .obs_cmts
            .iter()
            .any(|cmt| !model.endpoints.contains_key(cmt))
    }
    #[cfg(not(feature = "survival"))]
    {
        // Without the `survival` feature no non-Gaussian endpoint can be parsed, so
        // every observation is Gaussian and the predictor is always live.
        let _ = (model, subject);
        true
    }
}

/// Compute predictions using ODE integration. `pk_params_flat` is the flat RHS parameter
/// vector; `theta`/`eta` feed `OdeSpec::output_fn` for Form C (empty slices otherwise).
pub fn compute_predictions_ode(
    ode_spec: &crate::ode::OdeSpec,
    subject: &Subject,
    pk_params_flat: &[f64],
    theta: &[f64],
    eta: &[f64],
) -> Vec<f64> {
    // System resets (EVID=3/4) need the event-driven ODE walker; replicate the constant
    // params across every event slot.
    if subject.has_state_events() {
        let mut pk = PkParams::default();
        let n = pk_params_flat.len().min(crate::types::MAX_PK_PARAMS);
        pk.values[..n].copy_from_slice(&pk_params_flat[..n]);
        let pk_dose = vec![pk; subject.doses.len()];
        let pk_obs = vec![pk; subject.obs_times.len()];
        let pk_pk_only = vec![pk; subject.pk_only_times.len()];
        let pk_reset = vec![pk; subject.reset_times.len()];
        return crate::ode::ode_predictions_event_driven(
            ode_spec,
            subject,
            theta,
            eta,
            &pk_dose,
            &pk_obs,
            &pk_pk_only,
            &pk_reset,
        );
    }
    crate::ode::ode_predictions(ode_spec, pk_params_flat, theta, eta, subject)
}

/// Apply the FREM prediction override in place: each FREMTYPE > 0 observation becomes
/// `theta[k] + eta[m]` (indices from `frem_predictions`). No-op without `[frem]`. `eta` is
/// the BSV vector (IOV callers pass the BSV slice). Shared by the analytical and IOV/SAEM
/// prediction paths.
pub(crate) fn apply_frem_prediction_override(
    model: &crate::types::CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    preds: &mut [f64],
) {
    if let Some(ref fc) = model.frem_config {
        for (j, ft) in subject.fremtype.iter().enumerate() {
            if *ft > 0 {
                if let Some(&(theta_idx, eta_idx)) = fc.fremtype_to_indices.get(ft) {
                    if j < preds.len() {
                        preds[j] = theta[theta_idx] + eta[eta_idx];
                    }
                }
            }
        }
    }
}

/// Per-observation predictions for one subject at `(theta, eta)` — the unchecked
/// prediction kernel under [`crate::predict`], [`crate::fit`] and the simulators.
///
/// # No model/data checks
///
/// This kernel (and its [`compute_predictions_with_tv_into`] /
/// [`compute_predictions_with_tv_into_with_schedule`] variants) runs **none** of the
/// public entry points' checks: a name in
/// [`CompiledModel::referenced_covariates`](crate::types::CompiledModel::referenced_covariates)
/// absent from the subject's covariate data evaluates to `0.0` silently — including
/// `TAD`/`TAFD`/`MACHEPS` read in a readout or `[scaling]` (#1028). Run
/// [`crate::check_model_data`] first, or call [`crate::predict`] (`E_MISSING_COVARIATE`).
pub fn compute_predictions_with_tv(
    model: &crate::types::CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Vec<f64> {
    // Event snapshots allocate lazily, only on paths that materialize them.
    // Hot loops should reuse the buffer via `compute_predictions_with_tv_into`.
    let mut scratch = EventPkParams::default();
    compute_predictions_with_tv_into(model, subject, theta, eta, &mut scratch)
}

/// Same as [`compute_predictions_with_tv`] but writes per-event PK params into a
/// caller-owned scratch buffer, reused by hot loops across `eta` (static fast paths do not
/// touch it; an empty buffer is fine). Runs no model/data checks (see
/// [`compute_predictions_with_tv`]).
pub fn compute_predictions_with_tv_into(
    model: &crate::types::CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    scratch: &mut EventPkParams,
) -> Vec<f64> {
    compute_predictions_with_tv_into_with_schedule(model, subject, theta, eta, scratch, None)
}

/// Hot-path variant that also accepts a pre-built [`event_driven::EventSchedule`],
/// reused on the TV-cov event-driven paths (saving the merged-event sort and infusion
/// bounds per call). Ignored on the static fast path; `None` builds one on demand. Runs
/// no model/data checks (see [`compute_predictions_with_tv`]).
pub fn compute_predictions_with_tv_into_with_schedule(
    model: &crate::types::CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    scratch: &mut EventPkParams,
    schedule: Option<&event_driven::EventSchedule>,
) -> Vec<f64> {
    compute_predictions_with_tv_recycle_with_schedule(
        model,
        subject,
        theta,
        eta,
        scratch,
        schedule,
        Vec::new(),
    )
}

/// Inner-loop counterpart of [`compute_predictions_with_tv_into_with_schedule`]
/// that reuses the returned observation vector on allocation-sensitive repeated
/// evaluations of one subject.
pub(crate) fn compute_predictions_with_tv_recycle_with_schedule(
    model: &crate::types::CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    scratch: &mut EventPkParams,
    schedule: Option<&event_driven::EventSchedule>,
    mut recycled: Vec<f64>,
) -> Vec<f64> {
    // #905: an analytical subject with no Gaussian observation feeds the predictor nothing
    // (see `subject_feeds_analytical_pk`) and its dose compartment is unchecked, so the
    // walk could hit the unroutable-dose `panic!`. Decline with NaN, keyed on the primary
    // `ode_spec` before `effective_model_for_eval`, matching the validator exemption.
    if model.ode_spec.is_none() && !subject_feeds_analytical_pk(model, subject) {
        recycled.clear();
        recycled.resize(subject.obs_times.len(), f64::NAN);
        return recycled;
    }
    // A transit subject the closed form can't serve routes to its ODE twin (which ignores
    // the cached analytical `schedule`); every other model is unchanged (#486).
    let model = effective_model_for_eval(model, subject, theta, eta);
    let has_tv = subject.has_tv_covariates();
    let uses_time = model_uses_time_anywhere(model);

    let mut preds = if model.is_algebraic() {
        // Compartment-free model (#811): the readout is the prediction; these zeros are
        // overwritten by `apply_analytic_readout`.
        recycled.clear();
        recycled.resize(subject.obs_times.len(), 0.0);
        recycled
    } else if let Some(ref ode) = model.ode_spec {
        // ODE path. Resets (EVID=3/4) need the state-propagating event-driven
        // walker too, even without time-varying covariates — the plain
        // `ode_predictions` loop has no reset event.
        if has_tv || subject.has_state_events() || uses_time {
            compute_event_pk_params_into(model, subject, theta, eta, scratch);
            crate::ode::ode_predictions_event_driven(
                ode,
                subject,
                theta,
                eta,
                &scratch.dose,
                &scratch.obs,
                &scratch.pk_only,
                &scratch.reset,
            )
        } else if let Some(mr) = modified_release::mr_predictions(model, subject, theta, eta) {
            // Closed-form modified-release fast path (#860) for a static multi-route model
            // (#505/#856) with a canonical linear 1-/2-cpt disposition. Only in this static
            // branch: `identify_disposition`'s fixed-`p` probe is blind to `TIME`/`TAD`, so
            // the `model_uses_time_anywhere` guard excludes those (#1124). `None` falls
            // through to the ODE twin. Agreement with the twin is established elsewhere
            // (`modified_release::tests::reduces_to_ode_*`, `mr_matches_the_states_entry_point`
            // — `compute_predictions_with_states` deliberately has no MR branch — and
            // `modified_release::verify_against_ode_twin` in debug builds).
            mr
        } else {
            let pk = pk_params_at_time(model, theta, eta, &subject.covariates, 0.0);
            crate::ode::ode_predictions(ode, &pk.values, theta, eta, subject)
        }
    } else if absorption_walk::walk_eligible(model, subject) {
        // A walk-eligible transit/IG subject kept on the closed form: the absorption walk
        // serves it (#1560); the twin arm is a defensive fallback.
        compute_event_pk_params_into(model, subject, theta, eta, scratch);
        match absorption_walk::absorption_walk_predictions(
            model.pk_model,
            subject,
            &scratch.dose,
            &scratch.obs,
            &scratch.pk_only,
            None,
        ) {
            Some(preds) => preds,
            None => {
                let twin = model.effective_for(subject);
                return compute_predictions_with_tv_recycle_with_schedule(
                    twin, subject, theta, eta, scratch, None, recycled,
                );
            }
        }
    } else if (has_tv || subject.has_resets() || uses_time)
        && event_driven::supports_event_driven(model.pk_model)
    {
        compute_event_pk_params_into(model, subject, theta, eta, scratch);
        // Resolve modeled-`RATE` doses (#324/#394) with each dose's per-event PK snapshot.
        // Borrowed (no allocation) when all are `Fixed`.
        let resolved =
            crate::dosing::resolve_subject_doses_with(subject, model.active_dose_attr_map(), |k| {
                &scratch.dose[k].values
            });
        // A cached `EventSchedule` was built from the *unresolved* subject; reuse it only
        // when nothing was resolved.
        if let (std::borrow::Cow::Borrowed(_), Some(sched)) = (&resolved, schedule) {
            // Nothing was resolved (all doses already `Fixed`) → the cached schedule
            // is valid, reuse it.
            event_driven::event_driven_predictions_with_schedule(
                model.pk_model,
                &resolved,
                sched,
                &scratch.dose,
                &scratch.obs,
                &scratch.pk_only,
            )
        } else {
            // A modeled dose was resolved (or no cache) → rebuild the schedule from
            // the resolved subject (a cache built from the unresolved subject reads
            // `duration == 0`, and modeled duration is η-dependent anyway).
            event_driven::event_driven_predictions(
                model.pk_model,
                &resolved,
                &scratch.dose,
                &scratch.obs,
                &scratch.pk_only,
            )
        }
    } else {
        // No-TV fast path (or TV with unsupported model — see docstring).
        // The fast path is time-independent; the degraded unsupported-model
        // branch matches the existing single-snapshot TV behaviour.
        let pk = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);
        // Resolve any modeled-`RATE` doses (#394) before the closed-form math.
        let resolved =
            crate::dosing::resolve_subject_doses(subject, model.active_dose_attr_map(), &pk.values);
        compute_predictions_recycle(model.pk_model, &resolved, &pk, recycled)
    };

    // Analytical initial amounts (#521), before scaling so the divisor applies to the total.
    add_analytical_init(model, subject, theta, eta, &mut preds);

    // Analytic Form C readout (#650): replace the built-in central concentration
    // with `y = <expr>` before scaling/log-transform. No-op unless the model has
    // an `analytic_readout` (ODE Form C is applied inside `ode_predictions*`).
    apply_analytic_readout(model, subject, theta, eta, &mut preds);

    // `[scaling]` (single insertion point for every estimator; a no-op under Form C), then
    // LTBS and the FREM override: `to_observation_scale` is the one copy of this tail.
    to_observation_scale(model, subject, theta, eta, &mut preds);
    preds
}

#[cfg(test)]
#[path = "iov_scratch_tests.rs"]
mod iov_scratch_tests;

#[cfg(test)]
#[path = "parent_dose_attr_tests.rs"]
mod parent_dose_attr_tests;

#[cfg(test)]
mod tests {
    /// #1564: the IOV record memo reuses a parameter evaluation exactly when every input
    /// the program reads repeats — occasion, covariate map (by address), and the time unless
    /// the program cannot read `TIME` — and evaluates otherwise.
    #[test]
    fn record_param_memo_reuses_only_identical_inputs() {
        use std::cell::Cell;
        use std::collections::HashMap;
        let calls = Cell::new(0usize);
        let eval = |v: f64| {
            calls.set(calls.get() + 1);
            let mut p = PkParams::default();
            p.values[0] = v;
            p
        };
        let (a, b): (HashMap<String, f64>, HashMap<String, f64>) = (HashMap::new(), HashMap::new());
        let mut memo = RecordParamMemo {
            time_free: true,
            last: None,
            value: PkParams::default(),
        };
        assert_eq!(memo.get_ref(0, &a, 1.0, || eval(1.0)).values[0], 1.0);
        // Same occasion and map, another time, on a time-free program: reused.
        assert_eq!(memo.get_ref(0, &a, 2.0, || eval(9.0)).values[0], 1.0);
        assert_eq!(calls.get(), 1);
        // Another occasion, another map (even with equal contents): evaluated.
        assert_eq!(memo.get_ref(1, &a, 2.0, || eval(2.0)).values[0], 2.0);
        assert_eq!(memo.get_ref(1, &b, 2.0, || eval(3.0)).values[0], 3.0);
        assert_eq!(calls.get(), 3);
        // A program that reads `TIME` keys on the time too.
        let mut timed = RecordParamMemo {
            time_free: false,
            last: None,
            value: PkParams::default(),
        };
        timed.get_ref(0, &a, 1.0, || eval(4.0));
        assert_eq!(timed.get_ref(0, &a, 2.0, || eval(5.0)).values[0], 5.0);
        assert_eq!(timed.get_ref(0, &a, 2.0, || eval(6.0)).values[0], 5.0);
        assert_eq!(calls.get(), 5);
    }

    use super::*;
    use approx::assert_relative_eq;
    use std::collections::HashMap;

    fn bolus_dose(time: f64, amt: f64) -> DoseEvent {
        DoseEvent::new(time, amt, 1, 0.0, false, 0.0)
    }

    fn make_pk_params(cl: f64, v: f64) -> PkParams {
        let mut p = PkParams::default();
        p.values[0] = cl;
        p.values[1] = v;
        p
    }

    // ── records with the same `$PK` inputs share one evaluation ──

    fn covs(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
        pairs.iter().map(|&(k, v)| (k.to_string(), v)).collect()
    }

    /// Each row names the regression it catches: a key that leaves out a covariate the
    /// program reads, its group or its sign of zero shares two records that evaluate
    /// differently. A key that reads the map's unreferenced entries, or compares
    /// `NaN` by value, only loses sharing, and the rows for those pin that too.
    #[test]
    fn records_share_an_evaluation_only_when_every_pk_input_matches_bitwise() {
        let model =
            crate::parser::model_parser::parse_model_string(crate::test_models::ONE_CPT_IV_21)
                .expect("parses");
        assert_eq!(model.referenced_covariates, ["AGE", "WT"]);
        let base = covs(&[("WT", 70.0), ("AGE", 40.0)]);
        let maps = [
            /* 0 */ base.clone(),
            /* 1 equal values, another map */ base.clone(),
            /* 2 an unreferenced column differs */
            covs(&[("WT", 70.0), ("AGE", 40.0), ("FOO", 1.0)]),
            /* 3 */ covs(&[("WT", 95.0), ("AGE", 40.0)]),
            /* 4 back to the first, not adjacent */ base.clone(),
            /* 5 */ covs(&[("WT", f64::NAN), ("AGE", 40.0)]),
            /* 6 NaN matches NaN by bits */ covs(&[("WT", f64::NAN), ("AGE", 40.0)]),
            /* 7 */ covs(&[("WT", 0.0), ("AGE", 40.0)]),
            /* 8 -0 is not 0 */ covs(&[("WT", -0.0), ("AGE", 40.0)]),
            /* 9 WT missing is not WT = 0 */ covs(&[("AGE", 40.0)]),
            /* 10 */ covs(&[("AGE", 40.0)]),
            /* 11 the other covariate differs */ covs(&[("WT", 70.0), ("AGE", 41.0)]),
        ];
        let mut records: Vec<(Option<usize>, &HashMap<String, f64>)> =
            maps.iter().map(|m| (None, m)).collect();
        // 12, 13: the first record's inputs in another κ group; 14: in no group again.
        records.extend([(Some(1), &maps[0]), (Some(1), &maps[1])]);
        records.push((None, &maps[1]));
        // 15, 16: one map borrowed by consecutive records, as `subject.covariates` is.
        records.extend([(None, &maps[3]), (None, &maps[3])]);
        assert_eq!(
            first_record_with_same_pk_inputs(&model, records.iter().copied()),
            Some(vec![0, 0, 0, 3, 0, 5, 5, 7, 8, 9, 9, 11, 12, 12, 0, 3, 3])
        );
    }

    /// The straddle: one program with and without its `TIME` read, on records whose
    /// covariates and times all agree. Without `TIME` they share; with it there is no
    /// index at all, so each is its own, even the two at the same time.
    #[test]
    fn a_model_reading_time_shares_nothing() {
        let src = |cl: &str| {
            format!(
                r#"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  sigma EPS ~ 0.1
[individual_parameters]
  CL = {cl}
  V  = 10 * (WT/70)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ additive(EPS)
"#
            )
        };
        let parse =
            |cl: &str| crate::parser::model_parser::parse_model_string(&src(cl)).expect("parses");
        let with_time = parse("TVCL * exp(ETA_CL) * (1 + 0.01 * TIME)");
        let without = parse("TVCL * exp(ETA_CL)");
        assert!(crate::parser::model_parser::compiled_model_uses_time_builtin(&with_time));
        assert!(!crate::parser::model_parser::compiled_model_uses_time_builtin(&without));
        let m = covs(&[("WT", 70.0)]);
        let records = [(None, &m), (None, &m), (Some(0), &m), (Some(0), &m)];
        assert_eq!(
            first_record_with_same_pk_inputs(&without, records.iter().copied()),
            Some(vec![0, 0, 2, 2])
        );
        assert_eq!(
            first_record_with_same_pk_inputs(&with_time, records.iter().copied()),
            None
        );
    }

    /// A hand-built `pk_param_fn` has no program whose inputs the key could list, so it
    /// is evaluated at every record however alike they look.
    #[test]
    fn a_model_without_an_individual_parameter_program_shares_nothing() {
        let model = cl_from_cr_model();
        assert!(model.indiv_param_partials.indiv_param_program.is_none());
        let m = covs(&[("CR", 1.0)]);
        let records = [(None, &m), (None, &m), (Some(0), &m)];
        assert_eq!(
            first_record_with_same_pk_inputs(&model, records.iter().copied()),
            None
        );
    }

    /// The per-event materializer copies a snapshot from the record it shares with, in
    /// whichever list that record sits. Every snapshot must still be the bits of `$PK`
    /// evaluated at that record's own inputs, which this compares against directly. The
    /// fixture shares within a list, across lists (a reset and an EVID=2 row reusing an
    /// observation's), and non-adjacently, and asserts that it does.
    #[test]
    fn shared_per_event_snapshots_are_each_records_own_evaluation_bitwise() {
        let model =
            crate::parser::model_parser::parse_model_string(crate::test_models::ONE_CPT_IV_21)
                .expect("parses");
        let w = |wt: f64| covs(&[("WT", wt), ("AGE", 40.0)]);
        let mut s = make_subject_with_tv(w(70.0), vec![w(70.0), w(95.0)], Vec::new(), 2, 5);
        s.obs_covariates = vec![w(70.0), w(95.0), w(95.0), w(80.0), w(70.0)];
        s.pk_only_times = vec![3.5, 4.5];
        s.pk_only_covariates = vec![w(80.0), w(60.0)];
        s.reset_times = vec![2.5];
        s.reset_covariates = vec![w(95.0)];
        assert!(subject_needs_per_event_pk(&model, &s));
        let theta = model.default_params.theta.clone();
        let eta = [0.2];

        let ev = compute_event_pk_params(&model, &s, &theta, &eta);
        let own = |cov: &HashMap<String, f64>, t: f64| (model.pk_param_fn)(&theta, &eta, cov, t);
        let lists: [(&str, &[PkParams], Vec<PkParams>); 4] = [
            (
                "dose",
                &ev.dose,
                (0..2)
                    .map(|k| own(s.dose_cov(k), s.dose_param_time(k)))
                    .collect(),
            ),
            (
                "obs",
                &ev.obs,
                (0..5)
                    .map(|j| own(s.obs_cov(j), s.param_time_obs(j)))
                    .collect(),
            ),
            (
                "pk_only",
                &ev.pk_only,
                (0..2)
                    .map(|m| own(s.pk_only_cov(m), s.param_time_pk_only(m)))
                    .collect(),
            ),
            (
                "reset",
                &ev.reset,
                vec![own(s.reset_cov(0), s.param_time_at(s.reset_times[0]))],
            ),
        ];
        for (name, got, want) in &lists {
            assert_eq!(got.len(), want.len(), "{name}");
            for (i, (g, w)) in got.iter().zip(want).enumerate() {
                assert_eq!(
                    g.values.map(f64::to_bits),
                    w.values.map(f64::to_bits),
                    "{name} {i}: the snapshot is not its own record's evaluation"
                );
            }
        }
        // The sharing the comparison relies on: across the obs, EVID=2 and reset rows
        // the materializer passes in one call, obs 4 reuses obs 0, EVID=2 row 0 reuses
        // obs 3 and the reset reuses obs 1. A distinct WT per record reduces this test
        // to the unshared one.
        let records: Vec<_> = (0..5)
            .map(|j| (None, s.obs_cov(j)))
            .chain((0..2).map(|m| (None, s.pk_only_cov(m))))
            .chain([(None, s.reset_cov(0))])
            .collect();
        assert_eq!(
            first_record_with_same_pk_inputs(&model, records),
            Some(vec![0, 1, 1, 3, 0, 3, 6, 1])
        );
        // And the shared snapshots differ, so a copy from the wrong record shows.
        assert_ne!(ev.obs[0].cl(), ev.obs[1].cl());
        assert_ne!(ev.obs[3].cl(), ev.pk_only[1].cl());
    }

    // ── the sharing index is built once per scope (#2558) ──

    /// `CL` reads `WT` and `V` reads `AGE` when `wt`; otherwise `WT` is no input at all.
    fn wt_age_model(wt: bool) -> CompiledModel {
        let cl = if wt {
            "TVCL * exp(ETA_CL) * (WT/70)"
        } else {
            "TVCL * exp(ETA_CL)"
        };
        crate::parser::model_parser::parse_model_string(&format!(
            r#"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma EPS ~ 0.1
[individual_parameters]
  CL = {cl}
  V  = TVV * (AGE/40)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ additive(EPS)
"#
        ))
        .expect("parses")
    }

    /// A subject whose `WT` changes from record to record over a constant `AGE`, with as
    /// many resets as doses, so its dose list and its reset list are the same length.
    fn wt_varying_subject(wts: [f64; 5]) -> Subject {
        let w = |wt: f64| covs(&[("WT", wt), ("AGE", 40.0)]);
        let mut s = make_subject_with_tv(
            w(wts[0]),
            vec![w(wts[0]), w(wts[1])],
            wts.iter().map(|&x| w(x)).collect(),
            2,
            5,
        );
        s.reset_times = vec![2.5, 3.5];
        s.reset_covariates = vec![w(wts[2]), w(wts[3])];
        s
    }

    /// Every snapshot of `ev`, as bits.
    fn snapshot_bits(ev: &EventPkParams) -> Vec<Vec<u64>> {
        [&ev.dose, &ev.obs, &ev.pk_only, &ev.reset]
            .iter()
            .flat_map(|l| l.iter().map(|p| p.values.map(f64::to_bits).to_vec()))
            .collect()
    }

    /// Outside a scope each evaluation builds both of its indices; inside one the first
    /// evaluation builds them and every later one reuses them, at any η, and the
    /// snapshots are the same bits either way. Catches an EBE search whose scope stops
    /// holding the index, and a reused index that is not the one a fresh build gives.
    #[test]
    fn a_sharing_scope_builds_each_index_once_and_changes_no_snapshot() {
        let model = wt_age_model(true);
        let s = wt_varying_subject([70.0, 95.0, 95.0, 80.0, 70.0]);
        assert!(subject_needs_per_event_pk(&model, &s));
        let theta = model.default_params.theta.clone();
        let builds = sharing_scope::builds;
        let eval = |eta: f64| snapshot_bits(&compute_event_pk_params(&model, &s, &theta, &[eta]));

        let b0 = builds();
        let unscoped = [eval(0.2), eval(-0.3)];
        assert_eq!(
            builds() - b0,
            4,
            "outside a scope: dose + rest index per evaluation"
        );
        let scoped = {
            let _share = sharing_scope::SharingScope::enter(&model, &s);
            let b1 = builds();
            let scoped = [eval(0.2), eval(-0.3), eval(0.2)];
            assert_eq!(builds() - b1, 2, "inside a scope: dose + rest index, once");
            scoped
        };
        assert_eq!(
            scoped[0], unscoped[0],
            "η = 0.2: a reused index moved a snapshot"
        );
        assert_eq!(
            scoped[1], unscoped[1],
            "η = -0.3: a reused index moved a snapshot"
        );
        assert_eq!(scoped[2], unscoped[0]);
        // Closed, the scope keeps nothing.
        let b2 = builds();
        eval(0.2);
        assert_eq!(
            builds() - b2,
            2,
            "a closed scope must not keep serving its index"
        );
    }

    /// The scope is for one model, one subject, one site and one set of κ groups, and an
    /// index for anything else is built and handed back but not kept, so the scope's own
    /// still answers afterwards. Each pair below differs in one of those and has a
    /// different index, asserted, so a key that drops it hands back the wrong one.
    #[test]
    fn a_sharing_scope_rebuilds_for_another_model_subject_site_or_grouping() {
        use sharing_scope::{first_record_with_same_pk_inputs as scoped, ShareSite};
        let (with_wt, age_only) = (wt_age_model(true), wt_age_model(false));
        let s = wt_varying_subject([70.0, 95.0, 95.0, 80.0, 70.0]);
        let other = wt_varying_subject([70.0, 70.0, 95.0, 95.0, 95.0]);
        let builds = sharing_scope::builds;
        fn none(s: &Subject) -> Vec<(Option<usize>, &HashMap<String, f64>)> {
            (0..s.obs_times.len())
                .map(|j| (None, s.obs_cov(j)))
                .collect()
        }
        let direct = |m: &CompiledModel, r: &[(Option<usize>, &HashMap<String, f64>)]| {
            first_record_with_same_pk_inputs(m, r.iter().copied()).expect("parsed, no TIME")
        };
        let get = |m: &CompiledModel, sub: &Subject, site, r: &[_]| {
            scoped(m, sub, site, r.iter().copied())
                .expect("parsed, no TIME")
                .to_vec()
        };
        let (rs, ro) = (none(&s), none(&other));
        // The same records in other κ groups: obs 1 and 2 now share only within a group.
        let grouped: Vec<_> = rs
            .iter()
            .enumerate()
            .map(|(j, &(_, c))| (Some(j % 2), c))
            .collect();
        let want = direct(&with_wt, &rs);
        assert_eq!(want, [0, 1, 1, 3, 0]);
        assert_ne!(
            direct(&age_only, &rs),
            want,
            "the models must index s differently"
        );
        assert_ne!(
            direct(&with_wt, &ro),
            want,
            "the subjects must index differently"
        );
        assert_ne!(
            direct(&with_wt, &grouped),
            want,
            "the groupings must index differently"
        );

        let _share = sharing_scope::SharingScope::enter(&with_wt, &s);
        assert_eq!(get(&with_wt, &s, ShareSite::Recorder, &rs), want);
        let b = builds();
        assert_eq!(get(&with_wt, &s, ShareSite::Recorder, &rs), want);
        assert_eq!(builds(), b, "the scope's own index is reused");

        assert_eq!(
            get(&age_only, &s, ShareSite::Recorder, &rs),
            direct(&age_only, &rs),
            "another model (as the evaluation twin is) builds its own index"
        );
        assert_eq!(
            get(&with_wt, &other, ShareSite::Recorder, &ro),
            direct(&with_wt, &ro),
            "another subject builds its own index"
        );
        assert_eq!(builds(), b + 2);
        let b = builds();
        assert_eq!(get(&with_wt, &s, ShareSite::Recorder, &rs), want);
        assert_eq!(builds(), b, "neither of those replaced the scope's index");

        // Another site's list, the same length with the same (absent) groups, as the
        // recorder's and the f64 materializer's lists can be on a model without IOV.
        assert_eq!(rs.len(), ro.len());
        assert_eq!(
            get(&with_wt, &s, ShareSite::ObsPkOnlyReset, &ro),
            direct(&with_wt, &ro),
            "another site keeps its own index"
        );
        let b = builds();
        assert_eq!(get(&with_wt, &s, ShareSite::Recorder, &rs), want);
        assert_eq!(builds(), b, "and leaves the recorder's alone");

        assert_eq!(
            get(&with_wt, &s, ShareSite::Recorder, &grouped),
            direct(&with_wt, &grouped),
            "other κ groups rebuild"
        );
    }

    /// A scope opened inside another hands the outer one back when it closes; one opened
    /// for the pair already open changes nothing. Catches a guard that leaves its own
    /// state behind, which costs the outer search its index for the rest of the search.
    #[test]
    fn a_nested_sharing_scope_restores_the_outer_one() {
        use sharing_scope::{first_record_with_same_pk_inputs as scoped, ShareSite, SharingScope};
        let model = wt_age_model(true);
        let (s, other) = (
            wt_varying_subject([70.0, 95.0, 95.0, 80.0, 70.0]),
            wt_varying_subject([70.0, 70.0, 95.0, 95.0, 95.0]),
        );
        let builds = sharing_scope::builds;
        let ask = |sub: &Subject| {
            let r: Vec<_> = (0..5).map(|j| (None, sub.obs_cov(j))).collect();
            scoped(&model, sub, ShareSite::Dose, r.into_iter());
        };
        let _outer = SharingScope::enter(&model, &s);
        ask(&s);
        let b = builds();
        {
            let _inner = SharingScope::enter(&model, &other);
            ask(&other);
            ask(&other);
            assert_eq!(builds(), b + 1, "the inner scope keeps its own index");
        }
        let b = builds();
        ask(&s);
        assert_eq!(
            builds(),
            b,
            "the outer scope's index survives the inner one"
        );
        {
            let _again = SharingScope::enter(&model, &s);
            ask(&s);
        }
        ask(&s);
        assert_eq!(builds(), b, "re-entering the open pair keeps its index");
    }

    /// A guard that closes while one opened after it is still open must not hand back its
    /// own predecessor: the later guard would then reinstall a scope whose borrows have
    /// ended. A debug build fails at the out-of-order drop, and every scope is closed, so a
    /// release build only loses the index. Catches a `Drop` that restores unconditionally.
    #[test]
    #[cfg(debug_assertions)]
    fn an_out_of_order_sharing_scope_closes_every_scope() {
        use sharing_scope::{first_record_with_same_pk_inputs as scoped, ShareSite, SharingScope};
        let model = wt_age_model(true);
        let (s, other) = (
            wt_varying_subject([70.0, 95.0, 95.0, 80.0, 70.0]),
            wt_varying_subject([70.0, 70.0, 95.0, 95.0, 95.0]),
        );
        let ask = |sub: &Subject| {
            let r: Vec<_> = (0..5).map(|j| (None, sub.obs_cov(j))).collect();
            scoped(&model, sub, ShareSite::Dose, r.into_iter());
        };
        let closed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let first = SharingScope::enter(&model, &s);
            let second = SharingScope::enter(&model, &other);
            drop(first);
            drop(second);
        }));
        let msg = closed.expect_err("an out-of-order drop must fail a debug build");
        assert!(
            msg.downcast_ref::<&str>()
                .is_some_and(|m| m.contains("closed before one opened after it")),
            "the panic names the out-of-order close"
        );
        // Nothing is left open: neither subject's index is kept.
        for sub in [&s, &other] {
            let b = sharing_scope::builds();
            ask(sub);
            ask(sub);
            assert_eq!(
                sharing_scope::builds(),
                b + 2,
                "no scope may outlive its guard"
            );
        }
    }

    /// An EBE search opens the scope: across all its evaluations each site builds once.
    /// Without it a search of two evaluations already builds four f64 indices.
    #[test]
    fn an_ebe_search_builds_each_sharing_index_once() {
        let model = wt_age_model(true);
        let mut s = wt_varying_subject([70.0, 95.0, 95.0, 80.0, 70.0]);
        s.observations = vec![9.0, 6.0, 4.0, 3.0, 2.0];
        let b = sharing_scope::builds();
        let ebe = crate::estimation::inner_optimizer::find_ebe(
            &model,
            &s,
            &model.default_params,
            200,
            1e-8,
            None,
            None,
            0,
        );
        assert!(ebe.nll.is_finite());
        // Exactly one per site, so a site the search stops reaching fails as surely as a site
        // that rebuilds. This closed form's objective values come from the compiled plan
        // (`plan_value_route`), not the f64 predictor, so the f64 dose and obs/PK-only/reset
        // indices are never built: the one left is the recorder's (measured: 3 before the
        // plan served the value-only objective, 1 after).
        let built = sharing_scope::builds() - b;
        assert_eq!(
            built, 1,
            "indices built across one EBE search: want the recorder's index only, once \
             (the plan serves the objective's values, so no f64 index is built)"
        );
    }

    // ── the doses-only materializer is the full one's `dose` vector (#1235) ──

    /// `compute_dose_pk_params_into` must be **bit-identical** to `compute_event_pk_params(..).dose`:
    /// it exists only to skip the observation snapshots a doses-only caller never reads. Both arms
    /// of the gate are exercised and the subjects straddle it (**asserted**, so this cannot become
    /// a one-arm test).
    ///
    /// The two callers share `fill_dose_pk_params`, so a mutation inside the shared body moves both
    /// sides together and bit-equality is blind to it (the tautological-twin shape). Measured:
    ///
    /// | mutation | this test | `tests/typical_value_snapshot_sets.rs` |
    /// |---|---|---|
    /// | shared body reads `&subject.covariates` | **dies** (on the `assert_ne!`, not the bit compare) | arm A + both `D`/`R` arms |
    /// | shared body reads `0.0` for the time | **survives** (a tautology on that axis) | arm B |
    /// | the two callers diverge (doses-only always freezes) | **dies** | arms A, B + both `D`/`R` |
    ///
    /// So the `assert_ne!` non-degeneracy check is the only reason row 1 is visible here, and the
    /// time axis is covered by arm B. This is the Tier-1 statement of the contract at the code's own
    /// site, where `cargo test --lib` reaches it.
    #[test]
    fn dose_only_snapshots_match_the_full_event_builder() {
        let src = r#"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL) * (WT/70)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let model = crate::parser::model_parser::parse_full_model(src)
            .expect("the parity fixture parses")
            .model;
        let theta = model.default_params.theta.clone();
        let eta = vec![0.1_f64; model.n_eta + model.n_kappa];

        let base = |doses: Vec<DoseEvent>| Subject {
            id: "1".into(),
            doses,
            obs_times: vec![1.0, 4.0, 26.0],
            observations: vec![1.0; 3],
            obs_cmts: vec![1; 3],
            covariates: HashMap::from([("WT".to_string(), 70.0)]),
            cens: vec![0; 3],
            ..Default::default()
        };

        let static_subject = base(vec![bolus_dose(0.0, 100.0), bolus_dose(24.0, 100.0)]);
        let mut tv_subject = base(vec![bolus_dose(0.0, 100.0), bolus_dose(24.0, 100.0)]);
        tv_subject.dose_covariates = vec![
            HashMap::from([("WT".to_string(), 70.0)]),
            HashMap::from([("WT".to_string(), 95.0)]),
        ];
        tv_subject.obs_covariates = vec![
            HashMap::from([("WT".to_string(), 70.0)]),
            HashMap::from([("WT".to_string(), 70.0)]),
            HashMap::from([("WT".to_string(), 95.0)]),
        ];

        // The straddle itself: without it a later change to the gate could quietly put
        // both subjects on one arm and this would still pass.
        assert!(
            !subject_needs_per_event_pk(&model, &static_subject),
            "the static subject must take the constant arm"
        );
        assert!(
            subject_needs_per_event_pk(&model, &tv_subject),
            "the TV-covariate subject must take the per-event arm"
        );

        let mut doses_only = Vec::new();
        for (arm, subject) in [("static", &static_subject), ("per-event", &tv_subject)] {
            let full = compute_event_pk_params(&model, subject, &theta, &eta);
            compute_dose_pk_params_into(&model, subject, &theta, &eta, &mut doses_only);
            assert_eq!(
                doses_only.len(),
                full.dose.len(),
                "{arm} arm: dose snapshot count must match the full builder"
            );
            for (k, (a, b)) in doses_only.iter().zip(full.dose.iter()).enumerate() {
                assert_eq!(
                    a.values.map(f64::to_bits),
                    b.values.map(f64::to_bits),
                    "{arm} arm: dose {k} snapshot differs from the full builder's"
                );
            }
        }

        // Non-degeneracy: on the per-event arm the two doses must actually *differ*,
        // or every bit comparison above is satisfied by a builder that froze the
        // snapshot — the exact defect #1235 is about.
        compute_dose_pk_params_into(&model, &tv_subject, &theta, &eta, &mut doses_only);
        assert_ne!(
            doses_only[0].values.map(f64::to_bits),
            doses_only[1].values.map(f64::to_bits),
            "the per-event fixture must resolve different params at its two doses"
        );
    }

    // ── analytical [initial_conditions] (issue #521) ──

    #[test]
    fn analytical_init_central_one_cpt_iv_is_bolus_decay() {
        // An initial amount A₀ in the central compartment of a 1-cpt model
        // decays as the IV-bolus impulse: C(t) = (A₀/V)·exp(-(CL/V)·t).
        let (cl, v, a0) = (2.0_f64, 10.0_f64, 500.0_f64);
        let p = make_pk_params(cl, v);
        for &t in &[0.0_f64, 1.0, 5.0, 24.0] {
            let got = analytical_init_concentration(PkModel::OneCptIv, 1, a0, t, &p);
            let expected = (a0 / v) * (-(cl / v) * t).exp();
            assert_relative_eq!(got, expected, max_relative = 1e-12);
        }
    }

    #[test]
    fn analytical_init_central_oral_uses_cmt2_and_ignores_ka() {
        // For an oral model, `central` is cmt 2 and the init is a direct IV
        // bolus into central — KA must not enter (the amount is already there,
        // not absorbed through the depot). So it matches the 1-cpt decay and is
        // independent of KA.
        let (cl, v, a0) = (3.0_f64, 20.0_f64, 250.0_f64);
        let mut p = make_pk_params(cl, v);
        p.values[crate::types::PK_IDX_KA] = 1.7;
        let mut p_other_ka = p;
        p_other_ka.values[crate::types::PK_IDX_KA] = 0.3;
        let t = 4.0;
        let got = analytical_init_concentration(PkModel::OneCptOral, 2, a0, t, &p);
        let got_other = analytical_init_concentration(PkModel::OneCptOral, 2, a0, t, &p_other_ka);
        let expected = (a0 / v) * (-(cl / v) * t).exp();
        assert_relative_eq!(got, expected, max_relative = 1e-12);
        assert_relative_eq!(got, got_other, max_relative = 1e-12);
    }

    #[test]
    fn analytical_init_depot_oral_absorbs_through_ka() {
        // An init amount pre-loaded in the depot (cmt 1) of an oral model is
        // absorbed first-order with F=1, i.e. the oral closed form with D=A₀.
        let (cl, v, ka, a0) = (3.0_f64, 20.0_f64, 1.1_f64, 100.0_f64);
        let mut p = make_pk_params(cl, v);
        p.values[crate::types::PK_IDX_KA] = ka;
        let t = 2.0;
        let got = analytical_init_concentration(PkModel::OneCptOral, 1, a0, t, &p);
        let expected = crate::sens::one_cpt::one_cpt_oral_g::<f64>(a0, t, cl, v, ka, 1.0);
        assert_relative_eq!(got, expected, max_relative = 1e-12);
        // At t=0 nothing has absorbed yet → zero central concentration.
        let at0 = analytical_init_concentration(PkModel::OneCptOral, 1, a0, 0.0, &p);
        assert_relative_eq!(at0, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn iv_bolus_and_infusion_apply_f_matching_nonmem_closed_form() {
        // NONMEM anchor for #327/#419. With bioavailability F1 on an IV dose,
        // NONMEM delivers F1·AMT, so for a 1-cpt model (ADVAN1/TRANS2,
        // `F1 = THETA(3)` in $PK) the closed forms are:
        //   bolus:    C(t) = F·Dose/V · exp(-k·t)
        //   infusion: rate held at R, duration scaled to T_F = F·AMT/R (#419):
        //             C(t) = R/CL · (1 − exp(-k·t))                   (t ≤ T_F)
        //             C(t) = R/CL · (1 − exp(-k·T_F)) · exp(-k·(t−T_F)) (t > T_F)
        // For a *rate-defined* infusion NONMEM keeps the rate and scales the
        // duration (total exposure still F·AMT), so there is no amplitude factor
        // on the rate. The analytical superposition path (`predict_concentration`)
        // must reproduce these.
        let (cl, v, f) = (5.0_f64, 50.0_f64, 0.4_f64);
        let k = cl / v;
        let mut pk = make_pk_params(cl, v);
        pk.values[crate::types::PK_IDX_F] = f;

        // IV bolus: F scales the amount (unchanged by #419).
        let amt = 100.0;
        let doses = vec![bolus_dose(0.0, amt)];
        for &t in &[0.25_f64, 1.0, 4.0, 12.0] {
            let got = predict_concentration(PkModel::OneCptIv, &doses, t, &pk);
            let want = f * amt / v * (-k * t).exp();
            assert_relative_eq!(got, want, max_relative = 1e-12);
        }

        // IV infusion: R=25, raw T = AMT/R = 4 h; under F the duration scales to
        // T_F = F·AMT/R = 1.6 h with the rate held at R.
        let rate = 25.0;
        let t_inf = f * amt / rate;
        let doses = vec![DoseEvent::new(0.0, amt, 1, rate, false, 0.0)];
        for &t in &[1.0_f64, 4.0, 8.0] {
            let got = predict_concentration(PkModel::OneCptIv, &doses, t, &pk);
            let want = if t <= t_inf {
                rate / cl * (1.0 - (-k * t).exp())
            } else {
                rate / cl * (1.0 - (-k * t_inf).exp()) * (-k * (t - t_inf)).exp()
            };
            assert_relative_eq!(got, want, max_relative = 1e-12);
        }
    }

    #[test]
    fn consumes_pk_slot_matches_solver() {
        // Pin `PkModel::consumes_pk_slot` — the source of truth for the parser's
        // unused-param warning (#309) — to what the analytical solvers actually
        // read: perturbing a *consumed* slot must change the predicted
        // concentration, and perturbing a *non-consumed* slot must leave it
        // bit-identical. A future variant whose closed form reads a new slot but
        // isn't reflected in `consumes_pk_slot`/`required_pk_params` fails here,
        // closing the drift gap between the parser table and the solvers.
        use crate::types::{
            PK_IDX_CL, PK_IDX_F, PK_IDX_KA, PK_IDX_LAGTIME, PK_IDX_Q, PK_IDX_Q3, PK_IDX_V,
            PK_IDX_V2, PK_IDX_V3,
        };
        let named_slots = [
            PK_IDX_CL,
            PK_IDX_V,
            PK_IDX_Q,
            PK_IDX_V2,
            PK_IDX_KA,
            PK_IDX_F,
            PK_IDX_Q3,
            PK_IDX_V3,
            PK_IDX_LAGTIME,
        ];
        let baseline = || {
            let mut p = PkParams::default();
            p.values[PK_IDX_CL] = 1.0;
            p.values[PK_IDX_V] = 10.0;
            p.values[PK_IDX_Q] = 0.7;
            p.values[PK_IDX_V2] = 20.0;
            p.values[PK_IDX_KA] = 1.3;
            p.values[PK_IDX_F] = 0.6;
            p.values[PK_IDX_Q3] = 0.4;
            p.values[PK_IDX_V3] = 30.0;
            p.values[PK_IDX_LAGTIME] = 0.5;
            p
        };
        let doses = vec![bolus_dose(0.0, 100.0)];
        let t = 2.0;
        for model in [
            PkModel::OneCptIv,
            PkModel::OneCptOral,
            PkModel::TwoCptIv,
            PkModel::TwoCptOral,
            PkModel::ThreeCptIv,
            PkModel::ThreeCptOral,
        ] {
            let c0 = predict_concentration(model, &doses, t, &baseline());
            assert!(
                c0 > 0.0,
                "{model:?}: baseline concentration should be positive"
            );
            for &slot in &named_slots {
                let mut perturbed = baseline();
                perturbed.values[slot] *= 5.0;
                let c1 = predict_concentration(model, &doses, t, &perturbed);
                if model.consumes_pk_slot(slot) {
                    assert!(
                        (c1 - c0).abs() > 1e-9,
                        "{model:?}: slot {slot} is marked consumed but perturbing it \
                         did not change the prediction (c0={c0}, c1={c1})"
                    );
                } else {
                    assert_eq!(
                        c1, c0,
                        "{model:?}: slot {slot} is marked unused but perturbing it \
                         changed the prediction"
                    );
                }
            }
        }
    }

    #[test]
    fn test_superposition_single_dose() {
        let doses = vec![bolus_dose(0.0, 1000.0)];
        let pk = make_pk_params(10.0, 100.0);
        let c = predict_concentration(PkModel::OneCptIv, &doses, 0.0, &pk);
        assert_relative_eq!(c, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_superposition_two_doses() {
        let doses = vec![bolus_dose(0.0, 1000.0), bolus_dose(10.0, 1000.0)];
        let pk = make_pk_params(10.0, 100.0);
        let k: f64 = 10.0 / 100.0;

        // At t=10, first dose has decayed, second dose just given
        let c = predict_concentration(PkModel::OneCptIv, &doses, 10.0, &pk);
        let expected = (1000.0_f64 / 100.0) * (-k * 10.0).exp() + 1000.0 / 100.0;
        assert_relative_eq!(c, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_superposition_ignores_future_doses() {
        let doses = vec![bolus_dose(0.0, 1000.0), bolus_dose(100.0, 1000.0)];
        let pk = make_pk_params(10.0, 100.0);

        // At t=5, second dose hasn't happened yet
        let c_single =
            predict_concentration(PkModel::OneCptIv, &[bolus_dose(0.0, 1000.0)], 5.0, &pk);
        let c_two = predict_concentration(PkModel::OneCptIv, &doses, 5.0, &pk);
        assert_relative_eq!(c_single, c_two, epsilon = 1e-12);
    }

    /// #1133, the occasion-label route. NONMEM anchor J (`nonmem_anchor/reset_init_snapshot_J.ctl`):
    /// with `WT` flat and only `OCC` moving, `$PK` at the reset row runs under **that row's own
    /// occasion** (42.0, occasion 2), not the preceding record's (14.0, occasion 1). The ferx twin
    /// of that anchor reaches `OCC` as a covariate (κ is estimated); this test drives `predict_iov`
    /// with explicit κ so the occasion LABEL selects the seed. The record before the reset is
    /// `OCC = 1` and the reset row `OCC = 2`, so any rule inferring the occasion from the PRECEDING
    /// record gives 1 (what ferx did before) and only reading `reset_occasions` gives 2. The
    /// candidate seeds carry distinct κ (occasion 2: ln 3, occasion 1: ln 2, none: 0).
    #[test]
    fn iov_reset_takes_its_own_rows_occasion() {
        use crate::parser::model_parser::parse_model_string;
        let model = parse_model_string(
            "
[parameters]
  theta TVCL(5.0, 0.01, 100.0)
  theta TVV(50.0, 0.1, 1000.0)
  theta TVBASE(10.0, 0.01, 1000.0)
  omega ETA_CL ~ 0.0
  kappa KAPPA_B ~ 0.04
  sigma PROP ~ 0.01
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  BASE = TVBASE * exp(KAPPA_B)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central)  = BASE
  d/dt(central)  = -(CL/V) * central
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP)
",
        )
        .expect("fixture parses");

        let mut subj = Subject {
            id: "1".to_string(),
            obs_times: vec![1.0, 9.0],
            observations: vec![0.0; 2],
            obs_cmts: vec![1; 2],
            reset_times: vec![8.0],
            // The reset row itself is occasion 2 — the occasion it STARTS. The record
            // before it is occasion 1, which is what every neighbour-inferring rule
            // resolves to, and what ferx used before this was measured.
            reset_occasions: vec![2],
            cens: vec![0; 2],
            // t=1 is occasion 1, t=9 (after the reset) is occasion 2, so both are kappa
            // groups through an observation. (An occasion named only on a reset row is a
            // group too since #1153 — the end of this test.)
            occasions: vec![1, 2],
            ..Default::default()
        };

        let groups = crate::stats::likelihood::iov_occasion_groups(&subj);
        assert_eq!(groups.len(), 2, "occasions 1 and 2 must both be groups");
        let theta = model.default_params.theta.clone();
        let eta_bsv = vec![0.0];
        // Occasion 1 gets κ = ln 2, occasion 2 gets κ = ln 3.
        let k_of = |occ: u32| groups.iter().position(|(id, _)| *id == occ).expect("group");
        let mut kappas = vec![vec![0.0]; groups.len()];
        kappas[k_of(1)] = vec![2.0f64.ln()];
        kappas[k_of(2)] = vec![3.0f64.ln()];

        let own = predict_iov(&model, &subj, &theta, &eta_bsv, &kappas)[1];

        // Drop the row's own label: an unlabelled reset seeds under κ = 0 (#1153), not
        // under the preceding record's occasion 1 — a guess from a neighbour would give
        // `own / unlabelled = 1.5` here.
        subj.reset_occasions.clear();
        let unlabelled = predict_iov(&model, &subj, &theta, &eta_bsv, &kappas)[1];

        assert_relative_eq!(own / unlabelled, 3.0, epsilon = 1e-9);

        // An occasion ONLY the reset row carries (#1153) is a κ group of its own, so the
        // seed reads its κ: `BASE = 10·exp(κ₇)` and nothing else moves, so
        // `y(9) = 10·5·e^{−0.1}/50` exactly, against `10·e^{−0.1}/50` under κ = 0.
        subj.reset_occasions = vec![7];
        let groups = crate::stats::likelihood::iov_occasion_groups(&subj);
        assert_eq!(
            groups.iter().map(|g| g.0).collect::<Vec<_>>(),
            vec![1, 2, 7]
        );
        let mut kappas = vec![vec![0.0]; 3];
        kappas[0] = vec![2.0f64.ln()];
        kappas[2] = vec![5.0f64.ln()];
        let reset_only = predict_iov(&model, &subj, &theta, &eta_bsv, &kappas)[1];
        assert_relative_eq!(
            reset_only,
            10.0 * 5.0 * (-0.1f64).exp() / 50.0,
            epsilon = 1e-9
        );
    }

    /// Records at 0 (dose, occasion 1), 2 (pk-only), 5 (obs, occasion 1) and 8 (obs,
    /// occasion 2). Occasion 2 is a κ group through the t = 8 observation.
    fn pk_only_occasion_subject(pk_only_occasions: Vec<u32>) -> Subject {
        Subject {
            id: "1".into(),
            doses: vec![bolus_dose(0.0, 1000.0)],
            dose_occasions: vec![1],
            obs_times: vec![5.0, 8.0],
            observations: vec![1.0; 2],
            obs_cmts: vec![1; 2],
            occasions: vec![1, 2],
            cens: vec![0; 2],
            pk_only_times: vec![2.0],
            pk_only_raw_times: Vec::new(),
            reset_period_times: Vec::new(),
            pk_only_covariates: vec![HashMap::new()],
            pk_only_occasions,
            ..Default::default()
        }
    }

    /// #1876 / #1153: a pk-only row resolves to its own label, and to `None` (κ = 0)
    /// without one — never to a neighbouring record's, though the t = 0 dose (occasion 4)
    /// sits right before it.
    #[test]
    fn pk_only_row_occasion_is_its_own_label_or_none() {
        assert_eq!(
            pk_only_row_occasion(&pk_only_occasion_subject(vec![7]), 0),
            Some(7)
        );
        let mut s = pk_only_occasion_subject(Vec::new());
        s.dose_occasions = vec![4];
        assert_eq!(pk_only_row_occasion(&s, 0), None);
        // A label is not a group. `None` is the κ = 0 row in every provider.
        let occ_to_k = HashMap::from([(1u32, 0usize), (2, 1)]);
        let s = pk_only_occasion_subject(vec![2]);
        assert_eq!(pk_only_row_group(&s, 0, &occ_to_k), Some(1));
        let s = pk_only_occasion_subject(vec![9]);
        assert_eq!(pk_only_row_group(&s, 0, &occ_to_k), None);
    }

    /// #1153: a reset row and a dose resolve to their own label, and to `None` (κ = 0)
    /// without one. The reset at 3 sits right after a pk-only row labelled 3 and the dose
    /// is labelled 1, so a neighbour guess would give `Some(3)` or `Some(1)`; a dose with no
    /// label used to fall back to occasion `0`, which here names a real group.
    #[test]
    fn reset_and_dose_occasion_are_their_own_label_or_none() {
        let mut s = pk_only_occasion_subject(vec![3]);
        s.reset_times = vec![3.0];
        s.reset_occasions = vec![5];
        assert_eq!(reset_row_occasion(&s, 0), Some(5));
        let occ_to_k = HashMap::from([(0u32, 0usize), (1, 1), (3, 2), (5, 3)]);
        assert_eq!(reset_row_group(&s, 0, &occ_to_k), Some(3));
        s.reset_occasions.clear();
        assert_eq!(reset_row_occasion(&s, 0), None);
        assert_eq!(reset_row_group(&s, 0, &occ_to_k), None);

        assert_eq!(dose_row_occasion(&s, 0), Some(1));
        assert_eq!(dose_row_group(&s, 0, &occ_to_k), Some(1));
        s.dose_occasions.clear();
        assert_eq!(dose_row_occasion(&s, 0), None);
        assert_eq!(dose_row_group(&s, 0, &occ_to_k), None);
    }

    /// #1876 on the value: the pk-only row at `t_p = 2` governs `(0, 2]` under ITS
    /// occasion, so the t = 5 observation is `D/V·exp(−k₂·t_p − k₁·(5 − t_p))`, with
    /// `k_i = CL·exp(κ_i)/V`. κ₁ = ln 2 and κ₂ = ln 3, so the three conventions differ:
    ///
    /// | pk-only row under | `(0, 2]` rate | pred(5) |
    /// |---|---|---|
    /// | its own occasion: 2, or 9 that only it carries (κ₉ = ln 3) | 0.3 | 20·e^−1.2 |
    /// | κ = 0 (ferx before #1876) | 0.1 | 20·e^−0.8 |
    /// | the record before it (occasion 1) | 0.2 | 20·e^−1.0 |
    ///
    /// Run on the closed form and on an `[odes]` twin, which share
    /// `fill_iov_record_params` and nothing downstream of it.
    #[test]
    fn iov_pk_only_row_runs_under_its_own_occasion() {
        use crate::parser::model_parser::parse_model_string;
        let header = "
[parameters]
  theta TVCL(5.0, 0.01, 100.0)
  theta TVV(50.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.04
  kappa KAPPA_CL ~ 0.04
  sigma PROP ~ 0.01
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV
[structural_model]
";
        let tail = "
[error_model]
  DV ~ proportional(PROP)
";
        let analytical =
            parse_model_string(&format!("{header}  pk one_cpt_iv(cl=CL, v=V)\n{tail}"))
                .expect("analytical fixture parses");
        let ode = parse_model_string(&format!(
            "{header}  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  obs_scale = V
{tail}"
        ))
        .expect("ODE fixture parses");
        let own = 20.0 * (-1.2f64).exp();
        let kappa0 = 20.0 * (-0.8f64).exp();
        let previous = 20.0 * (-1.0f64).exp();
        // Measured worst relative errors (macOS arm64): analytical 2.0e-16; ODE 1.03e-5,
        // the same order in all three conventions, i.e. RK45 at the default `reltol = 1e-4`
        // and not the switch at t = 2. 1e-4 keeps ~10x headroom; the straddle needs 0.22.
        for (name, model, tol) in [("analytical", &analytical, 1e-12), ("ode", &ode, 1e-4)] {
            let theta = model.default_params.theta.clone();
            let pred = |labels: Vec<u32>| {
                let s = pk_only_occasion_subject(labels);
                let groups = crate::stats::likelihood::iov_occasion_groups(&s);
                let mut kappas = vec![vec![0.0]; groups.len()];
                // Occasion 9 is carried by the pk-only row alone, when it is so labelled.
                for (occ, kappa) in [(1, 2.0f64.ln()), (2, 3.0f64.ln()), (9, 3.0f64.ln())] {
                    if let Some(k) = groups.iter().position(|(id, _)| *id == occ) {
                        kappas[k] = vec![kappa];
                    }
                }
                predict_iov(model, &s, &theta, &[0.0], &kappas)[0]
            };
            let got = pred(vec![2]);
            assert!(got.is_finite(), "{name}: {got}");
            eprintln!(
                "{name}: pred(5) relative error {:e}",
                (got - own).abs() / own
            );
            assert!(
                (got - own).abs() / own < tol,
                "{name}: pred(5) = {got}, own-occasion value {own}"
            );
            // The straddle: the conventions differ by 22% and 49% of `own`, far above `tol`.
            for (wrong, what) in [(kappa0, "kappa = 0"), (previous, "previous record")] {
                assert!(
                    (got - wrong).abs() / own > 0.1,
                    "{name}: indistinguishable from the {what} convention"
                );
            }
            // A label no dose or observation carries gets a κ group of its own, so the
            // row runs under κ₉ = ln 3, not κ = 0.
            let only_pk_only = pred(vec![9]);
            assert!(
                (only_pk_only - own).abs() / own < tol,
                "{name}: an occasion only the pk-only row carries gave {only_pk_only}, \
                 its own kappa gives {own} (kappa = 0 is {kappa0})"
            );
            // No label (an in-memory subject): κ = 0 (#1153), not the occasion-1 dose before
            // it — the straddle above keeps the two conventions apart.
            let unlabelled = pred(Vec::new());
            assert!(
                (unlabelled - kappa0).abs() / kappa0 < tol,
                "{name}: unlabelled row gave {unlabelled}, kappa = 0 is {kappa0} \
                 (previous record is {previous})"
            );
        }
    }

    /// #1133 on the adaptive (decision-window) IOV materialiser: a reset row's snapshot is *its own
    /// covariates* under the κ of the decision window active at its time, like every other record.
    ///
    /// With `CL = TVCL·(WT/100)·exp(ETA_CL + KAPPA_CL)`, decisions at 0 and 24, resets at `t = 5`
    /// (window 0, κ = 0) and `t = 30` (window 1, κ = ln 2), each candidate convention differs:
    ///
    /// | reset | own row | previous record | next record |
    /// |---|---|---|---|
    /// | `t = 5`  | **0.5** (WT 50) | 1.0 (WT 100) | 0.1 (WT 10) |
    /// | `t = 30` | **1.6** (WT 80, κ = ln 2) | 2.0 (WT 100) | 0.2 (WT 10) |
    ///
    /// and the baseline κ instead of the window's gives 0.8 for the second.
    #[test]
    fn iov_event_pk_seeds_each_reset_from_its_own_row_and_window() {
        use crate::parser::model_parser::parse_model_string;
        let model = parse_model_string(
            "
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.0
  kappa KAPPA_CL ~ 0.0
  sigma PROP ~ 0.01
[individual_parameters]
  CL = TVCL * (WT / 100.0) * exp(ETA_CL + KAPPA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP)
",
        )
        .expect("fixture parses");

        let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
        let subj = Subject {
            id: "1".to_string(),
            obs_times: vec![1.0, 40.0],
            observations: vec![0.0; 2],
            obs_cmts: vec![1; 2],
            covariates: wt(100.0),
            obs_covariates: vec![wt(100.0), wt(10.0)],
            reset_times: vec![5.0, 30.0],
            // Each reset row carries a `WT` shared with no neighbouring record, so reading
            // one row early or late shows up in the assertions below.
            reset_covariates: vec![wt(50.0), wt(80.0)],
            cens: vec![0; 2],
            ..Default::default()
        };

        let theta = model.default_params.theta.clone();
        let eta_baseline = vec![0.0, 0.0]; // [ETA_CL, KAPPA_CL], κ = 0
        let eta_occ = vec![
            vec![0.0, 0.0],                    // window 0 (decision at t=0): κ = 0
            vec![0.0, std::f64::consts::LN_2], // window 1 (decision at t=24): κ = ln 2
        ];
        let decisions = vec![0.0, 24.0];
        let ev =
            compute_event_pk_params_iov(&model, &subj, &theta, &eta_baseline, &eta_occ, &decisions);

        assert_eq!(ev.reset.len(), 2, "one snapshot per reset row");
        assert_relative_eq!(ev.reset[0].values[0], 0.5, epsilon = 1e-12);
        assert_relative_eq!(ev.reset[1].values[0], 1.6, epsilon = 1e-12);
    }

    #[test]
    fn compute_predictions_routes_reset_subject_to_event_driven() {
        // A subject with a reset (EVID=3) must NOT use superposition:
        // `compute_predictions` should zero the state at the reset. Dose at
        // t=0, reset at t=5, obs at t=1 (positive) and t=6 (zero).
        let subj = Subject {
            id: "1".to_string(),
            doses: vec![bolus_dose(0.0, 1000.0)],
            obs_times: vec![1.0, 6.0],
            observations: vec![0.0; 2],
            obs_cmts: vec![1; 2],
            reset_times: vec![5.0],
            cens: vec![0; 2],
            ..Default::default()
        };
        let pk = make_pk_params(10.0, 100.0);
        let preds = compute_predictions(PkModel::OneCptIv, &subj, &pk);
        assert!(preds[0] > 0.0);
        assert_relative_eq!(preds[1], 0.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "modeled-RATE dose reached the analytical predictor")]
    fn compute_predictions_panics_on_modeled_dose() {
        // #324 / review #2: the analytical superposition predictor must reject a
        // modeled-RATE (RATE=-2) dose loudly *in release too* — a real `assert!`,
        // not the old `debug_assert!` that compiled out and let a 0-rate "infusion"
        // silently produce 0/NaN. Direct call (bypassing the public entrypoints'
        // gates) proves the predictor itself fails fast on an unvalidated subject.
        use crate::types::RateMode;
        let mut subj = make_subject_with_tv(HashMap::new(), Vec::new(), Vec::new(), 0, 1);
        subj.doses = vec![DoseEvent::modeled(
            0.0,
            100.0,
            1,
            false,
            0.0,
            RateMode::ModeledDuration,
        )];
        let pk = make_pk_params(10.0, 100.0);
        let _ = compute_predictions(PkModel::OneCptIv, &subj, &pk);
    }

    fn make_subject_with_tv(
        cov_const: HashMap<String, f64>,
        dose_cov: Vec<HashMap<String, f64>>,
        obs_cov: Vec<HashMap<String, f64>>,
        n_doses: usize,
        n_obs: usize,
    ) -> Subject {
        Subject {
            id: "1".to_string(),
            doses: (0..n_doses).map(|i| bolus_dose(i as f64, 100.0)).collect(),
            obs_times: (0..n_obs).map(|i| 1.0 + i as f64).collect(),
            observations: vec![0.0; n_obs],
            obs_cmts: vec![1; n_obs],
            covariates: cov_const,
            dose_covariates: dose_cov,
            obs_covariates: obs_cov,
            cens: vec![0; n_obs],
            ..Default::default()
        }
    }

    /// Build a tiny analytical CompiledModel where CL = covariate `CR` *
    /// theta[0] (so we can prove pk_param_fn was called with the right
    /// covariate snapshot per event).
    fn cl_from_cr_model() -> crate::types::CompiledModel {
        use crate::types::{CompiledModel, ErrorModel, ModelParameters, OmegaMatrix, SigmaVector};
        CompiledModel {
            name: "cl_from_cr".into(),
            error_model: ErrorModel::Additive,
            error_spec: crate::types::ErrorSpec::Single(ErrorModel::Additive),
            pk_param_fn: Box::new(|theta, _eta, cov, _t: f64| {
                let mut p = PkParams::default();
                let cr = cov.get("CR").copied().unwrap_or(1.0);
                p.values[crate::types::PK_IDX_CL] = theta[0] * cr;
                p.values[crate::types::PK_IDX_V] = 50.0;
                p
            }),
            n_theta: 1,
            n_eta: 0,
            theta_names: vec!["TVCL".into()],
            eta_names: Vec::new(),
            omega_init_as_sd: Vec::new(),
            pk_indices: vec![],
            eta_map: vec![],
            pk_idx_f64: vec![],
            sel_flat: vec![],
            referenced_covariates: vec!["CR".into()],
            indiv_param_names: Vec::new(),
            ..crate::types::test_helpers::bare_model(ModelParameters {
                theta: vec![1.0],
                theta_names: vec!["TVCL".into()],
                theta_lower: vec![0.0],
                theta_upper: vec![f64::INFINITY],
                theta_fixed: vec![false],
                omega_fixed: Vec::new(),
                sigma: SigmaVector {
                    values: vec![0.1],
                    names: vec!["EPS".into()],
                },
                ..crate::types::test_helpers::bare_params(OmegaMatrix::from_diagonal(
                    &[],
                    Vec::new(),
                ))
            })
        }
    }

    #[test]
    fn test_event_pk_params_no_tv_evaluates_once() {
        // No TV covariates → uses subject.covariates; result is identical
        // for every event.
        let mut covs = HashMap::new();
        covs.insert("CR".to_string(), 2.0);
        let subj = make_subject_with_tv(covs, Vec::new(), Vec::new(), 2, 3);
        let model = cl_from_cr_model();
        let ev = compute_event_pk_params(&model, &subj, &[10.0], &[]);
        assert_eq!(ev.dose.len(), 2);
        assert_eq!(ev.obs.len(), 3);
        // CL = theta * CR = 10 * 2 = 20 everywhere.
        for p in ev.dose.iter().chain(ev.obs.iter()) {
            assert_relative_eq!(p.cl(), 20.0, epsilon = 1e-12);
        }
    }

    /// The parameter-static arm must still emit one `pk_only` snapshot per EVID=2 record. Both
    /// event-driven walkers (`ode/predictions.rs`, `pk/event_driven.rs`) open with the same three
    /// unconditional length asserts (`assert_eq!`, so release aborts too), so a short `pk_only`
    /// aborts the run. The arm used to leave `pk_only` empty while filling `dose` and `obs`. It is
    /// reachable when a *parameter-static* subject is routed to an event-driven walker: by
    /// `subject.has_resets()` (EVID 3/4, predating #1124) or, since #1124, an `[odes]` RHS that
    /// reads model time. Both need only EVID=2 rows, because `prune_irrelevant_tv_covariates`
    /// clears `pk_only_covariates` but leaves `pk_only_times` populated, the state where
    /// `subject_needs_per_event_pk` is false and `pk_only_times` is not.
    #[test]
    fn event_pk_params_fills_pk_only_on_the_parameter_static_arm() {
        let mut covs = HashMap::new();
        covs.insert("CR".to_string(), 2.0);
        let mut subj = make_subject_with_tv(covs, Vec::new(), Vec::new(), 2, 3);
        // An EVID=2 row that survived covariate pruning: a time, but no snapshot.
        subj.pk_only_times = vec![0.5, 2.5];
        subj.pk_only_covariates = Vec::new();
        let model = cl_from_cr_model();
        // Precondition: this is the static arm, not the per-event one. Without
        // this the test would still pass via the `if` branch and prove nothing.
        assert!(
            !subject_needs_per_event_pk(&model, &subj),
            "fixture must exercise the parameter-static arm"
        );
        let ev = compute_event_pk_params(&model, &subj, &[10.0], &[]);
        assert_eq!(ev.dose.len(), subj.doses.len());
        assert_eq!(ev.obs.len(), subj.obs_times.len());
        assert_eq!(
            ev.pk_only.len(),
            subj.pk_only_times.len(),
            "pk_only must carry one snapshot per EVID=2 record; both event-driven \
             walkers assert this length unconditionally"
        );
        // CL = theta * CR = 10 * 2 = 20 on every row, pk_only included.
        for p in ev.dose.iter().chain(&ev.obs).chain(&ev.pk_only) {
            assert_relative_eq!(p.cl(), 20.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_event_pk_params_tv_uses_per_event_snapshot() {
        // Per-event snapshots drive different CL at each event.
        let mk = |cr: f64| {
            let mut h = HashMap::new();
            h.insert("CR".to_string(), cr);
            h
        };
        let dose_cov = vec![mk(1.0), mk(1.5)];
        let obs_cov = vec![mk(1.0), mk(2.0)];
        let subj = make_subject_with_tv(mk(1.0), dose_cov, obs_cov, 2, 2);
        let model = cl_from_cr_model();
        let ev = compute_event_pk_params(&model, &subj, &[10.0], &[]);
        assert_relative_eq!(ev.dose[0].cl(), 10.0, epsilon = 1e-12);
        assert_relative_eq!(ev.dose[1].cl(), 15.0, epsilon = 1e-12);
        assert_relative_eq!(ev.obs[0].cl(), 10.0, epsilon = 1e-12);
        assert_relative_eq!(ev.obs[1].cl(), 20.0, epsilon = 1e-12);
    }

    #[test]
    fn test_event_pk_params_time_builtin_uses_event_times_without_tv_covariates() {
        let model_str = "
[parameters]
  theta TVCL(10.0, 0.001, 100.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.1

[individual_parameters]
  if (TIME > 0.5) {
    CL = (TVCL + TIME) * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }

[structural_model]
  pk one_cpt_iv(cl=CL, v=time)

[error_model]
  DV ~ additive(EPS)
";
        let model = crate::parser::model_parser::parse_model_string(model_str)
            .expect("TIME/time built-ins parse");
        assert!(
            !model
                .referenced_covariates
                .iter()
                .any(|c| c == "TIME" || c == "time"),
            "TIME/time must not be tracked as data covariates: {:?}",
            model.referenced_covariates
        );

        let subj = make_subject_with_tv(HashMap::new(), Vec::new(), Vec::new(), 2, 2);
        assert!(!subj.has_tv_covariates(), "fixture must stay on no-TV path");

        let ev = compute_event_pk_params(&model, &subj, &model.default_params.theta, &[0.0]);

        assert_relative_eq!(ev.dose[0].cl(), 10.0, epsilon = 1e-12); // TIME = 0
        assert_relative_eq!(ev.dose[1].cl(), 11.0, epsilon = 1e-12); // TIME = 1
        assert_relative_eq!(ev.obs[0].cl(), 11.0, epsilon = 1e-12); // TIME = 1
        assert_relative_eq!(ev.obs[1].cl(), 12.0, epsilon = 1e-12); // TIME = 2
        assert_relative_eq!(ev.dose[0].v(), 0.0, epsilon = 1e-12); // time = 0
        assert_relative_eq!(ev.obs[1].v(), 2.0, epsilon = 1e-12); // time = 2
    }

    #[test]
    fn test_compute_predictions_length() {
        let subject = Subject {
            id: "1".to_string(),
            doses: vec![bolus_dose(0.0, 1000.0)],
            obs_times: vec![1.0, 2.0, 4.0, 8.0],
            observations: vec![0.0; 4],
            obs_cmts: vec![1; 4],
            cens: vec![0; 4],
            ..Default::default()
        };
        let pk = make_pk_params(10.0, 100.0);
        let preds = compute_predictions(PkModel::OneCptIv, &subject, &pk);
        assert_eq!(preds.len(), 4);
        // Predictions should be monotonically decreasing for IV bolus
        for i in 1..preds.len() {
            assert!(preds[i] < preds[i - 1]);
        }
    }

    fn oral_pk_params(cl: f64, v: f64, ka: f64) -> PkParams {
        let mut p = PkParams::default();
        p.values[crate::types::PK_IDX_CL] = cl;
        p.values[crate::types::PK_IDX_V] = v;
        p.values[crate::types::PK_IDX_KA] = ka;
        p
    }

    #[test]
    fn test_predict_concentration_lagtime_zero_matches_baseline() {
        // With lagtime = 0, predict_concentration must agree with the
        // pre-feature behavior at any observation time.
        let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        let pk = oral_pk_params(2.0, 20.0, 1.5); // CL/V = 0.1, KA = 1.5
        let c0 = predict_concentration(PkModel::OneCptOral, &doses, 2.0, &pk);
        // Bateman: C(t) = (F*Dose*KA) / (V*(KA-k)) * [exp(-k*t) - exp(-KA*t)]
        let k = 2.0_f64 / 20.0;
        let expected =
            (1.0 * 100.0 * 1.5) / (20.0 * (1.5 - k)) * ((-k * 2.0).exp() - (-1.5 * 2.0_f64).exp());
        assert_relative_eq!(c0, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_predict_concentration_lagtime_shifts_curve() {
        // Concentration at t with lagtime=L should equal the unlagged
        // concentration evaluated at (t - L).
        let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        let mut pk_lag = oral_pk_params(2.0, 20.0, 1.5);
        pk_lag.values[crate::types::PK_IDX_LAGTIME] = 1.5;
        let pk_nolag = oral_pk_params(2.0, 20.0, 1.5);

        // At t = 3.5 with lag=1.5, effective elapsed time is 2.0.
        let c_lag = predict_concentration(PkModel::OneCptOral, &doses, 3.5, &pk_lag);
        let c_no = predict_concentration(PkModel::OneCptOral, &doses, 2.0, &pk_nolag);
        assert_relative_eq!(c_lag, c_no, epsilon = 1e-12);
    }

    #[test]
    fn test_predict_concentration_lagtime_before_first_obs_returns_zero() {
        // Lagtime longer than the observation window: nothing has reached
        // the system yet, so concentration is exactly 0.
        let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        let mut pk = oral_pk_params(2.0, 20.0, 1.5);
        pk.values[crate::types::PK_IDX_LAGTIME] = 5.0;
        let c = predict_concentration(PkModel::OneCptOral, &doses, 1.0, &pk);
        assert_eq!(c, 0.0);
    }

    /// A non-finite dose arrival must come back `NaN`, never a drug-free `0.0` (#1189's silent-drop
    /// shape, closed on this engine by #1284).
    ///
    /// `predict_concentration` sums closed forms over doses instead of walking a timeline, so it
    /// has no `times_have_non_finite` to run. Both its arms test `t_eff` against `t`, every
    /// comparison against `NaN` is false, so the dose was skipped and the remaining trajectory
    /// returned as valid. Also the trailing `conc.max(0.0)` is an `f64::max`, which discards `NaN`
    /// (`0.0f64.max(NaN) == 0.0`, verified below), so even a dose that contributed `NaN` was floored
    /// to `0.0`: two independent ways to read finite and wrong.
    ///
    /// Both inputs to `t_eff` are exercised (a `NaN` `lagtime` from an overflowing `$PK`
    /// expression; a `NaN` `dose.time` from `DoseEvent::non_finite`, a modeled `D{n}`/`R{n}` out of
    /// domain); the finite control pins that the guard did not start rejecting everything.
    ///
    /// Mutation (run): deleting the `superposition_arrival_non_finite` early return in
    /// `predict_concentration` makes both non-finite arms return `0.0`; in
    /// `analytical_state_at_times`, the state half fires instead.
    #[test]
    fn non_finite_arrival_gives_nan_not_a_drug_free_zero() {
        // The floor that would have hidden it — measured here rather than recalled.
        assert_eq!(f64::NAN.max(0.0), 0.0, "f64::max discards NaN");

        let pk = make_pk_params(1.0, 10.0);
        let finite = predict_concentration(PkModel::OneCptIv, &[bolus_dose(0.0, 100.0)], 4.0, &pk);
        assert!(
            finite.is_finite() && finite > 0.0,
            "control: a finite dose still predicts ({finite})"
        );

        let mut nan_lag = pk;
        nan_lag.values[crate::types::PK_IDX_LAGTIME] = f64::NAN;
        let via_lag =
            predict_concentration(PkModel::OneCptIv, &[bolus_dose(0.0, 100.0)], 4.0, &nan_lag);
        assert!(via_lag.is_nan(), "NaN lagtime: got {via_lag}, want NaN");

        let via_dose_time =
            predict_concentration(PkModel::OneCptIv, &[bolus_dose(f64::NAN, 100.0)], 4.0, &pk);
        assert!(
            via_dose_time.is_nan(),
            "NaN dose time: got {via_dose_time}, want NaN"
        );

        // The *other* finite dose in the same subject must not rescue it: the
        // subject is non-finite, not "finite where it happens to have data".
        let mixed = predict_concentration(
            PkModel::OneCptIv,
            &[bolus_dose(0.0, 100.0), bolus_dose(f64::NAN, 100.0)],
            4.0,
            &pk,
        );
        assert!(mixed.is_nan(), "one bad dose poisons the sum: got {mixed}");

        // …and the **state** twin must say the same thing about the same subject.
        // It ran its own copy of this loop and had its own zero floor, so with the
        // guard on one side only it reported a drug-free `[[0.0]]` next to the
        // `NaN` above — a valid-looking compartment amount for a repelled subject
        // (#1399 review finding 1).
        let subj = Subject {
            id: "nan-arrival".into(),
            doses: vec![bolus_dose(f64::NAN, 100.0)],
            obs_times: vec![4.0, 8.0],
            ..Default::default()
        };
        let states = analytical_state_at_times(PkModel::OneCptIv, &subj, &pk, &subj.obs_times);
        assert_eq!(states.len(), 2, "one state vector per time");
        for (k, s) in states.iter().enumerate() {
            assert!(
                s.iter().all(|v| v.is_nan()),
                "state twin at time {k}: got {s:?}, want all NaN"
            );
        }
        // Straddle: the same walk on a finite dose still returns real amounts.
        let ok_subj = Subject {
            id: "finite".into(),
            doses: vec![bolus_dose(0.0, 100.0)],
            obs_times: vec![4.0],
            ..Default::default()
        };
        let ok_states =
            analytical_state_at_times(PkModel::OneCptIv, &ok_subj, &pk, &ok_subj.obs_times);
        assert!(
            ok_states[0].iter().all(|v| v.is_finite()) && ok_states[0][0] > 0.0,
            "control state: {:?}",
            ok_states[0]
        );
    }

    /// A `NaN` that enters the superposition **through the closed form's own inputs**, not the
    /// arrival, must survive the trailing zero floor (#1399 review). `conc.max(0.0)` discarded the
    /// `NaN` and reported a confident `0.0`. A non-finite bioavailability `F` is the reachable route:
    /// the arrival is finite, so `superposition_arrival_non_finite` does **not** cover it (asserted
    /// below: two gates rejecting the same inputs would be a test hole). Before the fix an analytical
    /// `one_cpt_iv(.., f=F)` with `F` overflowed at typical values returned `predict() = [0.0]` while
    /// the state twin (whose `if *s < 0.0` floor keeps `NaN`) returned `[[NaN]]`.
    ///
    /// Mutation (run): restoring the bare `conc.max(0.0)` makes the `F` arm return `0.0`.
    #[test]
    fn a_nan_from_the_closed_form_survives_the_zero_floor() {
        let doses = [bolus_dose(0.0, 100.0)];
        let mut pk = make_pk_params(1.0, 10.0);
        pk.values[crate::types::PK_IDX_F] = f64::NAN;

        // The arrival is finite, so the other guard cannot be what catches this.
        assert!(
            !superposition_arrival_non_finite(&doses, pk.lagtime()),
            "the arrival guard must NOT cover the F case — otherwise this test is \
             passing on the strength of a different gate"
        );

        let got = predict_concentration(PkModel::OneCptIv, &doses, 4.0, &pk);
        assert!(got.is_nan(), "NaN F: got {got}, want NaN");

        // A negative concentration is still floored — the fix preserves `NaN`, it does
        // not remove the floor.
        let mut neg = make_pk_params(1.0, 10.0);
        neg.values[crate::types::PK_IDX_F] = -1.0;
        let floored = predict_concentration(PkModel::OneCptIv, &doses, 4.0, &neg);
        assert_eq!(floored, 0.0, "a negative concentration still floors to 0");
    }

    #[test]
    fn test_infusion_with_lagtime_shifts_window() {
        // 1-cpt infusion, amt=100, rate=100 → duration = amt/rate = 1.0
        // (the `DoseEvent::new` signature is `(time, amt, cmt, rate, ss, ii)`
        // — duration is auto-computed from amt/rate; the trailing 0.0 is `ii`).
        // Starting at t_dose=2, with lagtime=0.5, the infusion effectively
        // runs 2.5..3.5.
        let dose = DoseEvent::new(2.0, 100.0, 1, 100.0, false, 0.0);
        debug_assert!(
            dose.is_infusion() && dose.duration > 0.0,
            "test must exercise the infusion branch"
        );
        let doses = vec![dose.clone()];
        let mut pk = make_pk_params(10.0, 100.0);
        pk.values[crate::types::PK_IDX_LAGTIME] = 0.5;

        // (a) Before lagged start: still zero.
        let c_pre = predict_concentration(PkModel::OneCptIv, &doses, 0.6, &pk);
        assert_eq!(c_pre, 0.0);

        // (b) Mid-infusion (lagged): pre-lag conc at tau = t - (dose.time + lag) = t - 2.5.
        // Compare against unlagged infusion at tau via the same formula.
        let pk_nolag = make_pk_params(10.0, 100.0);
        let c_lag = predict_concentration(PkModel::OneCptIv, &doses, 2.6, &pk);
        let c_no = predict_concentration(
            PkModel::OneCptIv,
            &[DoseEvent::new(0.1, 100.0, 1, 100.0, false, 0.0)],
            0.2,
            &pk_nolag,
        );
        // Both probe an elapsed time of 0.1 into a 1h infusion of rate=100.
        assert_relative_eq!(c_lag, c_no, epsilon = 1e-10);

        // (c) Post-infusion (lagged window ends at 3.5).
        let c_post = predict_concentration(PkModel::OneCptIv, &doses, 3.6, &pk);
        let c_post_nolag = predict_concentration(
            PkModel::OneCptIv,
            &[DoseEvent::new(0.0, 100.0, 1, 100.0, false, 0.0)],
            1.1,
            &pk_nolag,
        );
        assert_relative_eq!(c_post, c_post_nolag, epsilon = 1e-10);
    }

    // --- Steady-state (SS=1) integration with predict_concentration ---

    #[test]
    fn test_ss_iv_bolus_via_predict_matches_closed_form() {
        // Sanity check: predict_concentration with an SS=1 dose dispatches to
        // the closed-form SS formula (not the single-dose response).
        let ii = 12.0_f64;
        let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, true, ii)];
        let pk = make_pk_params(10.0, 100.0);
        let k = 10.0_f64 / 100.0;
        for &t in &[0.0, 3.0, 11.9, 24.0] {
            let c = predict_concentration(PkModel::OneCptIv, &doses, t, &pk);
            let expected = (1000.0 / 100.0) * (-k * t).exp() / (1.0 - (-k * ii).exp());
            assert_relative_eq!(c, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_ss_with_lagtime_identity_c_t_l_equals_c_t_minus_l() {
        // Acceptance criterion from issue #15: with SS+lagtime, the curve at
        // time t with lagtime L equals the unlagged SS curve at t - L. This
        // is the geometric-series identity under linear superposition.
        let ii = 12.0;
        let lagtime = 1.5;
        let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, true, ii)];

        let mut pk_lag = make_pk_params(10.0, 100.0);
        pk_lag.values[crate::types::PK_IDX_LAGTIME] = lagtime;
        let pk_nolag = make_pk_params(10.0, 100.0);

        // Sample several t > lagtime so both curves are defined.
        for &t in &[2.0, 5.0, 11.0, 13.5] {
            let c_lag = predict_concentration(PkModel::OneCptIv, &doses, t, &pk_lag);
            let c_no = predict_concentration(PkModel::OneCptIv, &doses, t - lagtime, &pk_nolag);
            assert_relative_eq!(c_lag, c_no, epsilon = 1e-12);
        }
    }

    /// NONMEM cross-check for SS + ALAG1 (issue #15). Reference PRED values
    /// were generated with NONMEM 7.5.1 (`ADVAN2 TRANS2`, `$ESTIMATION
    /// MAXEVAL=0`) for a 1-cpt oral model: CL=2, V=20, KA=1.5, ALAG1=1.5,
    /// a single `SS=1, II=24, AMT=100` dose into the depot, observed in the
    /// central compartment (`S2=V`). Control file + dataset documented in
    /// `tests/ss_lagtime_nonmem.rs`.
    ///
    /// The two earliest samples (t=0.5, 1.0 < ALAG1) exercise the
    /// previous-interval steady-state tail: NONMEM reports ~0.59 / ~0.56
    /// there, not 0. This is the analytical-path coverage of the wrap added
    /// to `predict_concentration`.
    #[test]
    fn test_ss_oral_with_lagtime_matches_nonmem() {
        let (cl, v, ka, lagtime, ii, amt) = (2.0, 20.0, 1.5, 1.5, 24.0, 100.0);
        let doses = vec![DoseEvent::new(0.0, amt, 1, 0.0, true, ii)];
        let mut pk = oral_pk_params(cl, v, ka);
        pk.values[crate::types::PK_IDX_LAGTIME] = lagtime;

        // (observation time, NONMEM PRED). t < 1.5 is the previous-interval
        // tail; t >= 1.5 is the current interval; t > II decays (naked SS).
        let nonmem: &[(f64, f64)] = &[
            (0.5, 0.59069),
            (1.0, 0.56188),
            (2.0, 3.07370),
            (4.0, 4.46240),
            (8.0, 3.07540),
            (12.0, 2.06170),
            (18.0, 1.13150),
            (23.0, 0.68628),
            (25.0, 0.56188),
            (30.0, 0.34080),
        ];
        for &(t, pred) in nonmem {
            let c = predict_concentration(PkModel::OneCptOral, &doses, t, &pk);
            assert_relative_eq!(c, pred, max_relative = 1e-4);
        }
    }

    #[test]
    fn test_ss_dose_does_not_contribute_before_its_record_time() {
        // SS=1 establishes steady state *at* the dose record time, so an SS
        // dose must not contribute to observations earlier than its own
        // record (the pre-arrival wrap only applies in [dose.time, t_eff)).
        // Regression guard: before the `t >= dose.time` gate, a future SS
        // record leaked a non-zero steady-state tail into earlier times.
        let (cl, v, ka, lagtime, ii, amt) = (2.0, 20.0, 1.5, 1.5, 24.0, 100.0);
        // SS dose recorded at t = 10 (not at the origin).
        let doses = vec![DoseEvent::new(10.0, amt, 1, 0.0, true, ii)];
        let mut pk = oral_pk_params(cl, v, ka);
        pk.values[crate::types::PK_IDX_LAGTIME] = lagtime;

        // Strictly before the record time → no contribution.
        for &t in &[0.0, 5.0, 9.9] {
            let c = predict_concentration(PkModel::OneCptOral, &doses, t, &pk);
            assert_eq!(c, 0.0, "future SS dose leaked into t={t}");
        }
        // In the record-to-arrival window [10, 11.5) → previous-interval tail.
        let c_pre = predict_concentration(PkModel::OneCptOral, &doses, 10.5, &pk);
        let dose0 = DoseEvent::new(0.0, amt, 1, 0.0, true, ii);
        // phase = (10.5 - 11.5) + II = 23.0
        let expected = one_cpt_oral_ss(&dose0, 23.0, cl, v, ka);
        assert_relative_eq!(c_pre, expected, max_relative = 1e-9);
    }

    #[test]
    fn test_ss_oral_via_predict_dispatches_to_ss_formula() {
        // With SS=1 and II large enough that exp(-k·II) is small, the SS
        // value should be close to but strictly above the single-dose value
        // at the same τ (the "+1" geometric-tail extra contribution).
        let ii = 24.0_f64;
        let cl = 2.0;
        let v = 20.0;
        let ka = 1.5;

        let doses_ss = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, ii)];
        let doses_single = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        let pk = oral_pk_params(cl, v, ka);

        let t = 4.0;
        let c_ss = predict_concentration(PkModel::OneCptOral, &doses_ss, t, &pk);
        let c_single = predict_concentration(PkModel::OneCptOral, &doses_single, t, &pk);
        assert!(
            c_ss > c_single,
            "SS conc {} should be greater than single-dose conc {}",
            c_ss,
            c_single
        );
        // With II = 24 and k = 0.1, KA = 1.5: exp(-k·II) ≈ 0.091, so the SS
        // tail adds ~10% to the slow term. Sanity: under 50%.
        assert!((c_ss / c_single) < 1.5);
    }

    // ── SS=1 resets the system at its record (#1576) ─────────────────────────
    //
    // The oracle is the closed form, written out here rather than read from the
    // engine: `C_single(τ) = D·ka/(V(ka−k))·(e^{−kτ} − e^{−ka·τ})` and
    // `C_ss(τ) = D·ka/(V(ka−k))·(e^{−kτ}/(1−e^{−k·II}) − e^{−ka·τ}/(1−e^{−ka·II}))`.
    // NONMEM ADVAN2 equals `C_ss` to 7.8e-16 on every reset row
    // (`nonmem_anchor/ss_reset`); these pin the analytic superposition to it.
    const SS_RESET_P: (f64, f64, f64, f64, f64) = (2.0, 20.0, 0.15, 100.0, 12.0); // CL V KA D II

    fn ss_reset_c_single(tau: f64) -> f64 {
        let (cl, v, ka, d, _) = SS_RESET_P;
        let k = cl / v;
        d * ka / (v * (ka - k)) * ((-k * tau).exp() - (-ka * tau).exp())
    }
    fn ss_reset_c_ss(tau: f64) -> f64 {
        let (cl, v, ka, d, ii) = SS_RESET_P;
        let k = cl / v;
        d * ka / (v * (ka - k))
            * ((-k * tau).exp() / (1.0 - (-k * ii).exp())
                - (-ka * tau).exp() / (1.0 - (-ka * ii).exp()))
    }
    fn ss_reset_bolus(t: f64) -> DoseEvent {
        DoseEvent::new(t, SS_RESET_P.3, 1, 0.0, false, 0.0)
    }
    fn ss_reset_ss(t: f64) -> DoseEvent {
        DoseEvent::new(t, SS_RESET_P.3, 1, 0.0, true, SS_RESET_P.4)
    }
    fn ss_reset_pk() -> PkParams {
        oral_pk_params(SS_RESET_P.0, SS_RESET_P.1, SS_RESET_P.2)
    }
    fn ss_reset_check(label: &str, doses: &[DoseEvent], cases: &[(f64, f64)]) {
        let pk = ss_reset_pk();
        for &(t, want) in cases {
            let got = predict_concentration(PkModel::OneCptOral, doses, t, &pk);
            assert!(got.is_finite(), "{label}: non-finite at t={t}");
            let rel = ((got - want) / want).abs();
            assert!(
                rel < 1e-11,
                "{label}: t={t} got {got} want {want} (rel {rel:e})"
            );
        }
    }

    /// Row 5: a bolus before a mid-timeline `SS=1` record is dead at and after it
    /// (was +10 … +24 %), and fully live before it.
    #[test]
    fn an_ss_record_wipes_an_earlier_bolus_in_analytic_superposition() {
        let doses = [ss_reset_bolus(0.0), ss_reset_ss(24.0)];
        ss_reset_check(
            "row 5",
            &doses,
            &[
                (10.0, ss_reset_c_single(10.0)),
                (24.5, ss_reset_c_ss(0.5)),
                (30.0, ss_reset_c_ss(6.0)),
                (35.0, ss_reset_c_ss(11.0)),
            ],
        );
    }

    /// Row 41, the TDM "SS on every visit" idiom (was +91 % at t = 12.5): only the
    /// latest reached record is live, so every interval reads `C_ss` exactly.
    #[test]
    fn ss_on_every_dose_record_reads_the_steady_state_in_every_interval() {
        let doses: Vec<_> = (0..4).map(|i| ss_reset_ss(12.0 * i as f64)).collect();
        ss_reset_check(
            "row 41",
            &doses,
            &[
                (6.0, ss_reset_c_ss(6.0)),
                (12.5, ss_reset_c_ss(0.5)),
                (30.0, ss_reset_c_ss(6.0)),
                (40.0, ss_reset_c_ss(4.0)),
                (60.0, ss_reset_c_ss(24.0)),
            ],
        );
    }

    /// Rows 51/52 in one test, both sides of the tie: a bolus row before a co-timed
    /// `SS=1` row is wiped, one after it superposes (NONMEM-measured row order).
    #[test]
    fn a_co_timed_bolus_is_wiped_before_the_ss_row_and_superposes_after_it() {
        ss_reset_check(
            "row 51 (bolus row, then SS row)",
            &[ss_reset_bolus(10.0), ss_reset_ss(10.0)],
            &[(10.5, ss_reset_c_ss(0.5)), (15.0, ss_reset_c_ss(5.0))],
        );
        ss_reset_check(
            "row 52 (SS row, then bolus row)",
            &[ss_reset_ss(10.0), ss_reset_bolus(10.0)],
            &[
                (10.5, ss_reset_c_ss(0.5) + ss_reset_c_single(0.5)),
                (15.0, ss_reset_c_ss(5.0) + ss_reset_c_single(5.0)),
            ],
        );
    }

    /// Row 4, the control: a non-SS dose after an `SS=1` record still superposes.
    #[test]
    fn a_non_ss_dose_after_an_ss_record_still_superposes() {
        ss_reset_check(
            "row 4",
            &[ss_reset_ss(0.0), ss_reset_bolus(120.0)],
            &[
                (100.0, ss_reset_c_ss(100.0)),
                (125.0, ss_reset_c_ss(125.0) + ss_reset_c_single(5.0)),
            ],
        );
    }

    /// Superposition keys the reset on the dose **record**, so a lagged dose recorded before
    /// an `SS=1` record but arriving after it is cancelled, as NONMEM does (#1587's measured
    /// row: dose at 9 with `ALAG1 = 2`, `SS=1` at 10; NONMEM 3.5948 at t = 11.5, `main` +9.8 %).
    /// The event-driven walkers still key on the arrival (#1587). With `lag < II` the `SS=1`
    /// dose reads its pre-arrival tail `C_ss(t − rec + II − lag)` before its own arrival.
    #[test]
    fn a_pending_lagged_dose_is_cancelled_by_an_ss_record_in_superposition() {
        let mut pk = ss_reset_pk();
        pk.values[crate::types::PK_IDX_LAGTIME] = 2.0;
        let doses = [ss_reset_bolus(9.0), ss_reset_ss(10.0)];
        let cases = [
            (11.5, ss_reset_c_ss(11.5)), // pre-arrival: phase II − lag = 10
            (13.0, ss_reset_c_ss(1.0)),  // one hour after the SS arrival at 12
        ];
        for (t, want) in cases {
            let got = predict_concentration(PkModel::OneCptOral, &doses, t, &pk);
            let rel = ((got - want) / want).abs();
            assert!(
                got.is_finite() && rel < 1e-11,
                "t={t}: got {got} want {want} (rel {rel:e})"
            );
        }
        // Differential: without the `SS=1` row the pending dose is live at 13.
        let alone = predict_concentration(PkModel::OneCptOral, &doses[..1], 13.0, &pk);
        assert!(
            alone > 0.1 * ss_reset_c_ss(1.0),
            "fixture degenerate: {alone}"
        );
    }

    /// A central infusion running across an `SS=1` record into the depot contributes nothing
    /// after the record on the superposition path (#1586's "infusion into central" row: +110 %
    /// before #1576). An infusion superposition cannot express (into the depot) routes to the
    /// event-driven walk, which stops it the same way since #1586
    /// (`an_ss_record_stops_a_running_infusion_in_the_cached_and_the_scanned_rates`).
    #[test]
    fn a_central_infusion_running_across_an_ss_record_stops_in_superposition() {
        let (cl, _, _, _, _) = SS_RESET_P;
        let infusion = DoseEvent::new(0.0, 200.0, 2, 10.0, false, 0.0); // 0–20 h into central
        let doses = [infusion, ss_reset_ss(10.0)];
        let k = cl / SS_RESET_P.1;
        ss_reset_check(
            "central infusion 0–20 h, SS=1 into the depot at 10",
            &doses,
            &[
                (5.0, 10.0 / cl * (1.0 - (-k * 5.0).exp())), // live before the record
                (10.5, ss_reset_c_ss(0.5)),
                (15.0, ss_reset_c_ss(5.0)),
                (21.0, ss_reset_c_ss(11.0)),
            ],
        );
    }

    /// The state twin (`[derived]`, sdtab states, depot readout) on row 41: the skip in
    /// `predict_concentration` alone leaves this wrong.
    #[test]
    fn analytical_states_reset_at_every_ss_record_too() {
        let (_, _, ka, d, ii) = SS_RESET_P;
        let subj = Subject {
            doses: (0..4).map(|i| ss_reset_ss(12.0 * i as f64)).collect(),
            ..make_subject_with_tv(HashMap::new(), vec![], vec![], 0, 0)
        };
        let times = [6.0, 12.5, 30.0, 40.0];
        let states = analytical_state_at_times(PkModel::OneCptOral, &subj, &ss_reset_pk(), &times);
        for (&t, st) in times.iter().zip(&states) {
            // Elapsed time since the latest reached record (records at 0, 12, 24, 36).
            let tau = t - (12.0 * (t / 12.0).floor()).min(36.0);
            let depot = d * (-ka * tau).exp() / (1.0 - (-ka * ii).exp());
            for (name, got, want) in [
                ("depot", st[0], depot),
                ("central", st[1], ss_reset_c_ss(tau)),
            ] {
                assert!(got.is_finite(), "{name} non-finite at t={t}");
                let rel = ((got - want) / want).abs();
                assert!(
                    rel < 1e-11,
                    "{name} at t={t}: got {got} want {want} (rel {rel:e})"
                );
            }
        }
    }

    // ── apply_scaling ───────────────────────────────────────────────────────

    /// Build the same `cl_from_cr_model` test fixture but with the
    /// requested `ScalingSpec` swapped in. Used by all apply_scaling tests
    /// so they exercise the production signature
    /// (`apply_scaling(&model, &subject, ...)`).
    fn model_with_scaling(scaling: crate::types::ScalingSpec) -> crate::types::CompiledModel {
        let mut m = cl_from_cr_model();
        m.scaling = scaling;
        m
    }

    fn one_subject_for_scaling() -> Subject {
        let mut cov = HashMap::new();
        cov.insert("CR".to_string(), 1.0);
        make_subject_with_tv(cov, Vec::new(), Vec::new(), 1, 3)
    }

    #[test]
    fn test_apply_scaling_none_is_noop() {
        let model = model_with_scaling(ScalingSpec::None);
        let subj = one_subject_for_scaling();
        let mut preds = vec![10.0, 20.0, 30.0];
        apply_scaling(&model, &subj, &[10.0], &[], &mut preds);
        assert_eq!(preds, vec![10.0, 20.0, 30.0]);
    }

    // ── analytic Form C readout (#650) ───────────────────────────────────────

    /// A degenerate `y = central / V` readout must reproduce the built-in
    /// concentration bit-for-bit (central amount = conc × V, so `/V` recovers
    /// conc), and an additive term must shift each prediction by exactly that
    /// amount — proving the readout replaces the concentration output correctly.
    #[test]
    fn analytic_form_c_readout_matches_and_shifts_concentration() {
        use crate::parser::model_parser::parse_model_string;
        let base = "\
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(50.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP)
";
        let plain = parse_model_string(base).expect("plain parses");
        let ident = parse_model_string(&format!("{base}[scaling]\n  y = central / V\n"))
            .expect("identity readout parses");
        let shifted = parse_model_string(&format!("{base}[scaling]\n  y = central / V + 2.0\n"))
            .expect("shifted readout parses");
        assert!(ident.analytic_readout.is_some());

        let subj = make_subject_with_tv(HashMap::new(), Vec::new(), Vec::new(), 1, 6);
        let theta = [1.0, 50.0];
        let c_plain = compute_predictions_with_tv(&plain, &subj, &theta, &[0.0]);
        let c_ident = compute_predictions_with_tv(&ident, &subj, &theta, &[0.0]);
        let c_shift = compute_predictions_with_tv(&shifted, &subj, &theta, &[0.0]);

        assert_eq!(c_plain.len(), 6);
        for i in 0..c_plain.len() {
            assert_relative_eq!(c_ident[i], c_plain[i], max_relative = 1e-12);
            assert_relative_eq!(c_shift[i], c_plain[i] + 2.0, epsilon = 1e-9);
            assert!(c_plain[i] > 0.0, "sanity: positive concentration");
        }
    }

    /// The readout's covariate is read from the **per-observation** snapshot, so a
    /// per-row flag (here `SEL`) selects the branch on that row — the analytic
    /// analogue of NONMEM's per-record `$ERROR`. Rows with `SEL==0` read `central/V`
    /// (concentration); rows with `SEL==1` read twice that.
    #[test]
    fn analytic_form_c_readout_uses_per_obs_covariate() {
        use crate::parser::model_parser::parse_model_string;
        let src = "\
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(50.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[scaling]
  y = if (SEL == 0) central / V else 2 * central / V
[error_model]
  DV ~ proportional(PROP)
";
        let model = parse_model_string(src).expect("parses");
        // 4 observations, alternating SEL flag per row.
        let obs_cov = vec![
            HashMap::from([("SEL".to_string(), 0.0)]),
            HashMap::from([("SEL".to_string(), 1.0)]),
            HashMap::from([("SEL".to_string(), 0.0)]),
            HashMap::from([("SEL".to_string(), 1.0)]),
        ];
        let subj = make_subject_with_tv(HashMap::new(), Vec::new(), obs_cov, 1, 4);
        let theta = [1.0, 50.0];
        let preds = compute_predictions_with_tv(&model, &subj, &theta, &[0.0]);
        // Odd rows (SEL==1) are exactly double the even rows' readout only if the
        // underlying concentration were equal — instead compare each row to its own
        // concentration via the ratio of the SEL==1 formula (2×) applied on its conc.
        assert_relative_eq!(
            preds[1] / preds[0],
            2.0 * (preds_conc(&subj, &theta, 1) / preds_conc(&subj, &theta, 0)),
            max_relative = 1e-9
        );
        assert_relative_eq!(
            preds[3] / preds[2],
            2.0 * (preds_conc(&subj, &theta, 3) / preds_conc(&subj, &theta, 2)),
            max_relative = 1e-9
        );
    }

    /// Helper: the plain built-in concentration at observation `i` for the fixture
    /// above (CL=1, V=50, single 100 mg IV bolus at t=0).
    fn preds_conc(subj: &Subject, theta: &[f64], i: usize) -> f64 {
        let mut pk = PkParams::default();
        pk.values[crate::types::PK_IDX_CL] = theta[0];
        pk.values[crate::types::PK_IDX_V] = theta[1];
        predict_concentration(PkModel::OneCptIv, &subj.doses, subj.obs_times[i], &pk)
    }

    /// **NONMEM-validation cross-check (#650).** An analytic Form C readout
    /// computes the same output as the *ODE* Form C readout, which is validated
    /// against NONMEM's per-record `$ERROR` (free/total protein-binding model,
    /// `docs/model-file/scaling.qmd`). On an IV 1-cpt model the analytic central
    /// amount (`concentration × V`) equals the ODE `central` state amount, so a
    /// saturable-binding total-vs-free readout `y = C + BMAX·C/(KD+C)` must give
    /// bit-for-bit-equal predictions on the analytic and ODE paths at matched
    /// parameters — inheriting the ODE path's NONMEM anchor.
    #[test]
    fn analytic_form_c_matches_ode_form_c_binding_readout() {
        use crate::parser::model_parser::parse_model_string;
        let readout = "  y = central / V + BMAX * (central / V) / (KD + central / V)\n";
        let params = "\
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVBMAX(3.0, 0.01, 100.0)
  theta TVKD(2.0, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP ~ 0.02 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV  * exp(ETA_V)
  BMAX = TVBMAX
  KD   = TVKD
";
        let analytic = parse_model_string(&format!(
            "{params}[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[scaling]\n{readout}[error_model]\n  DV ~ proportional(PROP)\n"
        ))
        .expect("analytic parses");
        // Equivalent ODE: `central` is the amount, disposition -(CL/V)·central,
        // IV bolus into central (cmt 1). Same readout replaces the state readout.
        let ode = parse_model_string(&format!(
            "{params}[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL / V) * central\n[scaling]\n{readout}[error_model]\n  DV ~ proportional(PROP)\n"
        ))
        .expect("ode parses");
        assert!(analytic.analytic_readout.is_some());
        assert!(ode.ode_spec.is_some());

        let subj = make_subject_with_tv(HashMap::new(), Vec::new(), Vec::new(), 1, 6);
        let theta = [0.2, 10.0, 3.0, 2.0];
        let eta = [0.1, -0.05];
        let a = compute_predictions_with_tv(&analytic, &subj, &theta, &eta);
        let o = compute_predictions_with_tv(&ode, &subj, &theta, &eta);
        assert_eq!(a.len(), o.len());
        for (av, ov) in a.iter().zip(o.iter()) {
            assert_relative_eq!(av, ov, max_relative = 1e-6, epsilon = 1e-9);
        }
    }

    /// Two-occasion IOV subject for the Form C readout cross-engine checks (#1079):
    /// a dose at t = 0 (occasion 1) and at t = 24 (occasion 2), two samples inside
    /// each occasion. No washout, so the occasion-2 samples carry occasion-1 drug.
    fn iov_readout_subject() -> Subject {
        let obs_times = vec![1.0, 3.0, 25.0, 27.0];
        let n = obs_times.len();
        Subject {
            id: "1".to_string(),
            doses: vec![bolus_dose(0.0, 100.0), bolus_dose(24.0, 100.0)],
            obs_times,
            observations: vec![0.0; n],
            obs_cmts: vec![1; n],
            cens: vec![0; n],
            occasions: vec![1, 1, 2, 2],
            dose_occasions: vec![1, 2],
            ..Default::default()
        }
    }

    /// **Regression (#1079): a Form C readout under IOV must be evaluated with the observation's
    /// own occasion κ.** `apply_analytic_readout` builds the readout's individual parameters from a
    /// `pk_param_fn` snapshot; under IOV `predict_iov` called it once with `eta_bsv` (κ held at 0).
    /// The concentration did carry the occasion's κ, but every individual parameter the readout
    /// read was frozen at its typical value, so an additive κ-carrying baseline was silently
    /// dropped from the prediction and objective (12-20% on the reproduction below).
    ///
    /// The oracle is cross-engine: the hand-written ODE twin applies the same readout inside the
    /// integrator against the per-observation parameter snapshot (also NONMEM's per-record
    /// `$ERROR` convention). A `Dual2`-vs-FD parity test passes against the wrong function, because
    /// the analytic path was seeded to match it. The decomposition is checked too: `y − central/V`
    /// must equal `BASE` evaluated at *that observation's* κ, pinning the fix to the per-occasion
    /// snapshot rather than a compensating error elsewhere.
    #[test]
    fn analytic_form_c_iov_readout_uses_occasion_kappa_matching_ode_twin() {
        use crate::parser::model_parser::parse_model_string;
        let params = "\
[parameters]
  theta TVCL(3.0, 0.01, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  theta TVBASE(2.0, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA_B ~ 0.04
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  BASE = TVBASE * exp(KAPPA_B)
";
        let tail = "\
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  iov_column = OCC
  ode_reltol = 1e-12
  ode_abstol = 1e-12
";
        // Closed-form and its hand-written ODE twin, same `[scaling]` readout.
        let analytic = parse_model_string(&format!(
            "{params}[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n\
             [scaling]\n  y = central / V + BASE\n{tail}"
        ))
        .expect("analytic parses");
        let ode = parse_model_string(&format!(
            "{params}[structural_model]\n  ode(states=[central])\n\
             [odes]\n  d/dt(central) = -(CL / V) * central\n\
             [scaling]\n  y = central / V + BASE\n{tail}"
        ))
        .expect("ode twin parses");
        // The readout's `V` divisor cancels the `conc × V` amount reconstruction, so
        // this (natural) readout cannot show the bug — pinned below so the fix does
        // not break the load-bearing cancellation.
        let conc_only = parse_model_string(&format!(
            "{params}[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n\
             [scaling]\n  y = central / V\n{tail}"
        ))
        .expect("identity readout parses");
        // Same model with no readout at all — the built-in concentration output.
        let plain = parse_model_string(&format!(
            "{params}[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n{tail}"
        ))
        .expect("plain parses");
        assert_eq!(analytic.n_kappa, 1, "one occasion random effect");
        assert!(analytic.analytic_readout.is_some() && analytic.ode_spec.is_none());
        assert!(ode.ode_spec.is_some());

        let subj = iov_readout_subject();
        let theta = [3.0, 20.0, 2.0];
        let eta_bsv = [0.1];
        let kappas = vec![vec![0.5], vec![-0.4]];

        let a = predict_iov(&analytic, &subj, &theta, &eta_bsv, &kappas);
        let o = predict_iov(&ode, &subj, &theta, &eta_bsv, &kappas);
        assert_eq!(a.len(), subj.obs_times.len());
        for (j, (av, ov)) in a.iter().zip(o.iter()).enumerate() {
            assert_relative_eq!(av, ov, max_relative = 1e-8, epsilon = 1e-12);
            assert!(av.is_finite(), "obs {j} finite");
        }

        // Decomposition: the readout output minus the concentration is `BASE` at the
        // observation's own occasion κ (`2·e^0.5` in occasion 1, `2·e^−0.4` in
        // occasion 2), NOT the typical value `2` the κ = 0 snapshot produced.
        let c = predict_iov(&plain, &subj, &theta, &eta_bsv, &kappas);
        let expected_base = [
            2.0 * 0.5_f64.exp(),
            2.0 * 0.5_f64.exp(),
            2.0 * (-0.4_f64).exp(),
            2.0 * (-0.4_f64).exp(),
        ];
        for (j, (av, cv)) in a.iter().zip(c.iter()).enumerate() {
            assert_relative_eq!(av - cv, expected_base[j], max_relative = 1e-10);
            assert!(
                (av - cv - 2.0).abs() > 0.5,
                "obs {j}: baseline must not collapse to the κ = 0 typical value"
            );
        }

        // The `y = central / V` cancellation still holds: the amount is rebuilt as
        // `conc × V` with the *same* `V` the readout divides by, so the per-occasion
        // evaluation must not perturb it (the round-trip is exact to rounding).
        let ident = predict_iov(&conc_only, &subj, &theta, &eta_bsv, &kappas);
        for (iv, cv) in ident.iter().zip(c.iter()) {
            assert_relative_eq!(iv, cv, max_relative = 1e-14);
        }
    }

    /// The `y = central / V` cancellation with a κ **on `V` itself** (#1079 fix note
    /// 3). The predictor rebuilds the central amount as `conc × V` and the readout
    /// divides by the same `V`; both now come from the observation's occasion, so the
    /// two must still cancel (exactly, up to the multiply/divide round-trip). Reading
    /// `V` from a κ = 0 snapshot on one side only would break this by `exp(κ)`.
    #[test]
    fn analytic_form_c_iov_identity_readout_cancels_kappa_on_v() {
        use crate::parser::model_parser::parse_model_string;
        let params = "\
[parameters]
  theta TVCL(3.0, 0.01, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA_V ~ 0.04
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV * exp(KAPPA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
";
        let tail = "\
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  iov_column = OCC
";
        let ident =
            parse_model_string(&format!("{params}[scaling]\n  y = central / V\n{tail}")).unwrap();
        let plain = parse_model_string(&format!("{params}{tail}")).unwrap();

        let subj = iov_readout_subject();
        let theta = [3.0, 20.0];
        let eta_bsv = [0.1];
        let kappas = vec![vec![0.5], vec![-0.4]];
        let y = predict_iov(&ident, &subj, &theta, &eta_bsv, &kappas);
        let c = predict_iov(&plain, &subj, &theta, &eta_bsv, &kappas);
        assert_eq!(y.len(), c.len());
        for (yv, cv) in y.iter().zip(c.iter()) {
            assert_relative_eq!(yv, cv, max_relative = 1e-14);
        }
    }

    #[test]
    fn test_apply_log_transform_noop_when_disabled() {
        let mut model = cl_from_cr_model();
        model.log_transform = false;
        let mut preds = vec![1.0, 2.0, 4.0];
        apply_log_transform(&model, &mut preds);
        assert_eq!(preds, vec![1.0, 2.0, 4.0]);
    }

    #[test]
    fn test_apply_log_transform_logs_with_floor() {
        let mut model = cl_from_cr_model();
        model.log_transform = true;
        let mut preds = vec![1.0, std::f64::consts::E, 0.0, -5.0];
        apply_log_transform(&model, &mut preds);
        assert_relative_eq!(preds[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(preds[1], 1.0, epsilon = 1e-12);
        // Non-positive predictions are floored, not turned into -inf/NaN.
        assert_relative_eq!(preds[2], LTBS_FLOOR.ln(), epsilon = 1e-9);
        assert_relative_eq!(preds[3], LTBS_FLOOR.ln(), epsilon = 1e-9);
        assert!(preds.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn test_ltbs_prediction_is_log_of_natural() {
        // The whole prediction pipeline (compute_predictions_with_tv) returns
        // ln(f) when LTBS is active, so log-scale predictions flow into every
        // downstream consumer (likelihood, IWRES/CWRES, IPRED, simulation).
        let subj = one_subject_for_scaling();
        let theta = [1.0];

        let mut natural_model = cl_from_cr_model();
        natural_model.log_transform = false;
        let natural = compute_predictions_with_tv(&natural_model, &subj, &theta, &[]);

        let mut ltbs_model = cl_from_cr_model();
        ltbs_model.log_transform = true;
        let logged = compute_predictions_with_tv(&ltbs_model, &subj, &theta, &[]);

        assert_eq!(natural.len(), logged.len());
        for (n, l) in natural.iter().zip(logged.iter()) {
            assert_relative_eq!(*l, n.max(LTBS_FLOOR).ln(), epsilon = 1e-9);
        }
    }

    #[test]
    fn test_ltbs_frem_predictions_are_raw_not_logged() {
        // Regression: FREM covariate predictions must be raw (theta + eta),
        // NOT log-transformed, even when the PK error model uses log_additive.
        // The override must happen AFTER apply_log_transform so the log does
        // not corrupt covariate predictions.
        let mut model = cl_from_cr_model();
        model.log_transform = true; // LTBS (log-additive)
        model.n_eta = 2;
        model.eta_names = vec!["ETA_CL".into(), "ETA_COV".into()];
        model.default_params.omega = crate::types::OmegaMatrix::from_diagonal(
            &[0.1, 100.0],
            vec!["ETA_CL".into(), "ETA_COV".into()],
        );
        // FREM config: FREMTYPE 100 -> (theta_idx=0 [TVCL], eta_idx=1 [ETA_COV])
        let mut map = std::collections::HashMap::new();
        map.insert(100u16, (0usize, 1usize));
        model.frem_config = Some(crate::types::FremConfig {
            fremtype_to_indices: map,
            covariate_sigma_index: 0,
        });

        let theta = [1.0]; // TVCL = 1.0
        let eta = [0.0, 5.0]; // ETA_COV = 5.0
                              // Expected FREM prediction: theta[0] + eta[1] = 1.0 + 5.0 = 6.0 (raw, NOT logged)

        // Subject with 2 obs: FREMTYPE 0 (PK) and FREMTYPE 100 (covariate)
        let mut subj = one_subject_for_scaling(); // has 3 obs
        subj.fremtype = vec![0, 100, 0];

        let preds = compute_predictions_with_tv(&model, &subj, &theta, &eta);
        // PK rows (0, 2) should be log-transformed
        assert!(preds[0].is_finite());
        // FREM row (1) must be raw: theta[0] + eta[1] = 6.0, not ln(6.0)
        assert_relative_eq!(preds[1], 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_apply_scaling_scalar_divides_in_place() {
        let model = model_with_scaling(ScalingSpec::ScalarScale(1000.0));
        let subj = one_subject_for_scaling();
        let mut preds = vec![1000.0, 2000.0, 3000.0];
        apply_scaling(&model, &subj, &[10.0], &[], &mut preds);
        for (got, exp) in preds.iter().zip([1.0, 2.0, 3.0].iter()) {
            assert_relative_eq!(got, exp, epsilon = 1e-12);
        }
    }

    /// Regression (#769 review): `predict_iov` must apply `[scaling]` to a subject
    /// that carries *dose* occasions but no *observation* occasions. In that state
    /// `iov_occasion_groups` is non-empty — it appends the dose-only occasion as a
    /// group with an EMPTY obs-index list — yet no observation is owned by any
    /// group, so the per-occasion scaling loop writes nothing. The base pass is
    /// what scales those observations; guarding it on `occ_groups.is_empty()`
    /// (rather than `subject.occasions.is_empty()`) skips it here and returns the
    /// predictions UNSCALED, off by the full scale factor — the #723 bug.
    ///
    /// The current data readers never build this state (they gate observation- and
    /// dose-occasion labels on the same column), so it is constructed directly to
    /// lock the guard against a future path that produces it.
    #[test]
    fn predict_iov_scales_subject_with_dose_only_occasions() {
        let scale = 1000.0;
        let mut model = cl_from_cr_model();
        // A consistent IOV model — `dose_occasions` are only populated for IOV
        // models. The κ block stays zero here, which is all this guard needs.
        model.n_kappa = 1;
        model.kappa_names = vec!["KAPPA_CL".to_string()];

        // Observations present + a dose occasion, but NO observation occasions.
        let mut dose_only = one_subject_for_scaling();
        dose_only.dose_occasions = vec![1; dose_only.doses.len()];
        assert!(
            dose_only.occasions.is_empty() && !dose_only.dose_occasions.is_empty(),
            "trigger state: dose occasions present, observation occasions absent"
        );

        // Unscaled reference (scaling off) — the predictions the scale divides.
        model.scaling = ScalingSpec::None;
        let raw = predict_iov(&model, &dose_only, &[1.0], &[], &[]);
        assert!(
            !raw.is_empty() && raw.iter().all(|v| v.is_finite() && *v > 0.0),
            "sanity: positive concentrations to scale, got {raw:?}"
        );

        // Scaling on: every prediction must be divided by `scale`, not left as-is.
        // Fails on the `occ_groups.is_empty()` guard (returns `raw`, `scale`× too big).
        model.scaling = ScalingSpec::ScalarScale(scale);
        let got = predict_iov(&model, &dose_only, &[1.0], &[], &[]);
        assert_eq!(got.len(), raw.len());
        for (g, r) in got.iter().zip(raw.iter()) {
            assert_relative_eq!(*g, r / scale, epsilon = 1e-12);
        }

        // And identical to the fully occasion-less subject (the known-correct #723
        // base-pass path, unchanged by #769): dose-only occasions must not move it.
        let occasionless = one_subject_for_scaling(); // occasions=[] AND dose_occasions=[]
        let reference = predict_iov(&model, &occasionless, &[1.0], &[], &[]);
        assert_eq!(got.len(), reference.len());
        for (g, r) in got.iter().zip(reference.iter()) {
            assert_relative_eq!(*g, r, epsilon = 1e-12);
        }

        // Mixed subject — observations DO carry occasions, plus a second dose in an
        // occasion with no co-timed sample. This takes the per-occasion (`else`)
        // branch and exercises the dose-only-group `continue`: the empty group is
        // skipped while every observation is still scaled per-occasion.
        let mut mixed = one_subject_for_scaling();
        mixed.doses.push(bolus_dose(0.0, 100.0));
        mixed.occasions = vec![1, 1, 1]; // all observations in occasion 1
        mixed.dose_occasions = vec![1, 2]; // second dose in occasion 2 (no observations)
        model.scaling = ScalingSpec::None;
        let raw_mixed = predict_iov(&model, &mixed, &[1.0], &[], &[]);
        assert!(raw_mixed.iter().all(|v| v.is_finite() && *v > 0.0));
        model.scaling = ScalingSpec::ScalarScale(scale);
        let got_mixed = predict_iov(&model, &mixed, &[1.0], &[], &[]);
        assert_eq!(got_mixed.len(), raw_mixed.len());
        for (g, r) in got_mixed.iter().zip(raw_mixed.iter()) {
            assert_relative_eq!(*g, r / scale, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_apply_scaling_marks_bad_scale_as_nan() {
        // When a ScaleFn returns 0 / NaN / inf at runtime, apply_scaling
        // replaces every prediction with NaN so the outer NLL goes NaN
        // and the optimizer rejects the step. Loud failure mode (vs the
        // earlier "skip silently" policy, which hid mis-scaled fits).
        let bad_returns = [0.0_f64, -1.0_f64, f64::NAN, f64::INFINITY];
        for &val in &bad_returns {
            let scale_fn: crate::types::ScaleFn = Box::new(move |_, _, _, _| val);
            let model = model_with_scaling(ScalingSpec::ExpressionScale {
                scale_fn,
                deriv: None,
            });
            let subj = one_subject_for_scaling();
            let mut preds = vec![1.0, 2.0, 3.0];
            apply_scaling(&model, &subj, &[10.0], &[], &mut preds);
            for p in &preds {
                assert!(
                    p.is_nan(),
                    "bad scale ({}) must produce NaN preds, got {}",
                    val,
                    p
                );
            }
        }
    }

    #[test]
    fn test_apply_scaling_scalar_bad_value_marks_nan() {
        // Defensive guard on the ScalarScale branch (hand-constructed
        // models bypass the parser's > 0 check).
        for &k in &[0.0_f64, -1.0_f64, f64::NAN, f64::INFINITY] {
            let model = model_with_scaling(ScalingSpec::ScalarScale(k));
            let subj = one_subject_for_scaling();
            let mut preds = vec![1.0, 2.0, 3.0];
            apply_scaling(&model, &subj, &[10.0], &[], &mut preds);
            for p in &preds {
                assert!(
                    p.is_nan(),
                    "bad scalar scale ({}) must produce NaN preds, got {}",
                    k,
                    p
                );
            }
        }
    }

    #[test]
    fn test_apply_scaling_expression_uses_covariate() {
        // scale_fn = WT / 70 — divisive convention so preds / (WT/70).
        let scale_fn: crate::types::ScaleFn =
            Box::new(|_theta, _eta, cov: &HashMap<String, f64>, _pk: &PkParams| {
                cov.get("WT").copied().unwrap_or(70.0) / 70.0
            });
        let model = model_with_scaling(ScalingSpec::ExpressionScale {
            scale_fn,
            deriv: None,
        });
        let mut subj = one_subject_for_scaling();
        subj.covariates.insert("WT".to_string(), 84.0); // scale = 84/70 = 1.2
        let mut preds = vec![12.0, 24.0];
        apply_scaling(&model, &subj, &[10.0], &[], &mut preds);
        assert_relative_eq!(preds[0], 10.0, epsilon = 1e-12); // 12 / 1.2
        assert_relative_eq!(preds[1], 20.0, epsilon = 1e-12); // 24 / 1.2
    }

    #[test]
    fn test_apply_scaling_expression_uses_pk_param() {
        // Regression for Phase 1.5 lift: scale_fn can now read individual
        // parameters from `pk`. Here `scale = V / 10` — V is at PK slot 1
        // (PK_IDX_V) for the cl_from_cr model fixture (CL=slot 0, V=slot 1).
        // The model sets V = 50 so scale = 5 and preds get divided by 5.
        let scale_fn: crate::types::ScaleFn =
            Box::new(|_theta, _eta, _cov: &HashMap<String, f64>, pk: &PkParams| {
                pk.values[crate::types::PK_IDX_V] / 10.0
            });
        let model = model_with_scaling(ScalingSpec::ExpressionScale {
            scale_fn,
            deriv: None,
        });
        let subj = one_subject_for_scaling();
        let mut preds = vec![100.0, 50.0];
        apply_scaling(&model, &subj, &[10.0], &[], &mut preds);
        // cl_from_cr_model sets V = 50, so scale = 5.
        assert_relative_eq!(preds[0], 20.0, epsilon = 1e-12);
        assert_relative_eq!(preds[1], 10.0, epsilon = 1e-12);
    }

    // ── PerCmt dispatch ────────────────────────────────────────────────────

    #[test]
    fn test_apply_scaling_per_cmt_dispatches_per_observation() {
        // Two observations on different CMTs get scaled by different
        // factors. CMT=1 → /1000, CMT=2 → /1.
        let mut map: HashMap<usize, ScalingSpec> = HashMap::new();
        map.insert(1, ScalingSpec::ScalarScale(1000.0));
        map.insert(2, ScalingSpec::ScalarScale(1.0));
        let model = model_with_scaling(ScalingSpec::PerCmt(map));

        let mut subj = one_subject_for_scaling();
        // The helper produces 3 obs all on CMT=1; override one to CMT=2.
        assert!(subj.obs_cmts.len() >= 2);
        subj.obs_cmts[0] = 1;
        subj.obs_cmts[1] = 2;
        subj.obs_cmts[2] = 1;

        let mut preds = vec![1000.0, 50.0, 2000.0];
        apply_scaling(&model, &subj, &[10.0], &[], &mut preds);
        assert_relative_eq!(preds[0], 1.0, epsilon = 1e-12); // 1000 / 1000
        assert_relative_eq!(preds[1], 50.0, epsilon = 1e-12); // 50 / 1
        assert_relative_eq!(preds[2], 2.0, epsilon = 1e-12); // 2000 / 1000
    }

    #[test]
    fn test_apply_scaling_per_cmt_missing_cmt_yields_nan() {
        // PerCmt map covers CMT=1 only, but obs[1] is on CMT=2.
        // Pre-fit validation catches this at fit() entry, but apply_scaling
        // is the defensive last line — must produce NaN, not silent zero.
        let mut map: HashMap<usize, ScalingSpec> = HashMap::new();
        map.insert(1, ScalingSpec::ScalarScale(1000.0));
        let model = model_with_scaling(ScalingSpec::PerCmt(map));

        let mut subj = one_subject_for_scaling();
        subj.obs_cmts[0] = 1;
        subj.obs_cmts[1] = 2; // not in map → NaN
        subj.obs_cmts[2] = 1;

        let mut preds = vec![1000.0, 50.0, 2000.0];
        apply_scaling(&model, &subj, &[10.0], &[], &mut preds);
        assert_relative_eq!(preds[0], 1.0, epsilon = 1e-12);
        assert!(preds[1].is_nan(), "missing CMT must produce NaN");
        assert_relative_eq!(preds[2], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_validate_per_cmt_scaling_errors_on_missing_entry() {
        // The pre-fit validation should produce a clear error naming the
        // missing CMTs rather than relying on the runtime NaN fallback.
        let mut map: HashMap<usize, ScalingSpec> = HashMap::new();
        map.insert(1, ScalingSpec::ScalarScale(1000.0));
        let model = model_with_scaling(ScalingSpec::PerCmt(map));

        let mut subj = one_subject_for_scaling();
        subj.obs_cmts = vec![1, 2, 3]; // 2 and 3 missing from map

        let err = validate_per_cmt_scaling(&model, std::slice::from_ref(&subj))
            .expect_err("missing CMTs must error");
        assert!(
            err.contains("[2, 3]"),
            "error should list missing CMTs explicitly, got: {}",
            err
        );
    }

    #[test]
    fn test_validate_per_cmt_scaling_passes_when_complete() {
        let mut map: HashMap<usize, ScalingSpec> = HashMap::new();
        map.insert(1, ScalingSpec::ScalarScale(1000.0));
        map.insert(2, ScalingSpec::ScalarScale(1.0));
        let model = model_with_scaling(ScalingSpec::PerCmt(map));

        let mut subj = one_subject_for_scaling();
        subj.obs_cmts = vec![1, 2, 1];

        validate_per_cmt_scaling(&model, std::slice::from_ref(&subj))
            .expect("complete PerCmt coverage must pass validation");
    }

    #[test]
    fn test_validate_per_cmt_scaling_noop_for_non_per_cmt() {
        // Non-PerCmt scaling never errors regardless of observed CMTs.
        let model = model_with_scaling(ScalingSpec::ScalarScale(1000.0));
        let subj = one_subject_for_scaling();
        validate_per_cmt_scaling(&model, std::slice::from_ref(&subj))
            .expect("non-PerCmt scaling never errors");
    }

    // ── build_obs_scale_array (Phase 2.5: shared FD/AD materialiser) ───────

    #[test]
    fn test_build_obs_scale_array_none() {
        let spec = ScalingSpec::None;
        let pk = PkParams::default();
        let cov = HashMap::new();
        let obs_cmts = vec![1, 1, 2];
        let arr = spec.build_obs_scale_array(&[], &[], &cov, &pk, &obs_cmts);
        assert_eq!(arr, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_build_obs_scale_array_scalar() {
        let spec = ScalingSpec::ScalarScale(1000.0);
        let pk = PkParams::default();
        let cov = HashMap::new();
        let obs_cmts = vec![1; 4];
        let arr = spec.build_obs_scale_array(&[], &[], &cov, &pk, &obs_cmts);
        assert_eq!(arr, vec![1000.0; 4]);
    }

    #[test]
    fn test_build_obs_scale_array_scalar_bad_value_yields_nan() {
        // Defensive: hand-constructed ScalarScale with invalid k materialises
        // NaN — feeds the FD path's loud-failure semantic and the AD path
        // gets a Const NaN that propagates to NaN gradient too.
        for &k in &[0.0_f64, -1.0, f64::NAN, f64::INFINITY] {
            let spec = ScalingSpec::ScalarScale(k);
            let pk = PkParams::default();
            let arr = spec.build_obs_scale_array(&[], &[], &HashMap::new(), &pk, &[1; 3]);
            assert!(arr.iter().all(|s| s.is_nan()), "bad k={} → all NaN", k);
        }
    }

    #[test]
    fn test_build_obs_scale_array_expression() {
        // Closure returns 5.0; array should be 5.0 for every obs.
        let scale_fn: crate::types::ScaleFn = Box::new(|_, _, _, _| 5.0);
        let spec = ScalingSpec::ExpressionScale {
            scale_fn,
            deriv: None,
        };
        let pk = PkParams::default();
        let arr = spec.build_obs_scale_array(&[], &[], &HashMap::new(), &pk, &[1; 3]);
        assert_eq!(arr, vec![5.0; 3]);
    }

    #[test]
    fn test_build_obs_scale_array_per_cmt_dispatches() {
        let mut map: HashMap<usize, ScalingSpec> = HashMap::new();
        map.insert(1, ScalingSpec::ScalarScale(1000.0));
        map.insert(2, ScalingSpec::ScalarScale(2.0));
        let spec = ScalingSpec::PerCmt(map);
        let pk = PkParams::default();
        let obs_cmts = vec![1, 2, 1, 2];
        let arr = spec.build_obs_scale_array(&[], &[], &HashMap::new(), &pk, &obs_cmts);
        assert_eq!(arr, vec![1000.0, 2.0, 1000.0, 2.0]);
    }

    #[test]
    fn test_build_obs_scale_array_per_cmt_missing_cmt_is_nan() {
        // CMT=3 is observed but not in the map. The runtime fit-time
        // validation should catch this earlier; if it doesn't (hand-built
        // CompiledModel), the array materialises NaN at the missing slot.
        let mut map: HashMap<usize, ScalingSpec> = HashMap::new();
        map.insert(1, ScalingSpec::ScalarScale(1000.0));
        let spec = ScalingSpec::PerCmt(map);
        let pk = PkParams::default();
        let obs_cmts = vec![1, 3, 1];
        let arr = spec.build_obs_scale_array(&[], &[], &HashMap::new(), &pk, &obs_cmts);
        assert_eq!(arr[0], 1000.0);
        assert!(arr[1].is_nan());
        assert_eq!(arr[2], 1000.0);
    }

    // ── Mixed bolus + infusion within a single subject (issue #176) ─────────
    //
    // Regression guard for the silent-wrong-answer bug on the superposition
    // path: before #176, `single_dose_concentration` branched on the model
    // enum (`OneCptIvBolus` vs `OneCptInfusion`), so an infusion event under
    // an IV-bolus model was treated as an instantaneous bolus of `AMT`,
    // ignoring the duration entirely. After #176 the dispatch picks per
    // dose from `dose.is_infusion()`, so both routes coexist in one record.

    #[test]
    fn test_mixed_bolus_and_infusion_in_single_subject_superposition() {
        // Same `OneCptIv` model, two doses with different administration:
        //   t=0:  bolus     AMT=100, RATE=0
        //   t=4:  infusion  AMT=500, RATE=50 → duration=10
        // The superposition prediction must equal the sum of the two
        // single-dose closed forms (one bolus, one infusion).
        let bolus = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
        let infusion = DoseEvent::new(4.0, 500.0, 1, 50.0, false, 0.0);
        debug_assert!(!bolus.is_infusion() && infusion.is_infusion());
        debug_assert!((infusion.duration - 10.0).abs() < 1e-12);
        let doses = vec![bolus.clone(), infusion.clone()];
        let pk = make_pk_params(10.0, 100.0);

        // Probe times that fall in three regimes:
        //   t = 2.0  → only the bolus has been given
        //   t = 8.0  → both active, infusion still running
        //   t = 20.0 → infusion has ended, post-infusion decay
        for &t in &[2.0, 8.0, 20.0] {
            let combined = predict_concentration(PkModel::OneCptIv, &doses, t, &pk);
            let from_bolus = predict_concentration(PkModel::OneCptIv, &[bolus.clone()], t, &pk);
            let from_infusion =
                predict_concentration(PkModel::OneCptIv, &[infusion.clone()], t, &pk);
            assert_relative_eq!(combined, from_bolus + from_infusion, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_mixed_dose_superposition_matches_event_driven_path() {
        // Cross-check: the analytical superposition path and the
        // event-driven analytical path must agree to roundoff on a
        // mixed-administration record. Before #176 the superposition path
        // was silently wrong (bolus formula applied to the infusion dose),
        // while the event-driven path was already correct — so these two
        // would have disagreed materially.
        use crate::pk::event_driven::event_driven_predictions;
        use crate::types::Subject;

        let doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0), // bolus
            DoseEvent::new(4.0, 500.0, 1, 50.0, false, 0.0), // infusion (dur=10)
        ];
        let obs_times = vec![1.0_f64, 3.0, 6.0, 10.0, 14.0, 20.0];
        let pk = make_pk_params(10.0, 100.0);

        let analytical: Vec<f64> = obs_times
            .iter()
            .map(|&t| predict_concentration(PkModel::OneCptIv, &doses, t, &pk))
            .collect();

        let n_obs = obs_times.len();
        let subject = Subject {
            id: "mixed".into(),
            doses: doses.clone(),
            obs_times: obs_times.clone(),
            observations: vec![0.0; n_obs],
            obs_cmts: vec![1; n_obs],
            cens: vec![0; n_obs],
            ..Default::default()
        };
        let pk_dose: Vec<PkParams> = vec![pk.clone(); doses.len()];
        let pk_obs: Vec<PkParams> = vec![pk.clone(); obs_times.len()];
        let event_driven =
            event_driven_predictions(PkModel::OneCptIv, &subject, &pk_dose, &pk_obs, &[]);

        for (i, (&a, &e)) in analytical.iter().zip(event_driven.iter()).enumerate() {
            let diff = (a - e).abs();
            assert!(
                diff < 1e-10,
                "mismatch at obs[{i}] t={}: superposition {a} vs event-driven {e} (|Δ|={diff})",
                obs_times[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // Regression tests for single_dose_states SS dispatch
    // -----------------------------------------------------------------------

    /// Bug regression: single_dose_states must use the SS formula for the
    /// central compartment when dose.ss = true.  Before the fix, oral and
    /// IV-infusion SS doses fell through to single-dose formulas, returning
    /// concentrations that were typically 5-15× too low at trough.
    ///
    /// For each model the test checks that the SS central state matches
    /// single_dose_concentration (the long-standing, validated scalar path).
    #[test]
    fn single_dose_states_central_matches_concentration_for_ss_doses() {
        use crate::pk::{
            one_cpt_infusion_ss, one_cpt_iv_bolus_ss, one_cpt_oral_f_ss, two_cpt_infusion_ss,
            two_cpt_iv_bolus_ss, two_cpt_oral_f_ss,
        };
        // --- helpers ---
        fn bolus_ss(amt: f64, ii: f64) -> DoseEvent {
            DoseEvent::new(0.0, amt, 1, 0.0, true, ii)
        }
        fn infusion_ss(amt: f64, rate: f64, ii: f64) -> DoseEvent {
            DoseEvent::new(0.0, amt, 1, rate, true, ii)
        }

        let cl = 5.0_f64;
        let v = 80.0_f64;
        let q = 2.0_f64;
        let v2 = 40.0_f64;
        let ka = 1.0_f64;
        let f_bio = 1.0_f64;
        let ii = 24.0_f64;
        let amt = 100.0_f64;
        let rate = 50.0_f64; // 2 h infusion
        let tau = 6.0_f64;

        let mut p1iv = PkParams::default();
        p1iv.values[crate::types::PK_IDX_CL] = cl;
        p1iv.values[crate::types::PK_IDX_V] = v;

        let mut p1oral = PkParams::default();
        p1oral.values[crate::types::PK_IDX_CL] = cl;
        p1oral.values[crate::types::PK_IDX_V] = v;
        p1oral.values[crate::types::PK_IDX_KA] = ka;
        p1oral.values[crate::types::PK_IDX_F] = f_bio;

        let mut p2iv = PkParams::default();
        p2iv.values[crate::types::PK_IDX_CL] = cl;
        p2iv.values[crate::types::PK_IDX_V] = v;
        p2iv.values[crate::types::PK_IDX_Q] = q;
        p2iv.values[crate::types::PK_IDX_V2] = v2;

        let mut p2oral = PkParams::default();
        p2oral.values[crate::types::PK_IDX_CL] = cl;
        p2oral.values[crate::types::PK_IDX_V] = v;
        p2oral.values[crate::types::PK_IDX_Q] = q;
        p2oral.values[crate::types::PK_IDX_V2] = v2;
        p2oral.values[crate::types::PK_IDX_KA] = ka;
        p2oral.values[crate::types::PK_IDX_F] = f_bio;

        // OneCptIv SS bolus — central index 0
        let d = bolus_ss(amt, ii);
        let states = single_dose_states(PkModel::OneCptIv, &d, tau, &p1iv);
        let expected = one_cpt_iv_bolus_ss(&d, tau, cl, v);
        assert!(
            approx::relative_eq!(states[0], expected, max_relative = 1e-9),
            "OneCptIv SS bolus: central mismatch"
        );

        // OneCptIv SS infusion — was using non-SS formula before fix
        let d = infusion_ss(amt, rate, ii);
        let states = single_dose_states(PkModel::OneCptIv, &d, tau, &p1iv);
        let expected = one_cpt_infusion_ss(&d, tau, cl, v);
        assert!(
            approx::relative_eq!(states[0], expected, max_relative = 1e-9),
            "OneCptIv SS infusion: central mismatch"
        );

        // OneCptOral SS bolus — was using one_cpt_oral_f (non-SS) before fix
        let d = bolus_ss(amt, ii);
        let states = single_dose_states(PkModel::OneCptOral, &d, tau, &p1oral);
        let expected = one_cpt_oral_f_ss(&d, tau, cl, v, ka, f_bio);
        assert!(
            approx::relative_eq!(states[1], expected, max_relative = 1e-9),
            "OneCptOral SS: central (index 1) mismatch"
        );

        // TwoCptIv SS bolus
        let d = bolus_ss(amt, ii);
        let states = single_dose_states(PkModel::TwoCptIv, &d, tau, &p2iv);
        let expected = two_cpt_iv_bolus_ss(&d, tau, cl, v, q, v2);
        assert!(
            approx::relative_eq!(states[0], expected, max_relative = 1e-9),
            "TwoCptIv SS bolus: central mismatch"
        );

        // TwoCptIv SS infusion — was using two_cpt_infusion (non-SS) before fix
        let d = infusion_ss(amt, rate, ii);
        let states = single_dose_states(PkModel::TwoCptIv, &d, tau, &p2iv);
        let expected = two_cpt_infusion_ss(&d, tau, cl, v, q, v2);
        assert!(
            approx::relative_eq!(states[0], expected, max_relative = 1e-9),
            "TwoCptIv SS infusion: central mismatch"
        );

        // TwoCptOral SS bolus — was using two_cpt_oral_f (non-SS) before fix
        let d = bolus_ss(amt, ii);
        let states = single_dose_states(PkModel::TwoCptOral, &d, tau, &p2oral);
        let expected = two_cpt_oral_f_ss(&d, tau, cl, v, q, v2, ka, f_bio);
        assert!(
            approx::relative_eq!(states[1], expected, max_relative = 1e-9),
            "TwoCptOral SS bolus: central (index 1) mismatch"
        );

        // TwoCptOral SS infusion central — was using two_cpt_infusion (non-SS) before fix
        let d = infusion_ss(amt, rate, ii);
        let states = single_dose_states(PkModel::TwoCptOral, &d, tau, &p2oral);
        let expected = two_cpt_infusion_ss(&d, tau, cl, v, q, v2);
        assert!(
            approx::relative_eq!(states[1], expected, max_relative = 1e-9),
            "TwoCptOral SS infusion: central (index 1) mismatch"
        );

        // SS states must exceed single-dose states (accumulation guard).
        let d_single = DoseEvent::new(0.0, amt, 1, 0.0, false, 0.0);
        let states_ss = single_dose_states(PkModel::OneCptOral, &bolus_ss(amt, ii), tau, &p1oral);
        let states_sd = single_dose_states(PkModel::OneCptOral, &d_single, tau, &p1oral);
        assert!(
            states_ss[1] > states_sd[1],
            "SS central must be > single-dose central (accumulation): ss={} sd={}",
            states_ss[1],
            states_sd[1]
        );
    }

    /// Transit is a 2-state `[depot, central]` model: `single_dose_states` must return the
    /// lumped unabsorbed-chain "depot" amount and the central concentration for a bolus
    /// (the `[derived]` / compartment-amount path), #386.
    #[test]
    fn single_dose_states_transit_returns_depot_and_central() {
        let mut p = PkParams::default();
        p.values[crate::types::PK_IDX_CL] = 0.5;
        p.values[crate::types::PK_IDX_V] = 10.0;
        p.values[crate::types::PK_IDX_N] = 3.0;
        p.values[crate::types::PK_IDX_MTT] = 1.5;
        let d = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
        let states = single_dose_states(PkModel::OneCptTransit, &d, 2.0, &p);
        assert_eq!(states.len(), 2, "transit exposes [depot, central]");
        assert!(
            states[0] > 0.0 && states[1] > 0.0,
            "both amounts positive mid-absorption: {states:?}"
        );
    }

    /// 2-cpt transit is a 3-state `[depot, central, peripheral]` model:
    /// `single_dose_states` must return the lumped unabsorbed-chain depot, the central
    /// concentration (matching `single_dose_concentration`), and the peripheral
    /// concentration — covering `two_cpt_transit_depot` / `_peripheral` (#386 PR D).
    #[test]
    fn single_dose_states_two_cpt_transit_returns_depot_central_periph() {
        let mut p = PkParams::default();
        p.values[crate::types::PK_IDX_CL] = 0.5;
        p.values[crate::types::PK_IDX_V] = 10.0; // V1
        p.values[crate::types::PK_IDX_Q] = 1.0;
        p.values[crate::types::PK_IDX_V2] = 20.0;
        p.values[crate::types::PK_IDX_N] = 3.0;
        p.values[crate::types::PK_IDX_MTT] = 1.5;
        let d = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
        let states = single_dose_states(PkModel::TwoCptTransit, &d, 2.0, &p);
        assert_eq!(
            states.len(),
            3,
            "2-cpt transit exposes [depot, central, periph]"
        );
        assert!(
            states.iter().all(|&x| x > 0.0),
            "all three amounts positive mid-absorption: {states:?}"
        );
        // Central state must equal the standalone concentration entry point.
        let c = single_dose_concentration(PkModel::TwoCptTransit, &d, 2.0, &p);
        assert!(
            (states[1] - c).abs() <= 1e-9 * c.abs().max(1.0),
            "central state {} vs single_dose_concentration {}",
            states[1],
            c
        );
    }

    /// IG is a 2-state `[depot, central]` model: `single_dose_states` must return the
    /// unabsorbed-IG "depot" amount and the central concentration for a bolus, covering
    /// the `single_dose_states` IG arm and `one_cpt_ig_depot` / `one_cpt_ig_f` (#790).
    /// Params in-domain (`ke = 0.05 < 1/(2·MAT·CV²) = 1.111`).
    #[test]
    fn single_dose_states_ig_returns_depot_and_central() {
        let mut p = PkParams::default();
        p.values[crate::types::PK_IDX_CL] = 0.5;
        p.values[crate::types::PK_IDX_V] = 10.0;
        p.values[crate::types::PK_IDX_MAT] = 1.5;
        p.values[crate::types::PK_IDX_CV2] = 0.3;
        let d = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
        let states = single_dose_states(PkModel::OneCptIg, &d, 2.0, &p);
        assert_eq!(states.len(), 2, "IG exposes [depot, central]");
        assert!(
            states[0] > 0.0 && states[1] > 0.0,
            "both amounts positive mid-absorption: {states:?}"
        );
        // Central state must equal the standalone concentration entry point.
        let c = single_dose_concentration(PkModel::OneCptIg, &d, 2.0, &p);
        assert!((states[1] - c).abs() <= 1e-9 * c.abs().max(1.0));
    }

    /// 2-cpt IG is a 3-state `[depot, central, peripheral]` model, covering
    /// `two_cpt_ig_depot` / `two_cpt_ig_f` / `two_cpt_ig_peripheral` (#790).
    #[test]
    fn single_dose_states_two_cpt_ig_returns_depot_central_periph() {
        let mut p = PkParams::default();
        p.values[crate::types::PK_IDX_CL] = 0.5;
        p.values[crate::types::PK_IDX_V] = 10.0; // V1
        p.values[crate::types::PK_IDX_Q] = 1.0;
        p.values[crate::types::PK_IDX_V2] = 20.0;
        p.values[crate::types::PK_IDX_MAT] = 1.5;
        p.values[crate::types::PK_IDX_CV2] = 0.3;
        let d = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
        let states = single_dose_states(PkModel::TwoCptIg, &d, 2.0, &p);
        assert_eq!(states.len(), 3, "2-cpt IG exposes [depot, central, periph]");
        assert!(
            states.iter().all(|&x| x > 0.0),
            "all three amounts positive mid-absorption: {states:?}"
        );
        let c = single_dose_concentration(PkModel::TwoCptIg, &d, 2.0, &p);
        assert!((states[1] - c).abs() <= 1e-9 * c.abs().max(1.0));
    }

    /// A `PkParams` whose `CL` and `V` pass the degenerate-parameter early return (#2104),
    /// so a direct call reaches the dispatch below it.
    fn pk_valid_cl_v() -> PkParams {
        let mut p = PkParams::default();
        p.values[crate::types::PK_IDX_CL] = 5.0;
        p.values[crate::types::PK_IDX_V] = 50.0;
        p
    }

    /// SS doses reroute to the ODE twin before reaching the closed forms (`effective_for`,
    /// #719), so the SS arms of `single_dose_concentration` / `single_dose_states` are
    /// `unreachable!` — calling them directly with an SS dose must hit those guards (covers the
    /// defensive arms).
    #[test]
    #[should_panic(expected = "one_cpt_ig SS doses reroute")]
    fn single_dose_concentration_one_cpt_ig_ss_unreachable() {
        let p = pk_valid_cl_v();
        let ss = DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0);
        let _ = single_dose_concentration(PkModel::OneCptIg, &ss, 2.0, &p);
    }

    #[test]
    #[should_panic(expected = "two_cpt_ig SS doses reroute")]
    fn single_dose_states_two_cpt_ig_ss_unreachable() {
        let p = pk_valid_cl_v();
        let ss = DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0);
        let _ = single_dose_states(PkModel::TwoCptIg, &ss, 2.0, &p);
    }

    /// SS doses reroute to the ODE twin (`effective_for`, #719) before reaching the transit
    /// closed form, so the SS arm of `single_dose_concentration` is `unreachable!`; calling it
    /// directly with an SS dose must hit that guard (covers the defensive arm + its message).
    #[test]
    #[should_panic(expected = "two_cpt_transit SS doses reroute")]
    fn single_dose_concentration_two_cpt_transit_ss_unreachable() {
        let p = pk_valid_cl_v();
        let ss = DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0);
        let _ = single_dose_concentration(PkModel::TwoCptTransit, &ss, 2.0, &p);
    }

    /// The mirror SS `unreachable!` arm in `single_dose_states`.
    #[test]
    #[should_panic(expected = "two_cpt_transit SS doses reroute")]
    fn single_dose_states_two_cpt_transit_ss_unreachable() {
        let p = pk_valid_cl_v();
        let ss = DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0);
        let _ = single_dose_states(PkModel::TwoCptTransit, &ss, 2.0, &p);
    }

    /// Regression: `single_dose_concentration` and `single_dose_states[central]` must
    /// agree for OneCptOral infusion doses.  Before the fix both functions used the
    /// oral Bateman formula for infusions (which bypass the depot), so both returned
    /// a consistent but wrong value.  After the fix both route to the IV infusion
    /// formula, keeping ipred == compartment_states[central] and matching NONMEM.
    #[test]
    fn single_dose_concentration_matches_states_for_one_cpt_oral_infusion() {
        fn infusion(amt: f64, rate: f64) -> DoseEvent {
            DoseEvent::new(0.0, amt, 1, rate, false, 0.0)
        }
        fn infusion_ss(amt: f64, rate: f64, ii: f64) -> DoseEvent {
            DoseEvent::new(0.0, amt, 1, rate, true, ii)
        }

        let cl = 3.0_f64;
        let v = 50.0_f64;
        let ka = 1.2_f64; // ka present but must NOT affect infusion formula
        let f_bio = 0.9_f64;
        let ii = 12.0_f64;
        let amt = 100.0_f64;
        let rate = 50.0_f64; // 2 h infusion
        let tau = 5.0_f64;

        let mut p = PkParams::default();
        p.values[crate::types::PK_IDX_CL] = cl;
        p.values[crate::types::PK_IDX_V] = v;
        p.values[crate::types::PK_IDX_KA] = ka;
        p.values[crate::types::PK_IDX_F] = f_bio;

        // Non-SS infusion: central (index 1) must equal single_dose_concentration
        let d = infusion(amt, rate);
        let conc = single_dose_concentration(PkModel::OneCptOral, &d, tau, &p);
        let states = single_dose_states(PkModel::OneCptOral, &d, tau, &p);
        assert!(
            approx::relative_eq!(conc, states[1], max_relative = 1e-9),
            "OneCptOral non-SS infusion: concentration={conc} ≠ central state={} \
             (depot is index 0, central is index 1)",
            states[1]
        );
        // Depot must be zero for an infusion (bypasses depot)
        assert_eq!(states[0], 0.0, "OneCptOral infusion: depot must be 0.0");

        // SS infusion
        let d = infusion_ss(amt, rate, ii);
        let conc = single_dose_concentration(PkModel::OneCptOral, &d, tau, &p);
        let states = single_dose_states(PkModel::OneCptOral, &d, tau, &p);
        assert!(
            approx::relative_eq!(conc, states[1], max_relative = 1e-9),
            "OneCptOral SS infusion: concentration={conc} ≠ central state={}",
            states[1]
        );
        assert_eq!(states[0], 0.0, "OneCptOral SS infusion: depot must be 0.0");

        // Bolus must be unchanged (oral formula correct for bolus)
        let d_bolus = DoseEvent::new(0.0, amt, 1, 0.0, false, 0.0);
        let conc = single_dose_concentration(PkModel::OneCptOral, &d_bolus, tau, &p);
        let states = single_dose_states(PkModel::OneCptOral, &d_bolus, tau, &p);
        assert!(
            approx::relative_eq!(conc, states[1], max_relative = 1e-9),
            "OneCptOral bolus: concentration={conc} ≠ central state={}",
            states[1]
        );
        assert!(states[0] > 0.0, "OneCptOral bolus: depot must be > 0");
    }

    /// Regression test: `single_dose_concentration` and `single_dose_states[central]`
    /// must agree for TwoCptOral infusion doses (both SS and non-SS).
    ///
    /// Before the fix, `single_dose_concentration` used the oral formula
    /// (`two_cpt_oral_f_ss` / `two_cpt_oral_f`) for infusion doses, while
    /// `single_dose_states` correctly used the IV infusion formula (since infusions
    /// bypass the depot).  This caused `ipred ≠ compartment_states[central]` for
    /// any TwoCptOral model with infusion doses.
    #[test]
    fn single_dose_concentration_matches_states_for_two_cpt_oral_infusion() {
        fn infusion(amt: f64, rate: f64) -> DoseEvent {
            DoseEvent::new(0.0, amt, 1, rate, false, 0.0)
        }
        fn infusion_ss(amt: f64, rate: f64, ii: f64) -> DoseEvent {
            DoseEvent::new(0.0, amt, 1, rate, true, ii)
        }

        let cl = 5.0_f64;
        let v = 80.0_f64;
        let q = 2.0_f64;
        let v2 = 40.0_f64;
        let ka = 1.2_f64;
        let f_bio = 1.0_f64;
        let ii = 24.0_f64;
        let amt = 200.0_f64;
        let rate = 100.0_f64; // 2 h infusion
        let tau = 8.0_f64;

        let mut p = PkParams::default();
        p.values[crate::types::PK_IDX_CL] = cl;
        p.values[crate::types::PK_IDX_V] = v;
        p.values[crate::types::PK_IDX_Q] = q;
        p.values[crate::types::PK_IDX_V2] = v2;
        p.values[crate::types::PK_IDX_KA] = ka;
        p.values[crate::types::PK_IDX_F] = f_bio;

        // Non-SS infusion: single_dose_concentration must equal central state
        let d = infusion(amt, rate);
        let conc = single_dose_concentration(PkModel::TwoCptOral, &d, tau, &p);
        let states = single_dose_states(PkModel::TwoCptOral, &d, tau, &p);
        // central is index 1 for TwoCptOral: [depot=0, central=1, periph=2]
        assert!(
            approx::relative_eq!(conc, states[1], max_relative = 1e-9),
            "TwoCptOral non-SS infusion: concentration={conc} ≠ central state={}",
            states[1]
        );

        // SS infusion
        let d = infusion_ss(amt, rate, ii);
        let conc = single_dose_concentration(PkModel::TwoCptOral, &d, tau, &p);
        let states = single_dose_states(PkModel::TwoCptOral, &d, tau, &p);
        assert!(
            approx::relative_eq!(conc, states[1], max_relative = 1e-9),
            "TwoCptOral SS infusion: concentration={conc} ≠ central state={}",
            states[1]
        );

        // Bolus should be unchanged (oral formula is correct for bolus)
        let d = DoseEvent::new(0.0, amt, 1, 0.0, false, 0.0);
        let conc = single_dose_concentration(PkModel::TwoCptOral, &d, tau, &p);
        let states = single_dose_states(PkModel::TwoCptOral, &d, tau, &p);
        assert!(
            approx::relative_eq!(conc, states[1], max_relative = 1e-9),
            "TwoCptOral bolus: concentration={conc} ≠ central state={}",
            states[1]
        );
    }

    /// Closes PR #327 review finding #5: bioavailability `F` must scale **every**
    /// element of the `single_dose_states` vector — depot, central, and all
    /// peripheral compartments — on every route. The analytical states are linear
    /// in the dose, so the invariant is `states(F) == F · states(F=1)`
    /// element-wise. This pins down what the central-only
    /// `single_dose_concentration` checks cannot see: a depot wrongly scaled, a
    /// scale missed on a peripheral arm, or `F` applied twice. `compartment_states`
    /// feeds `[derived]`/`[output]` expressions, so a silent error here would
    /// surface in user output.
    #[test]
    fn single_dose_states_scales_every_compartment_by_f() {
        let f = 0.4_f64;
        let tau = 6.0_f64;
        let ii = 12.0_f64;
        let amt = 100.0_f64;

        // Full structural params for up to three compartments; F set per call.
        let params = |f_bio: f64| {
            let mut p = PkParams::default();
            p.values[crate::types::PK_IDX_CL] = 3.0;
            p.values[crate::types::PK_IDX_V] = 50.0;
            p.values[crate::types::PK_IDX_Q] = 2.0;
            p.values[crate::types::PK_IDX_V2] = 40.0;
            p.values[crate::types::PK_IDX_KA] = 1.1;
            p.values[crate::types::PK_IDX_Q3] = 1.5;
            p.values[crate::types::PK_IDX_V3] = 30.0;
            p.values[crate::types::PK_IDX_F] = f_bio;
            p
        };
        let p1 = params(1.0);
        let pf = params(f);

        let models = [
            PkModel::OneCptIv,
            PkModel::OneCptOral,
            PkModel::TwoCptIv,
            PkModel::TwoCptOral,
            PkModel::ThreeCptIv,
            PkModel::ThreeCptOral,
        ];
        // Bolus into compartment 1, non-SS and SS. `F` scales the bioavailable
        // AMOUNT (`F·AMT` into central, or the oral depot load), so it scales
        // every compartment's state uniformly. Infusions are excluded here: for a
        // rate-defined infusion `F` scales the *duration* (#419), which reshapes
        // the profile rather than scaling it - that path is anchored by
        // `iv_bolus_and_infusion_apply_f_matching_nonmem_closed_form` and the
        // cross-path equality tests instead.
        let doses = [
            ("bolus", DoseEvent::new(0.0, amt, 1, 0.0, false, 0.0)),
            ("SS bolus", DoseEvent::new(0.0, amt, 1, 0.0, true, ii)),
        ];

        for model in models {
            for (label, dose) in &doses {
                let s1 = single_dose_states(model, dose, tau, &p1);
                let sf = single_dose_states(model, dose, tau, &pf);
                assert_eq!(
                    s1.len(),
                    sf.len(),
                    "{model:?} {label}: state-vector length must not depend on F"
                );
                // Guard against a vacuous pass (all-zero states scale trivially).
                assert!(
                    s1.iter().any(|&x| x.abs() > 1e-9),
                    "{model:?} {label}: all states ~0 at F=1 — test would be vacuous"
                );
                for (i, (&a, &b)) in s1.iter().zip(sf.iter()).enumerate() {
                    assert!(
                        approx::relative_eq!(b, f * a, max_relative = 1e-9, epsilon = 1e-12),
                        "{model:?} {label}: state[{i}] = {b} at F={f}, expected {} = F·{a} \
                         — F must scale every compartment exactly once",
                        f * a
                    );
                }
            }
        }
    }
}
