pub mod absorption;
pub mod absorption_walk;
pub mod analytical_absorption;
pub mod event_driven;
pub mod modified_release;
pub mod ode_template;
pub mod one_compartment;
pub mod sharing_scope;
pub mod three_compartment;
pub mod topology;
pub mod two_compartment;

use crate::types::{CompiledModel, DoseEvent, PkModel, PkParams, ScalingSpec, Subject};

pub use one_compartment::*;
pub use three_compartment::*;
pub use two_compartment::*;

#[inline]
pub fn model_uses_time_builtin(model: &CompiledModel) -> bool {
    crate::parser::model_parser::compiled_model_uses_time_builtin(model)
}

/// Does this model depend on model time *anywhere* — the individual-parameter
/// program (as [`model_uses_time_builtin`] alone reports) **or** the `[odes]`
/// right-hand side itself?
///
/// The routing guards that pick between the event-driven predictor, the
/// modified-release closed form, and the plain dense integration used to ask
/// only the first question. An `[odes]` RHS reading `TAD`/`TAFD`/`T`/`TIME` is
/// just as non-autonomous, and neither of the two static paths can represent it:
///
///   - the **closed form** identifies the disposition by probing the RHS at two
///     times with `tad` pinned to `0.0`, so a `TAD` term is invisible to it and
///     was silently dropped from the predictions — 8× wrong at 12 h (#1124);
///   - the **dense** predictor started its first segment at the dose *record*
///     rather than the arrival, so under a lagtime its `TAD` anchor had no
///     qualifying dose and it returned `NaN` for every observation (#1110).
///     #1073 has since given it a pre-arrival anchor, so that path is finite
///     too; the reroute stands on the closed-form defect above, and on keeping
///     one engine answering a non-autonomous RHS rather than two.
///
/// Routing these models to the event-driven predictor — which places doses on
/// its timeline at `d.time + lag`, so `TAD` anchors correctly — is what both
/// defects have in common, and it is the path the NONMEM `TAD`-in-`$DES` anchors
/// (`nonmem_anchor/tad_lag_{A,B}`) already validate.
///
/// **What the reroute is measured to preserve.** The predicate also moves models
/// the closed form declined for reasons of its own — `init(...)`, a
/// `dose_attr_map`, steady-state or infusion doses — off the *dense* path, which
/// they were on before. Measured, dense vs event-driven, worst relative
/// difference over the observations:
///
/// | model | measured |
/// |---|---|
/// | bare `TAFD` RHS, no lagtime, plain bolus | `6e-13` |
/// | `init(central) = BASE` + `TAFD` RHS | `4.8e-11` |
/// | `TAD` RHS **under a lagtime**, obs post-arrival | `5.8e-12` |
/// | …same, `lag = 3.0` (a larger anchor shift) | `5.3e-12` |
///
/// The lagtime rows are only measurable since #1073: before it the dense path
/// returned `NaN` for every observation on such a model — that being #1110, the
/// defect this reroute exists to route around — so there was nothing to compare
/// against. #1073 gave the dense predictor the same pre-arrival anchor, and the
/// two engines now agree.
///
/// One case where the two are still **not** comparable, so no agreement is
/// claimed for it: **a steady-state dose** whose RHS reads an *absolute* clock.
/// `T`/`TIME` under `SS=1` sits 67% from its own explicit dose train — an absolute
/// clock has no periodic limit for the run-in to converge to — and `TAFD` reads
/// `NaN`; both are #1139's remaining half. `TAD` under `SS=1` *is* comparable
/// since #1139: it reproduces its own explicit train to 3.8e-13 and NONMEM to
/// 7.4e-9. Rerouting neither caused nor cured any of this.
///
/// Regression tests: `rerouting_a_bare_tafd_rhs_does_not_move_the_predictions`
/// and `rerouting_an_init_seeded_rhs_does_not_move_the_predictions`.
///
/// **What the reroute costs, and why it is still whole-class.** The predicate is
/// deliberately not narrowed to the MR-eligible shape (`!spec.input_rate.
/// is_empty()`), even though that is the only shape #1124 itself could reach — a
/// model with no built-in forcing was declined by `mr_scope` anyway and was
/// already on the *dense* path. Measured, `ci-fast`, 2-state `[odes]`, two doses
/// over 24 h at `reltol = 1e-8`, dense vs event-driven:
///
/// | observations | ratio |
/// |---|---|
/// | 8  | 1.04× |
/// | 16 | 1.15× |
/// | 32 | 1.38× |
///
/// It grows with observation density because the dense driver breaks only at
/// dose/reset boundaries and reads observations through `saveat`, while the
/// event-driven walk calls `solve_ode` once per record and each call restarts the
/// step-size history. (For the MR-shaped cell the fix is actually about, the cost
/// is far larger and unavoidable — ~40×, closed form to integration.)
///
/// That is paid to keep **one** engine answering a non-autonomous RHS. Both
/// engines are correct here since #1073, but they are correct *differently*: they
/// break the timeline at different points, so value, gradient, and the `[derived]`
/// grid would agree only to solver tolerance and would drift apart under a
/// lagtime. Splitting the class by `input_rate` re-creates exactly the
/// value-vs-gradient and value-vs-`[derived]` engine splits that the gates in
/// `stats/likelihood.rs`, `api/output_columns.rs` and `sens/ode_provider.rs` each
/// had to be widened to close. Revisit with a measurement on a dense-sampling
/// design if the ratio starts to matter; the predicate itself is not a hot-loop
/// cost (both flags are resolved once in `build_ode_spec` at parse time, and this
/// is `#[inline]` over three `bool` reads).
/// **Which flag the `[odes]` half asks, and why it is the narrow one.** This
/// predicate asks whether the *PK dynamics* are non-autonomous, so it asks
/// `OdeRhsProgram::pk_reads_model_time` — the same walk with the injected
/// joint-PK-TTE `d/dt(__chz_<cmt>)` hazard lines excluded (#1166). A standard
/// joint model has a Weibull or Gompertz baseline hazard, which reads `TIME` by
/// definition, while its PK block is time-invariant; asking the wide flag made
/// every such model decline #570's shared solve (measured 1.82× on one objective
/// evaluation) and leave the dense driver, for a property it does not have.
///
/// The steady-state gradient gates (`sens/ode_provider.rs`) ask the PK-only
/// `pk_reads_absolute_time()` (#2093), which is safe only because a joint
/// PK-TTE subject with an SS dose is declined before them (#1210): SS
/// equilibration integrates the *augmented* system, `__chz` included, and a
/// time-reading hazard line breaks that state's cycle recurrence just as a
/// time-reading PK line would.
///
/// The Form C `readout_program` is deliberately **not** folded in either. Both
/// engines evaluate the readout under a `ModelTimeGuard`, so a `TIME`-reading
/// readout is already correct on the dense driver and folding it in would decline
/// the shared solve for nothing. `pk/modified_release.rs` hand-pairs it for a
/// closed-form-only reason (its linearity probe cannot see the readout's time
/// term), which does not apply here.
///
/// **What keeps a joint model out of the closed form is `identify_disposition`,
/// not this predicate.** `mr_scope` asks this predicate too, so before #1166 a
/// time-dependent hazard declined the closed form here; now it does not, and the
/// decline rests entirely on `identify_disposition`. That bails at
/// `n_states > 2`, which covers every joint model with two or more PK states —
/// but a *one*-state PK block plus the injected accumulator is exactly two and
/// reaches the probe. Measured on such a model (one state, a built-in forcing, a
/// Form C readout, and a hazard linear in the state so it clears the "no
/// autonomous term" check): `identify_disposition` still returns `None`, because
/// the accumulator makes the disposition triangular — the hazard row reads the
/// state but nothing returns from it — and that is not a disposition the closed
/// form recognises. So the exposure is closed, by a different gate than the one
/// that used to close it. The invariant is pinned by
/// `pk::modified_release::tests::a_joint_pk_tte_model_never_reaches_the_closed_form`
/// rather than enforced; whether to enforce it is #1209.
#[inline]
pub fn model_uses_time_anywhere(model: &CompiledModel) -> bool {
    model_uses_time_builtin(model)
        || model
            .ode_spec
            .as_ref()
            .is_some_and(|s| s.rhs_program.as_ref().is_some_and(|p| p.pk_reads_model_time()))
}

#[inline]
pub fn pk_params_at_time(
    model: &CompiledModel, theta: &[f64], eta: &[f64],
    covariates: &std::collections::HashMap<String, f64>, time: f64,
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
pub fn absorption_flip_flop_at(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64],
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
pub fn invalid_cl_v_records<K: AsRef<[f64]>>(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta_bsv: &[f64], kappas: &[K],
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
            subject.reset_times.iter().copied().zip(&params.reset).filter(|&(t, _)| !evid4_twin(t)),
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
/// `crate::api::check_absorption_flip_flop_no_twin`; see
/// [`crate::types::CompiledModel::absorption_ode_equivalent`] for what "no twin" covers
/// (an open-ended set, #1008).
pub fn effective_model_for_eval<'a>(
    model: &'a CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64],
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
pub fn effective_model_for_eval_iov<'a>(
    model: &'a CompiledModel, subject: &Subject, theta: &[f64], stacked_eta: &[f64],
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
pub fn absorption_walk_or_twin<'a>(
    model: &'a CompiledModel, subject: &Subject, ev: &EventPkParams,
) -> &'a CompiledModel {
    if absorption_walk::walk_serves(model.pk_model, subject, &ev.dose, &ev.obs, &ev.pk_only) {
        return model;
    }
    model.absorption_ode_equivalent.as_ref().map_or(model, |twin| twin.built())
}

/// `η` per record for an IOV subject: `[η_bsv, κ_g]` for occasion `occ` in group `g`
/// ([`crate::stats::likelihood::iov_occasion_groups`] order), `κ = 0` for an occasion
/// with no group — an occasion-less caller asks for `u32::MAX` and gets that.
pub fn iov_occasion_eta<'a, K: AsRef<[f64]>>(
    layout: &'a crate::stats::stacked_layout::StackedLayout, eta_bsv: &'a [f64], kappas: &'a [K],
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
pub fn fill_iov_record_params(
    model: &CompiledModel, subject: &Subject, theta: &[f64],
    combined_for: &dyn Fn(u32) -> Vec<f64>, dose: &mut Vec<PkParams>, obs: &mut Vec<PkParams>,
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
        push(pk_only, occ, subject.pk_only_cov(m), subject.param_time_pk_only(m));
    }
}

#[cfg(any(test, feature = "test-support"))]
thread_local! {
    /// Test switch: predict IOV subjects from per-record copies even where the record table
    /// applies, for the parity test that pins the two against each other.
    pub static IOV_TABLE_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[inline]
pub fn iov_table_off() -> bool {
    #[cfg(any(test, feature = "test-support"))]
    {
        IOV_TABLE_OFF.with(|c| c.get())
    }
    #[cfg(not(any(test, feature = "test-support")))]
    {
        false
    }
}

/// [`fill_iov_record_params`] into a table of distinct snapshots and one index per record
/// (#2549). An `ADDL`-implied dose that inherits its parent's attributes gets its own
/// entry first (copy-on-write).
pub fn fill_iov_record_table(
    model: &CompiledModel, subject: &Subject, theta: &[f64],
    combined_for: &dyn Fn(u32) -> Vec<f64>, table: &mut Vec<PkParams>, dose_ix: &mut Vec<u32>,
    obs_ix: &mut Vec<u32>, pk_only_ix: &mut Vec<u32>,
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
                table.push(pk_params_at_time(model, theta, &combined_for(occ), cov, time));
                (table.len() - 1) as u32
            }
        };
        *held = Some(i);
        out.push(i);
    };
    for d in 0..subject.doses.len() {
        let occ = dose_row_occasion(subject, d).unwrap_or(u32::MAX);
        push(table, &mut held, dose_ix, occ, subject.dose_cov(d), subject.dose_param_time(d));
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
        push(table, &mut held, obs_ix, occ, subject.obs_cov(j), subject.param_time_obs(j));
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
pub struct RecordParamMemo {
    pub time_free: bool,
    pub last: Option<(u32, *const std::collections::HashMap<String, f64>, u64)>,
    pub value: PkParams,
}

impl RecordParamMemo {
    /// `time_free` trusts the parsed program's `TIME` flag ([`model_uses_time_builtin`]),
    /// as routing does. A model with no parsed program never shares a time-free entry.
    pub fn new(model: &CompiledModel) -> Self {
        RecordParamMemo {
            time_free: model.indiv_param_partials.indiv_param_program.is_some()
                && !model_uses_time_builtin(model),
            last: None,
            value: PkParams::default(),
        }
    }

    /// The parameters for these inputs: the last evaluation's when every input matches,
    /// else a fresh `eval()`, kept for the next record.
    pub fn get_ref(
        &mut self, occ: u32, cov: &std::collections::HashMap<String, f64>, time: f64,
        eval: impl FnOnce() -> PkParams,
    ) -> &PkParams {
        if !self.same(occ, cov, time) {
            self.value = eval();
        }
        &self.value
    }

    /// Whether these inputs are the last call's, which they become.
    fn same(&mut self, occ: u32, cov: &std::collections::HashMap<String, f64>, time: f64) -> bool {
        let key = (occ, cov as *const _, if self.time_free { 0 } else { time.to_bits() });
        self.last.replace(key) == Some(key)
    }
}

/// Divide each prediction in-place by the scale derived from `model.scaling`
/// (**divisive**: `obs_scale = 1000` maps mg/L → mg/mL). `ScalingSpec::None` is a no-op.
/// `ExpressionScale` is evaluated on a subject-static `pk_param_fn` snapshot. Form C
/// (`y = <expr>`) does not come through here.
#[inline]
pub fn apply_scaling(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64], preds: &mut [f64],
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
pub fn obs_scale_divisors(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64], cmts: &[usize],
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
        static DEFAULT_PK: PkParams = PkParams { values: [0.0; crate::types::MAX_PK_PARAMS] };
        &DEFAULT_PK
    };
    model.scaling.build_obs_scale_array(theta, eta, &subject.covariates, pk_ref, cmts)
}

/// Divide one prediction by its `[scaling]` divisor, or `NaN` when the divisor is
/// not a finite positive number. [`ScalingSpec::build_obs_scale_array`] already
/// encodes invalid scales as `NaN`; the check here also covers a missing entry.
#[inline]
pub fn divide_by_obs_scale(pred: f64, scale: f64) -> f64 {
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
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64], preds: &mut [f64],
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
        Some(states.iter().map(|s| s.first().copied().unwrap_or(0.0)).collect())
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
        *pred = ar.readout.eval(&state, &pk_i.values, theta, eta, cov, obs_cmt, t);
    }
}

/// Layer analytical initial-compartment amounts onto `preds` (#521): for a linear
/// disposition, `A₀` in compartment `c` at t=0 is the impulse response of an F-bypassed
/// bolus. `A₀` is evaluated once per subject. Called **before** `[scaling]` and the
/// log-transform, matching the ODE path. No-op for ODE models and without
/// `[initial_conditions]`. `preds` is parallel to `subject.obs_times`.
pub fn add_analytical_init(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64], preds: &mut [f64],
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
pub fn add_analytical_init_with(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64], amount_pk: &PkParams,
    obs_decay_pk: Option<&[PkParams]>, preds: &mut [f64],
) {
    let first_reset = subject.reset_times.iter().copied().fold(f64::INFINITY, f64::min);
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
pub fn analytical_init_concentration(
    pk_model: PkModel, cmt: usize, a0: f64, t: f64, p: &PkParams,
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

/// Generic form of [`analytical_init_concentration`] over `PkNum`: in `Dual2` it gives
/// the exact `∂C/∂A₀`, `∂C/∂(CL,V,…)` for the analytic provider (#524). `F = 1`; returns 0
/// for unsupported compartments.
#[allow(clippy::too_many_arguments)]
pub fn analytical_init_concentration_g<T: crate::sens::num::PkNum>(
    pk_model: PkModel, cmt: usize, a0: T, t: T, cl: T, v: T, q: T, v2: T, ka: T, q3: T, v3: T,
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
        let missing: Vec<usize> =
            observed_cmts.iter().copied().filter(|c| !map.contains_key(c)).collect();
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
            let missing: Vec<usize> =
                observed_cmts.iter().copied().filter(|c| !map.contains_key(c)).collect();
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
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64],
) -> EventPkParams {
    let mut out = EventPkParams::with_capacity_for(subject);
    compute_event_pk_params_into(model, subject, theta, eta, &mut out);
    out
}

/// Whether a subject needs **per-event** PK rather than one t=0 snapshot: a covariate
/// changes on a dose/obs row (`has_tv_covariates`) or only on an EVID=2 row
/// (`pk_only_covariates`), or the model reads `TIME`. Single source of truth (#700) for
/// [`compute_event_pk_params_into`] and the adaptive driver's `event_pk` / `auc_target`.
pub fn subject_needs_per_event_pk(model: &CompiledModel, subject: &Subject) -> bool {
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
pub fn ode_cmt_states_approximate(model: &CompiledModel, subject: &Subject) -> bool {
    model.ode_spec.is_some() && (subject.has_tv_covariates() || model_uses_time_builtin(model))
}

/// The occasion label for an EVID=3/4 **reset row's** `$PK` on the `OCC`-column IOV path
/// (#1133, #1153): the row's own `OCC`, or `None` (κ = 0) when the subject stores no reset
/// labels. Measured in `nonmem_anchor/reset_init_snapshot_J.ctl` (`A_0` seeds under the
/// reset row's occasion: 42.0, not 14.0). No guess from a neighbouring record: a subject
/// read from data has every label or none. Takes the reset's **index**, not its time, so
/// close resets cannot alias.
pub fn reset_row_occasion(subject: &Subject, r: usize) -> Option<u32> {
    subject.reset_occasions.get(r).copied()
}

/// The occasion label for an EVID=2 or EVID=0/MDV=1 **pk-only row's** `$PK` on the
/// `OCC`-column IOV path (#1876): the row's own `OCC` (measured in
/// `nonmem_anchor/iov_pk_only_*_X.ctl`), or `None` (κ = 0) when the subject stores no
/// pk-only labels, as in [`reset_row_occasion`].
pub fn pk_only_row_occasion(subject: &Subject, m: usize) -> Option<u32> {
    subject.pk_only_occasions.get(m).copied()
}

/// The occasion label for dose `d`'s `$PK` on the IOV path (#2065), read by
/// [`predict_iov`] and both `sens/` walks: the dose's own label, or `None` (κ = 0) when
/// `dose_occasions` is short (#1153).
pub fn dose_row_occasion(subject: &Subject, d: usize) -> Option<u32> {
    subject.dose_occasions.get(d).copied()
}

/// The κ group of dose `d` ([`dose_row_occasion`] through `occ_to_k`), as the `sens/`
/// walks seed it. `None` is the κ = 0 dose, where production's `combined_for` lands on
/// a miss — the treatment [`pk_only_row_group`] gives an ungrouped pk-only row.
pub fn dose_row_group(
    subject: &Subject, d: usize, occ_to_k: &std::collections::HashMap<u32, usize>,
) -> Option<usize> {
    dose_row_occasion(subject, d).and_then(|occ| occ_to_k.get(&occ).copied())
}

/// The κ group of pk-only row `m` ([`pk_only_row_occasion`] through `occ_to_k`); `None`
/// is the κ = 0 row.
pub fn pk_only_row_group(
    subject: &Subject, m: usize, occ_to_k: &std::collections::HashMap<u32, usize>,
) -> Option<usize> {
    pk_only_row_occasion(subject, m).and_then(|occ| occ_to_k.get(&occ).copied())
}

/// The κ group of reset row `r` ([`reset_row_occasion`] through `occ_to_k`), as the
/// `sens/` providers seed it — [`pk_only_row_group`]'s rule for a reset row.
pub fn reset_row_group(
    subject: &Subject, r: usize, occ_to_k: &std::collections::HashMap<u32, usize>,
) -> Option<usize> {
    reset_row_occasion(subject, r).and_then(|occ| occ_to_k.get(&occ).copied())
}

/// Fill `out` with one `$PK` snapshot per dose — the single spelling of where the engine
/// resolves a dose attribute, shared by [`compute_event_pk_params_into`] and
/// [`compute_dose_pk_params_into`]. `constant` is `Some(p)` for a parameter-static
/// subject so the caller's snapshot is reused (one `$PK` call less per evaluation).
#[inline]
pub fn fill_dose_pk_params(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64],
    constant: Option<PkParams>, out: &mut Vec<PkParams>,
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
pub type PkInputs<'a> = (&'a std::collections::HashMap<String, f64>, f64);

/// The `$PK` inputs of each dose record, in order. This and its three siblings are the
/// one spelling of which snapshot and time a record's `$PK` reads, for the f64
/// materializer and for [`record_pk_inputs`].
pub fn dose_pk_inputs(subject: &Subject) -> impl Iterator<Item = PkInputs<'_>> {
    (0..subject.doses.len()).map(|k| (subject.dose_cov(k), subject.dose_param_time(k)))
}

/// The `$PK` inputs of each observation record, in order.
pub fn obs_pk_inputs(subject: &Subject) -> impl Iterator<Item = PkInputs<'_>> {
    (0..subject.obs_times.len()).map(|j| (subject.obs_cov(j), subject.param_time_obs(j)))
}

/// The `$PK` inputs of each PK-only (`EVID=2`) record, in order.
pub fn pk_only_pk_inputs(subject: &Subject) -> impl Iterator<Item = PkInputs<'_>> {
    (0..subject.pk_only_times.len())
        .map(|m| (subject.pk_only_cov(m), subject.param_time_pk_only(m)))
}

/// The `$PK` inputs of each reset (`EVID=3/4`) record, in order.
pub fn reset_pk_inputs(subject: &Subject) -> impl Iterator<Item = PkInputs<'_>> {
    (0..subject.reset_times.len())
        .map(|r| (subject.reset_cov(r), subject.param_time_at(subject.reset_times[r])))
}

/// Every dose, observation and PK-only record of `subject`, in that order, as
/// `(occasion label, covariate snapshot, parameter time)` — the records the closed-form
/// recorder evaluates `$PK` over. An unlabelled dose/PK-only row is `u32::MAX`, an
/// unlabelled observation 0.
pub fn record_pk_inputs(
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
/// parsed model is trusted to have the `pk_param_fn` its program describes: `pk_param_fn` is a
/// public field, and replacing it on a parsed model with a closure that reads a column the
/// program does not reference would share records that differ in that column, silently. No
/// code outside tests replaces it.
pub fn first_record_with_same_pk_inputs<'a>(
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
pub fn fill_per_record_pk_params(
    model: &CompiledModel, subject: &Subject, site: sharing_scope::ShareSite, theta: &[f64],
    eta: &[f64], records: &[&[(&std::collections::HashMap<String, f64>, f64)]],
    outs: &mut [&mut Vec<PkParams>],
) {
    debug_assert_eq!(records.len(), outs.len());
    // `locate` indexes each list from 0; both callers clear `outs` first.
    debug_assert!(outs.iter().all(|o| o.is_empty()));
    let first = sharing_scope::first_record_with_same_pk_inputs(
        model,
        subject,
        site,
        records.iter().flat_map(|r| r.iter().map(|&(cov, _)| (None, cov))),
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
pub const PARENT_DOSE_ATTRS: [crate::types::DoseAttr; 1] = [crate::types::DoseAttr::Lag];

/// `(k, parent, slot)` for every `ADDL`-implied dose `k` and every slot of
/// [`PARENT_DOSE_ATTRS`] its compartment reads, then the route lag of every input-rate
/// forcing its dose feeds. The slot is the one the engine resolves for that dose
/// (`ALAG{cmt}`, else the bare lag; the forcing's `lag_slot`). A route lag is a depot
/// `ALAG`, which NONMEM fixes at the parent row (`nonmem_anchor/addl_lag`).
pub fn parent_dose_attr_copies<'a>(
    model: &'a CompiledModel, subject: &'a Subject,
) -> impl Iterator<Item = (usize, usize, usize)> + 'a {
    let map = model.active_dose_attr_map();
    let forcings = model.ode_spec.as_ref().map_or(&[][..], |o| &o.input_rate[..]);
    (0..subject.doses.len()).flat_map(move |k| {
        let parent = subject.implied_dose_parent(k);
        let dose = &subject.doses[k];
        let cmt = dose.cmt_1based();
        let attrs = PARENT_DOSE_ATTRS.iter().filter_map(move |&attr| match attr {
            crate::types::DoseAttr::Lag => Some(map.lag_slot(cmt)),
            other => map.indexed_slot(other, cmt),
        });
        let route_lags =
            forcings.iter().filter(move |f| f.cmt == dose.cmt_idx()).filter_map(|f| f.lag_slot);
        parent.map(move |p| attrs.chain(route_lags).map(move |s| (k, p, s))).into_iter().flatten()
    })
}

/// Give each `ADDL`-implied dose its parent's [`PARENT_DOSE_ATTRS`] (#1909) and route lags
/// (#2367), as listed by [`parent_dose_attr_copies`]: the post-pass every per-dose builder
/// runs, so no second `$PK` call is needed.
pub fn inherit_parent_dose_attrs(model: &CompiledModel, subject: &Subject, dose: &mut [PkParams]) {
    for (k, p, slot) in parent_dose_attr_copies(model, subject) {
        dose[k].values[slot] = dose[p].values[slot];
    }
}

/// [`inherit_parent_dose_attrs`] over per-dose seed vectors of any scalar, for the
/// `sens/` walks whose dose snapshots are `Vec<T>` of duals.
pub fn inherit_parent_dose_attrs_g<T: Clone>(
    model: &CompiledModel, subject: &Subject, dose: &mut [Vec<T>],
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
pub fn push_pk_in_place(out: &mut Vec<PkParams>, f: impl FnOnce() -> PkParams) {
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
pub fn compute_dose_pk_params_into(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64], out: &mut Vec<PkParams>,
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
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64], out: &mut EventPkParams,
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
pub fn occasion_of(decision_times: &[f64], t: f64) -> Option<usize> {
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

/// Occasion-aware per-event PK for the adaptive IOV path (#701).
///
/// Like [`compute_event_pk_params`] but each observation / pk-only record is
/// evaluated with the eta of the **occasion (decision window)** active at its time
/// ([`occasion_of`]): `eta_occ[g]` for the window `g` containing the record, or
/// `eta_baseline` (BSV η with κ = 0) before the first decision. This is the IOV
/// analogue of [`predict_iov`]'s per-event occasion-κ selection, with occasion =
/// decision window instead of the OCC column. **Doses** are resolved the same way
/// (#931): a pre-scheduled base dose (#702) carries the PK of the occasion (decision
/// window) active at its administration time, so its bioavailability F resolves under
/// that occasion's κ — symmetric with a controller-injected dose, whose F is fixed
/// from the driver's per-decision LOCF snapshot at injection. A base dose administered
/// **before the first decision** has no open window, so — like an observation before the
/// first decision — it takes the baseline κ = 0 for its F; its disposition still flows
/// through the per-segment occasion PK (the standard event-driven split of dose-time
/// attributes from running clearance), so a loading dose given at t=0 with the first
/// decision at t>0 is dosed at baseline F but decays under the first window's occasion.
/// On the (common) dose-free
/// base subject `subject.doses` is empty, so `dose` stays empty and the result is
/// byte-identical to the pre-#931 path. It composes with #700's time-varying
/// covariates: each record's *covariate* snapshot (`dose_cov` / `obs_cov` /
/// `pk_only_cov`) is used alongside its occasion eta, so a model with both a TV
/// covariate and IOV κ is per-event correct in both.
///
/// **pk-only (EVID=2) convention — the decision window, not the `OCC` column.** Here
/// a pk-only record takes the κ of the decision window containing its time (like any
/// obs record), because in the reactive setting that observation genuinely occurs
/// *during* that occasion. The static [`predict_iov`] instead reads the row's own `OCC`
/// label ([`pk_only_row_occasion`], #1876). The two are consistent *within* their own
/// path (the adaptive driver's `governing_segment_occ_at` also uses [`occasion_of`], so
/// materialiser and driver agree); a future oracle that validates an adaptive IOV run
/// *with EVID=2 rows* against `predict_iov` agrees only where the row's `OCC` names the
/// decision window containing it.
///
/// **Base dose at an occasion boundary — agreement with [`predict_iov`].** The driver
/// assigns a *segment's* occasion from the record that ENDS it
/// (`governing_segment_occ_at`, #1073), and a base dose row is a record (#1148). So the
/// segment ending at a base dose takes that dose's occasion, as under `predict_iov`'s
/// end-of-interval rule. Measured on #2771: base doses at 0 and 24, observations at 12,
/// 36 and 60, decisions at 0, 24, 48 and 72 give `predict_iov`'s predictions to 2e-16
/// relative. (This `dose` snapshot only fixes each dose's *F* at its administration
/// occasion; the running disposition occasion is `governing_segment_occ_at`'s job.)
///
/// **A controller dose, and a hold.** Neither is a data record, but every decision is a
/// record of the driver (#2895), and in [`predict_iov`] on the run's static equivalent an
/// `EVID=2` row at each decision with no data row is too. The two then agree to the bit on
/// every record layout, a held decision included
/// (`the_run_equals_predict_iov_on_the_realized_doses_plus_a_record_per_decision`). Before
/// #2895 they parted wherever a decision had no data record before the next one.
///
/// **An `ADDL`-implied dose takes its governing record's occasion — agreement with
/// [`predict_iov`], not a third divergence** (#2371). Since #1909 an implied dose's
/// covariates and `TIME` are its governing record's (the first record at or after it;
/// [`crate::types::ImpliedDose`]), and so is its κ here: the occasion is resolved at
/// `Subject::dose_occasion_time`, the governing record's engine time, not at the dose's
/// own. `predict_iov` takes that record's `OCC`, so the two agree. Resolving κ at the
/// dose's own time instead gave one snapshot two occasions whenever a decision fell
/// between the dose and its governing record — window `g`'s κ with window `g + 1`'s
/// covariates — 5.6 % off `predict_iov` at the next observation on the regression
/// fixture, and 4.7 % with the covariate held constant, since the κ alone moves the
/// interval the implied dose ends. The parent-row alternative (covariates and `TIME` at
/// the dose's own time, as before #1909) fails that oracle on the κ alone. A dose record
/// is its own governing record, so nothing else moves.
pub fn compute_event_pk_params_iov(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta_baseline: &[f64],
    eta_occ: &[Vec<f64>], decision_times: &[f64],
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
pub const LTBS_FLOOR: f64 = 1e-12;

/// The post-engine tail of the `pk` sink: `[scaling]`, then
/// [`finish_on_observation_scale`]. Every non-IOV prediction sink ends here, so a new
/// tail step reaches all of them (#2129, #2156, #2259 were incomplete copies).
pub fn to_observation_scale(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64], preds: &mut [f64],
) {
    apply_scaling(model, subject, theta, eta, preds);
    finish_on_observation_scale(model, subject, theta, eta, preds);
}

/// The part of the `pk` sink's tail after `[scaling]`: the LTBS log-wrap, then the FREM
/// override. Sinks with a per-occasion/per-window `[scaling]` step (`predict_iov`,
/// `api::adaptive::adaptive_rows_to_model_scale`) call this directly. The FREM override
/// comes last (pseudo-observations are `θ_k + η_m` on the raw scale); it reads only BSV
/// slots of `eta_bsv`.
pub fn finish_on_observation_scale(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta_bsv: &[f64], preds: &mut [f64],
) {
    apply_log_transform(model, preds);
    apply_frem_prediction_override(model, subject, theta, eta_bsv, preds);
}

/// Log-transform predictions in place under log-transform-both-sides (LTBS); a no-op
/// otherwise. Applied at every prediction sink (NONMEM's `Y = LOG(F) + EPS`). The analytic
/// sensitivity paths (`sens::provider`, `sens::ode_provider`) must stay in sync.
#[inline]
pub fn apply_log_transform(model: &CompiledModel, preds: &mut [f64]) {
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
pub fn ltbs_log_g<T: crate::sens::num::PkNum>(p: T) -> T {
    p.guard_floor(LTBS_FLOOR).ln()
}

/// IOV predictions with per-dose occasion accounting (#104): per-event PK parameters carry
/// each event's occasion κ and the event-driven solver runs once. Its end-of-interval
/// convention eliminates a dose's carryover into a later occasion with the *later*
/// occasion's clearance, as NONMEM's continuous integration does.
///
/// `eta_bsv` is the BSV η (length `n_eta`); `kappas[k]` is the κ vector for the k-th
/// occasion group in `iov_occasion_groups` order, including dose-only, pk-only-only and
/// reset-only occasions. A dose, pk-only (`EVID=2`, or `EVID=0` with `MDV=1`) or reset
/// (`EVID=3/4`) row runs under its own occasion; zero κ only for a row with no label
/// (#1876, #1153). Every model has event-driven or ODE support, so the dispatch is total.
/// A Form C output (`y = <expr>`) referencing `KAPPA_*` would see κ = 0, so the parser
/// rejects it for IOV models (#107).
pub fn predict_iov(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta_bsv: &[f64], kappas: &[Vec<f64>],
) -> Vec<f64> {
    predict_iov_with_scratch(model, subject, theta, eta_bsv, kappas, &mut EventPkParams::default())
}

/// IOV predictions with caller-owned per-event parameter storage. Every event
/// is still evaluated at its current theta, eta, kappa, covariates and time;
/// only allocation capacity is reused, never numerical values.
pub fn predict_iov_with_scratch<K: AsRef<[f64]>>(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta_bsv: &[f64], kappas: &[K],
    scratch: &mut EventPkParams,
) -> Vec<f64> {
    predict_iov_with_scratch_and_schedule(model, subject, theta, eta_bsv, kappas, scratch, None)
}

/// [`predict_iov_with_scratch`] with the subject's cached
/// [`EventSchedule`](event_driven::EventSchedule) (#1452 review). A cached schedule is
/// valid only when nothing needed resolving (`Cow::Borrowed`), as on the non-IOV arm.
/// `None` rebuilds it.
#[allow(clippy::too_many_arguments)]
pub fn predict_iov_with_scratch_and_schedule<K: AsRef<[f64]>>(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta_bsv: &[f64], kappas: &[K],
    scratch: &mut EventPkParams, schedule: Option<&event_driven::EventSchedule>,
) -> Vec<f64> {
    let layout = crate::stats::stacked_layout::StackedLayout::new_cached(
        eta_bsv.len(),
        model.n_kappa,
        subject,
    );
    predict_iov_in(model, subject, &layout, theta, eta_bsv, kappas, scratch, schedule)
}

/// [`predict_iov_with_scratch_and_schedule`] over a caller-built occasion → group
/// `layout`, built once per subject by callers predicting it many times (#2037).
#[allow(clippy::too_many_arguments)]
pub fn predict_iov_in<K: AsRef<[f64]>>(
    model: &CompiledModel, subject: &Subject, layout: &crate::stats::stacked_layout::StackedLayout,
    theta: &[f64], eta_bsv: &[f64], kappas: &[K], scratch: &mut EventPkParams,
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
    let model = if walk { model } else { model.effective_for(subject) };
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
                    let view = |idx| event_driven::Indexed { table: dose_params.as_slice(), idx };
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
pub fn superposition_arrival_non_finite(doses: &[DoseEvent], lagtime: f64) -> bool {
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
    pk_model: PkModel, doses: &[DoseEvent], t: f64, pk_params: &PkParams,
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
/// in `(rate, duration)` via `DoseEvent::with_bioavailable_infusion` (#419). Matches
/// NONMEM's `F1` (#327).
pub fn route_f_scale(pk_model: PkModel, infusion: bool, p: &PkParams) -> f64 {
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
pub fn single_dose_concentration(
    pk_model: PkModel, dose: &DoseEvent, tau: f64, p: &PkParams,
) -> f64 {
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
pub fn central_raw(
    pk_model: PkModel, dose: &DoseEvent, tau: f64, p: &PkParams, infusion: bool,
) -> f64 {
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
///   OneCptIv:     `C_central`
///   OneCptOral:   [A_depot, C_central]
///   TwoCptIv:     [C_central, C_periph]
///   TwoCptOral:   [A_depot, C_central, C_periph]
///   ThreeCptIv:   [C_central, C_periph1, C_periph2]
///   ThreeCptOral: [A_depot, C_central, C_periph1, C_periph2]
///
/// Scaling is NOT applied — these are the raw analytical PK states.
pub fn single_dose_states(pk_model: PkModel, dose: &DoseEvent, tau: f64, p: &PkParams) -> Vec<f64> {
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
        PkModel::OneCptTransit => {
            vec![one_cpt_transit_depot(dose, tau, p.n_transit(), p.mtt(), p.f_bio()), central]
        }
        PkModel::TwoCptTransit => {
            let (q, v2, n, mtt, f) = (p.q(), p.v2(), p.n_transit(), p.mtt(), p.f_bio());
            vec![
                two_cpt_transit_depot(dose, tau, cl, v, q, v2, n, mtt, f),
                central,
                two_cpt_transit_peripheral(dose, tau, cl, v, q, v2, n, mtt, f),
            ]
        }
        // [depot = unabsorbed IG mass, central(, peripheral)].
        PkModel::OneCptIg => {
            vec![one_cpt_ig_depot(dose, tau, p.mat(), p.cv2(), p.f_bio()), central]
        }
        PkModel::TwoCptIg => {
            let (q, v2, mat, cv2, f) = (p.q(), p.v2(), p.mat(), p.cv2(), p.f_bio());
            vec![
                two_cpt_ig_depot(dose, tau, cl, v, q, v2, mat, cv2, f),
                central,
                two_cpt_ig_peripheral(dose, tau, cl, v, q, v2, mat, cv2, f),
            ]
        }
        PkModel::TwoCptIv => vec![central, two_cpt_iv_peripheral(dose, tau, cl, v, p.q(), p.v2())],
        PkModel::TwoCptOral if infusion => {
            vec![0.0, central, two_cpt_iv_peripheral(dose, tau, cl, v, p.q(), p.v2())]
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
    pk_model: PkModel, subject: &Subject, pk_params: &PkParams, times: &[f64],
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
    pk_model: PkModel, subject: &Subject, pk_params: &PkParams,
) -> Vec<Vec<f64>> {
    analytical_state_at_times(pk_model, subject, pk_params, &subject.obs_times)
}

/// Compute predictions AND full compartment states for all observations:
/// `(ipred_vec, compartment_states_vec)`. ODE states come from the same integration;
/// analytical states from superposition. Analytical subjects with resets, and IOV models,
/// get empty states (→ NaN in `[derived]`, `W_DERIVED_CMT_RESET_ANALYTICAL`).
pub fn compute_predictions_with_states(
    model: &crate::types::CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64],
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
pub fn compute_predictions_recycle(
    pk_model: PkModel, subject: &Subject, pk_params: &PkParams, mut out: Vec<f64>,
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
pub fn dose_needs_event_walk(pk_model: PkModel, subject: &Subject) -> bool {
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
pub fn subject_feeds_analytical_pk(model: &CompiledModel, subject: &Subject) -> bool {
    #[cfg(feature = "survival")]
    {
        // No non-Gaussian endpoints ⇒ every observation is Gaussian by construction.
        if model.endpoints.is_empty() {
            return true;
        }
        subject.obs_cmts.iter().any(|cmt| !model.endpoints.contains_key(cmt))
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
    ode_spec: &crate::ode::OdeSpec, subject: &Subject, pk_params_flat: &[f64], theta: &[f64],
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
pub fn apply_frem_prediction_override(
    model: &crate::types::CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64],
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
/// prediction kernel under `crate::predict`, `crate::fit` and the simulators.
///
/// # No model/data checks
///
/// This kernel (and its [`compute_predictions_with_tv_into`] /
/// [`compute_predictions_with_tv_into_with_schedule`] variants) runs **none** of the
/// public entry points' checks: a name in
/// [`CompiledModel::referenced_covariates`](crate::types::CompiledModel::referenced_covariates)
/// absent from the subject's covariate data evaluates to `0.0` silently — including
/// `TAD`/`TAFD`/`MACHEPS` read in a readout or `[scaling]` (#1028). Run
/// `crate::check_model_data` first, or call `crate::predict` (`E_MISSING_COVARIATE`).
pub fn compute_predictions_with_tv(
    model: &crate::types::CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64],
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
    model: &crate::types::CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64],
    scratch: &mut EventPkParams,
) -> Vec<f64> {
    compute_predictions_with_tv_into_with_schedule(model, subject, theta, eta, scratch, None)
}

/// Hot-path variant that also accepts a pre-built [`event_driven::EventSchedule`],
/// reused on the TV-cov event-driven paths (saving the merged-event sort and infusion
/// bounds per call). Ignored on the static fast path; `None` builds one on demand. Runs
/// no model/data checks (see [`compute_predictions_with_tv`]).
pub fn compute_predictions_with_tv_into_with_schedule(
    model: &crate::types::CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64],
    scratch: &mut EventPkParams, schedule: Option<&event_driven::EventSchedule>,
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
pub fn compute_predictions_with_tv_recycle_with_schedule(
    model: &crate::types::CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64],
    scratch: &mut EventPkParams, schedule: Option<&event_driven::EventSchedule>,
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
