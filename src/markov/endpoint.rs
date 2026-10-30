//! CTMM endpoint wiring (Phase 5, Track D — #759).
//!
//! Bridges the model layer to the pure numerical [`ctmm_data_term`] kernel in
//! [`crate::markov`]. Its job is everything the leaf deliberately does **not**
//! know about (§8.7 of `plans/tte-survival-markov.md`):
//!
//! - **build the generator** `Q(θ, η, cov)` from the endpoint's
//!   [`GeneratorFn`] (the `[markov_model] transition`
//!   intensities, diagonal filled row-sum-zero), once per subject
//!   (time-homogeneous — a drug-driven `Q(t)` is Phase 6);
//! - **collect the observations** — map each subject
//!   [`ObsRecord::DiscreteState`] row on the endpoint's CMT to a
//!   [`StateObs`], translating its raw DV code to the 0-based generator index via
//!   the endpoint's `state_codes` table; and
//! - **sum the per-subject NLL** across every CTMM endpoint, returning the raw
//!   (positive) `−Σ log P(Δt)[s,s′]`. Callers apply the same `2·` OFV-scale
//!   factor they apply to the TTE / binary terms.
//!
//! The whole module is gated behind `markov` (which implies `survival`).

use crate::markov::{ctmm_data_term, StateObs};
use crate::stats::likelihood::SUBJECT_SENTINEL_NLL;
use crate::types::{CompiledModel, EndpointLikelihood, GeneratorFn, ObsRecord, Subject};
use std::collections::HashMap;

/// Large finite objective returned for a subject whose CTMM data term cannot be
/// evaluated cleanly — mirrors the `survival` module's `1e20` convention (repel
/// the optimizer, never `−inf`/`NaN`/panic mid-fit). Two triggers, **both of
/// which are ruled out before the fit starts** by [`validate_ctmm_states`] and
/// the shared TV-covariate guard, so this is a defensive backstop, not an
/// expected path:
///
/// - a [`crate::markov::MarkovError`] from [`ctmm_data_term`] (structural/data
///   problem — validated pre-fit), and
/// - an observed DV code not present in the endpoint's `state_codes`
///   (validated pre-fit by [`validate_ctmm_states`]).
///
/// [`ctmm_data_term`]'s own parameter-driven degeneracy (an observed transition
/// whose probability underflowed) already returns its `1e20` sentinel *inside*
/// an `Ok`, so that live optimization path flows through the normal sum below.

/// Collect one subject's [`ObsRecord::DiscreteState`] rows on `cmt` into
/// [`StateObs`] in record order, translating each raw DV code to its 0-based
/// generator index through `state_codes`.
///
/// Returns `Err(bad_code)` if a row carries a DV code absent from `state_codes`
/// — a data/model mismatch [`validate_ctmm_states`] rejects before the fit, so
/// the caller treats it as the [`SUBJECT_SENTINEL_NLL`] backstop rather than a
/// per-iteration failure.
fn collect_state_obs(
    cmt: usize,
    state_codes: &[usize],
    records: &[ObsRecord],
) -> Result<Vec<StateObs>, usize> {
    let mut obs = Vec::new();
    for r in records {
        if let ObsRecord::DiscreteState {
            time,
            state,
            cmt: c,
            ..
        } = r
        {
            if *c != cmt {
                continue; // a DiscreteState row for a different endpoint's CMT
            }
            // Map the raw DV code → generator index. `state_codes` is tiny
            // (S states); a linear scan is cheaper than a HashMap here.
            let idx = state_codes.iter().position(|&code| code == *state);
            match idx {
                Some(i) => obs.push(StateObs {
                    time: *time,
                    state: i,
                }),
                None => return Err(*state),
            }
        }
    }
    Ok(obs)
}

/// Negative CTMM log-likelihood for one subject on a single `Ctmm` endpoint:
/// build `Q(θ,η,cov)`, gather its state observations, and score the transition
/// chain with [`ctmm_data_term`]. Returns the raw NLL (positive); the caller
/// applies the OFV-scale factor.
///
/// Evaluated fresh at each `eta` — including the perturbed `eta` of the FOCEI
/// FD-Hessian closure — so the generator's η-curvature is captured without any
/// separate gradient plumbing (the FD path differentiates through
/// `generator_fn`).
#[allow(clippy::too_many_arguments)]
fn ctmm_endpoint_nll(
    cmt: usize,
    n_states: usize,
    state_codes: &[usize],
    generator_fn: &GeneratorFn,
    records: &[ObsRecord],
    covariates: &HashMap<String, f64>,
    theta: &[f64],
    eta: &[f64],
) -> f64 {
    let obs = match collect_state_obs(cmt, state_codes, records) {
        Ok(o) => o,
        Err(_bad_code) => return SUBJECT_SENTINEL_NLL, // validated pre-fit; defensive
    };
    // Fewer than two states carry no transition — no contribution (also handled
    // inside `ctmm_data_term`, but skipping the generator build is cheaper).
    if obs.len() < 2 {
        return 0.0;
    }
    // Time-homogeneous path: no ODE state drives Q, so the state slice is empty.
    // The time-inhomogeneous (drug/PD-driven) path is scored by a separate integrator
    // (Phase 6, #817) that supplies the evolving state per gap — see `ctmm_subject_nll`.
    let q = generator_fn(theta, eta, covariates, &[]);
    debug_assert_eq!(
        q.nrows(),
        n_states,
        "generator_fn must return an n_states×n_states matrix"
    );
    // A transition intensity is an unconstrained user expression, so a covariate /
    // parameter regime (e.g. a linear covariate model, or a diverged η) can drive an
    // off-diagonal rate negative. A negative off-diagonal makes `q` a non-generator:
    // `expm(Q·Δt)` is no longer stochastic and an observed transition can score P > 1,
    // which `ctmm_data_term` clamps to a 0 penalty — the *minimum*, silently rewarding
    // the optimizer toward the unphysical region. Treat it as the degenerate case and
    // repel, exactly as the kernel does for an underflowed transition. The tiny
    // tolerance absorbs floating round-off on a rate that is physically zero.
    if super::has_negative_offdiagonal(&q) {
        return SUBJECT_SENTINEL_NLL;
    }
    match ctmm_data_term(&q, &obs) {
        Ok(nll) => nll,
        // Structural/data MarkovError — ruled out by validate_ctmm_states + the
        // datareader's sorted/finite-time guarantees before the fit. Repel
        // rather than panic in the hot loop.
        Err(_) => SUBJECT_SENTINEL_NLL,
    }
}

/// Exact `∂/∂η` of [`ctmm_subject_nll`] for time-homogeneous endpoints, at the same `1×`
/// scale as the value — the inner EBE gradient's CTMM block: the generator program over a
/// first-order η-dual (θ constant) through the scaling-and-squaring `expm` on that dual
/// (`CtmmTerm`). `None` — the caller falls back to FD for this point — when *any* CTMM
/// endpoint on the subject cannot be served exactly (no program, or a drug-driven generator
/// outside the dual state solve's scope) or in the degenerate regime (a negative off-diagonal rate, the `expm` /
/// underflow guards), where the value is a flat repellent with no meaningful derivative.
pub fn ctmm_subject_eta_grad(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<(f64, Vec<f64>)> {
    if subject.obs_records.is_empty() {
        return Some((0.0, vec![0.0; model.n_eta]));
    }
    let term = CtmmTerm { model, subject };
    let f = crate::survival::FixedTheta { theta, term: &term };
    if eta.is_empty() {
        use crate::survival::EtaScalar;
        return Some((f.eval::<f64>(&[])?, Vec::new()));
    }
    // η-only: the generator program reads θ as constants, so the dual carries one lane per
    // random effect, and `expm` on it costs (1 + n_η) exponentials per gap — under the
    // 2S×2S Fréchet solve a first-order (θ, η) jet needed — with no width cap.
    crate::survival::eta_gradient_by_bucket(eta, &f)
}

/// Every CTMM endpoint's data term on the subject as a function of `(θ, η)`, on any number
/// type: `Q` from the generator program, each gap's `expm(Q·Δt)` on that type (or, for a
/// drug-driven generator, the occupancy ODE, [`inhomogeneous_nll_g`]), `−Σ log P[u,v]` with
/// the value path's clamp and guards (declines). `None` for an endpoint without a program.
pub(crate) struct CtmmTerm<'a> {
    pub(crate) model: &'a CompiledModel,
    pub(crate) subject: &'a Subject,
}
impl crate::survival::SubjectTerm for CtmmTerm<'_> {
    fn eval<T: crate::sens::num::vm::VmNum>(&self, th: &[T], et: &[T]) -> Option<T> {
        let mut nll = T::from_f64(0.0);
        for (cmt, endpoint) in &self.model.endpoints {
            let EndpointLikelihood::Ctmm {
                n_states,
                state_codes,
                generator_program,
                ..
            } = endpoint
            else {
                continue;
            };
            let prog = generator_program.as_ref()?;
            let obs = collect_state_obs(*cmt, state_codes, &self.subject.obs_records).ok()?;
            if obs.len() < 2 {
                continue;
            }
            if prog.reads_states() {
                nll = nll
                    + inhomogeneous_nll_g(self.model, self.subject, prog, *n_states, &obs, th, et)?;
                continue;
            }
            let s = *n_states;
            let q = prog.eval_generator_g(th, et, &self.subject.covariates)?;
            // The value path's degeneracy guards, as declines: a negative off-diagonal
            // (a non-generator), a non-finite or enormous `expm` argument, a
            // non-positive probability — each a flat sentinel there, with no curvature.
            for j in 0..s {
                for k in 0..s {
                    if j != k && q[j * s + k].val() < -1e-12 {
                        return None;
                    }
                }
            }
            let mut expm = crate::ode::linear::Expm::<T>::new(s);
            let mut arg = vec![T::from_f64(0.0); s * s];
            for pair in obs.windows(2) {
                let (a, b) = (pair[0], pair[1]);
                let dt = b.time - a.time;
                if dt < 0.0 || (dt == 0.0 && a.state != b.state) {
                    return None;
                }
                if dt == 0.0 {
                    continue;
                }
                for (o, &x) in arg.iter_mut().zip(&q) {
                    *o = x * T::from_f64(dt);
                    if !(o.val().is_finite() && o.val().abs() <= crate::markov::MAX_EXP_ARG_ABS) {
                        return None;
                    }
                }
                let slot = expm.exp(&arg)?;
                let p = expm.get(slot)[a.state * s + b.state];
                if !(p.val().is_finite() && p.val() > 0.0) {
                    return None;
                }
                // The value path's clamp `p.min(1)`: past 1 the term is constant.
                let p = if p.val() > 1.0 { T::from_f64(1.0) } else { p };
                nll = nll - p.ln();
            }
        }
        Some(nll)
    }
}

/// Exact `∂²/∂η²` of [`ctmm_subject_nll`] for time-homogeneous endpoints: `Q` assembled
/// from the generator program over a second-order forward dual in η, each gap's
/// `P(Δt) = expm(Q·Δt)` by the scaling-and-squaring exponential on that dual type
/// (`crate::ode::linear::Expm`, one instance per endpoint so its Padé reuse spans the
/// gaps), and `−Σ log P[u,v]` — the second derivative of `expm` without a Fréchet solve
/// per direction pair. The FD stencil it replaces cost `2n(n+1)` likelihoods plus `2n` for
/// its steps, each re-exponentiating every gap. `None` — the caller keeps the stencil —
/// on the same declines as [`ctmm_subject_eta_grad`] (an inhomogeneous generator or one
/// without a program, the degenerate guards) or past the dual bucket.
pub(crate) fn ctmm_subject_eta_hessian(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<nalgebra::DMatrix<f64>> {
    crate::survival::eta_hessian_by_bucket(
        eta,
        &crate::survival::FixedTheta {
            theta,
            term: &CtmmTerm { model, subject },
        },
    )
}

/// Sum the CTMM-endpoint NLL over every `Ctmm` CMT for one subject, mirroring the
/// per-endpoint TTE / binary dispatch. Returns `0.0` when the model has no CTMM
/// endpoint or the subject has no discrete-state records for it.
///
/// This doubles as the FOCEI FD-Hessian closure at the interaction site:
/// evaluated at a perturbed η it rebuilds `Q(η)` and re-scores every transition,
/// so `data_term_hessian_fd` picks up the CTMM curvature w.r.t. η.
pub fn ctmm_subject_nll(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> f64 {
    if subject.obs_records.is_empty() {
        return 0.0;
    }
    let mut nll = 0.0;
    for (cmt, endpoint) in &model.endpoints {
        if let EndpointLikelihood::Ctmm {
            n_states,
            state_codes,
            generator_fn,
            generator_states,
            ..
        } = endpoint
        {
            nll += if generator_states.is_empty() {
                // Time-homogeneous (Phase 5): one constant Q per subject, P = expm(Q·Δt).
                ctmm_endpoint_nll(
                    *cmt,
                    *n_states,
                    state_codes,
                    generator_fn,
                    &subject.obs_records,
                    &subject.covariates,
                    theta,
                    eta,
                )
            } else {
                // Time-inhomogeneous (Phase 6, #817): Q depends on the evolving model
                // state, so each interval transition is integrated (occupancy ODE).
                ctmm_endpoint_nll_inhomogeneous(
                    model,
                    subject,
                    *cmt,
                    *n_states,
                    state_codes,
                    generator_fn,
                    theta,
                    eta,
                )
            };
        }
    }
    nll
}

/// Number of interior sub-nodes per observation gap at which the model state is
/// sampled to drive `Q(state(t))` across the interval. The occupancy ODE is then
/// integrated (adaptively) with the state linearly interpolated between sub-nodes;
/// concentration/PD states are smooth within a gap, so a modest grid is accurate,
/// and it is refined for free by the adaptive occupancy stepper on top. The
/// degenerate `SLOPE = 0` case (state-independent `Q`) is exact regardless.
#[cfg(feature = "markov")]
const INHOMOGENEOUS_SUBNODES_PER_GAP: usize = 16;

/// Negative CTMM log-likelihood for one subject on a **time-inhomogeneous** (drug/
/// PD-driven `Q(t)`) endpoint (Phase 6, #817).
///
/// The generator depends on the model's ODE state (a concentration `central/V`, a PD
/// response compartment, …), so `P(Δt_m)` has no closed form. The subject's ODE
/// system is solved once for `(θ, η)`; per observation gap the state is sampled on a
/// sub-grid and the occupancy ODE `dP/dτ = P·Q(state(t_m+τ))`, `P(0)=I` is integrated
/// with [`crate::markov::ctmm_inhomogeneous_transition`], the state linearly
/// interpolated between sub-nodes. The likelihood is the same
/// `−Σ_m log P(Δt_m)[s_m, s_{m+1}]`. Evaluated fresh at each `η` (including the
/// perturbed `η` of the FD paths), so the full `η`-dependence — through both the
/// generator and the PK/PD state trajectory — is captured without new gradient
/// plumbing.
#[cfg(feature = "markov")]
#[allow(clippy::too_many_arguments)]
fn ctmm_endpoint_nll_inhomogeneous(
    model: &CompiledModel,
    subject: &Subject,
    cmt: usize,
    n_states: usize,
    state_codes: &[usize],
    generator_fn: &GeneratorFn,
    theta: &[f64],
    eta: &[f64],
) -> f64 {
    let obs = match collect_state_obs(cmt, state_codes, &subject.obs_records) {
        Ok(o) => o,
        Err(_bad_code) => return SUBJECT_SENTINEL_NLL, // validated pre-fit; defensive
    };
    if obs.len() < 2 {
        return 0.0;
    }
    // An inhomogeneous intensity references an ODE state, so the model must be an ODE
    // model (guaranteed at parse: `generator_states` indices point into `ode_spec`).
    let Some(ode) = model.ode_spec.as_ref() else {
        return SUBJECT_SENTINEL_NLL;
    };

    // PK/PD disposition snapshot at (θ, η) — the ODE solve integrates under it, and
    // `init()` reads it (baseline covariate snapshot). A time-varying covariate on the
    // generator is rejected at fit setup, and so is one on a parameter this solve feeds
    // into the generator's states (CL, V, …; #1854) — both in
    // `api::check_survival_tv_covariates`. Every dose-record quantity — absorption kernel,
    // pathway fraction, route lag, zero-order `dur`, compartment lag, `F`, `D{n}`/`R{n}`,
    // the SS run-in — is read at its own dose record instead (#1575), always, as the
    // ODE hazard reads it (`survival::ode_cumhaz_hazard`).
    let pk = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);
    let mut pk_at_dose = Vec::new();
    crate::pk::compute_dose_pk_params_into(model, subject, theta, eta, &mut pk_at_dose);

    // One sub-grid per gap: `n_sub + 1` uniform nodes spanning [t_m, t_{m+1}] (gap m
    // occupies a contiguous block, so gaps sharing a boundary time keep independent
    // blocks). Solve the ODE state at every node in a single pass.
    let n_sub = INHOMOGENEOUS_SUBNODES_PER_GAP;
    let n_gaps = obs.len() - 1;
    let mut grid: Vec<f64> = Vec::with_capacity(n_gaps * (n_sub + 1));
    for w in obs.windows(2) {
        let (t0, t1) = (w[0].time, w[1].time);
        for i in 0..=n_sub {
            grid.push(t0 + (t1 - t0) * (i as f64) / (n_sub as f64));
        }
    }
    let states = crate::ode::predictions::ode_dense_solve_states_reading(
        ode,
        &pk.values,
        crate::ode::predictions::DoseReads::PerDose(&pk_at_dose),
        subject,
        &grid,
    );
    debug_assert_eq!(states.len(), grid.len());

    let cov = &subject.covariates;
    let mut nll = 0.0;
    for (m, w) in obs.windows(2).enumerate() {
        let (t0, t1) = (w[0].time, w[1].time);
        let dt = t1 - t0;
        let base = m * (n_sub + 1); // this gap's block start in `grid`/`states`

        // Out-of-order records (t_{m+1} < t_m): the homogeneous `ctmm_data_term`
        // rejects this with `MarkovError::TimeDecreased`; the inhomogeneous path would
        // instead return the identity `P(0)` and silently score a same-state pair as
        // `log 1 = 0`, understating the objective. Repel to match. (`dt == 0` with an
        // identical state is the legitimate zero-length gap and falls through.)
        if dt < 0.0 {
            return SUBJECT_SENTINEL_NLL;
        }

        // State at elapsed time τ ∈ [0, dt] within the gap, linearly interpolated
        // between the two bracketing sub-nodes, written into a reused scratch buffer so
        // the occupancy-ODE RHS (called at every RK45 stage of every gap) does not
        // allocate a fresh `Vec` per evaluation. `Q` off-diagonals are guarded inside
        // the generator's downstream use, but a negative off-diagonal from an
        // unconstrained intensity must still repel — checked at the sub-nodes below.
        let n_ode = states[base].len();
        let interp_scratch = std::cell::RefCell::new(vec![0.0_f64; n_ode]);
        let interp_into = |tau: f64, out: &mut [f64]| {
            if dt <= 0.0 {
                out.copy_from_slice(&states[base]);
                return;
            }
            let frac = (tau / dt).clamp(0.0, 1.0) * (n_sub as f64);
            let lo = (frac.floor() as usize).min(n_sub);
            let hi = (lo + 1).min(n_sub);
            let w_hi = frac - lo as f64;
            let (a, b) = (&states[base + lo], &states[base + hi]);
            for (o, (x, y)) in out.iter_mut().zip(a.iter().zip(b.iter())) {
                *o = x + (y - x) * w_hi;
            }
        };

        // Guard the generator at each sub-node — the inhomogeneous analogue of the
        // homogeneous `ctmm_data_term` up-front checks; `Q` is smooth in the state, so
        // the dense sub-nodes catch these (a driver that is non-monotone *strictly
        // between* two same-sign sub-nodes is the residual gap, not covered here):
        //   • a non-finite ODE state — e.g. a grid node the solve never reached and
        //     left as `NaN` — would poison the occupancy integration silently;
        //   • a non-finite or enormous rate (`|q·dt| > MAX_EXP_ARG_ABS`), a diverged
        //     intensity that would hang/overflow the stiff occupancy solve rather than
        //     return a usable `P` (mirrors the homogeneous `expm`-argument guard); and
        //   • a negative off-diagonal, an unconstrained intensity driven negative.
        // All repel the optimizer (sentinel) rather than scoring a bogus likelihood.
        for i in 0..=n_sub {
            let st = &states[base + i];
            if st.iter().any(|x| !x.is_finite()) {
                return SUBJECT_SENTINEL_NLL;
            }
            let q = generator_fn(theta, eta, cov, st);
            for j in 0..q.nrows() {
                for k in 0..q.ncols() {
                    let qjk = q[(j, k)];
                    if !qjk.is_finite() || qjk.abs() * dt > super::MAX_EXP_ARG_ABS {
                        return SUBJECT_SENTINEL_NLL;
                    }
                    if j != k && qjk < -1e-12 {
                        return SUBJECT_SENTINEL_NLL;
                    }
                }
            }
        }

        let p = crate::markov::ctmm_inhomogeneous_transition_with_opts(
            |tau| {
                let mut st = interp_scratch.borrow_mut();
                interp_into(tau, &mut st);
                generator_fn(theta, eta, cov, &st)
            },
            dt,
            n_states,
            &ode.effective_solver_opts(),
        );
        let prob = p[(obs[m].state, obs[m + 1].state)];
        // Underflowed / non-positive probability for an observed transition → repel.
        if !prob.is_finite() || prob <= 0.0 {
            return SUBJECT_SENTINEL_NLL;
        }
        nll -= prob.min(1.0).ln();
    }
    nll
}

/// A time-**inhomogeneous** endpoint's data term on any number type: the value path
/// ([`ctmm_endpoint_nll_inhomogeneous`]) evaluated generically. The ODE state at every
/// gap's sub-nodes from the dual state solve
/// ([`ode_states_at_g`](crate::sens::ode_provider::adjoint::ode_states_at_g)), the generator
/// program at the linearly interpolated state, and the occupancy ODE `dP/dτ = P·Q(τ)` on the
/// generic stepper ([`crate::ode::solve_ode_g`]) — so the derivative is that of the same
/// discretisation, with its steps chosen by the values. `None` on the value path's
/// degeneracy guards (the sentinel there), outside the dual state solve's scope, or on a
/// non-positive probability.
fn inhomogeneous_nll_g<T: crate::sens::num::vm::VmNum>(
    model: &CompiledModel,
    subject: &Subject,
    prog: &crate::parser::model_parser::CtmmGeneratorProgram,
    s: usize,
    obs: &[StateObs],
    theta: &[T],
    eta: &[T],
) -> Option<T> {
    let ode = model.ode_spec.as_ref()?;
    let n_sub = INHOMOGENEOUS_SUBNODES_PER_GAP;
    // The value path's grid, deduplicated (adjacent gaps share their boundary time).
    let mut grid: Vec<f64> = Vec::with_capacity((obs.len() - 1) * (n_sub + 1));
    for w in obs.windows(2) {
        let (t0, t1) = (w[0].time, w[1].time);
        if t1 < t0 {
            return None;
        }
        for i in 0..=n_sub {
            grid.push(t0 + (t1 - t0) * (i as f64) / (n_sub as f64));
        }
    }
    let mut times = grid.clone();
    times.dedup_by(|a, b| a.to_bits() == b.to_bits());
    let solved =
        crate::sens::ode_provider::adjoint::ode_states_at_g(model, subject, theta, eta, &times)?;
    let at = |t: f64| -> Option<&Vec<T>> {
        let i = times.binary_search_by(|x| x.total_cmp(&t)).ok()?;
        solved.states.get(i)
    };
    let (vars0, cov_vec) = prog.vars_g(theta, eta, &subject.covariates)?;
    // An uncertified closure, integrated as the value path's `solve_ode` integrates it.
    let opts = ode.effective_solver_opts().for_integration();
    let mut nll = T::from_f64(0.0);
    for (m, w) in obs.windows(2).enumerate() {
        let dt = w[1].time - w[0].time;
        if dt == 0.0 {
            if w[0].state != w[1].state {
                return None;
            }
            continue;
        }
        let nodes: Vec<&Vec<T>> = grid[m * (n_sub + 1)..(m + 1) * (n_sub + 1)]
            .iter()
            .map(|&t| at(t))
            .collect::<Option<_>>()?;
        // The value path's guards at the sub-nodes, as declines.
        let mut vars = vars0.clone();
        for st in &nodes {
            if st.iter().any(|x| !x.val().is_finite()) {
                return None;
            }
            let q = prog.generator_at_g(theta, eta, &cov_vec, &mut vars, st);
            for j in 0..s {
                for k in 0..s {
                    let v = q[j * s + k].val();
                    if !v.is_finite() || v.abs() * dt > crate::markov::MAX_EXP_ARG_ABS {
                        return None;
                    }
                    if j != k && v < -1e-12 {
                        return None;
                    }
                }
            }
        }
        let n_ode = nodes[0].len();
        let work = std::cell::RefCell::new((vars, vec![T::from_f64(0.0); n_ode]));
        let rhs = |u: &[T], _: &[T], tau: f64, du: &mut [T]| {
            let mut w = work.borrow_mut();
            let (vars, st) = &mut *w;
            // The state at `tau`, interpolated between the bracketing sub-nodes as the
            // value path interpolates it.
            let frac = (tau / dt).clamp(0.0, 1.0) * (n_sub as f64);
            let lo = (frac.floor() as usize).min(n_sub);
            let hi = (lo + 1).min(n_sub);
            let w_hi = T::from_f64(frac - lo as f64);
            for (o, (x, y)) in st.iter_mut().zip(nodes[lo].iter().zip(nodes[hi].iter())) {
                *o = *x + (*y - *x) * w_hi;
            }
            let q = prog.generator_at_g(theta, eta, &cov_vec, vars, st);
            // Forward Kolmogorov, `dP/dτ = P·Q`, row-major.
            for i in 0..s {
                for j in 0..s {
                    let mut acc = T::from_f64(0.0);
                    for k in 0..s {
                        acc = acc + u[i * s + k] * q[k * s + j];
                    }
                    du[i * s + j] = acc;
                }
            }
        };
        let mut u0 = vec![T::from_f64(0.0); s * s];
        for i in 0..s {
            u0[i * s + i] = T::from_f64(1.0);
        }
        let sol = crate::ode::solve_ode_g(&rhs, &u0, (0.0, dt), &[], &[dt], &opts);
        let p = sol.last()?.u[w[0].state * s + w[1].state];
        if !(p.val().is_finite() && p.val() > 0.0) {
            return None;
        }
        let p = if p.val() > 1.0 { T::from_f64(1.0) } else { p };
        nll = nll - p.ln();
    }
    Some(nll)
}

/// Fail-loud check that every CTMM record's observed DV code is one of the
/// endpoint's declared `state_codes`. Run once at fit setup: the datareader
/// accepts any non-negative integer DV on a discrete CMT (it cannot know which
/// codes a given endpoint declared), so the `Ctmm` endpoint must reject an
/// undeclared code itself rather than let [`ctmm_subject_nll`] silently fall back
/// to the sentinel. `records` may be a subject's full `obs_records`; only
/// `DiscreteState` rows on `cmt` are checked.
pub fn validate_ctmm_states(
    cmt: usize,
    state_codes: &[usize],
    records: &[ObsRecord],
) -> Result<(), String> {
    for r in records {
        if let ObsRecord::DiscreteState { state, cmt: c, .. } = r {
            if *c == cmt && !state_codes.contains(state) {
                let mut declared = state_codes.to_vec();
                declared.sort_unstable();
                return Err(format!(
                    "[markov_model] cmt = {cmt}: observed DV {state} is not one of the declared \
                     states {declared:?}. Every DV value on a CTMM CMT must match a `states` code."
                ));
            }
        }
    }
    Ok(())
}

/// Fail-loud check that a subject's CTMM observations on `cmt` are **non-decreasing
/// in time**, in record order. Run once at fit setup.
///
/// The datareader sorts a subject's *doses* by time but leaves its observation rows
/// in file order (the NONMEM convention is time-ordered input, but nothing enforces
/// it). An out-of-order pair would make [`ctmm_data_term`] return
/// [`MarkovError::TimeDecreased`](crate::markov::MarkovError::TimeDecreased), which
/// `ctmm_endpoint_nll` maps to the `SUBJECT_SENTINEL_NLL` backstop — silently
/// collapsing that subject's entire likelihood to `1e20` and biasing the population
/// fit with no diagnostic. Rejecting up front converts that silent corruption into a
/// clear error, mirroring [`validate_ctmm_states`]. Only `DiscreteState` rows on
/// `cmt` are inspected.
pub fn validate_ctmm_times(cmt: usize, records: &[ObsRecord]) -> Result<(), String> {
    let mut prev: Option<f64> = None;
    for r in records {
        if let ObsRecord::DiscreteState { time, cmt: c, .. } = r {
            if *c != cmt {
                continue;
            }
            if let Some(p) = prev {
                if *time < p {
                    return Err(format!(
                        "[markov_model] cmt = {cmt}: observation times must be non-decreasing \
                         within a subject, but {p} is followed by {time}. Sort each subject's \
                         CTMM rows by TIME (the datareader does not reorder observations)."
                    ));
                }
            }
            prev = Some(*time);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::GeneratorFn;
    use nalgebra::DMatrix;

    /// A constant 2-state generator `Q = [[-a, a], [b, -b]]`, ignoring θ/η/cov/state.
    fn const_gen(a: f64, b: f64) -> GeneratorFn {
        Box::new(
            move |_t: &[f64], _e: &[f64], _c: &HashMap<String, f64>, _s: &[f64]| {
                DMatrix::from_row_slice(2, 2, &[-a, a, b, -b])
            },
        )
    }

    fn disc(time: f64, state: usize, cmt: usize) -> ObsRecord {
        ObsRecord::DiscreteState {
            time,
            raw_time: time,
            state,
            cmt,
        }
    }

    /// NLL over a record slice with no covariates and no θ/η.
    fn nll(state_codes: &[usize], gen: &GeneratorFn, recs: &[ObsRecord]) -> f64 {
        ctmm_endpoint_nll(5, 2, state_codes, gen, recs, &HashMap::new(), &[], &[])
    }

    /// Closed-form 2-state occupancy: P(t)[0,0] for Q=[[-a,a],[b,-b]] is
    /// (b + a·e^{-(a+b)t})/(a+b). The endpoint NLL of a single 0→0 gap must equal
    /// −log of that.
    #[test]
    fn ctmm_endpoint_single_gap_matches_closed_form() {
        let (a, b, dt) = (0.7_f64, 0.3, 2.0);
        let lam = a + b;
        let p00 = (b + a * (-lam * dt).exp()) / lam;
        let got = nll(
            &[0, 1],
            &const_gen(a, b),
            &[disc(0.0, 0, 5), disc(dt, 0, 5)],
        );
        assert!((got - (-p00.ln())).abs() < 1e-12, "got {got}");
    }

    /// DV codes need not be 0-based/contiguous: states coded 1/2 map to generator
    /// indices 0/1 through `state_codes`, giving the identical likelihood.
    #[test]
    fn ctmm_endpoint_maps_noncontiguous_codes() {
        let (a, b, dt) = (0.5_f64, 0.5, 1.0);
        let lam = a + b;
        // 1 → 2 transition ⇒ generator index 0 → 1 ⇒ P[0,1] = a(1-e)/lam.
        let p01 = a * (1.0 - (-lam * dt).exp()) / lam;
        let got = nll(
            &[1, 2],
            &const_gen(a, b),
            &[disc(0.0, 1, 5), disc(dt, 2, 5)],
        );
        assert!((got - (-p01.ln())).abs() < 1e-12, "got {got}");
    }

    // ---- analytic η-gradient: dual generator + Van Loan chain (#759) ------------

    /// The mixed-effects CTMM of `tests/markov_smoke.rs`: a per-subject random effect on
    /// the awake→asleep intensity, no `[structural_model]` (an endpoint-only fit).
    const MIXED_CTMM: &str = r"
[parameters]
  theta LQ01(-0.7, -6.0, 3.0)
  theta LQ10(-1.2, -6.0, 3.0)
  omega ETA_Q ~ 0.1

[markov_model]
  type   = ctmm
  cmt    = 5
  states = [awake=0, asleep=1]
  transition awake  -> asleep = exp(LQ01 + ETA_Q)
  transition asleep -> awake  = exp(LQ10)
";

    fn ctmm_subject(states: &[(f64, usize)]) -> Subject {
        Subject {
            id: "1".into(),
            obs_records: states
                .iter()
                .map(|&(t, s)| disc(t, s, 5))
                .collect::<Vec<_>>(),
            ..Default::default()
        }
    }

    /// The load-bearing end-to-end check: the exact η-gradient — `∂Q/∂η` from replaying the
    /// parsed intensities over `Dual1`, chained through the adjoint Van Loan derivative of
    /// `expm` — must equal a central finite difference of the *same* subject NLL the fit
    /// minimizes. This is what pins the whole chain (parser → program → generator jets →
    /// Fréchet) to the function it claims to differentiate.
    #[test]
    fn ctmm_subject_eta_grad_matches_fd() {
        let model = crate::parser::model_parser::parse_model_string(MIXED_CTMM).expect("parse");
        assert_eq!(model.n_eta, 1);
        let subject = ctmm_subject(&[(0.0, 0), (1.0, 0), (2.3, 1), (3.1, 1), (5.0, 0)]);
        let theta = [-0.7_f64, -1.2];

        // Away from η = 0, so a sign error or a dropped term cannot hide.
        for &eta0 in &[-0.45_f64, 0.0, 0.6] {
            let eta = [eta0];
            let (v, g) = ctmm_subject_eta_grad(&model, &subject, &theta, &eta)
                .expect("homogeneous CTMM with a dual-evaluable program is analytic");
            // The gradient path's value must not drift from the value path's.
            let v_ref = ctmm_subject_nll(&model, &subject, &theta, &eta);
            assert!((v - v_ref).abs() < 1e-12, "value drift: {v} vs {v_ref}");

            let h = 1e-6;
            let fd = (ctmm_subject_nll(&model, &subject, &theta, &[eta0 + h])
                - ctmm_subject_nll(&model, &subject, &theta, &[eta0 - h]))
                / (2.0 * h);
            assert!(
                (g[0] - fd).abs() < 1e-6,
                "η = {eta0}: analytic {}, FD {fd}",
                g[0]
            );
            // A real gradient, not an accidental zero.
            assert!(g[0].abs() > 1e-3, "η = {eta0}: gradient suspiciously flat");
        }
    }

    /// A time-**inhomogeneous** (drug-driven) generator is differentiated exactly: the
    /// occupancy-ODE likelihood on a dual number type, the intensity reading the dual state
    /// solve. Two doses, the second on residual drug, so the state is live in every gap.
    /// The value matches the f64 path's; the η-gradient, the θ-score and the η-Hessian match
    /// Richardson differences (of the value path, and of the exact gradient). A state slot
    /// left unfilled (the intensity seeing `central = 0`) zeroes `∂/∂ETA_CL` and `∂/∂TVCL`
    /// and fails the first two.
    #[test]
    fn a_drug_driven_ctmm_is_differentiated_exactly() {
        use crate::types::DoseEvent;
        let src = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta LQ01(-0.7, -6.0, 3.0)
  theta LQ10(-1.2, -6.0, 3.0)
  theta SLOPE(0.1, -5.0, 5.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -CL/V * central

[error_model]
  DV ~ proportional(PROP)

[markov_model]
  type   = ctmm
  cmt    = 5
  states = [s0=0, s1=1]
  transition s0 -> s1 = exp(LQ01 + SLOPE * (central / V))
  transition s1 -> s0 = exp(LQ10)
";
        let mut model = crate::parser::model_parser::parse_model_string(src).expect("parse");
        if let Some(ode) = model.ode_spec.as_mut() {
            ode.solver_opts.reltol = 1e-12;
            ode.solver_opts.abstol = 1e-14;
        }
        let mut subject =
            ctmm_subject(&[(0.5, 0), (1.5, 1), (3.0, 1), (7.0, 0), (9.0, 1), (12.0, 0)]);
        subject.doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(6.0, 100.0, 1, 0.0, false, 0.0),
        ];
        let theta = model.default_params.theta.clone();
        let rich = |f: &dyn Fn(f64) -> f64, x: f64| {
            let d = |h: f64| (f(x + h) - f(x - h)) / (2.0 * h);
            let h = 1e-3 * (1.0 + x.abs());
            (4.0 * d(h / 2.0) - d(h)) / 3.0
        };
        let (mut worst_v, mut worst_g, mut worst_h) = (0.0f64, 0.0f64, 0.0f64);
        for eta0 in [-0.3, 0.25] {
            let (v, g) = ctmm_subject_eta_grad(&model, &subject, &theta, &[eta0])
                .expect("a drug-driven CTMM is served exactly");
            let v_ref = ctmm_subject_nll(&model, &subject, &theta, &[eta0]);
            assert!(v.is_finite() && v_ref.is_finite() && v_ref < 1e10);
            worst_v = worst_v.max((v - v_ref).abs() / (1.0 + v_ref.abs()));
            let fd = rich(&|e| ctmm_subject_nll(&model, &subject, &theta, &[e]), eta0);
            assert!(g[0].abs() > 1e-3, "η = {eta0}: ∂/∂ETA_CL is flat: {}", g[0]);
            worst_g = worst_g.max((g[0] - fd).abs() / (1.0 + fd.abs()));
            let h = ctmm_subject_eta_hessian(&model, &subject, &theta, &[eta0]).unwrap();
            let hd = rich(
                &|e| {
                    ctmm_subject_eta_grad(&model, &subject, &theta, &[e])
                        .unwrap()
                        .1[0]
                },
                eta0,
            );
            worst_h = worst_h.max((h[(0, 0)] - hd).abs() / (1.0 + hd.abs()));
            let (_, gt) = crate::sens::ode_provider::adjoint::non_gaussian_theta_score(
                &model,
                &subject,
                &theta,
                &[eta0],
            )
            .unwrap();
            for i in 0..theta.len() {
                let fd = rich(
                    &|t| {
                        let mut th = theta.clone();
                        th[i] = t;
                        ctmm_subject_nll(&model, &subject, &th, &[eta0])
                    },
                    theta[i],
                );
                worst_g = worst_g.max((gt[i] - fd).abs() / (1.0 + fd.abs()));
            }
            assert!(gt[0].abs() > 1e-3, "∂/∂TVCL is flat: {}", gt[0]);
        }
        eprintln!(
            "drug-driven CTMM: value {worst_v:.2e}, gradient {worst_g:.2e}, Hessian {worst_h:.2e}"
        );
        // Measured: the value bit-identical to the f64 path's (the same solves on the same
        // steps), the gradients 5.8e-7 and the Hessian 1.9e-7 off the Richardson differences —
        // the differences' own floor on an adaptively stepped likelihood. An unfilled state
        // slot or a dropped gap misses by O(1).
        assert!(worst_v < 1e-12, "value {worst_v:.2e}");
        assert!(worst_g < 3e-6, "gradient {worst_g:.2e}");
        assert!(worst_h < 1e-6, "Hessian {worst_h:.2e}");
    }

    /// #1187 at this objective. The inhomogeneous CTMM path reads its drug concentration
    /// from [`crate::ode::ode_dense_solve_states`], which resolved infusions through the
    /// unguarded `gated_infusions` — so for a `RATE>0` dose into a built-in absorption
    /// compartment the generator saw roughly twice the exposure and the **fitted** NLL was
    /// wrong. Nothing else covers this surface.
    ///
    /// Oracle is an explicit sub-dose train: N boluses spanning the same window. That is
    /// discriminating here precisely because a *bolus* into an input-rate compartment never
    /// reaches `gated_infusions` at all — only real infusions enter the active list — so the
    /// train is immune to the defect while the infusion is not.
    #[test]
    fn ctmm_nll_for_infusion_into_absorption_matches_a_subdose_train() {
        use crate::types::DoseEvent;

        // `first_order(ka=KA)` in central, so CMT 1 is the input-rate compartment; the
        // transition intensity reads `central / V`, making the generator drug-driven.
        let src = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(0.6, 0.01, 50.0)
  theta LQ01(-0.7, -6.0, 3.0)
  theta LQ10(-1.2, -6.0, 3.0)
  theta SLOPE(0.4, -5.0, 5.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = first_order(ka=KA) - CL/V * central

[error_model]
  DV ~ proportional(PROP)

[markov_model]
  type   = ctmm
  cmt    = 5
  states = [s0=0, s1=1]
  transition s0 -> s1 = exp(LQ01 + SLOPE * (central / V))
  transition s1 -> s0 = exp(LQ10)
";
        let model = crate::parser::model_parser::parse_model_string(src).expect("parse");
        let theta = &model.default_params.theta;
        let eta = [0.0_f64];

        let obs = [(0.0, 0), (2.0, 1), (5.0, 0), (9.0, 1)];
        let (amt, t_inf) = (100.0_f64, 3.0_f64);

        let mut infused = ctmm_subject(&obs);
        infused.doses = vec![DoseEvent::new(0.0, amt, 1, amt / t_inf, false, 0.0)];
        assert!(
            infused.doses[0].is_infusion(),
            "fixture must carry a real infusion or it tests nothing"
        );

        let n = 300usize;
        let mut train = ctmm_subject(&obs);
        train.doses = (0..n)
            .map(|k| {
                let tk = (k as f64 + 0.5) * t_inf / n as f64;
                DoseEvent::new(tk, amt / n as f64, 1, 0.0, false, 0.0)
            })
            .collect();

        let nll_infused = ctmm_subject_nll(&model, &infused, theta, &eta);
        let nll_train = ctmm_subject_nll(&model, &train, theta, &eta);

        // Non-degeneracy: the drug must actually move this objective, or the comparison
        // would hold for a generator that ignored `central` entirely.
        let drug_free = ctmm_subject_nll(&model, &ctmm_subject(&obs), theta, &eta);
        assert!(
            nll_train.is_finite() && (nll_train - drug_free).abs() > 1e-2,
            "CTMM NLL {nll_train} sits at its dose-free value {drug_free} — the concentration \
             is not reaching the generator, so the comparison would be vacuous"
        );

        let rel = (nll_infused - nll_train).abs() / nll_train.abs().max(1.0);
        assert!(
            rel < 1e-3,
            "CTMM NLL under an infusion into the absorption compartment is {nll_infused}, \
             but the equivalent sub-dose train gives {nll_train} (rel {rel:.2e}) — the \
             generator is seeing the wrong exposure"
        );
    }

    /// #1575 (M1). The drug-driven generator reads its concentration from a dense solve
    /// under the **baseline** snapshot. Until #1575 that solve read every dose-record
    /// quantity there too, so a covariate on an absorption parameter that differed between
    /// baseline and the dose records — `KA = 0.6 − WT`, `−0.4` at a baseline `WT = 1`, `0.6`
    /// at both doses — clamped the kernel and the generator saw **no drug**: the NLL equalled
    /// the drug-free one (measured 2.2867 vs 4.9106, −5.25 OFV), with every dose in domain.
    ///
    /// Bad and control differ only in the baseline `WT`, which reaches nothing but `KA`, and
    /// `KA` is a dose quantity — so their NLLs must be bit-identical. The straddle is
    /// asserted through `pk_param_fn`, and the control is pinned away from the drug-free NLL,
    /// so a solve that lost the dose in both arms cannot pass. Dies if
    /// `ctmm_endpoint_nll_inhomogeneous` hands the dense solve `Shared` reads.
    #[test]
    fn one_snapshot_ctmm_reads_dose_quantities_at_the_dose_record() {
        use crate::types::DoseEvent;

        let src = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(0.6, 0.01, 50.0)
  theta LQ01(-0.7, -6.0, 3.0)
  theta LQ10(-1.2, -6.0, 3.0)
  theta SLOPE(0.4, -5.0, 5.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA - WT

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = first_order(ka=KA) - CL/V * central

[error_model]
  DV ~ proportional(PROP)

[markov_model]
  type   = ctmm
  cmt    = 5
  states = [s0=0, s1=1]
  transition s0 -> s1 = exp(LQ01 + SLOPE * (central / V))
  transition s1 -> s0 = exp(LQ10)
";
        let model = crate::parser::model_parser::parse_model_string(src).expect("parse");
        let theta = &model.default_params.theta;
        let eta = [0.0_f64];
        let wt = |v: f64| HashMap::from([("WT".to_string(), v)]);
        let obs = [(0.0, 0), (2.0, 1), (5.0, 0), (9.0, 1), (14.0, 0)];
        let subject = |wt_baseline: f64| {
            let mut s = ctmm_subject(&obs);
            // Two doses, the second on residual drug.
            s.doses = vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(6.0, 100.0, 1, 0.0, false, 0.0),
            ];
            s.covariates = wt(wt_baseline);
            s.dose_covariates = vec![wt(0.0), wt(0.0)];
            s
        };
        let (bad, ctl) = (subject(1.0), subject(0.0));

        // The straddle: KA differs at the baseline snapshot the solve integrates under, and
        // is the control's in-domain value at both dose records.
        let ka_slot = model.pk_indices[model
            .indiv_param_names
            .iter()
            .position(|n| n == "KA")
            .expect("KA")];
        let ka = |cov: &HashMap<String, f64>, t: f64| {
            (model.pk_param_fn)(theta, &eta, cov, t).values[ka_slot]
        };
        assert_eq!(ka(&bad.covariates, 0.0), 0.6 - 1.0, "bad baseline KA");
        assert_eq!(ka(&ctl.covariates, 0.0), 0.6, "control baseline KA");
        for (k, d) in bad.doses.iter().enumerate() {
            assert_eq!(ka(&bad.dose_covariates[k], d.time), 0.6, "KA at dose {k}");
        }

        let (nll_bad, nll_ctl) = (
            ctmm_subject_nll(&model, &bad, theta, &eta),
            ctmm_subject_nll(&model, &ctl, theta, &eta),
        );
        let drug_free = ctmm_subject_nll(&model, &ctmm_subject(&obs), theta, &eta);
        assert!(
            nll_ctl.is_finite() && nll_ctl != SUBJECT_SENTINEL_NLL,
            "control NLL {nll_ctl}"
        );
        assert!(
            (nll_ctl - drug_free).abs() > 1e-1,
            "control NLL {nll_ctl} sits at the drug-free {drug_free}: the generator does not \
             see the dose, so the bit-identity below would be vacuous"
        );
        assert_eq!(
            nll_bad.to_bits(),
            nll_ctl.to_bits(),
            "the CTMM solve read KA at the baseline snapshot instead of the dose record: \
             {nll_bad} vs {nll_ctl} (drug-free {drug_free})"
        );
    }

    /// A subject with a single observation carries no transition ⇒ 0 contribution.
    #[test]
    fn ctmm_endpoint_single_obs_is_zero() {
        assert_eq!(nll(&[0, 1], &const_gen(0.4, 0.6), &[disc(0.0, 0, 5)]), 0.0);
    }

    /// An undeclared DV code hits the defensive sentinel in the hot path (and is
    /// what `validate_ctmm_states` rejects up front).
    #[test]
    fn ctmm_endpoint_unknown_code_sentinels() {
        let got = nll(
            &[0, 1],
            &const_gen(0.4, 0.6),
            &[disc(0.0, 0, 5), disc(1.0, 9, 5)],
        );
        assert_eq!(got, SUBJECT_SENTINEL_NLL);
    }

    /// A negative off-diagonal intensity (an unconstrained user expression driven
    /// negative) makes `Q` a non-generator; the endpoint repels with the sentinel
    /// rather than let `expm` produce a super-stochastic entry the kernel clamps to a
    /// 0 (minimum) penalty. `const_gen(a, b)` with `a < 0` puts `-a` at the `0→1`
    /// off-diagonal.
    #[test]
    fn ctmm_endpoint_negative_offdiagonal_sentinels() {
        let got = nll(
            &[0, 1],
            &const_gen(-0.4, 0.6), // q[0,1] = -0.4 < 0 ⇒ invalid generator
            &[disc(0.0, 0, 5), disc(1.0, 1, 5)],
        );
        assert_eq!(got, SUBJECT_SENTINEL_NLL);
        // A valid (non-negative off-diagonal) generator on the same data does not.
        let ok = nll(
            &[0, 1],
            &const_gen(0.4, 0.6),
            &[disc(0.0, 0, 5), disc(1.0, 1, 5)],
        );
        assert!(ok.is_finite() && ok < SUBJECT_SENTINEL_NLL, "got {ok}");
    }

    /// `validate_ctmm_times` rejects out-of-order observation times (which would
    /// otherwise collapse the subject to the silent sentinel) and passes sorted rows.
    #[test]
    fn validate_ctmm_times_rejects_out_of_order() {
        // Decreasing 2.0 → 1.0 on CMT 5.
        let recs = [disc(0.0, 0, 5), disc(2.0, 1, 5), disc(1.0, 0, 5)];
        let err = validate_ctmm_times(5, &recs).unwrap_err();
        assert!(err.contains("cmt = 5"), "msg: {err}");
        assert!(err.contains("non-decreasing"), "msg: {err}");
        // Sorted rows pass; equal times (Δt = 0) are allowed (non-decreasing).
        assert!(
            validate_ctmm_times(5, &[disc(0.0, 0, 5), disc(1.0, 1, 5), disc(1.0, 1, 5)]).is_ok()
        );
        // Out-of-order rows on a *different* CMT are ignored for this endpoint.
        assert!(validate_ctmm_times(5, &[disc(2.0, 0, 5), disc(1.0, 1, 6)]).is_ok());
    }

    /// `validate_ctmm_states` names the CMT, the offending DV, and the declared set.
    #[test]
    fn validate_rejects_undeclared_code() {
        let recs = [disc(0.0, 0, 5), disc(1.0, 2, 5)];
        let err = validate_ctmm_states(5, &[0, 1], &recs).unwrap_err();
        assert!(err.contains("cmt = 5"), "msg: {err}");
        assert!(err.contains("DV 2"), "msg: {err}");
        assert!(err.contains("[0, 1]"), "msg: {err}");
        // A clean subject passes; rows on another CMT are ignored.
        assert!(validate_ctmm_states(5, &[0, 1], &[disc(0.0, 0, 5), disc(1.0, 7, 6)]).is_ok());
    }

    /// Rows on a second endpoint's CMT are not folded into this endpoint's chain.
    #[test]
    fn collect_state_obs_filters_by_cmt() {
        let recs = [disc(0.0, 0, 5), disc(1.0, 3, 6), disc(2.0, 1, 5)];
        let obs = collect_state_obs(5, &[0, 1], &recs).unwrap();
        assert_eq!(obs.len(), 2);
        assert_eq!(obs[0].state, 0);
        assert_eq!(obs[1].state, 1);
        assert_eq!(obs[1].time, 2.0);
    }
}

#[cfg(test)]
mod exact_eta_hessian_tests {
    use super::*;

    /// The homogeneous CTMM η-Hessian (second derivative of `expm` on a dual) against the
    /// stencil it replaces, to the stencil's accuracy; and the inhomogeneous (drug-driven)
    /// endpoint, which has no generator program, declines to it.
    #[test]
    fn the_ctmm_eta_hessian_is_exact_on_a_homogeneous_generator() {
        let prep = crate::categorical::exact_eta_hessian_tests::load(
            r"
[parameters]
  theta LQ01(-0.7, -6.0, 3.0)
  theta LQ10(-1.2, -6.0, 3.0)
  omega ETA_A ~ 0.2
  omega ETA_B ~ 0.3
[markov_model]
  type   = ctmm
  cmt    = 5
  states = [awake=0, asleep=1]
  transition awake  -> asleep = exp(LQ01 + ETA_A)
  transition asleep -> awake  = exp(LQ10 + 0.5 * ETA_A + ETA_B)
",
            "data/ctmm_2state.csv",
        );
        let model = &prep.parsed.model;
        let theta = &model.default_params.theta;
        let eta = [0.09, -0.12];
        let mut worst = 0.0f64;
        let mut n = 0;
        for s in prep.population.subjects.iter().take(6) {
            if s.obs_records.is_empty() {
                continue;
            }
            let h = ctmm_subject_eta_hessian(model, s, theta, &eta).expect("in scope");
            let f = |e: &[f64]| ctmm_subject_nll(model, s, theta, e);
            let steps = crate::survival::shi_step_sizes(&f, &eta);
            let fd = crate::survival::data_term_hessian_fd(&f, &eta, &steps);
            let scale = fd.iter().fold(1e-300f64, |a, &v| a.max(v.abs()));
            for (a, b) in h.iter().zip(fd.iter()) {
                assert!(a.is_finite());
                worst = worst.max((a - b).abs() / scale);
            }
            n += 1;
        }
        assert!(n > 2, "too few subjects ({n})");
        eprintln!("CTMM η-Hessian vs the stencil: worst {worst:.2e}");
        // Measured 5.5e-5: the stencil's own error. A dropped clamp or gap misses by O(1).
        assert!(worst < 2e-4, "{worst:.2e}");

        let root = env!("CARGO_MANIFEST_DIR");
        let pd = crate::api::prepare_run(
            &format!("{root}/examples/ctmm_pd_2state.ferx"),
            Some(&format!("{root}/data/ctmm_pd_2state.csv")),
        )
        .unwrap();
        let m = &pd.parsed.model;
        if m.n_eta > 0 {
            let s = pd
                .population
                .subjects
                .iter()
                .find(|s| !s.obs_records.is_empty())
                .unwrap();
            let e = vec![0.0; m.n_eta];
            // The drug-driven example is served too (`a_drug_driven_ctmm_is_differentiated_exactly`
            // pins the values).
            let h = ctmm_subject_eta_hessian(m, s, &m.default_params.theta, &e)
                .expect("a drug-driven generator is served exactly");
            assert!(h.iter().all(|v| v.is_finite()));
        }
    }
}
