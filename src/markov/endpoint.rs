//! CTMM endpoint wiring (Phase 5, Track D — #759).
//!
//! Bridges the model layer to the pure numerical [`ctmm_data_term`] kernel in
//! [`crate::markov`]. Its job is everything the leaf deliberately does **not**
//! know about (§8.7 of `plans/tte-survival-markov.md`):
//!
//! - **build the generator** `Q(θ, η, cov)` from the endpoint's
//!   [`GeneratorFn`](crate::types::GeneratorFn) (the `[markov_model] transition`
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
const SUBJECT_SENTINEL_NLL: f64 = 1e20;

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

/// Exact `∂/∂η` of [`ctmm_subject_nll`] — the analytic counterpart of finite-differencing
/// the whole matrix-exponential likelihood (#759).
///
/// `Q` is rebuilt over `Dual1` with θ and η seeded
/// ([`CtmmGeneratorProgram::eval_generator_duals`]), giving an exact `∂Q/∂(θ,η)`; the η
/// columns are then chained through the Van Loan Fréchet derivative of `expm` by
/// [`ctmm_data_term_grad`]. Returns `(value, ∂/∂η)` with the gradient at the **same 1×
/// scale as the value** — the caller applies the objective's factor.
///
/// `None` — caller falls back to FD for this point — when *any* CTMM endpoint on the
/// subject cannot be served exactly:
///   * a time-**inhomogeneous** generator (no program: the likelihood is an occupancy ODE,
///     not an `expm`, so Van Loan does not apply);
///   * a `(θ, η)` width past the dual dispatch cap, or intensities we could not resolve;
///   * the degenerate regime — a negative off-diagonal rate, or the `expm`/underflow
///     guards — where the value is a flat `SUBJECT_SENTINEL_NLL` repellent with no
///     meaningful derivative.
///
/// Only the **η** block is returned today (the inner EBE loop is the consumer). The
/// program seeds the θ axes too, so an outer θ-gradient is a projection away — see the
/// non-Gaussian outer-gradient split (#486).
pub fn ctmm_subject_eta_grad(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<(f64, Vec<f64>)> {
    let n_eta = model.n_eta;
    let mut nll = 0.0;
    let mut grad = vec![0.0_f64; n_eta];
    if subject.obs_records.is_empty() {
        return Some((nll, grad));
    }
    for (cmt, endpoint) in &model.endpoints {
        let EndpointLikelihood::Ctmm {
            n_states,
            state_codes,
            generator_program,
            ..
        } = endpoint
        else {
            continue;
        };
        // No program ⇒ inhomogeneous, or not exactly representable. Decline the whole
        // subject rather than serve a partial gradient.
        let prog = generator_program.as_ref()?;

        let obs = collect_state_obs(*cmt, state_codes, &subject.obs_records).ok()?;
        if obs.len() < 2 {
            continue;
        }
        let (q, dq) = prog.eval_generator_duals(theta, eta, &subject.covariates)?;
        debug_assert_eq!(q.nrows(), *n_states);

        // Same degeneracy guard as the value path (`ctmm_endpoint_nll`): a negative
        // off-diagonal makes `Q` a non-generator and the value collapses to the flat
        // sentinel, which has no derivative.
        if super::has_negative_offdiagonal(&q) {
            return None;
        }

        // Project the (θ, η)-seeded jets onto the η axes. `dq` is laid out θ₀..θ_{T−1},
        // η₀..η_{E−1}, so the η block is exactly the tail — borrow it, no clones.
        //
        // A length mismatch would mean the program's axis layout disagrees with the model's
        // η count. Decline (→ FD) rather than zero-fill the missing axes: a zero jet is a
        // *wrong* gradient reported as analytic, which is the failure mode this whole
        // change set exists to remove.
        let n_theta = prog.n_theta_axis();
        if dq.len() != n_theta + n_eta {
            return None;
        }
        let (v, g) = crate::markov::ctmm_data_term_grad(&q, &dq[n_theta..], &obs).ok()??;
        nll += v;
        for (acc, gi) in grad.iter_mut().zip(g.iter()) {
            *acc += gi;
        }
    }
    Some((nll, grad))
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

    // PK/PD parameter snapshot at (θ, η) — the ODE solve reads it (baseline covariate
    // snapshot; a time-varying covariate on the generator is rejected at fit setup).
    let pk = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);

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
    let states = crate::ode::ode_dense_solve_states(ode, &pk.values, theta, eta, subject, &grid);
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
            &ode.solver_opts,
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
/// [`ctmm_endpoint_nll`] maps to the [`SUBJECT_SENTINEL_NLL`] backstop — silently
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

    /// A time-**inhomogeneous** (drug-driven) generator gets no program — its likelihood is
    /// an occupancy ODE, not an `expm`, so the Van Loan identity does not apply. It must
    /// decline to FD rather than silently return the homogeneous answer.
    #[test]
    fn inhomogeneous_ctmm_declines_the_analytic_grad() {
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
        let model = crate::parser::model_parser::parse_model_string(src).expect("parse");
        let subject = ctmm_subject(&[(0.0, 0), (1.0, 1)]);
        assert!(
            ctmm_subject_eta_grad(&model, &subject, &model.default_params.theta, &[0.1]).is_none(),
            "a state-driven generator must decline the expm-based analytic gradient"
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
