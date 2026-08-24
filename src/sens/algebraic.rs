//! Analytic sensitivities for a **compartment-free** (`$PRED`-equivalent) model
//! (issue #811).
//!
//! This is the degenerate case of the Form C readout jet, and the easiest one in
//! `sens/`: with no compartments there is no state to integrate, no dose walk, no
//! superposition and no `central = conc × V` reconstruction. The whole chain is
//!
//! ```text
//!   ∂y/∂θ = (∂y/∂p)·(∂p/∂θ)          ∂y/∂η = (∂y/∂p)·(∂p/∂η)
//! ```
//!
//! where `∂p/∂(θ,η)` comes from the individual-parameter program (the same
//! [`seed_pk_dual2`] / [`seed_pk_dual1`] seeders the ODE provider uses) and
//! `∂y/∂p` from evaluating the readout program over the seeded duals. Both orders
//! come out of one dual evaluation per observation, so the outer walk yields the
//! `η-η` Hessian and the `η-θ` cross block FOCEI needs without a second pass.
//!
//! # Matching production exactly
//!
//! The f64 predictor for such a model is `pk::apply_analytic_readout` over an
//! empty `state[]` (`pk/mod.rs`), and these walks must linearise about *that*
//! evaluation or the Dual2-vs-FD parity test fails. Three details carry the whole
//! agreement, and each mirrors a specific line there:
//!
//! - **Snapshot cadence.** Production re-evaluates the readout's individual
//!   parameters per observation only when a covariate can move them between rows
//!   (`has_tv_covariates`) or the model reads the `TIME` built-in; otherwise one
//!   subject-static snapshot at `(subject.covariates, t = 0)` drives every row.
//! - **Two different clocks.** The parameter snapshot uses the *observation* time
//!   (`obs_times[j]`), the readout evaluation uses the *raw data-file* time
//!   (`readout_time(j)`, the `$ERROR` convention, #1028). They differ only under
//!   stacked reset occasions, but they differ.
//! - **The output transform.** LTBS is applied to the readout output, through the
//!   shared generic [`apply_output_transform`] so the floor-then-log semantics
//!   (and their derivative at the floor) cannot drift from the f64 path.
//!
//! # Scope
//!
//! Declined (→ finite differences, loudly, via [`supported`]) for IOV (`n_kappa >
//! 0`), for a per-CMT readout, for a readout that is not dual-evaluable (a bare
//! θ/η the parser could not desugar, or a neural-network output), and past the
//! axis cap. Each is a *route* decline, not a silent wrong gradient — the same
//! contract the closed-form and ODE providers hold.

use super::dual1::Dual1;
use super::dual2::Dual2;
use super::ode_provider::{apply_output_transform, seed_pk_dual1, seed_pk_dual2, MAX_ODE_AXES};
use super::provider::{obs_sens_from_dual2, ObsGrad, ObsSens, SubjectSens};
use crate::parser::model_parser::{
    compiled_model_uses_time_builtin, IndivParamProgram, ModelTimeGuard, OdeOutputProgram,
};
use crate::types::{CompiledModel, Subject};

/// Widest `(θ, η)` axis count the dispatch ladders below instantiate.
///
/// Shares [`MAX_ODE_AXES`] rather than picking its own number: the inner walk's
/// `∂p/∂η` comes from `param_derivatives_at_cov`, whose own dispatch table is
/// bounded by that constant, so a model admitted here but wider than that would
/// take an analytic outer gradient against an FD inner — the scope split the ODE
/// provider forbids for the same reason.
///
/// Raising it is cheaper here than there (no integrator is monomorphised behind
/// these walks, only a bytecode readout eval), but it must be raised *together*
/// with `MAX_ODE_AXES` and its four tables, never alone.
const MAX_ALGEBRAIC_AXES: usize = MAX_ODE_AXES;

/// The two programs a compartment-free model's sensitivities are built from: the
/// individual-parameter program (`∂p/∂(θ,η)`) and the readout program (`∂y/∂p`).
///
/// `None` when either is absent or the readout is a shape these walks do not
/// serve — the single scope predicate [`supported`] and both entry points read it,
/// so the gate and the route cannot disagree (#637).
fn programs(model: &CompiledModel) -> Option<(&IndivParamProgram, &OdeOutputProgram)> {
    if !model.is_algebraic() {
        return None;
    }
    // IOV: the readout would have to be evaluated per occasion under a stacked
    // `(θ, η_bsv, κ)` seeding, as the f64 path already does (`predict_iov`).
    // Deferred — such a subject routes to FD, which differentiates that same
    // per-occasion predictor and is therefore correct, just slower.
    if model.n_kappa > 0 {
        return None;
    }
    let readout = model.analytic_readout.as_ref()?;
    // Per-CMT readouts carry their program on each branch rather than one uniform
    // program; the closed-form provider declines them too (`analytic_readout_dual_supported`).
    let prog_out = match &readout.readout {
        crate::ode::OdeReadout::Single(_) => readout.program.as_ref()?,
        _ => return None,
    };
    if !prog_out.is_dual_evaluable() {
        return None;
    }
    let prog_indiv = model
        .indiv_param_partials
        .indiv_param_program
        .as_ref()
        .filter(|p| p.n_theta_axis() == model.n_theta && p.n_eta_axis() == model.n_eta)?;
    Some((prog_indiv, prog_out))
}

/// Whether the analytic path here serves `model`. Callers that report a gradient
/// method must use this, not `is_algebraic()` alone: a compartment-free model
/// outside this scope is served correctly by finite differences, and saying
/// "analytic" while every subject falls back is the misreport #637 exists to
/// prevent.
pub(crate) fn supported(model: &CompiledModel) -> bool {
    programs(model).is_some() && (1..=MAX_ALGEBRAIC_AXES).contains(&(model.n_theta + model.n_eta))
}

/// Per-observation `(parameter snapshot time, readout time, covariates)`.
///
/// Mirrors `pk::apply_analytic_readout`'s cadence: one subject-static snapshot
/// unless a covariate or the `TIME` built-in can move the individual parameters
/// between rows. The two times are deliberately different — see the module note.
fn obs_context(
    subject: &Subject,
    j: usize,
    per_obs: bool,
) -> (f64, f64, &std::collections::HashMap<String, f64>) {
    let readout_time = subject.readout_time(j);
    if per_obs {
        (
            subject.obs_times.get(j).copied().unwrap_or(0.0),
            readout_time,
            subject.obs_cov(j),
        )
    } else {
        (0.0, readout_time, &subject.covariates)
    }
}

/// Full second-order sensitivities (`Dual2<M>`, `M = n_theta + n_eta`) for the
/// FOCE/FOCEI outer gradient.
fn run_obs<const M: usize>(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<SubjectSens> {
    let (prog_indiv, prog_out) = programs(model)?;
    let (n_theta, n_eta) = (model.n_theta, model.n_eta);
    debug_assert_eq!(M, n_theta + n_eta);
    let per_obs = subject.has_tv_covariates() || compiled_model_uses_time_builtin(model);

    // Scratch reused across observations; the readout eval clears both.
    let (mut vars, mut stack): (Vec<Dual2<M>>, Vec<Dual2<M>>) = (Vec::new(), Vec::new());
    // One static snapshot when the parameters cannot move between rows — the same
    // saving production makes, and the reason this is not simply "seed per row".
    let static_params: Option<Vec<Dual2<M>>> = (!per_obs)
        .then(|| seed_pk_dual2::<M>(model, prog_indiv, theta, eta, &subject.covariates, 0.0));

    let mut obs: Vec<ObsSens> = Vec::with_capacity(subject.obs_times.len());
    for j in 0..subject.obs_times.len() {
        let (param_time, readout_time, cov) = obs_context(subject, j, per_obs);
        let owned;
        let params: &[Dual2<M>] = match &static_params {
            Some(p) => p,
            None => {
                owned = seed_pk_dual2::<M>(model, prog_indiv, theta, eta, cov, param_time);
                &owned
            }
        };
        // A readout referencing `TIME` resolves `Op::PushTime` from the model-time
        // thread-local; enter this observation's readout clock, exactly as
        // `apply_analytic_readout` does. A no-op for a `TIME`-free readout.
        let y = {
            let _guard = ModelTimeGuard::enter(readout_time);
            prog_out.eval_output_g::<Dual2<M>>(&[], params, cov, &mut vars, &mut stack)
        };
        let y = apply_output_transform(model, y);
        obs.push(obs_sens_from_dual2(&y, n_theta, n_eta));
    }
    Some(SubjectSens { obs })
}

/// First-order η sensitivities (`Dual1<N>`, `N = n_eta`) for the inner EBE loop.
fn run_obs_grad<const N: usize>(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<Vec<ObsGrad>> {
    let (prog_indiv, prog_out) = programs(model)?;
    let n_eta = model.n_eta;
    debug_assert_eq!(N, n_eta);
    let per_obs = subject.has_tv_covariates() || compiled_model_uses_time_builtin(model);

    let (mut vars, mut stack): (Vec<Dual1<N>>, Vec<Dual1<N>>) = (Vec::new(), Vec::new());
    let static_params: Option<Vec<Dual1<N>>> = if per_obs {
        None
    } else {
        Some(seed_pk_dual1::<N>(
            model,
            prog_indiv,
            theta,
            eta,
            &subject.covariates,
            0.0,
        )?)
    };

    let mut out: Vec<ObsGrad> = Vec::with_capacity(subject.obs_times.len());
    for j in 0..subject.obs_times.len() {
        let (param_time, readout_time, cov) = obs_context(subject, j, per_obs);
        let owned;
        let params: &[Dual1<N>] = match &static_params {
            Some(p) => p,
            None => {
                owned = seed_pk_dual1::<N>(model, prog_indiv, theta, eta, cov, param_time)?;
                &owned
            }
        };
        let y = {
            let _guard = ModelTimeGuard::enter(readout_time);
            prog_out.eval_output_g::<Dual1<N>>(&[], params, cov, &mut vars, &mut stack)
        };
        let y = apply_output_transform(model, y);
        out.push(ObsGrad {
            f: y.value,
            df_deta: y.grad[..n_eta].to_vec(),
        });
    }
    Some(out)
}

/// Outer entry point: per-observation `(f, ∂f/∂η, ∂²f/∂η², ∂f/∂θ, ∂²f/∂η∂θ)`, or
/// `None` when this model is outside [`supported`] (caller → FD).
pub(crate) fn subject_sensitivities(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<SubjectSens> {
    // The ladder and `supported` are keyed on the same cap, so an in-scope model
    // can never fall through to `_ => None` and route to FD while the fit reports
    // "analytic" (#438/#466/#534 tripwire convention).
    macro_rules! disp {
        ($($m:literal),+) => {
            match model.n_theta + model.n_eta {
                $($m => run_obs::<$m>(model, subject, theta, eta),)+
                _ => None,
            }
        };
    }
    disp!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)
}

/// Inner entry point: per-observation `(f, ∂f/∂η)`, or `None` (caller → FD).
pub(crate) fn subject_eta_grad(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<Vec<ObsGrad>> {
    macro_rules! disp {
        ($($n:literal),+) => {
            match model.n_eta {
                $($n => run_obs_grad::<$n>(model, subject, theta, eta),)+
                _ => None,
            }
        };
    }
    disp!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)
}

const _: () = assert!(
    MAX_ALGEBRAIC_AXES == 24,
    "the two `disp!` ladders above enumerate 1..=24 explicitly; widening the cap \
     without widening both would let an in-scope model hit `_ => None` and fall \
     back to FD while the fit reports an analytic gradient"
);

#[cfg(test)]
#[path = "algebraic_tests.rs"]
mod tests;
