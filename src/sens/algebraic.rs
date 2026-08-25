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
//! Two pairs of walks: [`subject_sensitivities`] / [`subject_eta_grad`] for a
//! model with between-subject variability only, and
//! [`subject_sensitivities_iov`] / [`subject_eta_grad_iov`] for one that also
//! carries κ (see the IOV section below). The BSV walks decline a κ-carrying model
//! outright — its κ has nowhere to go in their axis layout — which is why
//! [`supported`] and [`supported_iov`] are separate predicates rather than one.
//!
//! Declined (→ finite differences, loudly, at the gate) for a per-CMT readout, for
//! a readout that is not dual-evaluable (a bare θ/η the parser could not desugar,
//! or a neural-network output), and past the axis cap. Each is a *route* decline,
//! not a silent wrong gradient — the same contract the closed-form and ODE
//! providers hold.

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
/// It is a **separate constant on purpose**, initialised to `MAX_ODE_AXES` rather
/// than spelled as it: the coupling is a real constraint today, but it is one
/// direction of dependency, and raising this cap should never be confused with
/// raising the ODE one (which drags four `disp!` ladders and the whole integrator
/// stack with it).
///
/// **Do not raise it.** Measured per gradient evaluation on a compartment-free
/// model (8 observations, one subject, release build):
///
/// | axes | analytic | FD of the predictor | ratio |
/// |---|---|---|---|
/// | 5  | 5.6 µs   | 4.7 µs  | 1.2× |
/// | 14 | 28.1 µs  | 10.8 µs | 2.6× |
/// | 24 | 141.8 µs | 26.6 µs | 5.3× |
///
/// `Dual2<M>` is O(M²) while the prediction it differentiates is a single scalar
/// expression per row, so the jet loses to finite differences here and loses
/// faster the wider it gets — the opposite of the ODE case, where one integration
/// dominates any number of dual lanes. The cap is therefore not a performance
/// cliff to be pushed back; past it, FD is the better route anyway. (Whether the
/// crossover means these models should prefer FD *below* the cap too is a real
/// question, but it is a change of default, not a constant — see the follow-up.)
pub(crate) const MAX_ALGEBRAIC_AXES: usize = MAX_ODE_AXES;

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
    // These are the **BSV** walks: their axis layout has no room for κ, so a
    // κ-carrying model is not theirs to serve. It is not declined outright —
    // [`supported_iov`] and the stacked walks below take it. Keep this predicate
    // and that one separate rather than merging them: the two layouts differ, and a
    // single "supported" that covered both would have to be re-derived at every
    // call site to know which pair of walks it licensed.
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

// ── Inter-occasion variability ──────────────────────────────────────────────
//
// A compartment-free model's κ enters through the individual parameters the
// readout reads, and nowhere else — there is no concentration to carry a
// per-occasion dynamic. That is why `pk::predict_iov` evaluates the readout **per
// occasion** for such a model instead of once from the BSV η, and it is the same
// reason these walks seed the κ axes rather than dropping them: a BSV-only jet
// would report `∂y/∂κ = 0` for a model whose whole between-arm structure lives
// there.
//
// The axis layout is the stacked one every IOV provider uses: θ on `0..n_theta`,
// η_bsv on `n_theta..n_theta+n_eta`, and occasion group `g`'s κ block at
// `n_theta + n_eta + g·n_kappa`. Widths are bucketed (`ODE_IOV_WIDTH_BUCKETS`)
// because the stacked count grows with the number of occasions — an MBMA study
// with many treatment arms is exactly that case — and monomorphising every width
// to 96 is the compile-cost trap #971 measured.

/// One observation's occasion-group index, or `None` when it carries no occasion
/// label (κ held at 0, matching `pk::predict_iov`).
type ObsGroups = Vec<Option<usize>>;

/// Per-observation occasion groups and the stacked-η width, or `None` when the
/// subject's labelling does not fit the stacked convention.
fn iov_layout(
    model: &CompiledModel,
    subject: &Subject,
    stacked_eta: &[f64],
) -> Option<(ObsGroups, usize)> {
    let occ_groups = crate::stats::likelihood::iov_occasion_groups(subject);
    if occ_groups.is_empty() {
        return None;
    }
    let n_stacked = model.n_eta + occ_groups.len() * model.n_kappa;
    if stacked_eta.len() != n_stacked {
        return None;
    }
    let occ_to_k = crate::stats::likelihood::iov_occ_to_k(&occ_groups);
    // An unlabelled observation is legitimate (production holds its κ at 0); an
    // observation whose label is not a known group is not, and declining beats
    // guessing which block its κ derivatives belong in.
    let groups = (0..subject.obs_times.len())
        .map(|j| match subject.occasions.get(j) {
            Some(occ) => occ_to_k.get(occ).copied().map(Some),
            None => Some(None),
        })
        .collect::<Option<ObsGroups>>()?;
    Some((groups, n_stacked))
}

/// Widest stacked `(θ, η_bsv, κ…)` axis count these IOV walks admit. Shares the
/// ODE IOV cap and its bucket ladder: the stacked width is the only thing that
/// grows with the occasion count, while the per-occasion derivative source
/// (`iov_combined_derivs_dyn`) runs over `(θ, η_bsv, κ_current)` and stays under
/// the ordinary 24-axis table — which `supported_iov` checks separately.
const MAX_ALGEBRAIC_IOV_AXES: usize = crate::sens::ode_provider::MAX_ODE_IOV_AXES;

/// Whether these IOV walks serve `model`. As with [`supported`], a caller that
/// reports a gradient method must consult this rather than `is_algebraic()`.
///
/// The per-occasion width bound (`n_theta + n_eta + n_kappa ≤ MAX_ALGEBRAIC_AXES`)
/// is the `iov_combined_derivs_dyn` table's, and is checked here rather than
/// discovered at the first observation — otherwise a model past it would report
/// analytic and then decline per subject.
pub(crate) fn supported_iov(model: &CompiledModel) -> bool {
    if !model.is_algebraic() || model.n_kappa == 0 {
        return false;
    }
    // `programs` declines every model with a κ (the non-IOV walks cannot serve
    // one), so re-do its readout/program checks here without that clause.
    let Some(readout) = model.analytic_readout.as_ref() else {
        return false;
    };
    let dual_ok = match (&readout.readout, &readout.program) {
        (crate::ode::OdeReadout::Single(_), Some(p)) => p.is_dual_evaluable(),
        _ => false,
    };
    // Under IOV the individual-parameter program is built over the **combined**
    // random-effect vector `[η_bsv, κ]`, so its η axis is `n_eta + n_kappa` — not
    // `n_eta`, which is what the non-IOV `programs` checks. Same convention as
    // `iov_analytical_supported`.
    let n_eff = model.n_eta + model.n_kappa;
    let prog_ok = model
        .indiv_param_partials
        .indiv_param_program
        .as_ref()
        .is_some_and(|p| p.n_theta_axis() == model.n_theta && p.n_eta_axis() == n_eff);
    dual_ok
        && prog_ok
        && (1..=MAX_ALGEBRAIC_AXES).contains(&(model.n_theta + model.n_eta + model.n_kappa))
}

/// The readout program and the individual-parameter program, for the IOV walks.
/// [`programs`] cannot be reused: it declines every κ-carrying model.
fn iov_programs(model: &CompiledModel) -> Option<(&IndivParamProgram, &OdeOutputProgram)> {
    if !supported_iov(model) {
        return None;
    }
    let readout = model.analytic_readout.as_ref()?;
    let prog_out = match &readout.readout {
        crate::ode::OdeReadout::Single(_) => readout.program.as_ref()?,
        _ => return None,
    };
    let prog_indiv = model.indiv_param_partials.indiv_param_program.as_ref()?;
    Some((prog_indiv, prog_out))
}

/// Second-order stacked-axis jet for the FOCE/FOCEI outer gradient under IOV.
fn run_obs_iov<const M: usize>(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    stacked_eta: &[f64],
) -> Option<SubjectSens> {
    let (prog_indiv, prog_out) = iov_programs(model)?;
    let (n_theta, n_eta, n_kappa) = (model.n_theta, model.n_eta, model.n_kappa);
    let n_eff = n_eta + n_kappa;
    let (groups, n_stacked) = iov_layout(model, subject, stacked_eta)?;
    let slot_row = crate::sens::provider::seed_dim_from_slots(prog_indiv.pk_slots_ref());
    let n_rows = prog_indiv.pk_slots_ref().len();
    let uses_time = compiled_model_uses_time_builtin(model);

    let (mut vars, mut stack): (Vec<Dual2<M>>, Vec<Dual2<M>>) = (Vec::new(), Vec::new());
    let mut obs: Vec<ObsSens> = Vec::with_capacity(subject.obs_times.len());
    for (j, &group) in groups.iter().enumerate() {
        // The occasion's combined effect: `[η_bsv, κ_g]`, or `[η_bsv, 0…]` for an
        // unlabelled observation — the same two cases `predict_iov` distinguishes.
        let combined = match group {
            Some(g) => {
                crate::stats::likelihood::iov_combined_effect(stacked_eta, n_eta, n_kappa, g)
            }
            None => crate::stats::likelihood::iov_combined_pk_only(stacked_eta, n_eta, n_kappa),
        };
        let cov = subject.obs_cov(j);
        let param_time = subject.obs_times.get(j).copied().unwrap_or(0.0);
        let pk = {
            let _guard = ModelTimeGuard::enter_if(uses_time, param_time);
            (model.pk_param_fn)(theta, &combined, cov, param_time)
        };
        let cd = {
            let _guard = ModelTimeGuard::enter_if(uses_time, param_time);
            crate::sens::provider::iov_combined_derivs_dyn(
                prog_indiv, n_theta, n_eff, n_rows, cov, theta, &combined,
            )?
        };

        // Combined column `c` → stacked axis. η_bsv is shared across occasions;
        // κ lands in this group's block, or is dropped when the row carries no
        // occasion (its κ is held at 0, so it has no derivative).
        let kappa_base = group.map(|g| n_theta + n_eta + g * n_kappa);
        let axis = |c: usize| -> Option<usize> {
            let ax = if c < n_eta {
                n_theta + c
            } else {
                kappa_base? + (c - n_eta)
            };
            (ax < M).then_some(ax)
        };
        let params: Vec<Dual2<M>> = (0..pk.values.len())
            .map(|s| {
                let val = pk.values[s];
                let Some(i) = slot_row.get(s).copied().flatten() else {
                    return Dual2::constant(val);
                };
                let n_th = n_theta.min(M);
                let mut grad = [0.0; M];
                let mut hess = [[0.0; M]; M];
                grad[..n_th].copy_from_slice(&cd.dtheta[i][..n_th]);
                for c in 0..n_eff {
                    let Some(ax) = axis(c) else { continue };
                    grad[ax] = cd.deta[i][c];
                    for d in 0..n_eff {
                        if let Some(bx) = axis(d) {
                            hess[ax][bx] = cd.d2eta[i][c][d];
                        }
                    }
                    // Writes the symmetric pair `hess[ax][m]` / `hess[m][ax]`, so
                    // the index is used on both sides of the matrix — an iterator
                    // over one row cannot express it.
                    #[allow(clippy::needless_range_loop)]
                    for m in 0..n_th {
                        let v = cd.d2eta_theta[i][c][m];
                        hess[ax][m] = v;
                        hess[m][ax] = v;
                    }
                }
                Dual2 {
                    value: val,
                    grad,
                    hess,
                }
            })
            .collect();

        let y = {
            let _guard = ModelTimeGuard::enter(subject.readout_time(j));
            prog_out.eval_output_g::<Dual2<M>>(&[], &params, cov, &mut vars, &mut stack)
        };
        let y = apply_output_transform(model, y);
        // Scatter over the **stacked** η, which is what the block-Ω assembly
        // (`prepare_stacked`) consumes for an IOV subject.
        obs.push(obs_sens_from_dual2(&y, n_theta, n_stacked));
    }
    Some(SubjectSens { obs })
}

/// First-order stacked-axis jet for the inner EBE loop under IOV.
fn run_obs_grad_iov<const N: usize>(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    stacked_eta: &[f64],
) -> Option<Vec<ObsGrad>> {
    let (prog_indiv, prog_out) = iov_programs(model)?;
    let (n_theta, n_eta, n_kappa) = (model.n_theta, model.n_eta, model.n_kappa);
    let n_eff = n_eta + n_kappa;
    let (groups, n_stacked) = iov_layout(model, subject, stacked_eta)?;
    let slot_row = crate::sens::provider::seed_dim_from_slots(prog_indiv.pk_slots_ref());
    let n_rows = prog_indiv.pk_slots_ref().len();
    let uses_time = compiled_model_uses_time_builtin(model);

    let (mut vars, mut stack): (Vec<Dual1<N>>, Vec<Dual1<N>>) = (Vec::new(), Vec::new());
    let mut out: Vec<ObsGrad> = Vec::with_capacity(subject.obs_times.len());
    for (j, &group) in groups.iter().enumerate() {
        let combined = match group {
            Some(g) => {
                crate::stats::likelihood::iov_combined_effect(stacked_eta, n_eta, n_kappa, g)
            }
            None => crate::stats::likelihood::iov_combined_pk_only(stacked_eta, n_eta, n_kappa),
        };
        let cov = subject.obs_cov(j);
        let param_time = subject.obs_times.get(j).copied().unwrap_or(0.0);
        let pk = {
            let _guard = ModelTimeGuard::enter_if(uses_time, param_time);
            (model.pk_param_fn)(theta, &combined, cov, param_time)
        };
        let cd = {
            let _guard = ModelTimeGuard::enter_if(uses_time, param_time);
            crate::sens::provider::iov_combined_derivs_dyn(
                prog_indiv, n_theta, n_eff, n_rows, cov, theta, &combined,
            )?
        };
        // Inner axes are the stacked η alone — no θ block, no Hessian.
        let kappa_base = group.map(|g| n_eta + g * n_kappa);
        let axis = |c: usize| -> Option<usize> {
            let ax = if c < n_eta {
                c
            } else {
                kappa_base? + (c - n_eta)
            };
            (ax < N).then_some(ax)
        };
        let params: Vec<Dual1<N>> = (0..pk.values.len())
            .map(|s| {
                let val = pk.values[s];
                let Some(i) = slot_row.get(s).copied().flatten() else {
                    return Dual1::constant(val);
                };
                let mut grad = [0.0; N];
                for c in 0..n_eff {
                    if let Some(ax) = axis(c) {
                        grad[ax] = cd.deta[i][c];
                    }
                }
                Dual1 { value: val, grad }
            })
            .collect();

        let y = {
            let _guard = ModelTimeGuard::enter(subject.readout_time(j));
            prog_out.eval_output_g::<Dual1<N>>(&[], &params, cov, &mut vars, &mut stack)
        };
        let y = apply_output_transform(model, y);
        out.push(ObsGrad {
            f: y.value,
            df_deta: y.grad[..n_stacked.min(N)].to_vec(),
        });
    }
    Some(out)
}

/// Outer IOV entry point over the stacked `[η_bsv, κ_0…κ_{K−1}]` vector.
///
/// Widths are **bucketed**: the runtime stacked count is rounded up to the next
/// entry of `ODE_IOV_WIDTH_BUCKETS` and the extra lanes are left zero, which is
/// inert (every seed guards its axis writes, every read indexes by the runtime
/// count). Enumerating all 96 widths instead would monomorphise this walk 96
/// times for no numerical gain — the compile-cost trap #971 measured.
pub(crate) fn subject_sensitivities_iov(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    stacked_eta: &[f64],
) -> Option<SubjectSens> {
    let dim = model.n_theta + stacked_eta.len();
    macro_rules! arms {
        ($($m:literal),+ $(,)?) => {
            match crate::sens::widths::bucket_for(dim, &crate::sens::ode_provider::ODE_IOV_WIDTH_BUCKETS)? {
                $($m => run_obs_iov::<$m>(model, subject, theta, stacked_eta),)+
                _ => None,
            }
        };
    }
    arms!(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 28,
        32, 40, 48, 56, 64, 80, 96
    )
}

/// Inner IOV entry point: `∂f/∂(stacked η)` per observation.
pub(crate) fn subject_eta_grad_iov(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    stacked_eta: &[f64],
) -> Option<Vec<ObsGrad>> {
    let dim = stacked_eta.len();
    macro_rules! arms {
        ($($n:literal),+ $(,)?) => {
            match crate::sens::widths::bucket_for(dim, &crate::sens::ode_provider::ODE_IOV_WIDTH_BUCKETS)? {
                $($n => run_obs_grad_iov::<$n>(model, subject, theta, stacked_eta),)+
                _ => None,
            }
        };
    }
    arms!(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 28,
        32, 40, 48, 56, 64, 80, 96
    )
}

// Both IOV ladders above enumerate `ODE_IOV_WIDTH_BUCKETS` literally, because
// `macro_rules!` cannot iterate a const. If the ladder is retuned there without
// editing both arm lists here, an in-scope subject would hit `_ => None` and fall
// back to FD silently — the failure this assert exists to turn into a compile error.
const _: () = assert!(
    crate::sens::widths::buckets_well_formed(
        &crate::sens::ode_provider::ODE_IOV_WIDTH_BUCKETS,
        MAX_ALGEBRAIC_IOV_AXES
    ),
    "the algebraic IOV ladders are written against ODE_IOV_WIDTH_BUCKETS; keep the \
     literal arm lists in step with it"
);

#[cfg(test)]
#[path = "algebraic_tests.rs"]
mod tests;
