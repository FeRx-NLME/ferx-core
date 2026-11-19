//! Fisher-scoring solve of the numerical θ/σ M-step (`mstep_optimizer = newton`).
//!
//! The M-step minimises a sum of per-observation Gaussian negative log-likelihoods at frozen
//! (SAEM) or importance-weighted (IMP/IMPMAP) random effects. For that objective the exact
//! score and the expected information
//! `I = Σⱼ [ ∇fⱼ∇fⱼᵀ / Vⱼ + ½ ∇Vⱼ∇Vⱼᵀ / Vⱼ² ]`
//! come out of the *same* adjoint recording
//! ([`obs_nll_subject_grad_fisher`](crate::estimation::fixed_eta_gradient::obs_nll_subject_grad_fisher)),
//! so one pass returns value, gradient and a positive-semidefinite curvature. The solve is a
//! damped Gauss–Newton (Fisher-scoring) iteration on them, warm-started at the previous
//! M-step's point: where BOBYQA spends `n + 1` value passes to build its first interpolation
//! model and more to move it, this takes a handful of fused passes.
//!
//! Same problem as the derivative-free solve: same packed space, same bounds, same
//! evaluation budget, and the same stopping rule — NLopt's, `|Δf| < ftol_abs ∨ |Δf| < ftol_rel ·
//! mean(|f|, |f'|)` ([`ftol_threshold`]), applied to the *accepted* step and, before taking one,
//! to the decrease the quadratic model predicts for it.

use crate::estimation::saem::fold_nll_grad;
use crate::stats::likelihood::SUBJECT_SENTINEL_NLL;
use nalgebra::{Cholesky, DMatrix, DVector};

/// One fused evaluation: objective value, packed gradient and row-major `n × n` information.
pub(crate) struct ValueGradInfo {
    pub val: f64,
    pub grad: Vec<f64>,
    pub info: Vec<f64>,
}

/// Fold per-subject `(value, gradient, information)` triples in subject order, with the
/// information when `want_info`. `None` when `want_info` and any subject has none (outside
/// the closed form's scope): the caller then keeps the derivative-free solve rather than
/// stepping on a partial curvature. Without `want_info` the information is ignored and the
/// result's is empty. The value and gradient fold is SAEM's [`fold_nll_grad`], the one
/// subject-ordered reduction of the numerical M-steps (#703).
pub(crate) fn fold_value_grad_info(
    per_subj: Vec<(f64, Vec<f64>, Option<Vec<f64>>)>, n: usize, want_info: bool,
) -> Option<ValueGradInfo> {
    let mut info = vec![0.0; if want_info { n * n } else { 0 }];
    let mut pairs = Vec::with_capacity(per_subj.len());
    for (v, g, i) in per_subj {
        if want_info {
            for (a, b) in info.iter_mut().zip(&i?) {
                *a += b;
            }
        }
        pairs.push((v, g));
    }
    let (val, grad) = fold_nll_grad(pairs, n);
    Some(ValueGradInfo { val, grad, info })
}

/// What an optimizer is handed for a value that is not finite: the crate's sentinel, a flat,
/// maximally bad point ([`SUBJECT_SENTINEL_NLL`]).
pub(crate) fn finite_value(val: f64) -> f64 {
    if val.is_finite() {
        val
    } else {
        SUBJECT_SENTINEL_NLL
    }
}

/// A non-finite gradient entry is zeroed, as the value's sentinel is flat.
pub(crate) fn finite_grad(grad: &mut [f64]) {
    for g in grad.iter_mut().filter(|g| !g.is_finite()) {
        *g = 0.0;
    }
}

/// What NLopt receives from a numerical M-step objective that was asked for its gradient: the
/// gradient written to `out` with [`finite_grad`] applied, and the value through
/// [`finite_value`]. The one cleanup SAEM's and IMP/IMPMAP's solves share.
pub(crate) fn nlopt_value_and_grad((val, grad): (f64, Vec<f64>), out: &mut [f64]) -> f64 {
    for (o, g) in out.iter_mut().zip(grad) {
        *o = g;
    }
    finite_grad(out);
    finite_value(val)
}

/// The decrease below which a step from objective `f_old` to `f_new` is not worth taking, for
/// the relative tolerance `ftol_rel` (also the absolute one): NLopt's `relstop` with
/// `ftol_abs = ftol_rel`, `ftol_rel · max(mean(|f_old|, |f_new|), 1)`. The `max` is the absolute
/// tolerance's floor — where the objective is near zero or passes through it, a purely
/// relative threshold collapses to nothing and the solve spends its budget on steps that
/// change nothing it can resolve. BOBYQA gets the same rule from NLopt (`set_ftol_abs`).
pub(crate) fn ftol_threshold(ftol_rel: f64, f_old: f64, f_new: f64) -> f64 {
    ftol_rel * (0.5 * (f_old.abs() + f_new.abs())).max(1.0)
}

/// Armijo sufficient-decrease constant of the backtracking line search (the textbook `1e-4`).
const ARMIJO_C1: f64 = 1e-4;
/// First nonzero Levenberg ridge, relative to the largest diagonal entry of the information.
const RIDGE_FLOOR: f64 = 1e-9;
/// Dampings tried before the iteration gives up and keeps its current point.
const MAX_DAMPINGS: u32 = 30;

/// Solve `(A + ridge·I) s = rhs` by Cholesky; `None` when not positive definite.
fn cholesky_solve(a: &[f64], m: usize, ridge: f64, rhs: &[f64]) -> Option<Vec<f64>> {
    let mut mat = DMatrix::from_row_slice(m, m, a);
    for i in 0..m {
        mat[(i, i)] += ridge;
    }
    let chol = Cholesky::new(mat)?;
    Some(chol.solve(&DVector::from_column_slice(rhs)).as_slice().to_vec())
}

#[cfg(test)]
thread_local! {
    /// Fused passes [`fisher_scoring`] still gets on this thread (`None`: no limit): past them
    /// the information is out of scope, so a driver test can make a solve decline at its start
    /// (`0`) or lose its information after one accepted step (`1`) — a stall no model reaches
    /// on demand.
    pub(crate) static FUSED_PASSES_ALLOWED: std::cell::Cell<Option<u32>> =
        const { std::cell::Cell::new(None) };
}

/// One fused evaluation, or its absence where [`FUSED_PASSES_ALLOWED`] is spent (test builds).
fn run_fused<F>(fused: &mut F, x: &[f64]) -> Option<ValueGradInfo>
where
    F: FnMut(&[f64]) -> Option<ValueGradInfo>,
{
    #[cfg(test)]
    if let Some(left) = FUSED_PASSES_ALLOWED.get() {
        if left == 0 {
            return None;
        }
        FUSED_PASSES_ALLOWED.set(Some(left - 1));
    }
    fused(x)
}

/// How [`fisher_scoring`] ended.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum NewtonOutcome {
    /// The information was out of scope at the start point (a subject the adjoint declined
    /// there): `x` is untouched and nothing was solved.
    Declined,
    /// The objective at the start point is the sentinel (a subject's prediction is not finite
    /// there): `x` is untouched and nothing was solved. The adjoint declined nothing; the
    /// derivative-free solve, which looks for a finite point, takes it from here.
    NonFinite,
    /// The iteration reached one of its own stopping rules — the tolerance (the step it would
    /// take is not worth a pass, or the one it took changed the objective by less than
    /// `ftol_rel`), every coordinate held at a bound, or the evaluation budget. `x` is the
    /// answer.
    Converged,
    /// No admissible step was left although the tolerance was not met — the information
    /// would not factor at any ridge (non-finite entries), or no damped step decreased the
    /// objective. `x` is the last accepted point (the start when none was); the caller
    /// continues from it with another solver.
    Stalled,
    /// An accepted point had no information (a subject the adjoint declined there), so the
    /// next step could not be taken. `x` is the last accepted point; the caller continues from
    /// it with another solver, and a fit says how many of its solves ended here.
    Lost,
}

/// Minimise an objective over the box `[lower, upper]` from `x` (updated in place) by Fisher
/// scoring. Returns how it ended and the evaluations it spent (see below).
///
/// `fused(x)` returns the value, gradient and information in one pass, or `None` where the
/// information is out of scope; `value(x)` is the same objective's value alone (a trial point
/// is judged on it, and only an accepted point that the iteration continues from pays for the
/// derivatives). Returns [`NewtonOutcome::Declined`] — `x` untouched, no evaluation spent —
/// when the information is out of scope at the start point, and [`NewtonOutcome::Stalled`]
/// when the iteration cannot take a step it has not converged for ([`NewtonOutcome::NonFinite`]
/// at a start whose objective is the sentinel); in each the caller keeps the derivative-free
/// solve (from `x`). `maxeval` bounds the evaluations of either kind (the
/// budget BOBYQA gets in value evaluations); **every pass counts as one evaluation, a value
/// pass and a fused pass alike**, and the count returned is what the caller subtracts from its
/// budget before handing the remainder to the solve it continues with, so one M-step never
/// spends more than `maxeval` in all.
///
/// A coordinate with `lower == upper`, or at a bound with the gradient pushing outward, is held
/// for that step; the step solves the remaining block of `I` (Levenberg-damped as far as needed
/// to factor it and to decrease the objective) and a trial point is clamped into the box.
///
/// The tolerance is tested on the *undamped* step. The Levenberg ridge is carried over from
/// earlier rejections, so a step at the carried ridge is shorter than the Fisher step and its
/// predicted (and realised) decrease understates what the point still has to give; a solve is
/// converged only when the plain Fisher step — or, where `I` does not factor, the one at the
/// smallest ridge that does — predicts a decrease inside `ftol_rel`, and an accepted step that
/// was damped never ends the solve on its own small decrease.
pub(crate) fn fisher_scoring<V, F>(
    x: &mut [f64], lower: &[f64], upper: &[f64], maxeval: u32, ftol_rel: f64, mut value: V,
    mut fused: F,
) -> (NewtonOutcome, u32)
where
    V: FnMut(&[f64]) -> f64,
    F: FnMut(&[f64]) -> Option<ValueGradInfo>,
{
    let n = x.len();
    let Some(mut cur) = run_fused(&mut fused, x) else {
        return (NewtonOutcome::Declined, 0);
    };
    let mut evals = 1u32;
    // A start whose objective is the sentinel (a subject's prediction is not finite there, as at
    // a start where a covariate-adjusted typical value is negative) has a flat value and
    // zeroed derivatives ([`finite_value`], [`finite_grad`]): every step predicts no decrease,
    // which would read as converged. It is not a solve of anything — the derivative-free
    // solve, which looks for a finite point, takes it from here.
    if cur.val >= SUBJECT_SENTINEL_NLL {
        return (NewtonOutcome::NonFinite, evals);
    }
    // Persists across iterations (Levenberg-Marquardt): a rejected step raises it, an accepted
    // one lowers it, and below the floor it is exactly zero — the plain Fisher step.
    let mut ridge = 0.0f64;
    let mut stalled = false;
    let mut lost = false;
    // Whether the objective has confirmed a step of this solve. Until it has, the quadratic
    // model's prediction alone does not end the solve (see `confirm` below).
    let mut stepped = false;
    while evals < maxeval {
        // The model predicts no worthwhile decrease at a start the objective has not yet
        // vouched for: the solve takes the one undamped Fisher step and lets the objective
        // decide, as BOBYQA's first design always evaluates. Where the objective is dominated
        // by a constant the model cannot see — a start at which a subject is scored by the
        // invalid-variance guard, 1.2e13 of it — the relative tolerance is the constant's, and
        // the prediction (0.08 there) is far inside it while the objective still has a smooth
        // part to give. A start that really is converged pays one value pass for it.
        let mut confirm = false;
        let free: Vec<usize> = (0..n)
            .filter(|&i| {
                lower[i] < upper[i]
                    && !(x[i] <= lower[i] && cur.grad[i] > 0.0)
                    && !(x[i] >= upper[i] && cur.grad[i] < 0.0)
            })
            .collect();
        let m = free.len();
        if m == 0 {
            break;
        }
        let g: Vec<f64> = free.iter().map(|&i| cur.grad[i]).collect();
        let mut a = vec![0.0f64; m * m];
        for (p, &i) in free.iter().enumerate() {
            for (q, &j) in free.iter().enumerate() {
                a[p * m + q] = cur.info[i * n + j];
            }
        }
        let max_diag = (0..m).map(|p| a[p * m + p]).fold(0.0f64, f64::max).max(f64::MIN_POSITIVE);
        let neg_g: Vec<f64> = g.iter().map(|v| -v).collect();
        // The decrease the quadratic model predicts for the undamped step, ½ gᵀA⁻¹g (at the
        // smallest ridge that factors, where A does not), against the tolerance: below it the
        // point is not worth a pass — converged, not stalled.
        let mut probe = 0.0f64;
        for _ in 0..MAX_DAMPINGS {
            if let Some(s) = cholesky_solve(&a, m, probe * max_diag, &neg_g) {
                let predicted = -0.5 * g.iter().zip(&s).map(|(a, b)| a * b).sum::<f64>();
                if !(predicted > ftol_threshold(ftol_rel, cur.val, cur.val - predicted)) {
                    if stepped {
                        return (NewtonOutcome::Converged, evals);
                    }
                    confirm = true;
                }
                break;
            }
            probe = (probe * 10.0).max(RIDGE_FLOOR);
        }
        // Levenberg damping: the carried ridge first (zero: the plain Fisher step); a block
        // that does not factor, or a step the objective rejects, multiplies the ridge
        // (relative to the largest diagonal entry) by ten from `RIDGE_FLOOR` — a larger ridge
        // is a shorter, more gradient-like step, so the sequence ends in a decrease.
        let mut accepted = None;
        let mut damped = false;
        for _ in 0..MAX_DAMPINGS {
            if evals >= maxeval {
                break;
            }
            let Some(s) = cholesky_solve(&a, m, ridge * max_diag, &neg_g) else {
                ridge = (ridge * 10.0).max(RIDGE_FLOOR);
                continue;
            };
            let mut trial = x.to_vec();
            for (p, &i) in free.iter().enumerate() {
                trial[i] = (x[i] + s[p]).clamp(lower[i], upper[i]);
            }
            if confirm && trial == x {
                // Stationary to the bit: nothing to confirm.
                return (NewtonOutcome::Converged, evals);
            }
            // The Armijo test needs a descent direction. The trial is the Fisher step clamped
            // into the box, and a clamped coordinate's shortened move can leave
            // `gᵀ(trial − x) ≥ 0` (a coordinate with a large step against its gradient
            // clamped to a small one, the others still moving uphill): then the sufficient-
            // decrease bound would admit an *increase* of the objective. That trial is not
            // judged — it costs no pass — and the ridge climbs toward the projected gradient
            // step, which is a descent direction (every free coordinate moves against its own
            // gradient, shortened or not).
            let slope: f64 = (0..n).map(|i| cur.grad[i] * (trial[i] - x[i])).sum();
            if slope >= 0.0 || slope.is_nan() {
                ridge = (ridge * 10.0).max(RIDGE_FLOOR);
                continue;
            }
            let v = value(&trial);
            evals += 1;
            if v.is_finite() && v <= cur.val + ARMIJO_C1 * slope {
                accepted = Some((trial, v));
                damped = ridge > 0.0;
                break;
            }
            if confirm {
                // The model predicted no gain and the objective agrees (the step is no
                // decrease): converged where it started.
                return (NewtonOutcome::Converged, evals);
            }
            ridge = (ridge * 10.0).max(RIDGE_FLOOR);
        }
        let Some((trial, v)) = accepted else {
            // Every damping tried with the budget left is a stall; the budget ending is an
            // ending.
            stalled = evals < maxeval;
            break;
        };
        // A damping so heavy that the step vanishes in rounding is "accepted" at the same
        // point: no progress, which is a stall.
        if trial == x {
            stalled = true;
            break;
        }
        ridge = if ridge * 0.1 < RIDGE_FLOOR { 0.0 } else { ridge * 0.1 };
        let decrease = cur.val - v;
        stepped = true;
        x.copy_from_slice(&trial);
        // Inside the tolerance on a plain Fisher step: this is the answer, and its
        // derivatives are not needed. A damped step's small decrease says how short the step
        // was, not how little is left; the undamped test at the next point decides.
        if !damped && decrease < ftol_threshold(ftol_rel, cur.val, v) {
            break;
        }
        // Another step from here needs the derivatives at the accepted point — and the budget
        // for the pass.
        if evals >= maxeval {
            break;
        }
        evals += 1;
        match run_fused(&mut fused, x) {
            Some(nx) => cur = nx,
            None => {
                lost = true;
                break;
            }
        }
    }
    let outcome = if lost {
        NewtonOutcome::Lost
    } else if stalled {
        NewtonOutcome::Stalled
    } else {
        NewtonOutcome::Converged
    };
    (outcome, evals)
}

/// What a fit's Newton solves did, for the warning the fit reports (the solves run inside
/// the M-steps, which return only their point). Counted by [`newton_solve`].
#[derive(Default)]
pub(crate) struct NewtonTally {
    /// Solves that ran Newton.
    solves: std::cell::Cell<u32>,
    /// Solves that lost the information — a subject whose recording the adjoint declined at the
    /// start or at an accepted point — and continued with the derivative-free solve.
    lost_information: std::cell::Cell<u32>,
    /// Solves that started where the objective is not finite (a subject's prediction is not
    /// finite there) and went to the derivative-free solve, which looks for a finite point.
    non_finite_start: std::cell::Cell<u32>,
    /// Solves that found no admissible step and continued with the derivative-free solve.
    stalled: std::cell::Cell<u32>,
}

impl NewtonTally {
    /// The fit's warning, `None` when every Newton solve ended on its own.
    pub(crate) fn warning(&self, label: &str) -> Option<String> {
        let (solves, lost, non_finite, stalled) = (
            self.solves.get(),
            self.lost_information.get(),
            self.non_finite_start.get(),
            self.stalled.get(),
        );
        if lost + non_finite + stalled == 0 {
            return None;
        }
        let mut causes = Vec::new();
        if lost > 0 {
            causes.push(format!(
                "{lost} lost the expected information at run time (the adjoint declined a \
                 subject's recording at that point)"
            ));
        }
        if non_finite > 0 {
            causes.push(format!(
                "{non_finite} started where the objective is not finite (a subject's \
                 prediction at that point is not finite)"
            ));
        }
        if stalled > 0 {
            causes.push(format!(
                "{stalled} found no admissible damped step before the tolerance was met"
            ));
        }
        Some(format!(
            "{label}: `mstep_optimizer = newton` handed {} of {solves} numerical M-steps to \
             `bobyqa` (the remainder of the M-step's evaluation budget, from the point Newton \
             reached): {}.",
            lost + non_finite + stalled,
            causes.join("; ")
        ))
    }
}

/// What [`newton_solve`] left for the caller.
pub(crate) struct NewtonSolve {
    /// The solve reached one of its own stopping rules: `x` is the M-step's answer.
    pub converged: bool,
    /// Evaluations spent (see [`fisher_scoring`]); `maxeval - evals` is what the derivative-free
    /// continuation of a solve that did not converge may still use.
    pub evals: u32,
}

/// [`fisher_scoring`] for an M-step, with the run-time information loss and stalls recorded in
/// `tally`: the one place SAEM's and IMP/IMPMAP's solves share.
#[allow(clippy::too_many_arguments)]
pub(crate) fn newton_solve<V, F>(
    x: &mut [f64], lower: &[f64], upper: &[f64], maxeval: u32, ftol_rel: f64, value: V, fused: F,
    tally: Option<&NewtonTally>,
) -> NewtonSolve
where
    V: FnMut(&[f64]) -> f64,
    F: FnMut(&[f64]) -> Option<ValueGradInfo>,
{
    let (outcome, evals) = fisher_scoring(x, lower, upper, maxeval, ftol_rel, value, fused);
    if let Some(t) = tally {
        t.solves.set(t.solves.get() + 1);
        if matches!(outcome, NewtonOutcome::Declined | NewtonOutcome::Lost) {
            t.lost_information.set(t.lost_information.get() + 1);
        } else if outcome == NewtonOutcome::NonFinite {
            t.non_finite_start.set(t.non_finite_start.get() + 1);
        } else if outcome == NewtonOutcome::Stalled {
            t.stalled.set(t.stalled.get() + 1);
        }
    }
    NewtonSolve { converged: outcome == NewtonOutcome::Converged, evals }
}

#[cfg(test)]
mod tests {
    use super::NewtonOutcome::{Converged, Declined, Lost, NonFinite, Stalled};
    use super::*;

    /// Gaussian regression `y_j = a·exp(b·t_j)`, additive σ fixed at 1: value, exact score and
    /// the Gauss-Newton information `JᵀJ` of `½ Σ (y - f)²`.
    fn exp_fit(x: &[f64]) -> Option<ValueGradInfo> {
        let ts = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0];
        let truth = (2.0f64, -0.7f64);
        let (a, b) = (x[0], x[1]);
        let mut out = ValueGradInfo { val: 0.0, grad: vec![0.0; 2], info: vec![0.0; 4] };
        for t in ts {
            let y = truth.0 * (truth.1 * t).exp();
            let f = a * (b * t).exp();
            let j = [(b * t).exp(), a * t * (b * t).exp()];
            let r = y - f;
            out.val += 0.5 * r * r;
            for p in 0..2 {
                out.grad[p] -= r * j[p];
                for q in 0..2 {
                    out.info[p * 2 + q] += j[p] * j[q];
                }
            }
        }
        Some(out)
    }

    /// Run the solver, counting evaluations of either kind.
    fn solve<F>(
        x: &mut [f64], lo: &[f64], hi: &[f64], maxeval: u32, ftol: f64, f: F,
    ) -> (NewtonOutcome, u32)
    where
        F: Fn(&[f64]) -> Option<ValueGradInfo>,
    {
        let calls = std::cell::Cell::new(0u32);
        let (ok, evals) = fisher_scoring(
            x,
            lo,
            hi,
            maxeval,
            ftol,
            |v| {
                calls.set(calls.get() + 1);
                f(v).unwrap().val
            },
            |v| {
                calls.set(calls.get() + 1);
                f(v)
            },
        );
        assert_eq!(evals, calls.get(), "the evaluations reported are the passes made");
        (ok, calls.get())
    }

    #[test]
    fn fisher_scoring_finds_the_zero_residual_optimum_in_a_few_passes() {
        let mut x = [1.5, -0.4];
        let (solved, calls) = solve(&mut x, &[-9.0; 2], &[9.0; 2], 100, 1e-12, exp_fit);
        assert_eq!(solved, Converged);
        assert!((x[0] - 2.0).abs() < 1e-5 && (x[1] + 0.7).abs() < 1e-5, "{x:?}");
        assert!(calls < 30, "{calls} passes");
    }

    #[test]
    fn fisher_scoring_respects_bounds_and_pinned_coordinates() {
        // Optimum a = 2 lies above the bound 1.8: the solve stops on the bound, b free.
        let mut x = [1.5, -0.4];
        assert_eq!(solve(&mut x, &[-9.0, -9.0], &[1.8, 9.0], 100, 1e-12, exp_fit).0, Converged);
        assert_eq!(x[0], 1.8);
        // A pinned coordinate never moves.
        let mut y = [1.5, -0.4];
        assert_eq!(solve(&mut y, &[1.5, -9.0], &[1.5, 9.0], 100, 1e-12, exp_fit).0, Converged);
        assert_eq!(y[0], 1.5);
        assert!(y[1] != -0.4);
    }

    #[test]
    fn fisher_scoring_declines_without_information_and_honours_the_budget() {
        let mut x = [1.5, -0.4];
        assert_eq!(
            fisher_scoring(&mut x, &[-9.0; 2], &[9.0; 2], 10, 1e-9, |_| 0.0, |_| None),
            (Declined, 0)
        );
        assert_eq!(x, [1.5, -0.4]);
        let mut z = [1.5, -0.4];
        let (_, calls) = solve(&mut z, &[-9.0; 2], &[9.0; 2], 3, 1e-15, exp_fit);
        assert!(calls <= 4, "{calls}");
    }

    /// A start at the sentinel objective is declined, not converged: its zeroed derivatives
    /// predict no decrease, which `fisher_scoring` would otherwise read as a solved M-step with
    /// `x` untouched (#3001). Mutation: delete the sentinel check and the first assertion fails.
    #[test]
    fn a_start_at_the_sentinel_objective_is_declined_not_converged() {
        let mut x = [1.5, -0.4];
        let sentinel = |_: &[f64]| {
            Some(ValueGradInfo {
                val: SUBJECT_SENTINEL_NLL,
                grad: vec![0.0; 2],
                info: vec![0.0; 4],
            })
        };
        let (outcome, _) = solve(&mut x, &[-9.0; 2], &[9.0; 2], 100, 1e-9, sentinel);
        assert_eq!(outcome, NonFinite);
        assert_eq!(x, [1.5, -0.4]);
        // The same flat information at a finite value *is* converged: the sentinel is the
        // discriminator, not the zero gradient.
        let flat =
            |_: &[f64]| Some(ValueGradInfo { val: 5.0, grad: vec![0.0; 2], info: vec![0.0; 4] });
        let mut y = [1.5, -0.4];
        assert_eq!(solve(&mut y, &[-9.0; 2], &[9.0; 2], 100, 1e-9, flat).0, Converged);
    }

    /// The tolerance has an absolute floor: where the objective is near zero a purely relative
    /// threshold (`ftol_rel · |f|`) collapses with it and the solve takes steps it cannot
    /// resolve until the budget ends. `f = x⁴` (minimum 0) under an information of 10 moves by
    /// damped steepest descent, a predicted decrease of `0.8x⁶` per step against `f = x⁴`: the
    /// relative rule is not met before `x ≈ 3.5e-4` (about 1e7 steps), the floored rule at
    /// `x ≈ 0.07` (about 250). The solve stops on the floor inside a budget the relative rule
    /// would spend whole.
    #[test]
    fn an_objective_near_zero_stops_on_the_absolute_floor() {
        let quartic = |v: &[f64]| {
            Some(ValueGradInfo {
                val: v[0].powi(4),
                grad: vec![4.0 * v[0].powi(3)],
                info: vec![10.0],
            })
        };
        let mut x = [1.0];
        let (out, calls) = solve(&mut x, &[-9.0], &[9.0], 5000, 1e-7, quartic);
        assert_eq!(out, Converged);
        assert!(
            calls < 1500 && x[0].abs() < 0.2,
            "{calls} passes, x = {x:?}: the floor ends the solve, not the budget"
        );
        // The threshold itself: NLopt's `relstop` with `ftol_abs = ftol_rel`.
        assert_eq!(ftol_threshold(1e-7, 0.0, 0.0), 1e-7);
        assert_eq!(ftol_threshold(1e-7, 4e3, 2e3), 1e-7 * 3e3);
        assert_eq!(ftol_threshold(1e-7, -4e3, 2e3), 1e-7 * 3e3, "across zero: sizes, not signs");
    }

    /// A start the objective has not vouched for is not converged on the model's word alone.
    /// The objective is `1e13 + ½(x − 3)²`: the constant is all the relative tolerance sees
    /// (`1e-7 · 1e13 = 1e6` against a predicted decrease of 0.005), the way a start scored by
    /// the invalid-variance guard is dominated by its 1e13 while the smooth part is small. The
    /// solve takes its one Fisher step and the objective confirms it (the answer is `x = 3`,
    /// two passes); a start that already is the optimum is stationary to the bit and costs one;
    /// and where the objective refuses the step the solve is converged where it started.
    #[test]
    fn a_start_the_objective_has_not_vouched_for_takes_one_step() {
        let plateau = |v: &[f64]| {
            Some(ValueGradInfo {
                val: 1e13 + 0.5 * (v[0] - 3.0).powi(2),
                grad: vec![v[0] - 3.0],
                info: vec![1.0],
            })
        };
        let mut x = [2.9];
        let (out, calls) = solve(&mut x, &[-9.0], &[9.0], 100, 1e-7, plateau);
        assert_eq!(out, Converged);
        assert_eq!(calls, 2, "fused at the start, one value at the step");
        assert!((x[0] - 3.0).abs() < 1e-12, "the step was taken: {x:?}");
        let mut at = [3.0];
        let (out, calls) = solve(&mut at, &[-9.0], &[9.0], 100, 1e-7, plateau);
        assert_eq!((out, calls, at), (Converged, 1, [3.0]), "stationary: nothing to confirm");
        // The objective disagrees with the model: the step is uphill, the solve stays put.
        let mut y = [2.9];
        let (out, calls) = fisher_scoring(
            &mut y,
            &[-9.0],
            &[9.0],
            100,
            1e-7,
            |v| 1e13 + 1.0 + (v[0] - 2.9).abs(),
            |_| Some(ValueGradInfo { val: 1e13, grad: vec![1e-3], info: vec![1.0] }),
        );
        assert_eq!((out, calls, y), (Converged, 2, [2.9]));
    }

    #[test]
    fn a_solve_already_inside_the_tolerance_takes_no_step() {
        let mut x = [2.0, -0.7];
        let (_, calls) = solve(&mut x, &[-9.0; 2], &[9.0; 2], 100, 1e-7, exp_fit);
        assert_eq!((calls, x), (1, [2.0, -0.7]));
    }

    #[test]
    fn a_step_inside_the_tolerance_costs_one_fused_pass_and_one_value() {
        // `½(x-3)²` with the curvature over-stated (0.8 for 1): the step from 2.9 overshoots
        // to 3.025, a decrease of 0.0046875 against a predicted 0.00625. With `ftol_rel`
        // 0.0047 the step is worth taking (prediction above the tolerance) and, once taken,
        // is inside it: the answer, without the derivatives at the new point.
        let quad = |v: &[f64]| {
            Some(ValueGradInfo {
                val: 1.0 + 0.5 * (v[0] - 3.0).powi(2),
                grad: vec![v[0] - 3.0],
                info: vec![0.8],
            })
        };
        let mut x = [2.9];
        let (_, calls) = solve(&mut x, &[-9.0], &[9.0], 100, 0.0047, quad);
        assert_eq!(calls, 2, "fused at the start + one value at the trial");
        assert!((x[0] - 3.025).abs() < 1e-12, "{x:?}");
        // Outside the tolerance the accepted point's derivatives are taken and the iteration goes on.
        let mut y = [2.9];
        let (_, calls) = solve(&mut y, &[-9.0], &[9.0], 100, 1e-9, quad);
        assert!(calls > 2 && (y[0] - 3.0).abs() < 1e-4, "{calls} {y:?}");
    }

    #[test]
    fn an_uphill_trial_is_backtracked() {
        // A curvature 100x too small proposes a step far past the minimum of ½x²; the damping
        // brings it back to a decrease.
        let mut x = [10.0];
        let (solved, _) = solve(&mut x, &[-1e3], &[1e3], 200, 1e-12, |v| {
            Some(ValueGradInfo { val: 0.5 * v[0] * v[0], grad: vec![v[0]], info: vec![0.01] })
        });
        assert!(solved == Converged && x[0].abs() < 1e-3, "{x:?}");
    }

    #[test]
    fn a_singular_information_block_is_damped_not_fatal() {
        // The second coordinate has gradient but no curvature: the ridge makes it factor.
        let mut x = [1.0, 1.0];
        let (solved, _) = solve(&mut x, &[-9.0; 2], &[9.0; 2], 100, 1e-12, |v| {
            Some(ValueGradInfo {
                val: 0.5 * v[0] * v[0] + 0.5 * v[1] * v[1],
                grad: v.to_vec(),
                info: vec![1.0, 0.0, 0.0, 0.0],
            })
        });
        assert!(solved == Converged && x[0].abs() < 1e-6 && x[1].abs() < 1.0, "{x:?}");
    }

    /// A solve that cannot take a step it has not converged for is `Stalled`, never `Converged`,
    /// and says where it stopped: information that does not factor at any ridge (non-finite
    /// entries) at the start, every damped step uphill, and a later point that lost its
    /// information. Each branch's own assertion carries the discriminator the others lack:
    /// `x` untouched for the first two, `x` at the accepted point for the third.
    #[test]
    fn a_solve_with_no_admissible_step_is_stalled_not_converged() {
        // ½(x-3)² with an information that is NaN: no ridge makes it factor.
        let nan_info = |v: &[f64]| {
            Some(ValueGradInfo {
                val: 0.5 * (v[0] - 3.0).powi(2),
                grad: vec![v[0] - 3.0],
                info: vec![f64::NAN],
            })
        };
        let mut x = [2.9];
        let (out, calls) = solve(&mut x, &[-9.0], &[9.0], 100, 1e-12, nan_info);
        assert_eq!((out, x), (Stalled, [2.9]));
        assert_eq!(calls, 1, "no value pass: nothing factored");
        // Factors, but the objective's value is uphill at every trial: all dampings rejected.
        let uphill = |v: &[f64]| {
            Some(ValueGradInfo {
                val: 1.0 + 0.5 * (v[0] - 3.0).powi(2) + (v[0] - 2.9).abs() * 10.0,
                grad: vec![v[0] - 3.0],
                info: vec![1.0],
            })
        };
        let mut y = [2.9];
        let (out, calls) = solve(&mut y, &[-9.0], &[9.0], 1000, 1e-12, uphill);
        assert_eq!((out, y), (Stalled, [2.9]));
        assert!(calls > 2, "the dampings were tried: {calls}");
        // A later point without information: the accepted point is kept, and it is a stall.
        let lost = |v: &[f64]| {
            (v[0] == 2.9).then(|| ValueGradInfo {
                val: 1.0 + 0.5 * (v[0] - 3.0).powi(2),
                grad: vec![v[0] - 3.0],
                info: vec![1.0],
            })
        };
        let mut z = [2.9];
        let (out, _) = fisher_scoring(
            &mut z,
            &[-9.0],
            &[9.0],
            100,
            1e-12,
            |v| 1.0 + 0.5 * (v[0] - 3.0).powi(2),
            lost,
        );
        assert!(out == Lost && (z[0] - 3.0).abs() < 1e-12, "{out:?} {z:?}");
    }

    /// The endings that are not stalls: a first step not worth a pass, every coordinate held at
    /// a bound, and an exhausted budget are all `Converged`.
    #[test]
    fn a_solve_that_reached_a_stopping_rule_is_converged() {
        let mut at_opt = [2.0, -0.7];
        assert_eq!(solve(&mut at_opt, &[-9.0; 2], &[9.0; 2], 100, 1e-7, exp_fit).0, Converged);
        // Held at the lower bound with the gradient pushing outward: no free coordinate.
        let mut held = [1.0];
        let up = |v: &[f64]| Some(ValueGradInfo { val: v[0], grad: vec![1.0], info: vec![1.0] });
        assert_eq!(solve(&mut held, &[1.0], &[9.0], 100, 1e-12, up).0, Converged);
        assert_eq!(held, [1.0]);
        let mut budget = [1.5, -0.4];
        assert_eq!(solve(&mut budget, &[-9.0; 2], &[9.0; 2], 3, 1e-15, exp_fit).0, Converged);
    }

    /// `f = 1 + ½ a (x - x*)²` with an information `κ` that understates the curvature `a`: the
    /// plain Fisher step overshoots, so the Levenberg ridge climbs until a step descends and
    /// the *accepted* step is damped.
    fn understated(a: f64, x_star: f64) -> impl Fn(&[f64]) -> Option<ValueGradInfo> {
        move |v: &[f64]| {
            Some(ValueGradInfo {
                val: 1.0 + 0.5 * a * (v[0] - x_star).powi(2),
                grad: vec![a * (v[0] - x_star)],
                info: vec![1.0],
            })
        }
    }

    /// The tolerance is tested on the undamped step. With the information 100x too small the
    /// ridge settles near 10-100 and is carried into every iteration: at `x = 3.1e-4` the
    /// carried-ridge step predicts `4.4e-5`, inside `ftol_rel = 1e-4`, while the plain Fisher
    /// step predicts `4.9e-4`, outside it — that point is not converged. Measured on this
    /// fixture: the carried-ridge test stopped at `3.13e-4`; the undamped test ends at `3.1e-6`.
    #[test]
    fn a_ridge_carried_over_does_not_declare_convergence() {
        let f = understated(100.0, 0.0);
        let mut x = [0.0316];
        let (out, _) = solve(&mut x, &[-9.0], &[9.0], 1000, 1e-4, f);
        assert_eq!(out, Converged);
        assert!(x[0].abs() < 1e-4, "stopped at {x:?}, a point still worth a pass");
        // Non-vacuity: it did take the damped steps (the start was 316x further out).
        assert!(x[0].abs() < 0.0316 / 10.0, "{x:?}");
    }

    /// A damped accepted step's small decrease says how short the step was, not how little is
    /// left. The first 12 trials are rejected, so the step that is accepted carries ridge
    /// `1e2` and decreases `f` by `1e-4`, inside `ftol_rel = 1e-3`; the point it lands on (`x =
    /// 0.099` of a quadratic whose plain step would predict `4.9e-3`) is far from converged.
    #[test]
    fn a_damped_accepted_step_does_not_end_the_solve_on_its_small_decrease() {
        let f = understated(1.0, 0.0);
        let rejected = std::cell::Cell::new(0u32);
        let mut x = [0.1];
        let (out, _) = fisher_scoring(
            &mut x,
            &[-9.0],
            &[9.0],
            1000,
            1e-3,
            |v| {
                rejected.set(rejected.get() + 1);
                if rejected.get() <= 12 {
                    f64::INFINITY
                } else {
                    f(v).unwrap().val
                }
            },
            &f,
        );
        assert_eq!(out, Converged);
        assert!(rejected.get() > 13, "the damped step was taken and not the last");
        assert!(x[0].abs() < 0.05, "stopped at {x:?}, far from converged");
    }

    /// One budget: every pass is one evaluation, the count is returned, and the solve never
    /// goes past `maxeval` — the pass for derivatives at an accepted point included.
    #[test]
    fn the_evaluations_spent_are_counted_and_never_exceed_the_budget() {
        for maxeval in 1..=12u32 {
            let mut x = [1.5, -0.4];
            let (_, calls) = solve(&mut x, &[-9.0; 2], &[9.0; 2], maxeval, 1e-15, exp_fit);
            assert!(calls <= maxeval, "budget {maxeval}: {calls} passes");
        }
    }

    /// The Cholesky solve: a positive-definite block is solved, a ridge turns a singular block
    /// into a solvable one, and a block that is not positive definite — or not finite — is
    /// `None` rather than a step.
    #[test]
    fn the_cholesky_solve_is_none_unless_the_block_is_positive_definite() {
        // [[4, 2], [2, 3]] s = [2, 1]  =>  s = [0.5, 0].
        let a = [4.0, 2.0, 2.0, 3.0];
        let s = cholesky_solve(&a, 2, 0.0, &[2.0, 1.0]).unwrap();
        assert!((s[0] - 0.5).abs() < 1e-14 && s[1].abs() < 1e-14, "{s:?}");
        // The ridge is added to the diagonal: (A + 1·I) s = [2, 1]  =>  [0.35, 0.05]·… by hand.
        let r = cholesky_solve(&a, 2, 1.0, &[2.0, 1.0]).unwrap();
        // [[5, 2], [2, 4]]: det 16, s = [(2*4 - 1*2)/16, (5*1 - 2*2)/16].
        assert!((r[0] - 6.0 / 16.0).abs() < 1e-14 && (r[1] - 1.0 / 16.0).abs() < 1e-14);
        // Singular: no solve at ridge 0, one with a ridge.
        let sing = [1.0, 1.0, 1.0, 1.0];
        assert!(cholesky_solve(&sing, 2, 0.0, &[1.0, 1.0]).is_none());
        assert!(cholesky_solve(&sing, 2, 1e-6, &[1.0, 1.0]).is_some());
        // Indefinite and non-finite.
        assert!(cholesky_solve(&[1.0, 2.0, 2.0, 1.0], 2, 0.0, &[1.0, 1.0]).is_none());
        assert!(cholesky_solve(&[f64::NAN], 1, 0.0, &[1.0]).is_none());
        assert!(cholesky_solve(&[f64::NEG_INFINITY], 1, 0.0, &[1.0]).is_none());
    }

    /// The tally counts what each solve did, and its warning names each cause with the other's
    /// negative: a solve that lost its information is not a stall, and a stall is not a loss.
    #[test]
    fn the_tally_names_a_lost_information_and_a_stall_apart() {
        let tally = NewtonTally::default();
        assert!(tally.warning("IMP").is_none(), "nothing ran");
        let quad = understated(1.0, 3.0);
        let run = |x0: f64, fused: &dyn Fn(&[f64]) -> Option<ValueGradInfo>| {
            let mut x = [x0];
            newton_solve(
                &mut x,
                &[-9.0],
                &[9.0],
                100,
                1e-12,
                |v| quad(v).unwrap().val,
                fused,
                Some(&tally),
            )
        };
        // Converged on its own: counted as a solve, not as a hand-over.
        assert!(run(2.9, &quad).converged);
        assert!(tally.warning("IMP").is_none());
        // A subject's information unavailable at the start: lost.
        let declined = run(2.9, &|_| None);
        assert!(!declined.converged && declined.evals == 0);
        let w = tally.warning("IMP").expect("a lost solve");
        assert!(
            w.starts_with("IMP:")
                && w.contains("1 of 2")
                && w.contains("lost the expected information"),
            "{w}"
        );
        assert!(!w.contains("no admissible"), "{w}");
        // An information that does not factor at any ridge: a stall, not a loss.
        let nan_info = |v: &[f64]| {
            Some(ValueGradInfo {
                val: 0.5 * (v[0] - 3.0).powi(2),
                grad: vec![v[0] - 3.0],
                info: vec![f64::NAN],
            })
        };
        assert!(!run(2.9, &nan_info).converged);
        let w = tally.warning("IMP").unwrap();
        assert!(w.contains("2 of 3") && w.contains("1 lost the expected information"), "{w}");
        assert!(w.contains("1 found no admissible damped step"), "{w}");
    }

    /// The warning names the cause each hand-over had, with the other causes' negatives: a start at
    /// the sentinel objective is not "the adjoint declined a subject's recording" (nothing was
    /// declined; a prediction is not finite) and a declined recording is not a non-finite start.
    /// Mutation: counting the sentinel start as `lost_information` fails the first block.
    #[test]
    fn the_tally_names_a_non_finite_start_apart_from_a_declined_recording() {
        let sentinel = |_: &[f64]| {
            Some(ValueGradInfo { val: SUBJECT_SENTINEL_NLL, grad: vec![0.0], info: vec![1.0] })
        };
        let run = |tally: &NewtonTally, fused: &dyn Fn(&[f64]) -> Option<ValueGradInfo>| {
            let mut x = [2.9];
            let s = newton_solve(&mut x, &[-9.0], &[9.0], 100, 1e-12, |_| 1.0, fused, Some(tally));
            assert!(!s.converged);
            assert_eq!(x, [2.9], "nothing was solved");
        };
        let t = NewtonTally::default();
        run(&t, &sentinel);
        let w = t.warning("SAEM").expect("a hand-over");
        assert!(
            w.contains("1 of 1") && w.contains("started where the objective is not finite"),
            "{w}"
        );
        assert!(
            !w.contains("lost the expected information") && !w.contains("adjoint declined"),
            "{w}"
        );
        let t = NewtonTally::default();
        run(&t, &|_| None);
        let w = t.warning("SAEM").expect("a hand-over");
        assert!(
            w.contains("1 lost the expected information") && w.contains("adjoint declined"),
            "{w}"
        );
        assert!(!w.contains("not finite"), "{w}");
    }

    /// The Armijo test is on a descent direction. Two free coordinates, `I = [[1, .9], [.9, 1]]`,
    /// `g = [1, .5]`: the Fisher step `[-2.89, +2.11]` clamps to `[-0.1, +2.11]` at the lower
    /// bound of the first, a projected step with `gᵀs = +0.95`. The objective at the start is 10
    /// and `5e-5` higher at every trial, which the bound `cur + 1e-4·slope` admitted for that
    /// slope — an increase accepted at ridge 0, then reported as converged. The solve must keep
    /// its start (stalled: no trial decreases the objective) rather than accept the increase.
    /// Mutation: dropping the `slope < 0` check makes the outcome `Converged` at a moved `x`.
    #[test]
    fn a_clamped_step_that_is_not_a_descent_direction_is_not_accepted() {
        let fused = |_: &[f64]| {
            Some(ValueGradInfo { val: 10.0, grad: vec![1.0, 0.5], info: vec![1.0, 0.9, 0.9, 1.0] })
        };
        let mut x = [0.1, 0.0];
        let (out, _) =
            fisher_scoring(&mut x, &[0.0, -9.0], &[9.0, 9.0], 200, 1e-9, |_| 10.0 + 5e-5, fused);
        assert_eq!((out, x), (Stalled, [0.1, 0.0]));
        // A real objective with those derivatives at the start (`f = ½(x − x*)ᵀ I (x − x*)`,
        // `x*` below the bound on the first coordinate): it descends, on the bound, and never
        // ends above where it started.
        let xs = [-2.79, 2.105];
        let f = move |v: &[f64]| {
            let d = [v[0] - xs[0], v[1] - xs[1]];
            let id = [d[0] + 0.9 * d[1], 0.9 * d[0] + d[1]];
            Some(ValueGradInfo {
                val: 0.5 * (d[0] * id[0] + d[1] * id[1]),
                grad: id.to_vec(),
                info: vec![1.0, 0.9, 0.9, 1.0],
            })
        };
        let start = f(&[0.1, 0.0]).unwrap().val;
        let mut y = [0.1, 0.0];
        let (out, _) = solve(&mut y, &[0.0, -9.0], &[9.0, 9.0], 200, 1e-12, f);
        assert_eq!(out, Converged);
        assert_eq!(y[0], 0.0, "held on the bound");
        assert!(f(&y).unwrap().val < start, "{y:?}");
    }

    #[test]
    fn folding_keeps_subject_order_and_declines_on_a_missing_information() {
        let ok = vec![
            (1.0, vec![1.0, 2.0], Some(vec![1.0, 0.0, 0.0, 1.0])),
            (2.0, vec![3.0, 4.0], Some(vec![2.0, 0.0, 0.0, 2.0])),
        ];
        let f = fold_value_grad_info(ok.clone(), 2, true).unwrap();
        assert_eq!((f.val, f.grad, f.info), (3.0, vec![4.0, 6.0], vec![3.0, 0.0, 0.0, 3.0]));
        // Without `want_info` the information is neither required nor summed.
        let v = fold_value_grad_info(ok, 2, false).unwrap();
        assert_eq!((v.val, v.grad, v.info), (3.0, vec![4.0, 6.0], vec![]));
        let bad = vec![(1.0, vec![0.0, 0.0], None)];
        assert!(fold_value_grad_info(bad.clone(), 2, true).is_none());
        assert!(fold_value_grad_info(bad, 2, false).is_some());
    }

    /// The shared cleanup: a non-finite value becomes the crate's sentinel and a non-finite
    /// gradient entry a zero; finite ones are untouched.
    #[test]
    fn non_finite_values_and_gradients_are_cleaned_to_the_sentinel_and_zero() {
        assert_eq!(finite_value(f64::NAN), SUBJECT_SENTINEL_NLL);
        assert_eq!(finite_value(f64::INFINITY), SUBJECT_SENTINEL_NLL);
        assert_eq!(finite_value(-2.5), -2.5);
        let mut g = [1.0, f64::NAN, f64::NEG_INFINITY, -0.5];
        finite_grad(&mut g);
        assert_eq!(g, [1.0, 0.0, 0.0, -0.5]);
        let mut out = [9.0; 2];
        assert_eq!(
            nlopt_value_and_grad((f64::NAN, vec![f64::NAN, 3.0]), &mut out),
            SUBJECT_SENTINEL_NLL
        );
        assert_eq!(out, [0.0, 3.0]);
    }
}
