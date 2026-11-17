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
//! evaluation budget, and the same stopping rule — `|Δf| < ftol_rel · |f|`, NLopt's
//! `ftol_rel`, applied to the *accepted* step and, before taking one, to the decrease the
//! quadratic model predicts for it.

/// One fused evaluation: objective value, packed gradient and row-major `n × n` information.
pub(crate) struct ValueGradInfo {
    pub val: f64,
    pub grad: Vec<f64>,
    pub info: Vec<f64>,
}

/// Fold per-subject `(value, gradient, information)` triples in subject order. `None` when any
/// subject has no information (outside the closed form's scope): the caller then keeps the
/// derivative-free solve rather than stepping on a partial curvature.
pub(crate) fn fold_value_grad_info(
    per_subj: Vec<(f64, Vec<f64>, Option<Vec<f64>>)>,
    n: usize,
) -> Option<ValueGradInfo> {
    let mut out = ValueGradInfo {
        val: 0.0,
        grad: vec![0.0; n],
        info: vec![0.0; n * n],
    };
    for (v, g, i) in per_subj {
        let i = i?;
        out.val += v;
        for (a, b) in out.grad.iter_mut().zip(&g) {
            *a += b;
        }
        for (a, b) in out.info.iter_mut().zip(&i) {
            *a += b;
        }
    }
    Some(out)
}

/// Armijo sufficient-decrease constant of the backtracking line search (the textbook `1e-4`).
const ARMIJO_C1: f64 = 1e-4;
/// First nonzero Levenberg ridge, relative to the largest diagonal entry of the information.
const RIDGE_FLOOR: f64 = 1e-9;
/// Dampings tried before the iteration gives up and keeps its current point.
const MAX_DAMPINGS: u32 = 30;

/// Solve `(A + ridge·I) s = rhs` by Cholesky; `None` when not positive definite.
fn cholesky_solve(a: &[f64], m: usize, ridge: f64, rhs: &[f64]) -> Option<Vec<f64>> {
    let mut l = vec![0.0f64; m * m];
    for i in 0..m {
        for j in 0..=i {
            let mut s = a[i * m + j] + if i == j { ridge } else { 0.0 };
            for k in 0..j {
                s -= l[i * m + k] * l[j * m + k];
            }
            if i == j {
                if !(s > 0.0) || !s.is_finite() {
                    return None;
                }
                l[i * m + i] = s.sqrt();
            } else {
                l[i * m + j] = s / l[j * m + j];
            }
        }
    }
    let mut y = vec![0.0f64; m];
    for i in 0..m {
        let mut s = rhs[i];
        for k in 0..i {
            s -= l[i * m + k] * y[k];
        }
        y[i] = s / l[i * m + i];
    }
    for i in (0..m).rev() {
        let mut s = y[i];
        for k in i + 1..m {
            s -= l[k * m + i] * y[k];
        }
        y[i] = s / l[i * m + i];
    }
    Some(y)
}

/// Minimise an objective over the box `[lower, upper]` from `x` (updated in place) by Fisher
/// scoring.
///
/// `fused(x)` returns the value, gradient and information in one pass, or `None` where the
/// information is out of scope; `value(x)` is the same objective's value alone (a trial point
/// is judged on it, and only an accepted point that the iteration continues from pays for the
/// derivatives). Returns `false` — `x` untouched — when the information is out of scope at the
/// start point, so the caller can keep the derivative-free solve. `maxeval` bounds the
/// evaluations of either kind (the budget BOBYQA gets in value evaluations).
///
/// A coordinate with `lower == upper`, or at a bound with the gradient pushing outward, is held
/// for that step; the step solves the remaining block of `I` (Levenberg-damped as far as needed
/// to factor it and to decrease the objective) and a trial point is clamped into the box.
pub(crate) fn fisher_scoring<V, F>(
    x: &mut [f64],
    lower: &[f64],
    upper: &[f64],
    maxeval: u32,
    ftol_rel: f64,
    mut value: V,
    mut fused: F,
) -> bool
where
    V: FnMut(&[f64]) -> f64,
    F: FnMut(&[f64]) -> Option<ValueGradInfo>,
{
    let n = x.len();
    let Some(mut cur) = fused(x) else {
        return false;
    };
    let mut evals = 1u32;
    // Persists across iterations (Levenberg-Marquardt): a rejected step raises it, an accepted
    // one lowers it, and below the floor it is exactly zero — the plain Fisher step.
    let mut ridge = 0.0f64;
    while evals < maxeval {
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
        let max_diag = (0..m)
            .map(|p| a[p * m + p])
            .fold(0.0f64, f64::max)
            .max(f64::MIN_POSITIVE);
        let neg_g: Vec<f64> = g.iter().map(|v| -v).collect();
        // Levenberg damping: the plain Fisher step first; a block that does not factor, or a
        // step the objective rejects, multiplies the ridge (relative to the largest diagonal
        // entry) by ten from `RIDGE_FLOOR` — a larger ridge is a shorter, more
        // gradient-like step, so the sequence ends in a decrease.
        let mut accepted = None;
        let mut first = true;
        for _ in 0..MAX_DAMPINGS {
            if evals >= maxeval {
                break;
            }
            let Some(s) = cholesky_solve(&a, m, ridge * max_diag, &neg_g) else {
                ridge = (ridge * 10.0).max(RIDGE_FLOOR);
                continue;
            };
            if first {
                first = false;
                // The decrease the quadratic model predicts, ½ gᵀ(A + ridge)⁻¹ g, against
                // the tolerance: below it the step is not worth a pass.
                let predicted = -0.5 * g.iter().zip(&s).map(|(a, b)| a * b).sum::<f64>();
                if !(predicted > ftol_rel * cur.val.abs()) {
                    break;
                }
            }
            let mut trial = x.to_vec();
            for (p, &i) in free.iter().enumerate() {
                trial[i] = (x[i] + s[p]).clamp(lower[i], upper[i]);
            }
            let slope: f64 = (0..n).map(|i| cur.grad[i] * (trial[i] - x[i])).sum();
            let v = value(&trial);
            evals += 1;
            if v.is_finite() && v <= cur.val + ARMIJO_C1 * slope {
                accepted = Some((trial, v));
                break;
            }
            ridge = (ridge * 10.0).max(RIDGE_FLOOR);
        }
        let Some((trial, v)) = accepted else {
            break;
        };
        ridge = if ridge * 0.1 < RIDGE_FLOOR {
            0.0
        } else {
            ridge * 0.1
        };
        let decrease = cur.val - v;
        x.copy_from_slice(&trial);
        // Inside the tolerance: this is the answer, and its derivatives are not needed.
        if decrease < ftol_rel * v.abs() {
            break;
        }
        // Another step from here needs the derivatives at the accepted point.
        evals += 1;
        match fused(x) {
            Some(nx) => cur = nx,
            None => break,
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gaussian regression `y_j = a·exp(b·t_j)`, additive σ fixed at 1: value, exact score and
    /// the Gauss-Newton information `JᵀJ` of `½ Σ (y - f)²`.
    fn exp_fit(x: &[f64]) -> Option<ValueGradInfo> {
        let ts = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0];
        let truth = (2.0f64, -0.7f64);
        let (a, b) = (x[0], x[1]);
        let mut out = ValueGradInfo {
            val: 0.0,
            grad: vec![0.0; 2],
            info: vec![0.0; 4],
        };
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
    fn solve<F>(x: &mut [f64], lo: &[f64], hi: &[f64], maxeval: u32, ftol: f64, f: F) -> (bool, u32)
    where
        F: Fn(&[f64]) -> Option<ValueGradInfo>,
    {
        let calls = std::cell::Cell::new(0u32);
        let ok = fisher_scoring(
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
        (ok, calls.get())
    }

    #[test]
    fn fisher_scoring_finds_the_zero_residual_optimum_in_a_few_passes() {
        let mut x = [1.5, -0.4];
        let (solved, calls) = solve(&mut x, &[-9.0; 2], &[9.0; 2], 100, 1e-12, exp_fit);
        assert!(solved);
        assert!(
            (x[0] - 2.0).abs() < 1e-5 && (x[1] + 0.7).abs() < 1e-5,
            "{x:?}"
        );
        assert!(calls < 30, "{calls} passes");
    }

    #[test]
    fn fisher_scoring_respects_bounds_and_pinned_coordinates() {
        // Optimum a = 2 lies above the bound 1.8: the solve stops on the bound, b free.
        let mut x = [1.5, -0.4];
        assert!(solve(&mut x, &[-9.0, -9.0], &[1.8, 9.0], 100, 1e-12, exp_fit).0);
        assert_eq!(x[0], 1.8);
        // A pinned coordinate never moves.
        let mut y = [1.5, -0.4];
        assert!(solve(&mut y, &[1.5, -9.0], &[1.5, 9.0], 100, 1e-12, exp_fit).0);
        assert_eq!(y[0], 1.5);
        assert!(y[1] != -0.4);
    }

    #[test]
    fn fisher_scoring_declines_without_information_and_honours_the_budget() {
        let mut x = [1.5, -0.4];
        assert!(!fisher_scoring(
            &mut x,
            &[-9.0; 2],
            &[9.0; 2],
            10,
            1e-9,
            |_| 0.0,
            |_| None
        ));
        assert_eq!(x, [1.5, -0.4]);
        let mut z = [1.5, -0.4];
        let (_, calls) = solve(&mut z, &[-9.0; 2], &[9.0; 2], 3, 1e-15, exp_fit);
        assert!(calls <= 4, "{calls}");
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
            Some(ValueGradInfo {
                val: 0.5 * v[0] * v[0],
                grad: vec![v[0]],
                info: vec![0.01],
            })
        });
        assert!(solved && x[0].abs() < 1e-3, "{x:?}");
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
        assert!(solved && x[0].abs() < 1e-6 && x[1].abs() < 1.0, "{x:?}");
    }

    #[test]
    fn folding_keeps_subject_order_and_declines_on_a_missing_information() {
        let ok = vec![
            (1.0, vec![1.0, 2.0], Some(vec![1.0, 0.0, 0.0, 1.0])),
            (2.0, vec![3.0, 4.0], Some(vec![2.0, 0.0, 0.0, 2.0])),
        ];
        let f = fold_value_grad_info(ok, 2).unwrap();
        assert_eq!(
            (f.val, f.grad, f.info),
            (3.0, vec![4.0, 6.0], vec![3.0, 0.0, 0.0, 3.0])
        );
        let bad = vec![(1.0, vec![0.0, 0.0], None)];
        assert!(fold_value_grad_info(bad, 2).is_none());
    }
}
