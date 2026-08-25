//! A-priori stiffness probe: one Jacobian eigenvalue read that picks the stepper for a
//! segment before any integration happens (`[fit_options] ode_method = auto`, issue #978).
//!
//! # What is measured
//!
//! Stiffness is a property of the **right-hand side**, not of the dataset: a system is stiff
//! when its fastest decaying mode is far faster than the interval being integrated, so an
//! explicit method's step is capped by *stability* rather than by accuracy. The fastest mode
//! is the most negative real part of an eigenvalue of `J = ∂f/∂u`, so the discriminator is
//!
//! ```text
//! λ_max = max_i |Re λ_i(J)|          (units: 1/time)
//! τ_fast = 1 / λ_max                 (the fastest timescale in the system)
//! ```
//!
//! and the probe calls a segment stiff when `λ_max ≥` [`STIFF_RE_LAMBDA_THRESHOLD`] — i.e.
//! when `τ_fast` is below [`STIFF_TAU_FAST`] in the model's own time unit.
//!
//! Measured per segment across the ferx-testdata ODE library (#978):
//!
//! | model family | `λ_max` across segments | verdict |
//! |---|---|---|
//! | cyclophosphamide | 20 – 234 | stiff everywhere |
//! | TMDD SAD (`cr`, `crib`, `ib`) | 1.7 – 126 | **straddles the threshold within one model** |
//! | TMDD SAD (`qss`, `mm`, `linear`) | 1.8 – 2.1 | non-stiff |
//! | busulfan, pembrolizumab | 0.1 – 4.6 | non-stiff |
//!
//! Note the TMDD row, because it is the one that decides the shape of this module. The issue
//! that proposed the probe measured those models at their *declared initial condition* only and
//! read them as uniformly non-stiff (`≤ 12.4`), which suggested a comfortable 7× gap between
//! the stiff and non-stiff families and a threshold with nothing near it. Per segment there is
//! no such gap: the same model reads 3.7 on its low-dose subjects and 126 on its high-dose ones.
//! Stiffness here is a property of *where the model is*, not of the model, and a single
//! per-model verdict cannot be right for both ends of a dose range. Hence a per-segment probe —
//! and hence the guard in `integrate_resolved_g`, which stops being a formality once a real
//! model is expected to cross the threshold in normal operation.
//!
//! **The threshold is dimensional.** `λ_max` is a rate, so a model written in days reads 24×
//! smaller than the same model written in hours. It is calibrated for the PK convention the
//! testdata library uses (hours), which is also the convention `ode_reltol`'s defaults assume;
//! a model on a very different clock should set `ode_method` explicitly rather than rely on
//! `auto`.
//!
//! # Where it is evaluated
//!
//! [`resolve_method`] is called by each driver on **that call's own initial state**, not once
//! per subject on the declared `init(...)` condition. That matters, and is not a detail: for
//! any model whose stiffness is carried by a binding term (`KON · C · R`), the fast eigenvalue
//! is identically zero before any drug is present, so the declared initial condition is the
//! one state at which such a model looks least stiff. Because the prediction path splits a
//! subject at every dose event and integrates each segment from its own post-dose state, the
//! probe sees the post-dose Jacobian on every segment after the first — the state where
//! binding stiffness actually lives — with no extra plumbing.
//!
//! Re-probing per segment also tracks parameter drift for free: the probe runs at whatever
//! `θ`/`η` the current likelihood evaluation carries, so a fit that walks into (or out of) a
//! stiff region of parameter space re-decides rather than riding a decision cached from the
//! initial estimates.
//!
//! # Consistency across `T`
//!
//! The probe reads [`PkNum::val`] only, exactly like every other comparison a driver makes.
//! The `T = f64` prediction and the `T = Dual2` sensitivity solve therefore resolve `auto` to
//! the *same* method for the same segment, so the analytic gradient differentiates the
//! trajectory the predictor reports rather than one produced by a different stepper.
//!
//! # Cost
//!
//! `n + 1` right-hand-side evaluations for the finite-difference Jacobian plus one `O(n³)`
//! eigensolve — the same work a *single* Rosenbrock step attempt does, against a segment that
//! takes tens to thousands of steps. Zero-length segments skip it entirely. On the testdata
//! models that read non-stiff, `auto` is not measurably slower than naming `rk45`; where it
//! fires it turns a 372 s fit into a 2.3 s one.

use crate::ode::solver::{OdeMethod, OdeSolverOptions};
use crate::sens::num::PkNum;

/// `|Re λ|max` at or above which [`resolve_method`] calls a segment stiff.
///
/// Calibrated on the ferx-testdata ODE library (see the module docs). It sits well above the
/// rates ordinary absorption/elimination produce — the non-stiff models there top out at 4.6,
/// and even the fast TMDD parameterizations sit at 1.8–2.1 — and well below the 20–234 the
/// stiff cyclophosphamide models read at every segment.
///
/// It is *not* a gap with nothing in it. The TMDD `cr`/`crib`/`ib` models sweep 1.7 to 126
/// across their own segments, so they cross it within a single fit, on the high-dose subjects.
/// That is the intended behaviour rather than a miscalibration — those segments really are
/// stiff, and fitting one of them under `auto` is 160× faster than under the explicit default —
/// but it does mean escalation is a routine event on real models, not a rare one.
pub const STIFF_RE_LAMBDA_THRESHOLD: f64 = 30.0;

/// The same threshold expressed as a timescale: a segment is stiff when its fastest mode
/// decays with a time constant below this, in the model's own time unit.
pub const STIFF_TAU_FAST: f64 = 1.0 / STIFF_RE_LAMBDA_THRESHOLD;

/// Relative finite-difference step for the probe Jacobian, floored so a compartment sitting
/// at zero (every state before its first dose) still gets a usable perturbation.
const JAC_FD_REL_STEP: f64 = 1e-7;

/// The stiff stepper `auto` escalates to.
///
/// [`Rodas4`](OdeMethod::Rodas4) is the stiff workhorse at the tolerances PK fits actually
/// run at; [`Rodas5P`](OdeMethod::Rodas5P) earns its extra stages only once the tolerance is
/// tight enough for its higher order to pay, which is the same `ode_reltol ≤ 1e-8` regime
/// where an ODE-form OFV is being matched against an analytical one.
const fn stiff_method_for(opts: &OdeSolverOptions) -> OdeMethod {
    if opts.reltol <= 1e-8 {
        OdeMethod::Rodas5P
    } else {
        OdeMethod::Rodas4
    }
}

/// `max_i |Re λ_i(∂f/∂u)|` at `(t, u)` — the a-priori stiffness discriminator.
///
/// Returns `None` when the Jacobian cannot be formed or scored: a non-finite right-hand side,
/// an empty system, or an eigensolve that does not converge. A caller that cannot classify
/// must not guess — see [`resolve_method`], which keeps the explicit default in that case.
///
/// The Jacobian is one-sided (`n + 1` evaluations rather than `2n`). One-sided FD carries
/// `O(h)` truncation error where central differences carry `O(h²)`, which would matter for a
/// derivative and does not matter here: the result is thresholded, not used, and it would take
/// an error of tens of percent to move a model across the threshold from where the measured
/// ones sit. Halving the evaluation count is what keeps the probe cheaper than one step of the
/// solver it is choosing.
pub(crate) fn max_abs_re_eigenvalue<T: PkNum>(
    rhs: &dyn Fn(&[T], &[T], f64, &mut [T]),
    u: &[T],
    params: &[T],
    t: f64,
) -> Option<f64> {
    let n = u.len();
    if n == 0 {
        return None;
    }

    let mut f0 = vec![T::from_f64(0.0); n];
    rhs(u, params, t, &mut f0);
    if f0.iter().any(|v| !v.val().is_finite()) {
        return None;
    }

    let mut u_pert = u.to_vec();
    let mut f1 = vec![T::from_f64(0.0); n];
    let mut jac = vec![0.0; n * n];
    for j in 0..n {
        let h = JAC_FD_REL_STEP * u[j].val().abs().max(1.0);
        u_pert[j] = u[j] + T::from_f64(h);
        rhs(&u_pert, params, t, &mut f1);
        u_pert[j] = u[j];
        for (i, f1i) in f1.iter().enumerate() {
            let d = (f1i.val() - f0[i].val()) / h;
            if !d.is_finite() {
                return None;
            }
            // Column-major: `jac[i + j * n]` is `∂f_i/∂u_j`, the layout `DMatrix` expects.
            jac[i + j * n] = d;
        }
    }

    let m = nalgebra::DMatrix::from_vec(n, n, jac);
    let eigs = m.complex_eigenvalues();
    let lambda_max = eigs.iter().fold(0.0_f64, |acc, z| acc.max(z.re.abs()));
    lambda_max.is_finite().then_some(lambda_max)
}

/// Resolve [`OdeMethod::Auto`] against the system this driver is about to integrate; every
/// other method is returned unchanged, so naming a solver explicitly still forces it.
///
/// The fallback on an unclassifiable system is the explicit default rather than a stiff
/// method, and that direction is deliberate. All three implicit steppers were measured
/// diverging on *some* model in the testdata library (`rodas4` on one TMDD parameterization,
/// `rodas5p` on another, `rosenbrock23` on a third) where the explicit family stayed clean, so
/// escalating on a guess risks trading a slow answer for a wrong one. An explicit method on a
/// stiff system is slow and loud (`min_step_clamped_steps`); a Rosenbrock method on a system
/// with a singular `W` is fast and wrong.
pub(crate) fn resolve_method<T: PkNum>(
    rhs: &dyn Fn(&[T], &[T], f64, &mut [T]),
    u: &[T],
    params: &[T],
    t: f64,
    opts: &OdeSolverOptions,
) -> OdeMethod {
    if opts.method != OdeMethod::Auto {
        return opts.method;
    }
    match max_abs_re_eigenvalue(rhs, u, params, t) {
        Some(lambda) if lambda >= STIFF_RE_LAMBDA_THRESHOLD => stiff_method_for(opts),
        _ => OdeMethod::EXPLICIT_FALLBACK,
    }
}

#[cfg(test)]
#[path = "stiffness_tests.rs"]
mod tests;
