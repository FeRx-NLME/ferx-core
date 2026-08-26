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
//! # Two things this discriminator cannot see
//!
//! Both are reasons to name a method by hand, and both matter more now that `auto` is the
//! default (#978) than they did when it was opt-in.
//!
//! **It is a rate, so it carries the model's time unit.** A model written in days reads 24×
//! smaller than the same model written in hours. The threshold is calibrated for the PK
//! convention the testdata library uses (hours), which is also what `ode_reltol`'s defaults
//! assume. A model on a minute clock reads 60× *larger* and can escalate wholesale — an oral
//! absorption model with `ka = 0.6/min` puts `λ_max` at 36 with nothing stiff anywhere in it.
//!
//! **It measures speed, not separation.** Textbook stiffness is a *ratio* — a fast mode
//! alongside a slow one — and this reads only the fast end. A system whose modes are all
//! equally fast is not stiff and an explicit method handles it comfortably, but it reads the
//! same as one that is: a transit-absorption chain written out in `[odes]` with `ktr = 50` has
//! every eigenvalue at `−50`, escalates, and pays a Jacobian and an `O(n³)` factorization per
//! step for nothing. Reading `|Re λ|` rather than `Re λ` widens this further, so a *growing*
//! mode escalates too — a diverging system is not a stiff one, and the guard rather than the
//! probe is what catches that.
//!
//! Both were accepted rather than fixed, because the alternatives are worse in ways the
//! measurement supports: a ratio `λ_max/λ_min` is undefined the moment any mode sits at zero,
//! which is every compartment before its first dose, and normalizing by the segment length
//! makes a long segment on a benign model read stiff. The escalations they cause are slower,
//! not wrong — the guard bounds the damage — and `ode_method = rk45` opts out completely.
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

/// Iteration cap for the probe's Schur decomposition.
///
/// Generous — PK Jacobians are small and well behaved, and nothing here has been observed to
/// need more than a handful of sweeps — but *finite*, which is the point: an uncapped QR
/// iteration on a pathological Jacobian would hang a worker thread, and this probe is only ever
/// a hint about which stepper to prefer. Failing to classify costs the explicit default;
/// failing to return costs the fit.
const EIGENSOLVE_MAX_ITERATIONS: usize = 1000;

/// The stiff stepper `auto` escalates to.
///
/// [`Rodas4`](OdeMethod::Rodas4) is the stiff workhorse at the tolerances PK fits actually
/// run at; [`Rodas5P`](OdeMethod::Rodas5P) earns its extra stages only once the tolerance is
/// tight enough for its higher order to pay.
///
/// The cut is `1e-9`, matching what [`OdeMethod`]'s own documentation tells a user choosing by
/// hand ("`rodas5p` at `ode_reltol ≤ 1e-9`"). It was `1e-8` here, which put a model on
/// `ode_reltol = 1e-8` on a *different* stepper than the docs would have sent its author to —
/// a gratuitous way for `auto` and a hand-written control stream to disagree.
const fn stiff_method_for(opts: &OdeSolverOptions) -> OdeMethod {
    if opts.reltol <= 1e-9 {
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
///
/// **Test-only since #1080 Part B.** [`resolve_method`] no longer calls it: it builds the
/// Jacobian itself so it can try [`gershgorin_abs_bound`] first and skip the eigensolve on the
/// segments that cannot possibly be stiff. This stays as the *exact* definition of the
/// discriminator, and the fast path is tested against it
/// (`the_gershgorin_fast_path_agrees_with_the_exact_eigensolve`) — which is the only way that
/// short-circuit is allowed to exist.
#[cfg(test)]
pub(crate) fn max_abs_re_eigenvalue<T: PkNum>(
    rhs: &dyn Fn(&[T], &[T], f64, &mut [T]),
    u: &[T],
    params: &[T],
    t: f64,
) -> Option<f64> {
    let jac = fd_jacobian(rhs, u, params, t)?;
    schur_max_abs_re(jac, u.len())
}

/// One-sided finite-difference Jacobian at `(t, u)`, column-major (`jac[i + j * n]` is
/// `∂f_i/∂u_j`, the layout `DMatrix` expects). `None` on an empty system or a non-finite
/// entry — a caller that cannot form the Jacobian must not guess.
fn fd_jacobian<T: PkNum>(
    rhs: &dyn Fn(&[T], &[T], f64, &mut [T]),
    u: &[T],
    params: &[T],
    t: f64,
) -> Option<Vec<f64>> {
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
            jac[i + j * n] = d;
        }
    }
    Some(jac)
}

/// `max_i |Re λ_i|` of a column-major `n × n` matrix, via a **bounded** Schur decomposition.
///
/// `DMatrix::complex_eigenvalues` is *not* usable here: it runs the Schur QR iteration with
/// `max_niter = 0`, which disables the iteration cap rather than setting it to zero, and
/// `.unwrap()`s the result — so a matrix it cannot reduce spins inside a fit's worker thread
/// instead of failing. The probe must be bounded above all else: it is a hint, and a hint is
/// never worth hanging for. `try_new` with an explicit cap gives back the `None` the callers
/// read as "cannot classify → stay explicit".
fn schur_max_abs_re(jac: Vec<f64>, n: usize) -> Option<f64> {
    let m = nalgebra::DMatrix::from_vec(n, n, jac);
    let schur = nalgebra::linalg::Schur::try_new(m, f64::EPSILON, EIGENSOLVE_MAX_ITERATIONS)?;
    let lambda_max = schur
        .complex_eigenvalues()
        .iter()
        .fold(0.0_f64, |acc, z| acc.max(z.re.abs()));
    lambda_max.is_finite().then_some(lambda_max)
}

/// Gershgorin's bound on `max_i |λ_i|` — and therefore on `max_i |Re λ_i|` — of a
/// column-major `n × n` matrix: every eigenvalue lies in some disc centred on a diagonal
/// entry with radius the sum of that row's off-diagonal magnitudes, so no eigenvalue can
/// exceed `max_i (|a_ii| + Σ_{j≠i} |a_ij|)`.
///
/// This is the probe's cheap sufficient test for *non*-stiffness (#1080 Part B item 3): `O(n²)`
/// adds against the `O(n³)` Schur iteration and its `DMatrix` allocation, and one-sided —
/// a bound below the threshold **proves** no eigenvalue reaches it, while a bound above it
/// proves nothing and the exact eigensolve still has to run. Since the escalation test is
/// `λ_max ≥ threshold`, that one-sidedness is exactly the direction that can be short-circuited
/// without changing a single decision.
fn gershgorin_abs_bound(jac: &[f64], n: usize) -> f64 {
    (0..n).fold(0.0_f64, |acc, i| {
        let row = (0..n).fold(0.0_f64, |sum, j| sum + jac[i + j * n].abs());
        acc.max(row)
    })
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
    let Some(jac) = fd_jacobian(rhs, u, params, t) else {
        return OdeMethod::EXPLICIT_FALLBACK;
    };
    // Cheap sufficient test first: if Gershgorin's bound on `|λ|max` is already below the
    // threshold, no eigenvalue can reach it and the eigensolve would only confirm the verdict
    // this line already gives. Ordinary absorption/elimination segments — the overwhelming
    // majority of what a fit integrates — exit here, which is what makes the probe cheap
    // enough to run per segment. Measured on a 3-state binding model
    // (`measure_probe_cost_against_segment_cost`): the probe on a non-stiff system drops from
    // 0.56 µs to 0.055 µs, taking `auto`'s overhead over a pinned `rk45` from 33% to 3% on a
    // bare segment and from 19% to 1.3% on a realistic observation grid. That is #1080 Part B
    // item 3's answer: per-subject latching would have traded away the per-segment probe —
    // the only layer that sees the fast-binding family `min_step_clamped_steps` is blind to,
    // and one whose verdict genuinely varies within a subject (a fast mode carried by
    // `KON · central` is identically zero before the first dose) — for a saving a sound bound
    // gets without giving up anything.
    if gershgorin_abs_bound(&jac, u.len()) < STIFF_RE_LAMBDA_THRESHOLD {
        return OdeMethod::EXPLICIT_FALLBACK;
    }
    match schur_max_abs_re(jac, u.len()) {
        Some(lambda) if lambda >= STIFF_RE_LAMBDA_THRESHOLD => stiff_method_for(opts),
        _ => OdeMethod::EXPLICIT_FALLBACK,
    }
}

#[cfg(test)]
#[path = "stiffness_tests.rs"]
mod tests;
