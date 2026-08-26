//! Tests for the Rosenbrock steppers.
//!
//! The load-bearing ones are the **measured convergence orders**: a Rosenbrock tableau is a
//! wall of transcribed decimals, and a single mistyped digit does not produce a wrong answer
//! so much as a quietly lower-order method. Order is therefore what pins the coefficients,
//! not a golden value.

use super::*;
use crate::ode::solver::{solve_ode, solve_ode_with_stats, OdeSolverStats};
use crate::sens::dual2::Dual2;
use approx::assert_relative_eq;

// ---------------------------------------------------------------------------------------
// Order-verification problem
// ---------------------------------------------------------------------------------------

/// Stiffness of the two components of [`order_rhs`].
const ORDER_LAM: f64 = -50.0;
const ORDER_MU: f64 = -30.0;

/// A nonlinear, non-autonomous, mildly stiff system with the **exact** solution
/// `y = (sin t, t²)`:
///
/// ```text
/// y0' = λ(y0 − sin t) + cos t  + 0.1 (y1² − t⁴)
/// y1' = μ(y1 − t²)   + 2t      + 0.05(y0³ − sin³t)
/// ```
///
/// Each correction term vanishes on the exact solution, so the closed form holds for any
/// `λ`, `μ` while the Jacobian stays fully populated and time-dependent — it exercises the
/// `J`, the `∂f/∂t` term and the nonlinearity at once, which a linear decay would not.
fn order_rhs(u: &[f64], _p: &[f64], t: f64, du: &mut [f64]) {
    du[0] = ORDER_LAM * (u[0] - t.sin()) + t.cos() + 0.1 * (u[1] * u[1] - t.powi(4));
    du[1] = ORDER_MU * (u[1] - t * t) + 2.0 * t + 0.05 * (u[0].powi(3) - t.sin().powi(3));
}

/// The same problem with mild rates. The stiff constants above force small steps on an
/// explicit method, and at small `h` the *step's* own error is below the interpolant's, so an
/// interpolation measurement there measures the step instead. Relaxing λ lets `h` be large
/// enough for the continuous extension's error to dominate — which is what
/// [`observed_interpolant_order`] needs to see.
fn mild_rhs(u: &[f64], _p: &[f64], t: f64, du: &mut [f64]) {
    du[0] = -2.0 * (u[0] - t.sin()) + t.cos() + 0.1 * (u[1] * u[1] - t.powi(4));
    du[1] = -1.0 * (u[1] - t * t) + 2.0 * t + 0.05 * (u[0].powi(3) - t.sin().powi(3));
}

fn order_exact(t: f64) -> [f64; 2] {
    [t.sin(), t * t]
}

/// Drive `method` over `[0, 2]` with `n_steps` **fixed** steps, bypassing the adaptive
/// controller (which would otherwise choose the step and make order unmeasurable).
///
/// Returns `(global error at t = 2, largest per-step embedded error estimate)`.
fn fixed_step_run(method: OdeMethod, n_steps: usize) -> (f64, f64) {
    // Driven straight through the `Stepper` trait, so this harness measures **any** method —
    // adding one to the enum brings it into every order test for free.
    let mut stepper = crate::ode::solver::make_stepper::<f64>(2, method);
    let opts = OdeSolverOptions {
        // Tolerances only scale the reported error norm; the step size here is fixed, so they
        // are irrelevant to the measurement. Keep them tight so the norm is not saturated.
        abstol: 1e-14,
        reltol: 1e-12,
        method,
        ..Default::default()
    };
    let mut u = order_exact(0.0).to_vec();
    let h = 2.0 / n_steps as f64;
    let mut t = 0.0;
    let mut est_max = 0.0_f64;

    for _ in 0..n_steps {
        // `err_norm` is the scaled RMS; recover a usable per-step magnitude from the state
        // difference the embedded pair implies by re-deriving it from the norm and the scale.
        let err_norm = stepper.attempt(&order_rhs, &u, &[], t, h, &opts);
        assert!(stepper.attempt_usable(), "{method:?} could not form a step");
        let scale = opts.abstol + opts.reltol * u[0].abs().max(u[1].abs());
        est_max = est_max.max(err_norm * scale);
        u.copy_from_slice(stepper.u_new());
        stepper.on_accept();
        t += h;
    }

    let exact = order_exact(t);
    let err = (u[0] - exact[0]).abs().max((u[1] - exact[1]).abs());
    (err, est_max)
}

/// Mean observed order across a halving sequence.
fn observed_order(method: OdeMethod, coarse: usize, refinements: usize) -> f64 {
    let (mut prev, _) = fixed_step_run(method, coarse);
    let mut n = coarse;
    let mut total = 0.0;
    for _ in 0..refinements {
        n *= 2;
        let (err, _) = fixed_step_run(method, n);
        total += (prev / err).log2();
        prev = err;
    }
    let order = total / refinements as f64;
    println!(
        "observed order  {method:?}: {order:.3}  (from n = {coarse}, {refinements} refinements)"
    );
    order
}

/// **Local** interpolation error of a method's continuous extension: take one step from an
/// *exact* initial condition, read the interpolant at the step midpoint, and compare with the
/// exact solution there. Starting from the exact solution isolates the interpolant — the
/// trajectory carries no accumulated error to mask it.
fn interpolant_local_error(method: OdeMethod, h: f64) -> f64 {
    let mut stepper = crate::ode::solver::make_stepper::<f64>(2, method);
    stepper.set_dense_required(true);
    let opts = OdeSolverOptions {
        abstol: 1e-14,
        reltol: 1e-12,
        method,
        ..Default::default()
    };
    // Start away from t = 0 so the problem's non-autonomous terms are active.
    let t0 = 0.3;
    let u = order_exact(t0).to_vec();
    stepper.attempt(&mild_rhs, &u, &[], t0, h, &opts);
    let mid = order_exact(t0 + 0.5 * h);
    (0..2).fold(0.0_f64, |acc, i| {
        acc.max((stepper.interpolate_component(0.5, &u, h, i) - mid[i]).abs())
    })
}

/// Measured order of the **continuous extension**, the analogue of [`observed_order`] for the
/// interpolant rather than the step.
///
/// This is what actually pins the Rodas dense-output `H` blocks. Endpoint continuity does not:
/// `Θ = 0 → y₀` and `Θ = 1 → y₁` hold structurally for *any* `H` whatsoever, so a test that
/// only checks the ends would pass with the coefficients deleted.
fn observed_interpolant_order(method: OdeMethod, coarse: f64, refinements: usize) -> f64 {
    let mut h = coarse;
    let mut prev = interpolant_local_error(method, h);
    let mut total = 0.0;
    for _ in 0..refinements {
        h *= 0.5;
        let err = interpolant_local_error(method, h);
        total += (prev / err).log2();
        prev = err;
    }
    let order = total / refinements as f64;
    println!("interpolant order  {method:?}: {order:.3}  (local error at h = {coarse} → {h})");
    order
}

/// The continuous extensions must earn their coefficients. A Rodas `H` block with a mistyped
/// digit still interpolates *exactly* at both step ends — that is structural — but its order
/// inside the step collapses, which is what this measures.
#[test]
fn interpolant_orders_pin_the_dense_output_blocks() {
    // Hermite (RK45, Vern7) is a cubic through two values and two derivatives: local order 4.
    for method in [OdeMethod::Rk45, OdeMethod::Vern7] {
        let order = observed_interpolant_order(method, 0.8, 3);
        assert!(
            (3.0..5.0).contains(&order),
            "{method:?} Hermite interpolant order {order:.2} outside [3.0, 5.0]"
        );
    }
    // The Rodas continuous extensions are built from their own `H` blocks; a wrong digit drops
    // these well below 3.
    for method in [OdeMethod::Rodas4, OdeMethod::Rodas5P] {
        let order = observed_interpolant_order(method, 0.8, 3);
        assert!(
            order > 3.0,
            "{method:?} dense-output order {order:.2} is too low — check its `h` block"
        );
    }
}

/// Rodas4 must converge at ~4th order. A transcription error in `a`, `c`, `α`, `d` or `γ`
/// collapses this well below the 3.3 floor (the floor is under 4 because the asymptotic
/// order is approached from below on this problem — 3.33 → 3.74 across the sequence).
#[test]
fn rodas4_converges_at_fourth_order() {
    let order = observed_order(OdeMethod::Rodas4, 50, 3);
    assert!(
        (3.3..4.6).contains(&order),
        "Rodas4 observed order {order:.2} outside [3.3, 4.6]"
    );
    // Absolute accuracy pins the coefficients themselves, not just their order.
    let (err, _) = fixed_step_run(OdeMethod::Rodas4, 100);
    assert!(
        err < 5e-8,
        "Rodas4 error at h=0.02 was {err:.3e}, expected < 5e-8"
    );
}

/// Rodas5P must converge at ~5th order and be sharply more accurate than Rodas4 at the same
/// step size — the whole reason to pay for its 8 stages.
#[test]
fn rodas5p_converges_at_fifth_order() {
    let order = observed_order(OdeMethod::Rodas5P, 25, 3);
    assert!(
        (4.0..5.7).contains(&order),
        "Rodas5P observed order {order:.2} outside [4.0, 5.7]"
    );
    let (err5p, _) = fixed_step_run(OdeMethod::Rodas5P, 100);
    let (err4, _) = fixed_step_run(OdeMethod::Rodas4, 100);
    assert!(err5p < 1e-9, "Rodas5P error at h=0.02 was {err5p:.3e}");
    assert!(
        err5p < err4 / 10.0,
        "Rodas5P ({err5p:.3e}) should be ≫10× more accurate than Rodas4 ({err4:.3e}) at h=0.02"
    );
}

/// `ode23s` is 2nd order — flat, not asymptotic, so this is a tight window.
#[test]
fn rosenbrock23_converges_at_second_order() {
    let order = observed_order(OdeMethod::Rosenbrock23, 25, 3);
    assert!(
        (1.8..2.3).contains(&order),
        "Rosenbrock23 observed order {order:.2} outside [1.8, 2.3]"
    );
}

/// The `ode23s` embedded estimate must scale as the **`O(h³)` local error** of the 2nd-order
/// solution — i.e. shrink ~8× per halving.
///
/// This is what pins the stage-3 time-derivative term to `h·γ·f_t`. Scaling it by `h`
/// instead (as some implementations do) leaves the solution identical but makes the estimate
/// scale ~4× per halving, i.e. an `O(h²)` estimate of an `O(h³)` error: the controller then
/// over-estimates the error on every non-autonomous problem and takes needlessly small steps.
#[test]
fn ros23_error_estimator_is_third_order() {
    let (_, e_coarse) = fixed_step_run(OdeMethod::Rosenbrock23, 50);
    let (_, e_mid) = fixed_step_run(OdeMethod::Rosenbrock23, 100);
    let (_, e_fine) = fixed_step_run(OdeMethod::Rosenbrock23, 200);
    for (a, b) in [(e_coarse, e_mid), (e_mid, e_fine)] {
        let ratio = a / b;
        assert!(
            ratio > 5.5,
            "estimator shrank only {ratio:.2}× per halving — an O(h²) estimate (~4×) means \
             the stage-3 f_t term lost its γ factor"
        );
    }
}

/// Verner 7(6) must converge at ~7th order — the entire reason it exists, since its value over
/// RK45 comes from order, not stability. A mistyped tableau digit collapses this.
#[test]
fn vern7_converges_at_seventh_order() {
    // Measured from h = 1/30 down. The order problem is mildly stiff (λ = −50), and Vern7 is
    // *explicit*: at coarser steps `|λh|` leaves its stability region and the run diverges,
    // which is the very distinction this method exists to make — it buys order, not stability.
    // Two refinements only, because an order-7 method reaches round-off almost immediately.
    // Measures 7.69 on this problem; the band is set around that rather than left wide.
    let order = observed_order(OdeMethod::Vern7, 60, 2);
    assert!(
        (6.5..8.2).contains(&order),
        "Vern7 observed order {order:.2} outside [6.5, 8.2]"
    );
    // At a step size where RK45 is still far off, the higher order must already show: this is
    // the accuracy-per-step advantage the tight-tolerance regime buys.
    let (err7, _) = fixed_step_run(OdeMethod::Vern7, 50);
    let (err45, _) = fixed_step_run(OdeMethod::Rk45, 50);
    assert!(
        err7 < err45,
        "Vern7 ({err7:.3e}) should beat RK45 ({err45:.3e}) per step at h=0.04"
    );
}

/// **Structural** verification of the transcribed Rosenbrock tableaus, independent of any
/// trajectory.
///
/// A Rosenbrock method is usually *published* as `(α_ij, Γ_ij, b_i)` but *implemented* in the
/// transformed form this module stores, `A = α·Γ⁻¹` and `C = diag(1/γ) − Γ⁻¹`. Inverting that
/// relation recovers the mathematical tableau from the implementation one, and the recovered
/// tableau has to satisfy the two definitions that fix the stage times and the `f_t` weights:
///
/// ```text
/// Σ_j α_ij = α_i  (the stage time)      Σ_j Γ_ij = γ_i  (the `h·γ_i·f_t` coefficient)
/// ```
///
/// Both are checked here without inverting anything: `Γ·1` is the solution of `Γ⁻¹ x = 1`, and
/// `Γ⁻¹ = diag(1/γ) − C` is lower triangular, so one forward substitution gives `x = Γ·1`.
/// Then `x` must equal `d`, and `A·x` must equal `alpha`.
///
/// This is what a measured convergence order cannot do: order is a property of the *whole*
/// method, so a compensating pair of errors could in principle survive it, and a coefficient
/// the test problem barely excites might not move it. This binds `a`, `c`, `alpha`, `d` and
/// `gamma` to each other elementwise — a single mistyped digit in any of the five breaks it by
/// many orders of magnitude.
#[test]
fn rosenbrock_tableaus_satisfy_their_defining_identities() {
    for (name, tab) in [("Rodas4", &RODAS4), ("Rodas5P", &RODAS5P)] {
        let s = tab.stages;
        // Γ⁻¹ = diag(1/γ) − C, lower triangular. Solve Γ⁻¹ x = 1 for x = Γ·1.
        let mut x = vec![0.0_f64; s];
        for i in 0..s {
            let mut rhs = 1.0;
            for j in 0..i {
                rhs -= -tab.c[i][j] * x[j]; // Γ⁻¹[i][j] = −C[i][j] off the diagonal
            }
            x[i] = rhs / (1.0 / tab.gamma);
        }

        for i in 0..s {
            // Σ_j Γ_ij == γ_i (stored as `d`).
            assert_relative_eq!(x[i], tab.d[i], epsilon = 1e-13);
            // Σ_j α_ij == α_i (stored as `alpha`), with α = A·Γ so Σ_j α_ij = (A·x)_i.
            let ax: f64 = (0..s).map(|j| tab.a[i][j] * x[j]).sum();
            assert_relative_eq!(ax, tab.alpha[i], epsilon = 1e-13);
        }
        println!("{name}: stage-time and γ_i identities hold for all {s} stages");
    }
}

// ---------------------------------------------------------------------------------------
// Agreement with the RK45 production path
// ---------------------------------------------------------------------------------------

/// Oral two-compartment PK: `p = [CL, V1, Q, V2, KA]`, states `[depot, central, periph]`.
fn pk_rhs<T: PkNum>(u: &[T], p: &[T], _t: f64, du: &mut [T]) {
    let (cl, v1, q, v2, ka) = (p[0], p[1], p[2], p[3], p[4]);
    du[0] = -ka * u[0];
    du[1] = ka * u[0] - (cl / v1) * u[1] - (q / v1) * u[1] + (q / v2) * u[2];
    du[2] = (q / v1) * u[1] - (q / v2) * u[2];
}

const PK_PARAMS: [f64; 5] = [3.0, 20.0, 2.0, 50.0, 1.0];
const PK_U0: [f64; 3] = [100.0, 0.0, 0.0];

fn pk_saveat() -> Vec<f64> {
    vec![0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
}

/// On a **non-stiff** model every method must reproduce the RK45 trajectory: the stiff
/// steppers are an alternative route to the same answer, not a different model.
#[test]
fn stiff_methods_match_rk45_on_a_nonstiff_pk_model() {
    let saveat = pk_saveat();
    let reference = solve_ode(
        &pk_rhs::<f64>,
        &PK_U0,
        (0.0, 24.0),
        &PK_PARAMS,
        &saveat,
        &OdeSolverOptions {
            abstol: 1e-12,
            reltol: 1e-10,
            ..Default::default()
        },
    );

    // (method, reltol, abstol, max_steps, tolerance vs RK45). Rosenbrock23 is 2nd order, so
    // matching a 1e-10 reference would cost ~10⁵ steps; it is checked at its own scale.
    let cases = [
        (OdeMethod::Rodas4, 1e-10, 1e-12, 10_000, 1e-7),
        (OdeMethod::Rodas5P, 1e-10, 1e-12, 10_000, 1e-7),
        (OdeMethod::Rosenbrock23, 1e-8, 1e-10, 500_000, 1e-4),
    ];
    for (method, reltol, abstol, max_steps, tol) in cases {
        let got = solve_ode(
            &pk_rhs::<f64>,
            &PK_U0,
            (0.0, 24.0),
            &PK_PARAMS,
            &saveat,
            &OdeSolverOptions {
                abstol,
                reltol,
                max_steps,
                method,
                ..Default::default()
            },
        );
        assert_eq!(got.len(), saveat.len(), "{method:?} lost save points");
        for (g, r) in got.iter().zip(reference.iter()) {
            assert_relative_eq!(g.t, r.t, epsilon = 1e-12);
            for (a, b) in g.u.iter().zip(r.u.iter()) {
                assert_relative_eq!(a, b, max_relative = tol, epsilon = 1e-9);
            }
        }
    }
}

/// The `saveat` contract is shared with RK45: one point per requested time, in order, and a
/// zero-width span short-circuits to `u0` at every requested time.
#[test]
fn saveat_contract_matches_rk45() {
    let saveat = pk_saveat();
    let opts = OdeSolverOptions {
        method: OdeMethod::Rodas4,
        ..Default::default()
    };
    let got = solve_ode(
        &pk_rhs::<f64>,
        &PK_U0,
        (0.0, 24.0),
        &PK_PARAMS,
        &saveat,
        &opts,
    );
    assert_eq!(got.len(), saveat.len());
    for (g, &t) in got.iter().zip(saveat.iter()) {
        assert_relative_eq!(g.t, t, epsilon = 1e-12);
        assert_eq!(g.u.len(), PK_U0.len());
    }

    let zero_span = solve_ode(
        &pk_rhs::<f64>,
        &PK_U0,
        (3.0, 3.0),
        &PK_PARAMS,
        &saveat,
        &opts,
    );
    assert_eq!(zero_span.len(), saveat.len());
    for p in &zero_span {
        assert_eq!(p.u, PK_U0.to_vec());
    }
}

/// One driver, two instantiations: the `f64` scalar path and the generic (`solve_ode_g`)
/// path now share a single loop and a single stepper, so at `T = f64` they must agree **bit
/// for bit**. This is the property that makes "wire a consumer once, get every method" true —
/// if the two ever diverged, a consumer would silently get a different trajectory depending
/// on which entry point it happened to call.
#[test]
fn scalar_and_generic_drivers_agree_bit_for_bit() {
    let saveat = pk_saveat();
    for method in [
        OdeMethod::Rk45,
        OdeMethod::Vern7,
        OdeMethod::Rosenbrock23,
        OdeMethod::Rodas4,
        OdeMethod::Rodas5P,
    ] {
        let opts = OdeSolverOptions {
            method,
            ..Default::default()
        };
        let scalar = solve_ode(
            &pk_rhs::<f64>,
            &PK_U0,
            (0.0, 24.0),
            &PK_PARAMS,
            &saveat,
            &opts,
        );
        let generic = crate::ode::solver::solve_ode_g::<f64>(
            &pk_rhs::<f64>,
            &PK_U0,
            (0.0, 24.0),
            &PK_PARAMS,
            &saveat,
            &opts,
        );
        assert_eq!(scalar.len(), generic.len(), "{method:?}");
        for (a, b) in scalar.iter().zip(generic.iter()) {
            assert_eq!(a.u, b.u, "{method:?} scalar vs generic driver");
        }
    }
}

/// The continuous extension must be **exact at both ends** of every step — `Θ = 0` returns the
/// step's start state and `Θ = 1` its end state — or interpolated readouts would disagree with
/// saved states at step boundaries. Checked through the public dense entry point by asking for
/// soft samples exactly at `saveat` times.
#[test]
fn interpolant_is_continuous_with_the_saved_states() {
    let saveat = pk_saveat();
    for method in [
        OdeMethod::Rk45,
        OdeMethod::Vern7,
        OdeMethod::Rosenbrock23,
        OdeMethod::Rodas4,
        OdeMethod::Rodas5P,
    ] {
        let opts = OdeSolverOptions {
            method,
            ..Default::default()
        };
        let (hard, soft) = crate::ode::solver::solve_ode_dense(
            &pk_rhs::<f64>,
            &PK_U0,
            (0.0, 24.0),
            &PK_PARAMS,
            &saveat,
            &saveat,
            &opts,
            None,
        );
        for (h, s) in hard.iter().zip(soft.iter()) {
            for (x, y) in h.u.iter().zip(s.u.iter()) {
                // Θ lands on 1.0 exactly at a save, so this is an identity, not an
                // approximation — a loose interpolant would show up immediately.
                assert_relative_eq!(x, y, max_relative = 1e-12, epsilon = 1e-14);
            }
        }
    }
}

/// Soft sampling must **not perturb the trajectory it samples** (#570): the hard saves have to
/// be bit-identical whether or not `interp_at` is requested — for every method, since the
/// interpolation is now the stepper's own continuous extension rather than extra output times.
/// And the interpolated values must be accurate, not merely continuous: each is compared
/// against a solve that lands on that time exactly.
#[test]
fn soft_sampling_does_not_perturb_the_step_sequence() {
    let saveat = pk_saveat();
    let interp_at = vec![0.4, 1.7, 3.3, 9.5, 20.0];
    for method in [
        OdeMethod::Rk45,
        OdeMethod::Vern7,
        OdeMethod::Rosenbrock23,
        OdeMethod::Rodas4,
        OdeMethod::Rodas5P,
    ] {
        let opts = OdeSolverOptions {
            method,
            ..Default::default()
        };
        let (hard_with, soft) = crate::ode::solver::solve_ode_dense(
            &pk_rhs::<f64>,
            &PK_U0,
            (0.0, 24.0),
            &PK_PARAMS,
            &saveat,
            &interp_at,
            &opts,
            None,
        );
        let without = solve_ode(
            &pk_rhs::<f64>,
            &PK_U0,
            (0.0, 24.0),
            &PK_PARAMS,
            &saveat,
            &opts,
        );
        for (a, b) in hard_with.iter().zip(without.iter()) {
            assert_eq!(
                a.u, b.u,
                "{method:?}: requesting soft samples changed the hard saves"
            );
        }

        assert_eq!(soft.len(), interp_at.len());

        // Accuracy of the interpolant itself, isolated from solver error: run *both* sides at
        // a tight tolerance, so what remains is the continuous extension's own error against a
        // solve that lands on those times exactly.
        let tight = OdeSolverOptions {
            abstol: 1e-12,
            reltol: 1e-10,
            method,
            ..Default::default()
        };
        let (_, soft_tight) = crate::ode::solver::solve_ode_dense(
            &pk_rhs::<f64>,
            &PK_U0,
            (0.0, 24.0),
            &PK_PARAMS,
            &saveat,
            &interp_at,
            &tight,
            None,
        );
        let landed = solve_ode(
            &pk_rhs::<f64>,
            &PK_U0,
            (0.0, 24.0),
            &PK_PARAMS,
            &interp_at,
            &tight,
        );
        // Interpolant accuracy is a property of the *continuous extension*, not of the step.
        // Rodas4/Rodas5P carry order-matched extensions and RK45's cubic Hermite matches its
        // own order well enough; Vern7 reads through the same cubic Hermite while its steps
        // are 7th-order accurate, so at tight tolerance its (larger) steps leave a visibly
        // bigger in-step error. That is a documented trade, not a defect — see
        // `crate::ode::verner`.
        // Vern7's continuous extension is the same cubic Hermite as RK45's and measures the
        // same order (`interpolant_orders_pin_the_dense_output_blocks`: 4.3 vs 4.9). What
        // differs is step *size* — at this tolerance Vern7 takes 2.8× fewer, so the same
        // relative position inside a step is a larger absolute distance, and its in-step error
        // lands near 1e-6 where the others sit below 1e-7. Bound set just above measured, not
        // waved through.
        let interp_tol = match method {
            OdeMethod::Vern7 => 1e-5,
            _ => 1e-6,
        };
        for (s, e) in soft_tight.iter().zip(landed.iter()) {
            for (x, y) in s.u.iter().zip(e.u.iter()) {
                assert_relative_eq!(x, y, max_relative = interp_tol, epsilon = 1e-9);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// The point of the exercise: a stiff system
// ---------------------------------------------------------------------------------------

/// Fast reversible binding on top of slow elimination — the TMDD/quasi-equilibrium shape.
/// `k_on`/`k_off` put an eigenvalue near `−1.1e4` while the observable dynamics run over
/// hours, so the explicit method is stability-limited, not accuracy-limited.
fn binding_rhs(u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]) {
    const K_ON: f64 = 1.0e4;
    const K_OFF: f64 = 1.0e3;
    const K_EL: f64 = 0.1;
    du[0] = -K_ON * u[0] + K_OFF * u[1] - K_EL * u[0];
    du[1] = K_ON * u[0] - K_OFF * u[1];
}

/// On the stiff system the Rosenbrock methods finish inside a normal step budget and agree
/// with a brute-force RK45 reference, while RK45 at the same budget cannot even reach the
/// end of the span — the failure mode is silent (the tail is freeze-padded), which is
/// exactly why a stiff option had to exist.
#[test]
fn rosenbrock_solves_a_stiff_binding_system_that_exhausts_rk45() {
    let saveat = vec![1.0, 6.0, 24.0];
    let u0 = [1.0, 0.0];

    // Brute-force explicit reference: same method as production, just given the (very large)
    // step budget stability demands.
    let reference = solve_ode(
        &binding_rhs,
        &u0,
        (0.0, 24.0),
        &[],
        &saveat,
        &OdeSolverOptions {
            abstol: 1e-14,
            reltol: 1e-11,
            max_steps: 5_000_000,
            ..Default::default()
        },
    );

    // RK45 on the production budget: stability, not accuracy, caps the step, so it burns the
    // whole budget without reaching t = 24.
    //
    // Named explicitly rather than taken from `Default`, which is `auto` since #978 — this
    // system is stiff enough that the probe escalates it, which is the whole point of `auto`
    // and the exact opposite of what this test needs to observe.
    let mut rk_stats = OdeSolverStats::default();
    let rk = solve_ode_with_stats(
        &binding_rhs,
        &u0,
        (0.0, 24.0),
        &[],
        &saveat,
        &OdeSolverOptions {
            abstol: 1e-10,
            reltol: 1e-8,
            method: OdeMethod::Rk45,
            ..Default::default()
        },
        Some(&mut rk_stats),
    );
    assert_eq!(
        rk_stats.attempted_steps, 10_000,
        "RK45 was expected to exhaust its step budget on this stiff system"
    );
    let rk_final_err = (rk[2].u[1] - reference[2].u[1]).abs() / reference[2].u[1].abs();
    assert!(
        rk_final_err > 1e-3,
        "RK45 unexpectedly landed within 0.1% at t=24 ({rk_final_err:.2e}) — the stiffness \
         premise of this test no longer holds"
    );

    for method in [
        OdeMethod::Rosenbrock23,
        OdeMethod::Rodas4,
        OdeMethod::Rodas5P,
    ] {
        let mut stats = OdeSolverStats::default();
        let got = solve_ode_with_stats(
            &binding_rhs,
            &u0,
            (0.0, 24.0),
            &[],
            &saveat,
            &OdeSolverOptions {
                abstol: 1e-10,
                reltol: 1e-8,
                method,
                ..Default::default()
            },
            Some(&mut stats),
        );
        assert!(
            stats.attempted_steps < 5_000,
            "{method:?} took {} steps on a 2-state stiff system",
            stats.attempted_steps
        );
        assert_eq!(stats.min_step_clamped_steps, 0, "{method:?} hit min_dt");
        for (g, r) in got.iter().zip(reference.iter()) {
            for (a, b) in g.u.iter().zip(r.u.iter()) {
                assert_relative_eq!(a, b, max_relative = 1e-4, epsilon = 1e-12);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// Sensitivity path
// ---------------------------------------------------------------------------------------

/// The `Dual2` instantiation must reproduce the `f64` trajectory in its value part: every
/// branch (LU pivot, accept/reject, step size) reads `.val()` only, so both runs take the
/// *same* step and pivot sequence.
///
/// Agreement is to solver precision rather than bit-for-bit because `Dual2` defines division
/// as `× recip`, one extra rounding per divide — the same reason the RK45 path's
/// `solve_ode_g_value_matches_scalar` compares at `1e-9`.
#[test]
fn dual_value_part_matches_the_scalar_solve() {
    let saveat = pk_saveat();
    let opts = OdeSolverOptions {
        abstol: 1e-12,
        reltol: 1e-10,
        method: OdeMethod::Rodas4,
        ..Default::default()
    };
    let scalar = solve_ode(
        &pk_rhs::<f64>,
        &PK_U0,
        (0.0, 24.0),
        &PK_PARAMS,
        &saveat,
        &opts,
    );

    let params: Vec<Dual2<2>> = vec![
        Dual2::<2>::var(PK_PARAMS[0], 0),
        Dual2::<2>::from_f64(PK_PARAMS[1]),
        Dual2::<2>::from_f64(PK_PARAMS[2]),
        Dual2::<2>::from_f64(PK_PARAMS[3]),
        Dual2::<2>::var(PK_PARAMS[4], 1),
    ];
    let u0: Vec<Dual2<2>> = PK_U0.iter().map(|&x| Dual2::<2>::from_f64(x)).collect();
    let dual = crate::ode::solver::solve_ode_g(
        &pk_rhs::<Dual2<2>>,
        &u0,
        (0.0, 24.0),
        &params,
        &saveat,
        &opts,
    );

    assert_eq!(dual.len(), scalar.len());
    for (d, s) in dual.iter().zip(scalar.iter()) {
        for (a, b) in d.u.iter().zip(s.u.iter()) {
            assert_relative_eq!(a.value, b, max_relative = 1e-9, epsilon = 1e-12);
        }
    }
}

/// `∂u/∂p` carried by the duals must match a finite difference of the scalar solve. This is
/// the property the whole method choice was made for: no Newton iteration means the step is
/// straight-line arithmetic, so differentiating it is exact for the step actually taken.
#[test]
fn dual_gradients_match_finite_differences_of_the_scalar_solve() {
    let saveat = pk_saveat();
    let opts = OdeSolverOptions {
        abstol: 1e-12,
        reltol: 1e-10,
        method: OdeMethod::Rodas5P,
        ..Default::default()
    };

    let params: Vec<Dual2<2>> = vec![
        Dual2::<2>::var(PK_PARAMS[0], 0),
        Dual2::<2>::from_f64(PK_PARAMS[1]),
        Dual2::<2>::from_f64(PK_PARAMS[2]),
        Dual2::<2>::from_f64(PK_PARAMS[3]),
        Dual2::<2>::var(PK_PARAMS[4], 1),
    ];
    let u0: Vec<Dual2<2>> = PK_U0.iter().map(|&x| Dual2::<2>::from_f64(x)).collect();
    let dual = crate::ode::solver::solve_ode_g(
        &pk_rhs::<Dual2<2>>,
        &u0,
        (0.0, 24.0),
        &params,
        &saveat,
        &opts,
    );

    // Central FD on CL (dim 0) and KA (dim 1).
    for (dim, p_idx) in [(0usize, 0usize), (1, 4)] {
        let h = 1e-4 * PK_PARAMS[p_idx];
        let mut lo = PK_PARAMS;
        let mut hi = PK_PARAMS;
        lo[p_idx] -= h;
        hi[p_idx] += h;
        let sol_lo = solve_ode(&pk_rhs::<f64>, &PK_U0, (0.0, 24.0), &lo, &saveat, &opts);
        let sol_hi = solve_ode(&pk_rhs::<f64>, &PK_U0, (0.0, 24.0), &hi, &saveat, &opts);
        for (k, d) in dual.iter().enumerate() {
            for state in 0..PK_U0.len() {
                let fd = (sol_hi[k].u[state] - sol_lo[k].u[state]) / (2.0 * h);
                assert_relative_eq!(
                    d.u[state].grad[dim],
                    fd,
                    max_relative = 1e-4,
                    epsilon = 1e-8
                );
            }
        }
    }
}

/// The sensitivity path gets in-step readouts too: `solve_ode_g_dense` is the `Dual2` twin of
/// `solve_ode_dense`, so a consumer that needs an interpolated state **with its derivative**
/// — a hazard value at an event time, say — has one to call, for every method. This is the
/// concrete form of "a new consumer written against the driver works everywhere".
#[test]
fn dual_dense_path_interpolates_values_and_derivatives() {
    let saveat = pk_saveat();
    let interp_at = vec![0.9, 5.5, 17.25];
    for method in [OdeMethod::Rk45, OdeMethod::Rodas5P] {
        let opts = OdeSolverOptions {
            abstol: 1e-12,
            reltol: 1e-10,
            method,
            ..Default::default()
        };
        let params: Vec<Dual2<2>> = vec![
            Dual2::<2>::var(PK_PARAMS[0], 0),
            Dual2::<2>::from_f64(PK_PARAMS[1]),
            Dual2::<2>::from_f64(PK_PARAMS[2]),
            Dual2::<2>::from_f64(PK_PARAMS[3]),
            Dual2::<2>::var(PK_PARAMS[4], 1),
        ];
        let u0: Vec<Dual2<2>> = PK_U0.iter().map(|&x| Dual2::<2>::from_f64(x)).collect();
        let (_, soft) = crate::ode::solver::solve_ode_g_dense(
            &pk_rhs::<Dual2<2>>,
            &u0,
            (0.0, 24.0),
            &params,
            &saveat,
            &interp_at,
            &opts,
            None,
        );
        assert_eq!(soft.len(), interp_at.len(), "{method:?}");

        // Value part: the same interpolated state the scalar dense path returns.
        let (_, soft_scalar) = crate::ode::solver::solve_ode_dense(
            &pk_rhs::<f64>,
            &PK_U0,
            (0.0, 24.0),
            &PK_PARAMS,
            &saveat,
            &interp_at,
            &opts,
            None,
        );
        for (a, b) in soft.iter().zip(soft_scalar.iter()) {
            for (x, y) in a.u.iter().zip(b.u.iter()) {
                assert_relative_eq!(x.value, y, max_relative = 1e-9, epsilon = 1e-12);
            }
        }

        // Derivative part: ∂/∂CL of the *interpolated* state, against central FD of the scalar
        // interpolated state — i.e. the jet really rides through the continuous extension.
        let hstep = 1e-4 * PK_PARAMS[0];
        let mut lo = PK_PARAMS;
        let mut hi = PK_PARAMS;
        lo[0] -= hstep;
        hi[0] += hstep;
        let soft_at = |p: &[f64; 5]| {
            crate::ode::solver::solve_ode_dense(
                &pk_rhs::<f64>,
                &PK_U0,
                (0.0, 24.0),
                p,
                &saveat,
                &interp_at,
                &opts,
                None,
            )
            .1
        };
        let (s_lo, s_hi) = (soft_at(&lo), soft_at(&hi));
        for (k, p) in soft.iter().enumerate() {
            for state in 0..PK_U0.len() {
                let fd = (s_hi[k].u[state] - s_lo[k].u[state]) / (2.0 * hstep);
                // The absolute floor is set by FD noise, not by the derivative: `depot` does
                // not depend on CL at all, so the analytic jet is exactly 0 while its finite
                // difference is round-off over `2h` (~5e-8 here). The relative bound is what
                // checks the two states that genuinely depend on CL.
                assert_relative_eq!(p.u[state].grad[0], fd, max_relative = 1e-4, epsilon = 1e-6);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// Failure handling
// ---------------------------------------------------------------------------------------

/// A right-hand side that goes non-finite makes the Jacobian — and therefore `W` — singular.
/// The stepper must give up and freeze-pad the remaining save points rather than spin to the
/// step budget or advance on a garbage stage solve. Mirrors RK45's divergence break.
#[test]
fn non_finite_rhs_terminates_and_pads_the_tail() {
    let rhs = |u: &[f64], _p: &[f64], t: f64, du: &mut [f64]| {
        du[0] = if t > 0.5 { f64::NAN } else { -u[0] };
    };
    let saveat = vec![0.25, 1.0, 2.0];
    let mut stats = OdeSolverStats::default();
    let got = solve_ode_with_stats(
        &rhs,
        &[1.0],
        (0.0, 2.0),
        &[],
        &saveat,
        &OdeSolverOptions {
            method: OdeMethod::Rodas4,
            ..Default::default()
        },
        Some(&mut stats),
    );
    assert_eq!(got.len(), saveat.len(), "tail must still be padded");
    assert!(
        stats.attempted_steps < 10_000,
        "expected an early break, not a full step budget"
    );
    // The pre-divergence save is still the correct decay value.
    assert_relative_eq!(got[0].u[0], (-0.25_f64).exp(), max_relative = 1e-4);
}

/// A zero-state system is a degenerate but reachable configuration (a model whose ODE block
/// compiles to nothing); it must not panic in workspace allocation or the LU.
#[test]
fn empty_state_vector_is_handled() {
    let rhs = |_u: &[f64], _p: &[f64], _t: f64, _du: &mut [f64]| {};
    let saveat = vec![1.0, 2.0];
    let got = solve_ode(
        &rhs,
        &[],
        (0.0, 2.0),
        &[],
        &saveat,
        &OdeSolverOptions {
            method: OdeMethod::Rodas5P,
            ..Default::default()
        },
    );
    assert_eq!(got.len(), saveat.len());
    assert!(got.iter().all(|p| p.u.is_empty()));
}

/// `[fit_options] ode_method` must reach the compiled model's solver options — the parse-time
/// `sync_ode_solver_opts` hop that `predict()` (which receives no fit options) depends on.
#[test]
fn fit_option_reaches_the_compiled_solver_options() {
    let src = "[parameters]
  theta TVCL(1.0, 0.01, 10.0)
  theta TVV(10.0, 0.1, 100.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_method = rodas4
";
    let model = crate::parser::model_parser::parse_full_model(src)
        .expect("model must parse")
        .model;
    assert_eq!(
        model.ode_spec.as_ref().unwrap().solver_opts.method,
        OdeMethod::Rodas4
    );
}

/// The TTE **event-time sampler** localizes a hazard crossing *inside* a step. It now does so
/// through whatever stepper is in use — bisecting that method's own continuous extension — so
/// it works for every method with no per-method wiring. The accumulator crosses `1.0` at
/// exactly `t = 2`, which every method must find.
#[test]
fn event_time_sampler_works_for_every_method() {
    use crate::ode::solver::{solve_ode_until_threshold, ThresholdCrossing};
    // A pure accumulator: CHZ grows at a constant rate, so it crosses 1.0 at t = 2.
    let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = 0.5;
    };
    for method in [
        OdeMethod::Rk45,
        OdeMethod::Vern7,
        OdeMethod::Rosenbrock23,
        OdeMethod::Rodas4,
        OdeMethod::Rodas5P,
    ] {
        let opts = OdeSolverOptions {
            method,
            ..Default::default()
        };
        let mut u = [0.0];
        match solve_ode_until_threshold(&rhs, &mut u, (0.0, 10.0), &[], &opts, 0, 1.0) {
            ThresholdCrossing::Crossed(t) => {
                assert_relative_eq!(t, 2.0, max_relative = 1e-6);
            }
            other => panic!("{method:?} must locate the crossing, got {other:?}"),
        }
        // …and a threshold beyond the horizon still reports a clean non-crossing.
        let mut u = [0.0];
        assert_eq!(
            solve_ode_until_threshold(&rhs, &mut u, (0.0, 1.0), &[], &opts, 0, 99.0),
            ThresholdCrossing::ReachedEnd,
            "{method:?}"
        );
    }
}

/// `OdeMethod::parse` is the `[fit_options] ode_method` surface: aliases resolve, unknown
/// tokens are rejected (rather than silently falling back to RK45), and every variant
/// round-trips through its canonical token.
#[test]
fn method_tokens_parse_and_round_trip() {
    assert_eq!(OdeMethod::parse("RK45"), Some(OdeMethod::Rk45));
    assert_eq!(OdeMethod::parse("ode23s"), Some(OdeMethod::Rosenbrock23));
    assert_eq!(OdeMethod::parse("ros23"), Some(OdeMethod::Rosenbrock23));
    assert_eq!(OdeMethod::parse("Rodas4"), Some(OdeMethod::Rodas4));
    assert_eq!(OdeMethod::parse("rodas5p"), Some(OdeMethod::Rodas5P));
    assert_eq!(OdeMethod::parse("vern7"), Some(OdeMethod::Vern7));
    assert_eq!(OdeMethod::parse("verner7"), Some(OdeMethod::Vern7));
    assert_eq!(OdeMethod::parse("lsoda"), None);
    assert_eq!(OdeMethod::parse("auto"), Some(OdeMethod::Auto));
    // The default is the probe (#978); the *fallback* it retreats to is RK45. These were the
    // same value until `auto` became the default and must not be conflated again — reading
    // `default()` as "the explicit method" is what would make the guard fall back to `auto`.
    assert_eq!(OdeMethod::default(), OdeMethod::Auto);
    assert_eq!(OdeMethod::EXPLICIT_FALLBACK, OdeMethod::Rk45);
    for m in [
        OdeMethod::Rk45,
        OdeMethod::Vern7,
        OdeMethod::Rosenbrock23,
        OdeMethod::Rodas4,
        OdeMethod::Rodas5P,
        OdeMethod::Auto,
    ] {
        assert_eq!(OdeMethod::parse(m.as_str()), Some(m));
    }
}

// ---------------------------------------------------------------------------------------
// Review #952: the stall diagnostic, and the two evaluation-count invariants
// ---------------------------------------------------------------------------------------

/// A right-hand side that is `NaN` everywhere, so `J` is `NaN`, `W = I/(γh) − J` fails to
/// factor, and no step can ever be formed at any `dt`.
fn nan_rhs(_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]) {
    du[0] = f64::NAN;
}

/// A Rosenbrock integration that stalls — `W` unusable even at `min_dt` — must say so in
/// [`OdeSolverStats::min_step_clamped_steps`].
///
/// This is the failure mode with no other tell. The driver stops and freeze-pads the
/// remaining `saveat` points with the last state, so the returned trajectory is finite,
/// positive and entirely plausible; the stats block is the only place the stall is visible,
/// and `min_step_clamped_steps` is the counter the `ode_method` docs send users to. Counting
/// the stall as a plain rejection (as it once was) left a stalled run presenting the same
/// clean `min_step_clamped_steps == 0` a healthy one does.
#[test]
fn a_stalled_rosenbrock_step_reports_a_min_step_clamp() {
    for method in [
        OdeMethod::Rosenbrock23,
        OdeMethod::Rodas4,
        OdeMethod::Rodas5P,
    ] {
        let mut stats = OdeSolverStats::default();
        let saveat = [0.5, 1.0];
        let got = solve_ode_with_stats(
            &nan_rhs,
            &[1.0],
            (0.0, 1.0),
            &[],
            &saveat,
            &OdeSolverOptions {
                method,
                ..Default::default()
            },
            Some(&mut stats),
        );

        // The tail is padded, not dropped — which is exactly why the stats have to speak.
        assert_eq!(got.len(), saveat.len(), "[{method:?}] saveat count");
        assert!(
            stats.min_step_clamped_steps > 0,
            "[{method:?}] a stall that freeze-pads the tail must be visible in \
             min_step_clamped_steps, got {stats:?}"
        );
    }
}

/// Counting harness: wrap a right-hand side and tally how many times it is evaluated.
fn counting_rhs<'a>(
    calls: &'a std::cell::Cell<usize>,
) -> impl Fn(&[f64], &[f64], f64, &mut [f64]) + 'a {
    move |u: &[f64], p: &[f64], t: f64, du: &mut [f64]| {
        calls.set(calls.get() + 1);
        order_rhs(u, p, t, du);
    }
}

/// A **rejected** Rosenbrock step must not re-derive `f0`, `J` and `f_t`.
///
/// A rejection leaves `(u, t)` exactly where they were and changes only `dt`, so those
/// `1 + n + 1` evaluations would reproduce the same numbers at full price — `rodas.f` and
/// `OrdinaryDiffEq.jl` both hold `J` across a retry. Only the `1/(γh)` diagonal and the LU
/// genuinely depend on `dt`.
#[test]
fn a_rejected_step_reuses_the_jacobian() {
    const N: usize = 2;
    for method in [
        OdeMethod::Rosenbrock23,
        OdeMethod::Rodas4,
        OdeMethod::Rodas5P,
    ] {
        let calls = std::cell::Cell::new(0usize);
        let rhs = counting_rhs(&calls);
        let mut stepper = crate::ode::solver::make_stepper::<f64>(N, method);
        let opts = OdeSolverOptions {
            method,
            ..Default::default()
        };
        let u = order_exact(0.3).to_vec();

        stepper.attempt(&rhs, &u, &[], 0.3, 1e-3, &opts);
        let first = calls.replace(0);

        // Same `(u, t)`, smaller `dt` — precisely a rejected step's retry. No `on_accept`.
        stepper.attempt(&rhs, &u, &[], 0.3, 5e-4, &opts);
        let retry = calls.get();

        assert_eq!(
            first - retry,
            N + 2,
            "[{method:?}] a retry should save exactly the f0 + {N}-column J + f_t \
             evaluations ({} → {retry})",
            first
        );
    }
}

/// On the dense path a non-FSAL pair must carry its end derivative into the next step.
///
/// `f(u_new, t + h)` is bought for the interpolant, and `(u_new, t + h)` *is* the next step's
/// `(u, t)` — so re-deriving it costs one extra `f` on every accepted step of every in-step
/// readout (TTE hazards, CTMM occupancy, adaptive-dosing monitors, `[output]` columns). For
/// Vern7 that is 11 evaluations per step instead of 10.
#[test]
fn vern7_carries_its_end_derivative_across_an_accepted_dense_step() {
    let calls = std::cell::Cell::new(0usize);
    let rhs = counting_rhs(&calls);
    let mut stepper = crate::ode::solver::make_stepper::<f64>(2, OdeMethod::Vern7);
    stepper.set_dense_required(true);
    let opts = OdeSolverOptions {
        method: OdeMethod::Vern7,
        ..Default::default()
    };

    let mut u = order_exact(0.3).to_vec();
    stepper.attempt(&rhs, &u, &[], 0.3, 1e-3, &opts);
    let first = calls.replace(0);
    u.copy_from_slice(stepper.u_new());
    stepper.on_accept();

    stepper.attempt(&rhs, &u, &[], 0.301, 1e-3, &opts);
    let second = calls.get();

    // 10 stages + the end derivative on the first step; the second inherits its `k1`.
    assert_eq!(first, 11, "first dense step: 10 stages + f_end");
    assert_eq!(
        second, 10,
        "the accepted step's f_end is the next step's k1, so the second step must not \
         re-evaluate it"
    );
}

/// The Rosenbrock helpers spell out their non-Rosenbrock arms rather than ending in a
/// catch-all, so that adding an [`OdeMethod`] fails to compile here instead of routing silently
/// into this stepper. `Auto` joined those arms in #978 — every driver resolves it to a concrete
/// method first, so reaching them means a caller bypassed the resolution, and the contract is
/// that it fails loudly at the boundary rather than integrating something wrong.
#[test]
fn the_rosenbrock_stepper_refuses_auto_loudly() {
    for build in [
        || {
            let _ = RosStepper::<f64>::new(1, OdeMethod::Auto);
        },
        || {
            let _ = method_gamma(OdeMethod::Auto);
        },
        || {
            let _ = method_err_exp(OdeMethod::Auto);
        },
    ] {
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(build));
        assert!(
            caught.is_err(),
            "a non-Rosenbrock method must not be accepted by the Rosenbrock stepper"
        );
    }
}
