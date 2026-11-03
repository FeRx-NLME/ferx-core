use super::*;
use crate::ode::solver::{solve_ode, OdeMethod, OdeSolverOptions};
use crate::sens::dual2::Dual2;

/// nalgebra's own `exp`. It runs the **same** algorithm (degree-13 Padé, Higham 2005) — it
/// matches `expm_g` bit-for-bit on two of the four fixtures below — so it checks the wiring
/// (coefficients, scaling rule, squaring loop), not the method. The closed-form tests below are
/// the independent references.
fn nalgebra_expm(a: &[f64], n: usize) -> Vec<f64> {
    let m = nalgebra::DMatrix::from_row_slice(n, n, a).exp();
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = m[(i, j)];
        }
    }
    out
}

fn max_rel_err(got: &[f64], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0.0_f64, |s, x| s.max(x.abs())).max(1e-300);
    got.iter().zip(want).fold(0.0_f64, |worst, (g, w)| {
        // Both sides: `worst.max(NaN)` keeps `worst`.
        assert!(
            g.is_finite() && w.is_finite(),
            "non-finite entry: got {g}, want {w}"
        );
        worst.max((g - w).abs() / scale)
    })
}

/// Two-compartment micro-constant generator `[[-(k10+k12), k21], [k12, -k21]]` times `h`.
fn two_cpt(k10: f64, k12: f64, k21: f64, h: f64) -> Vec<f64> {
    vec![-(k10 + k12) * h, k21 * h, k12 * h, -k21 * h]
}

#[test]
fn expm_matches_nalgebra_across_norms() {
    // Small norm (no squaring), a dosing interval of a 2-cpt model (a few squarings), the
    // #852 tacrolimus-style stiff liver over a day (|λ|max 50/h × 24 h: ~8 squarings), and a
    // dense non-normal 4×4 with mixed signs.
    let liver = {
        // depot → liver → central ⇄ periph; liver drains at QH/V4 = 50/h.
        let (ka, kl, klc, k10, k12, k21) = (1.2, 50.0, 49.0, 0.04, 0.3, 0.2);
        let h = 24.0;
        #[rustfmt::skip]
        let a = vec![
            -ka,  0.0,  0.0,            0.0,
             ka, -kl,   0.0,            0.0,
             0.0, klc, -(k10 + k12),    k21,
             0.0, 0.0,  k12,           -k21,
        ];
        a.into_iter().map(|x| x * h).collect::<Vec<_>>()
    };
    #[rustfmt::skip]
    let dense = vec![
        -1.3,  0.4,  2.0, -0.7,
         0.9, -2.2,  0.1,  1.5,
        -0.3,  1.1, -0.8,  0.6,
         2.4, -1.9,  0.5, -3.1,
    ];
    let cases: Vec<(&str, Vec<f64>, usize)> = vec![
        ("small", two_cpt(0.1, 0.3, 0.2, 0.5), 2),
        ("interval", two_cpt(0.1, 0.3, 0.2, 24.0), 2),
        ("stiff_liver", liver, 4),
        ("dense", dense, 4),
    ];
    for (name, a, n) in cases {
        let got = expm_g::<f64>(&a, n).expect("finite input");
        let err = max_rel_err(&got, &nalgebra_expm(&a, n));
        // Measured worst 1.5e-16 (dense; exactly 0 on interval and stiff_liver). The bound is
        // loose on purpose — this pins wiring, and a wrong coefficient misses by O(1).
        assert!(err < 1e-12, "{name}: expm_g vs nalgebra rel err {err:e}");
    }
}

#[test]
fn expm_matches_two_cpt_closed_form_through_squaring() {
    // Independent of Padé: the 2-cpt bolus solution in its macro-rate (Bateman) form, at a
    // span long enough that `expm_g` scales and squares (‖A‖₁·h ≈ 10 > θ13).
    let (k10, k12, k21, h) = (0.1, 0.3, 0.2, 24.0);
    let a = two_cpt(k10, k12, k21, h);
    let e = expm_g::<f64>(&a, 2).unwrap();
    let sum = k10 + k12 + k21;
    let disc = (sum * sum - 4.0 * k10 * k21).sqrt();
    let (alpha, beta) = (0.5 * (sum + disc), 0.5 * (sum - disc));
    let (ea, eb) = ((-alpha * h).exp(), (-beta * h).exp());
    // Column 0: unit bolus into central.
    let c11 = ((alpha - k21) * ea + (k21 - beta) * eb) / (alpha - beta);
    let c21 = k12 * (eb - ea) / (alpha - beta);
    for (got, want) in [(e[0], c11), (e[2], c21)] {
        let rel = (got - want).abs() / want.abs();
        // Measured ≤ 1.3e-15 after 1 squaring, against a formula that shares no code with Padé.
        assert!(rel < 1e-12, "{got} vs {want} (rel {rel:e})");
    }
}

#[test]
fn expm_of_defective_transit_chain_is_erlang() {
    // depot → t1 → t2 all at rate k: one Jordan block of size 3, the case an eigendecomposition
    // cannot represent. Column 0 of exp(A·t) is the Erlang-k impulse response.
    let (k, t) = (1.7, 2.3);
    #[rustfmt::skip]
    let a = vec![
        -k,  0.0, 0.0,
         k, -k,   0.0,
         0.0, k, -k,
    ];
    let at: Vec<f64> = a.iter().map(|x| x * t).collect();
    let e = expm_g::<f64>(&at, 3).expect("finite");
    let kt = k * t;
    let want = [(-kt).exp(), kt * (-kt).exp(), 0.5 * kt * kt * (-kt).exp()];
    for i in 0..3 {
        let rel = (e[i * 3] - want[i]).abs() / want[i];
        // Measured ≤ 8.7e-16; the bound is a rounding bound, not a convergence one.
        assert!(
            rel < 1e-13,
            "row {i}: {} vs {} (rel {rel:e})",
            e[i * 3],
            want[i]
        );
    }
}

#[test]
fn expm_exact_on_zero_and_nilpotent() {
    assert_eq!(
        expm_g::<f64>(&[0.0; 4], 2).unwrap(),
        vec![1.0, 0.0, 0.0, 1.0]
    );
    // [[0, c], [0, 0]] is nilpotent: exp = I + A exactly. A large `c` forces squarings, which
    // must not lose the exactness (A² = 0 at every scale).
    let c = 1.0e3;
    let e = expm_g::<f64>(&[0.0, c, 0.0, 0.0], 2).unwrap();
    assert_eq!(e[0], 1.0);
    assert_eq!(e[2], 0.0);
    assert_eq!(e[3], 1.0);
    // Measured exactly 0.
    assert!(
        (e[1] - c).abs() / c < 1e-14,
        "nilpotent off-diagonal {}",
        e[1]
    );
}

#[test]
fn expm_declines_non_finite_input() {
    assert!(expm_g::<f64>(&[f64::NAN, 0.0, 0.0, 1.0], 2).is_none());
    assert!(expm_g::<f64>(&[f64::INFINITY, 0.0, 0.0, 1.0], 2).is_none());
    // Finite but past the squaring cap: refused rather than looping.
    assert!(expm_g::<f64>(&[-1.0e300, 0.0, 0.0, -1.0], 2).is_none());
}

/// A 3×3 generator whose entries mix both parameters, so `exp(A(p))` has a nonzero gradient
/// and Hessian in every entry. Generic so one source serves the FD reference and the dual.
fn param_generator<T: PkNum>(p0: T, p1: T, h: f64) -> Vec<T> {
    let c = |x: f64| T::from_f64(x);
    let k12 = p0 * c(0.3);
    let k21 = p1;
    let k10 = p0 * p1 * c(0.2);
    let ka = c(1.1) + p1 * c(0.5);
    let hh = c(h);
    vec![
        -ka * hh,
        c(0.0),
        c(0.0),
        ka * hh,
        -(k10 + k12) * hh,
        k21 * hh,
        c(0.0),
        k12 * hh,
        -k21 * hh,
    ]
}

#[test]
fn expm_dual2_gradient_and_hessian_match_fd() {
    // h = 12 puts ‖A‖₁ past θ13, so the squaring loop is inside the differentiated path.
    let (p0, p1, h) = (0.8, 0.45, 12.0);
    let a_dual = param_generator(Dual2::<2>::var(p0, 0), Dual2::<2>::var(p1, 1), h);
    let norm1 = (0..3)
        .map(|j| (0..3).map(|i| a_dual[i * 3 + j].value.abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);
    assert!(
        norm1 > THETA13,
        "fixture must exercise squaring (‖A‖₁ = {norm1})"
    );
    let e = expm_g(&a_dual, 3).expect("finite");
    let f = |q0: f64, q1: f64, k: usize| expm_g::<f64>(&param_generator(q0, q1, h), 3).unwrap()[k];

    let (hg, hh) = (1e-6, 1e-4);
    for k in 0..9 {
        // Same computation, but the dual's arithmetic rounds independently: measured 1 ULP.
        approx::assert_relative_eq!(e[k].value, f(p0, p1, k), max_relative = 1e-14);
        let g0 = (f(p0 + hg, p1, k) - f(p0 - hg, p1, k)) / (2.0 * hg);
        let g1 = (f(p0, p1 + hg, k) - f(p0, p1 - hg, k)) / (2.0 * hg);
        approx::assert_relative_eq!(e[k].grad[0], g0, max_relative = 1e-6, epsilon = 1e-9);
        approx::assert_relative_eq!(e[k].grad[1], g1, max_relative = 1e-6, epsilon = 1e-9);
        let hxx = (f(p0 + hh, p1, k) - 2.0 * f(p0, p1, k) + f(p0 - hh, p1, k)) / (hh * hh);
        let hyy = (f(p0, p1 + hh, k) - 2.0 * f(p0, p1, k) + f(p0, p1 - hh, k)) / (hh * hh);
        let hxy = (f(p0 + hh, p1 + hh, k) - f(p0 + hh, p1 - hh, k) - f(p0 - hh, p1 + hh, k)
            + f(p0 - hh, p1 - hh, k))
            / (4.0 * hh * hh);
        approx::assert_relative_eq!(e[k].hess[0][0], hxx, max_relative = 1e-4, epsilon = 1e-6);
        approx::assert_relative_eq!(e[k].hess[1][1], hyy, max_relative = 1e-4, epsilon = 1e-6);
        approx::assert_relative_eq!(e[k].hess[0][1], hxy, max_relative = 1e-4, epsilon = 1e-6);
    }
}

// ── linear_segment_g ──────────────────────────────────────────────────────────────────────

/// One compartment with a constant infusion: `du/dt = −k·u + R`.
fn one_cpt_infusion<T: PkNum>(u: &[T], p: &[T], _t: f64, du: &mut [T]) {
    du[0] = p[1] - p[0] * u[0];
}

#[test]
fn segment_matches_closed_form_infusion() {
    let (k, r, u0) = (0.35, 12.0, 4.0);
    let (t0, t1) = (2.0, 9.0);
    let saveat = [3.0, 4.5, 6.0, 7.5, 9.0];
    let interp = [2.5, 8.25];
    let (hard, soft) = linear_segment_g::<f64>(
        &one_cpt_infusion,
        &[u0],
        (t0, t1),
        &[k, r],
        &saveat,
        &interp,
    )
    .expect("affine, autonomous");
    let exact = |t: f64| {
        let e = (-k * (t - t0)).exp();
        u0 * e + r / k * (1.0 - e)
    };
    for (pt, &t) in hard.iter().zip(&saveat).chain(soft.iter().zip(&interp)) {
        assert_eq!(pt.t, t);
        let rel = (pt.u[0] - exact(t)).abs() / exact(t);
        // Measured ≤ 1.2e-15 (chained propagation across the grid).
        assert!(
            rel < 1e-13,
            "t={t}: {} vs {} (rel {rel:e})",
            pt.u[0],
            exact(t)
        );
    }
}

#[test]
fn segment_reads_a_column_the_forcing_dwarfs() {
    // `b = f(0)` is cancelled out of every probe, so a unit probe reads `A_ij` to ~ε·|b_i|
    // absolutely. Each case puts the forcing many orders above the coefficient: a unit probe
    // reads A = 0 on the first and third, and the segment is declined (and integrated) on all
    // three; a probe of 1e12 still fails the bound. The first also needs the forcing column
    // of the augmented generator scaled down, or ‖M‖ = 1e20 exhausts the squaring budget.
    for (k, r, t) in [(1.0, 1e20, 1.0), (1e-3, 1e9, 24.0), (0.37, 3.3e19, 1.7)] {
        let (hard, _) =
            linear_segment_g::<f64>(&one_cpt_infusion, &[0.0], (0.0, t), &[k, r], &[t], &[])
                .expect("affine, autonomous");
        let exact = r / k * -(-k * t).exp_m1();
        let rel = (hard[0].u[0] - exact).abs() / exact;
        assert!(hard[0].u[0].is_finite());
        // Measured ≤ 3.2e-16 across the three.
        assert!(
            rel < 1e-14,
            "k={k}, r={r}: {} vs {exact} (rel {rel:e})",
            hard[0].u[0]
        );
    }
}

#[test]
fn segment_reads_jets_the_forcing_sensitivity_dwarfs() {
    // `du/dt = C·p^k − (1 + p)·u` at p = 0, u0 = U: the value of `b` is 0, but its first
    // (k = 1) or second (k = 2) derivative is of order C, so a probe below that magnitude
    // rounds ∂A/∂p from −1 to 0 while every value check passes. Exact, with E = e^{−t}:
    //   u = U·E;  k = 1: ∂u/∂p = C·(1 − E) − U·t·E,
    //                    ∂²u/∂p² = −2C·(1 − E) + 2C·t·E + U·t²·E;
    //             k = 2: ∂u/∂p = −U·t·E,  ∂²u/∂p² = 2C·(1 − E) + U·t²·E.
    // C spans 1e20 to 1e170 (a fixed probe of 2^500 ≈ 3e150 loses the last). U = 1 leaves
    // the jets of `b` as the only large magnitude in the model, so a probe sized from values
    // alone is too small there.
    let t = 1.0;
    let e = (-t).exp();
    for c in [1e20, 1e170] {
        for u0 in [c, 1.0] {
            for k in [1, 2] {
                let rhs = move |u: &[Dual2<1>], p: &[Dual2<1>], _t: f64, du: &mut [Dual2<1>]| {
                    let one = Dual2::<1>::from_f64(1.0);
                    let forcing = if k == 1 { p[0] } else { p[0] * p[0] };
                    du[0] = Dual2::<1>::from_f64(c) * forcing - (one + p[0]) * u[0];
                };
                let label = format!("C = {c:e}, u0 = {u0:e}, k = {k}");
                let (hard, _) = linear_segment_g(
                    &rhs,
                    &[Dual2::<1>::from_f64(u0)],
                    (0.0, t),
                    &[Dual2::<1>::var(0.0, 0)],
                    &[t],
                    &[],
                )
                .unwrap_or_else(|| panic!("{label}: affine, autonomous"));
                let got = hard[0].u[0];
                let (grad, hess) = if k == 1 {
                    (
                        c * (1.0 - e) - u0 * t * e,
                        -2.0 * c * (1.0 - e) + 2.0 * c * t * e + u0 * t * t * e,
                    )
                } else {
                    (-u0 * t * e, 2.0 * c * (1.0 - e) + u0 * t * t * e)
                };
                let rel_v = (got.val() - u0 * e).abs() / (u0 * e);
                let rel_g = (got.grad[0] - grad).abs() / grad.abs();
                let rel_h = (got.hess[0][0] - hess).abs() / hess.abs();
                assert!(
                    got.jets_finite() && got.val().is_finite(),
                    "{label}: {got:?}"
                );
                // Measured ≤ 1.3e-15 over all eight (value, ∂/∂p and ∂²/∂p²).
                assert!(
                    rel_v < 1e-14 && rel_g < 1e-14 && rel_h < 1e-14,
                    "{label}: value rel {rel_v:e}, ∂/∂p rel {rel_g:e}, ∂²/∂p² rel {rel_h:e}"
                );
            }
        }
    }
}

#[test]
fn segment_declines_an_rhs_whose_jets_are_not_affine() {
    // `du/dt = −u + p·u²` at p = 0: the value is `−u`, affine at every state, so every value
    // check passes — but ∂f/∂p = u² is not, and the column read for it (≈ s) disagrees with
    // the RHS's own derivative at `u0`. Only the derivative half of the self-check can see it;
    // accepting the segment returns a gradient that is off by the probe scale.
    let rhs = |u: &[Dual2<1>], p: &[Dual2<1>], _t: f64, du: &mut [Dual2<1>]| {
        du[0] = p[0] * u[0] * u[0] - u[0];
    };
    assert!(linear_segment_g(
        &rhs,
        &[Dual2::<1>::from_f64(1.0)],
        (0.0, 1.0),
        &[Dual2::<1>::var(0.0, 0)],
        &[1.0],
        &[],
    )
    .is_none());
}

/// Two compartments, depot absorption, constant infusion into central — the shape the
/// integrator comparison below runs on.
fn two_cpt_depot<T: PkNum>(u: &[T], p: &[T], _t: f64, du: &mut [T]) {
    let (ka, k10, k12, k21, rate) = (p[0], p[1], p[2], p[3], p[4]);
    du[0] = -ka * u[0];
    du[1] = ka * u[0] - (k10 + k12) * u[1] + k21 * u[2] + rate;
    du[2] = k12 * u[1] - k21 * u[2];
}

#[test]
fn segment_agrees_with_a_tight_integrator() {
    let p = [1.4, 0.12, 0.35, 0.18, 5.0];
    let u0 = [100.0, 3.0, 1.0];
    // Production saveat lies in (t0, t1] and always contains t1 (`integrate_segment`).
    let saveat = [0.5, 1.0, 2.0, 6.0, 12.0];
    let tight = OdeSolverOptions {
        reltol: 1e-12,
        abstol: 1e-12,
        max_steps: 1_000_000,
        method: OdeMethod::Rodas5P,
        ..Default::default()
    };
    let reference = solve_ode(&two_cpt_depot::<f64>, &u0, (0.0, 12.0), &p, &saveat, &tight);
    let (hard, _) =
        linear_segment_g::<f64>(&two_cpt_depot, &u0, (0.0, 12.0), &p, &saveat, &[]).unwrap();
    assert_eq!(hard.len(), reference.len());
    let mut worst = 0.0_f64;
    for (got, want) in hard.iter().zip(&reference) {
        assert_eq!(got.t, want.t);
        for (g, w) in got.u.iter().zip(&want.u) {
            // Both sides: `worst.max(NaN)` keeps `worst`, so a NaN on either would fold away.
            assert!(g.is_finite() && w.is_finite(), "linear {g}, rodas5p {w}");
            worst = worst.max((g - w).abs() / w.abs().max(1e-12));
        }
    }
    // Measured 4.3e-10 against rodas5p at 1e-12: the integrator's own error, not the
    // exponential's (the closed-form tests pin that at ~1e-15).
    assert!(
        worst < 1e-8,
        "linear vs tight rodas5p: worst rel err {worst:e}"
    );
}

#[test]
fn segment_declines_a_nonlinear_rhs() {
    // Michaelis–Menten elimination: the classifier rejects it structurally; this is the
    // runtime backstop behind that certification (public `solve_ode*` never get here).
    let mm = |u: &[f64], p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = -p[0] * u[0] / (p[1] + u[0]);
    };
    assert!(linear_segment_g::<f64>(&mm, &[50.0], (0.0, 4.0), &[10.0, 5.0], &[4.0], &[]).is_none());
    // The probe sits 2^20 above the model's magnitudes, so even with Km far above every state
    // the trajectory reaches, the column is read where saturation is ~1e-6 relative — far
    // outside the self-check's 1e-9 — and the segment is declined.
    assert!(linear_segment_g::<f64>(&mm, &[0.5], (0.0, 4.0), &[10.0, 1e12], &[4.0], &[]).is_none());
    // A nonlinear RHS built to agree with its probe secant at `u0 = 1` *and* at the probe
    // (2^20 here: b = 0 and u0 = 1 put the model's magnitude at 1):
    // `u·(1 + (u − 1)·(s − u)/s)` is `u` at both, so A = 1, b = 0 and the entry checks pass.
    // Only the states the propagation reaches (`u ≈ e^0.5`, where the RHS is ~1.65× A·u)
    // expose it.
    let s = 2f64.powi(20);
    let secant_twin = move |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = u[0] * (1.0 + (u[0] - 1.0) * ((s - u[0]) / s))
    };
    let mut at_u0 = [0.0];
    secant_twin(&[1.0], &[], 0.0, &mut at_u0);
    let mut at_probe = [0.0];
    secant_twin(&[s], &[], 0.0, &mut at_probe);
    assert_eq!(
        (at_u0[0], at_probe[0]),
        (1.0, s),
        "fixture must match its secant at both"
    );
    assert!(linear_segment_g::<f64>(&secant_twin, &[1.0], (0.0, 0.5), &[], &[0.5], &[]).is_none());
}

#[test]
fn segment_declines_forcing_that_switches_inside_the_span() {
    // An input that stops at t = 2 inside a (0, 4) span: piecewise in time, not LTI.
    let switched = |u: &[f64], _p: &[f64], t: f64, du: &mut [f64]| {
        du[0] = if t < 2.0 { 10.0 } else { 0.0 } - 0.3 * u[0];
    };
    assert!(linear_segment_g::<f64>(&switched, &[1.0], (0.0, 4.0), &[], &[4.0], &[]).is_none());
    // The same input over a span it does not switch inside is accepted.
    assert!(linear_segment_g::<f64>(&switched, &[1.0], (2.0, 4.0), &[], &[4.0], &[]).is_some());
}

#[test]
fn segment_dual_gradient_matches_fd() {
    let base = [1.4, 0.12, 0.35, 0.18, 5.0];
    let u0 = [100.0, 3.0, 1.0];
    let t_end = 10.0;
    let solve = |ka: f64, k10: f64| -> Vec<f64> {
        let p = [ka, k10, base[2], base[3], base[4]];
        linear_segment_g::<f64>(&two_cpt_depot, &u0, (0.0, t_end), &p, &[t_end], &[])
            .unwrap()
            .0[0]
            .u
            .clone()
    };
    let p_dual: Vec<Dual2<2>> = base
        .iter()
        .enumerate()
        .map(|(i, &v)| match i {
            0 => Dual2::<2>::var(v, 0),
            1 => Dual2::<2>::var(v, 1),
            _ => Dual2::<2>::from_f64(v),
        })
        .collect();
    let u0_dual: Vec<Dual2<2>> = u0.iter().map(|&v| Dual2::<2>::from_f64(v)).collect();
    let (hard, _) = linear_segment_g(
        &two_cpt_depot,
        &u0_dual,
        (0.0, t_end),
        &p_dual,
        &[t_end],
        &[],
    )
    .unwrap();
    let h = 1e-6;
    for i in 0..3 {
        let g0 = (solve(base[0] + h, base[1])[i] - solve(base[0] - h, base[1])[i]) / (2.0 * h);
        let g1 = (solve(base[0], base[1] + h)[i] - solve(base[0], base[1] - h)[i]) / (2.0 * h);
        approx::assert_relative_eq!(
            hard[0].u[i].grad[0],
            g0,
            max_relative = 1e-6,
            epsilon = 1e-8
        );
        approx::assert_relative_eq!(
            hard[0].u[i].grad[1],
            g1,
            max_relative = 1e-6,
            epsilon = 1e-8
        );
    }
}

/// `du/dt = −g(t)·u` with `g = 1 + 1000·[(t − ¼)(t − ½)(t − ¾)]²`: `g = 1` at the midpoint
/// (where A is read) and at both quarter-point checks, so the sampled self-check cannot see
/// the time dependence and the exact path returns `e^{−1}` — while the truth over (0, 1) is
/// `e^{−∫g} = e^{−(1 + 1000·107/107520)}`. The CTMM occupancy RHS (`markov`) is shaped like
/// this and reached `solve_ode` with the PK model's `Linear` route (#852 review).
fn time_varying_rate(u: &[f64], _p: &[f64], t: f64, du: &mut [f64]) {
    let q = (t - 0.25) * (t - 0.5) * (t - 0.75);
    du[0] = -(1.0 + 1000.0 * q * q) * u[0];
}

#[test]
fn public_solve_ode_never_takes_the_exact_path() {
    let linear = OdeSolverOptions {
        method: OdeMethod::Linear,
        reltol: 1e-10,
        abstol: 1e-12,
        ..Default::default()
    };
    let truth = (-(1.0 + 1000.0 * 107.0 / 107_520.0_f64)).exp();
    let frozen = (-1.0_f64).exp();
    // The straddle: on the certified route the fixture really does fool the exact path, so
    // the public assertions below are not passing because nothing could go wrong.
    let certified = crate::ode::solver::solve_ode_certified(
        &time_varying_rate,
        &[1.0],
        (0.0, 1.0),
        &[],
        &[1.0],
        &linear,
    );
    assert!(
        (certified[0].u[0] - frozen).abs() < 1e-14,
        "fixture must fool the sampled self-check: {}",
        certified[0].u[0]
    );
    let check = |label: &str, got: f64| {
        assert!(got.is_finite(), "{label}: {got}");
        assert!(
            (got - truth).abs() / truth < 1e-7,
            "{label}: {got} vs integrated truth {truth} (frozen-generator answer {frozen})"
        );
    };
    check(
        "solve_ode",
        solve_ode(&time_varying_rate, &[1.0], (0.0, 1.0), &[], &[1.0], &linear)[0].u[0],
    );
    let rhs_g = |u: &[Dual2<1>], _p: &[Dual2<1>], t: f64, du: &mut [Dual2<1>]| {
        let q = (t - 0.25) * (t - 0.5) * (t - 0.75);
        du[0] = u[0] * Dual2::<1>::from_f64(-(1.0 + 1000.0 * q * q));
    };
    let u0 = [Dual2::<1>::from_f64(1.0)];
    check(
        "solve_ode_g",
        crate::ode::solver::solve_ode_g(&rhs_g, &u0, (0.0, 1.0), &[], &[1.0], &linear)[0].u[0]
            .val(),
    );
    check(
        "solve_ode_g_dense",
        crate::ode::solver::solve_ode_g_dense(
            &rhs_g,
            &u0,
            (0.0, 1.0),
            &[],
            &[1.0],
            &[],
            &linear,
            None,
        )
        .0[0]
            .u[0]
            .val(),
    );
}

/// The drug-driven CTMM transition integrates its occupancy equation through public
/// `solve_ode` with the PK spec's options — `Linear` on a linear PK model. Its generator varies
/// in time, so it must be integrated: `P[0,0] = e^{−∫g}`, not the frozen `e^{−1}`.
#[cfg(feature = "markov")]
#[test]
fn ctmm_occupancy_ignores_the_pk_models_exact_route() {
    let linear = OdeSolverOptions {
        method: OdeMethod::Linear,
        reltol: 1e-10,
        abstol: 1e-12,
        ..Default::default()
    };
    let p = crate::markov::ctmm_inhomogeneous_transition_with_opts(
        |t| {
            let q = (t - 0.25) * (t - 0.5) * (t - 0.75);
            let g = 1.0 + 1000.0 * q * q;
            nalgebra::DMatrix::from_row_slice(2, 2, &[-g, g, 0.0, 0.0])
        },
        1.0,
        2,
        &linear,
    );
    let truth = (-(1.0 + 1000.0 * 107.0 / 107_520.0_f64)).exp();
    assert!(p[(0, 0)].is_finite(), "{p}");
    assert!(
        (p[(0, 0)] - truth).abs() / truth < 1e-7,
        "P[0,0] = {} vs {truth} (frozen generator: {})",
        p[(0, 0)],
        (-1.0_f64).exp()
    );
}

#[test]
fn segment_probe_stays_near_the_models_own_magnitudes() {
    // `((u / 1e-20) · 1e150 / 1e150) · 1e-20` is `u`, finite at every state this model
    // reaches, but overflows at a state near 2^500: a probe that large declined this
    // certified-linear model and integrated it. Sized from the model's magnitudes (here 1,
    // so 2^20) the intermediate peaks near 1e176 and the exact path runs.
    let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = -((((u[0] / 1e-20) * 1e150) / 1e150) * 1e-20);
    };
    let (hard, _) = linear_segment_g::<f64>(&rhs, &[1.0], (0.0, 2.0), &[], &[2.0], &[])
        .expect("certified-linear: must not be declined for an overflowing probe");
    let exact = (-2.0_f64).exp();
    assert!(
        (hard[0].u[0] - exact).abs() / exact < 1e-14,
        "{} vs {exact}",
        hard[0].u[0]
    );
}

#[test]
fn segment_keeps_a_large_forcing_jet_finite() {
    // `du/dt = 1e292·R` at R = 0, u0 = 0: every value is 0, so the generator's value is the
    // zero matrix and the exponential takes no squarings, while ∂b/∂R = 1e292. The Padé
    // numerator multiplies that jet by its first coefficient (~3.2e16) and overflowed it to
    // `inf`; the RHS reads no state, so no self-check reads the corrupted state jet. Exact:
    // u(t) = 1e292·R·t, so ∂u/∂R = 1e292·t and ∂²u/∂R² = 0.
    let big = 1e292;
    let rhs = |_u: &[Dual2<1>], p: &[Dual2<1>], _t: f64, du: &mut [Dual2<1>]| {
        du[0] = Dual2::<1>::from_f64(big) * p[0];
    };
    let t = 1.0;
    let got = linear_segment_g(
        &rhs,
        &[Dual2::<1>::from_f64(0.0)],
        (0.0, t),
        &[Dual2::<1>::var(0.0, 0)],
        &[t],
        &[],
    )
    .map(|(hard, _)| hard[0].u[0]);
    // Declining (and integrating) would also be safe; returning a non-finite jet is not.
    let got = got.expect("affine, autonomous: the exact path must serve this");
    assert!(got.jets_finite(), "{got:?}");
    assert_eq!(got.val(), 0.0);
    let rel_g = (got.grad[0] - big * t).abs() / (big * t);
    assert!(rel_g < 1e-14, "∂u/∂R = {:e} vs {:e}", got.grad[0], big * t);
    assert_eq!(got.hess[0][0], 0.0);
}

#[test]
fn expm_declines_an_overflowed_jet() {
    // Value zero (no squarings), derivative 1e300: the Padé numerator's first coefficient
    // (~3.2e16) overflows the jet while every value stays finite. Returning that as an exact
    // exponential would hand FOCEI an `inf` gradient; declining lets the caller integrate.
    let mut g = vec![Dual2::<1>::from_f64(0.0); 4];
    g[1] = Dual2::<1>::var(0.0, 0) * Dual2::<1>::from_f64(1e300);
    assert!(expm_g(&g, 2).is_none());
    // The same generator a hundred orders smaller is served: exp([[0, x], [0, 0]]) =
    // [[1, x], [0, 1]] (to the Padé quotient's rounding — 1 ulp on the diagonal, measured).
    g[1] = Dual2::<1>::var(0.0, 0) * Dual2::<1>::from_f64(1e200);
    let e = expm_g(&g, 2).expect("finite jets");
    assert!((e[0].val() - 1.0).abs() < 1e-15 && (e[3].val() - 1.0).abs() < 1e-15);
    assert_eq!(e[1].val(), 0.0);
    assert!(
        (e[1].grad[0] - 1e200).abs() / 1e200 < 1e-15,
        "{:e}",
        e[1].grad[0]
    );
}

#[test]
fn every_dual_overrides_jet_max_abs() {
    // `PkNum::jet_max_abs` defaults to 0.0 (correct only for `f64`). A dual that fell back to
    // it would make the propagator's derivative self-check vacuous — on exactly the solves the
    // gradient comes from — so each dual's override is pinned through the trait.
    use crate::sens::dual1::Dual1;
    fn through_trait<T: PkNum>(x: T) -> f64 {
        x.jet_max_abs()
    }
    assert_eq!(through_trait(2.5_f64), 0.0);
    assert_eq!(
        through_trait(Dual1::<2>::var(1.0, 1) * Dual1::<2>::from_f64(-3.0)),
        3.0
    );
    let x = Dual2::<1>::var(1.0, 0);
    assert_eq!(through_trait(x * x * x), 6.0); // ∂ = 3x² = 3, ∂² = 6x = 6: the Hessian counts
    assert!(through_trait(Dual2::<1>::var(1.0, 0) * Dual2::<1>::from_f64(f64::NAN)).is_nan());

    // The adjoint's duals (#486): the η-tangent tape type, and the covariance Hessian's
    // nested forward-over-reverse pair. The reverse-only `R` has no local jet.
    use crate::sens::rtape::{DualH, DualT, R};
    use crate::sens::vtape::DualV;
    let mut v = DualV::<2, 3>::constant(1.0);
    v.g = [0.5, -3.0];
    v.h = [0.0, 4.0, 0.0];
    assert_eq!(through_trait(v), 4.0, "DualV: the Hessian payload counts");
    v.h[0] = f64::NAN;
    assert!(through_trait(v).is_nan(), "DualV: NaN is not discarded");
    assert_eq!(through_trait(R::c(7.0)), 0.0);
    let mut t = DualT::<2, f64>::constant(1.0);
    t.g = [2.0, -5.0];
    assert_eq!(through_trait(t), 5.0, "DualT");
    let mut h = DualH::<1, 1, DualT<1, f64>>::constant(1.0);
    h.g[0].v = 0.25;
    h.h[0].g = [-6.0];
    assert_eq!(
        through_trait(h),
        6.0,
        "DualH: the base's jet of a Hessian entry counts"
    );
}

/// The pre-skip `matmul`: every product taken, structural zeros included.
fn matmul_reference<T: PkNum>(x: &[T], y: &[T], n: usize) -> Vec<T> {
    let mut out = vec![T::from_f64(0.0); n * n];
    for i in 0..n {
        for k in 0..n {
            for j in 0..n {
                out[i * n + j] = out[i * n + j] + x[i * n + k] * y[k * n + j];
            }
        }
    }
    out
}

/// Bits of every component, with `±0` identified (skipping a zero product may flip a zero's
/// sign and nothing else).
fn component_bits(v: &[crate::sens::vfree::DualF<2, 0>]) -> Vec<u64> {
    let norm = |x: f64| if x == 0.0 { 0 } else { x.to_bits() };
    v.iter()
        .flat_map(|d| std::iter::once(norm(d.v)).chain(d.g.iter().map(|&g| norm(g))))
        .collect()
}

/// A transit-like generator with structural zeros, a parameter-carrying chain, and one entry
/// that is numerically zero but carries a derivative (`θ − θ₀` at `θ = θ₀`).
fn sparse_generator() -> Vec<crate::sens::vfree::DualF<2, 0>> {
    use crate::sens::vfree::DualF;
    let n = 4;
    let z = DualF::<2, 0>::from_f64(0.0);
    let mut a = vec![z; n * n];
    for i in 0..n {
        a[i * n + i] = DualF {
            v: -0.7 - 0.1 * i as f64,
            g: [-1.0, 0.0],
            h: [],
        };
        if i + 1 < n {
            a[(i + 1) * n + i] = DualF {
                v: 0.7 + 0.1 * i as f64,
                g: [1.0, 0.3],
                h: [],
            };
        }
    }
    // Numerically zero, derivative live: must not be skipped.
    a[3] = DualF {
        v: 0.0,
        g: [0.0, 2.5],
        h: [],
    };
    a
}

/// The structural-zero skip in `matmul` is exact — bit-identical to taking every product —
/// while a zero that carries a derivative is still multiplied. Mutation: test the value
/// alone (`val() == 0`) → the `a[3]` jet is dropped and the `DualF` comparison fails.
#[test]
fn matmul_skips_structural_zeros_exactly_and_keeps_a_jet_carrying_zero() {
    let n = 4;
    let a = sparse_generator();
    let a2 = matmul(&a, &a, n);
    assert_eq!(
        component_bits(&a2),
        component_bits(&matmul_reference(&a, &a, n))
    );
    // (A·A)[0][3] = a00·a03 + a03·a33, so ∂/∂p₁ = −0.7·2.5 + (−1.0)·2.5 = −4.25 exactly.
    // Skipping the jet-carrying a03 as x (k = 3) leaves −1.75; the bitwise comparison
    // above catches it too, this names the number.
    assert_eq!(a2[3].g[1], -4.25, "the jet-carrying zero was dropped");
    let af: Vec<f64> = a.iter().map(|d| d.v).collect();
    let got = matmul(&af, &af, n);
    let want = matmul_reference(&af, &af, n);
    assert!(got.iter().zip(&want).all(|(g, w)| g == w));
}

/// A zero whose only derivative is second-order — value 0, gradient 0, Hessian live — is not
/// structural and is multiplied. Mutation: a predicate that reads value and gradient but not
/// the Hessian payload → the product's Hessian component differs from the dense reference.
#[test]
fn matmul_keeps_a_zero_whose_only_derivative_is_its_hessian() {
    use crate::sens::vfree::DualF;
    let n = 3;
    let c = |v: f64| DualF::<1, 1>::from_f64(v);
    let mut a = vec![c(0.0); n * n];
    for i in 0..n {
        a[i * n + i] = DualF {
            v: -0.5 - 0.2 * i as f64,
            g: [1.0],
            h: [0.3],
        };
    }
    a[2] = DualF {
        v: 0.0,
        g: [0.0],
        h: [1.5],
    };
    let got = matmul(&a, &a, n);
    let want = matmul_reference(&a, &a, n);
    let norm = |x: f64| if x == 0.0 { 0 } else { x.to_bits() };
    for (g, w) in got.iter().zip(&want) {
        assert_eq!(
            (norm(g.v), norm(g.g[0]), norm(g.h[0])),
            (norm(w.v), norm(w.g[0]), norm(w.h[0]))
        );
    }
    // (A·A)[0][2] = a00·a02 + a02·a22: the Hessian-only zero carries h = 1.5 into it.
    assert_ne!(got[2].h[0], 0.0, "the Hessian-only zero was dropped");
}

/// The outer gradient's type: a `DualV` on the θ-tape that is exactly zero with zero
/// η-jets (`θ − θ₀` at `θ = θ₀`) is not structural — its derivative lives on the tape,
/// not in the jets. The skipping `matmul` must give the same θ-gradient through a reverse
/// sweep as the dense reference. Mutation: `structural_zero` testing value and jets only
/// (`jet_max_abs() == 0`, dropping the tape check in `is_constant`) → the taped zero is
/// skipped and `∂S/∂p` loses the term.
#[test]
fn matmul_keeps_a_taped_zero_in_the_outer_gradient_type() {
    use crate::sens::vtape::{vadj_index, vadj_init, vsweep, vtape_reset, DualV};
    // S(p, q) = Σ_ij w_ij (A·A)_ij, with A's (0,3) entry the taped leaf p (value 0).
    fn sweep_ds(
        mul: fn(&[DualV<1, 0>], &[DualV<1, 0>], usize) -> Vec<DualV<1, 0>>,
    ) -> (f64, f64, f64) {
        let n = 4;
        vtape_reset(2);
        let p = DualV::<1, 0>::leaf(0.0, [0.0], []);
        let q = DualV::<1, 0>::leaf(0.8, [0.0], []);
        assert!(p.i != crate::sens::vtape::NONE, "premise: p is on the tape");
        let c = DualV::<1, 0>::constant;
        let mut a = vec![c(0.0); n * n];
        for i in 0..n {
            a[i * n + i] = q * c(-1.0 - 0.1 * i as f64);
            if i + 1 < n {
                a[(i + 1) * n + i] = q * c(0.5);
            }
        }
        a[3] = p;
        let a2 = mul(&a, &a, n);
        let mut s = c(0.0);
        for (k, v) in a2.iter().enumerate() {
            s = s + *v * c(1.0 + 0.25 * k as f64);
        }
        let adj = vadj_init::<1>();
        adj[vadj_index::<1, 1>(s.i, 0, 0)] = 1.0;
        let out = vsweep::<1, 1>(s.i as usize + 1);
        (
            s.v,
            out[vadj_index::<1, 1>(p.i, 0, 0)],
            out[vadj_index::<1, 1>(q.i, 0, 0)],
        )
    }
    let (sv, dp, dq) = sweep_ds(matmul::<DualV<1, 0>>);
    let (rv, rp, rq) = sweep_ds(matmul_reference::<DualV<1, 0>>);
    // The values are the dense loop's bit for bit. The adjoints are compared to rounding:
    // a taped type folds each entry into one tape node (`PkNum::dot_fold`), which sums the
    // same adjoints in another order than the dense loop's node per operation.
    assert_eq!(sv.to_bits(), rv.to_bits());
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-14 * (1.0 + b.abs());
    assert!(close(dq, rq), "∂S/∂q: skip {dq} vs dense {rq}");
    assert!(close(dp, rp), "∂S/∂p: skip {dp} vs dense {rp}");
    assert_ne!(
        dp, 0.0,
        "premise: the taped zero carries a θ-derivative into S"
    );

    // And against the function itself: central differences of the same S in plain `f64`.
    // S is quadratic in (p, q), so a central difference is exact up to rounding.
    fn s_f64(p: f64, q: f64) -> f64 {
        let n = 4;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            a[i * n + i] = q * (-1.0 - 0.1 * i as f64);
            if i + 1 < n {
                a[(i + 1) * n + i] = q * 0.5;
            }
        }
        a[3] = p;
        matmul_reference(&a, &a, n)
            .iter()
            .enumerate()
            .map(|(k, v)| v * (1.0 + 0.25 * k as f64))
            .sum()
    }
    let h = 1e-4;
    let fd_p = (s_f64(h, 0.8) - s_f64(-h, 0.8)) / (2.0 * h);
    let fd_q = (s_f64(0.0, 0.8 + h) - s_f64(0.0, 0.8 - h)) / (2.0 * h);
    // Measured: ∂S/∂p −1.52 vs FD −1.5199999999993, ∂S/∂q 11.424 vs 11.4240000000043 — about
    // 5e-13 relative, rounding only (S is quadratic); bound 1e-9.
    assert!(
        (dp - fd_p).abs() <= 1e-9 * (1.0 + fd_p.abs()),
        "∂S/∂p {dp} vs FD {fd_p}"
    );
    assert!(
        (dq - fd_q).abs() <= 1e-9 * (1.0 + fd_q.abs()),
        "∂S/∂q {dq} vs FD {fd_q}"
    );
}

/// Copilot's second dependency on the outer gradient's own type: an untaped `DualV` that is
/// zero in value and η-gradient but carries a second-order η-payload (`h`). It is not
/// constant, so it must be multiplied. Mutation: `DualV::is_constant` checking `g` but not
/// `h` → it is skipped and the product's `h` differs from the dense reference.
#[test]
fn matmul_keeps_an_outer_type_zero_whose_only_derivative_is_its_hessian() {
    use crate::sens::vtape::{DualV, NONE};
    let n = 3;
    let c = DualV::<1, 1>::constant;
    let mut a = vec![c(0.0); n * n];
    for i in 0..n {
        a[i * n + i] = DualV {
            v: -0.4 - 0.3 * i as f64,
            g: [1.0],
            h: [0.2],
            i: NONE,
        };
    }
    a[2] = DualV {
        v: 0.0,
        g: [0.0],
        h: [1.5],
        i: NONE,
    };
    assert!(
        !a[2].is_constant(),
        "premise: the payload makes it non-constant"
    );
    let got = matmul(&a, &a, n);
    let want = matmul_reference(&a, &a, n);
    let norm = |x: f64| if x == 0.0 { 0 } else { x.to_bits() };
    for (g, w) in got.iter().zip(&want) {
        assert_eq!(
            (norm(g.v), norm(g.g[0]), norm(g.h[0])),
            (norm(w.v), norm(w.g[0]), norm(w.h[0]))
        );
    }
    assert_ne!(got[2].h[0], 0.0, "the Hessian-only zero was dropped");
}

/// A structural zero is skipped in the per-step self-check only against a finite state.
/// State 1 has no self-term (a structural-zero diagonal) and accumulates from a huge
/// state 0 until it overflows: the dense check computes `0·inf = NaN`, fails, and the
/// segment declines, as its contract says for a state the extracted system cannot
/// reproduce. Skipping the zero regardless would pass the check and hand back `inf` as a
/// valid state. Mutation: drop `all_finite(xj)` from the self-check's skip → `Some`.
#[test]
fn a_segment_whose_state_overflows_declines() {
    let rhs = |u: &[f64], p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = -p[0] * u[0];
        du[1] = p[1] * u[0];
    };
    let (k, kin, t1) = (1e-9, 10.0, 1e8);
    let out = linear_segment_g(&rhs, &[1e300, 0.0], (0.0, t1), &[k, kin], &[t1], &[]);
    assert!(
        out.is_none(),
        "an overflowed state was returned as valid: {:?}",
        out.map(|(h, _)| h[0].u.clone())
    );
}

/// `matmul` keeps dense semantics against a non-finite factor: a structural zero times an
/// `inf` row is `NaN` in the dense product, so it is not skipped there. Mutation: drop the
/// `row_finite` guard → the zero is skipped and the entry comes out finite.
#[test]
fn matmul_does_not_skip_a_zero_against_a_non_finite_row() {
    let n = 2;
    let x = [0.0, 1.0, 2.0, 3.0];
    let y = [f64::INFINITY, 1.0, 1.0, 1.0];
    let got = matmul(&x, &y, n);
    let want = matmul_reference(&x, &y, n);
    assert!(want[0].is_nan(), "premise: the dense product is NaN");
    assert!(got[0].is_nan(), "the skip hid a NaN: {got:?}");
}

/// The kernel as it was before `Expm` (workspace + shared Padé steps), verbatim, so the
/// rewrite is pinned bit for bit against it rather than to a tolerance.
mod pre_expm {
    #![allow(clippy::all)]
    use super::super::{MAX_SQUARINGS, PADE13, SELF_CHECK_RTOL, THETA13};
    use crate::sens::linsolve::{lu_factor_in_place_g, lu_solve_in_place_g};
    use crate::sens::num::PkNum;

    /// Row-major `n×n` product `x·y`.
    ///
    /// A factor that is a **structural** zero — value 0 and no derivative at all, no jet and no
    /// tape node (`structural_zero`) — is skipped: its products add exactly 0 to every
    /// component, so the result is bit-identical (up to the sign of a zero). A PK generator is
    /// mostly such zeros (a transit chain is bidiagonal, its powers stay near-triangular), and in
    /// a dual type every skipped product is a full jet multiply. A zero that is only *numerically*
    /// zero — `θ − θ₀` at `θ = θ₀`, a taped or jet-carrying value — is not constant and is kept.
    fn matmul_old<T: PkNum>(x: &[T], y: &[T], n: usize) -> Vec<T> {
        use crate::sens::linsolve::{all_finite, structural_zero};
        // A skipped `x[i][k]` multiplies row `k` of `y`, which must be finite for the skipped
        // products to be exactly `±0` (`all_finite`).
        let row_finite: Vec<bool> = (0..n)
            .map(|k| y[k * n..(k + 1) * n].iter().all(|&v| all_finite(v)))
            .collect();
        let mut out = vec![T::from_f64(0.0); n * n];
        for i in 0..n {
            for k in 0..n {
                let xik = x[i * n + k];
                if row_finite[k] && structural_zero(xik) {
                    continue;
                }
                for j in 0..n {
                    out[i * n + j] = out[i * n + j] + xik * y[k * n + j];
                }
            }
        }
        out
    }

    /// `Σ_k c_k·M_k`, each `M_k` an `n×n` row-major matrix, plus `c_I·I` when `c_i` is nonzero.
    fn lincomb<T: PkNum>(terms: &[(f64, &[T])], c_i: f64, n: usize) -> Vec<T> {
        let mut out = vec![T::from_f64(0.0); n * n];
        for &(c, m) in terms {
            let c = T::from_f64(c);
            for (o, &v) in out.iter_mut().zip(m) {
                *o = *o + c * v;
            }
        }
        if c_i != 0.0 {
            for i in 0..n {
                out[i * n + i] = out[i * n + i] + T::from_f64(c_i);
            }
        }
        out
    }

    /// Matrix exponential `exp(A)` of the row-major `n×n` matrix `a`, by scaling and squaring with
    /// the degree-13 diagonal Padé approximant (Higham 2005).
    ///
    /// Returns `None` when the value part is non-finite, the norm needs more than
    /// [`MAX_SQUARINGS`] halvings, or the Padé denominator `V − U` is singular (it is not, for any
    /// finite input in range — the guard is for a diverged parameter set).
    ///
    /// The scaling exponent `s` is chosen from `‖A.val()‖₁`, so at `T = Dual2` the jets ride a
    /// computation fixed by the value part and the result is the exact derivative of the
    /// approximant the `f64` run returns. That approximant's Fréchet derivative is itself an
    /// accurate approximation of `exp`'s (Al-Mohy & Higham 2009), which the FD parity tests pin.
    pub(super) fn expm_old<T: PkNum>(a: &[T], n: usize) -> Option<Vec<T>> {
        debug_assert_eq!(a.len(), n * n, "expm_old: matrix must be n×n row-major");
        if n == 0 {
            return Some(Vec::new());
        }
        if a.iter().any(|x| !x.val().is_finite()) {
            return None;
        }
        // ‖A‖₁ — max absolute column sum of the value part.
        let norm1 = (0..n)
            .map(|j| (0..n).map(|i| a[i * n + j].val().abs()).sum::<f64>())
            .fold(0.0_f64, f64::max);
        let s = if norm1 > THETA13 {
            let s = (norm1 / THETA13).log2().ceil();
            if !(s <= MAX_SQUARINGS as f64) {
                return None;
            }
            s as u32
        } else {
            0
        };
        let scale = T::from_f64(0.5_f64.powi(s as i32));
        let a1: Vec<T> = a.iter().map(|&x| x * scale).collect();

        let b = &PADE13;
        let a2 = matmul_old(&a1, &a1, n);
        let a4 = matmul_old(&a2, &a2, n);
        let a6 = matmul_old(&a4, &a2, n);

        // U = A·[A6·(b13·A6 + b11·A4 + b9·A2) + b7·A6 + b5·A4 + b3·A2 + b1·I]
        let u_inner = lincomb(&[(b[13], &a6), (b[11], &a4), (b[9], &a2)], 0.0, n);
        let u_inner = matmul_old(&a6, &u_inner, n);
        let u_poly = lincomb(
            &[(1.0, &u_inner), (b[7], &a6), (b[5], &a4), (b[3], &a2)],
            b[1],
            n,
        );
        let u = matmul_old(&a1, &u_poly, n);
        // V = A6·(b12·A6 + b10·A4 + b8·A2) + b6·A6 + b4·A4 + b2·A2 + b0·I
        let v_inner = lincomb(&[(b[12], &a6), (b[10], &a4), (b[8], &a2)], 0.0, n);
        let v_inner = matmul_old(&a6, &v_inner, n);
        let v = lincomb(
            &[(1.0, &v_inner), (b[6], &a6), (b[4], &a4), (b[2], &a2)],
            b[0],
            n,
        );

        // Solve (V − U)·R = (V + U), one column at a time against one factorization.
        let mut q: Vec<T> = v.iter().zip(&u).map(|(&vi, &ui)| vi - ui).collect();
        let p: Vec<T> = v.iter().zip(&u).map(|(&vi, &ui)| vi + ui).collect();
        let mut piv = vec![0usize; n];
        if !lu_factor_in_place_g(&mut q, &mut piv, n) {
            return None;
        }
        let mut r = vec![T::from_f64(0.0); n * n];
        let mut col = vec![T::from_f64(0.0); n];
        for j in 0..n {
            for i in 0..n {
                col[i] = p[i * n + j];
            }
            lu_solve_in_place_g(&q, &piv, n, &mut col);
            for i in 0..n {
                r[i * n + j] = col[i];
            }
        }
        for _ in 0..s {
            r = matmul_old(&r, &r, n);
        }
        if r.iter().any(|x| !x.val().is_finite() || !x.jets_finite()) {
            return None;
        }
        Some(r)
    }

    pub(super) fn linear_segment_old<T: PkNum>(
        rhs: &dyn Fn(&[T], &[T], f64, &mut [T]),
        u0: &[T],
        t_span: (f64, f64),
        params: &[T],
        saveat: &[f64],
        interp_at: &[f64],
    ) -> Option<(
        Vec<crate::ode::solver::SolPointG<T>>,
        Vec<crate::ode::solver::SolPointG<T>>,
    )> {
        let n = u0.len();
        let (t0, t1) = t_span;
        let h = t1 - t0;
        if !(h > 0.0) || !h.is_finite() {
            return None;
        }
        let zero = T::from_f64(0.0);
        let t_mid = t0 + 0.5 * h;

        // ── Extract A (row-major n×n) and b at the midpoint: b = f(0), column j = (f(s·e_j) − b)/s.
        //
        // The difference cancels b, so it carries rounding of order ε·|b_i| — in the value and,
        // separately, in every derivative component — i.e. ε·|b_i|/(s·|A_ij|) relative to the
        // entry. A unit probe loses every digit wherever the forcing dwarfs the coefficient: an
        // infusion of 1e9 into k = 1e-3 reads A to ~1e-7, `du/dt = 1e20 − u` reads A = 0, and a
        // forcing whose *sensitivity* is large (`C·p` with C = 1e20) zeroes ∂A/∂p while the value
        // is fine. So the probe is set from the model's own magnitudes — the forcing, the entry
        // state, and the derivative components of both — times 2^20, rounded up to a power of
        // two so the multiply and divide are exact: `s·A_ij` then dominates `b_i` in every
        // component wherever `|A_ij|` is not itself tiny, and `fl(s·A + b) − b` is `s·A·(1 + δ)`.
        // Tying it to those magnitudes, rather than to one huge constant, keeps the RHS evaluated
        // within 2^20 of values it already produces, so a benign intermediate cannot overflow.
        // Whatever cancellation remains is not trusted: the self-check below compares the
        // derivative components too, and a column read wrongly in any of them declines the
        // segment rather than returning a wrong gradient.
        let mut probe = vec![zero; n];
        let mut b = vec![zero; n];
        rhs(&probe, params, t_mid, &mut b);
        let magnitude = b
            .iter()
            .chain(u0)
            .fold(1.0_f64, |m, x| m.max(x.val().abs()).max(x.jet_max_abs()));
        if !magnitude.is_finite() {
            return None;
        }
        let scale = 2f64.powi(magnitude.log2().ceil() as i32 + 20);
        if !scale.is_finite() {
            return None;
        }
        let mut a = vec![zero; n * n];
        let mut col = vec![zero; n];
        let inv_scale = T::from_f64(1.0 / scale);
        for j in 0..n {
            probe[j] = T::from_f64(scale);
            rhs(&probe, params, t_mid, &mut col);
            probe[j] = zero;
            for i in 0..n {
                a[i * n + j] = (col[i] - b[i]) * inv_scale;
            }
        }
        if a.iter().chain(&b).any(|x| !x.val().is_finite()) {
            return None;
        }

        // `f(x, t) == A·x + b` to rounding — the affine claim at one state and time.
        // In the value and in the derivative components alike: a column whose jets were lost to
        // cancellation reproduces the RHS's value exactly and its gradient not at all.
        let reproduces = |x: &[T], t: f64, direct: &mut [T]| -> bool {
            rhs(x, params, t, direct);
            (0..n).all(|i| {
                let mut assembled = b[i];
                let mut scale = b[i].val().abs();
                let mut jet_scale = b[i].jet_max_abs();
                for j in 0..n {
                    let (aij, xj) = (a[i * n + j], x[j]);
                    // A structural zero against a finite state adds exactly 0 to the sum and to
                    // both scales (the same skip, and the same guard, as the propagation).
                    if crate::sens::linsolve::structural_zero(aij)
                        && crate::sens::linsolve::all_finite(xj)
                    {
                        continue;
                    }
                    assembled = assembled + aij * xj;
                    scale += (aij.val() * xj.val()).abs();
                    jet_scale += aij.jet_max_abs() * xj.val().abs()
                        + aij.val().abs() * xj.jet_max_abs()
                        + aij.jet_max_abs() * xj.jet_max_abs();
                }
                let diff = direct[i] - assembled;
                direct[i].val().is_finite()
                    && diff.val().abs() <= SELF_CHECK_RTOL * scale + 1e-300
                    && diff.jet_max_abs() <= SELF_CHECK_RTOL * jet_scale + 1e-300
            })
        };

        // ── Self-check at the entry state, a quarter and three quarters through the span: catches
        // forcing that switches inside the span. Every propagated state is checked too, below.
        let mut direct = vec![zero; n];
        for t_check in [t0 + 0.25 * h, t0 + 0.75 * h] {
            if !reproduces(u0, t_check, &mut direct) {
                return None;
            }
        }

        // ── Augmented generator M = [[A, b/c], [0, 0]], (n+1)×(n+1); the propagated forcing
        // column is multiplied back by `c` below. Without `c` a forcing far above the rates sets
        // ‖M‖, and with it the squaring count: `du/dt = 1e20 − u` over unit time needs ~64
        // squarings and was declined. Scaling the column to the rates' magnitude leaves the
        // squaring count to `A`, where it belongs; `c` is a plain constant, so the jets are exact.
        let a_max = a.iter().fold(0.0_f64, |acc, x| acc.max(x.val().abs()));
        // Jets included: a forcing whose value is 0 but whose derivative is 1e292 leaves the
        // generator's value at zero (no squarings) while the Padé numerator multiplies that jet by
        // ~3e16 and overflows it.
        let b_max = b.iter().fold(0.0_f64, |acc, x| {
            acc.max(x.val().abs()).max(x.jet_max_abs())
        });
        let c = (b_max / a_max.max(1.0)).max(1.0);
        let inv_c = T::from_f64(1.0 / c);
        let m = n + 1;
        let mut gen = vec![zero; m * m];
        for i in 0..n {
            for j in 0..n {
                gen[i * m + j] = a[i * n + j];
            }
            gen[i * m + n] = b[i] * inv_c;
        }
        let c = T::from_f64(c);

        // Every requested time, clamped into the span, visited once in ascending order.
        let clamp = |t: f64| t.clamp(t0, t1);
        let mut times: Vec<f64> = saveat.iter().chain(interp_at).map(|&t| clamp(t)).collect();
        times.sort_by(|x, y| x.total_cmp(y));
        times.dedup();

        // Propagators keyed by the exact gap, so an evenly spaced grid pays one exponential.
        let mut cache: Vec<(u64, Vec<T>)> = Vec::new();
        let mut states: Vec<(f64, Vec<T>)> = Vec::with_capacity(times.len());
        let mut x = u0.to_vec();
        let mut t_prev = t0;
        for &t in &times {
            let dt = t - t_prev;
            if dt > 0.0 {
                let key = dt.to_bits();
                let idx = match cache.iter().position(|(k, _)| *k == key) {
                    Some(i) => i,
                    None => {
                        let scaled: Vec<T> = gen.iter().map(|&g| g * T::from_f64(dt)).collect();
                        cache.push((key, expm_old(&scaled, m)?));
                        cache.len() - 1
                    }
                };
                let e = &cache[idx].1;
                let mut next = vec![zero; n];
                // `x` is finite here: `u0` passed the magnitude check, and every propagated state
                // passed `reproduces`, which declines a non-finite one
                // (`a_segment_whose_state_overflows_declines`). So a skipped term is exactly `±0`.
                for i in 0..n {
                    let mut acc = e[i * m + n] * c;
                    for j in 0..n {
                        if crate::sens::linsolve::structural_zero(e[i * m + j]) {
                            continue;
                        }
                        acc = acc + e[i * m + j] * x[j];
                    }
                    next[i] = acc;
                }
                x = next;
                t_prev = t;
                // The affine claim along the trajectory, at the extraction time (an endpoint could
                // read the next segment's forcing). The probes and `u0` alone leave a nonlinear
                // RHS undetected whenever it agrees with its secant there — `du/dt = u²` from
                // `u0 = 1` is reproduced exactly at both — but not at the states it then reaches.
                if !reproduces(&x, t_mid, &mut direct) {
                    return None;
                }
            }
            states.push((t, x.clone()));
        }

        let lookup = |t: f64| -> Vec<T> {
            let tc = clamp(t);
            if tc <= t0 {
                return u0.to_vec();
            }
            // `times` is sorted and contains `tc` exactly.
            let i = states
                .binary_search_by(|(s, _)| s.total_cmp(&tc))
                .expect("every requested time was propagated");
            states[i].1.clone()
        };
        let at = |ts: &[f64]| -> Vec<crate::ode::solver::SolPointG<T>> {
            ts.iter()
                .map(|&t| crate::ode::solver::SolPointG { t, u: lookup(t) })
                .collect()
        };
        Some((at(saveat), at(interp_at)))
    }
}

/// A random generator: a PK-like sparse rate matrix with a forcing column, values and
/// tangents spread over magnitudes and signs (zeros and `−0.0` included).
fn random_rhs_matrix(seed: &mut u64, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut next = || {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        (*seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let r = next();
            a[i * n + j] = if i == j {
                -(0.05 + 3.0 * next())
            } else if r < 0.5 {
                0.0
            } else {
                0.02 + 2.0 * next()
            };
        }
    }
    let b: Vec<f64> = (0..n)
        .map(|_| if next() < 0.5 { 0.0 } else { 10.0 * next() })
        .collect();
    (a, b)
}

#[test]
fn the_expm_workspace_is_the_pre_expm_kernel_bit_for_bit() {
    use crate::sens::vfree::DualF;
    // Gaps at powers of two of each other (shared Padé steps), others that are not, a gap
    // repeated, and the tiny and long ends of a dosing timeline.
    let gaps = [
        0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 0.25, 3.0, 1.0, 48.0, 1e-3,
    ];
    let mut seed = 0x2545_F491_4F6C_DD1D_u64;
    let mut reused = 0;
    for case in 0..60 {
        let n = 1 + case % 5;
        let (a, b) = random_rhs_matrix(&mut seed, n);
        // A dual whose tangents scale the rates, and the forcing in the last tangent.
        let lift = |x: f64, k: usize| {
            DualF::<2, 0>::free(
                x,
                [x * 0.5 + if k % 3 == 0 { -0.0 } else { 0.0 }, k as f64],
                [],
            )
        };
        let ad: Vec<_> = a.iter().enumerate().map(|(k, &x)| lift(x, k)).collect();
        let bd: Vec<_> = b.iter().enumerate().map(|(k, &x)| lift(x, k + 1)).collect();
        // `expm` on its own, every gap through one workspace.
        let mut e64 = Expm::<f64>::new(n);
        let mut ed = Expm::<DualF<2, 0>>::new(n);
        let mut seen: Vec<u64> = Vec::new();
        for &h in &gaps {
            let ah: Vec<f64> = a.iter().map(|&x| x * h).collect();
            let adh: Vec<_> = ad.iter().map(|&x| x * DualF::constant(h)).collect();
            let want = pre_expm::expm_old(&ah, n)
                .map(|r| r.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
            let entries = e64.cache.len();
            let got = e64.exp(&ah).map(|slot| {
                e64.get(slot)
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>()
            });
            assert_eq!(want, got, "case {case}, f64, h = {h}");
            // A shared step: no new approximant for a gap that is not an exact repeat of an
            // earlier one (a repeat hits on the same key trivially and proves nothing).
            if e64.cache.len() == entries && !seen.contains(&h.to_bits()) {
                reused += 1;
            }
            seen.push(h.to_bits());
            let want = pre_expm::expm_old(&adh, n).map(|r| component_bits(&r));
            let got = ed.exp(&adh).map(|slot| component_bits(ed.get(slot)));
            assert_eq!(want, got, "case {case}, DualF, h = {h}");
        }
        // The whole segment: an affine RHS `A·u + b` read back off the closure.
        let rhs = |u: &[DualF<2, 0>], _p: &[DualF<2, 0>], _t: f64, du: &mut [DualF<2, 0>]| {
            for i in 0..n {
                let mut acc = bd[i];
                for j in 0..n {
                    acc = acc + ad[i * n + j] * u[j];
                }
                du[i] = acc;
            }
        };
        let u0: Vec<_> = (0..n)
            .map(|i| lift(100.0 * (i == 0) as u8 as f64, i))
            .collect();
        let saveat = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 36.0, 48.0];
        let want = pre_expm::linear_segment_old(&rhs, &u0, (0.0, 48.0), &[], &saveat, &[3.0]);
        let got = linear_segment_g(&rhs, &u0, (0.0, 48.0), &[], &saveat, &[3.0]);
        let bits = |o: &Option<(
            Vec<crate::ode::solver::SolPointG<DualF<2, 0>>>,
            Vec<crate::ode::solver::SolPointG<DualF<2, 0>>>,
        )>| {
            o.as_ref().map(|(h, s)| {
                h.iter()
                    .chain(s)
                    .map(|p| (p.t.to_bits(), component_bits(&p.u)))
                    .collect::<Vec<_>>()
            })
        };
        assert_eq!(bits(&want), bits(&got), "case {case}: segment");
        assert!(want.is_some(), "case {case}: the fixture must not decline");
    }
    // The fixture has to reach the shared-Padé arm, or this pins only the workspace.
    // Measured 344 over the 60 cases (exact repeats not counted); a break in power-of-two
    // sharing that keeps exact-key hits leaves none of them.
    assert!(reused > 300, "only {reused} Padé steps were shared");
}

#[test]
fn only_types_without_a_tape_are_pooled_and_shared() {
    use crate::sens::dual1::Dual1;
    use crate::sens::rtape::{DualT, R};
    use crate::sens::vfree::DualF;
    // The predicate is a trait item, so a type beyond any hand-kept list is covered.
    assert!(pade_shared::<f64>());
    assert!(pade_shared::<Dual1<3>>());
    assert!(pade_shared::<Dual2<2>>());
    assert!(pade_shared::<DualF<2, 0>>());
    assert!(pade_shared::<DualF<20, 0>>());
    assert!(!pade_shared::<R>());
    assert!(!pade_shared::<DualT<2, R>>());
    assert!(!pade_shared::<crate::sens::vtape::DualV<2, 0>>());
}

#[test]
fn a_taped_type_recomputes_every_exponential() {
    use crate::sens::rtape::{rinput, rtape_len, rtape_reset, R};
    // The same input twice records twice: sharing a result would drop the second recording.
    rtape_reset();
    let a: Vec<R> = [-0.5, 0.25, 0.0, -1.5].iter().map(|&x| rinput(x)).collect();
    let mut e = Expm::<R>::new(2);
    let first = e.exp(&a).expect("finite");
    let after_first = rtape_len();
    let second = e.exp(&a).expect("finite");
    assert!(
        rtape_len() > after_first,
        "the second call recorded nothing"
    );
    assert!(e.cache.iter().all(|c| !c.reusable && c.powers.len() == 1));
    assert_eq!(e.cache.len(), 2);
    let (x, y) = (e.get(first).to_vec(), e.get(second).to_vec());
    assert!(x
        .iter()
        .zip(&y)
        .all(|(p, q)| p.v.to_bits() == q.v.to_bits()));
    assert!(x.iter().zip(&y).any(|(p, q)| p.i != q.i), "same tape node");
}

#[test]
fn a_segment_reuses_the_threads_pade_work_bit_for_bit() {
    let pooled = pooled_pade_entries::<f64>;
    let (a, b) = (-0.8, 2.0);
    let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = a * u[0] + b;
        du[1] = 0.3 * u[0] - 0.1 * u[1];
    };
    let run = || {
        linear_segment_g(
            &rhs,
            &[1.0, 0.0],
            (0.0, 24.0),
            &[],
            &[1.0, 2.0, 4.0, 8.0],
            &[],
        )
    };
    let bits = |o: Option<(
        Vec<crate::ode::solver::SolPointG<f64>>,
        Vec<crate::ode::solver::SolPointG<f64>>,
    )>| {
        o.map(|(h, _)| {
            h.iter()
                .flat_map(|p| p.u.iter().map(|x| x.to_bits()))
                .collect::<Vec<_>>()
        })
    };
    let first = bits(run());
    let entries = pooled();
    assert!(entries > 0, "the first segment pooled nothing");
    // The second segment finds every approximant it needs: nothing new is computed.
    let second = bits(run());
    assert_eq!(first, second);
    assert_eq!(pooled(), entries);
    assert!(first.is_some());
}

#[test]
fn a_full_pool_entry_count_is_forgotten_at_the_next_segment() {
    // Fill the thread's workspace past the bound with segments of distinct generators
    // (none large enough to scale, so each is its own approximant)...
    let run = |k: usize| {
        let a = -1e-3 * (k + 1) as f64;
        let rhs = move |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = a * u[0] + 1.0;
        };
        linear_segment_g(&rhs, &[1.0], (0.0, 1.0), &[], &[1.0], &[]).expect("linear");
    };
    // One approximant each; the bound is checked at a segment's start, so `POOL_MAX_ENTRIES
    // + 1` segments end one past it.
    for k in 0..=POOL_MAX_ENTRIES {
        run(k);
    }
    assert!(pooled_pade_entries::<f64>() > POOL_MAX_ENTRIES);
    // ...and the next segment forgets them all before computing its own.
    run(POOL_MAX_ENTRIES + 5);
    assert!(
        pooled_pade_entries::<f64>() <= 2,
        "{} kept",
        pooled_pade_entries::<f64>()
    );
}
