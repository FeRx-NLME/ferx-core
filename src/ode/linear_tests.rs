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
    // three. The first also needs the forcing column of the augmented generator scaled down,
    // or ‖M‖ = 1e20 exhausts the squaring budget.
    for (k, r, t) in [(1.0, 1e20, 1.0), (1e-3, 1e9, 24.0), (0.37, 3.3e19, 1.7)] {
        let (hard, _) =
            linear_segment_g::<f64>(&one_cpt_infusion, &[0.0], (0.0, t), &[k, r], &[t], &[])
                .expect("affine, autonomous");
        let exact = r / k * -(-k * t).exp_m1();
        let rel = (hard[0].u[0] - exact).abs() / exact;
        assert!(hard[0].u[0].is_finite());
        // Measured ≤ 3.2e-16 across the three. A single rescale reads the third case's A to
        // 3e-13 (answer 8.3e-14 off), which this bound rejects.
        assert!(
            rel < 1e-14,
            "k={k}, r={r}: {} vs {exact} (rel {rel:e})",
            hard[0].u[0]
        );
    }
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
    // runtime backstop for a programmatic `OdeMethod::Linear` on an uncertified model.
    let mm = |u: &[f64], p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = -p[0] * u[0] / (p[1] + u[0]);
    };
    assert!(linear_segment_g::<f64>(&mm, &[50.0], (0.0, 4.0), &[10.0, 5.0], &[4.0], &[]).is_none());
    // `du/dt = u²` from `u0 = 1` agrees with its unit-probe secant (A = 1, b = 0) at `u0`
    // itself, so the entry-state checks pass; only the state the propagation reaches
    // (e^0.5 against the true 2) exposes it.
    let square = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| du[0] = u[0] * u[0];
    assert!(linear_segment_g::<f64>(&square, &[1.0], (0.0, 0.5), &[], &[0.5], &[]).is_none());
    // With Km far above every state reached, MM is affine to rounding at every checked point
    // and the backstop accepts it — which is why it is not the certification. Accepting it is
    // harmless because the answer is then right to that same precision: pinned here.
    let (hard, _) =
        linear_segment_g::<f64>(&mm, &[0.5], (0.0, 4.0), &[10.0, 1e12], &[4.0], &[]).unwrap();
    let linearised = 0.5 * (-10.0 / 1e12 * 4.0_f64).exp();
    assert!(
        (hard[0].u[0] - linearised).abs() < 1e-15,
        "{}",
        hard[0].u[0]
    );
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
