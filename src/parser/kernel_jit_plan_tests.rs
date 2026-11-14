//! Compiled plans against derivatives derived independently, by hand, in `f64` (no finite
//! differences): the recording type's linear solve (values factored at run time, derivatives by
//! refinement against that factorization) and its steady-state call (the implicit function
//! theorem), each to second order in η.
use super::*;
use crate::sens::num::PkNum;

/// The second-order layout over 2 η: value, `∂_0`, `∂_1`, `∂²_00`, `∂²_10`, `∂²_11`.
fn l2() -> Layout {
    Layout {
        n: 2,
        m: 3,
        th: false,
        wide: false,
        tan: 0,
    }
}

/// Record `f` over 2 η, build its plan, run it at `l2` at `eta`: each output's 6 lanes.
fn run2(f: impl Fn(&[Tr]) -> Vec<Tr>, record_at: [f64; 2], eta: [f64; 2]) -> Vec<[f64; 6]> {
    begin();
    let et: Vec<Tr> = (0..2).map(|k| Tr::eta(k, record_at[k])).collect();
    let y = f(&et);
    let rec = end(Some(y.iter().map(|v| v.reg()).collect())).expect("records");
    let plan = rec.plan(l2(), 2, 0).expect("plans");
    let l = l2().lanes();
    let mut u = vec![0.0; 2 * l];
    for k in 0..2 {
        u[k * l] = eta[k];
        u[k * l + 1 + k] = 1.0;
    }
    let mut du = vec![0.0; (y.len() + 1) * l];
    plan.forward_layout(l2(), &u, &[], &mut du).expect("runs");
    assert_eq!(du[y.len() * l], 1.0, "the plan's checks hold at {eta:?}");
    (0..y.len())
        .map(|j| std::array::from_fn(|c| du[j * l + c]))
        .collect()
}

fn close(got: f64, want: f64, what: &str) {
    assert!(
        got.is_finite() && want.is_finite(),
        "{what}: {got} vs {want}"
    );
    let rel = (got - want).abs() / (1.0 + want.abs());
    assert!(rel < 1e-12, "{what}: {got} vs {want} (rel {rel:e})");
}

/// `A(η)·x = b(η)`, 3×3, its first column small in the first row (so the solve pivots) and
/// the pivot row changing between the recording point and the evaluation point.
#[test]
fn a_recorded_linear_solve_is_exact_to_second_order_across_pivot_changes() {
    // A(η): entries affine and bilinear in η, so A′ and A″ are exact by hand.
    let a_of = |e: [f64; 2]| -> ([f64; 9], [[f64; 9]; 2], [[[f64; 9]; 2]; 2]) {
        let a = [
            1e-3 + e[0],
            2.0,
            1.0, //
            0.5,
            1.0 + e[1],
            -1.0, //
            2.0 + e[0] * e[1],
            0.25,
            3.0,
        ];
        let mut d = [[0.0; 9]; 2];
        d[0][0] = 1.0;
        d[1][4] = 1.0;
        d[0][6] = e[1];
        d[1][6] = e[0];
        let mut dd = [[[0.0; 9]; 2]; 2];
        dd[0][1][6] = 1.0;
        dd[1][0][6] = 1.0;
        (a, d, dd)
    };
    let b_of = |e: [f64; 2]| -> ([f64; 3], [[f64; 3]; 2], [[[f64; 3]; 2]; 2]) {
        let b = [1.0 + e[0] * e[0], e[1], 2.0];
        let mut d = [[0.0; 3]; 2];
        d[0][0] = 2.0 * e[0];
        d[1][1] = 1.0;
        let mut dd = [[[0.0; 3]; 2]; 2];
        dd[0][0][0] = 2.0;
        (b, d, dd)
    };
    let f = |e: &[Tr]| -> Vec<Tr> {
        let k = |c: f64| Tr::from_f64(c);
        let a = [
            k(1e-3) + e[0],
            k(2.0),
            k(1.0),
            k(0.5),
            k(1.0) + e[1],
            k(-1.0),
            k(2.0) + e[0] * e[1],
            k(0.25),
            k(3.0),
        ];
        let b = [k(1.0) + e[0] * e[0], e[1], k(2.0)];
        crate::sens::linsolve::solve_linear_system_g(&a, &b, 3).expect("nonsingular")
    };
    // Recorded where row 2 pivots first; evaluated where row 0 does as well.
    for eta in [[0.1, -0.2], [3.0, 0.5]] {
        let got = run2(f, [0.1, -0.2], eta);
        let (a, da, dda) = a_of(eta);
        let (b, db, ddb) = b_of(eta);
        let am = nalgebra::Matrix3::from_row_slice(&a);
        let inv = am.try_inverse().expect("nonsingular");
        let mv =
            |m: &[f64; 9], v: &nalgebra::Vector3<f64>| nalgebra::Matrix3::from_row_slice(m) * v;
        let x = inv * nalgebra::Vector3::from_row_slice(&b);
        let dx: Vec<nalgebra::Vector3<f64>> = (0..2)
            .map(|k| inv * (nalgebra::Vector3::from_row_slice(&db[k]) - mv(&da[k], &x)))
            .collect();
        // x_kl = A⁻¹(b_kl − A_kl·x − A_k·x_l − A_l·x_k).
        let ddx = |k: usize, l: usize| {
            inv * (nalgebra::Vector3::from_row_slice(&ddb[k][l])
                - mv(&dda[k][l], &x)
                - mv(&da[k], &dx[l])
                - mv(&da[l], &dx[k]))
        };
        for i in 0..3 {
            let what = |c: &str| format!("x[{i}] {c} at {eta:?}");
            close(got[i][0], x[i], &what("value"));
            close(got[i][1], dx[0][i], &what("∂0"));
            close(got[i][2], dx[1][i], &what("∂1"));
            close(got[i][3], ddx(0, 0)[i], &what("∂00"));
            close(got[i][4], ddx(1, 0)[i], &what("∂10"));
            close(got[i][5], ddx(1, 1)[i], &what("∂11"));
        }
    }
}

/// `u = P(u, η)`, `P_i = 0.3·e^{−u_i} + 0.1·u_{1−i} + c_i(η)`: a contraction whose fixed point
/// the call finds at run time (Picard on the values) and differentiates implicitly. Oracle:
/// `F(u, η) = u − P(u, η) = 0`, `F_u·u′ = c′`, `F_u·u″_kl = c_kl − F_uu[u_k, u_l]`, with
/// `F_u = I + diag(0.3e^{−u}) − 0.1·swap` and `F_uu` the diagonal `−0.3e^{−u}` terms.
#[test]
fn a_recorded_fixed_point_has_its_implicit_derivatives_to_second_order() {
    let c_of = |e: [f64; 2]| -> ([f64; 2], [[f64; 2]; 2], [[[f64; 2]; 2]; 2]) {
        let c = [e[0] * e[0] + 2.0 * e[1], e[0] * e[1] - 0.5];
        let d = [[2.0 * e[0], e[1]], [2.0, e[0]]];
        let mut dd = [[[0.0; 2]; 2]; 2];
        dd[0][0][0] = 2.0;
        dd[0][1][1] = 1.0;
        dd[1][0][1] = 1.0;
        (c, d, dd)
    };
    let f = |e: &[Tr]| -> Vec<Tr> {
        let k = |c: f64| Tr::from_f64(c);
        let c = [e[0] * e[0] + k(2.0) * e[1], e[0] * e[1] - k(0.5)];
        let p = |u: &[Tr]| -> Option<Vec<Tr>> {
            Some(
                (0..2)
                    .map(|i| k(0.3) * (-u[i]).exp() + k(0.1) * u[1 - i] + c[i])
                    .collect(),
            )
        };
        let solve: Box<SolveFn> = Box::new(|maps| {
            let mut u = vec![0.0; 2];
            for _ in 0..200 {
                let v = maps[0](&u)?;
                let done = v
                    .iter()
                    .zip(&u)
                    .all(|(a, b)| (a - b).abs() <= 1e-15 * (1.0 + a.abs()));
                u = v;
                if done {
                    break;
                }
            }
            Some((u, Found::Fixed(0)))
        });
        let maps: [&dyn Fn(&[Tr]) -> Option<Vec<Tr>>; 1] = [&p];
        record_fixed(2, &[], &[0.0, 0.0], &maps, solve).expect("records")
    };
    for eta in [[0.2, 0.1], [0.4, -0.3]] {
        let got = run2(f, [0.2, 0.1], eta);
        let (c, dc, ddc) = c_of(eta);
        // The value: Picard to convergence, as the call's search.
        let mut u = [0.0; 2];
        for _ in 0..200 {
            u = std::array::from_fn(|i| 0.3 * (-u[i]).exp() + 0.1 * u[1 - i] + c[i]);
        }
        let g = [0.3 * (-u[0]).exp(), 0.3 * (-u[1]).exp()];
        let fu = nalgebra::Matrix2::new(1.0 + g[0], -0.1, -0.1, 1.0 + g[1]);
        let fi = fu.try_inverse().expect("regular");
        let du: Vec<nalgebra::Vector2<f64>> = (0..2)
            .map(|k| fi * nalgebra::Vector2::new(dc[k][0], dc[k][1]))
            .collect();
        // F_uu[a, b]_i = −0.3e^{−u_i}·a_i·b_i.
        let ddu = |k: usize, l: usize| {
            let rhs = nalgebra::Vector2::new(
                ddc[k][l][0] + g[0] * du[k][0] * du[l][0],
                ddc[k][l][1] + g[1] * du[k][1] * du[l][1],
            );
            fi * rhs
        };
        for i in 0..2 {
            let what = |s: &str| format!("u[{i}] {s} at {eta:?}");
            close(got[i][0], u[i], &what("value"));
            close(got[i][1], du[0][i], &what("∂0"));
            close(got[i][2], du[1][i], &what("∂1"));
            close(got[i][3], ddu(0, 0)[i], &what("∂00"));
            close(got[i][4], ddu(1, 0)[i], &what("∂10"));
            close(got[i][5], ddu(1, 1)[i], &what("∂11"));
        }
    }
}

/// `exp(A)` by scaling and squaring with the squarings a run-time repeat call: `A` diagonal
/// with a norm that needs several squarings, its count different at the evaluation point from
/// the recording's. Oracle: `∂ e^{a} = a′e^{a}`, `∂² e^{a} = (a″ + a′a′)e^{a}`, per entry.
#[test]
fn a_recorded_matrix_exponential_squares_as_often_as_the_values_ask() {
    // a = −4η₀² − 6, d = −3η₀η₁ − 5: ‖A‖ ≈ 10–40, two to four squarings.
    let f = |e: &[Tr]| -> Vec<Tr> {
        let k = |c: f64| Tr::from_f64(c);
        let a = [
            k(-4.0) * e[0] * e[0] - k(6.0),
            k(0.0),
            k(0.0),
            k(-3.0) * e[0] * e[1] - k(5.0),
        ];
        let mut x = crate::ode::linear::Expm::<Tr>::new(2);
        let slot = x.exp(&a).expect("exponentiates");
        let p = x.get(slot);
        vec![p[0], p[3]]
    };
    for eta in [[0.3, 0.2], [1.5, 1.8]] {
        let got = run2(f, [0.3, 0.2], eta);
        let parts = [
            (
                -4.0 * eta[0] * eta[0] - 6.0,
                [-8.0 * eta[0], 0.0],
                [[-8.0, 0.0], [0.0, 0.0]],
            ),
            (
                -3.0 * eta[0] * eta[1] - 5.0,
                [-3.0 * eta[1], -3.0 * eta[0]],
                [[0.0, -3.0], [-3.0, 0.0]],
            ),
        ];
        for (i, (a, d, dd)) in parts.iter().enumerate() {
            let e = a.exp();
            let what = |s: &str| format!("exp[{i}] {s} at {eta:?}");
            let rel = |g: f64, w: f64, s: &str| {
                assert!(g.is_finite() && w.is_finite(), "{}", what(s));
                assert!(
                    (g - w).abs() <= 1e-13 * w.abs().max(1e-300),
                    "{}: {g} vs {w}",
                    what(s)
                );
            };
            rel(got[i][0], e, "value");
            rel(got[i][1], d[0] * e, "∂0");
            rel(got[i][2], d[1] * e, "∂1");
            rel(got[i][3], (dd[0][0] + d[0] * d[0]) * e, "∂00");
            rel(got[i][4], (dd[1][0] + d[1] * d[0]) * e, "∂10");
            rel(got[i][5], (dd[1][1] + d[1] * d[1]) * e, "∂11");
        }
    }
}

/// The reverse of `f`'s plan at the first-order layout, per output seeded on its value: the
/// η-gradient each reverse returns, which must be the forward's first-order lanes (a call's
/// sub-plans reverse inside the outer reverse — re-entrant).
fn rev_grads(f: impl Fn(&[Tr]) -> Vec<Tr>, record_at: [f64; 2], eta: [f64; 2]) -> Vec<[f64; 2]> {
    begin();
    let et: Vec<Tr> = (0..2).map(|k| Tr::eta(k, record_at[k])).collect();
    let y = f(&et);
    let rec = end(Some(y.iter().map(|v| v.reg()).collect())).expect("records");
    let l1 = Layout {
        n: 2,
        m: 0,
        th: false,
        wide: false,
        tan: 0,
    };
    let plan = rec.plan(l1, 2, 0).expect("plans");
    let l = l1.lanes();
    let mut u = vec![0.0; 2 * l];
    for k in 0..2 {
        u[k * l] = eta[k];
        u[k * l + 1 + k] = 1.0;
    }
    let fwd = plan.forward_kept(l1, &u, &[]).expect("runs");
    (0..y.len())
        .map(|j| {
            let mut du_bar = vec![0.0; y.len() * l];
            du_bar[j * l] = 1.0;
            let mut ub = vec![0.0; 2 * l];
            assert!(
                plan.reverse_from(&fwd, l1, 1, &mut ub, &mut [], &du_bar),
                "reverses"
            );
            [ub[0], ub[l]]
        })
        .collect()
}

/// `exp` of a 2×2 matrix in η, large enough to need squarings: the Padé solve and the repeat call.
fn case_expm(e: &[Tr]) -> Vec<Tr> {
    let k = |c: f64| Tr::from_f64(c);
    let a = [
        k(-4.0) * e[0] * e[0] - k(6.0),
        k(0.5) * e[1],
        k(0.0),
        k(-3.0) * e[0] * e[1] - k(5.0),
    ];
    let mut x = crate::ode::linear::Expm::<Tr>::new(2);
    let slot = x.exp(&a).expect("exponentiates");
    x.get(slot).to_vec()
}

/// The fixed point of a contraction whose constant term depends on η: the fixed-point call.
fn case_fixed(e: &[Tr]) -> Vec<Tr> {
    let k = |c: f64| Tr::from_f64(c);
    let c = [e[0] * e[0] + k(2.0) * e[1], e[0] * e[1] - k(0.5)];
    let p = |u: &[Tr]| -> Option<Vec<Tr>> {
        Some(
            (0..2)
                .map(|i| k(0.3) * (-u[i]).exp() + k(0.1) * u[1 - i] + c[i])
                .collect(),
        )
    };
    let solve: Box<SolveFn> = Box::new(|maps| {
        let mut u = vec![0.0; 2];
        for _ in 0..200 {
            u = maps[0](&u)?;
        }
        Some((u, Found::Fixed(0)))
    });
    let maps: [&dyn Fn(&[Tr]) -> Option<Vec<Tr>>; 1] = [&p];
    record_fixed(2, &[], &[0.0, 0.0], &maps, solve).expect("records")
}

/// A 2×2 solve with `A` and `b` in η: the recorded LU solve.
fn case_solve(e: &[Tr]) -> Vec<Tr> {
    let k = |c: f64| Tr::from_f64(c);
    let a = [k(1e-3) + e[0], k(2.0), k(0.5), k(1.0) + e[1]];
    let b = [k(1.0) + e[0] * e[1], e[1]];
    crate::sens::linsolve::solve_linear_system_g(&a, &b, 2).expect("nonsingular")
}

type Case = (&'static str, fn(&[Tr]) -> Vec<Tr>);
const CASES: [Case; 3] = [
    ("expm", case_expm),
    ("fixed point", case_fixed),
    ("solve", case_solve),
];

/// Every call kind's reverse (the fixed point's transposed steps, the repeat call's walk back,
/// the recorded solve's refinement) against its own forward's first-order lanes.
#[test]
fn every_call_kind_reverses_to_its_forward_derivatives() {
    for (name, f) in CASES {
        for eta in [[0.3, 0.2], [1.2, 0.9]] {
            let fw = run2(f, [0.3, 0.2], eta);
            let rv = rev_grads(f, [0.3, 0.2], eta);
            for (j, (a, b)) in fw.iter().zip(&rv).enumerate() {
                for k in 0..2 {
                    close(b[k], a[1 + k], &format!("{name} out {j} ∂{k} at {eta:?}"));
                }
            }
        }
    }
}

/// Each call kind's reverse seeded on a second-order lane — a third derivative, the order the
/// θ-adjoint of a Laplace / non-Gaussian term reads — against the forward tangent of that lane
/// (a different mechanism: no reverse, so nothing it holds fixed), and the two orders of
/// differentiation against each other (`∂³` is symmetric). A refinement or Newton step under a
/// reverse leaves an error one order past the steps the forward needs (the factorization it
/// holds fixed moves with η and θ), so with only the forward's steps the checks fail —
/// measured rel 7.3e-3 on the expm's Padé solve, 6.5e-3 on the fixed point (#2786: the CTMM
/// outer gradient was 1.4e-2 off). The scalar case worked by hand: two steps give 0.534 where
/// the exact `x‴/2` is 1.2594, three give it to rounding.
#[test]
fn every_call_kind_reverses_exactly_from_second_order_lanes() {
    let l2t = Layout { tan: 1, ..l2() };
    let (l, lt) = (l2().lanes(), l2t.lanes());
    // `∂²` lane of the pair `(k, m)`, `k ≥ m`: payload order 00, 10, 11.
    let h = |k: usize, m: usize| 3 + k * (k + 1) / 2 + m;
    for (name, f) in CASES {
        begin();
        let et: Vec<Tr> = (0..2).map(|k| Tr::eta(k, [0.3, 0.2][k])).collect();
        let y = f(&et);
        let rec = end(Some(y.iter().map(|v| v.reg()).collect())).expect("records");
        let plan = rec.plan(l2(), 2, 0).expect("plans");
        for eta in [[0.3, 0.2], [1.2, 0.9]] {
            // The reverse: per output and second-order lane, `∂/∂η_d` of that lane.
            let mut u = vec![0.0; 2 * l];
            for k in 0..2 {
                u[k * l] = eta[k];
                u[k * l + 1 + k] = 1.0;
            }
            let fwd = plan.forward_kept(l2(), &u, &[]).expect("runs");
            let third_rev = |j: usize, k: usize, m: usize| -> [f64; 2] {
                let mut du_bar = vec![0.0; y.len() * l];
                du_bar[j * l + h(k, m)] = 1.0;
                let mut ub = vec![0.0; 2 * l];
                assert!(
                    plan.reverse_from(&fwd, l2(), 1, &mut ub, &mut [], &du_bar),
                    "reverses"
                );
                [ub[0], ub[l]]
            };
            for d in 0..2 {
                // The forward tangent along `η_d` of every lane.
                let mut ut = vec![0.0; 2 * lt];
                for k in 0..2 {
                    ut[k * lt] = eta[k];
                    ut[k * lt + 1 + k] = 1.0;
                    ut[k * lt + l] = if k == d { 1.0 } else { 0.0 };
                }
                let mut du = vec![0.0; (y.len() + 1) * lt];
                plan.forward_layout(l2t, &ut, &[], &mut du).expect("runs");
                for j in 0..y.len() {
                    for k in 0..2 {
                        for m in 0..=k {
                            let want = du[j * lt + l + h(k, m)];
                            let got = third_rev(j, k, m)[d];
                            close(
                                got,
                                want,
                                &format!("{name} out {j} ∂³/∂{k}∂{m}∂{d} at {eta:?}"),
                            );
                        }
                    }
                }
            }
            for j in 0..y.len() {
                // ∂³/∂0∂0∂1 two ways, and ∂³/∂1∂1∂0.
                close(
                    third_rev(j, 0, 0)[1],
                    third_rev(j, 1, 0)[0],
                    &format!("{name} out {j} 001"),
                );
                close(
                    third_rev(j, 1, 1)[0],
                    third_rev(j, 1, 0)[1],
                    &format!("{name} out {j} 110"),
                );
            }
        }
    }
}

/// [`case_fixed`] with a third coordinate every map leaves where it is (`P_2(u) = u_2`, as a
/// steady-state run-in holds a hazard accumulator): `I − J` is singular in it, so the call
/// solves over the other two and the held one keeps the search's value with no derivative.
fn case_fixed_held(e: &[Tr]) -> Vec<Tr> {
    let k = |c: f64| Tr::from_f64(c);
    let c = [e[0] * e[0] + k(2.0) * e[1], e[0] * e[1] - k(0.5)];
    let p = |u: &[Tr]| -> Option<Vec<Tr>> {
        let mut y: Vec<Tr> = (0..2)
            .map(|i| k(0.3) * (-u[i]).exp() + k(0.1) * u[1 - i] + c[i])
            .collect();
        y.push(u[2]);
        Some(y)
    };
    let solve: Box<SolveFn> = Box::new(|maps| {
        let mut u = vec![0.0; 3];
        for _ in 0..200 {
            u = maps[0](&u)?;
        }
        Some((u, Found::Fixed(0)))
    });
    let maps: [&dyn Fn(&[Tr]) -> Option<Vec<Tr>>; 1] = [&p];
    record_fixed(3, &[2], &[0.0, 0.0, 0.0], &maps, solve).expect("records")
}

/// A held coordinate leaves the others' implicit derivatives as they are without it — to
/// second order, and in the reverse — and is itself a constant: before `held`, `I − J` was
/// singular there and the call failed at every derivative layout.
#[test]
fn a_held_coordinate_is_a_constant_of_the_fixed_point() {
    for eta in [[0.3, 0.2], [1.2, 0.9]] {
        let want = run2(case_fixed, [0.3, 0.2], eta);
        let got = run2(case_fixed_held, [0.3, 0.2], eta);
        for j in 0..2 {
            for c in 0..6 {
                close(
                    got[j][c],
                    want[j][c],
                    &format!("out {j} lane {c} at {eta:?}"),
                );
            }
        }
        assert_eq!(got[2], [0.0; 6], "the held coordinate at {eta:?}");
        let rw = rev_grads(case_fixed, [0.3, 0.2], eta);
        let rg = rev_grads(case_fixed_held, [0.3, 0.2], eta);
        for j in 0..2 {
            for k in 0..2 {
                close(
                    rg[j][k],
                    rw[j][k],
                    &format!("reverse out {j} ∂{k} at {eta:?}"),
                );
            }
        }
        assert_eq!(rg[2], [0.0; 2], "the held coordinate's reverse at {eta:?}");
    }
}

/// A plan is the recording of one integrator: the same model at the same address keys apart
/// when its solver options change (method, tolerance), or a fit under Rodas5P replays the
/// plan recorded under RK45 and returns RK45's trajectory (found on #2786 as a 4e-7 gap
/// between the plan's θ-adjoint and a forward dual on a Rodas5P case run after an RK45 one).
#[test]
fn the_solver_options_key_a_plan() {
    let mut model = crate::parser::model_parser::parse_model_string(
        "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  theta TVV(30.0, 1.0, 500.0)\n  \
         omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.01\n[individual_parameters]\n  \
         CL = TVCL * exp(ETA_CL)\n  V = TVV\n[structural_model]\n  \
         ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n\
         [scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP)\n",
    )
    .expect("parses");
    let subject = crate::types::Subject::default();
    let key = |m: &crate::types::CompiledModel| SubjectKey::new(m, &subject, 1, 2);
    let rk45 = key(&model);
    fn opts(m: &mut crate::types::CompiledModel) -> &mut crate::ode::solver::OdeSolverOptions {
        &mut m.ode_spec.as_mut().expect("an ODE model").solver_opts
    }
    opts(&mut model).method = crate::ode::solver::OdeMethod::Rodas5P;
    let rodas = key(&model);
    assert_ne!(rk45, rodas, "the method keys the plan");
    let reltol = opts(&mut model).reltol;
    opts(&mut model).reltol = reltol * 0.1;
    assert_ne!(rodas, key(&model), "the tolerance keys the plan");
    opts(&mut model).reltol = reltol;
    assert_eq!(rodas, key(&model), "the same options, the same key");
}

/// More than 63 η: the direction masks spill past one word, and the first-order η-gradient of
/// `Σ c_k·η_k²` and `Π (1 + 0.01·η_k)` is still exact at every η (the ones past 63 included).
#[test]
fn a_recording_over_70_eta_plans_and_has_its_first_order_gradient() {
    const N: usize = 70;
    let l1 = Layout {
        n: N as u16,
        m: 0,
        th: false,
        wide: false,
        tan: 0,
    };
    let eta: Vec<f64> = (0..N).map(|k| 0.1 + 0.01 * k as f64).collect();
    let c: Vec<f64> = (0..N).map(|k| 1.0 + 0.1 * k as f64).collect();
    begin();
    let et: Vec<Tr> = (0..N).map(|k| Tr::eta(k, eta[k])).collect();
    let mut sum = Tr::k(0.0);
    let mut prod = Tr::k(1.0);
    for k in 0..N {
        sum = sum + Tr::k(c[k]) * et[k] * et[k];
        prod = prod * (Tr::k(1.0) + Tr::k(0.01) * et[k]);
    }
    let rec = end(Some(vec![sum.reg(), prod.reg()])).expect("records");
    let plan = rec.plan(l1, N, 0).expect("plans past 63 η");
    let l = l1.lanes();
    let mut u = vec![0.0; N * l];
    for k in 0..N {
        u[k * l] = eta[k];
        u[k * l + 1 + k] = 1.0;
    }
    let mut du = vec![0.0; 3 * l];
    plan.forward_layout(l1, &u, &[], &mut du).expect("runs");
    assert_eq!(du[2 * l], 1.0, "the plan's checks hold");
    let p: f64 = eta.iter().map(|e| 1.0 + 0.01 * e).product();
    close(du[0], (0..N).map(|k| c[k] * eta[k] * eta[k]).sum(), "sum");
    close(du[l], p, "prod");
    for j in 0..N {
        close(du[1 + j], 2.0 * c[j] * eta[j], &format!("d sum/d eta {j}"));
        close(
            du[l + 1 + j],
            0.01 * p / (1.0 + 0.01 * eta[j]),
            &format!("d prod/d eta {j}"),
        );
    }
}

#[test]
fn dir_mask_ops_match_the_single_word_algebra_and_spill_past_it() {
    let (a, b) = (DirMask::eta(2), DirMask::eta(70));
    let u = a.clone() | &b;
    assert!(u.has(2) && u.has(70) && !u.has(3) && !u.has(69) && !u.has(200));
    assert!(u.any_below(71) && !u.any_below(2) && !b.any_below(70) && b.any_below(71));
    assert!(DirMask::FREE.no_dirs() && !DirMask::FREE.any_below(100) && !u.no_dirs());
    assert!(DirMask::ALL.has(500) && DirMask::ALL.any_below(1) && !DirMask::NONE.any_below(1));
    assert_eq!(&u & &DirMask::ALL, u);
    assert_eq!(&a & &b, DirMask::FREE);
    assert_eq!(&u & &DirMask::NONE, DirMask::NONE);
    assert_eq!(DirMask::NONE | &DirMask::FREE, DirMask::FREE);
}

/// A plan over 2 η (`η₀²`), for the variant-list tests.
fn square_plan() -> Entry {
    begin();
    let et: Vec<Tr> = (0..2).map(|k| Tr::eta(k, 1.0)).collect();
    let rec = end(Some(vec![(et[0] * et[0]).reg()])).expect("records");
    Arc::new(rec.plan(l2(), 2, 0).expect("plans"))
}

/// A forward stops at the first chunk whose guard fails: three chunks, the guard (`η₀² > 1`)
/// decided in the first, and the point on the other side of it runs that chunk alone.
#[test]
fn a_failed_guard_stops_the_forward_at_its_chunk() {
    begin();
    let e: Vec<Tr> = (0..2).map(|k| Tr::eta(k, [2.0, 1.0][k])).collect();
    let a = e[0] * e[0];
    assert!(a.br_cmp_c(crate::sens::num::BranchCmp::Gt, 1.0));
    Tr::chunk_mark();
    let b = e[1] * e[1] * e[0];
    Tr::chunk_mark();
    let c = b * e[1] + a;
    let rec = end(Some(vec![c.reg()])).expect("records");
    let plan = rec.plan(l2(), 2, 0).expect("plans");
    assert_eq!(plan.calls.len(), 3, "one call per chunk");
    let l = l2().lanes();
    let run = |eta: [f64; 2]| {
        let mut u = vec![0.0; 2 * l];
        for k in 0..2 {
            u[k * l] = eta[k];
            u[k * l + 1 + k] = 1.0;
        }
        let mut du = vec![0.0; 2 * l];
        CALLS_RUN.with(|n| n.set(0));
        let f = plan.forward_layout(l2(), &u, &[], &mut du).expect("runs");
        (CALLS_RUN.with(|n| n.get()), du[l], f)
    };
    let (ran, guard, _) = run([2.0, 1.5]);
    assert_eq!((ran, guard), (3, 1.0), "the guard holds: every call runs");
    let (ran, guard, f) = run([0.5, 1.5]);
    assert_eq!((ran, guard), (1, 0.0), "the guard fails in the first call");
    assert!(!plan.kept_guard(&f));
}

/// Variants are tried most recently serving first.
#[test]
fn variants_are_tried_most_recent_first() {
    let mut v = Variants::default();
    for _ in 0..3 {
        v = v.with(square_plan());
    }
    let order = |v: &Variants| v.order().map(|(i, _)| i).collect::<Vec<_>>();
    assert_eq!(order(&v), [2, 0, 1], "the newest first");
    v.hit(0);
    assert_eq!(order(&v), [0, 1, 2]);
    v.hit(1);
    assert_eq!(order(&v), [1, 2, 0]);
}

/// Past `MAX_VARIANTS` the oldest variant goes unless it served last; then the next oldest.
#[test]
fn a_full_variant_list_keeps_the_one_that_served_last() {
    let first = square_plan();
    let mut v = Variants::default().with(first.clone());
    let second = square_plan();
    v = v.with(second.clone());
    for _ in 2..MAX_VARIANTS {
        v = v.with(square_plan());
    }
    assert_eq!(v.len(), MAX_VARIANTS);
    v.hit(0);
    v = v.with(square_plan());
    assert_eq!(v.len(), MAX_VARIANTS);
    assert!(
        v.iter().any(|p| Arc::ptr_eq(p, &first)),
        "the touched one stays"
    );
    assert!(
        !v.iter().any(|p| Arc::ptr_eq(p, &second)),
        "the oldest other one goes"
    );
    assert_eq!(v.order().next().map(|(i, _)| i), Some(MAX_VARIANTS - 1));
}

/// `add` refreshes the adding thread's last-subject entry rather than invalidating every
/// thread's, and a later variant joins the list.
#[test]
fn add_refreshes_this_threads_last_subject() {
    let model = crate::parser::model_parser::parse_model_string(
        "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  theta TVV(30.0, 1.0, 500.0)\n  \
         omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.01\n[individual_parameters]\n  \
         CL = TVCL * exp(ETA_CL)\n  V = TVV\n[structural_model]\n  \
         ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n\
         [scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP)\n",
    )
    .expect("parses");
    let subject = crate::types::Subject::default();
    let key = SubjectKey::new(&model, &subject, 1, 2).with_variant(Desc::of_value(&0x6a7e_0001u64));
    let last = || LAST.with(|l| l.borrow().as_ref().map(|(_, k, ks)| (k.clone(), ks.len())));
    // Any fit or parse on another thread resets the global table, so what the second `add`
    // finds there is not this test's to assert: the variants it returns are, and `LAST` lists them.
    let ks = add(&key, square_plan());
    assert_eq!(last(), Some((key.clone(), ks.len())));
    let two = ks.with(square_plan());
    assert_eq!(two.len(), 2);
}

/// A registered fit's keys read the fingerprints computed at registration, the same values a
/// per-call hash gives, and dropping the scope releases them: a subject mutated afterwards keys
/// by its new contents. Dies when `print_of` serves an entry past its scope (a stale key) or
/// registers a different value than the per-call hash (two keys for one subject).
#[test]
fn a_registered_fit_keys_by_its_fingerprints_and_releases_them() {
    let model = crate::parser::model_parser::parse_model_string(
        "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.01\n\
         [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = 30.0\n[structural_model]\n  \
         pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(PROP)\n",
    )
    .expect("parses");
    let mut subjects = vec![crate::types::Subject::default()];
    subjects[0].obs_times = vec![1.0, 2.0];
    let prints = |k: SubjectKey| (k.model, k.subject);
    let hashed = prints(SubjectKey::new(&model, &subjects[0], 1, 1));
    {
        let _scope = register(&model, &subjects);
        let k = SubjectKey::new(&model, &subjects[0], 1, 1);
        assert_ne!(
            k.scope, 0,
            "a registered fit keys its plans in its own scope"
        );
        assert_eq!(prints(k), hashed, "registered = hashed");
    }
    subjects[0].obs_times.push(3.0);
    let moved = prints(SubjectKey::new(&model, &subjects[0], 1, 1));
    assert_ne!(moved, hashed, "a released subject keys by its new contents");
    let _scope = register(&model, &subjects);
    assert_eq!(prints(SubjectKey::new(&model, &subjects[0], 1, 1)), moved);
}

/// A registered fit's plans are its own: `reset()` (any parse, any unregistered caller on another
/// thread) leaves them, and the scope's end drops them. Dies when `reset` clears every scope (the
/// per-fit race this replaced) or a scope's plans outlive it (memory held across fits).
#[test]
fn a_fits_plans_survive_resets_and_leave_with_its_scope() {
    let model = crate::parser::model_parser::parse_model_string(
        "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.01\n\
         [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = 30.0\n[structural_model]\n  \
         pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(PROP)\n",
    )
    .expect("parses");
    let mut subjects = vec![crate::types::Subject::default()];
    subjects[0].obs_times = vec![0.5, 1.5];
    let scope = register(&model, &subjects);
    let key =
        SubjectKey::new(&model, &subjects[0], 1, 1).with_variant(Desc::of_value(&0x5c0_9eu64));
    assert_ne!(
        key,
        SubjectKey::new(&model, &subjects[0], 1, 1),
        "variant keys apart"
    );
    add(&key, square_plan());
    reset();
    assert!(
        matches!(lookup(&key), Lookup::Found(_)),
        "a reset leaves a registered fit's plans"
    );
    drop(scope);
    assert!(
        matches!(lookup(&key), Lookup::Missing),
        "the scope's end drops its plans"
    );
}

/// `add` hands back the variants it stored, so a caller serves from them and never looks the key
/// up again: a `reset` on another thread between the two (any fit or parse resets the global
/// table) made that lookup `Missing`, and the term came back `None`, scored as the 1e20 sentinel.
#[test]
fn add_returns_the_stored_variants_not_a_second_lookup() {
    let model = crate::parser::model_parser::parse_model_string(
        "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  theta TVV(30.0, 1.0, 500.0)\n  \
         omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.01\n[individual_parameters]\n  \
         CL = TVCL * exp(ETA_CL)\n  V = TVV\n[structural_model]\n  \
         ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n\
         [scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP)\n",
    )
    .expect("parses");
    let subject = crate::types::Subject::default();
    let key = SubjectKey::new(&model, &subject, 1, 2).with_variant(Desc::of_value(&0x6a7e_0002u64));
    let ks = add(&key, square_plan());
    assert_eq!(ks.len(), 1);
    // The interleaving, forced: the table is emptied right after `add`.
    reset();
    assert!(matches!(lookup(&key), Lookup::Missing));
    assert_eq!(ks.len(), 1, "the returned variants outlive the reset");
}

/// An ODE segment sampled strictly inside accepted steps (`interp_at`), as a hazard read-out
/// does: every state component at the hard point, then at each soft time.
fn case_ode_interp(method: crate::ode::solver::OdeMethod) -> impl Fn(&[Tr]) -> Vec<Tr> {
    move |e: &[Tr]| {
        let k = |c: f64| Tr::from_f64(c);
        let p = [k(0.8) * e[0].exp(), k(0.5) * e[1].exp()];
        let u0 = [k(1.0) + k(0.2) * e[1], k(0.4) + k(0.1) * e[0]];
        let rhs = |u: &[Tr], p: &[Tr], _t: Tr, du: &mut [Tr]| {
            du[0] = -p[0] * u[0] - p[1] * u[0] * u[1];
            du[1] = p[0] * u[0] - k(0.3) * u[1];
        };
        let opts = crate::ode::solver::OdeSolverOptions {
            reltol: 1e-3,
            abstol: 1e-5,
            method,
            auto_switch: false,
            ..Default::default()
        };
        let mut auto = crate::ode::solver::OdeAutoSwitchState::default();
        let (hard, soft) = Tr::ode_segment(
            &rhs,
            &u0,
            (0.0, 3.0),
            &p,
            &[3.0],
            &[0.7, 1.3, 2.9],
            &opts,
            &mut auto,
        )
        .expect("records as a loop call");
        hard.iter().chain(&soft).flat_map(|s| s.u.clone()).collect()
    }
}

/// A sample read off the continuous extension inside a step reverses exactly: the plan's reverse
/// η-adjoint of each output equals its own first-order forward tangent, under every stepper
/// family (Hermite for RK45, a tableau's own for Vern7 and Rodas5P, `ode23s`'s quadratic), and
/// the run's log does carry in-step read-outs strictly between a step's ends.
#[test]
fn an_interpolated_sample_reverses_like_the_forward_tangent() {
    use crate::ode::solver::OdeMethod;
    let l1 = Layout {
        n: 2,
        m: 0,
        th: false,
        wide: false,
        tan: 0,
    };
    let l = l1.lanes();
    for method in [
        OdeMethod::Rk45,
        OdeMethod::Vern7,
        OdeMethod::Rosenbrock23,
        OdeMethod::Rodas5P,
    ] {
        let f = case_ode_interp(method);
        begin();
        let et: Vec<Tr> = (0..2).map(|k| Tr::eta(k, [0.1, -0.2][k])).collect();
        let y = f(&et);
        let rec = end(Some(y.iter().map(|v| v.reg()).collect())).expect("records");
        let plan = rec.plan(l1, 2, 0).expect("plans");
        let eta = [0.15, -0.1];
        let mut u = vec![0.0; 2 * l];
        for k in 0..2 {
            u[k * l] = eta[k];
            u[k * l + 1 + k] = 1.0;
        }
        let mut du = vec![0.0; (y.len() + 1) * l];
        plan.forward_layout(l1, &u, &[], &mut du).expect("runs");
        assert_eq!(du[y.len() * l], 1.0, "{method:?}: the plan's checks hold");
        let fwd = plan.forward_kept(l1, &u, &[]).expect("runs");
        let inside = fwd
            .logs
            .iter()
            .flat_map(|lg| match lg {
                CallLog::Ode(lg) => lg.entries.as_slice(),
                _ => &[],
            })
            .filter(|e| matches!(e, ode_loop::Entry::Interp { theta, .. } if *theta > 0.0 && *theta < 1.0))
            .count();
        assert!(inside >= 6, "{method:?}: {inside} read-outs inside a step");
        // Measured worst relative gap over the 8 outputs × 2 η: 8.7e-17 (RK45), 7.3e-16 (Vern7),
        // 4.3e-11 (Rosenbrock23), 2.3e-11 (Rodas5P). The stiff pair's gap is the factorization's
        // refinement noise, as large on the hard end point (≤ 8.6e-12) as on the interpolated
        // samples, so it is not the read-out's; the bounds leave ~10× over each family.
        let tol = if matches!(method, OdeMethod::Rk45 | OdeMethod::Vern7) {
            1e-13
        } else {
            5e-10
        };
        let mut worst = 0.0f64;
        for j in 0..y.len() {
            let mut du_bar = vec![0.0; y.len() * l];
            du_bar[j * l] = 1.0;
            let mut ub = vec![0.0; 2 * l];
            assert!(
                plan.reverse_from(&fwd, l1, 1, &mut ub, &mut [], &du_bar),
                "{method:?}: output {j} reverses"
            );
            for (k, got) in [ub[0], ub[l]].into_iter().enumerate() {
                let want = du[j * l + 1 + k];
                let rel = (got - want).abs() / (1.0 + want.abs());
                assert!(rel.is_finite());
                assert!(
                    rel < tol,
                    "{method:?} out {j} ∂{k}: {got} vs {want} (rel {rel:e})"
                );
                worst = worst.max(rel);
            }
        }
        assert!(worst.is_finite(), "{method:?}: the gaps are finite");
    }
}

// ── The tiers of one chunk, bit for bit ────────────────────────────────────────────────

/// `exp(−(a0/a2)·a3)` over four inputs at the first-order layout over 3 η, `a3` carrying no
/// tangent and `a0`, `a2` none along η₂: output lane 3 is `exp(v)·(−1·t)` with `t` a structural
/// zero — the folded tiers write the constant `+0` where the dual operations give `−0`.
fn zero_tangent_chunk(live: bool) -> (ChunkProgram, Layout) {
    let code = vec![
        Flat::LoadState { d: 0, i: 0 },
        Flat::LoadState { d: 1, i: 1 },
        Flat::LoadState { d: 2, i: 2 },
        Flat::LoadState { d: 3, i: 3 },
        Flat::DivRaw { d: 4, a: 0, b: 2 },
        Flat::Mul { d: 5, a: 4, b: 3 },
        Flat::MulK {
            d: 6,
            a: 5,
            c: -1.0,
        },
        Flat::Unary {
            d: 7,
            op: Unary::Exp,
            a: 6,
        },
        Flat::Deriv { state: 0, a: 7 },
    ];
    let two = DirMask::eta(0) | &DirMask::eta(1);
    let spec = vec![
        (two.clone(), u16::MAX),
        (DirMask::ALL, u16::MAX),
        (two, u16::MAX),
        // `live`: the same zero tangent, but unknown to the folding (it computes the `-0`).
        (if live { DirMask::ALL } else { DirMask::NONE }, u16::MAX),
    ];
    let layout = Layout {
        n: 3,
        m: 0,
        th: false,
        wide: false,
        tan: 0,
    };
    let cp = ChunkProgram {
        code,
        n_regs: 8,
        spec,
        n_in: 4,
        compiled: Mutex::new(HashMap::new()),
        lane: Default::default(),
    };
    (cp, layout)
}

/// The inputs' lanes: values `v`, tangents of `a0`/`a2` along η₀, η₁ only, `a1` along all.
fn zero_tangent_inputs(v: [f64; 4], l: usize) -> Vec<f64> {
    let t = [
        [0.0, 0.7, -0.4, 0.0],
        [0.0, 0.3, 0.9, -0.5],
        [0.0, -0.6, 0.2, 0.0],
        [0.0; 4],
    ];
    let mut base = vec![0.0; 5 * l];
    for i in 0..4 {
        base[i * l] = v[i];
        base[i * l + 1..(i + 1) * l].copy_from_slice(&t[i][1..l]);
    }
    base
}

/// Every tier of the forward gives the same bits, signed zeros included, across the signs the
/// zero lane can take. Kills: the folded tier writing `+0` where the dual operations carry `-0`
/// (`canonical_zero` removed from either side).
#[test]
fn the_forward_tiers_agree_bit_for_bit_on_a_structurally_zero_tangent() {
    for v in [
        [1.3, 0.4, 2.0, 0.9],
        [-1.3, 0.4, -2.0, -0.9],
        [0.0, 1.0, 3.0, 2.0],
    ]
    .into_iter()
    .flat_map(|v| [(v, false), (v, true)])
    {
        let (v, live) = v;
        let (cp, layout) = zero_tangent_chunk(live);
        let l = layout.lanes();
        let t = cp.get(layout, false).expect("a kernel");
        let idx: Vec<u32> = (0..5).collect();
        let lits = [0.0f64];
        let mut interp = zero_tangent_inputs(v, l);
        unsafe {
            t.interpret(
                interp.as_mut_ptr(),
                std::ptr::null_mut(),
                idx.as_ptr(),
                lits.as_ptr(),
                1,
                0,
            )
        };
        t.promote(true, "none");
        assert_ne!(
            t.native.load(std::sync::atomic::Ordering::Acquire),
            0,
            "compiled"
        );
        let mut native = zero_tangent_inputs(v, l);
        unsafe {
            t.call(
                native.as_mut_ptr(),
                std::ptr::null_mut(),
                idx.as_ptr(),
                lits.as_ptr(),
            )
        };
        for c in 0..l {
            assert_eq!(
                interp[4 * l + c].to_bits(),
                native[4 * l + c].to_bits(),
                "{v:?} live={live} lane {c}: interpreter {:e}, native {:e}",
                interp[4 * l + c],
                native[4 * l + c]
            );
        }
        // The lane the fold knows is zero is the one the two used to disagree on.
        assert_eq!(native[4 * l + 3].to_bits(), 0.0f64.to_bits(), "{v:?}: a +0");
    }
}

/// The multi-seed reverse of the same chunk: native and interpreter agree bit for bit on
/// every input's adjoint lanes, from several seeds at once (`call_seeds`).
#[test]
fn the_reverse_tiers_agree_bit_for_bit_across_seeds() {
    let (cp, layout) = zero_tangent_chunk(false);
    let l = layout.lanes();
    let t = cp.get(layout, true).expect("a kernel");
    let idx: Vec<u32> = (0..5).collect();
    let lits = [0.0f64];
    let (seeds, stride) = (3, 5 * l);
    let init = || {
        let mut adj = vec![0.0; seeds * stride];
        for sd in 0..seeds {
            for c in 0..l {
                adj[sd * stride + 4 * l + c] = (sd as f64 + 1.0) * [1.0, -0.5, 0.25, 0.75][c];
            }
        }
        adj
    };
    let v = [1.3, 0.4, 2.0, 0.9];
    let (mut b0, mut b1) = (zero_tangent_inputs(v, l), zero_tangent_inputs(v, l));
    let (mut a0, mut a1) = (init(), init());
    unsafe {
        t.interpret(
            b0.as_mut_ptr(),
            a0.as_mut_ptr(),
            idx.as_ptr(),
            lits.as_ptr(),
            seeds,
            stride,
        )
    };
    t.promote(true, "none");
    unsafe {
        t.call_seeds(
            b1.as_mut_ptr(),
            a1.as_mut_ptr(),
            stride,
            seeds,
            idx.as_ptr(),
            lits.as_ptr(),
        )
    };
    assert!(a0[..4 * l].iter().any(|&x| x != 0.0), "a live adjoint");
    for k in 0..a0.len() {
        assert_eq!(
            a0[k].to_bits(),
            a1[k].to_bits(),
            "adjoint element {k}: {:e} vs {:e}",
            a0[k],
            a1[k]
        );
    }
}

/// `√(a0·a1 + a2) / (a1 − a0) · |a2|` over three dense inputs at the packed second-order
/// layout over 3 η (10 lanes): every dual operation's tangent and payload lanes run as `f64x2`
/// packs ([`simd`]), lane 0 and the odd lane out as scalars.
fn simd_chunk() -> (ChunkProgram, Layout) {
    let code = vec![
        Flat::LoadState { d: 0, i: 0 },
        Flat::LoadState { d: 1, i: 1 },
        Flat::LoadState { d: 2, i: 2 },
        Flat::Mul { d: 3, a: 0, b: 1 },
        Flat::Add { d: 4, a: 3, b: 2 },
        Flat::Unary {
            d: 5,
            op: Unary::Sqrt,
            a: 4,
        },
        Flat::Sub { d: 6, a: 1, b: 0 },
        Flat::DivRaw { d: 7, a: 5, b: 6 },
        Flat::Unary {
            d: 8,
            op: Unary::Abs,
            a: 2,
        },
        Flat::Mul { d: 9, a: 7, b: 8 },
        Flat::Deriv { state: 0, a: 9 },
        Flat::Deriv { state: 1, a: 7 },
    ];
    let layout = Layout {
        n: 3,
        m: 6,
        th: false,
        wide: false,
        tan: 0,
    };
    let cp = ChunkProgram {
        code,
        n_regs: 10,
        spec: vec![(DirMask::ALL, u16::MAX); 3],
        n_in: 3,
        compiled: Mutex::new(HashMap::new()),
        lane: Default::default(),
    };
    (cp, layout)
}

/// The inputs' lanes, every one live.
fn simd_inputs(l: usize) -> Vec<f64> {
    let mut base = vec![0.0; 5 * l];
    for i in 0..3 {
        for c in 0..l {
            base[i * l + c] = [1.3, 0.4, 2.0][i] + 0.37 * c as f64 - 0.11 * (i * c) as f64;
        }
    }
    base
}

/// The two-lane vector code gives the scalar code's bits, forward and multi-seed reverse,
/// and it is the vector code that ran: the chunk packs. Kills: a pack whose lanes are swapped
/// or read from the wrong operand, a vector store missing the forward's `+0` canonicalisation,
/// a reverse pack scheduled across the adjoint store it reads back.
#[test]
fn the_vector_tiers_agree_bit_for_bit_with_the_scalar_ones() {
    let (cp, layout) = simd_chunk();
    let l = layout.lanes();
    let idx: Vec<u32> = (0..5).collect();
    let lits = [0.0f64];
    // The forward.
    let t = cp.get(layout, false).expect("a kernel");
    let p = simd::packs(&t.f, l, false, 0).expect("the forward packs");
    let vec_ops = p
        .members
        .iter()
        .filter(|m| {
            !matches!(
                t.f.insts[m[0] as usize],
                Inst::Load { .. } | Inst::Store { .. }
            )
        })
        .count();
    assert!(vec_ops >= 20, "{vec_ops} vector operations");
    let mut interp = simd_inputs(l);
    unsafe {
        t.interpret(
            interp.as_mut_ptr(),
            std::ptr::null_mut(),
            idx.as_ptr(),
            lits.as_ptr(),
            1,
            0,
        )
    };
    t.promote(false, "speed");
    let mut native = simd_inputs(l);
    unsafe {
        t.call(
            native.as_mut_ptr(),
            std::ptr::null_mut(),
            idx.as_ptr(),
            lits.as_ptr(),
        )
    };
    assert!(interp[3 * l..].iter().all(|x| x.is_finite()));
    for c in 3 * l..5 * l {
        assert_eq!(interp[c].to_bits(), native[c].to_bits(), "forward lane {c}");
    }
    // The reverse, several seeds at once.
    let t = cp.get(layout, true).expect("a kernel");
    assert!(
        simd::packs(&t.f, l, true, t.prefix).is_some(),
        "the reverse packs"
    );
    let (seeds, stride) = (3, 5 * l);
    let init = || {
        let mut adj = vec![0.0; seeds * stride];
        for sd in 0..seeds {
            for c in 0..2 * l {
                adj[sd * stride + 3 * l + c] = (sd as f64 + 1.0) * (0.3 + 0.17 * c as f64);
            }
        }
        adj
    };
    let (mut b0, mut b1) = (simd_inputs(l), simd_inputs(l));
    let (mut a0, mut a1) = (init(), init());
    unsafe {
        t.interpret(
            b0.as_mut_ptr(),
            a0.as_mut_ptr(),
            idx.as_ptr(),
            lits.as_ptr(),
            seeds,
            stride,
        )
    };
    t.promote(false, "speed");
    unsafe {
        t.call_seeds(
            b1.as_mut_ptr(),
            a1.as_mut_ptr(),
            stride,
            seeds,
            idx.as_ptr(),
            lits.as_ptr(),
        )
    };
    assert!(a0.iter().all(|x| x.is_finite()));
    assert!(a0[..3 * l].iter().any(|&x| x != 0.0), "a live adjoint");
    for k in 0..a0.len() {
        assert_eq!(a0[k].to_bits(), a1[k].to_bits(), "adjoint element {k}");
    }
}

/// A covariate map's fingerprint ties each value to its name and does not depend on the map's
/// iteration order: hashing values alone, in iteration order, let swapped values collide and
/// equal maps key apart at random (#2786).
#[test]
fn a_covariate_value_keys_a_plan_by_its_name() {
    let subject = |pairs: &[(&str, f64)]| {
        let mut s = crate::types::Subject::default();
        s.covariates = pairs.iter().map(|&(k, v)| (k.to_string(), v)).collect();
        s
    };
    let a = subject(&[("WT", 70.0), ("AGE", 40.0)]);
    let swapped = subject(&[("WT", 40.0), ("AGE", 70.0)]);
    // Same contents built in the other order: a different map, possibly a different iteration.
    let again = subject(&[("AGE", 40.0), ("WT", 70.0)]);
    assert_ne!(
        subject_desc(&a),
        subject_desc(&swapped),
        "a value belongs to its name"
    );
    assert_eq!(
        subject_desc(&a),
        subject_desc(&again),
        "equal maps key equal whatever their iteration order"
    );
}

/// The exact Jacobian of a right-hand side with a kink (`kernel_jit::jacobian`, what a
/// Rosenbrock step's start body records) holds on both sides of it: `|u|`'s tangent is a
/// selection on the sign of `u`, not a branch. Recorded at `u = 0.5` and run at `u = −0.5`,
/// `J = sign(u)·p` must read `−p` with the plan's checks holding. As a branch the tangent was a
/// guard, and inside an ODE loop body a failed guard declines the whole segment: the
/// myelosuppression chain's `abs(CIRC0/rb − 1)` sent every subject whose `rb` crossed `CIRC0`
/// to FD (`an_exact_h_is_never_counted_as_redone`). Dies when the `Unary::Abs` rule branches.
#[test]
fn a_recorded_jacobian_holds_across_an_abs_kink() {
    let jac_of = |e: &[Tr]| -> Vec<Tr> {
        let rhs = |u: &[Tr], p: &[Tr], _t: Tr, du: &mut [Tr]| du[0] = u[0].abs() * p[0];
        let (mut f, mut j, mut ft) = ([Tr::k(0.0)], [Tr::k(0.0)], [Tr::k(0.0)]);
        jacobian(&rhs, &e[..1], &e[1..], Tr::k(0.0), &mut f, &mut j, &mut ft);
        vec![j[0]]
    };
    for (at, sign) in [(0.5, 1.0), (-0.5, -1.0)] {
        let out = run2(jac_of, [0.5, 3.0], [at, 3.0]);
        // Lanes: value, ∂/∂u, ∂/∂p, then the second order.
        close(out[0][0], sign * 3.0, "J");
        close(out[0][2], sign, "dJ/dp");
        close(out[0][1], 0.0, "dJ/du");
    }
}

/// Rows whose maps differ in more than one value key apart. The fixture is profile fixture
/// B2's ten studies (`nonmem_anchor/profile_logit_b2.csv`): per subject `STUDY`, `TRT` and the
/// level index `__level_PLACEBO`, the study's four levels across its rows. Summing `Fx`'s
/// unmixed per-entry hashes collided on them (studies 5, 6 and 7 on one fingerprint, 9 and 10
/// on another), and a colliding subject was served the first one's plan, whose gather reads
/// that subject's level θs: an exact gradient and Newton matrix of zero over its own levels
/// (the five `profile::tests::logit` failures). The descriptor is exact content (name-sorted
/// entries), so these rows can only key apart.
#[test]
fn subject_fingerprints_tell_apart_rows_differing_in_two_values() {
    let row = |study: f64, trt: f64, level: f64| -> HashMap<String, f64> {
        [("STUDY", study), ("TRT", trt), ("__level_PLACEBO", level)]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect()
    };
    let prints: Vec<Desc> = (1..=10)
        .map(|k| {
            let (study, first) = (k as f64, (4 * k - 3) as f64);
            let mut s = crate::types::Subject::default();
            s.covariates = row(study, 0.0, first);
            s.obs_covariates = (0..4)
                .flat_map(|v| [0.0, 1.0].map(|trt| row(study, trt, first + v as f64)))
                .collect();
            subject_desc(&s)
        })
        .collect();
    for (a, pa) in prints.iter().enumerate() {
        for (b, pb) in prints.iter().enumerate().skip(a + 1) {
            assert_ne!(
                pa,
                pb,
                "studies {} and {} share a fingerprint",
                a + 1,
                b + 1
            );
        }
    }
}

/// Covariate values that differ only in their `f64` top bits (whole numbers `1.0`..`6.0`: a level
/// index) key apart — and do so even when every descriptor hash collides (`COLLIDE`), because a hit
/// compares the descriptor's bytes. The old summed 64-bit fingerprint collapsed STUDY 2..6 to one
/// key and served a level-block subject another's plan (`level_two_block_nonmem_anchor` 13.8 OFV off).
#[test]
fn whole_number_covariate_values_key_apart() {
    let desc = |level: f64| {
        let mut s = crate::types::Subject::default();
        s.covariates = [("STUDY", level), ("__level_PLACEBO", level), ("OCC", 1.0)]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        subject_desc(&s)
    };
    COLLIDE.with(|c| c.set(true));
    let descs: Vec<Desc> = (1..=6).map(|l| desc(f64::from(l))).collect();
    COLLIDE.with(|c| c.set(false));
    assert!(descs.iter().all(|d| d.hash == 0), "the hashes all collide");
    for i in 0..descs.len() {
        for j in 0..i {
            assert_ne!(
                descs[i],
                descs[j],
                "levels {} and {} share a plan key",
                i + 1,
                j + 1
            );
        }
    }
}

/// Two never-parsed struct-literal models that differ only in what no hash sees (a closure) key
/// apart, whatever address each lands at, and a dropped model's plans go with it. Dies when the
/// key goes back to the model's address (a model built where a dropped one lived is served its
/// plan) or when the id's drop leaves plans behind (memory held for dead models).
#[test]
fn a_model_keys_its_plans_by_identity_and_they_leave_with_it() {
    let subject = crate::types::Subject::default();
    let build = |scale: f64| {
        let mut m =
            crate::types::test_helpers::analytical_model(crate::types::GradientMethod::default());
        m.pk_param_fn = Box::new(move |_, _, _, _| {
            let mut p = crate::types::PkParams::default();
            p.values[0] = scale;
            p
        });
        m
    };
    let (k1, id1, addr1) = {
        let m = build(1.0);
        let k = SubjectKey::new(&m, &subject, 1, 2).with_variant(Desc::of_value(&0x1d_e171u64));
        add(&k, square_plan());
        assert!(matches!(lookup(&k), Lookup::Found(_)));
        (k, m.plan_id.get(), &m as *const _ as usize)
    };
    // The first model is gone: its plans went with its identity.
    assert!(
        matches!(lookup(&k1), Lookup::Missing),
        "a dropped model's plans are gone"
    );
    // Build several, some usually at the freed address; the identity differs either way, so the
    // key must.
    let mut keep = Vec::new();
    for _ in 0..8 {
        let m = build(2.0);
        let same = &m as *const _ as usize == addr1;
        let k = SubjectKey::new(&m, &subject, 1, 2).with_variant(Desc::of_value(&0x1d_e171u64));
        assert_ne!(m.plan_id.get(), id1);
        assert_ne!(
            k, k1,
            "a model at a freed address keys apart (same address: {same})"
        );
        keep.push(m);
    }
    assert_ne!(
        SubjectKey::new(&keep[0], &subject, 1, 2),
        SubjectKey::new(&keep[1], &subject, 1, 2),
        "two live models identical but for a closure key apart"
    );
}

/// Plan identity is exact, never a hash alone (#2786): with every key descriptor hashed to `0`
/// on this thread ([`COLLIDE`]), subjects of different data, observed values, configurations
/// and layouts share one bucket, and equality alone still gives each its own plan. Dies when a
/// key compares a descriptor's hash instead of its bytes.
#[test]
fn colliding_keys_are_told_apart_by_their_content() {
    COLLIDE.with(|c| c.set(true));
    let model =
        crate::types::test_helpers::analytical_model(crate::types::GradientMethod::default());
    let mut a = crate::types::Subject::default();
    a.obs_times = vec![1.0];
    a.observations = vec![2.0];
    let mut b = a.clone();
    b.obs_times = vec![1.5];
    let mut a_dv = a.clone();
    a_dv.observations = vec![3.0];
    let bucket = |k: &SubjectKey| {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        k.hash(&mut h);
        h.finish()
    };
    let key = |s: &crate::types::Subject| SubjectKey::new(&model, s, 1, 2);
    let (ka, kb) = (key(&a), key(&b));
    assert_eq!(bucket(&ka), bucket(&kb), "the stub puts both in one bucket");
    assert_ne!(ka, kb, "different data are different keys");
    add(&ka, square_plan());
    assert!(
        matches!(lookup(&kb), Lookup::Missing),
        "b is not served a's plan"
    );
    assert!(
        matches!(lookup(&key(&a.clone())), Lookup::Found(_)),
        "an equal subject at another address is"
    );
    assert_ne!(
        ka.clone().with_observed(&a),
        ka.clone().with_observed(&a_dv)
    );
    let v = |x: u8| ka.clone().with_variant(Desc::of_value(&x));
    assert_ne!(v(1), v(2));
    assert_eq!(v(1), v(1));
    COLLIDE.with(|c| c.set(false));
}
