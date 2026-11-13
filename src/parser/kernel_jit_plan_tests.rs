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
