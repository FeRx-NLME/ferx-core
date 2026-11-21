//! Compiled plans against derivatives derived independently, by hand, in `f64` (no finite
//! differences): the recording type's linear solve (values factored at run time, derivatives by
//! refinement against that factorization) and its steady-state call (the implicit function
//! theorem), each to second order in η.
use super::*;
use crate::sens::num::PkNum;

/// The second-order layout over 2 η: value, `∂_0`, `∂_1`, `∂²_00`, `∂²_10`, `∂²_11`.
fn l2() -> Layout {
    Layout { n: 2, m: 3, th: false, wide: false, tan: 0 }
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
    (0..y.len()).map(|j| std::array::from_fn(|c| du[j * l + c])).collect()
}

fn close(got: f64, want: f64, what: &str) {
    assert!(got.is_finite() && want.is_finite(), "{what}: {got} vs {want}");
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
            Some((0..2).map(|i| k(0.3) * (-u[i]).exp() + k(0.1) * u[1 - i] + c[i]).collect())
        };
        let solve: Box<SolveFn> = Box::new(|maps| {
            let mut u = vec![0.0; 2];
            for _ in 0..200 {
                let v = maps[0](&u)?;
                let done = v.iter().zip(&u).all(|(a, b)| (a - b).abs() <= 1e-15 * (1.0 + a.abs()));
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
        let du: Vec<nalgebra::Vector2<f64>> =
            (0..2).map(|k| fi * nalgebra::Vector2::new(dc[k][0], dc[k][1])).collect();
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
        let a = [k(-4.0) * e[0] * e[0] - k(6.0), k(0.0), k(0.0), k(-3.0) * e[0] * e[1] - k(5.0)];
        let mut x = crate::ode::linear::Expm::<Tr>::new(2);
        let slot = x.exp(&a).expect("exponentiates");
        let p = x.get(slot);
        vec![p[0], p[3]]
    };
    for eta in [[0.3, 0.2], [1.5, 1.8]] {
        let got = run2(f, [0.3, 0.2], eta);
        let parts = [
            (-4.0 * eta[0] * eta[0] - 6.0, [-8.0 * eta[0], 0.0], [[-8.0, 0.0], [0.0, 0.0]]),
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
                assert!((g - w).abs() <= 1e-13 * w.abs().max(1e-300), "{}: {g} vs {w}", what(s));
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
    let l1 = Layout { n: 2, m: 0, th: false, wide: false, tan: 0 };
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
            assert!(plan.reverse_from(&fwd, l1, 1, &mut ub, &mut [], &du_bar), "reverses");
            [ub[0], ub[l]]
        })
        .collect()
}

/// `exp` of a 2×2 matrix in η, large enough to need squarings: the Padé solve and the repeat call.
fn case_expm(e: &[Tr]) -> Vec<Tr> {
    let k = |c: f64| Tr::from_f64(c);
    let a = [k(-4.0) * e[0] * e[0] - k(6.0), k(0.5) * e[1], k(0.0), k(-3.0) * e[0] * e[1] - k(5.0)];
    let mut x = crate::ode::linear::Expm::<Tr>::new(2);
    let slot = x.exp(&a).expect("exponentiates");
    x.get(slot).to_vec()
}

/// The fixed point of a contraction whose constant term depends on η: the fixed-point call.
fn case_fixed(e: &[Tr]) -> Vec<Tr> {
    let k = |c: f64| Tr::from_f64(c);
    let c = [e[0] * e[0] + k(2.0) * e[1], e[0] * e[1] - k(0.5)];
    let p = |u: &[Tr]| -> Option<Vec<Tr>> {
        Some((0..2).map(|i| k(0.3) * (-u[i]).exp() + k(0.1) * u[1 - i] + c[i]).collect())
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
const CASES: [Case; 3] = [("expm", case_expm), ("fixed point", case_fixed), ("solve", case_solve)];

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
                assert!(plan.reverse_from(&fwd, l2(), 1, &mut ub, &mut [], &du_bar), "reverses");
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
                            close(got, want, &format!("{name} out {j} ∂³/∂{k}∂{m}∂{d} at {eta:?}"));
                        }
                    }
                }
            }
            for j in 0..y.len() {
                // ∂³/∂0∂0∂1 two ways, and ∂³/∂1∂1∂0.
                close(third_rev(j, 0, 0)[1], third_rev(j, 1, 0)[0], &format!("{name} out {j} 001"));
                close(third_rev(j, 1, 1)[0], third_rev(j, 1, 0)[1], &format!("{name} out {j} 110"));
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
        let mut y: Vec<Tr> =
            (0..2).map(|i| k(0.3) * (-u[i]).exp() + k(0.1) * u[1 - i] + c[i]).collect();
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
                close(got[j][c], want[j][c], &format!("out {j} lane {c} at {eta:?}"));
            }
        }
        assert_eq!(got[2], [0.0; 6], "the held coordinate at {eta:?}");
        let rw = rev_grads(case_fixed, [0.3, 0.2], eta);
        let rg = rev_grads(case_fixed_held, [0.3, 0.2], eta);
        for j in 0..2 {
            for k in 0..2 {
                close(rg[j][k], rw[j][k], &format!("reverse out {j} ∂{k} at {eta:?}"));
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
    let l1 = Layout { n: N as u16, m: 0, th: false, wide: false, tan: 0 };
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
        close(du[l + 1 + j], 0.01 * p / (1.0 + 0.01 * eta[j]), &format!("d prod/d eta {j}"));
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
    assert!(v.iter().any(|p| Arc::ptr_eq(p, &first)), "the touched one stays");
    assert!(!v.iter().any(|p| Arc::ptr_eq(p, &second)), "the oldest other one goes");
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
        assert_ne!(k.scope, 0, "a registered fit keys its plans in its own scope");
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
    assert_ne!(key, SubjectKey::new(&model, &subjects[0], 1, 1), "variant keys apart");
    add(&key, square_plan());
    reset();
    assert!(matches!(lookup(&key), Lookup::Found(_)), "a reset leaves a registered fit's plans");
    drop(scope);
    assert!(matches!(lookup(&key), Lookup::Missing), "the scope's end drops its plans");
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
        let (hard, soft) =
            Tr::ode_segment(&rhs, &u0, (0.0, 3.0), &p, &[3.0], &[0.7, 1.3, 2.9], &opts, &mut auto)
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
    let l1 = Layout { n: 2, m: 0, th: false, wide: false, tan: 0 };
    let l = l1.lanes();
    for method in [OdeMethod::Rk45, OdeMethod::Vern7, OdeMethod::Rosenbrock23, OdeMethod::Rodas5P] {
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
        let tol = if matches!(method, OdeMethod::Rk45 | OdeMethod::Vern7) { 1e-13 } else { 5e-10 };
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
                assert!(rel < tol, "{method:?} out {j} ∂{k}: {got} vs {want} (rel {rel:e})");
                worst = worst.max(rel);
            }
        }
        assert!(worst.is_finite(), "{method:?}: the gaps are finite");
    }
}

// ── The tiers of one chunk, bit for bit ────────────────────────────────────────────────

/// `exp(−(a0/a2)·a3)` over four inputs at the first-order layout over 3 η, `a3` carrying no
/// tangent and `a0`, `a2` none along η₂: output lane 3 is `exp(v)·(−1·t)` with `t` a structural
/// zero — the folded tiers write the constant `+0` where the unfolded program gives `−0` (or, past `exp`'s
/// overflow, `NaN`).
fn zero_tangent_chunk(live: bool) -> (ChunkProgram, Layout) {
    let code = vec![
        Flat::LoadState { d: 0, i: 0 },
        Flat::LoadState { d: 1, i: 1 },
        Flat::LoadState { d: 2, i: 2 },
        Flat::LoadState { d: 3, i: 3 },
        Flat::DivRaw { d: 4, a: 0, b: 2 },
        Flat::Mul { d: 5, a: 4, b: 3 },
        Flat::MulK { d: 6, a: 5, c: -1.0 },
        Flat::Unary { d: 7, op: Unary::Exp, a: 6 },
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
    let layout = Layout { n: 3, m: 0, th: false, wide: false, tan: 0 };
    let cp = ChunkProgram { code, n_regs: 8, spec, n_in: 4, compiled: Mutex::new(HashMap::new()) };
    (cp, layout)
}

/// The inputs' lanes: values `v`, tangents of `a0`/`a2` along η₀, η₁ only, `a1` along all.
fn zero_tangent_inputs(v: [f64; 4], l: usize) -> Vec<f64> {
    let t = [[0.0, 0.7, -0.4, 0.0], [0.0, 0.3, 0.9, -0.5], [0.0, -0.6, 0.2, 0.0], [0.0; 4]];
    let mut base = vec![0.0; 5 * l];
    for i in 0..4 {
        base[i * l] = v[i];
        base[i * l + 1..(i + 1) * l].copy_from_slice(&t[i][1..l]);
    }
    base
}

/// Every tier of the forward gives the same bits, `NaN` and signed zeros included, across the
/// signs the zero lane can take and where `exp` overflows: `v = [-800, …, 1, 1]` makes the
/// output `exp(800)·(−1·t)` with `t` a structural zero — `inf · 0`, a `NaN`, wherever the
/// zero is computed, and a constant `0` where the folding knows it. All tiers (the interpreter,
/// the quick and the optimizing native) run the one folded function, so none can differ from
/// another. Kills: a tier that runs the program unfolded (the cold tier used to), or one that
/// patches its outputs (a `+0` canonicalisation on one side only).
#[test]
fn the_forward_tiers_agree_bit_for_bit_on_a_structurally_zero_tangent() {
    for v in [
        [1.3, 0.4, 2.0, 0.9],
        [-1.3, 0.4, -2.0, -0.9],
        [0.0, 1.0, 3.0, 2.0],
        // exp(800) = inf.
        [-800.0, 0.4, 1.0, 1.0],
    ]
    .into_iter()
    .flat_map(|v| [(v, false), (v, true)])
    {
        let (v, live) = v;
        let (cp, layout) = zero_tangent_chunk(live);
        let l = layout.lanes();
        let t = cp.get(layout, Sweep::Forward).expect("a kernel");
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
        for (quick, opt) in [(true, "none"), (true, "speed"), (false, "speed")] {
            t.promote(quick, opt);
            let mut native = zero_tangent_inputs(v, l);
            unsafe {
                t.call(native.as_mut_ptr(), std::ptr::null_mut(), idx.as_ptr(), lits.as_ptr())
            };
            for c in 0..l {
                assert_eq!(
                    interp[4 * l + c].to_bits(),
                    native[4 * l + c].to_bits(),
                    "{v:?} live={live} quick={quick} {opt} lane {c}: interpreter {:e}, native {:e}",
                    interp[4 * l + c],
                    native[4 * l + c]
                );
            }
        }
        // The overflowing value is really there (so the case reaches the `inf · 0` it is for):
        // the live zero lane is a `NaN`, the folded one the constant `0`.
        if v[0] == -800.0 {
            assert!(interp[4 * l].is_infinite(), "exp(800) is inf");
            assert_eq!(
                interp[4 * l + 3].is_nan(),
                live,
                "live={live}: the lane the folding knows to be zero is a constant, not inf·0"
            );
        }
    }
}

/// The multi-seed reverse of the same chunk: native and interpreter agree bit for bit on
/// every input's adjoint lanes, from several seeds at once (`call_seeds`).
#[test]
fn the_reverse_tiers_agree_bit_for_bit_across_seeds() {
    let (cp, layout) = zero_tangent_chunk(false);
    let l = layout.lanes();
    let t = cp.get(layout, Sweep::Reverse).expect("a kernel");
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
        t.interpret(b0.as_mut_ptr(), a0.as_mut_ptr(), idx.as_ptr(), lits.as_ptr(), seeds, stride)
    };
    t.promote(true, "none");
    unsafe {
        t.call_seeds(b1.as_mut_ptr(), a1.as_mut_ptr(), stride, seeds, idx.as_ptr(), lits.as_ptr())
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
        Flat::Unary { d: 5, op: Unary::Sqrt, a: 4 },
        Flat::Sub { d: 6, a: 1, b: 0 },
        Flat::DivRaw { d: 7, a: 5, b: 6 },
        Flat::Unary { d: 8, op: Unary::Abs, a: 2 },
        Flat::Mul { d: 9, a: 7, b: 8 },
        Flat::Deriv { state: 0, a: 9 },
        Flat::Deriv { state: 1, a: 7 },
    ];
    let layout = Layout { n: 3, m: 6, th: false, wide: false, tan: 0 };
    let cp = ChunkProgram {
        code,
        n_regs: 10,
        spec: vec![(DirMask::ALL, u16::MAX); 3],
        n_in: 3,
        compiled: Mutex::new(HashMap::new()),
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
/// or read from the wrong operand, a reverse pack scheduled across the adjoint store it reads back.
#[test]
fn the_vector_tiers_agree_bit_for_bit_with_the_scalar_ones() {
    let (cp, layout) = simd_chunk();
    let l = layout.lanes();
    let idx: Vec<u32> = (0..5).collect();
    let lits = [0.0f64];
    // The forward.
    let t = cp.get(layout, Sweep::Forward).expect("a kernel");
    let p = simd::packs(&t.ir().unwrap(), l, false, 0).expect("the forward packs");
    let vec_ops = p
        .members
        .iter()
        .filter(|m| {
            !matches!(t.ir().unwrap().insts[m[0] as usize], Inst::Load { .. } | Inst::Store { .. })
        })
        .count();
    assert!(vec_ops >= 20, "{vec_ops} vector operations");
    let mut interp = simd_inputs(l);
    unsafe {
        t.interpret(interp.as_mut_ptr(), std::ptr::null_mut(), idx.as_ptr(), lits.as_ptr(), 1, 0)
    };
    t.promote(false, "speed");
    let mut native = simd_inputs(l);
    unsafe { t.call(native.as_mut_ptr(), std::ptr::null_mut(), idx.as_ptr(), lits.as_ptr()) };
    assert!(interp[3 * l..].iter().all(|x| x.is_finite()));
    for c in 3 * l..5 * l {
        assert_eq!(interp[c].to_bits(), native[c].to_bits(), "forward lane {c}");
    }
    // The reverse, several seeds at once.
    let t = cp.get(layout, Sweep::Reverse).expect("a kernel");
    assert!(simd::packs(&t.ir().unwrap(), l, true, t.prefix).is_some(), "the reverse packs");
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
        t.interpret(b0.as_mut_ptr(), a0.as_mut_ptr(), idx.as_ptr(), lits.as_ptr(), seeds, stride)
    };
    t.promote(false, "speed");
    unsafe {
        t.call_seeds(b1.as_mut_ptr(), a1.as_mut_ptr(), stride, seeds, idx.as_ptr(), lits.as_ptr())
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
    assert_ne!(subject_desc(&a), subject_desc(&swapped), "a value belongs to its name");
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
            assert_ne!(pa, pb, "studies {} and {} share a fingerprint", a + 1, b + 1);
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
            assert_ne!(descs[i], descs[j], "levels {} and {} share a plan key", i + 1, j + 1);
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
    assert!(matches!(lookup(&k1), Lookup::Missing), "a dropped model's plans are gone");
    // Build several, some usually at the freed address; the identity differs either way, so the
    // key must.
    let mut keep = Vec::new();
    for _ in 0..8 {
        let m = build(2.0);
        let same = &m as *const _ as usize == addr1;
        let k = SubjectKey::new(&m, &subject, 1, 2).with_variant(Desc::of_value(&0x1d_e171u64));
        assert_ne!(m.plan_id.get(), id1);
        assert_ne!(k, k1, "a model at a freed address keys apart (same address: {same})");
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
    assert!(matches!(lookup(&kb), Lookup::Missing), "b is not served a's plan");
    assert!(
        matches!(lookup(&key(&a.clone())), Lookup::Found(_)),
        "an equal subject at another address is"
    );
    assert_ne!(ka.clone().with_observed(&a), ka.clone().with_observed(&a_dv));
    let v = |x: u8| ka.clone().with_variant(Desc::of_value(&x));
    assert_ne!(v(1), v(2));
    assert_eq!(v(1), v(1));
    COLLIDE.with(|c| c.set(false));
}

/// `sqrt` at a non-positive argument is flat in a compiled plan, as on every `PkNum` type
/// (#2892): a zero gradient and Hessian, not the `0.5 / 0 = inf` (`NaN` once multiplied by a
/// zero tangent) the unguarded rule puts in the jets. The `0.25` control straddles the gate
/// (live derivative `0.5/√x = 1`, curvature `−0.25/√x³ = −2`).
#[test]
fn sqrt_at_zero_is_flat_in_a_compiled_plan() {
    for x in [0.0, -0.0, -1e-18] {
        let (_, g, h) = derivs_of(&[x], 1, true, &|t| Some(vec![t[0].sqrt()])).unwrap();
        assert!(g[0][0] == 0.0 && h[0][0] == 0.0, "sqrt({x:e}): {g:?} {h:?}");
    }
    let (v, g, h) = derivs_of(&[0.25], 1, true, &|t| Some(vec![t[0].sqrt()])).unwrap();
    assert_eq!(v[0], 0.5);
    assert!((g[0][0] - 1.0).abs() < 1e-12 && (h[0][0] + 2.0).abs() < 1e-12);
}

/// A plan resolves each `(layout, direction)` once, in a slot read without a lock: the same
/// key gives the same kernels (one slice), and a key past the slots still serves, resolved per
/// call (so the tail of a plan's layouts never goes missing).
#[test]
fn resolved_kernels_are_shared_per_key_and_serve_past_the_slots() {
    begin();
    let et: Vec<Tr> = (0..2).map(|k| Tr::eta(k, 0.3)).collect();
    let y = vec![et[0] * et[1] + et[0]];
    let rec = end(Some(y.iter().map(|v| v.reg()).collect())).expect("records");
    let base = Layout { n: 2, m: 0, th: false, wide: false, tan: 0 };
    let plan = rec.plan(base, 2, 0).expect("plans");
    let mut shared = 0;
    for tan in 0..(RESOLVED_SLOTS as u8 + 4) {
        let lay = Layout { tan, ..base };
        let a = plan.fns(lay, Sweep::Forward).expect("resolves");
        let b = plan.fns(lay, Sweep::Forward).expect("resolves");
        assert_eq!(a.len(), b.len());
        if matches!(a, Fns::Shared(_)) {
            assert!(std::ptr::eq(a.as_ptr(), b.as_ptr()), "tan {tan}: one resolution");
            shared += 1;
        } else {
            assert!(matches!(b, Fns::Own(_)), "tan {tan}: past the slots, per call");
        }
    }
    assert_eq!(shared, RESOLVED_SLOTS, "the slots fill, in order, then stop");
}

// ---- the linear kernels' plan values are the native f64 path's bits (#2786) ---------------

fn k_expm<T: PkNum>(e: &[T], aug: bool, scale: f64) -> Vec<T> {
    let k = |c: f64| T::from_f64(c);
    let a = if aug {
        // The augmented generator of a one-state segment: [[-CL/V·Δ, b/c·Δ], [0, 0]].
        [-(k(0.31) * e[0].exp()) * k(scale), k(0.7) * e[1].exp() * k(scale), k(0.0), k(0.0)]
    } else {
        [
            -(k(0.31) * e[0].exp()) * k(scale),
            k(0.05) * e[1].exp() * k(scale),
            k(0.9) * e[0].exp() * k(scale),
            -(k(0.4) * e[1].exp()) * k(scale),
        ]
    };
    let mut x = crate::ode::linear::Expm::<T>::new(2);
    let slot = x.exp(&a).expect("exponentiates");
    x.get(slot).to_vec()
}

fn k_solve<T: PkNum>(e: &[T]) -> Vec<T> {
    let k = |c: f64| T::from_f64(c);
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
}

#[test]
fn the_linear_kernels_plan_values_are_the_native_bits() {
    type F = Box<dyn Fn(&[Tr]) -> Vec<Tr>>;
    type G = Box<dyn Fn(&[f64]) -> Vec<f64>>;
    let mut cases: Vec<(String, F, G)> = Vec::new();
    for aug in [true, false] {
        for scale in [0.5, 3.0, 24.0, 200.0] {
            cases.push((
                format!("expm aug={aug} scale={scale}"),
                Box::new(move |e| k_expm::<Tr>(e, aug, scale)),
                Box::new(move |e| k_expm::<f64>(e, aug, scale)),
            ));
        }
    }
    cases.push(("solve".into(), Box::new(k_solve::<Tr>), Box::new(k_solve::<f64>)));
    let mut bad = Vec::new();
    for (name, f, g) in &cases {
        for eta in [[0.0, 0.0], [0.1, -0.05], [-0.2, 0.15], [0.3, 0.2]] {
            let got = run2(f, [0.3, 0.2], eta);
            let want = g(&eta);
            for (j, (a, b)) in got.iter().zip(&want).enumerate() {
                if a[0].to_bits() != b.to_bits() {
                    bad.push(format!("{name} at {eta:?} out {j}: plan {} native {b}", a[0]));
                }
            }
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// Every field of a dose a recording can read keys its subject's plan (#2786 review #4):
/// `implied` (with each of its three fields), `evid4` and `off_at` each move the descriptor,
/// and a dose list equal in all of them does not. Dies when any one leaves `subject_desc`.
#[test]
fn every_dose_field_a_recording_reads_keys_the_plan() {
    use crate::types::{DoseEvent, ImpliedDose, Subject};
    let base = || {
        let mut s = Subject::default();
        s.doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
        ];
        s
    };
    let imp = |parent: usize, param_time: f64, gov_time: f64| {
        Some(ImpliedDose { parent, param_time, gov_time })
    };
    let with = |f: &dyn Fn(&mut DoseEvent)| {
        let mut s = base();
        f(&mut s.doses[1]);
        subject_desc(&s)
    };
    let plain = subject_desc(&base());
    assert_eq!(plain, with(&|_| {}), "an unchanged list keys equal");
    let implied = with(&|d| d.implied = imp(0, 24.0, 24.0));
    let variants = [
        ("implied", implied.clone()),
        ("parent", with(&|d| d.implied = imp(1, 24.0, 24.0))),
        ("param_time", with(&|d| d.implied = imp(0, 25.0, 24.0))),
        ("gov_time", with(&|d| d.implied = imp(0, 24.0, 25.0))),
        ("evid4", with(&|d| d.evid4 = true)),
        ("off_at", with(&|d| d.off_at = 30.0)),
    ];
    for (i, (name, d)) in variants.iter().enumerate() {
        assert_ne!(*d, plain, "{name} keys apart from the plain dose");
        for (other, e) in &variants[i + 1..] {
            assert_ne!(d, e, "{name} and {other} key apart");
        }
    }
}

/// A kept forward is adopted only by the plan that made it, identified by a process-unique
/// serial and not by its address (#2786 review #7): a freed plan's address is reused, and an
/// unrelated plan whose inputs match (eta = 0 at the initial theta) would adopt the stale forward.
/// Two live plans of one shape stand for the ABA pair: equal layout, buffer length and inputs,
/// differing in identity alone, so only the identity check can refuse. Dies when the check
/// goes back to an address, or to none.
#[test]
fn a_kept_forward_is_adopted_only_by_the_plan_that_made_it() {
    let (a, b) = (square_plan(), square_plan());
    assert!(a.chunky && b.chunky && a.layout == b.layout);
    assert_ne!(a.serial, b.serial, "serials are unique");
    let owner = 0x5eed_0007usize;
    let keep = |p: &Plan| {
        kept_lock().insert(
            owner,
            Kept {
                plan: p.serial,
                layout: p.layout,
                buf: vec![0.0; p.n_slots * p.lanes],
                scope: 0,
                model: 0,
            },
        );
    };
    let eta = [0.0, 0.0];
    keep(&a);
    assert!(b.adopt(owner, b.layout, &[], &eta).is_none(), "another plan's forward is refused");
    assert!(
        a.adopt(owner, a.layout, &[], &eta).is_some(),
        "its own is adopted (the refusal above was the identity, not the inputs)"
    );
    assert!(a.adopt(owner, a.layout, &[], &eta).is_none(), "and taken once");
}

/// A dropped model leaves nothing behind in the process-wide caches (#2786 review): the
/// templates it planned go, the chunk programs only its plans ran are freed (the chunk cache
/// holds them weakly), and a program's native code is freed with its last user. Dies when
/// `drop_model` leaves its templates in `templates()`, when the chunk cache owns its programs,
/// or when a compiled kernel's module is never freed.
#[test]
fn the_caches_shrink_when_a_model_drops() {
    let subject = crate::types::Subject::default();
    let m = crate::types::test_helpers::analytical_model(crate::types::GradientMethod::default());
    let id = m.plan_id.get();
    // A structure no other test records, so the counts below are this model's alone.
    let family =
        SubjectKey::new(&m, &subject, 1, 2).with_variant(Desc::of_value(&0xcac4_e5u64)).family();
    let rec = recorded(&family, || {
        let et: Vec<Tr> = (0..2).map(|k| Tr::eta(k, 1.0)).collect();
        let mut x = et[0];
        for _ in 0..23 {
            x = x * et[1] + et[0];
        }
        Some(vec![x.reg()])
    })
    .expect("records");
    let plan = rec.plan(l2(), 2, 0).expect("plans");
    assert_eq!(templates_of(id), 1, "the model's template is cached");
    let prog = plan
        .calls
        .iter()
        .find_map(|c| match c {
            Call::Chunk { prog, .. } => Some(prog.clone()),
            _ => None,
        })
        .expect("a chunk call");
    let mine = Arc::downgrade(&prog);
    let held_by_cache = |w: &std::sync::Weak<ChunkProgram>| {
        chunk_cache().lock().unwrap().values().flatten().any(|o| o.ptr_eq(w))
    };
    assert!(held_by_cache(&mine), "the program is shared by structure");
    // Compile it, so there is native code to free.
    let k = prog.get(l2(), Sweep::Forward).expect("a kernel");
    k.promote(true, "none");
    assert_ne!(k.native.load(std::sync::atomic::Ordering::Acquire), 0);
    drop((k, prog, plan));
    let freed0 = KERNELS_FREED.with(std::cell::Cell::get);
    // The template still holds the plan, so the program lives until the model goes.
    assert!(mine.upgrade().is_some(), "the template's plan runs it");
    drop(m);
    assert_eq!(templates_of(id), 0, "the dropped model's templates go");
    assert!(mine.upgrade().is_none(), "the program is freed");
    assert!(!held_by_cache(&mine), "no dead entry stays in the cache");
    assert!(KERNELS_FREED.with(std::cell::Cell::get) > freed0, "its native code is freed with it");
}

/// A bad mark is the subject's for every thread and every later caller, not for whichever
/// thread looked it up last (#2786 review): a thread's last-subject entry is dropped when the
/// mark is written, and a recording that finishes afterwards does not replace the mark.
/// Mutation: drop the `GENERATION` bump and the `LAST` reset from `mark_bad` -> the lookup
/// right after it still finds the plan; let `add` replace a `None` -> the lookup after the
/// second `add` finds a plan again, on this thread and on another.
#[test]
fn a_bad_mark_outlives_the_cached_plan_and_a_late_add() {
    let subject = crate::types::Subject::default();
    let m = crate::types::test_helpers::analytical_model(crate::types::GradientMethod::default());
    let key = SubjectKey::new(&m, &subject, 1, 2).with_variant(Desc::of_value(&0xbad_0001u64));
    add(&key, square_plan());
    assert!(matches!(lookup(&key), Lookup::Found(_)), "fixture: cached");
    mark_bad(&key, "forced");
    assert!(matches!(lookup(&key), Lookup::Bad), "the thread that cached the plan sees the mark");
    // A recording that was in flight when the mark landed serves its caller, and stores nothing.
    let served = add(&key, square_plan());
    assert_eq!(served.len(), 1, "the caller still has a plan to serve from");
    assert!(matches!(lookup(&key), Lookup::Bad), "a late add leaves the mark");
    std::thread::scope(|s| {
        s.spawn(|| assert!(matches!(lookup(&key), Lookup::Bad), "another thread"))
            .join()
            .expect("lookup");
    });
}

/// A finished fit's templates, families and kept forwards leave with its scope; the model's
/// scope-`0` state is not what they are keyed under (#2786 review): many fits of one model leave
/// the caches as they found them, and another fit's kept forward stays.
/// Mutation: key a family under scope `0` again (`SubjectKey::family`) -> the template's owner is
/// `(0, id)` and `templates_of` stays 1 after every fit; stop `drop_scope` clearing the fit's
/// kept forwards, or let `drop_model` clear them all -> the `kept_in` counts below.
#[test]
fn a_finished_fits_templates_and_kept_forwards_leave_with_its_scope() {
    let m = crate::types::test_helpers::analytical_model(crate::types::GradientMethod::default());
    let id = m.plan_id.get();
    let (a, b) = (vec![crate::types::Subject::default()], vec![crate::types::Subject::default()]);
    let guard_b = register(&m, &b);
    let scope_of =
        |s: &[crate::types::Subject]| registered(s.as_ptr() as usize).expect("registered").1;
    let scope_b = scope_of(&b);
    let keep = |owner: usize, scope: u64| {
        kept_lock().insert(
            owner,
            Kept { plan: 0, layout: l2(), buf: vec![0.0; 4], scope, model: m.plan_id.get() },
        );
    };
    keep(b.as_ptr() as usize, scope_b);
    for round in 0..4 {
        // On a thread that runs no fit: a registration inside a fit of the model joins it.
        let guard = super::in_fit(0, || register(&m, &a));
        let scope_a = scope_of(&a);
        assert_ne!(scope_a, scope_b, "each fit has a scope of its own");
        // A structure no other test records, so the counts are this model's alone.
        let family =
            SubjectKey::new(&m, &a[0], 1, 2).with_variant(Desc::of_value(&0xf17_0001u64)).family();
        assert_eq!(family.scope, scope_a, "the family belongs to the fit");
        let rec = recorded(&family, || {
            let et: Vec<Tr> = (0..2).map(|k| Tr::eta(k, 1.0)).collect();
            let mut x = et[0];
            for _ in 0..(11 + round) {
                x = x * et[1] + et[0];
            }
            Some(vec![x.reg()])
        })
        .expect("records");
        rec.plan(l2(), 2, 0).expect("plans");
        keep(a.as_ptr() as usize, scope_a);
        assert_eq!(templates_of(id), 1, "round {round}: the fit's template");
        assert_eq!(kept_in(scope_a), 1);
        drop(guard);
        assert_eq!(templates_of(id), 0, "round {round}: it left with the fit");
        assert_eq!(kept_in(scope_a), 0, "round {round}: so did its forward");
        assert!(lock_families().keys().all(|k| k.model != id), "round {round}: and its family");
        // Another model dropping is not this fit's business.
        drop_model(u64::MAX);
        assert_eq!(kept_in(scope_b), 1, "round {round}: B keeps its own");
    }
    drop(guard_b);
    assert_eq!(kept_in(scope_b), 0, "B's leaves with B");
}

/// `drop_model` frees the dropped model's unregistered kept forwards and no other model's
/// (#2786 round 2, finding 10). Mutation: `clear_kept(|k| k.scope == 0)` in `drop_model` (the
/// old `clear_kept(0)`) -> the second model's forward is gone.
#[test]
fn drop_model_clears_only_its_own_kept_forwards() {
    let (ma, mb) = (0x7a_0001u64, 0x7a_0002u64);
    let (oa, ob) = (0x7a_1001usize, 0x7a_1002usize);
    let put = |o: usize, model: u64| {
        kept_lock().insert(o, Kept { plan: 0, layout: l2(), buf: vec![0.0; 4], scope: 0, model });
    };
    put(oa, ma);
    put(ob, mb);
    drop_model(ma);
    let t = kept_lock();
    assert!(!t.contains_key(&oa), "the dropped model's forward is gone");
    assert!(t.contains_key(&ob), "another model's forward stays");
    drop(t);
    drop_model(mb);
    assert!(!kept_lock().contains_key(&ob));
}

/// The kept table's lock is taken from `Drop` (`drop_scope`, `Prints::drop`): one poisoned by a
/// panic elsewhere must not turn a drop into a second panic (#2786 round 2, finding 9).
/// Mutation: `.lock().unwrap()` in `kept_lock` -> the drop panics.
#[test]
fn a_poisoned_kept_table_does_not_panic_a_scope_drop() {
    let h = std::thread::spawn(|| {
        let _g = kept_table().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        panic!("poison the kept table");
    });
    assert!(h.join().is_err());
    assert!(kept_table().is_poisoned());
    drop_scope(0x7a_ffff);
    drop_model(0x7a_fffe);
    assert_eq!(kept_in(0x7a_ffff), 0);
    kept_table().clear_poison();
}

/// Each scope settles its own way: a point's failure is counted (the call took finite
/// differences) and not condemned, so a later call is re-served; a repelled point is neither
/// counted nor condemned; the model's failure marks the model (#2786 round 2, finding 1).
/// Mutations: drop `note_failure` from the `Point` arm -> not counted; make `Point` mark bad ->
/// not re-served; count `Repelled` -> the count is 1 after it.
#[test]
fn settling_a_failure_counts_a_point_declines_a_repelled_point_and_marks_a_model() {
    let new_model =
        || crate::types::test_helpers::analytical_model(crate::types::GradientMethod::default());
    let subs = vec![crate::types::Subject::default()];
    let key = |m: &crate::types::CompiledModel, tag: u64| {
        SubjectKey::new(m, &subs[0], 0, 0).with_variant(Desc::of_value(&tag))
    };
    // A repelled point: nothing noted, nothing marked.
    let m = new_model();
    let _g = register(&m, &subs);
    let k = key(&m, 0x5e_0010);
    settle_failure(&k, "overflowed", FailScope::Repelled);
    assert!(matches!(lookup(&k), Lookup::Missing));
    assert!(plan_failure_counts(&m, &subs).is_none());
    // A point's failure: counted, not condemned, and the next call is served.
    settle_failure(&k, "no steady state", FailScope::Point);
    assert!(matches!(lookup(&k), Lookup::Missing), "not condemned");
    let c = plan_failure_counts(&m, &subs).expect("counted");
    assert_eq!((c.any, c.inner, c.outer), (1, 1, 1));
    assert_eq!(c.why, "no steady state");
    add(&k, square_plan());
    assert!(matches!(lookup(&k), Lookup::Found(_)), "re-served");
    // The model's failure marks the model, and every subject's key with it.
    let m2 = new_model();
    let _g2 = register(&m2, &subs);
    let k2 = key(&m2, 0x5e_0011);
    settle_failure(&k2, "unsupported", FailScope::Model);
    assert!(matches!(lookup(&k2), Lookup::Bad));
    assert!(model_bad(&key(&m2, 0x5e_0012)).is_some(), "the model is bad");
    assert!(model_bad(&k).is_none(), "another model is not");
}

/// A failure at a point whose recorded values overflowed — the individual parameters, an
/// intermediate, a `NaN` guard — is the point's alone and is repelled, whatever it names: a
/// predictor that bailed ("the recorder returned None", the subject's scope on a sane point) and
/// a record-time run that failed (`Point`) alike; and a recording whose values were not all
/// finite says so (`finite()`), so a plan is not condemned for failing to serve at its own
/// point there (#2786 round 2, finding 2). The control side of each gate is asserted: on finite
/// values the scopes stay. Mutation: drop the `nonfinite` flag from `Rec::store` -> every
/// overflow row reads its finite-side scope.
#[test]
fn a_failure_at_an_overflowed_point_is_repelled_and_a_finite_one_keeps_its_scope() {
    let overflow = || Tr::eta(0, 800.0).exp();
    let fine = || Tr::eta(0, 1.0).exp();
    // A bailing predictor.
    begin();
    let x = overflow();
    assert!(!x.recorded().is_finite());
    assert_eq!(end(None).err().expect("fails").1, FailScope::Repelled);
    begin();
    let _ = fine();
    assert_eq!(end(None).err().expect("fails").1, FailScope::Subject);
    // A record-time run that failed.
    begin();
    let _ = overflow();
    fail_here("run");
    assert_eq!(end(Some(vec![])).err().expect("fails").1, FailScope::Repelled);
    begin();
    let _ = fine();
    fail_here("run");
    assert_eq!(end(Some(vec![])).err().expect("fails").1, FailScope::Point);
    // The model's own failure is not the point's, overflowed or not.
    begin();
    let _ = overflow();
    fail("unsupported".into());
    assert_eq!(end(Some(vec![])).err().expect("fails").1, FailScope::Model);
    // A recording that succeeds says whether its values were all finite: the guard NaN of
    // `inf - inf` is one even where every output is finite.
    begin();
    let (big, one) = (overflow(), fine());
    let nan = big - big;
    let rec = end(Some(vec![one.reg(), nan.reg()])).expect("records");
    assert!(!rec.finite());
    begin();
    let one = fine();
    let rec = end(Some(vec![one.reg()])).expect("records");
    assert!(rec.finite());
}

/// A subject whose structure is not its family's template's is recorded afresh, never failed: the
/// replay that meets another structure ("another structure than its template") is discarded by
/// [`recorded`], which records again without the template. The control side asserts the replay
/// engages for the template's own structure (`REPLAYS`), so the mismatch below is a replay that
/// ended, not one that never started. Mutation: return the replay's `Err` from `recorded`
/// instead of recording afresh -> `expect("records afresh")` fails.
#[test]
fn a_structure_other_than_the_template_is_recorded_afresh() {
    let subject = crate::types::Subject::default();
    let m = crate::types::test_helpers::analytical_model(crate::types::GradientMethod::default());
    let family =
        SubjectKey::new(&m, &subject, 1, 2).with_variant(Desc::of_value(&0x7e_4411u64)).family();
    let chain = |n: usize| {
        move || {
            let et: Vec<Tr> = (0..2).map(|k| Tr::eta(k, 1.0)).collect();
            let mut x = et[0];
            for _ in 0..n {
                x = x * et[1] + et[0];
            }
            Some(vec![x.reg()])
        }
    };
    let replays = || REPLAYS.with(std::cell::Cell::get);
    let first = recorded(&family, chain(23)).expect("records");
    first.plan(l2(), 2, 0).expect("plans");
    // The template's own structure replays.
    let before = replays();
    let same = recorded(&family, chain(23)).expect("records");
    same.plan(l2(), 2, 0).expect("plans");
    assert_eq!(replays(), before + 1, "the same structure replays");
    // Another structure: the replay ends at its first unmet decision, and the subject records
    // afresh.
    let before = replays();
    let other = recorded(&family, chain(5)).expect("records afresh");
    assert_eq!(replays(), before, "the other structure is not a replay");
    assert_eq!(other.outs.len(), 1);
    let plan = other.plan(l2(), 2, 0).expect("plans");
    // And it computes its own structure: `x ← x·η₁ + η₀`, five times from `η₀`, at (0.3, 0.2).
    let (e0, e1) = (0.3, 0.2);
    let mut want = e0;
    for _ in 0..5 {
        want = want * e1 + e0;
    }
    let l = l2().lanes();
    let mut u = vec![0.0; 2 * l];
    for (k, e) in [e0, e1].into_iter().enumerate() {
        u[k * l] = e;
        u[k * l + 1 + k] = 1.0;
    }
    let mut du = vec![0.0; 2 * l];
    plan.forward_layout(l2(), &u, &[], &mut du).expect("runs");
    close(du[0], want, "the afresh recording's value");
}

/// A family whose structure changes at every point (an event time that moves with η: a modeled
/// infusion duration crossing the observations) keeps no template once it is unstable. Its first
/// structure leaves one; at the `UNSTABLE_AFTER`th replay that meets another structure the
/// templates it kept are released — from its replay list and from the template map, whose key is
/// the structure and so bounds nothing — and twenty structures leave none, each of which would
/// otherwise be a recording kept for the rest of the fit. Every plan still computes its own
/// structure. Mutations: drop the `misses >= UNSTABLE_AFTER` early return in `recorded` -> the
/// replays go on and `misses` passes `UNSTABLE_AFTER`; drop the release in `missed` -> the
/// templates of the first structures stay kept.
#[test]
fn an_unstable_family_releases_its_templates() {
    let subject = crate::types::Subject::default();
    let m = crate::types::test_helpers::analytical_model(crate::types::GradientMethod::default());
    let family =
        SubjectKey::new(&m, &subject, 1, 2).with_variant(Desc::of_value(&0x7e_4412u64)).family();
    let owner = Some((family.scope, family.model));
    let kept = || {
        templates()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .values()
            .flatten()
            .filter(|t| t.owner == owner)
            .count()
    };
    let chain = |n: usize| {
        move || {
            let et: Vec<Tr> = (0..2).map(|k| Tr::eta(k, 1.0)).collect();
            let mut x = et[0];
            for _ in 0..n {
                x = x * et[1] + et[0];
            }
            Some(vec![x.reg()])
        }
    };
    let (e0, e1) = (0.3, 0.2);
    let l = l2().lanes();
    let mut u = vec![0.0; 2 * l];
    for (k, e) in [e0, e1].into_iter().enumerate() {
        u[k * l] = e;
        u[k * l + 1 + k] = 1.0;
    }
    for n in 1..=20 {
        let plan = recorded(&family, chain(n)).expect("records").plan(l2(), 2, 0).expect("plans");
        let mut want = e0;
        for _ in 0..n {
            want = want * e1 + e0;
        }
        let mut du = vec![0.0; 2 * l];
        plan.forward_layout(l2(), &u, &[], &mut du).expect("runs");
        close(du[0], want, "each structure's own value");
        if n == 1 {
            assert_eq!(kept(), 1, "a family's first structure leaves a template");
        }
    }
    let f = lock_families().get(&family).cloned().expect("the family is known");
    assert_eq!(f.misses, UNSTABLE_AFTER, "replays stop once the family is unstable");
    assert!(f.alts.is_empty(), "an unstable family replays nothing");
    assert_eq!(kept(), 0, "an unstable family's templates are released");
}

/// What a recording that declines the exact linear path leaves behind: nothing. The state of
/// the recording after `restore` is what it was at `snapshot` — instructions, registers, the
/// literal pool, hash-consing, loop calls, chunk marks, the guard and the template trace — and
/// recording the same expression again takes the same registers as the first time. Mutations:
/// leave the guard (`self.guard = s.guard`) -> the guard differs; leave the pool or the literals
/// in `cse` -> the re-recorded literal has no pool entry or reads a dead register; truncate
/// `code` but not `n_regs` -> the registers differ.
#[test]
fn restoring_a_recording_undoes_everything_recorded_since() {
    begin();
    let x = Tr::eta(0, 0.7);
    let y = x * Tr::k(2.5);
    let snap = rec(|r| r.snapshot());
    let shape = |r: &Rec| {
        (
            r.code.len(),
            r.n_regs,
            r.pool.clone(),
            r.loops.len(),
            r.marks.clone(),
            r.guard,
            r.guarded.len(),
            r.cse.len(),
            r.lits.len(),
        )
    };
    let at_snapshot = rec(|r| shape(r));
    // Something of everything: a literal, a dependent node, a guard, a chunk mark.
    let speculate = || {
        let z = (y + Tr::k(7.25)).exp();
        Tr::guard_cmp(z, ir::Cc::Gt, Tr::k(0.5));
        <Tr as crate::sens::num::RecHooks>::chunk_mark();
        z
    };
    let first = speculate();
    let after = rec(|r| shape(r));
    assert_ne!(after, at_snapshot, "the speculation recorded something");
    assert!(rec(|r| r.guard).is_some(), "and a guard");
    rec(|r| r.restore(snap));
    assert_eq!(rec(|r| shape(r)), at_snapshot, "restored");
    // The same recording again takes the same registers and the same shape.
    let second = speculate();
    assert_eq!(second.reg(), first.reg(), "the same registers");
    assert_eq!(rec(|r| shape(r)), after, "the same shape");
    end(Some(vec![second.reg()])).expect("records");
}

/// A right-hand side that is not affine in the state is a decline of the exact linear path, not a
/// failure of the recording: `affine_parts` says so (`false`), leaves no guard behind (its call on
/// the state's placeholder would have required `u > 0.5` of the variable), and the recording
/// ends `Ok`. Mutation: fail the recording there instead (`fail_subject`) -> `end` is `Err`;
/// skip the restore -> the guard on the placeholder state stays and `guard` is `Some`.
#[test]
fn a_right_hand_side_not_affine_in_the_state_declines_without_a_trace() {
    begin();
    // `f = −u²` with a branch on the state: not affine, and the branch would guard the variable.
    let rhs = |u: &[Tr], _p: &[Tr], _t: Tr, f: &mut [Tr]| {
        Tr::guard_cmp(u[0], ir::Cc::Gt, Tr::k(0.5));
        f[0] = -(u[0] * u[0]);
    };
    let (mut a, mut b) = (vec![Tr::k(0.0)], vec![Tr::k(0.0)]);
    assert!(!affine_parts(&rhs, &[], 0.0, &mut a, &mut b));
    assert!(rec(|r| r.guard).is_none(), "no guard on the variable stays");
    assert!(rec(|r| r.failed.is_none()), "the recording has not failed");
    let ok = end(Some(vec![])).expect("the recording ends Ok");
    assert!(ok.guard.is_none());
    // The control side: an affine right-hand side is served.
    begin();
    let aff = |u: &[Tr], _p: &[Tr], _t: Tr, f: &mut [Tr]| f[0] = -(Tr::k(0.3) * u[0]);
    let (mut a, mut b) = (vec![Tr::k(0.0)], vec![Tr::k(0.0)]);
    assert!(affine_parts(&aff, &[], 0.0, &mut a, &mut b));
    end(Some(vec![])).expect("records");
}

/// `y' = −p·y²` under `OdeMethod::Linear`, the exact linear path's request on a right-hand side it
/// cannot take: the recording declines the path (counted: `declined`) and records the stepper
/// instead, and the plan carries the value and the η-derivatives of the integration — against the
/// `f64` solve and its central difference. Mutation: fail the recording on the non-affine
/// right-hand side -> `recorded` is `Err`.
#[test]
fn a_non_affine_system_on_the_linear_method_is_integrated_by_the_plan() {
    fn solve<T: crate::sens::num::PkNum>(e0: T, e1: T) -> T {
        let k = |c: f64| T::from_f64(c);
        let p = [k(0.8) * e0.exp()];
        let u0 = [k(1.0) + k(0.2) * e1];
        let rhs = |u: &[T], p: &[T], _t: T, du: &mut [T]| du[0] = -(p[0] * u[0] * u[0]);
        let opts = crate::ode::solver::OdeSolverOptions {
            reltol: 1e-10,
            abstol: 1e-12,
            method: crate::ode::solver::OdeMethod::Linear,
            auto_switch: false,
            ..Default::default()
        };
        let mut auto = crate::ode::solver::OdeAutoSwitchState::default();
        crate::ode::solver::solve_ode_g_with_auto_state(
            &rhs,
            &u0,
            (0.0, 2.0),
            &p,
            &[2.0],
            &opts,
            &mut auto,
        )[0]
        .u[0]
    }
    let at = [0.3, 0.2];
    begin();
    let et: Vec<Tr> = (0..2).map(|k| Tr::eta(k, at[k])).collect();
    let y = solve(et[0], et[1]);
    let rec = end(Some(vec![y.reg()])).expect("records: the path is declined, not failed");
    assert_eq!(rec.declined, 1, "the exact linear path declined once");
    let plan = rec.plan(l2(), 2, 0).expect("plans");
    let l = l2().lanes();
    let mut u = vec![0.0; 2 * l];
    for k in 0..2 {
        u[k * l] = at[k];
        u[k * l + 1 + k] = 1.0;
    }
    let mut du = vec![0.0; 2 * l];
    plan.forward_layout(l2(), &u, &[], &mut du).expect("runs");
    let f = |a: f64, b: f64| solve::<f64>(a, b);
    assert!(
        (du[0] - f(at[0], at[1])).abs() < 1e-9 * (1.0 + du[0].abs()),
        "value {} vs {}",
        du[0],
        f(at[0], at[1])
    );
    let h = 1e-3;
    for k in 0..2 {
        let (mut p, mut q) = (at, at);
        p[k] += h;
        q[k] -= h;
        let fd = (f(p[0], p[1]) - f(q[0], q[1])) / (2.0 * h);
        assert!((du[1 + k] - fd).abs() < 2e-5 * (1.0 + fd.abs()), "∂{k}: {} vs FD {fd}", du[1 + k]);
    }
}

/// The derivative of the fourth derivative of `ln Γ` — the tangent of `Unary::Tetragamma`, which
/// the recorder had no rule for (it failed the model) — is the next of `ln Γ`'s jet, `pentagamma`.
/// Recorded through `jacobian` (the right-hand side `f = ψ″(u)`, whose `∂f/∂u` is the tangent) and
/// run at another point. Mutation: the rule back to `fail(..)` -> `end` is `Err`; the jet at the
/// wrong order (`3` for `4`) -> the value is `tetragamma`, not `pentagamma`.
#[test]
fn the_tangent_of_tetragamma_is_pentagamma() {
    use crate::stats::special::{pentagamma, tetragamma};
    let record_at = 1.7;
    begin();
    let u0 = Tr::eta(0, record_at);
    let rhs = |u: &[Tr], _p: &[Tr], _t: Tr, f: &mut [Tr]| f[0] = u[0].tetragamma();
    let (mut f, mut jac, mut ft) = (vec![Tr::k(0.0)], vec![Tr::k(0.0)], vec![Tr::k(0.0)]);
    jacobian(&rhs, &[u0], &[], Tr::k(0.0), &mut f, &mut jac, &mut ft);
    close(f[0].recorded(), tetragamma(record_at), "ψ″ at the recording point");
    close(jac[0].recorded(), pentagamma(record_at), "its tangent at the recording point");
    let rec = end(Some(vec![f[0].reg(), jac[0].reg()])).expect("records: the rule exists");
    let plan = rec.plan(l2(), 2, 0).expect("plans");
    let l = l2().lanes();
    for x in [0.9, 2.4] {
        let mut u = vec![0.0; 2 * l];
        u[0] = x;
        u[1] = 1.0;
        let mut du = vec![0.0; 3 * l];
        plan.forward_layout(l2(), &u, &[], &mut du).expect("runs");
        close(du[0], tetragamma(x), &format!("ψ″({x}) from the plan"));
        close(du[l], pentagamma(x), &format!("its tangent at {x} from the plan"));
    }
}

/// The ODE driver reads values, and a recording reaches it only through `ode_segment`; a nested
/// dual over a recording has none, so nothing may integrate one. The instantiation is a compile
/// error (`integrate_resolved_g_inner`'s `const` assertion on `RecHooks::ODE_DRIVER_OK`); this
/// pins the flag itself: the recording type and every plain type integrate, a dual of a recording
/// — at any depth — does not. Mutation: `ODE_DRIVER_OK = true` on `DualT` -> the last two
/// assertions fail.
#[test]
fn only_a_recording_with_a_segment_rule_or_a_plain_type_reaches_the_ode_driver() {
    use crate::sens::jet::DualT;
    use crate::sens::num::RecHooks;
    assert!(<Tr as RecHooks>::ODE_DRIVER_OK);
    assert!(<f64 as RecHooks>::ODE_DRIVER_OK);
    assert!(<DualT<1, f64> as RecHooks>::ODE_DRIVER_OK);
    assert!(<DualT<2, DualT<1, f64>> as RecHooks>::ODE_DRIVER_OK);
    assert!(!<DualT<1, Tr> as RecHooks>::ODE_DRIVER_OK);
    assert!(!<DualT<1, DualT<1, Tr>> as RecHooks>::ODE_DRIVER_OK);
}

/// A steady-state call whose map the recording cannot take (the map declines at the record
/// point) fails the recording as the point's: callers substitute a state for a `None` and keep
/// recording, which gave a plan with no steady state. Mutation: drop the `fail_here` in
/// `Tr::ss_fixed_point` -> `end` is `Ok`.
#[test]
fn a_steady_state_call_the_recording_cannot_take_fails_the_recording() {
    use crate::sens::num::RecHooks;
    begin();
    let declines: &dyn Fn(&[Tr]) -> Option<Vec<Tr>> = &|_| None;
    let out = <Tr as RecHooks>::ss_fixed_point(1, &[], &[0.0], &[declines], Box::new(|_| None));
    assert!(matches!(out, Some(None)), "the call is not taken");
    match end(Some(vec![])) {
        Err((_, scope)) => assert_eq!(scope, FailScope::Point),
        Ok(_) => panic!("the recording went on without the steady state"),
    }
}

/// Every `JetFn`'s native shim reaches its own jet at every order a plan reads: the shim table
/// listed indices 0..7 only, so `Exprel` (8) and `LnGamma` (9) read `NaN` in a compiled plan
/// (their recorded values were right). Each function's `k`-th derivative from a plan against
/// `JetFn::jet`. Mutation: shrink the shim table's index list back to `0..7` -> the last two
/// functions fail.
#[test]
fn every_jet_function_is_the_same_in_a_plan_as_in_its_table() {
    use crate::sens::num::JetFn;
    for f in JetFn::ALL {
        for k in 0..=4u8 {
            let at = 0.7;
            let want = f.jet(at)[k as usize];
            let got = run2(|e| vec![e[0].jet_k(f, k)], [at, at], [at, at])[0][0];
            if want.is_nan() {
                continue;
            }
            close(got, want, &format!("{f:?} order {k}"));
        }
    }
}

/// `periodic_ss_fixed_point_g`'s affinity self-check is a branch on a difference against a
/// tolerance: a plan recorded at one point serves another (the check holds at both), where an
/// exact-value guard on the solution and on each cycle's result held at the recording point
/// alone (`run2` asserts the plan's checks hold at the second point). The fixed point itself is
/// the closed form `b / (1 − e^{−k·II})`. Mutation: `br_value` reads back on `u_ss` / `within`
/// -> the checks fail at the other point (or the function returned `None` under a recording).
#[test]
fn the_ss_affinity_self_check_holds_at_another_point() {
    let ii = 12.0;
    let f = |e: &[Tr]| {
        let k = (e[0] * Tr::k(0.2) - Tr::k(1.5)).exp();
        let decay = (-(k * Tr::k(ii))).exp();
        let dose = Tr::k(5.0) * e[1].exp();
        let unforced = |u: &[Tr]| Some(vec![u[0] * decay]);
        let forced = |u: &[Tr]| Some(vec![u[0] * decay + dose]);
        let u =
            crate::dosing::periodic_ss_fixed_point_g::<Tr, _, _>(1, ii, 0.0, 0.0, unforced, forced)
                .expect("the affine map's fixed point");
        vec![u[0]]
    };
    let got = run2(f, [0.3, 0.2], [1.2, 0.9]);
    let k = (1.2f64 * 0.2 - 1.5).exp();
    let want = 5.0 * 0.9f64.exp() / (1.0 - (-k * ii).exp());
    close(got[0][0], want, "the fixed point at the second point");
}

/// Every native target a plan is compiled for probes its stack inline (#2786 review: the
/// analytic covariance R-matrix of a two-state ODE model segfaulted on Windows once its plan
/// reached the native tier). A frame past one page skips Windows' guard page without a probe;
/// Linux has no such requirement, so the crash cannot be reproduced here and this pins the
/// cause instead: the plan targets and the RHS JIT's take their flags from one
/// `jit::host_flags`, and every tier keeps the probe. Mutation: drop `enable_probestack` (or
/// build the plan target's flags from a fresh `settings::builder()` again, as before the fix)
/// -> the assertion names the tier.
#[test]
fn every_native_plan_tier_probes_its_stack() {
    use cranelift_codegen::settings::ProbestackStrategy;
    for (quick, opt) in [(true, "none"), (true, "speed"), (false, "speed")] {
        let isa = plan_isa(quick, opt).expect("a host target");
        let f = isa.flags();
        assert!(f.enable_probestack(), "tier ({quick}, {opt}): no stack probes");
        assert_eq!(
            f.probestack_strategy(),
            ProbestackStrategy::Inline,
            "tier ({quick}, {opt}): probes not inline"
        );
    }
}

/// Forward over reverse (#2980) against reverse over forward, through every call kind's
/// reverse (the repeat call's walk back, the fixed point's transposed steps, the recorded
/// solve): at the first-order layout with two tangent directions on η, one sweep seeded with
/// `ū` on the outputs' value lanes must give, on each η adjoint's tangent lane `a`, what the
/// reverse seeded with `ū` on tangent lane `a` gives on the value lane — and on the value
/// lanes, that reverse's tangent-lane-`a` adjoint (the plain gradient `ūᵀ·∂y/∂η`). Every base
/// lane is compared, the η-dual lanes included, so a sub-plan reverse that dropped the
/// adjoints' tangents (or read the value lanes for them) shows on the second-order entries.
#[test]
fn every_call_kind_sweeps_forward_over_reverse_as_a_reverse_per_direction() {
    let l1 = Layout { n: 2, m: 0, th: false, wide: false, tan: 0 };
    const K: usize = 2;
    let lt = Layout { tan: K as u8, ..l1 };
    let (lb, l) = (l1.lanes(), lt.lanes());
    for (name, f) in CASES {
        begin();
        let et: Vec<Tr> = (0..2).map(|k| Tr::eta(k, [0.3, 0.2][k])).collect();
        let y = f(&et);
        let rec = end(Some(y.iter().map(|v| v.reg()).collect())).expect("records");
        let plan = rec.plan(l1, 2, 0).expect("plans");
        let n_out = y.len();
        let u_bar: Vec<f64> = (0..n_out).map(|j| 0.7 - 0.4 * j as f64).collect();
        for eta in [[0.3, 0.2], [1.2, 0.9]] {
            // η's value and one-hot duals; direction `a` moves η_a's value.
            let mut u = vec![0.0; 2 * l];
            for k in 0..2 {
                u[k * l] = eta[k];
                u[k * l + 1 + k] = 1.0;
                u[k * l + lb + k] = 1.0;
            }
            let fwd = plan.forward_kept(lt, &u, &[]).expect("runs");
            let mut du_bar = vec![0.0; n_out * l];
            for j in 0..n_out {
                // The value lane and `∂y/∂η₀`'s, so the η-dual lanes' adjoints are live too.
                du_bar[j * l] = u_bar[j];
                du_bar[j * l + 1] = 0.5 * u_bar[j] + 0.3;
            }
            let mut fr = vec![0.0; 2 * l];
            assert!(
                plan.reverse_tangent_from(&fwd, 1, &mut fr, &mut [], &du_bar),
                "{name}: forward over reverse sweeps"
            );
            let mut du_rf = vec![0.0; K * n_out * l];
            for a in 0..K {
                for j in 0..n_out {
                    du_rf[a * n_out * l + j * l + lb + a] = u_bar[j];
                    du_rf[a * n_out * l + j * l + lb + K + a] = 0.5 * u_bar[j] + 0.3;
                }
            }
            let mut rf = vec![0.0; K * 2 * l];
            assert!(
                plan.reverse_from(&fwd, lt, K, &mut rf, &mut [], &du_rf),
                "{name}: reverse over forward sweeps"
            );
            let mut seen = 0.0f64;
            for a in 0..K {
                for k in 0..2 {
                    for x in 0..lb {
                        let what = |s: &str| format!("{name} η{k} lane {x} dir {a} {s} at {eta:?}");
                        let want = rf[a * 2 * l + k * l + x];
                        close(fr[k * l + lb + x * K + a], want, &what("tangent"));
                        close(
                            fr[k * l + x],
                            rf[a * 2 * l + k * l + lb + x * K + a],
                            &what("value"),
                        );
                        assert!(want.is_finite(), "{name}: non-finite second derivative");
                        seen = seen.max(want.abs());
                    }
                }
            }
            assert!(seen > 1e-3, "{name}: the second derivatives vanish, testing nothing");
        }
    }
}

/// `derivs_of`'s Hessians (#2980: one forward-over-reverse sweep, a seed per output) against
/// closed forms, on two outputs of three live inputs whose every second derivative is
/// distinct where it is not zero — so a swapped lane, a wrong input slot or
/// one output's seed leaking into another's row shows. `y₀ = x₀²x₁ + x₀e^{x₂}`,
/// `y₁ = e^{x₀x₂} + x₁³`.
///
/// Mutations: an input slot off by one in the read (`pb[((i + 1) % n) * l + 1 + j]`); every
/// output's seed in row 0 (both outputs' rows then coincide). (A transposed read is not one:
/// the matrix is symmetric.)
#[test]
fn derivs_hessians_match_closed_forms_on_several_live_inputs() {
    let x = [0.7f64, -1.3, 0.4];
    let f = |v: &[Tr]| -> Option<Vec<Tr>> {
        let (a, b, c) = (v[0], v[1], v[2]);
        Some(vec![a * a * b + a * c.exp(), (a * c).exp() + b * b * b])
    };
    let (vals, grad, hess) = derivs_of(&x, 2, true, &f).expect("served");
    let (a, b, c) = (x[0], x[1], x[2]);
    let (e2, e) = (c.exp(), (a * c).exp());
    let want_vals = [a * a * b + a * e2, e + b * b * b];
    let want_grad = [[2.0 * a * b + e2, a * a, a * e2], [c * e, 3.0 * b * b, a * e]];
    // Row-major `[i][j]`, `∂²y/∂x_i∂x_j`.
    let want_hess = [
        [[2.0 * b, 2.0 * a, e2], [2.0 * a, 0.0, 0.0], [e2, 0.0, a * e2]],
        [
            [c * c * e, 0.0, e * (1.0 + a * c)],
            [0.0, 6.0 * b, 0.0],
            [e * (1.0 + a * c), 0.0, a * a * e],
        ],
    ];
    for o in 0..2 {
        close(vals[o], want_vals[o], &format!("y{o}"));
        for i in 0..3 {
            close(grad[o][i], want_grad[o][i], &format!("∂y{o}/∂x{i}"));
            for j in 0..3 {
                close(hess[o][i * 3 + j], want_hess[o][i][j], &format!("∂²y{o}/∂x{i}∂x{j}"));
            }
        }
    }
}

/// A thread's hot table serves several subjects at once (the solve of one subject follows
/// another's, so one entry missed once per solve), evicts only on a slot collision, and a put
/// under a newer stamp releases the entries filled under the old one (a dropped scope's plans
/// must not stay pinned by the table). Mutation: back the table with one slot → the first
/// assertion fails; skip the stale drop in `put` → the `weak_count` one fails; compare `hk`
/// loosely in `get` → the collision row serves the wrong subject's variants.
/// (The stale drop in `get` is `the_hot_table_releases_a_dropped_scopes_plans_on_the_next_get`.)
#[test]
fn the_hot_table_serves_several_subjects_and_drops_stale_entries() {
    let key = |s: usize| HotKey {
        model: 0x1000,
        subject: 0x40_0000 + s * 0x40,
        n: 1,
        n_in: 2,
        kind: 0,
        class: 0,
    };
    let stamp = |gen: usize| HotStamp { prints: 1, gen };
    let (ka, kb) = {
        let ka = key(0);
        (ka, (1..).map(key).find(|k| k.slot() != ka.slot()).unwrap())
    };
    let (a, b): (Kernels, Kernels) = (Arc::new(Variants::default()), Arc::new(Variants::default()));
    let mut t = HotTable::default();
    t.put(ka, stamp(7), None, &a);
    t.put(kb, stamp(7), None, &b);
    let got = |t: &mut HotTable, k: &HotKey, g: usize| t.get(k, stamp(g), None);
    assert!(Arc::ptr_eq(&got(&mut t, &ka, 7).expect("a"), &a), "subject a still held");
    assert!(Arc::ptr_eq(&got(&mut t, &kb, 7).expect("b"), &b), "subject b held beside it");
    // A key that maps to a's slot evicts it, and is served as itself.
    let kc = (1..).map(|s| key(s * 7919)).find(|k| k.slot() == ka.slot() && *k != ka).unwrap();
    let c: Kernels = Arc::new(Variants::default());
    t.put(kc, stamp(7), None, &c);
    assert!(got(&mut t, &ka, 7).is_none(), "the collision evicted a");
    assert!(Arc::ptr_eq(&got(&mut t, &kc, 7).expect("c"), &c));
    // A put under a newer stamp releases what the old one filled.
    assert_eq!(Arc::weak_count(&b), 1, "fixture: the table holds b");
    t.put(kc, stamp(9), None, &c);
    assert_eq!(Arc::weak_count(&b), 0, "the stale entry was dropped, not kept");
    assert!(got(&mut t, &kb, 7).is_none() && got(&mut t, &kb, 9).is_none());
}

/// The variants of one subject (the plan kinds, the widths) do not share a slot: a lookup
/// alternates between them within one solve, and sharing one evicted the other on every call
/// (8000 of 8000 simulated variant pairs shared a slot while the variant was added after the
/// multiply). Mutation: restore the old hash (`(subject >> 4) * K + kind + n_in * 31 + n * 7`,
/// slot `(h >> 20) & 4095`) → the collision count is 256 and the second `get` misses.
#[test]
fn the_variants_of_one_subject_hold_different_hot_slots() {
    let key = |s: usize, kind: u8, n_in: usize| HotKey {
        model: 0x1000,
        subject: 0x40_0000 + s * 0x40,
        n: 3,
        n_in,
        kind,
        class: 0,
    };
    let by = |f: &dyn Fn(usize) -> (HotKey, HotKey)| {
        (0..256).filter(|&s| f(s).0.slot() == f(s).1.slot()).count()
    };
    // 256 pairs of a 1/4096 chance each: a handful at most (measured: 0).
    assert!(by(&|s| (key(s, 0, 2), key(s, 1, 2))) <= 4, "kind");
    assert!(by(&|s| (key(s, 0, 2), key(s, 0, 3))) <= 4, "n_in");
    let (a, b): (Kernels, Kernels) = (Arc::new(Variants::default()), Arc::new(Variants::default()));
    let s = (0..).find(|&s| key(s, 0, 2).slot() != key(s, 1, 2).slot()).unwrap();
    let stamp = HotStamp { prints: 1, gen: 7 };
    let mut t = HotTable::default();
    t.put(key(s, 0, 2), stamp, None, &a);
    t.put(key(s, 1, 2), stamp, None, &b);
    assert!(Arc::ptr_eq(&t.get(&key(s, 0, 2), stamp, None).expect("a"), &a));
    assert!(Arc::ptr_eq(&t.get(&key(s, 1, 2), stamp, None).expect("b"), &b));
}

/// The table does not keep a dropped scope's plans alive: it holds the variants weakly, so
/// they are freed when their owner drops them, with no further lookup on this thread, and the
/// entry then misses. Mutation: store the `Arc` (a strong clone) → `strong_count` is 2 and the
/// final `get` still serves the variants.
#[test]
fn the_hot_table_does_not_pin_released_variants() {
    let key = HotKey { model: 0x1000, subject: 0x40_0000, n: 1, n_in: 2, kind: 0, class: 0 };
    let stamp = HotStamp { prints: 1, gen: 7 };
    let a: Kernels = Arc::new(Variants::default());
    let mut t = HotTable::default();
    t.put(key, stamp, None, &a);
    assert_eq!(Arc::strong_count(&a), 1, "the table holds no strong reference");
    assert!(t.get(&key, stamp, None).is_some(), "served while owned");
    drop(a);
    assert!(t.get(&key, stamp, None).is_none(), "an entry whose variants were dropped misses");
}

/// A thread that only looks plans up (no put follows) lets go of a dropped scope's plans: a
/// `get` under a newer stamp releases every entry filled under the old one, for either half of
/// the stamp. Mutation: drop `release_stale` from `get` → the `weak_count` rows fail.
#[test]
fn the_hot_table_releases_a_dropped_scopes_plans_on_the_next_get() {
    let key = HotKey { model: 0x1000, subject: 0x40_0000, n: 1, n_in: 2, kind: 0, class: 0 };
    for (what, newer) in [
        ("generation", HotStamp { prints: 1, gen: 8 }),
        ("registry", HotStamp { prints: 2, gen: 7 }),
    ] {
        let a: Kernels = Arc::new(Variants::default());
        let mut t = HotTable::default();
        t.put(key, HotStamp { prints: 1, gen: 7 }, None, &a);
        assert_eq!(Arc::weak_count(&a), 1, "fixture: the table holds a");
        assert!(t.get(&key, newer, None).is_none(), "{what}: nothing served");
        assert_eq!(Arc::weak_count(&a), 0, "{what}: released by the get");
    }
}

/// A row filled under a layout is served only to a call whose layout has the same structure,
/// and a row filled with none only to a call with none. Mutation: drop the `(Some, Some)`
/// comparison (`same = true`) → the different-structure row serves; make the mixed arms `true`
/// → the none/Some rows serve.
#[test]
fn the_hot_table_compares_the_layout_structure() {
    use crate::stats::stacked_layout::StackedLayout;
    let subject_with = |occasions: Vec<u32>| crate::types::Subject {
        obs_times: vec![1.0; occasions.len()],
        occasions,
        ..Default::default()
    };
    let lay = |occ: Vec<u32>| StackedLayout::new(1, 1, &subject_with(occ));
    let (l_a, l_a2, l_b) = (lay(vec![1, 1, 2]), lay(vec![1, 1, 2]), lay(vec![1, 2, 2]));
    let desc = |l: &StackedLayout| Desc::of(|c| l.canon(c));
    assert_ne!(desc(&l_a).bytes(), desc(&l_b).bytes(), "fixture: layouts differ");
    let key = HotKey { model: 0x1000, subject: 0x40_0000, n: 1, n_in: 2, kind: 0, class: 0 };
    let stamp = HotStamp { prints: 1, gen: 7 };
    let ks: Kernels = Arc::new(Variants::default());
    let mut t = HotTable::default();
    t.put(key, stamp, Some(desc(&l_a)), &ks);
    assert!(t.get(&key, stamp, Some(&l_a2)).is_some(), "same structure");
    assert!(t.get(&key, stamp, Some(&l_b)).is_none(), "other structure");
    assert!(t.get(&key, stamp, None).is_none(), "layout vs none");
    t.put(key, stamp, None, &ks);
    assert!(t.get(&key, stamp, None).is_some(), "none vs none");
    assert!(t.get(&key, stamp, Some(&l_a)).is_none(), "none vs layout");
}

/// A matrix call or a repeat call whose record-time run overflows is at a point the value path
/// cannot score (its own `exp` returns `None` on the non-finite power), so the recording fails
/// as repelled, not counted as a plan failure: a diverged trial point (an extreme η) made the
/// squaring of a Padé approximant overflow and the subject reported `W_PLAN_FD_FALLBACK` for a
/// point nothing needed a derivative at (#3004). The controls on the other side of each gate
/// stay the point's: a count out of range (the repeat) and a finite product that is simply
/// recorded. Mutation: stop `MatSpec::run` noting a non-finite result to an open recording
/// -> the product row and the repeat row both read `Point`.
#[test]
fn an_overflowing_matrix_or_repeat_call_is_repelled_and_its_controls_keep_their_scope() {
    use crate::sens::num::RecHooks;
    let square = |v: &[Tr]| <Tr as RecHooks>::mat_mul(v, v, 1).flatten().expect("records");
    let repeat = |x: f64, count: f64| {
        begin();
        let (x0, n) = (Tr::eta(0, x), Tr::eta(1, count));
        let out = <Tr as RecHooks>::repeat(&[x0], n, &square);
        assert_eq!(matches!(out, Some(Some(_))), x.abs() < 1e50 && count <= 64.0);
        end(Some(vec![]))
    };
    // The product: finite records, an overflow is repelled.
    begin();
    let x = Tr::eta(0, 2.0);
    assert!(<Tr as RecHooks>::mat_mul(&[x], &[x], 1).flatten().is_some());
    assert!(end(Some(vec![])).is_ok(), "a finite product records");
    begin();
    let x = Tr::eta(0, 1e200);
    assert!(<Tr as RecHooks>::mat_mul(&[x], &[x], 1).flatten().is_none());
    assert_eq!(end(Some(vec![])).err().expect("fails").1, FailScope::Repelled);
    // The repeat: finite records, an overflowing iterate is repelled, a count out of range is
    // the point's.
    assert!(repeat(2.0, 3.0).is_ok(), "a finite repeat records");
    assert_eq!(repeat(1e100, 3.0).err().expect("fails").1, FailScope::Repelled);
    assert_eq!(repeat(2.0, 100.0).err().expect("fails").1, FailScope::Point);
    // A body whose inner solve is singular at a later iterate (`x -> (x-1)²/(x-1)`, singular at
    // `x = 1`): a real failure of the point, not an overflow, so it stays the point's. Mutation:
    // note a non-finite result unconditionally in `RepeatSpec::run` (the old reading of every
    // body `None`) -> this row reads `Repelled`.
    let solve_body = |v: &[Tr]| {
        let q = v[0] - Tr::k(1.0);
        <Tr as RecHooks>::mat_solve(&[q], &[q * q], 1).flatten().expect("records")
    };
    begin();
    let out = <Tr as RecHooks>::repeat(&[Tr::eta(0, 3.0)], Tr::eta(1, 3.0), &solve_body);
    assert!(matches!(out, Some(None)), "the third iterate is singular");
    assert_eq!(end(Some(vec![])).err().expect("fails").1, FailScope::Point);
    begin();
    let out = <Tr as RecHooks>::repeat(&[Tr::eta(0, 3.0)], Tr::eta(1, 2.0), &solve_body);
    assert!(matches!(out, Some(Some(_))), "two iterates are regular");
    assert!(end(Some(vec![])).is_ok());
}

/// A plan's kernels keep only what runs and lowers them (#3010): each compiled kernel's
/// instructions at their length and no `live`, which only the transforms that built it read,
/// and each chunk program's code at its length. A fit keeps thousands of kernels; on #3010's
/// 156-subject model this was 1.7 GB of IR at the peak, 0.8 GB after.
/// Mutation: drop `into_kernel` from `Tiered::new` (or the chunk code's `shrink_to_fit`) → the
/// kernel (or program) named in the message keeps its slack.
#[test]
fn a_plans_kernels_keep_only_what_runs_them() {
    begin();
    let et: Vec<Tr> = (0..2).map(|k| Tr::eta(k, 0.3)).collect();
    let y = (et[0] * et[1]).exp() + et[0] * et[0];
    let rec = end(Some(vec![y.reg()])).expect("records");
    let l1 = Layout { n: 2, m: 0, th: false, wide: false, tan: 0 };
    let plan = rec.plan(l1, 2, 0).expect("plans");
    let l = l1.lanes();
    let u = [0.3, 1.0, 0.0, 0.3, 0.0, 1.0];
    assert_eq!(u.len(), 2 * l);
    let fwd = plan.forward_kept(l1, &u, &[]).expect("runs");
    let mut du_bar = vec![0.0; l];
    du_bar[0] = 1.0;
    let mut ub = vec![0.0; 2 * l];
    assert!(plan.reverse_from(&fwd, l1, 1, &mut ub, &mut [], &du_bar));
    // Kernels seen: [forward, reverse].
    let mut seen = [0usize; 2];
    for c in &plan.calls {
        let Call::Chunk { prog, .. } = c else {
            continue;
        };
        assert_eq!(prog.code.capacity(), prog.code.len(), "a chunk program's code");
        for t in prog.compiled.lock().unwrap().values().flatten() {
            let what = if t.rev { "reverse" } else { "forward" };
            assert!(t.ir().unwrap().live.is_empty(), "the {what} kernel keeps `live`");
            assert_eq!(
                t.ir().unwrap().insts.capacity(),
                t.ir().unwrap().insts.len(),
                "the {what} kernel's instructions"
            );
            seen[t.rev as usize] += 1;
        }
    }
    assert!(seen[0] > 0 && seen[1] > 0, "kernels seen {seen:?}");
}

/// An unkept forward (`Plan::forward_unkept`, the exact tangent Jacobian of the M-step's
/// fixed-η score) writes the outputs' lanes and the guard exactly as the kept forward does —
/// here the value and both θ-tangents of `c = θ₁θ₀² + θ₀`, checked against their closed
/// forms — reports the guard as its result, and holds no run log or buffer afterwards.
#[test]
fn an_unkept_forward_writes_what_the_kept_one_does() {
    begin();
    let th: Vec<Tr> = (0..2).map(|j| Tr::theta(j, [2.0, 1.5][j])).collect();
    let a = th[0] * th[0];
    assert!(a.br_cmp_c(crate::sens::num::BranchCmp::Gt, 1.0));
    Tr::chunk_mark();
    let c = th[1] * a + th[0];
    let rec = end(Some(vec![c.reg()])).expect("records");
    let plan = rec.plan(Layout::F64, 0, 2).expect("plans");
    let layout = Layout { tan: 2, ..Layout::F64 };
    let l = layout.lanes();
    for (theta, held) in [([2.0_f64, 1.5], true), ([0.5, 1.5], false)] {
        let mut p = vec![0.0; 2 * l];
        for j in 0..2 {
            p[j * l] = theta[j];
            p[j * l + 1 + j] = 1.0;
        }
        let (mut kept, mut unkept) = (vec![0.0; 2 * l], vec![0.0; 2 * l]);
        let f = plan.forward_layout(layout, &[], &p, &mut kept).expect("runs");
        assert_eq!(plan.kept_guard(&f), held);
        assert_eq!(
            plan.forward_unkept(layout, &[], &p, &mut unkept),
            Some(held),
            "the result is the guard"
        );
        assert_eq!(unkept[l], f64::from(u8::from(held)), "the guard lane");
        if held {
            assert_eq!(unkept, kept, "the same lanes, bit for bit");
            let (t0, t1) = (theta[0], theta[1]);
            close(unkept[0], t1 * t0 * t0 + t0, "value");
            close(unkept[1], 2.0 * t0 * t1 + 1.0, "∂/∂θ₀");
            close(unkept[2], t0 * t0, "∂/∂θ₁");
        }
    }
}

/// #3060: a residual covariance `R` of pairs (`correlation cs(RHO) by PAIR`, `n/2` blocks of
/// 2) factored and solved on a recording. The dense factorisation records superlinearly many
/// instructions (here 618 at 16 rows, 2018 at 32: every structural zero below a pivot is a
/// division, and every product with it a node or a zero-fold guard) — which made the outer
/// gradient's kernel of a subject with 84 observations take gigabytes and seconds to
/// compile; the blocked one records `O(n)`. The probe is checked to see the dense growth, so
/// a blocked factorisation that went back to the dense loops would fail the linear bound.
#[test]
fn blocked_cholesky_records_linear_in_the_rows() {
    use crate::sens::focei_kernel::{chol, chol_blocks, chol_solve, chol_solve_blocks};
    use crate::stats::residual_error::RBlocks;
    let count = |n: usize, blocked: bool| -> usize {
        begin();
        let x = Tr::eta(0, 0.3);
        let group: Vec<usize> = (0..n).map(|j| j / 2).collect();
        let a: Vec<Vec<Tr>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| match (i == j, group[i] == group[j]) {
                        (true, _) => x + Tr::from_f64(2.0),
                        (false, true) => x * Tr::from_f64(0.5),
                        (false, false) => Tr::from_f64(0.0),
                    })
                    .collect()
            })
            .collect();
        let rhs: Vec<Tr> = (0..n).map(|j| x * Tr::from_f64(1.0 + j as f64)).collect();
        let z = if blocked {
            let blk = RBlocks::from_groups(&group);
            chol_solve_blocks(&chol_blocks(&a, &blk).expect("factors"), &rhs, &blk)
        } else {
            chol_solve(&chol(&a).expect("factors"), &rhs)
        };
        let s = z.into_iter().fold(Tr::from_f64(0.0), |acc, v| acc + v);
        end(Some(vec![s.reg()])).expect("records").code.len()
    };
    let (d16, d32) = (count(16, false), count(32, false));
    assert!(d32 > 3 * d16, "the probe sees the dense growth: {d16} → {d32}");
    let (b16, b32, b64) = (count(16, true), count(32, true), count(64, true));
    assert!(b32 <= 2 * b16 + 8 && b64 <= 2 * b32 + 8, "blocked: {b16} → {b32} → {b64}");
}

/// A phase boundary (#3077) releases the plans, templates and kept forwards of the fit the
/// thread runs and nothing else: the fit's bad marks and failure records stay (the subject does
/// not re-try a failed plan, and the fit still reports it at its end), another fit's plans,
/// templates and forwards stay, and a thread that runs no fit releases nothing.
/// Mutations: drop `|| v.is_none()` from the retain -> the bad mark is gone (`Missing`); retain
/// on anything but the scope -> B's plan / template / forward go; release under scope `0`
/// instead of returning -> the unregistered plan goes; skip `clear_kept` -> A's forward stays;
/// skip `lock_families().retain` / `prune_caches` -> A's template stays.
#[test]
fn a_phase_boundary_releases_only_the_running_fits_plans_and_keeps_its_bad_marks() {
    let ma = crate::types::test_helpers::analytical_model(crate::types::GradientMethod::default());
    let mb = crate::types::test_helpers::analytical_model(crate::types::GradientMethod::default());
    let (ida, idb) = (ma.plan_id.get(), mb.plan_id.get());
    let (a, b) = (vec![crate::types::Subject::default()], vec![crate::types::Subject::default()]);
    let guard_a = super::in_fit(0, || register(&ma, &a));
    let guard_b = super::in_fit(0, || register(&mb, &b));
    let scope_of =
        |s: &[crate::types::Subject]| registered(s.as_ptr() as usize).expect("registered").1;
    let (scope_a, scope_b) =
        (super::in_fit(guard_a.fit, || scope_of(&a)), super::in_fit(guard_b.fit, || scope_of(&b)));
    assert_ne!(scope_a, scope_b, "fixture: two fits");
    let variant = |v: u64| Desc::of_value(&v);
    // Per fit: a plan, a bad-marked plan, a template and a kept forward.
    // A structure per fit (`depth`): another fit's recording of the same one would replay its
    // template instead of planning one of its own.
    let fill = |m: &crate::types::CompiledModel,
                s: &[crate::types::Subject],
                scope: u64,
                depth: usize| {
        super::in_fit(scope, || {
            let key = SubjectKey::new(m, &s[0], 1, 2).with_variant(variant(0x3077_0001));
            add(&key, square_plan());
            let bad = SubjectKey::new(m, &s[0], 1, 2).with_variant(variant(0x3077_0002));
            mark_bad(&bad, "forced");
            let family =
                SubjectKey::new(m, &s[0], 1, 2).with_variant(variant(0x3077_0003)).family();
            let rec = recorded(&family, || {
                let et: Vec<Tr> = (0..2).map(|k| Tr::eta(k, 1.0)).collect();
                let mut x = et[0];
                for _ in 0..depth {
                    x = x * et[1] + et[0];
                }
                Some(vec![x.reg()])
            })
            .expect("records");
            rec.plan(l2(), 2, 0).expect("plans");
            kept_lock().insert(
                s.as_ptr() as usize,
                Kept { plan: 0, layout: l2(), buf: vec![0.0; 4], scope, model: m.plan_id.get() },
            );
            (key, bad)
        })
    };
    let (key_a, bad_a) = fill(&ma, &a, scope_a, 13);
    let (key_b, _) = fill(&mb, &b, scope_b, 14);
    // An unregistered plan (scope `0`), which a thread running no fit must leave alone.
    let mc = crate::types::test_helpers::analytical_model(crate::types::GradientMethod::default());
    let loose = crate::types::Subject::default();
    let key_0 = super::in_fit(0, || {
        let k = SubjectKey::new(&mc, &loose, 1, 2).with_variant(variant(0x3077_0004));
        add(&k, square_plan());
        k
    });
    assert_eq!(key_0.scope, 0, "fixture: unregistered");
    assert_eq!((templates_of(ida), templates_of(idb)), (1, 1), "fixture: a template each");
    assert_eq!((kept_in(scope_a), kept_in(scope_b)), (1, 1), "fixture: a forward each");

    super::in_fit(0, release_fit_plans);
    assert!(matches!(lookup(&key_0), Lookup::Found(_)), "no fit, nothing released");
    assert_eq!(kept_in(scope_a), 1, "no fit, no forward released");

    super::in_fit(scope_a, release_fit_plans);
    assert!(matches!(lookup(&key_a), Lookup::Missing), "A's plan is released");
    assert!(matches!(lookup(&bad_a), Lookup::Bad), "A's bad mark stays");
    assert_eq!(templates_of(ida), 0, "A's template is released");
    assert!(lock_families().keys().all(|k| k.scope != scope_a), "and its family");
    assert_eq!(kept_in(scope_a), 0, "A's forward is released");
    let reported = super::in_fit(scope_a, || plan_failures(&ma, &a));
    assert_eq!(reported, Some((1, "forced".to_string())), "its failure is still reported");
    assert!(matches!(lookup(&key_b), Lookup::Found(_)), "B's plan stays");
    assert_eq!(templates_of(idb), 1, "B's template stays");
    assert_eq!(kept_in(scope_b), 1, "B's forward stays");
    assert!(matches!(lookup(&key_0), Lookup::Found(_)), "the unregistered plan stays");
    drop((guard_a, guard_b));
    drop_model(ida);
}

/// A phase boundary frees every pool worker's thread-local buffers, not only the calling
/// thread's (#3077): each keeps the largest a pass ever needed. Mutations: release only the
/// calling thread (drop the `rayon::broadcast`) -> the other worker still holds its buffers;
/// leave one pool out of `ode_loop::release_thread_buffers` (`IDX`) -> its table stays.
#[test]
fn a_phase_boundary_frees_every_workers_pooled_buffers() {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().expect("a pool");
    let m = crate::types::test_helpers::analytical_model(crate::types::GradientMethod::default());
    let subs = vec![crate::types::Subject::default()];
    let pooled = || {
        POOL.with(|p| p.borrow().len())
            + LAST.with(|l| usize::from(l.borrow().is_some()))
            + super::ode_loop::pooled_buffers()
    };
    let counts = pool.install(|| {
        let guard = register(&m, &subs);
        let key = SubjectKey::new(&m, &subs[0], 1, 2).with_variant(Desc::of_value(&0x3077_0005u64));
        add(&key, square_plan());
        let fit = guard.fit;
        // Every worker pools a buffer and caches the plan it looked up last.
        rayon::broadcast(|_| {
            give(vec![0.0; 1 << 12]);
            super::in_fit(fit, || assert!(matches!(lookup(&key), Lookup::Found(_))));
            super::ode_loop::Log::default().give();
            super::ode_loop::pool_an_offset_table();
        });
        let before = rayon::broadcast(|_| pooled());
        release_fit_plans();
        let after = rayon::broadcast(|_| pooled());
        drop(guard);
        (before, after)
    });
    assert!(counts.0.iter().all(|&n| n >= 4), "fixture: each worker holds some: {:?}", counts.0);
    assert_eq!(counts.1, vec![0, 0], "each worker's are freed");
    drop(m);
}

/// `f`'s plan's reverses at `eta`, as their adjoints' bits: per output, the first-order reverse
/// from a first-order forward and from a second-order one (a prefix layout: the reverse reads
/// the leading lanes of a wider log), then the rows forward-over-reverse along both η
/// (`run_loop_rows`/`run_loop_rev_rows`) where the plan serves it. Also the loop logs' bytes,
/// kept and as the full log would have held them.
fn reverse_bits(plan: &Plan, n_out: usize, eta: [f64; 2]) -> (Vec<u64>, usize, usize, bool) {
    let mut bits = Vec::new();
    let mut bytes = (0, 0);
    for lay in [Layout { n: 2, m: 0, th: false, wide: false, tan: 0 }, l2()] {
        let l = lay.lanes();
        let mut u = vec![0.0; 2 * l];
        for k in 0..2 {
            u[k * l] = eta[k];
            u[k * l + 1 + k] = 1.0;
        }
        let fwd = plan.forward_kept(lay, &u, &[]).expect("runs");
        for lg in &fwd.logs {
            if let CallLog::Ode(lg) = lg {
                bytes.0 += lg.bytes();
                bytes.1 += lg.full_bytes();
            }
        }
        let l1 = Layout { n: 2, m: 0, th: false, wide: false, tan: 0 };
        let lr = l1.lanes();
        for j in 0..n_out {
            let mut du_bar = vec![0.0; n_out * lr];
            du_bar[j * lr] = 1.0;
            du_bar[j * lr + 1] = 0.5;
            let mut ub = vec![0.0; 2 * lr];
            assert!(plan.reverse_from(&fwd, l1, 1, &mut ub, &mut [], &du_bar), "reverses");
            assert!(ub.iter().all(|x| x.is_finite()), "finite adjoints");
            bits.extend(ub.iter().map(|x| x.to_bits()));
        }
    }
    let rows = plan.rows_serve(Layout::F64);
    if rows {
        let tu = [1.0, 0.0, 0.0, 1.0];
        let rf = plan.forward_rows(Layout::F64, &eta, &[], &tu, 2).expect("rows run");
        let du: Vec<f64> = (0..n_out).map(|j| 1.0 - 0.1 * j as f64).collect();
        let dut: Vec<f64> = (0..2 * n_out).map(|j| 0.3 + 0.05 * j as f64).collect();
        let (mut ub, mut ut) = (vec![0.0; 2], vec![0.0; 4]);
        assert!(plan.reverse_rows(&rf, &du, Some(&dut), &mut ub, &mut [], &mut ut, &mut []));
        assert!(ub.iter().chain(&ut).all(|x| x.is_finite()), "finite rows adjoints");
        bits.extend(ub.iter().chain(&ut).map(|x| x.to_bits()));
    }
    (bits, bytes.0, bytes.1, rows)
}

/// #3078: a loop call's log keeps a checkpoint — the lanes a step's bodies read that no body of
/// the step wrote — instead of every body's inputs, and the reverse regenerates the inputs by
/// running the step forward again. The reverse must be the full log's **bit for bit**: on every
/// stepper family (explicit RK45 and Vern7, the Rosenbrock pair carrying a Jacobian), from a
/// first-order forward, from a second-order one whose leading lanes the reverse reads, and
/// through the rows forward-over-reverse; and each replayed body input a body reads equals the
/// forward's, bit for bit (asserted inside every replay of a test build, so the whole suite
/// checks it). The memory it buys, measured on this two-state fixture, is printed and bounded.
///
/// Mutations: the rewind of a rejected step keeping `known` (a slot it wrote is then never
/// saved), and the clock not restored before a replayed body — each fails the replay's bitwise
/// check here. (A jump's state copy left out of the replay fails the switch fixture of
/// `checkpointed_loop_logs_give_the_full_logs_derivatives_bit_for_bit`; this fixture has no
/// switch.)
#[test]
fn a_checkpointed_ode_log_reverses_bit_for_bit_like_the_full_log() {
    use crate::ode::solver::OdeMethod;
    use ode_loop::{REPLAYED, REPLAY_FROM_FULL};
    for method in [OdeMethod::Rk45, OdeMethod::Vern7, OdeMethod::Rosenbrock23, OdeMethod::Rodas5P] {
        let f = case_ode_interp(method);
        begin();
        let et: Vec<Tr> = (0..2).map(|k| Tr::eta(k, [0.1, -0.2][k])).collect();
        let y = f(&et);
        let rec = end(Some(y.iter().map(|v| v.reg()).collect())).expect("records");
        let plan = rec.plan(l2(), 2, 0).expect("plans");
        for eta in [[0.15, -0.1], [0.9, 0.6]] {
            REPLAYED.with(|c| c.set(0));
            let (got, kept, full, rows) = reverse_bits(&plan, y.len(), eta);
            let replayed = REPLAYED.with(std::cell::Cell::get);
            REPLAY_FROM_FULL.with(|c| c.set(true));
            let (want, ..) = reverse_bits(&plan, y.len(), eta);
            REPLAY_FROM_FULL.with(|c| c.set(false));
            eprintln!(
                "{method:?} at {eta:?}: log {kept} B vs {full} B full ({:.1}×), {replayed} \
                 bodies replayed, rows {rows}",
                full as f64 / kept as f64
            );
            assert!(replayed > 0, "{method:?}: the reverse replayed no body");
            assert!(rows, "{method:?}: the rows path must be exercised");
            assert_eq!(got.len(), want.len());
            for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    a,
                    b,
                    "{method:?} at {eta:?}: adjoint {i}: {} vs {} from the full log",
                    f64::from_bits(*a),
                    f64::from_bits(*b)
                );
            }
            assert!(
                kept < full,
                "{method:?}: the checkpoint ({kept} B) is no smaller than {full} B"
            );
        }
    }
}

/// A phase run on the tighter ODE scope's own pool — the covariance step's stencil, the settle
/// check, the stagnation guard's decrement — leaves that pool's workers holding no plan handle
/// or buffer (#3077). The pool is cached for the next caller, and a phase boundary on the fit's
/// pool cannot reach it, so a worker's `LAST` would keep the plans the boundary pruned alive.
/// Mutation: drop the `release_pool_thread_caches` call from `with_fit_scope_ode` -> the second
/// visit finds the first one's buffers and plan handles on both workers.
#[test]
fn a_tighter_ode_scope_leaves_its_pools_workers_empty() {
    let m = crate::types::test_helpers::analytical_model(crate::types::GradientMethod::default());
    let subs = vec![crate::types::Subject::default()];
    let options = crate::types::FitOptions { threads: Some(2), ..Default::default() };
    // An override no other test arms, so the cached pool this leases is this test's alone.
    let tight = crate::ode::solver::OdeSolverOverride {
        reltol: Some(3.077e-9),
        abstol: Some(3.077e-11),
        ..Default::default()
    };
    let pooled = || {
        POOL.with(|p| p.borrow().len())
            + LAST.with(|l| usize::from(l.borrow().is_some()))
            + super::ode_loop::pooled_buffers()
    };
    let guard = register(&m, &subs);
    let fit = guard.fit;
    let key = SubjectKey::new(&m, &subs[0], 1, 2).with_variant(Desc::of_value(&0x3077_0006u64));
    add(&key, square_plan());
    let (filled, before) = crate::pool::with_fit_scope_ode(&options, tight, || {
        assert_eq!(rayon::current_num_threads(), 2, "fixture: the scope's own pool");
        rayon::broadcast(|_| {
            give(vec![0.0; 1 << 12]);
            super::in_fit(fit, || assert!(matches!(lookup(&key), Lookup::Found(_))));
            super::ode_loop::Log::default().give();
            super::ode_loop::pool_an_offset_table();
        });
        let filled = rayon::broadcast(|_| pooled());
        (filled, std::thread::current().id())
    })
    .expect("the tighter scope");
    assert!(filled.iter().all(|&n| n >= 4), "fixture: each worker holds some: {filled:?}");
    // The next caller of the same scope leases the same idle pool.
    let (after, again) = crate::pool::with_fit_scope_ode(&options, tight, || {
        let ids = rayon::broadcast(|_| std::thread::current().id());
        (rayon::broadcast(|_| pooled()), ids)
    })
    .expect("the tighter scope");
    assert!(again.contains(&before), "fixture: the cached pool is the one leased again");
    assert_eq!(after, vec![0, 0], "the scope's workers were left empty");
    drop(guard);
}
