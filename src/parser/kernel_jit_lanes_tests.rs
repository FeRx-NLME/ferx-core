//! The dual-granularity tier against the lowered scalar code it stands in for: each operation's
//! forward lanes bit for bit equal to [`ir::forward`]'s, on
//! every payload form a plan uses.
use super::super::super::{Flat, Unary};
use super::super::ir::{self, Cc, Layout};
use super::super::DirMask;
use super::{AtLayout, LaneProg};

fn layouts() -> Vec<(&'static str, Layout)> {
    let l = |n, m, th, wide| Layout {
        n,
        m,
        th,
        wide,
        tan: 0,
    };
    vec![
        ("f64", l(0, 0, false, false)),
        ("first", l(2, 0, false, false)),
        ("first-wide", l(2, 0, false, true)),
        ("triangle", l(3, 6, false, false)),
        ("triangle-wide", l(2, 3, false, true)),
        ("directional", l(3, 3, true, false)),
    ]
}

/// A deterministic spread of lane values (no RNG dependency): the value lane from `v`.
fn lanes_of(v: f64, lanes: usize, salt: u64) -> Vec<f64> {
    let mut x = salt.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..lanes)
        .map(|j| {
            if j == 0 {
                return v;
            }
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x % 2001) as f64 - 1000.0) / 700.0
        })
        .collect()
}

/// One operation on inputs `ins` (their value lanes), and its lanes forward both ways. `zero_tan` keeps input `i`'s tangents and payload zero.
fn check(name: &str, op: Flat, vals: &[f64], zero_tan: &[usize]) {
    check_mode(name, op, vals, zero_tan, false);
    // Every input without a tangent, and marked so: the value-only path.
    let all: Vec<usize> = (0..vals.len()).collect();
    check_mode(name, op, vals, &all, true);
}

fn check_mode(name: &str, op: Flat, vals: &[f64], zero_tan: &[usize], dead: bool) {
    for (lname, layout) in layouts() {
        let l = layout.lanes();
        let n_in = vals.len();
        // `r0..` the inputs, the operation into `r{n_in}`, then its output.
        let mut code: Vec<Flat> = (0..n_in)
            .map(|i| Flat::LoadState {
                d: i as u32,
                i: i as u32,
            })
            .collect();
        code.push(op);
        code.push(Flat::Deriv {
            state: 0,
            a: n_in as u32,
        });
        let n_regs = n_in + 1;
        let mut u: Vec<f64> = Vec::new();
        for (i, &v) in vals.iter().enumerate() {
            let mut x = lanes_of(v, l, 17 + i as u64 * 31 + layout.n as u64);
            if zero_tan.contains(&i) {
                x[1..].fill(0.0);
            }
            u.extend(x);
        }
        let what = format!("{name} on {lname}{}", if dead { " (dead)" } else { "" });

        // Forward: the lowered code against this tier.
        let f = ir::forward(&code, n_regs, layout);
        let mut du_ir = vec![0.0; l];
        {
            let (mut p, mut s) = (vec![0.0; 1], vec![0.0; 1]);
            let mut uu = u.clone();
            ir::interpret(&f, &mut [&mut uu, &mut p, &mut s, &mut du_ir]);
        }
        let prog = LaneProg::of(&code, n_regs, n_in).expect("no `if`");
        let spec: Vec<(DirMask, u16)> = (0..n_in)
            .map(|_| (if dead { DirMask::NONE } else { DirMask::ALL }, u16::MAX))
            .collect();
        let at = AtLayout::of(&prog, layout, &spec);
        let mut base = u.clone();
        base.extend(vec![0.0; l]);
        let idx: Vec<u32> = (0..=n_in as u32).collect();
        let mut regs = vec![0.0; n_regs * l];
        unsafe {
            prog.forward(
                &at,
                layout.wide,
                &mut regs,
                base.as_mut_ptr(),
                idx.as_ptr(),
                [0.0].as_ptr(),
                true,
                &mut vec![0.0; 10 * l],
            )
        };
        let du_lane = &base[n_in * l..];
        for (j, (a, b)) in du_ir.iter().zip(du_lane).enumerate() {
            // Dead: the folding's exactness — the same up to the sign of a zero.
            assert!(
                a.to_bits() == b.to_bits()
                    || (a.is_nan() && b.is_nan())
                    || (dead && (a == b || a.is_nan())),
                "{what}: forward lane {j}: lowered {a:e}, lanes {b:e}"
            );
        }
    }
}

#[test]
fn arithmetic_matches_the_lowered_code() {
    let (d, a, b) = (2, 0, 1);
    check("add", Flat::Add { d, a, b }, &[1.3, -0.4], &[]);
    check("sub", Flat::Sub { d, a, b }, &[1.3, -0.4], &[]);
    check("rsub", Flat::RSub { d, a, b }, &[1.3, -0.4], &[]);
    check("mul", Flat::Mul { d, a, b }, &[1.3, -0.4], &[]);
    check("mul-self", Flat::Mul { d: 1, a: 0, b: 0 }, &[1.3], &[]);
    check("div-raw", Flat::DivRaw { d, a, b }, &[1.3, -0.7], &[]);
    check("div", Flat::Div { d, a, b }, &[1.3, -0.7], &[]);
    check("div-guarded", Flat::Div { d, a, b }, &[1.3, 1e-31], &[]);
    for c in [2.5, -0.3] {
        let (d, a) = (1, 0);
        check("add-k", Flat::AddK { d, a, c }, &[0.9], &[]);
        check("sub-k", Flat::SubK { d, a, c }, &[0.9], &[]);
        check("rsub-k", Flat::RSubK { d, a, c }, &[0.9], &[]);
        check("mul-k", Flat::MulK { d, a, c }, &[0.9], &[]);
        check("div-k", Flat::DivK { d, a, c }, &[0.9], &[]);
    }
    check(
        "div-k-zero",
        Flat::DivK {
            d: 1,
            a: 0,
            c: 1e-31,
        },
        &[0.9],
        &[],
    );
}

#[test]
fn powers_match_the_lowered_code() {
    let (d, a, e) = (2, 0, 1);
    // A constant exponent (no tangent): the lifted `powf`, including at `a = 0`.
    check("pow-const", Flat::Pow { d, a, e }, &[1.7, 2.3], &[1]);
    check("pow-const-int", Flat::Pow { d, a, e }, &[1.7, 3.0], &[1]);
    check(
        "pow-const-zero-base",
        Flat::Pow { d, a, e },
        &[0.0, 2.0],
        &[1],
    );
    // An exponent with a tangent: `exp(e·ln a)`.
    check("pow-general", Flat::Pow { d, a, e }, &[1.7, 0.6], &[]);
}

#[test]
fn unaries_match_the_lowered_code() {
    let cases: [(Unary, f64); 18] = [
        (Unary::Exp, 0.4),
        (Unary::Ln, 1.9),
        (Unary::Ln, 1e-40),
        (Unary::LnRaw, 1.9),
        (Unary::Sqrt, 2.2),
        (Unary::InvLogit, 0.3),
        (Unary::Logit, 0.3),
        (Unary::Cos, 0.8),
        (Unary::Sin, 0.8),
        (Unary::Acos, 0.35),
        (Unary::LnGamma, 2.6),
        (Unary::Digamma, 2.6),
        (Unary::Trigamma, 2.6),
        (Unary::ExpRel(0), 0.45),
        (Unary::ExpRel(2), -1.3),
        (Unary::Abs, -0.6),
        (Unary::Abs, 0.6),
        (Unary::StopGrad, 0.6),
    ];
    for (op, x) in cases {
        check(
            &format!("{op:?}({x})"),
            Flat::Unary { d: 1, op, a: 0 },
            &[x],
            &[],
        );
    }
    for op in [Unary::Floor, Unary::Ceil, Unary::Round] {
        check(
            &format!("{op:?}"),
            Flat::Unary { d: 1, op, a: 0 },
            &[1.6],
            &[],
        );
    }
}

#[test]
fn the_gamma_primitive_matches_the_lowered_code() {
    for upper in [false, true] {
        for (i, j) in [(0, 0), (1, 0), (0, 1)] {
            check(
                &format!("gamma upper={upper} ({i},{j})"),
                Flat::GammaPq {
                    d: 2,
                    a: 0,
                    x: 1,
                    upper,
                    i,
                    j,
                },
                &[2.4, 1.7],
                &[],
            );
        }
    }
}

#[test]
fn selections_and_conditions_match_the_lowered_code() {
    // `r2 = r0 < r1`, then a select on it of `r0` and `r1`.
    for vals in [[0.3, 0.9], [0.9, 0.3]] {
        let code = Flat::Cmp {
            d: 2,
            cc: Cc::Lt,
            a: 0,
            b: 1,
        };
        check("cmp", code, &vals, &[]);
        check(
            "select",
            Flat::Select {
                d: 3,
                c: 2,
                a: 0,
                b: 1,
            },
            &[vals[0], vals[1], (vals[0] < vals[1]) as u8 as f64],
            &[2],
        );
    }
}
