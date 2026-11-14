//! The 1–3 compartment closed forms against a high-precision reference: values, first and
//! second parameter derivatives on a grid through every confluence the forms meet — `KA` equal
//! to, and `1e-12 … 1e-2` away from, each disposition rate; a near-zero elimination rate
//! (`1 − e^{−kt}` at `kt ~ 1e-6`); a fast-distribution and a near-double-root 3-cpt
//! disposition. The reference is a matrix exponential of the compartment generator in mpmath
//! at 60 digits, differentiated there — a third computation, outside both ferx engines
//! (`tests/data/closed_form_mpmath.txt`, one case per line:
//! `kind;params;t;ii;dur;value;grad;hess upper triangle;peripheral concentrations`).
//! The `oral*` rows also check the event-driven propagators from a loaded depot, and every row
//! with a peripheral column the `pk` peripheral closed forms.

use crate::sens::dual2::Dual2;
use crate::types::DoseEvent;

type D = Dual2<7>;

struct Case {
    kind: String,
    p: Vec<f64>,
    t: f64,
    ii: f64,
    dur: f64,
    value: f64,
    grad: Vec<f64>,
    hess: Vec<f64>,
    periph: Vec<f64>,
}

fn cases() -> Vec<Case> {
    let nums = |s: &str| -> Vec<f64> {
        s.split(',')
            .filter(|x| !x.is_empty())
            .map(|x| x.parse().expect("number"))
            .collect()
    };
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/closed_form_mpmath.txt"
    );
    std::fs::read_to_string(path)
        .expect("fixture")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split(';').collect();
            Case {
                kind: f[0].to_string(),
                p: nums(f[1]),
                t: f[2].parse().unwrap(),
                ii: f[3].parse().unwrap(),
                dur: f[4].parse().unwrap(),
                value: f[5].parse().unwrap(),
                grad: nums(f[6]),
                hess: nums(f[7]),
                periph: nums(f[8]),
            }
        })
        .collect()
}

/// The central concentration of a case through its closed form, on `Dual2` seeded on the
/// case's parameters.
fn closed_form(c: &Case) -> D {
    use crate::sens::{one_cpt as o, three_cpt as h, two_cpt as w};
    let x: Vec<D> = (0..7)
        .map(|i| c.p.get(i).map_or(D::constant(0.0), |&v| D::var(v, i)))
        .collect();
    let (t, one, amt) = (D::constant(c.t), D::constant(1.0), 100.0);
    let rate = if c.dur > 0.0 { amt / c.dur } else { 0.0 };
    match c.kind.as_str() {
        "oral1" => o::one_cpt_oral_g(amt, t, x[0], x[1], x[2], one),
        "oral1ss" => o::one_cpt_oral_ss_g(amt, t, c.ii, x[0], x[1], x[2], one),
        "inf1" => o::one_cpt_infusion_g(rate, c.dur, amt, t, x[0], x[1]),
        "inf1ss" => o::one_cpt_infusion_ss_g(rate, c.dur, amt, t, c.ii, x[0], x[1]),
        "oral2" => w::two_cpt_oral_g(amt, t, x[0], x[1], x[2], x[3], x[4], one),
        "oral2ss" => w::two_cpt_oral_ss_g(amt, t, c.ii, x[0], x[1], x[2], x[3], x[4], one),
        "bolus2" => w::two_cpt_iv_bolus_g(amt, t, x[0], x[1], x[2], x[3]),
        "inf2" => w::two_cpt_infusion_g(rate, c.dur, amt, t, x[0], x[1], x[2], x[3]),
        "oral3" => h::three_cpt_oral_g(amt, t, x[0], x[1], x[2], x[3], x[4], x[5], x[6], one),
        "oral3ss" => {
            h::three_cpt_oral_ss_g(amt, t, c.ii, x[0], x[1], x[2], x[3], x[4], x[5], x[6], one)
        }
        "bolus3" => h::three_cpt_iv_bolus_g(amt, t, x[0], x[1], x[2], x[3], x[4], x[5]),
        "inf3" => h::three_cpt_infusion_g(rate, c.dur, amt, t, x[0], x[1], x[2], x[3], x[4], x[5]),
        k => panic!("unknown kind {k}"),
    }
}

/// The central concentration of an `oral*` (non-SS) case through the event-driven
/// propagator, from the dose in the depot.
fn propagated(c: &Case) -> Option<D> {
    use crate::sens::propagate as p;
    let x: Vec<D> = (0..7)
        .map(|i| c.p.get(i).map_or(D::constant(0.0), |&v| D::var(v, i)))
        .collect();
    let (dt, z) = (D::constant(c.t), D::constant(0.0));
    let mut s = [D::constant(100.0), z, z, z];
    match c.kind.as_str() {
        "oral1" => {
            p::propagate_one_cpt_oral_g(&mut s, dt, x[0], x[1], x[2], z, z);
            Some(s[1] / x[1])
        }
        "oral2" => {
            p::propagate_two_cpt_oral_g(&mut s, dt, x[0], x[1], x[2], x[3], x[4], z, z, z);
            Some(s[1] / x[1])
        }
        "oral3" => {
            p::propagate_three_cpt_oral_g(
                &mut s, dt, x[0], x[1], x[2], x[3], x[4], x[5], x[6], z, z, z, z,
            );
            Some(s[1] / x[1])
        }
        _ => None,
    }
}

/// The peripheral concentrations of a case through the `pk` peripheral closed forms.
fn peripherals(c: &Case) -> Vec<f64> {
    use crate::pk::{three_compartment as h, two_compartment as w};
    let (amt, p) = (100.0, &c.p);
    let ss = c.kind.ends_with("ss");
    let rate = if c.dur > 0.0 { amt / c.dur } else { 0.0 };
    let dose = DoseEvent::new(0.0, amt, 1, rate, ss, c.ii);
    match c.kind.as_str() {
        "oral2" | "oral2ss" => {
            vec![w::two_cpt_oral_peripheral(
                &dose, c.t, p[0], p[1], p[2], p[3], p[4], 1.0,
            )]
        }
        "bolus2" | "inf2" => vec![w::two_cpt_iv_peripheral(&dose, c.t, p[0], p[1], p[2], p[3])],
        "oral3" | "oral3ss" => {
            h::three_cpt_oral_peripherals(&dose, c.t, p[0], p[1], p[2], p[3], p[4], p[5], p[6], 1.0)
                .to_vec()
        }
        "bolus3" | "inf3" => {
            h::three_cpt_iv_peripherals(&dose, c.t, p[0], p[1], p[2], p[3], p[4], p[5]).to_vec()
        }
        _ => vec![],
    }
}

/// Norm-wise relative error of `got` against `want` (`NaN` when `got` is not finite; a
/// non-finite reference panics).
fn rel(got: &[f64], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0.0_f64, |m, w| {
        assert!(w.is_finite(), "reference not finite");
        m.max(w.abs())
    });
    let mut worst = 0.0_f64;
    for (g, w) in got.iter().zip(want) {
        if !g.is_finite() {
            return f64::NAN;
        }
        worst = worst.max((g - w).abs()); // nan-ok: `g` checked above, `w` in `scale`'s fold
    }
    worst / scale
}

/// `(value, gradient, Hessian)` relative errors of `d` against a case's reference.
fn errors(c: &Case, d: &D) -> [f64; 3] {
    let n = c.p.len();
    let mut h = Vec::new();
    for i in 0..n {
        for j in i..n {
            h.push(d.hess[i][j]);
        }
    }
    [
        rel(&[d.value], &[c.value]),
        rel(&d.grad[..n], &c.grad),
        rel(&h, &c.hess),
    ]
}

/// Worst value / gradient / Hessian error over the grid, measured on Linux x86_64 (the table
/// this prints): `9.6e-16 / 2.3e-14 / 6.9e-13` (the Hessian worst at a steady-state oral `KA`
/// on the terminal rate), against `3e-5 / 7.5e3 / 1.3e13` before #2786 replaced the L'Hôpital
/// branches (and a value error of 100% where the propagators and peripherals dropped the
/// depot). ~10× headroom.
const BOUNDS: [f64; 3] = [1e-14, 2e-13, 1e-11];
/// The two corners where the residue form `Σ c·φ(λ)` itself cancels in the jets: modes a gap
/// `Δ` apart with `Δ·t ≪ 1` (2-cpt with `CL = 1e-5, Q = 1e-4`, `Δ·t = 7e-6`; 3-cpt with two
/// roots 1.2% apart from `Q3 = 1e-4`), where `c_λ`'s derivatives grow as `1/Δ²` and sum to
/// those of `Σ c = 1`. Measured `2.1e-16 / 1.2e-10 / 1.1e-6`; values stay at round-off.
const CORNER_BOUNDS: [f64; 3] = [1e-14, 1e-9, 1e-5];
/// Peripheral concentrations (values only): `7.6e-14` measured, `5e-12` in the corners.
const PERIPH_BOUND: f64 = 1e-12;
const CORNER_PERIPH_BOUND: f64 = 1e-10;

/// Whether a case is one of the residue-form corners [`CORNER_BOUNDS`] describes.
fn corner(c: &Case) -> bool {
    match c.kind.as_str() {
        "inf2" | "bolus2" => c.p[0] < 1e-4,
        "inf3" | "bolus3" => c.p[4] < 1e-3,
        _ => false,
    }
}

#[test]
fn closed_forms_match_the_mpmath_reference_through_every_confluence() {
    let cs = cases();
    assert!(cs.len() > 150, "fixture truncated: {} cases", cs.len());
    let mut worst: std::collections::BTreeMap<String, ([f64; 3], String)> = Default::default();
    let mut fails = Vec::new();
    let mut note = |fam: String, e: [f64; 3], c: &Case, bound: [f64; 3]| {
        let w = worst
            .entry(fam.clone())
            .or_insert(([0.0; 3], String::new()));
        for k in 0..3 {
            // `NaN` (a non-finite result) is the worst error there is.
            if e[k].is_nan() || e[k] > w.0[k] {
                w.0[k] = if e[k].is_nan() { f64::INFINITY } else { e[k] };
                if k == 2 || w.1.is_empty() {
                    w.1 = format!("{:?} t={}", c.p, c.t);
                }
            }
            if e[k].is_nan() || e[k] > bound[k] {
                fails.push(format!(
                    "{fam} [{k}] err {:.2e} at {:?} t={}",
                    e[k], c.p, c.t
                ));
            }
        }
    };
    for c in &cs {
        let (bound, periph_bound) = if corner(c) {
            (CORNER_BOUNDS, CORNER_PERIPH_BOUND)
        } else {
            (BOUNDS, PERIPH_BOUND)
        };
        note(c.kind.clone(), errors(c, &closed_form(c)), c, bound);
        if let Some(d) = propagated(c) {
            note(format!("{}-propagator", c.kind), errors(c, &d), c, bound);
        }
        let got = peripherals(c);
        if !got.is_empty() {
            let e = rel(&got, &c.periph);
            note(
                format!("{}-periph", c.kind),
                [e, 0.0, 0.0],
                c,
                [periph_bound, 1.0, 1.0],
            );
        }
    }
    for (fam, (e, at)) in &worst {
        eprintln!(
            "{fam:>18}: value {:.1e}  grad {:.1e}  hess {:.1e}   (worst at {at})",
            e[0], e[1], e[2]
        );
    }
    assert!(
        fails.is_empty(),
        "{} failures:\n{}",
        fails.len(),
        fails.join("\n")
    );
}
