//! `rhs_regcode` is a translation of the interpreter, so its tests are bit-identity tests:
//! every value the compiled program writes must be the interpreter's, bit for bit, on
//! `f64` and on the dual types the adjoint records over.
use super::super::{
    eval_statements_g, eval_statements_indexed_with_stack, parse_model_string, ModelTimeGuard,
    OdeRhsProgram,
};
use super::{RhsLayout, RhsRegProgram};
use crate::sens::num::vm::VmNum;
use crate::sens::num::PkNum;
use crate::sens::vfree::DualF;

/// xorshift64*, so the generated programs are the same on every run.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A random expression over the RHS's names, using every operation the compiler
/// translates (and literals that hit the guards: a near-zero divisor, `ln` of a
/// non-positive value, `0^x`).
fn expr(rng: &mut Rng, names: &[&str], depth: usize) -> String {
    if depth == 0 || rng.below(4) == 0 {
        return match rng.below(6) {
            0 => format!("{:.6}", rng.unit() * 4.0 - 1.0),
            1 => ["1e-40", "0", "2", "0.5", "-3"][rng.below(5)].to_string(),
            2 => "TIME".to_string(),
            _ => names[rng.below(names.len())].to_string(),
        };
    }
    let a = expr(rng, names, depth - 1);
    let b = expr(rng, names, depth - 1);
    match rng.below(15) {
        // An inline `if`, nested when an arm draws another, and the `min`/`max` that
        // desugar to one. An arm that ends in a leaf fuses with the op after the `if` on the
        // else path only (`x + (if (…) a else B)`), which `conditional` must reproduce.
        12 => format!("(if ({}) {a} else {b})", cond(rng, names, depth - 1)),
        13 => format!("min({a}, {b})"),
        14 => format!("max({a}, {b})"),
        0 => format!("({a} + {b})"),
        1 => format!("({a} - {b})"),
        2 => format!("({a} * {b})"),
        3 => format!("({a} / {b})"),
        4 => format!("({a} ^ 2)"),
        5 => format!("exp({a} / 8)"),
        6 => format!("ln({a})"),
        7 => format!("abs({a})"),
        8 => ["floor", "ceil", "round"][rng.below(3)].to_string() + &format!("({a})"),
        9 => format!("(-{a})"),
        10 => format!("(2 - {a})"),
        _ => format!("({a} * 3)"),
    }
}

/// A random condition: every comparison, the connectives and `present`, over operands that
/// are equal often enough (the seeds hold zeros) for `==`, `<=` and `>=` to split.
fn cond(rng: &mut Rng, names: &[&str], depth: usize) -> String {
    let d = depth.min(1);
    match rng.below(9) {
        0 => format!("({}) && ({})", cond(rng, names, 0), cond(rng, names, 0)),
        1 => format!("({}) || ({})", cond(rng, names, 0), cond(rng, names, 0)),
        2 => format!("!({})", cond(rng, names, 0)),
        3 => format!("present({})", expr(rng, names, d)),
        _ => {
            // A left operand opening with `(` parses as a parenthesised condition, so it
            // is a leaf.
            let op = ["<", "<=", ">", ">=", "==", "!="][rng.below(6)];
            format!("{} {op} {}", expr(rng, names, 0), expr(rng, names, d))
        }
    }
}

/// A model whose `[odes]` block is `odes`, parsed by the production parser.
fn program(odes: &str) -> OdeRhsProgram {
    program_states(3, odes)
}

/// [`program`] with `n` states `x1…xn` (observing `x2`).
fn program_states(n: usize, odes: &str) -> OdeRhsProgram {
    let states = (1..=n)
        .map(|i| format!("x{i}"))
        .collect::<Vec<_>>()
        .join(", ");
    let text = format!(
        "[parameters]
  theta TVA(1.0, 0.01, 10.0)
  theta TVB(2.0, 0.01, 10.0)
           theta TVC(0.5, 0.01, 10.0)
  omega ETA_A ~ 0.1
  sigma S ~ 0.1 (sd)

         [individual_parameters]
  A = TVA * exp(ETA_A)
  B = TVB
  C = TVC

         [structural_model]
  ode(obs_cmt=x2, states=[{states}])

         [odes]
{odes}

[error_model]
  DV ~ proportional(S)
"
    );
    let model = parse_model_string(&text).unwrap_or_else(|e| {
        panic!(
            "{e}
{text}"
        )
    });
    model.ode_spec.unwrap().rhs_program.unwrap()
}

/// A chain of `n` states: `x1' = −A·x1`, `xk' = A·x(k−1) − B·xk`.
fn chain_odes(n: usize) -> String {
    let mut lines = vec!["  d/dt(x1) = -A * x1".to_string()];
    for k in 2..=n {
        lines.push(format!("  d/dt(x{k}) = A * x{} - B * x{k}", k - 1));
    }
    lines.join(
        "
",
    )
}

fn random_odes(rng: &mut Rng) -> String {
    let mut names = vec!["x1", "x2", "x3", "A", "B", "C"];
    let mut lines = Vec::new();
    // Local assignments: a later expression reads the assigned value, not the slot's
    // value on entry (the renaming `lower` does).
    for local in ["L1", "L2"] {
        if rng.below(2) == 0 {
            lines.push(format!("  {local} = {}", expr(rng, &names, 3)));
            names.push(local);
        }
    }
    for s in ["x1", "x2", "x3"] {
        lines.push(format!("  d/dt({s}) = {}", expr(rng, &names, 4)));
    }
    lines.join("\n")
}

/// Values that reach the guards: negatives, zeros, one tiny value.
fn seed_values(rng: &mut Rng, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| match i % 5 {
            0 => rng.unit() * 6.0 - 3.0,
            1 => 0.0,
            2 => 1e-35,
            _ => rng.unit() * 3.0,
        })
        .collect()
}

fn layout(prog: &OdeRhsProgram) -> RhsLayout<'_> {
    RhsLayout {
        n_vars: prog.n_vars_total,
        state_count: prog.state_count,
        indiv_to_params_slot: &prog.indiv_to_params_slot,
        seed_slots: [
            prog.time_slot,
            prog.tafd_slot,
            prog.tad_slot,
            prog.macheps_slot,
        ],
    }
}

/// `vars` as the binding fills them from `u`, `params` and `seeds` (locals zero).
fn bind<T: VmNum>(prog: &OdeRhsProgram, u: &[T], params: &[T], seeds: &[T; 4]) -> Vec<T> {
    let mut vars = vec![T::from_f64(0.0); prog.n_vars_total];
    vars[..prog.state_count].copy_from_slice(&u[..prog.state_count]);
    for (i, &slot) in prog.indiv_to_params_slot.iter().enumerate() {
        vars[prog.state_count + i] = params[slot];
    }
    for (k, slot) in [
        prog.time_slot,
        prog.tafd_slot,
        prog.tad_slot,
        prog.macheps_slot,
    ]
    .into_iter()
    .enumerate()
    {
        if let Some(v) = vars.get_mut(slot) {
            *v = seeds[k];
        }
    }
    vars
}

fn bits_eq<T: VmNum>(a: &[T], b: &[T]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.same_bits(y))
}

/// Compiled vs the generic interpreter on one `T`; whether the program compiled.
fn check_on<T: VmNum>(
    prog: &OdeRhsProgram,
    u: &[T],
    params: &[T],
    seeds: &[T; 4],
    label: &str,
) -> bool {
    let Ok(reg) = RhsRegProgram::compile(&prog.stmts, &layout(prog)) else {
        return false;
    };
    let mut vars = bind(prog, u, params, seeds);
    let (mut du_ref, mut du_reg) = (vec![T::from_f64(9.0); 3], vec![T::from_f64(9.0); 3]);
    let mut stack = Vec::new();
    let _time = ModelTimeGuard::enter(1.75);
    eval_statements_g::<T>(
        &prog.stmts,
        &[],
        &[],
        &[],
        &mut vars,
        Some(&mut du_ref),
        &mut stack,
        &[],
        &[],
    );
    let mut regs = vec![T::from_f64(0.0); reg.n_regs()];
    reg.run(u, params, seeds, &mut du_reg, &mut regs);
    assert!(bits_eq(&du_ref, &du_reg), "{label}: du differs");
    true
}

#[test]
fn compiled_rhs_is_the_interpreter_bit_for_bit_on_random_programs() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut compiled = 0;
    let n_params = crate::types::MAX_PK_PARAMS + 2;
    for case in 0..250 {
        let odes = random_odes(&mut rng);
        let prog = program(&odes);
        let raw_u = seed_values(&mut rng, 3);
        let raw_p = seed_values(&mut rng, n_params);
        let raw_s = [1.75, rng.unit() * 5.0, rng.unit() * 2.0, f64::EPSILON];
        let label = format!("case {case}:\n{odes}");

        // f64 against the production `f64` interpreter (`eval_bytecode`'s lane VM), which
        // is the evaluator the scalar RHS replaced.
        if let Ok(reg) = RhsRegProgram::compile(&prog.stmts, &layout(&prog)) {
            let mut vars = bind(&prog, &raw_u, &raw_p, &raw_s);
            let (mut du_ref, mut du_reg) = (vec![9.0; 3], vec![9.0; 3]);
            let mut stack = Vec::new();
            let _time = ModelTimeGuard::enter(1.75);
            eval_statements_indexed_with_stack(
                &prog.stmts,
                &[],
                &[],
                &[],
                &mut vars,
                Some(&mut du_ref),
                &[],
                &mut stack,
            );
            let mut regs = vec![0.0; reg.n_regs()];
            reg.run(&raw_u, &raw_p, &raw_s, &mut du_reg, &mut regs);
            assert!(
                bits_eq(&du_ref, &du_reg),
                "{label}\nf64: du {du_ref:?} vs {du_reg:?}"
            );
        }
        // The duals: the inner gradient's tape-free first-order type, and the untaped
        // second-order one (value, tangents and the packed Hessian lanes all compared).
        let lift_f = |x: &[f64]| -> Vec<DualF<2, 0>> {
            x.iter()
                .enumerate()
                .map(|(i, &v)| DualF::<2, 0>::var(v, i % 2) * DualF::constant(0.5 + i as f64 / 7.0))
                .collect()
        };
        let (uf, pf, sf) = (lift_f(&raw_u), lift_f(&raw_p), lift_f(&raw_s));
        let sf: [DualF<2, 0>; 4] = [sf[0], sf[1], sf[2], sf[3]];
        compiled += usize::from(check_on(
            &prog,
            &uf,
            &pf,
            &sf,
            &format!("{label}\nDualF<2,0>"),
        ));
        let lift_v = |x: &[f64]| -> Vec<DualF<2, 3>> {
            x.iter()
                .enumerate()
                .map(|(i, &v)| {
                    let a = DualF::<2, 3>::var(v, i % 2);
                    a * a * DualF::constant(0.25) + DualF::var(0.3, (i + 1) % 2)
                })
                .collect()
        };
        let (uv, pv, sv) = (lift_v(&raw_u), lift_v(&raw_p), lift_v(&raw_s));
        let sv: [DualF<2, 3>; 4] = [sv[0], sv[1], sv[2], sv[3]];
        check_on(&prog, &uv, &pv, &sv, &format!("{label}\nDualV<2,3>"));
        // The first- and second-order forward duals (`Dual1`, `Dual2`), which the
        // sensitivity solves run.
        use crate::sens::dual1::Dual1;
        use crate::sens::dual2::Dual2;
        let lift_1 = |x: &[f64]| -> Vec<Dual1<2>> {
            x.iter()
                .enumerate()
                .map(|(i, &v)| Dual1::<2>::var(v, i % 2) * Dual1::constant(0.5 + i as f64 / 7.0))
                .collect()
        };
        let (u1, p1, s1) = (lift_1(&raw_u), lift_1(&raw_p), lift_1(&raw_s));
        let s1: [Dual1<2>; 4] = [s1[0], s1[1], s1[2], s1[3]];
        check_on(&prog, &u1, &p1, &s1, &format!("{label}\nDual1<2>"));
        let lift_2 = |x: &[f64]| -> Vec<Dual2<2>> {
            x.iter()
                .enumerate()
                .map(|(i, &v)| {
                    let a = <Dual2<2> as PkNum>::var(v, i % 2);
                    a * a * Dual2::constant(0.25) + <Dual2<2> as PkNum>::var(0.3, (i + 1) % 2)
                })
                .collect()
        };
        let (u2, p2, s2) = (lift_2(&raw_u), lift_2(&raw_p), lift_2(&raw_s));
        let s2: [Dual2<2>; 4] = [s2[0], s2[1], s2[2], s2[3]];
        check_on(&prog, &u2, &p2, &s2, &format!("{label}\nDual2<2>"));
    }
    // The generator only emits translatable programs, so they all compile; a collapse
    // here means the compiler started declining and the test checks nothing.
    assert_eq!(compiled, 250, "only {compiled} programs compiled");
}

#[test]
fn a_program_outside_the_translatable_shape_is_not_compiled() {
    // The ops whose f64 meaning the generic primitives do not reproduce.
    for odes in [
        "  d/dt(x1) = sqrt(x1)\n  d/dt(x2) = -x2\n  d/dt(x3) = 0",
        "  d/dt(x1) = inv_logit(x1)\n  d/dt(x2) = -x2\n  d/dt(x3) = 0",
        "  d/dt(x1) = logit(x1)\n  d/dt(x2) = -x2\n  d/dt(x3) = 0",
    ] {
        let prog = program(odes);
        assert!(
            RhsRegProgram::compile(&prog.stmts, &layout(&prog)).is_err(),
            "{odes}"
        );
        assert!(prog.reg.is_none(), "{odes}");
    }
    // And the production program carries the compiled form when it translates, an inline
    // `if` included (#2680).
    let prog = program("  d/dt(x1) = -A * x1\n  d/dt(x2) = A * x1 - B * x2\n  d/dt(x3) = C");
    assert!(prog.reg.is_some());
    let prog = program("  d/dt(x1) = if (x1 < 1) A else B\n  d/dt(x2) = -x2\n  d/dt(x3) = 0");
    assert!(prog.reg.is_some());
}

/// The register code's result, with the native function switched off.
fn run_without_jit<T: VmNum>(reg: &RhsRegProgram, u: &[T], p: &[T], s: &[T; 4]) -> Vec<T> {
    let mut du = vec![T::from_f64(9.0); 3];
    let mut regs = vec![T::from_f64(0.0); reg.n_regs()];
    super::RHS_JIT_DISABLED.with(|c| c.set(true));
    reg.run(u, p, s, &mut du, &mut regs);
    super::RHS_JIT_DISABLED.with(|c| c.set(false));
    du
}

/// A value of `T` from its lanes, contiguous or not.
fn from_lanes_any<T: VmNum>(l: &[f64]) -> T {
    if T::CONTIGUOUS {
        assert_eq!(std::mem::size_of::<T>(), l.len() * 8);
        // SAFETY: a contiguous `T` is exactly its lanes (`#[repr(C)]` `f64`s).
        unsafe { std::ptr::read(l.as_ptr().cast::<T>()) }
    } else {
        T::from_lanes(l)
    }
}

/// `x` lifted to `T`: tangents and payload of mixed sign, zeros and `−0.0`, so `0 − g` vs
/// `−g` and the `x·0` products are exercised.
fn lift_lanes<T: VmNum>(x: &[f64]) -> Vec<T> {
    let w = T::LANES.expect("a native layout").lanes();
    x.iter()
        .enumerate()
        .map(|(i, &v)| {
            let mut l = vec![v];
            for k in 1..w {
                l.push(match (i + k) % 5 {
                    0 => 0.0,
                    1 => 0.5 + i as f64 / 7.0,
                    2 => -1.25,
                    3 => -0.0,
                    _ => 0.375,
                });
            }
            from_lanes_any(&l)
        })
        .collect()
}

/// Native vs register code on one `T`.
fn check_native<T: VmNum>(reg: &RhsRegProgram, x: (&[T], &[T], &[T; 4]), label: &str) {
    assert!(
        reg.has_native::<T>(),
        "{label}: no native function for {}",
        std::any::type_name::<T>()
    );
    let want = run_without_jit(reg, x.0, x.1, x.2);
    let mut got = vec![T::from_f64(9.0); 3];
    let mut regs = vec![T::from_f64(0.0); reg.n_regs()];
    reg.run(x.0, x.1, x.2, &mut got, &mut regs);
    assert!(
        bits_eq(&want, &got),
        "{label}: native du differs ({})",
        std::any::type_name::<T>()
    );
}

fn check_native_lifted<T: VmNum>(
    reg: &RhsRegProgram,
    raw: (&[f64], &[f64], &[f64; 4]),
    label: &str,
) {
    let (u, p, s) = (
        lift_lanes::<T>(raw.0),
        lift_lanes::<T>(raw.1),
        lift_lanes::<T>(raw.2),
    );
    let s: [T; 4] = [s[0], s[1], s[2], s[3]];
    check_native(reg, (&u[..], &p[..], &s), label);
}

#[test]
fn native_rhs_is_the_register_code_bit_for_bit_on_random_programs() {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let n_params = crate::types::MAX_PK_PARAMS + 2;
    for case in 0..250 {
        let odes = random_odes(&mut rng);
        let prog = program(&odes);
        let reg = RhsRegProgram::compile(&prog.stmts, &layout(&prog))
            .unwrap_or_else(|e| panic!("case {case} did not compile ({e:?}):\n{odes}"));
        let raw_u = seed_values(&mut rng, 3);
        let raw_p = seed_values(&mut rng, n_params);
        let raw_s = [1.75, rng.unit() * 5.0, rng.unit() * 2.0, f64::EPSILON];
        let label = format!("case {case}:\n{odes}");
        let _time = ModelTimeGuard::enter(1.75);
        let raw = (&raw_u[..], &raw_p[..], &raw_s);
        check_native(&reg, raw, &label);
        // First order: the inner gradient's widths, inlined and out of line (`N ≥ 8`).
        check_native_lifted::<DualF<1, 0>>(&reg, raw, &label);
        check_native_lifted::<DualF<2, 0>>(&reg, raw, &label);
        check_native_lifted::<DualF<3, 0>>(&reg, raw, &label);
        check_native_lifted::<DualF<7, 0>>(&reg, raw, &label);
        check_native_lifted::<DualF<8, 0>>(&reg, raw, &label);
        check_native_lifted::<DualF<12, 0>>(&reg, raw, &label);
        // Second order: the packed η-Hessian payload, contiguous through `DualF`'s lanes.
        check_native_lifted::<DualF<2, 3>>(&reg, raw, &label);
        check_native_lifted::<DualF<4, 10>>(&reg, raw, &label);
        check_native_lifted::<DualF<3, 6>>(&reg, raw, &label);
        check_native_lifted::<DualF<8, 36>>(&reg, raw, &label);
    }
}

#[test]
fn a_too_wide_dual_keeps_the_register_code() {
    let prog = program("  d/dt(x1) = -A * x1 + 2\n  d/dt(x2) = A * x1 - B * x2\n  d/dt(x3) = C");
    let reg = prog.reg.as_ref().expect("translatable");
    // A payload past the lane cap has no native function.
    assert!(!reg.has_native::<DualF<24, 300>>());
    assert!(reg.has_native::<DualF<4, 10>>());
}

#[test]
fn native_fd_columns_are_the_lane_evaluator_bit_for_bit() {
    // The Rosenbrock Jacobian's batched columns: one native call per column against the
    // lane evaluator, on the production closures, over programs with and without `TIME`.
    let mut rng = Rng(0x94D0_49BB_1331_11EB);
    let n_params = crate::types::MAX_PK_PARAMS + 2;
    let (mut checked, mut split) = (0, 0);
    for case in 0..120 {
        let odes = random_odes(&mut rng);
        let prog = program(&odes);
        let Some(columns) = prog.columns.as_ref() else {
            continue;
        };
        assert!(
            prog.reg.as_ref().is_some_and(|r| r.has_native_f64()),
            "case {case}"
        );
        let u = seed_values(&mut rng, 3);
        let p = seed_values(&mut rng, n_params);
        let t = rng.unit() * 5.0;
        // Every state perturbed once, one twice with a negative step, past one lane pass.
        let cols: Vec<(usize, f64)> =
            [(0, 1e-7), (1, 2.5e-8), (2, 1e-6), (1, -3e-7), (0, 0.0)].repeat(2);
        let mut want = vec![7.0; cols.len() * 3];
        let mut got = vec![7.0; cols.len() * 3];
        super::RHS_JIT_DISABLED.with(|c| c.set(true));
        let ok_want = columns(&u, &p, t, &cols, &mut want);
        super::RHS_JIT_DISABLED.with(|c| c.set(false));
        let ok_got = columns(&u, &p, t, &cols, &mut got);
        if !ok_want && ok_got {
            // The lanes decline a batch whose columns take different arms of an `if`; the
            // native arm runs each column alone, as the caller's per-column fallback would.
            // Each is the RHS at its own point (the register code, bit for bit the
            // interpreter).
            let reg = prog.reg.as_ref().expect("native");
            let seeds = prog.binding.f64_seeds(&p, t);
            for (c, &(j, d)) in cols.iter().enumerate() {
                let mut x = u.clone();
                x[j] = u[j] + d;
                let mut du = vec![0.0; 3];
                let mut regs = vec![0.0; reg.n_regs()];
                super::RHS_JIT_DISABLED.with(|c| c.set(true));
                reg.run(&x, &p, &seeds, &mut du, &mut regs);
                super::RHS_JIT_DISABLED.with(|c| c.set(false));
                assert!(
                    bits_eq(&du, &got[c * 3..(c + 1) * 3]),
                    "case {case}, column {c}:\n{odes}\nscalar {du:?}\nnative {got:?}"
                );
            }
            split += 1;
            continue;
        }
        assert_eq!(ok_want, ok_got, "case {case}:\n{odes}");
        assert!(
            bits_eq(&want, &got),
            "case {case}:\n{odes}\nlanes {want:?}\nnative {got:?}"
        );
        checked += usize::from(ok_want);
    }
    assert!(
        checked > 100,
        "only {checked} programs ran the lane evaluator"
    );
    assert!(split > 0, "no batch split on an `if`");
}

/// A first-order type that records a reverse tape keeps the hoisted interpreter: with the
/// hoist, a lifted parameter-only value (`K = A / B` here) is recorded once per recording
/// and reused by every later call; compiled, it would be recorded again per call — a longer
/// tape whose sweep sums the same adjoints in another order. Three RHS calls on one
/// recording, production program vs the same model parsed with the compiler off: the same
/// tape length, the same output nodes and the same adjoints, bit for bit.
#[test]
fn a_taped_first_order_rhs_records_the_hoisted_tape() {
    use crate::sens::rtape::{rinput, rsweep, rtape_len, rtape_reset, R};
    let odes = "  K = A / B\n  d/dt(x1) = -K * x1\n  d/dt(x2) = K * x1 - C * x2\n  d/dt(x3) = x2";
    let production = program(odes);
    let interpreted = {
        let _off = super::super::RhsRegcodeOff::enter();
        program(odes)
    };
    assert!(production.reg.is_some() && interpreted.reg.is_none());
    assert!(production.hoist.is_some(), "the fixture must lift `A / B`");
    let record = |prog: &OdeRhsProgram| -> (usize, Vec<(u64, u32)>, Vec<u64>) {
        rtape_reset();
        let n_params = crate::types::MAX_PK_PARAMS + 2;
        let params: Vec<R> = (0..n_params).map(|i| rinput(0.5 + i as f64)).collect();
        let mut outs = Vec::new();
        let (mut vars, mut stack) = (Vec::new(), Vec::new());
        for (k, t) in [0.5, 1.0, 2.0].into_iter().enumerate() {
            let u: Vec<R> = (0..3).map(|i| rinput(1.0 + (i + k) as f64)).collect();
            let mut du = vec![R::c(0.0); 3];
            prog.eval_rhs_g::<R>(
                &u,
                &params,
                t,
                t,
                R::c(t),
                R::c(0.0),
                &mut du,
                &mut vars,
                &mut stack,
            );
            outs.extend(du);
        }
        let len = rtape_len();
        let sum = outs.iter().fold(R::c(0.0), |a, &b| a + b);
        let adj = rsweep::<1>(&[sum]);
        (
            len,
            outs.iter().map(|o| (o.v.to_bits(), o.i)).collect(),
            adj.iter().map(|a| a.to_bits()).collect(),
        )
    };
    let want = record(&interpreted);
    let got = record(&production);
    assert_eq!(want.0, got.0, "tape length");
    assert_eq!(want.1, got.1, "output values and nodes");
    assert_eq!(want.2, got.2, "adjoints");
}

/// The reverse pass of one program over one layout, two ways: interpreted, and as
/// forward-mode tangents of the same scalar code (`ir::interpret_tangent`, one input
/// direction at a time), the exact oracle. Returns the worst relative difference against
/// the oracle.
fn check_reverse(
    reg: &RhsRegProgram,
    layout: super::ir::Layout,
    rng: &mut Rng,
    label: &str,
) -> f64 {
    use super::ir;
    let w = layout.lanes();
    let fwd = ir::forward(&reg.code, reg.n_regs, layout);
    let rev = ir::reverse(&fwd);
    let n_params = crate::types::MAX_PK_PARAMS + 2;
    let lanes_of = |rng: &mut Rng, n: usize, first: &mut dyn FnMut(&mut Rng) -> f64| -> Vec<f64> {
        (0..n * w)
            .map(|j| {
                if j % w == 0 {
                    first(rng)
                } else {
                    rng.unit() * 2.0 - 1.0
                }
            })
            .collect()
    };
    let u = lanes_of(rng, 3, &mut |r: &mut Rng| r.unit() * 4.0 + 0.1);
    let p = lanes_of(rng, n_params, &mut |r: &mut Rng| r.unit() * 3.0 + 0.2);
    let mut sd = lanes_of(rng, 4, &mut |r: &mut Rng| r.unit() * 2.0);
    sd[3 * w] = f64::EPSILON;
    let du_bar: Vec<f64> = (0..3 * w).map(|_| rng.unit() * 2.0 - 1.0).collect();
    // Interpreted reverse pass.
    let (mut ub, mut pb, mut sb) = (vec![0.0; u.len()], vec![0.0; p.len()], vec![0.0; sd.len()]);
    {
        let (mut u_, mut p_, mut s_, mut du_) =
            (u.clone(), p.clone(), sd.clone(), vec![0.0; 3 * w]);
        let mut db = du_bar.clone();
        ir::interpret(
            &rev,
            &mut [
                &mut u_, &mut p_, &mut s_, &mut du_, &mut ub, &mut pb, &mut sb, &mut db,
            ],
        );
    }
    let du = vec![0.0; 3 * w];
    // The oracle: one forward tangent per input location.
    let mut worst = 0.0f64;
    let groups: [(usize, &Vec<f64>, &Vec<f64>); 3] = [(0, &u, &ub), (1, &p, &pb), (2, &sd, &sb)];
    for (g, xs, bar) in groups {
        for j in 0..xs.len() {
            let mut d = vec![
                vec![0.0; u.len()],
                vec![0.0; p.len()],
                vec![0.0; sd.len()],
                vec![0.0; 3 * w],
            ];
            d[g][j] = 1.0;
            ir::interpret_tangent(&fwd, &[&u, &p, &sd, &du], &mut d);
            let want: f64 = d[3].iter().zip(&du_bar).map(|(a, b)| a * b).sum();
            let got = bar[j];
            // An overflowed value (an `exp` past `f64::MAX`) makes the forward tangent of an
            // input it does not read `0·∞ = NaN`, where the adjoint is an honest `0`. The
            // converse — a non-finite adjoint against a finite tangent, as a dead `if` arm
            // leaking into the reverse pass would give — is still a failure.
            if want.is_nan() {
                continue;
            }
            assert!(
                want.is_finite() == got.is_finite(),
                "{label}: input ({g}, {j}): reverse {got} vs forward {want}"
            );
            if want.is_finite() {
                let rel = (got - want).abs() / (1.0 + want.abs());
                worst = worst.max(rel);
            }
        }
    }
    worst
}

#[test]
fn the_reverse_pass_is_the_forward_tangents_transposed_on_random_programs() {
    use super::ir::Layout;
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let layouts = [
        Layout::F64,
        Layout {
            n: 2,
            m: 0,
            th: false,
            wide: false,
            tan: 0,
        },
        Layout {
            n: 8,
            m: 0,
            th: false,
            wide: true,
            tan: 0,
        },
        Layout {
            n: 2,
            m: 3,
            th: false,
            wide: false,
            tan: 0,
        },
        Layout {
            n: 2,
            m: 2,
            th: true,
            wide: false,
            tan: 0,
        },
    ];
    let mut worst = 0.0f64;
    for case in 0..60 {
        let odes = random_odes(&mut rng);
        let prog = program(&odes);
        let reg = RhsRegProgram::compile(&prog.stmts, &layout(&prog))
            .unwrap_or_else(|e| panic!("case {case} did not compile ({e:?}):\n{odes}"));
        for &l in &layouts {
            let label = format!("case {case} {l:?}:\n{odes}");
            worst = worst.max(check_reverse(&reg, l, &mut rng, &label));
        }
    }
    // Measured: the reverse accumulation order against the forward one, rounding only.
    assert!(worst < 1e-12, "worst relative difference {worst:e}");
    eprintln!("reverse pass vs forward tangents: worst relative difference {worst:e}");
}

/// Forward-over-reverse (`ir::tangent` of `ir::reverse`) is the Hessian of `ū·f` applied
/// to a direction. The oracle derives the same entries a second way, forward-over-forward
/// (`ir::tangent` twice, which never runs the reverse pass): `∂²(ū·f)/∂uᵢ∂u_j` from a pair
/// of one-hot seeds. A wrong reverse rule, a wrong tangent rule or a tangent of the
/// reverse's own masking (`Func::live`) shows as a mismatch; the transform is also checked
/// against `ir::interpret_tangent` direction by direction.
#[test]
fn forward_over_reverse_is_the_hessian_on_random_programs() {
    use super::ir::{self, Layout};
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut worst = 0.0f64;
    for case in 0..40 {
        let odes = random_odes(&mut rng);
        let prog = program(&odes);
        let reg = RhsRegProgram::compile(&prog.stmts, &layout(&prog))
            .unwrap_or_else(|e| panic!("case {case} did not compile ({e:?}):\n{odes}"));
        let l = Layout::F64;
        let fwd = ir::forward(&reg.code, reg.n_regs, l);
        let n_params = crate::types::MAX_PK_PARAMS + 2;
        let u: Vec<f64> = (0..3).map(|_| rng.unit() * 4.0 + 0.1).collect();
        let p: Vec<f64> = (0..n_params).map(|_| rng.unit() * 3.0 + 0.2).collect();
        let mut sd: Vec<f64> = (0..4).map(|_| rng.unit() * 2.0).collect();
        sd[3] = f64::EPSILON;
        let du_bar: Vec<f64> = (0..3).map(|_| rng.unit() * 2.0 - 1.0).collect();
        let zeros = |n: usize| vec![0.0; n];
        // `ir::tangent` against the tangent interpreter, two directions at once.
        let t2 = ir::tangent(&fwd, 2);
        let mut dirs: Vec<Vec<f64>> = [3, n_params, 4, 3].iter().map(|&n| zeros(2 * n)).collect();
        for (a, d) in dirs.iter_mut().enumerate().take(3) {
            for (i, x) in d.iter_mut().enumerate() {
                *x = [0.5, -1.0, 0.0, 0.25, 1.5][(i + 3 * a) % 5];
            }
        }
        {
            let (mut u_, mut p_, mut s_, mut du_) = (u.clone(), p.clone(), sd.clone(), zeros(3));
            let [d0, d1, d2, d3] = &mut dirs[..] else {
                unreachable!()
            };
            ir::interpret(
                &t2,
                &mut [&mut u_, &mut p_, &mut s_, &mut du_, d0, d1, d2, d3],
            );
        }
        for j in 0..2 {
            let mut d: Vec<Vec<f64>> = dirs
                .iter()
                .map(|v| v.iter().skip(j).step_by(2).copied().collect())
                .collect();
            d[3] = zeros(3);
            ir::interpret_tangent(&fwd, &[&u, &p, &sd, &zeros(3)], &mut d);
            for (o, want) in d[3].iter().enumerate() {
                let got = dirs[3][2 * o + j];
                if want.is_nan() {
                    continue;
                }
                let rel = (got - want).abs() / (1.0 + want.abs());
                assert!(
                    rel < 1e-12,
                    "case {case} dir {j} out {o}: {got} vs {want}\n{odes}"
                );
            }
        }
        // Forward-over-reverse: row `i` of `H = ∂²(ū·f)/∂u²`, the tangent of `ū` along `eᵢ`.
        let rev = ir::reverse(&fwd);
        let fr = ir::tangent(&rev, 1);
        let ff = ir::tangent(&ir::tangent(&fwd, 1), 1);
        for i in 0..3 {
            let mut bufs: Vec<Vec<f64>> = vec![
                u.clone(),
                p.clone(),
                sd.clone(),
                zeros(3),
                zeros(3),
                zeros(n_params),
                zeros(4),
                du_bar.clone(),
            ];
            let mut tans: Vec<Vec<f64>> = bufs.iter().map(|b| zeros(b.len())).collect();
            tans[0][i] = 1.0;
            {
                let mut a: Vec<&mut [f64]> = bufs
                    .iter_mut()
                    .chain(tans.iter_mut())
                    .map(|v| v.as_mut_slice())
                    .collect();
                ir::interpret(&fr, &mut a);
            }
            for j in 0..3 {
                let got = tans[4][j];
                // Forward-over-forward: seed `u_j` inside, `uᵢ` outside.
                let mut b2: Vec<Vec<f64>> = vec![u.clone(), p.clone(), sd.clone(), zeros(3)];
                let mut t_in: Vec<Vec<f64>> = vec![zeros(3), zeros(n_params), zeros(4), zeros(3)];
                t_in[0][j] = 1.0;
                b2.extend(t_in);
                let mut t_out: Vec<Vec<f64>> = b2.iter().map(|b| zeros(b.len())).collect();
                t_out[0][i] = 1.0;
                b2.extend(t_out);
                {
                    let mut a: Vec<&mut [f64]> = b2.iter_mut().map(|v| v.as_mut_slice()).collect();
                    ir::interpret(&ff, &mut a);
                }
                // Argument 8 + 7: the outer tangent of the inner tangent of `du`.
                let want: f64 = b2[15].iter().zip(&du_bar).map(|(a, b)| a * b).sum();
                if !want.is_finite() {
                    continue;
                }
                assert!(
                    got.is_finite(),
                    "case {case} H[{i}][{j}]: {got} vs {want}\n{odes}"
                );
                worst = worst.max((got - want).abs() / (1.0 + want.abs()));
            }
        }
    }
    assert!(worst < 1e-11, "worst relative difference {worst:e}");
    eprintln!("forward-over-reverse vs forward-over-forward: worst relative difference {worst:e}");
}

/// `columns` over native and lane evaluator: the same acceptance, the same bits.
fn columns_agree(
    prog: &OdeRhsProgram,
    u: &[f64],
    p: &[f64],
    cols: &[(usize, f64)],
    out_len: usize,
) -> (bool, Vec<f64>) {
    let columns = prog.columns.as_ref().expect("lane-evaluable");
    let mut want = vec![7.0; out_len];
    let mut got = vec![7.0; out_len];
    super::RHS_JIT_DISABLED.with(|c| c.set(true));
    let ok_want = columns(u, p, 0.5, cols, &mut want);
    super::RHS_JIT_DISABLED.with(|c| c.set(false));
    let ok_got = columns(u, p, 0.5, cols, &mut got);
    assert_eq!(ok_want, ok_got, "cols {cols:?}, out {out_len}");
    assert!(
        bits_eq(&want, &got),
        "lanes {want:?}
native {got:?}"
    );
    (ok_got, got)
}

#[test]
fn native_columns_decline_where_the_lane_evaluator_declines() {
    let prog = program(&chain_odes(3));
    assert!(prog.reg.as_ref().is_some_and(|r| r.has_native_f64()));
    let p = vec![0.7; crate::types::MAX_PK_PARAMS + 2];
    let u = [1.0, 2.0, 3.0];
    let cols = [(0, 1e-7), (2, 1e-6)];
    // The accepted case, as the control: both run, and agree.
    assert!(columns_agree(&prog, &u, &p, &cols, 6).0);
    // `out` shorter than the columns need.
    assert!(!columns_agree(&prog, &u, &p, &cols, 5).0);
    // A column past the states, though inside a longer `u`: the lane evaluator reads only
    // the first `state_count` states, and so does the native arm.
    let long = [1.0, 2.0, 3.0, 4.0, 5.0];
    assert!(!columns_agree(&prog, &long, &p, &[(3, 1e-7)], 3).0);
    // And a longer `u` with in-range columns is fine.
    assert!(columns_agree(&prog, &long, &p, &cols, 6).0);
}

#[test]
fn native_columns_handle_more_states_than_the_stack_buffer() {
    // 40 states: `u` outgrows the 32-element stack copy and takes the heap one.
    let n = 40;
    let prog = program_states(n, &chain_odes(n));
    assert!(prog.reg.as_ref().is_some_and(|r| r.has_native_f64()));
    let p = vec![0.7; crate::types::MAX_PK_PARAMS + 2];
    let u: Vec<f64> = (0..n).map(|i| 1.0 + i as f64 / 7.0).collect();
    let cols = [(0, 1e-7), (31, 2e-7), (32, -1e-7), (n - 1, 3e-7)];
    let (ok, out) = columns_agree(&prog, &u, &p, &cols, cols.len() * n);
    assert!(ok);
    assert!(out.iter().all(|x| x.is_finite() && *x != 7.0));
}

#[test]
fn a_large_native_frame_runs_and_matches_the_register_code() {
    // 120 states at 8 lanes keep ~7.7 KB of prologue loads live: a frame past one page, which
    // needs stack probes on Windows (`enable_probestack`). Run it, and compare bits.
    let n = 120;
    let prog = program_states(n, &chain_odes(n));
    let reg = prog.reg.as_ref().expect("translatable");
    let lift = |v: f64| DualF::<7, 0>::free(v, [0.25, -0.5, 0.0, 1.5, -0.0, 2.0, 0.125], []);
    let u: Vec<_> = (0..n).map(|i| lift(1.0 + i as f64 / 9.0)).collect();
    let p: Vec<_> = (0..crate::types::MAX_PK_PARAMS + 2)
        .map(|i| lift(0.5 + i as f64))
        .collect();
    let s = [lift(0.0), lift(0.0), lift(0.0), lift(f64::EPSILON)];
    assert!(reg.has_native::<DualF<7, 0>>());
    let mut want = vec![DualF::<7, 0>::constant(9.0); n];
    let mut got = want.clone();
    let mut regs = vec![DualF::<7, 0>::constant(0.0); reg.n_regs()];
    super::RHS_JIT_DISABLED.with(|c| c.set(true));
    reg.run(&u, &p, &s, &mut want, &mut regs);
    super::RHS_JIT_DISABLED.with(|c| c.set(false));
    reg.run(&u, &p, &s, &mut got, &mut regs);
    assert!(bits_eq(&want, &got));
}

#[test]
fn the_jit_switch_reads_the_environment_value() {
    use super::jit::enabled_by;
    assert!(enabled_by(None));
    assert!(enabled_by(Some("0")));
    assert!(!enabled_by(Some("1")));
}

/// `reg` at states `x[..3]` and parameters `A`, `B`, `C` = `x[3..]`, each input lifted to `T`
/// by `lift(value, index)`; every other parameter slot a constant.
fn rhs_at<T: VmNum>(
    prog: &OdeRhsProgram,
    reg: &RhsRegProgram,
    x: &[f64; 6],
    lift: &dyn Fn(f64, usize) -> T,
) -> Vec<T> {
    let u: Vec<T> = (0..3).map(|i| lift(x[i], i)).collect();
    let mut p = vec![T::from_f64(0.9); crate::types::MAX_PK_PARAMS + 2];
    for (k, &slot) in prog.indiv_to_params_slot.iter().take(3).enumerate() {
        p[slot] = lift(x[3 + k], 3 + k);
    }
    let z = T::from_f64(0.0);
    let s = [T::from_f64(1.75), z, z, T::from_f64(f64::EPSILON)];
    let mut du = vec![T::from_f64(9.0); 3];
    let mut regs = vec![z; reg.n_regs()];
    reg.run(&u, &p, &s, &mut du, &mut regs);
    du
}

/// A taped type records an `if`'s live arm and nothing of the other, exactly as the
/// interpreter does: the production program (compiled) against the same model parsed with
/// the compiler off, on calls that take each arm of both `if`s — the same tape length, output
/// nodes and adjoints, bit for bit. Recording both arms would lengthen the tape. The second
/// `if` is a right operand whose else arm ends in a literal, so on that path the literal fuses
/// with the `*` after the `if` and on the other the `*` runs unfused (`conditional`).
#[test]
fn an_if_records_only_its_live_arm_on_a_tape() {
    use crate::sens::rtape::{rinput, rsweep, rtape_len, rtape_reset, R};
    let odes = "  d/dt(x1) = if (x1 < A) x1 * B else exp(x1) * C
  d/dt(x2) = x1 * (if (x2 > 1) B else 2)
  d/dt(x3) = x2";
    let production = program(odes);
    let interpreted = {
        let _off = super::super::RhsRegcodeOff::enter();
        program(odes)
    };
    assert!(production.reg.is_some() && interpreted.reg.is_none());
    // `eval_rhs_g` would keep the hoisted interpreter for `R` otherwise.
    assert!(production.hoist.is_none(), "the fixture must lift nothing");
    // `A` is params slot `indiv_to_params_slot[0]`, between 0.5 and 11.5: `x1` = 0.1 and 50
    // straddle it, `x2` = 0.5 and 2 straddle 1.
    let points = [[0.1, 0.5], [50.0, 2.0], [0.1, 2.0], [50.0, 0.5]];
    let record = |prog: &OdeRhsProgram, at: [f64; 2]| -> (usize, Vec<(u64, u32)>, Vec<u64>) {
        rtape_reset();
        let n_params = crate::types::MAX_PK_PARAMS + 2;
        let params: Vec<R> = (0..n_params).map(|i| rinput(0.5 + i as f64)).collect();
        let u = [rinput(at[0]), rinput(at[1]), rinput(3.0)];
        let mut du = vec![R::c(0.0); 3];
        let (mut vars, mut stack) = (Vec::new(), Vec::new());
        prog.eval_rhs_g::<R>(
            &u,
            &params,
            1.0,
            1.0,
            R::c(1.0),
            R::c(0.0),
            &mut du,
            &mut vars,
            &mut stack,
        );
        let len = rtape_len();
        let sum = du.iter().fold(R::c(0.0), |a, &b| a + b);
        let adj = rsweep::<1>(&[sum]);
        (
            len,
            du.iter().map(|o| (o.v.to_bits(), o.i)).collect(),
            adj.iter().map(|a| a.to_bits()).collect(),
        )
    };
    let mut lens = Vec::new();
    for at in points {
        let want = record(&interpreted, at);
        let got = record(&production, at);
        assert_eq!(want.0, got.0, "{at:?}: tape length");
        assert_eq!(want.1, got.1, "{at:?}: output values and nodes");
        assert_eq!(want.2, got.2, "{at:?}: adjoints");
        lens.push(got.0);
    }
    // The arms differ in what they record, so the points do reach both arms.
    lens.sort_unstable();
    lens.dedup();
    assert!(
        lens.len() > 1,
        "every point recorded the same tape: {lens:?}"
    );
}

/// The native code computes both arms of an `if`, so a dead arm's `NaN` is in the forward
/// values; its derivatives must still reach no input. `(x1 − 2)^B` at `x1 = 1.5` is a negative
/// base under a differentiated exponent — `NaN` — and the condition kills it. Over `f64`, a
/// first-order and a second-order layout: every input adjoint of the compiled reverse pass is
/// finite, `B`'s (read only in the dead arm) is exactly zero, and the native reverse pass is
/// the interpreted one bit for bit. The same through a taped block (`DualF`'s whole-call
/// recording), whose reverse rule is this reverse pass.
#[test]
fn a_dead_arm_leaks_no_nan_into_the_reverse_pass() {
    use super::ir::{self, Layout};
    let odes = "  d/dt(x1) = if (x1 > 1) A * x1 else (x1 - 2) ^ B
  d/dt(x2) = -x2
  d/dt(x3) = 0";
    let prog = program(odes);
    let reg = prog.reg.as_ref().expect("an `if` compiles");
    let n_params = crate::types::MAX_PK_PARAMS + 2;
    let b_slot = prog.indiv_to_params_slot[1];
    // The fixture's dead arm is `NaN` at this point.
    let dead = program("  d/dt(x1) = (x1 - 2) ^ B\n  d/dt(x2) = -x2\n  d/dt(x3) = 0");
    let p0: Vec<f64> = (0..n_params).map(|i| 0.7 + 0.31 * i as f64).collect();
    let s0 = [1.75, 0.0, 0.0, f64::EPSILON];
    let mut du = vec![0.0; 3];
    let mut regs = vec![0.0; dead.reg.as_ref().unwrap().n_regs()];
    let lifted = |v: f64| <DualF<1, 0> as PkNum>::var(v, 0);
    let mut du_d = vec![DualF::<1, 0>::constant(0.0); 3];
    let mut regs_d = vec![DualF::<1, 0>::constant(0.0); regs.len()];
    let p_d: Vec<_> = p0.iter().map(|&v| lifted(v)).collect();
    dead.reg.as_ref().unwrap().run(
        &[lifted(1.5), lifted(1.0), lifted(1.0)],
        &p_d,
        &[lifted(1.75), lifted(0.0), lifted(0.0), lifted(f64::EPSILON)],
        &mut du_d,
        &mut regs_d,
    );
    dead.reg
        .as_ref()
        .unwrap()
        .run(&[1.5, 1.0, 1.0], &p0, &s0, &mut du, &mut regs);
    assert!(
        du[0].is_nan() && du_d[0].v.is_nan(),
        "the dead arm must be NaN"
    );

    let layouts = [
        Layout::F64,
        Layout {
            n: 2,
            m: 0,
            th: false,
            wide: false,
            tan: 0,
        },
        Layout {
            n: 2,
            m: 3,
            th: false,
            wide: false,
            tan: 0,
        },
    ];
    for layout in layouts {
        let w = layout.lanes();
        let spread = |vals: &[f64]| -> Vec<f64> {
            vals.iter()
                .enumerate()
                .flat_map(|(i, &v)| {
                    (0..w).map(move |j| {
                        if j == 0 {
                            v
                        } else {
                            0.25 + 0.5 * ((i + j) % 3) as f64
                        }
                    })
                })
                .collect()
        };
        let u = spread(&[1.5, 1.0, 1.0]);
        let p = spread(&p0);
        let sd = spread(&s0);
        let fwd = ir::forward(&reg.code, reg.n_regs, layout);
        let rev = ir::reverse(&fwd);
        let du_bar = vec![1.0; 3 * w];
        let (mut ub, mut pb, mut sb) =
            (vec![0.0; u.len()], vec![0.0; p.len()], vec![0.0; sd.len()]);
        {
            let (mut u_, mut p_, mut s_, mut du_) =
                (u.clone(), p.clone(), sd.clone(), vec![0.0; 3 * w]);
            let mut db = du_bar.clone();
            ir::interpret(
                &rev,
                &mut [
                    &mut u_, &mut p_, &mut s_, &mut du_, &mut ub, &mut pb, &mut sb, &mut db,
                ],
            );
        }
        let all = || ub.iter().chain(&pb).chain(&sb);
        assert!(
            all().all(|a| a.is_finite()),
            "{layout:?}: a non-finite adjoint"
        );
        assert!(
            pb[b_slot * w..(b_slot + 1) * w].iter().all(|&a| a == 0.0),
            "{layout:?}: `B` has a derivative through the dead arm"
        );
        // The live arm `A·x1` does reach `x1`: the masking keeps it.
        assert!(ub[0] != 0.0, "{layout:?}: the live arm lost its derivative");
    }
}

/// Exact derivatives across an `if` against central finite differences of the `f64` program,
/// on both sides of every condition, through the register code (`Dual2`, the full Hessian)
/// and the native code (`DualF`, the packed one). `C` is read only by the conditions: its
/// derivatives are exactly zero, as the branch-wise RHS has none (#2649 is the integration's
/// concern, not the RHS's), and the differences agree since no step crosses a switch.
#[test]
fn derivatives_across_an_if_match_finite_differences() {
    use crate::sens::dual2::Dual2;
    let odes = "  d/dt(x1) = if (x1 * C < A) B * x1 * x1 else exp(B) * x2
  d/dt(x2) = -x2 * (if (x1 > C && x2 <= 1.5) A else B * B)
  d/dt(x3) = min(x1, x2) * A + max(x3, 1) * B";
    let prog = program(odes);
    let reg = prog.reg.as_ref().expect("an `if` compiles");
    // `[x1, x2, x3, A, B, C]`, the two points straddling every condition.
    let points = [
        [0.4, 1.2, 0.5, 2.0, 0.7, 1.1],
        [3.0, 1.4, 1.6, 0.9, 0.6, 1.3],
    ];
    let f = |x: &[f64; 6]| rhs_at(&prog, reg, x, &|v, _| v);
    // Each condition takes each side across the two points.
    let sides = |x: &[f64; 6]| {
        [
            x[0] * x[5] < x[3],
            x[0] > x[5] && x[1] <= 1.5,
            x[0] < x[1],
            x[2] > 1.0,
        ]
    };
    let (s0, s1) = (sides(&points[0]), sides(&points[1]));
    assert!(s0.iter().zip(&s1).all(|(a, b)| a != b), "{s0:?} vs {s1:?}");
    let h = 1e-5;
    let h2 = 1e-4;
    let mut worst_g = 0.0f64;
    let mut worst_h = 0.0f64;
    for x in &points {
        let y = f(x);
        let x2 = rhs_at(&prog, reg, x, &|v, i| <Dual2<6> as PkNum>::var(v, i));
        let xf = rhs_at(&prog, reg, x, &|v, i| <DualF<6, 21> as PkNum>::var(v, i));
        let step = |d: &[(usize, f64)]| {
            let mut z = *x;
            for &(i, e) in d {
                z[i] += e;
            }
            f(&z)
        };
        for o in 0..3 {
            assert_eq!(x2[o].value.to_bits(), y[o].to_bits(), "value, Dual2");
            assert_eq!(xf[o].v.to_bits(), y[o].to_bits(), "value, DualF");
            for i in 0..6 {
                let fd = (step(&[(i, h)])[o] - step(&[(i, -h)])[o]) / (2.0 * h);
                for (got, which) in [(x2[o].grad[i], "Dual2"), (xf[o].g[i], "DualF")] {
                    assert!(got.is_finite() && fd.is_finite());
                    worst_g = worst_g.max((got - fd).abs() / (1.0 + fd.abs()));
                    if i == 5 {
                        assert_eq!(got, 0.0, "{which}: ∂/∂C at {x:?}");
                    }
                }
                for j in 0..=i {
                    let fd = (step(&[(i, h2), (j, h2)])[o]
                        - step(&[(i, h2), (j, -h2)])[o]
                        - step(&[(i, -h2), (j, h2)])[o]
                        + step(&[(i, -h2), (j, -h2)])[o])
                        / (4.0 * h2 * h2);
                    let packed = xf[o].h[i * (i + 1) / 2 + j];
                    for got in [x2[o].hess[i][j], packed] {
                        assert!(got.is_finite() && fd.is_finite());
                        worst_h = worst_h.max((got - fd).abs() / (1.0 + fd.abs()));
                        if i == 5 {
                            assert_eq!(got, 0.0, "∂²/∂C∂· at {x:?}");
                        }
                    }
                }
            }
        }
    }
    // Measured: gradient 1.1e-11 (central differences at h = 1e-5), Hessian 2.2e-8 (second
    // differences at h = 1e-4, rounding-bound). The bounds keep ~10× headroom.
    eprintln!("derivatives across an if vs FD: gradient {worst_g:e}, Hessian {worst_h:e}");
    assert!(worst_g < 1e-10, "gradient vs FD {worst_g:e}");
    assert!(worst_h < 2e-7, "Hessian vs FD {worst_h:e}");
}

/// The hoisted pair (`prep` once per parameter set, `main` per call; what the steppers call)
/// on `if`s, against the register code. The first `if` reads parameters alone — every
/// instruction of it would be hoisted but for the rule that keeps an `if` out of `prep` — and
/// the second reads a state, with a hoisted value (`exp(A)`) in an arm. Parameter sets and
/// states straddle both conditions, on one cache.
#[test]
fn the_hoisted_pair_runs_an_if_bit_for_bit() {
    let odes = "  d/dt(x1) = (if (A < B) A / B else B / A) * x1
  d/dt(x2) = if (x1 < 1) exp(A) * x2 else C * x2
  d/dt(x3) = x2";
    let prog = program(odes);
    let reg = prog.reg.as_ref().expect("an `if` compiles");
    let native = reg.native_f64().expect("a hoisted pair");
    let mut cache = super::HoistCache::default();
    let s = [1.75, 0.0, 0.0, f64::EPSILON];
    let (a, b) = (prog.indiv_to_params_slot[0], prog.indiv_to_params_slot[1]);
    for (pa, pb) in [(0.5, 2.0), (3.0, 0.25)] {
        let mut p = vec![0.9; crate::types::MAX_PK_PARAMS + 2];
        p[a] = pa;
        p[b] = pb;
        for x1 in [0.3, 4.0] {
            let u = [x1, 1.5, 2.0];
            let want = run_without_jit(reg, &u, &p, &s);
            let mut got = vec![9.0; 3];
            assert!(native.run(&u, &p, &s, &mut got, &mut cache));
            assert!(
                bits_eq(&want, &got),
                "A {pa}, B {pb}, x1 {x1}: {want:?} vs {got:?}"
            );
        }
    }
}

// ---- The RHS-route note (#2681) ---------------------------------------------------------

use super::super::{Bytecode, Op, Statement};
use super::{rhs_route_note, NativeCause, RhsConstruct, RhsRefusal, RhsRoute, RhsSite};

/// The note `program(odes)` reports, which must exist.
fn note(odes: &str) -> String {
    let prog = program(odes);
    assert!(prog.reg.is_none(), "{odes}");
    let (code, msg, _) = prog
        .route_note()
        .unwrap_or_else(|| panic!("no note: {odes}"));
    assert_eq!(code, "W_ODE_RHS_INTERPRETED");
    assert!(msg.starts_with("W_ODE_RHS_INTERPRETED: "), "{msg}");
    msg
}

const REST: &str = "\n  d/dt(x2) = -x2\n  d/dt(x3) = 0";

/// One refusal per construct the `[odes]` parser can write, each named in the note with
/// the statement it is in — and, in the same test, a compiled program, which has none.
#[test]
fn each_parsable_refusal_names_its_construct_and_statement() {
    for (odes, construct, site) in [
        // An inline `if` compiles (#2680), so what an arm cannot translate is named as
        // itself, not as the `if`.
        (
            "  d/dt(x1) = if (x1 < 1) sqrt(x1) else B",
            "`sqrt`",
            "`d/dt(x1)`",
        ),
        ("  d/dt(x1) = sqrt(x1)", "`sqrt`", "`d/dt(x1)`"),
        ("  d/dt(x1) = inv_logit(x1)", "`inv_logit`", "`d/dt(x1)`"),
        ("  d/dt(x1) = logit(x1)", "`logit`", "`d/dt(x1)`"),
        ("  d/dt(x1) = tanh(x1)", "`tanh`", "`d/dt(x1)`"),
        ("  d/dt(x1) = softplus(x1)", "`softplus`", "`d/dt(x1)`"),
        ("  d/dt(x1) = sin(x1)", "`sin`", "`d/dt(x1)`"),
        ("  d/dt(x1) = cos(x1)", "`cos`", "`d/dt(x1)`"),
        ("  d/dt(x1) = pnorm(x1)", "`pnorm`", "`d/dt(x1)`"),
        // The site is the assignment, not the derivative that reads it.
        (
            "  E = sqrt(A)\n  d/dt(x1) = -E * x1",
            "`sqrt`",
            "the assignment to `E`",
        ),
        // A later statement: the first refusal in program order is the one named.
        (
            "  d/dt(x1) = -x1\n  d/dt(x2) = logit(x2) + sqrt(x1)\n  d/dt(x3) = 0",
            "`logit`",
            "`d/dt(x2)`",
        ),
        (
            "  if (A > 1) {\n    d/dt(x1) = -x1\n  } else {\n    d/dt(x1) = x1\n  }",
            "an `if (…) { … }` statement",
            "the [odes] block",
        ),
    ] {
        let odes = if odes.contains("d/dt(x3)") {
            odes.to_string()
        } else {
            format!("{odes}{REST}")
        };
        let msg = note(&odes);
        assert!(
            msg.contains(&format!("because of {construct} in {site}.")),
            "{odes}\n{msg}"
        );
    }
    // The other side of the gate, an inline `if` included.
    for odes in ["  d/dt(x1) = -A * x1", "  d/dt(x1) = if (x1 < 1) A else B"] {
        let prog = program(&format!("{odes}{REST}"));
        assert!(prog.reg.is_some(), "{odes}");
        assert!(
            prog.route_note().is_none(),
            "{odes}: a compiled RHS with native code has no note"
        );
    }
}

/// The refusals no `[odes]` block the parser accepts can write (θ, η and covariates are
/// rejected as undefined names there), on hand-built bytecode: each names its construct.
#[test]
fn each_unparsable_refusal_names_its_construct() {
    let prog = program(&format!("  d/dt(x1) = -A * x1{REST}"));
    let lay = layout(&prog);
    let refusal = |ops: Vec<Op>, constants: Vec<f64>| -> RhsRefusal {
        let mut bc = Bytecode::new();
        bc.max_stack = ops.len() + 1;
        bc.ops = ops;
        bc.constants = constants;
        RhsRegProgram::compile(&[Statement::DiffEqBc(0, bc)], &lay).expect_err("declined")
    };
    let x1 = Op::PushVar(0);
    for (ops, construct) in [
        (vec![x1, Op::PushConst(0), Op::Mod], "`mod`"),
        (vec![Op::PushTheta(0)], "a read of a θ"),
        (vec![Op::PushThetaGather(0)], "a read of a θ level"),
        (vec![Op::PushEta(0)], "a read of an η"),
        (vec![Op::PushCov(0)], "a read of a covariate"),
        (vec![Op::PushMixNum], "a read of the mixture class `MIXNUM`"),
        (
            vec![Op::PushNnOutput(0, 0)],
            "a read of a neural-network output",
        ),
        (vec![Op::PushTime], "a read of the `TIME` built-in"),
        (
            vec![Op::PushVar(10_000)],
            "a read of a variable outside the RHS's slots",
        ),
        // An `if` outside the shape the parser emits: a stray `Jump`, and a `JumpIfFalse`
        // with no `Jump` before its target.
        (vec![x1, Op::Jump(2)], "an inline `if`"),
        (
            vec![x1, Op::PushConst(0), Op::CmpLt, Op::JumpIfFalse(9)],
            "an inline `if`",
        ),
        // Unbalanced: nothing left on the stack.
        (vec![], "a statement the RHS compiler does not translate"),
    ] {
        let r = refusal(ops.clone(), vec![2.0]);
        assert_eq!(r.site, RhsSite::Deriv(0), "{ops:?}");
        let named = r.named(|s| format!("v{s}"));
        assert_eq!(named.construct, construct, "{ops:?}");
        assert_eq!(named.site, "`d/dt(v0)`");
        let (_, msg, details) = rhs_route_note(RhsRoute::Interpreter(&named)).unwrap();
        assert!(
            msg.contains(&format!("because of {construct} in `d/dt(v0)`.")),
            "{msg}"
        );
        assert_eq!(details["construct"], construct);
    }
}

/// A local read before the program assigns it is declined in `lower`, and named by the
/// local's own name.
#[test]
fn a_local_read_before_its_assignment_is_named() {
    let msg = note(&format!("  d/dt(x1) = -L * x1\n  L = A{REST}"));
    assert!(
        msg.contains("because of a read of `L` before it is assigned in `d/dt(x1)`."),
        "{msg}"
    );
}

/// Each sentence of the interpreter note, deleted in turn, reddens one assertion here, on a
/// function and on an `if (…) { … }` statement alike.
#[test]
fn every_sentence_of_the_interpreter_note_is_pinned() {
    let block_msg = note(&format!(
        "  if (A > 1) {{\n    d/dt(x1) = -x1\n  }} else {{\n    d/dt(x1) = x1\n  }}{REST}"
    ));
    let sqrt_msg = note(&format!("  d/dt(x1) = sqrt(x1){REST}"));
    for msg in [&block_msg, &sqrt_msg] {
        assert!(
            msg.contains(
                "the [odes] right-hand side runs on the bytecode interpreter, not compiled \
                 code, because of "
            ),
            "{msg}"
        );
        assert!(
            msg.contains(
                "The constructs that keep the interpreter are listed in the ODE models \
                 page of the documentation, under \"Native right-hand side\"."
            ),
            "{msg}"
        );
    }
    // The note ends with the pointer to the docs: nothing follows it.
    for msg in [&block_msg, &sqrt_msg] {
        assert!(msg.ends_with("under \"Native right-hand side\"."), "{msg}");
    }
}

/// The register-code note: the cause decided in order (switch, host, program), one
/// sentence per cause, and the shared lead and closing sentences.
#[test]
fn every_native_cause_has_its_own_sentence() {
    assert_eq!(NativeCause::of(false, false), NativeCause::Disabled);
    assert_eq!(NativeCause::of(false, true), NativeCause::Disabled);
    assert_eq!(NativeCause::of(true, false), NativeCause::Host);
    assert_eq!(NativeCause::of(true, true), NativeCause::Declined);
    let causes = [
        (
            NativeCause::Disabled,
            "the environment variable `FERX_NO_RHS_JIT=1` switches native code off",
            "disabled",
        ),
        (
            NativeCause::Host,
            "Cranelift cannot generate code for this host",
            "host",
        ),
        (
            NativeCause::Declined,
            "the native compiler did not build this program (it builds programs up to a \
             size limit)",
            "declined",
        ),
    ];
    for (cause, why, token) in causes {
        let (code, msg, details) = rhs_route_note(RhsRoute::RegisterCode(cause)).unwrap();
        assert_eq!(code, "W_ODE_RHS_NOT_NATIVE");
        assert!(
            msg.starts_with(
                "W_ODE_RHS_NOT_NATIVE: the [odes] right-hand side runs as register code \
                 rather than native code, because "
            ),
            "{msg}"
        );
        assert!(msg.contains(&format!("because {why}.")), "{msg}");
        for (other, other_why, _) in causes {
            if other != cause {
                assert!(!msg.contains(other_why), "{msg}");
            }
        }
        assert!(
            msg.ends_with(" The results are identical; only the speed differs."),
            "{msg}"
        );
        assert_eq!(details["cause"], token);
        assert_eq!(details["route"], "register_code");
    }
    assert!(rhs_route_note(RhsRoute::Native).is_none());
}

/// A compiled program without its `f64` native function reports the register-code note;
/// with it, none — both sides on one program.
#[test]
fn a_compiled_rhs_without_native_code_reports_the_register_code() {
    let prog = program(&format!("  d/dt(x1) = -A * x1{REST}"));
    assert!(prog.route_note().is_none());
    let was = super::RHS_JIT_DISABLED.with(|c| c.replace(true));
    let note = prog.route_note();
    super::RHS_JIT_DISABLED.with(|c| c.set(was));
    let (code, _, details) = note.expect("no native code is reported");
    assert_eq!(code, "W_ODE_RHS_NOT_NATIVE");
    // The test switch is neither the environment variable nor the host.
    assert_eq!(details["cause"], "declined");
    assert!(matches!(
        RhsConstruct::Function("sqrt"),
        RhsConstruct::Function(_)
    ));
}
