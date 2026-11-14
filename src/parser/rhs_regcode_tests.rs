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
use crate::sens::vtape::DualV;

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
/// translates (and literals that hit the guards: a near-zero divisor, `ln` and `sqrt` of a
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
    match rng.below(16) {
        // An inline `if`, nested when an arm draws another, and the `min`/`max` that
        // desugar to one. An arm that ends in a leaf fuses with the op after the `if` on the
        // else path only (`x + (if (…) a else B)`), which `conditional` must reproduce.
        12 => format!("(if ({}) {a} else {b})", cond(rng, names, depth - 1)),
        13 => format!("min({a}, {b})"),
        14 => format!("max({a}, {b})"),
        15 => format!("sqrt({a})"),
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
        let lift_v = |x: &[f64]| -> Vec<DualV<2, 3>> {
            x.iter()
                .enumerate()
                .map(|(i, &v)| {
                    let a = DualV::<2, 3>::var(v, i % 2);
                    a * a * DualV::constant(0.25) + DualV::var(0.3, (i + 1) % 2)
                })
                .collect()
        };
        let (uv, pv, sv) = (lift_v(&raw_u), lift_v(&raw_p), lift_v(&raw_s));
        let sv: [DualV<2, 3>; 4] = [sv[0], sv[1], sv[2], sv[3]];
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
        "  d/dt(x1) = tanh(x1)\n  d/dt(x2) = -x2\n  d/dt(x3) = 0",
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
    // `sqrt` translates (#2844).
    let prog = program("  d/dt(x1) = sqrt(x1)\n  d/dt(x2) = -x2\n  d/dt(x3) = 0");
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
        // Second order: the packed η-Hessian payload, contiguous and through `DualV`'s
        // lanes, and the directional payload.
        check_native_lifted::<DualF<2, 3>>(&reg, raw, &label);
        check_native_lifted::<DualF<4, 10>>(&reg, raw, &label);
        check_native_lifted::<DualV<3, 6>>(&reg, raw, &label);
        check_native_lifted::<DualV<8, 36>>(&reg, raw, &label);
        check_native_lifted::<DualV<2, 2, true>>(&reg, raw, &label);
    }
}

#[test]
fn a_taped_or_too_wide_dual_keeps_the_register_code() {
    let prog = program("  d/dt(x1) = -A * x1 + 2\n  d/dt(x2) = A * x1 - B * x2\n  d/dt(x3) = C");
    let reg = prog.reg.as_ref().expect("translatable");
    // A payload past the lane cap has no native function.
    assert!(!reg.has_native::<DualV<24, 300>>());
    assert!(reg.has_native::<DualV<4, 10>>());
    // A taped input keeps the register code (its operations must reach the tape).
    crate::sens::vtape::vtape_reset(3);
    let leaf = DualV::<2, 3>::leaf(1.5, [1.0, 0.0], [0.0; 3]);
    let lift = |v: f64| DualV::<2, 3>::free(v, [0.25, -0.5], [0.0; 3]);
    let u = [leaf, lift(2.0), lift(3.0)];
    let p: Vec<_> = (0..crate::types::MAX_PK_PARAMS + 2)
        .map(|i| lift(0.5 + i as f64))
        .collect();
    let s = [lift(0.0), lift(0.0), lift(0.0), lift(f64::EPSILON)];
    let mut du = vec![DualV::<2, 3>::constant(0.0); 3];
    let mut regs = vec![DualV::<2, 3>::constant(0.0); reg.n_regs()];
    reg.run(&u, &p, &s, &mut du, &mut regs);
    assert!(
        <DualV<2, 3> as PkNum>::is_taped(du[0]),
        "a taped state must produce a taped derivative"
    );
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

/// The reverse pass of one program over one layout, three ways: interpreted, native, and
/// as forward-mode tangents of the same scalar code (`ir::interpret_tangent`, one input
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
    // Native reverse pass: the lowering is bit for bit the interpreted one.
    let jit = reg.jit.as_ref().expect("native code");
    let vjp = jit.resolve_vjp(layout).expect("a reverse pass");
    let (mut ub2, mut pb2, mut sb2) = (vec![0.0; u.len()], vec![0.0; p.len()], vec![0.0; sd.len()]);
    let mut du = vec![0.0; 3 * w];
    // SAFETY: every buffer covers the program's extent at `w` lanes.
    unsafe {
        vjp.vjp(
            u.as_ptr(),
            p.as_ptr(),
            sd.as_ptr(),
            du.as_mut_ptr(),
            ub2.as_mut_ptr(),
            pb2.as_mut_ptr(),
            sb2.as_mut_ptr(),
            du_bar.as_ptr(),
        )
    };
    for (a, b) in ub
        .iter()
        .chain(&pb)
        .chain(&sb)
        .zip(ub2.iter().chain(&pb2).chain(&sb2))
    {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{label}: native reverse pass differs"
        );
    }
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
        },
        Layout {
            n: 8,
            m: 0,
            th: false,
            wide: true,
        },
        Layout {
            n: 2,
            m: 3,
            th: false,
            wide: false,
        },
        Layout {
            n: 2,
            m: 2,
            th: true,
            wide: false,
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

/// One taped recording of `reg` over `DualV<N, M, TH>`, swept from fixed output adjoints:
/// the adjoints of every leaf's every component. `block` records the call whole (one
/// block, reverse pass compiled); otherwise every operation is a tape node.
fn taped_leaf_adjoints<const N: usize, const M: usize, const TH: bool>(
    reg: &RhsRegProgram,
    raw: (&[f64], &[f64], &[f64; 4]),
    out_bar: &[f64],
    block: bool,
) -> (Vec<f64>, Vec<DualV<N, M, TH>>) {
    use crate::sens::vtape::{
        vadj_index, vadj_index_h, vadj_init, vsweep, vsweep_h, vtape_len, vtape_reset,
        RHS_BLOCK_OFF,
    };
    let nc = if TH { 1 + N + M } else { 1 + N };
    RHS_BLOCK_OFF.with(|c| c.set(!block));
    vtape_reset(nc);
    let tan = |i: usize| -> ([f64; N], [f64; M]) {
        let mut g = [0.0; N];
        let mut h = [0.0; M];
        for (k, x) in g.iter_mut().enumerate() {
            *x = [0.5, -1.25, 0.0, 0.375][(i + k) % 4];
        }
        for (k, x) in h.iter_mut().enumerate() {
            *x = [0.25, 0.0, -0.5][(i + k) % 3];
        }
        (g, h)
    };
    // Every parameter and two of the states are tape leaves; the third state and the seeds
    // are free (untaped inputs with tangents), as a recording's are.
    let mut leaves = Vec::new();
    let mut mk = |v: f64, i: usize, leaf: bool| {
        let (g, h) = tan(i);
        if leaf {
            let d = DualV::<N, M, TH>::leaf(v, g, h);
            leaves.push(d.i);
            d
        } else {
            DualV::<N, M, TH>::free(v, g, h)
        }
    };
    let u: Vec<_> = raw
        .0
        .iter()
        .enumerate()
        .map(|(i, &v)| mk(v, i, i != 1))
        .collect();
    let p: Vec<_> = raw
        .1
        .iter()
        .enumerate()
        .map(|(i, &v)| mk(v, 10 + i, true))
        .collect();
    let s: Vec<_> = raw
        .2
        .iter()
        .enumerate()
        .map(|(i, &v)| mk(v, 50 + i, false))
        .collect();
    let s: [DualV<N, M, TH>; 4] = [s[0], s[1], s[2], s[3]];
    let mut du = vec![DualV::<N, M, TH>::constant(0.0); 3];
    let mut regs = vec![DualV::<N, M, TH>::constant(0.0); reg.n_regs()];
    reg.run(&u, &p, &s, &mut du, &mut regs);
    RHS_BLOCK_OFF.with(|c| c.set(false));
    let adj = vadj_init::<1>();
    for (o, d) in du.iter().enumerate() {
        if <DualV<N, M, TH> as PkNum>::is_taped(*d) {
            for c in 0..nc {
                let k = if TH {
                    vadj_index_h::<N, M>(d.i, c)
                } else {
                    vadj_index::<N, 1>(d.i, c, 0)
                };
                // `+=`: two outputs can be one node (`abs(A)` returns `A`'s own node, so
                // `d/dt(x1) = abs(A)` and `d/dt(x3) = A` share it on the per-operation
                // tape), and its adjoint is the sum of both seeds.
                adj[k] += out_bar[o * nc + c];
            }
        }
    }
    let top = vtape_len();
    let a = if TH {
        vsweep_h::<N, M>(top)
    } else {
        vsweep::<N, 1>(top)
    };
    let mut out = Vec::new();
    for &id in &leaves {
        for c in 0..nc {
            out.push(if TH {
                a[vadj_index_h::<N, M>(id, c)]
            } else {
                a[vadj_index::<N, 1>(id, c, 0)]
            });
        }
    }
    (out, du)
}

fn check_taped_block<const N: usize, const M: usize, const TH: bool>(
    reg: &RhsRegProgram,
    raw: (&[f64], &[f64], &[f64; 4]),
    rng: &mut Rng,
    label: &str,
) -> f64 {
    let nc = if TH { 1 + N + M } else { 1 + N };
    let out_bar: Vec<f64> = (0..3 * nc).map(|_| rng.unit() * 2.0 - 1.0).collect();
    let (want, du_op) = taped_leaf_adjoints::<N, M, TH>(reg, raw, &out_bar, false);
    let (got, du_blk) = taped_leaf_adjoints::<N, M, TH>(reg, raw, &out_bar, true);
    // The values (and the untaped payload) are the register code's bit for bit.
    for (a, b) in du_op.iter().zip(&du_blk) {
        assert_eq!(a.v.to_bits(), b.v.to_bits(), "{label}: value differs");
        assert!(
            a.g.iter()
                .zip(&b.g)
                .all(|(x, y)| x.to_bits() == y.to_bits())
                && a.h
                    .iter()
                    .zip(&b.h)
                    .all(|(x, y)| x.to_bits() == y.to_bits()),
            "{label}: jets differ"
        );
    }
    let mut worst = 0.0f64;
    for (k, (w, g)) in want.iter().zip(&got).enumerate() {
        if w.is_nan() && g.is_nan() {
            continue;
        }
        assert!(
            w.is_finite() && g.is_finite(),
            "{label}: adjoint {k}: block {g} vs per-operation {w}"
        );
        worst = worst.max((g - w).abs() / (1.0 + w.abs()));
    }
    worst
}

#[test]
fn a_taped_rhs_block_is_the_per_operation_tape_to_rounding() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let n_params = crate::types::MAX_PK_PARAMS + 2;
    let mut worst = 0.0f64;
    let mut cases = 0;
    for case in 0..80 {
        let odes = random_odes(&mut rng);
        let prog = program(&odes);
        let reg = RhsRegProgram::compile(&prog.stmts, &layout(&prog))
            .unwrap_or_else(|e| panic!("case {case} did not compile ({e:?}):\n{odes}"));
        // Inputs away from the guards: the two tapes take the same branch everywhere.
        let raw_u: Vec<f64> = (0..3).map(|_| rng.unit() * 3.0 + 0.2).collect();
        let raw_p: Vec<f64> = (0..n_params).map(|_| rng.unit() * 3.0 + 0.2).collect();
        let raw_s = [
            1.75,
            rng.unit() * 5.0 + 0.1,
            rng.unit() * 2.0 + 0.1,
            f64::EPSILON,
        ];
        let raw = (&raw_u[..], &raw_p[..], &raw_s);
        let _time = ModelTimeGuard::enter(1.75);
        let label = format!("case {case}:\n{odes}");
        worst = worst.max(check_taped_block::<2, 0, false>(
            &reg, raw, &mut rng, &label,
        ));
        worst = worst.max(check_taped_block::<3, 6, false>(
            &reg, raw, &mut rng, &label,
        ));
        worst = worst.max(check_taped_block::<8, 0, false>(
            &reg, raw, &mut rng, &label,
        ));
        worst = worst.max(check_taped_block::<2, 2, true>(&reg, raw, &mut rng, &label));
        cases += 1;
    }
    assert_eq!(cases, 80);
    eprintln!("taped RHS block vs per-operation tape: worst relative difference {worst:e}");
    assert!(worst < 1e-12, "worst relative difference {worst:e}");
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
/// the interpreted one bit for bit. The same through a taped block (`DualV`'s whole-call
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
        },
        Layout {
            n: 2,
            m: 3,
            th: false,
            wide: false,
        },
    ];
    let jit = reg.jit.as_ref().expect("native code");
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
        let vjp = jit.resolve_vjp(layout).expect("a reverse pass");
        let (mut ub2, mut pb2, mut sb2) =
            (vec![0.0; u.len()], vec![0.0; p.len()], vec![0.0; sd.len()]);
        let mut du2 = vec![0.0; 3 * w];
        // SAFETY: every buffer covers the program's extent at `w` lanes.
        unsafe {
            vjp.vjp(
                u.as_ptr(),
                p.as_ptr(),
                sd.as_ptr(),
                du2.as_mut_ptr(),
                ub2.as_mut_ptr(),
                pb2.as_mut_ptr(),
                sb2.as_mut_ptr(),
                du_bar.as_ptr(),
            )
        };
        for (a, b) in all().zip(ub2.iter().chain(&pb2).chain(&sb2)) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{layout:?}: native reverse pass differs"
            );
        }
    }
    // A taped block: every leaf adjoint finite.
    let mut rng = Rng(0x5851_F42D_4C95_7F2D);
    let raw = (&[1.5, 1.0, 1.0][..], &p0[..], &s0);
    let _time = ModelTimeGuard::enter(1.75);
    let w2 = check_taped_block::<2, 0, false>(reg, raw, &mut rng, "dead arm, DualV<2,0>");
    let w3 = check_taped_block::<3, 6, false>(reg, raw, &mut rng, "dead arm, DualV<3,6>");
    assert!(
        w2 < 1e-12 && w3 < 1e-12,
        "block vs per-operation: {w2:e}, {w3:e}"
    );
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
            "  d/dt(x1) = if (x1 < 1) tanh(x1) else B",
            "`tanh`",
            "`d/dt(x1)`",
        ),
        ("  d/dt(x1) = inv_logit(x1)", "`inv_logit`", "`d/dt(x1)`"),
        ("  d/dt(x1) = logit(x1)", "`logit`", "`d/dt(x1)`"),
        ("  d/dt(x1) = tanh(x1)", "`tanh`", "`d/dt(x1)`"),
        ("  d/dt(x1) = softplus(x1)", "`softplus`", "`d/dt(x1)`"),
        ("  d/dt(x1) = sin(x1)", "`sin`", "`d/dt(x1)`"),
        ("  d/dt(x1) = cos(x1)", "`cos`", "`d/dt(x1)`"),
        ("  d/dt(x1) = pnorm(x1)", "`pnorm`", "`d/dt(x1)`"),
        // The site is the assignment, not the derivative that reads it.
        (
            "  E = tanh(A)\n  d/dt(x1) = -E * x1",
            "`tanh`",
            "the assignment to `E`",
        ),
        // A later statement: the first refusal in program order is the one named.
        (
            "  d/dt(x1) = -x1\n  d/dt(x2) = logit(x2) + sin(x1)\n  d/dt(x3) = 0",
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
    // `sqrt` among them (#2844), so its removal from the refusals is pinned here too.
    for odes in [
        "  d/dt(x1) = -A * x1",
        "  d/dt(x1) = if (x1 < 1) A else B",
        "  d/dt(x1) = sqrt(x1)",
        "  E = sqrt(A)\n  d/dt(x1) = -E * x1",
    ] {
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
    let tanh_msg = note(&format!("  d/dt(x1) = tanh(x1){REST}"));
    for msg in [&block_msg, &tanh_msg] {
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
    for msg in [&block_msg, &tanh_msg] {
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
        RhsConstruct::Function("tanh"),
        RhsConstruct::Function(_)
    ));
}

/// `structural_zeros` on `reg`'s own code, inputs marked by `zero` (states, then params,
/// then seeds).
fn zeros_of(reg: &RhsRegProgram, raw: (&[f64], &[f64], &[f64; 4]), zero: &[bool]) -> Vec<bool> {
    let (nu, np) = (raw.0.len(), raw.1.len());
    let zs = [
        zero[nu + np],
        zero[nu + np + 1],
        zero[nu + np + 2],
        zero[nu + np + 3],
    ];
    let mut out = vec![true; 3];
    super::structural_zeros(
        &reg.code,
        reg.n_regs,
        raw.0,
        raw.1,
        raw.2,
        &zero[..nu],
        &zero[nu..nu + np],
        &zs,
        &mut out,
    );
    out
}

/// The program on `Dual2<3>`: a structural-zero input is the constant `0`, every other one
/// moves along its own random combination of the three directions, so a derivative that is
/// not zero shows in the jets (no two inputs share a direction to cancel in).
fn dual2_outputs(
    reg: &RhsRegProgram,
    raw: (&[f64], &[f64], &[f64; 4]),
    zero: &[bool],
    rng: &mut Rng,
) -> Vec<crate::sens::dual2::Dual2<3>> {
    use crate::sens::dual2::Dual2;
    let mut k = 0;
    let mut lift = |v: f64| -> Dual2<3> {
        let z = zero[k];
        k += 1;
        if z {
            return Dual2::<3>::from_f64(0.0);
        }
        let mut x = Dual2::<3>::from_f64(v);
        for d in 0..3 {
            let c = rng.unit() * 2.0 - 1.0;
            x = x + <Dual2<3> as PkNum>::var(0.0, d) * Dual2::<3>::from_f64(c);
        }
        x
    };
    let u: Vec<_> = raw.0.iter().map(|&v| lift(v)).collect();
    let p: Vec<_> = raw.1.iter().map(|&v| lift(v)).collect();
    let s: Vec<_> = raw.2.iter().map(|&v| lift(v)).collect();
    let s = [s[0], s[1], s[2], s[3]];
    let mut du = vec![Dual2::<3>::from_f64(9.0); 3];
    let mut regs = vec![Dual2::<3>::from_f64(0.0); reg.n_regs()];
    reg.run(&u, &p, &s, &mut du, &mut regs);
    du
}

/// #2760: every derivative `structural_zeros` reports is identically zero — value `0`,
/// every first- and second-order derivative in every direction `0` — on random programs
/// over every operation the compiler translates, with inputs that are structural zeros,
/// live zeros (`0` with a derivative), tiny values and values that hit the guards. A
/// report that is wrong is a derivative the tape silently drops, so this is the test that
/// the analysis never over-reports.
#[test]
fn a_reported_structural_zero_is_identically_zero_on_random_programs() {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let n_params = crate::types::MAX_PK_PARAMS + 2;
    let (mut reported, mut zero_valued) = (0usize, 0usize);
    for case in 0..800 {
        let odes = random_odes(&mut rng);
        let prog = program(&odes);
        let reg = RhsRegProgram::compile(&prog.stmts, &layout(&prog))
            .unwrap_or_else(|e| panic!("case {case} did not compile ({e:?}):\n{odes}"));
        let raw_u = seed_values(&mut rng, 3);
        let raw_p = seed_values(&mut rng, n_params);
        let raw_s = [1.75, rng.unit() * 5.0, 0.0, f64::EPSILON];
        // An input that is exactly 0 is a structural zero three times in four, otherwise a
        // live zero, which must never be reported through.
        let zero: Vec<bool> = raw_u
            .iter()
            .chain(&raw_p)
            .chain(&raw_s)
            .map(|&v| v == 0.0 && rng.below(4) != 0)
            .collect();
        let raw = (&raw_u[..], &raw_p[..], &raw_s);
        let _time = ModelTimeGuard::enter(1.75);
        let dead = zeros_of(&reg, raw, &zero);
        let du = dual2_outputs(&reg, raw, &zero, &mut rng);
        for (o, (&d, x)) in dead.iter().zip(&du).enumerate() {
            zero_valued += usize::from(x.val() == 0.0);
            if !d {
                continue;
            }
            reported += 1;
            assert!(
                x.val() == 0.0 && x.jets_finite() && x.jet_max_abs() == 0.0,
                "case {case}: du[{o}] reported a structural zero but is {x:?}\n{odes}"
            );
        }
    }
    // The test checks nothing unless the analysis reports, and it reports a fraction of the
    // zero-valued outputs (it may under-report). Measured: 187 of 472 over 800 cases. At 400
    // it was 110 of 227 until the generator drew `sqrt` (#2844), which the analysis never
    // reports through, and then 82 of 221.
    eprintln!("{reported} reported of {zero_valued} zero-valued outputs");
    assert!(reported >= 100, "only {reported} outputs reported");
}

/// Every instruction kind [`super::makes_structural_zeros`] names is one the analysis marks a
/// zero from unflagged inputs, and the predicate names it (#2761 review). The random
/// programs cannot show this for comparisons and connectives, whose results only feed a
/// branch there, so each kind gets a two-or-three-instruction program whose one output is
/// that instruction's zero: params `[1, 2, 0, NaN]`, none flagged (the `0` a live zero). The
/// control, a sum of two live inputs, is the straddle: marks nothing, names nothing.
#[test]
fn each_zero_source_kind_marks_from_unflagged_inputs_and_is_named() {
    use super::Cc;
    use super::Flat;
    let (one, two, zero, nan) = (0u32, 1u32, 2u32, 3u32);
    let p = |d: u16, slot: u32| Flat::LoadParam { d, slot };
    let out = |a: u16| Flat::Deriv { state: 0, a };
    let cases: Vec<(&str, Vec<Flat>, bool)> = vec![
        (
            "literal 0",
            vec![Flat::LoadK { d: 0, c: 0.0 }, out(0)],
            true,
        ),
        (
            "product with 0",
            vec![p(0, one), Flat::MulK { d: 1, a: 0, c: 0.0 }, out(1)],
            true,
        ),
        (
            "comparison",
            vec![
                p(0, one),
                p(1, two),
                Flat::Cmp {
                    d: 2,
                    cc: Cc::Gt,
                    a: 0,
                    b: 1,
                },
                out(2),
            ],
            true,
        ),
        (
            "and",
            vec![
                p(0, one),
                p(1, zero),
                Flat::And { d: 2, a: 0, b: 1 },
                out(2),
            ],
            true,
        ),
        (
            "or",
            vec![p(0, zero), Flat::Or { d: 1, a: 0, b: 0 }, out(1)],
            true,
        ),
        (
            "not",
            vec![p(0, one), Flat::Not { d: 1, a: 0 }, out(1)],
            true,
        ),
        (
            "present",
            vec![p(0, nan), Flat::Present { d: 1, a: 0 }, out(1)],
            true,
        ),
        (
            "control: a sum",
            vec![p(0, one), p(1, two), Flat::Add { d: 2, a: 0, b: 1 }, out(2)],
            false,
        ),
    ];
    let params = [1.0, 2.0, 0.0, f64::NAN];
    for (name, code, source) in cases {
        let mut marked = [false];
        super::structural_zeros(
            &code,
            4,
            &[],
            &params,
            &[0.5; 4],
            &[],
            &[false; 4],
            &[false; 4],
            &mut marked,
        );
        assert_eq!(marked[0], source, "{name}: the analysis marks its output");
        assert_eq!(
            super::makes_structural_zeros(&code),
            source,
            "{name}: named as a source"
        );
    }
}

/// The skip `vtape::record_rhs` takes (#2761 review): for a program without an instruction
/// [`super::makes_structural_zeros`] names, inputs whose *loaded* ones
/// ([`super::loaded_inputs`]) are none of them structural zeros leave every output live, so
/// not running the analysis changes nothing. On the soundness test's random programs, with
/// zero-valued inputs flagged at random as there: a flag on a `params` slot the program does
/// not load must mark nothing (production `params` hold such slots, as untaped zeros). Two
/// straddles keep the predicate honest: programs that make zeros must reach a marked output
/// from unflagged inputs, and skippable cases must include ones with an unloaded input
/// flagged, or the loaded-only rule is never exercised.
#[test]
fn a_program_that_makes_no_zeros_marks_nothing_from_unflagged_loaded_inputs() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let n_params = crate::types::MAX_PK_PARAMS + 2;
    let (mut skippable, mut with_unloaded_flag, mut marked_by_a_source) = (0usize, 0usize, 0usize);
    for case in 0..400 {
        let odes = random_odes(&mut rng);
        let prog = program(&odes);
        let reg = RhsRegProgram::compile(&prog.stmts, &layout(&prog))
            .unwrap_or_else(|e| panic!("case {case} did not compile ({e:?}):\n{odes}"));
        let raw_u = seed_values(&mut rng, 3);
        let raw_p = seed_values(&mut rng, n_params);
        let raw_s = [1.75, rng.unit() * 5.0, 0.0, f64::EPSILON];
        let (nu, np) = (raw_u.len(), raw_p.len());
        let zero: Vec<bool> = raw_u
            .iter()
            .chain(&raw_p)
            .chain(&raw_s)
            .map(|&v| v == 0.0 && rng.below(2) == 0)
            .collect();
        let (lu, lp, ls) = super::loaded_inputs(&reg.code);
        let loaded_flagged = lu.iter().any(|&i| zero[i as usize])
            || lp.iter().any(|&k| zero[nu + k as usize])
            || (0..4).any(|k| ls[k] && zero[nu + np + k]);
        let _time = ModelTimeGuard::enter(1.75);
        let dead = zeros_of(&reg, (&raw_u[..], &raw_p[..], &raw_s), &zero);
        if super::makes_structural_zeros(&reg.code) {
            let none = vec![false; zero.len()];
            let dead_unflagged = zeros_of(&reg, (&raw_u[..], &raw_p[..], &raw_s), &none);
            marked_by_a_source += usize::from(dead_unflagged.iter().any(|&d| d));
        } else if !loaded_flagged {
            skippable += 1;
            with_unloaded_flag += usize::from(zero.iter().any(|&z| z));
            assert!(
                dead.iter().all(|&d| !d),
                "case {case}: no zero source and no loaded input flagged, but marked {dead:?}\n{odes}"
            );
        }
    }
    // Measured (Linux x86_64): 32 skippable, all 32 with an unloaded input flagged, 19
    // zero-source programs marked. All three arms must be reached for the test to say
    // anything; the floors sit well under the measurement.
    eprintln!(
        "{skippable} skippable ({with_unloaded_flag} with an unloaded input flagged), \
         {marked_by_a_source} zero-source programs marked"
    );
    assert!(
        skippable >= 15 && with_unloaded_flag >= 10 && marked_by_a_source >= 10,
        "{skippable}, {with_unloaded_flag}, {marked_by_a_source}"
    );
}

/// #2760: the analysis on the shape it exists for, and its guards. A gated term `G·k·x`
/// is a structural zero exactly when the gate is, and a structural zero against a value
/// that is not finite (`0·∞`, `0/0`) is not one, since IEEE arithmetic makes it `NaN`.
#[test]
fn a_gated_term_is_a_structural_zero_exactly_when_its_gate_is() {
    // x1..x3 are the states, A, B, C the individual parameters.
    let prog = program("  d/dt(x1) = C * A * x2\n  d/dt(x2) = C / B - A * x2\n  d/dt(x3) = C / C");
    let reg = RhsRegProgram::compile(&prog.stmts, &layout(&prog)).expect("compiles");
    let n_params = crate::types::MAX_PK_PARAMS + 2;
    let s = [1.75, 1.0, 0.5, f64::EPSILON];
    let mut p = vec![0.7; n_params];
    let mut zero = vec![false; 3 + n_params + 4];
    // C is the third individual parameter; the binding reads it from its params slot.
    let c = prog.indiv_to_params_slot[2];
    let _time = ModelTimeGuard::enter(1.75);
    // C = 0, a structural zero: x1' and x2's C/B term are dead; x2' still has −A·x2; C/C is
    // 0/0.
    p[c] = 0.0;
    zero[3 + c] = true;
    let u = [1.0, 2.0, 3.0];
    let d = zeros_of(&reg, (&u, &p, &s), &zero);
    assert_eq!(d, [true, false, false], "gate off");
    // The same zero with a derivative (a live 0): nothing is dead.
    zero[3 + c] = false;
    assert_eq!(
        zeros_of(&reg, (&u, &p, &s), &zero),
        [false; 3],
        "a live zero"
    );
    // The gate on: nothing is dead.
    p[c] = 1.0;
    assert_eq!(zeros_of(&reg, (&u, &p, &s), &zero), [false; 3], "gate on");
    // Gate off against an infinite state: `0·∞` is `NaN`, not a zero.
    p[c] = 0.0;
    zero[3 + c] = true;
    let u_inf = [1.0, f64::INFINITY, 3.0];
    assert_eq!(zeros_of(&reg, (&u_inf, &p, &s), &zero)[0], false, "0·inf");
}

/// One taped block recording of `reg` over `DualV<N, M, TH>` in which the inputs `zero`
/// marks are structural zeros (the constant `0`) and every other parameter and two of the
/// states are tape leaves, swept from fixed adjoints on every taped output. `keep_dead`
/// keeps the structural-zero outputs on the tape (`RHS_DEAD_KEPT`). Returns every leaf's
/// adjoint components, the outputs and the tape's length.
fn masked_block_adjoints<const N: usize, const M: usize, const TH: bool>(
    reg: &RhsRegProgram,
    raw: (&[f64], &[f64], &[f64; 4]),
    zero: &[bool],
    out_bar: &[f64],
    keep_dead: bool,
) -> (Vec<f64>, Vec<DualV<N, M, TH>>, usize) {
    use crate::sens::vtape::{
        vadj_index, vadj_index_h, vadj_init, vsweep, vsweep_h, vtape_len, vtape_reset,
        RHS_DEAD_KEPT,
    };
    let nc = if TH { 1 + N + M } else { 1 + N };
    RHS_DEAD_KEPT.with(|c| c.set(keep_dead));
    vtape_reset(nc);
    let mut leaves = Vec::new();
    let mut k = 0;
    let mut mk = |v: f64, leaf: bool| {
        let z = zero[k];
        k += 1;
        if z {
            return DualV::<N, M, TH>::constant(0.0);
        }
        let mut g = [0.0; N];
        let mut h = [0.0; M];
        for (j, x) in g.iter_mut().enumerate() {
            *x = [0.5, -1.25, 0.0, 0.375][(k + j) % 4];
        }
        for (j, x) in h.iter_mut().enumerate() {
            *x = [0.25, 0.0, -0.5][(k + j) % 3];
        }
        if leaf {
            let d = DualV::<N, M, TH>::leaf(v, g, h);
            leaves.push(d.i);
            d
        } else {
            DualV::<N, M, TH>::free(v, g, h)
        }
    };
    let u: Vec<_> = raw
        .0
        .iter()
        .enumerate()
        .map(|(i, &v)| mk(v, i != 1))
        .collect();
    let p: Vec<_> = raw.1.iter().map(|&v| mk(v, true)).collect();
    let s: Vec<_> = raw.2.iter().map(|&v| mk(v, false)).collect();
    let s: [DualV<N, M, TH>; 4] = [s[0], s[1], s[2], s[3]];
    let mut du = vec![DualV::<N, M, TH>::constant(0.0); 3];
    let mut regs = vec![DualV::<N, M, TH>::constant(0.0); reg.n_regs()];
    reg.run(&u, &p, &s, &mut du, &mut regs);
    RHS_DEAD_KEPT.with(|c| c.set(false));
    let top = vtape_len();
    let adj = vadj_init::<1>();
    for (o, d) in du.iter().enumerate() {
        if <DualV<N, M, TH> as PkNum>::is_taped(*d) {
            for c in 0..nc {
                let k = if TH {
                    vadj_index_h::<N, M>(d.i, c)
                } else {
                    vadj_index::<N, 1>(d.i, c, 0)
                };
                adj[k] = out_bar[o * nc + c];
            }
        }
    }
    let a = if TH {
        vsweep_h::<N, M>(top)
    } else {
        vsweep::<N, 1>(top)
    };
    let mut out = Vec::new();
    for &id in &leaves {
        for c in 0..nc {
            out.push(if TH {
                a[vadj_index_h::<N, M>(id, c)]
            } else {
                a[vadj_index::<N, 1>(id, c, 0)]
            });
        }
    }
    (out, du, top)
}

/// Dropped and kept recordings of one case: bit-identical values, jets and leaf adjoints;
/// how many outputs were dropped.
fn check_dead_outputs<const N: usize, const M: usize, const TH: bool>(
    reg: &RhsRegProgram,
    raw: (&[f64], &[f64], &[f64; 4]),
    zero: &[bool],
    rng: &mut Rng,
    label: &str,
) -> usize {
    let nc = if TH { 1 + N + M } else { 1 + N };
    let out_bar: Vec<f64> = (0..3 * nc).map(|_| rng.unit() * 2.0 - 1.0).collect();
    let (want, du_kept, top_kept) =
        masked_block_adjoints::<N, M, TH>(reg, raw, zero, &out_bar, true);
    let (got, du, top) = masked_block_adjoints::<N, M, TH>(reg, raw, zero, &out_bar, false);
    let same = |x: f64, y: f64| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan());
    let mut dropped = 0;
    for (o, (a, b)) in du_kept.iter().zip(&du).enumerate() {
        assert!(
            same(a.v, b.v)
                && a.g.iter().zip(&b.g).all(|(x, y)| same(*x, *y))
                && a.h.iter().zip(&b.h).all(|(x, y)| same(*x, *y)),
            "{label}: du[{o}] differs: kept {a:?}, dropped {b:?}"
        );
        if PkNum::is_taped(*a) && !PkNum::is_taped(*b) {
            dropped += 1;
            assert!(
                PkNum::is_constant(*b),
                "{label}: dropped du[{o}] is not a constant"
            );
        }
    }
    assert_eq!(
        top_kept - top,
        dropped,
        "{label}: one node per dropped output"
    );
    for (k, (w, g)) in want.iter().zip(&got).enumerate() {
        assert!(
            same(*w, *g),
            "{label}: leaf adjoint {k}: dropped {g:e} vs kept {w:e}"
        );
    }
    dropped
}

/// #2760: a block that leaves its structural-zero outputs off the tape is the block that
/// tapes them, bit for bit — every output's value, tangents and payload, and every leaf's
/// adjoint after a sweep from adjoints on every output — over the outer-gradient layouts and
/// the taped-payload one. The dropped outputs' local partials toward every taped input are
/// exact zeros, which the sweep already skips; this pins that, and that the drop happens.
#[test]
fn dropping_a_blocks_structural_zero_outputs_moves_no_bit() {
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let n_params = crate::types::MAX_PK_PARAMS + 2;
    let mut dropped = 0;
    // The gated shape first, then random programs.
    let mut cases = vec![
        "  d/dt(x1) = C * A * x2\n  d/dt(x2) = C / B - A * x2\n  d/dt(x3) = A * C * x1".to_string(),
    ];
    cases.extend((0..80).map(|_| random_odes(&mut rng)));
    for (case, odes) in cases.iter().enumerate() {
        let prog = program(odes);
        let reg = RhsRegProgram::compile(&prog.stmts, &layout(&prog))
            .unwrap_or_else(|e| panic!("case {case} did not compile ({e:?}):\n{odes}"));
        // Away from the guards, as the block-vs-per-operation test; C a structural zero in
        // the gated case, otherwise a random half of the states and parameters.
        let mut raw_u: Vec<f64> = (0..3).map(|_| rng.unit() * 3.0 + 0.2).collect();
        let mut raw_p: Vec<f64> = (0..n_params).map(|_| rng.unit() * 3.0 + 0.2).collect();
        let raw_s = [
            1.75,
            rng.unit() * 5.0 + 0.1,
            rng.unit() * 2.0 + 0.1,
            f64::EPSILON,
        ];
        let mut zero = vec![false; 3 + n_params + 4];
        if case == 0 {
            zero[3 + prog.indiv_to_params_slot[2]] = true;
        } else {
            for z in zero.iter_mut().take(3 + n_params) {
                *z = rng.below(2) == 0;
            }
        }
        for (i, z) in zero.iter().enumerate().take(3 + n_params) {
            if *z {
                if i < 3 {
                    raw_u[i] = 0.0;
                } else {
                    raw_p[i - 3] = 0.0;
                }
            }
        }
        let raw = (&raw_u[..], &raw_p[..], &raw_s);
        let _time = ModelTimeGuard::enter(1.75);
        let label = format!("case {case}:\n{odes}");
        let d = check_dead_outputs::<2, 0, false>(&reg, raw, &zero, &mut rng, &label);
        check_dead_outputs::<3, 6, false>(&reg, raw, &zero, &mut rng, &label);
        check_dead_outputs::<2, 2, true>(&reg, raw, &zero, &mut rng, &label);
        if case == 0 {
            assert_eq!(d, 2, "the gated case drops both gated derivatives");
        }
        dropped += d;
    }
    eprintln!("{dropped} outputs dropped over {} cases", cases.len());
    assert!(dropped >= 20, "only {dropped} outputs dropped");
}

/// #2760: the analysis reads values, so a structural zero times a factor whose value is
/// finite but whose tangent is not — a diverged upstream jet — is reported dead, while the
/// forward's lanes for it are `0·∞ = NaN`. Such an output must stay on the tape with its
/// `NaN`, as it was before the drop, rather than leave as a clean zero: the block drops an
/// output only when every lane the forward wrote is `0`. Both sides: the same program with a
/// finite tangent is dropped.
#[test]
fn a_structural_zero_against_a_non_finite_tangent_stays_on_the_tape() {
    let prog = program("  d/dt(x1) = C * A\n  d/dt(x2) = -A * x2\n  d/dt(x3) = B * x3");
    let reg = RhsRegProgram::compile(&prog.stmts, &layout(&prog)).expect("compiles");
    let n_params = crate::types::MAX_PK_PARAMS + 2;
    let (a, c) = (prog.indiv_to_params_slot[0], prog.indiv_to_params_slot[2]);
    let _time = ModelTimeGuard::enter(1.75);
    let run = |tangent: f64| -> DualV<2, 0, false> {
        crate::sens::vtape::vtape_reset(3);
        let leaf = |v: f64| DualV::<2, 0, false>::leaf(v, [0.5, -0.25], []);
        let u: Vec<_> = [1.0, 2.0, 3.0].iter().map(|&v| leaf(v)).collect();
        let mut p: Vec<_> = (0..n_params).map(|_| leaf(0.7)).collect();
        p[a] = DualV::<2, 0, false>::leaf(2.0, [tangent, 0.0], []);
        p[c] = DualV::<2, 0, false>::constant(0.0);
        let s = [DualV::<2, 0, false>::constant(1.75); 4];
        let mut du = vec![DualV::<2, 0, false>::constant(0.0); 3];
        let mut regs = vec![DualV::<2, 0, false>::constant(0.0); reg.n_regs()];
        reg.run(&u, &p, &s, &mut du, &mut regs);
        du[0]
    };
    let dropped = run(1.0);
    assert!(
        !PkNum::is_taped(dropped),
        "a finite tangent: the gated term is dropped"
    );
    let kept = run(f64::INFINITY);
    assert!(
        PkNum::is_taped(kept) && kept.g[0].is_nan(),
        "a non-finite tangent: the gated term stays taped with its NaN, got {kept:?}"
    );
}

/// The skip in `vtape::record_rhs` reads three things: the loaded states, the loaded
/// `params` slots and the loaded seeds (#2761 review round 1). The gated fixtures always
/// carry a structural-zero parameter (a gate that is off), so they decide on the `params`
/// clause alone and leave the other two unpinned. Here the program has no zero source and
/// every parameter is a live leaf, so a structural zero reaches it through a state or a
/// seed only: the recorder must still run the analysis and drop that output. The control,
/// with no structural zero among the inputs, tapes all three.
#[test]
fn a_structural_zero_state_or_seed_alone_makes_the_recorder_analyse() {
    let prog = program("  d/dt(x1) = A * x2\n  d/dt(x2) = B * TAD\n  d/dt(x3) = B * x3");
    let reg = RhsRegProgram::compile(&prog.stmts, &layout(&prog)).expect("compiles");
    assert!(
        !super::makes_structural_zeros(&reg.code),
        "the program must have no zero source, or the skip is never asked"
    );
    let n_params = crate::types::MAX_PK_PARAMS + 2;
    let _time = ModelTimeGuard::enter(1.75);
    // Which outputs are taped, with `x2` and `TAD` (seed 2) at the given values: a `0` is
    // the untaped constant, a structural zero.
    let taped = |x2: f64, tad: f64| -> [bool; 3] {
        crate::sens::vtape::vtape_reset(3);
        let leaf = |v: f64| DualV::<2, 0, false>::leaf(v, [0.5, -0.25], []);
        let mut u: Vec<_> = [1.0, x2, 3.0].iter().map(|&v| leaf(v)).collect();
        if x2 == 0.0 {
            u[1] = DualV::<2, 0, false>::constant(0.0);
        }
        let p: Vec<_> = (0..n_params).map(|_| leaf(0.7)).collect();
        let mut s = [DualV::<2, 0, false>::constant(1.75); 4];
        s[2] = DualV::<2, 0, false>::constant(tad);
        let mut du = vec![DualV::<2, 0, false>::constant(0.0); 3];
        let mut regs = vec![DualV::<2, 0, false>::constant(0.0); reg.n_regs()];
        reg.run(&u, &p, &s, &mut du, &mut regs);
        [
            PkNum::is_taped(du[0]),
            PkNum::is_taped(du[1]),
            PkNum::is_taped(du[2]),
        ]
    };
    assert_eq!(taped(2.0, 1.0), [true, true, true], "no structural zero");
    assert_eq!(taped(0.0, 1.0), [false, true, true], "a zero state alone");
    assert_eq!(taped(2.0, 0.0), [true, false, true], "a zero seed alone");
}

/// The QSS-TMDD shape that motivated compiling `sqrt` (#2844), plus a bare `sqrt` of a state
/// so the guard's inputs reach it directly.
const SQRT_ODES: &str = "  d/dt(x1) = sqrt(x1)
  d/dt(x2) = sqrt(x2 - A) * B
  d/dt(x3) = 0.5 * ((x3 - x2 - C) + sqrt((x3 - x2 - C) * (x3 - x2 - C) + 4 * C * x3))";

/// `sqrt` at its guard, on every route: the `f64` lane VM, the generic interpreter on the
/// duals, the register code and the native code, bit for bit — at a negative argument, `0`,
/// `−0.0`, a `NaN`, a tiny value and an ordinary one. And the guard's meaning, pinned on the
/// values: a negative or `NaN` argument floors to `+0`, `−0.0` passes through (the `v ≥ 0`
/// test, not `f64::max`, whose signed-zero result is not specified), and a non-positive
/// argument is a constant, with exactly zero jets — not the `inf · 0 = NaN` of a dual's own
/// `sqrt` at `0`, whose bits optimised code does not reproduce across routes (#2844).
/// (The tiny value is `1e-100`, not smaller: below ~`1e-123` a dual's `h2 = −0.25/s³`
/// overflows on every route alike, which is `sqrt`'s singularity, not the guard.)
#[test]
fn sqrt_at_its_guard_is_the_interpreter_bit_for_bit_on_every_route() {
    use crate::sens::dual1::Dual1;
    use crate::sens::dual2::Dual2;
    let prog = program(SQRT_ODES);
    let reg = prog.reg.as_ref().expect("`sqrt` compiles (#2844)");
    assert!(prog.route_note().is_none(), "{:?}", prog.route_note());
    let n_params = crate::types::MAX_PK_PARAMS + 2;
    let p: Vec<f64> = (0..n_params).map(|i| 0.4 + i as f64 / 5.0).collect();
    let s = [1.75, 0.0, 0.0, f64::EPSILON];
    for x in [-1.0, 0.0, -0.0, f64::NAN, 1e-100, 2.25] {
        let u = [x, x, x];
        let label = format!("x = {x:?}");
        // f64: the production lane VM against the register code.
        let mut vars = bind(&prog, &u, &p, &s);
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
        let du_lane = du_ref.clone();
        let mut regs = vec![0.0; reg.n_regs()];
        super::RHS_JIT_DISABLED.with(|c| c.set(true));
        reg.run(&u, &p, &s, &mut du_reg, &mut regs);
        super::RHS_JIT_DISABLED.with(|c| c.set(false));
        assert!(
            bits_eq(&du_ref, &du_reg),
            "{label}: f64 lane VM {du_ref:?} vs register {du_reg:?}"
        );
        // f64: the generic interpreter agrees with the lane VM too.
        assert!(check_on(&prog, &u, &p, &s, &label));
        // The guard's meaning on `d/dt(x1) = sqrt(x1)`.
        let want = match x {
            v if v.is_nan() || v < 0.0 => 0.0f64,
            v => v.sqrt(),
        };
        assert_eq!(du_lane[0].to_bits(), want.to_bits(), "{label}: sqrt(x1)");
        // The duals through the generic interpreter.
        let lift_f = |v: &[f64]| -> Vec<DualF<2, 0>> {
            v.iter()
                .enumerate()
                .map(|(i, &v)| DualF::<2, 0>::var(v, i % 2) * DualF::constant(0.5 + i as f64 / 7.0))
                .collect()
        };
        let (uf, pf, sf) = (lift_f(&u), lift_f(&p), lift_f(&s));
        assert!(check_on(
            &prog,
            &uf,
            &pf,
            &[sf[0], sf[1], sf[2], sf[3]],
            &label
        ));
        // The jets of `sqrt(x1)` on the register and the native code: zero where the
        // argument is not positive (`lift_f` keeps `x`'s sign and zeros), live otherwise.
        let sf4 = [sf[0], sf[1], sf[2], sf[3]];
        let mut du_native = vec![DualF::<2, 0>::constant(9.0); 3];
        let mut regs = vec![DualF::<2, 0>::constant(0.0); reg.n_regs()];
        reg.run(&uf, &pf, &sf4, &mut du_native, &mut regs);
        let a = uf[0].v; // `0.5 · x`
        let want_f = if a.is_nan() || a < 0.0 { 0.0 } else { a.sqrt() };
        for du in [run_without_jit(reg, &uf, &pf, &sf4), du_native] {
            assert_eq!(du[0].v.to_bits(), want_f.to_bits(), "{label}: DualF value");
            if x > 0.0 {
                assert!(du[0].g[0] > 0.0, "{label}: live jet {:?}", du[0].g);
            } else {
                assert!(
                    du[0].g.iter().all(|g| g.to_bits() == 0),
                    "{label}: jets {:?}",
                    du[0].g
                );
            }
        }
        let lift_v = |v: &[f64]| -> Vec<DualV<2, 3>> {
            v.iter()
                .enumerate()
                .map(|(i, &v)| DualV::<2, 3>::var(v, i % 2) + DualV::var(0.3, (i + 1) % 2))
                .collect()
        };
        let (uv, pv, sv) = (lift_v(&u), lift_v(&p), lift_v(&s));
        assert!(check_on(
            &prog,
            &uv,
            &pv,
            &[sv[0], sv[1], sv[2], sv[3]],
            &label
        ));
        let lift_1 = |v: &[f64]| -> Vec<Dual1<2>> {
            v.iter()
                .enumerate()
                .map(|(i, &v)| Dual1::<2>::var(v, i % 2))
                .collect()
        };
        let (u1, p1, s1) = (lift_1(&u), lift_1(&p), lift_1(&s));
        assert!(check_on(
            &prog,
            &u1,
            &p1,
            &[s1[0], s1[1], s1[2], s1[3]],
            &label
        ));
        let lift_2 = |v: &[f64]| -> Vec<Dual2<2>> {
            v.iter()
                .enumerate()
                .map(|(i, &v)| <Dual2<2> as PkNum>::var(v, i % 2))
                .collect()
        };
        let (u2, p2, s2) = (lift_2(&u), lift_2(&p), lift_2(&s));
        assert!(check_on(
            &prog,
            &u2,
            &p2,
            &[s2[0], s2[1], s2[2], s2[3]],
            &label
        ));
        // Native against the register code: f64, first- and second-order lanes.
        let raw = (&u[..], &p[..], &s);
        check_native(reg, raw, &label);
        check_native_lifted::<DualF<2, 0>>(reg, raw, &label);
        check_native_lifted::<DualF<8, 0>>(reg, raw, &label);
        check_native_lifted::<DualF<2, 3>>(reg, raw, &label);
        check_native_lifted::<DualV<3, 6>>(reg, raw, &label);
        check_native_lifted::<DualV<2, 2, true>>(reg, raw, &label);
    }
}

/// `sqrt`'s jets on the compiled routes against central finite differences of the `f64`
/// RHS: the register code (`Dual2`) and the native code (`DualF`, packed Hessian), at points
/// where every `sqrt` argument is positive, plus the reverse pass against forward tangents.
#[test]
fn sqrt_derivatives_match_finite_differences() {
    use crate::sens::dual2::Dual2;
    let prog = program(SQRT_ODES);
    let reg = prog.reg.as_ref().expect("`sqrt` compiles");
    // `[x1, x2, x3, A, B, C]`.
    let points = [
        [0.4, 1.2, 0.5, 0.3, 0.7, 1.1],
        [3.0, 2.4, 1.6, 0.9, 0.6, 1.3],
    ];
    let f = |x: &[f64; 6]| rhs_at(&prog, reg, x, &|v, _| v);
    let h = 1e-5;
    let h2 = 1e-4;
    let mut worst_g = 0.0f64;
    let mut worst_h = 0.0f64;
    let mut live = 0usize;
    for x in &points {
        assert!(x[1] - x[3] > 0.0, "every sqrt argument is positive");
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
                for got in [x2[o].grad[i], xf[o].g[i]] {
                    assert!(got.is_finite() && fd.is_finite());
                    worst_g = worst_g.max((got - fd).abs() / (1.0 + fd.abs()));
                }
                for j in 0..=i {
                    let fd = (step(&[(i, h2), (j, h2)])[o]
                        - step(&[(i, h2), (j, -h2)])[o]
                        - step(&[(i, -h2), (j, h2)])[o]
                        + step(&[(i, -h2), (j, -h2)])[o])
                        / (4.0 * h2 * h2);
                    live += usize::from(fd.abs() > 1e-3);
                    let packed = xf[o].h[i * (i + 1) / 2 + j];
                    for got in [x2[o].hess[i][j], packed] {
                        assert!(got.is_finite() && fd.is_finite());
                        worst_h = worst_h.max((got - fd).abs() / (1.0 + fd.abs()));
                    }
                }
            }
        }
    }
    // The second derivatives are genuinely non-zero (sqrt's curvature is under test).
    assert!(live > 10, "only {live} live Hessian entries");
    eprintln!("sqrt derivatives vs FD: gradient {worst_g:e}, Hessian {worst_h:e}");
    // Measured (macOS arm64): gradient 3.8e-11 (central differences at h = 1e-5), Hessian
    // 3.9e-8 (second differences at h = 1e-4, rounding-bound). The bounds keep ~10× headroom.
    assert!(worst_g < 4e-10, "gradient vs FD {worst_g:e}");
    assert!(worst_h < 4e-7, "Hessian vs FD {worst_h:e}");

    // The reverse pass (what a taped recording of the native block differentiates).
    use super::ir::Layout;
    let mut rng = Rng(0x5157_2844_0000_0001);
    let mut worst = 0.0f64;
    for l in [
        Layout::F64,
        Layout {
            n: 2,
            m: 0,
            th: false,
            wide: false,
        },
        Layout {
            n: 2,
            m: 2,
            th: true,
            wide: false,
        },
    ] {
        let w = check_reverse(reg, l, &mut rng, &format!("{l:?}"));
        assert!(w.is_finite(), "{l:?}: {w}");
        worst = worst.max(w);
    }
    assert!(worst < 1e-12, "reverse vs forward tangents {worst:e}");
}
