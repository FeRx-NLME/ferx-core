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
    match rng.below(12) {
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

/// A model whose `[odes]` block is `odes`, parsed by the production parser.
fn program(odes: &str) -> OdeRhsProgram {
    let text = format!(
        "[parameters]\n  theta TVA(1.0, 0.01, 10.0)\n  theta TVB(2.0, 0.01, 10.0)\n  \
         theta TVC(0.5, 0.01, 10.0)\n  omega ETA_A ~ 0.1\n  sigma S ~ 0.1 (sd)\n\n\
         [individual_parameters]\n  A = TVA * exp(ETA_A)\n  B = TVB\n  C = TVC\n\n\
         [structural_model]\n  ode(obs_cmt=x2, states=[x1, x2, x3])\n\n\
         [odes]\n{odes}\n\n[error_model]\n  DV ~ proportional(S)\n"
    );
    let model = parse_model_string(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
    model.ode_spec.unwrap().rhs_program.unwrap()
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
    let Some(reg) = RhsRegProgram::compile(&prog.stmts, &layout(prog)) else {
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
        if let Some(reg) = RhsRegProgram::compile(&prog.stmts, &layout(&prog)) {
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
    }
    // The generator only emits translatable programs, so they all compile; a collapse
    // here means the compiler started declining and the test checks nothing.
    assert_eq!(compiled, 250, "only {compiled} programs compiled");
}

#[test]
fn a_program_outside_the_translatable_shape_is_not_compiled() {
    // `if` (jumps), and the ops whose f64 meaning the generic primitives do not reproduce.
    for odes in [
        "  d/dt(x1) = if (x1 < 1) A else B\n  d/dt(x2) = -x2\n  d/dt(x3) = 0",
        "  d/dt(x1) = sqrt(x1)\n  d/dt(x2) = -x2\n  d/dt(x3) = 0",
        "  d/dt(x1) = inv_logit(x1)\n  d/dt(x2) = -x2\n  d/dt(x3) = 0",
        "  d/dt(x1) = logit(x1)\n  d/dt(x2) = -x2\n  d/dt(x3) = 0",
    ] {
        let prog = program(odes);
        assert!(
            RhsRegProgram::compile(&prog.stmts, &layout(&prog)).is_none(),
            "{odes}"
        );
        assert!(prog.reg.is_none(), "{odes}");
    }
    // And the production program carries the compiled form when it translates.
    let prog = program("  d/dt(x1) = -A * x1\n  d/dt(x2) = A * x1 - B * x2\n  d/dt(x3) = C");
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

/// Native vs register code on one `T` of `lanes` `f64`s.
fn check_native<T: VmNum>(
    reg: &RhsRegProgram,
    lanes: usize,
    x: (&[T], &[T], &[T; 4]),
    label: &str,
) {
    assert!(
        reg.has_native(lanes),
        "{label}: no native function for {lanes} lanes"
    );
    let want = run_without_jit(reg, x.0, x.1, x.2);
    let mut got = vec![T::from_f64(9.0); 3];
    let mut regs = vec![T::from_f64(0.0); reg.n_regs()];
    reg.run(x.0, x.1, x.2, &mut got, &mut regs);
    assert!(
        bits_eq(&want, &got),
        "{label}: native du differs ({lanes} lanes)"
    );
}

#[test]
fn native_rhs_is_the_register_code_bit_for_bit_on_random_programs() {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let n_params = crate::types::MAX_PK_PARAMS + 2;
    for case in 0..250 {
        let odes = random_odes(&mut rng);
        let prog = program(&odes);
        let reg = RhsRegProgram::compile(&prog.stmts, &layout(&prog))
            .unwrap_or_else(|| panic!("case {case} did not compile:\n{odes}"));
        let raw_u = seed_values(&mut rng, 3);
        let raw_p = seed_values(&mut rng, n_params);
        let raw_s = [1.75, rng.unit() * 5.0, rng.unit() * 2.0, f64::EPSILON];
        let label = format!("case {case}:\n{odes}");
        let _time = ModelTimeGuard::enter(1.75);
        check_native(&reg, 1, (&raw_u[..], &raw_p[..], &raw_s), &label);
        macro_rules! dual {
            ($n:literal) => {{
                // Tangents of mixed sign and zeros, so `0 − g` vs `−g` and the `x·0`
                // products are exercised.
                let lift = |x: &[f64]| -> Vec<DualF<$n, 0>> {
                    x.iter()
                        .enumerate()
                        .map(|(i, &v)| {
                            let mut g = [0.0; $n];
                            for (k, gk) in g.iter_mut().enumerate() {
                                *gk = match (i + k) % 4 {
                                    0 => 0.0,
                                    1 => 0.5 + i as f64 / 7.0,
                                    2 => -1.25,
                                    _ => -0.0,
                                };
                            }
                            DualF::free(v, g, [])
                        })
                        .collect()
                };
                let (u, p, s) = (lift(&raw_u), lift(&raw_p), lift(&raw_s));
                let s: [DualF<$n, 0>; 4] = [s[0], s[1], s[2], s[3]];
                check_native(&reg, $n + 1, (&u[..], &p[..], &s), &label);
            }};
        }
        dual!(1);
        dual!(2);
        dual!(3);
        dual!(7);
    }
}

#[test]
fn a_dual_wider_than_the_native_lanes_keeps_the_register_code() {
    // `DualF<8, 0>` runs its arithmetic out of line, with a different operation order for a
    // constant operand, so it must not reach a native function built for another width.
    let prog = program("  d/dt(x1) = -A * x1 + 2\n  d/dt(x2) = A * x1 - B * x2\n  d/dt(x3) = C");
    let reg = prog.reg.as_ref().expect("translatable");
    let lift = |v: f64| DualF::<8, 0>::free(v, [0.25; 8], []);
    let u = [lift(1.0), lift(2.0), lift(3.0)];
    let p: Vec<_> = (0..crate::types::MAX_PK_PARAMS + 2)
        .map(|i| lift(0.5 + i as f64))
        .collect();
    let s = [lift(0.0), lift(0.0), lift(0.0), lift(f64::EPSILON)];
    let want = run_without_jit(reg, &u, &p, &s);
    let mut got = vec![DualF::<8, 0>::constant(9.0); 3];
    let mut regs = vec![DualF::<8, 0>::constant(0.0); reg.n_regs()];
    reg.run(&u, &p, &s, &mut got, &mut regs);
    assert!(bits_eq(&want, &got));
    assert!(!reg.has_native(10));
}

#[test]
fn native_fd_columns_are_the_lane_evaluator_bit_for_bit() {
    // The Rosenbrock Jacobian's batched columns: one native call per column against the
    // lane evaluator, on the production closures, over programs with and without `TIME`.
    let mut rng = Rng(0x94D0_49BB_1331_11EB);
    let n_params = crate::types::MAX_PK_PARAMS + 2;
    let mut checked = 0;
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
}
