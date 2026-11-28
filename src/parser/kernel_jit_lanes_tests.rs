//! The lane-loop prototype against the production kernel, bit for bit, and its numbers.
use super::*;
use crate::sens::num::PkNum;

/// The packed η-Hessian layout over `n` η.
fn hess(n: u16) -> Layout {
    Layout { n, m: n * (n + 1) / 2, th: false, wide: false, tan: 0 }
}

/// The production forward kernel of `cp` at `layout` and the looped one, run on the same
/// inputs (every lane of every input filled, so a lane read that should have been a folded
/// constant shows): their outputs, bit for bit, through the interpreter and the native tiers.
/// Returns the number of output lanes compared.
fn agree(
    cp: &ChunkProgram, layout: Layout, n_out: usize, lits: &[f64], fill: impl Fn(usize) -> f64,
) -> usize {
    let l = layout.lanes();
    let slots = cp.n_in + n_out;
    let idx: Vec<u32> = (0..slots as u32).collect();
    let mut init = vec![0.0; slots * l];
    for (c, x) in init[..cp.n_in * l].iter_mut().enumerate() {
        *x = fill(c);
    }
    let t = cp.get(layout, Sweep::Forward).expect("a production kernel");
    let mut want = init.clone();
    unsafe {
        t.interpret(want.as_mut_ptr(), std::ptr::null_mut(), idx.as_ptr(), lits.as_ptr(), 1, 0)
    };
    t.promote(true, "speed");
    let mut native = init.clone();
    unsafe { t.call(native.as_mut_ptr(), std::ptr::null_mut(), idx.as_ptr(), lits.as_ptr()) };
    for c in cp.n_in * l..slots * l {
        assert_eq!(want[c].to_bits(), native[c].to_bits(), "production tiers, lane {c}");
    }
    for loop_min in [1, 2, 4, usize::MAX] {
        for (quick, opt) in [(true, "none"), (true, "speed"), (false, "speed")] {
            let k = Looped::new(cp, layout, Opts { loop_min }, quick, opt)
                .expect("the prototype lowers it");
            if loop_min == 1 {
                assert!(k.counts.unrolled_lanes == 0, "every group a loop at loop_min 1");
            }
            let mut got = init.clone();
            unsafe { k.call(got.as_mut_ptr(), idx.as_ptr(), lits.as_ptr()) };
            for c in cp.n_in * l..slots * l {
                assert_eq!(
                    want[c].to_bits(),
                    got[c].to_bits(),
                    "loop_min {loop_min} {quick}/{opt}: output slot {} lane {}: production {:e}, looped {:e}",
                    c / l - cp.n_in,
                    c % l,
                    want[c],
                    got[c]
                );
            }
        }
    }
    n_out * l
}

/// Every instruction the prototype lowers, over inputs with every kind of fixed lane: a dense
/// one, one carrying two of four directions, two η inputs (one-hot seeds), and literals. The
/// constants `0`, `−0`, `1` and `2` meet every fold (`x·0`, `x·1`, `0 − x`, `0/x`, a select of
/// equal constants), and the guards (`|b| < 1e-30`, `√` of a non-positive, `ln` below its floor)
/// take both sides across the input sets.
fn every_op_chunk() -> ChunkProgram {
    use Flat::*;
    let code = vec![
        LoadState { d: 0, i: 0 },
        LoadState { d: 1, i: 1 },
        LoadState { d: 2, i: 2 },
        LoadState { d: 3, i: 3 },
        LoadSeed { d: 4, k: 0 },
        LoadK { d: 5, c: 0.0 },
        LoadK { d: 6, c: -0.0 },
        LoadK { d: 7, c: 1.0 },
        LoadK { d: 8, c: 2.0 },
        Mul { d: 9, a: 0, b: 2 },
        Mul { d: 10, a: 9, b: 3 },
        Add { d: 11, a: 10, b: 1 },
        Sub { d: 12, a: 11, b: 2 },
        RSub { d: 13, a: 12, b: 4 },
        DivRaw { d: 14, a: 13, b: 1 },
        Div { d: 15, a: 14, b: 0 },
        Mul { d: 16, a: 15, b: 5 },
        Mul { d: 17, a: 15, b: 7 },
        Mul { d: 18, a: 17, b: 6 },
        Add { d: 19, a: 18, b: 5 },
        Sub { d: 20, a: 5, b: 19 },
        DivRaw { d: 21, a: 5, b: 19 },
        AddK { d: 22, a: 20, c: 0.5 },
        SubK { d: 23, a: 22, c: 0.0 },
        RSubK { d: 24, a: 23, c: 3.0 },
        MulK { d: 25, a: 24, c: 1.0 },
        MulK { d: 26, a: 25, c: 0.0 },
        MulK { d: 27, a: 25, c: -1.5 },
        DivK { d: 28, a: 27, c: 4.0 },
        DivK { d: 29, a: 27, c: 0.0 },
        Unary { d: 30, op: super::super::Unary::Exp, a: 28 },
        Unary { d: 31, op: super::super::Unary::Ln, a: 30 },
        Unary { d: 32, op: super::super::Unary::LnRaw, a: 30 },
        Unary { d: 33, op: super::super::Unary::Sqrt, a: 11 },
        Unary { d: 34, op: super::super::Unary::SqrtRhs, a: 12 },
        Unary { d: 35, op: super::super::Unary::Abs, a: 13 },
        Cmp { d: 36, cc: Cc::Lt, a: 11, b: 12 },
        Select { d: 37, c: 36, a: 33, b: 34 },
        Mul { d: 38, a: 37, b: 35 },
        Add { d: 39, a: 38, b: 31 },
        Mul { d: 40, a: 39, b: 32 },
        Add { d: 41, a: 40, b: 16 },
        Add { d: 42, a: 41, b: 21 },
        Add { d: 43, a: 42, b: 26 },
        Add { d: 44, a: 43, b: 29 },
        Unary { d: 45, op: super::super::Unary::Ln, a: 6 },
        Mul { d: 46, a: 8, b: 44 },
        Deriv { state: 0, a: 46 },
        Deriv { state: 1, a: 15 },
        Deriv { state: 2, a: 37 },
        Deriv { state: 3, a: 8 },
        Deriv { state: 4, a: 6 },
        Deriv { state: 5, a: 45 },
        Deriv { state: 6, a: 36 },
        // A stopped constant is a computed value to the folding, not the constant `1`.
        Unary { d: 47, op: super::super::Unary::StopGrad, a: 7 },
        Mul { d: 48, a: 47, b: 44 },
        Deriv { state: 7, a: 48 },
    ];
    let two = DirMask::eta(0) | &DirMask::eta(2);
    ChunkProgram {
        code,
        n_regs: 49,
        spec: vec![
            (DirMask::ALL, u16::MAX),
            (two, u16::MAX),
            (DirMask::eta(1), 1),
            (DirMask::eta(3), 3),
        ],
        n_in: 4,
        compiled: Mutex::new(HashMap::new()),
    }
}

/// The prototype is the production kernel bit for bit on every instruction it covers, at the
/// first-order and the packed second-order layouts, whatever the loop threshold (every group a
/// loop, every group written out, and between) and the Cranelift tier. Kills: a fold rule that
/// differs from `fold_with`'s (a `−0` where production keeps `+0`, a `NaN` where it folds
/// `inf·0`), a payload lane reading the wrong pair (`(k, l)` swapped onto a non-symmetric
/// operand), a loop reading a stale table row, a group mixing two signatures.
#[test]
fn the_looped_forward_is_the_production_kernel_bit_for_bit() {
    let cp = every_op_chunk();
    let mut lanes = 0;
    for layout in [Layout { n: 4, m: 0, th: false, wide: false, tan: 0 }, hess(4)] {
        for set in 0..4 {
            let v: [f64; 4] = [
                [1.3, 0.4, 2.0, 0.9],
                [-1.3, 0.4, -2.0, -0.9],
                [0.0, 1.0, 3.0, 2.0],
                [1e-31, -0.0, 1e3, -7.0],
            ][set];
            let l = layout.lanes();
            lanes += agree(&cp, layout, 8, &[0.75], |c| {
                let (i, j) = (c / l, c % l);
                if j == 0 {
                    v[i]
                } else {
                    0.3 - 0.07 * j as f64 + 0.19 * i as f64
                }
            });
        }
    }
    assert!(lanes > 0);
}

/// `n_s` states, `n_eta` η: a chain with Michaelis–Menten elimination from the central
/// compartment (so it is not affine in the state), each parameter `θ·exp(η)`.
fn ode_case(
    n_s: usize, n_eta: usize, method: crate::ode::solver::OdeMethod,
) -> impl Fn(&[Tr]) -> Vec<Tr> {
    move |e: &[Tr]| {
        let k = |c: f64| Tr::from_f64(c);
        let p: Vec<Tr> = (0..n_eta).map(|i| k(0.3 + 0.1 * i as f64) * e[i].exp()).collect();
        let u0: Vec<Tr> = (0..n_s).map(|i| if i == 0 { k(10.0) + p[0] } else { k(0.0) }).collect();
        let rhs = move |u: &[Tr], p: &[Tr], _t: Tr, du: &mut [Tr]| {
            let n = u.len();
            let np = p.len();
            let pk = |i: usize| p[i % np];
            du[0] = -pk(0) * u[0];
            let mm = pk(1) * u[1] / (pk(2) + u[1]);
            du[1] = pk(0) * u[0] - mm - pk(3) * u[1];
            for i in 2..n {
                du[i] = pk(i + 2) * u[i - 1] - pk(i + 3) * u[i];
            }
            du[1] = du[1] + pk(n + 3) * u[n - 1];
        };
        let opts = crate::ode::solver::OdeSolverOptions {
            reltol: 1e-4,
            abstol: 1e-6,
            method,
            auto_switch: false,
            ..Default::default()
        };
        let mut auto = crate::ode::solver::OdeAutoSwitchState::default();
        let (hard, _) = Tr::ode_segment(&rhs, &u0, (0.0, 4.0), &p, &[4.0], &[], &opts, &mut auto)
            .expect("records as a loop call");
        hard.iter().flat_map(|s| s.u.clone()).collect()
    }
}

/// The recorded step bodies (and the right-hand side) of `ode_case`.
fn ode_bodies(
    n_s: usize, n_eta: usize, method: crate::ode::solver::OdeMethod,
) -> Vec<(String, Arc<ChunkProgram>)> {
    let f = ode_case(n_s, n_eta, method);
    begin();
    let et: Vec<Tr> = (0..n_eta).map(|k| Tr::eta(k, 0.05 * k as f64 - 0.1)).collect();
    let y = f(&et);
    let rec = end(Some(y.iter().map(|v| v.reg()).collect())).expect("records");
    let mut out = Vec::new();
    for c in &rec.loops {
        if let CallKind::Ode(spec) = c {
            out.push(("rhs".to_string(), spec.rhs_program()));
            for (m, cp) in spec.step_programs() {
                out.push((format!("{m:?} step"), cp));
            }
        }
    }
    assert!(!out.is_empty(), "the recording has an ODE segment");
    out
}

/// The outputs a chunk writes: one past its highest `Deriv`.
fn n_out(cp: &ChunkProgram) -> usize {
    cp.code
        .iter()
        .filter_map(|f| {
            if let Flat::Deriv { state, .. } = *f {
                Some(state as usize + 1)
            } else {
                None
            }
        })
        .max()
        .unwrap_or(0)
}
/// The literals a chunk reads: one past its highest `LoadSeed`.
fn n_lits(cp: &ChunkProgram) -> usize {
    cp.code
        .iter()
        .filter_map(|f| if let Flat::LoadSeed { k, .. } = *f { Some(k as usize + 1) } else { None })
        .max()
        .unwrap_or(0)
}

/// A real recorded body: the right-hand side and both steps (RK45's, Rodas5P's) of a 3-state
/// segment over 3 η, at the packed η-Hessian, are the production kernel bit for bit.
#[test]
fn a_recorded_stiff_step_body_loops_bit_for_bit() {
    use crate::ode::solver::OdeMethod;
    let layout = hess(3);
    let l = layout.lanes();
    let mut covered = 0;
    for (name, cp) in ode_bodies(3, 3, OdeMethod::Rodas5P) {
        let lits: Vec<f64> = (0..n_lits(&cp).max(1)).map(|i| 0.01 + 0.37 * i as f64).collect();
        let n_o = n_out(&cp);
        if name.starts_with("Rodas5P") {
            // The stiff step reaches the lowering's host-call, stopped-value and select arms (the
            // factorization's pivoting), which the synthetic chunk does not.
            let has = |p: &dyn Fn(&Flat) -> bool| cp.code.iter().any(p);
            assert!(has(&|f| matches!(f, Flat::Host { .. })), "{name}: a host call");
            assert!(has(&|f| matches!(f, Flat::HostOut { .. })), "{name}: a host output");
            assert!(has(&|f| matches!(f, Flat::Select { .. })), "{name}: a select");
            assert!(
                has(&|f| matches!(f, Flat::Unary { op: super::super::Unary::StopGrad, .. })),
                "{name}: a stopped value"
            );
        }
        // Values near the recording's scale (the step's `dt` and `t` small and positive), every
        // derivative lane live.
        agree(&cp, layout, n_o, &lits, |c| {
            let (i, j) = (c / l, c % l);
            if j == 0 {
                0.05 + 0.11 * (i % 7) as f64
            } else {
                0.02 * (1.0 + (i * 13 + j * 7) as f64 % 11.0) - 0.1
            }
        });
        covered += 1;
        eprintln!("{name}: {} instructions, bit for bit", cp.code.len());
    }
    assert!(covered >= 3, "the rhs and both steps");
}

/// The prototype's numbers (#3080): a 9-state Rodas5P step body over 11 η at the packed
/// η-Hessian (78 lanes), unrolled (the production kernel) against looped, at the plan's large-
/// kernel tier (single-pass allocator, optimizer on). Run with
/// `cargo test --release --lib lane_loop_prototype_numbers -- --ignored --nocapture`.
#[test]
#[ignore = "measurement: prints the prototype's numbers"]
fn lane_loop_prototype_numbers() {
    use crate::ode::solver::OdeMethod;
    let n_eta = std::env::var("FERX_LANES_ETA").ok().and_then(|s| s.parse().ok()).unwrap_or(11);
    let n_s = std::env::var("FERX_LANES_STATES").ok().and_then(|s| s.parse().ok()).unwrap_or(9);
    let layout = hess(n_eta as u16);
    let l = layout.lanes();
    for (name, cp) in ode_bodies(n_s, n_eta, OdeMethod::Rodas5P) {
        let lits: Vec<f64> = (0..n_lits(&cp).max(1)).map(|i| 0.01 + 0.37 * i as f64).collect();
        let n_o = n_out(&cp);
        let slots = cp.n_in + n_o;
        let idx: Vec<u32> = (0..slots as u32).collect();
        let mut init = vec![0.0; slots * l];
        for (c, x) in init[..cp.n_in * l].iter_mut().enumerate() {
            let (i, j) = (c / l, c % l);
            *x = if j == 0 {
                0.05 + 0.11 * (i % 7) as f64
            } else {
                0.02 * (1.0 + (i * 13 + j * 7) as f64 % 11.0) - 0.1
            };
        }
        // Unrolled: IR, then the production tier-2 compile.
        let t0 = std::time::Instant::now();
        let t = cp.get(layout, Sweep::Forward).expect("a production kernel");
        let build_s = t0.elapsed().as_secs_f64();
        let ir_insts = t.n_insts;
        NATIVE_BYTES.with(|c| c.set(0));
        NATIVE_CLIF.with(|c| c.set(0));
        let t1 = std::time::Instant::now();
        assert!(t.promote(true, "speed"));
        let comp_s = t1.elapsed().as_secs_f64();
        let (bytes, clif) = (NATIVE_BYTES.with(|c| c.get()), NATIVE_CLIF.with(|c| c.get()));
        // `promote` installs the tier-2 entry point but leaves the call count at 0, so call 128
        // would cross `TIER1_CALLS` and compile the "none" tier inside the timed loop, replacing
        // the optimized entry point (review of #3086). Park the count past the last promotion
        // point: `count` then never promotes, and the loop times tier 2 alone.
        t.calls.store(TIER3_CALLS, Relaxed);
        let entry = t.native.load(Relaxed);
        let mut want = init.clone();
        let reps = 200;
        let t2 = std::time::Instant::now();
        for _ in 0..reps {
            unsafe { t.call(want.as_mut_ptr(), std::ptr::null_mut(), idx.as_ptr(), lits.as_ptr()) };
        }
        let call_us = t2.elapsed().as_secs_f64() / reps as f64 * 1e6;
        assert_eq!(
            (t.native.load(Relaxed), NATIVE_CLIF.with(|c| c.get())),
            (entry, clif),
            "{name}: the timed calls ran the tier-2 entry point and compiled nothing"
        );
        eprintln!(
            "{name}: {} flat instructions, {n_o} outputs, {l} lanes\n  unrolled: {ir_insts} IR insts, {clif} clif insts, {:.1} KiB native, IR build {:.2} s, compile {:.2} s, {call_us:.1} µs/call",
            cp.code.len(),
            bytes as f64 / 1024.0,
            build_s,
            comp_s
        );
        for loop_min in [2, 4, 8, 16] {
            let Some(k) = Looped::new(&cp, layout, Opts { loop_min }, true, "speed") else {
                eprintln!("  looped: not covered by the prototype");
                break;
            };
            let mut got = init.clone();
            let t3 = std::time::Instant::now();
            for _ in 0..reps {
                unsafe { k.call(got.as_mut_ptr(), idx.as_ptr(), lits.as_ptr()) };
            }
            let lcall_us = t3.elapsed().as_secs_f64() / reps as f64 * 1e6;
            let same = (cp.n_in * l..slots * l).all(|c| want[c].to_bits() == got[c].to_bits());
            let c = k.counts;
            eprintln!(
                "  looped (loop_min {loop_min}): {} clif insts, {:.1} KiB native, compile {:.3} s, {lcall_us:.1} µs/call, {} loops over {} lanes + {} lanes written out, scratch {:.1} KiB, tables {:.1} KiB, bit-identical: {same}",
                c.clif_insts,
                c.native_bytes as f64 / 1024.0,
                k.compile_s,
                c.loops,
                c.looped_lanes,
                c.unrolled_lanes,
                c.scratch as f64 * 8.0 / 1024.0,
                c.table as f64 * 4.0 / 1024.0,
            );
            assert!(same, "{name} loop_min {loop_min}: the looped kernel's bits");
        }
    }
}
