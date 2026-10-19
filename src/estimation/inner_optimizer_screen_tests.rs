//! The #2013 Sobol screen in the cold multi-start: its helpers, the flip-flop subject it
//! exists for, the unimodal subjects it must leave bit-identical, and its reproducibility
//! across threads and data order.

use super::*;
use crate::parser::model_parser::parse_full_model;
use crate::types::{DoseEvent, FitOptions, Population, Subject};

const TIMES: [f64; 15] = [
    0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 36.0, 48.0, 72.0, 96.0,
];

/// One-compartment oral concentration after a 100-unit dose, noise-free.
fn conc(cl: f64, v: f64, ka: f64, t: f64) -> f64 {
    let k = cl / v;
    100.0 * ka / (v * (ka - k)) * ((-k * t).exp() - (-ka * t).exp())
}

fn subject(id: &str, cl: f64, v: f64, ka: f64) -> Subject {
    let n = TIMES.len();
    Subject {
        id: id.into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: TIMES.to_vec(),
        observations: TIMES.iter().map(|&t| conc(cl, v, ka, t)).collect(),
        obs_cmts: vec![1; n],
        cens: vec![0; n],
        occasions: vec![1; n],
        ..Subject::default()
    }
}

/// A flip-flop configuration: the subject's `V` and the model's diagonal Ω.
struct Config {
    name: &'static str,
    v: f64,
    omega: [f64; 3],
    /// Individual objective of the mode a cold BFGS from η = 0 and every axis seed reach.
    trapped: f64,
    /// Individual objective of the deeper, off-axis mode.
    deeper: f64,
}

/// Both configurations of the #2013 report. The objectives are the values the solve
/// returns today; the screen has to carry the subject from the first to the second.
const CONFIGS: [Config; 2] = [
    Config {
        name: "V=10, Ω=[0.04, 0.1, 1]",
        v: 10.0,
        omega: [0.04, 0.1, 1.0],
        trapped: -42.336612,
        deeper: -50.155953,
    },
    Config {
        name: "V=5, Ω=[0.04, 0.04, 2]",
        v: 5.0,
        omega: [0.04, 0.04, 2.0],
        trapped: -21.073865,
        deeper: -44.016869,
    },
];

/// The flip-flop subject: `ka = 0.02` absorbs slower than `k = CL/V` eliminates, so the
/// data fit `(ka, k)` and its swap almost equally well.
fn flip_flop_subject(id: &str, cfg: &Config) -> Subject {
    subject(id, 1.0, cfg.v, 0.02)
}

/// Two ordinary subjects whose individual objective has a single mode.
fn controls() -> [Subject; 2] {
    [
        subject("ctl-a", 1.0, 10.0, 1.0),
        subject("ctl-b", 1.2, 9.0, 0.8),
    ]
}

fn model(omega: [f64; 3]) -> (CompiledModel, FitOptions) {
    let parsed = parse_full_model(&format!(
        "[parameters]
  theta TVCL(1.0, 0.01, 100)
  theta TVV(10.0, 0.1, 1000)
  theta TVKA(1.0, 0.001, 100)
  omega ETA_CL ~ {}
  omega ETA_V  ~ {}
  omega ETA_KA ~ {}
  sigma PROP_ERR ~ 0.05 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
",
        omega[0], omega[1], omega[2]
    ))
    .expect("flip-flop model parses");
    (parsed.model, parsed.fit_options)
}

/// A solve as the fit's stage runs it, with the screen set to `screen`.
fn solve(
    model: &CompiledModel,
    opts: &FitOptions,
    subj: &Subject,
    screen: usize,
    restarts: usize,
    eta_init: Option<&[f64]>,
) -> EbeResult {
    let stage = FitOptions {
        inner_screen: screen,
        ..opts.clone()
    };
    find_ebe_for_stage(
        model,
        subj,
        &model.default_params,
        stage.inner_maxiter,
        stage.inner_tol,
        eta_init,
        None,
        restarts,
        &stage,
    )
}

fn bits(v: &DVector<f64>) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn diag_omega(d: &[f64]) -> crate::types::OmegaMatrix {
    crate::types::OmegaMatrix::from_diagonal(d, d.iter().map(|_| "ETA".to_string()).collect())
}

#[test]
fn screen_seed_is_fnv1a_of_the_id_bytes() {
    // The published FNV-1a 64 test vectors: the offset basis for "", and "a".
    assert_eq!(screen_seed(""), 0xcbf2_9ce4_8422_2325);
    assert_eq!(screen_seed("a"), 0xaf63_dc4c_8601_ec8c);
    assert_eq!(screen_seed("foobar"), 0x8594_4171_f739_67e8);
    assert_ne!(screen_seed("12"), screen_seed("21"));
}

#[test]
fn omega_distance_is_the_prior_metric() {
    let omega = diag_omega(&[4.0, 1.0]);
    // Two prior SDs on the first axis is a distance of 1, not 2.
    assert!((omega_distance(&[2.0, 0.0], &[0.0, 0.0], &omega) - 1.0).abs() < 1e-12);
    // A difference, not a norm of either point.
    assert!((omega_distance(&[3.0, 5.0], &[1.0, 5.0], &omega) - 1.0).abs() < 1e-12);
    assert!((omega_distance(&[0.0, 0.0], &[0.0, 3.0], &omega) - 3.0).abs() < 1e-12);
}

#[test]
fn screen_starts_are_the_lowest_draws_of_the_scaled_prior() {
    use crate::estimation::importance_sampling::{inv_normal_cdf, sobol_uniforms};
    let omega = diag_omega(&[0.04, 0.1, 1.0]);
    let draws: Vec<Vec<f64>> = sobol_uniforms(3, 100, screen_seed("S1"))
        .into_iter()
        .map(|u| {
            let z = DVector::from_iterator(3, u.iter().map(|&ui| 2.0 * inv_normal_cdf(ui)));
            (&omega.chol * z).iter().copied().collect()
        })
        .collect();

    // A flat objective ties every draw, so the stable sort keeps the first three in Sobol
    // order: this pins the stream, its seed, the 2·chol(Ω) scale and the top-K count.
    let flat = screen_starts(&|_| 0.0, &omega, 3, 100, "S1");
    assert_eq!(flat, draws[..3].to_vec());

    // A real objective takes the three lowest, lowest first.
    let sq = |e: &[f64]| e.iter().map(|x| x * x).sum::<f64>();
    let got = screen_starts(&sq, &omega, 3, 100, "S1");
    let mut want = draws.clone();
    want.sort_by(|a, b| sq(a).total_cmp(&sq(b)));
    assert_eq!(got, want[..3].to_vec());

    // A draw whose objective is not finite is dropped, not ranked. `−∞` and a negative
    // NaN both sort ahead of every finite value under `total_cmp`, so a kept one would
    // take a top-K slot.
    let (first, second) = (draws[0].clone(), draws[1].clone());
    let not_finite = |e: &[f64]| {
        if e == first.as_slice() {
            f64::NEG_INFINITY
        } else if e == second.as_slice() {
            -f64::NAN
        } else {
            0.0
        }
    };
    assert_eq!(
        screen_starts(&not_finite, &omega, 3, 100, "S1"),
        draws[2..5].to_vec()
    );

    // Keyed by the ID: another ID draws another stream, the same ID the same one.
    assert_ne!(screen_starts(&|_| 0.0, &omega, 3, 100, "S2"), flat);
    assert_eq!(screen_starts(&|_| 0.0, &omega, 3, 100, "S1"), flat);

    // Nothing to draw.
    assert!(screen_starts(&|_| 0.0, &omega, 3, 0, "S1").is_empty());
    assert!(screen_starts(&|_| 0.0, &omega, 2, 100, "S1").is_empty());
}

/// The defect and its fix in one test: with the screen off the cold solve stays in the
/// shallower mode, with it on it reaches the deeper one — for both configurations and for
/// every subject ID tried, since the ID chooses the Sobol stream.
#[test]
fn the_screen_carries_a_flip_flop_subject_to_the_deeper_mode() {
    for cfg in &CONFIGS {
        let (m, opts) = model(cfg.omega);
        for i in 1..=8 {
            let id = format!("S{i}");
            let subj = flip_flop_subject(&id, cfg);

            let off = solve(&m, &opts, &subj, 0, 1, None);
            assert!(
                (off.nll - cfg.trapped).abs() < 1e-4,
                "{} {id}, inner_screen = 0: nll {} is not the trapped mode {}",
                cfg.name,
                off.nll,
                cfg.trapped
            );
            assert!(!off.screen_moved, "{} {id}: no screen ran", cfg.name);

            let on = solve(&m, &opts, &subj, 100, 1, None);
            assert!(
                (on.nll - cfg.deeper).abs() < 1e-4,
                "{} {id}, inner_screen = 100: nll {} is not the deeper mode {} (trapped {})",
                cfg.name,
                on.nll,
                cfg.deeper,
                cfg.trapped
            );
            assert!(
                on.screen_moved,
                "{} {id}: the screen moved the EBE",
                cfg.name
            );
            assert!(on.converged, "{} {id}: the deeper mode converged", cfg.name);
            // The two modes really are apart in the prior metric, so the distance gate
            // is not what admitted the move by accident.
            let d = omega_distance(
                on.eta.as_slice(),
                off.eta.as_slice(),
                &m.default_params.omega,
            );
            assert!(d > 1.0, "{} {id}: modes {d} apart in Ω⁻¹", cfg.name);
        }
    }
}

/// A unimodal subject is bit-identical with the screen on: every descent reconverges to the
/// incumbent's mode and none comes back lower by more than `1e-9`. (A loose `inner_tol`,
/// where one does, is the distance gate's test below.)
#[test]
fn the_screen_leaves_a_unimodal_subject_bit_identical() {
    for cfg in &CONFIGS {
        let (m, opts) = model(cfg.omega);
        for subj in controls() {
            let off = solve(&m, &opts, &subj, 0, 1, None);
            let on = solve(&m, &opts, &subj, 100, 1, None);
            assert_eq!(bits(&on.eta), bits(&off.eta), "{} {}", cfg.name, subj.id);
            assert_eq!(
                on.nll.to_bits(),
                off.nll.to_bits(),
                "{} {}",
                cfg.name,
                subj.id
            );
            assert_eq!(on.converged, off.converged);
            assert!(!on.screen_moved, "{} {}", cfg.name, subj.id);
        }
    }
}

/// The screen belongs to the cold multi-start: a warm start skips it, and
/// `inner_restarts = 0` turns it off with the rest of the multi-start.
#[test]
fn the_screen_runs_only_on_a_cold_multistart() {
    let cfg = &CONFIGS[0];
    let (m, opts) = model(cfg.omega);
    let subj = flip_flop_subject("S1", cfg);
    let trapped = solve(&m, &opts, &subj, 0, 1, None);

    let warm = solve(&m, &opts, &subj, 100, 1, Some(trapped.eta.as_slice()));
    assert!(!warm.screen_moved);
    assert!(
        (warm.nll - cfg.trapped).abs() < 1e-4,
        "a warm start from the trapped mode stays there: {}",
        warm.nll
    );

    let no_restarts = solve(&m, &opts, &subj, 100, 0, None);
    assert!(!no_restarts.screen_moved);
    assert_eq!(
        bits(&no_restarts.eta),
        bits(&solve(&m, &opts, &subj, 0, 0, None).eta)
    );
    assert!((no_restarts.nll - cfg.trapped).abs() < 1e-4);

    // Straddle: the same cold solve with restarts on does move, so neither result above is
    // a screen that could not have moved this subject anyway.
    assert!(solve(&m, &opts, &subj, 100, 1, None).screen_moved);
}

/// The population pass gives every subject the same EBE on 1 thread and on 8, and in
/// either data order: the Sobol stream is keyed by the subject's ID, not by the worker or
/// the subject's index.
#[test]
fn the_screen_is_reproducible_across_threads_and_data_order() {
    let cfg = &CONFIGS[0];
    let (m, opts) = model(cfg.omega);
    let mut subjects: Vec<Subject> = (1..=6)
        .map(|i| flip_flop_subject(&format!("S{i}"), cfg))
        .collect();
    subjects.extend(controls());
    let pop = |subjects: Vec<Subject>| Population {
        subjects,
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let policy = InnerSolvePolicy {
        screen: 100,
        ..InnerSolvePolicy::for_stage(&opts, &m)
    };
    let run = |population: &Population, threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .stack_size(32 * 1024 * 1024)
            .build()
            .unwrap();
        let (etas, _, _, _, moved) = pool.install(|| {
            run_inner_loop_warm_map(
                &m,
                population,
                &m.default_params,
                opts.inner_maxiter,
                opts.inner_tol,
                None,
                None,
                opts.min_obs_for_convergence_check as usize,
                opts.inner_restarts,
                policy,
                None,
                |s, ebe| (s.id.clone(), ebe.screen_moved),
            )
        });
        let mut by_id: Vec<(String, bool, Vec<u64>)> = moved
            .into_iter()
            .zip(&etas)
            .map(|((id, mv), e)| (id, mv, bits(e)))
            .collect();
        by_id.sort();
        by_id
    };

    let forward = pop(subjects.clone());
    let one = run(&forward, 1);
    assert_eq!(run(&forward, 8), one, "1 thread vs 8");
    subjects.reverse();
    assert_eq!(run(&pop(subjects), 8), one, "data order reversed");
    // Every flip-flop subject moved and neither control did, so the comparison above is
    // over subjects the screen actually changed.
    for (id, mv, _) in &one {
        assert_eq!(*mv, id.starts_with('S'), "{id}");
    }
}

#[test]
fn the_stage_policy_carries_the_screen_size() {
    let (m, opts) = model(CONFIGS[0].omega);
    assert_eq!(opts.inner_screen, 100, "on by default");
    for n in [0, 7] {
        let stage = FitOptions {
            inner_screen: n,
            ..opts.clone()
        };
        assert_eq!(InnerSolvePolicy::for_stage(&stage, &m).screen, n);
    }
    assert_eq!(
        InnerSolvePolicy::default().screen,
        0,
        "the historical solve"
    );
}

/// End to end through `fit()`: the final re-solve's warning names the flip-flop subject
/// and neither control, and `inner_screen = 0` silences it on the same data. Both on an
/// estimating fit and on `maxiter = 0`, whose evaluation-only solve is the other place a
/// fit reports its EBEs from.
#[test]
fn a_fit_names_the_subjects_the_screen_moved() {
    let cfg = &CONFIGS[0];
    let (m, opts) = model(cfg.omega);
    let mut subjects = vec![flip_flop_subject("S1", cfg)];
    subjects.extend(controls());
    let population = Population {
        subjects,
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let screen_warning = |screen: usize, outer_maxiter: usize| {
        let options = FitOptions {
            inner_screen: screen,
            outer_maxiter,
            run_covariance_step: false,
            threads: Some(1),
            ..opts.clone()
        };
        crate::api::fit(&m, &population, &m.default_params, &options)
            .expect("fit")
            .warnings
            .into_iter()
            .find(|w| w.contains("Sobol screen"))
    };
    for outer_maxiter in [2, 0] {
        let w = screen_warning(100, outer_maxiter)
            .unwrap_or_else(|| panic!("maxiter = {outer_maxiter}: the screen moved S1"));
        assert!(
            w.contains("of 1 subject (S1)"),
            "maxiter = {outer_maxiter}: {w}"
        );
        assert!(
            screen_warning(0, outer_maxiter).is_none(),
            "maxiter = {outer_maxiter}: no screen ran"
        );
    }
}

/// The Ω⁻¹ distance gate, on its own: under a loose `inner_tol` the incumbent and the
/// screen's descents all stop short inside the one basin, at slightly different points,
/// and some descents stop lower than the incumbent by more than `1e-9`. The objective gate
/// alone would take them and move a unimodal subject's EBE; the distance gate turns them
/// away. At the default `inner_tol` no descent comes back lower on these subjects, which
/// is why `the_screen_leaves_a_unimodal_subject_bit_identical` cannot see this gate.
#[test]
fn the_distance_gate_keeps_a_loosely_converged_unimodal_subject() {
    for cfg in &CONFIGS {
        let (m, opts) = model(cfg.omega);
        let loose = FitOptions {
            inner_tol: 1e-2,
            ..opts.clone()
        };
        let subj = &controls()[0];
        let off = solve(&m, &loose, subj, 0, 1, None);
        SCREEN_SAME_BASIN_REJECTS.with(|n| n.set(0));
        let on = solve(&m, &loose, subj, 100, 1, None);
        // Straddle: the gate actually turned a lower descent away (3 of 3 on macOS arm64
        // in both configurations), so the bit-identity below is the gate's doing.
        let rejects = SCREEN_SAME_BASIN_REJECTS.with(|n| n.get());
        assert!(
            rejects > 0,
            "{}: no screen descent came back lower within the basin",
            cfg.name
        );
        assert_eq!(bits(&on.eta), bits(&off.eta), "{}", cfg.name);
        assert_eq!(on.nll.to_bits(), off.nll.to_bits(), "{}", cfg.name);
        assert!(!on.screen_moved, "{}", cfg.name);
    }
}

/// Raw L2 gradient of the individual objective at the returned η̂.
fn grad_norm_at(m: &CompiledModel, subj: &Subject, eta: &DVector<f64>) -> f64 {
    let p = &m.default_params;
    let g = analytic_eta_nll_gradient(m, subj, &p.theta, eta.as_slice(), &p.omega, &p.sigma.values)
        .expect("the flip-flop model has an analytic inner gradient");
    let n = g.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(n.is_finite(), "gradient at {eta:?} is not finite");
    n
}

/// Above this raw gradient norm a returned η̂ is not a mode. The largest norm any
/// *converged* solve in the two sweeps below returns is 5.6e-4 (Linux x86_64, the axis
/// sweep's incumbent at `inner_maxiter = 42`), so the bound has ~18× headroom. The descents
/// the old `ebe_converged || ok` mislabelled stop at up to 3.5e-1 (screen) and 2.3e2 (axis
/// scan); seven screen cells and nine axis cells lie above the bound.
const NOT_A_MODE_GRAD: f64 = 1e-2;

/// Asserts the #2100 contract on one solve, and returns whether this cell straddles it:
/// the incumbent (`off`, the same cold solve without the block under test) converged, a
/// candidate from the block replaced it, and that candidate's own descent did not converge.
/// The old `ebe_converged || ok` reported exactly those cells converged.
fn check_converged_flag(
    what: &str,
    m: &CompiledModel,
    subj: &Subject,
    off: &EbeResult,
    on: &EbeResult,
) -> bool {
    let g = grad_norm_at(m, subj, &on.eta);
    assert!(
        !on.converged || g < NOT_A_MODE_GRAD,
        "{what}: η̂ reported converged at |∇| = {g:.3e} (nll {:.6}, incumbent {:.6})",
        on.nll,
        off.nll
    );
    off.converged && on.nll + 1e-9 < off.nll && !on.converged
}

/// #2100, the screen's acceptance block: a screen descent that stops at `inner_maxiter`
/// below a converged incumbent replaces it, and the returned EBE must then carry the
/// descent's own flag, not the incumbent's. Config 1 at a low `inner_maxiter` is where the
/// incumbent (the trapped mode) already converges and the screen's winning descent has not
/// yet met the deeper mode's gradient test.
#[test]
fn a_screen_descent_that_stopped_short_is_not_reported_converged() {
    let cfg = &CONFIGS[0];
    let (m, opts) = model(cfg.omega);
    let mut straddles = 0;
    for tol in [1e-5, 1e-3] {
        for max_iter in 5..=15 {
            let o = FitOptions {
                inner_maxiter: max_iter,
                inner_tol: tol,
                ..opts.clone()
            };
            for id in ["S1", "S2", "S3", "S4"] {
                let subj = flip_flop_subject(id, cfg);
                let off = solve(&m, &o, &subj, 0, 1, None);
                let on = solve(&m, &o, &subj, 100, 1, None);
                let what = format!("{id} inner_tol = {tol}, inner_maxiter = {max_iter}");
                if check_converged_flag(&what, &m, &subj, &off, &on) {
                    assert!(on.screen_moved, "{what}: the screen made the move");
                    straddles += 1;
                }
            }
        }
    }
    // Without a straddling cell the bound above would pass vacuously.
    assert!(
        straddles > 0,
        "no cell had a failed screen descent replace a converged incumbent"
    );
    // The other side: at the default `inner_maxiter` the same descent converges.
    let subj = flip_flop_subject("S1", cfg);
    let full = solve(&m, &opts, &subj, 100, 1, None);
    assert!(
        full.screen_moved && full.converged,
        "default inner_maxiter converges"
    );
}

/// #2100, the axis scan's acceptance block. A per-row covariate the model never reads makes
/// the subject a scan-all subject (`has_tv_covariates`), so the `±2·sd` seeds run on every
/// coordinate; with the screen off, a seed is the only thing that can move the EBE.
#[test]
fn an_axis_seed_that_stopped_short_is_not_reported_converged() {
    let cfg = &CONFIGS[0];
    let (m, opts) = model(cfg.omega);
    let mut subj = flip_flop_subject("S1", cfg);
    subj.obs_covariates = TIMES
        .iter()
        .map(|_| std::collections::HashMap::from([("UNUSED".to_string(), 1.0)]))
        .collect();
    assert!(subj.has_tv_covariates());
    let mut straddles = 0;
    for max_iter in 40..=60 {
        let o = FitOptions {
            inner_maxiter: max_iter,
            inner_tol: 1e-5,
            ..opts.clone()
        };
        let off = solve(&m, &o, &subj, 0, 0, None);
        let on = solve(&m, &o, &subj, 0, 1, None);
        assert!(!on.screen_moved);
        let what = format!("inner_maxiter = {max_iter}");
        if check_converged_flag(&what, &m, &subj, &off, &on) {
            straddles += 1;
        }
    }
    assert!(
        straddles > 0,
        "no cell had a failed axis seed replace a converged incumbent"
    );
    let full = solve(&m, &opts, &subj, 0, 1, None);
    assert!(
        (full.nll - cfg.deeper).abs() < 1e-4 && full.converged,
        "default inner_maxiter: the seed converges to the deeper mode, nll {}",
        full.nll
    );
}
