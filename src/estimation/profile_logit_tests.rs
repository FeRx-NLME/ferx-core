//! A profiled block read through a logit link (#2111): `y = inv_logit(c × block + rest)`.
//!
//! Fixture **B2** (`nonmem_anchor/profile_logit_b2.ferx`, the NONMEM anchor of
//! `tests/profiled_logit_levels.rs`): an observed proportion per study, visit and arm with
//! the placebo block and the study η inside the link. The prediction is nonlinear in both,
//! so the GLS matrix the affine route uses is only quasi-Newton here, and the solve takes
//! the adjoint's exact Hessian ([`ProfileCtx::exact_hessian`]).
//!
//! Every number below is measured on macOS arm64 unless marked; the Linux run of the same
//! tests agreed in every printed digit.

use super::*;

const B2: &str = "nonmem_anchor/profile_logit_b2.ferx";
const B2_DATA: &str = "nonmem_anchor/profile_logit_b2.csv";

fn b2(max_study: Option<usize>, edit: impl Fn(String) -> String) -> crate::api::PreparedRun {
    prep(B2, B2_DATA, max_study, edit)
}

/// B2 with one subject per arm (`ID = 2·STUDY − 1 + TRT`) and study 1's treated arm missing
/// its `TIME = 0` row. That subject reads `t4`, `t8` and the dependent `t12` but not `t0`,
/// which its control arm still reads (so the level exists), and reads the dependent through
/// `t0` as its slack: a level whose `moves` carry a `−1`. On B2 itself every subject reads
/// all four visits and keeps the θ form, so no move has one.
fn b2_slack() -> crate::api::PreparedRun {
    let csv = std::fs::read_to_string(root().join(B2_DATA)).unwrap();
    let mut lines = csv.lines();
    let header = lines.next().unwrap();
    assert!(
        header.starts_with("ID,STUDY,TIME,DV,CMT,EVID,MDV,TRT,"),
        "the data moved"
    );
    let kept: String = std::iter::once(format!("{header}\n"))
        .chain(lines.filter_map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            let (study, trt): (usize, usize) = (f[1].parse().unwrap(), f[7].parse().unwrap());
            (!(study == 1 && trt == 1 && f[2] == "0"))
                .then(|| format!("{},{}\n", 2 * study - 1 + trt, f[1..].join(",")))
        }))
        .collect();
    assert_eq!(
        kept.lines().count(),
        csv.lines().count() - 1,
        "the data moved"
    );
    let data = temp("b2_slack.csv");
    std::fs::write(&data, kept).unwrap();
    crate::api::prepare_run(
        root().join(B2).to_str().unwrap(),
        Some(data.to_str().unwrap()),
    )
    .unwrap_or_else(|e| panic!("{e}"))
}

fn laplace(t: String) -> String {
    assert!(t.contains("  method     = focei\n"), "the fixture moved");
    t.replace("  method     = focei\n", "  method     = laplace\n")
}

fn gradient_fd(t: String) -> String {
    t.replace("[fit_options]\n", "[fit_options]\n  gradient   = fd\n")
}

/// The solve from the inits with its work, and its levels.
fn solve_b2(p: &crate::api::PreparedRun, o: &FitOptions) -> (ProfileSolve, ProfileCounts) {
    let layout = layout_of(p);
    let c = ctx(p, &layout, o);
    let mut counts = ProfileCounts::default();
    let s = c
        .solve_counted(
            &p.init_params,
            &ProfileWarm::from_template(&layout, &p.init_params),
            &mut counts,
        )
        .unwrap_or_else(|e| panic!("the solve failed: {e:?}"));
    (s, counts)
}

/// [`FORCE_PROFILE_MATRIX_DECLINE`] for a scope, reset on drop, a panic included.
struct MatrixDecline;

impl MatrixDecline {
    /// Every subject's exact matrix declined.
    fn on() -> Self {
        Self::subject(usize::MAX)
    }

    /// Subject `i`'s exact matrix declined, every other one's served.
    fn subject(i: usize) -> Self {
        FORCE_PROFILE_MATRIX_DECLINE.with(|c| c.set(Some(i)));
        MatrixDecline
    }
}

impl Drop for MatrixDecline {
    fn drop(&mut self) {
        FORCE_PROFILE_MATRIX_DECLINE.with(|c| c.set(None));
    }
}

/// The worst level gap between two solves, in the first one's conditional SEs.
fn beta_gap_se(layout: &ProfileLayout, a: &ProfileSolve, b: &ProfileSolve) -> f64 {
    layout
        .coords
        .iter()
        .zip(&a.se_scale)
        .map(|(&k, se)| {
            let d = ((a.params.theta[k] - b.params.theta[k]) / se).abs();
            assert!(d.is_finite(), "level {k}: {d}");
            d
        })
        .fold(0.0, f64::max)
}

/// The step of the parity test's differenced reference, in conditional SEs.
const REF_STEP_SE: f64 = 3e-2;

/// Iterations on a Newton matrix with `dη̂/dβ` in it, exact or differenced; measured 3 on B2
/// under FOCEI and Laplace. The held GLS matrix takes 7 here, linear at ~0.1 per step, and
/// on a fixture with probabilities 0.1–0.9 does not converge in [`MAX_PROFILE_ITER`].
const NEWTON_MAX_ITER: usize = 4;

/// N-link and N-decline. The block is linked, every group says so, and the solve takes the
/// adjoint's exact matrix in at most [`NEWTON_MAX_ITER`] iterations, every matrix exact.
/// With the exact matrix declined it takes the second differences of the re-solved term,
/// counted as such, in as few iterations and to the same levels: the pair straddles the
/// matrix choice in one test. Under FOCEI and Laplace, whose objectives differ by 0.25 OFV
/// here.
///
/// Dies when either arm takes the held matrix (its iterations), when a decline is not
/// counted (the counts), and when either matrix is the wrong function (the levels).
#[test]
fn a_logit_linked_block_takes_the_exact_newton_matrix_and_differences_on_a_decline() {
    for (name, edit) in [
        ("focei", (|t| t) as fn(String) -> String),
        ("laplace", laplace),
    ] {
        let p = b2(None, edit);
        let o = options(&p);
        let layout = layout_of(&p);
        assert_eq!(layout.groups.len(), 10, "{name}: a group per study");
        assert!(layout.groups.iter().all(|g| g.linked), "{name}");
        let (exact, ce) = solve_b2(&p, &o);
        let (fd, cf) = {
            let _decline = MatrixDecline::on();
            solve_b2(&p, &o)
        };
        eprintln!(
            "{name}: exact {} it {ce:?}; declined {} it {cf:?}; ofv gap {:.3e}, levels {:.3e} SE",
            exact.iterations,
            fd.iterations,
            fd.ofv - exact.ofv,
            beta_gap_se(&layout, &exact, &fd)
        );
        assert!(
            exact.iterations <= NEWTON_MAX_ITER,
            "{name}: {}",
            exact.iterations
        );
        assert!(
            ce.exact_matrices > 0 && ce.fd_matrices == 0,
            "{name}: {ce:?}"
        );
        assert_eq!(ce.fd_gradients, 0, "{name}: {ce:?}");
        assert!(
            fd.iterations <= NEWTON_MAX_ITER,
            "{name}: {}",
            fd.iterations
        );
        assert!(
            cf.exact_matrices == 0 && cf.fd_matrices > 0,
            "{name}: {cf:?}"
        );
        assert!((fd.ofv - exact.ofv).abs() < 1e-8, "{name}");
        assert!(beta_gap_se(&layout, &exact, &fd) < 1e-6, "{name}");
    }
}

/// Off the adjoint's route (`gradient = fd`) the linked block takes the differenced matrix
/// with the reconverged gradient, in as few iterations, to the levels the exact route
/// reaches. Dies when the exact matrix is taken off the route (the counts), when the held
/// matrix is (the iterations), or when the linked gradient step is the affine one (the gap
/// to the exact route's levels grows with the step's truncation error).
#[test]
fn off_the_adjoint_route_a_linked_block_takes_the_differenced_matrix() {
    let p = b2(None, gradient_fd);
    let o = options(&p);
    assert!(!profile_gradient_is_analytic(&p.parsed.model, &o));
    let (fd, c) = solve_b2(&p, &o);
    assert!(c.exact_matrices == 0 && c.fd_matrices > 0, "{c:?}");
    assert!(c.adjoint_gradients == 0 && c.fd_gradients > 0, "{c:?}");
    let pa = b2(None, |t| t);
    let (exact, _) = solve_b2(&pa, &options(&pa));
    let layout = layout_of(&p);
    eprintln!(
        "gradient = fd: {} it, ofv gap {:.3e}, levels {:.3e} SE",
        fd.iterations,
        fd.ofv - exact.ofv,
        beta_gap_se(&layout, &exact, &fd)
    );
    assert!(fd.iterations <= NEWTON_MAX_ITER, "{}", fd.iterations);
    assert!((fd.ofv - exact.ofv).abs() < 1e-8);
    assert!(beta_gap_se(&layout, &exact, &fd) < 1e-6);
}

/// N-affine. An affine block is not linked, so the solve has no matrix choice: no matrix
/// is counted either way and F1 still lands in one step. Dies when `linked` is read as
/// `true` for every block.
#[test]
fn an_affine_block_is_not_linked_and_counts_no_matrix_choice() {
    let p = prep(F1, F1_DATA, None, |t| t);
    let layout = layout_of(&p);
    assert!(layout.groups.iter().all(|g| !g.linked));
    let o = options(&p);
    let c = ctx(&p, &layout, &o);
    let mut counts = ProfileCounts::default();
    let s = c
        .solve_counted(
            &p.init_params,
            &ProfileWarm::from_template(&layout, &p.init_params),
            &mut counts,
        )
        .unwrap();
    assert_eq!(s.iterations, 1);
    assert_eq!((counts.exact_matrices, counts.fd_matrices), (0, 0));
}

/// The `verbose` summary names the matrix counts on a linked block and only there. Both
/// sides of the gate in one test, so forcing either branch reddens it.
#[test]
fn the_profile_work_summary_names_the_matrix_counts_only_on_a_linked_block() {
    let affine = ProfileCounts {
        newton_steps: 3,
        resolves: 40,
        adjoint_gradients: 30,
        fd_gradients: 0,
        ..Default::default()
    };
    let linked = ProfileCounts {
        exact_matrices: 28,
        fd_matrices: 2,
        damped_matrices: 3,
        ..affine
    };
    let base = "Profile solves of the outer optimizer: 5 evaluations (0 solve failures, their \
                work counted), 3 Newton steps summed over groups, 40 EBE re-solves, level \
                gradients 30 adjoint / 0 finite-difference";
    assert_eq!(profile_work_summary(5, 0, &affine), base);
    assert_eq!(
        profile_work_summary(5, 0, &linked),
        format!("{base}, logit-linked Newton matrices 28 exact / 2 finite-difference, 3 damped")
    );
}

/// D3-link. On two studies the solve is the minimum over the levels (a dense FD Newton on
/// the cold re-solved objective), and the frozen-EBE shortcut lands above it, in one test,
/// under FOCEI and Laplace. Measured: solve − oracle 2.0e-10 / 2.9e-10, frozen − oracle
/// +5.2e-4 / +1.4e-6. The straddle bound is per method, a third of each: under Laplace the
/// frozen shortcut holds `½log|H|` at η̂ too and lands much closer, still four decades above
/// the solve's error. Dies when the total gradient is replaced by the held-η̂ partial (the
/// solve would land on the frozen point).
#[test]
fn the_linked_solve_is_the_minimum_and_the_frozen_shortcut_is_not() {
    for (name, edit, straddle) in [
        ("focei", (|t| t) as fn(String) -> String, 1.7e-4),
        ("laplace", laplace, 4.7e-7),
    ] {
        let p = b2(Some(2), edit);
        let layout = layout_of(&p);
        let o = options(&p);
        let c = ctx(&p, &layout, &o);
        let warm = ProfileWarm::from_template(&layout, &p.init_params);
        let s = c.solve(&p.init_params, &warm).unwrap();
        let (omin, _) = oracle_min(&c, &p.init_params, &warm.beta);
        let frozen = frozen_fixed_point(&c, &p.init_params, &warm.beta);
        let beta: Vec<f64> = layout.coords.iter().map(|&k| s.params.theta[k]).collect();
        let cold = total_ofv(&c, &p.init_params, &beta);
        eprintln!(
            "{name}: solve - oracle {:.3e}, frozen - oracle {:.3e}, solve - cold {:.3e}",
            s.ofv - omin,
            frozen - omin,
            s.ofv - cold
        );
        assert!((s.ofv - omin).abs() < 1e-8, "{name}");
        assert!((s.ofv - cold).abs() < 1e-8, "{name}: D0");
        assert!(frozen - omin > straddle, "{name}: the straddle collapsed");
    }
}

/// The exact Newton matrix against the second differences of the re-solved objective
/// ([`ProfileCtx::resolved_hessian`], the decline path's matrix), per subject at the inits
/// and at the solved point, under FOCEI and Laplace: the dual-vs-FD parity of the matrix
/// this PR adds. The reference shares nothing with the adjoint: it is `resolve` at shifted
/// levels, the EBE re-found at each point. The held GLS matrix is measured beside them, so
/// the fixture is shown to tell a matrix without `dη̂/dβ` apart. The third arm is
/// [`b2_slack`], whose study 1 reads a dependent level through a slack, so a move's `−1`
/// is read; the arm asserts that it is.
///
/// Dies when the matrix is read along the wrong moves (the slack's `−1` dropped: the third
/// arm), from the wrong objective (Laplace), or without `dη̂/dβ`.
///
/// The reference steps at [`REF_STEP_SE`], not the production [`LINKED_HESSIAN_STEP_SE`]:
/// the gap is the re-solves' noise over `h²`, not truncation. On the slack arm it was
/// 3.8e-5 at `1e-2` and 4.2e-4 at `3e-3` (macOS arm64); a Newton matrix needs none of
/// those digits, a parity bound does.
#[test]
fn the_exact_newton_matrix_matches_second_differences_of_the_resolved_objective() {
    let arms: [(&str, fn() -> crate::api::PreparedRun); 3] = [
        ("focei", || b2(None, |t| t)),
        ("laplace", || b2(None, laplace)),
        ("focei, a slack read", b2_slack),
    ];
    for (name, build) in arms {
        let p = build();
        let layout = layout_of(&p);
        let slack = layout
            .reads
            .iter()
            .flatten()
            .any(|r| r.moves.iter().any(|&(_, c)| c < 0.0));
        assert_eq!(slack, name.contains("slack"), "{name}: the fixture moved");
        let o = options(&p);
        let c = ctx(&p, &layout, &o);
        let sol = c
            .solve(
                &p.init_params,
                &ProfileWarm::from_template(&layout, &p.init_params),
            )
            .unwrap();
        let zero = DVector::zeros(c.model.n_eta);
        let mut worst = 0.0_f64;
        let mut held_gap = 0.0_f64;
        for params in [p.init_params.clone(), sol.params.clone()] {
            let x = pack_params(&params);
            let pt = crate::sens::ode_provider::adjoint::PackedPoint::new(&params, &x).unwrap();
            for i in 0..p.population.subjects.len() {
                let s = c.resolve(i, &params, &zero);
                let k = layout.reads[i].len();
                let se: Vec<f64> = layout.reads[i]
                    .iter()
                    .map(|r| sol.level_scale[r.level])
                    .collect();
                let exact = c.exact_hessian(i, &pt, &s).expect("the adjoint serves B2");
                let st: Vec<f64> = se.iter().map(|v| REF_STEP_SE * v).collect();
                let fd = c.resolved_hessian(i, &params, &s, &st);
                let held = c.held_hessian(i, &params, &s, &se);
                for a in 0..k {
                    for b in 0..k {
                        // In units of the level's own curvature, `se_a · se_b`.
                        let scale = se[a] * se[b];
                        let e = ((exact[(a, b)] - fd[(a, b)]) * scale).abs();
                        assert!(e.is_finite(), "{name} subject {i}: {e}");
                        worst = worst.max(e);
                        held_gap = held_gap.max(((held[(a, b)] - fd[(a, b)]) * scale).abs());
                    }
                }
            }
        }
        // Measured at REF_STEP_SE 1.2e-5 (FOCEI, Laplace) / 2.7e-5 (slack); held 0.14 / 0.56.
        eprintln!("{name}: exact - FD {worst:.3e}, held - FD {held_gap:.3e} (scaled)");
        assert!(worst < 1e-4, "{name}: {worst:.3e}");
        assert!(
            held_gap > 10.0 * worst,
            "{name}: the fixture cannot tell the two apart"
        );
    }
}

/// G2-link. The adjoint level gradient through the link against reconverged central
/// differences at `h = 1e-3` SE, at the inits and at the solved point. Measured 2.8e-8 SE
/// (2.8e-6 at `FD_STEP_SE = 1e-2`, the truncation term). Dies when the link's derivative is
/// dropped in the recorder.
#[test]
fn the_adjoint_level_gradient_through_the_link_matches_reconverged_differences() {
    let p = b2(None, |t| t);
    let layout = layout_of(&p);
    let o = options(&p);
    let c = ctx(&p, &layout, &o);
    let sol = c
        .solve(
            &p.init_params,
            &ProfileWarm::from_template(&layout, &p.init_params),
        )
        .unwrap();
    let zero = DVector::zeros(c.model.n_eta);
    let (mut fine, mut coarse) = (0.0_f64, 0.0_f64);
    for params in [p.init_params.clone(), sol.params.clone()] {
        let x = pack_params(&params);
        let pt = crate::sens::ode_provider::adjoint::PackedPoint::new(&params, &x).unwrap();
        for i in 0..p.population.subjects.len() {
            let s = c.resolve(i, &params, &zero);
            let se: Vec<f64> = layout.reads[i]
                .iter()
                .map(|r| sol.level_scale[r.level])
                .collect();
            let ga = c.total_gradient_adjoint(i, &pt, &s).expect("served");
            for (step, w) in [(1e-3, &mut fine), (FD_STEP_SE, &mut coarse)] {
                let st: Vec<f64> = se.iter().map(|v| step * v).collect();
                let gf = c.total_gradient_fd(i, &params, &s, &st);
                for ((a, b), v) in ga.iter().zip(&gf).zip(&se) {
                    let e = ((a - b) * v).abs();
                    assert!(e.is_finite(), "subject {i}: {e}");
                    *w = w.max(e);
                }
            }
        }
    }
    eprintln!("adjoint - FD: {fine:.3e} SE at h = 1e-3, {coarse:.3e} SE at h = FD_STEP_SE");
    assert!(fine < 3e-7, "{fine:.3e}");
}

/// Under three-node quadrature, at both anchors, the linked solve is the minimum over the
/// levels on two studies, and its matrices are counted. Dies when the solve ignores the
/// quadrature objective (the minimum moves) or the matrix route is not counted.
#[test]
fn a_linked_block_profiles_under_quadrature() {
    for (name, edit, anchor) in AGQ3 {
        let p = b2(Some(2), edit);
        assert_agq3(name, &p, anchor);
        let layout = layout_of(&p);
        let o = options(&p);
        let c = ctx(&p, &layout, &o);
        let warm = ProfileWarm::from_template(&layout, &p.init_params);
        let mut counts = ProfileCounts::default();
        let s = c.solve_counted(&p.init_params, &warm, &mut counts).unwrap();
        let (omin, _) = oracle_min(&c, &p.init_params, &warm.beta);
        eprintln!(
            "agq3 {name}: {} it, {counts:?}, solve - oracle {:.3e}",
            s.iterations,
            s.ofv - omin
        );
        assert!((s.ofv - omin).abs() < 1e-8, "{name}");
        assert!(
            counts.exact_matrices > 0 && counts.fd_matrices == 0,
            "{name}: {counts:?}"
        );
        // The exact matrix under quadrature is that objective's: against the differenced
        // one, which re-solves the quadrature term itself.
        let zero = DVector::zeros(c.model.n_eta);
        let x = pack_params(&s.params);
        let pt = crate::sens::ode_provider::adjoint::PackedPoint::new(&s.params, &x).unwrap();
        let mut worst = 0.0_f64;
        for i in 0..p.population.subjects.len() {
            let st = c.resolve(i, &s.params, &zero);
            let se: Vec<f64> = layout.reads[i]
                .iter()
                .map(|r| s.level_scale[r.level])
                .collect();
            let exact = c
                .exact_hessian(i, &pt, &st)
                .expect("served under quadrature");
            let steps: Vec<f64> = se.iter().map(|v| REF_STEP_SE * v).collect();
            let fd = c.resolved_hessian(i, &s.params, &st, &steps);
            for a in 0..se.len() {
                for b in 0..se.len() {
                    let e = ((exact[(a, b)] - fd[(a, b)]) * se[a] * se[b]).abs();
                    assert!(e.is_finite(), "{name} subject {i}");
                    worst = worst.max(e);
                }
            }
        }
        eprintln!("agq3 {name}: exact - FD {worst:.3e} (scaled)");
        assert!(worst < 1e-4, "{name}: {worst:.3e}");
    }
}

/// A group where the adjoint serves one subject's matrix and declines the other's: the
/// exact part and the differenced part are summed into one matrix, and the first pass's
/// differences are sized by the held pass. On [`b2_slack`] (two subjects per study),
/// declining subject 0 leaves study 1 mixed; the solve takes as few iterations and lands
/// on the levels of the undeclined one. Dies when the mixed arm drops either part (a
/// subject's curvature missing from the group: the iterations or the levels), or does not
/// count both.
#[test]
fn a_group_mixing_exact_and_differenced_matrices_solves_to_the_same_levels() {
    let p = b2_slack();
    let layout = layout_of(&p);
    let g0 = &layout.groups[layout.local[layout.reads[0][0].level].0];
    assert!(
        g0.subjects.len() >= 2 && g0.subjects.contains(&0),
        "the fixture moved"
    );
    let o = options(&p);
    let (all, _) = solve_b2(&p, &o);
    let (mixed, c) = {
        let _decline = MatrixDecline::subject(0);
        solve_b2(&p, &o)
    };
    eprintln!(
        "mixed: {} it {c:?}; ofv gap {:.3e}, levels {:.3e} SE",
        mixed.iterations,
        mixed.ofv - all.ofv,
        beta_gap_se(&layout, &all, &mixed)
    );
    assert!(c.exact_matrices > 0 && c.fd_matrices > 0, "{c:?}");
    assert!(mixed.iterations <= NEWTON_MAX_ITER, "{}", mixed.iterations);
    assert!((mixed.ofv - all.ofv).abs() < 1e-8);
    assert!(beta_gap_se(&layout, &all, &mixed) < 1e-6);
}

/// Through the link the Newton matrix is the true Hessian, and away from the optimum it is
/// indefinite. At TVL 2.5, EMAX −2 (two studies) the undamped solve failed `Singular`;
/// damped ([`damped_to_pd`]) it converges, and its point is the minimum of the cold
/// re-solved objective (the oracle started there, where its matrix is positive definite).
/// The second point, the inits, needs no damping, so the pair straddles the damping gate.
/// Dies when the damping is removed (the first point fails) or applied where the matrix is
/// already positive definite (the inits' count).
#[test]
fn an_indefinite_newton_matrix_is_damped_and_the_solve_still_finds_the_minimum() {
    let p = b2(Some(2), |t| t);
    let layout = layout_of(&p);
    let o = options(&p);
    let c = ctx(&p, &layout, &o);
    let mut far = p.init_params.clone();
    far.theta[0] = 2.5;
    far.theta[1] = -2.0;
    for (name, params, damped) in [("far", &far, true), ("inits", &p.init_params, false)] {
        let mut counts = ProfileCounts::default();
        let s = c
            .solve_counted(
                params,
                &ProfileWarm::from_template(&layout, params),
                &mut counts,
            )
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let beta: Vec<f64> = layout.coords.iter().map(|&k| s.params.theta[k]).collect();
        let (omin, _) = oracle_min(&c, params, &beta);
        eprintln!(
            "{name}: {} it, {} damped, solve - oracle {:.3e}",
            s.iterations,
            counts.damped_matrices,
            s.ofv - omin
        );
        assert_eq!(counts.damped_matrices > 0, damped, "{name}: {counts:?}");
        assert!((s.ofv - omin).abs() < 1e-8, "{name}");
    }
}
