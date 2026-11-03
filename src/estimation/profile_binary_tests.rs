//! A profiled block read by a `[binary_model] logit` predictor (#2111, binary half).
//!
//! Fixture **B1** (`nonmem_anchor/profile_logit_b1.ferx`, the NONMEM anchor A-B1 of
//! `tests/binary_logit_levels.rs`), profiled: 2400 Bernoulli rows, ten studies (one subject
//! each) × four visits × two arms, `logit p = TVL + PLACEBO[STUDY, TIME] + η + EMAX·TRT`.
//! The predictor's logit link is implicit, so the block is linked and the solve takes
//! PR-1's exact Newton matrix ([`ProfileCtx::exact_hessian`]). A binary row's term is the
//! Laplace form under FOCE, FOCEI and `laplace` alike (#2437), and the adjoint records the
//! same term, so the linked machinery needs no change for binary rows; these tests pin that.
//!
//! B1 has one degenerate property, named where it matters: under `sum_to_zero_within`
//! every subject reads all of its study's visits, so a level shift is nearly orthogonal to
//! the study η (`dη̂/dβ ≈ 0`). The held GLS matrix therefore still converges here, and the
//! frozen-EBE shortcut lands within 4.4e-8 of the minimum. D3 uses [`b1_unbalanced`], where
//! η absorbs a shift.
//!
//! Every number below was measured on macOS arm64 and again on Linux aarch64
//! (`tools/linux-test.sh`), which agreed in every printed digit.

use super::*;

const B1: &str = "nonmem_anchor/profile_logit_b1.ferx";
const B1_DATA: &str = "nonmem_anchor/profile_logit_b1.csv";

/// The method lines B1 is profiled under. The three one-node methods are one objective on
/// binary rows; the two quadrature arms are another one (`n_agq = 3`, each at its own
/// anchor; the binary η-Hessian is exact, so the anchors agree).
const ARMS: [(&str, &str); 5] = [
    ("laplace", "  method     = laplace\n"),
    ("focei", "  method     = focei\n"),
    ("foce", "  method     = foce\n"),
    ("agq3 laplace", "  method     = laplace\n  n_agq      = 3\n"),
    ("agq3 focei", "  method     = focei\n  n_agq      = 3\n"),
];

/// B1's model text, its block declared `profile`, under `method` (an [`ARMS`] line).
fn b1_text(method: &str) -> String {
    let t = std::fs::read_to_string(root().join(B1)).unwrap();
    for from in ["(0.0, -10.0, 10.0)\n", "  method     = laplace\n"] {
        assert!(t.contains(from), "the fixture moved: {from:?}");
    }
    t.replace("(0.0, -10.0, 10.0)\n", "(0.0) profile\n")
        .replace("  method     = laplace\n", method)
}

fn b1(max_study: Option<usize>, method: &'static str) -> crate::api::PreparedRun {
    prep(B1, B1_DATA, max_study, move |_| b1_text(method))
}

/// B1 cut to `max_study` studies with one subject per arm (`ID = 2·STUDY − 1 + TRT`), the
/// treated arm keeping only `TIME ≤ 4`. That subject reads two of its study's four visits,
/// whose levels do not sum to zero, so its η absorbs a level shift and the frozen-EBE
/// shortcut lands measurably above the minimum. On B1 itself it lands within 4.4e-8.
fn b1_unbalanced(method: &'static str, max_study: usize) -> crate::api::PreparedRun {
    let csv = std::fs::read_to_string(root().join(B1_DATA)).unwrap();
    let mut lines = csv.lines();
    let header = lines.next().unwrap();
    assert!(
        header.starts_with("ID,STUDY,TIME,DV,CMT,EVID,MDV,TRT"),
        "the data moved"
    );
    let kept: String = std::iter::once(format!("{header}\n"))
        .chain(lines.filter_map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            let study: usize = f[1].parse().unwrap();
            let time: f64 = f[2].parse().unwrap();
            let trt: usize = f[7].parse().unwrap();
            (study <= max_study && (trt == 0 || time <= 4.0))
                .then(|| format!("{},{}\n", 2 * study - 1 + trt, f[1..].join(",")))
        }))
        .collect();
    let data = temp("b1_unbalanced.csv");
    std::fs::write(&data, kept).unwrap();
    let model = temp("b1_unbalanced.ferx");
    std::fs::write(&model, b1_text(method)).unwrap();
    crate::api::prepare_run(model.to_str().unwrap(), Some(data.to_str().unwrap()))
        .unwrap_or_else(|e| panic!("{e}"))
}

/// The final row of `nonmem_anchor/results/profile_logit_b1_laplace.ext`, B1's
/// `METHOD=COND LAPLACE LIKE` optimum: θ in order, ω, the objective.
fn nonmem_optimum() -> (Vec<f64>, f64, f64) {
    let text =
        std::fs::read_to_string(root().join("nonmem_anchor/results/profile_logit_b1_laplace.ext"))
            .unwrap();
    let mut names: Vec<String> = Vec::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.first() {
            Some(&"ITERATION") => names = f.iter().map(|s| s.to_string()).collect(),
            Some(&"-1000000000") => {
                let get = |n: &str| -> f64 {
                    f[names.iter().position(|x| x == n).unwrap()]
                        .parse()
                        .unwrap()
                };
                return (
                    (1..=32).map(|k| get(&format!("THETA{k}"))).collect(),
                    get("OMEGA(1,1)"),
                    get("OBJ"),
                );
            }
            _ => {}
        }
    }
    panic!("profile_logit_b1_laplace.ext: no final row");
}

/// L0. The profiled block's levels come from the binary records: each study reads its own
/// three free visits and its dependent one (`level_rows`, #2562), so B1 has a linked group
/// per study and 30 coordinates. Dies when `subject_levels` walks the Gaussian clock only,
/// as it did before #2111's binary half: no subject reads a level, and the layout is empty,
/// so the solve would converge over nothing.
#[test]
fn a_binary_endpoints_records_define_the_profiled_levels() {
    let p = b1(None, ARMS[0].1);
    let active = crate::api::active_levels(&p.parsed.model, &p.population);
    assert_eq!(active.len(), 1);
    for (i, levels) in active[0].per_subject.iter().enumerate() {
        let want: Vec<usize> = (4 * i..4 * i + 4).collect();
        assert_eq!(levels, &want, "study {}", i + 1);
    }
    let layout = layout_of(&p);
    assert_eq!(layout.coords.len(), 30);
    assert_eq!(layout.groups.len(), 10, "a group per study");
    assert!(layout.groups.iter().all(|g| g.linked));
}

/// The same walk feeds the unprofiled `W_COV_RSR_ONE_SUBJECT_LEVELS` count. On B1 (one
/// subject per study, unprofiled), every one of the 40 levels is read by its study alone,
/// so an `rsr` step names them; under `r` there is no note. Before #2111's binary half the
/// count saw no binary record and the note was silent. Dies when the count walks the
/// Gaussian clock only.
#[test]
fn the_rsr_one_subject_note_counts_binary_records() {
    let p = prep(B1, B1_DATA, None, |t| t);
    let sandwich = crate::api::rsr_one_subject_levels_warning(
        &p.parsed.model,
        &p.population,
        Some(crate::types::CovarianceMethod::Sandwich),
    )
    .expect("every level is read by one study");
    assert!(
        sandwich.contains("40 of the 40 levels of theta PLACEBO[...]"),
        "{sandwich}"
    );
    assert!(crate::api::rsr_one_subject_levels_warning(
        &p.parsed.model,
        &p.population,
        Some(crate::types::CovarianceMethod::Hessian),
    )
    .is_none());
}

/// L0, the counted form: `theta PLACEBO[4]` read as `PLACEBO[VISIT]` has no level
/// declaration of its own, so the records that read it are the endpoints whose predictor
/// reads `VISIT`. Study 1 drops its last visit, so its set (`[0, 1, 2]`) differs from the
/// every-level superset as well as from the empty Gaussian-only set. Dies when a counted
/// block's records are left out (study 1 reads nothing), and when it is given every level.
#[test]
fn a_counted_block_read_by_the_binary_predictor_takes_its_levels_from_the_records() {
    let csv = std::fs::read_to_string(root().join(B1_DATA)).unwrap();
    let mut lines = csv.lines();
    let header = lines.next().unwrap();
    assert!(header.starts_with("ID,STUDY,TIME,"), "the data moved");
    let kept: String = std::iter::once(format!("{header},VISIT\n"))
        .chain(lines.filter_map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            let time: f64 = f[2].parse().unwrap();
            (!(f[1] == "1" && time == 12.0)).then(|| format!("{l},{}\n", time / 4.0 + 1.0))
        }))
        .collect();
    let data = temp("b1_visit.csv");
    std::fs::write(&data, kept).unwrap();
    let from = "PLACEBO[STUDY, TIME, contrast = sum_to_zero_within](0.0, -10.0, 10.0)\n";
    let text = std::fs::read_to_string(root().join(B1)).unwrap();
    assert!(text.contains(from), "the fixture moved");
    let text = text
        .replace(from, "PLACEBO[4](0.0) profile\n")
        .replace(
            "  TRT   continuous\n",
            "  TRT   continuous\n  VISIT categorical\n",
        )
        .replace("TVL + PLACEBO + ETA_S", "PLACEBO[VISIT] + ETA_S")
        .replace("  theta TVL(-0.3, -5.0, 5.0)\n", "");
    let model = temp("b1_visit.ferx");
    std::fs::write(&model, text).unwrap();
    let p = crate::api::prepare_run(model.to_str().unwrap(), Some(data.to_str().unwrap()))
        .unwrap_or_else(|e| panic!("{e}"));
    let active = crate::api::active_levels(&p.parsed.model, &p.population);
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].per_subject[0], vec![0, 1, 2], "study 1");
    for (i, levels) in active[0].per_subject.iter().enumerate().skip(1) {
        assert_eq!(levels, &vec![0, 1, 2, 3], "study {}", i + 1);
    }
}

/// N-bin. Under every method the solve takes the exact matrix in at most
/// [`NEWTON_MAX_ITER`] iterations (measured 3), every matrix exact and every gradient on the
/// adjoint; with the exact matrix declined it takes the differenced matrix, counted as
/// such, in as few iterations and to the same levels. The held GLS matrix (the block
/// marked unlinked) is measured beside them: it converges on B1 (see the module comment)
/// but takes 6 iterations, so the arm shows the matrix is read.
///
/// Dies when the exact matrix is never taken (the counts), when a decline takes the held
/// matrix (the iterations) or is not counted, and when either matrix is the wrong function
/// (the levels).
#[test]
fn a_binary_read_block_takes_the_exact_newton_matrix_under_every_method() {
    for (name, method) in ARMS {
        let p = b1(None, method);
        let o = options(&p);
        let mut layout = layout_of(&p);
        let (exact, ce) = solve_b2(&p, &o);
        let (fd, cf) = {
            let _decline = MatrixDecline::on();
            solve_b2(&p, &o)
        };
        for g in &mut layout.groups {
            g.linked = false;
        }
        let held = {
            let c = ctx(&p, &layout, &o);
            c.solve(
                &p.init_params,
                &ProfileWarm::from_template(&layout, &p.init_params),
            )
            .unwrap_or_else(|e| panic!("{name}: held {e:?}"))
        };
        eprintln!(
            "{name}: exact {} it {ce:?}; declined {} it, ofv gap {:.3e}, levels {:.3e} SE; \
             held {} it",
            exact.iterations,
            fd.iterations,
            fd.ofv - exact.ofv,
            beta_gap_se(&layout, &exact, &fd),
            held.iterations
        );
        assert!(
            exact.iterations <= NEWTON_MAX_ITER,
            "{name}: {}",
            exact.iterations
        );
        assert!(
            ce.exact_matrices > 0 && ce.fd_matrices == 0 && ce.fd_gradients == 0,
            "{name}: {ce:?}"
        );
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
        assert!(
            held.iterations > exact.iterations + 1,
            "{name}: the fixture cannot tell the held matrix apart ({} it)",
            held.iterations
        );
    }
}

/// D0-bin and M9. At NONMEM's optimum (its structural θ and ω, the levels solved) the
/// profiled objective is the cold unprofiled objective at the solved levels, and no higher
/// than the unprofiled one at NONMEM's own levels, under every method. The three one-node
/// methods are one objective on binary rows (the Laplace form, #2437), and three-node
/// quadrature is a different one, so the node count is read.
///
/// Measured: solve − cold 1.3e-10; profiled − unprofiled at NONMEM's levels −2.9e-9 (one
/// node), −8.7e-9 (three nodes); NONMEM's levels within 3.3e-5 SE; the one-node methods
/// equal to every printed digit; three nodes 1.22e-2 below one.
///
/// Dies when `fit_term` scores a binary row by any other form (the methods part), when the
/// quadrature arm is ignored (the node-count part), and when the solve stops off the
/// minimum (D0).
#[test]
fn the_profiled_binary_objective_is_the_unprofiled_one_under_every_method() {
    let (theta, omega, _) = nonmem_optimum();
    let mut ofv = Vec::new();
    for (name, method) in ARMS {
        let p = b1(None, method);
        let layout = layout_of(&p);
        let o = options(&p);
        let c = ctx(&p, &layout, &o);
        let mut nm = p.init_params.clone();
        nm.theta = theta.clone();
        nm.omega = crate::types::OmegaMatrix::from_diagonal(&[omega], vec!["ETA_S".into()]);
        let at_nm: Vec<f64> = layout.coords.iter().map(|&k| nm.theta[k]).collect();
        let unprofiled = total_ofv(&c, &nm, &at_nm);
        let s = c
            .solve(&nm, &ProfileWarm::from_template(&layout, &nm))
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let beta: Vec<f64> = layout.coords.iter().map(|&k| s.params.theta[k]).collect();
        let cold = total_ofv(&c, &nm, &beta);
        eprintln!(
            "{name}: profiled {:.10}, − cold {:.3e}, − unprofiled at NONMEM's levels {:.3e}",
            s.ofv,
            s.ofv - cold,
            s.ofv - unprofiled
        );
        assert!((s.ofv - cold).abs() < 1e-8, "{name}: D0");
        assert!(
            s.ofv - unprofiled < 1e-8,
            "{name}: above the unprofiled point"
        );
        assert!(
            s.ofv - unprofiled > -1e-6,
            "{name}: far below NONMEM's optimum"
        );
        ofv.push(s.ofv);
    }
    for (k, (name, _)) in ARMS.iter().enumerate().take(3).skip(1) {
        assert!((ofv[k] - ofv[0]).abs() < 1e-9, "{name} vs laplace");
    }
    assert!((ofv[3] - ofv[4]).abs() < 1e-9, "the two quadrature anchors");
    assert!(
        ofv[0] - ofv[3] > 1e-3,
        "three nodes read: {}",
        ofv[0] - ofv[3]
    );
}

/// D3-bin. On [`b1_unbalanced`] (two studies) the solve is the minimum over the levels, a
/// dense FD Newton on the cold re-solved objective, and at one node the frozen-EBE shortcut
/// lands above it, in one test. Measured: solve − oracle 8.2e-10 (one node) / 4.5e-12
/// (three); frozen − oracle 3.26e-3 under every one-node method. The bound is a third of
/// that.
///
/// Under quadrature the frozen shortcut lands within 3.8e-9 of the minimum, so that arm
/// asserts the minimum only: the quadrature marginal barely depends on its anchor, so
/// holding η̂ is not a distinct wrong answer there (#2107's AGQ tests take the same shape).
///
/// Dies when the total gradient is replaced by the held-η̂ partial (the one-node solve
/// lands on the frozen point), and when the solve stops off the minimum.
#[test]
fn the_binary_solve_is_the_minimum_and_the_frozen_shortcut_is_not() {
    for (name, method) in ARMS {
        let p = b1_unbalanced(method, 2);
        let layout = layout_of(&p);
        let o = options(&p);
        let c = ctx(&p, &layout, &o);
        let warm = ProfileWarm::from_template(&layout, &p.init_params);
        let s = c.solve(&p.init_params, &warm).unwrap();
        let (omin, _) = oracle_min(&c, &p.init_params, &warm.beta);
        eprintln!("{name}: solve − oracle {:.3e}", s.ofv - omin);
        assert!((s.ofv - omin).abs() < 1e-8, "{name}");
        if o.agq_nodes().is_none() {
            let frozen = frozen_fixed_point(&c, &p.init_params, &warm.beta);
            eprintln!("{name}: frozen − oracle {:.3e}", frozen - omin);
            assert!(frozen - omin > 1.1e-3, "{name}: the straddle collapsed");
        }
    }
}

/// B2 (the observed proportions, Gaussian rows on CMT 1, `y = inv_logit(LP)`) and B1's
/// Bernoulli rows (on CMT 2, `[binary_model] logit = LP`) in one model, `max_study`
/// studies: each subject carries both kinds of row at every visit, so one subject's term
/// sums a Gaussian and a binary likelihood over the same levels. Under `method`.
fn mixed(method: &'static str, max_study: usize) -> crate::api::PreparedRun {
    let text = std::fs::read_to_string(root().join("nonmem_anchor/profile_logit_b2.ferx")).unwrap();
    for from in ["[fit_options]\n", "  method     = focei\n"] {
        assert!(text.contains(from), "the fixture moved: {from:?}");
    }
    let text = text
        .replace(
            "[fit_options]\n",
            "[binary_model]\n  cmt   = 2\n  logit = LP\n\n[fit_options]\n",
        )
        .replace("  method     = focei\n", method);
    let b2 = std::fs::read_to_string(root().join("nonmem_anchor/profile_logit_b2.csv")).unwrap();
    let b1 = std::fs::read_to_string(root().join(B1_DATA)).unwrap();
    let mut b2 = b2.lines();
    let header = b2.next().unwrap();
    assert_eq!(
        header, "ID,STUDY,TIME,DV,CMT,EVID,MDV,TRT,NARM",
        "the data moved"
    );
    let mut rows: Vec<(usize, f64, usize, String)> = Vec::new();
    for (lines, cmt) in [
        (b2.collect::<Vec<_>>(), 1usize),
        (b1.lines().skip(1).collect(), 2),
    ] {
        for l in lines {
            let f: Vec<&str> = l.split(',').collect();
            let study: usize = f[1].parse().unwrap();
            if study > max_study {
                continue;
            }
            let time: f64 = f[2].parse().unwrap();
            let narm = f.get(8).copied().unwrap_or("30");
            let row = format!(
                "{},{},{},{},{cmt},{},{},{},{narm}\n",
                f[0], f[1], f[2], f[3], f[5], f[6], f[7]
            );
            rows.push((study, time, cmt, row));
        }
    }
    rows.sort_by(|a, b| (a.0, a.1, a.2).partial_cmp(&(b.0, b.1, b.2)).unwrap());
    let data = temp("mixed.csv");
    std::fs::write(
        &data,
        std::iter::once(format!("{header}\n"))
            .chain(rows.into_iter().map(|r| r.3))
            .collect::<String>(),
    )
    .unwrap();
    let model = temp("mixed.ferx");
    std::fs::write(&model, text).unwrap();
    crate::api::prepare_run(model.to_str().unwrap(), Some(data.to_str().unwrap()))
        .unwrap_or_else(|e| panic!("{e}"))
}

/// A model mixing Gaussian and binary rows, every subject reading the levels through
/// both: the solve is the minimum of the cold re-solved objective, on the exact matrix,
/// and the binary rows move it, so the minimum is not the Gaussian rows' alone (B2 on the
/// same studies). Under FOCEI and Laplace, which differ on the Gaussian rows. Dies when a
/// subject's term or matrix drops either kind of row (the oracle, the straddle), and when
/// the mixed group leaves the exact route (the counts).
#[test]
fn a_block_read_by_gaussian_and_binary_rows_is_solved_to_the_minimum() {
    for (name, method) in [ARMS[1], ARMS[0]] {
        let p = mixed(method, 2);
        assert!(p.parsed.model.has_binary(), "{name}");
        let layout = layout_of(&p);
        assert!(layout.groups.iter().all(|g| g.linked), "{name}");
        let o = options(&p);
        let c = ctx(&p, &layout, &o);
        let warm = ProfileWarm::from_template(&layout, &p.init_params);
        let mut counts = ProfileCounts::default();
        let s = c.solve_counted(&p.init_params, &warm, &mut counts).unwrap();
        let (omin, _) = oracle_min(&c, &p.init_params, &warm.beta);
        let gaussian = {
            let g = prep(
                "nonmem_anchor/profile_logit_b2.ferx",
                "nonmem_anchor/profile_logit_b2.csv",
                Some(2),
                move |t| t.replace("  method     = focei\n", method),
            );
            solve_b2(&g, &options(&g)).0
        };
        let shift = beta_gap_se(&layout, &s, &gaussian);
        eprintln!(
            "{name}: {} it {counts:?}, solve − oracle {:.3e}, levels vs Gaussian rows alone \
             {shift:.3e} SE",
            s.iterations,
            s.ofv - omin
        );
        assert!((s.ofv - omin).abs() < 1e-8, "{name}");
        assert!(
            counts.exact_matrices > 0 && counts.fd_matrices == 0 && counts.fd_gradients == 0,
            "{name}: {counts:?}"
        );
        // Measured 6.0e-2 / 5.5e-2 SE (FOCEI / Laplace), solve − oracle 1e-11: B2 aggregates B1's draws, so the two kinds of
        // row agree closely, but the shift is four decades above the 1e-6 SE the solve
        // reaches levels to, which is what a dropped binary row would leave.
        assert!(
            shift > 1e-2,
            "{name}: the binary rows do not move the levels"
        );
    }
}

/// P-bin and G2-bin: the dual-vs-FD parity of the binary Newton matrix and level gradient.
/// Per subject, at the inits and at the solved point, under `laplace` and three-node
/// quadrature: the exact matrix against the second differences of the re-solved term (the
/// reference shares nothing with the adjoint), and the adjoint level gradient against
/// reconverged central differences at `h = 1e-3` SE. The held GLS matrix is measured
/// beside them, so the fixture is shown to tell a matrix without `dη̂/dβ` apart.
///
/// Measured: matrix 3.0e-6 (scaled, at [`REF_STEP_SE`]), held 4.8e-2; gradient 1.0e-7 SE
/// (one node) / 9.3e-9 SE (three), every subject on the adjoint. The bounds keep 30× /
/// 10× headroom.
///
/// Dies when the binary row's term is recorded without the link's derivative or with the
/// wrong objective, or without `dη̂/dβ`.
#[test]
fn the_binary_newton_matrix_and_level_gradient_match_reconverged_differences() {
    for (name, method) in [ARMS[0], ARMS[3]] {
        let p = b1(None, method);
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
        let (mut matrix, mut held_gap, mut gradient) = (0.0_f64, 0.0_f64, 0.0_f64);
        for params in [p.init_params.clone(), sol.params.clone()] {
            let x = pack_params(&params);
            let pt = crate::sens::ode_provider::adjoint::PackedPoint::new(&params, &x).unwrap();
            for i in 0..p.population.subjects.len() {
                let s = c.resolve(i, &params, &zero);
                let se: Vec<f64> = layout.reads[i]
                    .iter()
                    .map(|r| sol.level_scale[r.level])
                    .collect();
                let exact = c
                    .exact_hessian(i, &pt, &s)
                    .unwrap_or_else(|| panic!("{name}: the adjoint serves subject {i}"));
                let st: Vec<f64> = se.iter().map(|v| REF_STEP_SE * v).collect();
                let fd = c.resolved_hessian(i, &params, &s, &st);
                let held = c.held_hessian(i, &params, &s, &se);
                for a in 0..se.len() {
                    for b in 0..se.len() {
                        let scale = se[a] * se[b];
                        let e = ((exact[(a, b)] - fd[(a, b)]) * scale).abs();
                        assert!(e.is_finite(), "{name} subject {i}: {e}");
                        matrix = matrix.max(e);
                        held_gap = held_gap.max(((held[(a, b)] - fd[(a, b)]) * scale).abs());
                    }
                }
                let ga = c
                    .total_gradient_adjoint(i, &pt, &s)
                    .unwrap_or_else(|| panic!("{name}: the adjoint serves subject {i}"));
                let st: Vec<f64> = se.iter().map(|v| 1e-3 * v).collect();
                let gf = c.total_gradient_fd(i, &params, &s, &st);
                for ((a, b), v) in ga.iter().zip(&gf).zip(&se) {
                    let e = ((a - b) * v).abs();
                    assert!(e.is_finite(), "{name} subject {i}: {e}");
                    gradient = gradient.max(e);
                }
            }
        }
        eprintln!(
            "{name}: matrix exact − FD {matrix:.3e}, held − FD {held_gap:.3e} (scaled); \
             gradient adjoint − FD {gradient:.3e} SE"
        );
        assert!(matrix < 1e-4, "{name}: {matrix:.3e}");
        assert!(
            held_gap > 10.0 * matrix,
            "{name}: the fixture cannot tell them apart"
        );
        assert!(gradient < 1e-6, "{name}: {gradient:.3e}");
    }
}
