use super::*;
use crate::estimation::parameterization::LOG_PACK_FLOOR;
use crate::types::{classify_warning, OmegaMatrix, SigmaVector, WarningCode};

/// A stand-in for `OuterResult`: the fields the decision reads and writes, plus the names
/// the `floored_in` closure reports and a penalty the `ranked` closure adds.
#[derive(Debug, Clone)]
struct Stub {
    params: ModelParameters,
    ofv: f64,
    converged: bool,
    evals: usize,
    warnings: Vec<String>,
    floored: Vec<String>,
    penalty: f64,
    cov_secs: f64,
}

impl StoppedFit for Stub {
    fn params(&self) -> &ModelParameters {
        &self.params
    }
    fn ofv(&self) -> f64 {
        self.ofv
    }
    fn converged_mut(&mut self) -> &mut bool {
        &mut self.converged
    }
    fn evals_mut(&mut self) -> &mut usize {
        &mut self.evals
    }
    fn warnings_mut(&mut self) -> &mut Vec<String> {
        &mut self.warnings
    }
    fn covariance_secs_mut(&mut self) -> &mut f64 {
        &mut self.cov_secs
    }
}

/// A2's collapsed stop (#2339): converged at 273.848987 with TVE0 on the floor.
fn floor_stop() -> Stub {
    Stub {
        params: template(),
        ofv: 273.848987,
        converged: true,
        evals: 30,
        warnings: Vec::new(),
        floored: names(),
        penalty: 0.0,
        cov_secs: 0.0,
    }
}

/// The restart A2 measured from that stop: NONMEM's optimum, off the floor, converged.
fn landed() -> Stub {
    Stub {
        params: template(),
        ofv: 101.911514,
        converged: true,
        evals: 81,
        warnings: Vec::new(),
        floored: Vec::new(),
        penalty: 0.0,
        cov_secs: 0.0,
    }
}

fn names() -> Vec<String> {
    vec!["TVE0".to_string()]
}

/// The probe evaluation: the stop's θ just above the floor, at OFV `ofv`.
fn probe_at(ofv: f64) -> Stub {
    Stub {
        ofv,
        ..floor_stop()
    }
}

fn resolve_probe(first: Stub, probe: Stub, restart: impl FnOnce(&Stub) -> Stub) -> Stub {
    resolve_floor_escape(
        first,
        &names(),
        &probe,
        restart,
        |s| s.ofv + s.penalty,
        |s| s.floored.clone(),
        false,
    )
}

fn resolve(first: Stub, probe: f64, restart: impl FnOnce(&Stub) -> Stub) -> Stub {
    resolve_probe(first, probe_at(probe), restart)
}

/// The gate, both sides in one test: a downhill probe restarts and adopts the landed fit; an
/// uphill probe (a genuine boundary optimum) never calls the restart and returns the stop
/// untouched. A gate stuck either way reddens one half.
#[test]
fn a_floor_stop_restarts_only_when_the_probe_is_downhill() {
    // A2's probe at TVE0 = 0.055 (start 55): ≈ 0.058 below the floor.
    let downhill = resolve(floor_stop(), 273.790, |_| landed());
    assert_eq!(
        downhill.ofv, 101.911514,
        "downhill probe must adopt the restart"
    );
    assert!(downhill.converged);
    assert_eq!(
        downhill.evals,
        81 + 30 + 1,
        "fit + probe + restart evaluations"
    );
    assert_eq!(downhill.warnings.len(), 1);
    assert_eq!(
        classify_warning(&downhill.warnings[0]).category,
        WarningCode::OptimizerHealth,
        "{}",
        downhill.warnings[0]
    );

    let uphill = resolve(floor_stop(), 273.9, |_| {
        panic!("an uphill probe must not restart")
    });
    assert_eq!(uphill.ofv, 273.848987);
    assert!(
        uphill.converged,
        "a genuine boundary optimum keeps its verdict"
    );
    assert_eq!(uphill.evals, 30);
    assert!(uphill.warnings.is_empty());
}

/// The threshold is strict and at `FLOOR_PROBE_GAIN`: a drop of exactly the gain is not a
/// slope, one over it is. Exact in floating point: `FLOOR_PROBE_GAIN − 0.0` is the gain. And
/// a non-finite probe is no evidence.
#[test]
fn the_probe_must_drop_by_more_than_the_gain() {
    let at_gain = Stub {
        ofv: FLOOR_PROBE_GAIN,
        ..floor_stop()
    };
    assert_eq!(at_gain.ofv - 0.0, FLOOR_PROBE_GAIN);
    let kept = resolve(at_gain, 0.0, |_| {
        panic!("a drop of exactly the gain must not restart")
    });
    assert!(kept.converged && kept.warnings.is_empty());
    let over = Stub {
        ofv: 2.0 * FLOOR_PROBE_GAIN,
        ..floor_stop()
    };
    assert_eq!(
        resolve(over, 0.0, |_| Stub {
            ofv: -1.0,
            ..landed()
        })
        .ofv,
        -1.0
    );
    let first = floor_stop();
    let kept = resolve(first.clone(), first.ofv - FLOOR_PROBE_GAIN * 0.5, |_| {
        panic!("half the gain must not restart")
    });
    assert!(kept.converged && kept.warnings.is_empty());
    let over = first.ofv - FLOOR_PROBE_GAIN * 2.0;
    assert_eq!(resolve(first.clone(), over, |_| landed()).ofv, 101.911514);
    for bad in [f64::NAN, f64::NEG_INFINITY] {
        let kept = resolve(first.clone(), bad, |_| {
            panic!("a {bad} probe must not restart")
        });
        assert!(kept.converged);
    }
}

/// A restart that ends no lower is dropped, and the stop it could not leave is not a
/// minimum: `converged` is demoted with a `convergence` warning, the evaluations of all three
/// runs counted.
#[test]
fn a_restart_that_ends_no_lower_leaves_the_stop_not_converged() {
    for worse in [273.848987, 300.0, f64::NAN] {
        let kept = resolve(floor_stop(), 273.790, |_| Stub {
            ofv: worse,
            evals: 40,
            ..landed()
        });
        assert_eq!(
            kept.ofv, 273.848987,
            "restart at {worse} must not be adopted"
        );
        assert!(
            !kept.converged,
            "a floor stop with a downhill probe is not a minimum"
        );
        assert_eq!(kept.evals, 30 + 40 + 1);
        assert_eq!(kept.warnings.len(), 1);
        assert_eq!(
            classify_warning(&kept.warnings[0]).category,
            WarningCode::Convergence,
            "{}",
            kept.warnings[0]
        );
    }
}

/// A restart that is lower but lands on the floor again is adopted (it is the lower point)
/// and is not converged either. Its message names the θ the *restart* floored — here
/// EMAX, not the first stop's TVE0 — and the restart's OFV, and says the estimates are the
/// restart's.
#[test]
fn a_restart_that_collapses_again_is_not_converged() {
    let kept = resolve(floor_stop(), 273.790, |_| Stub {
        ofv: 250.0,
        floored: vec!["EMAX".into()],
        ..landed()
    });
    assert_eq!(kept.ofv, 250.0);
    assert!(!kept.converged);
    assert_eq!(kept.evals, 81 + 30 + 1);
    assert_eq!(kept.warnings.len(), 1);
    let w = &kept.warnings[0];
    assert_eq!(
        classify_warning(w).category,
        WarningCode::Convergence,
        "{w}"
    );
    assert_eq!(
        w,
        &floor_refloored_warning(&names(), 273.848987, 273.79, &["EMAX".into()], 250.0)
    );
}

/// The gate judges the objective the adoption ranks: with a penalty, a probe that is lower
/// on the clean −2LL but not on the penalized objective does not restart, and one lower on
/// both does.
#[test]
fn the_probe_gate_ranks_on_the_penalized_objective() {
    let penalized_up = Stub {
        penalty: 1.0,
        ..probe_at(273.790)
    };
    let kept = resolve_probe(floor_stop(), penalized_up, |_| {
        panic!("a probe higher once penalized must not restart")
    });
    assert!(kept.converged && kept.warnings.is_empty());
    let penalized_down = Stub {
        penalty: -1.0,
        ..probe_at(273.849)
    };
    assert_eq!(
        resolve_probe(floor_stop(), penalized_down, |_| landed()).ofv,
        101.911514,
        "a probe lower once penalized restarts, though its clean OFV is higher"
    );
}

/// Every sentence of the two messages carries a claim; each is pinned to the text the
/// emitter builds.
#[test]
fn the_messages_name_the_coordinate_both_objectives_and_the_outcome() {
    let w = floor_restart_warning(&names(), 273.848987, 273.79, 101.911514);
    for part in [
        "TVE0 on ferx's internal 1e-10 log-packing floor",
        "the objective still falls as it rises",
        "OFV 273.848987 on the floor, 273.790000 just above it",
        "resumed from the best point seen with it reset to the initial value;",
        "reached OFV = 101.911514",
        "the reported estimates are from the resumed run",
    ] {
        assert!(w.contains(part), "missing {part:?} in {w}");
    }
    let two = floor_restart_warning(&["A".into(), "B".into()], 1.0, 0.5, 0.1);
    assert!(two.contains("A, B on") && two.contains("with them reset to the initial values;"));

    let w = floor_not_converged_warning(&names(), 273.848987, 273.79);
    for part in [
        "TVE0 on ferx's internal 1e-10 log-packing floor",
        "the objective still falls as it rises",
        "OFV 273.848987 on the floor, 273.790000 just above it",
        "a restart from the initial value found no lower point off the floor",
        "The fit is not reported as converged",
        "try a start closer to the expected estimate, or `optimizer = bobyqa`.",
    ] {
        assert!(w.contains(part), "missing {part:?} in {w}");
    }

    let w = floor_refloored_warning(&names(), 273.848987, 273.79, &["EMAX".into()], 250.0);
    for part in [
        "TVE0 on ferx's internal 1e-10 log-packing floor",
        "the objective still falls as it rises",
        "OFV 273.848987 on the floor, 273.790000 just above it",
        "resumed from the best point seen with it reset to the initial value;",
        "the resumed run reached OFV = 250.000000 but stopped with EMAX on the floor again",
        "The reported estimates are from the resumed run",
        "the fit is not reported as converged",
        "try a start closer to the expected estimate, or `optimizer = bobyqa`.",
    ] {
        assert!(w.contains(part), "missing {part:?} in {w}");
    }
}

/// Two θ ≥ 0 (internal floor) and one θ with a declared lower bound of 0.5, which is the
/// user's floor, not ferx's; one Ω diagonal; one σ.
fn template() -> ModelParameters {
    ModelParameters {
        residual_correlations: Vec::new(),
        residual_correlation_fixed: Vec::new(),
        theta: vec![55.0, 8.0, 2.0],
        theta_names: vec!["TVE0".into(), "EMAX".into(), "ET50".into()],
        theta_lower: vec![0.0, 0.0, 0.5],
        theta_upper: vec![200.0, 100.0, 100.0],
        theta_fixed: vec![false; 3],
        omega: OmegaMatrix::from_diagonal(&[0.25], vec!["ETA_EM".into()]),
        omega_fixed: vec![false],
        sigma: SigmaVector {
            values: vec![1.0],
            names: vec!["ADD_ERR".into()],
        },
        sigma_fixed: vec![false],
        omega_iov: None,
        kappa_fixed: Vec::new(),
        mixture: None,
    }
}

/// The floor question, per coordinate: a free θ ≥ 0 on the floor is floored; one off it, one
/// FIXed on it, one on the user's declared bound, and one whose start was itself on the
/// floor (nothing to restart to) are not.
#[test]
fn floored_thetas_are_free_on_the_internal_floor_and_started_above_it() {
    let start = template();
    let mut r = start.clone();
    r.theta = vec![LOG_PACK_FLOOR, 6.67, 0.5];
    assert_eq!(floored_thetas(&r, &start), vec![0]);

    r.theta[0] = 45.3;
    assert!(floored_thetas(&r, &start).is_empty(), "off the floor");

    r.theta[0] = LOG_PACK_FLOOR;
    let mut fixed = r.clone();
    fixed.theta_fixed[0] = true;
    assert!(floored_thetas(&fixed, &start).is_empty(), "FIXed");

    let mut low_start = start.clone();
    low_start.theta[0] = LOG_PACK_FLOOR;
    assert!(
        floored_thetas(&r, &low_start).is_empty(),
        "started on the floor"
    );

    // ET50 sits on its declared 0.5: the user's bound, not ferx's floor.
    assert!(!floored_thetas(&r, &start).contains(&2));
}

/// The probe moves only the floored θ, to `FLOOR_PROBE_REL` of its start; the restart puts
/// it back at its start and the Ω diagonal it dragged onto its rail with it, leaving the
/// rest of the stop (here EMAX) where it was.
#[test]
fn the_probe_and_the_restart_move_only_the_collapsed_coordinates() {
    let start = template();
    let mut r = start.clone();
    r.theta = vec![LOG_PACK_FLOOR, 6.67, 1.47];
    // On its rail exactly, as the A2 stop is: L = exp(OMEGA_CHOL_PACKED_LOWER), Ω = L².
    let rail = (2.0 * crate::estimation::parameterization::OMEGA_CHOL_PACKED_LOWER).exp();
    r.omega = OmegaMatrix::from_diagonal(&[rail], vec!["ETA_EM".into()]);

    let p = probe_params(&r, &start, &[0]);
    assert_eq!(p.theta, vec![FLOOR_PROBE_REL * 55.0, 6.67, 1.47]);
    assert_eq!(p.omega.matrix, r.omega.matrix);

    // The stop's flat-θ pre-flight FIXed ET50; the restart starts with the user's mask, so
    // its own pre-flight freezes ET50 again and warns (#2460 review, row 1).
    r.theta_fixed[2] = true;
    let q = restart_params(&r, &start);
    assert_eq!(q.theta_fixed, start.theta_fixed);
    assert!((q.theta[0] - 55.0).abs() < 1e-9, "{}", q.theta[0]);
    assert!((q.theta[1] - 6.67).abs() < 1e-12 && (q.theta[2] - 1.47).abs() < 1e-12);
    assert!(
        (q.omega.matrix[(0, 0)] - 0.25).abs() < 1e-12,
        "{}",
        q.omega.matrix
    );

    // An Ω diagonal off its rail is left alone.
    r.omega = OmegaMatrix::from_diagonal(&[0.07], vec!["ETA_EM".into()]);
    let q = restart_params(&r, &start);
    assert!(
        (q.omega.matrix[(0, 0)] - 0.07).abs() < 1e-12,
        "{}",
        q.omega.matrix
    );
}

/// `escape_floor` end to end over a fake optimizer, which answers by the options it is
/// handed and records each call: the fit from the start; the probe as an evaluation-only
/// run (`outer_maxiter = 0`, no covariance step) with TVE0 at `FLOOR_PROBE_REL` of its
/// start and everything else at the stop; the restart with the fit's own options from the
/// stop with TVE0 and the railed Ω back at their starts. A fit that never reaches the floor
/// runs once.
#[test]
fn escape_floor_probes_evaluation_only_then_restarts_from_the_reset_stop() {
    use std::cell::RefCell;
    let start = template();
    let options = FitOptions {
        run_covariance_step: true,
        ..FitOptions::default()
    };
    let rail = (2.0 * crate::estimation::parameterization::OMEGA_CHOL_PACKED_LOWER).exp();
    let mut stop = start.clone();
    stop.theta = vec![LOG_PACK_FLOOR, 6.67, 1.47];
    stop.omega = OmegaMatrix::from_diagonal(&[rail], vec!["ETA_EM".into()]);
    // The first run's flat-θ pre-flight FIXed ET50 on its result.
    stop.theta_fixed[2] = true;
    let calls: RefCell<Vec<(ModelParameters, usize, bool)>> = RefCell::new(Vec::new());
    let run = |p: &ModelParameters, o: &FitOptions| {
        calls
            .borrow_mut()
            .push((p.clone(), o.outer_maxiter, o.run_covariance_step));
        match calls.borrow().len() {
            1 => Stub {
                params: stop.clone(),
                ..floor_stop()
            },
            2 => Stub {
                params: p.clone(),
                ofv: 273.79,
                ..floor_stop()
            },
            _ => landed(),
        }
    };
    let r = escape_floor(&start, &options, run, |_| 0.0);
    assert_eq!(r.ofv, 101.911514);
    assert!(r.converged);
    let calls = calls.into_inner();
    assert_eq!(calls.len(), 3);
    assert_eq!(calls[0].0.theta, start.theta);
    assert_eq!((calls[0].1, calls[0].2), (options.outer_maxiter, true));
    assert_eq!(calls[1].0.theta, vec![FLOOR_PROBE_REL * 55.0, 6.67, 1.47]);
    assert_eq!(
        (calls[1].1, calls[1].2),
        (0, false),
        "the probe is an evaluation"
    );
    assert!(
        (calls[2].0.theta[0] - 55.0).abs() < 1e-9,
        "{}",
        calls[2].0.theta[0]
    );
    assert!((calls[2].0.theta[1] - 6.67).abs() < 1e-12);
    assert!((calls[2].0.omega.matrix[(0, 0)] - 0.25).abs() < 1e-12);
    assert_eq!(
        calls[2].0.theta_fixed, start.theta_fixed,
        "the restart starts with the user's FIX mask"
    );
    assert_eq!((calls[2].1, calls[2].2), (options.outer_maxiter, true));

    // Off the floor: one run, returned as is.
    let n = RefCell::new(0);
    let once = escape_floor(
        &start,
        &options,
        |_, _| {
            *n.borrow_mut() += 1;
            landed()
        },
        |_| 0.0,
    );
    assert_eq!((*n.borrow(), once.ofv), (1, 101.911514));

    // Evaluation only, or a non-finite objective: no probe either, floor or not.
    for (opts, ofv) in [
        (
            FitOptions {
                outer_maxiter: 0,
                ..options.clone()
            },
            273.848987,
        ),
        (options.clone(), f64::NAN),
    ] {
        let n = RefCell::new(0);
        escape_floor(
            &start,
            &opts,
            |_, _| {
                *n.borrow_mut() += 1;
                Stub {
                    params: stop.clone(),
                    ofv,
                    ..floor_stop()
                }
            },
            |_| 0.0,
        );
        assert_eq!(*n.borrow(), 1, "maxiter {} ofv {ofv}", opts.outer_maxiter);
    }
}

/// Row 7 of #2460's review: the run that is not reported still spent its covariance step,
/// and those seconds go to the kept result's covariance time — the restart's when it is
/// adopted (on or off the floor), the stop's when it is not — never to estimation time.
#[test]
fn the_discarded_runs_covariance_time_is_counted_as_covariance() {
    let stop = || Stub {
        cov_secs: 2.0,
        ..floor_stop()
    };
    let adopted = resolve(stop(), 273.790, |_| Stub {
        cov_secs: 3.0,
        ..landed()
    });
    assert_eq!(adopted.ofv, 101.911514);
    assert_eq!(adopted.cov_secs, 5.0, "restart adopted off the floor");

    let refloored = resolve(stop(), 273.790, |_| Stub {
        ofv: 250.0,
        floored: vec!["EMAX".into()],
        cov_secs: 3.0,
        ..landed()
    });
    assert_eq!(
        refloored.cov_secs, 5.0,
        "restart adopted on the floor again"
    );

    let dropped = resolve(stop(), 273.790, |_| Stub {
        ofv: 300.0,
        cov_secs: 3.0,
        ..landed()
    });
    assert_eq!(dropped.ofv, 273.848987);
    assert_eq!(dropped.cov_secs, 5.0, "restart not adopted");

    let untouched = resolve(stop(), 273.9, |_| {
        panic!("an uphill probe must not restart")
    });
    assert_eq!(untouched.cov_secs, 2.0, "no escape, no extra time");
}
