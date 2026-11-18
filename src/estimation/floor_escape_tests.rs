use super::*;
use crate::estimation::parameterization::LOG_PACK_FLOOR;
use crate::types::{classify_warning, OmegaMatrix, SigmaVector, WarningCode};

/// A stand-in for `OuterResult`: the fields the decision reads and writes, plus the names
/// the `floored_in` closure reports, the near-floor θ the `near_floor` closure reports with
/// the OFV of their probe, and a penalty the `ranked` closure adds.
#[derive(Debug, Clone)]
struct Stub {
    params: ModelParameters,
    ofv: f64,
    converged: bool,
    evals: usize,
    warnings: Vec<String>,
    floored: Vec<String>,
    near: Vec<NearFloorTheta>,
    near_at_ofv: Option<f64>,
    near_probe_ofv: f64,
    penalty: f64,
    cov_secs: f64,
    start_eval: Option<StartEval>,
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
    fn start_eval(&self) -> &Option<StartEval> {
        &self.start_eval
    }
    fn start_eval_mut(&mut self) -> &mut Option<StartEval> {
        &mut self.start_eval
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
        near: Vec::new(),
        near_at_ofv: None,
        near_probe_ofv: f64::NAN,
        penalty: 0.0,
        cov_secs: 0.0,
        start_eval: None,
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
        near: Vec::new(),
        near_at_ofv: None,
        near_probe_ofv: f64::NAN,
        penalty: 0.0,
        cov_secs: 0.0,
        start_eval: None,
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
        near_probe,
        false,
    )
    .0
}

/// The stub's `near_floor` hook: its near-floor θ, the restart's point re-scored (the same
/// fit, at `near_at_ofv` when set) and the probe as the same fit at `near_probe_ofv` (penalty kept, so the gate ranks
/// like the real one).
fn near_probe(s: &Stub) -> Option<NearFloorProbe<Stub>> {
    (!s.near.is_empty()).then(|| NearFloorProbe {
        thetas: s.near.clone(),
        inner_maxiter: FitOptions::default().inner_maxiter,
        at: Stub {
            ofv: s.near_at_ofv.unwrap_or(s.ofv),
            ..s.clone()
        },
        probe: Stub {
            ofv: s.near_probe_ofv,
            ..s.clone()
        },
    })
}

fn resolve(first: Stub, probe: f64, restart: impl FnOnce(&Stub) -> Stub) -> Stub {
    resolve_probe(first, probe_at(probe), restart)
}

/// The gate, both sides in one test: a downhill probe restarts and adopts the landed fit; an
/// uphill probe (a genuine boundary optimum) never calls the restart and returns the stop
/// with only the probe's one evaluation added — it ran, so `n_iterations` shows it, restart
/// or not (#2839 review, row 8). A gate stuck either way reddens one half.
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
    assert_eq!(uphill.evals, 30 + 1, "fit + probe, with no restart");
    assert!(uphill.warnings.is_empty());
}

/// The threshold is strict and at `FLOOR_PROBE_GAIN`: a drop of exactly the gain is not a
/// slope, one over it is. Exact in floating point: `FLOOR_PROBE_GAIN − 0.0` is the gain. And
/// a non-finite probe is no evidence. Every declined probe still ran, so each is counted
/// once — the non-finite one too.
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
    assert_eq!(
        kept.evals,
        floor_stop().evals + 1,
        "a flat probe is counted"
    );
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
    assert_eq!(
        kept.evals,
        first.evals + 1,
        "a probe under the gain is counted"
    );
    let over = first.ofv - FLOOR_PROBE_GAIN * 2.0;
    assert_eq!(resolve(first.clone(), over, |_| landed()).ofv, 101.911514);
    for bad in [f64::NAN, f64::NEG_INFINITY] {
        let kept = resolve(first.clone(), bad, |_| {
            panic!("a {bad} probe must not restart")
        });
        assert!(kept.converged);
        assert_eq!(
            kept.evals,
            first.evals + 1,
            "a {bad} probe ran, so it is counted"
        );
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
        &floor_refloored_warning(&names(), 273.848987, 273.79, &["EMAX".into()], None, 250.0)
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

    let w = floor_refloored_warning(&names(), 273.848987, 273.79, &["EMAX".into()], None, 250.0);
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
    let r = escape_floor(&start, &options, 4, run, |_| 0.0);
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
        4,
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
            4,
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

/// #2461 through the floor escape, on the numbers of the slow-tests failure at `dad870c0`:
/// the hand-written transit ODE started at −228.712965, walked to the TVNTR floor at
/// −59.310873, was restarted, and the restart reported `converged` at −191.926765. The
/// restart's own first evaluation (here −100) is above where it ends, so a check against
/// it passes; against the **fit's** start the report is 36.8 above, so it is demoted, says
/// so once, and keeps the first run's start. The control restart lands below that start
/// and stays converged, and a fit that never reaches the floor is judged the same way.
#[test]
fn the_report_is_held_against_the_first_runs_start() {
    use crate::estimation::outer_optimizer::ABOVE_START_TOKEN;
    use std::cell::Cell;
    let start = template();
    let options = FitOptions::default();
    let mut floored = start.clone();
    floored.theta[0] = LOG_PACK_FLOOR;
    let at = |ofv: f64| StartEval {
        ofv,
        guard_rejected: false,
        n_unconverged: 0,
        n_start_rejected: 0,
        rescored: None,
        warm_rescored_from: None,
    };
    let fit_start = at(-228.712965);
    let escape = |restart_ofv: f64| {
        let calls = Cell::new(0usize);
        escape_floor(
            &start,
            &options,
            4,
            |p, _| {
                calls.set(calls.get() + 1);
                match calls.get() {
                    1 => Stub {
                        params: floored.clone(),
                        ofv: -59.310873,
                        start_eval: Some(fit_start),
                        ..floor_stop()
                    },
                    2 => Stub {
                        params: p.clone(),
                        ofv: -59.409227,
                        ..floor_stop()
                    },
                    _ => Stub {
                        ofv: restart_ofv,
                        start_eval: Some(at(-100.0)),
                        ..landed()
                    },
                }
            },
            |_| 0.0,
        )
    };
    let tagged = |r: &Stub| {
        r.warnings
            .iter()
            .filter(|w| w.starts_with(ABOVE_START_TOKEN))
            .count()
    };
    let generic = |r: &Stub| {
        r.warnings
            .iter()
            .filter(|w| *w == "Outer optimization did not converge")
            .count()
    };

    let above = escape(-191.926765);
    assert_eq!(
        above.ofv, -191.926765,
        "the restart is still the one reported"
    );
    assert!(
        above.ofv < -100.0,
        "the fixture must sit below the restart's own start, or it cannot tell the two apart"
    );
    assert!(!above.converged, "{:?}", above.warnings);
    assert_eq!(
        (tagged(&above), generic(&above)),
        (1, 1),
        "{:?}",
        above.warnings
    );
    let w = above
        .warnings
        .iter()
        .find(|w| w.starts_with(ABOVE_START_TOKEN))
        .unwrap();
    assert!(
        w.contains("-191.926765") && w.contains("-228.712965"),
        "{w}"
    );
    assert_eq!(
        above.start_eval,
        Some(fit_start),
        "the fit's start, not the restart's"
    );

    let below = escape(-228.8);
    assert!(
        below.ofv < fit_start.ofv,
        "the control must land below the start"
    );
    assert!(below.converged, "{:?}", below.warnings);
    assert_eq!(tagged(&below), 0, "{:?}", below.warnings);

    // Off the floor: one run, judged against its own start, which is the fit's.
    let once = escape_floor(
        &start,
        &options,
        4,
        |_, _| Stub {
            ofv: -191.926765,
            start_eval: Some(fit_start),
            ..landed()
        },
        |_| 0.0,
    );
    assert!(!once.converged && tagged(&once) == 1, "{:?}", once.warnings);

    // A run already not converged carries the generic notice from its optimizer; above its
    // start it gains the tagged reason, but not the notice a second time (#2517 review,
    // row 2).
    let already = escape_floor(
        &start,
        &options,
        4,
        |_, _| Stub {
            ofv: -191.926765,
            converged: false,
            warnings: vec!["Outer optimization did not converge".to_string()],
            start_eval: Some(fit_start),
            ..landed()
        },
        |_| 0.0,
    );
    assert!(!already.converged, "{:?}", already.warnings);
    assert_eq!(
        (tagged(&already), generic(&already)),
        (1, 1),
        "{:?}",
        already.warnings
    );
}

/// #2619: a start that scored `∞` is re-scored once, at the start, evaluation-only, at the
/// default inner budget, and the report is held against that — every cell of the input space
/// in one test, so a gate stuck on either side of any branch reddens it. The figures are the
/// profile tests' F2 at `inner_maxiter = 1`: re-scored start 26.465827, report 66.022863.
#[test]
fn a_start_that_scored_infinite_is_rescored_once_and_judged_against_that() {
    use crate::estimation::outer_optimizer::ABOVE_START_TOKEN;
    use std::cell::RefCell;
    let start = template();
    let at = |ofv: f64| StartEval {
        ofv,
        guard_rejected: false,
        n_unconverged: 0,
        n_start_rejected: 0,
        rescored: None,
        warm_rescored_from: None,
    };
    let with_im = |inner_maxiter: usize| FitOptions {
        inner_maxiter,
        optimizer_trace: true,
        ..FitOptions::default()
    };
    // The fit returns `report` from a start that scored `first`, stopped away from the start
    // (so a re-score at the stop is told apart); any further call returns a run whose
    // recorded start is `rescore`. Every call's θ and options are kept.
    let mut stop = start.clone();
    stop.theta[1] *= 2.0;
    let case = |options: &FitOptions,
                first: Option<StartEval>,
                report: f64,
                rescore: Option<StartEval>,
                penalty: f64| {
        let calls = RefCell::new(Vec::<(Vec<f64>, FitOptions)>::new());
        let r = escape_floor(
            &start,
            options,
            2,
            |p, o| {
                calls.borrow_mut().push((p.theta.clone(), o.clone()));
                if calls.borrow().len() == 1 {
                    Stub {
                        params: stop.clone(),
                        ofv: report,
                        start_eval: first,
                        ..landed()
                    }
                } else {
                    Stub {
                        ofv: f64::INFINITY,
                        converged: false,
                        start_eval: rescore,
                        ..landed()
                    }
                }
            },
            |_| penalty,
        );
        (r, calls.into_inner())
    };
    let tagged = |r: &Stub| -> Vec<String> {
        r.warnings
            .iter()
            .filter(|w| w.starts_with(ABOVE_START_TOKEN))
            .cloned()
            .collect()
    };
    let inf = Some(at(f64::INFINITY));
    let healthy = Some(at(26.465827));
    let im1 = with_im(1);

    // E: re-scored finite, report above → one call more, at the start, with the re-score's
    // options; demoted, the message names both budgets and no guard.
    let (e, calls) = case(&im1, inf, 66.022863, healthy, 0.0);
    assert_eq!(calls.len(), 2, "the fit and one re-score");
    let (theta, o) = &calls[1];
    assert_ne!(
        stop.theta, start.theta,
        "the fixture must stop away from its start"
    );
    assert_eq!(
        theta, &start.theta,
        "re-scored at the start, not at the stop"
    );
    assert_eq!(
        (
            o.outer_maxiter,
            o.inner_maxiter,
            o.run_covariance_step,
            o.optimizer_trace
        ),
        (0, 200, false, false),
        "evaluation-only, at the default inner budget, writing no trace"
    );
    assert!(!e.converged, "{:?}", e.warnings);
    let w = tagged(&e);
    assert_eq!(w.len(), 1, "{:?}", e.warnings);
    assert!(
        w[0].contains("failed outright at `inner_maxiter` = 1")
            && w[0].contains("at `inner_maxiter` = 200")
            && w[0].contains("26.465827")
            && !w[0].contains("guard"),
        "{}",
        w[0]
    );
    assert_eq!(
        e.start_eval.and_then(|s| s.rescored),
        Some(StartRescore {
            run_inner_maxiter: 1,
            rescore_inner_maxiter: 200
        }),
        "the published start is the re-scored one"
    );
    // The report is ranked with its penalty, as before: 66.022863 + 5.
    let (pen, _) = case(&im1, inf, 66.022863, healthy, 5.0);
    assert!(tagged(&pen)[0].contains("71.022863"), "{:?}", pen.warnings);

    // F: re-scored finite, report within the bound → kept converged, no warning.
    let (f, calls) = case(&im1, inf, 26.465827, healthy, 0.0);
    assert_eq!(calls.len(), 2);
    assert!(f.converged && tagged(&f).is_empty(), "{:?}", f.warnings);

    // E, with the re-score itself guard-rejected: its own guard verdict and counts reach the
    // message beside the re-score sentence (#2626 review, row 1).
    let guarded = Some(StartEval {
        guard_rejected: true,
        n_unconverged: 1,
        ..at(26.465827)
    });
    let (eg, _) = case(&im1, inf, 66.022863, guarded, 0.0);
    let w = tagged(&eg);
    assert_eq!(w.len(), 1, "{:?}", eg.warnings);
    assert!(
        w[0].contains("failed outright at `inner_maxiter` = 1")
            && w[0].contains("EBE convergence guard rejected")
            && w[0].contains("1 of 2 subjects"),
        "{}",
        w[0]
    );

    // G / I: a re-score that is ∞, records no start, or is hard-rejected leaves the start
    // unscored — no comparison, the verdict kept.
    let hard = Some(StartEval {
        n_start_rejected: 1,
        ..at(26.465827)
    });
    for (label, rescore) in [("∞", inf), ("none", None), ("hard-rejected", hard)] {
        let (g, calls) = case(&im1, inf, 66.022863, rescore, 0.0);
        assert_eq!(calls.len(), 2, "{label}");
        assert!(
            g.converged && tagged(&g).is_empty(),
            "{label}: {:?}",
            g.warnings
        );
        // An unusable re-score is not published as the fit's start, except the hard-rejected
        // one, which is a scored start the check itself declines.
        let published = g.start_eval.and_then(|s| s.rescored).is_some();
        assert_eq!(
            published,
            label == "hard-rejected",
            "{label}: {:?}",
            g.start_eval
        );
    }

    // No re-score at all: a finite start (C, A), no start, a default or larger budget (the
    // same deterministic failure again), a non-finite report — which is what an
    // evaluation-only fit with a failed start reports, its one evaluation (#2626 review,
    // row 3) — and a cancelled fit.
    for (label, options, first, report) in [
        ("finite start", with_im(1), healthy, 20.0),
        ("finite start, above", with_im(1), healthy, 66.022863),
        ("no start", with_im(1), None, 66.022863),
        ("default budget", with_im(200), inf, 66.022863),
        ("larger budget", with_im(500), inf, 66.022863),
        (
            "evaluation-only fit",
            FitOptions {
                outer_maxiter: 0,
                ..with_im(1)
            },
            inf,
            f64::INFINITY,
        ),
        ("non-finite report", with_im(1), inf, f64::INFINITY),
        (
            "cancelled fit",
            FitOptions {
                cancel: Some({
                    let c = crate::cancel::CancelFlag::new();
                    c.cancel();
                    c
                }),
                ..with_im(1)
            },
            inf,
            66.022863,
        ),
    ] {
        let (_, calls) = case(&options, first, report, healthy, 0.0);
        assert_eq!(calls.len(), 1, "{label}: no re-score");
    }
    // The finite start above it is still demoted, on its own figure and without the
    // re-score sentence.
    let (a, _) = case(&im1, healthy, 66.022863, healthy, 0.0);
    assert!(!a.converged, "{:?}", a.warnings);
    assert!(
        !tagged(&a)[0].contains("failed outright"),
        "{:?}",
        a.warnings
    );
}

// ── #2629: a restart that leaves a θ just above the floor ───────────────────────────

/// T1. The near-floor question, per coordinate, on `template()` (EMAX starts at 8, so its
/// probe point is `FLOOR_PROBE_REL · 8 = 0.008`): below that point it is near the floor; at
/// it exactly, above it, FIXed, on a declared bound, started on the floor, or exactly on
/// the rail (that one is [`floored_thetas`]') it is not.
#[test]
fn near_floor_thetas_are_free_internal_and_below_their_probe_point() {
    let start = template();
    let mut r = start.clone();
    r.theta = vec![45.0, 0.5 * FLOOR_PROBE_REL * 8.0, 1.47];
    assert_eq!(near_floor_thetas(&r, &start), vec![1]);
    // F2's restart stop: EMAX = 3.31e-10, 1.197 ln-units above the floor.
    r.theta[1] = 3.31e-10;
    assert_eq!(near_floor_thetas(&r, &start), vec![1]);
    assert!(floored_thetas(&r, &start).is_empty(), "not on the rail");

    r.theta[1] = FLOOR_PROBE_REL * 8.0;
    assert!(
        near_floor_thetas(&r, &start).is_empty(),
        "at the probe point"
    );
    r.theta[1] = 0.01;
    assert!(near_floor_thetas(&r, &start).is_empty(), "above it");

    r.theta[1] = 3.31e-10;
    let mut fixed = r.clone();
    fixed.theta_fixed[1] = true;
    assert!(near_floor_thetas(&fixed, &start).is_empty(), "FIXed");

    // ET50 below its own probe point (0.002), but its floor is the user's declared 0.5.
    let mut declared = r.clone();
    declared.theta = vec![45.0, 6.67, 0.0015];
    assert!(near_floor_thetas(&declared, &start).is_empty(), "declared");

    let mut low_start = start.clone();
    low_start.theta[1] = LOG_PACK_FLOOR;
    assert!(
        near_floor_thetas(&r, &low_start).is_empty(),
        "started on the floor"
    );

    r.theta[1] = LOG_PACK_FLOOR;
    assert_eq!(floored_thetas(&r, &start), vec![1]);
    assert!(
        near_floor_thetas(&r, &start).is_empty(),
        "exactly on the rail is floored_thetas'"
    );
}

/// F2's near-floor EMAX after the restart (#2629).
fn near_emax() -> Vec<NearFloorTheta> {
    vec![NearFloorTheta {
        name: "EMAX".into(),
        value: 3.31e-10,
        probe: FLOOR_PROBE_REL * 8.0,
    }]
}

/// T2. The near-floor gate, both sides in one test: the restart lands lower with EMAX just
/// above the floor, and its probe drops by 2e-6 (over `FLOOR_PROBE_GAIN`) → not converged,
/// the refloored warning naming EMAX "below 1/1000 of its start"; the same restart with a drop of
/// 0.5e-6 → converged, the success warning. Before #2629 both were converged, so the pair
/// straddles the old behaviour. A probe that is non-finite is no evidence. The probe is an
/// evaluation either way.
#[test]
fn a_restart_left_just_above_the_floor_is_converged_only_when_its_probe_is_flat() {
    let restart = |probe_ofv: f64| {
        move |_: &Stub| Stub {
            ofv: 42.788001,
            near: near_emax(),
            near_probe_ofv: probe_ofv,
            ..landed()
        }
    };
    let down = resolve(floor_stop(), 273.790, restart(42.788001 - 2e-6));
    assert_eq!(
        down.ofv, 42.788001,
        "the lower point is still the one reported"
    );
    assert!(!down.converged, "{:?}", down.warnings);
    assert_eq!(
        down.evals,
        81 + 30 + 1 + 2,
        "fit + probe + restart + the near re-score and probe"
    );
    assert_eq!(down.warnings.len(), 1);
    let w = &down.warnings[0];
    assert_eq!(
        classify_warning(w).category,
        WarningCode::Convergence,
        "{w}"
    );
    for part in [
        "but stopped with EMAX below 1/1000 of its start (EMAX = 3.310e-10)",
        "where the objective still falls as it rises (OFV 42.787999 with it at EMAX = 8.000e-3, \
         against 42.788001 where it stopped).",
        "The reported estimates are from the resumed run",
        "the fit is not reported as converged",
    ] {
        assert!(w.contains(part), "missing {part:?} in {w}");
    }
    assert!(
        !w.contains("on the floor again") && !w.contains("TVE0 below"),
        "{w}"
    );

    for flat in [42.788001 - 0.5e-6, f64::NAN, f64::NEG_INFINITY] {
        let up = resolve(floor_stop(), 273.790, restart(flat));
        assert!(up.converged, "probe {flat}: {:?}", up.warnings);
        assert_eq!(up.evals, 81 + 30 + 1 + 2, "probe {flat}");
        assert_eq!(
            up.warnings,
            vec![floor_restart_warning(
                &names(),
                273.848987,
                273.79,
                42.788001
            )],
            "probe {flat}"
        );
    }
    // A re-score of the restart's own point that is invalid (`∞`, or the divergence
    // sentinel) is no reference for a probe below it either.
    for at in [f64::INFINITY, 1e21] {
        let up = resolve(floor_stop(), 273.790, |_| Stub {
            ofv: 42.788001,
            near: near_emax(),
            near_at_ofv: Some(at),
            near_probe_ofv: 42.0,
            ..landed()
        });
        assert!(up.converged, "re-score {at}: {:?}", up.warnings);
    }
    // The probe is judged against the re-score at its own budget, not the run's OFV: a
    // probe at 42.0 is 0.79 below the run's 42.788001 but above a re-score of 41.0.
    let up = resolve(floor_stop(), 273.790, |_| Stub {
        ofv: 42.788001,
        near: near_emax(),
        near_at_ofv: Some(41.0),
        near_probe_ofv: 42.0,
        ..landed()
    });
    assert!(up.converged, "{:?}", up.warnings);
    // Re-scored at 44 against the run's 42.788001, probe at 43.9: demoted, and the message
    // names the budget of the two re-scores, since one point now carries two OFVs (#2632
    // review, row 1). Where they print alike (above, and on F2) it does not.
    let down = resolve(floor_stop(), 273.790, |_| Stub {
        ofv: 42.788001,
        near: near_emax(),
        near_at_ofv: Some(44.0),
        near_probe_ofv: 43.9,
        ..landed()
    });
    assert!(!down.converged, "{:?}", down.warnings);
    let budget = FitOptions::default().inner_maxiter;
    assert!(
        down.warnings[0].contains(&format!(
            "reached OFV = 42.788001 but stopped with EMAX below 1/1000 of its start \
             (EMAX = 3.310e-10), where the objective still falls as it rises (OFV 43.900000 \
             with it at EMAX = 8.000e-3, against 44.000000 where it stopped, both scored at \
             `inner_maxiter = {budget}`)."
        )),
        "{}",
        down.warnings[0]
    );
    assert!(!w.contains("both scored"), "{w}");
}

/// The gate ranks the near probe on the penalized objective, as the first stop's does: a
/// probe 0.79 lower on the clean −2LL is downhill with no penalty and flat once it carries a
/// penalty of 1.
#[test]
fn the_near_floor_gate_ranks_on_the_penalized_objective() {
    let restart = |_: &Stub| Stub {
        ofv: 42.788001,
        near: near_emax(),
        near_probe_ofv: 42.0,
        ..landed()
    };
    let r = resolve(floor_stop(), 273.790, restart);
    assert!(!r.converged);
    let r = resolve_floor_escape(
        floor_stop(),
        &names(),
        &probe_at(273.790),
        restart,
        |s| s.ofv + s.penalty,
        |s| s.floored.clone(),
        |s| {
            near_probe(s).map(|p| NearFloorProbe {
                probe: Stub {
                    penalty: 1.0,
                    ..p.probe
                },
                ..p
            })
        },
        false,
    )
    .0;
    assert!(r.converged, "{:?}", r.warnings);
}

/// The message's input space: each kind of restart stop gets its own clause, in the
/// singular and the plural, and neither clause claims the other's θ.
#[test]
fn the_refloored_message_gives_each_kind_its_own_clause() {
    let near = NearFloorStop {
        thetas: near_emax(),
        inner_maxiter: 1000,
        at_ofv: 42.788001,
        probe_ofv: 42.766729,
    };
    let both = floor_refloored_warning(
        &names(),
        66.022863,
        66.014508,
        &["TVE0".into()],
        Some(&near),
        42.788001,
    );
    assert!(
        both.contains(
            "but stopped with TVE0 on the floor again, and with EMAX below 1/1000 of its start \
             (EMAX = 3.310e-10), where the objective still falls as it rises \
             (OFV 42.766729 with it at EMAX = 8.000e-3, against 42.788001 where it stopped). \
             The reported"
        ),
        "{both}"
    );
    let on = floor_refloored_warning(&names(), 66.0, 65.0, &["TVE0".into()], None, 42.0);
    assert!(
        on.contains("but stopped with TVE0 on the floor again. The reported")
            && !on.contains("below 1/1000 of"),
        "{on}"
    );
    let two = NearFloorStop {
        thetas: vec![
            near_emax().remove(0),
            NearFloorTheta {
                name: "ET50".into(),
                value: 1e-9,
                probe: 0.002,
            },
        ],
        inner_maxiter: 1000,
        at_ofv: 42.0,
        probe_ofv: 40.0,
    };
    let w = floor_refloored_warning(&names(), 66.0, 65.0, &[], Some(&two), 42.0);
    assert!(
        w.contains(
            "but stopped with EMAX, ET50 below 1/1000 of their starts (EMAX = 3.310e-10, \
             ET50 = 1.000e-9), where the objective still falls as they rise \
             (OFV 40.000000 with them at EMAX = 8.000e-3, ET50 = 2.000e-3, against \
             42.000000 where it stopped). The reported"
        ) && !w.contains("on the floor again"),
        "{w}"
    );
}

/// T3, end to end over a fake optimizer (as in
/// `escape_floor_probes_evaluation_only_then_restarts_from_the_reset_stop`): the restart
/// ends with TVE0 on the rail again **and** EMAX just above it. Only EMAX is probed — the
/// fourth call moves EMAX to its probe point and leaves TVE0 on the rail — and the warning
/// names both, each in its own clause. A restart off the floor makes no fourth call.
#[test]
fn escape_floor_probes_only_the_near_floor_theta_of_the_restart() {
    use std::cell::RefCell;
    let start = template();
    // A run budget below the default, so "both at one budget" is not the run's own
    // (#2632 review, row 3).
    let options = FitOptions {
        inner_maxiter: 1,
        ..FitOptions::default()
    };
    assert!(FitOptions::default().inner_maxiter > 1);
    let mut stop = start.clone();
    stop.theta = vec![LOG_PACK_FLOOR, 6.67, 1.47];
    let mut restarted = start.clone();
    restarted.theta = vec![LOG_PACK_FLOOR, 3.31e-10, 1.47];
    let calls: RefCell<Vec<(ModelParameters, usize, usize)>> = RefCell::new(Vec::new());
    let run = |p: &ModelParameters, o: &FitOptions| {
        calls
            .borrow_mut()
            .push((p.clone(), o.outer_maxiter, o.inner_maxiter));
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
            3 => Stub {
                params: restarted.clone(),
                ofv: 42.788001,
                ..landed()
            },
            // The re-score at the default budget differs from the run's 42.788001.
            4 => Stub {
                params: restarted.clone(),
                ofv: 42.79,
                ..landed()
            },
            _ => Stub {
                params: p.clone(),
                ofv: 42.766729,
                ..landed()
            },
        }
    };
    let r = escape_floor(&start, &options, 2, run, |_| 0.0);
    let calls = calls.into_inner();
    assert_eq!(
        calls.len(),
        5,
        "fit, probe, restart, the restart's re-score, near probe"
    );
    assert_eq!(calls[3].0.theta, restarted.theta, "the restart's point");
    assert_eq!(
        calls[4].0.theta,
        vec![LOG_PACK_FLOOR, FLOOR_PROBE_REL * 8.0, 1.47],
        "only EMAX is probed; TVE0 stays on the rail"
    );
    for c in &calls[3..] {
        assert_eq!(
            (c.1, c.2),
            (0, FitOptions::default().inner_maxiter),
            "both are evaluations, at the default budget, not the run's 1"
        );
    }
    assert_eq!(r.ofv, 42.788001);
    assert!(!r.converged);
    let w = r
        .warnings
        .iter()
        .find(|w| w.contains("log-packing floor"))
        .unwrap();
    assert!(
        w.contains(
            "but stopped with TVE0 on the floor again, and with EMAX below 1/1000 of its start \
             (EMAX = 3.310e-10)"
        ) && w.contains(&format!(
            "(OFV 42.766729 with it at EMAX = 8.000e-3, against 42.790000 where it stopped, \
             both scored at `inner_maxiter = {}`)",
            FitOptions::default().inner_maxiter
        )),
        "{w}"
    );

    // Cancelled during the restart: no re-score and no probe after it (#2632 review, row 6).
    let cancel = crate::cancel::CancelFlag::new();
    let cancelling = FitOptions {
        cancel: Some(cancel.clone()),
        ..options.clone()
    };
    let n = RefCell::new(0);
    escape_floor(
        &start,
        &cancelling,
        2,
        |p, _| {
            *n.borrow_mut() += 1;
            match *n.borrow() {
                1 => Stub {
                    params: stop.clone(),
                    ..floor_stop()
                },
                2 => Stub {
                    params: p.clone(),
                    ofv: 273.79,
                    ..floor_stop()
                },
                _ => {
                    cancel.cancel();
                    Stub {
                        params: restarted.clone(),
                        ofv: 42.788001,
                        ..landed()
                    }
                }
            }
        },
        |_| 0.0,
    );
    assert_eq!(*n.borrow(), 3, "no near-floor evaluations after a cancel");

    // A restart off the floor entirely: three calls, as before #2629.
    let n = RefCell::new(0);
    let r = escape_floor(
        &start,
        &options,
        2,
        |p, _| {
            *n.borrow_mut() += 1;
            match *n.borrow() {
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
        },
        |_| 0.0,
    );
    assert_eq!((*n.borrow(), r.converged), (3, true));
}

// ---------------------------------------------------------------------------------------
// Rail completion (#2832): a variance stopped short of its internal lower guard.
// ---------------------------------------------------------------------------------------

use crate::estimation::parameterization::{
    coordinate_names, pack_params, OMEGA_CHOL_PACKED_LOWER, SIGMA_PACK_LOWER,
};
use crate::types::{MixtureParams, WarningSeverity};

/// One of every coordinate kind the walk sees: a θ on its floor; five diagonal Ω — near the
/// rail (`ETA_A`), on it (`ETA_RAIL`), far above it (`ETA_BIG`), FIXed near it (`ETA_FIX`),
/// and one for the span's edge (`ETA_EDGE`); two σ, one near its −8 guard; a 2×2 κ block
/// with `L_11` and the off-diagonal both at −4.8; a class-2 Ω override of `ETA_A` near the
/// rail. Built in packed space, so every value is where the optimizer would read it.
fn rail_template() -> ModelParameters {
    let names = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let omega = OmegaMatrix::from_diagonal(
        &[0.1, 0.1, 0.1, 0.1, 0.1],
        names(&["ETA_A", "ETA_RAIL", "ETA_BIG", "ETA_FIX", "ETA_EDGE"]),
    );
    let kappa = OmegaMatrix::from_matrix(
        nalgebra::DMatrix::from_row_slice(2, 2, &[0.1, 0.01, 0.01, 0.1]),
        names(&["KAPPA_1", "KAPPA_2"]),
        false,
    );
    let sigma = SigmaVector {
        values: vec![0.1, 0.1],
        names: names(&["EPS_C", "EPS_BIG"]),
    };
    let base = ModelParameters {
        residual_correlations: Vec::new(),
        residual_correlation_fixed: Vec::new(),
        theta: vec![LOG_PACK_FLOOR],
        theta_names: vec!["TVE0".into()],
        theta_lower: vec![0.0],
        theta_upper: vec![200.0],
        theta_fixed: vec![false],
        omega: omega.clone(),
        omega_fixed: vec![false, false, false, true, false],
        sigma: sigma.clone(),
        sigma_fixed: vec![false, false],
        omega_iov: Some(kappa),
        kappa_fixed: vec![false, false],
        mixture: Some(MixtureParams {
            omega: vec![omega.clone(), omega],
            sigma: vec![sigma.clone(), sigma],
            omega_override_addr: vec![(1, 0)],
            omega_override_fixed: vec![false],
            sigma_override_addr: Vec::new(),
            sigma_override_fixed: Vec::new(),
        }),
    };
    let names = coordinate_names(&base);
    let at = |n: &str| {
        names
            .iter()
            .position(|x| x == n)
            .unwrap_or_else(|| panic!("{n} not in {names:?}"))
    };
    let mut x = pack_params(&base);
    let o = OMEGA_CHOL_PACKED_LOWER;
    for (n, v) in [
        ("ETA_A", o + 1.2),
        ("ETA_RAIL", o),
        ("ETA_BIG", -1.4),
        ("ETA_FIX", o + 1.2),
        ("ETA_EDGE", -1.0),
        ("EPS_C", SIGMA_PACK_LOWER + 1.2),
        ("EPS_BIG", -2.0),
        ("KAPPA_1", o + 1.2),
        ("KAPPA_2~KAPPA_1", o + 1.2),
        ("ETA_A_MIX2", o + 1.2),
    ] {
        x[at(n)] = v;
    }
    unpack_params(&x, &base)
}

fn rail_names(p: &ModelParameters) -> Vec<String> {
    let mut v: Vec<String> = near_rail_variances(p).into_iter().map(|c| c.name).collect();
    v.sort();
    v
}

/// `rail_template` with `ETA_EDGE` moved until its **packed** value (after the pack round
/// trip) is exactly `target`, or panics: the span's edge is tested on the value the walk
/// reads, not the one written.
fn with_edge_at(target: f64) -> ModelParameters {
    let base = rail_template();
    let i = coordinate_names(&base)
        .iter()
        .position(|n| n == "ETA_EDGE")
        .expect("ETA_EDGE");
    let mut x = pack_params(&base);
    let mut v = target;
    for _ in 0..64 {
        x[i] = v;
        let p = unpack_params(&x, &base);
        let got = pack_params(&p)[i];
        if got == target {
            return p;
        }
        v += target - got;
    }
    panic!("no ETA_EDGE value packs to exactly {target}");
}

/// T1. The candidates are the free variance coordinates within the span above their own
/// guard and not on it — Ω diagonal, σ (its guard is −8, not −6), a κ block's `L_11`, and
/// the `[mixture]` override, named with its class — and nothing else: not the θ on its
/// floor, the Ω on its rail, the Ω far above it, the FIXed Ω, or the κ off-diagonal.
#[test]
fn near_rail_variances_are_free_variances_within_the_span_and_off_the_guard() {
    let p = rail_template();
    assert_eq!(
        rail_names(&p),
        vec!["EPS_C", "ETA_A", "ETA_A_MIX2", "KAPPA_1"],
        "{:?}",
        near_rail_variances(&p)
    );
    let c = near_rail_variances(&p);
    let eps = c.iter().find(|c| c.name == "EPS_C").expect("EPS_C");
    assert_eq!(eps.family, RailFamily::ResidualError);
    assert!(
        (eps.value - (SIGMA_PACK_LOWER + 1.2).exp()).abs() < 1e-15,
        "σ reported on its SD scale: {}",
        eps.value
    );
    let eta = c.iter().find(|c| c.name == "ETA_A").expect("ETA_A");
    assert_eq!(eta.family, RailFamily::RandomEffect);
    assert!(
        (eta.value - (2.0 * (OMEGA_CHOL_PACKED_LOWER + 1.2)).exp()).abs() < 1e-15,
        "Ω reported as a variance: {}",
        eta.value
    );

    // The span's edge, straddled on the packed value the walk reads: in at exactly
    // guard + span, out just above it.
    let edge = OMEGA_CHOL_PACKED_LOWER + RAIL_PROBE_SPAN;
    let at_edge = with_edge_at(edge);
    assert!(
        rail_names(&at_edge).contains(&"ETA_EDGE".to_string()),
        "{:?}",
        rail_names(&at_edge)
    );
    let above = with_edge_at(edge + 1e-9);
    assert!(
        !rail_names(&above).contains(&"ETA_EDGE".to_string()),
        "{:?}",
        rail_names(&above)
    );
}

/// Row 5: a block's later row near the guard is its Cholesky factor `L_22`, not `Ω_22`.
/// `rail_template`'s κ block has `L_21` at −4.8, so with `L_22` at guard + 1.0 the variance
/// `Ω_22 = L_21² + L_22²` is 23.04 while the coordinate near the guard is `L_22 = e^−5`. The
/// first row (`L_11² = Ω_11`) stays a plain variance. Both sides of the gate in one test.
#[test]
fn a_block_rows_factor_is_reported_as_its_conditional_sd() {
    let base = rail_template();
    let names = coordinate_names(&base);
    let at = |n: &str| names.iter().position(|x| x == n).expect(n);
    let mut x = pack_params(&base);
    x[at("KAPPA_2")] = OMEGA_CHOL_PACKED_LOWER + 1.0;
    let p = unpack_params(&x, &base);
    let c = near_rail_variances(&p);
    let k2 = c
        .iter()
        .find(|c| c.name == "KAPPA_2")
        .expect("KAPPA_2 in the band");
    assert_eq!(k2.family, RailFamily::BlockFactor);
    assert!(
        (k2.value - (OMEGA_CHOL_PACKED_LOWER + 1.0).exp()).abs() < 1e-15,
        "the factor, not Ω_22: {}",
        k2.value
    );
    let k1 = c.iter().find(|c| c.name == "KAPPA_1").expect("KAPPA_1");
    assert_eq!(
        k1.family,
        RailFamily::RandomEffect,
        "the first row is its variance"
    );
}

/// The probe and restart start: the named coordinates on their guard (exactly, as the
/// optimizer reads them), everything else where it stopped, and the user's θ FIX mask.
#[test]
fn at_rail_params_puts_only_the_named_coordinates_on_their_guard() {
    let p = rail_template();
    let names = coordinate_names(&p);
    let at = |n: &str| names.iter().position(|x| x == n).expect(n);
    let mut stopped = p.clone();
    stopped.theta_fixed = vec![true];
    let start = rail_template();
    let q = at_rail_params(&stopped, &start, &[at("ETA_A_MIX2"), at("EPS_C")], 0.0);
    let (x0, x1) = (pack_params(&p), pack_params(&q));
    assert_eq!(x1[at("ETA_A_MIX2")], OMEGA_CHOL_PACKED_LOWER);
    assert_eq!(x1[at("EPS_C")], SIGMA_PACK_LOWER);
    for n in ["ETA_A", "ETA_BIG", "KAPPA_1", "KAPPA_2~KAPPA_1", "EPS_BIG"] {
        assert!((x1[at(n)] - x0[at(n)]).abs() < 1e-12, "{n} moved");
    }
    assert_eq!(q.theta_fixed, start.theta_fixed);
}

fn rail_candidates() -> Vec<RailCandidate> {
    vec![
        RailCandidate {
            index: 7,
            name: "ETA_CL_MIX2".into(),
            family: RailFamily::RandomEffect,
            value: 6.7e-5,
        },
        RailCandidate {
            index: 3,
            name: "ETA_SMALL".into(),
            family: RailFamily::RandomEffect,
            value: 1.0e-4,
        },
    ]
}

/// A stop short of the rail, as `mix_override_collapse.csv` stops on Linux.
fn short_stop() -> Stub {
    Stub {
        ofv: -37.238760,
        ..floor_stop()
    }
}

fn at_ofv(ofv: f64) -> Stub {
    Stub { ofv, ..landed() }
}

/// The probed indices and the restart's indices (`None`: never restarted) of one
/// `resolve_rail` call.
type RailCalls = (Vec<usize>, Option<Vec<usize>>);

/// The fake rail world of one resolve: the stop's candidates and per-index probe OFVs, the
/// resumed run's OFV, and the candidates and probe OFVs the resumed run itself shows
/// (`again`, empty for a resumed run on the guard). A fit is told apart by its OFV: the
/// stop's, or the resumed run's. `rescore` is the stop re-scored at the probes' budget
/// (`None`: it scores as the run did).
struct RailWorld<'a> {
    candidates: &'a [RailCandidate],
    probes: &'a [(usize, f64)],
    restart_ofv: f64,
    again: &'a [RailCandidate],
    again_probes: &'a [(usize, f64)],
    rescore: Option<f64>,
}

impl RailWorld<'_> {
    fn new<'a>(
        candidates: &'a [RailCandidate],
        probes: &'a [(usize, f64)],
        restart_ofv: f64,
    ) -> RailWorld<'a> {
        RailWorld {
            candidates,
            probes,
            restart_ofv,
            again: &[],
            again_probes: &[],
            rescore: None,
        }
    }

    /// Resolve `first` (whose OFV marks it as the stop) in this world; returns the
    /// result, the indices probed on the stop, and the restart's indices.
    fn resolve(&self, first: Stub, escape: Option<String>) -> (Stub, RailCalls) {
        use std::cell::RefCell;
        let stop_ofv = first.ofv;
        let is_stop = |s: &Stub| s.ofv.to_bits() == stop_ofv.to_bits();
        let probed = RefCell::new(Vec::new());
        let restarted = RefCell::new(None);
        let r = resolve_rail_completion(
            first,
            escape,
            |s: &Stub| {
                if is_stop(s) {
                    self.candidates.to_vec()
                } else {
                    self.again.to_vec()
                }
            },
            |s: &Stub| Stub {
                ofv: if is_stop(s) {
                    self.rescore.unwrap_or(s.ofv)
                } else {
                    s.ofv
                },
                ..s.clone()
            },
            |s: &Stub, i| {
                let table = if is_stop(s) {
                    probed.borrow_mut().push(i);
                    self.probes
                } else {
                    self.again_probes
                };
                at_ofv(table.iter().find(|(j, _)| *j == i).expect("probe").1)
            },
            |_, idx| {
                *restarted.borrow_mut() = Some(idx.to_vec());
                Stub {
                    evals: 40,
                    cov_secs: 1.5,
                    ..at_ofv(self.restart_ofv)
                }
            },
            |s| s.ofv + s.penalty,
            FitOptions::default().inner_maxiter,
            false,
        );
        (r, (probed.into_inner(), restarted.into_inner()))
    }
}

/// Resolve over `candidates` with per-index probe OFVs and a restart OFV; the resumed run
/// shows no candidate.
fn resolve_rail(
    first: Stub,
    probes: &[(usize, f64)],
    restart_ofv: f64,
    candidates: &[RailCandidate],
) -> (Stub, RailCalls) {
    RailWorld::new(candidates, probes, restart_ofv).resolve(first, None)
}

/// T2 (i)–(iii), (v): the per-candidate gate and the adoption, both sides. Only the downhill
/// candidate goes on the rail for the restart (a joint probe would have been dragged uphill
/// by `ETA_SMALL`); a probe within the gain, or uphill, leaves the stop bit-identical with
/// no warning; a restart that ends no lower keeps the stop, not converged.
#[test]
fn rail_completion_restarts_with_only_the_downhill_candidates_on_the_guard() {
    let cands = rail_candidates();

    // (v) + (ii): ETA_CL_MIX2 downhill, ETA_SMALL uphill → restart with [7] only, adopted.
    let (r, (probed, restarted)) = resolve_rail(
        Stub {
            cov_secs: 2.0,
            ..short_stop()
        },
        &[(7, -37.381570), (3, -37.20)],
        -37.396982,
        &cands,
    );
    assert_eq!(probed, vec![7, 3], "every candidate probed, alone");
    assert_eq!(restarted, Some(vec![7]));
    assert_eq!(r.ofv, -37.396982);
    assert!(r.converged);
    assert_eq!(
        r.evals,
        40 + 30 + 1 + 2,
        "restart + stop + the re-score + one per probe"
    );
    assert_eq!(
        r.cov_secs,
        1.5 + 2.0,
        "the discarded stop's covariance time is booked as covariance"
    );
    assert_eq!(r.warnings.len(), 1);
    assert!(
        r.warnings[0].contains("resumed from just above the guard"),
        "{}",
        r.warnings[0]
    );
    assert!(r.warnings[0].contains("ETA_CL_MIX2") && !r.warnings[0].contains("ETA_SMALL"));

    // (i): no downhill probe — one within the gain, one uphill — returns the stop untouched
    // but for its evaluation count, which books the re-score and both probes (row 8).
    let stop = short_stop();
    let (r, (probed, restarted)) = resolve_rail(
        stop.clone(),
        &[(7, stop.ofv - 0.5 * FLOOR_PROBE_GAIN), (3, -37.0)],
        -40.0,
        &cands,
    );
    assert_eq!(probed.len(), 2);
    assert_eq!(restarted, None);
    assert_eq!(r.ofv.to_bits(), stop.ofv.to_bits());
    assert_eq!(pack_params(&r.params), pack_params(&stop.params));
    assert_eq!((r.converged, r.evals), (true, stop.evals + 1 + 2));
    assert!(r.warnings.is_empty(), "{:?}", r.warnings);

    // A NaN probe is not downhill.
    let (_, (_, restarted)) =
        resolve_rail(short_stop(), &[(7, f64::NAN), (3, -37.0)], -40.0, &cands);
    assert_eq!(restarted, None);

    // (iii): downhill probe, restart not lower → the stop, not converged, with the
    // not-converged text; the restart's evaluations and covariance time are still booked.
    let (r, (_, restarted)) =
        resolve_rail(short_stop(), &[(7, -37.381570), (3, -37.0)], -37.0, &cands);
    assert_eq!(restarted, Some(vec![7]));
    assert_eq!(r.ofv, -37.238760);
    assert!(!r.converged);
    assert_eq!((r.evals, r.cov_secs), (30 + 40 + 1 + 2, 1.5));
    assert_eq!(r.warnings.len(), 1);
    assert!(
        r.warnings[0].contains("not reported as converged"),
        "{}",
        r.warnings[0]
    );
    // A restart exactly level with the stop is not lower, so not adopted.
    let (r, _) = resolve_rail(
        short_stop(),
        &[(7, -37.381570), (3, -37.0)],
        -37.238760,
        &cands,
    );
    assert!(!r.converged && r.ofv == -37.238760);
    // A NaN restart is not adopted either.
    let (r, _) = resolve_rail(
        short_stop(),
        &[(7, -37.381570), (3, -37.0)],
        f64::NAN,
        &cands,
    );
    assert!(!r.converged && r.ofv == -37.238760);
}

/// The gate ranks on the penalized objective, like the θ escape's: a probe whose OFV is
/// lower but whose penalty is higher by more is not downhill.
#[test]
fn the_rail_gate_ranks_on_the_penalized_objective() {
    let cands = &rail_candidates()[..1];
    let r = resolve_rail_completion(
        short_stop(),
        None,
        |_: &Stub| cands.to_vec(),
        |s: &Stub| s.clone(),
        |_: &Stub, _| Stub {
            penalty: 1.0,
            ..at_ofv(-37.381570)
        },
        |_, _| panic!("an uphill ranked probe must not restart"),
        |s| s.ofv + s.penalty,
        FitOptions::default().inner_maxiter,
        false,
    );
    assert!(r.converged && r.warnings.is_empty());
}

/// Row 4: the gate judges the probes against the stop **re-scored** at their budget, not
/// against the run's in-run OFV. Both sides of it in one test: a probe lower than the
/// in-run OFV but not than the re-score does not restart; one lower than the re-score
/// but not than the in-run OFV does. The warning then names the re-score and its budget,
/// since it prints differently from the run's OFV.
#[test]
fn the_rail_gate_judges_the_probes_against_the_rescored_stop() {
    let cands = &rail_candidates()[..1];
    // In-run −37.238760; re-scored −37.30. A probe at −37.25 is below the first only.
    let mut world = RailWorld::new(cands, &[(7, -37.25)], -37.40);
    world.rescore = Some(-37.30);
    let (r, (_, restarted)) = world.resolve(short_stop(), None);
    assert_eq!(restarted, None, "uphill of the re-score");
    assert!(r.converged && r.warnings.is_empty(), "{:?}", r.warnings);

    // In-run −37.238760; re-scored −37.20. A probe at −37.22 is below the re-score only.
    let mut world = RailWorld::new(cands, &[(7, -37.22)], -37.40);
    world.rescore = Some(-37.20);
    let (r, (_, restarted)) = world.resolve(short_stop(), None);
    assert_eq!(restarted, Some(vec![7]), "downhill of the re-score");
    let w = &r.warnings[0];
    assert!(
        w.contains(
            "stopped at OFV -37.238760 (re-scored -37.200000 at `inner_maxiter = 200`, as \
             the probes were) with"
        ),
        "{w}"
    );

    // A re-score that prints as the run's OFV adds nothing.
    let (r, _) = resolve_rail(short_stop(), &[(7, -37.381570)], -37.40, cands);
    assert!(
        r.warnings[0].contains("stopped at OFV -37.238760 with"),
        "{}",
        r.warnings[0]
    );

    // A re-score that is not finite probes nothing (and books only itself).
    let mut world = RailWorld::new(cands, &[(7, -37.381570)], -37.40);
    world.rescore = Some(f64::NAN);
    let (r, (probed, restarted)) = world.resolve(short_stop(), None);
    assert_eq!((probed.len(), restarted), (0, None));
    assert_eq!(r.evals, short_stop().evals + 1);
}

/// Row 3: the adopted resumed run is probed the same way. One left short of a guard again,
/// downhill there, is reported (it is lower) but not converged, with a text that names what
/// the **resumed** run left short. Both sides: a resumed run whose only candidate is uphill
/// stays converged. The re-check's evaluations are booked.
#[test]
fn an_adopted_rail_restart_left_short_again_is_not_converged() {
    let cands = &rail_candidates()[..1];
    let again = vec![RailCandidate {
        index: 5,
        name: "EPS_PROP".into(),
        family: RailFamily::ResidualError,
        value: 4.0e-4,
    }];
    let mut world = RailWorld::new(cands, &[(7, -37.381570)], -37.396982);
    world.again = &again;
    world.again_probes = &[(5, -37.40)];
    let (r, _) = world.resolve(short_stop(), None);
    assert_eq!(r.ofv, -37.396982, "still reported: it is lower");
    assert!(!r.converged);
    assert_eq!(r.evals, 40 + 30 + (1 + 1) + (1 + 1), "both probes booked");
    assert_eq!(r.warnings.len(), 1, "{:?}", r.warnings);
    let w = &r.warnings[0];
    for want in [
        "the random effect ETA_CL_MIX2 (",
        "reached OFV = -37.396982, but stopped with the residual error EPS_PROP (SD 4.000e-4; \
         OFV -37.400000 with it on the guard) just above the guard again.",
        "The reported estimates are from the resumed run, and the fit is not reported as \
         converged: tighten `outer_xtol`, or try another `optimizer`.",
    ] {
        assert!(w.contains(want), "missing {want:?} in {w}");
    }
    let entry = classify_warning(w);
    assert_eq!(entry.category, WarningCode::Convergence);

    // The resumed run's candidate is uphill: converged, the plain restart text.
    world.again_probes = &[(5, -37.0)];
    let (r, _) = world.resolve(short_stop(), None);
    assert!(r.converged);
    assert_eq!(r.evals, 40 + 30 + (1 + 1) + (1 + 1));
    assert!(!r.warnings[0].contains("again"), "{}", r.warnings[0]);
}

/// An adopted resumed run is a new result: it carries the θ floor escape's warning ahead of
/// its own, and an escape that left the stop not converged leaves it not converged too (the
/// resumed run starts from the escape's θ). With no escape it keeps its own verdict.
#[test]
fn an_adopted_rail_restart_carries_the_floor_escapes_warning_and_verdict() {
    let cands = &rail_candidates()[..1];
    let world = RailWorld::new(cands, &[(7, -37.381570)], -37.396982);
    let resolve = |converged: bool, escape: Option<String>| {
        world
            .resolve(
                Stub {
                    converged,
                    warnings: vec!["the stop's own".into()],
                    ..short_stop()
                },
                escape,
            )
            .0
    };
    let r = resolve(false, Some("the escape's".into()));
    assert!(!r.converged, "the escape's not-converged verdict stands");
    assert_eq!(r.warnings.len(), 2, "{:?}", r.warnings);
    assert_eq!(r.warnings[0], "the escape's");
    assert!(r.warnings[1].contains("resumed from just above the guard"));

    let r = resolve(true, Some("the escape's".into()));
    assert!(r.converged);
    assert_eq!(r.warnings[0], "the escape's");

    // No escape: an unconverged stop (an iteration cap, say) is replaced by a resumed run
    // that converged, with only the rail warning.
    let r = resolve(false, None);
    assert!(r.converged);
    assert_eq!(r.warnings.len(), 1, "{:?}", r.warnings);
}

/// The text's input space (plan §3), one assertion per sentence and per class.
#[test]
fn the_rail_messages_name_each_coordinate_by_family_and_say_what_happened() {
    let ov = RailCandidate {
        index: 7,
        name: "ETA_CL_MIX2".into(),
        family: RailFamily::RandomEffect,
        value: 6.7e-5,
    };
    let sig = RailCandidate {
        index: 9,
        name: "EPS_PROP".into(),
        family: RailFamily::ResidualError,
        value: 6.1e-4,
    };
    // A re-score that prints as the run's OFV, so no budget clause.
    let same = |ofv: f64| RailScore {
        at_ofv: ofv,
        inner_maxiter: 200,
    };
    // One random effect, adopted.
    let one = rail_restart_warning(
        &[(ov.clone(), -37.381570)],
        -37.238760,
        same(-37.238760),
        -37.396982,
    );
    for want in [
        "Outer optimization stopped at OFV -37.238760",
        "the random effect ETA_CL_MIX2 (variance 6.700e-5; OFV -37.381570 with it on the guard)",
        "just above ferx's internal lower guard, where the objective is still falling.",
        "It was resumed from just above the guard with it there and reached OFV = -37.396982",
        "the reported estimates are from the resumed run.",
    ] {
        assert!(one.contains(want), "missing {want:?} in {one}");
    }
    for not in [
        "THETA",
        "1e-10",
        "log-packing floor",
        "ETA_CL ",
        "residual error",
        "not reported as converged",
        "interior",
        "outer_xtol",
        "re-scored",
        "conditional SD",
    ] {
        assert!(!one.contains(not), "{not:?} in {one}");
    }
    // A block's later row: its factor, named as the conditional SD in its block (row 5),
    // never as a variance.
    let blk = RailCandidate {
        index: 4,
        name: "ETA_V".into(),
        family: RailFamily::BlockFactor,
        value: 1.4e-2,
    };
    let b = rail_restart_warning(&[(blk, -5.0)], -4.0, same(-4.0), -5.5);
    assert!(
        b.contains(
            "the random effect ETA_V (conditional SD in its block 1.400e-2; OFV -5.000000 \
             with it on the guard)"
        ),
        "{b}"
    );
    assert!(!b.contains("variance"), "{b}");
    // σ: its own family and scale, never "random effect".
    let s = rail_restart_warning(&[(sig.clone(), -10.0)], -9.0, same(-9.0), -10.5);
    assert!(
        s.contains(
            "the residual error EPS_PROP (SD 6.100e-4; OFV -10.000000 with it on the guard)"
        ),
        "{s}"
    );
    assert!(!s.contains("random effect"), "{s}");
    // Several: every name, plural agreement.
    let two = rail_restart_warning(
        &[(ov.clone(), -37.381570), (sig.clone(), -37.30)],
        -37.238760,
        same(-37.238760),
        -37.40,
    );
    assert!(
        two.contains("ETA_CL_MIX2") && two.contains("EPS_PROP"),
        "{two}"
    );
    assert!(two.contains("with them there"), "{two}");
    assert!(!two.contains("with it there"), "{two}");
    // Not converged: the verdict and the remedy, and never "reached OFV".
    let nc = rail_not_converged_warning(&[(ov, -37.381570)], -37.238760, same(-37.238760));
    for want in [
        "Outer optimization stopped at OFV -37.238760",
        "A run resumed from just above the guard with it there found no lower point",
        "so the fit is not reported as converged",
        "tighten `outer_xtol`",
        "or try another `optimizer`.",
    ] {
        assert!(nc.contains(want), "missing {want:?} in {nc}");
    }
    // Optimizer-neutral (row 6): rail completion runs after every optimizer.
    for not in ["reached OFV", "nlopt_lbfgs", "bobyqa", "[mixture]"] {
        assert!(!nc.contains(not), "{not:?} in {nc}");
    }
    let nc2 = rail_not_converged_warning(&[(sig.clone(), -1.0), (sig, -1.0)], 0.0, same(0.0));
    assert!(
        nc2.contains("with them there") && !nc2.contains("with it there"),
        "{nc2}"
    );
}

/// Holds the rail switch for a test's whole body, so a concurrent test turning rail
/// completion off cannot reach this one.
fn rail_lock() -> std::sync::MutexGuard<'static, ()> {
    rail_switch::LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Collapse completion (#3005) off until dropped, for a test of rail completion alone whose
/// fixture has a variance far below its start. Take [`rail_lock`] first.
struct CollapseOff;

impl CollapseOff {
    fn new() -> Self {
        rail_switch::COLLAPSE_OFF.store(true, std::sync::atomic::Ordering::SeqCst);
        CollapseOff
    }
}

impl Drop for CollapseOff {
    fn drop(&mut self) {
        rail_switch::COLLAPSE_OFF.store(false, std::sync::atomic::Ordering::SeqCst);
    }
}

/// `template()` with a second Ω diagonal, both near the rail: `ETA_EM` at guard + 1.2 and
/// `ETA_B` at guard + 0.8.
fn two_near_rail() -> ModelParameters {
    let mut p = template();
    let g = OMEGA_CHOL_PACKED_LOWER;
    p.omega = OmegaMatrix::from_diagonal(
        &[(2.0 * (g + 1.2)).exp(), (2.0 * (g + 0.8)).exp()],
        vec!["ETA_EM".into(), "ETA_B".into()],
    );
    p.omega_fixed = vec![false, false];
    p
}

fn packed_of(p: &ModelParameters, name: &str) -> f64 {
    let i = coordinate_names(p)
        .iter()
        .position(|n| n == name)
        .expect(name);
    pack_params(p)[i]
}

/// T2 (iv)–(v) end to end over a fake optimizer: each near-rail variance is probed alone,
/// evaluation-only, with only it on the guard; the restart runs with the fit's own options
/// and only the downhill one on the guard (`ETA_B`, uphill, stays where it stopped); the
/// adopted restart is reported. A fit with nothing near a rail runs once.
#[test]
fn escape_floor_probes_each_near_rail_variance_alone_then_resumes_the_downhill_ones() {
    use std::cell::RefCell;
    let _g = rail_lock();
    // `template()`'s ETA_EM starts at 0.25, so near its guard it is also a collapse candidate.
    let _off = CollapseOff::new();
    let start = template();
    // A small inner budget, so the probes' "at least the default" budget is visible (row 4).
    let options = FitOptions {
        inner_maxiter: 5,
        ..FitOptions::default()
    };
    let g = OMEGA_CHOL_PACKED_LOWER;
    let stop = two_near_rail();
    let calls: RefCell<Vec<(ModelParameters, usize)>> = RefCell::new(Vec::new());
    let budgets: RefCell<Vec<usize>> = RefCell::new(Vec::new());
    let r = escape_floor(
        &start,
        &options,
        4,
        |p, o| {
            calls.borrow_mut().push((p.clone(), o.outer_maxiter));
            budgets.borrow_mut().push(o.inner_maxiter);
            if calls.borrow().len() == 1 {
                return Stub {
                    params: stop.clone(),
                    ofv: 50.0,
                    ..landed()
                };
            }
            let (em, b) = (packed_of(p, "ETA_EM"), packed_of(p, "ETA_B"));
            let near = |x: f64| (x - (g + RAIL_RESTART_ABOVE)).abs() < 1e-12;
            let off_em = (em - (g + 1.2)).abs() < 1e-12;
            let (ofv, params) = match (em == g, b == g, o.outer_maxiter) {
                // The stop, re-scored: as it scored in the run.
                (false, false, 0) if off_em => (50.0, p.clone()),
                (true, false, 0) => (49.0, p.clone()),
                (false, true, 0) => (51.0, p.clone()),
                // The resumed run ends with ETA_EM on the guard, as BOBYQA does.
                (false, false, m) if m > 0 && near(em) => {
                    let i = coordinate_names(p)
                        .iter()
                        .position(|n| n == "ETA_EM")
                        .expect("ETA_EM");
                    (48.5, at_rail_params(p, &start, &[i], 0.0))
                }
                // The resumed run's re-check: ETA_B is still in the band, and uphill.
                (true, true, 0) => (49.0, p.clone()),
                other => panic!("unexpected call {other:?}"),
            };
            Stub {
                params,
                ofv,
                ..landed()
            }
        },
        |_| 0.0,
    );
    let calls = calls.into_inner();
    assert_eq!(
        calls.len(),
        7,
        "fit; re-score, two probes, restart; the resumed run's re-score and its one probe"
    );
    let maxiters: Vec<usize> = calls.iter().map(|c| c.1).collect();
    let m = options.outer_maxiter;
    assert_eq!(
        maxiters,
        vec![m, 0, 0, 0, m, 0, 0],
        "every probe is an evaluation"
    );
    let d = FitOptions::default().inner_maxiter;
    assert_eq!(
        budgets.into_inner(),
        vec![5, d, d, d, 5, d, d],
        "the runs keep the fit's budget; the re-scores and probes share at least the default"
    );
    assert_eq!(
        pack_params(&calls[1].0),
        pack_params(&stop),
        "the stop, re-scored"
    );
    assert_eq!(packed_of(&calls[2].0, "ETA_EM"), g);
    assert!((packed_of(&calls[2].0, "ETA_B") - (g + 0.8)).abs() < 1e-12);
    assert!((packed_of(&calls[3].0, "ETA_EM") - (g + 1.2)).abs() < 1e-12);
    assert_eq!(packed_of(&calls[3].0, "ETA_B"), g);
    // The probes sit on the guard; the resumed run starts just above it.
    assert!(
        (packed_of(&calls[4].0, "ETA_EM") - (g + RAIL_RESTART_ABOVE)).abs() < 1e-12,
        "{}",
        packed_of(&calls[4].0, "ETA_EM")
    );
    assert!(
        (packed_of(&calls[4].0, "ETA_B") - (g + 0.8)).abs() < 1e-12,
        "the uphill candidate stays where it stopped"
    );
    // The re-check probes the resumed run, not the stop: ETA_EM on the guard there.
    assert_eq!(packed_of(&calls[6].0, "ETA_EM"), g);
    assert_eq!(packed_of(&calls[6].0, "ETA_B"), g);
    assert_eq!(r.ofv, 48.5);
    assert!(r.converged);
    assert_eq!(r.warnings.len(), 1, "{:?}", r.warnings);
    assert!(r.warnings[0].contains("ETA_EM") && !r.warnings[0].contains("ETA_B"));

    // (iv): nothing near a rail (ETA_EM at 0.25) → one call.
    let n = RefCell::new(0);
    escape_floor(
        &start,
        &options,
        4,
        |_, _| {
            *n.borrow_mut() += 1;
            landed()
        },
        |_| 0.0,
    );
    assert_eq!(*n.borrow(), 1);

    // Off: the stop as the optimizer returned it.
    rail_switch::OFF.store(true, std::sync::atomic::Ordering::SeqCst);
    let n = RefCell::new(0);
    let off = escape_floor(
        &start,
        &options,
        4,
        |_, _| {
            *n.borrow_mut() += 1;
            Stub {
                params: stop.clone(),
                ofv: 50.0,
                ..landed()
            }
        },
        |_| 0.0,
    );
    rail_switch::OFF.store(false, std::sync::atomic::Ordering::SeqCst);
    assert_eq!((*n.borrow(), off.ofv), (1, 50.0));
}

/// Row 2: rail completion's own skips, each on a stop that **has** near-rail variances (so
/// the empty-candidates fast path cannot answer for it): an evaluation-only fit, a
/// cancelled one, and a non-finite objective each run once and come back as the optimizer
/// returned them. Deleting any one arm reddens its row.
#[test]
fn rail_completion_skips_evaluation_only_cancelled_and_non_finite_fits() {
    use std::cell::RefCell;
    let _g = rail_lock();
    let start = template();
    let stop = two_near_rail();
    assert_eq!(near_rail_variances(&stop).len(), 2, "fixture premise");
    let cancelled = {
        let flag = crate::cancel::CancelFlag::new();
        flag.cancel();
        FitOptions {
            cancel: Some(flag),
            ..FitOptions::default()
        }
    };
    for (what, options, ofv) in [
        (
            "evaluation-only",
            FitOptions {
                outer_maxiter: 0,
                ..FitOptions::default()
            },
            50.0,
        ),
        ("cancelled", cancelled, 50.0),
        ("non-finite", FitOptions::default(), f64::NAN),
    ] {
        let n = RefCell::new(0);
        let r = escape_floor(
            &start,
            &options,
            4,
            |_, _| {
                *n.borrow_mut() += 1;
                Stub {
                    params: stop.clone(),
                    ofv,
                    ..landed()
                }
            },
            |_| 0.0,
        );
        assert_eq!(*n.borrow(), 1, "{what}: one run, no rail probe");
        assert_eq!(pack_params(&r.params), pack_params(&stop), "{what}");
        assert!(
            r.warnings.iter().all(|w| !w.contains("guard")),
            "{what}: {:?}",
            r.warnings
        );
    }
}

/// T2 (vi): rail completion reads the run the θ escape reports. The θ restart lands with
/// `ETA_EM` near the rail (the first stop had it **on** the rail, where it is no candidate),
/// so the rail probe's start carries the θ restart's EMAX, not the first stop's.
#[test]
fn rail_completion_reads_the_floor_escapes_restart_not_the_first_stop() {
    use std::cell::RefCell;
    let _g = rail_lock();
    // `template()`'s ETA_EM starts at 0.25, so near its guard it is also a collapse candidate.
    let _off = CollapseOff::new();
    let start = template();
    let options = FitOptions::default();
    let g = OMEGA_CHOL_PACKED_LOWER;
    let rail = (2.0 * g).exp();
    let mut stop = start.clone();
    stop.theta = vec![LOG_PACK_FLOOR, 6.67, 1.47];
    stop.omega = OmegaMatrix::from_diagonal(&[rail], vec!["ETA_EM".into()]);
    let mut restarted = start.clone();
    restarted.theta = vec![50.0, 7.77, 1.5];
    restarted.omega = OmegaMatrix::from_diagonal(&[(2.0 * (g + 1.0)).exp()], vec!["ETA_EM".into()]);
    let calls: RefCell<Vec<ModelParameters>> = RefCell::new(Vec::new());
    let r = escape_floor(
        &start,
        &options,
        4,
        |p, _| {
            calls.borrow_mut().push(p.clone());
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
                3 => Stub {
                    params: restarted.clone(),
                    ..landed()
                },
                // The θ restart re-scored, as it scored in its run.
                4 => Stub {
                    params: p.clone(),
                    ..landed()
                },
                5 => Stub {
                    params: p.clone(),
                    ofv: 101.0,
                    ..landed()
                },
                // The rail restart, then its re-check (re-score and probe), all level.
                _ => Stub {
                    params: p.clone(),
                    ofv: 100.0,
                    ..landed()
                },
            }
        },
        |_| 0.0,
    );
    let calls = calls.into_inner();
    assert_eq!(
        calls.len(),
        8,
        "fit, θ probe, θ restart; rail re-score, probe, restart; the resumed run's re-check"
    );
    assert_eq!(
        pack_params(&calls[3]),
        pack_params(&restarted),
        "the rail re-score reads the θ restart"
    );
    assert_eq!(
        calls[4].theta[1], 7.77,
        "the rail probe starts from the θ restart"
    );
    assert_eq!(packed_of(&calls[4], "ETA_EM"), g);
    assert_eq!(r.ofv, 100.0);
    assert_eq!(r.warnings.len(), 2, "{:?}", r.warnings);
    assert!(r.warnings[0].contains("log-packing floor"));
    assert!(r.warnings[1].contains("resumed from just above the guard"));
}

/// A BOBYQA FOCEI fit of `src` on `data`, with rail completion on or off, and the
/// coordinates rail completion probed during it. Holds no lock: the caller holds
/// [`rail_lock`] across both arms.
fn rail_fit(src: &str, data: &str, on: bool) -> (crate::types::FitResult, Vec<String>) {
    use std::sync::atomic::Ordering;
    let model = crate::parser::model_parser::parse_model_string(src).expect("model parses");
    let pop = crate::read_nonmem_csv(std::path::Path::new(data), None, None).expect("data");
    let opts = FitOptions {
        method: crate::types::EstimationMethod::FoceI,
        optimizer: crate::types::Optimizer::Bobyqa,
        run_covariance_step: false,
        threads: Some(1),
        ..FitOptions::default()
    };
    rail_switch::PROBED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clear();
    rail_switch::OFF.store(!on, Ordering::SeqCst);
    let r = crate::fit(&model, &pop, &model.default_params, &opts);
    rail_switch::OFF.store(false, Ordering::SeqCst);
    let probed = std::mem::take(
        &mut *rail_switch::PROBED
            .lock()
            .unwrap_or_else(|e| e.into_inner()),
    );
    (r.expect("fit runs"), probed)
}

/// Rail completion left the fit exactly as the optimizer stopped it: OFV and packed
/// estimate bit for bit, and no rail warning.
fn assert_rail_untouched(on: &crate::types::FitResult, off: &crate::types::FitResult) {
    assert_eq!(
        on.ofv.to_bits(),
        off.ofv.to_bits(),
        "{} vs {}",
        on.ofv,
        off.ofv
    );
    let bits = |r: &crate::types::FitResult| -> Vec<u64> {
        r.packed_estimate
            .as_deref()
            .expect("packed estimate")
            .iter()
            .map(|x| x.to_bits())
            .collect()
    };
    assert_eq!(bits(on), bits(off));
    assert!(
        on.warnings
            .iter()
            .all(|w| !w.contains("resumed from just above the guard")),
        "{:?}",
        on.warnings
    );
}

/// T6. A variance already **on** the guard (`sim_oral_no_ka_iiv.csv`, no KA IIV in the
/// truth: BOBYQA puts ω_KA on the rail) is no candidate: it is not probed, and the fit is
/// bit-identical to one with rail completion off.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn a_variance_already_on_the_guard_is_not_probed() {
    let _g = rail_lock();
    let src = "[parameters]
  theta TVCL(0.12, 0.001, 10.0)
  theta TVV(7.7, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  omega ETA_CL ~ 0.08
  omega ETA_V  ~ 0.06
  omega ETA_KA_T6 ~ 0.05
  sigma PROP_ERR ~ 0.1 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA_T6)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
";
    let data = "tests/data/guard_hold/sim_oral_no_ka_iiv.csv";
    let (on, probed) = rail_fit(src, data, true);
    let (off, _) = rail_fit(src, data, false);
    let names = coordinate_names(
        &crate::parser::model_parser::parse_model_string(src)
            .unwrap()
            .default_params,
    );
    let ka = names
        .iter()
        .position(|n| n == "ETA_KA_T6")
        .expect("ETA_KA_T6");
    let x = on.packed_estimate.as_deref().expect("packed")[ka];
    eprintln!(
        "T6 ofv {:.9} ETA_KA_T6 packed {x:.9} probed {probed:?}",
        on.ofv
    );
    assert!(
        (x - OMEGA_CHOL_PACKED_LOWER).abs() < 1e-12,
        "fixture premise: ω_KA on the guard, got {x}"
    );
    assert!(!probed.contains(&"ETA_KA_T6".to_string()), "{probed:?}");
    assert_rail_untouched(&on, &off);

    // #3005: ω_KA fell from 0.05 to the guard, far below 1/1000 of its start, so collapse
    // completion probes it — upward, at ω² = 5e-5 — finds that uphill (the truth has no KA
    // IIV), and leaves the fit bit-identical to one with collapse completion off. A real fit
    // through the probe's non-empty, no-restart path.
    rail_switch::COLLAPSE_PROBED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clear();
    let (on_again, _) = rail_fit(src, data, true);
    let collapse_probed = std::mem::take(
        &mut *rail_switch::COLLAPSE_PROBED
            .lock()
            .unwrap_or_else(|e| e.into_inner()),
    );
    assert!(
        collapse_probed.contains(&"ETA_KA_T6".to_string()),
        "{collapse_probed:?}"
    );
    let collapse_off = {
        let _off = CollapseOff::new();
        rail_fit(src, data, true).0
    };
    assert_rail_untouched(&on_again, &collapse_off);
    assert!(
        on_again.warnings.iter().all(|w| !w.contains("re-inflated")),
        "{:?}",
        on_again.warnings
    );
}

/// T7. A small but **real** variance (`small_real_iiv.csv`, ω²_V = 4e-5 in the truth) is
/// estimated inside the probe span and off the guard: rail completion probes it, finds the
/// guard uphill, and leaves the fit bit-identical to one with rail completion off. The
/// probe's non-empty path is reached by a real fit, not only by the empty fast path T6
/// takes.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn a_small_real_variance_inside_the_span_is_probed_and_kept() {
    let _g = rail_lock();
    let src = "[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V_T7 ~ 0.0001
  sigma EPS ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V_T7)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";
    let data = "tests/data/guard_hold/small_real_iiv.csv";
    let (on, probed) = rail_fit(src, data, true);
    let (off, _) = rail_fit(src, data, false);
    let names = coordinate_names(
        &crate::parser::model_parser::parse_model_string(src)
            .unwrap()
            .default_params,
    );
    let v = names
        .iter()
        .position(|n| n == "ETA_V_T7")
        .expect("ETA_V_T7");
    let x = on.packed_estimate.as_deref().expect("packed")[v];
    eprintln!(
        "T7 ofv {:.9} ETA_V_T7 packed {x:.9} probed {probed:?}",
        on.ofv
    );
    assert!(
        x > OMEGA_CHOL_PACKED_LOWER + 0.1 && x <= OMEGA_CHOL_PACKED_LOWER + RAIL_PROBE_SPAN,
        "fixture premise: ω_V interior and inside the span, got {x}"
    );
    // Counted by this fixture's own name: fits from other tests in the binary push into the
    // same process-global record (#2839 review, row 1: measured `["ETA_CL", "ETA_CL",
    // "ETA_V"]` beside the `outer_optimizer` fits).
    let mine = probed.iter().filter(|n| n.as_str() == "ETA_V_T7").count();
    assert_eq!(mine, 1, "probed once: {probed:?}");
    assert_rail_untouched(&on, &off);
}

/// [`Stub`] plus the two start verdicts, for the one decision that reads them.
#[derive(Debug, Clone)]
struct WithVerdict {
    fit: Stub,
    left_init: Option<bool>,
    stationary_start: bool,
}

impl StoppedFit for WithVerdict {
    fn params(&self) -> &ModelParameters {
        self.fit.params()
    }
    fn ofv(&self) -> f64 {
        self.fit.ofv()
    }
    fn converged_mut(&mut self) -> &mut bool {
        self.fit.converged_mut()
    }
    fn evals_mut(&mut self) -> &mut usize {
        self.fit.evals_mut()
    }
    fn warnings_mut(&mut self) -> &mut Vec<String> {
        self.fit.warnings_mut()
    }
    fn covariance_secs_mut(&mut self) -> &mut f64 {
        self.fit.covariance_secs_mut()
    }
    fn start_eval(&self) -> &Option<StartEval> {
        self.fit.start_eval()
    }
    fn start_eval_mut(&mut self) -> &mut Option<StartEval> {
        self.fit.start_eval_mut()
    }
    fn start_verdict_mut(&mut self) -> Option<(&mut Option<bool>, &mut bool)> {
        Some((&mut self.left_init, &mut self.stationary_start))
    }
}

/// An adopted resumed run reports the fit's "left its start" verdict, not its own: it is
/// measured from the rail point it was resumed at, where a resumed run that barely moves
/// reads "never left" and may carry a start-probe verdict. A first run that left the user's
/// start makes the fit's answer `Some(true)`, with no stationary start. A first run that did
/// not leave it leaves the resumed run's own verdict standing.
#[test]
fn an_adopted_rail_restart_keeps_the_fits_left_its_start_verdict() {
    let cands = &rail_candidates()[..1];
    let stop_ofv = short_stop().ofv;
    let resolve = |first_left: Option<bool>| {
        resolve_rail_completion(
            WithVerdict {
                fit: short_stop(),
                left_init: first_left,
                stationary_start: false,
            },
            None,
            |s: &WithVerdict| {
                if s.fit.ofv == stop_ofv {
                    cands.to_vec()
                } else {
                    Vec::new()
                }
            },
            |s: &WithVerdict| s.clone(),
            |_: &WithVerdict, _| WithVerdict {
                fit: at_ofv(-37.381570),
                left_init: None,
                stationary_start: false,
            },
            |_, _| WithVerdict {
                fit: at_ofv(-37.396982),
                left_init: Some(false),
                stationary_start: true,
            },
            |s| s.fit.ofv + s.fit.penalty,
            FitOptions::default().inner_maxiter,
            false,
        )
    };
    let r = resolve(Some(true));
    assert_eq!(r.fit.ofv, -37.396982, "adopted");
    assert_eq!((r.left_init, r.stationary_start), (Some(true), false));
    let r = resolve(Some(false));
    assert_eq!((r.left_init, r.stationary_start), (Some(false), true));
    let r = resolve(None);
    assert_eq!((r.left_init, r.stationary_start), (Some(false), true));
}

/// T3. The adopted text is `optimizer_health` (a warning, so `status()` stays CONVERGED);
/// the not-converged text is `convergence` (critical).
#[test]
fn the_rail_messages_classify_as_health_and_convergence() {
    let c = RailCandidate {
        index: 7,
        name: "ETA_CL_MIX2".into(),
        family: RailFamily::RandomEffect,
        value: 6.7e-5,
    };
    let at = RailScore {
        at_ofv: -37.238760,
        inner_maxiter: 200,
    };
    let adopted = classify_warning(&rail_restart_warning(
        &[(c.clone(), -37.381570)],
        -37.238760,
        at,
        -37.396982,
    ));
    assert_eq!(
        (adopted.category, adopted.severity),
        (WarningCode::OptimizerHealth, WarningSeverity::Warning)
    );
    let kept = classify_warning(&rail_not_converged_warning(&[(c, -37.381570)], -37.2, at));
    assert_eq!(
        (kept.category, kept.severity),
        (WarningCode::Convergence, WarningSeverity::Critical)
    );
}

// ── Collapse completion (#3005) ────────────────────────────────────────────────────────

/// The #3005 C-QTc shape: a large Ω diagonal (`ETA_SLOPE ~ 600`), a small one
/// (`ETA_INT ~ 0.01`, so that one near its guard is not a collapse candidate), a FIXed one,
/// and an additive σ (SD 9), all as declared starts.
fn collapse_start() -> ModelParameters {
    ModelParameters {
        omega: OmegaMatrix::from_diagonal(
            &[0.01, 600.0, 600.0],
            vec!["ETA_INT".into(), "ETA_SLOPE".into(), "ETA_FIX".into()],
        ),
        omega_fixed: vec![false, false, true],
        sigma: SigmaVector {
            values: vec![9.0],
            names: vec!["ADD".into()],
        },
        sigma_fixed: vec![false],
        ..template()
    }
}

/// `base` with the packed coordinate `name` moved until its packed value (after the pack
/// round trip) is exactly `target`, or panics — so an edge is tested on the value the walk
/// reads, not the one written.
fn with_packed_at(base: &ModelParameters, name: &str, target: f64) -> ModelParameters {
    let i = coordinate_names(base)
        .iter()
        .position(|n| n == name)
        .unwrap_or_else(|| panic!("{name}"));
    let mut x = pack_params(base);
    let mut v = target;
    for _ in 0..64 {
        x[i] = v;
        let p = unpack_params(&x, base);
        let got = pack_params(&p)[i];
        if got == target {
            return p;
        }
        v += target - got;
    }
    panic!("no {name} value packs to exactly {target}");
}

/// #3005's stop: ω²_slope at 0.00622 (start 600), ω²_FIX moved the same way (it cannot have
/// moved in a real fit; here it pins the FIX gate), σ collapsed from SD 9 to e^-3, ETA_INT
/// where it started.
fn collapsed_stop() -> ModelParameters {
    let p = with_packed_at(&collapse_start(), "ETA_SLOPE", 0.5 * 0.00622f64.ln());
    let p = with_packed_at(&p, "ETA_FIX", 0.5 * 0.00622f64.ln());
    with_packed_at(&p, "ADD", -3.0)
}

/// `base` with the packed coordinate `name` written at `target`, read back within 1e-12:
/// for an edge straddled at a margin far wider than the round trip's error.
fn with_packed_near(base: &ModelParameters, name: &str, target: f64) -> ModelParameters {
    let i = coordinate_names(base)
        .iter()
        .position(|n| n == name)
        .unwrap_or_else(|| panic!("{name}"));
    let mut x = pack_params(base);
    x[i] = target;
    let p = unpack_params(&x, base);
    let got = pack_params(&p)[i];
    assert!((got - target).abs() < 1e-12, "{name}: {got} vs {target}");
    p
}

fn collapse_names(p: &ModelParameters) -> Vec<String> {
    let mut v: Vec<String> = collapsed_variances(p, &collapse_start())
        .into_iter()
        .map(|c| c.name)
        .collect();
    v.sort();
    v
}

/// The depth is the θ floor probe's 1/1000, on a variance: half its log on the packed
/// log-SD scale.
#[test]
fn the_collapse_depth_is_a_thousandfold_fall_in_variance() {
    assert!((collapse_depth() - 0.5 * 1000f64.ln()).abs() < 1e-15);
    assert!(((-2.0 * collapse_depth()).exp() - FLOOR_PROBE_REL).abs() < 1e-18);
}

/// The candidates are the free variance coordinates more than the depth below their start:
/// the collapsed Ω and σ, not the Ω at its start nor the FIXed one. The edge straddled on
/// the packed value the walk reads: just over `start − depth` out, just under it in.
#[test]
fn collapsed_variances_are_free_and_far_below_their_start() {
    let stop = collapsed_stop();
    assert_eq!(collapse_names(&stop), vec!["ADD", "ETA_SLOPE"]);
    let c = collapsed_variances(&stop, &collapse_start());
    let slope = c.iter().find(|c| c.name == "ETA_SLOPE").expect("ETA_SLOPE");
    assert_eq!(slope.family, RailFamily::RandomEffect);
    assert!((slope.value - 0.00622).abs() < 1e-12, "{}", slope.value);
    let add = c.iter().find(|c| c.name == "ADD").expect("ADD");
    assert_eq!(add.family, RailFamily::ResidualError);

    let s = pack_params(&collapse_start())[coordinate_names(&stop)
        .iter()
        .position(|n| n == "ETA_SLOPE")
        .expect("ETA_SLOPE")];
    // `start − depth` itself does not round-trip exactly through the Cholesky packing, so
    // this edge is straddled 1e-9 either side.
    let edge = s - collapse_depth();
    let at_edge = with_packed_near(&stop, "ETA_SLOPE", edge + 1e-9);
    assert!(!collapse_names(&at_edge).contains(&"ETA_SLOPE".to_string()));
    let under = with_packed_near(&stop, "ETA_SLOPE", edge - 1e-9);
    assert!(collapse_names(&under).contains(&"ETA_SLOPE".to_string()));

    // Where it landed is not the test: inside rail completion's span (which probes it the
    // other way) and on the guard itself it is still a candidate.
    let in_span = with_packed_at(&stop, "ETA_SLOPE", OMEGA_CHOL_PACKED_LOWER + 1.0);
    assert!(collapse_names(&in_span).contains(&"ETA_SLOPE".to_string()));
    assert!(near_rail_variances(&in_span)
        .iter()
        .any(|c| c.name == "ETA_SLOPE"));
    let on_guard = with_packed_at(&stop, "ETA_SLOPE", OMEGA_CHOL_PACKED_LOWER);
    assert!(collapse_names(&on_guard).contains(&"ETA_SLOPE".to_string()));
}

/// The probe and the restart point: the named coordinates at `below` under their packed
/// start (exactly), everything else where it stopped, and the user's θ FIX mask.
#[test]
fn below_start_params_moves_only_the_named_coordinates_relative_to_their_start() {
    let mut stop = collapsed_stop();
    stop.theta_fixed = vec![true, false, false];
    let start = collapse_start();
    let names = coordinate_names(&stop);
    let at = |n: &str| names.iter().position(|x| x == n).expect(n);
    let (x0, xs) = (pack_params(&stop), pack_params(&start));
    let probe = below_start_params(&stop, &start, &[at("ETA_SLOPE")], collapse_depth());
    let x1 = pack_params(&probe);
    assert!((x1[at("ETA_SLOPE")] - (xs[at("ETA_SLOPE")] - collapse_depth())).abs() < 1e-12);
    assert!(
        (probe.omega.matrix[(1, 1)] - 600.0 * FLOOR_PROBE_REL).abs() < 1e-9,
        "the probe is ω² at 1/1000 of its start: {}",
        probe.omega.matrix[(1, 1)]
    );
    for n in ["ETA_INT", "ETA_FIX", "ADD", "TVE0"] {
        assert!((x1[at(n)] - x0[at(n)]).abs() < 1e-12, "{n} moved");
    }
    assert_eq!(probe.theta_fixed, start.theta_fixed);
    let resumed = below_start_params(&stop, &start, &[at("ETA_SLOPE"), at("ADD")], 0.0);
    assert!((resumed.omega.matrix[(1, 1)] - 600.0).abs() < 1e-9);
    assert!((resumed.sigma.values[0] - 9.0).abs() < 1e-12);
}

fn slope_candidate() -> RailCandidate {
    RailCandidate {
        index: 4,
        name: "ETA_SLOPE".into(),
        family: RailFamily::RandomEffect,
        value: 0.00622,
    }
}

/// The #3005 stop as a stub: converged at 3327.580600.
fn qtc_stop() -> Stub {
    Stub {
        ofv: 3327.5806,
        ..floor_stop()
    }
}

/// One collapse resolve over a single candidate with probe OFV `probe` and resumed OFV
/// `restart`; the resumed run's own candidates (`again`) probe at `again_probe`. Returns the
/// result, the carried warnings, and whether it restarted.
fn resolve_collapse(
    first: Stub,
    carried: Vec<String>,
    probe: f64,
    restart: Option<f64>,
    again: Vec<RailCandidate>,
    again_probe: f64,
) -> (Stub, Vec<String>, bool) {
    let stop_ofv = first.ofv;
    let is_stop = move |s: &Stub| s.ofv.to_bits() == stop_ofv.to_bits();
    let restarted = std::cell::Cell::new(false);
    let (r, c) = resolve_short_stop(
        ShortStop::Collapse,
        first,
        carried,
        |s: &Stub| {
            if is_stop(s) {
                vec![slope_candidate()]
            } else {
                again.clone()
            }
        },
        |s: &Stub| s.clone(),
        |s: &Stub, _| at_ofv(if is_stop(s) { probe } else { again_probe }),
        |_, idx| {
            assert_eq!(idx, &[4]);
            restarted.set(true);
            restart.map(|ofv| Stub {
                evals: 40,
                ..at_ofv(ofv)
            })
        },
        |s| s.ofv + s.penalty,
        FitOptions::default().inner_maxiter,
        false,
    );
    (r, c, restarted.get())
}

/// The decision, both sides of each gate. A downhill probe resumes the fit with the variance
/// re-inflated and adopts a lower resumed run, which carries the earlier stages' warnings
/// ahead of its own and hands its own on; one within the gain does nothing but book its
/// evaluations; a resumed run no lower keeps the stop, not converged, and hands on only what
/// it was given (the stop still holds those).
#[test]
fn collapse_completion_re_inflates_a_downhill_variance_and_adopts_only_a_lower_run() {
    // Adopted: the #3005 numbers (probe ≈ 0.6 OFV-units of slope × 0.6, restart 3326.857).
    let (r, carried, restarted) = resolve_collapse(
        qtc_stop(),
        vec!["the θ escape's".into()],
        3327.5797,
        Some(3326.8566),
        Vec::new(),
        f64::NAN,
    );
    assert!(restarted);
    assert_eq!(r.ofv, 3326.8566);
    assert!(r.converged);
    assert_eq!(
        r.evals,
        40 + 30 + 1 + 1,
        "resumed + stop + re-score + one probe"
    );
    assert_eq!(r.warnings.len(), 2, "{:?}", r.warnings);
    assert_eq!(r.warnings[0], "the θ escape's");
    assert!(
        r.warnings[1].contains("re-inflated to the initial value"),
        "{}",
        r.warnings[1]
    );
    assert_eq!(carried, r.warnings, "a later stage carries both");

    // Within the gain: untouched but for the re-score and the probe.
    let stop = qtc_stop();
    let (r, carried, restarted) = resolve_collapse(
        stop.clone(),
        Vec::new(),
        stop.ofv - 0.5 * FLOOR_PROBE_GAIN,
        Some(3326.8566),
        Vec::new(),
        f64::NAN,
    );
    assert!(!restarted);
    assert_eq!((r.converged, r.evals), (true, stop.evals + 2));
    assert!(r.warnings.is_empty() && carried.is_empty());

    // Resumed no lower: the stop, not converged, with the not-converged text.
    let (r, carried, restarted) = resolve_collapse(
        qtc_stop(),
        vec!["the θ escape's".into()],
        3327.5797,
        Some(3327.5806),
        Vec::new(),
        f64::NAN,
    );
    assert!(restarted);
    assert_eq!(r.ofv, 3327.5806);
    assert!(!r.converged);
    assert_eq!(r.warnings.len(), 1, "{:?}", r.warnings);
    assert!(r.warnings[0].contains("not reported as converged"));
    assert_eq!(carried, vec!["the θ escape's".to_string()]);

    // Adopted but collapsed again, downhill there: reported, not converged.
    let (r, _, _) = resolve_collapse(
        qtc_stop(),
        Vec::new(),
        3327.5797,
        Some(3327.0),
        vec![RailCandidate {
            value: 0.004,
            ..slope_candidate()
        }],
        3326.9,
    );
    assert_eq!(r.ofv, 3327.0);
    assert!(!r.converged);
    assert!(
        r.warnings[0].contains("below 1/1000 of its initial value again"),
        "{}",
        r.warnings[0]
    );

    // Verification only (no restart allowed, #3005 review): a downhill probe keeps the stop
    // bit-identical, demotes it, books the probes, and hands on only what it was given.
    let stop = qtc_stop();
    let (r, carried, restarted) = resolve_collapse(
        stop.clone(),
        vec!["the rail stage's".into()],
        3327.5797,
        None,
        Vec::new(),
        f64::NAN,
    );
    assert!(restarted, "the restart closure was asked");
    assert_eq!(r.ofv.to_bits(), stop.ofv.to_bits());
    assert_eq!(pack_params(&r.params), pack_params(&stop.params));
    assert!(!r.converged);
    assert_eq!(r.evals, stop.evals + 2);
    assert_eq!(r.warnings.len(), 1, "{:?}", r.warnings);
    assert!(
        r.warnings[0].contains(
            "It was reached by a run that had already been resumed, so it was not resumed \
             again; the fit is not reported as converged: try another `optimizer`"
        ),
        "{}",
        r.warnings[0]
    );
    assert_eq!(
        classify_warning(&r.warnings[0]).category,
        WarningCode::Convergence
    );
    assert_eq!(carried, vec!["the rail stage's".to_string()]);
    // …and an uphill probe leaves a verified stop converged.
    let (r, _, _) = resolve_collapse(stop.clone(), Vec::new(), 3327.6, None, Vec::new(), 0.0);
    assert!(r.converged && r.warnings.is_empty());
}

/// Every sentence of the collapse texts, and their classes: the adopted one is
/// `optimizer_health` (a warning, so a converged fit stays converged), the others are
/// `convergence`. Each sentence is asserted so deleting it reddens this test; the rail
/// wording ("guard") must not leak into a collapse text, nor the reverse.
#[test]
fn the_collapse_messages_say_where_why_what_happened_and_what_to_do() {
    let at = RailScore {
        at_ofv: 3327.5806,
        inner_maxiter: 200,
    };
    let d = [(slope_candidate(), 3327.579700)];
    let adopted = collapse_restart_warning(&d, 3327.5806, at, 3326.856600);
    for want in [
        "Outer optimization stopped at OFV 3327.580600 with the random effect ETA_SLOPE \
         (variance 6.220e-3; OFV 3327.579700 with it raised to 1/1000 of its initial value) \
         below 1/1000 of its initial value, where the objective is still falling.",
        "Variances are optimized on a log scale, where the gradient of a collapsed one is \
         scaled down by its own size, so the optimizer read the point as stationary; it is \
         not a minimum.",
        "It was resumed with it re-inflated to the initial value and reached OFV = \
         3326.856600; the reported estimates are from the resumed run.",
    ] {
        assert!(adopted.contains(want), "missing {want:?} in {adopted}");
    }
    assert!(!adopted.contains("guard"), "{adopted}");
    let e = classify_warning(&adopted);
    assert_eq!(
        (e.category, e.severity),
        (WarningCode::OptimizerHealth, WarningSeverity::Warning)
    );

    let kept = collapse_not_converged_warning(&d, 3327.5806, at);
    assert!(
        kept.contains(
            "A run resumed with it re-inflated to the initial value found no lower point, so \
             the fit is not reported as converged: try another `optimizer` or `method = \
             gn_hybrid`, or start the variance closer to its expected estimate."
        ),
        "{kept}"
    );
    assert!(kept.contains("it is not a minimum."), "{kept}");
    let e = classify_warning(&kept);
    assert_eq!(
        (e.category, e.severity),
        (WarningCode::Convergence, WarningSeverity::Critical)
    );

    // Two coordinates: plural throughout.
    let add = RailCandidate {
        index: 6,
        name: "ADD".into(),
        family: RailFamily::ResidualError,
        value: 0.05,
    };
    let two = [(slope_candidate(), 3327.5797), (add.clone(), 3327.57)];
    let again = collapse_again_warning(&two, 3327.5806, at, 3327.0, &[(add, 3326.9)]);
    for want in [
        "ETA_SLOPE (variance 6.220e-3; OFV 3327.579700 with it raised to 1/1000 of its \
         initial value), the residual error ADD (SD 5.000e-2; OFV 3327.570000 with it raised to 1/1000 \
         of its initial value) below 1/1000 of their initial values, where",
        "It was resumed with them re-inflated to the initial values and reached OFV = \
         3327.000000, but stopped with the residual error ADD (SD 5.000e-2; OFV 3326.900000 \
         with it raised to 1/1000 of its initial value) below 1/1000 of its initial value \
         again.",
        "The reported estimates are from the resumed run, and the fit is not reported as \
         converged: try another `optimizer`",
    ] {
        assert!(again.contains(want), "missing {want:?} in {again}");
    }
    assert_eq!(classify_warning(&again).category, WarningCode::Convergence);

    // The rail texts keep their own wording.
    let rail = rail_restart_warning(&d, 3327.5806, at, 3326.8566);
    assert!(
        rail.contains("just above ferx's internal lower guard"),
        "{rail}"
    );
    assert!(
        rail.contains("OFV 3327.579700 with it on the guard"),
        "{rail}"
    );
    assert!(
        !rail.contains("1/1000") && !rail.contains("log scale"),
        "{rail}"
    );
}

/// End to end over a fake optimizer: the collapsed ω² is probed evaluation-only at 1/1000 of
/// its start, the fit is resumed with it back at its start and the fit's own options, and
/// the adopted run is reported. Collapse completion runs **before** rail completion, so a
/// resumed run that ends with a variance just above the guard is read by the rail stage and
/// carries the collapse warning. With the switch off the stop comes back as returned.
#[test]
fn escape_floor_re_inflates_a_collapsed_variance_then_hands_the_result_to_rail_completion() {
    use std::cell::RefCell;
    let _g = rail_lock();
    let start = collapse_start();
    let options = FitOptions::default();
    let stop = with_packed_at(&collapse_start(), "ETA_SLOPE", 0.5 * 0.00622f64.ln());
    let s = packed_of(&start, "ETA_SLOPE");
    let g = OMEGA_CHOL_PACKED_LOWER;
    let calls: RefCell<Vec<(ModelParameters, usize)>> = RefCell::new(Vec::new());
    let run = |p: &ModelParameters, o: &FitOptions| {
        calls.borrow_mut().push((p.clone(), o.outer_maxiter));
        let n = calls.borrow().len();
        let slope = packed_of(p, "ETA_SLOPE");
        let int = packed_of(p, "ETA_INT");
        let (ofv, params) = match (n, o.outer_maxiter) {
            (1, _) => (3327.5806, stop.clone()),
            // Collapse stage: the re-score, then the probe at start − depth.
            (2, 0) => (3327.5806, p.clone()),
            (3, 0) if (slope - (s - collapse_depth())).abs() < 1e-12 => (3327.5797, p.clone()),
            // The resumed run starts with ω² back at 600 and ends with ETA_INT near its
            // guard (packed g + 1.2).
            (4, m) if m > 0 && (slope - s).abs() < 1e-12 => {
                (3326.86, with_packed_at(p, "ETA_INT", g + 1.2))
            }
            // Its own collapse re-check finds nothing (ETA_SLOPE at its start), so the rail
            // stage: re-score, ETA_INT probed on the guard (uphill), done.
            (5, 0) => (3326.86, p.clone()),
            (6, 0) if int == g => (3326.95, p.clone()),
            other => panic!("unexpected call {other:?} (slope {slope}, int {int})"),
        };
        Stub {
            params,
            ofv,
            ..landed()
        }
    };
    let r = escape_floor(&start, &options, 44, run, |_| 0.0);
    let calls = calls.into_inner();
    assert_eq!(
        calls.len(),
        6,
        "fit; re-score, probe, resume; rail re-score, rail probe"
    );
    let probe = &calls[2].0;
    assert!(
        (probe.omega.matrix[(1, 1)] - 0.6).abs() < 1e-9,
        "probed at 1/1000 of its start: {}",
        probe.omega.matrix[(1, 1)]
    );
    assert!(
        (packed_of(probe, "ETA_INT") - packed_of(&stop, "ETA_INT")).abs() < 1e-12,
        "only the candidate moves"
    );
    assert_eq!(
        calls[3].1, options.outer_maxiter,
        "resumed with the fit's options"
    );
    assert!((calls[3].0.omega.matrix[(1, 1)] - 600.0).abs() < 1e-9);
    assert_eq!(r.ofv, 3326.86);
    assert!(r.converged);
    assert_eq!(r.warnings.len(), 1, "{:?}", r.warnings);
    assert!(r.warnings[0].contains("ETA_SLOPE") && r.warnings[0].contains("re-inflated"));

    // Off: one call, the stop as returned.
    rail_switch::COLLAPSE_OFF.store(true, std::sync::atomic::Ordering::SeqCst);
    let n = RefCell::new(0);
    let off = escape_floor(
        &start,
        &options,
        44,
        |_, _| {
            *n.borrow_mut() += 1;
            Stub {
                params: stop.clone(),
                ofv: 3327.5806,
                ..landed()
            }
        },
        |_| 0.0,
    );
    rail_switch::COLLAPSE_OFF.store(false, std::sync::atomic::Ordering::SeqCst);
    assert_eq!((*n.borrow(), off.ofv, off.converged), (1, 3327.5806, true));
}

/// The #3005 model: `lmer(dQTcF ~ TIME + ACTIVE + BLC + CONC + (CONC || ID))` as a ferx
/// model, with the issue's starts.
const QTC_MODEL: &str = "[parameters]
  theta INT(0.1, -100.0, 100.0)
  theta TEFF[TIME, contrast = ref](0.0, -100.0, 100.0)
  theta ACT(0.1, -100.0, 100.0)
  theta SLOPE(10, -1000.0, 1000.0)
  theta BL(-0.1, -10.0, 10.0)

  omega ETA_INT ~ 25
  omega ETA_SLOPE ~ 600

  sigma ADD ~ 80 (variance)

[covariates]
  CONC continuous
  ACTIVE continuous
  BLC continuous

[individual_parameters]
  B0 = INT + ETA_INT
  B2 = SLOPE + ETA_SLOPE

[structural_model]
  y = B0 + TEFF + ACT * ACTIVE + B2 * CONC + BL * BLC

[error_model]
  DV ~ additive(ADD)
";

/// The default FOCEI fit of [`QTC_MODEL`] to `qtc_sim.csv`, with collapse completion on or
/// off, and ω²_slope.
fn qtc_fit(on: bool) -> (crate::types::FitResult, f64) {
    use std::sync::atomic::Ordering;
    let mut parsed = crate::parser::model_parser::parse_full_model(QTC_MODEL).expect("model");
    let mut pop =
        crate::api::read_population_as_fitted(&parsed, "tests/data/variance_collapse/qtc_sim.csv")
            .expect("data")
            .0;
    crate::api::bind_theta_levels(&mut parsed, QTC_MODEL, &mut pop).expect("TEFF levels");
    let model = parsed.model;
    let opts = FitOptions {
        method: crate::types::EstimationMethod::FoceI,
        outer_maxiter: 1000,
        run_covariance_step: false,
        threads: Some(1),
        ..FitOptions::default()
    };
    rail_switch::COLLAPSE_OFF.store(!on, Ordering::SeqCst);
    let r = crate::fit(&model, &pop, &model.default_params, &opts);
    rail_switch::COLLAPSE_OFF.store(false, Ordering::SeqCst);
    let r = r.expect("fit runs");
    let w = r.omega[(1, 1)];
    (r, w)
}

/// #3005 end to end, on data simulated at ω²_slope = 652 whose ML optimum is ω²_slope =
/// 87.41 at OFV 3307.9166 (lme4 `REML = FALSE`, −2 log L − 704 ln 2π = 3307.916). With
/// collapse completion off the default optimizer throws ω²_slope from 600 to ~3e-5 — into
/// rail completion's span, whose downward probe is uphill — and stops `converged` 0.77
/// above the optimum. On, the probe at ω² = 0.6 is downhill, the resumed run from 600
/// reaches the optimum, and the warning names the variance. Measured on Linux x86_64 and
/// macOS arm64 alike: off 3308.688073, ω² 3.085e-5; on 3307.916556, ω² 87.4131.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn a_collapsed_variance_with_an_inward_slope_is_re_inflated_to_the_optimum() {
    let _g = rail_lock();
    let (off, w_off) = qtc_fit(false);
    let (on, w_on) = qtc_fit(true);
    eprintln!(
        "#3005 off {:.6} ω² {w_off:.3e} converged {}; on {:.6} ω² {w_on:.4} converged {}",
        off.ofv, off.converged, on.ofv, on.converged
    );
    assert!(off.ofv.is_finite() && on.ofv.is_finite());
    assert!(
        w_off < 0.6,
        "fixture premise: the uncompleted fit collapses ω²_slope below 1/1000 of its start, \
         got {w_off}"
    );
    assert!(
        (on.ofv - 3307.9166).abs() < 1e-3,
        "the lme4 optimum, got {:.6}",
        on.ofv
    );
    assert!((w_on - 87.41).abs() < 0.5, "ω²_slope {w_on}");
    assert!(off.ofv - on.ofv > 0.5, "{} vs {}", off.ofv, on.ofv);
    assert!(on.converged);
    let w = on
        .warnings
        .iter()
        .find(|w| w.contains("re-inflated to the initial value"))
        .unwrap_or_else(|| panic!("no collapse warning in {:?}", on.warnings));
    assert!(w.contains("the random effect ETA_SLOPE (variance "), "{w}");
}

/// The world of a rail restart that collapses a variance (#3005 review): `ETA_INT` (start
/// 1e-4) stops just above its guard and is downhill there; every rail restart lands it on
/// the guard but throws `ETA_SLOPE` (start 600) to 0.0062, where raising it is downhill;
/// every collapse restart re-inflates `ETA_SLOPE` and, while `int_bounces`, leaves `ETA_INT`
/// short of its guard again. Each evaluation lowers the OFV by 0.5 from the last run.
/// Returns the report and the call log (`outer_maxiter`, packed ETA_SLOPE, packed ETA_INT).
fn rail_then_collapse_world(int_bounces: bool) -> (Stub, Vec<(usize, f64, f64)>) {
    use std::cell::RefCell;
    let mut start = collapse_start();
    start.omega = OmegaMatrix::from_diagonal(
        &[1e-4, 600.0, 600.0],
        vec!["ETA_INT".into(), "ETA_SLOPE".into(), "ETA_FIX".into()],
    );
    let g = OMEGA_CHOL_PACKED_LOWER;
    let s = packed_of(&start, "ETA_SLOPE");
    let collapsed = 0.5 * 0.00622f64.ln();
    let short = with_packed_at(&start, "ETA_INT", g + 1.2);
    let options = FitOptions::default();
    let log: RefCell<Vec<(usize, f64, f64)>> = RefCell::new(Vec::new());
    let ofv = RefCell::new(100.0);
    let at = RefCell::new((f64::NAN, f64::NAN));
    let r = escape_floor(
        &start,
        &options,
        44,
        |p, o| {
            let (slope, int) = (packed_of(p, "ETA_SLOPE"), packed_of(p, "ETA_INT"));
            log.borrow_mut().push((o.outer_maxiter, slope, int));
            let n = log.borrow().len();
            let last = *ofv.borrow();
            let params = if n == 1 {
                short.clone()
            } else if o.outer_maxiter == 0 {
                // A probe (any point other than the last run's) is downhill by 0.5, a
                // re-score is level.
                let probing = *at.borrow() != (slope, int);
                return Stub {
                    params: p.clone(),
                    ofv: if probing { last - 0.5 } else { last },
                    ..landed()
                };
            } else if (int - (g + RAIL_RESTART_ABOVE)).abs() < 1e-12 {
                // A rail restart: ETA_INT onto the guard, ETA_SLOPE collapsed.
                let q = at_rail_params(
                    p,
                    &start,
                    &[coordinate_names(p)
                        .iter()
                        .position(|n| n == "ETA_INT")
                        .expect("ETA_INT")],
                    0.0,
                );
                with_packed_at(&q, "ETA_SLOPE", collapsed)
            } else if (slope - s).abs() < 1e-12 {
                // A collapse restart.
                if int_bounces {
                    with_packed_at(p, "ETA_INT", g + 1.2)
                } else {
                    p.clone()
                }
            } else {
                panic!("unexpected run {n}: slope {slope}, int {int}");
            };
            let next = last - 1.0;
            *ofv.borrow_mut() = next;
            *at.borrow_mut() = (
                packed_of(&params, "ETA_SLOPE"),
                packed_of(&params, "ETA_INT"),
            );
            Stub {
                params,
                ofv: if n == 1 { 100.0 } else { next },
                ..landed()
            }
        },
        |_| 0.0,
    );
    (r, log.into_inner())
}

/// #3005 review: a rail restart is a full outer optimization, and the run it adopts can
/// collapse a variance. Collapse completion reads that run again: here it re-inflates
/// `ETA_SLOPE`, and the rail stage, given nothing new, does not run a second time. The
/// report carries both stages' warnings and is converged.
#[test]
fn a_rail_restart_that_collapses_a_variance_is_read_by_collapse_completion_again() {
    let _g = rail_lock();
    let (r, log) = rail_then_collapse_world(false);
    let runs: Vec<usize> = log
        .iter()
        .enumerate()
        .filter(|(_, c)| c.0 > 0)
        .map(|(i, _)| i + 1)
        .collect();
    assert_eq!(
        log.len(),
        7,
        "fit; rail re-score, probe, restart; collapse re-score, probe, restart: {log:?}"
    );
    assert_eq!(runs, vec![1, 4, 7], "three full runs");
    assert!(r.converged, "{:?}", r.warnings);
    assert_eq!(r.warnings.len(), 2, "{:?}", r.warnings);
    assert!(r.warnings[0].contains("resumed from just above the guard"));
    assert!(
        r.warnings[1].contains("ETA_SLOPE") && r.warnings[1].contains("re-inflated"),
        "{}",
        r.warnings[1]
    );
}

/// The bound: when the rail stage still adopts a run in the last of `COMPLETION_ROUNDS`
/// rounds, that run is verified for collapse without a restart; its downhill probe leaves it
/// reported but not converged. Without the verification this fit came back `converged`
/// with `ETA_SLOPE` collapsed.
#[test]
fn a_collapse_left_by_the_last_rail_restart_is_verified_and_not_converged() {
    let _g = rail_lock();
    assert_eq!(COMPLETION_ROUNDS, 2, "the call log below is two rounds");
    let (r, log) = rail_then_collapse_world(true);
    assert_eq!(
        log.len(),
        12,
        "fit; rail ×3; collapse ×3; rail ×3; collapse verification: re-score, probe: {log:?}"
    );
    assert!(
        log[10..].iter().all(|c| c.0 == 0),
        "the verification restarts nothing: {log:?}"
    );
    assert!(!r.converged);
    let last = r.warnings.last().expect("a warning");
    assert!(
        last.contains("ETA_SLOPE") && last.contains("not resumed again"),
        "{last}"
    );
}

/// The world of a collapse restart that floors a θ (#3005 review, round 2): the stop has
/// `ETA_SLOPE` collapsed (raising it is downhill); the collapse restart re-inflates it, is
/// lower, and leaves `TVE0` (start 55, `≥ 0`) on the log-packing floor, where the θ probe is
/// downhill; the θ restart ends at `theta_restart_ofv` with TVE0 = 50. Returns the report
/// and the call log (`outer_maxiter`, TVE0).
fn collapse_then_floor_world(theta_restart_ofv: f64) -> (Stub, Vec<(usize, f64)>) {
    use std::cell::RefCell;
    let start = collapse_start();
    let s = packed_of(&start, "ETA_SLOPE");
    let stop = with_packed_at(&start, "ETA_SLOPE", 0.5 * 0.00622f64.ln());
    let options = FitOptions::default();
    let log: RefCell<Vec<(usize, f64)>> = RefCell::new(Vec::new());
    let r = escape_floor(
        &start,
        &options,
        44,
        |p, o| {
            let (tve0, slope) = (p.theta[0], packed_of(p, "ETA_SLOPE"));
            log.borrow_mut().push((o.outer_maxiter, tve0));
            let n = log.borrow().len();
            let (ofv, params) = match (n, o.outer_maxiter) {
                (1, _) => (100.0, stop.clone()),
                (2, 0) => (100.0, p.clone()),
                (3, 0) if (slope - (s - collapse_depth())).abs() < 1e-12 => (99.5, p.clone()),
                (4, m) if m > 0 && (slope - s).abs() < 1e-12 => {
                    let mut q = p.clone();
                    q.theta[0] = LOG_PACK_FLOOR;
                    (98.0, q)
                }
                // The θ floor escape on the adopted run: probe at 1/1000 of TVE0's start …
                (5, 0) if (tve0 - FLOOR_PROBE_REL * 55.0).abs() < 1e-12 => (97.5, p.clone()),
                // … and the restart from TVE0's start.
                (6, m) if m > 0 && (tve0 - 55.0).abs() < 1e-9 => {
                    let mut q = p.clone();
                    q.theta[0] = 50.0;
                    (theta_restart_ofv, q)
                }
                other => panic!("unexpected call {other:?} (TVE0 {tve0}, slope {slope})"),
            };
            Stub {
                params,
                ofv,
                ..landed()
            }
        },
        |_| 0.0,
    );
    (r, log.into_inner())
}

/// #3005 review, round 2: a run a completion adopts goes through the θ floor escape again.
/// Here the collapse restart is lower but floors TVE0 with the objective still falling;
/// the θ restart is adopted, carries the collapse warning ahead of its own, and — being a
/// new run — is read by both completions again (no candidates, so no calls). When the θ
/// restart ends no lower, the collapse restart is kept, not converged. Without the second
/// escape this fit came back `converged` at 98 with TVE0 on the floor.
#[test]
fn a_completion_restart_that_floors_a_theta_goes_through_the_floor_escape_again() {
    let _g = rail_lock();
    let (r, log) = collapse_then_floor_world(96.0);
    assert_eq!(
        log.len(),
        6,
        "fit; collapse re-score, probe, restart; θ probe, θ restart: {log:?}"
    );
    assert_eq!(r.ofv, 96.0);
    assert_eq!(r.params.theta[0], 50.0);
    assert!(r.converged, "{:?}", r.warnings);
    assert_eq!(r.warnings.len(), 2, "{:?}", r.warnings);
    assert!(r.warnings[0].contains("re-inflated"), "{}", r.warnings[0]);
    assert!(
        r.warnings[1].contains("TVE0") && r.warnings[1].contains("log-packing floor"),
        "{}",
        r.warnings[1]
    );

    let (r, log) = collapse_then_floor_world(98.5);
    assert_eq!(log.len(), 6, "{log:?}");
    assert_eq!(r.ofv, 98.0, "the collapse restart is kept");
    assert_eq!(r.params.theta[0], LOG_PACK_FLOOR);
    assert!(!r.converged);
    assert!(
        r.warnings.iter().any(|w| w.contains("re-inflated"))
            && r.warnings
                .last()
                .is_some_and(|w| w.contains("TVE0") && w.contains("not reported as converged")),
        "{:?}",
        r.warnings
    );
}
