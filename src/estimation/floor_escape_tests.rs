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
    );
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
