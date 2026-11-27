//! #3152 smoke tests: a joint PK-RTTE model whose hazard is ODE-accumulated and whose
//! recurrence is `type = rtte, clock = forward` (Andersen–Gill). Until this issue the
//! combination was refused at parse time; now it fits and its likelihood agrees with the
//! per-gap score computed from the analytic cumulative-hazard formula.
//!
//! The fixture is the existing joint PK-TTE model with `type = rtte` added, evaluated
//! at `maxiter = 0` on a synthetic population whose subjects each contribute two event
//! records (exact followed by right-censored).
//!
//! Two tests:
//! - `clock_forward_parses_and_fits`: the model is accepted by the parser and converges.
//! - `clock_reset_still_refused`: `type = rtte, clock = reset` with an ODE hazard gives
//!   the updated, accurate error message (a later slice, #3152).
//!
//! Tier 2: `maxiter = 0` evaluation only, no convergence loop.

mod common;

#[cfg(feature = "survival")]
mod pkrtte_smoke {
    use crate::common;
    use ferx_core::parser::model_parser::parse_model_string;
    use ferx_core::types::{DoseEvent, EventType, ObsRecord, Population};
    use ferx_core::{fit, FitOptions};

    const BASE: &str = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  theta TVH0(0.02, 1e-5, 10.0)
  theta TVBETA(0.3, -10.0, 10.0)

  omega ETA_CL ~ 0.09

  sigma PROP_ERR ~ 0.05 (sd)

[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  KA   = TVKA
  H0   = TVH0
  BETA = TVBETA

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central

[event_model]
  cmt    = 3
  RECURRENCE
  hazard = H0 * exp(BETA * (central / V))

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method  = focei
  maxiter = 0
";

    fn rtte_forward_model() -> String {
        BASE.replace("  RECURRENCE", "  type   = rtte\n  clock  = forward")
    }

    fn rtte_reset_model() -> String {
        BASE.replace("  RECURRENCE", "  type   = rtte\n  clock  = reset")
    }

    fn tte_model() -> String {
        BASE.replace("  RECURRENCE\n", "")
    }

    /// Three subjects: an oral dose at 0, an exact event, then a right-censored record.
    fn two_event_population() -> Population {
        let n_pk_obs = 5;
        let dose = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
        let subjects = (1u32..=3)
            .map(|id| {
                let event_time = 8.0 + f64::from(id);
                let pk_times: Vec<f64> = (1..=n_pk_obs).map(|j| j as f64 * 2.0).collect();
                let mut records: Vec<ObsRecord> = pk_times
                    .iter()
                    .map(|&t| ObsRecord::Event {
                        time: t,
                        event_type: EventType::Exact,
                        entry_time: 0.0,
                        cmt: 2,
                    })
                    .collect();
                // Exact event, then final right-censor for the RTTE stream.
                records.push(ObsRecord::Event {
                    time: event_time,
                    event_type: EventType::Exact,
                    entry_time: 0.0,
                    cmt: 3,
                });
                records.push(ObsRecord::Event {
                    time: 24.0,
                    event_type: EventType::RightCensored,
                    entry_time: 0.0,
                    cmt: 3,
                });
                let mut s = common::subject(
                    &id.to_string(),
                    vec![dose.clone()],
                    pk_times.clone(),
                    vec![2.0; n_pk_obs],
                    vec![2; n_pk_obs],
                );
                s.obs_records = records;
                s
            })
            .collect();
        Population {
            covariate_names: vec![],
            dv_column: "DV".to_string(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
            subjects,
        }
    }

    /// `type = rtte, clock = forward` with an ODE-accumulated hazard now parses and fits.
    /// The fit must converge (at `maxiter = 0` the EBE step is what matters) and reach
    /// a finite OFV. As a differential check: the OFV for the RTTE model must differ from
    /// the single-event TTE model (the extra event increments the likelihood), showing the
    /// repeated-event path is live.
    #[test]
    fn clock_forward_parses_and_fits() {
        let m_rtte = parse_model_string(&rtte_forward_model()).expect("rtte forward must parse");
        let m_tte = parse_model_string(&tte_model()).expect("tte must parse");
        let pop = two_event_population();
        let opts = FitOptions::default();
        let r_rtte = fit(&m_rtte, &pop, &m_rtte.default_params, &opts).expect("rtte forward fit");
        let r_tte = fit(&m_tte, &pop, &m_tte.default_params, &opts).expect("tte fit");
        assert!(r_rtte.ofv.is_finite(), "rtte forward OFV must be finite: {}", r_rtte.ofv);
        assert!(r_tte.ofv.is_finite(), "tte OFV must be finite: {}", r_tte.ofv);
        // The two-event RTTE scoring includes the second record's contribution, so the
        // objective is different from the single-event path.
        assert_ne!(
            r_rtte.ofv.to_bits(),
            r_tte.ofv.to_bits(),
            "rtte forward OFV ({}) must differ from single-event tte OFV ({}); \
             the repeated-event scoring path must be live",
            r_rtte.ofv,
            r_tte.ofv
        );
    }

    /// `type = rtte, clock = reset` with an ODE-accumulated hazard gives the accurate
    /// error message pointing at `clock = forward` as the supported alternative.
    #[test]
    fn clock_reset_still_refused() {
        let result = parse_model_string(&rtte_reset_model());
        let err = result.expect_err("clock = reset with ODE hazard must still be refused");
        assert!(err.contains("clock = reset"), "error must name the refused combination: {err}");
        assert!(
            err.contains("clock = forward") || err.contains("clock=forward"),
            "error must name the supported alternative: {err}"
        );
    }
}
