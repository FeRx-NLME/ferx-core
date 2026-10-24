//! The span certificate on an ODE model in the periods layout (#1877): a kappa on
//! one occasion, and a level of that occasion, move the next occasion's records
//! through the state they carry over the boundary. `spanned` measures a unit's
//! kappa column over the whole subject, against every level of the subject.
//! Binder-only fixtures: FD of the f64 predictor, no fit.

use super::absorption::{agrees_with_joint_oracle, jacobian, try_bind, unit_rank};
use super::*;
use crate::parser::model_parser::LevelContrast;

/// Three subjects, one study each, two periods: a bolus of 100 at 0 opens
/// occasion 1 and one at 12 opens occasion 2, with residual drug from the first
/// (`CL / V` = 0.05/h). Occasion 2 is sampled at 12 (after its dose, so its
/// level is the one the dose row reads), 14, 16 and 20; occasion 1 at the last
/// `occ1` of 1, 2, 4 and 8. `W`, the record index, varies within each occasion.
/// The dose rows carry their occasion's `OCC`, so a branch on it switches with
/// the kappa.
fn periods_pop(occ1: usize) -> Population {
    const OCC1: [f64; 4] = [1.0, 2.0, 4.0, 8.0];
    const OCC2: [f64; 4] = [12.0, 14.0, 16.0, 20.0];
    let mut pop = population(3, 0);
    for s in pop.subjects.iter_mut() {
        let rows: Vec<(f64, u32)> = OCC1[4 - occ1..]
            .iter()
            .map(|&t| (t, 1))
            .chain(OCC2.iter().map(|&t| (t, 2)))
            .collect();
        let n = rows.len();
        s.obs_times = rows.iter().map(|r| r.0).collect();
        s.occasions = rows.iter().map(|r| r.1).collect();
        s.obs_covariates = rows
            .iter()
            .enumerate()
            .map(|(j, r)| {
                HashMap::from([
                    ("OCC".to_string(), r.1 as f64),
                    ("W".to_string(), 1.0 + j as f64),
                ])
            })
            .collect();
        s.observations = vec![1.0; n];
        s.obs_cmts = vec![1; n];
        s.cens = vec![0; n];
        s.doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ];
        s.dose_occasions = vec![1, 2];
        s.dose_covariates = [1.0, 2.0]
            .iter()
            .map(|&o| HashMap::from([("OCC".to_string(), o), ("W".to_string(), 0.0)]))
            .collect();
    }
    pop
}

/// A one-compartment bolus ODE with `KAPPA_CL` declared and a `[STUDY, OCC]`
/// block: `ip` the `[individual_parameters]` body after `V`, `y` the readout.
/// `ETA_V` sits on an unread `Z`, so no η shares a direction with the kappa.
fn ode_model(ip: &str, y: &str, contrast: &str) -> String {
    let modifier = if contrast.is_empty() {
        String::new()
    } else {
        format!(", contrast = {contrast}")
    };
    format!(
        r#"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(20.0, 0.1, 500.0)
  theta PLACEBO[STUDY, OCC{modifier}](0.0, -10.0, 10.0)
  omega ETA_V ~ 0.09
  kappa KAPPA_CL ~ 0.09
  sigma ADD ~ 0.1
[individual_parameters]
  V = TVV
  Z = TVV * exp(ETA_V)
{ip}
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y = {y}
[error_model]
  DV ~ additive(ADD)
"#
    )
}

/// The measured clause over occasions, up to its caveat.
const MEASURED: &str = "the random effect `KAPPA_CL` moves `y`, at every record of each \
                        occasion, exactly as some combination of this block's levels does";

/// #1877. Three ODE periods fixtures, each against the joint oracle, whose kappa
/// column is the whole subject's:
///
/// - **A** (absorbed): the kappa and the block on `CL`, in branches on `W` the
///   walk cannot name a site for. Each occasion's kappa and level are one
///   function, carry-over included. Refused with the measured clause.
/// - **B** (identified): occasion 1 sampled once; there the block shifts the
///   readout, while from occasion 2 it rides on `CL` with the kappa. On its own
///   one record occasion 1's kappa is proportional to its level (residual 0),
///   but it also moves occasion 2's records through the drug it leaves behind,
///   which neither level reproduces (residual 0.45 over the subject). Binds.
/// - **C** (absorbed): the kappa on `F1`, the block scaling the readout. Occasion
///   1's kappa moves occasion 2's records by its residual drug, which is
///   proportional to occasion 2's level there — a one-compartment state is one
///   exponential — and so is absorbed by the two levels together, not by its own.
///   Refused with the measured clause.
///
/// The oracle verdicts are asserted too, so the three straddle absorption.
///
/// Mutations — the unit's columns restricted to its own records, the pre-#1877
/// reading (B is refused); the unit's own levels only (C binds); both, with every
/// level of the subject on the unit's records (B is refused). Equivalent, so not
/// killed: the kappa bumped on every occasion at once (#1874's M8), whose column
/// is the sum of the occasions' and lies in the span whenever each does; before
/// #1877, under the restriction, A killed it.
#[test]
fn a_kappa_carried_over_is_measured_over_the_whole_subject() {
    let a_ip = "  if (W > 2.5) {\n    CL = TVCL * exp(PLACEBO + KAPPA_CL)\n  } else {\n    \
                CL = 2 * TVCL * exp(PLACEBO + KAPPA_CL)\n  }";
    let b_ip = "  if (OCC > 1) {\n    CL = TVCL * exp(PLACEBO + KAPPA_CL)\n    P1 = 0\n  } \
                else {\n    CL = TVCL * exp(KAPPA_CL)\n    P1 = PLACEBO\n  }";
    let c_ip = "  CL = TVCL\n  F1 = exp(KAPPA_CL)";
    let cells: [(&str, &str, &str, usize, bool); 3] = [
        ("A", a_ip, "central / V", 4, true),
        ("B", b_ip, "central / V + P1", 1, false),
        ("C", c_ip, "central / V * exp(PLACEBO)", 4, true),
    ];
    let mut wrong = Vec::new();
    for (tag, ip, y, occ1, absorbed) in cells {
        let pop = periods_pop(occ1);
        let text = |c: &str| ode_model(ip, y, c);
        let jac = jacobian(&text("none"), &pop);
        let free = unit_rank(&jac, "KAPPA_CL", None);
        let under = unit_rank(&jac, "KAPPA_CL", Some(LevelContrast::Unconstrained));
        if free == 0 || (under == 0) != absorbed {
            wrong.push(format!(
                "{tag}: fixture premise, rank {free} without the block, {under} with it"
            ));
        }
        for c in ["sum_to_zero", "ref", "none"] {
            match (try_bind(&text(c), &pop), absorbed) {
                (Err(e), true) if e.contains(MEASURED) => {}
                (Ok(_), false) => {}
                (got, _) => wrong.push(format!("{tag} {c}: {got:?}")),
            }
        }
        agrees_with_joint_oracle(tag, &text, &pop, &mut wrong);
    }
    assert!(wrong.is_empty(), "carry-over:\n{}", wrong.join("\n"));
}
