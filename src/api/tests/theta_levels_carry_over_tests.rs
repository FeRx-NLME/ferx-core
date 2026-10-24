//! The span certificate on an ODE model in the periods layout (#1877): a kappa on
//! one occasion, and a level of that occasion, move the next occasion's records
//! through the state they carry over the boundary. `spanned` measures a unit's
//! kappa column over the whole subject, against every level of the subject.
//! Binder-only fixtures: FD of the f64 predictor, no fit.

use super::absorption::{
    agrees_with_joint_oracle, jacobian, joint_oracle, try_bind, unit_rank, EXPLICIT,
};
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

/// Three subjects, one study each, three periods into an oral depot: occasion
/// 1 is a dose at 0 alone — 100, 60 or 150, so the subjects differ — and
/// occasions 2 and 3 open with a dose of 100 at 12 and at 24, observed at 12,
/// 14, 16, 20 and at 24, 26, 28, 32. `W`, the record index, is 0 on the first
/// dose row, so `1 + 0.1 * W` is 1 there.
fn dose_only_pop() -> Population {
    let mut pop = population(3, 0);
    for (i, s) in pop.subjects.iter_mut().enumerate() {
        let rows: Vec<(f64, u32)> = [12.0, 14.0, 16.0, 20.0]
            .iter()
            .map(|&t| (t, 2))
            .chain([24.0, 26.0, 28.0, 32.0].iter().map(|&t| (t, 3)))
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
            DoseEvent::new(0.0, [100.0, 60.0, 150.0][i], 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
        ];
        s.dose_occasions = vec![1, 2, 3];
        s.dose_covariates = [(1.0, 0.0), (2.0, 4.5), (3.0, 8.5)]
            .iter()
            .map(|&(o, w)| HashMap::from([("OCC".to_string(), o), ("W".to_string(), w)]))
            .collect();
    }
    pop
}

/// [`ode_model`] with an oral depot: the dose lands in `depot`, absorbed at
/// `KA = 0.5`/h, so a dose's residual twelve hours on is not proportional to a
/// fresh dose's absorption curve.
fn oral_model(ip: &str, contrast: &str) -> String {
    ode_model(ip, "central / V", contrast)
        .replace(
            "  ode(obs_cmt=central, states=[central])",
            "  ode(obs_cmt=central, states=[depot, central])",
        )
        .replace(
            "  d/dt(central) = -(CL/V) * central",
            "  d/dt(depot) = -0.5 * depot\n  d/dt(central) = 0.5 * depot - (CL/V) * central",
        )
}

/// #1877 review r1. A dose-only occasion has a κ group of its own
/// (`iov_occasion_groups`) and no record, so a unit list taken from the records
/// never measures it. In an ODE model such an occasion's kappa reaches `y` only
/// through its dose's attributes — the parameters between records are the
/// observed occasions' — so here the kappa is on `F1`, which from occasion 2 on
/// also reads the block, through a factor distributed over both (no site the
/// walk can name): occasion 2's and 3's kappas are their levels' twins.
///
/// - **D** (identified): occasion 1's kappa scales its dose, whose residual
///   moves occasion 2's and 3's records in a shape no level reproduces — dose 1
///   is long absorbed by 12, doses 2 and 3 are fresh. Binds.
/// - **D0** (absorbed under `none`, the twin): the kappa is multiplied by
///   `OCC - 1`, so occasion 1's column is zero and measures nothing. Refused
///   under `none` with the measured clause; under the global contrasts the sum of
///   the levels — a shift of doses 2 and 3 only, which no θ reproduces — is
///   coded out, and the oracle and the binder both bind. Judged contrast by
///   contrast against [`joint_oracle`], not with `agrees_with_joint_oracle`: its
///   auto clause fails here, `auto` resolving to `sum_to_zero_within` where global
///   absorbs nothing. That is the measured clause's `auto` and not this unit
///   list, since D0's one extra unit measures nothing.
///
/// Mutation — the units taken from the observation records only, the reading
/// before review (D is refused). The twin keeps a fix that binds every
/// dose-only layout visible.
#[test]
fn a_dose_only_occasion_is_measured_through_its_carry_over() {
    let ip = |k: &str| {
        format!("  CL = TVCL\n  G = 1 + 0.1 * W\n  F1 = exp(PLACEBO * (OCC - 1) * G + {k} * G)")
    };
    let cells = [
        ("D", ip("KAPPA_CL"), false),
        ("D0", ip("KAPPA_CL * (OCC - 1)"), true),
    ];
    let pop = dose_only_pop();
    let mut wrong = Vec::new();
    for (tag, ip, absorbed) in &cells {
        let text = |c: &str| oral_model(ip, c);
        let jac = jacobian(&text("none"), &pop);
        let free = unit_rank(&jac, "KAPPA_CL", None);
        let under = unit_rank(&jac, "KAPPA_CL", Some(LevelContrast::Unconstrained));
        if free == 0 || (under == 0) != *absorbed {
            wrong.push(format!(
                "{tag}: fixture premise, rank {free} without the block, {under} with it"
            ));
        }
        if !absorbed {
            for c in ["sum_to_zero", "ref", "none"] {
                if let Err(e) = try_bind(&text(c), &pop) {
                    wrong.push(format!("{tag} {c}: refused: {e}"));
                }
            }
            agrees_with_joint_oracle(tag, &text, &pop, &mut wrong);
            continue;
        }
        for (contrast, token) in EXPLICIT {
            let absorbs = !joint_oracle(&jac, contrast).0.is_empty();
            match (try_bind(&text(token), &pop), absorbs) {
                (Err(e), true) if e.contains(MEASURED) => {}
                (Ok(_), false) => {}
                (got, _) => wrong.push(format!("{tag} {token}: oracle {absorbs}, {got:?}")),
            }
        }
    }
    assert!(wrong.is_empty(), "dose-only:\n{}", wrong.join("\n"));
}
