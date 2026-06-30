//! `ode_template NAME(...)` disposition generation (#322 Phase 0b).
//!
//! `ode_template two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)` in
//! `[structural_model]` is **lowering sugar**: ferx generates the standard
//! disposition ODE for the named model and feeds it through the ordinary ODE
//! pipeline, so the user gets the explicit ODE form without hand-writing it.
//! There is no new runtime path — `ode_template` desugars to exactly the
//! `ode(obs_cmt=…, states=[…])` + `[odes]` + `[scaling] obs_scale=…` a user
//! would type by hand (see `parser::model_parser::apply_ode_template`).
//!
//! The transcription rules are the ones codified and verified by
//! `tests/analytical_ode_equivalence.rs` (ferx-r#127): ODE states carry
//! **amounts**; the observed concentration is read out via
//! `obs_scale = V` (`V1` for multi-compartment models); inter-compartmental
//! flux uses micro-constants `k10 = CL/V1`, `k12 = Q/V1`, `k21 = Q/V2`, …;
//! absorption adds a `depot` state (`-KA*depot` out, `+KA*depot` into central).
//! Bioavailability `F` and lag time are applied by the engine at the dose
//! (reserved PK slots), never baked into the RHS — so they are declared as
//! individual parameters by the user, exactly as for a hand-written ODE model.
//!
//! `ode_template`'s parameter signature matches the analytical `pk NAME(...)`
//! signature for the same model, including `ka` for the oral routes: even when
//! the user overrides the depot equation with `transit(...)`, the generated
//! `central` equation still needs the `ka` depot→central transfer constant, so
//! the generated model is runnable as written.

use crate::types::PkModel;
use std::collections::HashMap;

/// A standard PK disposition lowered from `ode_template NAME(...)` to the
/// hand-written ODE form.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedDisposition {
    /// State compartment names, in order (e.g. `["depot", "central", "periph"]`).
    pub states: Vec<String>,
    /// Observed compartment name. Always `central` for the standard models.
    pub obs_cmt: String,
    /// `(state, full "d/dt(state) = …" line)` for each generated state equation.
    pub odes: Vec<(String, String)>,
    /// `[scaling] obs_scale = <expr>` right-hand side — the central volume.
    pub obs_scale: String,
}

/// Generate the disposition ODE for `ode_template model_name(params)`.
///
/// `params` maps each lowercased role (`cl`, `v1`, `ka`, …) to the user's
/// individual-parameter variable name. Every required role must be present and
/// no extra roles are allowed — a missing or unknown role is a parse error
/// (matching the analytical `pk` model's required/unknown-parameter rules), so
/// the generated equations never reference an unmapped name.
pub fn generate(
    model_name: &str,
    params: &HashMap<String, String>,
) -> Result<GeneratedDisposition, String> {
    // Name → model and the required-role set both come from the shared analytical
    // `PkModel` tables (`from_name` / `required_pk_params`), so `ode_template`'s
    // accepted names and required parameters can never drift from the analytical
    // `pk NAME(...)` signature for the same model (Ron #363). The role names are
    // the conventional keys (`cl`, `v1`, `ka`, …) carried alongside each slot.
    let model = PkModel::from_name(model_name).ok_or_else(|| {
        format!(
            "Unknown ode_template model: {model_name}. Valid names are one_cpt_iv, \
             one_cpt_oral, one_cpt_transit, two_cpt_iv, two_cpt_oral, two_cpt_transit, \
             three_cpt_iv, three_cpt_oral."
        )
    })?;
    let name = model.canonical_name();
    let required: Vec<&'static str> = model
        .required_pk_params()
        .iter()
        .map(|(_, role)| *role)
        .collect();

    for &role in &required {
        if !params.contains_key(role) {
            return Err(format!(
                "ode_template {name} requires `{role}`, which is not mapped. \
                 Map it as `{role}=VARNAME` in ode_template {name}(...). \
                 Required parameters: {}.",
                required.join(", ")
            ));
        }
    }
    let mut extra: Vec<&str> = params
        .keys()
        .map(String::as_str)
        .filter(|k| !required.contains(k))
        .collect();
    if !extra.is_empty() {
        extra.sort_unstable();
        return Err(format!(
            "ode_template {name}: unknown parameter(s) `{}`; valid names are {}.",
            extra.join(", "),
            required.join(", ")
        ));
    }

    // Safe after the required-role check above.
    let g = |role: &str| params.get(role).expect("required role present").as_str();

    let dt = |state: &str, rhs: String| (state.to_string(), format!("d/dt({state}) = {rhs}"));

    let (states, obs_scale, odes): (Vec<&str>, &str, Vec<(String, String)>) = match model {
        PkModel::OneCptTransit => {
            // The analytic `pk one_cpt_transit` desugars to the Savic transit forcing
            // delivered straight into central (#386), the ODE analogue of the
            // exponential-tilting closed form.
            let (cl, v, n, mtt) = (g("cl"), g("v"), g("n"), g("mtt"));
            (
                vec!["central"],
                v,
                vec![dt(
                    "central",
                    format!("transit(n={n}, mtt={mtt}) - ({cl}/{v}) * central"),
                )],
            )
        }
        PkModel::OneCptIv => {
            let (cl, v) = (g("cl"), g("v"));
            (
                vec!["central"],
                v,
                vec![dt("central", format!("-({cl}/{v}) * central"))],
            )
        }
        PkModel::OneCptOral => {
            let (cl, v, ka) = (g("cl"), g("v"), g("ka"));
            (
                vec!["depot", "central"],
                v,
                vec![
                    dt("depot", format!("-{ka} * depot")),
                    dt("central", format!("{ka} * depot - ({cl}/{v}) * central")),
                ],
            )
        }
        PkModel::TwoCptIv => {
            let (cl, v1, q, v2) = (g("cl"), g("v1"), g("q"), g("v2"));
            (
                vec!["central", "periph"],
                v1,
                vec![
                    dt(
                        "central",
                        format!("-({cl}/{v1} + {q}/{v1}) * central + ({q}/{v2}) * periph"),
                    ),
                    dt(
                        "periph",
                        format!("({q}/{v1}) * central - ({q}/{v2}) * periph"),
                    ),
                ],
            )
        }
        PkModel::TwoCptOral => {
            let (cl, v1, q, v2, ka) = (g("cl"), g("v1"), g("q"), g("v2"), g("ka"));
            (
                vec!["depot", "central", "periph"],
                v1,
                vec![
                    dt("depot", format!("-{ka} * depot")),
                    dt(
                        "central",
                        format!(
                            "{ka} * depot - ({cl}/{v1} + {q}/{v1}) * central + ({q}/{v2}) * periph"
                        ),
                    ),
                    dt(
                        "periph",
                        format!("({q}/{v1}) * central - ({q}/{v2}) * periph"),
                    ),
                ],
            )
        }
        PkModel::TwoCptTransit => {
            // The analytic `pk two_cpt_transit` desugars to the Savic transit forcing
            // delivered straight into central, on a 2-cpt disposition (#386 PR D) —
            // the ODE analogue of the exponential-tilting closed form.
            let (cl, v1, q, v2, n, mtt) = (g("cl"), g("v1"), g("q"), g("v2"), g("n"), g("mtt"));
            (
                vec!["central", "periph"],
                v1,
                vec![
                    dt(
                        "central",
                        format!(
                            "transit(n={n}, mtt={mtt}) \
                             - ({cl}/{v1} + {q}/{v1}) * central + ({q}/{v2}) * periph"
                        ),
                    ),
                    dt(
                        "periph",
                        format!("({q}/{v1}) * central - ({q}/{v2}) * periph"),
                    ),
                ],
            )
        }
        PkModel::ThreeCptIv => {
            let (cl, v1, q2, v2, q3, v3) = (g("cl"), g("v1"), g("q2"), g("v2"), g("q3"), g("v3"));
            (
                vec!["central", "periph1", "periph2"],
                v1,
                vec![
                    dt(
                        "central",
                        format!(
                            "-({cl}/{v1} + {q2}/{v1} + {q3}/{v1}) * central \
                             + ({q2}/{v2}) * periph1 + ({q3}/{v3}) * periph2"
                        ),
                    ),
                    dt(
                        "periph1",
                        format!("({q2}/{v1}) * central - ({q2}/{v2}) * periph1"),
                    ),
                    dt(
                        "periph2",
                        format!("({q3}/{v1}) * central - ({q3}/{v3}) * periph2"),
                    ),
                ],
            )
        }
        PkModel::ThreeCptOral => {
            let (cl, v1, q2, v2, q3, v3, ka) = (
                g("cl"),
                g("v1"),
                g("q2"),
                g("v2"),
                g("q3"),
                g("v3"),
                g("ka"),
            );
            (
                vec!["depot", "central", "periph1", "periph2"],
                v1,
                vec![
                    dt("depot", format!("-{ka} * depot")),
                    dt(
                        "central",
                        format!(
                            "{ka} * depot - ({cl}/{v1} + {q2}/{v1} + {q3}/{v1}) * central \
                             + ({q2}/{v2}) * periph1 + ({q3}/{v3}) * periph2"
                        ),
                    ),
                    dt(
                        "periph1",
                        format!("({q2}/{v1}) * central - ({q2}/{v2}) * periph1"),
                    ),
                    dt(
                        "periph2",
                        format!("({q3}/{v1}) * central - ({q3}/{v3}) * periph2"),
                    ),
                ],
            )
        }
    };

    Ok(GeneratedDisposition {
        states: states.into_iter().map(String::from).collect(),
        obs_cmt: "central".to_string(),
        odes,
        obs_scale: obs_scale.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn one_cpt_iv_disposition() {
        let g = generate("one_cpt_iv", &map(&[("cl", "CL"), ("v", "V")])).unwrap();
        assert_eq!(g.states, vec!["central"]);
        assert_eq!(g.obs_cmt, "central");
        assert_eq!(g.obs_scale, "V");
        assert_eq!(g.odes[0].0, "central");
        assert_eq!(g.odes[0].1, "d/dt(central) = -(CL/V) * central");
    }

    #[test]
    fn two_cpt_oral_uses_mapped_var_names() {
        // Non-default names exercise the substitution (clearance is CLP here).
        let g = generate(
            "two_cpt_oral",
            &map(&[
                ("cl", "CLP"),
                ("v1", "VC"),
                ("q", "QQ"),
                ("v2", "VP"),
                ("ka", "KABS"),
            ]),
        )
        .unwrap();
        assert_eq!(g.states, vec!["depot", "central", "periph"]);
        assert_eq!(g.obs_scale, "VC");
        let lines: Vec<&str> = g.odes.iter().map(|(_, l)| l.as_str()).collect();
        assert_eq!(lines[0], "d/dt(depot) = -KABS * depot");
        assert_eq!(
            lines[1],
            "d/dt(central) = KABS * depot - (CLP/VC + QQ/VC) * central + (QQ/VP) * periph"
        );
        assert_eq!(
            lines[2],
            "d/dt(periph) = (QQ/VC) * central - (QQ/VP) * periph"
        );
    }

    #[test]
    fn three_cpt_iv_has_three_states_and_two_peripherals() {
        let g = generate(
            "three_cpt_iv",
            &map(&[
                ("cl", "CL"),
                ("v1", "V1"),
                ("q2", "Q2"),
                ("v2", "V2"),
                ("q3", "Q3"),
                ("v3", "V3"),
            ]),
        )
        .unwrap();
        assert_eq!(g.states, vec!["central", "periph1", "periph2"]);
        assert_eq!(g.obs_scale, "V1");
        // Assert the full micro-constant RHS, not just the shape — a wrong
        // cross-term (q2/q3 ↔ v2/v3 swap) would otherwise only surface in the
        // slow equivalence test.
        let lines: Vec<&str> = g.odes.iter().map(|(_, l)| l.as_str()).collect();
        assert_eq!(
            lines[0],
            "d/dt(central) = -(CL/V1 + Q2/V1 + Q3/V1) * central + (Q2/V2) * periph1 + (Q3/V3) * periph2"
        );
        assert_eq!(
            lines[1],
            "d/dt(periph1) = (Q2/V1) * central - (Q2/V2) * periph1"
        );
        assert_eq!(
            lines[2],
            "d/dt(periph2) = (Q3/V1) * central - (Q3/V3) * periph2"
        );
    }

    #[test]
    fn compartment_aliases_resolve() {
        // The long `*_compartment_*` aliases generate the same disposition.
        let a = generate(
            "two_cpt_iv",
            &map(&[("cl", "CL"), ("v1", "V1"), ("q", "Q"), ("v2", "V2")]),
        )
        .unwrap();
        let b = generate(
            "two_compartment_iv",
            &map(&[("cl", "CL"), ("v1", "V1"), ("q", "Q"), ("v2", "V2")]),
        )
        .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn missing_required_role_errors() {
        // Oral model without `ka` — the generated central eqn would reference an
        // unmapped transfer constant, so this must be rejected (not silently run).
        let err = generate(
            "two_cpt_oral",
            &map(&[("cl", "CL"), ("v1", "V1"), ("q", "Q"), ("v2", "V2")]),
        )
        .unwrap_err();
        assert!(err.contains("requires `ka`"), "got: {err}");
    }

    #[test]
    fn unknown_role_errors() {
        let err = generate(
            "one_cpt_iv",
            &map(&[("cl", "CL"), ("v", "V"), ("ka", "KA")]),
        )
        .unwrap_err();
        assert!(err.contains("unknown parameter"), "got: {err}");
        assert!(err.contains("ka"), "got: {err}");
    }

    #[test]
    fn unknown_model_errors() {
        let err = generate("four_cpt_oral", &map(&[("cl", "CL")])).unwrap_err();
        assert!(err.contains("Unknown ode_template model"), "got: {err}");
    }

    #[test]
    fn required_roles_track_pkmodel_required_params() {
        // The drift guard Ron asked for (#363): `generate` derives its required
        // roles from `PkModel::required_pk_params` rather than a private copy, so it
        // must require *exactly* that set for every model — accepting the full set,
        // and rejecting the omission of any single required role by name. Add a 7th
        // model or rename a role and this fails unless `generate` is updated too.
        for model in [
            PkModel::OneCptIv,
            PkModel::OneCptOral,
            PkModel::TwoCptIv,
            PkModel::TwoCptOral,
            PkModel::ThreeCptIv,
            PkModel::ThreeCptOral,
        ] {
            let name = model.canonical_name();
            let roles: Vec<&str> = model.required_pk_params().iter().map(|(_, r)| *r).collect();

            // Exactly the required roles mapped → generates successfully, and the
            // generated state count matches the model's compartment count.
            let full: Vec<(&str, &str)> = roles.iter().map(|r| (*r, "X")).collect();
            let g = generate(name, &map(&full))
                .unwrap_or_else(|e| panic!("{name} should accept its required roles: {e}"));
            assert_eq!(g.states.len(), g.odes.len(), "{name}: one ODE per state");

            // Drop each required role in turn → a parse error naming that role.
            for omit in &roles {
                let partial: Vec<(&str, &str)> = roles
                    .iter()
                    .filter(|r| *r != omit)
                    .map(|r| (*r, "X"))
                    .collect();
                let err = generate(name, &map(&partial)).unwrap_err();
                assert!(
                    err.contains(&format!("requires `{omit}`")),
                    "{name}: omitting `{omit}` should error naming it, got: {err}"
                );
            }
        }
    }
}
