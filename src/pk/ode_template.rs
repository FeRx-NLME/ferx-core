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

/// Canonical model key for a name, accepting the same aliases as the analytical
/// `pk` parser (`one_cpt_iv` / `one_compartment_iv`, …).
fn canonical(name: &str) -> Option<&'static str> {
    match name {
        "one_cpt_iv" | "one_compartment_iv" => Some("one_cpt_iv"),
        "one_cpt_oral" | "one_compartment_oral" => Some("one_cpt_oral"),
        "two_cpt_iv" | "two_compartment_iv" => Some("two_cpt_iv"),
        "two_cpt_oral" | "two_compartment_oral" => Some("two_cpt_oral"),
        "three_cpt_iv" | "three_compartment_iv" => Some("three_cpt_iv"),
        "three_cpt_oral" | "three_compartment_oral" => Some("three_cpt_oral"),
        _ => None,
    }
}

/// The role names (lowercased keys, as in `pk NAME(...)`) each model requires.
fn required_roles(model: &str) -> &'static [&'static str] {
    match model {
        "one_cpt_iv" => &["cl", "v"],
        "one_cpt_oral" => &["cl", "v", "ka"],
        "two_cpt_iv" => &["cl", "v1", "q", "v2"],
        "two_cpt_oral" => &["cl", "v1", "q", "v2", "ka"],
        "three_cpt_iv" => &["cl", "v1", "q2", "v2", "q3", "v3"],
        "three_cpt_oral" => &["cl", "v1", "q2", "v2", "q3", "v3", "ka"],
        _ => &[],
    }
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
    let model = canonical(model_name).ok_or_else(|| {
        format!(
            "Unknown ode_template model: {model_name}. Valid names are one_cpt_iv, \
             one_cpt_oral, two_cpt_iv, two_cpt_oral, three_cpt_iv, three_cpt_oral."
        )
    })?;

    let required = required_roles(model);
    for &role in required {
        if !params.contains_key(role) {
            return Err(format!(
                "ode_template {model} requires `{role}`, which is not mapped. \
                 Map it as `{role}=VARNAME` in ode_template {model}(...). \
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
            "ode_template {model}: unknown parameter(s) `{}`; valid names are {}.",
            extra.join(", "),
            required.join(", ")
        ));
    }

    // Safe after the required-role check above.
    let g = |role: &str| params.get(role).expect("required role present").as_str();

    let dt = |state: &str, rhs: String| (state.to_string(), format!("d/dt({state}) = {rhs}"));

    let (states, obs_scale, odes): (Vec<&str>, &str, Vec<(String, String)>) = match model {
        "one_cpt_iv" => {
            let (cl, v) = (g("cl"), g("v"));
            (
                vec!["central"],
                v,
                vec![dt("central", format!("-({cl}/{v}) * central"))],
            )
        }
        "one_cpt_oral" => {
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
        "two_cpt_iv" => {
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
        "two_cpt_oral" => {
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
        "three_cpt_iv" => {
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
        "three_cpt_oral" => {
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
        _ => unreachable!("canonical() already gated the model name"),
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
}
