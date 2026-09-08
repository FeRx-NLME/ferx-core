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

/// How the dose enters `central`, when it is not the template's own bolus,
/// first-order depot or absorption forcing (#1257).
///
/// A search that proposes Pharmpy's `ABSORPTION(ZO)` or `ABSORPTION(WEIBULL)`
/// is not swapping a `pk` template — neither has one. It is replacing the
/// **input term** of a generated disposition with an ODE input function, and
/// stating that as a value here keeps the transcription in one place: the
/// caller says which model, not which text.
///
/// Both non-default forms feed `central` directly, so they are accepted only
/// on a template that has no depot and no forcing of its own — the `*_iv`
/// family. Anything else is an error naming the clash rather than a model
/// with two input terms.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum InputForm {
    /// Whatever the named template does: a bolus into `central` (`*_iv`), a
    /// first-order `depot` (`*_oral`), or the transit / inverse-Gaussian
    /// forcing of `*_transit` / `*_ig`.
    #[default]
    Template,
    /// `zero_order(dur = …)` into `central` — NONMEM's modeled duration
    /// (`RATE = −2`, `D1`), Pharmpy's `ABSORPTION(ZO)`.
    ZeroOrder {
        /// The `[individual_parameters]` name holding the input duration.
        dur: String,
    },
    /// `weibull(td = …, beta = …)` into `central` — Pharmpy's
    /// `ABSORPTION(WEIBULL)`, whose depot-with-a-time-varying-`KAW` spelling
    /// integrates to the same input rate.
    Weibull {
        /// The Weibull scale parameter (Pharmpy's `LAMBDA`).
        td: String,
        /// The Weibull shape parameter (Pharmpy's `K`).
        beta: String,
    },
}

impl InputForm {
    /// How the form is named in an error message.
    pub fn label(&self) -> &'static str {
        match self {
            InputForm::Template => "the template's own",
            InputForm::ZeroOrder { .. } => "zero-order",
            InputForm::Weibull { .. } => "Weibull",
        }
    }
}

/// How `central` eliminates, when it is not the template's own first-order
/// `CL/V` (#1257).
///
/// The saturable forms are Pharmpy's parameterisation
/// (`modeling/odes.py::_do_michaelis_menten_elimination`): the flux out of
/// `central` is `(CLMM·KM/(KM + C) + CL)·A/V` with `C = A/V`, i.e. the usual
/// `Vmax·C/(KM + C)` with `Vmax = CLMM·KM`. Zero-order elimination has no
/// variant of its own — it is [`MichaelisMenten`](Self::MichaelisMenten) with
/// `KM` fixed far below the observed concentrations, which is also how
/// Pharmpy writes it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum EliminationForm {
    /// `CL/V · A`, fused into the outflow bracket with the inter-compartmental
    /// rate constants, exactly as the template writes it.
    #[default]
    FirstOrder,
    /// `CLMM·KM·C/(KM + C)` with `C = A/V`.
    ///
    /// The Michaelis-Menten clearance is the template's **own `cl` binding** —
    /// there is no second name. Pharmpy renames `CL` to `CLMM` at this point;
    /// ferx keeps the name, which is what carries the parameter's initial
    /// estimate and its η across the move. Taking a separate name here would
    /// leave the template's `cl` binding declared and read by nothing, i.e. a
    /// free parameter the objective cannot see.
    MichaelisMenten {
        /// The `[individual_parameters]` name holding the Michaelis constant.
        km: String,
    },
    /// `CL·C + CLMM·KM·C/(KM + C)` — Pharmpy's `MIX-FO-MM`. Here `CLMM` *is* a
    /// second parameter, alongside the template's `cl`.
    MixedFoMm {
        /// The saturable clearance.
        clmm: String,
        /// The Michaelis constant.
        km: String,
    },
}

impl EliminationForm {
    /// The MFL elimination mode this form spells.
    pub fn label(&self) -> &'static str {
        match self {
            EliminationForm::FirstOrder => "first-order",
            EliminationForm::MichaelisMenten { .. } => "Michaelis-Menten",
            EliminationForm::MixedFoMm { .. } => "mixed first-order / Michaelis-Menten",
        }
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
    generate_variant(
        model_name,
        params,
        &InputForm::Template,
        &EliminationForm::FirstOrder,
    )
}

/// [`generate`], with the central compartment's input and elimination terms
/// replaced (#1257).
///
/// This is the one transcription of a standard PK disposition into ODE text:
/// [`generate`] is this function at its defaults, and the
/// `generate_is_the_default_variant` test pins the two byte-for-byte. The
/// variants exist because `ABSORPTION(ZO)`, `ABSORPTION(WEIBULL)` and every
/// `ELIMINATION` other than `FO` have no analytic `pk` template at all — they
/// are reachable only as `[odes]` text, and a model-space search has to write
/// that text without owning a second copy of the disposition
/// ([`crate::edit::StructuralSpec`], `ferx-tools::modelsearch`).
///
/// Only the `central` equation changes; the depot and peripheral equations are
/// the template's own, which is why a caller writes exactly one override line.
pub fn generate_variant(
    model_name: &str,
    params: &HashMap<String, String>,
    input: &InputForm,
    elimination: &EliminationForm,
) -> Result<GeneratedDisposition, String> {
    // Name → model and the required-role set both come from the shared analytical
    // `PkModel` tables (`from_name` / `required_pk_params`), so `ode_template`'s
    // accepted names and required parameters can never drift from the analytical
    // `pk NAME(...)` signature for the same model (Ron #363). The role names are
    // the conventional keys (`cl`, `v1`, `ka`, …) carried alongside each slot.
    let model = PkModel::from_name(model_name).ok_or_else(|| {
        format!(
            "Unknown ode_template model: {model_name}. Valid names are one_cpt_iv, \
             one_cpt_oral, one_cpt_transit, one_cpt_ig, two_cpt_iv, two_cpt_oral, \
             two_cpt_transit, two_cpt_ig, three_cpt_iv, three_cpt_oral."
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

    // ── The topology, once ──────────────────────────────────────────────────
    // Everything that differs between the ten templates: whether there is a
    // depot (and its `ka`), what the template's own input term into `central`
    // is, which volume `central` is read out on, and the peripheral chain as
    // `(state, q, vp)`. Every equation below is composed from this, so a
    // variant cannot disagree with the plain form about where a compartment
    // connects — the one-implementation rule, applied to generated text.
    #[allow(clippy::type_complexity)]
    let (depot_ka, template_input, vc, periphs): (
        Option<&str>,
        String,
        &str,
        Vec<(&'static str, &str, &str)>,
    ) = match model {
        // The analytic `pk one_cpt_transit` / `pk *_ig` desugar to the Savic
        // transit (#386) and Freijer & Post inverse-Gaussian (#790) forcings
        // delivered straight into central — the ODE analogue of their
        // exponential-tilting closed forms.
        PkModel::OneCptTransit => (
            None,
            format!("transit(n={}, mtt={})", g("n"), g("mtt")),
            g("v"),
            vec![],
        ),
        PkModel::OneCptIg => (
            None,
            format!("igd(mat={}, cv2={})", g("mat"), g("cv2")),
            g("v"),
            vec![],
        ),
        PkModel::OneCptIv => (None, String::new(), g("v"), vec![]),
        PkModel::OneCptOral => (
            Some(g("ka")),
            format!("{} * depot", g("ka")),
            g("v"),
            vec![],
        ),
        PkModel::TwoCptIv => (
            None,
            String::new(),
            g("v1"),
            vec![("periph", g("q"), g("v2"))],
        ),
        PkModel::TwoCptOral => (
            Some(g("ka")),
            format!("{} * depot", g("ka")),
            g("v1"),
            vec![("periph", g("q"), g("v2"))],
        ),
        PkModel::TwoCptTransit => (
            None,
            format!("transit(n={}, mtt={})", g("n"), g("mtt")),
            g("v1"),
            vec![("periph", g("q"), g("v2"))],
        ),
        PkModel::TwoCptIg => (
            None,
            format!("igd(mat={}, cv2={})", g("mat"), g("cv2")),
            g("v1"),
            vec![("periph", g("q"), g("v2"))],
        ),
        PkModel::ThreeCptIv => (
            None,
            String::new(),
            g("v1"),
            vec![("periph1", g("q2"), g("v2")), ("periph2", g("q3"), g("v3"))],
        ),
        PkModel::ThreeCptOral => (
            Some(g("ka")),
            format!("{} * depot", g("ka")),
            g("v1"),
            vec![("periph1", g("q2"), g("v2")), ("periph2", g("q3"), g("v3"))],
        ),
    };
    let cl = g("cl");

    // ── The input term ──────────────────────────────────────────────────────
    let input_term = match input {
        InputForm::Template => template_input,
        other => {
            // A replacement input feeds `central` itself, so the template must
            // not already deliver the dose somewhere: a depot would keep its
            // own bolus and its `ka` flux, and a `transit()` / `igd()` forcing
            // would be a second input function on one equation. Both are
            // rejected here rather than left to the parser, whose message would
            // blame generated text the user never wrote.
            if depot_ka.is_some() || !template_input.is_empty() {
                return Err(format!(
                    "ode_template {name}: {} absorption feeds `central` directly, so it needs a \
                     template with no absorption of its own — the `*_iv` family. `{name}` \
                     already delivers the dose through {}.",
                    other.label(),
                    if depot_ka.is_some() {
                        "a `depot` compartment"
                    } else {
                        "its own input-rate function"
                    }
                ));
            }
            match other {
                InputForm::ZeroOrder { dur } => format!("zero_order(dur={dur})"),
                InputForm::Weibull { td, beta } => format!("weibull(td={td}, beta={beta})"),
                InputForm::Template => unreachable!("matched above"),
            }
        }
    };

    // ── The elimination term ────────────────────────────────────────────────
    // First-order elimination shares one bracket with the inter-compartmental
    // rate constants, which is how the plain form has always been written; a
    // saturable one cannot, so its distribution terms are emitted separately
    // below. The saturable expressions are Pharmpy's rate × amount in its own
    // parenthesisation, so the two are comparable term by term.
    let elimination_term = match elimination {
        EliminationForm::FirstOrder => {
            let mut terms = format!("{cl}/{vc}");
            for (_, q, _) in &periphs {
                terms.push_str(&format!(" + {q}/{vc}"));
            }
            format!("({terms}) * central")
        }
        EliminationForm::MichaelisMenten { km } => {
            format!("(({cl} * {km} / ({km} + central / {vc})) / {vc}) * central")
        }
        EliminationForm::MixedFoMm { clmm, km } => {
            format!("(({cl} + {clmm} * {km} / ({km} + central / {vc})) / {vc}) * central")
        }
    };

    // ── `central` ───────────────────────────────────────────────────────────
    let mut central = if input_term.is_empty() {
        format!("-{elimination_term}")
    } else {
        format!("{input_term} - {elimination_term}")
    };
    if !matches!(elimination, EliminationForm::FirstOrder) {
        for (_, q, _) in &periphs {
            central.push_str(&format!(" - ({q}/{vc}) * central"));
        }
    }
    for (state, q, vp) in &periphs {
        central.push_str(&format!(" + ({q}/{vp}) * {state}"));
    }

    // ── Assembly ────────────────────────────────────────────────────────────
    let mut states: Vec<String> = Vec::with_capacity(2 + periphs.len());
    let mut odes: Vec<(String, String)> = Vec::with_capacity(2 + periphs.len());
    if let Some(ka) = depot_ka {
        states.push("depot".to_string());
        odes.push(dt("depot", format!("-{ka} * depot")));
    }
    states.push("central".to_string());
    odes.push(dt("central", central));
    for (state, q, vp) in &periphs {
        states.push((*state).to_string());
        odes.push(dt(
            state,
            format!("({q}/{vc}) * central - ({q}/{vp}) * {state}"),
        ));
    }

    Ok(GeneratedDisposition {
        states,
        obs_cmt: "central".to_string(),
        odes,
        obs_scale: vc.to_string(),
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

    /// Every role of a template, bound to its own upper-cased name — so a
    /// snapshot below reads as the equation a user would have written.
    fn roles_of(name: &str) -> Vec<(String, String)> {
        PkModel::from_name(name)
            .unwrap_or_else(|| panic!("{name} is a template"))
            .required_pk_params()
            .iter()
            .map(|(_, role)| (role.to_string(), role.to_uppercase()))
            .collect()
    }

    fn owned(pairs: &[(String, String)]) -> HashMap<String, String> {
        pairs.iter().cloned().collect()
    }

    /// The generated `central` line for `name` under one variant.
    fn central_of(name: &str, input: &InputForm, elimination: &EliminationForm) -> String {
        let g = generate_variant(name, &owned(&roles_of(name)), input, elimination)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        g.odes
            .iter()
            .find(|(state, _)| state == "central")
            .map(|(_, line)| line.clone())
            .unwrap_or_else(|| panic!("{name} generates a central equation"))
    }

    /// The regression guard for #1257's refactor: every one of the ten
    /// templates, transcribed as it was before the disposition was composed
    /// from one topology table rather than written out per model.
    ///
    /// These strings are the *previous* implementation's literal output, so
    /// the test fails on any drift in spacing, parenthesisation or term
    /// order — none of which changes the model, all of which would change
    /// every generated `.ferx` file a search writes and every diff a user
    /// reads. Byte-identity is the claim; a looser assertion would not have
    /// caught the fused-bracket split the variants needed.
    #[test]
    fn plain_form_is_byte_identical_across_every_template() {
        let expected: &[(&str, &[&str])] = &[
            ("one_cpt_iv", &["d/dt(central) = -(CL/V) * central"]),
            (
                "one_cpt_oral",
                &[
                    "d/dt(depot) = -KA * depot",
                    "d/dt(central) = KA * depot - (CL/V) * central",
                ],
            ),
            (
                "one_cpt_transit",
                &["d/dt(central) = transit(n=N, mtt=MTT) - (CL/V) * central"],
            ),
            (
                "one_cpt_ig",
                &["d/dt(central) = igd(mat=MAT, cv2=CV2) - (CL/V) * central"],
            ),
            (
                "two_cpt_iv",
                &[
                    "d/dt(central) = -(CL/V1 + Q/V1) * central + (Q/V2) * periph",
                    "d/dt(periph) = (Q/V1) * central - (Q/V2) * periph",
                ],
            ),
            (
                "two_cpt_oral",
                &[
                    "d/dt(depot) = -KA * depot",
                    "d/dt(central) = KA * depot - (CL/V1 + Q/V1) * central + (Q/V2) * periph",
                    "d/dt(periph) = (Q/V1) * central - (Q/V2) * periph",
                ],
            ),
            (
                "two_cpt_transit",
                &[
                    "d/dt(central) = transit(n=N, mtt=MTT) - (CL/V1 + Q/V1) * central + (Q/V2) * periph",
                    "d/dt(periph) = (Q/V1) * central - (Q/V2) * periph",
                ],
            ),
            (
                "two_cpt_ig",
                &[
                    "d/dt(central) = igd(mat=MAT, cv2=CV2) - (CL/V1 + Q/V1) * central + (Q/V2) * periph",
                    "d/dt(periph) = (Q/V1) * central - (Q/V2) * periph",
                ],
            ),
            (
                "three_cpt_iv",
                &[
                    "d/dt(central) = -(CL/V1 + Q2/V1 + Q3/V1) * central + (Q2/V2) * periph1 + (Q3/V3) * periph2",
                    "d/dt(periph1) = (Q2/V1) * central - (Q2/V2) * periph1",
                    "d/dt(periph2) = (Q3/V1) * central - (Q3/V3) * periph2",
                ],
            ),
            (
                "three_cpt_oral",
                &[
                    "d/dt(depot) = -KA * depot",
                    "d/dt(central) = KA * depot - (CL/V1 + Q2/V1 + Q3/V1) * central + (Q2/V2) * periph1 + (Q3/V3) * periph2",
                    "d/dt(periph1) = (Q2/V1) * central - (Q2/V2) * periph1",
                    "d/dt(periph2) = (Q3/V1) * central - (Q3/V3) * periph2",
                ],
            ),
        ];
        // Every template is covered, so a new one cannot join the family
        // without a snapshot: `PkModel` has exactly these ten.
        assert_eq!(expected.len(), 10, "one snapshot per `PkModel` template");
        for (name, lines) in expected {
            let g =
                generate(name, &owned(&roles_of(name))).unwrap_or_else(|e| panic!("{name}: {e}"));
            let got: Vec<&str> = g.odes.iter().map(|(_, l)| l.as_str()).collect();
            assert_eq!(&got, lines, "{name}");
        }
    }

    /// `generate` is `generate_variant` at its defaults — not a second
    /// transcription that happens to agree today.
    #[test]
    fn generate_is_the_default_variant() {
        for name in [
            "one_cpt_iv",
            "one_cpt_oral",
            "one_cpt_transit",
            "one_cpt_ig",
            "two_cpt_iv",
            "two_cpt_oral",
            "two_cpt_transit",
            "two_cpt_ig",
            "three_cpt_iv",
            "three_cpt_oral",
        ] {
            let params = owned(&roles_of(name));
            assert_eq!(
                generate(name, &params).unwrap(),
                generate_variant(
                    name,
                    &params,
                    &InputForm::Template,
                    &EliminationForm::FirstOrder
                )
                .unwrap(),
                "{name}"
            );
        }
    }

    /// Zero-order and Weibull absorption replace the input term of an `*_iv`
    /// template and leave the disposition alone.
    #[test]
    fn replacement_input_feeds_central() {
        assert_eq!(
            central_of(
                "one_cpt_iv",
                &InputForm::ZeroOrder { dur: "DUR".into() },
                &EliminationForm::FirstOrder,
            ),
            "d/dt(central) = zero_order(dur=DUR) - (CL/V) * central"
        );
        assert_eq!(
            central_of(
                "two_cpt_iv",
                &InputForm::Weibull {
                    td: "TD".into(),
                    beta: "BETA".into(),
                },
                &EliminationForm::FirstOrder,
            ),
            "d/dt(central) = weibull(td=TD, beta=BETA) - (CL/V1 + Q/V1) * central + (Q/V2) * periph"
        );
        // The peripheral equation is the template's own, untouched.
        let g = generate_variant(
            "two_cpt_iv",
            &owned(&roles_of("two_cpt_iv")),
            &InputForm::Weibull {
                td: "TD".into(),
                beta: "BETA".into(),
            },
            &EliminationForm::FirstOrder,
        )
        .unwrap();
        assert_eq!(g.states, vec!["central", "periph"]);
        assert_eq!(
            g.odes[1].1,
            "d/dt(periph) = (Q/V1) * central - (Q/V2) * periph"
        );
    }

    /// A replacement input on a template that already absorbs is refused —
    /// the depot would keep its own bolus, or the equation would carry two
    /// input functions.
    #[test]
    fn replacement_input_needs_a_bolus_template() {
        for name in ["one_cpt_oral", "two_cpt_oral", "three_cpt_oral"] {
            let err = generate_variant(
                name,
                &owned(&roles_of(name)),
                &InputForm::ZeroOrder { dur: "DUR".into() },
                &EliminationForm::FirstOrder,
            )
            .unwrap_err();
            assert!(err.contains("`depot` compartment"), "{name}: {err}");
            assert!(err.contains("zero-order"), "{name}: {err}");
        }
        for name in ["one_cpt_transit", "two_cpt_ig"] {
            let err = generate_variant(
                name,
                &owned(&roles_of(name)),
                &InputForm::Weibull {
                    td: "TD".into(),
                    beta: "BETA".into(),
                },
                &EliminationForm::FirstOrder,
            )
            .unwrap_err();
            assert!(err.contains("input-rate function"), "{name}: {err}");
            assert!(err.contains("Weibull"), "{name}: {err}");
        }
    }

    /// Saturable elimination: the `cl` binding becomes the Michaelis-Menten
    /// clearance, and the distribution terms leave the (now non-linear)
    /// elimination bracket.
    #[test]
    fn saturable_elimination_splits_the_outflow_bracket() {
        assert_eq!(
            central_of(
                "one_cpt_iv",
                &InputForm::Template,
                &EliminationForm::MichaelisMenten { km: "KM".into() },
            ),
            "d/dt(central) = -((CL * KM / (KM + central / V)) / V) * central"
        );
        assert_eq!(
            central_of(
                "two_cpt_iv",
                &InputForm::Template,
                &EliminationForm::MichaelisMenten { km: "KM".into() },
            ),
            "d/dt(central) = -((CL * KM / (KM + central / V1)) / V1) * central \
             - (Q/V1) * central + (Q/V2) * periph"
        );
        assert_eq!(
            central_of(
                "two_cpt_oral",
                &InputForm::Template,
                &EliminationForm::MixedFoMm {
                    clmm: "CLMM".into(),
                    km: "KM".into(),
                },
            ),
            "d/dt(central) = KA * depot \
             - ((CL + CLMM * KM / (KM + central / V1)) / V1) * central \
             - (Q/V1) * central + (Q/V2) * periph"
        );
    }

    /// The two replacements compose: an ODE candidate may change absorption
    /// and elimination in the same move.
    #[test]
    fn input_and_elimination_compose() {
        assert_eq!(
            central_of(
                "one_cpt_iv",
                &InputForm::ZeroOrder { dur: "DUR".into() },
                &EliminationForm::MichaelisMenten { km: "KM".into() },
            ),
            "d/dt(central) = zero_order(dur=DUR) - ((CL * KM / (KM + central / V)) / V) * central"
        );
    }

    /// Saturable elimination on a forcing template keeps the forcing: the
    /// transit chain and Michaelis-Menten elimination are an allowed pair in
    /// Pharmpy's move table, so the composition has to be reachable.
    #[test]
    fn saturable_elimination_keeps_a_transit_forcing() {
        assert_eq!(
            central_of(
                "two_cpt_transit",
                &InputForm::Template,
                &EliminationForm::MichaelisMenten { km: "KM".into() },
            ),
            "d/dt(central) = transit(n=N, mtt=MTT) \
             - ((CL * KM / (KM + central / V1)) / V1) * central \
             - (Q/V1) * central + (Q/V2) * periph"
        );
    }
}
