//! Which `[odes]` models take the exact linear path (#852), and what `ode_method = linear`
//! says when a model cannot.
//!
//! The route is decided in one place, [`OdeSpec::effective_solver_opts`], so these tests ask
//! that function directly: every integration reads it, and a model that resolves to
//! `OdeMethod::Linear` there is propagated by the matrix exponential everywhere.

use crate::ode::OdeMethod;
use crate::parser::model_parser::parse_full_model;

/// A model whose `[odes]` body is `odes`, with optional `[fit_options]` lines.
fn source(odes: &str, fit_options: &str) -> String {
    format!(
        "[parameters]
  theta TVCL(3.0, 0.01, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  theta TVKA(1.1, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
{odes}

[error_model]
  DV ~ proportional(PROP)

[fit_options]
  method = foce
{fit_options}
"
    )
}

/// Depot absorption written as a state: linear, autonomous, no forcing.
const LINEAR: &str = "  d/dt(depot) = -KA * depot
  d/dt(central) = KA * depot - (CL / V) * central";

fn effective_method(odes: &str, fit_options: &str) -> OdeMethod {
    let model = parse_full_model(&source(odes, fit_options))
        .unwrap_or_else(|e| panic!("fixture must parse: {e}"))
        .model;
    model
        .ode_spec
        .as_ref()
        .expect("an [odes] model")
        .effective_solver_opts()
        .method
}

#[test]
fn auto_resolves_to_linear_only_for_a_certified_system() {
    assert_eq!(effective_method(LINEAR, ""), OdeMethod::Linear);
    assert_eq!(
        effective_method(LINEAR, "  ode_method = auto"),
        OdeMethod::Linear
    );
    assert_eq!(
        effective_method(LINEAR, "  ode_method = linear"),
        OdeMethod::Linear
    );
    // Each way out of eligibility keeps `auto` on the integrator.
    let saturable = "  d/dt(depot) = -KA * depot
  d/dt(central) = KA * depot - CL * central / (V + central)";
    assert_eq!(effective_method(saturable, ""), OdeMethod::Auto);
    let reads_tad = "  d/dt(depot) = -KA * depot
  d/dt(central) = KA * depot - (CL / V) * central * (1 + 0 * TAD)";
    assert_eq!(effective_method(reads_tad, ""), OdeMethod::Auto);
    let forcing = "  d/dt(depot) = 0 * depot
  d/dt(central) = first_order(ka=KA) - (CL / V) * central";
    assert_eq!(effective_method(forcing, ""), OdeMethod::Auto);
}

#[test]
fn a_named_integrator_opts_a_linear_model_out() {
    for (token, method) in [
        ("rk45", OdeMethod::Rk45),
        ("rodas5p", OdeMethod::Rodas5P),
        ("vern7", OdeMethod::Vern7),
    ] {
        assert_eq!(
            effective_method(LINEAR, &format!("  ode_method = {token}")),
            method,
            "`ode_method = {token}` must reach the integrator on a linear model"
        );
    }
}

/// The parse error for `ode_method = linear` on `odes` (optionally with an extra block).
fn linear_error(odes: &str, extra_block: &str) -> String {
    let src = format!("{}{extra_block}", source(odes, "  ode_method = linear"));
    match parse_full_model(&src) {
        Ok(_) => panic!("`ode_method = linear` must be refused on:\n{odes}\n{extra_block}"),
        Err(e) => e,
    }
}

/// One case per ineligibility reason. Each asserts the clause that names its own cause, so
/// deleting any one sentence of the message reddens exactly the case that needs it, and every
/// case asserts the shared lead and remedy.
#[test]
fn ode_method_linear_names_why_a_model_cannot_take_the_exact_path() {
    let remedy = "Use `ode_method = auto` (the default) to integrate it.";
    let lead =
        "fit option `ode_method = linear`: this [odes] model cannot be propagated exactly — ";

    let e = linear_error(
        "  d/dt(depot) = -KA * depot
  d/dt(central) = KA * depot - CL * central / (V + central)",
        "",
    );
    assert!(e.contains("not linear in the states"), "{e}");
    assert!(e.contains(remedy), "{e}");
    assert!(e.starts_with(lead), "{e}");

    let e = linear_error(
        "  d/dt(depot) = -KA * depot
  d/dt(central) = KA * depot - (CL / V) * central * (1 + 0 * TIME)",
        "",
    );
    assert!(e.contains("reads model time"), "{e}");
    assert!(e.contains(remedy), "{e}");
    assert!(e.starts_with(lead), "{e}");

    let e = linear_error(
        "  d/dt(depot) = 0 * depot
  d/dt(central) = first_order(ka=KA) - (CL / V) * central",
        "",
    );
    assert!(e.contains("absorption input function"), "{e}");
    assert!(e.contains("depot state instead"), "{e}");
    assert!(e.contains(remedy), "{e}");
    assert!(e.starts_with(lead), "{e}");

    let e = linear_error(LINEAR, "\n[diffusion]\n  central ~ 0.01\n");
    assert!(e.contains("stochastic ([diffusion]) model"), "{e}");
    assert!(e.contains(remedy), "{e}");
    assert!(e.starts_with(lead), "{e}");
}

#[test]
fn ode_method_linear_is_accepted_on_a_certified_model() {
    parse_full_model(&source(LINEAR, "  ode_method = linear"))
        .unwrap_or_else(|e| panic!("a linear model must accept `ode_method = linear`: {e}"));
}

#[test]
fn ode_method_linear_is_refused_on_an_analytical_model() {
    // The assertion has nothing to hold on a model without [odes]; accepting it silently would
    // let a deleted or misspelled [odes] block pass as "certified exact".
    let analytical = source(LINEAR, "  ode_method = linear")
        .replace(
            "ode(obs_cmt=central, states=[depot, central])",
            "pk one_cpt_oral(cl=CL, v=V, ka=KA)",
        )
        .replace(&format!("[odes]\n{LINEAR}\n"), "");
    assert!(
        !analytical.contains("[odes]"),
        "fixture must have no [odes] block"
    );
    let e = match parse_full_model(&analytical) {
        Ok(_) => panic!("`ode_method = linear` must be refused on an analytical model"),
        Err(e) => e,
    };
    assert!(
        e.contains(
            "fit option `ode_method = linear` applies to an [odes] model, and this model has none,"
        ),
        "{e}"
    );
    assert!(
        e.contains("so there is nothing for the option to act on; remove it."),
        "{e}"
    );
    // The same model without the option parses: the refusal is the option's, not the fixture's.
    parse_full_model(&analytical.replace("  ode_method = linear", ""))
        .unwrap_or_else(|e| panic!("analytical twin must parse: {e}"));
}
