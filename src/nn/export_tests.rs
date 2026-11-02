//! Tests for the DCM → plain population-PK export (#1331).
//!
//! Two halves, split by what they need:
//!
//! - **Emission** (every build). The committed fixture `tests/data/dcm_export/`
//!   — a model plus a θ vector — exported and compared against the committed
//!   `expected_analytic.ferx`, reparsed in the build without `nn` (the build
//!   ferx-r ships, where `[covariate_nn]` itself does not parse), and checked for
//!   the two properties the emission alone can lose: a function the parser does
//!   not know, and a weight printed short of round-trip precision.
//! - **Parity** (`nn` only). The exported model evaluated against the network it
//!   came from, bit for bit: the same function written two ways, so the target
//!   is equality, not a tolerance.

use super::*;
use crate::parser::model_parser::{parse_model_string, SUPPORTED_UNARY_FNS};

const FIXTURE_MODEL: &str = include_str!("../../tests/data/dcm_export/model.ferx");
const FIXTURE_THETA: &str = include_str!("../../tests/data/dcm_export/theta.csv");
const FIXTURE_EXPECTED: &str = include_str!("../../tests/data/dcm_export/expected_analytic.ferx");
const FIXTURE_FITTED: &str = include_str!("../../tests/data/dcm_export/expected_fitted.ferx");

fn fixture_theta() -> (Vec<String>, Vec<f64>) {
    FIXTURE_THETA
        .lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let (n, v) = l.split_once(',').expect("name,value");
            (n.to_string(), v.parse::<f64>().expect("a number"))
        })
        .unzip()
}

fn export(
    text: &str,
    names: &[String],
    theta: &[f64],
    form: DcmWeightForm,
) -> Result<String, String> {
    let mut opts = DcmExportOptions::default();
    opts.weight_form = form;
    export_dcm_as_analytic(text, names, theta, &opts)
}

fn export_fixture(form: DcmWeightForm) -> String {
    let (names, theta) = fixture_theta();
    export(FIXTURE_MODEL, &names, &theta, form).expect("fixture exports")
}

/// The lines the export generated: every output line that is not an input line.
fn generated_lines(exported: &str, original: &str) -> Vec<String> {
    let orig: HashSet<&str> = original.lines().collect();
    exported
        .lines()
        .filter(|l| !orig.contains(l))
        .map(str::to_string)
        .collect()
}

// ── Emission (every build) ──────────────────────────────────────────────────

/// The exported text of the committed fixture is the committed text.
///
/// The whole emission is pinned in the build without `nn`: a sign flip, a
/// transposed `W` index, a dropped layer, a changed activation spelling or a
/// truncated weight each change a line here. The parity tests below say the
/// text is *right*; this says it has not moved. To regenerate after a deliberate
/// change, write `export_fixture(DcmWeightForm::Literal)` over the file and
/// review the diff line by line.
#[test]
fn fixture_export_matches_the_committed_text() {
    let got = export_fixture(DcmWeightForm::Literal);
    if got != FIXTURE_EXPECTED {
        let first = got
            .lines()
            .zip(FIXTURE_EXPECTED.lines())
            .position(|(a, b)| a != b);
        panic!(
            "export of tests/data/dcm_export/model.ferx drifted from expected_analytic.ferx \
             (first differing line: {first:?})\n--- got ---\n{got}"
        );
    }
}

/// The export parses in the current build — in particular in the default build,
/// which has no `nn` and refuses the `[covariate_nn]` input outright — and
/// carries no trace of the network block or its references.
#[test]
fn exported_model_parses_without_a_network_block() {
    for form in [DcmWeightForm::Literal, DcmWeightForm::FixedTheta] {
        let text = export_fixture(form);
        parse_model_string(&text)
            .unwrap_or_else(|e| panic!("{form:?} export does not parse: {e}\n{text}"));
        for line in text.lines() {
            let code = strip_line_comment(line);
            assert!(
                !code.contains("[covariate_nn"),
                "{form:?}: block left behind: {line}"
            );
            assert!(
                !code.contains("TYPICAL_PK.") && !code.contains("TYPICAL_PK ."),
                "{form:?}: a network reference survived the rewrite: {line}"
            );
        }
    }
    #[cfg(not(feature = "nn"))]
    {
        let err = parse_model_string(FIXTURE_MODEL).expect_err("no nn: the DCM must not parse");
        assert!(err.contains("--features nn"), "{err}");
    }
}

/// Every function the export calls is one the parser knows.
///
/// Before #1332 an unknown name parsed as the identity, so `tanh(z)` emitted
/// ahead of the builtin would have exported a model that fits, converges and
/// computes `z`. The parser now refuses an unknown name, and this pins the
/// export to the whitelist the parser checks against, so a new activation
/// cannot be spelled with a name that does not exist.
#[test]
fn every_emitted_function_is_a_parser_builtin() {
    let (names, theta) = fixture_theta();
    let mut texts = vec![export_fixture(DcmWeightForm::Literal)];
    // Every activation, as hidden and as output, so each spelling is checked.
    for act in ALL_ACTIVATIONS {
        let model = FIXTURE_MODEL
            .replace(
                "activation = tanh",
                &format!("activation = {}", act.as_str()),
            )
            .replace(
                "output     = softplus",
                &format!("output     = {}", act.as_str()),
            );
        texts.push(export(&model, &names, &theta, DcmWeightForm::Literal).unwrap());
    }
    let mut seen = HashSet::new();
    for text in &texts {
        for line in generated_lines(text, FIXTURE_MODEL) {
            let code = strip_line_comment(&line);
            let bytes = code.as_bytes();
            for word in identifiers(code) {
                let end = word.as_ptr() as usize - code.as_ptr() as usize + word.len();
                let rest = code[end..].trim_start();
                if rest.starts_with('(') && bytes.len() > end {
                    assert!(
                        word == "if" || SUPPORTED_UNARY_FNS.contains(&word),
                        "the export calls `{word}(...)`, which the parser does not know:\n{line}"
                    );
                    seen.insert(word.to_string());
                }
            }
        }
    }
    // The check above must have seen the calls it guards, or it passes vacuously.
    for f in ["tanh", "softplus", "inv_logit", "exp", "if"] {
        assert!(seen.contains(f), "no `{f}(` was emitted; saw {seen:?}");
    }
}

/// Each weight appears in the text as a literal that reads back as the same
/// `f64`. A fixed-precision format (`{:.6}`) would print a different model whose
/// difference is invisible in the output.
#[test]
fn every_weight_is_printed_at_round_trip_precision() {
    let (names, theta) = fixture_theta();
    let text = export_fixture(DcmWeightForm::Literal);
    let mut literals: Vec<u64> = Vec::new();
    for line in generated_lines(&text, FIXTURE_MODEL) {
        let code = strip_line_comment(&line);
        for tok in code.split(|c: char| !(c.is_ascii_alphanumeric() || c == '.' || c == '_')) {
            if tok.starts_with(|c: char| c.is_ascii_digit()) {
                if let Ok(v) = tok.parse::<f64>() {
                    literals.push(v.to_bits());
                }
            }
        }
    }
    let mut n_weights = 0;
    for (n, v) in names.iter().zip(&theta) {
        if n.starts_with("W_") || n.starts_with("B_") {
            n_weights += 1;
            assert!(
                literals.contains(&v.abs().to_bits()),
                "weight {n} = {v:?} is not in the text at full precision"
            );
        }
    }
    assert_eq!(
        n_weights, 23,
        "the fixture's 2-3-2-2 network has 23 weights"
    );
}

/// `FixedTheta` declares every weight as a FIXed θ holding its exact value, and
/// the equations reference them by name rather than inlining them.
#[test]
fn fixed_theta_form_declares_each_weight_once_at_its_value() {
    let (names, theta) = fixture_theta();
    let text = export_fixture(DcmWeightForm::FixedTheta);
    let model = parse_model_string(&text).expect("parses");
    for (n, v) in names.iter().zip(&theta) {
        if !(n.starts_with("W_") || n.starts_with("B_")) {
            continue;
        }
        let i = model
            .default_params
            .theta_names
            .iter()
            .position(|t| t == n)
            .unwrap_or_else(|| panic!("{n} not declared"));
        assert_eq!(model.default_params.theta[i].to_bits(), v.to_bits(), "{n}");
        assert!(model.default_params.theta_fixed[i], "{n} must be FIX");
        assert_eq!(text.matches(&format!("theta {n}(")).count(), 1, "{n}");
    }
    let generated = generated_lines(&text, FIXTURE_MODEL).join("\n");
    assert!(
        generated.contains("W_TYPICAL_PK_3_2_1 * H2_TYPICAL_PK_1"),
        "{generated}"
    );
}

#[test]
fn a_model_without_a_network_is_an_error() {
    let plain = FIXTURE_MODEL
        .lines()
        .filter(|l| !l.contains("covariate_nn") && !l.contains(" = ["))
        .collect::<Vec<_>>()
        .join("\n");
    let err = export(&plain, &[], &[], DcmWeightForm::Literal).unwrap_err();
    assert!(err.contains("no [covariate_nn] block"), "{err}");
}

#[test]
fn a_weight_missing_from_the_fit_is_named() {
    let (mut names, theta) = fixture_theta();
    let i = names
        .iter()
        .position(|n| n == "W_TYPICAL_PK_2_1_3")
        .unwrap();
    names[i] = "SOMETHING_ELSE".into();
    let err = export(FIXTURE_MODEL, &names, &theta, DcmWeightForm::Literal).unwrap_err();
    assert!(
        err.contains("W_TYPICAL_PK_2_1_3") && err.contains("not among"),
        "{err}"
    );
}

#[test]
fn a_non_finite_weight_is_refused() {
    let (names, mut theta) = fixture_theta();
    let i = names.iter().position(|n| n == "B_TYPICAL_PK_3_2").unwrap();
    theta[i] = f64::NAN;
    let err = export(FIXTURE_MODEL, &names, &theta, DcmWeightForm::Literal).unwrap_err();
    assert!(
        err.contains("B_TYPICAL_PK_3_2") && err.contains("non-finite"),
        "{err}"
    );
}

#[test]
fn mismatched_theta_lengths_are_refused() {
    let (names, theta) = fixture_theta();
    let err = export(FIXTURE_MODEL, &names, &theta[1..], DcmWeightForm::Literal).unwrap_err();
    assert!(err.contains("theta_names has"), "{err}");
}

/// A generated name the model already uses is an error, not a silent shadowing:
/// the reassignment would change whatever the model's own variable fed.
#[test]
fn a_generated_name_the_model_already_uses_is_refused() {
    let (names, theta) = fixture_theta();
    let model = FIXTURE_MODEL.replace(
        "  V  = TVV * exp(ETA_V)",
        "  H1_TYPICAL_PK_2 = 1.0\n  V  = TVV * exp(ETA_V) * H1_TYPICAL_PK_2",
    );
    let err = export(&model, &names, &theta, DcmWeightForm::Literal).unwrap_err();
    assert!(
        err.contains("`H1_TYPICAL_PK_2`") && err.contains("already uses"),
        "{err}"
    );
    // Case-folded: a differently cased spelling collides too.
    let model = FIXTURE_MODEL.replace("TVV", "typical_pk_cl");
    let err = export(&model, &names, &theta, DcmWeightForm::Literal).unwrap_err();
    assert!(err.contains("`TYPICAL_PK_CL`"), "{err}");
}

#[test]
fn a_reference_to_an_undeclared_output_is_refused() {
    let (names, theta) = fixture_theta();
    let model = FIXTURE_MODEL.replace("TYPICAL_PK . KA", "TYPICAL_PK.V2");
    let err = export(&model, &names, &theta, DcmWeightForm::Literal).unwrap_err();
    assert!(
        err.contains("TYPICAL_PK.V2") && err.contains("Known outputs: CL, KA"),
        "{err}"
    );
}

#[test]
fn a_time_input_is_refused() {
    // Same input count, so the fixture's weight names still match.
    let (names, theta) = fixture_theta();
    let model = FIXTURE_MODEL.replace("inputs     = [WT, CRCL]", "inputs     = [WT, TIME]");
    let err = export(&model, &names, &theta, DcmWeightForm::Literal).unwrap_err();
    assert!(err.contains("TIME") && err.contains("#1942"), "{err}");
}

/// A network input written into an equation must come back as the data column.
/// Each name below parses in the DCM, where the network looks the column up by
/// name, but would read something else once it is an expression: a built-in, a
/// θ or a variable — or not parse at all.
#[test]
fn an_input_that_would_not_read_the_data_column_is_refused() {
    let (names, theta) = fixture_theta();
    let with_inputs = |inputs: &str| {
        FIXTURE_MODEL.replace(
            "inputs     = [WT, CRCL]",
            &format!("inputs     = [{inputs}]"),
        )
    };
    // A built-in the parser resolves before any data column.
    let err = export(
        &with_inputs("WT, MIXNUM"),
        &names,
        &theta,
        DcmWeightForm::Literal,
    )
    .unwrap_err();
    assert!(err.contains("`MIXNUM`") && err.contains("mixture"), "{err}");
    // Not an identifier at all.
    let err = export(
        &with_inputs("WT, CRCL.MLMIN"),
        &names,
        &theta,
        DcmWeightForm::Literal,
    )
    .unwrap_err();
    assert!(
        err.contains("`CRCL.MLMIN`") && err.contains("not an identifier"),
        "{err}"
    );
    // A θ of the same name: the equation would read the θ.
    let model = FIXTURE_MODEL.replace(
        "  theta TVV(30, 1, 100)",
        "  theta TVV(30, 1, 100)\n  theta CRCL(1, 0.1, 10)",
    );
    let err = export(&model, &names, &theta, DcmWeightForm::Literal).unwrap_err();
    assert!(
        err.contains("`CRCL`") && err.contains("not read as a data covariate"),
        "{err}"
    );
    // The control: the fixture's own inputs are read as columns, so the check
    // above is not refusing everything.
    let model = parse_model_string(&export_fixture(DcmWeightForm::Literal)).unwrap();
    for input in ["WT", "CRCL"] {
        assert!(
            model.referenced_covariates.iter().any(|c| c == input),
            "{input}"
        );
    }
}

/// A variable the model assigns under an input's name is refused too: the
/// equation would read the variable, not the column.
#[test]
fn an_input_the_model_assigns_is_refused() {
    let (names, theta) = fixture_theta();
    let model = FIXTURE_MODEL.replace(
        "  V  = TVV * exp(ETA_V)",
        "  WT = 70\n  V  = TVV * exp(ETA_V)",
    );
    let err = export(&model, &names, &theta, DcmWeightForm::Literal).unwrap_err();
    assert!(err.contains("WT"), "{err}");
}

/// A `NAME.OUTPUT` read outside `[individual_parameters]` — a kappa weight,
/// which the parser resolves with the networks in scope but no locals — would
/// become a bare output name there, read as a data column. The control reads the
/// same output only inside `[individual_parameters]` and exports.
#[test]
fn an_output_read_outside_individual_parameters_is_refused() {
    let (names, theta) = fixture_theta();
    let with_kappa = |weight: &str| {
        FIXTURE_MODEL
            .replace(
                "  omega ETA_KA ~ 0.2",
                &format!("  omega ETA_KA ~ 0.2\n  kappa KAPPA_CL ~ 0.09 weight = {weight}"),
            )
            .replace("exp(ETA_CL)", "exp(ETA_CL + KAPPA_CL)")
    };
    let control = export(&with_kappa("NARM"), &names, &theta, DcmWeightForm::Literal);
    assert!(control.is_ok(), "{control:?}");
    let err = export(
        &with_kappa("TYPICAL_PK.CL"),
        &names,
        &theta,
        DcmWeightForm::Literal,
    )
    .unwrap_err();
    assert!(
        err.contains("`TYPICAL_PK.CL` is read outside [individual_parameters]"),
        "{err}"
    );
    assert!(err.contains("`TYPICAL_PK_CL` as a data column"), "{err}");
}

/// Without a `[parameters]` block there is nowhere to declare the `FixedTheta`
/// weights — and no valid model either, since the parser requires the block. The
/// refusal says so instead of blaming the weight references.
#[test]
fn a_model_without_a_parameters_block_is_refused_for_the_missing_block() {
    let (names, theta) = fixture_theta();
    let model = FIXTURE_MODEL.replace("[parameters]\n", "");
    for form in [DcmWeightForm::Literal, DcmWeightForm::FixedTheta] {
        let err = export(&model, &names, &theta, form).unwrap_err();
        assert!(
            err.contains("Missing [parameters] block"),
            "{form:?}: {err}"
        );
    }
}

/// The network's units must not become individual parameters: an ODE model
/// gives each top-level `[individual_parameters]` assignment a slot in a layout
/// of fewer than 128, and two 64-unit hidden layers are 130 units. The export of
/// such a network parses, and the units are absent from the parameter list.
#[test]
fn a_wide_network_on_an_ode_model_exports_to_a_model_that_parses() {
    let model = FIXTURE_MODEL
        .replace("layers     = [3, 2]", "layers     = [64, 64]")
        .replace(
            "  pk one_cpt_oral(cl=CL, v=V, ka=KA)",
            "  ode(obs_cmt=central, states=[depot, central])\n\n[odes]\n  \
             d/dt(depot)   = -KA * depot\n  d/dt(central) = KA * depot / V - CL/V * central",
        );
    let names = parse_nn_block_fields(
        "TYPICAL_PK",
        &[
            "inputs = [WT, CRCL]".into(),
            "outputs = [CL, KA]".into(),
            "layers = [64, 64]".into(),
            "activation = tanh".into(),
        ],
    )
    .unwrap()
    .weight_theta_names();
    assert_eq!(names.len(), 2 * 64 + 64 + 64 * 64 + 64 + 64 * 2 + 2);
    let theta: Vec<f64> = (0..names.len())
        .map(|k| ((k as f64) * 0.37).sin())
        .collect();
    let text = export(&model, &names, &theta, DcmWeightForm::Literal).expect("exports");
    let plain = parse_model_string(&text).expect("the export parses");
    assert!(plain.ode_spec.is_some(), "the fixture must be an ODE model");
    for p in &plain.indiv_param_names {
        assert!(
            !p.starts_with("H1_") && !p.starts_with("H2_") && !p.starts_with("Z_"),
            "network unit `{p}` became an individual parameter"
        );
    }
    assert!(plain.indiv_param_names.iter().any(|p| p == "CL"));
}

/// A model may read a weight θ directly; the parser allows it (#1016). The θ
/// leaves with the block, so `Literal` inlines the weight's value at the
/// reference and `FixedTheta` declares the θ again and keeps the name. Left as
/// a bare name, the reference would parse as a data column — a different model.
#[test]
fn a_direct_read_of_a_weight_theta_survives_the_export() {
    let (names, theta) = fixture_theta();
    let model = FIXTURE_MODEL.replace(
        "  V  = TVV * exp(ETA_V)",
        "  V  = TVV * exp(ETA_V + 0.01 * B_TYPICAL_PK_3_1 - W_TYPICAL_PK_1_1_1)",
    );
    let b = theta[names.iter().position(|n| n == "B_TYPICAL_PK_3_1").unwrap()];
    let w = theta[names
        .iter()
        .position(|n| n == "W_TYPICAL_PK_1_1_1")
        .unwrap()];
    assert!(
        b < 0.0 && w > 0.0,
        "one negative and one positive reference"
    );

    let literal = export(&model, &names, &theta, DcmWeightForm::Literal).unwrap();
    assert!(
        literal.contains(&format!(
            "  V  = TVV * exp(ETA_V + 0.01 * (-{:?}) - {w:?})",
            b.abs()
        )),
        "{literal}"
    );
    let plain = parse_model_string(&literal).unwrap();
    assert!(
        plain
            .referenced_covariates
            .iter()
            .all(|c| !c.starts_with("W_") && !c.starts_with("B_")),
        "{:?}",
        plain.referenced_covariates
    );

    let fixed = export(&model, &names, &theta, DcmWeightForm::FixedTheta).unwrap();
    assert!(fixed.contains("exp(ETA_V + 0.01 * B_TYPICAL_PK_3_1 - W_TYPICAL_PK_1_1_1)"));
    let plain = parse_model_string(&fixed).unwrap();
    assert!(plain
        .referenced_covariates
        .iter()
        .all(|c| !c.starts_with("B_")));
}

/// The validation parse does not read a `[priors] from_fit` file: the path is
/// relative to the model, which a text transform does not know, so resolving it
/// against the working directory would refuse a valid model. The block itself is
/// kept in the export.
#[test]
fn a_from_fit_prior_is_kept_and_not_read() {
    let (names, theta) = fixture_theta();
    let model =
        format!("{FIXTURE_MODEL}\n[priors]\n  from_fit = \"no/such/dir/parent-fit.json\"\n");
    let text = export(&model, &names, &theta, DcmWeightForm::Literal)
        .expect("a prior file that does not resolve here does not block the export");
    assert!(
        text.contains("[priors]\n  from_fit = \"no/such/dir/parent-fit.json\"\n"),
        "{text}"
    );
    // Control: the parser itself does read the file, so the strip is what lets
    // the export through.
    assert!(parse_model_string(&text).is_err());
}

/// Paths and other quoted strings survive byte for byte, even when they spell
/// a network reference or a weight θ name. `[data]` / `[priors]` / `[fit_options]`
/// are copied whole; a quoted string in any other block is skipped by the
/// rewrite too, so the guarantee does not rest on the block list alone.
#[test]
fn quoted_paths_are_never_rewritten() {
    let (names, theta) = fixture_theta();
    let tail = "\n[data]\n  path = \"TYPICAL_PK.CL.csv\"\n\
                \n[priors]\n  from_fit = \"fits/W_TYPICAL_PK_1_1_1-B_TYPICAL_PK_3_2.json\"\n";
    let model = format!("{FIXTURE_MODEL}{tail}");
    for form in [DcmWeightForm::Literal, DcmWeightForm::FixedTheta] {
        let text = export(&model, &names, &theta, form).unwrap();
        assert!(text.ends_with(tail), "{form:?}: paths changed:\n{text}");
    }
    // Unquoted, only the block rule protects it: `[data]` is copied whole.
    let tail = "\n[data]\n  path = TYPICAL_PK.CL.csv\n";
    let text = export(
        &format!("{FIXTURE_MODEL}{tail}"),
        &names,
        &theta,
        DcmWeightForm::Literal,
    )
    .unwrap();
    assert!(text.ends_with(tail), "unquoted path changed:\n{text}");
    // A quoted span inside an expression block is skipped as well.
    let mut weight_literals = HashMap::new();
    weight_literals.insert("W_TYPICAL_PK_1_1_1".to_string(), 0.5);
    let nets: Vec<(NnBlockSpec, Vec<f64>)> = Vec::new();
    let names_table = NameTable {
        outputs: BTreeMap::new(),
    };
    let line = r#"  X = W_TYPICAL_PK_1_1_1 * 2   # "W_TYPICAL_PK_1_1_1""#;
    let got = rewrite_refs(line, &nets, &names_table, &weight_literals).unwrap();
    assert_eq!(got, r#"  X = 0.5 * 2   # "W_TYPICAL_PK_1_1_1""#);
    let line = r#"  label = "W_TYPICAL_PK_1_1_1 'x'" W_TYPICAL_PK_1_1_1"#;
    let got = rewrite_refs(line, &nets, &names_table, &weight_literals).unwrap();
    assert_eq!(got, r#"  label = "W_TYPICAL_PK_1_1_1 'x'" 0.5"#);
    assert_eq!(
        identifiers(r#"a "b c" d 'e'"#).collect::<Vec<_>>(),
        vec!["a", "d"]
    );
}

/// The fixture's fit: its θ, and Ω / Σ estimates that differ from every value the
/// model file declares, so an export that kept a starting value shows.
fn fixture_fit(model_text: Option<&str>) -> crate::types::FitResult {
    let (names, theta) = fixture_theta();
    crate::types::FitResult {
        theta,
        theta_names: names,
        eta_names: ["ETA_CL", "ETA_V", "ETA_KA"].map(String::from).to_vec(),
        omega: nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![
            0.0912, 0.0437, 0.3125,
        ])),
        sigma: vec![0.1234],
        sigma_names: vec!["PROP_ERR".into()],
        model_text: model_text.map(str::to_string),
        // Finite, so the fit round-trips through JSON (a `NaN` is written `null`).
        shrinkage_eps: 0.0,
        iwres_lag1_r: 0.0,
        dw_statistic: 2.0,
        ..crate::types::test_helpers::empty_fit_result()
    }
}

/// `export_dcm_from_fit` exports only when the fit was made with the networks
/// the model declares. The weight names carry the block name and shape alone, so
/// each change below keeps every weight name the export looks up — and without
/// the check would silently export the fit's weights under a different network.
#[test]
fn a_fit_made_with_a_different_network_is_refused() {
    let fit = fixture_fit(Some(FIXTURE_MODEL));
    // Control: the model the fit was made with exports — under `network_only`
    // identically to the name-lookup entry point.
    let mut network_only = DcmExportOptions::default();
    network_only.network_only = true;
    let got = export_dcm_from_fit(FIXTURE_MODEL, &fit, &network_only).unwrap();
    assert_eq!(got, FIXTURE_EXPECTED);
    // Comments, layout and `init` do not define the network.
    let cosmetic = FIXTURE_MODEL
        .replace(
            "inputs     = [WT, CRCL]   # both normalised",
            "inputs = [WT,CRCL]",
        )
        .replace(
            "  output     = softplus",
            "  output     = softplus\n  init       = [2.0, 1.0]",
        );
    export_dcm_from_fit(&cosmetic, &fit, &DcmExportOptions::default()).unwrap();

    for (what, from, to) in [
        (
            "inputs",
            "inputs     = [WT, CRCL]",
            "inputs     = [WT, AGE]",
        ),
        (
            "inputs",
            "inputs     = [WT, CRCL]",
            "inputs     = [CRCL, WT]",
        ),
        ("outputs", "outputs    = [CL, KA]", "outputs    = [CL, V]"),
        ("activation", "activation = tanh", "activation = sigmoid"),
        ("output", "output     = softplus", "output     = exp"),
        (
            "center",
            "center     = [70, -5]",
            "center     = [70, -5.000000001]",
        ),
        ("scale", "scale      = [12, 23.5]", "scale      = [12, 23]"),
        ("center", "  center     = [70, -5]\n", ""),
    ] {
        let model = FIXTURE_MODEL.replace(from, to);
        assert_ne!(model, FIXTURE_MODEL);
        let err = export_dcm_from_fit(&model, &fit, &DcmExportOptions::default()).unwrap_err();
        assert!(
            err.contains(&format!("{what}:"))
                && err.contains("not the network the fit was made with"),
            "{from} -> {to}: {err}"
        );
        // The name lookup alone does not see it: every weight is still found.
        let (names, theta) = fixture_theta();
        let _ = export(&model, &names, &theta, DcmWeightForm::Literal);
    }
    let renamed = FIXTURE_MODEL.replace("covariate_nn TYPICAL_PK", "covariate_nn OTHER");
    let err = export_dcm_from_fit(&renamed, &fit, &DcmExportOptions::default()).unwrap_err();
    assert!(
        err.contains("[OTHER]") && err.contains("[TYPICAL_PK]"),
        "{err}"
    );
    // A fit without model text cannot be checked.
    let err = export_dcm_from_fit(
        FIXTURE_MODEL,
        &fixture_fit(None),
        &DcmExportOptions::default(),
    )
    .unwrap_err();
    assert!(err.contains("no model text"), "{err}");
}

/// The committed `fit.json` the CLI test reads is this fixture's fit: the
/// fixture's θ and model text. Regenerate it by setting
/// `FERX_WRITE_DCM_FIT_JSON=1` (in an `nn` build, so the file carries the
/// `neural_networks` field both builds can read).
#[test]
fn the_committed_fit_json_is_the_fixture_fit() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/dcm_export/fit.json"
    );
    let fit = fixture_fit(Some(FIXTURE_MODEL));
    if std::env::var_os("FERX_WRITE_DCM_FIT_JSON").is_some() {
        std::fs::write(path, serde_json::to_string(&fit).unwrap()).unwrap();
    }
    let read: crate::types::FitResult =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(read.theta_names, fit.theta_names);
    assert_eq!(
        read.theta.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        fit.theta.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(read.model_text, fit.model_text);
    assert_eq!(read.omega, fit.omega);
    assert_eq!(read.sigma, fit.sigma);
}

/// The committed fitted export: the fixture at its fit, every parameter at the
/// fit's estimate. Regenerate with `FERX_WRITE_DCM_FITTED=1`.
#[test]
fn fitted_export_matches_the_committed_text() {
    let fit = fixture_fit(Some(FIXTURE_MODEL));
    let got = export_dcm_from_fit(FIXTURE_MODEL, &fit, &DcmExportOptions::default()).unwrap();
    if std::env::var_os("FERX_WRITE_DCM_FITTED").is_some() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/dcm_export/expected_fitted.ferx"
        );
        std::fs::write(path, &got).unwrap();
    }
    assert_eq!(got, FIXTURE_FITTED);
}

/// Every estimate lands in the export, bit for bit on the scale the fit reports
/// it on: a θ, a variance-declared Ω and an `(sd)`-declared Σ.
#[test]
fn the_fitted_export_declares_every_estimate_of_the_fit() {
    let fit = fixture_fit(Some(FIXTURE_MODEL));
    let text = export_dcm_from_fit(FIXTURE_MODEL, &fit, &DcmExportOptions::default()).unwrap();
    let p = parse_model_string(&text).unwrap().default_params;
    let i = p.theta_names.iter().position(|n| n == "TVV").unwrap();
    assert_eq!(p.theta[i].to_bits(), 31.5f64.to_bits());
    assert_eq!(p.omega.eta_names, fit.eta_names);
    for (got, want) in p.omega.matrix.iter().zip(fit.omega.iter()) {
        assert_eq!(got.to_bits(), want.to_bits());
    }
    assert_eq!(p.sigma.values[0].to_bits(), 0.1234f64.to_bits());
    assert!(text.contains("theta TVV(31.5, 1, 100)"), "{text}");
    assert!(text.contains("sigma PROP_ERR ~ 0.1234 (sd)"), "{text}");
    assert!(text.contains("is the fit's estimate"), "{text}");
}

/// A declaration keeps its scale — the `(sd)` tag and an inline prior on it
/// still mean what they did — and a block declaration over several lines, a
/// correlated `block_sigma` and a kappa are written as well. Values that cross a
/// scale read back within the rounding of the conversion.
#[test]
fn declarations_keep_their_scale_and_blocks_are_written() {
    let model = FIXTURE_MODEL
        .replace(
            "  omega ETA_CL ~ 0.1\n  omega ETA_V  ~ 0.05\n",
            "  block_omega (ETA_CL, ETA_V) = [0.1,\n     0.01, 0.05]   # CL-V\n",
        )
        .replace(
            "  omega ETA_KA ~ 0.2",
            "  omega ETA_KA ~ 0.4 (sd) prior(0.4, rse = 30%)\n  kappa KAPPA_CL ~ 0.09 weight = NARM",
        )
        .replace(
            "  sigma PROP_ERR ~ 0.1 (sd)",
            "  block_sigma (PROP_ERR, ADD_ERR) = [0.01, 0.001, 0.04]",
        )
        .replace("exp(ETA_CL)", "exp(ETA_CL + KAPPA_CL)")
        .replace(
            "DV ~ proportional(PROP_ERR)",
            "DV ~ combined(PROP_ERR, ADD_ERR)",
        );
    let mut fit = fixture_fit(Some(&model));
    fit.omega[(0, 1)] = 0.0123;
    fit.omega[(1, 0)] = 0.0123;
    fit.omega_iov = Some(nalgebra::DMatrix::from_element(1, 1, 0.0456));
    fit.kappa_names = vec!["KAPPA_CL".into()];
    fit.sigma = vec![0.1234, 0.25];
    fit.sigma_names = vec!["PROP_ERR".into(), "ADD_ERR".into()];
    fit.residual_correlations = vec![crate::types::ResidualCorrelation {
        sigma_i: 0,
        sigma_j: 1,
        rho: 0.3,
    }];
    let text = export_dcm_from_fit(&model, &fit, &DcmExportOptions::default()).unwrap();
    assert!(
        text.contains("block_omega (ETA_CL, ETA_V) = [0.0912, 0.0123, 0.0437]"),
        "{text}"
    );
    assert!(
        text.contains("omega ETA_KA ~ 0.5590169943749475 (sd) prior(0.4, rse = 30%)"),
        "{text}"
    );
    assert!(
        text.contains("kappa KAPPA_CL ~ 0.0456 weight = NARM"),
        "{text}"
    );
    let p = parse_model_string(&text).unwrap().default_params;
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-15 * b.abs();
    assert!(close(p.omega.matrix[(2, 2)], 0.3125), "{}", p.omega.matrix);
    assert!(close(p.sigma.values[0], 0.1234) && close(p.sigma.values[1], 0.25));
    assert!(
        close(p.residual_correlations[0].rho, 0.3),
        "{:?}",
        p.residual_correlations
    );
    assert_eq!(
        p.omega_iov.unwrap().matrix[(0, 0)].to_bits(),
        0.0456f64.to_bits()
    );
}

/// An estimate the export cannot declare is an error naming it, with the
/// way out; `network_only` takes it and keeps the model's own values.
#[test]
fn an_estimate_with_no_declaration_is_refused_unless_network_only() {
    let mut fit = fixture_fit(Some(FIXTURE_MODEL));
    fit.theta_names.push("TVEXTRA".into());
    fit.theta.push(2.0);
    let err = export_dcm_from_fit(FIXTURE_MODEL, &fit, &DcmExportOptions::default()).unwrap_err();
    assert!(
        err.contains("theta TVEXTRA") && err.contains("--network-only"),
        "{err}"
    );
    let mut fit = fixture_fit(Some(FIXTURE_MODEL));
    fit.eta_names[1] = "ETA_OTHER".into();
    let err = export_dcm_from_fit(FIXTURE_MODEL, &fit, &DcmExportOptions::default()).unwrap_err();
    assert!(err.contains("omega ETA_OTHER"), "{err}");
    let mut fit = fixture_fit(Some(FIXTURE_MODEL));
    fit.sigma_names[0] = "ADD_ERR".into();
    let err = export_dcm_from_fit(FIXTURE_MODEL, &fit, &DcmExportOptions::default()).unwrap_err();
    assert!(err.contains("sigma ADD_ERR"), "{err}");
    let mut opts = DcmExportOptions::default();
    opts.network_only = true;
    let text = export_dcm_from_fit(FIXTURE_MODEL, &fit, &opts).unwrap();
    assert!(text.contains("theta TVV(30, 1, 100)"), "{text}");
    assert!(
        text.contains("keeps the value this file declared"),
        "{text}"
    );
}

/// A repeated network input is named as such, not blamed on a second block.
#[test]
fn a_repeated_input_or_output_is_named() {
    let (names, theta) = fixture_theta();
    for (from, to, dup) in [
        ("[WT, CRCL]", "[WT, wt]", "`inputs` lists `wt`"),
        ("[CL, KA]", "[CL, CL]", "`outputs` lists `CL`"),
    ] {
        let model = FIXTURE_MODEL.replacen(from, to, 1);
        let err = export(&model, &names, &theta, DcmWeightForm::Literal).unwrap_err();
        assert!(err.contains(dup), "{err}");
    }
}

/// A block this build cannot parse is left out of the validation parse and kept
/// in the export, so a joint DCM exports from the default-feature `ferx`.
#[cfg(not(feature = "survival"))]
#[test]
fn a_block_this_build_cannot_parse_is_kept_and_not_validated() {
    let (names, theta) = fixture_theta();
    let model =
        format!("{FIXTURE_MODEL}\n[event_model]\n  cmt = 2\n  hazard = 0.01 * TYPICAL_PK.CL\n");
    let text = export(&model, &names, &theta, DcmWeightForm::Literal).unwrap();
    // The straddle: the export itself does not parse in this build.
    assert!(parse_model_string(&text).is_err());
    assert!(
        text.contains("[event_model]\n  cmt = 2\n  hazard = 0.01 * TYPICAL_PK_CL"),
        "{text}"
    );
}

/// The expression language's `/` returns 0 below a 1e-30 divisor; a network
/// dividing by such a scale cannot be written with it.
#[test]
fn a_scale_below_the_division_guard_is_refused() {
    let (names, theta) = fixture_theta();
    let model = FIXTURE_MODEL.replace("scale      = [12, 23.5]", "scale      = [12, 1e-31]");
    let err = export(&model, &names, &theta, DcmWeightForm::Literal).unwrap_err();
    assert!(err.contains("CRCL") && err.contains("1e-30"), "{err}");
    let model = FIXTURE_MODEL.replace("scale      = [12, 23.5]", "scale      = [12]");
    let err = export(&model, &names, &theta, DcmWeightForm::Literal).unwrap_err();
    assert!(err.contains("one entry per input"), "{err}");
}

/// The reference rewrite touches code only: a comment mentioning the network is
/// copied verbatim, a number's exponent is not mistaken for an identifier, and
/// `NAME . OUT` with spaces is a reference (the tokenizer drops the spaces).
#[test]
fn reference_rewrite_is_confined_to_code() {
    let (names, theta) = fixture_theta();
    let model = FIXTURE_MODEL.replace(
        "  V  = TVV * exp(ETA_V)",
        "  V  = TVV * 1e0 * exp(ETA_V)   # not TYPICAL_PK.CL",
    );
    let text = export(&model, &names, &theta, DcmWeightForm::Literal).unwrap();
    assert!(
        text.contains("  V  = TVV * 1e0 * exp(ETA_V)   # not TYPICAL_PK.CL\n"),
        "{text}"
    );
    assert!(
        text.contains("  KA = TYPICAL_PK_KA * exp(ETA_KA)\n"),
        "{text}"
    );
    assert!(text.contains("  CL = TYPICAL_PK_CL * exp(ETA_CL)   # clearance from the network\n"));
}

const ALL_ACTIVATIONS: [Activation; 6] = [
    Activation::Identity,
    Activation::Relu,
    Activation::Softplus,
    Activation::Tanh,
    Activation::Sigmoid,
    Activation::Exp,
];

// ── Parity (nn only) ────────────────────────────────────────────────────────

#[cfg(feature = "nn")]
mod parity {
    use super::*;
    use crate::nn::{CovariateMapper, MlpMapper, NamedMlpMapper};
    use crate::types::{CompiledModel, PkParams};
    use std::collections::HashMap;

    /// The fixture network with `hidden` / `output` activations and an `init`-free
    /// weight vector chosen so every defect class can show: negative weights and
    /// biases, and an output layer whose second head swings across softplus's
    /// whole domain — past both `|z| > 20` thresholds — as `WT` sweeps the grid.
    fn model_text(hidden: Activation, output: Activation) -> String {
        FIXTURE_MODEL
            .replace(
                "activation = tanh",
                &format!("activation = {}", hidden.as_str()),
            )
            .replace(
                "output     = softplus",
                &format!("output     = {}", output.as_str()),
            )
    }

    /// The fixture θ with the output layer rescaled so the KA head reaches both
    /// softplus tails. Unit 1 of layer 2 is driven hard by `WT`, so its output
    /// sweeps (−1, 1) under `tanh` and the KA pre-activation `≈ 40·h − 2` crosses
    /// −20 and +20.
    fn parity_theta(model: &CompiledModel) -> Vec<f64> {
        let (names, fixture) = fixture_theta();
        let mut theta = model.default_params.theta.clone();
        for (n, v) in names.iter().zip(&fixture) {
            let i = model
                .default_params
                .theta_names
                .iter()
                .position(|t| t == n)
                .unwrap_or_else(|| panic!("{n}"));
            theta[i] = *v;
        }
        let set = |theta: &mut Vec<f64>, n: &str, v: f64| {
            let i = model
                .default_params
                .theta_names
                .iter()
                .position(|t| t == n)
                .unwrap();
            theta[i] = v;
        };
        set(&mut theta, "W_TYPICAL_PK_1_1_1", 3.0);
        set(&mut theta, "W_TYPICAL_PK_2_1_1", 3.0);
        set(&mut theta, "W_TYPICAL_PK_3_2_1", 40.0);
        set(&mut theta, "W_TYPICAL_PK_3_2_2", -0.5);
        set(&mut theta, "B_TYPICAL_PK_3_2", -2.0);
        theta
    }

    /// The exported model's θ, carrying the DCM's value for every θ the two
    /// share by name. The export rewrites only the network: `TVV` keeps the model
    /// file's declared value, so a comparison at the fitted `TVV` sets it here.
    fn plain_theta(plain: &CompiledModel, dcm: &CompiledModel, dcm_theta: &[f64]) -> Vec<f64> {
        let p = &plain.default_params;
        p.theta_names
            .iter()
            .zip(&p.theta)
            .map(|(n, v)| {
                dcm.default_params
                    .theta_names
                    .iter()
                    .position(|t| t == n)
                    .map_or(*v, |i| dcm_theta[i])
            })
            .collect()
    }

    fn weights_of(model: &CompiledModel, theta: &[f64]) -> Vec<f64> {
        let nn = &model.covariate_nns[0];
        theta[nn.weights_offset..nn.weights_offset + nn.mapper.n_weights()].to_vec()
    }

    /// The covariate grid: `WT` across and beyond the population (driving the
    /// softplus tails), `CRCL` on both sides of its negative `center`.
    fn grid() -> Vec<HashMap<String, f64>> {
        let mut out = Vec::new();
        for k in 0..=40 {
            let wt = 20.0 + 3.5 * k as f64;
            for crcl in [-40.0, -5.0, 0.0, 37.25, 140.0] {
                out.push(HashMap::from([
                    ("WT".to_string(), wt),
                    ("CRCL".to_string(), crcl),
                ]));
            }
        }
        out
    }

    fn pk(
        model: &CompiledModel,
        theta: &[f64],
        eta: &[f64],
        cov: &HashMap<String, f64>,
    ) -> PkParams {
        (model.pk_param_fn)(theta, eta, cov, 0.0)
    }

    /// Exported model ≡ network, bit for bit, for every (hidden, output) pair.
    ///
    /// Three readings at each grid point, all compared on the exact bits of
    /// every `PkParams` slot: the network's own `forward_raw` against the
    /// exported model's `CL` / `KA` (η = 0), and the DCM's `pk_param_fn` against
    /// the exported model's at a non-zero η, which is the composition a fit
    /// evaluates. Measured realised error: 0 ULP on every pair (the comparison
    /// is `to_bits` equality, so there is no tolerance to report).
    #[test]
    fn export_reproduces_the_network_bit_for_bit_for_every_activation_pair() {
        let eta = [0.3, -0.2, 0.15];
        for hidden in ALL_ACTIVATIONS {
            for output in ALL_ACTIVATIONS {
                let text = model_text(hidden, output);
                let dcm = parse_model_string(&text).expect("the DCM parses under nn");
                let theta = parity_theta(&dcm);
                let exported = export(
                    &text,
                    &dcm.default_params.theta_names,
                    &theta,
                    DcmWeightForm::Literal,
                )
                .unwrap();
                let plain = parse_model_string(&exported)
                    .unwrap_or_else(|e| panic!("{hidden:?}/{output:?}: {e}\n{exported}"));
                assert!(plain.covariate_nns.is_empty());
                let plain_theta = plain_theta(&plain, &dcm, &theta);
                let mapper = &dcm.covariate_nns[0].mapper;
                let weights = weights_of(&dcm, &theta);
                for cov in grid() {
                    let y = mapper.forward_raw(&weights, &cov).unwrap();
                    let p0 = pk(&plain, &plain_theta, &[0.0; 3], &cov);
                    let ctx = format!("{hidden:?}/{output:?} at {cov:?}");
                    assert_eq!(
                        p0.cl().to_bits(),
                        y[0].to_bits(),
                        "CL {ctx}: {} vs {}",
                        p0.cl(),
                        y[0]
                    );
                    assert_eq!(
                        p0.ka().to_bits(),
                        y[1].to_bits(),
                        "KA {ctx}: {} vs {}",
                        p0.ka(),
                        y[1]
                    );

                    let a = pk(&dcm, &theta, &eta, &cov);
                    let b = pk(&plain, &plain_theta, &eta, &cov);
                    for s in 0..a.values.len() {
                        assert_eq!(
                            a.values[s].to_bits(),
                            b.values[s].to_bits(),
                            "slot {s} {ctx}: {} vs {}",
                            a.values[s],
                            b.values[s]
                        );
                    }
                }
            }
        }
    }

    /// The parity grid reaches all three softplus branches on the KA head — or a
    /// naive `log(1 + exp(z))` emission, which differs only in the tails, would
    /// pass the test above. Read through a twin network with an identity head,
    /// which gives the pre-activation `z` directly.
    #[test]
    fn the_parity_grid_reaches_both_softplus_tails() {
        let dcm = parse_model_string(&model_text(Activation::Tanh, Activation::Softplus)).unwrap();
        let theta = parity_theta(&dcm);
        let weights = weights_of(&dcm, &theta);
        let src = &dcm.covariate_nns[0].mapper;
        let mlp = MlpMapper::new(
            src.mlp().layer_sizes().to_vec(),
            Activation::Tanh,
            Activation::Identity,
        )
        .unwrap();
        let twin = NamedMlpMapper::new(
            mlp,
            src.input_names().to_vec(),
            vec!["CL".into(), "KA".into()],
        )
        .unwrap()
        .with_normalization(src.input_center().to_vec(), src.input_scale().to_vec())
        .unwrap();
        let (mut above, mut below, mut middle) = (0, 0, 0);
        for cov in grid() {
            let z = twin.forward_raw(&weights, &cov).unwrap()[1];
            assert!(z.is_finite());
            if z > 20.0 {
                above += 1;
            } else if z < -20.0 {
                below += 1;
            } else {
                middle += 1;
            }
        }
        assert!(
            above > 0 && below > 0 && middle > 0,
            "z>20: {above}, z<-20: {below}, else: {middle}"
        );
    }

    /// A covariate the data leaves blank is `NaN` in the subject's map, and the
    /// network gives `NaN` for every output that reads it. The export must too:
    /// a `present(X)` guard on the normalisation would hand back a finite,
    /// different value — a subject the DCM refuses becomes one the export fits.
    #[test]
    fn a_missing_input_is_nan_in_the_export_as_in_the_network() {
        let text = model_text(Activation::Tanh, Activation::Softplus);
        let dcm = parse_model_string(&text).unwrap();
        let theta = parity_theta(&dcm);
        let exported = export(
            &text,
            &dcm.default_params.theta_names,
            &theta,
            DcmWeightForm::Literal,
        )
        .unwrap();
        let plain = parse_model_string(&exported).unwrap();
        let weights = weights_of(&dcm, &theta);
        for missing in ["WT", "CRCL"] {
            let mut cov = HashMap::from([("WT".to_string(), 64.0), ("CRCL".to_string(), 90.0)]);
            cov.insert(missing.to_string(), f64::NAN);
            let y = dcm.covariate_nns[0]
                .mapper
                .forward_raw(&weights, &cov)
                .unwrap();
            let p = pk(&plain, &plain_theta(&plain, &dcm, &theta), &[0.0; 3], &cov);
            assert!(
                y[0].is_nan() && y[1].is_nan(),
                "network with {missing} = NaN: {y:?}"
            );
            assert!(
                p.cl().is_nan() && p.ka().is_nan(),
                "export with {missing} = NaN must be NaN like the network, got CL={} KA={}",
                p.cl(),
                p.ka()
            );
            // The rest of the model is untouched by the missing input.
            assert_eq!(
                p.v().to_bits(),
                pk(&dcm, &theta, &[0.0; 3], &cov).v().to_bits()
            );
        }
    }

    /// A covariate *absent from the map* — a sparse snapshot, a direct
    /// `pk_param_fn` call — is `0.0` to every expression, and the network reads
    /// it the same way, so the export agrees there too. Distinct from the `NaN`
    /// case above: the key is removed, not set.
    #[test]
    fn an_absent_input_key_agrees_between_the_export_and_the_network() {
        let text = model_text(Activation::Tanh, Activation::Softplus);
        let dcm = parse_model_string(&text).unwrap();
        let theta = parity_theta(&dcm);
        let exported = export(
            &text,
            &dcm.default_params.theta_names,
            &theta,
            DcmWeightForm::Literal,
        )
        .unwrap();
        let plain = parse_model_string(&exported).unwrap();
        let plain_th = plain_theta(&plain, &dcm, &theta);
        let eta = [0.2, -0.1, 0.3];
        for missing in ["WT", "CRCL"] {
            let mut cov = HashMap::from([("WT".to_string(), 64.0), ("CRCL".to_string(), 90.0)]);
            cov.remove(missing);
            assert!(!cov.contains_key(missing));
            let a = pk(&dcm, &theta, &eta, &cov);
            let b = pk(&plain, &plain_th, &eta, &cov);
            assert!(a.cl().is_finite() && a.ka().is_finite());
            for s in 0..a.values.len() {
                assert_eq!(
                    a.values[s].to_bits(),
                    b.values[s].to_bits(),
                    "slot {s}, {missing} absent"
                );
            }
        }
    }

    /// A direct read of a weight θ evaluates to the same value after the export,
    /// in both weight forms.
    #[test]
    fn a_direct_weight_read_matches_the_network_bit_for_bit() {
        let text = model_text(Activation::Tanh, Activation::Softplus).replace(
            "  V  = TVV * exp(ETA_V)",
            "  V  = TVV * exp(ETA_V + 0.01 * B_TYPICAL_PK_3_1 - W_TYPICAL_PK_1_1_1)",
        );
        let dcm = parse_model_string(&text).unwrap();
        let theta = parity_theta(&dcm);
        for form in [DcmWeightForm::Literal, DcmWeightForm::FixedTheta] {
            let exported = export(&text, &dcm.default_params.theta_names, &theta, form).unwrap();
            let plain = parse_model_string(&exported).unwrap();
            let plain_th = plain_theta(&plain, &dcm, &theta);
            for cov in grid() {
                let a = pk(&dcm, &theta, &[0.1, 0.2, 0.3], &cov);
                let b = pk(&plain, &plain_th, &[0.1, 0.2, 0.3], &cov);
                for s in 0..a.values.len() {
                    assert_eq!(
                        a.values[s].to_bits(),
                        b.values[s].to_bits(),
                        "{form:?} slot {s}"
                    );
                }
            }
        }
    }

    /// `FixedTheta` computes the same function as `Literal`.
    #[test]
    fn fixed_theta_form_matches_the_network_bit_for_bit() {
        let text = model_text(Activation::Relu, Activation::Exp);
        let dcm = parse_model_string(&text).unwrap();
        let theta = parity_theta(&dcm);
        let exported = export(
            &text,
            &dcm.default_params.theta_names,
            &theta,
            DcmWeightForm::FixedTheta,
        )
        .unwrap();
        let plain = parse_model_string(&exported).unwrap();
        let eta = [-0.1, 0.4, 0.0];
        for cov in grid() {
            let a = pk(&dcm, &theta, &eta, &cov);
            let b = pk(&plain, &plain_theta(&plain, &dcm, &theta), &eta, &cov);
            for s in 0..a.values.len() {
                assert_eq!(
                    a.values[s].to_bits(),
                    b.values[s].to_bits(),
                    "slot {s} at {cov:?}"
                );
            }
        }
    }
}
