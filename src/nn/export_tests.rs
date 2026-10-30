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
