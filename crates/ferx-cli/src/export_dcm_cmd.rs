//! `ferx export-dcm` — write a fitted `[covariate_nn]` model out as a plain
//! population PK model (#1331).
//!
//! Thin by construction: read the fit, call
//! [`ferx_core::nn::export::export_dcm_from_fit`], write the text. The
//! transform itself lives in `ferx-core`.

use ferx_core::nn::export::{export_dcm_from_fit, DcmExportOptions, DcmWeightForm};
use ferx_core::types::FitResult;
use std::path::Path;

pub const EXPORT_DCM_USAGE: &str = "\
Usage: ferx export-dcm <run.fitrx>                       [options]
       ferx export-dcm <model.ferx> --fit <run.fitrx|run.json> [options]

Writes a fitted deep compartment model (a model with [covariate_nn] blocks) out
as an ordinary population PK model: each network becomes plain
[individual_parameters] equations at the fitted weights. The result has no
[covariate_nn] block and loads without the `nn` feature. Each network's
equations give exactly what the fitted network does, and every other theta /
omega / sigma / kappa is written at the fit's estimate, so the export is the
fitted model. A declaration the estimate cannot be written into (a vector
theta, a [covariate_model] theta, a [mixture] class override) is an error;
--network-only then keeps the model file's values for everything but the
networks.

With a separate model file, the fit must have been made with the same networks
(same inputs, outputs, layers, activations, center and scale); a mismatch is an
error rather than an export of the weights under a different network.

The weights are read at full precision, so the fit must be a `.fitrx` bundle
or a `--output-format json` fit file. The `{model}-fit.yaml` report is not
accepted: it rounds to 6 decimals and does not list the network weights.

  --fit PATH        the fit to read the weights from, when the first argument
                    is the model file
  -o, --output PATH write the exported model to PATH
                    (default: {model}-analytic.ferx next to the input)
  --fixed-theta     declare each weight as a FIXed theta and reference it by
                    name, instead of inlining the values
  --network-only    write only the networks at the fit's values; keep every
                    other parameter at the value the model file declares
  -h, --help        print this help and exit
";

/// The parsed arguments: one positional and the flags, in any order.
#[derive(Debug, Default, PartialEq)]
struct Args {
    input: String,
    fit: Option<String>,
    output: Option<String>,
    fixed_theta: bool,
    network_only: bool,
}

fn parse_args(args: &[String]) -> Result<Args, String> {
    let mut parsed = Args::default();
    let mut input = None;
    let mut it = args.iter().skip(2);
    while let Some(a) = it.next() {
        let slot = match a.as_str() {
            "--fit" => &mut parsed.fit,
            "-o" | "--output" => &mut parsed.output,
            "--fixed-theta" => {
                parsed.fixed_theta = true;
                continue;
            }
            "--network-only" => {
                parsed.network_only = true;
                continue;
            }
            _ if a.starts_with('-') => return Err(format!("unexpected argument `{a}`")),
            _ if input.is_none() => {
                input = Some(a.clone());
                continue;
            }
            _ => return Err(format!("unexpected argument `{a}`")),
        };
        match it.next() {
            Some(v) if !v.starts_with('-') => *slot = Some(v.clone()),
            _ => return Err(format!("{a} requires a value")),
        }
    }
    parsed.input = input.ok_or("expected a .fitrx bundle or a model file")?;
    Ok(parsed)
}

fn read_fit(path: &Path) -> Result<FitResult, String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    match ext.as_str() {
        "fitrx" => ferx_core::io::fitrx::load_fit_without_population(path)
            .map(|l| l.fit)
            .map_err(|e| format!("failed to read `{}`: {e}", path.display())),
        "json" => {
            let text = std::fs::read_to_string(path)
                .map_err(|e| format!("failed to read `{}`: {e}", path.display()))?;
            serde_json::from_str(&text).map_err(|e| {
                format!(
                    "`{}` is not a ferx fit JSON (written by `--output-format json`): {e}",
                    path.display()
                )
            })
        }
        "yaml" | "yml" => Err(format!(
            "`{}`: the fit YAML report rounds to 6 decimals and does not list the \
             network weights; pass the `.fitrx` bundle or a `--output-format json` fit",
            path.display()
        )),
        other => Err(format!(
            "`{}`: unsupported fit file extension `{other}` (expected `.fitrx` or `.json`)",
            path.display()
        )),
    }
}

/// `(model text, fit, default output path)` from the arguments.
fn load(args: &Args) -> Result<(String, FitResult, std::path::PathBuf), String> {
    let first = Path::new(&args.input);
    let is_fitrx = first
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("fitrx"));
    let (text, fit) = if is_fitrx {
        if args.fit.is_some() {
            return Err("--fit is only for a model-file first argument".into());
        }
        let loaded = ferx_core::io::fitrx::load_fit_without_population(first)
            .map_err(|e| format!("failed to read `{}`: {e}", first.display()))?;
        (loaded.model_source, loaded.fit)
    } else {
        let fit_path = args
            .fit
            .as_deref()
            .ok_or("a model file needs --fit <run.fitrx|run.json>")?;
        let text = std::fs::read_to_string(first)
            .map_err(|e| format!("cannot read the model file `{}`: {e}", first.display()))?;
        (text, read_fit(Path::new(fit_path))?)
    };
    let stem = first
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    let out = first.with_file_name(format!("{stem}-analytic.ferx"));
    Ok((text, fit, out))
}

pub fn run(args: &[String]) -> i32 {
    if args.iter().skip(2).any(|a| a == "-h" || a == "--help") {
        print!("{EXPORT_DCM_USAGE}");
        return 0;
    }
    let loaded = parse_args(args).and_then(|a| load(&a).map(|l| (a, l)));
    let (parsed, (text, fit, default_out)) = match loaded {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error: {e}");
            eprint!("{EXPORT_DCM_USAGE}");
            return 2;
        }
    };
    let out = parsed
        .output
        .clone()
        .map(std::path::PathBuf::from)
        .unwrap_or(default_out);
    let inputs = std::iter::once(&parsed.input).chain(&parsed.fit);
    if let Some(input) = inputs.map(Path::new).find(|i| same_file(i, &out)) {
        eprintln!(
            "Error: the output `{}` is the input `{}`; writing it would overwrite the \
             original. Pass a different -o",
            out.display(),
            input.display()
        );
        return 2;
    }
    let mut opts = DcmExportOptions::default();
    if parsed.fixed_theta {
        opts.weight_form = DcmWeightForm::FixedTheta;
    }
    opts.network_only = parsed.network_only;
    let input = Path::new(&parsed.input);
    match write_export(&text, &fit, &opts, &out, model_dir(input)) {
        Ok(prior_checked) => {
            println!("Wrote {}", out.display());
            if parsed.network_only {
                println!(
                    "Note: --network-only: every theta, omega and sigma outside the \
                     networks keeps the value the model file declares."
                );
            }
            for note in notes(&text, input, &out, prior_checked) {
                println!("Note: {note}");
            }
            0
        }
        Err(e) => {
            eprintln!("Error: {e}");
            1
        }
    }
}

/// Do `a` and `b` name the same existing file?
fn same_file(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

/// The `key = value` entries of `[data]` / `[priors]` that name a file.
fn file_entries(text: &str) -> Vec<(&'static str, String)> {
    let mut block = String::new();
    let mut out = Vec::new();
    for line in text.lines() {
        let code = line.split('#').next().unwrap_or_default().trim();
        if let Some(inner) = code.strip_prefix('[').and_then(|c| c.strip_suffix(']')) {
            block = inner
                .split_whitespace()
                .next()
                .unwrap_or_default()
                .to_string();
            continue;
        }
        let Some((key, value)) = code.split_once('=') else {
            continue;
        };
        let key = match (block.as_str(), key.trim()) {
            ("data", "path") => "[data] path",
            ("priors", "from_fit") => "[priors] from_fit",
            _ => continue,
        };
        let value = value.trim().trim_matches(|c| c == '"' || c == '\'');
        out.push((key, value.to_string()));
    }
    out
}

/// The directory the model's relative paths resolved against, when known: the
/// model file's own. A `.fitrx` bundle does not record it.
fn model_dir(input: &Path) -> Option<&Path> {
    let bundle = input
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("fitrx"));
    (!bundle).then(|| input.parent().unwrap_or(Path::new("")))
}

/// What the user should know about the written export that the export cannot
/// say for itself.
///
/// A relative `[data] path` or `[priors] from_fit` is copied as written, and
/// resolves against wherever the export is loaded from. A `from_fit` prior was
/// made for the network model: its network-weight entries match no free theta
/// in the export and are skipped when it loads. `prior_checked` says whether
/// [`check_prior_import`] could load the export with it.
fn notes(text: &str, input: &Path, out: &Path, prior_checked: bool) -> Vec<String> {
    let out_dir = out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let canonical = |d: &Path| d.canonicalize().unwrap_or_else(|_| d.to_path_buf());
    let mut notes = Vec::new();
    for (key, value) in file_entries(text) {
        let relative = !Path::new(&value).is_absolute() && !value.starts_with('~');
        match model_dir(input) {
            _ if !relative => {}
            None => notes.push(format!(
                "{key} = \"{value}\" is relative and copied as written; the bundle does \
                 not record the original model's directory, so check that it resolves \
                 against `{}`",
                out_dir.display()
            )),
            Some(dir) if canonical(dir) != canonical(out_dir) => notes.push(format!(
                "{key} = \"{value}\" is relative and copied as written; it now resolves \
                 against `{}`, not the model's directory `{}`",
                out_dir.display(),
                dir.display()
            )),
            Some(_) => {}
        }
        if key == "[priors] from_fit" {
            notes.push(if prior_checked {
                format!(
                    "{key} = \"{value}\" was made for the network model; its network-weight \
                     entries match no free theta in the export and are skipped (the export \
                     was loaded with it: the other entries still import)"
                )
            } else {
                format!(
                    "{key} = \"{value}\" was made for the network model; its network-weight \
                     entries match no free theta in the export and are skipped. It could not \
                     be loaded from here to check the rest: if only network weights imported \
                     from it, the export will not load until `from_fit` is removed"
                )
            });
        }
    }
    notes
}

/// `text` with its `[priors] from_fit` pointed at `path`.
fn with_from_fit(text: &str, path: &Path) -> String {
    let mut block = "";
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        let code = line.split('#').next().unwrap_or_default().trim();
        if let Some(inner) = code.strip_prefix('[').and_then(|c| c.strip_suffix(']')) {
            block = if inner.trim() == "priors" {
                "priors"
            } else {
                ""
            };
        }
        let key = code.split_once('=').map(|(k, _)| k.trim());
        if block == "priors" && key == Some("from_fit") {
            out.push_str(&format!("  from_fit = \"{}\"\n", path.display()));
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// Load the export with its `[priors] from_fit` resolved where the model
/// resolved it, so a prior source whose only importable estimates were network
/// weights — which the export no longer has free — is refused here rather than
/// when the export is next loaded. `Ok(true)` when it loaded, `Ok(false)` when it
/// could not be checked (no `from_fit`, a bundle's unknown directory, a source
/// not found from here, or a block this build cannot parse).
fn check_prior_import(exported: &str, model_dir: Option<&Path>) -> Result<bool, String> {
    let Some((_, value)) = file_entries(exported)
        .into_iter()
        .find(|(k, _)| *k == "[priors] from_fit")
    else {
        return Ok(false);
    };
    let path = Path::new(&value);
    let resolved = match model_dir {
        _ if path.is_absolute() => path.to_path_buf(),
        Some(dir) => dir.join(path),
        None => return Ok(false),
    };
    if !resolved.is_file() {
        return Ok(false);
    }
    match ferx_core::parse_model_string(&with_from_fit(exported, &resolved)) {
        Ok(_) => Ok(true),
        Err(e) if e.contains("[priors] from_fit") => Err(format!(
            "the export would not load: {e} The network's weights are constants in the \
             export, so a prior on them no longer applies; remove `from_fit` from \
             [priors], or point it at a fit with estimates for the other parameters"
        )),
        Err(_) => Ok(false),
    }
}

/// Export `text` at `fit` — after checking `fit` was made with the networks
/// `text` declares — check its prior import, and write it to `out`. Returns
/// whether the `from_fit` prior was checked.
fn write_export(
    text: &str,
    fit: &FitResult,
    opts: &DcmExportOptions,
    out: &Path,
    model_dir: Option<&Path>,
) -> Result<bool, String> {
    let exported = export_dcm_from_fit(text, fit, opts)?;
    let checked = check_prior_import(&exported, model_dir)?;
    std::fs::write(out, exported).map_err(|e| format!("cannot write `{}`: {e}", out.display()))?;
    Ok(checked)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Vec<String> {
        std::iter::once("ferx")
            .chain(v.iter().copied())
            .map(String::from)
            .collect()
    }

    #[test]
    fn help_returns_0() {
        assert_eq!(run(&args(&["export-dcm", "--help"])), 0);
    }

    #[test]
    fn missing_input_and_stray_flags_are_usage_errors() {
        assert_eq!(run(&args(&["export-dcm"])), 2);
        assert_eq!(run(&args(&["export-dcm", "m.ferx", "--bogus"])), 2);
        // A model file without --fit.
        assert_eq!(run(&args(&["export-dcm", "m.ferx"])), 2);
    }

    /// Flags may come before, after or around the positional.
    #[test]
    fn flags_and_the_input_parse_in_any_order() {
        let want = Args {
            input: "m.ferx".into(),
            fit: Some("run.json".into()),
            output: Some("out.ferx".into()),
            fixed_theta: true,
            network_only: false,
        };
        for order in [
            &[
                "m.ferx",
                "--fit",
                "run.json",
                "-o",
                "out.ferx",
                "--fixed-theta",
            ][..],
            &[
                "-o",
                "out.ferx",
                "--fixed-theta",
                "m.ferx",
                "--fit",
                "run.json",
            ],
            &[
                "--fit",
                "run.json",
                "m.ferx",
                "--output",
                "out.ferx",
                "--fixed-theta",
            ],
        ] {
            let mut a = vec!["export-dcm"];
            a.extend_from_slice(order);
            assert_eq!(parse_args(&args(&a)).as_ref(), Ok(&want), "{order:?}");
        }
        let one = |v: &[&str]| parse_args(&args(&[&["export-dcm"][..], v].concat()));
        assert_eq!(one(&["run.fitrx"]).unwrap().input, "run.fitrx");
        assert!(one(&["a.ferx", "b.ferx"])
            .unwrap_err()
            .contains("unexpected argument `b.ferx`"));
        assert!(one(&["a.ferx", "-o"])
            .unwrap_err()
            .contains("-o requires a value"));
        assert!(one(&["-o", "--fit", "x"])
            .unwrap_err()
            .contains("-o requires a value"));
        assert!(one(&["--fixed-theta"])
            .unwrap_err()
            .contains("expected a .fitrx"));
    }

    /// `-o` naming the model or the fit is refused before anything is written.
    #[test]
    fn an_output_that_is_an_input_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("m.ferx");
        let fit = dir.path().join("run.json");
        std::fs::write(&model, "keep").unwrap();
        std::fs::write(&fit, "keep").unwrap();
        let (m, f) = (model.to_str().unwrap(), fit.to_str().unwrap());
        for out in [m, f] {
            assert_eq!(run(&args(&["export-dcm", m, "--fit", f, "-o", out])), 2);
        }
        assert_eq!(std::fs::read_to_string(&model).unwrap(), "keep");
        assert_eq!(std::fs::read_to_string(&fit).unwrap(), "keep");
        assert!(same_file(&model, &dir.path().join(".").join("m.ferx")));
        assert!(!same_file(&model, &dir.path().join("new.ferx")));
    }

    /// A relative path is flagged when the export lands elsewhere, and only
    /// then; a `from_fit` prior is always flagged.
    #[test]
    fn relative_paths_and_from_fit_priors_are_noted() {
        let text =
            "[data]\n  path = \"../data/pk.csv\"  # rel\n[priors]\n  from_fit = prev.fitrx\n";
        let here = Path::new("models/m.ferx");
        let same = notes(text, here, Path::new("models/m-analytic.ferx"), true);
        assert_eq!(same.len(), 1, "{same:?}");
        assert!(
            same[0].contains("the other entries still import"),
            "{same:?}"
        );
        let unchecked = notes(text, here, Path::new("models/m-analytic.ferx"), false);
        assert!(
            unchecked[0].contains("could not be loaded from here"),
            "{unchecked:?}"
        );
        let moved = notes(text, here, Path::new("out/m.ferx"), true);
        assert!(
            moved
                .iter()
                .any(|n| n.contains("\"../data/pk.csv\" is relative")
                    && n.contains(
                        "now resolves against `out`, not the model's directory `models`"
                    )),
            "{moved:?}"
        );
        assert!(
            moved
                .iter()
                .any(|n| n.contains("\"prev.fitrx\" is relative")),
            "{moved:?}"
        );
        // A bundle does not know the original directory, and says so rather than
        // claiming the path moved.
        let bundle = notes(
            text,
            Path::new("run.fitrx"),
            Path::new("run-analytic.ferx"),
            false,
        );
        let rel: Vec<_> = bundle
            .iter()
            .filter(|n| n.contains("is relative"))
            .collect();
        assert_eq!(rel.len(), 2, "{bundle:?}");
        assert!(
            rel.iter().all(
                |n| n.contains("does not record the original model's directory")
                    && n.contains("against `.`")
                    && !n.contains("now resolves")
            ),
            "{bundle:?}"
        );
        let absolute = "[data]\n  path = \"/abs/pk.csv\"\n";
        assert!(notes(absolute, Path::new("run.fitrx"), Path::new("x.ferx"), false).is_empty());
    }

    /// A `from_fit` prior whose only importable estimates were network weights
    /// would leave the export unloadable; it is refused before anything is
    /// written. The control source also estimates `TVV` and exports.
    #[test]
    fn a_from_fit_prior_that_would_import_nothing_is_refused() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/dcm_export");
        let fit_path = root.join("fit.json");
        let model = std::fs::read_to_string(root.join("model.ferx")).unwrap();
        let fit: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&fit_path).unwrap()).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let m = dir.path().join("m.ferx");
        std::fs::write(
            &m,
            format!("{model}\n[priors]\n  from_fit = \"src.json\"\n"),
        )
        .unwrap();
        let out = dir.path().join("out.ferx");
        let run_with = |source_tvv: &str| {
            let mut src = fit.clone();
            let n = src["theta"].as_array().unwrap().len();
            src["se_theta"] = serde_json::json!(vec![0.05; n]);
            src["theta_names"][0] = serde_json::json!(source_tvv);
            std::fs::write(dir.path().join("src.json"), src.to_string()).unwrap();
            let _ = std::fs::remove_file(&out);
            let code = run(&args(&[
                "export-dcm",
                m.to_str().unwrap(),
                "--fit",
                fit_path.to_str().unwrap(),
                "-o",
                out.to_str().unwrap(),
            ]));
            (code, out.exists())
        };
        assert_eq!(fit["theta_names"][0], "TVV");
        assert_eq!(run_with("OTHER"), (1, false));
        let text = std::fs::read_to_string(&m).unwrap();
        let err = check_prior_import(&text_export(&text, &fit_path), Some(dir.path())).unwrap_err();
        assert!(
            err.contains("no prior could be imported") && err.contains("remove `from_fit`"),
            "{err}"
        );
        assert_eq!(run_with("TVV"), (0, true));
    }

    fn text_export(model: &str, fit_path: &Path) -> String {
        export_dcm_from_fit(
            model,
            &read_fit(fit_path).unwrap(),
            &DcmExportOptions::default(),
        )
        .unwrap()
    }

    #[test]
    fn the_yaml_report_is_refused_for_its_precision() {
        let err = read_fit(Path::new("run-fit.yaml")).unwrap_err();
        assert!(err.contains("6 decimals"), "{err}");
    }

    /// The committed fixture, through the CLI's own path: model file + JSON fit.
    #[test]
    fn writes_the_committed_export_of_the_fixture() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/dcm_export");
        let fit = read_fit(&root.join("fit.json")).unwrap();
        let model = std::fs::read_to_string(root.join("model.ferx")).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dcm-analytic.ferx");
        write_export(&model, &fit, &DcmExportOptions::default(), &out, None).unwrap();
        let got = std::fs::read_to_string(&out).unwrap();
        let want = std::fs::read_to_string(root.join("expected_fitted.ferx")).unwrap();
        assert_eq!(got, want);
        let mut network_only = DcmExportOptions::default();
        network_only.network_only = true;
        write_export(&model, &fit, &network_only, &out, None).unwrap();
        let got = std::fs::read_to_string(&out).unwrap();
        let want = std::fs::read_to_string(root.join("expected_analytic.ferx")).unwrap();
        assert_eq!(got, want);
        // A model whose network is not the fit's is an error, not a file.
        let other = model.replace("activation = tanh", "activation = relu");
        let err = write_export(&other, &fit, &DcmExportOptions::default(), &out, None).unwrap_err();
        assert!(
            err.contains("not the network the fit was made with"),
            "{err}"
        );
    }
}
