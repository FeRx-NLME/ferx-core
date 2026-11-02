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
    match write_export(&text, &fit, &opts, &out) {
        Ok(()) => {
            println!("Wrote {}", out.display());
            if parsed.network_only {
                println!(
                    "Note: --network-only: every theta, omega and sigma outside the \
                     networks keeps the value the model file declares."
                );
            }
            for note in notes(&text, Path::new(&parsed.input), &out) {
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

/// What the user should know about the written export that the export cannot
/// say for itself.
///
/// A relative `[data] path` or `[priors] from_fit` is copied as written, and
/// resolves against wherever the export is loaded from; the original resolved it
/// against its own directory, which for a `.fitrx` this command does not know. A
/// `from_fit` prior was made for the network model: its network-weight entries
/// match no free theta in the export and are skipped when it is loaded.
fn notes(text: &str, input: &Path, out: &Path) -> Vec<String> {
    let dir = |p: &Path| {
        p.parent()
            .map(|d| d.canonicalize().unwrap_or_else(|_| d.to_path_buf()))
    };
    let from_bundle = input
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("fitrx"));
    let moved = from_bundle || dir(input) != dir(out);
    let mut notes = Vec::new();
    for (key, value) in file_entries(text) {
        let relative = !Path::new(&value).is_absolute() && !value.starts_with('~');
        if relative && moved {
            notes.push(format!(
                "{key} = \"{value}\" is relative and copied as written; it now resolves \
                 against `{}`, not the original model's directory",
                out.parent()
                    .filter(|p| !p.as_os_str().is_empty())
                    .unwrap_or(Path::new("."))
                    .display()
            ));
        }
        if key == "[priors] from_fit" {
            notes.push(format!(
                "{key} = \"{value}\" was made for the network model; its network-weight \
                 entries match no free theta in the export and are skipped when it loads"
            ));
        }
    }
    notes
}

/// Export `text` at `fit` — after checking `fit` was made with the networks
/// `text` declares — and write it to `out`.
fn write_export(
    text: &str,
    fit: &FitResult,
    opts: &DcmExportOptions,
    out: &Path,
) -> Result<(), String> {
    let exported = export_dcm_from_fit(text, fit, opts)?;
    std::fs::write(out, exported).map_err(|e| format!("cannot write `{}`: {e}", out.display()))
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
        let same = notes(text, here, Path::new("models/m-analytic.ferx"));
        assert_eq!(same.len(), 1, "{same:?}");
        assert!(same[0].contains("network-weight"), "{same:?}");
        let moved = notes(text, here, Path::new("out/m.ferx"));
        assert!(
            moved
                .iter()
                .any(|n| n.contains("\"../data/pk.csv\" is relative") && n.contains("`out`")),
            "{moved:?}"
        );
        assert!(
            moved
                .iter()
                .any(|n| n.contains("\"prev.fitrx\" is relative")),
            "{moved:?}"
        );
        let bundle = notes(text, Path::new("run.fitrx"), Path::new("run-analytic.ferx"));
        assert!(
            bundle.iter().any(|n| n.contains("is relative")),
            "{bundle:?}"
        );
        let absolute = "[data]\n  path = \"/abs/pk.csv\"\n";
        assert!(notes(absolute, Path::new("run.fitrx"), Path::new("x.ferx")).is_empty());
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
        write_export(&model, &fit, &DcmExportOptions::default(), &out).unwrap();
        let got = std::fs::read_to_string(&out).unwrap();
        let want = std::fs::read_to_string(root.join("expected_fitted.ferx")).unwrap();
        assert_eq!(got, want);
        let mut network_only = DcmExportOptions::default();
        network_only.network_only = true;
        write_export(&model, &fit, &network_only, &out).unwrap();
        let got = std::fs::read_to_string(&out).unwrap();
        let want = std::fs::read_to_string(root.join("expected_analytic.ferx")).unwrap();
        assert_eq!(got, want);
        // A model whose network is not the fit's is an error, not a file.
        let other = model.replace("activation = tanh", "activation = relu");
        let err = write_export(&other, &fit, &DcmExportOptions::default(), &out).unwrap_err();
        assert!(
            err.contains("not the network the fit was made with"),
            "{err}"
        );
    }
}
