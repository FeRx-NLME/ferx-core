//! `ferx export-dcm` — write a fitted `[covariate_nn]` model out as a plain
//! population PK model (#1331).
//!
//! Thin by construction: read the fit, call
//! [`ferx_core::nn::export::export_dcm_as_analytic`], write the text. The
//! transform itself lives in `ferx-core`.

use ferx_core::nn::export::{export_dcm_from_fit, DcmExportOptions, DcmWeightForm};
use ferx_core::types::FitResult;
use std::path::Path;

pub const EXPORT_DCM_USAGE: &str = "\
Usage: ferx export-dcm <run.fitrx>                       [-o <out.ferx>] [--fixed-theta]
       ferx export-dcm <model.ferx> --fit <run.fitrx|run.json> [-o <out.ferx>] [--fixed-theta]

Writes a fitted deep compartment model (a model with [covariate_nn] blocks) out
as an ordinary population PK model: each network becomes plain
[individual_parameters] equations at the fitted weights. The result has no
[covariate_nn] block, loads without the `nn` feature, and predicts exactly
what the fitted network does. Only the networks are rewritten: the other
theta / omega / sigma declarations keep the values the model file gives them.

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
  -h, --help        print this help and exit
";

fn value<'a>(args: &'a [String], names: &[&str]) -> Result<Option<&'a str>, String> {
    let Some(i) = args.iter().position(|a| names.contains(&a.as_str())) else {
        return Ok(None);
    };
    match args.get(i + 1) {
        Some(v) if !v.starts_with('-') => Ok(Some(v.as_str())),
        _ => Err(format!("{} requires a value", args[i])),
    }
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
fn load(args: &[String]) -> Result<(String, FitResult, std::path::PathBuf), String> {
    let first = args
        .get(2)
        .filter(|a| !a.starts_with('-'))
        .ok_or("expected a .fitrx bundle or a model file")?;
    let first = Path::new(first);
    let fit_flag = value(args, &["--fit"])?;
    let is_fitrx = first
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("fitrx"));
    let (text, fit) = if is_fitrx {
        if fit_flag.is_some() {
            return Err("--fit is only for a model-file first argument".into());
        }
        let loaded = ferx_core::io::fitrx::load_fit_without_population(first)
            .map_err(|e| format!("failed to read `{}`: {e}", first.display()))?;
        (loaded.model_source, loaded.fit)
    } else {
        let fit_path = fit_flag.ok_or("a model file needs --fit <run.fitrx|run.json>")?;
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
    let known_valued = ["--fit", "-o", "--output"];
    let mut i = 3;
    while i < args.len() {
        let a = args[i].as_str();
        if known_valued.contains(&a) {
            i += 2;
            continue;
        }
        if a != "--fixed-theta" {
            eprintln!("Error: unexpected argument `{a}`");
            eprint!("{EXPORT_DCM_USAGE}");
            return 2;
        }
        i += 1;
    }
    let (text, fit, default_out) = match load(args) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error: {e}");
            eprint!("{EXPORT_DCM_USAGE}");
            return 2;
        }
    };
    let out = match value(args, &["-o", "--output"]) {
        Ok(Some(p)) => std::path::PathBuf::from(p),
        Ok(None) => default_out,
        Err(e) => {
            eprintln!("Error: {e}");
            return 2;
        }
    };
    let fixed = args.iter().any(|a| a == "--fixed-theta");
    match write_export(&text, &fit, fixed, &out) {
        Ok(()) => {
            println!("Wrote {}", out.display());
            0
        }
        Err(e) => {
            eprintln!("Error: {e}");
            1
        }
    }
}

/// Export `text` at `fit`'s weights — after checking `fit` was made with the
/// networks `text` declares — and write it to `out`.
fn write_export(text: &str, fit: &FitResult, fixed_theta: bool, out: &Path) -> Result<(), String> {
    let mut opts = DcmExportOptions::default();
    if fixed_theta {
        opts.weight_form = DcmWeightForm::FixedTheta;
    }
    let exported = export_dcm_from_fit(text, fit, &opts)?;
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
        write_export(&model, &fit, false, &out).unwrap();
        let got = std::fs::read_to_string(&out).unwrap();
        let want = std::fs::read_to_string(root.join("expected_analytic.ferx")).unwrap();
        assert_eq!(got, want);
        // A model whose network is not the fit's is an error, not a file.
        let other = model.replace("activation = tanh", "activation = relu");
        let err = write_export(&other, &fit, false, &out).unwrap_err();
        assert!(
            err.contains("not the network the fit was made with"),
            "{err}"
        );
    }
}
