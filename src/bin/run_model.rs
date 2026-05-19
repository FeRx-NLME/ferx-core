use std::env;
use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();

    // Subcommand dispatch — checked before the positional model-path logic
    // so `ferx translate ...` doesn't collide with `ferx <model.ferx> ...`.
    if args.get(1).map(String::as_str) == Some("translate") {
        run_translate(&args[2..]);
        return;
    }

    if args.len() < 2 {
        eprintln!("Usage: ferx <model.ferx|model.mod|model.cpp> --data <data.csv> [--threads N|auto] [--output <run.fitrx>] [--include-data]");
        eprintln!("       ferx <model> --simulate                       [--threads N|auto] [--output <run.fitrx>]");
        eprintln!("       ferx translate <model.mod|model.cpp>          [-o out.ferx] [--check]");
        eprintln!();
        eprintln!("Fits a NLME model and writes sdtab.csv with residuals.");
        eprintln!("Data must be in NONMEM format (ID, TIME, DV, EVID, AMT, CMT, ...)");
        eprintln!();
        eprintln!("--threads N    use N rayon workers (N > 0)");
        eprintln!("--threads 0    use rayon default (one worker per logical CPU)");
        eprintln!("--threads auto alias for --threads 0");
        eprintln!();
        eprintln!("--output PATH  also write a portable .fitrx fit bundle (zip of JSON+CSV)");
        eprintln!("--include-data embed the input --data CSV inside the .fitrx (off by default)");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let data_path = args
        .iter()
        .position(|a| a == "--data")
        .and_then(|i| args.get(i + 1));
    let simulate = args.iter().any(|a| a == "--simulate");
    let threads = parse_threads_flag(&args);
    let output_path = parse_output_flag(&args);
    let include_data = args.iter().any(|a| a == "--include-data");
    if include_data && output_path.is_none() {
        eprintln!("Warning: --include-data has no effect without --output");
    }
    if include_data && data_path.is_none() {
        eprintln!("Warning: --include-data ignored (no --data file to embed)");
    }

    // Configure rayon's global pool before any parallel work starts. build_global()
    // is once-per-process — correct for a CLI binary. Without a --threads flag we
    // leave rayon's default (one worker per logical CPU) in place.
    if let Some(n) = threads {
        if let Err(e) = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
        {
            eprintln!(
                "Warning: failed to configure thread pool with {} threads: {}",
                n, e
            );
        }
    }

    let t_start = Instant::now();
    let result = if let Some(csv_path) = data_path {
        ferx_core::run_model_with_data(model_path, csv_path)
    } else if simulate {
        ferx_core::run_model_simulate(model_path)
    } else {
        eprintln!("Error: specify --data <file.csv> or --simulate");
        std::process::exit(1);
    };
    let elapsed = t_start.elapsed();

    match result {
        Ok((fit_result, population)) => {
            // Derive model name from model file path
            let model_name = std::path::Path::new(model_path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model");

            let sdtab_path = format!("{}-sdtab.csv", model_name);
            match ferx_core::io::output::write_sdtab_csv(&fit_result, &population, &sdtab_path) {
                Ok(()) => eprintln!("Residuals written to {}", sdtab_path),
                Err(e) => eprintln!("Warning: failed to write sdtab: {}", e),
            }

            let yaml_path = format!("{}-fit.yaml", model_name);
            match ferx_core::io::output::write_estimates_yaml(&fit_result, &yaml_path) {
                Ok(()) => eprintln!("Estimates written to {}", yaml_path),
                Err(e) => eprintln!("Warning: failed to write estimates: {}", e),
            }

            if let Some(out) = &output_path {
                let model_source = std::fs::read_to_string(model_path).unwrap_or_default();
                let include = if include_data {
                    data_path.map(std::path::PathBuf::from)
                } else {
                    None
                };
                let opts = ferx_core::io::fitrx::SaveFitOptions {
                    include_data: include,
                };
                match ferx_core::io::fitrx::save_fit(
                    &fit_result,
                    &population,
                    &model_source,
                    std::path::Path::new(out),
                    opts,
                ) {
                    Ok(()) => eprintln!("Fit bundle written to {}", out),
                    Err(e) => eprintln!("Warning: failed to write fit bundle: {}", e),
                }
            }

            let elapsed_secs = elapsed.as_secs_f64();
            eprintln!("Elapsed fit time: {:.3}s", elapsed_secs);

            // Write timing file alongside outputs
            let timing_path = format!("{}-timing.txt", model_name);
            if let Ok(()) = std::fs::write(
                &timing_path,
                format!("elapsed_seconds={:.6}\n", elapsed_secs),
            ) {
                eprintln!("Timing written to {}", timing_path);
            }

            println!("\nFit completed!");
            println!("OFV: {:.4}", fit_result.ofv);
            println!("Elapsed: {:.3}s", elapsed_secs);
            for (name, val) in fit_result.theta_names.iter().zip(fit_result.theta.iter()) {
                println!("  {} = {:.6}", name, val);
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}

/// Parse the optional `--output` flag. Returns `None` when absent; exits with
/// an error message when present but missing its value, mirroring `--threads`.
/// Handle `ferx translate <input.mod> [-o out.ferx] [--check] [--force]`.
fn run_translate(args: &[String]) {
    let input = match args.iter().find(|a| !a.starts_with('-')) {
        Some(p) => p.clone(),
        None => {
            eprintln!("Usage: ferx translate <model.mod|model.cpp> [-o out.ferx] [--check] [--force]");
            std::process::exit(1);
        }
    };
    let output = parse_o_flag(args);
    let check_only = args.iter().any(|a| a == "--check");
    let force = args.iter().any(|a| a == "--force");

    let path = Path::new(&input);
    let result = ferx_core::translate_mrgsolve_file(path);
    let ferx_text = match result {
        Ok(s) => s,
        Err(e) => {
            eprintln!("translate: {e}");
            std::process::exit(1);
        }
    };

    if check_only {
        eprintln!("translate: {} parsed cleanly ({} bytes of .ferx)", path.display(), ferx_text.len());
        return;
    }

    match output {
        Some(out_path) => {
            let out = Path::new(&out_path);
            if out.exists() && !force {
                eprintln!(
                    "translate: refusing to overwrite existing {} (pass --force to override)",
                    out.display()
                );
                std::process::exit(1);
            }
            if let Err(e) = std::fs::write(out, ferx_text) {
                eprintln!("translate: failed to write {}: {}", out.display(), e);
                std::process::exit(1);
            }
            eprintln!("translate: wrote {}", out.display());
        }
        None => print!("{}", ferx_text),
    }
}

fn parse_o_flag(args: &[String]) -> Option<String> {
    args.iter()
        .position(|a| a == "-o" || a == "--output")
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn parse_output_flag(args: &[String]) -> Option<String> {
    let idx = args.iter().position(|a| a == "--output")?;
    match args.get(idx + 1) {
        Some(v) if !v.starts_with("--") => Some(v.clone()),
        _ => {
            eprintln!("Error: --output requires a path (e.g. --output run1.fitrx)");
            std::process::exit(1);
        }
    }
}

/// Parse the optional `--threads` flag. Returns `None` when the flag is
/// absent, when its value is `0`, or when its value is `auto` — all of which
/// mean "leave rayon's default pool alone". Exits the process on a missing
/// value or any other non-parseable input so typos don't silently fall
/// through to the default.
fn parse_threads_flag(args: &[String]) -> Option<usize> {
    let idx = args.iter().position(|a| a == "--threads")?;
    let value = args.get(idx + 1).unwrap_or_else(|| {
        eprintln!("Error: --threads requires a value (positive integer, 0, or 'auto')");
        std::process::exit(1);
    });
    if value.eq_ignore_ascii_case("auto") || value == "0" {
        return None;
    }
    match value.parse::<usize>() {
        Ok(n) if n > 0 => Some(n),
        _ => {
            eprintln!(
                "Error: --threads expects a positive integer, 0, or 'auto'; got '{}'",
                value
            );
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_output_flag, parse_threads_flag};

    fn args(extra: &[&str]) -> Vec<String> {
        std::iter::once("ferx")
            .chain(extra.iter().copied())
            .map(String::from)
            .collect()
    }

    #[test]
    fn absent_flag_is_none() {
        assert_eq!(parse_threads_flag(&args(&["model.ferx"])), None);
    }

    #[test]
    fn positive_integer_parses() {
        assert_eq!(parse_threads_flag(&args(&["--threads", "4"])), Some(4));
    }

    #[test]
    fn zero_means_default() {
        assert_eq!(parse_threads_flag(&args(&["--threads", "0"])), None);
    }

    #[test]
    fn auto_means_default() {
        assert_eq!(parse_threads_flag(&args(&["--threads", "auto"])), None);
        assert_eq!(parse_threads_flag(&args(&["--threads", "AUTO"])), None);
    }

    #[test]
    fn output_absent_is_none() {
        assert_eq!(parse_output_flag(&args(&["model.ferx"])), None);
    }

    #[test]
    fn output_returns_path() {
        assert_eq!(
            parse_output_flag(&args(&["--output", "run1.fitrx"])),
            Some("run1.fitrx".to_string())
        );
    }
}
