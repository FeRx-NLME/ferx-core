# Rust API

ferx-core can be used as a Rust library for embedding population PK estimation in your own applications.

## Adding as a Dependency

```toml
[dependencies]
ferx-core = { path = "../ferx-core" }
```

## Quick Example

```rust
use ferx_core::*;
use std::path::Path;

fn main() -> Result<(), String> {
    // Parse model and data
    let parsed = parse_full_model_file(Path::new("model.ferx"))?;
    let population = read_nonmem_csv(Path::new("data.csv"), None)?;

    // Build initial parameters and options
    let (init_params, options) = build_fit_inputs(&parsed)?;

    // Run estimation
    let result = fit(&parsed.model, &population, &init_params, &options)?;

    // Access results
    println!("OFV: {}", result.ofv);
    for (name, val) in result.theta_names.iter().zip(result.theta.iter()) {
        println!("  {} = {:.6}", name, val);
    }

    Ok(())
}
```

## API Sections

- [Core Types](types.md) -- `CompiledModel`, `Population`, `FitResult`, `FitOptions`, etc.
- [Fitting Functions](fitting.md) -- `fit()`, `fit_from_files()`, `run_model_with_data()`
- [Simulation](simulation.md) -- `simulate()`, `simulate_with_seed()`, `predict()`
- [Parsing](parsing.md) -- `parse_model_file()`, `parse_full_model_file()`, `read_nonmem_csv()`
