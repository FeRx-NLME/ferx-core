#![cfg_attr(feature = "autodiff", feature(autodiff))]

// Require autodiff unless the `ci` feature is set (for CI without Enzyme toolchain).
#[cfg(all(not(feature = "autodiff"), not(feature = "ci")))]
compile_error!(
    "The `autodiff` feature is required. Finite-difference gradients are not supported. \
     Build with: cargo build --release (autodiff is now the default feature), \
     or explicitly: cargo build --release --features autodiff. \
     Requires the Enzyme Rust toolchain: rustup toolchain install enzyme"
);

pub mod ad;
pub mod api;
pub mod build_info;
pub mod cancel;
pub mod diagnostics;
pub mod estimation;
pub mod io;
#[cfg(feature = "nn")]
pub mod nn;
pub mod ode;
pub mod parser;
pub mod pk;
pub mod stats;
pub mod suggest_start;
pub mod types;

pub use api::{
    check_model_data, check_model_data_warnings, check_model_options, fit, fit_from_files, predict,
    run_from_file, run_model_simulate, run_model_with_data, run_model_with_data_inits, simulate,
    simulate_with_seed, simulate_with_uncertainty, validate_model_file, SimulateUncertaintyOptions,
};
pub use cancel::CancelFlag;
pub use diagnostics::{CheckReport, Diagnostic, Severity};
pub use estimation::run_sir::run_sir;
pub use estimation::uncertainty_samples::UncertaintyMethod;
pub use io::datareader::{read_nonmem_csv, read_nonmem_csv_with_covariates};
pub use parser::model_parser::{parse_full_model_file, parse_model_file, parse_model_string};
pub use suggest_start::{inits_from_nca, NcaInit, SuggestedStart};
pub use types::*;
