# Installation

ferx-core builds with a recent **nightly** Rust toolchain (pinned in
`rust-toolchain.toml`) — `cargo build` is all you need. There is no longer any
custom compiler or autodiff toolchain to set up: the exact FOCE/FOCEI and HMC
gradients now come from hand-rolled analytic `Dual2` sensitivities, so the old
Enzyme/`--features autodiff` path has been **retired**.

> **Upgrading from an older setup?** If you previously built a custom `enzyme`
> nightly from source, you can remove it (`rustup toolchain uninstall enzyme`) —
> it is no longer used. A stock nightly suffices.

Pick your platform:

- [**Linux**](#linux) — fully supported
- [**macOS**](#macos) — supported (Intel and Apple Silicon)
- [**Windows**](#windows) — supported (native, or via WSL2)

---

## Prerequisites

### Linux

Tested on Ubuntu 22.04/24.04; other distributions work with package-manager adjustments.

```bash
# rustup + nightly (provides cargo):
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
rustup toolchain install nightly --profile minimal

# Build tools + the NLopt C library (the only non-Rust dependency):
sudo apt update
sudo apt install -y build-essential pkg-config libnlopt-dev
```

**Do not use snap's rustup** — its filesystem confinement breaks on non-standard
home directories (common on enterprise servers). Use the rustup.rs installer above.

### macOS

Supported on Intel and Apple Silicon.

```bash
xcode-select --install          # Xcode Command Line Tools (clang, git)
brew install nlopt

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
rustup toolchain install nightly --profile minimal
```

### Windows

ferx-core builds natively on Windows with the MSVC toolchain. Install
[rustup](https://rustup.rs) and a nightly toolchain (`rustup toolchain install
nightly`); the NLopt crate builds via CMake (install the
[Build Tools for Visual Studio](https://visualstudio.microsoft.com/downloads/) if
the link step complains). WSL2 with the [Linux](#linux) instructions also works
and is convenient if you already use a Unix workflow.

---

## Building ferx-core from source

```bash
git clone https://github.com/FeRx-NLME/ferx-core
cd ferx-core

cargo build --release
# Binary at target/release/ferx
```

The `release` profile uses `lto = "fat"`, so the first build's final link step is
single-threaded and takes a few minutes; Cargo caches dependencies, so later
builds are fast. Extra cores do not speed up the final LTO link — that is normal,
not a hang.

### Build options

```bash
# Debug build (faster compile, slower runtime)
cargo build

# Quick type-check without building
cargo check

# Lints
cargo clippy

# CI configuration (release-level opt, thin LTO — what CI uses)
cargo build --release --no-default-features --features ci
```

The `ci` feature is a no-op kept for compatibility with existing
`--features ci` invocations; it does not change behaviour.

### Verify the build

```bash
cargo run --release --bin ferx -- examples/warfarin.ferx --data data/warfarin.csv
```

This should print a successful model fit with parameter estimates.

---

## Installing the ferx R package

The R package handles the Rust build for you:

```r
pak::pak("FeRx-NLME/ferx-r")
# or: devtools::install_github("FeRx-NLME/ferx-r")
```

See the [ferx R package README](https://github.com/FeRx-NLME/ferx-r) for API usage.

### Cancelling a running fit (Ctrl-C)

`ferx_fit()` runs the estimator on a worker thread and polls for R interrupts on
the main thread every ~100 ms, so **Ctrl-C** (or RStudio's red stop button) aborts
the fit cleanly. The worker exits at the next safe checkpoint — typically within a
second or two, but up to one inner-loop evaluation for heavy ODE models. The call
then returns with an R error:

```
Error: ferx_fit: cancelled by user
```

Ctrl-Z (SIGTSTP) will *not* abort the fit — it suspends the whole R process to the
shell. Use Ctrl-C instead.

---

## Dependencies

ferx-core depends on these crates (managed automatically by Cargo):

| Crate | Purpose |
|-------|---------|
| `nalgebra` | Linear algebra (matrices, Cholesky) |
| `nlopt` | Nonlinear optimization (SLSQP, L-BFGS, MMA) |
| `rayon` | Parallel computation |
| `rand`, `rand_distr` | Random number generation (SAEM, SIR) |
| `csv` | CSV data file reading |
| `regex` | Model file expression parsing |

The `nlopt` crate requires the NLopt C library. Most platforms handle this
automatically; if the build fails on NLopt, install it via your system package
manager:

```bash
# macOS
brew install nlopt

# Ubuntu/Debian
sudo apt install libnlopt-dev

# Fedora
sudo dnf install NLopt-devel
```

---

## Troubleshooting

### `"error: the option 'Z' is only accepted on the nightly compiler"`
Your shell (or R) is finding a non-nightly `rustc`. Check `rustc --version` and
your `PATH`. For R, verify `Sys.which("rustc")` and `~/.Renviron`. ferx-core
builds on the nightly toolchain pinned in `rust-toolchain.toml`.

### The build fails on NLopt
Install the NLopt C library for your platform (see [Dependencies](#dependencies)).

### `gradient = ad` errors with `E_AD_RETIRED`
The Enzyme autodiff path was retired. Use `gradient = auto` (the exact analytic
gradient where it is in scope, finite differences otherwise) or `gradient = fd`.
