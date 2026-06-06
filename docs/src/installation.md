# Installation

ferx-core requires a nightly Rust toolchain with Enzyme for automatic differentiation. The **supported way to get one is to build `rustc` from source with Enzyme enabled**, so that `rustc`, LLVM, and Enzyme are all compiled from the same tree and are guaranteed version-compatible. This is also how the [ferx-r Docker image](https://github.com/ferx-core-NLME/ferx-r) builds its toolchain.

> ## ⚠️ Do not build a standalone Enzyme *plugin* against a separately-installed LLVM
>
> An older approach — `rustup`-install a prebuilt nightly, build the Enzyme plugin (`libEnzyme-<N>.so`/`.dylib`) against a Homebrew/apt LLVM, and copy it into the nightly's sysroot — **does not work**. The plugin's LLVM and the LLVM that the prebuilt `rustc` was compiled against are different builds (even at the same major version), and the mismatch makes Enzyme **hang indefinitely in `llvm::Constant::getNullValue`** the moment it differentiates *anything* — even a trivial scalar function, in both forward and reverse mode.
>
> Worse, it fails silently: the usual `rustc +enzyme -Z autodiff=Enable - </dev/null` check still prints `error[E0601]: main function not found` and looks like success, because that check never actually runs differentiation. The only reliable verification is to **compile and run a real `#[autodiff_forward]`/`#[autodiff_reverse]` function** (see [Verify](#4-verify-the-toolchain)).
>
> If you previously set up the toolchain this way, remove it (`rustup toolchain uninstall enzyme`) and rebuild from source as below.

## Two separate one-time costs

Autodiff setup has **two distinct one-time steps**. Don't conflate them — they happen at different times and only the first is documented on this page:

| Step | What it is | Roughly | When |
|---|---|---|---|
| **1. Build the Enzyme toolchain** | Compile a custom nightly `rustc` + LLVM with Enzyme from source (**this page**) | ~45–60 min | **Once per machine** — reused forever. |
| **2. Compile ferx / ferx-core** | Build the crates and link them | **CPU-dependent**: ~15–30 min on a fast machine, up to ~1–2 h on an older laptop (autodiff); faster for FD | **First build only** — Cargo caches dependencies, so later builds take minutes. |

Step 2 is slower on the autodiff path because it uses **fat LTO** (required for cross-crate Enzyme correctness; see [Building ferx-core](#building-ferx-core-from-source) below). The finite-difference build skips step 1 entirely and uses thin LTO in step 2, so it links noticeably faster.

---

Pick your platform:

- [**Linux**](#linux) — fully supported
- [**macOS**](#macos) — supported (Intel and Apple Silicon)
- [**Windows**](#windows) — **native Enzyme autodiff not supported** (use [WSL2](#windows) or Docker; the ferx R package runs natively with finite-difference gradients)

The **step 1** toolchain build takes **~45–60 min** and needs **~30 GB** of free disk (it compiles LLVM and a stage-1 `rustc`). It's a one-time cost.

---

## Linux

Tested on Ubuntu 22.04/24.04. Other distributions work with package-manager adjustments.

### 1. Install build dependencies + rustup nightly (for `cargo`)

**Do not use snap's rustup** — its filesystem confinement breaks on non-standard home directories (common on enterprise servers).

```bash
# Remove snap rustup if you had it:
sudo snap remove rustup 2>/dev/null || true

# Build deps for compiling rustc + LLVM from source:
sudo apt update
sudo apt install -y cmake ninja-build clang g++ python3 \
                    build-essential curl git pkg-config libssl-dev libzstd-dev

# rustup + nightly — provides cargo and lets us `rustup toolchain link` later:
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
rustup toolchain install nightly --profile minimal
```

You do **not** install a separate LLVM — the source build compiles its own (this is the whole point: matching LLVM and Enzyme).

Now jump to [Build the Enzyme toolchain from source](#3-build-the-enzyme-toolchain-from-source).

---

## macOS

Supported on Intel and Apple Silicon.

### 1. Install build dependencies + rustup nightly (for `cargo`)

```bash
xcode-select --install          # Xcode Command Line Tools (clang, git)
brew install cmake ninja python3

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
rustup toolchain install nightly --profile minimal
```

No separate Homebrew LLVM is needed (or wanted) — the source build compiles its own matching LLVM. Continue below.

---

## 3. Build the Enzyme toolchain from source

Identical on Linux and macOS. This compiles a stage-1 `rustc` with Enzyme integrated into its own LLVM.

```bash
git clone --depth 1 https://github.com/rust-lang/rust /tmp/rust-src
cd /tmp/rust-src

./configure \
    --release-channel=nightly \
    --enable-llvm-enzyme \
    --enable-llvm-link-shared \
    --enable-ninja \
    --disable-docs \
    --set llvm.download-ci-llvm=false   # REQUIRED: Enzyme cannot build against CI LLVM

./x build --stage 1 library         # ~45–60 min; builds LLVM + stage-1 rustc + std

# Register the freshly-built toolchain under the name ferx pins to:
rustup toolchain link enzyme build/host/stage1
rustc +enzyme --version --verbose
```

The flags: `--enable-llvm-enzyme` builds and links the Enzyme plugin into rustc's own LLVM; `--enable-llvm-link-shared` builds LLVM as a shared library (required for the Enzyme plugin to attach). `--set llvm.download-ci-llvm=false` is the critical one: by default the Rust build downloads a prebuilt "CI" LLVM, which Enzyme cannot attach to. Forcing a from-source LLVM is what makes the toolchain actually work.

> The stage-1 toolchain bundles only `rustc` — no `cargo`, `rustfmt`, or `clippy`. That's fine for building: `cargo` comes from the `nightly` you installed in step 1 and rustup falls back to it automatically (a `"cargo is unavailable for the active toolchain"` *info* line is harmless). But because `rustfmt`/`clippy` are absent, run those on nightly explicitly — `cargo +nightly fmt`, `cargo +nightly clippy --no-default-features --features ci`. In particular the repo's pre-commit hook runs `cargo fmt` and will **fail under the `enzyme` pin** unless you format on nightly first (or commit with `--no-verify` for non-Rust changes).

## 4. Verify the toolchain

**Do not rely on the `E0601` check** — it passes even for a broken toolchain. Instead compile and run a real `#[autodiff_forward]` function (as below). This is the only check that detects the silent `getNullValue` hang:

```bash
cat > /tmp/ad_check.rs <<'EOF'
#![feature(autodiff)]
use std::autodiff::autodiff_forward;

// d/dx of f(x,y) = exp(x*y) + x*x  is  y*exp(x*y) + 2x.
#[autodiff_forward(df, Dual, Const, Dual)]
#[inline(never)]
fn f(x: f64, y: f64) -> f64 { (x * y).exp() + x * x }

fn main() {
    let (val, dfdx) = df(1.5, 1.0, 2.0);          // x=1.5, dx=1, y=2
    let expect = 2.0 * (1.5 * 2.0_f64).exp() + 2.0 * 1.5;
    assert!((dfdx - expect).abs() < 1e-9);
    println!("autodiff OK: f={val:.4}, df/dx={dfdx:.4}");
}
EOF

rustc +enzyme -Zautodiff=Enable -Clto=fat -Copt-level=2 /tmp/ad_check.rs -o /tmp/ad_check
/tmp/ad_check
```

Expected: the compile finishes **in a few seconds** and prints `autodiff OK: f=22.3355, df/dx=43.1711`. If `rustc` instead **hangs** (pins a core at 100% CPU and never returns), your toolchain has the LLVM/Enzyme mismatch described in the warning at the top — rebuild from source with `--set llvm.download-ci-llvm=false`.

> **The ferx R package runs this check for you.** Installing [ferx-r](https://github.com/FeRx-NLME/ferx-r) on the autodiff path runs the same compile-and-run self-test automatically (under a timeout) before the long fat-LTO build, and aborts with a pointer back here if the toolchain hangs — so a broken toolchain fails fast instead of wedging `rustc`. You only need to run the check above by hand when setting the toolchain up outside the R package (e.g. for the `ferx` CLI). Set `FERX_AD_PREFLIGHT_SKIP=1` to skip it once verified.

### Make the toolchain permanent

The clone lives in `/tmp/rust-src`, and `rustup toolchain link enzyme build/host/stage1` points the `enzyme` toolchain *into* that tree. macOS periodically purges `/tmp`, which would silently break the toolchain. Relocate the stage-1 build (~640 MB once pruned) to a permanent path and re-link:

```bash
# Copy, dereferencing the LLVM/Enzyme dylib symlinks into real files (-L):
cp -aL /tmp/rust-src/build/host/stage1 ~/.local/share/enzyme-toolchain

# cp prints a few "directory causes a cycle" warnings for the bundled rust
# source tree — harmless. Prune it (it's only for IDE source browsing, not
# building) so the copy stays ~640 MB instead of ballooning to ~30 GB:
rm -rf ~/.local/share/enzyme-toolchain/lib/rustlib/rustc-src \
       ~/.local/share/enzyme-toolchain/lib/rustlib/src

rustup toolchain uninstall enzyme
rustup toolchain link enzyme ~/.local/share/enzyme-toolchain

# Re-run the verification above against the new path, then reclaim ~8 GB:
rm -rf /tmp/rust-src
```

The `-L` is required: the dylibs inside `stage1` are symlinks into the build tree, so plain `cp -a` would leave them dangling once `/tmp/rust-src` is deleted. The prune is required because `-L` otherwise dereferences the source-tree symlinks (which also contain a build-dir cycle) and bloats the copy to ~30 GB.

### Multi-user servers

Stage the built toolchain in `/opt/enzyme-toolchain` so all users share one build (same `cp -aL` + prune as above — `-L` to materialize the dylib symlinks, prune to avoid the ~30 GB source-tree blowup):

```bash
sudo cp -aL /tmp/rust-src/build/host/stage1 /opt/enzyme-toolchain
sudo rm -rf /opt/enzyme-toolchain/lib/rustlib/rustc-src /opt/enzyme-toolchain/lib/rustlib/src
sudo chown -R root:root /opt/enzyme-toolchain
sudo chmod -R a+rX /opt/enzyme-toolchain
```

Each user links it into their own rustup once:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y   # for cargo
rustup toolchain link enzyme /opt/enzyme-toolchain
```

For R users, add to `~/.Renviron`:
```
PATH=/opt/enzyme-toolchain/bin:${HOME}/.cargo/bin:${PATH}
RUSTUP_TOOLCHAIN=enzyme
```

---

## Windows

**Building the Enzyme toolchain from source is not currently supported on Windows.** The Enzyme-enabled Rust source build is exercised only on Linux and macOS, the `-Z autodiff=Enable` path is untested on `x86_64-pc-windows-msvc`, and there's no Windows CI to back a green signal.

Note that the [ferx R package](https://github.com/FeRx-NLME/ferx-r) *does* install and run natively on Windows using finite-difference gradients (no Enzyme). Only native Enzyme autodiff is unavailable - for that, use one of the workarounds below.

### Workarounds for Windows users

- **WSL2 (recommended)**: install Ubuntu under WSL2 and follow the [Linux instructions](#linux). Performance is near-native for CPU-bound model fitting.
- **Docker**: use the ferx-r Docker image, which ships the toolchain prebuilt.
- **Remote dev**: SSH into a Linux server or cloud VM.

Once upstream Enzyme integration is distributed via rustup (tracked at [rust-lang/rust#124509](https://github.com/rust-lang/rust/issues/124509)), Windows support should become straightforward. No ETA; the feature is still experimental in rustc.

---

## Building ferx-core from source

Once your Enzyme toolchain is set up and **verified**:

```bash
git clone https://github.com/ferx-core-NLME/ferx-core
cd ferx-core

RUSTFLAGS="-Z autodiff=Enable" cargo build --release --features autodiff
# Binary at target/release/ferx
```

> **Expect a long build, scaling with single-core CPU speed.** A
> `--release --features autodiff` build is the slowest configuration. Because the
> dominant step is single-threaded (see below), wall-clock time tracks your
> per-core speed almost linearly: **~15–30 min on a fast modern CPU (recent
> desktop or native Apple Silicon arm64), up to ~1–2 h on an older / mobile
> laptop CPU** (which may also thermal-throttle under sustained single-core load).
> Extra cores do **not** help. Two costs compound:
> - The release profile uses `lto = "fat"`, which re-optimizes the whole dependency
>   graph in one final, largely **single-threaded** link/codegen step. Your other CPU
>   cores will look mostly idle during it — that's normal, not a hang.
> - Enzyme runs its differentiation passes during LLVM codegen on top of that.
>
> Fat LTO is *required* for cross-crate Enzyme correctness, so it can't be dialed down
> for an autodiff release build. Any source change re-triggers this final step, so the
> wait recurs on every rebuild. To check it's progressing rather than stuck, confirm a
> `rustc --crate-name ferx` process is running (`ps aux | grep rustc`); the final LTO
> step shows no per-crate `.rlib` churn because all dependencies are already compiled.
>
> A build that pins a core at 100% CPU **forever** (sampling shows `rustc` stuck in
> `getNullValue` inside `CreateForwardDiff`/`CreatePrimalAndGradient`) is not slow — it's
> the broken-toolchain symptom from the warning at the top of this page. Re-run the
> [autodiff verification](#4-verify-the-toolchain): if the tiny scalar check also hangs,
> the problem is your toolchain, not ferx-core. For faster iteration, use the debug or
> `--features ci` builds under "Build options" below.

> **`RUSTFLAGS="-Z autodiff=Enable"` is required for every `--features autodiff` build.**
> The `autodiff` feature won't compile without it — you'll get
> `error: using the autodiff feature requires -Z autodiff=Enable`.
>
> Pass it inline as shown (it's the form CI and the R package use), or export it for
> the current shell with `export RUSTFLAGS="-Z autodiff=Enable"`.
>
> Do **not** bake it into a committed `.cargo/config.toml`: that flag is nightly-only
> and applies to *all* builds, which breaks the stable toolchain and the no-Enzyme
> `--features ci` path (including CI). If you want it persistent on your own machine,
> keep it in a *git-ignored* local `.cargo/config.toml` or a shell/`direnv` export.

### Build options

```bash
# Debug build (faster compile, slower runtime)
RUSTFLAGS="-Z autodiff=Enable" cargo build --features autodiff

# Quick type-check without building
RUSTFLAGS="-Z autodiff=Enable" cargo check --features autodiff

# Lints
RUSTFLAGS="-Z autodiff=Enable" cargo clippy --features autodiff

# CI build without autodiff (uses finite differences — no Enzyme, no RUSTFLAGS needed)
cargo build --release --no-default-features --features ci
```

The `ci` feature is the right choice for development on machines without the full Enzyme toolchain — at the cost of finite-difference (rather than exact) gradients. It builds on plain nightly and does not need the `RUSTFLAGS` flag. This is also the configuration ferx-core's CI uses.

### Verify the build

```bash
RUSTFLAGS="-Z autodiff=Enable" cargo run --release --features autodiff --bin ferx -- examples/warfarin.ferx --simulate
```

Should print a successful model fit with parameter estimates.

---

## Installing the ferx R package

The R package handles all of the above automatically — you do **not** set `RUSTFLAGS` yourself. Just:

```r
devtools::install_github("ferx-core-NLME/ferx-r")
```

The package's build is driven by its `src/Makevars`, which:

1. **Probes for the `enzyme` toolchain** with `rustup toolchain list | grep -q '^enzyme'`.
2. **If Enzyme is present**, it builds the autodiff path: it writes `rustflags = ["-Z", "autodiff=Enable"]` into a generated `rust/.cargo/config.toml` and pins `channel = "enzyme"` in a generated `rust/rust-toolchain.toml` (both build artifacts, not committed), so the flag is supplied for you.
3. **If Enzyme is missing**, it silently falls back to the stable path (`--no-default-features --features ci,nn`, thin LTO) so the install still succeeds — with finite-difference gradients instead of autodiff. The package prints a notice on load when built this way.

You can override the probe with an environment variable before installing:

- `FERX_NO_AUTODIFF=1` — force the stable (no-autodiff) build even if Enzyme is installed.
- `FERX_NO_AUTODIFF=0` — force the autodiff build, erroring out if the `enzyme` toolchain isn't found.

To confirm which path your installed package took, call `ferx:::ferx_rust_autodiff_enabled()` in R.

This is why a committed `.cargo/config.toml` is discouraged for the core repo: the R package's `Makevars` already manages the flag conditionally per machine, generating the config only when Enzyme is actually available. A committed flag in `ferx-core` would conflict with that logic and break the no-Enzyme install.

See the [ferx R package README](https://github.com/ferx-core-NLME/ferx-r) for API usage.

### Cancelling a running fit (Ctrl-C)

`ferx_fit()` runs the estimator on a worker thread and polls for R interrupts on the main thread every ~100 ms, so **Ctrl-C** (or RStudio's red stop button) aborts the fit cleanly. The worker exits at the next safe checkpoint — typically within a second or two, but up to one inner-loop evaluation for heavy ODE models. The call then returns with an R error:

```
Error: ferx_fit: cancelled by user
```

Ctrl-Z (SIGTSTP) will *not* abort the fit — it suspends the whole R process to the shell. Use Ctrl-C instead.

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

The `nlopt` crate requires the NLopt C library. Most platforms handle this automatically; if the build fails on NLopt, install it via your system package manager:
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

### `rustc` hangs forever during an `--features autodiff` build (or during the verify step)
The classic symptom of a **mismatched LLVM/Enzyme toolchain**: one core at 100% CPU, no progress, and a profiler (`sample <pid>`) shows `rustc` stuck in `llvm::Constant::getNullValue` under `EnzymeLogic::CreateForwardDiff` (forward mode) or `CreatePrimalAndGradient`/`ZeroMemory` (reverse mode). This happens when the `enzyme` toolchain was assembled by copying a separately-built Enzyme plugin into a prebuilt nightly's sysroot. **Fix:** rebuild `rustc` from source with `--enable-llvm-enzyme` and `--set llvm.download-ci-llvm=false` (see [step 3](#3-build-the-enzyme-toolchain-from-source)), then confirm with the [real autodiff verification](#4-verify-the-toolchain).

### `"error: the option 'Z' is only accepted on the nightly compiler"`
Your shell (or R) is finding a non-nightly `rustc`. Check `rustc --version` and your `PATH`. For R, verify `Sys.which("rustc")` and `~/.Renviron`.

### `"custom toolchain 'enzyme' specified in override file ... is not installed"`
Run `rustup toolchain link enzyme <path-to-stage1>` (e.g. `build/host/stage1`, or `/opt/enzyme-toolchain` on shared installs).

### `"not a directory: '/<path>/lib'"` from `rustup toolchain link`
Permission issue — the user running `toolchain link` can't read the target path. On shared installs, run `sudo chmod -R a+rX /opt/enzyme-toolchain`.

### `./x build` fails downloading or attaching LLVM
Make sure `--set llvm.download-ci-llvm=false` is in your `./configure` line. Enzyme cannot attach to the prebuilt CI LLVM; the build must compile LLVM from source.

### `"incorrect value 'X' for unstable option 'autodiff'"`
Valid autodiff values change between nightly builds. Test with `rustc +enzyme -Zautodiff=Enable ...`; if `Enable` is rejected, try `LooseTypes`, `Inline`, or `PrintTA`.

### `"Enzyme: cannot handle (forward) unknown intrinsic llvm.maximumnum"`
Recent rustc lowers `f64::max()`/`f64::min()` to intrinsics Enzyme can't yet differentiate. This is a ferx-core code-level issue — AD-instrumented functions must use manual `if` expressions. Should not happen on released ferx-core versions; if it does, please file an issue.

### `"cargo is unavailable for the active toolchain"` (info, not error)
The stage-1 toolchain has no bundled `cargo`; rustup falls back to nightly's `cargo`, which works. Ignore it, or `cp ~/.rustup/toolchains/nightly-*/bin/cargo /opt/enzyme-toolchain/bin/`.

### Refreshing when upstream nightly rolls forward
If a new ferx-core release references stdlib items your toolchain doesn't have (e.g. `autodiff_forward` not found), re-pull `rust-lang/rust` and rebuild the stage-1 toolchain from source.
