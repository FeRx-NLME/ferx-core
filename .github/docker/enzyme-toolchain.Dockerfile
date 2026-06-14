# syntax=docker/dockerfile:1.7
#
# Lean Enzyme-enabled Rust toolchain image for ferx-core CI  (issue #281)
# =======================================================================
#
# Why this exists
# ---------------
# ferx-core's *default* build uses Enzyme autodiff for inner-loop gradients,
# but every normal CI job builds `--features ci` (finite differences only) on
# plain nightly. So the autodiff path — and the `autodiff_fd_consistency`
# regression guard that protects the whole #278 bug class — are exercised
# NOWHERE automatically. Enzyme needs a custom rustc+LLVM built from source
# (no prebuilt rustup channel; the plugin-only shortcut hangs — see
# docs/src/installation.md), which is far too slow to build inside every run.
#
# This image bakes that toolchain ONCE. The per-run `autodiff.yml` job then
# just pulls it and compiles/tests ferx-core. The slow build runs on a
# GitHub-hosted runner via `enzyme-toolchain-image.yml`; nobody builds it
# locally. Published (public) to ghcr.io/ferx-nlme/enzyme-toolchain.
#
# Recipe adapted from ferx-r/Dockerfile (the project's proven Enzyme build),
# stripped of the R / RStudio layers so the CI image stays lean.

# ===========================================================================
# Stage 1: build a stage-1 rustc with Enzyme + its own matching LLVM.
# The ~20-30 GB build tree stays in this throwaway stage.
#   * download-ci-llvm=false is REQUIRED: Enzyme can't be built against CI LLVM.
#   * llvm.targets=X86 trims LLVM to the host backend (vs ferx-r, which builds
#     all targets) — cuts build time/size for an x86_64-only CI image. Drop it
#     if the build misbehaves.
# ===========================================================================
FROM ubuntu:24.04 AS enzyme-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build build-essential curl git ca-certificates \
        python3 pkg-config libssl-dev libzstd-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CARGO_HOME=/opt/cargo \
    RUSTUP_HOME=/opt/rustup \
    PATH=/opt/cargo/bin:/usr/local/bin:/usr/bin:/bin

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain nightly --profile minimal

RUN set -eux; \
    git clone --depth 1 https://github.com/rust-lang/rust /tmp/rust-src; \
    cd /tmp/rust-src; \
    ./configure \
        --release-channel=nightly \
        --enable-llvm-enzyme \
        --enable-llvm-link-shared \
        --enable-ninja \
        --disable-docs \
        --set llvm.download-ci-llvm=false \
        --set llvm.targets=X86 ; \
    ./x build --stage 1 library; \
    rustup toolchain link enzyme build/host/stage1; \
    rustc +enzyme --version --verbose

# ===========================================================================
# Stage 2: lean runtime = ubuntu + enzyme toolchain + cargo + ferx-core's
# C build deps (the `nlopt` crate builds NLopt from source via cmake; `lld`
# for the fat-LTO autodiff link).
# ===========================================================================
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build lld build-essential curl git ca-certificates \
        python3 pkg-config libssl-dev libzstd-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CARGO_HOME=/opt/cargo \
    RUSTUP_HOME=/opt/rustup \
    PATH=/opt/cargo/bin:/usr/local/bin:/usr/bin:/bin

# cargo + the rustup proxies come from a normal nightly; the Enzyme *rustc* is
# the linked `enzyme` toolchain copied from the builder stage. rust-src is
# pulled in because some autodiff builds want it.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain nightly --profile minimal \
    && rustup component add rust-src --toolchain nightly \
    && chmod -R a+rX /opt/cargo /opt/rustup

COPY --from=enzyme-builder /tmp/rust-src/build/host/stage1 /opt/enzyme-toolchain
RUN rustup toolchain link enzyme /opt/enzyme-toolchain \
    && rustup default enzyme \
    && chmod -R a+rX /opt/enzyme-toolchain \
    && rustc +enzyme --version --verbose

ENV RUSTUP_TOOLCHAIN=enzyme

# ── Notes ──────────────────────────────────────────────────────────────────
#  * cargo-in-toolchain: cargo comes from the nightly proxies installed above;
#    the `enzyme` toolchain supplies only the Enzyme rustc. This is the proven
#    ferx-r pattern (link `build/host/stage1` + `rustup default enzyme`). If a
#    future rust/rustup ever needs cargo inside the linked toolchain, either add
#    `./x build --stage 1 cargo` in stage 1 or set
#    `RUSTC=$(rustup which --toolchain enzyme rustc)` in the consumer job.
#  * Verification depth: `rustc +enzyme --version` only proves the toolchain
#    LOADS, not that differentiation WORKS (installation.md warns a
#    mismatched-LLVM Enzyme can pass --version yet hang on real autodiff). The
#    real functional check is ferx-core's `autodiff_fd_consistency` suite, run by
#    autodiff.yml against this image. A future hardening is a tiny inline
#    #[autodiff] smoke compile+run here to fail the monthly rebuild earlier.
