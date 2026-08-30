//! Model-development tooling built on top of [`ferx_core`].
//!
//! # Boundary rule
//!
//! > **`ferx-core`** = one model, one dataset, one fit.
//! > **`ferx-tools`** = many fits, resampling, model-space search.
//!
//! If it calls `fit()` more than once, it belongs here. Statistical kernels
//! (npde, CWRES, shrinkage, covariance/SE, SIR, FREM, VPC statistics) stay in
//! `ferx-core`; only the *orchestration* of repeated fits lives in this crate.
//!
//! The dependency is strictly one-way (A1): `ferx-core` never depends on
//! `ferx-tools`. Everything this crate needs must therefore be `pub` in
//! `ferx-core` — which means it is also reachable from the R wrapper, which
//! consumes `ferx-core` as an ordinary external crate (A4). An API that only
//! makes sense "because we're in the same workspace" is the wrong API, and
//! `api/ferx-core-public-api.txt` is the CI-enforced record of every item that
//! crossed the line (A2).
//!
//! This crate is currently a placeholder: the workspace mechanics and the API
//! gate land first (#1114 P1), the tools themselves after the prerequisite API
//! gaps G1 (thread-pool control) and G2 (quiet inner fits) are closed.

/// The `ferx-core` version this build of `ferx-tools` links against.
///
/// Deliberately routed through `ferx_core`'s *public* API rather than
/// `env!("CARGO_PKG_VERSION")`: it is the smallest thing that fails to compile
/// if the one-way dependency is ever broken or mis-wired, so the placeholder
/// crate still proves the link it exists to establish.
pub fn core_version() -> &'static str {
    ferx_core::build_info::BUILD_INFO.ferx_version
}

#[cfg(test)]
mod tests {
    use super::core_version;

    #[test]
    fn core_version_is_reported_through_the_public_api() {
        let v = core_version();
        assert!(!v.is_empty(), "ferx-core reported an empty version");
        // `x.y.z` — enough to catch a wire-up that returns some unrelated string.
        assert_eq!(
            v.split('.').count(),
            3,
            "unexpected ferx-core version shape: {v}"
        );
    }
}
