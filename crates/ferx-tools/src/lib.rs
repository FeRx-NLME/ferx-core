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
//! First inhabitant: [`bootstrap`] (#1140), the non-parametric case bootstrap
//! at `PsN::bootstrap` feature parity. It is the boundary rule made concrete —
//! 200+ calls to [`ferx_core::fit`] over resampled data, and no numerics of its
//! own.

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

    /// #529 convention guard, `ferx-tools` half: every `pub` options struct
    /// this crate exposes must implement `Default` so a caller can build it
    /// with `..Default::default()` and survive an added field. A new options
    /// struct that forgets it fails to compile here.
    #[test]
    fn every_public_options_struct_implements_default() {
        fn assert_default<T: Default>() -> T {
            T::default()
        }
        let _ = assert_default::<crate::bootstrap::BootstrapOptions>();
        let _ = assert_default::<crate::gam::GamOptions>();
        let _ = assert_default::<crate::search::RunOptions>();
    }

    #[test]
    fn core_version_is_reported_through_the_public_api() {
        let v = core_version();
        assert!(!v.is_empty(), "ferx-core reported an empty version");
        // A leading numeric major component is enough to catch a wire-up that
        // returns some unrelated string. Deliberately NOT `split('.').count() == 3`:
        // a SemVer pre-release or build metadata (`0.4.0-rc.1`) yields four
        // components and would fail for a reason unrelated to what this guards.
        let major = v.split('.').next().unwrap_or("");
        assert!(
            !major.is_empty() && major.bytes().all(|b| b.is_ascii_digit()),
            "unexpected ferx-core version shape: {v}"
        );
    }
}

pub mod bootstrap;
pub mod gam;
pub mod search;
