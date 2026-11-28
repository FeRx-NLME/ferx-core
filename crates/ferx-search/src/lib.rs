// SPDX-License-Identifier: LGPL-3.0-or-later
//! Model-space search built on top of [`ferx_core`].
//!
//! # Boundary rule
//!
//! > **`ferx-core`** = one model, one dataset, one fit.
//! > **`ferx-tools`** = many fits of a given model: resampling and screening.
//! > **`ferx-search`** = many fits over a space of models.
//!
//! [`search`] holds the shared layer (the `.ferxsearch` configuration, the MFL
//! parser, the candidate runner and its journal); every other module is one
//! search tool on top of it, and [`amd`] runs them in order.
//!
//! The dependency is strictly one-way (#3185): `ferx-search` → `ferx-tools` →
//! `ferx-core`. From `ferx-tools` this crate uses only
//! [`ferx_tools::lanes::run_in_lanes`], and neither crate below may depend on
//! this one. `tests/public_api_boundary.rs` in `ferx-core` pins it.
//!
//! # License
//!
//! LGPL-3.0-or-later (#3183), unlike the MIT crates below it; see the crate's
//! `NOTICE`, `COPYING.LESSER` and `COPYING`. Releases of this code inside
//! `ferx-tools`, up to and including 0.4.0, remain MIT.

pub mod allometry;
pub mod amd;
pub mod covsearch;
pub mod globalsearch;
pub mod iivsearch;
pub mod iovsearch;
pub mod modelsearch;
pub mod ruvsearch;
pub mod search;

#[cfg(test)]
mod tests {
    /// #529, trait-level half for `ferx-search`: this crate's options types
    /// really do resolve `Default`, so a caller can build them with
    /// `..Default::default()`.
    ///
    /// A **named spot check, not the inventory guard**: see the test of the same
    /// name in `ferx-tools`, and `every_public_options_type_implements_default`
    /// in `ferx-core`'s `tests/public_api_boundary.rs`, which scans all of
    /// `crates/`.
    #[test]
    fn every_public_options_struct_implements_default() {
        fn assert_default<T: Default>() -> T {
            T::default()
        }
        let _ = assert_default::<crate::search::RunOptions>();
        let _ = assert_default::<crate::modelsearch::ModelsearchOptions>();
        let _ = assert_default::<crate::ruvsearch::RuvsearchOptions>();
        let _ = assert_default::<crate::iivsearch::IivsearchOptions>();
        let _ = assert_default::<crate::iovsearch::IovsearchOptions>();
        let _ = assert_default::<crate::amd::AmdOptions>();
        let _ = assert_default::<crate::globalsearch::GlobalsearchOptions>();
    }
}
