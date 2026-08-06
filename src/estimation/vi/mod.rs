//! Variational inference for NLME models.
//!
//! VI is an alternative way to **marginalize the random effects** — a peer of
//! FOCE/FOCEI, Laplace/AGQ, SAEM and IMP, not a technique specific to any one
//! model class. Instead of profiling `η` out at its conditional mode (FOCE) or
//! sampling it (SAEM), VI posits a tractable posterior `q_φᵢ(η)` per subject and
//! optimizes its parameters *jointly* with the population parameters.
//!
//! # The objective
//!
//! For each subject, maximize the evidence lower bound
//!
//! ```text
//! ELBOᵢ = E_{η∼qᵢ}[ log p(yᵢ | η, θ, Σ) + log p(η | Ω) − log qᵢ(η) ]  ≤  log p(yᵢ)
//! ```
//!
//! Because `p(η | Ω)` is Gaussian and `qᵢ` is Gaussian, this regroups into
//!
//! ```text
//! ELBOᵢ = E_{η∼qᵢ}[ log p(yᵢ | η, θ, Σ) ]  −  KL( qᵢ ‖ N(0, Ω) )
//!         └──── Monte Carlo (the only ────┘  └──── closed form ────┘
//!                 stochastic piece)
//! ```
//!
//! so only the data term needs sampling. That is a large variance reduction over
//! estimating the whole bound by Monte Carlo, and it is what lets the `Ω` update
//! be taken in closed form rather than by a noisy gradient step.
//!
//! # Reference and departures from it
//!
//! Janssen A, Bennis FC, Cnossen MH, Mathôt RAA. *Mixed effect estimation in deep
//! compartment models: variational methods outperform first-order
//! approximations.* J Pharmacokinet Pharmacodyn (2024) 51:797–808.
//! <https://doi.org/10.1007/s10928-024-09931-w>
//!
//! Two deliberate departures from their implementation:
//!
//! 1. They estimate the whole ELBO by Monte Carlo, including the `log p(η)` and
//!    `log q` terms. We take the KL analytically (above).
//! 2. Consequently `Ω` need not be stepped by the optimizer at all — under the
//!    analytic KL its ELBO-maximizing value is exactly `(1/N)Σᵢ(Sᵢ + μᵢμᵢᵀ)`.
//!    Their Fig. 3 shows `Ω` fluctuating badly during optimization; this removes
//!    that failure mode by construction.
//!
//! What we keep from them: the full-rank Gaussian family, and the path-derivative
//! ("sticking the landing", Roeder et al. 2017) gradient estimator for the
//! fallback Monte-Carlo-KL path.
//!
//! # Status
//!
//! Building bottom-up. Present: the variational families ([`family`]) and the
//! stochastic optimizer ([`adam`]). Still to come: ELBO assembly, `run_vi`, and
//! the `EstimationMethod::Vi` wiring.

pub mod adam;
pub mod elbo;
pub mod family;
pub mod run;

pub use adam::{AdamConfig, AdamState, PolyakAverager};
pub use elbo::{
    analytic_eta_grad_available, closed_form_omega, population_neg_elbo,
    unsupported_data_term_reason, ElboConfig, ElboEval, EtaGradMode, PackedLayout,
};
pub use family::{FullRank, KlTerm, MeanField, VariationalFamily};
pub use run::run_vi;
