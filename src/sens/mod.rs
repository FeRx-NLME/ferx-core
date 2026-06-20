//! Analytic PK-parameter sensitivities for the closed-form (analytical) PK
//! models — the clean-slate replacement for Enzyme AD / FD-of-Jacobian in the
//! FOCEI outer gradient (issue #367).
//!
//! Goal: provide `f`, `∂f/∂pk`, `∂²f/∂pk²` for each built-in PK solution exactly
//! and noise-free, so the gradient assembly can chain them to `∂f/∂η`, `∂f/∂θ`,
//! `∂²f/∂η²`, `∂²f/∂η∂θ` via closed-form chain rule (η enters only as
//! `pk = tv·exp(sel·η)`; θ through `tv_fn`).
//!
//! Two implementation styles ship together and are kept in lockstep (a per-kernel
//! parity test cross-checks them for every PK model / dose kind):
//!   * hand-written closed-form derivatives (`*_explicit`) — the default fast path
//!     the provider selects when a kernel covers every dose; and
//!   * the same solution evaluated over the [`dual2::Dual2`] forward-2nd-order
//!     dual number — the general path, used for lagtime / oral-infusion / `F`-on-IV
//!     and any dose the explicit kernels don't cover.

pub mod dual1;
pub mod dual2;
pub mod jet;
pub mod num;
pub mod ode_provider;
pub mod one_cpt;
pub mod one_cpt_explicit;
pub mod propagate;
pub mod provider;
pub mod three_cpt;
pub mod three_cpt_explicit;
pub mod two_cpt;
pub mod two_cpt_explicit;
