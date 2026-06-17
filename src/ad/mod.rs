#[cfg(feature = "autodiff")]
pub mod ad_gradients;
#[cfg(feature = "autodiff")]
pub mod dual;
#[cfg(feature = "autodiff")]
pub mod event_driven_ad;
#[cfg(feature = "autodiff")]
pub mod event_driven_ad_jac;

#[cfg(feature = "autodiff")]
pub use dual::Dual;
