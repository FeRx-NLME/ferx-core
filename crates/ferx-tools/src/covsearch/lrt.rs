//! The likelihood-ratio test an SCM step decides on (#1180).
//!
//! Two nested models, the *reduced* one with fewer free parameters, and
//! `ΔOFV = OFV_reduced − OFV_extended`. Under the null that the extra
//! parameters are zero, `ΔOFV ~ χ²(df)` with `df` the number of extra free
//! parameters, so the p-value is the upper tail `P(χ²_df ≥ ΔOFV)`. This is
//! Pharmpy's `modeling.lrt` and PsN's `gof_pval`, and the two agree on every
//! case that matters: `df ≥ 1`, a finite ΔOFV, the tail from the regularized
//! incomplete gamma function.
//!
//! What `df` is: the difference in [`FitResult::n_parameters`] — every *free*
//! θ, Ω and σ — between the two fits. That is Pharmpy's
//! `len(model.parameters.nonfixed)` difference and PsN's `nthetas` difference
//! (PsN counts θ only, but an SCM step never touches Ω or σ so the two are the
//! same number). Read off the fits rather than the edit so a relation whose θ
//! the file `FIX`es, or a categorical whose level count only the data knows,
//! is counted as it was estimated, not as it was written.
//!
//! [`FitResult::n_parameters`]: ferx_core::FitResult::n_parameters

use ferx_core::stats::special::regularized_gamma_q;

/// `P(χ²_df ≥ x)`: the survival function of the chi-square distribution.
///
/// `1.0` for `x ≤ 0` (a model that got *worse* with more parameters has
/// nothing to show) and for `df == 0`, where the test is undefined and the
/// caller has already refused the step — see [`Lrt::forward`].
pub fn chi_square_sf(x: f64, df: usize) -> f64 {
    // `x.is_nan()` is spelled out: `!(x > 0.0)` would read the same but hides
    // the NaN arm.
    if df == 0 || x.is_nan() || x <= 0.0 {
        return 1.0;
    }
    if !x.is_finite() {
        return 0.0;
    }
    regularized_gamma_q(df as f64 / 2.0, x / 2.0).clamp(0.0, 1.0)
}

/// One likelihood-ratio comparison, as a search step reports it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lrt {
    /// `OFV_reduced − OFV_extended`: positive when the extended model fits
    /// better.
    pub dofv: f64,
    /// Extra free parameters in the extended model.
    pub df: usize,
    /// `P(χ²_df ≥ ΔOFV)`.
    pub p_value: f64,
    /// The significance level the comparison was made at.
    pub alpha: f64,
    /// `p_value ≤ alpha` — the extended model's extra parameters are
    /// significant.
    pub significant: bool,
}

impl Lrt {
    /// Compare a reduced and an extended fit at level `alpha`.
    ///
    /// `n_reduced` / `n_extended` are the fits' free-parameter counts. Returns
    /// `Err` when the extended model does not actually add a free parameter:
    /// that is not a test with a degenerate answer, it is a candidate that
    /// cannot be judged — a relation whose θ is `FIX`ed, or two candidates
    /// that compiled to the same parameter vector — and the step has to say so
    /// rather than pass it through `χ²(0)`.
    pub fn forward(
        ofv_reduced: f64,
        n_reduced: usize,
        ofv_extended: f64,
        n_extended: usize,
        alpha: f64,
    ) -> Result<Lrt, String> {
        if n_extended <= n_reduced {
            return Err(format!(
                "the extended model has {n_extended} free parameter{} against the reduced \
                 model's {n_reduced}; a likelihood-ratio test needs at least one more",
                if n_extended == 1 { "" } else { "s" }
            ));
        }
        let df = n_extended - n_reduced;
        let dofv = ofv_reduced - ofv_extended;
        let p_value = if dofv.is_finite() {
            chi_square_sf(dofv, df)
        } else {
            f64::NAN
        };
        Ok(Lrt {
            dofv,
            df,
            p_value,
            alpha,
            // `NaN <= alpha` is false: a comparison that cannot be computed
            // is not significant, and the caller reports the NaN.
            significant: p_value <= alpha,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `scipy.stats.chi2.sf`, to the digits scipy prints.
    #[test]
    fn chi_square_tail_matches_scipy() {
        // chi2.sf(3.841459, 1) == 0.05
        assert!((chi_square_sf(3.841_459, 1) - 0.05).abs() < 1e-6);
        // chi2.sf(6.634897, 1) == 0.01
        assert!((chi_square_sf(6.634_897, 1) - 0.01).abs() < 1e-6);
        // chi2.sf(10.827566, 1) == 0.001
        assert!((chi_square_sf(10.827_566, 1) - 0.001).abs() < 1e-6);
        // chi2.sf(9.21034, 2) == 0.01
        assert!((chi_square_sf(9.210_34, 2) - 0.01).abs() < 1e-6);
        // chi2.sf(13.2767, 4) == 0.01
        assert!((chi_square_sf(13.2767, 4) - 0.01).abs() < 1e-5);
        // chi2.sf(20.0, 3) == 0.00017
        assert!((chi_square_sf(20.0, 3) - 0.000_170_0).abs() < 1e-6);
    }

    #[test]
    fn chi_square_tail_edge_cases() {
        assert_eq!(chi_square_sf(0.0, 1), 1.0);
        assert_eq!(chi_square_sf(-5.0, 1), 1.0);
        assert_eq!(chi_square_sf(f64::NAN, 1), 1.0);
        assert_eq!(chi_square_sf(5.0, 0), 1.0);
        assert_eq!(chi_square_sf(f64::INFINITY, 2), 0.0);
        // Monotone in x and in df.
        assert!(chi_square_sf(5.0, 1) < chi_square_sf(4.0, 1));
        assert!(chi_square_sf(5.0, 2) > chi_square_sf(5.0, 1));
    }

    #[test]
    fn forward_test_reads_the_fits_parameter_counts() {
        // PsN's tabulated cutoff for p = 0.01, df = 1 is 6.63.
        let just_over = Lrt::forward(100.0, 10, 100.0 - 6.64, 11, 0.01).unwrap();
        assert_eq!(just_over.df, 1);
        assert!(just_over.significant, "{just_over:?}");
        let just_under = Lrt::forward(100.0, 10, 100.0 - 6.62, 11, 0.01).unwrap();
        assert!(!just_under.significant, "{just_under:?}");

        // Two extra parameters (a hockey stick, or a three-level categorical)
        // need 9.21 at the same level.
        let two_df = Lrt::forward(100.0, 10, 100.0 - 6.64, 12, 0.01).unwrap();
        assert_eq!(two_df.df, 2);
        assert!(!two_df.significant, "{two_df:?}");
        assert!(
            Lrt::forward(100.0, 10, 100.0 - 9.3, 12, 0.01)
                .unwrap()
                .significant
        );
    }

    #[test]
    fn a_worse_extended_model_is_never_significant() {
        let t = Lrt::forward(100.0, 10, 100.5, 11, 0.5).unwrap();
        assert!(t.dofv < 0.0);
        assert_eq!(t.p_value, 1.0);
        assert!(!t.significant);
    }

    #[test]
    fn an_extended_model_that_adds_no_parameter_is_refused() {
        let e = Lrt::forward(100.0, 10, 90.0, 10, 0.01).unwrap_err();
        assert!(e.contains("needs at least one more"), "{e}");
        let e = Lrt::forward(100.0, 10, 90.0, 9, 0.01).unwrap_err();
        assert!(
            e.contains("9 free parameters against the reduced model's 10"),
            "{e}"
        );
    }

    #[test]
    fn a_nan_ofv_is_reported_not_selected() {
        let t = Lrt::forward(100.0, 10, f64::NAN, 11, 0.01).unwrap();
        assert!(t.p_value.is_nan());
        assert!(!t.significant);
    }
}
