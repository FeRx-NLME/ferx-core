use super::nca::SubjectNca;
/// Population pooling: aggregate per-subject NCA estimates to geometric means.
use crate::cold_sort::ColdSort;

/// Population-level NCA summary (geometric means across subjects).
#[derive(Debug, Clone)]
#[allow(dead_code)] // vss/lambda_z/c0/tmax_median exposed for diagnostics
pub struct PopNca {
    pub cl_f: Option<f64>,
    pub v_f: Option<f64>,
    pub vss: Option<f64>,
    pub lambda_z: Option<f64>,
    pub ka: Option<f64>,
    pub c0: Option<f64>,
    pub tmax_median: f64,
    /// Inter-subject CV² of CL/F, for omega initialisation.
    pub cl_cv2: Option<f64>,
    /// Apparent lag time (oral models only); median across subjects with detectable lag.
    pub tlag: Option<f64>,
    /// Q from biexponential peeling (2/3-cpt); None until try_biexp_peel runs.
    pub q_peel: Option<f64>,
    /// V2 from biexponential peeling (2/3-cpt); None until try_biexp_peel runs.
    pub v2_peel: Option<f64>,
}

/// Minimum number of subjects needed for a pooled parameter estimate to be trusted.
const MIN_SUBJECTS: usize = 2;

fn geomean_min_n(iter: impl Iterator<Item = f64>) -> Option<f64> {
    let vals: Vec<f64> = iter.filter(|&v| v > 0.0 && v.is_finite()).collect();
    if vals.len() < MIN_SUBJECTS {
        None
    } else {
        geomean(vals.into_iter())
    }
}

pub fn pool_nca(subjects: &[SubjectNca]) -> PopNca {
    let cl_f = geomean_min_n(subjects.iter().filter_map(|s| s.cl_f));
    let v_f = geomean_min_n(subjects.iter().filter_map(|s| s.v_f));
    let vss = geomean_min_n(subjects.iter().filter_map(|s| s.vss));
    let lambda_z = geomean_min_n(subjects.iter().filter_map(|s| s.lambda_z));
    let ka = geomean_min_n(subjects.iter().filter_map(|s| s.ka));
    let c0 = geomean(
        subjects
            .iter()
            .filter_map(|s| s.c0)
            .filter(|&v| v > 0.0 && v.is_finite()),
    );

    // Median Tmax
    let mut tmaxes: Vec<f64> = subjects
        .iter()
        .map(|s| s.tmax)
        .filter(|t| t.is_finite())
        .collect();
    tmaxes.cold_sort_by(&|a, b| a.partial_cmp(b).unwrap());
    let tmax_median = median_f64(&tmaxes);

    // Median lag time — use median rather than geomean because tlag is often a
    // discrete sampling-time value and a few subjects with no detectable lag
    // should not pull the estimate to zero.
    let tlag = {
        let mut tlags: Vec<f64> = subjects
            .iter()
            .filter_map(|s| s.tlag)
            .filter(|&t| t > 0.0 && t.is_finite())
            .collect();
        tlags.cold_sort_by(&|a, b| a.partial_cmp(b).unwrap());
        if tlags.is_empty() {
            None
        } else {
            Some(median_f64(&tlags))
        }
    };

    // Inter-subject CV² of CL/F: var(log(CL_i / CL_pop))
    let cl_cv2 = cl_f.and_then(|cl_pop| {
        let log_devs: Vec<f64> = subjects
            .iter()
            .filter_map(|s| s.cl_f)
            .filter(|&v| v > 0.0 && v.is_finite())
            .map(|cl_i| (cl_i / cl_pop).ln())
            .collect();
        if log_devs.len() < 3 {
            None
        } else {
            let mean: f64 = log_devs.iter().sum::<f64>() / log_devs.len() as f64;
            let var: f64 = log_devs.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                / (log_devs.len() - 1) as f64;
            if var.is_finite() && var > 0.0 {
                Some(var)
            } else {
                None
            }
        }
    });

    PopNca {
        cl_f,
        v_f,
        vss,
        lambda_z,
        ka,
        c0,
        tmax_median,
        cl_cv2,
        tlag,
        q_peel: None,
        v2_peel: None,
    }
}

fn geomean(iter: impl Iterator<Item = f64>) -> Option<f64> {
    let mut sum_ln = 0.0;
    let mut n = 0usize;
    for v in iter {
        sum_ln += v.ln();
        n += 1;
    }
    if n == 0 {
        None
    } else {
        let gm = (sum_ln / n as f64).exp();
        if gm.is_finite() && gm > 0.0 {
            Some(gm)
        } else {
            None
        }
    }
}

fn median_f64(sorted: &[f64]) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::suggest_start::nca::SubjectNca;

    fn make_nca(cl_f: f64, v_f: f64, ka: f64) -> SubjectNca {
        SubjectNca {
            cl_f: Some(cl_f),
            v_f: Some(v_f),
            vss: None,
            lambda_z: Some(cl_f / v_f),
            ka: Some(ka),
            c0: None,
            tlag: None,
            tmax: 1.0,
            cmax: 10.0,
            auc_inf: Some(v_f / cl_f),
            mrt: None,
        }
    }

    #[test]
    fn test_geomean_basic() {
        let vals = vec![1.0, 2.0, 4.0];
        let gm = geomean(vals.into_iter()).unwrap();
        assert!(
            (gm - 2.0).abs() < 1e-10,
            "geomean([1,2,4]) should be 2.0, got {gm}"
        );
    }

    #[test]
    fn test_pool_nca_geometric_mean() {
        // Three subjects with CL/F = 0.1, 0.2, 0.4 → geomean = 0.2
        let subjects = vec![
            make_nca(0.1, 8.0, 1.0),
            make_nca(0.2, 8.0, 1.0),
            make_nca(0.4, 8.0, 1.0),
        ];
        let pop = pool_nca(&subjects);
        let cl = pop.cl_f.unwrap();
        assert!(
            (cl - 0.2).abs() / 0.2 < 1e-9,
            "pooled CL should be 0.2, got {cl}"
        );
    }

    #[test]
    fn test_pool_nca_drops_nan() {
        let mut subjects = vec![make_nca(0.1, 8.0, 1.0), make_nca(0.2, 8.0, 1.0)];
        // Add a subject with failed CL estimate
        subjects.push(SubjectNca {
            cl_f: None,
            ..make_nca(0.0, 8.0, 1.0)
        });
        let pop = pool_nca(&subjects);
        // Should still produce a valid estimate from the 2 valid subjects
        assert!(pop.cl_f.is_some());
        let cl = pop.cl_f.unwrap();
        assert!((cl - (0.1f64 * 0.2).sqrt()).abs() / (0.1f64 * 0.2).sqrt() < 1e-9);
    }

    #[test]
    fn test_cl_cv2_computed() {
        // CL values: 0.1, 0.2, 0.4 — large IIV
        let subjects = vec![
            make_nca(0.1, 8.0, 1.0),
            make_nca(0.2, 8.0, 1.0),
            make_nca(0.4, 8.0, 1.0),
        ];
        let pop = pool_nca(&subjects);
        assert!(pop.cl_cv2.is_some());
        assert!(pop.cl_cv2.unwrap() > 0.0);
    }
}
