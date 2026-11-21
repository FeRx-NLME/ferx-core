//! A fit leaves no plan handle or cached buffer on its pool's workers (#3077), also when its
//! last plan consumer is not the ODE settle check, which releases around itself only when it
//! applies.

use super::fit_inner;
use crate::parser::model_parser::rhs_regcode::kernel_jit::thread_cache_count;
use crate::types::{EstimationMethod, FitOptions};
use std::path::Path;

/// A closed-form model, where the settle check does not apply, with SIR as the fit's last plan
/// consumer, run on a pool the test owns so its workers can be read after the fit. Mutation: drop `fit_inner`'s `ReleaseOnDrop` -> the
/// workers keep the plans they looked up last and the buffers their passes pooled (the scope's
/// end prunes the shared maps only).
#[test]
fn a_fit_whose_settle_check_does_not_apply_leaves_its_workers_empty() {
    let parsed =
        crate::parser::model_parser::parse_full_model_file(Path::new("examples/warfarin.ferx"))
            .expect("parse");
    let model = &parsed.model;
    assert!(model.integrator_spec().is_none(), "fixture: the settle check does not apply");
    let (population, _) = crate::api::read_population_for(
        model,
        &parsed.covariate_decls,
        "data/warfarin.csv",
        None,
        parsed.fit_options.iov_column.as_deref(),
        None,
        &parsed.column_map,
    )
    .expect("read");
    let options = FitOptions {
        method: EstimationMethod::FoceI,
        outer_maxiter: 2,
        // SIR re-solves every subject's modes on the workers after the covariance step's own
        // release, so it is the last plan consumer and the workers hold its caches at the end.
        run_covariance_step: true,
        sir: true,
        sir_samples: 40,
        sir_resamples: 10,
        sir_seed: Some(3077),
        verbose: false,
        ..FitOptions::default()
    };
    let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().expect("a pool");
    let after = pool.install(|| {
        let r = fit_inner(model, &population, &population, &model.default_params, &options)
            .expect("the fit");
        assert!(r.sir_ess.is_some(), "fixture: SIR ran: {:?}", r.warnings);
        rayon::broadcast(|_| thread_cache_count())
    });
    assert_eq!(after, vec![0, 0], "every worker's caches left with the fit");
}
