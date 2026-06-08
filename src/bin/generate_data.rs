//! Generate NONMEM-format CSV datasets for all examples.
//! Usage: cargo run --bin generate_data

use ferx_core::api::SimulationResult;
use ferx_core::*;
use std::collections::HashMap;
use std::io::Write;

fn main() {
    generate_warfarin();
    generate_two_cpt_iv();
    generate_two_cpt_oral_cov();
    generate_mm_oral();
    eprintln!("All datasets generated in data/");
}

fn write_nonmem_csv(
    path: &str,
    subjects: &[(
        String,
        f64,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        HashMap<String, f64>,
    )],
    // (id, dose_amt, obs_times, observations, dose_times, covariates)
    cov_names: &[&str],
    dose_cmt: usize,
) {
    let mut f = std::fs::File::create(path).unwrap();

    // Header
    let mut header = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV".to_string();
    for cov in cov_names {
        header.push(',');
        header.push_str(&cov.to_uppercase());
    }
    writeln!(f, "{}", header).unwrap();

    for (id, dose_amt, obs_times, observations, dose_times, covariates) in subjects {
        // Dose records
        for &dt in dose_times {
            let mut line = format!("{},{},.,1,{},{},0,1", id, dt, dose_amt, dose_cmt);
            for cov in cov_names {
                let v = covariates.get(*cov).copied().unwrap_or(0.0);
                line.push_str(&format!(",{:.1}", v));
            }
            writeln!(f, "{}", line).unwrap();
        }

        // Observation records
        for (j, &t) in obs_times.iter().enumerate() {
            let dv = observations[j];
            let mdv = if dv <= 0.001 { 1 } else { 0 };
            let dv_str = if mdv == 1 {
                ".".to_string()
            } else {
                format!("{:.4}", dv)
            };
            let mut line = format!("{},{},{},0,.,{},0,{}", id, t, dv_str, dose_cmt, mdv);
            for cov in cov_names {
                let v = covariates.get(*cov).copied().unwrap_or(0.0);
                line.push_str(&format!(",{:.1}", v));
            }
            writeln!(f, "{}", line).unwrap();
        }
    }

    eprintln!("  Written: {}", path);
}

fn simulate_subjects(
    model: &CompiledModel,
    params: &ModelParameters,
    n_subjects: usize,
    dose_amt: f64,
    dose_cmt: usize,
    obs_times: &[f64],
    seed: u64,
    covariates_fn: Option<&dyn Fn(usize) -> HashMap<String, f64>>,
) -> Vec<(
    String,
    f64,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    HashMap<String, f64>,
)> {
    let subjects: Vec<Subject> = (1..=n_subjects)
        .map(|i| {
            let cov = covariates_fn.map(|f| f(i)).unwrap_or_default();
            Subject {
                id: format!("{}", i),
                doses: vec![DoseEvent::new(0.0, dose_amt, dose_cmt, 0.0, false, 0.0)],
                obs_times: obs_times.to_vec(),
                obs_raw_times: Vec::new(),
                observations: vec![0.0; obs_times.len()],
                obs_cmts: vec![1; obs_times.len()],
                covariates: cov,
                dose_covariates: Vec::new(),
                obs_covariates: Vec::new(),
                pk_only_times: Vec::new(),
                pk_only_covariates: Vec::new(),
                reset_times: Vec::new(),
                cens: vec![0; obs_times.len()],
                occasions: Vec::new(),
                dose_occasions: Vec::new(),
                #[cfg(feature = "survival")]
                obs_records: vec![],
            }
        })
        .collect();

    let pop = Population {
        subjects: subjects.clone(),
        covariate_names: vec![],
        dv_column: "dv".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };

    let sim = simulate_with_seed(model, &pop, params, 1, seed);

    subjects
        .into_iter()
        .map(|subj| {
            let sims: Vec<&SimulationResult> = sim.iter().filter(|s| s.id == subj.id).collect();
            let obs: Vec<f64> = sims
                .iter()
                .map(|s| s.outcome.continuous_value().max(0.0))
                .collect();
            (
                subj.id,
                dose_amt,
                subj.obs_times,
                obs,
                vec![0.0],
                subj.covariates,
            )
        })
        .collect()
}

// ─── Warfarin ───────────────────────────────────────────────────────────────

fn generate_warfarin() {
    eprintln!("Generating warfarin dataset...");
    let model = build_warfarin_model();
    let params = build_warfarin_true_params();
    let obs_times = vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 48.0, 72.0, 96.0, 120.0];

    let subjects = simulate_subjects(&model, &params, 10, 100.0, 1, &obs_times, 42, None);
    write_nonmem_csv("data/warfarin.csv", &subjects, &[], 1);
}

fn build_warfarin_model() -> CompiledModel {
    let theta_names = vec!["TVCL".into(), "TVV".into(), "TVKA".into()];
    let eta_names = vec!["ETA_CL".into(), "ETA_V".into(), "ETA_KA".into()];
    let omega = OmegaMatrix::from_diagonal(&[0.07, 0.02, 0.40], eta_names.clone());
    let sigma = SigmaVector {
        values: vec![0.01],
        names: vec!["PROP_ERR".into()],
    };
    let default_params = ModelParameters {
        theta: vec![0.134, 8.1, 1.0],
        theta_names: theta_names.clone(),
        theta_lower: vec![0.001, 0.1, 0.01],
        theta_upper: vec![10.0, 500.0, 50.0],
        theta_fixed: vec![false; 3],
        omega,
        omega_fixed: vec![false; 3],
        sigma,
        sigma_fixed: vec![false; 1],
        omega_iov: None,
        kappa_fixed: Vec::new(),
    };
    let pk_param_fn: PkParamFn =
        Box::new(|theta: &[f64], eta: &[f64], _: &HashMap<String, f64>| {
            let mut p = PkParams::default();
            p.values[PK_IDX_CL] = theta[0] * eta[0].exp();
            p.values[PK_IDX_V] = theta[1] * eta[1].exp();
            p.values[PK_IDX_KA] = theta[2] * eta[2].exp();
            p
        });
    CompiledModel {
        name: "warfarin".into(),
        pk_model: PkModel::OneCptOral,
        error_model: ErrorModel::Proportional,
        error_spec: ferx_core::types::ErrorSpec::Single(ErrorModel::Proportional),
        pk_param_fn,
        n_theta: 3,
        n_eta: 3,
        n_epsilon: 1,
        theta_names,
        eta_names,
        default_params,
        omega_init_as_sd: vec![false; 3],
        sigma_init_as_sd: vec![false; 1],
        kappa_init_as_sd: Vec::new(),
        tv_fn: None,
        pk_indices: vec![PK_IDX_CL, PK_IDX_V, PK_IDX_KA],

        eta_map: (0..3).map(|i| i as i32).collect(),

        pk_idx_f64: [PK_IDX_CL, PK_IDX_V, PK_IDX_KA]
            .iter()
            .map(|&i| i as f64)
            .collect(),

        sel_flat: {
            // n_tv = 3, n_eta = 3, each tv uses its positional eta.
            let mut v = vec![0.0f64; 3 * 3];
            for i in 0..3 {
                v[i * 3 + i] = 1.0;
            }
            v
        },
        ode_spec: None,
        diffusion_theta_start: None,
        diffusion_state_indices: Vec::new(),
        bloq_method: BloqMethod::Drop,
        mu_refs: HashMap::new(),
        kappa_mu_refs: HashMap::new(),
        referenced_covariates: Vec::new(),
        gradient_method: GradientMethod::default(),
        parse_warnings: Vec::new(),
        eta_param_info: Vec::new(),
        theta_transform: Vec::new(),
        n_kappa: 0,
        kappa_names: Vec::new(),
        indiv_param_names: vec!["CL".into(), "V".into(), "KA".into()],
        indiv_param_partials: crate::types::IndivParamPartials::empty(),
        #[cfg(feature = "nn")]
        covariate_nns: Vec::new(),
        scaling: ScalingSpec::None,
        log_transform: false,
        dv_pre_logged: false,
        derived_exprs: vec![],
        output_columns: vec![],
        #[cfg(feature = "survival")]
        endpoints: std::collections::HashMap::new(),
    }
}

fn build_warfarin_true_params() -> ModelParameters {
    ModelParameters {
        theta: vec![0.134, 8.1, 1.0],
        theta_names: vec!["TVCL".into(), "TVV".into(), "TVKA".into()],
        theta_lower: vec![0.001, 0.1, 0.01],
        theta_upper: vec![10.0, 500.0, 50.0],
        theta_fixed: vec![false; 3],
        omega: OmegaMatrix::from_diagonal(
            &[0.07, 0.02, 0.40],
            vec!["ETA_CL".into(), "ETA_V".into(), "ETA_KA".into()],
        ),
        omega_fixed: vec![false; 3],
        sigma: SigmaVector {
            values: vec![0.01],
            names: vec!["PROP_ERR".into()],
        },
        sigma_fixed: vec![false; 1],
        omega_iov: None,
        kappa_fixed: Vec::new(),
    }
}

// ─── Two-compartment IV ─────────────────────────────────────────────────────

fn generate_two_cpt_iv() {
    eprintln!("Generating two_cpt_iv dataset...");
    let theta_names = vec!["TVCL".into(), "TVV1".into(), "TVQ".into(), "TVV2".into()];
    let eta_names = vec![
        "ETA_CL".into(),
        "ETA_V1".into(),
        "ETA_Q".into(),
        "ETA_V2".into(),
    ];
    let omega = OmegaMatrix::from_diagonal(&[0.10, 0.10, 0.10, 0.10], eta_names.clone());
    let sigma = SigmaVector {
        values: vec![0.01],
        names: vec!["PROP_ERR".into()],
    };
    let params = ModelParameters {
        theta: vec![5.0, 15.0, 3.0, 30.0],
        theta_names: theta_names.clone(),
        theta_lower: vec![0.1, 1.0, 0.01, 1.0],
        theta_upper: vec![100.0, 500.0, 100.0, 500.0],
        theta_fixed: vec![false; 4],
        omega,
        omega_fixed: vec![false; 4],
        sigma,
        sigma_fixed: vec![false; 1],
        omega_iov: None,
        kappa_fixed: Vec::new(),
    };
    let pk_param_fn: PkParamFn =
        Box::new(|theta: &[f64], eta: &[f64], _: &HashMap<String, f64>| {
            let mut p = PkParams::default();
            p.values[PK_IDX_CL] = theta[0] * eta[0].exp();
            p.values[PK_IDX_V] = theta[1] * eta[1].exp();
            p.values[PK_IDX_Q] = theta[2] * eta[2].exp();
            p.values[PK_IDX_V2] = theta[3] * eta[3].exp();
            p
        });
    let model = CompiledModel {
        name: "two_cpt_iv".into(),
        pk_model: PkModel::TwoCptIv,
        error_model: ErrorModel::Proportional,
        error_spec: ferx_core::types::ErrorSpec::Single(ErrorModel::Proportional),
        pk_param_fn,
        n_theta: 4,
        n_eta: 4,
        n_epsilon: 1,
        theta_names,
        eta_names,
        default_params: params.clone(),
        omega_init_as_sd: vec![false; 4],
        sigma_init_as_sd: vec![false; 1],
        kappa_init_as_sd: Vec::new(),
        tv_fn: None,
        pk_indices: vec![PK_IDX_CL, PK_IDX_V, PK_IDX_Q, PK_IDX_V2],

        eta_map: (0..4).map(|i| i as i32).collect(),

        pk_idx_f64: [PK_IDX_CL, PK_IDX_V, PK_IDX_Q, PK_IDX_V2]
            .iter()
            .map(|&i| i as f64)
            .collect(),

        sel_flat: {
            let mut v = vec![0.0f64; 4 * 4];
            for i in 0..4 {
                v[i * 4 + i] = 1.0;
            }
            v
        },
        ode_spec: None,
        diffusion_theta_start: None,
        diffusion_state_indices: Vec::new(),
        bloq_method: BloqMethod::Drop,
        mu_refs: HashMap::new(),
        kappa_mu_refs: HashMap::new(),
        referenced_covariates: Vec::new(),
        gradient_method: GradientMethod::default(),
        parse_warnings: Vec::new(),
        eta_param_info: Vec::new(),
        theta_transform: Vec::new(),
        n_kappa: 0,
        kappa_names: Vec::new(),
        indiv_param_names: vec!["CL".into(), "V".into(), "Q".into(), "V2".into()],
        indiv_param_partials: crate::types::IndivParamPartials::empty(),
        #[cfg(feature = "nn")]
        covariate_nns: Vec::new(),
        scaling: ScalingSpec::None,
        log_transform: false,
        dv_pre_logged: false,
        derived_exprs: vec![],
        output_columns: vec![],
        #[cfg(feature = "survival")]
        endpoints: std::collections::HashMap::new(),
    };
    let obs_times = vec![0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 48.0, 72.0];
    let subjects = simulate_subjects(&model, &params, 15, 100.0, 1, &obs_times, 123, None);
    write_nonmem_csv("data/two_cpt_iv.csv", &subjects, &[], 1);
}

// ─── Two-compartment oral with covariates ───────────────────────────────────

fn generate_two_cpt_oral_cov() {
    eprintln!("Generating two_cpt_oral_cov dataset...");
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    let theta_names: Vec<String> = vec![
        "TVCL",
        "TVV1",
        "TVQ",
        "TVV2",
        "TVKA",
        "THETA_WT",
        "THETA_CRCL",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    let eta_names: Vec<String> = vec!["ETA_CL", "ETA_V1", "ETA_Q", "ETA_V2", "ETA_KA"]
        .into_iter()
        .map(String::from)
        .collect();
    let omega = OmegaMatrix::from_diagonal(&[0.10, 0.10, 0.05, 0.05, 0.15], eta_names.clone());
    let sigma = SigmaVector {
        values: vec![0.02],
        names: vec!["PROP_ERR".into()],
    };
    let params = ModelParameters {
        theta: vec![5.0, 50.0, 10.0, 100.0, 1.2, 0.75, 0.50],
        theta_names: theta_names.clone(),
        theta_lower: vec![0.1, 1.0, 0.1, 1.0, 0.01, 0.01, 0.01],
        theta_upper: vec![100.0, 500.0, 100.0, 500.0, 10.0, 5.0, 5.0],
        theta_fixed: vec![false; 7],
        omega,
        omega_fixed: vec![false; 5],
        sigma,
        sigma_fixed: vec![false; 1],
        omega_iov: None,
        kappa_fixed: Vec::new(),
    };
    let pk_param_fn: PkParamFn =
        Box::new(|theta: &[f64], eta: &[f64], cov: &HashMap<String, f64>| {
            let wt = cov.get("wt").copied().unwrap_or(70.0);
            let crcl = cov.get("crcl").copied().unwrap_or(100.0);
            let mut p = PkParams::default();
            p.values[PK_IDX_CL] = theta[0]
                * (wt / 70.0).powf(theta[5])
                * (crcl / 100.0).powf(theta[6])
                * eta[0].exp();
            p.values[PK_IDX_V] = theta[1] * (wt / 70.0).powf(theta[5]) * eta[1].exp();
            p.values[PK_IDX_Q] = theta[2] * eta[2].exp();
            p.values[PK_IDX_V2] = theta[3] * eta[3].exp();
            p.values[PK_IDX_KA] = theta[4] * eta[4].exp();
            p
        });
    let model = CompiledModel {
        name: "two_cpt_oral_cov".into(),
        pk_model: PkModel::TwoCptOral,
        error_model: ErrorModel::Proportional,
        error_spec: ferx_core::types::ErrorSpec::Single(ErrorModel::Proportional),
        pk_param_fn,
        n_theta: 7,
        n_eta: 5,
        n_epsilon: 1,
        theta_names,
        eta_names,
        default_params: params.clone(),
        omega_init_as_sd: vec![false; 5],
        sigma_init_as_sd: vec![false; 1],
        kappa_init_as_sd: Vec::new(),
        tv_fn: None,
        pk_indices: vec![PK_IDX_CL, PK_IDX_V, PK_IDX_Q, PK_IDX_V2, PK_IDX_KA],

        eta_map: (0..5).map(|i| i as i32).collect(),

        pk_idx_f64: [PK_IDX_CL, PK_IDX_V, PK_IDX_Q, PK_IDX_V2, PK_IDX_KA]
            .iter()
            .map(|&i| i as f64)
            .collect(),

        sel_flat: {
            let mut v = vec![0.0f64; 5 * 5];
            for i in 0..5 {
                v[i * 5 + i] = 1.0;
            }
            v
        },
        ode_spec: None,
        diffusion_theta_start: None,
        diffusion_state_indices: Vec::new(),
        bloq_method: BloqMethod::Drop,
        mu_refs: HashMap::new(),
        kappa_mu_refs: HashMap::new(),
        referenced_covariates: Vec::new(),
        gradient_method: GradientMethod::default(),
        parse_warnings: Vec::new(),
        eta_param_info: Vec::new(),
        theta_transform: Vec::new(),
        n_kappa: 0,
        kappa_names: Vec::new(),
        indiv_param_names: vec![
            "CL".into(),
            "V".into(),
            "Q".into(),
            "V2".into(),
            "KA".into(),
        ],
        indiv_param_partials: crate::types::IndivParamPartials::empty(),
        #[cfg(feature = "nn")]
        covariate_nns: Vec::new(),
        scaling: ScalingSpec::None,
        log_transform: false,
        dv_pre_logged: false,
        derived_exprs: vec![],
        output_columns: vec![],
        #[cfg(feature = "survival")]
        endpoints: std::collections::HashMap::new(),
    };

    // Generate random covariates (matching Julia seed 456)
    let mut rng = rand::rngs::StdRng::seed_from_u64(456);
    let normal = Normal::new(0.0, 1.0).unwrap();
    // Generate covariates first, then build subjects manually.
    let obs_times = vec![0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0, 36.0, 48.0];
    let n_subjects = 30;

    let covs: Vec<HashMap<String, f64>> = (0..n_subjects)
        .map(|_| {
            let wt = (70.0 + 15.0 * normal.sample(&mut rng) as f64).clamp(45.0, 120.0);
            let crcl = (90.0 + 25.0 * normal.sample(&mut rng) as f64).clamp(30.0, 150.0);
            let mut m = HashMap::new();
            m.insert("wt".into(), wt);
            m.insert("crcl".into(), crcl);
            m
        })
        .collect();

    let subjects: Vec<Subject> = (0..n_subjects)
        .map(|i| Subject {
            id: format!("{}", i + 1),
            doses: vec![DoseEvent::new(0.0, 250.0, 1, 0.0, false, 0.0)],
            obs_times: obs_times.clone(),
            obs_raw_times: Vec::new(),
            observations: vec![0.0; obs_times.len()],
            obs_cmts: vec![1; obs_times.len()],
            covariates: covs[i].clone(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; obs_times.len()],
            occasions: Vec::new(),
            dose_occasions: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        })
        .collect();
    let pop = Population {
        subjects,
        covariate_names: vec!["wt".into(), "crcl".into()],
        dv_column: "dv".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };
    let sim = simulate_with_seed(&model, &pop, &params, 1, 456);

    let result: Vec<_> = pop
        .subjects
        .iter()
        .map(|subj| {
            let sims: Vec<_> = sim.iter().filter(|s| s.id == subj.id).collect();
            let obs: Vec<f64> = sims
                .iter()
                .map(|s| s.outcome.continuous_value().max(0.0))
                .collect();
            (
                subj.id.clone(),
                250.0,
                subj.obs_times.clone(),
                obs,
                vec![0.0],
                subj.covariates.clone(),
            )
        })
        .collect();

    write_nonmem_csv("data/two_cpt_oral_cov.csv", &result, &["wt", "crcl"], 1);
}

// ─── Michaelis-Menten ODE ───────────────────────────────────────────────────

fn generate_mm_oral() {
    eprintln!("Generating mm_oral dataset...");
    let theta_names = vec!["TVVMAX".into(), "TVKM".into(), "TVV".into(), "TVKA".into()];
    let eta_names = vec!["ETA_VMAX".into(), "ETA_V".into()];
    let omega = OmegaMatrix::from_diagonal(&[0.15, 0.10], eta_names.clone());
    let sigma = SigmaVector {
        values: vec![0.02],
        names: vec!["PROP_ERR".into()],
    };
    let params = ModelParameters {
        theta: vec![4.0, 6.0, 12.0, 1.5],
        theta_names: theta_names.clone(),
        theta_lower: vec![0.1, 0.1, 1.0, 0.05],
        theta_upper: vec![50.0, 100.0, 200.0, 20.0],
        theta_fixed: vec![false; 4],
        omega,
        omega_fixed: vec![false; 2],
        sigma,
        sigma_fixed: vec![false; 1],
        omega_iov: None,
        kappa_fixed: Vec::new(),
    };
    let pk_param_fn: PkParamFn =
        Box::new(|theta: &[f64], eta: &[f64], _: &HashMap<String, f64>| {
            let mut p = PkParams::default();
            p.values[0] = theta[0] * eta[0].exp(); // VMAX
            p.values[1] = theta[1]; // KM
            p.values[2] = theta[2] * eta[1].exp(); // V
            p.values[4] = theta[3]; // KA
            p
        });
    let ode_rhs: Box<dyn Fn(&[f64], &[f64], f64, &mut [f64]) + Send + Sync> =
        Box::new(|u: &[f64], params: &[f64], _t: f64, du: &mut [f64]| {
            let (depot, central) = (u[0], u[1]);
            let (vmax, km, v, ka) = (params[0], params[1], params[2], params[4]);
            du[0] = -ka * depot;
            du[1] = ka * depot / v - vmax * central / (km + central);
        });
    let ode_spec = ferx_core::ode::OdeSpec {
        rhs: ode_rhs,
        n_states: 2,
        state_names: vec!["depot".into(), "central".into()],
        readout: ferx_core::ode::OdeReadout::ObsCmt(1),
        diffusion_var: Vec::new(),
        init_fn: None,
    };
    let model = CompiledModel {
        name: "mm_oral".into(),
        pk_model: PkModel::OneCptOral,
        error_model: ErrorModel::Proportional,
        error_spec: ferx_core::types::ErrorSpec::Single(ErrorModel::Proportional),
        pk_param_fn,
        n_theta: 4,
        n_eta: 2,
        n_epsilon: 1,
        theta_names,
        eta_names,
        default_params: params.clone(),
        omega_init_as_sd: vec![false; 2],
        sigma_init_as_sd: vec![false; 1],
        kappa_init_as_sd: Vec::new(),
        tv_fn: None,
        pk_indices: vec![0, 2],

        eta_map: (0..2).map(|i| i as i32).collect(),

        pk_idx_f64: vec![0.0, 2.0],

        sel_flat: {
            let mut v = vec![0.0f64; 2 * 2];
            for i in 0..2 {
                v[i * 2 + i] = 1.0;
            }
            v
        },
        ode_spec: Some(ode_spec),
        diffusion_theta_start: None,
        diffusion_state_indices: Vec::new(),
        bloq_method: BloqMethod::Drop,
        mu_refs: HashMap::new(),
        kappa_mu_refs: HashMap::new(),
        referenced_covariates: Vec::new(),
        gradient_method: GradientMethod::default(),
        parse_warnings: Vec::new(),
        eta_param_info: Vec::new(),
        theta_transform: Vec::new(),
        n_kappa: 0,
        kappa_names: Vec::new(),
        indiv_param_names: vec!["VMAX".into(), "KM".into(), "V".into(), "KA".into()],
        indiv_param_partials: crate::types::IndivParamPartials::empty(),
        #[cfg(feature = "nn")]
        covariate_nns: Vec::new(),
        scaling: ScalingSpec::None,
        log_transform: false,
        dv_pre_logged: false,
        derived_exprs: vec![],
        output_columns: vec![],
        #[cfg(feature = "survival")]
        endpoints: std::collections::HashMap::new(),
    };
    let obs_times = vec![
        0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 24.0, 36.0, 48.0,
    ];
    let subjects = simulate_subjects(&model, &params, 20, 200.0, 1, &obs_times, 1234, None);
    write_nonmem_csv("data/mm_oral.csv", &subjects, &[], 1);
}
