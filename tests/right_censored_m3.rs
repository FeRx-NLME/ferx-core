use ferx_core::{fit, parse_model_string, read_nonmem_csv, FitOptions};
use std::io::Write;

const MODEL_SRC: &str = r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.20 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
  maxiter = 1
  gradient = fd
  covariance = false
  bloq_method = m3
";

fn write_csv(contents: &str) -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::new().expect("create temp csv");
    f.write_all(contents.as_bytes()).expect("write temp csv");
    f
}

#[test]
fn cens_negative_one_dataset_fits_with_m3() {
    let csv = "ID,TIME,DV,EVID,MDV,AMT,CMT,CENS\n\
               1,0,.,1,1,100,1,0\n\
               1,1,9.0,0,0,.,1,0\n\
               1,2,12.0,0,0,.,1,-1\n\
               2,0,.,1,1,100,1,0\n\
               2,1,8.5,0,0,.,1,0\n\
               2,2,11.5,0,0,.,1,-1\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).expect("read CENS=-1 dataset");
    assert_eq!(pop.subjects[0].cens, vec![0, -1]);
    assert_eq!(pop.subjects[1].cens, vec![0, -1]);

    let model = parse_model_string(MODEL_SRC).expect("model parses");
    let mut opts = FitOptions::default();
    opts.outer_maxiter = 1;
    opts.run_covariance_step = false;

    let result = fit(&model, &pop, &model.default_params, &opts).expect("M3 fit accepts CENS=-1");
    assert!(result.ofv.is_finite());
}
