//! usage: fitrx-cost <model.ferx> <data.csv> <out.fitrx> <reps>
use ferx_core::io::fitrx::{load_fit_without_population, save_fit, SaveFitOptions};
use std::path::{Path, PathBuf};
use std::time::Instant;
fn med(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let reps: usize = a[4].parse().unwrap();
    let t0 = Instant::now();
    let (fit, pop) = ferx_core::run_model_with_data(&a[1], Some(&a[2])).expect("fit");
    let t_fit = t0.elapsed().as_secs_f64();
    let src = fit.model_text.clone().unwrap_or_default();
    let out = Path::new(&a[3]);
    let mut ts = vec![];
    for _ in 0..reps {
        let t = Instant::now();
        save_fit(&fit, &pop, &src, out, SaveFitOptions::default()).unwrap();
        ts.push(t.elapsed().as_secs_f64());
    }
    let size = std::fs::metadata(out).unwrap().len();
    let mut tl = vec![];
    for _ in 0..reps {
        let t = Instant::now();
        load_fit_without_population(out).unwrap();
        tl.push(t.elapsed().as_secs_f64());
    }
    let wd = out.with_extension("data.fitrx");
    save_fit(&fit, &pop, &src, &wd, SaveFitOptions { include_data: Some(PathBuf::from(&a[2])) }).unwrap();
    let sized = std::fs::metadata(&wd).unwrap().len();
    println!("{{\"subjects\":{},\"obs\":{},\"fit_s\":{:.4},\"save_s\":{:.5},\"load_s\":{:.5},\"bytes\":{},\"bytes_with_data\":{}}}",
        fit.n_subjects, fit.n_obs, t_fit, med(ts), med(tl), size, sized);
}
