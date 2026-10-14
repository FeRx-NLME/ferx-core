//! NONMEM anchor for the `TIME` that `[individual_parameters]` reads at a record: the TIME as
//! written, for every record kind (#1969).
//!
//! The engines hold individual parameters constant per record, taking them from the record that
//! ends each interval, which is NONMEM's rule. They order and integrate on an internal clock that
//! differs from the data's `TIME` in two ways: the reader moves a trough (or an `EVID=2` row)
//! listed before a co-timed dose one ULP down (#614, #1810), and it shifts a reset period that
//! restarts `TIME` past the previous one. Before #1969 the parameters read that internal clock, so
//! `if (TIME >= 24)` at a trough written at 24 took the earlier branch, and every record of a
//! restarted period read a time past the step. NONMEM's `$PK` reads the TIME as written.
//!
//! # The control streams
//!
//! `nonmem_anchor/param_time_<case>.ctl`, each a 30-subject FOCEI fit (`METHOD=1 INTER`) by the
//! `nonmem:7.6.0-anchor` build; ferx is evaluated once at NONMEM's optimum (`maxiter = 0`) on a
//! closed-form and an `[odes]` twin, so the two sides share no solver.
//!
//! | case | data | what reads `TIME` | governing record |
//! |---|---|---|---|
//! | `moved` | trough at 24/48 listed before its dose | `CL` | the moved observation |
//! | `pkonly` | an `EVID=2` row at 24/48 listed before the dose | `CL` | the moved pk-only row |
//! | `reset` | `moved` plus a period 2 restarting `TIME` at 0 (`EVID=4`) | `CL` | every period-2 record |
//! | `control` | the dose listed first; nothing moved | `CL` | an observation, unmoved |
//! | `reset_f` | two periods, oral, no observation at a dose | `F` | the period-2 dose at 0 |
//! | `reset_seed` | two periods, `A_0`/`init` = `BASE` | `BASE` | the period-2 reset row |
//!
//! IV 100 at 0/24/48 (`reset_f`, `reset_seed`: 0/24 per period), so every later dose lands on
//! residual drug. The `control` case agrees with NONMEM before and after the fix: it pins that the
//! fix changes only records whose two clocks differ. Measured before the fix (main `944ca32d`):
//! `moved` 1006.9 OFV off, `pkonly` 20.6, `reset` 9799.7, `reset_f` 3374.3, `reset_seed` 921.7.
//!
//! Raw NONMEM output is committed under `nonmem_anchor/results/param_time_<case>.*`.
//!
//! Tier 2: `maxiter = 0`, one objective evaluation per cell, no convergence loop.

use std::collections::BTreeMap;
use std::path::PathBuf;

use ferx_core::{fit, prepare_run, FitResult, Population};

fn anchor(name: &str) -> String {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("nonmem_anchor");
    p.push(name);
    p.to_string_lossy().into_owned()
}

fn whitespace_rows(name: &str) -> Vec<Vec<String>> {
    let path = anchor(&format!("results/{name}"));
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .lines()
        .map(|l| l.split_whitespace().map(str::to_string).collect())
        .collect()
}

/// `#OBJV` at NONMEM's optimum: the `OBJ` column of the `.ext`'s `-1000000000` row.
fn nm_ofv(case: &str) -> f64 {
    let rows = whitespace_rows(&format!("param_time_{case}.ext"));
    let header = rows
        .iter()
        .find(|r| r.first().map(String::as_str) == Some("ITERATION"))
        .expect(".ext header");
    let obj = header.iter().position(|h| h == "OBJ").expect("OBJ column");
    let last = rows
        .iter()
        .find(|r| r.first().map(String::as_str) == Some("-1000000000"))
        .expect(".ext final row");
    last[obj].parse().expect("OBJ value")
}

/// NONMEM's `IPRED` per subject, at the observation rows (`EVID = 0`) in file order, from the
/// `.tab` written at `FORMAT=s1PE23.16`.
fn nm_ipred(case: &str) -> BTreeMap<String, Vec<f64>> {
    let rows = whitespace_rows(&format!("param_time_{case}.tab"));
    let header = rows
        .iter()
        .find(|r| r.first().map(String::as_str) == Some("ID"))
        .expect(".tab header");
    let at = |label: &str| header.iter().position(|h| h == label).expect(label);
    let (id, evid, ipred) = (at("ID"), at("EVID"), at("IPRED"));
    let mut out: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for r in rows
        .iter()
        .filter(|r| r.len() == header.len() && r[0] != "ID")
    {
        let num = |c: usize| -> f64 { r[c].parse().expect("a .tab number") };
        if num(evid) == 0.0 {
            out.entry(format!("{}", num(id) as i64))
                .or_default()
                .push(num(ipred));
        }
    }
    out
}

fn run(case: &str, engine: &str) -> (FitResult, Population) {
    let run = prepare_run(
        &anchor(&format!("param_time_{case}_{engine}.ferx")),
        Some(&anchor(&format!("param_time_{case}.csv"))),
    )
    .unwrap_or_else(|e| panic!("{case}/{engine}: the anchor model and dataset load: {e}"));
    let r = fit(
        &run.parsed.model,
        &run.population,
        &run.init_params,
        &run.parsed.fit_options,
    )
    .unwrap_or_else(|e| panic!("{case}/{engine}: the evaluation runs: {e}"));
    (r, run.population)
}

/// `|ΔOFV|` against NONMEM, every cell. Measured worst: 2.093e-8 (`reset_seed`/ode; the other
/// eleven cells 3e-10 to 1.6e-8), so ~10× headroom. Before the fix the smallest gap was 20.6.
const OFV_TOL: f64 = 2e-7;
/// `|ΔIPRED| / max(|IPRED|, 1e-3)` against NONMEM's full-precision `.tab`, every observation.
/// Measured worst: 7.387e-8 (`pkonly`/ode), so ~13× headroom. Before the fix the trough at 24
/// was off by up to 8× (`moved`) and 62× (`reset`).
const IPRED_TOL: f64 = 1e-6;

fn assert_matches_nonmem(case: &str, engine: &str) {
    let want_ofv = nm_ofv(case);
    let want_ipred = nm_ipred(case);
    assert_eq!(want_ipred.len(), 30, "{case}: thirty subjects in the .tab");
    let (r, _) = run(case, engine);

    // Every fixture must run the analytic (Dual) inner gradient: an FD fallback would leave the
    // dual seeders this fix also changes unexercised.
    assert!(
        !r.warnings
            .iter()
            .any(|w| w.contains("finite-difference inner gradients")),
        "{case}/{engine}: a subject fell back to FD inner gradients: {:?}",
        r.warnings
    );

    assert!(r.ofv.is_finite(), "{case}/{engine}: OFV {}", r.ofv);
    let d_ofv = (r.ofv - want_ofv).abs();

    let mut worst_ipred = 0.0f64;
    let mut n = 0usize;
    for s in &r.subjects {
        let want = want_ipred
            .get(&s.id)
            .unwrap_or_else(|| panic!("{case}: no .tab rows for subject {}", s.id));
        assert_eq!(
            s.ipred.len(),
            want.len(),
            "{case}/{engine}: subject {} observation count",
            s.id
        );
        for (g, w) in s.ipred.iter().zip(want) {
            assert!(g.is_finite(), "{case}/{engine}: subject {} IPRED {g}", s.id);
            worst_ipred = worst_ipred.max((g - w).abs() / w.abs().max(1e-3));
            n += 1;
        }
    }
    assert_eq!(n, r.subjects.iter().map(|s| s.n_obs).sum::<usize>());
    eprintln!(
        "{case}/{engine}: ferx {:.10} NONMEM {want_ofv:.10} |dOFV| {d_ofv:.3e} \
         worst rel |dIPRED| {worst_ipred:.3e} over {n} observations",
        r.ofv
    );
    assert!(
        d_ofv < OFV_TOL,
        "{case}/{engine}: ferx OFV {} vs NONMEM {want_ofv} (|d| {d_ofv:.3e})",
        r.ofv
    );
    assert!(
        worst_ipred < IPRED_TOL,
        "{case}/{engine}: worst relative IPRED gap {worst_ipred:.3e} against NONMEM's .tab"
    );
}

#[test]
fn a_trough_listed_before_its_dose_reads_the_written_time() {
    assert_matches_nonmem("moved", "analytic");
    assert_matches_nonmem("moved", "ode");
}

#[test]
fn a_pk_only_row_listed_before_its_dose_reads_the_written_time() {
    assert_matches_nonmem("pkonly", "analytic");
    assert_matches_nonmem("pkonly", "ode");
}

#[test]
fn a_reset_period_that_restarts_time_reads_the_written_time() {
    assert_matches_nonmem("reset", "analytic");
    assert_matches_nonmem("reset", "ode");
}

#[test]
fn a_dose_listed_first_is_unchanged() {
    assert_matches_nonmem("control", "analytic");
    assert_matches_nonmem("control", "ode");
}

#[test]
fn a_period_two_dose_reads_f_at_its_written_time() {
    assert_matches_nonmem("reset_f", "analytic");
    assert_matches_nonmem("reset_f", "ode");
}

#[test]
fn a_reset_row_reseeds_init_at_its_written_time() {
    assert_matches_nonmem("reset_seed", "ode");
}

/// The sdtab individual-parameter columns report the parameters the prediction used: `CL` and
/// `IND` at every observation against NONMEM's `.tab`, so the trough written at 24 shows the
/// post-step `CL` (and period 2 of `reset` the pre-step one from TIME 0). `IND` carries no η, so
/// its branch must match exactly; `CL`'s measured worst relative gap is 1.746e-9 (`moved`; `reset`
/// 1.639e-9), the EBE agreement, so 2e-8 is ~11× headroom. Before the fix the trough at 24
/// reported `IND = 1` against NONMEM's `IND = 1.956`.
#[test]
fn the_sdtab_parameter_columns_read_the_written_time() {
    for case in ["moved", "reset"] {
        let src = std::fs::read_to_string(anchor(&format!("param_time_{case}_analytic.ferx")))
            .expect("anchor model");
        let dir = tempfile::tempdir().expect("tempdir");
        let model = dir.path().join("m.ferx");
        std::fs::write(&model, format!("{src}\n[output]\n  CL IND\n")).expect("write model");
        let run = prepare_run(
            model.to_str().expect("utf-8"),
            Some(&anchor(&format!("param_time_{case}.csv"))),
        )
        .expect("load");
        let r = fit(
            &run.parsed.model,
            &run.population,
            &run.init_params,
            &run.parsed.fit_options,
        )
        .expect("evaluate");

        // NONMEM's CL and IND at each observation row, per subject.
        let rows = whitespace_rows(&format!("param_time_{case}.tab"));
        let header = rows.iter().find(|r| r[0] == "ID").expect("header").clone();
        let at = |l: &str| header.iter().position(|h| h == l).expect(l);
        let (id, evid, cl, ind) = (at("ID"), at("EVID"), at("CL"), at("IND"));
        let mut want: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();
        for r in rows
            .iter()
            .filter(|r| r.len() == header.len() && r[0] != "ID")
        {
            let n = |c: usize| -> f64 { r[c].parse().expect("number") };
            if n(evid) == 0.0 {
                want.entry(format!("{}", n(id) as i64))
                    .or_default()
                    .push((n(cl), n(ind)));
            }
        }

        let mut worst = 0.0f64;
        let mut stepped_at_trough = 0usize;
        for (s, pop_s) in r.subjects.iter().zip(&run.population.subjects) {
            let col = |name: &str| -> &Vec<f64> {
                &s.extra_columns
                    .iter()
                    .find(|(n, _)| n == name)
                    .unwrap_or_else(|| panic!("{case}: no {name} column"))
                    .1
            };
            let (got_cl, got_ind) = (col("CL"), col("IND"));
            let w = &want[&s.id];
            assert_eq!(got_cl.len(), w.len(), "{case}: subject {} rows", s.id);
            for (j, (&(wcl, wind), (&gcl, &gind))) in
                w.iter().zip(got_cl.iter().zip(got_ind)).enumerate()
            {
                assert!(gcl.is_finite() && gind.is_finite(), "{case}: {gcl} {gind}");
                // IND is 1 or TVIND (no η): the branch must match NONMEM's exactly.
                assert!(
                    (gind - wind).abs() < 1e-12 * wind.abs(),
                    "{case}: subject {} obs {j}: IND {gind} vs NONMEM {wind}",
                    s.id
                );
                worst = worst.max((gcl - wcl).abs() / wcl.abs());
                if pop_s.param_time_obs(j) == 24.0 && pop_s.obs_times[j] < 24.0 {
                    assert!(gind > 1.0, "{case}: the trough at 24 reports IND {gind}");
                    stepped_at_trough += 1;
                }
            }
        }
        eprintln!("{case}: sdtab CL worst rel gap {worst:.3e}");
        assert!(
            stepped_at_trough > 0,
            "{case}: no moved trough at 24 was checked"
        );
        assert!(
            worst < 2e-8,
            "{case}: sdtab CL vs NONMEM, worst rel {worst:.3e}"
        );
    }
}

/// The fixtures straddle the step: in each, the governing record sits on one side of
/// `TIME = 24` on the engine clock and on the other side as written. Without this the data could
/// quietly stop crossing the step (a reader change that stops moving the trough, say) and the
/// anchors above would agree with NONMEM for a reason unrelated to the clock the parameters
/// read. The control asserts the opposite: no record of it has two clocks.
#[test]
fn every_fixture_straddles_the_step_and_the_control_does_not() {
    let pop = |case: &str| run_population(case);

    // moved: the trough at 24 runs below 24 on the engine clock.
    let p = pop("moved");
    let s = &p.subjects[0];
    let j = (0..s.obs_times.len())
        .find(|&j| s.param_time_obs(j) == 24.0)
        .expect("moved: an observation written at 24");
    assert!(
        s.obs_times[j] < 24.0,
        "moved: engine time {}",
        s.obs_times[j]
    );

    // pkonly: the EVID=2 row at 24 runs below 24; the observation there is not the one moved.
    let p = pop("pkonly");
    let s = &p.subjects[0];
    let m = (0..s.pk_only_times.len())
        .find(|&m| s.param_time_pk_only(m) == 24.0)
        .expect("pkonly: an EVID=2 row written at 24");
    assert!(
        s.pk_only_times[m] < 24.0,
        "pkonly: engine time {}",
        s.pk_only_times[m]
    );

    // reset: period 2's records run past 73 but were written from 0.
    let p = pop("reset");
    let s = &p.subjects[0];
    let r = *s.reset_times.last().expect("reset: an EVID=4 row");
    assert!(r >= 73.0, "reset: period 2 engine start {r}");
    assert_eq!(s.param_time_at(r), 0.0, "reset: period 2 written start");
    let j = (0..s.obs_times.len())
        .find(|&j| s.obs_times[j] > r && s.param_time_obs(j) < 24.0)
        .expect("reset: a period-2 observation written before 24");
    assert!(s.obs_times[j] >= 24.0);

    // reset_f: the period-2 dose at 0 runs past 24.
    let p = pop("reset_f");
    let s = &p.subjects[0];
    let d = s
        .doses
        .iter()
        .find(|d| d.time >= 24.0 && s.param_time_at(d.time) == 0.0)
        .expect("reset_f: a period-2 dose written at 0");
    assert!(d.time >= 24.0, "reset_f: engine time {}", d.time);

    // reset_seed: the reset row, which seeds `init`, runs past 24 but was written at 0.
    let p = pop("reset_seed");
    let s = &p.subjects[0];
    let r = *s.reset_times.last().expect("reset_seed: an EVID=4 row");
    assert!(r >= 24.0, "reset_seed: engine time {r}");
    assert_eq!(s.param_time_at(r), 0.0, "reset_seed: written time");

    // control: every record's engine time is its written time.
    let p = pop("control");
    for s in &p.subjects {
        for j in 0..s.obs_times.len() {
            assert_eq!(s.obs_times[j], s.param_time_obs(j), "control: obs {j}");
        }
        for m in 0..s.pk_only_times.len() {
            assert_eq!(
                s.pk_only_times[m],
                s.param_time_pk_only(m),
                "control: pk-only {m}"
            );
        }
        for d in &s.doses {
            assert_eq!(
                d.time,
                s.param_time_at(d.time),
                "control: dose at {}",
                d.time
            );
        }
        assert!(s.reset_times.is_empty() && s.reset_period_times.is_empty());
    }
}

fn run_population(case: &str) -> Population {
    let engine = if case == "reset_seed" {
        "ode"
    } else {
        "analytic"
    };
    prepare_run(
        &anchor(&format!("param_time_{case}_{engine}.ferx")),
        Some(&anchor(&format!("param_time_{case}.csv"))),
    )
    .unwrap_or_else(|e| panic!("{case}: load: {e}"))
    .population
}
