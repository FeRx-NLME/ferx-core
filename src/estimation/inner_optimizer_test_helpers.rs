//! IOV fixtures and the warm-start assertion the #2217 / #2261 tests share across modules:
//! the outer optimizer's caches, the covariance step, the profile solve, SAEM, VI, SIR and
//! the reconverged finite-difference gradients all check, through
//! [`assert_starts_at_modes`], that an IOV inner solve began at the mode it was handed.

use nalgebra::DVector;

use super::{InnerRecording, IovStart};
use crate::types::{CompiledModel, FitOptions, Optimizer, Population};

/// A model file and its data, loaded as `fit` would, cut to the first `n_subj` subjects.
pub(crate) fn load_iov_fixture(
    src: &str,
    data: &str,
    n_subj: usize,
) -> (CompiledModel, Population, FitOptions) {
    let parsed = crate::parser::model_parser::parse_full_model(src).expect("model parses");
    let (mut pop, _) = crate::api::read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        data,
        None,
        parsed.fit_options.iov_column.as_deref(),
        None,
        &parsed.column_map,
    )
    .expect("data loads");
    pop.subjects.truncate(n_subj);
    // `fit`'s preparation of a nested-level or serially-correlated population (each a
    // no-op otherwise); `optimize_population` is called below it.
    crate::stats::nested_levels::assign_level_cells(&parsed.model, &mut pop);
    crate::stats::residual_error::assign_residual_groups(&parsed.model, &mut pop);
    let opts = FitOptions {
        optimizer: Optimizer::Bfgs,
        outer_maxiter: 1,
        run_covariance_step: false,
        verbose: false,
        ..parsed.fit_options
    };
    (parsed.model, pop, opts)
}

/// `warfarin_iov` (one-level `iov_column`), plus a copy of its first subject with no
/// occasion labels, which an IOV model solves for η alone.
pub(crate) fn warfarin_iov_fixture() -> (CompiledModel, Population, FitOptions) {
    let (model, mut pop, opts) = load_iov_fixture(
        include_str!("../../examples/warfarin_iov.ferx"),
        "data/warfarin_iov.csv",
        2,
    );
    let mut no_occ = pop.subjects[0].clone();
    no_occ.id = "NO_OCC".into();
    no_occ.occasions.clear();
    no_occ.dose_occasions.clear();
    no_occ.reset_occasions.clear();
    no_occ.pk_only_occasions.clear();
    pop.subjects.push(no_occ);
    (model, pop, opts)
}

/// [`warfarin_iov_fixture`] with its closed form written as `[odes]`: an ODE + IOV model,
/// whose outer caches restart κ at 0 (#2508).
pub(crate) fn warfarin_iov_ode_fixture() -> (CompiledModel, Population, FitOptions) {
    let src = include_str!("../../examples/warfarin_iov.ferx");
    let pk = "  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n";
    assert_eq!(src.matches(pk).count(), 1, "anchor {pk:?} moved");
    let src = src.replacen(
        pk,
        "  ode(states=[depot, central])\n\n[odes]\n  d/dt(depot)   = -KA * depot\n  \
         d/dt(central) =  KA * depot - (CL/V) * central\n\n[scaling]\n  y = central / V\n",
        1,
    );
    let (model, mut pop, opts) = load_iov_fixture(&src, "data/warfarin_iov.csv", 2);
    assert!(
        model.ode_spec.is_some() && model.n_kappa > 0,
        "fixture: must be an ODE + IOV model"
    );
    let mut no_occ = pop.subjects[0].clone();
    no_occ.id = "NO_OCC".into();
    no_occ.occasions.clear();
    no_occ.dose_occasions.clear();
    no_occ.reset_occasions.clear();
    no_occ.pk_only_occasions.clear();
    pop.subjects.push(no_occ);
    (model, pop, opts)
}

/// `nested_levels_3` (exact3): study → arm nested below the subject.
pub(crate) fn nested_levels_fixture() -> (CompiledModel, Population, FitOptions) {
    load_iov_fixture(
        include_str!("../../nonmem_anchor/nested_levels_3.ferx"),
        "nonmem_anchor/nested_levels_3.csv",
        3,
    )
}

/// `warfarin_iov` with a residual `correlation ar1(RHO) by OCC` line: kappa and a serial
/// residual correlation on one model. Its optimizer leg takes 8 s for one subject in a
/// debug build, so its test drives the first evaluation directly.
pub(crate) fn kappa_correlation_fixture() -> (CompiledModel, Population, FitOptions) {
    let src = include_str!("../../examples/warfarin_iov.ferx");
    let (sigma, error) = (
        "  sigma PROP_ERR ~ 0.2 (sd)\n",
        "  DV ~ proportional(PROP_ERR)\n",
    );
    for anchor in [sigma, error] {
        assert_eq!(src.matches(anchor).count(), 1, "anchor {anchor:?} moved");
    }
    let src = src
        .replacen(sigma, &format!("{sigma}  theta RHO(0.3, -0.99, 0.99)\n"), 1)
        .replacen(error, &format!("{error}  correlation ar1(RHO) by OCC\n"), 1);
    let (model, pop, opts) = load_iov_fixture(&src, "data/warfarin_iov.csv", 3);
    assert!(
        model.residual_serial.is_some(),
        "fixture: the correlation line must reach the model"
    );
    // ...and the data: with every `obs_rgroup` 0 the correlation is inert and this is
    // plain `warfarin_iov` again (measured: 11 rows in group 1, 11 in group 2).
    for s in &pop.subjects {
        let groups: std::collections::BTreeSet<u32> =
            s.obs_rgroup.iter().copied().filter(|&g| g > 0).collect();
        assert!(
            groups.len() >= 2,
            "fixture: subject {} has residual groups {:?}; the correlation is inert",
            s.id,
            s.obs_rgroup
        );
    }
    (model, pop, opts)
}

/// `f` on a private one-thread rayon pool. Both sides of every #2217 bit-identity twin run
/// through here, so neither shares the global pool with concurrently running tests (the
/// in-process IOV nondeterminism of #1197 needs such a co-runner). [`record_inner_solves`]
/// sees only this thread's solves, so every recording runs here too.
///
/// [`record_inner_solves`]: super::record_inner_solves
pub(crate) fn on_one_thread<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .stack_size(crate::FIT_RAYON_STACK_SIZE)
        .build()
        .unwrap()
        .install(f)
}

/// #2261: every start of `rec` that `select` picks began at its subject's mode, bit for
/// bit — the BSV block at `etas[i] + μ` and every cell at `kappas[i]` — and the number of
/// such IOV starts, which must be positive (a selection that picks nothing tests nothing).
/// A subject with no κ cells is skipped. A fixture whose κ̂ are all 0 is refused: there a
/// warm start that dropped the κ block would start at the same point.
///
/// `label` names the call site, so a failure says which site lost κ̂.
pub(crate) fn assert_starts_at_modes(
    label: &str,
    rec: &InnerRecording,
    select: impl Fn(&IovStart) -> bool,
    pop: &Population,
    etas: &[DVector<f64>],
    kappas: &[Vec<DVector<f64>>],
) -> usize {
    assert_these_start_at_modes(
        label,
        rec.starts.iter().filter(|s| select(s)),
        pop,
        etas,
        kappas,
    )
}

/// #2261, the invariant over a whole fit: every population pass of `rec` that is handed
/// warm starts (not all zero) hands each IOV subject (one whose EBE has κ cells) `η̂` then every cell's κ̂,
/// and the subject's IOV solve in that pass starts at that κ block, bit for bit. Returns the
/// number of such (pass, subject) pairs, which must be positive. A site that hands `η̂`
/// alone fails here, naming the pass.
pub(crate) fn assert_every_warm_pass_is_joint(
    label: &str,
    rec: &InnerRecording,
    model: &CompiledModel,
) -> usize {
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let n_eta = model.n_eta;
    let mut checked = 0;
    for p in &rec.passes {
        let Some(init) = &p.init else { continue };
        // A cold start spelled as zeros (the identity-scale pre-flight's `cold_etas`)
        // carries no mode to lose.
        if p.kappas.is_empty() || init.iter().all(|&v| v == 0.0) {
            continue;
        }
        let joint = n_eta + p.kappas.iter().map(Vec::len).sum::<usize>();
        assert_eq!(
            init.len(),
            joint,
            "{label}: pass {} handed subject {} a warm start of width {}, not η̂ and κ̂ \
             ({joint}): the κ block was dropped",
            p.pass,
            p.subject,
            init.len()
        );
        let start = rec
            .starts
            .iter()
            .find(|s| s.pass == Some(p.pass) && s.subject == p.subject)
            .unwrap_or_else(|| {
                panic!(
                    "{label}: pass {} subject {}: no IOV start",
                    p.pass, p.subject
                )
            });
        let got: Vec<u64> = start.kappas.iter().flat_map(|c| bits(c)).collect();
        assert_eq!(
            got,
            bits(&init[n_eta..]),
            "{label}: pass {} subject {}: κ block of the start is not the κ̂ it was handed",
            p.pass,
            p.subject
        );
        checked += 1;
    }
    assert!(checked > 0, "{label}: no warm pass reached an IOV subject");
    checked
}

/// #2261: every inner solve of `rec` for a subject `select` picks ran at the stencil
/// tolerance `stencil_inner_tol(stage_tol)`, and `stage_tol` is looser than it, so a stencil
/// that dropped the tightening would show. Returns the number of solves, which must be
/// positive.
pub(crate) fn assert_solves_at_stencil_tol(
    label: &str,
    rec: &InnerRecording,
    select: impl Fn(&str) -> bool,
    stage_tol: f64,
) -> usize {
    let want = super::stencil_inner_tol(stage_tol);
    assert!(
        want < stage_tol,
        "{label}: fixture: the stage tolerance {stage_tol:e} is already at the stencil's"
    );
    let mut n = 0;
    for (subject, tol) in rec.solve_tols.iter().filter(|(s, _)| select(s)) {
        assert_eq!(
            tol.to_bits(),
            want.to_bits(),
            "{label}: subject {subject} re-solved at {tol:e}, not the stencil tolerance {want:e}"
        );
        n += 1;
    }
    assert!(n > 0, "{label}: no solve was selected");
    n
}

/// #2261, for a call whose re-solves warm-start from modes no single list holds (a profile
/// solve's line-search trials, say): every start `select` picks began, bit for bit, at the
/// mode some earlier IOV solve of the same subject returned — η̂ + μ and every cell's κ̂.
/// Returns the number checked, which must be positive. A start whose η block matches an
/// earlier mode while its κ block matches none is the defect this exists to catch, and the
/// message says so.
pub(crate) fn assert_starts_at_previous_modes(
    label: &str,
    rec: &InnerRecording,
    select: impl Fn(&IovStart) -> bool,
) -> usize {
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let mut checked = 0;
    for s in rec.starts.iter().filter(|s| select(s)) {
        let earlier: Vec<_> = rec.ends[..s.ends_before]
            .iter()
            .filter(|e| e.subject == s.subject)
            .collect();
        let eta_matches: Vec<_> = earlier
            .iter()
            .filter(|e| {
                let psi: Vec<u64> = e
                    .eta
                    .iter()
                    .zip(&s.mu)
                    .map(|(a, m)| (a + m).to_bits())
                    .collect();
                psi == bits(&s.psi)
            })
            .collect();
        assert!(
            !eta_matches.is_empty(),
            "{label}: subject {}: η block of the start {:?} is no earlier mode's η̂ + μ",
            s.subject,
            s.psi
        );
        let got: Vec<Vec<u64>> = s.kappas.iter().map(|c| bits(c)).collect();
        let hit = eta_matches
            .iter()
            .find(|e| e.kappas.iter().map(|c| bits(c)).collect::<Vec<_>>() == got);
        let Some(hit) = hit else {
            panic!(
                "{label}: subject {}: κ block of the start {:?} is not the κ̂ of the mode its η \
                 block came from ({:?})",
                s.subject,
                s.kappas,
                eta_matches.iter().map(|e| &e.kappas).collect::<Vec<_>>()
            );
        };
        assert!(
            hit.kappas.iter().flatten().any(|&v| v != 0.0),
            "{label}: subject {}: fixture: the mode's κ̂ are all 0, so a dropped κ block \
             cannot show",
            s.subject
        );
        checked += 1;
    }
    assert!(checked > 0, "{label}: no IOV start was selected");
    checked
}

/// [`assert_starts_at_modes`] over an explicit list of starts, for a caller that selects
/// them by position.
pub(crate) fn assert_these_start_at_modes<'a>(
    label: &str,
    starts: impl IntoIterator<Item = &'a IovStart>,
    pop: &Population,
    etas: &[DVector<f64>],
    kappas: &[Vec<DVector<f64>>],
) -> usize {
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let mut checked = 0;
    for s in starts {
        let i = pop
            .subjects
            .iter()
            .position(|p| p.id == s.subject)
            .unwrap_or_else(|| panic!("{label}: start for unknown subject {}", s.subject));
        let want_kappas = &kappas[i];
        if want_kappas.is_empty() {
            continue;
        }
        assert!(
            want_kappas.iter().flatten().any(|&v| v != 0.0),
            "{label}: subject {}: fixture: every κ̂ is 0, so a dropped κ block cannot show",
            s.subject
        );
        let want_psi: Vec<u64> = etas[i]
            .iter()
            .zip(&s.mu)
            .map(|(e, m)| (e + m).to_bits())
            .collect();
        assert_eq!(
            bits(&s.psi),
            want_psi,
            "{label}: subject {}: η block of the start {:?} is not η̂ {:?} + μ {:?}",
            s.subject,
            s.psi,
            etas[i].as_slice(),
            s.mu
        );
        let got: Vec<Vec<u64>> = s.kappas.iter().map(|c| bits(c)).collect();
        let want: Vec<Vec<u64>> = want_kappas.iter().map(|c| bits(c.as_slice())).collect();
        assert_eq!(
            got,
            want,
            "{label}: subject {}: κ block of the start {:?} is not κ̂ {:?}",
            s.subject,
            s.kappas,
            want_kappas.iter().map(|c| c.as_slice()).collect::<Vec<_>>()
        );
        checked += 1;
    }
    assert!(checked > 0, "{label}: no IOV start was selected");
    checked
}
