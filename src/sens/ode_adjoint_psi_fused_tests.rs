//! The quadrature gradient's Ψ θ-adjoint read off A's and the mode response rows'
//! (`agq_taped`): `∂Ψ/∂θ = ∂A/∂θ − Σ_r (∇_η A)_r ∂y_r/∂θ`, against Ψ's own sweep
//! (`PSI_SWEEP_UNFUSED`). Each case asserts which route it took (`PSI_FUSED_HITS`), so the
//! fused read is what the parity covers and IOV is what the fallback covers.

use super::tests::{ebe_for, moved_params, subject_shell};
use super::*;
use crate::estimation::parameterization::pack_params;
use crate::parser::model_parser::parse_model_string;
use crate::types::DoseEvent;

const HEAD: &str = "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  theta TVV(30.0, 1.0, 500.0)\n  \
    theta TVKA(1.2, 0.05, 20.0)\n  block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.06]\n  \
    omega ETA_KA ~ 0.2\n";

fn model_src(iov: bool, ode: bool) -> String {
    let (kappa, kterm, fit) = if iov {
        (
            "  kappa KAPPA_CL ~ 0.04\n",
            " + KAPPA_CL",
            "[fit_options]\n  iov_column = OCC\n",
        )
    } else {
        ("", "", "")
    };
    let structural = if ode {
        "[structural_model]\n  ode(states=[depot, central])\n[odes]\n  d/dt(depot) = -KA * depot\n  \
         d/dt(central) = KA * depot - (CL/V) * central\n[scaling]\n  y = central / V\n"
    } else {
        "[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n"
    };
    format!(
        "{HEAD}{kappa}  sigma PROP_ERR ~ 0.1 (sd)\n  sigma ADD_ERR ~ 0.05 (sd)\n[individual_parameters]\n  \
         CL = TVCL * exp(ETA_CL{kterm})\n  V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n\
         {structural}[error_model]\n  DV ~ combined(PROP_ERR, ADD_ERR)\n{fit}"
    )
}

fn subject_for(model: &CompiledModel) -> Subject {
    let times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.5, 14.0, 18.0, 24.0];
    let mut s = subject_shell(
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &times,
    );
    if model.n_kappa > 0 {
        s.occasions = times
            .iter()
            .map(|&t| if t < 12.0 { 1 } else { 2 })
            .collect();
        s.dose_occasions = vec![1, 2];
    }
    let eta_ref: Vec<f64> = vec![0.1, -0.05, 0.2, 0.05][..model.n_eta + model.n_kappa].to_vec();
    let preds =
        crate::pk::compute_predictions_with_tv(model, &s, &model.default_params.theta, &eta_ref);
    s.observations = preds
        .iter()
        .enumerate()
        .map(|(j, p)| p * (1.0 + 0.15 * (1.3 * j as f64).sin()))
        .collect();
    s
}

fn worst_rel(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let gaps: Vec<f64> = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / (1.0 + y.abs()))
        .collect();
    assert!(gaps.iter().all(|e| e.is_finite()), "{a:?} vs {b:?}");
    gaps.into_iter().fold(0.0, f64::max)
}

/// Laplace (exact anchor, one node) and AGQ-3 under the FOCEI anchor, closed form and ODE:
/// the fused Ψ adjoint equals Ψ's own sweep, the mode response is untouched, and IOV —
/// whose rows are the BSV block only — keeps the sweep.
#[test]
fn psi_read_off_a_and_the_response_rows_matches_its_own_sweep() {
    for (iov, ode, objective) in [
        (
            false,
            false,
            Objective::Agq {
                n_agq: 1,
                exact: true,
            },
        ),
        (
            false,
            true,
            Objective::Agq {
                n_agq: 1,
                exact: true,
            },
        ),
        (
            false,
            false,
            Objective::Agq {
                n_agq: 3,
                exact: false,
            },
        ),
        (
            true,
            false,
            Objective::Agq {
                n_agq: 1,
                exact: true,
            },
        ),
    ] {
        let model = super::tests::tight(parse_model_string(&model_src(iov, ode)).unwrap());
        let s = subject_for(&model);
        let params = moved_params(&model);
        let x = pack_params(&params);
        let eta = ebe_for(&model, &s, &params, 1e-12, true);
        let pt = PackedPoint::new(&params, &x).unwrap();
        let go =
            || subject_adjoint_gradient_and_eta_dx_at(&model, &s, &pt, eta.as_slice(), objective);
        PSI_FUSED_HITS.with(|c| c.set(0));
        let (g_on, d_on) = go();
        let hits = PSI_FUSED_HITS.with(|c| c.get());
        PSI_SWEEP_UNFUSED.with(|c| c.set(true));
        let (g_off, d_off) = go();
        PSI_SWEEP_UNFUSED.with(|c| c.set(false));
        let case = format!("iov={iov} ode={ode} {objective:?}");
        assert_eq!(
            hits > 0,
            !iov,
            "{case}: fused only where the rows cover every effect"
        );
        let (g_on, g_off) = (g_on.expect("in scope"), g_off.expect("in scope"));
        let w = worst_rel(&g_on, &g_off);
        // Same quantity by two summation orders: measured worst 2.9e-15 (Laplace, closed
        // form), so 1e-13 keeps ~35× headroom.
        assert!(w < 1e-13, "{case}: worst gradient gap {w:.2e}");
        let (d_on, d_off) = (d_on.expect("response"), d_off.expect("response"));
        for (a, b) in d_on.iter().zip(&d_off) {
            assert_eq!(a, b, "{case}: the response is the same sweeps either way");
        }
    }
}
