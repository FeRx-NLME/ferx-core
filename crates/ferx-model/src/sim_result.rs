//! The result of a simulation run. Its own crate-level module, not under `api`, so the endpoint
//! simulators below `api` can build one (#3041); `api` re-exports it at its public path.
use crate::types::SimOutcome;

/// A single simulated observation.
///
/// `draw` is the uncertainty draw index (1-based). For `simulate()` /
/// `simulate_with_seed()`, which use point-estimate parameters, `draw` is
/// always `1`. For `simulate_with_uncertainty()` it spans
/// `1..=n_uncertainty_draws`. `sim` is the replicate index *within* a draw.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub draw: usize,
    pub sim: usize,
    pub id: String,
    /// For Gaussian rows: the scheduled observation time from the subject's grid.
    /// For TTE rows: the sampled event time (equals `SimOutcome::Event::time`; the
    /// outer field exists for uniform iteration without matching on `outcome`).
    pub time: f64,
    /// CMT column value for this observation row. For Gaussian subjects this mirrors the data
    /// file's CMT (e.g. 1 for a central-compartment PK endpoint — not necessarily 0). For TTE
    /// rows (requires `survival` feature) it matches the `[event_model] cmt` declaration.
    pub cmt: usize,
    /// Individual prediction at η (Gaussian path only; NAN for non-Gaussian).
    pub ipred: f64,
    /// Simulated observation outcome.  For Gaussian: `SimOutcome::Continuous { value }`.
    /// For TTE (requires `survival` feature): `SimOutcome::Event { time, observed }`.
    pub outcome: SimOutcome,
    /// The mixture class (`MIXNUM`, 1-based) this row's subject was simulated under in its
    /// `(draw, sim)` replicate (#2372): `Some(k)` on a `[mixture]` model, `None` otherwise.
    /// NONMEM's `$TABLE MIXNUM` under `$SIMULATION`. Every row of one subject in one
    /// replicate carries the same class; the class is redrawn per replicate (#2163).
    pub mixnum: Option<usize>,
}
