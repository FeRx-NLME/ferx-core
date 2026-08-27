use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

// Re-export the milestone-2 sensitivity partials placeholder so
// `CompiledModel` can hold one and external callers (test fixtures and the
// `generate_data` data-generation binary) can construct an empty one. The
// inner `Expression` AST stays parser-private — only `IndivParamPartials::empty`
// and the `Debug`/`Clone` derives are reachable from outside the crate.
pub use crate::parser::model_parser::IndivParamPartials;
pub use crate::parser::model_parser::ThetaBlocks;

/// How a dose's infusion `rate`/`duration` are determined.
///
/// NONMEM overloads the `RATE` column with negative codes that make the
/// infusion **parameter-driven** rather than data-driven (see
/// [`crate::io`] data-format docs):
///   - `RATE = -2` → the infusion *duration* is the model parameter `D{cmt}`
///     ([`RateMode::ModeledDuration`]); the rate is then `amt / duration`.
///   - `RATE = -1` → the infusion *rate* is the model parameter `R{cmt}`
///     ([`RateMode::ModeledRate`]); the duration is then `amt / rate`.
///
/// The modeled values are not known at parse/read time (they depend on the
/// per-iteration `theta`/`eta`/covariates), so a coded dose stores its mode
/// here and is resolved to a concrete ([`RateMode::Fixed`]) dose per iteration
/// by [`DoseEvent::resolve_rate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RateMode {
    /// `RATE ≥ 0`: `rate`/`duration` are the literal stored values. The default
    /// keeps every existing `DoseEvent` construction (and serialized data)
    /// behaving exactly as before.
    #[default]
    Fixed,
    /// `RATE = -2`: infusion duration is the modeled parameter `D{cmt}` resolved
    /// from the dose compartment via the model's `DoseAttrMap`.
    ModeledDuration,
    /// `RATE = -1`: infusion *rate* is the modeled parameter `R{cmt}` resolved
    /// from the dose compartment via the model's `DoseAttrMap`. The duration is
    /// then `amt / rate` (the NONMEM-faithful mirror of [`Self::ModeledDuration`]).
    ModeledRate,
}

/// How an infusion's `(rate, duration)` was *specified*, which fixes how
/// bioavailability `F` reshapes it (NONMEM convention; issue #419).
///
/// `F` always delivers the same total exposure `F·amt`, but for `F ≠ 1` it keeps
/// one of `(rate, duration)` fixed and scales the other:
///   - [`Self::RateDefined`] (`RATE>0` data **and** `RATE=-1` → `R{cmt}`): hold
///     the rate, scale the duration to `F·amt/rate`.
///   - [`Self::DurationDefined`] (`RATE=-2` → `D{cmt}`): hold the duration, scale
///     the rate to `F·amt/duration` (ferx's original behaviour for every infusion,
///     correct only for this case).
///
/// Unlike [`RateMode`], this tag is **persistent** — it is *not* consumed by
/// [`DoseEvent::resolve_rate`] (which collapses the mode to [`RateMode::Fixed`]),
/// because the `F` rule is applied downstream, after resolution. It is the single
/// piece of state that lets [`DoseEvent::bioavailable_infusion`] stay the one
/// source of truth across every prediction path. Only meaningful for infusions; a
/// bolus never reads it. Defaults to [`Self::RateDefined`] — the NONMEM default and
/// the correct value for every `RATE>0` dose and every synthetic infusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InfusionDef {
    /// Rate-specified: `RATE>0` (data) and `RATE=-1` (`R{cmt}`). `F` scales the
    /// duration.
    #[default]
    RateDefined,
    /// Duration-specified: `RATE=-2` (`D{cmt}`). `F` scales the rate.
    DurationDefined,
}

/// Clamp `x` to a lower `floor`: returns `x` when `x > floor`, otherwise `floor`.
///
/// The single home for the "pull a transient mid-fit excursion back off the
/// domain wall" clamp shared by [`DoseEvent::DURATION_FLOOR`] (modeled infusion
/// duration `D ≤ 0`) and [`crate::pk::absorption::PreparedInputRate::MIN_PARAM`]
/// (transit `mtt`, inverse-Gaussian `mat`/`cv2` ≤ 0). Both keep a downstream `amt/D` /
/// `ktr.ln()` finite at the wall so the optimiser can climb back to the interior
/// without perturbing a converged (interior) fit. Centralising it keeps the
/// `NaN`-falls-to-floor subtlety in one place — every `>` is false for `NaN`, so
/// a `NaN` input also returns `floor`. The explicit comparison (not `f64::max`)
/// is what makes the `NaN`-to-`floor` behaviour deliberate rather than incidental.
#[inline]
pub(crate) fn clamp_above_floor(x: f64, floor: f64) -> f64 {
    if x > floor {
        x
    } else {
        floor
    }
}

/// A single dose event (bolus, infusion, or oral)
#[derive(Debug, Clone)]
pub struct DoseEvent {
    pub time: f64,
    pub amt: f64,
    /// 1-based target compartment as authored (`CMT`), with `CMT=0` meaning
    /// NONMEM's *default dose compartment* (compartment 1). **Private on purpose**
    /// (#912): read it through [`Self::cmt_idx`] (0-based state index),
    /// [`Self::cmt_1based`] (1-based compartment, `0`→`1`), or [`Self::cmt_raw`]
    /// (the literal authored value). Keeping the field private is what makes the
    /// recurring `cmt - 1` underflow bug (#899) unrepresentable at the call site —
    /// a bare `- 1` cannot be written against it.
    cmt: usize,
    pub rate: f64,
    pub duration: f64,
    pub ss: bool,
    pub ii: f64,
    /// How `rate`/`duration` are determined. [`RateMode::Fixed`] for ordinary
    /// (data-driven) doses; a modeled variant for a NONMEM coded `RATE`, which
    /// is resolved per iteration by [`Self::resolve_rate`].
    pub rate_mode: RateMode,
    /// How this infusion was *specified*, which fixes how bioavailability `F`
    /// reshapes it (see [`InfusionDef`] and [`Self::bioavailable_infusion`], #419).
    /// Persistent across [`Self::resolve_rate`]. Only read for infusions.
    pub infusion_def: InfusionDef,
}

impl DoseEvent {
    pub fn new(time: f64, amt: f64, cmt: usize, rate: f64, ss: bool, ii: f64) -> Self {
        let duration = if rate > 0.0 { amt / rate } else { 0.0 };
        Self {
            time,
            amt,
            cmt,
            rate,
            duration,
            ss,
            ii,
            rate_mode: RateMode::Fixed,
            // A data-driven dose is rate-specified (`RATE>0`); a bolus never reads
            // this. Either way `RateDefined` is the correct default (#419).
            infusion_def: InfusionDef::RateDefined,
        }
    }

    /// Construct a dose whose infusion `rate`/`duration` are *modeled* (a NONMEM
    /// coded `RATE`). The concrete `rate`/`duration` are unknown until the
    /// per-iteration parameters are available, so they are left at `0.0` and
    /// filled in by [`Self::resolve_rate`]; until then [`Self::is_infusion`]
    /// still reports `true` from the mode.
    pub fn modeled(time: f64, amt: f64, cmt: usize, ss: bool, ii: f64, mode: RateMode) -> Self {
        // The definition tag mirrors the coded mode and persists past
        // `resolve_rate` (which clears `rate_mode` to `Fixed`): `RATE=-1`/`R{cmt}`
        // is rate-specified, `RATE=-2`/`D{cmt}` is duration-specified (#419).
        let infusion_def = match mode {
            RateMode::ModeledDuration => InfusionDef::DurationDefined,
            RateMode::Fixed | RateMode::ModeledRate => InfusionDef::RateDefined,
        };
        Self {
            time,
            amt,
            cmt,
            rate: 0.0,
            duration: 0.0,
            ss,
            ii,
            rate_mode: mode,
            infusion_def,
        }
    }

    /// Domain floor for a modeled infusion `duration` when clamping a transient
    /// mid-fit excursion (see [`Self::resolve_rate`]). Mirrors
    /// [`crate::pk::absorption::PreparedInputRate::MIN_PARAM`]: far below any
    /// realistic duration, so it never perturbs a converged fit — it only keeps
    /// a transient `D ≤ 0` (or `NaN`) from turning `amt / D` into a non-finite
    /// rate. `NaN` falls to the floor (every `>` is false for `NaN`).
    pub(crate) const DURATION_FLOOR: f64 = 1e-8;

    /// Domain floor for a modeled infusion `rate` (`RATE = -1` → `R{cmt}`), the
    /// mirror of [`Self::DURATION_FLOOR`]. A transient `R ≤ 0` (or `NaN`)
    /// mid-search would otherwise make `amt / R` (the implied duration)
    /// non-finite; clamping `R` to this floor keeps it finite (delivering the
    /// dose over a very long duration) without perturbing a converged fit, whose
    /// optimum is interior. `NaN` falls to the floor (every `>` is false for
    /// `NaN`).
    pub(crate) const RATE_FLOOR: f64 = 1e-8;

    /// Resolve a modeled-`RATE` dose into a concrete ([`RateMode::Fixed`]) dose
    /// for this iteration's per-dose `PkParams` (`params` = `PkParams::values`).
    ///
    /// **Single source of truth** for the modeled-`RATE` rule. Every prediction
    /// entrypoint maps its doses through this *before* integrating, so all
    /// downstream machinery (ODE forcing, SS equilibration, the break-time
    /// timeline) sees only a concrete `rate`/`duration` and a new dose-application
    /// path cannot silently diverge — the recurring failure mode that F (#327),
    /// lag (#369), and duration/rate (#324) each had to thread through every path.
    ///
    /// It is **`F`-agnostic**: it derives `(rate, duration)` from the *raw*
    /// `amt`, leaving bioavailability to the existing per-compartment `F`
    /// machinery downstream (so `F` is applied exactly once — `F·amt` delivered
    /// over `D`, matching NONMEM's `F·RATE` for an infusion).
    ///
    /// f64-only / FD-only by construction. The ODE engine has no analytic-
    /// sensitivity path, and the analytical engine routes any subject with a modeled dose to FD
    /// (see `resolve_gradient_method`, #394) precisely because resolving a duration
    /// or rate here would drop its `∂/∂η`, so no `Dual` twin is ever needed. A
    /// transient `D ≤ 0` is clamped to [`Self::DURATION_FLOOR`]; a transient
    /// `R ≤ 0` to [`Self::RATE_FLOOR`].
    pub(crate) fn resolve_rate(&self, attr_map: &DoseAttrMap, params: &[f64]) -> DoseEvent {
        match self.rate_mode {
            RateMode::Fixed => self.clone(),
            RateMode::ModeledDuration => {
                // The slot's existence is an invariant enforced by
                // `check_model_data` (a `RATE=-2` dose with no matching `D{cmt}`
                // is rejected before any prediction runs).
                let slot = attr_map
                    .indexed_slot(DoseAttr::Duration, self.cmt)
                    .expect("modeled-duration dose slot validated by check_model_data");
                let d_raw = params.get(slot).copied().unwrap_or(0.0);
                let duration = clamp_above_floor(d_raw, Self::DURATION_FLOOR);
                DoseEvent {
                    rate: self.amt / duration,
                    duration,
                    rate_mode: RateMode::Fixed,
                    ..self.clone()
                }
            }
            RateMode::ModeledRate => {
                // Mirror of `ModeledDuration`: the `R{cmt}` slot's existence is an
                // invariant enforced by `check_model_data` (a `RATE=-1` dose with
                // no matching `R{cmt}` is rejected before any prediction runs).
                let slot = attr_map
                    .indexed_slot(DoseAttr::Rate, self.cmt)
                    .expect("modeled-rate dose slot validated by check_model_data");
                let r_raw = params.get(slot).copied().unwrap_or(0.0);
                let rate = clamp_above_floor(r_raw, Self::RATE_FLOOR);
                DoseEvent {
                    rate,
                    duration: self.amt / rate,
                    rate_mode: RateMode::Fixed,
                    ..self.clone()
                }
            }
        }
    }

    pub fn is_infusion(&self) -> bool {
        // A modeled-duration dose is an infusion even before `resolve_rate` fills
        // in the concrete `rate` (which is `0.0` until then).
        self.rate > 0.0 || !self.is_fixed()
    }

    /// 0-based state-vector index for this dose's target compartment.
    ///
    /// `CMT` is 1-based, and `CMT=0` is NONMEM's *default dose compartment* —
    /// compartment 1 — so both `CMT=0` and `CMT=1` map to index `0` (#899, #912).
    /// This is the single named home for the `cmt → state index` conversion:
    /// before it, ~a dozen sites open-coded `cmt - 1` (which underflows on
    /// `CMT=0`), `cmt.saturating_sub(1)`, or `if cmt >= 1 { cmt - 1 }` (which
    /// silently drops `CMT=0`), and they disagreed — the four-behaviour bug #899
    /// set out to remove. Index a state vector, or match a 0-based
    /// `InputRateForcing::cmt`, through this — never a bare `- 1`.
    #[inline]
    pub fn cmt_idx(&self) -> usize {
        self.cmt.saturating_sub(1)
    }

    /// 1-based target compartment with `CMT=0` resolved to compartment 1 (#899).
    ///
    /// The 1-based complement of [`Self::cmt_idx`], for comparing against a
    /// 1-based compartment number or a set of them (`f.cmt + 1`, `ALAG{n}`'s
    /// index). `CMT=0` and `CMT=1` both return `1`. Use this — not the raw
    /// `self.cmt` — whenever a `CMT=0` dose must be treated as compartment 1,
    /// otherwise the default dose compartment silently misses the lookup.
    #[inline]
    pub fn cmt_1based(&self) -> usize {
        self.cmt.max(1)
    }

    /// The literal authored `CMT`, `0` included — the escape hatch for the few
    /// sites that need the raw value rather than a resolved index: detecting a
    /// `CMT=0` dose (`d.cmt_raw() == 0`), echoing it in a diagnostic, or passing
    /// it to a callee that performs its own `0`→`1` mapping (e.g.
    /// [`DoseAttrMap::indexed_slot`], which does `cmt.max(1)` internally). Prefer
    /// [`Self::cmt_idx`] or [`Self::cmt_1based`]; reach for this only when the
    /// *unresolved* value is what you actually mean.
    ///
    /// `pub(crate)` on purpose (#912 review): the raw value is the one that can be
    /// mis-decremented (`cmt_raw() - 1` re-opens the underflow the private field
    /// closes), so it is kept off the public surface where adversarial review does
    /// not reach. The public read API is the two *resolved* accessors above.
    #[inline]
    pub(crate) fn cmt_raw(&self) -> usize {
        self.cmt
    }

    /// True when this dose's `rate`/`duration` are concrete (data-driven), i.e.
    /// [`RateMode::Fixed`] — either an ordinary dose or one already passed through
    /// [`Self::resolve_rate`]. False for a still-modeled NONMEM coded `RATE`.
    ///
    /// **Single source of truth** for "is this dose resolved?". Every prediction
    /// path that snapshots `rate`/`duration` (the ODE resolve shadows, and the
    /// analytical / AD tripwires) tests this rather than re-spelling the
    /// `matches!(rate_mode, Fixed)` predicate, so the second modeled variant
    /// (`RATE=-1` → `Rn`, #324) changed "resolved" in exactly one place instead
    /// of across every dose-application site.
    pub fn is_fixed(&self) -> bool {
        matches!(self.rate_mode, RateMode::Fixed)
    }

    /// Bioavailable `(rate, duration)` for this **resolved** infusion under
    /// bioavailability `f_bio` (issue #419).
    ///
    /// **Single source of truth** for the mode-aware `F`-on-infusion rule. NONMEM
    /// keeps the *specified* quantity and scales the other so total exposure is
    /// `F·amt` either way:
    ///   - [`InfusionDef::RateDefined`] (`RATE>0`, `RATE=-1`): `(rate, F·duration)`.
    ///   - [`InfusionDef::DurationDefined`] (`RATE=-2`): `(F·rate, duration)`.
    ///
    /// Every prediction path derives its infusion `F` handling from this: the
    /// injected/closed-form rate is the returned `rate`, and the infusion window /
    /// break-time end is the returned `duration`. A bolus uses
    /// [`PkParams::bioavailable_amount`] instead (`F·amt`); the oral depot bakes `F`
    /// in (so the analytical path passes the `1.0` branch of `route_f_scale`). The
    /// analytic-sensitivity engine (`sens/`) applies `F` as a `route_f_scale`
    /// post-multiply, so it declines a rate-defined infusion under `F ≠ 1` (which
    /// reshapes rather than scales) to the FD gradient. At `F = 1` both arms are
    /// the no-op identity, so every infusion is unchanged.
    pub(crate) fn bioavailable_infusion(&self, f_bio: f64) -> (f64, f64) {
        match self.infusion_def {
            InfusionDef::RateDefined => (self.rate, f_bio * self.duration),
            InfusionDef::DurationDefined => (f_bio * self.rate, self.duration),
        }
    }

    /// A clone of this infusion with `rate`/`duration` replaced by the
    /// bioavailable pair from [`Self::bioavailable_infusion`]. Lets the analytical
    /// superposition closed forms (which read `dose.rate`/`dose.duration`) consume
    /// the `F`-reshaped infusion without an extra post-multiply (#419).
    pub(crate) fn with_bioavailable_infusion(&self, f_bio: f64) -> DoseEvent {
        let (rate, duration) = self.bioavailable_infusion(f_bio);
        DoseEvent {
            rate,
            duration,
            ..self.clone()
        }
    }

    /// A clone of this dose repositioned to the origin of a single steady-state
    /// cycle: `time = 0`, with `ss`/`ii` cleared so the event-driven equilibration
    /// walk replays it as an ordinary within-cycle dose. Preserves the
    /// bioavailability-reshaped `(rate, duration)` this dose already carries —
    /// unlike [`Self::new`], which would recompute `duration = amt/rate` and drop
    /// `F` (#419). Lives here because it constructs a `DoseEvent` by functional
    /// update, which the now-private `cmt` field forbids outside this
    /// module (#912).
    pub(crate) fn synthetic_cycle_dose(&self) -> DoseEvent {
        DoseEvent {
            time: 0.0,
            ss: false,
            ii: 0.0,
            ..self.clone()
        }
    }
}

/// Fixed-layout PK parameters — replaces HashMap<String, f64> for AD compatibility.
///
/// Index convention:
///   0: CL      (clearance)
///   1: V       (volume, or V1 for 2-cmt)
///   2: Q/Q2    (intercompartmental clearance, central ↔ peripheral 1; 2-cmt and 3-cmt)
///   3: V2      (peripheral volume 1; 2-cmt and 3-cmt)
///   4: KA      (absorption rate constant, oral only)
///   5: F       (bioavailability, default 1.0)
///   6: Q3      (intercompartmental clearance, 3-cmt: central ↔ peripheral 2)
///   7: V3      (peripheral volume 2, 3-cmt only)
///   8: LAGTIME (dose/absorption lagtime, default 0.0; equivalent to NONMEM ALAG)
///
/// Slots 0–8 are the named PK parameters above. Slots 9.. are spare capacity
/// for ODE models, whose `[individual_parameters]` may declare additional
/// "structural" parameters (rate constants, Emax/EC50, baselines, …) beyond the
/// named PK slots. `ode_param_slots` routes canonical names to slots 0–8 and
/// structural names to the remaining free slots, while keeping slots
/// `PK_IDX_F` and `PK_IDX_LAGTIME` reserved so an undeclared F/lagtime keeps its
/// default rather than being aliased by a structural parameter (issue #122).
/// The headroom here therefore bounds how many structural parameters an ODE
/// model can declare.
///
/// This must be a compile-time constant because the hand-rolled `Dual2`
/// analytic sensitivities (`src/sens/provider.rs`) build stack arrays
/// `[Dual2<M>; MAX_PK_PARAMS]` inside the per-observation loop, and Rust
/// array lengths must be known at compile time (a `Vec` there would mean a
/// heap allocation per observation on the FOCE/FOCEI hot path). The cost of
/// a larger ceiling is stack, on two fronts: `PkParams` holds
/// `MAX_PK_PARAMS * 8` bytes (~1 KB at 128, negligible), but the `Dual2`
/// arrays scale with both the slot count and the dual width `M` (capped at
/// `MAX_SCALE_AXES = 16`) — up to ~279 KB per array at the current ceiling,
/// constructed per observation on rayon worker threads (~2 MB stacks). See
/// `tests/large_ode_slot_headroom.rs` for the stack-safety smoke tests.
pub const MAX_PK_PARAMS: usize = 128;

pub const PK_IDX_CL: usize = 0;
pub const PK_IDX_V: usize = 1;
pub const PK_IDX_Q: usize = 2;
pub const PK_IDX_V2: usize = 3;
pub const PK_IDX_KA: usize = 4;
pub const PK_IDX_F: usize = 5;
pub const PK_IDX_Q3: usize = 6;
pub const PK_IDX_V3: usize = 7;
pub const PK_IDX_LAGTIME: usize = 8;
/// Transit-compartment count `n` (Savic continuous-N) for the analytic
/// `one_cpt_transit` absorption model (#386). Lives in the spare slot region
/// (≥9); only that model reads it, so it does not shrink ODE structural headroom.
pub const PK_IDX_N: usize = 9;
/// Mean transit time `mtt` for the analytic `one_cpt_transit` model (#386).
pub const PK_IDX_MTT: usize = 10;
/// Mean absorption time `mat` (`μ`) for the analytic `one_cpt_ig` inverse-Gaussian
/// absorption model (#790). Spare-region slot (≥9), read only by the IG models.
pub const PK_IDX_MAT: usize = 11;
/// Relative dispersion `cv2` (`CV²`) of the inverse-Gaussian absorption-time
/// distribution for the analytic `one_cpt_ig` model (#790); shape `λ = mat/cv2`.
pub const PK_IDX_CV2: usize = 12;

/// The engine-reserved PK slots: bioavailability (`F`) and absorption lag
/// (`lagtime`). `ode_param_slots` keeps these free for an undeclared F/lagtime
/// (issue #122), both the analytical and ODE engines apply them to the dose
/// itself rather than the RHS, and the "computed but never used" census exempts
/// parameters routed here. Single source of truth so those sites can't drift.
pub(crate) const RESERVED_PK_SLOTS: [usize; 2] = [PK_IDX_F, PK_IDX_LAGTIME];

/// When the engine consumes a dose attribute, and therefore whether a model that
/// *also* reads the same individual parameter can be diagnosed without a dataset
/// in hand. See [`DoseAttrConsumption::of`] and issue #993.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DoseAttrConsumption {
    /// Applied to **every** dose that reaches the compartment, whatever the dataset
    /// says: bioavailability (`F`, `F{n}`) scales the amount, absorption lag
    /// (`LAGTIME`/`ALAG`, `ALAG{n}`/`LAGTIME{n}`) shifts the event time. A model
    /// that also reads such a parameter on the prediction path applies it twice for
    /// any dosed subject, so the parser can reject it outright.
    EveryDose,
    /// Consulted only when a dose carries a **coded `RATE`** — `D{n}` on `RATE=-2`
    /// (modeled duration), `R{n}` on `RATE=-1` (modeled rate). On an ordinary
    /// dataset the parameter is inert, so `R1` as a plain rate constant is a
    /// perfectly correct model; the collision can only be judged with the data, in
    /// [`crate::api::check_modeled_dose_rates`].
    CodedRateOnly,
}

impl DoseAttrConsumption {
    /// The dose-attribute meaning an `[individual_parameters]` name carries on an
    /// ODE model, or `None` for an ordinary parameter. Returns the attribute, when
    /// the engine consumes it, and the compartment it applies to (`None` for the
    /// bare all-compartment forms).
    ///
    /// The engine applies these at the **dose event**, never through the `[odes]`
    /// RHS, so they are the one class of parameter that is load-bearing while
    /// textually absent — which is why the "computed but never used" census exempts
    /// them. The flip side is that a model which *also* reads one on the prediction
    /// path gets it applied twice, silently (#993). Single source of truth for both
    /// halves: the census exemption and the double-use diagnostic.
    ///
    /// `slot` is the parameter's `ode_param_slots` assignment. That is what makes a
    /// **bare** `F`/`LAGTIME`/`ALAG` a dose attribute — the name alone is not
    /// enough, since the routing is what binds it to [`PK_IDX_F`] /
    /// [`PK_IDX_LAGTIME`]. Pass `None` for an analytical model, whose parameters
    /// take no such slot; the compartment-indexed forms are recognised by name
    /// through [`DoseAttr::from_indexed_name`] on either engine.
    ///
    /// Deliberately **not** a list of forbidden names: `F` and `LAGTIME` are the
    /// correct spellings for what they mean, and `D{n}`/`R{n}` are what a NONMEM
    /// user reaches for. The name is fine; the name read as something *else* in the
    /// same model is the defect (#995).
    pub fn of(name: &str, slot: Option<usize>) -> Option<(DoseAttr, Self, Option<usize>)> {
        // Bare `F` / `LAGTIME` / `ALAG`: a dose attribute only because
        // `ode_param_slots` routed it to a reserved slot. Keyed off the real routing
        // rather than the name so it cannot drift from `ode_param_slots`.
        if let Some(s) = slot {
            if s == PK_IDX_F {
                return Some((DoseAttr::F, Self::EveryDose, None));
            }
            if s == PK_IDX_LAGTIME {
                return Some((DoseAttr::Lag, Self::EveryDose, None));
            }
        }
        // Compartment-indexed `F{n}` / `ALAG{n}` / `LAGTIME{n}` / `D{n}` / `R{n}`,
        // via the same predicate that builds `DoseAttrMap`.
        let (attr, cmt) = DoseAttr::from_indexed_name(name)?;
        let when = match attr {
            DoseAttr::F | DoseAttr::Lag => Self::EveryDose,
            DoseAttr::Duration | DoseAttr::Rate => Self::CodedRateOnly,
        };
        Some((attr, when, Some(cmt)))
    }
}

/// A dose-modifying attribute that NONMEM keys by **compartment** — `Fn`
/// (bioavailability), `ALAGn` (absorption lag), `Dn` (modeled infusion
/// *duration*, `RATE=-2`), `Rn` (modeled infusion *rate*, `RATE=-1`). A dose
/// into compartment `n` uses the attribute declared for `n`; ferx additionally
/// honours a bare `F`/`lagtime` as the all-compartment default (see
/// [`DoseAttrMap`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DoseAttr {
    /// Bioavailability fraction (`Fn`); default 1.0.
    F,
    /// Absorption/dose lag time (`ALAGn`); default 0.0.
    Lag,
    /// Modeled infusion duration (`Dn`, `RATE=-2`). No bare default — a coded
    /// `RATE=-2` row with no matching `Dn` is an error (as in NONMEM).
    Duration,
    /// Modeled infusion rate (`Rn`, `RATE=-1`). No bare default — see above.
    Rate,
}

/// Resolves a dose's effective bioavailability / lag / modeled duration / rate
/// from the per-dose [`PkParams`] vector, keyed by the dose's **compartment**.
///
/// NONMEM makes these dose attributes compartment-indexed (`F1`/`F2`,
/// `ALAG1`/`ALAG2`, `D1`/`D2`, `R1`/`R2`); the single `PK_IDX_F`/`PK_IDX_LAGTIME`
/// slots can only carry one value, which silently mis-applies when a subject is
/// dosed into more than one compartment (the ODE-engine case; the analytical
/// engine has a single fixed route, so this collapses to the bare slot there).
/// This map is the **single source of truth** for "which slot holds attribute
/// `a` for compartment `c`", so every dose-application path (ODE RHS, analytical
/// infusion, FD gradient) resolves identically and a new path cannot drift.
///
/// Resolution order for a dose into 1-based compartment `cmt`:
///   1. the indexed entry `(attr, cmt)` if the model declared one (`Fn`/`ALAGn`/…);
///   2. for `F`/`Lag` only, the bare slot (`PK_IDX_F` = 1.0, `PK_IDX_LAGTIME` = 0.0)
///      as the all-compartment default — preserving pre-existing bare-`F`/`lagtime`
///      models unchanged;
///   3. `Duration`/`Rate` have no bare fallback (a coded `RATE` with no matching
///      `Dn`/`Rn` parameter is rejected upstream), so [`Self::indexed_slot`]
///      returns `None` and the caller errors.
#[derive(Debug, Clone, Default)]
pub struct DoseAttrMap {
    /// `(attribute, 1-based compartment) -> PkParams slot`. Empty for the common
    /// single-route / bare-`F`/`lagtime` model, where every lookup falls through
    /// to the reserved slot.
    indexed: HashMap<(DoseAttr, usize), usize>,
    /// The entries whose parameter is **also** read on the model's prediction path
    /// (`[odes]` RHS / `[scaling]` readout), carrying the parameter's name for the
    /// diagnostic. Only `Duration`/`Rate` ever land here: an `F{n}`/`ALAG{n}` in the
    /// same position is already a parse error, since those apply to *every* dose,
    /// whereas a `D{n}`/`R{n}` is inert unless the data codes `RATE=-2`/`-1`. That
    /// makes the collision data-dependent, so it is judged in
    /// `api::validation::check_modeled_dose_rates` rather than by the parser (#993).
    prediction_path_reads: HashMap<(DoseAttr, usize), String>,
}

impl DoseAttr {
    /// Short noun for this attribute, for diagnostics ("bioavailability", …).
    /// Kept beside [`DoseAttrConsumption::of`] so a message can never name one
    /// attribute and describe another.
    pub fn noun(self) -> &'static str {
        match self {
            DoseAttr::F => "bioavailability",
            DoseAttr::Lag => "absorption lag",
            DoseAttr::Duration => "modeled infusion duration",
            DoseAttr::Rate => "modeled infusion rate",
        }
    }

    /// What the engine does with it at the dose event, phrased to complete
    /// "the engine already …".
    pub fn applied_as(self) -> &'static str {
        match self {
            DoseAttr::F => "scales each dose amount by it",
            DoseAttr::Lag => "delays each dose event by it",
            DoseAttr::Duration => "uses it as the infusion duration (RATE=-2)",
            DoseAttr::Rate => "uses it as the infusion rate (RATE=-1)",
        }
    }

    /// Recognise a compartment-indexed dose-attribute parameter name, returning
    /// `(attr, 1-based compartment)`. Case-insensitive; the numeric suffix must
    /// be a positive integer (so `F0` is *not* an attribute, and bare `F` /
    /// `lagtime` — handled by the reserved slots — return `None`).
    ///
    /// Recognised: `F{n}` (bioavailability), `ALAG{n}` / `LAGTIME{n}` (lag),
    /// `D{n}` (modeled infusion *duration*, `RATE=-2`; #324), and `R{n}` (modeled
    /// infusion *rate*, `RATE=-1`; #324). `S{n}` is excluded — that is the
    /// `[scaling]` block's compartment scale, a separate concept.
    ///
    /// Both `D{n}` and `R{n}` carry a collision risk (a `D`- or `R`-prefixed name
    /// could be an ordinary ODE rate constant), so neither is forcibly *reserved*
    /// here: recognising the name merely makes the [`DoseAttrMap`] entry available
    /// (harmless if never dosed against — the entry is consulted only by
    /// [`DoseEvent::resolve_rate`] when a `RATE=-2`/`-1` dose targets compartment
    /// `n`), and the data-driven gate (`E_MODELED_DURATION_NO_PARAM` /
    /// `E_MODELED_RATE_NO_PARAM`) lives in `check_model_data`. NONMEM treats `D{n}`
    /// / `R{n}` as reserved `$PK` names the same way, so a model that names a
    /// non-dose parameter `R1` while also dosing `RATE=-1` into compartment 1 is
    /// the user's collision to resolve, not ours.
    ///
    /// Recognising a name does not by itself make it a dose attribute — the
    /// caller still gates on engine (compartment-indexed `F`/`Lag`/`Duration` are
    /// ODE-only) and on the compartment existing. `alag` and `lagtime` both map
    /// to [`DoseAttr::Lag`], matching the existing bare `alag`/`lagtime` aliases.
    pub fn from_indexed_name(name: &str) -> Option<(DoseAttr, usize)> {
        let lower = name.to_ascii_lowercase();
        // The prefixes are mutually exclusive — no name starts with two of them,
        // and none is a prefix of another (`lagtime`/`alag`/`f`/`d`/`r` all differ
        // in their first byte except the two lag aliases, which are disjoint) — so
        // the iteration order does not affect the result.
        for (prefix, attr) in [
            ("lagtime", DoseAttr::Lag),
            ("alag", DoseAttr::Lag),
            ("f", DoseAttr::F),
            ("d", DoseAttr::Duration),
            ("r", DoseAttr::Rate),
        ] {
            if let Some(suffix) = lower.strip_prefix(prefix) {
                // The suffix must be a pure positive integer; `f_bio`, `cl`, etc.
                // (non-numeric or empty suffixes) are not attributes.
                if !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit()) {
                    if let Ok(cmt) = suffix.parse::<usize>() {
                        if cmt >= 1 {
                            return Some((attr, cmt));
                        }
                    }
                }
            }
        }
        None
    }
}

impl DoseAttrMap {
    /// Record that compartment `cmt`'s `attr` is held in PkParams `slot`.
    pub fn insert(&mut self, attr: DoseAttr, cmt: usize, slot: usize) {
        self.indexed.insert((attr, cmt), slot);
    }

    /// Record that `name` — the `D{cmt}`/`R{cmt}` parameter for `attr` — is also
    /// read on the prediction path, so a coded-`RATE` dose that lands on it can be
    /// reported as a double use once the data is known (#993). Recording is
    /// unconditional and cheap; whether it *matters* depends on the dataset, which
    /// is why nothing here rejects.
    pub fn mark_prediction_path_read(&mut self, attr: DoseAttr, cmt: usize, name: &str) {
        self.prediction_path_reads
            .insert((attr, cmt), name.to_string());
    }

    /// The parameter name for `(attr, cmt)` when it is read on the prediction path,
    /// else `None`. See [`Self::mark_prediction_path_read`].
    pub fn prediction_path_read(&self, attr: DoseAttr, cmt: usize) -> Option<&str> {
        self.prediction_path_reads
            .get(&(attr, cmt))
            .map(String::as_str)
    }

    /// `true` when no compartment-indexed attribute is recorded — the common
    /// case (bare-`F`/`lagtime` model, no `D{cmt}`). Lets the dose-resolution
    /// step short-circuit before scanning a subject's doses: with no indexed
    /// slot there can be no modeled-`RATE` dose to resolve.
    pub fn is_empty(&self) -> bool {
        self.indexed.is_empty()
    }

    /// The PkParams slot holding `attr` for compartment `cmt`, if the model
    /// declared a compartment-indexed parameter for it.
    ///
    /// `cmt` is 1-based, and `CMT=0` — NONMEM's *default dose compartment* — is
    /// compartment 1, so it resolves to the `{attr}1` entry (#899). Without the
    /// `max(1)` a dose written `CMT=0` misses the indexed map and silently falls
    /// back to the bare slot: on a model declaring `F1 = 0.6` and nothing else,
    /// `PkParams::default()` leaves `PK_IDX_F` at **1.0**, so the dose would be
    /// delivered at full amount with no error and no warning — the same silent
    /// class #899 exists to remove, and the last *dose-application* site where a
    /// `CMT=0` dose was silently mis-applied. (One validation gate lingered past
    /// this: `check_absorption_closed_form_support` still *rejected* a `CMT=0`
    /// dose on a closed-form absorption model as non-depot — a loud error, not a
    /// silent mis-dose — corrected in the same spirit by the #913 review.)
    ///
    /// Analytical models are unaffected: per-compartment `F{cmt}`/`ALAG{cmt}` are
    /// ODE-only (`model_parser`, which rejects an unbound `F{cmt}` on a `pk(...)`
    /// model outright), so their `indexed` map holds only `D{cmt}`/`R{cmt}` — and
    /// a `CMT=0` modeled-`RATE` dose is an infusion by `DoseEvent::is_infusion`,
    /// which `check_dose_compartments` rejects on both engines.
    pub fn indexed_slot(&self, attr: DoseAttr, cmt: usize) -> Option<usize> {
        self.indexed.get(&(attr, cmt.max(1))).copied()
    }

    /// Whether any compartment-indexed entry for `attr` is declared (e.g. `F1`/`F2`
    /// for [`DoseAttr::F`]). The bare slot (`PK_IDX_F`/`PK_IDX_LAGTIME`) is the
    /// fallback and is *not* recorded here, so this is true only for the
    /// per-compartment form.
    pub fn has_indexed_attr(&self, attr: DoseAttr) -> bool {
        self.indexed.keys().any(|&(a, _)| a == attr)
    }

    /// Bioavailability for a dose into 1-based `cmt`: `F{cmt}` if declared, else
    /// the bare `PK_IDX_F` slot (default 1.0 when the model has no `F` at all).
    pub fn f_bio(&self, cmt: usize, params: &[f64]) -> f64 {
        self.resolve_or(DoseAttr::F, cmt, PK_IDX_F, 1.0, params)
    }

    /// Lag time for a dose into 1-based `cmt`: `ALAG{cmt}` if declared, else the
    /// bare `PK_IDX_LAGTIME` slot (default 0.0 when the model has no lag at all).
    pub fn lagtime(&self, cmt: usize, params: &[f64]) -> f64 {
        self.resolve_or(DoseAttr::Lag, cmt, PK_IDX_LAGTIME, 0.0, params)
    }

    /// The PkParams slot holding the lag time for a dose into 1-based `cmt`:
    /// `ALAG{cmt}` if compartment-indexed, else the bare `PK_IDX_LAGTIME`. Single-
    /// sources the slot-resolution fallback that callers (e.g. the ODE sensitivity
    /// walk's `dose_lag_slot`) would otherwise re-implement.
    pub fn lag_slot(&self, cmt: usize) -> usize {
        self.indexed_slot(DoseAttr::Lag, cmt)
            .unwrap_or(PK_IDX_LAGTIME)
    }

    /// Read `attr` for `cmt` from its indexed slot, else from `bare_slot`, else
    /// `dflt` (both reads are bounds-checked so a short `params` slice — e.g. an
    /// analytical model whose vector stops at slot 8 — cannot panic).
    fn resolve_or(
        &self,
        attr: DoseAttr,
        cmt: usize,
        bare_slot: usize,
        dflt: f64,
        params: &[f64],
    ) -> f64 {
        let slot = self.indexed_slot(attr, cmt).unwrap_or(bare_slot);
        params.get(slot).copied().unwrap_or(dflt)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PkParams {
    pub values: [f64; MAX_PK_PARAMS],
}

impl Default for PkParams {
    fn default() -> Self {
        let mut v = [0.0; MAX_PK_PARAMS];
        v[PK_IDX_F] = 1.0; // bioavailability defaults to 1
        Self { values: v }
    }
}

impl PkParams {
    pub fn cl(&self) -> f64 {
        self.values[PK_IDX_CL]
    }
    pub fn v(&self) -> f64 {
        self.values[PK_IDX_V]
    }
    pub fn q(&self) -> f64 {
        self.values[PK_IDX_Q]
    }
    pub fn v2(&self) -> f64 {
        self.values[PK_IDX_V2]
    }
    pub fn ka(&self) -> f64 {
        self.values[PK_IDX_KA]
    }
    /// Transit-compartment count `n` (Savic continuous-N) for the analytic
    /// `one_cpt_transit` model (#386); `0` for every other model.
    pub fn n_transit(&self) -> f64 {
        self.values[PK_IDX_N]
    }
    /// Mean transit time `mtt` for the analytic `one_cpt_transit` model (#386).
    pub fn mtt(&self) -> f64 {
        self.values[PK_IDX_MTT]
    }
    /// Mean absorption time `mat` (`μ`) for the analytic `one_cpt_ig` model (#790);
    /// `0` for every other model.
    pub fn mat(&self) -> f64 {
        self.values[PK_IDX_MAT]
    }
    /// Relative dispersion `cv2` (`CV²`) for the analytic `one_cpt_ig` model (#790);
    /// the IG shape is `λ = mat/cv2`.
    pub fn cv2(&self) -> f64 {
        self.values[PK_IDX_CV2]
    }
    pub fn f_bio(&self) -> f64 {
        self.values[PK_IDX_F]
    }

    /// Bioavailable dose **amount**: `F · amount`. `F` scales the input on every
    /// instantaneous route — IV bolus and oral-depot load alike — matching NONMEM's
    /// `F1` (#327).
    ///
    /// Single source of truth for `F` on a *bolus/amount*. The infusion *shape*
    /// rule (which of rate/duration `F` scales) lives in
    /// [`DoseEvent::bioavailable_infusion`] (#419); together they cover every route.
    /// The prediction paths derive their `F` handling from these:
    /// * event-driven (`pk/event_driven.rs`) calls them directly;
    /// * the analytical superposition path (`pk/mod.rs`) applies the amount rule as
    ///   a post-multiply via `route_f_scale` (oral-depot closed forms bake `F` in,
    ///   so they take the `1.0` branch) and feeds the infusion rule in via
    ///   [`DoseEvent::with_bioavailable_infusion`];
    /// * the analytic-sensitivity engine (`sens/`) post-multiplies by `F`
    ///   (`route_f_scale`) and declines the #419 reshaping case (rate-defined
    ///   infusion, `F ≠ 1`) to the FD gradient.
    ///
    /// A change to either rule must be mirrored in those sites.
    pub(crate) fn bioavailable_amount(&self, amount: f64) -> f64 {
        self.f_bio() * amount
    }
    pub fn q3(&self) -> f64 {
        self.values[PK_IDX_Q3]
    }
    pub fn v3(&self) -> f64 {
        self.values[PK_IDX_V3]
    }
    pub fn lagtime(&self) -> f64 {
        self.values[PK_IDX_LAGTIME]
    }

    /// Map a PK parameter name to its index in the fixed-size array.
    ///
    /// `"alag"` is accepted as an alias for `"lagtime"` for NONMEM familiarity.
    pub fn name_to_index(name: &str) -> Option<usize> {
        match name {
            "cl" => Some(PK_IDX_CL),
            "v" | "v1" => Some(PK_IDX_V),
            "q" | "q2" => Some(PK_IDX_Q),
            "v2" => Some(PK_IDX_V2),
            "ka" => Some(PK_IDX_KA),
            "f" => Some(PK_IDX_F),
            "q3" => Some(PK_IDX_Q3),
            "v3" => Some(PK_IDX_V3),
            "lagtime" | "alag" => Some(PK_IDX_LAGTIME),
            // Savic transit-absorption params for `pk one_cpt_transit` (#386).
            "n" => Some(PK_IDX_N),
            "mtt" => Some(PK_IDX_MTT),
            // Inverse-Gaussian absorption params for `pk one_cpt_ig` (#790).
            "mat" => Some(PK_IDX_MAT),
            "cv2" => Some(PK_IDX_CV2),
            _ => None,
        }
    }

    /// Build from named HashMap (bridge for parser compatibility)
    pub fn from_hashmap(map: &HashMap<String, f64>) -> Self {
        let mut p = Self::default();
        if let Some(&v) = map.get("cl") {
            p.values[PK_IDX_CL] = v;
        }
        if let Some(&v) = map.get("v") {
            p.values[PK_IDX_V] = v;
        }
        if let Some(&v) = map.get("v1") {
            p.values[PK_IDX_V] = v;
        }
        if let Some(&v) = map.get("q") {
            p.values[PK_IDX_Q] = v;
        }
        if let Some(&v) = map.get("q2") {
            p.values[PK_IDX_Q] = v;
        }
        if let Some(&v) = map.get("v2") {
            p.values[PK_IDX_V2] = v;
        }
        if let Some(&v) = map.get("ka") {
            p.values[PK_IDX_KA] = v;
        }
        if let Some(&v) = map.get("f") {
            p.values[PK_IDX_F] = v;
        }
        if let Some(&v) = map.get("q3") {
            p.values[PK_IDX_Q3] = v;
        }
        if let Some(&v) = map.get("v3") {
            p.values[PK_IDX_V3] = v;
        }
        if let Some(&v) = map.get("lagtime").or_else(|| map.get("alag")) {
            p.values[PK_IDX_LAGTIME] = v;
        }
        p
    }
}

/// A single subject with dosing and observation data
#[derive(Debug, Clone, Default)]
pub struct Subject {
    pub id: String,
    pub doses: Vec<DoseEvent>,
    pub obs_times: Vec<f64>,
    /// Original (unshifted) observation times from the data file's TIME column,
    /// parallel to `obs_times`. For subjects with stacked reset occasions whose
    /// TIME restarts (see `io/datareader`), `obs_times` carries the internal
    /// monotonic timeline while this keeps the raw value for the user-clock
    /// diagnostics: sdtab/covtab TIME and `predict()`/`simulate()` TIME.
    /// `[derived]` integral windows use raw times so per-occasion AUC is
    /// correct. Populated for every observation read from a CSV (equal to
    /// `obs_times` when no shift occurred); empty only for in-memory subjects,
    /// where consumers fall back to `obs_times`.
    pub obs_raw_times: Vec<f64>,
    pub observations: Vec<f64>,
    pub obs_cmts: Vec<usize>,
    /// Subject-representative covariate values (first non-missing value per
    /// covariate). Used by the AD fast path and as a fallback when neither
    /// `dose_covariates` nor `obs_covariates` is populated.
    pub covariates: HashMap<String, f64>,
    /// Per-dose covariate snapshot (LOCF), parallel to `doses`. When the
    /// dataset has no time-varying covariates, this is empty and consumers
    /// fall back to `covariates`. NONMEM-equivalent: the value of `$PK`
    /// inputs at each dose record.
    pub dose_covariates: Vec<HashMap<String, f64>>,
    /// Per-observation covariate snapshot (LOCF), parallel to `obs_times`.
    /// Same fallback semantics as `dose_covariates`.
    pub obs_covariates: Vec<HashMap<String, f64>>,
    /// Times of EVID=2 "other event" rows (typically covariate-change
    /// markers). Only populated when the subject has time-varying
    /// covariates — for time-constant covariates these rows are no-ops
    /// (NONMEM-equivalent: $PK runs but with the same values).
    /// The event-driven propagators see them as a third event kind that
    /// does not mutate compartment amounts but does refresh the
    /// piecewise-constant rate matrix from the row's covariate values.
    pub pk_only_times: Vec<f64>,
    /// Per-EVID-2 covariate snapshot (LOCF), parallel to `pk_only_times`.
    /// Empty when no TV covariates.
    pub pk_only_covariates: Vec<HashMap<String, f64>>,
    /// Times of system-reset events (NONMEM EVID=3 "reset" and EVID=4
    /// "reset + dose"). At each of these times every compartment amount is
    /// set back to zero. For EVID=4 the row's dose is *also* recorded in
    /// `doses`; the reset is applied first (state-propagating paths use a
    /// `Reset < Dose` tie-break at a shared time). Empty for the common case
    /// of no reset rows — superposition stays valid and consumers skip the
    /// state-propagating path. Resets break dose superposition, so a subject
    /// with any reset is forced onto the event-driven analytical / ODE path.
    pub reset_times: Vec<f64>,
    /// Censoring flag per observation (0 = quantified, 1 = below LLOQ, -1 = above ULOQ).
    /// On censored rows, `observations[j]` holds the corresponding LOQ limit.
    pub cens: Vec<i8>,
    /// Occasion index per observation row (parallel to `obs_times`).
    /// Empty when no IOV column is present in the data.
    pub occasions: Vec<u32>,
    /// `L2` grouping id per observation row (parallel to `obs_times`), the
    /// NONMEM data item that ties several records into one correlated
    /// observation unit. Rows sharing an `L2` value (within a subject) are
    /// correlated together by `block_sigma`; a `0` (or empty vector when the
    /// data has no `L2` column) means "ungrouped", and those rows fall back to
    /// the `(time, occasion)` pairing rule. This is what lets a user pair, e.g.,
    /// the total and unbound rows of one blood draw explicitly rather than
    /// relying on co-temporal row order (#827).
    pub obs_l2: Vec<i64>,
    /// Occasion index per dose event (parallel to `doses`).
    /// Empty when no IOV column is present in the data.
    pub dose_occasions: Vec<u32>,
    /// FREM observation type per observation (parallel to `obs_times`).
    /// 0 = PK observation, 100/200/300/... = covariate observation.
    /// Empty when FREMTYPE column is absent from the data.
    pub fremtype: Vec<u16>,
    /// Non-Gaussian observation records (TTE events, discrete states, counts).
    /// Empty for all-Gaussian subjects. Populated by the data reader when the
    /// model declares a non-Gaussian endpoint for the row's CMT.
    ///
    /// Compiled unconditionally (Phase 4.0): the discrete-state / count plumbing
    /// is the shared foundation for the categorical (Track C) and Markov
    /// (Track D) endpoints, so it must be present on the default build. Only the
    /// TTE `Event` variant it can hold stays behind the `survival` feature.
    pub obs_records: Vec<ObsRecord>,
}

impl Subject {
    pub fn has_censored_observation(&self) -> bool {
        self.cens.iter().any(|&c| c != 0)
    }

    /// True when the subject carries per-event covariate snapshots (i.e. at
    /// least one covariate was time-varying in the source data). When false,
    /// callers can use `covariates` directly and skip the per-event evaluation
    /// loop.
    pub fn has_tv_covariates(&self) -> bool {
        !self.dose_covariates.is_empty() || !self.obs_covariates.is_empty()
    }

    /// True when this subject carries any system-reset event (EVID=3 or
    /// EVID=4). Resets zero all compartment amounts, which dose superposition
    /// cannot express, so the prediction dispatcher routes these subjects onto
    /// the state-propagating event-driven analytical / ODE path regardless of
    /// whether they have time-varying covariates.
    pub fn has_resets(&self) -> bool {
        !self.reset_times.is_empty()
    }

    /// True when any dose record on this subject is flagged steady-state
    /// (SS=1). Used to gate paths that don't yet honour SS (event-driven,
    /// AD propagators, ODE) — see `estimation/inner_optimizer.rs` and the
    /// SS warning in `api.rs`.
    pub fn has_ss_doses(&self) -> bool {
        self.doses.iter().any(|d| d.ss)
    }

    /// True when any dose is a *periodic* steady-state dose: flagged `SS=1`
    /// **and** carrying a positive dosing interval `II > 0`. This is the precise
    /// predicate the analytic-sensitivity gates use to decline steady-state
    /// combinations they don't yet handle (the dual SS-equilibration needs the
    /// `II` recurrence). Distinct from [`has_ss_doses`](Self::has_ss_doses),
    /// which tests the `SS` flag alone; the single source of truth for the
    /// `d.ss && d.ii > 0.0` gate scan shared by `ode_provider.rs` and
    /// `estimation/inner_optimizer.rs`.
    pub fn has_periodic_ss_dose(&self) -> bool {
        self.doses.iter().any(|d| d.ss && d.ii > 0.0)
    }

    /// True when every dose carries concrete (`Fixed`) `rate`/`duration` — i.e.
    /// no dose is still a modeled NONMEM coded `RATE` awaiting
    /// [`DoseEvent::resolve_rate`]. The common case (no coded doses) is `true`,
    /// so the ODE resolve shadows return `Cow::Borrowed` and the analytical / AD
    /// tripwires pass. Single source of truth alongside [`DoseEvent::is_fixed`]
    /// (#324 / #383): a future coded variant changes "resolved" in one place.
    pub fn all_doses_fixed(&self) -> bool {
        self.doses.iter().all(|d| d.is_fixed())
    }

    /// True when any dose is a *rate-defined* infusion (`RATE>0` data or
    /// `RATE=-1` → `R{cmt}`), i.e. one whose infusion *window* bioavailability
    /// `F` reshapes (NONMEM scales the duration to `F·amt/rate`; #419). A
    /// duration-defined infusion (`RATE=-2`) is excluded — `F` scales its rate,
    /// not its window. Used to decide whether a cached
    /// [`crate::pk::event_driven::EventSchedule`] (whose break times bake in the
    /// window) would go stale as `F` varies across the inner search - the same
    /// reason [`CompiledModel::has_lagtime`] gates that cache.
    pub fn has_rate_defined_infusion(&self) -> bool {
        self.doses
            .iter()
            .any(|d| d.is_infusion() && matches!(d.infusion_def, InfusionDef::RateDefined))
    }

    /// Time of the first dose of the reset-occasion containing `obs_time`,
    /// used as the TAFD (time-after-first-dose) reference. For a subject with
    /// no resets this is just the earliest dose (the conventional TAFD origin);
    /// for stacked reset occasions it is the first dose at or after the most
    /// recent reset, so TAFD resets per occasion instead of measuring from the
    /// very first occasion across the shifted timeline (issue #195 review).
    /// `obs_time` is on the same (internal, possibly shifted) clock as
    /// `doses[*].time` and `reset_times`, so the result is correct regardless
    /// of any occasion shift. Returns `f64::INFINITY` when the occasion
    /// containing `obs_time` has no dose at or after its reset — which includes
    /// the no-doses-at-all case and a pure-reset (EVID=3) occasion with no
    /// subsequent dose; callers treat a non-finite result as an undefined TAFD
    /// (NaN).
    pub fn occasion_first_dose_time(&self, obs_time: f64) -> f64 {
        let seg_start = self
            .reset_times
            .iter()
            .copied()
            .filter(|&r| r <= obs_time + 1e-9)
            .fold(f64::NEG_INFINITY, f64::max);
        self.doses
            .iter()
            .map(|d| d.time)
            .filter(|&t| t >= seg_start - 1e-9)
            .fold(f64::INFINITY, f64::min)
    }

    /// Covariate snapshot at observation index `j`. Falls back to the
    /// subject-static `covariates` map when per-event snapshots aren't present.
    pub fn obs_cov(&self, j: usize) -> &HashMap<String, f64> {
        self.obs_covariates.get(j).unwrap_or(&self.covariates)
    }

    /// Covariate snapshot at dose index `k`. Same fallback as `obs_cov`.
    pub fn dose_cov(&self, k: usize) -> &HashMap<String, f64> {
        self.dose_covariates.get(k).unwrap_or(&self.covariates)
    }

    /// The `TIME` a *record-level* expression sees at observation index `j`: the
    /// raw data-file time ([`Subject::obs_raw_times`]), falling back to the
    /// internal monotonic [`Subject::obs_times`] for in-memory subjects that
    /// don't carry it.
    ///
    /// This is the user clock, and it is the convention every other per-record
    /// object already uses — sdtab/covtab `TIME`, `predict()`/`simulate()` `TIME`,
    /// `[derived]` integral windows, and the custom residual-magnitude model
    /// ([`ModelParameters::ruv_obs_mult`], documented as "matching what NONMEM's
    /// `$ERROR` sees"). A `[scaling]` Form C readout is the `$ERROR` twin — it is
    /// anchored against NONMEM's `$ERROR` in
    /// `tests/scaling_time_readout_nonmem_anchor.rs` — so it reads the same clock
    /// (#1028). The two differ only for subjects with stacked reset occasions
    /// whose data TIME restarts, where `obs_times` is the shifted integrator
    /// timeline; the integrator itself keeps using `obs_times`, exactly as
    /// NONMEM's `$DES` clock is not its `$ERROR` `TIME`.
    pub fn readout_time(&self, j: usize) -> f64 {
        self.obs_raw_times
            .get(j)
            .copied()
            .unwrap_or_else(|| self.obs_times.get(j).copied().unwrap_or(0.0))
    }

    /// Covariate snapshot at EVID=2 row index `m`. Same fallback as
    /// the others — for time-constant covariates this returns the
    /// subject-static map.
    pub fn pk_only_cov(&self, m: usize) -> &HashMap<String, f64> {
        self.pk_only_covariates.get(m).unwrap_or(&self.covariates)
    }

    /// Names of covariates whose value **varies within this subject**, i.e. is
    /// not constant across the per-observation / per-dose / per-EVID2 LOCF
    /// snapshots. Returns an empty vector when the subject has no time-varying
    /// covariates (the common case — snapshots are empty or all equal).
    ///
    /// This is the *which-covariate* companion to the boolean
    /// [`has_tv_covariates`](Self::has_tv_covariates): guards that only care
    /// whether a *specific* referenced covariate varies (e.g. survival hazards,
    /// #741) intersect this set with their reference list, rather than rejecting
    /// on the mere presence of any time-varying covariate. Unlike
    /// `has_tv_covariates`, it also inspects `pk_only_covariates` (EVID=2 rows),
    /// so a covariate that changes only on an "other event" row is detected.
    /// Names are returned sorted for a deterministic diagnostic.
    pub fn time_varying_covariate_names(&self) -> Vec<String> {
        // First value seen per covariate; a later differing value marks it varying.
        let mut first_seen: HashMap<&str, f64> = HashMap::new();
        let mut varying: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let snapshots = self
            .obs_covariates
            .iter()
            .chain(self.dose_covariates.iter())
            .chain(self.pk_only_covariates.iter());
        for snap in snapshots {
            for (name, &val) in snap {
                match first_seen.get(name.as_str()) {
                    // Exact inequality is correct: values are LOCF copies of the
                    // parsed CSV f64, so a genuine change differs bit-for-bit and a
                    // carried-forward value is identical. A NaN covariate (which
                    // should not occur) compares unequal and is conservatively flagged.
                    Some(&prev) if prev != val => {
                        varying.insert(name.clone());
                    }
                    Some(_) => {}
                    None => {
                        first_seen.insert(name.as_str(), val);
                    }
                }
            }
        }
        varying.into_iter().collect()
    }
}

/// Summary of records excluded by `[data_selection]` `ignore`/`accept` rules.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Default)]
pub struct ExclusionSummary {
    /// Subject IDs that had all records removed (zero doses and observations remaining).
    pub excluded_subject_ids: Vec<String>,
    /// Number of observation records (EVID==0, MDV==0) excluded.
    pub n_obs_excluded: usize,
    /// Number of dose records (EVID 1/4) excluded.
    pub n_dose_excluded: usize,
    /// Number of other records excluded that are neither a scored observation
    /// nor a dose — EVID==2 (other event), EVID==3 (reset), and missing-DV
    /// observation rows (EVID==0, MDV==1). Tracked so the reported counts sum to
    /// every excluded record and the summary can't read all-zeros while rows
    /// were dropped.
    pub n_other_excluded: usize,
    /// Total CSV records read before any filtering.
    pub n_records_total: usize,
    /// `ignore` / `ignore_subjects` clauses that matched at least one record,
    /// in declaration order.
    pub fired_ignore: Vec<String>,
    /// `accept` clauses that rejected at least one record, in declaration order.
    pub fired_accept: Vec<String>,
}

/// A collection of subjects
#[derive(Debug, Clone)]
pub struct Population {
    pub subjects: Vec<Subject>,
    pub covariate_names: Vec<String>,
    pub dv_column: String,
    /// All column headers from the data CSV in original order (ID, TIME, DV, AMT, ...,
    /// covariates). Preserved verbatim so downstream consumers can echo a NONMEM-style
    /// `$INPUT` line. Empty for in-memory `Population` values that were not read from a file.
    pub input_columns: Vec<String>,
    /// Present when `[data_selection]` rules were applied; `None` when no filtering
    /// was requested or the population was constructed in-memory.
    pub exclusions: Option<ExclusionSummary>,
    /// Non-fatal warnings generated while reading the dataset (e.g. ADDL with missing II,
    /// unparseable OCC values). Propagated into `FitResult.warnings` by `fit()`.
    pub warnings: Vec<String>,
}

impl Population {
    pub fn n_obs(&self) -> usize {
        self.subjects.iter().map(|s| s.observations.len()).sum()
    }

    /// Drop per-event covariate snapshots that don't carry any
    /// variation in covariates the model actually references.
    ///
    /// The data reader populates `dose_covariates` / `obs_covariates` /
    /// `pk_only_covariates` whenever *any* non-standard CSV column
    /// varies within a subject — including pure time-tracker columns
    /// like `DAY` or `STIME` that no model expression touches. The
    /// downstream prediction path then takes the heavy event-driven
    /// route (one `pk_param_fn` call per event, plus state propagation
    /// across each interval) instead of the analytical superposition
    /// fast path that runs `pk_param_fn` once per subject. This pre-fit
    /// pass clears the snapshots for any subject whose model-referenced
    /// covariates are all time-constant, so the existing
    /// `has_tv_covariates()`-based dispatcher routes those subjects
    /// through the fast path automatically.
    ///
    /// Returns the number of subjects whose snapshots were cleared
    /// (for diagnostic / warning purposes).
    pub fn prune_irrelevant_tv_covariates(&mut self, referenced: &[String]) -> usize {
        let mut pruned = 0;
        for subj in &mut self.subjects {
            if subj.dose_covariates.is_empty()
                && subj.obs_covariates.is_empty()
                && subj.pk_only_covariates.is_empty()
            {
                continue; // already on the fast path
            }
            let mut any_relevant_tv = false;
            'covs: for cov in referenced {
                let base = subj.covariates.get(cov).copied();
                for snap in subj
                    .dose_covariates
                    .iter()
                    .chain(subj.obs_covariates.iter())
                    .chain(subj.pk_only_covariates.iter())
                {
                    if snap.get(cov).copied() != base {
                        any_relevant_tv = true;
                        break 'covs;
                    }
                }
            }
            if !any_relevant_tv {
                subj.dose_covariates.clear();
                subj.obs_covariates.clear();
                subj.pk_only_covariates.clear();
                pruned += 1;
            }
        }
        pruned
    }
}

/// Whether a declared covariate is continuous or categorical.
///
/// Both kinds are carried as `f64` in the data path (categoricals must be
/// numerically coded — see [`CovariateDecl`]). The distinction is metadata for
/// downstream consumers (R-side summary statistics, covariate-search
/// algorithms) that treat the two differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CovariateKind {
    Continuous,
    Categorical,
}

impl CovariateKind {
    /// Lowercase label used in the `[covariates]` block and in output.
    pub fn label(&self) -> &'static str {
        match self {
            CovariateKind::Continuous => "continuous",
            CovariateKind::Categorical => "categorical",
        }
    }
}

/// One entry from the optional `[covariates]` DSL block: a dataset column the
/// modeller declares as a covariate, tagged continuous or categorical. This is
/// a declaration of *availability* — it does not imply the covariate is used in
/// the structural model.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CovariateDecl {
    /// Column name, case-sensitive, matching the CSV header.
    pub name: String,
    pub kind: CovariateKind,
}

/// A single row of the [`CovariateTable`], echoing one input dataset record.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq)]
pub struct CovariateRow {
    pub id: String,
    pub time: f64,
    /// EVID of the source row (0=obs, 1=dose, 2=other, 3=reset, 4=reset+dose).
    pub evid: u32,
    /// Covariate values, parallel to [`CovariateTable::names`]. A missing value
    /// (blank / `.` / `NA` in the source) is encoded as `f64::NAN`.
    pub values: Vec<f64>,
}

/// Echo of the declared covariate columns from the input dataset, one row per
/// input record (including dose / EVID rows — unlike the observation-only
/// sdtab). Produced at data-read time when a `[covariates]` block is present,
/// and attached to [`FitResult::covariate_table`]. Missing values are
/// `f64::NAN`. Restricted to declared columns to bound memory.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Default, PartialEq)]
pub struct CovariateTable {
    /// Declared covariate names, in declaration order. Parallel to each row's
    /// `values` and to `kinds`.
    pub names: Vec<String>,
    /// Continuous/categorical tag per covariate, parallel to `names`.
    pub kinds: Vec<CovariateKind>,
    pub rows: Vec<CovariateRow>,
}

/// Between-subject variability matrix (Omega)
#[derive(Debug, Clone)]
pub struct OmegaMatrix {
    pub matrix: DMatrix<f64>,
    pub chol: DMatrix<f64>,
    pub eta_names: Vec<String>,
    pub diagonal: bool,
    /// Which (i,j) entries are free parameters (not structural zeros).
    /// Diagonal entries are always free. Off-diagonals are free only when
    /// both etas belong to the same `block_omega` declaration; cross-block
    /// and standalone-vs-block entries are structural zeros and stay false.
    /// Used by the SAEM M-step to zero sampling correlations that bleed into
    /// structurally-absent entries via `(1/N) Σ ηη^T`.
    pub free_mask: DMatrix<bool>,
    /// Pre-computed Ω⁻¹. Cached at construction so per-call code paths
    /// (`individual_nll_into`, SAEM MH proposals) don't have to clone the
    /// matrix, run Cholesky, and invert on every evaluation.
    pub inv: DMatrix<f64>,
    /// Pre-computed `log|Ω| = 2·Σᵢ log(L_ii)`. Same motivation as `inv`.
    pub log_det: f64,
}

impl OmegaMatrix {
    pub fn from_matrix_with_mask(
        m: DMatrix<f64>,
        names: Vec<String>,
        diagonal: bool,
        free_mask: DMatrix<bool>,
    ) -> Self {
        let n = m.nrows();
        // If the input matrix is PD we use it as-is. If Cholesky fails we
        // regularise (eigenvalue floor) and switch to the regularised
        // matrix from here on, so `matrix`, `chol`, `inv`, and `log_det`
        // all describe the *same* matrix downstream — otherwise the
        // FOCE inner loop's eta_prior (read from `matrix`) would be
        // inconsistent with the cached `inv` (computed from `m_reg`).
        let (matrix, chol, inv) = match m.clone().cholesky() {
            Some(c) => (m, c.l(), c.inverse()),
            None => {
                let eig = m.clone().symmetric_eigen();
                let min_eig = eig.eigenvalues.min();
                let reg = if min_eig < 0.0 { -min_eig + 1e-8 } else { 1e-8 };
                let m_reg = &m + DMatrix::identity(n, n) * reg;
                let c = m_reg
                    .clone()
                    .cholesky()
                    .expect("Regularized matrix must be PD");
                (m_reg, c.l(), c.inverse())
            }
        };
        // log|Ω| = 2·Σᵢ log(L_ii). Negative or zero diagonals shouldn't
        // happen post-regularisation but we fall back to f64::INFINITY so
        // downstream NLL code can short-circuit cleanly instead of NaNing.
        let mut log_det = 0.0;
        for i in 0..n {
            let lii = chol[(i, i)];
            if lii > 0.0 {
                log_det += lii.ln();
            } else {
                log_det = f64::INFINITY;
                break;
            }
        }
        log_det *= 2.0;
        Self {
            matrix,
            chol,
            eta_names: names,
            diagonal,
            free_mask,
            inv,
            log_det,
        }
    }

    pub fn from_matrix(m: DMatrix<f64>, names: Vec<String>, diagonal: bool) -> Self {
        let n = m.nrows();
        // Infer free_mask: diagonal entries always free; for non-diagonal
        // matrices, off-diagonals are free iff non-zero. This is the correct
        // inference when reconstructing an OmegaMatrix from a final estimate
        // matrix where the original block structure has already been imposed.
        // For initial parsing of `block_omega` declarations, use
        // `from_matrix_with_mask` directly so cross-block zeros are preserved.
        let mut free_mask = DMatrix::from_element(n, n, false);
        for i in 0..n {
            for j in 0..n {
                free_mask[(i, j)] = i == j || (!diagonal && m[(i, j)] != 0.0);
            }
        }
        Self::from_matrix_with_mask(m, names, diagonal, free_mask)
    }

    pub fn from_diagonal(variances: &[f64], names: Vec<String>) -> Self {
        let n = variances.len();
        let mut m = DMatrix::zeros(n, n);
        for i in 0..n {
            m[(i, i)] = variances[i];
        }
        Self::from_matrix(m, names, true)
    }

    /// Construct from a known lower-Cholesky factor `L` such that Ω = L Lᵀ.
    /// Avoids the Cholesky factorisation that `from_matrix*` runs, which the
    /// SAEM/FOCEI hot paths hit on every NLopt M-step and every outer
    /// iteration via `unpack_params`. Ω⁻¹ is computed from `L` directly:
    /// Ω⁻¹ = L⁻ᵀ L⁻¹, where L⁻¹ comes from one lower-triangular solve
    /// against the identity.
    pub fn from_chol_factor(
        l: DMatrix<f64>,
        names: Vec<String>,
        diagonal: bool,
        free_mask: DMatrix<bool>,
    ) -> Self {
        let n = l.nrows();
        let matrix = &l * l.transpose();
        // L⁻¹ via lower-triangular solve: L · X = I ⇒ X = L⁻¹.
        // `solve_lower_triangular(&I)` returns Some(_) iff L is non-singular;
        // L Lᵀ is PD by construction so a positive-diagonal L is guaranteed
        // unless the caller hands us a degenerate factor — fall back to a
        // full Cholesky on the materialised matrix in that degenerate case
        // so the cache is at least populated rather than panicking.
        let identity = DMatrix::<f64>::identity(n, n);
        let inv = match l.solve_lower_triangular(&identity) {
            Some(l_inv) => &l_inv.transpose() * &l_inv,
            None => matrix
                .clone()
                .cholesky()
                .expect("L Lᵀ should be PD by construction")
                .inverse(),
        };
        let mut log_det = 0.0;
        for i in 0..n {
            let lii = l[(i, i)];
            if lii > 0.0 {
                log_det += lii.ln();
            } else {
                log_det = f64::INFINITY;
                break;
            }
        }
        log_det *= 2.0;
        Self {
            matrix,
            chol: l,
            eta_names: names,
            diagonal,
            free_mask,
            inv,
            log_det,
        }
    }

    pub fn dim(&self) -> usize {
        self.matrix.nrows()
    }
}

/// Residual error parameters (Sigma)
#[derive(Debug, Clone)]
pub struct SigmaVector {
    pub values: Vec<f64>,
    pub names: Vec<String>,
}

/// Fixed correlation between two named residual-error terms.
///
/// `sigma_i` and `sigma_j` index [`SigmaVector::values`]. The covariance used
/// at runtime is `rho * sigma_i * sigma_j`, so the existing positive SD
/// parameterization remains unchanged while off-diagonal residual covariance is
/// carried into subject-level R matrices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualCorrelation {
    pub sigma_i: usize,
    pub sigma_j: usize,
    pub rho: f64,
}

/// Full set of model parameters
#[derive(Debug, Clone)]
pub struct ModelParameters {
    pub theta: Vec<f64>,
    pub theta_names: Vec<String>,
    pub theta_lower: Vec<f64>,
    pub theta_upper: Vec<f64>,
    /// Per-theta FIX flags (NONMEM-style). Fixed thetas are not estimated;
    /// they retain their initial value and receive SE = 0 in the cov step.
    pub theta_fixed: Vec<bool>,
    pub omega: OmegaMatrix,
    /// Per-eta FIX flags. For diagonal omegas: flag fixes the variance.
    /// For block omegas: all etas within a fixed block share `true`, and
    /// every Cholesky element whose row *or* column is flagged is held
    /// constant during optimization. Parser rejects fixing an eta that
    /// belongs to a block unless the whole block is fixed.
    pub omega_fixed: Vec<bool>,
    pub sigma: SigmaVector,
    /// Per-sigma FIX flags.
    pub sigma_fixed: Vec<bool>,
    /// Inter-occasion variability matrix (Omega_IOV). `None` when no `kappa`
    /// declarations appear in the model file.  Always diagonal for Option A.
    pub omega_iov: Option<OmegaMatrix>,
    /// Per-kappa FIX flags (parallel to `omega_iov` diagonal).
    pub kappa_fixed: Vec<bool>,
    /// Per-class Omega/Sigma for `$MIXTURE` models (#977). `Some` iff the model
    /// carries a `[mixture]` block; the base (class-1) Omega/Sigma stay in
    /// `omega`/`sigma`, while this holds the full per-class matrices (class 1 is
    /// a copy of the base) plus the packing addresses of the per-class
    /// overrides. `None` for a single-population model.
    pub mixture: Option<MixtureParams>,
}

/// Numeric per-class Omega/Sigma for a `$MIXTURE` model (#977 Phase 2).
///
/// The mixing-probability coefficients are ordinary thetas (they ride the theta
/// vector and pack identically), so this struct only carries the per-class
/// random-effect / residual variances. Non-overridden classes hold a copy of the
/// base Omega/Sigma; each `[mixture]` `omega(k)` / `sigma(k)` declaration becomes
/// one entry in the corresponding `*_override_addr` list, which drives the
/// per-class packed segment in `estimation::parameterization`.
#[derive(Debug, Clone)]
pub struct MixtureParams {
    /// Per-class Omega, length K. Index `c` is class `c + 1`; index 0 (class 1)
    /// equals the base `ModelParameters::omega`.
    pub omega: Vec<OmegaMatrix>,
    /// Per-class Sigma, length K. Index 0 (class 1) equals the base sigma.
    pub sigma: Vec<SigmaVector>,
    /// Packed-order addresses of the Omega overrides: `(class_index_0based,
    /// eta_index)`. Class index is >= 1 (class 1 is the base and is never
    /// overridden). One packed scalar — `ln` of the class matrix's Cholesky
    /// diagonal — is emitted per entry, in this order.
    pub omega_override_addr: Vec<(usize, usize)>,
    /// FIX flag per Omega override (parallel to `omega_override_addr`).
    pub omega_override_fixed: Vec<bool>,
    /// Packed-order addresses of the Sigma overrides: `(class_index_0based,
    /// sigma_index)`.
    pub sigma_override_addr: Vec<(usize, usize)>,
    /// FIX flag per Sigma override (parallel to `sigma_override_addr`).
    pub sigma_override_fixed: Vec<bool>,
}

impl ModelParameters {
    /// Return `true` if any parameter is marked FIX.
    pub fn has_any_fixed(&self) -> bool {
        self.theta_fixed.iter().any(|&b| b)
            || self.omega_fixed.iter().any(|&b| b)
            || self.sigma_fixed.iter().any(|&b| b)
            || self.kappa_fixed.iter().any(|&b| b)
    }
}

/// One mixing-probability expression for a `$MIXTURE` class (#977).
///
/// The inner `Expression` AST is parser-private (`pub(crate)`, mirroring
/// [`IndivParamPartials`]); external users see the class index and form but
/// cannot pattern-match the tree. Evaluated per subject over theta + covariates
/// to yield the class logit (or probability); softmax-normalized across classes.
#[derive(Debug, Clone)]
pub struct MixingExpr {
    /// 1-based class this expression scores. For the `logit` form only classes
    /// `1..=n_classes-1` carry an expression (the last class is the softmax
    /// reference with implicit logit 0).
    pub class: usize,
    /// `true` for `logit(k) = …` (raw logit, softmax-normalized); `false` for
    /// `p(k) = …` (probability, must lie in (0,1) and is renormalized).
    pub is_logit: bool,
    /// Mixing expression over theta + covariates. Parser-private AST.
    pub(crate) expr: crate::parser::model_parser::Expression,
}

/// A per-class Omega/Sigma override declared in a `[mixture]` block via
/// `omega(k) NAME ~ var` / `sigma(k) NAME ~ var` (#977). An omitted class shares
/// the base (class-1) Omega/Sigma. Phase 1 stores the raw declaration; Phase 2
/// builds the per-class numeric matrices from it.
#[derive(Debug, Clone)]
pub struct MixtureClassOverride {
    /// 1-based class this override applies to (2..=n_classes; class 1 is base).
    pub class: usize,
    /// Random-effect (eta) or residual (sigma) name being overridden — must name
    /// a base declaration.
    pub name: String,
    /// Initial value: variance for omega, variance-or-SD for sigma per `as_sd`.
    pub init: f64,
    /// `true` when written on the SD scale (`(sd)` suffix).
    pub as_sd: bool,
    /// FIX flag.
    pub fixed: bool,
}

/// `$MIXTURE` model structure attached to [`CompiledModel`] (#977).
///
/// Holds the number of subpopulations, the per-class mixing expressions, and any
/// per-class Omega/Sigma overrides. The class-specific *typical values* are not
/// stored here — they live in `[individual_parameters]` as `MIXNUM`-branched
/// expressions and are shared/split entirely by the user.
///
/// `#[non_exhaustive]`: this struct is built by the parser, and #996 had to add a
/// field to it — a source-breaking change for any downstream crate constructing
/// it with a struct literal. Marking it non-exhaustive makes the next field
/// addition non-breaking; construct it inside this crate (where literals are
/// still allowed) or read it field-by-field.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MixtureSpec {
    /// Number of subpopulations K (>= 2).
    pub n_classes: usize,
    /// Mixing-probability expressions (see [`MixingExpr`]). Both forms score
    /// classes `1..=K-1` (`n_classes - 1` entries); the last class is implicit —
    /// the softmax reference (logit 0) for the `logit` form, or the probability
    /// complement `1 − Σ` for the `p` form.
    pub mixing: Vec<MixingExpr>,
    /// Covariate names referenced by the mixing expressions (for data checks).
    pub logit_covariates: Vec<String>,
    /// Per-class Omega overrides (empty ⇒ all classes share the base Omega).
    pub omega_overrides: Vec<MixtureClassOverride>,
    /// Per-class Sigma overrides (empty ⇒ all classes share the base Sigma).
    pub sigma_overrides: Vec<MixtureClassOverride>,
    /// Class-aware mu-references detected from `[individual_parameters]` (#996).
    ///
    /// Empty unless a typical value is written as a `MIXNUM`-switched chain whose
    /// every arm is the *same* log-mu-ref pattern on the *same* eta, e.g.
    /// `CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)`.
    /// This is the only structured per-class θ mapping in the codebase — the
    /// class-specific typical values otherwise live purely as `MIXNUM`-branched
    /// expressions (see the type doc above).
    ///
    /// What each estimator does with it differs:
    ///
    /// - **IMP / IMPMAP** apply the responsibility-weighted per-class shift
    ///   `log θ_k += (Σ_i PMIX_ic · η̄_ic) / (Σ_i PMIX_ic)` to every anchor here,
    ///   switched and class-shared alike.
    /// - **SAEM** keeps genuinely class-*switched* anchors on the numerical
    ///   M-step — its hard per-subject class draw makes the per-class η mean a
    ///   biased classification-EM statistic — and takes the closed form only for
    ///   a class-*shared* anchor, i.e. one whose theta is the same in every class
    ///   slot. Those are also the ones `CompiledModel::mu_refs` can express.
    ///
    /// An eta appears here at most once, and only when its class-aware
    /// assignment is the *last* one for that eta, so this never contradicts
    /// `CompiledModel::mu_refs`.
    pub mu_refs: Vec<MixtureMuRef>,
}

/// A `MIXNUM`-switched log-mu-reference: one eta paired with one anchor theta
/// **per class** (#996).
///
/// Produced by the parser when every arm of a `MIXNUM` chain matches the same
/// mu-ref pattern on the same eta. `theta_names[c]` is the anchor for class
/// `c + 1`; the same theta name may repeat (a class-shared typical value), which
/// is what makes the degenerate one-theta case reduce to the classical pooled
/// mu-ref update.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MixtureMuRef {
    /// Eta this mu-reference is attached to.
    pub eta_name: String,
    /// Anchor theta name per class; `theta_names[c]` serves class `c + 1`.
    /// Length is always the mixture's `n_classes`.
    pub theta_names: Vec<String>,
    /// Always `true` today — only log-mu-ref patterns (`THETA*exp(ETA)` /
    /// `exp(log(THETA)+ETA)`) are detected, mirroring `get_mu_ref_pairs`'s
    /// filter. Kept explicit so an additive class-aware pattern can be added
    /// without changing the consumer's shape.
    pub log_transformed: bool,
}

/// Supported PK structural models.
///
/// IV (bolus and/or infusion) administration is represented by a single
/// variant per compartment count; the bolus-vs-infusion choice is made
/// per dose event from the dataset's RATE column (see
/// `DoseEvent::is_infusion`). This mirrors NONMEM, nlmixr2, and Monolix
/// and lets a subject mix bolus and infusion doses in one record.
/// Oral routes remain a separate variant because they have a distinct
/// model structure (absorption rate constant KA, bioavailability F).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PkModel {
    OneCptIv,
    OneCptOral,
    /// 1-cpt with Savic transit-compartment absorption via the analytic
    /// exponential-tilting closed form (#386). Requires `cl`, `v`, `n` (transit
    /// count), `mtt` (mean transit time); `f`/`lagtime` optional as for oral.
    OneCptTransit,
    /// 1-cpt with Freijer & Post inverse-Gaussian absorption via the analytic
    /// exponential-tilting closed form (#790) — the same `igd(mat, cv2)` density the
    /// ODE forcing path implements, skipping the ODE solve. Requires `cl`, `v`,
    /// `mat` (mean absorption time), `cv2` (relative dispersion); `f`/`lagtime`
    /// optional as for oral.
    OneCptIg,
    TwoCptIv,
    TwoCptOral,
    /// 2-cpt with Savic transit-compartment absorption via the analytic
    /// exponential-tilting closed form (#386 PR D). Requires `cl`, `v1`, `q`,
    /// `v2`, `n` (transit count), `mtt` (mean transit time); `f`/`lagtime`
    /// optional as for oral.
    TwoCptTransit,
    /// 2-cpt with Freijer & Post inverse-Gaussian absorption via the analytic
    /// exponential-tilting closed form (#790). Requires `cl`, `v1`, `q`, `v2`,
    /// `mat` (mean absorption time), `cv2` (relative dispersion); `f`/`lagtime`
    /// optional as for oral.
    TwoCptIg,
    ThreeCptIv,
    ThreeCptOral,
}

impl PkModel {
    /// Canonical PK slots that MUST be mapped in a `[structural_model]` `pk(...)`
    /// line for this model, each paired with the conventional name shown in
    /// parser errors. `f`/`lagtime` are intentionally absent — they are optional
    /// and default to 1.0 / 0.0 (see `PkParams::default`).
    ///
    /// Slots are canonical (`name_to_index` values), so the `v`/`v1` and `q`/`q2`
    /// aliases satisfy the same requirement. The display names mirror the
    /// "Required Parameters" table in `docs/model-file/structural-model.qmd`;
    /// the parser enforces what that table documents (issue #309).
    pub(crate) fn required_pk_params(&self) -> &'static [(usize, &'static str)] {
        match self {
            PkModel::OneCptIv => &[(PK_IDX_CL, "cl"), (PK_IDX_V, "v")],
            PkModel::OneCptOral => &[(PK_IDX_CL, "cl"), (PK_IDX_V, "v"), (PK_IDX_KA, "ka")],
            PkModel::OneCptTransit => &[
                (PK_IDX_CL, "cl"),
                (PK_IDX_V, "v"),
                (PK_IDX_N, "n"),
                (PK_IDX_MTT, "mtt"),
            ],
            PkModel::OneCptIg => &[
                (PK_IDX_CL, "cl"),
                (PK_IDX_V, "v"),
                (PK_IDX_MAT, "mat"),
                (PK_IDX_CV2, "cv2"),
            ],
            PkModel::TwoCptIv => &[
                (PK_IDX_CL, "cl"),
                (PK_IDX_V, "v1"),
                (PK_IDX_Q, "q"),
                (PK_IDX_V2, "v2"),
            ],
            PkModel::TwoCptOral => &[
                (PK_IDX_CL, "cl"),
                (PK_IDX_V, "v1"),
                (PK_IDX_Q, "q"),
                (PK_IDX_V2, "v2"),
                (PK_IDX_KA, "ka"),
            ],
            PkModel::TwoCptTransit => &[
                (PK_IDX_CL, "cl"),
                (PK_IDX_V, "v1"),
                (PK_IDX_Q, "q"),
                (PK_IDX_V2, "v2"),
                (PK_IDX_N, "n"),
                (PK_IDX_MTT, "mtt"),
            ],
            PkModel::TwoCptIg => &[
                (PK_IDX_CL, "cl"),
                (PK_IDX_V, "v1"),
                (PK_IDX_Q, "q"),
                (PK_IDX_V2, "v2"),
                (PK_IDX_MAT, "mat"),
                (PK_IDX_CV2, "cv2"),
            ],
            PkModel::ThreeCptIv => &[
                (PK_IDX_CL, "cl"),
                (PK_IDX_V, "v1"),
                (PK_IDX_Q, "q2"),
                (PK_IDX_V2, "v2"),
                (PK_IDX_Q3, "q3"),
                (PK_IDX_V3, "v3"),
            ],
            PkModel::ThreeCptOral => &[
                (PK_IDX_CL, "cl"),
                (PK_IDX_V, "v1"),
                (PK_IDX_Q, "q2"),
                (PK_IDX_V2, "v2"),
                (PK_IDX_Q3, "q3"),
                (PK_IDX_V3, "v3"),
                (PK_IDX_KA, "ka"),
            ],
        }
    }

    /// The canonical short model name (e.g. `one_cpt_oral`), used in parser
    /// diagnostics. Long-form aliases (`one_compartment_oral`) normalise to this.
    ///
    /// Deliberately the inverse of the string→`PkModel` match in
    /// `parse_structural_model` (which additionally accepts the long-form
    /// aliases, so the two can't be a single bidirectional table);
    /// `canonical_name_round_trips_through_parser` guards them against drift.
    pub(crate) fn canonical_name(&self) -> &'static str {
        match self {
            PkModel::OneCptIv => "one_cpt_iv",
            PkModel::OneCptOral => "one_cpt_oral",
            PkModel::OneCptTransit => "one_cpt_transit",
            PkModel::OneCptIg => "one_cpt_ig",
            PkModel::TwoCptIv => "two_cpt_iv",
            PkModel::TwoCptOral => "two_cpt_oral",
            PkModel::TwoCptTransit => "two_cpt_transit",
            PkModel::TwoCptIg => "two_cpt_ig",
            PkModel::ThreeCptIv => "three_cpt_iv",
            PkModel::ThreeCptOral => "three_cpt_oral",
        }
    }

    /// The (1-based) compartments the **analytical** engine can deliver a
    /// zero-order infusion into — i.e. the only compartments a modeled infusion
    /// *duration* `D{cmt}` (`RATE=-2`, #324/#394) may target. Single source of
    /// truth, kept in lockstep with the infusion-routing `match (pk_model, d.cmt)`
    /// in [`crate::pk::event_driven`] (and the equivalent superposition dispatch):
    /// the **central** compartment for every model, plus the **peripheral**
    /// compartment(s) for the 2-/3-cpt IV models, and — since #400 — the oral
    /// **depot** (cmt 1), a zero-order release into the depot followed by
    /// first-order `ka` absorption into central, and — since #375 — the oral
    /// models' **peripheral**(s) too (the oral propagators reuse the IV forced
    /// response; nothing flows back into the depot). That makes the set
    /// `1..=n_states` for all six walk-supported models; transit/IG stay `&[]`
    /// (they have no walk, and reroute an infusion to their ODE twin). A
    /// `D{cmt}` outside this set is rejected at parse time rather than silently
    /// mis-routed (no-TV path) or panicking (event-driven path).
    pub(crate) fn infusable_compartments(&self) -> &'static [usize] {
        self.topology().infusable_compartments()
    }

    /// Resolve a `[structural_model]` model name (canonical or long-form alias,
    /// e.g. `one_cpt_iv` / `one_compartment_iv`) to its `PkModel`. `None` for any
    /// unrecognised name (including the retired `*_bolus` / `*_infusion` spellings,
    /// which the parser handles separately with a migration error).
    ///
    /// The single source of truth for name → model, shared by the analytical `pk`
    /// parser (`parse_structural_model`) and the `ode_template` desugarer, so the
    /// accepted aliases can't drift between the two paths. The inverse of
    /// `canonical_name` (which omits the aliases); `from_name_round_trips_and_accepts_aliases`
    /// and `canonical_name_round_trips_through_parser` pin them together.
    pub(crate) fn from_name(name: &str) -> Option<PkModel> {
        match name {
            "one_cpt_iv" | "one_compartment_iv" => Some(PkModel::OneCptIv),
            "one_cpt_oral" | "one_compartment_oral" => Some(PkModel::OneCptOral),
            "one_cpt_transit" | "one_compartment_transit" => Some(PkModel::OneCptTransit),
            "one_cpt_ig" | "one_compartment_ig" => Some(PkModel::OneCptIg),
            "two_cpt_iv" | "two_compartment_iv" => Some(PkModel::TwoCptIv),
            "two_cpt_oral" | "two_compartment_oral" => Some(PkModel::TwoCptOral),
            "two_cpt_transit" | "two_compartment_transit" => Some(PkModel::TwoCptTransit),
            "two_cpt_ig" | "two_compartment_ig" => Some(PkModel::TwoCptIg),
            "three_cpt_iv" | "three_compartment_iv" => Some(PkModel::ThreeCptIv),
            "three_cpt_oral" | "three_compartment_oral" => Some(PkModel::ThreeCptOral),
            _ => None,
        }
    }

    /// Whether this is an **absorption** model — the dose enters an input
    /// compartment (cmt 1) and is absorbed into central (cmt 2), as opposed to IV
    /// (dose directly into central, cmt 1). True for first-order oral *and* the
    /// `one_cpt_transit` model. The name is historical: what it gates is the
    /// compartment layout / dose routing, not `ka` specifically (oral reads `ka`,
    /// transit reads `n`/`mtt`; which slots a model actually consumes is
    /// [`PkModel::consumes_pk_slot`] via [`PkModel::required_pk_params`], not this
    /// predicate). (`f` is read by every model since #327 — it scales IV doses too.)
    pub(crate) fn is_oral(&self) -> bool {
        matches!(
            self,
            PkModel::OneCptOral
                | PkModel::OneCptTransit
                | PkModel::OneCptIg
                | PkModel::TwoCptOral
                | PkModel::TwoCptTransit
                | PkModel::TwoCptIg
                | PkModel::ThreeCptOral
        )
    }

    /// Whether the analytical solver for this model actually reads the given PK
    /// slot. This is the single source of truth for "is a mapped param used", and
    /// it mirrors what the `pk/` closed forms consume — pinned to real solver
    /// behaviour by `consumes_pk_slot_matches_solver` in `pk/mod.rs`, so a future
    /// variant that reads a new slot can't silently drift from this:
    ///   - every required structural slot (`required_pk_params`);
    ///   - `lagtime`, applied to *every* dose (`predict_concentration` shifts the
    ///     effective dose time for IV and oral alike);
    ///   - `f` (bioavailability), applied to *every* dose and route — IV bolus,
    ///     infusion, and oral depot — scaling the bioavailable amount/rate
    ///     (#327). Both the superposition and event-driven analytical paths read
    ///     it for every model, IV included, so it is never inert.
    pub(crate) fn consumes_pk_slot(&self, slot: usize) -> bool {
        slot == PK_IDX_LAGTIME
            || slot == PK_IDX_F
            || self.required_pk_params().iter().any(|(s, _)| *s == slot)
    }
}

/// Supported residual error models
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorModel {
    Additive,
    Proportional,
    Combined,
}

/// How a sigma parameter enters the residual error model.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum SigmaType {
    Proportional,
    Additive,
}

impl ErrorModel {
    /// Return the `SigmaType` for each sigma, in the order they appear in `FitResult.sigma`.
    pub fn sigma_types(self) -> Vec<SigmaType> {
        match self {
            ErrorModel::Proportional => vec![SigmaType::Proportional],
            ErrorModel::Additive => vec![SigmaType::Additive],
            ErrorModel::Combined => vec![SigmaType::Proportional, SigmaType::Additive],
        }
    }

    /// Number of sigma parameters this error model consumes.
    pub fn n_sigma(self) -> usize {
        match self {
            ErrorModel::Additive | ErrorModel::Proportional => 1,
            ErrorModel::Combined => 2,
        }
    }
}

/// Closure signature for [`ErrorSelector::eval`]: maps one observation's
/// covariate snapshot to a 0-based endpoint key into `ErrorSpec::Selected
/// .endpoints` (issue #658). Total — the parser requires a final `else`, so
/// every observation resolves to some declared endpoint.
pub type ErrorSelectFn = Box<dyn Fn(&HashMap<String, f64>) -> usize + Send + Sync>;

/// Covariate-driven residual-error endpoint selector (issue #658).
///
/// Maps one observation's covariate snapshot to a 0-based endpoint key into the
/// `ErrorSpec::Selected.endpoints` map, resolving a user `if (COV …) { … } else
/// { … }` in `[error_model]`. The selection depends only on covariates — never
/// on θ/η or the prediction — so it is constant across the inner EBE loop and
/// the FOCE/FOCEI outer iterations (the σ-gradient rides the existing per-CMT
/// σ channel once the endpoint is chosen). The closure mirrors the boxed-closure
/// pattern used by [`ScaleFn`]; `branch_labels` carries source-order branch
/// descriptions purely for `Debug`/diagnostics.
pub struct ErrorSelector {
    /// Covariate map → 0-based endpoint key. Total: the parser requires a final
    /// `else`, so every observation resolves to some declared endpoint.
    pub eval: ErrorSelectFn,
    /// Human-readable branch descriptions (source order), for `Debug` output.
    pub branch_labels: Vec<String>,
}

impl std::fmt::Debug for ErrorSelector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ErrorSelector({:?})", self.branch_labels)
    }
}

/// Residual error specification for a model.
///
/// `Single` applies one error model to every observation (the default and the
/// only form analytical-PK models support). `PerCmt` dispatches a distinct
/// error model per observed compartment — the multi-endpoint / simultaneous
/// PK-PD case (issue #14). The map key is the 1-based CMT index from the data
/// file's CMT column, matching `subject.obs_cmts[i]`. `Selected` (issue #658)
/// dispatches on a key resolved per observation from a covariate `if/else`
/// selector instead of the CMT column; its `endpoints` map is keyed by the
/// selector's 0-based branch index and is otherwise identical in shape/handling
/// to `PerCmt` (every downstream method treats the two alike — see the shared
/// `PerCmt(map) | Selected { endpoints: map, .. }` arms).
pub enum ErrorSpec {
    /// One error model for all observations.
    Single(ErrorModel),
    /// Per-CMT error models. Each endpoint carries its own `ErrorModel` and the
    /// indices into the flat global `sigma.values` vector that supply its
    /// sigmas (declaration order in the `[parameters]` block).
    PerCmt(HashMap<usize, EndpointError>),
    /// Covariate-selected error models (issue #658). `selector` resolves each
    /// observation's covariate snapshot to a 0-based key into `endpoints`; the
    /// `EndpointError` values carry the same `error_model` + `sigma_idx` layout
    /// as `PerCmt`.
    Selected {
        selector: ErrorSelector,
        endpoints: HashMap<usize, EndpointError>,
    },
}

impl std::fmt::Debug for ErrorSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorSpec::Single(em) => f.debug_tuple("Single").field(em).finish(),
            ErrorSpec::PerCmt(map) => f.debug_tuple("PerCmt").field(map).finish(),
            ErrorSpec::Selected {
                selector,
                endpoints,
            } => f
                .debug_struct("Selected")
                .field("selector", selector)
                .field("endpoints", endpoints)
                .finish(),
        }
    }
}

impl Default for ErrorSpec {
    fn default() -> Self {
        ErrorSpec::Single(ErrorModel::Additive)
    }
}

impl ErrorSpec {
    /// Per-observation endpoint dispatch keys for `subject`.
    ///
    /// For `Single`/`PerCmt` this is exactly the data CMT column
    /// (`subject.obs_cmts`, borrowed with no allocation). For `Selected` the
    /// covariate selector is evaluated on each observation's covariate snapshot
    /// (`subject.obs_cov(j)`, which falls back to the subject-level covariates
    /// when no per-observation snapshot exists — the per-row Form C semantics).
    /// The returned keys are what every residual-dispatch method
    /// (`variance_at`, `sigma_loadings`, `dvar_df`, …) expects as its `cmt`
    /// argument.
    pub fn obs_keys<'a>(&self, subject: &'a Subject) -> std::borrow::Cow<'a, [usize]> {
        match self {
            ErrorSpec::Selected { selector, .. } => std::borrow::Cow::Owned(
                (0..subject.obs_cmts.len())
                    .map(|j| (selector.eval)(subject.obs_cov(j)))
                    .collect(),
            ),
            _ => std::borrow::Cow::Borrowed(&subject.obs_cmts),
        }
    }

    /// Endpoint dispatch key for a single observation `j` of `subject`. The
    /// `O(1)` analogue of [`obs_keys`](Self::obs_keys) for callers that resolve
    /// one observation at a time (e.g. the per-observation simulation loop),
    /// avoiding a full key-vector allocation per call for `Selected` specs.
    pub fn obs_key(&self, subject: &Subject, j: usize) -> usize {
        match self {
            ErrorSpec::Selected { selector, .. } => (selector.eval)(subject.obs_cov(j)),
            _ => subject.obs_cmts[j],
        }
    }
}

/// Per-observation multiplier for one flat sigma slot (#484).
///
/// Evaluates the residual-error *magnitude factor* for a single observation
/// from `(theta, observation-covariate map, TIME)`. The factor multiplies that
/// sigma's loading, so the slot's variance contribution becomes
/// `(loading · factor · sigma)²`. Inputs are deliberately limited to
/// quantities that do **not** depend on the random effects (η) or the
/// prediction beyond the built-in proportional loading, so the multiplier is
/// constant across the inner EBE loop for a fixed θ. The covariate map must
/// supply any model covariates the expression names; `TIME` is provided by the
/// parser's event-time built-in.
pub type RuvMagFn = Box<dyn Fn(&[f64], &HashMap<String, f64>, f64) -> f64 + Send + Sync>;

/// Closure signature for a kappa's `weight = <expr>` (#1031): `(theta,
/// covariates, time) -> weight`. Same inputs as [`RuvMagFn`] and for the same
/// reason — a variance weight may not depend on the random effects it scales.
pub type KappaWeightFn = Box<dyn Fn(&[f64], &HashMap<String, f64>, f64) -> f64 + Send + Sync>;

/// A kappa's declared sample-size weight (#1031): `kappa K ~ γ² weight = W`
/// means `κ_ik ~ N(0, Ω_IOV / W_ik)`.
///
/// Carried on [`CompiledModel::kappa_weights`] for reporting and validation
/// only — the scaling itself is desugared into the individual-parameter
/// expressions at parse time.
pub struct KappaWeight {
    /// The weight expression exactly as written in the model file.
    pub expr: String,
    /// Evaluates the weight for one record's covariate snapshot.
    pub eval: KappaWeightFn,
}

impl std::fmt::Debug for KappaWeight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KappaWeight")
            .field("expr", &self.expr)
            .finish_non_exhaustive()
    }
}

/// Custom residual-error magnitude (#484): one optional multiplier per flat
/// sigma slot, indexed positionally to match `ErrorSpec::Single`'s sigma
/// consumption order. `per_sigma[k] == None` means slot `k` is a bare sigma
/// (multiplier ≡ 1, the legacy path); `Some(f)` makes the slot's magnitude
/// vary with TIME / covariates / theta.
///
/// `#[non_exhaustive]`: `per_sigma_deriv` (#576/#486) is the first field added
/// since this type shipped; marking the struct non-exhaustive now means a
/// future field addition can't again break an external struct-literal
/// construction — build via [`Default::default()`] plus field assignment (or
/// `..Default::default()`) instead.
#[derive(Default)]
#[non_exhaustive]
pub struct RuvMagnitude {
    pub per_sigma: Vec<Option<RuvMagFn>>,
    /// Whether any slot's magnitude expression references a `θ` (#484/#1029).
    ///
    /// A θ-free magnitude — every `weight = <covariate expr>` model, and the
    /// TIME/covariate-only #484 forms — is a per-observation *constant* for a
    /// fixed dataset, so an analytic θ gradient that chains only through the
    /// prediction (`∂nll/∂f · ∂f/∂θ`) is still exact. A θ-*dependent* one adds a
    /// direct `∂V/∂θ` channel that such a gradient omits, so the estimators
    /// without one (the Gauss-Newton Laplace/Sheiner–Beal gradients) consult
    /// this flag and route those subjects to their magnitude-aware
    /// finite-difference fallback instead.
    pub theta_dependent: bool,
    /// Parallel to `per_sigma`: the compiled `Dual1` θ-derivative program for the
    /// same non-bare slot (#576/#486), or `None` for a bare slot. Lets the
    /// analytic outer θ/σ gradient differentiate the magnitude directly — a new
    /// direct-θ term in `∂R/∂θ` — instead of falling back to FD. The runtime
    /// closure in `per_sigma` still serves every value-only caller (inner loop,
    /// IWRES, simulation).
    pub per_sigma_deriv: Vec<Option<crate::parser::model_parser::RuvMagDerivProgram>>,
}

impl std::fmt::Debug for RuvMagnitude {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let slots: Vec<&str> = self
            .per_sigma
            .iter()
            .map(|s| if s.is_some() { "expr" } else { "bare" })
            .collect();
        write!(f, "RuvMagnitude({:?})", slots)
    }
}

impl RuvMagnitude {
    /// Whether any slot carries a non-trivial multiplier. A `RuvMagnitude` whose
    /// slots are all `None` behaves identically to no custom magnitude.
    pub fn is_active(&self) -> bool {
        self.per_sigma.iter().any(|s| s.is_some())
    }

    /// Per-sigma multiplier vector for one observation. Slot `k` evaluates its
    /// program; bare slots return `1.0`. The returned vector has length
    /// `per_sigma.len()`.
    pub fn eval_obs(
        &self,
        theta: &[f64],
        obs_covariates: &HashMap<String, f64>,
        time: f64,
    ) -> Vec<f64> {
        self.per_sigma
            .iter()
            .map(|slot| match slot {
                Some(f) => f(theta, obs_covariates, time),
                None => 1.0,
            })
            .collect()
    }

    /// Per-sigma `∂(multiplier)/∂θ` vector for one observation (#576/#486). Slot
    /// `k` evaluates its `Dual1` program's gradient; a bare slot (multiplier ≡ 1)
    /// contributes an all-zero row. `None` when any active slot's program
    /// declines (θ-axis count mismatch / beyond `MAX_RUV_MAG_AXES`) — the caller
    /// then falls back to FD for the magnitude's direct-θ gradient channel.
    pub fn eval_obs_theta_grad(
        &self,
        theta: &[f64],
        obs_covariates: &HashMap<String, f64>,
        time: f64,
    ) -> Option<Vec<Vec<f64>>> {
        self.per_sigma_deriv
            .iter()
            .map(|slot| match slot {
                Some(p) => p.theta_grad(theta, obs_covariates, time),
                None => Some(vec![0.0; theta.len()]),
            })
            .collect()
    }
}

/// One endpoint of a multi-endpoint (`ErrorSpec::PerCmt`) residual error model.
#[derive(Debug, Clone)]
pub struct EndpointError {
    pub error_model: ErrorModel,
    /// Positions into the flat global `sigma.values` supplying this endpoint's
    /// sigmas. Length equals `error_model.n_sigma()`.
    pub sigma_idx: Vec<usize>,
}

/// Transformation applied to a theta on the natural scale.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThetaTransform {
    /// Theta is on the natural scale (no transformation).
    Identity,
    /// Theta is on the log scale; back-transform = exp(theta).
    Log,
    /// Theta is on the logit scale: `inv_logit(THETA + ETA)`. User sets THETA
    /// on the logit scale (e.g. logit(0.7) ≈ 0.847).
    Logit,
    /// Theta is on the probability scale: `inv_logit(logit(THETA) + ETA)`.
    /// User sets THETA directly in (0,1) (e.g. 0.70 for 70% bioavailability).
    LogitProbability,
}

/// Distribution / parameterisation of an ETA random effect.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum EtaParamType {
    /// `TVCL * exp(ETA)` or `exp(THETA + ETA)` — log-normal.
    LogNormal,
    /// `TVCL + ETA` — normal (additive).
    Additive,
    /// `inv_logit(THETA + ETA)` — logit-normal; THETA on the logit scale.
    Logit,
    /// `inv_logit(logit(THETA) + ETA)` — logit-normal; THETA on the (0,1) scale.
    LogitProbability,
    /// Pattern not automatically recognised.
    Custom,
}

/// Per-ETA transformation metadata, carried in `FitResult`.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct EtaParamInfo {
    pub eta_name: String,
    pub param_type: EtaParamType,
    /// Theta paired with this ETA. Set only when the ETA is added directly to a single THETA in
    /// the same expression (e.g. `THETA * exp(ETA)` or `inv_logit(THETA + ETA)`).
    /// Not set for mu-ref patterns like `TVCL * exp(ETA)` where the THETA is a scale factor.
    pub linked_theta: Option<String>,
    /// Name of the individual parameter this ETA appears in (e.g. `"CL"`).
    pub individual_param_name: String,
}

/// PK parameter function: maps `(theta, eta, covariates, time)` -> PkParams.
///
/// `time` is the event/observation time on the PK timeline. It feeds the `TIME`
/// built-in (`Expression::Time`) and the analytical `pk_time_mapping` slots: the
/// closure sets the model-time thread-local from it for the duration of the
/// evaluation. Threading it through the signature (rather than relying on the
/// caller to wrap in `with_model_time`) makes the compiler force every call site
/// to supply the time, so no path can silently evaluate `TIME` at `0.0` (#610).
/// Callers for which `TIME` is irrelevant (the model does not use the built-in)
/// may pass any value, e.g. `0.0`.
pub type PkParamFn =
    Box<dyn Fn(&[f64], &[f64], &HashMap<String, f64>, f64) -> PkParams + Send + Sync>;

/// Closure signature for `CompiledModel::tv_fn`: the typical-value evaluator.
/// Receives `(theta, covariates)` and returns one typical value per
/// `[individual_parameters]` assignment, in declaration order — parallel to
/// `pk_indices` and `eta_map`.
pub type TvFn = Box<dyn Fn(&[f64], &HashMap<String, f64>) -> Vec<f64> + Send + Sync>;

/// Closure signature for `[scaling] obs_scale = <expr>` (Form B). Receives
/// `(theta, eta, covariates, pk_params)` and returns the per-subject scale
/// factor used to divide the raw prediction. `pk_params` is the subject-
/// static evaluation of `model.pk_param_fn`, so the scale expression can
/// reference individual parameters (e.g. `obs_scale = 1000 / V`) — the
/// closure looks up V via its PK slot in `pk_params.values`.
pub type ScaleFn =
    Box<dyn Fn(&[f64], &[f64], &HashMap<String, f64>, &PkParams) -> f64 + Send + Sync>;

/// A non-zero initial compartment amount on an **analytical** PK model
/// (`[initial_conditions] init(NAME) = <expr>`; issue #521). The analytical
/// engine has no state vector to seed, so an initial amount `A₀` in
/// compartment `cmt` is layered onto the dose-driven prediction as the
/// closed-form impulse response of an `F`-bypassed bolus of `A₀` into `cmt`
/// at t=0 — exact for the linear superposition models. This mirrors NONMEM's
/// `A_0(cmt)` and the ODE path's `init(...)` directive, which seeds the
/// integrator state instead.
///
/// `amount_fn` shares the [`ScaleFn`] signature: it receives
/// `(theta, eta, covariates, pk_params)` and returns `A₀`, so the expression
/// may reference thetas, etas, covariates, and individual parameters (e.g.
/// `init(central) = CONC0 * V`).
///
/// `#[non_exhaustive]`: constructed only inside the crate (the parser), and the
/// field set grew once already (`amount_deriv`, #524). The marker keeps adding
/// further compiled-program fields a non-breaking change for external callers.
#[non_exhaustive]
pub struct AnalyticalInit {
    /// 1-based compartment index the initial amount is deposited into.
    pub cmt: usize,
    /// Evaluates the initial amount `A₀` for a subject. See type docs.
    pub amount_fn: ScaleFn,
    /// The same `A₀` expression compiled to a `Dual2`-differentiable program
    /// (issue #524), so the analytic FOCE/FOCEI provider can differentiate the
    /// init impulse `A₀ · kernel(t, pk)` exactly instead of falling back to
    /// finite differences. Reuses [`ScaleDerivProgram`] — the init amount has the
    /// same `(θ, η, individual-param, covariate)` shape as an `obs_scale`
    /// expression. `None` only for hand-constructed inits with no parsed
    /// expression (those keep the FD fallback).
    pub amount_deriv: Option<crate::parser::model_parser::ScaleDerivProgram>,
}

/// A full Form C output readout (`[scaling] y = <expr>`) lifted onto an
/// **analytical** PK model (issue #650).
///
/// Analytical closed forms return the built-in central-compartment
/// *concentration*. A Form C readout **replaces** that concentration with an
/// arbitrary expression that may reference compartment **amounts** by name
/// (`central` always; `depot` for oral models — peripheral amounts are rejected
/// at parse, mirroring `[initial_conditions]`), individual parameters, thetas,
/// etas, covariates, and `if/else`. This is the analytic analogue of the ODE
/// Form C readout stored on [`crate::ode::OdeSpec::readout`]; analytical models
/// have `ode_spec == None`, so the readout lives here instead.
///
/// The predictor reconstructs the compartment-amount vector (central =
/// concentration × `V`; depot = superposed `F·D·exp(-ka·t)`) in `state_names`
/// order and feeds it, plus individual params / covariates, to the shared
/// `PkNum`-generic evaluator [`crate::parser::model_parser::OdeOutputProgram::eval_output_g`]
/// — the same one the ODE provider uses — so FOCE/FOCEI sensitivities stay
/// analytic over `Dual2`/`Dual1` (no finite-difference fallback on the
/// supported paths).
///
/// `#[non_exhaustive]`: constructed only inside the crate (the parser).
#[non_exhaustive]
pub struct AnalyticReadout {
    /// The compiled readout: `OdeReadout::Single` for uniform `y = <expr>`,
    /// `OdeReadout::PerCmt` for per-CMT `y[CMT=N] = <expr>`. Reuses the ODE
    /// readout enum and its f64 `OdeOutputFn` closure(s).
    pub readout: crate::ode::OdeReadout,
    /// Sensitivity program for the uniform `Single` readout (issue #650), so the
    /// analytic provider can differentiate it over `Dual2`/`Dual1`. `None` for
    /// per-CMT (each `PerCmtReadout` carries its own program) and for readouts
    /// that are not dual-evaluable (those fall back to FD, matching ODE).
    pub program: Option<crate::parser::model_parser::OdeOutputProgram>,
    /// Canonical compartment names in `state[]` order — `["central"]` for IV
    /// models, `["depot", "central"]` for oral. Parallel to the amount vector
    /// the predictor builds and to the readout program's `n_states` layout.
    pub state_names: Vec<String>,
}

impl AnalyticReadout {
    /// Whether the readout reads the oral **depot** amount (state slot 0 in the
    /// `["depot", "central"]` layout).
    ///
    /// The depot amount is reconstructed by dose superposition, which is invalid
    /// across EVID=3/4 resets — a reset zeros the compartments and the closed
    /// form has no way to restart the accumulation. So `fit`/`predict` reject a
    /// depot-referencing readout on a subject that carries a reset, rather than
    /// silently feeding a zero depot into the readout (issue #650 review). A
    /// readout that only reads `central` is unaffected (the central concentration
    /// is already reset-correct).
    pub(crate) fn references_depot(&self) -> bool {
        if self.state_names.first().map(String::as_str) != Some("depot") {
            return false;
        }
        match &self.readout {
            crate::ode::OdeReadout::ObsCmt(_) => false,
            crate::ode::OdeReadout::Single(_) => {
                self.program.as_ref().is_some_and(|p| p.references_state(0))
            }
            crate::ode::OdeReadout::PerCmt(map) => map
                .values()
                .any(|r| r.program.as_ref().is_some_and(|p| p.references_state(0))),
        }
    }
}

/// How the structural model's raw output is mapped to the observed `DV`.
///
/// Set by the `[scaling]` block in `.ferx` model files. The convention is
/// **divisive**: `pred_scaled = pred_raw / scale`. This matches the natural
/// reading of `obs_scale = V/1000` as "divide amount by V/1000 to get
/// concentration in the user's units."
///
/// Forms A/B (this enum) post-multiply analytical and ODE predictions
/// uniformly at the end of the prediction dispatcher. Form C (ODE-only
/// `y = <expr>`) is handled inside the ODE timeline loop via
/// `OdeSpec::output_fn` instead — it replaces the state readout entirely,
/// so it doesn't share the post-multiply path.
#[derive(Default)]
pub enum ScalingSpec {
    /// No scaling: prediction is returned as-is.
    #[default]
    None,
    /// Constant divisor applied to every prediction.
    ScalarScale(f64),
    /// Per-subject divisor evaluated from `(theta, eta, covariates, pk)`.
    /// Used for expressions like `obs_scale = 1000 / V`. `deriv` is the same
    /// expression compiled to a `Dual2`-differentiable program (issue #367), so
    /// the analytic sensitivity provider can differentiate `f / scale` exactly;
    /// `None` for hand-constructed specs / closures with no parsed expression
    /// (those fall back to finite differences).
    ExpressionScale {
        scale_fn: ScaleFn,
        deriv: Option<crate::parser::model_parser::ScaleDerivProgram>,
    },
    /// Per-CMT dispatch for multi-analyte models (parent+metabolite,
    /// sum-of-moieties, free vs total, ...). Key is the 1-based CMT
    /// index from the data file's CMT column (matches
    /// `subject.obs_cmts[i]`, which is `usize`). Each entry is one of
    /// `None` / `ScalarScale` / `ExpressionScale` (no nested `PerCmt`
    /// — parser enforces).
    ///
    /// Fit-time validation requires every observed CMT in the population
    /// to have an entry; missing entries fall through to NaN predictions
    /// at runtime as a defensive guard against hand-constructed
    /// CompiledModels.
    PerCmt(HashMap<usize, ScalingSpec>),
}

impl ScalingSpec {
    /// Materialise the per-observation scale factors for one subject.
    ///
    /// Returns a `Vec<f64>` of length `obs_cmts.len()`. The vector is the
    /// canonical source of truth used by both the FD path
    /// (`pk::apply_scaling`) and the AD path (threaded as a `Const` slice
    /// into the four AD entry points). Keeping a single materialiser
    /// avoids drift between the two — every change to the scaling
    /// semantics is felt by both paths automatically.
    ///
    /// All variants are subject-static: the closure (`ExpressionScale`)
    /// is evaluated at most once per subject. AD therefore treats the
    /// scale as a constant w.r.t. eta. For a genuinely eta-independent
    /// scale (`WT/70`, `TVV/1000` — covariates/thetas only) this materialised
    /// value is the whole story. For a scale that depends on eta (e.g.
    /// `obs_scale = V` with `V = TVV*exp(ETA_V)`, or `1000/V`) this frozen
    /// materialisation drops `d obs_scale / d eta`, which is why the live
    /// FOCE/FOCEI gradient does NOT take the η/θ derivative from it: since #486
    /// the inner and outer gradients apply the exact quotient rule analytically
    /// (`provider::apply_expression_scale[_inner]`), with FD kept only for
    /// LTBS / TV-cov / IOV combinations. See `docs/model-file/scaling.qmd`.
    /// (The retired-AD-era classifier `inner_optimizer::analytical_ad_unsupported`
    /// / `ScalingSpec::breaks_ad_inner_gradient` is vestigial and routes nothing.)
    ///
    /// Invalid scale values (0, negative, NaN, inf — e.g. from a covariate
    /// that's missing, or from a `1/(TVV-x)` near a singularity) propagate
    /// as `NaN` so the downstream divide produces a NaN prediction and
    /// the outer NLL goes NaN, matching the established loud-failure
    /// semantic used everywhere else in the scaling path.
    pub fn build_obs_scale_array(
        &self,
        theta: &[f64],
        eta: &[f64],
        covariates: &HashMap<String, f64>,
        pk: &PkParams,
        obs_cmts: &[usize],
    ) -> Vec<f64> {
        let n = obs_cmts.len();
        match self {
            Self::None => vec![1.0; n],
            Self::ScalarScale(k) => {
                let v = if *k > 0.0 && k.is_finite() {
                    *k
                } else {
                    f64::NAN
                };
                vec![v; n]
            }
            Self::ExpressionScale { scale_fn, .. } => {
                let s = scale_fn(theta, eta, covariates, pk);
                let v = if s > 0.0 && s.is_finite() {
                    s
                } else {
                    f64::NAN
                };
                vec![v; n]
            }
            Self::PerCmt(map) => obs_cmts
                .iter()
                .map(|cmt| match map.get(cmt) {
                    Some(inner) => {
                        let s = match inner {
                            Self::None => 1.0,
                            Self::ScalarScale(k) => *k,
                            Self::ExpressionScale { scale_fn, .. } => {
                                scale_fn(theta, eta, covariates, pk)
                            }
                            Self::PerCmt(_) => {
                                // Nested PerCmt is rejected at parse time;
                                // a hand-constructed CompiledModel that
                                // bypasses validation lands here. Return
                                // NaN (loud failure → NaN OFV) rather than
                                // 1.0, which would silently produce a
                                // mis-scaled fit in release builds where
                                // the debug_assert is stripped. (Caught by
                                // Copilot review on PR #85.)
                                debug_assert!(false, "nested PerCmt rejected at parse time");
                                f64::NAN
                            }
                        };
                        if s > 0.0 && s.is_finite() {
                            s
                        } else {
                            f64::NAN
                        }
                    }
                    // Validation in `fit()` catches this; defensive NaN.
                    None => f64::NAN,
                })
                .collect(),
        }
    }

    /// Returns true if this spec is a Form C readout shape that the AD
    /// path can't represent (currently only Form C lives outside
    /// `ScalingSpec` — see `OdeReadout::requires_fd`).
    ///
    /// `ScalingSpec` variants — including `ExpressionScale` and
    /// `PerCmt` — are all AD-compatible now via the per-observation scale
    /// array produced by `build_obs_scale_array`. Always returns `false`.
    /// Kept for symmetry with `OdeReadout::requires_fd` and forward
    /// compatibility (e.g. a future variant that genuinely can't fold
    /// into a Const slice).
    #[inline]
    pub fn requires_fd(&self) -> bool {
        match self {
            Self::None | Self::ScalarScale(_) | Self::ExpressionScale { .. } | Self::PerCmt(_) => {
                false
            }
        }
    }

    /// Returns true when materialising this spec needs a `pk_param_fn`
    /// evaluation. Only `ExpressionScale` consults `pk` (either directly
    /// or as an inner entry inside `PerCmt`). Lets callers skip the
    /// pk eval — which may be expensive on models with parsed expressions
    /// or NN forward passes — for the common `None` / `ScalarScale`
    /// cases. (Caught by Copilot review on PR #85.)
    #[inline]
    pub fn needs_pk_eval(&self) -> bool {
        match self {
            Self::None | Self::ScalarScale(_) => false,
            Self::ExpressionScale { .. } => true,
            Self::PerCmt(map) => map.values().any(Self::needs_pk_eval),
        }
    }

    /// **VESTIGIAL** (retained for the retired Enzyme-AD inner path; not consulted
    /// by any live routing — its only caller, `inner_optimizer::analytical_ad_unsupported`,
    /// is itself dead). Historically returned `true` when an `ExpressionScale` made the
    /// frozen-materialisation AD inner gradient unsafe, so the inner loop fell back to FD.
    ///
    /// Post-#486 this is **no longer how an η-dependent `obs_scale` is handled**: the live
    /// FOCE/FOCEI inner and outer gradients apply the exact η/θ quotient rule analytically
    /// (`provider::apply_expression_scale[_inner]`); only LTBS / TV-cov / IOV combinations
    /// route to FD, via their own gates — not this predicate. The historical ~12 OFV gap
    /// referenced the frozen-scale AD approximation, which is no longer used. Kept solely
    /// so the dead classifier compiles; do not re-wire it into routing without first
    /// removing the analytic `ExpressionScale` handling.
    #[inline]
    pub fn breaks_ad_inner_gradient(&self) -> bool {
        match self {
            Self::None | Self::ScalarScale(_) => false,
            Self::ExpressionScale { .. } => true,
            Self::PerCmt(map) => map.values().any(Self::breaks_ad_inner_gradient),
        }
    }
}

impl std::fmt::Debug for ScalingSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "ScalingSpec::None"),
            Self::ScalarScale(k) => write!(f, "ScalingSpec::ScalarScale({})", k),
            Self::ExpressionScale { .. } => write!(f, "ScalingSpec::ExpressionScale {{ .. }}"),
            Self::PerCmt(map) => {
                let cmts: Vec<usize> = {
                    let mut v: Vec<usize> = map.keys().copied().collect();
                    v.sort();
                    v
                };
                write!(f, "ScalingSpec::PerCmt({:?})", cmts)
            }
        }
    }
}

/// Associates an ETA with its mu-referencing anchor theta.
#[derive(Debug, Clone)]
pub struct MuRef {
    pub theta_name: String,
    /// true for patterns THETA*exp(ETA) or exp(log(THETA)+ETA); false for THETA+ETA
    pub log_transformed: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
//  Non-Gaussian endpoint types  (Phase 1: TTE / survival)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "survival")]
/// Censoring type for a TTE event record.
#[derive(Debug, Clone)]
pub enum EventType {
    Exact,
    RightCensored,
    /// Event occurred in the half-open interval (left, right].
    IntervalCensored {
        left: f64,
        right: f64,
    },
}

/// A single non-Gaussian observation record on a subject.
///
/// Compiled unconditionally (Phase 4.0) so the categorical/count (Track C) and
/// Markov (Track D) endpoints share one observation stream on the default build.
/// Only the TTE `Event` variant stays behind the `survival` feature — the same
/// mixed-`cfg` shape `SimOutcome` already uses. The record variant is chosen by
/// the CMT's declared endpoint, never guessed from the DV value (§8.1).
#[derive(Debug, Clone)]
pub enum ObsRecord {
    /// TTE / survival event observation (gated behind the `survival` feature).
    #[cfg(feature = "survival")]
    Event {
        time: f64,
        event_type: EventType,
        /// Left truncation / delayed entry time (0.0 when none).
        /// The likelihood conditions on survival past entry_time:
        ///   H_eff(T) = H(T) − H(entry_time)
        entry_time: f64,
        cmt: usize,
    },
    /// Discrete-state observation: an integer category or Markov state index.
    /// Serves binary/ordinal (Track C) and DTMM/CTMM state observations
    /// (Track D); the CMT's declared endpoint disambiguates the meaning.
    /// Discrete-state (binary / categorical / Markov) observation. `time` is the
    /// **internal** occasion-shifted clock the engine and likelihood run on;
    /// `raw_time` is the TIME the input CSV carried, and is what every reported row
    /// (sdtab, `simulate()`, `predict_categorical`) must use so output joins back to
    /// the input and to covtab. They differ only for reset-delimited occasions.
    DiscreteState {
        time: f64,
        raw_time: f64,
        state: usize,
        cmt: usize,
    },
    /// Non-negative integer count observation (Poisson / negative-binomial, Track C).
    /// Carries both clocks for the same reason [`Self::DiscreteState`] does — reporting
    /// must use `raw_time` or the rows cannot be joined back to the input.
    Count {
        time: f64,
        raw_time: f64,
        count: u32,
        cmt: usize,
    },
}

#[cfg(feature = "survival")]
/// Analytic parametric hazard families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HazardFamily {
    Exponential,
    Weibull,
    Gompertz,
}

#[cfg(feature = "survival")]
/// Closure type for computing hazard parameters from (theta, eta, covariates).
///
/// Return layout by family:
///   Exponential: `[lambda]`
///   Weibull:     `[scale, shape]`
///   Gompertz:    `[alpha, gamma, loghr_term]`  (loghr_term cumulates log-hazard
///                 contributions from covariates; 0.0 when no covariates)
pub type HazardParamFn =
    Box<dyn Fn(&[f64], &[f64], &HashMap<String, f64>) -> Vec<f64> + Send + Sync>;

#[cfg(feature = "survival")]
/// Hazard specification for a TTE endpoint.
pub enum HazardSpec {
    Analytic {
        family: HazardFamily,
        param_fn: HazardParamFn,
    },
    /// Drug-driven hazard accumulated as an extra ODE state (joint PK-TTE, Phase 2).
    ///
    /// The parser appends a synthetic cumulative-hazard compartment to the model's
    /// ODE system (`__chz' = hazard`, initial value 0); `chz_state` is that
    /// compartment's index in the ODE state vector. The TTE likelihood reads
    /// `H(t)` from the integrated state `u[chz_state]` and `h(t)` from its
    /// derivative `du[chz_state]`, rather than from a closed-form family.
    OdeAccumulated { chz_state: usize },
}

#[cfg(feature = "survival")]
impl std::fmt::Debug for HazardSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HazardSpec::Analytic { family, .. } => {
                write!(f, "HazardSpec::Analytic({family:?})")
            }
            HazardSpec::OdeAccumulated { chz_state } => {
                write!(f, "HazardSpec::OdeAccumulated(chz_state={chz_state})")
            }
        }
    }
}

#[cfg(feature = "survival")]
/// Hazard clock for a **repeated** TTE (RTTE) endpoint — how time is measured
/// between successive events (§3.3 of `plans/tte-survival-markov.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtteClock {
    /// Clock-forward / total time (Andersen–Gill), the default. The hazard is a
    /// function of *absolute* time; the cumulative hazard accumulates continuously
    /// over `[0, T]` and is **not** reset at events. NLL `Σ_k log h(t_k) − H(T)`.
    Forward,
    /// Clock-reset / gap time (renewal). The hazard depends on time *since the
    /// previous event*; the accumulator resets to 0 at each event. NLL
    /// `Σ_k log h(Δ_k) − Σ_k H(Δ_k)`. For an analytic hazard family this is exact with
    /// no ODE — the closed form is evaluated on each gap `Δ_k`
    /// ([`crate::survival::rtte_reset_nll_from_curves`]). A drug-driven ODE hazard would
    /// need the selective per-state reset (§8.8.6) and is still rejected at parse.
    Reset,
}

#[cfg(feature = "survival")]
/// Whether a TTE endpoint observes a single event per subject (standard TTE /
/// competing risks) or repeated events (RTTE). Orthogonal to [`HazardSpec`]'s
/// analytic-vs-ODE axis: a recurrent endpoint can carry either hazard shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TteRecurrence {
    /// Standard TTE: at most one event (plus optional censoring) per subject.
    /// The per-record likelihood terms are independent (Phase 1 / 1b / 2).
    Single,
    /// Repeated TTE (RTTE): multiple events per subject on one endpoint. The
    /// cumulative-hazard terms couple across a subject's records — see
    /// [`crate::survival::rtte_forward_nll_from_curves`].
    Repeated { clock: RtteClock },
}

#[cfg(feature = "survival")]
/// Link function for a categorical endpoint's linear predictor. Only `Logit`
/// (log-odds) is implemented in Phase 4 slice 1; probit/cloglog are rejected at
/// parse with a "not yet supported" error so the surface stays forward-compatible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkFn {
    /// Logistic (log-odds) link: `p = 1 / (1 + exp(-lp))`.
    Logit,
}

#[cfg(feature = "survival")]
/// Closure computing a categorical endpoint's linear predictor `lp` from
/// `(theta, eta, covariates)` — the log-odds for a `Binary` logit endpoint.
/// Mirrors [`HazardParamFn`]: evaluated once per observation record with a
/// *baseline* covariate snapshot (no time argument), so a time-varying covariate
/// on the predictor would be silently frozen and is guarded at parse.
pub type LinearPredictorFn =
    Box<dyn Fn(&[f64], &[f64], &HashMap<String, f64>) -> f64 + Send + Sync>;

#[cfg(feature = "markov")]
/// Closure building a CTMM generator matrix `Q(θ, η, covariates, state)` — the
/// S×S transition-intensity (generator) matrix for a continuous-time Markov
/// endpoint.
///
/// The off-diagonals `q_jk ≥ 0` (`j ≠ k`) are the `[markov_model] transition`
/// intensities; the builder fills the diagonal row-sum-zero
/// (`q_jj = −Σ_{k≠j} q_jk`) so the caller always receives a valid generator.
///
/// The `state` slice is the model's **ODE state vector at the current time**,
/// supplied only on the **time-inhomogeneous** (drug/PD-driven `Q(t)`, Phase 6,
/// #817) path so an intensity may reference an ODE state by name (e.g. a
/// concentration `central/V` or a PD response compartment). The builder resolves
/// those names against `state` via the endpoint's
/// [`generator_states`](EndpointLikelihood::Ctmm::generator_states) index map.
/// On the **time-homogeneous** (Phase 5) path no intensity references a state, so
/// callers pass an empty slice and the argument is ignored — the generator is
/// then a pure function of a *baseline* `(θ, η, covariates)` snapshot, and a
/// time-varying covariate on an intensity would be silently frozen (guarded at
/// parse/fit-setup).
pub type GeneratorFn = Box<
    dyn Fn(&[f64], &[f64], &HashMap<String, f64>, &[f64]) -> nalgebra::DMatrix<f64> + Send + Sync,
>;

#[cfg(feature = "survival")]
/// Per-CMT endpoint likelihood specification.
pub enum EndpointLikelihood {
    Gaussian(EndpointError),
    Tte {
        hazard: HazardSpec,
        /// Single event (standard TTE) vs. repeated events (RTTE). Defaults to
        /// `Single`; set to `Repeated` only when the model declares `type = rtte`.
        recurrence: TteRecurrence,
        /// Covariate names referenced by this hazard's **closed-form parameters**
        /// (analytic families — scale/shape/alpha/gamma/loghr). The analytic hazard
        /// is evaluated once per record with a *baseline* covariate snapshot (the
        /// [`HazardParamFn`] takes no time argument), so a time-varying covariate
        /// here would be silently frozen; [`crate::api::check_survival_tv_covariates`]
        /// rejects that up front (#741). Empty for [`HazardSpec::OdeAccumulated`],
        /// whose covariate dependencies flow through the ODE system and are guarded
        /// by the whole-subject time-varying-covariate check instead.
        hazard_covariates: Vec<String>,
    },
    /// Binary / Bernoulli endpoint — mixed-effects logistic regression (§3.5 of
    /// `plans/categorical-endpoint-phase4.md`). `state ∈ {0,1}` is observed via
    /// [`ObsRecord::DiscreteState`]; the data term is `−Σ[y·log p + (1−y)·log(1−p)]`
    /// with `p = link⁻¹(lp)`.
    Binary {
        link: LinkFn,
        /// Linear predictor `lp` (log-odds) as a function of (theta, eta, covariates).
        lp_fn: LinearPredictorFn,
        /// Covariate names referenced by the linear predictor (including any reached
        /// transitively through `[individual_parameters]`). Evaluated at a baseline
        /// snapshot (the [`LinearPredictorFn`] takes no time argument), so a covariate
        /// listed here that turns out to be time-varying in the data is rejected at fit
        /// setup by [`crate::api::check_survival_tv_covariates`] — the categorical
        /// analogue of [`Self::Tte`]'s `hazard_covariates` (#741).
        lp_covariates: Vec<String>,
    },
    /// Continuous-time Markov model (CTMM) endpoint — the discrete state
    /// `s ∈ {0..S−1}` is observed via [`ObsRecord::DiscreteState`] at irregular
    /// times, and the per-subject data term is `−Σ_m log P(Δt_m)[s_m, s_{m+1}]`
    /// with the transition-probability matrix `P(Δt) = expm(Q·Δt)` (§3.4/§8.7 of
    /// `plans/tte-survival-markov.md`). **Time-homogeneous** (Phase 5): the
    /// generator `Q(θ,η,cov)` is built once per subject from a baseline covariate
    /// snapshot; a drug-driven `Q(t)` is Phase 6. Gated behind `markov`
    /// (which implies `survival`).
    #[cfg(feature = "markov")]
    Ctmm {
        /// Number of states `S` (the generator is `S×S`).
        n_states: usize,
        /// Data DV code for each state, indexed by 0-based generator index:
        /// `state_codes[i]` is the integer DV value that denotes state `i`. The
        /// endpoint maps an observed [`ObsRecord::DiscreteState`] `state` (the raw
        /// DV code) back to its generator index through this table, so states may
        /// be coded with any non-negative integers (e.g. `1`/`2`), not only
        /// `0..S−1`. Length `== n_states`; codes are unique (checked at parse).
        state_codes: Vec<usize>,
        /// Builds the generator `Q(θ,η,cov,state)` — see [`GeneratorFn`].
        generator_fn: GeneratorFn,
        /// Resolved, dual-evaluable form of the same intensities, for the **analytic**
        /// gradient: it yields `∂Q/∂(θ,η)` exactly, which the endpoint chains through the
        /// Van Loan Fréchet derivative of `expm` instead of finite-differencing the whole
        /// matrix-exponential likelihood (#759).
        ///
        /// `None` — and the endpoint keeps its FD gradient — when the intensities are
        /// time-**inhomogeneous** (an intensity reads an ODE state, so `Q` is a functional
        /// of the PK trajectory and the likelihood is an occupancy ODE rather than an
        /// `expm`), or when the program cannot be represented exactly (see
        /// `build_ctmm_generator_program`).
        generator_program: Option<crate::parser::model_parser::CtmmGeneratorProgram>,
        /// Covariate names referenced by any transition intensity (including any
        /// reached transitively through `[individual_parameters]`). Evaluated at a
        /// baseline snapshot, so a covariate listed here that is time-varying in
        /// the data is rejected at fit setup by
        /// [`crate::api::check_survival_tv_covariates`] — the CTMM analogue of
        /// [`Self::Tte`]'s `hazard_covariates` and [`Self::Binary`]'s
        /// `lp_covariates` (#741).
        generator_covariates: Vec<String>,
        /// ODE **state** names referenced by any transition intensity, paired with
        /// their index in the model's ODE state vector: `(state_name, ode_index)`.
        /// Non-empty **iff** the endpoint is time-**inhomogeneous** (drug/PD-driven
        /// `Q(t)`, Phase 6 #817) — an intensity references a model state, so the
        /// generator must be re-evaluated as the state evolves over each observation
        /// gap (occupancy ODE, [`crate::markov::ctmm_inhomogeneous_transition`])
        /// rather than built once per subject. **Empty** ⇒ time-homogeneous (Phase 5,
        /// `expm(Q·Δt)`). [`generator_fn`](Self::Ctmm::generator_fn) reads its `state`
        /// argument through this map.
        generator_states: Vec<(String, usize)>,
    },
    // Ordinal, Poisson, NegBin, Dtmm deferred to Phase 4/4b
}

#[cfg(feature = "survival")]
impl std::fmt::Debug for EndpointLikelihood {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EndpointLikelihood::Gaussian(e) => write!(f, "Gaussian({e:?})"),
            EndpointLikelihood::Tte {
                hazard, recurrence, ..
            } => {
                write!(f, "Tte({hazard:?}, {recurrence:?})")
            }
            EndpointLikelihood::Binary { link, .. } => write!(f, "Binary({link:?})"),
            #[cfg(feature = "markov")]
            EndpointLikelihood::Ctmm {
                n_states,
                state_codes,
                ..
            } => write!(f, "Ctmm(n_states={n_states}, codes={state_codes:?})"),
        }
    }
}

/// Outcome of a single simulated observation — replaces the previous `dv_sim: f64` field.
///
/// `Continuous` preserves the existing Gaussian path unchanged.
/// `Event` carries TTE-specific outputs (gated behind `survival` feature).
/// `Category`/`Count` (Phase 4.0) are the shared categorical/count/Markov draw
/// shapes; they compile unconditionally but have no producer until Track C/D.
#[derive(Debug, Clone)]
pub enum SimOutcome {
    /// Gaussian continuous prediction + residual noise (the only variant before Phase 1).
    Continuous { value: f64 },
    /// TTE event: simulated event time and whether it occurred before the censoring horizon.
    #[cfg(feature = "survival")]
    Event { time: f64, observed: bool },
    /// Categorical / discrete-state draw (binary, ordinal, DTMM/CTMM state) — Phase 4+.
    Category { state: usize },
    /// Count draw (Poisson / negative-binomial) — Phase 4+.
    Count { count: u32 },
}

impl SimOutcome {
    /// Extract the continuous value for Gaussian outcomes; NAN for all others.
    ///
    /// Calling this on a TTE `Event` row is a logic error; a `debug_assert` fires
    /// in debug builds to catch misuse early.
    pub fn continuous_value(&self) -> f64 {
        match self {
            SimOutcome::Continuous { value } => *value,
            #[cfg(feature = "survival")]
            SimOutcome::Event { .. } => {
                debug_assert!(
                    false,
                    "continuous_value() called on a TTE Event row — filter by CMT type first"
                );
                f64::NAN
            }
            SimOutcome::Category { .. } | SimOutcome::Count { .. } => {
                debug_assert!(
                    false,
                    "continuous_value() called on a categorical/count row — filter by CMT type first"
                );
                f64::NAN
            }
        }
    }
}

/// Prediction for one (subject, time, CMT) point — the predict-side analogue of
/// [`SimOutcome`] (`plans/tte-survival-markov.md` §8.8.1).
///
/// A non-Gaussian endpoint's prediction is a **probability vector or a curve, never
/// one scalar**, which is why the legacy [`crate::api::PredictionResult`] (`pred: f64`)
/// cannot carry it. That struct is unchanged and still backs the Gaussian
/// [`crate::predict`]; this enum backs the per-endpoint predictors
/// ([`crate::api::predict_categorical`] today).
///
/// Only variants with a **live producer** are declared. The plan also sketches
/// `Survival` and `Rate` arms; TTE prediction already ships as the richer
/// [`crate::api::SurvivalPredictionResult`] (S/H/h + CIF + median/mean, more than a
/// three-field variant would hold), and `Rate` waits on the Poisson/count slice —
/// adding either before it can be produced would be an untestable dead arm.
#[derive(Debug, Clone, PartialEq)]
pub enum Prediction {
    /// Gaussian continuous prediction (the existing scalar path).
    Continuous { pred: f64 },
    /// Category probabilities `P(Y = k)`, indexed by the endpoint's own state
    /// order. Binary uses `[1 − p, p]` so index == the observed 0/1 DV code.
    /// CTMM occupancy `π(t)` reuses this variant (#820), indexed by generator
    /// state; callers map back to DV codes through the endpoint's `state_codes`.
    CatProbs { probs: Vec<f64> },
}

impl Prediction {
    /// `P(Y = k)` for state index `k`, or `None` when this is not a categorical
    /// prediction or `k` is out of range. Keeps callers from indexing `probs`
    /// directly and panicking on a shape they did not check.
    pub fn prob(&self, k: usize) -> Option<f64> {
        match self {
            Prediction::CatProbs { probs } => probs.get(k).copied(),
            Prediction::Continuous { .. } => None,
        }
    }
}

/// A prediction attached to a specific endpoint CMT — the multi-endpoint analogue
/// of [`crate::api::PredictionResult`], which has no CMT field because it predates
/// per-CMT endpoints.
#[derive(Debug, Clone, PartialEq)]
pub struct EndpointPredictionResult {
    pub id: String,
    /// CMT of the endpoint that produced this prediction.
    pub cmt: usize,
    /// Raw data TIME of the observation being predicted (matches sdtab / input).
    pub time: f64,
    /// Observed outcome at this record, when the dataset carries one — the 0/1 DV
    /// for a binary endpoint. Paired here so a caller can form a residual without
    /// re-joining against the input population.
    pub observed: Option<f64>,
    pub prediction: Prediction,
}

/// A compiled model ready for estimation
pub struct CompiledModel {
    pub name: String,
    pub pk_model: PkModel,
    pub error_model: ErrorModel,
    /// Residual error specification. For single-endpoint models this is
    /// `ErrorSpec::Single(error_model)`; for multi-endpoint models it carries
    /// the per-CMT dispatch and `error_model` holds a representative endpoint.
    pub error_spec: ErrorSpec,
    /// Fixed residual-error correlations declared by `block_sigma`.
    ///
    /// Empty for ordinary diagonal `$SIGMA` models. When non-empty, the
    /// residual-error helpers use each observation's sigma loadings to add both
    /// within-observation covariance terms (for `combined(...)` endpoints) and
    /// cross-observation covariance terms for paired same-time/same-occasion
    /// endpoint rows. This covers NONMEM `$SIGMA BLOCK` combined-error models
    /// and paired multi-endpoint records such as total/unbound assays.
    pub residual_correlations: Vec<ResidualCorrelation>,
    pub pk_param_fn: PkParamFn,
    pub n_theta: usize,
    /// Number of between-subject variability (BSV) ETAs.
    pub n_eta: usize,
    /// Number of inter-occasion variability (IOV) kappa parameters.
    /// Zero when no `kappa` declarations are present.
    pub n_kappa: usize,
    pub n_epsilon: usize,
    pub theta_names: Vec<String>,
    /// BSV ETA names only (length == n_eta).
    pub eta_names: Vec<String>,
    /// IOV kappa names (length == n_kappa). Empty when no IOV.
    pub kappa_names: Vec<String>,
    /// Names of the individual parameters declared at the top level of the
    /// `[individual_parameters]` block, in declaration order. Parallel to
    /// `pk_indices`; for analytical models the i-th name is the variable
    /// whose value lands in `PkParams.values[pk_indices[i]]`. For ODE
    /// models the i-th name is written sequentially into slot `i` by
    /// `pk_param_fn`. Used by the FFI to label per-subject EBE individual
    /// parameter values (e.g. `CL`, `V`, `Ka`).
    ///
    /// Bound: `pk_param_fn` writes at most `MAX_PK_PARAMS` slots (the size
    /// of the fixed `PkParams.values` array). For analytical models the
    /// parser already routes assignments through that fixed slot table, so
    /// excess names are not possible. For ODE models with more than
    /// `MAX_PK_PARAMS` top-level `[individual_parameters]` assignments,
    /// names beyond index `MAX_PK_PARAMS - 1` will appear in this list but
    /// `pk_param_fn` won't store their values — downstream consumers will
    /// read either zero or NaN for those slots. In practice no PK model
    /// approaches this limit.
    pub indiv_param_names: Vec<String>,
    /// Symbolic partial derivatives of every top-level `[individual_parameters]`
    /// assignment w.r.t. each θ and η axis, precomputed at parse time. Outer
    /// Vec is parallel to `indiv_param_names`; inner Vecs have length
    /// `n_theta` (user-declared θ) and `n_eta + n_kappa` (extended η) on the
    /// θ and η sides respectively.
    ///
    /// Reserved for a future analytical-η-gradient path. The original
    /// downstream consumers (Tier 4a milestones 3-5: augmented ODE RHS,
    /// Form C readout sensitivities, `gradient = sens` estimator wiring)
    /// were reverted in #145 after the `gradient = sens` path failed to
    /// deliver a wall-time win at low n_η. The partials themselves are
    /// still produced at parse time and are exercised by parser unit
    /// tests; they're kept on `CompiledModel` so the primitive is in
    /// place when a future symbolic-gradient consumer lands.
    ///
    /// Field itself is `pub` so external test fixtures and the
    /// `generate_data` binary can write `IndivParamPartials::empty()` into
    /// it. The inner Expression AST stays private — outside callers can
    /// only stuff in an empty placeholder, not read or mutate the
    /// parser-produced partials.
    #[allow(dead_code)] // no runtime consumer after #145; see field doc.
    pub indiv_param_partials: IndivParamPartials,
    pub default_params: ModelParameters,
    /// Per-eta flag (parallel to `eta_names` / omega diagonal): `true` when
    /// the user wrote `omega NAME ~ X (sd)` and the parser squared the value.
    /// Pure display metadata — the stored `default_params.omega` is always on
    /// the variance scale. Always `false` for etas declared inside a
    /// `block_omega`, which is variance-only.
    pub omega_init_as_sd: Vec<bool>,
    /// Per-sigma flag (parallel to `default_params.sigma.values`): `true` when
    /// the user wrote `sigma NAME ~ X (sd)`. Since #56 the .ferx default for
    /// sigma is variance — the parser `sqrt`s the variance-scale input into
    /// the internal SD representation. With `(sd)` the value is stored as-is.
    pub sigma_init_as_sd: Vec<bool>,
    /// Per-kappa flag (parallel to `kappa_names`): `true` when the user wrote
    /// `kappa NAME ~ X (sd)`. Empty when no kappa declarations are present.
    pub kappa_init_as_sd: Vec<bool>,
    /// Per-kappa sample-size weight (parallel to `kappa_names`, #1031): `Some`
    /// when the user wrote `kappa NAME ~ X weight = <expr>`, declaring
    /// `κ_ik ~ N(0, Ω_IOV / W_ik)` — the between-treatment-arm variability of a
    /// longitudinal MBMA, whose arm-level random effect scales with the number
    /// of subjects behind the arm. Empty when no kappa declarations are present.
    ///
    /// The scaling itself is applied by the parser, which desugars every
    /// reference to a weighted kappa in `[individual_parameters]` into
    /// `KAPPA / sqrt(W)`. That reparameterisation is exactly the hand-written
    /// form it replaces (`κ/√W` with `κ ~ N(0, Ω_IOV)` *is* `κ' ~ N(0, Ω_IOV/W)`),
    /// so every estimator, sensitivity, and diagnostic sees a model it already
    /// supports, and the estimated Ω_IOV stays the unweighted γ² a published
    /// analysis reports. This field is what the model carries for *reporting* —
    /// the source text plus an evaluator for the effective SD at a typical arm
    /// size (`γ/√W`), which is the number a reader actually needs.
    pub kappa_weights: Vec<Option<KappaWeight>>,
    /// Detected mu-referencing relationships: eta_name → (theta_name, log_transformed).
    /// Populated by the parser; empty map means no mu-referencing detected.
    pub mu_refs: HashMap<String, MuRef>,
    /// Same as `mu_refs` but for IOV kappa parameters (kappa_name → MuRef).
    pub kappa_mu_refs: HashMap<String, MuRef>,
    /// Computes covariate-adjusted typical values per subject for AD.
    /// Returns one value per `[individual_parameters]` assignment (in
    /// declaration order), evaluated with eta = 0. Covariates and theta are
    /// folded in; only eta is differentiated. The AD inner loop then
    /// computes `pk[pk_indices[i]] = tv[i] * exp(dot(sel_flat[i,:], eta))`,
    /// so `tv.len() == pk_indices.len() == eta_map.len() == sel_flat.len() / n_eta`,
    /// and the eta application is driven by `sel_flat` rather than being
    /// positional. When `Some`, enables AD gradient computation in the
    /// inner loop; when `None` (e.g. ODE models), falls back to FD.
    pub tv_fn: Option<TvFn>,
    /// Maps each `[individual_parameters]` assignment (by declaration order)
    /// to its PK parameter slot. E.g. for a model with CL, V, KA:
    /// `[PK_IDX_CL, PK_IDX_V, PK_IDX_KA] = [0, 1, 4]`. Parallel to the
    /// output of `tv_fn` and to `eta_map`; used by AD to route each tv
    /// value to the correct PK slot. Note: the index here is the
    /// assignment/tv index, *not* the eta index — see `eta_map` for the
    /// latter (they diverge when some params are eta-free).
    pub pk_indices: Vec<usize>,
    /// Per-tv eta index: `eta_map[i]` is the eta index referenced by the
    /// i-th [individual_parameters] assignment, or -1 if the assignment
    /// references no eta (e.g. `Q = TVQ`). Parallel to `pk_indices` and the
    /// output of `tv_fn`; used by the AD path to correctly combine eta
    /// with each tv slot. Before this field existed the AD loop assumed
    /// `pk_indices.len() == n_eta` with 1:1 positional correspondence,
    /// which silently misaligned eta and produced NaN gradients for models
    /// with eta-free PK parameters like 2-cpt where `Q` is fixed.
    pub eta_map: Vec<i32>,
    /// Precomputed `pk_indices` as `Vec<f64>` — the form the AD functions
    /// actually want. Cached here so each BFGS gradient call doesn't
    /// reallocate and recast a tiny vector; on a 110k-find_ebe fit that
    /// saves several million allocations.
    pub pk_idx_f64: Vec<f64>,
    /// Precomputed one-hot eta selector (row-major, n_tv × n_eta) derived
    /// from `eta_map`. Same motivation as `pk_idx_f64`: built once, reused
    /// for every AD gradient evaluation.
    pub sel_flat: Vec<f64>,
    /// ODE specification. When `Some`, predictions use ODE integration instead of
    /// analytical PK equations. The `pk_param_fn` output is flattened and passed
    /// to the ODE RHS function as the parameter vector.
    pub ode_spec: Option<crate::ode::OdeSpec>,
    /// Compartment-indexed modeled-dose attributes (`D{cmt}` for `RATE=-2`) for
    /// **analytical** PK models (#324, #394). ODE models carry their own map on
    /// [`crate::ode::OdeSpec::dose_attr_map`] and leave this `Default` (empty) —
    /// the analytical dispatch paths read this field, the ODE paths read the
    /// `OdeSpec` one. Empty for the common analytical model with no `RATE=-2`
    /// dosing; populated by the parser when a `D{cmt}` individual parameter is
    /// declared. Used by [`crate::pk::compute_predictions`] callers to resolve
    /// modeled-duration doses to a concrete `rate`/`duration` before the
    /// closed-form math (mirrors the ODE `resolve_subject_doses` step).
    pub dose_attr_map: DoseAttrMap,
    /// Index of the first diffusion theta in the theta vector, and the parallel
    /// mapping from diffusion-theta index to ODE state index.
    /// `None` when no `[diffusion]` block is present.
    /// Used by `ekf_p_obs` to read current diffusion variances from `theta`
    /// without requiring mutation of `ode_spec` during estimation.
    pub diffusion_theta_start: Option<usize>,
    /// For each diffusion theta (offset from `diffusion_theta_start`),
    /// the index of the ODE state it applies to. Parallel to the diffusion
    /// theta slice of `theta`. Empty when `diffusion_theta_start` is `None`.
    pub diffusion_state_indices: Vec<usize>,
    /// Mirror of [`FitOptions::bloq_method`] so likelihood/AD paths can read
    /// it without threading the options struct through every call site.
    /// Set by [`fit_from_files`](crate::fit_from_files) automatically;
    /// callers invoking [`fit`](crate::fit) with a hand-built `CompiledModel`
    /// must set this field to match `options.bloq_method` themselves.
    pub bloq_method: BloqMethod,
    /// Covariate names referenced by any expression in the model (preserved
    /// in the case the modeller wrote). Validated against the data's covariate
    /// columns before a fit so that a missing/misspelt covariate fails loudly
    /// instead of silently evaluating to zero.
    pub referenced_covariates: Vec<String>,
    /// Mirror of [`FitOptions::gradient_method`] so the inner loop can
    /// dispatch at runtime without threading the options struct through
    /// every call site. Set by [`fit_from_files`](crate::fit_from_files)
    /// automatically; callers invoking [`fit`](crate::fit) with a
    /// hand-built `CompiledModel` must set this field to match
    /// `options.gradient_method` themselves. A mismatch is not detected —
    /// `find_ebe` reads this field, not `options`.
    pub gradient_method: GradientMethod,
    /// Warnings generated at parse time (e.g. mu-referencing disabled for
    /// conditional parameters).  Prepended to `FitResult.warnings` by `fit()`.
    pub parse_warnings: Vec<String>,
    /// True when an individual parameter is assigned inside an `if`-branch that
    /// references an ETA (e.g. `if (WT>70) { CL = TVCL*exp(ETA_CL) } else {...}`).
    /// Set by the parser. The analytical AD kernels can't represent the branch
    /// structure, so `inner_optimizer::analytical_ad_unsupported` routes such
    /// models to FD. (Structured replacement for matching the "conditional
    /// parameter" `parse_warnings` string.)
    pub has_conditional_eta_params: bool,
    /// Per-ETA transformation metadata derived from the `[individual_parameters]`
    /// expressions at parse time. Length ≤ n_eta (only ETAs whose expression was
    /// classified are present). Forwarded into `FitResult`.
    pub eta_param_info: Vec<EtaParamInfo>,
    /// Per-theta transformation: `theta_transform[i]` describes whether theta i
    /// is used on the natural (Identity), log, or logit scale. Length == n_theta.
    pub theta_transform: Vec<ThetaTransform>,
    /// Parsed `[covariate_nn NAME]` blocks (one entry per block in the model
    /// file). Empty when the `nn` feature is off or no block is present.
    ///
    /// Consumed by `build_pk_param_fn` and `tv_fn`: each NN's forward output is
    /// pre-computed once per call and looked up by `Expression::NnOutput` via
    /// the `NAME.OUTPUT` dot-access syntax. Weights live in `theta` starting at
    /// `weights_offset` for `mapper.n_weights()` slots, so they participate in
    /// the optimizer vector like any other theta.
    #[cfg(feature = "nn")]
    pub covariate_nns: Vec<crate::nn::CovariateNn>,
    /// How the structural model's raw output is mapped to the observed `DV`.
    /// Default `ScalingSpec::None` preserves the historical behaviour where
    /// the prediction is returned unchanged. Forms A/B from the `[scaling]`
    /// block populate this field; Form C lives on `ode_spec.output_fn`.
    pub scaling: ScalingSpec,
    /// Log-transform-both-sides (LTBS) active. When `true`, the effective
    /// prediction is `log(f)` and the residual error is additive on the log
    /// scale (matching NONMEM's `Y = LOG(F) + EPS`). Set by the parser when
    /// the `[error_model]` uses `log(DV) ~ additive(...)` or
    /// `DV ~ log_additive(...)`. The prediction sinks log-wrap their output
    /// when this is set, so IPRED/PRED/IWRES/CWRES and simulated DV are all
    /// reported on the log scale.
    pub log_transform: bool,
    /// Only meaningful when `log_transform` is `true`. `true` means the data's
    /// `DV` column is *already* on the log scale (case 1, `DV ~ log_additive`),
    /// so `fit()` must NOT log-transform it again; `false` means `DV` is on the
    /// natural scale (case 2, `log(DV) ~ additive`) and `fit()` log-transforms
    /// the observations once at load. Ignored when `log_transform` is `false`.
    pub dv_pre_logged: bool,
    /// Derived expression specifications from [derived] block.
    /// Empty when no [derived] block is present. Evaluated post-fit.
    pub derived_exprs: Vec<DerivedExprSpec>,
    /// Column names from [output] block. Validated at fit time.
    pub output_columns: Vec<String>,
    /// Per-CMT non-Gaussian endpoint specifications.
    /// Empty for models with only Gaussian observations.
    /// Keyed by the CMT value declared in `[event_model]` / future blocks.
    #[cfg(feature = "survival")]
    pub endpoints: HashMap<usize, EndpointLikelihood>,
    /// FREM configuration. When `Some`, the model uses FREMTYPE-based
    /// observation dispatch: covariate pseudo-observations use individual
    /// parameter values as predictions and a near-zero additive sigma.
    pub frem_config: Option<FremConfig>,
    /// IIV on residual error (NONMEM `Y = IPRED + EPS*EXP(ETA)`). When `Some(k)`,
    /// eta index `k` is a random effect that scales the residual standard
    /// deviation per subject: the residual variance for every observation is
    /// multiplied by `exp(2*eta[k])`. The eta is declared as an ordinary
    /// `omega` in `[parameters]` and wired here via `iiv_on_ruv = NAME` in
    /// `[error_model]`; it is NOT referenced by any individual parameter, so it
    /// carries no `EtaParamInfo` entry and the PK closure ignores it. See #409.
    pub residual_error_eta: Option<usize>,
    /// Non-zero initial compartment amounts for **analytical** PK models, from
    /// the `[initial_conditions]` block (issue #521). Empty for the common case
    /// (zero initial state) and for ODE models (which seed state via
    /// `ode_spec.init_fn` instead). Each entry layers a closed-form
    /// `F`-bypassed bolus impulse onto the dose-driven prediction; see
    /// [`AnalyticalInit`] and `pk::add_analytical_init`.
    pub analytical_init: Vec<AnalyticalInit>,
    /// Full Form C output readout (`[scaling] y = <expr>`) for **analytical** PK
    /// models (issue #650). `None` for the common case (built-in concentration
    /// readout), for Forms A/B (`obs_scale`, stored in [`Self::scaling`]), and
    /// for ODE models (whose Form C readout lives on `ode_spec.readout`). When
    /// `Some`, the readout replaces the built-in concentration prediction; see
    /// [`AnalyticReadout`].
    pub analytic_readout: Option<AnalyticReadout>,
    /// Custom residual-error magnitude (#484). `None` for the common case where
    /// every sigma is a bare parameter (the legacy variance path). `Some` when
    /// the `[error_model]` makes a sigma's magnitude an expression of TIME /
    /// covariates / theta; the FOCE/FOCEI likelihood then scales each
    /// observation's sigma loadings by [`RuvMagnitude::eval_obs`].
    pub ruv_magnitude: Option<RuvMagnitude>,
    /// A synthesized ODE representation of an analytical absorption model that carries one,
    /// used as a runtime fallback when the closed form cannot serve a particular subject. The
    /// analytic transit (`one_cpt_transit` / `two_cpt_transit`, #386) and inverse-Gaussian
    /// (`one_cpt_ig` / `two_cpt_ig`, #790) closed forms assume constant parameters over each
    /// absorption window, so a mid-profile `TIME` switch or time-varying covariates route to
    /// this exact ODE (`transit()` / `igd()`) equivalent instead (a full boxed sub-model, built
    /// eagerly at parse time so it reuses the whole ODE prediction/sensitivity path unchanged).
    /// It is also the target of the parameter-dependent flip-flop reroute (`ke` outside the
    /// tilting convergence domain).
    ///
    /// `None` means the model has **no ODE fallback**, for any of three reasons: it is not a
    /// closed-form absorption model at all; it is one the desugar declines by name (a
    /// `lagtime`/`f` mapping that shadows a reserved slot, a user `[odes]` / `[scaling]` /
    /// `[initial_conditions]` block, a dose-attribute parameter in a disposition role); or the
    /// twin *was* reconstructed and its own parse rejected it (#1008), in which case the model
    /// carries a `W_ABSORPTION_TWIN_DECLINED` [`Self::parse_warnings`] entry naming the reason —
    /// read it with [`absorption_twin_decline_reason`]. The cause set is open-ended by
    /// construction (the twin is an ODE model, so every ODE-scoped parse check applies to it and
    /// not to the analytical primary), so do not treat `None` as "outside the desugar's scope".
    /// See [`CompiledModel::effective_for`], [`AbsorptionOdeEquivalent`], and the parser's
    /// `absorption_ode_equivalent_source` (#486, #790, #1008).
    pub absorption_ode_equivalent: Option<AbsorptionOdeEquivalent>,
    /// `$MIXTURE`-style discrete latent subpopulations (#977). `Some` when the
    /// model file carries a `[mixture]` block: the subject marginal becomes a
    /// covariate-weighted mixture over `n_classes` class-conditional
    /// likelihoods, and the reserved `MIXNUM` index (1..=K) selects
    /// class-specific typical values in `[individual_parameters]`. `None` for an
    /// ordinary single-population model. Phase 1 (#977) parses and validates the
    /// spec; `fit()` errors until the objective is wired.
    pub mixture: Option<MixtureSpec>,
}

/// The ODE representation of an analytical absorption model that carries one (a plain
/// `one_cpt_transit` / `two_cpt_transit`, or `one_cpt_ig` / `two_cpt_ig`), built **eagerly at
/// parse time** from the equivalent's reconstructed `.ferx` source.
///
/// It used to be built lazily (a `OnceLock` compiled the source on first use, so a fit whose
/// subjects never hit the fallback paid no extra parse), with a build failure turned into a
/// `.expect()` panic on the grounds that the source "is reconstructed from already-validated
/// blocks". That reasoning was structurally wrong: the twin is an **ODE** model while the
/// primary is **analytical**, so every ODE-scoped parse check runs on the twin and not on the
/// primary, and a model the primary accepts can produce a twin the parser rejects — which
/// surfaced as an internal panic mid-fit, the first time a TV-covariate / `TIME` / IOV / SS /
/// infusion subject rerouted (#1003 hit it twice; #1008 found a third live door, an individual
/// parameter named `CENTRAL`/`PERIPH` colliding with the twin's state names). Building at
/// parse time makes the failure a parse-time **decline** instead: `parse_full_model` attaches
/// `None`, records a parse warning, and the model stays closed-form, exactly as if the desugar
/// had declined it. The cost is one extra parse of the twin source per transit/IG model
/// (~12 ms measured at `ci-test` opt level, ≈ the primary's own parse; paid once, where the
/// lazy path paid the same parse on first reroute). See [`CompiledModel::effective_for`]
/// (#486, #790, #1008).
pub struct AbsorptionOdeEquivalent {
    built: Box<CompiledModel>,
}

impl AbsorptionOdeEquivalent {
    /// Parse the reconstructed twin source into the boxed sub-model. `Err` carries the twin
    /// parser's own message; the caller (`parse_full_model`) declines the twin and records it
    /// as a parse warning rather than failing the primary parse (#1008). The source contains
    /// no `pk one_cpt_transit`/…/`pk two_cpt_ig` line, so this parse cannot recurse into the
    /// desugar that produced it.
    pub(crate) fn build(source: &str) -> Result<Self, String> {
        let built = crate::parser::model_parser::parse_model_string(source)?;
        Ok(Self {
            built: Box::new(built),
        })
    }

    /// The ODE equivalent sub-model.
    pub(crate) fn built(&self) -> &CompiledModel {
        &self.built
    }

    /// Stamp the call-time ODE solver tolerances onto the twin so rerouted subjects integrate
    /// at the requested accuracy. Called by [`CompiledModel::sync_ode_solver_opts`]. Without
    /// this, a closed-form transit/IG primary is analytic — its own `ode_spec` stamp is a
    /// no-op — so every rerouted (TV-cov / `TIME` / IOV) subject silently integrated at the
    /// tolerance baked into the twin's re-emitted `[fit_options]` (#814). The twin is itself
    /// an ODE model, so this stamps its `ode_spec.solver_opts`; it carries no nested twin, so
    /// the twin-branch in `sync_ode_solver_opts` is a no-op.
    pub(crate) fn sync_solver_opts(&mut self, opts: &FitOptions) {
        self.built.sync_ode_solver_opts(opts);
    }
}

/// The marker that separates the fixed `W_ABSORPTION_TWIN_DECLINED` text from the twin parser's
/// own reason (#1008). Written by [`absorption_twin_declined_warning`] and read back by
/// [`absorption_twin_decline_reason`], so the round-trip is one constant rather than two
/// independent substring guesses. The fixed text contains no other occurrence, so splitting on
/// the first one recovers the reason exactly — including a reason that itself says "Reason:".
const TWIN_DECLINE_REASON_MARKER: &str = " Reason: ";

/// The parse warning recorded when an absorption model's ODE twin was reconstructed but its own
/// parse rejected it (#1008).
///
/// The reason goes **last**, behind [`TWIN_DECLINE_REASON_MARKER`], for two reasons: the fixed
/// consequence text (which features the model just lost) reads first, and the variable tail is
/// recoverable verbatim by [`absorption_twin_decline_reason`] so the up-front rejection messages
/// in `api::validation` can quote it. The twin parser's message is trimmed and given a sentence
/// terminator — most parser errors carry none, and without this the reason ran into the
/// following sentence.
pub(crate) fn absorption_twin_declined_warning(reason: &str) -> String {
    let reason = reason.trim();
    let terminator = if reason.ends_with(['.', '!', '?']) {
        ""
    } else {
        "."
    };
    format!(
        "This absorption model's ODE equivalent could not be built, so the model stays \
         closed-form. Subjects needing the ODE fallback (time-varying covariates, a \
         `TIME`-dependent parameter, IOV, steady-state or infusion doses, or the flip-flop \
         regime) will be rejected with an explicit error instead of silently rerouting \
         (W_ABSORPTION_TWIN_DECLINED, #1008).{TWIN_DECLINE_REASON_MARKER}{reason}{terminator}"
    )
}

/// The `#1008` twin-decline reason `model` recorded at parse time, or `None` when its twin was
/// never declined that way (it built, or the desugar declined it by name before reconstructing
/// a source).
///
/// The reason lives in [`CompiledModel::parse_warnings`] rather than in a field of its own:
/// `CompiledModel` is not `#[non_exhaustive]`, so a new public field would be a breaking change
/// for struct-literal construction, and `parse_warnings` is already the model's parse-diagnostics
/// channel. That makes the `W_ABSORPTION_TWIN_DECLINED` token load-bearing — it is what
/// `classify_warning` and `api::validation::parse_warning_to_code` key on too — so it is written
/// and read in one place here.
pub(crate) fn absorption_twin_decline_reason(model: &CompiledModel) -> Option<&str> {
    model
        .parse_warnings
        .iter()
        .find(|w| w.contains("W_ABSORPTION_TWIN_DECLINED"))
        .and_then(|w| w.split_once(TWIN_DECLINE_REASON_MARKER))
        .map(|(_, reason)| reason)
}

/// FREM (Full Random Effects Model) configuration.
///
/// Maps FREMTYPE observation-type values to (theta_index, eta_index) pairs
/// so the likelihood can compute covariate pseudo-observation predictions
/// as `theta[theta_idx] + eta[eta_idx]` and use a near-zero additive sigma.
#[derive(Debug, Clone)]
pub struct FremConfig {
    /// Maps FREMTYPE value (100, 200, ...) → (theta_index, eta_index).
    /// For FREMTYPE observations, the prediction is
    /// `theta[theta_idx] + eta[eta_idx]`.
    pub fremtype_to_indices: HashMap<u16, (usize, usize)>,
    /// Index into `sigma_values` for the covariate error sigma (EPSCOV).
    pub covariate_sigma_index: usize,
}

/// Inner-loop (per-subject EBE) gradient method.
///
/// The inner optimizer is BFGS; what differs across variants is how the
/// gradient of the individual NLL w.r.t. ETA is computed.
///
/// - `Auto` (default): use the exact analytic `Dual2` sensitivities whenever
///   the model is in scope for them (analytical PK path: `tv_fn` populated,
///   no LTBS / expression-scale inner / SDE), else fall back to `Fd`. The
///   resolved per-subject route is reported in the startup banner.
/// - `Fd`: central finite differences on the forward NLL. Performs `2·n_eta`
///   forward evaluations per gradient, so cost scales linearly with the
///   number of random effects. Always available, including for ODE models.
/// - `Ad`: retired. The Enzyme automatic-differentiation path was removed in
///   favour of the analytic `Dual2` provider; requesting it errors with
///   `E_AD_RETIRED`. Retained as a parse-then-error alias so old model files
///   surface a clear migration message. Use `Auto` or `Fd` instead.
///
/// ## Numerical equivalence
///
/// The analytic `Dual2` gradient is exact up to floating-point roundoff; FD
/// introduces `O(1e-9)` noise per component. For well-conditioned problems
/// both converge to the same OFV within line-search tolerance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GradientMethod {
    #[default]
    Auto,
    Ad,
    Fd,
}

impl CompiledModel {
    /// Vector and data-bound θ level blocks declared by `[parameters]`.
    ///
    /// This metadata lives behind the existing opaque `indiv_param_partials`
    /// field so the feature remains compatible with external `CompiledModel`
    /// struct literals.
    pub fn theta_blocks(&self) -> &ThetaBlocks {
        &self.indiv_param_partials.theta_blocks
    }

    /// Returns true when this model uses ODE integration; false for analytical PK.
    pub fn is_ode_based(&self) -> bool {
        self.ode_spec.is_some()
    }

    /// Returns true for a **compartment-free** (`$PRED`-equivalent) model — one
    /// whose `[structural_model]` declares its prediction as an equation with no
    /// compartments underneath (issue #811).
    ///
    /// Such a model has no ODE spec *and* no closed form: its prediction is the
    /// readout alone. [`Self::pk_model`] holds a placeholder that must never be
    /// dispatched on, so every site that reads it — the closed-form predictors,
    /// the dose/absorption validators, the analytic sensitivity gates, the
    /// starting-estimate heuristics — branches on this predicate first.
    ///
    /// **Derived, not stored**, so the two cannot drift: an
    /// [`AnalyticReadout`] with an EMPTY `state_names` *is* the marker, and one
    /// can only be built by the parser (the type is `#[non_exhaustive]` and
    /// every compartment-bearing readout carries at least `central`).
    pub fn is_algebraic(&self) -> bool {
        self.ode_spec.is_none()
            && self
                .analytic_readout
                .as_ref()
                .is_some_and(|ar| ar.state_names.is_empty())
    }

    /// The model that should actually serve `subject`'s predictions / sensitivities.
    ///
    /// For a closed-form absorption model (transit `one_cpt_transit` / `two_cpt_transit`, or
    /// inverse-Gaussian `one_cpt_ig` / `two_cpt_ig`) whose closed form cannot cope with this
    /// subject — a `TIME`-dependent structural parameter or time-varying covariates make the
    /// disposition switch mid-absorption, which the per-dose absorption convolution assumes
    /// constant, or IOV (`n_kappa > 0`, #719) needs cross-occasion dose carryover (#104) that
    /// the superposition cannot express, or a **steady-state** (`SS=1`) dose (#719) needs the
    /// periodic pulse train equilibrated through the absorption kernel — which the ODE twin
    /// does (`ode::predictions::equilibrate_ss_state`) but the closed form has no periodic-sum
    /// form for — return its exact ODE (`transit()` / `igd()`) equivalent
    /// (`absorption_ode_equivalent`, built at parse time); otherwise `self`. Constant-parameter,
    /// non-IOV, non-SS subjects keep the fast, exact closed form. The equivalent shares this
    /// model's θ/η layout, so callers pass the same parameter vector (#486, #790). A no-op
    /// (`self`) for every model without an ODE twin.
    pub fn effective_for<'a>(&'a self, subject: &Subject) -> &'a CompiledModel {
        if let Some(eq) = &self.absorption_ode_equivalent {
            if crate::parser::model_parser::compiled_model_uses_time_builtin(self)
                || subject.has_tv_covariates()
                || self.n_kappa > 0
                // Steady-state doses reroute to the ODE twin (#719): the closed-form
                // superposition has no periodic-sum SS form, but the twin equilibrates the
                // dose through the absorption kernel (a pulse-train trough + periodic forward
                // `R_in`). The SS + lagtime combination the twin can't yet serve is rejected
                // upfront by `check_absorption_closed_form_support`, so it never reaches here.
                || subject.has_periodic_ss_dose()
                // Infusion doses reroute to the ODE twin too (#719 gap 2): the closed form
                // absorbs an instantaneous bolus, but the twin delivers the dose as a zero-order
                // source feeding the kernel (`R_in_inf`). The SS-infusion / zero-order-window
                // combinations the twin can't yet serve are rejected upfront.
                || subject.doses.iter().any(|d| d.is_infusion())
            {
                return eq.built();
            }
        }
        self
    }

    /// The compartment-indexed dose-attribute map (`D{cmt}` for `RATE=-2`, …) for
    /// **this model's engine**: the `OdeSpec`'s map for ODE models, the analytical
    /// `dose_attr_map` field otherwise. Single source of truth for "which map
    /// applies", so a caller cannot accidentally read the empty analytical default
    /// on an ODE model (or vice versa) and silently resolve nothing (#383/#394).
    pub(crate) fn active_dose_attr_map(&self) -> &DoseAttrMap {
        match &self.ode_spec {
            Some(ode) => &ode.dose_attr_map,
            None => &self.dose_attr_map,
        }
    }

    /// Mutable [`Self::active_dose_attr_map`], for the parser to annotate the map
    /// after both engines' maps have been built — currently only
    /// [`DoseAttrMap::mark_prediction_path_read`] (#993), which is discovered when
    /// `[scaling]` is parsed, after the ODE spec already exists.
    pub(crate) fn active_dose_attr_map_mut(&mut self) -> &mut DoseAttrMap {
        match &mut self.ode_spec {
            Some(ode) => &mut ode.dose_attr_map,
            None => &mut self.dose_attr_map,
        }
    }

    /// Copy the configured ODE solver tolerances from `opts` onto this model's
    /// [`OdeSpec`] (no-op for the `OdeSpec` of an analytical model). Call this once
    /// after the model file's `[fit_options]` and any call-time `settings` overrides
    /// have been merged into `opts`, so the integrator uses the requested accuracy.
    /// The parser calls it at parse time, so `.ferx` `[fit_options]` and any
    /// entry that integrates the parsed spec as-is (`predict`, `fit_from_files`)
    /// already use the configured accuracy.
    ///
    /// Also carries the same tolerances into the absorption ODE twin
    /// ([`AbsorptionOdeEquivalent`]) when the model has one, so a closed-form
    /// transit/IG model — analytic itself, but whose TV-cov / `TIME` / IOV subjects
    /// integrate on the twin — honors a call-time `ode_reltol`/`ode_abstol`/
    /// `ode_max_steps` on the rerouted path too (#814). This runs even though the
    /// primary's own `ode_spec` is `None`, so on such a model the call is no longer a
    /// full no-op.
    ///
    /// Note: [`fit`](crate::fit) takes `&CompiledModel` and does **not** call
    /// this. The integrator reads [`OdeSpec::solver_opts`], never
    /// `FitOptions::ode_reltol` directly, so a caller that merges call-time
    /// `settings` into its own `FitOptions` (as the R wrapper's `ferx_fit`
    /// does) must re-apply this on an owned model *before* `fit` for those
    /// overrides to reach the solver. Idempotent.
    pub fn sync_ode_solver_opts(&mut self, opts: &FitOptions) {
        if let Some(ode) = self.ode_spec.as_mut() {
            ode.solver_opts.reltol = opts.ode_reltol;
            ode.solver_opts.abstol = opts.ode_abstol;
            ode.solver_opts.max_steps = opts.ode_max_steps;
            ode.solver_opts.method = opts.ode_method;
            ode.solver_opts.stiff_abort_after = opts.ode_stiff_abort_after;
            ode.solver_opts.auto_switch = opts.ode_auto_switch;
        }
        // Also carry the call-time tolerances into the absorption ODE twin (#814). A
        // closed-form transit/IG primary is analytic — the block above is a no-op — but its
        // TV-cov / `TIME` / IOV subjects integrate on the twin (`effective_for`), which must
        // honor the requested accuracy rather than the parse-default baked into its source.
        // Runs whether or not the primary carries an `ode_spec`.
        if let Some(eq) = self.absorption_ode_equivalent.as_mut() {
            eq.sync_solver_opts(opts);
        }
    }

    /// Returns true when the model has a `[diffusion]` block (SDE / EKF path).
    pub fn is_sde(&self) -> bool {
        self.diffusion_theta_start.is_some()
    }

    /// True for the **closed-form LTBS** path — the one that runs the analytic
    /// `g = ln(f)` *inner* EBE gradient whose ~1e-9 provider-vs-predictor gap makes the
    /// covariance step tolerance-sensitive (PR #665). ODE-LTBS shares `solve_ode_g` with
    /// the objective (no gap), so it does not need the tightened fit/covariance
    /// tolerances. Used by [`FitOptions::effective_inner_tol`] /
    /// [`FitOptions::effective_cov_inner_tol`].
    ///
    /// **IOV is included as of #486.** The `n_kappa == 0` carve-out that used to sit here
    /// was justified solely by LTBS × IOV routing its inner loop to FD — which
    /// `analytic_inner_common_bail` no longer does, since `run_obs_iov_eta` now applies
    /// the same `ln` jet over the stacked axes. A κ-carrying closed-form LTBS model
    /// therefore runs the *same* analytic ln-jet inner gradient, carries the *same* gap,
    /// and needs the same tolerances; leaving it out would have produced quietly inflated
    /// standard errors with unchanged point estimates. This predicate must move with
    /// `analytic_inner_common_bail`'s `log_transform` scope, not independently of it.
    pub fn uses_closed_form_ltbs_inner(&self) -> bool {
        self.log_transform && self.ode_spec.is_none()
    }

    /// Returns true when the model has a time-to-event (`[event_model]`)
    /// endpoint. The analytical single-snapshot AD kernel computes the
    /// PK-observation NLL, not the hazard/survival likelihood, so its
    /// eta-gradient through the hazard (especially the shape parameters) is
    /// wrong - `tte_weibull` / `tte_gompertz` diverged ~2-5 OFV from FD under
    /// AD, so TTE models take the FD inner gradient (via the live gates
    /// `analytic_inner_grad_supported` / `subject_has_survival_records`, not the
    /// vestigial `analytical_ad_unsupported`).
    #[cfg(feature = "survival")]
    pub fn has_tte(&self) -> bool {
        self.endpoints
            .values()
            .any(|e| matches!(e, EndpointLikelihood::Tte { .. }))
    }

    /// CMTs routed to a `Tte` endpoint, sorted ascending. Keeps the
    /// "is this a TTE endpoint" predicate in one place (shared with
    /// [`has_tte`](Self::has_tte)); callers that need the cause list — e.g.
    /// building one censoring template row per cause in `run_model_simulate` —
    /// use this instead of inlining the `matches!` filter.
    #[cfg(feature = "survival")]
    pub fn tte_cmts(&self) -> Vec<usize> {
        let mut c: Vec<usize> = self
            .endpoints
            .iter()
            .filter_map(|(cmt, ep)| matches!(ep, EndpointLikelihood::Tte { .. }).then_some(*cmt))
            .collect();
        c.sort_unstable();
        c
    }

    /// True if any endpoint is a **repeated** TTE (RTTE) endpoint. Drives the
    /// Laplace-bias warning in `fit()`: RTTE ω² is severely underestimated under
    /// FOCE/FOCEI (Karlsson et al. 2009), so an RTTE fit whose final stage is
    /// Laplace-based warns and recommends `method = saem`/`imp`. It does **not**
    /// change the method — the default estimator stays `FoceI`.
    #[cfg(feature = "survival")]
    pub fn has_rtte(&self) -> bool {
        self.endpoints.values().any(|e| {
            matches!(
                e,
                EndpointLikelihood::Tte {
                    recurrence: TteRecurrence::Repeated { .. },
                    ..
                }
            )
        })
    }

    /// True if any endpoint is a [`Binary`](EndpointLikelihood::Binary) (logistic)
    /// endpoint — Phase 4 categorical, Track C (#760).
    #[cfg(feature = "survival")]
    pub fn has_binary(&self) -> bool {
        self.endpoints
            .values()
            .any(|e| matches!(e, EndpointLikelihood::Binary { .. }))
    }

    /// CMTs routed to a `Binary` endpoint, sorted ascending. The datareader routes
    /// these rows to [`ObsRecord::DiscreteState`] via `ObsRouting.discrete`.
    #[cfg(feature = "survival")]
    pub fn binary_cmts(&self) -> Vec<usize> {
        let mut c: Vec<usize> = self
            .endpoints
            .iter()
            .filter_map(|(cmt, ep)| matches!(ep, EndpointLikelihood::Binary { .. }).then_some(*cmt))
            .collect();
        c.sort_unstable();
        c
    }

    /// True if any endpoint is a [`Ctmm`](EndpointLikelihood::Ctmm)
    /// (continuous-time Markov) endpoint — Phase 5, Track D (#759).
    #[cfg(feature = "markov")]
    pub fn has_ctmm(&self) -> bool {
        self.endpoints
            .values()
            .any(|e| matches!(e, EndpointLikelihood::Ctmm { .. }))
    }

    /// CMTs routed to a `Ctmm` endpoint, sorted ascending. The datareader routes
    /// these rows to [`ObsRecord::DiscreteState`] via `ObsRouting.discrete`, the
    /// same discrete-state plumbing the `Binary` endpoint uses.
    #[cfg(feature = "markov")]
    pub fn ctmm_cmts(&self) -> Vec<usize> {
        let mut c: Vec<usize> = self
            .endpoints
            .iter()
            .filter_map(|(cmt, ep)| matches!(ep, EndpointLikelihood::Ctmm { .. }).then_some(*cmt))
            .collect();
        c.sort_unstable();
        c
    }

    /// True if the model carries any **discrete** (non-Gaussian, non-TTE) endpoint — the
    /// binary / Bernoulli family today; ordinal / Poisson / negative-binomial extend this
    /// match. This is the family-agnostic gate for the discrete data term and its FD-Hessian
    /// contribution, so a new discrete family is enabled here and in `discrete_subject_nll`
    /// without editing a likelihood dispatch site. Equals [`has_binary`](Self::has_binary)
    /// today.
    #[cfg(feature = "survival")]
    pub fn has_discrete(&self) -> bool {
        self.endpoints
            .values()
            .any(|e| matches!(e, EndpointLikelihood::Binary { .. }))
    }

    /// True if the model carries **any non-Gaussian endpoint** (TTE/RTTE, the
    /// Phase-4 categorical/count families, or a Phase-5 CTMM). This is the
    /// predicate the estimation machinery gates on when choosing the **FD-Hessian
    /// Laplace** inner/outer path and disabling the analytic outer gradient —
    /// those choices are correct for every non-Gaussian data term, not TTE
    /// specifically. Equals [`has_tte`](Self::has_tte) exactly when no
    /// discrete/Markov endpoint is present, so existing TTE behaviour is
    /// unchanged.
    #[cfg(feature = "survival")]
    pub fn has_non_gaussian(&self) -> bool {
        #[cfg(feature = "markov")]
        let ctmm = self.has_ctmm();
        #[cfg(not(feature = "markov"))]
        let ctmm = false;
        self.has_tte() || self.has_discrete() || ctmm
    }

    /// Always false without the `survival` feature - TTE endpoints can't be
    /// parsed, so no model can carry one.
    #[cfg(not(feature = "survival"))]
    pub fn has_tte(&self) -> bool {
        false
    }

    /// Always false without the `survival` feature - no non-Gaussian endpoint
    /// (TTE or categorical) can be parsed.
    #[cfg(not(feature = "survival"))]
    pub fn has_non_gaussian(&self) -> bool {
        false
    }

    /// Always false without the `survival` feature - `[binary_model]` can't be parsed.
    #[cfg(not(feature = "survival"))]
    pub fn has_binary(&self) -> bool {
        false
    }

    /// Always false without the `survival` feature - no discrete (categorical/count)
    /// endpoint can be parsed.
    #[cfg(not(feature = "survival"))]
    pub fn has_discrete(&self) -> bool {
        false
    }

    /// Returns true when `[individual_parameters]` declares `LAGTIME` (or its
    /// `ALAG` alias). Used by the prediction dispatcher and inner optimizer
    /// to choose between cached-schedule / AD fast paths and the lagtime-
    /// aware slow paths.
    ///
    /// Checks both routes by which lagtime can be wired in:
    ///   1. Analytical PK: `pk_indices` contains `PK_IDX_LAGTIME` when the
    ///      `[structural_model]` line includes `lagtime=` / `alag=`.
    ///   2. ODE: the LAGTIME/ALAG slot is populated by name in
    ///      `build_pk_param_fn`'s ODE branch (sequential pk_indices do not
    ///      reflect this), so we fall back to scanning `indiv_param_names`.
    pub fn has_lagtime(&self) -> bool {
        if self.pk_indices.contains(&PK_IDX_LAGTIME) {
            return true;
        }
        self.indiv_param_names.iter().any(|n| {
            let u = n.to_uppercase();
            // Bare `lagtime`/`alag` apply on any engine. A compartment-indexed
            // `ALAGn`/`LAGTIMEn` (issue #369) only routes lag on the ODE engine
            // — the analytical path has a single fixed dose route, where such a
            // name lands in an unused spare slot — so gate it on `ode_spec`.
            u == "LAGTIME"
                || u == "ALAG"
                || (self.ode_spec.is_some()
                    && matches!(DoseAttr::from_indexed_name(n), Some((DoseAttr::Lag, _))))
        })
    }

    /// True when any built-in absorption input-rate forcing carries a per-route
    /// lag (`fn(..., lag=L)`, #857) — the forcing switches on at its own
    /// `t_dose + lag_cmt + lag_route`, distinct from the compartment lag
    /// ([`Self::has_lagtime`]). Used to route a route-lag subject onto the
    /// event-driven analytic walk (`integrate_tvcov_g`), where each route's onset
    /// discontinuity is injected as its own rate-on saltation (#859) — mirroring
    /// how `has_lagtime` routes an estimated compartment lag there. A model with
    /// no `ode_spec` (analytical PK) has no input-rate forcings, so this is false.
    pub(crate) fn has_route_absorption_lag(&self) -> bool {
        self.ode_spec.as_ref().is_some_and(|o| o.has_route_lag())
    }

    /// Compartment-scoped variant of [`Self::has_lagtime`]: true only when a
    /// lag actually resolves for a dose into 1-based `cmt` — a bare
    /// `LAGTIME`/`ALAG` (which, per [`DoseAttrMap::lagtime`]'s fallback, applies
    /// to every compartment lacking its own indexed override) or an
    /// `ALAG{cmt}`/`LAGTIME{cmt}` indexed specifically for this compartment.
    /// Unlike `has_lagtime`, a lag declared on a *different* compartment
    /// (`ALAG2` while `cmt` is 1) does not count — used to scope the SS+lag
    /// rejection (#719 gap 1) to the actual SS-dosed compartment.
    pub fn has_lagtime_on_cmt(&self, cmt: usize) -> bool {
        if self.pk_indices.contains(&PK_IDX_LAGTIME) {
            return true;
        }
        self.indiv_param_names.iter().any(|n| {
            let u = n.to_uppercase();
            u == "LAGTIME"
                || u == "ALAG"
                || (self.ode_spec.is_some()
                    && matches!(
                        DoseAttr::from_indexed_name(n),
                        Some((DoseAttr::Lag, indexed_cmt)) if indexed_cmt == cmt
                    ))
        })
    }

    /// True when `subject` has a steady-state dose into a built-in absorption
    /// input-rate compartment that the analytic dual SS fixed point does **not**
    /// serve: SS into a `zero_order` spanning window (the pointwise `R_in` fixed
    /// point doesn't build it), or SS combined with an absorption lag on the dosed
    /// compartment — either a compartment `lagtime`/`ALAG` ([`Self::has_lagtime_on_cmt`])
    /// or a per-route `lag=` on the forcing ([`crate::pk::absorption::InputRateForcing`]'s
    /// `lag_slot`, #859). All are hard-rejected upstream (`E_ABSORPTION_SS_ZERO_ORDER`
    /// / `E_ABSORPTION_SS_LAG`, `api::check_absorption_dosing` — whose `has_ss_lag`
    /// covers a per-route lag via `route_lag_cmts`); the ODE support gates
    /// (`ode_tvcov_supported`, `ode_iov_subject_supported`) and the IOV FD-reason
    /// attribution (`iov_fd_reason`) re-check it as belt-and-suspenders so the
    /// analytic-vs-FD routing stays self-consistent if either upstream check is
    /// relaxed. This must therefore mirror the upstream predicate exactly — the
    /// per-route `lag_slot` clause was added with #859 (which made a `first_order`
    /// route lag analytic, so without it a relaxed upstream reject would route an
    /// SS + route-lag model onto the analytic SS seed, which does not carry the
    /// per-route onset). Single source of truth for those three sites (#835 review)
    /// — keeping the scope in one place so it cannot drift between them (cf. the
    /// #814 attribution-drift class of bug). Scoped to the SS-dosed compartment:
    /// an SS dose into a plain compartment on a model that separately declares
    /// absorption for a different route is unaffected (its `R_in` is inert for
    /// that dose).
    pub(crate) fn ss_absorption_out_of_scope(&self, subject: &Subject) -> bool {
        let Some(ode) = self.ode_spec.as_ref() else {
            return false;
        };
        subject.doses.iter().any(|d| {
            // Match the forcing 0-based and probe the lagtime 1-based so a `CMT=0` dose (the
            // default dose compartment == compartment 1) is scoped like `CMT=1` (#899, #913 review)
            // — otherwise a `CMT=0` SS dose into a zero-order / lagged absorption compartment
            // slips this out-of-scope guard and is silently mis-served.
            d.ss && d.ii > 0.0
                && ode.input_rate.iter().any(|f| {
                    f.cmt == d.cmt_idx()
                        && (f.kind == crate::pk::absorption::InputRateKind::ZeroOrder
                            || f.lag_slot.is_some()
                            || self.has_lagtime_on_cmt(d.cmt_1based()))
                })
        })
    }

    /// True when the model wires in a bioavailability `F`/`Fn` parameter (on
    /// either engine). Mirrors [`Self::has_lagtime`]: the analytical route puts
    /// [`PK_IDX_F`] in `pk_indices` (from `f=` on the `[structural_model]`
    /// line), while both engines may instead name it `F` (any case) in
    /// `[individual_parameters]`; a compartment-indexed `Fn` routes on the ODE
    /// engine only. Used with [`Subject::has_rate_defined_infusion`] to skip the
    /// event-driven [`crate::pk::event_driven::EventSchedule`] cache when `F`
    /// could reshape an infusion window across the inner search (#419).
    pub fn has_bioavailability(&self) -> bool {
        if self.pk_indices.contains(&PK_IDX_F) {
            return true;
        }
        self.indiv_param_names.iter().any(|n| {
            let u = n.to_uppercase();
            u == "F"
                || (self.ode_spec.is_some()
                    && matches!(DoseAttr::from_indexed_name(n), Some((DoseAttr::F, _))))
        })
    }

    /// Residual variance for one observation, dispatching on its compartment.
    /// Thin convenience wrapper over [`ErrorSpec::variance_at`] for call sites
    /// that already hold the `&CompiledModel`.
    pub fn residual_variance_at(&self, cmt: usize, f_pred: f64, sigma: &[f64]) -> f64 {
        self.error_spec.variance_at_with_correlations(
            cmt,
            f_pred,
            sigma,
            &self.residual_correlations,
        )
    }

    /// Residual variance for one observation with a per-observation custom
    /// magnitude (#484). `mult` is the per-sigma multiplier vector for this
    /// observation (from [`RuvMagnitude::eval_obs`]); a `None` falls back to the
    /// plain [`residual_variance_at`](Self::residual_variance_at).
    pub fn residual_variance_at_scaled(
        &self,
        cmt: usize,
        f_pred: f64,
        sigma: &[f64],
        mult: Option<&[f64]>,
    ) -> f64 {
        match mult {
            Some(m) => self.error_spec.variance_at_scaled(
                cmt,
                f_pred,
                sigma,
                &self.residual_correlations,
                m,
            ),
            None => self.residual_variance_at(cmt, f_pred, sigma),
        }
    }

    /// Per-observation residual-magnitude multipliers for `subject` at `theta`
    /// (#484), as an `[obs][sigma-slot]` matrix, or `None` when no custom
    /// magnitude is active. Evaluated once per outer iteration (it depends on
    /// θ, observation covariates, and TIME — never on η), so the FOCE/FOCEI
    /// data term, Laplace curvature term, and inner EBE objective can all share
    /// one matrix and stay mutually consistent. The TIME fed to each row is the
    /// raw data-file time ([`Subject::readout_time`], matching what NONMEM's
    /// `$ERROR` sees), falling back to `obs_times` for in-memory subjects.
    pub fn ruv_obs_mult(&self, subject: &Subject, theta: &[f64]) -> Option<Vec<Vec<f64>>> {
        let rm = self.ruv_magnitude.as_ref()?;
        if !rm.is_active() {
            return None;
        }
        let n = subject.observations.len();
        let mut out = Vec::with_capacity(n);
        for j in 0..n {
            out.push(rm.eval_obs(theta, subject.obs_cov(j), subject.readout_time(j)));
        }
        Some(out)
    }

    /// Per-observation `∂(residual-magnitude multiplier)/∂θ` for `subject` at
    /// `theta` (#576/#486), as an `[obs][sigma-slot][theta]` tensor, or `None`
    /// when no custom magnitude is active *or* any active slot's `Dual1` program
    /// declines (θ-axis count beyond [`crate::parser::model_parser::MAX_RUV_MAG_AXES`]
    /// — the analytic outer gradient's own gate,
    /// [`crate::sens::provider::analytic_outer_gradient_available`], bounds
    /// `model.n_theta` against the same constant, so this should only return
    /// `None` here for a model that gate already declined). Feeds the outer θ/σ
    /// gradient's direct-θ channel (`sens_outer_gradient::prepare_stacked`) — the
    /// magnitude enters `R` through θ directly, not only through the prediction.
    pub fn ruv_obs_mult_theta_grad(
        &self,
        subject: &Subject,
        theta: &[f64],
    ) -> Option<Vec<Vec<Vec<f64>>>> {
        let rm = self.ruv_magnitude.as_ref()?;
        if !rm.is_active() {
            return None;
        }
        let n = subject.observations.len();
        let mut out = Vec::with_capacity(n);
        for j in 0..n {
            out.push(rm.eval_obs_theta_grad(theta, subject.obs_cov(j), subject.readout_time(j))?);
        }
        Some(out)
    }

    /// True when a residual error model is defined for `cmt` *and* its sigmas
    /// resolve in `sigma`, so a DV (assay-noised) observation can be drawn there.
    /// False for a [`ErrorSpec::PerCmt`] spec that omits `cmt` (or whose endpoint
    /// `sigma_idx` falls outside `sigma`), or a `Single` model whose `sigma` is
    /// shorter than its error model needs (`< n_sigma` — e.g. a `Combined` model
    /// carrying one σ, down to no `[error_model]` at all). `simulate_adaptive`
    /// consults this before drawing a DV monitor so an un-modeled compartment is a
    /// typed error rather than a fabricated σ (#391 S1.5 edge a) — and so
    /// [`residual_variance_at`](Self::residual_variance_at), which indexes
    /// `sigma[0..n_sigma]` for a `Single` model (and the endpoint's `sigma_idx` for
    /// a `PerCmt` one), is never reached with a `sigma` too short to cover it
    /// (which would otherwise **panic** for `Single` or return **NaN** for
    /// `PerCmt`). The two branches are symmetric: each requires `sigma` to cover
    /// every index the variance computation will read.
    pub fn has_residual_error_for_cmt(&self, cmt: usize, sigma: &[f64]) -> bool {
        match &self.error_spec {
            ErrorSpec::Single(em) => sigma.len() >= em.n_sigma(),
            ErrorSpec::PerCmt(map) => map.get(&cmt).is_some_and(|ep| {
                // The endpoint must carry enough sigma indices for its model *and*
                // every one must resolve in `sigma`; `variance_at` reads
                // `sigma_idx[0..n_sigma]`, so either shortfall yields a NaN variance.
                ep.sigma_idx.len() >= ep.error_model.n_sigma()
                    && ep.sigma_idx.iter().all(|&i| i < sigma.len())
            }),
            // `Selected` dispatches on a covariate-resolved branch key, not `cmt`,
            // so "residual error for this CMT" isn't meaningful per-CMT. Every
            // observation resolves to some endpoint (the parser requires a final
            // `else`), so residual error is defined iff *every* endpoint is
            // well-formed against `sigma`. (Selected is not combined with the
            // adaptive-dosing monitor path that consults this guard.)
            ErrorSpec::Selected { endpoints, .. } => {
                !endpoints.is_empty()
                    && endpoints.values().all(|ep| {
                        ep.sigma_idx.len() >= ep.error_model.n_sigma()
                            && ep.sigma_idx.iter().all(|&i| i < sigma.len())
                    })
            }
        }
    }

    /// Whether a custom residual-error magnitude (#484) is active. The
    /// per-observation magnitude expression makes the residual variance depend
    /// on θ directly (not only through the prediction `f`). The analytic gradient
    /// now carries this via a direct-θ channel (`mag_variance_dtheta` in
    /// `sens_outer_gradient::prepare_stacked` / `theta_block`; the inner loop via
    /// `residual_inner_obs`), so **both** loops are analytic on FOCE and FOCEI
    /// (#644/#659) — including combined with `iiv_on_ruv` on the closed-form
    /// (#673/#677) and ODE (#486) paths, since the outer assembly is
    /// provider-agnostic. Per-subject FD fallbacks remain only for magnitude
    /// combined with an M3-censored row, a correlated `block_sigma`, or more than
    /// `MAX_RUV_MAG_AXES` θ (see #486's remaining-FD register). The FD forward NLL
    /// is itself magnitude-aware, so those fallbacks stay exact.
    #[inline]
    pub fn has_custom_ruv_magnitude(&self) -> bool {
        self.ruv_magnitude.as_ref().is_some_and(|m| m.is_active())
    }

    /// Exact number of free coordinates the outer optimizer searches over.
    /// This uses the same packed FIX and structural-zero masks as estimation,
    /// including diagonal/separate BSV and IOV blocks and mixture overrides.
    pub fn free_packed_dim(&self) -> usize {
        let p = &self.default_params;
        let fixed = crate::estimation::parameterization::packed_fixed_mask(p);
        let structural = crate::estimation::parameterization::omega_structural_zero_mask(p);
        debug_assert_eq!(fixed.len(), structural.len());
        fixed
            .iter()
            .zip(structural.iter())
            .filter(|(is_fixed, is_structural)| !**is_fixed && !**is_structural)
            .count()
    }

    /// Whether any kappa carries a `weight = <expr>` modifier (#1031).
    #[inline]
    pub fn has_weighted_kappa(&self) -> bool {
        self.kappa_weights.iter().any(|w| w.is_some())
    }

    /// Whether an *active* custom residual magnitude depends on `θ`
    /// (see [`RuvMagnitude::theta_dependent`]).
    ///
    /// `false` for the θ-free case — every `weight = <covariate expr>` model
    /// (#1029) and the TIME/covariate-only #484 forms — where an analytic θ
    /// gradient that chains through the prediction alone is still exact.
    #[inline]
    pub fn has_theta_dependent_ruv_magnitude(&self) -> bool {
        self.ruv_magnitude
            .as_ref()
            .is_some_and(|m| m.is_active() && m.theta_dependent)
    }

    /// Multiplicative factor applied to the residual *variance* for a subject
    /// whose random-effect vector is `eta`, from the IIV-on-RUV term
    /// (`Y = IPRED + EPS*EXP(ETA)`). Returns `exp(2*eta[k])` when
    /// `residual_error_eta == Some(k)` and `k` is in range, else `1.0`.
    ///
    /// Because every residual-error model writes the variance as
    /// `(scale·σ)²` terms summed, multiplying the whole variance by
    /// `exp(2*eta_k)` is exactly equivalent to scaling the residual SD by
    /// `exp(eta_k)` — i.e. `EPS·EXP(ETA)` — for additive, proportional, and
    /// combined alike.
    #[inline]
    pub fn residual_var_scale(&self, eta: &[f64]) -> f64 {
        match self.residual_error_eta {
            Some(k) => match eta.get(k) {
                Some(&e) => (2.0 * e).exp(),
                None => 1.0,
            },
            None => 1.0,
        }
    }

    /// Residual *variance* for resampling observation row `j` of `subject`
    /// during simulation or NPDE, including the FREM and IIV-on-RUV splits that
    /// a bare [`residual_variance_at`](Self::residual_variance_at) call misses:
    ///
    /// * FREM covariate pseudo-observations (`FREMTYPE > 0`) follow the additive
    ///   model `Y = θ+η+ε` with the covariate sigma (EPSCOV), independent of the
    ///   PK error model, and are **not** scaled by the PK residual-error IIV —
    ///   mirroring the likelihood (`stats/likelihood.rs`) and the IWRES path
    ///   (`api.rs`). Their `f_pred` is the `θ+η` override, so feeding it into the
    ///   PK error model would scale the noise by the covariate magnitude (e.g.
    ///   proportional error), which is wrong.
    /// * All other rows use the PK error model scaled by `ruv_scale` (pass
    ///   [`residual_var_scale`](Self::residual_var_scale)).
    ///
    /// `mult` is the per-sigma custom-magnitude multiplier row (#484) for this
    /// observation, from [`ruv_obs_mult`](Self::ruv_obs_mult); `None` (or a row
    /// of ones) reproduces the legacy unscaled variance. FREM covariate rows are
    /// never magnitude-scaled — the multiplier acts on the PK residual only,
    /// mirroring the likelihood and IWRES paths.
    #[inline]
    pub fn sim_residual_variance(
        &self,
        subject: &Subject,
        j: usize,
        f_pred: f64,
        sigma: &[f64],
        ruv_scale: f64,
        mult: Option<&[f64]>,
    ) -> f64 {
        if let Some(ref fc) = self.frem_config {
            if subject.fremtype.get(j).copied().unwrap_or(0) > 0 {
                let s = sigma[fc.covariate_sigma_index];
                return (s * s).max(1e-12);
            }
        }
        self.residual_variance_at_scaled(self.error_spec.obs_key(subject, j), f_pred, sigma, mult)
            * ruv_scale
    }

    /// Canonical compartment names for analytical models, used in `[derived]` expressions.
    /// For ODE models use `ode_spec.state_names` instead.
    /// Returns a `'static` slice so it can be used in `DerivedContext` without lifetime issues.
    pub fn analytical_compartment_names(&self) -> &'static [String] {
        debug_assert!(
            self.ode_spec.is_none(),
            "analytical_compartment_names called on an ODE model — use ode_spec.state_names instead"
        );
        // A compartment-free model (#811) has no compartments to name. Its
        // `pk_model` is a placeholder, so the match below would advertise
        // `["central"]` — and `[derived]` would then resolve a bare `central` to a
        // compartment reference that reads NaN forever, instead of reporting it as
        // the undefined name it is.
        if self.is_algebraic() {
            return &[];
        }
        use std::sync::OnceLock;
        // Exhaustive `match` on `pk_model` so adding an 11th `PkModel` variant is a
        // COMPILE error here (a missing arm), not a runtime index-out-of-bounds on a
        // fixed-size array. Each arm lazily materialises its own owned `Vec<String>`
        // from the compartment-name literals in `pk::topology` — the single source of
        // truth — that the public `&'static [String]` API returns.
        macro_rules! cached_names {
            ($lock:ident) => {{
                static $lock: OnceLock<Vec<String>> = OnceLock::new();
                $lock
                    .get_or_init(|| {
                        self.pk_model
                            .topology()
                            .compartment_names
                            .iter()
                            .map(|s| s.to_string())
                            .collect()
                    })
                    .as_slice()
            }};
        }
        match self.pk_model {
            PkModel::OneCptIv => cached_names!(ONE_CPT_IV),
            PkModel::OneCptOral => cached_names!(ONE_CPT_ORAL),
            PkModel::OneCptTransit => cached_names!(ONE_CPT_TRANSIT),
            PkModel::OneCptIg => cached_names!(ONE_CPT_IG),
            PkModel::TwoCptIv => cached_names!(TWO_CPT_IV),
            PkModel::TwoCptOral => cached_names!(TWO_CPT_ORAL),
            PkModel::TwoCptTransit => cached_names!(TWO_CPT_TRANSIT),
            PkModel::TwoCptIg => cached_names!(TWO_CPT_IG),
            PkModel::ThreeCptIv => cached_names!(THREE_CPT_IV),
            PkModel::ThreeCptOral => cached_names!(THREE_CPT_ORAL),
        }
    }
}

impl std::fmt::Debug for CompiledModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledModel")
            .field("name", &self.name)
            .field("pk_model", &self.pk_model)
            .field("error_model", &self.error_model)
            .field("error_spec", &self.error_spec)
            .field("n_theta", &self.n_theta)
            .field("n_eta", &self.n_eta)
            .field("n_kappa", &self.n_kappa)
            .finish()
    }
}

/// One discrete-endpoint (binary) observation's post-fit diagnostics — the
/// non-Gaussian analogue of a row of [`SubjectResult`]'s parallel `ipred`/`iwres`
/// vectors (`plans/tte-survival-markov.md` §8.8.5).
///
/// Kept as its own row type rather than appended to those vectors because a discrete
/// record is not on the Gaussian observation grid: it has no residual variance, no
/// CWRES, and its `ipred` is a probability rather than a concentration. Every column
/// the Gaussian rows carry but a discrete row cannot define is emitted as an empty
/// sdtab cell.
#[cfg(feature = "survival")]
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq)]
pub struct DiscreteObsDiagnostic {
    /// CMT of the endpoint that produced this record.
    pub cmt: usize,
    /// Raw data TIME (joins back to the input CSV).
    pub time: f64,
    /// Observed outcome (the 0/1 DV for a binary endpoint).
    pub dv: f64,
    /// Population prediction `P(Y = 1)` at η = 0.
    pub pred: f64,
    /// Individual prediction `P(Y = 1)` at the subject's EBE η.
    pub ipred: f64,
    /// Standardized (Pearson) residual `(y − p)/√(p(1−p))` at the EBE. NaN at a
    /// degenerate `p ∈ {0,1}`, where it is undefined.
    pub iwres: f64,
}

/// Per-subject estimation results
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct SubjectResult {
    pub id: String,
    #[serde(with = "crate::serde_nalgebra::dvector")]
    pub eta: DVector<f64>,
    pub ipred: Vec<f64>,
    pub pred: Vec<f64>,
    pub iwres: Vec<f64>,
    pub cwres: Vec<f64>,
    /// Normalized prediction distribution errors (simulation-based, decorrelated
    /// within subject). Empty unless `[fit_options] npde_nsim > 0`. Populated
    /// post-fit by [`crate::stats::npde`]; emitted as the `NPDE` sdtab column.
    pub npde: Vec<f64>,
    /// Normalized prediction discrepancies (simulation-based, no decorrelation).
    /// Empty unless `[fit_options] npde_nsim > 0`. Emitted as the `NPD` column.
    pub npd: Vec<f64>,
    pub ofv_contribution: f64,
    pub cens: Vec<i8>,
    /// Number of observations for this subject (MDV=0 rows).
    pub n_obs: usize,
    /// Posterior class-membership probabilities `PMIX_ik` for a `[mixture]` model
    /// (length = number of classes `K`); `None` for non-mixture fits. Formed from
    /// the converged fit as `PMIX_ik ∝ p_ik · exp(−nll_ik)` (normalised over `k`)
    /// and emitted as the `PMIX_1..PMIX_K` sdtab columns. (#977)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pmix: Option<Vec<f64>>,
    /// Most-probable class `MIXEST_i` (1-based, NONMEM `MIXEST` convention) for a
    /// `[mixture]` model; `None` for non-mixture fits. `argmax_k PMIX_ik`, emitted
    /// as the `MIXEST` sdtab column. (#977)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mixest: Option<usize>,
    /// Extra sdtab columns from [derived] and [output] blocks, computed
    /// post-fit. Each entry is (column_name, per-observation values). Subject-
    /// level aggregates (max, AUC, tmax) are repeated across all observation rows.
    pub extra_columns: Vec<(String, Vec<f64>)>,
    /// Per-observation TAD computed with individual lagtime. Populated by
    /// `compute_extra_output_columns` whenever the model has a lagtime or [derived]/[output]
    /// blocks exist. Empty if those conditions are not met; output.rs falls back to
    /// a lagtime=0 approximation in that case (correct for the common case).
    pub per_obs_tad: Vec<f64>,
    /// Full compartment state vector at each observation time. For ODE models this
    /// is the raw solver state `u[i]` in whatever units the ODE defines (typically
    /// amounts for standard PK ODEs). For SDE models (`[diffusion]` block), these are
    /// the deterministic ODE states, not EKF-filtered states. For analytical models
    /// this is the natural per-compartment value (amounts for depot, concentrations for
    /// central/peripheral). Indexed `[obs_j][cmt_i]`.
    /// Empty for IOV subjects and analytical TV-covariate subjects (those cases yield
    /// NaN in `[derived]`; see W_DERIVED_CMT_IOV_UNSUPPORTED / W_DERIVED_CMT_TV_ANALYTICAL).
    /// Scaling (`apply_scaling`, Form A/C) is never applied here — only `ipred` is scaled.
    pub compartment_states: Vec<Vec<f64>>,
    /// Post-fit diagnostics for this subject's **discrete** (binary) endpoint records,
    /// one entry per record (§8.8.5). Empty for models with no discrete endpoint, which
    /// keeps every existing `FitResult` serialization byte-identical.
    #[cfg(feature = "survival")]
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub discrete_rows: Vec<DiscreteObsDiagnostic>,
}

/// Per-subject conditional distribution of the random effects, produced by the
/// opt-in SAEM conditional-distribution pass (`saem_conddist = true`, #257).
///
/// All per-subject vectors are indexed in the same order as the fit's subjects.
/// This is the SAEM analogue of saemix `conddist.saemix` / Monolix's
/// "Conditional Distribution" task — the distribution `p(η_i | y_i; θ̂)` rather
/// than just its mode (the EBE on `SubjectResult.eta`).
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct CondDist {
    /// Conditional mean of η per subject: `cond_mean[i]` has length `n_eta`.
    pub cond_mean: Vec<Vec<f64>>,
    /// Conditional SD of η per subject (sample SD over the retained draws).
    pub cond_sd: Vec<Vec<f64>>,
    /// Retained draws per subject: `samples[i]` is `nsamp × n_eta`. Empty for
    /// every subject unless `saem_conddist_keep_samples` was set.
    pub samples: Vec<Vec<Vec<f64>>>,
    /// Distribution-based η-shrinkage per eta:
    /// `1 - SD_over_subjects(cond_mean[·][j]) / sqrt(Ω_jj)`. `NaN` when there
    /// are fewer than two subjects.
    pub shrinkage: Vec<f64>,
    /// Number of retained draws per subject (after burn-in).
    pub nsamp: usize,
    /// Burn-in sweeps discarded before accumulation.
    pub burnin: usize,
}

// ── Derived expression types ──────────────────────────────────────────────────

/// Context threaded into every [derived] expression evaluation.
pub struct DerivedContext<'a> {
    pub theta: &'a [f64],
    pub eta: &'a [f64],
    pub indiv_params: &'a HashMap<String, f64>,
    pub covariates: &'a HashMap<String, f64>,
    pub ipred: f64,
    pub pred: f64,
    pub dv: f64,
    pub time: f64,
    pub tafd: f64,
    pub tad: f64,
    pub prev_derived: &'a HashMap<String, f64>,
    /// Raw compartment state at this observation time. For ODE models this is
    /// the solver state `u[i]`; for SDE models these are deterministic ODE states
    /// (not EKF-filtered). For analytical models it follows the convention in
    /// the `compartment_states` docs on `SubjectResult`. Empty slice for IOV subjects,
    /// analytical TV-covariate subjects, and grid-integral points when
    /// `uses_compartments` is false.
    pub compartments: &'a [f64],
    /// Names parallel to `compartments` — ODE state names or analytical names.
    pub compartment_names: &'a [String],
}

pub type DerivedEvalFn = Box<dyn Fn(&DerivedContext<'_>) -> f64 + Send + Sync>;
pub type DerivedFilterFn = Box<dyn Fn(&DerivedContext<'_>) -> bool + Send + Sync>;

pub struct DerivedExprSpec {
    pub name: String,
    pub kind: DerivedKind,
    /// True if any sub-expression in `kind` references `compartments[i]` or a
    /// named ODE state variable.  Used to gate `W_DERIVED_CMT_*` warnings so
    /// they fire only when compartment states are actually requested.
    pub uses_compartments: bool,
}

pub enum DerivedKind {
    /// Evaluated independently at each observation row.
    PerRow { eval: DerivedEvalFn },
    /// Reduction over observation rows (max/min/tmax), optionally filtered.
    /// The scalar is repeated for all rows of the subject.
    Aggregate {
        func: AggFunction,
        value: DerivedEvalFn,
        filter: Option<DerivedFilterFn>,
    },
    /// Numeric integration over a time window.
    Integral {
        integrand: DerivedEvalFn,
        /// When `Some`, only time points where this evaluates to true contribute.
        condition: Option<DerivedFilterFn>,
        /// True when integrand references DV → use observation times only.
        data_based: bool,
        /// True when the integrand references `compartments[i]` or a named ODE
        /// state variable. When true and the integral uses a fine grid, the ODE
        /// solver (or analytical formula) is re-run at grid points for accuracy.
        uses_compartments: bool,
        window: IntegralWindow,
        step: IntegralStep,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggFunction {
    Max,
    Min,
    Tmax,
}

#[derive(Debug, Clone)]
pub enum IntegralWindow {
    Explicit {
        from: f64,
        to: f64,
    },
    /// One integral per period-aligned window; each observation gets its window's value.
    Periodic {
        period: f64,
        anchor: f64,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum IntegralStep {
    /// Use observation times only (DV integrals; fallback when model unavailable).
    ObsTimes,
    /// Fine internal grid with this step size (hours).
    Fixed(f64),
    /// Auto: (to − from) / 500.
    Auto,
}

/// How per-occasion kappa random effects are treated in the IS marginal likelihood.
///
/// `Marginalized` — kappa is jointly sampled with eta using a block-diagonal
/// posterior Hessian. The IS -2LL integrates over both η and κ uncertainty,
/// making it directly comparable to NONMEM's `$EST METHOD=IMP LAPLACIAN=1`.
/// `FixedAtMode` — kappa is held at its EBE (κ̂) and only eta is sampled. This is a
/// *partial* marginal likelihood that ignores κ uncertainty. (Legacy; no longer
/// used for IOV models.)
/// `NotApplicable` — model has no kappa declarations.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum KappaTreatment {
    NotApplicable,
    FixedAtMode,
    Marginalized,
}

/// Result of the importance-sampling marginal log-likelihood step.
///
/// Produced by the `Imp` stage in a method chain (`methods = [..., imp]`).
/// Surfaced on `FitResult.importance_sampling`.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct ImportanceSamplingResult {
    /// `−2 · Σᵢ log p(yᵢ | θ)` estimated by importance sampling. Lower-bias
    /// alternative to the FOCE/Laplace OFV when subject posteriors are
    /// non-Gaussian (sparse data, strong nonlinearity). For IOV models,
    /// this uses joint (eta, kappa) sampling and is directly comparable
    /// to the FOCE OFV.
    pub minus2_log_likelihood: f64,
    /// Monte-Carlo standard error on `minus2_log_likelihood`. Scales with
    /// `1/sqrt(n_samples)`; halve by quadrupling `imp_samples`.
    pub mc_standard_error: f64,
    /// `(subject_id, ESS/K)` for every subject whose normalized effective sample
    /// size fraction fell below `FitOptions::imp_low_ess_threshold`. Empty list
    /// means every subject's proposal matched its posterior well.
    pub low_ess_subjects: Vec<(String, f64)>,
    /// Number of importance samples drawn per subject (`FitOptions::imp_samples`).
    pub n_samples: usize,
    /// Student-t proposal degrees of freedom (`FitOptions::imp_proposal_df`).
    pub proposal_df: f64,
    /// Minimum across-subject normalized ESS fraction (ESS / K). 1.0 = ideal,
    /// near 0 = degenerate proposal for at least one subject.
    pub ess_min: f64,
    /// Median across-subject normalized ESS fraction.
    pub ess_median: f64,
    /// Treatment of per-occasion kappa random effects.  See [`KappaTreatment`].
    pub kappa_treatment: KappaTreatment,
}

/// One row of the IMPMAP per-iteration parameter trace.
///
/// Analogous to one line in NONMEM's `.ext` file for `METHOD=IMPMAP`.
/// Positive `iteration` values are EM iterations; special negative values
/// mark the final (averaged) estimate and standard errors.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct ImpmapTraceRow {
    /// EM iteration number (1-based). Special values:
    /// `-1_000_000_000` = final averaged estimate,
    /// `-1_000_000_001` = standard errors (when covariance step ran).
    pub iteration: i64,
    pub theta: Vec<f64>,
    /// Lower triangle of the omega matrix, row-major: `(0,0), (1,0), (1,1), …`
    pub omega_lower_tri: Vec<f64>,
    pub sigma: Vec<f64>,
    /// Objective function value (−2·log-likelihood from importance sampling).
    pub ofv: f64,
}

/// Per-iteration parameter trace from IMPMAP, analogous to NONMEM `.ext`.
///
/// Surfaced on `FitResult.impmap_trace` when the final estimating stage is
/// IMPMAP. Column names follow NONMEM convention (`THETA1`, `OMEGA(1,1)`, …).
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Default)]
pub struct ImpmapTrace {
    pub rows: Vec<ImpmapTraceRow>,
    pub theta_names: Vec<String>,
    /// e.g. `"OMEGA(1,1)"`, `"OMEGA(2,1)"`, `"OMEGA(2,2)"`, …
    pub omega_names: Vec<String>,
    pub sigma_names: Vec<String>,
}

/// Posterior summary for a single scalar parameter, computed across all
/// post-warmup, post-thinning draws from every chain.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct PosteriorSummary {
    /// Parameter name (e.g. `TVCL`, `OMEGA(1,1)`, `SIGMA(1)`).
    pub name: String,
    pub mean: f64,
    pub sd: f64,
    /// 2.5% posterior quantile (lower 95% credible bound).
    pub q025: f64,
    pub median: f64,
    /// 97.5% posterior quantile (upper 95% credible bound).
    pub q975: f64,
    /// Split-R̂ convergence diagnostic. Values near 1.0 indicate the chains
    /// have mixed; `> 1.01` flags non-convergence.
    pub rhat: f64,
    /// Bulk effective sample size (mixing of the centre of the distribution).
    pub ess_bulk: f64,
    /// Tail effective sample size (mixing of the 5%/95% quantiles).
    pub ess_tail: f64,
    /// Monte-Carlo standard error of the posterior mean.
    pub mcse: f64,
}

/// Result of a full MCMC Bayesian fit (`EstimationMethod::Bayes`). Surfaced on
/// [`FitResult::bayes`]. Carries posterior summaries + convergence diagnostics
/// instead of a single point estimate; the optimizer-style fields on
/// `FitResult` (theta/omega/sigma) are populated with the posterior means so
/// downstream consumers that expect a point estimate still work.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct BayesResult {
    /// Per-parameter posterior summaries, ordered θ, then Ω entries, then Σ.
    pub summaries: Vec<PosteriorSummary>,
    /// Number of independent chains run.
    pub n_chains: usize,
    /// Warmup sweeps per chain (discarded from the posterior).
    pub n_warmup: usize,
    /// Retained sampling draws per chain (post-warmup, post-thinning).
    pub n_draws_per_chain: usize,
    /// Total divergent HMC transitions across all chains. Non-zero counts
    /// indicate posterior geometry the sampler could not traverse reliably.
    pub n_divergent: usize,
    /// Worst (largest) split-R̂ across all parameters; convenience for a
    /// single-number convergence check.
    pub max_rhat: f64,
    /// Raw posterior draws, row-major `[chain][draw][param]` flattened, retained
    /// only when the caller requests them (large). `None` otherwise.
    pub draws: Option<Vec<f64>>,
}

/// Outcome of the post-estimation covariance step.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq)]
pub enum CovarianceStatus {
    /// User set `covariance = false`; step was not attempted.
    NotRequested,
    /// Covariance matrix was successfully computed.
    Computed,
    /// Step was attempted but failed (e.g. singular Hessian).
    Failed,
    /// FD Hessian was non-PD; SIR was run as a fallback and succeeded.
    SirFallback,
}

/// What to do when the covariance step produces a non-positive-definite Hessian.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CovarianceFallback {
    /// Do nothing; leave the covariance step as failed (default).
    #[default]
    None,
    /// Run SIR with a proposal built from the rectified (|eigenvalue|) Hessian,
    /// inflated 4× for heavier tails. Parameter uncertainty is then reported as
    /// 95% credible intervals from the SIR posterior quantiles instead of
    /// `H⁻¹`-based standard errors.
    Sir,
}

/// Which estimator to use for the parameter covariance matrix, mirroring
/// NONMEM's `$COVARIANCE MATRIX=` options. All three share the same FD Hessian
/// `R` (the observed information) and per-subject score cross-product
/// `S = Σᵢ gᵢgᵢᵀ`; they differ only in how those are combined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CovarianceMethod {
    /// `R⁻¹` — inverse observed-information (Hessian) matrix. The model-based
    /// covariance; assumes the model is correctly specified (default, NONMEM
    /// `MATRIX=R`).
    #[default]
    Hessian,
    /// `S⁻¹` — inverse cross-product (outer-product-of-gradients) matrix. The
    /// empirical-information covariance (NONMEM `MATRIX=S`).
    CrossProduct,
    /// `R⁻¹ S R⁻¹` — the Huber–White "sandwich". Robust to model
    /// mis-specification; NONMEM's default (`MATRIX=RSR`).
    Sandwich,
}

/// Severity level for a structured warning entry.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum WarningSeverity {
    Critical,
    Warning,
    Info,
}

/// Stable, typed taxonomy for a structured warning (issue #778).
///
/// Each variant is the machine-branchable *code* an agent or the R wrapper keys
/// off — a stable contract, unlike free-text message wording. Serialized (and
/// via [`WarningCode::as_str`]) as a fixed snake_case token; those tokens are a
/// public API surface and must not change once released. Add variants as new
/// warning classes appear; never repurpose an existing token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum WarningCode {
    /// Optimizer did not reach the convergence criterion.
    Convergence,
    /// Informational note about the covariance step (e.g. its evaluation cost).
    CovarianceStep,
    /// The covariance step failed — no standard errors are available.
    CovarianceFailed,
    /// The covariance step succeeded but was regularized / degraded (eigenvalue
    /// floor applied, or a non-finite off-diagonal stencil); SEs may be
    /// over-optimistic.
    CovarianceRegularized,
    /// Correlation / condition-number problem in the covariance.
    ConditionNumber,
    /// Optimizer-health issue (trust-radius collapse, degeneracy).
    OptimizerHealth,
    /// Residual (IWRES) autocorrelation — Durbin–Watson out of range.
    DwAutocorrelation,
    /// ETA distribution departs from normality (Shapiro–Wilk).
    EtaNormality,
    /// An experimental feature (SDE, neural-network components) was used.
    Experimental,
    /// BLOQ / M3 censoring handling caveat.
    BloqMethod,
    /// Sampling-importance-resampling (SIR) issue.
    Sir,
    /// Importance-sampling issue (ESS = 0, proposal collapse).
    ImportanceSampling,
    /// EPS (residual) shrinkage is notably high / negative.
    EpsShrinkage,
    /// One or more ETA (random-effect) shrinkages exceed the threshold — the
    /// data poorly inform those individual random effects.
    EtaShrinkage,
    /// One or more THETA estimates are pinned to an optimizer bound — a sign of
    /// non-identifiability or a too-tight bound.
    BoundaryEstimate,
    /// One or more OMEGA / SIGMA optimizer coordinates are pinned to an internal
    /// runaway guard. Unlike a THETA bound, this is an implementation safety
    /// limit rather than a user-declared constraint, so the affected fit result
    /// is not an interior optimum.
    ParameterAtRunawayGuard,
    /// One or more THETA estimates have a large relative standard error — poorly
    /// estimated / imprecise parameters.
    InflatedRse,
    /// One or more parameter pairs are highly correlated in the covariance —
    /// over-parameterization / non-identifiability.
    HighCorrelation,
    /// Dataset-quality issue (missing DV, ADDL/II, non-positive DV, …).
    DataQuality,
    /// Omega structure caveat (mixed lognormal / additive block).
    OmegaStructure,
    /// A gradient / sampler fallback was taken (e.g. HMC → Metropolis-Hastings).
    GradientFallback,
    /// Mu-referencing note or missing-mu-reference caveat.
    MuReferencing,
    /// Optimizer-configuration note (`global_search`).
    OptimizerConfig,
    /// Multi-start informational note.
    MultiStart,
    /// The run was cancelled by the user.
    Cancelled,
    /// Thread-count efficiency note.
    Threads,
    /// A simulation-time, per-subject diagnostic — a degenerate hazard draw, or an
    /// over-large recurrent-event stream that was skipped/censored — surfaced by
    /// `simulate_with_options_diag` and echoed into `FitResult.warnings` on a
    /// `--simulate` run (#762, #763). Distinct from the fit-time optimizer/data codes:
    /// it flags a pathological simulated subject, not an estimation problem.
    Simulation,
    /// A transit / inverse-Gaussian absorption closed form entered the flip-flop
    /// regime (disposition rate ≥ the tilting abscissa), where it returns an
    /// identically-zero profile. Either auto-rerouted to the ODE twin (an
    /// informational heads-up, #776) or — for a twin-less model at a subject's
    /// fitted EBE — a silently degenerate likelihood contribution the η = 0
    /// fit-start reject could not catch (#785).
    FlipFlop,
    /// An analytic transit / inverse-Gaussian model's ODE twin could not be built, so
    /// the model keeps no ODE fallback (#1008). The twin is an ODE model while the
    /// primary is analytical, so an ODE-scoped parse check can reject a twin whose
    /// primary the parser accepts — most often an individual parameter named after one
    /// of the twin's own states (`CENTRAL` / `PERIPH`). The model still fits as a pure
    /// closed form; what it loses is the reroute, so any subject that needs it (TV
    /// covariates, `TIME`-dependent parameters, IOV, SS / infusion doses, the flip-flop
    /// regime) is rejected with an explicit error rather than mis-served.
    AbsorptionTwinDeclined,
    /// A non-fixed theta whose outer gradient is ≈ 0 at the initial estimate — it
    /// has no effect on the objective (unmapped, or dropped from the structural /
    /// scaling model). The pre-flight guard freezes it at its initial value so the
    /// remaining parameters can be estimated instead of the whole fit dying on an
    /// eval-1 optimizer `Failure` (#826).
    FlatParameter,
    /// An ODE-solver diagnostic from the post-fit pass (#1080 Part B): steps clamped at the
    /// minimum step size (a stability-limited, freeze-padded segment), a stiff escalation the
    /// `auto` guard discarded, or a segment cut short by `ode_stiff_abort_after`. Distinct
    /// from the optimizer codes — it says the *integration* under the final estimates was not
    /// clean, whatever the optimizer made of it.
    OdeSolver,
    /// Unrecognised message — fallback bucket.
    General,
}

impl WarningCode {
    /// The stable snake_case token (identical to the serde representation).
    /// This is the string an agent / the R wrapper branches on.
    pub fn as_str(&self) -> &'static str {
        match self {
            WarningCode::Convergence => "convergence",
            WarningCode::CovarianceStep => "covariance_step",
            WarningCode::CovarianceFailed => "covariance_failed",
            WarningCode::CovarianceRegularized => "covariance_regularized",
            WarningCode::ConditionNumber => "condition_number",
            WarningCode::OptimizerHealth => "optimizer_health",
            WarningCode::DwAutocorrelation => "dw_autocorrelation",
            WarningCode::EtaNormality => "eta_normality",
            WarningCode::Experimental => "experimental",
            WarningCode::BloqMethod => "bloq_method",
            WarningCode::Sir => "sir",
            WarningCode::ImportanceSampling => "importance_sampling",
            WarningCode::EpsShrinkage => "eps_shrinkage",
            WarningCode::EtaShrinkage => "eta_shrinkage",
            WarningCode::BoundaryEstimate => "boundary_estimate",
            WarningCode::ParameterAtRunawayGuard => "parameter_at_runaway_guard",
            WarningCode::InflatedRse => "inflated_rse",
            WarningCode::HighCorrelation => "high_correlation",
            WarningCode::DataQuality => "data_quality",
            WarningCode::OmegaStructure => "omega_structure",
            WarningCode::GradientFallback => "gradient_fallback",
            WarningCode::MuReferencing => "mu_referencing",
            WarningCode::OptimizerConfig => "optimizer_config",
            WarningCode::MultiStart => "multi_start",
            WarningCode::Cancelled => "cancelled",
            WarningCode::Threads => "threads",
            WarningCode::Simulation => "simulation",
            WarningCode::FlipFlop => "flip_flop",
            WarningCode::AbsorptionTwinDeclined => "absorption_twin_declined",
            WarningCode::FlatParameter => "flat_parameter",
            WarningCode::OdeSolver => "ode_solver",
            WarningCode::General => "general",
        }
    }
}

impl std::fmt::Display for WarningCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A structured warning with severity, typed code, and message.
///
/// Populated in parallel with `FitResult.warnings` (which remains for
/// backward compatibility). The `category` is a typed [`WarningCode`] — the
/// stable, machine-branchable taxonomy — replacing the former free-text
/// category string (issue #778).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WarningEntry {
    pub severity: WarningSeverity,
    /// Typed warning code (stable snake_case serde token).
    pub category: WarningCode,
    /// Human-readable message. For messages that carry a multi-stage chain
    /// prefix such as `[FOCEI] ...`, only the body after the prefix is stored
    /// here; the method tag is moved into `source_method`. For unprefixed
    /// messages this is identical to the corresponding entry in
    /// `FitResult.warnings`.
    pub message: String,
    /// For multi-stage chains, the method that produced this warning.
    pub source_method: Option<String>,
    /// Optional structured payload with machine-readable numbers behind the
    /// message (e.g. `{"durbin_watson": 1.2}` or `{"condition_number": 1.4e6}`),
    /// so an agent reads the value directly instead of parsing prose. Populated
    /// by native at-source emitters; `None` on the string-classified fallback
    /// path ([`classify_warning`]). Omitted from JSON when `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

/// Classify a free-text warning message into a structured `WarningEntry`.
///
/// This is the single source of truth for warning severity/category in
/// ferx-core. The R wrapper consumes the structured output directly and never
/// re-classifies message text. Multi-stage chain prefixes (`[FOCEI] ...`) are
/// stripped into `source_method`; the remaining message is matched against the
/// fixed category vocabulary. Unrecognised messages fall back to
/// `Warning`/`general`.
pub fn classify_warning(raw: &str) -> WarningEntry {
    // Strip a leading "[METHOD] " chain prefix into source_method.
    let (source_method, msg) = if let Some(rest) = raw.strip_prefix('[') {
        if let Some(idx) = rest.find(']') {
            let tag = &rest[..idx];
            let body = rest[idx + 1..].trim_start();
            (Some(tag.to_string()), body.to_string())
        } else {
            (None, raw.to_string())
        }
    } else {
        (None, raw.to_string())
    };

    let lower = msg.to_lowercase();

    // (severity, category) keyed off distinctive substrings. Order matters:
    // more specific patterns first.
    // `if_same_then_else`: several arms below deliberately share an outcome —
    // notably the four `DataQuality` arms (ADDL/II, IOV occasion, missing DV,
    // and the LTBS/SS/EVID data group). Each is a distinct warning family that
    // merely classifies the same today, so they are kept as separate,
    // individually commented arms; collapsing them into one `||` condition
    // would erase the grouping and make re-categorising a single family a
    // rewrite instead of a one-line edit. Scoped to this statement rather than
    // the whole fn so a future duplicated arm in another chain still lints.
    #[allow(clippy::if_same_then_else)]
    let (severity, category) = if lower.contains("w_absorption_twin_declined") {
        // #1008: the analytic absorption model kept no ODE twin. Not an estimation problem —
        // the fit that follows is a pure closed form — but the model has silently lost the
        // reroute, so it is worth its own code rather than the `general` bucket.
        //
        // First in the chain, ahead of *every* prose arm, because this message interpolates
        // the twin parser's own error text: any substring an arm below matches ("condition
        // number", "ill-conditioned", "did not converge", …) could arrive inside that reason
        // and misclassify a decline as a Critical estimation failure. The message also *lists*
        // the flip-flop reroute among the features the model just lost, which is how the
        // `flip-flop regime` arm claimed it during development. An explicit `W_` token beats
        // a prose match, so it is tested before any of them rather than just before the one
        // that happened to collide.
        (
            WarningSeverity::Warning,
            WarningCode::AbsorptionTwinDeclined,
        )
    } else if lower.contains("w_ode_solver_escalation_note") {
        // #1080 Part B: the informational half — `ode_method = auto` escalated and the stiff
        // method coped. Its own token, so re-classifying the plain message text recovers the
        // `Info` severity the emitter used rather than promoting a routine note to a warning.
        (WarningSeverity::Info, WarningCode::OdeSolver)
    } else if lower.contains("w_ode_solver_diagnostics") {
        // #1080 Part B: the post-fit solver-statistics pass. Matched on its `W_` token and
        // placed with the other token arm, ahead of the prose ones — the message names
        // methods and quotes counters, and "did not converge" style prose could plausibly be
        // added to it later without anyone remembering this chain exists.
        (WarningSeverity::Warning, WarningCode::OdeSolver)
    } else if lower.contains("did not converge")
        || lower.contains("without convergence")
        || lower.contains("no multi-start run converged")
    {
        (WarningSeverity::Critical, WarningCode::Convergence)
    } else if lower.contains("covariance step failed")
        || lower.contains("covariance failed")
        || (lower.contains("covariance step") && lower.contains("not positive definite"))
    {
        // "ses not available" intentionally omitted — too broad.
        // The compound check catches format_non_pd_warning ("Covariance step:
        // Hessian is not positive definite") whose prefix differs from "failed:"
        // messages. It must be compound so that "SIR failed: covariance not
        // positive definite" (no "covariance step" token) still routes to "sir".
        (WarningSeverity::Critical, WarningCode::CovarianceFailed)
    } else if lower.contains("covariance step regularized")
        || lower.contains("off-diagonal fd stencil")
    {
        // Regularisation warning and off-diagonal NaN soft warning both indicate a
        // degraded-but-present covariance step result. Severity within the message
        // (minor/moderate/severe, or "over-optimistic") informs guidance.
        (WarningSeverity::Warning, WarningCode::CovarianceRegularized)
    } else if lower.contains("ill-conditioned") || lower.contains("condition number") {
        // Note: "covariance step failed: Hessian has ill-conditioned entries" contains
        // "ill-conditioned" but is caught by "covariance step failed" above (else-if chain).
        // Any future covariance message that contains "ill-conditioned" but NOT "failed:"
        // would land here instead — keep this ordering in mind when adding new messages.
        (WarningSeverity::Critical, WarningCode::ConditionNumber)
    } else if lower.starts_with("w_tte_degenerate") || lower.starts_with("w_rtte_degenerate") {
        // Simulation-time per-subject hazard diagnostics (#762/#763). Must precede the
        // generic "degenerate" → optimizer-health branch below, whose substring these
        // otherwise match — these flag a pathological *simulated subject*, not an
        // estimation/optimizer problem.
        (WarningSeverity::Warning, WarningCode::Simulation)
    } else if lower.contains("flip-flop regime") {
        // Transit / IG absorption closed form in the flip-flop regime (#776/#785):
        // auto-rerouted to the ODE twin (informational) or, twin-less at a fitted
        // EBE, a silently degenerate subject. Precedes the generic "degenerate"
        // branch so a flip-flop message never misroutes to optimizer-health.
        (WarningSeverity::Warning, WarningCode::FlipFlop)
    } else if lower.contains("trust radius") || lower.contains("degenerate") {
        (WarningSeverity::Warning, WarningCode::OptimizerHealth)
    } else if lower.contains("autocorrelation") || lower.contains("durbin") {
        (WarningSeverity::Warning, WarningCode::DwAutocorrelation)
    } else if lower.contains("shapiro") || lower.contains("non-normal") {
        (WarningSeverity::Warning, WarningCode::EtaNormality)
    } else if lower.contains("experimental feature") {
        // Experimental-feature notices (issue #175): SDE and neural-network
        // components emit a runtime warning so results are applied with caution.
        (WarningSeverity::Warning, WarningCode::Experimental)
    } else if lower.contains("m3 bloq")
        || lower.contains("bloq handling")
        || lower.contains("m3 censoring")
        || lower.contains("censoring handling")
    {
        (WarningSeverity::Warning, WarningCode::BloqMethod)
    } else if lower.contains("sir failed")
        || lower.contains("sir requested")
        || lower.starts_with("sir:")
        || lower.starts_with("sir fallback:")
    {
        // The `sir:` / `sir fallback:` prefixes carry the proposal-conditioning
        // diagnostics (#1021) — a rank-deficient or bound-shrunk proposal. They
        // reach this arm because no earlier pattern matches them: the shrinkage
        // message mentions "the covariance step" but not "not positive definite",
        // so the compound covariance arm above does not claim it.
        (WarningSeverity::Warning, WarningCode::Sir)
    } else if lower.contains("ess = 0") || lower.contains("proposal collapse") {
        (WarningSeverity::Warning, WarningCode::ImportanceSampling)
    } else if lower.contains("eps shrinkage") {
        (WarningSeverity::Warning, WarningCode::EpsShrinkage)
    } else if lower.starts_with("eta shrinkage") || lower.contains(" eta shrinkage") {
        // Word-boundary match so "beta shrinkage" (or similar) does not collide.
        (WarningSeverity::Warning, WarningCode::EtaShrinkage)
    } else if lower.contains("internal optimizer runaway guard") {
        (
            WarningSeverity::Warning,
            WarningCode::ParameterAtRunawayGuard,
        )
    } else if lower.contains("optimizer bound") {
        // Distinctive phrase; the eps-shrinkage message's "sigma at a bound" is
        // matched earlier and never reaches here.
        (WarningSeverity::Warning, WarningCode::BoundaryEstimate)
    } else if lower.contains("relative standard error") {
        (WarningSeverity::Warning, WarningCode::InflatedRse)
    } else if lower.contains("highly correlated") {
        (WarningSeverity::Warning, WarningCode::HighCorrelation)
    } else if lower.starts_with("w_addl_missing_ii") || lower.contains("addl > 0 but ii") {
        (WarningSeverity::Warning, WarningCode::DataQuality)
    } else if lower.starts_with("w_iov_occ_missing")
        || lower.contains("missing or unparseable values in iov_column")
    {
        (WarningSeverity::Warning, WarningCode::DataQuality)
    } else if lower.starts_with("w_missing_dv") {
        (WarningSeverity::Warning, WarningCode::DataQuality)
    } else if lower.contains("ltbs")
        || lower.contains("non-positive dv")
        || lower.contains("ss=1 dose")
        || lower.contains("ss=1 infusion")
        || lower.contains("evid=3/4")
        || lower.contains("lagtime evaluates")
    {
        (WarningSeverity::Warning, WarningCode::DataQuality)
    } else if lower.contains("mixed lognormal") || lower.contains("mixed log-normal") {
        (WarningSeverity::Warning, WarningCode::OmegaStructure)
    } else if lower.contains("hmc is unavailable") {
        // "falls back to" intentionally removed: no emitted message uses that exact
        // phrase. The SAEM HMC message is fully covered by "hmc is unavailable".
        (WarningSeverity::Info, WarningCode::GradientFallback)
    } else if lower.contains("not mu-referenced") {
        (WarningSeverity::Warning, WarningCode::MuReferencing)
    } else if lower.contains("mu-ref") || lower.contains("mu-referencing") {
        (WarningSeverity::Info, WarningCode::MuReferencing)
    } else if lower.contains("global_search disabled") {
        // Runtime failure: CRS2-LM init failed — the optimiser ran without global search.
        (WarningSeverity::Warning, WarningCode::OptimizerConfig)
    } else if lower.contains("global_search") {
        (WarningSeverity::Info, WarningCode::OptimizerConfig)
    } else if lower.contains("multi-start") {
        (WarningSeverity::Info, WarningCode::MultiStart)
    } else if lower.contains("cancelled by user") {
        (WarningSeverity::Info, WarningCode::Cancelled)
    } else if lower.contains("threads configured") || lower.contains("threads than subjects") {
        (WarningSeverity::Info, WarningCode::Threads)
    } else if lower.contains("n\u{00b2} ofv")
        || lower.contains("n^2 ofv")
        || (lower.contains("parameters") && lower.contains("covariance step:"))
    {
        (WarningSeverity::Info, WarningCode::CovarianceStep)
    } else if lower.contains("has no effect on the objective") {
        // Pre-flight flat-theta guard (#826): a theta with an ≈0 outer gradient at
        // the initial estimate was frozen at its init so the fit could proceed.
        (WarningSeverity::Warning, WarningCode::FlatParameter)
    } else {
        (WarningSeverity::Warning, WarningCode::General)
    };

    WarningEntry {
        severity,
        category,
        message: msg,
        source_method,
        details: None,
    }
}

/// Full fit result
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct FitResult {
    /// Final method in the chain (same as `method_chain.last()`).
    pub method: EstimationMethod,
    /// Full sequence of methods executed, in order. Always has at least one entry.
    pub method_chain: Vec<EstimationMethod>,
    pub converged: bool,
    pub ofv: f64,
    pub aic: f64,
    pub bic: f64,
    pub theta: Vec<f64>,
    pub theta_names: Vec<String>,
    /// Names of the random effects (etas), parallel to the omega diagonal.
    pub eta_names: Vec<String>,
    #[serde(with = "crate::serde_nalgebra::dmatrix")]
    pub omega: DMatrix<f64>,
    pub sigma: Vec<f64>,
    /// Names of the sigma parameters, parallel to `sigma`.
    pub sigma_names: Vec<String>,
    /// Residual error model (additive, proportional, combined).
    ///
    /// For multi-endpoint (per-CMT) models this is only the *representative*
    /// endpoint's error model; it does not describe the other endpoints.
    /// `sigma_types` (parallel to `sigma`/`sigma_names`) carries the correct
    /// per-sigma classification and should be preferred by consumers that need
    /// to distinguish endpoints.
    pub error_model: ErrorModel,
    #[serde(with = "crate::serde_nalgebra::option_dmatrix")]
    pub covariance_matrix: Option<DMatrix<f64>>,
    pub se_theta: Option<Vec<f64>>,
    /// Standard errors for omega elements.
    ///
    /// - **Diagonal omega**: length = n_eta, one SE per variance.
    /// - **Block omega**: length = n_eta·(n_eta+1)/2, column-major lower
    ///   triangle (same layout as the packed Cholesky). Use
    ///   [`omega_se_at`] to index by (i, j).
    pub se_omega: Option<Vec<f64>>,
    pub se_sigma: Option<Vec<f64>>,
    /// FIX flags carried through from the model so the output layer can
    /// render `FIXED` for SE columns rather than the (meaningless) zero
    /// they acquire from the reduced-Hessian covariance step.
    pub theta_fixed: Vec<bool>,
    pub omega_fixed: Vec<bool>,
    pub sigma_fixed: Vec<bool>,
    /// Per-eta SD-init flag (parallel to the omega diagonal): `true` when the
    /// initial value was written as `omega NAME ~ X (sd)`. Lets downstream
    /// printers annotate the estimate with `[initial specified as SD]`.
    /// Always `false` for block-omega entries.
    pub omega_init_as_sd: Vec<bool>,
    /// Per-sigma SD-init flag (parallel to `sigma`): `true` when the initial
    /// value was written as `sigma NAME ~ X (sd)`, `false` for the variance
    /// default.
    pub sigma_init_as_sd: Vec<bool>,
    pub subjects: Vec<SubjectResult>,
    pub n_obs: usize,
    pub n_subjects: usize,
    pub n_parameters: usize,
    pub n_iterations: usize,
    pub interaction: bool,
    pub warnings: Vec<String>,
    /// Structured counterpart to `warnings` — same entries with severity and category metadata.
    pub warnings_structured: Vec<WarningEntry>,
    // SIR results (optional)
    pub sir_ci_theta: Option<Vec<(f64, f64)>>,
    pub sir_ci_omega: Option<Vec<(f64, f64)>>,
    pub sir_ci_sigma: Option<Vec<(f64, f64)>>,
    pub sir_ess: Option<f64>,
    /// Resampled packed parameter vectors retained from the SIR step, available
    /// when `FitOptions.sir_keep_samples = true`. Each `Vec<f64>` is a draw in
    /// the packed parameter space — same layout as `pack_params`:
    /// `[log-theta, Cholesky-omega, log-sigma]`, with the IOV Cholesky block
    /// appended when the model has kappa declarations.
    /// Consumed by `simulate_with_uncertainty()` with `UncertaintyMethod::Sir`.
    pub sir_resamples_packed: Option<Vec<Vec<f64>>>,
    /// Importance-sampling marginal log-likelihood result. `Some` when an
    /// `Imp` stage ran in the method chain (`methods = [..., imp]`). The
    /// `−2 log L_IS` value is lower-bias than the FOCE/Laplace `ofv` for
    /// sparsely-sampled subjects and is the preferred quantity for AIC/BIC
    /// model comparison in those settings. See [`ImportanceSamplingResult`].
    pub importance_sampling: Option<ImportanceSamplingResult>,
    /// Per-iteration IMPMAP parameter trace, analogous to NONMEM `.ext` file
    /// output. `Some` when the final estimating stage was IMPMAP.
    pub impmap_trace: Option<ImpmapTrace>,
    /// Full MCMC Bayesian result. `Some` when `method = bayes` was run;
    /// carries posterior summaries + convergence diagnostics. See [`BayesResult`].
    pub bayes: Option<BayesResult>,
    // IOV results (present when kappa declarations exist in the model)
    #[serde(with = "crate::serde_nalgebra::option_dmatrix")]
    pub omega_iov: Option<DMatrix<f64>>,
    pub kappa_names: Vec<String>,
    pub kappa_fixed: Vec<bool>,
    /// Per-kappa SD-init flag (parallel to the `omega_iov` diagonal). Same
    /// semantics as `omega_init_as_sd` — `true` when the user wrote
    /// `kappa NAME ~ X (sd)`. Always `false` for block_kappa entries.
    pub kappa_init_as_sd: Vec<bool>,
    /// Per-kappa sample-size weight source text (parallel to `kappa_names`,
    /// #1031): `Some("NARM")` for `kappa K ~ γ² weight = NARM`. Empty for a
    /// model with no weighted kappa, so an existing fit YAML is byte-identical.
    ///
    /// The estimate reported for a weighted kappa is the *unweighted* γ² — the
    /// quantity a published MBMA reports — and the arm's effective SD is
    /// `γ/√W`; see `kappa_weight_typical` for the arm size that goes with it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub kappa_weights: Vec<Option<String>>,
    /// Median weight over this dataset's subject-occasions, parallel to
    /// `kappa_weights` (#1031). `Some(200.0)` next to `weight = NARM` means the
    /// median arm carried 200 subjects, so the effective between-arm SD there is
    /// `γ/√200` — the number a reader needs to judge a γ that looks alarming
    /// undivided. `None` when that kappa has no weight.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub kappa_weight_typical: Vec<Option<f64>>,
    pub se_kappa: Option<Vec<f64>>,
    /// Pooled kappa shrinkage: one value per kappa parameter, averaged over all
    /// subject-occasion pairs.  Empty when `n_kappa == 0`.
    pub shrinkage_kappa: Vec<f64>,
    /// Per-occasion kappa shrinkage: `shrinkage_kappa_by_occ[occ_idx][kappa_idx]`.
    /// `occ_idx` is the 0-based position within each subject's own occasion list
    /// (order in which distinct OCC values first appear in that subject's rows),
    /// **not** the raw OCC column value.  For unbalanced designs where subjects
    /// have different OCC sequences, a given `occ_idx` may map to different OCC
    /// values across subjects — use `shrinkage_kappa` (pooled) in that case.
    /// Empty when `n_kappa == 0` or only one occasion is present.
    pub shrinkage_kappa_by_occ: Vec<Vec<f64>>,
    /// Per-subject, per-occasion kappa EBEs.
    /// `ebe_kappas[i][k]` is the kappa vector for subject i, occasion k.
    /// Outer vec is empty when `n_kappa == 0`.
    #[serde(with = "crate::serde_nalgebra::vec_vec_dvector")]
    pub ebe_kappas: Vec<Vec<DVector<f64>>>,
    /// Estimated OFV evaluations saved by the SAEM mu-ref gradient step M-step.
    /// Non-None only when method=saem and mu_referencing=true.
    pub saem_mu_ref_m_step_evals_saved: Option<u64>,
    /// Number of subjects that used HMC at least once during the SAEM E-step.
    /// `None` when `n_leapfrog = 0` (MH-only) or for non-SAEM methods.
    pub saem_n_subjects_hmc: Option<usize>,
    /// Gradient method used in the inner (per-subject EBE) BFGS loop.
    pub gradient_method_inner: String,
    /// Gradient method used in the outer (population parameter) optimizer.
    pub gradient_method_outer: String,
    /// True when the model uses ODE integration; false for analytical PK.
    pub uses_ode_solver: bool,
    /// True when the model has a `[diffusion]` block (SDE / EKF likelihood).
    pub uses_sde: bool,
    /// Number of Rayon worker threads used during this fit.
    pub n_threads_used: usize,
    /// NLopt algorithms requested but not available in this platform build.
    pub nlopt_missing_algorithms: Vec<String>,
    /// Estimated OFV evaluations for the covariance step (n_params²), set
    /// when `run_covariance_step = true` and `n_parameters > 30`.
    pub covariance_n_evals_estimated: Option<usize>,
    /// Path to the per-iteration optimizer trace CSV, present when
    /// `FitOptions::optimizer_trace = true`.
    pub trace_path: Option<String>,
    /// Number of outer iterations in which at least one subject had an
    /// unconverged EBE.  Always `0` for SAEM (which uses MH sampling).
    pub ebe_convergence_warnings: u32,
    /// Worst-case number of unconverged subjects in a single outer iteration.
    pub max_unconverged_subjects: u32,
    /// Total number of times the Nelder-Mead fallback was invoked across all
    /// subjects and all outer iterations.  Always `0` for SAEM.
    pub total_ebe_fallbacks: u32,
    /// Outcome of the post-estimation covariance step.
    pub covariance_status: CovarianceStatus,
    /// ETA shrinkage per random effect: `1 - SD(eta_hat_k) / sqrt(omega_kk)`.
    /// `NaN` when `omega_kk` is zero. Computed from the conditional **mode**
    /// (EBE); for the distribution-based counterpart see
    /// `cond_dist.shrinkage`.
    pub shrinkage_eta: Vec<f64>,
    /// Per-subject conditional distribution of the random effects from the
    /// opt-in SAEM conditional-distribution pass. `Some` only when
    /// `method = saem` and `saem_conddist = true`; `None` otherwise (#257).
    pub cond_dist: Option<CondDist>,
    /// EPS shrinkage: `1 - SD(IWRES)`.  `NaN` when fewer than 2 valid residuals.
    pub shrinkage_eps: f64,
    /// Pooled lag-1 Pearson correlation of IWRES across subjects.
    /// `NaN` when no subject has ≥ 2 valid IWRES values.
    pub iwres_lag1_r: f64,
    /// Pooled Durbin-Watson statistic for IWRES within subjects.
    /// 2.0 = no autocorrelation; < 1.5 = positive; > 2.5 = negative.
    /// `NaN` when no subject has ≥ 2 valid IWRES values.
    pub dw_statistic: f64,
    /// Wall-clock time for the complete fit in seconds.
    pub wall_time_secs: f64,
    /// Wall-clock time spent converging each stage of `method_chain`, parallel
    /// to it, in seconds. Excludes the covariance step (see
    /// `covariance_wall_time_secs`), which only ever runs on the last
    /// estimating stage (#615).
    pub method_wall_times_secs: Vec<f64>,
    /// Wall-clock time spent on the post-estimation covariance step (FD
    /// Hessian / SIR-fallback proposal construction), in seconds. `0.0` when
    /// the covariance step was not run (#713).
    pub covariance_wall_time_secs: f64,
    /// Model name (from the `.ferx` file or "Unnamed").
    pub model_name: String,
    /// ferx-core library version (from Cargo.toml at compile time).
    pub ferx_version: String,
    /// OS/arch/Docker/username snapshot of the machine the fit ran on (#704).
    pub environment: crate::environment::EnvironmentInfo,
    /// Per-ETA transformation metadata (see `EtaParamInfo`). Used by the R
    /// layer to pick the correct CI / CV% formula for each random effect.
    pub eta_param_info: Vec<EtaParamInfo>,
    /// Per-theta transformation (Identity / Log / Logit), parallel to `theta`.
    /// Tells the R layer whether a theta must be back-transformed before display.
    pub theta_transform: Vec<ThetaTransform>,
    /// Per-sigma type (Proportional / Additive), parallel to `sigma`.
    pub sigma_types: Vec<SigmaType>,
    /// Eigenvalues of the correlation matrix of free (non-fixed) parameters,
    /// sorted descending. `None` when the covariance step was not run, failed,
    /// or fewer than two free parameters exist.
    pub cov_eigenvalues: Option<Vec<f64>>,
    /// Ratio of the largest to smallest eigenvalue of the correlation matrix of
    /// free parameters. `f64::INFINITY` when the smallest eigenvalue is
    /// non-positive (signals a near-singular parameter space). `None` when
    /// `cov_eigenvalues` is `None`.
    pub cov_condition_number: Option<f64>,
    /// Whether each BSV eta is lognormally parameterised (`true`) or
    /// additive/unknown (`false`). Parallel to `eta_names` / omega diagonal.
    pub eta_log_transformed: Vec<bool>,
    /// Parameter-level correlation matrix for BSV omega.  Entry `[i,j]` uses
    /// the lognormal formula `(exp(ω_ij)−1)/√((exp(ω_ii)−1)(exp(ω_jj)−1))`
    /// when both etas are lognormal, otherwise falls back to
    /// `ω_ij/√(ω_ii·ω_jj)`.  `None` when omega is diagonal (no off-diagonals).
    #[serde(with = "crate::serde_nalgebra::option_dmatrix")]
    pub omega_param_corr: Option<DMatrix<f64>>,
    /// Parameter-level correlation matrix for IOV block kappa, analogous to
    /// `omega_param_corr`.  `None` when `omega_iov` is absent or diagonal.
    #[serde(with = "crate::serde_nalgebra::option_dmatrix")]
    pub omega_iov_param_corr: Option<DMatrix<f64>>,
    /// Path to the `.ferx` model file used for this fit, as supplied by the
    /// caller. `Some` when the fit was launched via `fit_from_files` or
    /// `run_model_with_data`; `None` when `fit()` was called with an in-memory
    /// `CompiledModel`. Stored verbatim (no canonicalisation) so paths don't
    /// leak the runner's home directory into shared `.fitrx` bundles.
    pub model_path: Option<String>,
    /// Path to the NONMEM-format CSV data file used for this fit, as supplied
    /// by the caller. `Some` / `None` follows the same rules as `model_path`.
    pub data_path: Option<String>,
    /// SHA-256 hex digest (64 chars, lowercase) of the model file bytes at
    /// fit time. Used by `run_sir` to refuse stale data when the caller
    /// re-supplies a model or asks the function to re-read from `model_path`.
    /// Computed only when the fit was launched from a file path.
    pub model_hash: Option<String>,
    /// SHA-256 hex digest of the data file bytes at fit time. Same semantics
    /// as `model_hash`.
    pub data_hash: Option<String>,
    /// Verbatim content of the `.ferx` model file. `Some` when the fit was
    /// launched via `fit_from_files` / CLI or loaded from a `.fitrx` bundle;
    /// `None` for in-memory `fit()` callers who never had a file path.
    pub model_text: Option<String>,
    /// Initial theta values as supplied to the optimizer, parallel to `theta`
    /// and `theta_names`.
    pub theta_init: Vec<f64>,
    /// Initial omega matrix (variance scale), same layout as `omega`.
    #[serde(with = "crate::serde_nalgebra::dmatrix")]
    pub omega_init: DMatrix<f64>,
    /// Initial sigma values, parallel to `sigma` and `sigma_names`.
    pub sigma_init: Vec<f64>,
    /// `(min_time, max_time)` across all observation records. `None` only when
    /// there are no observations at all.
    pub obs_time_range: Option<(f64, f64)>,
    /// Gradient of the objective function at the best-OFV parameter point,
    /// in the packed parameter space (log-theta, Cholesky-omega, log-sigma).
    /// `Some` only for NLopt gradient-based runs (SLSQP, L-BFGS, MMA) when at
    /// least one gradient-requesting iteration improved the OFV; `None` for
    /// BOBYQA (derivative-free), built-in BFGS, GN, and SAEM.
    pub final_gradient: Option<Vec<f64>>,
    // ── Run settings (for runlog / reproducibility) ──────────────────────────
    /// Outer optimizer used for this fit, as a lowercase label ("bobyqa",
    /// "slsqp", "nlopt_lbfgs", "mma", "bfgs", "lbfgs", "trust_region"). When the
    /// `optimizer = auto` default resolved the choice, the label is the compound
    /// form `"auto (<resolved>)"` — e.g. `"auto (nlopt_lbfgs)"` — recording both
    /// the setting and what actually ran. SAEM/GN/IMP report their own fixed
    /// labels ("saem", "gn", "imp-bobyqa", "impmap-bobyqa"). Always populated;
    /// the label is the same regardless of method chain length. Consumers that
    /// match on the label should accept the `auto (...)` prefix (#490).
    pub optimizer: String,
    /// Number of random multi-starts attempted. 1 means a single fit from
    /// the model-file initial values (no multi-start).
    pub n_starts: usize,
    /// Seed used to perturb initial values across multi-starts.  `None` when
    /// `n_starts == 1` (no perturbation applied) or when no seed was set and
    /// the run used a random seed derived from the system clock.
    pub multi_start_seed: Option<u64>,
    /// Seed used for the SAEM MCMC E-step.  `None` for non-SAEM methods or
    /// when no explicit seed was set in `[fit_options]`.
    pub saem_seed: Option<u64>,
    /// Seed used for the SIR resampling step.  `None` when SIR was not run or
    /// no explicit seed was set.
    pub sir_seed: Option<u64>,
    /// Seed used for the importance-sampling Monte Carlo step.  `None` when IS
    /// was not run or no explicit seed was set.
    pub imp_seed: Option<u64>,
    /// Effective RNG seed used for the simulation-based NPDE/NPD diagnostics —
    /// the value actually fed to the simulator, including the built-in default
    /// when `[fit_options] npde_seed` was left unset, so the diagnostic is
    /// reproducible from this field alone. `None` when NPDE did not run
    /// (`npde_nsim = 0`).
    pub npde_seed: Option<u64>,
    /// LOQ censoring handling method: "drop" (treat CENS rows as ordinary) or
    /// "m3" (M3 likelihood for censored observations).
    pub bloq_method: String,
    /// Maximum number of outer optimizer iterations allowed.
    pub outer_maxiter: usize,
    /// Gradient-norm convergence tolerance for the outer optimizer.
    pub outer_gtol: f64,
    /// NCA initialisation method used to derive starting values, if any.
    /// One of "nca", "nca_sweep", "nca_ebe", or `None` when the model-file
    /// initial values were used directly.
    pub inits_from_nca: Option<String>,
    /// Names of covariate columns present in the dataset, in the order they
    /// appear in the data file's header.  Mirrors the NONMEM `$INPUT` echo —
    /// lets `ferx_runlog()` report which covariates were available without
    /// requiring the caller to re-read the CSV.  Empty for in-memory `fit()`
    /// calls that never touch a file.
    pub covariate_names: Vec<String>,
    /// All column headers from the data CSV in original order (ID, TIME, DV, AMT, ...,
    /// covariates), analogous to NONMEM `$INPUT`. Empty for in-memory `fit()` calls.
    pub input_columns: Vec<String>,
    /// One entry per `[covariate_nn NAME]` block in the model, populated by
    /// `fit()` from `CompiledModel.covariate_nns`. Empty when the `nn`
    /// feature is off or no block is declared. Output writers
    /// (`write_estimates_yaml`, `print_results`, `.fitrx`) use this to
    /// collapse the wall of per-weight thetas (`W_NN_l_i_j`, `B_NN_l_i`)
    /// into a single readable `neural_networks:` summary section — see
    /// `plans/dcm-and-low-dim-node.md` "Option E".
    #[cfg(feature = "nn")]
    pub neural_networks: Vec<NeuralNetworkInfo>,
    /// Echo of the declared covariate columns from the input dataset (ID, TIME,
    /// EVID + one column per declared covariate, one row per input record).
    /// `Some` only when the model has a `[covariates]` block AND the fit was
    /// launched from a data file; `None` for the in-memory [`fit`] entry point
    /// (which has no raw rows) or when no `[covariates]` block is declared.
    /// Missing values are `f64::NAN`. See [`CovariateTable`].
    pub covariate_table: Option<CovariateTable>,
    /// Record-level exclusion statistics; `Some` when `[data_selection]` rules
    /// were active during the fit (or the caller supplied `ignore`/`accept`
    /// expressions).  `None` means no filtering was requested.
    pub exclusions: Option<ExclusionSummary>,
    /// The optimizer's **exact** final packed parameter vector (log-theta,
    /// Cholesky-omega lower triangle, log-sigma, over the free parameters), lifted
    /// from [`OuterResult::packed_estimate`]. Lets [`run_covariance`] reproduce the
    /// inline covariance step's FD-Hessian bit-for-bit by reusing this exact
    /// Cholesky factor instead of re-decomposing `omega` (which is not the
    /// round-trip inverse of the stored `L·Lᵀ`; the FD Hessian amplifies the
    /// difference on ill-conditioned ω directions — the divergence #816 review
    /// surfaced). `Some` for every packed-Cholesky-space optimizer — the NLopt and
    /// BFGS outer loops, the trust region, and Gauss-Newton (pure and hybrid). `None`
    /// for SAEM and importance-sampling (which rebuild `omega` from the reported
    /// matrix, so both paths re-decompose identically and already agree), for Bayes
    /// (no Hessian covariance step), and for hand-built results.
    ///
    /// **Transient (`#[serde(skip)]`):** it is an in-process optimisation for a
    /// `run_covariance` called right after a fit; a fit reloaded from `.fitrx`
    /// carries `None` and `run_covariance` falls back to re-packing from `omega`
    /// (the documented "~1e-5, platform dependent" path). Never serialized, so no
    /// `.fitrx` / ferx-r format change.
    #[serde(skip)]
    pub packed_estimate: Option<Vec<f64>>,
    /// True when this result was reconstructed from a `.fitrx` checkpoint by
    /// [`crate::io::fitrx::load_fit`] rather than produced by a live fit.
    ///
    /// The checkpoint format does not round-trip per-record discrete-endpoint
    /// diagnostics (`SubjectResult::discrete_rows`), so a restored fit of a binary /
    /// categorical model has none. This flag lets [`crate::io::output::write_sdtab_csv`]
    /// refuse a restored fit whose model declares a binary endpoint (rows genuinely
    /// dropped) while still writing a live fit. It must be paired with the model's
    /// declared endpoints — parsed from [`Self::model_text`] — because the *reconstructed
    /// population* is not a usable signal: `load_fit` re-reads the data with default
    /// (Gaussian) routing, so binary DVs come back as continuous observations and no
    /// [`ObsRecord::DiscreteState`] survives. Runtime-only: `#[serde(skip)]`, so no
    /// `.fitrx` / JSON / ferx-r format change, and a live fit always carries `false`.
    /// Removed once diagnostics round-trip (#911).
    #[serde(skip)]
    pub restored_from_checkpoint: bool,
}

impl FitResult {
    /// Schema version for the machine-readable JSON serialization (#777).
    ///
    /// Bump on any breaking change to the JSON shape (renamed/removed fields,
    /// changed nesting). Additive fields do not require a bump. Agent/tool
    /// consumers pin on this via the top-level `schema_version` key.
    pub const JSON_SCHEMA_VERSION: u32 = 1;

    /// Serialize this fit to a complete, agent-friendly `serde_json::Value`,
    /// with a top-level `schema_version` key injected alongside every field.
    ///
    /// This is the in-process counterpart to writing `{model}-fit.json`:
    /// Rust/R callers get the structured payload without a file round-trip.
    /// Non-finite floats (`NaN`/`±Inf`) serialize to JSON `null`, matching the
    /// empty-cell convention of the YAML/CSV writers.
    pub fn to_json_value(&self) -> serde_json::Value {
        let mut v = serde_json::to_value(self).expect("FitResult is serializable");
        if let serde_json::Value::Object(map) = &mut v {
            map.insert(
                "schema_version".to_string(),
                serde_json::json!(Self::JSON_SCHEMA_VERSION),
            );
        }
        v
    }
}

/// Look up the SE for omega element (i, j) from the `se_omega` vector.
///
/// `se_omega` may be diagonal-only (length = n_eta) or full lower-triangle
/// (length = n_eta·(n_eta+1)/2, column-major).  Returns `None` when
/// `se_omega` is `None`, the index is out of bounds, or the format is
/// diagonal and an off-diagonal element is requested.
pub fn omega_se_at(se_omega: &Option<Vec<f64>>, n_eta: usize, i: usize, j: usize) -> Option<f64> {
    let se = se_omega.as_ref()?;
    let (r, c) = if i >= j { (i, j) } else { (j, i) }; // ensure r >= c
    let n_lt = crate::estimation::parameterization::omega_packed_len(n_eta, false);
    if se.len() == n_lt && n_lt != n_eta {
        // Full lower-triangle format (block omega). The packed index is the
        // shared column-major `chol_lt_idx` (same convention as `pack_params`).
        //
        // DELIBERATE behavior change vs the old inline offset (NOT pure motion):
        // an out-of-range `r >= n_eta` now returns `None`. The old code did
        // `se.get(col_offset + (r - c))`, which for e.g. `(3,0)` on an n_eta=3
        // block returned `Some(se[3])` — a silently-wrong SE for a flat index
        // that happened to land inside the packed vector. `None` is the correct
        // answer. This is only reachable by an external caller: every in-repo
        // caller loops `i,j < n_eta`, so the guard is inert internally. The
        // guard also keeps `chol_lt_idx`'s `debug_assert!(i < n)` from firing.
        if r < n_eta {
            se.get(crate::estimation::parameterization::chol_lt_idx(
                r, c, n_eta,
            ))
            .copied()
        } else {
            None
        }
    } else {
        // Diagonal-only format: only (i, i) is available.
        if r == c {
            se.get(r).copied()
        } else {
            None
        }
    }
}

/// Minimal per-NN metadata carried on `FitResult` so output writers can
/// summarise NN weights without re-walking `theta_names` to detect them.
#[cfg(feature = "nn")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NeuralNetworkInfo {
    /// Block name from the model file (e.g. `TYPICAL_PK`).
    pub name: String,
    /// Layer shape including input and output dimensions
    /// (e.g. `[2, 16, 5]` for 2 inputs → 16 hidden → 5 outputs).
    pub shape: Vec<usize>,
    /// Hidden-layer activation name (e.g. `"tanh"`, `"relu"`).
    pub hidden_activation: String,
    /// Output-layer activation name.
    pub output_activation: String,
    /// Total weight + bias count for this NN.
    pub n_weights: usize,
    /// Index into `FitResult.theta` (and `FitResult.theta_names`) where
    /// this NN's contiguous weight block starts.
    pub weights_offset: usize,
    /// Input covariate names in declaration order.
    pub input_names: Vec<String>,
    /// PK output names in declaration order.
    pub output_names: Vec<String>,
    /// Per-input `center`, in `input_names` order (all-zero when the model
    /// declares no normalisation).
    ///
    /// Recorded because the network computes on `(x − center) / scale`, not on
    /// the raw covariate: without these the reported `weights_offset` slice of
    /// `theta` cannot be evaluated on new data. Consumers that reconstruct the
    /// network from the fit output (`ferx-r`'s `.fitrx` round-trip) need both
    /// vectors. `init` is deliberately **not** recorded — it is a starting value
    /// for the optimizer, superseded by the fitted weights, and says nothing
    /// about what the converged network computes.
    ///
    /// `serde(default)` so a `.fitrx` bundle written before this field existed still
    /// deserialises; such a bundle predates `center`/`scale` entirely, so the empty
    /// vector it yields means "no transform" and should be read as the identity.
    #[serde(default)]
    pub input_center: Vec<f64>,
    /// Per-input `scale`, in `input_names` order (all-one when the model declares
    /// no normalisation). See [`NeuralNetworkInfo::input_center`].
    #[serde(default)]
    pub input_scale: Vec<f64>,
}

/// How the IOV occasion partition — which data rows belong to which occasion —
/// is determined. IOV *random effects* (`kappa`) are always declared in the
/// model DSL; this only controls where the occasion **labels** come from.
///
/// The chosen rule populates `Subject::occasions` / `Subject::dose_occasions`,
/// the same vectors the dataset-column path fills, so everything downstream
/// (`split_obs_by_occasion`, the inner-loop kappa expansion) is unchanged.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum IovOccasionRule {
    /// Default: occasion labels come from the dataset column named by
    /// [`FitOptions::iov_column`] (or none, if that is unset).
    #[default]
    Column,
    /// Dose-triggered: each administration begins a new occasion. An
    /// observation takes the occasion of the most recent dose at or before its
    /// time (dose-before-observation tie-break at equal TIME). ADDL-expanded
    /// doses each start their own occasion.
    PerDose,
    /// Time-window breakpoints. The (strictly increasing) edges define bins:
    /// `time(24, 48)` → occasions `[-inf,24) = 0`, `[24,48) = 1`, `[48,inf) = 2`.
    /// A row's occasion is the number of breakpoints at or before its time.
    TimeWindows(Vec<f64>),
}

/// Options for fit()
#[derive(Debug, Clone)]
pub struct FitOptions {
    /// Primary estimation method (used when `methods` is empty).
    /// When `methods` is non-empty, `method` is ignored for execution and
    /// is set to the final method in the chain for backwards-compatible reporting.
    pub method: EstimationMethod,
    /// Sequence of estimation methods to run. Each stage's converged parameters
    /// are used as the initial values for the next stage. The final stage
    /// produces the reported fit (covariance, diagnostics, OFV). Leave empty
    /// to run a single stage using `method`.
    pub methods: Vec<EstimationMethod>,
    pub outer_maxiter: usize,
    pub outer_gtol: f64,
    /// Relative **step** tolerance for the derivative-free outer optimizer
    /// (`Bobyqa`): NLopt's `xtol_rel`, controlling how far the trust radius must
    /// shrink to declare success. Default `1e-4` (~0.01% move in the natural-scale
    /// parameter). Other optimizers use their own fixed internal tolerances.
    pub outer_xtol: f64,
    /// Relative **objective** tolerance for the derivative-free outer optimizer
    /// (`Bobyqa`): NLopt's `ftol_rel`, the relative OFV change below which the
    /// optimizer stops. `None` (the default) auto-selects per model: `1e-8` for a
    /// pure-TTE model (an *exactly*-evaluated hazard objective, where the looser
    /// historical `1e-6` stopped short on a near-flat frailty-ω² ridge — #469 — and
    /// the reported variance landed above its own minimum), and `1e-6` otherwise.
    /// The `1e-6` floor for non-TTE fits is deliberate: on a *noisy* objective
    /// (ODE solver error, FD-inner FOCE) a tighter `ftol` is unreachable, so BOBYQA
    /// grinds toward its maxeval budget instead of converging. `Some(x)` pins `x`
    /// for every model, overriding the auto-selection.
    pub outer_ftol: Option<f64>,
    pub inner_maxiter: usize,
    pub inner_tol: f64,
    /// Guarded multi-start count for the inner EBE search (`[fit_options]
    /// inner_restarts`, default `1`; set `0` to disable). A multimodal individual objective —
    /// e.g. saturable protein binding, which admits a high-V/low-conc and a
    /// low-V/high-conc basin for the same data — can trap a single warm-started
    /// BFGS in the wrong basin, inflating that subject's objective. When `> 0`,
    /// a subject re-solves the EBE **on a cold start** from `inner_restarts`
    /// Ω-scaled alternate seeds and keeps the lowest-objective mode; the outer
    /// loop's warm start then carries the chosen basin forward, so the scan runs
    /// once per subject per fit (≈0 overhead). Subjects on the event-driven path
    /// (system resets / time-varying covariates) re-seed every random effect;
    /// every other subject is probed per coordinate for a weakly-identified
    /// (flat individual objective / high per-subject shrinkage) random effect and
    /// re-seeds only those — the #891 case, where a distant lower posterior mode
    /// hides from a single descent. Unimodal, well-identified subjects are
    /// unaffected — an alternate seed that reconverges to the same mode is not
    /// accepted — so `0` and the untriggered subjects stay bit-identical.
    pub inner_restarts: usize,
    /// Inner EBE-reconvergence tolerance used **only by the covariance step**
    /// (`[fit_options] cov_inner_tol`), decoupled from the fit's `inner_tol`. The
    /// covariance R-matrix is a second-difference of the reconverged OFV, and that
    /// reconvergence is far more sensitive to EBE precision than the fit itself —
    /// on a flat / weakly-identified surface an EBE converged only to a loose
    /// `inner_tol` can leave enough residual noise to perturb the standard errors
    /// (e.g. the heavily-censored M3 + IOV case in #654).
    ///
    /// `None` (default) uses `inner_tol`, tightened to at most
    /// [`LTBS_COV_INNER_TOL`](Self::LTBS_COV_INNER_TOL) (`1e-8`) **for LTBS models**
    /// (whose `g = ln(f)` covariance Hessian needs it); non-LTBS models keep
    /// `inner_tol`, so their SEs are byte-identical. Set explicitly to opt any model
    /// in (e.g. `cov_inner_tol = 1e-11` for the #654 heavily-censored M3 + IOV case)
    /// or to override the LTBS default. See [`FitOptions::effective_cov_inner_tol`].
    pub cov_inner_tol: Option<f64>,
    /// RK45 ODE solver relative tolerance (`[fit_options] ode_reltol`, or via
    /// `ferx_fit(settings = list(ode_reltol = ...))`). Default `1e-4`. Only
    /// affects ODE models. The default reproduces analytical closed forms in
    /// PRED to ~1e-4, but the FOCE OFV amplifies solver error, so a tighter
    /// value (e.g. `1e-10`) is needed for the ODE-form OFV to match the
    /// analytical OFV. Copied onto `OdeSpec::solver_opts` via
    /// [`CompiledModel::sync_ode_solver_opts`].
    pub ode_reltol: f64,
    /// RK45 ODE solver absolute tolerance (`[fit_options] ode_abstol`).
    /// Default `1e-6`. See [`FitOptions::ode_reltol`].
    pub ode_abstol: f64,
    /// RK45 ODE solver maximum step count per integration segment
    /// (`[fit_options] ode_max_steps`). Default `10000`. Raise if a tight
    /// `ode_reltol` exhausts the step budget on stiff multi-compartment
    /// segments. See [`FitOptions::ode_reltol`].
    pub ode_max_steps: usize,
    /// ODE stepper (`[fit_options] ode_method`). Default
    /// [`Auto`](crate::ode::OdeMethod::Auto) (#978) — an a-priori Jacobian eigenvalue probe
    /// picks the stepper per integration segment, keeping the explicit
    /// [`Rk45`](crate::ode::OdeMethod::Rk45) on everything that is not stability-limited.
    /// Naming a method pins it and skips the probe; `rk45` reproduces the pre-#978 default
    /// exactly. The stiff alternatives (`rosenbrock23`, `rodas4`, `rodas5p`) are
    /// linearly-implicit; see [`crate::ode::OdeMethod`] for when they pay and which
    /// integration paths honour them. Copied onto `OdeSpec::solver_opts` via
    /// [`CompiledModel::sync_ode_solver_opts`], alongside the tolerances.
    pub ode_method: crate::ode::OdeMethod,
    /// Abort an ODE segment once this many of its steps have clamped at the solver's
    /// minimum step size (`[fit_options] ode_stiff_abort_after`, #708/#1080). Default
    /// `None` — grind on to [`ode_max_steps`](Self::ode_max_steps), the pre-#1080
    /// behaviour.
    ///
    /// A clamp means the step failed its error test and was accepted anyway because `dt`
    /// could not shrink further, i.e. the segment is stability-limited rather than
    /// accuracy-limited. Setting a budget bounds what such a segment costs; the tail it
    /// did not integrate is freeze-padded with the last state, so this is a
    /// cost/accuracy trade, not a free win — the fit's ODE-solver warning reports how
    /// many segments were cut short. Prefer `ode_method = auto` (the default) or a named
    /// stiff method first; reach for this when a fit is grinding and you want it to say
    /// so quickly. Copied onto `OdeSpec::solver_opts` via
    /// [`CompiledModel::sync_ode_solver_opts`].
    pub ode_stiff_abort_after: Option<u32>,
    /// Let `ode_method = auto` change stepper **inside** an integration segment, not only at
    /// its start (`[fit_options] ode_auto_switch`, #1080 Part C). Default `true`; ignored
    /// under a named `ode_method`, which is pinned.
    ///
    /// The a-priori probe reads the Jacobian at the segment's entry state, which is the state
    /// a binding model looks least stiff at — both factors of `KON · C · R` are zero when the
    /// dose lands. With this on, the probe is re-read every 25 accepted steps and the stepper
    /// is swapped in place when the verdict changes, keeping the state integrated so far.
    /// Setting it to `false` restores one method per segment, chosen at its start. Copied onto
    /// `OdeSpec::solver_opts` via [`CompiledModel::sync_ode_solver_opts`].
    pub ode_auto_switch: bool,
    pub run_covariance_step: bool,
    /// *Initial* relative step size for the finite-difference Hessian in the
    /// covariance step. The actual step for parameter i is
    /// `fd_hessian_step * (1 + |x_hat[i]|)`. Default `1e-2`. ferx halves this
    /// automatically (up to 8×) if a diagonal stencil comes back non-finite, so
    /// manual tuning is rarely needed for overflow; decrease (e.g. `1e-3`) for
    /// smoother OFV surfaces where FD noise is the main concern.
    /// When `true` (the default) and the model is in analytic-covariance scope, the
    /// covariance R-matrix is the **exact analytic Hessian** of the FOCE/FOCEI marginal
    /// (#436), assembled from third-order sensitivities instead of second-differencing the
    /// reconverged objective. Out-of-scope models fall back to the finite-difference stencil
    /// regardless — it is correct for everything — so this key only ever chooses between two
    /// routes to the same quantity. Set `false` to force finite differences.
    pub analytic_cov_hessian: bool,
    pub fd_hessian_step: f64,
    /// What to do when the FD Hessian is non-positive-definite.
    /// Default [`CovarianceFallback::None`] leaves the covariance step as failed.
    /// [`CovarianceFallback::Sir`] runs SIR with a fallback proposal covariance
    /// built from the rectified (`|eigenvalue|`) Hessian, inflated 4×.
    pub covariance_fallback: CovarianceFallback,
    /// Which covariance estimator to assemble, mirroring NONMEM `$COV MATRIX=`.
    /// Default [`CovarianceMethod::Hessian`] (`R⁻¹`). [`CovarianceMethod::CrossProduct`]
    /// (`S⁻¹`) and [`CovarianceMethod::Sandwich`] (`R⁻¹SR⁻¹`) add the per-subject
    /// score cross-product `S`; currently supported for FOCEI and IOV fits.
    pub covariance_method: CovarianceMethod,
    /// Whether `covariance_method` was set explicitly rather than left at its
    /// default. Exists only so the high-dimension routing guard (#1064) can
    /// tell "the user asked for `MATRIX=R`" from "nobody said anything": at a
    /// few hundred free parameters the FD-of-OFV `R` matrix reconverges every
    /// subject's EBEs at each of `n(n+1)/2` stencil points, so a *defaulted*
    /// `Hessian` routes to the cross-product and an explicit one is honoured
    /// with a warning about the cost.
    pub covariance_method_set: bool,
    pub interaction: bool,
    pub verbose: bool,
    /// Outer-loop (population parameter) optimizer. Defaults to
    /// [`Optimizer::Auto`], which picks `nlopt_lbfgs` when the analytic FOCE/FOCEI
    /// gradient is available and `bobyqa` when only finite differences are (#490).
    pub optimizer: Optimizer,
    /// Inner-loop (EBE) optimizer. `Auto` keeps the size-based default; any other
    /// value pins the inner solver with no dimension-based switching.
    pub inner_optimizer: InnerOptimizer,
    pub lbfgs_memory: usize,
    /// Run a gradient-free global pre-search (NLopt GN_CRS2_LM) before local optimization.
    pub global_search: bool,
    /// Max evaluations for the global pre-search (0 = auto).
    pub global_maxeval: usize,
    // SAEM-specific options
    pub saem_n_exploration: usize,
    pub saem_n_convergence: usize,
    /// Number of block-kernel MH proposals per subject per SAEM outer
    /// iteration. The default 20 mixes well on hard cold-start surfaces
    /// (e.g. Emax PKPD with stressful initial values, where chains at 3
    /// proposals can lock the M-step into a degenerate basin with the
    /// PD-curve thetas at boundary values) and, together with the
    /// componentwise kernel (run automatically for multi-η models), keeps a
    /// block Ω from collapsing to a near rank-1 correlation matrix. The
    /// componentwise sweep count `max(2, n_mh_steps / n_eta)` is derived from
    /// this value (the kernel is skipped entirely for single-η models).
    /// Reduce to 3-10 for the older/faster behaviour on simpler
    /// well-identified models; raise (30-50) only when the diagnostic shows
    /// the M-step is still tracking correlated samples.
    pub saem_n_mh_steps: usize,
    pub saem_adapt_interval: usize,
    /// Exploration-phase cap on the stochastic-approximation step for the
    /// **numerical θ/σ M-step** (issue #1011); `None` uses the
    /// `MSTEP_SA_MAX_STEP` default of 0.03.
    ///
    /// The M-step result is blended in as `θ ← θ + γ_θ·(θ* − θ)` rather than
    /// assigned, because assigning it outright is `argmax` of a *single* MCMC η
    /// draw rather than the SA average of `E[argmax]` — a Monte-Carlo bias that
    /// does not decay with iteration count for a θ with no ETA. This is the
    /// θ-side counterpart of the Ω cap; in the convergence phase the cap lifts
    /// and the full decaying `γ = 1/(k−k1)` applies either way.
    ///
    /// Smaller damps harder. **`1.0` disables the damping**, reproducing the
    /// pre-#1011 assignment exactly. Must be in `(0, 1]`.
    ///
    /// Ignored when the numerical M-step has no θ to estimate (every θ
    /// mu-referenced or `FIX`), and for mixture models — see
    /// `estimation::saem::damps_numerical_mstep`.
    pub saem_mstep_damping: Option<f64>,
    /// Number of initial exploration iterations during which the BSV/IOV Ω
    /// M-step is suppressed (Ω held at its initial value) while the MH chain
    /// warms up. Prevents the iteration-1 Ω collapse on sparse data, where a
    /// cold-start chain (η = 0, few MH steps) yields a tiny `(1/N)Σηηᵀ` that
    /// the γ=1 M-step would otherwise install as Ω, starving the proposal.
    /// Clamped to `saem_n_exploration` at use. `0` disables the burn-in.
    pub saem_omega_burnin: usize,
    pub saem_seed: Option<u64>,
    /// Run a post-fit conditional-distribution pass after SAEM converges.
    ///
    /// When `true`, after the population parameters are fixed the existing MH
    /// kernels are re-run per subject (warm-started at the EBE mode) and the
    /// draws are *accumulated* to estimate each subject's conditional mean and
    /// SD of η — the analogue of saemix `conddist.saemix` / Monolix's
    /// "Conditional Distribution" task. Default `false` (mode-only output,
    /// unchanged behaviour). See `docs/estimation/saem.qmd` (#257).
    pub saem_conddist: bool,
    /// Number of retained MH sweeps per subject in the conditional-distribution
    /// pass (after burn-in). Larger values tighten the conditional mean/SD
    /// estimates at linear wall cost. Only used when `saem_conddist = true`.
    pub saem_conddist_nsamp: usize,
    /// Burn-in sweeps discarded before accumulation in the conditional-
    /// distribution pass, to forget the EBE-mode warm start. Only used when
    /// `saem_conddist = true`.
    pub saem_conddist_burnin: usize,
    /// Retain the raw per-subject draws (not just mean/SD) from the
    /// conditional-distribution pass on the result. Adds
    /// `n_subjects * nsamp * n_eta * 8` bytes; default `false`.
    pub saem_conddist_keep_samples: bool,
    /// Number of leapfrog steps per HMC proposal in the SAEM E-step.
    /// `0` (default) uses the Metropolis-Hastings random-walk sampler.
    /// A positive value (e.g. `3`) enables HMC; requires an analytical PK
    /// model (the HMC `∂NLL/∂η` is the `Dual2` analytic gradient) — falls
    /// back to MH otherwise.
    pub saem_n_leapfrog: usize,
    // Bayes (Gibbs-within-HMC) options — see EstimationMethod::Bayes.
    /// Number of warmup (burn-in + adaptation) sweeps per chain, discarded from
    /// the reported posterior. HMC step size / leapfrog count adapt during this
    /// phase. Default 1000.
    pub bayes_warmup: usize,
    /// Number of post-warmup sampling sweeps retained per chain (before
    /// thinning). Default 1000.
    pub bayes_iters: usize,
    /// Number of independent chains (run with distinct seeds; used for
    /// split-R̂ / cross-chain diagnostics). Default 4.
    pub bayes_chains: usize,
    /// Keep every `bayes_thin`-th sampling draw. `1` (default) keeps all draws.
    pub bayes_thin: usize,
    /// Base RNG seed for the Bayes sampler. Chain `c` uses a seed derived from
    /// this. `None` draws a nondeterministic seed.
    pub bayes_seed: Option<u64>,
    /// Levenberg-Marquardt damping factor for Gauss-Newton (0 = pure GN).
    pub gn_lambda: f64,
    // SIR options
    pub sir: bool,
    pub sir_samples: usize,
    pub sir_resamples: usize,
    pub sir_seed: Option<u64>,
    /// When `true` and SIR is enabled, the resampled packed parameter vectors
    /// are retained on `FitResult.sir_resamples_packed` for downstream use by
    /// `simulate_with_uncertainty()`. Adds `n_resamples * n_packed * 8` bytes
    /// to the result; default `false`.
    pub sir_keep_samples: bool,
    /// Degrees of freedom for the Student-t SIR proposal distribution.
    /// Heavier tails than the normal improve ESS for parameters near boundaries
    /// (omega variances, constrained thetas). Default 5.0 follows Dosne (2017).
    /// Set to a large value (e.g. 100.0) to recover near-normal behaviour.
    pub sir_df: f64,
    // Importance-sampling options (consumed by the `Imp` chain stage; ignored
    // otherwise). By default `imp` is a Monte-Carlo EM **estimator** matching
    // NONMEM `METHOD=IMP`: the conditional mode + first-order variance are found
    // only on the first iteration, then the proposal is re-centered from the
    // previous iteration's importance-sample mean/covariance, and θ/Ω/σ are
    // updated from the importance-weighted posterior moments each iteration. Set
    // `imp_eval_only = true` (NONMEM `EONLY=1`) to instead evaluate
    // `−2 log L = −2 Σᵢ log ∫ p(yᵢ|η,θ)p(η|θ) dη` at the fixed input parameters
    // without updating them.
    /// Gauss–Hermite nodes **per random effect** for `method = laplace` and
    /// `method = focei` (#251). **Default 1.**
    ///
    /// `n_agq = 1` is the classical single-point method (Laplace under the exact anchor,
    /// FOCEI under the Gauss-Newton anchor). `n_agq > 1` is adaptive Gauss–Hermite
    /// quadrature: it refines the marginal toward the exact integral and is what makes the
    /// exact-anchor path unbiased on strongly non-Gaussian integrands. The tensor grid costs
    /// `n_agq^n_eta` likelihood evaluations per subject per outer iteration, so raise it
    /// deliberately — odd values are conventional (they keep a node exactly at the mode).
    /// Capped by [`crate::estimation::agq::MAX_AGQ_NODES`], with the *grid* additionally
    /// capped by [`crate::estimation::agq::MAX_AGQ_GRID`].
    pub n_agq: usize,
    /// Number of importance samples per subject. Default 1000. Recommended
    /// 2000–5000 for publication-quality MC SE (cost scales linearly).
    pub imp_samples: usize,
    /// Degrees of freedom for the Student-t proposal. Default 5.0 (heavy-tailed
    /// — robust to mild proposal misspecification). Must be ≥ 1. The token
    /// `normal` (parsed to `f64::INFINITY`) selects a multivariate-normal
    /// proposal.
    pub imp_proposal_df: f64,
    /// RNG seed for the IS sampling. `None` falls back to a fixed default so
    /// runs are reproducible across invocations.
    pub imp_seed: Option<u64>,
    /// Subjects with normalized effective sample size below this fraction
    /// (ESS / K) are flagged in the result. Default 0.1. Set to 0 to silence
    /// the flag entirely.
    pub imp_low_ess_threshold: f64,
    /// Number of MCEM iterations for the estimating `imp` path (ignored when
    /// `imp_eval_only`). Default 200.
    pub imp_iterations: usize,
    /// Number of terminal iterations whose parameters are averaged to form the
    /// reported estimate (Monte-Carlo variance reduction). Default 50. Ignored
    /// when `imp_eval_only`.
    pub imp_averaging: usize,
    /// When `true`, `imp` evaluates `−2 log L` at the fixed input parameters and
    /// does not estimate (NONMEM `IMP EONLY=1`); it must then be the terminal
    /// chain stage. When `false` (default), `imp` is an MCEM estimator
    /// (NONMEM `METHOD=IMP`).
    pub imp_eval_only: bool,
    // IMPMAP (Importance Sampling assisted by Mode A Posteriori) options,
    // consumed by the `Impmap` estimating stage. IMPMAP runs a Monte-Carlo EM
    // loop: each iteration re-centers a per-subject importance-sampling proposal
    // at the freshly-computed conditional mode (MAP) and first-order variance,
    // then updates θ/Ω/σ from the importance-weighted posterior moments.
    /// Number of MCEM iterations (M-step parameter updates). Default 200.
    pub impmap_iterations: usize,
    /// Importance samples drawn per subject per iteration (K). Default 300.
    /// Larger K reduces Monte-Carlo noise in the M-step at linear cost.
    pub impmap_samples: usize,
    /// Proposal degrees of freedom. `f64::INFINITY` selects a multivariate
    /// normal proposal (parsed from `normal`); a finite value selects a
    /// heavier-tailed Student-t. Default `4.0` (Student-t). A Gaussian proposal
    /// (`= normal`, NONMEM's IMPMAP default) has lighter tails than the posterior
    /// of weakly-identified parameters, so importance weights blow up in the tail
    /// and bias the M-step moments; the heavier-tailed t default avoids that.
    pub impmap_proposal_df: f64,
    /// RNG seed for the IMPMAP sampling. `None` falls back to a fixed default so
    /// runs are reproducible across invocations.
    pub impmap_seed: Option<u64>,
    /// Number of terminal iterations whose parameters are averaged to form the
    /// reported estimate (Monte-Carlo variance reduction). Default 50.
    pub impmap_averaging: usize,
    /// Subjects whose normalized effective sample size (ESS / K) falls below
    /// this fraction are flagged as poorly-sampled. Default 0.1.
    pub impmap_low_ess_threshold: f64,
    /// When `true`, IMPMAP collects per-iteration parameter values into
    /// `FitResult.impmap_trace` (analogous to NONMEM `.ext` output). Default `false`.
    pub impmap_trace: bool,
    /// Number of additional random starting points for per-subject MAP
    /// (analogous to NONMEM MCETA). 0 = single start (current behaviour).
    pub impmap_mceta: usize,
    /// Use Sobol quasi-random sequences for IS draws instead of pseudo-random.
    /// Only applies to MVN proposals (impmap_proposal_df = normal). Default false.
    pub impmap_sobol: bool,
    /// FREM only: Rao-Blackwellise the covariate ETAs (integrate them analytically,
    /// sample only the PK ETAs) in IMP/IMPMAP importance sampling. Default `true`
    /// — strongly recommended, since brute-force sampling of the near-singular
    /// covariate dimensions has very poor ESS. Set `false` only to diagnose the
    /// RB path against the full-dimensional sampler.
    pub frem_rao_blackwell: bool,
    /// Adaptive importance-sample count for IMP (NONMEM `AUTO`/`STDOBJ`). When
    /// `true` (the default), `imp_samples` is the *starting* count and is ramped
    /// up (×2 per iteration, capped at 10000) whenever the objective's Monte-Carlo
    /// standard deviation exceeds 1.0, so high-dimensional / FREM fits reach a
    /// low-noise objective automatically instead of carrying a sample-count-
    /// dependent M-step bias. Low-dimensional, well-sampled fits never trip the
    /// threshold, so there is no cost there. Set `false` to pin the sample count.
    pub imp_auto: bool,
    /// Adaptive importance-sample count for IMPMAP (NONMEM `AUTO`/`STDOBJ`). As
    /// [`FitOptions::imp_auto`] but ramps `impmap_samples`. Default `true`.
    pub impmap_auto: bool,
    /// Minimum ISCALE factor for adaptive IS proposal scaling (NONMEM ISCALE_MIN).
    /// The proposal covariance is multiplied by iscale² to improve IS efficiency.
    /// Set `iscale_min == iscale_max == 1.0` to disable. Default 0.1.
    pub iscale_min: f64,
    /// Maximum ISCALE factor for adaptive IS proposal scaling (NONMEM ISCALE_MAX).
    /// Default 10.0.
    pub iscale_max: f64,
    /// Defensive-mixture weight for the IMP/IMPMAP importance-sampling proposal
    /// (issue #528). Each subject draws an `imp_defensive_alpha` fraction of its
    /// samples from the prior `N(0, Ω)` instead of the mode-centred t-proposal,
    /// and every sample is scored under the resulting mixture density. Because
    /// the prior is guaranteed to cover the conditional posterior, this bounds
    /// each importance weight by `p(y|η)/alpha`, so no single sample from a
    /// weakly-identified subject (e.g. an analytical `[initial_conditions]`
    /// baseline whose V cancels in the amplitude) can hijack the importance-
    /// weighted M-step and walk θ to the bounds. (It bounds the weights, not the
    /// raw ESS — a sharp interior likelihood spike can still keep ESS low.)
    /// Applies to the estimating IMP/IMPMAP phases (including the FREM
    /// Rao-Blackwell path) and the eval-only IS marginal-likelihood pass. Must be
    /// in `[0, 1)`. Default `0.0` — the mixture is **opt-in**, so the default
    /// reproduces the pre-#528 single-proposal sampler and stays bit-comparable
    /// with NONMEM (which has no defensive mixture); set a small positive value
    /// (e.g. `0.1`) to enable the rescue for weakly-identified models. For an
    /// IMPMAP stage the option may also be written as `impmap_defensive_alpha`
    /// (an alias for this same field). Enabling the mixture disables Sobol QMC and
    /// raises the per-subject ESS floor by ≈`alpha` (so `imp_low_ess_threshold`
    /// flags fewer subjects).
    pub imp_defensive_alpha: f64,
    /// How LOQ-censored observations are handled.
    /// See [`BloqMethod`]. Defaults to `Drop` (backward-compatible: no effect
    /// when the data has no CENS column).
    pub bloq_method: BloqMethod,
    /// Number of Monte-Carlo replicates per subject used to compute the
    /// simulation-based NPDE/NPD diagnostics after the fit. `0` (default)
    /// disables the computation entirely — no `NPDE`/`NPD` columns are emitted.
    /// A typical value is `1000` (the `npde`-package default). Cost scales
    /// linearly with the replicate count. See [`crate::stats::npde`].
    pub npde_nsim: usize,
    /// RNG seed for the NPDE/NPD simulation. `None` falls back to a fixed
    /// default so the diagnostic is reproducible across invocations.
    pub npde_seed: Option<u64>,
    /// Maximum CG iterations for the Steihaug subproblem solver (trust-region only).
    /// `None` (default) uses a size-adaptive budget of `ceil(sqrt(n_params)).clamp(5, n_params)`,
    /// which is 5 for typical NLME problems (n_params ≈ 7–15) and grows with model size.
    /// `Some(n)` pins the budget to `n` — set to `50` to recover the previous fixed behaviour.
    pub steihaug_max_iters: Option<usize>,
    /// If true (default), use automatically detected mu-referencing to centre
    /// ETA starting points on the current population mean at each outer step.
    /// Set to false to disable for comparison purposes.
    pub mu_referencing: bool,
    /// Number of rayon worker threads used for the per-subject parallel loops
    /// (inner EBE search, SAEM MH steps, SIR weighting, likelihood reductions).
    /// `None` (default) leaves rayon's global pool alone, which means one
    /// worker per logical CPU. `Some(n)` runs the fit inside a scoped local
    /// pool of `n` threads — so the setting is per-call, not process-wide,
    /// and different fits can use different thread counts.
    pub threads: Option<usize>,
    /// Number of independent optimizations to run from perturbed starting values.
    /// `1` (default) is a single run — no behaviour change. When `> 1`, runs are
    /// launched in parallel via rayon; the result with the lowest OFV among
    /// converged runs is returned. Start 0 always uses the exact user initials;
    /// starts 1..n are log-space or additive perturbations of size `start_sigma`.
    /// Useful for models with local minima (nonlinear elimination, full-block omega,
    /// many covariates). Nested rayon parallelism (multi-start × per-subject) is
    /// safe — rayon's work-stealing pool handles it without oversubscription.
    pub n_starts: usize,
    /// Log-space standard deviation of the perturbation applied to initial theta
    /// values for starts 1..n_starts. Log-packed thetas are multiplied by
    /// `exp(N(0, start_sigma))`; identity-packed thetas (negative lower bound)
    /// are shifted by `start_sigma * N(0,1)`. Default `0.3` (≈ 30% CV).
    pub start_sigma: f64,
    /// RNG seed for the multi-start theta perturbations. Independent of
    /// `saem_seed` so that changing the SAEM seed for SAEM convergence does
    /// not silently alter which perturbed starts are tried for FOCE multi-start
    /// runs. Default `None` falls back to `42`.
    pub multi_start_seed: Option<u64>,
    /// Name of the column in the dataset that identifies the occasion for each row.
    /// When `Some`, `read_nonmem_csv` populates `Subject::occasions` / `dose_occasions`
    /// and the inner loop estimates per-occasion kappas alongside the BSV etas.
    /// Requires at least one `kappa` declaration in the model's `[parameters]` block.
    pub iov_column: Option<String>,
    /// Model-side rule for deriving the IOV occasion partition. Default
    /// [`IovOccasionRule::Column`] reproduces the historical behaviour (occasions
    /// come from `iov_column`). When set to `PerDose` or `TimeWindows`, `fit()`
    /// derives `Subject::occasions` / `dose_occasions` from each subject's timeline
    /// and — if `iov_column` is also set — overrides the column, emitting a warning.
    pub iov_occasion: IovOccasionRule,
    /// Optional cooperative cancellation token. When present and flipped by
    /// another thread, the outer/inner/SAEM/GN loops exit at the next safe
    /// point and `fit()` returns `Err("cancelled by user")`. Default `None`.
    pub cancel: Option<crate::cancel::CancelFlag>,
    /// Keys the user explicitly set, in the order they were applied. Populated
    /// by `parse_fit_options` / `apply_fit_option`. Used by `fit()` to warn
    /// when a key is set that the selected estimation method does not consume.
    pub user_set_keys: Vec<String>,
    /// Inner-loop gradient method. Default [`GradientMethod::Auto`] uses the
    /// exact analytic `Dual2` sensitivities whenever the model has an
    /// analytical PK path (`tv_fn` populated) and is in scope; otherwise falls
    /// back to FD. See [`GradientMethod`] for the full contract.
    pub gradient_method: GradientMethod,
    /// How often, in gradient evaluations, to re-solve each subject's inner
    /// EBE loop (η̂ and the FOCE Hessian) during the population gradient
    /// instead of holding it fixed.
    ///
    /// - `0` (default) — never reconverge on non-IOV models: use the cheap
    ///   fixed-EBE analytic gradient.
    /// - `1` — reconverge on every gradient evaluation.
    /// - `N` — reconverge on evals `0, N, 2N, …` and use the cheap fixed-EBE
    ///   gradient in between.
    ///
    /// The reconverged gradient captures the inner-solution response term that
    /// the fixed-EBE gradient omits — the term whose absence stalls SLSQP well
    /// above the derivative-free optimum on ill-conditioned non-IOV fits (see
    /// the `focei-slsqp-fixed-ebe-gradient-bias` note) — at ~5–6× the
    /// per-gradient cost (the inner loop runs once per perturbed component).
    /// A larger `N` amortizes that cost while still periodically correcting the
    /// search direction. IOV models (`n_kappa > 0`) always reconverge and
    /// ignore this setting.
    pub reconverge_gradient_interval: usize,
    /// When `true`, write a per-iteration optimizer trace CSV to a temp file
    /// and store its path in `FitResult::trace_path`. Default: `false`.
    pub optimizer_trace: bool,
    /// Apply an additional scaling layer on top of the existing log/Cholesky
    /// parameterization, dividing each transformed coordinate by its initial
    /// magnitude so they are O(1) when passed to the outer optimizer.
    ///
    /// **Default: `false`** (changed in issue #99). The scaling layer is *not*
    /// trajectory-transparent: although the OFV value is unchanged at any
    /// fixed point, the layer rescales the gradient the optimizer sees, and
    /// that gradient feeds the identity-Hessian overshoot cap
    /// (`cap_scaled_gradient`, on every SLSQP eval and the first L-BFGS eval), the
    /// quasi-Newton Hessian estimate, and the xtol/ftol termination — all of
    /// which act in the scaled coordinate system, so the *trajectory* and stop
    /// point differ. Because the scaling-enabled path only ever runs on
    /// log/Cholesky-packed coordinates (it auto-disables when any
    /// identity-packed theta is present), dividing by `|log value|` is
    /// counterproductive: a coordinate like `ln(V) = ln(20) ≈ 3` gets scale 3,
    /// so the optimizer's unit step becomes a 3-unit move in log space — an
    /// e³ ≈ 20× multiplicative jump in V. That large step both overshoots and,
    /// through the uniform gradient cap, starves the step in every other
    /// dimension (notably OMEGA), which then halts on xtol ~2.5 OFV units
    /// short of the minimum (issue #99; the earlier SAD_SCEN1 slowdown noted
    /// in `optimize_nlopt` is the same effect). The `false` default reproduces
    /// the well-tested pre-scaling-layer behaviour; `true` is left as an
    /// opt-in for experimentation.
    pub scale_params: bool,
    /// Parameter-scaling strategy for the outer optimizer. When non-`None` this
    /// supersedes [`scale_params`]: `Rescale2` (nlmixr2-style bound-half-width
    /// normalisation) is the recommended setting for gradient-based optimizers
    /// and substantially improves cold-start convergence (see
    /// [`ParameterScaling`]). **Default: `Auto`** — applies `Rescale2` to the
    /// gradient-based optimizers that benefit (`Bfgs`/`Lbfgs`/`NloptLbfgs`/`Slsqp`)
    /// and leaves the derivative-free default `Bobyqa` unscaled (where `Rescale2`
    /// distorts its trust-region model). Set via `[fit_options]` key
    /// `parameter_scaling = none|abs|rescale2` to override.
    pub parameter_scaling: ParameterScaling,
    /// Fraction of subjects allowed to have unconverged EBEs before the outer
    /// optimizer rejects the current parameter step (returns OFV = ∞).  Set to
    /// `1.0` to disable the guard (old behaviour).  Default: `0.1`.
    pub max_unconverged_frac: f64,
    /// Minimum number of observations a subject must have for its EBE to count
    /// toward `max_unconverged_frac`.  Subjects below this threshold are
    /// excluded from the convergence fraction but still run normally.
    /// Default: `2`.
    pub min_obs_for_convergence_check: u32,
    /// Enable the outer-loop stagnation guard. When `true` (default), the
    /// NLopt-based outer optimizers short-circuit once recent evals show
    /// no OFV improvement above 1e-3 over a window of `3*(n+1).max(50)`
    /// evals — letting SLSQP / L-BFGS terminate in microseconds via their
    /// own xtol/ftol instead of burning through the remaining maxeval
    /// budget at full inner-loop cost. Set to `false` to disable when
    /// you want the optimizer to run to its natural termination criterion
    /// (or to `outer_maxiter`), e.g. for debugging or for problems with
    /// very slow-but-real OFV improvements below the stagnation threshold.
    pub stagnation_guard: bool,
    /// When an inner EBE BFGS solve fails and falls back to Nelder–Mead, warm-start
    /// the simplex from the BFGS partial η̂ instead of cold-starting from the prior
    /// mode η=0. A weakly-identified η (e.g. an unidentifiable peripheral volume)
    /// drives BFGS onto the steep prior slope far from 0, and NM slides down it to
    /// the mode in far fewer iterations than refining from 0 in the flat basin —
    /// substantially fewer prediction walks on fallback-heavy fits (≈1.7× on a
    /// 2-cpt unidentifiable-V2 benchmark).
    ///
    /// **Opt-in (default `false`).** Warm-starting moves the fallback subjects'
    /// EBEs, which perturbs the outer optimiser's trajectory: harmless for the
    /// derivative-free BOBYQA default, but it can derail a gradient-based outer
    /// optimiser (e.g. MMA) into a worse basin on some models. Enable it only
    /// after validating the OFV/estimates on your model + outer optimiser; leave
    /// it `false` for the historical (cold-restart) behaviour.
    pub ebe_warm_start: bool,
    /// When `Some(_)`, call [`crate::suggest_start::inits_from_nca`] with the
    /// selected strategy before the optimizer loop to derive NCA-based starting
    /// values from the data. Useful when the model file's defaults are far from
    /// the truth. `None` (the default) disables it.
    pub inits_from_nca: Option<crate::suggest_start::NcaInit>,
    /// Expression strings for `[data_selection] ignore = ...` / `ignore_subjects`.
    /// Each string may contain `&&`-joined sub-expressions (all must hold).
    /// A record is excluded when any clause evaluates to `true`.
    /// Stored verbatim for logging; compiled to `FilterClause` at read time.
    pub ignore_exprs: Vec<String>,
    /// Expression strings for `[data_selection] accept = ...`.
    /// A record is excluded when any clause evaluates to `false`.
    pub accept_exprs: Vec<String>,
    /// Subject IDs to exclude wholesale (syntactic sugar for `ignore = ID == X`).
    /// Compared as strings against `Subject::id`.
    pub ignore_subjects: Vec<String>,
    /// FREM prediction map: `"TV_WT/ETA_WT_FREM:100, TV_AGE/ETA_AGE_FREM:200"`.
    /// Maps theta/eta pairs to FREMTYPE values.
    pub frem_predictions: Option<String>,
    /// FREM covariate sigma name (e.g. "EPSCOV").
    pub frem_sigma: Option<String>,
    /// Enable periodic checkpointing so an interrupted run can be resumed from
    /// its last saved state (#755). When `true` *and* `checkpoint_path` is set,
    /// the current population estimates (θ/Ω/Σ), stage index, iteration and OFV
    /// are written to `checkpoint_path` no more often than every
    /// `checkpoint_interval_secs` seconds, and the file is removed on successful
    /// completion. A run shorter than the interval never writes a checkpoint.
    /// Settable from `[fit_options]`; default `true`.
    pub checkpoint: bool,
    /// Minimum wall-clock seconds between checkpoint writes. Settable from
    /// `[fit_options]`; default `300` (5 minutes).
    pub checkpoint_interval_secs: u64,
    /// Where to read/write the `.tmp` checkpoint. Set by the file runner from the
    /// model name (`{stem}.tmp`); `None` disables checkpointing regardless of
    /// `checkpoint`. Runtime-only — not settable from `[fit_options]`.
    pub checkpoint_path: Option<String>,
    /// SHA-256 of the model file at fit time; stored in the checkpoint and
    /// checked on resume so an edited model never silently continues from stale
    /// state. Runtime-only; set by the file runner alongside `checkpoint_path`.
    pub checkpoint_model_hash: Option<String>,
    /// SHA-256 of the data CSV at fit time; checked on resume like
    /// `checkpoint_model_hash`. Runtime-only.
    pub checkpoint_data_hash: Option<String>,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            method: EstimationMethod::FoceI,
            methods: Vec::new(),
            outer_maxiter: 500,
            outer_gtol: 1e-6,
            // BOBYQA trust-region stop tolerances (NLopt xtol_rel/ftol_rel).
            // `outer_ftol = None` auto-selects 1e-8 for pure-TTE (exact objective)
            // and 1e-6 elsewhere (#469): the historical 1e-6 stopped the derivative-
            // free outer optimizer short on the near-flat frailty-ω² ridge (read
            // 0.204 vs the NONMEM/nlmixr2 0.175 consensus; at 1e-8 it lands 0.176),
            // but on a *noisy* objective (ODE/FD-inner) 1e-8 is unreachable and
            // BOBYQA grinds to maxeval, so non-TTE fits keep 1e-6. Override either
            // via `[fit_options] outer_xtol`/`outer_ftol`.
            outer_xtol: 1e-4,
            outer_ftol: None,
            inner_maxiter: 200,
            // 1e-5, not the looser 1e-4 that an earlier comment justified as
            // matching "NONMEM's ~3-SIGDIGITS inner loop" — that conflated the
            // *outer* control (NSIG/SIGDIGITS, default 3) with the *inner*
            // conditional precision (SIGL, default ~10 significant digits), which
            // NONMEM runs far tighter. A loose inner tolerance leaves residual
            // noise in each subject's EBE solution, which propagates into the
            // marginal OFV the *outer* optimizer sees. On models with a noisy or
            // flat marginal surface (FD-inner FOCE such as LTBS) that noise made
            // the derivative-free BOBYQA outer optimizer false-converge a few OFV
            // units above the true minimum. Tightening to 1e-5 removes enough of
            // that noise to reach NONMEM's minimum (LTBS now matches to <0.001
            // OFV), at ~1.5x the per-fit cost. Note tighter is not uniformly
            // better: at 1e-6 some ill-conditioned fits (3x3 block-Ω,
            // KA=KE+exp(...) coupling) over-converge the inner Hessian and BOBYQA
            // navigates into a worse basin — 1e-5 sits below the LTBS noise floor
            // while staying clear of that pathology. It does change the converged
            // point versus 1e-4; the previous "no measurable OFV change" claim
            // only held for well-conditioned fits. Override via `inner_tol = ...`
            // in `[fit_options]` (loosen for speed; tighten with care).
            inner_tol: 1e-5,
            // On by default (one Ω-scaled restart pass): guards against inner-EBE
            // local minima on multimodal individual objectives at ≈0 cost, since
            // it fires only for reset / TV-cov subjects on a cold start and only
            // *accepts* a strictly-lower mode. Unimodal subjects stay bit-identical;
            // set `inner_restarts = 0` to disable. See `FitOptions::inner_restarts`.
            inner_restarts: 1,
            cov_inner_tol: None,
            // ODE solver tolerances: match OdeSolverOptions::default() so the
            // engine default is unchanged. Opt into tighter accuracy per model
            // via `[fit_options] ode_reltol = ...` (see FitOptions::ode_reltol).
            ode_reltol: 1e-4,
            ode_abstol: 1e-6,
            ode_max_steps: 10_000,
            ode_method: crate::ode::OdeMethod::Auto,
            ode_stiff_abort_after: None,
            ode_auto_switch: true,
            run_covariance_step: true,
            fd_hessian_step: 1e-2,
            covariance_fallback: CovarianceFallback::None,
            covariance_method: CovarianceMethod::Hessian,
            covariance_method_set: false,
            analytic_cov_hessian: true,
            interaction: true,
            verbose: true,
            // `Auto` resolves per model (see `Optimizer::resolve_auto`): the
            // gradient-based NLopt L-BFGS when the exact analytic FOCE/FOCEI
            // gradient is available, and the derivative-free BOBYQA otherwise.
            // BOBYQA is the fallback because the fixed-EBE FD gradient that the
            // gradient-based optimizers rely on is biased on ill-conditioned fits
            // (ODE/PD models, sparse data, Hill-ridge identifiability), where
            // SLSQP can declare convergence hundreds of OFV units above the true
            // minimum. See `docs/estimation/optimizers.qmd` for the benchmarking
            // (#490) behind these defaults and how to pin an optimizer explicitly.
            optimizer: Optimizer::Auto,
            inner_optimizer: InnerOptimizer::Auto,
            lbfgs_memory: 5,
            global_search: false,
            global_maxeval: 0,
            saem_n_exploration: 150,
            saem_n_convergence: 250,
            saem_n_mh_steps: 20,
            saem_adapt_interval: 50,
            saem_mstep_damping: None,
            saem_omega_burnin: 20,
            saem_seed: None,
            saem_conddist: false,
            saem_conddist_nsamp: 200,
            saem_conddist_burnin: 20,
            saem_conddist_keep_samples: false,
            saem_n_leapfrog: 0,
            bayes_warmup: 1000,
            bayes_iters: 1000,
            bayes_chains: 4,
            bayes_thin: 1,
            bayes_seed: None,
            gn_lambda: 0.01,
            sir: false,
            sir_samples: 1000,
            sir_resamples: 250,
            sir_seed: None,
            sir_keep_samples: false,
            sir_df: 5.0,
            n_agq: 1,
            imp_samples: 1000,
            imp_proposal_df: 5.0,
            imp_seed: None,
            imp_low_ess_threshold: 0.1,
            imp_iterations: 200,
            imp_averaging: 50,
            imp_eval_only: false,
            impmap_iterations: 200,
            impmap_samples: 300,
            impmap_proposal_df: 4.0,
            impmap_seed: None,
            impmap_averaging: 50,
            impmap_low_ess_threshold: 0.1,
            impmap_trace: false,
            impmap_mceta: 0,
            impmap_sobol: false,
            frem_rao_blackwell: true,
            imp_auto: true,
            impmap_auto: true,
            iscale_min: 0.1,
            iscale_max: 10.0,
            imp_defensive_alpha: 0.0,
            bloq_method: BloqMethod::Drop,
            npde_nsim: 0,
            npde_seed: None,
            steihaug_max_iters: None,
            mu_referencing: true,
            threads: None,
            n_starts: 1,
            start_sigma: 0.3,
            multi_start_seed: None,
            iov_column: None,
            iov_occasion: IovOccasionRule::Column,
            cancel: None,
            user_set_keys: Vec::new(),
            gradient_method: GradientMethod::default(),
            reconverge_gradient_interval: 0,
            optimizer_trace: false,
            scale_params: false,
            parameter_scaling: ParameterScaling::Auto,
            max_unconverged_frac: 0.1,
            min_obs_for_convergence_check: 2,
            stagnation_guard: true,
            ebe_warm_start: false,
            inits_from_nca: None,
            ignore_exprs: Vec::new(),
            accept_exprs: Vec::new(),
            ignore_subjects: Vec::new(),
            frem_predictions: None,
            frem_sigma: None,
            checkpoint: true,
            checkpoint_interval_secs: 300,
            checkpoint_path: None,
            checkpoint_model_hash: None,
            checkpoint_data_hash: None,
        }
    }
}

/// LOQ censoring handling.
///
/// `Drop` — CENS rows are kept as ordinary observations (no special treatment). If
/// the dataset has no CENS column, every row is treated as quantified and this is
/// equivalent to the pre-M3 behavior.
///
/// `M3` — Beal's M3 method: each censored observation contributes a normal-tail
/// probability instead of a Gaussian residual term. LLOQ is read from DV on
/// CENS=1 rows; ULOQ is read from DV on CENS=-1 rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BloqMethod {
    Drop,
    M3,
}

impl BloqMethod {
    pub fn label(self) -> &'static str {
        match self {
            BloqMethod::Drop => "drop",
            BloqMethod::M3 => "m3",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Optimizer {
    /// Pick the outer optimizer automatically from the model (the default).
    /// Resolves to [`Optimizer::NloptLbfgs`] when the exact analytic FOCE/FOCEI
    /// gradient is available (the model is in the sensitivity provider's scope
    /// and the user did not force `gradient_method = fd`), and to
    /// [`Optimizer::Bobyqa`] otherwise (ODE/PD models, LTBS, SDE, or
    /// `gradient_method = fd`, where the outer loop must fall back to finite
    /// differences). Benchmarking across ~10 real FOCEI datasets (#490) found
    /// NLopt L-BFGS fastest-to-optimum on every analytic-gradient problem, while
    /// BOBYQA was both fastest and most reliable when only finite differences
    /// are available. See [`Optimizer::resolve_auto`].
    Auto,
    Bfgs,
    Lbfgs,
    /// NLopt LD_SLSQP — Sequential Least Squares Programming. Gradient-based;
    /// fast per iteration on smooth, well-conditioned analytical PK models where
    /// the fixed-EBE finite-difference gradient is a faithful proxy for the true
    /// gradient. On ill-conditioned fits (ODE/PD models, sparse data, Hill-ridge
    /// identifiability) the fixed-EBE bias can drive SLSQP to declare convergence
    /// hundreds of OFV units above the true minimum — pair with
    /// `reconverge_gradient_interval = 1` if it stalls, or switch to `Bobyqa`
    /// (the default; see `FitOptions::default`).
    Slsqp,
    /// NLopt LD_LBFGS
    NloptLbfgs,
    /// NLopt LD_MMA — Method of Moving Asymptotes
    Mma,
    /// NLopt LN_BOBYQA — derivative-free quadratic interpolation, default outer
    /// optimizer. Re-evaluates the FOCE objective (and the inner EBE loop) at
    /// every trial point, so it never sees the fixed-EBE gradient bias that can
    /// stall gradient-based optimizers; consistently reaches a lower OFV than
    /// SLSQP on ODE/PD models, sparse data, and Hill-ridge problems. Needs more
    /// outer evaluations than SLSQP to triangulate a quadratic from scratch, but
    /// each evaluation is cheap (no FD gradient sweep). See
    /// `docs/estimation/optimizers.qmd` for the cefepime and Emax PKPD
    /// validations behind the default choice.
    Bobyqa,
    /// Newton trust-region with Steihaug CG subproblem (via argmin)
    TrustRegion,
}

/// Inner-loop (EBE) optimizer, set via `[fit_options] inner_optimizer`. Lets the
/// user pin the per-subject solver explicitly instead of the size-based default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InnerOptimizer {
    /// Size-based selection (the historical behaviour): dense BFGS below
    /// [`crate::estimation::inner_optimizer::INNER_LBFGS_MIN_DIM`], limited-memory
    /// L-BFGS at/above it. Dense BFGS Newton-converges in a few steps and wins for
    /// the typical small `n_eta`; L-BFGS wins for high-dimensional inner problems
    /// (large IOV). Nelder–Mead remains the on-failure fallback in every mode.
    #[default]
    Auto,
    /// Always dense BFGS, regardless of inner dimension.
    Bfgs,
    /// Always limited-memory L-BFGS, regardless of inner dimension.
    Lbfgs,
    /// Always Nelder–Mead (derivative-free).
    NelderMead,
}

/// Parameter-scaling strategy for the outer optimizer. Maps the packed
/// parameter vector into a better-conditioned space before the optimizer sees
/// it (and maps gradients/bounds back). Distinct from the legacy
/// [`FitOptions::scale_params`] bool, which it supersedes when set to a
/// non-`None` value.
///
/// On `two_cpt_oral_cov` (FOCEI, mu-referencing), `Rescale2` + `bfgs` reaches
/// OFV −1198.97 from a cold start — matching nlmixr2 (−1199.24) — where the
/// unscaled gradient-based optimizers stall near −1152/−1192. This mirrors
/// nlmixr2's finding that parameter scaling, not gradient exactness, is the
/// lever for cold-start robustness of *gradient-based* optimizers.
///
/// Crucially, `Rescale2` is **harmful to the derivative-free default `Bobyqa`**
/// (e.g. it drops `emax_pkpd` from OFV −36.76 to −13.51 and `three_cpt_iv` from
/// −730.6 to −715.9): rescaling the trust region of a gradient-free optimizer
/// distorts its quadratic model. Hence the default is [`Auto`](Self::Auto),
/// which applies `Rescale2` only to the gradient-based optimizers that benefit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum ParameterScaling {
    /// **Default.** Apply `Rescale2` for the gradient-based optimizers that
    /// benefit from it (`Bfgs`, `Lbfgs`, `NloptLbfgs`, `Slsqp`) and no scaling
    /// otherwise — so the derivative-free `Bobyqa` default (where `Rescale2` is
    /// harmful) and `Mma`/`TrustRegion` are left unscaled, with the legacy
    /// `scale_params` / IOV-auto-enable still applying in that unscaled branch.
    /// `Slsqp` is scaled because the bound-half-width rescaling fixes its
    /// cold-start convergence on IOV models (#335).
    #[default]
    Auto,
    /// No scaling: fall back to the legacy `scale_params` bool (and the IOV+SLSQP
    /// auto-enable). Preserves the pre-`Auto` unscaled behaviour for any optimizer.
    None,
    /// Normalise each coordinate by `|packed value|` (the legacy `compute_scale`
    /// strategy). O(1) for log-packed thetas; 1.0 fallback near zero.
    Abs,
    /// nlmixr2-style: normalise each coordinate by the half-width of its bound
    /// range, mapping it toward `(−1, 1)`. The recommended scaling for
    /// gradient-based optimizers (`bfgs`/`lbfgs`); harmful to `Bobyqa`.
    Rescale2,
}

impl Optimizer {
    pub fn label(self) -> &'static str {
        // NB: the `bfgs`/`lbfgs` labels no longer round-trip through the parser —
        // `apply_fit_option` resolves both keywords to `NloptLbfgs` (see #483).
        // The labels stay as the variant's own name for truthful run provenance;
        // only a Rust caller that constructs `Optimizer::Bfgs`/`Lbfgs` directly can
        // produce them, and those variants are slated for removal.
        match self {
            Optimizer::Auto => "auto",
            Optimizer::Bfgs => "bfgs",
            Optimizer::Lbfgs => "lbfgs",
            Optimizer::Slsqp => "slsqp",
            Optimizer::NloptLbfgs => "nlopt_lbfgs",
            Optimizer::Mma => "mma",
            Optimizer::Bobyqa => "bobyqa",
            Optimizer::TrustRegion => "trust_region",
        }
    }

    /// Resolve [`Optimizer::Auto`] to a concrete optimizer for `model`; every
    /// other variant is returned unchanged (idempotent). `auto` picks
    /// [`Optimizer::NloptLbfgs`] when the exact analytic FOCE/FOCEI gradient is
    /// available — the model is in the sensitivity provider's scope and the user
    /// did not force `gradient_method = fd` — and [`Optimizer::Bobyqa`]
    /// otherwise. This mirrors the analytic-vs-FD predicate the outer loop uses
    /// (see `build_info::gradient_method_outer`), so `auto` lands on the
    /// gradient-based optimizer exactly when there is an exact gradient to feed
    /// it (#490).
    ///
    /// `interaction` is `options.interaction` (`true` for `method = focei`,
    /// `false` for plain `method = foce`) — a custom residual-magnitude model's
    /// direct-θ channel is FOCEI-only (#486 review), so a magnitude-active model
    /// under plain FOCE must still resolve to `Bobyqa` even though
    /// `analytic_outer_gradient_available` (interaction-agnostic) says the model
    /// is in scope.
    pub fn resolve_auto(self, model: &CompiledModel, interaction: bool) -> Optimizer {
        if self != Optimizer::Auto {
            return self;
        }
        // #1064: BOBYQA builds a quadratic interpolation model of the whole
        // parameter space. That is the right trade at the handful-of-parameters
        // scale every other model in this codebase lives at, and hopeless at the
        // several-hundred an unstructured-placebo MBMA model reaches — the
        // interpolation set alone is O(n²) points before the first useful step.
        // Above the threshold `auto` takes the gradient-based optimizer even
        // though the gradient is finite-difference, because O(n) FD passes beat
        // O(n²) interpolation.
        if model.free_packed_dim() > BOBYQA_MAX_DIM {
            return Optimizer::NloptLbfgs;
        }
        // Use the single shared predicate so `auto` can never disagree with the
        // outer loop's actual gradient dispatch (#490 review): resolving to a
        // gradient-based optimizer while the loop ran FD would feed it a noisy
        // gradient.
        if crate::sens::provider::analytic_outer_gradient_for_interaction(model, interaction) {
            Optimizer::NloptLbfgs
        } else {
            Optimizer::Bobyqa
        }
    }
}

/// Free packed dimension above which [`Optimizer::Auto`] refuses BOBYQA (#1064).
///
/// BOBYQA's interpolation set grows quadratically in the parameter count, so a
/// derivative-free quadratic model stops being cheaper than a finite-difference
/// gradient well before the several-hundred θ an unstructured-placebo MBMA model
/// declares. 64 is the point past which the derivative-free model, rather than
/// the objective, dominates the wall clock.
pub const BOBYQA_MAX_DIM: usize = 64;

/// Free packed dimension above which a *defaulted* `covariance_method = r`
/// routes to the cross-product (#1064).
///
/// The `R` matrix is a finite-difference Hessian of the objective that
/// reconverges every subject's EBEs at each of `n(n+1)/2` stencil points. At
/// n = 100 that is ~5,000 reconverged population objectives; at n = 800 it is
/// ~320,000, which will not finish. The cross-product `S` needs one pass.
pub const COV_HESSIAN_MAX_DIM: usize = 100;

/// Estimation method
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstimationMethod {
    Foce,
    FoceI,
    FoceGn,
    FoceGnHybrid,
    Saem,
    /// Importance Sampling (NONMEM `METHOD=IMP`). By **default an estimator**: a
    /// Monte-Carlo EM loop that finds each subject's conditional mode and
    /// first-order variance only on the first iteration, then re-centers the
    /// importance-sampling proposal from the previous iteration's
    /// importance-sample mean/covariance, updating θ/Ω/σ from the
    /// importance-weighted posterior moments each iteration. Reports the IS
    /// `−2 log L` on `FitResult.importance_sampling` and a Laplace OFV on `ofv`.
    ///
    /// With `imp_eval_only = true` (NONMEM `IMP EONLY=1`) it instead *evaluates*
    /// `−2 log L_IS` at the fixed input parameters without updating them; in that
    /// mode it must be the terminal chain stage (it consumes the prior stage's
    /// params + EBEs + per-subject Hessians, or evaluates at the initial
    /// parameters when standalone). Contrast [`Impmap`](Self::Impmap), which
    /// re-evaluates the mode/variance *every* iteration (more robust, costlier).
    Imp,
    /// Importance Sampling assisted by Mode A Posteriori (NONMEM `METHOD=IMPMAP`).
    /// Like estimating [`Imp`](Self::Imp) this *is* an estimator, but its E-step
    /// re-evaluates each subject's conditional mode and first-order variance
    /// *every* iteration (as in FOCE/ITS) to center a multivariate-normal
    /// importance-sampling proposal — more robust than `Imp` on high-dimensional,
    /// rich-data problems — and its M-step updates θ/Ω/σ from the
    /// importance-weighted posterior moments.
    Impmap,
    /// Full MCMC Bayesian estimation (Path A — Gibbs-within-HMC, NONMEM
    /// `METHOD=BAYES` parity). Draws from the joint posterior
    /// `p(θ, Ω, Σ, {ηᵢ} | y)` by alternating a per-subject η block (reusing the
    /// SAEM HMC / MH kernel) with conjugate population draws (Ω: inverse-Wishart,
    /// σ²: inverse-gamma, mu-referenced θ: normal). Reports posterior summaries +
    /// convergence diagnostics on `FitResult.bayes`, not a point estimate.
    Bayes,
    /// The **Laplace approximation** with the *exact* Hessian — NONMEM `$EST METHOD=1
    /// LAPLACIAN` — generalised to adaptive Gauss–Hermite quadrature by the `n_agq` argument
    /// (#251).
    ///
    /// The estimator evaluates the **exact** conditional likelihood on a Gauss–Hermite grid
    /// of `n_agq` nodes per random effect, laid around each subject's EBE mode with the exact
    /// posterior Hessian as the scaling. At **`n_agq = 1`** the one-point rule sits at the
    /// mode with weight `√π` and the sum collapses, term for term, to
    /// `(2π)^(d/2)·|H|^(−½)·exp(l(η̂))` — Laplace exactly. At **`n_agq > 1`** it is adaptive
    /// Gauss–Hermite quadrature: the marginal improves toward the exact integral, and because
    /// it integrates the model's real likelihood rather than a Gaussian-residual surrogate it
    /// is the only method here that handles non-Gaussian endpoints (TTE, categorical) without
    /// approximation. Deterministic, so unlike [`Saem`](Self::Saem) / [`Imp`](Self::Imp) its
    /// OFV carries no Monte-Carlo noise. Cost is `n_agq^n_eta` likelihood evaluations per
    /// subject per iteration; see [`crate::estimation::agq::MAX_AGQ_GRID`].
    ///
    /// **The Hessian anchor is what distinguishes it from [`FoceI`](Self::FoceI)** — which is
    /// the *same* quadrature machinery with the *Gauss-Newton* Hessian `CᵀC + Ω⁻¹` (dropping
    /// `∂²f/∂η²`) as the scaling. `Laplace` uses the exact Hessian, so at one node it carries
    /// the curvature of the η-dependent residual variance the Gauss-Newton form discards —
    /// which is why it reproduces NONMEM's LAPLACIAN (six significant figures on warfarin)
    /// where FOCEI lands on a different value. See [`FitOptions::hessian_anchor`].
    ///
    /// (There is no separate `Agq` method: adaptive quadrature *is* `Laplace` with
    /// `n_agq > 1`. The old `method = agq` token is rejected by the parser with a note.)
    Laplace,
}

/// Which Hessian scales the Gauss–Hermite grid (and enters the `½log|H|` term).
///
/// The quadrature identity holds for any positive-definite scaling: the anchor changes the
/// proposal — and, at one node, the whole objective — but not the `n → ∞` limit. So a single
/// machinery serves both estimator families, parameterised by this choice:
/// [`Laplace`](EstimationMethod::Laplace) → [`Exact`](Self::Exact),
/// [`FoceI`](EstimationMethod::FoceI) → [`GaussNewton`](Self::GaussNewton).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HessianAnchor {
    /// Exact conditional Hessian `∂²nll/∂b²`. `Laplace` / adaptive GH quadrature.
    Exact,
    /// Gauss-Newton (Almquist) Hessian `H̃ = Ω⁻¹ + Σ pⱼaⱼaⱼᵀ`. FOCEI and its quadrature
    /// refinement. Always positive-definite and needs only first `f`-derivatives, so its
    /// `½log|H̃|` gradient is fully analytic.
    GaussNewton,
}

impl EstimationMethod {
    pub fn label(self) -> &'static str {
        match self {
            EstimationMethod::Foce => "FOCE",
            EstimationMethod::FoceI => "FOCEI",
            EstimationMethod::FoceGn => "FOCE-GN",
            EstimationMethod::FoceGnHybrid => "FOCE-GN-Hybrid",
            EstimationMethod::Saem => "SAEM",
            EstimationMethod::Imp => "IMP",
            EstimationMethod::Impmap => "IMPMAP",
            EstimationMethod::Bayes => "BAYES",
            EstimationMethod::Laplace => "LAPLACE",
        }
    }
}

impl FitOptions {
    /// Returns the sequence of methods to execute. If `methods` is non-empty it
    /// is returned as-is; otherwise a single-element chain wrapping `method`.
    pub fn method_chain(&self) -> Vec<EstimationMethod> {
        if self.methods.is_empty() {
            vec![self.method]
        } else {
            self.methods.clone()
        }
    }

    /// The AGQ node count when this (stage's) method is AGQ, else `None`.
    ///
    /// The **single predicate** every AGQ-aware branch consults — the population objective
    /// ([`crate::estimation::outer_optimizer::pop_nll_opts`]), the outer-gradient dispatch,
    /// and the optimizer/report classification. Keeping it in one place is what stops the
    /// objective and the gradient from disagreeing about which method is running: an
    /// outer loop minimising the AGQ objective while fed the analytic *FOCE* gradient
    /// would converge, silently, to the wrong parameters.
    ///
    /// Reads `self.method` (the per-stage method — `api::fit_inner` rewrites it for each
    /// stage of a chain), so it is correct inside a chained fit too.
    pub fn agq_nodes(&self) -> Option<usize> {
        match self.method {
            // Laplace is the exact-anchor quadrature: `n_agq = 1` is Laplace, `> 1` is
            // adaptive Gauss–Hermite quadrature. Both route through the same objective, the
            // same analytic gradient, and the same covariance stencil.
            EstimationMethod::Laplace => Some(self.n_agq.max(1)),
            // FOCEI with `n_agq > 1` is the Gauss-Newton-anchored quadrature refinement, live
            // for every in-analytic-scope model; `n_agq = 1` (the default) is plain FOCEI and
            // takes its own path (`None`). `check_model_options` rejects `focei, n_agq > 1`
            // only *outside* the analytic sensitivity scope (`E_FOCEI_NAGQ_UNSUPPORTED`).
            EstimationMethod::FoceI if self.n_agq > 1 => Some(self.n_agq),
            _ => None,
        }
    }

    /// Which Hessian scales the quadrature grid for this (stage's) method. Only meaningful
    /// when [`agq_nodes`](Self::agq_nodes) is `Some`: `Laplace` anchors on the exact
    /// conditional Hessian, `FoceI` on the Gauss-Newton `H̃`. See [`HessianAnchor`].
    pub fn hessian_anchor(&self) -> HessianAnchor {
        match self.method {
            EstimationMethod::FoceI => HessianAnchor::GaussNewton,
            _ => HessianAnchor::Exact,
        }
    }

    /// Covariance-step inner EBE-reconvergence tolerance that **closed-form LTBS**
    /// models tighten to when `cov_inner_tol` is unset. The
    /// covariance R-matrix reconverges EBEs under the `g = ln(f)` wrap, which
    /// amplifies a loosely-converged EBE into the Hessian and inflates the standard
    /// errors (empirically ~65% on warfarin θ SEs at the `1e-5` default; correct at
    /// `≤ 1e-8`). This is applied **only for the closed-form LTBS** path that actually
    /// runs the analytic `ln(f)` inner gradient (including under IOV as of #486 — see
    /// [`CompiledModel::uses_closed_form_ltbs_inner`]): blanket-tightening the covariance
    /// step over-converges some ill-conditioned inner Hessians into an indefinite
    /// covariance, and ODE-LTBS shares `solve_ode_g` so it has no such gap. Set
    /// `cov_inner_tol` explicitly to opt any other model in (e.g. the #654 M3 + IOV case,
    /// which is not LTBS).
    pub const LTBS_COV_INNER_TOL: f64 = 1e-8;

    /// Effective inner EBE-reconvergence tolerance for the covariance step: an
    /// explicit `cov_inner_tol` when set; otherwise the fit's `inner_tol`, tightened
    /// to at most [`LTBS_COV_INNER_TOL`](Self::LTBS_COV_INNER_TOL) only when
    /// `ltbs_closed_form` (the closed-form LTBS path, IOV included — see
    /// [`CompiledModel::uses_closed_form_ltbs_inner`]). Every other model keeps
    /// `inner_tol`, so its SEs are byte-identical.
    pub fn effective_cov_inner_tol(&self, ltbs_closed_form: bool) -> f64 {
        self.cov_inner_tol.unwrap_or_else(|| {
            if ltbs_closed_form {
                self.inner_tol.min(Self::LTBS_COV_INNER_TOL)
            } else {
                self.inner_tol
            }
        })
    }

    /// Ceiling the fit's inner EBE tolerance is tightened to for **closed-form LTBS**
    /// models. That path's analytic `g = ln(f)` inner gradient makes
    /// the marginal surface slightly noisier (the ~1e-9 provider-vs-
    /// `compute_predictions` gap), so at the loose default the outer optimiser lands
    /// nondeterministically on flat Ω directions and the standard errors of
    /// weakly-identified variances become process-dependent. Converging the fit's
    /// inner loop to at least `1e-6` pins the flat directions; the covariance step
    /// then reconverges tighter still ([`effective_cov_inner_tol`](Self::effective_cov_inner_tol)).
    /// ODE-LTBS (shares `solve_ode_g`, no gap) does not need this and is left at
    /// `inner_tol`. LTBS + IOV *is* included as of #486, when its inner loop stopped
    /// routing to FD.
    pub const LTBS_FIT_INNER_TOL: f64 = 1e-6;

    /// Effective **fit** inner EBE tolerance for this model: the fit's `inner_tol`,
    /// tightened to at most [`LTBS_FIT_INNER_TOL`](Self::LTBS_FIT_INNER_TOL) for the
    /// closed-form LTBS path (`ltbs_closed_form`, IOV included) **unless the user set
    /// `inner_tol` explicitly** — an explicit tolerance (e.g. the documented
    /// accuracy-for-speed `inner_tol = 1e-4`) is always honoured. A fit already
    /// tighter than the ceiling keeps its own value; every other model is unaffected.
    pub fn effective_inner_tol(&self, ltbs_closed_form: bool) -> f64 {
        let user_set_inner_tol = self.user_set_keys.iter().any(|k| k == "inner_tol");
        if ltbs_closed_form && !user_set_inner_tol {
            self.inner_tol.min(Self::LTBS_FIT_INNER_TOL)
        } else {
            self.inner_tol
        }
    }

    /// Check `user_set_keys` against the selected method chain. Returns one
    /// warning per key that isn't consumed by any method in the chain, listing
    /// the method-specific keys that *are* applicable so the user can correct
    /// the mistake. Framework-level keys (covariance/verbose/sir/bloq/threads/
    /// mu_referencing) are omitted from the suggestion list — they apply to
    /// every method and are exposed as top-level arguments in the wrappers.
    pub fn unsupported_keys_warnings(&self) -> Vec<String> {
        if self.user_set_keys.is_empty() {
            return Vec::new();
        }
        let chain = self.method_chain();
        // Applicability = framework keys ∪ (method-specific keys for each
        // stage in the chain). A key is legit as long as *some* stage
        // consumes it.
        let mut applicable: std::collections::BTreeSet<&'static str> =
            std::collections::BTreeSet::new();
        applicable.extend(framework_keys().iter().copied());
        for &m in &chain {
            applicable.extend(method_specific_keys(m).iter().copied());
        }
        // Only method-specific keys get surfaced as "available" — listing
        // framework keys here would conflate the two layers.
        let mut method_only: std::collections::BTreeSet<&'static str> =
            std::collections::BTreeSet::new();
        for &m in &chain {
            method_only.extend(method_specific_keys(m).iter().copied());
        }
        let chain_label: String = if chain.len() == 1 {
            chain[0].label().to_string()
        } else {
            chain
                .iter()
                .map(|m| m.label())
                .collect::<Vec<_>>()
                .join(" → ")
        };
        let available: Vec<&'static str> = method_only.iter().copied().collect();

        let mut seen = std::collections::HashSet::new();
        let mut warnings = Vec::new();
        for key in &self.user_set_keys {
            // `method` / `methods` select the chain itself — they can't be
            // "wrong for the method" in the way other options can.
            if key == "method" || key == "methods" {
                continue;
            }
            if applicable.contains(key.as_str()) {
                continue;
            }
            if !seen.insert(key.clone()) {
                continue;
            }
            warnings.push(format!(
                "fit option `{}` is not used by method `{}` and will be ignored. \
                 Method-specific options for `{}`: {}",
                key,
                chain_label,
                chain_label,
                available.join(", ")
            ));
        }
        warnings
    }

    /// Warn when no estimation method was set anywhere — neither in the model
    /// file's `[fit_options]` nor by the caller (CLI / R / Python wrapper).
    /// In that case `method` carries its `FOCEI` default, and the user may not
    /// realise the engine picked it for them. Wrappers mark an explicit method
    /// by pushing `"method"` onto `user_set_keys` (the parser does the same for
    /// a model-file `method = ...` line), so the absence of that key is the
    /// reliable signal that the value defaulted. Returns `None` once a method
    /// has been specified anywhere.
    pub fn method_default_warning(&self) -> Option<String> {
        if self.user_set_keys.iter().any(|k| k == "method") {
            return None;
        }
        Some(format!(
            "No estimation method was specified in the model file or the call; \
             defaulting to `{}`. Set `method` in [fit_options] or pass it \
             explicitly to silence this warning.",
            self.method.label()
        ))
    }
}

/// Framework-level fit-option keys: consumed by every method and typically
/// exposed as dedicated top-level arguments in the language wrappers
/// (`covariance`, `verbose`, `bloq_method`, `threads`, `sir`, ...). Kept
/// separate from `method_specific_keys` so the "unsupported option" warning
/// can list only method-specific suggestions without conflating the layers.
pub fn framework_keys() -> &'static [&'static str] {
    &[
        "covariance",
        "covariance_method",
        "covariance_fallback",
        "analytic_cov_hessian",
        "fd_hessian_step",
        "verbose",
        "sir",
        "sir_samples",
        "sir_resamples",
        "sir_seed",
        "sir_keep_samples",
        "sir_df",
        "bloq_method",
        "bloq",
        "npde_nsim",
        "npde_seed",
        "mu_referencing",
        "threads",
        "n_starts",
        "start_sigma",
        "multi_start_seed",
        "gradient",
        "gradient_method",
        "iov_column",
        "iov_occasion",
        "optimizer_trace",
        "scale_params",
        "parameter_scaling",
        "max_unconverged_frac",
        "min_obs_for_convergence_check",
        "ebe_warm_start",
        "inits_from_nca",
        "frem_predictions",
        "frem_sigma",
        // ODE-solver knobs: applied to the integrator at parse time via
        // `sync_ode_solver_opts` (gated on the model being an ODE model, not on
        // the estimation method), so they are framework-level — every method
        // that integrates an ODE model honours them. Listing them here keeps
        // `unsupported_keys_warnings` from falsely flagging them as "ignored".
        "ode_reltol",
        "ode_abstol",
        "ode_max_steps",
        "ode_method",
        "ode_stiff_abort_after",
        "ode_auto_switch",
        // Checkpoint / restart (#755): the periodic resume-point writer is driven
        // by the fit runner independent of the estimation method, so these are
        // framework-level — every method honours them.
        "checkpoint",
        "checkpoint_interval_secs",
    ]
}

/// Fit-option keys that are meaningful only for a particular estimation
/// method (or family of methods). `method` / `methods` are omitted — those
/// select the chain itself and can't be "wrong for the method". Framework-
/// wide keys live in `framework_keys`.
pub fn method_specific_keys(m: EstimationMethod) -> &'static [&'static str] {
    match m {
        EstimationMethod::Foce => &[
            "maxiter",
            "inner_maxiter",
            "inner_tol",
            "inner_restarts",
            "inner_optimizer",
            "optimizer",
            "outer_xtol",
            "outer_ftol",
            "steihaug_max_iters",
            "global_search",
            "global_maxeval",
            "stagnation_guard",
            "reconverge_gradient_interval",
        ],
        // FOCEI accepts `n_agq`: `= 1` (default) is plain FOCEI, `> 1` is the Gauss-Newton-
        // anchored quadrature refinement.
        EstimationMethod::FoceI => &[
            "n_agq",
            "maxiter",
            "inner_maxiter",
            "inner_tol",
            "inner_restarts",
            "inner_optimizer",
            "optimizer",
            "outer_xtol",
            "outer_ftol",
            "steihaug_max_iters",
            "global_search",
            "global_maxeval",
            "stagnation_guard",
            "reconverge_gradient_interval",
        ],
        // Laplace is the exact-anchor quadrature; `n_agq` (default 1) is its node count.
        // `n_agq = 1` is Laplace, `> 1` is adaptive Gauss–Hermite quadrature.
        EstimationMethod::Laplace => &[
            "n_agq",
            "maxiter",
            "inner_maxiter",
            "inner_tol",
            "inner_restarts",
            "inner_optimizer",
            "optimizer",
            "outer_xtol",
            "outer_ftol",
            "steihaug_max_iters",
            "global_search",
            "global_maxeval",
            "stagnation_guard",
            "reconverge_gradient_interval",
        ],
        EstimationMethod::FoceGn => &[
            "maxiter",
            "inner_maxiter",
            "inner_tol",
            "inner_restarts",
            "inner_optimizer",
            "gn_lambda",
        ],
        EstimationMethod::FoceGnHybrid => &[
            "maxiter",
            "inner_maxiter",
            "inner_tol",
            "inner_restarts",
            "inner_optimizer",
            "optimizer",
            "outer_xtol",
            "outer_ftol",
            "steihaug_max_iters",
            "global_search",
            "global_maxeval",
            "stagnation_guard",
            "gn_lambda",
            "reconverge_gradient_interval",
        ],
        EstimationMethod::Saem => &[
            "inner_maxiter",
            "inner_tol",
            "inner_optimizer",
            "n_exploration",
            "n_convergence",
            "n_mh_steps",
            "n_leapfrog",
            "saem_n_leapfrog",
            "adapt_interval",
            "mstep_damping",
            "saem_mstep_damping",
            "omega_burnin",
            "conddist",
            "saem_conddist",
            "conddist_nsamp",
            "conddist_burnin",
            "conddist_keep_samples",
            "seed",
            "saem_seed",
        ],
        EstimationMethod::Imp => &[
            "imp_samples",
            "imp_proposal_df",
            "imp_seed",
            "imp_low_ess_threshold",
            "imp_iterations",
            "imp_averaging",
            "imp_eval_only",
            "inner_maxiter",
            "inner_tol",
            "iscale_min",
            "iscale_max",
            "frem_rao_blackwell",
            "imp_auto",
            "imp_defensive_alpha",
        ],
        EstimationMethod::Impmap => &[
            "inner_maxiter",
            "inner_tol",
            "inner_optimizer",
            "impmap_iterations",
            "impmap_samples",
            "impmap_proposal_df",
            "impmap_seed",
            "impmap_averaging",
            "impmap_low_ess_threshold",
            "impmap_trace",
            "impmap_mceta",
            "impmap_sobol",
            "iscale_min",
            "iscale_max",
            "frem_rao_blackwell",
            "impmap_auto",
            // Both spellings write the same field; `impmap_*` is the canonical
            // IMPMAP form, `imp_*` is accepted so neither is wrongly flagged
            // "unsupported" (issue #528).
            "impmap_defensive_alpha",
            "imp_defensive_alpha",
        ],
        EstimationMethod::Bayes => &[
            "inner_maxiter",
            "inner_tol",
            "n_mh_steps",
            "n_leapfrog",
            "bayes_warmup",
            "bayes_iters",
            "bayes_chains",
            "bayes_thin",
            "bayes_seed",
        ],
    }
}

/// Trial design specification parsed from [simulation] block
#[derive(Debug, Clone)]
pub struct SimulationSpec {
    pub n_subjects: usize,
    pub dose_amt: f64,
    pub dose_cmt: usize,
    pub obs_times: Vec<f64>,
    pub seed: u64,
    /// Administrative censoring horizon for TTE endpoints (`[simulation] horizon`).
    /// `Some(t)` makes `t` the right-censoring window for **every** simulated TTE
    /// cause, overriding the per-record `observation_window` so a re-simulated
    /// event-bearing subject censors at the planned study end rather than drawing
    /// unbounded (the competing-risks VPC fix — see `survival::simulate_tte`).
    /// `None` falls back to the per-record window. Required when the model has a
    /// TTE endpoint and synthetic subjects are generated.
    pub horizon: Option<f64>,
    /// Optional per-subject covariates: (name, values) — length must equal n_subjects
    pub covariates: Vec<(String, Vec<f64>)>,
}

/// Full parsed model including simulation spec and fit options
pub struct ParsedModel {
    pub model: CompiledModel,
    pub simulation: Option<SimulationSpec>,
    /// Parsed `[adaptive_dosing]` block (#391 S2): a declarative, file-driven
    /// reactive controller. `None` when the block is absent. This slice (S2.1)
    /// only parses and validates the spec; compiling it to a controller and
    /// running it against `model` is a later step (S2.2+).
    pub adaptive_dosing: Option<crate::sim::adaptive::AdaptiveDosingSpec>,
    pub fit_options: FitOptions,
    /// Declarations from the optional `[covariates]` block. `None` when the
    /// block is absent (legacy auto-detect: every non-standard CSV column is a
    /// covariate). `Some` (possibly empty) when present, in which case it is
    /// authoritative — only listed columns are covariates, and each must exist
    /// in the data and be numerically coded. Drives the file-based readers and
    /// the [`FitResult::covariate_table`].
    pub covariate_decls: Option<Vec<CovariateDecl>>,
    /// `path` from the optional `[data]` block, resolved relative to the model
    /// file's own directory by `parse_full_model_file` (NONMEM `$DATA`
    /// analogue, #690). `None` when the block is absent — callers must then
    /// supply a dataset path externally, as before. An externally supplied
    /// path (CLI `--data`, R) always overrides this — see
    /// [`crate::api::resolve_data_path`].
    pub data_path: Option<String>,
    /// Canonical-role → actual-header column remappings from the optional
    /// `[data]` block (#730, NONMEM `$INPUT TIME=TAFD` analogue). Each entry is
    /// `(canonical_role_lowercase, actual_csv_header)` — e.g. `("time",
    /// "TAFD")` for `TIME = TAFD`. Empty when the block declares no mappings.
    /// The reader renames each mapped header to its canonical role, so the
    /// column serves that role and is excluded from covariate auto-detection.
    pub column_map: Vec<(String, String)>,
    /// 1-based source line of each unnamed `[block]` header, keyed by the
    /// lowercased block type (e.g. `"individual_parameters" -> 7`). Used by
    /// `ferx check` to attach a block-level location to diagnostics. Empty when
    /// a model is constructed programmatically rather than parsed from text.
    pub block_lines: std::collections::HashMap<String, usize>,
}

/// Factories that build minimal `CompiledModel` instances for unit tests.
/// Exposed `pub(crate)` (gated on `#[cfg(test)]`) so other modules' tests
/// can construct models without duplicating the boilerplate.
#[cfg(test)]
#[path = "types_test_helpers.rs"]
pub(crate) mod test_helpers;

#[cfg(test)]
#[path = "types_tests.rs"]
mod tests;
