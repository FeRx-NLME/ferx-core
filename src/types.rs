use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

// Re-export the milestone-2 sensitivity partials placeholder so
// `CompiledModel` can hold one and external callers (test fixtures and the
// `generate_data` data-generation binary) can construct an empty one. The
// inner `Expression` AST stays parser-private — only `IndivParamPartials::empty`
// and the `Debug`/`Clone` derives are reachable from outside the crate.
pub use crate::parser::model_parser::IndivParamPartials;

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
    pub cmt: usize,
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
pub const MAX_PK_PARAMS: usize = 16;

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

/// The engine-reserved PK slots: bioavailability (`F`) and absorption lag
/// (`lagtime`). `ode_param_slots` keeps these free for an undeclared F/lagtime
/// (issue #122), both the analytical and ODE engines apply them to the dose
/// itself rather than the RHS, and the "computed but never used" census exempts
/// parameters routed here. Single source of truth so those sites can't drift.
pub(crate) const RESERVED_PK_SLOTS: [usize; 2] = [PK_IDX_F, PK_IDX_LAGTIME];

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
}

impl DoseAttr {
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

    /// `true` when no compartment-indexed attribute is recorded — the common
    /// case (bare-`F`/`lagtime` model, no `D{cmt}`). Lets the dose-resolution
    /// step short-circuit before scanning a subject's doses: with no indexed
    /// slot there can be no modeled-`RATE` dose to resolve.
    pub fn is_empty(&self) -> bool {
        self.indexed.is_empty()
    }

    /// The PkParams slot holding `attr` for compartment `cmt`, if the model
    /// declared a compartment-indexed parameter for it.
    pub fn indexed_slot(&self, attr: DoseAttr, cmt: usize) -> Option<usize> {
        self.indexed.get(&(attr, cmt)).copied()
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
#[derive(Debug, Clone)]
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
    #[cfg(feature = "survival")]
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

    /// Covariate snapshot at EVID=2 row index `m`. Same fallback as
    /// the others — for time-constant covariates this returns the
    /// subject-static map.
    pub fn pk_only_cov(&self, m: usize) -> &HashMap<String, f64> {
        self.pk_only_covariates.get(m).unwrap_or(&self.covariates)
    }
}

/// Summary of records excluded by `[data_selection]` `ignore`/`accept` rules.
#[derive(Debug, Clone, Default)]
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
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, Default, PartialEq)]
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
                if i == j {
                    free_mask[(i, j)] = true;
                } else if !diagonal && m[(i, j)] != 0.0 {
                    free_mask[(i, j)] = true;
                }
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
    TwoCptIv,
    TwoCptOral,
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
            PkModel::TwoCptIv => "two_cpt_iv",
            PkModel::TwoCptOral => "two_cpt_oral",
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
    /// first-order `ka` absorption into central. Notably this still EXCLUDES oral
    /// peripherals, which the closed forms cannot infuse into. A `D{cmt}` outside
    /// this set is rejected at parse time rather than silently mis-routed (no-TV
    /// path) or panicking (event-driven path).
    pub(crate) fn infusable_compartments(&self) -> &'static [usize] {
        match self {
            PkModel::OneCptIv => &[1],
            // Oral: cmt 1 = depot (zero-order-into-depot, #400), cmt 2 = central
            // (depot-bypassing infusion).
            PkModel::OneCptOral => &[1, 2],
            // Transit models a bolus absorbed through the Gamma transit chain;
            // modeled-duration infusions are unsupported in v1, so a `D{cmt}` on a
            // transit model is rejected at parse (#386).
            PkModel::OneCptTransit => &[],
            PkModel::TwoCptIv => &[1, 2],
            PkModel::TwoCptOral => &[1, 2],
            PkModel::ThreeCptIv => &[1, 2, 3],
            PkModel::ThreeCptOral => &[1, 2],
        }
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
            "two_cpt_iv" | "two_compartment_iv" => Some(PkModel::TwoCptIv),
            "two_cpt_oral" | "two_compartment_oral" => Some(PkModel::TwoCptOral),
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
                | PkModel::TwoCptOral
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorModel {
    Additive,
    Proportional,
    Combined,
}

/// How a sigma parameter enters the residual error model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// Residual error specification for a model.
///
/// `Single` applies one error model to every observation (the default and the
/// only form analytical-PK models support). `PerCmt` dispatches a distinct
/// error model per observed compartment — the multi-endpoint / simultaneous
/// PK-PD case (issue #14). The map key is the 1-based CMT index from the data
/// file's CMT column, matching `subject.obs_cmts[i]`.
#[derive(Debug, Clone)]
pub enum ErrorSpec {
    /// One error model for all observations.
    Single(ErrorModel),
    /// Per-CMT error models. Each endpoint carries its own `ErrorModel` and the
    /// indices into the flat global `sigma.values` vector that supply its
    /// sigmas (declaration order in the `[parameters]` block).
    PerCmt(HashMap<usize, EndpointError>),
}

impl Default for ErrorSpec {
    fn default() -> Self {
        ErrorSpec::Single(ErrorModel::Additive)
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

impl ErrorSpec {
    /// Observation-level loadings onto the flat residual-error vector.
    ///
    /// For an observation with prediction `f`, additive error contributes a
    /// coefficient of `1`, proportional error contributes `f`, and combined
    /// error contributes both. Multiplying these loadings by the residual
    /// sigma covariance matrix yields the scalar variance on the diagonal and
    /// the cross-observation covariance off diagonal.
    pub fn sigma_loadings(&self, cmt: usize, f: f64, n_sigma: usize) -> Vec<(usize, f64)> {
        match self {
            ErrorSpec::Single(em) => match em {
                ErrorModel::Additive => {
                    if n_sigma > 0 {
                        vec![(0, 1.0)]
                    } else {
                        Vec::new()
                    }
                }
                ErrorModel::Proportional => {
                    if n_sigma > 0 {
                        vec![(0, f)]
                    } else {
                        Vec::new()
                    }
                }
                ErrorModel::Combined => {
                    let mut out = Vec::with_capacity(2);
                    if n_sigma > 0 {
                        out.push((0, f));
                    }
                    if n_sigma > 1 {
                        out.push((1, 1.0));
                    }
                    out
                }
            },
            ErrorSpec::PerCmt(map) => match map.get(&cmt) {
                Some(ep) => match ep.error_model {
                    ErrorModel::Additive => ep
                        .sigma_idx
                        .first()
                        .copied()
                        .map(|i| vec![(i, 1.0)])
                        .unwrap_or_default(),
                    ErrorModel::Proportional => ep
                        .sigma_idx
                        .first()
                        .copied()
                        .map(|i| vec![(i, f)])
                        .unwrap_or_default(),
                    ErrorModel::Combined => {
                        let mut out = Vec::with_capacity(2);
                        if let Some(&i) = ep.sigma_idx.first() {
                            out.push((i, f));
                        }
                        if let Some(&i) = ep.sigma_idx.get(1) {
                            out.push((i, 1.0));
                        }
                        out
                    }
                },
                None => Vec::new(),
            },
        }
    }

    /// `∂(sigma loading coefficient)/∂f` for each slot an observation loads on,
    /// parallel in shape to [`sigma_loadings`].
    ///
    /// Every loading coefficient is affine in the prediction `f`: the
    /// proportional slot loads `f` (slope `1`), the additive slot loads the
    /// constant `1` (slope `0`). Returning the slopes with the *same slot
    /// presence* as [`sigma_loadings`] lets the dense-`R` derivative
    /// ([`crate::stats::residual_error::compute_dr_df_matrices`]) reuse the exact
    /// bilinear cross-covariance assembly with the value loadings replaced by
    /// these slopes — the off-diagonal `R` is linear in each observation's
    /// loadings, so `∂R_jk/∂f_j = cross(slopes_j, values_k)`.
    pub fn sigma_loading_slopes(&self, cmt: usize, n_sigma: usize) -> Vec<(usize, f64)> {
        match self {
            ErrorSpec::Single(em) => match em {
                ErrorModel::Additive => {
                    if n_sigma > 0 {
                        vec![(0, 0.0)]
                    } else {
                        Vec::new()
                    }
                }
                ErrorModel::Proportional => {
                    if n_sigma > 0 {
                        vec![(0, 1.0)]
                    } else {
                        Vec::new()
                    }
                }
                ErrorModel::Combined => {
                    let mut out = Vec::with_capacity(2);
                    if n_sigma > 0 {
                        out.push((0, 1.0));
                    }
                    if n_sigma > 1 {
                        out.push((1, 0.0));
                    }
                    out
                }
            },
            ErrorSpec::PerCmt(map) => match map.get(&cmt) {
                Some(ep) => match ep.error_model {
                    ErrorModel::Additive => ep
                        .sigma_idx
                        .first()
                        .copied()
                        .map(|i| vec![(i, 0.0)])
                        .unwrap_or_default(),
                    ErrorModel::Proportional => ep
                        .sigma_idx
                        .first()
                        .copied()
                        .map(|i| vec![(i, 1.0)])
                        .unwrap_or_default(),
                    ErrorModel::Combined => {
                        let mut out = Vec::with_capacity(2);
                        if let Some(&i) = ep.sigma_idx.first() {
                            out.push((i, 1.0));
                        }
                        if let Some(&i) = ep.sigma_idx.get(1) {
                            out.push((i, 0.0));
                        }
                        out
                    }
                },
                None => Vec::new(),
            },
        }
    }

    /// Residual variance including fixed residual correlations from
    /// `block_sigma`.
    pub fn variance_at_with_correlations(
        &self,
        cmt: usize,
        f_pred: f64,
        sigma: &[f64],
        correlations: &[ResidualCorrelation],
    ) -> f64 {
        if correlations.is_empty() {
            return self.variance_at(cmt, f_pred, sigma);
        }
        // The correlated variance is exactly [`variance_at_scaled`] with an
        // all-ones multiplier: an empty `mult` makes every loading's multiplier
        // default to 1.0. Delegate so the loading + cross-covariance formula
        // lives in one place (#484 review #9) and the magnitude path can never
        // silently diverge from the bare-sigma path.
        self.variance_at_scaled(cmt, f_pred, sigma, correlations, &[])
    }

    /// Residual variance with a per-observation custom magnitude (#484).
    ///
    /// Like [`variance_at_with_correlations`] but scales each sigma loading by
    /// the per-observation multiplier `mult[idx]` (1-based on the flat sigma
    /// vector, supplied by [`RuvMagnitude::eval_obs`]). A `mult` of all ones
    /// reproduces [`variance_at_with_correlations`] exactly. Cross terms from
    /// `block_sigma` scale by the product of the two loadings' multipliers.
    ///
    /// [`variance_at_with_correlations`]: ErrorSpec::variance_at_with_correlations
    pub fn variance_at_scaled(
        &self,
        cmt: usize,
        f_pred: f64,
        sigma: &[f64],
        correlations: &[ResidualCorrelation],
        mult: &[f64],
    ) -> f64 {
        let loadings = self.sigma_loadings(cmt, f_pred, sigma.len());
        if loadings.is_empty() {
            return f64::NAN;
        }
        let m = |idx: usize| mult.get(idx).copied().unwrap_or(1.0);
        let mut v = 0.0;
        for &(idx, coeff) in &loadings {
            let Some(&s) = sigma.get(idx) else {
                return f64::NAN;
            };
            let c = coeff * m(idx);
            v += c * c * s * s;
        }
        for corr in correlations {
            let ci = loadings
                .iter()
                .find(|(idx, _)| *idx == corr.sigma_i)
                .map(|(_, coeff)| *coeff * m(corr.sigma_i));
            let cj = loadings
                .iter()
                .find(|(idx, _)| *idx == corr.sigma_j)
                .map(|(_, coeff)| *coeff * m(corr.sigma_j));
            let (Some(ci), Some(cj)) = (ci, cj) else {
                continue;
            };
            let (Some(&si), Some(&sj)) = (sigma.get(corr.sigma_i), sigma.get(corr.sigma_j)) else {
                return f64::NAN;
            };
            v += 2.0 * ci * cj * corr.rho * si * sj;
        }
        v.max(1e-12)
    }

    /// `SigmaType` for each entry of the flat global sigma vector, as a `Vec`
    /// of length `n_sigma`, so `FitResult` can label/scale every sigma.
    ///
    /// `Single` stamps the error model's own sigma types into the leading
    /// slots and leaves any further declared sigmas as `Additive`. `PerCmt`
    /// stamps the type of every sigma index each endpoint owns; sigmas not
    /// referenced by any endpoint default to `Additive`. Either way the
    /// returned length always equals `n_sigma`.
    pub fn sigma_types(&self, n_sigma: usize) -> Vec<SigmaType> {
        let mut out = vec![SigmaType::Additive; n_sigma];
        match self {
            ErrorSpec::Single(em) => {
                for (i, t) in em.sigma_types().into_iter().enumerate() {
                    if i < out.len() {
                        out[i] = t;
                    }
                }
            }
            ErrorSpec::PerCmt(map) => {
                for ep in map.values() {
                    let types = ep.error_model.sigma_types();
                    for (k, &idx) in ep.sigma_idx.iter().enumerate() {
                        if idx < out.len() {
                            out[idx] = types[k];
                        }
                    }
                }
            }
        }
        out
    }

    /// `d(residual variance)/d(prediction f)` for one observation at `cmt`.
    ///
    /// The score term the SAEM M-step needs alongside the variance. Additive
    /// endpoints contribute 0; proportional/combined endpoints contribute
    /// `2·f·σ_prop²` (σ_prop is the endpoint's proportional sigma, which is the
    /// first sigma for both `Proportional` and `Combined`). `Single` ignores
    /// `cmt`; `PerCmt` dispatches on the endpoint registered for `cmt`.
    pub fn dvar_df(&self, cmt: usize, f: f64, sigma: &[f64]) -> f64 {
        let (em, prop_sigma) = match self {
            ErrorSpec::Single(em) => (*em, sigma.first().copied().unwrap_or(0.0)),
            ErrorSpec::PerCmt(map) => match map.get(&cmt) {
                Some(ep) => (
                    ep.error_model,
                    ep.sigma_idx
                        .first()
                        .and_then(|&i| sigma.get(i))
                        .copied()
                        .unwrap_or(0.0),
                ),
                None => return 0.0,
            },
        };
        match em {
            ErrorModel::Additive => 0.0,
            ErrorModel::Proportional | ErrorModel::Combined => 2.0 * f * prop_sigma * prop_sigma,
        }
    }

    /// Flat sigma-vector index of `cmt`'s proportional sigma (the slot a
    /// magnitude multiplier scales for the `f`-derivative chain): slot `0` for
    /// `Single`, the endpoint's first `sigma_idx` for `PerCmt`. `None` when
    /// `PerCmt` has no endpoint registered for `cmt`, or that endpoint declares
    /// no sigma at all — both cases the `_scaled` derivative callers treat as
    /// "no residual error here" (`0.0`). Single source for
    /// [`dvar_df_scaled`](Self::dvar_df_scaled) and
    /// [`d2var_df2_scaled`](Self::d2var_df2_scaled) so the two can't resolve a
    /// different slot for the same `(cmt, mult)` (#486 review).
    fn prop_sigma_slot(&self, cmt: usize) -> Option<usize> {
        match self {
            ErrorSpec::Single(_) => Some(0),
            ErrorSpec::PerCmt(map) => map.get(&cmt)?.sigma_idx.first().copied(),
        }
    }

    /// `dvar_df` with a per-observation custom magnitude (#484).
    ///
    /// The proportional loading carries the multiplier `m_prop`, so the
    /// variance's `f`-dependent term is `(f·m_prop·σ_prop)²` and its `f`
    /// derivative scales by `m_prop²`. The proportional sigma is the first sigma
    /// slot for both `Proportional` and `Combined`; `mult` is indexed on the
    /// flat sigma vector. A `mult` of all ones reproduces [`dvar_df`].
    ///
    /// [`dvar_df`]: ErrorSpec::dvar_df
    pub fn dvar_df_scaled(&self, cmt: usize, f: f64, sigma: &[f64], mult: &[f64]) -> f64 {
        let Some(prop_slot) = self.prop_sigma_slot(cmt) else {
            return 0.0;
        };
        let m = mult.get(prop_slot).copied().unwrap_or(1.0);
        self.dvar_df(cmt, f, sigma) * m * m
    }

    /// `d²(residual variance)/d(prediction f)²` for one observation at `cmt`.
    ///
    /// Additive endpoints contribute 0 (variance is `σ_add²`, independent of f).
    /// Proportional and combined endpoints contribute `2·σ_prop²` (variance has
    /// a `f²·σ_prop²` term, so the second derivative w.r.t. f is constant).
    /// `Single` ignores `cmt`; `PerCmt` dispatches on the endpoint registered
    /// for `cmt`. Used by the Almquist Laplace FOCEI gradient's θ-axis β_j
    /// chain — keeping the per-CMT routing here lets the same closed-form
    /// gradient handle multi-endpoint models without changing the call site.
    pub fn d2var_df2(&self, cmt: usize, sigma: &[f64]) -> f64 {
        let (em, prop_sigma) = match self {
            ErrorSpec::Single(em) => (*em, sigma.first().copied().unwrap_or(0.0)),
            ErrorSpec::PerCmt(map) => match map.get(&cmt) {
                Some(ep) => (
                    ep.error_model,
                    ep.sigma_idx
                        .first()
                        .and_then(|&i| sigma.get(i))
                        .copied()
                        .unwrap_or(0.0),
                ),
                None => return 0.0,
            },
        };
        match em {
            ErrorModel::Additive => 0.0,
            ErrorModel::Proportional | ErrorModel::Combined => 2.0 * prop_sigma * prop_sigma,
        }
    }

    /// `d²(residual variance)/d(prediction f)²` with a per-observation custom
    /// magnitude (#484/#576) — the second-derivative analogue of [`dvar_df_scaled`].
    ///
    /// The proportional loading carries the multiplier `m_prop`, so the variance's
    /// `f²·(m_prop·σ_prop)²` term differentiates twice to `2·(m_prop·σ_prop)²`,
    /// i.e. [`d2var_df2`] scaled by `m_prop²` — exactly the `m²` factor
    /// [`dvar_df_scaled`] applies to `dvar_df`. Keeping the same scaling here lets
    /// the M3 censored curvature (which differentiates the same `v(f)`) stay
    /// internally consistent under a custom RUV magnitude, and the FOCEI outer
    /// θ/σ gradient's direct-θ channel (`sens_outer_gradient::prepare_stacked`)
    /// consume the same magnitude-scaled second derivative. A `mult` of all ones
    /// reproduces [`d2var_df2`].
    ///
    /// [`dvar_df_scaled`]: ErrorSpec::dvar_df_scaled
    /// [`dvar_df`]: ErrorSpec::dvar_df
    /// [`d2var_df2`]: ErrorSpec::d2var_df2
    pub fn d2var_df2_scaled(&self, cmt: usize, sigma: &[f64], mult: &[f64]) -> f64 {
        let Some(prop_slot) = self.prop_sigma_slot(cmt) else {
            return 0.0;
        };
        let m = mult.get(prop_slot).copied().unwrap_or(1.0);
        self.d2var_df2(cmt, sigma) * m * m
    }

    /// `d(residual variance)/d(log σ_k)` for one observation at `cmt`, where
    /// `k` indexes the flat global sigma vector. Zero when `σ_k` does not enter
    /// this observation's endpoint, so the SAEM sigma-gradient can sum this over
    /// every observation and have each sigma pick up only its own endpoint's
    /// contributions. (`Proportional` slot → `2·σ_k²·f²`, `Additive` slot →
    /// `2·σ_k²`.)
    pub fn dvar_dlogsigma(&self, cmt: usize, k: usize, f: f64, sigma: &[f64]) -> f64 {
        let sk = match sigma.get(k) {
            Some(&s) => s,
            None => return 0.0,
        };
        let sk2 = sk * sk;
        // Resolve which `SigmaType` (if any) global index `k` plays for this
        // observation's endpoint.
        let stype = match self {
            ErrorSpec::Single(em) => em.sigma_types().get(k).copied(),
            ErrorSpec::PerCmt(map) => match map.get(&cmt) {
                Some(ep) => ep
                    .sigma_idx
                    .iter()
                    .position(|&i| i == k)
                    .and_then(|p| ep.error_model.sigma_types().get(p).copied()),
                None => None,
            },
        };
        match stype {
            Some(SigmaType::Proportional) => 2.0 * sk2 * f * f,
            Some(SigmaType::Additive) => 2.0 * sk2,
            None => 0.0,
        }
    }

    /// Residual variance for one observation, dispatching on its compartment.
    ///
    /// For `Single` the `cmt` is ignored and the full `sigma` slice is used
    /// (the back-compat path). For `PerCmt` the endpoint registered for `cmt`
    /// selects the error model and slices `sigma` by its `sigma_idx`. Returns
    /// `NaN` when `cmt` has no registered endpoint, or when an endpoint's
    /// `sigma_idx` points outside `sigma` — defensive guards mirroring the
    /// scaling path. Fit-time validation rejects an uncovered CMT up front,
    /// and `build_error_spec` resolves indices against the real sigma vector,
    /// so a `NaN` here is only reachable via a hand-constructed model.
    /// Whether the residual variance depends on the prediction `f` for any
    /// endpoint (proportional or combined). When `false` (purely additive),
    /// the variance is constant in `f`, so FOCE's choice of evaluation point
    /// (linearized `f0` vs population `f(η=0)`) is irrelevant and the cheap
    /// path stays bit-identical. Used to gate the FOCE population-variance
    /// (`f(η=0)`) treatment in the marginal, analytical gradient, and
    /// covariance step.
    pub fn has_f_dependent_variance(&self) -> bool {
        match self {
            ErrorSpec::Single(em) => !matches!(em, ErrorModel::Additive),
            ErrorSpec::PerCmt(map) => map
                .values()
                .any(|ep| !matches!(ep.error_model, ErrorModel::Additive)),
        }
    }

    /// Global `sigma.values` indices of the additive component of every
    /// `Combined` endpoint (the second sigma slot). De-duplicated; empty when
    /// no endpoint is combined.
    pub fn combined_additive_sigma_indices(&self) -> Vec<usize> {
        match self {
            ErrorSpec::Single(ErrorModel::Combined) => vec![1],
            ErrorSpec::Single(_) => Vec::new(),
            ErrorSpec::PerCmt(map) => {
                let mut out = Vec::new();
                for endpoint in map.values() {
                    if matches!(endpoint.error_model, ErrorModel::Combined) {
                        if let Some(&idx) = endpoint.sigma_idx.get(1) {
                            if !out.contains(&idx) {
                                out.push(idx);
                            }
                        }
                    }
                }
                out
            }
        }
    }

    pub fn variance_at(&self, cmt: usize, f_pred: f64, sigma: &[f64]) -> f64 {
        use crate::stats::residual_error::residual_variance;
        match self {
            ErrorSpec::Single(em) => residual_variance(*em, f_pred, sigma),
            ErrorSpec::PerCmt(map) => match map.get(&cmt) {
                Some(ep) => {
                    // Slice length is tied to the endpoint's error model
                    // (1 for additive/proportional, 2 for combined); the max
                    // is 2, so a stack buffer avoids a per-observation alloc.
                    let n = ep.error_model.n_sigma();
                    let mut buf = [0.0f64; 2];
                    for k in 0..n.min(2) {
                        match ep.sigma_idx.get(k).and_then(|&i| sigma.get(i)) {
                            Some(&v) => buf[k] = v,
                            None => return f64::NAN, // malformed spec / sigma length
                        }
                    }
                    residual_variance(ep.error_model, f_pred, &buf[..n.min(2)])
                }
                None => f64::NAN,
            },
        }
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone)]
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
pub enum ScalingSpec {
    /// No scaling: prediction is returned as-is.
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

impl Default for ScalingSpec {
    fn default() -> Self {
        Self::None
    }
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

#[cfg(feature = "survival")]
/// A single non-Gaussian observation record on a subject.
#[derive(Debug, Clone)]
pub enum ObsRecord {
    Event {
        time: f64,
        event_type: EventType,
        /// Left truncation / delayed entry time (0.0 when none).
        /// The likelihood conditions on survival past entry_time:
        ///   H_eff(T) = H(T) − H(entry_time)
        entry_time: f64,
        cmt: usize,
    },
    // DiscreteState and Count variants deferred to Phase 4/5
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
/// Per-CMT endpoint likelihood specification.
pub enum EndpointLikelihood {
    Gaussian(EndpointError),
    Tte { hazard: HazardSpec },
    // Binary, Ordinal, Poisson, NegBin, Ctmm, Dtmm deferred to Phase 4/5
}

#[cfg(feature = "survival")]
impl std::fmt::Debug for EndpointLikelihood {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EndpointLikelihood::Gaussian(e) => write!(f, "Gaussian({e:?})"),
            EndpointLikelihood::Tte { hazard } => write!(f, "Tte({hazard:?})"),
        }
    }
}

/// Outcome of a single simulated observation — replaces the previous `dv_sim: f64` field.
///
/// `Continuous` preserves the existing Gaussian path unchanged.
/// `Event` carries TTE-specific outputs (gated behind `survival` feature).
#[derive(Debug, Clone)]
pub enum SimOutcome {
    /// Gaussian continuous prediction + residual noise (the only variant before Phase 1).
    Continuous { value: f64 },
    /// TTE event: simulated event time and whether it occurred before the censoring horizon.
    #[cfg(feature = "survival")]
    Event { time: f64, observed: bool },
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
        }
    }
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
    pub tv_fn: Option<Box<dyn Fn(&[f64], &HashMap<String, f64>) -> Vec<f64> + Send + Sync>>,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradientMethod {
    Auto,
    Ad,
    Fd,
}

impl Default for GradientMethod {
    fn default() -> Self {
        Self::Auto
    }
}

impl CompiledModel {
    /// Returns true when this model uses ODE integration; false for analytical PK.
    pub fn is_ode_based(&self) -> bool {
        self.ode_spec.is_some()
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

    /// Copy the configured ODE solver tolerances from `opts` onto this model's
    /// [`OdeSpec`] (no-op for analytical models). Call this once after the
    /// model file's `[fit_options]` and any call-time `settings` overrides have
    /// been merged into `opts`, so the integrator uses the requested accuracy.
    /// The parser calls it at parse time, so `.ferx` `[fit_options]` and any
    /// entry that integrates the parsed spec as-is (`predict`, `fit_from_files`)
    /// already use the configured accuracy.
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
        }
    }

    /// Returns true when the model has a `[diffusion]` block (SDE / EKF path).
    pub fn is_sde(&self) -> bool {
        self.diffusion_theta_start.is_some()
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

    /// Always false without the `survival` feature - TTE endpoints can't be
    /// parsed, so no model can carry one.
    #[cfg(not(feature = "survival"))]
    pub fn has_tte(&self) -> bool {
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
        if self.pk_indices.iter().any(|&i| i == PK_IDX_LAGTIME) {
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

    /// True when the model wires in a bioavailability `F`/`Fn` parameter (on
    /// either engine). Mirrors [`Self::has_lagtime`]: the analytical route puts
    /// [`PK_IDX_F`] in `pk_indices` (from `f=` on the `[structural_model]`
    /// line), while both engines may instead name it `F` (any case) in
    /// `[individual_parameters]`; a compartment-indexed `Fn` routes on the ODE
    /// engine only. Used with [`Subject::has_rate_defined_infusion`] to skip the
    /// event-driven [`crate::pk::event_driven::EventSchedule`] cache when `F`
    /// could reshape an infusion window across the inner search (#419).
    pub fn has_bioavailability(&self) -> bool {
        if self.pk_indices.iter().any(|&i| i == PK_IDX_F) {
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
    /// raw data-file time (`obs_raw_times`, matching what NONMEM's `$ERROR`
    /// sees), falling back to `obs_times` for in-memory subjects.
    pub fn ruv_obs_mult(&self, subject: &Subject, theta: &[f64]) -> Option<Vec<Vec<f64>>> {
        let rm = self.ruv_magnitude.as_ref()?;
        if !rm.is_active() {
            return None;
        }
        let n = subject.observations.len();
        let mut out = Vec::with_capacity(n);
        for j in 0..n {
            let time = subject
                .obs_raw_times
                .get(j)
                .copied()
                .unwrap_or_else(|| subject.obs_times.get(j).copied().unwrap_or(0.0));
            out.push(rm.eval_obs(theta, subject.obs_cov(j), time));
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
            let time = subject
                .obs_raw_times
                .get(j)
                .copied()
                .unwrap_or_else(|| subject.obs_times.get(j).copied().unwrap_or(0.0));
            out.push(rm.eval_obs_theta_grad(theta, subject.obs_cov(j), time)?);
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
        }
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
    /// One predicate in the `iiv_on_ruv` × {plain | IOV | M3} × {closed-form | ODE}
    /// routing decision. **As of #486 this gate forces FD for no combination** — every
    /// `iiv_on_ruv` model (closed-form / ODE, plain / IOV / M3 BLOQ, including the triples)
    /// is served by the shared, provider-agnostic gradient assembly. The censored ×
    /// residual-eta cross-terms (`h·z` inner column, `C·z`/`C·m·a` true-Hessian / mixed
    /// blocks, the σ-cross) live in `residual_inner_obs` (inner) and `prepare` /
    /// `prepare_stacked` (outer), keyed on `subject.cens[j]` and `residual_error_eta` over
    /// whatever `ObsSens` the walk emits. So the **non-IOV ODE M3 BLOQ + `iiv_on_ruv`**
    /// combination — the last holdout — is analytic too (the #547 pattern: the ODE walk
    /// emits the same per-observation shape the closed-form M3 + `iiv_on_ruv` assembly
    /// already handled; the ODE and closed-form packed gradients are bit-identical and both
    /// match reconverged FD to ~1e-7, inner and outer, #486). The predicate is retained as
    /// the canonical "does `iiv_on_ruv` force FD here?" marker — now uniformly `false` — so
    /// the inner/outer gates and their tests document the closed frontier in one place.
    ///
    /// **This is NOT the single source of truth for the full routing** — the decision
    /// is spread across several predicates that must move together, so a future scope
    /// change has to touch all the relevant ones in lockstep or the inner and outer
    /// loops desync:
    /// - this gate (consulted by [`analytic_outer_gradient_available`](crate::sens::provider::analytic_outer_gradient_available)
    ///   and the inner bails [`analytic_inner_common_bail`] / the ODE inner gate);
    /// - [`iov_analytical_supported`](crate::sens::provider::iov_analytical_supported)
    ///   (closed-form IOV: declines M3, served for plain/`iiv_on_ruv`);
    /// - [`ode_iov_supported`](crate::sens::ode_provider::ode_iov_supported)
    ///   (ODE IOV + `iiv_on_ruv`, including the M3 triple, routed here);
    /// - [`analytical_supported`](crate::sens::provider::analytical_supported)
    ///   (closed-form M3 + `iiv_on_ruv`, #4c).
    ///
    /// Net effect today: analytic for **every** `iiv_on_ruv` combination —
    /// **closed-form M3 + `iiv_on_ruv`** (#4c), **closed-form IOV + `iiv_on_ruv`**
    /// (#4b/#474), **ODE IOV + `iiv_on_ruv`** incl. the M3 triple (#486, via
    /// `ode_iov_supported`), and **non-IOV ODE M3 + `iiv_on_ruv`** (#486).
    #[inline]
    pub fn iiv_on_ruv_forces_fd(&self) -> bool {
        // No remaining `iiv_on_ruv` combination forces FD (#486); see the doc above.
        let _ = self;
        false
    }

    /// Whether a custom residual-error magnitude (#484) is active. The
    /// per-observation magnitude expression makes the residual variance depend
    /// on θ directly (not only through the prediction `f`), which the analytic
    /// inner/outer gradient kernels do not yet carry — so both gradient loops
    /// fall back to finite differences when this is `true`. The FD forward NLL
    /// is itself magnitude-aware, so FD stays exact. Removing this gate is the
    /// Phase 5 follow-up (AD through the magnitude program).
    #[inline]
    pub fn has_custom_ruv_magnitude(&self) -> bool {
        self.ruv_magnitude.as_ref().is_some_and(|m| m.is_active())
    }

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
        self.residual_variance_at_scaled(subject.obs_cmts[j], f_pred, sigma, mult) * ruv_scale
    }

    /// Canonical compartment names for analytical models, used in `[derived]` expressions.
    /// For ODE models use `ode_spec.state_names` instead.
    /// Returns a `'static` slice so it can be used in `DerivedContext` without lifetime issues.
    pub fn analytical_compartment_names(&self) -> &'static [String] {
        debug_assert!(
            self.ode_spec.is_none(),
            "analytical_compartment_names called on an ODE model — use ode_spec.state_names instead"
        );
        use std::sync::OnceLock;
        macro_rules! names {
            ($lock:ident, $($name:expr),+) => {{
                static $lock: OnceLock<Vec<String>> = OnceLock::new();
                $lock.get_or_init(|| vec![$($name.to_string()),+]).as_slice()
            }};
        }
        match self.pk_model {
            PkModel::OneCptIv => names!(ONE_CMT_IV, "central"),
            PkModel::OneCptOral => names!(ONE_CMT_ORAL, "depot", "central"),
            PkModel::OneCptTransit => names!(ONE_CMT_TRANSIT, "depot", "central"),
            PkModel::TwoCptIv => names!(TWO_CMT_IV, "central", "peripheral"),
            PkModel::TwoCptOral => names!(TWO_CMT_ORAL, "depot", "central", "peripheral"),
            PkModel::ThreeCptIv => names!(THREE_CMT_IV, "central", "peripheral1", "peripheral2"),
            PkModel::ThreeCptOral => names!(
                THREE_CMT_ORAL,
                "depot",
                "central",
                "peripheral1",
                "peripheral2"
            ),
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

/// Per-subject estimation results
#[derive(Debug, Clone)]
pub struct SubjectResult {
    pub id: String,
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
}

/// Per-subject conditional distribution of the random effects, produced by the
/// opt-in SAEM conditional-distribution pass (`saem_conddist = true`, #257).
///
/// All per-subject vectors are indexed in the same order as the fit's subjects.
/// This is the SAEM analogue of saemix `conddist.saemix` / Monolix's
/// "Conditional Distribution" task — the distribution `p(η_i | y_i; θ̂)` rather
/// than just its mode (the EBE on `SubjectResult.eta`).
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KappaTreatment {
    NotApplicable,
    FixedAtMode,
    Marginalized,
}

/// Result of the importance-sampling marginal log-likelihood step.
///
/// Produced by the `Imp` stage in a method chain (`methods = [..., imp]`).
/// Surfaced on `FitResult.importance_sampling`.
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone, Default)]
pub struct ImpmapTrace {
    pub rows: Vec<ImpmapTraceRow>,
    pub theta_names: Vec<String>,
    /// e.g. `"OMEGA(1,1)"`, `"OMEGA(2,1)"`, `"OMEGA(2,2)"`, …
    pub omega_names: Vec<String>,
    pub sigma_names: Vec<String>,
}

/// Posterior summary for a single scalar parameter, computed across all
/// post-warmup, post-thinning draws from every chain.
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone, PartialEq)]
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

/// A structured warning with severity, category, and message.
///
/// Populated in parallel with `FitResult.warnings` (which remains for
/// backward compatibility). The `category` is a fixed lowercase vocabulary:
/// `convergence`, `covariance_step`, `optimizer_health`, `dw_autocorrelation`,
/// `bloq_method`, `sir`, `importance_sampling`, `data_quality`,
/// `omega_structure`, `ebe_convergence`, `gradient_fallback`,
/// `mu_referencing`, `optimizer_config`, `multi_start`, `cancelled`,
/// `threads`, `condition_number`, `eta_normality`, `eps_shrinkage`,
/// `experimental`, `general` (fallback for unrecognised messages).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WarningEntry {
    pub severity: WarningSeverity,
    /// Fixed lowercase category string (see type-level docs).
    pub category: String,
    /// Human-readable message. For messages that carry a multi-stage chain
    /// prefix such as `[FOCEI] ...`, only the body after the prefix is stored
    /// here; the method tag is moved into `source_method`. For unprefixed
    /// messages this is identical to the corresponding entry in
    /// `FitResult.warnings`.
    pub message: String,
    /// For multi-stage chains, the method that produced this warning.
    pub source_method: Option<String>,
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
    let (severity, category) = if lower.contains("did not converge")
        || lower.contains("without convergence")
        || lower.contains("no multi-start run converged")
    {
        (WarningSeverity::Critical, "convergence")
    } else if lower.contains("covariance step failed")
        || lower.contains("covariance failed")
        || (lower.contains("covariance step") && lower.contains("not positive definite"))
    {
        // "ses not available" intentionally omitted — too broad.
        // The compound check catches format_non_pd_warning ("Covariance step:
        // Hessian is not positive definite") whose prefix differs from "failed:"
        // messages. It must be compound so that "SIR failed: covariance not
        // positive definite" (no "covariance step" token) still routes to "sir".
        (WarningSeverity::Critical, "covariance_step")
    } else if lower.contains("covariance step regularized")
        || lower.contains("off-diagonal fd stencil")
    {
        // Regularisation warning and off-diagonal NaN soft warning both indicate a
        // degraded-but-present covariance step result. Severity within the message
        // (minor/moderate/severe, or "over-optimistic") informs guidance; the
        // category is always covariance_step.
        (WarningSeverity::Warning, "covariance_step")
    } else if lower.contains("ill-conditioned") || lower.contains("condition number") {
        // Note: "covariance step failed: Hessian has ill-conditioned entries" contains
        // "ill-conditioned" but is caught by "covariance step failed" above (else-if chain).
        // Any future covariance message that contains "ill-conditioned" but NOT "failed:"
        // would land here instead — keep this ordering in mind when adding new messages.
        (WarningSeverity::Critical, "condition_number")
    } else if lower.contains("trust radius") || lower.contains("degenerate") {
        (WarningSeverity::Warning, "optimizer_health")
    } else if lower.contains("autocorrelation") || lower.contains("durbin") {
        (WarningSeverity::Warning, "dw_autocorrelation")
    } else if lower.contains("shapiro") || lower.contains("non-normal") {
        (WarningSeverity::Warning, "eta_normality")
    } else if lower.contains("experimental feature") {
        // Experimental-feature notices (issue #175): SDE and neural-network
        // components emit a runtime warning so results are applied with caution.
        (WarningSeverity::Warning, "experimental")
    } else if lower.contains("m3 bloq")
        || lower.contains("bloq handling")
        || lower.contains("m3 censoring")
        || lower.contains("censoring handling")
    {
        (WarningSeverity::Warning, "bloq_method")
    } else if lower.contains("sir failed") || lower.contains("sir requested") {
        (WarningSeverity::Warning, "sir")
    } else if lower.contains("ess = 0") || lower.contains("proposal collapse") {
        (WarningSeverity::Warning, "importance_sampling")
    } else if lower.contains("eps shrinkage") {
        (WarningSeverity::Warning, "eps_shrinkage")
    } else if lower.starts_with("w_addl_missing_ii") || lower.contains("addl > 0 but ii") {
        (WarningSeverity::Warning, "data_quality")
    } else if lower.starts_with("w_iov_occ_missing")
        || lower.contains("missing or unparseable values in iov_column")
    {
        (WarningSeverity::Warning, "data_quality")
    } else if lower.starts_with("w_missing_dv") {
        (WarningSeverity::Warning, "data_quality")
    } else if lower.contains("ltbs")
        || lower.contains("non-positive dv")
        || lower.contains("ss=1 dose")
        || lower.contains("ss=1 infusion")
        || lower.contains("evid=3/4")
        || lower.contains("lagtime evaluates")
    {
        (WarningSeverity::Warning, "data_quality")
    } else if lower.contains("mixed lognormal") || lower.contains("mixed log-normal") {
        (WarningSeverity::Warning, "omega_structure")
    } else if lower.contains("hmc is unavailable") {
        // "falls back to" intentionally removed: no emitted message uses that exact
        // phrase. The SAEM HMC message is fully covered by "hmc is unavailable".
        (WarningSeverity::Info, "gradient_fallback")
    } else if lower.contains("not mu-referenced") {
        (WarningSeverity::Warning, "mu_referencing")
    } else if lower.contains("mu-ref") || lower.contains("mu-referencing") {
        (WarningSeverity::Info, "mu_referencing")
    } else if lower.contains("global_search disabled") {
        // Runtime failure: CRS2-LM init failed — the optimiser ran without global search.
        (WarningSeverity::Warning, "optimizer_config")
    } else if lower.contains("global_search") {
        (WarningSeverity::Info, "optimizer_config")
    } else if lower.contains("multi-start") {
        (WarningSeverity::Info, "multi_start")
    } else if lower.contains("cancelled by user") {
        (WarningSeverity::Info, "cancelled")
    } else if lower.contains("threads configured") || lower.contains("threads than subjects") {
        (WarningSeverity::Info, "threads")
    } else if lower.contains("n\u{00b2} ofv")
        || lower.contains("n^2 ofv")
        || (lower.contains("parameters") && lower.contains("covariance step:"))
    {
        (WarningSeverity::Info, "covariance_step")
    } else {
        (WarningSeverity::Warning, "general")
    };

    WarningEntry {
        severity,
        category: category.to_string(),
        message: msg,
        source_method,
    }
}

/// Full fit result
#[derive(Debug, Clone)]
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
    pub omega_iov: Option<DMatrix<f64>>,
    pub kappa_names: Vec<String>,
    pub kappa_fixed: Vec<bool>,
    /// Per-kappa SD-init flag (parallel to the `omega_iov` diagonal). Same
    /// semantics as `omega_init_as_sd` — `true` when the user wrote
    /// `kappa NAME ~ X (sd)`. Always `false` for block_kappa entries.
    pub kappa_init_as_sd: Vec<bool>,
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
    /// Model name (from the `.ferx` file or "Unnamed").
    pub model_name: String,
    /// ferx-core library version (from Cargo.toml at compile time).
    pub ferx_version: String,
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
    pub omega_param_corr: Option<DMatrix<f64>>,
    /// Parameter-level correlation matrix for IOV block kappa, analogous to
    /// `omega_param_corr`.  `None` when `omega_iov` is absent or diagonal.
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
    let n_lt = n_eta * (n_eta + 1) / 2;
    if se.len() == n_lt && n_lt != n_eta {
        // Full lower-triangle format (block omega).
        let col_offset = if c == 0 {
            0
        } else {
            c * n_eta - c * (c - 1) / 2
        };
        se.get(col_offset + (r - c)).copied()
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
    pub run_covariance_step: bool,
    /// *Initial* relative step size for the finite-difference Hessian in the
    /// covariance step. The actual step for parameter i is
    /// `fd_hessian_step * (1 + |x_hat[i]|)`. Default `1e-2`. ferx halves this
    /// automatically (up to 8×) if a diagonal stencil comes back non-finite, so
    /// manual tuning is rarely needed for overflow; decrease (e.g. `1e-3`) for
    /// smoother OFV surfaces where FD noise is the main concern.
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
    /// that gradient feeds the SLSQP overshoot cap (`cap_slsqp_gradient`), the
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
            // ODE solver tolerances: match OdeSolverOptions::default() so the
            // engine default is unchanged. Opt into tighter accuracy per model
            // via `[fit_options] ode_reltol = ...` (see FitOptions::ode_reltol).
            ode_reltol: 1e-4,
            ode_abstol: 1e-6,
            ode_max_steps: 10_000,
            run_covariance_step: true,
            fd_hessian_step: 1e-2,
            covariance_fallback: CovarianceFallback::None,
            covariance_method: CovarianceMethod::Hessian,
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

/// Estimation method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
        "optimizer_trace",
        "scale_params",
        "parameter_scaling",
        "max_unconverged_frac",
        "min_obs_for_convergence_check",
        "ebe_warm_start",
        "inits_from_nca",
        "frem_predictions",
        "frem_sigma",
        // RK45 ODE-solver knobs: applied to the integrator at parse time via
        // `sync_ode_solver_opts` (gated on the model being an ODE model, not on
        // the estimation method), so they are framework-level — every method
        // that integrates an ODE model honours them. Listing them here keeps
        // `unsupported_keys_warnings` from falsely flagging them as "ignored".
        "ode_reltol",
        "ode_abstol",
        "ode_max_steps",
    ]
}

/// Fit-option keys that are meaningful only for a particular estimation
/// method (or family of methods). `method` / `methods` are omitted — those
/// select the chain itself and can't be "wrong for the method". Framework-
/// wide keys live in `framework_keys`.
pub fn method_specific_keys(m: EstimationMethod) -> &'static [&'static str] {
    match m {
        EstimationMethod::Foce | EstimationMethod::FoceI => &[
            "maxiter",
            "inner_maxiter",
            "inner_tol",
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
            "inner_optimizer",
            "gn_lambda",
        ],
        EstimationMethod::FoceGnHybrid => &[
            "maxiter",
            "inner_maxiter",
            "inner_tol",
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
pub(crate) mod test_helpers {
    use super::*;
    use std::collections::HashMap;

    /// Build an analytical-PK model (`tv_fn = Some`, `ode_spec = None`).
    pub(crate) fn analytical_model(gradient_method: GradientMethod) -> CompiledModel {
        make_compiled_model(false, gradient_method)
    }

    /// Build an ODE-backed model (`tv_fn = None`, `ode_spec = Some`).
    pub(crate) fn ode_model(gradient_method: GradientMethod) -> CompiledModel {
        make_compiled_model(true, gradient_method)
    }

    fn make_compiled_model(with_ode: bool, gradient_method: GradientMethod) -> CompiledModel {
        CompiledModel {
            name: "test".into(),
            pk_model: PkModel::OneCptOral,
            error_model: ErrorModel::Additive,
            error_spec: ErrorSpec::Single(ErrorModel::Additive),
            residual_correlations: Vec::new(),
            pk_param_fn: Box::new(|_, _, _, _| PkParams::default()),
            n_theta: 1,
            n_eta: 1,
            n_epsilon: 1,
            n_kappa: 0,
            theta_names: vec!["CL".into()],
            eta_names: vec!["ETA_CL".into()],
            kappa_names: Vec::new(),
            indiv_param_names: vec!["CL".into()],
            indiv_param_partials: IndivParamPartials::empty(),
            default_params: ModelParameters {
                theta: vec![1.0],
                theta_names: vec!["CL".into()],
                theta_lower: vec![0.0],
                theta_upper: vec![f64::INFINITY],
                theta_fixed: vec![false],
                omega: OmegaMatrix::from_diagonal(&[0.1], vec!["ETA_CL".into()]),
                omega_fixed: vec![false],
                sigma: SigmaVector {
                    values: vec![0.1],
                    names: vec!["EPS".into()],
                },
                sigma_fixed: vec![false],
                omega_iov: None,
                kappa_fixed: Vec::new(),
            },
            omega_init_as_sd: vec![false],
            sigma_init_as_sd: vec![false],
            kappa_init_as_sd: Vec::new(),
            mu_refs: HashMap::new(),
            kappa_mu_refs: HashMap::new(),
            // Analytical models populate tv_fn; ODE models leave it None.
            tv_fn: if with_ode {
                None
            } else {
                Some(Box::new(|_t, _c| vec![1.0]))
            },
            pk_indices: vec![],
            eta_map: vec![],
            pk_idx_f64: vec![],
            sel_flat: vec![],
            diffusion_theta_start: None,
            diffusion_state_indices: Vec::new(),
            ode_spec: if with_ode {
                Some(crate::ode::OdeSpec {
                    rhs: Box::new(|_y, _p, _t, _dy| {}),
                    n_states: 2,
                    state_names: vec!["depot".into(), "central".into()],
                    readout: crate::ode::OdeReadout::ObsCmt(0),
                    diffusion_var: Vec::new(),
                    init_fn: None,
                    solver_opts: crate::ode::OdeSolverOptions::default(),
                    input_rate: Vec::new(),
                    rhs_program: None,
                    readout_program: None,
                    indiv_param_program: None,
                    dose_attr_map: Default::default(),
                })
            } else {
                None
            },
            dose_attr_map: Default::default(),
            bloq_method: BloqMethod::Drop,
            referenced_covariates: vec![],
            gradient_method,
            parse_warnings: Vec::new(),
            has_conditional_eta_params: false,
            eta_param_info: Vec::new(),
            theta_transform: Vec::new(),
            #[cfg(feature = "nn")]
            covariate_nns: Vec::new(),
            scaling: ScalingSpec::None,
            log_transform: false,
            dv_pre_logged: false,
            derived_exprs: vec![],
            output_columns: vec![],
            #[cfg(feature = "survival")]
            endpoints: HashMap::new(),
            frem_config: None,
            residual_error_eta: None,
            analytical_init: Vec::new(),
            analytic_readout: None,
            ruv_magnitude: None,
        }
    }

    /// A 1-cpt IV model with **proportional** error whose CL scales with the
    /// `WT` covariate (`CL = TVCL·WT/70`), paired with a subject whose `WT`
    /// rises across observation rows (70 → 140 → 210). Because `WT` is
    /// time-varying, the TV-aware predictor and the baseline-only
    /// `pk_param_fn(subject.covariates)` disagree on every obs row after the
    /// first — the fixture shared by the simulation / NPDE / NCA-sweep
    /// TV-covariate regression tests (#506). `n_eta = 0` so the only spread in
    /// simulated replicates is residual error, which keeps NPD deterministic.
    pub(crate) fn tv_cov_iv_model_and_subject() -> (CompiledModel, Subject) {
        let params = ModelParameters {
            theta: vec![5.0, 50.0],
            theta_names: vec!["TVCL".into(), "TVV".into()],
            theta_lower: vec![0.1, 5.0],
            theta_upper: vec![50.0, 500.0],
            theta_fixed: vec![false; 2],
            omega: OmegaMatrix::from_diagonal(&[], vec![]),
            omega_fixed: Vec::new(),
            sigma: SigmaVector {
                values: vec![0.05],
                names: vec!["PROP_ERR".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: None,
            kappa_fixed: Vec::new(),
        };
        let model = CompiledModel {
            name: "tv_cov_iv".into(),
            pk_model: PkModel::OneCptIv,
            error_model: ErrorModel::Proportional,
            error_spec: ErrorSpec::Single(ErrorModel::Proportional),
            residual_correlations: Vec::new(),
            pk_param_fn: Box::new(
                |theta: &[f64], _eta: &[f64], cov: &HashMap<String, f64>, _t: f64| {
                    let mut p = PkParams::default();
                    let wt = cov.get("WT").copied().unwrap_or(70.0);
                    p.values[0] = theta[0] * (wt / 70.0);
                    p.values[1] = theta[1];
                    p
                },
            ),
            n_theta: 2,
            n_eta: 0,
            n_epsilon: 1,
            n_kappa: 0,
            kappa_names: Vec::new(),
            theta_names: vec!["TVCL".into(), "TVV".into()],
            eta_names: Vec::new(),
            indiv_param_names: vec!["CL".into(), "V".into()],
            indiv_param_partials: IndivParamPartials::empty(),
            default_params: params.clone(),
            omega_init_as_sd: Vec::new(),
            sigma_init_as_sd: vec![false],
            kappa_init_as_sd: Vec::new(),
            mu_refs: HashMap::new(),
            kappa_mu_refs: HashMap::new(),
            tv_fn: None,
            pk_indices: vec![0, 1],
            eta_map: Vec::new(),
            pk_idx_f64: vec![0.0, 1.0],
            sel_flat: Vec::new(),
            ode_spec: None,
            dose_attr_map: Default::default(),
            diffusion_theta_start: None,
            diffusion_state_indices: Vec::new(),
            bloq_method: BloqMethod::Drop,
            referenced_covariates: vec!["WT".into()],
            gradient_method: GradientMethod::Fd,
            parse_warnings: Vec::new(),
            has_conditional_eta_params: false,
            eta_param_info: Vec::new(),
            theta_transform: Vec::new(),
            #[cfg(feature = "nn")]
            covariate_nns: Vec::new(),
            scaling: ScalingSpec::None,
            log_transform: false,
            dv_pre_logged: false,
            derived_exprs: vec![],
            output_columns: vec![],
            #[cfg(feature = "survival")]
            endpoints: HashMap::new(),
            frem_config: None,
            residual_error_eta: None,
            analytical_init: Vec::new(),
            analytic_readout: None,
            ruv_magnitude: None,
        };

        let mut baseline_cov = HashMap::new();
        baseline_cov.insert("WT".to_string(), 70.0);
        let subject = Subject {
            id: "TVCOV".to_string(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 2.0, 3.0],
            obs_raw_times: Vec::new(),
            observations: vec![0.0, 0.0, 0.0],
            obs_cmts: vec![1, 1, 1],
            covariates: baseline_cov,
            dose_covariates: vec![HashMap::from([("WT".to_string(), 70.0)])],
            obs_covariates: vec![
                HashMap::from([("WT".to_string(), 70.0)]),
                HashMap::from([("WT".to_string(), 140.0)]),
                HashMap::from([("WT".to_string(), 210.0)]),
            ],
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0, 0, 0],
            occasions: vec![1, 1, 1],
            dose_occasions: vec![1],
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        (model, subject)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Transit (`OneCptTransit`) descriptors: canonical name, the analytic 2-state
    /// `[depot, central]` layout, and no modeled-duration infusions (#386).
    #[test]
    fn one_cpt_transit_descriptors() {
        assert_eq!(PkModel::OneCptTransit.canonical_name(), "one_cpt_transit");
        assert!(PkModel::OneCptTransit.infusable_compartments().is_empty());
    }

    /// `sim_residual_variance` must split FREM covariate pseudo-observations
    /// (FREMTYPE>0) off the PK error model: they use the additive covariate
    /// sigma (EPSCOV) and ignore the IIV-on-RUV scaling, while ordinary PK rows
    /// use the PK error model scaled by `ruv_scale`. Regression for the FREM
    /// simulation corruption introduced when sim/NPDE moved onto the TV-cov
    /// dispatcher (which writes the θ+η override into FREM rows).
    #[test]
    fn sim_residual_variance_splits_frem_rows_from_pk_error() {
        let mut model = test_helpers::analytical_model(GradientMethod::Fd);
        // Proportional PK error: V_pk = (f·σ)².
        model.error_model = ErrorModel::Proportional;
        model.error_spec = ErrorSpec::Single(ErrorModel::Proportional);
        // FREMTYPE 100 maps to (θ[0], η[0]); the covariate sigma sits at index 1.
        model.frem_config = Some(FremConfig {
            fremtype_to_indices: std::collections::HashMap::from([(100u16, (0usize, 0usize))]),
            covariate_sigma_index: 1,
        });

        let subject = Subject {
            id: "S".into(),
            doses: vec![],
            obs_times: vec![1.0, 2.0],
            obs_raw_times: vec![],
            observations: vec![0.0, 0.0],
            obs_cmts: vec![1, 1],
            covariates: std::collections::HashMap::new(),
            dose_covariates: vec![],
            obs_covariates: vec![],
            pk_only_times: vec![],
            pk_only_covariates: vec![],
            reset_times: vec![],
            cens: vec![0, 0],
            occasions: vec![],
            dose_occasions: vec![],
            // Row 0 = PK observation, row 1 = covariate pseudo-observation.
            fremtype: vec![0, 100],
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };

        let sigma = vec![0.3, 0.5]; // [pk_prop_sigma, frem_cov_sigma]
        let ruv = 4.0; // exercise the IIV-on-RUV scaling on the PK row
        let f = 2.0;

        // PK row: (f·σ_pk)² · ruv_scale.
        let pk_var = model.sim_residual_variance(&subject, 0, f, &sigma, ruv, None);
        assert!((pk_var - (f * 0.3) * (f * 0.3) * ruv).abs() < 1e-12);

        // FREM row: covariate σ², independent of `f_pred` and `ruv_scale`.
        let frem_var = model.sim_residual_variance(&subject, 1, 999.0, &sigma, ruv, None);
        assert!((frem_var - 0.5 * 0.5).abs() < 1e-12);

        // Without a FREM config every row uses the PK error model.
        model.frem_config = None;
        let pk_only = model.sim_residual_variance(&subject, 1, f, &sigma, 1.0, None);
        assert!((pk_only - (f * 0.3) * (f * 0.3)).abs() < 1e-12);
    }

    #[test]
    fn sim_residual_variance_applies_custom_magnitude() {
        // #484 review #3: simulate() must draw residual error with the custom
        // magnitude, not a constant SD. A multiplier m on the proportional slot
        // scales the variance by m²; an all-ones / None multiplier is the legacy
        // variance exactly.
        let mut model = test_helpers::analytical_model(GradientMethod::Fd);
        model.error_model = ErrorModel::Proportional;
        model.error_spec = ErrorSpec::Single(ErrorModel::Proportional);
        model.frem_config = None;

        let subject = Subject {
            id: "S".into(),
            doses: vec![],
            obs_times: vec![1.0],
            obs_raw_times: vec![],
            observations: vec![0.0],
            obs_cmts: vec![1],
            covariates: std::collections::HashMap::new(),
            dose_covariates: vec![],
            obs_covariates: vec![],
            pk_only_times: vec![],
            pk_only_covariates: vec![],
            reset_times: vec![],
            cens: vec![0],
            occasions: vec![],
            dose_occasions: vec![],
            fremtype: vec![0],
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let sigma = vec![0.3];
        let f = 2.0;

        // Bare row: (f·σ)².
        let bare = model.sim_residual_variance(&subject, 0, f, &sigma, 1.0, None);
        assert!((bare - (f * 0.3) * (f * 0.3)).abs() < 1e-12);

        // Magnitude 2 on the proportional slot: (f·2·σ)² = 4·bare.
        let scaled = model.sim_residual_variance(&subject, 0, f, &sigma, 1.0, Some(&[2.0]));
        assert!((scaled - 4.0 * (f * 0.3) * (f * 0.3)).abs() < 1e-12);

        // An all-ones multiplier reproduces the bare variance.
        let unit = model.sim_residual_variance(&subject, 0, f, &sigma, 1.0, Some(&[1.0]));
        assert!((unit - bare).abs() < 1e-12);
    }

    #[test]
    fn resolve_auto_picks_nlopt_lbfgs_when_analytic() {
        // Analytical PK model with `gradient = auto` → exact FOCE gradient is
        // available, so `auto` resolves to the gradient-based NLopt L-BFGS (#490).
        let m = test_helpers::analytical_model(GradientMethod::Auto);
        assert_eq!(
            Optimizer::Auto.resolve_auto(&m, true),
            Optimizer::NloptLbfgs,
            "analytic-gradient model should resolve auto → nlopt_lbfgs"
        );
    }

    #[test]
    fn resolve_auto_picks_bobyqa_when_fd_only() {
        // ODE model (no analytic scope) and an analytical model with
        // `gradient = fd` both fall back to finite differences → bobyqa.
        let ode = test_helpers::ode_model(GradientMethod::Auto);
        let forced_fd = test_helpers::analytical_model(GradientMethod::Fd);
        for m in [&ode, &forced_fd] {
            assert_eq!(
                Optimizer::Auto.resolve_auto(m, true),
                Optimizer::Bobyqa,
                "FD-only model should resolve auto → bobyqa"
            );
        }
    }

    /// #486 σ-magnitude FOCE port: a custom residual-magnitude model is analytic on
    /// **both** loops now (the Sheiner–Beal FOCE marginal threads `mult(θ)` through
    /// its `R⁰`), so `auto` resolves to `NloptLbfgs` under plain `method = foce`
    /// (`interaction = false`) as well as under FOCEI (`interaction = true`).
    #[test]
    fn resolve_auto_analytic_for_custom_magnitude_both_loops() {
        let content = "[parameters]\n  theta TVCL(0.2)\n  theta TVV(10.0)\n  theta RUV_LATE(1.5, 0.0, 10.0)\n  omega ETA_CL ~ 0.09\n  sigma PROP_ERR ~ 0.04\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(PROP_ERR * (1.0 + RUV_LATE * TIME / 48.0))\n";
        let m = crate::parser::model_parser::parse_model_string(content).expect("parse");
        assert!(m.has_custom_ruv_magnitude());
        assert!(crate::sens::provider::analytic_outer_gradient_available(&m));
        assert_eq!(
            Optimizer::Auto.resolve_auto(&m, false),
            Optimizer::NloptLbfgs,
            "FOCE + custom magnitude now has the analytic gradient"
        );
        assert_eq!(
            Optimizer::Auto.resolve_auto(&m, true),
            Optimizer::NloptLbfgs,
            "FOCEI + custom magnitude has the analytic gradient"
        );
    }

    #[test]
    fn resolve_auto_is_identity_for_concrete_optimizers() {
        // Every non-`Auto` optimizer is returned unchanged, regardless of model.
        let m = test_helpers::analytical_model(GradientMethod::Auto);
        for opt in [
            Optimizer::Bfgs,
            Optimizer::Lbfgs,
            Optimizer::Slsqp,
            Optimizer::NloptLbfgs,
            Optimizer::Mma,
            Optimizer::Bobyqa,
            Optimizer::TrustRegion,
        ] {
            assert_eq!(opt.resolve_auto(&m, true), opt);
        }
    }

    #[test]
    fn required_pk_params_match_docs_table() {
        // Locks the per-model required slots to the "Required Parameters" table
        // in docs/model-file/structural-model.qmd (issue #309).
        use PkModel::*;
        let cases: &[(PkModel, &[&str])] = &[
            (OneCptIv, &["cl", "v"]),
            (OneCptOral, &["cl", "v", "ka"]),
            (TwoCptIv, &["cl", "v1", "q", "v2"]),
            (TwoCptOral, &["cl", "v1", "q", "v2", "ka"]),
            (ThreeCptIv, &["cl", "v1", "q2", "v2", "q3", "v3"]),
            (ThreeCptOral, &["cl", "v1", "q2", "v2", "q3", "v3", "ka"]),
        ];
        for (model, expected_names) in cases {
            let req = model.required_pk_params();
            let names: Vec<&str> = req.iter().map(|(_, n)| *n).collect();
            assert_eq!(&names, expected_names, "wrong required names for {model:?}");
            // Every (slot, name) pair must be self-consistent with name_to_index,
            // so the parser's key→slot canonicalisation lines up with this table.
            for (slot, name) in req {
                assert_eq!(
                    PkParams::name_to_index(name),
                    Some(*slot),
                    "slot/name mismatch for `{name}` in {model:?}"
                );
            }
            // f / lagtime are optional and must never appear as required.
            assert!(
                !req.iter()
                    .any(|(s, _)| *s == PK_IDX_F || *s == PK_IDX_LAGTIME),
                "{model:?} must not require f/lagtime"
            );
        }
    }

    #[test]
    fn canonical_name_covers_all_variants() {
        use PkModel::*;
        assert_eq!(OneCptIv.canonical_name(), "one_cpt_iv");
        assert_eq!(OneCptOral.canonical_name(), "one_cpt_oral");
        assert_eq!(TwoCptIv.canonical_name(), "two_cpt_iv");
        assert_eq!(TwoCptOral.canonical_name(), "two_cpt_oral");
        assert_eq!(ThreeCptIv.canonical_name(), "three_cpt_iv");
        assert_eq!(ThreeCptOral.canonical_name(), "three_cpt_oral");
    }

    #[test]
    fn from_name_round_trips_and_accepts_aliases() {
        use PkModel::*;
        // `from_name` is the single source of name → model for both the `pk` parser
        // and the `ode_template` desugarer (Ron #363). It must be the inverse of
        // `canonical_name` on the canonical spelling and accept every long-form
        // alias the parser historically accepted.
        let cases: &[(PkModel, &str)] = &[
            (OneCptIv, "one_compartment_iv"),
            (OneCptOral, "one_compartment_oral"),
            (TwoCptIv, "two_compartment_iv"),
            (TwoCptOral, "two_compartment_oral"),
            (ThreeCptIv, "three_compartment_iv"),
            (ThreeCptOral, "three_compartment_oral"),
        ];
        for (model, alias) in cases {
            assert_eq!(
                PkModel::from_name(model.canonical_name()),
                Some(*model),
                "canonical name of {model:?} must round-trip"
            );
            assert_eq!(
                PkModel::from_name(alias),
                Some(*model),
                "alias `{alias}` must resolve to {model:?}"
            );
        }
        // Unknown names and the retired `*_bolus` / `*_infusion` spellings do NOT
        // resolve — the parser maps the retired ones to a migration error itself,
        // so `from_name` must return `None` for them (not a wrong variant).
        for none in [
            "four_cpt_oral",
            "one_cpt_iv_bolus",
            "two_cpt_infusion",
            "",
            "pk",
        ] {
            assert_eq!(PkModel::from_name(none), None, "`{none}` must not resolve");
        }
    }

    #[test]
    fn error_spec_has_f_dependent_variance() {
        use std::collections::HashMap;
        // Single endpoint: only additive is f-independent.
        assert!(!ErrorSpec::Single(ErrorModel::Additive).has_f_dependent_variance());
        assert!(ErrorSpec::Single(ErrorModel::Proportional).has_f_dependent_variance());
        assert!(ErrorSpec::Single(ErrorModel::Combined).has_f_dependent_variance());

        // PerCmt: f-dependent if ANY endpoint is non-additive.
        let ep = |em: ErrorModel| EndpointError {
            error_model: em,
            sigma_idx: vec![0],
        };
        let mut all_additive = HashMap::new();
        all_additive.insert(1usize, ep(ErrorModel::Additive));
        all_additive.insert(2usize, ep(ErrorModel::Additive));
        assert!(!ErrorSpec::PerCmt(all_additive).has_f_dependent_variance());

        let mut mixed = HashMap::new();
        mixed.insert(1usize, ep(ErrorModel::Additive));
        mixed.insert(2usize, ep(ErrorModel::Proportional));
        assert!(ErrorSpec::PerCmt(mixed).has_f_dependent_variance());
    }

    #[test]
    fn combined_additive_sigma_indices_picks_second_slot() {
        use std::collections::HashMap;
        // Single combined: additive component is sigma index 1.
        assert_eq!(
            ErrorSpec::Single(ErrorModel::Combined).combined_additive_sigma_indices(),
            vec![1]
        );
        // Non-combined single specs have no additive-combined slot.
        assert!(ErrorSpec::Single(ErrorModel::Additive)
            .combined_additive_sigma_indices()
            .is_empty());
        assert!(ErrorSpec::Single(ErrorModel::Proportional)
            .combined_additive_sigma_indices()
            .is_empty());

        let ep = |em: ErrorModel, idx: Vec<usize>| EndpointError {
            error_model: em,
            sigma_idx: idx,
        };

        // PerCmt: returns the global index of each combined endpoint's
        // second sigma slot, de-duplicated; non-combined endpoints contribute
        // nothing.
        let mut map = HashMap::new();
        map.insert(1usize, ep(ErrorModel::Proportional, vec![0]));
        map.insert(2usize, ep(ErrorModel::Combined, vec![1, 3]));
        let mut got = ErrorSpec::PerCmt(map).combined_additive_sigma_indices();
        got.sort_unstable();
        assert_eq!(got, vec![3]);

        // Two combined endpoints that share the same additive sigma index
        // collapse to a single entry.
        let mut shared = HashMap::new();
        shared.insert(1usize, ep(ErrorModel::Combined, vec![0, 2]));
        shared.insert(2usize, ep(ErrorModel::Combined, vec![1, 2]));
        assert_eq!(
            ErrorSpec::PerCmt(shared).combined_additive_sigma_indices(),
            vec![2]
        );
    }

    #[test]
    fn classify_warning_convergence_is_critical() {
        let w = classify_warning("Outer optimization did not converge");
        assert_eq!(w.severity, WarningSeverity::Critical);
        assert_eq!(w.category, "convergence");
        assert!(w.source_method.is_none());
    }

    #[test]
    fn classify_warning_covariance_is_critical() {
        let w = classify_warning("Covariance step failed");
        assert_eq!(w.severity, WarningSeverity::Critical);
        assert_eq!(w.category, "covariance_step");
    }

    #[test]
    fn classify_warning_dw_is_warning() {
        let w = classify_warning("Positive IWRES autocorrelation detected (Durbin-Watson = 1.20).");
        assert_eq!(w.severity, WarningSeverity::Warning);
        assert_eq!(w.category, "dw_autocorrelation");
    }

    #[test]
    fn classify_warning_mu_ref_is_info() {
        let w = classify_warning("mu-ref: CL, V");
        assert_eq!(w.severity, WarningSeverity::Info);
        assert_eq!(w.category, "mu_referencing");
    }

    #[test]
    fn classify_warning_saem_non_mu_ref_is_warning() {
        let w = classify_warning(
            "SAEM: individual parameter(s) not mu-referenced: V. This can strongly \
             affect convergence; prefer forms such as `CL = TVCL * exp(ETA_CL)` when possible.",
        );
        assert_eq!(w.severity, WarningSeverity::Warning);
        assert_eq!(w.category, "mu_referencing");
    }

    #[test]
    fn classify_warning_strips_chain_prefix() {
        let w = classify_warning("[FOCEI] Covariance step failed");
        assert_eq!(w.source_method.as_deref(), Some("FOCEI"));
        assert_eq!(w.message, "Covariance step failed");
        assert_eq!(w.severity, WarningSeverity::Critical);
        assert_eq!(w.category, "covariance_step");
    }

    #[test]
    fn classify_warning_unknown_falls_back_to_general() {
        let w = classify_warning("some entirely novel message");
        assert_eq!(w.severity, WarningSeverity::Warning);
        assert_eq!(w.category, "general");
    }

    /// Round-trip table covering every literal warning message emitted by the
    /// engine, with the (severity, category) it should classify to.
    ///
    /// This protects the classifier contract from quiet regressions when a
    /// message gets a typo fix or a wording change. If you edit a message
    /// string in `outer_optimizer.rs`, `saem.rs`, `gauss_newton.rs`,
    /// `trust_region.rs`, or `api.rs`, mirror the edit here so the test
    /// keeps reflecting the engine's actual output.
    ///
    /// Messages built by `format!(..)` are exercised with a representative
    /// instantiation: the substrings the classifier matches on must remain
    /// present after interpolation.
    #[test]
    fn classify_warning_roundtrips_every_engine_message() {
        use WarningSeverity::*;
        let table: &[(&str, WarningSeverity, &str)] = &[
            // -- outer_optimizer.rs ----------------------------------------
            (
                "Outer optimization did not converge",
                Critical,
                "convergence",
            ),
            ("Covariance step failed", Critical, "covariance_step"),
            // global_search has two arms: explicit "disabled" is a runtime
            // failure (Warning); a bare mention without "disabled" is
            // informational.
            (
                "global_search disabled: bad seed",
                Warning,
                "optimizer_config",
            ),
            ("global_search reached eval cap", Info, "optimizer_config"),
            ("cancelled by user", Info, "cancelled"),
            // -- gauss_newton.rs -------------------------------------------
            (
                "Gauss-Newton: trust radius collapsed",
                Warning,
                "optimizer_health",
            ),
            (
                "Gauss-Newton: degenerate BHHH Hessian, trust radius collapsed",
                Warning,
                "optimizer_health",
            ),
            (
                "Gauss-Newton: max iterations reached without convergence",
                Critical,
                "convergence",
            ),
            // -- saem.rs ---------------------------------------------------
            (
                "Covariance step failed \u{2014} SEs not available",
                Critical,
                "covariance_step",
            ),
            (
                "saem_n_leapfrog > 0 but HMC is unavailable (requires an analytical PK model \
                 the Dual2 gradient supports); falling back to Metropolis-Hastings",
                Info,
                "gradient_fallback",
            ),
            // -- trust_region.rs ------------------------------------------
            (
                "Trust-region did not converge: line search stalled",
                Critical,
                "convergence",
            ),
            // -- api.rs ----------------------------------------------------
            (
                "Positive IWRES autocorrelation detected (Durbin-Watson = 1.20).",
                Warning,
                "dw_autocorrelation",
            ),
            (
                "Negative IWRES autocorrelation detected (Durbin-Watson = 2.80).",
                Warning,
                "dw_autocorrelation",
            ),
            (
                "M3 censoring under FOCE uses non-interaction (Sheiner–Beal) semantics",
                Warning,
                "bloq_method",
            ),
            (
                "SIR failed: covariance not positive definite",
                Warning,
                "sir",
            ),
            (
                "SIR requested but covariance matrix is not available",
                Warning,
                "sir",
            ),
            (
                "IMP: 2 subject(s) had ESS = 0 (proposal collapse)",
                Warning,
                "importance_sampling",
            ),
            (
                "LTBS (log(DV) ~ ...): 3 observation(s) with non-positive DV",
                Warning,
                "data_quality",
            ),
            (
                "W_MISSING_DV: 2 observation row(s) (EVID=0) had a missing DV",
                Warning,
                "data_quality",
            ),
            (
                "Stochastic differential equations ([diffusion] / Extended Kalman \
                 Filter) are an EXPERIMENTAL feature: validated only on a small set",
                Warning,
                "experimental",
            ),
            (
                "Neural-network model components ([covariate_nn] / deep compartment \
                 models) are an EXPERIMENTAL feature: validated only on a small set",
                Warning,
                "experimental",
            ),
            (
                "block omega: ETA_CL x ETA_V have mixed lognormal / additive parameterisations",
                Warning,
                "omega_structure",
            ),
            ("mu-ref: CL, V, KA", Info, "mu_referencing"),
            (
                "SAEM: individual parameter(s) not mu-referenced: V. This can strongly \
                 affect convergence; prefer forms such as `CL = TVCL * exp(ETA_CL)` when possible.",
                Warning,
                "mu_referencing",
            ),
            (
                "Multi-start: best result from start 3/8",
                Info,
                "multi_start",
            ),
            (
                "12 threads configured but only 10 subject(s)",
                Info,
                "threads",
            ),
            ("SAEM with more threads than subjects/2", Info, "threads"),
            (
                "Covariance step: 35 parameters \u{2192} n\u{00b2} OFV evaluations",
                Info,
                "covariance_step",
            ),
            (
                "Covariance step: 35 parameters -> n^2 OFV evaluations",
                Info,
                "covariance_step",
            ),
            // format_non_pd_warning path (NonPdHessian variant).
            (
                "Covariance step: Hessian is not positive definite. \
                 Eigenvalues: [8.4000, 2.1000, -0.0100]. SE estimates not available.",
                Critical,
                "covariance_step",
            ),
            // Covariance step regularised — present in all three severity tiers.
            (
                "Covariance step regularized: eigenvalue floor applied to FD Hessian \
                 (1 of 3 free-block eigenvalues clipped; min eig = 1.2e-6, floor = 8.4e-14; \
                 severity: minor). Standard errors are likely reliable.",
                Warning,
                "covariance_step",
            ),
            (
                "Covariance step regularized: eigenvalue floor applied to FD Hessian \
                 (3 of 5 free-block eigenvalues clipped; min eig = 1.2e-6, floor = 8.4e-14; \
                 severity: severe). Standard errors are likely unreliable; \
                 SIR-based confidence intervals are recommended.",
                Warning,
                "covariance_step",
            ),
            // Off-diagonal NaN soft warning — Success result, but correlation missing.
            (
                "Covariance step: off-diagonal FD stencil(s) non-finite for theta[CL], sigma[1]. \
                 Cross-partial correlation set to 0; SE for these parameter(s) \
                 may be over-optimistic. Try tuning fd_hessian_step.",
                Warning,
                "covariance_step",
            ),
            // Chain-prefixed off-diagonal warning.
            (
                "[FOCEI] Covariance step: off-diagonal FD stencil(s) non-finite for theta[CL]. \
                 Cross-partial correlation set to 0; SE for these parameter(s) \
                 may be over-optimistic. Try tuning fd_hessian_step.",
                Warning,
                "covariance_step",
            ),
            // Unusable messages introduced in commit 2.
            (
                "Covariance step failed: base OFV is non-finite at convergence \
                 (likely numerical overflow or underflow in model evaluation). \
                 SE estimates not available.",
                Critical,
                "covariance_step",
            ),
            (
                "Covariance step failed: Hessian has ill-conditioned entries for the \
                 following parameter(s) — theta[CL] (non-finite diagonal); \
                 sigma[1] (non-finite off-diagonal). SE estimates not available.",
                Critical,
                "covariance_step",
            ),
            // Omega near-singular (tiny positive eigenvalue) — "near-singular" descriptor.
            (
                "Covariance step failed: Omega matrix is near-singular at \
                 convergence (min eigenvalue = 1.2e-10; eigenvalues: [0.5000, 1.2e-10]). \
                 SE estimates not available.",
                Critical,
                "covariance_step",
            ),
            // Omega truly non-PD (negative eigenvalue) — "not positive definite" descriptor.
            (
                "Covariance step failed: Omega matrix is not positive definite at \
                 convergence (min eigenvalue = -1.0e-3; eigenvalues: [0.5000, -1.0e-3]). \
                 SE estimates not available.",
                Critical,
                "covariance_step",
            ),
            // SIR message that also contains "not positive definite" — must
            // still route to "sir", NOT to covariance_step.
            (
                "SIR failed: covariance not positive definite",
                Warning,
                "sir",
            ),
            // -- chain-prefixed (multi-stage) -----------------------------
            (
                "[FOCEI] Covariance step failed",
                Critical,
                "covariance_step",
            ),
            (
                "[FOCEI] Covariance step: Hessian is not positive definite. \
                 Eigenvalues: [2.1000, -0.0100]. SE estimates not available.",
                Critical,
                "covariance_step",
            ),
            (
                "[SAEM] individual parameter(s) not mu-referenced: V",
                Warning,
                "mu_referencing",
            ),
            (
                "[FOCEI] Outer optimization did not converge",
                Critical,
                "convergence",
            ),
        ];

        let mut failures: Vec<String> = Vec::new();
        for (msg, want_sev, want_cat) in table {
            let got = classify_warning(msg);
            if got.severity != *want_sev || got.category != *want_cat {
                failures.push(format!(
                    "  {msg:?} -> expected ({:?}, {:?}), got ({:?}, {:?})",
                    want_sev, want_cat, got.severity, got.category
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "classify_warning round-trip failures:\n{}",
            failures.join("\n")
        );
    }

    #[test]
    fn is_ode_based_false_for_analytical() {
        let m = test_helpers::analytical_model(GradientMethod::Auto);
        assert!(!m.is_ode_based());
    }

    #[test]
    fn is_ode_based_true_for_ode() {
        let m = test_helpers::ode_model(GradientMethod::Auto);
        assert!(m.is_ode_based());
    }

    #[test]
    fn has_bioavailability_detects_f_on_either_engine() {
        // Baseline test models declare no F → false on both engines.
        assert!(!test_helpers::analytical_model(GradientMethod::Auto).has_bioavailability());
        assert!(!test_helpers::ode_model(GradientMethod::Auto).has_bioavailability());

        // Analytical route: `f=` on `[structural_model]` puts PK_IDX_F in pk_indices.
        let mut m = test_helpers::analytical_model(GradientMethod::Auto);
        m.pk_indices = vec![PK_IDX_F];
        assert!(m.has_bioavailability());

        // Either engine: a bare `F` (any case) in `[individual_parameters]`.
        let mut m = test_helpers::analytical_model(GradientMethod::Auto);
        m.indiv_param_names = vec!["CL".into(), "f".into()];
        assert!(m.has_bioavailability());

        // ODE engine only: a compartment-indexed `Fn` routes via the DoseAttrMap.
        let mut m = test_helpers::ode_model(GradientMethod::Auto);
        m.indiv_param_names = vec!["CL".into(), "F1".into()];
        assert!(m.has_bioavailability());

        // The same `F1` on the analytical engine is not bioavailability (no ode_spec).
        let mut m = test_helpers::analytical_model(GradientMethod::Auto);
        m.indiv_param_names = vec!["CL".into(), "F1".into()];
        assert!(!m.has_bioavailability());
    }

    #[test]
    fn error_spec_single_ignores_cmt() {
        let spec = ErrorSpec::Single(ErrorModel::Proportional);
        // Proportional: V = (f * sigma)^2 = (10 * 0.1)^2 = 1.0, regardless of CMT.
        assert!((spec.variance_at(1, 10.0, &[0.1]) - 1.0).abs() < 1e-12);
        assert!((spec.variance_at(7, 10.0, &[0.1]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn error_spec_per_cmt_dispatches_model_and_sigma_slice() {
        // Flat sigma vector [prop_pk, add_pd]; CMT=2 uses idx 0 (proportional),
        // CMT=3 uses idx 1 (additive).
        let mut map = HashMap::new();
        map.insert(
            2,
            EndpointError {
                error_model: ErrorModel::Proportional,
                sigma_idx: vec![0],
            },
        );
        map.insert(
            3,
            EndpointError {
                error_model: ErrorModel::Additive,
                sigma_idx: vec![1],
            },
        );
        let spec = ErrorSpec::PerCmt(map);
        let sigma = [0.1, 2.0];

        // CMT=2 proportional: (10 * 0.1)^2 = 1.0 (independent of additive sigma).
        assert!((spec.variance_at(2, 10.0, &sigma) - 1.0).abs() < 1e-12);
        // CMT=3 additive: 2.0^2 = 4.0 (independent of prediction).
        assert!((spec.variance_at(3, 10.0, &sigma) - 4.0).abs() < 1e-12);
        assert!((spec.variance_at(3, 999.0, &sigma) - 4.0).abs() < 1e-12);

        // Unregistered CMT → NaN guard.
        assert!(spec.variance_at(1, 10.0, &sigma).is_nan());
    }

    #[test]
    fn error_spec_single_sigma_types_padded_to_n_sigma() {
        // Single proportional with extra declared sigmas: leading slot is
        // proportional, the rest default to additive, length == n_sigma.
        let spec = ErrorSpec::Single(ErrorModel::Proportional);
        assert_eq!(
            spec.sigma_types(3),
            vec![
                SigmaType::Proportional,
                SigmaType::Additive,
                SigmaType::Additive
            ]
        );
        // Combined stamps both leading slots.
        let combined = ErrorSpec::Single(ErrorModel::Combined);
        assert_eq!(
            combined.sigma_types(2),
            vec![SigmaType::Proportional, SigmaType::Additive]
        );
    }

    #[test]
    fn variance_at_scaled_unit_mult_matches_legacy() {
        // A unit multiplier must reproduce the plain variance bit-for-bit.
        let combined = ErrorSpec::Single(ErrorModel::Combined);
        let sigma = [0.3, 2.0];
        let f = 5.0;
        let base = combined.variance_at_with_correlations(1, f, &sigma, &[]);
        let scaled = combined.variance_at_scaled(1, f, &sigma, &[], &[1.0, 1.0]);
        assert_eq!(base, scaled);
    }

    #[test]
    fn variance_at_scaled_scales_proportional_loading() {
        // mult = [2, 1] on combined → (2·f·σ_p)² + σ_a².
        let combined = ErrorSpec::Single(ErrorModel::Combined);
        let sigma = [0.3, 2.0];
        let f = 5.0;
        let v = combined.variance_at_scaled(1, f, &sigma, &[], &[2.0, 1.0]);
        let expected = (2.0 * f * 0.3).powi(2) + 2.0_f64.powi(2);
        assert!((v - expected).abs() < 1e-12, "got {v}, want {expected}");
    }

    #[test]
    fn dvar_df_scaled_scales_by_mprop_squared() {
        // dvar_df scales by the proportional multiplier squared.
        let combined = ErrorSpec::Single(ErrorModel::Combined);
        let sigma = [0.3, 2.0];
        let f = 7.0;
        let base = combined.dvar_df(1, f, &sigma);
        let scaled = combined.dvar_df_scaled(1, f, &sigma, &[3.0, 1.0]);
        assert!((scaled - base * 9.0).abs() < 1e-12);
    }

    #[test]
    fn d2var_df2_scaled_scales_by_mprop_squared() {
        // d2var_df2 takes the same m² proportional-loading factor as dvar_df_scaled,
        // so the M3 censored curvature is built from a consistent v(f). A unit mult
        // reproduces the unscaled value; additive stays 0 at any mult.
        let combined = ErrorSpec::Single(ErrorModel::Combined);
        let sigma = [0.3, 2.0];
        let base = combined.d2var_df2(1, &sigma);
        assert!((combined.d2var_df2_scaled(1, &sigma, &[3.0, 1.0]) - base * 9.0).abs() < 1e-12);
        assert!((combined.d2var_df2_scaled(1, &sigma, &[1.0, 1.0]) - base).abs() < 1e-12);
        assert_eq!(
            ErrorSpec::Single(ErrorModel::Additive).d2var_df2_scaled(1, &[0.5], &[4.0]),
            0.0
        );
    }

    #[test]
    fn error_spec_per_cmt_variance_at_out_of_range_idx_is_nan() {
        // Hand-constructed endpoint whose sigma_idx points past the sigma slice
        // must yield NaN, not panic.
        let mut map = HashMap::new();
        map.insert(
            2,
            EndpointError {
                error_model: ErrorModel::Proportional,
                sigma_idx: vec![5], // out of range for a length-1 sigma vector
            },
        );
        let spec = ErrorSpec::PerCmt(map);
        assert!(spec.variance_at(2, 10.0, &[0.1]).is_nan());
    }

    #[test]
    fn error_spec_dvar_df_matches_legacy_single_formulas() {
        let f = 7.0;
        // Additive: 0. Proportional/Combined: 2*f*sigma_prop^2.
        assert_eq!(
            ErrorSpec::Single(ErrorModel::Additive).dvar_df(1, f, &[0.5]),
            0.0
        );
        assert!(
            (ErrorSpec::Single(ErrorModel::Proportional).dvar_df(1, f, &[0.3]) - 2.0 * f * 0.09)
                .abs()
                < 1e-12
        );
        // Combined uses the first (proportional) sigma.
        assert!(
            (ErrorSpec::Single(ErrorModel::Combined).dvar_df(1, f, &[0.3, 2.0]) - 2.0 * f * 0.09)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn error_spec_per_cmt_dvar_df_and_dlogsigma_dispatch_by_endpoint() {
        // CMT=2 proportional on sigma idx 0; CMT=3 additive on sigma idx 1.
        let mut map = HashMap::new();
        map.insert(
            2,
            EndpointError {
                error_model: ErrorModel::Proportional,
                sigma_idx: vec![0],
            },
        );
        map.insert(
            3,
            EndpointError {
                error_model: ErrorModel::Additive,
                sigma_idx: vec![1],
            },
        );
        let spec = ErrorSpec::PerCmt(map);
        let sigma = [0.3, 2.0];
        let f = 7.0;

        // dvar_df: proportional endpoint -> 2*f*0.3^2; additive -> 0.
        assert!((spec.dvar_df(2, f, &sigma) - 2.0 * f * 0.09).abs() < 1e-12);
        assert_eq!(spec.dvar_df(3, f, &sigma), 0.0);

        // dvar_dlogsigma: each sigma only contributes through its own endpoint.
        // sigma 0 (proportional, CMT=2): 2*0.3^2*f^2 at CMT=2, 0 at CMT=3.
        assert!((spec.dvar_dlogsigma(2, 0, f, &sigma) - 2.0 * 0.09 * f * f).abs() < 1e-12);
        assert_eq!(spec.dvar_dlogsigma(3, 0, f, &sigma), 0.0);
        // sigma 1 (additive, CMT=3): 2*2.0^2 at CMT=3, 0 at CMT=2.
        assert!((spec.dvar_dlogsigma(3, 1, f, &sigma) - 2.0 * 4.0).abs() < 1e-12);
        assert_eq!(spec.dvar_dlogsigma(2, 1, f, &sigma), 0.0);
    }

    #[test]
    fn error_spec_per_cmt_sigma_types_maps_each_slot() {
        let mut map = HashMap::new();
        map.insert(
            2,
            EndpointError {
                error_model: ErrorModel::Proportional,
                sigma_idx: vec![0],
            },
        );
        map.insert(
            3,
            EndpointError {
                error_model: ErrorModel::Additive,
                sigma_idx: vec![1],
            },
        );
        let spec = ErrorSpec::PerCmt(map);
        assert_eq!(
            spec.sigma_types(2),
            vec![SigmaType::Proportional, SigmaType::Additive]
        );
    }

    #[test]
    fn test_lagtime_name_to_index_and_default() {
        assert_eq!(PkParams::name_to_index("lagtime"), Some(PK_IDX_LAGTIME));
        // NONMEM-style alias maps to the same slot.
        assert_eq!(PkParams::name_to_index("alag"), Some(PK_IDX_LAGTIME));
        assert_eq!(PK_IDX_LAGTIME, 8);
        // Slots 0–8 are named PK params; the rest are ODE structural-param
        // headroom (see MAX_PK_PARAMS docs / issue #122).
        assert!(MAX_PK_PARAMS > PK_IDX_LAGTIME);

        let default = PkParams::default();
        assert_eq!(default.lagtime(), 0.0);
        // F still defaults to 1.0 (unchanged).
        assert_eq!(default.f_bio(), 1.0);
    }

    #[test]
    fn dose_attr_map_resolves_indexed_then_bare_then_default() {
        // params: slot 5 = bare F, slot 8 = bare lag, slots 9/10 = F2/ALAG2.
        let mut params = [0.0f64; MAX_PK_PARAMS];
        params[PK_IDX_F] = 0.8; // bare F
        params[PK_IDX_LAGTIME] = 0.5; // bare lag
        params[9] = 0.3; // F for compartment 2
        params[10] = 1.25; // ALAG for compartment 2

        let mut map = DoseAttrMap::default();
        map.insert(DoseAttr::F, 2, 9);
        map.insert(DoseAttr::Lag, 2, 10);

        // Compartment 1 has no indexed entry -> falls through to the bare slot.
        assert_eq!(map.f_bio(1, &params), 0.8);
        assert_eq!(map.lagtime(1, &params), 0.5);
        // Compartment 2 is overridden by its indexed slot.
        assert_eq!(map.f_bio(2, &params), 0.3);
        assert_eq!(map.lagtime(2, &params), 1.25);
        // indexed_slot exposes the raw mapping (used by upstream validation).
        assert_eq!(map.indexed_slot(DoseAttr::F, 2), Some(9));
        assert_eq!(map.indexed_slot(DoseAttr::F, 1), None);
    }

    #[test]
    fn dose_attr_from_indexed_name_recognizes_f_lag_duration_and_rate() {
        use DoseAttr::*;
        // Bioavailability and both lag spellings, case-insensitive.
        assert_eq!(DoseAttr::from_indexed_name("F1"), Some((F, 1)));
        assert_eq!(DoseAttr::from_indexed_name("f2"), Some((F, 2)));
        assert_eq!(DoseAttr::from_indexed_name("ALAG1"), Some((Lag, 1)));
        assert_eq!(DoseAttr::from_indexed_name("alag3"), Some((Lag, 3)));
        assert_eq!(DoseAttr::from_indexed_name("LAGTIME2"), Some((Lag, 2)));
        // Modeled infusion duration D{n} (RATE=-2; #324), case-insensitive.
        assert_eq!(DoseAttr::from_indexed_name("D1"), Some((Duration, 1)));
        assert_eq!(DoseAttr::from_indexed_name("d2"), Some((Duration, 2)));
        // Modeled infusion rate R{n} (RATE=-1; #324), case-insensitive.
        assert_eq!(DoseAttr::from_indexed_name("R1"), Some((Rate, 1)));
        assert_eq!(DoseAttr::from_indexed_name("r2"), Some((Rate, 2)));

        // Bare forms and zero index are not compartment-indexed attributes.
        assert_eq!(DoseAttr::from_indexed_name("F"), None);
        assert_eq!(DoseAttr::from_indexed_name("lagtime"), None);
        assert_eq!(DoseAttr::from_indexed_name("alag"), None);
        assert_eq!(DoseAttr::from_indexed_name("F0"), None);
        assert_eq!(DoseAttr::from_indexed_name("D0"), None);
        assert_eq!(DoseAttr::from_indexed_name("D"), None);
        assert_eq!(DoseAttr::from_indexed_name("R0"), None);
        assert_eq!(DoseAttr::from_indexed_name("R"), None);

        // Must not capture canonical PK names, the [scaling] `S{n}` names, or
        // non-numeric suffixes — including `D`/`R`-prefixed words (`delta`,
        // `rate`, `RUV`, `R2D2`).
        for n in [
            "CL", "V1", "V2", "Q2", "Q3", "KA", "S1", "S2", "f_bio", "rate", "delta", "decay",
            "RUV", "R2D2",
        ] {
            assert_eq!(DoseAttr::from_indexed_name(n), None, "{n} must not match");
        }

        // An all-digit suffix that overflows usize fails to parse -> not an
        // attribute (exercises the parse-error guard, not just the success path).
        assert_eq!(
            DoseAttr::from_indexed_name(&format!("R{}", "9".repeat(40))),
            None
        );
    }

    #[test]
    fn dose_attr_map_empty_yields_engine_defaults() {
        // An empty map (the common bare-/single-route model) must reproduce the
        // pre-existing defaults: F = 1.0, lag = 0.0, even for a params slice too
        // short to hold the reserved slots (cannot panic).
        let map = DoseAttrMap::default();
        let full = PkParams::default().values; // F slot already 1.0
        assert_eq!(map.f_bio(1, &full), 1.0);
        assert_eq!(map.lagtime(1, &full), 0.0);

        let short: [f64; 3] = [2.0, 3.0, 4.0];
        assert_eq!(map.f_bio(1, &short), 1.0);
        assert_eq!(map.lagtime(1, &short), 0.0);
        // Duration/Rate have no bare fallback -> no indexed slot means None.
        assert_eq!(map.indexed_slot(DoseAttr::Duration, 1), None);
        assert_eq!(map.indexed_slot(DoseAttr::Rate, 1), None);
    }

    #[test]
    fn dose_event_resolve_rate_modeled_duration_matches_explicit_infusion() {
        // RATE=-2 with D{1} in slot 9: a 100-unit dose over D = 5 must resolve to
        // the same (rate, duration) as an explicit RATE = 100/5 = 20 infusion.
        let mut map = DoseAttrMap::default();
        map.insert(DoseAttr::Duration, 1, 9);
        let mut params = [0.0; MAX_PK_PARAMS];
        params[9] = 5.0; // D1

        let modeled = DoseEvent::modeled(0.0, 100.0, 1, false, 0.0, RateMode::ModeledDuration);
        assert!(
            modeled.is_infusion(),
            "modeled dose is an infusion pre-resolve"
        );
        let resolved = modeled.resolve_rate(&map, &params);

        assert_eq!(resolved.rate_mode, RateMode::Fixed);
        assert_eq!(resolved.duration, 5.0);
        assert_eq!(resolved.rate, 20.0);
        // Bit-equal to the hand-written explicit infusion (the #324 invariant).
        let explicit = DoseEvent::new(0.0, 100.0, 1, 20.0, false, 0.0);
        assert_eq!(resolved.rate, explicit.rate);
        assert_eq!(resolved.duration, explicit.duration);
        // cmt/time/amt/ss/ii are preserved through resolution.
        assert_eq!(resolved.cmt, 1);
        assert_eq!(resolved.amt, 100.0);

        // A Fixed dose is returned unchanged (the common, allocation-cheap path).
        let same = explicit.resolve_rate(&map, &params);
        assert_eq!(
            (same.rate, same.duration, same.rate_mode),
            (20.0, 5.0, RateMode::Fixed)
        );
    }

    #[test]
    fn dose_event_resolve_rate_clamps_nonpositive_duration() {
        // A transient D <= 0 (or NaN) mid-search clamps to DURATION_FLOOR so
        // rate = amt / D stays finite (mirrors PreparedInputRate::MIN_PARAM).
        let mut map = DoseAttrMap::default();
        map.insert(DoseAttr::Duration, 1, 9);
        let modeled = DoseEvent::modeled(0.0, 100.0, 1, false, 0.0, RateMode::ModeledDuration);

        for bad in [0.0, -3.0, f64::NAN] {
            let mut params = [0.0; MAX_PK_PARAMS];
            params[9] = bad;
            let r = modeled.resolve_rate(&map, &params);
            assert_eq!(r.duration, DoseEvent::DURATION_FLOOR, "clamp at floor");
            assert!(r.rate.is_finite() && r.rate > 0.0, "rate finite");
        }
    }

    #[test]
    fn dose_event_resolve_rate_modeled_rate_matches_explicit_infusion() {
        // RATE=-1 with R{1} in slot 9: a 100-unit dose at R = 20 must resolve to
        // the same (rate, duration) as an explicit RATE = 20 infusion (the
        // #324 invariant, mirror of the modeled-duration case).
        let mut map = DoseAttrMap::default();
        map.insert(DoseAttr::Rate, 1, 9);
        let mut params = [0.0; MAX_PK_PARAMS];
        params[9] = 20.0; // R1

        let modeled = DoseEvent::modeled(0.0, 100.0, 1, false, 0.0, RateMode::ModeledRate);
        assert!(
            modeled.is_infusion(),
            "modeled-rate dose is an infusion pre-resolve"
        );
        let resolved = modeled.resolve_rate(&map, &params);

        assert_eq!(resolved.rate_mode, RateMode::Fixed);
        assert_eq!(resolved.rate, 20.0);
        // amt / rate = 100 / 20; bit-equal to the hand-written explicit infusion.
        assert_eq!(resolved.duration, 5.0);
        let explicit = DoseEvent::new(0.0, 100.0, 1, 20.0, false, 0.0);
        assert_eq!(resolved.rate, explicit.rate);
        assert_eq!(resolved.duration, explicit.duration);
        assert_eq!(resolved.cmt, 1);
        assert_eq!(resolved.amt, 100.0);
    }

    #[test]
    fn dose_event_resolve_rate_clamps_nonpositive_rate() {
        // A transient R <= 0 (or NaN) mid-search clamps to RATE_FLOOR so the
        // implied duration = amt / R stays finite (mirror of the duration clamp).
        let mut map = DoseAttrMap::default();
        map.insert(DoseAttr::Rate, 1, 9);
        let modeled = DoseEvent::modeled(0.0, 100.0, 1, false, 0.0, RateMode::ModeledRate);

        for bad in [0.0, -3.0, f64::NAN] {
            let mut params = [0.0; MAX_PK_PARAMS];
            params[9] = bad;
            let r = modeled.resolve_rate(&map, &params);
            assert_eq!(r.rate, DoseEvent::RATE_FLOOR, "clamp at floor");
            assert!(
                r.duration.is_finite() && r.duration > 0.0,
                "duration finite"
            );
        }
    }

    #[test]
    fn clamp_above_floor_passes_through_or_clamps() {
        // The shared clamp behind DURATION_FLOOR / MIN_PARAM: > floor passes through,
        // <= floor (and NaN — every `>` is false for NaN) returns the floor.
        let floor = 1e-8;
        assert_eq!(clamp_above_floor(5.0, floor), 5.0, "above floor passes");
        assert_eq!(
            clamp_above_floor(2e-8, floor),
            2e-8,
            "just above floor passes"
        );
        assert_eq!(
            clamp_above_floor(floor, floor),
            floor,
            "at floor → floor (not >)"
        );
        assert_eq!(clamp_above_floor(0.0, floor), floor, "zero clamps to floor");
        assert_eq!(
            clamp_above_floor(-3.0, floor),
            floor,
            "negative clamps to floor"
        );
        assert_eq!(
            clamp_above_floor(f64::NAN, floor),
            floor,
            "NaN clamps to floor"
        );
    }

    #[test]
    fn test_lagtime_from_hashmap_primary_and_alias() {
        let mut m = HashMap::new();
        m.insert("lagtime".to_string(), 1.5);
        let p = PkParams::from_hashmap(&m);
        assert_eq!(p.lagtime(), 1.5);

        let mut m_alias = HashMap::new();
        m_alias.insert("alag".to_string(), 2.0);
        let p_alias = PkParams::from_hashmap(&m_alias);
        assert_eq!(p_alias.lagtime(), 2.0);
    }

    /// Guard the SAEM MH-step default. An early value (3) was too low for hard
    /// cold-start surfaces — the chain didn't decorrelate between SAEM outer
    /// iterations, so the single-draw stochastic M-step received sticky
    /// correlated ETAs and locked the population-θ M-step into a degenerate
    /// basin (observed on Emax PKPD: PD-curve thetas pinned to boundary, ~150
    /// OFV units worse than the correct basin). The default was raised to 10,
    /// then to 20 alongside the componentwise eta kernel and the damped Ω
    /// stochastic-approximation step — both added to stop a block (correlated)
    /// Ω collapsing to a near rank-1 correlation matrix (UVM 2-cpt: every
    /// off-diagonal correlation → ~0.99, one variance → 0). The larger default
    /// also sizes the componentwise sweep count (`max(2, n_mh_steps / n_eta)`).
    ///
    /// If a future change drops the default below ~5, re-run both the Emax PKPD
    /// basin regression and the UVM block-Ω collapse regression in the
    /// experiment repo before merging — both fail silently (OFV looks fine;
    /// parameters wrong).
    #[test]
    fn saem_n_mh_steps_default_is_20() {
        let opts = FitOptions::default();
        assert_eq!(
            opts.saem_n_mh_steps, 20,
            "saem_n_mh_steps default changed — see comment above this test \
             for the basin-trap and block-Ω-collapse regression rationale \
             before adjusting."
        );
    }

    #[test]
    fn impmap_proposal_df_default_is_finite_t() {
        // IMPMAP defaults to a Student-t proposal (df = 4), not a Gaussian: the
        // MVN tails are too light for weakly-identified posteriors and bias the
        // M-step moments (absorption-param drift, #411). Do not revert to ∞
        // without re-checking that regression.
        let opts = FitOptions::default();
        assert_eq!(opts.impmap_proposal_df, 4.0);
        assert!(opts.impmap_proposal_df.is_finite());
    }

    // ── small pure helpers ───────────────────────────────────────────────────

    /// Bare subject with no doses/observations; tests mutate the fields they
    /// exercise.
    fn bare_subject(id: &str) -> Subject {
        Subject {
            id: id.to_string(),
            doses: Vec::new(),
            obs_times: Vec::new(),
            obs_raw_times: Vec::new(),
            observations: Vec::new(),
            obs_cmts: Vec::new(),
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: Vec::new(),
            occasions: Vec::new(),
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: Vec::new(),
        }
    }

    fn dose(ss: bool) -> DoseEvent {
        DoseEvent::new(0.0, 100.0, 1, 0.0, ss, 0.0)
    }

    #[test]
    fn subject_has_censored_observation_reflects_cens_flags() {
        let mut s = bare_subject("1");
        s.cens = vec![0, 0];
        assert!(!s.has_censored_observation());
        s.cens = vec![0, 1];
        assert!(s.has_censored_observation());
        s.cens = vec![-1, 0];
        assert!(s.has_censored_observation());
    }

    #[test]
    fn subject_has_ss_doses_detects_steady_state_flag() {
        let mut s = bare_subject("1");
        s.doses = vec![dose(false), dose(false)];
        assert!(!s.has_ss_doses());
        s.doses.push(dose(true));
        assert!(s.has_ss_doses());
    }

    #[test]
    fn subject_has_periodic_ss_dose_requires_ss_flag_and_positive_ii() {
        let mut s = bare_subject("1");
        // No doses → false.
        assert!(!s.has_periodic_ss_dose());
        // SS flag but `II = 0` (not a periodic SS dose) → false, unlike
        // `has_ss_doses`, which keys on the flag alone.
        s.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 0.0)];
        assert!(s.has_ss_doses());
        assert!(!s.has_periodic_ss_dose());
        // Periodic SS dose (`SS=1`, `II > 0`) → true.
        s.doses.push(DoseEvent::new(0.0, 100.0, 1, 0.0, true, 24.0));
        assert!(s.has_periodic_ss_dose());
        // `II > 0` without the SS flag → false.
        s.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 24.0)];
        assert!(!s.has_periodic_ss_dose());
    }

    #[test]
    fn subject_has_rate_defined_infusion_distinguishes_infusion_modes() {
        let mut s = bare_subject("1");
        // No doses → false.
        assert!(!s.has_rate_defined_infusion());

        // Bolus (RATE=0) is not an infusion → false.
        s.doses = vec![dose(false)];
        assert!(!s.has_rate_defined_infusion());

        // Duration-defined infusion (RATE=-2 → D{cmt}): it *is* an infusion, but F
        // scales its rate, not its window, so it must not count (#419).
        s.doses = vec![DoseEvent::modeled(
            0.0,
            100.0,
            1,
            false,
            0.0,
            RateMode::ModeledDuration,
        )];
        assert!(s.doses[0].is_infusion());
        assert!(!s.has_rate_defined_infusion());

        // Rate-defined infusion (RATE>0 data) → true.
        s.doses = vec![DoseEvent::new(0.0, 100.0, 1, 10.0, false, 0.0)];
        assert!(s.has_rate_defined_infusion());

        // A rate-defined infusion alongside a bolus still trips it.
        s.doses = vec![dose(false), DoseEvent::new(0.0, 100.0, 1, 10.0, false, 0.0)];
        assert!(s.has_rate_defined_infusion());
    }

    #[test]
    fn subject_pk_only_cov_falls_back_to_static_map() {
        let mut s = bare_subject("1");
        s.covariates.insert("WT".to_string(), 70.0);
        // No per-EVID-2 snapshots → static map.
        assert_eq!(s.pk_only_cov(0).get("WT"), Some(&70.0));
        // With a snapshot at index 0 → that snapshot.
        let mut snap = HashMap::new();
        snap.insert("WT".to_string(), 80.0);
        s.pk_only_covariates = vec![snap];
        assert_eq!(s.pk_only_cov(0).get("WT"), Some(&80.0));
        // Out-of-range index → fall back to static map.
        assert_eq!(s.pk_only_cov(5).get("WT"), Some(&70.0));
    }

    #[test]
    fn prune_irrelevant_tv_covariates_clears_only_constant_referenced_covs() {
        // Subject A: the referenced covariate (WT) is constant across snapshots
        // but an unreferenced one (DAY) varies → snapshots should be pruned.
        let mut a = bare_subject("A");
        a.covariates = HashMap::from([("WT".to_string(), 70.0), ("DAY".to_string(), 1.0)]);
        a.obs_covariates = vec![
            HashMap::from([("WT".to_string(), 70.0), ("DAY".to_string(), 1.0)]),
            HashMap::from([("WT".to_string(), 70.0), ("DAY".to_string(), 2.0)]),
        ];
        // Subject B: the referenced covariate (WT) genuinely varies → keep.
        let mut b = bare_subject("B");
        b.covariates = HashMap::from([("WT".to_string(), 70.0)]);
        b.obs_covariates = vec![
            HashMap::from([("WT".to_string(), 70.0)]),
            HashMap::from([("WT".to_string(), 90.0)]),
        ];

        let mut pop = Population {
            subjects: vec![a, b],
            covariate_names: vec!["WT".to_string(), "DAY".to_string()],
            dv_column: "DV".to_string(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        };
        let pruned = pop.prune_irrelevant_tv_covariates(&["WT".to_string()]);
        assert_eq!(pruned, 1, "only subject A should be pruned");
        assert!(
            pop.subjects[0].obs_covariates.is_empty(),
            "A pruned to fast path"
        );
        assert!(
            !pop.subjects[1].obs_covariates.is_empty(),
            "B keeps TV snapshots"
        );
    }

    #[test]
    fn covariate_kind_label_strings() {
        assert_eq!(CovariateKind::Continuous.label(), "continuous");
        assert_eq!(CovariateKind::Categorical.label(), "categorical");
    }

    #[test]
    fn scaling_spec_requires_fd_is_always_false() {
        assert!(!ScalingSpec::None.requires_fd());
        assert!(!ScalingSpec::ScalarScale(1000.0).requires_fd());
        assert!(!ScalingSpec::PerCmt(HashMap::new()).requires_fd());
    }

    #[test]
    fn estimation_method_label_covers_all_variants() {
        assert_eq!(EstimationMethod::Foce.label(), "FOCE");
        assert_eq!(EstimationMethod::FoceI.label(), "FOCEI");
        assert_eq!(EstimationMethod::FoceGn.label(), "FOCE-GN");
        assert_eq!(EstimationMethod::FoceGnHybrid.label(), "FOCE-GN-Hybrid");
        assert_eq!(EstimationMethod::Saem.label(), "SAEM");
        assert_eq!(EstimationMethod::Imp.label(), "IMP");
    }

    #[test]
    fn model_parameters_has_any_fixed_tracks_every_flag_vec() {
        let mut mp = test_helpers::analytical_model(GradientMethod::Fd).default_params;
        mp.theta_fixed = vec![false];
        mp.omega_fixed = vec![false];
        mp.sigma_fixed = vec![false];
        mp.kappa_fixed = Vec::new();
        assert!(!mp.has_any_fixed());
        mp.sigma_fixed = vec![true];
        assert!(mp.has_any_fixed());
    }

    #[test]
    fn pk_params_from_hashmap_maps_named_fields_and_aliases() {
        let map = HashMap::from([
            ("cl".to_string(), 5.0),
            ("v1".to_string(), 30.0), // v1 alias → V index
            ("q".to_string(), 2.0),
            ("v2".to_string(), 50.0),
            ("ka".to_string(), 1.2),
            ("f".to_string(), 0.8),
            ("q3".to_string(), 0.5),
            ("v3".to_string(), 100.0),
        ]);
        let p = PkParams::from_hashmap(&map);
        assert_eq!(p.values[PK_IDX_CL], 5.0);
        assert_eq!(p.values[PK_IDX_V], 30.0);
        assert_eq!(p.values[PK_IDX_Q], 2.0);
        assert_eq!(p.values[PK_IDX_V2], 50.0);
        assert_eq!(p.values[PK_IDX_KA], 1.2);
        assert_eq!(p.values[PK_IDX_F], 0.8);
        assert_eq!(p.values[PK_IDX_Q3], 0.5);
        assert_eq!(p.values[PK_IDX_V3], 100.0);
    }

    // ── block_sigma / correlated residual error (PR #545) ──────────────────

    #[test]
    fn sigma_loadings_single_matches_error_model_shape() {
        // Additive loads only the additive sigma (coeff 1); proportional loads
        // the proportional sigma (coeff f); combined loads both, in the
        // (prop, add) order that `residual_variance` assumes positionally.
        let add = ErrorSpec::Single(ErrorModel::Additive);
        assert_eq!(add.sigma_loadings(0, 5.0, 1), vec![(0, 1.0)]);

        let prop = ErrorSpec::Single(ErrorModel::Proportional);
        assert_eq!(prop.sigma_loadings(0, 5.0, 1), vec![(0, 5.0)]);

        let comb = ErrorSpec::Single(ErrorModel::Combined);
        assert_eq!(comb.sigma_loadings(0, 5.0, 2), vec![(0, 5.0), (1, 1.0)]);
    }

    #[test]
    fn sigma_loadings_single_respects_n_sigma_guards() {
        // With no sigmas available the loadings are empty; a 1-sigma combined
        // model exposes only the proportional loading.
        let add = ErrorSpec::Single(ErrorModel::Additive);
        assert!(add.sigma_loadings(0, 5.0, 0).is_empty());
        let prop = ErrorSpec::Single(ErrorModel::Proportional);
        assert!(prop.sigma_loadings(0, 5.0, 0).is_empty());
        let comb = ErrorSpec::Single(ErrorModel::Combined);
        assert_eq!(comb.sigma_loadings(0, 5.0, 1), vec![(0, 5.0)]);
    }

    #[test]
    fn sigma_loadings_per_cmt_dispatches_by_endpoint() {
        // Endpoint in cmt 1 is combined over global sigma indices (2, 3);
        // cmt 2 is additive over index 0. Loadings must use those global
        // indices, and an unmapped compartment yields no loadings.
        let mut map = std::collections::HashMap::new();
        map.insert(
            1usize,
            EndpointError {
                error_model: ErrorModel::Combined,
                sigma_idx: vec![2, 3],
            },
        );
        map.insert(
            2usize,
            EndpointError {
                error_model: ErrorModel::Additive,
                sigma_idx: vec![0],
            },
        );
        let spec = ErrorSpec::PerCmt(map);
        assert_eq!(spec.sigma_loadings(1, 7.0, 4), vec![(2, 7.0), (3, 1.0)]);
        assert_eq!(spec.sigma_loadings(2, 7.0, 4), vec![(0, 1.0)]);
        assert!(spec.sigma_loadings(9, 7.0, 4).is_empty());
    }

    #[test]
    fn sigma_loading_slopes_match_loading_shape_with_f_derivatives() {
        // Slopes have the SAME slot presence as `sigma_loadings` but carry
        // ∂coeff/∂f: 0 for the additive (constant) slot, 1 for the proportional
        // slot. This keeps the dense-R ∂R/∂f cross-covariance assembly consistent.
        let add = ErrorSpec::Single(ErrorModel::Additive);
        assert_eq!(add.sigma_loading_slopes(0, 1), vec![(0, 0.0)]);
        assert!(add.sigma_loading_slopes(0, 0).is_empty());

        let prop = ErrorSpec::Single(ErrorModel::Proportional);
        assert_eq!(prop.sigma_loading_slopes(0, 1), vec![(0, 1.0)]);
        assert!(prop.sigma_loading_slopes(0, 0).is_empty());

        let comb = ErrorSpec::Single(ErrorModel::Combined);
        assert_eq!(comb.sigma_loading_slopes(0, 2), vec![(0, 1.0), (1, 0.0)]);
        assert_eq!(comb.sigma_loading_slopes(0, 1), vec![(0, 1.0)]);

        // PerCmt dispatch over global sigma indices, mirroring sigma_loadings.
        let mut map = std::collections::HashMap::new();
        map.insert(
            1usize,
            EndpointError {
                error_model: ErrorModel::Combined,
                sigma_idx: vec![2, 3],
            },
        );
        map.insert(
            2usize,
            EndpointError {
                error_model: ErrorModel::Proportional,
                sigma_idx: vec![0],
            },
        );
        map.insert(
            3usize,
            EndpointError {
                error_model: ErrorModel::Additive,
                sigma_idx: vec![1],
            },
        );
        let spec = ErrorSpec::PerCmt(map);
        assert_eq!(spec.sigma_loading_slopes(1, 4), vec![(2, 1.0), (3, 0.0)]);
        assert_eq!(spec.sigma_loading_slopes(2, 4), vec![(0, 1.0)]);
        assert_eq!(spec.sigma_loading_slopes(3, 4), vec![(1, 0.0)]);
        assert!(spec.sigma_loading_slopes(9, 4).is_empty());
    }

    #[test]
    fn variance_at_with_correlations_empty_delegates_to_variance_at() {
        // No correlations ⇒ identical to the plain diagonal variance.
        let spec = ErrorSpec::Single(ErrorModel::Combined);
        let sigma = [0.2, 1.0];
        let plain = spec.variance_at(0, 10.0, &sigma);
        let corr = spec.variance_at_with_correlations(0, 10.0, &sigma, &[]);
        assert_relative_eq!(plain, corr, epsilon = 1e-12);
    }

    #[test]
    fn variance_at_with_correlations_combined_adds_cross_term() {
        // V = (10·0.2)² + 1² + 2·10·0.5·0.2·1 = 4 + 1 + 2 = 7.
        let spec = ErrorSpec::Single(ErrorModel::Combined);
        let corr = ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.5,
        };
        let v = spec.variance_at_with_correlations(1, 10.0, &[0.2, 1.0], &[corr]);
        assert_relative_eq!(v, 7.0, epsilon = 1e-12);
    }

    #[test]
    fn variance_at_with_correlations_per_cmt_combined() {
        // Same combined cross term, but reached through a PerCmt endpoint whose
        // sigmas live at global indices (0, 1).
        let mut map = std::collections::HashMap::new();
        map.insert(
            1usize,
            EndpointError {
                error_model: ErrorModel::Combined,
                sigma_idx: vec![0, 1],
            },
        );
        let spec = ErrorSpec::PerCmt(map);
        let corr = ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.5,
        };
        let v = spec.variance_at_with_correlations(1, 10.0, &[0.2, 1.0], &[corr]);
        assert_relative_eq!(v, 7.0, epsilon = 1e-12);
    }

    #[test]
    fn variance_at_with_correlations_nan_when_no_loadings() {
        // An observation in an unmapped compartment has no loadings ⇒ NaN.
        let map = std::collections::HashMap::new();
        let spec = ErrorSpec::PerCmt(map);
        let corr = ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.5,
        };
        let v = spec.variance_at_with_correlations(1, 10.0, &[0.2, 1.0], &[corr]);
        assert!(v.is_nan());
    }

    #[test]
    fn variance_at_with_correlations_nan_when_sigma_index_missing() {
        // A PerCmt combined endpoint whose sigma_idx (0, 1) points past a
        // length-1 sigma slice: the diagonal accumulation hits a missing sigma
        // and returns NaN rather than silently dropping a term.
        let mut map = std::collections::HashMap::new();
        map.insert(
            1usize,
            EndpointError {
                error_model: ErrorModel::Combined,
                sigma_idx: vec![0, 1],
            },
        );
        let spec = ErrorSpec::PerCmt(map);
        let corr = ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.5,
        };
        let v = spec.variance_at_with_correlations(1, 10.0, &[0.2], &[corr]);
        assert!(v.is_nan());
    }

    #[test]
    fn variance_at_with_correlations_skips_unloaded_correlation() {
        // The correlation names sigma index 2, which the combined endpoint
        // never loads, so the cross term is skipped and V is the plain
        // diagonal (10·0.2)² + 1² = 5.
        let spec = ErrorSpec::Single(ErrorModel::Combined);
        let corr = ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 2,
            rho: 0.5,
        };
        let v = spec.variance_at_with_correlations(0, 10.0, &[0.2, 1.0, 0.3], &[corr]);
        assert_relative_eq!(v, 5.0, epsilon = 1e-12);
    }

    #[test]
    fn residual_variance_at_threads_correlations_from_model() {
        // The `CompiledModel` convenience wrapper must apply its own
        // `residual_correlations`, not just the diagonal.
        let mut model = test_helpers::analytical_model(GradientMethod::Fd);
        model.error_spec = ErrorSpec::Single(ErrorModel::Combined);
        model.error_model = ErrorModel::Combined;
        model.residual_correlations = vec![ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.5,
        }];
        let v = model.residual_variance_at(0, 10.0, &[0.2, 1.0]);
        assert_relative_eq!(v, 7.0, epsilon = 1e-12);
    }

    #[test]
    fn has_residual_error_for_cmt_covers_single_and_per_cmt() {
        let mut model = test_helpers::analytical_model(GradientMethod::Fd);

        // Single error model: defined for every cmt iff sigma covers its n_sigma.
        model.error_spec = ErrorSpec::Single(ErrorModel::Proportional);
        assert!(model.has_residual_error_for_cmt(1, &[0.1]));
        assert!(model.has_residual_error_for_cmt(99, &[0.1]));
        assert!(
            !model.has_residual_error_for_cmt(1, &[]),
            "no sigma ⇒ no error model"
        );
        // A Combined model needs two sigmas; one is too short — `residual_variance`
        // would otherwise index `sigma[1]` and PANIC (symmetric to the PerCmt
        // out-of-range guard below, never a fabricated σ).
        model.error_spec = ErrorSpec::Single(ErrorModel::Combined);
        assert!(model.has_residual_error_for_cmt(1, &[0.1, 0.2]));
        assert!(
            !model.has_residual_error_for_cmt(1, &[0.1]),
            "Combined needs 2 sigmas; 1 is too short to draw a DV"
        );

        // PerCmt: defined only for compartments present in the map.
        let mut map = std::collections::HashMap::new();
        map.insert(
            2usize,
            EndpointError {
                error_model: ErrorModel::Proportional,
                sigma_idx: vec![0],
            },
        );
        model.error_spec = ErrorSpec::PerCmt(map);
        assert!(
            model.has_residual_error_for_cmt(2, &[0.1]),
            "cmt 2 has an endpoint"
        );
        assert!(
            !model.has_residual_error_for_cmt(1, &[0.1]),
            "cmt 1 is not in the map"
        );
        // An endpoint whose sigma_idx falls outside `sigma` is not drawable —
        // `residual_variance_at` would otherwise return NaN (symmetric to the
        // Single empty-sigma guard, never a fabricated σ).
        assert!(
            !model.has_residual_error_for_cmt(2, &[]),
            "cmt 2's sigma_idx is out of range for an empty sigma"
        );
        // A Combined endpoint needs two sigma indices; one is too few — `variance_at`
        // would read `sigma_idx[1]` (absent) and return NaN even though sigma is long.
        let mut combined_map = std::collections::HashMap::new();
        combined_map.insert(
            2usize,
            EndpointError {
                error_model: ErrorModel::Combined,
                sigma_idx: vec![0], // only one index for a two-sigma model
            },
        );
        model.error_spec = ErrorSpec::PerCmt(combined_map);
        assert!(
            !model.has_residual_error_for_cmt(2, &[0.1, 0.2]),
            "a Combined endpoint with too few sigma indices is not drawable"
        );
    }
}
