//! Single per-`PkModel` compartment-topology descriptor.
//!
//! Consolidates the per-model lookup tables that previously re-encoded the same
//! compartment geometry independently in `event_driven::state_layout`,
//! `propagate::state_layout_g`, `pk/mod::analytical_state_at_times` (inline
//! `n_states`), `types::analytical_compartment_names`,
//! `types::infusable_compartments`, `supports_event_driven` / `walk_supports`,
//! and the two `match (pk_model, d.cmt)` rate-routing blocks in
//! `propagate_with_bounds` / `active_rates_g`.
//!
//! Every symbol here is an integer / name / boolean lookup feeding vec sizing,
//! an index read, rate-channel selection, or a supported/unsupported branch. No
//! f64 arithmetic lives here, so consolidation is provably numerically inert.

use crate::types::PkModel;

/// Which internal rate accumulator a 1-based dose compartment routes into for
/// the closed-form event walk. Returned by [`PkTopology::dose_channel`]; callers
/// map each arm onto their own `rate_*` accumulator and decide the `None` policy
/// — since #375 both walks `panic!` on it (silently dropping an infusion from
/// the *gradient* only would make FOCE differentiate a different dosing history
/// than it predicted), and `check_dose_compartments` makes that unreachable from
/// a validated call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Channel {
    Central,
    Periph1,
    Periph2,
    Depot,
}

/// Compartment geometry for one `PkModel`. `'static` throughout so callers keep
/// their existing `&'static` return signatures.
pub(crate) struct PkTopology {
    /// State-vector dimension. Invariant: `== compartment_names.len()`
    /// (asserted by `topology_fields_are_internally_consistent`).
    pub n_states: usize,
    /// Index of `"central"` within `compartment_names` (the read-out slot).
    /// Stored (not derived) for a literal transcription; the test above pins it
    /// to `compartment_names.iter().position(|n| *n == "central")`.
    pub central_slot: usize,
    /// Ordered analytical compartment names (the state-vector layout).
    pub compartment_names: &'static [&'static str],
    /// Rate channel for each 1-based dose compartment; index = `cmt - 1`.
    /// `None` / out-of-range = not an infusable compartment for this model.
    pub channels: &'static [Option<Channel>],
    /// 1-based compartments a modeled infusion (`D{cmt}`/`R{cmt}`) may target.
    /// Invariant: `== {i+1 | channels[i] == Some(_)}` (asserted by the
    /// consistency test).
    pub infusable: &'static [usize],
    /// Has an event-driven closed-form walk implementation (the six
    /// non-transit/IG analytical models).
    pub event_walk_supported: bool,
}

impl PkModel {
    /// The compartment topology for this model. `'static` table lookup.
    pub(crate) fn topology(self) -> &'static PkTopology {
        match self {
            PkModel::OneCptIv => &ONE_CPT_IV,
            PkModel::OneCptOral => &ONE_CPT_ORAL,
            PkModel::OneCptTransit => &ONE_CPT_TRANSIT,
            PkModel::OneCptIg => &ONE_CPT_IG,
            PkModel::TwoCptIv => &TWO_CPT_IV,
            PkModel::TwoCptOral => &TWO_CPT_ORAL,
            PkModel::TwoCptTransit => &TWO_CPT_TRANSIT,
            PkModel::TwoCptIg => &TWO_CPT_IG,
            PkModel::ThreeCptIv => &THREE_CPT_IV,
            PkModel::ThreeCptOral => &THREE_CPT_ORAL,
        }
    }
}

impl PkTopology {
    /// `(n_states, central_slot)` — the tuple the two `state_layout*` facades return.
    #[inline]
    pub(crate) fn state_layout(&self) -> (usize, usize) {
        (self.n_states, self.central_slot)
    }

    /// Rate channel for a 1-based dose compartment. `None` = not infusable for
    /// this model; every caller panics on it (see [`Channel`]). Preserves the old
    /// `match (pk_model, d.cmt) { .. , _ => .. }` fall-through: any cmt not
    /// explicitly mapped yields `None`, exactly the arms the old `_` caught.
    /// Since #375 the oral peripherals map to `Some(Periph1)`/`Some(Periph2)`,
    /// so for the six walk-supported models the only in-range `None` is `cmt 0`
    /// (`checked_sub(1)`), NONMEM's default dose compartment — which has no
    /// meaning for a zero-order input and is rejected by
    /// `check_dose_compartments`.
    #[inline]
    pub(crate) fn dose_channel(&self, cmt_1based: usize) -> Option<Channel> {
        cmt_1based
            .checked_sub(1)
            .and_then(|i| self.channels.get(i).copied().flatten())
    }

    /// 1-based compartments a modeled infusion may target.
    #[inline]
    pub(crate) fn infusable_compartments(&self) -> &'static [usize] {
        self.infusable
    }
}

use Channel::{Central, Depot, Periph1, Periph2};

static ONE_CPT_IV: PkTopology = PkTopology {
    n_states: 1,
    central_slot: 0,
    compartment_names: &["central"],
    channels: &[Some(Central)],
    infusable: &[1],
    event_walk_supported: true,
};
static ONE_CPT_ORAL: PkTopology = PkTopology {
    n_states: 2,
    central_slot: 1,
    compartment_names: &["depot", "central"],
    channels: &[Some(Depot), Some(Central)],
    infusable: &[1, 2],
    event_walk_supported: true,
};
// Transit models a bolus absorbed through the Gamma transit chain; modeled-duration
// infusions are unsupported in v1, so a `D{cmt}` on a transit model is rejected at parse
// (#386) — hence `infusable: &[]`. There is no event walk, so `channels: &[]` too. Do NOT
// populate these to "fix an oversight": routing would accept a `D1` that #386 rejects.
static ONE_CPT_TRANSIT: PkTopology = PkTopology {
    n_states: 2,
    central_slot: 1,
    compartment_names: &["depot", "central"],
    channels: &[],
    infusable: &[],
    event_walk_supported: false,
};
// Inverse-Gaussian, like transit: absorbs an instantaneous bolus through the IG
// absorption-time density; modeled-duration infusions are unsupported, so a `D{cmt}` on an
// IG model is rejected at parse (#790) — hence `infusable: &[]` and no walk (`channels: &[]`).
static ONE_CPT_IG: PkTopology = PkTopology {
    n_states: 2,
    central_slot: 1,
    compartment_names: &["depot", "central"],
    channels: &[],
    infusable: &[],
    event_walk_supported: false,
};
static TWO_CPT_IV: PkTopology = PkTopology {
    n_states: 2,
    central_slot: 0,
    compartment_names: &["central", "peripheral"],
    channels: &[Some(Central), Some(Periph1)],
    infusable: &[1, 2],
    event_walk_supported: true,
};
static TWO_CPT_ORAL: PkTopology = PkTopology {
    n_states: 3,
    central_slot: 1,
    compartment_names: &["depot", "central", "peripheral"],
    // The oral peripheral is infusable since #375: the depot takes no inflow from
    // the disposition sub-system, so a rate into the peripheral drives exactly the
    // central/peripheral pair the 2-cpt IV model has, and the oral propagator
    // reuses that IV forced response.
    channels: &[Some(Depot), Some(Central), Some(Periph1)],
    infusable: &[1, 2, 3],
    event_walk_supported: true,
};
// 2-cpt transit: same as the 1-cpt transit — modeled-duration infusion rejected at parse
// (#386), no event walk. `channels`/`infusable` intentionally empty.
static TWO_CPT_TRANSIT: PkTopology = PkTopology {
    n_states: 3,
    central_slot: 1,
    compartment_names: &["depot", "central", "peripheral"],
    channels: &[],
    infusable: &[],
    event_walk_supported: false,
};
// 2-cpt inverse-Gaussian: same as the 1-cpt IG (#790). `channels`/`infusable` empty.
static TWO_CPT_IG: PkTopology = PkTopology {
    n_states: 3,
    central_slot: 1,
    compartment_names: &["depot", "central", "peripheral"],
    channels: &[],
    infusable: &[],
    event_walk_supported: false,
};
static THREE_CPT_IV: PkTopology = PkTopology {
    n_states: 3,
    central_slot: 0,
    compartment_names: &["central", "peripheral1", "peripheral2"],
    channels: &[Some(Central), Some(Periph1), Some(Periph2)],
    infusable: &[1, 2, 3],
    event_walk_supported: true,
};
static THREE_CPT_ORAL: PkTopology = PkTopology {
    n_states: 4,
    central_slot: 1,
    compartment_names: &["depot", "central", "peripheral1", "peripheral2"],
    // Both oral peripherals are infusable since #375 — same reduction to the
    // 3-cpt IV forced response as `TWO_CPT_ORAL`.
    channels: &[Some(Depot), Some(Central), Some(Periph1), Some(Periph2)],
    infusable: &[1, 2, 3, 4],
    event_walk_supported: true,
};

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [PkModel; 10] = [
        PkModel::OneCptIv,
        PkModel::OneCptOral,
        PkModel::OneCptTransit,
        PkModel::OneCptIg,
        PkModel::TwoCptIv,
        PkModel::TwoCptOral,
        PkModel::TwoCptTransit,
        PkModel::TwoCptIg,
        PkModel::ThreeCptIv,
        PkModel::ThreeCptOral,
    ];

    /// Drift guard for the stored-but-derivable geometry fields.
    ///
    /// `n_states`/`central_slot` are pinned to their canonical derivation from
    /// `compartment_names`. For `infusable` vs `channels` we assert only the
    /// genuinely-invariant DIRECTION — every routable compartment must be a
    /// parse-accepted infusion target (`channels[i].is_some() ⇒ infusable ∋ i+1`) —
    /// NOT full equality. The two fields answer different questions (`infusable` is
    /// the parse-time gate for `D{cmt}`/`R{cmt}`; `channels` is the walk's rate-accumulator
    /// routing), and enforcing equality would force a future PR that legitimately accepts an
    /// infusion into a no-walk model to invent a fake routing arm. So `infusable` may be a
    /// superset of the channel set; it may not contain a compartment the walk can't route.
    #[test]
    fn topology_fields_are_internally_consistent() {
        for m in ALL {
            let t = m.topology();
            assert_eq!(t.n_states, t.compartment_names.len(), "{m:?} n_states");
            assert_eq!(
                t.central_slot,
                t.compartment_names
                    .iter()
                    .position(|n| *n == "central")
                    .unwrap(),
                "{m:?} central_slot"
            );
            let channel_set: Vec<usize> = t
                .channels
                .iter()
                .enumerate()
                .filter_map(|(i, c)| c.map(|_| i + 1))
                .collect();
            // Every routable compartment must be a parse-accepted infusion target.
            for &cmt in &channel_set {
                assert!(
                    t.infusable.contains(&cmt),
                    "{m:?}: cmt {cmt} is routable but not in infusable {:?}",
                    t.infusable
                );
            }
            // For a WALK-SUPPORTED model `infusable` must EQUAL the channel set: an
            // extra entry would let the parser accept a `D{cmt}` that `dose_channel`
            // can't route, panicking the event walk at fit time. (A no-walk transit/IG
            // model may legitimately keep `infusable` a superset — direction-only above.)
            if t.event_walk_supported {
                assert_eq!(
                    t.infusable, channel_set,
                    "{m:?}: walk-supported model's infusable must equal its channel set"
                );
            }
        }
    }

    /// Pin the six event-walk-supported models (== old supports_event_driven).
    #[test]
    fn event_walk_supported_matches_the_six_analytical_models() {
        for m in ALL {
            let expect = matches!(
                m,
                PkModel::OneCptIv
                    | PkModel::OneCptOral
                    | PkModel::TwoCptIv
                    | PkModel::TwoCptOral
                    | PkModel::ThreeCptIv
                    | PkModel::ThreeCptOral
            );
            assert_eq!(m.topology().event_walk_supported, expect, "{m:?}");
        }
    }

    /// dose_channel reproduces the old 12-arm (pk_model, cmt) routing table,
    /// and returns None for every arm the old `_` caught (cmt 0, oral periph).
    #[test]
    fn dose_channel_reproduces_routing_table() {
        use PkModel::*;
        let cases = [
            (OneCptIv, 1, Some(Channel::Central)),
            (OneCptOral, 1, Some(Channel::Depot)),
            (OneCptOral, 2, Some(Channel::Central)),
            (TwoCptIv, 1, Some(Channel::Central)),
            (TwoCptIv, 2, Some(Channel::Periph1)),
            (TwoCptOral, 1, Some(Channel::Depot)),
            (TwoCptOral, 2, Some(Channel::Central)),
            // Oral peripherals gained a rate channel in #375 — the oral
            // propagators reuse the IV forced response, so an infusion into them
            // is routable rather than rejected.
            (TwoCptOral, 3, Some(Channel::Periph1)),
            (ThreeCptIv, 1, Some(Channel::Central)),
            (ThreeCptIv, 2, Some(Channel::Periph1)),
            (ThreeCptIv, 3, Some(Channel::Periph2)),
            (ThreeCptOral, 1, Some(Channel::Depot)),
            (ThreeCptOral, 2, Some(Channel::Central)),
            (ThreeCptOral, 3, Some(Channel::Periph1)),
            (ThreeCptOral, 4, Some(Channel::Periph2)),
            // Still unroutable: cmt 0 (an infusion has no default compartment)
            // and anything past the end of the state vector.
            (OneCptIv, 0, None),
            (OneCptIv, 2, None),
            (TwoCptOral, 0, None),
            (TwoCptOral, 4, None),
            (ThreeCptOral, 0, None),
            (ThreeCptOral, 5, None),
        ];
        for (m, cmt, want) in cases {
            assert_eq!(m.topology().dose_channel(cmt), want, "{m:?} cmt {cmt}");
        }
    }
}
