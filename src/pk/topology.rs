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
/// (event_driven panics on an unsupported infusion cmt; active_rates_g skips).
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
    /// Absorption model (dose enters an input/depot cmt), incl. transit/IG.
    pub is_oral: bool,
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
    /// this model (caller decides panic vs skip). Preserves the old
    /// `match (pk_model, d.cmt) { .. , _ => .. }` fall-through: any cmt not
    /// explicitly mapped (incl. cmt 0 and peripherals of oral models) yields
    /// `None`, exactly the arms the old `_` caught.
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
    is_oral: false,
    event_walk_supported: true,
};
static ONE_CPT_ORAL: PkTopology = PkTopology {
    n_states: 2,
    central_slot: 1,
    compartment_names: &["depot", "central"],
    channels: &[Some(Depot), Some(Central)],
    infusable: &[1, 2],
    is_oral: true,
    event_walk_supported: true,
};
static ONE_CPT_TRANSIT: PkTopology = PkTopology {
    n_states: 2,
    central_slot: 1,
    compartment_names: &["depot", "central"],
    channels: &[],
    infusable: &[],
    is_oral: true,
    event_walk_supported: false,
};
static ONE_CPT_IG: PkTopology = PkTopology {
    n_states: 2,
    central_slot: 1,
    compartment_names: &["depot", "central"],
    channels: &[],
    infusable: &[],
    is_oral: true,
    event_walk_supported: false,
};
static TWO_CPT_IV: PkTopology = PkTopology {
    n_states: 2,
    central_slot: 0,
    compartment_names: &["central", "peripheral"],
    channels: &[Some(Central), Some(Periph1)],
    infusable: &[1, 2],
    is_oral: false,
    event_walk_supported: true,
};
static TWO_CPT_ORAL: PkTopology = PkTopology {
    n_states: 3,
    central_slot: 1,
    compartment_names: &["depot", "central", "peripheral"],
    channels: &[Some(Depot), Some(Central)],
    infusable: &[1, 2],
    is_oral: true,
    event_walk_supported: true,
};
static TWO_CPT_TRANSIT: PkTopology = PkTopology {
    n_states: 3,
    central_slot: 1,
    compartment_names: &["depot", "central", "peripheral"],
    channels: &[],
    infusable: &[],
    is_oral: true,
    event_walk_supported: false,
};
static TWO_CPT_IG: PkTopology = PkTopology {
    n_states: 3,
    central_slot: 1,
    compartment_names: &["depot", "central", "peripheral"],
    channels: &[],
    infusable: &[],
    is_oral: true,
    event_walk_supported: false,
};
static THREE_CPT_IV: PkTopology = PkTopology {
    n_states: 3,
    central_slot: 0,
    compartment_names: &["central", "peripheral1", "peripheral2"],
    channels: &[Some(Central), Some(Periph1), Some(Periph2)],
    infusable: &[1, 2, 3],
    is_oral: false,
    event_walk_supported: true,
};
static THREE_CPT_ORAL: PkTopology = PkTopology {
    n_states: 4,
    central_slot: 1,
    compartment_names: &["depot", "central", "peripheral1", "peripheral2"],
    channels: &[Some(Depot), Some(Central)],
    infusable: &[1, 2],
    is_oral: true,
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

    /// The three stored-but-derivable fields must agree with their canonical
    /// derivation for every variant. This is the drift guard that lets the
    /// descriptor be a flat literal.
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
            let derived: Vec<usize> = t
                .channels
                .iter()
                .enumerate()
                .filter_map(|(i, c)| c.map(|_| i + 1))
                .collect();
            assert_eq!(t.infusable, derived.as_slice(), "{m:?} infusable");
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
            (ThreeCptIv, 1, Some(Channel::Central)),
            (ThreeCptIv, 2, Some(Channel::Periph1)),
            (ThreeCptIv, 3, Some(Channel::Periph2)),
            (ThreeCptOral, 1, Some(Channel::Depot)),
            (ThreeCptOral, 2, Some(Channel::Central)),
            // old `_` arms -> None
            (OneCptIv, 0, None),
            (OneCptIv, 2, None),
            (TwoCptOral, 3, None),
            (ThreeCptOral, 3, None),
        ];
        for (m, cmt, want) in cases {
            assert_eq!(m.topology().dose_channel(cmt), want, "{m:?} cmt {cmt}");
        }
    }
}
