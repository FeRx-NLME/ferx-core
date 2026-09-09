//! From a resolved MFL covariate effect to the `[covariate_model]` line a
//! candidate carries (#1180).
//!
//! The `.ferxsearch` space says `COVARIATE?(CL, WT, pow)`; the candidate says
//! `CL ~ WT power(center = median)`. The mapping is the one the search page
//! publishes (`lin` → `linear`, `piece_lin` → `hockey`, `exp` →
//! `exponential`, `pow` → `power`, `cat` → `categorical`), with the centre
//! written out as the data-derived statistic PsN's `scm` uses — the median
//! for a continuous covariate, the most common level for a categorical one —
//! and the θ left to the block's defaults, which are PsN's inits and bounds
//! verbatim. Nothing here invents a number.
//!
//! MFL's operator argument — `COVARIATE?(CL, WT, lin, +)` — maps onto the
//! block's trailing operator token (#1313). ferx's additive effect is
//! null-at-zero where Pharmpy's is not, which the coverage check states as a
//! note rather than a gap; see [`ferx_core::CovariateOp`].

use ferx_core::edit::Relation;
use ferx_core::{CovariateForm, CovariateOp, CovariateStat};

use crate::search::mfl::{CovariateEffect, Mode as _};
use crate::search::CovariateEffectSpec;

/// One effect the search can add to or remove from a model.
///
/// A [`CovariateEffectSpec`] narrowed to what covsearch handles: the
/// coverage check (#1179) has already refused `cat2` and `custom`, so this is
/// the remaining five forms under either operator.
#[derive(Debug, Clone, PartialEq)]
pub struct Effect {
    pub parameter: String,
    pub covariate: String,
    pub form: CovariateForm,
    /// Whether the effect is a factor on the parameter or a term added to it.
    /// Part of the effect's identity, not of its pair: `CL ~ WT linear` and
    /// `CL ~ WT linear +` are two candidates competing for the one
    /// `[covariate_model]` line the pair is allowed.
    pub op: CovariateOp,
}

impl Effect {
    /// Build from a resolved space entry, naming an effect the search cannot
    /// take (which the coverage check should already have refused).
    pub fn from_spec(spec: &CovariateEffectSpec) -> Result<Effect, String> {
        let form = match spec.effect {
            CovariateEffect::Lin => CovariateForm::Linear,
            CovariateEffect::PieceLin => CovariateForm::Hockey,
            CovariateEffect::Exp => CovariateForm::Exponential,
            CovariateEffect::Pow => CovariateForm::Power,
            CovariateEffect::Cat => CovariateForm::Categorical,
            CovariateEffect::Cat2 | CovariateEffect::Custom => {
                return Err(format!(
                    "covsearch: `{}` on {}-{} has no `[covariate_model]` form; the coverage \
                     check should have refused it",
                    spec.effect.label(),
                    spec.parameter,
                    spec.covariate
                ))
            }
        };
        let op = match spec.op {
            crate::search::mfl::CovariateOp::Multiply => CovariateOp::Multiply,
            crate::search::mfl::CovariateOp::Add => CovariateOp::Add,
        };
        Ok(Effect {
            parameter: spec.parameter.clone(),
            covariate: spec.covariate.clone(),
            form,
            op,
        })
    }

    /// The `(parameter, covariate)` pair — one `[covariate_model]` line per
    /// pair, so this is what "already in the model" is tested on.
    pub fn pair(&self) -> (&str, &str) {
        (&self.parameter, &self.covariate)
    }

    /// `CL-WT`, the feature-vector key and the report's pair column.
    pub fn pair_key(&self) -> String {
        format!("{}-{}", self.parameter, self.covariate)
    }

    /// The `[covariate_model]` spelling of the form: `power`, `linear`, …
    pub fn form_label(&self) -> &'static str {
        self.form.label()
    }

    /// `CL-WT-power`, or `CL-WT-power-add` for an additive effect: the
    /// effect's short name in candidate ids and messages.
    ///
    /// The suffix appears only under `+` so every id a multiplicative search
    /// produced before #1313 is unchanged — and so two candidates on the same
    /// pair and form, differing only in operator, cannot collide.
    pub fn label(&self) -> String {
        match self.op {
            CovariateOp::Multiply => format!("{}-{}", self.pair_key(), self.form_label()),
            CovariateOp::Add => format!("{}-{}-add", self.pair_key(), self.form_label()),
        }
    }

    /// The `[covariate_model]` line this effect adds.
    ///
    /// The centring statistic is written explicitly rather than left to the
    /// block's default so the final model reads as what ran; the θ clause is
    /// omitted so the block supplies PsN's default init and bounds for the
    /// form, computed from the data it is bound to.
    pub fn relation(&self) -> Relation {
        let center = match self.form {
            CovariateForm::Categorical => CovariateStat::Mode,
            _ => CovariateStat::Median,
        };
        Relation {
            parameter: self.parameter.clone(),
            covariate: self.covariate.clone(),
            form: self.form.clone(),
            op: self.op,
            center: Some(center),
            fix: None,
            thetas: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::mfl::CovariateOp;

    fn spec(effect: CovariateEffect, op: CovariateOp) -> CovariateEffectSpec {
        CovariateEffectSpec {
            parameter: "CL".into(),
            covariate: "WT".into(),
            effect,
            op,
            optional: true,
        }
    }

    #[test]
    fn every_mfl_form_maps_onto_its_block_form() {
        let cases = [
            (CovariateEffect::Lin, CovariateForm::Linear, "linear"),
            (CovariateEffect::PieceLin, CovariateForm::Hockey, "hockey"),
            (
                CovariateEffect::Exp,
                CovariateForm::Exponential,
                "exponential",
            ),
            (CovariateEffect::Pow, CovariateForm::Power, "power"),
            (
                CovariateEffect::Cat,
                CovariateForm::Categorical,
                "categorical",
            ),
        ];
        for (mfl, form, label) in cases {
            let e = Effect::from_spec(&spec(mfl, CovariateOp::Multiply)).unwrap();
            assert_eq!(e.form, form);
            assert_eq!(e.form_label(), label);
            assert_eq!(e.label(), format!("CL-WT-{label}"));
            assert_eq!(e.pair(), ("CL", "WT"));
        }
    }

    #[test]
    fn the_relation_is_centred_like_psn_scm_and_leaves_theta_to_the_block() {
        let power = Effect::from_spec(&spec(CovariateEffect::Pow, CovariateOp::Multiply)).unwrap();
        let r = power.relation();
        assert_eq!(r.center, Some(CovariateStat::Median));
        assert_eq!(r.fix, None);
        assert!(r.thetas.is_empty());

        let cat = Effect::from_spec(&spec(CovariateEffect::Cat, CovariateOp::Multiply)).unwrap();
        assert_eq!(cat.relation().center, Some(CovariateStat::Mode));
    }

    #[test]
    fn forms_without_a_block_spelling_are_refused_by_name() {
        let e = Effect::from_spec(&spec(CovariateEffect::Cat2, CovariateOp::Multiply)).unwrap_err();
        assert!(e.contains("`cat2` on CL-WT"), "{e}");
        let e =
            Effect::from_spec(&spec(CovariateEffect::Custom, CovariateOp::Multiply)).unwrap_err();
        assert!(e.contains("`custom`"), "{e}");
        // A form gap is a form gap under either operator.
        let e = Effect::from_spec(&spec(CovariateEffect::Cat2, CovariateOp::Add)).unwrap_err();
        assert!(e.contains("`cat2` on CL-WT"), "{e}");
    }

    #[test]
    fn the_mfl_operator_becomes_the_blocks_trailing_operator_token() {
        let mul = Effect::from_spec(&spec(CovariateEffect::Pow, CovariateOp::Multiply)).unwrap();
        assert_eq!(mul.op, ferx_core::CovariateOp::Multiply);
        assert_eq!(mul.relation().op, ferx_core::CovariateOp::Multiply);
        assert_eq!(mul.label(), "CL-WT-power");

        let add = Effect::from_spec(&spec(CovariateEffect::Pow, CovariateOp::Add)).unwrap();
        assert_eq!(add.op, ferx_core::CovariateOp::Add);
        // The operator has to reach the relation the candidate writes, not just
        // the effect: `relation()` is the only thing the model text sees.
        assert_eq!(add.relation().op, ferx_core::CovariateOp::Add);
        // The two are one candidate each and must not collide on their id: the
        // pair is the same, and one `[covariate_model]` line is all it gets.
        assert_eq!(add.label(), "CL-WT-power-add");
        assert_eq!(add.pair(), mul.pair());
        assert_ne!(add.label(), mul.label());
    }
}
