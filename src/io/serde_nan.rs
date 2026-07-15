//! Shared NaN-as-`null` serde adapters for JSON bundles (`.fitrx`, checkpoints).
//!
//! JSON has no representation for NaN/±Inf. `serde_json` writes them as `null`
//! on serialize, which is correct per the spec but means the default
//! `Deserialize` for `f64` then fails when reading the value back. These modules
//! make the round-trip lossless: non-finite serializes to `null`, and `null`
//! deserializes back to `NaN`. Use via `#[serde(with = "...")]` on fields that
//! may legitimately carry NaN (shrinkage, `cov_condition_number` when the
//! Hessian is singular, the OFV before the first eval, etc.):
//!
//! - `crate::io::serde_nan::scalar`  for `f64`
//! - `crate::io::serde_nan::vec`     for `Vec<f64>`
//! - `crate::io::serde_nan::vec_vec` for `Vec<Vec<f64>>`
//! - `crate::io::serde_nan::opt`     for `Option<f64>`

use serde::{Serialize, Serializer};

/// A single `f64` that serializes as itself when finite, else as JSON `null`.
/// The one place the scalar element rule lives; the sequence serializers reuse
/// it so there is no second copy of the finite check.
struct NanNull(f64);

impl Serialize for NanNull {
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        if self.0.is_finite() {
            ser.serialize_f64(self.0)
        } else {
            ser.serialize_none()
        }
    }
}

/// A slice of `f64` serialized as a sequence of [`NanNull`] elements. Reused by
/// both `vec` and `vec_vec` so the inner element writer exists exactly once.
struct NanNullSlice<'a>(&'a [f64]);

impl Serialize for NanNullSlice<'_> {
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let mut seq = ser.serialize_seq(Some(self.0.len()))?;
        for v in self.0 {
            seq.serialize_element(&NanNull(*v))?;
        }
        seq.end()
    }
}

/// `f64` field: non-finite -> `null`, `null` -> `NaN`.
pub mod scalar {
    use super::NanNull;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(value: &f64, ser: S) -> Result<S::Ok, S::Error> {
        NanNull(*value).serialize(ser)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<f64, D::Error> {
        let opt: Option<f64> = Option::deserialize(de)?;
        Ok(opt.unwrap_or(f64::NAN))
    }
}

/// `Vec<f64>` field: each non-finite element -> `null`, `null` -> `NaN`.
pub mod vec {
    use super::NanNullSlice;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(value: &Vec<f64>, ser: S) -> Result<S::Ok, S::Error> {
        NanNullSlice(value.as_slice()).serialize(ser)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<Vec<f64>, D::Error> {
        let opts: Vec<Option<f64>> = Vec::deserialize(de)?;
        Ok(opts.into_iter().map(|o| o.unwrap_or(f64::NAN)).collect())
    }
}

/// `Vec<Vec<f64>>` field: each non-finite element -> `null`, `null` -> `NaN`.
pub mod vec_vec {
    use super::NanNullSlice;
    use serde::{de::Deserialize, ser::SerializeSeq, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(value: &Vec<Vec<f64>>, ser: S) -> Result<S::Ok, S::Error> {
        let mut outer = ser.serialize_seq(Some(value.len()))?;
        for inner in value {
            outer.serialize_element(&NanNullSlice(inner.as_slice()))?;
        }
        outer.end()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<Vec<Vec<f64>>, D::Error> {
        let outer: Vec<Vec<Option<f64>>> = Vec::deserialize(de)?;
        Ok(outer
            .into_iter()
            .map(|inner| inner.into_iter().map(|o| o.unwrap_or(f64::NAN)).collect())
            .collect())
    }
}

/// `Option<f64>` field: `Some(non-finite)` and `None` both -> `null`.
pub mod opt {
    use serde::{de::Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(value: &Option<f64>, ser: S) -> Result<S::Ok, S::Error> {
        match value {
            Some(v) if v.is_finite() => ser.serialize_some(v),
            _ => ser.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<Option<f64>, D::Error> {
        Option::<f64>::deserialize(de)
    }
}
