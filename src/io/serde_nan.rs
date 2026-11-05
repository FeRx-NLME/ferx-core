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
//! - `crate::io::serde_nan::opt_vec` for `Option<Vec<f64>>`

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

/// `Option<Vec<f64>>` field: `None` -> `null`, `Some(v)` -> `v` with each
/// non-finite element -> `null`, and back. Pair it with
/// `skip_serializing_if = "Option::is_none"` + `default` so an absent vector is
/// an absent key rather than a `null` one.
pub mod opt_vec {
    use super::NanNullSlice;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(value: &Option<Vec<f64>>, ser: S) -> Result<S::Ok, S::Error> {
        match value {
            Some(v) => ser.serialize_some(&NanNullSlice(v.as_slice())),
            None => ser.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<Option<Vec<f64>>, D::Error> {
        let opts: Option<Vec<Option<f64>>> = Option::deserialize(de)?;
        Ok(opts.map(|v| v.into_iter().map(|o| o.unwrap_or(f64::NAN)).collect()))
    }
}

/// A JSON value read back with every `null` that lands on a plain `f64` (or
/// `f32`) as `NaN` — the inverse of `serde_json` writing a non-finite float as
/// `null`, for a whole type at once rather than one annotated field at a time.
///
/// Every other `null` reads as it would from the bare [`serde_json::Value`]: an
/// `Option` becomes `None`, a unit becomes `()`, anything else is an error.
/// Arrays, objects and newtypes are walked so a nested `f64` gets the same rule;
/// an enum's payload is read by the bare `Value` (no NaN-able enum payload is
/// written today).
pub(crate) struct NullAsNan(pub(crate) serde_json::Value);

impl<'de> serde::Deserializer<'de> for NullAsNan {
    type Error = serde_json::Error;

    fn deserialize_any<V: serde::de::Visitor<'de>>(self, v: V) -> Result<V::Value, Self::Error> {
        use serde_json::Value;
        match self.0 {
            Value::Array(items) => v.visit_seq(NullAsNanSeq(items.into_iter())),
            Value::Object(map) => v.visit_map(NullAsNanMap {
                entries: map.into_iter(),
                value: None,
            }),
            other => serde::Deserializer::deserialize_any(other, v),
        }
    }

    fn deserialize_f64<V: serde::de::Visitor<'de>>(self, v: V) -> Result<V::Value, Self::Error> {
        match self.0 {
            serde_json::Value::Null => v.visit_f64(f64::NAN),
            other => serde::Deserializer::deserialize_f64(other, v),
        }
    }

    fn deserialize_f32<V: serde::de::Visitor<'de>>(self, v: V) -> Result<V::Value, Self::Error> {
        self.deserialize_f64(v)
    }

    fn deserialize_option<V: serde::de::Visitor<'de>>(self, v: V) -> Result<V::Value, Self::Error> {
        match self.0 {
            serde_json::Value::Null => v.visit_none(),
            other => v.visit_some(NullAsNan(other)),
        }
    }

    fn deserialize_newtype_struct<V: serde::de::Visitor<'de>>(
        self,
        _name: &'static str,
        v: V,
    ) -> Result<V::Value, Self::Error> {
        v.visit_newtype_struct(self)
    }

    fn deserialize_enum<V: serde::de::Visitor<'de>>(
        self,
        name: &'static str,
        variants: &'static [&'static str],
        v: V,
    ) -> Result<V::Value, Self::Error> {
        serde::Deserializer::deserialize_enum(self.0, name, variants, v)
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 char str string bytes byte_buf
        unit unit_struct seq tuple tuple_struct map struct identifier ignored_any
    }
}

struct NullAsNanSeq(std::vec::IntoIter<serde_json::Value>);

impl<'de> serde::de::SeqAccess<'de> for NullAsNanSeq {
    type Error = serde_json::Error;

    fn next_element_seed<T: serde::de::DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>, Self::Error> {
        self.0
            .next()
            .map(|v| seed.deserialize(NullAsNan(v)))
            .transpose()
    }
}

struct NullAsNanMap {
    entries: serde_json::map::IntoIter,
    value: Option<serde_json::Value>,
}

impl<'de> serde::de::MapAccess<'de> for NullAsNanMap {
    type Error = serde_json::Error;

    fn next_key_seed<K: serde::de::DeserializeSeed<'de>>(
        &mut self,
        seed: K,
    ) -> Result<Option<K::Value>, Self::Error> {
        let Some((key, value)) = self.entries.next() else {
            return Ok(None);
        };
        self.value = Some(value);
        seed.deserialize(serde_json::Value::String(key)).map(Some)
    }

    fn next_value_seed<V: serde::de::DeserializeSeed<'de>>(
        &mut self,
        seed: V,
    ) -> Result<V::Value, Self::Error> {
        let value = self.value.take().expect("next_key_seed runs first");
        seed.deserialize(NullAsNan(value))
    }
}
