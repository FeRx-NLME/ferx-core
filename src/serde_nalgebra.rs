//! serde shims for `nalgebra` `DMatrix`/`DVector`.
//!
//! nalgebra's own `serde-serialize` feature is off in this crate, and its
//! internal representation would serialize as an opaque storage blob anyway.
//! For the machine-readable JSON fit output (#777) we want a stable, agent-
//! friendly shape instead:
//!
//! - a matrix serializes as `{ "rows": r, "cols": c, "data": [..] }`, with
//!   `data` laid out **row-major**;
//! - a vector serializes as a flat JSON array.
//!
//! Use via `#[serde(with = "crate::serde_nalgebra::<module>")]` on a field.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Wire representation of a dense matrix. `data` is row-major.
#[derive(Serialize, Deserialize)]
struct MatrixRepr {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

fn to_repr(m: &DMatrix<f64>) -> MatrixRepr {
    let (rows, cols) = (m.nrows(), m.ncols());
    let mut data = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            data.push(m[(r, c)]);
        }
    }
    MatrixRepr { rows, cols, data }
}

fn from_repr<E: serde::de::Error>(repr: MatrixRepr) -> Result<DMatrix<f64>, E> {
    if repr.data.len() != repr.rows * repr.cols {
        return Err(E::custom(format!(
            "matrix data length {} does not match rows*cols = {}*{}",
            repr.data.len(),
            repr.rows,
            repr.cols
        )));
    }
    // `from_row_slice` consumes the slice row-major, matching `to_repr`.
    Ok(DMatrix::from_row_slice(repr.rows, repr.cols, &repr.data))
}

/// `#[serde(with)]` module for a plain `DMatrix<f64>` field.
pub mod dmatrix {
    use super::*;

    pub fn serialize<S: Serializer>(m: &DMatrix<f64>, s: S) -> Result<S::Ok, S::Error> {
        to_repr(m).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<DMatrix<f64>, D::Error> {
        from_repr(MatrixRepr::deserialize(d)?)
    }
}

/// `#[serde(with)]` module for an `Option<DMatrix<f64>>` field.
pub mod option_dmatrix {
    use super::*;

    pub fn serialize<S: Serializer>(m: &Option<DMatrix<f64>>, s: S) -> Result<S::Ok, S::Error> {
        m.as_ref().map(to_repr).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<DMatrix<f64>>, D::Error> {
        match Option::<MatrixRepr>::deserialize(d)? {
            Some(repr) => Ok(Some(from_repr(repr)?)),
            None => Ok(None),
        }
    }
}

/// `#[serde(with)]` module for a plain `DVector<f64>` field (flat array).
pub mod dvector {
    use super::*;

    pub fn serialize<S: Serializer>(v: &DVector<f64>, s: S) -> Result<S::Ok, S::Error> {
        v.as_slice().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<DVector<f64>, D::Error> {
        Ok(DVector::from_vec(Vec::<f64>::deserialize(d)?))
    }
}

/// `#[serde(with)]` module for a `Vec<Vec<DVector<f64>>>` field (per-subject,
/// per-occasion IOV EBEs). Serializes as a nested array of flat arrays.
pub mod vec_vec_dvector {
    use super::*;

    #[allow(clippy::ptr_arg)]
    pub fn serialize<S: Serializer>(v: &Vec<Vec<DVector<f64>>>, s: S) -> Result<S::Ok, S::Error> {
        let nested: Vec<Vec<&[f64]>> = v
            .iter()
            .map(|inner| inner.iter().map(|dv| dv.as_slice()).collect())
            .collect();
        nested.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Vec<Vec<DVector<f64>>>, D::Error> {
        let nested = Vec::<Vec<Vec<f64>>>::deserialize(d)?;
        Ok(nested
            .into_iter()
            .map(|inner| inner.into_iter().map(DVector::from_vec).collect())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct Holder {
        #[serde(with = "dmatrix")]
        m: DMatrix<f64>,
        #[serde(with = "option_dmatrix")]
        om: Option<DMatrix<f64>>,
        #[serde(with = "dvector")]
        v: DVector<f64>,
        #[serde(with = "vec_vec_dvector")]
        vv: Vec<Vec<DVector<f64>>>,
    }

    #[test]
    fn matrix_is_row_major_and_round_trips() {
        // 2x3 with distinct entries so a transpose bug would show.
        let m = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let h = Holder {
            m: m.clone(),
            om: Some(m.clone()),
            v: DVector::from_vec(vec![7.0, 8.0]),
            vv: vec![vec![DVector::from_vec(vec![1.0, 2.0])], vec![]],
        };
        let json = serde_json::to_value(&h).unwrap();
        // Row-major: first row [1,2,3] precedes second row [4,5,6].
        assert_eq!(json["m"]["rows"], 2);
        assert_eq!(json["m"]["cols"], 3);
        assert_eq!(
            json["m"]["data"],
            serde_json::json!([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        );
        assert_eq!(json["v"], serde_json::json!([7.0, 8.0]));

        let back: Holder = serde_json::from_value(json).unwrap();
        assert_eq!(back, h);
    }

    #[test]
    fn none_option_round_trips() {
        let h = Holder {
            m: DMatrix::zeros(1, 1),
            om: None,
            v: DVector::zeros(0),
            vv: vec![],
        };
        let s = serde_json::to_string(&h).unwrap();
        let back: Holder = serde_json::from_str(&s).unwrap();
        assert_eq!(back, h);
    }

    #[test]
    fn mismatched_length_errors() {
        let bad = serde_json::json!({
            "m": {"rows": 2, "cols": 2, "data": [1.0, 2.0, 3.0]},
            "om": null,
            "v": [],
            "vv": []
        });
        assert!(serde_json::from_value::<Holder>(bad).is_err());
    }
}
