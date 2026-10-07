//! One read of a `.ferx` model file (#1752).
//!
//! An entry point that parses a model file, binds it to data, hashes it and stores
//! its text needs all four to describe the **same** version of the file. Reading
//! the file once per use let an edit between the reads bind the new text against
//! the old parse, and stamp `model_hash` / `model_text` from a file that was never
//! fitted. [`ModelSource`] is that one read: the bytes are hashed, decoded and
//! parsed, and every later use takes them from here.

use std::path::{Path, PathBuf};

use crate::io::hash::sha256_bytes;
use crate::types::ParsedModel;

/// A model file read once: its text, the SHA-256 of its bytes, and the parse of
/// that text. Every field describes the same read of the file.
#[non_exhaustive]
pub struct ModelSource {
    /// The path the file was read from, as given.
    pub path: PathBuf,
    /// The file's contents, decoded as UTF-8. This is the text to hand to
    /// [`crate::bind_theta_levels`] / `bind_covariate_stats` and to
    /// store as `FitResult::model_text`.
    pub text: String,
    /// Lowercase hex SHA-256 of the file's bytes, as `FitResult::model_hash`
    /// stores it.
    pub hash: String,
    /// The parse of [`ModelSource::text`]. Relative paths in it (`[data] path`,
    /// `[priors] from_fit`) resolve against the file's directory.
    pub parsed: ParsedModel,
}

/// Which step of the read failed, so a caller can give each its own code
/// (`ferx check` reports an unreadable file as `E_MODEL_READ`, a parse failure by
/// the parse's own code).
pub(crate) enum ModelSourceError {
    /// The file could not be read, or is not UTF-8.
    Read(String),
    /// The file's hash is not the one the caller expected.
    Hash(String),
    /// The text does not parse.
    Parse(String),
}

impl ModelSourceError {
    pub(crate) fn into_message(self) -> String {
        match self {
            Self::Read(m) | Self::Hash(m) | Self::Parse(m) => m,
        }
    }
}

impl ModelSource {
    /// Read `path` once, hash its bytes and parse its text.
    ///
    /// An unreadable file fails with `cannot read the model file {path}: {e}`; a
    /// file that does not parse fails with the parser's message.
    pub fn read(path: impl AsRef<Path>) -> Result<Self, String> {
        Self::load(path.as_ref(), None, "", "").map_err(ModelSourceError::into_message)
    }

    /// [`ModelSource::read`], refusing a file whose SHA-256 is not
    /// `expected_hash` (when given). The hash is compared **before** the text is
    /// decoded or parsed, so an edited file is reported as edited even when the
    /// edit no longer parses. `entry` names the caller in every failure message
    /// (`run_sir`, `ferx_sir`, …).
    pub fn read_verified(
        path: impl AsRef<Path>,
        expected_hash: Option<&str>,
        entry: &str,
    ) -> Result<Self, String> {
        Self::load(path.as_ref(), expected_hash, &format!("{entry}: "), "")
            .map_err(ModelSourceError::into_message)
    }

    /// The one implementation behind [`ModelSource::read`] / `read_verified`.
    /// `prefix` opens every message; `because` closes the read and hash failures
    /// (the post-hoc resolver says there why the file is needed at all).
    pub(crate) fn load(
        path: &Path,
        expected_hash: Option<&str>,
        prefix: &str,
        because: &str,
    ) -> Result<Self, ModelSourceError> {
        let shown = path.display();
        let bytes = std::fs::read(path).map_err(|e| {
            ModelSourceError::Read(format!(
                "{prefix}cannot read the model file {shown}: {e}.{because}"
            ))
        })?;
        let hash = sha256_bytes(&bytes);
        if let Some(expected) = expected_hash {
            if hash != expected {
                return Err(ModelSourceError::Hash(format!(
                    "{prefix}model hash mismatch for {shown}. Stored: {expected}, current: \
                     {hash}. The .ferx file has changed since the fit was produced — \
                     refusing to run against stale source.{because}"
                )));
            }
        }
        let text = String::from_utf8(bytes).map_err(|e| {
            ModelSourceError::Read(format!("{prefix}the model file {shown} is not UTF-8: {e}"))
        })?;
        let parsed = crate::parser::model_parser::parse_full_model_source(&text, path)
            .map_err(|e| ModelSourceError::Parse(format!("{prefix}{e}")))?;
        Ok(Self {
            path: path.to_path_buf(),
            text,
            hash,
            parsed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::ModelSource;
    use crate::io::hash::{sha256_bytes, sha256_file};

    const MODEL: &str =
        "[parameters]\n  theta TVCL(1.0, 0.1, 10.0)\n  theta TVV(10.0, 1.0, 100.0)\n  \
                         omega ETA_CL ~ 0.1\n  sigma PROP ~ 0.1\n\n[individual_parameters]\n  \
                         CL = TVCL * exp(ETA_CL)\n  V = TVV\n\n[structural_model]\n  \
                         pk one_cpt_iv(cl=CL, v=V)\n\n[error_model]\n  DV ~ proportional(PROP)\n";

    fn file(text: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.ferx");
        std::fs::write(&path, text).unwrap();
        (dir, path)
    }

    /// The read's fields are one read of the file: `text` is its bytes, `hash`
    /// their SHA-256 (as `sha256_file` and `FitResult::model_hash` spell it), and
    /// `parsed` the parse of that text.
    #[test]
    fn a_read_is_the_files_bytes_their_hash_and_their_parse() {
        let (_d, path) = file(MODEL);
        let src = ModelSource::read(&path).unwrap();
        assert_eq!(src.text, MODEL);
        assert_eq!(src.hash, sha256_file(&path).unwrap());
        assert_eq!(src.hash, sha256_bytes(MODEL.as_bytes()));
        assert_eq!(src.parsed.model.theta_names, ["TVCL", "TVV"]);
        assert_eq!(src.path, path);
    }

    /// `read_verified` compares the hash **before** it parses: a file edited into
    /// text that no longer parses is reported as edited, not as a parse error. Both
    /// sides of the ordering in one test: the same broken text with no expected
    /// hash is the parse error, and the untouched file verifies.
    ///
    /// Mutation — parse before comparing the hash: the mismatch assertions.
    #[test]
    fn a_hash_mismatch_is_reported_before_the_parse() {
        let (_d, path) = file(MODEL);
        let stored = sha256_bytes(MODEL.as_bytes());
        let ok = ModelSource::read_verified(&path, Some(&stored), "run_sir").unwrap();
        assert_eq!(ok.text, MODEL);

        std::fs::write(&path, "[parameters]\n  theta TVCL(\n").unwrap();
        let err = ModelSource::read_verified(&path, Some(&stored), "run_sir")
            .err()
            .expect("an edited file is refused");
        let want = format!(
            "run_sir: model hash mismatch for {}. Stored: {stored}, current: ",
            path.display()
        );
        assert!(err.starts_with(&want), "{err}");
        assert!(
            err.ends_with(
                "The .ferx file has changed since the fit was produced — refusing to run \
                 against stale source."
            ),
            "{err}"
        );

        let parse = ModelSource::read_verified(&path, None, "run_sir")
            .err()
            .expect("the edit does not parse");
        assert!(parse.starts_with("run_sir: "), "{parse}");
        assert!(!parse.contains("hash"), "{parse}");
    }

    /// An unreadable file names the path and the OS's reason; `read` carries no
    /// entry prefix, `read_verified` carries its entry.
    #[test]
    fn an_unreadable_file_names_its_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing.ferx");
        let err = ModelSource::read(&path).err().expect("missing");
        let want = format!("cannot read the model file {}: ", path.display());
        assert!(err.starts_with(&want), "{err}");
        let err = ModelSource::read_verified(&path, None, "ferx_sir")
            .err()
            .expect("missing");
        assert!(err.starts_with(&format!("ferx_sir: {want}")), "{err}");
    }

    /// Bytes that are not UTF-8 are a read failure, after the hash check.
    #[test]
    fn a_file_that_is_not_utf8_is_a_read_failure() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.ferx");
        std::fs::write(&path, [0xff, 0xfe, 0x00]).unwrap();
        let err = ModelSource::read(&path).err().expect("not UTF-8");
        let want = format!("the model file {} is not UTF-8: ", path.display());
        assert!(err.starts_with(&want), "{err}");
    }
}
