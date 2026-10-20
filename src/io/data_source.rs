//! One read of a dataset (#2140), the data-side twin of
//! [`ModelSource`](crate::ModelSource).
//!
//! An entry point that reads a population and stamps `FitResult::data_hash` needs
//! both to describe the **same** version of the file. Reading the path once for
//! the population and again for the hash let an edit between the reads (a CSV
//! rewritten while a long fit runs) be recorded as the fitted data. [`DataSource`]
//! is that one read: the bytes are taken once and hashed, and the reader parses
//! those bytes ([`crate::read_population_with_source`]).

use std::path::{Path, PathBuf};

use crate::io::datareader::ERR_OPEN_CSV;
use crate::io::hash::sha256_bytes;

/// A dataset read once: its bytes and their SHA-256. Every field describes the
/// same read.
#[non_exhaustive]
pub struct DataSource {
    /// The path the bytes were read from, as given; `None` for
    /// [`DataSource::from_bytes`].
    pub path: Option<PathBuf>,
    /// The dataset's contents, as read.
    pub bytes: Vec<u8>,
    /// Lowercase hex SHA-256 of [`DataSource::bytes`], as `FitResult::data_hash`
    /// stores it.
    pub hash: String,
}

impl DataSource {
    /// Read `path` once and hash its bytes.
    ///
    /// An unreadable file fails with the message
    /// [`read_population_with`](crate::read_population_with) gives the same file:
    /// `Failed to open CSV: {e}`.
    pub fn read(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| format!("{ERR_OPEN_CSV}{e}"))?;
        Ok(Self {
            path: Some(path.to_path_buf()),
            hash: sha256_bytes(&bytes),
            bytes,
        })
    }

    /// A dataset the caller already holds in memory, hashed.
    pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> Self {
        let bytes = bytes.into();
        Self {
            path: None,
            hash: sha256_bytes(&bytes),
            bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DataSource;
    use crate::io::hash::{sha256_bytes, sha256_file};

    const CSV: &str = "ID,TIME,DV,EVID,AMT\n1,0,.,1,100\n1,1,5.0,0,.\n";

    /// The read's fields are one read of the file: `bytes` is its contents and
    /// `hash` their SHA-256, spelled as `sha256_file` (and so `data_hash`) spells it.
    #[test]
    fn a_read_is_the_files_bytes_and_their_hash() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("d.csv");
        std::fs::write(&path, CSV).unwrap();
        let src = DataSource::read(&path).unwrap();
        assert_eq!(src.bytes, CSV.as_bytes());
        assert_eq!(src.hash, sha256_file(&path).unwrap());
        assert_eq!(src.path.as_deref(), Some(path.as_path()));

        let mem = DataSource::from_bytes(CSV);
        assert_eq!(mem.hash, sha256_bytes(CSV.as_bytes()));
        assert_eq!(mem.hash, src.hash);
        assert!(mem.path.is_none());
    }

    /// An unreadable file fails as the reader's own open does, so a caller that
    /// switches to a source keeps the message it had.
    #[test]
    fn an_unreadable_file_is_the_readers_open_failure() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing.csv");
        let err = DataSource::read(&path).err().expect("missing");
        let os = std::fs::File::open(&path).unwrap_err();
        assert_eq!(err, format!("Failed to open CSV: {os}"));
    }
}
