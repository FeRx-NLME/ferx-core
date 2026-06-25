//! SHA-256 hashing for model and data file integrity checks.
//!
//! Used to detect whether the `.ferx` model file or the data CSV has changed
//! between a fit and a downstream operation like `run_sir` that re-uses the
//! fit's stored paths.

use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::Path;

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

/// Compute the SHA-256 of `bytes`, returning the lowercase hex digest (64 chars).
pub fn sha256_bytes(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex_lower(h.finalize().as_ref())
}

/// Read `path` and return its SHA-256 hex digest. Streams the file in 64 KiB
/// chunks so large data CSVs don't have to be held in memory twice.
pub fn sha256_file(path: &Path) -> Result<String, String> {
    let mut f = File::open(path).map_err(|e| format!("opening {}: {}", path.display(), e))?;
    let mut h = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = f
            .read(&mut buf)
            .map_err(|e| format!("reading {}: {}", path.display(), e))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex_lower(h.finalize().as_ref()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn empty_string_matches_known_digest() {
        assert_eq!(
            sha256_bytes(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn known_digest_for_abc() {
        assert_eq!(
            sha256_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn file_and_bytes_agree() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let payload = b"hello world\nferx-core\n";
        tmp.as_file().write_all(payload).unwrap();
        tmp.as_file().sync_all().unwrap();
        assert_eq!(sha256_file(tmp.path()).unwrap(), sha256_bytes(payload));
    }
}
