//! Runtime environment snapshot attached to each fit (#704): OS, CPU
//! architecture, whether the process is running inside a container, and the
//! OS username — recorded once per fit for troubleshooting and
//! reproducibility, alongside `FitResult.wall_time_secs`/`n_threads_used`.

/// Snapshot of the machine and account a fit executed on.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EnvironmentInfo {
    /// Target OS family (`std::env::consts::OS`), e.g. `"linux"`, `"macos"`, `"windows"`.
    pub os: String,
    /// Target CPU architecture (`std::env::consts::ARCH`), e.g. `"x86_64"`, `"aarch64"`.
    pub arch: String,
    /// `true` when `/.dockerenv` exists or `/proc/1/cgroup` names a container runtime.
    /// Always `false` on platforms without a `/proc` filesystem.
    pub in_docker: bool,
    /// OS username the fit ran under (`$USER`/`%USERNAME%`), `"unknown"` if neither is set.
    pub username: String,
}

impl Default for EnvironmentInfo {
    /// Placeholder for `.fitrx` bundles saved before this field existed —
    /// distinguishable from a real detection, which never returns "unknown" arch/OS.
    fn default() -> Self {
        EnvironmentInfo {
            os: "unknown".to_string(),
            arch: "unknown".to_string(),
            in_docker: false,
            username: "unknown".to_string(),
        }
    }
}

/// Detect the current process's environment. Cheap (a couple of env lookups
/// and, on Linux, one small file read) — call once per fit.
pub fn detect() -> EnvironmentInfo {
    EnvironmentInfo {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        in_docker: detect_docker(),
        username: detect_username(),
    }
}

fn detect_docker() -> bool {
    if std::path::Path::new("/.dockerenv").exists() {
        return true;
    }
    std::fs::read_to_string("/proc/1/cgroup")
        .map(|s| s.contains("docker") || s.contains("kubepods") || s.contains("containerd"))
        .unwrap_or(false)
}

fn detect_username() -> String {
    std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .unwrap_or_else(|_| "unknown".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_returns_nonempty_os_and_arch() {
        let env = detect();
        assert!(!env.os.is_empty());
        assert!(!env.arch.is_empty());
    }

    #[test]
    fn default_is_distinguishable_placeholder() {
        let d = EnvironmentInfo::default();
        assert_eq!(d.os, "unknown");
        assert_eq!(d.arch, "unknown");
        assert!(!d.in_docker);
    }
}
