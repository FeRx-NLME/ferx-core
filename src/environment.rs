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
    /// `true` when `/.dockerenv` exists, `KUBERNETES_SERVICE_HOST` is set, or
    /// `/proc/1/cgroup` names a container runtime. Best-effort: a cgroup-v2 host
    /// with a cgroup namespace (common for containerd/CRI-O pods) hides the
    /// container's cgroup path, which is why the Kubernetes env var check
    /// exists as a fallback. Always `false` on platforms without `/proc`.
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
        in_docker: docker_signal(
            std::path::Path::new("/.dockerenv").exists(),
            std::env::var("KUBERNETES_SERVICE_HOST").is_ok(),
            std::fs::read_to_string("/proc/1/cgroup").ok(),
        ),
        username: username_from_env(std::env::var("USER").ok(), std::env::var("USERNAME").ok()),
    }
}

/// Pure decision logic for `in_docker`, factored out of `detect()` so every
/// branch (dockerenv marker, Kubernetes env var, cgroup substrings, and the
/// all-false case) is directly unit-testable without touching the real
/// filesystem or process environment.
fn docker_signal(has_dockerenv: bool, has_k8s_service_host: bool, cgroup: Option<String>) -> bool {
    if has_dockerenv || has_k8s_service_host {
        return true;
    }
    cgroup
        .map(|s| s.contains("docker") || s.contains("kubepods") || s.contains("containerd"))
        .unwrap_or(false)
}

/// Pure decision logic for `username`: `$USER` wins, then `%USERNAME%`, then
/// the `"unknown"` sentinel — factored out so both fallback branches are
/// directly unit-testable without mutating process-global env vars.
fn username_from_env(user: Option<String>, username: Option<String>) -> String {
    user.or(username).unwrap_or_else(|| "unknown".to_string())
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

    #[test]
    fn docker_signal_true_on_dockerenv_marker() {
        assert!(docker_signal(true, false, None));
    }

    #[test]
    fn docker_signal_true_on_kubernetes_service_host() {
        // Covers cgroup-v2 hosts with a cgroup namespace, where a pod's own
        // cgroup path (e.g. containerd/CRI-O under runc) reads back as "0::/"
        // with no identifying substring — the env var is always set instead.
        assert!(docker_signal(false, true, Some("0::/".to_string())));
    }

    #[test]
    fn docker_signal_true_on_cgroup_substring() {
        for marker in ["docker", "kubepods", "containerd"] {
            let cgroup = format!("0::/system.slice/{marker}-abc123.scope");
            assert!(
                docker_signal(false, false, Some(cgroup.clone())),
                "expected {cgroup:?} to be detected as a container"
            );
        }
    }

    #[test]
    fn docker_signal_false_when_no_signal_present() {
        assert!(!docker_signal(false, false, Some("0::/".to_string())));
        assert!(!docker_signal(false, false, None));
    }

    #[test]
    fn username_prefers_user_over_username() {
        assert_eq!(
            username_from_env(Some("alice".into()), Some("BOB".into())),
            "alice"
        );
    }

    #[test]
    fn username_falls_back_to_windows_username() {
        assert_eq!(username_from_env(None, Some("BOB".into())), "BOB");
    }

    #[test]
    fn username_falls_back_to_unknown_sentinel() {
        assert_eq!(username_from_env(None, None), "unknown");
    }
}
