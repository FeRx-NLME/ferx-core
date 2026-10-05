#!/usr/bin/env bash
#
# Run a cargo command on Linux, in Docker, against this checkout — so a fit number
# measured locally is a number `main`'s CI would also produce. `--help` prints usage;
# this header is the why (#1688).
#
# Linux is the reference platform for fit numbers. macOS and Linux disagree on fits
# because of the OS math library, not the CPU: at cff801db `per_route_lag` reaches
# OFV -688.936445811360 on Linux arm64 AND on Linux amd64, bit-identical down to the
# covariance eigenvalues, while native macOS arm64 stops at -420.33 and fails. CI
# (`slow-tests.yml`) runs on ubuntu-latest, so "is this red on main?" is a Linux
# question, and a Mac-only red answers a different one.
#
# The container layout is load-bearing; each piece was measured on #1688:
#
#   - The tree is mounted READ-ONLY at /ro and tar-copied to /src (without target/,
#     .git and the ignored local dirs). The `gen_*_anchor` tests rewrite committed
#     `nonmem_anchor/*.csv`, so a plain `:ro` mount fails them with ReadOnlyFilesystem
#     and a read-write mount lets a Linux run overwrite tracked files in your tree.
#     tar keeps mtimes and /src is a fixed path, so cargo's fingerprints hold across
#     runs: a warm re-run does not recompile.
#   - /src/.git is written as `gitdir: <host absolute git dir>`, and the git common dir
#     is mounted read-only at its OWN absolute path. Tests that call `git ls-files`
#     (path filter, AGENTS.md, public-API boundary) need a working repo, and a
#     worktree's `.git` file points at an absolute host path.
#   - CARGO_BUILD_JOBS defaults to 1. On Docker Desktop's default VM (8 GB) the full
#     slow sweep is OOM-killed at the default job count and at 2; at 1 it completes in
#     ~21 min with a 6.9 GB peak. Raise it with --jobs only after raising Docker's memory.
#   - RAYON_NUM_THREADS=1, as ci.yml and slow-tests.yml set it.
#
# Before cargo runs, the container prints one marker line:
#
#     ferx-platform: Linux/<arch> <libc version> <rustc -V>
#
# Log readers (ferx-red-diff) grep for it anywhere in the log, not on line 1 — a
# `docker pull` can precede it.
#
# Not a CI gate: opt-in local tooling, so it is deliberately absent from preflight.sh.
set -euo pipefail

usage() {
  cat <<'EOF'
usage: tools/linux-test.sh [--amd64] [--jobs N] [--dry-run] -- <cargo args>

Runs `cargo <cargo args>` on Linux in Docker (image rustlang/rust:nightly) against a
read-only copy of this checkout. Exit status is cargo's.

  --amd64     run linux/amd64 (emulated on Apple silicon; slow). Default: the host's
              native Linux arch. Each arch has its own cache volumes.
  --jobs N    CARGO_BUILD_JOBS (default 1: higher is OOM-killed on an 8 GB Docker VM)
  --dry-run   print the docker command and the in-container script; run nothing

Example (the slow-tests.yml core leg):
  tools/linux-test.sh -- test --workspace --exclude docs-lint --no-default-features \
      --features ci,survival,slow-tests --profile ci-test --no-fail-fast

The docker binary is $FERX_DOCKER (default: docker).

Caches live in Docker volumes, not in this tree:
  ferx-linux-target, ferx-linux-registry   (and *-amd64 under --amd64)
To reclaim the space:
  docker volume rm ferx-linux-target ferx-linux-registry
EOF
}

amd64=0
jobs=1
dry_run=0
while [ $# -gt 0 ]; do
  case "$1" in
    --amd64) amd64=1; shift ;;
    --jobs)
      if [ $# -lt 2 ] || ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
        echo "linux-test: --jobs needs a positive integer" >&2
        usage >&2
        exit 2
      fi
      jobs="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    *)
      echo "linux-test: unknown option '$1' (cargo args go after --)" >&2
      usage >&2
      exit 2 ;;
  esac
done
if [ $# -eq 0 ]; then
  echo "linux-test: no cargo args (put them after --)" >&2
  usage >&2
  exit 2
fi

docker_bin="${FERX_DOCKER:-docker}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tree="$(git -C "$script_dir" rev-parse --show-toplevel)"
git_dir="$(git -C "$tree" rev-parse --absolute-git-dir)"
# A worktree's git dir lives under the common dir; a main checkout's IS the common dir.
git_common="$(cd "$tree" && cd "$(git rev-parse --git-common-dir)" && pwd)"

image="rustlang/rust:nightly"
suffix=""
platform_args=()
if [ "$amd64" -eq 1 ]; then
  # A target dir shared across arches fails at run time with `exec format error`
  # rather than rebuilding, so each arch gets its own volumes.
  suffix="-amd64"
  platform_args=(--platform linux/amd64)
fi

cargo_cmd="cargo$(printf ' %q' "$@")"

# Runs inside the container. `%q`-quoted values are substituted on the host.
inner="set -euo pipefail
tar -C /ro --exclude=./target --exclude=./.git --exclude=./.claude --exclude=./.cargo -cf - . \\
  | { mkdir -p /src && tar -C /src -xf -; }
printf 'gitdir: %s\\n' $(printf '%q' "$git_dir") > /src/.git
git config --global --add safe.directory '*'
if ! command -v cmake >/dev/null 2>&1; then
  { apt-get update -qq && apt-get install -y -qq cmake; } >/dev/null 2>&1 \\
    || { echo 'linux-test: could not install cmake (nlopt needs it)' >&2; exit 2; }
fi
cd /src
echo \"ferx-platform: Linux/\$(uname -m) \$(ldd --version 2>&1 | sed -n 1p) \$(rustc -V)\"
set +e
$cargo_cmd 2> >(tee /tmp/cargo.stderr >&2)
code=\$?
wait
if [ \"\$code\" -ne 0 ] && grep -q 'SIGKILL' /tmp/cargo.stderr; then
  echo 'linux-test: rustc was killed — lower --jobs or raise Docker'\"'\"'s memory' >&2
fi
exit \"\$code\""

docker_argv=(
  "$docker_bin" run --rm
  "${platform_args[@]+"${platform_args[@]}"}"
  -v "$tree:/ro:ro"
  -v "$git_common:$git_common:ro"
  -v "ferx-linux-target$suffix:/target"
  -v "ferx-linux-registry$suffix:/usr/local/cargo/registry"
  -e CARGO_TARGET_DIR=/target
  -e RAYON_NUM_THREADS=1
  -e "CARGO_BUILD_JOBS=$jobs"
  "$image"
  bash -c "$inner"
)

if [ "$dry_run" -eq 1 ]; then
  echo "# docker command:"
  printf '%q ' "${docker_argv[@]:0:${#docker_argv[@]}-1}"
  printf '"$inner"\n'
  echo "# inner script (\$inner):"
  printf '%s\n' "$inner"
  exit 0
fi

if ! command -v "$docker_bin" >/dev/null 2>&1; then
  echo "linux-test: docker not found ('$docker_bin' is not an executable on PATH; set FERX_DOCKER)" >&2
  exit 2
fi
if ! "$docker_bin" info >/dev/null 2>&1; then
  echo "linux-test: docker daemon not reachable (docker info failed) — start Docker Desktop and retry" >&2
  exit 2
fi

exec "${docker_argv[@]}"
