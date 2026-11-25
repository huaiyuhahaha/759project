#!/usr/bin/env bash
# Shared read-only discovery for the Stage 1 Euler probes.
set -euo pipefail

section() { printf '\n== %s ==\n' "$*"; }

show_command() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
}

required() {
    show_command "$@"
    local rc=0
    "$@" || rc=$?
    if (( rc != 0 )); then
        printf 'ERROR: required command failed (exit %s).\n' "$rc" >&2
        return "$rc"
    fi
}

optional() {
    show_command "$@"
    local rc=0
    "$@" || rc=$?
    if (( rc != 0 )); then
        printf 'UNAVAILABLE OR FAILED: optional probe exited %s; continuing.\n' "$rc"
    fi
    return 0
}

bounded_optional() {
    local seconds=$1
    shift
    if command -v timeout >/dev/null 2>&1; then
        optional timeout --kill-after=2s "${seconds}s" "$@"
    else
        show_command "$@"
        printf 'SKIPPED: timeout is unavailable; this optional probe needs a time limit.\n'
    fi
}

require_tool() {
    if ! command -v "$1" >/dev/null 2>&1; then
        printf 'ERROR: required tool is unavailable: %s\n' "$1" >&2
        return 1
    fi
}

init_modules() {
    if command -v module >/dev/null 2>&1; then
        return 0
    fi
    local init rc
    for init in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh \
        /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash; do
        if [[ -r "$init" ]]; then
            printf '+ source %q (initialize this probe shell only)\n' "$init"
            # Site initialization scripts may legitimately reference unset variables.
            set +u
            rc=0
            # shellcheck disable=SC1090
            source "$init" || rc=$?
            set -u
            if (( rc != 0 )); then
                printf 'Module initialization returned %s.\n' "$rc"
            fi
            if command -v module >/dev/null 2>&1; then
                return 0
            fi
        fi
    done
    printf 'UNAVAILABLE: no module command or usable module initialization was found.\n'
    return 1
}

probe_modules() {
    local output_dir=$1 rc=0
    section 'Modules available now'
    # Failure to create a requested artifact is essential, even for optional discovery.
    : > "$output_dir/modules-avail.txt"
    if ! init_modules; then
        printf 'Module discovery could not run.\n' > "$output_dir/modules-avail.txt"
        return 0
    fi
    show_command module avail
    module avail > "$output_dir/modules-avail.txt" 2>&1 || rc=$?
    printf 'Full output: %s/modules-avail.txt (exit %s)\n' "$output_dir" "$rc"
    if (( rc != 0 )); then
        printf 'UNAVAILABLE OR FAILED: module avail; retaining its output.\n'
    fi
    printf 'Filtered module output:\n'
    if ! grep -Ei 'conda|miniforge|anaconda|mamba|python|petsc|slepc|mpi|openblas|gcc' \
        "$output_dir/modules-avail.txt"; then
        printf 'No matching module lines; see the full output.\n'
    fi
    optional module list

    section 'System compiler selected after E3 129413'
    printf 'No compiler module is loaded or purged; historical gcc/14.3.0 is incompatible.\n'
    optional /usr/bin/gcc --version
    optional /usr/bin/g++ --version
}

probe_tools() {
    section 'Existing tools and versions in the original environment'
    local tool
    for tool in conda mamba micromamba python python3 gcc g++ mpiexec mpirun \
        mpicc mpicxx srun sbatch sinfo scontrol numactl taskset quota lfs; do
        if command -v "$tool" >/dev/null 2>&1; then
            printf '\n%s resolves to: ' "$tool"
            command -v "$tool"
            case "$tool" in
                quota|lfs) ;;  # Availability here; actual quota queries are separate.
                *) bounded_optional 15 "$tool" --version ;;
            esac
        else
            printf 'UNAVAILABLE: %s\n' "$tool"
        fi
    done
    if command -v conda >/dev/null 2>&1; then
        bounded_optional 20 conda info --base
    fi
}

probe_storage() {
    section 'Filesystem capacity and user quotas'
    local account path
    account=$(id -un)
    local -a candidates=("$HOME")
    for path in "${SCRATCH:-}" "${SCRATCH_DIR:-}" "${TMPDIR:-}" \
        "${SLURM_TMPDIR:-}" /scratch "/scratch/$account" /lscratch /localscratch; do
        if [[ -n "$path" && -d "$path" ]]; then
            candidates+=("$path")
        fi
    done
    printf 'Only existing paths are probed. Free filesystem space is not a user quota.\n'
    local -A seen=()
    for path in "${candidates[@]}"; do
        if [[ -n "${seen[$path]:-}" ]]; then
            continue
        fi
        seen[$path]=1
        bounded_optional 15 df -h -- "$path"
        bounded_optional 15 df -i -- "$path"
        if command -v lfs >/dev/null 2>&1; then
            # Non-Lustre paths or unsupported quota operations are logged, not fatal.
            bounded_optional 15 lfs quota -u "$account" "$path"
        fi
    done
    if command -v quota >/dev/null 2>&1; then
        bounded_optional 20 quota -s
    else
        printf 'UNAVAILABLE: quota; no site-specific quota command is assumed.\n'
    fi
    if ! command -v lfs >/dev/null 2>&1; then
        printf 'UNAVAILABLE: lfs; Lustre quota queries were not run.\n'
    fi
}
