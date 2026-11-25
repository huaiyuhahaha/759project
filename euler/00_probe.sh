#!/usr/bin/env bash
# Run on the Euler login node: bash euler/00_probe.sh
# No installations, builds, or job submissions are performed.
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd -- "$repo_root"
# shellcheck source=euler/probe_common.sh
source "$repo_root/euler/probe_common.sh"

results_root="$repo_root/results/euler"
mkdir -p -- "$results_root"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
output_dir=$(mktemp -d "$results_root/E0_login_${stamp}_XXXXXX")
printf 'Euler login probe output: %s\n' "$output_dir"

main() {
    section 'E0 login probe'
    printf 'UTC start: %s\n' "$(date -u +%FT%TZ)"
    printf 'Repository: %s\nOutput: %s\n' "$repo_root" "$output_dir"
    required hostname
    required uname -a
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        printf 'WARNING: SLURM_JOB_ID=%s; this probe is intended for a login node.\n' "$SLURM_JOB_ID"
    fi
    probe_modules "$output_dir"
    probe_tools
    probe_storage

    section 'Instruction partition and Slurm metadata'
    require_tool timeout
    require_tool sbatch
    require_tool sinfo
    require_tool scontrol
    required timeout --kill-after=2s 20s sinfo -p instruction -o '%P %a %l %D %c %m %f'
    required timeout --kill-after=2s 20s sinfo -p instruction -N -o '%N %t %c %m %f'
    required timeout --kill-after=2s 20s scontrol show partition instruction
    printf '\nE0 login probe completed at %s.\n' "$(date -u +%FT%TZ)"
    printf 'Output: %s\n' "$output_dir"
    printf 'From the repository root, submit the compute probe with:\n'
    printf '  sbatch euler/00_probe_node.sbatch\n'
}

# pipefail preserves an essential-probe or log-write failure while retaining output.
main 2>&1 | tee "$output_dir/env.txt"
