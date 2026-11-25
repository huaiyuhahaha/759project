#!/usr/bin/env bash
# Submit from the repository root after the login probe creates results/euler/.
#SBATCH --partition=instruction
#SBATCH --constraint=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --job-name=baselines_E0
#SBATCH --output=results/euler/E0_%j.slurm.out
#SBATCH --error=results/euler/E0_%j.slurm.err
set -euo pipefail

cd -- "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
repo_root=$PWD
if [[ ! -f "$repo_root/bench/davidson_bench.cpp" ]]; then
    printf 'ERROR: submit this job from the repository root.\n' >&2
    exit 1
fi
# shellcheck source=euler/probe_common.sh
source "$repo_root/euler/probe_common.sh"

job_id=${SLURM_JOB_ID:?This script must be run with sbatch}
if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
    printf 'ERROR: unexpected Slurm job ID: %s\n' "$job_id" >&2
    exit 1
fi
output_dir="$repo_root/results/euler/E0_$job_id"
# Never overwrite an earlier probe with the same job ID (for example, a requeue).
mkdir -- "$output_dir"
printf 'Euler compute probe output: %s\n' "$output_dir"

main() {
    section 'E0 compute-node probe'
    printf 'UTC start: %s\n' "$(date -u +%FT%TZ)"
    printf 'Repository: %s\nOutput: %s\nJob ID: %s\n' "$repo_root" "$output_dir" "$job_id"
    required hostname
    required uname -a

    section 'Slurm allocation'
    # An explicit allowlist avoids recording unrelated environment secrets.
    local name
    for name in SLURM_JOB_ID SLURM_JOB_NAME SLURM_JOB_PARTITION SLURM_JOB_NODELIST \
        SLURM_JOB_NUM_NODES SLURM_NTASKS SLURM_CPUS_PER_TASK SLURM_JOB_CPUS_PER_NODE \
        SLURM_CPUS_ON_NODE SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURMD_NODENAME; do
        printf '%s=%s\n' "$name" "${!name-<unset>}"
    done
    require_tool timeout
    require_tool scontrol
    required timeout --kill-after=2s 20s scontrol show job "$job_id"
    required timeout --kill-after=2s 20s scontrol show node "${SLURMD_NODENAME:?Missing Slurm node name}"

    section 'CPU topology and actual process affinity'
    require_tool lscpu
    required lscpu
    required lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE
    local cpu_list
    cpu_list=$(awk '/^Cpus_allowed_list:/ { print $2 }' /proc/self/status)
    if [[ -z "$cpu_list" ]]; then
        printf 'ERROR: could not read the process CPU affinity from /proc/self/status.\n' >&2
        return 1
    fi
    printf 'Cpus_allowed_list=%s\n' "$cpu_list"
    printf 'Do not infer physical cores by dividing cpus-per-task by two; use this topology.\n'
    if command -v taskset >/dev/null 2>&1; then
        optional taskset -pc "$$"
    else
        printf 'UNAVAILABLE: taskset; /proc affinity is recorded above.\n'
    fi
    if command -v numactl >/dev/null 2>&1; then
        bounded_optional 15 numactl -H
    else
        printf 'UNAVAILABLE: numactl; lscpu topology is recorded above.\n'
    fi

    probe_modules "$output_dir"
    probe_tools

    section 'Conda-forge reachability diagnostic'
    printf 'This optional HEAD request downloads no packages. Compute jobs will not require internet.\n'
    if command -v curl >/dev/null 2>&1; then
        optional curl --head --location --fail --silent --show-error \
            --connect-timeout 5 --max-time 12 https://conda.anaconda.org/conda-forge/noarch/repodata.json
    elif command -v wget >/dev/null 2>&1; then
        bounded_optional 15 wget --spider --server-response --timeout=10 --tries=1 \
            https://conda.anaconda.org/conda-forge/noarch/repodata.json
    else
        printf 'UNAVAILABLE: curl and wget; network reachability was not tested.\n'
    fi
    printf '\nE0 compute probe completed at %s.\n' "$(date -u +%FT%TZ)"
    printf 'Output: %s\n' "$output_dir"
}

main 2>&1 | tee "$output_dir/env.txt"
