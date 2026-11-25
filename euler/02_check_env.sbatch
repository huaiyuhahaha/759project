#!/usr/bin/env bash
# Submit after E1 succeeds, from the repository root. No downloads or installs.
#SBATCH --partition=instruction
#SBATCH --constraint=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --job-name=baselines_E2
#SBATCH --output=results/euler/E2_%j.slurm.out
#SBATCH --error=results/euler/E2_%j.slurm.err
set -euo pipefail

cd -- "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
repo_root=$PWD
# shellcheck source=euler/probe_common.sh
source "$repo_root/euler/probe_common.sh"
# shellcheck source=euler/compiler_common.sh
source "$repo_root/euler/compiler_common.sh"
job_id=${SLURM_JOB_ID:?Run this script with sbatch}
[[ "$job_id" =~ ^[0-9]+$ ]] || { printf 'ERROR: invalid job ID.\n' >&2; exit 1; }
output_dir="$repo_root/results/euler/E2_$job_id"
mkdir -- "$output_dir"

main() {
    section 'E2 environment and numerical checks'
    printf 'UTC start: %s\nJob ID: %s\nOutput: %s\n' "$(date -u +%FT%TZ)" "$job_id" "$output_dir"
    required hostname
    required lscpu
    required lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE
    required scontrol show job "$job_id"

    local env_prefix
    if [[ -n "${BASELINES_ENV_PREFIX:-}" ]]; then
        env_prefix=$BASELINES_ENV_PREFIX
    elif [[ -f "$repo_root/.euler/baselines-prefix.txt" ]]; then
        env_prefix=$(cat "$repo_root/.euler/baselines-prefix.txt")
    else
        printf 'ERROR: no E1 environment prefix; run bash euler/01_setup_env.sh first.\n' >&2
        return 1
    fi
    if [[ "$env_prefix" != /* || "$env_prefix" == *$'\n'* || ! -x "$env_prefix/bin/python" \
          || ! -x "$env_prefix/bin/mpiexec" || ! -f "$env_prefix/conda-meta/history" ]]; then
        printf 'ERROR: invalid/incomplete E1 environment: %s\n' "$env_prefix" >&2
        return 1
    fi
    printf 'Environment prefix: %s\n' "$env_prefix"
    printf '%s\n' "$env_prefix" > "$output_dir/environment-prefix.txt"
    cp -- "$repo_root/euler/env/environment.yml" "$output_dir/requested-environment.yml"

    section 'C++ build with system /usr/bin/g++ (no compiler module)'
    # Isolate the clean system build environment from conda Python/MPI.
    local cpp_rc=0
    (
        set -euo pipefail
        prepare_system_compiler "$output_dir/build" || exit 1
        build=(/usr/bin/g++ -std=c++17 -O3 -march=native -fopenmp -Wall -Wextra -pedantic
               "-I$repo_root" "$repo_root/matrix.cpp" "$repo_root/davidson.cpp"
               "$repo_root/bench/davidson_bench.cpp" -o "$output_dir/davidson_bench")
        show_command "${build[@]}"
        "${build[@]}" || exit 1
        "$output_dir/davidson_bench" 64 dump "$output_dir/input_n64.bin" \
            > "$output_dir/cpp_dump_n64.json" || exit 1
        cat "$output_dir/cpp_dump_n64.json" || exit 1
        bytes=$(wc -c < "$output_dir/input_n64.bin")
        if [[ "$bytes" -ne $((8 * (64 * 64 + 64 * 4))) ]]; then
            printf 'ERROR: n=64 dump has wrong length: %s bytes\n' "$bytes" >&2
            exit 1
        fi
    ) || cpp_rc=$?
    printf 'C++ build/dump exit: %s\n' "$cpp_rc"
    printf '%s\n' "$cpp_rc" > "$output_dir/cpp_exit_code.txt"

    section 'Conda libraries, binding, and numerical checks'
    local checks_rc=0
    # Invoke the selected prefix directly; inherited user/module libraries must
    # not silently select another Python, MPI, PETSc, or BLAS installation.
    env -u PYTHONPATH -u PYTHONHOME -u LD_PRELOAD -u LD_LIBRARY_PATH \
        -u PETSC_DIR -u PETSC_ARCH -u SLEPC_DIR \
        -u OMP_PROC_BIND -u OMP_PLACES -u GOMP_CPU_AFFINITY -u KMP_AFFINITY \
        PATH="$env_prefix/bin:/usr/bin:/bin" CONDA_PREFIX="$env_prefix" \
        PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
        "$env_prefix/bin/python" -B "$repo_root/euler/run_env_checks.py" \
        --output-dir "$output_dir" || checks_rc=$?
    if (( cpp_rc != 0 || checks_rc != 0 )); then
        printf 'E2 FAIL: C++ exit=%s, Python checks exit=%s. Return all logs.\n' "$cpp_rc" "$checks_rc" >&2
        return 1
    fi
    printf 'E2 PASS at %s. Output: %s\n' "$(date -u +%FT%TZ)" "$output_dir"
}

main 2>&1 | tee "$output_dir/env.txt"
