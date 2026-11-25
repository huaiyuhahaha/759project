#!/usr/bin/env bash
# Submit from the repo root after E2 passes. Existing environment only.
#SBATCH --partition=instruction
#SBATCH --constraint=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:09:00
#SBATCH --job-name=baselines_E3
#SBATCH --output=results/euler/E3_%j.slurm.out
#SBATCH --error=results/euler/E3_%j.slurm.err
set -euo pipefail

cd -- "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
repo_root=$PWD
# shellcheck source=euler/probe_common.sh
source "$repo_root/euler/probe_common.sh"
# shellcheck source=euler/compiler_common.sh
source "$repo_root/euler/compiler_common.sh"
job_id=${SLURM_JOB_ID:?Run this script with sbatch}
[[ "$job_id" =~ ^[0-9]+$ ]] || { printf 'ERROR: invalid job ID.\n' >&2; exit 1; }
output_dir="$repo_root/results/euler/E3_$job_id"
mkdir -- "$output_dir"

main() {
    section 'E3 complete baseline pipeline smoke'
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
        printf 'ERROR: no E1 environment prefix.\n' >&2
        return 1
    fi
    if [[ "$env_prefix" != /* || "$env_prefix" == *$'\n'* \
        || ! -x "$env_prefix/bin/python" || ! -x "$env_prefix/bin/mpiexec" \
        || ! -f "$env_prefix/.baselines-owner.json" ]]; then
        printf 'ERROR: invalid or incomplete baseline environment: %s\n' "$env_prefix" >&2
        return 1
    fi
    printf '%s\n' "$env_prefix" > "$output_dir/environment-prefix.txt"
    cp -- "$repo_root/euler/env/environment.yml" "$output_dir/requested-environment.yml"
    cp -- "$env_prefix/.baselines-owner.json" "$output_dir/environment-owner.json"

    section 'Compile both C++ wrappers with system /usr/bin/g++ (no compiler module)'
    mkdir -- "$output_dir/build"
    (
        prepare_system_compiler "$output_dir/build"
        for name in davidson_bench cpp_baselines_bench; do
            build=(/usr/bin/g++ -std=c++17 -O3 -march=native -fopenmp -Wall -Wextra -pedantic
                   "-I$repo_root" "$repo_root/matrix.cpp" "$repo_root/davidson.cpp"
                   "$repo_root/bench/$name.cpp" -o "$output_dir/build/$name")
            show_command "${build[@]}"
            "${build[@]}"
        done
    )

    section 'Verify this job environment and run all solver configurations'
    # Clean the inherited module/user library overrides, as in the passing E2.
    clean=(env -u PYTHONPATH -u PYTHONHOME -u LD_PRELOAD -u LD_LIBRARY_PATH
           -u PETSC_DIR -u PETSC_ARCH -u SLEPC_DIR
           -u OMP_PROC_BIND -u OMP_PLACES -u GOMP_CPU_AFFINITY -u KMP_AFFINITY
           PATH="$env_prefix/bin:/usr/bin:/bin" CONDA_PREFIX="$env_prefix"
           PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
           OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
           MPIR_CVAR_ENABLE_GPU=0)
    "${clean[@]}" "$env_prefix/bin/python" -B "$repo_root/euler/inspect_env.py" --output-dir "$output_dir"
    "${clean[@]}" "$env_prefix/bin/python" -B "$repo_root/bench/run_baselines.py" \
        --sizes 64 128 256 --cores 1 2 --smoke --min-time 0 --max-runs 3 \
        --step E3 --output-dir "$output_dir" --environment-json "$output_dir/environment.json" \
        --cxx /usr/bin/g++ \
        --dumper "$output_dir/build/davidson_bench" --cpp-binary "$output_dir/build/cpp_baselines_bench" \
        --mpiexec "$env_prefix/bin/mpiexec" --timeout 60
    printf 'E3 PASS at %s. Output: %s\n' "$(date -u +%FT%TZ)" "$output_dir"
}

main 2>&1 | tee "$output_dir/env.txt"
