#!/usr/bin/env bash
# Shared E4/E5/E6 job body; arguments after the step are forwarded to the driver.
# Called by sbatch scripts, never installs packages or initializes modules.
set -euo pipefail

step=${1:?Expected E4, E5 or E6}
shift
[[ "$step" == E4 || "$step" == E5 || "$step" == E6 ]] || { printf 'ERROR: expected E4, E5 or E6.\n' >&2; exit 1; }
cd -- "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
repo_root=$PWD
# shellcheck source=euler/probe_common.sh
source "$repo_root/euler/probe_common.sh"
# shellcheck source=euler/compiler_common.sh
source "$repo_root/euler/compiler_common.sh"
job_id=${SLURM_JOB_ID:?Run this script with sbatch}
[[ "$job_id" =~ ^[0-9]+$ ]] || { printf 'ERROR: invalid job ID.\n' >&2; exit 1; }
output_dir="$repo_root/results/euler/${step}_$job_id"
mkdir -- "$output_dir"

main() {
    section "$step Davidson baselines"
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

    section 'Build with system /usr/bin/g++ (no compiler module)'
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

    section 'Verify libraries and execute the requested sweep'
    clean=(env -u PYTHONPATH -u PYTHONHOME -u LD_PRELOAD -u LD_LIBRARY_PATH
           -u PETSC_DIR -u PETSC_ARCH -u SLEPC_DIR
           -u OMP_PROC_BIND -u OMP_PLACES -u GOMP_CPU_AFFINITY -u KMP_AFFINITY
           PATH="$env_prefix/bin:/usr/bin:/bin" CONDA_PREFIX="$env_prefix"
           PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
           OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
           MPIR_CVAR_ENABLE_GPU=0)
    local -a inspect_prefix=() driver_prefix=() placement_args=()
    if [[ "$step" == E6 ]]; then
        section 'E6 hardware capture and mandatory single-socket/NUMA gate'
        "${clean[@]}" "$env_prefix/bin/python" -B "$repo_root/euler/record_hardware.py" --output-dir "$output_dir"
        "${clean[@]}" "$env_prefix/bin/python" -B "$repo_root/euler/single_socket.py" \
            prepare --output-dir "$output_dir" --cores 8
        driver_prefix=("$env_prefix/bin/python" -B "$repo_root/euler/single_socket.py"
                       exec --plan "$output_dir/single_socket.json")
        inspect_prefix=("${driver_prefix[@]}" --first-cpu --)
        driver_prefix+=(--)
        placement_args=(--single-socket-plan "$output_dir/single_socket.json")
    fi
    "${clean[@]}" "${inspect_prefix[@]}" "$env_prefix/bin/python" -B "$repo_root/euler/inspect_env.py" --output-dir "$output_dir"
    "${clean[@]}" "${driver_prefix[@]}" "$env_prefix/bin/python" -B "$repo_root/bench/run_baselines.py" \
        --step "$step" --output-dir "$output_dir" --environment-json "$output_dir/environment.json" \
        --cxx /usr/bin/g++ --dumper "$output_dir/build/davidson_bench" \
        --cpp-binary "$output_dir/build/cpp_baselines_bench" --mpiexec "$env_prefix/bin/mpiexec" \
        "${placement_args[@]}" "$@"
    printf '%s PASS at %s. Output: %s\n' "$step" "$(date -u +%FT%TZ)" "$output_dir"
}

main "$@" 2>&1 | tee "$output_dir/env.txt"
