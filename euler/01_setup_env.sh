#!/usr/bin/env bash
# Euler LOGIN NODE only: bash euler/01_setup_env.sh
# Optional override: BASELINES_ENV_PREFIX=/absolute/shared/path/baselines
# Recover a completed install whose final marker failed:
# bash euler/01_setup_env.sh --adopt-existing results/euler/E1_login_<timestamp>_<suffix>
# Reuses a matching environment created by this script; never updates other envs.
set -euo pipefail

adopt_from=''
if (( $# != 0 )); then
    if [[ $# != 2 || $1 != --adopt-existing || -z $2 ]]; then
        printf 'Usage: bash euler/01_setup_env.sh [--adopt-existing PRIOR_E1_DIRECTORY]\n' >&2
        exit 2
    fi
    adopt_from=$2
fi

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd -- "$repo_root"
# shellcheck source=euler/probe_common.sh
source "$repo_root/euler/probe_common.sh"

if [[ -n "${SLURM_JOB_ID:-}" || -n "${SLURM_STEP_ID:-}" ]]; then
    printf 'ERROR: run E1 on the login node, outside any Slurm allocation.\n' >&2
    exit 1
fi
if [[ $(hostname -s) != euler-login-* ]]; then
    printf 'ERROR: E0 identified euler-login-3; run E1 on an Euler login host (euler-login-*).\n' >&2
    exit 1
fi
if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
    printf 'ERROR: this environment is intended for the Linux x86_64 platform observed in E0.\n' >&2
    exit 1
fi
for tool in python3 realpath sha256sum timeout tee mktemp; do
    require_tool "$tool"
done
bootstrap_python=$(type -P python3)
# -I isolates Python settings, but not the dynamic linker's library search.
# Clear module/user library overrides on EVERY Python invocation, including
# after choose_conda has loaded Miniforge into this shell's environment.
bootstrap() {
    env -u LD_LIBRARY_PATH -u LD_PRELOAD -u PYTHONHOME -u PYTHONPATH \
        "$bootstrap_python" -I -B "$@"
}

environment_python() {
    env -u LD_LIBRARY_PATH -u LD_PRELOAD -u PYTHONHOME -u PYTHONPATH \
        PATH="$env_prefix/bin:/usr/bin:/bin" CONDA_PREFIX="$env_prefix" \
        "$env_prefix/bin/python" -I -B "$@"
}

runtime_root="$repo_root/.euler"
env_prefix=$(realpath -m -- "${BASELINES_ENV_PREFIX:-$runtime_root/envs/baselines}")
spec_file="$repo_root/euler/env/environment.yml"
if [[ -n "$adopt_from" ]]; then
    adopt_from=$(realpath -e -- "$adopt_from")
    if [[ ! -d "$adopt_from" || ! -d "$env_prefix" ]]; then
        printf 'ERROR: adoption requires an existing prefix and prior E1 output directory; nothing will be installed.\n' >&2
        exit 1
    fi
fi
mkdir -p -- "$repo_root/results/euler"
output_dir=$(mktemp -d "$repo_root/results/euler/E1_login_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
printf 'Euler E1 output: %s\n' "$output_dir"

conda_cmd=''
conda_base=''
pyscf_source=''

try_conda() {
    local candidate=$1 base
    [[ -x "$candidate" ]] || return 1
    show_command "$candidate" --version
    timeout --kill-after=2s 20s "$candidate" --version || return 1
    base=$(timeout --kill-after=2s 20s "$candidate" info --base) || return 1
    if [[ ! -d "$base" ]]; then
        printf 'UNUSABLE: conda did not return an existing base directory.\n'
        return 1
    fi
    conda_cmd=$candidate
    conda_base=$base
    return 0
}

choose_conda() {
    local existing candidate installer checksum expected actual download_dir
    existing=$(type -P conda || true)
    section 'Choose conda without changing shell startup files'
    if init_modules; then
        if module load conda/miniforge/24.3.0; then
            optional module list
            candidate=$(type -P conda || true)
            if [[ -n "$candidate" ]] && try_conda "$candidate"; then
                printf 'Using the E0-observed conda/miniforge/24.3.0 module.\n'
                return 0
            fi
        else
            printf 'The E0-observed Miniforge module did not load; checking existing conda.\n'
        fi
    fi
    for candidate in "$existing" "${CONDA_EXE:-}" "$runtime_root/miniforge/bin/conda"; do
        if [[ -n "$candidate" ]] && try_conda "$candidate"; then
            printf 'Using existing conda: %s\n' "$candidate"
            return 0
        fi
    done

    if [[ -n "$adopt_from" ]]; then
        printf 'ERROR: adoption needs a usable existing conda; it never installs Miniforge or packages.\n' >&2
        return 1
    fi

    section 'No usable conda found; install project-local Miniforge'
    if [[ -e "$runtime_root/miniforge" || -L "$runtime_root/miniforge" ]]; then
        printf 'ERROR: existing Miniforge path is unusable; inspect it before retrying. No files removed.\n' >&2
        return 1
    fi
    require_tool curl
    download_dir=$(mktemp -d "$runtime_root/miniforge-download_XXXXXX")
    installer="$download_dir/Miniforge3-Linux-x86_64.sh"
    checksum="$installer.sha256"
    local release_url=https://github.com/conda-forge/miniforge/releases/latest/download
    required curl --fail --location --retry 2 --connect-timeout 15 --max-time 600 \
        --output "$installer" "$release_url/Miniforge3-Linux-x86_64.sh"
    required curl --fail --location --retry 2 --connect-timeout 15 --max-time 60 \
        --output "$checksum" "$release_url/Miniforge3-Linux-x86_64.sh.sha256"
    expected=$(awk 'NR == 1 {print $1}' "$checksum")
    actual=$(sha256sum "$installer")
    actual=${actual%% *}
    if [[ ! "$expected" =~ ^[[:xdigit:]]{64}$ || "$actual" != "${expected,,}" ]]; then
        printf 'ERROR: official Miniforge checksum did not match; installer was not executed.\n' >&2
        return 1
    fi
    printf '%s  %s\n' "$actual" "$installer" | tee "$output_dir/miniforge-sha256.txt"
    cp -- "$checksum" "$output_dir/miniforge-official.sha256"
    # Batch mode does not run conda init or edit shell startup files.
    required bash "$installer" -b -p "$runtime_root/miniforge"
    if ! try_conda "$runtime_root/miniforge/bin/conda"; then
        printf 'ERROR: project-local Miniforge installation did not provide usable conda.\n' >&2
        return 1
    fi
}

check_existing_prefix() {
    if [[ ! -O "$env_prefix" || ! -f "$env_prefix/conda-meta/history" \
        || ! -f "$env_prefix/.baselines-owner.json" ]]; then
        printf 'ERROR: existing prefix is not an owned, recognized baseline environment: %s\n' "$env_prefix" >&2
        printf 'For a completed E1 install with a missing marker, use --adopt-existing PRIOR_E1_DIRECTORY. Otherwise inspect the prefix; it was not modified.\n' >&2
        return 1
    fi
    pyscf_source=$(bootstrap - "$env_prefix" "$repo_root" "$spec_file" <<'PY'
import hashlib, json, os, pathlib, sys
prefix, repo, spec = map(pathlib.Path, sys.argv[1:])
record = json.loads((prefix / '.baselines-owner.json').read_text())
expected = {'schema': 1, 'uid': os.getuid(), 'prefix': str(prefix), 'repo': str(repo),
            'spec_sha256': hashlib.sha256(spec.read_bytes()).hexdigest(), 'status': 'ready'}
for key, value in expected.items():
    if record.get(key) != value:
        raise SystemExit(f'ERROR: existing environment {key} differs or is incomplete; no update performed.')
if record.get('pyscf_source') not in ('conda-forge', 'pip'):
    raise SystemExit('ERROR: unrecognized PySCF provenance.')
print(record['pyscf_source'])
PY
    )
    printf 'Reusing matching owned environment without installing or updating packages.\n'
    cp -- "$env_prefix/.baselines-owner.json" "$output_dir/environment-owner.json"
}

prepare_adoption() {
    section 'Explicitly validate an existing E1 environment without reinstalling'
    if [[ ! -O "$env_prefix" || ! -f "$env_prefix/conda-meta/history" \
        || ! -x "$env_prefix/bin/python" || ! -x "$env_prefix/bin/mpiexec" ]]; then
        printf 'ERROR: adoption requires an owned conda prefix with Python, MPI and history.\n' >&2
        return 1
    fi
    if [[ -e "$env_prefix/.baselines-owner.json" || -L "$env_prefix/.baselines-owner.json" ]]; then
        # An explicit adoption request cannot bypass a conflicting existing marker.
        check_existing_prefix
    else
        # This recovery path covers the conda-forge install observed in E1.
        # validate_adoption below verifies provenance before any ready marker.
        pyscf_source=conda-forge
    fi
    printf 'Prior E1 evidence: %s\nNo package search, creation, installation or update will run.\n' "$adopt_from"
}

validate_adoption() {
    bootstrap - "$env_prefix" "$repo_root" "$spec_file" "$adopt_from" "$output_dir" <<'PY'
import hashlib, json, pathlib, sys
prefix, repo, spec, prior, output = map(pathlib.Path, sys.argv[1:])
def require(condition, message):
    if not condition:
        raise SystemExit('ERROR: adoption refused: ' + message)
require(prior.parent == (repo / 'results/euler').resolve() and prior.name.startswith('E1_login_'),
        'evidence must be an E1_login_* directory in this repository/results/euler')
for filename in ('environment-requested.yml', 'environment-effective.yml'):
    require((prior / filename).read_bytes() == spec.read_bytes(), filename + ' differs from the current specification')
old_versions = json.loads((prior / 'versions.json').read_text())
new_versions = json.loads((output / 'versions.json').read_text())
for versions in (old_versions, new_versions):
    require(pathlib.Path(versions['prefix']).resolve() == prefix, 'environment prefix differs')
    require(versions['pyscf_source'] == 'conda-forge', 'this recovery requires conda-forge PySCF')
require(old_versions['versions'] == new_versions['versions'], 'installed distribution versions changed')
def inventory(path):
    rows = json.loads(path.read_text())
    records = {row['name']: (row['version'], row.get('build_string', row.get('build')),
                            row['channel'], row['platform']) for row in rows}
    require(len(records) == len(rows), 'duplicate package names in inventory')
    require(records.get('pyscf', (None, None, None))[2] == 'conda-forge', 'PySCF is not a conda-forge package')
    return records
require(inventory(prior / 'conda-packages.json') == inventory(output / 'conda-packages.json'),
        'package versions, builds, channels or platforms differ from the prior E1 inventory')
def frozen(path):
    return sorted(line.strip() for line in path.read_text().splitlines() if line.strip())
require(frozen(prior / 'pip-freeze.txt') == frozen(output / 'pip-freeze.txt'), 'pip distribution inventory changed')
evidence = ('environment-requested.yml', 'environment-effective.yml', 'versions.json',
            'conda-packages.json', 'pip-freeze.txt')
record = {'adopted_from': str(prior), 'prefix': str(prefix), 'validation': 'passed',
          'prior_artifact_sha256': {name: hashlib.sha256((prior / name).read_bytes()).hexdigest() for name in evidence},
          'pip_check': 'passed before adoption', 'native_runtime_validation': 'pending E2'}
(output / 'adoption.json').write_text(json.dumps(record, indent=2) + '\n')
print('Adoption evidence and current metadata match; pip check passed. Native runtime checks remain pending E2.')
PY
}

create_prefix() {
    local search_rc=0
    section 'Check current conda-forge PySCF availability'
    show_command "$conda_cmd" search --override-channels --channel conda-forge --json pyscf
    "$conda_cmd" search --override-channels --channel conda-forge --json pyscf \
        > "$output_dir/pyscf-conda-search.json" 2> "$output_dir/pyscf-conda-search.stderr" || search_rc=$?
    cat "$output_dir/pyscf-conda-search.stderr"
    pyscf_source=$(bootstrap - "$output_dir/pyscf-conda-search.json" "$search_rc" <<'PY'
import json, pathlib, sys
data = json.loads(pathlib.Path(sys.argv[1]).read_text())
if sys.argv[2] == '0' and data.get('pyscf'):
    print('conda-forge')
elif data.get('exception_name') == 'PackagesNotFoundError':
    print('pip')
else:
    raise SystemExit('ERROR: package search failed or was inconclusive; this is not evidence that PySCF is unavailable. See retained JSON/stderr.')
PY
    )
    if [[ "$pyscf_source" == pip ]]; then
        printf 'Conda reports PackagesNotFoundError for PySCF; use a recorded PyPI wheel fallback.\n'
        printf 'Reason: PackagesNotFoundError from conda-forge search; see pyscf-conda-search.json.\n' \
            > "$output_dir/pyscf-fallback-reason.txt"
        sed '/^  - pyscf$/d' "$spec_file" > "$output_dir/environment-effective.yml"
    else
        printf 'PySCF is present on conda-forge; keep it in the shared environment solve.\n'
        cp -- "$spec_file" "$output_dir/environment-effective.yml"
    fi

    section 'Create only the requested new environment'
    mkdir -p -- "$(dirname -- "$env_prefix")"
    required "$conda_cmd" env create --yes --no-default-packages \
        --prefix "$env_prefix" --file "$output_dir/environment-effective.yml"
    if [[ "$pyscf_source" == pip ]]; then
        mkdir -- "$output_dir/wheels"
        required environment_python -m pip --isolated download --index-url https://pypi.org/simple \
            --only-binary=:all: --no-deps --dest "$output_dir/wheels" pyscf
        local -a wheels=("$output_dir"/wheels/pyscf-*.whl)
        if [[ ${#wheels[@]} != 1 || ! -f "${wheels[0]}" ]]; then
            printf 'ERROR: expected exactly one downloaded PySCF wheel.\n' >&2
            return 1
        fi
        sha256sum "${wheels[0]}" | tee "$output_dir/pyscf-wheel-sha256.txt"
        required environment_python -m pip --isolated install --no-index --no-deps "${wheels[0]}"
    fi
}

record_environment() {
    section 'Actual installed versions and exact environment exports'
    # Keep reuse/adoption logs usable as complete evidence for later recovery.
    if [[ "$pyscf_source" == conda-forge ]]; then
        cp -- "$spec_file" "$output_dir/environment-effective.yml"
    else
        sed '/^  - pyscf$/d' "$spec_file" > "$output_dir/environment-effective.yml"
    fi
    required "$conda_cmd" list --prefix "$env_prefix" --show-channel-urls
    "$conda_cmd" list --prefix "$env_prefix" --json > "$output_dir/conda-packages.json"
    "$conda_cmd" list --prefix "$env_prefix" --explicit --md5 > "$output_dir/conda-explicit.txt"
    "$conda_cmd" env export --prefix "$env_prefix" > "$output_dir/environment-exact.yml"
    "$conda_cmd" env export --prefix "$env_prefix" --from-history > "$output_dir/environment-history.yml"
    environment_python -m pip freeze --all > "$output_dir/pip-freeze.txt"
    required environment_python -m pip check
    environment_python - "$env_prefix" "$output_dir" "$pyscf_source" <<'PY'
import importlib.metadata as md
import json, pathlib, platform, sys
prefix, output = map(pathlib.Path, sys.argv[1:3])
records = {r['name']: r for r in json.loads((output / 'conda-packages.json').read_text())}
needed = ('python', 'numpy', 'scipy', 'h5py', 'threadpoolctl', 'petsc', 'petsc4py',
          'slepc', 'slepc4py', 'mpi', 'mpich', 'mpi4py', 'libblas', 'libcblas', 'liblapack')
for name in needed:
    if name not in records:
        raise SystemExit(f'ERROR: missing required conda package {name}')
def build(name):
    return records[name].get('build_string', records[name].get('build', ''))
for name in ('petsc', 'slepc'):
    if not build(name).startswith('real_'):
        raise SystemExit(f'ERROR: {name} is not a CPU real-scalar build.')
for name in ('libblas', 'libcblas', 'liblapack'):
    if 'openblas' not in build(name):
        raise SystemExit(f'ERROR: {name} did not resolve to OpenBLAS.')
if build('mpi') != 'mpich' or not (prefix / 'bin/mpiexec').is_file():
    raise SystemExit('ERROR: expected the actual MPICH launcher and MPI variant.')
if sys.version_info[:2] != (3, 12) or pathlib.Path(sys.prefix).resolve() != prefix.resolve():
    raise SystemExit('ERROR: unexpected Python version or environment prefix.')
names = ('numpy', 'scipy', 'h5py', 'threadpoolctl', 'pyscf', 'petsc4py', 'slepc4py', 'mpi4py')
versions = {'python': platform.python_version(), **{name: md.version(name) for name in names}}
summary = {'prefix': str(prefix), 'pyscf_source': sys.argv[3], 'versions': versions,
           'native_imports_blas_linkage_and_float64_validation': 'pending E2'}
(output / 'versions.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
PY
}

main() {
    section 'E1 login-node environment setup'
    printf 'UTC: %s\nHost: %s\nPrefix: %s\nLogs: %s\n' \
        "$(date -u +%FT%TZ)" "$(hostname)" "$env_prefix" "$output_dir"
    printf 'No shell startup files, system packages, or unrelated environments will be changed.\n'
    probe_storage
    printf 'E0 found no usable quota command. Filesystem capacity does not establish personal quota.\n'
    printf 'No scratch installation location was established by E0; the default is project-local.\n'
    mkdir -p -- "$runtime_root/pkgs"
    bounded_optional 15 df -h -- "$runtime_root"
    cp -- "$spec_file" "$output_dir/environment-requested.yml"

    # Process-local configuration only; do not run conda config --set or conda init.
    cat > "$output_dir/condarc" <<'CONFIG'
channels:
  - conda-forge
channel_priority: strict
auto_update_conda: false
show_channel_urls: true
CONFIG
    export CONDARC="$output_dir/condarc"
    export CONDA_PKGS_DIRS="$runtime_root/pkgs"
    export CONDA_CHANNEL_PRIORITY=strict
    export CONDA_AUTO_UPDATE_CONDA=false
    export PYTHONNOUSERSITE=1
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
    choose_conda
    printf 'Conda executable: %s\nConda base: %s\nPackage cache: %s\n' \
        "$conda_cmd" "$conda_base" "$CONDA_PKGS_DIRS"
    if [[ "$env_prefix" == "$(realpath -m -- "$conda_base")" ]]; then
        printf 'ERROR: the target cannot be the conda base environment.\n' >&2
        return 1
    fi
    if [[ -n "$adopt_from" ]]; then
        prepare_adoption
    elif [[ -e "$env_prefix" || -L "$env_prefix" ]]; then
        check_existing_prefix
    else
        create_prefix
    fi
    record_environment
    if [[ -n "$adopt_from" ]]; then
        validate_adoption
    fi
    bootstrap - "$env_prefix" "$repo_root" "$spec_file" "$pyscf_source" "$output_dir" "$adopt_from" <<'PY'
import hashlib, json, os, pathlib, sys
prefix, repo, spec = map(pathlib.Path, sys.argv[1:4])
record = {'schema': 1, 'uid': os.getuid(), 'prefix': str(prefix), 'repo': str(repo),
          'spec_sha256': hashlib.sha256(spec.read_bytes()).hexdigest(), 'status': 'ready',
          'pyscf_source': sys.argv[4], 'e1_artifacts': sys.argv[5]}
if sys.argv[6]:
    record['adopted_from'] = sys.argv[6]
text = json.dumps(record, indent=2) + '\n'
marker = prefix / '.baselines-owner.json'
if not marker.exists():
    with marker.open('x') as stream:
        stream.write(text)
(pathlib.Path(sys.argv[5]) / 'environment-owner.json').write_text(marker.read_text())
PY
    local prefix_file
    prefix_file=$(mktemp "$runtime_root/.baselines-prefix_XXXXXX")
    printf '%s\n' "$env_prefix" > "$prefix_file"
    mv -f -- "$prefix_file" "$runtime_root/baselines-prefix.txt"
    printf '\nE1 complete. Prefix pointer: %s/baselines-prefix.txt\n' "$runtime_root"
    printf 'Output: %s\nNext: submit euler/02_check_env.sbatch; native runtime checks remain pending E2.\n' "$output_dir"
}

# Preserve failures and logs. Failed/incomplete prefixes are never deleted or updated.
main 2>&1 | tee "$output_dir/env.txt"
