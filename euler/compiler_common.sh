#!/usr/bin/env bash
# Shared E2/E3 and future E4/E5 compiler policy. Call inside a build subshell.
set -euo pipefail

prepare_system_compiler() {
    local build_dir=$1
    # Do not initialize, load, unload or purge modules: even purge can evaluate
    # the incompatible historical gcc module. Remove implicit search overrides.
    unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH \
        COMPILER_PATH GCC_EXEC_PREFIX LD_LIBRARY_PATH LD_PRELOAD \
        CPPFLAGS CFLAGS CXXFLAGS LDFLAGS
    export PATH=/usr/bin:/bin CC=/usr/bin/gcc CXX=/usr/bin/g++
    if [[ ! -x /usr/bin/g++ ]]; then
        printf 'ERROR: required system compiler /usr/bin/g++ is unavailable.\n' >&2
        return 1
    fi
    mkdir -p -- "$build_dir" || return 1
    {
        printf 'Compiler policy: system /usr/bin/g++; no compiler module.\n'
        printf 'Reason: E3 129413 exposed incompatible gcc/14.3.0 C++ headers.\n'
        printf 'Flags: -std=c++17 -O3 -march=native -fopenmp -Wall -Wextra -pedantic\n'
        printf 'Resolved compiler: '
        readlink -f /usr/bin/g++ || return 1
        /usr/bin/g++ --version || return 1
        printf 'Target: '
        /usr/bin/g++ -dumpmachine || return 1
        printf 'libstdc++: '
        /usr/bin/g++ -print-file-name=libstdc++.so || return 1
    } > "$build_dir/compiler.txt" || return 1
    cat "$build_dir/compiler.txt" || return 1
    # E2's original dump wrapper does not include filesystem, so explicitly
    # check the C++17 headers and OpenMP runtime needed by the E3 wrapper.
    cat > "$build_dir/compiler_probe.cpp" <<'CPP' || return 1
#include <filesystem>
#include <omp.h>
int main() {
    return std::filesystem::path(".").empty() || omp_get_max_threads() < 1;
}
CPP
    show_command /usr/bin/g++ -std=c++17 -O3 -march=native -fopenmp \
        -Wall -Wextra -pedantic "$build_dir/compiler_probe.cpp" -o "$build_dir/compiler_probe"
    /usr/bin/g++ -std=c++17 -O3 -march=native -fopenmp -Wall -Wextra -pedantic \
        "$build_dir/compiler_probe.cpp" -o "$build_dir/compiler_probe" || return 1
    OMP_NUM_THREADS=1 "$build_dir/compiler_probe" || return 1
    printf 'System compiler filesystem/OpenMP check: PASS\n'
}
