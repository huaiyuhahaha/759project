"""Build and run the gemm correctness checks (checks/gemm_check.cpp).

The micro-kernel is chosen at compile time, so the checks are built once per
target and every build is run:
  - native:  -march=native (AVX-512 kernel on AVX-512 CPUs, else AVX2+FMA);
  - avx2:    -mavx2 -mfma (skipped if this CPU lacks AVX2/FMA);
  - generic: no -march flag, i.e. baseline x86-64, which uses the scalar kernel;
  - avx512-shape: the AVX-512 tile shape (12x16) and blocking built from GCC
    generic vectors (-DGEMM_TEST_AVX512_SHAPE), runnable without AVX-512.
    It covers everything in the AVX-512 build except the six intrinsic
    wrappers, which only the native build on an AVX-512 CPU runs.
On an AVX-512 CPU the native build also checks Huaiyu's
cache_register_multiplication() from matrix_optimize_4.cpp.

OMP_NUM_THREADS is left as set by the caller (default: all CPUs available).
Exits 0 only if every build compiles and passes.

Usage (needs g++ with OpenMP; works from any directory):
    python3 checks/run_gemm_check.py              # -O3 builds
    python3 checks/run_gemm_check.py --sanitize   # ASan + UBSan builds, leak check on
"""

import argparse
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGETS = {"native": ["-march=native"], "avx2": ["-mavx2", "-mfma"], "generic": [],
           # AVX-512 tile shape and blocking on portable vectors (see gemm.cpp), so the
           # AVX-512 configuration is exercised on CPUs without AVX-512 too.
           "avx512-shape": ["-march=native", "-DGEMM_TEST_AVX512_SHAPE", "-Wno-psabi"]}


def cpu_flags():
    try:
        text = pathlib.Path("/proc/cpuinfo").read_text()
    except OSError:
        return set()
    for line in text.splitlines():
        if line.startswith("flags"):
            return set(line.split(":", 1)[1].split())
    return set()


def defines(target_flags):
    """Preprocessor macros the compiler defines for these target flags."""
    out = subprocess.run(["g++", *target_flags, "-dM", "-E", "-x", "c++", "/dev/null"],
                         capture_output=True, text=True, check=True).stdout
    return {line.split()[1] for line in out.splitlines() if line.startswith("#define")}


parser = argparse.ArgumentParser(description="Build and run the gemm correctness checks.")
parser.add_argument("--sanitize", action="store_true",
                    help="build with AddressSanitizer + UBSan; leaks and UB fail the run")
args = parser.parse_args()

opt = (["-O1", "-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=undefined"]
       if args.sanitize else ["-O3"])
env = dict(os.environ)
if args.sanitize:
    env.setdefault("ASAN_OPTIONS", "detect_leaks=1")

flags = cpu_flags()
results = {}
with tempfile.TemporaryDirectory() as tmp:
    for name, target in TARGETS.items():
        if name == "avx2" and not {"avx2", "fma"} <= flags:
            print(f"== {name}: skipped (CPU lacks AVX2/FMA)")
            continue
        sources = [ROOT / "gemm.cpp", ROOT / "matrix.cpp", ROOT / "checks" / "gemm_check.cpp"]
        extra = []
        if "__AVX512F__" in defines(target):
            sources.append(ROOT / "matrix_optimize_4.cpp")
            extra.append("-DCHECK_HUAIYU")
        exe = pathlib.Path(tmp) / f"gemm_check_{name}"
        build = ["g++", "-std=c++17", *opt, *target, *extra, "-fopenmp",
                 "-Wall", "-Wextra", "-pedantic", f"-I{ROOT}", *map(str, sources),
                 "-o", str(exe)]
        print(f"\n== {name}: build: {' '.join(build)}", flush=True)
        if subprocess.run(build).returncode != 0:
            results[name] = "build error"
            continue
        rc = subprocess.run([str(exe)], env=env).returncode
        results[name] = "PASS" if rc == 0 else f"FAIL (exit {rc})"

print("\nSummary:")
for name, status in results.items():
    print(f"  {name:12} {status}")
ok = bool(results) and all(s == "PASS" for s in results.values())
print("OVERALL:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
