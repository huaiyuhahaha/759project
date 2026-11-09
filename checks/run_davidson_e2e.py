"""Build and run the serial Davidson end-to-end checks.

For each n in CASES the fixture A (diagonal 1..n, both adjacent diagonals 0.2)
is built here from the spec, and NumPy's eigvalsh(A)[:2] is passed to the C++
check as the independent reference. Every case runs on both backends (serial,
and OpenMP with up to 8 threads); the dependent-vector drop criterion is
checked last. Exits 0 only if every run passes.

Usage (needs g++ with OpenMP and NumPy; works from any directory):
    python3 checks/run_davidson_e2e.py              # -O2 build
    python3 checks/run_davidson_e2e.py --sanitize   # ASan + UBSan build, leak check on
"""

import argparse
import os
import pathlib
import subprocess
import sys
import tempfile

# Pin BLAS threads before NumPy loads its BLAS.
for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[var] = "1"

import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CASES = (8, 64)  # must match kCases in checks/davidson_e2e.cpp
BACKENDS = ("serial", "openmp")
OMP_THREADS = min(8, len(os.sched_getaffinity(0)))


def reference(n):
    A = np.diag(np.arange(1.0, n + 1)) + 0.2 * (np.eye(n, k=1) + np.eye(n, k=-1))
    return [repr(float(x)) for x in np.linalg.eigvalsh(A)[:2]]


parser = argparse.ArgumentParser(description="Build and run the Davidson E2E checks.")
parser.add_argument("--sanitize", action="store_true",
                    help="build with AddressSanitizer + UBSan; leaks and UB fail the run")
args = parser.parse_args()

if args.sanitize:
    flags = ["-O1", "-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=undefined"]
else:
    flags = ["-O2"]
env = {**os.environ, "OMP_NUM_THREADS": str(OMP_THREADS)}
if args.sanitize:
    env.setdefault("ASAN_OPTIONS", "detect_leaks=1")

with tempfile.TemporaryDirectory() as tmp:
    exe = pathlib.Path(tmp) / "davidson_e2e"
    build = ["g++", "-std=c++17", *flags, "-fopenmp", "-Wall", "-Wextra", "-pedantic",
             f"-I{ROOT}", str(ROOT / "matrix.cpp"), str(ROOT / "davidson.cpp"),
             str(ROOT / "checks" / "davidson_e2e.cpp"), "-o", str(exe)]
    print("build:", " ".join(build))
    print(f"OpenMP backend threads: {OMP_THREADS}; BLAS threads: 1", flush=True)
    if subprocess.run(build).returncode != 0:
        sys.exit("FAIL: build error")

    runs = [(f"{backend} n={n}", [backend, str(n), *reference(n)])
            for backend in BACKENDS for n in CASES]
    runs.append(("lindep criterion", ["lindep"]))
    codes = []
    for name, run_args in runs:
        print(f"\n== {name}: {' '.join(run_args)}", flush=True)
        rc = subprocess.run([str(exe), *run_args], env=env).returncode
        print(f"== {name}: exit code {rc}", flush=True)
        codes.append(rc)

ok = all(rc == 0 for rc in codes)
print("\nOVERALL:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
