"""Build and run the Davidson end-to-end checks.

For each n in CASES the fixture A (diagonal 1..n, both adjacent diagonals 0.2)
is built here from the spec, and NumPy's eigvalsh(A)[:2] is passed to the C++
check as the independent reference. Every case runs on serial, OpenMP (with
up to 8 threads), and CUDA; the dependent-vector drop criterion is checked
last. Exits 0 only if every run passes.

Usage (needs nvcc, g++ with OpenMP, and NumPy; works from any directory):
    python3 checks/run_davidson_e2e.py              # -O2 nvcc build (serial + openmp + cuda)
    python3 checks/run_davidson_e2e.py --sanitize   # ASan + UBSan g++ build (CPU backends only)
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
BACKENDS = ("serial", "openmp", "cuda")
OMP_THREADS = min(8, len(os.sched_getaffinity(0)))


def reference(n):
    A = np.diag(np.arange(1.0, n + 1)) + 0.2 * (np.eye(n, k=1) + np.eye(n, k=-1))
    return [repr(float(x)) for x in np.linalg.eigvalsh(A)[:2]]


parser = argparse.ArgumentParser(description="Build and run the Davidson E2E checks.")
parser.add_argument("--sanitize", action="store_true",
                    help="build CPU backends with AddressSanitizer + UBSan; leaks and UB fail the run")
args = parser.parse_args()

env = {**os.environ, "OMP_NUM_THREADS": str(OMP_THREADS)}
if args.sanitize:
    env.setdefault("ASAN_OPTIONS", "detect_leaks=1")

with tempfile.TemporaryDirectory() as tmp:
    exe = pathlib.Path(tmp) / "davidson_e2e"
    if args.sanitize:
        backends = ("serial", "openmp")
        flags = ["-O1", "-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=undefined"]
        build = ["g++", "-std=c++17", *flags, "-fopenmp", "-Wall", "-Wextra", "-pedantic",
                 f"-I{ROOT}", str(ROOT / "matrix.cpp"), str(ROOT / "davidson.cpp"),
                 str(ROOT / "checks" / "davidson_e2e.cpp"), "-o", str(exe)]
    else:
        backends = BACKENDS
        build = ["nvcc", "-std=c++17", "-O2", "-arch=native", "-DUSE_CUDA",
                 "-Xcompiler", "-fopenmp,-Wall,-Wextra",
                 f"-I{ROOT}", str(ROOT / "matrix.cpp"), str(ROOT / "davidson.cpp"),
                 str(ROOT / "davidson.cu"), str(ROOT / "checks" / "davidson_e2e.cpp"),
                 "-o", str(exe)]

    print("build:", " ".join(build))
    print(f"OpenMP backend threads: {OMP_THREADS}; BLAS threads: 1", flush=True)
    if subprocess.run(build).returncode != 0:
        sys.exit("FAIL: build error")

    runs = [(f"{backend} n={n}", [backend, str(n), *reference(n)])
            for backend in backends for n in CASES]
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