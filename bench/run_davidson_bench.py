"""Benchmark the Davidson solver: serial baseline vs the OpenMP backend.

Builds bench/davidson_bench.cpp once (both backends live in the same binary, so
compiler flags are identical), then for each n runs the serial backend and the
OpenMP backend at 1, 2, 4, 8 threads (capped at the CPUs available). Each run
is its own process: same seeded matrix, guess and tolerance, one warm-up solve
and five timed solves. BLAS thread counts are pinned to 1 (the solver itself
does not call BLAS). Exits nonzero if any run fails its checks or if an OpenMP
run disagrees with the serial run on iterations or eigenvalues.

Usage (needs g++ with OpenMP; works from any directory):
    python3 bench/run_davidson_bench.py [--sizes 512 1024 2048] [--threads 1 2 4 8]
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
FLAGS = ["-std=c++17", "-O3", "-march=native", "-fopenmp", "-Wall", "-Wextra", "-pedantic"]
KERNELS = ("matvec", "orthogonalize", "project", "eigensolve", "ritz", "precondition")
EIG_MATCH_TOL = 1e-12  # max |eigenvalue(OpenMP) - eigenvalue(serial)|

parser = argparse.ArgumentParser(description="Benchmark serial vs OpenMP Davidson.")
parser.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048])
parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
args = parser.parse_args()

available = len(os.sched_getaffinity(0))
threads = [t for t in args.threads if t <= available]

env = dict(os.environ)
for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    env[var] = "1"
env["OMP_PLACES"] = "cores"     # one thread per physical core
env["OMP_PROC_BIND"] = "close"


def run(exe, n, backend, nthreads):
    proc = subprocess.run([str(exe), str(n), backend], capture_output=True, text=True,
                          env={**env, "OMP_NUM_THREADS": str(nthreads)})
    sys.stderr.write(proc.stderr)
    if not proc.stdout.strip():
        sys.exit(f"FAIL: n={n} {backend} produced no result (exit {proc.returncode})")
    res = json.loads(proc.stdout.strip().splitlines()[-1])
    res["exit_code"] = proc.returncode
    return res


failures = []
results = []
with tempfile.TemporaryDirectory() as tmp:
    exe = pathlib.Path(tmp) / "davidson_bench"
    build = ["g++", *FLAGS, f"-I{ROOT}", str(ROOT / "matrix.cpp"), str(ROOT / "davidson.cpp"),
             str(ROOT / "bench" / "davidson_bench.cpp"), "-o", str(exe)]
    print("build:", " ".join(build))
    print(f"CPUs available: {available}; OpenMP threads: {threads}; "
          "OMP_PLACES=cores OMP_PROC_BIND=close; BLAS threads = 1", flush=True)
    if subprocess.run(build).returncode != 0:
        sys.exit("FAIL: build error")

    for n in args.sizes:
        serial = run(exe, n, "serial", 1)
        results.append(serial)
        for t in threads:
            results.append(run(exe, n, "openmp", t))
        print(f"  n={n} done", flush=True)

first = results[0]
print(f"\nk={first['k']}  tol={first['tol']}  seed={first['seed']}  "
      f"warm-up={first['warmup']}  timed runs={first['runs']}  flags: {' '.join(FLAGS)}\n")
header = (f"{'n':>5} {'backend':>8} {'thr':>3} {'median ms':>10} {'speedup':>8} {'iters':>5} "
          f"{'subspace':>8} {'max resid (indep)':>17} {'max|dEig| vs serial':>19} {'exit':>4}")
print(header)
print("-" * len(header))
serial_by_n = {}
for r in results:
    if r["backend"] == "serial":
        serial_by_n[r["n"]] = r
    s = serial_by_n[r["n"]]
    speedup = s["median"] / r["median"]
    deig = max(abs(a - b) for a, b in zip(r["eigenvalues"], s["eigenvalues"]))
    print(f"{r['n']:>5} {r['backend']:>8} {r['threads']:>3} {1e3 * r['median']:>10.2f} "
          f"{speedup:>8.2f} {r['iterations']:>5} {r['subspace']:>8} {r['max_residual']:>17.3e} "
          f"{deig:>19.3e} {r['exit_code']:>4}")
    label = f"n={r['n']} {r['backend']} threads={r['threads']}"
    if r["exit_code"] != 0 or not r["ok"]:
        failures.append(f"{label}: run checks failed")
    if r["iterations"] != s["iterations"] or deig > EIG_MATCH_TOL:
        failures.append(f"{label}: differs from serial (iterations or eigenvalues)")


def print_profile(title, rows):
    print(f"\n{title}")
    print(f"{'n':>5} {'thr':>3} {'solve ms':>9} " + " ".join(f"{k:>14}" for k in KERNELS)
          + f" {'other':>14}")
    for r in rows:
        total = r["median"]
        cells = [r["profile"][k] for k in KERNELS]
        other = total - sum(cells)
        print(f"{r['n']:>5} {r['threads']:>3} {1e3 * total:>9.2f} "
              + " ".join(f"{1e3 * c:>7.2f} ({100 * c / total:>3.0f}%)" for c in cells)
              + f" {1e3 * other:>7.2f} ({100 * other / total:>3.0f}%)")


print_profile("Serial kernel profile, ms (share of median solve), median of timed runs:",
              [r for r in results if r["backend"] == "serial"])
if threads:
    top = max(threads)
    print_profile(f"OpenMP kernel profile at {top} threads:",
                  [r for r in results if r["backend"] == "openmp" and r["threads"] == top])

print()
for f in failures:
    print("FAIL:", f)
print("OVERALL:", "FAIL" if failures else "PASS")
sys.exit(1 if failures else 0)
