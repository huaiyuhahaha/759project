"""Benchmark the Davidson solver: serial baseline, OpenMP backend, and PySCF.

Builds bench/davidson_bench.cpp once; both C++ backends live in the same
binary, so compiler flags are identical. For each n it first writes the inputs
with `davidson_bench <n> dump`: the seeded matrix and the initialize_vector(4, A)
guess as raw float64. It computes NumPy eigvalsh(A)[:4] from that file as the
reference. It then runs on those inputs:
  - the C++ serial backend;
  - the C++ OpenMP backend at 1, 2, 4, 8 threads (capped at the CPUs available);
  - PySCF's davidson1 at 1 and 8 threads, via bench/pyscf_davidson_bench.py.
Each run is its own process: one warm-up solve and five timed solves of the
same inputs, timing only the solve. Every run reports a checksum of the input
bytes it used, and every checksum must equal the dump's.

Threads: every run gets OMP_PLACES=cores and OMP_PROC_BIND=close. C++ runs pin
BLAS threads to 1 (the C++ solver does not call BLAS). PySCF runs set
OMP_NUM_THREADS and the BLAS thread counts to the run's thread count, since
their matvec is a NumPy/OpenBLAS dgemv. C++ matvecs equal the final subspace
size because the C++ loop never restarts.

Exits nonzero if any of these fails:
  - a run fails its own checks (for PySCF this includes all four convergence flags);
  - an input checksum differs from the dump's;
  - an eigenvalue differs from eigvalsh by 1e-8 or more;
  - a C++ OpenMP run disagrees with C++ serial on iterations or eigenvalues.

Usage (needs g++ with OpenMP, NumPy and PySCF; works from any directory):
    python3 bench/run_davidson_bench.py [--sizes 512 1024 2048] [--threads 1 2 4 8]
                                        [--pyscf-threads 1 8]
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile

BLAS_VARS = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS")
# This driver uses NumPy only for the eigvalsh reference; pin its BLAS first.
for var in BLAS_VARS:
    os.environ[var] = "1"

import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
FLAGS = ["-std=c++17", "-O3", "-march=native", "-fopenmp", "-Wall", "-Wextra", "-pedantic"]
KERNELS = ("matvec", "orthogonalize", "project", "eigensolve", "ritz", "precondition")
NROOTS = 4             # kRoots in davidson_bench.cpp
EIG_MATCH_TOL = 1e-12  # max |eigenvalue(C++ OpenMP) - eigenvalue(C++ serial)|
REF_ACCEPT = 1e-8      # max |eigenvalue - NumPy eigvalsh| for every run

parser = argparse.ArgumentParser(description="Benchmark serial vs OpenMP Davidson, and PySCF.")
parser.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048])
parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
parser.add_argument("--pyscf-threads", type=int, nargs="+", default=[1, 8])
args = parser.parse_args()

available = len(os.sched_getaffinity(0))
threads = [t for t in args.threads if t <= available]
pyscf_threads = [t for t in args.pyscf_threads if t <= available]

env = dict(os.environ)          # BLAS thread counts already 1
env["OMP_PLACES"] = "cores"     # one thread per physical core
env["OMP_PROC_BIND"] = "close"


def parse(proc, label):
    sys.stderr.write(proc.stderr)
    lines = proc.stdout.strip().splitlines()
    if not lines:
        sys.exit(f"FAIL: {label} produced no result (exit {proc.returncode})")
    for line in lines[:-1]:  # e.g. PySCF log warnings
        print(f"  [{label}] {line}", flush=True)
    res = json.loads(lines[-1])
    res["exit_code"] = proc.returncode
    return res


def run_cpp(exe, n, backend, nthreads):
    proc = subprocess.run([str(exe), str(n), backend], capture_output=True, text=True,
                          env={**env, "OMP_NUM_THREADS": str(nthreads)})
    return parse(proc, f"n={n} {backend}")


def run_pyscf(n, inputs, nthreads):
    t = str(nthreads)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "bench" / "pyscf_davidson_bench.py"), str(n), str(inputs)],
        capture_output=True, text=True,
        env={**env, "OMP_NUM_THREADS": t, **{v: t for v in BLAS_VARS}})
    return parse(proc, f"n={n} pyscf threads={t}")


failures = []
results = []
reference = {}
dump_checksum = {}
compiler = subprocess.run(["g++", "--version"], capture_output=True, text=True).stdout
with tempfile.TemporaryDirectory() as tmp:
    exe = pathlib.Path(tmp) / "davidson_bench"
    build = ["g++", *FLAGS, f"-I{ROOT}", str(ROOT / "matrix.cpp"), str(ROOT / "davidson.cpp"),
             str(ROOT / "bench" / "davidson_bench.cpp"), "-o", str(exe)]
    print("build:", " ".join(build))
    print(f"compiler: {compiler.splitlines()[0] if compiler else 'unknown'}")
    print(f"CPUs available: {available}; C++ OpenMP threads: {threads}; "
          f"PySCF threads: {pyscf_threads}; OMP_PLACES=cores OMP_PROC_BIND=close; "
          "BLAS threads = 1 for C++ runs, = run threads for PySCF runs", flush=True)
    if subprocess.run(build).returncode != 0:
        sys.exit("FAIL: build error")

    for n in args.sizes:
        inputs = pathlib.Path(tmp) / f"inputs_{n}.bin"
        dump = parse(subprocess.run([str(exe), str(n), "dump", str(inputs)],
                                    capture_output=True, text=True), f"n={n} dump")
        if dump["exit_code"] != 0:
            sys.exit(f"FAIL: n={n} dump failed")
        dump_checksum[n] = dump["input_checksum"]
        A = np.fromfile(inputs, dtype="<f8", count=n * n).reshape(n, n)
        reference[n] = np.linalg.eigvalsh(A)[:NROOTS]  # outside every timer
        del A

        results.append(run_cpp(exe, n, "serial", 1))
        for t in threads:
            results.append(run_cpp(exe, n, "openmp", t))
        for t in pyscf_threads:
            results.append(run_pyscf(n, inputs, t))
        print(f"  n={n} done", flush=True)

serial_by_n = {r["n"]: r for r in results if r["backend"] == "serial"}
omp_by_nt = {(r["n"], r["threads"]): r for r in results if r["backend"] == "openmp"}
for r in results:
    s = serial_by_n[r["n"]]
    r["speedup"] = s["median"] / r["median"]
    r["ref_err"] = max(abs(a - b) for a, b in zip(r["eigenvalues"], reference[r["n"]]))
    label = f"n={r['n']} {r['backend']} threads={r['threads']}"
    if r["exit_code"] != 0 or not r["ok"]:
        failures.append(f"{label}: run checks failed")
    if r["input_checksum"] != dump_checksum[r["n"]]:
        failures.append(f"{label}: input checksum {r['input_checksum']} != dump "
                        f"{dump_checksum[r['n']]}")
    if len(r["eigenvalues"]) != NROOTS or not r["ref_err"] < REF_ACCEPT:
        failures.append(f"{label}: eigenvalues differ from NumPy eigvalsh by {r['ref_err']:.3e}")
    if r["backend"] == "pyscf" and not (len(r["conv"]) == NROOTS and all(r["conv"])):
        failures.append(f"{label}: convergence flags {r['conv']}")
    if r["backend"] == "openmp":
        r["deig"] = max(abs(a - b) for a, b in zip(r["eigenvalues"], s["eigenvalues"]))
        if r["iterations"] != s["iterations"] or r["deig"] > EIG_MATCH_TOL:
            failures.append(f"{label}: differs from serial (iterations or eigenvalues)")

first = results[0]
print(f"\nk={first['k']}  tol={first['tol']}  seed={first['seed']}  "
      f"warm-up={first['warmup']}  timed runs={first['runs']}  flags: {' '.join(FLAGS)}")
print("input checksums (dump = every run): "
      + "  ".join(f"n={n}: {c}" for n, c in dump_checksum.items()) + "\n")

print("C++ solver")
header = (f"{'n':>5} {'backend':>8} {'thr':>3} {'median ms':>10} {'speedup':>8} {'iters':>5} "
          f"{'subspace':>8} {'max resid (indep)':>17} {'max|dEig| vs serial':>19} "
          f"{'max|eig-eigvalsh|':>17} {'exit':>4}")
print(header)
print("-" * len(header))
for r in results:
    if r["backend"] == "pyscf":
        continue
    print(f"{r['n']:>5} {r['backend']:>8} {r['threads']:>3} {1e3 * r['median']:>10.2f} "
          f"{r['speedup']:>8.2f} {r['iterations']:>5} {r['subspace']:>8} "
          f"{r['max_residual']:>17.3e} {r.get('deig', 0.0):>19.3e} {r['ref_err']:>17.3e} "
          f"{r['exit_code']:>4}")


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

pyscf_rows = [r for r in results if r["backend"] == "pyscf"]
if pyscf_rows:
    print("\nPySCF davidson1 vs C++ at matching thread counts (same input bytes and guess)")
    header = (f"{'n':>5} {'thr':>3} {'solver':>10} {'median ms':>10} {'vs C++ serial':>13} "
              f"{'PySCF/C++ omp':>13} {'iters':>5} {'matvecs':>7} {'conv flags':>10} "
              f"{'max resid (indep)':>17} {'||XtX-I||_F':>11} {'max|eig-eigvalsh|':>17} "
              f"{'exit':>4}")
    print(header)
    print("-" * len(header))
    for p in pyscf_rows:
        n, t = p["n"], p["threads"]
        cpp_rows = ([("C++ serial", serial_by_n[n])] if t == 1 else []) + \
                   ([("C++ OpenMP", omp_by_nt[(n, t)])] if (n, t) in omp_by_nt else [])
        for name, r in cpp_rows:
            print(f"{n:>5} {t:>3} {name:>10} {1e3 * r['median']:>10.2f} {r['speedup']:>13.2f} "
                  f"{'':>13} {r['iterations']:>5} {r['subspace']:>7} "
                  f"{'all' if r['converged'] else 'NOT':>10} {r['max_residual']:>17.3e} "
                  f"{r['orth_error']:>11.3e} {r['ref_err']:>17.3e} {r['exit_code']:>4}")
        omp = omp_by_nt.get((n, t))
        ratio = f"{p['median'] / omp['median']:.2f}" if omp else "-"
        flags = "".join("T" if c else "F" for c in p["conv"])
        print(f"{n:>5} {t:>3} {'PySCF':>10} {1e3 * p['median']:>10.2f} {p['speedup']:>13.2f} "
              f"{ratio:>13} {p['iterations']:>5} {p['matvecs']:>7} {flags:>10} "
              f"{p['max_residual']:>17.3e} {p['orth_error']:>11.3e} {p['ref_err']:>17.3e} "
              f"{p['exit_code']:>4}")

    v, st = pyscf_rows[0]["versions"], pyscf_rows[0]["settings"]
    print("\nPySCF runs: " + ", ".join(f"{k} {val}" for k, val in v.items()))
    print(f"davidson1: nroots={NROOTS}  tol_residual={st['tol_residual']:g}  "
          f"tol (eigenvalue change)={st['eig_change_tol']:g}  max_cycle={st['max_cycle']}  "
          f"max_space={st['max_space_arg']} (effective {st['effective_max_space']})  "
          f"lindep={st['lindep']:g}  precond: C++ guarded diagonal, floor "
          f"{st['denom_floor']:g}, shift e[0] (chosen by davidson1)")
    for p in pyscf_rows:
        tr = p["thread_report"]
        pools = ", ".join(f"{q['library']}={q['num_threads']}" for q in tr.get("threadpools", []))
        print(f"  n={p['n']:>4} thr={p['threads']}: restarts={p['settings']['restarts']}  "
              f"deterministic={p['deterministic']}  pyscf.lib.num_threads={tr['pyscf_lib_num_threads']}"
              f"\n      thread pools: {pools}"
              f"\n      OS threads by allowed CPUs: {tr['thread_cpu_affinity']}")

print()
for f in failures:
    print("FAIL:", f)
print("OVERALL:", "FAIL" if failures else "PASS")
sys.exit(1 if failures else 0)
