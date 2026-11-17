"""Benchmark matrix multiply: naive baseline, blocked gemm() and OpenBLAS dgemm.

Builds bench/gemm_bench.cpp once with -O3 -march=native -fopenmp (plus
OpenBLAS, and on AVX-512 CPUs Huaiyu's matrix_optimize_4.cpp), then times
square C = A * B for each size:
  - naive    the original Matrix::operator* (verbatim copy in bench/original_matrix.cpp),
             1 thread (sizes <= --naive-max);
  - loop     the same loop on raw pointers, auto-vectorized, 1 thread (sizes <= --naive-max);
  - huaiyu   cache_register_multiplication(), 1 thread (AVX-512 builds only);
  - blocked  gemm() at every thread count;
  - openblas cblas_dgemm at every thread count.
Each configuration is its own process: one warm-up, then --runs timed runs;
the table shows the median. Every result is checked against OpenBLAS.

Thread placement: a run with t threads is restricted (sched_setaffinity) to
one logical CPU on each of the first t physical cores, so no two threads can
share a core through SMT/hyper-threading. (OpenBLAS does not pin its threads;
given both SMT siblings of each core, the OS occasionally put two of them on
one core and a whole run came out 20-50% slow.) OpenMP runs also get
OMP_PLACES=cores OMP_PROC_BIND=close. OpenBLAS runs do NOT get OMP_PROC_BIND:
libgomp would pin the main thread to one core at startup and OpenBLAS's
worker threads inherit that mask, putting every BLAS thread on one core.

OpenBLAS picks its kernels at run time from the CPU model (DYNAMIC_ARCH). Old
versions misdetect new CPUs (0.3.20 reports a Raptor Lake i7-14700HX as
"Prescott", an SSE3 core, and runs ~3x slower). The core it used is printed;
override it with --openblas-coretype (e.g. Haswell for AVX2, SkylakeX for AVX-512).

--sweep instead times the blocked gemm over a grid of (mc, kc) block sizes at
--sweep-size, on 1 thread and on the largest thread count.

Usage (works from any directory):
    python3 bench/run_gemm_bench.py [--sizes 256 512 1024 2048 4096] [--threads 1 2 4 8]
        [--naive-max 2048] [--runs 7] [--openblas-coretype Haswell] [--no-openblas]
        [--blas-flags "-lopenblas"] [--thin 1 4 8 16] [--sweep] [--sweep-size 2048]
"""

import argparse
import json
import os
import pathlib
import shlex
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
FLAGS = ["-std=c++17", "-O3", "-march=native", "-fopenmp", "-Wall", "-Wextra", "-pedantic"]

parser = argparse.ArgumentParser(description="Benchmark naive, blocked and OpenBLAS matrix multiply.")
parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
parser.add_argument("--naive-max", type=int, default=2048,
                    help="largest size for the naive and loop variants")
parser.add_argument("--runs", type=int, default=7, help="timed runs per configuration")
parser.add_argument("--naive-runs", type=int, default=3,
                    help="timed runs for the naive and loop variants")
parser.add_argument("--openblas-coretype", default=None, help="sets OPENBLAS_CORETYPE")
parser.add_argument("--no-openblas", action="store_true", help="build and run without OpenBLAS")
parser.add_argument("--blas-flags", default="-lopenblas", help="compiler/linker flags for OpenBLAS")
parser.add_argument("--thin", type=int, nargs="*", default=[],
                    help="also time n x n times n x c (Davidson block matvec) for these c, "
                         "at the largest size and every thread count")
parser.add_argument("--sweep", action="store_true", help="sweep blocking sizes instead")
parser.add_argument("--sweep-size", type=int, default=2048)
args = parser.parse_args()


def physical_cores():
    """Allowed CPUs grouped by physical core, in CPU order."""
    allowed = sorted(os.sched_getaffinity(0))
    groups = {}
    for cpu in allowed:
        path = pathlib.Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        try:
            key = path.read_text().strip()
        except OSError:
            key = str(cpu)
        groups.setdefault(key, []).append(cpu)
    return sorted(groups.values(), key=lambda g: g[0])


CORES = physical_cores()
threads = [t for t in args.threads if 1 <= t <= len(CORES)]
if not threads:
    sys.exit(f"FAIL: no requested thread count fits the {len(CORES)} available cores")
skipped = sorted(set(args.threads) - set(threads))


def defines(flags):
    out = subprocess.run(["g++", *flags, "-dM", "-E", "-x", "c++", "/dev/null"],
                         capture_output=True, text=True, check=True).stdout
    return {line.split()[1] for line in out.splitlines() if line.startswith("#define")}


have_avx512 = "__AVX512F__" in defines(FLAGS)
have_openblas = not args.no_openblas

base_env = {k: v for k, v in os.environ.items()
            if k not in ("OMP_PLACES", "OMP_PROC_BIND", "OMP_NUM_THREADS")}
for var in ("OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    base_env[var] = "1"
if args.openblas_coretype:
    base_env["OPENBLAS_CORETYPE"] = args.openblas_coretype


def run(exe, variant, m, n, k, t, runs, blocking=None):
    cpus = [group[0] for group in CORES[:t]]  # one logical CPU per physical core
    env = dict(base_env, OMP_NUM_THREADS=str(t))
    if variant == "openblas":
        env["OPENBLAS_NUM_THREADS"] = str(t)
    else:
        env.update(OMP_PLACES="cores", OMP_PROC_BIND="close")
    cmd = [str(exe), variant, str(m), str(n), str(k), str(t), str(runs)]
    if blocking:
        cmd += [str(b) for b in blocking]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                          preexec_fn=lambda: os.sched_setaffinity(0, cpus))
    sys.stderr.write(proc.stderr)
    lines = proc.stdout.strip().splitlines()
    if not lines:
        sys.exit(f"FAIL: {' '.join(cmd)} produced no result (exit {proc.returncode})")
    res = json.loads(lines[-1])
    res["exit_code"] = proc.returncode
    res["cpus"] = cpus
    return res


failures = []
results = []
compiler = subprocess.run(["g++", "--version"], capture_output=True, text=True).stdout
model = next((line.split(":", 1)[1].strip() for line in
              pathlib.Path("/proc/cpuinfo").read_text().splitlines()
              if line.startswith("model name")), "unknown")
with tempfile.TemporaryDirectory() as tmp:
    exe = pathlib.Path(tmp) / "gemm_bench"
    sources = [ROOT / "gemm.cpp", ROOT / "matrix.cpp", ROOT / "bench" / "original_matrix.cpp",
               ROOT / "bench" / "gemm_bench.cpp"]
    extra = []
    if have_avx512:
        sources.append(ROOT / "matrix_optimize_4.cpp")
        extra.append("-DHAVE_HUAIYU")
    if have_openblas:
        extra.append("-DHAVE_OPENBLAS")
    build = ["g++", *FLAGS, *extra, f"-I{ROOT}", *map(str, sources), "-o", str(exe)]
    if have_openblas:
        build += shlex.split(args.blas_flags)
    print("build:", " ".join(build))
    print(f"compiler: {compiler.splitlines()[0] if compiler else 'unknown'}")
    print(f"CPU: {model}; physical cores available: {len(CORES)}; thread counts: {threads}"
          + (f" (skipped {skipped}: more than the available cores)" if skipped else ""))
    print("placement: t threads on one logical CPU of each of the first t physical cores; "
          "OpenMP runs: OMP_PLACES=cores OMP_PROC_BIND=close", flush=True)
    if subprocess.run(build).returncode != 0:
        sys.exit("FAIL: build error")

    if args.sweep:
        s = args.sweep_size
        grid_mc = [48, 72, 96, 144, 192, 288]
        grid_kc = [128, 192, 256, 384, 512]
        print(f"\nblocking sweep, blocked gemm, n={s}, GFLOP/s (median of {args.runs}); "
              "mc is rounded up to a multiple of MR")
        for t in sorted({1, max(threads)}):
            print(f"\nthreads={t}")
            corner = "mc \\ kc"
            print(f"{corner:>8} " + " ".join(f"{kc:>7}" for kc in grid_kc))
            for mc in grid_mc:
                row = []
                for kc in grid_kc:
                    r = run(exe, "blocked", s, s, s, t, args.runs, (mc, kc, 4096))
                    if r["exit_code"] != 0 or not r["ok"]:
                        failures.append(f"sweep mc={mc} kc={kc} t={t}: check failed")
                    row.append(r["gflops"])
                print(f"{mc:>8} " + " ".join(f"{g:>7.1f}" for g in row), flush=True)
        print("\nOVERALL:", "PASS" if not failures else "FAIL")
        for f in failures:
            print("  " + f)
        sys.exit(0 if not failures else 1)

    shapes = [(s, s, s) for s in args.sizes]
    if args.thin:
        big = max(args.sizes)
        shapes += [(big, c, big) for c in args.thin]
    for (m, n, k) in shapes:
        square = m == n == k
        if square and m <= args.naive_max:
            results.append(run(exe, "naive", m, n, k, 1, args.naive_runs))
            results.append(run(exe, "loop", m, n, k, 1, args.naive_runs))
        if square and have_avx512:
            results.append(run(exe, "huaiyu", m, n, k, 1, args.runs))
        for t in threads:
            results.append(run(exe, "blocked", m, n, k, t, args.runs))
            if have_openblas:
                results.append(run(exe, "openblas", m, n, k, t, args.runs))
        print(f"  m={m} n={n} k={k} done", flush=True)

for r in results:
    if r["exit_code"] != 0 or not r["ok"]:
        failures.append(f"{r['variant']} m={r['m']} n={r['n']} k={r['k']} t={r['threads']}: "
                        f"check failed (max_err {r['max_err']}, exit {r['exit_code']})")

by_key = {(r["variant"], r["m"], r["n"], r["k"], r["threads"]): r for r in results}
first = results[0]
openblas_cores = sorted({r["openblas_core"] for r in results if r["variant"] == "openblas"})
print(f"\nkernel: {first['kernel']}; blocking mc={first['mc']} kc={first['kc']} nc={first['nc']}; "
      f"runs: {args.runs} (naive, loop {args.naive_runs}); OpenBLAS core: "
      f"{', '.join(openblas_cores) if openblas_cores else 'not run'}")
print("speedup: vs blocked on 1 thread; effic.: speedup / threads; "
      "% of OpenBLAS: OpenBLAS time / this time, at the same thread count\n")
header = (f"{'m':>5} {'n':>5} {'k':>5} {'variant':>8} {'thr':>3} {'median ms':>11} "
          f"{'GFLOP/s':>8} {'speedup':>8} {'effic.':>7} {'% of OpenBLAS':>13} {'max err':>9}")
print(header)
print("-" * len(header))
for r in results:
    key = (r["m"], r["n"], r["k"])
    base = by_key.get(("blocked", *key, 1))
    ob = by_key.get(("openblas", *key, r["threads"]))
    speed = eff = pct = ""
    if r["variant"] == "blocked" and base:
        sp = base["median"] / r["median"]
        speed, eff = f"{sp:.2f}", f"{sp / r['threads']:.2f}"
    if r["variant"] in ("blocked", "huaiyu", "naive", "loop") and ob:
        pct = f"{100 * ob['median'] / r['median']:.1f}"
    print(f"{r['m']:>5} {r['n']:>5} {r['k']:>5} {r['variant']:>8} {r['threads']:>3} "
          f"{1e3 * r['median']:>11.3f} {r['gflops']:>8.2f} {speed:>8} {eff:>7} {pct:>13} "
          f"{r['max_err']:>9.2e}")

print("\nOVERALL:", "PASS" if not failures else "FAIL")
for f in failures:
    print("  " + f)
sys.exit(0 if not failures else 1)
