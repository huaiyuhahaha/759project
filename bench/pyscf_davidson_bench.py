"""PySCF davidson1 on the C++ benchmark's inputs.

    python3 bench/pyscf_davidson_bench.py <n> <input-file>

Run through bench/run_davidson_bench.py. That driver writes <input-file> with
`davidson_bench <n> dump`, which contains the C++ benchmark's seeded matrix A
and its initialize_vector(4, A) guess as raw float64 bytes. It also sets the
thread environment (OMP_NUM_THREADS, *_NUM_THREADS) before this process starts.

Solver call: pyscf.lib.linalg_helper.davidson1 with nroots=4, the four guess
columns as x0, and tol_residual=1e-8 (the C++ residual tolerance). davidson1
also requires every |eigenvalue change| < tol; that tolerance is set to its
documented default (1e-12) and recorded. max_cycle=100 matches the C++
max_iter. max_space is davidson1's default, also recorded. aop applies A to
each vector with A.dot(x), one matvec per basis vector as in the C++ solver.

lindep is 1e-16 = kLinDepTol^2 rather than davidson1's default of 1e-14.
davidson1 drops root k when ||r_k||^2 <= lindep, so with 1e-14 no residual
below 1e-7 is ever refined and tol_residual=1e-8 cannot be reached: it stops
on linear dependence with roots unconverged. davidson1 normalizes each
preconditioned vector before orthogonalizing it, so lindep=1e-16 also gives
exactly the C++ test (remaining norm <= 1e-8 x original norm => dropped).

The preconditioner is the C++ solver's guarded diagonal one:
    r_i / (e - A_ii), with |e - A_ii| < 1e-8 clamped to +-1e-8 (sign kept).
davidson1 itself chooses the shift: it calls precond(r_k, e[0], x_k), so every
root uses the lowest Ritz value. The C++ loop uses each root's own value.

Only the davidson1 call is timed: one warm-up solve, then five timed solves,
reporting the median. All checks run after the timed solves. The four
convergence flags must all be True. Residuals ||A x_k - e_k x_k||_2 and
||X^T X - I||_F are recomputed from A. The script prints one JSON line;
exit code 0 = ok, 1 = check failed, 2 = bad input.
"""

import json
import os
import platform
import statistics
import sys
import time

import numpy as np
import scipy
import pyscf
from pyscf.lib import linalg_helper, num_threads

try:
    import threadpoolctl
except ImportError:  # optional: only used to record the thread pools
    threadpoolctl = None

NROOTS = 4
TOL_RESIDUAL = 1e-8     # residual-norm criterion, same as kTol in davidson_bench.cpp
EIG_CHANGE_TOL = 1e-12  # davidson1 `tol`: |e_k - e_k(previous cycle)| < tol (its default)
MAX_CYCLE = 100         # kMaxIter in davidson_bench.cpp
MAX_SPACE = 12          # davidson1 default (it adds (nroots - 1) * 4 internally)
LINDEP = 1e-16          # kLinDepTol^2; davidson1 default 1e-14 cannot reach 1e-8 (see above)
DENOM_FLOOR = 1e-8      # kDenomFloor in davidson.cpp
WARMUP, RUNS = 1, 5
RESIDUAL_SLACK = 10.0   # same acceptance as davidson_bench.cpp
ORTH_ACCEPT = 1e-8


def input_checksum(*arrays):
    """Same function as input_checksum() in davidson_bench.cpp."""
    words = np.concatenate([np.ascontiguousarray(a, dtype="<f8").ravel() for a in arrays])
    words = words.view("<u8")
    weights = np.arange(words.size, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    return f"{int(np.sum(words * weights, dtype=np.uint64)):016x}"


def guarded_diag_precond(diag):
    def precond(r, e, x):
        d = e - diag
        small = np.abs(d) < DENOM_FLOOR
        d[small] = np.copysign(DENOM_FLOOR, d[small])
        return r / d
    return precond


class CountingMatvec:
    """aop for davidson1: one A.dot(x) per vector; counts cycles and matvecs."""

    def __init__(self, A):
        self.A = A
        self.calls = 0
        self.matvecs = 0

    def __call__(self, xs):
        self.calls += 1
        self.matvecs += len(xs)
        return [self.A.dot(x) for x in xs]


def thread_report():
    env_vars = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS", "OMP_PLACES", "OMP_PROC_BIND")
    report = {"env": {v: os.environ.get(v) for v in env_vars},
              "pyscf_lib_num_threads": num_threads()}
    if threadpoolctl is not None:
        report["threadpools"] = [
            {"api": p["internal_api"], "version": p.get("version"),
             "threading_layer": p.get("threading_layer"), "num_threads": p["num_threads"],
             "library": os.path.basename(p["filepath"])}
            for p in threadpoolctl.threadpool_info()]
    # Allowed CPUs of every OS thread in this process (e.g. {"0-15": 9}).
    affinity = {}
    for tid in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{tid}/status") as f:
                for line in f:
                    if line.startswith("Cpus_allowed_list:"):
                        cpus = line.split(":", 1)[1].strip()
                        affinity[cpus] = affinity.get(cpus, 0) + 1
        except OSError:
            pass
    report["thread_cpu_affinity"] = affinity
    return report


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    n = int(sys.argv[1])
    data = np.fromfile(sys.argv[2], dtype="<f8")
    if data.size != n * n + n * NROOTS:
        print(f"input has {data.size} values, expected {n * n + n * NROOTS}", file=sys.stderr)
        return 2
    A = data[: n * n].reshape(n, n)
    guess = data[n * n:].reshape(n, NROOTS)
    checksum = input_checksum(A, guess)
    diag = np.diag(A).copy()
    precond = guarded_diag_precond(diag)

    space_seen = {"effective_max_space": None, "restarts": 0}

    def record_space(envs):  # warm-up only: davidson1's internal subspace limit
        space_seen["effective_max_space"] = envs["max_space"]
        space_seen["restarts"] += bool(envs["fresh_start"])

    times, eig_runs = [], []
    for run in range(WARMUP + RUNS):
        x0 = [np.ascontiguousarray(guess[:, k]) for k in range(NROOTS)]
        aop = CountingMatvec(A)
        callback = record_space if run == 0 else None
        t0 = time.perf_counter()
        conv, e, x = linalg_helper.davidson1(
            aop, x0, precond, tol=EIG_CHANGE_TOL, tol_residual=TOL_RESIDUAL,
            max_cycle=MAX_CYCLE, max_space=MAX_SPACE, lindep=LINDEP, nroots=NROOTS,
            callback=callback)
        elapsed = time.perf_counter() - t0
        if run >= WARMUP:
            times.append(elapsed)
        eig_runs.append(np.array(e, dtype=float))

    # Independent checks, outside the timer.
    conv = [bool(c) for c in np.atleast_1d(conv)]
    e = np.array(e, dtype=float)
    X = np.column_stack(x)
    residuals = [float(np.linalg.norm(A.dot(X[:, k]) - e[k] * X[:, k]))
                 for k in range(X.shape[1])]
    orth_error = float(np.linalg.norm(X.T @ X - np.eye(X.shape[1]), "fro"))
    max_residual = max(residuals) if residuals else float("nan")
    deterministic = all(np.array_equal(r, eig_runs[0]) for r in eig_runs)
    ok = (len(conv) == NROOTS and all(conv) and e.size == NROOTS and X.shape == (n, NROOTS)
          and max_residual < RESIDUAL_SLACK * TOL_RESIDUAL and orth_error < ORTH_ACCEPT)

    result = {
        "n": n, "backend": "pyscf", "threads": int(os.environ.get("OMP_NUM_THREADS", "0")),
        "k": NROOTS, "warmup": WARMUP, "runs": RUNS, "times": times,
        "median": statistics.median(times),
        "settings": {"tol_residual": TOL_RESIDUAL, "eig_change_tol": EIG_CHANGE_TOL,
                     "max_cycle": MAX_CYCLE, "max_space_arg": MAX_SPACE,
                     "lindep": LINDEP, "denom_floor": DENOM_FLOOR, **space_seen},
        "conv": conv, "converged": bool(conv) and all(conv),
        "iterations": aop.calls, "matvecs": aop.matvecs,
        "eigenvalues": e.tolist(), "deterministic": deterministic,
        "residuals": residuals, "max_residual": max_residual, "orth_error": orth_error,
        "input_checksum": checksum,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "pyscf": pyscf.__version__,
                     "threadpoolctl": threadpoolctl.__version__ if threadpoolctl else None},
        "thread_report": thread_report(),
        "ok": ok,
    }
    print(json.dumps(result))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
