#!/usr/bin/env python3
"""Stage E2 numerical/environment checks, not the production benchmark.

  python -B euler/check_numerics.py --solver pyscf --output-dir /tmp/e2-local
  mpiexec -n 2 python -B euler/check_numerics.py --solver slepc --output-dir PATH
  python -B euler/check_numerics.py --solver pyscf --sizes 64 128 --output-dir PATH

Set affinity and BLAS thread limits before Python starts. Official API references
and alignment decisions: docs/baselines/design.md. These small synthetic inputs
are NOT the C++ benchmark's input stream.
"""

import argparse
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
import time
import traceback

NROOTS = 4
TOL = 1e-8
REPEATS = 2
THREAD_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "GOMP_CPU_AFFINITY", "KMP_AFFINITY",
)


def clean_json(value):
    """Keep failure records valid JSON, including non-finite numerical results."""
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if hasattr(value, "tolist"):
        return clean_json(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def encode(value):
    return json.dumps(clean_json(value), sort_keys=True, allow_nan=False)


def rank_environment(rank, extra_versions=None):
    import numpy as np
    import pyscf
    import scipy
    import threadpoolctl

    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "pyscf": pyscf.__version__,
                "threadpoolctl": threadpoolctl.__version__}
    versions.update(extra_versions or {})
    tasks = {}
    for status in Path("/proc/self/task").glob("*/status"):
        try:
            line = next(x for x in status.read_text().splitlines()
                        if x.startswith("Cpus_allowed_list:"))
            cpus = line.split(":", 1)[1].strip()
            tasks[cpus] = tasks.get(cpus, 0) + 1
        except (OSError, StopIteration):
            continue
    actual = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    expected = os.environ.get("BASELINES_EXPECTED_CPUS")
    expected_cpu = int(expected.split(",")[rank]) if expected else None
    pools = threadpoolctl.threadpool_info()
    pool_failures = [{"library": pool.get("filepath"), "num_threads": pool.get("num_threads")}
                     for pool in pools if pool.get("num_threads") != 1]
    if not pools:
        pool_failures.append({"reason": "threadpoolctl reported no pools; single-thread BLAS is unverified"})
    return {"rank": rank, "pid": os.getpid(), "hostname": socket.gethostname(),
            "affinity": actual, "expected_cpu": expected_cpu,
            "affinity_gate": actual == [expected_cpu] if expected else True,
            "affinity_check_requested": bool(expected), "thread_cpu_affinity": tasks,
            "threadpools": pools, "threadpool_gate": not pool_failures,
            "threadpool_gate_failures": pool_failures,
            "thread_env": {name: os.environ.get(name) for name in THREAD_VARS}, "versions": versions}


def smoke_inputs(n):
    import numpy as np

    rng = np.random.default_rng(759)
    upper = np.triu(rng.uniform(-1.0, 1.0, size=(n, n)) / n, k=1)
    matrix = upper + upper.T
    np.fill_diagonal(matrix, np.arange(1, n + 1, dtype=np.float64))
    return matrix, np.eye(n, NROOTS, dtype=np.float64)


def accuracy(matrix, values, vectors, reference, converged):
    import numpy as np

    values, vectors = np.asarray(values, dtype=np.float64), np.asarray(vectors, dtype=np.float64)
    shape_ok = values.shape == (NROOTS,) and vectors.shape == (len(matrix), NROOTS)
    finite = bool(np.isfinite(values).all() and np.isfinite(vectors).all())
    if not shape_ok or not finite:
        return {"gate_pass": False, "finite": finite, "shape_ok": shape_ok,
                "eigenvalues": values.tolist(), "failure_reason": "missing or non-finite eigenpairs"}
    residuals = np.linalg.norm(matrix @ vectors - vectors * values, axis=0)
    errors = np.abs(values - reference)
    orth_error = float(np.linalg.norm(vectors.T @ vectors - np.eye(NROOTS), "fro"))
    checks = {"converged": bool(converged), "eigenvalue_error": bool(np.max(errors) < TOL),
              "residual": bool(np.max(residuals) < TOL), "orthogonality": orth_error < TOL}
    return {"gate_pass": all(checks.values()), "checks": checks, "finite": finite,
            "shape_ok": shape_ok, "eigenvalues": values.tolist(),
            "reference_eigenvalues": reference.tolist(), "eigenvalue_errors": errors.tolist(),
            "residuals": residuals.tolist(), "max_eigenvalue_error": float(np.max(errors)),
            "max_residual": float(np.max(residuals)), "orth_error": orth_error, "residual_slack": 1.0,
            "failure_reason": ", ".join(name for name, passed in checks.items() if not passed) or None}


def repeated_state(samples):
    import numpy as np

    first, second = samples
    values_match = (len(first["eigenvalues"]) == len(second["eigenvalues"])
                    and bool(np.allclose(first["eigenvalues"], second["eigenvalues"], rtol=0.0, atol=1e-12)))
    counts_match = first["iterations"] == second["iterations"]
    return {"eigenvalues_match_atol_1e_12": values_match,
            "iterations_match": counts_match, "passed": values_match and counts_match}


def check_pyscf(n, output_dir, helpers):
    import numpy as np
    from pyscf.lib import linalg_helper

    matrix, guess = smoke_inputs(n)
    checksum = helpers.input_checksum(matrix, guess)
    reference = np.linalg.eigvalsh(matrix)[:NROOTS]
    samples = []
    settings = {"nroots": NROOTS, "tol_residual": TOL, "tol": 1e-12,
                "max_cycle": 100, "max_space_argument": 12, "lindep": 1e-16,
                "denominator_floor": 1e-8, "shift": "lowest Ritz value for every root"}
    for repeat in range(REPEATS):
        operator = helpers.CountingMatvec(matrix)
        state = {"effective_max_space": None, "fresh_start_callbacks": 0}

        def observe(envs):
            state["effective_max_space"] = int(envs["max_space"])
            state["fresh_start_callbacks"] += int(bool(envs["fresh_start"]))

        initial = [guess[:, k].copy() for k in range(NROOTS)]
        precond = helpers.guarded_diag_precond(np.diag(matrix).copy())
        start = time.perf_counter()
        conv, values, vectors = linalg_helper.davidson1(
            operator, initial, precond, tol=1e-12, tol_residual=TOL,
            max_cycle=100, max_space=12, lindep=1e-16, nroots=NROOTS, callback=observe)
        elapsed = time.perf_counter() - start
        flags = [bool(flag) for flag in np.atleast_1d(conv)]
        vectors = np.column_stack(vectors) if len(vectors) else np.empty((n, 0))
        sample = accuracy(matrix, values, vectors, reference, len(flags) == NROOTS and all(flags))
        sample.update({"repeat": repeat, "elapsed_s": elapsed, "convergence_flags": flags,
                       "iterations": operator.calls, "iteration_definition": "operator callback count",
                       "matvecs": operator.matvecs, "settings": {**settings, **state}})
        samples.append(sample)
    lifecycle = repeated_state(samples)
    unchanged = checksum == helpers.input_checksum(matrix, guess)
    environment = rank_environment(0)
    return {"samples": samples, "input_checksum": checksum, "lifecycle": lifecycle,
            "input_unchanged": unchanged, "rank_environment": [environment],
            "numerical_gate_pass": all(s["gate_pass"] for s in samples),
            "affinity_gate_pass": environment["affinity_gate"],
            "threadpool_gate_pass": environment["threadpool_gate"],
            "ok": all(s["gate_pass"] for s in samples) and lifecycle["passed"]
                  and unchanged and environment["affinity_gate"] and environment["threadpool_gate"]}


def event_counts(events, stage):
    return {name: int(event.getPerfInfo(stage.id)["count"]) for name, event in events.items()}


def check_slepc(n, output_dir, helpers, comm, MPI):
    import numpy as np
    import mpi4py
    import petsc4py
    import slepc4py
    from petsc4py import PETSc
    from slepc4py import SLEPc

    rank, ranks = comm.rank, comm.size
    if np.dtype(PETSc.ScalarType) != np.dtype(np.float64):
        raise RuntimeError(f"real float64 PETSc required; got {PETSc.ScalarType}")
    matrix, guess = comm.bcast(smoke_inputs(n) if rank == 0 else None, root=0)
    checksum = helpers.input_checksum(matrix, guess)
    checksums = comm.allgather(checksum)
    if any(value != checksums[0] for value in checksums):
        raise RuntimeError("synthetic input checksum differs across ranks")
    reference = comm.bcast(np.linalg.eigvalsh(matrix)[:NROOTS] if rank == 0 else None, root=0)
    petsc_comm = PETSc.COMM_WORLD
    if petsc_comm.getSize() != ranks or petsc_comm.getRank() != rank:
        raise RuntimeError("PETSc and mpi4py COMM_WORLD disagree; check the MPI build and launcher")
    mat = PETSc.Mat().createDense(size=(n, n), comm=petsc_comm)
    mat.setUp()
    lo, hi = mat.getOwnershipRange()
    local = mat.getDenseArray()
    if local.shape != (hi - lo, n):
        raise RuntimeError(f"unexpected local dense shape {local.shape}, ownership {(lo, hi)}")
    local[:, :] = matrix[lo:hi, :]
    del local
    mat.assemble()
    owned = comm.gather((lo, hi, mat.getDenseArray(readonly=True).copy()), root=0)
    ownership = None
    if rank == 0:
        rebuilt, coverage = np.empty_like(matrix), np.zeros(n, dtype=int)
        for begin, end, slab in owned:
            rebuilt[begin:end] = slab
            coverage[begin:end] += 1
        ownership = {"ranges": [[begin, end] for begin, end, _ in owned],
                     "each_row_owned_once": bool(np.all(coverage == 1)),
                     "matrix_exactly_matches_input": bool(np.array_equal(rebuilt, matrix)),
                     "reconstructed_input_checksum": helpers.input_checksum(rebuilt, guess)}
    ownership = comm.bcast(ownership, root=0)

    # Calibrate the raw event outside solver stages with three known MatMult calls.
    if not PETSc.Log.isActive():
        PETSc.Log.begin()
    events = {name: PETSc.Log.Event(name) for name in ("MatMult", "BVMatMult")}
    calibration = PETSc.Log.Stage(f"E2_calibration_n{n}")
    x, y = mat.createVecRight(), mat.createVecLeft()
    xlo, xhi = x.getOwnershipRange()
    test_vector = np.linspace(0.25, 1.25, n)
    x.getArray()[:] = test_vector[xlo:xhi]
    calibration.push()
    before = event_counts(events, calibration)
    for _ in range(3):
        mat.mult(x, y)
    after = event_counts(events, calibration)
    calibration.pop()
    ylo, yhi = y.getOwnershipRange()
    expected = matrix[ylo:yhi] @ test_vector
    local_error = float(np.max(np.abs(y.getArray(readonly=True) - expected), initial=0))
    layout_error = comm.allreduce(local_error, op=MPI.MAX)
    layout_limit = 1e-12 * max(1.0, float(np.max(np.abs(matrix @ test_vector))))
    calibration_counts = comm.allgather(after["MatMult"] - before["MatMult"])
    ownership.update({"matvec_layout_error": layout_error, "matvec_layout_tolerance": layout_limit,
                      "matvec_layout_pass": layout_error < layout_limit,
                      "known_three_matmult_event_counts_by_rank": calibration_counts,
                      "matmult_calibration_pass": all(count == 3 for count in calibration_counts)})
    x.destroy()
    y.destroy()

    eps = SLEPc.EPS().create(comm=petsc_comm)
    eps.setType(SLEPc.EPS.Type.GD)
    eps.setProblemType(SLEPc.EPS.ProblemType.HEP)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
    eps.setDimensions(nev=NROOTS, ncv=24)
    eps.setTolerances(tol=TOL, max_it=100)
    eps.setConvergenceTest(SLEPc.EPS.Conv.ABS)
    eps.setExtraction(SLEPc.EPS.Extraction.RITZ)
    eps.setGDBlockSize(NROOTS)
    eps.setGDInitialSize(NROOTS)
    eps.setGDKrylovStart(False)
    eps.setGDDoubleExpansion(False)
    st = eps.getST()
    st.setType(SLEPc.ST.Type.PRECOND)
    st.setShift(0.0)
    st.getKSP().setType(PETSc.KSP.Type.PREONLY)
    st.getKSP().getPC().setType(PETSc.PC.Type.JACOBI)

    samples = []
    for repeat in range(REPEATS):
        eps.reset()
        eps.setOperators(mat)
        # EPSReset destroys BV objects even though it preserves EPS options.
        eps.getBV().setMatMultMethod(SLEPc.BV.MatMultType.VECS)
        initial = []
        for k in range(NROOTS):
            vector = mat.createVecRight()
            begin, end = vector.getOwnershipRange()
            vector.getArray()[:] = guess[begin:end, k]
            initial.append(vector)
        eps.setInitialSpace(initial)
        stage = PETSc.Log.Stage(f"E2_slepc_n{n}_repeat{repeat}")
        stage.push()
        before = event_counts(events, stage)
        comm.Barrier()
        start = time.perf_counter()
        eps.setUp()
        eps.solve()
        local_elapsed = time.perf_counter() - start
        comm.Barrier()
        after = event_counts(events, stage)
        stage.pop()
        elapsed = comm.allreduce(local_elapsed, op=MPI.MAX)
        rank_times = comm.gather(local_elapsed, root=0)
        raw_counts = comm.gather({name: after[name] - before[name] for name in events}, root=0)
        nconv, reason = eps.getConverged(), int(eps.getConvergedReason())
        if len(set(comm.allgather((nconv, reason, eps.getIterationNumber())))) != 1:
            raise RuntimeError("SLEPc convergence diagnostics differ across ranks")
        settings = {"type": eps.getType(), "problem_type": int(eps.getProblemType()),
                    "which": int(eps.getWhichEigenpairs()), "dimensions": eps.getDimensions(),
                    "tolerances": eps.getTolerances(), "convergence_test": int(eps.getConvergenceTest()),
                    "extraction": int(eps.getExtraction()), "gd_block_size": eps.getGDBlockSize(),
                    "gd_initial_size": eps.getGDInitialSize(), "gd_krylov_start": eps.getGDKrylovStart(),
                    "gd_double_expansion": eps.getGDDoubleExpansion(), "gd_restart": eps.getGDRestart(),
                    "st_type": st.getType(), "shift": float(st.getShift()),
                    "ksp_type": st.getKSP().getType(), "pc_type": st.getKSP().getPC().getType(),
                    "bv_type": eps.getBV().getType(), "bv_matmult_method": int(eps.getBV().getMatMultMethod()),
                    "bv_orthogonalization": eps.getBV().getOrthogonalization()}
        configuration_ok = (
            settings["type"] == "gd" and settings["problem_type"] == SLEPc.EPS.ProblemType.HEP
            and settings["which"] == SLEPc.EPS.Which.SMALLEST_REAL
            and tuple(settings["dimensions"][:2]) == (NROOTS, 24)
            and tuple(settings["tolerances"]) == (TOL, 100)
            and settings["convergence_test"] == SLEPc.EPS.Conv.ABS
            and settings["extraction"] == SLEPc.EPS.Extraction.RITZ
            and settings["gd_block_size"] == NROOTS and settings["gd_initial_size"] == NROOTS
            and not settings["gd_krylov_start"] and not settings["gd_double_expansion"]
            and settings["st_type"] == "precond" and settings["shift"] == 0.0
            and settings["ksp_type"] == "preonly" and settings["pc_type"] == "jacobi"
            and settings["bv_matmult_method"] == SLEPc.BV.MatMultType.VECS)
        view_path = output_dir / f"slepc_n{n}_r{ranks}_repeat{repeat}.view.txt"
        viewer = PETSc.Viewer().createASCII(str(view_path), mode="w", comm=petsc_comm)
        eps.view(viewer)
        viewer.destroy()
        vector = mat.createVecRight()
        values, imaginary, columns = [], [], []
        for k in range(min(NROOTS, nconv)):
            eigenvalue = complex(eps.getEigenpair(k, vector))
            values.append(eigenvalue.real)
            imaginary.append(eigenvalue.imag)
            begin, end = vector.getOwnershipRange()
            pieces = comm.gather((begin, end, vector.getArray(readonly=True).copy()), root=0)
            if rank == 0:
                column = np.empty(n)
                for begin, end, piece in pieces:
                    column[begin:end] = piece
                columns.append(column)
        vector.destroy()
        for vector in initial:
            vector.destroy()
        sample = None
        if rank == 0:
            vectors = np.column_stack(columns) if columns else np.empty((n, 0))
            sample = accuracy(matrix, values, vectors, reference, reason > 0 and nconv >= NROOTS)
            sample.update({"repeat": repeat, "elapsed_s": elapsed, "rank_elapsed_s": rank_times,
                           "converged_reason": reason, "nconv": nconv,
                           "iterations": eps.getIterationNumber(), "iteration_definition": "SLEPc outer iterations",
                           "eigenvalue_imaginary_parts": imaginary, "settings": settings,
                           "configuration_gate": bool(configuration_ok),
                           "eps_view_path": str(view_path), "raw_event_counts_by_rank": raw_counts,
                           "matvecs": None,
                           "matvec_count_note": "Raw PETSc events only; three-call calibration does not establish coverage of every GD operator path."})
            real_values = all(abs(v) < 1e-12 for v in imaginary)
            sample["real_eigenvalue_gate"] = real_values
            extra_failures = ([] if configuration_ok else ["effective configuration mismatch"])
            if not real_values:
                extra_failures.append("non-real eigenvalues for a real symmetric problem")
            if sample.get("failure_reason"):
                extra_failures.insert(0, sample["failure_reason"])
            sample["failure_reason"] = ", ".join(extra_failures) or None
            sample["gate_pass"] = sample["gate_pass"] and bool(configuration_ok) and real_values
        samples.append(comm.bcast(sample, root=0))

    log_path = output_dir / f"slepc_n{n}_r{ranks}.petsc-log.txt"
    viewer = PETSc.Viewer().createASCII(str(log_path), mode="w", comm=petsc_comm)
    PETSc.Log.view(viewer)
    viewer.destroy()
    versions = {"petsc": PETSc.Sys.getVersion(), "slepc": SLEPc.Sys.getVersion(),
                "petsc4py": petsc4py.__version__, "slepc4py": slepc4py.__version__,
                "mpi4py": mpi4py.__version__, "mpi_library": MPI.Get_library_version(),
                "petsc_scalar_type": str(np.dtype(PETSc.ScalarType))}
    environment = comm.gather(rank_environment(rank, versions), root=0)
    resources = None
    if rank == 0:
        resources = {"affinity_gate_pass": all(info["affinity_gate"] for info in environment),
                     "threadpool_gate_pass": all(info["threadpool_gate"] for info in environment)}
    resources = comm.bcast(resources, root=0)
    lifecycle = repeated_state(samples)
    lifecycle["settings_preserved"] = clean_json(samples[0]["settings"]) == clean_json(samples[1]["settings"])
    lifecycle["passed"] = lifecycle["passed"] and lifecycle["settings_preserved"]
    unchanged = checksum == helpers.input_checksum(matrix, guess)
    ownership_ok = all(ownership[key] for key in
                       ("each_row_owned_once", "matrix_exactly_matches_input", "matvec_layout_pass"))
    result = {"samples": samples, "input_checksum": checksum, "input_checksums_by_rank": checksums,
              "input_unchanged": unchanged, "lifecycle": lifecycle, "ownership": ownership,
              "petsc_log_path": str(log_path), "rank_environment": environment,
              "numerical_gate_pass": all(s["gate_pass"] for s in samples), **resources,
              "count_diagnostics": {"calibration_pass": ownership["matmult_calibration_pass"],
                                    "solver_matvec_equivalence_verified": False,
                                    "required_for_numerical_gate": False},
              "ok": all(s["gate_pass"] for s in samples) and lifecycle["passed"]
                    and unchanged and ownership_ok and all(resources.values())}
    eps.destroy()
    mat.destroy()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("slepc", "pyscf"), required=True)
    sizes = parser.add_mutually_exclusive_group()
    sizes.add_argument("--size", type=int, default=100)
    sizes.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    requested_sizes = args.sizes or [args.size]
    if any(n < 24 for n in requested_sizes) or len(set(requested_sizes)) != len(requested_sizes):
        parser.error("sizes must be distinct integers >= 24 (the GD subspace dimension)")
    comm = None
    rank = int(os.environ.get("PMI_RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0")))
    try:
        if args.solver == "slepc":
            from mpi4py import MPI
            comm, rank = MPI.COMM_WORLD, MPI.COMM_WORLD.rank
            import slepc4py
            slepc4py.init([sys.argv[0]], comm=comm)
        else:
            launcher_size = int(os.environ.get("PMI_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", "1")))
            if launcher_size != 1:
                raise RuntimeError("run the PySCF checker as one process, not under multiple MPI ranks")
        ranks = comm.size if comm is not None else 1
        expected = os.environ.get("BASELINES_EXPECTED_CPUS")
        if expected:
            cpus = [int(cpu) for cpu in expected.split(",")]
            if len(cpus) != ranks or len(set(cpus)) != ranks or min(cpus) < 0:
                raise ValueError("BASELINES_EXPECTED_CPUS must list one distinct nonnegative CPU per rank")
        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            for n in requested_sizes:
                path = args.output_dir / f"{args.solver}_n{n}_r{ranks}.json"
                if path.exists():
                    raise FileExistsError(f"refusing to overwrite {path}; choose a fresh output directory")
        if comm is not None:
            comm.Barrier()
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
        import pyscf_davidson_bench as helpers
        passed = True
        for n in requested_sizes:
            result = (check_slepc(n, args.output_dir, helpers, comm, MPI) if args.solver == "slepc"
                      else check_pyscf(n, args.output_dir, helpers))
            result.update({"record_type": "E2_numerical_check", "solver": args.solver, "n": n,
                           "nroots": NROOTS, "ranks": ranks, "repeat_count": REPEATS,
                           "input_kind": "synthetic NumPy default_rng seed759; not final C++ benchmark inputs",
                           "evidence_location": "euler" if os.environ.get("SLURM_JOB_ID") else "local",
                           "job_id": os.environ.get("SLURM_JOB_ID"), "hostname": socket.gethostname(),
                           "timing_purpose": "environment smoke only; exclude from benchmark report",
                           "mpi_boundary_barriers_timed": False})
            if rank == 0:
                with (args.output_dir / f"{args.solver}_n{n}_r{ranks}.json").open("x", encoding="utf-8") as stream:
                    stream.write(encode(result) + "\n")
                print(encode(result), flush=True)
            passed = passed and result["ok"]
        if comm is not None:
            passed = comm.bcast(passed if rank == 0 else None, root=0)
        return 0 if passed else 1
    except Exception as exc:
        error = {"record_type": "E2_error", "solver": args.solver, "rank": rank,
                 "hostname": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID"),
                 "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "ok": False}
        print(encode(error), file=sys.stderr, flush=True)
        try:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            path = args.output_dir / f"{args.solver}_error_rank{rank}_pid{os.getpid()}.json"
            with path.open("x", encoding="utf-8") as stream:
                stream.write(encode(error) + "\n")
        except OSError as write_error:
            print(f"could not save error artifact: {write_error}", file=sys.stderr, flush=True)
        if comm is not None and comm.size > 1:
            comm.Abort(1)  # A peer may still be inside a collective; do not wait for it.
        return 1


if __name__ == "__main__":
    sys.exit(main())
