#!/usr/bin/env python3
"""SLEPc GD on the shared C++ input dump (Stage 3 baseline runner).

Launch with the environment's verified MPI launcher and one physical CPU per
rank, with BLAS/OpenMP threads fixed to one before Python starts. See
docs/baselines/design.md for the alignment and the validated E2 lifecycle.

Official interfaces checked before implementation:
https://slepc.upv.es/release/slepc4py/reference/slepc4py.SLEPc.EPS.html
https://slepc.upv.es/release/manualpages/EPS/EPSReset.html
https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.Mat.html
https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.LogEvent.html

Only setUp()+solve() is timed. Input, reset, initial-vector copies, barriers,
eigenvector gathering, validation, logging, and output are outside that timer.
Raw PETSc event counts are diagnostic; vector-equivalent matvecs remain null.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time
import traceback

NROOTS = 4
TOL = 1e-8
MAX_IT = 100
NCV = 24
# PETSc's MatMatMult event names are registered in src/mat/interface/dlregismat.c.
EVENT_NAMES = ("MatMult", "BVMatMult", "MatMatMultSym", "MatMatMultNum")
COUNT_NOTE = ("Raw per-rank PETSc event counts only; calibration does not "
              "establish coverage of every GD operator path. Do not sum rank "
              "counts as independent global matrix applications.")


def configured_eps(PETSc, SLEPc):
    eps = SLEPc.EPS().create(comm=PETSc.COMM_WORLD)
    eps.setType(SLEPc.EPS.Type.GD)
    eps.setProblemType(SLEPc.EPS.ProblemType.HEP)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
    eps.setDimensions(nev=NROOTS, ncv=NCV)
    eps.setTolerances(tol=TOL, max_it=MAX_IT)
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
    return eps


def effective_settings(eps):
    st, bv = eps.getST(), eps.getBV()
    return {
        "type": eps.getType(), "problem_type": int(eps.getProblemType()),
        "which": int(eps.getWhichEigenpairs()), "dimensions": eps.getDimensions(),
        "tolerances": eps.getTolerances(), "convergence_test": int(eps.getConvergenceTest()),
        "extraction": int(eps.getExtraction()), "gd_block_size": eps.getGDBlockSize(),
        "gd_initial_size": eps.getGDInitialSize(), "gd_krylov_start": eps.getGDKrylovStart(),
        "gd_double_expansion": eps.getGDDoubleExpansion(), "gd_restart": eps.getGDRestart(),
        "st_type": st.getType(), "shift": float(st.getShift()),
        "ksp_type": st.getKSP().getType(), "pc_type": st.getKSP().getPC().getType(),
        "bv_type": bv.getType(), "bv_matmult_method": int(bv.getMatMultMethod()),
        "bv_orthogonalization": bv.getOrthogonalization(),
        "preconditioner_note": "Fixed zero-shift Jacobi; not the guarded Ritz-dependent C++/PySCF denominator.",
        "timed_region": "EPS.setUp + EPS.solve", "mpi_boundary_barriers_timed": False,
        "eigenvector_extraction_timed": False,
    }


def configuration_matches(settings, SLEPc):
    return bool(
        settings["type"] == "gd" and settings["problem_type"] == SLEPc.EPS.ProblemType.HEP
        and settings["which"] == SLEPc.EPS.Which.SMALLEST_REAL
        and tuple(settings["dimensions"][:2]) == (NROOTS, NCV)
        and tuple(settings["tolerances"]) == (TOL, MAX_IT)
        and settings["convergence_test"] == SLEPc.EPS.Conv.ABS
        and settings["extraction"] == SLEPc.EPS.Extraction.RITZ
        and settings["gd_block_size"] == NROOTS and settings["gd_initial_size"] == NROOTS
        and not settings["gd_krylov_start"] and not settings["gd_double_expansion"]
        and settings["st_type"] == "precond" and settings["shift"] == 0.0
        and settings["ksp_type"] == "preonly" and settings["pc_type"] == "jacobi"
        and settings["bv_matmult_method"] == SLEPc.BV.MatMultType.VECS
    )


def event_counts(events, stage):
    return {name: int(event.getPerfInfo(stage.id)["count"]) for name, event in events.items()}


def write_view(PETSc, path, view):
    viewer = PETSc.Viewer().createASCII(str(path), mode="w", comm=PETSc.COMM_WORLD)
    try:
        view(viewer)
    finally:
        viewer.destroy()


def make_matrix(matrix, comm, PETSc, common, np):
    """Fill PETSc's actual local dense array, respecting its storage layout."""
    n = len(matrix)
    mat = PETSc.Mat().createDense(size=(n, n), comm=PETSc.COMM_WORLD)
    mat.setUp()
    lo, hi = mat.getOwnershipRange()
    local = mat.getDenseArray()
    if local.shape != (hi - lo, n):
        raise RuntimeError(f"unexpected local dense shape {local.shape}, ownership {(lo, hi)}")
    local[:, :] = matrix[lo:hi, :]
    del local
    mat.assemble()
    local = mat.getDenseArray(readonly=True)
    matches = bool(np.array_equal(local, matrix[lo:hi, :]))
    fingerprint = common.input_checksum(local)
    del local
    ranges = comm.allgather((lo, hi))
    coverage = np.zeros(n, dtype=int)
    for begin, end in ranges:
        coverage[begin:end] += 1
    ownership = {
        "ranges": ranges, "each_row_owned_once": bool(np.all(coverage == 1)),
        "matrix_exactly_matches_input_by_rank": comm.allgather(matches),
        "local_matrix_checksums": comm.allgather(fingerprint), "matrix_type": mat.getType(),
    }
    if not ownership["each_row_owned_once"] or not all(ownership["matrix_exactly_matches_input_by_rank"]):
        raise RuntimeError(f"distributed dense matrix does not match shared input: {ownership}")
    return mat, fingerprint, ownership


def calibrate_counts(mat, matrix, comm, MPI, PETSc, events, np):
    """Diagnostic three-call calibration and independent distributed layout test."""
    n = mat.getSize()[0]
    stage = PETSc.Log.Stage("baseline_matmult_calibration")
    x, y = mat.createVecRight(), mat.createVecLeft()
    test_vector = np.linspace(0.25, 1.25, n)
    lo, hi = x.getOwnershipRange()
    x.getArray()[:] = test_vector[lo:hi]
    stage.push()
    try:
        before = event_counts(events, stage)
        for _ in range(3):
            mat.mult(x, y)
        after = event_counts(events, stage)
    finally:
        stage.pop()
    lo, hi = y.getOwnershipRange()
    expected = matrix[lo:hi] @ test_vector
    error = comm.allreduce(float(np.max(np.abs(y.getArray(readonly=True) - expected), initial=0)), op=MPI.MAX)
    scale = comm.allreduce(float(np.max(np.abs(expected), initial=0)), op=MPI.MAX)
    limit = 1e-12 * max(1.0, scale)
    counts = comm.allgather({name: after[name] - before[name] for name in events})
    x.destroy()
    y.destroy()
    return {
        "known_three_matmult_counts_by_rank": counts,
        "calibration_pass": all(count["MatMult"] == 3 for count in counts),
        "matvec_layout_error": error, "matvec_layout_tolerance": limit,
        "matvec_layout_pass": error < limit,
        "solver_matvec_equivalence_verified": False, "required_for_numerical_gate": False,
    }


def solve_once(eps, mat, guess, repeat, view_path, events, comm, MPI, PETSc, SLEPc, np):
    eps.reset()
    eps.setOperators(mat)
    # EPSReset destroys BV; this option must be restored on the replacement.
    eps.getBV().setMatMultMethod(SLEPc.BV.MatMultType.VECS)
    initial = []
    for k in range(NROOTS):
        vector = mat.createVecRight()
        lo, hi = vector.getOwnershipRange()
        vector.getArray()[:] = guess[lo:hi, k]
        initial.append(vector)
    eps.setInitialSpace(initial)
    stage = PETSc.Log.Stage(f"baseline_slepc_repeat{repeat}")
    stage.push()
    try:
        before = event_counts(events, stage)
        comm.Barrier()
        start = time.perf_counter()
        eps.setUp()
        eps.solve()
        local_elapsed = time.perf_counter() - start
        comm.Barrier()
        after = event_counts(events, stage)
    finally:
        stage.pop()
    elapsed = comm.allreduce(local_elapsed, op=MPI.MAX)
    rank_times = comm.gather(local_elapsed, root=0)
    counts = comm.gather({name: after[name] - before[name] for name in events}, root=0)
    nconv, reason, iterations = eps.getConverged(), int(eps.getConvergedReason()), eps.getIterationNumber()
    if len(set(comm.allgather((nconv, reason, iterations)))) != 1:
        raise RuntimeError("SLEPc convergence diagnostics differ across ranks")
    settings = effective_settings(eps)
    settings_by_rank = comm.allgather(settings)
    settings_match = all(value == settings_by_rank[0] for value in settings_by_rank)
    configured = configuration_matches(settings, SLEPc) and settings_match
    write_view(PETSc, view_path, eps.view)
    vector = mat.createVecRight()
    values, imaginary, columns = [], [], []
    for k in range(min(NROOTS, nconv)):
        eigenvalue = complex(eps.getEigenpair(k, vector))
        values.append(eigenvalue.real)
        imaginary.append(eigenvalue.imag)
        lo, hi = vector.getOwnershipRange()
        pieces = comm.gather((lo, hi, vector.getArray(readonly=True).copy()), root=0)
        if comm.rank == 0:
            column = np.empty(len(guess))
            for begin, end, piece in pieces:
                column[begin:end] = piece
            columns.append(column)
    vector.destroy()
    for vector in initial:
        vector.destroy()
    details = {
        "elapsed_s": elapsed, "rank_elapsed_s": rank_times,
        "converged_reason": reason, "nconv": nconv, "iterations": iterations,
        "iteration_definition": "SLEPc outer iterations", "settings": settings,
        "configuration_gate": configured, "settings_match_across_ranks": settings_match,
        "eigenvalue_imaginary_parts": imaginary, "real_eigenvalue_gate": all(abs(v) < 1e-12 for v in imaginary),
        "eps_view_path": str(view_path), "raw_event_counts_by_rank": counts,
        "matvecs": None, "matvec_count_method": "unverified PETSc event diagnostics",
        "matvec_count_note": COUNT_NOTE,
    }
    if comm.rank == 0:
        details["raw_event_counts_min"] = {name: min(count[name] for count in counts) for name in events}
        details["raw_event_counts_max"] = {name: max(count[name] for count in counts) for name in events}
        details["rank_elapsed_min_s"] = min(rank_times)
        details["rank_elapsed_max_s"] = max(rank_times)
    vectors = np.column_stack(columns) if columns else np.empty((len(guess), 0))
    return values, vectors, details


def all_rank_resources(args, common, comm):
    report = common.resource_report([args.cpus[comm.rank]], threads=1)
    report.update({"rank": comm.rank, "hostname": socket.gethostname(), "pid": os.getpid()})
    reports = comm.allgather(report)
    return {"ok": all(item["ok"] for item in reports), "ranks": reports,
            "errors": [f"rank {item['rank']}: {error}" for item in reports for error in item["errors"]]}


def run(args, common, comm, MPI, stream, np, PETSc, SLEPc):
    import mpi4py
    import petsc4py
    import slepc4py
    import threadpoolctl

    if comm.size != args.cores:
        raise ValueError(f"--cores={args.cores} but MPI launched {comm.size} ranks")
    if PETSc.COMM_WORLD.getSize() != comm.size or PETSc.COMM_WORLD.getRank() != comm.rank:
        raise RuntimeError("PETSc and mpi4py COMM_WORLD disagree")
    if np.dtype(PETSc.ScalarType) != np.dtype(np.float64):
        raise RuntimeError(f"real float64 PETSc required; got {PETSc.ScalarType}")
    matrix, guess, checksum = common.load_inputs(args.n, args.input)
    # load_inputs returns views into one allocation. Detach the small guess so
    # releasing the full input on non-root ranks actually releases its memory.
    guess = guess.copy()
    reference = common.load_reference(args.reference)
    checksums = comm.allgather(checksum)
    reference_checksums = comm.allgather(common.input_checksum(reference))
    if len(set(checksums)) != 1 or len(set(reference_checksums)) != 1:
        raise RuntimeError("input or reference checksum differs across MPI ranks")
    matrix.flags.writeable = False
    guess.flags.writeable = False
    reference.flags.writeable = False
    initial_resources = all_rank_resources(args, common, comm)
    if not initial_resources["ok"]:
        raise RuntimeError(f"resource gate failed before timing: {initial_resources['errors']}")
    mat, fingerprint, ownership = make_matrix(matrix, comm, PETSc, common, np)
    if not PETSc.Log.isActive():
        PETSc.Log.begin()
    # An unobserved named event can be zero; it is never treated as verified work.
    events = {name: PETSc.Log.Event(name) for name in EVENT_NAMES}
    diagnostics = calibrate_counts(mat, matrix, comm, MPI, PETSc, events, np)
    if not diagnostics["matvec_layout_pass"]:
        raise RuntimeError(f"distributed matrix-vector layout check failed: {diagnostics}")
    if comm.rank != 0:
        matrix = None  # Only rank zero needs the full original matrix for checks.
    versions = {
        "python": platform.python_version(), "numpy": np.__version__,
        "petsc": PETSc.Sys.getVersion(), "slepc": SLEPc.Sys.getVersion(),
        "petsc4py": petsc4py.__version__, "slepc4py": slepc4py.__version__,
        "mpi4py": mpi4py.__version__, "mpi_library": MPI.Get_library_version(),
        "mpi_vendor": MPI.get_vendor(), "threadpoolctl": threadpoolctl.__version__,
        "petsc_scalar_type": str(np.dtype(PETSc.ScalarType)),
    }
    rank_versions = comm.allgather(versions)
    eps = configured_eps(PETSc, SLEPc)
    samples = []
    first_settings = None
    repeat = 0
    log_path = args.view_dir / "petsc-log.txt" if args.log_view else None
    while True:
        proceed = common.should_continue(args, samples) if comm.rank == 0 else None
        if not comm.bcast(proceed, root=0):
            break
        view_path = args.view_dir / f"repeat{repeat:03d}.view.txt"
        values, vectors, details = solve_once(
            eps, mat, guess, repeat, view_path, events, comm, MPI, PETSc, SLEPc, np)
        unchanged_local = common.input_checksum(mat.getDenseArray(readonly=True)) == fingerprint
        unchanged_by_rank = comm.allgather(unchanged_local)
        resources = all_rank_resources(args, common, comm)
        sample = None
        if comm.rank == 0:
            checked = common.accuracy(matrix, values, vectors, reference,
                                      details["converged_reason"] > 0 and details["nconv"] >= NROOTS)
            if first_settings is None:
                first_settings = details["settings"]
            settings_preserved = first_settings == details["settings"]
            unchanged = all(unchanged_by_rank) and common.input_checksum(matrix, guess) == checksum
            failures = [checked["failure_reason"]] if checked.get("failure_reason") else []
            extra_gates = {
                "resource gate": resources["ok"], "effective configuration": details["configuration_gate"],
                "settings retained across reset": settings_preserved,
                "real eigenvalues": details["real_eigenvalue_gate"], "input unchanged": unchanged,
            }
            failures.extend(name for name, passed in extra_gates.items() if not passed)
            sample = {
                **common.base_record(args, "slepc"), "record_type": "sample",
                "sample_index": repeat, "sample_id": f"{args.config_id}:{repeat}",
                "phase": "warmup" if repeat == 0 else "measured", **checked, **details,
                "ranks": comm.size, "omp_threads": 1, "blas_threads": 1,
                "input_checksum": checksum, "input_checksums_by_rank": checksums,
                "reference_checksum": reference_checksums[0], "input_unchanged": unchanged,
                "operator_unchanged_by_rank": unchanged_by_rank,
                "versions": versions, "versions_by_rank": rank_versions, "resources": resources,
                "settings_preserved": settings_preserved, "ownership": ownership,
                "count_diagnostics": diagnostics, "petsc_log_path": str(log_path) if log_path else None,
                "profiled": bool(args.log_view), "smoke": bool(args.smoke),
                "gate_pass": bool(checked["gate_pass"] and all(extra_gates.values())),
                "failure_reason": "; ".join(failures) or None,
            }
            common.write_json(stream, sample)
            samples.append(sample)
        # Root owns samples and makes every stopping decision; no rank-local loop.
        repeat += 1
    if log_path is not None:
        write_view(PETSc, log_path, PETSc.Log.view)
    eps.destroy()
    mat.destroy()
    summary = None
    if comm.rank == 0:
        summary = common.summarize(args, "slepc", samples)
        summary.update({"ranks": comm.size, "omp_threads": 1, "blas_threads": 1,
                        "input_checksums_by_rank": checksums, "ownership": ownership,
                        "count_diagnostics": diagnostics, "versions_by_rank": rank_versions,
                        "matvec_count_note": COUNT_NOTE,
                        "eps_view_paths": [sample["eps_view_path"] for sample in samples],
                        "petsc_log_path": str(log_path) if log_path else None,
                        "profiled": bool(args.log_view), "smoke": bool(args.smoke)})
        common.write_json(sys.stdout, summary)
    passed = comm.bcast(bool(summary["gate_pass"]) if comm.rank == 0 else None, root=0)
    return 0 if passed else 1


def main():
    comm, stream, args, common = None, None, None, None
    rank = int(os.environ.get("PMI_RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0")))
    # baseline_common imports no MPI: the selected launcher's communicator is
    # initialized here before PETSc/SLEPc, using the E2-validated init sequence.
    try:
        import baseline_common as common
        parser = argparse.ArgumentParser(description=__doc__)
        common.add_arguments(parser)
        parser.add_argument("--view-dir", type=Path)
        parser.add_argument("--log-view", action="store_true",
                            help="save optional PETSc diagnostic log outside solve timers")
        args = parser.parse_args()
        common.validate_args(args)
        if args.n < NCV:
            raise ValueError(f"n must be at least {NCV}, the fixed GD subspace dimension")
        from mpi4py import MPI
        comm, rank = MPI.COMM_WORLD, MPI.COMM_WORLD.rank
        import numpy as np
        import slepc4py
        slepc4py.init([sys.argv[0]], comm=comm)
        from petsc4py import PETSc
        from slepc4py import SLEPc

        args.view_dir = args.view_dir or args.samples.parent / (args.samples.stem + "_views")
        if rank == 0:
            args.samples.parent.mkdir(parents=True, exist_ok=True)
            # Refuse old output rather than silently mixing separate runs.
            stream = args.samples.open("x", encoding="utf-8")
            args.view_dir.mkdir(parents=True, exist_ok=False)
        comm.Barrier()
        return run(args, common, comm, MPI, stream, np, PETSc, SLEPc)
    except Exception as exc:
        error = {
            "schema_version": 1, "record_type": "error", "solver": "slepc", "backend": "slepc",
            "rank": rank, "hostname": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID"),
            "config_id": args.config_id if args is not None else None,
            "error_type": type(exc).__name__, "error": str(exc), "failure_reason": str(exc),
            "traceback": traceback.format_exc(), "gate_pass": False, "ok": False,
        }
        def emit_error(destination):
            if common is not None:
                common.write_json(destination, error)
            else:
                print(json.dumps(error, sort_keys=True, allow_nan=False), file=destination, flush=True)

        try:
            emit_error(sys.stderr)
            if args is not None:
                path = args.samples.parent / f"{args.samples.stem}_error_rank{rank}_pid{os.getpid()}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("x", encoding="utf-8") as error_stream:
                    emit_error(error_stream)
            if stream is not None:
                emit_error(stream)
            if rank == 0:
                emit_error(sys.stdout)
        except OSError as write_error:
            print(f"could not save error artifact: {write_error}", file=sys.stderr, flush=True)
        finally:
            if comm is not None and comm.size > 1:
                comm.Abort(1)  # A peer may still be blocked in a collective.
        return 1
    finally:
        if stream is not None:
            stream.close()


if __name__ == "__main__":
    sys.exit(main())
