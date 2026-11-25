"""Opt-in Euler baseline mode; the original PySCF runner keeps its defaults.

Invoke through pyscf_davidson_bench.py with --baseline. The driver must set
process affinity and thread limits before Python imports its numerical libraries.
Each sample times only davidson1, including its operator/callback instrumentation.
"""

import argparse
from functools import partial
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np

import baseline_common as common
import pyscf_davidson_bench as legacy


BINDING_VARS = ("OMP_PROC_BIND", "OMP_PLACES", "GOMP_CPU_AFFINITY", "KMP_AFFINITY")


def versions():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": legacy.scipy.__version__, "pyscf": legacy.pyscf.__version__,
            "threadpoolctl": (legacy.threadpoolctl.__version__
                              if legacy.threadpoolctl is not None else None)}


def settings():
    return {"nroots": legacy.NROOTS, "problem": "real symmetric",
            "which": "smallest eigenvalues", "tol_residual": legacy.TOL_RESIDUAL,
            "eig_change_tol": legacy.EIG_CHANGE_TOL, "max_cycle": legacy.MAX_CYCLE,
            "max_space_arg": legacy.MAX_SPACE, "lindep": legacy.LINDEP,
            "denom_floor": legacy.DENOM_FLOOR,
            "preconditioner": "guarded r_i / (shift - A_ii)",
            "preconditioner_shift": "lowest Ritz value for every root (chosen by davidson1)",
            "timing_region": "davidson1 call, including operator and observation callbacks",
            "residual_slack": 1.0}


def resource_state(args):
    report = common.resource_report(args.cpus, args.cores)
    report["pyscf_lib_num_threads"] = int(legacy.num_threads())
    report["binding_env"] = {name: os.environ.get(name) for name in BINDING_VARS}
    errors = list(report["errors"])
    if report["pyscf_lib_num_threads"] != args.cores:
        errors.append("pyscf.lib.num_threads() does not match --cores")
    if any(value for value in report["binding_env"].values()):
        errors.append("binding variables must be cleared before Python starts")
    report["errors"] = errors
    report["ok"] = bool(report["ok"] and not errors)
    return report


def solve_once(matrix, guess, profiler=None):
    """Construct fresh per-call state, then return the timed native result."""
    initial = [guess[:, k].copy() for k in range(legacy.NROOTS)]
    operator = legacy.CountingMatvec(matrix)
    preconditioner = legacy.guarded_diag_precond(np.diag(matrix).copy())
    state = {"effective_max_space": None, "fresh_start_callbacks": 0,
             "observation_callbacks": 0}

    def observe(envs):
        state["effective_max_space"] = int(envs["max_space"])
        state["fresh_start_callbacks"] += int(bool(envs["fresh_start"]))
        state["observation_callbacks"] += 1

    lifecycle = {
        "fresh_operator": True,
        "fresh_preconditioner": True,
        "initial_vectors_are_independent_copies": all(
            not np.shares_memory(vector, guess) for vector in initial),
        "initial_vectors_match_input": bool(np.array_equal(np.column_stack(initial), guess)),
        "initial_guess_checksum": common.input_checksum(np.column_stack(initial)),
    }
    solver = legacy.linalg_helper.davidson1
    if profiler is not None:
        solver = partial(profiler.runcall, solver)
    start = time.perf_counter()
    conv, values, vectors = solver(
        operator, initial, preconditioner,
        tol=legacy.EIG_CHANGE_TOL, tol_residual=legacy.TOL_RESIDUAL,
        max_cycle=legacy.MAX_CYCLE, max_space=legacy.MAX_SPACE,
        lindep=legacy.LINDEP, nroots=legacy.NROOTS, callback=observe)
    elapsed = time.perf_counter() - start
    flags = [bool(flag) for flag in np.atleast_1d(conv)]
    values = np.asarray(values)
    vectors = np.column_stack(vectors) if len(vectors) else np.empty((matrix.shape[0], 0))
    return {"values": values, "vectors": vectors, "flags": flags,
            "elapsed_s": elapsed, "iterations": operator.calls, "matvecs": operator.matvecs,
            "observed": state, "lifecycle": lifecycle}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_arguments(parser)
    parser.add_argument("--baseline", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--profile", type=Path,
                        help="optional cProfile output for measured davidson1 calls only; also writes a .txt sibling")
    args = parser.parse_args(argv)
    try:
        common.validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    samples = []
    stream = None
    failure = None
    current_resources = None
    checksum = None
    profiler = None
    profile_stream = profile_text_stream = None
    profile_calls = 0
    current_versions = versions()
    fixed_settings = settings()
    metadata = common.base_record(args, "pyscf")
    try:
        args.samples.parent.mkdir(parents=True, exist_ok=True)
        stream = args.samples.open("x", encoding="utf-8")
        if args.profile is not None:
            import cProfile
            args.profile.parent.mkdir(parents=True, exist_ok=True)
            profile_text_path = args.profile.with_suffix(".txt")
            if profile_text_path == args.profile:
                raise ValueError("--profile must use a filename with a suffix other than .txt")
            if args.profile.exists() or profile_text_path.exists():
                raise FileExistsError("profile output already exists; choose a fresh path")
            profile_stream = args.profile.open("xb")
            profile_text_stream = profile_text_path.open("x", encoding="utf-8")
            profiler = cProfile.Profile()
        matrix, guess, checksum = common.load_inputs(args.n, args.input)
        reference = common.load_reference(args.reference)
        # The solver gets copied guesses. Its operator may only read the matrix.
        matrix.setflags(write=False)
        guess.setflags(write=False)
        while common.should_continue(args, samples):
            sample_index = len(samples)
            before = resource_state(args)
            current_resources = before
            if not before["ok"]:
                raise RuntimeError("resource check failed before solve: " + "; ".join(before["errors"]))
            before_checksum = common.input_checksum(matrix, guess)
            if before_checksum != checksum:
                raise RuntimeError("input checksum changed before solve")
            active_profiler = profiler if sample_index > 0 else None
            profile_calls += int(active_profiler is not None)
            result = solve_once(matrix, guess, active_profiler)
            after_checksum = common.input_checksum(matrix, guess)
            current_resources = resource_state(args)
            current_resources["before_solve"] = before
            flags = result["flags"]
            checks = common.accuracy(matrix, result["values"], result["vectors"], reference,
                                     len(flags) == legacy.NROOTS and all(flags))
            lifecycle = result["lifecycle"]
            lifecycle["input_unchanged"] = before_checksum == after_checksum == checksum
            lifecycle["input_checksum_before"] = before_checksum
            lifecycle["input_checksum_after"] = after_checksum
            fresh_state = (lifecycle["initial_vectors_are_independent_copies"]
                           and lifecycle["initial_vectors_match_input"]
                           and lifecycle["input_unchanged"])
            # Repeated results are diagnostics; correctness does not require bitwise determinism.
            if samples:
                first = samples[0]
                first_values = np.asarray(first["eigenvalues"])
                lifecycle["iterations_match_warmup"] = result["iterations"] == first["iterations"]
                lifecycle["matvecs_match_warmup"] = result["matvecs"] == first["matvecs"]
                lifecycle["eigenvalues_match_warmup_atol_1e_12"] = bool(
                    first_values.shape == result["values"].shape
                    and np.allclose(first_values, result["values"], rtol=0.0, atol=1e-12))
            errors = ([checks["failure_reason"]] if checks.get("failure_reason") else [])
            errors.extend(current_resources["errors"])
            if not fresh_state:
                errors.append("fresh-state or input-immutability check failed")
            sample = {**metadata, "record_type": "sample", "sample_index": sample_index,
                      "phase": "warmup" if sample_index == 0 else "measured",
                      "elapsed_s": result["elapsed_s"], **checks,
                      "gate_pass": bool(checks["gate_pass"] and current_resources["ok"] and fresh_state),
                      "failure_reason": "; ".join(errors) or None,
                      "input_checksum": checksum, "iterations": result["iterations"],
                      "matvecs": result["matvecs"], "iteration_definition": "operator callback count",
                      "matvec_count_method": "number of vectors passed to the operator",
                      "settings": {**fixed_settings, **result["observed"]},
                      "versions": current_versions, "resources": current_resources,
                      "profiled": active_profiler is not None,
                      "profile_path": str(args.profile) if active_profiler is not None else None,
                      "convergence_flags": flags, "lifecycle": lifecycle}
            common.write_json(stream, sample)
            samples.append(sample)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        traceback.print_exc(file=sys.stderr)
        if stream is not None:
            common.write_json(stream, {**metadata, "record_type": "error", "gate_pass": False,
                                       "failure_reason": failure, "sample_index": len(samples),
                                       "input_checksum": checksum, "settings": fixed_settings,
                                       "profiled": profile_calls > 0,
                                       "profile_path": str(args.profile) if profile_stream is not None else None,
                                       "versions": current_versions, "resources": current_resources})
    finally:
        if profiler is not None:
            try:
                import marshal
                import pstats
                # cProfile.dump_stats uses this marshal format. Writing to the
                # exclusively opened stream avoids overwriting an existing file.
                profiler.create_stats()
                marshal.dump(profiler.stats, profile_stream)
                profile_stream.flush()
                if profiler.stats:
                    pstats.Stats(profiler, stream=profile_text_stream).strip_dirs().sort_stats(
                        "cumulative").print_stats(60)
                else:
                    profile_text_stream.write("No measured solver call was profiled.\n")
            except Exception as exc:
                message = f"profile write failed: {type(exc).__name__}: {exc}"
                failure = "; ".join(filter(None, [failure, message]))
                traceback.print_exc(file=sys.stderr)
                if stream is not None:
                    common.write_json(stream, {**metadata, "record_type": "error", "gate_pass": False,
                                               "failure_reason": message, "profiled": profile_calls > 0,
                                               "profile_path": str(args.profile)})
        for profile_file in (profile_stream, profile_text_stream):
            if profile_file is not None:
                profile_file.close()
        if stream is not None:
            stream.close()

    summary = common.summarize(args, "pyscf", samples)
    summary.update(profiled=profile_calls > 0,
                   profile_path=str(args.profile) if profile_stream is not None else None)
    if failure is not None:
        summary.update(ok=False, gate_pass=False, strict_gate_pass=False,
                       stop_reason="error", failure_reason=failure)
        if not samples:
            summary.update(input_checksum=checksum, settings=fixed_settings,
                           versions=current_versions, resources=current_resources)
    common.write_json(sys.stdout, summary)
    return 0 if summary["gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
