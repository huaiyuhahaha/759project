"""Shared input, accuracy, repeat and artifact rules for the opt-in baselines."""

import argparse
import json
import math
import os
from pathlib import Path
import socket
import statistics

import numpy as np

NROOTS = 4
TOL = 1e-8


def parse_cpus(value):
    try:
        cpus = [int(cpu) for cpu in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("CPUs must be comma-separated integers") from exc
    if not cpus or min(cpus) < 0 or len(set(cpus)) != len(cpus):
        raise argparse.ArgumentTypeError("CPUs must be distinct nonnegative integers")
    return cpus


def add_arguments(parser):
    parser.add_argument("n", type=int)
    parser.add_argument("input", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--cores", type=int, required=True)
    parser.add_argument("--cpus", type=parse_cpus, required=True)
    parser.add_argument("--config-id", default="standalone")
    parser.add_argument("--min-time", type=float, default=1.0)
    parser.add_argument("--min-runs", type=int, default=5)
    parser.add_argument("--max-runs", type=int, default=200)
    parser.add_argument("--smoke", action="store_true")


def validate_args(args):
    if args.n < 24:
        raise ValueError("n must be >= 24 (the aligned GD subspace dimension)")
    if args.cores < 1 or len(args.cpus) != args.cores or len(set(args.cpus)) != args.cores:
        raise ValueError("--cpus must contain exactly one distinct CPU per core")
    if min(args.cpus) < 0:
        raise ValueError("CPU IDs must be nonnegative")
    if not math.isfinite(args.min_time) or args.min_time < 0:
        raise ValueError("--min-time must be finite and nonnegative")
    if args.min_runs < 1 or not 1 <= args.max_runs <= 200:
        raise ValueError("--min-runs must be positive and --max-runs between 1 and 200")
    if args.smoke and args.max_runs > 3:
        raise ValueError("smoke mode permits at most three measured solves")


def input_checksum(*arrays):
    """Same position-weighted uint64 checksum as the unchanged C++ dump runner."""
    words = np.concatenate([np.ascontiguousarray(a, dtype="<f8").ravel() for a in arrays]).view("<u8")
    weights = np.arange(words.size, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    return f"{int(np.sum(words * weights, dtype=np.uint64)):016x}"


def load_inputs(n, path):
    path = Path(path)
    if path.stat().st_size != 8 * (n * n + n * NROOTS):
        raise ValueError("input file length does not match n*n + n*4 float64 values")
    data = np.fromfile(path, dtype="<f8")
    matrix = data[:n*n].reshape(n, n)
    guess = data[n*n:].reshape(n, NROOTS)
    if not np.isfinite(data).all() or not np.array_equal(matrix, matrix.T):
        raise ValueError("input must be finite with an exactly symmetric matrix")
    return matrix, guess, input_checksum(matrix, guess)


def load_reference(path):
    path = Path(path)
    if path.stat().st_size != 8 * NROOTS:
        raise ValueError("reference must contain exactly four float64 values")
    values = np.fromfile(path, dtype="<f8")
    if not np.isfinite(values).all() or np.any(np.diff(values) < 0):
        raise ValueError("reference eigenvalues must be finite and sorted")
    return values


def accuracy(matrix, values, vectors, reference, converged):
    values = np.asarray(values, dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.float64)
    shape_ok = values.shape == (NROOTS,) and vectors.shape == (len(matrix), NROOTS)
    finite = bool(np.isfinite(values).all() and np.isfinite(vectors).all())
    result = {"converged": bool(converged), "eigenvalues": values.tolist(),
              "reference_eigenvalues": np.asarray(reference).tolist(), "finite": finite,
              "shape_ok": shape_ok, "residual_slack": 1.0}
    if not shape_ok or not finite:
        result.update(gate_pass=False, max_eigenvalue_error=None, max_residual=None,
                      orth_error=None, residuals=[], eigenvalue_errors=[],
                      failure_reason="missing or non-finite eigenpairs")
        return result
    residuals = np.linalg.norm(matrix @ vectors - vectors * values, axis=0)
    errors = np.abs(values - reference)
    orth_error = float(np.linalg.norm(vectors.T @ vectors - np.eye(NROOTS), "fro"))
    checks = {"converged": bool(converged), "eigenvalue_error": bool(np.max(errors) < TOL),
              "residual": bool(np.max(residuals) < TOL), "orthogonality": orth_error < TOL}
    result.update(gate_pass=all(checks.values()), checks=checks, residuals=residuals.tolist(),
                  eigenvalue_errors=errors.tolist(), max_eigenvalue_error=float(np.max(errors)),
                  max_residual=float(np.max(residuals)), orth_error=orth_error,
                  failure_reason=", ".join(k for k, ok in checks.items() if not ok) or None)
    return result


def resource_report(cpus, threads):
    import threadpoolctl

    actual = sorted(os.sched_getaffinity(0))
    pools = threadpoolctl.threadpool_info()
    errors = []
    if actual != sorted(cpus):
        errors.append(f"affinity {actual} differs from requested {sorted(cpus)}")
    if not any(pool.get("user_api") == "blas" for pool in pools):
        errors.append("no BLAS thread pool detected")
    for pool in pools:
        if pool.get("num_threads") != threads:
            errors.append(f"{pool.get('filepath')} has {pool.get('num_threads')} threads; expected {threads}")
    masks = {}
    for status in Path("/proc/self/task").glob("*/status"):
        try:
            line = next(row for row in status.read_text().splitlines() if row.startswith("Cpus_allowed_list:"))
            mask = line.split(":", 1)[1].strip()
            masks[mask] = masks.get(mask, 0) + 1
        except (OSError, StopIteration):
            continue
    return {"ok": not errors, "errors": errors, "expected_cpus": list(cpus),
            "affinity": actual, "threadpools": pools, "thread_cpu_affinity": masks,
            "thread_env": {k: os.environ.get(k) for k in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
                "OMP_PROC_BIND", "OMP_PLACES", "GOMP_CPU_AFFINITY", "KMP_AFFINITY")}}


def base_record(args, solver):
    return {"schema_version": 1, "solver": solver, "backend": solver, "n": args.n,
            "k": NROOTS, "nroots": NROOTS, "cores": args.cores,
            "ranks": args.cores if solver == "slepc" else 1, "config_id": args.config_id,
            "hostname": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID"),
            "evidence_location": "euler" if os.environ.get("SLURM_JOB_ID") else "local",
            "smoke": bool(args.smoke)}


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(v) for v in value]
    if hasattr(value, "tolist"):
        return clean_json(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(stream, record):
    stream.write(json.dumps(clean_json(record), sort_keys=True, allow_nan=False) + "\n")
    stream.flush()


def should_continue(args, samples):
    if not samples:
        return True
    if not samples[-1].get("gate_pass", False):
        return False
    measured = [sample for sample in samples if sample["phase"] == "measured"]
    if len(measured) >= args.max_runs:
        return False
    return len(measured) < args.min_runs or sum(s["elapsed_s"] for s in measured) < args.min_time


def summarize(args, solver, samples):
    measured = [s for s in samples if s.get("phase") == "measured"]
    times = [s.get("elapsed_s") for s in measured]
    valid_times = all(isinstance(t, (int, float)) and math.isfinite(t) and t >= 0 for t in times)
    passed = bool(samples) and all(s.get("gate_pass", False) for s in samples)
    passed = passed and valid_times
    total = sum(times) if valid_times else None
    met_time, met_runs = total is not None and total >= args.min_time, len(times) >= args.min_runs
    reason = "failed_gate" if not passed else "targets_met" if met_time and met_runs else "max_runs"
    result = {**base_record(args, solver), "record_type": "summary", "runs": len(times),
              "warmup": sum(s.get("phase") == "warmup" for s in samples),
              "warmups": sum(s.get("phase") == "warmup" for s in samples),
              "sample_count": len(samples), "times": times,
              "median": statistics.median(times) if times and valid_times else None,
              "median_ms": statistics.median(times) * 1000 if times and valid_times else None,
              "min_ms": min(times) * 1000 if times and valid_times else None,
              "max_ms": max(times) * 1000 if times and valid_times else None,
              "total_timed_s": total, "min_time_s": args.min_time,
              "min_runs": args.min_runs, "max_runs": args.max_runs,
              "minimum_time_met": met_time, "minimum_runs_met": met_runs,
              "stop_reason": reason, "all_converged": bool(samples) and all(s.get("converged", False) for s in samples),
              "gate_pass": passed, "strict_gate_pass": passed, "ok": passed,
              "residual_slack": 1.0,
              "failure_reason": "; ".join(dict.fromkeys(str(s.get("failure_reason") or "failed sample gate")
                                                        for s in samples if not s.get("gate_pass"))) or None}
    for key, target in (("max_eigenvalue_error", "max_eigenvalue_error"),
                        ("max_residual", "max_residual"), ("orth_error", "max_orth_error")):
        values = [s.get(key) for s in samples]
        result[target] = max(values) if values and all(isinstance(v, (int, float)) and math.isfinite(v) for v in values) else None
    result["orth_error"] = result["max_orth_error"]
    for key in ("iterations", "matvecs"):
        values = [s.get(key) for s in measured]
        valid = bool(values) and all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in values)
        result[key + "_min"] = min(values) if valid else None
        result[key + "_median"] = statistics.median(values) if valid else None
        result[key + "_max"] = max(values) if valid else None
    ratios = [1000 * s["elapsed_s"] / s["matvecs"] for s in measured
              if valid_times and isinstance(s.get("matvecs"), (int, float))
              and math.isfinite(s["matvecs"]) and s["matvecs"] > 0]
    result["median_ms_per_matvec"] = statistics.median(ratios) if ratios and len(ratios) == len(measured) else None
    for key in ("input_checksum", "iteration_definition", "matvec_count_method", "settings", "versions", "resources"):
        result[key] = samples[-1].get(key) if samples else None
    result["eps_view_paths"] = [s["eps_view_path"] for s in samples if s.get("eps_view_path")]
    return result
