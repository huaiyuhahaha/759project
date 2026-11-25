#!/usr/bin/env python3
"""Run aligned Davidson baselines; every configuration retains samples and logs.

Local example (no SLEPc required):
  python3 -B bench/run_baselines.py --sizes 64 128 --cores 1 --solvers cpp_serial cpp_openmp pyscf \
      --smoke --min-time 0 --max-runs 3 --output-dir /tmp/baselines-smoke
Euler: submit euler/03_smoke.sbatch. This driver never installs packages.
"""

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shlex
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
SOLVERS = ("cpp_serial", "cpp_openmp", "pyscf", "slepc")
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
CLEAR_KEYS = ("OMP_PROC_BIND", "OMP_PLACES", "GOMP_CPU_AFFINITY", "KMP_AFFINITY",
              "PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "LD_LIBRARY_PATH", "PETSC_DIR", "PETSC_ARCH",
              "SLEPC_DIR", "HYDRA_HOST_FILE", "HYDRA_BINDING", "HYDRA_MAPPING", "HYDRA_MEMBIND",
              "HYDRA_TOPOLIB", "HYDRA_TOPO_DEBUG", "HYDRA_LAUNCHER_EXTRA_ARGS")
COMPILER_CLEAR_KEYS = ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "OBJC_INCLUDE_PATH",
                       "LIBRARY_PATH", "COMPILER_PATH", "GCC_EXEC_PREFIX", "LD_LIBRARY_PATH",
                       "LD_PRELOAD", "CPPFLAGS", "CFLAGS", "CXXFLAGS", "LDFLAGS")
CSV_FIELDS = (
    "schema_version", "step", "job_id", "hostname", "source_revision", "sweep", "config_id",
    "n", "nroots", "solver", "cores", "ranks", "omp_threads", "blas_threads", "input_checksum",
    "warmups", "runs", "min_runs", "max_runs", "min_time_s", "total_timed_s", "minimum_time_met",
    "minimum_runs_met", "stop_reason", "median_ms", "min_ms", "max_ms", "iterations_min",
    "iterations_median", "iterations_max", "iteration_definition", "matvecs_min", "matvecs_median",
    "matvecs_max", "matvec_count_method", "median_ms_per_matvec", "all_converged",
    "max_eigenvalue_error", "max_residual", "max_orth_error", "residual_slack", "strict_gate_pass",
    "gate_pass", "failure_reason", "settings_json", "versions_json", "resources_json",
    "cpu_mapping_json", "raw_path", "sample_ids_json", "env_path", "eps_view_paths_json",
    "exit_code", "smoke", "environment_verified", "evidence_location",
    "profiled", "profile_kind", "profile_paths_json",
    "placement_policy", "socket", "numa_node", "memory_policy", "placement_path",
    "rank_placement_paths_json")


def command(command_args, directory, label, env, timeout):
    """No shell; retain both streams even if startup, solver or timeout fails."""
    directory.mkdir(parents=True, exist_ok=True)
    out, err = directory / f"{label}.out", directory / f"{label}.err"
    args = list(map(str, command_args))
    record = {"command": args, "stdout": str(out), "stderr": str(err), "timeout_s": timeout,
              "environment": {k: env.get(k) for k in (*THREAD_KEYS, *CLEAR_KEYS, *COMPILER_CLEAR_KEYS,
                                                     "PATH", "CC", "CXX", "MPIR_CVAR_ENABLE_GPU")}}
    print("+ " + shlex.join(args), flush=True)
    try:
        with out.open("x") as stdout, err.open("x") as stderr:
            proc = subprocess.Popen(args, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                record["exit_code"] = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                record.update(exit_code=124, error="configuration timed out; process group terminated")
    except OSError as exc:
        record.update(exit_code=127, error=str(exc))
    (directory / f"{label}.command.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def last_summary(path):
    for line in reversed(Path(path).read_text(errors="replace").splitlines()):
        try:
            result = json.loads(line)
        except ValueError:
            continue  # MPI diagnostics such as UCX startup notices remain in the log.
        if isinstance(result, dict) and result.get("record_type") == "summary":
            return result
    return None


def source_metadata():
    files = [REPO / name for name in ("matrix.cpp", "matrix.h", "davidson.cpp", "davidson.h")]
    files += sorted((REPO / "bench").glob("*.py")) + sorted((REPO / "bench").glob("*.cpp"))
    files += sorted((REPO / "euler").glob("*.py")) + sorted((REPO / "euler").glob("*.sbatch"))
    files += sorted((REPO / "euler").glob("*.sh"))
    metadata = {"sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    for name, args in (("revision", ["rev-parse", "HEAD"]), ("status", ["status", "--short"])):
        try:
            proc = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, timeout=10)
            metadata[name] = proc.stdout.strip() if proc.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            metadata[name] = None
    return metadata


def build_tools(args, out, env):
    build = out / "build"
    build.mkdir(exist_ok=True)
    build_env = dict(env)
    if os.environ.get("SLURM_JOB_ID"):
        if args.cxx != "/usr/bin/g++":
            raise ValueError("Euler builds require /usr/bin/g++ without a compiler module")
        for key in COMPILER_CLEAR_KEYS:
            build_env.pop(key, None)
        build_env.update(PATH="/usr/bin:/bin", CC="/usr/bin/gcc", CXX="/usr/bin/g++")
    binaries = {"dumper": args.dumper, "cpp": args.cpp_binary}
    compiler_recorded = False
    for key, source in (("dumper", "davidson_bench.cpp"), ("cpp", "cpp_baselines_bench.cpp")):
        if binaries[key] is not None:
            binaries[key] = binaries[key].resolve()
            if not binaries[key].is_file():
                raise ValueError(f"missing {key} executable: {binaries[key]}")
            continue
        if key == "cpp" and not any(s.startswith("cpp_") for s in args.solvers):
            continue
        if not compiler_recorded:
            version = command([args.cxx, "--version"], build, "compiler_version", build_env, 15)
            if version["exit_code"]:
                raise RuntimeError("compiler version check failed; see build/compiler_version.err")
            compiler_recorded = True
        binaries[key] = build / key
        flags = [args.cxx, "-std=c++17", "-O3", "-march=native", "-fopenmp", "-Wall", "-Wextra", "-pedantic"]
        cmd = [*flags, f"-I{REPO}", REPO / "matrix.cpp", REPO / "davidson.cpp", REPO / "bench" / source,
               "-o", binaries[key]]
        result = command(cmd, build, f"compile_{key}", build_env, 120)
        if result["exit_code"]:
            raise RuntimeError(f"{key} compilation failed; see {result['stderr']}")
    for key, binary in binaries.items():
        if binary is not None:
            (build / f"{key}.sha256").write_text(hashlib.sha256(binary.read_bytes()).hexdigest() + "\n")
    return binaries


def check_sample(sample, expected, reference, np):
    errors = []
    for key in ("solver", "n", "cores", "input_checksum"):
        if sample.get(key) != expected[key]:
            errors.append(f"{key} mismatch")
    if sample.get("sample_index") != expected["sample_index"] or sample.get("phase") != expected["phase"]:
        errors.append("sample order or warm-up/measured phase mismatch")
    elapsed = sample.get("elapsed_s")
    if not isinstance(elapsed, (float, int)) or not math.isfinite(elapsed) or elapsed < 0:
        errors.append("invalid elapsed time")
    values = np.asarray(sample.get("eigenvalues", []), dtype=float)
    residuals = np.asarray(sample.get("residuals", []), dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all() or np.max(np.abs(values - reference)) >= 1e-8:
        errors.append("eigenvalues fail shared reference check")
    if residuals.shape != (4,) or not np.isfinite(residuals).all() or np.any(residuals >= 1e-8) or np.any(residuals < 0):
        errors.append("residual gate failed")
    # CSV aggregates come from validated per-root evidence, never a runner's
    # possibly inconsistent summary maxima.
    sample["max_eigenvalue_error"] = float(np.max(np.abs(values - reference))) if values.shape == (4,) and np.isfinite(values).all() else None
    sample["max_residual"] = float(np.max(residuals)) if residuals.shape == (4,) and np.isfinite(residuals).all() else None
    orth = sample.get("orth_error")
    if not isinstance(orth, (float, int)) or not math.isfinite(orth) or not 0 <= orth < 1e-8:
        errors.append("orthogonality gate failed")
    if not sample.get("converged"):
        errors.append("solver did not converge")
    if not isinstance(sample.get("resources"), dict) or not sample["resources"].get("ok", False):
        errors.append("resource verification failed")
    sample["driver_gate_pass"] = not errors
    sample["gate_pass"] = bool(sample.get("gate_pass")) and not errors
    if errors:
        sample["failure_reason"] = "; ".join(filter(None, [sample.get("failure_reason"), *errors]))


def blas_paths(resources):
    result = set()
    if not isinstance(resources, dict):
        return result
    for pool in resources.get("threadpools", []):
        if pool.get("user_api") == "blas" and pool.get("filepath"):
            result.add(str(Path(pool["filepath"]).resolve()))
    for rank in resources.get("ranks", []):
        result.update(blas_paths(rank))
    return result


def run_configuration(args, out, raw, writer, csv_stream, tools, n, solver, cpus, sweep,
                      checksum, reference, provenance, environment, base_env, common, np):
    cores = len(cpus)
    config_id = f"s{sweep}_n{n}_{solver}_p{cores}"
    directory = out / "configs" / config_id
    directory.mkdir(parents=True)
    sample_path = directory / "samples.jsonl"
    profile_kind, profile_paths = None, []
    placement = {}
    if args.single_socket_plan:
        from single_socket import load_verified_plan, validate_selected, command_prefix
        plan = load_verified_plan(args.single_socket_plan, requiredcores=max(args.cores))
        validate_selected(cpus, plan)
        placement = {"placement_policy": "single_socket_single_numa_node",
                     "socket": plan["socket"], "numa_node": plan["numa_node"],
                     "memory_policy": plan["memory_policy"]["mode"], "cpu_mapping": cpus,
                     "placement_path": "single_socket.json"}
        (directory / "placement.json").write_text(json.dumps(placement, indent=2) + "\n")
    options = SimpleNamespace(n=n, cores=cores, cpus=cpus, config_id=config_id,
                              min_time=args.min_time, min_runs=args.min_runs, max_runs=args.max_runs, smoke=args.smoke)
    flags = ["--reference", out / f"reference_n{n}.bin", "--samples", sample_path,
             "--cores", str(cores), "--cpus", ",".join(map(str, cpus)), "--config-id", config_id,
             "--min-time", str(args.min_time), "--min-runs", str(args.min_runs), "--max-runs", str(args.max_runs)]
    if args.smoke:
        flags.append("--smoke")
    env = dict(base_env)
    for key in CLEAR_KEYS:
        env.pop(key, None)
    for key in THREAD_KEYS:
        env[key] = str(cores if solver == "pyscf" else 1)
    env.update(OMP_DYNAMIC="FALSE", PYTHONDONTWRITEBYTECODE="1", MPIR_CVAR_ENABLE_GPU="0")
    if os.environ.get("SLURM_JOB_ID"):
        env["PYTHONNOUSERSITE"] = "1"
    input_file = out / f"input_n{n}.bin"
    if solver.startswith("cpp_"):
        backend = solver.removeprefix("cpp_")
        env["OMP_NUM_THREADS"] = str(cores)
        if solver == "cpp_openmp":
            env.update(OMP_PROC_BIND="TRUE", OMP_PLACES=",".join("{" + str(cpu) + "}" for cpu in cpus))
        cmd = ["taskset", "-c", ",".join(map(str, cpus)), tools["cpp"], str(n), input_file, backend, *flags]
    elif solver == "pyscf":
        cmd = ["taskset", "-c", ",".join(map(str, cpus)), sys.executable, "-B",
               REPO / "bench/pyscf_davidson_bench.py", str(n), input_file, "--baseline", *flags]
        if args.profile_pyscf:
            profile_kind = "cprofile"
            profile_paths = [directory / "profile.prof", directory / "profile.txt"]
            cmd.extend(["--profile", profile_paths[0]])
    else:
        env["MPIEXEC_TIMEOUT"] = str(max(1, int(args.timeout) - 5))
        rank_prefix = []
        if args.single_socket_plan:
            rank_prefix = [sys.executable, "-B", REPO / "euler/single_socket.py", "exec-rank",
                           "--plan", args.single_socket_plan.resolve(), "--cores", str(cores),
                           "--evidence-dir", directory / "rank_placement", "--"]
        cmd = [args.mpiexec, "-launcher", "fork", "-hosts", "localhost", "-bind-to",
               "user:" + ",".join(map(str, cpus)), "-n", str(cores), *rank_prefix, sys.executable, "-B",
               REPO / "bench/slepc_davidson_bench.py", str(n), input_file, *flags,
               "--view-dir", directory / "views"]
        if args.log_view:
            cmd.append("--log-view")
            profile_kind = "petsc_log"
            profile_paths = [directory / "views" / "petsc-log.txt"]
    if args.single_socket_plan:
        # The prefix takes effect before Python, OpenMP, or the local MPI
        # launcher starts. Fork/exec children inherit the task memory policy.
        if solver != "slepc":
            cmd = cmd[3:]  # Replace the legacy taskset prefix with the verified one.
        cmd = [*command_prefix(plan, cpus), *cmd]
    execution = command(cmd, directory, "runner", env, args.timeout)
    samples, errors = [], []
    rank_placement_paths = []
    if args.single_socket_plan and solver == "slepc":
        for rank, cpu in enumerate(cpus):
            path = directory / "rank_placement" / f"rank-{rank}.json"
            rank_placement_paths.append(str(path.relative_to(out)))
            try:
                evidence = json.loads(path.read_text())
                if (evidence.get("rank") != rank or evidence.get("mpi_size") != cores
                        or evidence.get("verified_before_solver_exec") is not True
                        or evidence.get("cpus") != [cpu]
                        or evidence.get("socket") != plan["socket"]
                        or evidence.get("numa_node") != plan["numa_node"]
                        or evidence.get("memory_policy") != plan["memory_policy"]["mode"]):
                    raise ValueError("rank placement differs from selected topology/policy")
            except (OSError, ValueError) as exc:
                errors.append(f"rank {rank} placement evidence failed: {exc}")
    metadata = {"step": args.step, "sweep": sweep, "source_revision": provenance.get("revision"),
                "config_id": config_id, "job_id": os.environ.get("SLURM_JOB_ID"), "hostname": socket.gethostname(),
                **placement}
    if sample_path.exists():
        for line in sample_path.read_text().splitlines():
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("sample record is not a JSON object")
                if record.get("record_type") == "sample":
                    expected = {"solver": solver, "n": n, "cores": cores, "input_checksum": checksum,
                                "sample_index": len(samples), "phase": "warmup" if not samples else "measured"}
                    check_sample(record, expected, reference, np)
                    if environment and solver in ("pyscf", "slepc"):
                        providers = set(environment["blas_alignment"]["openblas_paths"])
                        if blas_paths(record.get("resources", {})) != providers:
                            record["gate_pass"] = False
                            record["failure_reason"] = "runner BLAS providers differ from this job's verified environment"
                    samples.append(record)
                elif record.get("record_type") == "error":
                    errors.append(str(record.get("error") or record.get("failure_reason") or "runner error"))
                else:
                    raise ValueError("unknown sample record type")
                record.update(metadata)
                common.write_json(raw, record)
            except (ValueError, TypeError, KeyError) as exc:
                errors.append(f"invalid raw record: {exc}")
    else:
        errors.append("runner did not produce a sample file")
    summary = common.summarize(options, solver, samples)
    reported = last_summary(execution["stdout"])
    if execution["exit_code"]:
        errors.append(f"runner exit {execution['exit_code']}: {execution.get('error', 'see runner logs')}")
    if reported is None:
        errors.append("runner did not emit a final summary")
    elif reported.get("runs") != summary["runs"] or reported.get("input_checksum") != checksum or not reported.get("ok"):
        errors.append("runner summary failed or disagrees with collected samples")
    if not samples or summary["warmups"] != 1 or summary["runs"] > args.max_runs:
        errors.append("invalid sample counts")
    if summary["ok"] and common.should_continue(options, samples):
        errors.append("runner stopped before repeat targets or cap")
    for path in profile_paths:
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty requested profile: {path.name}")
    if errors:
        summary.update(ok=False, gate_pass=False, strict_gate_pass=False, stop_reason="runner_failure",
                       failure_reason="; ".join(filter(None, [summary.get("failure_reason"), *errors])))
        common.write_json(raw, {**common.base_record(options, solver), **metadata,
                                "record_type": "error", "error": summary["failure_reason"], "ok": False})
    summary.update(metadata, input_checksum=checksum, exit_code=execution["exit_code"],
                   omp_threads=1 if solver == "slepc" else cores, blas_threads=cores if solver == "pyscf" else 1,
                   environment_verified=bool(environment), cpu_mapping=cpus,
                   raw_path=str(sample_path.relative_to(out)), env_path="environment.json" if environment else "driver_environment.json",
                   sample_ids=[s["sample_index"] for s in samples],
                   rank_placement_paths=rank_placement_paths,
                   profiled=bool(profile_kind), profile_kind=profile_kind,
                   profile_paths=[str(path.relative_to(out)) for path in profile_paths])
    common.write_json(raw, summary)
    with (directory / "summary.json").open("x") as stream:
        common.write_json(stream, summary)
    row = dict(summary)
    for source, target in (("settings", "settings_json"), ("versions", "versions_json"),
                           ("resources", "resources_json"), ("cpu_mapping", "cpu_mapping_json"),
                           ("sample_ids", "sample_ids_json"), ("eps_view_paths", "eps_view_paths_json"),
                           ("profile_paths", "profile_paths_json"),
                           ("rank_placement_paths", "rank_placement_paths_json")):
        row[target] = json.dumps(common.clean_json(summary.get(source)), sort_keys=True, allow_nan=False)
    writer.writerow({key: row.get(key) for key in CSV_FIELDS})
    csv_stream.flush()
    print(f"{config_id}: {'PASS' if summary['ok'] else 'FAIL'}; {summary['runs']} measured solves; {summary['stop_reason']}", flush=True)
    return summary["ok"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--cores", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--solvers", choices=SOLVERS, nargs="+", default=list(SOLVERS))
    parser.add_argument("--min-time", type=float, default=1.0)
    parser.add_argument("--min-runs", type=int)
    parser.add_argument("--max-runs", type=int, default=200)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sweeps", type=int, default=1)
    parser.add_argument("--step", default="local")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--environment-json", type=Path)
    parser.add_argument("--single-socket-plan", type=Path,
                        help="require verified E6 placement before any numerical work")
    parser.add_argument("--dumper", type=Path)
    parser.add_argument("--cpp-binary", type=Path)
    parser.add_argument("--cxx", default="/usr/bin/g++" if os.environ.get("SLURM_JOB_ID") else os.environ.get("CXX", "g++"))
    parser.add_argument("--mpiexec", default=str(Path(sys.executable).parent / "mpiexec"))
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--log-view", action="store_true")
    parser.add_argument("--profile-pyscf", action="store_true",
                        help="profile measured PySCF solver calls; keep these timings separate from E4")
    args = parser.parse_args()
    args.min_runs = args.min_runs if args.min_runs is not None else 3 if args.smoke else 5
    if any(n < 24 for n in args.sizes) or any(p < 1 for p in args.cores):
        parser.error("sizes must be >=24 and core counts positive")
    if any(len(values) != len(set(values)) for values in (args.sizes, args.cores, args.solvers)):
        parser.error("sizes, cores and solvers must not contain duplicates")
    if args.sweeps < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("sweeps and timeout must be positive")
    if args.profile_pyscf and "pyscf" not in args.solvers:
        parser.error("--profile-pyscf requires pyscf in --solvers")
    if args.step == "E6" and not args.single_socket_plan:
        parser.error("E6 requires --single-socket-plan")
    sys.path.insert(0, str(REPO / "euler"))
    plan = None
    if args.single_socket_plan:
        from single_socket import load_verified_plan, verify_runtime, command_prefix
        try:
            plan = load_verified_plan(args.single_socket_plan, requiredcores=max(args.cores))
            # The shared job launches this process through single_socket.py exec.
            # Check again before importing numerical libraries or touching inputs.
            runtime = verify_runtime(plan)
        except (ValueError, RuntimeError, OSError) as exc:
            parser.error(f"E6 placement gate: {exc}")
    # The driver does reference eigensolves at one BLAS thread. Child solver
    # processes receive their own limits before any numerical libraries load.
    for key in THREAD_KEYS:
        os.environ[key] = "1"
    for key in ("OMP_PROC_BIND", "OMP_PLACES", "GOMP_CPU_AFFINITY", "KMP_AFFINITY"):
        os.environ.pop(key, None)
    import numpy as np
    import threadpoolctl
    import baseline_common as common
    from run_env_checks import physical_cpus
    try:
        common.validate_args(SimpleNamespace(n=min(args.sizes), cores=1, cpus=[0], min_time=args.min_time,
                                             min_runs=args.min_runs, max_runs=args.max_runs, smoke=args.smoke))
    except ValueError as exc:
        parser.error(str(exc))
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "run.json"
    if manifest_path.exists() or (out / "raw.jsonl").exists() or (out / "summary.csv").exists():
        parser.error("output already contains benchmark artifacts; choose a fresh directory/job")
    manifest = {"schema_version": 1, "arguments": vars(args), "hostname": socket.gethostname(),
                "job_id": os.environ.get("SLURM_JOB_ID"), "status": "running"}
    with manifest_path.open("x") as stream:
        common.write_json(stream, manifest)
    try:
        if int(os.environ.get("SLURM_JOB_NUM_NODES", "1")) != 1:
            raise ValueError("this driver supports exactly one node")
        needed = max(args.cores) if any(s != "cpp_serial" for s in args.solvers) else 1
        if plan:
            cpus = plan["selected"]
            mapping = dict(plan)
            mapping["driver_runtime"] = runtime
            if args.single_socket_plan.resolve() != out / "single_socket.json":
                with (out / "single_socket.json").open("xb") as stream:
                    stream.write(args.single_socket_plan.read_bytes())
        else:
            allowed, physical = physical_cpus()
            if len(physical) < needed:
                raise ValueError(f"requested {needed} physical cores; allocation provides {len(physical)}")
            cpus = [entry["cpu"] for entry in physical]
            mapping = {"allowed_logical_cpus": allowed, "physical_cores": physical, "selected": cpus[:needed]}
        (out / "cpu_mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
        provenance = source_metadata()
        (out / "source.json").write_text(json.dumps(provenance, indent=2) + "\n")
        environment = None
        if args.environment_json:
            environment = json.loads(args.environment_json.read_text())
            if not environment.get("ok") or environment["hostname"] != socket.gethostname() or environment.get("job_id") != os.environ.get("SLURM_JOB_ID"):
                raise ValueError("environment evidence must pass and belong to this job/node")
            if environment["blas_alignment"]["status"] != "verified_shared_loaded_dependency":
                raise ValueError("shared BLAS dependency evidence is unverified")
            if args.environment_json.resolve() != out / "environment.json":
                with (out / "environment.json").open("xb") as stream:
                    stream.write(args.environment_json.read_bytes())
        elif os.environ.get("SLURM_JOB_ID"):
            raise ValueError("Euler runs require --environment-json from this job's inspection")
        config_text = io.StringIO()
        with contextlib.redirect_stdout(config_text):
            np.show_config()
        driver_env = {"python": platform.python_version(), "executable": sys.executable,
                      "numpy": np.__version__, "numpy_show_config": config_text.getvalue(),
                      "threadpools": threadpoolctl.threadpool_info(), "cpu_mapping": mapping,
                      "job_id": os.environ.get("SLURM_JOB_ID"), "hostname": socket.gethostname(),
                      "environment_verified": bool(environment), "source": provenance}
        (out / "driver_environment.json").write_text(json.dumps(driver_env, indent=2) + "\n")
        if not (out / "env.txt").exists():
            (out / "env.txt").write_text(json.dumps(driver_env, indent=2) + "\n")
        env = dict(os.environ)
        command(["lscpu"], out, "lscpu", env, 15)
        tools = build_tools(args, out, env)
        all_ok = True
        with (out / "raw.jsonl").open("x") as raw, (out / "summary.csv").open("x", newline="") as csv_stream:
            writer = csv.DictWriter(csv_stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            inputs = {}
            for n in args.sizes:
                dump_cmd = [tools["dumper"], str(n), "dump", out / f"input_n{n}.bin"]
                if plan:
                    dump_cmd = [*command_prefix(plan, cpus[:1]), *dump_cmd]
                result = command(dump_cmd, out, f"dump_n{n}", env, args.timeout)
                if result["exit_code"]:
                    raise RuntimeError(f"input dump failed for n={n}; see dump logs")
                dump = json.loads(Path(result["stdout"]).read_text())
                matrix, guess, checksum = common.load_inputs(n, out / f"input_n{n}.bin")
                if checksum != dump["input_checksum"]:
                    raise ValueError(f"C++/Python input checksum mismatch for n={n}")
                reference_start = time.perf_counter()
                with threadpoolctl.threadpool_limits(limits=1):
                    reference = np.linalg.eigvalsh(matrix)[:4]
                reference_elapsed = time.perf_counter() - reference_start
                if not np.isfinite(reference).all():
                    raise ValueError(f"nonfinite eigvalsh reference for n={n}")
                reference.astype("<f8").tofile(out / f"reference_n{n}.bin")
                reference_info = {"n": n, "input_checksum": checksum, "eigenvalues": reference.tolist(),
                                  "method": "numpy.linalg.eigvalsh(A)[:4]", "blas_threads": 1,
                                  "elapsed_s": reference_elapsed,
                                  "numpy": np.__version__, "input_sha256": hashlib.sha256((out / f"input_n{n}.bin").read_bytes()).hexdigest()}
                (out / f"reference_n{n}.json").write_text(json.dumps(reference_info, indent=2) + "\n")
                inputs[n] = (checksum, reference)
                del matrix, guess
            for sweep in range(1, args.sweeps + 1):
                for n in args.sizes:
                    checksum, reference = inputs[n]
                    for solver in args.solvers:
                        for cores in ([1] if solver == "cpp_serial" else args.cores):
                            passed = run_configuration(args, out, raw, writer, csv_stream, tools, n, solver,
                                                       cpus[:cores], sweep, checksum, reference, provenance,
                                                       environment, env, common, np)
                            all_ok = passed and all_ok
        manifest.update(status="complete", ok=all_ok)
        print("OVERALL " + ("PASS" if all_ok else "FAIL"), flush=True)
        return 0 if all_ok else 1
    except Exception as exc:
        manifest.update(status="failed", ok=False, error=f"{type(exc).__name__}: {exc}")
        (out / "driver_error.json").write_text(json.dumps({"record_type": "error", "error": manifest["error"], "ok": False}) + "\n")
        print(manifest["error"], file=sys.stderr, flush=True)
        return 1
    finally:
        with manifest_path.open("w") as stream:
            common.write_json(stream, manifest)


if __name__ == "__main__":
    sys.exit(main())
