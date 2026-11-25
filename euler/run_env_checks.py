"""E2 only: verify the environment and one/two-rank GD on one allocated node."""

import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys


def physical_cpus():
    """Choose one allowed logical CPU per physical socket/core, in stable order."""
    groups = {}
    allowed = sorted(os.sched_getaffinity(0))
    for cpu in allowed:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = tuple(int((topology / name).read_text())
                    for name in ("physical_package_id", "core_id"))
        groups.setdefault(key, []).append(cpu)
    return allowed, [{"socket": key[0], "core": key[1], "cpu": min(cpus),
                      "allowed_siblings": cpus} for key, cpus in sorted(groups.items())]


def capture(command, label, out, env, timeout=50):
    """Retain both streams and clean up the local process group on timeout."""
    print(f"\n== {label}\n+ {shlex.join(map(str, command))}", flush=True)
    result = {"label": label, "command": list(map(str, command)), "timeout_s": timeout}
    stdout_path, stderr_path = out / f"{label}.out", out / f"{label}.err"
    try:
        with stdout_path.open("x") as stdout, stderr_path.open("x") as stderr:
            proc = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr,
                                    start_new_session=True)
            try:
                result["exit_code"] = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                result.update(exit_code=124, error="Timed out; terminated local process group")
    except OSError as exc:
        result.update(exit_code=127, error=str(exc))
    for path in (stdout_path, stderr_path):
        if path.exists():
            print(path.read_text(errors="replace"), end="", flush=True)
    result.update(stdout=str(stdout_path.name), stderr=str(stderr_path.name))
    print(f"{label}: exit {result['exit_code']}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "checks.json").exists():
        parser.error("output already contains checks.json; use a fresh job directory")
    repo = Path(__file__).resolve().parent.parent
    prefix = Path(sys.prefix)
    python = prefix / "bin/python"
    mpiexec = prefix / "bin/mpiexec"
    summary = {"stage": "E2", "hostname": socket.gethostname(),
               "job_id": os.environ.get("SLURM_JOB_ID"), "prefix": str(prefix),
               "checks": [], "ok": False}
    try:
        if int(os.environ.get("SLURM_JOB_NUM_NODES", "1")) != 1:
            raise ValueError("E2's fork launcher requires exactly one allocated node")
        allowed, physical = physical_cpus()
        if len(physical) < 2:
            raise ValueError(f"Need two physical cores; allowed CPUs {allowed} provide {len(physical)}")
        cpus = [entry["cpu"] for entry in physical[:2]]
        summary["cpu_mapping"] = {"allowed_logical_cpus": allowed,
                                  "physical_cores": physical, "selected": cpus}
        (out / "cpu_mapping.json").write_text(json.dumps(summary["cpu_mapping"], indent=2) + "\n")
        print("Physical-core selection:", json.dumps(summary["cpu_mapping"]), flush=True)
        env = dict(os.environ)
        for key in ("OMP_PROC_BIND", "OMP_PLACES", "GOMP_CPU_AFFINITY", "KMP_AFFINITY",
                    "HYDRA_HOST_FILE", "HYDRA_BINDING", "HYDRA_MAPPING", "HYDRA_MEMBIND",
                    "HYDRA_TOPOLIB", "HYDRA_TOPO_DEBUG", "HYDRA_LAUNCHER_EXTRA_ARGS"):
            env.pop(key, None)
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[key] = "1"
        env.update(OMP_DYNAMIC="FALSE", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
                   MPIR_CVAR_ENABLE_GPU="0", MPIEXEC_TIMEOUT="45")
        for label, command in (
            ("mpiexec_info", [str(mpiexec), "-info"]),
            ("mpiexec_binding_help", [str(mpiexec), "-bind-to", "-help"]),
            ("environment", ["taskset", "--cpu-list", str(cpus[0]), str(python), "-B",
                             str(repo / "euler/inspect_env.py"), "--output-dir", str(out)]),
        ):
            timeout = 90 if label == "environment" else 10
            summary["checks"].append(capture(command, label, out, env, timeout=timeout))
        checker = str(repo / "euler/check_numerics.py")
        for solver, ranks in (("pyscf", 1), ("slepc", 1), ("slepc", 2)):
            selected_text = ",".join(map(str, cpus[:ranks]))
            child_env = {**env, "BASELINES_EXPECTED_CPUS": selected_text}
            command = [str(python), "-B", checker, "--solver", solver, "--size", "100",
                       "--output-dir", str(out)]
            if solver == "slepc":
                command = [str(mpiexec), "-launcher", "fork", "-hosts", "localhost",
                           "-bind-to", f"user:{selected_text}", "-n", str(ranks), *command]
            else:
                command = ["taskset", "--cpu-list", selected_text, *command]
            summary["checks"].append(capture(command, f"{solver}_{ranks}rank", out, child_env))
        summary["ok"] = all(row["exit_code"] == 0 for row in summary["checks"])
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        print(f"E2 ERROR: {summary['error']}", file=sys.stderr, flush=True)
    finally:
        (out / "checks.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("E2 Python checks:", "PASS" if summary["ok"] else "FAIL", flush=True)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
