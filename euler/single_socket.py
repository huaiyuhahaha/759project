#!/usr/bin/env python3
"""Fail-closed, standard-library placement gate for the E6 baseline job.

The plan uses sysfs physical core IDs. lscpu renumbers cores on some machines
(including the recorded E4 node), so its core *groups*, rather than numeric
core IDs, must agree with sysfs. All checks happen outside solver timers.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys


SYSFS = Path("/sys/devices/system/cpu")


def cpu_list(value):
    """Parse Linux CPU-list notation, rejecting duplicate or invalid IDs."""
    result = []
    for part in value.strip().split(","):
        if not re.fullmatch(r"\d+(?:-\d+)?", part):
            raise RuntimeError(f"Invalid CPU list: {value!r}")
        limits = [int(item) for item in part.split("-")]
        start, end = limits[0], limits[-1]
        if end < start:
            raise RuntimeError(f"Reversed CPU range: {part}")
        result.extend(range(start, end + 1))
    if len(result) != len(set(result)):
        raise RuntimeError(f"Duplicate CPU IDs in {value!r}")
    return sorted(result)


def inspect_topology(allowed=None, sysfs_root=SYSFS, lscpu_output=None):
    """Read and cross-check the allocation; arguments permit synthetic tests."""
    allowed = sorted(os.sched_getaffinity(0) if allowed is None else allowed)
    if not allowed or len(allowed) != len(set(allowed)):
        raise RuntimeError("The allowed CPU set is empty or contains duplicates")
    command = ["lscpu", "--parse=CPU,CORE,SOCKET,NODE"]
    if lscpu_output is None:
        lscpu_output = subprocess.check_output(command, text=True)
    parsed = {}
    for line in lscpu_output.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        columns = line.strip().split(",")
        if len(columns) != 4 or not columns[0].isdigit():
            raise RuntimeError(f"Invalid lscpu row: {line!r}")
        cpu = int(columns[0])
        if cpu in parsed:
            raise RuntimeError(f"Duplicate lscpu row for CPU {cpu}")
        parsed[cpu] = columns[1:]
    evidence, groups = [], {}
    sysfs_to_lscpu, lscpu_to_sysfs = {}, {}
    for cpu in allowed:
        if cpu not in parsed or any(not item.isdigit() for item in parsed[cpu]):
            raise RuntimeError(f"lscpu has no complete core/socket/NUMA mapping for CPU {cpu}")
        lc_core, lc_socket, lc_node = map(int, parsed[cpu])
        cpu_path = Path(sysfs_root) / f"cpu{cpu}"
        topology = cpu_path / "topology"
        package = int((topology / "physical_package_id").read_text().strip())
        core = int((topology / "core_id").read_text().strip())
        siblings = cpu_list((topology / "thread_siblings_list").read_text())
        nodes = [int(item.name[4:]) for item in cpu_path.glob("node[0-9]*")
                 if re.fullmatch(r"node\d+", item.name)]
        if package < 0 or core < 0 or len(nodes) != 1 or cpu not in siblings:
            raise RuntimeError(f"Incomplete/ambiguous sysfs topology for CPU {cpu}")
        node = nodes[0]
        if (package, node) != (lc_socket, lc_node):
            raise RuntimeError(f"sysfs/lscpu socket or NUMA mismatch for CPU {cpu}")
        key, lc_key = (package, core), (lc_socket, lc_core)
        if sysfs_to_lscpu.setdefault(key, lc_key) != lc_key:
            raise RuntimeError(f"lscpu splits sysfs physical core {key}")
        if lscpu_to_sysfs.setdefault(lc_key, key) != key:
            raise RuntimeError(f"lscpu merges distinct sysfs physical cores at CPU {cpu}")
        groups.setdefault(key, []).append(cpu)
        evidence.append({"cpu": cpu, "socket": package, "core": core,
                         "numa_node": node, "thread_siblings": siblings,
                         "lscpu_core": lc_core})
    by_cpu = {item["cpu"]: item for item in evidence}
    physical = []
    allowed_set = set(allowed)
    for (package, core), cpus in sorted(groups.items()):
        first = by_cpu[min(cpus)]
        for cpu in cpus:
            entry = by_cpu[cpu]
            if (entry["thread_siblings"] != first["thread_siblings"] or
                    set(entry["thread_siblings"]) & allowed_set != set(cpus) or
                    entry["numa_node"] != first["numa_node"]):
                raise RuntimeError(f"Inconsistent SMT siblings or NUMA mapping for CPU {cpu}")
        physical.append({"socket": package, "core": core, "cpu": min(cpus),
                         "allowed_siblings": sorted(cpus), "numa_node": first["numa_node"]})
    return {"allowed_logical_cpus": allowed, "physical_cores": physical,
            "verification": {"sysfs_root": str(sysfs_root), "lscpu_command": command,
                             "lscpu_csv": lscpu_output, "cpu_evidence": evidence,
                             "core_comparison": "physical-core grouping; IDs may be renumbered"}}


def choose_cores(topology, requiredcores):
    """Choose the largest socket first, then its largest single NUMA domain."""
    if requiredcores < 1:
        raise RuntimeError("The required physical core count must be positive")
    physical = topology["physical_cores"]
    counts = {}
    for item in physical:
        counts[item["socket"]] = counts.get(item["socket"], 0) + 1
    if not counts:
        raise RuntimeError("No allocated physical cores found")
    package = min(counts, key=lambda key: (-counts[key], key))
    node_counts = {}
    for item in physical:
        if item["socket"] == package:
            node_counts[item["numa_node"]] = node_counts.get(item["numa_node"], 0) + 1
    node = min(node_counts, key=lambda key: (-node_counts[key], key))
    if node_counts[node] < requiredcores:
        raise RuntimeError(
            f"Socket {package} has the most allocated physical cores ({counts[package]}), "
            f"but its largest NUMA node {node} has only {node_counts[node]}; "
            f"need {requiredcores} physical cores on one socket and NUMA node")
    selected = sorted(item["cpu"] for item in physical
                      if item["socket"] == package and item["numa_node"] == node)[:requiredcores]
    return {"socket": package, "numa_node": node, "selected": selected,
            "expectedcores": requiredcores,
            "socket_physical_core_counts": {str(key): value for key, value in sorted(counts.items())},
            "selected_socket_numa_physical_core_counts": {
                str(key): value for key, value in sorted(node_counts.items())},
            "selection_rule": "most allocated physical cores per socket, then per NUMA node; "
                              "ties use lowest ID; ascending representative CPUs"}


def _recorded_tools(plan):
    taskset = plan["taskset"]
    if not Path(taskset).is_file() or not os.access(taskset, os.X_OK):
        raise RuntimeError(f"Recorded taskset is unavailable: {taskset}")
    policy = plan["memory_policy"]
    found = shutil.which("numactl")
    if policy["mode"] == "numactl_bind":
        if not found or os.path.realpath(found) != os.path.realpath(policy["numactl"]):
            raise RuntimeError("numactl availability/path changed since placement preparation")
    elif policy["mode"] == "pinned_first_touch":
        if found:
            raise RuntimeError("numactl became available after choosing pinned first-touch; prepare again")
    else:
        raise RuntimeError(f"Unknown memory policy: {policy['mode']}")


def validate_selected(cpus, plan):
    """Recheck the launch's CPU subset against current affinity/sysfs/lscpu."""
    cpus = list(cpus)
    selected = plan["selected"]
    if len(selected) != plan["expectedcores"] or len(selected) != len(set(selected)):
        raise RuntimeError("The placement plan has invalid/duplicate selected CPUs")
    if not cpus or len(cpus) != len(set(cpus)) or not set(cpus) <= set(selected):
        raise RuntimeError("Launch CPUs must be distinct members of the selected CPU set")
    if plan["host"] != socket.gethostname() or plan["slurm_job_id"] != os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("The placement plan belongs to a different host or Slurm job")
    current = set(os.sched_getaffinity(0))
    if not set(cpus) <= current:
        raise RuntimeError(f"Launch CPUs {cpus} are outside current affinity {sorted(current)}")
    if not current <= set(plan["allowed_logical_cpus"]):
        raise RuntimeError("Current affinity exceeds the allocation recorded in the placement plan")
    topology = inspect_topology()
    recorded = {item["cpu"]: item for item in plan["verification"]["cpu_evidence"]}
    live = {item["cpu"]: item for item in topology["verification"]["cpu_evidence"]}
    physical = set()
    for cpu in cpus:
        item = live[cpu]
        if cpu not in recorded or item != recorded[cpu]:
            raise RuntimeError(f"Topology changed since preparation for CPU {cpu}")
        if (item["socket"], item["numa_node"]) != (plan["socket"], plan["numa_node"]):
            raise RuntimeError(f"CPU {cpu} is outside the chosen socket/NUMA node")
        key = (item["socket"], item["core"])
        if key in physical:
            raise RuntimeError("Launch CPUs include SMT siblings of the same physical core")
        physical.add(key)
    _recorded_tools(plan)
    return {"cpus": cpus, "socket": plan["socket"], "numa_node": plan["numa_node"],
            "physical_cores": len(physical), "memory_policy": plan["memory_policy"]["mode"]}


def load_verified_plan(path, requiredcores=None, cpus=None):
    plan = json.loads(Path(path).read_text())
    if plan.get("schema_version") != 1 or not plan.get("verified"):
        raise RuntimeError("Unsupported or unverified single-socket placement plan")
    if requiredcores is not None and plan["expectedcores"] != requiredcores:
        raise RuntimeError(f"Expected {requiredcores} selected cores, got {plan['expectedcores']}")
    validate_selected(plan["selected"] if cpus is None else cpus, plan)
    plan["_plan_path"] = str(Path(path).resolve())
    return plan


def _verify_memory_policy(plan):
    command = [plan["memory_policy"]["numactl"], "--show"]
    result = subprocess.run(command, text=True, capture_output=True)
    fields = {line.split(":", 1)[0].strip(): line.split(":", 1)[1].strip()
              for line in result.stdout.splitlines() if ":" in line}
    node = str(plan["numa_node"])
    if (result.returncode != 0 or fields.get("policy") != "bind" or
            fields.get("membind") != node or fields.get("cpubind") != node):
        raise RuntimeError(f"Cannot verify numactl CPU/memory binding: {result.stdout}\n{result.stderr}")
    return {"command": command, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def verify_runtime(plan, cpus=None):
    """Gate the driver itself before it allocates or first-touches matrices."""
    cpus = plan["selected"] if cpus is None else list(cpus)
    if os.environ.get("E6_SINGLE_SOCKET_PLAN") != plan.get("_plan_path"):
        raise RuntimeError("The E6 driver must be launched through single_socket.py exec")
    if set(os.sched_getaffinity(0)) != set(cpus):
        raise RuntimeError("Current affinity must equal the launch's selected physical CPUs")
    evidence = validate_selected(cpus, plan)
    if plan["memory_policy"]["mode"] == "numactl_bind":
        evidence["numactl_show"] = _verify_memory_policy(plan)
    return evidence


def command_prefix(plan, cpus):
    """Validate and return an explicit NUMA-policy/taskset command prefix."""
    validate_selected(cpus, plan)
    prefix = []
    if plan["memory_policy"]["mode"] == "numactl_bind":
        prefix = [plan["memory_policy"]["numactl"],
                  f"--cpunodebind={plan['numa_node']}", f"--membind={plan['numa_node']}"]
    return prefix + [plan["taskset"], "-c", ",".join(map(str, cpus))]


def prepare(output_dir, requiredcores):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "single_socket.json"
    if destination.exists():
        raise RuntimeError(f"Refusing to overwrite placement plan: {destination}")
    topology = inspect_topology()
    taskset = shutil.which("taskset")
    if not taskset:
        raise RuntimeError("taskset is required for explicit physical-CPU placement")
    numactl = shutil.which("numactl")
    plan = {"schema_version": 1, "verified": False,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "taskset": taskset, **topology, **choose_cores(topology, requiredcores),
            "memory_policy": {"mode": "numactl_bind" if numactl else "pinned_first_touch",
                              "numactl": numactl,
                              "description": "explicit CPU-node and memory-node binding" if numactl else
                              "numactl unavailable; matrix allocation/first touch runs on selected CPUs"}}
    # Probe both affinity and memory policy before any solver/input allocation.
    probe_code = "import json,os; print(json.dumps(sorted(os.sched_getaffinity(0))))"
    prefix = command_prefix(plan, plan["selected"])
    probe = subprocess.run(prefix + [sys.executable, "-c", probe_code], text=True, capture_output=True)
    plan["affinity_probe"] = {"command": prefix + [sys.executable, "-c", probe_code],
                              "returncode": probe.returncode, "stdout": probe.stdout,
                              "stderr": probe.stderr}
    if probe.returncode != 0 or json.loads(probe.stdout) != plan["selected"]:
        raise RuntimeError(f"Placement/NUMA binding probe failed: {probe.stderr or probe.stdout}")
    if numactl:
        command = prefix + [numactl, "--show"]
        result = subprocess.run(command, text=True, capture_output=True)
        plan["memory_policy"]["probe"] = {"command": command, "returncode": result.returncode,
                                          "stdout": result.stdout, "stderr": result.stderr}
        fields = {line.split(":", 1)[0].strip(): line.split(":", 1)[1].strip()
                  for line in result.stdout.splitlines() if ":" in line}
        if (result.returncode != 0 or fields.get("policy") != "bind" or
                fields.get("membind") != str(plan["numa_node"]) or
                fields.get("cpubind") != str(plan["numa_node"])):
            raise RuntimeError(f"Cannot verify numactl memory binding: {result.stdout}\n{result.stderr}")
    validate_selected(plan["selected"], plan)
    plan["verified"] = True
    with destination.open("x") as output:
        json.dump(plan, output, indent=2)
        output.write("\n")
    print(f"Verified {requiredcores} distinct physical cores on socket {plan['socket']}, "
          f"NUMA node {plan['numa_node']}: {plan['selected']}; "
          f"memory policy: {plan['memory_policy']['mode']}", flush=True)
    print(destination, flush=True)
    return plan


def _mpi_rank(cores):
    rank = int(os.environ.get("PMI_RANK", "-1"))
    size = int(os.environ.get("PMI_SIZE", "-1"))
    if size != cores or not 0 <= rank < size:
        raise RuntimeError(f"Expected PMI_SIZE={cores} and a valid PMI_RANK; got size={size}, rank={rank}")
    return rank


def _exec_bound(plan, cpus, command, rank=None, cores=None, evidence_dir=None):
    """Exec, retaining Hydra's PMI descriptors across the binding trampoline."""
    prefix = command_prefix(plan, cpus)
    trampoline = [sys.executable, "-B", str(Path(__file__).resolve()), "_bound_exec",
                  "--plan", plan["_plan_path"], "--cpus", ",".join(map(str, cpus))]
    if rank is not None:
        trampoline += ["--rank", str(rank), "--cores", str(cores),
                       "--evidence-dir", str(evidence_dir)]
    wrapped = prefix + trampoline + ["--"] + command
    print(f"E6 launch: {wrapped}", flush=True)
    os.environ["E6_SINGLE_SOCKET_PLAN"] = plan["_plan_path"]
    os.execv(wrapped[0], wrapped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--cores", type=int, default=8)
    exec_parser = subparsers.add_parser("exec")
    exec_parser.add_argument("--plan", type=Path, required=True)
    exec_parser.add_argument("--first-cpu", action="store_true")
    exec_parser.add_argument("command", nargs=argparse.REMAINDER)
    rank_parser = subparsers.add_parser("exec-rank", help="Reapply memory binding after Hydra binds a rank")
    rank_parser.add_argument("--plan", type=Path, required=True)
    rank_parser.add_argument("--cores", type=int, required=True)
    rank_parser.add_argument("--evidence-dir", type=Path, required=True)
    rank_parser.add_argument("command", nargs=argparse.REMAINDER)
    bound_parser = subparsers.add_parser("_bound_exec", help="Internal post-binding verification trampoline")
    bound_parser.add_argument("--plan", type=Path, required=True)
    bound_parser.add_argument("--cpus", required=True)
    bound_parser.add_argument("--rank", type=int)
    bound_parser.add_argument("--cores", type=int)
    bound_parser.add_argument("--evidence-dir", type=Path)
    bound_parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            prepare(args.output_dir, args.cores)
        else:
            command = args.command[1:] if args.command[:1] == ["--"] else args.command
            if not command:
                raise RuntimeError("exec requires a command after --")
            if args.action == "exec":
                plan = load_verified_plan(args.plan)
                cpus = plan["selected"][:1] if args.first_cpu else plan["selected"]
                _exec_bound(plan, cpus, command)
            elif args.action == "exec-rank":
                rank = _mpi_rank(args.cores)
                selected = json.loads(args.plan.read_text())["selected"]
                if args.cores > len(selected):
                    raise RuntimeError("MPI size exceeds the selected physical core count")
                cpus = [selected[rank]]
                if set(os.sched_getaffinity(0)) != set(cpus):
                    raise RuntimeError(f"Hydra rank {rank} affinity is not expected singleton {cpus}")
                plan = load_verified_plan(args.plan, cpus=cpus)
                _exec_bound(plan, cpus, command, rank, args.cores, args.evidence_dir)
            else:
                cpus = cpu_list(args.cpus)
                plan = load_verified_plan(args.plan, cpus=cpus)
                evidence = verify_runtime(plan, cpus)
                if args.rank is not None:
                    if _mpi_rank(args.cores) != args.rank or cpus != [plan["selected"][args.rank]]:
                        raise RuntimeError("MPI rank identity/CPU changed across the binding trampoline")
                    evidence.update({"rank": args.rank, "mpi_size": args.cores,
                                     "host": plan["host"], "slurm_job_id": plan["slurm_job_id"],
                                     "plan_path": plan["_plan_path"], "verified_before_solver_exec": True})
                    args.evidence_dir.mkdir(parents=True, exist_ok=True)
                    with (args.evidence_dir / f"rank-{args.rank}.json").open("x") as output:
                        json.dump(evidence, output, indent=2)
                        output.write("\n")
                os.execvpe(command[0], command, os.environ)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"E6 PLACEMENT GATE FAILED: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
