"""Read and cross-check returned Euler evidence without modifying it.

Scientific failures remain rows. Contradictory or missing integrity evidence raises
ValueError so a report cannot quietly turn corrupted inputs into valid results.
Ordinary files and losslessly archived evidence use the same checks. Omitted
matrix dumps retain explicitly labeled packaging-time fingerprints; their bytes
are not represented as freshly rehashed. Only the standard library is required;
this module never runs a solver.
"""

import csv
import io
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import struct

from baseline_storage import (exists, input_fingerprint, open_text, read_bytes,
                              read_text, sha256, size)


INT_FIELDS = ("schema_version", "sweep", "n", "nroots", "cores", "ranks", "warmups",
              "runs", "min_runs", "max_runs", "exit_code", "omp_threads", "blas_threads")
BOOL_FIELDS = ("gate_pass", "strict_gate_pass", "all_converged", "minimum_time_met",
               "minimum_runs_met", "smoke", "profiled", "environment_verified")
FLOAT_FIELDS = ("min_time_s", "total_timed_s", "median_ms", "min_ms", "max_ms",
                "iterations_min", "iterations_median", "iterations_max", "matvecs_min",
                "matvecs_median", "matvecs_max", "median_ms_per_matvec",
                "max_eigenvalue_error", "max_residual", "max_orth_error", "residual_slack")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(read_text(path))


def close(a, b):
    if a is None or b is None:
        return a is b
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-18)


def numeric(values):
    return all(isinstance(v, (int, float)) and not isinstance(v, bool)
               and math.isfinite(v) for v in values)


def artifact(directory, manifest, value):
    """Map recorded Euler absolute paths into the returned job directory."""
    require(isinstance(value, str) and value, "empty artifact path")
    original = PurePosixPath(value)
    if original.is_absolute():
        try:
            original = original.relative_to(manifest["arguments"]["output_dir"])
        except ValueError as exc:
            raise ValueError(f"artifact escapes original job directory: {value}") from exc
    require(".." not in original.parts, f"artifact escapes job directory: {value}")
    path = directory.joinpath(*original.parts)
    require(exists(path) and size(path) > 0, f"missing/empty artifact: {path}")
    return path


def typed_row(row, line):
    row = dict(row, csv_line=line)
    for key in INT_FIELDS:
        row[key] = int(row[key])
    for key in FLOAT_FIELDS:
        row[key] = float(row[key]) if row.get(key) else None
        require(row[key] is None or math.isfinite(row[key]), f"nonfinite CSV {key}, line {line}")
    for key in BOOL_FIELDS:
        require(row.get(key) in ("True", "False"), f"invalid CSV boolean {key}, line {line}")
        row[key] = row[key] == "True"
    for key in ("socket", "numa_node"):
        if row.get(key):
            row[key] = int(row[key])
    return row


def check_memory_policy(evidence, node):
    """Verify saved numactl output; allowed memory nodes alone are not policy."""
    fields = {line.split(":", 1)[0].strip(): line.split(":", 1)[1].strip()
              for line in evidence.get("stdout", "").splitlines() if ":" in line}
    require(evidence.get("returncode") == 0 and fields.get("policy") == "bind"
            and fields.get("membind") == str(node) and fields.get("cpubind") == str(node),
            "NUMA binding probe does not verify the chosen node")


def check_single_socket(directory, manifest, mapping):
    """Cross-check saved E6 topology without querying this analysis machine."""
    plan_path = artifact(directory, manifest, manifest["arguments"]["single_socket_plan"])
    require(plan_path == directory / "single_socket.json", "unexpected E6 placement artifact")
    plan = read_json(plan_path)
    require(plan.get("verified") is True and plan.get("schema_version") == 1, "unverified E6 plan")
    require(all(mapping.get(key) == value for key, value in plan.items()), "plan/mapping mismatch")
    require(plan["host"] == manifest["hostname"] and str(plan["slurm_job_id"]) == str(manifest["job_id"]),
            "placement plan belongs to a different job/node")
    allowed, selected = plan["allowed_logical_cpus"], plan["selected"]
    require(len(set(allowed)) == len(allowed) and len(allowed) >= 8, "invalid allocated CPU set")
    require(plan["expectedcores"] == len(selected) == len(set(selected)) == 8,
            "E6 requires eight distinct selected CPUs")
    require(set(selected) <= set(allowed), "selected CPUs escape allocation")
    saved = plan["verification"]["cpu_evidence"]
    evidence = {entry["cpu"]: entry for entry in saved}
    require(len(evidence) == len(saved) and set(evidence) == set(allowed), "sysfs evidence coverage mismatch")
    lscpu = {}
    for line in plan["verification"]["lscpu_csv"].splitlines():
        if not line or line.startswith("#"):
            continue
        cells = line.split(",")
        require(len(cells) == 4 and all(value.isdigit() for value in cells), "invalid lscpu topology")
        cpu, core, socket, node = map(int, cells)
        require(cpu not in lscpu, "duplicate lscpu CPU")
        lscpu[cpu] = (core, socket, node)
    groups, forward, reverse = {}, {}, {}
    for cpu in allowed:
        item = evidence[cpu]
        require(cpu in lscpu, "allocated CPU missing from lscpu")
        lc_core, lc_socket, lc_node = lscpu[cpu]
        require((item["socket"], item["numa_node"]) == (lc_socket, lc_node), "sysfs/lscpu socket/NUMA mismatch")
        require(item["lscpu_core"] == lc_core, "saved lscpu core mismatch")
        key, lc_key = (item["socket"], item["core"]), (lc_socket, lc_core)
        require(forward.setdefault(key, lc_key) == lc_key and reverse.setdefault(lc_key, key) == key,
                "sysfs/lscpu physical-core group mismatch")
        groups.setdefault(key, []).append(cpu)
    physical = []
    for (socket, core), cpus in sorted(groups.items()):
        first = evidence[min(cpus)]
        for cpu in cpus:
            item = evidence[cpu]
            require(cpu in item["thread_siblings"] and item["thread_siblings"] == first["thread_siblings"]
                    and set(item["thread_siblings"]) & set(allowed) == set(cpus)
                    and item["numa_node"] == first["numa_node"], "inconsistent SMT/NUMA evidence")
        physical.append({"socket": socket, "core": core, "cpu": min(cpus),
                         "allowed_siblings": sorted(cpus), "numa_node": first["numa_node"]})
    require(physical == plan["physical_cores"], "physical core inventory contradicts sysfs")
    counts = {socket: sum(item["socket"] == socket for item in physical) for socket, _ in groups}
    socket = min(counts, key=lambda key: (-counts[key], key))
    nodes = {item["numa_node"] for item in physical if item["socket"] == socket}
    node_counts = {node: sum(item["socket"] == socket and item["numa_node"] == node for item in physical) for node in nodes}
    node = min(node_counts, key=lambda key: (-node_counts[key], key))
    wanted = sorted(item["cpu"] for item in physical if item["socket"] == socket and item["numa_node"] == node)[:8]
    require(node_counts[node] >= 8 and plan["socket"] == socket and plan["numa_node"] == node
            and selected == wanted, "selected CPUs violate largest-socket/single-NUMA rule")
    require(plan["socket_physical_core_counts"] == {str(k): v for k, v in counts.items()}
            and plan["selected_socket_numa_physical_core_counts"] == {str(k): v for k, v in node_counts.items()},
            "recorded allocation counts contradict topology")
    probe = plan["affinity_probe"]
    require(probe["returncode"] == 0 and json.loads(probe["stdout"]) == selected, "affinity probe failed")
    policy = plan["memory_policy"]
    require(policy["mode"] in ("numactl_bind", "pinned_first_touch"), "unknown memory policy")
    runtime = mapping["driver_runtime"]
    for key, value in (("cpus", selected), ("socket", socket), ("numa_node", node),
                       ("physical_cores", 8), ("memory_policy", policy["mode"])):
        require(runtime.get(key) == value, f"driver runtime placement mismatch: {key}")
    if policy["mode"] == "numactl_bind":
        require(policy.get("numactl"), "missing numactl executable")
        check_memory_policy(policy["probe"], node)
        check_memory_policy(runtime["numactl_show"], node)
    else:
        require(policy.get("numactl") is None, "first-touch fallback despite recorded numactl")
    hardware = read_json(directory / "hardware.json")
    require(hardware["hostname"] == manifest["hostname"] and str(hardware["job_id"]) == str(manifest["job_id"]),
            "hardware belongs to a different job/node")
    for n in manifest["arguments"]["sizes"]:
        dump = read_json(directory / f"dump_n{n}.command.json")
        prefix = [plan["taskset"], "-c", str(selected[0])]
        if policy["mode"] == "numactl_bind":
            prefix = [policy["numactl"], f"--cpunodebind={node}", f"--membind={node}"] + prefix
        require(dump["exit_code"] == 0 and dump["command"][:len(prefix)] == prefix,
                "input dump was not pinned to the first selected CPU")
    return plan


def check_placement_record(record, row):
    for key in ("placement_policy", "socket", "numa_node", "memory_policy", "placement_path"):
        require(record.get(key) == row[key], f"{row['config_id']}: placement metadata mismatch: {key}")
    require(record.get("cpu_mapping") == json.loads(row["cpu_mapping_json"]), "placement CPU metadata mismatch")


def check_configuration_placement(directory, manifest, plan, row, warnings):
    config = directory / "configs" / row["config_id"]
    cpus = plan["selected"][:row["cores"]]
    expected = {"placement_policy": "single_socket_single_numa_node", "socket": plan["socket"],
                "numa_node": plan["numa_node"], "memory_policy": plan["memory_policy"]["mode"],
                "placement_path": "single_socket.json"}
    require(all(row[key] == value for key, value in expected.items()), "CSV placement contradicts verified plan")
    check_placement_record(read_json(config / "placement.json"), row)
    summary = read_json(config / "summary.json")
    check_placement_record(summary, row)
    for key in (*INT_FIELDS, *FLOAT_FIELDS, *BOOL_FIELDS):
        require(close(summary.get(key), row[key]) if key in FLOAT_FIELDS else summary.get(key) == row[key],
                f"{row['config_id']}: saved summary/CSV mismatch: {key}")
    for key in ("settings", "versions", "resources", "sample_ids", "rank_placement_paths"):
        require(summary.get(key) == json.loads(row[key + "_json"]), f"saved summary/CSV {key} mismatch")
    command = read_json(config / "runner.command.json")
    require(command["exit_code"] == row["exit_code"], "runner/CSV exit mismatch")
    prefix = [plan["taskset"], "-c", ",".join(map(str, cpus))]
    if plan["memory_policy"]["mode"] == "numactl_bind":
        prefix = [plan["memory_policy"]["numactl"], f"--cpunodebind={plan['numa_node']}",
                  f"--membind={plan['numa_node']}"] + prefix
    require(command["command"][:len(prefix)] == prefix, "solver launch lacks verified placement prefix")
    rank_paths = json.loads(row["rank_placement_paths_json"])
    if row["solver"] != "slepc":
        require(not rank_paths, "non-MPI configuration has rank placement records")
        return
    wanted = [f"configs/{row['config_id']}/rank_placement/rank-{rank}.json" for rank in range(row["cores"])]
    require(rank_paths == wanted, "rank evidence path coverage mismatch")
    args = command["command"]
    mpi = [manifest["arguments"]["mpiexec"], "-launcher", "fork", "-hosts", "localhost", "-bind-to",
           "user:" + ",".join(map(str, cpus)), "-n", str(row["cores"])]
    require(args[len(prefix):len(prefix) + len(mpi)] == mpi, "MPI rank binding changed")
    require("exec-rank" in args and "--plan" in args and args[args.index("--plan") + 1] == manifest["arguments"]["single_socket_plan"],
            "MPI launch lacks post-Hydra placement check")
    for rank, value in enumerate(rank_paths):
        try:
            evidence = read_json(artifact(directory, manifest, value))
        except (OSError, ValueError):
            require(not row["gate_pass"], "passing MPI configuration lacks rank placement evidence")
            warnings.append(f"{row['config_id']}: failed rank {rank} has no placement evidence")
            continue
        checks = {"rank": rank, "mpi_size": row["cores"], "cpus": [cpus[rank]], "physical_cores": 1,
                  "socket": plan["socket"], "numa_node": plan["numa_node"],
                  "memory_policy": plan["memory_policy"]["mode"], "host": manifest["hostname"],
                  "slurm_job_id": manifest["job_id"], "verified_before_solver_exec": True,
                  "plan_path": manifest["arguments"]["single_socket_plan"]}
        require(all(evidence.get(key) == value for key, value in checks.items()), "rank placement contradicts plan/identity")
        if plan["memory_policy"]["mode"] == "numactl_bind":
            check_memory_policy(evidence["numactl_show"], plan["numa_node"])


def check_resources(sample, cpus, providers):
    resources = sample.get("resources") or {}
    if not sample.get("gate_pass"):
        return
    require(resources.get("ok") is True, "passing sample has failed resources")
    solver = sample["solver"]
    if solver.startswith("cpp_"):
        require(resources["selected_cpus"] == cpus, "C++ CPU selection mismatch")
        masks = [r["cpus"] for r in resources["thread_affinities"]]
        require(sorted(masks) == sorted([[cpu] for cpu in cpus]), "C++ team affinity mismatch")
        return
    ranks = resources["ranks"] if solver == "slepc" else [resources]
    require(len(ranks) == (len(cpus) if solver == "slepc" else 1), "rank resource count mismatch")
    for rank, item in enumerate(ranks):
        expected = [cpus[rank]] if solver == "slepc" else cpus
        threads = 1 if solver == "slepc" else len(cpus)
        require(item["ok"] and item["affinity"] == expected, "Python affinity mismatch")
        pools = item["threadpools"]
        require(all(pool["num_threads"] == threads for pool in pools), "thread-pool mismatch")
        loaded = {pool["filepath"] for pool in pools if pool["user_api"] == "blas"}
        require(loaded == providers, "sample BLAS provider differs from inspected environment")
        for snapshot in [item] + ([item["before_solve"]] if "before_solve" in item else []):
            require(snapshot.get("ok") and snapshot.get("affinity") == expected,
                    "before/after solve affinity verification failed")
            masks = snapshot.get("thread_cpu_affinity", {})
            require(masks, "missing operating-system thread affinity evidence")
            for mask, count in masks.items():
                actual = set()
                for part in mask.split(","):
                    bounds = list(map(int, part.split("-")))
                    require(len(bounds) in (1, 2) and bounds[0] <= bounds[-1], "invalid CPU range")
                    actual.update(range(bounds[0], bounds[-1] + 1))
                require(count > 0 and actual and actual <= set(expected),
                        "an operating-system thread escapes the selected cores")


def check_sample(sample, row, reference, cpus, providers, index):
    label = f"{row['config_id']} sample {index}"
    for key in ("solver", "n", "cores", "input_checksum", "hostname", "job_id", "config_id"):
        require(str(sample.get(key)) == str(row[key]), f"{label}: {key} mismatch")
    require(sample["sample_index"] == index, f"{label}: sample order mismatch")
    require(sample["phase"] == ("warmup" if index == 0 else "measured"), f"{label}: phase mismatch")
    require(numeric([sample["elapsed_s"]]) and sample["elapsed_s"] >= 0, f"{label}: bad time")
    values, residuals = sample.get("eigenvalues", []), sample.get("residuals", [])
    finite_pairs = len(values) == len(residuals) == 4 and numeric(values + residuals)
    if finite_pairs:
        errors = [abs(a - b) for a, b in zip(values, reference)]
        require(close(sample["max_eigenvalue_error"], max(errors)), f"{label}: eigenvalue aggregate mismatch")
        require(close(sample["max_residual"], max(residuals)), f"{label}: residual aggregate mismatch")
        strict = (sample.get("converged") and max(errors) < 1e-8
                  and all(0 <= value < 1e-8 for value in residuals)
                  and numeric([sample.get("orth_error")]) and 0 <= sample["orth_error"] < 1e-8)
    else:
        strict = False
    require(not sample["gate_pass"] or strict, f"{label}: passing flag contradicts accuracy")
    if sample["gate_pass"] and row["solver"] == "slepc":
        for key in ("configuration_gate", "settings_preserved", "real_eigenvalue_gate", "input_unchanged"):
            require(sample.get(key) is True, f"{label}: failed {key}")
        require(all(sample["operator_unchanged_by_rank"]), f"{label}: changed distributed operator")
    if sample["gate_pass"] and row["solver"] == "pyscf":
        for key in ("initial_vectors_are_independent_copies", "initial_vectors_match_input", "input_unchanged"):
            require(sample["lifecycle"].get(key) is True, f"{label}: failed {key}")
    check_resources(sample, cpus, providers)


def check_aggregates(row, samples):
    label = row["config_id"]
    measured = [sample for sample in samples if sample["phase"] == "measured"]
    times = [sample["elapsed_s"] for sample in measured]
    require(row["runs"] == len(measured) and row["warmups"] == len(samples) - len(measured),
            f"{label}: run count mismatch")
    require(row["warmups"] in (0, 1), f"{label}: invalid warm-up count")
    wanted = {"total_timed_s": sum(times), "median_ms": statistics.median(times) * 1000 if times else None,
              "min_ms": min(times) * 1000 if times else None, "max_ms": max(times) * 1000 if times else None}
    for key, field in (("max_eigenvalue_error", "max_eigenvalue_error"),
                       ("max_residual", "max_residual"), ("orth_error", "max_orth_error")):
        values = [sample.get(key) for sample in samples]
        wanted[field] = max(values) if values and numeric(values) else None
    for key in ("iterations", "matvecs"):
        values = [sample.get(key) for sample in measured]
        valid = values and numeric(values)
        for operation, suffix in ((min, "min"), (statistics.median, "median"), (max, "max")):
            wanted[f"{key}_{suffix}"] = operation(values) if valid else None
    ratios = [sample["elapsed_s"] * 1000 / sample["matvecs"] for sample in measured
              if numeric([sample.get("matvecs")]) and sample["matvecs"] > 0]
    wanted["median_ms_per_matvec"] = statistics.median(ratios) if ratios and len(ratios) == len(times) else None
    for key, value in wanted.items():
        require(close(row[key], value), f"{label}: CSV {key} contradicts raw samples")
    require(row["all_converged"] == (bool(samples) and all(s.get("converged") for s in samples)),
            f"{label}: convergence aggregate mismatch")
    require(row["minimum_time_met"] == (sum(times) >= row["min_time_s"]), f"{label}: time target mismatch")
    require(row["minimum_runs_met"] == (len(times) >= row["min_runs"]), f"{label}: count target mismatch")
    require(not row["gate_pass"] or (samples and all(s["gate_pass"] for s in samples)
                                    and row["strict_gate_pass"] and row["exit_code"] == 0),
            f"{label}: passing configuration contradicts samples/exit")
    if row["gate_pass"]:
        require(row["warmups"] == 1 and 1 <= len(times) <= row["max_runs"],
                f"{label}: passing configuration violates warm-up/run limits")
        targets_met = len(times) >= row["min_runs"] and sum(times) >= row["min_time_s"]
        require(targets_met or len(times) == row["max_runs"],
                f"{label}: stopped before repeat targets or cap")
        expected_stop = "targets_met" if targets_met else "max_runs"
        require(row["stop_reason"] == expected_stop, f"{label}: repeat stop reason mismatch")
        require(not (len(times) - 1 >= row["min_runs"] and sum(times[:-1]) >= row["min_time_s"]),
                f"{label}: continued after repeat targets were already met")


def load_job(path, step):
    """Return typed CSV rows and reproducible audit facts for E4/E5/E6."""
    require(step in ("E4", "E5", "E6"), "only E4, E5 and E6 are report inputs")
    path = Path(path).resolve()
    directory = path.parent if path.name == "summary.csv" else path
    manifest = read_json(directory / "run.json")
    environment = read_json(directory / "environment.json")
    mapping = read_json(directory / "cpu_mapping.json")
    args = manifest["arguments"]
    require(args["step"] == step, f"expected {step}, got {args['step']}")
    require(not args["smoke"], "smoke runs cannot supply production/profile evidence")
    plan = None
    if step == "E6":
        require(args["sizes"] == [64, 128, 256, 512, 1024, 2048, 4096]
                and args["cores"] == [1, 2, 4, 8] and args["sweeps"] == 2
                and args["solvers"] == ["cpp_serial", "cpp_openmp", "pyscf", "slepc"],
                "E6 experiment differs from required size/core/sweep grid")
        require(args["min_runs"] == 5 and args["min_time"] == 1 and args["max_runs"] == 200
                and not args["log_view"] and not args["profile_pyscf"], "E6 repeat/profile policy changed")
        plan = check_single_socket(directory, manifest, mapping)
        require(environment["cpu_affinity"] == plan["selected"][:1], "E6 inspection was not pinned")
    require(environment["ok"] and environment["blas_alignment"]["status"] == "verified_shared_loaded_dependency",
            "environment/shared BLAS inspection did not pass")
    for key in ("job_id", "hostname"):
        require(str(environment[key]) == str(manifest[key]), f"environment {key} mismatch")
    with io.StringIO(read_text(directory / "summary.csv"), newline="") as stream:
        rows = [typed_row(row, line) for line, row in enumerate(csv.DictReader(stream), 2)]
    by_id, references, checksums, warnings = {}, {}, {}, []
    fingerprints = {}
    providers = set(environment["blas_alignment"]["openblas_paths"])
    for n in args["sizes"]:
        info = read_json(directory / f"reference_n{n}.json")
        require(info["n"] == n and len(info["eigenvalues"]) == 4, "invalid reference metadata")
        fingerprint = input_fingerprint(directory, n)
        require(fingerprint["sha256"] == info["input_sha256"], f"input SHA mismatch at n={n}")
        require(fingerprint["size"] == 8 * (n * n + 4 * n), "input length mismatch")
        require(fingerprint["validation"] in ("bytes", "packaging_manifest"), "unknown input validation basis")
        fingerprints[str(n)] = fingerprint
        values = list(struct.unpack("<4d", read_bytes(directory / f"reference_n{n}.bin")))
        require(values == info["eigenvalues"] and numeric(values), "reference binary/JSON mismatch")
        references[n], checksums[str(n)] = values, info["input_checksum"]
    for row in rows:
        label = row["config_id"]
        require(label not in by_id, f"duplicate configuration {label}")
        by_id[label] = row
        for key in ("job_id", "hostname"):
            require(str(row[key]) == str(manifest[key]), f"CSV {key} mismatch")
        require(row["step"] == step and not row["smoke"] and row["environment_verified"], "CSV evidence type mismatch")
        require(step == "E5" or not row["profiled"], "profiled timing cannot enter production results")
        require(row["min_runs"] == args["min_runs"] and row["max_runs"] == args["max_runs"]
                and close(row["min_time_s"], args["min_time"]), f"{label}: repeat policy differs from manifest")
        require(row["input_checksum"] == checksums[str(row["n"])], "configuration input checksum mismatch")
        cpus = mapping["selected"][:row["cores"]]
        require(json.loads(row["cpu_mapping_json"]) == cpus, "configuration CPU selection mismatch")
        if plan:
            check_configuration_placement(directory, manifest, plan, row, warnings)
        try:
            raw_path = artifact(directory, manifest, row["raw_path"])
            records = [json.loads(line) for line in read_text(raw_path).splitlines()]
        except ValueError:
            require(not row["gate_pass"] and row["runs"] == row["warmups"] == 0,
                    f"{label}: missing samples for a nonempty/passing configuration")
            records = []
            warnings.append(f"{label}: failed before producing local samples")
        samples = [record for record in records if record.get("record_type") == "sample"]
        row["samples"] = samples
        for index, sample in enumerate(samples):
            require(step == "E5" or not sample.get("profiled", False), "profiled sample cannot enter production results")
            check_sample(sample, row, references[row["n"]], cpus, providers, index)
            if sample.get("eps_view_path"):
                artifact(directory, manifest, sample["eps_view_path"])
        check_aggregates(row, samples)
        if row["solver"] == "slepc" and row["gate_pass"]:
            require(all(sample.get("eps_view_path") for sample in samples), "missing mandatory EPS view")
        for value in json.loads(row["eps_view_paths_json"]):
            artifact(directory, manifest, value)
        for value in json.loads(row["profile_paths_json"]):
            try:
                artifact(directory, manifest, value)
            except ValueError:
                require(not row["gate_pass"], f"{label}: passing row has a missing requested profile")
                warnings.append(f"{label}: failed configuration has missing profile {value}")
        if step == "E5" and row["gate_pass"]:
            require(row["profiled"] and json.loads(row["profile_paths_json"]), "successful E5 row has no profile")
    # Stream aggregate JSON to avoid retaining a second copy of every sample.
    raw_samples, raw_summaries, errors, seen_samples = 0, set(), 0, set()
    with open_text(directory / "raw.jsonl") as stream:
        for line in stream:
            record = json.loads(line)
            kind, label = record["record_type"], record["config_id"]
            require(label in by_id, "aggregate raw record has no CSV row")
            row = by_id[label]
            if plan:
                check_placement_record(record, row)
            if kind == "sample":
                require(step == "E5" or not record.get("profiled", False), "profiled aggregate sample cannot enter production results")
                key = (label, record["sample_index"])
                require(key not in seen_samples, "duplicate aggregate sample")
                seen_samples.add(key)
                original = row["samples"][record["sample_index"]]
                for field in ("phase", "elapsed_s", "input_checksum", "converged", "eigenvalues", "residuals", "orth_error"):
                    require(record[field] == original[field], f"aggregate/local sample mismatch: {field}")
                require(not row["gate_pass"] or record["gate_pass"], "passing row has failed aggregate sample")
                raw_samples += 1
            elif kind == "summary":
                require(label not in raw_summaries, "duplicate aggregate summary")
                raw_summaries.add(label)
                require(record["gate_pass"] == row["gate_pass"] and record["runs"] == row["runs"], "raw summary disagrees with CSV")
            elif kind == "error":
                errors += 1
            else:
                raise ValueError(f"unknown raw record type {kind}")
    require(raw_samples == sum(len(row["samples"]) for row in rows), "aggregate sample count mismatch")
    require(raw_summaries == set(by_id), "aggregate summary coverage mismatch")
    expected = {f"s{sweep}_n{n}_{solver}_p{cores}" for sweep in range(1, args["sweeps"] + 1)
                for n in args["sizes"] for solver in args["solvers"]
                for cores in ([1] if solver == "cpp_serial" else args["cores"])}
    require(set(by_id) <= expected, "unexpected configuration in CSV")
    missing = sorted(expected - set(by_id))
    if missing:
        warnings.append("Incomplete job: expected configurations are missing")
    failed = [row["config_id"] for row in rows if not row["gate_pass"]]
    if manifest.get("status") != "complete" or not manifest.get("ok"):
        warnings.append("Job manifest does not report a complete successful run")
    maxima = {key: max((row[key] for row in rows if row[key] is not None), default=None)
              for key in ("max_eigenvalue_error", "max_residual", "max_orth_error")}
    audit = {"step": step, "job_id": str(manifest["job_id"]), "configurations": len(rows),
             "expected_configurations": len(expected), "samples": raw_samples,
             "measured_samples": sum(row["runs"] for row in rows), "warmups": sum(row["warmups"] for row in rows),
             "error_records": errors, "failed_configurations": failed,
             "capped_below_target": [row["config_id"] for row in rows if row["stop_reason"] == "max_runs" and not row["minimum_time_met"]],
             "maximum_accuracy": maxima, "missing_expected_configurations": missing,
             "input_checksums": checksums, "input_fingerprints": fingerprints, "warnings": warnings,
             "input_validation_note": "bytes = freshly rehashed input dump; packaging_manifest = hash and size "
                                      "verified during packaging, with input bytes subsequently omitted",
             "hashes": {name: sha256(directory / name) for name in
                        ("summary.csv", "raw.jsonl", "run.json", "environment.json", "source.json")}}
    if plan:
        audit["placement"] = {"selected_cpus": plan["selected"], "socket": plan["socket"],
                              "numa_node": plan["numa_node"], "memory_policy": plan["memory_policy"]["mode"],
                              "allocated_logical_cpus": len(plan["allowed_logical_cpus"]),
                              "allocated_physical_cores": len(plan["physical_cores"]),
                              "socket_physical_core_counts": plan["socket_physical_core_counts"],
                              "rank_placement_records": sum(len(json.loads(r["rank_placement_paths_json"])) for r in rows),
                              "nested_subsets_verified": True, "sysfs_lscpu_grouping_verified": True}
        audit["hashes"].update({name: sha256(directory / name) for name in
                                ("single_socket.json", "cpu_mapping.json", "hardware.json", "build/compiler.txt")})
    if exists(directory / "package.json"):
        package = read_json(directory / "package.json")
        require(package.get("schema_version") == 1 and package["archive"]["path"] == "evidence.zip",
                "unsupported compact evidence package")
        archive = directory / package["archive"]["path"]
        require(size(archive) == package["archive"]["size"]
                and sha256(archive) == package["archive"]["sha256"], "evidence archive differs from packaging manifest")
        audit["retention"] = {"mode": "compact_evidence_archive", "archive": package["archive"],
                              "manifest_sha256": sha256(directory / "package.json"),
                              "archived_members": len(package["members"]),
                              "omitted_input_dumps": sorted(package["verified_omitted_inputs"]),
                              "input_validation": {n: fp["validation"] for n, fp in fingerprints.items()}}
    return {"directory": directory, "manifest": manifest, "environment": environment,
            "cpu_mapping": mapping, "rows": rows, "audit": audit}


def compare_jobs(e4, e6):
    """Require comparable numerical work while recording intentional harness changes."""
    require(e4["audit"]["step"] == "E4" and e6["audit"]["step"] == "E6", "comparison requires E4 then E6")
    sources = [read_json(job["directory"] / "source.json")["sha256"] for job in (e4, e6)]
    numerical = ("matrix.cpp", "matrix.h", "davidson.cpp", "davidson.h", "bench/baseline_common.py",
                 "bench/cpp_baselines_bench.cpp", "bench/davidson_bench.cpp", "bench/pyscf_baselines.py",
                 "bench/pyscf_davidson_bench.py", "bench/slepc_davidson_bench.py")
    for name in numerical:
        require(sources[0].get(name) and sources[0].get(name) == sources[1].get(name), f"E4/E6 numerical source mismatch: {name}")
    changed = sorted(name for name in set(sources[0]) | set(sources[1]) if sources[0].get(name) != sources[1].get(name))
    allowed_changes = {"bench/run_baselines.py", "euler/run_baseline_job.sh", "euler/06_single_socket.sbatch",
                       "euler/single_socket.py", "euler/record_hardware.py"}
    require(set(changed) <= allowed_changes, "E4/E6 unexpected source change outside placement harness")
    for key in ("sizes", "solvers", "sweeps", "min_time", "min_runs", "max_runs", "cxx", "mpiexec", "log_view", "profile_pyscf"):
        require(e4["manifest"]["arguments"][key] == e6["manifest"]["arguments"][key], f"E4/E6 manifest setting mismatch: {key}")
    require(read_bytes(e4["directory"] / "build/compiler.txt") == read_bytes(e6["directory"] / "build/compiler.txt"),
            "E4/E6 compiler identity/flags differ")
    for key in ("python", "prefix", "packages", "petsc", "slepc", "mpi", "blas_alignment"):
        require(e4["environment"][key] == e6["environment"][key], f"E4/E6 environment mismatch: {key}")
    pools = [sorted(job["environment"]["threadpools"], key=lambda pool: pool["filepath"]) for job in (e4, e6)]
    require(pools[0] == pools[1], "E4/E6 loaded thread-pool libraries differ")
    inputs = {}
    for n in e4["manifest"]["arguments"]["sizes"]:
        inputs[str(n)] = {}
        for kind in ("input", "reference"):
            name = f"{kind}_n{n}.bin"
            hashes = [job["audit"]["input_fingerprints"][str(n)]["sha256"] if kind == "input"
                      else sha256(job["directory"] / name) for job in (e4, e6)]
            require(hashes[0] == hashes[1], f"E4/E6 {kind} fingerprints differ at n={n}")
            inputs[str(n)][kind + "_sha256"] = hashes[0]
        inputs[str(n)]["input_validation"] = {job["audit"]["step"]: job["audit"]["input_fingerprints"][str(n)]["validation"]
                                             for job in (e4, e6)}
    def normalized_settings(row):
        settings = json.loads(row["settings_json"])
        if "thread_environment" in settings:
            settings["thread_environment"].pop("OMP_PLACES", None)
        return settings
    old = {(r["sweep"], r["n"], r["solver"], r["cores"]): r for r in e4["rows"]}
    compared = 0
    for row in e6["rows"]:
        key = (row["sweep"], row["n"], row["solver"], row["cores"])
        previous = old.get(key)
        if previous and previous["gate_pass"] and row["gate_pass"]:
            require(normalized_settings(previous) == normalized_settings(row), f"E4/E6 solver settings differ: {row['config_id']}")
            require(json.loads(previous["versions_json"]) == json.loads(row["versions_json"]), f"E4/E6 runner versions differ: {row['config_id']}")
            compared += 1
    return {"e4_job_id": e4["audit"]["job_id"], "e6_job_id": e6["audit"]["job_id"],
            "same_hostname": e4["manifest"]["hostname"] == e6["manifest"]["hostname"],
            "numerical_sources_identical": list(numerical), "changed_harness_sources": changed,
            "compiler_flags_identical": True, "numerical_environment_identical": True,
            "compared_settings_rows": compared, "settings_excluded_placement_fields": ["thread_environment.OMP_PLACES"],
            "identical_inputs_and_references": inputs,
            "interpretation": "Separate shared-node jobs; placement comparison does not isolate every source of timing variation."}
