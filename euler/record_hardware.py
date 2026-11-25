#!/usr/bin/env python3
"""Record E6 hardware evidence without requiring optional system utilities."""
import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import re
import shlex
import socket
import subprocess

MISSING = "not recorded"
SLURM_KEYS = (
    "SLURM_JOB_ID SLURM_JOB_NAME SLURM_JOB_PARTITION SLURM_JOB_NODELIST "
    "SLURMD_NODENAME SLURM_JOB_NUM_NODES SLURM_NTASKS SLURM_CPUS_PER_TASK "
    "SLURM_JOB_CPUS_PER_NODE SLURM_CPUS_ON_NODE SLURM_CPU_BIND "
    "SLURM_CPU_BIND_LIST SLURM_MEM_PER_NODE SLURM_CONSTRAINT "
    "SLURM_JOB_CONSTRAINTS"
).split()
RUNTIME_KEYS = (
    "CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CONDA_PREFIX "
    "OMP_NUM_THREADS OMP_THREAD_LIMIT OMP_DYNAMIC OMP_PROC_BIND OMP_PLACES "
    "GOMP_CPU_AFFINITY KMP_AFFINITY OPENBLAS_NUM_THREADS MKL_NUM_THREADS "
    "BLIS_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS "
    "PYTHONNOUSERSITE MPICH_VERSION MPICH_GPU_SUPPORT_ENABLED "
    "HYDRA_BINDING HYDRA_TOPO_DEBUG MPIR_CVAR_ENABLE_GPU"
).split()


class Evidence:
    def __init__(self, raw_dir):
        self.raw_dir = raw_dir
        self.unavailable = {}
        self.commands = {}
        self.files = {}

    def read(self, path):
        try:
            return Path(path).read_text(encoding="utf-8", errors="replace").strip()
        except OSError as exc:
            self.unavailable[str(path)] = str(exc)
            return MISSING

    def save(self, name, text):
        target = self.raw_dir / name
        with target.open("x", encoding="utf-8") as handle:
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
        return f"hardware/{name}"

    def raw_file(self, name, source):
        value = self.read(source)
        self.files[name] = {
            "source": source, "artifact": self.save(name, value),
            "status": MISSING if value == MISSING else "recorded",
        }
        return value

    def command(self, name, arguments):
        error = None
        try:
            completed = subprocess.run(
                arguments, capture_output=True, text=True, errors="replace",
                timeout=20, check=False,
            )
            out, err, code = completed.stdout, completed.stderr, completed.returncode
        except OSError as exc:
            out, err, code, error = "", str(exc), 127, str(exc)
        except subprocess.TimeoutExpired as exc:
            def decoded(value):
                return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
            out, err, code = decoded(exc.stdout), decoded(exc.stderr), 124
            error = "command timed out after 20 seconds"
        self.commands[name] = {
            "argv": arguments, "exit_code": code,
            "stdout": self.save(name + ".out", out),
            "stderr": self.save(name + ".err", err),
            "status": "recorded" if code == 0 else MISSING,
        }
        if error:
            self.commands[name]["error"] = error
        print(f"hardware: {name}: exit {code}", flush=True)
        return out if code == 0 else MISSING


def colon_fields(text):
    return dict(line.split(":", 1) for line in text.splitlines() if ":" in line)


def cpuinfo_records(text):
    wanted = ("processor", "vendor_id", "model name", "cpu family", "model",
              "stepping", "cpu MHz", "physical id", "core id", "cpu cores", "siblings")
    records = []
    for block in text.split("\n\n"):
        values = {key.strip(): value.strip() for key, value in colon_fields(block).items()}
        if "processor" in values:
            records.append({key: values.get(key, MISSING) for key in wanted})
    return records or MISSING


def sysfs_records(evidence):
    caches, frequencies, nodes = [], [], []
    cpu_root = Path("/sys/devices/system/cpu")
    cache_keys = ("level", "type", "size", "shared_cpu_list", "shared_cpu_map",
                  "coherency_line_size", "number_of_sets", "ways_of_associativity", "id")
    frequency_keys = ("scaling_min_freq", "scaling_max_freq", "scaling_cur_freq",
                      "cpuinfo_min_freq", "cpuinfo_max_freq", "scaling_governor")
    for cpu in sorted(cpu_root.glob("cpu[0-9]*")):
        if not re.fullmatch(r"cpu\d+", cpu.name):
            continue
        for cache in sorted((cpu / "cache").glob("index[0-9]*")):
            caches.append({"cpu": cpu.name, "index": cache.name, "source": str(cache),
                           **{key: evidence.read(cache / key) for key in cache_keys}})
        frequency = cpu / "cpufreq"
        frequencies.append({
            "cpu": cpu.name, "source": str(frequency), "frequency_unit": "kHz",
            "values": {key: evidence.read(frequency / key) for key in frequency_keys}
            if frequency.is_dir() else MISSING,
        })
    for node in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
        if re.fullmatch(r"node\d+", node.name):
            nodes.append({"node": node.name, "source": str(node),
                          **{key: evidence.read(node / key)
                             for key in ("cpulist", "meminfo", "distance")}})
    result = {"caches": caches or MISSING, "cpufreq": frequencies or MISSING,
              "numa_nodes": nodes or MISSING,
              "online_cpus": evidence.read(cpu_root / "online"),
              "online_nodes": evidence.read("/sys/devices/system/node/online")}
    evidence.save("sysfs.json", json.dumps(result, indent=2, sort_keys=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "hardware.json"
    raw_dir = args.output_dir / "hardware"
    if target.exists() or raw_dir.exists():
        parser.error("hardware.json or hardware/ already exists; refusing to overwrite evidence")
    raw_dir.mkdir()
    evidence = Evidence(raw_dir)
    for name, arguments in (
        ("lscpu", ["lscpu"]),
        ("lscpu-json", ["lscpu", "--json"]),
        ("lscpu-extended", ["lscpu", "--extended=CPU,CORE,SOCKET,NODE,ONLINE,MHZ"]),
        ("lscpu-caches", ["lscpu", "--caches"]),
        ("uname", ["uname", "-a"]),
        ("numactl-hardware", ["numactl", "--hardware"]),
        ("numactl-show", ["numactl", "--show"]),
    ):
        evidence.command(name, arguments)
    cpuinfo = cpuinfo_records(evidence.raw_file("cpuinfo.txt", "/proc/cpuinfo"))
    memory = evidence.raw_file("meminfo.txt", "/proc/meminfo")
    memory_fields = {key.strip(): value.strip() for key, value in colon_fields(memory).items()}
    release = evidence.raw_file("os-release.txt", "/etc/os-release")
    os_fields = {}
    for line in release.splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            try:
                os_fields[key] = " ".join(shlex.split(value))
            except ValueError:
                os_fields[key] = value
    evidence.raw_file("process-status.txt", "/proc/self/status")
    slurm = {key: os.environ.get(key, MISSING) for key in SLURM_KEYS}
    job_id = os.environ.get("SLURM_JOB_ID")
    hostname = socket.gethostname()
    if job_id:
        evidence.command("slurm-job", ["scontrol", "show", "job", job_id])
    else:
        evidence.commands["slurm-job"] = {"status": MISSING, "reason": "SLURM_JOB_ID is unset"}
    evidence.command("slurm-node", ["scontrol", "show", "node",
                                    os.environ.get("SLURMD_NODENAME", hostname)])
    if os.environ.get("SLURM_JOB_PARTITION"):
        evidence.command("slurm-partition", ["scontrol", "show", "partition",
                                             os.environ["SLURM_JOB_PARTITION"]])
    else:
        evidence.commands["slurm-partition"] = {"status": MISSING, "reason": "SLURM_JOB_PARTITION is unset"}
    sysfs = sysfs_records(evidence)
    result = {
        "schema_version": 1,
        "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hostname": hostname, "job_id": job_id or MISSING,
        "kernel": platform.uname()._asdict(), "os_release": os_fields or MISSING,
        "cpuinfo_logical_cpus": cpuinfo, "microarchitecture": MISSING,
        "memory_total": memory_fields.get("MemTotal", MISSING), "sysfs": sysfs,
        "slurm_environment": slurm,
        "runtime_environment_at_capture": {key: os.environ.get(key, MISSING) for key in RUNTIME_KEYS},
        "commands": evidence.commands, "files": evidence.files,
        "unavailable_files": evidence.unavailable,
        "notes": [
            "Clock readings are observations at capture time, not guaranteed operating frequencies.",
            "Cache rows repeat shared caches; shared_cpu_list identifies sharing, so do not sum all rows.",
            "Microarchitecture marketing names are not inferred from CPU model or family.",
            "This captures the job environment before solver launch; per-configuration affinity is recorded separately.",
        ],
    }
    with target.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"hardware: evidence saved to {target}", flush=True)


if __name__ == "__main__":
    main()
