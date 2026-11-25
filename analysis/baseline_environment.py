"""Render environment tables from returned artifacts, without host inference."""

from collections import defaultdict
import json
import os
from pathlib import Path, PurePosixPath
import re
from baseline_storage import exists, read_text


MISSING = "not recorded"
LABELS = {"cpp_serial": "C++ serial", "cpp_openmp": "C++ OpenMP",
          "pyscf": "PySCF", "slepc": "SLEPc"}


def _read(path, json_value=False):
    try:
        value = read_text(path)
        return (json.loads(value) if value.strip() else {}) if json_value else value
    except FileNotFoundError:
        return {} if json_value else ""


def _shown(value):
    return MISSING if value is None or value == "" else str(value)


def _version(value):
    return ".".join(map(str, value)) if isinstance(value, (tuple, list)) else _shown(value)


def _table(headers, rows):
    def line(cells):
        return "| " + " | ".join(str(x).replace("|", "\\|").replace("\n", " ") for x in cells) + " |"
    return "\n".join([line(headers), line(["---"] * len(headers)), *(line(row) for row in rows)])


def _link(label, target, output):
    return f"[{label}]({Path(os.path.relpath(target, Path(output).parent)).as_posix()})"


def _fields(text):
    return dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_/:]*)=([^\s]+)", text))


def _size_bytes(value):
    match = re.fullmatch(r"([\d.]+)([KMG]?)", str(value))
    if not match:
        return None
    return float(match[1]) * 1024 ** {"": 0, "K": 1, "M": 2, "G": 3}[match[2]]


def _binary_size(value):
    if value is None:
        return MISSING
    for power, label in ((3, "GiB"), (2, "MiB"), (1, "KiB")):
        if value >= 1024 ** power:
            return f"{value / 1024 ** power:g} {label}"
    return f"{value:g} bytes"


def _range(values, digits=3):
    values = [float(x) for x in values if x not in (None, "", MISSING)]
    if not values:
        return MISSING
    low, high = min(values), max(values)
    return f"{low:.{digits}f}" if low == high else f"{low:.{digits}f}–{high:.{digits}f}"


def collect_environment(job):
    """Return parsed job evidence; optional absent artifacts stay explicit."""
    directory = job["directory"]
    hardware = _read(directory / "hardware.json", True)
    environment = job.get("environment") or _read(directory / "environment.json", True)
    topology = _read(directory / "single_socket.json", True)
    lscpu = _read(directory / "hardware/lscpu-json.out", True)
    fields = {item["field"].rstrip(":").strip(): item["data"] for item in lscpu.get("lscpu", [])}
    compiler = _read(directory / "build/compiler.txt")
    compiler_fields = {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                       for line in compiler.splitlines() if ":" in line}
    compiler_version = next((line for line in compiler.splitlines() if line.startswith("g++ (")), MISSING)
    slurm = {kind: _fields(_read(directory / f"hardware/slurm-{kind}.out"))
             for kind in ("job", "node", "partition")}
    caches = defaultdict(dict)
    for cache in hardware.get("sysfs", {}).get("caches", []):
        if not isinstance(cache, dict):
            continue
        group = (cache.get("level"), cache.get("type"))
        shared = cache.get("shared_cpu_list", MISSING)
        size = _size_bytes(cache.get("size"))
        if shared in caches[group] and caches[group][shared] != size:
            raise ValueError("Contradictory cache size for a shared cache instance")
        caches[group][shared] = size
    commands = {}
    for row in job.get("rows", []):
        key = (row["solver"], int(row["cores"]))
        value = _read(directory / "configs" / row["config_id"] / "runner.command.json", True)
        if key not in commands:
            commands[key] = (value, row["config_id"])
        elif value.get("environment") != commands[key][0].get("environment"):
            raise ValueError(f"Runtime environment varies within solver/core configuration {key}")
    return {"hardware": hardware, "environment": environment, "topology": topology,
            "lscpu": fields, "compiler": compiler_fields, "compiler_version": compiler_version,
            "slurm": slurm, "caches": caches, "commands": commands}


def build_environment(job, output):
    """Build an artifact-linked Markdown section for the primary E6 job."""
    directory = job["directory"]
    data = collect_environment(job)
    hardware, env, topology = (data[key] for key in ("hardware", "environment", "topology"))
    cpu, slurm = data["lscpu"], data["slurm"]
    def source(label, path):
        target = directory / path
        return _link(label, target, output) if exists(target) else MISSING
    raw_hardware = source("hardware.json", "hardware.json")
    cpu_source = source("lscpu", "hardware/lscpu.out")
    job_source = source("Slurm job", "hardware/slurm-job.out")
    node_source = source("Slurm node", "hardware/slurm-node.out")
    software_source = source("environment.json", "environment.json")
    selected = topology.get("selected", [])
    frequencies = hardware.get("sysfs", {}).get("cpufreq", [])
    selected_freq = [entry["values"].get("scaling_cur_freq") for entry in frequencies
                     if isinstance(entry, dict) and isinstance(entry.get("values"), dict)
                     and entry.get("cpu") in {f"cpu{x}" for x in selected}]
    selected_freq = [float(value) / 1000 for value in selected_freq if value not in (None, MISSING)]
    governors = sorted({entry["values"].get("scaling_governor", MISSING) for entry in frequencies
                        if isinstance(entry, dict) and isinstance(entry.get("values"), dict)})
    memory = hardware.get("memory_total", MISSING)
    memory_match = re.fullmatch(r"(\d+) kB", memory)
    memory_text = memory + (f" ({int(memory_match[1]) / 1024**2:.2f} GiB, unit conversion)" if memory_match else "")
    kernel = hardware.get("kernel", {})
    node_type = slurm["node"].get("ActiveFeatures", MISSING)
    rows = [
        ["Node / recorded type", f"{hardware.get('hostname', MISSING)}; Slurm active features: `{node_type}`", node_source],
        ["CPU", cpu.get("Model name", MISSING), cpu_source],
        ["Microarchitecture label", hardware.get("microarchitecture", MISSING), raw_hardware],
        ["Sockets, physical cores, SMT", f"{cpu.get('Socket(s)', MISSING)} sockets × {cpu.get('Core(s) per socket', MISSING)} cores/socket; "
         f"{cpu.get('Thread(s) per core', MISSING)} threads/core; {cpu.get('CPU(s)', MISSING)} logical CPUs", cpu_source],
        ["Clock evidence", f"Recorded CPU limits {cpu.get('CPU min MHz', MISSING)}–{cpu.get('CPU max MHz', MISSING)} MHz; "
         f"selected-CPU snapshot {_range(selected_freq)} MHz; governor {', '.join(governors) or MISSING}; "
         f"boost {cpu.get('Frequency boost', MISSING)}. Nominal/base clock: {MISSING}.", raw_hardware],
        ["Memory", f"OS MemTotal {memory_text}; Slurm RealMemory {slurm['node'].get('RealMemory', MISSING)} MB", raw_hardware + "; " + node_source],
        ["OS", hardware.get("os_release", {}).get("PRETTY_NAME", MISSING), source("os-release", "hardware/os-release.txt")],
        ["Kernel", f"{kernel.get('system', MISSING)} {kernel.get('release', MISSING)}; {kernel.get('version', MISSING)}", raw_hardware],
        ["Slurm allocation", f"partition `{slurm['job'].get('Partition', MISSING)}`, constraint `{slurm['job'].get('Features', MISSING)}`; "
         f"{slurm['job'].get('NumNodes', MISSING)} node, {slurm['job'].get('NumCPUs', MISSING)} logical CPUs, "
         f"memory request {slurm['job'].get('MinMemoryNode', MISSING)}, wall limit {slurm['job'].get('TimeLimit', MISSING)}", job_source],
        ["Sharing / scheduler", f"node state {slurm['node'].get('State', MISSING)}, CPUAlloc={slurm['node'].get('CPUAlloc', MISSING)} "
         f"at capture; Slurm {slurm['node'].get('Version', MISSING)}; partition OverSubscribe={slurm['partition'].get('OverSubscribe', MISSING)}", node_source + "; " + source("partition", "hardware/slurm-partition.out")],
    ]
    parts = ["## Environment and hardware", _table(["Item", "Recorded value", "Evidence"], rows),
             "Clock readings are startup snapshots, not measured solve frequencies. The artifacts do not record the marketing label “Zen 4”; it is therefore not supplied as a measured hardware field."]
    cache_rows = []
    for (level, kind), instances in sorted(data["caches"].items()):
        sizes = set(instances.values())
        size = _binary_size(next(iter(sizes))) if len(sizes) == 1 else ", ".join(sorted(map(_binary_size, sizes)))
        total = sum(instances.values()) if None not in sizes else None
        cache_rows.append([f"L{level} {kind}", size, len(instances), _binary_size(total)])
    parts += ["Cache instances are deduplicated by level, type and shared CPU list; totals cover the whole node. "
              + source("Raw cache sharing/size evidence", "hardware/sysfs.json") + ".",
              _table(["Cache", "Size per instance", "Distinct instances", "Whole-node total"], cache_rows or [[MISSING] * 4])]
    numa_rows = []
    for node in hardware.get("sysfs", {}).get("numa_nodes", []):
        if isinstance(node, dict):
            match = re.search(r"MemTotal:\s*(\d+ kB)", node.get("meminfo", ""))
            numa_rows.append([node.get("node", MISSING), node.get("cpulist", MISSING),
                              match[1] if match else MISSING, node.get("distance", MISSING)])
    parts += ["NUMA layout (distance columns follow node order). " + raw_hardware + ".",
              _table(["NUMA node", "Logical CPUs", "Node MemTotal", "Distance row"], numa_rows or [[MISSING] * 4])]
    policy = topology.get("memory_policy", {})
    parts.append(f"E6 selects socket **{topology.get('socket', MISSING)}**, NUMA node **{topology.get('numa_node', MISSING)}**. "
                 + source("The placement plan", "single_socket.json") + " records distinct physical cores and the original allocation. "
                 + source("Driver placement", "cpu_mapping.json") + " records its verified allocation/first-touch context.")
    parts.append(_table(["Physical cores p", "Selected logical CPU IDs (one per physical core)", "Socket", "NUMA node"],
                        [[p, ",".join(map(str, selected[:p])) or MISSING, topology.get("socket", MISSING), topology.get("numa_node", MISSING)]
                         for p in job["manifest"]["arguments"]["cores"]]))
    if policy.get("mode") == "pinned_first_touch":
        parts.append("Memory policy: **pinned first-touch**. `numactl` was unavailable, so `numactl -H` and an explicit `--membind` policy were **not recorded**. "
                     "The driver, input dumper, solver processes and MPI ranks were pinned before matrix allocation/first-touch. "
                     "Resident-page locality was not measured; pinning alone cannot prove every page stayed local. "
                     + source("numactl availability error", "hardware/numactl-hardware.err") + "; "
                     + source("rank 0 pre-solver evidence", "configs/s1_n64_slepc_p8/rank_placement/rank-0.json") + ".")
    else:
        parts.append(f"Memory policy: **{policy.get('mode', MISSING)}**. " + source("Policy and verification", "single_socket.json") + "; "
                     + source("numactl hardware", "hardware/numactl-hardware.out") + ".")
    compiler = data["compiler"]
    parts += ["### Software and binding", _table(["Component", "Recorded version / setting", "Evidence"], [
        ["C++ compiler", f"{compiler.get('Resolved compiler', MISSING)}; {data['compiler_version']}; target {compiler.get('Target', MISSING)}", source("compiler.txt", "build/compiler.txt")],
        ["C++ flags", "`" + compiler.get("Flags", MISSING) + "`", source("compiler.txt", "build/compiler.txt")],
        ["Python", env.get("python", MISSING), software_source],
        *[[name, _version(details.get("version")), software_source] for name, details in sorted(env.get("packages", {}).items())],
        ["PETSc (native)", _version(env.get("petsc", {}).get("version")) + "; scalar type " + env.get("petsc", {}).get("scalar_dtype", MISSING), software_source],
        ["SLEPc (native)", _version(env.get("slepc", {}).get("version")), software_source],
        ["MPI", env.get("mpi", {}).get("vendor", MISSING) + " " + _version(env.get("mpi", {}).get("vendor_version")), software_source],
        *[[pool.get("internal_api", MISSING), f"version {_version(pool.get('version'))}; threading variant {pool.get('threading_layer', MISSING)}; `{pool.get('filepath', MISSING)}`", software_source]
          for pool in env.get("threadpools", [])],
    ])]
    alignment = env.get("blas_alignment", {})
    providers = alignment.get("dependency_provider_matches", {})
    if alignment.get("status") == "verified_shared_loaded_dependency" and providers.get("numpy") == providers.get("petsc") and providers.get("numpy"):
        parts.append("NumPy and native PETSc were verified to load the **same OpenBLAS library**, also shared by SciPy/SLEPc in the recorded dependency check: "
                     + ", ".join(f"`{path}`" for path in providers["numpy"]) + ". " + software_source + ".")
    else:
        parts.append("NumPy/PETSc shared BLAS identity: **not recorded as verified**. " + software_source + ".")
    kernels = sorted({pool["architecture"] for pool in env.get("threadpools", []) if pool.get("architecture")})
    if kernels:
        parts.append("The BLAS-reported kernel architecture is `" + ", ".join(kernels) + "`; this is a library dispatch label, not the host CPU model. " + software_source + ".")
    owner = _read(directory / "environment-owner.json", True)
    parts.append(f"Environment prefix: `{env.get('prefix', MISSING)}`. PySCF package source: {owner.get('pyscf_source', MISSING)}. "
                 + source("Environment ownership/provenance", "environment-owner.json") + ".")
    if owner.get("e1_artifacts"):
        install = directory.parent / PurePosixPath(owner["e1_artifacts"]).name
        snapshot = install / "conda-packages.json"
        if snapshot.is_file():
            parts.append("The " + _link("E1 complete package/version inventory", snapshot, output) + " and "
                         + _link("explicit environment lock", install / "conda-explicit.txt", output)
                         + " record installation provenance. E6's runtime versions above are the direct execution evidence; a fresh complete package inventory was not recorded in E6. "
                         "Versions absent from runtime records, including the loaded OpenMP runtime version, remain **not recorded**.")
    thread_rows = []
    for (solver, cores), (command, label) in sorted(data["commands"].items()):
        variables = command.get("environment", {})
        keys = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
        limits = [variables.get(key) for key in keys]
        thread_rows.append([LABELS.get(solver, solver), cores, _shown(variables.get("OMP_NUM_THREADS")),
                            _shown(limits[0]) if len(set(limits)) == 1 else ", ".join(f"{key}={_shown(variables.get(key))}" for key in keys),
                            "unset" if variables.get("OMP_PROC_BIND") is None else variables["OMP_PROC_BIND"],
                            "unset" if variables.get("OMP_PLACES") is None else variables["OMP_PLACES"],
                            source("command", f"configs/{label}/runner.command.json")])
    parts += ["The following values come from retained launch environments and are checked for consistency across sizes/sweeps. "
              "“BLAS/aux limits” covers `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `BLIS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` and `NUMEXPR_NUM_THREADS`; "
              "a configured limit does not mean every corresponding library was loaded.",
              _table(["Solver", "p", "OMP_NUM_THREADS", "BLAS/aux limits", "OMP_PROC_BIND", "OMP_PLACES", "Launch evidence"], thread_rows)]
    observed = [value[0].get("environment", {}) for value in data["commands"].values()]
    unset_keys = ("GOMP_CPU_AFFINITY", "KMP_AFFINITY", "HYDRA_BINDING", "HYDRA_MAPPING", "HYDRA_MEMBIND", "HYDRA_TOPOLIB", "HYDRA_TOPO_DEBUG", "HYDRA_HOST_FILE", "HYDRA_LAUNCHER_EXTRA_ARGS")
    if observed and all(values.get(key) is None for values in observed for key in unset_keys):
        parts.append("Recorded as unset for every launch: " + ", ".join(f"`{key}`" for key in unset_keys) + ".")
    gpu_values = sorted({str(values.get("MPIR_CVAR_ENABLE_GPU", MISSING)) for values in observed})
    dynamic = set()
    dynamic_source = None
    for row in job.get("rows", []):
        if row["solver"].startswith("cpp_"):
            settings = json.loads(row["settings_json"]) if row.get("settings_json") else row.get("settings", {})
            if "OMP_DYNAMIC" in settings.get("thread_environment", {}):
                dynamic.add(settings["thread_environment"]["OMP_DYNAMIC"])
                dynamic_source = source("C++ sample settings", f"configs/{row['config_id']}/summary.json")
    dynamic_text = ("C++ records `OMP_DYNAMIC`=" + "/".join(sorted(dynamic)) + "; " + dynamic_source + ". " if dynamic else "")
    parts.append("`MPIR_CVAR_ENABLE_GPU`=" + "/".join(gpu_values or [MISSING]) + ". " + dynamic_text
                 + "`OMP_DYNAMIC` for Python launches and `OMP_THREAD_LIMIT` were **not recorded** in the returned runtime snapshots. "
                 "CPU process masks, OpenMP worker masks and MPI rank masks are retained in each configuration's sample resources; "
                 "the tables use one hardware thread per selected physical core.")
    mpi_entries = [(key, value) for key, value in data["commands"].items() if key[0] == "slepc"]
    if mpi_entries:
        (_, cores), (command, label) = max(mpi_entries, key=lambda entry: entry[0][1])
        argv = command.get("command", [])
        launcher = next((arg for arg in argv if PurePosixPath(arg).name == "mpiexec"), MISSING)
        options = []
        for name in ("-launcher", "-hosts", "-bind-to", "-n"):
            options.extend([name, argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else MISSING])
        parts.append("Recorded MPI launch example: `" + launcher + " " + " ".join(options) + "`. "
                     "The per-rank placement wrapper runs after Hydra and records its verified singleton CPU and memory policy before SLEPc starts. "
                     + source("MPI command", f"configs/{label}/runner.command.json") + "; "
                     + source("rank 0 evidence", f"configs/{label}/rank_placement/rank-0.json") + ".")
    return "\n\n".join(parts) + "\n"
