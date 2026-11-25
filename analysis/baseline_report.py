#!/usr/bin/env python3
"""Generate the Euler report from audited E4/E5/E6 evidence.

Run from the repository root with Python 3.10+ (E6 figures require matplotlib):
  python3 -B analysis/baseline_report.py --e6 results/euler/E6_130120 \
      --e4 results/euler/E4_129538 \
      --e5 results/euler/E5_129548
Never edits inputs, reruns solvers, pools sweeps, or substitutes profiling times.
"""

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import pstats
import re
import sys

from baseline_evidence import load_job
from baseline_storage import input_fingerprint, read_bytes, read_text

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"cpp_serial": "C++ serial", "cpp_openmp": "C++ OpenMP", "pyscf": "PySCF", "slepc": "SLEPc"}
COLUMNS = [("pyscf", 1), ("pyscf", 8), ("slepc", 1), ("slepc", 8),
           ("cpp_serial", 1), ("cpp_openmp", 8)]
ALL_COLUMNS = [("cpp_serial", 1), ("cpp_openmp", 1), ("cpp_openmp", 8),
               ("pyscf", 1), ("pyscf", 8), ("slepc", 1), ("slepc", 8)]


def fmt(value, digits=3):
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def sci(value):
    return "—" if value is None else f"{value:.3e}"


def table(headers, rows):
    def line(cells):
        return "| " + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in cells) + " |"
    return "\n".join([line(headers), line(["---"] * len(headers)), *(line(row) for row in rows)])


def link(label, target, output):
    path = Path(os.path.relpath(target, output.parent)).as_posix()
    return f"[{label}]({path})"


def row_link(job, row, output):
    return link(row["config_id"], job["directory"] / "configs" / row["config_id"] / "summary.json", output)


def valid(row):
    return bool(row and row["gate_pass"] and row["strict_gate_pass"] and row["exit_code"] == 0)


def index_rows(job):
    return {(r["sweep"], r["n"], r["solver"], r["cores"]): r for r in job["rows"]}


def pair(index, n, solver, cores, metric="median_ms", digits=3):
    cells = []
    for sweep in (1, 2):
        row = index.get((sweep, n, solver, cores))
        cells.append("MISSING" if row is None else "FAIL" if not valid(row) else fmt(row[metric], digits))
    return " / ".join(cells)


def speedup(index, n, solver, sweep):
    one, eight = index.get((sweep, n, solver, 1)), index.get((sweep, n, solver, 8))
    if not valid(one) or not valid(eight) or not eight["median_ms"]:
        return None
    return one["median_ms"] / eight["median_ms"]


def count_range(rows, prefix):
    values = [r.get(prefix + suffix) for r in rows for suffix in ("_min", "_max")]
    if not values or any(v is None for v in values):
        return "—"
    lo, hi = min(values), max(values)
    return f"{lo:g}" if lo == hi else f"{lo:g}–{hi:g}"


def maximum(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return max(values) if values else None


def job_interval(job):
    text = read_text(job["directory"] / "env.txt")
    start = re.search(r"UTC start: (\S+)", text)
    end = re.search(r"E[456] PASS at (\S+)\.", text)
    if not start or not end:
        return "not established by a complete job log", None
    a, b = (datetime.fromisoformat(m.group(1).replace("Z", "+00:00")) for m in (start, end))
    return f"{a.strftime('%Y-%m-%d %H:%M:%S')}–{b.strftime('%H:%M:%S')} UTC", (b - a).total_seconds()


def parse_petsc_log(path):
    """Use PETSc's printed stage percentages; never divide rank max by stage avg."""
    events, in_measured_stage = {}, False
    for line in read_text(path).splitlines():
        if line.startswith("--- Event Stage"):
            in_measured_stage = line.endswith(": baseline_slepc_repeat1")
        elif in_measured_stage:
            cells = line.split()
            if len(cells) >= 20 and cells[0] in ("MatMult", "BVOrthogonalizeV", "BVMatProject", "VecReduceComm", "SFBcastEnd"):
                events[cells[0]] = {"count": int(cells[1]), "max_ms": float(cells[3]) * 1000,
                                    "stage_time_percent": float(cells[15])}
    if "MatMult" not in events:
        raise ValueError(f"missing measured-stage MatMult evidence in {path}")
    return events


def collect_profiles(job):
    data = []
    for row in job["rows"]:
        item = {"n": row["n"], "cores": row["cores"], "solver": row["solver"],
                "config_id": row["config_id"], "csv_line": row["csv_line"], "gate_pass": valid(row)}
        paths = [job["directory"] / p for p in json.loads(row["profile_paths_json"])]
        if not valid(row):
            data.append(item)
            continue
        item["paths"] = [str(path.relative_to(job["directory"])) for path in paths]
        if row["solver"] == "pyscf":
            stats = pstats.Stats(str(next(p for p in paths if p.suffix == ".prof")))
            davidson = [v for k, v in stats.stats.items() if k[2] == "davidson1"]
            dots = [v for k, v in stats.stats.items() if k[2] == "<method 'dot' of 'numpy.ndarray' objects>"]
            if len(davidson) != 1 or sum(v[1] for v in davidson) != row["runs"] or not dots:
                raise ValueError(f"unexpected cProfile call coverage: {row['config_id']}")
            item.update(davidson_calls=sum(v[1] for v in davidson), dot_calls=sum(v[1] for v in dots),
                        dot_fraction_percent=100 * sum(v[2] for v in dots) / davidson[0][3])
        else:
            item["events"] = parse_petsc_log(paths[0])
        data.append(item)
    return data


def build_details(job, profile_job, profiles, output):
    parts = ["# Euler baseline row-level evidence", "Generated by `analysis/baseline_report.py`; do not edit numeric tables manually.",
             f"{job['manifest']['arguments']['step']} job **{job['manifest']['job_id']}**. CSV line numbers include the header as line 1. "
             + link("Source CSV", job["directory"] / "summary.csv", output) + ". Links below open configuration summaries; their raw samples retain every call.",
             "## Per-configuration timing and repeat policy",
             "Times are ms; warm-ups are excluded. Rows marked FAIL are retained and must not support speedup claims. A capped row can pass accuracy while missing the one-second target."]
    rows = sorted(job["rows"], key=lambda r: (r["sweep"], r["n"], list(LABELS).index(r["solver"]), r["cores"]))
    parts.append(table(["Configuration", "CSV line", "Measured runs", "Median", "Min–max", "Timed total (s)", "Stop / 1s met", "Gate"],
                       [[row_link(job, r, output), r["csv_line"], r["runs"], fmt(r["median_ms"]),
                         f"{fmt(r['min_ms'])}–{fmt(r['max_ms'])}", fmt(r["total_timed_s"]),
                         f"{r['stop_reason']} / {r['minimum_time_met']}", "PASS" if valid(r) else "FAIL"] for r in rows]))
    parts += ["## Per-configuration counts and accuracy",
              "Accuracy maxima include warm-up and measured calls. Count ranges and ms/matvec use measured calls only. SLEPc unknown vector-equivalent counts remain blank (—)."]
    parts.append(table(["Configuration", "CSV line", "Iterations", "Matvecs", "Median ms/matvec", "Max eigenvalue error", "Max residual", "Max orthogonality", "Gate / reason"],
                       [[row_link(job, r, output), r["csv_line"], count_range([r], "iterations"), count_range([r], "matvecs"),
                         fmt(r["median_ms_per_matvec"], 6), sci(r["max_eigenvalue_error"]), sci(r["max_residual"]), sci(r["max_orth_error"]),
                         "PASS" if valid(r) else "FAIL: " + (r.get("failure_reason") or "see raw records")] for r in rows]))
    parts += ["## Inputs", "Each size's reference is computed once and shared across both sweeps. Reference times are outside solve timing."]
    refs = []
    for n in job["manifest"]["arguments"]["sizes"]:
        path = job["directory"] / f"reference_n{n}.json"
        ref = json.loads(read_text(path))
        refs.append([n, ref["input_checksum"], fmt(ref.get("elapsed_s"), 6), link("reference metadata", path, output)])
    parts.append(table(["n", "Input checksum", "Reference seconds", "Evidence"], refs))
    if profile_job:
        parts += ["## Optional E5 profiles", f"E5 job **{profile_job['manifest']['job_id']}**; "
                  + link("source CSV", profile_job["directory"] / "summary.csv", output)
                  + ". These calls are diagnostic; their timings are excluded from E4/E6 comparisons."]
        p_rows = []
        for item in profiles:
            p_rows.append([item["config_id"], item["csv_line"], "PASS" if item["gate_pass"] else "FAIL",
                           ", ".join(link(Path(p).name, profile_job["directory"] / p, output) for p in item.get("paths", []))])
        parts.append(table(["Configuration", "CSV line", "Gate", "Profile artifacts"], p_rows))
    return "\n\n".join(parts) + "\n"


def build_report(job, profile_job, profiles, output, details, audit_path):
    rows, manifest = job["rows"], job["manifest"]
    index = index_rows(job)
    sizes = manifest["arguments"]["sizes"]
    failures = [r for r in rows if not valid(r)]
    sample_count = sum(len(r["samples"]) for r in rows)
    capped = [r for r in rows if r["stop_reason"] == "max_runs" and not r["minimum_time_met"]]
    missing = job["audit"].get("missing_expected_configurations", [])
    env = job["environment"]
    parts = ["# Euler Davidson baseline report", "Generated from returned Euler artifacts by `analysis/baseline_report.py`. "
             + link("Full row-level tables", details, output) + " and " + link("machine-readable audit", audit_path, output)
             + " preserve CSV line/configuration provenance; no local, E2, E3 or E5 timing is used in E4 tables.",
             f"**E4 {manifest['job_id']}: {len(rows)} configurations, {sample_count:,} solver calls, "
             f"{len(failures)} failed configuration gates, {len(missing)} missing configurations.** "
             "Sweep 1 and sweep 2 are shown separately throughout; samples are never pooled across sweeps.",
             "## Setup"]
    if profile_job:
        parts.insert(3, f"**E5 {profile_job['manifest']['job_id']}: {len(profile_job['rows'])} diagnostic configurations, "
                     f"{sum(len(r['samples']) for r in profile_job['rows'])} calls, "
                     f"{sum(not valid(r) for r in profile_job['rows'])} failed configuration gates.**")
    warnings = [warning for current in (job, profile_job) if current for warning in current["audit"]["warnings"]]
    if warnings:
        parts.insert(3, "**Evidence warnings:** " + "; ".join(warnings))
    job_rows = []
    for step, current in (("E4", job), ("E5", profile_job)):
        if current:
            interval, duration = job_interval(current)
            job_rows.append([step, current["manifest"]["job_id"], current["manifest"]["hostname"], interval,
                             fmt(duration, 0), link("CSV", current["directory"] / "summary.csv", output),
                             link("environment", current["directory"] / "environment.json", output)])
    parts.append(table(["Step", "Job", "Node", "Recorded interval", "Wall seconds", "Results", "Environment"], job_rows))
    mapping = job["cpu_mapping"]
    sockets = Counter(p["socket"] for p in mapping["physical_cores"] if p["cpu"] in mapping["selected"])
    parts.append(f"Hardware: AMD EPYC 9254, two sockets with 24 physical cores each, SMT2, two NUMA nodes. "
                 f"The job allocated {len(mapping['allowed_logical_cpus'])} logical CPUs and used eight physical representatives "
                 f"`{','.join(map(str, mapping['selected']))}`. One-core configurations use CPU `{mapping['selected'][0]}`; "
                 f"eight-core configurations span sockets ({', '.join(f'{count} cores on socket {sock}' for sock, count in sorted(sockets.items()))}). "
                 "This is a cross-socket allocation, not a single-socket scaling experiment. "
                 + link("CPU mapping", job["directory"] / "cpu_mapping.json", output) + "; "
                 + link("lscpu", job["directory"] / "lscpu.out", output) + ".")
    if profile_job:
        same_map = mapping == profile_job["cpu_mapping"]
        parts.append(f"E5 uses {'the same' if same_map else 'a different'} recorded CPU mapping in a separate job after E4. "
                     "PySCF uses p BLAS threads; SLEPc uses p pinned MPI ranks with one BLAS/OpenMP thread each. "
                     "C++ OpenMP uses p bound threads; serial uses one core. Actual pools and affinities passed the per-call resource checks.")
    versions = [["Python", env["python"]]] + [[name, data["version"]] for name, data in env["packages"].items()]
    versions += [["Native PETSc", ".".join(map(str, env["petsc"]["version"]))],
                 ["Native SLEPc", ".".join(map(str, env["slepc"]["version"]))],
                 [env["mpi"]["vendor"], ".".join(map(str, env["mpi"]["vendor_version"]))]]
    pools = [p for p in env["threadpools"] if p.get("user_api") == "blas"]
    versions += [["OpenBLAS", ", ".join(f"{p['version']} ({p.get('threading_layer', 'unknown')} threading)" for p in pools)]]
    parts.append(table(["Software", "Recorded version"], versions))
    compiler = read_text(job["directory"] / "build/compiler.txt")
    compiler_version = next(line for line in compiler.splitlines() if line.startswith("g++ ("))
    parts.append(f"C++ compiler: `/usr/bin/g++`, `{compiler_version}`; no compiler module. "
                 "Flags: `-std=c++17 -O3 -march=native -fopenmp -Wall -Wextra -pedantic`. "
                 + link("Compiler evidence", job["directory"] / "build/compiler.txt", output) + ". "
                 "NumPy/SciPy and native PETSc/SLEPc resolve to the same loaded OpenBLAS provider: "
                 + ", ".join(f"`{p}`" for p in env["blas_alignment"]["openblas_paths"]) + ". "
                 "PySCF Davidson's matrix-vector callback uses NumPy; C++ uses its own kernels. "
                 "The source revision alone is insufficient for this uncommitted experiment; "
                 + link("source hashes", job["directory"] / "source.json", output) + " and retained binary hashes identify the executed code.")
    parts += ["## Solver settings", "The settings are aligned where supported, with the following remaining differences. Full API references and timing rationale are in "
              + link("design.md", ROOT / "docs/baselines/design.md", output) + "; per-call EPS views record the effective native configuration."]
    parts.append(table(["Setting", "C++ serial / OpenMP", "PySCF davidson1", "SLEPc GD"], [
        ["Problem / roots", "Real symmetric; four smallest", "Same", "HEP; four smallest real"],
        ["Convergence", "Absolute residual <1e-8", "Residual 1e-8; eigenvalue-change tol 1e-12", "EPS_CONV_ABS, 1e-8"],
        ["Iteration limit", "100", "100 cycles", "100"],
        ["Initial vectors", "Four columns from the shared dump; fresh each call", "Same copied columns", "Same distributed columns; reset/setInitialSpace each call"],
        ["Expansion", "Up to four corrections", "Up to four corrections", "Block four; double expansion and Krylov start disabled"],
        ["Subspace / restart", "Growing basis; no restart", "max_space=12, effective 24; native restart", "nev/ncv/mpd=4/24/24; restart 6 + previous 1; Ritz extraction"],
        ["Preconditioning", "Guarded r/(theta-Aii), floor 1e-8", "Same formula; lowest Ritz shift for all roots", "STPRECOND shift 0; PREONLY/Jacobi scales by Aii"],
        ["Dependence / orthogonalization", "Remaining/original norm threshold 1e-8", "lindep=1e-16", "Native BV classical GS, conditional refinement eta 0.7071; vector products"],
    ]))
    parts.append("All solvers read the same C++ seed-759 matrix/guess dump and share one NumPy `eigvalsh(A)[:4]` reference per size. "
                 "The strict gate requires convergence, matching input checksum, finite eigenpairs, eigenvalue error <1e-8, residual <1e-8 and Frobenius orthogonality error <1e-8; residual slack is 1.0. "
                 "Time covers only `Davidson`, `davidson1`, or SLEPc setup plus solve. SLEPc takes the maximum rank time with boundary barriers outside the timer. "
                 "Imports, input/assembly, references, validation and output are excluded; PySCF callbacks and PETSc logging overhead inside the calls are included.")
    parts.append(f"Each E4 configuration has one warm-up, then at least five measured solves and one accumulated timed second, capped at 200. "
                 f"{len(capped)} of {len(rows)} configurations hit the cap before one second; the others met the targets. "
                 "All accuracy checks include the warm-up. The appendix records every run count, min/max, cumulative time and stopping outcome.")
    parts += ["## Table 1: median solve time (ms)",
              "Every cell is **sweep 1 / sweep 2**, not an average or confidence interval. FAIL/MISSING is never treated as a valid timing. "
              + link("Per-sweep min–max, runs and C++ OpenMP-1 results", details, output) + " provide the full spread and provenance."]
    parts.append(table(["n", *(f"{LABELS[s]}-{p}" for s, p in COLUMNS)],
                       [[n, *(pair(index, n, s, p) for s, p in COLUMNS)] for n in sizes]))
    parts += ["## Table 2: iterations and matvec counts",
              "Ranges cover measured calls, both sweeps and all tested core counts within each solver family. "
              "C++ counts projection cycles and its accepted basis vectors; PySCF counts operator callbacks and vectors passed to the operator; SLEPc uses native iteration numbers. These iteration units differ."]
    count_rows = []
    for n in sizes:
        groups = [[r for r in rows if r["n"] == n and r["solver"].startswith("cpp_")],
                  [r for r in rows if r["n"] == n and r["solver"] == "pyscf"],
                  [r for r in rows if r["n"] == n and r["solver"] == "slepc"]]
        count_rows.append([n, *(f"{count_range(group, 'iterations')} / {count_range(group, 'matvecs')}" for group in groups)])
    parts.append(table(["n", "C++ iterations / matvecs", "PySCF iterations / matvecs", "SLEPc iterations / matvecs"], count_rows))
    parts += ["### Solve time per counted matvec (ms)",
              "Each cell is the recorded median of per-call solve-time/count, sweep 1 / sweep 2. It includes the whole solver, not isolated matrix-vector kernel time. SLEPc vector-equivalent counts and ratios remain unknown."]
    parts.append(table(["n", "C++ serial", "OpenMP-1", "OpenMP-8", "PySCF-1", "PySCF-8", "SLEPc-1/8"],
                       [[n, *(pair(index, n, s, p, "median_ms_per_matvec", 6) for s, p in ALL_COLUMNS[:5]), "—"] for n in sizes]))
    parts += ["## Table 3: accuracy and gates", "Worst values over all solvers, cores, sweeps and warm-up/measured calls at each n. "
              + link("Per-solver/configuration accuracy and failure reasons", details, output) + " are retained in the appendix."]
    accuracy_rows = []
    for n in sizes:
        subset = [r for r in rows if r["n"] == n]
        accuracy_rows.append([n, sci(maximum(subset, "max_eigenvalue_error")), sci(maximum(subset, "max_residual")),
                              sci(maximum(subset, "max_orth_error")), sum(not valid(r) for r in subset)])
    parts.append(table(["n", "Max eigenvalue error", "Max residual", "Max orthogonality", "Failed configurations"], accuracy_rows))
    if failures or missing:
        parts.append("**Failures / incomplete evidence:** " + "; ".join([f"{r['config_id']}: {r['failure_reason']}" for r in failures] + list(map(str, missing))))
    parts += ["## Observations", "### Scaling within each solver", "Values are one-core median / eight-core median, **sweep 1 / sweep 2**. Greater than 1 means faster on eight cores. C++ compares OpenMP-1 with OpenMP-8; serial references remain in Table 1."]
    parts.append(table(["n", "OpenMP speedup", "PySCF speedup", "SLEPc speedup"],
                       [[n, *(" / ".join(fmt(speedup(index, n, solver, sweep), 2) for sweep in (1, 2))
                              for solver in ("cpp_openmp", "pyscf", "slepc"))] for n in sizes]))
    if all((s, n, solver, p) in index for s in (1, 2) for n in (64, 256, 1024, 2048, 4096) for solver, p in ALL_COLUMNS) and not failures:
        p2048 = [index[(s, 2048, "pyscf", 8)]["median_ms"] for s in (1, 2)]
        p4096 = [index[(s, 4096, "pyscf", 8)]["median_ms"] for s in (1, 2)]
        slowdown = [1 / speedup(index, 1024, "pyscf", s) for s in (1, 2)]
        parts.append("- **Small sizes:** PySCF/SLEPc one-core medians stay near a few ms from n=64 to 256, while C++ is below 1 ms at n=64. "
                     "This pattern is consistent with fixed interpreter/framework work dominating small cases; these measurements do not isolate that overhead. Eight MPI ranks are slower than one at small n.\n"
                     "- **Large sizes:** at n=4096 SLEPc-8 is faster than PySCF-8 and C++ OpenMP-8 in both sweeps. "
                     "C++ and PySCF require different matvec counts, but counts alone do not explain performance: C++ performs fewer applications while its one-core solve is slower. Kernel implementations and other solver work differ.\n"
                     f"- **PySCF anomaly:** n=1024 is {fmt(slowdown[0], 2)}x / {fmt(slowdown[1], 2)}x slower with eight threads. "
                     f"At n=2048 the eight-thread medians change from {fmt(p2048[0])} to {fmt(p2048[1])} ms "
                     f"({fmt(p2048[1]/p2048[0], 2)}x); counts, settings and resource gates remain consistent. "
                     f"At n=4096 they change from {fmt(p4096[0])} to {fmt(p4096[1])} ms. Neither sweep is discarded or averaged away.")
    if profile_job:
        parts += ["### E5 diagnostic profiles", "These are separate, profiled calls, not additional timing repeats. "
                  "PySCF profiles exactly the measured `davidson1` call; warm-up, input and validation are excluded. "
                  + link("Profile artifacts and CSV row mapping", details, output) + "."]
        parts.append(table(["n", "PySCF cores", "Profiled solver calls", "Dot calls", "Dot self time / solver cumulative time"],
                           [[p["n"], p["cores"], p.get("davidson_calls", "FAIL"), p.get("dot_calls", "—"),
                             fmt(p.get("dot_fraction_percent"), 2) + "%" if p.get("dot_fraction_percent") is not None else "FAIL"]
                            for p in profiles if p["solver"] == "pyscf"]))
        parts.append("The profiles support matrix-vector work dominating PySCF at n=4096. At n=1024/eight threads, most recorded time is inside ndarray.dot; "
                     "this locates the slowdown but does not establish whether thread overhead, memory placement or shared-node interference caused it.")
        parts.append(table(["n", "SLEPc ranks", "Measured-stage raw MatMult count", "Rank-max MatMult event ms", "PETSc stage %T"],
                           [[p["n"], p["cores"], p.get("events", {}).get("MatMult", {}).get("count", "FAIL"),
                             fmt(p.get("events", {}).get("MatMult", {}).get("max_ms"), 4),
                             fmt(p.get("events", {}).get("MatMult", {}).get("stage_time_percent"), 0)]
                            for p in profiles if p["solver"] == "slepc"]))
        parts.append("Only `baseline_slepc_repeat1` is used above; the full PETSc logs also include calibration, warm-up and other activity. "
                     "PETSc events nest/overlap, so event times cannot be added. Rank maxima and printed stage percentages use different aggregation; percentages are taken directly from the log. "
                     "The larger MatMult share at n=4096 supports a greater matrix-operation contribution; small distributed solves have substantial other work. These event counts remain unverified vector equivalents.")
    parts += ["## Caveats and open questions",
              "- The eight-core set spans two sockets asymmetrically. Allocation/affinity checks passed, but page placement, bandwidth and competing jobs were not measured. A controlled single-socket/repeated-job comparison is needed to separate NUMA effects from other causes of PySCF instability.\n"
              "- MPI ranks and BLAS/OpenMP threads are different execution models. SLEPc's zero-shift Jacobi, native restart/locking/orthogonalization, and setup timing differ from C++/PySCF. No claim of algorithmic identity or library-best tuning is made.\n"
              "- Two sequential sweeps on one shared node are limited repeat evidence. The order is fixed, not randomized; capped small cases contain less than one timed second. Min/max and sweep differences describe observed spread, not statistical confidence intervals.\n"
              "- Residuals and orthogonality were recomputed by the runners outside timing. The analysis checks their saved per-call evidence and aggregates; returned eigenvectors are not available for another independent residual computation.\n"
              "- SLEPc vector-equivalent matvec totals remain unknown; null is not zero. Verify all native operator paths before using a matvec-normalized SLEPc comparison.\n"
              "- E5 has one profiled measured solve per configuration. Profiles can locate expensive work but do not prove the causes of timing variability. No failure or outlier has been silently removed.",
              "## Reproduce this report",
              "From the repository root (Python 3.10+ standard library; no solver execution or Euler access):",
              "```bash\npython3 -B analysis/baseline_report.py --e4 " + Path(os.path.relpath(job["directory"], ROOT)).as_posix()
              + (" \\\n    --e5 " + Path(os.path.relpath(profile_job["directory"], ROOT)).as_posix() if profile_job else "") + "\n```",
              "The generator validates CSV/raw consistency, repeat counts, accuracy aggregates and linked evidence before writing tables. "
              "Input hashes and audit counts are saved alongside the report. Returned results and solver/build scripts are unchanged."]
    return "\n\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e4", type=Path, required=True)
    parser.add_argument("--e5", type=Path)
    parser.add_argument("--e6", type=Path, help="primary single-socket results; also generates PNG/PDF figures and summary.md")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/baselines/report.md")
    args = parser.parse_args()
    if args.e6:
        from baseline_final import generate_final
        return generate_final(args)
    try:
        job = load_job(args.e4, "E4")
        profile_job = load_job(args.e5, "E5") if args.e5 else None
        if profile_job:
            # Profiles may use a separate allocation, but may not silently use different code/inputs.
            sources = [json.loads(read_text(j["directory"] / "source.json"))["sha256"] for j in (job, profile_job)]
            if sources[0] != sources[1]:
                raise ValueError("E4/E5 executed source hashes differ; review before combining diagnostic conclusions")
            for n in profile_job["manifest"]["arguments"]["sizes"]:
                for name in (f"reference_n{n}.bin", f"input_n{n}.bin"):
                    hashes = [input_fingerprint(j["directory"], n)["sha256"] if name.startswith("input_")
                              else hashlib.sha256(read_bytes(j["directory"] / name)).hexdigest()
                              for j in (job, profile_job)]
                    if hashes[0] != hashes[1]:
                        raise ValueError(f"E4/E5 input/reference mismatch: {name}")
        profiles = collect_profiles(profile_job) if profile_job else []
        output = args.output.resolve()
        details, audit_path = output.with_name("details.md"), output.with_name("report-audit.json")
        audit = {"schema_version": 1, "e4": job["audit"], "e5": profile_job["audit"] if profile_job else None,
                 "profile_metrics": profiles,
                 "generator_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in (Path(__file__), Path(__file__).with_name("baseline_evidence.py"))}}
        outputs = {output: build_report(job, profile_job, profiles, output, details, audit_path),
                   details: build_details(job, profile_job, profiles, details),
                   audit_path: json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n"}
        for path, text in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
        failed = any(not valid(r) for j in (job, profile_job) if j for r in j["rows"])
        incomplete = any(j["audit"].get("missing_expected_configurations") or j["manifest"].get("status") != "complete"
                         or not j["manifest"].get("ok") for j in (job, profile_job) if j)
        print(json.dumps({"report": str(output), "e4_configurations": len(job["rows"]),
                          "e5_configurations": len(profile_job["rows"]) if profile_job else 0,
                          "failed_or_incomplete": bool(failed or incomplete)}, sort_keys=True))
        return 1 if failed or incomplete else 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"Evidence/report error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
