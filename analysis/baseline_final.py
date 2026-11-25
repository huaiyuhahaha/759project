"""Final E6-primary report, comparison tables and one-page slide summary."""

import hashlib
import json
from pathlib import Path
import sys

from baseline_evidence import compare_jobs, load_job
from baseline_environment import build_environment
from baseline_figures import make_figures
from baseline_storage import input_fingerprint, read_bytes, read_text
from baseline_report import (ROOT, LABELS, ALL_COLUMNS, build_details, collect_profiles,
                             count_range, fmt, index_rows, job_interval, link, maximum,
                             pair, row_link, sci, table, valid)

PARALLEL = ("cpp_openmp", "pyscf", "slepc")


def ratio(index, sweep, n, solver, cores):
    one, many = index.get((sweep, n, solver, 1)), index.get((sweep, n, solver, cores))
    return one["median_ms"] / many["median_ms"] if valid(one) and valid(many) and many["median_ms"] else None


def ratios(index, n, solver, cores, efficiency=False):
    values = [ratio(index, sweep, n, solver, cores) for sweep in (1, 2)]
    return " / ".join(fmt(value * 100 / cores if efficiency and value is not None else value,
                          2 if efficiency else 3) for value in values)


def main_table(job):
    index = index_rows(job)
    return table(["n", *(f"{LABELS[s]}-{p}" for s, p in ALL_COLUMNS)],
                 [[n, *(pair(index, n, s, p) for s, p in ALL_COLUMNS)]
                  for n in job["manifest"]["arguments"]["sizes"]])


def cross_table(e4, e6):
    old, new = index_rows(e4), index_rows(e6)
    rows = []
    for n in e6["manifest"]["arguments"]["sizes"]:
        for solver in PARALLEL:
            factors = []
            for sweep in (1, 2):
                a, b = old.get((sweep, n, solver, 8)), new.get((sweep, n, solver, 8))
                factors.append(fmt(a["median_ms"] / b["median_ms"] if valid(a) and valid(b) and b["median_ms"] else None))
            rows.append([n, LABELS[solver], pair(old, n, solver, 8), pair(new, n, solver, 8), " / ".join(factors)])
    return table(["n", "Solver, 8 cores", "E4 ms (cross-socket)", "E6 ms (single-socket)", "E4 / E6 time"], rows)


def scaling_table(job):
    index = index_rows(job)
    rows = []
    for n in (1024, 2048, 4096):
        for solver in LABELS:
            for p in ([1] if solver == "cpp_serial" else (1, 2, 4, 8)):
                rows.append([n, LABELS[solver], p, pair(index, n, solver, p),
                             ratios(index, n, solver, p), ratios(index, n, solver, p, efficiency=True)])
    return table(["n", "Solver", "Cores", "Median ms", "Speedup", "Efficiency (%)"], rows)


def findings(e4, e6):
    index, old = index_rows(e6), index_rows(e4)
    if any(not valid(r) for r in e6["rows"]) or any(not valid(index.get((s, n, solver, p))) for s in (1, 2) for n in (1024, 2048, 4096)
           for solver, p in ALL_COLUMNS):
        return ["Some required rows failed or are missing; no complete scaling or instability conclusion is drawn. See the gate and row-level tables."]
    py2048 = [index[(s, 2048, "pyscf", 8)]["median_ms"] for s in (1, 2)]
    old2048 = [old[(s, 2048, "pyscf", 8)]["median_ms"] for s in (1, 2)]
    slow1024 = " / ".join(fmt(1 / ratio(index, s, 1024, "pyscf", 8), 2) for s in (1, 2))
    return [
        f"**Correctness and placement:** all {len(e6['rows'])} E6 configurations pass strict numerical/resource gates; "
        f"{e6['audit']['measured_samples']:,} measured solves are retained. "
        f"{len(e6['audit']['capped_below_target'])} configurations hit the 200-call cap below one timed second. "
        "[Accuracy table](#accuracy) and [repeat details](details.md) document these limits.",
        f"**C++ OpenMP scales well at n=4096:** eight-core medians are {pair(index, 4096, 'cpp_openmp', 8)} ms, "
        f"with {ratios(index, 4096, 'cpp_openmp', 8)}× speedup and "
        f"{ratios(index, 4096, 'cpp_openmp', 8, True)}% efficiency against OpenMP-1. "
        "[Scaling table, n=4096/OpenMP](#scaling).",
        f"**SLEPc-8 is fastest among the tested eight-core solvers at n=4096:** "
        f"{pair(index, 4096, 'slepc', 8)} ms, with {ratios(index, 4096, 'slepc', 8)}× speedup "
        f"and {ratios(index, 4096, 'slepc', 8, True)}% efficiency. "
        "This compares the recorded solver configurations, which differ in algorithms/settings. "
        "[Main table, n=4096](#main-times); [scaling table](#scaling).",
        f"**Single-socket placement did not remove PySCF-8 instability.** At n=1024 it remains {slow1024}× slower "
        f"than PySCF-1. At n=2048 its medians are {pair(index, 2048, 'pyscf', 8)} ms, a "
        f"{fmt(max(py2048)/min(py2048), 2)}× sweep difference (E4: {fmt(max(old2048)/min(old2048), 2)}×). "
        f"At n=4096 PySCF-8 takes {pair(index, 4096, 'pyscf', 8)} ms versus PySCF-1's "
        f"{pair(index, 4096, 'pyscf', 1)} ms. Counts and recorded binding checks remain consistent; "
        "the cause is unresolved. [Main table](#main-times), [comparison table](#cross-socket) and [counts](#counts).",
        "**Placement comparisons differ by solver:** at n=4096 E6 OpenMP-8 and SLEPc-8 are faster than their E4 counterparts, "
        "while PySCF-8 is slower in both sweeps. The larger allocation, chosen cores and separate shared-node jobs "
        "also differ, so these are observed E4/E6 differences rather than isolated causal NUMA effects. "
        "[Comparison table, n=4096](#cross-socket).",
    ]


def settings_section(job, output):
    examples = [next(r for r in job["rows"] if r["solver"] == s and r["cores"] == 1) for s in ("cpp_serial", "pyscf", "slepc")]
    return "\n\n".join([
        "## Problem, solver settings and timing",
        "The task is to compute the four smallest eigenpairs of each real, dense symmetric n×n matrix. "
        "All solvers read the same seed-759 C++ matrix and four-vector initial-guess dump. "
        "Each size shares one NumPy `eigvalsh(A)[:4]` reference across both sweeps. "
        "The strict gates require convergence, identical input checksums, eigenvalue error <1e-8, "
        "every recomputed residual <1e-8, and Frobenius orthogonality error <1e-8; residual slack is 1.0.",
        table(["Setting", "C++ serial / OpenMP", "PySCF davidson1", "SLEPc GD"], [
            ["Roots / problem", "Four smallest; real symmetric", "Same", "HEP; four smallest real"],
            ["Convergence / limit", "Absolute residual 1e-8; 100 iterations", "Residual 1e-8; eigenvalue-change 1e-12; 100 cycles", "EPS_CONV_ABS 1e-8; 100 iterations"],
            ["Initial state", "Same four vectors; fresh per call", "Same copied vectors; fresh per call", "Same distributed vectors; reset/setInitialSpace per call"],
            ["Expansion", "One correction per unconverged root", "Up to four corrections", "Block four; double expansion/Krylov start off"],
            ["Subspace / restart", "Growing basis; no explicit cap/restart", "max_space=12; effective 24; native restart", "nev/ncv/mpd=4/24/24; restart 6 + previous 1"],
            ["Preconditioner", "Guarded per-root r/(theta-Aii), floor 1e-8", "Guarded denominator; lowest Ritz shift for every root", "Zero-shift STPRECOND/PREONLY/Jacobi; scales by Aii"],
            ["Dependence / orthogonalization", "Remaining/original norm threshold 1e-8", "lindep=1e-16", "Native BV classical GS; conditional refinement eta 0.7071"],
            ["Timed region", "Davidson call (existing internal profiling included)", "davidson1 call including callbacks", "EPS.setUp + EPS.solve; maximum elapsed over ranks"],
        ]),
        "Recorded settings: " + ", ".join(row_link(job, r, output) for r in examples)
        + "; per-call EPS views are linked from these records. API rationale and remaining differences: "
        + link("design.md", ROOT / "docs/baselines/design.md", output) + ".",
        "Input generation, file I/O, MPI matrix assembly, references, validation and output are outside solve timing. "
        "MPI boundary barriers are also outside the timer; the maximum rank elapsed time is retained. "
        "PETSc event logging inside the call remains included. Each configuration performs one warm-up, then "
        "at least five measured solves AND at least one accumulated timed second, capped at 200 measured solves. "
        "Each repeat starts from the same initial state. Warm-ups contribute to correctness checks but not timing medians.",
    ])


def build_final_report(e4, e6, e5, profiles, output, audit_path, figures):
    index, rows = index_rows(e6), e6["rows"]
    parts = ["# Davidson baselines on Euler: final report",
             f"**Primary result: E6 job {e6['manifest']['job_id']}, single-socket physical-core scaling.** "
             f"The experiment compares C++ serial/OpenMP, PySCF and SLEPc over n=64–4096 with 1/2/4/8 cores. "
             f"E4 job {e4['manifest']['job_id']} supplies the earlier cross-socket comparison. "
             "All tables and figures are generated from returned artifacts by `analysis/baseline_report.py`; no local smoke timings are included.",
             "**Reading the numbers:** every paired cell is **sweep 1 / sweep 2**. These are medians of separate solve repetitions; "
             "samples are not pooled and a pair is not a confidence interval. FAIL/MISSING never contributes a valid timing. "
             + link("Complete E6 rows, ranges, counts and evidence", output.with_name("details.md"), output) + "; "
             + link("E4 row-level comparison", output.with_name("e4-details.md"), output) + "; "
             + link("machine-readable audit", audit_path, output) + ".",
             "## Jobs and validation"]
    job_rows = []
    for job in (e6, e4, e5):
        if job:
            interval, duration = job_interval(job)
            a = job["audit"]
            job_rows.append([a["step"], a["job_id"], a["configurations"], a["measured_samples"], a["warmups"],
                             len(a["failed_configurations"]), len(a["missing_expected_configurations"]), fmt(duration, 0),
                             link("CSV", job["directory"] / "summary.csv", output)])
    parts.append(table(["Step", "Job", "Configs", "Measured solves", "Warm-ups", "Failed", "Missing", "Wall seconds", "Evidence"], job_rows))
    parts.append(f"E6 has {len(e6['audit']['capped_below_target'])} capped configurations below one timed second; "
                 f"E4 has {len(e4['audit']['capped_below_target'])}. They remain in the tables with their repeat outcomes in the appendices. "
                 "The audit cross-checks CSV aggregates, local/combined raw samples, strict gates, input/reference bytes, "
                 "placement and rank records. E4/E6 numerical source hashes, build identity/flags, library versions, "
                 "inputs/references and aligned settings agree; the placement harness and affinity settings intentionally differ.")
    if (e6["directory"] / "package.json").is_file():
        parts.append("**Retained evidence:** small CSV/JSON, compiler/hardware records and configuration summaries remain directly browsable. "
                     "Original per-call timing records and EPS views are losslessly retained in each job's `evidence.zip`; "
                     "the reader validates the archive hash and reads members without extraction. Large generated input dumps and compiled executables "
                     "are excluded from the team branch. Input byte lengths and SHA-256 values were verified during packaging and retained "
                     "in `package.json`; a checkout without those dumps compares the recorded fingerprints instead of rehashing absent bytes. "
                     "Reference binaries remain available. " + link("E6 packaging manifest", e6["directory"] / "package.json", output)
                     + "; " + link("E6 raw evidence", e6["directory"] / "evidence.zip", output) + ".")
    warnings = [w for j in (e6, e4, e5) if j for w in j["audit"]["warnings"]]
    if warnings:
        parts.append("**Evidence warnings:** " + "; ".join(warnings))
    parts += [build_environment(e6, output), settings_section(e6, output),
              '<a id="main-times"></a>\n\n## Main table: E6 median solve time (ms)',
              "Single-socket 1- and 8-core results; C++ serial is only a one-core reference. "
              "All seven sizes appear below; the 2/4-core measurements are included in scaling and the full appendix.", main_table(e6),
              '<a id="cross-socket"></a>\n\n## Secondary table: E4 versus E6 at eight cores',
              "E4 selected CPUs `[14,15,42,43,44,45,46,47]` (two physical cores on socket 0, six on socket 1). "
              "E6 uses one socket/NUMA node, as verified above. The ratio E4/E6 >1 means E6 was faster. "
              "Sweep numbers align presentation only: these were separate jobs, not simultaneous paired controls. "
              "No sole causal attribution to socket placement is justified.", cross_table(e4, e6),
              '<a id="scaling"></a>\n\n## Scaling: speedup and parallel efficiency',
              "For each solver and sweep, speedup is its one-core median divided by its p-core median; "
              "efficiency is speedup/p, shown as a percentage. OpenMP uses **OpenMP-1**, not C++ serial, as its denominator. "
              "Serial has only a one-core point; multi-core speedup is not applicable.", scaling_table(e6),
              '<a id="counts"></a>\n\n## Iterations, matvec counts and time per matvec',
              "Ranges span measured solves, both sweeps and all tested core counts within each solver family. "
              "C++ counts projection cycles and accepted basis-vector applications; PySCF counts operator callbacks "
              "and vectors passed to the operator; SLEPc reports native iteration numbers. Iteration units differ."]
    count_rows = []
    for n in e6["manifest"]["arguments"]["sizes"]:
        groups = [[r for r in rows if r["n"] == n and r["solver"].startswith("cpp_")],
                  [r for r in rows if r["n"] == n and r["solver"] == "pyscf"],
                  [r for r in rows if r["n"] == n and r["solver"] == "slepc"]]
        count_rows.append([n, *(f"{count_range(g, 'iterations')} / {count_range(g, 'matvecs')}" for g in groups)])
    parts += [table(["n", "C++ iterations / matvecs", "PySCF iterations / matvecs", "SLEPc iterations / matvecs"], count_rows),
              "**SLEPc matvec totals are not verified vector-equivalent counts.** Native PETSc event counts may omit "
              "other operator paths and cannot be summed across ranks as vector applications. Unknown totals and "
              "time-per-matvec remain **—**, never zero. The following values are medians of each measured call's "
              "whole solve time / counted matvec; they are not isolated matrix-vector kernel timings.",
              table(["n", *(f"{LABELS[s]}-{p}" for s, p in ALL_COLUMNS)],
                    [[n, *(pair(index, n, s, p, "median_ms_per_matvec", 6) for s, p in ALL_COLUMNS)]
                     for n in e6["manifest"]["arguments"]["sizes"]]),
              '<a id="accuracy"></a>\n\n## Accuracy',
              "Worst recorded values over warm-ups, measured solves, solvers, core counts and sweeps at each n; "
              "all three error thresholds are strictly <1e-8."]
    parts.append(table(["n", "Max eigenvalue error", "Max residual", "Max orthogonality error", "Failed configs"],
                       [[n, *(sci(maximum([r for r in rows if r["n"] == n], metric))
                               for metric in ("max_eigenvalue_error", "max_residual", "max_orth_error")),
                         sum(not valid(r) for r in rows if r["n"] == n)]
                        for n in e6["manifest"]["arguments"]["sizes"]]))
    parts += ["## Figures", "Lines show the two sweeps separately; no smoothing or confidence band is fitted. "
              "The plotted values and configuration IDs are retained in "
              + link("figure-data.json", output.parent / "figures/figure-data.json", output) + "."]
    for name, figure in figures["figures"].items():
        png, pdf = (output.parent / "figures" / f"{name}.{ext}" for ext in ("png", "pdf"))
        parts += ["!" + link(figure["caption"], png, output), figure["caption"] + " " + link("PDF", pdf, output) + "."]
    parts += ["## Findings", "\n".join("- " + item for item in findings(e4, e6))]
    if e5:
        parts += ["## E5 profiles: diagnostic context", "E5 was a separate cross-socket job with one profiled measured solve "
                  "per configuration. Its timings never enter E6 or E4 timing/scaling tables. PySCF cProfile covers "
                  "the measured davidson1 call; PETSc's full-process log also contains calibration and warm-up. "
                  "Only the named `baseline_slepc_repeat1` stage is extracted below. "
                  + link("Profile artifacts and CSV rows", output.with_name("e4-details.md"), output) + ".",
                  table(["n", "PySCF cores", "Profiled solver calls", "Dot calls", "Dot self / solver cumulative (%)"],
                        [[p["n"], p["cores"], p.get("davidson_calls", "FAIL"), p.get("dot_calls", "—"),
                          fmt(p.get("dot_fraction_percent"), 2)] for p in profiles if p["solver"] == "pyscf"]),
                  table(["n", "SLEPc ranks", "Raw MatMult event count", "Rank-max event ms", "Printed stage %T"],
                        [[p["n"], p["cores"], p.get("events", {}).get("MatMult", {}).get("count", "—"),
                          fmt(p.get("events", {}).get("MatMult", {}).get("max_ms"), 4),
                          fmt(p.get("events", {}).get("MatMult", {}).get("stage_time_percent"), 0)]
                         for p in profiles if p["solver"] == "slepc"]),
                  "Events can nest/overlap; do not add event times. Rank maxima and stage percentages have different "
                  "aggregation rules, so the percentage is read directly from PETSc's log. Profiles locate work "
                  "in E5 but do not establish the cause of the E6 PySCF anomaly."]
    parts += ["## Limits and reproducibility",
              "- E6 used a shared node, a larger allocation than E4, fixed solver order and two sequential sweeps. "
              "Competing work, bandwidth and physical page residency were not measured. Pinned first-touch is a placement policy, not proof of page locality.\n"
              "- MPI ranks and OpenMP/BLAS threads are different execution models. Preconditioning, restart/locking/orthogonalization "
              "and setup timing differ; these are aligned baselines, not identical algorithms or claims of each library's best tuning.\n"
              "- Min/max and sweep differences describe observed spread, not confidence intervals. Short capped cases and all outliers are retained.\n"
              "- Runners recomputed residuals/orthogonality outside timing. Analysis validates saved evidence and aggregates; "
              "eigenvectors were not returned for another independent residual computation.\n"
              "- Missing hardware/software values are labeled not recorded. SLEPc vector-equivalent counts remain unresolved.",
              "Regenerate locally with Python 3.10+ and matplotlib; no solver execution or Euler connection is needed:",
              "```bash\npython3 -B analysis/baseline_report.py --e6 " + e6["directory"].relative_to(ROOT).as_posix()
              + " \\\n    --e4 " + e4["directory"].relative_to(ROOT).as_posix()
              + (" \\\n    --e5 " + e5["directory"].relative_to(ROOT).as_posix() if e5 else "") + "\n```",
              "The generator writes this report, summary.md, E6/E4 row appendices, report-audit.json and PNG/PDF figures. "
              "Artifact contradictions stop generation; scientific failures remain explicit rows. Returned results are read only."]
    return "\n\n".join(parts) + "\n"


def build_summary(e4, e6, output):
    index = index_rows(e6)
    if any(not valid(r) for r in e6["rows"]) or e6["audit"]["missing_expected_configurations"]:
        return ("# Euler Davidson baselines — incomplete evidence\n\n" + main_table(e6)
                + "\n\nFailed/missing configurations are retained; see report.md and details.md before drawing scaling conclusions.\n")
    py = [index[(s, 2048, "pyscf", 8)]["median_ms"] for s in (1, 2)]
    a = e6["audit"]
    parts = ["# Euler Davidson baselines — slide summary",
             f"**E6 job {a['job_id']}:** four lowest eigenpairs, n=64–4096; nested 1/2/4/8 physical cores "
             f"on socket {e6['cpu_mapping']['socket']} / NUMA {e6['cpu_mapping']['numa_node']}. "
             "Numactl unavailable: pinned first-touch. **Median solve times below are ms, sweep 1 / sweep 2.**",
             main_table(e6),
             f"- **All {len(e6['rows'])} configurations passed:** {a['measured_samples']:,} measured solves; "
             f"{len(a['capped_below_target'])} configurations reached the 200-call cap below one timed second. [Accuracy/repeats](report.md#accuracy).\n"
             f"- **OpenMP, n=4096:** {ratios(index, 4096, 'cpp_openmp', 8)}× speedup and "
             f"{ratios(index, 4096, 'cpp_openmp', 8, True)}% efficiency at eight cores. [Scaling](report.md#scaling).\n"
             f"- **SLEPc-8 is fastest at n=4096** in these settings: {pair(index, 4096, 'slepc', 8)} ms; "
             f"{ratios(index, 4096, 'slepc', 8)}× speedup. Matvec-normalized SLEPc timing remains unverified. [Counts](report.md#counts).\n"
             f"- **PySCF instability persists:** n=2048 PySCF-8 differs {fmt(max(py)/min(py), 2)}× between sweeps; "
             "at n=1024 and 4096 eight threads are slower than one in both sweeps. [Main table](report.md#main-times).\n"
             "- **E4→E6 at n=4096:** OpenMP-8 and SLEPc-8 improved; PySCF-8 worsened. Separate shared-node jobs "
             "do not isolate a causal NUMA effect. [Comparison](report.md#cross-socket).",
             "[Full report, setup and caveats](report.md) · [E6 per-row evidence](details.md) · "
             "[E4 comparison evidence](e4-details.md) · [PNG/PDF figures](report.md#figures)."]
    return "\n\n".join(parts) + "\n"


def generate_final(args):
    try:
        e4, e6 = load_job(args.e4, "E4"), load_job(args.e6, "E6")
        e5 = load_job(args.e5, "E5") if args.e5 else None
        comparison = compare_jobs(e4, e6)
        if e5:
            if json.loads(read_text(e4["directory"] / "source.json"))["sha256"] != json.loads(read_text(e5["directory"] / "source.json"))["sha256"]:
                raise ValueError("E4/E5 source hashes differ")
            for n in e5["manifest"]["arguments"]["sizes"]:
                for name in (f"input_n{n}.bin", f"reference_n{n}.bin"):
                    hashes = [input_fingerprint(j["directory"], n)["sha256"] if name.startswith("input_")
                              else hashlib.sha256(read_bytes(j["directory"] / name)).hexdigest() for j in (e4, e5)]
                    if hashes[0] != hashes[1]:
                        raise ValueError(f"E4/E5 input/reference mismatch: {name}")
        profiles = collect_profiles(e5) if e5 else []
        output = args.output.resolve()
        figures = make_figures(e6, output.parent / "figures")
        audit_path = output.with_name("report-audit.json")
        audit = {"schema_version": 2, "primary": "e6", "e4": e4["audit"], "e6": e6["audit"],
                 "e5": e5["audit"] if e5 else None, "e4_e6_comparison": comparison,
                 "profile_metrics": profiles, "figures": figures,
                 "generator_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in Path(__file__).parent.glob("baseline_*.py")}}
        outputs = {output: build_final_report(e4, e6, e5, profiles, output, audit_path, figures),
                   output.with_name("details.md"): build_details(e6, None, [], output.with_name("details.md")),
                   output.with_name("e4-details.md"): build_details(e4, e5, profiles, output.with_name("e4-details.md")),
                   output.with_name("summary.md"): build_summary(e4, e6, output),
                   audit_path: json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n"}
        for path, text in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
        failed = any(j["audit"]["failed_configurations"] or j["audit"]["missing_expected_configurations"]
                     or j["manifest"].get("status") != "complete" or not j["manifest"].get("ok") for j in (e4, e6, e5) if j)
        print(json.dumps({"report": str(output), "e6_configurations": len(e6["rows"]),
                          "figures": len(figures["figures"]), "failed_or_incomplete": bool(failed)}))
        return int(bool(failed))
    except (OSError, ValueError, KeyError, ImportError) as error:
        print(f"Evidence/report error: {error}", file=sys.stderr)
        return 2
