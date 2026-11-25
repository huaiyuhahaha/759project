"""Publication/export figures from audited rows; never execute a solver."""

import hashlib
import json
import os
from pathlib import Path
import tempfile


LABELS = {"cpp_serial": "C++ serial", "cpp_openmp": "C++ OpenMP", "pyscf": "PySCF", "slepc": "SLEPc"}
COLORS = {"cpp_serial": "#333333", "cpp_openmp": "#0072B2", "pyscf": "#D55E00", "slepc": "#009E73"}


def make_figures(job, destination):
    # Keep font/cache state outside the repository and returned evidence.
    with tempfile.TemporaryDirectory(prefix="baseline-mpl-") as cache:
        old = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
            from matplotlib.ticker import ScalarFormatter
            return _draw(job, destination, matplotlib, plt, Line2D, ScalarFormatter)
        finally:
            if old is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = old


def _draw(job, destination, matplotlib, plt, Line2D, ScalarFormatter):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    index = {(r["sweep"], r["n"], r["solver"], r["cores"]): r for r in job["rows"]}
    sizes = job["manifest"]["arguments"]["sizes"]
    cores = job["manifest"]["arguments"]["cores"]
    job_id = job["manifest"]["job_id"]
    manifest = {"job_id": job_id, "matplotlib_version": matplotlib.__version__,
                "sweep_convention": "separate series; no pooling, smoothing or confidence intervals",
                "figures": {}, "series": []}
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
                         "legend.fontsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.dpi": 180, "axes.spines.top": False, "axes.spines.right": False})

    def value(sweep, n, solver, p, metric):
        row = index.get((sweep, n, solver, p))
        if not row or not row["gate_pass"] or not row["strict_gate_pass"] or row["exit_code"]:
            return None
        return row.get(metric)

    def series(ax, name, sweep, solver, p, xs, ys, ids):
        ax.plot(xs, [float("nan") if y is None else y for y in ys],
                color=COLORS[solver], linestyle="-" if sweep == 1 else "--",
                marker="o" if sweep == 1 else "s", markersize=4, linewidth=1.6,
                alpha=1 if sweep == 1 else .75, label=f"{LABELS[solver]} · sweep {sweep}")
        manifest["series"].append({"figure": name, "sweep": sweep, "solver": solver,
                                   "cores": p, "x": xs, "y": ys, "config_ids": ids})

    def save(fig, name, caption):
        paths = []
        for suffix in ("png", "pdf"):
            path = destination / f"{name}.{suffix}"
            metadata = {"Creator": "analysis/baseline_report.py", "CreationDate": None,
                        "ModDate": None} if suffix == "pdf" else {"Software": "analysis/baseline_report.py"}
            fig.savefig(path, bbox_inches="tight", metadata=metadata)
            paths.append({"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        manifest["figures"][name] = {"caption": caption, "files": paths}
        plt.close(fig)

    for p in (1, 8):
        name = f"time_vs_n_{p}core"
        fig, ax = plt.subplots(figsize=(8.5, 5.2), layout="constrained")
        for solver in LABELS:
            if p == 8 and solver == "cpp_serial":
                continue
            for sweep in (1, 2):
                series(ax, name, sweep, solver, p, sizes,
                       [value(sweep, n, solver, p, "median_ms") for n in sizes],
                       [f"s{sweep}_n{n}_{solver}_p{p}" for n in sizes])
        ax.set(xscale="log", yscale="log", xlabel="Matrix dimension n", ylabel="Median solve time (ms)",
               title=f"E6 job {job_id} · {p} physical core{'s' if p > 1 else ''}")
        ax.set_xticks(sizes)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.grid(True, which="both", alpha=.2)
        ax.legend(ncol=2)
        save(fig, name, f"Log-log solve-time medians on {p} cores; both sweeps shown separately.")

    for metric in ("speedup", "efficiency"):
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 4), sharey=True, layout="constrained")
        for ax, n in zip(axes, (1024, 2048, 4096)):
            for solver in ("cpp_openmp", "pyscf", "slepc"):
                for sweep in (1, 2):
                    one = value(sweep, n, solver, 1, "median_ms")
                    ys = []
                    for p in cores:
                        many = value(sweep, n, solver, p, "median_ms")
                        result = one / many if one is not None and many else None
                        ys.append(result / p if result is not None and metric == "efficiency" else result)
                    series(ax, metric, sweep, solver, None, cores, ys,
                           [f"s{sweep}_n{n}_{solver}_p{p}" for p in cores])
            ax.plot(cores, cores if metric == "speedup" else [1] * len(cores),
                    color="#777777", linestyle=":", linewidth=1.2)
            ax.set(title=f"n = {n}", xlabel="Physical cores", xticks=cores, ylim=(0, None))
            ax.grid(True, alpha=.2)
        axes[0].set_ylabel("Speedup T₁ / Tₚ" if metric == "speedup" else "Parallel efficiency (T₁ / Tₚ) / p")
        handles = [Line2D([0], [0], color=COLORS[s], label=LABELS[s]) for s in ("cpp_openmp", "pyscf", "slepc")]
        handles += [Line2D([0], [0], color="black", linestyle=style, marker=mark, label=f"Sweep {i}")
                    for i, style, mark in ((1, "-", "o"), (2, "--", "s"))]
        handles.append(Line2D([0], [0], color="#777777", linestyle=":", label="Ideal"))
        fig.legend(handles=handles, loc="outside lower center", ncol=6)
        fig.suptitle(f"E6 job {job_id} · single-socket {metric}")
        save(fig, metric, "Same-sweep one-core reference for each solver; efficiency is speedup / cores.")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), layout="constrained")
    name = "time_per_matvec"
    for ax, p in zip(axes, (1, 8)):
        for solver in ("cpp_serial", "cpp_openmp", "pyscf"):
            if p == 8 and solver == "cpp_serial":
                continue
            for sweep in (1, 2):
                series(ax, name, sweep, solver, p, sizes,
                       [value(sweep, n, solver, p, "median_ms_per_matvec") for n in sizes],
                       [f"s{sweep}_n{n}_{solver}_p{p}" for n in sizes])
        ax.set(xscale="log", yscale="log", xlabel="Matrix dimension n",
               ylabel="Median solve time / counted matvec (ms)", title=f"{p} physical core{'s' if p > 1 else ''}")
        ax.set_xticks(sizes)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(True, which="both", alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(f"E6 job {job_id} · whole solve normalized by counted vector applications")
    save(fig, name, "SLEPc omitted: vector-equivalent counts are unverified. This is not isolated matvec kernel time.")
    (destination / "figure-data.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest
