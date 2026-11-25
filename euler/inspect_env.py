#!/usr/bin/env python3
"""E2 environment evidence, run once (not through mpiexec).

Usage: <environment>/bin/python -B euler/inspect_env.py --output-dir <E2 directory>
The caller sets BLAS/OpenMP threads to one before startup. Nothing is installed.
environment.json is retained even when imports or alignment checks fail.
"""

import argparse
import contextlib
from datetime import datetime, timezone
import importlib
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import sys
import traceback


PACKAGES = ("numpy", "scipy", "pyscf", "petsc4py", "slepc4py", "mpi4py", "threadpoolctl")
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "BLIS_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES",
               "GOMP_CPU_AFFINITY", "KMP_AFFINITY")
LIBRARY_PATTERN = re.compile(r"blas|lapack|libmkl|libblis|libflexiblas|libmpi|libmpich|libpetsc|libslepc", re.I)


def failure(report, message):
    report["failures"].append(message)
    print(f"FAIL: {message}", file=sys.stderr, flush=True)


def capture(function):
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        function()
    return output.getvalue()


def mapped_libraries():
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            path = fields[5].removesuffix(" (deleted)")
            if LIBRARY_PATTERN.search(Path(path).name):
                paths.add(str(Path(path).resolve()))
    return sorted(paths)


def dependencies(path):
    """Retain ldd's complete evidence and resolved dependency paths."""
    command = ["ldd", str(path)]
    if not shutil.which("ldd"):
        return {"command": command, "error": "ldd is unavailable", "paths": []}
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error), "paths": []}
    paths = re.findall(r"(?m)^\s*(?:\S+\s+=>\s+)?(/[^\n(]+?)\s+\(", result.stdout)
    return {"command": command, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr,
            "paths": sorted({str(Path(p.strip()).resolve()) for p in paths})}


def inspect_dependencies(report, targets):
    """Inspect imported extensions and native PETSc/SLEPc libraries actually loaded.

    Standalone ldd on an extension does not inherit libraries already loaded by
    Python. Retain that diagnostic, but use the mapped native library for its
    BLAS dependency chain. Missing or ambiguous mappings cannot establish it.
    """
    provenance = {name: {"source": "imported module or package library", "path": str(path)}
                  for name, path in targets.items()}
    evidence = {name: dependencies(path) for name, path in targets.items()}
    for name in ("petsc", "slepc"):
        pattern = re.compile(rf"lib{name}\.so(?:\.\d+)*")
        candidates = sorted({path for path in report.get("mapped_libraries", [])
                             if pattern.fullmatch(Path(path).name)})
        status = "selected" if len(candidates) == 1 else "missing" if not candidates else "ambiguous"
        provenance[name] = {"source": "/proc/self/maps", "status": status,
                            "candidates": candidates, "path": candidates[0] if len(candidates) == 1 else None}
        if len(candidates) == 1:
            evidence[name] = dependencies(candidates[0])
        else:
            evidence[name] = {"error": f"{status} mapped native lib{name} library", "paths": []}
    report["dependency_targets"] = provenance
    report["dependencies"] = evidence


def check_blas(report):
    pools = [p for p in report.get("threadpools", []) if p.get("user_api") == "blas"]
    providers = sorted({str(Path(p["filepath"]).resolve()) for p in pools
                        if p.get("internal_api") == "openblas" and p.get("filepath")})
    others = [p for p in pools if p.get("internal_api") != "openblas"]
    evidence = report.get("dependencies", {})
    required = ("numpy", "scipy", "petsc", "slepc")
    matches = {name: sorted(set(evidence.get(name, {}).get("paths", [])) & set(providers))
               for name in required}
    reasons = []
    if not providers:
        reasons.append("no loaded OpenBLAS provider was identified")
    elif len(providers) != 1:
        reasons.append("multiple distinct OpenBLAS libraries are loaded")
    if others:
        reasons.append("a non-OpenBLAS BLAS provider is loaded")
    for name in required:
        item = evidence.get(name, {})
        if item.get("error"):
            reasons.append(f"{name} dependency inspection: {item['error']}")
        elif item.get("returncode") != 0 or "not found" in item.get("stdout", ""):
            reasons.append(f"{name} dependency inspection failed or contains missing libraries")
        elif len(providers) == 1 and matches[name] != providers:
            reasons.append(f"{name} dependencies do not establish use of the observed OpenBLAS")
    status = "verified_shared_loaded_dependency" if not reasons else "mismatch_or_unresolved"
    report["blas_alignment"] = {
        "status": status, "openblas_paths": providers, "other_blas_pools": others,
        "dependency_provider_matches": matches, "reasons": reasons,
        "required_dependency_targets": list(required),
        "scope": "NumPy/SciPy extension dependencies and uniquely mapped native PETSc/SLEPc "
                 "library dependencies after tiny BLAS operations; standalone PETSc/SLEPc "
                 "extension ldd results are supplemental diagnostics. "
                 "PySCF Davidson's matrix-vector callback uses NumPy. Extra PySCF library "
                 "dependencies and all detected thread pools are retained separately.",
    }
    for reason in reasons:
        failure(report, f"BLAS alignment: {reason}")


def inspect(report):
    modules = {}
    report["packages"] = {}
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            modules[name] = module
            report["packages"][name] = {"version": getattr(module, "__version__", None),
                                        "file": getattr(module, "__file__", None)}
        except Exception as error:
            failure(report, f"import {name}: {type(error).__name__}: {error}")

    np = modules.get("numpy")
    petsc = slepc = mpi = None
    for package, attribute in (("mpi4py", "MPI"), ("petsc4py", "PETSc"), ("slepc4py", "SLEPc")):
        if package not in modules:
            continue
        try:
            leaf = importlib.import_module(f"{package}.{attribute}")
            if attribute == "MPI":
                mpi = leaf
                if mpi.COMM_WORLD.size != 1:
                    raise RuntimeError("inspect_env.py must run as one process, without mpiexec")
            elif attribute == "PETSc":
                petsc = leaf
            else:
                slepc = leaf
        except Exception as error:
            failure(report, f"load {package}.{attribute}: {type(error).__name__}: {error}")
    if mpi is not None and mpi.COMM_WORLD.size != 1:
        # Only rank zero may write a diagnostic artifact on accidental multi-rank use.
        return mpi.COMM_WORLD.rank == 0

    targets = {}
    if mpi is not None:
        vendor, version = mpi.get_vendor()
        report["mpi"] = {"vendor": vendor, "vendor_version": version,
                         "library_version": mpi.Get_library_version(),
                         "standard_version": mpi.Get_version(),
                         "size": mpi.COMM_WORLD.size, "file": mpi.__file__}
        targets["mpi"] = mpi.__file__
        if vendor != "MPICH":
            failure(report, f"expected the environment's MPICH; mpi4py reports {vendor!r}")
    if petsc is not None:
        report["petsc"] = {"version": petsc.Sys.getVersion(),
                           "version_info": petsc.Sys.getVersionInfo(),
                           "config": modules["petsc4py"].get_config(), "file": petsc.__file__}
        targets["petsc_extension"] = petsc.__file__
        if np is not None:
            dtype = str(np.dtype(petsc.ScalarType))
            report["petsc"]["scalar_dtype"] = dtype
            if np.dtype(petsc.ScalarType) != np.dtype("float64"):
                failure(report, f"PETSc must use real float64 scalars; found {dtype}")
    if slepc is not None:
        report["slepc"] = {"version": slepc.Sys.getVersion(),
                           "config": modules["slepc4py"].get_config(), "file": slepc.__file__}
        targets["slepc_extension"] = slepc.__file__

    report["operations"] = {}
    if np is not None:
        report["numpy_show_config"] = capture(np.show_config)
        matrix = np.asfortranarray(np.arange(64, dtype=np.float64).reshape(8, 8) / 100 + np.eye(8))
        vector = np.ones(8)
        reference = matrix @ vector
        report["operations"]["numpy_dot"] = bool(np.all(np.isfinite(reference)))
        for name in ("numpy._core._multiarray_umath", "numpy.core._multiarray_umath"):
            try:
                targets["numpy"] = importlib.import_module(name).__file__
                break
            except ImportError:
                continue
        if "scipy" in modules:
            try:
                blas = importlib.import_module("scipy.linalg.blas")
                result = blas.dgemv(1.0, matrix, vector)
                report["operations"]["scipy_dgemv"] = bool(np.allclose(result, reference))
                targets["scipy"] = importlib.import_module("scipy.linalg._fblas").__file__
            except Exception as error:
                failure(report, f"SciPy BLAS exercise: {error}")
        if "pyscf" in modules:
            try:
                lib = importlib.import_module("pyscf.lib")
                report["pyscf_lib_num_threads"] = lib.num_threads()
                result = lib.dot(matrix, matrix)
                report["operations"]["pyscf_dot"] = bool(np.allclose(result, matrix @ matrix))
                lib_dir = Path(modules["pyscf"].__file__).parent / "lib"
                for path in sorted(lib_dir.glob("libnp_helper*.so")):
                    targets[f"pyscf_{path.name}"] = str(path)
            except Exception as error:
                failure(report, f"PySCF BLAS exercise: {error}")
        if petsc is not None:
            mat = x = y = None
            try:
                data = np.asfortranarray(matrix, dtype=petsc.ScalarType)
                mat = petsc.Mat().createDense(size=data.shape, array=data, comm=petsc.COMM_SELF)
                mat.assemble()
                x, y = mat.createVecRight(), mat.createVecLeft()
                x.set(1.0)
                mat.mult(x, y)
                report["operations"]["petsc_dense_matmult"] = bool(np.allclose(y.getArray(readonly=True), reference))
            except Exception as error:
                failure(report, f"PETSc dense MatMult exercise: {error}")
            finally:
                for obj in (y, x, mat):
                    if obj is not None:
                        obj.destroy()
    for name, ok in report["operations"].items():
        if not ok:
            failure(report, f"tiny {name} operation returned incorrect/non-finite values")

    if "threadpoolctl" in modules:
        report["threadpools"] = modules["threadpoolctl"].threadpool_info()
        for pool in report["threadpools"]:
            if pool.get("num_threads") != 1:
                failure(report, f"expected one thread in every loaded pool: {pool}")
    if report.get("pyscf_lib_num_threads", 1) != 1:
        failure(report, "PySCF lib.num_threads() is not one")
    try:
        report["mapped_libraries"] = mapped_libraries()
    except OSError as error:
        failure(report, f"cannot inspect Linux mapped libraries: {error}")
    inspect_dependencies(report, targets)
    check_blas(report)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "environment.json"
    if path.exists():
        parser.error(f"refusing to overwrite {path}; use a fresh output directory")
    report = {"record_type": "environment_check", "schema_version": 1,
              "utc": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname(),
              "job_id": os.environ.get("SLURM_JOB_ID"), "python": platform.python_version(),
              "executable": sys.executable, "prefix": sys.prefix, "platform": platform.platform(),
              "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "thread_environment": {name: os.environ.get(name) for name in THREAD_VARS},
              "failures": []}
    try:
        if not inspect(report):
            return 1
    except Exception as error:
        failure(report, f"environment inspection: {type(error).__name__}: {error}")
        report["traceback"] = traceback.format_exc()
    report["ok"] = not report["failures"]
    encoded = json.dumps(report, indent=2, sort_keys=True, default=str, allow_nan=False)
    path.write_text(encoded + "\n", encoding="utf-8")
    print(encoded, flush=True)
    print(f"Environment evidence: {path}", flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
