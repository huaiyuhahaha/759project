# Euler Davidson baseline design

The final study compares C++ serial/OpenMP, PySCF `davidson1`, and SLEPc generalized Davidson on the same dense symmetric inputs. [report.md](report.md) contains the generated results, hardware/software evidence and limitations; [summary.md](summary.md) is the slide version. [STATUS.md](STATUS.md) records the team handoff.

## Experiments retained

| Experiment | Job | Purpose | Core counts |
|---|---|---|---|
| E4 | 129538 | Earlier cross-socket comparison, two sweeps | 1, 8 |
| E5 | 129548 | Diagnostic PySCF/PETSc profiles; excluded from timing comparisons | 1, 8 |
| E6 | 130120 | Primary single-socket scaling, two sweeps | 1, 2, 4, 8 |

E4/E6 use n=64,128,256,512,1024,2048,4096 and four lowest eigenpairs. C++ serial is a one-core reference. E5 profiles n=1024/4096 with one warm-up and one measured call. Numerical solver sources, input/reference fingerprints, compiler identity and library versions agree between E4 and E6; placement-only harness changes are recorded separately.

## Inputs, settings and correctness

`bench/davidson_bench.cpp` generates a seed-759 matrix followed by four initial vectors, little-endian float64 in row-major order. Dump length is `8*(n*n+4*n)`. Every solver checks the shared input checksum. A one-thread NumPy `eigvalsh(A)[:4]` reference is computed once per size outside timing. Repeats restore the same initial state.

All solvers require four converged roots, maximum eigenvalue error <1e-8, every independently recomputed residual <1e-8, and Frobenius orthogonality error <1e-8. Residual slack is 1.0. Failed configurations remain visible and cannot supply valid speedups.

| Setting | C++ | PySCF | SLEPc |
|---|---|---|---|
| Problem / roots | Real symmetric, four smallest | Same | HEP, four smallest real |
| Absolute residual / iteration limit | 1e-8 / 100 | 1e-8 / 100 cycles; eigenvalue-change tolerance 1e-12 | EPS_CONV_ABS, 1e-8 / 100 |
| Initial space / expansion | Four shared vectors; one correction per unconverged root | Same; up to four corrections | setInitialSpace, initial size/block size 4; no double expansion or Krylov start |
| Subspace / restart | Growing basis; no explicit cap or restart | max_space=12, effective 24; native restart | nev/ncv/mpd=4/24/24; recorded restart 6 + previous 1; Ritz extraction |
| Preconditioning | Per-root guarded r/(theta-Aii), floor 1e-8 | Guarded denominator, lowest Ritz shift for every root | STPRECOND shift 0, PREONLY/Jacobi scales by Aii |
| Dependence / orthogonalization | Relative norm threshold 1e-8 | lindep=1e-16 | Native BV classical GS, conditional refinement eta 0.7071 |

These implementations are aligned baselines, not identical algorithms or claims of each library's best tuning. Actual settings, native EPS views and convergence reasons are retained with the results. SLEPc resets EPS, reattaches the operator and restores the initial vectors before each call; BV options are reapplied after reset.

## Timing and counts

Time only C++ `Davidson`, PySCF `davidson1`, or SLEPc `EPS.setUp + EPS.solve`. Exclude imports, process startup, matrix creation/assembly, file I/O, reference eigensolves, validation and output. PySCF callbacks, PETSc logging inside the call, and C++ internal profiling are included. SLEPc eigenvector extraction is outside timing; C++/PySCF return vectors from the timed API.

MPI uses a barrier before/after the timed region and retains the maximum rank elapsed time. Boundary barriers and the final reduction are outside timing. Each E4/E6 configuration has one warm-up, then at least five measured solves AND at least one accumulated timed second, capped at 200. Medians, minima, maxima, sample counts and stopping outcomes are retained; capped cases do not imply one second was measured. The two sweeps are never pooled.

C++ counts projection cycles and accepted basis-vector applications. PySCF counts operator callbacks and vectors passed to its operator. SLEPc reports native iteration numbers. PETSc event counts are diagnostic and do not establish total vector-equivalent applications across all operator paths. SLEPc matvec-normalized time remains unknown, not zero. Other time-per-matvec values normalize the whole solve, not an isolated kernel.

## Compiler, libraries and placement

Euler builds use absolute `/usr/bin/g++`, recorded GCC 14.3.1, with `-std=c++17 -O3 -march=native -fopenmp -Wall -Wextra -pedantic`. No compiler module is loaded or purged. The old gcc/14.3.0 module had incompatible C++ headers on the current platform; gnu15/15.2.0 remains a documented fallback, not an automatic selection.

Runtime inspection verifies NumPy and native PETSc use the same OpenBLAS provider (also shared by SciPy/SLEPc). Versions, threading variants, MPI implementation and actual worker masks are recorded. Setting an environment variable alone is not proof of the loaded library or affinity.

E4 selected CPUs 14,15,42-47 across two sockets. E6 requests 48 logical CPUs; on the recorded two-socket SMT2 host this provides enough physical cores to select eight on one socket. The runtime gate, not the allocation estimate, is authoritative. It groups sysfs physical cores and cross-checks lscpu socket/NUMA membership and core equivalence classes; lscpu core IDs can be renumbered. Select the socket with most allocated physical cores, then its largest NUMA domain; break ties by lowest ID. Use nested prefixes for p=1/2/4/8 and reject duplicated SMT siblings, inconsistent topology or insufficient cores.

E6 selected CPUs 3,4,6,13,16,17,18,19 on socket 0 / NUMA node 0. PySCF uses p BLAS threads within the selected mask; C++ OpenMP uses p singleton-bound workers; SLEPc uses p singleton-bound MPI ranks with one BLAS/OpenMP thread each. The driver and dumper are pinned before matrix allocation/first-touch. When available, numactl must successfully apply and verify CPU-node and memory-node binding. It was absent in E6, so the policy is pinned first-touch; actual resident-page locality was not measured.

MPICH Hydra may reset an inherited memory policy during rank binding. The per-rank wrapper therefore verifies the inherited singleton CPU, reapplies the selected policy after Hydra, verifies it, then executes SLEPc. All 210 E6 rank-placement records pass. Shared-node interference, fixed solver order, different allocations and separate job conditions prevent assigning every E4/E6 timing difference solely to NUMA.

## Retained evidence and regeneration

Small CSV/JSON summaries, software/hardware/compiler records, Slurm logs, reference values and per-configuration summaries are directly browsable in Git. Exact raw timing records and repeated EPS views are losslessly packed in each final job's `evidence.zip`; `package.json` records member hashes and the archive hash. The analysis reads these members without extraction. These original timing observations cannot be regenerated from solver source alone.

Generated matrix dumps and compiled executables are excluded. Input lengths and SHA-256 hashes were checked against the original bytes during packaging; later archive-only analysis compares those recorded fingerprints. This is explicitly distinguished from rehashing available input bytes. Duplicate transfer archives, environment installation logs, discovery/smoke results, local environments and editor settings are excluded from the team branch. The small installation package inventory is retained only as version provenance.

Run `python3 -B analysis/baseline_report.py --e6 results/euler/E6_130120 --e4 results/euler/E4_129538 --e5 results/euler/E5_129548` with Python 3.10+ and matplotlib. The script audits recorded evidence and generates the report, summary, appendices, audit and PNG/PDF figures; it does not execute solvers or install environments.

## API and methodology references

- [SLEPc EPS manual](https://slepc.upv.es/release/documentation/manual/eps.html), [EPSGD](https://slepc.upv.es/release/manualpages/EPS/EPSGD.html), and [slepc4py EPS API](https://slepc.upv.es/release/slepc4py/reference/slepc4py.SLEPc.EPS.html).
- [Absolute convergence](https://slepc.upv.es/release/manualpages/EPS/EPSSetConvergenceTest.html), [initial space](https://slepc.upv.es/release/manualpages/EPS/EPSSetInitialSpace.html), [dimensions](https://slepc.upv.es/release/manualpages/EPS/EPSSetDimensions.html), [block size](https://slepc.upv.es/release/manualpages/EPS/EPSGDSetBlockSize.html), [double expansion](https://slepc.upv.es/release/manualpages/EPS/EPSGDSetDoubleExpansion.html), and [GD restart](https://slepc.upv.es/release/manualpages/EPS/EPSGDSetRestart.html).
- [STPRECOND](https://slepc.upv.es/release/manualpages/ST/STPRECOND.html), [PETSc PREONLY](https://petsc.org/release/manualpages/KSP/KSPPREONLY/), [Jacobi](https://petsc.org/release/manualpages/PC/PCJACOBI/), and [EPSReset](https://slepc.upv.es/release/manualpages/EPS/EPSReset.html).
- [BV matmult method](https://slepc.upv.es/release/manualpages/BV/BVSetMatMultMethod.html), [PETSc logging](https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.Log.html) and [LogEvent](https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.LogEvent.html).
- [Linux NUMA policy](https://docs.kernel.org/admin-guide/mm/numa_memory_policy.html), [numactl manual](https://github.com/numactl/numactl/blob/master/numactl.8), and [MPICH 5.0.1 Hydra binding implementation](https://github.com/pmodels/mpich/blob/v5.0.1/src/pm/hydra/lib/tools/topo/hwloc/topo_hwloc.c).
