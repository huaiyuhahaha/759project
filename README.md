# ECE 759 Parallel Davidson Eigensolver

A C++17 Davidson eigensolver for the lowest few eigenpairs of a dense real symmetric matrix. It has two backends: a serial baseline and an OpenMP version. Both are checked against NumPy and benchmarked against PySCF's Davidson solver.

## What it computes

`Davidson(A, guess, nroots, tol, max_iter, backend)` ([davidson.h](davidson.h)) returns the `nroots` smallest eigenvalues λ₁ ≤ … ≤ λ_k of a symmetric n×n matrix `A`, along with orthonormal eigenvectors X (n×k).

It stops when every root satisfies ‖A x_k − λ_k x_k‖₂ < `tol`. The result also reports:
- the convergence flag;
- the number of iterations and expansions;
- the number of dropped corrections;
- the final subspace size;
- the residual norms;
- per-kernel timings (`DavidsonProfile`).

```cpp
#include "davidson.h"
Matrix guess = initialize_vector(4, A);   // unit vectors at the 4 smallest diagonal entries
DavidsonResult r = Davidson(A, guess, 4, 1e-8, 100, Backend::OpenMP);  // default: Backend::Serial
```

## The Davidson loop ([davidson.cpp](davidson.cpp))

1. **Initial guesses.** Each column of `guess` is orthonormalized against the basis V using two passes of Gram–Schmidt. A vector is dropped as dependent if its remaining norm is ≤ 1e-8 × its original norm (`kLinDepTol`). Each kept vector v gets one matvec, and A·v is stored, so A is applied only once per basis vector.
2. **Projection.** The loop forms H = Vᵀ(AV) (m×m) from the stored products.
3. **Jacobi diagonalization.** `MatrixJacobi` runs classical Jacobi rotations on H:
   - it rotates out the largest off-diagonal entry each time;
   - it stops when every off-diagonal entry is ≤ 1e-12 · max|H|, or after 50·m² rotations;
   - it accumulates the rotations as eigenvectors.

   The Ritz values are then sorted in ascending order.
4. **Residual check.** For each of the `nroots` lowest Ritz pairs (θ, y), the loop computes:
   - x = V y;
   - A x = (AV) y;
   - r = A x − θ x.

   It stops, converged, when every ‖r‖₂ < `tol`.
5. **Subspace expansion.** Each unconverged root gets a diagonal-preconditioned correction δᵢ = rᵢ / (θ − Aᵢᵢ). Any denominator with |θ − Aᵢᵢ| < 1e-8 is clamped to ±1e-8, keeping its sign.
   - Corrections are orthonormalized as in step 1, and dependent ones are dropped.
   - If every correction is dropped, the raw residuals are tried instead.
   - If nothing new can be added, the solver stops unconverged. It also stops unconverged after `max_iter` iterations.
   - There is no restart, so the subspace grows by at most `nroots` vectors per iteration.

## What OpenMP parallelizes

Only the matvec A·v (`matvec_omp`): `#pragma omp parallel for schedule(static)` over the rows of A. Each row's dot product is computed by one thread in the same order as the serial loop, so the OpenMP results are bitwise identical to serial.

Everything else stays serial and is shared by both backends: Gram–Schmidt, projection, Jacobi, Ritz vectors, residuals and the preconditioner. The thread count comes from `OMP_NUM_THREADS`.

The matvec was chosen from measurements. On the benchmark matrices it takes 96% / 98% / 99% of the serial solve at n = 512 / 1024 / 2048.

## Solvers in this repo

| Name | Role | Where |
|---|---|---|
| C++ serial | Validated baseline (`Backend::Serial`) | [davidson.cpp](davidson.cpp) |
| C++ OpenMP | Parallel backend; threads only the matvec (`Backend::OpenMP`) | [davidson.cpp](davidson.cpp) |
| PySCF `davidson1` | Outside Davidson implementation for comparison; timed | [bench/pyscf_davidson_bench.py](bench/pyscf_davidson_bench.py) |
| NumPy `eigvalsh` | Exact reference eigenvalues for checking results; never timed | [checks/run_davidson_e2e.py](checks/run_davidson_e2e.py), [bench/run_davidson_bench.py](bench/run_davidson_bench.py) |

**Status:**
- **Implemented:** the serial and OpenMP backends, the E2E checks, and the benchmark.
- **Planned:** a CUDA backend. There is no CUDA code yet.
- **Optional extension:** atomic SCF built on this solver. Not implemented.

## Layout

```
matrix.h/.cpp                  dense row-major Matrix
davidson.h/.cpp                initialize_vector, MatrixJacobi, orthonormalize_against, Davidson
checks/davidson_e2e.cpp        E2E check program (n=8, n=64, and the drop-criterion check)
checks/run_davidson_e2e.py     builds it and runs every case on both backends
bench/davidson_bench.cpp       C++ timing program; also writes benchmark inputs for PySCF
bench/pyscf_davidson_bench.py  PySCF davidson1 runner
bench/run_davidson_bench.py    builds and runs the full benchmark and prints the tables
```

There is no Makefile: the two `run_*.py` scripts build their own binaries with `g++` in a temporary directory.

## Prerequisites (tested versions)

- Linux, because the scripts use `os.sched_getaffinity`. Tested on WSL2 Ubuntu 22.04.5.
- `g++` with C++17 and OpenMP: tested with g++ 11.4.0. The sanitizer check also needs ASan/UBSan, which come with Ubuntu's g++.
- Python 3 with NumPy for the checks: tested with Python 3.10.12 and NumPy 2.1.1.
- For the benchmark, also PySCF (tested 2.15.0, with SciPy 1.14.1 installed). threadpoolctl (tested 3.8.0) is optional; it only records thread-pool sizes.

```bash
python3 -m pip install --user numpy pyscf threadpoolctl
```

## Commands

Run from the repo root in a Linux/WSL shell. Each command exits 0 only if everything passes.

```bash
python3 checks/run_davidson_e2e.py              # E2E checks, -O2, serial and OpenMP backends
python3 checks/run_davidson_e2e.py --sanitize   # same checks under ASan + UBSan, with leak detection
python3 bench/run_davidson_bench.py             # full benchmark (options: --sizes, --threads, --pyscf-threads)
```

From Windows, prefix each command with `wsl -d Ubuntu --cd "<repo path>" --`.

## Correctness: E2E n = 64

The test matrix is tridiagonal, with diagonal 1..64 and both neighbouring diagonals 0.2.
- Settings: k = 2, guess `initialize_vector(2, A)`, solver tolerance 1e-10.
- Reference: NumPy `eigvalsh(A)[:2]`.
- Pass criteria: every error below 1e-8, and the final subspace must be smaller than 64.

The serial and OpenMP backends gave identical results in both the normal and sanitizer builds:

| Converged | Iterations | Expansions | Dropped corrections | Final subspace | Max \|λ − eigvalsh\| | Max ‖Ax − λx‖₂ | ‖XᵀX − I‖_F |
|---|---|---|---|---|---|---|---|
| yes | 9 | 8 | 7 | 10 | 0.0 | 1.51e-11 | 1.13e-15 |

The same script also runs:
- an n = 8 case on the same type of matrix;
- a unit check that `orthonormalize_against` drops a vector exactly when its remaining/original norm ratio is ≤ 1e-8, at scales 1e-6, 1 and 1e6.

The "at least one dropped correction" assertion applies only to these tridiagonal test matrices; it is not a general property of the solver.

## Performance snapshot

**Setup:**
- Machine: AMD Ryzen 7 7745HX (8 cores / 16 threads), WSL2 Ubuntu 22.04.
- Compiler flags: g++ 11.4.0, `-O3 -march=native -fopenmp`.
- Matrices: dense, with Aᵢᵢ = i + 1 and Aᵢⱼ = Aⱼᵢ = u/n, where u is uniform in [−1, 1) from `mt19937_64` with seed 759. This makes them strictly diagonally dominant.
- Solve: k = 4, guess `initialize_vector(4, A)`, residual tolerance 1e-8.
- Timing: each configuration runs in its own process, with one warm-up solve and then five timed solves. Only the solve is timed, and the table shows the median.
- Inputs: PySCF reads the exact matrix and guess bytes that the C++ benchmark writes, and checksums confirm the bytes match.

| n | C++ serial | C++ OpenMP-8 | PySCF-1 | PySCF-8 |
|---|---|---|---|---|
| 512 | 10.09 | 1.48 | 4.21 | 4.08 |
| 1024 | 34.23 | 4.14 | 6.59 | 5.06 |
| 2048 | 125.71 | 14.93 | 26.91 | 10.83 |

All times are median ms. "-1" and "-8" are thread counts.

**Speedups**, listed for n = 512 / 1024 / 2048:
- C++ OpenMP-8 vs C++ serial: 6.8× / 8.3× / 8.4×.
- PySCF-1 vs C++ serial: 2.4× / 5.2× / 4.7×.
- PySCF-8 vs PySCF-1: 1.03× / 1.30× / 2.48×.
- C++ OpenMP-8 vs PySCF-8: C++ is 2.76× and 1.22× faster at n = 512 and 1024. At n = 2048, PySCF-8 is 1.38× faster than C++ OpenMP-8.

**Correctness in this run:**
- Every solve converged; all four of PySCF's convergence flags were True.
- Independently recomputed max residual ≤ 8.3e-9, max |λ − `eigvalsh`| ≤ 2.0e-13, and ‖XᵀX − I‖_F ≤ 5.0e-15.
- C++ OpenMP eigenvalues were bitwise identical to serial.

**Why the numbers look like this (these matrices only):**
- C++ converges in 3–4 iterations (12–15 matvecs). PySCF takes 12 iterations (31–32 matvecs).
- PySCF is still faster on one thread because its matvec is an OpenBLAS `dgemv`. The C++ matvec loop calls `Matrix::cols()`, defined in another source file, on every element, which keeps the compiler from optimizing it. That is a known, unfixed weakness of the baseline.
- PySCF threads only that `dgemv`, so it gains little from extra threads at small n.

**Thread settings and affinity** (these differ between C++ and PySCF):
- All runs set `OMP_PLACES=cores` and `OMP_PROC_BIND=close`.
- C++ OpenMP threads are pinned one per physical core. BLAS threads are set to 1, though the C++ solver doesn't call BLAS.
- PySCF runs set `OMP_NUM_THREADS` and `OPENBLAS/MKL/BLIS_NUM_THREADS` to 1 or 8. libgomp pins the main Python thread to core 0 (CPUs 0–1), but the OpenBLAS worker threads are not pinned and can run on all 16 logical CPUs.

**PySCF settings:**
- `davidson1` is called with `nroots=4` and `tol_residual=1e-8`. It also requires every eigenvalue change between iterations to be below `tol=1e-12` (its default).
- `lindep=1e-16`, not the default 1e-14. With 1e-14, PySCF stops refining any root whose residual is below about 1e-7, so it cannot reach 1e-8.
- `max_cycle=100`, matching the C++ iteration limit.
- `max_space=12`, the default. PySCF raises this internally to 24 for 4 roots, and each solve restarted once.
- The matvec function applies `A.dot(x)` to each vector separately.
- The preconditioner is the C++ guarded diagonal one. However, `davidson1` passes the lowest Ritz value as the shift for every root, whereas the C++ solver uses each root's own θ.
