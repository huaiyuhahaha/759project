# ECE 759 Parallel Davidson Eigensolver

A C++17 / CUDA Davidson eigensolver for the lowest few eigenpairs of a dense real symmetric matrix. It provides three backends: a serial baseline (`Backend::Serial`), a multi-threaded CPU version (`Backend::OpenMP`), and a GPU-resident implementation (`Backend::CUDA`). All three backends are verified against NumPy and benchmarked against PySCF's `davidson1` solver.

## What it computes

`Davidson(A, guess, nroots, tol, max_iter, backend)` ([davidson.h](davidson.h)) returns the `nroots` smallest eigenvalues λ₁ ≤ … ≤ λ_k of a symmetric n×n matrix `A`, along with orthonormal eigenvectors X (n×k).

It stops when every root satisfies ‖A x_k − λ_k x_k‖₂ < `tol`. The returned `DavidsonResult` also reports:
- the convergence flag;
- the number of iterations and subspace expansions;
- the number of dropped corrections;
- the final subspace size;
- the residual norms;
- per-kernel timings (`DavidsonProfile`).

```cpp
#include "davidson.h"
Matrix guess = initialize_vector(4, A);   // unit vectors at the 4 smallest diagonal entries
DavidsonResult r = Davidson(A, guess, 4, 1e-8, 100, Backend::CUDA);  // Serial, OpenMP, or CUDA
```

## The Davidson loop ([davidson.cpp](davidson.cpp))

1. **Initial guesses.** Each column of `guess` is orthonormalized against the basis V using two passes of Classical Gram–Schmidt (CGS2). A vector is dropped as linearly dependent if its remaining norm is ≤ 1e-8 times its original norm (`kLinDepTol`). Each kept vector v gets one matrix-vector product, and A·v is stored so A is applied only once per basis vector.
2. **Projection.** The loop forms the small symmetric projected matrix H = Vᵀ(AV) (m×m) from the stored products.
3. **Jacobi diagonalization.** `MatrixJacobi` runs classical Jacobi rotations on H:
   - it zeros out the largest off-diagonal entry at each step;
   - it stops when every off-diagonal entry is ≤ 1e-12 · max|H|, or after 50·m² rotations;
   - it accumulates the rotations as eigenvectors Q.

   The Ritz values are then sorted in ascending order.
4. **Residual check.** For each of the `nroots` lowest Ritz pairs (θ, y), the loop computes:
   - x = V y;
   - A x = (AV) y;
   - r = A x − θ x.

   It stops, converged, when every ‖r‖₂ < `tol`.
5. **Subspace expansion.** Each unconverged root gets a diagonal-preconditioned correction δᵢ = rᵢ / (θ − Aᵢᵢ) (or rᵢ / (Aᵢᵢ − θ) up to sign), clamping small denominators (`kDenomFloor` = 1e-8 on CPU; `kPrecondEps` = 1e-12 fallback on GPU).
   - Corrections are orthonormalized as in step 1, and dependent ones are dropped.
   - On CPU, if every correction is dropped, the raw residuals are tried instead.
   - If nothing new can be added, the solver stops unconverged. It also stops unconverged after `max_iter` iterations.
   - There is no restart, so the subspace grows by at most `nroots` vectors per iteration.

## What OpenMP parallelizes

Only the matrix-vector product A·v (`matvec_omp`): `#pragma omp parallel for schedule(static)` over the rows of A. Each row's dot product is computed by one thread in the same order as the serial loop, so the OpenMP results are bitwise identical to serial.

Everything else stays serial and is shared with the baseline: Gram–Schmidt, projection, Jacobi diagonalization, Ritz vectors, residuals, and preconditioning. The thread count comes from `OMP_NUM_THREADS`. On the benchmark matrices, `matvec` takes 96%, 98%, and 99% of the serial solve time at n = 512, 1024, and 2048.

## What CUDA parallelizes ([davidson.cu](davidson.cu))

`DavidsonCUDA` keeps A, the subspace basis V, and AV resident in GPU device memory across iterations, transferring only the small m×m projected matrix H to the host for `MatrixJacobi`:

- **Memory layout:** A is stored row-major (`row * n + col`) for coalesced row reads during matrix-vector multiplication, while V, AV, X, AX, and T are stored column-major (`col * n + row`) so each n-element vector is contiguous for 1D coalesced kernel accesses.
- **Blocked matrix-vector product (`matvec_block_kernel`):** Multiplies A against only the newly added basis columns (`new_vecs = m - computed_av`) using a 2D grid of 16×16 thread blocks (`kTile = 16`) with shared-memory tiling (`tileA` and `tileV`).
- **Rayleigh–Ritz projection (`project_kernel`):** Launches one 256-thread block per upper-triangular entry (i, j) of H and computes vᵢᵀ(Avⱼ) via two-stage warp-shuffle reduction (`block_reduce_sum`).
- **Ritz vector reconstruction (`ritz_vectors_kernel`):** Stages the m × nroots eigenvector matrix Q in shared memory and computes X = VQ and AX = (AV)Q in parallel across all n rows.
- **Fused residual and preconditioning (`fused_residual_precond_kernel`):** Computes r = AX_r − λ_r X_r, accumulates ‖r‖₂² via warp-shuffle reduction and `atomicAdd`, and applies the diagonal Jacobi preconditioner in a single pass over global memory.
- **GPU Gram–Schmidt (`gpu_orthonormalize_against`):** Runs two-pass Classical Gram–Schmidt on device vectors using `dot_kernel`, `axpy_kernel`, and `scale_kernel`.

## Solvers in this repo

| Name | Role | Where |
|---|---|---|
| C++ serial | Validated baseline (`Backend::Serial`) | [davidson.cpp](davidson.cpp) |
| C++ OpenMP | Multi-core CPU backend; threads the matvec (`Backend::OpenMP`) | [davidson.cpp](davidson.cpp) |
| C++ CUDA | GPU-resident solver (`Backend::CUDA`) | [davidson.cu](davidson.cu), [davidson.cpp](davidson.cpp) |
| PySCF `davidson1` | External Davidson implementation for comparison; timed | [bench/pyscf_davidson_bench.py](bench/pyscf_davidson_bench.py) |
| NumPy `eigvalsh` | Exact reference eigenvalues for checking results; never timed | [checks/run_davidson_e2e.py](checks/run_davidson_e2e.py), [bench/run_davidson_bench.py](bench/run_davidson_bench.py) |

**Status:**
- **Implemented:** the serial, OpenMP, and CUDA backends, the E2E checks, and the full benchmark suite.
- **Optional extension:** atomic SCF built on this solver (not implemented).

## Layout

```
matrix.h/.cpp                  dense row-major Matrix
davidson.h/.cpp                initialize_vector, MatrixJacobi, orthonormalize_against, Davidson
davidson.cu                    CUDA kernels and DavidsonCUDA GPU-resident solver
checks/davidson_e2e.cpp        E2E check program (n=8, n=64, and the drop-criterion check)
checks/run_davidson_e2e.py     builds with nvcc and runs every case on serial, openmp, and cuda
bench/davidson_bench.cpp       C++ / CUDA timing program; also writes benchmark inputs for PySCF
bench/pyscf_davidson_bench.py  PySCF davidson1 runner
bench/run_davidson_bench.py    builds and runs the full benchmark and prints the tables
```

There is no Makefile: the two `run_*.py` scripts compile their binaries in a temporary directory using `nvcc` (forwarding `-fopenmp` and host optimization flags to `g++`) or `g++` alone when `--no-cuda` is passed.

## Prerequisites

- Linux or WSL2 (scripts use `os.sched_getaffinity`). Tested on WSL2 Ubuntu 24.04.1.
- NVIDIA CUDA Toolkit (`nvcc`) and a CUDA-capable GPU (compute capability ≥ 6.0 for `double` `atomicAdd`). Tested with CUDA 13.3 on an NVIDIA RTX 4000 Ada (`sm_89`).
- `g++` with C++17 and OpenMP (tested with `g++ 13.3.0`).
- Python 3 with NumPy, SciPy, PySCF, and `threadpoolctl` (tested with Python 3.12.3, NumPy 2.5.5, SciPy 1.18.1, PySCF 2.15.1, `threadpoolctl 3.8.0`):

```bash
python3 -m venv ~/pyenv
source ~/pyenv/bin/activate
pip install numpy scipy pyscf threadpoolctl
```

## Commands

Run from the repo root (or `checks/` / `bench/`) in a Linux/WSL shell with the virtual environment activated. Each command exits `0` only if every check passes:

```bash
python3 checks/run_davidson_e2e.py              # E2E checks, -O2, serial, openmp, and cuda backends
python3 checks/run_davidson_e2e.py --sanitize   # CPU checks under ASan + UBSan with leak detection
python3 bench/run_davidson_bench.py             # full benchmark (options: --sizes, --threads, --pyscf-threads, --no-cuda)
```

From Windows PowerShell or Command Prompt, prefix each command with `wsl -d Ubuntu --cd "<repo path>" --`.

## Correctness: E2E n = 8 and n = 64

The test matrix is tridiagonal, with diagonal 1..n and both neighboring diagonals equal to 0.2.
- Settings: k = 2, guess `initialize_vector(2, A)`, solver tolerance 1e-10.
- Reference: NumPy `eigvalsh(A)[:2]`.
- Pass criteria: every eigenvalue error below 1e-8, residual norm within tolerance, and final subspace smaller than n.

All three backends (`serial`, `openmp`, and `cuda`) converged to identical eigenvalues and subspace dimensions on n = 64:

| Backend | Converged | Iterations | Expansions | Dropped corrections | Final subspace | Max \|λ − eigvalsh\| | Max ‖Ax − λx‖₂ | ‖XᵀX − I‖_F |
|---|---|---|---|---|---|---|---|---|
| `serial` | yes | 9 | 8 | 7 | 10 | 0.0 | 1.510e-11 | 1.132e-15 |
| `openmp` | yes | 9 | 8 | 7 | 10 | 0.0 | 1.510e-11 | 1.132e-15 |
| `cuda` | yes | 9 | 8 | 7 | 10 | 0.0 | 1.510e-11 | 1.132e-15 |

On n = 8, all three backends converged in 7 iterations (final subspace 8) with max eigenvalue error 2.220e-16, max residual 9.564e-14, and ‖XᵀX − I‖_F = 1.735e-15. The script also runs a unit check verifying that `orthonormalize_against` drops a vector when its remaining-to-original norm ratio is ≤ 1e-8 across scales 1e-6, 1, and 1e6.

## Performance snapshot

**Setup:**
- Hardware: NVIDIA RTX 4000 Ada Generation (`sm_89`, 20 GB VRAM) and 24-core host CPU under WSL2 Ubuntu 24.04.1.
- Compiler flags: `nvcc -std=c++17 -O3 -arch=native -DUSE_CUDA -Xcompiler -march=native,-fopenmp,-Wall,-Wextra` (`nvcc 13.3` / `g++ 13.3.0`).
- Matrices: dense symmetric strictly diagonally dominant matrices with Aᵢᵢ = i + 1 and Aᵢⱼ = Aⱼᵢ = u/n, where u ∈ [−1, 1) from `mt19937_64` with seed 759.
- Solve: k = 4, guess `initialize_vector(4, A)`, residual tolerance 1e-8.
- Timing: each configuration runs in its own process with one warm-up solve followed by five timed solves; tables report median wall-clock time in milliseconds.
- Inputs: PySCF reads the exact matrix and guess bytes dumped by `davidson_bench`, verified via 64-bit position-weighted checksums (`n=512: 22b609d39ba4198c`, `n=1024: 3d70c0a8d9e64af0`, `n=2048: f559e02c1377bad8`).

### End-to-end median solve times (ms)

| n | C++ serial | C++ OpenMP-1 | C++ OpenMP-2 | C++ OpenMP-4 | C++ OpenMP-8 | C++ CUDA (GPU) | PySCF-1 | PySCF-8 |
|---|---|---|---|---|---|---|---|---|
| 512 | 8.25 | 7.61 | 3.96 | 2.29 | 1.25 | 21.34 | 3.12 | 3.24 |
| 1024 | 25.74 | 25.21 | 12.86 | 7.04 | 3.86 | 16.15 | 8.31 | 3.07 |
| 2048 | 97.22 | 97.24 | 50.88 | 27.56 | 13.69 | 39.17 | 33.21 | 8.60 |

**Speedups vs C++ serial** (n = 512 / 1024 / 2048):
- **C++ OpenMP-8:** 6.58× / 6.67× / 7.10×
- **C++ CUDA:** 0.39× / 1.59× / 2.48×
- **PySCF-8:** 2.55× / 8.38× / 11.30×

### Kernel profile breakdown (median ms and share of solve)

| Backend | n | Total (ms) | `matvec` | `orthogonalize` | `project` | `eigensolve` | `ritz` | `precondition` | `other` |
|---|---|---|---|---|---|---|---|---|---|
| Serial | 512 | 8.25 | 7.94 (96%) | 0.05 (1%) | 0.08 (1%) | 0.11 (1%) | 0.03 (0%) | 0.01 (0%) | 0.02 (0%) |
| Serial | 1024 | 25.74 | 25.29 (98%) | 0.09 (0%) | 0.16 (1%) | 0.09 (0%) | 0.08 (0%) | 0.03 (0%) | 0.01 (0%) |
| Serial | 2048 | 97.22 | 96.39 (99%) | 0.17 (0%) | 0.19 (0%) | 0.04 (0%) | 0.13 (0%) | 0.18 (0%) | 0.11 (0%) |
| OpenMP-8 | 512 | 1.25 | 0.96 (77%) | 0.05 (4%) | 0.08 (7%) | 0.11 (8%) | 0.03 (2%) | 0.01 (1%) | 0.01 (1%) |
| OpenMP-8 | 1024 | 3.86 | 3.42 (89%) | 0.09 (2%) | 0.16 (4%) | 0.08 (2%) | 0.08 (2%) | 0.02 (1%) | 0.02 (0%) |
| OpenMP-8 | 2048 | 13.69 | 12.86 (94%) | 0.16 (1%) | 0.19 (1%) | 0.04 (0%) | 0.15 (1%) | 0.15 (1%) | 0.12 (1%) |
| CUDA | 512 | 21.34 | 0.32 (1%) | 17.40 (82%) | 0.17 (1%) | 0.39 (2%) | 1.17 (6%) | 0.00 (0%) | 1.89 (9%) |
| CUDA | 1024 | 16.15 | 0.62 (4%) | 10.87 (67%) | 0.16 (1%) | 0.26 (2%) | 0.96 (6%) | 0.00 (0%) | 3.28 (20%) |
| CUDA | 2048 | 39.17 | 1.26 (3%) | 11.50 (29%) | 0.13 (0%) | 0.26 (1%) | 0.91 (2%) | 0.00 (0%) | 25.11 (64%) |

**Correctness across all benchmark runs:**
- Every solve converged; all four PySCF convergence flags were `True`.
- Independent max residual ≤ 8.31e-9, max |λ − eigvalsh| ≤ 3.05e-13, and ‖XᵀX − I‖_F ≤ 4.96e-15.
- C++ OpenMP eigenvalues were bitwise identical to serial; C++ CUDA eigenvalues matched serial within 8.88e-16 (0.0 at n = 2048).

**Performance analysis:**
- **GPU `matvec` acceleration:** `matvec_block_kernel` achieves **24.8×** (n = 512), **40.8×** (n = 1024), and **76.5×** (n = 2048) speedups over C++ serial, reducing the O(n²) matrix-vector products to 1%–4% of total GPU solve time.
- **GPU latency and transfer bottlenecks:** At matrix sizes n ≤ 2048, total CUDA runtime is dominated by two overheads rather than compute:
  1. `orthogonalize` (10.87–17.40 ms): `gpu_orthonormalize_against` launches individual `dot_kernel` and `axpy_kernel` calls inside a double loop over existing basis vectors, incurring a blocking device-to-host `cudaMemcpy` synchronization on every inner product.
  2. `other` (1.89 ms → 25.11 ms as n grows from 512 to 2048): `DavidsonCUDA` flattens the n×n matrix on the host, calls `cudaMalloc`, and transfers A (32 MB at n = 2048) over PCIe inside each timed call.
- **C++ vs PySCF iteration count:** C++ converges in 3–4 iterations (12–15 matvecs) without restarting, whereas PySCF `davidson1` (default `max_space=12`, effective space 24) restarts once and takes 12 iterations (31–32 matvecs), using OpenBLAS `dgemv` for its matrix-vector products.

**Thread settings and affinity:**
- All runs set `OMP_PLACES=cores` and `OMP_PROC_BIND=close`.
- C++ OpenMP threads are pinned one per physical core with BLAS threads pinned to 1.
- PySCF runs set `OMP_NUM_THREADS` and `OPENBLAS/MKL/BLIS_NUM_THREADS` to 1 or 8; `libgomp` pins the main Python thread to core 0 while OpenBLAS worker threads run across available cores.