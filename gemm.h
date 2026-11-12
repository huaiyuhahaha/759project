#ifndef GEMM_H
#define GEMM_H

// Cache-blocked, SIMD, OpenMP-parallel double-precision matrix multiply
//     C += A * B
// for row-major A (m x k), B (k x n) and C (m x n). The structure follows the
// GotoBLAS/BLIS design: blocks of A and B are copied ("packed") into
// contiguous buffers sized for the cache levels, and a register-tiled
// micro-kernel computes one MR x NR tile of C at a time. See gemm.cpp.

// Cache-blocking sizes, in matrix elements.
struct GemmBlocking {
    int mc;  // rows of A per task: an mc x kc block of packed A should fit in L2
    int kc;  // depth of each rank-kc update: MR x kc and kc x NR panels should fit in L1
    int nc;  // columns of B packed at once: the kc x nc block should fit in L3
};

// Defaults tuned for the compiled micro-kernel (see gemm.cpp).
GemmBlocking gemm_default_blocking();

// Shape of the compiled micro-kernel tile and a short description such as
// "avx2+fma 6x8". The kernel is chosen at compile time from the target ISA
// (-march=native picks AVX-512 if available, else AVX2+FMA, else scalar).
int gemm_mr();
int gemm_nr();
const char* gemm_kernel_name();

// C += A * B with row-major storage and leading dimensions (row strides)
// lda >= k, ldb >= n, ldc >= n. Only the m x n region of C is written.
// C must not overlap A or B.
// nthreads <= 0 means omp_get_max_threads() (OMP_NUM_THREADS); without
// OpenMP the call is single-threaded. Blocking sizes are rounded up to
// multiples of the micro-tile; any values >= 1 are valid.
// Throws std::invalid_argument on negative sizes or too-small leading dimensions.
void gemm(int m, int n, int k,
          const double* A, int lda,
          const double* B, int ldb,
          double* C, int ldc,
          int nthreads = 0);

void gemm(int m, int n, int k,
          const double* A, int lda,
          const double* B, int ldb,
          double* C, int ldc,
          int nthreads, const GemmBlocking& blocking);

#endif
