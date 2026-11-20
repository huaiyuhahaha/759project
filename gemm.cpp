#include "gemm.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>

#if defined(__AVX512F__) || (defined(__AVX2__) && defined(__FMA__))
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

// Loop structure (GotoBLAS/BLIS), for C (m x n) += A (m x k) * B (k x n):
//
//   for pc in steps of kc:                        rank-kc update
//     pack A[:, pc:pc+kc] into MR-row panels      (all m rows; shared by threads)
//     for jc in steps of nc:
//       pack B[pc:pc+kc, jc:jc+nc] into NR-column panels   (shared by threads)
//       parallel for (ic block of mc rows, jr panel of NR columns):
//         for ir in steps of MR within the ic block:
//           micro-kernel: C[ir:ir+MR, jr:jr+NR] += Apanel(MR x kc) * Bpanel(kc x NR)
//
// The micro-kernel keeps the MR x NR tile of C in vector registers for the
// whole kc loop, so C is read and written once per rank-kc update. A
// B micro-panel (kc x NR) stays in L1 while the thread sweeps the A panels
// of its mc-row block, which stays in L2. Packing makes both operands
// contiguous and zero-pads partial panels, so the kernel never needs bounds
// checks; partial tiles of C go through a small temporary tile instead.
//
// Parallelization: OpenMP threads split the (ic, jr) iteration space, so
// every thread writes disjoint tiles of C and no reduction is needed. Packing
// is split across threads too. The implicit barrier after each `omp for`
// keeps packing and computing in order.

namespace {

using idx = std::ptrdiff_t;

// ---- SIMD abstraction: one vector register type and five operations. ----
#if defined(__AVX512F__) || defined(GEMM_TEST_AVX512_SHAPE)
// AVX-512 tile shape and blocking, shared by the real kernel and its test build.
constexpr int VW = 8;   // doubles per vector
constexpr int MR = 12;  // tile rows:   12 x 2 accumulators + 2 B vectors + 1 broadcast = 27 of 32 zmm
constexpr int NV = 2;   // tile vectors per row (NR = 16)
constexpr GemmBlocking kDefaultBlocking{72, 768, 4096};  // best in an Euler sweep (EPYC 9254, Zen 4)
#if defined(GEMM_TEST_AVX512_SHAPE)
// Test only (checks/run_gemm_check.py): the same shape on GCC generic 8-double
// vectors, so the AVX-512 configuration of the packing, blocking and edge code
// runs on CPUs without AVX-512. Only the intrinsic wrappers below differ.
typedef double vreg __attribute__((vector_size(64)));
constexpr const char* kKernelName = "emulated avx512 shape 12x16 (test only)";
inline vreg vzero() { return vreg{}; }
inline vreg vload(const double* p) { vreg v; std::memcpy(&v, p, sizeof v); return v; }
inline void vstore(double* p, vreg v) { std::memcpy(p, &v, sizeof v); }
inline vreg vbroadcast(double x) { return vreg{x, x, x, x, x, x, x, x}; }
inline vreg vfmadd(vreg a, vreg b, vreg c) { return a * b + c; }
inline vreg vadd(vreg a, vreg b) { return a + b; }
#else
using vreg = __m512d;
constexpr const char* kKernelName = "avx512f 12x16";
inline vreg vzero() { return _mm512_setzero_pd(); }
inline vreg vload(const double* p) { return _mm512_loadu_pd(p); }
inline void vstore(double* p, vreg v) { _mm512_storeu_pd(p, v); }
inline vreg vbroadcast(double x) { return _mm512_set1_pd(x); }
inline vreg vfmadd(vreg a, vreg b, vreg c) { return _mm512_fmadd_pd(a, b, c); }
inline vreg vadd(vreg a, vreg b) { return _mm512_add_pd(a, b); }
#endif
#elif defined(__AVX2__) && defined(__FMA__)
using vreg = __m256d;
constexpr int VW = 4;
constexpr int MR = 6;   // 6 x 2 accumulators + 2 B vectors + 1 broadcast = 15 of 16 ymm
constexpr int NV = 2;   // NR = 8
constexpr const char* kKernelName = "avx2+fma 6x8";
constexpr GemmBlocking kDefaultBlocking{72, 256, 4096};  // best in a local sweep; = BLIS haswell
inline vreg vzero() { return _mm256_setzero_pd(); }
inline vreg vload(const double* p) { return _mm256_loadu_pd(p); }
inline void vstore(double* p, vreg v) { _mm256_storeu_pd(p, v); }
inline vreg vbroadcast(double x) { return _mm256_set1_pd(x); }
inline vreg vfmadd(vreg a, vreg b, vreg c) { return _mm256_fmadd_pd(a, b, c); }
inline vreg vadd(vreg a, vreg b) { return _mm256_add_pd(a, b); }
#else
struct vreg { double x; };
constexpr int VW = 1;
constexpr int MR = 4;
constexpr int NV = 4;   // NR = 4
constexpr const char* kKernelName = "scalar 4x4";
constexpr GemmBlocking kDefaultBlocking{64, 256, 1024};
inline vreg vzero() { return {0.0}; }
inline vreg vload(const double* p) { return {*p}; }
inline void vstore(double* p, vreg v) { *p = v.x; }
inline vreg vbroadcast(double x) { return {x}; }
inline vreg vfmadd(vreg a, vreg b, vreg c) { return {a.x * b.x + c.x}; }
inline vreg vadd(vreg a, vreg b) { return {a.x + b.x}; }
#endif

constexpr int NR = NV * VW;

idx ceil_div(idx a, idx b) { return (a + b - 1) / b; }
idx round_up(idx a, idx b) { return ceil_div(a, b) * b; }

// C[0:MR, 0:NR] (row stride ldc) += Ap * Bp, where
//   Ap is an MR x kc panel stored column by column: Ap[p * MR + r] = A(r, p);
//   Bp is a kc x NR panel stored row by row:       Bp[p * NR + c] = B(p, c).
// The loops over r and v have compile-time bounds and are fully unrolled, so
// acc lives in registers.
inline void micro_kernel(int kc, const double* __restrict Ap,
                         const double* __restrict Bp, double* __restrict C, idx ldc) {
    vreg acc[MR][NV];
#pragma GCC unroll 16
    for (int r = 0; r < MR; ++r)
#pragma GCC unroll 4
        for (int v = 0; v < NV; ++v)
            acc[r][v] = vzero();

    // Start fetching the C tile now; it is only needed after the kc loop.
    // Its rows are ldc apart, a pattern the hardware prefetchers miss, so
    // without this every tile waits on DRAM latency once C outgrows L3.
    // Each row is NR doubles and may be unaligned, so touch one element per
    // 64-byte line plus the last element (2 lines per row for AVX2, 3 for AVX-512).
#pragma GCC unroll 16
    for (int r = 0; r < MR; ++r) {
#pragma GCC unroll 4
        for (int c = 0; c < NR; c += 8)
            __builtin_prefetch(C + r * ldc + c, 1);
        __builtin_prefetch(C + r * ldc + NR - 1, 1);
    }

    for (int p = 0; p < kc; ++p) {
        vreg b[NV];
#pragma GCC unroll 4
        for (int v = 0; v < NV; ++v)
            b[v] = vload(Bp + v * VW);
#pragma GCC unroll 16
        for (int r = 0; r < MR; ++r) {
            const vreg a = vbroadcast(Ap[r]);
#pragma GCC unroll 4
            for (int v = 0; v < NV; ++v)
                acc[r][v] = vfmadd(a, b[v], acc[r][v]);
        }
        Ap += MR;
        Bp += NR;
    }

#pragma GCC unroll 16
    for (int r = 0; r < MR; ++r)
#pragma GCC unroll 4
        for (int v = 0; v < NV; ++v) {
            double* c = C + r * ldc + v * VW;
            vstore(c, vadd(vload(c), acc[r][v]));
        }
}

// Same as micro_kernel, but only the top-left mr x nr part of the tile exists in C.
void micro_kernel_edge(int kc, const double* Ap, const double* Bp,
                       double* C, idx ldc, int mr, int nr) {
    alignas(64) double tile[MR * NR] = {};
    micro_kernel(kc, Ap, Bp, tile, NR);
    for (int r = 0; r < mr; ++r)
        for (int c = 0; c < nr; ++c)
            C[r * ldc + c] += tile[r * NR + c];
}

// Copy A[0:mr, 0:kc] (row stride lda) into an MR x kc panel; rows mr..MR-1 are zero.
void pack_A_panel(int mr, int kc, const double* A, idx lda, double* Ap) {
    for (int r = 0; r < mr; ++r) {
        const double* row = A + r * lda;
        for (int p = 0; p < kc; ++p)
            Ap[p * MR + r] = row[p];
    }
    for (int r = mr; r < MR; ++r)
        for (int p = 0; p < kc; ++p)
            Ap[p * MR + r] = 0.0;
}

// Copy B[0:kc, 0:nr] (row stride ldb) into a kc x NR panel; columns nr..NR-1 are zero.
void pack_B_panel(int kc, int nr, const double* B, idx ldb, double* Bp) {
    for (int p = 0; p < kc; ++p) {
        const double* row = B + p * ldb;
        double* out = Bp + p * NR;
        for (int c = 0; c < nr; ++c)
            out[c] = row[c];
        for (int c = nr; c < NR; ++c)
            out[c] = 0.0;
    }
}

// 64-byte-aligned scratch buffer that grows on demand and is reused across
// calls, so repeated small multiplies do not pay for allocation and page faults.
class PackBuffer {
public:
    PackBuffer() = default;
    PackBuffer(const PackBuffer&) = delete;
    PackBuffer& operator=(const PackBuffer&) = delete;
    ~PackBuffer() { std::free(data_); }

    double* get(std::size_t count) {
        if (count > capacity_) {
            const std::size_t bytes = round_up(static_cast<idx>(count * sizeof(double)), 64);
            void* p = std::aligned_alloc(64, bytes);
            if (!p)
                throw std::bad_alloc();
            std::free(data_);
            data_ = static_cast<double*>(p);
            capacity_ = count;
        }
        return data_;
    }

private:
    double* data_ = nullptr;
    std::size_t capacity_ = 0;
};

int resolve_threads(int nthreads) {
#ifdef _OPENMP
    return nthreads > 0 ? nthreads : omp_get_max_threads();
#else
    (void)nthreads;
    return 1;
#endif
}

}  // namespace

GemmBlocking gemm_default_blocking() { return kDefaultBlocking; }
int gemm_mr() { return MR; }
int gemm_nr() { return NR; }
const char* gemm_kernel_name() { return kKernelName; }

void gemm(int m, int n, int k, const double* A, int lda, const double* B, int ldb,
          double* C, int ldc, int nthreads) {
    gemm(m, n, k, A, lda, B, ldb, C, ldc, nthreads, kDefaultBlocking);
}

void gemm(int m, int n, int k, const double* A, int lda, const double* B, int ldb,
          double* C, int ldc, int nthreads, const GemmBlocking& blocking) {
    if (m < 0 || n < 0 || k < 0)
        throw std::invalid_argument("gemm: dimensions must be nonnegative");
    if (lda < k || ldb < n || ldc < n)
        throw std::invalid_argument("gemm: leading dimension smaller than row length");
    if (m == 0 || n == 0 || k == 0)
        return;  // C += 0
    if (!A || !B || !C)
        throw std::invalid_argument("gemm: null matrix pointer");

    const int threads = resolve_threads(nthreads);

    // Clamp the blocking to the problem and to multiples of the tile.
    const idx kc = std::min<idx>(std::max(blocking.kc, 1), k);
    const idx nc = std::min<idx>(round_up(std::max(blocking.nc, 1), NR), round_up(n, NR));
    const idx a_panels = ceil_div(m, MR);

    // The m rows are split into ic_blocks row blocks of about mc rows. Block ib
    // covers A panels [ib * a_panels / ic_blocks, (ib + 1) * a_panels / ic_blocks),
    // so block sizes differ by at most one panel and the statically scheduled
    // (row block, column panel) tasks all cost about the same. (Rounding m up
    // to blocks of exactly mc would leave one short block, and the thread that
    // gets it would idle at the barrier.)
    idx ic_blocks = ceil_div(a_panels, ceil_div(std::max(blocking.mc, 1), MR));
    // With few column panels (thin B), use more row blocks so every thread gets several tasks.
    const idx jr_panels = ceil_div(nc, NR);
    const idx min_tasks = 4 * static_cast<idx>(threads);
    if (threads > 1 && ic_blocks * jr_panels < min_tasks)
        ic_blocks = std::min(a_panels, ceil_div(min_tasks, jr_panels));

    static thread_local PackBuffer a_buffer, b_buffer;
    double* const Apack = a_buffer.get(static_cast<std::size_t>(a_panels * MR * kc));
    double* const Bpack = b_buffer.get(static_cast<std::size_t>(kc * nc));

#pragma omp parallel num_threads(threads) if (threads > 1)
    for (idx pc = 0; pc < k; pc += kc) {
        const int kcur = static_cast<int>(std::min(kc, k - pc));

#pragma omp for schedule(static)
        for (idx ip = 0; ip < a_panels; ++ip) {
            const idx i0 = ip * MR;
            pack_A_panel(static_cast<int>(std::min<idx>(MR, m - i0)), kcur,
                         A + i0 * lda + pc, lda, Apack + ip * MR * kcur);
        }

        for (idx jc = 0; jc < n; jc += nc) {
            const idx ncur = std::min(nc, n - jc);
            const idx jpanels = ceil_div(ncur, NR);

#pragma omp for schedule(static)
            for (idx jp = 0; jp < jpanels; ++jp) {
                const idx j0 = jp * NR;
                pack_B_panel(kcur, static_cast<int>(std::min<idx>(NR, ncur - j0)),
                             B + pc * ldb + jc + j0, ldb, Bpack + jp * NR * kcur);
            }

            // Iterations are ordered ic-major, so a static chunk keeps one
            // thread on the same mc-row block of packed A while it walks the
            // B panels.
#pragma omp for collapse(2) schedule(static)
            for (idx ib = 0; ib < ic_blocks; ++ib) {
                for (idx jp = 0; jp < jpanels; ++jp) {
                    const double* Bp = Bpack + jp * NR * kcur;
                    const idx j = jc + jp * NR;
                    const int nr = static_cast<int>(std::min<idx>(NR, n - j));
                    const idx i_begin = ib * a_panels / ic_blocks * MR;
                    const idx i_end = std::min<idx>(m, (ib + 1) * a_panels / ic_blocks * MR);
                    for (idx i = i_begin; i < i_end; i += MR) {
                        const double* Ap = Apack + (i / MR) * MR * kcur;
                        const int mr = static_cast<int>(std::min<idx>(MR, m - i));
                        double* Cij = C + i * ldc + j;
                        if (mr == MR && nr == NR)
                            micro_kernel(kcur, Ap, Bp, Cij, ldc);
                        else
                            micro_kernel_edge(kcur, Ap, Bp, Cij, ldc, mr, nr);
                    }
                }
            }
        }
    }
}
