// Time one matrix-multiply implementation on C (m x n) += A (m x k) * B (k x n).
//
// Usage: gemm_bench <variant> <m> <n> <k> <threads> [runs] [mc kc nc]
//   variant  naive    the original Matrix::operator* (master 0e014e3), compiled from
//                     the verbatim copy in bench/original_matrix.cpp; as before,
//                     it allocates its zeroed result inside the timed call
//            loop     the same i-k-j loop on raw restrict pointers, which the compiler
//                     vectorizes (no blocking, no packing, single-threaded)
//            huaiyu   cache_register_multiplication() from matrix_optimize_4.cpp
//                     (single-threaded; only in AVX-512 builds with -DHAVE_HUAIYU)
//            blocked  gemm() from gemm.h with <threads> OpenMP threads
//            openblas cblas_dgemm with <threads> OpenBLAS threads (-DHAVE_OPENBLAS)
//   runs     timed runs after one untimed warm-up (default 5)
//   mc kc nc blocking for the blocked variant (default gemm_default_blocking())
//
// A and B are uniform in [-1, 1) from mt19937_64 seeded with 759. Every run
// starts from C = 0 (zeroed outside the timer) and times only the multiply.
// The result of the last run is compared with a reference (OpenBLAS if
// available, else gemm() on one thread). Since every |a|, |b| <= 1, each entry
// must be within 2 (k + 2) eps k of the reference.
//
// Prints one JSON line: variant, sizes, threads, kernel, blocking, median/min
// seconds, GFLOP/s at the median (2 m n k flops), max |C - ref|, ok, and the
// core type OpenBLAS selected (its DYNAMIC_ARCH detection can be wrong on new
// CPUs; override with OPENBLAS_CORETYPE).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "gemm.h"
#include "matrix.h"
#include "original_matrix.h"

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef HAVE_OPENBLAS
#include <cblas.h>
#endif
#ifdef HAVE_HUAIYU
void cache_register_multiplication(const double* A, const double* B, double* C,
                                   int A_rows, int B_cols, int mutual_dim, int block_mu_dim);
#endif

namespace {

// The naive loop on raw pointers without the per-access bounds checks:
// what -O3 auto-vectorization gives for an unblocked i-k-j loop.
void loop_multiply(int m, int n, int k, const double* __restrict A,
                   const double* __restrict B, double* __restrict C) {
    for (int i = 0; i < m; ++i) {
        double* __restrict c = C + static_cast<std::size_t>(i) * n;
        for (int p = 0; p < k; ++p) {
            const double a = A[static_cast<std::size_t>(i) * k + p];
            const double* __restrict b = B + static_cast<std::size_t>(p) * n;
            for (int j = 0; j < n; ++j)
                c[j] += a * b[j];
        }
    }
}

[[noreturn]] void usage(const char* msg) {
    std::fprintf(stderr, "gemm_bench: %s\nusage: gemm_bench <naive|loop|huaiyu|blocked|openblas> "
                         "<m> <n> <k> <threads> [runs] [mc kc nc]\n", msg);
    std::exit(2);
}

int parse_int(const char* s, const char* what) {
    char* end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (*s == '\0' || *end != '\0' || v < 0 || v > (1L << 30))
        usage((std::string("bad ") + what).c_str());
    return static_cast<int>(v);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6 && argc != 7 && argc != 10)
        usage("wrong number of arguments");
    const std::string variant = argv[1];
    const int m = parse_int(argv[2], "m"), n = parse_int(argv[3], "n"),
              k = parse_int(argv[4], "k"), threads = parse_int(argv[5], "threads");
    const int runs = argc >= 7 ? parse_int(argv[6], "runs") : 5;
    if (threads < 1 || runs < 1)
        usage("threads and runs must be >= 1");
    GemmBlocking blocking = gemm_default_blocking();
    if (argc == 10)
        blocking = {parse_int(argv[7], "mc"), parse_int(argv[8], "kc"), parse_int(argv[9], "nc")};

    bool known = variant == "naive" || variant == "loop" || variant == "blocked";
#ifdef HAVE_OPENBLAS
    known = known || variant == "openblas";
    // Must match OPENBLAS_NUM_THREADS, which run_gemm_bench.py sets the same
    // way; never raise the count above the one OpenBLAS was started with.
    openblas_set_num_threads(variant == "openblas" ? threads : 1);
#endif
#ifdef HAVE_HUAIYU
    known = known || variant == "huaiyu";
#endif
    if (!known)
        usage(("variant not available in this build: " + variant).c_str());
    if ((variant == "naive" || variant == "loop" || variant == "huaiyu") && threads != 1)
        usage("naive, loop and huaiyu are single-threaded; use threads = 1");

    Matrix A(m, k), B(k, n), C(m, n);
    std::mt19937_64 rng(759);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const std::size_t asize = static_cast<std::size_t>(m) * k;
    const std::size_t bsize = static_cast<std::size_t>(k) * n;
    const std::size_t csize = static_cast<std::size_t>(m) * n;
    for (std::size_t i = 0; i < asize; ++i) A.data()[i] = u(rng);
    for (std::size_t i = 0; i < bsize; ++i) B.data()[i] = u(rng);

    // Copies of A and B in the original Matrix class, for the naive variant only.
    const bool naive = variant == "naive";
    original_matrix::Matrix Aorig(naive ? m : 0, naive ? k : 0), Borig(naive ? k : 0, naive ? n : 0);
    original_matrix::Matrix Corig(0, 0);
    if (naive) {
        for (int i = 0; i < m; ++i)
            std::copy_n(A.data() + static_cast<std::size_t>(i) * k, k, Aorig[i]);
        for (int i = 0; i < k; ++i)
            std::copy_n(B.data() + static_cast<std::size_t>(i) * n, n, Borig[i]);
    }

    auto multiply = [&] {
        if (variant == "naive") {
            Corig = Aorig * Borig;
        } else if (variant == "loop") {
            loop_multiply(m, n, k, A.data(), B.data(), C.data());
        } else if (variant == "blocked") {
            gemm(m, n, k, A.data(), k, B.data(), n, C.data(), n, threads, blocking);
#ifdef HAVE_OPENBLAS
        } else if (variant == "openblas") {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A.data(), k,
                        B.data(), n, 1.0, C.data(), n);
#endif
#ifdef HAVE_HUAIYU
        } else if (variant == "huaiyu") {
            cache_register_multiplication(A.data(), B.data(), C.data(), m, n, k, 256);
#endif
        }
    };

    using Clock = std::chrono::steady_clock;
    std::vector<double> times;
    for (int r = 0; r <= runs; ++r) {  // r == 0 is the warm-up
        std::fill_n(C.data(), csize, 0.0);
        const auto t0 = Clock::now();
        multiply();
        const double dt = std::chrono::duration<double>(Clock::now() - t0).count();
        if (r > 0)
            times.push_back(dt);
    }
    if (naive)  // outside the timer: move the original-class result into C for the check
        for (int i = 0; i < m; ++i)
            std::copy_n(Corig[i], n, C.data() + static_cast<std::size_t>(i) * n);
    std::sort(times.begin(), times.end());
    const double median = runs % 2 ? times[runs / 2]
                                   : 0.5 * (times[runs / 2 - 1] + times[runs / 2]);

    // Reference, outside every timer. OpenBLAS keeps the thread count it was
    // started with (OPENBLAS_NUM_THREADS: 1, or <threads> for the openblas
    // variant): raising it at run time crashes OpenBLAS 0.3.28 inside its own
    // threaded dgemm (segfault in dgemm_itcopy with 4+ threads).
    std::vector<double> ref(csize, 0.0);
#ifdef HAVE_OPENBLAS
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A.data(), k,
                B.data(), n, 0.0, ref.data(), n);
    const char* ref_name = "openblas";
#else
    gemm(m, n, k, A.data(), k, B.data(), n, ref.data(), n, 1);
    const char* ref_name = "blocked-1thread";
#endif
    double max_err = 0;
    for (std::size_t i = 0; i < csize; ++i)
        max_err = std::max(max_err, std::abs(C.data()[i] - ref[i]));
    const double allowed = 2.0 * (k + 2) * std::numeric_limits<double>::epsilon() * k;
    const bool ok = std::isfinite(max_err) && max_err <= allowed;

    const double gflops = median > 0 ? 2.0 * m * n * k / median * 1e-9 : 0.0;
#ifdef HAVE_OPENBLAS
    const char* openblas_core = openblas_get_corename();
    std::string openblas_config = openblas_get_config();  // e.g. "... USE_OPENMP ..."
    while (!openblas_config.empty() && openblas_config.back() == ' ')
        openblas_config.pop_back();
#else
    const char* openblas_core = "none";
    const std::string openblas_config = "none";
#endif
    std::printf("{\"variant\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"threads\": %d, "
                "\"kernel\": \"%s\", \"mc\": %d, \"kc\": %d, \"nc\": %d, \"runs\": %d, "
                "\"median\": %.9g, \"min\": %.9g, \"gflops\": %.6g, \"reference\": \"%s\", "
                "\"max_err\": %.3g, \"ok\": %s, \"openblas_core\": \"%s\", "
                "\"openblas_config\": \"%s\"}\n",
                variant.c_str(), m, n, k, threads, gemm_kernel_name(), blocking.mc, blocking.kc,
                blocking.nc, runs, median, times.front(), gflops, ref_name, max_err,
                ok ? "true" : "false", openblas_core, openblas_config.c_str());
    return ok ? 0 : 1;
}
