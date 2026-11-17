// Correctness checks for gemm() (gemm.h) and Matrix::operator*.
//
// Every gemm call is compared with a plain triple-loop reference of C0 + A*B.
// An entry passes if |C - ref| <= 2 (k + 2) eps (|C0| + sum_p |A_ip| |B_pj|),
// i.e. within a few times the standard dot-product rounding bound of both
// computations. The cases cover:
//   - empty and 1-sized dimensions, sizes just below/at/above the micro-tile;
//   - padded leading dimensions: padding of A and B holds NaN (so reading it
//     poisons the result) and padding of C holds a sentinel that must survive;
//   - tiny blocking sizes that force many kc/nc/mc blocks and partial blocks;
//   - 1, 2, 3, 7 and the maximum number of threads;
//   - thin shapes like the Davidson block matvec A (n x n) * V (n x 4);
//   - n larger than the default nc (several column blocks);
//   - concurrent calls from inside an OpenMP parallel region, nested off and on;
//   - argument validation, and Matrix::operator* including empty shapes.
// With -DCHECK_HUAIYU (AVX-512 builds only) it also checks
// cache_register_multiplication() from matrix_optimize_4.cpp.
//
// Usage: gemm_check            exits 0 only if every check passes.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "gemm.h"
#include "matrix.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef CHECK_HUAIYU
void cache_register_multiplication(const double* A, const double* B, double* C,
                                   int A_rows, int B_cols, int mutual_dim, int block_mu_dim);
#endif

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();
constexpr double kSentinel = 1234.5;

int g_checks = 0;
int g_failures = 0;

void fail(const std::string& msg) {
    ++g_failures;
    std::printf("FAIL: %s\n", msg.c_str());
}

int max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

struct Problem {
    int m, n, k, lda, ldb, ldc;
    std::vector<double> A, B, C0;
};

Problem make_problem(int m, int n, int k, int pad, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    Problem p{m, n, k, k + pad, n + pad, n + 2 * pad, {}, {}, {}};
    p.A.assign(static_cast<std::size_t>(m) * p.lda, nan);
    p.B.assign(static_cast<std::size_t>(k) * p.ldb, nan);
    p.C0.assign(static_cast<std::size_t>(m) * p.ldc, kSentinel);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < k; ++j)
            p.A[static_cast<std::size_t>(i) * p.lda + j] = u(rng);
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < n; ++j)
            p.B[static_cast<std::size_t>(i) * p.ldb + j] = u(rng);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            p.C0[static_cast<std::size_t>(i) * p.ldc + j] = u(rng);
    return p;
}

// Compare C (same layout as p.C0) with C0 + A*B; padding must still hold kSentinel.
void verify(const Problem& p, const std::vector<double>& C, const std::string& label) {
    ++g_checks;
    double worst = 0;  // largest error / allowed error
    int bad_i = -1, bad_j = -1;
    for (int i = 0; i < p.m; ++i) {
        for (int j = 0; j < p.ldc; ++j) {
            const std::size_t ij = static_cast<std::size_t>(i) * p.ldc + j;
            if (j >= p.n) {
                if (C[ij] != kSentinel) {
                    fail(label + ": padding of C modified at (" + std::to_string(i) + ", " +
                         std::to_string(j) + ")");
                    return;
                }
                continue;
            }
            double ref = p.C0[ij], bound = std::abs(p.C0[ij]);
            for (int q = 0; q < p.k; ++q) {
                const double a = p.A[static_cast<std::size_t>(i) * p.lda + q];
                const double b = p.B[static_cast<std::size_t>(q) * p.ldb + j];
                ref += a * b;
                bound += std::abs(a * b);
            }
            const double allowed = 2.0 * (p.k + 2) * kEps * bound + 1e-300;
            const double ratio = std::abs(C[ij] - ref) / allowed;
            if (!(ratio <= worst)) {  // also catches NaN
                worst = std::isnan(ratio) ? std::numeric_limits<double>::infinity() : ratio;
                bad_i = i;
                bad_j = j;
            }
        }
    }
    if (!(worst <= 1.0))
        fail(label + ": entry (" + std::to_string(bad_i) + ", " + std::to_string(bad_j) +
             ") error is " + std::to_string(worst) + "x the allowed bound");
}

std::string shape(int m, int n, int k) {
    return "m=" + std::to_string(m) + " n=" + std::to_string(n) + " k=" + std::to_string(k);
}

void check_gemm(int m, int n, int k, int pad, int threads, const GemmBlocking* blocking,
                std::mt19937_64& rng) {
    const Problem p = make_problem(m, n, k, pad, rng);
    std::vector<double> C = p.C0;
    if (blocking)
        gemm(m, n, k, p.A.data(), p.lda, p.B.data(), p.ldb, C.data(), p.ldc, threads, *blocking);
    else
        gemm(m, n, k, p.A.data(), p.lda, p.B.data(), p.ldb, C.data(), p.ldc, threads);
    std::string label = "gemm " + shape(m, n, k) + " pad=" + std::to_string(pad) +
                        " threads=" + std::to_string(threads);
    if (blocking)
        label += " blocking=" + std::to_string(blocking->mc) + "/" +
                 std::to_string(blocking->kc) + "/" + std::to_string(blocking->nc);
    verify(p, C, label);
}

void expect_throw(const std::string& label, const std::function<void()>& f) {
    ++g_checks;
    try {
        f();
    } catch (const std::invalid_argument&) {
        return;
    }
    fail(label + ": expected std::invalid_argument");
}

void check_matrix_multiply(int m, int n, int k, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    Matrix A(m, k), B(k, n);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < k; ++j)
            A[i][j] = u(rng);
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < n; ++j)
            B[i][j] = u(rng);
    const Matrix C = A * B;
    ++g_checks;
    if (C.rows() != m || C.cols() != n) {
        fail("Matrix::operator* " + shape(m, n, k) + ": wrong result shape");
        return;
    }
    Problem p{m, n, k, k, n, n, {}, {}, {}};
    p.A.assign(A.data(), A.data() + static_cast<std::size_t>(m) * k);
    p.B.assign(B.data(), B.data() + static_cast<std::size_t>(k) * n);
    p.C0.assign(static_cast<std::size_t>(m) * n, 0.0);
    verify(p, std::vector<double>(C.data(), C.data() + static_cast<std::size_t>(m) * n),
           "Matrix::operator* " + shape(m, n, k));
}

#ifdef CHECK_HUAIYU
void check_huaiyu(int m, int n, int k, int depth_block, std::mt19937_64& rng) {
    const Problem p = make_problem(m, n, k, 0, rng);
    std::vector<double> C = p.C0;
    cache_register_multiplication(p.A.data(), p.B.data(), C.data(), m, n, k, depth_block);
    verify(p, C, "cache_register_multiplication " + shape(m, n, k) +
                     " block_mu_dim=" + std::to_string(depth_block));
}
#endif

}  // namespace

int main() {
    std::mt19937_64 rng(759);
    const int MR = gemm_mr(), NR = gemm_nr(), T = max_threads();
    const GemmBlocking def = gemm_default_blocking();
    std::printf("kernel %s (MR=%d NR=%d), default blocking mc=%d kc=%d nc=%d, max threads %d\n",
                gemm_kernel_name(), MR, NR, def.mc, def.kc, def.nc, T);

    std::vector<int> thread_counts{1, 2, 3, 7, T};
    std::sort(thread_counts.begin(), thread_counts.end());
    thread_counts.erase(std::unique(thread_counts.begin(), thread_counts.end()),
                        thread_counts.end());

    // 1. Small and edge shapes, default blocking, with and without padding.
    std::vector<std::vector<int>> shapes = {
        {0, 5, 5}, {5, 0, 5}, {5, 5, 0}, {0, 0, 0}, {1, 1, 1}, {1, 7, 3}, {7, 1, 3}, {3, 3, 1},
        {37, 53, 41}, {101, 97, 103}, {200, 13, 300}, {13, 200, 300}, {257, 255, 259},
    };
    for (int dm : {-1, 0, 1})
        for (int dn : {-1, 0, 1})
            for (int k : {1, 17})
                shapes.push_back({MR + dm, NR + dn, k});
    for (const auto& s : shapes)
        for (int pad : {0, 3})
            for (int t : {1, T})
                check_gemm(s[0], s[1], s[2], pad, t, nullptr, rng);

    // 2. Tiny blocking: many partial kc/nc/mc blocks; every thread count.
    const GemmBlocking tiny{2 * MR, 7, 3 * NR};
    const GemmBlocking odd{MR + 1, 5, NR + 1};  // rounded up to tile multiples
    const GemmBlocking ones{1, 1, 1};
    for (int t : thread_counts) {
        check_gemm(100, 90, 50, 2, t, &tiny, rng);
        check_gemm(3 * MR + 1, 5 * NR - 1, 23, 1, t, &odd, rng);
        check_gemm(29, 31, 13, 0, t, &ones, rng);
    }

    // 3. Larger problems and thin (Davidson-like) shapes at full threads.
    check_gemm(500, 600, 700, 1, T, nullptr, rng);
    check_gemm(1000, 4, 1000, 0, T, nullptr, rng);
    check_gemm(1000, 1, 1000, 0, T, nullptr, rng);
    check_gemm(4, 1000, 1000, 0, T, nullptr, rng);
    check_gemm(1, 1, 5000, 0, T, nullptr, rng);

    // n > default nc: several column blocks (jc loop), the last one partial.
    check_gemm(2 * MR + 1, def.nc + 3 * NR + 1, 70, 1, T, nullptr, rng);

    // 4. Repeated calls reuse the packing buffers; a smaller call after a larger one.
    check_gemm(300, 300, 300, 0, T, nullptr, rng);
    check_gemm(10, 10, 10, 0, T, nullptr, rng);

#ifdef _OPENMP
    // 4b. Concurrent calls from inside a parallel region. Each calling thread
    // has its own (thread_local) packing buffers; run once with nested
    // parallelism off (inner team of 1) and once on (inner teams of 2).
    for (int levels : {1, 2}) {
        const int saved_levels = omp_get_max_active_levels();
        omp_set_max_active_levels(levels);
        constexpr int kCallers = 4, kRounds = 3;
        std::vector<Problem> probs;
        for (int c = 0; c < kCallers * kRounds; ++c)
            probs.push_back(make_problem(40 + 13 * c, 30 + 11 * c, 50 + 17 * c, c % 2, rng));
        std::vector<std::vector<double>> Cs;
        for (const Problem& p : probs)
            Cs.push_back(p.C0);
#pragma omp parallel num_threads(kCallers)
        {
            // Each thread does several calls of different sizes, so buffers grow and are reused.
            for (int c = omp_get_thread_num(); c < kCallers * kRounds; c += omp_get_num_threads()) {
                const Problem& p = probs[c];
                gemm(p.m, p.n, p.k, p.A.data(), p.lda, p.B.data(), p.ldb, Cs[c].data(), p.ldc, 2);
            }
        }
        omp_set_max_active_levels(saved_levels);
        for (int c = 0; c < kCallers * kRounds; ++c)
            verify(probs[c], Cs[c], "gemm inside a parallel region, max_active_levels=" +
                                        std::to_string(levels) + ", call " + std::to_string(c));
    }
#endif

    // 5. Argument validation.
    std::vector<double> buf(16, 0.0);
    expect_throw("negative m", [&] { gemm(-1, 2, 2, buf.data(), 2, buf.data(), 2, buf.data(), 2); });
    expect_throw("lda < k", [&] { gemm(2, 2, 3, buf.data(), 2, buf.data(), 2, buf.data(), 2); });
    expect_throw("ldb < n", [&] { gemm(2, 3, 2, buf.data(), 2, buf.data(), 2, buf.data(), 3); });
    expect_throw("ldc < n", [&] { gemm(2, 3, 2, buf.data(), 2, buf.data(), 3, buf.data(), 2); });
    expect_throw("null A", [&] { gemm(2, 2, 2, nullptr, 2, buf.data(), 2, buf.data(), 2); });

    // 6. Matrix::operator*.
    check_matrix_multiply(37, 29, 53, rng);
    check_matrix_multiply(64, 64, 64, rng);
    check_matrix_multiply(0, 3, 5, rng);
    check_matrix_multiply(4, 3, 0, rng);  // 4x0 times 0x3 is the 4x3 zero matrix
    expect_throw("Matrix::operator* mismatch", [&] { (void)(Matrix(2, 3) * Matrix(2, 3)); });

#ifdef CHECK_HUAIYU
    // 7. Huaiyu's single-threaded AVX-512 version (C += A*B, no leading dimensions).
    for (int depth_block : {7, 64, 256})
        for (const auto& s : std::vector<std::vector<int>>{
                 {16, 8, 5}, {37, 53, 41}, {15, 7, 3}, {101, 263, 300}, {1, 1, 1}})
            check_huaiyu(s[0], s[1], s[2], depth_block, rng);
#endif

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
