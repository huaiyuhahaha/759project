// Davidson benchmark on seeded dense symmetric diagonally dominant matrices.
//
//   davidson_bench <n> <serial|openmp>   time the C++ solver
//   davidson_bench <n> dump <file>       write the benchmark inputs for PySCF
//
// Times only the Davidson() call: one warm-up solve, then kRuns measured solves
// on the same matrix and guess. Prints one JSON object with the timings, the
// median per-kernel profile, and an independent check of the returned pairs.
// dump writes A (n x n) then the guess (n x kRoots), both row-major
// little-endian float64, so another solver gets exactly the same bytes; both
// modes print the same input_checksum of those bytes.
// Run through bench/run_davidson_bench.py, which fixes build flags and threads.

#include "davidson.h"
#include "matrix.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr int kRoots = 4;
constexpr double kTol = 1e-8;  // Davidson residual-norm stopping criterion
constexpr int kMaxIter = 100;
constexpr int kWarmup = 1;
constexpr int kRuns = 5;
constexpr std::uint64_t kSeed = 759;
// Independent residual must be within this factor of the solver tolerance.
constexpr double kResidualSlack = 10.0;

// A_ii = i + 1 and A_ij = A_ji = u / n with u uniform in [-1, 1), so every
// off-diagonal row sum is below (n - 1) / n < 1 <= A_ii: strictly diagonally
// dominant. mt19937_64 and the bit-to-double mapping are fully specified, so
// the matrix is identical across compilers and runs.
Matrix make_matrix(int n) {
    std::mt19937_64 rng(kSeed);
    Matrix A(n, n);
    for (int i = 0; i < n; ++i) {
        A[i][i] = i + 1.0;
        for (int j = i + 1; j < n; ++j) {
            const double u = static_cast<double>(rng() >> 11) * 0x1.0p-53;  // [0, 1)
            A[i][j] = A[j][i] = (2.0 * u - 1.0) / n;
        }
    }
    return A;
}

// Position-weighted sum of the 64-bit words of A then guess (row-major), mod
// 2^64. Every weight is odd, so changing any single value changes the sum.
// bench/pyscf_davidson_bench.py computes the same function with NumPy.
std::uint64_t input_checksum(const Matrix& A, const Matrix& guess) {
    std::uint64_t h = 0, pos = 0;
    auto add = [&](const Matrix& M) {
        for (int i = 0; i < M.rows(); ++i)
            for (int j = 0; j < M.cols(); ++j) {
                std::uint64_t w;
                std::memcpy(&w, &M[i][j], sizeof w);
                h += w * (2 * pos++ + 1);
            }
    };
    add(A);
    add(guess);
    return h;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const std::size_t h = v.size() / 2;
    return v.size() % 2 ? v[h] : 0.5 * (v[h - 1] + v[h]);
}

using Clock = std::chrono::steady_clock;

int run(int n, Backend backend, const char* backend_name) {
    const Matrix A = make_matrix(n);
    const Matrix guess = initialize_vector(kRoots, A);

    std::vector<double> times;
    std::vector<DavidsonProfile> profiles;
    std::vector<double> first_eigs;
    bool deterministic = true;
    DavidsonResult res;
    for (int run = 0; run < kWarmup + kRuns; ++run) {
        const auto t0 = Clock::now();
        DavidsonResult r = Davidson(A, guess, kRoots, kTol, kMaxIter, backend);
        const double t = std::chrono::duration<double>(Clock::now() - t0).count();
        if (run >= kWarmup) {
            times.push_back(t);
            profiles.push_back(r.profile);
        }
        if (run == 0)
            first_eigs = r.eigenvalues;
        else if (r.eigenvalues != first_eigs)
            deterministic = false;
        res = std::move(r);
    }

    // Independent check: plain serial loops over A, not the solver's kernels.
    const Matrix& X = res.eigenvectors;
    double max_resid = 0, orth_sq = 0;
    for (int k = 0; k < kRoots; ++k) {
        double r2 = 0;
        for (int i = 0; i < n; ++i) {
            const double* row = A[i];
            double ax = 0;
            for (int j = 0; j < n; ++j)
                ax += row[j] * X[j][k];
            const double r = ax - res.eigenvalues[k] * X[i][k];
            r2 += r * r;
        }
        const double resid = std::sqrt(r2);
        if (!(resid <= max_resid))
            max_resid = resid;  // also propagates NaN
        for (int l = 0; l < kRoots; ++l) {
            double g = 0;
            for (int i = 0; i < n; ++i)
                g += X[i][k] * X[i][l];
            const double e = g - (k == l ? 1.0 : 0.0);
            orth_sq += e * e;
        }
    }
    const double orth_err = std::sqrt(orth_sq);
    const bool ok = res.converged && deterministic && max_resid < kResidualSlack * kTol &&
                    orth_err < 1e-8;

    auto kernel = [&](double DavidsonProfile::*field) {
        std::vector<double> v;
        for (const DavidsonProfile& p : profiles)
            v.push_back(p.*field);
        return median(v);
    };
#ifdef _OPENMP
    const int threads = backend == Backend::OpenMP ? omp_get_max_threads() : 1;
#else
    const int threads = 1;
#endif

    std::printf("{\"n\": %d, \"backend\": \"%s\", \"threads\": %d, \"k\": %d, \"tol\": %g, "
                "\"seed\": %llu, \"warmup\": %d, \"runs\": %d, \"times\": [",
                n, backend_name, threads, kRoots, kTol,
                static_cast<unsigned long long>(kSeed), kWarmup, kRuns);
    for (std::size_t i = 0; i < times.size(); ++i)
        std::printf("%s%.9g", i ? ", " : "", times[i]);
    std::printf("], \"median\": %.9g, \"profile\": {\"matvec\": %.9g, \"orthogonalize\": %.9g, "
                "\"project\": %.9g, \"eigensolve\": %.9g, \"ritz\": %.9g, "
                "\"precondition\": %.9g}, ",
                median(times), kernel(&DavidsonProfile::matvec),
                kernel(&DavidsonProfile::orthogonalize), kernel(&DavidsonProfile::project),
                kernel(&DavidsonProfile::eigensolve), kernel(&DavidsonProfile::ritz),
                kernel(&DavidsonProfile::precondition));
    std::printf("\"converged\": %s, \"deterministic\": %s, \"iterations\": %d, "
                "\"expansions\": %d, \"dropped\": %d, \"subspace\": %d, \"eigenvalues\": [",
                res.converged ? "true" : "false", deterministic ? "true" : "false",
                res.iterations, res.expansions, res.dropped_corrections, res.subspace_size);
    for (int k = 0; k < kRoots; ++k)
        std::printf("%s%.17g", k ? ", " : "", res.eigenvalues[k]);
    std::printf("], \"max_residual\": %.6g, \"orth_error\": %.6g, \"input_checksum\": \"%016llx\", "
                "\"ok\": %s}\n",
                max_resid, orth_err,
                static_cast<unsigned long long>(input_checksum(A, guess)), ok ? "true" : "false");
    return ok ? 0 : 1;
}

int dump(int n, const char* path) {
    const Matrix A = make_matrix(n);
    const Matrix guess = initialize_vector(kRoots, A);
    std::FILE* f = std::fopen(path, "wb");
    if (!f) {
        std::fprintf(stderr, "cannot open '%s' for writing\n", path);
        return 1;
    }
    bool ok = true;
    for (int i = 0; i < n; ++i)
        ok = ok && std::fwrite(A[i], sizeof(double), n, f) == static_cast<std::size_t>(n);
    for (int i = 0; i < n; ++i)
        ok = ok && std::fwrite(guess[i], sizeof(double), kRoots, f) ==
                       static_cast<std::size_t>(kRoots);
    ok = (std::fclose(f) == 0) && ok;
    if (!ok) {
        std::fprintf(stderr, "write to '%s' failed\n", path);
        return 1;
    }
    std::printf("{\"n\": %d, \"k\": %d, \"seed\": %llu, \"input_checksum\": \"%016llx\"}\n", n,
                kRoots, static_cast<unsigned long long>(kSeed),
                static_cast<unsigned long long>(input_checksum(A, guess)));
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const bool dumping = argc == 4 && std::strcmp(argv[2], "dump") == 0;
    if (argc != 3 && !dumping) {
        std::fprintf(stderr, "usage: %s <n> <serial|openmp>\n       %s <n> dump <file>\n",
                     argv[0], argv[0]);
        return 2;
    }
    char* end = nullptr;
    const long n = std::strtol(argv[1], &end, 10);
    if (*end != '\0' || n < kRoots || n > 1 << 15) {
        std::fprintf(stderr, "bad n '%s'\n", argv[1]);
        return 2;
    }
    if (dumping) {
        try {
            return dump(static_cast<int>(n), argv[3]);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "exception: %s\n", e.what());
            return 1;
        }
    }
    Backend backend;
    if (std::strcmp(argv[2], "serial") == 0)
        backend = Backend::Serial;
    else if (std::strcmp(argv[2], "openmp") == 0)
        backend = Backend::OpenMP;
    else {
        std::fprintf(stderr, "bad backend '%s'\n", argv[2]);
        return 2;
    }
    try {
        return run(static_cast<int>(n), backend, argv[2]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "exception: %s\n", e.what());
        return 1;
    }
}
