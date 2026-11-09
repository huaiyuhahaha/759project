// End-to-end checks of the serial Davidson solver.
//
// Build and run with:  python3 checks/run_davidson_e2e.py [--sanitize]
// The driver computes the NumPy reference eigvalsh(A)[:2] for each case and runs
//     davidson_e2e <serial|openmp> <n> <ref0> <ref1>   one case on one backend
//     davidson_e2e lindep                              the dependent-vector drop criterion
// Exit status: 0 = PASS, 1 = FAIL, 2 = bad usage.

#include "davidson.h"
#include "matrix.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <vector>

namespace {

constexpr int kRoots = 2;
constexpr double kAccept = 1e-8;      // threshold for every reported error
constexpr double kSolverTol = 1e-10;  // Davidson residual-norm stopping criterion
constexpr int kMaxIter = 50;

struct Case {
    int n;
    bool must_stop_short_of_n;  // converge with final subspace size < n
};
// Fixture for every case: diagonal 1..n, both adjacent diagonals 0.2.
// n=8 fills the whole space; n=64 must converge in a proper subspace.
constexpr Case kCases[] = {{8, false}, {64, true}};

// Running maximum that keeps NaN once seen, so a NaN error can never pass.
double worst(double current, double value) {
    return (value > current || std::isnan(value)) ? value : current;
}

struct Checks {
    bool ok = true;
    void require(bool cond, const char* what) {
        if (!cond) {
            std::printf("  failed: %s\n", what);
            ok = false;
        }
    }
    int finish() const {
        std::printf("%s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
};

int run_case(const Case& c, Backend backend, const char* backend_name,
             const double (&ref)[kRoots]) {
    const int n = c.n;
    Matrix A(n, n);
    for (int i = 0; i < n; ++i) {
        A[i][i] = i + 1;
        if (i + 1 < n)
            A[i][i + 1] = A[i + 1][i] = 0.2;
    }

    const Matrix guess = initialize_vector(kRoots, A);
    const DavidsonResult res = Davidson(A, guess, kRoots, kSolverTol, kMaxIter, backend);
    const Matrix& X = res.eigenvectors;

    std::printf("%s, n=%d, k=%d: converged=%s  iterations=%d  expansions=%d  "
                "dropped corrections=%d  final subspace=%d\n",
                backend_name, n, kRoots, res.converged ? "yes" : "no", res.iterations,
                res.expansions, res.dropped_corrections, res.subspace_size);

    Checks checks;
    if (static_cast<int>(res.eigenvalues.size()) != kRoots || X.rows() != n ||
        X.cols() != kRoots) {
        checks.require(false, "result has the expected shape");
        return checks.finish();
    }

    // All errors below are recomputed from A and the returned pairs; the
    // solver's own residual_norms are not used.
    double eig_err = 0, max_resid = 0, orth_sq = 0;
    for (int k = 0; k < kRoots; ++k) {
        const double lambda = res.eigenvalues[k];
        const double err = std::abs(lambda - ref[k]);
        eig_err = worst(eig_err, err);

        double r2 = 0;
        for (int i = 0; i < n; ++i) {
            double ax = 0;
            for (int j = 0; j < n; ++j)
                ax += A[i][j] * X[j][k];
            const double r = ax - lambda * X[i][k];
            r2 += r * r;
        }
        const double resid = std::sqrt(r2);
        max_resid = worst(max_resid, resid);
        std::printf("  root %d: lambda=%.15f  numpy=%.15f  |diff|=%.3e  ||Ax-lambda x||=%.3e\n",
                    k, lambda, ref[k], err, resid);

        for (int l = 0; l < kRoots; ++l) {
            double g = 0;
            for (int i = 0; i < n; ++i)
                g += X[i][k] * X[i][l];
            const double e = g - (k == l ? 1.0 : 0.0);
            orth_sq += e * e;
        }
    }
    const double orth_err = std::sqrt(orth_sq);

    std::printf("max eigenvalue error   = %.3e\n", eig_err);
    std::printf("max ||Ax - lambda x||2 = %.3e\n", max_resid);
    std::printf("||X^T X - I||_F        = %.3e\n", orth_err);

    checks.require(res.converged, "solver converged");
    checks.require(res.expansions >= 1, "at least one basis expansion");
    if (c.must_stop_short_of_n)
        checks.require(res.subspace_size < n, "final subspace size < n");
    // Specific to this tridiagonal fixture (not a general solver property):
    // unit-vector guesses keep the basis spanned by e_0..e_{m-1}, so both roots'
    // residuals lie along e_m and the second correction is dependent on the
    // first one appended in the same cycle: the loop must drop it.
    checks.require(res.dropped_corrections >= 1, "at least one dependent correction dropped");
    checks.require(eig_err < kAccept, "max eigenvalue error < 1e-8");
    checks.require(max_resid < kAccept, "max ||Ax - lambda x||_2 < 1e-8");
    checks.require(orth_err < kAccept, "||X^T X - I||_F < 1e-8");
    return checks.finish();
}

// orthonormalize_against must drop a candidate exactly when its remaining norm
// after orthogonalization is tiny relative to its original norm, at any scale.
int run_lindep_check() {
    const double r2 = std::sqrt(2.0), r3 = std::sqrt(3.0), r6 = std::sqrt(6.0);
    // Orthonormal basis b0, b1 of a plane in R^3, and unit w orthogonal to both.
    const std::vector<std::vector<double>> basis = {{1 / r2, 1 / r2, 0},
                                                    {1 / r6, -1 / r6, 2 / r6}};
    const std::vector<double> w = {1 / r3, -1 / r3, -1 / r3};

    // v = s (3 b0 - 4 b1 + 5 rho w) has original norm 5 s sqrt(1 + rho^2) and
    // remaining norm 5 s rho, so remaining/original ~= rho.
    const double rhos[] = {0.0, 1e-12, 0.1 * kLinDepTol, 10 * kLinDepTol, 1e-3};
    const double scales[] = {1e-6, 1.0, 1e6};

    std::printf("drop threshold kLinDepTol = %.0e (remaining/original norm)\n", kLinDepTol);
    Checks checks;
    for (double rho : rhos) {
        for (double s : scales) {
            std::vector<double> v(3);
            for (int i = 0; i < 3; ++i)
                v[i] = s * (3 * basis[0][i] - 4 * basis[1][i] + 5 * rho * w[i]);
            const bool expect_kept = rho > kLinDepTol;
            const bool kept = orthonormalize_against(basis, v);
            std::printf("  remaining/original ~ %-7.0e scale %-5.0e: %-7s (expected %s)\n",
                        rho, s, kept ? "kept" : "dropped", expect_kept ? "kept" : "dropped");
            checks.require(kept == expect_kept, "drop decision matches the relative criterion");
            if (kept) {
                double norm2 = 0, along_w = 0, max_overlap = 0;
                for (int i = 0; i < 3; ++i) {
                    norm2 += v[i] * v[i];
                    along_w += v[i] * w[i];
                }
                for (const auto& b : basis) {
                    double o = 0;
                    for (int i = 0; i < 3; ++i)
                        o += b[i] * v[i];
                    max_overlap = worst(max_overlap, std::abs(o));
                }
                checks.require(std::abs(std::sqrt(norm2) - 1) < 1e-12, "kept vector has unit norm");
                checks.require(max_overlap < 1e-12, "kept vector is orthogonal to the basis");
                checks.require(std::abs(along_w) > 1 - 1e-6, "kept vector points along w");
            }
        }
    }
    return checks.finish();
}

int usage(const char* prog) {
    std::fprintf(stderr, "usage: %s <serial|openmp> <n> <ref0> <ref1>   (NumPy eigvalsh(A)[:2])\n"
                         "       %s lindep\n", prog, prog);
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::strcmp(argv[1], "lindep") == 0)
            return run_lindep_check();
        if (argc != 3 + kRoots)
            return usage(argv[0]);

        Backend backend;
        if (std::strcmp(argv[1], "serial") == 0)
            backend = Backend::Serial;
        else if (std::strcmp(argv[1], "openmp") == 0)
            backend = Backend::OpenMP;
        else
            return usage(argv[0]);

        char* end = nullptr;
        const long n = std::strtol(argv[2], &end, 10);
        const Case* found = nullptr;
        for (const Case& c : kCases)
            if (*end == '\0' && c.n == n)
                found = &c;
        if (!found) {
            std::fprintf(stderr, "no case defined for n='%s'\n", argv[2]);
            return 2;
        }
        double ref[kRoots];
        for (int k = 0; k < kRoots; ++k) {
            const char* arg = argv[3 + k];
            ref[k] = std::strtod(arg, &end);
            if (end == arg || *end != '\0' || !std::isfinite(ref[k])) {
                std::fprintf(stderr, "bad reference value '%s'\n", arg);
                return 2;
            }
        }
        return run_case(*found, backend, argv[1], ref);
    } catch (const std::exception& e) {
        std::printf("FAIL: exception: %s\n", e.what());
        return 1;
    }
}
