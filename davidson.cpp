#include <iostream>
#include <vector>
#include "matrix.h"
#include "davidson.h"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <utility>


Matrix initialize_vector(int size, const Matrix& M) {
    int cols=M.cols();
    Matrix U(cols, size);
    std::vector<double> diag(cols);
    for (int i = 0; i < cols; ++i) {
        diag[i] = M[i][i];
    }
    
    
std::vector<int> idx(cols);
std::iota(idx.begin(), idx.end(), 0);

std::partial_sort(
    idx.begin(),
    idx.begin() + size,
    idx.end(),
    [&](int i, int j) {
        return diag[i] < diag[j];
    }
);
    
    for (int i = 0; i < size; ++i) {

        U[idx[i]][i] = 1.0;
        
    }
    return U;
}

Matrix MatrixJacobi(Matrix& A, int max_iter, Matrix* eigenvectors) {

    // This is a solver that the davidson method needs to call to the full diagonalization of the projected small matrix.
    if (A.rows() != A.cols() || max_iter < 0)
        throw std::invalid_argument("Jacobi requires a square matrix and nonnegative max_iter");

    const int n = A.rows();
    if (eigenvectors) {
        *eigenvectors = Matrix(n, n);
        for (int k = 0; k < n; ++k)
            (*eigenvectors)[k][k] = 1.0;
    }
    double scale = 0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            scale = std::max(scale, std::abs(A[i][j]));
    const double tolerance = 1e-12 * scale;

    for (int iter = 0; iter < max_iter; ++iter) {
        int i = 0, j = 0;
        double offdiag_max = 0;
        for (int alpha = 0; alpha < n; ++alpha) {
            for (int beta = alpha + 1; beta < n; ++beta) {
                if (std::abs(A[alpha][beta]) > offdiag_max) {
                    offdiag_max = std::abs(A[alpha][beta]);
                    i = alpha;
                    j = beta;
                }
            }
        }
        if (offdiag_max <= tolerance)
            break;

        const double aii = A[i][i], ajj = A[j][j], aij = A[i][j];
        const double tau = (0.5 * ajj - 0.5 * aii) / aij;
        const double t = std::copysign(1.0, tau)
                       / (std::abs(tau) + std::hypot(1.0, tau));
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double s = t * c;

        for (int k = 0; k < n; ++k) {
            if (k == i || k == j)
                continue;
            const double aki = A[k][i], akj = A[k][j];
            A[k][i] = A[i][k] = c * aki - s * akj;
            A[k][j] = A[j][k] = s * aki + c * akj;
        }
        A[i][i] = aii - t * aij;
        A[j][j] = ajj + t * aij;
        A[i][j] = A[j][i] = 0;

        if (eigenvectors) {
            // Accumulate Q <- Q J, the same column rotation applied to A above.
            Matrix& Q = *eigenvectors;
            for (int k = 0; k < n; ++k) {
                const double qki = Q[k][i], qkj = Q[k][j];
                Q[k][i] = c * qki - s * qkj;
                Q[k][j] = s * qki + c * qkj;
            }
        }
    }
    return A;
}

namespace {

using Vec = std::vector<double>;

// |theta - A_ii| below this is clamped (sign kept) in the diagonal preconditioner.
constexpr double kDenomFloor = 1e-8;
// Jacobi rotation budget per entry of the projected matrix (typical need is ~3).
constexpr long long kJacobiRotationsPerEntry = 50;

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

double dot(const Vec& a, const Vec& b) {
    double s = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        s += a[i] * b[i];
    return s;
}

Vec matvec(const Matrix& A, const Vec& v) {
    Vec out(v.size());
    for (int i = 0; i < A.rows(); ++i) {
        const double* row = A[i];
        double s = 0;
        for (int j = 0; j < A.cols(); ++j)
            s += row[j] * v[j];
        out[i] = s;
    }
    return out;
}

// OpenMP backend for the dominant kernel. The row loop body is kept identical
// to matvec() so the comparison isolates threading; each row is summed in the
// same order by one thread, so the result matches matvec() bitwise.
Vec matvec_omp(const Matrix& A, const Vec& v) {
    Vec out(v.size());
    const int rows = A.rows();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const double* row = A[i];
        double s = 0;
        for (int j = 0; j < A.cols(); ++j)
            s += row[j] * v[j];
        out[i] = s;
    }
    return out;
}

}  // namespace

bool orthonormalize_against(const std::vector<Vec>& basis, Vec& v) {
    const double norm0 = std::sqrt(dot(v, v));
    if (!(norm0 > 0) || !std::isfinite(norm0))
        return false;
    for (int pass = 0; pass < 2; ++pass) {
        for (const Vec& b : basis) {
            const double proj = dot(b, v);
            for (std::size_t i = 0; i < v.size(); ++i)
                v[i] -= proj * b[i];
        }
    }
    const double norm = std::sqrt(dot(v, v));
    if (!(norm > kLinDepTol * norm0))
        return false;
    for (double& x : v)
        x /= norm;
    return true;
}

DavidsonResult Davidson(const Matrix& A, const Matrix& guess, int nroots,
                        double tol, int max_iter, Backend backend) {
    const int n = A.rows();
    if (A.cols() != n)
        throw std::invalid_argument("Davidson requires a square matrix");
    if (guess.rows() != n)
        throw std::invalid_argument("Davidson guess must have one row per matrix row");
    if (nroots < 1 || nroots > n)
        throw std::invalid_argument("Davidson nroots must be in [1, n]");
    if (!(tol > 0) || max_iter < 1)
        throw std::invalid_argument("Davidson requires tol > 0 and max_iter >= 1");

    DavidsonResult result;
    DavidsonProfile& prof = result.profile;
    std::vector<Vec> V, AV;  // orthonormal basis vectors and A times each one
    auto add_to_basis = [&](Vec v) {
        auto t0 = Clock::now();
        const bool independent = orthonormalize_against(V, v);
        prof.orthogonalize += seconds_since(t0);
        if (!independent)
            return false;
        t0 = Clock::now();
        AV.push_back(backend == Backend::OpenMP ? matvec_omp(A, v) : matvec(A, v));
        prof.matvec += seconds_since(t0);
        V.push_back(std::move(v));
        return true;
    };

    for (int c = 0; c < guess.cols(); ++c) {
        Vec v(n);
        for (int i = 0; i < n; ++i)
            v[i] = guess[i][c];
        add_to_basis(std::move(v));
    }
    if (static_cast<int>(V.size()) < nroots)
        throw std::invalid_argument("Davidson guess needs at least nroots independent columns");

    result.eigenvalues.assign(nroots, 0.0);
    result.residual_norms.assign(nroots, 0.0);
    result.eigenvectors = Matrix(n, nroots);
    std::vector<Vec> residuals(nroots, Vec(n));

    for (int iter = 1; iter <= max_iter; ++iter) {
        const int m = static_cast<int>(V.size());

        // Rayleigh-Ritz: H = V^T A V = Q diag(theta) Q^T.
        auto t0 = Clock::now();
        Matrix H(m, m);
        for (int i = 0; i < m; ++i)
            for (int j = i; j < m; ++j)
                H[i][j] = H[j][i] = dot(V[i], AV[j]);
        prof.project += seconds_since(t0);

        t0 = Clock::now();
        Matrix Q(0, 0);
        const long long budget = kJacobiRotationsPerEntry * m * m;
        MatrixJacobi(H, static_cast<int>(std::min<long long>(budget, INT_MAX)), &Q);

        std::vector<int> order(m);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return H[a][a] < H[b][b]; });
        prof.eigensolve += seconds_since(t0);

        // Ritz pairs x = V y, residuals r = A V y - theta x for the nroots lowest.
        t0 = Clock::now();
        bool all_converged = true;
        for (int k = 0; k < nroots; ++k) {
            const int col = order[k];
            const double theta = H[col][col];
            Vec x(n, 0.0), ax(n, 0.0);
            for (int b = 0; b < m; ++b) {
                const double y = Q[b][col];
                for (int i = 0; i < n; ++i) {
                    x[i] += y * V[b][i];
                    ax[i] += y * AV[b][i];
                }
            }
            Vec& r = residuals[k];
            for (int i = 0; i < n; ++i) {
                r[i] = ax[i] - theta * x[i];
                result.eigenvectors[i][k] = x[i];
            }
            result.eigenvalues[k] = theta;
            result.residual_norms[k] = std::sqrt(dot(r, r));
            if (!(result.residual_norms[k] < tol))
                all_converged = false;
        }
        prof.ritz += seconds_since(t0);
        result.iterations = iter;
        result.subspace_size = m;
        if (all_converged) {
            result.converged = true;
            return result;
        }
        if (iter == max_iter)
            break;

        // Expand with diagonal-preconditioned corrections for unconverged roots.
        int added = 0;
        for (int k = 0; k < nroots; ++k) {
            if (result.residual_norms[k] < tol)
                continue;
            const auto tp = Clock::now();
            Vec delta(n);
            for (int i = 0; i < n; ++i) {
                double d = result.eigenvalues[k] - A[i][i];
                if (std::abs(d) < kDenomFloor)
                    d = std::copysign(kDenomFloor, d);
                delta[i] = residuals[k][i] / d;
            }
            prof.precondition += seconds_since(tp);
            if (add_to_basis(std::move(delta)))
                ++added;
            else
                ++result.dropped_corrections;
        }
        // If every correction was dependent, fall back to the raw residuals,
        // which are orthogonal to the basis in exact arithmetic.
        if (added == 0) {
            for (int k = 0; k < nroots; ++k)
                if (!(result.residual_norms[k] < tol) && add_to_basis(residuals[k]))
                    ++added;
        }
        if (added == 0)
            break;  // stagnated: no new independent direction
        ++result.expansions;
    }
    return result;
}
