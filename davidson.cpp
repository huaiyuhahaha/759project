#include <iostream>
#include <vector>
#include "matrix.h"
#include "davidson.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>


Matrix initialize_vector(int size, const Matrix& M) {
    int cols=M.cols();
    Matrix U(cols, size);
    double* diag=new double[cols];
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

Matrix MatrixJacobi(Matrix& A, int max_iter) {

    // This is a solver that the davidson method needs to call to the full diagonalization of the projected small matrix.
    if (A.rows() != A.cols() || max_iter < 0)
        throw std::invalid_argument("Jacobi requires a square matrix and nonnegative max_iter");

    const int n = A.rows();
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
    }
    return A;
}
