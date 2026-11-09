#ifndef DAVIDSON_H
#define DAVIDSON_H

#include <vector>

#include "matrix.h"

Matrix initialize_vector(int size, const Matrix& M);
// Diagonalize a real symmetric A in place using at most max_iter rotations.
// If eigenvectors is non-null it receives the accumulated rotations Q, so that
// column k of Q is the eigenvector for the final A[k][k] (original A = Q diag(A) Q^T).
Matrix MatrixJacobi(Matrix& A, int max_iter, Matrix* eigenvectors = nullptr);

// A candidate basis vector whose norm after orthogonalization is <= this
// fraction of its original norm is treated as linearly dependent and dropped.
constexpr double kLinDepTol = 1e-8;

// Orthogonalize v against an orthonormal basis (two Gram-Schmidt passes) and
// normalize it. Returns false (v is then unusable) if v is zero, non-finite,
// or its remaining norm is <= kLinDepTol times its original norm.
bool orthonormalize_against(const std::vector<std::vector<double>>& basis,
                            std::vector<double>& v);

// Serial is the validated baseline. OpenMP runs the dominant measured kernel,
// the A v matvec, across threads (OMP_NUM_THREADS); all other kernels are
// shared with Serial. Without -fopenmp the OpenMP backend runs single-threaded.
enum class Backend { Serial, OpenMP };

// Wall-clock seconds spent in each kernel during one Davidson call.
struct DavidsonProfile {
    double matvec = 0;         // A v for every vector added to the basis
    double orthogonalize = 0;  // Gram-Schmidt of guesses and corrections
    double project = 0;        // H = V^T (A V)
    double eigensolve = 0;     // Jacobi on H, then sorting
    double ritz = 0;           // Ritz vectors, A x, residuals and their norms
    double precondition = 0;   // diagonal-preconditioned corrections
};

struct DavidsonResult {
    DavidsonProfile profile;
    bool converged = false;              // every requested residual norm < tol
    int iterations = 0;                  // project/solve/residual cycles performed
    int expansions = 0;                  // cycles that appended at least one basis vector
    int dropped_corrections = 0;         // corrections rejected by orthonormalize_against
    int subspace_size = 0;               // basis size used in the final cycle
    std::vector<double> eigenvalues;     // nroots lowest Ritz values, ascending
    std::vector<double> residual_norms;  // ||A x_k - eigenvalues[k] x_k||_2
    Matrix eigenvectors{0, 0};           // n x nroots; column k pairs with eigenvalues[k]
};

// Serial Davidson for the nroots lowest eigenpairs of a real symmetric A.
// guess (n x k, k >= nroots) seeds the search basis, e.g. initialize_vector(k, A).
// Stops when every residual norm is below tol (converged), or unconverged when
// max_iter cycles have run or no new independent direction can be added.
DavidsonResult Davidson(const Matrix& A, const Matrix& guess, int nroots,
                        double tol, int max_iter, Backend backend = Backend::Serial);

#endif
