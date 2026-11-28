#include <iostream>
#include <vector>
#include "matrix_comp.h"
#include "davidson_new.h"
#include <algorithm>
#include <climits>
#include <limits>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>


Matrix<double> MatrixJacobi(Matrix<double>& A, Matrix<double>& Q,int max_iter=100,double tol=1e-12) {

    // This is a solver that the davidson method needs to call to the full diagonalization of the projected small matrix.
    if (A.rows() != A.cols() || max_iter < 0)
        throw std::invalid_argument("Jacobi requires a square matrix and nonnegative max_iter");

    const int n = A.rows();
    for (int k = 0; k < n; ++k)
        Q[k][k] = 1.0;
    double scale = 0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            scale = std::max(scale, std::abs(A[i][j]));
    const double tolerance = tol * scale;

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
        if (offdiag_max <= tolerance) {
            std::cout << "Jacobi converged after " << iter << " iterations with offdiag_max = " << offdiag_max << "\n";
            break;
        }

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

        // Accumulate Q <- Q J, the same column rotation applied to A above.
        for (int k = 0; k < n; ++k) {
            const double qki = Q[k][i], qkj = Q[k][j];
            Q[k][i] = c * qki - s * qkj;
            Q[k][j] = s * qki + c * qkj;
        }
        
    }
    return A;
}

// Jacobi rotation budget per entry of the projected matrix (typical need is ~3).
constexpr long long kJacobiRotationsPerEntry = 50;

// Only the first active rows of V are basis vectors. The remaining rows
// are workspace for this cycle's corrections.
std::pair<Matrix<double>, std::vector<double>> largest_ritz_pairs(
        const Matrix<double>& A, const Matrix<double>& V, const Matrix<double>& V_T, int active, int nroots) {
    Matrix full_projection = (V * A) * V_T;
    Matrix<double> projected(active, active);
    for (int b = 0; b < active; ++b)
        std::copy_n(full_projection[b], active, projected[b]);

    Matrix<double> Q(active, active);
    const int rotations = static_cast<int>(std::min<double>(
        INT_MAX, static_cast<double>(kJacobiRotationsPerEntry) * active * active));
    MatrixJacobi(projected, Q, rotations);

    std::vector<int> order(active);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + nroots, order.end(),
        [&](int i, int j) { return projected[i][i] > projected[j][j]; });

    Matrix<double> coefficients(V.rows(), nroots);
    std::vector<double> values(nroots);
    for (int root = 0; root < nroots; ++root) {
        values[root] = projected[order[root]][order[root]];
        for (int b = 0; b < active; ++b)
            coefficients[b][root] = Q[b][order[root]];
    }
    return {V_T * coefficients, std::move(values)};
}

// Each column of X is a full-space Ritz vector.
void compute_residuals(const Matrix<double>& X, const std::vector<double>& values,
                       Matrix<double>& residuals, std::vector<double>& norms) {
    std::fill(norms.begin(), norms.end(), 0.0);
    for (int i = 0; i < X.rows(); ++i) {
        for (int root = 0; root < X.cols(); ++root) {
            residuals[i][root] -= values[root] * X[i][root];
            norms[root] = std::hypot(norms[root], residuals[i][root]);
        }
    }
}

void restart_subspace(const Matrix<double>& X, Matrix<double>& V, Matrix<double>& V_T) {
    for (int i = 0; i < X.rows(); ++i) {
        for (int root = 0; root < X.cols(); ++root)
            V[root][i] = V_T[i][root] = X[i][root];
        std::fill(V_T[i] + X.cols(), V_T[i] + V.rows(), 0.0);
    }
    for (int b = X.cols(); b < V.rows(); ++b)
        std::fill_n(V[b], V.cols(), 0.0);
}

auto initialize_subspace(int size, const Matrix<double>& M, int expand_iter) {
    int cols = M.cols();
    double* diag = new double[cols+1];
    for (int i = 0; i < cols; ++i)
    {
        diag[i] = M[i][i];
    }
    diag[cols] = -std::numeric_limits<double>::infinity(); // sentinel for largest values
    int* diag_ind = new int[size];
    std::fill_n(diag_ind, size, cols); 
    for (int i = 0; i < cols; ++i)
    {
        if (diag[i] >= diag[diag_ind[0]])
        {
            diag_ind[0] = i;
            int j=1;
            while(j < size && diag[i] >= diag[diag_ind[j]])
            {

                diag_ind[j-1]=diag_ind[j];
                diag_ind[j]=i;
                ++j;

            }
        }
    }

    

    Matrix<double> subspace(size+ expand_iter*size, cols);
    Matrix<double> subspace_T(cols, size+ expand_iter*size);
    for (int i=0; i < size; ++i)
        {
            subspace[i][diag_ind[i]] = 1;
            subspace_T[diag_ind[i]][i] = 1;

        }

    delete[] diag;
    delete[] diag_ind;
    return std::pair{std::move(subspace), std::move(subspace_T)};
}

double preconditioner(double diag_val, double eigval, double eta=1e-6) {
    const double d = diag_val - eigval;
    return d / (d*d + eta*eta);
}

int append_subspace(const Matrix<double>& A, Matrix<double> & subspace, Matrix<double> & subspace_T, const Matrix<double> & new_vectors, const Matrix<double> & M, int start, int nroots) {


// M is the diagonalized matrix in the subspace
int cols = M.cols();
    double* diag = new double[cols+1];
    for (int i = 0; i < cols; ++i)
    {
        diag[i] = M[i][i];
    }
    diag[cols] = -std::numeric_limits<double>::infinity(); // sentinel for largest values
    int* diag_ind = new int[nroots];
    std::fill_n(diag_ind, nroots, cols); 
    for (int i = 0; i < cols; ++i)
    {
        if (diag[i] >= diag[diag_ind[0]])
        {
            diag_ind[0] = i;
            int j=1;
            while(j < nroots && diag[i] >= diag[diag_ind[j]])
            {

                diag_ind[j-1]=diag_ind[j];
                diag_ind[j]=i;
                ++j;

            }
        }
    }

    Matrix<double> selSubEigvec(subspace.rows(), nroots);
    Matrix<double> selSubEigval(nroots, nroots);



    //compute new subspace vectors
    for (int j=0; j< nroots; ++j)
    {
        for (int i=0; i< new_vectors.rows(); ++i)
        {
            selSubEigvec[i][j] = new_vectors[i][diag_ind[j]];

        }
        selSubEigval[j][j]=M[diag_ind[j]][diag_ind[j]];
    }

    // compute new full space vectors from subspace vectors

    Matrix<double> subspace_exp_T(A.rows(),nroots);
    subspace_exp_T = subspace_T*selSubEigvec;



    //construct residual vector from ritz pair

    Matrix t= A*subspace_exp_T-subspace_exp_T*selSubEigval;
    Matrix<double> t_T(t.cols(), t.rows());
    //rescale the vectors

    int inactive = 0;
    for (int i = 0; i < t.cols(); ++i)
    {
        for(int j=0; j < t.rows(); ++j)
        {

            t[j][i]=preconditioner(A[j][j], selSubEigval[i][i])*t[j][i];
            t_T[i][j] = t[j][i];

        }
    }






    // orthogonalize the residual with the current subspace, normalize and add into the subspace
   

    for (int i = 0; i < t.cols(); ++i)

    {
        auto projector = subspace_T*subspace;
        Matrix<double> t_i(t.rows(), 1);
        std::copy_n(t_T[i], t.rows(), t_i[0]);
        const double original_norm = std::sqrt(
            std::inner_product(t_T[i], t_T[i] + t.rows(), t_T[i], 0.0));

        //orthogonalize t_i against the current subspace
        for (int pass = 0; pass < 2; ++pass)
            t_i = t_i - projector* t_i;

        //normalize t_i
        double norm = 0.0;
        for (int j = 0; j < t_i.rows(); ++j)
        {
            norm += t_i[j][0]*t_i[j][0];
        }
        norm = std::sqrt(norm);
        if (!(norm > std::max(1e-12, 1e-8 * original_norm)) || !std::isfinite(norm)) {
            ++inactive;
            continue;
        }
        
        //add t_i into the subspace
        for (int j=0; j < t_i.rows(); ++j)
        {
            t_i[j][0] /= norm;
            subspace[start+i-inactive][j]=t_i[j][0];
            subspace_T[j][start+i-inactive]=t_i[j][0];
        }



    }





    return inactive;
}

bool block_search(const Matrix<double>& A, const Matrix<double>& guess, int nroots,
                  double tol, int max_iter,
                  double (* /*select_shift*/)(double), int expand_iter, int /*expand_size*/) {
    const int n = A.rows();
    if (A.cols() != n)
        throw std::invalid_argument("Davidson requires a square matrix");
    if (guess.rows() != n)
        throw std::invalid_argument("Davidson guess must have one row per matrix row");
    if (nroots < 1 || nroots > n)
        throw std::invalid_argument("Davidson nroots must be in [1, n]");
    if (!(tol > 0) || max_iter < 1)
        throw std::invalid_argument("Davidson requires tol > 0 and max_iter >= 1");
    if (expand_iter < 1 || (1LL + expand_iter) * nroots > INT_MAX)
        throw std::invalid_argument("Invalid Davidson expansion count");

    auto [V, V_T] = initialize_subspace(nroots, A, expand_iter);
    std::vector<double> norms(nroots);

    for (int iter=1; iter<=max_iter; ++iter) {
        int active = nroots;
        
        
        
        for(int j=1; j<=expand_iter && active<n; ++j){
            Matrix U=V*A;
            Matrix full_projection=U*V_T;
            Matrix<double> A_sub(active, active);
            for (int b=0; b<active; ++b)
                std::copy_n(full_projection[b], active, A_sub[b]);
            Matrix<double> Q(active, active);
            Matrix S=MatrixJacobi(A_sub, Q);

            const int count = std::min(nroots, n-active);
            const int inactive = append_subspace(A, V, V_T, Q, S, active, count);
            active += count-inactive;
            if (inactive == count)
                break;


            

            
        



        }

        auto [X, values] = largest_ritz_pairs(A, V, V_T, active, nroots);
        Matrix residuals = A * X;
        compute_residuals(X, values, residuals, norms);
        if (std::all_of(norms.begin(), norms.end(), [tol](double norm) {
                return std::isfinite(norm) && norm < tol;
            }))
            return true;
        restart_subspace(X, V, V_T);
    }
    return false;
}
