#include "davidson.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err));                             \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

namespace {

constexpr int kTile = 16;
constexpr int kBlockSize = 256;
constexpr double kPrecondEps = 1e-12;

using Clock = std::chrono::steady_clock;
double elapsed_sec(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}


// Warp & Block Reduction Helper
__device__ double block_reduce_sum(double val) {
    __shared__ double warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (lane == 0) warp_sums[warp_id] = val;
    __syncthreads();

    int num_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < num_warps) ? warp_sums[lane] : 0.0;
    if (warp_id == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }
    return val;
}


__global__ void matvec_block_kernel(const double* __restrict__ A,
                                    const double* __restrict__ V_block,
                                    double* __restrict__ W,
                                    int n, int b) {
    __shared__ double tileA[kTile][kTile];
    __shared__ double tileV[kTile][kTile];

    int row = blockIdx.y * kTile + threadIdx.y;
    int col = blockIdx.x * kTile + threadIdx.x;

    double sum = 0.0;
    int num_tiles = (n + kTile - 1) / kTile;

    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * kTile + threadIdx.x;
        int v_row = t * kTile + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] =
            (row < n && a_col < n) ? A[row * n + a_col] : 0.0;
        tileV[threadIdx.y][threadIdx.x] =
            (col < b && v_row < n) ? V_block[col * n + v_row] : 0.0;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < kTile; ++k) {
            sum += tileA[threadIdx.y][k] * tileV[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < n && col < b) {
        W[col * n + row] = sum;
    }
}

__global__ void project_kernel(const double* __restrict__ V,
                               const double* __restrict__ AV,
                               double* __restrict__ H,
                               int n, int m) {
    int i = blockIdx.y;
    int j = blockIdx.x;
    if (i >= m || j >= m || j < i) return;

    const double* vi = V + static_cast<size_t>(i) * n;
    const double* avj = AV + static_cast<size_t>(j) * n;

    double partial = 0.0;
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        partial += vi[k] * avj[k];
    }

    double total = block_reduce_sum(partial);
    if (threadIdx.x == 0) {
        H[i * m + j] = total;
        H[j * m + i] = total;
    }
}

__global__ void ritz_vectors_kernel(const double* __restrict__ V,
                                    const double* __restrict__ AV,
                                    const double* __restrict__ Q,
                                    double* __restrict__ X,
                                    double* __restrict__ AX,
                                    int n, int m, int nroots) {
    extern __shared__ double s_Q[];
    int q_size = m * nroots;
    for (int idx = threadIdx.x; idx < q_size; idx += blockDim.x) {
        s_Q[idx] = Q[idx];
    }
    __syncthreads();

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    for (int r = 0; r < nroots; ++r) {
        double xr = 0.0;
        double axr = 0.0;
        for (int k = 0; k < m; ++k) {
            double q_kr = s_Q[k * nroots + r];
            xr += V[static_cast<size_t>(k) * n + row] * q_kr;
            axr += AV[static_cast<size_t>(k) * n + row] * q_kr;
        }
        X[static_cast<size_t>(r) * n + row] = xr;
        AX[static_cast<size_t>(r) * n + row] = axr;
    }
}

__global__ void fused_residual_precond_kernel(const double* __restrict__ X_r,
                                              const double* __restrict__ AX_r,
                                              const double* __restrict__ diagA,
                                              double* __restrict__ T_r,
                                              double* __restrict__ d_norm_sq,
                                              double lambda, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double r_sq = 0.0;

    if (idx < n) {
        double res = AX_r[idx] - lambda * X_r[idx];
        r_sq = res * res;

        double denom = diagA[idx] - lambda;
        T_r[idx] = (fabs(denom) > kPrecondEps) ? (res / denom) : res;
    }

    double block_sum = block_reduce_sum(r_sq);
    if (threadIdx.x == 0 && block_sum > 0.0) {
        atomicAdd(d_norm_sq, block_sum);
    }
}

__global__ void dot_kernel(const double* __restrict__ a,
                           const double* __restrict__ b,
                           double* __restrict__ d_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double val = (idx < n) ? (a[idx] * b[idx]) : 0.0;
    double block_sum = block_reduce_sum(val);
    if (threadIdx.x == 0 && block_sum != 0.0) {
        atomicAdd(d_out, block_sum);
    }
}

__global__ void axpy_kernel(double* __restrict__ y,
                            const double* __restrict__ x,
                            double alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] += alpha * x[idx];
    }
}

__global__ void scale_kernel(double* __restrict__ x, double alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] *= alpha;
    }
}

// Host helper: GPU dot product
double gpu_dot(const double* d_a, const double* d_b, double* d_scalar, int n) {
    CUDA_CHECK(cudaMemset(d_scalar, 0, sizeof(double)));
    int grid = (n + kBlockSize - 1) / kBlockSize;
    dot_kernel<<<grid, kBlockSize>>>(d_a, d_b, d_scalar, n);
    double result = 0.0;
    CUDA_CHECK(cudaMemcpy(&result, d_scalar, sizeof(double), cudaMemcpyDeviceToHost));
    return result;
}

// Host helper: Two-pass Classical Gram-Schmidt on GPU
bool gpu_orthonormalize_against(const double* d_V, int m, double* d_v,
                                double* d_scalar, int n) {
    double orig_norm_sq = gpu_dot(d_v, d_v, d_scalar, n);
    if (!std::isfinite(orig_norm_sq) || orig_norm_sq <= 0.0) return false;
    double orig_norm = std::sqrt(orig_norm_sq);

    int grid = (n + kBlockSize - 1) / kBlockSize;
    for (int pass = 0; pass < 2; ++pass) {
        for (int j = 0; j < m; ++j) {
            const double* d_vj = d_V + static_cast<size_t>(j) * n;
            double proj = gpu_dot(d_vj, d_v, d_scalar, n);
            axpy_kernel<<<grid, kBlockSize>>>(d_v, d_vj, -proj, n);
        }
    }

    double new_norm_sq = gpu_dot(d_v, d_v, d_scalar, n);
    if (!std::isfinite(new_norm_sq) || new_norm_sq <= 0.0) return false;
    double new_norm = std::sqrt(new_norm_sq);

    if (new_norm <= kLinDepTol * orig_norm) return false;

    scale_kernel<<<grid, kBlockSize>>>(d_v, 1.0 / new_norm, n);
    return true;
}

}  // namespace

DavidsonResult DavidsonCUDA(const Matrix& A, const Matrix& guess, int nroots,
                            double tol, int max_iter) {
    DavidsonResult res;
    const int n = A.rows();
    const int k_init = guess.cols();
    const int max_subspace = std::min(n, k_init + max_iter * nroots);

    // Flatten A (row-major) and extract diagonal
    std::vector<double> h_A(static_cast<size_t>(n) * n);
    std::vector<double> h_diagA(n);
    for (int i = 0; i < n; ++i) {
        h_diagA[i] = A[i][i];
        for (int j = 0; j < n; ++j) {
            h_A[static_cast<size_t>(i) * n + j] = A[i][j];
        }
    }

    // Allocate persistent GPU memory
    double *d_A = nullptr, *d_diagA = nullptr;
    double *d_V = nullptr, *d_AV = nullptr;
    double *d_X = nullptr, *d_AX = nullptr, *d_T = nullptr;
    double *d_H = nullptr, *d_Q = nullptr, *d_scalar = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, static_cast<size_t>(n) * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_diagA, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_V, static_cast<size_t>(max_subspace) * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_AV, static_cast<size_t>(max_subspace) * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_X, static_cast<size_t>(nroots) * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_AX, static_cast<size_t>(nroots) * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_T, static_cast<size_t>(nroots) * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_H, static_cast<size_t>(max_subspace) * max_subspace * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Q, static_cast<size_t>(max_subspace) * nroots * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_scalar, sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_diagA, h_diagA.data(), n * sizeof(double), cudaMemcpyHostToDevice));

    // Seed initial basis V on GPU and orthonormalize
    int m = 0;
    std::vector<double> h_col(n);
    auto t0 = Clock::now();
    for (int j = 0; j < k_init && m < max_subspace; ++j) {
        for (int i = 0; i < n; ++i) h_col[i] = guess[i][j];
        double* d_dest = d_V + static_cast<size_t>(m) * n;
        CUDA_CHECK(cudaMemcpy(d_dest, h_col.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        if (gpu_orthonormalize_against(d_V, m, d_dest, d_scalar, n)) {
            ++m;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    res.profile.orthogonalize += elapsed_sec(t0);

    int computed_av = 0;
    std::vector<double> h_H_buf;
    std::vector<double> h_Q_buf;
    res.eigenvalues.resize(nroots, 0.0);
    res.residual_norms.resize(nroots, 0.0);

    for (int iter = 0; iter < max_iter; ++iter) {
        if (m < nroots) break;
        res.iterations = iter + 1;
        res.subspace_size = m;

        
        int new_vecs = m - computed_av;
        if (new_vecs > 0) {
            t0 = Clock::now();
            dim3 block(kTile, kTile);
            dim3 grid((new_vecs + kTile - 1) / kTile, (n + kTile - 1) / kTile);
            matvec_block_kernel<<<grid, block>>>(
                d_A,
                d_V + static_cast<size_t>(computed_av) * n,
                d_AV + static_cast<size_t>(computed_av) * n,
                n, new_vecs);
            CUDA_CHECK(cudaDeviceSynchronize());
            res.profile.matvec += elapsed_sec(t0);
            computed_av = m;
        }

        
        t0 = Clock::now();
        {
            dim3 grid(m, m);
            project_kernel<<<grid, kBlockSize>>>(d_V, d_AV, d_H, n, m);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        res.profile.project += elapsed_sec(t0);

        t0 = Clock::now();
        h_H_buf.resize(static_cast<size_t>(m) * m);
        CUDA_CHECK(cudaMemcpy(h_H_buf.data(), d_H, h_H_buf.size() * sizeof(double),
                              cudaMemcpyDeviceToHost));

        Matrix H_mat(m, m);
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < m; ++j)
                H_mat[i][j] = h_H_buf[i * m + j];

        Matrix Q_mat(m, m);
        MatrixJacobi(H_mat, 100 * m * m, &Q_mat);

        std::vector<int> order(m);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return H_mat[a][a] < H_mat[b][b]; });

        h_Q_buf.resize(static_cast<size_t>(m) * nroots);
        for (int r = 0; r < nroots; ++r) {
            int col = order[r];
            res.eigenvalues[r] = H_mat[col][col];
            for (int k = 0; k < m; ++k) {
                h_Q_buf[k * nroots + r] = Q_mat[k][col];
            }
        }
        CUDA_CHECK(cudaMemcpy(d_Q, h_Q_buf.data(), h_Q_buf.size() * sizeof(double),
                              cudaMemcpyHostToDevice));
        res.profile.eigensolve += elapsed_sec(t0);

        
        t0 = Clock::now();
        int grid1d = (n + kBlockSize - 1) / kBlockSize;
        size_t smem_bytes = static_cast<size_t>(m) * nroots * sizeof(double);
        ritz_vectors_kernel<<<grid1d, kBlockSize, smem_bytes>>>(
            d_V, d_AV, d_Q, d_X, d_AX, n, m, nroots);

        bool all_converged = true;
        for (int r = 0; r < nroots; ++r) {
            CUDA_CHECK(cudaMemset(d_scalar, 0, sizeof(double)));
            fused_residual_precond_kernel<<<grid1d, kBlockSize>>>(
                d_X + static_cast<size_t>(r) * n,
                d_AX + static_cast<size_t>(r) * n,
                d_diagA,
                d_T + static_cast<size_t>(r) * n,
                d_scalar, res.eigenvalues[r], n);

            double norm_sq = 0.0;
            CUDA_CHECK(cudaMemcpy(&norm_sq, d_scalar, sizeof(double), cudaMemcpyDeviceToHost));
            res.residual_norms[r] = std::sqrt(norm_sq);
            if (res.residual_norms[r] >= tol) {
                all_converged = false;
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        res.profile.ritz += elapsed_sec(t0);  
        if (all_converged) {
            res.converged = true;
            break;
        }

        
        t0 = Clock::now();
        int added = 0;
        for (int r = 0; r < nroots && m < max_subspace; ++r) {
            if (res.residual_norms[r] < tol) continue;

            double* d_dest = d_V + static_cast<size_t>(m) * n;
            CUDA_CHECK(cudaMemcpy(d_dest, d_T + static_cast<size_t>(r) * n,
                                  n * sizeof(double), cudaMemcpyDeviceToDevice));

            if (gpu_orthonormalize_against(d_V, m, d_dest, d_scalar, n)) {
                ++m;
                ++added;
            } else {
                ++res.dropped_corrections;
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        res.profile.orthogonalize += elapsed_sec(t0);

        if (added == 0) break;
        ++res.expansions;
    }

    
    res.eigenvectors = Matrix(n, nroots);
    std::vector<double> h_X(static_cast<size_t>(nroots) * n);
    CUDA_CHECK(cudaMemcpy(h_X.data(), d_X, h_X.size() * sizeof(double), cudaMemcpyDeviceToHost));
    for (int r = 0; r < nroots; ++r) {
        for (int i = 0; i < n; ++i) {
            res.eigenvectors[i][r] = h_X[static_cast<size_t>(r) * n + i];
        }
    }

    cudaFree(d_A);
    cudaFree(d_diagA);
    cudaFree(d_V);
    cudaFree(d_AV);
    cudaFree(d_X);
    cudaFree(d_AX);
    cudaFree(d_T);
    cudaFree(d_H);
    cudaFree(d_Q);
    cudaFree(d_scalar);

    return res;
}