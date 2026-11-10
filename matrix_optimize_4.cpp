#include <immintrin.h>
#include <algorithm>
#include <cstddef>
#include <stdexcept>

void kernel(const double* A, const double* B, double* C, int size, int C_cols, int C_rows) {
// A has 16 rows and 'size' columns stored in column-major order
// B should be 'size' rows and 8 columns stored in row-major order
// this function does outer product of A and B in the most inner loop.

 __m512d c0 = _mm512_loadu_pd(C + 0);
        __m512d c1 = _mm512_loadu_pd(C + C_cols);
        __m512d c2 = _mm512_loadu_pd(C + 2*C_cols);
        __m512d c3 = _mm512_loadu_pd(C + 3*C_cols);
        __m512d c4 = _mm512_loadu_pd(C + 4*C_cols);
        __m512d c5 = _mm512_loadu_pd(C + 5*C_cols);
        __m512d c6 = _mm512_loadu_pd(C + 6*C_cols);
        __m512d c7 = _mm512_loadu_pd(C + 7*C_cols);
        __m512d c8 = _mm512_loadu_pd(C + 8*C_cols);
        __m512d c9 = _mm512_loadu_pd(C + 9*C_cols);
        __m512d c10 = _mm512_loadu_pd(C + 10*C_cols);
        __m512d c11 = _mm512_loadu_pd(C + 11*C_cols);
        __m512d c12 = _mm512_loadu_pd(C + 12*C_cols);
        __m512d c13 = _mm512_loadu_pd(C + 13*C_cols);
        __m512d c14 = _mm512_loadu_pd(C + 14*C_cols);
        __m512d c15 = _mm512_loadu_pd(C + 15*C_cols);

    for (int k = 0; k < size; ++k) 
    {

       



        
            __m512d a0 = _mm512_set1_pd(A[0 + k*C_rows]);
            __m512d a1 = _mm512_set1_pd(A[1 + k*C_rows]);
            __m512d a2 = _mm512_set1_pd(A[2 + k*C_rows]);
            __m512d a3 = _mm512_set1_pd(A[3 + k*C_rows]);   
            __m512d a4 = _mm512_set1_pd(A[4 + k*C_rows]);
            __m512d a5 = _mm512_set1_pd(A[5 + k*C_rows]);
            __m512d a6 = _mm512_set1_pd(A[6 + k*C_rows]);
            __m512d a7 = _mm512_set1_pd(A[7 + k*C_rows]);
            __m512d a8 = _mm512_set1_pd(A[8 + k*C_rows]);
            __m512d a9 = _mm512_set1_pd(A[9 + k*C_rows]);
            __m512d a10 = _mm512_set1_pd(A[10 + k*C_rows]);
            __m512d a11 = _mm512_set1_pd(A[11 + k*C_rows]);
            __m512d a12 = _mm512_set1_pd(A[12 + k*C_rows]);
            __m512d a13 = _mm512_set1_pd(A[13 + k*C_rows]);
            __m512d a14 = _mm512_set1_pd(A[14 + k*C_rows]);
            __m512d a15 = _mm512_set1_pd(A[15 + k*C_rows]);

            __m512d b = _mm512_loadu_pd(B + k*8);

            
            c0 = _mm512_fmadd_pd(a0, b, c0);
            c1 = _mm512_fmadd_pd(a1, b, c1);
            c2 = _mm512_fmadd_pd(a2, b, c2);
            c3 = _mm512_fmadd_pd(a3, b, c3);
            c4 = _mm512_fmadd_pd(a4, b, c4);
            c5 = _mm512_fmadd_pd(a5, b, c5);
            c6 = _mm512_fmadd_pd(a6, b, c6);
            c7 = _mm512_fmadd_pd(a7, b, c7);
            c8 = _mm512_fmadd_pd(a8, b, c8);
            c9 = _mm512_fmadd_pd(a9, b, c9);
            c10 = _mm512_fmadd_pd(a10, b, c10);
            c11 = _mm512_fmadd_pd(a11, b, c11);
            c12 = _mm512_fmadd_pd(a12, b, c12);
            c13 = _mm512_fmadd_pd(a13, b, c13);
            c14 = _mm512_fmadd_pd(a14, b, c14);
            c15 = _mm512_fmadd_pd(a15, b, c15);
                
            
        
        
    }

    _mm512_storeu_pd(C + 0, c0);
    _mm512_storeu_pd(C + C_cols, c1);
    _mm512_storeu_pd(C + 2*C_cols, c2);
    _mm512_storeu_pd(C + 3*C_cols, c3);
    _mm512_storeu_pd(C + 4*C_cols, c4);
    _mm512_storeu_pd(C + 5*C_cols, c5);
    _mm512_storeu_pd(C + 6*C_cols, c6);
    _mm512_storeu_pd(C + 7*C_cols, c7);
    _mm512_storeu_pd(C + 8*C_cols, c8);
    _mm512_storeu_pd(C + 9*C_cols, c9);
    _mm512_storeu_pd(C + 10*C_cols, c10);
    _mm512_storeu_pd(C + 11*C_cols, c11);
    _mm512_storeu_pd(C + 12*C_cols, c12);
    _mm512_storeu_pd(C + 13*C_cols, c13);
    _mm512_storeu_pd(C + 14*C_cols, c14);
    _mm512_storeu_pd(C + 15*C_cols, c15);


}

// Only edge tiles use a temporary C buffer. A and B are already zero-padded.
static void edge_kernel(const double* A, const double* B, double* C,
                        int depth, int C_cols, int rows, int cols) {
    double edge[16 * 8]{};
    for (int r = 0; r < rows; ++r)
        std::copy_n(C + static_cast<std::size_t>(r) * C_cols, cols, edge + r * 8);

    kernel(A, B, edge, depth, 8, 16);

    for (int r = 0; r < rows; ++r)
        std::copy_n(edge + r * 8, cols, C + static_cast<std::size_t>(r) * C_cols);
}

void cache_register_multiplication(const double* A, const double* B, double* C,
                                  int A_rows, int B_cols, int mutual_dim,
                                  int block_mu_dim) {
    if (A_rows < 0 || B_cols < 0 || mutual_dim < 0 || block_mu_dim <= 0)
        throw std::invalid_argument("Dimensions must be nonnegative and block depth positive");
    if (A_rows == 0 || B_cols == 0 || mutual_dim == 0)
        return; // C += A*B leaves C unchanged for an empty product.

    constexpr int B_block_cols = 256;
    constexpr int B_tiles_per_block = B_block_cols / 8;

    // Calculate full tile counts and remainder dimensions once.
    const int A_row_blocks = A_rows / 16;
    const int remaining_rows = A_rows % 16;
    const int full_rows = A_row_blocks * 16;
    const int B_col_blocks = B_cols / 8;
    const int remaining_cols = B_cols % 8;
    const int full_cols = B_col_blocks * 8;
    const int mutual_blocks = mutual_dim / block_mu_dim;
    const int remaining_depth = mutual_dim % block_mu_dim;
    const int depth_blocks = mutual_blocks + (remaining_depth != 0);

    // Group the complete 8-column tiles into blocks of up to 256 columns.
    const int full_B_blocks = B_col_blocks / B_tiles_per_block;
    const int remaining_B_tiles = B_col_blocks % B_tiles_per_block;
    const int B_blocks = full_B_blocks + (remaining_B_tiles != 0);
    const int packed_A_tiles = A_row_blocks + (remaining_rows != 0);
    const int max_depth = std::min(block_mu_dim, mutual_dim);

    double* A_block = new double[static_cast<std::size_t>(packed_A_tiles) * 16 * max_depth];
    double* B_block = new double[static_cast<std::size_t>(max_depth) * B_block_cols];

    for (int k = 0; k < depth_blocks; ++k) {
        const int depth = (k < mutual_blocks) ? block_mu_dim : remaining_depth;
        const int k0 = k * block_mu_dim;
        const std::size_t A_tile_size = static_cast<std::size_t>(16) * depth;
        const std::size_t B_tile_size = static_cast<std::size_t>(8) * depth;

        // Pack the full A tiles, each as a contiguous 16 x depth panel.
        for (int i = 0; i < A_row_blocks; ++i) {
            double* tile = A_block + i * A_tile_size;
            for (int r = 0; r < 16; ++r) {
                for (int p = 0; p < depth; ++p) {
                    tile[static_cast<std::size_t>(p) * 16 + r] =
                        A[static_cast<std::size_t>(16 * i + r) * mutual_dim + k0 + p];
                }
            }
        }

        // Pack the bottom A tile separately, padding its missing rows.
        double* bottom_A = A_block + A_row_blocks * A_tile_size;
        if (remaining_rows != 0) {
            std::fill_n(bottom_A, A_tile_size, 0.0);
            for (int r = 0; r < remaining_rows; ++r) {
                for (int p = 0; p < depth; ++p) {
                    bottom_A[static_cast<std::size_t>(p) * 16 + r] =
                        A[static_cast<std::size_t>(full_rows + r) * mutual_dim + k0 + p];
                }
            }
        }

        for (int jb = 0; jb < B_blocks; ++jb) {
            const int tiles = (jb < full_B_blocks) ? B_tiles_per_block : remaining_B_tiles;
            const int col0 = jb * B_block_cols;

            // Pack complete 8-column B tiles for this column block.
            for (int j = 0; j < tiles; ++j) {
                double* tile = B_block + j * B_tile_size;
                for (int p = 0; p < depth; ++p) {
                    for (int c = 0; c < 8; ++c) {
                        tile[static_cast<std::size_t>(p) * 8 + c] =
                            B[static_cast<std::size_t>(k0 + p) * B_cols + col0 + j * 8 + c];
                    }
                }
            }

            // 1. Main region: full tiles only, with no edge checks in this loop.
            for (int i = 0; i < A_row_blocks; ++i) {
                for (int j = 0; j < tiles; ++j) {
                    kernel(A_block + i * A_tile_size, B_block + j * B_tile_size,
                           C + static_cast<std::size_t>(16 * i) * B_cols + col0 + j * 8,
                           depth, B_cols, 16);
                }
            }

            // 2. Bottom edge: reuse this packed B block before replacing it.
            if (remaining_rows != 0) {
                for (int j = 0; j < tiles; ++j) {
                    edge_kernel(bottom_A, B_block + j * B_tile_size,
                                C + static_cast<std::size_t>(full_rows) * B_cols + col0 + j * 8,
                                depth, B_cols, remaining_rows, 8);
                }
            }
        }

        if (remaining_cols != 0) {
            // Pack the rightmost B tile, padding its missing columns.
            std::fill_n(B_block, B_tile_size, 0.0);
            for (int p = 0; p < depth; ++p) {
                for (int c = 0; c < remaining_cols; ++c) {
                    B_block[static_cast<std::size_t>(p) * 8 + c] =
                        B[static_cast<std::size_t>(k0 + p) * B_cols + full_cols + c];
                }
            }

            // 3. Right edge: full row tiles only.
            for (int i = 0; i < A_row_blocks; ++i) {
                edge_kernel(A_block + i * A_tile_size, B_block,
                            C + static_cast<std::size_t>(16 * i) * B_cols + full_cols,
                            depth, B_cols, 16, remaining_cols);
            }

            // 4. Bottom-right corner.
            if (remaining_rows != 0) {
                edge_kernel(bottom_A, B_block,
                            C + static_cast<std::size_t>(full_rows) * B_cols + full_cols,
                            depth, B_cols, remaining_rows, remaining_cols);
            }
        }
    }

    delete[] A_block;
    delete[] B_block;
}
