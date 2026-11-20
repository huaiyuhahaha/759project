// Verbatim copy of matrix.h from master 0e014e3 (before gemm()), wrapped in
// namespace original_matrix. bench/gemm_bench.cpp times its operator* as the
// "naive" baseline. Do not edit: it must stay identical to the original code.
#ifndef ORIGINAL_MATRIX_H
#define ORIGINAL_MATRIX_H

#include <cstddef>

namespace original_matrix {

// Dense row-major matrix. Access elements with matrix[row][column].
class Matrix {
public:
    Matrix(int rows, int cols);
    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);
    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(Matrix&& other) noexcept;
    ~Matrix();

    int rows() const noexcept;
    int cols() const noexcept;

    double* operator[](int row);
    const double* operator[](int row) const;

    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;

private:
    int rows_;
    int cols_;
    double* data_;

    std::size_t size() const noexcept;
    void swap(Matrix& other) noexcept;
    static std::size_t checked_size(int rows, int cols);
    void check_row(int row) const;
    void require_same_shape(const Matrix& other, const char* operation) const;
};

}  // namespace original_matrix

#endif
