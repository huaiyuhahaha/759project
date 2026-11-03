#ifndef MATRIX_H
#define MATRIX_H

#include <cstddef>

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

#endif
