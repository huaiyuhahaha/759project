#ifndef MATRIX_COMP_H
#define MATRIX_COMP_H

#include <cstddef>
#include <complex>

// Dense row-major matrix. Access elements with matrix[row][column].
template <typename T>
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

    // A shape change reallocates storage, preserving overlapping entries and
    // zero-initializing new entries. Existing row pointers are invalidated.
    void resize(int rows, int cols);

    // Unchecked access: row and column indices must be in range.
    T* operator[](int row);
    const T* operator[](int row) const;

    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;

private:
    int rows_;
    int cols_;
    T* data_;

    std::size_t size() const noexcept;
    void swap(Matrix& other) noexcept;
    static std::size_t checked_size(int rows, int cols);
    void require_same_shape(const Matrix& other, const char* operation) const;
};

// Multiplication has separate real and complex implementations.
template <>
Matrix<double> Matrix<double>::operator*(const Matrix<double>& other) const;

template <>
Matrix<std::complex<double>> Matrix<std::complex<double>>::operator*(
    const Matrix<std::complex<double>>& other) const;

#endif
