#include <iostream>
#include <vector>
#include "matrix_comp.h"
#include <complex>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
template <typename T>
Matrix<T>::Matrix(int rows, int cols)
    : rows_(rows), cols_(cols), data_(new T[checked_size(rows, cols)]{})
{}

template <typename T>
Matrix<T>::Matrix(const Matrix& other)
    : rows_(other.rows_), cols_(other.cols_), data_(new T[other.size()])
{
    std::copy_n(other.data_, size(), data_);
}

template <typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix& other) {
    if (this != &other) {
        Matrix copy(other);
        swap(copy);
    }
    return *this;
}

template <typename T>
Matrix<T>::Matrix(Matrix&& other) noexcept
    : rows_(other.rows_), cols_(other.cols_), data_(other.data_)
{
    other.rows_ = 0;
    other.cols_ = 0;
    other.data_ = nullptr;
}

template <typename T>
Matrix<T>& Matrix<T>::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        delete[] data_;
        rows_ = other.rows_;
        cols_ = other.cols_;
        data_ = other.data_;
        other.rows_ = 0;
        other.cols_ = 0;
        other.data_ = nullptr;
    }
    return *this;
}

template <typename T>
Matrix<T>::~Matrix() { delete[] data_; }

template <typename T>
int Matrix<T>::rows() const noexcept { return rows_; }
template <typename T>
int Matrix<T>::cols() const noexcept { return cols_; }

template <typename T>
void Matrix<T>::resize(int rows, int cols) {
    if (rows == rows_ && cols == cols_)
        return;

    Matrix resized(rows, cols);
    const int kept_rows = std::min(rows_, rows);
    const int kept_cols = std::min(cols_, cols);
    if (cols == cols_ && kept_cols > 0 && kept_rows > 0) {
        std::copy_n(data_, static_cast<std::size_t>(kept_rows) * cols, resized.data_);
    } else if (kept_cols > 0) {
        for (int r = 0; r < kept_rows; ++r)
            std::copy_n((*this)[r], kept_cols, resized[r]);
    }
    swap(resized);
}

template <typename T>
std::size_t Matrix<T>::size() const noexcept {
    return static_cast<std::size_t>(rows_) * static_cast<std::size_t>(cols_);
}

template <typename T>       
void Matrix<T>::swap(Matrix& other) noexcept {
    std::swap(rows_, other.rows_);
    std::swap(cols_, other.cols_);
    std::swap(data_, other.data_);
}

template <typename T>
T* Matrix<T>::operator[](int row) {
    return data_ + static_cast<std::size_t>(row) * cols_;
}

template <typename T>
const T* Matrix<T>::operator[](int row) const {
    return data_ + static_cast<std::size_t>(row) * cols_;
}

template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix& other) const {
    require_same_shape(other, "addition");
    Matrix result(rows_, cols_);
    for (std::size_t i = 0; i < size(); ++i)
        result.data_[i] = data_[i] + other.data_[i];
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator-(const Matrix& other) const {
    require_same_shape(other, "subtraction");
    Matrix result(rows_, cols_);
    for (std::size_t i = 0; i < size(); ++i)
        result.data_[i] = data_[i] - other.data_[i];
    return result;
}

template <typename T>
std::size_t Matrix<T>::checked_size(int rows, int cols) {
    if (rows < 0 || cols < 0)
        throw std::invalid_argument("Matrix dimensions must be nonnegative");
    const auto r = static_cast<std::size_t>(rows);
    const auto c = static_cast<std::size_t>(cols);
    if (c != 0 && r > std::numeric_limits<std::size_t>::max() / c)
        throw std::overflow_error("Matrix size is too large");
    return r * c;
}

template <typename T>
void Matrix<T>::require_same_shape(const Matrix& other, const char* operation) const {
    if (rows_ != other.rows_ || cols_ != other.cols_)
        throw std::invalid_argument(std::string("Matrix ") + operation +
                                    " requires matching dimensions");
}


template class Matrix<double>;
template class Matrix<std::complex<double>>;
